"""周期モジュレーター（仕様 7.4 節 / 8 節）。

どれも時刻を `offset + shape(t) * amplitude` に写します。`shape` は -1〜1
（pulse と random-step だけ 0〜1 のほうが読みやすいのでそちら）に正規化されています。
**時刻だけの純粋な関数** なので、どこから何度呼んでも同じ値が出ます。
"""

from __future__ import annotations

import math

from movo.expression._compat import (
    TAU,
    clamp,
    fbm_1d,
    is_nullish,
    js_mod,
    js_number,
    js_round,
    lerp,
    sample_polyline,
    value_noise_1d,
)

MODULATOR_TYPES = [
    "sine",
    "cosine",
    "triangle",
    "square",
    "sawtooth",
    "pulse",
    "noise",
    "random-step",
    "custom-curve",
    "audio-reactive",
    "shake",
    "beat",
    "snap",
]

COMBINE_MODES = ["add", "multiply", "replace", "min", "max", "average"]


def _default_resolve(value, fallback):
    """既定の «数の読み方»。数値でなければ既定値。"""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback


def _wrap01(t):
    return js_mod(js_mod(t, 1) + 1, 1)


def _smooth_gate(p, edge, softness):
    if softness <= 0:
        return 1 if p >= edge else 0
    return clamp((p - edge) / softness, 0, 1)


def _get(spec, key, default=None):
    return spec.get(key, default) if isinstance(spec, dict) else default


def evaluate_modulator(spec, time, ctx=None):
    """モジュレーター 1 つを評価して数を返す。

    `ctx` は `{"seed": int, "audio": {...}, "bpm": float, "resolve_number": fn}`。
    """
    ctx = ctx or {}
    resolve = ctx.get("resolve_number") or _default_resolve
    kind = _get(spec, "type") or "sine"
    frequency = resolve(_get(spec, "frequency"), 1)
    amplitude = resolve(_get(spec, "amplitude"), 1)
    offset = resolve(_get(spec, "offset"), 0)
    phase = resolve(_get(spec, "phase"), 0)
    seed = (ctx.get("seed") or 0) + (_get(spec, "seedOffset") or 0)
    t = time * frequency + phase

    if kind == "sine":
        shape = math.sin(t * TAU)
    elif kind == "cosine":
        shape = math.cos(t * TAU)
    elif kind == "triangle":
        p = _wrap01(t)
        shape = p * 4 - 1 if p < 0.5 else 3 - p * 4
    elif kind == "square":
        duty = resolve(_get(spec, "duty"), 0.5)
        shape = 1 if _wrap01(t) < duty else -1
    elif kind == "sawtooth":
        shape = _wrap01(t) * 2 - 1
    elif kind == "pulse":
        # 0〜1 のゲート。`width` は 1 周期のうち «開いている» 割合。
        raw_width = _get(spec, "width")
        if is_nullish(raw_width):
            raw_width = _get(spec, "duty")
        width = clamp(resolve(raw_width, 0.5), 0, 1)
        soft = clamp(resolve(_get(spec, "softness"), 0), 0, 0.5)
        p = _wrap01(t)
        if soft <= 0:
            shape = 1 if p < width else 0
        else:
            rise = _smooth_gate(p, 0, soft)
            fall = 1 - _smooth_gate(p, width, soft)
            shape = clamp(min(rise, fall), 0, 1)
    elif kind == "noise":
        octaves = max(1, js_round(resolve(_get(spec, "octaves"), 1)))
        shape = fbm_1d(t, seed, octaves) if octaves > 1 else value_noise_1d(t, seed)
    elif kind == "random-step":
        steps = math.floor(t)
        shape = value_noise_1d(steps + 0.5, seed)
    elif kind == "snap":
        # «段階的な回転»。15 度や 45 度に吸い付く機械的な動きを作る。
        # 式で floor(time * 4) % 4 * 15 と書けますが、何をしたいのかが伝わりません。
        step = max(0.0001, resolve(_get(spec, "step"), 15))
        rate = max(0.0001, resolve(_get(spec, "rate"), 4))
        from_value = resolve(_get(spec, "from"), -step * 2)
        to_value = resolve(_get(spec, "to"), step * 2)
        index = math.floor(time * rate)
        span = max(step, to_value - from_value)
        if (_get(spec, "mode") or "pingpong") == "random":
            raw = from_value + value_noise_1d(index + 0.5, seed) * 0.5 * span + span * 0.5
        else:
            steps = max(1, js_round(span / step))
            cycle = steps * 2
            position = int(js_mod(js_mod(index, cycle) + cycle, cycle))
            raw = from_value + (position if position <= steps else cycle - position) * step
        # 呼び出し側で amplitude を掛けるので、ここでは «度» をそのまま出す。
        shape = js_round(raw / step) * step
    elif kind == "custom-curve":
        curve = _get(spec, "curve")
        if is_nullish(curve):
            curve = _get(spec, "points")
        if not curve:
            shape = 0
        else:
            p = _wrap01(t)
            if isinstance(curve[0], (list, tuple)):
                shape = sample_polyline(curve, p)[1]
            else:
                scaled = p * (len(curve) - 1)
                i = min(len(curve) - 2, math.floor(scaled))
                nxt = curve[i + 1] if i + 1 < len(curve) else curve[i]
                shape = lerp(curve[i], nxt, scaled - i)
    elif kind == "shake":
        # 一瞬だけ揺らす減衰振動。常時振動しないよう decay を既定で持たせる。
        start = resolve(_get(spec, "start"), 0)
        decay = resolve(_get(spec, "decay"), 6)
        elapsed = time - start
        if elapsed < 0:
            shape = 0
        else:
            envelope = math.exp(-elapsed * decay) if decay > 0 else 1
            if _get(spec, "random") is False:
                jitter = math.sin(elapsed * frequency * TAU)
            else:
                jitter = value_noise_1d(elapsed * frequency * 2, seed)
            shape = jitter * envelope
    elif kind == "beat":
        # BPM に同期したパルス。拍の頭で 1、そこから decay で減衰する。
        bpm = resolve(_get(spec, "bpm"), ctx.get("bpm") if ctx.get("bpm") is not None else 120)
        division = max(0.03125, resolve(_get(spec, "division"), 1))
        decay = resolve(_get(spec, "decay"), 6)
        beat_length = (60 / max(1, bpm)) * division
        phase_offset = resolve(_get(spec, "beatOffset"), 0)
        position = js_mod(js_mod(time - phase_offset, beat_length) + beat_length, beat_length)
        shape = math.exp(-position * decay) if decay > 0 else 1 - position / beat_length
    elif kind == "audio-reactive":
        audio = ctx.get("audio") or {"level": 0, "bands": []}
        band = _get(spec, "band")
        level = audio.get("level") or 0
        bands = audio.get("bands")
        if band is not None and isinstance(bands, (list, tuple)) and len(bands):
            index = int(clamp(js_round(js_number(band)), 0, len(bands) - 1))
            level = bands[index] if index < len(bands) else 0
        smoothing = clamp(resolve(_get(spec, "smoothing"), 0), 0, 0.99)
        previous = _get(spec, "__previous")
        if is_nullish(previous):
            previous = level
        shape = level * (1 - smoothing) + previous * smoothing
    else:
        shape = 0

    value = offset + shape * amplitude
    clamp_spec = _get(spec, "clamp")
    if isinstance(clamp_spec, (list, tuple)) and len(clamp_spec) == 2:
        value = clamp(value, clamp_spec[0], clamp_spec[1])
    return value


def combine_values(values, mode="add", base=0):
    """複数のモジュレーターの結果をまとめる（仕様 8 節）。"""
    values = list(values)
    if len(values) == 0:
        return base
    if mode == "replace":
        return values[-1]
    if mode == "multiply":
        acc = 1 if base == 0 else base
        for v in values:
            acc *= v
        return acc
    if mode == "min":
        return min([values[0] if base == 0 and values else base] + values)
    if mode == "max":
        return max([values[0] if base == 0 and values else base] + values)
    if mode == "average":
        return sum(values) / len(values)
    acc = base
    for v in values:
        acc += v
    return acc


def list_modulators() -> list[str]:
    return list(MODULATOR_TYPES)


__all__ = [
    "COMBINE_MODES",
    "MODULATOR_TYPES",
    "combine_values",
    "evaluate_modulator",
    "list_modulators",
]
