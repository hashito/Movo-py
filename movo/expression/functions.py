"""式から呼べる関数と定数。

**ここには、ファイル・ネットワーク・時刻・プロセスに触れるものを 1 つも置きません。**
それがサンドボックスの中身そのものです（仕様 23 章 / 34 章）。
新しい関数を足すときは «外の世界を読まないか» «同じ入力から必ず同じ値が出るか» を
確かめてください。決定性が崩れると、同じ JSON から違う動画が出ます。
"""

from __future__ import annotations

import math

from ._compat import (
    NAN,
    TAU,
    UNDEFINED,
    clamp,
    create_random,
    fbm_1d,
    hash_string,
    is_finite_number,
    is_nullish,
    js_mod,
    js_round,
    js_sign,
    js_string,
    js_trunc,
    js_truthy,
    lerp,
    smoothstep,
    to_degrees,
    to_radians,
    u32,
    value_noise_1d,
    value_noise_2d,
)

CONSTANTS = {
    "PI": math.pi,
    "TAU": TAU,
    "E": math.e,
    "DEG2RAD": math.pi / 180,
    "RAD2DEG": 180 / math.pi,
    "Infinity": math.inf,
}


def _num(value, fallback=0):
    """JS 版の `num()`。«本物の有限な数» でなければ既定値へ落とす。

    文字列の `"3"` は 3 になりません（JS も `typeof v === 'number'` で見ています）。
    """
    return value if is_finite_number(value) else fallback


def create_function_library(seed: int = 0) -> dict:
    """式から見える関数の辞書を作る。`seed` で乱数とノイズが決まる。"""
    seed = int(seed) if is_finite_number(seed) else 0
    random_streams: dict[str, object] = {}

    def stream_for(name):
        key = str(name)
        s = random_streams.get(key)
        if s is None:
            s = create_random(u32(seed ^ hash_string(key)))
            random_streams[key] = s
        return s

    def _clamp(x, lo=UNDEFINED, hi=UNDEFINED):
        return clamp(_num(x), _num(lo), _num(hi, 1))

    def _round(x, digits=UNDEFINED):
        d = math.pow(10, max(0, js_round(_num(digits))))
        return js_round(_num(x) * d) / d

    def _mod(x, y=UNDEFINED):
        m = _num(y)
        if m == 0:
            return 0
        return js_mod(js_mod(_num(x), m) + m, m)

    def _remap(x, in_min=UNDEFINED, in_max=UNDEFINED, out_min=UNDEFINED, out_max=UNDEFINED, do_clamp=True):
        """ある範囲の値を別の範囲へ写す。既定では範囲外を切り詰める。"""
        a = _num(in_min)
        b = _num(in_max, 1)
        if a == b:
            return _num(out_min)
        t = (_num(x) - a) / (b - a)
        if do_clamp:
            t = clamp(t, 0, 1)
        return lerp(_num(out_min), _num(out_max, 1), t)

    def _ping_pong(x, length=UNDEFINED):
        ll = max(1e-9, _num(length, 1))
        t = js_mod(js_mod(_num(x), ll * 2) + ll * 2, ll * 2)
        return ll * 2 - t if t > ll else t

    def _wave(t, frequency=UNDEFINED, amplitude=UNDEFINED, phase=UNDEFINED):
        return math.sin((_num(t) * _num(frequency, 1) + _num(phase)) * TAU) * _num(amplitude, 1)

    def _triangle(t, frequency=UNDEFINED):
        p = js_mod(js_mod(_num(t) * _num(frequency, 1), 1) + 1, 1)
        return p * 4 - 1 if p < 0.5 else 3 - p * 4

    def _square(t, frequency=UNDEFINED, duty=UNDEFINED):
        p = js_mod(js_mod(_num(t) * _num(frequency, 1), 1) + 1, 1)
        return 1 if p < _num(duty, 0.5) else -1

    def _sawtooth(t, frequency=UNDEFINED):
        p = js_mod(js_mod(_num(t) * _num(frequency, 1), 1) + 1, 1)
        return p * 2 - 1

    def _wiggle_at(t, frequency=UNDEFINED, amplitude=UNDEFINED, seed_offset=UNDEFINED):
        """AE の wiggle 相当。滑らかなノイズで揺らす。"""
        return fbm_1d(_num(t) * _num(frequency, 1), seed + js_round(_num(seed_offset)) * 977, 3) * _num(
            amplitude, 1
        )

    def _beat_at(t, bpm=UNDEFINED, division=UNDEFINED, decay=UNDEFINED):
        """BPM に同期した減衰パルス（拍の頭で 1）。"""
        length = (60 / max(1, _num(bpm, 120))) * max(0.03125, _num(division, 1))
        position = js_mod(js_mod(_num(t), length) + length, length)
        d = _num(decay, 6)
        return math.exp(-position * d) if d > 0 else 1 - position / length

    def _beat_index_at(t, bpm=UNDEFINED, division=UNDEFINED):
        length = (60 / max(1, _num(bpm, 120))) * max(0.03125, _num(division, 1))
        return math.floor(_num(t) / length)

    def _noise(x, y=UNDEFINED):
        if y is UNDEFINED:
            return value_noise_1d(_num(x), seed)
        return value_noise_2d(_num(x), _num(y), seed)

    def _fbm(x, octaves=UNDEFINED):
        return fbm_1d(_num(x), seed, max(1, js_round(_num(octaves, 4))))

    def _random(stream_or_min=UNDEFINED, maximum=UNDEFINED):
        if isinstance(stream_or_min, str):
            return stream_for(stream_or_min)()
        r = stream_for("default")()
        if stream_or_min is UNDEFINED:
            return r
        if maximum is UNDEFINED:
            return r * _num(stream_or_min, 1)
        return _num(stream_or_min) + r * (_num(maximum, 1) - _num(stream_or_min))

    def _hash_random(key=UNDEFINED):
        h = hash_string(js_string(key)) ^ seed
        return u32(h) / 4294967296

    def _default_to(value, fallback=UNDEFINED):
        if is_nullish(value) or (isinstance(value, float) and math.isnan(value)):
            return fallback
        return value

    def _min(*args):
        values = [_num(a, math.inf) for a in args]
        return min(values) if values else math.inf

    def _max(*args):
        values = [_num(a, -math.inf) for a in args]
        return max(values) if values else -math.inf

    def _pow(x, y=UNDEFINED):
        try:
            return float(math.pow(_num(x), _num(y)))
        except (OverflowError, ValueError):
            return math.inf

    def _log(x=UNDEFINED):
        return math.log(max(1e-12, _num(x)))

    return {
        # --- 三角関数 -------------------------------------------------------
        "sin": lambda x=UNDEFINED: math.sin(_num(x)),
        "cos": lambda x=UNDEFINED: math.cos(_num(x)),
        "tan": lambda x=UNDEFINED: math.tan(_num(x)),
        "asin": lambda x=UNDEFINED: math.asin(clamp(_num(x), -1, 1)),
        "acos": lambda x=UNDEFINED: math.acos(clamp(_num(x), -1, 1)),
        "atan": lambda x=UNDEFINED: math.atan(_num(x)),
        "atan2": lambda y=UNDEFINED, x=UNDEFINED: math.atan2(_num(y), _num(x)),
        "sinh": lambda x=UNDEFINED: math.sinh(_num(x)),
        "cosh": lambda x=UNDEFINED: math.cosh(_num(x)),
        "tanh": lambda x=UNDEFINED: math.tanh(_num(x)),
        "degrees": lambda x=UNDEFINED: to_degrees(_num(x)),
        "radians": lambda x=UNDEFINED: to_radians(_num(x)),
        # --- 算術 -----------------------------------------------------------
        "abs": lambda x=UNDEFINED: abs(_num(x)),
        "sign": lambda x=UNDEFINED: js_sign(_num(x)),
        "floor": lambda x=UNDEFINED: math.floor(_num(x)),
        "ceil": lambda x=UNDEFINED: math.ceil(_num(x)),
        "round": _round,
        "trunc": lambda x=UNDEFINED: js_trunc(_num(x)),
        "sqrt": lambda x=UNDEFINED: math.sqrt(max(0, _num(x))),
        "cbrt": lambda x=UNDEFINED: math.copysign(abs(_num(x)) ** (1 / 3), _num(x)),
        "pow": _pow,
        "exp": lambda x=UNDEFINED: math.exp(_num(x)),
        "log": _log,
        "log2": lambda x=UNDEFINED: math.log2(max(1e-12, _num(x))),
        "log10": lambda x=UNDEFINED: math.log10(max(1e-12, _num(x))),
        "hypot": lambda *args: math.hypot(*[_num(a) for a in args]) if args else 0.0,
        "mod": _mod,
        "min": _min,
        "max": _max,
        "sum": lambda *args: sum(_num(a) for a in args),
        # --- 補間 -----------------------------------------------------------
        "clamp": _clamp,
        "lerp": lambda a=UNDEFINED, b=UNDEFINED, t=UNDEFINED: lerp(_num(a), _num(b), _num(t)),
        "mix": lambda a=UNDEFINED, b=UNDEFINED, t=UNDEFINED: lerp(_num(a), _num(b), _num(t)),
        "smoothstep": lambda e0=UNDEFINED, e1=UNDEFINED, x=UNDEFINED: smoothstep(
            _num(e0), _num(e1, 1), _num(x)
        ),
        "step": lambda edge=UNDEFINED, x=UNDEFINED: 0 if _num(x) < _num(edge) else 1,
        "remap": _remap,
        "pingPong": _ping_pong,
        # --- 周期波 ---------------------------------------------------------
        "wave": _wave,
        "triangle": _triangle,
        "square": _square,
        "sawtooth": _sawtooth,
        # --- MV 向けヘルパー（時刻を明示的に渡す版） ------------------------
        "wiggleAt": _wiggle_at,
        "beatAt": _beat_at,
        "beatIndexAt": _beat_index_at,
        # --- ノイズと乱数（決定的） -----------------------------------------
        "noise": _noise,
        "fbm": _fbm,
        "random": _random,
        "hashRandom": _hash_random,
        # --- 論理 -----------------------------------------------------------
        # 真偽の判定は JS に合わせる（空配列は «真»）。Python の bool() とは違う。
        "if": lambda condition=UNDEFINED, when_true=UNDEFINED, when_false=UNDEFINED: (
            when_true if js_truthy(condition) else when_false
        ),
        "select": lambda condition=UNDEFINED, when_true=UNDEFINED, when_false=UNDEFINED: (
            when_true if js_truthy(condition) else when_false
        ),
        "isFinite": lambda x=UNDEFINED: is_finite_number(x),
        "defaultTo": _default_to,
        # --- ベクトル -------------------------------------------------------
        "length": lambda x=UNDEFINED, y=UNDEFINED: math.hypot(_num(x), _num(y)),
        "distance": lambda x1=UNDEFINED, y1=UNDEFINED, x2=UNDEFINED, y2=UNDEFINED: math.hypot(
            _num(x2) - _num(x1), _num(y2) - _num(y1)
        ),
        "angle": lambda x1=UNDEFINED, y1=UNDEFINED, x2=UNDEFINED, y2=UNDEFINED: to_degrees(
            math.atan2(_num(y2) - _num(y1), _num(x2) - _num(x1))
        ),
    }


__all__ = ["CONSTANTS", "NAN", "create_function_library"]
