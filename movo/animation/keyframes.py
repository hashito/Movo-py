"""キーフレームのサンプリング。

値は数値・色（`#rrggbb`）・数値の配列・文字列を受けます。
数値と配列は補間し、それ以外は «切り替わる»（ステップ）扱いです。
"""

from __future__ import annotations

import math
import re

from movo.expression._compat import (
    UNDEFINED,
    clamp,
    is_finite_number,
    js_round,
    js_string,
    parse_color,
)

from .easing import get_easing

_COLOR_RE = re.compile(r"^(#|rgba?\(|hsla?\()")


def _is_color_string(value) -> bool:
    return isinstance(value, str) and bool(_COLOR_RE.match(value.strip()))


def _is_number(value) -> bool:
    """JS の `typeof v === 'number'`。Python の真偽値は数に数えない。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _interpolate(a, b, t, eased_value_space=False):
    if _is_number(a) and _is_number(b):
        return a + (b - a) * t
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        length = max(len(a), len(b))
        out = []
        for i in range(length):
            av = a[i] if i < len(a) and _is_number(a[i]) else 0
            bv = b[i] if i < len(b) and _is_number(b[i]) else av
            out.append(av + (bv - av) * t)
        return out
    if _is_color_string(a) and _is_color_string(b):
        ca = parse_color(a)
        cb = parse_color(b)

        def mix(x, y):
            return js_round(x + (y - x) * t)

        alpha = ca["a"] + (cb["a"] - ca["a"]) * t
        return f"rgba({mix(ca['r'], cb['r'])}, {mix(ca['g'], cb['g'])}, {mix(ca['b'], cb['b'])}, {js_string(alpha)})"
    if isinstance(a, dict) and isinstance(b, dict) and not eased_value_space:
        out = {}
        for key in list(a.keys()) + [k for k in b.keys() if k not in a]:
            other = b[key] if key in b else a.get(key)
            out[key] = _interpolate(a.get(key), other, t, True)
        return out
    return b if t >= 1 else a


def _time_of(keyframe) -> float:
    time = keyframe.get("time") if isinstance(keyframe, dict) else None
    return time if is_finite_number(time) else 0.0


def sample_keyframes(keyframes, time, extrapolate=None):
    """キーフレームの列を `time` 秒でサンプルする。

    `extrapolate` は範囲外の扱い: `"hold"`（既定）/ `"loop"` / `"pingPong"` / `"extend"`。
    """
    if not isinstance(keyframes, (list, tuple)) or len(keyframes) == 0:
        return UNDEFINED
    sorted_frames = sorted(keyframes, key=_time_of) if len(keyframes) > 1 else list(keyframes)
    first = sorted_frames[0]
    last = sorted_frames[-1]
    if len(sorted_frames) == 1:
        return first.get("value")

    start = _time_of(first)
    end = _time_of(last)
    span = end - start
    t = time
    mode = extrapolate if extrapolate else "hold"

    if span > 0 and (t < start or t > end):
        if mode == "loop":
            t = start + math.fmod(math.fmod(t - start, span) + span, span)
        elif mode == "pingPong":
            p = math.fmod(math.fmod(t - start, span * 2) + span * 2, span * 2)
            t = start + (span * 2 - p if p > span else p)

    if t <= start:
        if mode == "extend" and len(sorted_frames) > 1:
            k0 = sorted_frames[0]
            k1 = sorted_frames[1]
            if _is_number(k0.get("value")) and _is_number(k1.get("value")) and _time_of(k1) != _time_of(k0):
                slope = (k1["value"] - k0["value"]) / (_time_of(k1) - _time_of(k0))
                return k0["value"] + slope * (t - _time_of(k0))
        return first.get("value")
    if t >= end:
        if mode == "extend" and len(sorted_frames) > 1:
            k0 = sorted_frames[-2]
            k1 = last
            if _is_number(k0.get("value")) and _is_number(k1.get("value")) and _time_of(k1) != _time_of(k0):
                slope = (k1["value"] - k0["value"]) / (_time_of(k1) - _time_of(k0))
                return k1["value"] + slope * (t - _time_of(k1))
        return last.get("value")

    i = 0
    while i < len(sorted_frames) - 2 and _time_of(sorted_frames[i + 1]) <= t:
        i += 1
    a = sorted_frames[i]
    b = sorted_frames[i + 1]
    duration = _time_of(b) - _time_of(a)
    raw = 1.0 if duration <= 0 else clamp((t - _time_of(a)) / duration, 0, 1)
    if a.get("hold") or b.get("easing") == "hold":
        return a.get("value")
    # 仕様どおり、イージングは «向かう先» のキーフレームが持つ。
    easing_spec = b.get("easing")
    if easing_spec is None:
        easing_spec = a.get("easingOut", "linear")
    easing = get_easing(easing_spec)
    return _interpolate(a.get("value"), b.get("value"), easing(raw))


def keyframe_range(keyframes):
    """キーフレームの列が覆う時間の幅。"""
    if not isinstance(keyframes, (list, tuple)) or len(keyframes) == 0:
        return None
    times = [_time_of(k) for k in keyframes]
    return {"start": min(times), "end": max(times)}


__all__ = ["keyframe_range", "sample_keyframes"]
