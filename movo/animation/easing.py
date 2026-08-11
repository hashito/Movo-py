"""イージングと cubic-bezier。キーフレーム補間から使います。"""

from __future__ import annotations

import math
import re

from movo.expression._compat import clamp, is_finite_number, js_number

_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1
_C4 = (2 * math.pi) / 3
_C5 = (2 * math.pi) / 4.5


def _bounce_out(x):
    n1 = 7.5625
    d1 = 2.75
    if x < 1 / d1:
        return n1 * x * x
    if x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    if x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    x -= 2.625 / d1
    return n1 * x * x + 0.984375


def cubic_bezier(x1, y1, x2, y2):
    """CSS と同じ 3 次ベジェ。二分法で解く（安定していて短い）。"""

    def sample_x(t):
        return 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t**3

    def sample_y(t):
        return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t**3

    def eased(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1
        lo = 0.0
        hi = 1.0
        t = x
        for _ in range(24):
            value = sample_x(t)
            if abs(value - x) < 1e-5:
                break
            if value < x:
                lo = t
            else:
                hi = t
            t = (lo + hi) / 2
        return sample_y(t)

    return eased


def _ease_in_out_expo(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return math.pow(2, 20 * x - 10) / 2
    return (2 - math.pow(2, -20 * x + 10)) / 2


def _ease_in_out_elastic(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return -(math.pow(2, 20 * x - 10) * math.sin((20 * x - 11.125) * _C5)) / 2
    return (math.pow(2, -20 * x + 10) * math.sin((20 * x - 11.125) * _C5)) / 2 + 1


EASINGS: dict = {
    "linear": lambda x: x,
    "hold": lambda x: 0,
    "step": lambda x: 1 if x >= 1 else 0,
    "stepEnd": lambda x: 1 if x >= 1 else 0,
    "stepStart": lambda x: 1 if x > 0 else 0,
    "ease": lambda x: cubic_bezier(0.25, 0.1, 0.25, 1)(x),
    "easeIn": lambda x: x * x,
    "easeOut": lambda x: 1 - (1 - x) * (1 - x),
    "easeInOut": lambda x: 2 * x * x if x < 0.5 else 1 - math.pow(-2 * x + 2, 2) / 2,
    "easeInQuad": lambda x: x * x,
    "easeOutQuad": lambda x: 1 - (1 - x) * (1 - x),
    "easeInOutQuad": lambda x: 2 * x * x if x < 0.5 else 1 - math.pow(-2 * x + 2, 2) / 2,
    "easeInCubic": lambda x: x**3,
    "easeOutCubic": lambda x: 1 - math.pow(1 - x, 3),
    "easeInOutCubic": lambda x: 4 * x**3 if x < 0.5 else 1 - math.pow(-2 * x + 2, 3) / 2,
    "easeInQuart": lambda x: x**4,
    "easeOutQuart": lambda x: 1 - math.pow(1 - x, 4),
    "easeInOutQuart": lambda x: 8 * x**4 if x < 0.5 else 1 - math.pow(-2 * x + 2, 4) / 2,
    "easeInQuint": lambda x: x**5,
    "easeOutQuint": lambda x: 1 - math.pow(1 - x, 5),
    "easeInOutQuint": lambda x: 16 * x**5 if x < 0.5 else 1 - math.pow(-2 * x + 2, 5) / 2,
    "easeInSine": lambda x: 1 - math.cos((x * math.pi) / 2),
    "easeOutSine": lambda x: math.sin((x * math.pi) / 2),
    "easeInOutSine": lambda x: -(math.cos(math.pi * x) - 1) / 2,
    "easeInExpo": lambda x: 0 if x == 0 else math.pow(2, 10 * x - 10),
    "easeOutExpo": lambda x: 1 if x == 1 else 1 - math.pow(2, -10 * x),
    "easeInOutExpo": _ease_in_out_expo,
    "easeInCirc": lambda x: 1 - math.sqrt(max(0.0, 1 - math.pow(x, 2))),
    "easeOutCirc": lambda x: math.sqrt(max(0.0, 1 - math.pow(x - 1, 2))),
    "easeInOutCirc": lambda x: (
        (1 - math.sqrt(max(0.0, 1 - math.pow(2 * x, 2)))) / 2
        if x < 0.5
        else (math.sqrt(max(0.0, 1 - math.pow(-2 * x + 2, 2))) + 1) / 2
    ),
    "easeInBack": lambda x: _C3 * x**3 - _C1 * x * x,
    "easeOutBack": lambda x: 1 + _C3 * math.pow(x - 1, 3) + _C1 * math.pow(x - 1, 2),
    "easeInOutBack": lambda x: (
        (math.pow(2 * x, 2) * ((_C2 + 1) * 2 * x - _C2)) / 2
        if x < 0.5
        else (math.pow(2 * x - 2, 2) * ((_C2 + 1) * (x * 2 - 2) + _C2) + 2) / 2
    ),
    "easeInElastic": lambda x: (
        0 if x == 0 else 1 if x == 1 else -math.pow(2, 10 * x - 10) * math.sin((x * 10 - 10.75) * _C4)
    ),
    "easeOutElastic": lambda x: (
        0 if x == 0 else 1 if x == 1 else math.pow(2, -10 * x) * math.sin((x * 10 - 0.75) * _C4) + 1
    ),
    "easeInOutElastic": _ease_in_out_elastic,
    "easeInBounce": lambda x: 1 - _bounce_out(1 - x),
    "easeOutBounce": _bounce_out,
    "easeInOutBounce": lambda x: (
        (1 - _bounce_out(1 - 2 * x)) / 2 if x < 0.5 else (1 + _bounce_out(2 * x - 1)) / 2
    ),
}

# 書き手の癖に合わせた別名。
EASINGS["ease-in"] = EASINGS["easeIn"]
EASINGS["ease-out"] = EASINGS["easeOut"]
EASINGS["ease-in-out"] = EASINGS["easeInOut"]
EASINGS["bounce"] = EASINGS["easeOutBounce"]
EASINGS["elastic"] = EASINGS["easeOutElastic"]
EASINGS["back"] = EASINGS["easeOutBack"]

_CUBIC_BEZIER_RE = re.compile(r"^cubic-?bezier\(([^)]+)\)$", re.IGNORECASE)
_NORMALISE_RE = re.compile(r"[-_\s]")


def list_easings() -> list[str]:
    return sorted(EASINGS.keys())


def get_easing(spec):
    """イージングの指定を関数にする。

    名前・`[x1,y1,x2,y2]`・`{"type": "cubicBezier", "points": [...]}` を受けます。
    知らない名前は linear に落とします（動画が出なくなるより «動く» を選ぶ）。
    """
    if not spec:
        return EASINGS["linear"]
    if callable(spec):
        return spec
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return cubic_bezier(spec[0], spec[1], spec[2], spec[3])
    if isinstance(spec, dict):
        points = spec.get("points")
        if isinstance(points, (list, tuple)) and len(points) == 4:
            return cubic_bezier(*points)
        return EASINGS["linear"]
    if isinstance(spec, str):
        m = _CUBIC_BEZIER_RE.match(spec.strip())
        if m:
            parts = [js_number(v.strip()) for v in m.group(1).split(",")]
            if len(parts) == 4 and all(is_finite_number(p) for p in parts):
                return cubic_bezier(*parts)
        found = EASINGS.get(spec)
        if found is not None:
            return found
        found = EASINGS.get(_NORMALISE_RE.sub("", spec))
        return found if found is not None else EASINGS["linear"]
    return EASINGS["linear"]


def apply_easing(spec, t):
    return get_easing(spec)(clamp(t, 0, 1))


__all__ = ["EASINGS", "apply_easing", "cubic_bezier", "get_easing", "list_easings"]
