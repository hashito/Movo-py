"""色の解釈と変換。色は ``{"r","g","b"}` が 0..255、``"a"`` が 0..1 の辞書で扱います。

**辞書のキーは JS 版のまま ``r`` / ``g`` / ``b`` / ``a`` です。** プロジェクト JSON に
そのまま書かれる形なので、Python 側で名前を変えるわけにはいきません。

## なぜ NumPy を使っていないか

ここは **1 フレームに数十回**しか通りません（レイヤーの色を 1 回引くだけ）。
配列にする確保のコストのほうが大きいので、素の Python が最速です。
画素の配列に色を «当てる» のは :mod:`movo.core.bitmap` と renderer の仕事です。
"""

from __future__ import annotations

import re
from typing import Any

from .math import clamp, js_round

#: CSS の色名のうち、実際に使われるものだけ。全部入れても使われないので絞っています。
NAMED: dict[str, tuple[int, int, int, float]] = {
    "transparent": (0, 0, 0, 0.0),
    "black": (0, 0, 0, 1.0),
    "white": (255, 255, 255, 1.0),
    "red": (255, 0, 0, 1.0),
    "green": (0, 128, 0, 1.0),
    "lime": (0, 255, 0, 1.0),
    "blue": (0, 0, 255, 1.0),
    "yellow": (255, 255, 0, 1.0),
    "cyan": (0, 255, 255, 1.0),
    "magenta": (255, 0, 255, 1.0),
    "gray": (128, 128, 128, 1.0),
    "grey": (128, 128, 128, 1.0),
    "silver": (192, 192, 192, 1.0),
    "orange": (255, 165, 0, 1.0),
    "purple": (128, 0, 128, 1.0),
    "pink": (255, 192, 203, 1.0),
    "brown": (165, 42, 42, 1.0),
    "navy": (0, 0, 128, 1.0),
    "teal": (0, 128, 128, 1.0),
    "gold": (255, 215, 0, 1.0),
}

Color = dict[str, float]

_DEFAULT: Color = {"r": 0, "g": 0, "b": 0, "a": 1.0}

_RGB_RE = re.compile(r"^rgba?\(([^)]+)\)$")
_HSL_RE = re.compile(r"^hsla?\(([^)]+)\)$")
_SPLIT_RE = re.compile(r"[,\s/]+")


def _to_number(text: str, fallback: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return fallback


def parse_color(value: Any, fallback: Color | None = None) -> Color:
    """``#rgb`` / ``#rrggbbaa`` / ``rgb()`` / ``hsl()`` / 色名 / 解析済みの辞書を読む。

    **読めなかったときは例外にせず ``fallback`` を返します。** 色が 1 つ
    間違っているだけで動画が出ないのは割に合わないからです。書き間違いは
    ``movo validate`` が拾います。
    """
    fb = dict(_DEFAULT if fallback is None else fallback)
    if value is None:
        return fb
    if isinstance(value, dict):
        return {
            "r": clamp(js_round(value.get("r", 0) or 0), 0, 255),
            "g": clamp(js_round(value.get("g", 0) or 0), 0, 255),
            "b": clamp(js_round(value.get("b", 0) or 0), 0, 255),
            "a": clamp(1.0 if value.get("a") is None else value["a"], 0, 1),
        }
    if isinstance(value, (list, tuple)):
        r, g, b = (list(value) + [0, 0, 0])[:3]
        a = value[3] if len(value) > 3 else 1.0
        return {
            "r": clamp(js_round(r), 0, 255),
            "g": clamp(js_round(g), 0, 255),
            "b": clamp(js_round(b), 0, 255),
            "a": clamp(a, 0, 1),
        }

    text = str(value).strip().lower()
    if text in NAMED:
        r, g, b, a = NAMED[text]
        return {"r": r, "g": g, "b": b, "a": a}

    if text.startswith("#"):
        hex_text = text[1:]
        try:
            if len(hex_text) in (3, 4):
                expand = lambda c: int(c + c, 16)  # noqa: E731
                return {
                    "r": expand(hex_text[0]),
                    "g": expand(hex_text[1]),
                    "b": expand(hex_text[2]),
                    "a": expand(hex_text[3]) / 255 if len(hex_text) == 4 else 1.0,
                }
            if len(hex_text) in (6, 8):
                return {
                    "r": int(hex_text[0:2], 16),
                    "g": int(hex_text[2:4], 16),
                    "b": int(hex_text[4:6], 16),
                    "a": int(hex_text[6:8], 16) / 255 if len(hex_text) == 8 else 1.0,
                }
        except ValueError:
            return fb
        return fb

    m = _RGB_RE.match(text)
    if m:
        parts = [p for p in _SPLIT_RE.split(m.group(1)) if p]
        nums = [_to_number(p) for p in parts]
        nums += [0.0] * (3 - len(nums))
        return {
            "r": clamp(js_round(nums[0]), 0, 255),
            "g": clamp(js_round(nums[1]), 0, 255),
            "b": clamp(js_round(nums[2]), 0, 255),
            "a": clamp(nums[3], 0, 1) if len(nums) > 3 else 1.0,
        }

    m = _HSL_RE.match(text)
    if m:
        parts = [p for p in _SPLIT_RE.split(m.group(1)) if p]
        if len(parts) < 3:
            return fb
        h = _to_number(parts[0].rstrip("deg")) / 360
        s = _to_number(parts[1].rstrip("%")) / 100
        lightness = _to_number(parts[2].rstrip("%")) / 100
        a = clamp(_to_number(parts[3].rstrip("%"), 1.0), 0, 1) if len(parts) > 3 else 1.0
        r, g, b = hsl_to_rgb(h, s, lightness)
        return {"r": r, "g": g, "b": b, "a": a}

    return fb


def _hue_to_rgb(p: float, q: float, t: float) -> float:
    tt = t
    if tt < 0:
        tt += 1
    if tt > 1:
        tt -= 1
    if tt < 1 / 6:
        return p + (q - p) * 6 * tt
    if tt < 1 / 2:
        return q
    if tt < 2 / 3:
        return p + (q - p) * (2 / 3 - tt) * 6
    return p


def hsl_to_rgb(h: float, s: float, lightness: float) -> tuple[int, int, int]:
    """HSL（すべて 0..1）を 0..255 の RGB にする。"""
    if s == 0:
        v = js_round(clamp(lightness, 0, 1) * 255)
        return (v, v, v)
    q = lightness * (1 + s) if lightness < 0.5 else lightness + s - lightness * s
    p = 2 * lightness - q
    return (
        js_round(clamp(_hue_to_rgb(p, q, h + 1 / 3), 0, 1) * 255),
        js_round(clamp(_hue_to_rgb(p, q, h), 0, 1) * 255),
        js_round(clamp(_hue_to_rgb(p, q, h - 1 / 3), 0, 1) * 255),
    )


def rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """0..255 の RGB を 0..1 の HSL にする。"""
    rn = r / 255
    gn = g / 255
    bn = b / 255
    high = max(rn, gn, bn)
    low = min(rn, gn, bn)
    lightness = (high + low) / 2
    if high == low:
        return (0.0, 0.0, lightness)
    d = high - low
    s = d / (2 - high - low) if lightness > 0.5 else d / (high + low)
    if high == rn:
        h = ((gn - bn) / d + (6 if gn < bn else 0)) / 6
    elif high == gn:
        h = ((bn - rn) / d + 2) / 6
    else:
        h = ((rn - gn) / d + 4) / 6
    return (h, s, lightness)


def color_to_css(c: Color) -> str:
    return f"rgba({c['r']}, {c['g']}, {c['b']}, {c['a']})"


def mix_color(a: Color, b: Color, t: float) -> Color:
    """2 色の線形補間（アルファは乗算しません）。"""
    return {
        "r": js_round(a["r"] + (b["r"] - a["r"]) * t),
        "g": js_round(a["g"] + (b["g"] - a["g"]) * t),
        "b": js_round(a["b"] + (b["b"] - a["b"]) * t),
        "a": a["a"] + (b["a"] - a["a"]) * t,
    }


def color_to_rgba8(c: Color) -> tuple[int, int, int, int]:
    """``Bitmap.fill()`` に渡せる 0..255 の 4 つ組にする。"""
    return (
        int(clamp(js_round(c["r"]), 0, 255)),
        int(clamp(js_round(c["g"]), 0, 255)),
        int(clamp(js_round(c["b"]), 0, 255)),
        int(clamp(js_round(c["a"] * 255), 0, 255)),
    )
