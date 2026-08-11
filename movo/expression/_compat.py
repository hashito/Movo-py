"""JS の «型変換の意味» を写した小さな道具箱。

`movo.core` にあるもの（エラー・数学・乱数・色・バージョン）はここで **再輸出するだけ**
です。実装を持ちません。乱数とノイズを 2 か所に持つと、片方だけ直したときに
«同じ JSON から違う動画» が出て、しかも絵は出てしまうので気付けません。

ここに実装があるのは、**JS に素直に書き直すと数が変わってしまう変換** だけです。

- `Number(x)` は `""` を 0 にする。`float("")` は例外
- `%` は JS が «割られる数の符号»、Python は «割る数の符号»（-1 % 3 が 2 になる）
- `Boolean([])` は JS で真。Python の `bool([])` は偽
- `String(1.0)` は JS で "1"。Python の `str(1.0)` は "1.0"
- JS には `undefined` がある。`null` と «そこに何も無い» は別物

`Math.round`（0.5 は上へ）と 32 ビットの折り返しは `movo.core.math` / `movo.core.rng`
にあるものを使います。
"""

from __future__ import annotations

import math
import re

from movo.core.color import hsl_to_rgb, parse_color
from movo.core.errors import ErrorCodes, MovoError, MovoValidationError
from movo.core.math import (
    DEG,
    TAU,
    catmull_rom,
    clamp,
    inverse_lerp,
    js_round,
    lerp,
    sample_polyline,
    smoothstep,
    to_degrees,
    to_radians,
)
from movo.core.rng import (
    RandomSource,
    create_random,
    fbm1d as fbm_1d,
    hash_string,
    value_noise_1d,
    value_noise_2d,
    value_noise_3d,
)
from movo.core.version import (
    MOVO_JSON_VERSION,
    MOVO_VERSION,
    is_compatible_json_version,
)

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# undefined
# ---------------------------------------------------------------------------


class _Undefined:
    """JS の `undefined`。

    JSON に `undefined` は無いので «辞書にキーが無い» で足りることが多いのですが、
    式の評価と animation の解決では `null`（値として null が書いてある）と
    `undefined`（そこに何も無い）を区別しないと結果が変わります。
    たとえば `resolve_animated` の «サンプルできなかったら既定値» の判定や、
    `apply_animations` が組む指定の «キーはあるが値は無い» という状態です。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "undefined"


UNDEFINED = _Undefined()


def is_nullish(value) -> bool:
    """JS の `x == null`（null または undefined）。"""
    return value is None or value is UNDEFINED


# ---------------------------------------------------------------------------
# 32 ビット整数（JS の `>>> 0` と `Math.imul`）
# ---------------------------------------------------------------------------


def u32(x: int) -> int:
    """JS の `x >>> 0`。"""
    return x & 0xFFFFFFFF


def i32(x: int) -> int:
    """JS の `x | 0`（符号付き 32 ビットへ畳む）。"""
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def imul(a: int, b: int) -> int:
    """JS の `Math.imul(a, b)`。符号付き 32 ビットの積。"""
    return i32(u32(a) * u32(b))


# ---------------------------------------------------------------------------
# JS の型変換
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def js_number(value) -> float:
    """JS の `Number(value)`。変換できなければ NaN。

    `Number("")` が 0、`Number(null)` が 0、`Number(undefined)` が NaN という
    細かいところまで写しています。相対単位や拍の判定がここに乗っています。
    """
    if value is None:
        return 0.0
    if value is UNDEFINED:
        return NAN
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0.0
        if text in ("Infinity", "+Infinity"):
            return INF
        if text == "-Infinity":
            return -INF
        try:
            if text[:2].lower() == "0x":
                return float(int(text, 16))
            if text[:2].lower() == "0o":
                return float(int(text, 8))
            if text[:2].lower() == "0b":
                return float(int(text, 2))
        except ValueError:
            return NAN
        if not _NUMERIC_RE.match(text):
            return NAN
        try:
            return float(text)
        except ValueError:
            return NAN
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        if len(value) == 1:
            return js_number(value[0])
        return NAN
    return NAN


def is_finite_number(value) -> bool:
    """JS の `typeof v === 'number' && Number.isFinite(v)`。

    Python では `True` が `int` の仲間なので、真偽値を数として扱わないよう外します。
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def js_trunc(x: float):
    if math.isnan(x) or math.isinf(x):
        return x
    return math.trunc(x)


def js_sign(x: float):
    if math.isnan(x):
        return NAN
    return (x > 0) - (x < 0)


def js_mod(a: float, b: float) -> float:
    """JS の `%`。余りの符号は «割られる数» に従う。"""
    if b == 0 or math.isnan(a) or math.isnan(b) or math.isinf(a):
        return NAN
    if math.isinf(b):
        return a
    return math.fmod(a, b)


def js_string(value) -> str:
    """JS の `String(value)`。`str(1.0)` が "1.0" になる罠を避ける。"""
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_integer() and abs(value) < 1e21:
            return str(int(value))
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join("" if is_nullish(v) else js_string(v) for v in value)
    return "[object Object]"


def js_truthy(value) -> bool:
    """JS の真偽判定。**空配列・空辞書は真** なので Python の `bool()` は使えない。"""
    if value is UNDEFINED or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0 and not math.isnan(value)
    if isinstance(value, str):
        return value != ""
    return True


def is_plain_object(value) -> bool:
    """JSON のオブジェクト（配列でない入れ物）か。"""
    return isinstance(value, dict)


__all__ = [
    "DEG",
    "ErrorCodes",
    "INF",
    "MOVO_JSON_VERSION",
    "MOVO_VERSION",
    "MovoError",
    "MovoValidationError",
    "NAN",
    "RandomSource",
    "TAU",
    "UNDEFINED",
    "catmull_rom",
    "clamp",
    "create_random",
    "fbm_1d",
    "hash_string",
    "hsl_to_rgb",
    "i32",
    "imul",
    "inverse_lerp",
    "is_compatible_json_version",
    "is_finite_number",
    "is_nullish",
    "is_plain_object",
    "js_mod",
    "js_number",
    "js_round",
    "js_sign",
    "js_string",
    "js_trunc",
    "js_truthy",
    "lerp",
    "parse_color",
    "sample_polyline",
    "smoothstep",
    "to_degrees",
    "to_radians",
    "u32",
    "value_noise_1d",
    "value_noise_2d",
    "value_noise_3d",
]
