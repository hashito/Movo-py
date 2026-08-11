"""ソフトウェアラスタライザ。

ここには 2 つのラスタライザが入っています。

- **走査線の多角形塗り** — 図形・字形・マスクに使います。nonzero と even-odd の
  両方に対応し、縦は 4 倍サンプリング、横は被覆率を面積で正確に出します
- **テクスチャ付き三角形** — メッシュ変形と 3D に使います。左上規則で塗るので、
  隣り合う三角形が二重に塗られません

**重い部分は `movo.renderer.kernels` の Numba 関数に出してあります。**
このファイルは «形を組み立てて配列に直し、カーネルへ渡す» 係です。

    多角形 1 枚の塗り（1280x720）
      NumPy で囲む矩形を一括判定 … 30.4 ms
      Numba でコンパイルした走査線 … 0.296 ms（103 倍）

**「NumPy でベクトル化すれば速い」はラスタライザには当てはまりません。**
NumPy 版は O(矩形の面積 x 辺の数)、走査線は O(辺 x 行) です。

いっぽう **合成モード 22 種は全画面に一様**なので、`composite` / `blend_rgb`
（同じ大きさの 2 枚を重ねる）は NumPy の一括演算です。画素ごとに分岐が要る
`hue` / `saturation` / `color` / `luminosity` も NumPy で書けます。

**ただし `draw_bitmap`（位置をずらして重ねる）は Numba です。** 大きさが違う
2 枚の «重なる範囲だけ» を触るので、NumPy で書くと `astype` の中間配列が
10 本ほどでき、466x466 を 960x540 へ 1 枚重ねるのに 94 ms かかっていました
（カーネルなら 1.5 ms）。**帯域ではなく «中間配列を作る手間» が支配的なときは
Numba が勝ちます。** 「全画面は NumPy」を機械的に当てはめないでください。
"""

from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Sequence

import numpy as np

from . import kernels
from .kernels import (
    composite_bitmap_kernel,
    draw_textured_triangle_kernel,
    fill_coverage_rgba,
    fill_coverage_solid,
    rasterize_contours_kernel,
    stroke_capacity,
    stroke_offsets_size,
    stroke_to_contours_kernel,
)

# ── 依存の解き方 ────────────────────────────────────────────────
#
# `movo.core` は別の担当が並行して移植しています。**まだ無いときでも
# このファイルだけで動くように**、最小限の実装を用意しておきます。
try:  # pragma: no cover - core が入ったらそちらを使う
    from movo.core.color import parse_color as _core_parse_color
except Exception:  # pragma: no cover
    _core_parse_color = None

try:  # pragma: no cover
    from movo.core.bitmap import Bitmap
except Exception:  # pragma: no cover
    Bitmap = None


def clamp(value: float, lo: float, hi: float) -> float:
    """`lo` と `hi` で挟む。JS 版の `clamp` と同じ（NaN は考えません）。"""
    return lo if value < lo else (hi if value > hi else value)


_NAMED_COLORS = {
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

_HEX_RE = re.compile(r"^#([0-9a-f]{3,8})$", re.I)
_RGB_RE = re.compile(r"^rgba?\(([^)]+)\)$")
_HSL_RE = re.compile(r"^hsla?\(([^)]+)\)$")


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    if s == 0:
        v = round(clamp(l, 0, 1) * 255)
        return (v, v, v)

    def hue2rgb(p: float, q: float, t: float) -> float:
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

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (
        _js_round(clamp(hue2rgb(p, q, h + 1 / 3), 0, 1) * 255),
        _js_round(clamp(hue2rgb(p, q, h), 0, 1) * 255),
        _js_round(clamp(hue2rgb(p, q, h - 1 / 3), 0, 1) * 255),
    )


def _js_round(v: float) -> int:
    """JS の `Math.round`（0.5 は **常に上へ**）。Python の `round` は偶数丸めです。"""
    return int(math.floor(v + 0.5))


def js_number(value) -> str:
    """JS が数を文字列にするときと同じ書き方にする。

    `rgba(57, 197, 187, 1)` のような文字列を組み立てるのに使います。Python の
    `str(1.0)` は `"1.0"` ですが、JS は `"1"` です。**この文字列は保存された
    プロジェクトやログにそのまま出る**ので、JS 版と食い違わせません。
    """
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    return repr(value) if isinstance(value, float) else str(value)


def parse_color(value, fallback=(0, 0, 0, 1.0)) -> tuple[float, float, float, float]:
    """色を `(r, g, b, a)` にする。r/g/b は 0..255、a は 0..1。

    `#rgb` `#rgba` `#rrggbb` `#rrggbbaa` `rgb()/rgba()` `hsl()/hsla()`、
    CSS 風の色名、`{"r":..,"g":..,"b":..,"a":..}` を受け付けます。
    """
    if _core_parse_color is not None:  # pragma: no cover
        c = _core_parse_color(value)
        if isinstance(c, dict):
            return (c.get("r", 0), c.get("g", 0), c.get("b", 0), c.get("a", 1.0))
        return tuple(c)  # type: ignore[return-value]

    if value is None:
        return tuple(fallback)  # type: ignore[return-value]
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        a = value[3] if len(value) > 3 else 1.0
        return (
            clamp(_js_round(value[0]), 0, 255),
            clamp(_js_round(value[1]), 0, 255),
            clamp(_js_round(value[2]), 0, 255),
            clamp(a, 0, 1),
        )
    if isinstance(value, dict):
        return (
            clamp(_js_round(value.get("r", 0)), 0, 255),
            clamp(_js_round(value.get("g", 0)), 0, 255),
            clamp(_js_round(value.get("b", 0)), 0, 255),
            clamp(value.get("a", 1.0), 0, 1),
        )
    text = str(value).strip().lower()
    named = _NAMED_COLORS.get(text)
    if named:
        return named
    if text.startswith("#"):
        hexpart = text[1:]
        if len(hexpart) in (3, 4):
            expand = lambda c: int(c + c, 16)  # noqa: E731
            try:
                return (
                    expand(hexpart[0]),
                    expand(hexpart[1]),
                    expand(hexpart[2]),
                    expand(hexpart[3]) / 255 if len(hexpart) == 4 else 1.0,
                )
            except ValueError:
                return tuple(fallback)  # type: ignore[return-value]
        if len(hexpart) in (6, 8):
            try:
                return (
                    int(hexpart[0:2], 16),
                    int(hexpart[2:4], 16),
                    int(hexpart[4:6], 16),
                    int(hexpart[6:8], 16) / 255 if len(hexpart) == 8 else 1.0,
                )
            except ValueError:
                return tuple(fallback)  # type: ignore[return-value]
        return tuple(fallback)  # type: ignore[return-value]
    m = _RGB_RE.match(text)
    if m:
        parts = [p for p in re.split(r"[,\s/]+", m.group(1)) if p]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                nums.append(0.0)
        while len(nums) < 3:
            nums.append(0.0)
        return (
            clamp(_js_round(nums[0]), 0, 255),
            clamp(_js_round(nums[1]), 0, 255),
            clamp(_js_round(nums[2]), 0, 255),
            clamp(nums[3], 0, 1) if len(nums) > 3 else 1.0,
        )
    m = _HSL_RE.match(text)
    if m:
        parts = [p for p in re.split(r"[,\s/]+", m.group(1)) if p]
        try:
            h = float(re.sub(r"[^\d.eE+-].*$", "", parts[0])) / 360
            s = float(re.sub(r"[^\d.eE+-].*$", "", parts[1])) / 100
            l = float(re.sub(r"[^\d.eE+-].*$", "", parts[2])) / 100
        except (ValueError, IndexError):
            return tuple(fallback)  # type: ignore[return-value]
        a = clamp(float(parts[3]), 0, 1) if len(parts) > 3 else 1.0
        r, g, b = _hsl_to_rgb(h, s, l)
        return (r, g, b, a)
    return tuple(fallback)  # type: ignore[return-value]


def mix_color(a, b, t):
    """2 色を混ぜる（ストレートアルファのまま線形補間）。"""
    return (
        _js_round(a[0] + (b[0] - a[0]) * t),
        _js_round(a[1] + (b[1] - a[1]) * t),
        _js_round(a[2] + (b[2] - a[2]) * t),
        a[3] + (b[3] - a[3]) * t,
    )


# ══════════════════════════════════════════════════════════════════
# 合成モード
# ══════════════════════════════════════════════════════════════════

#: 使える合成モードの一覧。
#:
#: `blend` はスキーマ上ただの文字列なので、**綴りを間違えても黙って `normal`**
#: になります。気付けるように一覧を公開し、意味検証で照合します。
#: **並び順が `kernels` の番号そのもの**なので、途中に足さないでください。
BLEND_MODES = [
    "normal",
    "add",
    "subtract",
    "screen",
    "multiply",
    "overlay",
    "hardLight",
    "softLight",
    "colorDodge",
    "colorBurn",
    "linearBurn",
    "linearLight",
    "vividLight",
    "pinLight",
    "darken",
    "lighten",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
]

_BLEND_IDS = {name: index for index, name in enumerate(BLEND_MODES)}

#: 3 チャンネルまとめて計算する必要がある合成モード。
NON_SEPARABLE = frozenset({"hue", "saturation", "color", "luminosity"})

#: ⚠ **`draw_bitmap`（ビットマップどうしの重ね）で効く合成モードはこれだけです。**
#:
#: `fill_coverage` は `BLEND_MODES` の 22 種すべてを見ますが、ビットマップを
#: 重ねる経路は **9 種しか見ません**。JS 版がそういう作りだからです
#: （`core/src/bitmap.js` が `raster.js` とは別に短い一覧と `blendChannel` を
#: 持っていて、そこに無い名前は黙って `normal` になります）。
#:
#: 直したくなりますが、**直すと JS 版と絵が変わります**。`softLight` を指定した
#: プロジェクトは JS 版では `normal` で出ているので、Python 版だけ効かせると
#: 同じ JSON から違う動画が出てしまいます。互換のためにこの «穴» ごと移します。
COMPOSITE_BLEND_MODES = frozenset(
    {"normal", "add", "screen", "multiply", "overlay", "darken", "lighten", "difference", "subtract"}
)


def blend_id(name: str | None) -> int:
    """合成モードの名前を番号にする。知らない名前は `normal`（JS 版と同じ）。"""
    return _BLEND_IDS.get(name or "normal", 0)


def _u8_array(v: np.ndarray) -> np.ndarray:
    """JS の `Uint8ClampedArray` への代入と同じ丸め方（五捨五入）で uint8 にする。"""
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def blend_rgb(cb: np.ndarray, cs: np.ndarray, blend: str) -> np.ndarray:
    """**全画面ぶんの** 合成モード計算（NumPy の一括演算）。

    `cb`（下）と `cs`（上）は 0..255 の float 配列で、末尾の次元が RGB の 3 です。
    画素ごとの分岐が要る `hue` / `saturation` / `color` / `luminosity` も
    `np.where` で書けるので、ここは NumPy が最速です（メモリ帯域が支配的）。
    """
    if blend == "normal" or blend not in _BLEND_IDS:
        return cs
    if blend in NON_SEPARABLE:
        return _blend_non_separable_np(blend, cb, cs)
    if blend == "add":
        return np.minimum(255.0, cb + cs)
    if blend == "subtract":
        return np.maximum(0.0, cb - cs)
    if blend == "screen":
        return 255.0 - ((255.0 - cb) * (255.0 - cs)) / 255.0
    if blend == "multiply":
        return (cb * cs) / 255.0
    if blend == "overlay":
        return np.where(cb < 128.0, (2.0 * cb * cs) / 255.0, 255.0 - (2.0 * (255.0 - cb) * (255.0 - cs)) / 255.0)
    if blend == "hardLight":
        return np.where(cs < 128.0, (2.0 * cb * cs) / 255.0, 255.0 - (2.0 * (255.0 - cb) * (255.0 - cs)) / 255.0)
    if blend == "softLight":
        b = cb / 255.0
        sv = cs / 255.0
        d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(np.maximum(b, 0.0)))
        out = np.where(sv <= 0.5, b - (1.0 - 2.0 * sv) * b * (1.0 - b), b + (2.0 * sv - 1.0) * (d - b))
        return out * 255.0
    if blend == "colorDodge":
        denom = np.where(cs >= 255.0, 1.0, 255.0 - cs)
        return np.where(cs >= 255.0, 255.0, np.minimum(255.0, (cb * 255.0) / denom))
    if blend == "colorBurn":
        denom = np.where(cs <= 0.0, 1.0, cs)
        return np.where(cs <= 0.0, 0.0, 255.0 - np.minimum(255.0, ((255.0 - cb) * 255.0) / denom))
    if blend == "linearBurn":
        return np.maximum(0.0, cb + cs - 255.0)
    if blend == "linearLight":
        return np.clip(cb + 2.0 * cs - 255.0, 0.0, 255.0)
    if blend == "vividLight":
        low_d = np.where(cs <= 0.0, 1.0, 2.0 * cs)
        low = np.where(cs <= 0.0, 0.0, 255.0 - np.minimum(255.0, ((255.0 - cb) * 255.0) / low_d))
        high_d = np.where(cs >= 255.0, 1.0, 2.0 * (255.0 - cs))
        high = np.where(cs >= 255.0, 255.0, np.minimum(255.0, (cb * 255.0) / high_d))
        return np.where(cs < 128.0, low, high)
    if blend == "pinLight":
        return np.where(cs < 128.0, np.minimum(cb, 2.0 * cs), np.maximum(cb, 2.0 * cs - 255.0))
    if blend == "darken":
        return np.minimum(cb, cs)
    if blend == "lighten":
        return np.maximum(cb, cs)
    if blend == "difference":
        return np.abs(cb - cs)
    if blend == "exclusion":
        return cb + cs - (2.0 * cb * cs) / 255.0
    return cs


def _lum_np(c: np.ndarray) -> np.ndarray:
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]


def _clip_color_np(c: np.ndarray) -> np.ndarray:
    l = _lum_np(c)[..., None]
    mn = c.min(axis=-1, keepdims=True)
    mx = c.max(axis=-1, keepdims=True)
    out = c
    denom = np.where(l - mn == 0, 1.0, l - mn)
    lowered = l + ((out - l) * l) / denom
    out = np.where(mn < 0, lowered, out)
    denom2 = np.where(mx - l == 0, 1.0, mx - l)
    raised = l + ((out - l) * (255.0 - l)) / denom2
    return np.where(mx > 255.0, raised, out)


def _set_lum_np(c: np.ndarray, l: np.ndarray) -> np.ndarray:
    d = l - _lum_np(c)
    return _clip_color_np(c + d[..., None])


def _set_sat_np(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """最大・中間・最小の位置を保ったまま幅だけ `s` にそろえる。

    `argsort(kind="stable")` を使うのは、同じ値が並んだときの順を JS の
    `Array.sort`（安定）と合わせるためです。
    """
    order = np.argsort(c, axis=-1, kind="stable")
    take = lambda idx: np.take_along_axis(c, idx[..., None], axis=-1)[..., 0]  # noqa: E731
    min_i = order[..., 0]
    mid_i = order[..., 1]
    max_i = order[..., 2]
    cmin = take(min_i)
    cmid = take(mid_i)
    cmax = take(max_i)
    span = np.where(cmax - cmin == 0, 1.0, cmax - cmin)
    mid_value = np.where(cmax > cmin, ((cmid - cmin) * s) / span, 0.0)
    max_value = np.where(cmax > cmin, s, 0.0)
    out = np.zeros_like(c)
    np.put_along_axis(out, mid_i[..., None], mid_value[..., None], axis=-1)
    np.put_along_axis(out, max_i[..., None], max_value[..., None], axis=-1)
    return out


def _sat_np(c: np.ndarray) -> np.ndarray:
    return c.max(axis=-1) - c.min(axis=-1)


def _blend_non_separable_np(blend: str, cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    if blend == "hue":
        return _set_lum_np(_set_sat_np(cs, _sat_np(cb)), _lum_np(cb))
    if blend == "saturation":
        return _set_lum_np(_set_sat_np(cb, _sat_np(cs)), _lum_np(cb))
    if blend == "color":
        return _set_lum_np(cs, _lum_np(cb))
    return _set_lum_np(cb, _lum_np(cs))  # luminosity


def composite(dst: np.ndarray, src: np.ndarray, alpha: float = 1.0, blend: str = "normal") -> None:
    """`dst` の上に `src` を合成モード付きで重ねる（その場で書き換え）。

    **全画面ぶんを 1 回の NumPy 演算で処理します。** 画素ごとに Python の `for` を
    回すと 1280x720 で 720 ミリ秒かかりますが、これなら 13 ミリ秒ほどです。

    :param dst: `(h, w, 4)` の uint8。書き換えられます
    :param src: `(h, w, 4)` の uint8。同じ形であること
    """
    if alpha <= 0:
        return
    # JS は `(a / 255) * alpha` の順で割ってから掛けます。`a * (alpha / 255)` と
    # まとめると float64 の丸めが変わり、a = 33 から食い違って画素が 1 ずれます。
    sa = src[..., 3].astype(np.float64) / 255.0 * alpha
    if not sa.any():
        return
    da = dst[..., 3].astype(np.float64) / 255.0
    out_a = sa + da * (1.0 - sa)
    safe = np.where(out_a > 0, out_a, 1.0)
    cb = dst[..., :3].astype(np.float64)
    cs = src[..., :3].astype(np.float64)
    mixed = blend_rgb(cb, cs, blend)
    sa3 = sa[..., None]
    da3 = da[..., None]
    result = (mixed * sa3 + cb * da3 * (1.0 - sa3)) / safe[..., None]
    dst[..., :3] = np.where(out_a[..., None] > 0, _u8_array(result), dst[..., :3])
    dst[..., 3] = _u8_array(out_a * 255.0)


def draw_bitmap(dst, src, dx: float = 0, dy: float = 0, alpha: float = 1.0, blend: str = "normal"):
    """`dst` の (dx, dy) に `src` を重ねる（JS 版の `compositeBitmap` と同じ）。

    **`Bitmap.draw` ではなくこちらを使ってください。** 丸め方が違うと 1 ずれます
    （JS の `Uint8ClampedArray` は五捨五入、NumPy の `astype` は切り捨て）。
    影・枠・文字を重ねるところは «JS 版と同じ絵» が要るので、ここに寄せます。

    ⚠ **合成モードは 9 種しか効きません**（`COMPOSITE_BLEND_MODES`）。
    `fill_coverage` の 22 種とは別なので、取り違えないでください。JS 版の
    `compositeBitmap` が自前の短い一覧で照合していて、そこに無い名前は黙って
    `normal` に落ちるためです。互換のためにその «穴» ごと移植しています。
    """
    if alpha <= 0:
        return dst
    dst_data = _bitmap_data(dst)
    src_data = _bitmap_data(src)
    if src_data.size == 0:
        return dst
    # **ここを NumPy で書かないでください。** 1 枚重ねるだけで float64 の中間配列が
    # 10 本ほどでき、466x466 を 960x540 へ重ねるのに 94 ms かかっていました
    # （`astype` は毎回コピーを作ります）。カーネルなら 1.5 ms です。
    #
    # 帯域ではなく «中間配列を作る手間» が支配的なので、全画面に一様な演算
    # （`composite` や `blend_rgb`）とは逆の判断になります。エフェクト側が
    # «gather が支配的な処理はベクトル化しないほうが速い» と言っているのと同じ話です。
    #
    # 重なる範囲の刈り取りもカーネルの中でやります。ここで配列を切ると
    # 非連続なビューになって、かえって遅くなるためです。
    ox = _js_round(dx)
    oy = _js_round(dy)
    # 一覧に無いモードは `normal`。**`blend_id` に丸投げしないこと**（あちらは
    # 22 種を知っているので、JS 版なら効かないはずのモードが効いてしまいます）。
    mode = blend_id(blend) if blend in COMPOSITE_BLEND_MODES else 0
    composite_bitmap_kernel(dst_data, src_data, ox, oy, float(alpha), mode)
    return dst


def blend_pixel(data: np.ndarray, x: int, y: int, r, g, b, sa, blend: str = "normal") -> None:
    """1 画素だけ合成する。**普段は使わないでください**（Python の呼び出しが重い）。

    互換のために残してあります。まとまった量を塗るときは `fill_coverage` か
    `composite` を使ってください。
    """
    kernels.blend_pixel(data, y, x, float(r), float(g), float(b), float(sa), blend_id(blend))


# ══════════════════════════════════════════════════════════════════
# 走査線ラスタライザ
# ══════════════════════════════════════════════════════════════════


class Region:
    """カバレッジ（被覆率）と、その «触るべき範囲»。

    `coverage` は `(height, width)` の float32 です。JS 版は 1 次元でしたが、
    2 次元で持つほうが NumPy と噛み合います（速度は変わりません）。

    `max_x < min_x` のときは «何も塗るものがない» を意味します。
    """

    __slots__ = ("coverage", "min_x", "min_y", "max_x", "max_y")

    def __init__(self, coverage: np.ndarray, min_x: int, min_y: int, max_x: int, max_y: int) -> None:
        self.coverage = coverage
        self.min_x = int(min_x)
        self.min_y = int(min_y)
        self.max_x = int(max_x)
        self.max_y = int(max_y)

    @property
    def is_empty(self) -> bool:
        return self.max_x < self.min_x or self.max_y < self.min_y

    @property
    def width(self) -> int:
        return self.coverage.shape[1]

    @property
    def height(self) -> int:
        return self.coverage.shape[0]


def pack_contours(contours: Iterable[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """輪郭の並びを «1 本のつながった配列 + 切れ目» に詰め直す。

    Numba へ渡せるのは NumPy 配列だけなので、Python のリストのリストは
    ここで «平らな float64 の列» に直します。
    """
    flat: list[np.ndarray] = []
    offsets = [0]
    cursor = 0
    for contour in contours:
        if contour is None:
            continue
        arr = np.asarray(contour, dtype=np.float64).ravel()
        if arr.size < 4:
            # 点が 2 つ未満の輪郭は辺が張れないので捨てます（JS 版と同じ）
            continue
        flat.append(arr)
        cursor += arr.size
        offsets.append(cursor)
    if not flat:
        return np.empty(0, np.float64), np.zeros(1, np.int64)
    return np.concatenate(flat), np.asarray(offsets, np.int64)


def rasterize_contours(
    contours: Iterable[Sequence[float]],
    width: int,
    height: int,
    fill_rule: str = "nonzero",
    coverage: np.ndarray | None = None,
) -> Region:
    """輪郭の集まりからカバレッジを作る。

    :param contours: それぞれ `[x0, y0, x1, y1, ...]` の並び。**常に閉じたもの**として扱います
    :param fill_rule: `"nonzero"`（既定）か `"evenodd"`
    :param coverage: 使い回したい配列があれば渡せます（0 埋めしてから使います）
    """
    width = int(width)
    height = int(height)
    if coverage is None:
        coverage = np.zeros((height, width), np.float32)
    else:
        coverage.fill(0)
    verts, offsets = pack_contours(contours)
    bbox = np.zeros(4, np.int64)
    if verts.size == 0:
        return Region(coverage, 0, 0, -1, -1)
    rasterize_contours_kernel(
        verts, offsets, width, height, 1 if fill_rule == "evenodd" else 0, coverage, bbox
    )
    return Region(coverage, bbox[0], bbox[1], bbox[2], bbox[3])


def _bitmap_data(bitmap) -> np.ndarray:
    """`Bitmap` でも生の NumPy 配列でも受け取れるようにする。"""
    data = getattr(bitmap, "data", bitmap)
    if data.ndim == 1:  # pragma: no cover - 1 次元で持っている実装への保険
        h = bitmap.height
        w = bitmap.width
        data = data.reshape(h, w, 4)
    return data


def fill_coverage(bitmap, region: Region, color, alpha: float = 1.0, blend: str = "normal"):
    """カバレッジを通して 1 色を乗せる。"""
    if region.is_empty:
        return bitmap
    r, g, b, a = parse_color(color)
    data = _bitmap_data(bitmap)
    max_x = min(region.max_x, data.shape[1] - 1)
    max_y = min(region.max_y, data.shape[0] - 1)
    min_x = max(region.min_x, 0)
    min_y = max(region.min_y, 0)
    if max_x < min_x or max_y < min_y:
        return bitmap
    fill_coverage_solid(
        data, region.coverage, min_x, min_y, max_x, max_y,
        float(r), float(g), float(b), float(a), float(alpha), blend_id(blend),
    )
    return bitmap


def fill_coverage_colors(bitmap, region: Region, colors: np.ndarray, alpha: float = 1.0, blend: str = "normal"):
    """カバレッジを通して «画素ごとに違う色» を乗せる。

    :param colors: `(囲む矩形の高さ, 幅, 4)` の float64。RGB は 0..255、A は 0..1
    """
    if region.is_empty:
        return bitmap
    data = _bitmap_data(bitmap)
    max_x = min(region.max_x, data.shape[1] - 1)
    max_y = min(region.max_y, data.shape[0] - 1)
    min_x = max(region.min_x, 0)
    min_y = max(region.min_y, 0)
    if max_x < min_x or max_y < min_y:
        return bitmap
    fill_coverage_rgba(
        data, region.coverage, min_x, min_y, max_x, max_y,
        np.ascontiguousarray(colors, dtype=np.float64), float(alpha), blend_id(blend),
    )
    return bitmap


def fill_coverage_with(bitmap, region: Region, shader: Callable, alpha: float = 1.0, blend: str = "normal"):
    """シェーダで色を作りながら塗る（グラデーション用）。

    **シェーダは «ベクトル化» の約束です。** `shader(xs, ys)` に `(h, w)` の
    座標配列を渡すので、`(h, w, 4)` の色（RGB 0..255・A 0..1）を返してください。
    1 画素ずつ Python の関数を呼ぶと、そこだけで数百ミリ秒かかります。
    """
    if region.is_empty:
        return bitmap
    data = _bitmap_data(bitmap)
    max_x = min(region.max_x, data.shape[1] - 1)
    max_y = min(region.max_y, data.shape[0] - 1)
    min_x = max(region.min_x, 0)
    min_y = max(region.min_y, 0)
    if max_x < min_x or max_y < min_y:
        return bitmap
    ys, xs = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    colors = shader(xs.astype(np.float64), ys.astype(np.float64))
    return fill_coverage_colors(bitmap, region, colors, alpha, blend)


def fill_contours(bitmap, contours, color, fill_rule: str = "nonzero", alpha: float = 1.0, blend: str = "normal"):
    """輪郭をそのままビットマップへ塗る（作って塗るだけの近道）。"""
    data = _bitmap_data(bitmap)
    region = rasterize_contours(contours, data.shape[1], data.shape[0], fill_rule)
    return fill_coverage(bitmap, region, color, alpha, blend)


# ══════════════════════════════════════════════════════════════════
# テクスチャ付き三角形
# ══════════════════════════════════════════════════════════════════


def draw_textured_triangle(
    dst,
    src,
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    alpha: float = 1.0,
    blend: str = "normal",
    clamp_edge: bool = False,
    tint=None,
    depth=None,
):
    """テクスチャを貼った三角形を描く。

    :param a: `(x, y, u, v)`。`u` `v` はテクスチャの «画素座標»
    :param tint: `(r, g, b, a)` で色を被せる（任意）
    :param depth: `{"buffer": (h,w) の float32, "z": [z0,z1,z2], "test": bool, "write": bool}`

    `depth` を渡すと «深度バッファ» を使って前後関係を画素単位で決めます。
    渡さなければ «描いた順» に上書きするだけなので、既存の呼び出しは変わりません。
    """
    if alpha <= 0:
        return
    dst_data = _bitmap_data(dst)
    src_data = _bitmap_data(src)
    vx = np.array([a[0], b[0], c[0]], np.float64)
    vy = np.array([a[1], b[1], c[1]], np.float64)
    vu = np.array([a[2], b[2], c[2]], np.float64)
    vv = np.array([a[3], b[3], c[3]], np.float64)

    if tint is None:
        tr = tg = tb = ta = 0.0
        use_tint = 0
    else:
        tr, tg, tb, ta = float(tint[0]), float(tint[1]), float(tint[2]), float(tint[3])
        use_tint = 1

    if depth is None:
        depth_buffer = np.zeros((1, 1), np.float32)
        vz = np.zeros(3, np.float64)
        use_depth = 0
        depth_test = 0
        depth_write = 0
    else:
        depth_buffer = depth["buffer"]
        if depth_buffer.ndim == 1:  # pragma: no cover
            depth_buffer = depth_buffer.reshape(dst_data.shape[0], dst_data.shape[1])
        vz = np.asarray(depth["z"], np.float64)
        use_depth = 1
        depth_test = 0 if depth.get("test") is False else 1
        depth_write = 0 if depth.get("write") is False else 1

    draw_textured_triangle_kernel(
        dst_data, src_data, vx, vy, vu, vv, float(alpha), blend_id(blend),
        1 if clamp_edge else 0, tr, tg, tb, ta, use_tint,
        depth_buffer, vz, use_depth, depth_test, depth_write,
    )


# ══════════════════════════════════════════════════════════════════
# 形を作る道具
# ══════════════════════════════════════════════════════════════════


def stroke_to_contours(points: Sequence[float], thickness: float, closed: bool = False) -> list[np.ndarray]:
    """折れ線を «塗れる形» に変える（線を太らせる）。

    **回り方（winding）は全部そろえてあります。**（Movo の issue #74）
    素直に並べると «辺の四角形» が時計回り、«継ぎ目の円» が反時計回りになり、
    重なったところで nonzero 塗りが打ち消し合って **穴が開きます**。
    円弧やトリムした線のように細かく折れると、点線のように見えてしまいます。
    """
    half = max(0.05, thickness / 2)
    pts = np.asarray(points, dtype=np.float64).ravel()
    count = pts.size // 2
    if count < 2:
        return []
    verts = np.empty(stroke_capacity(count, closed), np.float64)
    offsets = np.empty(stroke_offsets_size(count, closed), np.int64)
    n = stroke_to_contours_kernel(pts, float(half), 1 if closed else 0, verts, offsets)
    return [verts[offsets[i] : offsets[i + 1]].copy() for i in range(n)]


def circle_contour(cx: float, cy: float, radius: float, segments: int = 32) -> np.ndarray:
    angles = np.arange(segments, dtype=np.float64) / segments * (2 * math.pi)
    out = np.empty(segments * 2, np.float64)
    out[0::2] = cx + np.cos(angles) * radius
    out[1::2] = cy + np.sin(angles) * radius
    return out


def ellipse_contour(cx: float, cy: float, rx: float, ry: float, segments: int = 48, rotation: float = 0.0) -> np.ndarray:
    angles = np.arange(segments, dtype=np.float64) / segments * (2 * math.pi)
    cos = math.cos(rotation)
    sin = math.sin(rotation)
    x = np.cos(angles) * rx
    y = np.sin(angles) * ry
    out = np.empty(segments * 2, np.float64)
    out[0::2] = cx + x * cos - y * sin
    out[1::2] = cy + x * sin + y * cos
    return out


def rect_contour(x: float, y: float, width: float, height: float, radius: float = 0.0) -> list[float]:
    """長方形（`radius` を付けると角丸）。角は 8 分割で近似します。"""
    if radius <= 0:
        return [x, y, x + width, y, x + width, y + height, x, y + height]
    r = min(radius, width / 2, height / 2)
    points: list[float] = []

    def corner(cx: float, cy: float, start_angle: float) -> None:
        steps = 8
        for i in range(steps + 1):
            angle = start_angle + (i / steps) * (math.pi / 2)
            points.append(cx + math.cos(angle) * r)
            points.append(cy + math.sin(angle) * r)

    corner(x + width - r, y + r, -math.pi / 2)
    corner(x + width - r, y + height - r, 0.0)
    corner(x + r, y + height - r, math.pi / 2)
    corner(x + r, y + r, math.pi)
    return points


def flatten_quadratic(out: list, x0, y0, cx, cy, x1, y1, tolerance: float = 0.25) -> None:
    """2 次ベジェを折れ線にする（`out` の末尾に足していきます）。

    分割数は «制御点の膨らみ» から決めるので、同じ入力からは必ず同じ点列が
    出ます（決定性）。
    """
    distance = math.hypot(cx - (x0 + x1) / 2, cy - (y0 + y1) / 2)
    steps = int(clamp(math.ceil(math.sqrt(distance / tolerance) * 2), 2, 48))
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        out.append(mt * mt * x0 + 2 * mt * t * cx + t * t * x1)
        out.append(mt * mt * y0 + 2 * mt * t * cy + t * t * y1)


def flatten_cubic(out: list, x0, y0, c1x, c1y, c2x, c2y, x1, y1, tolerance: float = 0.25) -> None:
    """3 次ベジェを折れ線にする。"""
    distance = math.hypot(c1x - x0, c1y - y0) + math.hypot(c2x - c1x, c2y - c1y) + math.hypot(x1 - c2x, y1 - c2y)
    steps = int(clamp(math.ceil(math.sqrt(distance / tolerance) * 1.5), 3, 64))
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        out.append(mt**3 * x0 + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t**3 * x1)
        out.append(mt**3 * y0 + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t**3 * y1)


# ══════════════════════════════════════════════════════════════════
# 領域拡張
# ══════════════════════════════════════════════════════════════════


def expand_region_with(bitmap_class, bitmap, spec: dict, scale: float = 1.0):
    """描画領域を外へ広げる。回転や変形で内容が切れるのを防ぎます。

    `fill` で端の扱いを選べます。

      transparent … 透明のまま（既定）
      edge        … 端 1px を外へ伸ばす（回転しても隙間ができない）
      <色>        … その色で塗る

    :param spec: `{top, right, bottom, left, all, fill}`（論理 px）
    :param scale: content のスーパーサンプリング倍率
    :returns: `{"bitmap":…, "top":…, "right":…, "bottom":…, "left":…}` か None
    """
    all_value = spec.get("all", 0) or 0
    top = max(0, _js_round(spec.get("top", all_value) if spec.get("top") is not None else all_value))
    right = max(0, _js_round(spec.get("right", all_value) if spec.get("right") is not None else all_value))
    bottom = max(0, _js_round(spec.get("bottom", all_value) if spec.get("bottom") is not None else all_value))
    left = max(0, _js_round(spec.get("left", all_value) if spec.get("left") is not None else all_value))
    if top == 0 and right == 0 and bottom == 0 and left == 0:
        return None

    px = lambda v: _js_round(v * scale)  # noqa: E731
    pt, pr, pb, pl = px(top), px(right), px(bottom), px(left)
    src = _bitmap_data(bitmap)
    height0, width0 = src.shape[0], src.shape[1]
    width = width0 + pl + pr
    height = height0 + pt + pb
    # 上限を設けておかないと、指定ミスで巨大なビットマップを作ってしまう
    if width > 16384 or height > 16384:
        return None

    out = bitmap_class(width, height)
    dst = _bitmap_data(out)
    fill = spec.get("fill", "transparent") or "transparent"
    if fill not in ("transparent", "edge"):
        r, g, b, a = parse_color(fill, (0, 0, 0, 0.0))
        dst[..., 0] = int(r)
        dst[..., 1] = int(g)
        dst[..., 2] = int(b)
        dst[..., 3] = _u8_array(np.float64(a * 255))

    dst[pt : pt + height0, pl : pl + width0] = src

    if fill == "edge":
        # 端 1px を外側へ複製する。角は最寄りの端の色になる。
        ys = np.clip(np.arange(height) - pt, 0, height0 - 1)
        xs = np.clip(np.arange(width) - pl, 0, width0 - 1)
        stretched = src[ys[:, None], xs[None, :]]
        inside = np.zeros((height, width), bool)
        inside[pt : pt + height0, pl : pl + width0] = True
        dst[...] = np.where(inside[..., None], dst, stretched)

    return {"bitmap": out, "top": top, "right": right, "bottom": bottom, "left": left}


__all__ = [
    "BLEND_MODES",
    "COMPOSITE_BLEND_MODES",
    "NON_SEPARABLE",
    "Region",
    "blend_id",
    "blend_pixel",
    "blend_rgb",
    "circle_contour",
    "clamp",
    "composite",
    "draw_bitmap",
    "draw_textured_triangle",
    "ellipse_contour",
    "expand_region_with",
    "fill_contours",
    "fill_coverage",
    "fill_coverage_colors",
    "fill_coverage_with",
    "flatten_cubic",
    "flatten_quadratic",
    "js_number",
    "mix_color",
    "pack_contours",
    "parse_color",
    "rasterize_contours",
    "rect_contour",
    "stroke_to_contours",
]
