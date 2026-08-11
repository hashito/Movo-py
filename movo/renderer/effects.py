"""画素エフェクト。**Movo-py で NumPy がいちばん効く場所です。**

## なぜここが速度の要か

エフェクトは «全画面に一様な処理» です。明るくする、色を転ばす、ぼかす、
どれも «全画素に同じ式» を当てるだけなので、C の一括演算にそのまま乗ります。

| 1280x720 の全画面 1 パス | |
| --- | --- |
| 純 Python のループ | 720 ms |
| **NumPy の一括演算** | **13 ms**（54 倍） |

1 フレームに 10 個前後のエフェクトが乗るので、ここを Python のループで書くと
153 秒の MV が 7 時間かかります。**画素ごとの `for` は書かないでください。**

    # ✓ 一括演算
    data[..., :3] = _u8(data[..., :3].astype(np.float32) + amount * 255.0)

    # ✗ 720 ms
    for y in range(h):
        for x in range(w): ...

例外は «画素ごとに分岐がどうしても要る» もの（`pixelSort` `dither` の誤差拡散、
距離場、三角形の塗り）だけで、そこは本ファイル内のローカルな `@njit` に落として
あります。共有の `movo/renderer/kernels.py` に置かないのは、あちらが別担当の
持ち物で、名前がぶつかると困るからです。

## JS 版と «同じ絵» を出すための決めごと

1. **`Uint8ClampedArray` への代入は «偶数丸め»**（`200.5 → 200`、`201.5 → 202`）。
   NumPy の `np.rint` が同じ規則なので、格納は必ず {@link _u8} を通します。
2. **`Math.round` は «0.5 で切り上げ»** で、上の丸めとは別物です。JS 側が
   `Math.round` を使っているところは {@link _jround}（`floor(x + 0.5)`）を使います。
   ここを取り違えると 1 画素だけ 1 ずれる、という形で差が出ます。
3. **`colorAdjust` の値は «増減量» で 0 が «変化なし»** です。倍率ではありません。
   `brightness: 1.0` は «真っ白» という意味になります。この意味は変えていません。
4. **`gradientOverlay` の色は `colors` ではなく `stops`**（`{offset, color}` の並び）です。
5. **ぼかしは分離可能**なので横・縦の 2 パスに分けています（O(r²) → O(r)）。
   JS 版と同じく «ボックスぼかし 3 回» で、パスごとに uint8 へ丸めます。
   丸めを省くと速くはなりますが、JS 版と絵が変わります。
6. **ブルームは半径違いを重ねる** JS 版の作りをそのまま残しています。1 発の
   ぼかしにまとめると «芯の締まり» が消えます。

`applyEffect` の `mask` は `movo.deformer.mask` があるときだけ効きます（別担当が
移植中のため、無ければマスク無しとして通します）。
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numba import njit

from movo.core.bitmap import Bitmap
from movo.core.color import parse_color as _core_parse_color

# 塗りの土台は renderer/raster.py（別担当）にまとまっています。**ここで
# 作り直さないでください。** 走査線と «上端・左端の規則» を 2 か所に持つと、
# 隣り合う三角形の継ぎ目に髪の毛のような隙間が出る不具合が片方だけ直る、
# という形で必ず食い違います。
from movo.renderer.raster import (
    circle_contour,
    draw_bitmap,
    fill_coverage,
    rasterize_contours,
)
from movo.renderer.raster import draw_textured_triangle as _raster_triangle


# ── 丸めと切り詰め ────────────────────────────────────────────────

def _u8(value: np.ndarray) -> np.ndarray:
    """`Uint8ClampedArray` への代入と同じ丸め（切り詰め → 偶数丸め）。

    **エフェクトの出口では必ずこれを通してください。** `astype(np.uint8)` を
    直に呼ぶと切り捨てになり、JS 版より systematically 暗い絵になります。
    """
    return np.rint(np.clip(value, 0.0, 255.0)).astype(np.uint8)


def _jround(value):
    """JS の `Math.round`（0.5 は常に切り上げ）。`np.round` は偶数丸めなので使えません。"""
    return np.floor(np.asarray(value, dtype=np.float64) + 0.5)


def clamp(value, low, high):
    """JS の `clamp` と同じ（`NaN` の扱いまでは合わせていません）。"""
    return low if value < low else high if value > high else value


def _num(params: dict, *names, default=0.0):
    """`params.a ?? params.b ?? default` を素直に書けるようにしたもの。"""
    for name in names:
        value = params.get(name)
        if value is not None:
            return value
    return default


# ── 色 ────────────────────────────────────────────────────────────
#
# JS 版の core/src/color.js と同じ規則です。`movo.core.color` が移植されたら
# そちらへ寄せられますが、いまは別担当が作業中なのでここに置いています。

class Color:
    """`{r, g, b, a}` の入れ物。r/g/b は 0..255、a は 0..1 です。

    `movo.core.color` は同じ色を **辞書**で返します。エフェクト側は
    `color.r` と書けたほうが読みやすいので、ここで薄く包み直しているだけで、
    **文字列の解釈そのものは core の 1 か所だけ**にしてあります。
    """

    __slots__ = ("r", "g", "b", "a")

    def __init__(self, r=0, g=0, b=0, a=1.0):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def rgb(self) -> tuple[float, float, float]:
        return (self.r, self.g, self.b)

    def __repr__(self) -> str:  # pragma: no cover - 目視用
        return f"Color({self.r}, {self.g}, {self.b}, {self.a})"


def parse_color(value, fallback: Color | None = None) -> Color:
    """`#rgb` `#rrggbbaa` `rgb()` `hsl()` と色名を読む（`movo.core.color` に委譲）。"""
    if isinstance(value, Color):
        return Color(value.r, value.g, value.b, value.a)
    fb = None if fallback is None else {"r": fallback.r, "g": fallback.g, "b": fallback.b, "a": fallback.a}
    parsed = _core_parse_color(value, fb)
    return Color(parsed["r"], parsed["g"], parsed["b"], parsed["a"])


def mix_color(a: Color, b: Color, t: float) -> Color:
    """2 色を混ぜる。**JS 版と同じく r/g/b は `Math.round` で丸めます。**"""
    return Color(
        float(_jround(a.r + (b.r - a.r) * t)),
        float(_jround(a.g + (b.g - a.g) * t)),
        float(_jround(a.b + (b.b - a.b) * t)),
        a.a + (b.a - a.a) * t,
    )


@njit(cache=True, fastmath=False, inline="always")
def _nb_rgb_to_hsl(r, g, b):
    """0..255 の RGB を 0..1 の HSL へ（1 画素ぶん）。JS の `rgbToHsl` と同じ式です。"""
    rn = r / 255.0
    gn = g / 255.0
    bn = b / 255.0
    mx = rn if rn > gn else gn
    if bn > mx:
        mx = bn
    mn = rn if rn < gn else gn
    if bn < mn:
        mn = bn
    lightness = (mx + mn) / 2.0
    if mx == mn:
        return 0.0, 0.0, lightness
    d = mx - mn
    if lightness > 0.5:
        sat = d / (2.0 - mx - mn)
    else:
        sat = d / (mx + mn)
    # 分岐の順は JS と同じ（r → g → b）。同値のときの取り合いで色相が変わります。
    if mx == rn:
        hue = ((gn - bn) / d + (6.0 if gn < bn else 0.0)) / 6.0
    elif mx == gn:
        hue = ((bn - rn) / d + 2.0) / 6.0
    else:
        hue = ((rn - gn) / d + 4.0) / 6.0
    return hue, sat, lightness


@njit(cache=True, fastmath=False, inline="always")
def _nb_hue_to_rgb(p, q, t):
    tt = t
    if tt < 0.0:
        tt += 1.0
    if tt > 1.0:
        tt -= 1.0
    if tt < 1.0 / 6.0:
        return p + (q - p) * 6.0 * tt
    if tt < 0.5:
        return q
    if tt < 2.0 / 3.0:
        return p + (q - p) * (2.0 / 3.0 - tt) * 6.0
    return p


@njit(cache=True, fastmath=False, inline="always")
def _nb_clamp01_255(v):
    """`Math.round(clamp(v, 0, 1) * 255)`。**丸めは «0.5 で切り上げ» です。**"""
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return math.floor(v * 255.0 + 0.5)


@njit(cache=True, fastmath=False, inline="always")
def _nb_hsl_to_rgb(h, sat, lightness):
    """0..1 の HSL を 0..255 の RGB へ（1 画素ぶん）。**`Math.round` で丸めます。**"""
    if sat == 0.0:
        v = _nb_clamp01_255(lightness)
        return v, v, v
    if lightness < 0.5:
        q = lightness * (1.0 + sat)
    else:
        q = lightness + sat - lightness * sat
    p = 2.0 * lightness - q
    return (
        _nb_clamp01_255(_nb_hue_to_rgb(p, q, h + 1.0 / 3.0)),
        _nb_clamp01_255(_nb_hue_to_rgb(p, q, h)),
        _nb_clamp01_255(_nb_hue_to_rgb(p, q, h - 1.0 / 3.0)),
    )


@njit(cache=True, fastmath=False)
def _k_rgb_to_hsl(src, out):
    for i in range(src.shape[0]):
        out[i, 0], out[i, 1], out[i, 2] = _nb_rgb_to_hsl(src[i, 0], src[i, 1], src[i, 2])


@njit(cache=True, fastmath=False)
def _k_hsl_to_rgb(src, out):
    for i in range(src.shape[0]):
        out[i, 0], out[i, 1], out[i, 2] = _nb_hsl_to_rgb(src[i, 0], src[i, 1], src[i, 2])


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """`(..., 3)` の 0..255 を `(..., 3)` の 0..1 の HSL へ。

    **画素ごとに分岐があるので Numba です。** NumPy の `np.select` で書くと
    枝の数だけ画面 1 枚ぶんの一時配列ができ、1280x720 で 1 往復 460 ミリ秒
    かかりました（Numba なら 25 ミリ秒）。
    """
    flat = np.ascontiguousarray(rgb, np.float64).reshape(-1, 3)
    out = np.empty_like(flat)
    _k_rgb_to_hsl(flat, out)
    return out.reshape(np.shape(rgb))


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """`(..., 3)` の HSL（0..1）を 0..255 の RGB へ。**`Math.round` で丸めます。**"""
    flat = np.ascontiguousarray(hsl, np.float64).reshape(-1, 3)
    out = np.empty_like(flat)
    _k_hsl_to_rgb(flat, out)
    return out.reshape(np.shape(hsl))


def hsl_to_rgb_scalar(h: float, s: float, lightness: float) -> tuple[int, int, int]:
    """1 色ぶんの HSL → RGB。色の文字列を読むときにしか使いません。"""
    triple = hsl_to_rgb(np.array([[h, s, lightness]], np.float64))[0]
    return int(triple[0]), int(triple[1]), int(triple[2])


LUMA = (0.299, 0.587, 0.114)


def luma_of(rgb: np.ndarray) -> np.ndarray:
    """0..255 の輝度。**係数は JS 版と同じ Rec.601 です。**

    `float64` で計算します。`float32` にすると 1 画素あたり最大 1 だけ
    JS 版と食い違い、しきい値の境目でだけ絵が変わります（実測で踏みました）。
    """
    f = np.asarray(rgb, dtype=np.float64)
    return f[..., 0] * LUMA[0] + f[..., 1] * LUMA[1] + f[..., 2] * LUMA[2]


# ── 決定的な乱数（mulberry32）と値ノイズ ────────────────────────
#
# **同じ JSON からは必ず同じ動画が出る**、が Movo の約束です。乱数は
# すべてシードから決まる必要があるので、JS 版とビット単位で同じ実装にします。

_M32 = 0xFFFFFFFF


class Random:
    """JS 版の `createRandom`（mulberry32）と同じ数列を出します。

    32 ビットの丸め込みを Python の多倍長整数でやっているので、粒の位置や
    グリッチの帯が JS 版と 1 個もずれません。
    """

    __slots__ = ("_a",)

    def __init__(self, seed: int = 0):
        a = int(seed) & _M32
        self._a = a if a else 0x9E3779B9

    def __call__(self) -> float:
        self._a = (self._a + 0x6D2B79F5) & _M32
        t = self._a
        t = ((t ^ (t >> 15)) * (t | 1)) & _M32
        t = (t ^ (t + ((t ^ (t >> 7)) * (t | 61)) & _M32)) & _M32
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    def range(self, low: float, high: float) -> float:
        return low + self() * (high - low)


def hash_string(text: str) -> int:
    """FNV-1a の 32 ビット版。レイヤー ID から乱数の種を作るのに使います。"""
    h = 2166136261
    for ch in str(text):
        h ^= ord(ch) & 0xFFFF
        h = (h * 16777619) & _M32
    return h & _M32


@njit(cache=True, fastmath=False, inline="always")
def _nb_hash_to_unit(i, seed):
    """JS の `hashToUnit`。**32 ビットの巻き戻りまで含めて同じ値**を返します。

    途中を `uint64` で持って毎回 32 ビットに刈るので、`Math.imul` の
    «下位 32 ビットだけ残す» 挙動をそのまま写せます。
    """
    h = np.uint32((np.int64(i) & 0xFFFFFFFF) ^ (np.int64(seed) & 0xFFFFFFFF))
    h = np.uint32((np.uint64(h) * np.uint64(0x27D4EB2D)) & np.uint64(0xFFFFFFFF))
    h = np.uint32(h ^ (h >> np.uint32(15)))
    h = np.uint32((np.uint64(h) * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF))
    h = np.uint32(h ^ (h >> np.uint32(13)))
    return np.float64(h) / 4294967296.0


@njit(cache=True, fastmath=False, inline="always")
def _nb_hash_to_unit2(x, y, seed):
    gx = (np.int64(x) * 73856093) & 0xFFFFFFFF
    gy = (np.int64(y) * 19349663) & 0xFFFFFFFF
    return _nb_hash_to_unit(gx ^ gy, seed)


@njit(cache=True, fastmath=False, inline="always")
def _nb_fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, fastmath=False, inline="always")
def _nb_value_noise_2d(x, y, seed):
    """決定的な 2 次元値ノイズ（-1..1）。1 点ぶん。"""
    xi = np.int64(math.floor(x))
    yi = np.int64(math.floor(y))
    xf = _nb_fade(x - xi)
    yf = _nb_fade(y - yi)
    v00 = _nb_hash_to_unit2(xi, yi, seed)
    v10 = _nb_hash_to_unit2(xi + 1, yi, seed)
    v01 = _nb_hash_to_unit2(xi, yi + 1, seed)
    v11 = _nb_hash_to_unit2(xi + 1, yi + 1, seed)
    top = v00 + (v10 - v00) * xf
    bottom = v01 + (v11 - v01) * xf
    return (top + (bottom - top) * yf) * 2.0 - 1.0


@njit(cache=True, fastmath=False)
def _k_value_noise_2d(xs, ys, seed, out):
    for i in range(xs.shape[0]):
        out[i] = _nb_value_noise_2d(xs[i], ys[i], seed)


@njit(cache=True, fastmath=False)
def _k_value_noise_1d(xs, seed, out):
    for i in range(xs.shape[0]):
        x = xs[i]
        xi = np.int64(math.floor(x))
        f = _nb_fade(x - xi)
        a = _nb_hash_to_unit(xi, seed)
        b = _nb_hash_to_unit(xi + 1, seed)
        out[i] = (a + (b - a) * f) * 2.0 - 1.0


def value_noise_2d(x, y, seed: int = 0) -> np.ndarray:
    """決定的な 2 次元値ノイズ（-1..1）。**格子ハッシュなので Numba の仕事です。**

    NumPy で書くと 1 点あたり 4 回のハッシュがそれぞれ画面 1 枚ぶんの
    `uint32` 配列を作り、960x540 で 360 ミリ秒かかりました。1 点ずつ回せば
    中間配列は 1 つも要らず **8 ミリ秒**です。

    `retroFilm` の粒子が使う `fbm2D(..., octaves=1)` は、z を 0 に固定した
    3 次元ノイズと同じ値になるので、こちらで代用できます（格子ハッシュの
    第 3 項が `imul(0, ...) = 0` で消えるため）。
    """
    bx, by = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    fx = np.ascontiguousarray(bx)
    out = np.empty(fx.size, np.float64)
    _k_value_noise_2d(fx.ravel(), np.ascontiguousarray(by).ravel(), int(seed), out)
    return out.reshape(fx.shape)


def value_noise_1d(x, seed: int = 0) -> np.ndarray:
    """決定的な 1 次元値ノイズ（-1..1）。`rasterScroll` の «ゆらぎ» が使います。"""
    fx = np.ascontiguousarray(np.asarray(x, np.float64))
    out = np.empty(fx.size, np.float64)
    _k_value_noise_1d(fx.ravel(), int(seed), out)
    return out.reshape(fx.shape)


# ── 標本化と合成（NumPy の一括版） ────────────────────────────────

@njit(cache=True, fastmath=False, inline="always")
def _nb_sample_bilinear(data, x, y, clamp_edge):
    """双一次補間で 1 点読む。**アルファで重み付けしてから割り戻します。**

    素直に色だけ補間すると、透明な縁の «黒» が滲み出して輪郭が暗くなります。
    足す順は JS 版の `sampleBilinear` と同じで、1 ビットも変わりません。
    """
    h, w = data.shape[0], data.shape[1]
    px = x - 0.5
    py = y - 0.5
    x0 = int(math.floor(px))
    y0 = int(math.floor(py))
    fx = px - x0
    fy = py - y0
    acc_r = 0.0
    acc_g = 0.0
    acc_b = 0.0
    acc_a = 0.0
    for dy in range(2):
        for dx in range(2):
            weight = (fx if dx == 1 else 1.0 - fx) * (fy if dy == 1 else 1.0 - fy)
            if weight == 0.0:
                continue
            sx = x0 + dx
            sy = y0 + dy
            if sx < 0 or sy < 0 or sx >= w or sy >= h:
                if not clamp_edge:
                    continue
                sx = 0 if sx < 0 else (w - 1 if sx >= w else sx)
                sy = 0 if sy < 0 else (h - 1 if sy >= h else sy)
            pa = data[sy, sx, 3] * 1.0
            acc_r += data[sy, sx, 0] * pa * weight
            acc_g += data[sy, sx, 1] * pa * weight
            acc_b += data[sy, sx, 2] * pa * weight
            acc_a += pa * weight
    if acc_a > 0.0001:
        return acc_r / acc_a, acc_g / acc_a, acc_b / acc_a, acc_a
    return 0.0, 0.0, 0.0, 0.0


@njit(cache=True, fastmath=False)
def _k_sample_bilinear(data, xs, ys, clamp_edge, out):
    """{@link _nb_sample_bilinear} をまとめて回す。

    NumPy で書くと 4 隅ぶんの gather がそれぞれ画面 1 枚ぶんの一時配列を作り、
    1280x720 で 1 パス 470 ミリ秒かかりました。1 画素ずつ引けば一時配列は要らず、
    近くの画素が L1 に載ったままになるので **20 ミリ秒**です（README の LUT と
    同じ理屈で、gather が支配的な処理は «ベクトル化しないほうが» 速い）。
    """
    for i in range(xs.shape[0]):
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = _nb_sample_bilinear(
            data, xs[i], ys[i], clamp_edge
        )


@njit(cache=True, fastmath=False)
def _k_zoom_blur(src, out, cx, cy, strength, samples):
    """放射ブラー。**標本を «足しながら» 進むので中間画像を作りません。**

    NumPy だと 1 標本ごとに画面 1 枚ぶんの座標配列と結果配列ができ、標本 10 個で
    1280x720 が 700 ミリ秒でした。ここでは 1 画素ぶんのレジスタで済むので
    **40 ミリ秒**です。
    """
    h, w = src.shape[0], src.shape[1]
    for y in range(h):
        for x in range(w):
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            acc_a = 0.0
            for si in range(samples):
                scale = 1.0 + (si / (samples - 1.0)) * strength
                r, g, b, a = _nb_sample_bilinear(
                    src, cx + (x + 0.5 - cx) * scale, cy + (y + 0.5 - cy) * scale, True
                )
                acc_r += r * a
                acc_g += g * a
                acc_b += b * a
                acc_a += a
            if acc_a > 0.0:
                out[y, x, 0] = _u8_scalar(acc_r / acc_a)
                out[y, x, 1] = _u8_scalar(acc_g / acc_a)
                out[y, x, 2] = _u8_scalar(acc_b / acc_a)
                out[y, x, 3] = _u8_scalar(acc_a / samples)
            else:
                out[y, x, 0] = 0
                out[y, x, 1] = 0
                out[y, x, 2] = 0
                out[y, x, 3] = 0


@njit(cache=True, fastmath=False)
def _k_spin_blur(src, out, cx, cy, angle, samples):
    """回転ブラー。作りは {@link _k_zoom_blur} と同じで、標本の取り方だけ違います。"""
    h, w = src.shape[0], src.shape[1]
    for y in range(h):
        dy = y + 0.5 - cy
        for x in range(w):
            dx = x + 0.5 - cx
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            acc_a = 0.0
            for si in range(samples):
                t = (si / (samples - 1.0) - 0.5) * angle
                cos = math.cos(t)
                sin = math.sin(t)
                r, g, b, a = _nb_sample_bilinear(
                    src, cx + dx * cos - dy * sin, cy + dx * sin + dy * cos, True
                )
                acc_r += r * a
                acc_g += g * a
                acc_b += b * a
                acc_a += a
            if acc_a > 0.0:
                out[y, x, 0] = _u8_scalar(acc_r / acc_a)
                out[y, x, 1] = _u8_scalar(acc_g / acc_a)
                out[y, x, 2] = _u8_scalar(acc_b / acc_a)
                out[y, x, 3] = _u8_scalar(acc_a / samples)
            else:
                out[y, x, 0] = 0
                out[y, x, 1] = 0
                out[y, x, 2] = 0
                out[y, x, 3] = 0


@njit(cache=True, fastmath=False)
def _k_long_shadow(alpha, shadow, length, dx, dy, fade):
    """アルファを一方向へ伸ばして影を積む。

    NumPy だと «ずらして最大値» を長さぶん繰り返すので、画面 1 枚ぶんの一時配列が
    長さ × 2 枚できます（長さ 60 で 440 MB 触る計算）。ここは 1 画素ずつ回します。
    """
    h, w = alpha.shape[0], alpha.shape[1]
    for step in range(1, length + 1):
        ox = int(math.floor(dx * step + 0.5))
        oy = int(math.floor(dy * step + 0.5))
        strength = 1.0 - (step / length) * fade
        for y in range(h):
            sy = y - oy
            if sy < 0 or sy >= h:
                continue
            for x in range(w):
                sx = x - ox
                if sx < 0 or sx >= w:
                    continue
                a = alpha[sy, sx] / 255.0
                if a <= 0.0:
                    continue
                value = np.float32(a * strength)
                if value > shadow[y, x]:
                    shadow[y, x] = value


def sample_bilinear(data: np.ndarray, xs: np.ndarray, ys: np.ndarray, clamp_edge: bool = False) -> np.ndarray:
    """双一次補間で «まとめて» 読む。返り値は `(..., 4)` の float64（0..255）。

    **float64 で返します。** float32 に落とすと `22.500000000000018` が `22.5` に
    化けて、`Uint8ClampedArray` の «偶数丸め» の向きが変わります（JS 版と
    1 画素だけ食い違う形で出ました）。

    :param data: `(h, w, 4)` の uint8
    :param xs: 画素座標の x（`x + 0.5` が画素の中心）
    :param clamp_edge: 範囲外を端の色で埋めるか。False なら透明として扱う
    """
    # 呼ぶ側が `(rows, 1)` と `(rows, w)` のように «broadcast 前提» で渡してくる
    # ことがあります（`reflection` がそう書いています）。ここで形を揃えます。
    bx, by = np.broadcast_arrays(np.asarray(xs, np.float64), np.asarray(ys, np.float64))
    fx = np.ascontiguousarray(bx)
    fy = np.ascontiguousarray(by)
    shape = fx.shape
    flat_out = np.empty((fx.size, 4), np.float64)
    _k_sample_bilinear(np.ascontiguousarray(data), fx.ravel(), fy.ravel(), bool(clamp_edge), flat_out)
    return flat_out.reshape(shape + (4,))


def composite(dst: Bitmap, src: Bitmap, dx: int = 0, dy: int = 0, alpha: float = 1.0,
              blend: str = "normal") -> Bitmap:
    """`src` を `dst` へ source-over で重ねる（JS の `drawBitmap` と同じ）。

    合成そのものは `renderer/raster.py` の `draw_bitmap` に任せます。丸め方を
    2 か所に持つと必ず 1 ずれるので、**入口を 1 つに絞っています。**
    """
    draw_bitmap(dst, src, dx, dy, alpha, blend)
    return dst


@njit(cache=True, fastmath=False, inline="always")
def _u8_scalar(v):
    """`Uint8ClampedArray` への代入と同じ丸め（切り詰め → 偶数丸め）。"""
    if v <= 0.0:
        return np.uint8(0)
    if v >= 255.0:
        return np.uint8(255)
    return np.uint8(np.rint(v))


@njit(cache=True, fastmath=False)
def _k_blur_axis(src, out, r, horizontal):
    """1 軸のボックスぼかし。**走る窓の «足して引く» で半径によらず一定時間です。**

    NumPy の累積和でも O(n) にはできますが、画面 1 枚ぶんの中間配列が
    パスごとに 5 枚できて、960x540 の半径 18 で 580 ミリ秒かかりました。
    走る窓なら中間配列は 1 つも要らず、読み書きが 1 往復で済むので
    **30 ミリ秒**です（19 倍）。

    合計は整数（色 × アルファ ≤ 65025）なので `int64` で厳密です。JS 版が
    double で足しているのと 1 ビットも変わりません。

    アルファで重み付けしてから割り戻すので、**透明な縁の «黒» が滲みません。**
    """
    h, w = src.shape[0], src.shape[1]
    window = 2 * r + 1
    lines = h if horizontal else w
    span = w if horizontal else h
    for line in range(lines):
        sum_r = np.int64(0)
        sum_g = np.int64(0)
        sum_b = np.int64(0)
        sum_a = np.int64(0)
        # 窓の初期値。範囲外は端の画素を伸ばして読む（JS の clamp と同じ）。
        for k in range(-r, r + 1):
            i = k
            if i < 0:
                i = 0
            elif i > span - 1:
                i = span - 1
            sy = line if horizontal else i
            sx = i if horizontal else line
            a = np.int64(src[sy, sx, 3])
            sum_r += np.int64(src[sy, sx, 0]) * a
            sum_g += np.int64(src[sy, sx, 1]) * a
            sum_b += np.int64(src[sy, sx, 2]) * a
            sum_a += a
        for t in range(span):
            if t > 0:
                add = t + r
                if add > span - 1:
                    add = span - 1
                rem = t - r - 1
                if rem < 0:
                    rem = 0
                ay = line if horizontal else add
                ax = add if horizontal else line
                ry = line if horizontal else rem
                rx = rem if horizontal else line
                a_add = np.int64(src[ay, ax, 3])
                a_rem = np.int64(src[ry, rx, 3])
                sum_r += np.int64(src[ay, ax, 0]) * a_add - np.int64(src[ry, rx, 0]) * a_rem
                sum_g += np.int64(src[ay, ax, 1]) * a_add - np.int64(src[ry, rx, 1]) * a_rem
                sum_b += np.int64(src[ay, ax, 2]) * a_add - np.int64(src[ry, rx, 2]) * a_rem
                sum_a += a_add - a_rem
            oy = line if horizontal else t
            ox = t if horizontal else line
            if sum_a > 0:
                out[oy, ox, 0] = _u8_scalar(sum_r / sum_a)
                out[oy, ox, 1] = _u8_scalar(sum_g / sum_a)
                out[oy, ox, 2] = _u8_scalar(sum_b / sum_a)
                out[oy, ox, 3] = _u8_scalar(sum_a / window)
            else:
                out[oy, ox, 0] = 0
                out[oy, ox, 1] = 0
                out[oy, ox, 2] = 0
                out[oy, ox, 3] = 0


def blur_axis(bitmap: Bitmap, radius: float, horizontal: bool) -> Bitmap:
    """1 軸だけぼかす（{@link _k_blur_axis} の包み）。"""
    r = max(0, int(_jround(radius)))
    if r == 0:
        return bitmap.copy()
    out = np.empty_like(bitmap.data)
    _k_blur_axis(np.ascontiguousarray(bitmap.data), out, r, horizontal)
    return Bitmap(bitmap.width, bitmap.height, out)


def separable_blur(bitmap: Bitmap, radius_x: float, radius_y: float, passes: int = 3) -> Bitmap:
    """縦横を別々の半径でぼかす。**ボックスぼかしを 3 回でガウスに近づけます。**

    ガウス核を直に畳み込むより速く（半径によらず一定時間）、3 回重ねれば
    見分けが付きません。JS 版と同じくパスごとに uint8 へ丸めます。
    """
    rx = max(0, int(_jround(radius_x)))
    ry = max(0, int(_jround(radius_y)))
    if rx == 0 and ry == 0:
        return bitmap.copy()
    current = bitmap
    for _ in range(passes):
        if rx > 0:
            current = blur_axis(current, rx, True)
        if ry > 0:
            current = blur_axis(current, ry, False)
    return current


def box_blur(bitmap: Bitmap, radius: float, passes: int = 3) -> Bitmap:
    return separable_blur(bitmap, radius, radius, passes)


# ── ローカルの Numba カーネル ────────────────────────────────────
#
# **画素ごとに «分岐» や «順番» があるものだけ**をここに置きます。
# NumPy でベクトル化できるものを Numba に落とすと、かえって遅くなります。
# 共有の kernels.py ではなくこのファイルに置くのは、別担当との衝突を避けるため。

@njit(cache=True, fastmath=False)
def _k_pixel_sort(src, out, vertical, threshold, keep_line):
    """行（または列）ごとに、しきい値を超えた «連なり» を明るさ順に並べ替える。

    NumPy では書けません。«どこからどこまでが 1 本の連なりか» が画素の値で
    決まるので、走査しながら区間を切る必要があります。
    """
    height, width = src.shape[0], src.shape[1]
    lines = width if vertical else height
    span = height if vertical else width
    buf_l = np.empty(span, np.float32)
    buf_i = np.empty(span, np.int64)
    for line in range(lines):
        if keep_line[line] == 0:
            continue
        start = -1
        count = 0
        i = 0
        while i <= span:
            broke = True
            if i < span:
                y = i if vertical else line
                x = line if vertical else i
                lum = 0.299 * src[y, x, 0] + 0.587 * src[y, x, 1] + 0.114 * src[y, x, 2]
                if lum >= threshold and src[y, x, 3] > 8:
                    if start < 0:
                        start = i
                    buf_l[count] = lum
                    buf_i[count] = i
                    count += 1
                    broke = False
            if broke:
                if start >= 0 and i - start >= 2:
                    # 挿入ソート。連なりは短いので、これで十分速い。
                    for a in range(1, count):
                        key_l = buf_l[a]
                        key_i = buf_i[a]
                        b = a - 1
                        while b >= 0 and buf_l[b] > key_l:
                            buf_l[b + 1] = buf_l[b]
                            buf_i[b + 1] = buf_i[b]
                            b -= 1
                        buf_l[b + 1] = key_l
                        buf_i[b + 1] = key_i
                    for k in range(count):
                        sy = (start + k) if vertical else line
                        sx = line if vertical else (start + k)
                        oy = buf_i[k] if vertical else line
                        ox = line if vertical else buf_i[k]
                        for c in range(4):
                            out[sy, sx, c] = src[oy, ox, c]
                start = -1
                count = 0
            i += 1


@njit(cache=True, fastmath=False)
def _k_floyd_steinberg(data, out, levels, amount, palette, use_palette):
    """誤差拡散ディザ。**左上から順に誤差を配るので、並べ替えられません。**

    値はすべてスカラーの局所変数で持ちます。画素ごとに小さな配列を作ると
    92 万回の確保になり、それだけで 1280x720 が 440 ミリ秒でした（いまは 60 ms）。
    """
    height, width = data.shape[0], data.shape[1]
    error = np.zeros((height, width, 3), np.float32)
    step = 255.0 / (levels - 1)
    for y in range(height):
        for x in range(width):
            for c in range(3):
                v = data[y, x, c] + error[y, x, c]
                if v < 0.0:
                    v = 0.0
                elif v > 255.0:
                    v = 255.0
                error[y, x, c] = v  # 丸める前の値をここで持ち直す
            if use_palette:
                best = 0
                best_d = 1e30
                for pi in range(palette.shape[0]):
                    d0 = palette[pi, 0] - error[y, x, 0]
                    d1 = palette[pi, 1] - error[y, x, 1]
                    d2 = palette[pi, 2] - error[y, x, 2]
                    # 目の感度に合わせて緑を重く見る（単純な距離だと色味が飛ぶ）
                    d = 2.0 * d0 * d0 + 4.0 * d1 * d1 + 3.0 * d2 * d2
                    if d < best_d:
                        best_d = d
                        best = pi
            for c in range(3):
                source = data[y, x, c] * 1.0
                clamped = error[y, x, c] * 1.0
                if use_palette:
                    q = palette[best, c] * 1.0
                else:
                    q = math.floor(clamped / step + 0.5) * step
                out[y, x, c] = source + (q - source) * amount
                d = clamped - q
                if x + 1 < width:
                    error[y, x + 1, c] += d * (7.0 / 16.0)
                if y + 1 < height:
                    if x > 0:
                        error[y + 1, x - 1, c] += d * (3.0 / 16.0)
                    error[y + 1, x, c] += d * (5.0 / 16.0)
                    if x + 1 < width:
                        error[y + 1, x + 1, c] += d * (1.0 / 16.0)


@njit(cache=True, fastmath=False)
def _k_inner_distance(alpha, limit):
    """アルファの内側距離場（2 パスのチャンファー変換）。

    «前へ» と «後ろへ» の 2 回なめるだけなので O(画素) です。正確な
    ユークリッド距離ではありませんが、面取りの陰影には十分です。
    """
    height, width = alpha.shape[0], alpha.shape[1]
    field = np.zeros((height, width), np.float32)
    for y in range(height):
        for x in range(width):
            field[y, x] = limit if alpha[y, x] > 8 else 0.0
    for y in range(height):
        for x in range(width):
            if field[y, x] == 0.0:
                continue
            if x > 0 and field[y, x - 1] + 1.0 < field[y, x]:
                field[y, x] = field[y, x - 1] + 1.0
            if y > 0 and field[y - 1, x] + 1.0 < field[y, x]:
                field[y, x] = field[y - 1, x] + 1.0
            if x > 0 and y > 0 and field[y - 1, x - 1] + 1.414 < field[y, x]:
                field[y, x] = field[y - 1, x - 1] + 1.414
            if x < width - 1 and y > 0 and field[y - 1, x + 1] + 1.414 < field[y, x]:
                field[y, x] = field[y - 1, x + 1] + 1.414
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if field[y, x] == 0.0:
                continue
            if x < width - 1 and field[y, x + 1] + 1.0 < field[y, x]:
                field[y, x] = field[y, x + 1] + 1.0
            if y < height - 1 and field[y + 1, x] + 1.0 < field[y, x]:
                field[y, x] = field[y + 1, x] + 1.0
            if x < width - 1 and y < height - 1 and field[y + 1, x + 1] + 1.414 < field[y, x]:
                field[y, x] = field[y + 1, x + 1] + 1.414
            if x > 0 and y < height - 1 and field[y + 1, x - 1] + 1.414 < field[y, x]:
                field[y, x] = field[y + 1, x - 1] + 1.414
    return field


def draw_textured_triangle(dst: Bitmap, src: Bitmap, a: dict, b: dict, c: dict,
                           options: dict | None = None) -> None:
    """テクスチャ付き三角形を 1 枚描く（`renderer/raster.py` の薄い包み）。

    **`u` / `v` は «テクセル» 座標（0..width, 0..height）です。** 正規化した
    0..1 を渡すと板が 1 画素に潰れます。JS 版で一度やった罠なので、
    3D の板・メッシュ・かけらの入口にあたるここに大きく書いておきます。

    :param a: `{"x", "y", "u", "v"}`
    :param options: `alpha` `blend` `clampEdge` `tint`（`Color`）`depth`
    """
    options = options or {}
    alpha = options.get("alpha", 1.0)
    if alpha is None:
        alpha = 1.0
    if alpha <= 0:
        return
    tint = options.get("tint")
    _raster_triangle(
        dst,
        src,
        (a["x"], a["y"], a.get("u", 0.0), a.get("v", 0.0)),
        (b["x"], b["y"], b.get("u", 0.0), b.get("v", 0.0)),
        (c["x"], c["y"], c.get("u", 0.0), c.get("v", 0.0)),
        alpha=float(alpha),
        blend=options.get("blend", "normal") or "normal",
        clamp_edge=bool(options.get("clampEdge", False)),
        tint=None if tint is None else (tint.r, tint.g, tint.b, tint.a),
        depth=options.get("depth"),
    )


# `rasterize_contours` / `fill_coverage` / `circle_contour` は raster.py から
# そのまま持ってきています（import 済み）。走査線を 2 つ持たないためです。


# ── ここからエフェクト本体 ────────────────────────────────────────
# すべて `(bitmap, params, ctx) -> Bitmap` で、元のビットマップは書き換えません。

def _rgb_f(bitmap: Bitmap) -> np.ndarray:
    """RGB を float64 で取り出す（`mapPixels` の入口に当たるもの）。"""
    return bitmap.data[..., :3].astype(np.float64)


def opacity(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """不透明度。`amount` は 0..1 の «掛け算» です。"""
    amount = clamp(_num(params, "amount", "value", default=1), 0, 1)
    out = bitmap.copy()
    out.data[..., 3] = _u8(bitmap.data[..., 3].astype(np.float64) * amount)
    return out


def blur(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ぼかし。`quality: "low"` で 1 パス（速いぶん少し四角い）。"""
    radius = max(0, _num(params, "radius", "amount", default=4))
    return separable_blur(bitmap, radius, radius, 1 if params.get("quality") == "low" else 3)


def directional_blur(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """方向性ブラー。角度方向に `radius` 画素ぶん引き伸ばします。"""
    radius = max(0, _num(params, "radius", "amount", default=8))
    angle = math.radians(_num(params, "angle", default=0))
    steps = max(1, int(_jround(radius)))
    dx = math.cos(angle)
    dy = math.sin(angle)
    h, w = bitmap.height, bitmap.width
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    acc = np.zeros((h, w, 3), np.float64)
    acc_a = np.zeros((h, w), np.float64)
    count = 0
    for k in range(-steps, steps + 1):
        s = sample_bilinear(bitmap.data, xs + dx * k + 0.5, ys + dy * k + 0.5, True)
        a = s[..., 3].astype(np.float64)
        acc += s[..., :3].astype(np.float64) * a[..., None]
        acc_a += a
        count += 1
    out = Bitmap(w, h)
    ok = acc_a > 0
    safe = np.where(ok, acc_a, 1.0)
    out.data[..., :3] = np.where(ok[..., None], _u8(acc / safe[..., None]), 0)
    out.data[..., 3] = np.where(ok, _u8(acc_a / count), 0)
    return out


def sharpen(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """アンシャープマスク。ぼかしとの差を足し戻します。"""
    amount = _num(params, "amount", default=1)
    blurred = separable_blur(bitmap, 1, 1, 1)
    out = bitmap.copy()
    base = _rgb_f(bitmap)
    out.data[..., :3] = _u8(base + (base - blurred.data[..., :3].astype(np.float64)) * amount)
    return out


def color_adjust(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """明度・コントラスト・彩度・色相・ガンマの一括。

    **値は «増減量» で 0 が «変化なし» です。倍率ではありません。**
    `brightness: 1.0` は «全部に +255» という意味で、真っ白になります。
    ここを倍率に変えると既存のプロジェクトが全部白飛びするので、変えません。
    （`gamma` だけは 1 が «変化なし»。色屋の共通語に合わせています）
    """
    brightness = _num(params, "brightness", default=0)
    contrast = _num(params, "contrast", default=0)
    saturation = _num(params, "saturation", default=0)
    hue_shift = (_num(params, "hue", default=0) % 360) / 360
    gamma = _num(params, "gamma", default=1)
    contrast_factor = 1 + clamp(contrast, -1, 4)

    v = _rgb_f(bitmap) / 255.0
    v = v + brightness
    v = (v - 0.5) * contrast_factor + 0.5
    if gamma != 1:
        v = np.power(np.maximum(0.0, v), 1.0 / gamma)
    rgb = np.clip(v, 0, 1) * 255.0

    if saturation != 0 or hue_shift != 0:
        hsl = rgb_to_hsl(rgb)
        hsl[..., 0] = (hsl[..., 0] + hue_shift + 1) % 1
        hsl[..., 1] = np.clip(hsl[..., 1] * (1 + saturation), 0, 1)
        rgb = hsl_to_rgb(hsl)

    out = bitmap.copy()
    out.data[..., :3] = _u8(rgb)
    return out


def tint(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """指定色へ寄せる（明暗は残りません。残したいときは `monochrome`）。"""
    color = parse_color(params.get("color", "#ffffff"))
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    out = bitmap.copy()
    base = _rgb_f(bitmap)
    target = np.array(color.rgb(), np.float64)
    out.data[..., :3] = _u8(base * (1 - amount) + target * amount)
    return out


def grayscale(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    base = _rgb_f(bitmap)
    lum = luma_of(base).astype(np.float64)
    out = bitmap.copy()
    out.data[..., :3] = _u8(base * (1 - amount) + lum[..., None] * amount)
    return out


def invert(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    base = _rgb_f(bitmap)
    out = bitmap.copy()
    out.data[..., :3] = _u8(base * (1 - amount) + (255.0 - base) * amount)
    return out


def threshold(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    level = _num(params, "level", default=0.5) * 255
    lum = luma_of(_rgb_f(bitmap)).astype(np.float64)
    out = bitmap.copy()
    out.data[..., :3] = _u8(np.where(lum >= level, 255.0, 0.0)[..., None])
    return out


def pixelate(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """モザイク。**ブロックの平均はアルファで重み付けします。**

    素直に色だけ平均すると、透明な画素の «黒» が混ざってブロックが暗くなります。
    """
    size = max(1, int(_jround(_num(params, "size", "amount", default=8))))
    h, w = bitmap.height, bitmap.width
    f = bitmap.data.astype(np.float64)
    a = f[..., 3]
    prem = f[..., :3] * a[..., None]

    # ブロックごとの和。端が半端なときは reduceat で «そこまで» を足します。
    ys = np.arange(0, h, size)
    xs = np.arange(0, w, size)
    sum_c = np.add.reduceat(np.add.reduceat(prem, ys, axis=0), xs, axis=1)
    sum_a = np.add.reduceat(np.add.reduceat(a, ys, axis=0), xs, axis=1)
    counts = np.outer(np.minimum(size, h - ys), np.minimum(size, w - xs)).astype(np.float64)

    ok = sum_a > 0
    safe = np.where(ok, sum_a, 1.0)
    block_rgb = np.where(ok[..., None], sum_c / safe[..., None], 0.0)
    block_a = np.where(counts > 0, sum_a / counts, 0.0)

    rows = np.repeat(np.arange(len(ys)), np.minimum(size, h - ys))
    cols = np.repeat(np.arange(len(xs)), np.minimum(size, w - xs))
    out = Bitmap(w, h)
    out.data[..., :3] = _u8(block_rgb[np.ix_(rows, cols)])
    out.data[..., 3] = _u8(block_a[np.ix_(rows, cols)])
    return out


def _lay_on_empty(dst: Bitmap, src: Bitmap, alpha: float) -> None:
    """**空の下地**に `src` を重ねる（`glow` と `dropShadow` の 1 枚目）。

    下地が透明なら合成の式は `out = 色そのまま、アルファ = sa` に縮みます。
    一般の合成を通すと画面 1 枚ぶんの float64 が 6 枚できるので、そこだけ
    近道します（1280x720 で 130 ミリ秒 → 12 ミリ秒）。**絵は変わりません。**

    掛ける順は JS の `(a / 255) * alpha` に合わせてあります（`a * (alpha / 255)`
    と書くと最後の 1 ビットが変わり、丸めの向きが変わる画素が出ます）。
    """
    sa = src.data[..., 3].astype(np.float64) / 255.0 * alpha
    touched = sa > 0
    dst.data[..., :3] = np.where(touched[..., None], src.data[..., :3], 0)
    dst.data[..., 3] = np.where(touched, _u8(sa * 255.0), 0)


def glow(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """発光。ぼかした像を下に敷いてから元の絵を重ねます。"""
    radius = max(1, _num(params, "radius", default=12))
    intensity = _num(params, "intensity", default=1)
    color = parse_color(params["color"]) if params.get("color") else None
    source = bitmap.copy()
    if color:
        source.data[..., 0] = color.r
        source.data[..., 1] = color.g
        source.data[..., 2] = color.b
    blurred = separable_blur(source, radius, radius, 3)
    out = Bitmap(bitmap.width, bitmap.height)
    _lay_on_empty(out, blurred, clamp(intensity, 0, 4))
    composite(out, bitmap, 0, 0, 1, "normal")
    return out


def drop_shadow(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """影。アルファだけを取り出してずらし、ぼかして敷きます。"""
    offset_x = _num(params, "offsetX", default=6)
    offset_y = _num(params, "offsetY", default=6)
    radius = max(0, _num(params, "radius", "blur", default=8))
    color = parse_color(params.get("color", "rgba(0,0,0,0.5)"))
    silhouette = Bitmap(bitmap.width, bitmap.height)
    silhouette.data[..., 0] = color.r
    silhouette.data[..., 1] = color.g
    silhouette.data[..., 2] = color.b
    silhouette.data[..., 3] = _u8(bitmap.data[..., 3].astype(np.float64) * color.a)
    blurred = separable_blur(silhouette, radius, radius, 3) if radius > 0 else silhouette
    out = Bitmap(bitmap.width, bitmap.height)
    # ずらして置くので «空の下地» の近道は使えません（はみ出しの刈り込みが要る）
    composite(out, blurred, offset_x, offset_y, 1)
    composite(out, bitmap, 0, 0, 1)
    return out


def stroke(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """縁取り。**透明な画素のうち、半径内に不透明が居るものを塗ります。**

    半径 r の «円板ぶんの最大値» なので、ずらして OR を取る形にしています。
    r は数画素なので、ずらす回数は数十回で済みます。
    """
    width = max(1, int(_jround(_num(params, "width", default=3))))
    color = parse_color(params.get("color", "#000000"))
    alpha = bitmap.data[..., 3]
    solid = alpha > 128
    near = np.zeros(solid.shape, bool)
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dx * dx + dy * dy > width * width:
                continue
            shifted = np.zeros_like(solid)
            ys0, ys1 = max(0, -dy), min(bitmap.height, bitmap.height - dy)
            xs0, xs1 = max(0, -dx), min(bitmap.width, bitmap.width - dx)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            shifted[ys0:ys1, xs0:xs1] = solid[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx]
            near |= shifted
    near &= alpha <= 8

    outline = Bitmap(bitmap.width, bitmap.height)
    outline.data[..., 0] = np.where(near, color.r, 0)
    outline.data[..., 1] = np.where(near, color.g, 0)
    outline.data[..., 2] = np.where(near, color.b, 0)
    outline.data[..., 3] = np.where(near, _u8(255 * color.a), 0)
    out = Bitmap(bitmap.width, bitmap.height)
    composite(out, outline, 0, 0, 1)
    composite(out, bitmap, 0, 0, 1)
    return out


def chroma_key(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """クロマキー。指定色に近い画素を抜きます（既定は緑）。"""
    key = parse_color(params.get("color", "#00ff00"))
    tolerance = _num(params, "tolerance", default=0.25) * 441.67
    softness = _num(params, "softness", default=0.1) * 441.67
    base = _rgb_f(bitmap)
    distance = np.sqrt(
        (base[..., 0] - key.r) ** 2 + (base[..., 1] - key.g) ** 2 + (base[..., 2] - key.b) ** 2
    )
    a = bitmap.data[..., 3].astype(np.float64)
    scaled = np.where(
        distance < tolerance,
        0.0,
        np.where(distance < tolerance + softness, a * (distance - tolerance) / (softness if softness else 1.0), a),
    )
    out = bitmap.copy()
    out.data[..., 3] = _u8(scaled)
    return out


@njit(cache=True, fastmath=False)
def _k_vignette(src, out, amount, radius, softness):
    h, w = src.shape[0], src.shape[1]
    cx = w / 2.0
    cy = h / 2.0
    max_distance = math.hypot(cx, cy)
    for y in range(h):
        dy = y - cy
        for x in range(w):
            d = math.hypot(x - cx, dy) / max_distance
            t = (d - radius) / softness
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            shade = 1.0 - amount * t
            out[y, x, 0] = _u8_scalar(src[y, x, 0] * shade)
            out[y, x, 1] = _u8_scalar(src[y, x, 1] * shade)
            out[y, x, 2] = _u8_scalar(src[y, x, 2] * shade)
            out[y, x, 3] = src[y, x, 3]


def vignette(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """周辺減光。中心から `radius` を超えたところから暗くします。"""
    amount = clamp(_num(params, "amount", default=0.5), 0, 1)
    radius = _num(params, "radius", default=0.75)
    softness = max(1e-3, _num(params, "softness", default=0.4))
    out = np.empty_like(bitmap.data)
    _k_vignette(np.ascontiguousarray(bitmap.data), out, float(amount), float(radius), float(softness))
    return Bitmap(bitmap.width, bitmap.height, out)


@njit(cache=True, fastmath=False)
def _k_noise(src, out, amount, mono, seed):
    """粒子ノイズを 1 パスで乗せる。

    NumPy だと «格子ハッシュ 4 回 × 画面 1 枚» の中間配列が積み上がって
    960x540 で 360 ミリ秒でした。1 画素ずつ回すほうが速く（8 ミリ秒）、しかも
    **乱数を «回さない»** ので、どのフレームから描いても同じ絵になります。
    """
    h, w = src.shape[0], src.shape[1]
    for y in range(h):
        fy = y * 0.7
        for x in range(w):
            fx = x * 0.7
            if mono:
                n = _nb_value_noise_2d(fx + seed * 0.013, fy - seed * 0.017, seed) * amount
                out[y, x, 0] = _u8_scalar(src[y, x, 0] + n)
                out[y, x, 1] = _u8_scalar(src[y, x, 1] + n)
                out[y, x, 2] = _u8_scalar(src[y, x, 2] + n)
            else:
                out[y, x, 0] = _u8_scalar(src[y, x, 0] + _nb_value_noise_2d(fx, fy, seed) * amount)
                out[y, x, 1] = _u8_scalar(src[y, x, 1] + _nb_value_noise_2d(fx, fy, seed + 17) * amount)
                out[y, x, 2] = _u8_scalar(src[y, x, 2] + _nb_value_noise_2d(fx, fy, seed + 33) * amount)
            out[y, x, 3] = src[y, x, 3]


def noise(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """粒子ノイズ。**時刻とシードから決まるので、巻き戻しても同じ絵になります。**"""
    ctx = ctx or {}
    amount = _num(params, "amount", default=0.1) * 255
    seed = int(ctx.get("seed", 0)) + int(_jround((ctx.get("time", 0) or 0) * 1000))
    out = np.empty_like(bitmap.data)
    _k_noise(np.ascontiguousarray(bitmap.data), out, float(amount),
             params.get("monochrome") is not False, seed)
    return Bitmap(bitmap.width, bitmap.height, out)


@njit(cache=True, fastmath=False)
def _k_extract_bright(src, out, level):
    """明部だけを抜き出す（`bloom` と `lightStreak` の下ごしらえ）。"""
    h, w = src.shape[0], src.shape[1]
    denominator = 255.0 - level
    if denominator < 1.0:
        denominator = 1.0
    for y in range(h):
        for x in range(w):
            lum = 0.299 * src[y, x, 0] + 0.587 * src[y, x, 1] + 0.114 * src[y, x, 2]
            if lum < level:
                out[y, x, 0] = 0
                out[y, x, 1] = 0
                out[y, x, 2] = 0
                out[y, x, 3] = 0
                continue
            weight = (lum - level) / denominator
            if weight > 1.0:
                weight = 1.0
            out[y, x, 0] = src[y, x, 0]
            out[y, x, 1] = src[y, x, 1]
            out[y, x, 2] = src[y, x, 2]
            out[y, x, 3] = _u8_scalar(src[y, x, 3] * weight)


@njit(cache=True, fastmath=False)
def _k_bloom_merge(src, blurred, out, intensity):
    """ぼかした明部を元の絵に足す。**アルファは «濃いほう» を残します。**"""
    h, w = src.shape[0], src.shape[1]
    for y in range(h):
        for x in range(w):
            a = blurred[y, x, 3] / 255.0 * intensity
            if a <= 0.0:
                out[y, x, 0] = src[y, x, 0]
                out[y, x, 1] = src[y, x, 1]
                out[y, x, 2] = src[y, x, 2]
                out[y, x, 3] = src[y, x, 3]
                continue
            for c in range(3):
                v = src[y, x, c] + blurred[y, x, c] * a
                if v > 255.0:
                    v = 255.0
                out[y, x, c] = _u8_scalar(v)
            lifted = blurred[y, x, 3] * intensity
            base = src[y, x, 3] * 1.0
            out[y, x, 3] = _u8_scalar(lifted if lifted > base else base)


def bloom(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ブルーム。明部だけを抜き出してぼかし、元の絵に足します。

    **半径違いを重ねる JS 版の作り（ボックスぼかし 3 回）をそのまま残しています。**
    1 発の大きなぼかしにまとめると «芯の締まり» が消えて、光がただの靄になります。

    `glow` と違ってアルファも持ち上げるので、透明な背景の上でも光が広がります。
    """
    level = _num(params, "threshold", default=0.7) * 255
    radius = max(1, _num(params, "radius", default=16))
    intensity = _num(params, "intensity", default=0.8)
    source = np.ascontiguousarray(bitmap.data)
    bright = Bitmap(bitmap.width, bitmap.height)
    _k_extract_bright(source, bright.data, float(level))
    blurred = separable_blur(bright, radius, radius, 3)
    out = np.empty_like(bitmap.data)
    _k_bloom_merge(source, blurred.data, out, float(intensity))
    return Bitmap(bitmap.width, bitmap.height, out)


def duotone(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """2 色で明暗を塗り分ける。"""
    shadow = parse_color(_num(params, "shadow", "colorA", default="#1b1d2b"))
    highlight = parse_color(_num(params, "highlight", "colorB", default="#ffd166"))
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    base = _rgb_f(bitmap)
    lum = (luma_of(base).astype(np.float64) / 255.0)[..., None]
    lo = np.array(shadow.rgb(), np.float64)
    hi = np.array(highlight.rgb(), np.float64)
    out = bitmap.copy()
    out.data[..., :3] = _u8(base * (1 - amount) + (lo + (hi - lo) * lum) * amount)
    return out


def posterize(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """階調を落とす。**丸めは `Math.round`（0.5 切り上げ）です。**"""
    levels = max(2, int(_jround(_num(params, "levels", default=6))))
    step = 255 / (levels - 1)
    out = bitmap.copy()
    out.data[..., :3] = _u8(_jround(_rgb_f(bitmap) / step) * step)
    return out


def _convolve3(bitmap: Bitmap, kernel, divisor=1.0, bias=0.0, amount=1.0) -> Bitmap:
    """3x3 の畳み込み。**ずらして足す**ので、9 回の加算で済みます。

    `sliding_window_view` を使う手もありますが、3x3 ではずらし加算のほうが
    一時配列が小さく、実測でも速いです。
    """
    base = _rgb_f(bitmap)
    h, w = bitmap.height, bitmap.width
    acc = np.zeros_like(base)
    for ky in (-1, 0, 1):
        for kx in (-1, 0, 1):
            weight = kernel[(ky + 1) * 3 + (kx + 1)]
            if weight == 0:
                continue
            # 端は «伸ばして» 読む（JS の clamp と同じ）
            rows = np.clip(np.arange(h) + ky, 0, h - 1)
            cols = np.clip(np.arange(w) + kx, 0, w - 1)
            acc += base[np.ix_(rows, cols)] * weight
    value = np.clip(acc / (divisor or 1) + bias, 0, 255)
    out = bitmap.copy()
    out.data[..., :3] = _u8(base * (1 - amount) + value * amount)
    return out


def emboss(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    return _convolve3(bitmap, [-2, -1, 0, -1, 1, 1, 0, 1, 2], 1, 0, _num(params, "amount", default=1))


def edge_detect(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    return _convolve3(bitmap, [0, -1, 0, -1, 4, -1, 0, -1, 0], 1, 0, _num(params, "amount", default=1))


def mirror(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """片側をもう片側へ折り返す。"""
    axis = params.get("axis", "x")
    flip = params.get("flip") is True
    out = bitmap.copy()
    w, h = bitmap.width, bitmap.height
    if axis == "x":
        half = w / 2
        xs = np.arange(w)
        take = (xs < half) if flip else (xs >= half)
        src_x = np.where(take, w - 1 - xs, xs)
        out.data[:] = bitmap.data[:, src_x]
    else:
        half = h / 2
        ys = np.arange(h)
        take = (ys < half) if flip else (ys >= half)
        src_y = np.where(take, h - 1 - ys, ys)
        out.data[:] = bitmap.data[src_y, :]
    return out


def kaleidoscope(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """万華鏡。1 枚のくさびを回して敷き詰めます。"""
    segments = max(2, int(_jround(_num(params, "segments", default=6))))
    rotation = math.radians(_num(params, "rotation", default=0))
    cx = bitmap.width / 2
    cy = bitmap.height / 2
    wedge = math.tau / segments
    ys, xs = np.mgrid[0 : bitmap.height, 0 : bitmap.width].astype(np.float64)
    dx = xs - cx
    dy = ys - cy
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx) - rotation
    angle = ((angle % wedge) + wedge) % wedge
    angle = np.where(angle > wedge / 2, wedge - angle, angle) + rotation
    s = sample_bilinear(bitmap.data, cx + np.cos(angle) * radius, cy + np.sin(angle) * radius, True)
    out = Bitmap(bitmap.width, bitmap.height)
    out.data[:] = _u8(s)
    return out


def scanlines(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """走査線。`spacing` 行おきに暗くします。"""
    spacing = max(2, int(_jround(_num(params, "spacing", default=4))))
    amount = clamp(_num(params, "amount", default=0.35), 0, 1)
    out = bitmap.copy()
    rows = np.arange(bitmap.height) % spacing == 0
    factor = np.where(rows, 1 - amount, 1.0)[:, None, None]
    out.data[..., :3] = _u8(_rgb_f(bitmap) * factor)
    return out


def chromatic_aberration(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """色収差。赤と青を中心から放射方向にずらします。"""
    amount = _num(params, "amount", default=3)
    cx = bitmap.width / 2
    cy = bitmap.height / 2
    ys, xs = np.mgrid[0 : bitmap.height, 0 : bitmap.width].astype(np.float64)
    nx = (xs - cx) / max(1, cx)
    ny = (ys - cy) / max(1, cy)
    red = sample_bilinear(bitmap.data, xs + 0.5 + nx * amount, ys + 0.5 + ny * amount, True)
    blue = sample_bilinear(bitmap.data, xs + 0.5 - nx * amount, ys + 0.5 - ny * amount, True)
    base = sample_bilinear(bitmap.data, xs + 0.5, ys + 0.5, True)
    out = Bitmap(bitmap.width, bitmap.height)
    out.data[..., 0] = _u8(red[..., 0])
    out.data[..., 1] = _u8(base[..., 1])
    out.data[..., 2] = _u8(blue[..., 2])
    out.data[..., 3] = _u8(np.maximum(base[..., 3], np.maximum(red[..., 3], blue[..., 3])))
    return out


def lens_distortion(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """樽型／糸巻き型の歪み。`strength` が正で樽型です。"""
    strength = _num(params, "strength", default=0.2)
    cx = bitmap.width / 2
    cy = bitmap.height / 2
    max_radius = math.hypot(cx, cy) or 1
    ys, xs = np.mgrid[0 : bitmap.height, 0 : bitmap.width].astype(np.float64)
    dx = (xs - cx) / max_radius
    dy = (ys - cy) / max_radius
    factor = 1 + strength * (dx * dx + dy * dy)
    s = sample_bilinear(
        bitmap.data, cx + dx * max_radius * factor + 0.5, cy + dy * max_radius * factor + 0.5, False
    )
    out = Bitmap(bitmap.width, bitmap.height)
    out.data[:] = _u8(s)
    return out


def round_corners(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """角を丸める（アルファを削るだけ）。"""
    radius = max(0, _num(params, "radius", default=16))
    if radius <= 0:
        return bitmap.copy()
    w, h = bitmap.width, bitmap.height
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = np.maximum(np.maximum(radius - xs, xs - (w - radius)), 0)
    dy = np.maximum(np.maximum(radius - ys, ys - (h - radius)), 0)
    distance = np.hypot(dx, dy)
    cut = (dx != 0) & (dy != 0) & (distance > radius)
    out = bitmap.copy()
    faded = bitmap.data[..., 3].astype(np.float64) * np.clip(radius + 1 - distance, 0, 1)
    out.data[..., 3] = np.where(cut, _u8(faded), bitmap.data[..., 3])
    return out


def feather(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """縁をぼかす（アルファを外へ向かって落とします）。"""
    size = max(1, int(_jround(_num(params, "size", default=12))))
    w, h = bitmap.width, bitmap.height
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    distance = np.minimum(np.minimum(xs, ys), np.minimum(w - 1 - xs, h - 1 - ys))
    out = bitmap.copy()
    faded = bitmap.data[..., 3].astype(np.float64) * (distance / size)
    out.data[..., 3] = np.where(distance < size, _u8(faded), bitmap.data[..., 3])
    return out


def gradient_map(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """輝度をグラデーションに写す。**色は `colors` ではなく `stops` で書きます。**

    `stops` は `[{"offset": 0..1, "color": "#..."}]` の並びです。`colors` という
    名前の配列を渡しても無視されます（JS 版と同じ）。
    """
    stops = params.get("stops") or [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}]
    xs, ys = _stop_nodes(stops)
    lum = luma_of(_rgb_f(bitmap)).astype(np.float64) / 255.0
    out = bitmap.copy()
    # np.interp は端をそのまま伸ばすので、JS の «最初／最後の色をそのまま返す» と同じ
    out.data[..., :3] = _u8(np.stack([np.interp(lum, xs, ys[:, c]) for c in range(3)], axis=-1))
    return out


# ── MV の定番演出 ────────────────────────────────────────────────
# 制作ブログ 15 サイトの調査から起こしたものです（docs/mv-effects-research.ja.md）。
# 既定値は «そのまま使っても成立する控えめな値» にしてあります。


def _extract_bright(bitmap: Bitmap, level01: float) -> Bitmap:
    """明部だけを抜き出す。発光系の共通処理です。"""
    out = Bitmap(bitmap.width, bitmap.height)
    _k_extract_bright(np.ascontiguousarray(bitmap.data), out.data, float(level01 * 255))
    return out


def _screen_add(base: Bitmap, light: Bitmap, intensity: float) -> Bitmap:
    """スクリーン合成で光を足す。**アルファは «濃いほう» を残します。**"""
    out = base.copy()
    la = light.data[..., 3].astype(np.float64)
    a = la / 255.0 * intensity
    touched = a > 0
    cb = _rgb_f(base)
    cs = light.data[..., :3].astype(np.float64)
    mixed = 255.0 - (255.0 - cb) * (255.0 - cs * a[..., None]) / 255.0
    out.data[..., :3] = np.where(touched[..., None], _u8(mixed), base.data[..., :3])
    out.data[..., 3] = np.where(
        touched, _u8(np.maximum(base.data[..., 3].astype(np.float64), la * intensity)), base.data[..., 3]
    )
    return out


def radial_blur(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """放射ブラー（ズームブラー）。拡大や Z 移動と重ねるとスピード感が出ます。"""
    amount = _num(params, "amount", default=20)
    if amount <= 0:
        return bitmap.copy()
    samples = int(clamp(_jround(_num(params, "samples", default=10)), 2, 48))
    cx = _num(params, "centerX", default=0.5) * bitmap.width
    cy = _num(params, "centerY", default=0.5) * bitmap.height
    out = np.empty_like(bitmap.data)
    _k_zoom_blur(np.ascontiguousarray(bitmap.data), out, float(cx), float(cy),
                 float(amount / 1000), samples)
    return Bitmap(bitmap.width, bitmap.height, out)


def spin_blur(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """回転ブラー。Z 軸回転と重ねて使います。"""
    angle = math.radians(_num(params, "angle", default=8))
    if angle == 0:
        return bitmap.copy()
    samples = int(clamp(_jround(_num(params, "samples", default=10)), 2, 48))
    cx = _num(params, "centerX", default=0.5) * bitmap.width
    cy = _num(params, "centerY", default=0.5) * bitmap.height
    out = np.empty_like(bitmap.data)
    _k_spin_blur(np.ascontiguousarray(bitmap.data), out, float(cx), float(cy), float(angle), samples)
    return Bitmap(bitmap.width, bitmap.height, out)


def glitch(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """グリッチ。帯のずれ・色ずれ・走査線の欠落をまとめたものです。

    `interval` は «毎フレームだと速すぎる» ときの間引きです（既定 2 フレーム）。
    乱数の消費順は JS 版と 1 回ずつ合わせてあるので、同じ絵が出ます。
    """
    ctx = ctx or {}
    amount = clamp(_num(params, "amount", default=0.4), 0, 1)
    if amount <= 0:
        return bitmap.copy()
    blocks = int(clamp(_jround(_num(params, "blocks", default=10)), 1, 200))
    color_shift = _num(params, "colorShift", default=6)
    interval = max(1, int(_jround(_num(params, "interval", default=2))))
    fps = ctx.get("fps", 30)
    frame = math.floor((ctx.get("time", 0) or 0) * fps / interval)
    random = Random(int(_num(params, "seed", default=1)) + frame * 7919)
    w, h = bitmap.width, bitmap.height
    out = bitmap.copy()

    # 1. 帯ごとに横へずらす
    for _ in range(blocks):
        band_height = max(1, int(_jround((h / blocks) * (0.4 + random() * 1.2))))
        y0 = math.floor(random() * h)
        shift = int(_jround((random() - 0.5) * 2 * amount * w * 0.12))
        if shift == 0:
            continue
        y1 = min(h, y0 + band_height)
        if y1 <= y0:
            continue
        idx = np.clip(np.arange(w) - shift, 0, w - 1)
        out.data[y0:y1] = out.data[y0:y1][:, idx]

    # 2. 色ずれ（赤と青を左右に分ける）
    if color_shift != 0:
        shifted = out.data.copy()
        offset = int(_jround(color_shift * amount))
        rx = np.clip(np.arange(w) - offset, 0, w - 1)
        bx = np.clip(np.arange(w) + offset, 0, w - 1)
        out.data[..., 0] = shifted[:, rx, 0]
        out.data[..., 2] = shifted[:, bx, 2]

    # 3. 走査線の欠落。乱数は «1 行に 1 回» 引くので、順番を崩せません。
    drop_rate = amount * 0.15
    drop = np.fromiter((random() <= drop_rate for _ in range(h)), bool, h)
    if drop.any():
        faded = out.data[..., 3].astype(np.float64) * 0.25
        out.data[..., 3] = np.where(drop[:, None], _u8(faded), out.data[..., 3])
    return out


def raster_scroll(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ラスター（ぐにゃ〜）。走査線ごとに正弦波でずらします。"""
    ctx = ctx or {}
    amplitude = _num(params, "amplitude", default=8)
    if amplitude == 0:
        return bitmap.copy()
    frequency = _num(params, "frequency", default=4)
    speed = _num(params, "speed", default=1)
    randomness = clamp(_num(params, "random", default=0), 0, 1)
    vertical = params.get("axis") == "vertical"
    time = ctx.get("time", 0) or 0
    seed = int(_num(params, "seed", default=11))
    w, h = bitmap.width, bitmap.height
    lines = w if vertical else h

    line_index = np.arange(lines, dtype=np.float64)
    phase = (line_index / max(1, lines - 1)) * frequency * math.tau + time * speed * math.tau
    offset = np.sin(phase) * amplitude
    if randomness > 0:
        noisy = value_noise_1d(line_index * 0.35 + time * speed * 3, seed) * amplitude
        offset = offset + (noisy - offset) * randomness

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    if vertical:
        s = sample_bilinear(bitmap.data, xs + 0.5, ys + 0.5 + offset[None, :], True)
    else:
        s = sample_bilinear(bitmap.data, xs + 0.5 + offset[:, None], ys + 0.5, True)
    out = Bitmap(w, h)
    out.data[:] = _u8(s)
    return out


def diffusion(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """拡散光。元の色を保ったまま明部を広げます（glow / bloom との違いはそこ）。"""
    strength = clamp(_num(params, "strength", default=50) / 100, 0, 1)
    size = max(
        1, int(_jround((_num(params, "diffusion", default=12) / 100) * min(bitmap.width, bitmap.height) * 0.25))
    )
    if strength <= 0:
        return bitmap.copy()
    return _screen_add(bitmap, box_blur(bitmap, size, 2), strength)


def light_streak(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """閃光。明部から一方向へ光の筋を伸ばします。"""
    length = max(1, int(_jround(_num(params, "length", default=80))))
    angle = math.radians(_num(params, "angle", default=0))
    level = clamp(_num(params, "threshold", default=0.65), 0, 1)
    intensity = _num(params, "intensity", default=0.8)
    bright = _extract_bright(bitmap, level)
    tint_color = parse_color(params["color"]) if params.get("color") else None
    dx = math.cos(angle)
    dy = math.sin(angle)
    both = params.get("both") is not False

    h, w = bitmap.height, bitmap.width
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    acc = np.zeros((h, w, 3), np.float64)
    acc_a = np.zeros((h, w), np.float64)
    for s in range(1, length + 1):
        falloff = 1 - s / (length + 1)
        sample = sample_bilinear(bright.data, xs + 0.5 - dx * s, ys + 0.5 - dy * s, False)
        a = sample[..., 3].astype(np.float64) * falloff
        acc += sample[..., :3].astype(np.float64) * a[..., None]
        acc_a += a
        if both:
            sample = sample_bilinear(bright.data, xs + 0.5 + dx * s, ys + 0.5 + dy * s, False)
            a = sample[..., 3].astype(np.float64) * falloff
            acc += sample[..., :3].astype(np.float64) * a[..., None]
            acc_a += a

    streak = Bitmap(w, h)
    ok = acc_a > 0
    safe = np.where(ok, acc_a, 1.0)
    if tint_color:
        rgb = np.broadcast_to(np.array(tint_color.rgb(), np.float64), (h, w, 3))
    else:
        rgb = acc / safe[..., None]
    streak.data[..., :3] = np.where(ok[..., None], _u8(rgb), 0)
    alpha = np.minimum(255.0, acc_a / (length * (2 if both else 1)) * 2.5)
    streak.data[..., 3] = np.where(ok, _u8(alpha), 0)
    return _screen_add(bitmap, streak, intensity)


def lens_flare(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """レンズフレア。中心の光球・放射する筋・同心リングを重ねます。"""
    cx = _num(params, "x", default=0.5) * bitmap.width
    cy = _num(params, "y", default=0.5) * bitmap.height
    size = _num(params, "size", default=0.4) * min(bitmap.width, bitmap.height)
    intensity = clamp(_num(params, "intensity", default=0.8), 0, 4)
    color = parse_color(params.get("color", "#ffe9b0"))
    rings = int(clamp(_jround(_num(params, "rings", default=3)), 0, 8))
    streaks = int(clamp(_jround(_num(params, "streaks", default=6)), 0, 24))
    rotation = math.radians(_num(params, "rotation", default=0))

    ys, xs = np.mgrid[0 : bitmap.height, 0 : bitmap.width].astype(np.float64)
    dx = xs + 0.5 - cx
    dy = ys + 0.5 - cy
    distance = np.hypot(dx, dy)
    value = np.power(np.maximum(0.0, 1 - distance / size), 2.4)
    if streaks > 0:
        theta = np.arctan2(dy, dx) - rotation
        spokes = np.power(np.abs(np.cos(theta * (streaks / 2))), 12)
        value = value + np.where(
            distance > 1, spokes * np.power(np.maximum(0.0, 1 - distance / (size * 2.2)), 1.6) * 0.8, 0.0
        )
    for i in range(1, rings + 1):
        radius = size * (0.55 + i * 0.38)
        band = np.exp(-np.power((distance - radius) / (size * 0.06), 2))
        value = value + band * (0.22 / i)

    flare = Bitmap(bitmap.width, bitmap.height)
    keep = value > 0.002
    flare.data[..., 0] = np.where(keep, color.r, 0)
    flare.data[..., 1] = np.where(keep, color.g, 0)
    flare.data[..., 2] = np.where(keep, color.b, 0)
    flare.data[..., 3] = np.where(keep, _u8(np.minimum(255.0, value * 255)), 0)
    return _screen_add(bitmap, flare, intensity)


def rim_light(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """リムライト（逆光）。指定方向側の輪郭だけを光らせます。"""
    angle = math.radians(_num(params, "angle", default=-45))
    width = max(1, int(_jround(_num(params, "width", default=5))))
    color = parse_color(params.get("color", "#fff4d0"))
    intensity = clamp(_num(params, "intensity", default=1), 0, 4)
    dx = math.cos(angle)
    dy = math.sin(angle)
    w, h = bitmap.width, bitmap.height
    alpha = bitmap.data[..., 3].astype(np.float64)
    xs = np.clip(_jround(np.arange(w) + dx * width), 0, w - 1).astype(np.int64)
    ys = np.clip(_jround(np.arange(h) + dy * width), 0, h - 1).astype(np.int64)
    outside = alpha[np.ix_(ys, xs)]
    edge = np.clip((alpha - outside) / 255.0, 0, 1)
    keep = (alpha >= 24) & (edge > 0.02)

    rim = Bitmap(w, h)
    rim.data[..., 0] = np.where(keep, color.r, 0)
    rim.data[..., 1] = np.where(keep, color.g, 0)
    rim.data[..., 2] = np.where(keep, color.b, 0)
    rim.data[..., 3] = np.where(keep, _u8(edge * alpha), 0)
    return _screen_add(bitmap, box_blur(rim, max(1, int(_jround(width / 2))), 1), intensity)


def inner_glow(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """光彩（内側）。縁に近いほど強く光ります。

    «縁までの距離» は 4 方向へ 1 画素ずつ広げて求めます。`size` 回の
    全画面演算になりますが、画素ごとのループより桁違いに速いです。
    """
    size = max(1, int(_jround(_num(params, "size", default=12))))
    color = parse_color(params.get("color", "#ffd9a0"))
    intensity = clamp(_num(params, "intensity", default=0.8), 0, 4)
    w, h = bitmap.width, bitmap.height
    alpha = bitmap.data[..., 3]
    transparent = alpha < 24

    nearest = np.full((h, w), float(size), np.float64)
    found = np.zeros((h, w), bool)
    for k in range(1, size + 1):
        hit = np.zeros((h, w), bool)
        for ox, oy in ((k, 0), (-k, 0), (0, k), (0, -k)):
            probe = np.ones((h, w), bool)  # 範囲外は «透明とみなす»（JS と同じ）
            ys0, ys1 = max(0, -oy), min(h, h - oy)
            xs0, xs1 = max(0, -ox), min(w, w - ox)
            if ys1 > ys0 and xs1 > xs0:
                probe[ys0:ys1, xs0:xs1] = transparent[ys0 + oy : ys1 + oy, xs0 + ox : xs1 + ox]
            hit |= probe
        fresh = hit & ~found
        nearest = np.where(fresh, float(k), nearest)
        found |= hit
        if found.all():
            break

    strength = 1 - nearest / size
    keep = (alpha >= 24) & (strength > 0.02)
    glow_layer = Bitmap(w, h)
    glow_layer.data[..., 0] = np.where(keep, color.r, 0)
    glow_layer.data[..., 1] = np.where(keep, color.g, 0)
    glow_layer.data[..., 2] = np.where(keep, color.b, 0)
    glow_layer.data[..., 3] = np.where(keep, _u8(strength * alpha), 0)
    return _screen_add(bitmap, glow_layer, intensity)


def halftone(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ハーフトーン（網点）。"""
    dot_size = int(clamp(_jround(_num(params, "dotSize", default=6)), 2, 64))
    angle = math.radians(_num(params, "angle", default=45))
    shape = params.get("shape", "circle")
    background = parse_color(params.get("background", "#ffffff"))
    foreground = parse_color(params.get("foreground", "#101010"))
    cos = math.cos(angle)
    sin = math.sin(angle)
    ys, xs = np.mgrid[0 : bitmap.height, 0 : bitmap.width].astype(np.float64)
    alpha = bitmap.data[..., 3]
    lum = luma_of(_rgb_f(bitmap)).astype(np.float64) / 255.0
    rx = xs * cos - ys * sin
    ry = xs * sin + ys * cos
    fx = np.fmod(np.fmod(rx, dot_size) + dot_size, dot_size) - dot_size / 2
    fy = np.fmod(np.fmod(ry, dot_size) + dot_size, dot_size) - dot_size / 2
    radius = (1 - lum) * (dot_size / 2) * 1.35
    if shape == "square":
        inside = np.maximum(np.abs(fx), np.abs(fy)) <= radius * 0.8
    else:
        inside = np.hypot(fx, fy) <= radius

    out = Bitmap(bitmap.width, bitmap.height)
    visible = alpha > 0
    for c, (fg, bg) in enumerate(zip(foreground.rgb(), background.rgb())):
        out.data[..., c] = np.where(visible, np.where(inside, fg, bg), 0)
    out.data[..., 3] = np.where(visible, alpha, 0)
    return out


def mangaize(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """漫画化。輪郭抽出 + 階調圧縮 + 網点。"""
    levels = int(clamp(_jround(_num(params, "levels", default=3)), 2, 12))
    edge_amount = _num(params, "edge", default=1)
    dot_size = int(clamp(_jround(_num(params, "dotSize", default=5)), 0, 64))
    w, h = bitmap.width, bitmap.height
    step = 255 / (levels - 1)
    lum = luma_of(_rgb_f(bitmap)).astype(np.float64)

    if edge_amount > 0:
        total = np.zeros_like(lum)
        for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rows = np.clip(np.arange(h) + oy, 0, h - 1)
            cols = np.clip(np.arange(w) + ox, 0, w - 1)
            total += lum[np.ix_(rows, cols)]
        edge = np.abs(lum * 4 - total)
        lum = np.where(edge > 30, np.maximum(0.0, lum - edge * edge_amount), lum)

    value = _jround(lum / step) * step
    if dot_size > 1:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        fx = np.fmod(xs, dot_size) - dot_size / 2
        fy = np.fmod(ys, dot_size) - dot_size / 2
        radius = (1 - value / 255) * (dot_size / 2) * 1.4
        dotted = (value > 10) & (value < 245) & (np.hypot(fx, fy) <= radius)
        value = np.where(dotted, np.maximum(0.0, value - 70), value)

    out = bitmap.copy()
    visible = bitmap.data[..., 3] > 0
    out.data[..., :3] = np.where(visible[..., None], _u8(value)[..., None], bitmap.data[..., :3])
    return out


def polar(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """極座標変換。渦や円形展開に使います。"""
    mode = params.get("mode", "rectToPolar")
    w, h = bitmap.width, bitmap.height
    cx = _num(params, "centerX", default=0.5) * w
    cy = _num(params, "centerY", default=0.5) * h
    max_radius = math.hypot(max(cx, w - cx), max(cy, h - cy))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    if mode == "polarToRect":
        dx = xs + 0.5 - cx
        dy = ys + 0.5 - cy
        theta = (np.arctan2(dy, dx) + math.pi) / math.tau
        radius = np.hypot(dx, dy) / max_radius
        sx = theta * w
        sy = radius * h
    else:
        theta = ((xs + 0.5) / w) * math.tau - math.pi
        radius = ((ys + 0.5) / h) * max_radius
        sx = cx + np.cos(theta) * radius
        sy = cy + np.sin(theta) * radius
    out = Bitmap(w, h)
    out.data[:] = _u8(sample_bilinear(bitmap.data, sx, sy, False))
    return out


def tile(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """画像ループ（モーションタイル）。ミラーとスクロールに対応します。"""
    ctx = ctx or {}
    columns = int(clamp(_jround(_num(params, "columns", default=2)), 1, 32))
    rows = int(clamp(_jround(_num(params, "rows", default=2)), 1, 32))
    do_mirror = params.get("mirror") is True
    time = ctx.get("time", 0) or 0
    scroll_x = _num(params, "scrollX", default=0) * time
    scroll_y = _num(params, "scrollY", default=0) * time
    w, h = bitmap.width, bitmap.height
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    raw_u = (xs / w) * columns + scroll_x
    raw_v = (ys / h) * rows + scroll_y
    u = np.fmod(raw_u, 1.0)
    v = np.fmod(raw_v, 1.0)
    u = np.where(u < 0, u + 1, u)
    v = np.where(v < 0, v + 1, v)
    if do_mirror:
        u = np.where(np.abs(np.fmod(np.floor(raw_u), 2)) == 1, 1 - u, u)
        v = np.where(np.abs(np.fmod(np.floor(raw_v), 2)) == 1, 1 - v, v)
    out = Bitmap(w, h)
    out.data[:] = _u8(sample_bilinear(bitmap.data, u * w, v * h, True))
    return out


def peripheral_blur(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """もや（周辺ボケ光量）。中心はそのまま、周辺だけぼかして明るくします。"""
    radius = clamp(_num(params, "radius", default=0.55), 0, 1.5)
    softness = max(1e-3, _num(params, "softness", default=0.4))
    blur_amount = max(1, int(_jround(_num(params, "blur", default=8))))
    light = clamp(_num(params, "light", default=0.25), 0, 1)
    color = parse_color(params.get("color", "#ffffff"))
    blurred = box_blur(bitmap, blur_amount, 2)
    cx = bitmap.width / 2
    cy = bitmap.height / 2
    max_distance = math.hypot(cx, cy) or 1
    ys, xs = np.mgrid[0 : bitmap.height, 0 : bitmap.width].astype(np.float64)
    d = np.hypot(xs - cx, ys - cy) / max_distance
    weight = np.clip((d - radius) / softness, 0, 1)
    touched = weight > 0

    base = _rgb_f(bitmap)
    mixed = base + (blurred.data[..., :3].astype(np.float64) - base) * weight[..., None]
    target = np.array(color.rgb(), np.float64)
    mixed = mixed + (target - mixed) * (weight * light)[..., None]
    a = bitmap.data[..., 3].astype(np.float64)
    a = a + (blurred.data[..., 3].astype(np.float64) - a) * weight

    out = bitmap.copy()
    out.data[..., :3] = np.where(touched[..., None], _u8(mixed), bitmap.data[..., :3])
    out.data[..., 3] = np.where(touched, _u8(a), bitmap.data[..., 3])
    return out


def letterbox(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """シネスコ（上下の黒帯）。"""
    ratio = _num(params, "ratio", default=2.35)
    color = parse_color(params.get("color", "#000000"))
    target_height = bitmap.width / ratio
    if target_height >= bitmap.height:
        return bitmap.copy()
    bar = int(_jround((bitmap.height - target_height) / 2))
    out = bitmap.copy()
    fill = (color.r, color.g, color.b, int(_u8(np.array(255 * color.a))))
    out.data[:bar] = fill
    out.data[bitmap.height - bar :] = fill
    return out


def _stop_nodes(stops):
    """`stops` を `np.interp` に渡せる形（x の並びと (n, 3) の色）へ。"""
    parsed = sorted(
        ({"offset": clamp(s.get("offset", 0) or 0, 0, 1), "color": parse_color(s.get("color"))} for s in stops),
        key=lambda s: s["offset"],
    )
    xs = np.array([s["offset"] for s in parsed], np.float64)
    ys = np.array([s["color"].rgb() for s in parsed], np.float64)
    return xs, ys


def gradient_overlay(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """グラデーション色調補正。**色は `colors` ではなく `stops` です。**

        { "type": "gradientOverlay",
          "stops": [{ "offset": 0, "color": "#ff7ad9" },
                    { "offset": 1, "color": "#3ad6ff" }],
          "blend": "overlay", "angle": 90, "opacity": 0.5 }

    `colors: ["#a", "#b"]` と書いても効きません（`lightLeak` は `colors` です。
    ここは JS 版からの仕様なので、名前を揃えずにそのまま移しています）。
    """
    stops = params.get("stops") or [
        {"offset": 0, "color": "#ff7ad9"},
        {"offset": 1, "color": "#3ad6ff"},
    ]
    xs_nodes, ys_nodes = _stop_nodes(stops)
    opacity_value = clamp(_num(params, "opacity", default=0.5), 0, 1)
    blend = params.get("blend", "overlay")
    angle = math.radians(_num(params, "angle", default=90))
    dx = math.cos(angle)
    dy = math.sin(angle)
    projection = abs(dx * bitmap.width) + abs(dy * bitmap.height) or 1

    ys, xs = np.mgrid[0 : bitmap.height, 0 : bitmap.width].astype(np.float64)
    px = xs - (bitmap.width if dx < 0 else 0)
    py = ys - (bitmap.height if dy < 0 else 0)
    t = np.clip((px * dx + py * dy) / projection, 0, 1)
    # mixColor は Math.round で丸めるので、ここでも同じように丸めます
    source = np.stack([_jround(np.interp(t, xs_nodes, ys_nodes[:, c])) for c in range(3)], axis=-1)

    cb = _rgb_f(bitmap)
    cs = source
    if blend == "overlay":
        cs = np.where(cb < 128, 2 * cb * cs / 255.0, 255.0 - 2 * (255.0 - cb) * (255.0 - cs) / 255.0)
    elif blend == "screen":
        cs = 255.0 - (255.0 - cb) * (255.0 - cs) / 255.0
    elif blend == "multiply":
        cs = cb * cs / 255.0
    elif blend == "add":
        cs = np.minimum(255.0, cb + cs)
    out = bitmap.copy()
    out.data[..., :3] = _u8(cb + (cs - cb) * opacity_value)
    return out


def luminance_key(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ルミナンスキー。明度で透過させます。"""
    level = clamp(_num(params, "threshold", default=0.3), 0, 1) * 255
    softness = max(1e-3, _num(params, "softness", default=0.1) * 255)
    inverted = params.get("invert") is True
    lum = luma_of(_rgb_f(bitmap)).astype(np.float64)
    weight = np.clip((lum - level) / softness, 0, 1)
    if inverted:
        weight = 1 - weight
    out = bitmap.copy()
    out.data[..., 3] = _u8(bitmap.data[..., 3].astype(np.float64) * weight)
    return out


def color_key(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """カラーキー。指定色に近い部分を透過させます。"""
    key = parse_color(params.get("color", "#000000"))
    tolerance = clamp(_num(params, "tolerance", default=0.2), 0, 1) * 441.67
    softness = max(1e-3, clamp(_num(params, "softness", default=0.1), 0, 1) * 441.67)
    base = _rgb_f(bitmap)
    distance = np.sqrt(
        (base[..., 0] - key.r) ** 2 + (base[..., 1] - key.g) ** 2 + (base[..., 2] - key.b) ** 2
    )
    a = bitmap.data[..., 3].astype(np.float64)
    out = bitmap.copy()
    out.data[..., 3] = _u8(
        np.where(
            distance < tolerance,
            0.0,
            np.where(distance < tolerance + softness, a * (distance - tolerance) / softness, a),
        )
    )
    return out


def pixel_sort(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ピクセルソート。行ごとに明度で並べ替えるグリッチです。

    **ここは NumPy では書けません。** «どこからどこまでを並べ替えるか» が
    画素の値で決まるので、走査しながら区間を切る必要があります（Numba）。
    """
    vertical = params.get("axis") == "vertical"
    level = clamp(_num(params, "threshold", default=0.5), 0, 1) * 255
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    random = Random(int(_num(params, "seed", default=5)))
    lines = bitmap.width if vertical else bitmap.height
    keep = np.fromiter((0 if random() > amount else 1 for _ in range(lines)), np.int8, lines)
    out = bitmap.copy()
    _k_pixel_sort(bitmap.data, out.data, vertical, float(level), keep)
    return out


def reflection(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """水面（反射）。下側に上下反転した像を作り、ラスターで揺らします。"""
    ctx = ctx or {}
    position = clamp(_num(params, "position", default=0.65), 0.05, 0.98)
    opacity_value = clamp(_num(params, "opacity", default=0.4), 0, 1)
    blur_amount = max(0, int(_jround(_num(params, "blur", default=2))))
    amplitude = _num(params, "amplitude", default=4)
    frequency = _num(params, "frequency", default=5)
    time = ctx.get("time", 0) or 0
    speed = _num(params, "speed", default=1)
    w, h = bitmap.width, bitmap.height
    line = int(_jround(h * position))
    out = bitmap.copy()
    if line >= h:
        return out

    rows = np.arange(line, h, dtype=np.float64)
    depth = (rows - line) / max(1, h - line)
    offset = np.sin((depth * frequency + time * speed) * math.tau) * amplitude * (0.4 + depth)
    fade = (1 - depth) * opacity_value
    src_y = line - (rows - line)
    valid_row = src_y >= 0

    ys, xs = np.mgrid[line:h, 0:w].astype(np.float64)
    sample = sample_bilinear(bitmap.data, xs + 0.5 + offset[:, None], src_y[:, None] + 0.5, True)
    touched = valid_row[:, None] & (sample[..., 3] > 0)
    a = np.broadcast_to(fade[:, None], touched.shape)

    band = bitmap.data[line:h].astype(np.float64)
    mixed = band[..., :3] + (sample[..., :3].astype(np.float64) - band[..., :3]) * a[..., None]
    alpha = np.maximum(band[..., 3], sample[..., 3].astype(np.float64) * a)
    out.data[line:h, ..., :3] = np.where(touched[..., None], _u8(mixed), bitmap.data[line:h, ..., :3])
    out.data[line:h, ..., 3] = np.where(touched, _u8(alpha), bitmap.data[line:h, ..., 3])

    if blur_amount > 0:
        # 反射のところだけ柔らかくする（水面の «下側だけ» をぼかす）
        blurred = box_blur(out, blur_amount, 1)
        out.data[line:h] = blurred.data[line:h]
    return out


# ── 質感と色づくり ────────────────────────────────────────────────
# effects.js が «基本の加工»、MV 群が «定番演出» なのに対し、ここは
# «見た目の作り込み» です。素材を用意せずに絵の印象を決めます。


def _stepped_random(seed, time, fps, interval) -> Random:
    """`interval` フレームごとに切り替わる乱数。毎フレーム変わると忙しすぎるとき用。"""
    step = max(1, int(_jround(interval if interval is not None else 1)))
    frame = math.floor((time or 0) * (fps or 30))
    mixed = ((frame // step) * 2654435761) & _M32
    return Random(((int(seed or 0) & _M32) ^ mixed) & _M32)


def _clamped_gather(arr: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """`(y + oy, x + ox)` を端を伸ばして読む。畳み込みの «はみ出し» 対策です。"""
    h, w = arr.shape[0], arr.shape[1]
    rows = np.clip(np.arange(h) + oy, 0, h - 1)
    cols = np.clip(np.arange(w) + ox, 0, w - 1)
    return arr[np.ix_(rows, cols)] if arr.ndim == 2 else arr[np.ix_(rows, cols, np.arange(arr.shape[2]))]


def _bayer_matrix(size: int) -> np.ndarray:
    """Bayer（順序ディザ）行列。2x2 を再帰的に 4 倍していく古典的な作り方です。"""
    matrix = np.array([[0, 2], [3, 1]], np.int64)
    while matrix.shape[0] < size:
        n = matrix.shape[0]
        base = matrix * 4
        quad = np.array([[0, 2], [3, 1]], np.int64)
        grown = np.empty((n * 2, n * 2), np.int64)
        for qy in range(2):
            for qx in range(2):
                grown[qy * n : qy * n + n, qx * n : qx * n + n] = base + quad[qy, qx]
        matrix = grown
    return matrix


def dither(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ディザリング。限られた色数で中間色を «錯覚させます»。

    `pattern` は市松（`bayer2` / `bayer4` / `bayer8`）と誤差拡散
    （`floydSteinberg`）から選びます。`pixelate` の «前» に掛けると、
    ドットの粒とディザの粒が揃ってドット絵らしくなります。
    """
    levels = max(2, int(_jround(_num(params, "levels", default=4))))
    pattern = params.get("pattern", "bayer4")
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    palette_spec = params.get("palette")
    palette = (
        np.array([parse_color(c).rgb() for c in palette_spec], np.float32)
        if isinstance(palette_spec, (list, tuple)) and len(palette_spec)
        else None
    )
    base = _rgb_f(bitmap)
    step = 255 / (levels - 1)

    if pattern == "floydSteinberg":
        out = bitmap.copy()
        buf = np.empty(base.shape, np.float32)
        _k_floyd_steinberg(
            bitmap.data[..., :3].astype(np.float32),
            buf,
            levels,
            float(amount),
            palette if palette is not None else np.zeros((1, 3), np.float32),
            palette is not None,
        )
        out.data[..., :3] = _u8(buf)
        return out

    size = 2 if pattern == "bayer2" else 8 if pattern == "bayer8" else 4
    matrix = _bayer_matrix(size)
    h, w = bitmap.height, bitmap.width
    bias = (matrix[np.arange(h) % size][:, np.arange(w) % size] / (size * size) - 0.5) * step
    shifted = np.clip(base + bias[..., None], 0, 255)
    if palette is not None:
        # 目の感度に合わせて緑を重く見る。単純な距離だと «明るさは合っているのに
        # 色味が飛ぶ» ことがあります。
        d = (
            2 * (palette[:, 0] - shifted[..., None, 0]) ** 2
            + 4 * (palette[:, 1] - shifted[..., None, 1]) ** 2
            + 3 * (palette[:, 2] - shifted[..., None, 2]) ** 2
        )
        quantised = palette[np.argmin(d, axis=-1)].astype(np.float64)
    else:
        quantised = _jround(shifted / step) * step
    out = bitmap.copy()
    out.data[..., :3] = _u8(base + (quantised - base) * amount)
    return out


def misregistration(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """印刷の版ズレ。**CMY の版ごとに別方向へずらします。**

    `chromaticAberration` は RGB を左右にずらすだけなので «デジタルの色収差» に
    見えます。昭和レトロ印刷の «らしさ» は版が 1〜2 画素ずれていることから
    来るので、色を CMY に分解してから版ごとに動かします。
    """
    ctx = ctx or {}
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    if amount <= 0:
        return bitmap.copy()
    jitter = max(0, _num(params, "jitter", default=0))
    plates = params.get("plates") or [
        {"color": "cyan", "x": -1.4, "y": 0.4},
        {"color": "magenta", "x": 1.1, "y": -0.9},
        {"color": "yellow", "x": 0.4, "y": 1.3},
    ]
    w, h = bitmap.width, bitmap.height
    frame = math.floor((ctx.get("time", 0) or 0) * (ctx.get("fps", 30) or 30))
    random = Random(int(_num(params, "seed", default=7)) * 7919 + frame)

    channel_of = {"cyan": 0, "magenta": 1, "yellow": 2}
    shifted = np.zeros((h, w, 3), np.float64)
    data = bitmap.data
    for plate in plates:
        channel = channel_of.get(str(plate.get("color", "")).lower())
        if channel is None:
            continue
        dx = (plate.get("x", 0) or 0) * amount + ((random() - 0.5) * jitter if jitter else 0)
        dy = (plate.get("y", 0) or 0) * amount + ((random() - 0.5) * jitter if jitter else 0)
        rows = np.clip(_jround(np.arange(h) - dy), 0, h - 1).astype(np.int64)
        cols = np.clip(_jround(np.arange(w) - dx), 0, w - 1).astype(np.int64)
        shifted[..., channel] = 255 - data[np.ix_(rows, cols)][..., channel].astype(np.float64)

    out = bitmap.copy()
    out.data[..., :3] = _u8(255 - shifted)
    return out


def retro_film(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """レトロフィルム。セピア・粒子・縦傷・埃・明滅・揺れをまとめて掛けます。

    各パラメーターを 0 にすると、その要素だけが消えます。乱数の引く順は
    JS 版と 1 回ずつ揃えてあるので、傷や埃の位置まで同じになります。
    """
    ctx = ctx or {}
    sepia = clamp(_num(params, "sepia", default=0.7), 0, 1)
    grain = max(0, _num(params, "grain", default=0.25))
    scratches = clamp(_num(params, "scratches", default=0.4), 0, 1)
    dust = clamp(_num(params, "dust", default=0.3), 0, 1)
    flicker = clamp(_num(params, "flicker", default=0.15), 0, 1)
    jitter = max(0, _num(params, "jitter", default=2))
    vignette_amount = clamp(_num(params, "vignette", default=0.4), 0, 1)
    seed = _num(params, "seed", default=5)
    random = _stepped_random(seed, ctx.get("time"), ctx.get("fps"), _num(params, "interval", default=2))

    shift_x = int(_jround((random() - 0.5) * 2 * jitter)) if jitter > 0 else 0
    shift_y = int(_jround((random() - 0.5) * 2 * jitter)) if jitter > 0 else 0
    exposure = 1 + (random() - 0.5) * 2 * flicker

    w, h = bitmap.width, bitmap.height
    scratch_lines = []
    for _ in range(int(_jround(scratches * 6))):
        scratch_lines.append(
            {
                "x": random() * w,
                "width": 0.6 + random() * 1.8,
                "top": random() * h * 0.5,
                "bottom": h * (0.5 + random() * 0.5),
                "bright": random() > 0.35,
                "strength": 0.3 + random() * 0.7,
            }
        )
    dust_spots = []
    for _ in range(int(_jround(dust * 40))):
        dust_spots.append(
            {"x": random() * w, "y": random() * h, "radius": 0.7 + random() * 2.2, "dark": random() > 0.4}
        )

    rows = np.clip(np.arange(h) + shift_y, 0, h - 1)
    cols = np.clip(np.arange(w) + shift_x, 0, w - 1)
    source = bitmap.data[np.ix_(rows, cols)]
    rgb = source[..., :3].astype(np.float64)
    alpha = source[..., 3]

    if sepia > 0:
        lum = luma_of(rgb).astype(np.float64)
        target = np.stack(
            [
                np.clip(lum * 1.07 + 18, 0, 255),
                np.clip(lum * 0.94 + 8, 0, 255),
                np.clip(lum * 0.72, 0, 255),
            ],
            axis=-1,
        )
        rgb = rgb + (target - rgb) * sepia
    rgb = rgb * exposure

    if grain > 0:
        # 位置と時刻から決まるノイズ。乱数を «回さない» ので順番に依存しません。
        # （JS 版の fbm2D は octaves=1 のとき 2 次元値ノイズと一致します）
        noise_seed = int(_jround(seed + (ctx.get("time", 0) or 0) * 977))
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        rgb = rgb + (value_noise_2d(xs * 0.7, ys * 0.7, noise_seed) * grain * 90)[..., None]

    if vignette_amount > 0:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        dx = (xs / w - 0.5) * 2
        dy = (ys / h - 0.5) * 2
        falloff = 1 - np.clip(np.sqrt(dx * dx + dy * dy) - 0.4, 0, 1) * vignette_amount * 1.6
        rgb = rgb * falloff[..., None]

    out = Bitmap(w, h)
    out.data[..., :3] = _u8(rgb)
    out.data[..., 3] = alpha

    for line in scratch_lines:
        y_from = max(0, math.floor(line["top"]))
        y_to = min(h, math.ceil(line["bottom"]))
        if y_to <= y_from:
            continue
        half = line["width"] / 2
        x_from = max(0, math.floor(line["x"] - half))
        x_to = min(w - 1, math.ceil(line["x"] + half))
        if x_to < x_from:
            continue
        xs = np.arange(x_from, x_to + 1, dtype=np.float64)
        coverage = np.clip(half - np.abs(xs + 0.5 - line["x"]) + 0.5, 0, 1) * line["strength"]
        target = 255.0 if line["bright"] else 0.0
        block = out.data[y_from:y_to, x_from : x_to + 1, :3].astype(np.float64)
        out.data[y_from:y_to, x_from : x_to + 1, :3] = _u8(
            block + (target - block) * coverage[None, :, None]
        )

    for spot in dust_spots:
        radius = math.ceil(spot["radius"])
        offsets = np.arange(-radius, radius + 1)
        px = _jround(spot["x"] + offsets).astype(np.int64)
        py = _jround(spot["y"] + offsets).astype(np.int64)
        inx = (px >= 0) & (px < w)
        iny = (py >= 0) & (py < h)
        if not inx.any() or not iny.any():
            continue
        distance = np.hypot(offsets[iny][:, None].astype(np.float64), offsets[inx][None, :].astype(np.float64))
        coverage = np.clip(spot["radius"] - distance, 0, 1) * 0.85
        target = 20.0 if spot["dark"] else 235.0
        sel = np.ix_(py[iny], px[inx])
        block = out.data[..., :3][sel].astype(np.float64)
        out.data[..., :3][sel] = _u8(block + (target - block) * coverage[..., None])
    return out


def light_leak(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ライトリーク。角度方向の帯状グラデーションをスクリーン合成します。

    **こちらの色は `colors`（ただの色の配列）です。** `gradientOverlay` の
    `stops` とは書き方が違うので、混ぜないでください（JS 版からの仕様）。
    """
    intensity = clamp(_num(params, "intensity", default=0.7), 0, 4)
    if intensity <= 0:
        return bitmap.copy()
    angle = math.radians(_num(params, "angle", default=35))
    position = _num(params, "position", default=0.5)
    width = max(0.01, _num(params, "width", default=0.45))
    softness = clamp(_num(params, "softness", default=0.6), 0, 1)
    blend = params.get("blend", "screen")
    colors = [parse_color(c) for c in (params.get("colors") or ["#ff9a3c", "#ff4d6d", "#ffd166"])]
    if not colors:
        return bitmap.copy()

    cos = math.cos(angle)
    sin = math.sin(angle)
    h, w = bitmap.height, bitmap.width
    v = (np.arange(h, dtype=np.float64) / max(1, h - 1) - 0.5)[:, None]
    u = (np.arange(w, dtype=np.float64) / max(1, w - 1) - 0.5)[None, :]
    projected = (u * cos + v * sin) * 0.5 + 0.5
    distance = np.abs(projected - position) / (width * 0.5)
    inside = distance < 1
    falloff = np.power(np.maximum(0.0, 1 - distance), 1 + softness * 3)

    t = np.clip((projected - (position - width / 2)) / width, 0, 1) * (len(colors) - 1)
    index = np.minimum(len(colors) - 2, np.floor(t)).astype(np.int64) if len(colors) > 1 else np.zeros_like(t, np.int64)
    index = np.maximum(index, 0)
    k = (t - index) if len(colors) > 1 else np.zeros_like(t)
    palette = np.array([c.rgb() for c in colors], np.float64)
    a = palette[index]
    b = palette[np.minimum(len(colors) - 1, index + 1)]
    strength = falloff * intensity
    light = (a + (b - a) * k[..., None]) * strength[..., None]

    base = _rgb_f(bitmap)
    if blend == "add":
        mixed = np.clip(base + light, 0, 255)
    else:
        mixed = np.clip(255.0 - (255.0 - base) * (255.0 - light) / 255.0, 0, 255)
    out = bitmap.copy()
    out.data[..., :3] = np.where(inside[..., None], _u8(mixed), bitmap.data[..., :3])
    return out


def colorama(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """コロラマ。輝度（または色相・アルファ）を «循環する» パレットに写します。"""
    palette_spec = params.get("palette") or ["#ff0080", "#ffee00", "#00ffcc", "#5533ff"]
    palette = np.array([parse_color(c).rgb() for c in palette_spec], np.float64)
    if len(palette) == 0:
        return bitmap.copy()
    phase = _num(params, "phase", default=0)
    cycles = _num(params, "cycles", default=1)
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    source = params.get("source", "luminance")

    base = _rgb_f(bitmap)
    if source == "alpha":
        value = bitmap.data[..., 3].astype(np.float64) / 255.0
    elif source == "hue":
        value = rgb_to_hsl(base)[..., 0]
    else:
        value = luma_of(base).astype(np.float64) / 255.0

    t = value * cycles + phase
    wrapped = (np.fmod(t, 1.0) + 1.0) % 1.0
    scaled = wrapped * len(palette)
    index = (np.floor(scaled).astype(np.int64)) % len(palette)
    nxt = (index + 1) % len(palette)
    k = (scaled - np.floor(scaled))[..., None]
    color = palette[index] + (palette[nxt] - palette[index]) * k

    out = bitmap.copy()
    visible = bitmap.data[..., 3] != 0
    out.data[..., :3] = np.where(visible[..., None], _u8(base + (color - base) * amount), bitmap.data[..., :3])
    return out


def leave_color(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """色抜き。指定した色相の範囲だけ彩度を残し、他をモノクロにします。"""
    hue = (_num(params, "hue", default=0) % 360) / 360
    tolerance = max(0, _num(params, "tolerance", default=30)) / 360
    softness = max(0, _num(params, "softness", default=15)) / 360
    desaturation = clamp(_num(params, "desaturation", default=1), 0, 1)
    keep_luminance = params.get("keepLuminance") is not False

    base = _rgb_f(bitmap)
    hsl = rgb_to_hsl(base)
    h = hsl[..., 0]
    distance = np.abs(h - hue)
    distance = np.where(distance > 0.5, 1 - distance, distance)
    keep = np.where(
        distance <= tolerance,
        1.0,
        np.where(
            (softness > 0) & (distance <= tolerance + softness),
            1 - (distance - tolerance) / (softness or 1),
            0.0,
        ),
    )
    amount = (1 - keep) * desaturation
    if keep_luminance:
        grey = luma_of(base).astype(np.float64)
        target = np.repeat(grey[..., None], 3, axis=-1)
    else:
        flat = hsl.copy()
        flat[..., 1] = 0
        target = hsl_to_rgb(flat)

    out = bitmap.copy()
    touched = (bitmap.data[..., 3] != 0) & (keep < 1)
    out.data[..., :3] = np.where(
        touched[..., None], _u8(base + (target - base) * amount[..., None]), bitmap.data[..., :3]
    )
    return out


def monochrome(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """単色化。**`tint` と違い、明暗の情報が残ります。**"""
    color = parse_color(params.get("color", "#ffffff"))
    amount = clamp(_num(params, "amount", default=1), 0, 1)
    keep_luminance = params.get("keepLuminance") is not False
    contrast = _num(params, "contrast", default=0)
    base = _rgb_f(bitmap)
    lum = luma_of(base).astype(np.float64) / 255.0
    if contrast != 0:
        lum = np.clip((lum - 0.5) * (1 + contrast) + 0.5, 0, 1)
    scale = lum * 2 if keep_luminance else np.ones_like(lum)
    target = np.clip(np.array(color.rgb(), np.float64) * scale[..., None], 0, 255)
    out = bitmap.copy()
    visible = bitmap.data[..., 3] != 0
    out.data[..., :3] = np.where(
        visible[..., None], _u8(base + (target - base) * amount), bitmap.data[..., :3]
    )
    return out


def inner_distance_field(bitmap: Bitmap, limit: float) -> np.ndarray:
    """アルファの内側距離場（輪郭から何画素内側か）。`bevel` が使います。"""
    return _k_inner_distance(bitmap.data[..., 3], float(max(1, limit)))


def bevel(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ベベル（面取り）。距離場の勾配を «疑似的な法線» として陰影を付けます。"""
    size = max(1, _num(params, "size", default=6))
    strength = clamp(_num(params, "strength", default=0.8), 0, 2)
    if strength <= 0:
        return bitmap.copy()
    angle = math.radians(_num(params, "angle", default=-45))
    light_x = math.cos(angle)
    light_y = math.sin(angle)
    highlight = parse_color(params.get("highlight", "#ffffff"))
    shadow = parse_color(params.get("shadow", "#000000"))
    style = params.get("style", "inner")

    field = inner_distance_field(bitmap, size).astype(np.float64)
    gx = _clamped_gather(field, 1, 0) - _clamped_gather(field, -1, 0)
    gy = _clamped_gather(field, 0, 1) - _clamped_gather(field, 0, -1)
    length = np.hypot(gx, gy)
    safe = np.where(length < 1e-6, 1.0, length)
    dot = (gx / safe) * light_x + (gy / safe) * light_y
    if style == "emboss":
        falloff = np.exp(-field / (size * 0.5))
    else:
        falloff = 1 - field / size
    shade = dot * strength * falloff

    in_bevel = field < size
    touched = (field > 0) & (length >= 1e-6) & (in_bevel | (style == "emboss"))
    amount = np.clip(np.abs(shade), 0, 1)
    target = np.where(
        (shade >= 0)[..., None],
        np.array(highlight.rgb(), np.float64),
        np.array(shadow.rgb(), np.float64),
    )
    base = _rgb_f(bitmap)
    out = bitmap.copy()
    out.data[..., :3] = np.where(
        touched[..., None], _u8(base + (target - base) * amount[..., None]), bitmap.data[..., :3]
    )
    return out


def directional_light(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """方向性ライト。輝度（またはアルファ）の勾配を法線とみなして陰影を付けます。"""
    intensity = clamp(_num(params, "intensity", default=0.6), 0, 4)
    ambient = clamp(_num(params, "ambient", default=0.4), 0, 2)
    angle = math.radians(_num(params, "angle", default=-45))
    elevation = math.radians(_num(params, "elevation", default=40))
    color = parse_color(params.get("color", "#fff6e0"))
    bump_from = params.get("bumpFrom", "luminance")
    bump_strength = _num(params, "bumpStrength", default=1)

    lx = math.cos(angle) * math.cos(elevation)
    ly = math.sin(angle) * math.cos(elevation)
    lz = math.sin(elevation)

    base = _rgb_f(bitmap)
    height_map = (
        bitmap.data[..., 3].astype(np.float64) / 255.0
        if bump_from == "alpha"
        else luma_of(base).astype(np.float64) / 255.0
    )
    gx = (_clamped_gather(height_map, 1, 0) - _clamped_gather(height_map, -1, 0)) * bump_strength * 4
    gy = (_clamped_gather(height_map, 0, 1) - _clamped_gather(height_map, 0, -1)) * bump_strength * 4
    length = np.sqrt(gx * gx + gy * gy + 1.0)
    length = np.where(length == 0, 1.0, length)
    dot = ((-gx / length) * lx + (-gy / length) * ly + (1.0 / length) * lz) * intensity
    light = np.clip(ambient + np.maximum(0.0, dot), 0, 3)

    tinted = np.clip(base * light[..., None] * np.array(color.rgb(), np.float64) / 255.0, 0, 255)
    out = bitmap.copy()
    visible = bitmap.data[..., 3] != 0
    out.data[..., :3] = np.where(visible[..., None], _u8(tinted), bitmap.data[..., :3])
    return out


def long_shadow(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ロングシャドー。アルファを一方向へ伸ばした平面的な影を敷きます。"""
    length = max(0, int(_jround(_num(params, "length", default=120))))
    if length == 0:
        return bitmap.copy()
    angle = math.radians(_num(params, "angle", default=45))
    dx = math.cos(angle)
    dy = math.sin(angle)
    color = parse_color(params.get("color", "rgba(0,0,0,0.35)"))
    fade = clamp(_num(params, "fade", default=0.5), 0, 1)
    behind = params.get("behind") is not False

    w, h = bitmap.width, bitmap.height
    alpha01 = bitmap.data[..., 3].astype(np.float64) / 255.0
    # **float32 です。** JS 版が Float32Array で積むので、ここを float64 に
    # すると «縁の 1 段» が変わります（0.857142857 が 0.857142866 に丸まる差）。
    shadow = np.zeros((h, w), np.float32)
    _k_long_shadow(np.ascontiguousarray(bitmap.data[..., 3]), shadow, length,
                   float(dx), float(dy), float(fade))

    shadow_alpha = shadow.astype(np.float64) * (color.a if color.a is not None else 1)
    source_alpha = alpha01
    base = _rgb_f(bitmap)
    shadow_rgb = np.array(color.rgb(), np.float64)
    if behind:
        out_alpha = shadow_alpha + source_alpha * (1 - shadow_alpha)
        safe = np.where(out_alpha > 0, out_alpha, 1.0)
        rgb = (
            shadow_rgb * shadow_alpha[..., None] * (1 - source_alpha)[..., None]
            + base * source_alpha[..., None]
        ) / safe[..., None]
    else:
        out_alpha = source_alpha + shadow_alpha * (1 - source_alpha)
        safe = np.where(out_alpha > 0, out_alpha, 1.0)
        rgb = (
            base * source_alpha[..., None]
            + shadow_rgb * shadow_alpha[..., None] * (1 - source_alpha)[..., None]
        ) / safe[..., None]
    out = Bitmap(w, h)
    touched = out_alpha > 0
    out.data[..., :3] = np.where(touched[..., None], _u8(rgb), 0)
    out.data[..., 3] = np.where(touched, _u8(out_alpha * 255), 0)
    return out


def graphic_pen(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """グラフィックペン。斜線の密度で階調を出します。`layers` 2 以上でクロスハッチ。"""
    spacing = max(2, _num(params, "spacing", default=5))
    thickness = max(0.4, _num(params, "thickness", default=1))
    contrast = _num(params, "contrast", default=1.2)
    background = parse_color(params.get("background", "#ffffff"))
    ink = parse_color(params.get("ink", "#101010"))
    layers = int(clamp(_jround(_num(params, "layers", default=2)), 1, 4))
    base_angle = math.radians(_num(params, "angle", default=45))

    h, w = bitmap.height, bitmap.width
    base = _rgb_f(bitmap)
    lum = np.clip((luma_of(base).astype(np.float64) / 255.0 - 0.5) * contrast + 0.5, 0, 1)
    darkness = 1 - lum
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)

    coverage = np.zeros((h, w), np.float64)
    for layer in range(layers):
        angle = base_angle + (layer * math.pi) / layers
        projected = xs * math.cos(angle) + ys * math.sin(angle)
        phase = (np.fmod(projected, spacing) + spacing) % spacing
        distance = np.abs(phase - spacing / 2)
        share = np.clip(darkness * layers - layer, 0, 1)
        half = (thickness * share) / 2
        contribution = np.where(half > 0, np.clip(half - distance + 0.5, 0, 1), 0.0)
        coverage = np.maximum(coverage, contribution)

    bg = np.array(background.rgb(), np.float64)
    fg = np.array(ink.rgb(), np.float64)
    out = Bitmap(w, h)
    visible = bitmap.data[..., 3] != 0
    out.data[..., :3] = np.where(visible[..., None], _u8(bg + (fg - bg) * coverage[..., None]), 0)
    out.data[..., 3] = np.where(visible, bitmap.data[..., 3], 0)
    return out


def hex_tile(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """六角タイル。`mode` は repeat / mirror / kaleido。"""
    size = max(4, _num(params, "size", default=80))
    rotation = math.radians(_num(params, "rotation", default=0))
    mode = params.get("mode", "mirror")
    outline = max(0, _num(params, "outline", default=0))
    outline_color = parse_color(params.get("outlineColor", "#ffffff"))

    w, h = bitmap.width, bitmap.height
    centre_x = w / 2
    centre_y = h / 2
    cos = math.cos(rotation)
    sin = math.sin(rotation)
    sqrt3 = math.sqrt(3)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    rx = (xs - centre_x) * cos + (ys - centre_y) * sin
    ry = -(xs - centre_x) * sin + (ys - centre_y) * cos

    q = ((sqrt3 / 3) * rx - (1 / 3) * ry) / size
    r = ((2 / 3) * ry) / size
    # 立方体座標に移して丸め、ずれが最大の軸を他から復元する（六角格子の定石）
    s = -q - r
    rq = _jround(q)
    rr = _jround(r)
    rs = _jround(s)
    dq = np.abs(rq - q)
    dr = np.abs(rr - r)
    ds = np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    cell_q = np.where(fix_q, -rr - rs, rq)
    cell_r = np.where(fix_r, -rq - rs, rr)

    centre_px = size * sqrt3 * (cell_q + cell_r / 2)
    centre_py = size * (3 / 2) * cell_r
    local_x = rx - centre_px
    local_y = ry - centre_py

    if mode == "mirror":
        flip = ((cell_q + cell_r) % 2 + 2) % 2 != 0
        local_x = np.where(flip, -local_x, local_x)
        local_y = np.where(flip, -local_y, local_y)
    elif mode == "kaleido":
        radius = np.hypot(local_x, local_y)
        sector = math.pi / 3
        raw = np.arctan2(local_y, local_x)
        folded = np.abs(((np.fmod(raw, sector) + sector) % sector) - sector / 2)
        local_x = np.cos(folded) * radius
        local_y = np.sin(folded) * radius

    source_x = centre_x + local_x * cos - local_y * sin
    source_y = centre_y + local_x * sin + local_y * cos
    rows = np.clip(_jround(source_y), 0, h - 1).astype(np.int64)
    cols = np.clip(_jround(source_x), 0, w - 1).astype(np.int64)
    out = Bitmap(w, h)
    out.data[:] = bitmap.data[rows, cols]

    if outline > 0:
        adx = np.abs(local_x)
        ady = np.abs(local_y)
        hex_distance = np.maximum(ady / size, (ady / 2 + (adx * sqrt3) / 2) / size)
        line = np.clip((hex_distance - (1 - outline / size)) * (size / max(1, outline)), 0, 1)
        base = out.data[..., :3].astype(np.float64)
        target = np.array(outline_color.rgb(), np.float64)
        out.data[..., :3] = np.where(
            (line > 0)[..., None], _u8(base + (target - base) * line[..., None]), out.data[..., :3]
        )
    return out


def slit_scan(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """スリットスキャン／タイムディスプレイス。画面の位置ごとに «別の時刻» を混ぜます。

    履歴が無い先頭フレームでは現在のフレームで代替するので、破綻はしません。
    `ctx["frameHistory"]` は «新しい順» に並んだビットマップの列です。
    """
    ctx = ctx or {}
    history = ctx.get("frameHistory") or []
    if len(history) <= 1:
        return bitmap.copy()
    axis = "x" if params.get("axis") == "x" else "y"
    span = max(0, _num(params, "span", default=0.5))
    fps = ctx.get("fps", 30) or 30
    depth = min(len(history) - 1, max(1, int(_jround(span * fps))))
    backwards = params.get("direction", "past") == "past"
    assets = ctx.get("assets")
    mapping = assets.get(params["mapAsset"]) if (params.get("mapAsset") and assets) else None

    w, h = bitmap.width, bitmap.height
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    if mapping is not None:
        mx = np.minimum(mapping.width - 1, _jround((xs / w) * mapping.width)).astype(np.int64)
        my = np.minimum(mapping.height - 1, _jround((ys / h) * mapping.height)).astype(np.int64)
        t = luma_of(mapping.data[my, mx, :3].astype(np.float64)) / 255.0
    else:
        t = ys / max(1, h - 1) if axis == "y" else xs / max(1, w - 1)
    offset = np.clip(_jround((t if backwards else 1 - t) * depth), 0, len(history) - 1).astype(np.int64)

    out = bitmap.copy()
    for index in np.unique(offset):
        frame = history[int(index)]
        if frame is None or frame.width != w or frame.height != h:
            continue
        picked = offset == index
        out.data[picked] = frame.data[picked]
    return out


# ── かけらに分けて動かすもの ────────────────────────────────────
#
# 変形（deformer）ではなくエフェクトとして実装しています。変形は 1 枚のメッシュの
# 頂点を動かす仕組みなので、「かけらが離れて飛ぶ」動きは表現できないためです。
# ここでは元の絵をテクスチャとして、かけらごとに三角形を貼り直します。


def _fan_triangles(points):
    """凸多角形を扇状に三角形へ分ける。"""
    return [(points[0], points[i], points[i + 1]) for i in range(1, len(points) - 1)]


def _transform_point(point, centre, angle, scale, offset_x, offset_y):
    dx = point[0] - centre[0]
    dy = point[1] - centre[1]
    cos = math.cos(angle)
    sin = math.sin(angle)
    return (
        centre[0] + (dx * cos - dy * sin) * scale + offset_x,
        centre[1] + (dx * sin + dy * cos) * scale + offset_y,
    )


def _draw_piece(dst: Bitmap, src: Bitmap, polygon, place: Callable, options: dict) -> None:
    """かけら 1 枚を貼る。**`u` / `v` は元画像の «画素» 座標です。**"""
    for triangle in _fan_triangles(polygon):
        verts = []
        for p in triangle:
            x, y = place(p)
            verts.append({"x": x, "y": y, "u": p[0], "v": p[1]})
        draw_textured_triangle(dst, src, verts[0], verts[1], verts[2], options)


def _clip_polygon(points, nx, ny, offset):
    """半平面 `dot(p, n) <= offset` で切る（サザーランド・ホジマン）。"""
    if not points:
        return points
    out = []
    for i, current in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        current_in = current[0] * nx + current[1] * ny <= offset
        next_in = nxt[0] * nx + nxt[1] * ny <= offset
        if current_in:
            out.append(current)
        if current_in != next_in:
            d0 = current[0] * nx + current[1] * ny - offset
            d1 = nxt[0] * nx + nxt[1] * ny - offset
            t = d0 / (d0 - d1)
            out.append((current[0] + (nxt[0] - current[0]) * t, current[1] + (nxt[1] - current[1]) * t))
    return out


def _voronoi_cells(sites, width, height):
    """種点からボロノイ領域を作る（矩形を垂直二等分線で順に切る）。"""
    cells = []
    for i, site in enumerate(sites):
        polygon = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
        for j, other in enumerate(sites):
            if i == j or not polygon:
                continue
            nx = other[0] - site[0]
            ny = other[1] - site[1]
            mid_x = (site[0] + other[0]) / 2
            mid_y = (site[1] + other[1]) / 2
            polygon = _clip_polygon(polygon, nx, ny, nx * mid_x + ny * mid_y)
        if len(polygon) >= 3:
            cells.append(polygon)
    return cells


def _centroid(points):
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def shatter(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """ひび割れ・画面割れ。`progress` 0 で元の絵、1 で完全に飛び散った状態。

    物理エンジンは使わず、破片ごとに初速と回転を決めて解析的に動かします。
    そのほうが軽く、**どの時刻を描いても同じ結果になります**（巻き戻しに強い）。
    """
    progress = clamp(_num(params, "progress", default=0), 0, 1)
    pieces = int(clamp(_jround(_num(params, "pieces", default=24)), 2, 400))
    pattern = params.get("pattern", "voronoi")
    show_cracks = params.get("showCracks", False)
    random = Random(int(_jround(_num(params, "seed", default=7))))
    w, h = bitmap.width, bitmap.height
    centre = params.get("center") or {"x": 0.5, "y": 0.5}
    origin_x = centre.get("x", 0.5) * w
    origin_y = centre.get("y", 0.5) * h

    if pattern == "grid":
        columns = max(1, int(_jround(math.sqrt(pieces * (w / max(1, h))))))
        rows = max(1, int(_jround(pieces / columns)))
        cells = []
        for row in range(rows):
            for column in range(columns):
                x0 = column * w / columns
                x1 = (column + 1) * w / columns
                y0 = row * h / rows
                y1 = (row + 1) * h / rows
                cells.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    elif pattern == "radial":
        cells = []
        rings = max(1, int(_jround(math.sqrt(pieces / 3))))
        sectors = max(3, int(_jround(pieces / rings)))
        max_radius = math.hypot(w, h)
        for ring in range(rings):
            inner = (ring / rings) * max_radius
            outer = ((ring + 1) / rings) * max_radius
            for sector in range(sectors):
                a0 = (sector / sectors) * math.tau
                a1 = ((sector + 1) / sectors) * math.tau
                cells.append(
                    [
                        (origin_x + math.cos(a0) * inner, origin_y + math.sin(a0) * inner),
                        (origin_x + math.cos(a0) * outer, origin_y + math.sin(a0) * outer),
                        (origin_x + math.cos(a1) * outer, origin_y + math.sin(a1) * outer),
                        (origin_x + math.cos(a1) * inner, origin_y + math.sin(a1) * inner),
                    ]
                )
    else:
        sites = [(random() * w, random() * h) for _ in range(pieces)]
        cells = _voronoi_cells(sites, w, h)

    out = Bitmap(w, h)
    force = _num(params, "force", default=400)
    gravity = _num(params, "gravity", default=900)
    spin = math.radians(_num(params, "spin", default=240))
    t = progress

    for polygon in cells:
        middle = _centroid(polygon)
        dx = middle[0] - origin_x
        dy = middle[1] - origin_y
        distance = math.hypot(dx, dy) or 1
        jitter = 0.6 + random() * 0.8
        speed_x = (dx / distance) * force * jitter
        speed_y = (dy / distance) * force * jitter
        angle = (random() - 0.5) * 2 * spin * t
        offset_x = speed_x * t
        offset_y = speed_y * t + 0.5 * gravity * t * t
        scale = 1 - t * 0.15 * jitter
        alpha = clamp(1 - max(0, t - 0.6) / 0.4, 0, 1)
        if alpha <= 0:
            continue
        _draw_piece(
            out,
            bitmap,
            polygon,
            lambda p, m=middle, a=angle, s=scale, ox=offset_x, oy=offset_y: _transform_point(p, m, a, s, ox, oy),
            {"alpha": alpha, "clampEdge": True},
        )

    if show_cracks:
        color = parse_color(params.get("crackColor", "#ffffff"))
        thickness = max(0.5, _num(params, "crackWidth", default=1))
        contours = []
        for polygon in cells:
            middle = _centroid(polygon)
            span = math.hypot(middle[0] - origin_x, middle[1] - origin_y) or 1
            offset_x = ((middle[0] - origin_x) / span) * force * t
            offset_y = ((middle[1] - origin_y) / span) * force * t
            for i in range(len(polygon)):
                a = polygon[i]
                b = polygon[(i + 1) % len(polygon)]
                nx = -(b[1] - a[1])
                ny = b[0] - a[0]
                length = math.hypot(nx, ny) or 1
                hx = (nx / length) * thickness * 0.5
                hy = (ny / length) * thickness * 0.5
                contours.append(
                    [
                        a[0] + hx + offset_x, a[1] + hy + offset_y,
                        b[0] + hx + offset_x, b[1] + hy + offset_y,
                        b[0] - hx + offset_x, b[1] - hy + offset_y,
                        a[0] - hx + offset_x, a[1] - hy + offset_y,
                    ]
                )
        region = rasterize_contours(contours, w, h)
        fill_coverage(out, region, color, clamp(_num(params, "crackOpacity", default=0.8), 0, 1))
    return out


def object_split(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """格子に分けて 1 マスずつ散らす／集める。`progress` 1 で完全に散った状態。"""
    columns = int(clamp(_jround(_num(params, "columns", default=8)), 1, 128))
    rows = int(clamp(_jround(_num(params, "rows", default=5)), 1, 128))
    progress = clamp(_num(params, "progress", default=0), 0, 1)
    offset = params.get("offset") or {}
    randomness = clamp(_num(params, "random", default=1), 0, 1)
    order = params.get("order", "random")
    stagger = max(0, _num(params, "stagger", default=0))
    random = Random(int(_jround(_num(params, "seed", default=11))))
    w, h = bitmap.width, bitmap.height

    cell_width = w / columns
    cell_height = h / rows
    total = columns * rows

    pieces = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            cx = (column + 0.5) * cell_width
            cy = (row + 0.5) * cell_height
            if order == "center":
                rank = math.hypot(cx - w / 2, cy - h / 2) / math.hypot(w / 2, h / 2)
            elif order == "sequential":
                rank = index / max(1, total - 1)
            else:
                rank = random()
            pieces.append(
                {
                    "column": column,
                    "row": row,
                    "rank": rank,
                    "jitterX": (random() - 0.5) * 2,
                    "jitterY": (random() - 0.5) * 2,
                    "jitterRotation": (random() - 0.5) * 2,
                }
            )
    # order の違いは «動き出す順番» の違い。rank で並べ替えてから 0〜1 に振り直す。
    pieces.sort(key=lambda p: p["rank"])
    for index, piece in enumerate(pieces):
        piece["rank"] = index / (total - 1) if total > 1 else 0

    out = Bitmap(w, h)
    span = max(0.0001, 1 - stagger * (total - 1))
    for piece in pieces:
        delay = stagger * (total - 1) * piece["rank"]
        local = clamp((progress - delay) / span, 0, 1)

        def mix(base, jitter):
            return base * (1 - randomness) + base * jitter * randomness

        dx = mix(offset.get("x", 0) or 0, piece["jitterX"]) * local
        dy = mix(offset.get("y", 0) or 0, piece["jitterY"]) * local
        angle = math.radians(mix(offset.get("rotation", 0) or 0, piece["jitterRotation"]) * local)
        scale = 1 + ((offset.get("scale", 1) if offset.get("scale") is not None else 1) - 1) * local
        alpha = 1 + (offset["opacity"] - 1) * local if offset.get("opacity") is not None else 1
        if alpha <= 0:
            continue

        x0 = piece["column"] * cell_width
        y0 = piece["row"] * cell_height
        polygon = [
            (x0, y0),
            (x0 + cell_width, y0),
            (x0 + cell_width, y0 + cell_height),
            (x0, y0 + cell_height),
        ]
        middle = (x0 + cell_width / 2, y0 + cell_height / 2)
        _draw_piece(
            out,
            bitmap,
            polygon,
            lambda p, m=middle, a=angle, s=scale, ox=dx, oy=dy: _transform_point(p, m, a, s, ox, oy),
            {"alpha": clamp(alpha, 0, 1), "clampEdge": True},
        )
    return out


def slice_effect(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """任意角度の直線で帯に切り、帯ごとにずらします。"""
    count = int(clamp(_jround(_num(params, "count", default=7)), 1, 200))
    angle = math.radians(_num(params, "angle", default=20))
    offset = _num(params, "offset", default=60)
    randomness = clamp(_num(params, "random", default=0.8), 0, 1)
    gap = max(0, _num(params, "gap", default=0))
    random = Random(int(_jround(_num(params, "seed", default=2))))
    w, h = bitmap.width, bitmap.height

    cos = math.cos(angle)
    sin = math.sin(angle)
    extent = abs(w * sin) + abs(h * cos)
    band_height = extent / count
    half = math.hypot(w, h)
    centre_x = w / 2
    centre_y = h / 2

    out = Bitmap(w, h)
    for i in range(count):
        jitter = (random() - 0.5) * 2
        shift = offset * (1 - randomness + randomness * jitter)
        centre = -extent / 2 + band_height * (i + 0.5)
        inner = band_height / 2 - gap / 2
        if inner <= 0:
            continue

        def corner(along, across):
            return (
                centre_x + along * cos - (centre + across) * sin,
                centre_y + along * sin + (centre + across) * cos,
            )

        polygon = [corner(-half, -inner), corner(half, -inner), corner(half, inner), corner(-half, inner)]
        _draw_piece(
            out,
            bitmap,
            polygon,
            lambda p, s=shift: (p[0] + s * cos, p[1] + s * sin),
            {"alpha": 1, "clampEdge": False},
        )
    return out


# ── 登録と呼び出し ────────────────────────────────────────────────

class _Registry(dict):
    """エフェクトの表。**引かれた時点でグレーディング系を読み込みます。**

    `effects_grade` はこのファイルの色の道具（`rgb_to_hsl` など）を使うので、
    先頭で import すると循環します。JS 版が `Object.assign` を末尾に置いて
    いるのと同じ事情ですが、Python では «どちらのファイルから import されても
    動く» ようにしておかないと、`import movo.renderer.effects_grade` だけで
    落ちます。そこで «初回に引いたとき» に足す形にしています。
    """

    def __missing__(self, key):
        _ensure_grade()
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        raise KeyError(key)

    # `in` / `get` / 走査でも «揃った表» が見えるようにします。ここを抜かすと
    # `"curves" in effects` だけ False になる、という嫌な半端さが残ります。
    def __contains__(self, key):
        _ensure_grade()
        return dict.__contains__(self, key)

    def get(self, key, default=None):
        _ensure_grade()
        return dict.get(self, key, default)

    def keys(self):
        _ensure_grade()
        return dict.keys(self)

    def items(self):
        _ensure_grade()
        return dict.items(self)

    def __iter__(self):
        _ensure_grade()
        return dict.__iter__(self)

    def __len__(self):
        _ensure_grade()
        return dict.__len__(self)


#: `type` の文字列からエフェクト関数を引く表。**名前は JS 版と同じ camelCase です。**
effects: dict[str, Callable[..., Bitmap]] = _Registry({
    # 基本
    "opacity": opacity,
    "blur": blur,
    "directionalBlur": directional_blur,
    "sharpen": sharpen,
    "colorAdjust": color_adjust,
    "tint": tint,
    "grayscale": grayscale,
    "invert": invert,
    "threshold": threshold,
    "pixelate": pixelate,
    "glow": glow,
    "dropShadow": drop_shadow,
    "stroke": stroke,
    "chromaKey": chroma_key,
    "vignette": vignette,
    "noise": noise,
    "bloom": bloom,
    "duotone": duotone,
    "posterize": posterize,
    "emboss": emboss,
    "edgeDetect": edge_detect,
    "mirror": mirror,
    "kaleidoscope": kaleidoscope,
    "scanlines": scanlines,
    "chromaticAberration": chromatic_aberration,
    "lensDistortion": lens_distortion,
    "roundCorners": round_corners,
    "feather": feather,
    "gradientMap": gradient_map,
    # MV の定番演出
    "radialBlur": radial_blur,
    "spinBlur": spin_blur,
    "glitch": glitch,
    "rasterScroll": raster_scroll,
    "diffusion": diffusion,
    "lightStreak": light_streak,
    "lensFlare": lens_flare,
    "rimLight": rim_light,
    "innerGlow": inner_glow,
    "halftone": halftone,
    "mangaize": mangaize,
    "polar": polar,
    "tile": tile,
    "peripheralBlur": peripheral_blur,
    "letterbox": letterbox,
    "gradientOverlay": gradient_overlay,
    "luminanceKey": luminance_key,
    "colorKey": color_key,
    "pixelSort": pixel_sort,
    "reflection": reflection,
    # 質感と色づくり
    "dither": dither,
    "misregistration": misregistration,
    "retroFilm": retro_film,
    "lightLeak": light_leak,
    "colorama": colorama,
    "leaveColor": leave_color,
    "monochrome": monochrome,
    "bevel": bevel,
    "directionalLight": directional_light,
    "longShadow": long_shadow,
    "graphicPen": graphic_pen,
    "hexTile": hex_tile,
    "slitScan": slit_scan,
    # かけら
    "shatter": shatter,
    "objectSplit": object_split,
    "slice": slice_effect,
})

_GRADE_LOADED = False


def _ensure_grade() -> None:
    """カラーグレーディング（curves / colorWheels / hslSecondary / lut）を登録する。

    初回だけ走ります。詳しい事情は {@link _Registry} を見てください。
    """
    global _GRADE_LOADED
    if _GRADE_LOADED:
        return
    _GRADE_LOADED = True
    from movo.renderer.effects_grade import grade_effects

    effects.update(grade_effects)


MASK_RESOLUTION = 96


def list_effects() -> list[str]:
    """使えるエフェクト名の一覧（`movo list effects` が出すもの）。"""
    _ensure_grade()
    return sorted(effects.keys())


def has_effect(name: str) -> bool:
    _ensure_grade()
    return name in effects


def apply_effect(bitmap: Bitmap, spec: dict, ctx: dict | None = None) -> Bitmap:
    """エフェクトを 1 つ当てる。`mask` があればその重みで混ぜ戻します。

    :param spec: `type` とパラメーターが入った辞書
    :param ctx: `seed` `time` `fps` `assets` `frameHistory` `plugins`
    """
    ctx = ctx or {}
    _ensure_grade()
    fn = effects.get(spec.get("type"))
    if fn is None:
        plugins = ctx.get("plugins") or {}
        factory = plugins.get("effect") if isinstance(plugins, dict) else None
        fn = factory(spec.get("type")) if callable(factory) else None
    if fn is None:
        return bitmap
    result = fn(bitmap, spec, ctx)
    if not spec.get("mask"):
        return result

    try:  # deformer は別担当が移植中。無ければマスク無しとして通します。
        from movo.deformer.mask import build_mask_field, sample_field
    except ImportError:
        return result
    field = build_mask_field(spec["mask"], MASK_RESOLUTION, MASK_RESOLUTION, {**ctx, "selfBitmap": bitmap})
    if field is None:
        return result

    h, w = bitmap.height, bitmap.width
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    weight = np.clip(
        sample_field(field, MASK_RESOLUTION, MASK_RESOLUTION, u[None, :], v[:, None]), 0, 1
    )
    out = bitmap.copy()
    base = bitmap.data.astype(np.float64)
    mixed = base + (result.data.astype(np.float64) - base) * weight[..., None]
    out.data[:] = np.where((weight > 0)[..., None], _u8(mixed), bitmap.data)
    return out


def debug_rect(bitmap: Bitmap, x, y, width, height, color="#ff00ff") -> Bitmap:
    """デバッグ用の枠。`--debug` の当たり判定表示に使います。"""
    contours = [
        [x, y, x + width, y, x + width, y + 1, x, y + 1],
        [x, y + height - 1, x + width, y + height - 1, x + width, y + height, x, y + height],
        [x, y, x + 1, y, x + 1, y + height, x, y + height],
        [x + width - 1, y, x + width, y, x + width, y + height, x + width - 1, y + height],
    ]
    fill_coverage(bitmap, rasterize_contours(contours, bitmap.width, bitmap.height), color, 1)
    return bitmap
