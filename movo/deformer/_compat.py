"""変形が使う «配列版» の道具。

`movo.core` にも同じ名前の関数があります（`fbm2d` `value_noise_1d`
`sample_polyline` `clamp`）。**あちらは 1 点ずつ、ここは «頂点ぶんまとめて»**
という違いだけで、値は同じです（`tests/test_parity_deformer.py` で JS 版と
突き合わせています）。

変形は 1 レイヤーで数千頂点を一度に動かすので、点ごとに呼ぶと Python の
関数呼び出しだけで時間が溶けます。core 側は «式の中から 1 点だけ引く» 用途
なので、両方あるのが正しい形です。core の `fbm2d_grid` は «等間隔の格子»
専用で、メッシュの頂点（不等間隔）には使えません。

## 乱数を NumPy に載せるときの注意

JS の `Math.imul` は «下位 32 ビットの掛け算» です。XOR も加算も下位 32 ビットは
合同なので、**`uint32` の配列のまま計算すれば一致します**。符号付きに直す
必要はありません（直すとかえって桁が化けます）。

`Math.floor` は負の数を «下» へ丸めます。NumPy の `np.floor` も同じなので
そのまま使えます（`int()` や `//` に置き換えると負の座標でずれます）。
"""

from __future__ import annotations

import math

import numpy as np

from movo.core.math import TAU, js_round  # noqa: F401  （js_round は JS の Math.round と同じ丸め）


def clamp(value, low, high):
    """`low` と `high` の間に丸める。

    **`movo.core.math.clamp` をそのまま使わないのはわざとです。** あちらは
    スカラー用（`value < minimum` の比較）で、変形は «頂点ぶんの配列» を
    そのまま通します。値はどちらも同じで、配列を受けられるかだけが違います。
    """
    if isinstance(value, np.ndarray):
        return np.clip(value, low, high)
    return low if value < low else (high if value > high else value)


def _u32(values) -> np.ndarray:
    """整数（負でも可）を uint32 の配列にする。JS の ToUint32 と同じ。"""
    return np.asarray(values, np.int64).astype(np.uint32)


def _imul(a: np.ndarray, b) -> np.ndarray:
    """`Math.imul` の配列版。

    **いったん uint64 に上げてから下位 32 ビットを取ります。** uint32 のまま
    掛けても答えは同じ（巻き戻りが定義されている）のですが、NumPy が
    «overflow encountered» と警告を出し、本当に困る桁あふれが埋もれます。
    """
    return ((a.astype(np.uint64) * np.uint64(b & 0xFFFFFFFF)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _hash_to_unit(i, seed) -> np.ndarray:
    """整数から 0〜1 の値を作る（JS 版 `hashToUnit`）。"""
    h = _imul(_u32(i) ^ _u32(seed), 0x27D4EB2D)
    h = h ^ (h >> np.uint32(15))
    h = _imul(h, 0x85EBCA6B)
    h = h ^ (h >> np.uint32(13))
    return h.astype(np.float64) / 4294967296.0


def _hash_to_unit3(x, y, z, seed) -> np.ndarray:
    return _hash_to_unit(
        _imul(_u32(x), 73856093) ^ _imul(_u32(y), 19349663) ^ _imul(_u32(z), 83492791),
        seed,
    )


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def value_noise_1d(x, seed: int = 0):
    """1 次元の値ノイズ（−1〜1）。配列でもスカラーでも通ります。"""
    x = np.asarray(x, np.float64)
    i = np.floor(x)
    f = x - i
    a = _hash_to_unit(i, seed)
    b = _hash_to_unit(i + 1, seed)
    out = (a + (b - a) * _fade(f)) * 2 - 1
    return out


def value_noise_3d(x, y, z, seed: int = 0):
    """3 次元の値ノイズ（−1〜1）。

    第 3 軸があると、模様が «流れる» のではなく **形が変わって**いきます。
    `evolution` を進めたときの見え方がここで決まります。
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    xi = np.floor(x)
    yi = np.floor(y)
    zi = np.floor(z)
    xf = _fade(x - xi)
    yf = _fade(y - yi)
    zf = _fade(z - zi)

    def corner(dx, dy, dz):
        return _hash_to_unit3(xi + dx, yi + dy, zi + dz, seed)

    def lerp(a, b, t):
        return a + (b - a) * t

    z0 = lerp(lerp(corner(0, 0, 0), corner(1, 0, 0), xf), lerp(corner(0, 1, 0), corner(1, 1, 0), xf), yf)
    z1 = lerp(lerp(corner(0, 0, 1), corner(1, 0, 1), xf), lerp(corner(0, 1, 1), corner(1, 1, 1), xf), yf)
    return lerp(z0, z1, zf) * 2 - 1


def fbm2d(x, y, options: dict | None = None):
    """フラクタルノイズ。`fbm` は −1〜1、`turbulent` と `ridged` は 0〜1。

    :param options: ``seed`` / ``z`` / ``octaves`` / ``lacunarity`` / ``gain`` / ``type``
    """
    options = options or {}
    seed = int(options.get("seed") or 0)
    z = float(options.get("z") or 0)
    octaves = max(1, min(10, round(options.get("octaves", 4))))
    lacunarity = float(options.get("lacunarity", 2))
    gain = float(options.get("gain", 0.5))
    kind = options.get("type") or "fbm"

    px = np.asarray(x, np.float64)
    py = np.asarray(y, np.float64)
    total = np.zeros(np.broadcast(px, py).shape, np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    # 格子ノイズは軸に沿った模様が出やすいので、オクターブごとに座標を少し回します。
    cos = math.cos(0.5)
    sin = math.sin(0.5)
    for i in range(octaves):
        sample = value_noise_3d(px * freq, py * freq, z * freq, seed + i * 1013)
        rx = px * cos - py * sin
        py = px * sin + py * cos
        px = rx
        if kind == "turbulent":
            total = total + np.abs(sample) * amp
        elif kind == "ridged":
            ridge = 1 - np.abs(sample)
            total = total + ridge * ridge * amp
        else:
            total = total + sample * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    if norm == 0:
        return np.zeros_like(total)
    return total / norm


def _catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def sample_polyline(points: np.ndarray, t):
    """折れ線を Catmull-Rom で読む（JS 版 `samplePolyline` の配列版）。

    :param points: ``(N, 2)``
    :param t: 0〜1。配列で渡すと全部まとめて返します
    """
    points = np.asarray(points, np.float64)
    n = len(points)
    t = np.asarray(t, np.float64)
    if n == 0:
        return np.zeros(t.shape + (2,), np.float64)
    if n == 1:
        return np.broadcast_to(points[0], t.shape + (2,)).copy()
    clamped = np.clip(t, 0.0, 1.0)
    scaled = clamped * (n - 1)
    i = np.minimum(np.floor(scaled).astype(np.int64), n - 2)
    local = scaled - i
    p0 = points[np.maximum(0, i - 1)]
    p1 = points[i]
    p2 = points[i + 1]
    p3 = points[np.minimum(n - 1, i + 2)]
    local2 = local[..., None]
    return _catmull_rom(p0, p1, p2, p3, local2)


def warn(message: str) -> None:
    """core の logger があればそちらへ、無ければ標準エラーへ。"""
    try:  # pragma: no cover
        from movo.core.logger import logger  # type: ignore

        logger.warn(message)
    except Exception:  # pragma: no cover
        import sys

        print(f"⚠ {message}", file=sys.stderr)


__all__ = ["TAU", "clamp", "fbm2d", "value_noise_1d", "value_noise_3d", "sample_polyline", "warn"]
