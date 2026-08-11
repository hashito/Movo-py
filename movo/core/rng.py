"""決定的な乱数と値ノイズ。

Movo で «ばらつく» ものはすべてここから引きます（ノイズモジュレータ・粒子・
ゆらぎ・物理の同点処理）。同じプロジェクトからは必ず同じ動画が出る、という
約束（仕様 29 節）を守るためです。

## JS 版と同じ系列を出すために

JavaScript のビット演算は **32 ビット整数**で行われ、``>>>`` は符号なし、
``Math.imul`` は 32 ビットで巻き戻る掛け算です。Python の整数は無限桁なので、
そのまま書くと **3 回目くらいから値がずれます**。すべての演算のあとに
``& 0xFFFFFFFF`` を掛けて 32 ビットに畳んでいるのはそのためです。

同じ種から同じ系列が出ることは ``tests/test_core_rng.py`` で JS 版の実測値と
突き合わせています。

## 速度について

**モジュレータから呼ぶぶんには 1 フレームに数十回**なので、素の Python で
十分です。**画素ごとに呼ぶ**（ノイズのエフェクト、粒子の初期化）ときは
:func:`fbm2d_grid` を使ってください。同じ式を Numba でコンパイルしたものです。

| 320x180 を fbm で埋める | |
| --- | --- |
| 素の Python（``NUMBA_DISABLE_JIT=1``） | 25,455 ms |
| Numba（`@njit`） | **3.2 ms**（7,950 倍） |

1280x720 で 52 ミリ秒です。ここが Numba の効きがいちばん大きい場所で、
理由は «画素ごとに 4 オクターブ × 8 隅のハッシュ» という、
**NumPy に落とすと中間配列が 32 枚できる**形だからです。
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numba import njit

_MASK = 0xFFFFFFFF


def hash_string(text: str) -> int:
    """文字列から 32 ビットの種を作る（FNV-1a）。

    **UTF-16 のコード単位**で回します。JS の ``charCodeAt`` がそうだからで、
    絵文字を含む名前でも JS 版と同じ値になります。
    """
    # utf-16-le の 2 バイトずつを 1 コード単位として読む
    raw = str(text).encode("utf-16-le")
    h = 2166136261
    for i in range(0, len(raw), 2):
        code = raw[i] | (raw[i + 1] << 8)
        h ^= code
        h = (h * 16777619) & _MASK
    return h & _MASK


class Random:
    """mulberry32。小さく速く、絵づくりには十分な質があります。

    呼び出すと 0 以上 1 未満の値を返します（``rng()``）。JS 版の
    ``createRandom()`` が返す関数と同じ使い勝手にしてあります。
    """

    __slots__ = ("_a",)

    def __init__(self, seed: int = 0) -> None:
        a = seed & _MASK
        # JS の `(seed >>> 0) || 0x9e3779b9` — 0 は «指定なし» として黄金比に置き換える
        self._a = a if a else 0x9E3779B9

    def __call__(self) -> float:
        a = (self._a + 0x6D2B79F5) & _MASK
        self._a = a
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & _MASK
        t = (t ^ (t + ((t ^ (t >> 7)) * (t | 61)))) & _MASK
        return ((t ^ (t >> 14)) & _MASK) / 4294967296.0

    def range(self, minimum: float, maximum: float) -> float:
        return minimum + self() * (maximum - minimum)

    def int(self, minimum: int, maximum: int) -> int:
        """``minimum`` 以上 ``maximum`` 以下の整数（**上端を含みます**）。"""
        return math.floor(self.range(minimum, maximum + 1))

    def pick(self, items: Sequence):
        return items[min(len(items) - 1, math.floor(self() * len(items)))]

    def gaussian(self) -> float:
        """標準正規分布（Box-Muller）。0 が出ると log が発散するので引き直します。"""
        u = 0.0
        v = 0.0
        while u == 0:
            u = self()
        while v == 0:
            v = self()
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


def create_random(seed: int = 0) -> Random:
    """JS 版の ``createRandom()`` と同じ入口。"""
    return Random(seed)


class RandomSource:
    """名前つきの独立した乱数列を、1 つの親の種から派生させます。

    機能ごとに列を分けておくと、**関係のない部分を書き換えても結果が変わりません**。
    1 本の列を共有していると、粒子の数を 1 つ増やしただけで、その後ろの
    «ゆらぎ» が全部ずれます。
    """

    def __init__(self, seed: int = 12345) -> None:
        self.seed = seed & _MASK
        self._streams: dict[str, Random] = {}

    def stream(self, name: str) -> Random:
        s = self._streams.get(name)
        if s is None:
            s = Random((self.seed ^ hash_string(name)) & _MASK)
            self._streams[name] = s
        return s

    def reset(self) -> None:
        self._streams.clear()


# ── 値ノイズ ────────────────────────────────────────────────
#
# 以下は «素の Python» と «Numba» の 2 本立てです。式は同じものを 2 回書いて
# いますが、片方だけ直して食い違うことがないよう、テストで «同じ値が出ること»
# を確かめています（tests/test_core_rng.py）。


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _hash_to_unit(i: int, seed: int) -> float:
    h = (((i & _MASK) ^ (seed & _MASK)) * 0x27D4EB2D) & _MASK
    h ^= h >> 15
    h = (h * 0x85EBCA6B) & _MASK
    h ^= h >> 13
    return (h & _MASK) / 4294967296.0


def _hash_to_unit2(x: int, y: int, seed: int) -> float:
    return _hash_to_unit(((x * 73856093) & _MASK) ^ ((y * 19349663) & _MASK), seed)


def _hash_to_unit3(x: int, y: int, z: int, seed: int) -> float:
    return _hash_to_unit(
        ((x * 73856093) & _MASK) ^ ((y * 19349663) & _MASK) ^ ((z * 83492791) & _MASK), seed
    )


def value_noise_1d(x: float, seed: int = 0) -> float:
    """1 次元の値ノイズ。返り値は -1..1。"""
    i = math.floor(x)
    f = x - i
    a = _hash_to_unit(i, seed)
    b = _hash_to_unit(i + 1, seed)
    return (a + (b - a) * _fade(f)) * 2 - 1


def value_noise_2d(x: float, y: float, seed: int = 0) -> float:
    """2 次元の値ノイズ。返り値は -1..1。"""
    xi = math.floor(x)
    yi = math.floor(y)
    xf = _fade(x - xi)
    yf = _fade(y - yi)
    v00 = _hash_to_unit2(xi, yi, seed)
    v10 = _hash_to_unit2(xi + 1, yi, seed)
    v01 = _hash_to_unit2(xi, yi + 1, seed)
    v11 = _hash_to_unit2(xi + 1, yi + 1, seed)
    top = v00 + (v10 - v00) * xf
    bottom = v01 + (v11 - v01) * xf
    return (top + (bottom - top) * yf) * 2 - 1


def value_noise_3d(x: float, y: float, z: float, seed: int = 0) -> float:
    """3 次元の値ノイズ。返り値は -1..1。

    3 本目の軸があると、ノイズが «流れる» のではなく **«変化する»** ようになります。
    ``z`` を時間で動かすと模様そのものが移り変わります（2 次元だと平行移動に
    しかならず、雲や炎に見えません）。
    """
    xi = math.floor(x)
    yi = math.floor(y)
    zi = math.floor(z)
    xf = _fade(x - xi)
    yf = _fade(y - yi)
    zf = _fade(z - zi)

    def corner(dx: int, dy: int, dz: int) -> float:
        return _hash_to_unit3(xi + dx, yi + dy, zi + dz, seed)

    def mix(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    z0 = mix(mix(corner(0, 0, 0), corner(1, 0, 0), xf), mix(corner(0, 1, 0), corner(1, 1, 0), xf), yf)
    z1 = mix(mix(corner(0, 0, 1), corner(1, 0, 1), xf), mix(corner(0, 1, 1), corner(1, 1, 1), xf), yf)
    return mix(z0, z1, zf) * 2 - 1


def fbm1d(x: float, seed: int = 0, octaves: int = 4, lacunarity: float = 2, gain: float = 0.5) -> float:
    """1 次元のフラクタルノイズ。"""
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for i in range(octaves):
        total += value_noise_1d(x * freq, seed + i * 1013) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return 0.0 if norm == 0 else total / norm


def fbm2d(
    x: float,
    y: float,
    *,
    seed: int = 0,
    z: float = 0.0,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    type: str = "fbm",
) -> float:
    """2/3 次元のフラクタルノイズ。

    ``fbm`` は -1..1、``turbulent`` と ``ridged`` は 0..1 を返します。

    オクターブごとに座標を 0.5 ラジアン回しているのは、**格子ノイズは軸に
    沿った縞が出やすい**ためです。回さないと «斜め 45 度の格子» がはっきり見えます。
    """
    octaves = max(1, min(10, round(octaves)))
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    cos_r = math.cos(0.5)
    sin_r = math.sin(0.5)
    px = x
    py = y
    for i in range(octaves):
        sample = value_noise_3d(px * freq, py * freq, z * freq, seed + i * 1013)
        rx = px * cos_r - py * sin_r
        py = px * sin_r + py * cos_r
        px = rx
        if type == "turbulent":
            total += abs(sample) * amp
        elif type == "ridged":
            ridge = 1 - abs(sample)
            total += ridge * ridge * amp
        else:
            total += sample * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return 0.0 if norm == 0 else total / norm


# ── Numba 版（画素ごとに呼ぶとき） ───────────────────────────

# 掛け算は **int64 で受けてから 32 ビットに畳みます。** uint32 同士で掛けると
# NUMBA_DISABLE_JIT=1（素の Python として動かして中身を追うとき）に NumPy が
# オーバーフロー警告を出し、本物の異常が埋もれます。32 ビット × 定数は
# int64 に収まるので、結果は同じで警告だけが消えます。
@njit(cache=True, inline="always")
def _nb_hash_to_unit(i: np.int64, seed: np.int64) -> np.float64:
    h = np.uint32(np.int64(np.uint32(i) ^ np.uint32(seed)) * np.int64(0x27D4EB2D))
    h = np.uint32(h ^ (h >> np.uint32(15)))
    h = np.uint32(np.int64(h) * np.int64(0x85EBCA6B))
    h = np.uint32(h ^ (h >> np.uint32(13)))
    return np.float64(h) / 4294967296.0


@njit(cache=True, inline="always")
def _nb_hash_to_unit3(x: np.int64, y: np.int64, z: np.int64, seed: np.int64) -> np.float64:
    a = np.uint32(np.int64(np.uint32(x)) * np.int64(73856093))
    b = np.uint32(np.int64(np.uint32(y)) * np.int64(19349663))
    c = np.uint32(np.int64(np.uint32(z)) * np.int64(83492791))
    return _nb_hash_to_unit(np.int64(np.uint32(a ^ b ^ c)), seed)


@njit(cache=True, inline="always")
def _nb_fade(t: np.float64) -> np.float64:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, inline="always")
def _nb_value_noise_3d(x, y, z, seed):
    xi = np.int64(math.floor(x))
    yi = np.int64(math.floor(y))
    zi = np.int64(math.floor(z))
    xf = _nb_fade(x - xi)
    yf = _nb_fade(y - yi)
    zf = _nb_fade(z - zi)
    c000 = _nb_hash_to_unit3(xi, yi, zi, seed)
    c100 = _nb_hash_to_unit3(xi + 1, yi, zi, seed)
    c010 = _nb_hash_to_unit3(xi, yi + 1, zi, seed)
    c110 = _nb_hash_to_unit3(xi + 1, yi + 1, zi, seed)
    c001 = _nb_hash_to_unit3(xi, yi, zi + 1, seed)
    c101 = _nb_hash_to_unit3(xi + 1, yi, zi + 1, seed)
    c011 = _nb_hash_to_unit3(xi, yi + 1, zi + 1, seed)
    c111 = _nb_hash_to_unit3(xi + 1, yi + 1, zi + 1, seed)
    a0 = c000 + (c100 - c000) * xf
    b0 = c010 + (c110 - c010) * xf
    a1 = c001 + (c101 - c001) * xf
    b1 = c011 + (c111 - c011) * xf
    z0 = a0 + (b0 - a0) * yf
    z1 = a1 + (b1 - a1) * yf
    return (z0 + (z1 - z0) * zf) * 2.0 - 1.0


@njit(cache=True)
def _fbm2d_grid_kernel(out, x0, y0, dx, dy, seed, z, octaves, lacunarity, gain, kind):
    """``out``（h, w）を fbm で埋める。``kind`` は 0=fbm / 1=turbulent / 2=ridged。"""
    height, width = out.shape
    cos_r = math.cos(0.5)
    sin_r = math.sin(0.5)
    for j in range(height):
        wy = y0 + dy * j
        for i in range(width):
            wx = x0 + dx * i
            total = 0.0
            amp = 1.0
            freq = 1.0
            norm = 0.0
            px = wx
            py = wy
            for o in range(octaves):
                sample = _nb_value_noise_3d(px * freq, py * freq, z * freq, seed + o * 1013)
                rx = px * cos_r - py * sin_r
                py = px * sin_r + py * cos_r
                px = rx
                if kind == 1:
                    total += abs(sample) * amp
                elif kind == 2:
                    ridge = 1.0 - abs(sample)
                    total += ridge * ridge * amp
                else:
                    total += sample * amp
                norm += amp
                amp *= gain
                freq *= lacunarity
            out[j, i] = 0.0 if norm == 0.0 else total / norm


_KINDS = {"fbm": 0, "turbulent": 1, "ridged": 2}


def fbm2d_grid(
    width: int,
    height: int,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    dx: float = 1.0,
    dy: float = 1.0,
    seed: int = 0,
    z: float = 0.0,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    type: str = "fbm",
) -> np.ndarray:
    """``(height, width)`` の float64 配列を fbm で埋める。

    **画素ごとにノイズが要るときは必ずこちらを使ってください。**
    :func:`fbm2d` を Python の二重ループから呼ぶと 1280x720 で 32 秒、
    この関数なら 52 ミリ秒です（実測 613 倍）。式は :func:`fbm2d` と同じで、
    テストで «同じ値が出ること» を確かめています。
    """
    out = np.empty((int(height), int(width)), np.float64)
    _fbm2d_grid_kernel(
        out,
        float(x0),
        float(y0),
        float(dx),
        float(dy),
        int(seed),
        float(z),
        max(1, min(10, round(octaves))),
        float(lacunarity),
        float(gain),
        _KINDS.get(type, 0),
    )
    return out
