"""すべてのモジュールが使う小さな数学。

.. note::
   このファイルは ``movo/core/math.py`` ですが、標準ライブラリの ``math`` は
   問題なく import できます。Python 3 の import は既定で絶対 import なので、
   ``import math`` は常に標準ライブラリを指します。

**スカラ用**です。画素の配列に対しては使わないでください（Python の関数呼び出しが
1 画素ごとに入ると 1280x720 で 700 ミリ秒かかります）。配列には NumPy の
``np.clip`` などを直接当ててください。
"""

from __future__ import annotations

import math as _math
from typing import Sequence

TAU = _math.pi * 2
DEG = _math.pi / 180

#: 6 要素の 2D アフィン行列 ``[a, b, c, d, e, f]``。
Matrix = tuple[float, float, float, float, float, float]


def js_round(value: float) -> int:
    """JavaScript の ``Math.round`` と同じ丸め。

    **Python の組み込み ``round`` とは結果が違います。** Python は «偶数丸め»
    （``round(0.5) == 0``）ですが、JS は «常に +∞ 方向»（``Math.round(0.5) == 1``）です。
    色の変換やサンプル値の量子化でここがずれると、JS 版と 1 ずれた画素が出ます。
    """
    if value != value:  # NaN
        return 0
    return _math.floor(value + 0.5)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else (maximum if value > maximum else value)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, v: float) -> float:
    if a == b:
        return 0.0
    return (v - a) / (b - a)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp(inverse_lerp(edge0, edge1, x), 0, 1)
    return t * t * (3 - 2 * t)


def to_radians(deg: float) -> float:
    return deg * DEG


def to_degrees(rad: float) -> float:
    return rad / DEG


def approximately(a: float, b: float, epsilon: float = 1e-9) -> bool:
    return abs(a - b) <= epsilon


class Mat2D:
    """2D アフィン行列。``[a, c, e / b, d, f / 0, 0, 1]`` を ``[a, b, c, d, e, f]`` で持ちます。

    NumPy の 3x3 行列にしていないのは、**1 レイヤーごとに数回しか掛けないから**です。
    ここで ndarray を作ると確保のコストのほうが大きくなります（実測で 6 倍遅い）。
    画素に当てる段では、この 6 要素を Numba のカーネルへ数値のまま渡します。
    """

    @staticmethod
    def identity() -> Matrix:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    @staticmethod
    def multiply(m1: Sequence[float], m2: Sequence[float]) -> Matrix:
        return (
            m1[0] * m2[0] + m1[2] * m2[1],
            m1[1] * m2[0] + m1[3] * m2[1],
            m1[0] * m2[2] + m1[2] * m2[3],
            m1[1] * m2[2] + m1[3] * m2[3],
            m1[0] * m2[4] + m1[2] * m2[5] + m1[4],
            m1[1] * m2[4] + m1[3] * m2[5] + m1[5],
        )

    @staticmethod
    def translate(m: Sequence[float], x: float, y: float) -> Matrix:
        return Mat2D.multiply(m, (1, 0, 0, 1, x, y))

    @staticmethod
    def rotate(m: Sequence[float], radians: float) -> Matrix:
        c = _math.cos(radians)
        s = _math.sin(radians)
        return Mat2D.multiply(m, (c, s, -s, c, 0, 0))

    @staticmethod
    def scale(m: Sequence[float], sx: float, sy: float) -> Matrix:
        return Mat2D.multiply(m, (sx, 0, 0, sy, 0, 0))

    @staticmethod
    def skew(m: Sequence[float], kx: float, ky: float) -> Matrix:
        return Mat2D.multiply(m, (1, ky, kx, 1, 0, 0))

    @staticmethod
    def apply(m: Sequence[float], x: float, y: float) -> tuple[float, float]:
        return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])

    @staticmethod
    def invert(m: Sequence[float]) -> Matrix | None:
        """逆行列。**潰れている（行列式がほぼ 0）ときは None を返します。**

        例外にしないのは、``scaleX: 0`` のレイヤーが普通にあるからです。
        呼ぶ側は «描かない» と解釈すれば済みます。
        """
        det = m[0] * m[3] - m[1] * m[2]
        if abs(det) < 1e-12:
            return None
        inv = 1.0 / det
        return (
            m[3] * inv,
            -m[1] * inv,
            -m[2] * inv,
            m[0] * inv,
            (m[2] * m[5] - m[3] * m[4]) * inv,
            (m[1] * m[4] - m[0] * m[5]) * inv,
        )

    @staticmethod
    def from_transform(t: dict) -> Matrix:
        """解決済みの transform から行列を組む。

        **キー名は JSON のまま**（``scaleX`` / ``skewY``）です。プロジェクト JSON を
        JS 版とそのまま共用するので、ここで snake_case に直すわけにはいきません。
        """
        m = Mat2D.identity()
        m = Mat2D.translate(m, t.get("x", 0) or 0, t.get("y", 0) or 0)
        m = Mat2D.rotate(m, to_radians(t.get("rotation", 0) or 0))
        if t.get("skewX") or t.get("skewY"):
            m = Mat2D.skew(m, t.get("skewX", 0) or 0, t.get("skewY", 0) or 0)
        sx = t.get("scaleX", 1)
        sy = t.get("scaleY", 1)
        return Mat2D.scale(m, 1 if sx is None else sx, 1 if sy is None else sy)


def solve2x2(a: float, b: float, c: float, d: float, x: float, y: float) -> tuple[float, float] | None:
    """2x2 の連立一次方程式を解く。特異なら None。"""
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    return ((x * d - b * y) / det, (a * y - x * c) / det)


def catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def sample_polyline(points: Sequence[Sequence[float]], t: float) -> list[float]:
    """折れ線を Catmull-Rom で滑らかにたどる。``t`` は 0..1。"""
    if len(points) == 0:
        return [0.0, 0.0]
    if len(points) == 1:
        return [points[0][0], points[0][1]]
    clamped = clamp(t, 0, 1)
    scaled = clamped * (len(points) - 1)
    i = min(int(_math.floor(scaled)), len(points) - 2)
    local = scaled - i
    p0 = points[max(0, i - 1)]
    p1 = points[i]
    p2 = points[i + 1]
    p3 = points[min(len(points) - 1, i + 2)]
    return [
        catmull_rom(p0[0], p1[0], p2[0], p3[0], local),
        catmull_rom(p0[1], p1[1], p2[1], p3[1], local),
    ]
