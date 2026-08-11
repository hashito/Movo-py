"""剛体の状態を «1 枚の配列» で持つための置き場。

## なぜこうしたか

JS 版の `Body` は `{position: {x, y}, velocity: {x, y}, ...}` という
オブジェクトでした。Python でそのまま書くと、1 ステップの積分だけで
剛体の数 × 20 回の属性アクセスが起き、**Numba へ渡す前に配列へ詰め直す
手間のほうが計算より高くつきます**。

そこで剛体の値は最初から `(剛体数, NS)` の `float64` 配列に置き、
`Body.position.x` は **その配列を覗く窓**にしました。

  - Numba のカーネルには配列をそのまま渡せる（詰め直しゼロ）
  - `body.position.x` という JS 版と同じ書き味は残る

配列は `World` が持ち、剛体が増えると倍々で伸ばします。伸びたときに
`Body` が古い配列を掴んだままにならないよう、間に `_Bank` を挟んで
**参照を 1 か所にまとめて**あります。
"""

from __future__ import annotations

import numpy as np

# float64 の列。ここを増やすときは NS も直すこと。
PX, PY, VX, VY, ANG, AV, FX, FY, TQ, INV_M, INV_I, FRIC, REST, LDAMP, ADAMP, GSCALE, RADIUS, MASS, INERTIA = range(19)
NS = 19

# int64 の列。
BTYPE, FIXROT, SENSOR, CGROUP, CMASK, STYPE, VSTART, VCOUNT = range(8)
NI = 8

# BTYPE の値
STATIC, DYNAMIC, KINEMATIC = 0, 1, 2
# STYPE の値
CIRCLE, POLYGON = 0, 1

_TYPE_NAMES = {STATIC: "static", DYNAMIC: "dynamic", KINEMATIC: "kinematic"}
_TYPE_CODES = {"static": STATIC, "dynamic": DYNAMIC, "kinematic": KINEMATIC}


class Bank:
    """剛体の値をまとめて持つ入れ物。`Body` はここを覗きます。"""

    __slots__ = ("S", "I", "count")

    def __init__(self, capacity: int = 1) -> None:
        self.S = np.zeros((capacity, NS), np.float64)
        self.I = np.zeros((capacity, NI), np.int64)
        self.count = 0

    def allocate(self) -> int:
        """1 行ぶん確保して行番号を返す。足りなければ倍に伸ばします。"""
        if self.count >= self.S.shape[0]:
            capacity = max(4, self.S.shape[0] * 2)
            grown_s = np.zeros((capacity, NS), np.float64)
            grown_i = np.zeros((capacity, NI), np.int64)
            grown_s[: self.count] = self.S[: self.count]
            grown_i[: self.count] = self.I[: self.count]
            self.S = grown_s
            self.I = grown_i
        row = self.count
        self.count += 1
        return row


class Vec2View:
    """`body.position.x` と書けるようにするための «配列を覗く窓»。"""

    __slots__ = ("_body", "_cx", "_cy")

    def __init__(self, body, cx: int, cy: int) -> None:
        self._body = body
        self._cx = cx
        self._cy = cy

    @property
    def x(self) -> float:
        return float(self._body._bank.S[self._body._row, self._cx])

    @x.setter
    def x(self, value: float) -> None:
        self._body._bank.S[self._body._row, self._cx] = value

    @property
    def y(self) -> float:
        return float(self._body._bank.S[self._body._row, self._cy])

    @y.setter
    def y(self, value: float) -> None:
        self._body._bank.S[self._body._row, self._cy] = value

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __eq__(self, other) -> bool:
        if isinstance(other, Vec2View):
            return self.as_tuple() == other.as_tuple()
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return self.as_tuple() == (other[0], other[1])
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"({self.x}, {self.y})"
