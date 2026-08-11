"""柔らかいものと粒（JS 版 packages/physics/src/soft.js の移植）。

`SoftChain` は «質点を距離拘束で繋いだ 1 本の紐» です。髪・ネクタイ・
スカートの裾はこれで足ります。描画側は出てきた点列に沿ってレイヤーの
メッシュを曲げます。

## Python 版で変えたところ（値は変えていません）

点も粒も **`float64` の 2 次元配列**で持ちます。

  - `SoftChain.points` … ``(N, 5)`` の x, y, 前の x, 前の y, 固定か
  - `ParticleSystem.particles` … ``(生きている数, 10)``

JS 版はどちらもオブジェクトの配列でした。粒 5,000 個の 1 ステップを実測すると

| | |
| --- | --- |
| 純 Python | 2.33 ms |
| **Numba** | **0.17 ms**（13 倍） |

です。1 フレーム 2.33ms は 5 分の MV で 21 秒。紐は 8 節・6 反復で
**0.0014 ms** なので、こちらは «無いのと同じ» 速さになりました。

**乱数だけは Python 側に残しています。** 粒の «生まれ方» は JS 版と同じ
mulberry32 の系列でなければならず、Numba で書き直すと系列が変わるからです。
生成は 1 ステップに数個なので、ここが遅くなることはありません。
"""

from __future__ import annotations

import math

import numpy as np

from . import _kernels as K
from ._compat import clamp, create_random

# points の列
_X, _Y, _PX, _PY, _PINNED = range(5)
# particles の列
_P_X, _P_Y, _P_VX, _P_VY, _P_LIFE, _P_MAXLIFE, _P_SIZE, _P_ROT, _P_SPIN, _P_SEED = range(10)
_P_COLUMNS = 10


class SoftChain:
    """紐。ヴァーレ積分と距離拘束の緩和で動きます。

    :param options: ``id`` / ``origin`` / ``length`` / ``segments`` /
        ``stiffness`` / ``damping`` / ``gravityScale`` / ``angle`` / ``iterations``
    """

    def __init__(self, **options) -> None:
        self.id = options.get("id", "softChain")
        self.segments = max(1, round(options.get("segments", 8)))
        self.length = float(options.get("length", 200))
        self.stiffness = clamp(float(options.get("stiffness", 0.7)), 0.0, 1.0)
        self.damping = clamp(float(options.get("damping", 0.15)), 0.0, 1.0)
        self.gravity_scale = float(options.get("gravityScale", 1.0))
        self.iterations = max(1, round(options.get("iterations", 6)))
        self.wind = [0.0, 0.0]
        origin = options.get("origin") or {}
        if isinstance(origin, dict):
            self.origin = [float(origin.get("x", 0.0)), float(origin.get("y", 0.0))]
        else:
            self.origin = [float(origin[0]), float(origin[1])]

        angle = math.radians(float(options.get("angle", 90)))
        segment_length = self.length / self.segments
        self.segment_length = segment_length
        self.points = np.zeros((self.segments + 1, 5), np.float64)
        for i in range(self.segments + 1):
            x = self.origin[0] + math.cos(angle) * segment_length * i
            y = self.origin[1] + math.sin(angle) * segment_length * i
            self.points[i, _X] = x
            self.points[i, _Y] = y
            self.points[i, _PX] = x
            self.points[i, _PY] = y
            self.points[i, _PINNED] = 1.0 if i == 0 else 0.0

    def set_origin(self, x: float, y: float) -> None:
        """付け根を動かす。残りは拘束を通じて付いてきます。"""
        self.origin[0] = float(x)
        self.origin[1] = float(y)
        self.points[0, _X] = float(x)
        self.points[0, _Y] = float(y)

    def set_wind(self, x: float, y: float) -> None:
        self.wind[0] = float(x)
        self.wind[1] = float(y)

    def step(self, h: float, world=None) -> None:
        gravity = world.gravity if world is not None else (0.0, 980.0)
        gx = gravity[0] * self.gravity_scale + self.wind[0]
        gy = gravity[1] * self.gravity_scale + self.wind[1]
        K.soft_chain_step(
            self.points, self.segments, h, gx, gy, self.damping, self.stiffness,
            self.segment_length, self.iterations, self.origin[0], self.origin[1],
        )

    def to_normalized_path(self, width: float, height: float) -> list[list[float]]:
        """`pathDeform` に渡す 0〜1 の折れ線。"""
        w = width or 1
        h = height or 1
        return [
            [(float(p[_X]) - self.origin[0]) / w + 0.5, (float(p[_Y]) - self.origin[1]) / h]
            for p in self.points
        ]


class ParticleSystem:
    """粒を撒く。

    :param options: JS 版 `new ParticleSystem({...})` と同じキー
    """

    def __init__(self, **options) -> None:
        self.id = options.get("id", "particles")
        self.max_particles = min(20000, max(1, round(options.get("maxParticles", 400))))
        self.rate = float(options.get("rate", 60))
        self.lifetime = float(options.get("lifetime", 2))
        self.lifetime_variance = float(options.get("lifetimeVariance", 0.3))
        self.gravity_scale = float(options.get("gravityScale", 1.0))
        self.drag = float(options.get("drag", 0.0))
        self.emitter = {
            "x": float(options.get("x", 0.0)),
            "y": float(options.get("y", 0.0)),
            "width": float(options.get("width", 0.0)),
            "height": float(options.get("height", 0.0)),
        }
        self.speed = float(options.get("speed", 200))
        self.speed_variance = float(options.get("speedVariance", 0.4))
        self.direction = float(options.get("direction", -90))
        self.spread = float(options.get("spread", 30))
        self.size = float(options.get("size", 8))
        self.size_variance = float(options.get("sizeVariance", 0.4))
        self.size_over_life = float(options.get("sizeOverLife", 1))
        self.spin = float(options.get("spin", 0))
        self.color = options.get("color", "#ffffff")
        self.end_color = options.get("endColor")
        self.fade_in = float(options.get("fadeIn", 0.05))
        self.fade_out = float(options.get("fadeOut", 0.4))
        self.bounce = float(options.get("bounce", 0.0))
        floor_y = options.get("floorY")
        self.floor_y = None if floor_y is None else float(floor_y)
        # 事前に進めておく秒数。雪や星のように «最初から画面に散っている» 演出は
        # これを指定しないと 0 秒時点が空っぽになります。
        self.prewarm = max(0.0, min(30.0, float(options.get("prewarm", 0))))
        self._seed = int(options.get("seed", 12345))
        self.random = create_random(self._seed)
        self._data = np.zeros((self.max_particles, _P_COLUMNS), np.float64)
        self._count = 0
        self._accumulator = 0.0
        self.time = 0.0

    @property
    def particles(self) -> np.ndarray:
        """生きている粒だけの眺め（``(数, 10)``）。列は `_P_*` を参照。"""
        return self._data[: self._count]

    def reset(self) -> None:
        """最初の状態に戻す。**乱数も同じ種から引き直します**（巻き戻しても同じ絵）。"""
        self._count = 0
        self._accumulator = 0.0
        self.time = 0.0
        self.random = create_random(self._seed)

    def warmup(self, world=None, step: float = 1.0 / 30.0) -> None:
        """`prewarm` 秒ぶん空回ししてから本番に入る。

        巻き戻し時の `reset()` のあとにも同じ回数だけ回すので、結果は決定的です。
        """
        if self.prewarm <= 0:
            return
        steps = round(self.prewarm / step)
        for _ in range(steps):
            self.step(step, world)
        self.time = 0.0

    def _spawn(self) -> None:
        """粒を 1 つ生む。**乱数を引く順番は JS 版と同じでなければなりません。**"""
        if self._count >= self.max_particles:
            return
        r = self.random
        angle = math.radians(self.direction + (r() - 0.5) * self.spread)
        speed = self.speed * (1 + (r() - 0.5) * 2 * self.speed_variance)
        row = self._data[self._count]
        row[_P_X] = self.emitter["x"] + (r() - 0.5) * self.emitter["width"]
        row[_P_Y] = self.emitter["y"] + (r() - 0.5) * self.emitter["height"]
        row[_P_VX] = math.cos(angle) * speed
        row[_P_VY] = math.sin(angle) * speed
        row[_P_LIFE] = 0.0
        row[_P_MAXLIFE] = max(0.05, self.lifetime * (1 + (r() - 0.5) * 2 * self.lifetime_variance))
        row[_P_SIZE] = max(0.5, self.size * (1 + (r() - 0.5) * 2 * self.size_variance))
        row[_P_ROT] = r() * 360
        row[_P_SPIN] = (r() - 0.5) * 2 * self.spin
        row[_P_SEED] = r()
        self._count += 1

    def step(self, h: float, world=None) -> None:
        gravity = world.gravity if world is not None else (0.0, 980.0)
        gx = gravity[0] * self.gravity_scale
        gy = gravity[1] * self.gravity_scale
        self._accumulator += self.rate * h
        while self._accumulator >= 1:
            self._spawn()
            self._accumulator -= 1
        self._count = K.particles_step(
            self._data, self._count, h, gx, gy, self.drag,
            0.0 if self.floor_y is None else self.floor_y,
            0 if self.floor_y is None else 1,
            self.bounce,
        )
        self.time += h

    def render(self) -> dict:
        """描画側へ渡す «まとめて描ける» 形。

        JS 版は粒ごとの辞書を並べて返していました。Python では 1 万個ぶんの
        辞書を毎フレーム作ると、それだけで描画より時間がかかります。
        **列ごとの配列**で返し、描画側が一括で扱えるようにしています。
        JS 版と同じ «辞書の並び» が欲しいときは `render_list()` を使ってください。
        """
        n = self._count
        data = self._data[:n]
        t = np.divide(data[:, _P_LIFE], data[:, _P_MAXLIFE], out=np.zeros(n), where=data[:, _P_MAXLIFE] > 0)
        opacity = np.ones(n)
        if self.fade_in > 0:
            rising = t < self.fade_in
            opacity[rising] = t[rising] / self.fade_in
        if self.fade_out > 0:
            falling = t > 1 - self.fade_out
            opacity[falling] = np.minimum(opacity[falling], (1 - t[falling]) / self.fade_out)
        return {
            "x": data[:, _P_X].copy(),
            "y": data[:, _P_Y].copy(),
            "size": data[:, _P_SIZE] * (1 + (self.size_over_life - 1) * t),
            "rotation": data[:, _P_ROT].copy(),
            "opacity": np.clip(opacity, 0.0, 1.0),
            "progress": t,
            # 進行方向へ向ける／伸ばす（alignToVelocity / stretch）ために使います
            "vx": data[:, _P_VX].copy(),
            "vy": data[:, _P_VY].copy(),
            "seed": data[:, _P_SEED].copy(),
        }

    def render_list(self) -> list[dict]:
        """JS 版と同じ «粒ごとの辞書» の並び。少数のときや検証用。"""
        columns = self.render()
        return [
            {key: float(values[i]) for key, values in columns.items()}
            for i in range(self._count)
        ]
