"""パーティクル（粒）の生成・進行・描画。

雨・雪・桜吹雪・紙吹雪・泡・火花・煙といった «舞い物» を作ります。既定値は
`particle_presets.py` にまとめてあり、`emitter` に書いた値が上書きします。

## 速度の設計

粒の状態は **«配列の束»（structure of arrays）** で持ちます。JS 版はオブジェクトの
配列でしたが、Python でそれをやると 1 万粒 × 4500 フレームで手が付けられません。

| 1 万粒を 1 ステップ進める | |
| --- | --- |
| 粒ごとの Python ループ | 約 12 ms |
| **NumPy の一括演算** | **約 0.15 ms** |

一方 **«生む» ところは 1 フレームに数十個**しかないので、素直な Python の
ループのままです。ここを配列にしても速くならず、乱数を引く順が崩れて
JS 版と粒の位置が変わるほうが損です。

描画は Numba の走査線に 1 回だけ降ります。粒ごとに全画面の被覆率バッファを
確保すると 900 粒で 3 GB 触ることになるので、**粒の囲む矩形の中だけ**を
塗るカーネルにしてあります。

## 決定性

乱数は mulberry32（`movo.renderer.effects.Random`）で、**JS 版と同じ数列**です。
生む順・引く順まで合わせてあるので、同じ JSON からは粒 1 個までは同じ絵に
なります。巻き戻したら `reset()` してから `warmup()` を同じ回数まわします。
"""

from __future__ import annotations

import math

import numpy as np

from movo.core.bitmap import Bitmap
from movo.renderer.effects import (
    Random,
    _jround,
    circle_contour,
    clamp,
    draw_textured_triangle,
    hash_string,
    njit,
    parse_color,
)
from movo.renderer.particle_presets import resolve_preset

#: 1 つのエミッターが持てる粒の上限。ここを外すと «rate の書き間違い» で
#: メモリを食い潰せてしまいます。
MAX_PARTICLES = 20000

# 粒 1 個ぶんの «列» の名前。**`p_` を付けてあるのは、エミッターの設定
# （`self.size` `self.spin` など同名のスカラー）と衝突させないためです。**
_FIELDS = ("p_x", "p_y", "p_vx", "p_vy", "p_life", "p_max_life", "p_size", "p_rotation", "p_spin", "p_seed")


class ParticleSystem:
    """粒の集合。`step()` で進め、`render()` で «描くための一覧» を出します。

    :param options: エミッターの設定（プリセット + `emitter` の上書き）
    """

    def __init__(self, options: dict | None = None):
        options = options or {}
        self.id = options.get("id", "particles")
        self.max_particles = int(min(MAX_PARTICLES, max(1, _jround(options.get("maxParticles", 400) or 400))))
        self.rate = _opt(options, "rate", 60)
        self.lifetime = _opt(options, "lifetime", 2)
        self.lifetime_variance = _opt(options, "lifetimeVariance", 0.3)
        self.gravity_scale = _opt(options, "gravityScale", 1)
        self.drag = _opt(options, "drag", 0)
        self.emitter = {
            "x": _opt(options, "x", 0),
            "y": _opt(options, "y", 0),
            "width": _opt(options, "width", 0),
            "height": _opt(options, "height", 0),
        }
        self.speed = _opt(options, "speed", 200)
        self.speed_variance = _opt(options, "speedVariance", 0.4)
        self.direction = _opt(options, "direction", -90)
        self.spread = _opt(options, "spread", 30)
        self.size = _opt(options, "size", 8)
        self.size_variance = _opt(options, "sizeVariance", 0.4)
        self.size_over_life = _opt(options, "sizeOverLife", 1)
        self.spin = _opt(options, "spin", 0)
        self.color = _opt(options, "color", "#ffffff")
        self.end_color = options.get("endColor")
        self.fade_in = _opt(options, "fadeIn", 0.05)
        self.fade_out = _opt(options, "fadeOut", 0.4)
        self.bounce = _opt(options, "bounce", 0)
        self.floor_y = options.get("floorY")
        # 事前に進めておく秒数。雪や星のように «最初から画面に散っている» 演出は
        # これを指定しないと 0 秒時点が空っぽになります。
        self.prewarm = max(0.0, min(30.0, _opt(options, "prewarm", 0)))
        self._seed = int(options.get("seed", 12345) or 0)
        self.random = Random(self._seed)
        self.time = 0.0
        self._accumulator = 0.0
        self._alloc = 0
        self.count = 0
        for name in _FIELDS:
            setattr(self, name, np.zeros(0, np.float64))

    # ── 生成と巻き戻し ────────────────────────────────────────

    def reset(self) -> None:
        """粒を全部捨てて、乱数も種からやり直す（巻き戻しのため）。"""
        self.count = 0
        self._accumulator = 0.0
        self.time = 0.0
        self.random = Random(self._seed)
        for name in _FIELDS:
            setattr(self, name, np.zeros(0, np.float64))
        self._alloc = 0

    def warmup(self, world=None, step: float = 1 / 30) -> None:
        """`prewarm` 秒ぶん空回ししてから本番に入る。

        **`reset()` のあとにも同じ回数まわすので、結果は決定的です。**
        """
        if self.prewarm <= 0:
            return
        for _ in range(int(_jround(self.prewarm / step))):
            self.step(step, world)
        self.time = 0.0

    def _grow(self, extra: int) -> None:
        """配列を広げる。倍々にしておくと、毎フレームの確保がほぼ起きません。"""
        need = self.count + extra
        if need <= self._alloc:
            return
        alloc = max(16, self._alloc)
        while alloc < need:
            alloc *= 2
        for name in _FIELDS:
            old = getattr(self, name)
            grown = np.zeros(alloc, np.float64)
            grown[: self.count] = old[: self.count]
            setattr(self, name, grown)
        self._alloc = alloc

    def _spawn_batch(self, how_many: int) -> None:
        """粒を生む。**乱数の «引く順» は JS 版と 1 個ずつ同じです。**

        上限に達したら «乱数を引かずに» 打ち切ります。ここで先に引いてしまうと、
        上限に当たったときだけ以降の数列がずれます。
        """
        if how_many <= 0:
            return
        rows = []
        r = self.random
        for _ in range(how_many):
            if self.count + len(rows) >= self.max_particles:
                # **`return` ではなく `break`。** ここで抜けてしまうと、直前まで
                # 作った粒がまるごと捨てられます（上限に当たった 1 フレームだけ
                # 粒が 1 個足りない、という形で出ました）。
                break
            angle = math.radians(self.direction + (r() - 0.5) * self.spread)
            speed = self.speed * (1 + (r() - 0.5) * 2 * self.speed_variance)
            rows.append(
                (
                    self.emitter["x"] + (r() - 0.5) * self.emitter["width"],
                    self.emitter["y"] + (r() - 0.5) * self.emitter["height"],
                    math.cos(angle) * speed,
                    math.sin(angle) * speed,
                    0.0,
                    max(0.05, self.lifetime * (1 + (r() - 0.5) * 2 * self.lifetime_variance)),
                    max(0.5, self.size * (1 + (r() - 0.5) * 2 * self.size_variance)),
                    r() * 360,
                    (r() - 0.5) * 2 * self.spin,
                    r(),
                )
            )
        if not rows:
            return
        self._grow(len(rows))
        block = np.array(rows, np.float64)
        for i, name in enumerate(_FIELDS):
            getattr(self, name)[self.count : self.count + len(rows)] = block[:, i]
        self.count += len(rows)

    # ── 進行 ──────────────────────────────────────────────────

    def step(self, h: float, world=None) -> None:
        """1 ステップ進める。**ここが NumPy の一括演算です。**"""
        gravity = _gravity_of(world)
        gx = gravity[0] * self.gravity_scale
        gy = gravity[1] * self.gravity_scale

        self._accumulator += self.rate * h
        spawns = 0
        while self._accumulator >= 1:
            spawns += 1
            self._accumulator -= 1
        self._spawn_batch(spawns)

        n = self.count
        if n:
            life = self.p_life[:n]
            life += h
            alive = life < self.p_max_life[:n]
            if not alive.all():
                # 生き残りだけを前へ詰める。**並び順は変えません**（JS の splice と
                # 同じ相対順序を保たないと、色の割り当てが変わります）。
                keep = np.flatnonzero(alive)
                for name in _FIELDS:
                    arr = getattr(self, name)
                    arr[: keep.size] = arr[:n][keep]
                self.count = int(keep.size)
                n = self.count

        if n:
            drag_factor = 1 / (1 + self.drag * h)
            vx = self.p_vx[:n]
            vy = self.p_vy[:n]
            vx[:] = (vx + gx * h) * drag_factor
            vy[:] = (vy + gy * h) * drag_factor
            self.p_x[:n] += vx * h
            self.p_y[:n] += vy * h
            self.p_rotation[:n] += self.p_spin[:n] * h
            if self.floor_y is not None:
                y = self.p_y[:n]
                below = y > self.floor_y
                if below.any():
                    y[below] = self.floor_y
                    vy[below] = -vy[below] * self.bounce
                    vx[below] *= 0.9
        self.time += h

    # ── 描画のための一覧 ──────────────────────────────────────

    def render(self) -> dict:
        """位置・大きさ・回転・不透明度の «束»。描画側はこれだけ見ます。"""
        n = self.count
        t = np.divide(self.p_life[:n], self.p_max_life[:n], out=np.zeros(n), where=self.p_max_life[:n] != 0)
        opacity = np.ones(n)
        if self.fade_in > 0:
            opacity = np.where(t < self.fade_in, t / self.fade_in, opacity)
        if self.fade_out > 0:
            opacity = np.minimum(opacity, np.where(t > 1 - self.fade_out, (1 - t) / self.fade_out, opacity))
        return {
            "x": self.p_x[:n].copy(),
            "y": self.p_y[:n].copy(),
            "size": self.p_size[:n] * (1 + (self.size_over_life - 1) * t),
            "rotation": self.p_rotation[:n].copy(),
            "opacity": np.clip(opacity, 0, 1),
            "progress": t,
            # 進行方向へ向ける／伸ばす（alignToVelocity / stretch）ために使う
            "vx": self.p_vx[:n].copy(),
            "vy": self.p_vy[:n].copy(),
            "seed": self.p_seed[:n].copy(),
            "count": n,
        }


def _opt(options: dict, name: str, fallback):
    value = options.get(name)
    return fallback if value is None else value


def _gravity_of(world):
    """物理ワールドから重力を取り出す。無ければ «下向き 980»（JS 版の既定）。"""
    if world is None:
        return (0.0, 980.0)
    gravity = getattr(world, "gravity", None)
    if gravity is None and isinstance(world, dict):
        gravity = world.get("gravity")
    if gravity is None:
        return (0.0, 980.0)
    if isinstance(gravity, dict):
        return (gravity.get("x", 0) or 0, gravity.get("y", 980) if gravity.get("y") is not None else 980)
    return (getattr(gravity, "x", 0), getattr(gravity, "y", 980))


def create_particle_system(emitter: dict, width: float, height: float, seed: int = 12345,
                           layer_id: str = "particles") -> ParticleSystem:
    """`emitter` の記述からシステムを組む。

    プリセットの寸法・速度は **1080p 基準**で書いてあるので、ここで解像度に
    合わせて掛け直します。利用者が `emitter` に直接書いた値はそのプロジェクトの
    座標系なので触りません。
    """
    emitter = emitter or {}
    defaults: dict = {}
    name = emitter.get("preset")
    if name:
        resolved = resolve_preset(name, width, height)
        if resolved is None:
            raise ValueError(f'知らないパーティクルプリセット "{name}" です')
        defaults = resolved
        scale = height / 1080
        if scale != 1:
            if isinstance(defaults.get("size"), (int, float)):
                defaults["size"] = max(0.5, defaults["size"] * scale)
            if isinstance(defaults.get("speed"), (int, float)):
                defaults["speed"] *= scale
        # 0 秒時点で画面が空っぽにならないよう、寿命ぶんだけ空回ししておく
        defaults.setdefault("prewarm", (defaults.get("lifetime", 2)) * 0.9)
    options = {
        "id": layer_id,
        "seed": (int(seed) ^ hash_string(layer_id)) & 0xFFFFFFFF,
        **defaults,
        **emitter,
    }
    return ParticleSystem(options)


# ── 粒 1 個の形 ──────────────────────────────────────────────────

def particle_contour(x: float, y: float, radius: float, rotation: float, vx: float, vy: float,
                     seed: float, emitter: dict) -> list[float]:
    """粒 1 個の輪郭を作る。

    `shape` で形を選び、`alignToVelocity` で進行方向へ向け、`stretch` で進行
    方向へ伸ばします。**雨や火花は «速度方向に伸びた線» にすると一気にそれ
    らしくなります。**
    """
    shape = emitter.get("shape", "circle")
    stretch = emitter.get("stretch", 1) if emitter.get("stretch") is not None else 1
    align = emitter.get("alignToVelocity") is True
    speed = math.hypot(vx, vy)
    angle = math.atan2(vy, vx) if (align and speed > 1e-3) else math.radians(rotation)
    cos = math.cos(angle)
    sin = math.sin(angle)

    def place(lx, ly):
        return (x + lx * cos - ly * sin, y + lx * sin + ly * cos)

    long = radius * stretch

    if shape == "square":
        pts = [place(-long, -radius), place(long, -radius), place(long, radius), place(-long, radius)]
        return [c for p in pts for c in p]
    if shape == "triangle":
        pts = [place(long, 0), place(-long * 0.7, -radius), place(-long * 0.7, radius)]
        return [c for p in pts for c in p]
    if shape == "line":
        half = max(0.35, radius * 0.35)
        pts = [place(-long, -half), place(long, -half), place(long, half), place(-long, half)]
        return [c for p in pts for c in p]
    if shape == "star":
        points = []
        for i in range(10):
            a = angle + (i / 10) * math.tau - math.pi / 2
            r = radius if i % 2 == 0 else radius * 0.45
            points.append(x + math.cos(a) * r * (stretch if i % 2 == 0 else 1))
            points.append(y + math.sin(a) * r)
        return points
    if shape == "irregularQuad":
        # 種から決まる «歪んだ多角形»。紙吹雪や破片に使います。
        sides = int(clamp(_jround(emitter.get("shapeSides", 4) or 4), 3, 8))
        random = Random(int(_jround(seed * 1e6)))
        points = []
        for i in range(sides):
            a = angle + (i / sides) * math.tau
            r = radius * (0.6 + random() * 0.7)
            points.append(x + math.cos(a) * r * stretch)
            points.append(y + math.sin(a) * r)
        return points
    if stretch == 1:
        return circle_contour(x, y, radius, 16)
    # 伸ばした円＝楕円を進行方向に向ける
    points = []
    for i in range(20):
        a = (i / 20) * math.tau
        px, py = place(math.cos(a) * long, math.sin(a) * radius)
        points.append(px)
        points.append(py)
    return points


# ── 描画 ─────────────────────────────────────────────────────────

@njit(cache=True, fastmath=False)
def _k_fill_particles(dst, flat, starts, counts, colors, alphas):
    """粒をまとめて塗る。**粒ごとの «囲む矩形» の中だけ**を触ります。

    粒ごとに全画面の被覆率バッファを取ると、900 粒で 3 GB ぶん触ることになり、
    そこだけで 1 フレーム数秒かかります。ここでは矩形ぶんの小さな作業配列を
    使い回すので、粒の面積の合計にしか比例しません。
    """
    height, width = dst.shape[0], dst.shape[1]
    subsamples = 4
    weight = 1.0 / subsamples
    max_edges = 0
    for k in range(counts.shape[0]):
        if counts[k] > max_edges:
            max_edges = counts[k]
    if max_edges < 2:
        return
    cx = np.empty(max_edges, np.float64)
    cd = np.empty(max_edges, np.int64)

    for k in range(starts.shape[0]):
        n = counts[k]
        if n < 3:
            continue
        base = starts[k]
        min_x = 1e30
        max_x = -1e30
        min_y = 1e30
        max_y = -1e30
        for i in range(n):
            px = flat[base + i * 2]
            py = flat[base + i * 2 + 1]
            if px < min_x:
                min_x = px
            if px > max_x:
                max_x = px
            if py < min_y:
                min_y = py
            if py > max_y:
                max_y = py
        x0 = int(math.floor(min_x))
        x1 = int(math.ceil(max_x))
        y0 = int(math.floor(min_y))
        y1 = int(math.ceil(max_y))
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > width - 1:
            x1 = width - 1
        if y1 > height - 1:
            y1 = height - 1
        if x0 > x1 or y0 > y1:
            continue

        span = x1 - x0 + 1
        coverage = np.zeros(span, np.float64)
        cr = colors[k, 0]
        cg = colors[k, 1]
        cb = colors[k, 2]
        ca = colors[k, 3] * alphas[k]
        if ca <= 0.0:
            continue

        for y in range(y0, y1 + 1):
            for i in range(span):
                coverage[i] = 0.0
            for s in range(subsamples):
                sy = y + (s + 0.5) / subsamples
                found = 0
                for i in range(n):
                    j = (i + 1) % n
                    ay = flat[base + i * 2 + 1]
                    by = flat[base + j * 2 + 1]
                    if ay == by:
                        continue
                    lo = ay if ay < by else by
                    hi = ay if ay > by else by
                    if sy < lo or sy >= hi:
                        continue
                    ax = flat[base + i * 2]
                    bx = flat[base + j * 2]
                    t = (sy - ay) / (by - ay)
                    cx[found] = ax + (bx - ax) * t
                    cd[found] = 1 if by > ay else -1
                    found += 1
                if found < 2:
                    continue
                for i in range(1, found):
                    kx = cx[i]
                    kd = cd[i]
                    j = i - 1
                    while j >= 0 and cx[j] > kx:
                        cx[j + 1] = cx[j]
                        cd[j + 1] = cd[j]
                        j -= 1
                    cx[j + 1] = kx
                    cd[j + 1] = kd
                winding = 0
                for i in range(found - 1):
                    winding += cd[i]
                    if winding == 0:
                        continue
                    sa = cx[i]
                    sb = cx[i + 1]
                    if sb <= sa:
                        continue
                    if sa < x0:
                        sa = x0 * 1.0
                    if sb > x1 + 1:
                        sb = (x1 + 1) * 1.0
                    if sb <= sa:
                        continue
                    first = int(math.floor(sa)) - x0
                    last = int(math.floor(sb - 1e-9)) - x0
                    if first < 0:
                        first = 0
                    if last > span - 1:
                        last = span - 1
                    if first == last:
                        coverage[first] += (sb - sa) * weight
                    else:
                        coverage[first] += (first + x0 + 1 - sa) * weight
                        for xi in range(first + 1, last):
                            coverage[xi] += weight
                        coverage[last] += (sb - (last + x0)) * weight

            for i in range(span):
                cov = coverage[i]
                if cov <= 0.0005:
                    continue
                if cov > 1.0:
                    cov = 1.0
                sa = cov * ca
                if sa <= 0.0:
                    continue
                x = x0 + i
                da = dst[y, x, 3] / 255.0
                out_a = sa + da * (1.0 - sa)
                if out_a <= 0.0:
                    continue
                for c in range(3):
                    src = cr if c == 0 else (cg if c == 1 else cb)
                    value = (src * sa + dst[y, x, c] * da * (1.0 - sa)) / out_a
                    if value < 0.0:
                        value = 0.0
                    elif value > 255.0:
                        value = 255.0
                    dst[y, x, c] = np.uint8(np.rint(value))
                av = out_a * 255.0
                if av > 255.0:
                    av = 255.0
                dst[y, x, 3] = np.uint8(np.rint(av))


def render_particles(system: ParticleSystem, emitter: dict, width: int, height: int,
                     transform: dict | None = None, scale: float = 1.0, sprite: Bitmap | None = None) -> Bitmap:
    """粒を 1 枚のビットマップに描く。

    粒は **ワールド座標で進む**ので、エミッターからの相対で描きます。そうすると
    レイヤーのトランスフォームでまとめて動かす・回す・拡大することができます。

    :param emitter: `color` `endColor` `shape` `stretch` `alignToVelocity` など
    :param sprite: 素材を貼るとき。無ければ `shape` の図形を塗ります
    """
    emitter = emitter or {}
    transform = transform or {"x": 0, "y": 0}
    buffer = Bitmap(int(_jround(width * scale)), int(_jround(height * scale)))
    snapshot = system.render()
    n = snapshot["count"]
    if n == 0:
        return buffer

    centre_x = (width / 2) * scale
    centre_y = (height / 2) * scale
    xs = (snapshot["x"] - (transform.get("x", 0) or 0)) * scale + centre_x
    ys = (snapshot["y"] - (transform.get("y", 0) or 0)) * scale + centre_y

    if sprite is not None:
        # 素材を貼る場合は、粒ごとに回した四角形（三角形 2 枚）として描きます。
        sizes = snapshot["size"] * scale
        for i in range(n):
            half = sizes[i] / 2
            angle = math.radians(snapshot["rotation"][i])
            cos = math.cos(angle)
            sin = math.sin(angle)
            corners = []
            for lx, ly in ((-half, -half), (half, -half), (half, half), (-half, half)):
                corners.append((xs[i] + lx * cos - ly * sin, ys[i] + lx * sin + ly * cos))
            uv = [(0.0, 0.0), (float(sprite.width), 0.0), (float(sprite.width), float(sprite.height)),
                  (0.0, float(sprite.height))]
            options = {"alpha": float(snapshot["opacity"][i]), "clampEdge": True}
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                draw_textured_triangle(
                    buffer,
                    sprite,
                    {"x": corners[a][0], "y": corners[a][1], "u": uv[a][0], "v": uv[a][1]},
                    {"x": corners[b][0], "y": corners[b][1], "u": uv[b][0], "v": uv[b][1]},
                    {"x": corners[c][0], "y": corners[c][1], "u": uv[c][0], "v": uv[c][1]},
                    options,
                )
        return buffer

    start_color = parse_color(emitter.get("color", "#ffffff"))
    end_color = parse_color(emitter["endColor"]) if emitter.get("endColor") else None

    contours = []
    colors = np.empty((n, 4), np.float64)
    for i in range(n):
        radius = (snapshot["size"][i] / 2) * scale
        contours.append(
            particle_contour(
                float(xs[i]), float(ys[i]), float(radius), float(snapshot["rotation"][i]),
                float(snapshot["vx"][i]), float(snapshot["vy"][i]), float(snapshot["seed"][i]), emitter,
            )
        )
        if end_color is None:
            colors[i] = (start_color.r, start_color.g, start_color.b, start_color.a)
        else:
            k = clamp(float(snapshot["progress"][i]), 0, 1)
            colors[i] = (
                float(_jround(start_color.r + (end_color.r - start_color.r) * k)),
                float(_jround(start_color.g + (end_color.g - start_color.g) * k)),
                float(_jround(start_color.b + (end_color.b - start_color.b) * k)),
                start_color.a + (end_color.a - start_color.a) * k,
            )

    flat = np.concatenate([np.asarray(c, np.float64) for c in contours])
    starts = np.zeros(n, np.int64)
    counts = np.zeros(n, np.int64)
    offset = 0
    for i, c in enumerate(contours):
        starts[i] = offset
        counts[i] = len(c) // 2
        offset += len(c)
    _k_fill_particles(buffer.data, flat, starts, counts, colors, snapshot["opacity"].astype(np.float64))
    return buffer


__all__ = [
    "MAX_PARTICLES",
    "ParticleSystem",
    "create_particle_system",
    "particle_contour",
    "render_particles",
]
