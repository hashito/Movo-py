"""movo-physics-2d — 内蔵の 2 次元剛体エンジン（JS 版 world.js の移植）。

力積ベース・**固定タイムステップ**・決定的。同じ剛体を同じ回数進めれば
必ず同じ軌跡になります。単位は «画面の画素» と «秒» なので
`gravity: {x: 0, y: 980}` がそのまま読めます。

## 決定性について（ここが Movo の土台です）

並列レンダリングは «どのフレームからでも 0 フレーム目から追いつける» ことに
依存しています。だから

  - 乱数は使いません（粒子だけがシード付きで使います）
  - 剛体の並び順・接触を作る順番は JS 版のまま（i < j の二重ループ）
  - `dt` は呼び手が渡さなければ `timeStep` 固定

を守っています。**「まとめて計算できるから」と順番を変えないでください。**

## 状態の持ち方

剛体の値は `World` が持つ 1 枚の `float64` 配列に置いてあります
（詳しくは `_state.py`）。`body.position.x` はその配列を覗く窓です。
Numba のカーネルへ «詰め直しゼロ» で渡すための作りです。
"""

from __future__ import annotations

import math

import numpy as np

from . import _kernels as K
from ._compat import clamp
from ._state import (
    ADAMP, ANG, AV, BTYPE, CGROUP, CIRCLE, CMASK, DYNAMIC, FIXROT, FRIC, FX, FY,
    GSCALE, INERTIA, INV_I, INV_M, LDAMP, MASS, POLYGON, PX, PY,
    RADIUS, REST, SENSOR, STYPE, TQ, VCOUNT, VSTART, VX, VY,
    Bank, Vec2View, _TYPE_CODES, _TYPE_NAMES,
)
from .shapes import Shape, circle_shape, shape_area, shape_inertia

_body_counter = 0


class Body:
    """剛体 1 つ。

    JS 版と同じ書き味（`body.position.x`）のまま、値は配列に置いています。

    :param options: JS 版 `new Body({...})` と同じキー
    """

    __slots__ = ("id", "shape", "user_data", "_bank", "_row", "position", "velocity", "force")

    def __init__(self, **options) -> None:
        global _body_counter
        _body_counter += 1
        # JS 版は Math.random() で id を作っていました。**決定性のために連番に
        # 変えています**（id が出力に混ざる場面はありませんが、揃えておく方が安全）。
        self.id = options.get("id") or f"body-{_body_counter:06d}"
        self.shape: Shape = options.get("shape") or circle_shape(25)
        self.user_data = options.get("userData")

        self._bank = Bank(1)
        self._row = self._bank.allocate()
        self.position = Vec2View(self, PX, PY)
        self.velocity = Vec2View(self, VX, VY)
        self.force = Vec2View(self, FX, FY)

        S = self._bank.S
        I = self._bank.I
        r = self._row
        kind = options.get("bodyType") or options.get("type") or "dynamic"
        I[r, BTYPE] = _TYPE_CODES.get(kind, DYNAMIC)
        S[r, PX] = float(options.get("x", 0.0))
        S[r, PY] = float(options.get("y", 0.0))
        S[r, VX] = float(options.get("velocityX", 0.0))
        S[r, VY] = float(options.get("velocityY", 0.0))
        S[r, ANG] = float(options.get("angle", 0.0))
        S[r, AV] = float(options.get("angularVelocity", 0.0))
        S[r, FRIC] = float(options.get("friction", 0.3))
        S[r, REST] = clamp(float(options.get("restitution", 0.2)), 0.0, 1.0)
        S[r, LDAMP] = float(options.get("linearDamping", 0.0))
        S[r, ADAMP] = float(options.get("angularDamping", 0.0))
        S[r, GSCALE] = float(options.get("gravityScale", 1.0))
        I[r, FIXROT] = 1 if options.get("fixedRotation", False) else 0
        I[r, SENSOR] = 1 if options.get("sensor", False) else 0
        I[r, CGROUP] = int(options.get("collisionGroup", 1))
        I[r, CMASK] = int(options.get("collisionMask", 0xFFFFFFFF))
        self._write_shape()

        mass = options.get("mass")
        if mass is None:
            mass = max(0.05, shape_area(self.shape) / 10000)
        self.set_mass(mass)

    # ── 形と質量 ────────────────────────────────────────────────

    def _write_shape(self) -> None:
        I = self._bank.I
        S = self._bank.S
        r = self._row
        if self.shape.type == "circle":
            I[r, STYPE] = CIRCLE
            S[r, RADIUS] = self.shape.radius
            I[r, VCOUNT] = 0
        else:
            I[r, STYPE] = POLYGON
            I[r, VCOUNT] = len(self.shape.vertices)

    def set_mass(self, mass: float) -> None:
        """質量を決め直す。慣性モーメントも形から作り直します。"""
        S = self._bank.S
        r = self._row
        m = max(1e-6, float(mass))
        S[r, MASS] = m
        is_dynamic = self._bank.I[r, BTYPE] == DYNAMIC
        S[r, INV_M] = 1.0 / m if is_dynamic else 0.0
        inertia = shape_inertia(self.shape, m)
        S[r, INERTIA] = inertia
        fixed = self._bank.I[r, FIXROT] != 0
        S[r, INV_I] = 1.0 / inertia if (is_dynamic and not fixed and inertia > 0) else 0.0

    # ── 値の出し入れ ────────────────────────────────────────────

    def _get(self, column: int) -> float:
        return float(self._bank.S[self._row, column])

    def _set(self, column: int, value: float) -> None:
        self._bank.S[self._row, column] = value

    @property
    def type(self) -> str:
        return _TYPE_NAMES[int(self._bank.I[self._row, BTYPE])]

    @property
    def is_dynamic(self) -> bool:
        return self._bank.I[self._row, BTYPE] == DYNAMIC

    @property
    def angle(self) -> float:
        return self._get(ANG)

    @angle.setter
    def angle(self, value: float) -> None:
        self._set(ANG, value)

    @property
    def angular_velocity(self) -> float:
        return self._get(AV)

    @angular_velocity.setter
    def angular_velocity(self, value: float) -> None:
        self._set(AV, value)

    @property
    def torque(self) -> float:
        return self._get(TQ)

    @torque.setter
    def torque(self, value: float) -> None:
        self._set(TQ, value)

    @property
    def mass(self) -> float:
        return self._get(MASS)

    @property
    def inertia(self) -> float:
        return self._get(INERTIA)

    @property
    def inv_mass(self) -> float:
        return self._get(INV_M)

    @property
    def inv_inertia(self) -> float:
        return self._get(INV_I)

    @property
    def restitution(self) -> float:
        return self._get(REST)

    @property
    def friction(self) -> float:
        return self._get(FRIC)

    @property
    def sensor(self) -> bool:
        return self._bank.I[self._row, SENSOR] != 0

    # ── 力と速度 ────────────────────────────────────────────────

    def apply_force(self, x: float, y: float) -> None:
        self._set(FX, self._get(FX) + x)
        self._set(FY, self._get(FY) + y)

    def apply_impulse(self, ix: float, iy: float, contact_x=None, contact_y=None) -> None:
        """力積を加える。接触点を渡すと回転にも効きます。"""
        if not self.is_dynamic:
            return
        inv_m = self._get(INV_M)
        self._set(VX, self._get(VX) + ix * inv_m)
        self._set(VY, self._get(VY) + iy * inv_m)
        if contact_x is not None:
            rx = contact_x - self._get(PX)
            ry = contact_y - self._get(PY)
            self._set(AV, self._get(AV) + (rx * iy - ry * ix) * self._get(INV_I))

    def velocity_at(self, x: float, y: float) -> tuple[float, float]:
        """世界座標 `(x, y)` にある «その剛体の一点» の速度。"""
        rx = x - self._get(PX)
        ry = y - self._get(PY)
        av = self._get(AV)
        return (self._get(VX) - av * ry, self._get(VY) + av * rx)

    def aabb(self) -> tuple[float, float, float, float]:
        from .shapes import shape_aabb

        return shape_aabb(self.shape, (self.position.x, self.position.y), self.angle)

    def state(self) -> dict:
        """描画側へ渡す «見えている値» だけの写し。"""
        return {
            "position": (self.position.x, self.position.y),
            "velocity": (self.velocity.x, self.velocity.y),
            "angle": self.angle,
            "angularVelocity": self.angular_velocity,
            "speed": math.hypot(self.velocity.x, self.velocity.y),
        }


class World:
    """剛体・拘束・柔らかいものをまとめて進める世界。

    :param config: ``gravity`` / ``timeStep`` / ``subSteps`` / ``iterations``
        / ``bounds`` / ``pixelsPerMeter``
    """

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        gravity = config.get("gravity") or {}
        self.gravity = (float(gravity.get("x", 0.0)), float(gravity.get("y", 980.0)))
        self.time_step = float(config.get("timeStep", 1.0 / 60.0))
        self.sub_steps = max(1, round(config.get("subSteps", 2)))
        self.iterations = max(1, round(config.get("iterations", 8)))
        self.bounds = config.get("bounds")
        self.pixels_per_meter = float(config.get("pixelsPerMeter", 100))
        self.bodies: list[Body] = []
        self.constraints: list[dict] = []
        self.soft_bodies: list = []
        self.time = 0.0
        self.step_count = 0
        self.contacts: list[dict] = []
        self.unstable = False
        self._by_id: dict[str, Body] = {}
        self._bank = Bank(4)
        self._verts = np.zeros((0, 2), np.float64)
        self._verts_dirty = True
        self._manifolds = None

    # ── 出し入れ ────────────────────────────────────────────────

    def add_body(self, body: Body) -> Body:
        """剛体を加える。値はこの世界の配列へ引っ越します。"""
        row = self._bank.allocate()
        self._bank.S[row] = body._bank.S[body._row]
        self._bank.I[row] = body._bank.I[body._row]
        body._bank = self._bank
        body._row = row
        self.bodies.append(body)
        self._by_id[body.id] = body
        self._verts_dirty = True
        return body

    def remove_body(self, body: Body) -> None:
        """剛体を外す。**並び順は保ちます**（順番が変わると結果が変わるため）。"""
        if body not in self.bodies:
            return
        index = self.bodies.index(body)
        S = self._bank.S
        I = self._bank.I
        count = self._bank.count
        # 外した剛体は自前の入れ物へ «値ごと» 戻します。そのまま別の世界へ
        # 入れ直せるようにするためで、消してしまうと位置が原点に飛びます。
        detached = Bank(1)
        detached.allocate()
        detached.S[0] = S[index]
        detached.I[0] = I[index]

        self.bodies.pop(index)
        self._by_id.pop(body.id, None)
        S[index: count - 1] = S[index + 1: count]
        I[index: count - 1] = I[index + 1: count]
        self._bank.count -= 1
        for k in range(index, len(self.bodies)):
            self.bodies[k]._row = k
        body._bank = detached
        body._row = 0
        self._verts_dirty = True

    def body_by_id(self, body_id: str):
        return self._by_id.get(body_id)

    def add_constraint(self, constraint: dict) -> dict:
        self.constraints.append(constraint)
        return constraint

    def add_soft_body(self, soft):
        self.soft_bodies.append(soft)
        return soft

    # ── 1 ステップ ──────────────────────────────────────────────

    def _rebuild_vertices(self) -> None:
        """多角形の頂点を 1 本の配列にまとめる（Numba へ渡すため）。"""
        total = sum(len(b.shape.vertices) for b in self.bodies if b.shape.type != "circle")
        verts = np.zeros((max(1, total), 2), np.float64)
        cursor = 0
        for k, body in enumerate(self.bodies):
            if body.shape.type == "circle":
                self._bank.I[k, VSTART] = 0
                self._bank.I[k, VCOUNT] = 0
                continue
            n = len(body.shape.vertices)
            verts[cursor: cursor + n] = body.shape.vertices
            self._bank.I[k, VSTART] = cursor
            self._bank.I[k, VCOUNT] = n
            cursor += n
        self._verts = verts
        self._verts_dirty = False

    def _ensure_manifold_buffers(self, n: int):
        pairs = max(1, n * (n - 1) // 2)
        if self._manifolds is None or self._manifolds[0].shape[0] < pairs:
            self._manifolds = (
                np.zeros((pairs, 2), np.int64),      # 剛体の組
                np.zeros((pairs, 2), np.float64),    # 法線
                np.zeros(pairs, np.float64),         # めり込み量
                np.zeros((pairs, 4), np.float64),    # 反発・摩擦・センサー・接触点数
                np.zeros((pairs, 2, 2), np.float64),  # 接触点の座標
                np.zeros((pairs, 2, 3), np.float64),  # 法線力積・接線力積・bias
            )
        return self._manifolds

    def step(self, dt: float | None = None) -> None:
        """固定タイムステップで 1 回だけ進める。"""
        if dt is None:
            dt = self.time_step
        n = len(self.bodies)
        if self._verts_dirty:
            self._rebuild_vertices()
        S = self._bank.S
        I = self._bank.I
        m_idx, m_norm, m_pen, m_extra, m_cpt, m_cimp = self._ensure_manifold_buffers(n)

        if self.bounds:
            bounds = np.array(
                [
                    float(self.bounds.get("minX", np.nan)),
                    float(self.bounds.get("minY", np.nan)),
                    float(self.bounds.get("maxX", np.nan)),
                    float(self.bounds.get("maxY", np.nan)),
                    float(self.bounds.get("restitution", 0.4)),
                ],
                np.float64,
            )
            has_bounds = 1
        else:
            bounds = np.zeros(5, np.float64)
            has_bounds = 0

        h = dt / self.sub_steps
        gx, gy = self.gravity
        count = 0
        for _ in range(self.sub_steps):
            K.integrate_velocities(S, I, n, h, gx, gy)
            count = K.build_manifolds(S, I, self._verts, n, m_idx, m_norm, m_pen, m_extra, m_cpt, m_cimp)
            K.solve_velocity(S, I, m_idx, m_norm, m_extra, m_cpt, m_cimp, count, self.iterations)
            # 拘束は «種類ごとに書き方が違う» ので Python 側で解きます。
            # 本数は普通ひと桁なので、ここが遅くなることはありません。
            for _ in range(self.iterations):
                self._solve_joints(h)
            K.integrate_positions(S, I, self._verts, n, h, bounds, has_bounds)
            K.solve_positions(S, m_idx, m_norm, m_pen, m_extra, count)
            for soft in self.soft_bodies:
                soft.step(h, self)
            self.time += h
        self.contacts = self._collect_contacts(count)
        self.step_count += 1
        self._check_stability()

    def _collect_contacts(self, count: int) -> list[dict]:
        """カーネルの結果を «見て分かる» 形に戻す。描画とデバッグ用です。"""
        if count == 0:
            return []
        m_idx, m_norm, m_pen, m_extra, m_cpt, _ = self._manifolds
        out = []
        for m in range(count):
            points = int(m_extra[m, 3])
            out.append(
                {
                    "bodyA": self.bodies[int(m_idx[m, 0])],
                    "bodyB": self.bodies[int(m_idx[m, 1])],
                    "normal": (float(m_norm[m, 0]), float(m_norm[m, 1])),
                    "penetration": float(m_pen[m]),
                    "restitution": float(m_extra[m, 0]),
                    "friction": float(m_extra[m, 1]),
                    "sensor": m_extra[m, 2] != 0.0,
                    "contacts": [(float(m_cpt[m, k, 0]), float(m_cpt[m, k, 1])) for k in range(points)],
                }
            )
        return out

    def _solve_joints(self, h: float) -> None:
        for constraint in self.constraints:
            if constraint.get("enabled") is False:
                continue
            solve_constraint(constraint, h)

    def _check_stability(self) -> None:
        """NaN になった剛体を原点へ戻す。**黙って壊れたまま進めません。**"""
        for body in self.bodies:
            if math.isfinite(body.position.x) and math.isfinite(body.position.y):
                continue
            if not self.unstable:
                self.unstable = True
                _warn(f'physics became unstable for body "{body.id}"; its state was reset')
            body.position.x = 0.0
            body.position.y = 0.0
            body.velocity.x = 0.0
            body.velocity.y = 0.0
            body.angular_velocity = 0.0

    def snapshot(self) -> dict:
        return {
            "time": self.time,
            "bodies": [{"id": b.id, **b.state()} for b in self.bodies],
            "softBodies": [
                {"id": s.id, "points": [(p[0], p[1]) for p in s.points]} for s in self.soft_bodies
            ],
        }


def _warn(message: str) -> None:
    try:  # pragma: no cover - core が入ればそちらへ
        from movo.core.logger import logger  # type: ignore

        logger.warn(message)
    except Exception:  # pragma: no cover
        import sys

        print(f"⚠ {message}", file=sys.stderr)


# ── 拘束 ──────────────────────────────────────────────────────


def _anchor_world(body: Body, anchor) -> tuple[float, float]:
    """取り付け点を世界座標へ。`None` なら重心そのもの。"""
    if not anchor:
        return (body.position.x, body.position.y)
    cos = math.cos(body.angle)
    sin = math.sin(body.angle)
    if isinstance(anchor, dict):
        ax = float(anchor.get("x", 0.0))
        ay = float(anchor.get("y", 0.0))
    else:
        ax = float(anchor[0])
        ay = float(anchor[1])
    return (body.position.x + ax * cos - ay * sin, body.position.y + ax * sin + ay * cos)


def solve_constraint(constraint: dict, h: float) -> None:
    """拘束を 1 本解く（JS 版 `solveConstraint` の移植）。

    種類は spring / rope / distance / pin / hinge。どれも «位置か速度を
    少しずつ直す» という形に揃えてあります。
    """
    body_a = constraint.get("bodyA")
    body_b = constraint.get("bodyB")
    kind = constraint.get("type")
    if body_a is None or body_b is None:
        return
    pa = _anchor_world(body_a, constraint.get("anchorA", constraint.get("anchor")))
    pb = _anchor_world(body_b, constraint.get("anchorB"))
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    distance = math.hypot(dx, dy)

    if kind == "spring":
        rest = float(constraint.get("restLength", 0.0))
        if distance < 1e-6:
            return
        nx = dx / distance
        ny = dy / distance
        stretch = distance - rest
        va = body_a.velocity_at(pa[0], pa[1])
        vb = body_b.velocity_at(pb[0], pb[1])
        relative = (vb[0] - va[0]) * nx + (vb[1] - va[1]) * ny
        force = -float(constraint.get("stiffness", 100)) * stretch - float(constraint.get("damping", 5)) * relative
        ix = nx * force * h
        iy = ny * force * h
        body_a.apply_impulse(-ix, -iy, pa[0], pa[1])
        body_b.apply_impulse(ix, iy, pb[0], pb[1])
        return

    if kind == "rope":
        max_length = constraint.get("length", constraint.get("restLength", 100))
        max_length = float(max_length)
        if distance <= max_length or distance < 1e-6:
            return
        _positional_correction(body_a, body_b, pa, pb, dx, dy, distance, max_length, 1.0)
        return

    if kind in ("distance", "pin", "hinge"):
        if kind == "distance":
            target = constraint.get("length", constraint.get("restLength", distance))
            target = float(target)
        else:
            target = 0.0
        stiffness = float(constraint.get("stiffness", 1))
        if distance < 1e-9 and target == 0.0:
            pass  # すでに重なっている
        else:
            _positional_correction(body_a, body_b, pa, pb, dx, dy, distance, target, clamp(stiffness, 0.0, 1.0))
        if kind == "hinge" and (constraint.get("minAngle") is not None or constraint.get("maxAngle") is not None):
            relative_angle = (body_b.angle - body_a.angle) * 180 / math.pi
            low = constraint.get("minAngle")
            high = constraint.get("maxAngle")
            low = -math.inf if low is None else float(low)
            high = math.inf if high is None else float(high)
            correction = 0.0
            if relative_angle < low:
                correction = (low - relative_angle) * math.pi / 180
            elif relative_angle > high:
                correction = (high - relative_angle) * math.pi / 180
            if correction != 0.0:
                total_inv = body_a.inv_inertia + body_b.inv_inertia
                if total_inv > 0:
                    body_b.angle = body_b.angle + (correction * body_b.inv_inertia) / total_inv
                    body_a.angle = body_a.angle - (correction * body_a.inv_inertia) / total_inv
                    body_b.angular_velocity = body_b.angular_velocity * 0.5
                    body_a.angular_velocity = body_a.angular_velocity * 0.5


def _positional_correction(body_a, body_b, pa, pb, dx, dy, distance, target, strength) -> None:
    total_inv_mass = body_a.inv_mass + body_b.inv_mass
    if total_inv_mass <= 0:
        return
    length = 1e-9 if distance < 1e-9 else distance
    nx = dx / length
    ny = dy / length
    error = (length - target) * strength
    correction_x = nx * error
    correction_y = ny * error
    body_a.position.x = body_a.position.x + (correction_x * body_a.inv_mass) / total_inv_mass
    body_a.position.y = body_a.position.y + (correction_y * body_a.inv_mass) / total_inv_mass
    body_b.position.x = body_b.position.x - (correction_x * body_b.inv_mass) / total_inv_mass
    body_b.position.y = body_b.position.y - (correction_y * body_b.inv_mass) / total_inv_mass

    # 拘束方向の速度を抜く。残しておくと «直したそばから引っ張り合う» ので。
    va = body_a.velocity_at(pa[0], pa[1])
    vb = body_b.velocity_at(pb[0], pb[1])
    relative = (vb[0] - va[0]) * nx + (vb[1] - va[1]) * ny
    if relative == 0:
        return
    impulse = -relative / total_inv_mass
    body_a.apply_impulse(-nx * impulse, -ny * impulse, pa[0], pa[1])
    body_b.apply_impulse(nx * impulse, ny * impulse, pb[0], pb[1])
