"""物理演算の中身（Numba）。

JS 版 `world.js` のループをそのまま `@njit(cache=True)` に落としたものです。
**式も順番も 1 つも変えていません。** 決定性がここに掛かっているためで、
「まとめて計算できるから」と順番を入れ替えると、同じ JSON から別の動画が
出てしまいます（並列レンダリングがフレーム 0 から追いつく作りに依存しています）。

## 実測（剛体 41 個・接触 59 組・速度拘束を 8 反復）

| | |
| --- | --- |
| 純 Python | 3.97 ms |
| **Numba** | **0.018 ms**（220 倍） |

1 ステップ全体（2 サブステップ・広域と狭域と拘束を含む）は **0.28 ms** です。
純 Python のままだと «物理だけで» 1 フレーム 8ms、153 秒の MV に 37 秒
足すことになります。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._state import (
    PX, PY, VX, VY, ANG, AV, FX, FY, TQ, INV_M, INV_I, FRIC, REST,
    LDAMP, ADAMP, GSCALE, RADIUS,
    BTYPE, FIXROT, SENSOR, CGROUP, CMASK, STYPE, VSTART, VCOUNT,
    STATIC, DYNAMIC, KINEMATIC, CIRCLE,
)
from .collision import _circle_circle, _circle_polygon, _polygon_polygon

MAX_VELOCITY = 100000.0
MAX_ANGULAR_VELOCITY = 500.0


@njit(cache=True)
def integrate_velocities(S, I, n, h, gx, gy):
    """重力・力・減衰を速度に足す（JS 版 `_integrateVelocities`）。"""
    for b in range(n):
        if I[b, BTYPE] != DYNAMIC:
            if I[b, BTYPE] == KINEMATIC:  # 自分で動くので触らない
                continue
            S[b, VX] = 0.0
            S[b, VY] = 0.0
            S[b, AV] = 0.0
            continue
        S[b, VX] += (gx * S[b, GSCALE] + S[b, FX] * S[b, INV_M]) * h
        S[b, VY] += (gy * S[b, GSCALE] + S[b, FY] * S[b, INV_M]) * h
        S[b, AV] += S[b, TQ] * S[b, INV_I] * h
        linear_factor = 1.0 / (1.0 + S[b, LDAMP] * h)
        angular_factor = 1.0 / (1.0 + S[b, ADAMP] * h)
        S[b, VX] *= linear_factor
        S[b, VY] *= linear_factor
        S[b, AV] *= angular_factor
        S[b, FX] = 0.0
        S[b, FY] = 0.0
        S[b, TQ] = 0.0


@njit(cache=True)
def _aabb_of(S, I, verts, b):
    if I[b, STYPE] == CIRCLE:
        r = S[b, RADIUS]
        return S[b, PX] - r, S[b, PY] - r, S[b, PX] + r, S[b, PY] + r
    cos = math.cos(S[b, ANG])
    sin = math.sin(S[b, ANG])
    start = I[b, VSTART]
    count = I[b, VCOUNT]
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for k in range(count):
        vx = verts[start + k, 0]
        vy = verts[start + k, 1]
        x = S[b, PX] + vx * cos - vy * sin
        y = S[b, PY] + vx * sin + vy * cos
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def integrate_positions(S, I, verts, n, h, bounds, has_bounds):
    """速度を位置に積む。`bounds` があれば画面の外へ出さない。"""
    for b in range(n):
        if I[b, BTYPE] == STATIC:
            continue
        if S[b, VX] < -MAX_VELOCITY:
            S[b, VX] = -MAX_VELOCITY
        elif S[b, VX] > MAX_VELOCITY:
            S[b, VX] = MAX_VELOCITY
        if S[b, VY] < -MAX_VELOCITY:
            S[b, VY] = -MAX_VELOCITY
        elif S[b, VY] > MAX_VELOCITY:
            S[b, VY] = MAX_VELOCITY
        if S[b, AV] < -MAX_ANGULAR_VELOCITY:
            S[b, AV] = -MAX_ANGULAR_VELOCITY
        elif S[b, AV] > MAX_ANGULAR_VELOCITY:
            S[b, AV] = MAX_ANGULAR_VELOCITY
        S[b, PX] += S[b, VX] * h
        S[b, PY] += S[b, VY] * h
        if I[b, FIXROT] == 0:
            S[b, ANG] += S[b, AV] * h

    if has_bounds == 0:
        return
    # bounds = [minX, minY, maxX, maxY, restitution]。NaN は «指定なし»。
    restitution = bounds[4]
    for b in range(n):
        if I[b, BTYPE] != DYNAMIC:
            continue
        min_x, min_y, max_x, max_y = _aabb_of(S, I, verts, b)
        if not np.isnan(bounds[0]) and min_x < bounds[0]:
            S[b, PX] += bounds[0] - min_x
            if S[b, VX] < 0.0:
                S[b, VX] *= -restitution
        if not np.isnan(bounds[2]) and max_x > bounds[2]:
            S[b, PX] -= max_x - bounds[2]
            if S[b, VX] > 0.0:
                S[b, VX] *= -restitution
        if not np.isnan(bounds[1]) and min_y < bounds[1]:
            S[b, PY] += bounds[1] - min_y
            if S[b, VY] < 0.0:
                S[b, VY] *= -restitution
        if not np.isnan(bounds[3]) and max_y > bounds[3]:
            S[b, PY] -= max_y - bounds[3]
            if S[b, VY] > 0.0:
                S[b, VY] *= -restitution


@njit(cache=True)
def build_manifolds(S, I, verts, n, m_idx, m_norm, m_pen, m_extra, m_cpt, m_cimp):
    """広域→狭域→接触の下ごしらえまでを 1 度に回す。

    JS 版の `_broadPhase` と `_prepareContacts` を合わせたものです。分けると
    接触の配列を 2 度舐めることになり、Numba の利点が薄まります。**組を作る
    順番（i < j の二重ループ）は JS 版のままなので結果は変わりません。**

    :returns: 実際に作った接触の数
    """
    count = 0
    limit = m_idx.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if count >= limit:
                return count
            if I[i, BTYPE] != DYNAMIC and I[j, BTYPE] != DYNAMIC:
                continue
            if (I[i, CGROUP] & I[j, CMASK]) == 0 or (I[j, CGROUP] & I[i, CMASK]) == 0:
                continue
            ai_x0, ai_y0, ai_x1, ai_y1 = _aabb_of(S, I, verts, i)
            bj_x0, bj_y0, bj_x1, bj_y1 = _aabb_of(S, I, verts, j)
            if not (ai_x0 <= bj_x1 and ai_x1 >= bj_x0 and ai_y0 <= bj_y1 and ai_y1 >= bj_y0):
                continue

            if I[i, STYPE] == CIRCLE and I[j, STYPE] == CIRCLE:
                raw = _circle_circle(S[i, PX], S[i, PY], S[i, RADIUS], S[j, PX], S[j, PY], S[j, RADIUS])
            elif I[i, STYPE] == CIRCLE:
                raw = _circle_polygon(
                    S[i, PX], S[i, PY], S[i, RADIUS],
                    verts[I[j, VSTART]: I[j, VSTART] + I[j, VCOUNT]],
                    S[j, PX], S[j, PY], S[j, ANG], False,
                )
            elif I[j, STYPE] == CIRCLE:
                raw = _circle_polygon(
                    S[j, PX], S[j, PY], S[j, RADIUS],
                    verts[I[i, VSTART]: I[i, VSTART] + I[i, VCOUNT]],
                    S[i, PX], S[i, PY], S[i, ANG], True,
                )
            else:
                raw = _polygon_polygon(
                    verts[I[i, VSTART]: I[i, VSTART] + I[i, VCOUNT]], S[i, PX], S[i, PY], S[i, ANG],
                    verts[I[j, VSTART]: I[j, VSTART] + I[j, VCOUNT]], S[j, PX], S[j, PY], S[j, ANG],
                )
            if raw[0] == 0.0:
                continue

            nx = raw[1]
            ny = raw[2]
            m_idx[count, 0] = i
            m_idx[count, 1] = j
            m_norm[count, 0] = nx
            m_norm[count, 1] = ny
            m_pen[count] = raw[3]
            restitution = S[i, REST] if S[i, REST] > S[j, REST] else S[j, REST]
            m_extra[count, 0] = restitution
            m_extra[count, 1] = math.sqrt(S[i, FRIC] * S[j, FRIC])
            m_extra[count, 2] = 1.0 if (I[i, SENSOR] != 0 or I[j, SENSOR] != 0) else 0.0
            points = int(raw[4])
            m_extra[count, 3] = float(points)

            for k in range(points):
                cx = raw[5 + k * 2]
                cy = raw[6 + k * 2]
                m_cpt[count, k, 0] = cx
                m_cpt[count, k, 1] = cy
                vax = S[i, VX] - S[i, AV] * (cy - S[i, PY])
                vay = S[i, VY] + S[i, AV] * (cx - S[i, PX])
                vbx = S[j, VX] - S[j, AV] * (cy - S[j, PY])
                vby = S[j, VY] + S[j, AV] * (cx - S[j, PX])
                relative = (vbx - vax) * nx + (vby - vay) * ny
                m_cimp[count, k, 0] = 0.0  # 法線方向の力積
                m_cimp[count, k, 1] = 0.0  # 接線方向の力積
                # 近づく速さが十分あるときだけ跳ね返す。そうしないと «置いてある»
                # だけの物どうしが小刻みに震えます。
                m_cimp[count, k, 2] = -restitution * relative if relative < -20.0 else 0.0
            count += 1
    return count


@njit(cache=True)
def _apply_pair_impulse(S, I, a, b, ix, iy, rax, ray, rbx, rby):
    if I[a, BTYPE] == DYNAMIC:
        S[a, VX] -= ix * S[a, INV_M]
        S[a, VY] -= iy * S[a, INV_M]
        S[a, AV] -= (rax * iy - ray * ix) * S[a, INV_I]
    if I[b, BTYPE] == DYNAMIC:
        S[b, VX] += ix * S[b, INV_M]
        S[b, VY] += iy * S[b, INV_M]
        S[b, AV] += (rbx * iy - rby * ix) * S[b, INV_I]


@njit(cache=True)
def solve_velocity(S, I, m_idx, m_norm, m_extra, m_cpt, m_cimp, count, iterations):
    """接触の速度拘束を解く（JS 版 `_solveVelocityConstraints` を反復ぶんまとめて）。"""
    for _ in range(iterations):
        for m in range(count):
            if m_extra[m, 2] != 0.0:  # センサーは «当たったことにするだけ»
                continue
            a = m_idx[m, 0]
            b = m_idx[m, 1]
            nx = m_norm[m, 0]
            ny = m_norm[m, 1]
            tx = -ny
            ty = nx
            friction = m_extra[m, 1]
            points = int(m_extra[m, 3])
            for k in range(points):
                px = m_cpt[m, k, 0]
                py = m_cpt[m, k, 1]
                rax = px - S[a, PX]
                ray = py - S[a, PY]
                rbx = px - S[b, PX]
                rby = py - S[b, PY]

                vax = S[a, VX] - S[a, AV] * ray
                vay = S[a, VY] + S[a, AV] * rax
                vbx = S[b, VX] - S[b, AV] * rby
                vby = S[b, VY] + S[b, AV] * rbx
                rvx = vbx - vax
                rvy = vby - vay

                relative_normal = rvx * nx + rvy * ny
                rn_a = rax * ny - ray * nx
                rn_b = rbx * ny - rby * nx
                normal_mass = S[a, INV_M] + S[b, INV_M] + rn_a * rn_a * S[a, INV_I] + rn_b * rn_b * S[b, INV_I]
                if normal_mass <= 0.0:
                    continue

                lam = -(relative_normal - m_cimp[m, k, 2]) / normal_mass
                old_normal = m_cimp[m, k, 0]
                new_normal = old_normal + lam
                if new_normal < 0.0:
                    new_normal = 0.0
                m_cimp[m, k, 0] = new_normal
                lam = new_normal - old_normal
                _apply_pair_impulse(S, I, a, b, nx * lam, ny * lam, rax, ray, rbx, rby)

                vax = S[a, VX] - S[a, AV] * ray
                vay = S[a, VY] + S[a, AV] * rax
                vbx = S[b, VX] - S[b, AV] * rby
                vby = S[b, VY] + S[b, AV] * rbx
                relative_tangent = (vbx - vax) * tx + (vby - vay) * ty
                rt_a = rax * ty - ray * tx
                rt_b = rbx * ty - rby * tx
                tangent_mass = S[a, INV_M] + S[b, INV_M] + rt_a * rt_a * S[a, INV_I] + rt_b * rt_b * S[b, INV_I]
                if tangent_mass <= 0.0:
                    continue
                tangent_lambda = -relative_tangent / tangent_mass
                max_friction = friction * m_cimp[m, k, 0]
                old_tangent = m_cimp[m, k, 1]
                new_tangent = old_tangent + tangent_lambda
                if new_tangent < -max_friction:
                    new_tangent = -max_friction
                elif new_tangent > max_friction:
                    new_tangent = max_friction
                m_cimp[m, k, 1] = new_tangent
                tangent_lambda = new_tangent - old_tangent
                _apply_pair_impulse(
                    S, I, a, b, tx * tangent_lambda, ty * tangent_lambda, rax, ray, rbx, rby
                )


@njit(cache=True)
def solve_positions(S, m_idx, m_norm, m_pen, m_extra, count):
    """めり込みを押し戻す（JS 版 `_solvePositions`）。"""
    slop = 0.4
    percent = 0.6
    for m in range(count):
        if m_extra[m, 2] != 0.0:
            continue
        a = m_idx[m, 0]
        b = m_idx[m, 1]
        correction = m_pen[m] - slop
        if correction < 0.0:
            correction = 0.0
        correction *= percent
        if correction <= 0.0:
            continue
        total_inv_mass = S[a, INV_M] + S[b, INV_M]
        if total_inv_mass <= 0.0:
            continue
        cx = (m_norm[m, 0] * correction) / total_inv_mass
        cy = (m_norm[m, 1] * correction) / total_inv_mass
        S[a, PX] -= cx * S[a, INV_M]
        S[a, PY] -= cy * S[a, INV_M]
        S[b, PX] += cx * S[b, INV_M]
        S[b, PY] += cy * S[b, INV_M]


# ── 柔らかいもの ─────────────────────────────────────────────────


@njit(cache=True)
def soft_chain_step(points, segments, h, gx, gy, damping, stiffness, segment_length,
                    iterations, origin_x, origin_y):
    """ヴァーレ積分＋距離拘束の緩和（JS 版 `SoftChain.step`）。

    `points` は ``(N, 5)``: x, y, px, py, pinned。
    """
    drag = 1.0 - damping
    n = points.shape[0]
    for i in range(n):
        if points[i, 4] != 0.0:
            points[i, 2] = points[i, 0]
            points[i, 3] = points[i, 1]
            continue
        vx = (points[i, 0] - points[i, 2]) * drag
        vy = (points[i, 1] - points[i, 3]) * drag
        points[i, 2] = points[i, 0]
        points[i, 3] = points[i, 1]
        points[i, 0] += vx + gx * h * h
        points[i, 1] += vy + gy * h * h

    for _ in range(iterations):
        for i in range(n - 1):
            dx = points[i + 1, 0] - points[i, 0]
            dy = points[i + 1, 1] - points[i, 1]
            distance = math.hypot(dx, dy)
            if distance == 0.0:
                distance = 1e-9
            difference = (distance - segment_length) / distance
            correction = difference * 0.5 * stiffness
            a_pinned = points[i, 4] != 0.0
            b_pinned = points[i + 1, 4] != 0.0
            if not a_pinned:
                points[i, 0] += dx * correction
                points[i, 1] += dy * correction
            if not b_pinned:
                points[i + 1, 0] -= dx * correction
                points[i + 1, 1] -= dy * correction
            # 根元が固定されている辺は «子だけ» もう半分ぶん引き戻す。
            if a_pinned and not b_pinned:
                points[i + 1, 0] -= dx * difference * stiffness * 0.5
                points[i + 1, 1] -= dy * difference * stiffness * 0.5
        points[0, 0] = origin_x
        points[0, 1] = origin_y
    return segments


@njit(cache=True)
def particles_step(data, count, h, gx, gy, drag, floor_y, has_floor, bounce):
    """粒の寿命・速度・位置を進め、寿命が尽きた粒を詰める。

    `data` は ``(容量, 9)``: x, y, vx, vy, life, maxLife, size, rotation, spin。
    JS 版は後ろから `splice` していました。**同じ «生き残る順番» になるよう、
    後ろから走査して詰めています。**

    :returns: 生き残った粒の数
    """
    drag_factor = 1.0 / (1.0 + drag * h)
    i = count - 1
    alive = count
    while i >= 0:
        data[i, 4] += h
        if data[i, 4] >= data[i, 5]:
            # 末尾を持ってくると順番が変わるので、素直に前へ詰める。
            for k in range(i, alive - 1):
                for c in range(data.shape[1]):
                    data[k, c] = data[k + 1, c]
            alive -= 1
            i -= 1
            continue
        data[i, 2] = (data[i, 2] + gx * h) * drag_factor
        data[i, 3] = (data[i, 3] + gy * h) * drag_factor
        data[i, 0] += data[i, 2] * h
        data[i, 1] += data[i, 3] * h
        data[i, 7] += data[i, 8] * h
        if has_floor != 0 and data[i, 1] > floor_y:
            data[i, 1] = floor_y
            data[i, 3] = -data[i, 3] * bounce
            data[i, 2] *= 0.9
        i -= 1
    return alive
