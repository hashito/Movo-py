"""狭域の当たり判定（JS 版 packages/physics/src/collision.js の移植）。

円／円・円／多角形・多角形／多角形（SAT と参照面クリップ）の 3 つだけです。
接触点は最大 2 つ。2 次元で箱を積み上げるにはこれで足ります。

## なぜ Numba か

ここは **辺の数 × 頂点の数** だけ内積を回します。多角形どうしなら
1 組で数百回の掛け算になり、剛体が 30 個あれば 1 ステップで数万回です。
NumPy で «辺ごとに一時配列を作る» と、配列を作る手間のほうが高くつきます
（README の「ラスタライザは NumPy では遅くなる」と同じ話です）。

実測（18 頂点どうし・10,000 組）:

| | |
| --- | --- |
| 純 Python | 1,095 ms |
| **Numba** | **85 ms**（13 倍） |

多角形の SAT は «辺ごとに全頂点へ内積» なので、頂点数が増えると効きます。

戻り値は «長さ 10 の float64 配列» にしています。Numba から
辞書やクラスを返すと型推論が効かず、`nopython` を外すことになるためです。

    [0] 当たったか（1/0）
    [1] 法線 x        A から B へ向く
    [2] 法線 y
    [3] めり込み量
    [4] 接触点の数（1 か 2）
    [5..8] 接触点（x, y, x, y）
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

EPSILON = 1e-9


@njit(cache=True)
def _world_vertices(verts, px, py, angle):
    """局所座標の頂点を世界座標へ移す。"""
    cos = math.cos(angle)
    sin = math.sin(angle)
    n = verts.shape[0]
    out = np.empty((n, 2), np.float64)
    for i in range(n):
        x = verts[i, 0]
        y = verts[i, 1]
        out[i, 0] = px + x * cos - y * sin
        out[i, 1] = py + x * sin + y * cos
    return out


@njit(cache=True)
def _face_normals(v):
    """各辺の外向き法線（JS 版と同じ «(ey, -ex) を正規化» の向き）。"""
    n = v.shape[0]
    out = np.empty((n, 2), np.float64)
    for i in range(n):
        j = (i + 1) % n
        ex = v[j, 0] - v[i, 0]
        ey = v[j, 1] - v[i, 1]
        length = math.hypot(ex, ey)
        if length == 0.0:
            length = 1.0
        out[i, 0] = ey / length
        out[i, 1] = -ex / length
    return out


@njit(cache=True)
def _project(v, ax, ay):
    lo = np.inf
    hi = -np.inf
    for i in range(v.shape[0]):
        d = v[i, 0] * ax + v[i, 1] * ay
        if d < lo:
            lo = d
        if d > hi:
            hi = d
    return lo, hi


@njit(cache=True)
def _circle_circle(pax, pay, ra, pbx, pby, rb):
    out = np.zeros(9, np.float64)
    dx = pbx - pax
    dy = pby - pay
    distance = math.hypot(dx, dy)
    radius_sum = ra + rb
    if distance >= radius_sum:
        return out
    if distance < EPSILON:
        nx, ny = 0.0, -1.0
    else:
        nx, ny = dx / distance, dy / distance
    penetration = radius_sum - distance
    out[0] = 1.0
    out[1] = nx
    out[2] = ny
    out[3] = penetration
    out[4] = 1.0
    out[5] = pax + nx * (ra - penetration / 2)
    out[6] = pay + ny * (ra - penetration / 2)
    return out


@njit(cache=True)
def _circle_polygon(cx, cy, radius, pverts, ppx, ppy, pangle, flipped):
    """円と多角形。`flipped` が真なら «多角形が A» として返します。"""
    out = np.zeros(9, np.float64)
    v = _world_vertices(pverts, ppx, ppy, pangle)
    normals = _face_normals(v)
    n = v.shape[0]

    best_distance = -np.inf
    best_index = 0
    for i in range(n):
        d = normals[i, 0] * (cx - v[i, 0]) + normals[i, 1] * (cy - v[i, 1])
        if d > radius:
            return out
        if d > best_distance:
            best_distance = d
            best_index = i

    ax = v[best_index, 0]
    ay = v[best_index, 1]
    bx = v[(best_index + 1) % n, 0]
    by = v[(best_index + 1) % n, 1]

    if best_distance < EPSILON:
        # 円の中心が多角形の内側にある。
        nx = normals[best_index, 0]
        ny = normals[best_index, 1]
        penetration = radius - best_distance
        contact_x = cx - nx * radius
        contact_y = cy - ny * radius
    else:
        ex = bx - ax
        ey = by - ay
        length_sq = ex * ex + ey * ey
        if length_sq == 0.0:
            length_sq = 1.0
        t = ((cx - ax) * ex + (cy - ay) * ey) / length_sq
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        closest_x = ax + ex * t
        closest_y = ay + ey * t
        dx = cx - closest_x
        dy = cy - closest_y
        distance = math.hypot(dx, dy)
        if distance > radius:
            return out
        if distance < EPSILON:
            nx = normals[best_index, 0]
            ny = normals[best_index, 1]
        else:
            nx = dx / distance
            ny = dy / distance
        penetration = radius - distance
        contact_x = closest_x
        contact_y = closest_y

    out[0] = 1.0
    # 法線は必ず «A から B へ»。円が A のときだけ向きを返します。
    if flipped:
        out[1] = nx
        out[2] = ny
    else:
        out[1] = -nx
        out[2] = -ny
    out[3] = penetration
    out[4] = 1.0
    out[5] = contact_x
    out[6] = contact_y
    return out


@njit(cache=True)
def _clip(p0x, p0y, p1x, p1y, dx, dy, offset, buf):
    """半平面で線分を切る。`buf` に最大 2 点書き、書いた数を返します。"""
    d0 = p0x * dx + p0y * dy - offset
    d1 = p1x * dx + p1y * dy - offset
    count = 0
    if d0 <= 0.0:
        buf[0] = p0x
        buf[1] = p0y
        count = 1
    if d1 <= 0.0 and count < 2:
        buf[count * 2] = p1x
        buf[count * 2 + 1] = p1y
        count += 1
    if d0 * d1 < 0.0 and count < 2:
        t = d0 / (d0 - d1)
        buf[count * 2] = p0x + (p1x - p0x) * t
        buf[count * 2 + 1] = p0y + (p1y - p0y) * t
        count += 1
    return count


@njit(cache=True)
def _polygon_polygon(averts, apx, apy, aangle, bverts, bpx, bpy, bangle):
    out = np.zeros(9, np.float64)
    va = _world_vertices(averts, apx, apy, aangle)
    vb = _world_vertices(bverts, bpx, bpy, bangle)
    na = _face_normals(va)
    nb = _face_normals(vb)

    min_overlap = np.inf
    axis_x = 0.0
    axis_y = 0.0
    has_axis = False
    from_a = True

    for i in range(na.shape[0]):
        pa_lo, pa_hi = _project(va, na[i, 0], na[i, 1])
        pb_lo, pb_hi = _project(vb, na[i, 0], na[i, 1])
        overlap = min(pa_hi - pb_lo, pb_hi - pa_lo)
        if overlap <= 0.0:
            return out
        if overlap < min_overlap:
            min_overlap = overlap
            axis_x = na[i, 0]
            axis_y = na[i, 1]
            has_axis = True
            from_a = True
    for i in range(nb.shape[0]):
        pa_lo, pa_hi = _project(va, nb[i, 0], nb[i, 1])
        pb_lo, pb_hi = _project(vb, nb[i, 0], nb[i, 1])
        overlap = min(pa_hi - pb_lo, pb_hi - pa_lo)
        if overlap <= 0.0:
            return out
        if overlap < min_overlap:
            min_overlap = overlap
            axis_x = nb[i, 0]
            axis_y = nb[i, 1]
            has_axis = True
            from_a = False
    if not has_axis:
        return out

    # 法線を A から B へ向ける。
    cdx = bpx - apx
    cdy = bpy - apy
    nx = axis_x
    ny = axis_y
    if nx * cdx + ny * cdy < 0.0:
        nx = -nx
        ny = -ny

    if from_a:
        ref = va
        inc = vb
        rnx = nx
        rny = ny
    else:
        ref = vb
        inc = va
        rnx = -nx
        rny = -ny

    ref_normals = _face_normals(ref)
    ref_index = 0
    best = -np.inf
    for i in range(ref_normals.shape[0]):
        d = ref_normals[i, 0] * rnx + ref_normals[i, 1] * rny
        if d > best:
            best = d
            ref_index = i
    r0x = ref[ref_index, 0]
    r0y = ref[ref_index, 1]
    r1x = ref[(ref_index + 1) % ref.shape[0], 0]
    r1y = ref[(ref_index + 1) % ref.shape[0], 1]

    inc_normals = _face_normals(inc)
    inc_index = 0
    worst = np.inf
    for i in range(inc_normals.shape[0]):
        d = inc_normals[i, 0] * rnx + inc_normals[i, 1] * rny
        if d < worst:
            worst = d
            inc_index = i
    i0x = inc[inc_index, 0]
    i0y = inc[inc_index, 1]
    i1x = inc[(inc_index + 1) % inc.shape[0], 0]
    i1y = inc[(inc_index + 1) % inc.shape[0], 1]

    rdx = r1x - r0x
    rdy = r1y - r0y
    ref_length = math.hypot(rdx, rdy)
    if ref_length == 0.0:
        ref_length = 1.0
    tx = rdx / ref_length
    ty = rdy / ref_length

    out[0] = 1.0
    out[1] = nx
    out[2] = ny
    out[3] = min_overlap

    buf = np.empty(4, np.float64)
    count = _clip(i0x, i0y, i1x, i1y, -tx, -ty, -(r0x * tx + r0y * ty), buf)
    if count < 2:
        out[4] = 1.0
        out[5] = (i0x + i1x) / 2
        out[6] = (i0y + i1y) / 2
        return out
    i0x, i0y, i1x, i1y = buf[0], buf[1], buf[2], buf[3]
    count = _clip(i0x, i0y, i1x, i1y, tx, ty, r1x * tx + r1y * ty, buf)
    if count < 2:
        out[4] = 1.0
        out[5] = (i0x + i1x) / 2
        out[6] = (i0y + i1y) / 2
        return out

    ref_offset = r0x * rnx + r0y * rny
    kept = 0
    for k in range(2):
        px = buf[k * 2]
        py = buf[k * 2 + 1]
        separation = px * rnx + py * rny - ref_offset
        if separation <= 0.5:
            out[5 + kept * 2] = px
            out[6 + kept * 2] = py
            kept += 1
    if kept == 0:
        out[4] = 1.0
        out[5] = (buf[0] + buf[2]) / 2
        out[6] = (buf[1] + buf[3]) / 2
        return out
    out[4] = float(kept)
    return out


# ── Python から使う面 ─────────────────────────────────────────────

_EMPTY = np.zeros((0, 2), np.float64)


def collide(a, b):
    """2 つの剛体の接触を調べる。当たっていなければ `None`。

    :returns: ``{"bodyA", "bodyB", "normal", "penetration", "contacts"}``
        法線は **必ず A から B へ**向きます。
    """
    ta = a.shape.type
    tb = b.shape.type
    if ta == "circle" and tb == "circle":
        raw = _circle_circle(a.position.x, a.position.y, a.shape.radius, b.position.x, b.position.y, b.shape.radius)
    elif ta == "circle" and tb == "polygon":
        raw = _circle_polygon(
            a.position.x, a.position.y, a.shape.radius, b.shape.vertices, b.position.x, b.position.y, b.angle, False
        )
    elif ta == "polygon" and tb == "circle":
        raw = _circle_polygon(
            b.position.x, b.position.y, b.shape.radius, a.shape.vertices, a.position.x, a.position.y, a.angle, True
        )
    elif ta == "polygon" and tb == "polygon":
        raw = _polygon_polygon(
            a.shape.vertices, a.position.x, a.position.y, a.angle,
            b.shape.vertices, b.position.x, b.position.y, b.angle,
        )
    else:
        return None

    if raw[0] == 0.0:
        return None
    count = int(raw[4])
    contacts = [(float(raw[5 + i * 2]), float(raw[6 + i * 2])) for i in range(count)]
    return {
        "bodyA": a,
        "bodyB": b,
        "normal": (float(raw[1]), float(raw[2])),
        "penetration": float(raw[3]),
        "contacts": contacts,
    }


def aabb_overlap(a, b) -> bool:
    """外接矩形どうしが重なるか。``(minX, minY, maxX, maxY)`` を渡します。"""
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]
