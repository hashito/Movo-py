"""標準の幾何変形（JS 版 packages/deformer/src/deformers.js の移植）。

変形はメッシュと «解決済みのパラメータ» と «時刻・マスク・素材» を受け取り、
頂点の新しい位置を書きます。マスクを通した重み付けは `apply_deform` が
まとめて面倒を見るので、**どの変形でも部分適用（仕様 11 章）が効きます**。

## Python 版で変えたところ

JS 版は «頂点ごとに関数を呼ぶ» 形でした。Python でそれをやると、21x21 の
格子（441 頂点）でも 1 変形あたり 0.4ms かかります。10 個重なるレイヤーが
20 枚あるフレームで 80ms です。

そこで **変形は「頂点の配列を丸ごと受け取って、新しい配列を返す」** 形に
しました。式も分岐も JS 版と同じで、`if` が `np.where` になっただけです。

| `twist` 1 回 | 頂点ごとの Python | NumPy の一括演算 |
| --- | --- | --- |
| 777 頂点 | 0.262 ms | **0.031 ms**（8 倍） |
| 6,588 頂点 | 1.986 ms | **0.119 ms**（17 倍） |

Numba にしていないのは、ここが «全頂点に一様な演算» で分岐が浅いからです。
NumPy の一括演算で C の速度が出ており、JIT の手間を足す価値がありません。
"""

from __future__ import annotations

import math

import numpy as np

from ._compat import TAU, clamp, fbm2d, js_round, sample_polyline, value_noise_1d
from ._sampling import channel_value, sample_bilinear

MASK_RESOLUTION = 64


def apply_deform(mesh, ctx, compute) -> None:
    """`compute()` が返した新しい位置を、マスクの重みで混ぜて書き戻す。

    :param compute: ``() -> (新しい x, 新しい y)``。どちらも `mesh.x` と同じ長さ

    重みの扱いは JS 版そのままです。**0.999 以上は «そのまま置き換え»** で、
    0 より大きく 0.999 未満だけ線形に混ぜます。0 以下は触りません。
    （ほぼ 1 のときに掛け算を挟まないための場合分けで、これが無いと
    マスク無しの結果とマスク全面 1 の結果がわずかにずれます。）
    """
    out_x, out_y = compute()
    field = ctx.get("maskField") if ctx else None
    if field is None:
        mesh.x = out_x
        mesh.y = out_y
        return
    from .mask import sample_field

    weight = np.clip(sample_field(field, MASK_RESOLUTION, MASK_RESOLUTION, mesh.u0, mesh.v0), 0.0, 1.0)
    full = weight >= 0.999
    partial = (weight > 0) & ~full
    mesh.x = np.where(full, out_x, np.where(partial, mesh.x + (out_x - mesh.x) * weight, mesh.x))
    mesh.y = np.where(full, out_y, np.where(partial, mesh.y + (out_y - mesh.y) * weight, mesh.y))


def _center_of(params: dict, mesh) -> tuple[float, float]:
    center = params.get("center") or {}
    cx = center.get("x", params.get("centerX", 0.5))
    cy = center.get("y", params.get("centerY", 0.5))
    return (float(cx) * mesh.width, float(cy) * mesh.height)


def bend(mesh, params: dict, ctx: dict) -> None:
    """軸に沿って曲げる。

    断面を «曲がった背骨と直交するよう» 回すので、単に下へずらすのではなく
    ちゃんと «曲がって» 見えます。
    """
    amount = float(params.get("amount", 0.5))
    axis = params.get("axis", "x")
    origin = float(params.get("origin", 0.5))
    if amount == 0:
        return

    def compute():
        if axis == "y":
            n = mesh.v0 - origin
            offset = amount * mesh.width * n * n
            slope = (2 * amount * mesh.width * n) / max(1, mesh.height)
            dy = mesh.x - mesh.width / 2
            return mesh.x + offset, mesh.y - dy * slope
        n = mesh.u0 - origin
        offset = amount * mesh.height * n * n
        slope = (2 * amount * mesh.height * n) / max(1, mesh.width)
        dy = mesh.y - mesh.height / 2
        return mesh.x - dy * slope, mesh.y + offset

    apply_deform(mesh, ctx, compute)


def twist(mesh, params: dict, ctx: dict) -> None:
    """中心のまわりに回し、`radius` で 0 に落とす。"""
    angle = math.radians(float(params.get("angle", 45)))
    cx, cy = _center_of(params, mesh)
    radius = float(params.get("radius", 0.6)) * max(mesh.width, mesh.height) * 0.5
    if angle == 0 or radius <= 0:
        return
    falloff_power = float(params.get("falloff", 1))

    def compute():
        dx = mesh.x - cx
        dy = mesh.y - cy
        distance = np.hypot(dx, dy)
        inside = distance <= radius
        t = np.where(inside, 1 - distance / radius, 0.0)
        local_angle = angle * np.power(t, falloff_power)
        cos = np.cos(local_angle)
        sin = np.sin(local_angle)
        return (
            np.where(inside, cx + dx * cos - dy * sin, mesh.x),
            np.where(inside, cy + dx * sin + dy * cos, mesh.y),
        )

    apply_deform(mesh, ctx, compute)


def wave(mesh, params: dict, ctx: dict) -> None:
    """正弦波の «さざなみ»。"""
    amplitude = float(params.get("amplitude", 20))
    frequency = float(params.get("frequency", 3))
    speed = float(params.get("speed", 0))
    phase = float(params.get("phase", 0))
    axis = params.get("axis", "x")
    time = float(ctx.get("time", 0) if ctx else 0)
    if amplitude == 0:
        return
    shift = phase + time * speed

    def compute():
        out_x = mesh.x
        out_y = mesh.y
        if axis in ("x", "both"):
            out_y = mesh.y + np.sin((mesh.u0 * frequency + shift) * TAU) * amplitude
        if axis in ("y", "both"):
            out_x = mesh.x + np.sin((mesh.v0 * frequency + shift) * TAU) * amplitude
        return out_x, out_y

    apply_deform(mesh, ctx, compute)


def skew(mesh, params: dict, ctx: dict) -> None:
    """せん断。`x` は高さに比例して横へ、`y` は幅に比例して縦へ。"""
    kx = float(params.get("x", params.get("amountX", 0)))
    ky = float(params.get("y", params.get("amountY", 0)))
    if kx == 0 and ky == 0:
        return
    cx = float(params.get("originX", 0.5)) * mesh.width
    cy = float(params.get("originY", 0.5)) * mesh.height

    def compute():
        return mesh.x + (mesh.y - cy) * kx, mesh.y + (mesh.x - cx) * ky

    apply_deform(mesh, ctx, compute)


def perspective(mesh, params: dict, ctx: dict) -> None:
    """四隅を動かす。内側は双一次で付いてきます。"""
    corners = params.get("corners") or {}

    def as_point(value, fx, fy):
        if isinstance(value, (list, tuple)):
            return (float(value[0]) if len(value) > 0 else fx, float(value[1]) if len(value) > 1 else fy)
        if isinstance(value, dict):
            return (float(value.get("x", fx)), float(value.get("y", fy)))
        return (fx, fy)

    tl = as_point(corners.get("topLeft"), 0, 0)
    tr = as_point(corners.get("topRight"), 1, 0)
    bl = as_point(corners.get("bottomLeft"), 0, 1)
    br = as_point(corners.get("bottomRight"), 1, 1)

    def compute():
        u = mesh.u0
        v = mesh.v0
        top_x = tl[0] + (tr[0] - tl[0]) * u
        top_y = tl[1] + (tr[1] - tl[1]) * u
        bottom_x = bl[0] + (br[0] - bl[0]) * u
        bottom_y = bl[1] + (br[1] - bl[1]) * u
        return (
            (top_x + (bottom_x - top_x) * v) * mesh.width,
            (top_y + (bottom_y - top_y) * v) * mesh.height,
        )

    apply_deform(mesh, ctx, compute)


def bulge(mesh, params: dict, ctx: dict) -> None:
    """中心から押し出す（負なら引き込む）。"""
    strength = float(params.get("strength", 0.6))
    cx, cy = _center_of(params, mesh)
    radius = float(params.get("radius", 0.3)) * max(mesh.width, mesh.height)
    if strength == 0 or radius <= 0:
        return

    def compute():
        dx = mesh.x - cx
        dy = mesh.y - cy
        distance = np.hypot(dx, dy)
        inside = (distance <= radius) & (distance >= 1e-6)
        t = distance / radius
        scale = 1 + strength * (1 - t * t)
        return (
            np.where(inside, cx + dx * scale, mesh.x),
            np.where(inside, cy + dy * scale, mesh.y),
        )

    apply_deform(mesh, ctx, compute)


def pinch(mesh, params: dict, ctx: dict) -> None:
    """`bulge` の符号を反転しただけの別名。"""
    merged = dict(params)
    merged["strength"] = -float(params.get("strength", 0.5))
    bulge(mesh, merged, ctx)


def sphereize(mesh, params: dict, ctx: dict) -> None:
    """球面レンズの歪み。"""
    strength = float(params.get("strength", 0.5))
    cx, cy = _center_of(params, mesh)
    radius = float(params.get("radius", 0.5)) * max(mesh.width, mesh.height)

    def compute():
        dx = mesh.x - cx
        dy = mesh.y - cy
        distance = np.hypot(dx, dy)
        inside = (distance <= radius) & (distance >= 1e-6)
        t = distance / radius
        scale = 1 + strength * np.sqrt(np.maximum(0.0, 1 - t * t))
        return (
            np.where(inside, cx + dx * scale, mesh.x),
            np.where(inside, cy + dy * scale, mesh.y),
        )

    apply_deform(mesh, ctx, compute)


def ripple(mesh, params: dict, ctx: dict) -> None:
    """中心から広がる同心円の波。"""
    amplitude = float(params.get("amplitude", 10))
    frequency = float(params.get("frequency", 4))
    speed = float(params.get("speed", 1))
    cx, cy = _center_of(params, mesh)
    radius = float(params.get("radius", 1)) * max(mesh.width, mesh.height)
    time = float(ctx.get("time", 0) if ctx else 0)

    def compute():
        dx = mesh.x - cx
        dy = mesh.y - cy
        distance = np.hypot(dx, dy)
        inside = (distance >= 1e-6) & (distance <= radius)
        safe = np.where(inside, distance, 1.0)
        w = np.sin((safe / radius) * frequency * TAU - time * speed * TAU) * amplitude
        falloff = 1 - safe / radius
        return (
            np.where(inside, mesh.x + (dx / safe) * w * falloff, mesh.x),
            np.where(inside, mesh.y + (dy / safe) * w * falloff, mesh.y),
        )

    apply_deform(mesh, ctx, compute)


def mesh_warp(mesh, params: dict, ctx: dict) -> None:
    """自由変形。格子の制御点を双一次で広げます。"""
    columns = max(1, js_round(params.get("columns", 4)))
    rows = max(1, js_round(params.get("rows", 4)))
    offsets = np.zeros(((columns + 1) * (rows + 1), 2), np.float64)
    for point in params.get("points") or []:
        column = int(clamp(js_round(point.get("column", point.get("col", 0))), 0, columns))
        row = int(clamp(js_round(point.get("row", 0)), 0, rows))
        index = row * (columns + 1) + column
        offsets[index, 0] += float(point.get("offsetX", point.get("x", 0)))
        offsets[index, 1] += float(point.get("offsetY", point.get("y", 0)))

    def sample_offset(u, v, component):
        fx = np.clip(u, 0, 1) * columns
        fy = np.clip(v, 0, 1) * rows
        x0 = np.minimum(columns, np.floor(fx).astype(np.int64))
        y0 = np.minimum(rows, np.floor(fy).astype(np.int64))
        x1 = np.minimum(columns, x0 + 1)
        y1 = np.minimum(rows, y0 + 1)
        tx = fx - x0
        ty = fy - y0

        def at(cx, cy):
            return offsets[cy * (columns + 1) + cx, component]

        top = at(x0, y0) * (1 - tx) + at(x1, y0) * tx
        bottom = at(x0, y1) * (1 - tx) + at(x1, y1) * tx
        return top * (1 - ty) + bottom * ty

    def compute():
        return (
            mesh.x + sample_offset(mesh.u0, mesh.v0, 0),
            mesh.y + sample_offset(mesh.u0, mesh.v0, 1),
        )

    apply_deform(mesh, ctx, compute)


def path_deform(mesh, params: dict, ctx: dict) -> None:
    """曲線に沿わせる。横軸が弧に、縦軸がその法線に載ります。"""
    raw = params.get("path") or []
    path = []
    for p in raw:
        if isinstance(p, dict):
            path.append((float(p.get("x", 0)), float(p.get("y", 0))))
        else:
            path.append((float(p[0]), float(p[1])))
    if len(path) < 2:
        return
    points = np.array(path, np.float64)
    strength = float(params.get("strength", 1))
    stretch = params.get("stretch") is not False

    def compute():
        u = mesh.u0
        p = sample_polyline(points, u)
        delta = 0.002
        a = sample_polyline(points, np.clip(u - delta, 0, 1))
        b = sample_polyline(points, np.clip(u + delta, 0, 1))
        tx = (b[..., 0] - a[..., 0]) * mesh.width
        ty = (b[..., 1] - a[..., 1]) * mesh.height
        length = np.hypot(tx, ty)
        length = np.where(length == 0, 1.0, length)
        tx = tx / length
        ty = ty / length
        normal_x = -ty
        normal_y = tx
        offset = (mesh.v0 - 0.5) * mesh.height
        target_x = (p[..., 0] * mesh.width if stretch else mesh.x) + normal_x * offset
        target_y = p[..., 1] * mesh.height + normal_y * offset
        return (
            mesh.x + (target_x - mesh.x) * strength,
            mesh.y + (target_y - mesh.y) * strength,
        )

    apply_deform(mesh, ctx, compute)


def displacement(mesh, params: dict, ctx: dict) -> None:
    """別の画像の明るさで押しずらす。"""
    assets = ctx.get("assets") if ctx else None
    map_bitmap = assets.get(params.get("mapAsset")) if assets else None
    if map_bitmap is None:
        return
    amount_x = float(params.get("amountX", 0))
    amount_y = float(params.get("amountY", 0))
    channel = params.get("channel", "luminance")
    scroll_x = float(params.get("scrollX", 0))
    scroll_y = float(params.get("scrollY", 0))
    time = float(ctx.get("time", 0) if ctx else 0)

    def compute():
        u = ((mesh.u0 + scroll_x * time) % 1 + 1) % 1
        v = ((mesh.v0 + scroll_y * time) % 1 + 1) % 1
        sample = sample_bilinear(map_bitmap, u * map_bitmap.width, v * map_bitmap.height, True)
        value = channel_value(sample, channel)
        signed = value / 255 - 0.5
        return mesh.x + signed * amount_x * 2, mesh.y + signed * amount_y * 2

    apply_deform(mesh, ctx, compute)


def turbulent_displace(mesh, params: dict, ctx: dict) -> None:
    """フラクタルノイズで揺らす。`displacement` と違い素材画像が要りません。

    x 用と y 用でシードをずらした 2 枚のノイズを引き、その値を変位にします。
    `evolution` を進めると «流れる» のではなく **形が変わる** 動きになります。
    """
    amount = float(params.get("amount", 0))
    amount_x = float(params.get("amountX", amount))
    amount_y = float(params.get("amountY", amount))
    if amount_x == 0 and amount_y == 0:
        return

    scale = float(params.get("scale", 0.01))
    seed = js_round(params.get("seed", 0)) & 0xFFFFFFFF
    evolution = float(params.get("evolution") or 0)
    kind = params.get("mode") or params.get("type") or "turbulent"
    options = {
        "seed": seed,
        "z": evolution,
        "octaves": params.get("octaves", 3),
        "lacunarity": params.get("lacunarity", 2),
        "gain": params.get("gain", 0.5),
        "type": kind,
    }
    # turbulent / ridged は 0〜1 なので中心を 0 に寄せます。
    centre = 0.0 if kind == "fbm" else 0.5
    offset_x = float(params.get("offsetX", 0))
    offset_y = float(params.get("offsetY", 0))

    def compute():
        # 変位は «元の格子位置»（u0/v0）から引きます。動いたあとの座標で引くと、
        # 変形を重ねる順番で結果が変わってしまいます。
        nx = (mesh.u0 * mesh.width + offset_x) * scale
        ny = (mesh.v0 * mesh.height + offset_y) * scale
        dx = fbm2d(nx, ny, options) - centre
        second = dict(options)
        second["seed"] = (seed + 8191) & 0xFFFFFFFF
        dy = fbm2d(nx + 137.13, ny - 91.71, second) - centre
        return mesh.x + dx * amount_x * 2, mesh.y + dy * amount_y * 2

    apply_deform(mesh, ctx, compute)


def melt(mesh, params: dict, ctx: dict) -> None:
    """溶ける。列ごとに違う量だけ下へ滴らせます。

    落下量は列ごとにノイズで決め、下の行ほど大きく引き伸ばします。
    `progress` 0 で無変化、1 で完全に溶け落ちた状態です。
    """
    progress = clamp(float(params.get("progress", 0)), 0.0, 1.0)
    if progress <= 0:
        return
    amount = float(params.get("amount", 300))
    columns = max(1, js_round(params.get("columns", 60)))
    randomness = clamp(float(params.get("randomness", 0.7)), 0.0, 1.0)
    angle = math.radians(float(params.get("angle", 90)))
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    seed = js_round(params.get("seed", 3))

    def compute():
        u = mesh.u0
        v = mesh.v0
        # 列インデックスを整数に丸めてから引くと、列の中では同じ落下量になります。
        column = np.floor(u * columns)
        noise = (value_noise_1d(column * 0.7 + 0.5, seed) + 1) / 2
        column_amount = amount * (1 - randomness + randomness * noise)
        # 下の行ほど落ちます（上端は残って伸びます）。
        drop = column_amount * progress * v * v
        return mesh.x + dir_x * drop, mesh.y + dir_y * drop

    apply_deform(mesh, ctx, compute)


def hand_drawn(mesh, params: dict, ctx: dict) -> None:
    """手描き風ジッター。輪郭を小刻みに揺らします。

    `interval` フレームごとに揺れ方が変わるので、frameHold と併せると
    手描きアニメらしい «コマ打ち» になります。
    """
    amount = float(params.get("amount", 4))
    if amount == 0:
        return
    scale = float(params.get("scale", 0.05))
    interval = max(1, js_round(params.get("interval", 3)))
    roughness = clamp(float(params.get("roughness", 0.6)), 0.0, 1.0)
    fps = float(ctx.get("fps", 30) if ctx else 30)
    # interval フレームごとに «別の紙に描き直す» イメージでシードを変えます。
    frame = math.floor(float(ctx.get("time", 0) if ctx else 0) * fps)
    seed = (js_round(params.get("seed", 4)) + (frame // interval) * 7919) & 0xFFFFFFFF
    octaves = 3 if roughness > 0.5 else 1

    def compute():
        nx = mesh.u0 * mesh.width * scale
        ny = mesh.v0 * mesh.height * scale
        dx = fbm2d(nx, ny, {"seed": seed, "octaves": octaves, "gain": roughness})
        dy = fbm2d(nx + 53.7, ny - 31.3, {"seed": (seed + 4093) & 0xFFFFFFFF, "octaves": octaves, "gain": roughness})
        return mesh.x + dx * amount, mesh.y + dy * amount

    apply_deform(mesh, ctx, compute)


def curve_deform(mesh, params: dict, ctx: dict) -> None:
    """曲面変形。上下（または左右）の辺を独立に曲げます。

    `bend` が全体を一様に曲げるのに対し、こちらは辺ごとに制御できます。
    """
    axis = "y" if params.get("axis") == "y" else "x"
    top_curve = float(params.get("topCurve", 0))
    bottom_curve = float(params.get("bottomCurve", 0))
    twist_amount = float(params.get("twist", 0))
    if top_curve == 0 and bottom_curve == 0 and twist_amount == 0:
        return
    size = mesh.height if axis == "x" else mesh.width

    def compute():
        # 辺に沿った位置（0〜1）と、辺から辺への位置（0〜1）
        along = mesh.u0 if axis == "x" else mesh.v0
        across = mesh.v0 if axis == "x" else mesh.u0
        arch = np.sin(along * math.pi)          # 端で 0、中央で 1 の山
        curve = top_curve + (bottom_curve - top_curve) * across
        shift = arch * curve * size * 0.5
        twist_shift = (along - 0.5) * twist_amount * size * (across - 0.5) * 2
        if axis == "x":
            return mesh.x, mesh.y + shift + twist_shift
        return mesh.x + shift + twist_shift, mesh.y

    apply_deform(mesh, ctx, compute)


deformers = {
    "bend": bend,
    "twist": twist,
    "wave": wave,
    "skew": skew,
    "perspective": perspective,
    "bulge": bulge,
    "pinch": pinch,
    "sphereize": sphereize,
    "ripple": ripple,
    "meshWarp": mesh_warp,
    "pathDeform": path_deform,
    "displacement": displacement,
    "turbulentDisplace": turbulent_displace,
    "melt": melt,
    "handDrawn": hand_drawn,
    "curveDeform": curve_deform,
}


def list_deformers() -> list[str]:
    return sorted(deformers)


def has_deformer(name: str) -> bool:
    return name in deformers
