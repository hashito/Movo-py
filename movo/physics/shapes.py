"""当たり判定の形（JS 版 packages/physics/src/shapes.js の移植）。

**内部にあるのは円と凸多角形の 2 つだけです。** 矩形・カプセル・メッシュ・
アルファ輪郭はどれもこの 2 つに畳んでから使います。狭域判定（narrow phase）を
小さく保つための JS 版の設計をそのまま持ってきました。

## Python 版で変えたところ

頂点は `{x, y}` の配列ではなく **`(N, 2)` の `float64` 配列**で持ちます。
Numba のカーネルへそのまま渡せるようにするためで、値と順番は JS 版と
1 つも変わりません（凸包の並びも同じです）。

アルファ輪郭の «不透明画素を探す» 走査だけは NumPy の一括演算にしてあります。
1280x720 を純 Python で舐めると 1 枚 0.7 秒（README の «純 Python の 1 パス»
と同じ数字）で、ここだけで書き出しが止まって見えるためです。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Shape:
    """円か凸多角形。`type` で分かれます。

    :param type: ``"circle"`` か ``"polygon"``
    :param radius: 円のときの半径
    :param vertices: 多角形のときの頂点 ``(N, 2)``。重心を原点に寄せてあります
    :param centroid: 元の座標系での重心（`polygon_shape` だけが埋めます）
    """

    type: str = "circle"
    radius: float = 25.0
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float64))
    centroid: tuple[float, float] = (0.0, 0.0)


def _as_points(values) -> list[tuple[float, float]]:
    """`[x, y]` でも `{x, y}` でも受け取れるようにする（JS 版と同じ寛容さ）。"""
    out: list[tuple[float, float]] = []
    for v in values:
        if isinstance(v, dict):
            out.append((float(v.get("x", 0.0)), float(v.get("y", 0.0))))
        elif hasattr(v, "x") and hasattr(v, "y"):
            out.append((float(v.x), float(v.y)))
        else:
            out.append((float(v[0]), float(v[1])))
    return out


def circle_shape(radius: float) -> Shape:
    """円。半径 0 以下は 0.01 に丸めます（0 半径は当たり判定が壊れるため）。"""
    return Shape(type="circle", radius=max(0.01, float(radius)))


def polygon_shape(vertices) -> Shape:
    """点列から凸多角形を作る。凸包を取り、重心を原点へ寄せます。"""
    points = _as_points(vertices)
    hull = convex_hull(points)
    cx, cy = polygon_centroid(hull)
    verts = np.array([(p[0] - cx, p[1] - cy) for p in hull], np.float64)
    return Shape(type="polygon", vertices=verts, centroid=(cx, cy))


def rectangle_shape(width: float, height: float) -> Shape:
    """矩形。頂点の並びは JS 版と同じ «左上→右上→右下→左下» です。"""
    hw = max(0.01, float(width) / 2)
    hh = max(0.01, float(height) / 2)
    verts = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)], np.float64)
    return Shape(type="polygon", vertices=verts, centroid=(0.0, 0.0))


def capsule_shape(length: float, radius: float) -> Shape:
    """カプセル。両端を 8 分割した «角の丸い多角形» で近似します（JS 版と同じ 16 分割）。"""
    half = max(0.01, float(length) / 2)
    r = max(0.01, float(radius))
    steps = 8
    verts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        angle = -math.pi / 2 + (i / steps) * math.pi
        verts.append((half + math.cos(angle) * r, math.sin(angle) * r))
    for i in range(steps + 1):
        angle = math.pi / 2 + (i / steps) * math.pi
        verts.append((-half + math.cos(angle) * r, math.sin(angle) * r))
    return Shape(type="polygon", vertices=np.array(verts, np.float64), centroid=(0.0, 0.0))


def alpha_outline_shape(bitmap, options: dict | None = None) -> Shape:
    """画像の不透明な部分から凸多角形を作る。

    **凹んだ形は少し «太って» 当たります。** 凸包しか作らないからです。
    完全な凸分解より遥かに安く、MV の用途では困らない、という JS 版の
    判断をそのまま引き継いでいます。

    :param bitmap: `movo.core.bitmap.Bitmap`（`data` は ``(h, w, 4)`` の uint8）
    :param options: ``threshold`` / ``simplify`` / ``width`` / ``height``
    """
    options = options or {}
    threshold = options.get("threshold")
    threshold = 16 if threshold is None else float(threshold)
    width = options.get("width")
    height = options.get("height")
    scale_x = (bitmap.width if width is None else float(width)) / bitmap.width
    scale_y = (bitmap.height if height is None else float(height)) / bitmap.height
    step = max(1, round(min(bitmap.width, bitmap.height) / 48))

    # ここは «行ごと・列ごとの端» を探すだけなので NumPy の一括演算が効きます。
    # 純 Python の二重ループだと 1280x720 で 0.7 秒かかりました。
    alpha = bitmap.data[..., 3]
    opaque = alpha > threshold
    points: list[tuple[float, float]] = []

    rows = np.arange(0, bitmap.height, step)
    row_any = opaque[rows].any(axis=1)
    row_left = opaque[rows].argmax(axis=1)
    row_right = bitmap.width - 1 - opaque[rows][:, ::-1].argmax(axis=1)
    for k, y in enumerate(rows):
        if row_any[k]:
            points.append((float(row_left[k]), float(y)))
            points.append((float(row_right[k]), float(y)))

    cols = np.arange(0, bitmap.width, step)
    col = opaque[:, cols]
    col_any = col.any(axis=0)
    col_top = col.argmax(axis=0)
    col_bottom = bitmap.height - 1 - col[::-1].argmax(axis=0)
    for k, x in enumerate(cols):
        if col_any[k]:
            points.append((float(x), float(col_top[k])))
            points.append((float(x), float(col_bottom[k])))

    if len(points) < 3:
        # 不透明な画素が 1 つも無い。矩形に落として «描いても消えない» ようにします。
        return rectangle_shape(
            bitmap.width if width is None else width,
            bitmap.height if height is None else height,
        )

    centered = [
        ((p[0] - bitmap.width / 2) * scale_x, (p[1] - bitmap.height / 2) * scale_y)
        for p in points
    ]
    hull = convex_hull(centered)
    simplify = options.get("simplify")
    if simplify:
        hull = simplify_polygon(hull, float(simplify))
    return Shape(type="polygon", vertices=np.array(hull, np.float64), centroid=(0.0, 0.0))


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew の monotone chain。JS 版と同じ並び（下側→上側）で返します。"""
    if len(points) < 3:
        return list(points)
    ordered = sorted(points, key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    upper.pop()
    lower.pop()
    hull = lower + upper
    return hull if len(hull) >= 3 else list(points)


def simplify_polygon(points: list[tuple[float, float]], tolerance: float):
    """外形の変化が `tolerance` 未満の頂点を落とす。"""
    if len(points) <= 4 or tolerance <= 0:
        return points
    out = []
    n = len(points)
    for i in range(n):
        prev = points[(i - 1 + n) % n]
        current = points[i]
        nxt = points[(i + 1) % n]
        area = abs(
            (current[0] - prev[0]) * (nxt[1] - prev[1])
            - (nxt[0] - prev[0]) * (current[1] - prev[1])
        ) / 2
        if area > tolerance:
            out.append(current)
    return out if len(out) >= 3 else points


def polygon_centroid(points) -> tuple[float, float]:
    """多角形の重心。面積がほぼ 0 のときは点の平均に落とします。"""
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        cross = a[0] * b[1] - b[0] * a[1]
        area += cross
        cx += (a[0] + b[0]) * cross
        cy += (a[1] + b[1]) * cross
    if abs(area) < 1e-9:
        k = n or 1
        return (sum(p[0] for p in points) / k, sum(p[1] for p in points) / k)
    area *= 0.5
    return (cx / (6 * area), cy / (6 * area))


def shape_area(shape: Shape) -> float:
    """面積。質量を書かなかったときの既定値づくりに使います。"""
    if shape.type == "circle":
        return math.pi * shape.radius * shape.radius
    v = shape.vertices
    if len(v) == 0:
        return 0.0
    rolled = np.roll(v, -1, axis=0)
    return float(abs(np.sum(v[:, 0] * rolled[:, 1] - rolled[:, 0] * v[:, 1])) / 2)


def shape_inertia(shape: Shape, mass: float) -> float:
    """重心まわりの慣性モーメント。"""
    if shape.type == "circle":
        return 0.5 * mass * shape.radius * shape.radius
    v = shape.vertices
    n = len(v)
    if n == 0:
        return mass
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        ax, ay = v[i]
        bx, by = v[(i + 1) % n]
        cross = abs(ax * by - bx * ay)
        numerator += cross * (ax * ax + ax * bx + bx * bx + ay * ay + ay * by + by * by)
        denominator += cross
    if denominator < 1e-9:
        return mass
    return (mass / 6) * (numerator / denominator)


def shape_aabb(shape: Shape, position, angle: float) -> tuple[float, float, float, float]:
    """世界座標での外接矩形 ``(minX, minY, maxX, maxY)``。"""
    px = position[0] if not hasattr(position, "x") else position.x
    py = position[1] if not hasattr(position, "y") else position.y
    if shape.type == "circle":
        r = shape.radius
        return (px - r, py - r, px + r, py + r)
    cos = math.cos(angle)
    sin = math.sin(angle)
    v = shape.vertices
    xs = px + v[:, 0] * cos - v[:, 1] * sin
    ys = py + v[:, 0] * sin + v[:, 1] * cos
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def create_shape(spec: dict | None, context: dict | None = None) -> Shape:
    """プロジェクト JSON の記述から形を作る。

    :param spec: ``{"type": "circle", "radius": 30}`` のような指定
    :param context: ``bitmap`` / ``width`` / ``height`` / ``alphaOutline``
    """
    context = context or {}
    spec = spec or {}
    kind = spec.get("type", "rectangle")
    ctx_w = context.get("width")
    ctx_h = context.get("height")

    if kind == "circle":
        radius = spec.get("radius")
        if radius is None:
            radius = max(ctx_w if ctx_w is not None else 50, ctx_h if ctx_h is not None else 50) / 2
        return circle_shape(radius)

    if kind == "capsule":
        length = spec.get("length", ctx_w if ctx_w is not None else 100)
        radius = spec.get("radius")
        if radius is None:
            radius = (ctx_h if ctx_h is not None else 50) / 2
        return capsule_shape(length, radius)

    if kind in ("polygon", "mesh"):
        pts = spec.get("points")
        if isinstance(pts, (list, tuple)) and len(pts) >= 3:
            return polygon_shape(pts)
        return rectangle_shape(
            spec.get("width", ctx_w if ctx_w is not None else 100),
            spec.get("height", ctx_h if ctx_h is not None else 100),
        )

    if kind == "alpha-outline":
        bitmap = context.get("bitmap")
        if bitmap is None:
            # 画像が無ければ矩形に落とす。ここで例外にすると «描けるのに落ちる» ので。
            return rectangle_shape(
                spec.get("width", ctx_w if ctx_w is not None else 100),
                spec.get("height", ctx_h if ctx_h is not None else 100),
            )
        if context.get("alphaOutline") is False:
            return rectangle_shape(
                ctx_w if ctx_w is not None else bitmap.width,
                ctx_h if ctx_h is not None else bitmap.height,
            )
        return alpha_outline_shape(
            bitmap,
            {
                "threshold": spec.get("threshold"),
                "simplify": spec.get("simplify"),
                "width": ctx_w,
                "height": ctx_h,
            },
        )

    return rectangle_shape(
        spec.get("width", ctx_w if ctx_w is not None else 100),
        spec.get("height", ctx_h if ctx_h is not None else 100),
    )
