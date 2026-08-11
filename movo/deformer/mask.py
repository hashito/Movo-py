"""マスクの評価（JS 版 packages/deformer/src/mask.js の移植）。

マスクはレイヤーの内容ボックス上に **0〜1 の重み場**を作ります。変形は
その重みで変位を割り引き、画素エフェクトはその重みで元と混ぜます。
「腕だけ曲げる」「顔だけぼかす」ができるのはこれのおかげです。

座標は **正規化**（内容ボックスに対する割合）です。仕様書の例と同じ書き方に
なるようにしてあります。

## Python 版で変えたところ

JS 版は 64x64 の場を «画素ごとの二重ループ» で作っていました。Python では
1 枚あたり 4,096 回の Python ループになり、変形が 10 個乗るレイヤーで
1 フレーム 20ms 以上かかります。

**すべて NumPy の一括演算に置き換えました。** 形は同じで、`for` が
座標グリッドの演算になっただけです。

| 64x64 の `ellipse` ＋ `feather: 0.3` 1 枚 | |
| --- | --- |
| 画素ごとの Python ループ | 40.6 ms |
| **NumPy の一括演算** | **0.31 ms**（133 倍） |

膨張（`expand`）とぼかし（`feather`）も同じです。ぼかしは JS 版と同じ
«端では届いた画素だけで平均する» 箱ぼかしを 2 回、累積和で作っています。
"""

from __future__ import annotations

import math

import numpy as np

from ._compat import clamp, warn
from ._sampling import channel_value, sample_bilinear

MASK_TYPES = [
    "rectangle",
    "ellipse",
    "sector",
    "diagonal",
    "polygon",
    "path",
    "image",
    "alpha",
    "layer",
    "segmentation",
]


def _grid(width: int, height: int):
    """画素の中心（0〜1）の格子。`(v, u)` の順に返します。"""
    u = (np.arange(width, dtype=np.float64) + 0.5) / width
    v = (np.arange(height, dtype=np.float64) + 0.5) / height
    return np.meshgrid(u, v)


def build_mask_field(mask: dict | None, width: int, height: int, ctx: dict | None = None):
    """重み場を作る。`None` は «マスク無し»（どこも全力）という意味です。

    :param mask: 解決済みのマスク記述
    :param width: 場の解像度（画素）
    :param height: 同上
    :param ctx: ``assets`` / ``selfBitmap`` / ``layerAlpha``
    :returns: ``(height*width,)`` の float32、または `None`
    """
    ctx = ctx or {}
    if not mask or not mask.get("type"):
        return None
    kind = mask["type"]

    cx = float(mask.get("x", 0.5))
    cy = float(mask.get("y", 0.5))
    mw = float(mask.get("width", 1))
    mh = float(mask.get("height", 1))
    rotation = math.radians(float(mask.get("rotation", 0)))
    cos = math.cos(-rotation)
    sin = math.sin(-rotation)

    u, v = _grid(width, height)

    def rotated():
        dx = u - cx
        dy = v - cy
        return dx * cos - dy * sin, dx * sin + dy * cos

    if kind == "rectangle":
        dx, dy = rotated()
        field = ((np.abs(dx) <= mw / 2) & (np.abs(dy) <= mh / 2)).astype(np.float32)

    elif kind == "ellipse":
        rx = max(1e-6, mw / 2)
        ry = max(1e-6, mh / 2)
        dx, dy = rotated()
        field = (((dx / rx) ** 2 + (dy / ry) ** 2) <= 1).astype(np.float32)

    elif kind == "sector":
        # 扇型。放射ワイプや時計状の登場に使います。
        start_angle = math.radians(float(mask.get("startAngle", -90)))
        end_angle = math.radians(float(mask.get("endAngle", 90)))
        inner_radius = float(mask.get("innerRadius", 0))
        outer_radius = float(mask.get("outerRadius", 1.5))
        sweep = end_angle - start_angle
        turn = math.pi * 2
        # 一周ぶん指定されたときは剰余が 0 になってしまうので全周として扱います。
        span = turn if abs(sweep) >= turn - 1e-9 else ((sweep % turn) + turn) % turn
        dx = u - cx
        dy = v - cy
        radius = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx) - start_angle
        theta = ((theta % turn) + turn) % turn
        inside_ring = (radius >= inner_radius) & (radius <= outer_radius)
        field = (inside_ring & (theta <= span)).astype(np.float32)

    elif kind == "diagonal":
        # 角度と幅で指定する帯。width を 0 から広げると斜めに現れ、
        # 1.5 以上で全面になります。シーンチェンジの定番です。
        angle = math.radians(float(mask.get("angle", -45)))
        nx = math.cos(angle)
        ny = math.sin(angle)
        centre = float(mask.get("center", 0.5))
        band = max(0.0, float(mask.get("width", 1)))
        half = band / 2
        projected = (u - 0.5) * nx + (v - 0.5) * ny + 0.5
        field = (np.abs(projected - centre) <= half).astype(np.float32)

    elif kind == "polygon":
        points = _as_pairs(mask.get("points") or [])
        if len(points) < 3:
            return None
        field = point_in_polygon(u, v, points).astype(np.float32)

    elif kind == "path":
        # 折れ線の «まわり» を通すマスク（#73 でトリムと `d` に対応）。
        polylines = _mask_path_polylines(mask)
        # None は «形が書かれていない»（＝マスク無し）。空リストは «トリムで消えた»
        # （＝どこも通さない）。同じ「線が無い」でも意味が逆なので分けます。
        if polylines is None:
            return None
        thickness = max(1e-4, float(mask.get("thickness", 0.1)))
        half = thickness / 2
        inside = np.zeros(u.shape, bool)
        for polyline in polylines:
            inside |= _distance_to_polyline(u, v, polyline) <= half
        field = inside.astype(np.float32)

    elif kind == "image":
        assets = ctx.get("assets")
        bitmap = assets.get(mask.get("asset")) if assets else None
        if bitmap is None:
            warn(f'mask asset "{mask.get("asset")}" is unavailable; the mask is ignored')
            return None
        sample = sample_bilinear(bitmap, u * bitmap.width, v * bitmap.height, True)
        field = (channel_value(sample, mask.get("channel", "luminance")) / 255).astype(np.float32)

    elif kind == "alpha":
        bitmap = ctx.get("selfBitmap")
        if bitmap is None:
            return None
        sample = sample_bilinear(bitmap, u * bitmap.width, v * bitmap.height, True)
        field = (sample[..., 3] / 255).astype(np.float32)

    elif kind == "layer":
        resolve = ctx.get("layerAlpha")
        bitmap = resolve(mask.get("layer")) if resolve else None
        if bitmap is None:
            warn(f'mask layer "{mask.get("layer")}" was not rendered before this one; the mask is ignored')
            return None
        sample = sample_bilinear(bitmap, u * bitmap.width, v * bitmap.height, True)
        field = (sample[..., 3] / 255).astype(np.float32)

    elif kind == "segmentation":
        warn('mask type "segmentation" needs an AI segmentation plugin; falling back to the full layer')
        return None

    else:
        warn(f'unknown mask type "{kind}"; the mask is ignored')
        return None

    field = np.ascontiguousarray(field.reshape(height, width), np.float32)
    if mask.get("expand"):
        field = _expand_field(field, float(mask["expand"]))
    if mask.get("feather"):
        field = _feather_field(field, float(mask["feather"]))
    if mask.get("invert"):
        field = 1 - field
    opacity = mask.get("opacity")
    if isinstance(opacity, (int, float)):
        field = field * clamp(float(opacity), 0.0, 1.0)
    return field.ravel().astype(np.float32)


def _as_pairs(values) -> list[tuple[float, float]]:
    out = []
    for p in values:
        if isinstance(p, dict):
            out.append((float(p.get("x", 0)), float(p.get("y", 0))))
        else:
            out.append((float(p[0]), float(p[1])))
    return out


def _mask_path_polylines(mask: dict):
    """`path` マスクの折れ線を作る（#73）。

    座標は **0〜1 の正規化**です。書き方は 2 つ:

        {"type": "path", "path": [[0.1, 0.2], [0.9, 0.5]]}   点列（従来どおり）
        {"type": "path", "d": "M10 10 C ..."}                 SVG のパス文字列

    `d` は **そのままでは 0〜1 に収まりません**。`viewBox` を書けばそれで写し、
    書かなければパス自身の外接矩形で 0〜1 に伸ばします。

    :returns: 折れ線のリスト。`None` は «形が書かれていない»
    """
    if isinstance(mask.get("d"), str):
        try:
            from movo.core.svg_path import (  # type: ignore
                is_trim_active, path_to_subpaths, subpaths_bounds, trim_subpaths,
            )
        except Exception:
            warn("mask.d（SVG パス）には movo.core.svg_path が要ります（まだ移植されていません）")
            return None
        subpaths = path_to_subpaths(mask["d"])
        if not subpaths:
            return None
        view_box = mask.get("viewBox") if isinstance(mask.get("viewBox"), (list, tuple)) else None
        if view_box is not None and len(view_box) != 4:
            view_box = None
        bounds = subpaths_bounds(subpaths)
        origin_x = view_box[0] if view_box else bounds["minX"]
        origin_y = view_box[1] if view_box else bounds["minY"]
        span_x = (view_box[2] or 1) if view_box else bounds["width"]
        span_y = (view_box[3] or 1) if view_box else bounds["height"]
        scaled = []
        for subpath in subpaths:
            pts = list(subpath["points"])
            for i in range(0, len(pts), 2):
                pts[i] = (pts[i] - origin_x) / span_x
                pts[i + 1] = (pts[i + 1] - origin_y) / span_y
            scaled.append({"points": pts, "closed": subpath["closed"]})
        subpaths = scaled
        if is_trim_active(mask.get("trim")):
            subpaths = trim_subpaths(subpaths, mask["trim"])
    else:
        pairs = mask.get("path") or mask.get("points") or []
        if len(pairs) < 2:
            return None
        flat = []
        for p in _as_pairs(pairs):
            flat += [p[0], p[1]]
        subpaths = [{"points": flat, "closed": mask.get("closed") is True}]

    out = []
    for subpath in subpaths:
        pts = subpath["points"]
        pairs = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
        # 閉じたサブパスは «最初に戻る 1 辺» も距離の対象にします。
        if subpath.get("closed") and len(pairs) >= 2:
            pairs.append(pairs[0])
        if len(pairs) >= 2:
            out.append(np.array(pairs, np.float64))
    return out


def point_in_polygon(x, y, points) -> np.ndarray:
    """奇偶規則で «中か» を判定する。配列でまとめて通せます。"""
    pts = np.asarray(points, np.float64)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    inside = np.zeros(np.broadcast(x, y).shape, bool)
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        # JS 版と同じ «yi > y と yj > y が違うとき» の判定。
        crosses = (yi > y) != (yj > y)
        denom = yj - yi
        safe = denom if denom != 0 else 1.0
        limit = ((xj - xi) * (y - yi)) / safe + xi
        inside ^= crosses & (x < limit)
        j = i
    return inside


def _distance_to_polyline(x, y, points: np.ndarray) -> np.ndarray:
    """折れ線までの最短距離。線分ごとに一括で測って最小を取ります。"""
    best = np.full(np.broadcast(x, y).shape, np.inf)
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq <= 0:
            t = np.zeros_like(best)
        else:
            t = np.clip(((x - x0) * dx + (y - y0) * dy) / length_sq, 0.0, 1.0)
        best = np.minimum(best, np.hypot(x - (x0 + dx * t), y - (y0 + dy * t)))
    return best


def _expand_field(field: np.ndarray, amount: float) -> np.ndarray:
    """マスクを太らせる（負なら痩せさせる）。円形の構造要素です。"""
    height, width = field.shape
    radius = round(abs(amount) * min(width, height))
    if radius <= 0:
        return field
    grow = amount > 0
    out = field.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            if dx == 0 and dy == 0:
                continue
            # ずらした場を «届く範囲だけ» 重ねます。届かない所は元の値のまま。
            ys = slice(max(0, dy), min(height, height + dy))
            xs = slice(max(0, dx), min(width, width + dx))
            ys_src = slice(max(0, -dy), min(height, height - dy))
            xs_src = slice(max(0, -dx), min(width, width - dx))
            target = out[ys_src, xs_src]
            other = field[ys, xs]
            out[ys_src, xs_src] = np.maximum(target, other) if grow else np.minimum(target, other)
    return out


def _box_blur_1d(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """端では «届いた画素だけ» で平均する箱ぼかし（JS 版と同じ数え方）。"""
    moved = np.moveaxis(values, axis, -1)
    n = moved.shape[-1]
    padded = np.concatenate([np.zeros(moved.shape[:-1] + (1,), np.float64), np.cumsum(moved, axis=-1)], axis=-1)
    index = np.arange(n)
    low = np.maximum(0, index - radius)
    high = np.minimum(n, index + radius + 1)
    total = padded[..., high] - padded[..., low]
    count = (high - low).astype(np.float64)
    return np.moveaxis(np.where(count > 0, total / count, 0.0), -1, axis)


def _feather_field(field: np.ndarray, feather: float) -> np.ndarray:
    """箱ぼかしを 2 回。安くて柔らかい «ぼかし縁» です。

    **1 パスごとに float32 に落としています。** JS 版は途中結果を
    `Float32Array` に書くので、そこで丸めが 1 回入ります。float64 のまま
    通すと «より正確» になりますが、JS 版と 1e-7 ずれ、そのずれが
    変形（画素単位）で 1e-6 まで拡大しました。**同じ絵を出すほうが優先です。**
    """
    height, width = field.shape
    radius = max(1, round(feather * min(width, height) * 0.5))
    out = field.astype(np.float32)
    for _ in range(2):
        out = _box_blur_1d(out.astype(np.float64), radius, axis=1).astype(np.float32)  # 横
        out = _box_blur_1d(out.astype(np.float64), radius, axis=0).astype(np.float32)  # 縦
    return out


def sample_field(field, width: int, height: int, u, v):
    """重み場を双一次で読む（座標は 0〜1）。`field` が `None` なら 1。"""
    if field is None:
        return np.ones_like(np.asarray(u, np.float64))
    grid = np.asarray(field, np.float64).reshape(height, width)
    x = np.clip(np.asarray(u, np.float64) * width - 0.5, 0, width - 1)
    y = np.clip(np.asarray(v, np.float64) * height - 0.5, 0, height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(width - 1, x0 + 1)
    y1 = np.minimum(height - 1, y0 + 1)
    fx = x - x0
    fy = y - y0
    a = grid[y0, x0]
    b = grid[y0, x1]
    c = grid[y1, x0]
    d = grid[y1, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
