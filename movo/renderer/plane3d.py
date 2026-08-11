"""レイヤーを «3 次元に置かれた板» として描く。

これまでの 2.5D は、レイヤーに `z` を持たせて «縮尺と視差» だけを付けるもの
でした。板は常に画面と平行なので、床に倒したり、奥へ傾けたりできません。

ここでは `transform.rotationX` / `rotationY` が入っているときだけ、板の 4 隅を
3 次元で回してから投影し、テクスチャ付き三角形 2 枚として描きます。

    rotationX: -90  … 床（手前に倒れる）
    rotationX:  90  … 天井
    rotationY:  45  … 奥へ開いた壁

**どちらも書かないレイヤーはこの経路を通りません。** 既存のプロジェクトが
1 画素も変わらないことが、この機能でいちばん大事な条件です。

## 罠（JS 版で踏んだもの）

1. **テクスチャ座標は «テクセル» 単位（0..width）です。** 0..1 の正規化座標を
   渡すと、左上の 1 画素だけを引き伸ばした絵になります。
2. **ラスタライズは投影後の «画面» 座標で行います。** 板を «遠くに大きく» 置いても
   実寸ぶんのバッファは要りません（これを取り違えて 1 フレーム 8 秒になりました）。
3. **`lookAt` を書いたときの上方向。** 画面の y は下向きなので、内部では反転して
   扱います。上方向のまま外積に入れると `right` の符号が逆になり、左右が
   入れ替わった絵（壁が反対側に出る）になります。
"""

from __future__ import annotations

import math

from movo.core.bitmap import Bitmap
from movo.renderer.effects import draw_textured_triangle


def is_plane3d(transform: dict | None) -> bool:
    """そのレイヤーが «3 次元の板» として描かれるか。"""
    if not transform:
        return False
    return (transform.get("rotationX", 0) or 0) != 0 or (transform.get("rotationY", 0) or 0) != 0


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (0.0, 0.0, 1.0) if length < 1e-9 else (v[0] / length, v[1] / length, v[2] / length)


def camera_basis(eye: dict, look_at: dict | None, up: dict | None) -> dict:
    """カメラの «向き» を作る。

    `lookAt` を書かないときは «+z を向いた» カメラになります。その場合の基底は
    単位行列そのものなので、**投影式は 2.5D のときと完全に一致します**（既存の
    プロジェクトの見た目が変わりません）。

    画面の y は下向きなので、上方向の既定は `(0, -1, 0)` です。
    """
    if not look_at:
        return {"right": (1.0, 0.0, 0.0), "down": (0.0, 1.0, 0.0), "forward": (0.0, 0.0, 1.0)}
    forward = _normalize(
        _sub((look_at["x"], look_at["y"], look_at["z"]), (eye["x"], eye["y"], eye["z"]))
    )
    # 利用者は «上方向» で書きたいが、画面の y は下向きなので内部では反転して扱う
    up = up or {}
    up_vector = _normalize((up.get("x", 0) or 0, up.get("y", -1) if up.get("y") is not None else -1, up.get("z", 0) or 0))
    down_vector = (-up_vector[0], -up_vector[1], -up_vector[2])
    # 上方向と視線が平行だと外積が潰れるので、その場合だけ別の軸を使う
    reference = (0.0, 0.0, 1.0) if abs(_dot(down_vector, forward)) > 0.999 else down_vector
    right = _normalize(_cross(reference, forward))
    down = _cross(forward, right)
    return {"right": right, "down": down, "forward": forward}


def project_plane(transform: dict, rect: dict, camera: dict) -> dict:
    """板の 4 隅を 3 次元で回してから投影する。

    **回す順番は Z → X → Y です。** Z（画面内の回転）は既存の `rotation` と同じ
    意味になるよう最初に掛け、そのあとで奥行き方向へ倒します。逆にすると、
    傾けたあとで画面内回転が効くことになり、`rotation` だけを書いていたときと
    挙動が変わります。

    `rect` は **«レイヤー内での板の位置と大きさ»** です。板の «見た目の大きさ» で
    作ってしまうと、内容がビットマップのどこにあるかを無視することになり、
    絵が縮んだり消えたりします。

    :returns: `{"corners", "depths", "depth", "visible", "facing"}`
    """
    scale_x = transform.get("scaleX", 1) if transform.get("scaleX") is not None else 1
    scale_y = transform.get("scaleY", 1) if transform.get("scaleY") is not None else 1
    left = rect["left"] * scale_x
    top = rect["top"] * scale_y
    width = rect["width"] * scale_x
    height = rect["height"] * scale_y

    rz = math.radians(transform.get("rotation", 0) or 0)
    rx = math.radians(transform.get("rotationX", 0) or 0)
    ry = math.radians(transform.get("rotationY", 0) or 0)
    cz, sz = math.cos(rz), math.sin(rz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)

    # 回転は «板の中心» ではなく transform.x/y が指す点まわりに掛かる（2D と揃える）
    local = [
        (left, top),
        (left + width, top),
        (left + width, top + height),
        (left, top + height),
    ]
    world = []
    for lx, ly in local:
        x, y, z = lx, ly, 0.0
        t = x * cz - y * sz
        y = x * sz + y * cz
        x = t
        t = y * cx - z * sx
        z = y * sx + z * cx
        y = t
        t = x * cy + z * sy
        z = -x * sy + z * cy
        x = t
        world.append((x + transform["x"], y + transform["y"], z + (transform.get("z", 0) or 0)))

    eye = camera["eye"]
    basis = camera["basis"]
    reference = camera["referenceDistance"]
    centre_x = camera["centreX"]
    centre_y = camera["centreY"]

    corners = []
    depths = []
    depth_sum = 0.0
    visible = True
    for point in world:
        v = _sub(point, (eye["x"], eye["y"], eye["z"]))
        cam_z = _dot(v, basis["forward"])
        # カメラの真横・後ろに回った頂点は投影できない。板ごと描画を諦める。
        if cam_z <= 1:
            visible = False
        k = reference / max(1, cam_z)
        corners.append({"x": centre_x + _dot(v, basis["right"]) * k, "y": centre_y + _dot(v, basis["down"]) * k})
        depths.append(cam_z)
        depth_sum += cam_z

    # 表裏の判定。投影後の符号付き面積が負なら «裏を向いている»。
    area = (corners[1]["x"] - corners[0]["x"]) * (corners[2]["y"] - corners[0]["y"]) - (
        corners[2]["x"] - corners[0]["x"]
    ) * (corners[1]["y"] - corners[0]["y"])
    facing = 0.0 if area == 0 else math.copysign(1.0, area)
    return {
        "corners": corners,
        "depths": depths,
        "depth": depth_sum / 4,
        "visible": visible,
        "facing": facing,
    }


def draw_plane(destination: Bitmap, bitmap: Bitmap, corners, options: dict | None = None) -> None:
    """投影した 4 隅にビットマップを貼る（三角形 2 枚）。

    :param corners: 左上 → 右上 → 右下 → 左下
    :param options: `alpha` `blend` `tint` `doubleSided` `facing` `depth`

    `depth` を渡すと板どうしが «交差して» 見えるようになります。渡さなければ
    従来どおり «描いた順» の重なりです。
    """
    options = options or {}
    alpha = options.get("alpha", 1)
    if alpha is None:
        alpha = 1
    if alpha <= 0:
        return
    # 背面を描かない設定なら、裏を向いた時点で何もしない
    if options.get("doubleSided") is False and (options.get("facing") or 0) < 0:
        return

    # ★ テクスチャ座標は «画素単位»（0..width）。0..1 を渡すと板が 1 画素に潰れます。
    w = bitmap.width
    h = bitmap.height
    uv = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]

    def vertex(i):
        return {"x": corners[i]["x"], "y": corners[i]["y"], "u": uv[i][0], "v": uv[i][1]}

    shared = {
        "alpha": alpha,
        "blend": options.get("blend", "normal") or "normal",
        "tint": options.get("tint"),
        "clampEdge": True,
    }
    depth = options.get("depth")

    def depth_for(a, b, c):
        if not depth:
            return None
        return {
            "buffer": depth["buffer"],
            "z": [depth["values"][a], depth["values"][b], depth["values"][c]],
            "test": depth.get("test", True),
            "write": depth.get("write", True),
        }

    draw_textured_triangle(destination, bitmap, vertex(0), vertex(1), vertex(2), {**shared, "depth": depth_for(0, 1, 2)})
    draw_textured_triangle(destination, bitmap, vertex(0), vertex(2), vertex(3), {**shared, "depth": depth_for(0, 2, 3)})


__all__ = ["camera_basis", "draw_plane", "is_plane3d", "project_plane"]
