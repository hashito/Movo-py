"""3D メッシュ（OBJ）の読み込みと描画。

板を 3 次元に置けるようになっても（`plane3d.py`）、作れるのは «平らな面» だけ
です。立体そのものを置きたい場面 — ステージのセット、小道具、ロゴの押し出し —
ではモデルを読めた方が早い。

**OBJ を選んだ理由。** テキスト形式なので自前パーサが数十行で済み、外部依存が
要りません。glTF は JSON + バイナリで、実装量が一桁増えるわりに、この用途で
得られるものは «アニメーション付きモデルが読める» ことくらいです。

描画は `plane3d.py` と同じテクスチャ付き三角形 + 深度バッファです。三角形ごとに
奥行きを持たせているので、へこんだ形でも自分自身の前後関係が正しく出ます。

## 陰影について

`drawTexturedTriangle` の `tint` は «その色へ寄せる» ものなので、**暗くするには
黒へ寄せます。** 白へ寄せると陰影ではなく白飛びになります。

平行光源（太陽）と点光源（電球）の両方を持てます。点光源の減衰は物理どおりの
逆二乗ではなく «範囲で正規化した滑らかな落ち方» にしました。逆二乗だと近づいた
瞬間に真っ白になって、扱いにくいからです。
"""

from __future__ import annotations

import math

import numpy as np

from movo.core.bitmap import Bitmap
from movo.renderer.effects import Color, draw_textured_triangle


def parse_obj(text: str) -> dict:
    """OBJ を読む。

    対応するのは `v`（頂点）・`vt`（テクスチャ座標）・`vn`（法線）・`f`（面）だけ
    です。`f` は 3 点でも 4 点以上でもよく、扇状に三角形へ割ります。マテリアル
    （`usemtl`）は名前だけ拾って «面のまとまり» として持ち、色は呼ぶ側が決めます。

    :returns: `{"positions", "uvs", "normals", "faces"}`
    """
    positions: list[list[float]] = []
    uvs: list[list[float]] = []
    normals: list[list[float]] = []
    faces: list[dict] = []
    material = None

    def index(raw: str, length: int) -> int:
        """OBJ の添字は 1 始まりで、負の値は «末尾からの相対» を意味します。"""
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return -1
        if value == 0:
            return -1
        return value - 1 if value > 0 else length + value

    def number(token: str) -> float:
        try:
            value = float(token)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        head = parts[0]
        if head == "v":
            positions.append([number(p) for p in (parts + ["0", "0", "0"])[1:4]])
        elif head == "vt":
            uvs.append([number(p) for p in (parts + ["0", "0"])[1:3]])
        elif head == "vn":
            normals.append([number(p) for p in (parts + ["0", "0", "0"])[1:4]])
        elif head == "usemtl":
            material = parts[1] if len(parts) > 1 else None
        elif head == "f":
            corners = []
            for token in parts[1:]:
                bits = token.split("/")
                v = index(bits[0], len(positions))
                t = index(bits[1], len(uvs)) if len(bits) > 1 and bits[1] else -1
                n = index(bits[2], len(normals)) if len(bits) > 2 and bits[2] else -1
                corners.append((v, t, n))
            # 扇状に三角形へ割る。凸でない面はきれいに割れないが、OBJ の面は
            # ほぼ凸なので実用上は足りる。
            for i in range(1, len(corners) - 1):
                faces.append(
                    {
                        "v": [corners[0][0], corners[i][0], corners[i + 1][0]],
                        "t": [corners[0][1], corners[i][1], corners[i + 1][1]],
                        "n": [corners[0][2], corners[i][2], corners[i + 1][2]],
                        "material": material,
                    }
                )
    return {"positions": positions, "uvs": uvs, "normals": normals, "faces": faces}


def normalize_bounds(positions) -> dict:
    """モデルを «原点中心・長辺 1» に正規化するための情報を出す。

    OBJ の単位はモデルごとにばらばら（メートルだったり 100 倍だったり）なので、
    そのまま置くと画面から消えるか点にしか見えません。`transform.scale` を
    «画面上の高さ» として書けるよう、読み込み時に正規化しておきます。
    """
    if not positions:
        return {"center": [0.0, 0.0, 0.0], "scale": 1.0, "size": [0.0, 0.0, 0.0]}
    arr = np.asarray(positions, np.float64)
    low = arr.min(axis=0)
    high = arr.max(axis=0)
    size = (high - low).tolist()
    center = ((low + high) / 2).tolist()
    longest = max(size) or 1.0
    return {"center": center, "scale": 1.0 / longest, "size": size}


def _rotation(rotation_z: float, rotation_x: float, rotation_y: float):
    """回転を «3 つの軸» から組む。**順番は plane3d と揃えて Z → X → Y。**"""
    cz, sz = math.cos(math.radians(rotation_z)), math.sin(math.radians(rotation_z))
    cx, sx = math.cos(math.radians(rotation_x)), math.sin(math.radians(rotation_x))
    cy, sy = math.cos(math.radians(rotation_y)), math.sin(math.radians(rotation_y))

    def rotate(p):
        x, y, z = p[0], p[1], p[2]
        t = x * cz - y * sz
        y = x * sz + y * cz
        x = t
        t = y * cx - z * sx
        z = y * sx + z * cx
        y = t
        t = x * cy + z * sy
        z = -x * sy + z * cy
        x = t
        return (x, y, z)

    return rotate


def _project(points, transform, camera, bounds, size, rotate):
    """頂点をワールド → カメラ空間 → 画面へ。**頂点ごとに 1 度だけ計算します。**"""
    eye = camera["eye"]
    basis = camera["basis"]
    reference = camera["referenceDistance"]
    centre_x = camera["centreX"]
    centre_y = camera["centreY"]
    tx = transform["x"]
    ty = transform["y"]
    tz = transform.get("z", 0) or 0

    screen: list[tuple[float, float] | None] = []
    depth: list[float] = []
    worlds: list[tuple[float, float, float]] = []
    for p in points:
        local = rotate(
            (
                (p[0] - bounds["center"][0]) * bounds["scale"] * size,
                (p[1] - bounds["center"][1]) * bounds["scale"] * size,
                (p[2] - bounds["center"][2]) * bounds["scale"] * size,
            )
        )
        world = (local[0] + tx, local[1] + ty, local[2] + tz)
        worlds.append(world)
        v = (world[0] - eye["x"], world[1] - eye["y"], world[2] - eye["z"])
        cam_z = v[0] * basis["forward"][0] + v[1] * basis["forward"][1] + v[2] * basis["forward"][2]
        depth.append(cam_z)
        if cam_z <= 1:
            screen.append(None)
            continue
        k = reference / cam_z
        screen.append(
            (
                centre_x + (v[0] * basis["right"][0] + v[1] * basis["right"][1] + v[2] * basis["right"][2]) * k,
                centre_y + (v[0] * basis["down"][0] + v[1] * basis["down"][1] + v[2] * basis["down"][2]) * k,
            )
        )
    return screen, depth, worlds


def draw_mesh(destination: Bitmap, model: dict, texture: Bitmap | None, transform: dict, camera: dict,
              options: dict | None = None) -> int:
    """メッシュを描く。

    :param model: `parse_obj` の結果（`bounds` を持っていればそれを使います）
    :param texture: 貼る絵。無ければ単色
    :param options: `alpha` `blend` `color` `shading` `light` `pointLights`
                    `doubleSided` `depth`
    :returns: 実際に描いた三角形の数
    """
    options = options or {}
    alpha = options.get("alpha", 1)
    if alpha is None:
        alpha = 1
    if alpha <= 0:
        return 0

    positions = model["positions"]
    uvs = model["uvs"]
    faces = model["faces"]
    normals = model["normals"]
    bounds = model.get("bounds") or normalize_bounds(positions)
    scale = transform.get("scaleY") if transform.get("scaleY") is not None else transform.get("scaleX", 1)
    if scale is None:
        scale = 1
    size = scale * (transform.get("meshSize", 200) if transform.get("meshSize") is not None else 200)
    rotate = _rotation(transform.get("rotation", 0) or 0, transform.get("rotationX", 0) or 0,
                       transform.get("rotationY", 0) or 0)
    shading = options.get("shading", 0.55) if options.get("shading") is not None else 0.55
    # 平行光源。既定は «左上手前» から。法線が無いモデルでは面から計算する。
    light = options.get("light") or [-0.4, -0.7, 0.6]
    light_length = math.sqrt(light[0] ** 2 + light[1] ** 2 + light[2] ** 2) or 1
    point_lights = options.get("pointLights") or []

    screen, depth, worlds = _project(positions, transform, camera, bounds, size, rotate)

    base_color = options.get("color") or Color(200, 200, 210, 1.0)
    # テクスチャが無いときは «1 画素の単色画像» を貼る。三角形ラスタライザは常に
    # テクスチャを引く作りなので、単色用の分岐を増やすより素直。
    flat = None
    if texture is None:
        flat = Bitmap(1, 1)
        flat.data[0, 0] = (base_color.r, base_color.g, base_color.b, 255)
    drawn = 0

    for face in faces:
        a = screen[face["v"][0]]
        b = screen[face["v"][1]]
        c = screen[face["v"][2]]
        # 1 頂点でもカメラの後ろなら、その面ごと諦める。切り取りまではやらない。
        if a is None or b is None or c is None:
            continue

        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if options.get("doubleSided") is False and area < 0:
            continue

        # 陰影。法線があればそれを、無ければ面から求めた法線を使う。
        if face["n"][0] >= 0 and face["n"][0] < len(normals):
            normal = rotate(normals[face["n"][0]])
        else:
            p0 = positions[face["v"][0]]
            p1 = positions[face["v"][1]]
            p2 = positions[face["v"][2]]
            e1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
            e2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
            normal = rotate(
                (
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                )
            )
        nl = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) or 1
        lambert = max(
            0.0,
            (normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]) / (nl * light_length),
        )
        if point_lights:
            # 面の中心。点光源からの向きと距離はここで測る。
            p0 = worlds[face["v"][0]]
            p1 = worlds[face["v"][1]]
            p2 = worlds[face["v"][2]]
            cx = (p0[0] + p1[0] + p2[0]) / 3
            cy = (p0[1] + p1[1] + p2[1]) / 3
            cz = (p0[2] + p1[2] + p2[2]) / 3
            for lamp in point_lights:
                dx = (lamp.get("x", 0) or 0) - cx
                dy = (lamp.get("y", 0) or 0) - cy
                dz = (lamp.get("z", 0) or 0) - cz
                distance = math.sqrt(dx * dx + dy * dy + dz * dz) or 1
                # 逆二乗だと «近づいた瞬間に真っ白» で扱いにくいので、範囲で
                # 正規化した滑らかな落ち方にする。
                lamp_range = lamp.get("range", 600) if lamp.get("range") is not None else 600
                falloff = max(0.0, 1 - distance / lamp_range) ** 2
                if falloff <= 0:
                    continue
                facing = max(0.0, (normal[0] * dx + normal[1] * dy + normal[2] * dz) / (nl * distance))
                lambert += facing * falloff * (lamp.get("intensity", 1) if lamp.get("intensity") is not None else 1)
            lambert = min(1.6, lambert)

        # ★ tint は «その色へ寄せる» ので、暗くするには «黒へ» 寄せる
        level = 1 - shading + shading * lambert
        tint = Color(0, 0, 0, max(0.0, 1 - level)) if shading > 0 else None

        def uv_for(i, face=face):
            t = face["t"][i]
            if texture is None or t < 0 or t >= len(uvs):
                return (0.0, 0.0)
            # OBJ の v は下が 0。画像は上が 0 なので反転する。
            return (uvs[t][0] * texture.width, (1 - uvs[t][1]) * texture.height)

        ua, va = uv_for(0)
        ub, vb = uv_for(1)
        uc, vc = uv_for(2)
        draw_textured_triangle(
            destination,
            texture if texture is not None else flat,
            {"x": a[0], "y": a[1], "u": ua, "v": va},
            {"x": b[0], "y": b[1], "u": ub, "v": vb},
            {"x": c[0], "y": c[1], "u": uc, "v": vc},
            {
                "alpha": alpha,
                "blend": options.get("blend", "normal") or "normal",
                "clampEdge": True,
                "tint": tint,
                "depth": (
                    {
                        "buffer": options["depth"],
                        "z": [depth[face["v"][0]], depth[face["v"][1]], depth[face["v"][2]]],
                    }
                    if options.get("depth") is not None
                    else None
                ),
            },
        )
        drawn += 1
    return drawn


def draw_floor_shadow(destination: Bitmap, model: dict, transform: dict, camera: dict, options: dict) -> int:
    """床への投影影。

    **影があるかないかで «置かれている» 感じがまるで変わります。** ただしシャドウ
    マップは重く、この規模の用途には過剰です。ここでは «光源から見て、モデルを
    床の平面へ潰したもの» を黒く描くだけにしています。凹凸は落ちますが、接地感は
    十分に出ます。

    平行光源のときは向きだけで潰し、点光源のときは光源を通る直線で潰します。
    """
    floor_y = options.get("floorY")
    if floor_y is None or not math.isfinite(floor_y):
        return 0
    opacity = options.get("opacity", 0.35) if options.get("opacity") is not None else 0.35
    if opacity <= 0:
        return 0

    positions = model["positions"]
    faces = model["faces"]
    bounds = model.get("bounds") or normalize_bounds(positions)
    scale = transform.get("scaleY") if transform.get("scaleY") is not None else transform.get("scaleX", 1)
    if scale is None:
        scale = 1
    size = scale * (transform.get("meshSize", 200) if transform.get("meshSize") is not None else 200)
    rotate = _rotation(transform.get("rotation", 0) or 0, transform.get("rotationX", 0) or 0,
                       transform.get("rotationY", 0) or 0)
    eye = camera["eye"]
    basis = camera["basis"]
    reference = camera["referenceDistance"]
    centre_x = camera["centreX"]
    centre_y = camera["centreY"]
    lamp = options.get("lightPosition")
    direction = options.get("light") or [-0.4, -0.7, 0.6]

    shadow_color = options.get("color") or Color(0, 0, 0, 1.0)
    flat = Bitmap(1, 1)
    flat.data[0, 0] = (shadow_color.r, shadow_color.g, shadow_color.b, 255)

    screen: list[tuple[float, float, float] | None] = []
    for p in positions:
        local = rotate(
            (
                (p[0] - bounds["center"][0]) * bounds["scale"] * size,
                (p[1] - bounds["center"][1]) * bounds["scale"] * size,
                (p[2] - bounds["center"][2]) * bounds["scale"] * size,
            )
        )
        world = (local[0] + transform["x"], local[1] + transform["y"], local[2] + (transform.get("z", 0) or 0))

        if lamp:
            dx = world[0] - (lamp.get("x", 0) or 0)
            dy = world[1] - (lamp.get("y", 0) or 0)
            dz = world[2] - (lamp.get("z", 0) or 0)
        else:
            # 平行光源は «光が進む向き»。面から見た向きの逆に伸ばす。
            dx, dy, dz = -direction[0], -direction[1], -direction[2]
        # 光が床と平行、または上を向いていると影が落ちない
        if abs(dy) < 1e-6 or (floor_y - world[1]) / dy <= 0:
            screen.append(None)
            continue
        t = (floor_y - world[1]) / dy
        on_floor = (world[0] + dx * t, floor_y, world[2] + dz * t)
        v = (on_floor[0] - eye["x"], on_floor[1] - eye["y"], on_floor[2] - eye["z"])
        cam_z = v[0] * basis["forward"][0] + v[1] * basis["forward"][1] + v[2] * basis["forward"][2]
        if cam_z <= 1:
            screen.append(None)
            continue
        k = reference / cam_z
        screen.append(
            (
                centre_x + (v[0] * basis["right"][0] + v[1] * basis["right"][1] + v[2] * basis["right"][2]) * k,
                centre_y + (v[0] * basis["down"][0] + v[1] * basis["down"][1] + v[2] * basis["down"][2]) * k,
                cam_z,
            )
        )

    drawn = 0
    for face in faces:
        a = screen[face["v"][0]]
        b = screen[face["v"][1]]
        c = screen[face["v"][2]]
        if a is None or b is None or c is None:
            continue
        draw_textured_triangle(
            destination,
            flat,
            {"x": a[0], "y": a[1], "u": 0.0, "v": 0.0},
            {"x": b[0], "y": b[1], "u": 0.0, "v": 0.0},
            {"x": c[0], "y": c[1], "u": 0.0, "v": 0.0},
            {
                "alpha": opacity,
                "blend": "normal",
                "clampEdge": True,
                # 影は «床の上» なので深度を書かない。書くとモデル本体が影に負ける。
                "depth": (
                    {"buffer": options["depth"], "z": [a[2], b[2], c[2]], "write": False}
                    if options.get("depth") is not None
                    else None
                ),
            },
        )
        drawn += 1
    return drawn


__all__ = ["draw_floor_shadow", "draw_mesh", "normalize_bounds", "parse_obj"]
