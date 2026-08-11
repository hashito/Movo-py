"""3D の板・OBJ メッシュ・エフェクトグラフ。

こちらも «絵» は JS 版と突き合わせて確かめてあります（開発時の比較で
`triangle` `coverage` `plane` `planeDraw` `meshDraw` `meshShadow` がいずれも
1 画素も違いませんでした）。ここで固定するのは **崩れたときに気付きにくい
決めごと**です。

- テクスチャ座標は «テクセル» 単位。0..1 を渡すと板が 1 画素に潰れる
- 回す順番は Z → X → Y。逆にすると `rotation` だけ書いた既存の絵が変わる
- `lookAt` を書かないカメラは 2.5D と «完全に同じ» 投影になる
"""

from __future__ import annotations

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.renderer.effect_graph import EffectGraphError, apply_effect_graph
from movo.renderer.mesh3d import draw_floor_shadow, draw_mesh, normalize_bounds, parse_obj
from movo.renderer.plane3d import camera_basis, draw_plane, is_plane3d, project_plane

CUBE_OBJ = """
# 1 辺 2 の立方体
v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
usemtl shell
f 1/1 2/2 3/3 4/4
f 5/1 8/4 7/3 6/2
f 1/1 5/2 6/3 2/4
f 3/1 7/2 8/3 4/4
f 1/1 4/2 8/3 5/4
f 2/1 6/2 7/3 3/4
"""


def checker(width=40, height=28) -> Bitmap:
    ys, xs = np.mgrid[0:height, 0:width]
    data = np.zeros((height, width, 4), np.uint8)
    data[..., 0] = (xs * 6) % 256
    data[..., 1] = (ys * 9) % 256
    data[..., 2] = 200
    data[..., 3] = 255
    return Bitmap(width, height, data)


def camera(width=40, height=28, distance=600.0, look_at=None):
    eye = {"x": 0, "y": 0, "z": -distance}
    return {
        "eye": eye,
        "basis": camera_basis(eye, look_at, {"x": 0, "y": -1, "z": 0} if look_at else None),
        "referenceDistance": distance,
        "centreX": width / 2,
        "centreY": height / 2,
    }


# ── 3D の板 ──────────────────────────────────────────────────────


def test_only_rotated_layers_take_the_3d_path():
    """**どちらも書かないレイヤーは 3D 経路を通らないこと。**

    既存のプロジェクトが 1 画素も変わらないことが、この機能でいちばん大事な
    条件です。
    """
    assert not is_plane3d({"x": 10, "y": 20, "rotation": 45})
    assert not is_plane3d(None)
    assert is_plane3d({"rotationX": -35})
    assert is_plane3d({"rotationY": 24})


def test_camera_without_look_at_is_the_identity_basis():
    """`lookAt` を書かないと «+z を向いた» カメラ＝ 2.5D と同じ投影になります。"""
    basis = camera_basis({"x": 0, "y": 0, "z": -600}, None, None)
    assert basis["right"] == (1.0, 0.0, 0.0)
    assert basis["down"] == (0.0, 1.0, 0.0)
    assert basis["forward"] == (0.0, 0.0, 1.0)


def test_flat_plane_projects_to_the_original_rectangle():
    """傾けていない板は «そのままの矩形» に落ちること（2.5D との地続き）。"""
    rect = {"left": -20, "top": -14, "width": 40, "height": 28}
    transform = {"x": 0, "y": 0, "z": 0, "scaleX": 1, "scaleY": 1}
    projected = project_plane(transform, rect, camera())
    assert projected["visible"]
    assert projected["corners"][0]["x"] == pytest.approx(0)
    assert projected["corners"][2]["x"] == pytest.approx(40)
    assert projected["facing"] > 0


def test_rotation_order_is_z_then_x_then_y():
    """**回す順番は Z → X → Y。** 逆にすると `rotation` だけの絵が変わります。"""
    rect = {"left": -20, "top": -10, "width": 40, "height": 20}
    base = {"x": 0, "y": 0, "z": 0, "scaleX": 1, "scaleY": 1}
    zx = project_plane({**base, "rotation": 30, "rotationX": 40}, rect, camera())
    xz = project_plane({**base, "rotation": 40, "rotationX": 30}, rect, camera())
    assert zx["corners"][0]["x"] != pytest.approx(xz["corners"][0]["x"])


def test_plane_behind_the_camera_is_not_visible():
    transform = {"x": 0, "y": 0, "z": -1200, "rotationX": 20, "scaleX": 1, "scaleY": 1}
    projected = project_plane(transform, {"left": -20, "top": -14, "width": 40, "height": 28}, camera())
    assert not projected["visible"]


def test_back_face_is_skipped_when_single_sided():
    """裏を向いた板は `doubleSided: false` で描かれないこと。"""
    dst = Bitmap(40, 28)
    corners = [{"x": 30, "y": 4}, {"x": 6, "y": 4}, {"x": 6, "y": 24}, {"x": 30, "y": 24}]
    draw_plane(dst, checker(), corners, {"alpha": 1, "doubleSided": False, "facing": -1})
    assert dst.is_empty


def test_plane_uses_texel_uv_not_normalised():
    """**テクスチャ座標は «テクセル» 単位（0..width）です。**

    0..1 の正規化座標を渡すと «左上の 1 画素を引き伸ばした» 絵になります。
    板の中で色が変わっていれば、そうなっていない証拠になります。
    """
    dst = Bitmap(40, 28)
    corners = [{"x": 4, "y": 4}, {"x": 36, "y": 4}, {"x": 36, "y": 24}, {"x": 4, "y": 24}]
    draw_plane(dst, checker(), corners, {"alpha": 1})
    inside = dst.data[6:22, 6:34, :3]
    assert len(np.unique(inside.reshape(-1, 3), axis=0)) > 8


def test_plane_depth_buffer_keeps_the_nearer_one():
    """深度を渡すと «手前» が残ること。"""
    dst = Bitmap(40, 28)
    corners = [{"x": 4, "y": 4}, {"x": 36, "y": 4}, {"x": 36, "y": 24}, {"x": 4, "y": 24}]
    buffer = np.full((28, 40), np.inf, np.float32)
    near = Bitmap(4, 4)
    near.data[...] = (255, 0, 0, 255)
    far = Bitmap(4, 4)
    far.data[...] = (0, 0, 255, 255)
    draw_plane(dst, near, corners, {"alpha": 1, "depth": {"buffer": buffer, "values": [100.0] * 4}})
    draw_plane(dst, far, corners, {"alpha": 1, "depth": {"buffer": buffer, "values": [900.0] * 4}})
    assert dst.data[14, 20, 0] == 255  # 奥の青には負けない


# ── OBJ メッシュ ─────────────────────────────────────────────────


def test_parse_obj_reads_vertices_and_fans_quads():
    model = parse_obj(CUBE_OBJ)
    assert len(model["positions"]) == 8
    assert len(model["uvs"]) == 4
    # 4 点の面は «扇状に» 三角形 2 枚へ割る
    assert len(model["faces"]) == 12
    assert model["faces"][0]["material"] == "shell"


def test_parse_obj_handles_negative_indices():
    """OBJ の負の添字は «末尾からの相対» という意味です。"""
    model = parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
    assert model["faces"][0]["v"] == [0, 1, 2]


def test_normalize_bounds_puts_the_model_at_the_origin():
    """**単位がばらばらの OBJ を «原点中心・長辺 1» に揃えます。**

    そのまま置くと画面から消えるか点にしか見えないので、読み込み時に
    正規化しておいて `scale` を «画面上の大きさ» として書けるようにします。
    """
    bounds = normalize_bounds([[-1, -1, -1], [1, 1, 1]])
    assert bounds["center"] == [0.0, 0.0, 0.0]
    assert bounds["scale"] == pytest.approx(0.5)


def test_normalize_bounds_of_an_empty_model():
    assert normalize_bounds([])["scale"] == 1.0


def test_draw_mesh_paints_and_counts_triangles():
    model = parse_obj(CUBE_OBJ)
    model["bounds"] = normalize_bounds(model["positions"])
    dst = Bitmap(40, 28)
    depth = np.full((28, 40), np.inf, np.float32)
    drawn = draw_mesh(
        dst, model, None,
        {"x": 0, "y": 0, "z": 0, "rotation": 15, "rotationX": 25, "rotationY": 40, "scaleY": 1, "meshSize": 30},
        camera(distance=400.0), {"alpha": 1, "shading": 0.6, "depth": depth, "doubleSided": False},
    )
    assert drawn == 6  # 立方体なので 12 枚のうち手前の 6 枚
    assert not dst.is_empty


def test_shading_darkens_towards_black_not_white():
    """**陰影は «黒へ寄せて» 作ります。** 白へ寄せると白飛びになります。"""
    model = parse_obj(CUBE_OBJ)
    model["bounds"] = normalize_bounds(model["positions"])
    transform = {"x": 0, "y": 0, "z": 0, "rotationX": 25, "rotationY": 40, "scaleY": 1, "meshSize": 30}
    lit = Bitmap(40, 28)
    draw_mesh(lit, model, None, transform, camera(distance=400.0), {"shading": 0})
    shaded = Bitmap(40, 28)
    draw_mesh(shaded, model, None, transform, camera(distance=400.0), {"shading": 1})
    painted = shaded.data[..., 3] > 0
    assert shaded.data[painted][..., :3].mean() < lit.data[painted][..., :3].mean()


def test_floor_shadow_needs_a_floor():
    model = parse_obj(CUBE_OBJ)
    model["bounds"] = normalize_bounds(model["positions"])
    transform = {"x": 0, "y": 0, "z": 0, "scaleY": 1, "meshSize": 30}
    dst = Bitmap(40, 28)
    assert draw_floor_shadow(dst, model, transform, camera(distance=400.0), {"floorY": None}) == 0
    assert draw_floor_shadow(dst, model, transform, camera(distance=400.0), {"floorY": 20, "opacity": 0}) == 0
    assert draw_floor_shadow(dst, model, transform, camera(distance=400.0), {"floorY": 20, "opacity": 0.4}) > 0


# ── エフェクトグラフ ─────────────────────────────────────────────


def test_graph_without_nodes_returns_the_source():
    src = checker()
    assert apply_effect_graph(src, None) is src
    assert apply_effect_graph(src, {"nodes": []}) is src


def test_graph_runs_the_chain():
    src = checker()
    graph = {
        "nodes": [{"id": "s", "type": "source"}, {"id": "g", "type": "grayscale", "amount": 1}],
        "connections": [{"from": "s", "to": "g"}],
        "output": "g",
    }
    out = apply_effect_graph(src, graph)
    assert np.array_equal(out.data[..., 0], out.data[..., 1])


def test_graph_evaluates_each_node_once():
    """**枝分かれして合流しても、同じ枝は 1 回しか計算しません。**"""
    calls = {"n": 0}

    def counting(bitmap, params, ctx):
        calls["n"] += 1
        return bitmap.copy()

    graph = {
        "nodes": [
            {"id": "s", "type": "source"},
            {"id": "a", "type": "counting"},
            {"id": "l", "type": "blend"},
            {"id": "r", "type": "blend"},
            {"id": "out", "type": "blend"},
        ],
        "connections": [
            {"from": "s", "to": "a"},
            {"from": "a", "to": "l"},
            {"from": "a", "to": "r"},
            {"from": "l", "to": "out"},
            {"from": "r", "to": "out"},
        ],
        "output": "out",
    }
    apply_effect_graph(checker(), graph, {"plugins": {"effect": lambda name: counting if name == "counting" else None}})
    assert calls["n"] == 1


def test_graph_rejects_a_cycle():
    graph = {
        "nodes": [{"id": "a", "type": "blur"}, {"id": "b", "type": "blur"}],
        "connections": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
        "output": "b",
    }
    with pytest.raises(EffectGraphError):
        apply_effect_graph(checker(), graph)


def test_graph_rejects_an_unknown_output():
    graph = {"nodes": [{"id": "a", "type": "blur"}], "output": "nope"}
    with pytest.raises(EffectGraphError):
        apply_effect_graph(checker(), graph)


def test_graph_skips_unknown_connections():
    """知らないノードを指した接続は «飛ばして» 描き切ること（絵は出す）。"""
    graph = {
        "nodes": [{"id": "s", "type": "source"}],
        "connections": [{"from": "ghost", "to": "s"}],
        "output": "s",
    }
    assert apply_effect_graph(checker(), graph).width == 40
