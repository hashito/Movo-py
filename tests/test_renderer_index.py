"""レンダラー本体（`movo/renderer/index.py`）の «繋ぎ目» を押さえる。

移植を 6 体で並列に進めたので、壊れたのは中身ではなく **モジュール同士の
繋ぎ目**でした。ここに置いてあるのは全部 «実際に踏んだ食い違い» です。
中身の正しさ（画素）は `test_effects_parity.py` などが見ています。

見ているもの:

  - `Mesh.draw` の options 辞書 → `draw_textured_triangle` のキーワード引数
  - `_audio_at` が `(3, フレーム数)` の NumPy 配列を受けられること
  - パーティクルの `ParticleSystem` が «描く側と組になっているほう» であること
  - シーンのトランジションが NumPy 経路で 1 枚ぶん動くこと
  - シーン合成・アンカー・親子・トラックマットが端まで通ること
  - 同じ JSON からは同じフレーム（決定性）
"""

from __future__ import annotations

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.deformer.mesh import Mesh
from movo.renderer.index import Renderer, apply_scene_transition
from movo.timeline import build_timeline

BASE = {
    "project": {"name": "t", "seed": 7},
    "video": {"width": 64, "height": 36, "fps": 10, "duration": 2, "background": "#000000"},
    "render": {"quality": "standard"},
}


def project_with(layers, **scene):
    return {**BASE, "scenes": [{"id": "s", "duration": 2, "layers": layers, **scene}]}


def make_renderer(project):
    timeline = build_timeline(project)
    return Renderer(project=project, timeline=timeline, assets=None).prepare()


CIRCLE = {
    "id": "ball",
    "type": "shape",
    "shape": {"type": "circle", "radius": 10, "fill": "#ff0000"},
    "transform": {"x": 32, "y": 18},
}


# ── 繋ぎ目 1: Mesh.draw の options ──────────────────────────────


def test_mesh_draw_unpacks_the_options_dict():
    """`Mesh.draw` は options 辞書、`draw_textured_triangle` はキーワード引数。

    そのまま渡していたので `alpha` の位置に辞書が入り、
    `'<=' not supported between instances of 'dict' and 'int'` で落ちていました。
    """
    src = Bitmap(8, 8)
    src.data[...] = 255
    dst = Bitmap(16, 16)
    Mesh.grid(8, 8, 4, 8, 8).draw(dst, src, [1, 0, 0, 1, 0, 0], {"alpha": 1.0, "clampEdge": False})
    assert dst.data[..., 3].max() == 255


def test_identity_mesh_covers_the_same_area_as_the_split_one():
    """変形していない格子を 4 隅だけで描いても絵は変わらない（速度のための近道）。"""
    src = Bitmap(8, 8)
    src.data[...] = 255
    fast = Bitmap(16, 16)
    slow = Bitmap(16, 16)
    Mesh.grid(8, 8, 4, 8, 8).draw(fast, src, [1, 0, 0, 1, 2, 2], {"alpha": 1.0})
    bent = Mesh.grid(8, 8, 4, 8, 8)
    bent.x[0] += 1e-3  # わずかに動かして «分割したまま» の経路へ入れる
    bent.x[0] -= 1e-3
    bent.alpha[0] = 0.999999  # アルファが一様でないので分割経路を通る
    bent.draw(slow, src, [1, 0, 0, 1, 2, 2], {"alpha": 1.0})
    assert np.abs(fast.data.astype(int) - slow.data.astype(int)).max() <= 1


# ── 繋ぎ目 2: 音の包絡 ──────────────────────────────────────────


def test_audio_at_accepts_the_numpy_bands_from_analyze_envelope():
    """`analyze_envelope` の `bands` は `(3, フレーム数)` の NumPy 配列。

    `bands or []` と書くと「真理値が曖昧」で落ちます。
    """
    renderer = make_renderer(project_with([CIRCLE]))
    renderer.audio_envelope = {
        "levels": np.linspace(0, 1, 20, dtype=np.float32),
        "bands": np.zeros((3, 20), np.float32),
    }
    renderer.audio_envelope["bands"][1, 5] = 0.5
    assert renderer._audio_at(0.5)["bands"] == [0.0, 0.5, 0.0]
    # 包絡が無くても «静かな音» として答える（式が壊れないように）
    renderer.audio_envelope = None
    assert renderer._audio_at(0.5) == {"level": 0, "bands": [0, 0, 0]}


# ── 繋ぎ目 3: パーティクル ──────────────────────────────────────


def test_particle_system_is_the_one_the_drawing_side_expects():
    """`ParticleSystem` は physics と renderer の 2 か所に移植されています。

    描く側（`particles.render_particles`）は renderer 版の `render()` を呼ぶので、
    physics 版を渡すと `prepare()` の時点で落ちます。
    """
    from movo.renderer.particles import ParticleSystem as DrawableParticleSystem

    renderer = make_renderer(
        project_with([{"id": "p", "type": "particle", "emitter": {"rate": 20, "lifetime": 1}}])
    )
    system = renderer.particles["p"]
    assert isinstance(system, DrawableParticleSystem)
    assert "count" in system.render()


# ── 繋ぎ目 4: トランジション（画素ループを NumPy にした経路）────


@pytest.mark.parametrize("kind", ["fade", "wipe", "iris", "dissolve", "jaws", "matte", "slide", "zoom", "flash"])
def test_every_transition_type_returns_a_usable_result(kind):
    buffer = Bitmap(32, 18)
    buffer.data[...] = 255
    applied = apply_scene_transition(buffer, {"type": kind, "in": 1.0}, 0.3, 2.0, seed=3)
    assert 0 <= applied["alpha"] <= 1
    assert "offsetX" in applied and "offsetY" in applied
    assert buffer.data.dtype == np.uint8


def test_a_finished_transition_leaves_the_buffer_alone():
    buffer = Bitmap(8, 8)
    buffer.data[...] = 200
    applied = apply_scene_transition(buffer, {"type": "wipe", "in": 0.5}, 1.0, 2.0)
    assert applied == {"alpha": 1, "offsetX": 0, "offsetY": 0}
    assert (buffer.data == 200).all()


# ── 端まで通る ────────────────────────────────────────────────


def test_render_frame_draws_the_layer_at_its_anchor():
    """既定のアンカーは中心。`transform.x/y` が «図形の中心» になる。"""
    renderer = make_renderer(project_with([CIRCLE]))
    frame = renderer.render_frame(5)
    assert frame.width == 64 and frame.height == 36
    # 中心は赤、四隅は背景
    assert frame.data[18, 32, 0] > 200 and frame.data[18, 32, 1] < 40
    assert frame.data[0, 0, 0] == 0


def test_parent_transform_is_inherited():
    layers = [
        {"id": "root", "type": "shape", "shape": {"type": "rect", "width": 1, "height": 1},
         "transform": {"x": 20, "y": 0}},
        {**CIRCLE, "parent": "root", "transform": {"x": 12, "y": 18}},
    ]
    renderer = make_renderer(project_with(layers))
    frame = renderer.render_frame(5)
    # 親の +20 が乗るので、円の中心は x=32 に来る
    assert frame.data[18, 32, 0] > 200
    assert frame.data[18, 12, 0] < 40


def test_track_matte_cuts_the_layer_with_another_one():
    layers = [
        {"id": "cut", "type": "shape", "matte": True,
         "shape": {"type": "circle", "radius": 4, "fill": "#ffffff"},
         "transform": {"x": 32, "y": 18}},
        {**CIRCLE, "trackMatte": {"layer": "cut", "type": "alpha"}},
    ]
    renderer = make_renderer(project_with(layers))
    frame = renderer.render_frame(5)
    assert frame.data[18, 32, 0] > 200  # マットの内側は残る
    # 半径 10 の円だが、半径 4 のマットの外は抜ける（背景の黒が見える）。
    # **アルファではなく色を見ます。** 合成先は不透明な背景なので、
    # 抜けたところもアルファは 255 のままです。
    assert frame.data[18, 41, 0] < 40


def test_scene_background_is_drawn_under_the_layers():
    renderer = make_renderer(project_with([CIRCLE], background="#0000ff"))
    frame = renderer.render_frame(5)
    assert tuple(frame.data[0, 0, :3]) == (0, 0, 255)


def test_the_same_project_gives_the_same_frame():
    """同じ JSON からは同じ動画（決定性）。粒のように乱数を使うものを混ぜる。"""
    project = project_with(
        [CIRCLE, {"id": "p", "type": "particle", "emitter": {"rate": 40, "lifetime": 1, "size": 3}}]
    )
    first = make_renderer(project).render_frame(9)
    second = make_renderer(project).render_frame(9)
    assert np.array_equal(first.data, second.data)


def test_rendering_out_of_order_matches_a_sequential_render():
    """フレーム 9 だけを描いても、頭から描いたときの 9 と同じ絵になる。

    並列レンダリングはこれが成り立つことに乗っています。
    """
    project = project_with(
        [{"id": "p", "type": "particle", "emitter": {"rate": 40, "lifetime": 1, "size": 3}}]
    )
    jumped = make_renderer(project).render_frame(9)
    sequential_renderer = make_renderer(project)
    for frame in range(10):
        last = sequential_renderer.render_frame(frame)
    assert np.array_equal(jumped.data, last.data)


def test_an_unknown_layer_type_is_skipped_not_fatal():
    renderer = make_renderer(project_with([{"id": "x", "type": "nope"}, CIRCLE]))
    frame = renderer.render_frame(5)
    assert frame.data[18, 32, 0] > 200  # 知らない種別を飛ばして、残りは描かれる
