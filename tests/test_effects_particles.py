"""パーティクル（粒）。

**いちばん大事なのは決定性です。** 同じ JSON からは同じ動画が出る、という
Movo の約束は、粒の «生む順» と «乱数を引く順» が JS 版と 1 回ずつ揃っている
ことで成り立っています。開発時に JS 版と突き合わせ、40 個の粒の位置・速度・
回転・寿命が 1e-13 まで一致することを確かめてあります。

ここではその «崩れやすいところ» を固定します。
"""

from __future__ import annotations



import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.renderer.particle_presets import PARTICLE_PRESETS, list_particle_presets, resolve_preset
from movo.renderer.particles import (
    ParticleSystem,
    create_particle_system,
    particle_contour,
    render_particles,
)

WORLD = {"gravity": {"x": 0, "y": 980}}


def system(**overrides) -> ParticleSystem:
    options = {
        "id": "p", "seed": 4242, "maxParticles": 40, "rate": 90, "lifetime": 1.2,
        "speed": 200, "direction": -80, "spread": 50, "size": 6, "spin": 90,
        "x": 20, "y": 14, "width": 10, "height": 4,
    }
    options.update(overrides)
    return ParticleSystem(options)


def run(sys_: ParticleSystem, steps: int = 10) -> dict:
    for _ in range(steps):
        sys_.step(1 / 30, WORLD)
    return sys_.render()


# ── 決定性 ────────────────────────────────────────────────────────


def test_same_seed_gives_the_same_particles():
    a = run(system())
    b = run(system())
    for key in ("x", "y", "vx", "vy", "rotation", "size"):
        assert np.array_equal(a[key], b[key])


def test_reset_and_replay_is_identical():
    """**巻き戻しても同じ絵になること。** 並列レンダリングの前提です。"""
    sys_ = system()
    first = run(sys_)
    sys_.reset()
    second = run(sys_)
    assert np.array_equal(first["x"], second["x"])
    assert np.array_equal(first["seed"], second["seed"])


def test_different_seed_gives_different_particles():
    assert not np.array_equal(run(system())["x"], run(system(seed=99))["x"])


def test_warmup_is_deterministic_after_reset():
    """`prewarm` は `reset()` のあとにも同じ回数まわすので結果が変わりません。"""
    sys_ = system(prewarm=0.5)
    sys_.warmup(WORLD, 1 / 30)
    first = run(sys_)
    sys_.reset()
    sys_.warmup(WORLD, 1 / 30)
    assert np.array_equal(first["x"], run(sys_)["x"])


def test_warmup_fills_the_screen_at_time_zero():
    """**`prewarm` を入れないと 0 秒時点が空っぽ**になります（雪や星で困る）。"""
    cold = system(prewarm=0)
    warm = system(prewarm=0.5)
    warm.warmup(WORLD, 1 / 30)
    assert cold.count == 0
    assert warm.count > 0
    assert warm.time == 0.0


# ── 上限と寿命 ────────────────────────────────────────────────────


def test_max_particles_is_respected_without_dropping_a_batch():
    """**上限に当たったフレームでも、その手前までの粒は残ること。**

    `_spawn_batch` を `break` ではなく `return` で抜けると、作りかけの粒が
    まるごと捨てられて «1 フレームだけ 1 個足りない» という形で出ます。
    """
    sys_ = system(maxParticles=40, rate=90, lifetime=10)
    counts = []
    for _ in range(20):
        sys_.step(1 / 30, WORLD)
        counts.append(sys_.count)
    assert max(counts) == 40
    assert counts == sorted(counts)  # 減らずに増えて上限で止まる


def test_particles_die_at_their_lifetime():
    sys_ = system(lifetime=0.2, lifetimeVariance=0, rate=30, maxParticles=100)
    for _ in range(6):
        sys_.step(1 / 30, WORLD)
    born = sys_.count
    for _ in range(20):
        sys_.step(1 / 30, WORLD)
    assert sys_.count <= born + 1  # 生まれた数だけ死んでいく（際限なく増えない）


def test_gravity_pulls_particles_down():
    sys_ = system(rate=30, speed=0, spread=0, drag=0, gravityScale=1)
    snapshot = run(sys_, 10)
    assert snapshot["vy"].max() > 0


def test_floor_bounces():
    sys_ = system(rate=30, speed=0, spread=0, floorY=20, bounce=0.5)
    snapshot = run(sys_, 20)
    assert snapshot["y"].max() <= 20 + 1e-9


def test_fade_in_and_out_clamp_opacity():
    snapshot = run(system(fadeIn=0.2, fadeOut=0.3))
    assert snapshot["opacity"].min() >= 0
    assert snapshot["opacity"].max() <= 1


# ── プリセット ────────────────────────────────────────────────────


def test_the_seven_presets_are_available():
    assert list_particle_presets() == ["bubble", "confetti", "rain", "sakura", "smoke", "snow", "spark"]


def test_size_and_speed_scale_with_the_stage():
    """**プリセットの寸法は 1080p 基準**なので、解像度に合わせて掛け直します。"""
    full = create_particle_system({"preset": "snow"}, 1920, 1080)
    half = create_particle_system({"preset": "snow"}, 960, 540)
    assert half.size == pytest.approx(full.size / 2)
    assert half.speed == pytest.approx(full.speed / 2)


def test_emitter_overrides_the_preset():
    """利用者が書いた値はそのプロジェクトの座標系。**掛け直しません。**"""
    made = create_particle_system({"preset": "snow", "size": 40, "rate": 5}, 960, 540)
    assert made.size == 40
    assert made.rate == 5


def test_presets_get_a_prewarm_by_default():
    made = create_particle_system({"preset": "sakura"}, 1920, 1080)
    assert made.prewarm > 0


def test_unknown_preset_is_reported():
    with pytest.raises(ValueError, match="プリセット"):
        create_particle_system({"preset": "there-is-no-such-preset"}, 1920, 1080)


def test_stage_dependent_presets_use_the_width():
    rain = resolve_preset("rain", 1920, 1080)
    assert rain["width"] == 1920
    assert rain["y"] < 0  # 画面の上から降らせる
    assert callable(PARTICLE_PRESETS["rain"])
    assert isinstance(PARTICLE_PRESETS["spark"], dict)


# ── 粒 1 個の形 ──────────────────────────────────────────────────


@pytest.mark.parametrize("shape", ["circle", "square", "triangle", "line", "star", "irregularQuad"])
def test_every_shape_makes_a_closed_polygon(shape):
    points = particle_contour(20.0, 14.0, 4.0, 30.0, 10.0, -5.0, 0.5, {"shape": shape})
    assert len(points) >= 6
    assert len(points) % 2 == 0


def test_align_to_velocity_turns_the_particle():
    """**雨や火花は «速度方向に伸ばす» と一気にそれらしくなります。**"""
    straight = particle_contour(20.0, 14.0, 4.0, 0.0, 0.0, 0.0, 0.5, {"shape": "line", "stretch": 4})
    aligned = particle_contour(20.0, 14.0, 4.0, 0.0, 0.0, 30.0, 0.5,
                               {"shape": "line", "stretch": 4, "alignToVelocity": True})
    assert straight != aligned


def test_irregular_quad_is_seeded():
    a = particle_contour(20.0, 14.0, 4.0, 0.0, 0.0, 0.0, 0.25, {"shape": "irregularQuad"})
    b = particle_contour(20.0, 14.0, 4.0, 0.0, 0.0, 0.0, 0.25, {"shape": "irregularQuad"})
    c = particle_contour(20.0, 14.0, 4.0, 0.0, 0.0, 0.0, 0.75, {"shape": "irregularQuad"})
    assert a == b
    assert a != c


# ── 描画 ─────────────────────────────────────────────────────────


def test_render_paints_something():
    sys_ = system()
    run(sys_, 10)
    buffer = render_particles(sys_, {"color": "#ff0000"}, 40, 28)
    assert not buffer.is_empty
    painted = buffer.data[..., 3] > 0
    assert buffer.data[painted][..., 0].max() > 0


def test_render_with_no_particles_is_empty():
    buffer = render_particles(system(), {}, 40, 28)
    assert buffer.is_empty


def test_end_color_shifts_over_the_lifetime():
    """`endColor` を書くと寿命の進みで色が移ること。"""
    sys_ = system(lifetime=1.0, lifetimeVariance=0)
    run(sys_, 15)
    start_only = render_particles(sys_, {"color": "#ff0000"}, 40, 28)
    with_end = render_particles(sys_, {"color": "#ff0000", "endColor": "#0000ff"}, 40, 28)
    assert not np.array_equal(start_only.data, with_end.data)


def test_render_with_a_sprite():
    sprite = Bitmap(4, 4)
    sprite.data[...] = (0, 255, 0, 255)
    sys_ = system()
    run(sys_, 10)
    buffer = render_particles(sys_, {}, 40, 28, sprite=sprite)
    assert not buffer.is_empty


def test_render_is_relative_to_the_emitter():
    """粒はワールド座標で進むので、**エミッターからの相対で描きます。**

    こうしておくとレイヤーのトランスフォームでまとめて動かせます。
    """
    sys_ = system()
    run(sys_, 10)
    here = render_particles(sys_, {"color": "#ffffff"}, 40, 28, {"x": 20, "y": 14})
    moved = render_particles(sys_, {"color": "#ffffff"}, 40, 28, {"x": 0, "y": 0})
    assert not np.array_equal(here.data, moved.data)


def test_spawn_uses_the_emitter_box():
    """箱の大きさが 0 なら、粒はエミッターの点から生まれること。

    速度と重力を止めて «生まれた位置そのもの» を見ます（進めてしまうと
    1 ステップぶん動いた後の位置になります）。
    """
    sys_ = system(width=0, height=0, rate=30, x=20, y=14, speed=0, gravityScale=0)
    sys_.step(1 / 30, WORLD)
    assert sys_.p_x[0] == pytest.approx(20)
    assert sys_.p_y[0] == pytest.approx(14)


def test_direction_and_spread_are_degrees():
    """`direction` は **度**で、-90 が «真上» です（画面の y は下向き）。"""
    sys_ = system(direction=-90, spread=0, speed=100, speedVariance=0, rate=30, gravityScale=0, drag=0)
    sys_.step(1 / 30, WORLD)
    assert sys_.p_vy[0] == pytest.approx(-100.0)
    assert sys_.p_vx[0] == pytest.approx(0.0, abs=1e-9)
