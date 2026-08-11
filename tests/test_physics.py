"""物理演算の検査（JS 版 tests/physics.test.js の移植）。

**同じ検査を同じ順番で**並べてあります。JS 版が守っている性質が
Python 版でも成り立っているか、1 対 1 で見比べられるようにするためです。

JS 版との «数値そのものの» 突き合わせは `tests/test_parity.py` にあります。
"""

from __future__ import annotations

import math

import pytest

from movo.core.bitmap import Bitmap
from movo.physics import (
    Body,
    ParticleSystem,
    SoftChain,
    World,
    alpha_outline_shape,
    capsule_shape,
    circle_shape,
    collide,
    create_shape,
    movo_physics_2d,
    polygon_shape,
    rectangle_shape,
)


def falling_ball_world():
    world = World({"gravity": {"x": 0, "y": 1000}, "timeStep": 1 / 120, "subSteps": 2, "iterations": 10})
    world.add_body(Body(id="floor", bodyType="static", shape=rectangle_shape(1000, 40), x=500, y=800))
    ball = Body(
        id="ball", bodyType="dynamic", shape=circle_shape(50), x=500, y=200,
        mass=1, restitution=0, friction=0.5, linearDamping=0.4,
    )
    world.add_body(ball)
    return world, ball


def test_gravity_accelerates_a_dynamic_body():
    world, ball = falling_ball_world()
    start_y = ball.position.y
    world.step(1 / 60)
    assert ball.position.y > start_y
    assert ball.velocity.y > 0


def test_static_bodies_never_move():
    world, _ = falling_ball_world()
    floor = world.body_by_id("floor")
    before = floor.position.as_tuple()
    for _ in range(60):
        world.step(1 / 60)
    assert floor.position.as_tuple() == before
    assert floor.inv_mass == 0


def test_ball_comes_to_rest_on_the_floor():
    world, ball = falling_ball_world()
    for _ in range(300):
        world.step(1 / 60)
    # 床の上面は 800 - 20 = 780。半径 50 なので中心は 730 あたりで止まる。
    assert abs(ball.position.y - 730) < 4, f"止まった位置が {ball.position.y}"
    assert abs(ball.velocity.y) < 30, f"止まったときの速度が {ball.velocity.y}"


def test_restitution_makes_a_ball_bounce():
    world = World({"gravity": {"x": 0, "y": 1000}, "timeStep": 1 / 120, "subSteps": 2})
    world.add_body(Body(id="floor", bodyType="static", shape=rectangle_shape(1000, 40), x=500, y=800))
    ball = Body(id="ball", bodyType="dynamic", shape=circle_shape(50), x=500, y=300, mass=1, restitution=0.9)
    world.add_body(ball)
    saw_upward = False
    for _ in range(200):
        world.step(1 / 60)
        if ball.velocity.y < -100:
            saw_upward = True
    assert saw_upward, "跳ね返っていない"


def test_simulation_is_deterministic():
    world_a, ball_a = falling_ball_world()
    world_b, ball_b = falling_ball_world()
    for _ in range(120):
        world_a.step(1 / 60)
        world_b.step(1 / 60)
    assert ball_a.position.x == ball_b.position.x
    assert ball_a.position.y == ball_b.position.y


def test_collision_covers_all_shape_pairs():
    c1 = Body(id="c1", shape=circle_shape(10), x=0, y=0)
    c2 = Body(id="c2", shape=circle_shape(10), x=15, y=0)
    manifold = collide(c1, c2)
    assert manifold is not None
    assert 4 < manifold["penetration"] < 6
    assert manifold["normal"][0] > 0.9

    far = Body(id="far", shape=circle_shape(10), x=100, y=0)
    assert collide(c1, far) is None

    box = Body(id="box", shape=rectangle_shape(40, 40), x=20, y=0)
    cp = collide(c1, box)
    assert cp is not None
    assert len(cp["contacts"]) >= 1

    box2 = Body(id="box2", shape=rectangle_shape(40, 40), x=35, y=0)
    pp = collide(box, box2)
    assert pp is not None
    assert len(pp["contacts"]) >= 1


def test_collision_groups_and_masks_filter_pairs():
    world = World({"gravity": {"x": 0, "y": 0}})
    world.add_body(Body(id="a", shape=circle_shape(10), x=0, y=0, collisionGroup=1, collisionMask=1))
    world.add_body(Body(id="b", shape=circle_shape(10), x=5, y=0, collisionGroup=2, collisionMask=2))
    world.step(1 / 60)
    assert world.contacts == []


def test_distance_constraint_keeps_bodies_apart():
    world = World({"gravity": {"x": 0, "y": 1000}, "timeStep": 1 / 120})
    anchor = Body(id="anchor", bodyType="static", shape=circle_shape(5), x=0, y=0)
    weight = Body(id="weight", bodyType="dynamic", shape=circle_shape(10), x=0, y=50, mass=1)
    world.add_body(anchor)
    world.add_body(weight)
    world.add_constraint({"type": "distance", "bodyA": anchor, "bodyB": weight, "length": 200, "stiffness": 1})
    for _ in range(240):
        world.step(1 / 60)
    distance = math.hypot(weight.position.x - anchor.position.x, weight.position.y - anchor.position.y)
    assert abs(distance - 200) < 12, f"距離が {distance} まで流れた"


def test_spring_pulls_towards_rest_length():
    world = World({"gravity": {"x": 0, "y": 0}, "timeStep": 1 / 120})
    anchor = Body(id="anchor", bodyType="static", shape=circle_shape(5), x=0, y=0)
    head = Body(id="head", bodyType="dynamic", shape=circle_shape(10), x=0, y=300, mass=1)
    world.add_body(anchor)
    world.add_body(head)
    world.add_constraint(
        {"type": "spring", "bodyA": anchor, "bodyB": head, "restLength": 80, "stiffness": 120, "damping": 10}
    )
    for _ in range(600):
        world.step(1 / 60)
    assert abs(head.position.y - 80) < 40, f"ばねが {head.position.y} で落ち着いた"


def test_hinge_angle_limits():
    world = World({"gravity": {"x": 0, "y": 0}})
    upper = Body(id="upper", bodyType="dynamic", shape=rectangle_shape(20, 100), x=0, y=0, mass=1)
    lower = Body(id="lower", bodyType="dynamic", shape=rectangle_shape(20, 100), x=0, y=100, mass=1)
    world.add_body(upper)
    world.add_body(lower)
    lower.angle = math.pi  # 180 度。制限より外
    world.add_constraint(
        {"type": "hinge", "bodyA": upper, "bodyB": lower, "anchor": [0, 50], "minAngle": -20, "maxAngle": 140}
    )
    for _ in range(60):
        world.step(1 / 60)
    relative = (lower.angle - upper.angle) * 180 / math.pi
    assert relative <= 141, f"相対角が {relative}"


def test_shape_factories_and_mass_properties():
    assert circle_shape(-5).radius > 0
    assert len(rectangle_shape(40, 20).vertices) == 4
    assert len(capsule_shape(100, 20).vertices) > 8
    assert polygon_shape([[0, 0], [10, 0], [10, 10]]).type == "polygon"
    body = Body(shape=rectangle_shape(100, 100), mass=4)
    assert body.mass == 4
    assert body.inertia > 0
    assert body.inv_inertia > 0
    fixed = Body(shape=rectangle_shape(10, 10), fixedRotation=True)
    assert fixed.inv_inertia == 0


def test_alpha_outline_builds_a_hull():
    bitmap = Bitmap(64, 64)
    bitmap.data[16:48, 16:48, 3] = 255
    shape = alpha_outline_shape(bitmap, {})
    assert shape.type == "polygon"
    assert len(shape.vertices) >= 3
    max_x = float(shape.vertices[:, 0].max())
    assert 14 <= max_x <= 16.5, f"凸包の半幅が {max_x}"

    empty = alpha_outline_shape(Bitmap(16, 16), {"width": 10, "height": 10})
    assert len(empty.vertices) == 4

    downgraded = create_shape({"type": "alpha-outline"}, {"bitmap": bitmap, "width": 32, "height": 32, "alphaOutline": False})
    assert len(downgraded.vertices) == 4


def test_soft_chain_hangs_and_follows_its_anchor():
    world = World({"gravity": {"x": 0, "y": 980}, "timeStep": 1 / 120})
    chain = SoftChain(segments=6, length=180, origin={"x": 100, "y": 100}, stiffness=0.9, damping=0.1)
    world.add_soft_body(chain)
    for _ in range(120):
        world.step(1 / 60)
    tip = chain.points[-1]
    assert tip[1] > chain.points[0][1], "先端は付け根より下にあるはず"
    span = math.hypot(tip[0] - 100, tip[1] - 100)
    assert span <= 190, f"紐が {span} まで伸びた"

    chain.set_origin(400, 100)
    for _ in range(120):
        world.step(1 / 60)
    assert abs(chain.points[0][0] - 400) < 1e-6
    tip_after = chain.points[-1]
    assert tip_after[0] > 300, f"先端が付け根に付いてこない（{tip_after[0]}）"
    assert math.hypot(tip_after[0] - 400, tip_after[1] - 100) <= 190


def test_wind_pushes_a_soft_chain():
    world = World({"gravity": {"x": 0, "y": 980}, "timeStep": 1 / 120})
    chain = SoftChain(segments=6, length=180, origin={"x": 0, "y": 0})
    world.add_soft_body(chain)
    chain.set_wind(600, 0)
    for _ in range(120):
        world.step(1 / 60)
    assert chain.points[-1][0] > 5


def test_particles_emit_age_and_expire():
    world = World({"gravity": {"x": 0, "y": 500}})
    system = ParticleSystem(rate=60, lifetime=0.5, lifetimeVariance=0, speed=100, seed=3)
    for _ in range(30):
        system.step(1 / 60, world)
    assert len(system.particles) > 0
    rendered = system.render_list()
    assert len(rendered) == len(system.particles)
    for particle in rendered:
        assert 0 <= particle["opacity"] <= 1
        assert math.isfinite(particle["x"]) and math.isfinite(particle["y"])

    system.rate = 0
    for _ in range(120):
        system.step(1 / 60, world)
    assert len(system.particles) == 0, "粒が消えていない"


def test_particle_emission_is_reproducible():
    def make():
        return ParticleSystem(rate=40, lifetime=1, speed=200, seed=11)

    a = make()
    b = make()
    world = World({})
    for _ in range(20):
        a.step(1 / 60, world)
        b.step(1 / 60, world)
    assert [round(p * 1000) for p in a.particles[:, 0]] == [round(p * 1000) for p in b.particles[:, 0]]


def test_engine_plugin_surface():
    world = movo_physics_2d.create_world({"gravity": {"x": 0, "y": 100}, "timeStep": 1 / 60})
    body = movo_physics_2d.add_body(world, {"id": "b", "shape": circle_shape(10), "x": 0, "y": 0, "mass": 1})
    movo_physics_2d.step(world, 1 / 60)
    state = movo_physics_2d.get_body_state(body)
    assert state["position"][1] > 0
    movo_physics_2d.set_body_transform(body, 5, 6, 0.5)
    assert body.position.x == 5
    assert "softChain" in movo_physics_2d.features()
    movo_physics_2d.dispose(world)
    assert world.bodies == []


def test_unstable_state_is_caught_and_reset():
    world = World({})
    body = Body(id="x", bodyType="dynamic", shape=circle_shape(5), x=0, y=0, mass=1)
    world.add_body(body)
    body.position.x = math.nan
    world.step(1 / 60)
    assert math.isfinite(body.position.x)
    assert world.unstable is True


def test_particles_reset_returns_to_the_same_sequence():
    world = World({"gravity": {"x": 0, "y": 500}})
    system = ParticleSystem(seed=4242, rate=40, lifetime=2, speed=100)

    def run():
        for _ in range(30):
            system.step(1 / 30, world)
        return [round(p * 1000) for p in system.particles[:, 0]]

    first = run()
    system.reset()
    assert run() == first


def test_particles_prewarm():
    world = World({"gravity": {"x": 0, "y": 500}})
    cold = ParticleSystem(seed=7, rate=60, lifetime=3, speed=80)
    warm = ParticleSystem(seed=7, rate=60, lifetime=3, speed=80, prewarm=2.5)
    warm.warmup(world, 1 / 30)
    assert len(cold.particles) == 0
    assert len(warm.particles) > 20, f"空回ししたのに {len(warm.particles)} 個しかない"
    assert warm.time == 0


def test_particles_prewarm_is_deterministic():
    world = World({"gravity": {"x": 0, "y": 500}})
    system = ParticleSystem(seed=11, rate=50, lifetime=2, speed=90, prewarm=1.5)
    system.warmup(world, 1 / 30)
    first = [round(p * 1000) for p in system.particles[:, 1]]
    system.reset()
    system.warmup(world, 1 / 30)
    assert [round(p * 1000) for p in system.particles[:, 1]] == first


def test_prewarm_is_clamped():
    assert ParticleSystem(prewarm=500).prewarm == 30
    assert ParticleSystem(prewarm=-5).prewarm == 0


@pytest.mark.parametrize("removed", ["a", "b"])
def test_remove_body_keeps_the_others_intact(removed):
    """並び順を保ったまま外せること。順番が変わると結果まで変わります。"""
    world = World({"gravity": {"x": 0, "y": 0}})
    a = world.add_body(Body(id="a", shape=circle_shape(5), x=1, y=2, mass=1))
    b = world.add_body(Body(id="b", shape=circle_shape(5), x=3, y=4, mass=1))
    c = world.add_body(Body(id="c", shape=circle_shape(5), x=5, y=6, mass=1))
    target = a if removed == "a" else b
    world.remove_body(target)
    remaining = {body.id: body.position.as_tuple() for body in world.bodies}
    assert "c" in remaining and remaining["c"] == (5.0, 6.0)
    # 外した剛体も値を持ったまま
    assert target.position.as_tuple() == ((1.0, 2.0) if removed == "a" else (3.0, 4.0))
    assert c.position.as_tuple() == (5.0, 6.0)
