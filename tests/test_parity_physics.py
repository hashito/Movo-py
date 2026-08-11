"""JS 版と «数値そのもの» を突き合わせる（物理演算）。

移植でいちばん怖いのは «それらしく動くが JS 版とは違う» 状態です。
テストが «床で止まる» のような性質しか見ていないと、係数を 1 つ間違えても
通ってしまいます。そこで **JS 版を実際に走らせた軌跡**を
`tests/data/parity_physics.json` に置き、そこと突き合わせています。

基準の作り直しかた（JS 版に手が入ったとき）:

    node tests/data/parity_physics.mjs > tests/data/parity_physics.json

## どれくらい合っていれば «同じ» か

**どちらも float64 で、演算の順番も揃えてあるので、原理的には完全一致します。**
実際、下の許容差 1e-9 は «JSON に 10 桁で書いた» ぶんの丸めしか見ていません。
桁落ちで少しずつ離れる作りにはなっていない、ということの確認です。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from movo.physics import (
    Body,
    ParticleSystem,
    SoftChain,
    World,
    capsule_shape,
    circle_shape,
    polygon_shape,
    rectangle_shape,
)

GOLDEN = json.loads((Path(__file__).parent / "data" / "parity_physics.json").read_text("utf-8"))
TOLERANCE = 1e-9


def _trace(build, steps: int = 240):
    world, watch = build()
    rows = []
    for i in range(steps):
        world.step(1 / 60)
        if i % 20 == 0:
            rows.append([round(v, 10) for v in watch()])
    return rows


def _assert_same(label: str, rows):
    expected = GOLDEN[label]
    assert len(rows) == len(expected), f"{label}: 行数が違う"
    for r, (got, want) in enumerate(zip(rows, expected)):
        assert len(got) == len(want), f"{label}[{r}]: 列数が違う"
        for c, (a, b) in enumerate(zip(got, want)):
            assert abs(a - b) <= TOLERANCE + abs(b) * 1e-9, f"{label}[{r}][{c}]: {a} ≠ {b}（JS 版）"


def test_falling_ball_matches_js():
    def build():
        world = World({"gravity": {"x": 0, "y": 1000}, "timeStep": 1 / 120, "subSteps": 2, "iterations": 10})
        world.add_body(Body(id="floor", bodyType="static", shape=rectangle_shape(1000, 40), x=500, y=800))
        ball = Body(
            id="ball", bodyType="dynamic", shape=circle_shape(50), x=500, y=200,
            mass=1, restitution=0.4, friction=0.5, linearDamping=0.4,
        )
        world.add_body(ball)
        return world, lambda: [ball.position.x, ball.position.y, ball.velocity.y, ball.angle]

    _assert_same("falling-ball", _trace(build))


def test_box_stack_matches_js():
    """箱を積む。**多角形どうしの接触点クリップ**がずれると真っ先に崩れます。"""

    def build():
        world = World({"gravity": {"x": 30, "y": 900}, "timeStep": 1 / 60, "subSteps": 2, "iterations": 8})
        world.add_body(Body(id="ground", bodyType="static", shape=rectangle_shape(800, 40), x=400, y=600))
        boxes = []
        for i in range(5):
            b = Body(
                id=f"box{i}", bodyType="dynamic", shape=rectangle_shape(60, 60),
                x=400 + i * 3, y=500 - i * 70, mass=2, restitution=0.15, friction=0.4, angle=i * 0.05,
            )
            world.add_body(b)
            boxes.append(b)

        def watch():
            out = []
            for b in boxes:
                out += [b.position.x, b.position.y, b.angle, b.angular_velocity]
            return out

        return world, watch

    _assert_same("box-stack", _trace(build))


def test_capsule_and_bounds_match_js():
    """カプセル・三角・円と、画面の縁での跳ね返り。"""

    def build():
        world = World({
            "gravity": {"x": 0, "y": 980}, "timeStep": 1 / 60, "subSteps": 3, "iterations": 6,
            "bounds": {"minX": 0, "maxX": 400, "minY": 0, "maxY": 400, "restitution": 0.5},
        })
        a = Body(id="cap", bodyType="dynamic", shape=capsule_shape(80, 20), x=200, y=100, mass=1.5, restitution=0.5, angle=0.7)
        b = Body(id="tri", bodyType="dynamic", shape=polygon_shape([[0, 0], [50, 0], [25, 40]]), x=190, y=20, mass=1, restitution=0.3)
        c = Body(id="ball", bodyType="dynamic", shape=circle_shape(25), x=210, y=250, mass=3, restitution=0.6)
        world.add_body(a)
        world.add_body(b)
        world.add_body(c)
        return world, lambda: [
            a.position.x, a.position.y, a.angle,
            b.position.x, b.position.y, b.angle,
            c.position.x, c.position.y,
        ]

    _assert_same("capsule-mix", _trace(build))


def test_constraints_match_js():
    def build():
        world = World({"gravity": {"x": 0, "y": 1000}, "timeStep": 1 / 120, "iterations": 8})
        anchor = Body(id="anchor", bodyType="static", shape=circle_shape(5), x=0, y=0)
        w1 = Body(id="w1", bodyType="dynamic", shape=circle_shape(10), x=30, y=50, mass=1)
        w2 = Body(id="w2", bodyType="dynamic", shape=circle_shape(10), x=60, y=90, mass=2)
        world.add_body(anchor)
        world.add_body(w1)
        world.add_body(w2)
        world.add_constraint({"type": "distance", "bodyA": anchor, "bodyB": w1, "length": 120, "stiffness": 0.8})
        world.add_constraint({"type": "spring", "bodyA": w1, "bodyB": w2, "restLength": 60, "stiffness": 200, "damping": 6})
        world.add_constraint({"type": "rope", "bodyA": anchor, "bodyB": w2, "length": 250})
        return world, lambda: [
            w1.position.x, w1.position.y, w2.position.x, w2.position.y, w1.velocity.x, w2.velocity.y,
        ]

    _assert_same("constraints", _trace(build))


def test_soft_chain_matches_js():
    def build():
        world = World({"gravity": {"x": 0, "y": 980}, "timeStep": 1 / 120})
        chain = SoftChain(segments=8, length=200, origin={"x": 100, "y": 100}, stiffness=0.75, damping=0.12)
        chain.set_wind(120, -30)
        world.add_soft_body(chain)

        def watch():
            out = []
            for p in chain.points:
                out += [float(p[0]), float(p[1])]
            return out

        return world, watch

    _assert_same("soft-chain", _trace(build))


def test_particles_match_js():
    """**乱数列そのもの**の突き合わせ。1 つずれれば全部ずれます。"""
    world = World({"gravity": {"x": 20, "y": 500}})
    system = ParticleSystem(
        seed=20240801, rate=45, lifetime=1.2, lifetimeVariance=0.4, speed=180, speedVariance=0.5,
        spread=60, direction=-80, size=10, sizeVariance=0.5, spin=90, drag=0.4,
        floorY=300, bounce=0.5, width=40, height=20, x=5, y=7,
    )
    expected = GOLDEN["particles"]
    frame = 0
    for i in range(90):
        system.step(1 / 30, world)
        if i % 10 != 0:
            continue
        want = expected[frame]
        frame += 1
        got = system.particles
        assert len(got) == len(want), f"{i} ステップ目の粒の数が違う: {len(got)} ≠ {len(want)}"
        for k, row in enumerate(want):
            for c in range(10):
                assert abs(float(got[k, c]) - row[c]) <= TOLERANCE + abs(row[c]) * 1e-9, (
                    f"{i} ステップ目・{k} 番目・列 {c}: {got[k, c]} ≠ {row[c]}（JS 版）"
                )


def test_golden_file_covers_every_case():
    """基準ファイルに «見ていない項目» が残っていないこと。"""
    assert set(GOLDEN) == {"falling-ball", "box-stack", "capsule-mix", "constraints", "soft-chain", "particles"}
