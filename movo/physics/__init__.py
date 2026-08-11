"""movo-physics — 公開 API と «物理エンジンプラグイン» の実装。

内蔵エンジンも、プラグインが使うのと同じ面（仕様 26 章）で外に出しています。
別のエンジンに差し替えても Movo の他の部分に手を入れずに済むようにするためです。

内容は JS 版 `packages/physics/src` と `packages/character/src` の移植です。
IK とリグをここに置いた理由は `rig.py` の冒頭に書いてあります。
"""

from __future__ import annotations

from .collision import aabb_overlap, collide
from .ik import clamp_angle, solve_fabrik, solve_two_bone
from .rig import apply_ik, build_rig, normalize_motion, resolve_rig_pose
from .shapes import (
    Shape,
    alpha_outline_shape,
    capsule_shape,
    circle_shape,
    convex_hull,
    create_shape,
    polygon_centroid,
    polygon_shape,
    rectangle_shape,
    shape_aabb,
    shape_area,
    shape_inertia,
    simplify_polygon,
)
from .soft import ParticleSystem, SoftChain
from .world import Body, World, solve_constraint

PHYSICS_FEATURES = [
    "gravity",
    "velocity",
    "acceleration",
    "friction",
    "airResistance",
    "collision",
    "restitution",
    "rotation",
    "spring",
    "distanceConstraint",
    "hingeConstraint",
    "rope",
    "pendulum",
    "particles",
    "softChain",
]

BODY_TYPES = ["static", "dynamic", "kinematic"]


class MovoPhysics2D:
    """内蔵エンジン。`PhysicsEnginePlugin` と同じ形をしています。"""

    name = "movo-physics-2d"
    version = "1.0.0"

    def create_world(self, config: dict | None = None) -> World:
        return World(config or {})

    def add_body(self, world: World, spec) -> Body:
        body = spec if isinstance(spec, Body) else Body(**spec)
        return world.add_body(body)

    def add_constraint(self, world: World, constraint: dict) -> dict:
        return world.add_constraint(constraint)

    def step(self, world: World, delta_time: float | None = None) -> None:
        world.step(world.time_step if delta_time is None else delta_time)

    def get_body_state(self, body: Body) -> dict:
        return body.state()

    def set_body_transform(self, body: Body, x: float, y: float, angle=None) -> None:
        body.position.x = x
        body.position.y = y
        if angle is not None:
            body.angle = angle

    def dispose(self, world: World) -> None:
        world.bodies.clear()
        world.constraints.clear()
        world.soft_bodies.clear()
        world._bank.count = 0
        world._verts_dirty = True

    def features(self) -> list[str]:
        return list(PHYSICS_FEATURES)


movo_physics_2d = MovoPhysics2D()


def describe_physics() -> dict:
    """`movo list physics` が出す一覧。"""
    return {
        "engine": "movo-physics-2d (built in, deterministic, fixed time step)",
        "bodyTypes": BODY_TYPES,
        "shapes": ["circle", "rectangle", "capsule", "polygon", "mesh", "alpha-outline"],
        "constraints": ["spring", "hinge", "distance", "pin", "rope"],
        "soft": ["softChain (hair, cloth, ties)", "particle systems"],
        "controlModes": ["physics", "animation", "blend", "follow", "override"],
    }


__all__ = [
    "Body", "World", "solve_constraint", "SoftChain", "ParticleSystem",
    "collide", "aabb_overlap", "Shape", "circle_shape", "rectangle_shape",
    "polygon_shape", "capsule_shape", "alpha_outline_shape", "create_shape",
    "convex_hull", "simplify_polygon", "polygon_centroid", "shape_area",
    "shape_inertia", "shape_aabb", "solve_fabrik", "solve_two_bone", "clamp_angle",
    "build_rig", "resolve_rig_pose", "apply_ik", "normalize_motion",
    "PHYSICS_FEATURES", "BODY_TYPES", "movo_physics_2d", "describe_physics",
    "MovoPhysics2D",
]
