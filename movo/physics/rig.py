"""人物リグ（JS 版 packages/character/src/rig.js の移植）。

リグは «部品の木» です。部品ごとに画像・親からのずれ・自分の中の軸・
動かせる回転／拡大／不透明度を持ちます。部品自身が変形を持てるので、
肘を回しながら腕を曲げる、といったことができます。

`movo/character/` を作らず `physics/` に置いているのは、README の構成表で
**physics が «剛体・拘束・IK»** を受け持つと決まっているからです。IK と
リグは同じ «骨を並べる» 計算なので、離すと共通部分が二重になります。
"""

from __future__ import annotations

import math

from ._compat import Mat2D, apply_animations, resolve_animated
from .ik import clamp_angle, solve_fabrik


class RigError(ValueError):
    """リグの書き方が壊れているとき。core の MovoError が入れば置き換えます。"""


def build_rig(spec: dict) -> dict:
    """JSON の記述から «描く順に並んだ» リグを組む。

    深さ優先で並べるので、**親が必ず子より先に変換されます**。
    """
    parts = [dict(part) for part in (spec.get("parts") or [])]
    by_id = {part.get("id"): part for part in parts}
    for part in parts:
        parent = part.get("parent")
        if parent and parent not in by_id:
            raise RigError(f'rig part "{part.get("id")}" references unknown parent "{parent}"')

    order: list[dict] = []

    def visit(part: dict, depth: int) -> None:
        if depth > 64:
            raise RigError(f'rig hierarchy for "{part.get("id")}" is cyclic')
        order.append(part)
        for child in [p for p in parts if p.get("parent") == part.get("id")]:
            visit(child, depth + 1)

    for part in [p for p in parts if not p.get("parent")]:
        visit(part, 0)
    for part in parts:
        if not any(part is entry for entry in order):
            order.append(part)

    return {"parts": parts, "byId": by_id, "order": order, "ik": spec.get("ik") or [], "id": spec.get("id")}


def resolve_rig_pose(rig: dict, ctx: dict | None = None, options: dict | None = None) -> dict:
    """1 フレームぶんの姿勢を求める。

    :returns: 部品 id → ``{part, matrix, rotation, worldRotation, origin,
        opacity, scaleX, scaleY, modifiers}``
    """
    options = options or {}
    pose: dict[str, dict] = {}
    motion = options.get("motion") or {}
    motion_tracks = motion.get("tracks") or []

    for part in rig["order"]:
        state = {
            "rotation": _num(resolve_animated(part.get("rotation"), ctx, 0), 0),
            "scaleX": _num(resolve_animated(part.get("scaleX"), ctx, 1), 1),
            "scaleY": _num(resolve_animated(part.get("scaleY"), ctx, 1), 1),
            "opacity": _num(resolve_animated(part.get("opacity"), ctx, 1), 1),
            "offsetX": 0.0,
            "offsetY": 0.0,
        }
        apply_animations(state, part.get("animations"), ctx)

        # `<部品id>.rotation` の形で書かれたモーショントラック
        for track in motion_tracks:
            if track.get("part") != part.get("id"):
                continue
            prop = str(track.get("property") or "rotation")
            if prop.startswith("transform."):
                prop = prop[len("transform."):]
            value = resolve_animated(
                {
                    "keyframes": track.get("keyframes"),
                    "expression": track.get("expression"),
                    "modulator": track.get("modulator"),
                    "modulators": track.get("modulators"),
                    "loop": motion.get("duration") if motion.get("loop") else None,
                },
                ctx,
                state.get(prop, 0),
            )
            if value is not None:
                state[prop] = (state.get(prop, 0) + value) if track.get("relative") else value

        override = (options.get("partOverrides") or {}).get(part.get("id"))
        if override:
            state.update(override)

        position = part.get("position") or [0, 0]
        parent_pose = pose.get(part.get("parent")) if part.get("parent") else None
        matrix = list(parent_pose["matrix"]) if parent_pose else Mat2D.identity()
        matrix = Mat2D.translate(matrix, (position[0] or 0) + state["offsetX"], (position[1] or 0) + state["offsetY"])
        matrix = Mat2D.rotate(matrix, math.radians(state["rotation"]))
        matrix = Mat2D.scale(matrix, state["scaleX"], state["scaleY"])

        ox, oy = Mat2D.apply(matrix, 0, 0)
        pose[part.get("id")] = {
            "part": part,
            "matrix": matrix,
            "rotation": state["rotation"],
            "worldRotation": (parent_pose["worldRotation"] if parent_pose else 0) + state["rotation"],
            "origin": (ox, oy),
            "opacity": (parent_pose["opacity"] if parent_pose else 1) * state["opacity"],
            "scaleX": state["scaleX"],
            "scaleY": state["scaleY"],
            "modifiers": part.get("modifiers") or [],
        }

    for ik in rig.get("ik") or []:
        if ik.get("enabled") is False:
            continue
        apply_ik(rig, pose, ik, ctx)

    return pose


def apply_ik(rig: dict, pose: dict, ik: dict, ctx=None) -> None:
    """IK を 1 本適用し、解いた結果を回転として書き戻す。

    影響を受けた部品とその子孫の行列を組み直します。
    """
    chain = [rig["byId"].get(part_id) for part_id in (ik.get("chain") or [])]
    chain = [part for part in chain if part]
    if len(chain) < 2:
        return

    joints = [list(pose[part["id"]]["origin"]) for part in chain]
    # 先端は最後の関節から «骨 1 本ぶん» 伸びた所にあります。
    last = chain[-1]
    last_pose = pose[last["id"]]
    tip_length = last.get("length")
    if tip_length is None:
        tip_length = _bone_length(chain[-1]) or 60
    tip_angle = math.radians(last_pose["worldRotation"])
    joints.append([
        last_pose["origin"][0] + math.cos(tip_angle) * tip_length,
        last_pose["origin"][1] + math.sin(tip_angle) * tip_length,
    ])

    target_spec = ik.get("target") or {}
    target = (
        _num(resolve_animated(target_spec.get("x"), ctx, joints[-1][0]), 0),
        _num(resolve_animated(target_spec.get("y"), ctx, joints[-1][1]), 0),
    )
    solved = solve_fabrik(joints, target, {"iterations": ik.get("iterations", 10), "strength": ik.get("strength", 1)})

    # 解いた関節の位置を «局所の回転» に戻す
    for i, part in enumerate(chain):
        entry = pose[part["id"]]
        frm = solved[i]
        to = solved[i + 1]
        world_angle = math.atan2(to[1] - frm[1], to[0] - frm[0])
        parent_pose = pose.get(part.get("parent")) if part.get("parent") else None
        parent_world = math.radians(parent_pose["worldRotation"] if parent_pose else 0)
        rest = _rest_angle_for(chain[i + 1] if i + 1 < len(chain) else None)
        local = world_angle - parent_world - rest
        local = clamp_angle(local, part.get("minAngle"), part.get("maxAngle"))
        entry["rotation"] = math.degrees(local)
        entry["worldRotation"] = math.degrees(world_angle - rest)

    # 鎖と、その下にぶら下がっているものの行列を作り直す
    affected = {part["id"] for part in chain}
    for part in rig["order"]:
        if part["id"] not in affected and not (part.get("parent") and part["parent"] in affected):
            continue
        affected.add(part["id"])
        entry = pose[part["id"]]
        parent_pose = pose.get(part.get("parent")) if part.get("parent") else None
        position = part.get("position") or [0, 0]
        matrix = list(parent_pose["matrix"]) if parent_pose else Mat2D.identity()
        matrix = Mat2D.translate(matrix, position[0] or 0, position[1] or 0)
        matrix = Mat2D.rotate(matrix, math.radians(entry["rotation"]))
        matrix = Mat2D.scale(matrix, entry["scaleX"], entry["scaleY"])
        entry["matrix"] = matrix
        entry["origin"] = Mat2D.apply(matrix, 0, 0)
        entry["worldRotation"] = (parent_pose["worldRotation"] if parent_pose else 0) + entry["rotation"]


def normalize_motion(motion):
    """モーションの記述を `{tracks: [...]}` の形に揃える。"""
    if not motion:
        return None
    if isinstance(motion, list):
        return {"tracks": motion}
    if isinstance(motion.get("tracks"), list):
        return motion
    tracks = []
    for part_id, properties in motion.items():
        if not isinstance(properties, dict):
            continue
        for prop, spec in properties.items():
            entry = {"part": part_id, "property": prop}
            entry.update(spec if isinstance(spec, dict) else {"value": spec})
            tracks.append(entry)
    return {"tracks": tracks}


def _bone_length(child):
    position = (child or {}).get("position")
    if not position:
        return None
    length = math.hypot(position[0] or 0, position[1] or 0)
    return length if length > 0 else None


def _rest_angle_for(child_part):
    """この部品から出る骨の «素の向き»（ラジアン）。"""
    if not child_part or not child_part.get("position"):
        return 0.0
    position = child_part["position"]
    return math.atan2(position[1] or 0, position[0] or 0)


def _num(value, fallback):
    return fallback if value is None else float(value)
