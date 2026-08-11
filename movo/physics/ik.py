"""逆運動学（JS 版 packages/character/src/ik.js の移植）。

FABRIK を使います。長さがいくつの鎖でも扱えて、収束が速く、**届かない
ところを狙われても «真っ直ぐ伸びる» という素直な壊れ方**をするからです。

Numba にしていないのは、関節がふつう 2〜4 個しかないからです。
JIT の呼び出し手間のほうが計算より大きく、実測で 3 倍遅くなりました。
"""

from __future__ import annotations

import math


def solve_fabrik(joints, target, options: dict | None = None) -> list[list[float]]:
    """鎖の先端が `target` に届くように関節を並べ直す。

    :param joints: 関節の座標（根元が先頭）。``[(x, y), ...]``
    :param target: 目標 ``(x, y)``
    :param options: ``iterations`` / ``strength`` / ``tolerance``
    :returns: 新しい関節の座標
    """
    options = options or {}
    iterations = max(1, round(options.get("iterations", 10)))
    strength = float(options.get("strength", 1))
    tolerance = float(options.get("tolerance", 0.5))
    points = [[float(j[0]), float(j[1])] for j in joints]
    if len(points) < 2:
        return points

    lengths = []
    total_length = 0.0
    for i in range(len(points) - 1):
        length = math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        lengths.append(length)
        total_length += length

    root = [points[0][0], points[0][1]]
    goal = [
        root[0] + (float(target[0]) - root[0]) * strength,
        root[1] + (float(target[1]) - root[1]) * strength,
    ]
    distance_to_target = math.hypot(goal[0] - root[0], goal[1] - root[1])

    if distance_to_target > total_length:
        # 届かない。目標へ向かって真っ直ぐ伸ばします。
        dx = (goal[0] - root[0]) / (distance_to_target or 1)
        dy = (goal[1] - root[1]) / (distance_to_target or 1)
        for i in range(1, len(points)):
            points[i][0] = points[i - 1][0] + dx * lengths[i - 1]
            points[i][1] = points[i - 1][1] + dy * lengths[i - 1]
        return points

    for _ in range(iterations):
        tip = points[-1]
        if math.hypot(tip[0] - goal[0], tip[1] - goal[1]) < tolerance:
            break
        # 後ろ向き：先端を目標に乗せてから根元へ辿る
        points[-1] = [goal[0], goal[1]]
        for i in range(len(points) - 2, -1, -1):
            dx = points[i][0] - points[i + 1][0]
            dy = points[i][1] - points[i + 1][1]
            distance = math.hypot(dx, dy) or 1e-9
            ratio = lengths[i] / distance
            points[i] = [points[i + 1][0] + dx * ratio, points[i + 1][1] + dy * ratio]
        # 前向き：根元を元の場所へ戻してから先端へ辿る
        points[0] = [root[0], root[1]]
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            distance = math.hypot(dx, dy) or 1e-9
            ratio = lengths[i - 1] / distance
            points[i] = [points[i - 1][0] + dx * ratio, points[i - 1][1] + dy * ratio]
    return points


def solve_two_bone(root, target, length_a: float, length_b: float, flip: bool = False) -> tuple[float, float]:
    """2 本の骨を «解析的に» 解く。肘の向きを決め打ちしたいときに使います。

    :returns: ``(角度A, 角度B)``。ラジアン・世界座標。
    """
    dx = float(target[0]) - float(root[0])
    dy = float(target[1]) - float(root[1])
    distance = min(math.hypot(dx, dy), length_a + length_b - 1e-6)
    base_angle = math.atan2(dy, dx)
    cos_a = (distance * distance + length_a * length_a - length_b * length_b) / ((2 * distance * length_a) or 1e-9)
    angle_offset = math.acos(max(-1.0, min(1.0, cos_a)))
    angle_a = base_angle + (-angle_offset if flip else angle_offset)
    cos_b = (length_a * length_a + length_b * length_b - distance * distance) / ((2 * length_a * length_b) or 1e-9)
    interior = math.acos(max(-1.0, min(1.0, cos_b)))
    angle_b = angle_a + (math.pi - interior if flip else interior - math.pi)
    return (angle_a, angle_b)


def clamp_angle(radians: float, min_degrees=None, max_degrees=None) -> float:
    """角度（ラジアン）を度で書かれた範囲に収める。範囲が無ければ素通し。"""
    if min_degrees is None and max_degrees is None:
        return radians
    degrees = radians * 180 / math.pi
    while degrees > 180:
        degrees -= 360
    while degrees < -180:
        degrees += 360
    if min_degrees is not None and degrees < min_degrees:
        degrees = float(min_degrees)
    if max_degrees is not None and degrees > max_degrees:
        degrees = float(max_degrees)
    return degrees * math.pi / 180
