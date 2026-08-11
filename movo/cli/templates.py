"""`movo init` が作るプロジェクトの雛形。

どのテンプレートも **そのまま render できます**。サンプル画像は手続き的に作るので、
新しいプロジェクトに «ダウンロード» が要りません。

画像の生成は **NumPy の一括演算** で書いています。JS 版は画素ごとの二重ループ
でしたが、Python でそれをやると 512x512 の 1 枚に数秒かかります
（README.ja.md の «判断 1» と同じ理由です）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

TEMPLATES = ["basic", "text", "physics", "character", "showcase"]


def _base_project(options: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://movo.dev/schema/project-v1.json",
        "movoVersion": "1.0",
        "project": {"name": options.get("name"), "seed": 12345},
        "video": {
            "width": options.get("width") or 1920,
            "height": options.get("height") or 1080,
            "fps": options.get("fps") or 30,
            "duration": options.get("duration") or 6,
            "background": "#0f1220",
        },
        "assets": {},
        "variables": {},
        "plugins": [],
        "scenes": [],
        "audio": [],
        "render": {"quality": "standard"},
        "output": {"format": "mp4", "codec": "h264"},
    }


def build_template(template: str, options: dict[str, Any]) -> dict[str, Any]:
    project = _base_project(options)
    if template == "text":
        return _text_template(project, options)
    if template == "physics":
        return _physics_template(project)
    if template == "character":
        return _character_template(project, options)
    if template == "showcase":
        return _showcase_template(project, options)
    return _basic_template(project, options)


def _basic_template(project: dict, options: dict) -> dict:
    width = project["video"]["width"]
    height = project["video"]["height"]
    project["assets"] = {"logo": "assets/images/logo.png"}
    project["scenes"] = [
        {
            "id": "main",
            "start": 0,
            "duration": project["video"]["duration"],
            "layers": [
                {
                    "id": "title",
                    "type": "text",
                    "text": options.get("name"),
                    "style": {"size": 96, "color": "#ffffff", "align": "center", "family": options.get("font")},
                    "transform": {"x": width / 2, "y": 260, "anchorX": 0.5, "anchorY": 0.5},
                    "animations": [
                        {
                            "property": "transform.opacity",
                            "keyframes": [
                                {"time": 0, "value": 0},
                                {"time": 1, "value": 1, "easing": "easeOut"},
                            ],
                        }
                    ],
                },
                {
                    "id": "logo",
                    "type": "image",
                    "asset": "logo",
                    "transform": {"x": width / 2, "y": height / 2 + 60, "anchorX": 0.5, "anchorY": 0.5},
                    "animations": [
                        {
                            "property": "transform.y",
                            "modulator": {
                                "type": "sine",
                                "frequency": 0.5,
                                "amplitude": 24,
                                "offset": height / 2 + 60,
                            },
                        },
                        {"property": "transform.rotation", "expression": "sin(time * 1.2) * 8"},
                    ],
                    "modifiers": [
                        {"id": "wobble", "type": "wave", "axis": "x", "amplitude": 6, "frequency": 2, "speed": 1}
                    ],
                },
            ],
        }
    ]
    return project


def _text_template(project: dict, options: dict) -> dict:
    width = project["video"]["width"]
    height = project["video"]["height"]
    project["video"]["background"] = "#101820"
    project["scenes"] = [
        {
            "id": "opening",
            "start": 0,
            "duration": 3,
            "transition": {"type": "fade", "duration": 0.4},
            "layers": [
                {
                    "id": "headline",
                    "type": "text",
                    "text": options.get("name"),
                    "style": {
                        "size": 120,
                        "color": "#ffd166",
                        "align": "center",
                        "family": options.get("font"),
                        "stroke": {"color": "#00000088", "width": 6},
                    },
                    "transform": {"x": width / 2, "y": height / 2, "anchorX": 0.5, "anchorY": 0.5},
                    "animations": [
                        {
                            "property": f"transform.scale{axis}",
                            "keyframes": [
                                {"time": 0, "value": 0.7},
                                {"time": 0.8, "value": 1, "easing": "easeOutBack"},
                            ],
                        }
                        for axis in ("X", "Y")
                    ],
                }
            ],
        },
        {
            "id": "body",
            "start": 3,
            "duration": 3,
            "layers": [
                {
                    "id": "lines",
                    "type": "text",
                    "text": "JSON で\n動画を組み立てる",
                    "style": {
                        "size": 72,
                        "color": "#ffffff",
                        "align": "center",
                        "lineHeight": 1.4,
                        "family": options.get("font"),
                    },
                    "transform": {"x": width / 2, "y": height / 2, "anchorX": 0.5, "anchorY": 0.5},
                    "modifiers": [{"type": "wave", "axis": "x", "amplitude": 8, "frequency": 1.5, "speed": 0.8}],
                }
            ],
        },
    ]
    return project


def _physics_template(project: dict) -> dict:
    project["video"]["background"] = "#1b1d2b"
    project["assets"] = {"ball": "assets/images/ball.png"}
    project["physicsWorld"] = {
        "engine": "movo-physics-2d",
        "gravity": {"x": 0, "y": 1600},
        "timeStep": 0.0166667,
        "subSteps": 2,
        "iterations": 8,
    }
    balls = [
        {
            "id": f"ball{i}",
            "type": "image",
            "asset": "ball",
            "transform": {"x": 560 + i * 220, "y": 120 + i * 60, "width": 120, "height": 120, "anchorX": 0.5, "anchorY": 0.5},
            "physics": {
                "type": "rigidBody",
                "bodyType": "dynamic",
                "shape": {"type": "circle", "radius": 60},
                "mass": 1,
                "friction": 0.25,
                "restitution": 0.72,
                "linearDamping": 0.02,
                "angularDamping": 0.05,
            },
        }
        for i in range(4)
    ]
    project["scenes"] = [
        {
            "id": "main",
            "start": 0,
            "duration": project["video"]["duration"],
            "layers": [
                {
                    "id": "floor",
                    "type": "shape",
                    "shape": {"type": "rectangle", "width": 1720, "height": 60, "radius": 12, "fill": "#3d4468"},
                    "transform": {"x": 960, "y": 960, "anchorX": 0.5, "anchorY": 0.5},
                    "physics": {
                        "type": "rigidBody",
                        "bodyType": "static",
                        "shape": {"type": "rectangle", "width": 1720, "height": 60},
                        "friction": 0.4,
                        "restitution": 0.4,
                    },
                },
                {
                    "id": "wallLeft",
                    "type": "shape",
                    "shape": {"type": "rectangle", "width": 60, "height": 900, "fill": "#2a2f4a"},
                    "transform": {"x": 120, "y": 520, "anchorX": 0.5, "anchorY": 0.5},
                    "physics": {
                        "type": "rigidBody",
                        "bodyType": "static",
                        "shape": {"type": "rectangle", "width": 60, "height": 900},
                        "restitution": 0.6,
                    },
                },
                {
                    "id": "wallRight",
                    "type": "shape",
                    "shape": {"type": "rectangle", "width": 60, "height": 900, "fill": "#2a2f4a"},
                    "transform": {"x": 1800, "y": 520, "anchorX": 0.5, "anchorY": 0.5},
                    "physics": {
                        "type": "rigidBody",
                        "bodyType": "static",
                        "shape": {"type": "rectangle", "width": 60, "height": 900},
                        "restitution": 0.6,
                    },
                },
                *balls,
            ],
        }
    ]
    return project


def _character_template(project: dict, options: dict) -> dict:
    project["assets"] = {
        "body": "assets/images/part-body.png",
        "head": "assets/images/part-head.png",
        "armUpper": "assets/images/part-arm-upper.png",
        "armLower": "assets/images/part-arm-lower.png",
    }
    project["characters"] = {
        "person01": {
            "parts": [
                {"id": "body", "asset": "body", "parent": None, "pivot": [0.5, 0.5]},
                {"id": "head", "asset": "head", "parent": "body", "position": [0, -170], "pivot": [0.5, 0.9]},
                {
                    "id": "upperArmRight",
                    "asset": "armUpper",
                    "parent": "body",
                    "position": [70, -100],
                    "pivot": [0.5, 0.1],
                },
                {
                    "id": "lowerArmRight",
                    "asset": "armLower",
                    "parent": "upperArmRight",
                    "position": [0, 150],
                    "pivot": [0.5, 0.1],
                },
            ],
            "ik": [],
        }
    }
    project["scenes"] = [
        {
            "id": "main",
            "start": 0,
            "duration": project["video"]["duration"],
            "layers": [
                {
                    "id": "person",
                    "type": "character",
                    "character": "person01",
                    "transform": {"x": 960, "y": 620, "anchorX": 0.5, "anchorY": 0.5},
                    "motion": {
                        "tracks": [
                            {
                                "part": "upperArmRight",
                                "property": "rotation",
                                "modulator": {"type": "sine", "frequency": 0.8, "amplitude": 35},
                            },
                            {
                                "part": "lowerArmRight",
                                "property": "rotation",
                                "modulator": {"type": "sine", "frequency": 0.8, "amplitude": 25, "phase": 0.25},
                            },
                            {
                                "part": "head",
                                "property": "rotation",
                                "modulator": {"type": "sine", "frequency": 0.4, "amplitude": 6},
                            },
                        ]
                    },
                },
                {
                    "id": "caption",
                    "type": "text",
                    "text": "パーツアニメーション",
                    "style": {"size": 64, "color": "#ffffff", "align": "center", "family": options.get("font")},
                    "transform": {"x": 960, "y": 980, "anchorX": 0.5, "anchorY": 0.5},
                },
            ],
        }
    ]
    return project


def _showcase_template(project: dict, options: dict) -> dict:
    project["video"]["duration"] = 10
    project["assets"] = {
        "logo": "assets/images/logo.png",
        "ball": "assets/images/ball.png",
        "noiseMap": "assets/images/noise.png",
    }
    project["physicsWorld"] = {"gravity": {"x": 0, "y": 1400}, "timeStep": 0.0166667, "subSteps": 2, "iterations": 8}
    drops = [
        {
            "id": f"drop{i}",
            "type": "image",
            "asset": "ball",
            "transform": {"x": 700 + i * 260, "y": 140, "width": 120, "height": 120, "anchorX": 0.5, "anchorY": 0.5},
            "physics": {
                "type": "rigidBody",
                "bodyType": "dynamic",
                "shape": {"type": "circle", "radius": 60},
                "mass": 1,
                "restitution": 0.78,
                "friction": 0.2,
            },
        }
        for i in range(3)
    ]
    project["scenes"] = [
        {
            "id": "opening",
            "start": 0,
            "duration": 4,
            "transition": {"type": "fade", "duration": 0.5},
            "layers": [
                {
                    "id": "title",
                    "type": "text",
                    "text": options.get("name"),
                    "style": {"size": 110, "color": "#ffffff", "align": "center", "family": options.get("font")},
                    "transform": {"x": 960, "y": 420, "anchorX": 0.5, "anchorY": 0.5},
                    "animations": [
                        {
                            "property": "transform.rotation",
                            "modulators": [
                                {"type": "sine", "frequency": 0.6, "amplitude": 3},
                                {"type": "noise", "frequency": 6, "amplitude": 0.8},
                            ],
                            "combine": "add",
                        }
                    ],
                    "modifiers": [
                        {
                            "id": "titleWave",
                            "type": "wave",
                            "axis": "x",
                            "amplitude": {"keyframes": [{"time": 0, "value": 0}, {"time": 2, "value": 14}]},
                            "frequency": 2,
                            "phase": {"expression": "time * 0.8"},
                        }
                    ],
                },
                {
                    "id": "logo",
                    "type": "image",
                    "asset": "logo",
                    "transform": {"x": 960, "y": 720, "anchorX": 0.5, "anchorY": 0.5, "scale": 0.8},
                    "modifiers": [
                        {
                            "type": "twist",
                            "angle": {"expression": "sin(time) * 30"},
                            "center": {"x": 0.5, "y": 0.5},
                            "radius": 0.7,
                        },
                        {"type": "displacement", "mapAsset": "noiseMap", "amountX": 8, "amountY": 8, "scrollX": 0.05},
                    ],
                },
            ],
        },
        {
            "id": "physics",
            "start": 4,
            "duration": 6,
            "layers": [
                {
                    "id": "ground",
                    "type": "shape",
                    "shape": {
                        "type": "rectangle",
                        "width": 1700,
                        "height": 50,
                        "radius": 8,
                        "fill": {
                            "type": "linear",
                            "angle": 90,
                            "stops": [{"offset": 0, "color": "#5b6bb5"}, {"offset": 1, "color": "#232946"}],
                        },
                    },
                    "transform": {"x": 960, "y": 940, "anchorX": 0.5, "anchorY": 0.5},
                    "physics": {
                        "type": "rigidBody",
                        "bodyType": "static",
                        "shape": {"type": "rectangle", "width": 1700, "height": 50},
                        "restitution": 0.5,
                    },
                },
                *drops,
                {
                    "id": "sparks",
                    "type": "particle",
                    "transform": {"x": 960, "y": 900},
                    "emitter": {
                        "rate": 90,
                        "lifetime": 1.4,
                        "speed": 380,
                        "spread": 70,
                        "direction": -90,
                        "size": 10,
                        "color": "#ffd166",
                        "endColor": "#ef476f",
                        "gravityScale": 0.6,
                    },
                },
                {
                    "id": "label",
                    "type": "text",
                    "text": "重力・衝突・パーティクル",
                    "style": {"size": 56, "color": "#ffffff", "align": "center", "family": options.get("font")},
                    "transform": {"x": 960, "y": 160, "anchorX": 0.5, "anchorY": 0.5},
                },
            ],
        },
    ]
    return project


# ── サンプル画像（すべて NumPy の一括演算で作ります）─────────────


def build_sample_assets(template: str) -> dict[str, bytes]:
    """`movo render` が init 直後に通るようにする、手続き的なサンプル画像。"""
    from . import bridge

    files: dict[str, Any] = {"assets/images/logo.png": _make_logo(512, 512)}
    if template in ("physics", "showcase"):
        files["assets/images/ball.png"] = _make_ball(240)
    if template == "showcase":
        files["assets/images/noise.png"] = _make_noise(256, 256)
    if template == "character":
        files["assets/images/part-body.png"] = _make_rounded_rect(200, 320, "#4361ee")
        files["assets/images/part-head.png"] = _make_circle(160, "#f4a261")
        files["assets/images/part-arm-upper.png"] = _make_rounded_rect(60, 170, "#3a56d4")
        files["assets/images/part-arm-lower.png"] = _make_rounded_rect(52, 150, "#2f45ad")
    return {name: bytes(bridge.encode_png(_as_bitmap(rgba))) for name, rgba in files.items()}


def _as_bitmap(rgba: np.ndarray):
    from movo.core.bitmap import Bitmap

    height, width = rgba.shape[:2]
    return Bitmap(width, height, np.ascontiguousarray(rgba, dtype=np.uint8))


def _grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:height, 0:width]
    return xs.astype(np.float64), ys.astype(np.float64)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _make_logo(width: int, height: int) -> np.ndarray:
    xs, ys = _grid(width, height)
    cx, cy = width / 2, height / 2
    radius = min(width, height) * 0.42
    distance = np.hypot(xs - cx, ys - cy)
    alpha = np.clip(radius - distance, 0, 1)

    rgba = np.zeros((height, width, 4), np.float64)
    t = (xs / width + ys / height) / 2
    rgba[..., 0] = 60 + t * 120
    rgba[..., 1] = 120 + (1 - t) * 90
    rgba[..., 2] = 220
    rgba[..., 3] = alpha * 255

    # 円板から «M» を白く抜く。線分ごとの距離を一括で計算します。
    stroke = radius * 0.16
    points = [
        (cx - radius * 0.5, cy + radius * 0.45),
        (cx - radius * 0.5, cy - radius * 0.45),
        (cx, cy + radius * 0.05),
        (cx + radius * 0.5, cy - radius * 0.45),
        (cx + radius * 0.5, cy + radius * 0.45),
    ]
    near = np.zeros((height, width), bool)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy or 1
        t2 = np.clip(((xs - x0) * dx + (ys - y0) * dy) / length_sq, 0, 1)
        near |= np.hypot(xs - (x0 + dx * t2), ys - (y0 + dy * t2)) < stroke / 2
    mark = near & (rgba[..., 3] >= 8)
    rgba[mark, 0:3] = 255
    return rgba


def _make_ball(size: int) -> np.ndarray:
    xs, ys = _grid(size, size)
    c = size / 2
    radius = size / 2 - 1
    dx, dy = xs - c, ys - c
    distance = np.hypot(dx, dy)
    inside = distance <= radius
    shade = np.clip(1 - np.hypot(dx + radius * 0.3, dy + radius * 0.35) / (radius * 1.6), 0, None)
    tone = 0.45 + shade * 0.75
    rgba = np.zeros((size, size, 4), np.float64)
    rgba[..., 0] = np.where(inside, 230 * tone, 0)
    rgba[..., 1] = np.where(inside, 90 * tone, 0)
    rgba[..., 2] = np.where(inside, 120 * tone, 0)
    rgba[..., 3] = np.where(inside, np.clip(radius - distance, 0, 1) * 255, 0)
    return rgba


def _make_noise(width: int, height: int) -> np.ndarray:
    """value ノイズの模様。

    `movo.core.rng` の `value_noise_2d` が来たらそちらを使います（同じシードで
    JS 版と同じ模様になるため）。無い間は決定的な擬似乱数で代用します
    — **その部分は後で繋ぐ**（模様が違うだけで、テンプレートの動作は同じです）。
    """
    from . import bridge

    noise = bridge.pick("movo.core.rng", "value_noise_2d", "valueNoise2D")
    xs, ys = _grid(width, height)
    if not getattr(noise, "movo_not_connected", False):
        values = np.vectorize(lambda x, y: noise(x / 24, y / 24, 4242))(xs, ys)
        values = (values + 1) / 2
    else:
        generator = np.random.default_rng(4242)
        coarse = generator.random(((height // 24) + 2, (width // 24) + 2))
        yi = (ys / 24).astype(int)
        xi = (xs / 24).astype(int)
        values = coarse[yi, xi]
    gray = np.round(values * 255)
    rgba = np.zeros((height, width, 4), np.float64)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = 255
    return rgba


def _make_rounded_rect(width: int, height: int, color: str) -> np.ndarray:
    xs, ys = _grid(width, height)
    radius = min(width, height) * 0.3
    dx = np.maximum.reduce([radius - xs, xs - (width - radius), np.zeros_like(xs)])
    dy = np.maximum.reduce([radius - ys, ys - (height - radius), np.zeros_like(ys)])
    inside = np.hypot(dx, dy) <= radius
    r, g, b = _hex_to_rgb(color)
    shade = 0.75 + 0.25 * (1 - ys / height)
    rgba = np.zeros((height, width, 4), np.float64)
    rgba[..., 0] = np.where(inside, r * shade, 0)
    rgba[..., 1] = np.where(inside, g * shade, 0)
    rgba[..., 2] = np.where(inside, b * shade, 0)
    rgba[..., 3] = np.where(inside, 255, 0)
    return rgba


def _make_circle(size: int, color: str) -> np.ndarray:
    xs, ys = _grid(size, size)
    c = size / 2
    distance = np.hypot(xs - c, ys - c)
    inside = distance <= c - 1
    r, g, b = _hex_to_rgb(color)
    rgba = np.zeros((size, size, 4), np.float64)
    rgba[..., 0] = np.where(inside, r, 0)
    rgba[..., 1] = np.where(inside, g, 0)
    rgba[..., 2] = np.where(inside, b, 0)
    rgba[..., 3] = np.where(inside, np.clip(c - distance, 0, 1) * 255, 0)
    return rgba
