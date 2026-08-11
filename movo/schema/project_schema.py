"""Movo プロジェクト JSON のスキーマ（v1）。

仕様 37 節の決めごと: **後方互換を保つ**。未知のプロパティはほぼどこでも通します。
プラグインが足したキーや将来のキーワードで、古い版が壊れないようにするためです。
ここで捕まえたいのは «本当に間違っている値»（負の周波数・違う enum・id 抜け）です。

**JSON のキー名は JS 版のまま**（`movoVersion` `frameHold` `timeRemap` …）です。
Python の書き方に寄せて `movo_version` にすると、JS 版で書いた JSON が動かなくなります。
"""

from __future__ import annotations

LAYER_TYPES = [
    "image",
    "video",
    "text",
    "shape",
    "audio",
    "character",
    "particle",
    "group",
    "composition",
    "shader",
    "fractalNoise",
    "frameBuffer",
    "starfield",
    "linePath",
    "neonPath",
    "metaball",
    "waterSurface",
    "primitive3d",
    "spotlight",
    "mesh",
    "shapeAnim",
    "waveform",
    "speedLines",
    "custom",
]

DEFORMER_TYPES = [
    "bend",
    "twist",
    "wave",
    "skew",
    "perspective",
    "bulge",
    "meshWarp",
    "pathDeform",
    "displacement",
    "ripple",
    "pinch",
    "sphereize",
    "turbulentDisplace",
    "melt",
    "handDrawn",
    "curveDeform",
]

EFFECT_TYPES = [
    # カラーグレーディング
    "curves",
    "colorWheels",
    "hslSecondary",
    "lut",
    # 質感と色づくり
    "retroFilm",
    "lightLeak",
    "colorama",
    "leaveColor",
    "monochrome",
    "bevel",
    "directionalLight",
    "longShadow",
    "graphicPen",
    "hexTile",
    "slitScan",
    # かけらに分けて動かすもの
    "shatter",
    "objectSplit",
    "slice",
    # MV 制作ブログの調査から追加
    "radialBlur",
    "spinBlur",
    "glitch",
    "rasterScroll",
    "diffusion",
    "lightStreak",
    "lensFlare",
    "rimLight",
    "innerGlow",
    "halftone",
    "mangaize",
    "polar",
    "tile",
    "peripheralBlur",
    "letterbox",
    "gradientOverlay",
    "luminanceKey",
    "colorKey",
    "pixelSort",
    "reflection",
    "bloom",
    "duotone",
    "posterize",
    "emboss",
    "edgeDetect",
    "mirror",
    "kaleidoscope",
    "scanlines",
    "chromaticAberration",
    "lensDistortion",
    "roundCorners",
    "feather",
    "blur",
    "directionalBlur",
    "sharpen",
    "colorAdjust",
    "tint",
    "grayscale",
    "invert",
    "threshold",
    "pixelate",
    "glow",
    "dropShadow",
    "stroke",
    "chromaKey",
    "vignette",
    "noise",
    "gradientMap",
    "opacity",
]

MASK_TYPES = [
    "rectangle",
    "ellipse",
    "sector",
    "diagonal",
    "polygon",
    "path",
    "image",
    "alpha",
    "layer",
    "segmentation",
]

SHAPE_TYPES = ["circle", "rectangle", "capsule", "polygon", "mesh", "alpha-outline"]

MODULATOR_TYPES = [
    "sine",
    "cosine",
    "triangle",
    "square",
    "sawtooth",
    "pulse",
    "noise",
    "random-step",
    "custom-curve",
    "audio-reactive",
    "shake",
    "beat",
]

RENDERER_TYPES = ["canvas-2d", "svg", "webgl", "gpu", "headless-browser", "custom"]

QUALITY_PRESETS = ["draft", "preview", "standard", "high", "ultra"]

PHYSICS_CONTROL_MODES = ["physics", "animation", "blend", "follow", "override"]

PLUGIN_KINDS = [
    "asset-provider",
    "renderer",
    "effect",
    "deformer",
    "modulator",
    "physics-engine",
    "audio-processor",
    "exporter",
    "cli-command",
    "layer-type",
]


def _number(**extra):
    return {"type": "number", **extra}


def _ref(name):
    return {"$ref": f"#/definitions/{name}"}


project_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://movo.dev/schema/project-v1.json",
    "title": "Movo Project",
    "type": "object",
    "required": ["video"],
    "properties": {
        "$schema": {"type": "string"},
        "movoVersion": {"type": "string", "pattern": r"^\d+(\.\d+)*$"},
        "project": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "author": {"type": "string"},
                "seed": {"type": "integer"},
                "bpm": {
                    "oneOf": [
                        _number(exclusiveMinimum=0, maximum=400),
                        {
                            # 曲から取る。解析は正規化より前に走り、ここは数値に潰される。
                            "type": "object",
                            "required": ["fromAudio"],
                            "properties": {
                                "fromAudio": {"type": "string"},
                                "fallback": _number(exclusiveMinimum=0, maximum=400),
                                "minBpm": _number(exclusiveMinimum=0, maximum=400),
                                "maxBpm": _number(exclusiveMinimum=0, maximum=400),
                            },
                        },
                    ]
                },
                "root": {"type": "string"},
            },
        },
        "video": {
            "type": "object",
            "required": ["width", "height"],
            "properties": {
                "width": {"type": "integer", "minimum": 1, "maximum": 16384},
                "height": {"type": "integer", "minimum": 1, "maximum": 16384},
                "fps": _number(exclusiveMinimum=0, maximum=240),
                "duration": _number(minimum=0),
                "background": _ref("color"),
                "pixelAspect": _number(exclusiveMinimum=0),
                "colorSpace": {"type": "string", "enum": ["srgb", "rec709", "linear"]},
                # 画面の端に置いた文字は、別のアスペクト比に組み替えたときに最初に
                # 切れます。«ここより外には出さない» 割合を書いておくと、意味検証が
                # はみ出しを警告します（エラーにはしません）。
                "safeArea": {
                    "type": "object",
                    "properties": {
                        "x": _number(minimum=0, maximum=0.5),
                        "y": _number(minimum=0, maximum=0.5),
                    },
                },
            },
        },
        # 同じ JSON から 16:9 / 9:16 / 1:1 を出すための «違うところだけ»。
        "variants": {"type": "object", "additionalProperties": _ref("variant")},
        "assets": {"type": "object", "additionalProperties": _ref("asset")},
        "variables": {"type": "object"},
        # 2.5D カメラ。レイヤーの transform.z と組み合わせて視差を作る。
        "camera": {
            "type": "object",
            "properties": {
                "x": _ref("animatedNumber"),
                "y": _ref("animatedNumber"),
                "z": _ref("animatedNumber"),
                "fov": _ref("animatedNumber"),
                "referenceDistance": _ref("animatedNumber"),
                "dof": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "focusZ": _ref("animatedNumber"),
                        "aperture": _ref("animatedNumber"),
                    },
                },
            },
        },
        # プリセット（エイリアス）: レイヤーに書く内容の断片に名前を付けて再利用する
        "presets": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "extends": {
                        "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]
                    },
                    "presetMerge": {"type": "string", "enum": ["concat", "replace"]},
                },
            },
        },
        # プロジェクト直下の use は、スキルを 1 枚のシーンとして展開する
        "use": {
            "oneOf": [_ref("skillUse"), {"type": "array", "items": _ref("skillUse")}]
        },
        "fonts": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "family": {"type": "string"},
                            "index": {"type": "integer", "minimum": 0},
                            # ウェイト・イタリックごとにファイルを分ける。
                            # 書かなかった面は «太さ優先» で近い面に落ちる。
                            "regular": _ref("fontFace"),
                            "bold": _ref("fontFace"),
                            "italic": _ref("fontFace"),
                            "boldItalic": _ref("fontFace"),
                        },
                    },
                ]
            },
        },
        "plugins": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "version": {"type": "string"},
                            "path": {"type": "string"},
                            "kind": {"type": "string", "enum": PLUGIN_KINDS},
                            "config": {"type": "object"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                ]
            },
        },
        "compositions": {"type": "object", "additionalProperties": _ref("composition")},
        "characters": {"type": "object", "additionalProperties": _ref("rig")},
        "physicsWorld": _ref("physicsWorld"),
        "scenes": {"type": "array", "items": _ref("scene")},
        "layers": {
            "type": "array",
            "description": "Shorthand for a single scene spanning the whole video.",
            "items": _ref("layer"),
        },
        "audio": {"type": "array", "items": _ref("audioTrack")},
        "render": _ref("render"),
        # 1 つでも配列でも書けます。配列にすると 1 回描いて何通りも書き出します。
        "output": {
            "oneOf": [_ref("output"), {"type": "array", "minItems": 1, "items": _ref("output")}]
        },
        "deterministic": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "seed": {"type": "integer"},
                "fixedTimeStep": {"type": "boolean"},
                "lockPluginVersions": {"type": "boolean"},
            },
        },
        "cache": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}, "directory": {"type": "string"}},
        },
        "security": {
            "type": "object",
            "properties": {
                "allowNetwork": {"type": "boolean"},
                "allowPlugins": {"type": "array", "items": {"type": "string"}},
                "allowFilesystemOutside": {"type": "boolean"},
                "maxDownloadSizeMB": _number(minimum=0),
            },
        },
        "ai": {
            "type": "object",
            "properties": {
                "defaultProvider": {"type": "string"},
                "providers": {"type": "object"},
            },
        },
    },
    "definitions": {
        "color": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "r": _number(minimum=0, maximum=255),
                        "g": _number(minimum=0, maximum=255),
                        "b": _number(minimum=0, maximum=255),
                        "a": _number(minimum=0, maximum=1),
                    },
                },
            ],
            "errorMessage": 'must be a colour such as "#ff0000", "rgba(0,0,0,0.5)" or {r,g,b,a}',
        },
        "easing": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                {"type": "object"},
            ]
        },
        "keyframe": {
            "type": "object",
            "required": ["time"],
            "properties": {
                "time": _number(),
                "value": {},
                "easing": _ref("easing"),
                # type: "matte" 用。グレースケール画像 1 枚で任意のワイプになります。
                "asset": {"type": "string"},
                "invert": {"type": "boolean"},
                "generator": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["fractalNoise"]},
                        "scale": _number(exclusiveMinimum=0),
                    },
                },
                "hold": {"type": "boolean"},
            },
        },
        "modulator": {
            "type": "object",
            "required": ["type"],
            "properties": {
                # 組み込みは MODULATOR_TYPES。プラグインが追加できるよう enum にはしない
                # （未知の名前は意味検証で警告になる）。
                "type": {"type": "string", "minLength": 1},
                "frequency": _ref("positiveAnimatedNumber"),
                "amplitude": _ref("animatedNumber"),
                "offset": _ref("animatedNumber"),
                "phase": _ref("animatedNumber"),
                "duty": _number(minimum=0, maximum=1),
                "width": _number(minimum=0, maximum=1),
                "softness": _number(minimum=0, maximum=0.5),
                "octaves": {"type": "integer", "minimum": 1, "maximum": 8},
                "seedOffset": {"type": "integer"},
                "curve": {"type": "array"},
                "points": {"type": "array"},
                "band": {"type": "integer", "minimum": 0},
                "smoothing": _number(minimum=0, maximum=0.99),
                "start": _number(),
                "decay": _number(minimum=0),
                "random": {"type": "boolean"},
                "bpm": _number(exclusiveMinimum=0, maximum=400),
                "division": _number(exclusiveMinimum=0),
                "beatOffset": _number(),
                "clamp": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            },
        },
        "animatedNumber": {
            "oneOf": [
                {"type": "number"},
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "value": {},
                        "base": {},
                        "keyframes": {"type": "array", "items": _ref("keyframe")},
                        "expression": {"type": "string"},
                        "modulator": _ref("modulator"),
                        "modulators": {"type": "array", "items": _ref("modulator")},
                        "combine": {
                            "type": "string",
                            "enum": ["add", "multiply", "replace", "min", "max", "average"],
                        },
                        "delay": _number(),
                        "timeScale": _number(),
                        "loop": {"oneOf": [{"type": "number"}, {"type": "boolean"}]},
                        "loopDuration": _number(minimum=0),
                        "extrapolate": {
                            "type": "string",
                            "enum": ["hold", "loop", "pingPong", "extend"],
                        },
                        "clamp": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "round": {"type": "boolean"},
                    },
                },
            ],
            "errorMessage": "must be a number, or an object with keyframes / expression / modulator(s)",
        },
        "positiveAnimatedNumber": {
            "oneOf": [
                {"type": "number", "exclusiveMinimum": 0},
                {"type": "string"},
                {"type": "object"},
            ],
            "errorMessage": "must be greater than 0",
        },
        "animatedValue": {
            "oneOf": [
                {"type": "number"},
                {"type": "string"},
                {"type": "boolean"},
                {"type": "array"},
                {"type": "object"},
            ]
        },
        "animation": {
            "type": "object",
            "required": ["property"],
            "properties": {
                "id": {"type": "string"},
                "property": {"type": "string", "minLength": 1},
                "value": {},
                "keyframes": {"type": "array", "items": _ref("keyframe")},
                "expression": {"type": "string"},
                "modulator": _ref("modulator"),
                "modulators": {"type": "array", "items": _ref("modulator")},
                "combine": {
                    "type": "string",
                    "enum": ["add", "multiply", "replace", "min", "max", "average"],
                },
                "relative": {"type": "boolean"},
                "enabled": {"type": "boolean"},
                "delay": _number(),
                "timeScale": _number(),
                "startTime": _number(),
                "endTime": _number(),
                "loop": {"oneOf": [{"type": "number"}, {"type": "boolean"}]},
                "loopDuration": _number(minimum=0),
                "extrapolate": {"type": "string", "enum": ["hold", "loop", "pingPong", "extend"]},
                "clamp": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "round": {"type": "boolean"},
            },
        },
        # バリアント（アスペクト比違い）の «違うところだけ»。
        # ここは意図的にゆるくしています。プロジェクトのどのキーでも上書きできて
        # よく、書き方の正しさは «畳んだあとの JSON» を検証すれば分かるからです。
        "variant": {
            "type": "object",
            "properties": {
                "video": {"type": "object"},
                "render": {"type": "object"},
                "output": {"type": "object"},
                "project": {"type": "object"},
                "camera": {"type": "object"},
                # id をキーにしたレイヤーの部分上書き（配列の添字ではなく名前で指す）
                "layers": {"type": "object", "additionalProperties": {"type": "object"}},
            },
        },
        "transform": {
            "type": "object",
            "properties": {
                "x": _ref("animatedNumber"),
                "y": _ref("animatedNumber"),
                # z を書くと project.camera が効く（2.5D の視差）
                "z": _ref("animatedNumber"),
                "rotation": _ref("animatedNumber"),
                "scale": _ref("animatedNumber"),
                "scaleX": _ref("animatedNumber"),
                "scaleY": _ref("animatedNumber"),
                "skewX": _ref("animatedNumber"),
                "skewY": _ref("animatedNumber"),
                "opacity": _ref("animatedNumber"),
                "anchorX": _ref("animatedNumber"),
                "anchorY": _ref("animatedNumber"),
                "width": _ref("animatedNumber"),
                "height": _ref("animatedNumber"),
                "motionPath": _ref("motionPath"),
            },
        },
        "mask": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"type": "string", "enum": MASK_TYPES},
                "x": _ref("animatedNumber"),
                "y": _ref("animatedNumber"),
                "width": _ref("animatedNumber"),
                "height": _ref("animatedNumber"),
                "rotation": _ref("animatedNumber"),
                "points": {"type": "array"},
                "path": {"type": "array"},
                # SVG のパス文字列でマスクの形を書く。`viewBox` を書けばその座標系で、
                # 書かなければパス自身の外接矩形で 0〜1 に写します。
                "d": {"type": "string"},
                "viewBox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "closed": {"type": "boolean"},
                # トリムパス。線が «描かれていく» 演出の基本です。
                "trim": _ref("trimPath"),
                "thickness": _ref("animatedNumber"),
                "startAngle": _ref("animatedNumber"),
                "endAngle": _ref("animatedNumber"),
                "innerRadius": _ref("animatedNumber"),
                "outerRadius": _ref("animatedNumber"),
                "asset": {"type": "string"},
                "layer": {"type": "string"},
                "channel": {
                    "type": "string",
                    "enum": ["red", "green", "blue", "alpha", "luminance"],
                },
                "label": {"type": "string"},
                "feather": _ref("animatedNumber"),
                "invert": {"type": "boolean"},
                "expand": _ref("animatedNumber"),
            },
        },
        # トリムパス。`shape` と `mask` の «形» を長さの割合で切ります。
        "trimPath": {
            "type": "object",
            "properties": {
                "start": _ref("animatedNumber"),
                "end": _ref("animatedNumber"),
                "offset": _ref("animatedNumber"),
                # 'each'（既定）は全サブパスを同時に、'sequential' は 1 本ずつ順に。
                "mode": {"type": "string", "enum": ["each", "sequential"]},
                # 線幅を書いていない図形をトリムしたときに使う «代わりの線幅»。
                "width": _ref("animatedNumber"),
                "enabled": {"type": "boolean"},
            },
        },
        # shape レイヤーの形。プラグインが種類を足せるよう、未知のキーは許します。
        "shape": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "minLength": 1},
                "kind": {"type": "string", "minLength": 1},
                # SVG のパス文字列。文字列の配列にすればサブパスを分けて書けます。
                "d": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                # SVG をそのまま貼る／`.svg` 素材を指す。取り込むのは «形» だけ。
                "svg": {"type": "string"},
                "asset": {"type": "string"},
                "trim": _ref("trimPath"),
                "closed": {"type": "boolean"},
                "fillRule": {"type": "string", "enum": ["nonzero", "evenodd"]},
            },
        },
        "modifier": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "minLength": 1},
                "enabled": {"type": "boolean"},
                "mask": _ref("mask"),
                "amount": _ref("animatedNumber"),
                "axis": {"type": "string", "enum": ["x", "y", "both"]},
                "origin": _ref("animatedNumber"),
                "angle": _ref("animatedNumber"),
                "radius": _ref("animatedNumber"),
                "strength": _ref("animatedNumber"),
                "amplitude": _ref("animatedNumber"),
                "frequency": _ref("positiveAnimatedNumber"),
                "speed": _ref("animatedNumber"),
                "phase": _ref("animatedNumber"),
                "columns": {"type": "integer", "minimum": 1, "maximum": 256},
                "rows": {"type": "integer", "minimum": 1, "maximum": 256},
                "points": {"type": "array"},
                "corners": {"type": "object"},
                "center": {"type": "object"},
                "path": {"type": "array"},
                "mapAsset": {"type": "string"},
                "amountX": _ref("animatedNumber"),
                "amountY": _ref("animatedNumber"),
            },
        },
        "effectNode": {
            "type": "object",
            "required": ["id", "type"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "type": {"type": "string", "minLength": 1},
            },
        },
        "effectGraph": {
            "type": "object",
            "required": ["nodes"],
            "properties": {
                "nodes": {"type": "array", "items": _ref("effectNode")},
                "connections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["from", "to"],
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "input": {"type": "string"},
                        },
                    },
                },
                "output": {"type": "string"},
            },
        },
        "physicsShape": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"type": "string", "enum": SHAPE_TYPES},
                "radius": _number(exclusiveMinimum=0),
                "width": _number(exclusiveMinimum=0),
                "height": _number(exclusiveMinimum=0),
                "length": _number(exclusiveMinimum=0),
                "points": {"type": "array"},
                "asset": {"type": "string"},
                "threshold": _number(minimum=0, maximum=255),
                "simplify": _number(minimum=0),
            },
        },
        "physics": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["rigidBody", "softChain", "particle", "none"]},
                "bodyType": {"type": "string", "enum": ["static", "dynamic", "kinematic"]},
                "shape": _ref("physicsShape"),
                "mass": _number(exclusiveMinimum=0),
                "friction": _number(minimum=0),
                "restitution": _number(minimum=0, maximum=1),
                "linearDamping": _number(minimum=0),
                "angularDamping": _number(minimum=0),
                "gravityScale": _number(),
                "fixedRotation": {"type": "boolean"},
                "velocity": {"type": "object"},
                "angularVelocity": _number(),
                "collisionGroup": {"type": "integer"},
                "collisionMask": {"type": "integer"},
                "sensor": {"type": "boolean"},
                "segments": {"type": "integer", "minimum": 1, "maximum": 128},
                "stiffness": _number(minimum=0),
                "damping": _number(minimum=0),
                "source": {"type": "string"},
                "wind": {"type": "object"},
                "control": _ref("physicsControl"),
            },
        },
        "physicsControl": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": PHYSICS_CONTROL_MODES},
                "animationWeight": _number(minimum=0, maximum=1),
                "physicsWeight": _number(minimum=0, maximum=1),
                "followStiffness": _number(minimum=0),
                "followDamping": _number(minimum=0),
                "from": _number(),
                "to": _number(),
            },
        },
        "physicsConstraint": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["spring", "hinge", "distance", "pin", "rope"]},
                "bodyA": {"type": "string"},
                "bodyB": {"type": "string"},
                "anchor": {"type": "array", "items": {"type": "number"}},
                "anchorA": {"type": "array", "items": {"type": "number"}},
                "anchorB": {"type": "array", "items": {"type": "number"}},
                "restLength": _number(minimum=0),
                "length": _number(minimum=0),
                "stiffness": _number(minimum=0),
                "damping": _number(minimum=0),
                "minAngle": _number(),
                "maxAngle": _number(),
            },
        },
        "physicsWorld": {
            "type": "object",
            "properties": {
                "engine": {"type": "string"},
                "gravity": {"type": "object", "properties": {"x": _number(), "y": _number()}},
                "timeStep": _number(exclusiveMinimum=0, maximum=1),
                "subSteps": {"type": "integer", "minimum": 1, "maximum": 32},
                "iterations": {"type": "integer", "minimum": 1, "maximum": 64},
                "pixelsPerMeter": _number(exclusiveMinimum=0),
                "bounds": {"type": "object"},
                "constraints": {"type": "array", "items": _ref("physicsConstraint")},
                "enabled": {"type": "boolean"},
            },
        },
        "rigPart": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "asset": {"type": "string"},
                "parent": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                "position": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "pivot": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "rotation": _ref("animatedNumber"),
                "scaleX": _ref("animatedNumber"),
                "scaleY": _ref("animatedNumber"),
                "opacity": _ref("animatedNumber"),
                "zIndex": {"type": "number"},
                "length": _number(minimum=0),
                "modifiers": {"type": "array", "items": _ref("modifier")},
                "animations": {"type": "array", "items": _ref("animation")},
            },
        },
        "rig": {
            "type": "object",
            "required": ["parts"],
            "properties": {
                "id": {"type": "string"},
                "parts": {"type": "array", "items": _ref("rigPart"), "minItems": 1},
                "ik": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["chain"],
                        "properties": {
                            "type": {"type": "string", "enum": ["ik"]},
                            "chain": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                            "target": {"type": "object"},
                            "iterations": {"type": "integer", "minimum": 1, "maximum": 200},
                            "strength": _number(minimum=0, maximum=1),
                            "enabled": {"type": "boolean"},
                        },
                    },
                },
            },
        },
        "asset": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            # 'lut' はカラーグレーディングのルック（.cube）。
                            # 'svg' はベクタのロゴ。取り込むのは «パスの形» だけ。
                            "enum": [
                                "image",
                                "video",
                                "audio",
                                "font",
                                "data",
                                "lyrics",
                                "lut",
                                "svg",
                                "ai-image",
                                "ai-character",
                                "ai-audio",
                                "ai-video",
                                "generated",
                            ],
                        },
                        "path": {"type": "string"},
                        "url": {"type": "string"},
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                        "prompt": {"type": "string"},
                        "negativePrompt": {"type": "string"},
                        "size": {"type": "string"},
                        "transparent": {"type": "boolean"},
                        "parts": {"type": "array", "items": {"type": "string"}},
                        "styleReference": {"type": "string"},
                        "seed": {"type": "integer"},
                        "cache": {"type": "boolean"},
                        "fallback": {"type": "string"},
                        "placeholder": {"type": "object"},
                    },
                },
            ]
        },
        "audioTrack": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "asset": {"type": "string"},
                "path": {"type": "string"},
                "start": _number(minimum=0),
                "offset": _number(minimum=0),
                "duration": _number(minimum=0),
                "volume": _ref("animatedNumber"),
                "pan": _ref("animatedNumber"),
                "loop": {"type": "boolean"},
                "fadeIn": _number(minimum=0),
                "fadeOut": _number(minimum=0),
                "enabled": {"type": "boolean"},
                # このトラックが鳴っている間、別のトラックを下げる。
                "ducks": {"type": "array", "items": _ref("audioDuck")},
            },
        },
        # オートダッキング。ナレーションの裏で BGM を下げるための指定。
        "audioDuck": {
            "type": "object",
            "required": ["target"],
            "properties": {
                # 下げる相手。audioTrack の id、無ければ asset 名で指す。
                "target": {"type": "string"},
                # 下げ幅（dB）。負で書くのが自然だが、正で書かれても下げる向きに読む。
                "amount": _number(minimum=-60, maximum=60),
                "attack": _number(minimum=0, maximum=10),
                "release": _number(minimum=0, maximum=10),
                # これを超えたら «鳴っている» とみなす線（dBFS）。
                "threshold": _number(minimum=-100, maximum=0),
                # 閾値を割ってから戻し始めるまでの間（秒）。
                "hold": _number(minimum=0, maximum=10),
            },
        },
        # 基礎アニメーションの参照（レイヤー）。展開後には残らない。
        "animationUse": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "required": ["animation"],
                    "properties": {"animation": {"type": "string"}, "with": {"type": "object"}},
                },
            ]
        },
        "fontFace": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string"},
                        "index": {"type": "integer", "minimum": 0},
                    },
                },
            ]
        },
        "sceneSkillUse": {
            "type": "object",
            "required": ["scene"],
            "properties": {
                "scene": {"type": "string"},
                "id": {"type": "string"},
                "with": {"type": "object"},
            },
        },
        "skillUse": {
            "type": "object",
            "required": ["skill"],
            "properties": {
                "skill": {"type": "string"},
                "id": {"type": "string"},
                "start": _number(minimum=0),
                "with": {"type": "object"},
            },
        },
        "layer": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "id": {"type": "string"},
                "use": {
                    "oneOf": [_ref("animationUse"), {"type": "array", "items": _ref("animationUse")}]
                },
                "preset": {
                    "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]
                },
                "presetMerge": {"type": "string", "enum": ["concat", "replace"]},
                "type": {"type": "string", "enum": LAYER_TYPES},
                "name": {"type": "string"},
                "enabled": {"type": "boolean"},
                # false にすると video.safeArea のはみ出し検査から外れる
                "safeArea": {"type": "boolean"},
                "start": _number(),
                "end": _number(),
                "duration": _number(minimum=0),
                "zIndex": {"type": "number"},
                "blend": {"type": "string"},
                "asset": {"type": "string"},
                "composition": {"type": "string"},
                "character": {"type": "string"},
                "rig": _ref("rig"),
                "text": {"oneOf": [{"type": "string"}, _ref("textContent")]},
                "transform": _ref("transform"),
                "animations": {"type": "array", "items": _ref("animation")},
                "modifiers": {"type": "array", "items": _ref("modifier")},
                "effects": {"type": "array", "items": _ref("modifier")},
                "effectGraph": _ref("effectGraph"),
                "physics": {"oneOf": [_ref("physics"), {"type": "null"}]},
                "physicsControl": _ref("physicsControl"),
                "layers": {"type": "array", "items": _ref("layer")},
                "mask": _ref("mask"),
                "motion": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                "parts": {"type": "array"},
                "shape": _ref("shape"),
                "style": _ref("textStyle"),
                "font": {"type": "object"},
                "emitter": {"type": "object"},
                "particles": {"type": "object"},
                "shader": {"type": "object"},
                "fractalNoise": _ref("fractalNoise"),
                "frameBuffer": _ref("frameBuffer"),
                "starfield": {"type": "object"},
                "linePath": {"type": "object"},
                "neonPath": {"type": "object"},
                "metaball": {"type": "object"},
                "waterSurface": {"type": "object"},
                "primitive3d": {"type": "object"},
                "spotlight": {"type": "object"},
                "mesh": {"type": "object"},
                "shapeAnim": {"type": "object"},
                "waveform": _ref("waveform"),
                "speedLines": _ref("speedLines"),
                "karaoke": _ref("karaoke"),
                "textBox": {"type": "object"},
                "textPath": {"type": "object"},
                "strokeOrder": {"type": "object"},
                "counter": _ref("counter"),
                "frameHold": _ref("frameHold"),
                "timeRemap": _ref("timeRemap"),
                # shape.trim の別名（レイヤーの性質として書きたくなるため）
                "trim": _ref("trimPath"),
                "echo": _ref("echo"),
                "frameEcho": _ref("frameEcho"),
                "regionExpand": _ref("regionExpand"),
                "kenBurns": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "object"},
                        "to": {"type": "object"},
                        "easing": {"type": "string"},
                        "duration": {"oneOf": [{"type": "number"}, {"type": "null"}]},
                    },
                },
                "parent": {"type": "string"},
                "matte": {"type": "boolean"},
                "trackMatte": _ref("trackMatte"),
                "motionBlur": _ref("motionBlur"),
                "repeater": _ref("repeater"),
                "textAnimator": _ref("textAnimator"),
                "motionPath": _ref("motionPath"),
                "offset": _number(),
                "timeScale": _number(),
                "loop": {"oneOf": [{"type": "boolean"}, {"type": "number"}]},
                "layerType": {"type": "string"},
            },
        },
        "regionExpand": {
            "type": "object",
            "properties": {
                "all": _number(minimum=0),
                "top": _number(minimum=0),
                "right": _number(minimum=0),
                "bottom": _number(minimum=0),
                "left": _number(minimum=0),
                "fill": {"type": "string"},
            },
        },
        "frameBuffer": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["below", "scene", "frame"]},
                "clear": {"type": "boolean"},
            },
        },
        "fractalNoise": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["fbm", "turbulent", "ridged"]},
                "octaves": {"type": "integer", "minimum": 1, "maximum": 10},
                "scale": _number(exclusiveMinimum=0),
                "scaleY": _number(exclusiveMinimum=0),
                "lacunarity": _number(exclusiveMinimum=0),
                "gain": _number(minimum=0, maximum=1),
                "evolution": _ref("animatedNumber"),
                "scrollX": _ref("animatedNumber"),
                "scrollY": _ref("animatedNumber"),
                "contrast": _ref("animatedNumber"),
                "brightness": _ref("animatedNumber"),
                "colorA": _ref("color"),
                "colorB": _ref("color"),
                "resolution": {"type": "integer", "minimum": 8, "maximum": 2048},
                "seed": {"type": "integer"},
            },
        },
        "speedLines": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 4, "maximum": 2000},
                "innerRadius": _ref("animatedNumber"),
                "outerRadius": _ref("animatedNumber"),
                "thickness": _ref("animatedNumber"),
                "density": _ref("animatedNumber"),
                "jitter": _number(minimum=0, maximum=1),
                "color": _ref("color"),
                "centerX": _number(),
                "centerY": _number(),
                "seed": {"type": "integer"},
                "speed": _number(),
            },
        },
        "karaoke": {
            "type": "object",
            "properties": {
                "progress": _ref("animatedNumber"),
                "color": _ref("color"),
                "mode": {"type": "string", "enum": ["fill", "wipe"]},
                "softness": _number(minimum=0, maximum=1),
            },
        },
        # ── テキスト（リッチテキスト / 日本語組版）
        #
        # `style` の «昔からある» 項目（size・color・align など）は、キーフレームや
        # 式で書けるようにあえて型を書いていません。ここで縛ると
        # {"size": {"keyframes": [...]}} のような正しい書き方が弾かれます。
        "textRun": {
            "type": "object",
            "properties": {
                "t": {"type": "string"},
                "text": {"type": "string"},
                "content": {"type": "string"},
                "color": _ref("color"),
                "size": _number(exclusiveMinimum=0),
                "sizeScale": _number(exclusiveMinimum=0),
                "family": {"type": "string"},
                "bold": {"type": "boolean"},
                "italic": {"type": "boolean"},
            },
        },
        # 枠に収める自動縮小。batch では «人が見て直す» 工程が無いので、
        # 曲や歌詞を差し替えたときのはみ出しをここで止めます。
        "textFit": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["shrink", "wrap", "off"]},
                # 数値は px、文字列は「80%」。% の基準は maxWidth か transform.width。
                "maxWidth": {"oneOf": [{"type": "number"}, {"type": "string"}]},
                "maxHeight": {"oneOf": [{"type": "number"}, {"type": "string"}]},
                "maxLines": {"type": "integer", "minimum": 1},
                "minSize": _number(exclusiveMinimum=0, maximum=1),
                "basis": _number(exclusiveMinimum=0),
                "enabled": {"type": "boolean"},
            },
        },
        "textRuby": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "syntax": {"type": "string", "enum": ["bracket"]},
                "size": _number(exclusiveMinimum=0, maximum=1),
                "offset": _number(),
            },
        },
        "textStyle": {
            "type": "object",
            "properties": {
                "runs": {"type": "array", "items": _ref("textRun")},
                "markup": {"type": "boolean"},
                "kinsoku": {
                    "oneOf": [
                        {"type": "string", "enum": ["off", "normal", "strict"]},
                        {"type": "boolean"},
                    ]
                },
                "fit": _ref("textFit"),
                "ruby": _ref("textRuby"),
            },
        },
        # レイヤーの `text` は «文字列» か «本文＋スタイル» のどちらでも書けます。
        "textContent": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "value": {"type": "string"},
                "runs": {"type": "array", "items": _ref("textRun")},
                "markup": {"type": "boolean"},
                "kinsoku": {
                    "oneOf": [
                        {"type": "string", "enum": ["off", "normal", "strict"]},
                        {"type": "boolean"},
                    ]
                },
                "fit": _ref("textFit"),
                "ruby": _ref("textRuby"),
            },
        },
        "counter": {
            "type": "object",
            "properties": {
                "from": _number(),
                "to": _number(),
                "progress": _ref("animatedNumber"),
                "decimals": {"type": "integer", "minimum": 0, "maximum": 8},
                "pad": {"type": "integer", "minimum": 0, "maximum": 16},
                "prefix": {"type": "string"},
                "suffix": {"type": "string"},
                "separator": {"type": "boolean"},
                "easing": _ref("easing"),
            },
        },
        "frameHold": {
            "type": "object",
            "properties": {
                "fps": _number(exclusiveMinimum=0, maximum=240),
                "enabled": {"type": "boolean"},
            },
        },
        # 速度ランプ・フリーズフレーム・逆再生。「出力の何秒で素材の何秒を見せるか」
        # の対応表なので、同じ値を 2 つ並べればフリーズ、値を下げれば逆再生です。
        "timeRemap": {
            "type": "object",
            "required": ["keyframes"],
            "properties": {
                "keyframes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["time", "value"],
                        "properties": {
                            "time": _number(minimum=0),
                            "value": _number(),
                            "easing": {"type": "string"},
                        },
                    },
                },
                # hold: コマ落ち（既定） / blend: 前後のフレームを混ぜる
                "mode": {"type": "string", "enum": ["hold", "blend"]},
                "extrapolate": {"type": "string", "enum": ["hold", "loop", "pingPong"]},
                "enabled": {"type": "boolean"},
            },
        },
        "echo": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 32},
                "delay": _number(exclusiveMinimum=0),
                "opacity": _number(minimum=0, maximum=1),
                "scale": _number(exclusiveMinimum=0),
                "rotation": _number(),
                "offsetX": _number(),
                "offsetY": _number(),
                "behind": {"type": "boolean"},
            },
        },
        "frameEcho": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "count": {"type": "integer", "minimum": 1, "maximum": 32},
                "delayFrames": {"type": "integer", "minimum": 1, "maximum": 60},
                "decay": _number(minimum=0, maximum=1),
                "blend": {"type": "string"},
                "tint": _ref("color"),
                "tintAmount": _number(minimum=0, maximum=1),
                "scale": _number(exclusiveMinimum=0),
            },
        },
        "trackMatte": {
            "type": "object",
            "required": ["layer"],
            "properties": {
                "layer": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": ["alpha", "luma"]},
                "invert": {"type": "boolean"},
            },
        },
        "motionBlur": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "samples": {"type": "integer", "minimum": 1, "maximum": 32},
                "shutter": _number(minimum=0, maximum=1),
            },
        },
        "repeater": {
            "type": "object",
            "required": ["count"],
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 200},
                "reverse": {"type": "boolean"},
                "offset": {
                    "type": "object",
                    "properties": {
                        "x": _number(),
                        "y": _number(),
                        "rotation": _number(),
                        "scale": _number(exclusiveMinimum=0),
                        "opacity": _number(minimum=0, maximum=1),
                    },
                },
            },
        },
        "textAnimator": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "enum": ["character", "word", "line"]},
                "stagger": _number(minimum=0),
                "duration": _number(exclusiveMinimum=0),
                "delay": _number(),
                "easing": _ref("easing"),
                "order": {"type": "string", "enum": ["forward", "reverse", "random", "center"]},
                "seed": {"type": "integer"},
                "loop": {"type": "boolean"},
                "loopDuration": _number(exclusiveMinimum=0),
                "from": {
                    "type": "object",
                    "properties": {
                        "opacity": _number(minimum=0, maximum=1),
                        "x": _number(),
                        "y": _number(),
                        "scale": _number(minimum=0),
                        "rotation": _number(),
                        "random": {
                            "type": "object",
                            "properties": {
                                "x": _number(),
                                "y": _number(),
                                "scale": _number(minimum=0),
                                "rotation": _number(),
                            },
                        },
                    },
                },
            },
        },
        "motionPath": {
            "type": "object",
            "required": ["points"],
            "properties": {
                "points": {"type": "array", "minItems": 2},
                "progress": _ref("animatedNumber"),
                "closed": {"type": "boolean"},
                "autoOrient": {"type": "boolean"},
                "orientOffset": _number(),
                "offsetX": _number(),
                "offsetY": _number(),
            },
        },
        "waveform": {
            "type": "object",
            "properties": {
                "style": {"type": "string", "enum": ["bars", "wave", "mirror"]},
                "bars": {"type": "integer", "minimum": 2, "maximum": 512},
                "color": _ref("color"),
                "endColor": _ref("color"),
                "gap": _number(minimum=0, maximum=0.9),
                "gain": _number(minimum=0),
                "radius": _number(minimum=0),
                "thickness": _number(minimum=0),
                "window": _number(exclusiveMinimum=0),
            },
        },
        "transition": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["fade", "wipe", "slide", "iris", "dissolve", "zoom", "flash", "jaws", "matte"],
                },
                "duration": _number(minimum=0),
                "in": _number(minimum=0),
                "out": _number(minimum=0),
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "vertical", "horizontal"],
                },
                "teeth": {"type": "integer", "minimum": 1, "maximum": 200},
                "depth": _number(minimum=0, maximum=1),
                "shape": {"type": "string", "enum": ["spikes", "waves", "blocks"]},
                "softness": _number(minimum=0, maximum=1),
                "centerX": _number(minimum=0, maximum=1),
                "centerY": _number(minimum=0, maximum=1),
                "scale": _number(exclusiveMinimum=0),
                "blur": _number(minimum=0),
                "color": _ref("color"),
                "easing": _ref("easing"),
            },
        },
        "scene": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "start": _number(minimum=0),
                "duration": _number(minimum=0),
                "end": _number(minimum=0),
                "background": _ref("color"),
                "enabled": {"type": "boolean"},
                "transition": _ref("transition"),
                "layers": {"type": "array", "items": _ref("layer")},
                "physicsWorld": _ref("physicsWorld"),
                "use": {
                    "oneOf": [_ref("skillUse"), {"type": "array", "items": _ref("skillUse")}]
                },
            },
        },
        "composition": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "duration": _number(minimum=0),
                "background": _ref("color"),
                "layers": {"type": "array", "items": _ref("layer")},
                "scenes": {"type": "array", "items": _ref("scene")},
            },
        },
        "render": {
            "type": "object",
            "properties": {
                "quality": {"type": "string", "enum": QUALITY_PRESETS},
                "renderer": {"type": "string", "enum": RENDERER_TYPES},
                "superSample": {"type": "integer", "minimum": 1, "maximum": 4},
                "threads": {"type": "integer", "minimum": 1, "maximum": 64},
                "frameHistory": {"type": "integer", "minimum": 1, "maximum": 240},
                "deformation": {
                    "type": "object",
                    "properties": {
                        "meshResolution": {"type": "integer", "minimum": 2, "maximum": 256}
                    },
                },
                "physics": {
                    "type": "object",
                    "properties": {
                        "subSteps": {"type": "integer", "minimum": 1, "maximum": 32},
                        "iterations": {"type": "integer", "minimum": 1, "maximum": 64},
                    },
                },
                "effects": {
                    "type": "object",
                    "properties": {"samples": {"type": "integer", "minimum": 1, "maximum": 64}},
                },
            },
        },
        "output": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["mp4", "webm", "mov", "gif", "png-sequence", "wav"],
                },
                "codec": {"type": "string"},
                "path": {"type": "string"},
                "crf": {"type": "integer", "minimum": 0, "maximum": 63},
                "bitrate": {"type": "string"},
                "pixelFormat": {"type": "string"},
                "audioCodec": {"type": "string"},
                "audioBitrate": {"type": "string"},
                "preset": {"type": "string"},
                "loop": {"type": "integer", "minimum": 0},
                "colors": {"type": "integer", "minimum": 2, "maximum": 256},
                # ラウドネス正規化。書いたときだけ動く（既定は従来のピーク正規化）。
                "loudness": _ref("loudness"),
            },
        },
        # EBU R128 / ITU-R BS.1770 のラウドネス正規化。
        "loudness": {
            "oneOf": [
                # false と書けば «やらない»。true は既定値（−14 LUFS / −1 dBTP）。
                {"type": "boolean"},
                {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        # 目標ラウドネス（LUFS）。配信先に合わせるなら −14 あたり。
                        "target": _number(minimum=-70, maximum=0),
                        # トゥルーピークの天井（dBTP）。0 を超えると変換時に歪む。
                        "truePeak": _number(minimum=-20, maximum=0),
                        # 合わせたつもりの基準。測定はどれも BS.1770 で同じ。
                        "standard": {
                            "type": "string",
                            "enum": ["ebu-r128", "itu-bs1770", "atsc-a85", "streaming"],
                        },
                    },
                },
            ]
        },
    },
}

__all__ = [
    "DEFORMER_TYPES",
    "EFFECT_TYPES",
    "LAYER_TYPES",
    "MASK_TYPES",
    "MODULATOR_TYPES",
    "PHYSICS_CONTROL_MODES",
    "PLUGIN_KINDS",
    "QUALITY_PRESETS",
    "RENDERER_TYPES",
    "SHAPE_TYPES",
    "project_schema",
]
