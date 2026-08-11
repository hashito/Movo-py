"""movo.schema — プロジェクト JSON の検証と正規化。

検証は 2 段です。

  1. 構造検証: `project_schema.py` の JSON Schema に当てる
  2. 意味検証: スキーマでは書けないもの（レイヤー id の重複・宣言していない
     素材の参照・壊れた式・知らない変形の名前・破綻する物理の値 …）

正規化の順番は **結果を変えます**。`normalize_project` の中のコメントを読んでから
並べ替えてください。
"""

from __future__ import annotations

import copy
import math

from movo.expression import ExpressionEngine
from movo.expression._compat import (
    MOVO_JSON_VERSION,
    MovoValidationError,
    is_compatible_json_version,
    is_finite_number,
    js_number,
    js_round,
)

from .extends import apply_extends, merge_deep, strip_json_comments
from .musical_time import (
    is_musical_time,
    musical_units,
    resolve_musical_time,
    to_seconds,
)
from .params import (
    RECIPE_SCHEMA,
    build_recipe,
    check_recipe_assets,
    expand_params,
    list_params,
    param_overrides_from,
    parse_param_assignments,
    prepare_project,
    read_recipe,
    resolve_params,
    write_recipe,
)
from .presets import describe_presets, resolve_presets
from .project_schema import (
    DEFORMER_TYPES,
    EFFECT_TYPES,
    LAYER_TYPES,
    MASK_TYPES,
    MODULATOR_TYPES,
    PHYSICS_CONTROL_MODES,
    PLUGIN_KINDS,
    QUALITY_PRESETS,
    RENDERER_TYPES,
    SHAPE_TYPES,
    project_schema,
)
from .relative_units import (
    axis_of_key,
    find_unresolved_relative_units,
    is_relative_unit,
    relative_to_pixels,
    resolve_relative_units,
)
from .structure import find_section, resolve_structure, structure_of
from .validator import SchemaValidator, join_path
from .variants import apply_variant, expand_all_variants, list_variants, variant_names

_validator = SchemaValidator(project_schema)

QUALITY_SETTINGS = {
    "draft": {
        "superSample": 1,
        "meshResolution": 8,
        "physicsSubSteps": 1,
        "physicsIterations": 4,
        "effectSamples": 1,
        "alphaOutline": False,
    },
    "preview": {
        "superSample": 1,
        "meshResolution": 12,
        "physicsSubSteps": 2,
        "physicsIterations": 6,
        "effectSamples": 2,
        "alphaOutline": False,
    },
    "standard": {
        "superSample": 1,
        "meshResolution": 20,
        "physicsSubSteps": 2,
        "physicsIterations": 8,
        "effectSamples": 4,
        "alphaOutline": True,
    },
    "high": {
        "superSample": 2,
        "meshResolution": 32,
        "physicsSubSteps": 4,
        "physicsIterations": 12,
        "effectSamples": 8,
        "alphaOutline": True,
    },
    "ultra": {
        "superSample": 3,
        "meshResolution": 48,
        "physicsSubSteps": 6,
        "physicsIterations": 16,
        "effectSamples": 16,
        "alphaOutline": True,
    },
}


def validate_structure(project):
    """構造検証だけ。`{"valid": bool, "issues": [...]}` を返す。"""
    return _validator.validate(project)


def validate_project(
    project,
    file=None,
    base_dir=None,
    set_values=None,
    params=None,
    known_deformers=None,
    known_effects=None,
    known_modulators=None,
):
    """構造検証と意味検証を通す。"""
    # 拍・小節の書き方（"4bar"）は «秒の糖衣» なので、構造検証にかける前に
    # 秒へ直します。スキーマ側で «数値または拍の文字列» を 100 か所以上に
    # 書き足すより、ここで 1 回ほどく方が取りこぼしがありません。
    # 呼び出し元の JSON は変えないので、複製の上で行います。
    subject = project
    issues: list[dict] = []
    if isinstance(project, dict):
        try:
            # 継承と params は «秒に直す» より前。土台や params 側にも拍で書けるように。
            # 相対単位（"50%" / "7vh"）も同じ段で px に直します。«セーフエリアからの
            # はみ出し» を見るには、比較できる数値になっている必要があるためです。
            subject = resolve_relative_units(
                resolve_musical_time(
                    resolve_structure(
                        resolve_params(
                            apply_extends(copy.deepcopy(project), file=file, base_dir=base_dir),
                            file=file,
                            set_values=set_values,
                            params=params,
                        )
                    )
                )
            )
        except Exception as error:
            issues.append(
                {
                    "path": getattr(error, "path", None) or "project.bpm",
                    "message": getattr(error, "reason", None) or str(error),
                }
            )
            subject = project

    structural = validate_structure(subject)
    issues.extend(structural["issues"])
    warnings: list[dict] = []

    if isinstance(project, dict):
        if project.get("movoVersion") and not is_compatible_json_version(project["movoVersion"]):
            warnings.append(
                {
                    "path": "movoVersion",
                    "message": (
                        f"project targets Movo JSON {project['movoVersion']} "
                        f"but this build implements {MOVO_JSON_VERSION}"
                    ),
                }
            )
        # 意味検証も «秒に直したあと» を見る。キーフレームが表示区間に収まって
        # いるかの判定などは、拍のままでは比較できないため。
        _semantic_checks(
            subject,
            issues,
            warnings,
            known_deformers=known_deformers,
            known_effects=known_effects,
            known_modulators=known_modulators,
        )

    return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}


def assert_valid_project(project, file=None, **options):
    """検証を通らなければ MovoValidationError を投げる。"""
    result = validate_project(project, file=file, **options)
    if not result["valid"]:
        raise MovoValidationError(result["issues"], file)
    return result


# ---------------------------------------------------------------------------
# 意味検証
# ---------------------------------------------------------------------------

#: ai-character が作る派生素材の既定のパーツ名。
_DEFAULT_AI_PARTS = [
    "head",
    "body",
    "upperArmLeft",
    "lowerArmLeft",
    "handLeft",
    "upperArmRight",
    "lowerArmRight",
    "handRight",
    "legLeft",
    "legRight",
    "hair",
    "mouth",
    "eyeLeft",
    "eyeRight",
]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_list(value):
    return value if isinstance(value, list) else []


def _semantic_checks(project, issues, warnings, known_deformers, known_effects, known_modulators):
    seed = js_number((project.get("project") or {}).get("seed"))
    engine = ExpressionEngine(seed=int(seed) if is_finite_number(seed) else 0)
    assets = project.get("assets") or {}
    asset_names = set(assets.keys())
    # ai-character は "<素材名>.<パーツ名>" という派生素材を生成するので、
    # それも宣言済みとして扱う（既定パーツ名も許可する）。
    for name, declaration in assets.items():
        if not isinstance(declaration, dict) or declaration.get("type") != "ai-character":
            continue
        for part in declaration.get("parts") or _DEFAULT_AI_PARTS:
            asset_names.add(f"{name}.{part}")
    composition_names = set((project.get("compositions") or {}).keys())
    preset_names = set((project.get("presets") or {}).keys())
    character_names = set((project.get("characters") or {}).keys())
    layer_ids: dict[str, str] = {}

    def check_expression(source, path):
        if not isinstance(source, str):
            return
        result = engine.check(source)
        if not result["ok"]:
            issues.append({"path": path, "message": result["message"]})

    def check_modulator(modulator, path):
        if not isinstance(modulator, dict) or not isinstance(modulator.get("type"), str):
            return
        if known_modulators is not None and modulator["type"] not in known_modulators:
            warnings.append(
                {
                    "path": join_path(path, "type"),
                    "message": (
                        f'unknown modulator type "{modulator["type"]}" '
                        "— it evaluates to 0 unless a plugin provides it"
                    ),
                }
            )

    def walk_animated(value, path):
        if not isinstance(value, (dict, list)):
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk_animated(item, f"{path}[{index}]")
            return
        if isinstance(value.get("expression"), str):
            check_expression(value["expression"], join_path(path, "expression"))
        if value.get("modulator"):
            check_modulator(value["modulator"], join_path(path, "modulator"))
        if isinstance(value.get("modulators"), list):
            for index, m in enumerate(value["modulators"]):
                check_modulator(m, f"{join_path(path, 'modulators')}[{index}]")
        for key, child in value.items():
            if key == "expression":
                continue
            walk_animated(child, join_path(path, key))

    def check_modifier(modifier, path):
        if not isinstance(modifier, dict):
            return
        if (
            known_deformers is not None
            and modifier.get("type")
            and modifier["type"] not in known_deformers
            and not (known_effects is not None and modifier["type"] in known_effects)
        ):
            warnings.append(
                {
                    "path": join_path(path, "type"),
                    "message": f'unknown modifier type "{modifier["type"]}" — it will be skipped',
                }
            )
        if modifier.get("mapAsset") and modifier["mapAsset"] not in asset_names:
            issues.append(
                {
                    "path": join_path(path, "mapAsset"),
                    "message": f'asset "{modifier["mapAsset"]}" is not declared in assets',
                }
            )
        mask = modifier.get("mask")
        if isinstance(mask, dict) and mask.get("asset") and mask["asset"] not in asset_names:
            issues.append(
                {
                    "path": join_path(path, "mask.asset"),
                    "message": f'asset "{mask["asset"]}" is not declared in assets',
                }
            )
        walk_animated(modifier, path)

    def check_keyframe_range(value, path, visible_end, label):
        """キーフレームが «表示区間» に収まっているかを見る。

        最後のキーフレームがレイヤーの消える時刻より後にあると、その動きは
        途中で止まったまま終わります（カラオケ塗りが最後まで塗られない、
        ゲージが伸び切らない、など）。書いた本人には気付きにくいので警告します。
        """
        if visible_end is None or not isinstance(value, dict):
            return
        keyframes = value.get("keyframes")
        if not isinstance(keyframes, list) or not keyframes:
            return
        last = -math.inf
        for keyframe in keyframes:
            time = keyframe.get("time") if isinstance(keyframe, dict) else None
            if _is_number(time) and time > last:
                last = time
        # 0.05 秒はフレーム 1〜2 枚ぶんの誤差。ここまでは許す。
        if last > visible_end + 0.05:
            warnings.append(
                {
                    "path": path,
                    "message": (
                        f"{label}の最後のキーフレームが {_plain(last)} 秒にありますが、"
                        f"このレイヤーは {visible_end:.2f} 秒で消えます"
                        "（動きが途中で止まったまま終わります）"
                    ),
                }
            )

    def check_flash_rate(layer, path):
        """書き方の時点で «明らかに点滅が速すぎる» ものに注意する。

        実際のフレームを測る検査は書き出し時に走りますが、20 分待ってから
        «危険でした» と言われるのは遅いので、分かるものだけ先に伝えます。
        光過敏性発作のガイドライン（ITU-R BT.1702 / WCAG 2.3.1）は
        毎秒 3 回を超える閃光を危険としています。
        """
        bpm = js_number((project.get("project") or {}).get("bpm"))
        uses = _as_list(layer.get("use"))

        for index, entry in enumerate(uses):
            if not isinstance(entry, dict) or entry.get("animation") != "beat-flash":
                continue
            if not is_finite_number(bpm) or bpm <= 0:
                continue
            # beat-flash の division は «何拍ごとか»。大きいほど間隔が空く。
            division = js_number((entry.get("with") or {}).get("division", 1))
            if not is_finite_number(division) or division == 0:
                division = 1
            per_second = bpm / 60 / division
            if per_second > 3:
                warnings.append(
                    {
                        "path": f"{join_path(path, 'use')}[{index}]",
                        "message": (
                            f"拍ごとの点滅が毎秒 {per_second:.1f} 回になります"
                            "（光過敏性発作のガイドラインは毎秒 3 回まで）。"
                            "間隔を空けるか、点滅する面積を画面の 1/4 未満にしてください"
                        ),
                    }
                )

        # glitch-flicker の interval は «何フレームごとに変わるか»。
        # 1 なら毎フレーム変わるので、24fps でも毎秒 12 往復になる。
        flicker_interval = None
        for entry in uses:
            if isinstance(entry, dict) and entry.get("animation") == "glitch-flicker":
                flicker_interval = (entry.get("with") or {}).get("interval")
                break
        if flicker_interval is not None and js_number(flicker_interval) == 1:
            warnings.append(
                {
                    "path": join_path(path, "use"),
                    "message": (
                        "glitch-flicker の interval が 1（毎フレーム）です。"
                        "画面全体に掛けると光過敏性発作の危険があります。2 以上にしてください"
                    ),
                }
            )

    def check_layer(layer, path, scene_duration=None):
        if not isinstance(layer, dict):
            return

        # タイムリマップは «そのレイヤーの時間» を曲げるだけなので、物理演算には
        # 効きません。物理は固定タイムステップで世界ごと進めるためです。
        time_remap = layer.get("timeRemap")
        if time_remap and time_remap.get("enabled") is not False and layer.get("physics"):
            warnings.append(
                {
                    "path": join_path(path, "timeRemap"),
                    "message": (
                        "timeRemap は physics に効きません"
                        "（物理は固定タイムステップで世界ごと進むため）。"
                        "速度を変えたいなら physics を外してください"
                    ),
                }
            )

        # レイヤーが見えている区間の «終わり»（レイヤー相対の秒数）
        start = layer["start"] if _is_number(layer.get("start")) else 0
        visible_end = None
        if _is_number(layer.get("end")):
            visible_end = layer["end"]
        elif _is_number(layer.get("duration")):
            visible_end = start + layer["duration"]
        elif _is_number(scene_duration):
            visible_end = scene_duration

        if layer.get("id"):
            if layer["id"] in layer_ids:
                issues.append(
                    {
                        "path": join_path(path, "id"),
                        "message": (
                            f'duplicate layer id "{layer["id"]}" '
                            f"(also at {layer_ids[layer['id']]})"
                        ),
                    }
                )
            else:
                layer_ids[layer["id"]] = path
        if layer.get("asset") and layer["asset"] not in asset_names:
            issues.append(
                {
                    "path": join_path(path, "asset"),
                    "message": f'asset "{layer["asset"]}" is not declared in assets',
                }
            )
        preset = layer.get("preset")
        preset_list = preset if isinstance(preset, list) else ([preset] if preset else [])
        for name in preset_list:
            if name not in preset_names:
                message = (
                    f'preset "{name}" is not declared in presets '
                    f"(declared: {', '.join(preset_names)})"
                    if preset_names
                    else f'preset "{name}" is not declared — add it to project.presets'
                )
                issues.append({"path": join_path(path, "preset"), "message": message})
        if (
            layer.get("type") == "composition"
            and layer.get("composition")
            and layer["composition"] not in composition_names
        ):
            issues.append(
                {
                    "path": join_path(path, "composition"),
                    "message": f'composition "{layer["composition"]}" is not declared',
                }
            )
        if (
            layer.get("type") == "character"
            and layer.get("character")
            and layer["character"] not in character_names
            and not layer.get("rig")
        ):
            issues.append(
                {
                    "path": join_path(path, "character"),
                    "message": f'character "{layer["character"]}" is not declared in characters',
                }
            )
        if _is_number(layer.get("start")) and _is_number(layer.get("end")) and layer["end"] < layer["start"]:
            issues.append(
                {"path": join_path(path, "end"), "message": "end must not be earlier than start"}
            )
        check_flash_rate(layer, path)
        for index, animation in enumerate(_as_list(layer.get("animations"))):
            animation_path = f"{join_path(path, 'animations')}[{index}]"
            if isinstance(animation, dict) and animation.get("expression"):
                check_expression(animation["expression"], join_path(animation_path, "expression"))
            if (
                isinstance(animation, dict)
                and not animation.get("keyframes")
                and not animation.get("expression")
                and not animation.get("modulator")
                and not animation.get("modulators")
                and "value" not in animation
            ):
                issues.append(
                    {
                        "path": animation_path,
                        "message": "animation needs one of value, keyframes, expression, modulator or modulators",
                    }
                )
            walk_animated(animation, animation_path)
            check_keyframe_range(animation, animation_path, visible_end, "アニメーション")
        # 塗り・カウンター・枠の伸びなど «進み具合» を持つものも同じように見る
        for key, label in (("karaoke", "カラオケ塗り"), ("counter", "カウンター")):
            node = layer.get(key)
            if isinstance(node, dict) and node.get("progress"):
                check_keyframe_range(
                    node["progress"], join_path(path, f"{key}.progress"), visible_end, label
                )
        text_box = layer.get("textBox")
        if isinstance(text_box, dict):
            reveal = text_box.get("reveal")
            if isinstance(reveal, dict) and reveal.get("progress"):
                check_keyframe_range(
                    reveal["progress"], join_path(path, "textBox.reveal.progress"), visible_end, "枠の出現"
                )
        for key in ("linePath", "neonPath"):
            node = layer.get(key)
            if isinstance(node, dict) and node.get("end"):
                check_keyframe_range(
                    node["end"], join_path(path, f"{key}.end"), visible_end, "線の伸び"
                )
        for index, modifier in enumerate(_as_list(layer.get("modifiers"))):
            check_modifier(modifier, f"{join_path(path, 'modifiers')}[{index}]")
        for index, effect in enumerate(_as_list(layer.get("effects"))):
            check_modifier(effect, f"{join_path(path, 'effects')}[{index}]")
        if layer.get("transform"):
            walk_animated(layer["transform"], join_path(path, "transform"))
        physics = layer.get("physics")
        if (
            isinstance(physics, dict)
            and isinstance(physics.get("shape"), dict)
            and physics["shape"].get("type") == "alpha-outline"
            and not layer.get("asset")
            and not physics["shape"].get("asset")
        ):
            warnings.append(
                {
                    "path": join_path(path, "physics.shape"),
                    "message": "alpha-outline needs an image asset; falling back to a rectangle",
                }
            )
        for index, child in enumerate(_as_list(layer.get("layers"))):
            check_layer(child, f"{join_path(path, 'layers')}[{index}]", scene_duration)

    for scene_index, scene in enumerate(_as_list(project.get("scenes"))):
        scene_path = f"scenes[{scene_index}]"
        if _is_number(scene.get("duration")):
            scene_duration = scene["duration"]
        else:
            video_duration = (project.get("video") or {}).get("duration")
            scene_duration = video_duration if video_duration is not None else None
        for index, layer in enumerate(_as_list(scene.get("layers"))):
            check_layer(layer, f"{scene_path}.layers[{index}]", scene_duration)
    for index, layer in enumerate(_as_list(project.get("layers"))):
        check_layer(layer, f"layers[{index}]")

    for name, composition in (project.get("compositions") or {}).items():
        for index, layer in enumerate(_as_list(composition.get("layers"))):
            check_layer(layer, f"compositions.{name}.layers[{index}]")
        for scene_index, scene in enumerate(_as_list(composition.get("scenes"))):
            for index, layer in enumerate(_as_list(scene.get("layers"))):
                check_layer(
                    layer, f"compositions.{name}.scenes[{scene_index}].layers[{index}]"
                )

    for name, rig in (project.get("characters") or {}).items():
        ids = set()
        for index, part in enumerate(_as_list(rig.get("parts"))):
            part_path = f"characters.{name}.parts[{index}]"
            if part.get("id") in ids:
                issues.append(
                    {
                        "path": join_path(part_path, "id"),
                        "message": f'duplicate part id "{part.get("id")}"',
                    }
                )
            ids.add(part.get("id"))
            if part.get("asset") and part["asset"] not in asset_names:
                issues.append(
                    {
                        "path": join_path(part_path, "asset"),
                        "message": f'asset "{part["asset"]}" is not declared in assets',
                    }
                )
        for index, part in enumerate(_as_list(rig.get("parts"))):
            if part.get("parent") and part["parent"] not in ids:
                issues.append(
                    {
                        "path": f"characters.{name}.parts[{index}].parent",
                        "message": f'unknown parent part "{part["parent"]}"',
                    }
                )
        for index, ik in enumerate(_as_list(rig.get("ik"))):
            for chain_index, part_id in enumerate(_as_list(ik.get("chain"))):
                if part_id not in ids:
                    issues.append(
                        {
                            "path": f"characters.{name}.ik[{index}].chain[{chain_index}]",
                            "message": f'unknown part "{part_id}"',
                        }
                    )

    for index, track in enumerate(_as_list(project.get("audio"))):
        if track.get("asset") and track["asset"] not in asset_names:
            issues.append(
                {
                    "path": f"audio[{index}].asset",
                    "message": f'asset "{track["asset"]}" is not declared in assets',
                }
            )
        if not track.get("asset") and not track.get("path"):
            issues.append(
                {"path": f"audio[{index}]", "message": "audio track needs an asset or a path"}
            )

    _check_relative_units(project, warnings)
    _check_safe_area(project, warnings)

    world = project.get("physicsWorld")
    if isinstance(world, dict) and world.get("timeStep") is not None and world["timeStep"] > 1 / 20:
        warnings.append(
            {
                "path": "physicsWorld.timeStep",
                "message": "large time steps make the simulation unstable; 1/60 is recommended",
            }
        )

    render = project.get("render")
    if isinstance(render, dict) and render.get("quality") and render["quality"] not in QUALITY_PRESETS:
        issues.append(
            {"path": "render.quality", "message": f"must be one of: {', '.join(QUALITY_PRESETS)}"}
        )


def _plain(value):
    """メッセージに出す数（`3.0` ではなく `3`）。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _check_relative_units(project, warnings):
    """直せなかった相対単位を伝える。

    `"50%"` は今までどおり NaN → 既定値に落ちるだけなので、動画は出ます。
    出るのに «思ったところに無い» のがいちばん困るので、書いた場所を言います。
    """
    for found in find_unresolved_relative_units(project):
        warnings.append(
            {
                "path": found["path"],
                "message": (
                    f'"{found["value"]}" を解決できませんでした（{found["reason"]}）。'
                    "px で書くか、vw / vh を使ってください"
                ),
            }
        )


def _check_safe_area(project, warnings):
    """`video.safeArea` からはみ出しているレイヤーを警告する。

        "video": { "width": 1920, "height": 1080, "safeArea": { "x": 0.05, "y": 0.05 } }

    端に寄せた文字は、9:16 に組み替えたときにいちばん先に切れます。**エラーでは
    なく警告** なのは、画面いっぱいの背景のようにはみ出して当然のものがあるからです。

    判定は «最初のフレームの、書いてある値だけ» で行います。アニメーションで
    外へ出るもの、テキストの実寸、親の変換までは追いません。追おうとすると
    レンダリングと同じ計算が要り、検証が «速く済む» という取り柄を失います。
    """
    video = project.get("video") or {}
    safe_area = video.get("safeArea")
    if not isinstance(safe_area, dict):
        return
    width = _finite(video.get("width"))
    height = _finite(video.get("height"))
    if width is None or height is None:
        return

    margin_x = _ratio_of(safe_area.get("x")) * width
    margin_y = _ratio_of(safe_area.get("y")) * height
    safe = {
        "left": margin_x,
        "right": width - margin_x,
        "top": margin_y,
        "bottom": height - margin_y,
    }
    # 1px 未満のはみ出しは丸め誤差なので見逃す
    slack = 1

    def check(layer, path):
        if not isinstance(layer, dict) or layer.get("enabled") is False:
            return
        # `"safeArea": false` は «これは意図的に外に出している» という宣言。
        if layer.get("safeArea") is False:
            return
        box = _layer_box(layer)
        if not box:
            return
        # 画面を覆い切っているものは背景。はみ出して «当然» なので黙ります。
        # ここを警告すると、どの作例でも背景 1 枚が必ず鳴り、警告を読まない癖がつきます。
        if box["left"] <= 0 and box["top"] <= 0 and box["right"] >= width and box["bottom"] >= height:
            return
        outside = []
        if box["left"] < safe["left"] - slack:
            outside.append(f"左が {js_round(safe['left'] - box['left'])}px")
        if box["right"] > safe["right"] + slack:
            outside.append(f"右が {js_round(box['right'] - safe['right'])}px")
        if box["top"] < safe["top"] - slack:
            outside.append(f"上が {js_round(safe['top'] - box['top'])}px")
        if box["bottom"] > safe["bottom"] + slack:
            outside.append(f"下が {js_round(box['bottom'] - safe['bottom'])}px")
        if not outside:
            return
        warnings.append(
            {
                "path": join_path(path, "transform"),
                "message": (
                    f'レイヤー "{layer.get("id", "?")}" がセーフエリアからはみ出しています'
                    f"（{' / '.join(outside)}）。"
                    "別のアスペクト比に組み替えたときに切れます"
                ),
            }
        )

    # シーン直下のレイヤーだけを見る。入れ子のレイヤーの座標は «親からの相対» で、
    # 画面の座標と直接くらべられないためです。
    for scene_index, scene in enumerate(_as_list(project.get("scenes"))):
        for index, layer in enumerate(_as_list(scene.get("layers"))):
            check(layer, f"scenes[{scene_index}].layers[{index}]")
    for index, layer in enumerate(_as_list(project.get("layers"))):
        check(layer, f"layers[{index}]")


def _ratio_of(value) -> float:
    """`0.05` は «画面の 5%»。半分を超える余白は成り立たないので 0.5 で止めます。"""
    n = js_number(value)
    if not is_finite_number(n) or n < 0:
        return 0
    return min(0.5, n)


def _layer_box(layer):
    """書いてある値からレイヤーの箱を組み立てる。

    大きさが分からないもの（原寸の画像・テキスト）は «点» として見ます。
    点が内側にあるかぎり警告しません。当てずっぽうで大きさを仮定すると、
    出ないはずの警告が出て «警告を読まない» 癖がつくからです。
    """
    transform = layer.get("transform") or {}
    x = _finite(transform.get("x"))
    y = _finite(transform.get("y"))
    if x is None and y is None:
        return None
    box_width = _finite(transform.get("width"))
    if box_width is None:
        box_width = _finite((layer.get("shape") or {}).get("width"))
    if box_width is None:
        box_width = 0
    box_height = _finite(transform.get("height"))
    if box_height is None:
        box_height = _finite((layer.get("shape") or {}).get("height"))
    if box_height is None:
        box_height = 0
    # レンダラと同じ既定値。group / composition だけ左上原点です。
    default_anchor = 0 if layer.get("type") in ("group", "composition") else 0.5
    anchor_x = _finite(transform.get("anchorX"))
    anchor_x = default_anchor if anchor_x is None else anchor_x
    anchor_y = _finite(transform.get("anchorY"))
    anchor_y = default_anchor if anchor_y is None else anchor_y
    px = x if x is not None else 0
    py = y if y is not None else 0
    return {
        "left": px - box_width * anchor_x,
        "right": px + box_width * (1 - anchor_x),
        "top": py - box_height * anchor_y,
        "bottom": py + box_height * (1 - anchor_y),
    }


def _finite(value):
    """数として読めれば返し、読めなければ None。

    **書かれていないキー（None）は None のままにします。** JS の `Number(null)` は
    0 ですが、ここへ来る None は `.get()` が返した «キーが無い»（JS の undefined）
    です。0 として扱うと、大きさを書いていないレイヤーの箱が «幅 0» ではなく
    «幅 0 の点» にならず、セーフエリアの判定が狂います。
    """
    if value is None:
        return None
    n = js_number(value)
    return n if is_finite_number(n) else None


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------


def _expand_ken_burns(project):
    """`kenBurns` を transform のキーフレームへ展開する。

    静止画をゆっくり寄せながら流す定番演出を 1 行で書けるようにするための糖衣です。
    `duration` を省略するとレイヤーの表示区間全体を使います。
    """
    video_duration = (project.get("video") or {}).get("duration")
    if video_duration is None:
        video_duration = 10

    def apply_to(layers, scene_duration):
        for layer in layers or []:
            if not isinstance(layer, dict):
                continue
            if isinstance(layer.get("layers"), list):
                apply_to(layer["layers"], scene_duration)
            spec = layer.get("kenBurns")
            if not spec:
                continue
            del layer["kenBurns"]

            source = spec.get("from") or {}
            target = spec.get("to") or {}
            start = layer.get("start")
            start = 0 if start is None else start
            if spec.get("duration") is not None:
                length = spec["duration"]
            elif layer.get("duration") is not None:
                length = layer["duration"]
            else:
                end = layer.get("end")
                length = (end if end is not None else scene_duration) - start
            end_time = start + max(0.01, length)
            easing = spec.get("easing") or "easeInOutSine"

            pairs = [
                ("transform.scaleX", source.get("scale", 1), target.get("scale", 1.1), False),
                ("transform.scaleY", source.get("scale", 1), target.get("scale", 1.1), False),
                ("transform.x", source.get("x", 0), target.get("x", 0), True),
                ("transform.y", source.get("y", 0), target.get("y", 0), True),
            ]
            animations = []
            for prop, a, b, relative in pairs:
                # 動きが無い軸はキーフレームを作らない（無駄な評価を避ける）
                if a == b:
                    continue
                animation = {"property": prop}
                if relative:
                    animation["relative"] = True
                animation["keyframes"] = [
                    {"time": start, "value": a},
                    {"time": end_time, "value": b, "easing": easing},
                ]
                animations.append(animation)
            if not animations:
                continue
            # レイヤー自身の指定を後に置く（そちらが優先される）
            layer["animations"] = [*animations, *(layer.get("animations") or [])]

    for scene in project.get("scenes") or []:
        if isinstance(scene, dict):
            duration = scene.get("duration")
            apply_to(scene.get("layers"), duration if duration is not None else video_duration)
    for composition in (project.get("compositions") or {}).values():
        if not isinstance(composition, dict):
            continue
        duration = composition.get("duration")
        apply_to(composition.get("layers"), duration if duration is not None else video_duration)
        for scene in composition.get("scenes") or []:
            if isinstance(scene, dict):
                scene_duration = scene.get("duration")
                apply_to(
                    scene.get("layers"),
                    scene_duration if scene_duration is not None else video_duration,
                )


def normalize_project(project, file=None, base_dir=None, set_values=None, params=None):
    """既定値を埋めて «後段が完全な記録として扱える» 形にする。

    **この関数の中の順番は結果を変えます。** JS 版（packages/schema/src/index.js）と
    同じ並びです。並べ替える前に、各段のコメントの «なぜその位置か» を読んでください。
    """
    # 継承 → params 展開 → 曲の区間 → 拍の変換 → プリセット → 相対単位 → kenBurns。
    # どれも書いていない JSON では何もしない（従来どおり）。
    cloned = resolve_params(
        apply_extends(copy.deepcopy(project or {}), file=file, base_dir=base_dir),
        file=file,
        set_values=set_values,
        params=params,
    )
    video = cloned.get("video") or {}
    normalized_video = {
        "width": video.get("width", 1920),
        "height": video.get("height", 1080),
        "fps": video.get("fps", 30),
    }
    # `duration` は «書かれていなければ無い» ままにします。None を入れると
    # 構造検証が «number ではない» と言い出します。
    if video.get("duration") is not None:
        normalized_video["duration"] = video["duration"]
    normalized_video["background"] = video.get("background", "#000000")
    normalized_video["pixelAspect"] = video.get("pixelAspect", 1)
    normalized_video["colorSpace"] = video.get("colorSpace", "srgb")
    # セーフエリアは検証でしか使いませんが、正規化で落とすと
    # «normalize_project を通したものを検証する» 経路で消えてしまいます。
    if video.get("safeArea"):
        normalized_video["safeArea"] = video["safeArea"]
    cloned["video"] = normalized_video

    cloned["project"] = {"name": "untitled", "seed": 12345, **(cloned.get("project") or {})}
    cloned["presets"] = cloned.get("presets") or {}
    cloned["assets"] = cloned.get("assets") or {}
    cloned["variables"] = cloned.get("variables") or {}
    cloned["fonts"] = cloned.get("fonts") or {}
    cloned["plugins"] = cloned.get("plugins") or []
    cloned["compositions"] = cloned.get("compositions") or {}
    cloned["characters"] = cloned.get("characters") or {}
    cloned["audio"] = cloned.get("audio") or []
    # output は配列でも書けます。正規化では «形» を保ったまま既定値だけ入れます。
    if isinstance(cloned.get("output"), list):
        cloned["output"] = [
            {"format": "mp4", "codec": "h264", **(entry or {})} for entry in cloned["output"]
        ]
    else:
        cloned["output"] = {"format": "mp4", "codec": "h264", **(cloned.get("output") or {})}
    cloned["security"] = {
        "allowNetwork": True,
        "maxDownloadSizeMB": 100,
        **(cloned.get("security") or {}),
    }
    cloned["deterministic"] = {
        "enabled": True,
        "fixedTimeStep": True,
        **(cloned.get("deterministic") or {}),
    }
    if cloned["deterministic"].get("seed") is None:
        cloned["deterministic"]["seed"] = cloned["project"]["seed"]

    requested_render = cloned.get("render") or {}
    quality = requested_render.get("quality") or "standard"
    preset = QUALITY_SETTINGS.get(quality) or QUALITY_SETTINGS["standard"]
    render = {
        # 既知でないキー（motionBlur、frameHistory など）もそのまま残す
        **requested_render,
        "quality": quality,
        "renderer": requested_render.get("renderer") or "canvas-2d",
        "superSample": (
            requested_render["superSample"]
            if requested_render.get("superSample") is not None
            else preset["superSample"]
        ),
        "deformation": {
            "meshResolution": preset["meshResolution"],
            **(requested_render.get("deformation") or {}),
        },
        "physics": {
            "subSteps": preset["physicsSubSteps"],
            "iterations": preset["physicsIterations"],
            **(requested_render.get("physics") or {}),
        },
        "effects": {"samples": preset["effectSamples"], **(requested_render.get("effects") or {})},
        "alphaOutline": (
            requested_render["alphaOutline"]
            if requested_render.get("alphaOutline") is not None
            else preset["alphaOutline"]
        ),
    }
    if requested_render.get("threads") is not None:
        render["threads"] = requested_render["threads"]
    cloned["render"] = render

    # トップレベルの `layers` は «動画いっぱいの 1 シーン» の略記。
    if not cloned.get("scenes"):
        scene = {"id": "main", "start": 0, "layers": cloned.get("layers") or []}
        if cloned["video"].get("duration") is not None:
            scene["duration"] = cloned["video"]["duration"]
        cloned["scenes"] = [scene]
    cloned.pop("layers", None)

    # 曲の区間（from: { section }）を実際の秒に畳む。**小節の解決より前**です。
    # ここで数値にしておかないと、区間から来た尺が二重に変換されます。
    resolve_structure(cloned)

    # 拍・小節の書き方（"4bar" / "8beat"）を秒へ直す。プリセットを畳み込む前に
    # やるのは、プリセット側にも拍で書けるようにするためです。
    resolve_musical_time(cloned)

    # プリセット（エイリアス）をレイヤーへ畳み込む。ここで解決しておけば
    # レンダラ・タイムライン以降は preset を知らなくてよい。
    resolve_presets(cloned)

    # 相対単位（"50%" / "7vh"）を px へ直す。プリセットを畳んだ «あと» なのは、
    # プリセット側にも相対単位で書けるようにするためです。kenBurns の展開より
    # «前» なのは、あちらが from / to を数値として読むためです。
    resolve_relative_units(cloned)

    # kenBurns をキーフレームへ展開する。手で書いたキーフレームと同じ結果になる。
    _expand_ken_burns(cloned)

    cloned["physicsWorld"] = {
        "engine": "movo-physics-2d",
        "gravity": {"x": 0, "y": 980},
        "timeStep": 1 / 60,
        "subSteps": cloned["render"]["physics"]["subSteps"],
        "iterations": cloned["render"]["physics"]["iterations"],
        "pixelsPerMeter": 100,
        "enabled": True,
        **(cloned.get("physicsWorld") or {}),
    }

    return cloned


__all__ = [
    "DEFORMER_TYPES",
    "EFFECT_TYPES",
    "LAYER_TYPES",
    "MASK_TYPES",
    "MODULATOR_TYPES",
    "MOVO_JSON_VERSION",
    "PHYSICS_CONTROL_MODES",
    "PLUGIN_KINDS",
    "QUALITY_PRESETS",
    "QUALITY_SETTINGS",
    "RECIPE_SCHEMA",
    "RENDERER_TYPES",
    "SHAPE_TYPES",
    "SchemaValidator",
    "apply_extends",
    "apply_variant",
    "assert_valid_project",
    "axis_of_key",
    "build_recipe",
    "check_recipe_assets",
    "describe_presets",
    "expand_all_variants",
    "expand_params",
    "find_section",
    "find_unresolved_relative_units",
    "is_musical_time",
    "is_relative_unit",
    "join_path",
    "list_params",
    "list_variants",
    "merge_deep",
    "musical_units",
    "normalize_project",
    "param_overrides_from",
    "parse_param_assignments",
    "prepare_project",
    "project_schema",
    "read_recipe",
    "relative_to_pixels",
    "resolve_musical_time",
    "resolve_params",
    "resolve_presets",
    "resolve_relative_units",
    "resolve_structure",
    "strip_json_comments",
    "structure_of",
    "to_seconds",
    "validate_project",
    "validate_structure",
    "variant_names",
    "write_recipe",
]
