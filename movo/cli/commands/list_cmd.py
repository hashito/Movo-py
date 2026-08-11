"""`movo list <種類>` — この版が何をできるかを一覧する。

**未接続のものは «0 件» と出します。** ここでエラーにすると «何が使えるか調べる»
という目的が果たせません。移植の途中でどこまで揃ったかを見るのにも使えます。
"""

from __future__ import annotations

import json
import os
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import read_project_file

KINDS = [
    "effects",
    "deformers",
    "physics",
    "modulators",
    "easings",
    "functions",
    "layers",
    "formats",
    "masks",
    "renderers",
    "particles",
    "presets",
    "profiles",
    "blends",
]


def list_command(positional: list[str], options: dict[str, Any]) -> Any:
    kind = positional[0] if positional else None
    if not kind:
        logger.info(f'使い方: movo list <{"|".join(KINDS)}>')
        logger.info("  presets はプロジェクトごとの定義なので、ファイルを指定します: movo list presets movo.json")
        return KINDS
    # presets だけはプロジェクトの中身なので、ファイルを読む
    if kind == "presets":
        return _list_presets(positional[1] if len(positional) > 1 else "movo.json", options)
    data = _collect(kind)
    if data is None:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE, f'不明な種類 "{kind}"', hint=f'使えるのは: {", ".join(KINDS)}'
        )
    if options.get("json"):
        say(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return data
    _print(kind, data)
    return data


def _list_presets(file: str, options: dict[str, Any]) -> Any:
    read = read_project_file(file)
    describe_presets = bridge.pick("movo.schema.presets", "describe_presets", "describePresets")
    presets = describe_presets(read["raw"])
    if options.get("json"):
        say(json.dumps(presets, ensure_ascii=False, indent=2, default=str))
        return presets
    logger.info(style.bold(f"プリセット {len(presets)} 件"))
    if not presets:
        logger.info("  定義がありません。project.presets に書くとレイヤーから preset で参照できます。")
        return presets
    for preset in presets:
        logger.info(f'  {preset["name"]}')
        if preset.get("description"):
            logger.info(f'    {preset["description"]}')
        if preset.get("extends"):
            logger.info(f'    継承: {", ".join(preset["extends"])}')
        logger.info(f'    内容: {", ".join(preset.get("provides", []))}')
    logger.info("")
    logger.info('  使い方: レイヤーに "preset": "<名前>" または "preset": ["<名前>", ...]')
    return presets


def _collect(kind: str) -> Any:
    if kind == "effects":
        return bridge.listing("movo.renderer.effects", "list_effects", "listEffects")
    if kind == "deformers":
        return bridge.listing("movo.deformer", "describe_deformers", "describeDeformers")
    if kind == "physics":
        return bridge.listing("movo.physics", "describe_physics", "describePhysics")
    if kind == "modulators":
        return {
            "types": bridge.listing("movo.animation.modulators", "list_modulators", "listModulators"),
            "combine": bridge.listing("movo.animation.modulators", "COMBINE_MODES"),
        }
    if kind == "easings":
        return bridge.listing("movo.animation.easing", "list_easings", "listEasings")
    if kind == "functions":
        engine_class = bridge.pick("movo.expression", "ExpressionEngine")
        if getattr(engine_class, "movo_not_connected", False):
            return {"functions": [], "constants": []}
        engine = engine_class()
        return {"functions": engine.list_functions(), "constants": engine.list_constants()}
    if kind == "layers":
        return bridge.listing("movo.schema.project_schema", "LAYER_TYPES")
    if kind == "formats":
        return bridge.listing("movo.exporters", "list_exporters", "listExporters")
    if kind == "masks":
        return bridge.listing("movo.deformer.mask", "MASK_TYPES")
    if kind == "renderers":
        return bridge.listing("movo.renderer", "RENDERER_KINDS")
    if kind == "particles":
        return bridge.listing("movo.renderer.particle_presets", "list_particle_presets", "listParticlePresets")
    if kind == "blends":
        return bridge.listing("movo.renderer.raster", "BLEND_MODES")
    if kind == "profiles":
        # movo compare --target <名前> で使える «作風の目標値»
        #
        # **作業ディレクトリを渡します。** 渡さないと `list_profiles` は組み込みしか
        # 読まず、`./profiles/*.json` に置いた自作の作風が一覧に出ません
        # （`profile_library` の説明どおり «組み込み → プロジェクト固有» の順で
        # 上書きできるのに、一覧からは見えない、という食い違いになっていました）。
        list_profiles = bridge.pick("movo.core.profile_library", "list_profiles", "listProfiles")
        try:
            entries = list(list_profiles(os.getcwd()) or [])
        except Exception:  # noqa: BLE001 - 一覧のために止まりたくはない
            entries = []
        return [
            {
                "name": entry.get("name"),
                "description": (
                    f'{entry.get("label")} — {entry.get("note", "")}'.strip() if entry.get("label") else entry.get("note")
                ),
            }
            for entry in entries
            if isinstance(entry, dict)
        ]
    return None


def _print(kind: str, data: Any) -> None:
    logger.info(style.bold(f"movo {kind}"))
    logger.info("")
    if isinstance(data, list):
        if not data:
            logger.info("  （0 件 — この種類を持つモジュールがまだ繋がっていません）")
            return
        if isinstance(data[0], dict):
            for entry in data:
                label = entry.get("format") or entry.get("name") or json.dumps(entry, ensure_ascii=False)
                extra = " (ffmpeg が必要)" if entry.get("requiresFfmpeg") else ""
                logger.info(f"  {label}{extra}")
            return
        columns = 3
        width = max(len(str(d)) for d in data) + 3
        for i in range(0, len(data), columns):
            logger.info(("  " + "".join(str(d).ljust(width) for d in data[i : i + columns])).rstrip())
        logger.info("")
        logger.info(f"  {len(data)} 件")
        return
    for key, value in data.items():
        if isinstance(value, list):
            logger.info(f'  {style.bold(key)}: {", ".join(map(str, value))}')
        elif isinstance(value, dict):
            logger.info(f"  {style.bold(key)}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {style.bold(key.ljust(14))} {value}")
