"""`movo validate` — スキーマ検証と意味検証。"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import validate_file


def validate_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    file = (positional[0] if positional else None) or "movo.json"
    result = validate_file(file)

    if options.get("json"):
        payload = {
            "file": result["file"],
            "valid": result.get("valid", False)
            and (not options.get("strict") or not result.get("warnings")),
            "issues": result.get("issues", []),
            "warnings": result.get("warnings", []),
        }
        say(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        if not payload["valid"]:
            raise SystemExit(3)
        return payload

    short_file = os.path.relpath(result["file"], os.getcwd())
    issues = result.get("issues", [])
    warnings = result.get("warnings", [])
    for issue in issues:
        logger.error(style.bold("MOVO_SCHEMA_INVALID"))
        logger.info(f"  File:   {short_file}")
        logger.info(f'  Path:   {issue.get("path") or "(root)"}')
        logger.info(f'  Reason: {issue.get("message")}')
        logger.info("")
    for warning in warnings:
        logger.warn(f'{warning.get("path")}: {warning.get("message")}')

    if issues:
        logger.error(f"検証に失敗しました（エラー {len(issues)} 件、警告 {len(warnings)} 件）")
        raise SystemExit(3)

    # タイムラインの概要を出して、作り手が構造を確かめられるようにする。
    # result["raw"] はスキル展開後なので、use を書いたプロジェクトでも実際の構成が出る。
    normalize_project = bridge.pick("movo.schema", "normalize_project", "normalizeProject")
    build_timeline = bridge.pick("movo.timeline", "build_timeline", "buildTimeline")
    all_layers = bridge.pick("movo.timeline", "all_layers", "allLayers")
    project = normalize_project(result["raw"])
    timeline = build_timeline(project)
    layers = all_layers(timeline)

    logger.success(f"{short_file} は有効です")
    logger.info(
        f'  {timeline["width"]}x{timeline["height"]} @ {timeline["fps"]}fps / '
        f'{timeline["duration"]:.2f}秒 / {timeline["frameCount"]} フレーム'
    )
    logger.info(
        f'  シーン: {len(timeline["scenes"])}  レイヤー: {len(layers)}  '
        f'素材: {len(project.get("assets") or {})}'
    )
    by_type = Counter(layer.get("type") for layer in layers)
    if by_type:
        logger.info("  レイヤー種別: " + ", ".join(f"{t}×{n}" for t, n in by_type.items()))
    if result.get("skillsUsed"):
        logger.info("  スキル: " + ", ".join(f'{e["skill"]}({e["layers"]})' for e in result["skillsUsed"]))
    if warnings:
        logger.info(f"  警告 {len(warnings)} 件")
    if options.get("strict") and warnings:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f"--strict が指定されているため警告 {len(warnings)} 件を失敗として扱います",
        )
    return {**result, "timeline": timeline}
