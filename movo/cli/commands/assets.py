"""`movo assets generate` と `movo assets plan`。

`plan` は **API を 1 回も叩きません**。何がどのプロバイダーへ、どんな指示で
送られるかを先に読めるようにするためのものです（課金と内容の両方を、
実行する前に確かめられます）。
"""

from __future__ import annotations

import json
import os
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import create_session, read_project_file


def assets_command(positional: list[str], options: dict[str, Any]) -> Any:
    action = positional[0] if positional else None
    file = positional[1] if len(positional) > 1 else "movo.json"
    if action == "plan":
        return _assets_plan(file, options)
    if action == "generate":
        return _assets_generate(file, options)
    raise MovoError(
        ErrorCodes.MOVO_CLI_USAGE,
        f'不明なサブコマンド "movo assets {action or ""}"',
        hint="使えるのは: movo assets generate <file> | movo assets plan <file>",
    )


def _assets_plan(file: str, options: dict[str, Any]) -> Any:
    read = read_project_file(file)
    normalize_project = bridge.pick("movo.schema", "normalize_project", "normalizeProject")
    plan_ai_assets = bridge.pick("movo.ai", "plan_ai_assets", "planAiAssets")
    planned = plan_ai_assets(normalize_project(read["raw"]))

    if options.get("json"):
        say(json.dumps({"assets": planned, "count": len(planned)}, ensure_ascii=False, indent=2, default=str))
        return planned
    if not planned:
        logger.info("AI で生成する素材は宣言されていません。")
        return planned
    logger.info(style.bold(f"AI 素材 {len(planned)} 件"))
    for entry in planned:
        logger.info("")
        logger.info(f'  {style.bold(entry["asset"])}  ({entry["type"]})')
        logger.info(f'    provider : {entry["provider"]}')
        logger.info(f'    model    : {entry["model"]}')
        logger.info(f'    prompt   : {_truncate(entry.get("prompt") or "(なし)", 90)}')
        if entry.get("parts"):
            logger.info(f'    parts    : {", ".join(entry["parts"])}')
    logger.info("")
    logger.info("  実際に生成するには: movo assets generate <file>")
    logger.info("  API キー: movo config set openai.apiKey <キー>")
    return planned


def _assets_generate(file: str, options: dict[str, Any]) -> Any:
    session = create_session(
        file, {"generate_assets": True, "no_cache": options.get("force") is True, "dry_run_ai": False}
    )
    generator = session.get("generator")
    if generator is None:
        raise MovoError(
            ErrorCodes.MOVO_RENDERER_UNAVAILABLE,
            "AI 素材の生成（movo.ai）はまだ移植されていません — 後で繋ぐ",
            hint="素材を手で置けば movo render は動きます",
        )
    generated = generator.generated
    plan = generator.plan

    if options.get("json"):
        say(json.dumps({"generated": generated, "plan": plan}, ensure_ascii=False, indent=2, default=str))
        return generated

    if not plan:
        logger.info("AI で生成する素材は宣言されていません。")
        return generated
    reused = len([entry for entry in plan if entry.get("cached")])
    logger.success(f"AI 素材: 新規 {len(generated)} 件 / 再利用 {reused} 件")
    for entry in generated:
        logger.info(f'  {entry["asset"]} → {os.path.relpath(entry["path"], os.getcwd())}')
    placeholders = session["assets"].stats().get("placeholders", 0)
    if placeholders:
        logger.warn(f"{placeholders} 件はプレースホルダのままです（API キーの設定を確認してください）")
    return generated


def _truncate(text: str, length: int) -> str:
    value = " ".join(str(text).split())
    return f"{value[: length - 1]}…" if len(value) > length else value
