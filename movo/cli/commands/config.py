"""`movo config` — API キーと既定値。プロジェクト JSON の外に置きます。"""

from __future__ import annotations

import json
import sys
from typing import Any

from ..config_store import (
    config_file_path,
    get_config_value,
    is_secret_key,
    list_config,
    mask_secret,
    set_config_value,
    unset_config_value,
)
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError


def config_command(positional: list[str], options: dict[str, Any]) -> Any:
    action = positional[0] if positional else "list"
    key = positional[1] if len(positional) > 1 else None
    value = " ".join(positional[2:])

    if action == "set":
        if not key:
            raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "movo config set <key> <value>")
        if not value:
            # 標準入力から読むと、シェルの履歴に秘密が残りません。
            stdin_value = _read_stdin()
            if not stdin_value:
                raise MovoError(
                    ErrorCodes.MOVO_CLI_USAGE,
                    f'"{key}" に入れる値がありません',
                    hint=f"movo config set {key} <値>   もしくは  echo <値> | movo config set {key}",
                )
            file = set_config_value(key, stdin_value.strip())
            logger.success(f"{key} を保存しました ({file})")
            return {"key": key, "file": file}
        file = set_config_value(key, _coerce(value))
        shown = mask_secret(value) if is_secret_key(key) else value
        logger.success(f"{key} = {shown}  →  {file}")
        return {"key": key, "file": file}

    if action == "get":
        if not key:
            raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "movo config get <key>")
        current = get_config_value(key)
        if current is None:
            logger.info(f"{key} は設定されていません")
            return {"key": key, "value": None}
        display = mask_secret(current) if is_secret_key(key) else current
        if options.get("json"):
            say(json.dumps({"key": key, "value": display}, ensure_ascii=False, indent=2))
        else:
            logger.info(f"{key} = {display}")
        return {"key": key, "value": display}

    if action == "unset":
        if not key:
            raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "movo config unset <key>")
        file = unset_config_value(key)
        logger.success(f"{key} を削除しました ({file})")
        return {"key": key, "file": file}

    if action == "path":
        say(config_file_path())
        return {"path": config_file_path()}

    entries = list_config()
    if options.get("json"):
        say(json.dumps(entries, ensure_ascii=False, indent=2, default=str))
        return entries
    logger.info(style.bold(f"設定ファイル: {config_file_path()}"))
    if not entries:
        logger.info("  （空）")
        logger.info("")
        logger.info("  例: movo config set openai.apiKey sk-...")
        logger.info("      movo config set gemini.apiKey ...")
        return entries
    logger.info("")
    for entry in entries:
        suffix = style.gray("(環境変数)") if entry.get("source") == "env" else ""
        logger.info(f'  {str(entry["key"]).ljust(28)} {entry["value"]}  {suffix}')
    return entries


def _coerce(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    # 「書いたとおりに戻る」ものだけ数値にします（"007" を 7 にしない）。
    if str(int(number)) == value:
        return int(number)
    if str(number) == value:
        return number
    return value


def _read_stdin() -> str:
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        return sys.stdin.read()
    except Exception:  # noqa: BLE001
        return ""
