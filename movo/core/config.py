"""利用者ごとの設定（API キー・既定値）。

**秘密はプロジェクト JSON に置きません**（仕様 34 節）。環境変数を先に見て、
次に ``~/.movo/config.json`` を読みます。ファイルは他人から読めない権限
（0600）で書きます。

プロジェクト JSON に API キーが入ると、そのまま git に入り、そのまま公開されます。
**«うっかり» を仕組みで防ぐ**ためにここを分けています。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .platform import movo_home

_SECRET_KEY_PATTERN = re.compile(r"(apikey|token|secret|password)$", re.IGNORECASE)


def config_file_path() -> str:
    return str(Path(movo_home()) / "config.json")


def load_config() -> dict[str, Any]:
    """設定を読む。**壊れていても例外にせず空にします。**

    設定ファイルが壊れているだけでレンダリングまで止まるのは割に合いません。
    """
    try:
        with open(config_file_path(), encoding="utf-8") as handle:
            parsed = json.load(handle)
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(config: dict[str, Any]) -> str:
    path = Path(config_file_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    # 先に権限を絞ってから書きます。書いてから chmod すると、その一瞬だけ
    # 他人に読める状態ができます。
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass  # Windows では意味を持たない
    return str(path)


def _env_name_for(key: str) -> str:
    """``openai.apiKey`` → ``MOVO_OPENAI_API_KEY``。JS 版と同じ変換規則です。"""
    name = re.sub(r"[.-]", "_", key)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return f"MOVO_{name.upper()}"


def get_config_value(key: str, fallback: Any = None) -> Any:
    """ドット区切りの鍵で引く。**環境変数が最優先**、次に設定ファイル。

    環境変数を先にするのは、CI やコンテナで «ファイルを置かずに» 渡せる
    ようにするためです。
    """
    env = os.environ.get(_env_name_for(key))
    if env:
        return env
    node: Any = load_config()
    for part in key.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return fallback
    return fallback if node is None else node


def set_config_value(key: str, value: Any) -> str:
    config = load_config()
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return save_config(config)


def unset_config_value(key: str) -> str:
    config = load_config()
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            return config_file_path()
        node = node[part]
    node.pop(parts[-1], None)
    return save_config(config)


def is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_PATTERN.search(key))


def mask_secret(value: Any) -> str:
    """秘密を伏せる。**両端だけ残します**（本人が «どの鍵か» を見分けられるように）。"""
    text = "" if value is None else str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * max(4, len(text) - 8)}{text[-4:]}"


def list_config() -> list[dict[str, Any]]:
    """設定を ``key = value`` の並びに平たくする。秘密は伏せます。"""
    out: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], prefix: str) -> None:
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, key)
            else:
                out.append({"key": key, "value": mask_secret(v) if is_secret_key(key) else v, "source": "config"})

    walk(load_config(), "")
    for key in ("openai.apiKey", "gemini.apiKey"):
        env = os.environ.get(_env_name_for(key))
        if env:
            out.append({"key": key, "value": mask_secret(env), "source": "env"})
    return out
