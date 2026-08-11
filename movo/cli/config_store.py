"""設定（API キーなど）の置き場。

**プロジェクト JSON には絶対に書きません。** JSON は共有されるものなので、
そこに API キーが載ると、共有した瞬間に漏れます。`~/.movo/config.json` に
本人だけが読める権限（600）で置きます。

`movo.core.config` が用意できたらそちらを使います（下の `_core`）。
ここに実装があるのは、**設定は描画が繋がっていなくても要る** からです
（`movo config set` は初日の最初のコマンドになり得ます）。
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from . import bridge

# 環境変数のほうが強い、という決めごと。CI では «ファイルを置かずに» 渡したい
# ためです。名前は JS 版と揃えてあります。
ENVIRONMENT_KEYS = {
    "openai.apiKey": "MOVO_OPENAI_API_KEY",
    "gemini.apiKey": "MOVO_GEMINI_API_KEY",
    "stability.apiKey": "MOVO_STABILITY_API_KEY",
}


def movo_home() -> Path:
    override = os.environ.get("MOVO_HOME")
    return Path(override) if override else Path.home() / ".movo"


def config_file_path() -> str:
    core = _core()
    if core is not None:
        function = getattr(core, "config_file_path", None) or getattr(core, "configFilePath", None)
        if function:
            return function()
    return str(movo_home() / "config.json")


def _core():
    return bridge.optional_module("movo.core.config")


def _read() -> dict[str, Any]:
    path = Path(config_file_path())
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(data: dict[str, Any]) -> str:
    path = Path(config_file_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        # 600。Windows では意味を持ちませんが、失敗はしません。
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return str(path)


def is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return "apikey" in lowered or "token" in lowered or "secret" in lowered or "password" in lowered


def mask_secret(value: Any) -> str:
    text = str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * (len(text) - 8)}{text[-4:]}"


def get_config_value(key: str) -> Any:
    core = _core()
    if core is not None and hasattr(core, "get_config_value"):
        return core.get_config_value(key)
    environment = ENVIRONMENT_KEYS.get(key)
    if environment and os.environ.get(environment):
        return os.environ[environment]
    node: Any = _read()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_config_value(key: str, value: Any) -> str:
    core = _core()
    if core is not None and hasattr(core, "set_config_value"):
        return core.set_config_value(key, value)
    data = _read()
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return _write(data)


def unset_config_value(key: str) -> str:
    core = _core()
    if core is not None and hasattr(core, "unset_config_value"):
        return core.unset_config_value(key)
    data = _read()
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            return _write(data)
        node = node[part]
    node.pop(parts[-1], None)
    return _write(data)


def list_config() -> list[dict[str, Any]]:
    """設定の一覧。**秘密は伏せた形で返します。**

    一覧を出しただけで端末の履歴に API キーが残るのは避けたいので、
    伏せるかどうかを «出す側» ではなくここで決めています。
    """
    core = _core()
    if core is not None and hasattr(core, "list_config"):
        return core.list_config()
    entries: list[dict[str, Any]] = []

    def walk(node: Any, prefix: str = "") -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, path)
            else:
                entries.append(
                    {"key": path, "value": mask_secret(value) if is_secret_key(path) else value, "source": "file"}
                )

    walk(_read())
    known = {entry["key"] for entry in entries}
    for key, environment in ENVIRONMENT_KEYS.items():
        if os.environ.get(environment) and key not in known:
            entries.append({"key": key, "value": mask_secret(os.environ[environment]), "source": "env"})
    return sorted(entries, key=lambda e: e["key"])
