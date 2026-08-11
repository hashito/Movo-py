"""`movo plugin list` と `movo plugin create`。

プラグインは **許可リストに載っているものだけ** を読みます（`security.allowPlugins`）。
プロジェクト JSON を渡されただけで任意のコードが走るのは危険なので、
そこは JS 版と同じ約束のままです。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import read_project_file

PLUGIN_KINDS = ["deformer", "effect", "modulator", "layerType", "exporter"]

PLUGIN_TEMPLATE = '''"""__NAME__ — Movo プラグイン。

プロジェクトの movo.json に次を足すと読み込まれます。

    "plugins": [{ "name": "__NAME__" }],
    "security": { "allowPlugins": ["__NAME__"] }

許可リストに書かないと読み込まれません（JSON を受け取っただけで任意の
コードが走らないようにするためです）。
"""

import numpy as np


def register(api):
    """Movo から 1 回だけ呼ばれます。ここで名前を登録します。"""

    @api.effect("__NAME__-tint")
    def tint(bitmap, params, context):
        """全画面に色を乗せるだけの例。

        **画素ごとの for を書かないでください。** NumPy の一括演算で書きます
        （純 Python のループは 1280x720 で 720 ミリ秒かかります）。
        """
        amount = float(params.get("amount", 0.2))
        colour = np.array(params.get("color", (255, 120, 60)), dtype=np.float32)
        rgb = bitmap.data[..., :3].astype(np.float32)
        bitmap.data[..., :3] = (rgb * (1 - amount) + colour * amount).astype(np.uint8)
        return bitmap
'''


def plugin_command(positional: list[str], options: dict[str, Any]) -> Any:
    action = positional[0] if positional else "list"
    if action == "list":
        return _plugin_list(positional[1] if len(positional) > 1 else "movo.json", options)
    if action == "create":
        return _plugin_create(positional[1] if len(positional) > 1 else None, options)
    if action == "kinds":
        if options.get("json"):
            say(json.dumps(PLUGIN_KINDS, ensure_ascii=False, indent=2))
            return PLUGIN_KINDS
        logger.info("\n".join(f"  {k}" for k in PLUGIN_KINDS))
        return PLUGIN_KINDS
    raise MovoError(
        ErrorCodes.MOVO_CLI_USAGE,
        f'不明なサブコマンド "movo plugin {action}"',
        hint="使えるのは: movo plugin list | movo plugin create <名前> | movo plugin kinds",
    )


def _plugin_list(file: str, options: dict[str, Any]) -> Any:
    project: dict[str, Any]
    project_root = str(Path.cwd())
    try:
        read = read_project_file(file)
        normalize_project = bridge.pick("movo.schema", "normalize_project", "normalizeProject")
        project = normalize_project(read["raw"])
        project_root = read["projectRoot"]
    except Exception:  # noqa: BLE001 - 読めなくても «一覧» は出したい
        project = {"plugins": [], "security": {}}

    load_plugins = bridge.pick("movo.plugin_sdk", "load_plugins", "loadPlugins")
    if getattr(load_plugins, "movo_not_connected", False):
        # プラグインの読み込みは «後で繋ぐ»。宣言だけは読めるので、そこは出します。
        declared = project.get("plugins") or []
        if options.get("json"):
            say(json.dumps({"plugins": [], "declared": declared, "loaded": False}, ensure_ascii=False, indent=2))
            return []
        logger.warn("プラグインの読み込み（movo.plugin_sdk）はまだ移植されていません — 後で繋ぐ")
        logger.info(f"  宣言されているプラグイン: {len(declared)} 件")
        for entry in declared:
            logger.info(f'    {entry.get("name") if isinstance(entry, dict) else entry}')
        return []

    registry = load_plugins(project, project_root=project_root, strict=False)
    entries = registry.list()
    if options.get("json"):
        say(json.dumps({"plugins": entries, "declared": project.get("plugins") or []}, ensure_ascii=False, indent=2))
        return entries
    if not entries:
        logger.info("読み込まれたプラグインはありません。")
        logger.info("  雛形を作る: movo plugin create my-effect")
        return entries
    logger.info(style.bold(f"プラグイン {len(entries)} 件"))
    for plugin in entries:
        logger.info(f'  {plugin["name"]}@{plugin["version"]}')
        logger.info(f'    種類  : {", ".join(plugin.get("kinds") or []) or "(未宣言)"}')
        if plugin.get("permissions"):
            logger.info(f'    権限  : {", ".join(plugin["permissions"])}')
    return entries


def _plugin_create(name: str | None, options: dict[str, Any]) -> dict[str, Any]:
    if not name:
        raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "プラグイン名が要ります", hint="movo plugin create my-effect")
    directory = Path("plugins") / name
    entry = directory / "__init__.py"
    if entry.exists() and not options.get("force"):
        raise MovoError(ErrorCodes.MOVO_CLI_USAGE, f"{entry} は既にあります", hint="上書きするなら --force")
    directory.mkdir(parents=True, exist_ok=True)
    entry.write_text(PLUGIN_TEMPLATE.replace("__NAME__", name), encoding="utf-8")
    (directory / "README.md").write_text(
        "\n".join(
            [
                f"# {name}",
                "",
                "Movo プラグインです。",
                "",
                "プロジェクトの movo.json に次を追加すると読み込まれます。",
                "",
                "```json",
                "{",
                f'  "plugins": [{{ "name": "{name}" }}],',
                f'  "security": {{ "allowPlugins": ["{name}"] }}',
                "}",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.success(f"プラグインの雛形を作成しました: {entry}")
    logger.info("  movo.json の plugins と security.allowPlugins に名前を追加してください。")
    return {"name": name, "path": str(entry)}
