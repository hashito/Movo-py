"""目標値（プロファイル）の置き場。

スキルと同じ考え方で、**組み込み → プロジェクト固有** の順に読み、後から
読んだものが勝ちます。自分の作風を ``profiles/`` に置けば、同梱のスタイルと
同じように名前で呼べます。

1. ``movo/library/profiles/*.json``（組み込み）
2. ``<プロジェクト>/profiles/*.json``（自作・上書き）
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import ErrorCodes, MovoError

#: 組み込みのプロファイルの置き場。**パッケージの中**を指します
#: （EXE に固めたときも同じ相対位置に入るため）。
BUILTIN_ROOT = Path(__file__).resolve().parent.parent / "library" / "profiles"


def _read_dir(directory: str | os.PathLike[str] | None) -> list[dict[str, Any]]:
    """1 つのディレクトリから読む。無ければ空。"""
    if not directory:
        return []
    path = Path(directory)
    if not path.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for file in sorted(path.iterdir()):
        if file.suffix != ".json":
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID, f"プロファイルを読めません: {file}", cause=error
            ) from error
        entry = dict(data)
        entry["name"] = data.get("name") or file.stem
        entry["source"] = str(file)
        out.append(entry)
    return out


def list_profiles(project_dir: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """使えるプロファイルを列挙する。名前が同じならプロジェクト側が勝ちます。"""
    merged: dict[str, dict[str, Any]] = {}
    for entry in _read_dir(BUILTIN_ROOT):
        merged[entry["name"]] = {**entry, "builtin": True}
    if project_dir:
        for entry in _read_dir(Path(project_dir) / "profiles"):
            merged[entry["name"]] = {**entry, "builtin": False}
    return sorted(merged.values(), key=lambda e: e["name"])


def load_profile_target(
    name_or_path: str, project_dir: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """名前かファイルパスから目標値を引く。

    **ファイルとして存在すればファイル優先**です。``./x.json`` と名前 ``x`` が
    ぶつかったときに «手元のファイルが無視される» のが一番困るためです。
    """
    candidate = Path(name_or_path)
    if candidate.is_file():
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return {
            "name": data.get("name") or candidate.stem,
            "label": data.get("label"),
            "target": data.get("target", data),
        }

    available = list_profiles(project_dir)
    hit = next((entry for entry in available if entry["name"] == name_or_path), None)
    if hit is None:
        raise MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND,
            f"プロファイルが見つかりません: {name_or_path}",
            hint=f"使えるのは {' / '.join(e['name'] for e in available)}（movo list profiles でも見られます）",
        )
    return {"name": hit["name"], "label": hit.get("label"), "target": hit.get("target", hit)}
