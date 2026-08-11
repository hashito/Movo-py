"""`movo init` — 新しいプロジェクトを作る。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..console import logger, style
from ..errors import ErrorCodes, MovoError
from ..templates import TEMPLATES, build_sample_assets, build_template

DIRECTORIES = [
    "assets/images",
    "assets/videos",
    "assets/audio",
    "assets/fonts",
    "assets/generated",
    "characters/motions",
    "compositions",
    "plugins",
    "cache",
    "output",
]

GITIGNORE = """cache/
output/
assets/generated/
*.log
"""


def init_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    name = positional[0] if positional else None
    if not name:
        raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "プロジェクト名が要ります", hint="movo init my-video")
    template = options.get("template") or "basic"
    if template not in TEMPLATES:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            f'不明なテンプレート "{template}"',
            hint=f'使えるのは: {", ".join(TEMPLATES)}',
        )

    root = Path(name).resolve()
    if root.exists() and any(root.iterdir()) and not options.get("force"):
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            f'"{name}" は既にあり、空ではありません',
            hint="それでも書き込むなら --force",
        )

    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)

    project = build_template(
        template,
        {
            "name": root.name,
            "width": options.get("width"),
            "height": options.get("height"),
            "fps": options.get("fps"),
            "font": options.get("font"),
        },
    )
    (root / "movo.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (root / "README.md").write_text(_readme_for(root.name, template), encoding="utf-8")

    # サンプル画像は PNG のエンコーダ（movo.core.png）が要ります。まだ繋がって
    # いない間も **プロジェクトの雛形だけは作れる** ようにしておきます
    # （素材が無くても JSON の書き方は確かめられるので、そのほうが役に立ちます）。
    written = 0
    try:
        assets = build_sample_assets(template)
    except Exception as error:  # noqa: BLE001
        logger.warn(f"サンプル画像は作れませんでした（{error}）")
        logger.warn("  assets/images/ に画像を置いてから movo render してください")
        assets = {}
    for relative, data in assets.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written += 1

    shown = os.path.relpath(root, os.getcwd()) or "."
    logger.success(f"プロジェクトを作成しました: {shown}")
    logger.info("")
    logger.info(f'  {style.bold("次の手順")}')
    logger.info(f"    cd {shown}")
    logger.info("    movo validate movo.json")
    logger.info("    movo render movo.json")
    logger.info("")
    others = ", ".join(t for t in TEMPLATES if t != template)
    logger.info(f"  テンプレート: {template}（他: {others}）")
    return {"root": str(root), "template": template, "files": written + 3}


def _readme_for(name: str, template: str) -> str:
    return f"""# {name}

Movo で作成した動画プロジェクトです（テンプレート: `{template}`）。

## 使い方

```bash
movo validate movo.json      # JSON を検証
movo render movo.json        # 動画を書き出す（output/ 以下）
movo render movo.json --jobs auto   # 区間に割って同時に描く（長い曲向け）
movo frame movo.json -t 1.5  # 1 フレームだけ確認
movo preview movo.json       # ブラウザでプレビュー
```

ffmpeg が無い環境では MP4 の代わりにアニメーション GIF が出力されます。
`movo doctor` で実行環境を確認できます。

## ディレクトリ

| パス | 内容 |
| --- | --- |
| `movo.json` | プロジェクト定義 |
| `assets/` | 画像・動画・音声・フォント |
| `characters/` | リグとモーション |
| `compositions/` | 再利用する部品 |
| `plugins/` | ローカルプラグイン |
| `cache/` | 生成物のキャッシュ（コミット不要） |
| `output/` | 書き出し先（コミット不要） |
"""
