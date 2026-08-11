"""`movo skill` — 基礎アニメーションとスキルを一覧・確認・描画する。

  movo skill list                      すべて一覧
  movo skill list --animations         基礎アニメーションだけ
  movo skill list --skills             スキルだけ
  movo skill list --scenes             シーンスキルだけ
  movo skill list --movies             ムービースキルだけ
  movo skill show cutin-title          入力値と中身を表示
  movo skill render cutin-title --set text=タイトル -o tmp/out.mp4
  movo skill render lyric-mv --set title=夜明けまで --set bpm=92 -o tmp/a.mp4
  movo skill expand cutin-title --set text=タイトル -o project.json
  movo skill new my-skill              雛形を作る
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from movo.skill import (
    ANIMATION_CATEGORIES,
    SKILL_CATEGORIES,
    SkillRegistry,
    build_skill_project,
    find_dead_inputs,
    parse_input_assignments,
)

from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import create_session, render_video

ACTIONS = ["list", "show", "render", "expand", "new"]

# よく使う解像度の近道。
PRESETS = {
    "1080p": {"width": 1920, "height": 1080},
    "720p": {"width": 1280, "height": 720},
    "480p": {"width": 854, "height": 480},
    "square": {"width": 1080, "height": 1080},
    "vertical": {"width": 1080, "height": 1920},
    "shorts": {"width": 1080, "height": 1920},
    "thumb": {"width": 640, "height": 360},
}

# 種類の見出し。一覧・詳細の両方で使う。
KIND_LABELS = {
    "animation": "基礎アニメーション",
    "skill": "スキル",
    "scene": "シーンスキル",
    "movie": "ムービースキル",
}


def skill_command(positional: list[str], options: dict[str, Any]) -> Any:
    action = (positional[0] if positional else None) or "list"
    name = positional[1] if len(positional) > 1 else None
    registry = SkillRegistry().load(project_root=os.getcwd())

    if action == "list":
        return _skill_list(registry, options)
    if action == "show":
        return _skill_show(registry, name, options)
    if action == "render":
        return _skill_render(registry, name, options)
    if action == "expand":
        return _skill_expand(registry, name, options)
    if action == "new":
        return _skill_new(name, options)

    raise MovoError(
        ErrorCodes.MOVO_CLI_USAGE,
        f'不明なサブコマンド "movo skill {action}"',
        hint="使えるのは: " + " / ".join(f"movo skill {a}" for a in ACTIONS),
    )


def _inputs_from(options: dict[str, Any]) -> dict[str, Any]:
    """入力値を集める（--set key=value を何度でも、--with '{"json":1}' も可）。"""
    assignments = parse_input_assignments(options.get("set"))
    from_json: dict[str, Any] = {}
    if isinstance(options.get("with"), str):
        try:
            from_json = json.loads(options["with"])
        except ValueError as error:
            raise MovoError(ErrorCodes.MOVO_CLI_USAGE, f"--with の JSON を読めません: {error}") from error
    if isinstance(options.get("inputs"), str):
        file = Path(options["inputs"]).resolve()
        if not file.exists():
            raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f'入力値ファイルが見つかりません: {options["inputs"]}')
        from_json.update(json.loads(file.read_text(encoding="utf-8")))
    return {**from_json, **assignments}


def _overrides_from(options: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("width", "height", "fps", "seed", "duration", "bpm"):
        if isinstance(options.get(key), (int, float)) and not isinstance(options.get(key), bool):
            overrides[key] = options[key]
    for key in ("quality", "background"):
        if isinstance(options.get(key), str):
            overrides[key] = options[key]
    if isinstance(options.get("preset"), str):
        preset = PRESETS.get(options["preset"])
        if preset is None:
            raise MovoError(
                ErrorCodes.MOVO_CLI_USAGE,
                f'不明なプリセット "{options["preset"]}"',
                hint=f'使えるのは: {", ".join(PRESETS)}',
            )
        overrides.update(preset)
    return overrides


def _skill_list(registry: SkillRegistry, options: dict[str, Any]) -> Any:
    wanted = [
        ("animation", options.get("animations") is True or options.get("animation") is True),
        ("skill", options.get("skills") is True or options.get("skill") is True),
        ("scene", options.get("scenes") is True or options.get("scene") is True),
        ("movie", options.get("movies") is True or options.get("movie") is True),
    ]
    picked = [kind for kind, on in wanted if on]
    # 絞り込みが無いときは «全種類»。1 つでも指定されたらそれだけを出す。
    kinds = picked or ["animation", "skill", "scene", "movie"]

    def keep(entry) -> bool:
        if isinstance(options.get("category"), str) and entry["category"] != options["category"]:
            return False
        if isinstance(options.get("tag"), str) and options["tag"] not in entry["tags"]:
            return False
        if isinstance(options.get("grep"), str):
            needle = options["grep"].lower()
            hay = f'{entry["name"]} {entry["title"]} {entry["description"]} {" ".join(entry["tags"])}'.lower()
            if needle not in hay:
                return False
        return True

    collected = {kind: [e for e in registry.list(kind) if keep(e)] for kind in kinds}

    if options.get("json"):
        payload = {}
        for kind in kinds:
            payload[f"{kind}s"] = [
                {
                    "name": entry["name"],
                    "title": entry["title"],
                    "description": entry["description"],
                    "category": entry["category"],
                    "tags": entry["tags"],
                    "inputs": {
                        key: {
                            "type": definition.get("type", "text"),
                            "label": definition.get("label"),
                            "default": definition.get("default"),
                            "required": bool(definition.get("required")),
                        }
                        for key, definition in entry["inputs"].items()
                    },
                    "source": entry["source"],
                }
                for entry in collected[kind]
            ]
        say(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return payload

    for kind in kinds:
        entries = collected[kind]
        heading = KIND_LABELS.get(kind, kind)
        order = ANIMATION_CATEGORIES if kind == "animation" else SKILL_CATEGORIES
        logger.info("")
        logger.info(style.bold(f"{heading} {len(entries)} 件"))
        if not entries:
            logger.info("  該当なし")
            continue
        groups: dict[str, list] = {}
        for entry in entries:
            groups.setdefault(entry["category"], []).append(entry)
        categories = sorted(groups, key=lambda c: (order.index(c) if c in order else 99, c))
        for category in categories:
            logger.info(f'  {style.dim(f"[{category}]")}')
            for entry in groups[category]:
                mark = style.dim(" *") if entry["source"] == "project" else ""
                logger.info(f'    {entry["name"].ljust(18)} {entry["description"] or entry["title"]}{mark}')
    logger.info("")
    logger.info(style.dim("  * はプロジェクト側の定義（animations/ skills/ scenes/ movies/ で上書き可能）"))
    logger.info("  詳細: movo skill show <名前>   描画: movo skill render <スキル名> --set key=値")
    return collected


def _describe_input(key: str, definition: dict) -> str:
    kind = definition.get("type", "text")
    bits = [kind]
    if definition.get("required"):
        bits.append("必須")
    if definition.get("options"):
        bits.append("|".join(map(str, definition["options"])))
    if definition.get("min") is not None or definition.get("max") is not None:
        bits.append(f'{definition.get("min", "-∞")}..{definition.get("max", "∞")}')
    if "default" in definition:
        fallback = " 既定=" + json.dumps(definition["default"], ensure_ascii=False)
    else:
        fallback = ""
    # f 文字列の中で同じ引用符を入れ子にすると Python 3.11 では構文エラーになるので、
    # 組み立てを先に済ませてから差し込みます。
    kinds = style.dim("(" + ", ".join(bits) + ")")
    label = definition.get("label", "")
    return f"    {key.ljust(16)} {kinds}{fallback}\n      {label}"


def _skill_show(registry: SkillRegistry, name: str | None, options: dict[str, Any]) -> Any:
    if not name:
        raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "名前が必要です", hint="movo skill show cutin-title")
    entry = registry.find(name)
    if entry is None:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'"{name}" というスキル／基礎アニメーションはありません',
            hint="一覧は movo skill list",
        )
    if options.get("json"):
        say(json.dumps(entry["definition"], ensure_ascii=False, indent=2))
        return entry

    kind = KIND_LABELS.get(entry["kind"], entry["kind"])
    definition = entry["definition"]
    logger.info("")
    heading = style.dim(f'{kind} / {entry["category"]} / v{entry["version"]}')
    logger.info(f'{style.bold(entry["name"])}  {heading}')
    logger.info(f'  {entry["title"]}')
    if entry["description"]:
        logger.info(f'  {entry["description"]}')
    if entry["tags"]:
        logger.info(f'  タグ: {", ".join(entry["tags"])}')
    if entry["file"]:
        logger.info(f'  定義: {entry["file"]}')
    if entry.get("learnedFrom"):
        logger.info("  学習元:")
        for url in entry["learnedFrom"]:
            logger.info(f"    {url}")

    inputs = list(entry["inputs"].items())
    logger.info("")
    logger.info(f'  {style.bold("入力値")} {len(inputs)} 件')
    if not inputs:
        logger.info("    （なし）")
    # 「宣言されているのに本文から参照されていない」入力は、使う人からは
    # «変えても何も起きない壊れた項目» に見えます。ここで印を付けておきます。
    dead = set(find_dead_inputs(registry, entry["name"]))
    for key, definition_entry in inputs:
        logger.info(_describe_input(key, definition_entry))
        if key in dead:
            logger.warn("      ↑ この入力はどこからも参照されていません（変えても結果は変わりません）")

    logger.info("")
    if entry["kind"] == "movie":
        sequence = definition.get("sequence") or []
        logger.info(f'  {style.bold("シーンの並び")} {len(sequence)} 件（repeat 展開前）')
        for item in sequence:
            bars = (item.get("with") or {}).get("bars")
            note = style.dim(f"{bars} 小節") if bars is not None else ""
            logger.info(f'    {str(item.get("scene", "?")).ljust(18)} {note}')
        logger.info("")
        logger.info(f'  試す: movo skill render {entry["name"]} --set title=タイトル -o tmp/{entry["name"]}.mp4')
    elif entry["kind"] == "scene":
        layers = (definition.get("scene") or {}).get("layers") or []
        logger.info(f'  {style.bold("生成するレイヤー")} {len(layers)} 件（repeat 展開前）')
        for layer in layers:
            logger.info(_layer_line(layer))
        logger.info("")
        duration = (definition.get("scene") or {}).get("duration", "4bar")
        logger.info(f"  尺: {json.dumps(duration, ensure_ascii=False)}（project.bpm から秒になります）")
        logger.info("  使い方（scenes に書く）:")
        logger.info(f'    {{ "use": "{entry["name"]}", "with": {{ "bars": 4 }} }}')
    elif entry["kind"] == "skill":
        layers = definition.get("layers") or []
        logger.info(f'  {style.bold("生成するレイヤー")} {len(layers)} 件（repeat 展開前）')
        for layer in layers:
            logger.info(_layer_line(layer))
        logger.info("")
        example = " ".join(
            "--set " + key + "=" + json.dumps(d.get("default", "値"), ensure_ascii=False).replace('"', "")
            for key, d in inputs[:3]
            if d.get("required") or "default" in d
        )
        logger.info(f'  試す: movo skill render {entry["name"]} {example} -o tmp/{entry["name"]}.mp4')
    else:
        produces = definition.get("produces") or {}
        summary = ", ".join(f"{k}×{len(v) if isinstance(v, list) else 1}" for k, v in produces.items())
        logger.info(f'  {style.bold("レイヤーに合成するもの")}: {summary or "なし"}')
        logger.info("")
        logger.info("  使い方（レイヤーに書く）:")
        extra = f', "with": {{ "{inputs[0][0]}": ... }}' if inputs else ""
        logger.info(f'    "use": [{{ "animation": "{entry["name"]}"{extra} }}]')
    logger.info("")
    return entry


def _to_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _layer_line(layer: dict) -> str:
    """`movo skill show` の «生成するレイヤー» 1 行分。"""
    uses = [u if isinstance(u, str) else u.get("animation") for u in _to_list(layer.get("use"))]
    note = style.dim("use: " + ", ".join(map(str, uses))) if uses else ""
    return f'    {str(layer.get("type", "?")).ljust(12)} {note}'


def _build_for(registry: SkillRegistry, name: str | None, options: dict[str, Any]) -> dict:
    if not name:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE, "スキル名が必要です", hint="movo skill render cutin-title --set text=タイトル"
        )
    return build_skill_project(registry, name, _inputs_from(options), _overrides_from(options))


def _skill_expand(registry: SkillRegistry, name: str | None, options: dict[str, Any]) -> dict:
    built = _build_for(registry, name, options)
    text = json.dumps(built["project"], ensure_ascii=False, indent=2) + "\n"
    if options.get("output"):
        target = Path(options["output"]).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        logger.success(f'スキル "{name}" を展開しました → {target}')
        logger.info(f"  そのまま描画: movo render {target}")
        return {"path": str(target), "project": built["project"]}
    say(text.rstrip("\n"))
    return {"project": built["project"]}


def _skill_render(registry: SkillRegistry, name: str | None, options: dict[str, Any]) -> dict:
    built = _build_for(registry, name, options)
    project = built["project"]
    entry = built["entry"]
    logger.step(
        f'スキル {style.bold(entry["name"])} を描画します  '
        f'{project["video"]["width"]}x{project["video"]["height"]} @ {project["video"]["fps"]}fps  '
        f'{project["video"]["duration"]:.2f}s'
    )

    session = create_session(
        f'{entry["name"]}.json',
        {
            "inline_project": project,
            "project_root": os.getcwd(),
            "skill_registry": registry,
            "quality": options.get("quality"),
            "renderer": options.get("renderer"),
            "super_sample": options.get("superSample"),
            "no_cache": options.get("noCache") is True or options.get("cache") is False,
            "generate_assets": options.get("generate") is not False,
            "strict_plugins": False,
            "dry_run_ai": options.get("dryRun") is True,
        },
    )

    output = options.get("output") or str(Path("tmp") / f'{entry["name"]}.{"gif" if options.get("format") == "gif" else "mp4"}')
    result = render_video(
        session,
        {
            "output": output,
            "format": options.get("format"),
            "from": options.get("from") if isinstance(options.get("from"), (int, float)) else None,
            "to": options.get("to") if isinstance(options.get("to"), (int, float)) else None,
            "quiet": options.get("quiet"),
        },
    )
    logger.info(
        f'  {result["frames"]} フレーム / {result["elapsedSeconds"]:.1f} 秒 '
        f'({result["fpsAchieved"]:.1f} fps) / 形式 {result["format"]}'
    )
    if options.get("json"):
        say(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


SKILL_TEMPLATE = {
    "skill": {
        "name": "__NAME__",
        "title": "新しいスキル",
        "description": "ここに説明を書きます",
        "category": "title",
        "tags": [],
    },
    "inputs": {
        "text": {"type": "text", "label": "表示する文字", "default": "テキスト"},
        "color": {"type": "color", "label": "文字色", "default": "#ffffff"},
        "duration": {"type": "number", "label": "長さ（秒）", "default": 3, "min": 0.2, "max": 60},
    },
    "duration": "${duration}",
    "video": {"width": 1920, "height": 1080, "background": "#101018"},
    "layers": [
        {
            "id": "${_id}-text",
            "type": "text",
            "text": "${text}",
            "style": {"size": 96, "color": "${color}", "align": "center", "weight": "bold"},
            "transform": {"x": "${centerX}", "y": "${centerY}", "anchorX": 0.5, "anchorY": 0.5},
            "use": [{"animation": "pop-in", "with": {"duration": 0.4}}],
        }
    ],
}

SCENE_TEMPLATE = {
    "skill": {
        "name": "__NAME__",
        "kind": "scene",
        "title": "新しいシーンスキル",
        "description": "ここに説明を書きます",
        "category": "title",
        "tags": [],
    },
    "inputs": {
        "title": {"type": "text", "label": "表示する文字", "default": "タイトル"},
        "bars": {"type": "number", "label": "尺（小節）", "default": 4, "min": 0.25, "max": 64},
        "color": {"type": "color", "label": "文字色", "default": "#ffffff"},
    },
    "video": {"width": 1920, "height": 1080},
    "scene": {
        # 小節で書いておくと、project.bpm を変えるだけで曲に合います
        "duration": "${bars}bar",
        "transition": {"type": "fade", "in": 0.4, "out": 0.4},
        "layers": [
            {
                "id": "${_id}-text",
                "type": "text",
                "text": "${title}",
                "style": {"size": 96, "color": "${color}", "align": "center", "weight": "bold"},
                "transform": {"x": "${centerX}", "y": "${centerY}", "anchorX": 0.5, "anchorY": 0.5},
                "use": [{"animation": "rise-in", "with": {"duration": 0.8}}],
            }
        ],
    },
}

MOVIE_TEMPLATE = {
    "skill": {
        "name": "__NAME__",
        "kind": "movie",
        "title": "新しいムービースキル",
        "description": "ここに説明を書きます",
        "category": "lyric",
        "tags": [],
    },
    "inputs": {
        "title": {"type": "text", "label": "曲名", "default": "無題"},
        "bpm": {"type": "number", "label": "BPM", "default": 150, "min": 40, "max": 300},
        "lines": {"type": "textList", "label": "歌詞（1 行ずつ）", "default": []},
    },
    "video": {"width": 1920, "height": 1080, "fps": 30, "background": "#0a0a12"},
    "project": {"bpm": 150},
    "sequence": [
        {"scene": "mv-intro", "with": {"title": "${title}", "bars": 4}},
        {"scene": "mv-verse", "with": {"lines": "${lines}", "bars": 8}},
        {"scene": "mv-chorus", "with": {"hook": "${lines[0]}", "bars": 8}},
        {"scene": "mv-outro", "with": {"title": "${title}", "bars": 4}},
    ],
}

ANIMATION_TEMPLATE = {
    "animation": {
        "name": "__NAME__",
        "title": "新しい基礎アニメーション",
        "description": "ここに説明を書きます",
        "category": "in",
        "tags": [],
    },
    "inputs": {
        "delay": {"type": "number", "label": "開始を遅らせる（秒）", "default": 0, "min": 0},
        "duration": {"type": "number", "label": "長さ（秒）", "default": 0.5, "min": 0.01},
    },
    "produces": {
        "animations": [
            {
                "property": "transform.opacity",
                "keyframes": [
                    {"time": "${delay}", "value": 0, "easing": "easeOut"},
                    {"time": "${delay + duration}", "value": 1},
                ],
            }
        ]
    },
}


def _skill_new(name: str | None, options: dict[str, Any]) -> dict:
    if not name:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "名前が必要です",
            hint="movo skill new my-skill / movo skill new my-move --animation",
        )
    # 種類ごとに «置き場» が決まっているので、フラグから 1 つ選ぶ
    if options.get("animation") is True or options.get("animations") is True:
        kind = "animation"
    elif options.get("scene") is True or options.get("scenes") is True:
        kind = "scene"
    elif options.get("movie") is True or options.get("movies") is True:
        kind = "movie"
    else:
        kind = "skill"
    layout = {
        "animation": ("animations", ANIMATION_TEMPLATE),
        "skill": ("skills", SKILL_TEMPLATE),
        "scene": ("scenes", SCENE_TEMPLATE),
        "movie": ("movies", MOVIE_TEMPLATE),
    }[kind]
    directory = Path(layout[0]).resolve()
    target = directory / f"{name}.json"
    if target.exists() and not options.get("force"):
        raise MovoError(ErrorCodes.MOVO_CLI_USAGE, f"{target} は既にあります", hint="上書きするなら --force")
    template = json.loads(json.dumps(layout[1], ensure_ascii=False).replace("__NAME__", name))
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.success(f"雛形を作成しました: {target}")
    if kind == "animation":
        logger.info(f'  レイヤーに "use": [{{ "animation": "{name}" }}] と書くと使えます。')
    elif kind == "scene":
        logger.info(f'  scenes に "use": "{name}" と書くと 1 シーンになります。')
        logger.info(f"  描画: movo skill render {name} -o tmp/{name}.mp4")
    else:
        logger.info(f"  確認: movo skill show {name}")
        logger.info(f"  描画: movo skill render {name} -o tmp/{name}.mp4")
    return {"path": str(target)}
