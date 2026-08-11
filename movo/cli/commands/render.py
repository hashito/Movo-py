"""`movo render` / `movo frame` / `movo frames`。"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .. import bridge
from ..args import param_options_from
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..parallel import render_video_parallel, resolve_job_count
from ..pipeline import (
    create_session,
    read_project_file,
    render_video,
    resolve_output_path,
    resolve_range,
    write_lock_file,
)


def _session_options_from(options: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
    return {
        # «素材だけ差し替えて作り直す»: --set / --params / --save-recipe を渡す
        **param_options_from(options),
        # アスペクト比バリアント。畳むのは prepare_project なので、ここでは
        # «どれを選んだか» を渡すだけです。
        "variant": variant if variant is not None else _variant_from(options),
        "quality": options.get("quality"),
        "renderer": options.get("renderer"),
        "super_sample": options.get("superSample"),
        "seed": options.get("seed"),
        "no_cache": options.get("noCache") is True or options.get("cache") is False,
        "generate_assets": options.get("generate") is not False,
        "strict_plugins": options.get("strict") is not False,
        "dry_run_ai": options.get("dryRun") is True,
    }


def _variant_from(options: dict[str, Any]) -> str | None:
    value = options.get("variant")
    return value if isinstance(value, str) and value != "" else None


def _wants_all_variants(options: dict[str, Any]) -> bool:
    """`--all-variants` が指定されたか。

    このキーは args.py の «真偽値» 表に載せてありますが、`--all-variants
    movo.json` と書かれた場合に備えて «飲み込んだファイル名» も拾い直します。
    """
    return options.get("allVariants") not in (None, False)


def _all_variants_file(options: dict[str, Any]) -> str | None:
    value = options.get("allVariants")
    return value if isinstance(value, str) else None


def _fill_output_template(template: str, values: dict[str, str]) -> str:
    """`-o "tmp/{name}-{variant}.mp4"` の差し替え。"""
    out = str(template)
    for key in ("name", "variant"):
        out = out.replace("{" + key + "}", str(values.get(key, "{" + key + "}")))
    return out


def _variant_output_path(session, template: str | None, name: str, variant: str, output_format: str | None) -> str:
    """バリアントごとの書き出し先。

    `-o` に `{variant}` を書いてあればそれに従います。書いていなければ、既定の
    書き出し先の «拡張子の手前» に名前を挟みます。同じ名前に 3 回書き込んで
    «最後の 1 本しか残らない» のを防ぐためです。
    """
    if isinstance(template, str) and template != "":
        return _fill_output_template(template, {"name": name, "variant": variant})
    declared = session["project"].get("output") or {}
    if isinstance(declared, list):
        declared = declared[0] if declared else {}
    base = resolve_output_path(
        project_output=declared.get("path"),
        project_root=session["projectRoot"],
        name=name,
        format=output_format or declared.get("format") or "mp4",
    )
    path = Path(base)
    return str(path.with_name(f"{path.stem}-{variant}{path.suffix}"))


def _render_options_from(options: dict[str, Any], output: str | None) -> dict[str, Any]:
    """`render_video` に渡す «この 1 本をどう描くか» の指定。

    単発と並列で同じものを渡したいので 1 か所にまとめています。並列のときは
    ここに `jobs` と `cli_options`（子プロセスへ引き継ぐ指定）が足されます。
    """
    return {
        "output": output,
        "format": options.get("format"),
        "from": options.get("from") if isinstance(options.get("from"), (int, float)) else None,
        "to": options.get("to") if isinstance(options.get("to"), (int, float)) else None,
        "scene": options.get("scene") if isinstance(options.get("scene"), str) else None,
        "quiet": options.get("quiet"),
        # `--no-audio`: 音を動画に入れない（音に反応するアニメーションはそのまま）
        "audio": options.get("audio"),
        # `--no-check-flash`: 光過敏性発作の検査をこの 1 回だけ止める
        "check_flash": options.get("checkFlash"),
        # `--warmup`: 区間の頭で残像の履歴を作り直すための助走フレーム数。
        # 並列レンダリングの親が子に渡すためのもので、普段は書きません。
        "warmup": options.get("warmup"),
    }


def render_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    file = (positional[0] if positional else None) or _all_variants_file(options) or "movo.json"
    if _wants_all_variants(options):
        return _render_all_variants(file, options)

    variant = _variant_from(options)
    session = create_session(file, _session_options_from(options))
    _report_session(session, variant)

    # **`--jobs` は «区間に割って同時に描く»。** 使えないと分かったときは
    # render_video_parallel が理由を言って 1 本に落とすので、ここでは分岐だけです。
    jobs = resolve_job_count(options.get("jobs"))
    render = render_video_parallel if jobs > 1 else render_video

    output = options.get("output")
    if variant:
        # `--variant` を付けたときも名前を分けます。素の 1 本と同じ名前に書くと、
        # 16:9 を出したあと 9:16 を出した時点で前の 1 本が消えるためです。
        output = _variant_output_path(
            session,
            options.get("output"),
            (session["project"].get("project") or {}).get("name") or "output",
            variant,
            options.get("format"),
        )

    result = render(
        session,
        {
            **_render_options_from(options, output),
            "jobs": jobs,
            # 子プロセスに «同じ絵になる指定» を引き継ぐために、素の指定ごと渡します。
            "cli_options": options,
            "keep_parts": options.get("keepParts") is True,
        },
    )

    if options.get("lock"):
        lock_path = write_lock_file(session)
        logger.info(f"  ロックファイル: {os.path.relpath(lock_path, os.getcwd())}")

    logger.info(
        f'  {result["frames"]} フレーム / {result["elapsedSeconds"]:.1f} 秒 '
        f'({result["fpsAchieved"]:.1f} fps) / 形式 {result["format"]}'
        + (f' / {result["jobs"]} 並列 {result["chunks"]} 区間' if result.get("parallel") else "")
    )
    if options.get("json"):
        say(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def _render_all_variants(file: str, options: dict[str, Any]) -> dict[str, Any]:
    """宣言されているバリアントを順に書き出す（`--all-variants`）。

    1 本ずつ `create_session` からやり直します。解像度が変わればレイアウトも
    素材の読み込みサイズも変わるので、途中から作り直す近道はありません。
    その代わり «同じ JSON から同じ動画» の性質はバリアントごとに保たれます。
    """
    read = read_project_file(file)
    apply_extends = bridge.pick("movo.schema", "apply_extends", "applyExtends")
    variant_names = bridge.pick("movo.schema", "variant_names", "variantNames")
    inherited = apply_extends(copy.deepcopy(read["raw"]), file=file)
    declared = set((inherited.get("variants") or {}).keys())
    names = variant_names(inherited)
    if len(names) <= 1:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            f'{Path(file).name} に variants が宣言されていません',
            hint='"variants": { "shorts": { "video": { "width": 1080, "height": 1920 } } } のように書きます',
        )

    logger.step(f'バリアント {len(names)} 本: {" / ".join(names)}')
    # バリアントは «1 本ずつ» 出しますが、その 1 本の中は `--jobs` で割れます。
    # 3 本を同時に走らせないのは、解像度が違うと素材の読み込みサイズも変わり、
    # 3 本ぶんの素材を同時に抱えることになるためです。
    jobs = resolve_job_count(options.get("jobs"))
    render = render_video_parallel if jobs > 1 else render_video
    results = []
    for variant in names:
        # 素のまま（base）は «バリアントを選ばない» ことなので None を渡します。
        selected = variant if variant in declared else None
        session = create_session(file, _session_options_from(options, selected))
        _report_session(session, variant)
        result = render(
            session,
            {
                **_render_options_from(
                    options,
                    _variant_output_path(
                        session,
                        options.get("output"),
                        (session["project"].get("project") or {}).get("name") or "output",
                        variant,
                        options.get("format"),
                    ),
                ),
                "jobs": jobs,
                # 子には «このバリアント» を名指しで渡します（--all-variants は
                # 渡しません。子まで全バリアントを描き始めてしまうためです）。
                "cli_options": {**options, "variant": selected, "allVariants": None},
                "keep_parts": options.get("keepParts") is True,
            },
        )
        logger.info(
            f'  {variant}: {result["frames"]} フレーム / {result["elapsedSeconds"]:.1f} 秒 / {result["path"]}'
        )
        results.append({**result, "variant": variant})
    if options.get("json"):
        say(json.dumps({"variants": results}, ensure_ascii=False, indent=2, default=str))
    return {"variants": results}


def frame_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    file = (positional[0] if positional else None) or "movo.json"
    session = create_session(file, _session_options_from(options))
    timeline = session["timeline"]

    if isinstance(options.get("frame"), (int, float)):
        frame_index = round(options["frame"])
    else:
        time_value = options.get("time") if isinstance(options.get("time"), (int, float)) else 0
        frame_index = round(time_value * timeline["fps"])
    if frame_index < 0 or frame_index >= timeline["frameCount"]:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            f'frame {frame_index} は範囲外です（0..{timeline["frameCount"] - 1}、尺 {timeline["duration"]} 秒）',
        )

    _report_session(session)
    bitmap = session["renderer"].render_frame(frame_index)
    if options.get("output"):
        output_path = Path(options["output"]).resolve()
    else:
        output_path = Path(session["projectRoot"]) / "output" / f"frame-{frame_index:05d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(bridge.encode_png(bitmap)))
    logger.success(
        f'フレーム {frame_index}（{frame_index / timeline["fps"]:.3f} 秒）→ {os.path.relpath(output_path, os.getcwd())}'
    )
    result = {"frame": frame_index, "path": str(output_path)}
    if options.get("json"):
        say(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def frames_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    file = (positional[0] if positional else None) or "movo.json"
    session = create_session(file, _session_options_from(options))
    _report_session(session)
    rng = resolve_range(
        session["timeline"],
        {
            "from": options.get("from") if isinstance(options.get("from"), (int, float)) else None,
            "to": options.get("to") if isinstance(options.get("to"), (int, float)) else None,
            "scene": options.get("scene"),
        },
    )
    directory = (
        str(Path(options["output"]).resolve())
        if options.get("output")
        else str(Path(session["projectRoot"]) / "output" / "frames")
    )
    result = render_video(
        session,
        {
            "output": directory,
            "format": "png-sequence",
            "from": rng.get("startTime"),
            "to": rng.get("endTime"),
            "scene": options.get("scene"),
            "quiet": options.get("quiet"),
        },
    )
    if options.get("json"):
        say(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def _report_session(session, variant: str | None = None) -> None:
    timeline = session["timeline"]
    project = session["project"]
    renderer = session["renderer"]
    assets = session.get("assets")
    name = (project.get("project") or {}).get("name", "")
    label = f"{style.bold(name)} [{variant}]" if variant else style.bold(name)
    logger.step(
        f'{label}  {timeline["width"]}x{timeline["height"]} @ {timeline["fps"]}fps  '
        f'{timeline["duration"]:.2f}s  品質 {project["render"]["quality"]}'
    )
    details = []
    stats = assets.stats() if assets is not None and hasattr(assets, "stats") else {}
    if stats:
        details.append(f'素材 {stats.get("images", 0) + stats.get("audio", 0)}')
        if stats.get("placeholders"):
            details.append(f'プレースホルダ {stats["placeholders"]}')
    for attribute, label_text in (("bodies", "物理ボディ"), ("particles", "パーティクル"), ("soft_chains", "ソフトチェーン")):
        holder = getattr(renderer, attribute, None)
        if holder:
            details.append(f"{label_text} {len(holder)}")
    if getattr(renderer, "render_scale", 1) > 1:
        details.append(f"スーパーサンプリング {renderer.render_scale}x")
    if details:
        logger.verbose("  " + " / ".join(details))
    for error in session.get("assetErrors") or []:
        reason = error.get("reason") if isinstance(error, dict) else str(error)
        logger.verbose(f"  素材の警告: {reason}")
