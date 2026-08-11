"""すべての CLI コマンドが共有するレンダリングの筋道。

JSON → スキーマ検証 → プラグイン読み込み → 素材の解決 → タイムライン展開 →
音のミックス → フレーム描画 → 書き出し。

**このファイルは «繋ぎ» です。** 実際の検証・描画・書き出しは `movo.schema` /
`movo.renderer` / `movo.exporters` が持っていて、それらは別の担当が移植中です。
未接続のものを使おうとすると `movo.cli.bridge.NotConnectedError` が
«〈どのモジュール〉が未接続です — その部分は後で繋ぐ» と名指しで止めます。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from movo import __version__

from . import bridge
from .console import logger
from .errors import ErrorCodes, MovoError, MovoValidationError


def strip_json_comments(text: str) -> str:
    """`//` と `/* */` を許す。プロジェクト JSON に注釈を書けるようにするため。"""
    from movo.skill import strip_json_comments as _strip

    return _strip(text)


def read_project_file(file: str | os.PathLike[str]) -> dict[str, Any]:
    """プロジェクトファイルを読んで解釈する。"""
    absolute = Path(file).resolve()
    if not absolute.exists():
        raise MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND,
            f"project file not found: {file}",
            hint='新しく作るなら "movo init <名前>"',
        )
    try:
        text = absolute.read_text(encoding="utf-8")
    except OSError as error:
        raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"cannot read {file}: {error}", cause=error) from error
    try:
        parsed = json.loads(strip_json_comments(text))
    except ValueError as error:
        raise MovoError(ErrorCodes.MOVO_SCHEMA_INVALID, f"invalid JSON: {error}", file=str(absolute)) from error

    declared_root = (parsed.get("project") or {}).get("root") if isinstance(parsed, dict) else None
    project_root = (absolute.parent / declared_root).resolve() if declared_root else absolute.parent
    return {"raw": parsed, "file": str(absolute), "projectRoot": str(project_root)}


def expand_skills(raw: dict, project_root: str, file: str | None = None, registry=None) -> dict:
    """スキルと基礎アニメーションを展開する。

    プロジェクト JSON の `use`（スキル）と `layer.use`（基礎アニメーション）を
    解決して、素の Movo JSON に落とします。検証と正規化はこの後に走ります。
    """
    from movo.skill import SkillRegistry, expand_project_skills

    registry = registry or SkillRegistry().load(project_root=project_root)
    result = expand_project_skills(raw, registry, file=file)
    if result["used"]:
        summary = ", ".join(f'{u["skill"]}({u["layers"]} レイヤー)' for u in result["used"])
        logger.verbose(f"スキルを展開しました: {summary}")
    return {"project": result["project"], "used": result["used"], "registry": registry}


def validate(raw: dict, file: str | None = None, throw_on_error: bool = True, registry=None) -> dict:
    """検証。プラグインが登録した名前も「既知」として扱う。"""
    validate_project = bridge.pick("movo.schema", "validate_project", "validateProject")
    known = {
        "known_deformers": set(bridge.listing("movo.deformer", "list_deformers", "listDeformers")),
        "known_effects": set(bridge.listing("movo.renderer.effects", "list_effects", "listEffects")),
        "known_modulators": set(bridge.listing("movo.animation.modulators", "list_modulators", "listModulators")),
    }
    result = validate_project(raw, file=file, **known)
    if not result.get("valid", True) and throw_on_error:
        raise MovoValidationError(result.get("issues", []), file)
    return result


class Session(dict):
    """1 回のレンダリングに要るものをひとまとめにしたもの。

    `dict` にしてあるのは `--json` でそのまま出せるようにするためではなく、
    **属性でも添字でも読めるようにして、移植の途中で呼び方が揺れても
    壊れないようにする**ためです（JS 版はオブジェクトの分割代入で受けています）。
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def create_session(file: str | None, options: dict[str, Any] | None = None) -> Session:
    """描くために要るものを全部そろえる。

    `inline_project` を渡すとファイルを読まずにその JSON を使います
    （`movo skill render` のように、スキルから組み立てた JSON を直接描く場合）。
    """
    options = options or {}

    # **インラインのときは «元になった JSON» を覚えておきます。** `file` には
    # 呼び出し元が渡した名前（`make-mv` では出力先の .mp4）がそのまま入るので、
    # `file` を見て «プロジェクトファイルがある» と判断すると、並列レンダリングの
    # 子プロセスが .mp4 を JSON として読みに行きます（実際にそうなり、
    # `'utf-8' codec can't decode byte 0xaf` という無関係な顔で落ちました）。
    inline_project = options.get("inline_project")
    if inline_project is not None:
        read = {
            "raw": inline_project,
            "file": str(Path(file or "skill.json").resolve()),
            "projectRoot": str(Path(options.get("project_root") or os.getcwd()).resolve()),
        }
    else:
        read = read_project_file(file or "movo.json")
    raw_source, absolute, project_root = read["raw"], read["file"], read["projectRoot"]

    # «素材だけ差し替えて同じ動画を作り直す»。継承と params をここで畳んでおけば
    # 以降は素の Movo JSON として扱える。--save-recipe の «作り方» もここで書き出す。
    prepare_project = bridge.pick("movo.schema.params", "prepare_project", "prepareProject")
    source = prepare_project(raw_source, file=absolute, **_param_kwargs(options))

    # スキルを先に展開して、素の JSON にしてから検証・正規化する
    skills = expand_skills(source, project_root, absolute, options.get("skill_registry"))
    raw = skills["project"]

    # project.bpm.fromAudio。"4bar" の解決が project.bpm を数値として読むので、
    # 検証・正規化より «前» に潰しておく必要がある。
    # 設定は **JS 版と同じ綴りの辞書** で渡します（`projectRoot` など）。
    resolve_bpm = bridge.pick("movo.audio", "resolve_bpm_from_audio", "resolveBpmFromAudio")
    if not getattr(resolve_bpm, "movo_not_connected", False):
        resolve_bpm(raw, None, {"projectRoot": project_root})

    normalize_project = bridge.pick("movo.schema", "normalize_project", "normalizeProject")
    validation = validate(raw, file=absolute, throw_on_error=True)
    for warning in validation.get("warnings", []):
        logger.warn(f'{warning.get("path")}: {warning.get("message")}')

    project = normalize_project(raw)
    if options.get("quality"):
        project["render"]["quality"] = options["quality"]
        renormalised = normalize_project({**raw, "render": {**(raw.get("render") or {}), "quality": options["quality"]}})
        project["render"] = renormalised["render"]
    if options.get("renderer"):
        project["render"]["renderer"] = options["renderer"]
    if options.get("super_sample"):
        project["render"]["superSample"] = options["super_sample"]
    if options.get("seed") is not None:
        project["project"]["seed"] = options["seed"]
        project.setdefault("deterministic", {})["seed"] = options["seed"]

    cache_directory = (project.get("cache") or {}).get("directory")
    cache_root = str(Path(project_root) / (cache_directory or "cache"))
    cache = bridge.pick("movo.core.cache", "Cache")(
        cache_root,
        enabled=False if options.get("no_cache") else (project.get("cache") or {}).get("enabled") is not False,
        namespace_salt={"movo": __version__, "quality": project["render"]["quality"]},
    )

    assets = bridge.pick("movo.core.assets", "AssetStore")(
        project_root=project_root,
        assets=project.get("assets") or {},
        cache=cache,
        security=project.get("security") or {},
        seed=project["project"].get("seed"),
    )
    asset_errors = assets.resolve_all(strict=False, generate=options.get("generate_assets") is not False)

    build_timeline = bridge.pick("movo.timeline", "build_timeline", "buildTimeline")
    timeline = build_timeline(project)

    audio = None
    envelope = None
    mix_audio = bridge.pick("movo.audio", "mix_project_audio", "mixProjectAudio")
    if not getattr(mix_audio, "movo_not_connected", False):
        mixed = mix_audio(
            project,
            assets,
            {
                "duration": timeline["duration"],
                "fps": timeline["fps"],
                "seed": project["project"].get("seed"),
            },
        )
        if mixed:
            audio = mixed["audio"]
            analyze_envelope = bridge.pick("movo.audio", "analyze_envelope", "analyzeEnvelope")
            envelope = analyze_envelope(audio, timeline["fps"], timeline["frameCount"])
            logger.verbose(f'mixed {mixed.get("tracks")} audio track(s)')

    renderer = _build_renderer(
        {
            "project": project,
            "timeline": timeline,
            "assets": assets,
            "cache": cache,
            "project_root": project_root,
            "audio_envelope": envelope,
            "audio": audio,
        }
    )

    return Session(
        file=absolute,
        # ファイルから読んだのなら None。インラインならその元の JSON。
        # 並列レンダリングの子はこれを受け取って «親とまったく同じ» 組み立てを
        # やり直します（ファイルが無くても割れるようにするため）。
        inlineProject=inline_project,
        projectRoot=project_root,
        raw=raw,
        source=source,
        skillRegistry=skills["registry"],
        skillsUsed=skills["used"],
        project=project,
        timeline=timeline,
        cache=cache,
        assets=assets,
        assetErrors=asset_errors or [],
        renderer=renderer,
        audio=audio,
        envelope=envelope,
        validation=validation,
    )


def _build_renderer(arguments: dict[str, Any]):
    """レンダラーを組み立てる。

    **レンダラーは別の担当が移植中です。** 受け取り方が «キーワード引数» に
    なるか «設定の辞書 1 個» になるかまだ決まっていないので、両方を試します。
    決まったらここは 1 行に縮められます — **その部分は後で繋ぐ**。

    受け取れない引数があった場合も、名前を捨てずに «その引数は要らなかった»
    として落とし直します。黙って `None` を渡すと、音に反応する動きが
    «エラーも出ないのに効かない» という形で壊れるためです。
    """
    renderer_class = bridge.pick("movo.renderer", "Renderer")
    try:
        renderer = renderer_class(**arguments)
    except TypeError as error:
        if getattr(renderer_class, "movo_not_connected", False):
            raise
        logger.verbose(f"レンダラーの受け取り方が違うようなので、設定の辞書で渡し直します（{error}）")
        renderer = renderer_class(arguments)
    prepare = getattr(renderer, "prepare", None)
    if callable(prepare):
        renderer = prepare() or renderer
    return renderer


def _param_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    """`prepare_project` に渡す params 関連の指定だけを取り出す。"""
    # **キー名は `prepare_project` の引数名に合わせます。**
    # `set` は Python の組み込み関数と同じ綴りなので、schema 側は `set_values`、
    # `format` も組み込みと紛れるので `output_format` という名前になっています。
    # ここを揃え忘れて `--set` がまるごと効かない状態になっていました。
    return {
        "set_values": options.get("set"),
        "params": options.get("params"),
        "save_recipe": options.get("save_recipe"),
        "output": options.get("output"),
        "output_format": options.get("format"),
        "variant": options.get("variant"),
    }


def resolve_range(timeline: Any, options: dict[str, Any]) -> dict[str, Any]:
    """描く範囲（秒 → フレーム番号）を決める。timeline 側の実装に任せます。"""
    function = bridge.pick("movo.timeline", "resolve_range", "resolveRange")
    return function(
        timeline,
        from_=options.get("from"),
        to=options.get("to"),
        scene=options.get("scene"),
    )


def create_flash_guard(project: dict, timeline: Any, options: dict[str, Any] | None = None):
    """光過敏性発作（PSE）の検査係を作る。作らない（＝検査しない）なら None。

    単発でも並列でも同じ設定で測りたいので、作る場所を 1 か所にまとめています。
    `render.flashGuard.enabled: false`（作者が切った）か `--no-check-flash`
    （その 1 回だけ切った）のときだけ None になります。
    """
    options = options or {}
    settings = (project.get("render") or {}).get("flashGuard") or {}
    if settings.get("enabled") is False or options.get("check_flash") is False:
        return None
    guard_class = bridge.pick("movo.core.flash_guard", "FlashGuard")
    if getattr(guard_class, "movo_not_connected", False):
        return None
    # 閾値の名前は **プロジェクト JSON と同じ綴り** で渡します（FlashGuard 側が
    # その名前で受けます）。None は «指定なし» として捨てられるので、
    # `flashGuard` を書いていないプロジェクトでも既定値がそのまま効きます。
    return guard_class(
        width=timeline["width"],
        height=timeline["height"],
        fps=timeline["fps"],
        maxPerSecond=settings.get("maxPerSecond"),
        luminanceDelta=settings.get("luminanceDelta"),
        areaRatio=settings.get("areaRatio"),
    )


def resolve_output_path(
    explicit: str | None = None,
    project_output: str | None = None,
    project_root: str = ".",
    name: str = "output",
    format: str = "mp4",
    suffix: str = "",
) -> str:
    if explicit:
        return str(Path(explicit).resolve())
    if project_output:
        return str((Path(project_root) / project_output).resolve())
    default_extension = bridge.pick("movo.exporters", "default_extension_for", "defaultExtensionFor")
    extension = ".mp4"
    if not getattr(default_extension, "movo_not_connected", False):
        extension = default_extension(format)
    elif format == "webm":
        extension = ".webm"
    elif format == "gif":
        extension = ".gif"
    elif format == "wav":
        extension = ".wav"
    # suffix は «1 回描いて何通りも書き出す» ときに、2 本目以降の名前が
    # ぶつからないようにするためのものです（output/name-gif.gif など）。
    if format == "png-sequence":
        base = Path("output") / f"frames{suffix}"
    else:
        base = Path("output") / f"{name}{suffix}{extension}"
    return str((Path(project_root) / base).resolve())


def render_video(session: Session, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """範囲を決めてフレームを描き、ファイルに書き出す。"""
    options = options or {}
    project = session["project"]
    timeline = session["timeline"]
    renderer = session["renderer"]
    project_root = session["projectRoot"]
    rng = resolve_range(timeline, options)

    # `output` は 1 つでも配列でも書けます。**1 回描いて何通りも書き出す** ためです。
    # 1 本ごとにプロセスを起こしていた頃は、mp4 と gif が欲しいときに «同じ絵を
    # 2 回描いて» いました。描画はいちばん重い工程なので、そこを 1 回にまとめる
    # 意味は大きいです。
    #
    # コマンドラインで `-o` や `-f` を指定したときは、**先頭の 1 本だけ**を
    # それで置き換えます（「この 1 本が欲しい」という指定だと読むのが自然なので）。
    declared = project.get("output")
    requests = declared if isinstance(declared, list) else [declared or {}]
    if not requests:
        requests = [{}]

    negotiate_format = bridge.pick("movo.exporters", "negotiate_format", "negotiateFormat")
    targets = []
    for index, declaration in enumerate(requests):
        requested = (options.get("format") if index == 0 else None) or declaration.get("format") or "mp4"
        negotiated = negotiate_format(requested)
        if negotiated.get("downgraded"):
            logger.warn(negotiated.get("reason", ""))
        targets.append(
            {
                "declaration": declaration,
                "format": negotiated["format"],
                "path": resolve_output_path(
                    explicit=options.get("output") if index == 0 else None,
                    project_output=declaration.get("path"),
                    project_root=project_root,
                    name=(project.get("project") or {}).get("name") or "output",
                    format=negotiated["format"],
                    # 2 本目以降で名前がぶつからないよう、形式を名前に混ぜます
                    suffix=f'-{negotiated["format"]}' if index > 0 and not declaration.get("path") else "",
                ),
            }
        )
    # 以降、1 本目を «代表» として扱います（戻り値やログの互換のため）
    output_format = targets[0]["format"]
    output_path = targets[0]["path"]

    audio_path = _write_audio_sidecar(session, options, output_format, output_path)

    create_exporter = bridge.pick("movo.exporters", "create_exporter", "createExporter")
    exporters = [
        create_exporter(
            target["format"],
            width=timeline["width"],
            height=timeline["height"],
            fps=timeline["fps"],
            output_path=target["path"],
            output=target["declaration"],
            # gif / png 連番には音を渡しません（入れ物が音を持てないため）
            audio_path=None if target["format"] in ("png-sequence", "gif") else audio_path,
            audio=session.get("audio"),
            start_index=rng["startFrame"],
            stride=(
                max(1, round(timeline["fps"] / (target["declaration"].get("gifFps") or 15)))
                if target["format"] == "gif"
                else 1
            ),
        )
        for target in targets
    ]

    for exporter in exporters:
        exporter.begin()
    total = rng["endFrame"] - rng["startFrame"] + 1
    progress = None if options.get("quiet") else logger.progress(total, "render")
    started = time.perf_counter()

    # 光過敏性発作（PSE）の検査。既定で回します。書き出したフレームをそのまま
    # 測るだけなので、追加のレンダリングは要りません。
    flash_guard = create_flash_guard(project, timeline, options)

    # ── 助走（warmup）──────────────────────────────────────
    # **区間の途中から描き始めても «前のフレームの記憶» が要る** ものがあります。
    # frameEcho（実フレームの残像）と slitScan は、直前の何フレームかを覚えていて
    # 初めて正しい絵になります。何も助走せずに区間の頭を描くと、そこだけ残像が
    # 消えた絵になり、**繋ぎ目で絵が変わってしまいます**。
    #
    # そこで «書き出さずに描くだけ» のフレームを前に足します。物理とパーティクルは
    # 0 フレーム目から追いつき直すので助走は要りませんが、履歴はレンダラーが
    # 持っているだけなので、こうして作り直すしかありません。
    warmup = max(0, round(float(options.get("warmup") or 0)))
    warmup_from = max(0, rng["startFrame"] - warmup)
    if warmup_from < rng["startFrame"]:
        logger.verbose(f'助走 {rng["startFrame"] - warmup_from} フレーム（残像の履歴を作り直します）')
        for frame in range(warmup_from, rng["startFrame"]):
            renderer.render_frame(frame)

    # 並列レンダリングの子プロセスは «何フレーム描いたか» を親に伝えます。
    # 親から渡された報告口（multiprocessing の Queue）に流すだけです。
    report = options.get("report_progress")

    for frame in range(rng["startFrame"], rng["endFrame"] + 1):
        bitmap = renderer.render_frame(frame)
        for exporter in exporters:
            exporter.write_frame(bitmap, frame)
        if flash_guard is not None:
            flash_guard.push(bitmap)
        done = frame - rng["startFrame"] + 1
        if progress is not None:
            progress.update(done)
        if report is not None and (done % 5 == 0 or done == total):
            report(done)
        if options.get("on_frame"):
            options["on_frame"](frame, bitmap)

    flash_report = flash_guard.report() if flash_guard is not None else None
    results = [exporter.end() for exporter in exporters]
    result = results[0]
    if flash_report and not flash_report.get("ok", True):
        describe = bridge.pick("movo.core.flash_guard", "describe_flash_report", "describeFlashReport")
        if not getattr(describe, "movo_not_connected", False):
            for line in describe(flash_report):
                logger.warn(line)
    if progress is not None:
        progress.done(f'{result["frames"]} frames -> {_relative(result["path"])}')
    elapsed = time.perf_counter() - started
    for extra in results[1:]:
        logger.info(f'  -> {_relative(extra["path"])}')

    return {
        **result,
        "format": output_format,
        # 2 本目以降も呼び出し元から見えるようにしておきます
        "outputs": [
            {"path": r["path"], "format": targets[i]["format"], "frames": r["frames"]} for i, r in enumerate(results)
        ],
        "flashReport": flash_report,
        "elapsedSeconds": elapsed,
        "fpsAchieved": total / max(0.001, elapsed),
        "range": rng,
        "audioPath": audio_path,
    }


def _write_audio_sidecar(session: Session, options: dict, output_format: str, output_path: str) -> str | None:
    """音を wav に落として、書き出しに渡せる場所へ置く。

    `audio: False`（CLI では `--no-audio`）は «音を動画に入れない» という指定です。
    音そのものは混ぜたまま（＝音に反応するアニメーションはそのまま動く）なので、
    **絵は変わりません**。並列レンダリングの子プロセスがこれを使います。区間ごとに
    音を入れると、どの区間も曲の頭から鳴ってしまうためです。
    """
    if not session.get("audio"):
        return None
    if options.get("audio") is False:
        return None
    if output_format in ("png-sequence", "gif"):
        return None

    encoded = bridge.encode_wav(session["audio"])
    if bridge.find_ffmpeg():
        # 音を «mix.wav» という固定名にすると、同じフォルダの別プロジェクトを
        # 同時にレンダリングしたときに互いのファイルを上書きしてしまいます。
        # ffmpeg は -shortest で書き出すので、他人の（短い）音を掴んだ側は
        # 動画まで途中で切られます。中身のハッシュを名前に入れて避けます。
        import hashlib

        digest = hashlib.sha256(bytes(encoded)).hexdigest()[:12]
        cache_root = getattr(session["cache"], "root", None) or str(Path(session["projectRoot"]) / "cache")
        audio_path = Path(cache_root) / "audio" / f"mix-{digest}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(bytes(encoded))
        return str(audio_path)

    sidecar = Path(output_path).with_suffix(".wav")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(bytes(encoded))
    logger.warn(f"ffmpeg が無いので、音は別ファイルに書きました: {sidecar}")
    return None


def _relative(target: str) -> str:
    try:
        return os.path.relpath(target, os.getcwd())
    except ValueError:
        return target


def write_lock_file(session: Session) -> str:
    """あとで同じ動画を作り直せるように movo.lock.json を書く。"""
    ffmpeg = bridge.find_ffmpeg()
    lock = {
        "movoVersion": __version__,
        "generatedAt": None,
        "project": {
            "file": Path(session["file"]).name,
            "seed": session["project"]["project"].get("seed"),
            "fps": session["timeline"]["fps"],
            "duration": session["timeline"]["duration"],
            "width": session["timeline"]["width"],
            "height": session["timeline"]["height"],
        },
        "quality": session["project"]["render"]["quality"],
        "ffmpeg": (ffmpeg or {}).get("version"),
        "assets": sorted((session["project"].get("assets") or {}).keys()),
    }
    target = Path(session["projectRoot"]) / "movo.lock.json"
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(target)


def validate_file(file: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """副作用の無い構造検証（`movo validate` 用）。"""
    options = options or {}
    read = read_project_file(file)
    raw_source, absolute, project_root = read["raw"], read["file"], read["projectRoot"]

    # create_session と同じ順で畳む。ここを揃えないと «render は通るのに
    # validate だけ落ちる» ということが起きる（params / extends / fromAudio）。
    prepare_project = bridge.pick("movo.schema.params", "prepare_project", "prepareProject")
    if getattr(prepare_project, "movo_not_connected", False):
        source = raw_source
    else:
        source = prepare_project(raw_source, file=absolute, **_param_kwargs(options))

    skills = expand_skills(source, project_root, absolute)
    raw = skills["project"]
    resolve_bpm = bridge.pick("movo.audio", "resolve_bpm_from_audio", "resolveBpmFromAudio")
    if not getattr(resolve_bpm, "movo_not_connected", False):
        resolve_bpm(raw, None, {"projectRoot": project_root})

    result = validate(raw, file=absolute, throw_on_error=False)
    # raw はスキル展開後の JSON。呼び出し側はこれを使うこと。ファイルを読み直すと
    # use が未展開のままなので、レイヤー数の集計がずれる。
    base = {**result, "file": absolute, "raw": raw, "source": source, "skillsUsed": skills["used"]}
    if options.get("normalize") and result.get("valid"):
        normalize_project = bridge.pick("movo.schema", "normalize_project", "normalizeProject")
        project = normalize_project(raw)
        build_timeline = bridge.pick("movo.timeline", "build_timeline", "buildTimeline")
        return {**base, "timeline": build_timeline(project), "project": project}
    return base
