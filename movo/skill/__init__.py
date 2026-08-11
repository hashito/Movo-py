"""movo-skill — 基礎アニメーションとスキル。

2 階層に分かれています。

  基礎アニメーション (animation)
    1 レイヤーに対する動きの部品。`animations` / `modifiers` / `effects` /
    `textAnimator` / `transform` を生成してレイヤーに合成する。
    例: fade-in, pop-in, float, beat-bounce, handheld

  スキル (skill)
    入力値からレイヤー群（＝場面まるごと）を生成するテンプレート。
    例: cutin-title, lyric-line, glitch-overlay, weather

  シーンスキル (kind: "scene")
    シーン «まるごと» のテンプレート。`scenes` に並べて使う。
    尺は小節で書けるので、BPM を変えるだけで曲に合います。
    例: mv-intro, mv-verse, mv-chorus, mv-outro, rich-intro, rich-chorus

  ムービースキル (kind: "movie")
    シーンスキルを順番に並べたもの＝動画 1 本ぶん。入力値だけで 1 本になる。
    例: lyric-mv, hype-lyric-mv, rich-mv

どれも JSON なので、プロジェクトの `animations/` `skills/` `scenes/` `movies/`
に置くだけで追加・上書きできます。定義そのものは JS 版から **1 文字も変えずに**
`movo/skill/library/` へ持ってきています（JSON なので移植の必要がありません）。
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from movo.cli.console import logger
from movo.cli.errors import ErrorCodes, MovoError, reason_of

from .responsive_time import fit_animations_to_clip, merge_protect, resolve_protect
from .template import (
    LYRIC_FUNCTIONS,
    create_skill_engine,
    expand_template,
    is_timed_line,
    resolve_inputs,
)

__all__ = [
    "KINDS",
    "SKILL_CATEGORIES",
    "ANIMATION_CATEGORIES",
    "SkillRegistry",
    "builtin_library_root",
    "expand_animation",
    "apply_animation_uses",
    "expand_skill",
    "expand_scene_skill",
    "expand_movie_skill",
    "expand_project_skills",
    "build_skill_project",
    "build_scene_project",
    "build_movie_project",
    "parse_input_assignments",
    "coerce_assignment",
    "find_dead_inputs",
    "expand_template",
    "resolve_inputs",
    "fit_animations_to_clip",
    "resolve_protect",
    "merge_protect",
    "is_timed_line",
    "create_skill_engine",
    "LYRIC_FUNCTIONS",
]


def builtin_library_root() -> Path:
    """組み込みスキル定義の置き場。

    **単体 EXE（PyInstaller）では中身が展開先に置かれます。** `__file__` から
    たどると EXE の中を指してしまい «1 件も読めない» ので、`sys._MEIPASS` を
    先に見ます（`tools/build_exe.py` が `--add-data` で同じ相対位置に入れます）。
    """
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        candidate = Path(bundled) / "movo" / "skill" / "library"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent / "library"


def strip_json_comments(text: str) -> str:
    """`//` と `/* */` を取り除く。文字列の中は触らない。"""
    out: list[str] = []
    in_string = False
    in_line = False
    in_block = False
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < length else ""
        if in_line:
            if ch == "\n":
                in_line = False
                out.append(ch)
            i += 1
            continue
        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            out.append(ch)
            if ch == "\\":
                out.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def read_jsonc(file: str | os.PathLike[str]) -> Any:
    """JSON からコメントを取り除いて読む（プロジェクト JSON と同じ書式を許す）。"""
    text = Path(file).read_text(encoding="utf-8")
    return json.loads(strip_json_comments(text))


def _list_json_files(directory: Path) -> list[Path]:
    try:
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".json")
    except OSError:
        return []


# 種類ごとの «置き場» と «登録簿の名前»。
#
# レイヤースキルとシーンスキルは名前空間を分けています。同じ "intro" でも
# «レイヤー群» と «シーンまるごと» では使う場所が違うため、混ざると
# 「どちらの意味で書いたのか」が読み手に分からなくなるからです。
KINDS = ["animation", "skill", "scene", "movie"]

KIND_DIRECTORIES = [
    ("animations", "animation"),
    ("skills", "skill"),
    ("scenes", "scene"),
    ("movies", "movie"),
]


def _header_of(definition: Any) -> dict | None:
    """見出し（name / title / kind）の置き場。

    シーンスキルにも `"skill": {"kind": "scene"}` と書かせているのは、本文側の
    キー（`scene` / `sequence`）と見出しを分けておかないと、«レイヤーの入った
    scene» を見出しと読み違えてしまうためです。
    """
    if not isinstance(definition, dict):
        return None
    return definition.get("skill") or definition.get("animation")


class SkillEntry(dict):
    """登録簿の 1 件。`dict` なのは `--json` でそのまま出せるようにするためです。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class SkillRegistry:
    """基礎アニメーションとスキルの登録簿。"""

    def __init__(self) -> None:
        self.animations: dict[str, SkillEntry] = {}
        self.skills: dict[str, SkillEntry] = {}
        self.scene_skills: dict[str, SkillEntry] = {}
        self.movies: dict[str, SkillEntry] = {}

    def load(self, project_root: str | os.PathLike[str] | None = None, builtin: bool = True) -> "SkillRegistry":
        """組み込みライブラリと、プロジェクト固有の定義を読み込む。

        同じ名前があればプロジェクト側が優先される（上書きできる）。
        """
        root = builtin_library_root()
        for directory, kind in KIND_DIRECTORIES:
            if builtin:
                self._load_directory(root / directory, kind, "builtin")
            if project_root:
                self._load_directory(Path(project_root) / directory, kind, "project")
        return self

    def _load_directory(self, directory: Path, kind: str, source: str) -> None:
        for file in _list_json_files(directory):
            try:
                definition = read_jsonc(file)
            except Exception as error:  # noqa: BLE001
                logger.warn(f"{kind} の読み込みに失敗しました: {file} ({error})")
                continue
            try:
                self.register(definition, kind=kind, source=source, file=str(file))
            except MovoError as error:
                logger.warn(f"{file}: {reason_of(error)}")

    def register(
        self,
        definition: dict,
        kind: str | None = None,
        source: str = "inline",
        file: str | None = None,
    ) -> SkillEntry:
        """定義を 1 件登録する。

        種類は «ヘッダーの kind» が最優先です。置き場（skills/ scenes/ movies/）で
        決めてしまうと、フォルダを移しただけで意味が変わってしまうためです。
        """
        header = _header_of(definition)
        declared = (header or {}).get("kind")
        if declared in KINDS:
            resolved_kind = declared
        elif kind is not None:
            resolved_kind = kind
        else:
            resolved_kind = "skill" if isinstance(definition, dict) and "skill" in definition else "animation"

        if not header or not header.get("name"):
            raise MovoError(ErrorCodes.MOVO_SCHEMA_INVALID, f"{resolved_kind} の定義に name がありません", file=file)

        entry = SkillEntry(
            kind=resolved_kind,
            name=header["name"],
            title=header.get("title", header["name"]),
            description=header.get("description", ""),
            category=header.get("category", "other"),
            tags=header.get("tags", []),
            version=header.get("version", "1.0.0"),
            learnedFrom=header.get("learnedFrom", []),
            inputs=definition.get("inputs", {}) if isinstance(definition, dict) else {},
            definition=definition,
            source=source,
            file=file,
        )
        self._map(resolved_kind)[entry["name"]] = entry
        return entry

    def _map(self, kind: str) -> dict[str, SkillEntry]:
        if kind == "skill":
            return self.skills
        if kind == "scene":
            return self.scene_skills
        if kind == "movie":
            return self.movies
        return self.animations

    def animation(self, name: str | None) -> SkillEntry | None:
        return self.animations.get(name) if name else None

    def skill(self, name: str | None) -> SkillEntry | None:
        return self.skills.get(name) if name else None

    def scene_skill(self, name: str | None) -> SkillEntry | None:
        """シーンスキル（`scenes` に並べられるもの）。"""
        return self.scene_skills.get(name) if name else None

    def movie(self, name: str | None) -> SkillEntry | None:
        """ムービースキル（動画 1 本ぶん）。"""
        return self.movies.get(name) if name else None

    def find(self, name: str | None) -> SkillEntry | None:
        """種類を問わず引く（movo skill show / render の入口で使う）。"""
        return self.movie(name) or self.scene_skill(name) or self.skill(name) or self.animation(name)

    def list(self, kind: str | None = None) -> list[SkillEntry]:
        if kind in KINDS:
            return sorted(self._map(kind).values(), key=lambda e: e["name"])
        found: list[SkillEntry] = []
        for k in KINDS:
            found.extend(self._map(k).values())
        return sorted(found, key=lambda e: e["name"])

    def names(self) -> dict[str, list[str]]:
        """名前の一覧（検証で「既知の名前」として使う）。"""
        return {
            "animations": list(self.animations),
            "skills": list(self.skills),
            "scenes": list(self.scene_skills),
            "movies": list(self.movies),
        }


def _suggest(name: str, candidates) -> str | None:
    """よく似た名前を提案する（打ち間違いの救済）。"""
    best = None
    best_score = float("inf")
    for candidate in candidates:
        score = _distance(name, candidate)
        if score < best_score:
            best_score = score
            best = candidate
    return best if best_score <= max(2, len(name) // 3) else None


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[len(b)]


def _to_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def expand_animation(
    registry: SkillRegistry, use: dict, scope: dict | None = None, options: dict | None = None
) -> dict:
    """基礎アニメーションを 1 件展開して、レイヤーに合成する形を返す。"""
    scope = scope or {}
    options = options or {}
    # scale を書いていない呼び出しでも壊れないようにしておく（等倍として扱う）
    scaled = {"scale": 1, **scope}
    entry = registry.animation(use.get("animation"))
    if entry is None:
        hint = _suggest(str(use.get("animation")), registry.animations)
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'基礎アニメーション "{use.get("animation")}" が見つかりません',
            path=options.get("path"),
            hint=(
                f'もしかして "{hint}" ですか？ 一覧は movo skill list --animations'
                if hint
                else "一覧は movo skill list --animations"
            ),
        )
    values, issues = resolve_inputs(
        entry["inputs"], use.get("with") or {}, {"name": entry["name"], "scale": scaled["scale"]}
    )
    if issues:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            " / ".join(issue["message"] for issue in issues),
            path=options.get("path"),
        )
    engine = options.get("engine") or create_skill_engine(options.get("seed", 0))
    produced = expand_template(
        entry["definition"].get("produces", {}),
        {**scaled, **values, "_name": entry["name"]},
        {"engine": engine, "path": f'animations.{entry["name"]}', "file": entry.get("file")},
    )
    return {
        "animations": produced.get("animations", []),
        "modifiers": produced.get("modifiers", []),
        "effects": produced.get("effects", []),
        "textAnimator": produced.get("textAnimator"),
        "transform": produced.get("transform"),
        "style": produced.get("style"),
        # frameEcho / frameHold / echo / mask など、レイヤー直下の項目をまとめて足す口
        "layer": produced.get("layer"),
    }


def _normalize_uses(use: Any) -> list[dict]:
    if not use:
        return []
    items = use if isinstance(use, list) else [use]
    out = []
    for item in items:
        normalised = {"animation": item} if isinstance(item, str) else item
        if isinstance(normalised, dict) and normalised.get("animation"):
            out.append(normalised)
    return out


def apply_animation_uses(layer: Any, registry: SkillRegistry, scope: dict, options: dict | None = None) -> Any:
    """レイヤーの `use` を解決して、基礎アニメーションを合成したレイヤーを返す。

    合成の順番は「基礎アニメーション → レイヤー自身の指定」です。同じプロパティを
    両方が動かす場合、後に評価されるレイヤー自身の指定が勝ちます。
    """
    options = options or {}
    if not isinstance(layer, dict):
        return layer

    # スキル展開が «このレイヤーはこの尺・この保護区間で畳む» と書き置いた指示。
    # 基礎アニメーションを合成したあとでないとキーフレームが出そろわないので、
    # 畳むのはここまで遅らせています。展開の途中でしか使わない印なので、
    # 取り出したら出力からは消します。
    instruction = layer.get("_timeFit") or options.get("fit")
    current = layer
    if "_timeFit" in layer:
        current = {k: v for k, v in layer.items() if k != "_timeFit"}

    fit_options = {
        "clipEnd": (instruction or {}).get("end"),
        "protect": (instruction or {}).get("protect"),
        "bpm": (scope or {}).get("bpm"),
        "timeSignature": (scope or {}).get("timeSignature"),
        "path": current.get("id"),
    }
    # 入れ子のレイヤーにも同じ指示を渡す（子に end があれば子の end が勝つ）
    child_options = {**options, "fit": instruction} if instruction else options

    uses = _normalize_uses(current.get("use"))
    if not uses:
        # 自分に use が無くても、子レイヤーには付いているかもしれない
        with_children = current
        if isinstance(current.get("layers"), list):
            with_children = {
                **current,
                "layers": [apply_animation_uses(child, registry, scope, child_options) for child in current["layers"]],
            }
        # use も保護区間の指示も無いレイヤーは «手書き» とみなして畳みません
        # （手書きのプロジェクトは movo validate が警告する、という役割分担のまま）。
        if not instruction and with_children.get("timeProtect") is None:
            return with_children
        return fit_animations_to_clip(with_children, fit_options)

    merged = {k: v for k, v in current.items() if k != "use"}
    collected: dict[str, list] = {"animations": [], "modifiers": [], "effects": []}
    text_animator: dict | None = None
    transform: dict | None = None
    style_extra: dict | None = None
    extra: dict | None = None

    for use in uses:
        produced = expand_animation(registry, use, scope, options)
        collected["animations"].extend(produced["animations"])
        collected["modifiers"].extend(produced["modifiers"])
        collected["effects"].extend(produced["effects"])
        if produced.get("textAnimator"):
            text_animator = {**(text_animator or {}), **produced["textAnimator"]}
        if produced.get("transform"):
            transform = {**(transform or {}), **produced["transform"]}
        if produced.get("style"):
            style_extra = {**(style_extra or {}), **produced["style"]}
        if produced.get("layer"):
            extra = {**(extra or {}), **produced["layer"]}

    merged["animations"] = [*collected["animations"], *_to_list(current.get("animations"))]
    merged["modifiers"] = [*collected["modifiers"], *_to_list(current.get("modifiers"))]
    merged["effects"] = [*collected["effects"], *_to_list(current.get("effects"))]
    for key in ("animations", "modifiers", "effects"):
        if not merged[key]:
            del merged[key]
    if text_animator:
        merged["textAnimator"] = {**text_animator, **(current.get("textAnimator") or {})}
    if transform:
        merged["transform"] = {**transform, **(current.get("transform") or {})}
    if style_extra:
        merged["style"] = {**style_extra, **(current.get("style") or {})}
    # produces.layer は frameEcho / frameHold のようなレイヤー直下の項目。
    # ここでもレイヤー自身に書いたものが勝つ。
    if extra:
        for key, value in extra.items():
            if current.get(key) is None:
                merged[key] = value

    # 入れ子のレイヤーも解決する
    if isinstance(merged.get("layers"), list):
        merged["layers"] = [apply_animation_uses(child, registry, scope, child_options) for child in merged["layers"]]

    # 基礎アニメーションの時間はレイヤーの尺を知らずに書かれているので、ここで
    # はみ出しを畳んでおく（短い尺でも動きが終わるようにする）。保護区間があれば
    # «頭と尻» はそのまま残る。
    return fit_animations_to_clip(merged, fit_options)


def _scene_seconds(duration: Any, bpm: Any, name: str) -> float:
    """`"4bar"` でも `"2.5"` でも `4` でも秒にする（尺の合計を出すため）。"""
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        value = float(duration)
        return value if value == value else 4.0
    from .responsive_time import _musical_seconds

    seconds = _musical_seconds(duration, bpm, None, f"{name}.duration")
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return 4.0
    return value if value == value else 4.0


def expand_skill(registry: SkillRegistry, use: dict, context: dict | None = None) -> dict:
    """スキルを 1 件展開する（レイヤー群を返す）。"""
    context = context or {}
    entry = registry.skill(use.get("skill"))
    if entry is None:
        hint = _suggest(str(use.get("skill")), registry.skills)
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'スキル "{use.get("skill")}" が見つかりません',
            path=context.get("path"),
            hint=f'もしかして "{hint}" ですか？ 一覧は movo skill list' if hint else "一覧は movo skill list",
        )

    engine = context.get("engine") or create_skill_engine(context.get("seed", 0))
    definition = entry["definition"]
    video = definition.get("video") or {}
    width = context.get("width") or video.get("width") or 1920
    height = context.get("height") or video.get("height") or 1080
    fps = context.get("fps") or video.get("fps") or 30
    instance_id = use.get("id") or entry["name"]
    start = use.get("start") or 0

    # scale はスキルの設計解像度に対する倍率。文字サイズや半径などの
    # 「ピクセルで書いた寸法」に掛けておくと、どの解像度でも同じ絵になる。
    design_height = video.get("height") or 1080
    scale = height / design_height

    values, issues = resolve_inputs(entry["inputs"], use.get("with") or {}, {"name": entry["name"], "scale": scale})
    if issues:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID, " / ".join(i["message"] for i in issues), path=context.get("path")
        )

    scope = {
        **values,
        "_id": instance_id,
        "_name": entry["name"],
        "_start": start,
        "width": width,
        "height": height,
        "fps": fps,
        "scale": scale,
        "bpm": context.get("bpm") or values.get("bpm") or 120,
        "centerX": width / 2,
        "centerY": height / 2,
    }
    # duration は他の値から計算できるようにしておく。"6bar" のような «小節» も
    # 受けます。ここで秒に直すのは、レイヤーの end に入れる値だからです。
    duration_template = use.get("duration")
    if duration_template is None:
        duration_template = definition.get("duration", video.get("duration", 4))
    duration = expand_template(
        duration_template, scope, {"engine": engine, "path": f'{entry["name"]}.duration', "file": entry.get("file")}
    )
    scope["_duration"] = _scene_seconds(duration, scope["bpm"], entry["name"])

    expanded = expand_template(
        {
            "layers": definition.get("layers", []),
            "assets": definition.get("assets", {}),
            "audio": definition.get("audio", []),
            "physicsWorld": definition.get("physicsWorld"),
            "characters": definition.get("characters", {}),
        },
        scope,
        {"engine": engine, "path": entry["name"], "file": entry.get("file")},
    )

    # 尺追従の保護区間。スキル全体の既定を先に読んでおく（レイヤーの timeProtect が勝つ）。
    protect = resolve_protect(
        definition.get("responsiveTime"),
        {
            "bpm": scope["bpm"],
            "timeSignature": context.get("timeSignature"),
            "path": f'{entry["name"]}.responsiveTime',
        },
    )

    # レイヤー ID の衝突を避けるため、id を書いていないレイヤーには自動で付ける
    layers = []
    for index, layer in enumerate(expanded.get("layers") or []):
        with_id = dict(layer)
        if not with_id.get("id"):
            with_id["id"] = f'{instance_id}-{with_id.get("type", "layer")}-{index}'
        # スキル内の時間はスキル基準なので、埋め込み位置ぶんずらす
        if start != 0:
            with_id["start"] = (with_id.get("start") or 0) + start
            if with_id.get("end") is not None:
                with_id["end"] += start
            elif with_id.get("duration") is None:
                with_id["end"] = start + scope["_duration"]
        elif with_id.get("end") is None and with_id.get("duration") is None:
            with_id["end"] = scope["_duration"]
        # レイヤーの `use` はこの先（apply_animation_uses）で解決されるので、
        # «どう畳むか» を書き置いて持ち回る。ここで畳んで終わりにすると、
        # 基礎アニメーションのキーフレームが畳まれないまま残る。
        effective = merge_protect(
            protect,
            resolve_protect(
                with_id.get("timeProtect"),
                {
                    "bpm": scope["bpm"],
                    "timeSignature": context.get("timeSignature"),
                    "path": f'{with_id["id"]}.timeProtect',
                },
            ),
        )
        if effective:
            with_id["_timeFit"] = {"protect": effective}
        layers.append(
            fit_animations_to_clip(
                with_id,
                {
                    "protect": effective,
                    "bpm": scope["bpm"],
                    "timeSignature": context.get("timeSignature"),
                    "path": with_id["id"],
                },
            )
        )

    return {
        "layers": layers,
        "assets": expanded.get("assets") or {},
        "audio": expanded.get("audio") or [],
        "physicsWorld": expanded.get("physicsWorld"),
        "characters": expanded.get("characters") or {},
        "duration": scope["_duration"],
        "inputs": values,
        "entry": entry,
    }


def expand_scene_skill(registry: SkillRegistry, use: dict, context: dict | None = None) -> dict:
    """シーンスキルを 1 件展開して «シーン 1 つぶん» を返す。

    尺は `"${bars}bar"` のような «小節の文字列» のまま返します。秒に直すのは
    正規化（normalize_project）の仕事で、そこまで文字列で運べば «BPM を変える
    だけで全カットが曲に合う» という性質が最後まで残るからです。
    """
    context = context or {}
    entry = registry.scene_skill(use.get("scene"))
    if entry is None:
        hint = _suggest(str(use.get("scene")), registry.scene_skills)
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'シーンスキル "{use.get("scene")}" が見つかりません',
            path=context.get("path"),
            hint=(
                f'もしかして "{hint}" ですか？ 一覧は movo skill list --scenes'
                if hint
                else "一覧は movo skill list --scenes"
            ),
        )

    engine = context.get("engine") or create_skill_engine(context.get("seed", 0))
    definition = entry["definition"]
    source = definition.get("scene") or {}
    video = definition.get("video") or {}
    width = context.get("width") or video.get("width") or 1920
    height = context.get("height") or video.get("height") or 1080
    fps = context.get("fps") or video.get("fps") or 30
    design_height = video.get("height") or 1080
    scale = height / design_height
    instance_id = use.get("id") or entry["name"]

    values, issues = resolve_inputs(entry["inputs"], use.get("with") or {}, {"name": entry["name"], "scale": scale})
    if issues:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID, " / ".join(i["message"] for i in issues), path=context.get("path")
        )

    project_block = definition.get("project") or {}
    bpm = context.get("bpm") or values.get("bpm") or project_block.get("bpm") or 120
    scope = {
        **values,
        "_id": instance_id,
        "_name": entry["name"],
        "width": width,
        "height": height,
        "fps": fps,
        "scale": scale,
        "bpm": bpm,
        "centerX": width / 2,
        "centerY": height / 2,
    }

    duration_template = use.get("duration")
    if duration_template is None:
        duration_template = source.get("duration", definition.get("duration", "4bar"))
    duration = expand_template(
        duration_template, scope, {"engine": engine, "path": f'{entry["name"]}.duration', "file": entry.get("file")}
    )
    # 式の中で «残り何秒か» を使いたいことがあるので、秒でも渡しておく。
    scope["_duration"] = _scene_seconds(duration, bpm, entry["name"])

    expanded = expand_template(
        {
            "layers": source.get("layers", []),
            "transition": source.get("transition"),
            "background": source.get("background"),
            "use": source.get("use"),
            "assets": definition.get("assets", {}),
            "audio": definition.get("audio", []),
            "characters": definition.get("characters", {}),
            "physicsWorld": definition.get("physicsWorld"),
        },
        scope,
        {"engine": engine, "path": entry["name"], "file": entry.get("file")},
    )

    # 尺追従の保護区間。シーンスキルは «6 小節ぶんの絵を 12 小節で使う» という
    # 使われ方をするので、ここが効きどころです。
    protect = resolve_protect(
        definition.get("responsiveTime"),
        {"bpm": bpm, "timeSignature": context.get("timeSignature"), "path": f'{entry["name"]}.responsiveTime'},
    )

    # レイヤーには «尺いっぱい» を既定にしたいので end を足しません。足すとシーンを
    # 伸ばしたときにレイヤーだけ先に消えてしまいます。
    #
    # ただし «畳むときの尺» は要るので、保護区間を書いたスキルにだけ `_timeFit`
    # として «シーンの秒数» を持たせます。end を足す代わりにこれを渡すのは、
    # **«いつ消えるか» と «いつまでに動き終わるか» は別の話** だからです。
    layers = []
    for index, layer in enumerate(expanded.get("layers") or []):
        with_id = {**layer, "id": layer.get("id") or f'{instance_id}-{layer.get("type", "layer")}-{index}'}
        if protect or with_id.get("timeProtect") is not None:
            with_id["_timeFit"] = {"end": scope["_duration"], "protect": protect}
        layers.append(with_id)

    scene: dict[str, Any] = {"id": instance_id, "duration": duration, "layers": layers}
    if use.get("start") is not None:
        scene["start"] = use["start"]
    if expanded.get("transition"):
        scene["transition"] = expanded["transition"]
    if expanded.get("background"):
        scene["background"] = expanded["background"]
    # シーンスキルの中からレイヤースキルを呼べるようにする（後段の expand_uses が拾う）
    if expanded.get("use"):
        scene["use"] = expanded["use"]
    if expanded.get("physicsWorld"):
        scene["physicsWorld"] = expanded["physicsWorld"]

    return {
        "scene": scene,
        "assets": expanded.get("assets") or {},
        "audio": expanded.get("audio") or [],
        "characters": expanded.get("characters") or {},
        "physicsWorld": expanded.get("physicsWorld"),
        "duration": scope["_duration"],
        "inputs": values,
        "entry": entry,
    }


def _scene_skill_use(scene: Any, registry: SkillRegistry) -> dict | None:
    """シーンの `use` が «シーンスキルの呼び出し» かどうかを見分ける。

    `"use": "mv-intro"` と `"use": [{"skill": "lyric-line"}]` は書き方が近いので、
    «シーンスキルとして登録されている名前か» で判定します。名前空間を分けてある
    ので取り違えは起きません。
    """
    if not isinstance(scene, dict):
        return None
    use = scene.get("use")
    if isinstance(use, str):
        if registry.scene_skill(use):
            return {"scene": use, "with": scene.get("with"), "id": scene.get("id")}
        return None
    if isinstance(use, dict) and isinstance(use.get("scene"), str):
        return {
            "scene": use["scene"],
            "with": use.get("with") or scene.get("with"),
            "id": use.get("id") or scene.get("id"),
        }
    return None


def expand_movie_skill(registry: SkillRegistry, use: dict, context: dict | None = None) -> dict:
    """ムービースキルの `sequence` を «シーンの並び» に展開する。"""
    context = context or {}
    entry = registry.movie(use.get("movie"))
    if entry is None:
        hint = _suggest(str(use.get("movie")), registry.movies)
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'ムービースキル "{use.get("movie")}" が見つかりません',
            path=context.get("path"),
            hint=(
                f'もしかして "{hint}" ですか？ 一覧は movo skill list --movies'
                if hint
                else "一覧は movo skill list --movies"
            ),
        )

    engine = context.get("engine") or create_skill_engine(context.get("seed", 0))
    definition = entry["definition"]
    video = definition.get("video") or {}
    width = context.get("width") or video.get("width") or 1920
    height = context.get("height") or video.get("height") or 1080
    fps = context.get("fps") or video.get("fps") or 30
    scale = height / (video.get("height") or 1080)

    values, issues = resolve_inputs(entry["inputs"], use.get("with") or {}, {"name": entry["name"], "scale": scale})
    if issues:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID, " / ".join(i["message"] for i in issues), path=context.get("path")
        )

    project_block = definition.get("project") or {}
    bpm = context.get("bpm") or values.get("bpm") or project_block.get("bpm") or 120
    scope = {
        **values,
        "_name": entry["name"],
        "width": width,
        "height": height,
        "fps": fps,
        "scale": scale,
        "bpm": bpm,
        "centerX": width / 2,
        "centerY": height / 2,
    }

    # sequence 自体もテンプレート。when / repeat が使えるので «歌詞の行数だけ
    # A メロを繰り返す» のような組み立てが書けます。
    expanded = expand_template(
        {
            "sequence": definition.get("sequence", []),
            "assets": definition.get("assets", {}),
            "audio": definition.get("audio", []),
            "characters": definition.get("characters", {}),
            "presets": definition.get("presets", {}),
            "camera": definition.get("camera"),
        },
        scope,
        {"engine": engine, "path": entry["name"], "file": entry.get("file")},
    )

    scenes: list[dict] = []
    assets = dict(expanded.get("assets") or {})
    characters = dict(expanded.get("characters") or {})
    audio = list(expanded.get("audio") or [])
    total = 0.0
    seen: dict[str, int] = {}

    for index, item in enumerate(expanded.get("sequence") or []):
        if not isinstance(item, dict) or not isinstance(item.get("scene"), str):
            continue
        # 同じシーンスキルを 2 回並べても id がぶつからないようにする
        count = seen.get(item["scene"], 0) + 1
        seen[item["scene"]] = count
        scene_id = item.get("id") or (f'{item["scene"]}-{count}' if count > 1 else item["scene"])
        result = expand_scene_skill(
            registry,
            {
                "scene": item["scene"],
                "id": scene_id,
                "with": item.get("with") or {},
                "duration": item.get("duration"),
            },
            {
                "width": width,
                "height": height,
                "fps": fps,
                "bpm": bpm,
                # 拍子も渡さないと、シーン側の "1/2bar" が 4 拍子として解釈される
                "timeSignature": context.get("timeSignature") or project_block.get("timeSignature"),
                "seed": context.get("seed"),
                "engine": engine,
                "path": f'{entry["name"]}.sequence[{index}]',
            },
        )
        scenes.append(result["scene"])
        assets.update(result["assets"])
        characters.update(result["characters"])
        audio.extend(result["audio"])
        total += result["duration"]

    return {
        "scenes": scenes,
        "assets": assets,
        "audio": audio,
        "characters": characters,
        "presets": expanded.get("presets") or {},
        "camera": expanded.get("camera"),
        "duration": total,
        # 尺は «小節» で書かれているので、bpm を持ち帰らないと秒に直せない。
        # 拾い忘れると «"4bar" は拍の単位ですが project.bpm がありません» になる。
        "bpm": bpm,
        "seed": project_block.get("seed"),
        "video": definition.get("video"),
        "inputs": values,
        "entry": entry,
    }


def expand_project_skills(raw: dict | None, registry: SkillRegistry, file: str | None = None) -> dict:
    """プロジェクト JSON の中のスキル参照をすべて展開する。

    対応する書き方:
      project.movie          → 動画 1 本ぶん（シーンの並びごと差し込む）
      project.use            → 最初のシーンに追加
      scene.use （シーンスキル名） → そのシーンをまるごと差し替える
      scene.use （スキル名）       → そのシーンに追加
      layer.use              → 基礎アニメーションを合成

    @returns {"project": ..., "used": [...]}
    """
    project = copy.deepcopy(raw or {})
    project_block = project.get("project") or {}
    engine = create_skill_engine(project_block.get("seed") or 0)
    used: list[dict] = []

    video = project.get("video") or {}
    base_context = {
        "width": video.get("width") or 1920,
        "height": video.get("height") or 1080,
        "fps": video.get("fps") or 30,
        "bpm": project_block.get("bpm"),
        # 保護区間を "1/2bar" と書けるようにするため、拍子もここから配ります。
        "timeSignature": project_block.get("timeSignature"),
        "seed": project_block.get("seed") or 0,
        "engine": engine,
        "file": file,
    }
    animation_scope = {
        "width": base_context["width"],
        "height": base_context["height"],
        "fps": base_context["fps"],
        # 1080p を基準にした倍率。移動量や揺れ幅を px で書いた基礎アニメーションが
        # 解像度に追従できるようにする。
        "scale": base_context["height"] / 1080,
        "bpm": base_context["bpm"] or 120,
        "timeSignature": base_context["timeSignature"],
        "centerX": base_context["width"] / 2,
        "centerY": base_context["height"] / 2,
    }

    def merge_skill(target: dict, result: dict) -> None:
        project["assets"] = {**(result.get("assets") or {}), **(project.get("assets") or {})}
        project["characters"] = {**(result.get("characters") or {}), **(project.get("characters") or {})}
        if result.get("audio"):
            project["audio"] = [*(project.get("audio") or []), *result["audio"]]
        if result.get("physicsWorld"):
            project["physicsWorld"] = {**result["physicsWorld"], **(project.get("physicsWorld") or {})}
        target["layers"] = [*(target.get("layers") or []), *result["layers"]]

    def expand_uses(holder: dict, path: str) -> None:
        uses = holder.get("use")
        if not uses:
            return
        items = uses if isinstance(uses, list) else [uses]
        holder.pop("use", None)
        for index, use in enumerate(items):
            normalised = {"skill": use} if isinstance(use, str) else use
            if not isinstance(normalised, dict) or not normalised.get("skill"):
                continue
            result = expand_skill(registry, normalised, {**base_context, "path": f"{path}.use[{index}]"})
            merge_skill(holder, result)
            used.append(
                {
                    "skill": normalised["skill"],
                    "id": normalised.get("id") or normalised["skill"],
                    "layers": len(result["layers"]),
                }
            )

    # 0) プロジェクト直下の movie（動画 1 本ぶん）。
    #    シーンの並びを丸ごと持ってくるので、いちばん先に処理します。
    if project.get("movie"):
        request = {"movie": project["movie"]} if isinstance(project["movie"], str) else project["movie"]
        name = request.get("movie") or request.get("use") or request.get("name")
        project.pop("movie", None)
        result = expand_movie_skill(
            registry, {"movie": name, "with": request.get("with") or {}}, {**base_context, "path": "movie"}
        )
        # プロジェクトに書いてあるものが «上書き» になるよう、スキル側を先に置く。
        project["assets"] = {**result["assets"], **(project.get("assets") or {})}
        project["characters"] = {**result["characters"], **(project.get("characters") or {})}
        project["presets"] = {**result["presets"], **(project.get("presets") or {})}
        if result.get("camera") and not project.get("camera"):
            project["camera"] = result["camera"]
        if result["audio"]:
            project["audio"] = [*(project.get("audio") or []), *result["audio"]]
        project["scenes"] = [*result["scenes"], *(project.get("scenes") or [])]
        # video と bpm もスキル側の既定を敷く。プロジェクトに書いてあればそちらが勝つ。
        # bpm を敷かないと、シーンの尺（"4bar"）を秒に直せず検証で落ちる。
        project["video"] = {**(result.get("video") or {}), **(project.get("video") or {})}
        if project["video"].get("duration") is None:
            project["video"]["duration"] = result["duration"]
        project["project"] = {**(project.get("project") or {})}
        if project["project"].get("bpm") is None:
            project["project"]["bpm"] = result["bpm"]
        if project["project"].get("seed") is None and result.get("seed") is not None:
            project["project"]["seed"] = result["seed"]
        used.append({"skill": name, "id": name, "layers": len(result["scenes"])})

    # 1) シーンごとの use
    if isinstance(project.get("scenes"), list):
        # 1a) シーンスキル（シーンまるごとの差し替え）
        new_scenes = []
        for index, scene in enumerate(project["scenes"]):
            request = _scene_skill_use(scene, registry)
            if request is None:
                new_scenes.append(scene)
                continue
            result = expand_scene_skill(registry, request, {**base_context, "path": f"scenes[{index}]"})
            project["assets"] = {**result["assets"], **(project.get("assets") or {})}
            project["characters"] = {**result["characters"], **(project.get("characters") or {})}
            if result["audio"]:
                project["audio"] = [*(project.get("audio") or []), *result["audio"]]
            used.append(
                {"skill": request["scene"], "id": result["scene"]["id"], "layers": len(result["scene"]["layers"])}
            )
            # 呼び出し側に書いた id / start / duration などが勝つ（部分的に上書きできる）
            overrides = {k: v for k, v in scene.items() if k not in ("use", "with", "layers")}
            new_scenes.append(
                {
                    **result["scene"],
                    **overrides,
                    "layers": [*result["scene"]["layers"], *(scene.get("layers") or [])],
                }
            )
        project["scenes"] = new_scenes
        # 1b) レイヤースキル
        for index, scene in enumerate(project["scenes"]):
            if isinstance(scene, dict):
                expand_uses(scene, f"scenes[{index}]")

    # 2) プロジェクト直下の use（シーンが無ければ 1 つ作る）
    if project.get("use"):
        if not isinstance(project.get("scenes"), list) or not project["scenes"]:
            project["scenes"] = [{"id": "main", "start": 0, "layers": []}]
        holder = {"layers": project["scenes"][0].get("layers") or [], "use": project["use"]}
        project.pop("use", None)
        expand_uses(holder, "project")
        project["scenes"][0]["layers"] = holder["layers"]

    # 3) レイヤーの use（基礎アニメーション）
    def map_layers(layers):
        return [
            apply_animation_uses(layer, registry, animation_scope, {"engine": engine, "seed": base_context["seed"]})
            for layer in (layers or [])
        ]

    if isinstance(project.get("scenes"), list):
        for scene in project["scenes"]:
            if isinstance(scene, dict):
                scene["layers"] = map_layers(scene.get("layers"))
    if isinstance(project.get("layers"), list):
        project["layers"] = map_layers(project["layers"])
    for composition in (project.get("compositions") or {}).values():
        if isinstance(composition.get("layers"), list):
            composition["layers"] = map_layers(composition["layers"])
        for scene in composition.get("scenes") or []:
            scene["layers"] = map_layers(scene.get("layers"))

    return {"project": project, "used": used}


def build_skill_project(
    registry: SkillRegistry, name: str, inputs: dict | None = None, overrides: dict | None = None
) -> dict:
    """スキル 1 つを単体で動画にするためのプロジェクト JSON を組み立てる。"""
    inputs = dict(inputs or {})
    overrides = overrides or {}
    # 同じ入口で «レイヤー群 / シーン / 動画 1 本» のどれでも書き出せるようにする。
    # 利用者から見ると「名前を渡して描く」という 1 つの操作だからです。
    if registry.movie(name):
        return build_movie_project(registry, name, inputs, overrides)
    if registry.scene_skill(name):
        return build_scene_project(registry, name, inputs, overrides)
    entry = registry.skill(name)
    if entry is None:
        # 名前空間は分けているが、打ち間違いの救済はまたいで探す
        hint = _suggest(name, [*registry.skills, *registry.scene_skills, *registry.movies])
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'スキル "{name}" が見つかりません',
            hint=f'もしかして "{hint}" ですか？' if hint else "一覧は movo skill list",
        )

    definition = entry["definition"]
    video = definition.get("video") or {}
    width = overrides.get("width") or video.get("width") or 1920
    height = overrides.get("height") or video.get("height") or 1080
    fps = overrides.get("fps") or video.get("fps") or 30
    project_block = definition.get("project") or {}
    # 優先順位は「CLI の指定 → 利用者が渡した入力値 → スキルの既定 → 全体の既定」。
    # 以前はスキルの既定が入力値より強く、--set bpm=90 が効かなかった。
    seed = overrides.get("seed") or inputs.get("seed") or project_block.get("seed") or 12345
    bpm = overrides.get("bpm") or inputs.get("bpm") or project_block.get("bpm")

    # --duration は「スキル全体の長さ」なので、duration という入力を持つスキルには
    # その入力としても渡す（そうしないと中の動きだけ元の長さのままになる）
    with_inputs = dict(inputs)
    if (
        overrides.get("duration") is not None
        and entry["inputs"].get("duration")
        and with_inputs.get("duration") is None
    ):
        with_inputs["duration"] = overrides["duration"]

    result = expand_skill(
        registry,
        {
            "skill": name,
            "id": overrides.get("id") or "skill",
            "with": with_inputs,
            "duration": overrides.get("duration"),
        },
        {"width": width, "height": height, "fps": fps, "bpm": bpm, "seed": seed},
    )

    project = {
        "movoVersion": "1.0",
        "project": {
            "name": overrides.get("name") or entry["name"],
            "description": entry["description"],
            "seed": seed,
            **({"bpm": bpm} if bpm else {}),
        },
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": result["duration"],
            "background": overrides.get("background") or video.get("background") or "#000000",
        },
        "assets": result["assets"],
        "characters": result["characters"],
        **({"physicsWorld": result["physicsWorld"]} if result.get("physicsWorld") else {}),
        "scenes": [{"id": "main", "start": 0, "duration": result["duration"], "layers": result["layers"]}],
        **({"audio": result["audio"]} if result["audio"] else {}),
        "render": {
            "quality": overrides.get("quality") or (definition.get("render") or {}).get("quality") or "standard",
            **(definition.get("render") or {}),
        },
        "output": {
            "format": "mp4",
            "codec": "h264",
            **(definition.get("output") or {}),
            **(overrides.get("output") or {}),
        },
    }

    # スキルが返したレイヤーには基礎アニメーションの `use` が残っているので解決する
    expanded = expand_project_skills(project, registry)
    return {"project": expanded["project"], "result": result, "entry": entry}


def build_scene_project(
    registry: SkillRegistry, name: str, inputs: dict | None = None, overrides: dict | None = None
) -> dict:
    """シーンスキル 1 つを単体で動画にする（`movo skill render mv-intro`）。"""
    inputs = dict(inputs or {})
    overrides = overrides or {}
    entry = registry.scene_skill(name)
    definition = entry["definition"]
    video = definition.get("video") or {}
    width = overrides.get("width") or video.get("width") or 1920
    height = overrides.get("height") or video.get("height") or 1080
    fps = overrides.get("fps") or video.get("fps") or 30
    project_block = definition.get("project") or {}
    seed = overrides.get("seed") or inputs.get("seed") or project_block.get("seed") or 12345
    # 小節で書いた尺は bpm が無いと秒に直せないので、必ず入れる。
    bpm = overrides.get("bpm") or inputs.get("bpm") or project_block.get("bpm") or 120

    result = expand_scene_skill(
        registry,
        {
            "scene": name,
            "id": overrides.get("id") or entry["name"],
            "with": inputs,
            "duration": overrides.get("duration"),
        },
        {"width": width, "height": height, "fps": fps, "bpm": bpm, "seed": seed},
    )

    project = {
        "movoVersion": "1.0",
        "project": {
            "name": overrides.get("name") or entry["name"],
            "description": entry["description"],
            "seed": seed,
            "bpm": bpm,
        },
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": result["duration"],
            "background": overrides.get("background") or video.get("background") or "#000000",
        },
        "assets": result["assets"],
        "characters": result["characters"],
        **({"physicsWorld": result["physicsWorld"]} if result.get("physicsWorld") else {}),
        "presets": definition.get("presets") or {},
        "scenes": [{**result["scene"], "start": 0}],
        **({"audio": result["audio"]} if result["audio"] else {}),
        "render": {"quality": overrides.get("quality") or "standard", **(definition.get("render") or {})},
        "output": {
            "format": "mp4",
            "codec": "h264",
            **(definition.get("output") or {}),
            **(overrides.get("output") or {}),
        },
    }

    expanded = expand_project_skills(project, registry)
    return {"project": expanded["project"], "result": result, "entry": entry}


def build_movie_project(
    registry: SkillRegistry, name: str, inputs: dict | None = None, overrides: dict | None = None
) -> dict:
    """ムービースキルを 1 本の動画にする（`movo skill render lyric-mv`）。

    「スキル用 JSON と入力値があれば動画になる」という設計の到達点で、ここから
    先はプロジェクト JSON を 1 行も書かずに書き出せます。
    """
    inputs = dict(inputs or {})
    overrides = overrides or {}
    entry = registry.movie(name)
    definition = entry["definition"]
    video = definition.get("video") or {}
    width = overrides.get("width") or video.get("width") or 1920
    height = overrides.get("height") or video.get("height") or 1080
    fps = overrides.get("fps") or video.get("fps") or 30
    project_block = definition.get("project") or {}
    seed = overrides.get("seed") or inputs.get("seed") or project_block.get("seed") or 12345
    bpm = overrides.get("bpm") or inputs.get("bpm") or project_block.get("bpm") or 120

    result = expand_movie_skill(
        registry,
        {"movie": name, "with": inputs},
        {"width": width, "height": height, "fps": fps, "bpm": bpm, "seed": seed},
    )

    project = {
        "movoVersion": "1.0",
        "project": {
            "name": overrides.get("name") or entry["name"],
            "description": entry["description"],
            "seed": seed,
            "bpm": bpm,
        },
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            # --duration は «1 本の長さ» なので、指定があればそちらを使う
            "duration": overrides.get("duration") or result["duration"],
            "background": overrides.get("background") or video.get("background") or "#000000",
        },
        "assets": result["assets"],
        "characters": result["characters"],
        "presets": result["presets"],
        **({"camera": result["camera"]} if result.get("camera") else {}),
        "scenes": result["scenes"],
        **({"audio": result["audio"]} if result["audio"] else {}),
        "render": {"quality": overrides.get("quality") or "standard", **(definition.get("render") or {})},
        "output": {
            "format": "mp4",
            "codec": "h264",
            **(definition.get("output") or {}),
            **(overrides.get("output") or {}),
        },
    }

    expanded = expand_project_skills(project, registry)
    return {"project": expanded["project"], "result": result, "entry": entry}


def coerce_assignment(text: str) -> Any:
    """コマンドラインの文字列を素直な型に寄せる。

    宣言された型への変換は `resolve_inputs` が行うので、ここは「配列やオブジェクトを
    渡せるようにする」ことと、数値・真偽を扱いやすくすることが目的です。
    """
    trimmed = text.strip()
    if trimmed == "true":
        return True
    if trimmed == "false":
        return False
    if trimmed == "null":
        return None
    if trimmed:
        try:
            number = float(trimmed)
        except ValueError:
            number = None
        if number is not None and number == number and abs(number) != float("inf"):
            # 整数に見える書き方（"174"）は整数で返します。"174.0" と書いたときは
            # 小数のままにします（**書いたとおりに返す**のが、名前に使われたときに
            # 効きます。JS 版で "01" が 1 になって連番が崩れた例があります）。
            if re.fullmatch(r"[+-]?\d+", trimmed):
                return int(trimmed)
            return number
    if trimmed[:1] in ("[", "{"):
        try:
            return json.loads(trimmed)
        except ValueError:
            return text  # JSON でなければ文字列のまま（"{}" を含む歌詞など）
    return text


def parse_input_assignments(items: Any) -> dict[str, Any]:
    """`--set key=value` 形式の文字列を入力オブジェクトにする。"""
    out: dict[str, Any] = {}
    if items is None:
        return out
    # --set を 1 回だけ書いた場合は文字列で来る
    values = items if isinstance(items, list) else [items]
    for item in values:
        text = str(item)
        index = text.find("=")
        if index < 0:
            raise MovoError(
                ErrorCodes.MOVO_CLI_USAGE,
                f'--set は key=value の形式で指定してください: "{text}"',
                hint="例: --set text=タイトル --set bpm=174",
            )
        out[text[:index].strip()] = coerce_assignment(text[index + 1 :])
    return out


SKILL_CATEGORIES = [
    "title",
    "lyric",
    "transition",
    "background",
    "overlay",
    "audio",
    "physics",
    "character",
    "look",
    "other",
]

ANIMATION_CATEGORIES = ["in", "out", "loop", "accent", "text", "camera", "other"]


def find_dead_inputs(registry: SkillRegistry, name: str) -> list[str]:
    """「宣言されているのに、どこからも参照されていない入力値」を探す。

    スキルを直しているうちに、レイヤーだけ消して入力の宣言が残る、ということが
    起きます（`mv-hype` の `second` が実際にそうなっていました）。入力欄には出るのに
    変えても何も起きないので、**使う人からは «壊れている» ように見えます。**

    判定は **本文に名前が出てくるか** で行います。最初は «既定値で組み立てた結果と、
    その入力だけ変えた結果を比べる» という素直な方法にしていましたが、`photo-slide`
    の `overlap` を誤って «死んでいる» と報告しました。既定では `assets` が空で
    `repeat` が 0 回になり、参照している行がそもそも展開されないためです。
    **«既定値では通らない枝» と «本当に消し忘れ» を、出力の比較では区別できません。**
    """
    entry = registry.find(name)
    if not entry or not entry.get("inputs"):
        return []
    # 入力の «宣言» 自体に名前が出るのは当たり前なので、そこは見ません。
    # `definition` の中にも `inputs` が入っているので、**両方**落とします
    # （片方だけ落として «1 件も死んでいない» という結果を一度出しました）。
    body = dict(entry.get("definition") or entry)
    body.pop("inputs", None)
    body.pop("skill", None)
    source = json.dumps(body, ensure_ascii=False)

    dead = []
    for key in entry["inputs"]:
        # 識別子として出てくるかを見ます。`size` が `letterSpacing` に埋もれて
        # 見つかった、のような取りこぼしを防ぐため前後を区切ります。
        pattern = rf"(^|[^A-Za-z0-9_$]){re.escape(key)}([^A-Za-z0-9_$]|$)"
        if not re.search(pattern, source):
            dead.append(key)
    return dead
