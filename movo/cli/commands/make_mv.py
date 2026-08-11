"""`movo make-mv <音声ファイル>` — 曲を渡すと、その曲に合わせた MV を 1 本作る。

手順は 3 つだけです。

  1. 曲を解析して BPM・1 拍目・区間（イントロ／A メロ／サビ…）を出す
  2. ムービースキルの並びを、実際の区間に割り当てる
  3. 曲の長さぶん書き出す

«カット尺を小節で決める» のが MV でいちばん効く設計判断でした。それを人が
計算しなくて済むようにしたのがこのコマンドです。曲を差し替えれば BPM も区間も
そのぶん変わり、カット割りが勝手に追従します。
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from movo.skill import SkillRegistry, build_movie_project, parse_input_assignments

from .. import bridge
from ..console import logger, style
from ..errors import ErrorCodes, MovoError
from ..parallel import render_video_parallel, resolve_job_count
from ..pipeline import create_session, render_video


def plan_sequence(
    sections: list[dict], bpm: float, beats_per_bar: int = 4, options: dict[str, Any] | None = None
) -> list[dict]:
    """解析した区間を、ムービースキルの並びへ割り当てる。

    区間の数はスキルの並びの数と一致しません（曲によって 2 個のことも 8 個のことも
    ある）。そこで «盛り上がりの順位» で対応づけます。いちばん静かな先頭を intro、
    いちばん盛り上がるところを chorus、という具合です。
    """
    options = options or {}
    bar_seconds = (60 / bpm) * beats_per_bar

    def bars(seconds: float) -> int:
        return max(1, round(seconds / bar_seconds))

    if not sections:
        return []

    # 区間が 3 つ未満のときは «曲の起伏が取れなかった» ということ。ループ素材や
    # 平坦な曲で普通に起きます。1 区間のまま流すと «1 カット 15 小節» という MV に
    # なってしまうので、定型の構成（4 : 8 : 8 : 4）へ割り直します。
    if len(sections) < 3:
        total = bars(sections[-1]["end"] - sections[0]["start"])
        share = [("mv-intro", 4 / 24), ("mv-verse", 8 / 24), ("mv-chorus", 8 / 24), ("mv-outro", 4 / 24)]
        plan = []
        used = 0
        for index, (scene, ratio) in enumerate(share):
            # 端数は最後のシーンで吸収して、合計が曲の長さと合うようにする
            count = max(1, total - used) if index == len(share) - 1 else max(1, round(total * ratio))
            used += count
            plan.append({"scene": scene, "bars": count, "start": sections[0]["start"]})
        return plan

    # ラベルはそのまま使わず、«位置と勢い» から決め直す。解析側のラベルは順位で
    # 付いているだけなので、MV の構成としては据わりが悪いことがある。
    plan: list[dict] = []
    last = len(sections) - 1
    peak = max(range(len(sections)), key=lambda i: sections[i].get("energy", 0))

    # 1 カットの上限。区間がこれより長ければ «同じ種類のカットを続ける» 形に割ります。
    # 96 小節を 1 カットで通す MV は無く、実際には 4〜8 小節で切り替わります。
    max_bars = max(2, round(options.get("maxBars") or 8))

    # «どれくらい激しいシーンに寄せるか»（0 で従来どおり）。勢いの強い A メロを
    # mv-hype に差し替えます。**決め（mv-chorus）とイントロ・アウトロは変えません。**
    # ずっと激しいと «うるさいだけ» になるので、落差を残すのがこの機能の要点です。
    intensity = min(1.0, max(0.0, float(options.get("intensity") or 0)))
    energies = [s.get("energy", 0) for s in sections]
    lowest = min(energies)
    span = max(energies) - lowest

    for index, section in enumerate(sections):
        if index == 0:
            scene = "mv-intro"
        elif index == last and index != peak:
            scene = "mv-outro"
        elif index == peak:
            scene = "mv-chorus"
        else:
            scene = "mv-verse"

        # 正規化した勢いが閾値を超えた A メロだけを激しいシーンに寄せます。
        # intensity が 0 なら «1 より大きい» は起きないので、従来どおりになります。
        if scene == "mv-verse" and intensity > 0:
            normalised = (section.get("energy", 0) - lowest) / span if span > 0 else 0
            if normalised > 1 - intensity:
                scene = "mv-hype"

        remaining = bars(section["end"] - section["start"])
        # イントロとアウトロは切らない。曲の «入りと終わり» は 1 カットで見せる。
        limit = remaining if scene in ("mv-intro", "mv-outro") else max_bars
        offset = 0
        while remaining > 0:
            # 端数が 1 小節だけ余ると忙しないので、最後の 2 カットで分け合う
            take = limit + 1 if remaining - limit == 1 else min(limit, remaining)
            plan.append({"scene": scene, "bars": take, "start": section["start"] + offset * bar_seconds})
            offset += take
            remaining -= take
    return plan


def _distribute_lines(lines: list[str], plan: list[dict]) -> list[list[str]]:
    """歌詞（1 行 1 文）を、A メロ／サビへ配る。

    行数が足りなければ繰り返し、余れば切ります。歌詞を渡さなかったときは
    «曲名だけ» の構成になります。
    """
    if not lines:
        return [[] for _ in plan]
    cursor = 0
    out = []
    for item in plan:
        if item["scene"] in ("mv-intro", "mv-outro"):
            out.append([])
            continue
        # 1 シーンあたり 2〜3 行。小節数から決める。
        want = max(1, min(3, round(item["bars"] / 3)))
        taken = []
        for _ in range(want):
            taken.append(lines[cursor % len(lines)])
            cursor += 1
        out.append(taken)
    return out


#: 役割として認める語尾。`plan_sequence` は `mv-chorus` のような名前を出しますが、
#: ムービースキルによってシーン名の頭は違います（`rich-chorus` など）。
#: **語尾だけを役割として見て、頭はスキルに合わせます。**
ROLE_SUFFIXES = ("intro", "verse", "chorus", "hype", "outro", "bridge", "burst")

#: その役割のシーンがスキルに無いときの代わり。
#:
#: ⚠ **`hype` の代わりに `burst` を先に選んではいけません。** `rich-burst` は
#: 自分の説明に «同じサビが何度も来る曲で «1 回だけ» 差すための枠» と書いてある
#: とおりの «崩し» で、`duotone` と `invert` で色を振り切ります。`--intensity` を
#: 上げたときに `hype` がここへ落ちると、**21 シーン中 13 シーンが反転した紫**に
#: なりました（絵が壊れているようにしか見えません）。盛り上がりの代わりは
#: «大きいサビ» であって «崩し» ではありません。
ROLE_FALLBACKS = {"hype": ("chorus", "burst"), "bridge": ("verse",), "burst": ("chorus",)}


def scene_roles(template: list[dict]) -> dict[str, str]:
    """テンプレートの並びから «役割 → そのスキルでのシーン名» を作る。

    これが無いと `make-mv --style` は **シーン名が `mv-` で始まるスキルにしか
    使えません**。`rich-mv`（`rich-intro` / `rich-chorus` …）を指定すると、
    構成は組めるのに `with` が 1 つも当たらず、色も画像も歌詞も既定値に戻った
    MV が出ていました（8 カット中 5 カットが空でした）。
    """
    roles: dict[str, str] = {}
    for item in template:
        name = item.get("scene") if isinstance(item, dict) else None
        if not isinstance(name, str):
            continue
        role = name.rsplit("-", 1)[-1]
        if role in ROLE_SUFFIXES and role not in roles:
            roles[role] = name
    return roles


def resolve_scene(name: str, roles: dict[str, str]) -> str:
    """`mv-chorus` を、そのスキルで実際に使われているシーン名に直す。"""
    if not roles:
        return name
    role = name.rsplit("-", 1)[-1]
    if role in roles:
        return roles[role]
    for alternative in ROLE_FALLBACKS.get(role, ()):
        if alternative in roles:
            return roles[alternative]
    return name


def distribute_timed_lines(timed: list[dict], plan: list[dict], bar_seconds: float) -> list[list[str]]:
    """**時刻の付いた歌詞**を、カットの時間範囲で切り分ける。

    `_distribute_lines` との違いはここだけです。あちらは «小節数から 2〜3 行» と
    決め打ちで、同じ行を何度も出したり、歌っていないところに歌詞を置いたりします。
    こちらは **そのカットの時間に実際に歌われている行だけ** を渡します。

    範囲に «掛かっている» 行も拾います（`overlap=True`）。歌はカットの切れ目とは
    無関係に続くので、またぐ行を落とすと «サビの 1 行目だけ消える» が起きます。

    イントロ・アウトロにも歌が乗っていれば渡します。**曲が決めることで、
    シーンの種類が決めることではない**からです（歌い出しがイントロの絵に
    重なる MV はいくらでもあります）。

    ⚠ **返すのは文字列ではなく `{text, at, for}` のままです。** スキル側は
    `lyricAt` / `lyricFor` で «その行がシーンの何秒から何秒か» を読みます
    （`movo/skill/template.py` の `LYRIC_FUNCTIONS`）。ここで `line["text"]`
    だけ取り出すと、**時刻を使える唯一の場所の直前で時刻を捨てる**ことになり、
    歌詞は出るのにタイミングだけ等分に戻る、という気付きにくい壊れ方をします。
    `slice_lyrics` はシーンの頭を 0 秒に直して返すので、そのまま渡せます。
    """
    from movo.core import slice_lyrics

    out: list[list[dict]] = []
    for item in plan:
        start = item["start"]
        end = start + item["bars"] * bar_seconds
        out.append(slice_lyrics(timed, start, end, overlap=True))
    return out


def build_sequence(
    plan: list[dict],
    template: list[dict],
    per_scene: list[list] | None = None,
    *,
    authoritative: bool = False,
) -> list[dict]:
    """曲の構成から、ムービースキルの `sequence` を組み直す。

    **もとの並びから «そのシーン種別に何を渡しているか» を借ります。** ここを
    作り直してしまうと、タイトルも色も歌詞もスキルの既定値に戻ります（実際に
    «無題» と «スキルの既定の歌詞» が並ぶ MV を作ってしまいました）。借りる形に
    しておけば、利用者が書いたムービースキルでも同じように動きます。
    """
    per_scene = per_scene or []
    template_for: dict[str, dict] = {}
    for item in template:
        if isinstance(item, dict) and isinstance(item.get("scene"), str) and item["scene"] not in template_for:
            template_for[item["scene"]] = item.get("with") or {}

    # 同じ種類のカットが «何本目か» を数えます。全体の通し番号だと、間に別の種別が
    # 挟まったときに振り分けが偏るので、種別ごとに数えます。
    # 役割（intro / verse / chorus / hype / outro）を、そのスキルのシーン名へ移します。
    roles = scene_roles(template)

    seen: dict[str, int] = {}
    out = []
    for index, item in enumerate(plan):
        scene = resolve_scene(item["scene"], roles)
        nth = seen.get(scene, 0) + 1
        seen[scene] = nth
        per = {**template_for.get(scene, {}), "bars": item["bars"]}
        group = per_scene[index] if index < len(per_scene) else None
        if group:
            # 時刻つきの行（辞書）は **そのまま**。文字列に畳むと時刻が消えます。
            per["lines"] = group if isinstance(group[0], dict) else "\n".join(group)
            # **`hook`（決め文句）を使うシーンは `lines` を見ません。** サビの
            # シーンがまさにそれで、時刻つきの歌詞を渡しているのに «スキルの
            # 既定の決め文句» が 8 小節ぶん出続けていました。そのカットで最初に
            # 歌われる行を決め文句にします。
            if "hook" in per:
                head = group[0]
                per["hook"] = head["text"] if isinstance(head, dict) else head
        elif authoritative:
            # **歌の無いカットには «歌詞なし» を明示します。** 空を «指定なし» と
            # 見なすと、テンプレートの `${lines}` が生き残り、曲全体の歌詞が
            # まるごとそのカットに落ちます（2 小節の間奏に 28 行が詰まって、
            # 1 行 0.09 秒という «読めない上に警告が並ぶ» 状態になりました）。
            per["lines"] = []
        # 同じ種類のカットが続くと «同じ絵が並ぶだけ» になります。文字だけの構成
        # では、歌詞の位置を 1 つおきに変えるだけでも «場面が変わった» と分かります。
        if scene.rsplit("-", 1)[-1] == "verse":
            per["y"] = 180 if nth % 2 == 1 else 280
        out.append({"scene": scene, "with": per})
    return out


def parse_asset_assignments(items: Any) -> dict[str, dict]:
    """`--asset 名前=パス` を素材の宣言にする。

    **これが無いと `make-mv` から画像を 1 枚も使えません。** `rich-mv` は
    `artAsset` / `figureAsset` という入力を持っていますが、素材そのものを
    宣言する手段がどこにも無く、名前を指しても «そんな素材は無い» で
    終わっていました（入力だけあって配線が無い状態でした）。

    種類は拡張子から決めます。書き分けさせるほどの情報量が無いためです。
    """
    out: dict[str, dict] = {}
    if items is None:
        return out
    values = items if isinstance(items, list) else [items]
    for item in values:
        text = str(item)
        index = text.find("=")
        if index < 0:
            raise MovoError(
                ErrorCodes.MOVO_CLI_USAGE,
                f'--asset は 名前=パス の形式で指定してください: "{text}"',
                hint="例: --asset art=assets/free/susuki.png --asset figure=assets/free/tree.png",
            )
        name = text[:index].strip()
        path = Path(text[index + 1:].strip())
        if not path.exists():
            raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"素材が見つかりません: {path}")
        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            kind = "image"
        elif suffix in (".mp4", ".webm", ".mov"):
            kind = "video"
        elif suffix in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            kind = "audio"
        elif suffix in (".ttf", ".otf"):
            kind = "font"
        elif suffix in (".lrc", ".srt", ".vtt"):
            kind = "lyrics"
        else:
            kind = "data"
        out[name] = {"type": kind, "path": str(path.resolve())}
    return out


def _read_timed_lyrics(path: Any) -> list[dict] | None:
    """`--lyrics <.lrc|.srt|.vtt|.json>` を読む。指定が無ければ ``None``。"""
    if not path:
        return None
    from movo.core import parse_lyrics

    file = Path(str(path))
    if not file.exists():
        raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"歌詞ファイルが見つかりません: {path}")
    return parse_lyrics(file.read_text(encoding="utf-8"), file=str(file))


def make_mv_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    audio_path = positional[0] if positional else None
    if not audio_path:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "movo make-mv <音声ファイル> [オプション]",
            hint="例: movo make-mv song.wav --title 夜明けまで -o tmp/mv.mp4",
        )
    absolute = Path(audio_path).resolve()
    if not absolute.exists():
        raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"音声ファイルが見つかりません: {audio_path}")

    # ── 1. 曲を調べる ──────────────────────────────────────────
    logger.info(f"{style.bold(absolute.name)} を解析しています…")
    decode_audio_file = bridge.pick("movo.audio", "decode_audio_file", "decodeAudioFile")
    analyze_audio = bridge.pick("movo.audio", "analyze_audio", "analyzeAudio")
    beats_per_bar = int(options.get("beatsPerBar") or 4)
    audio = decode_audio_file(str(absolute))
    settings: dict[str, Any] = {"beatsPerBar": beats_per_bar}
    if options.get("minBpm"):
        settings["minBpm"] = options["minBpm"]
    if options.get("maxBpm"):
        settings["maxBpm"] = options["maxBpm"]
    analysis = analyze_audio(audio, settings)
    bpm = round(analysis["bpm"] * 100) / 100
    logger.info(
        f'  BPM {bpm}（確からしさ {analysis["confidence"]:.2f}） / '
        f'{len(analysis["sections"])} 区間 / {analysis["duration"]:.1f} 秒'
    )
    if analysis["confidence"] < 0.35:
        logger.warn("  BPM の確からしさが低めです。ずれていたら --min-bpm / --max-bpm で範囲を絞ってください")

    # ── 2. 構成を組む ──────────────────────────────────────────
    plan = plan_sequence(
        analysis["sections"],
        bpm,
        beats_per_bar,
        {"maxBars": options.get("maxBars"), "intensity": options.get("intensity")},
    )
    lines = [line.strip() for line in str(options.get("lines") or "").replace("\\n", "\n").split("\n") if line.strip()]

    # **時刻付きの歌詞があれば、そちらが優先です。** 小節数から機械的に配るのは
    # «いつ歌われるか» を知らないときの次善策でしかありません。
    # .lrc は `movo lyrics align` で作れます。
    timed = _read_timed_lyrics(options.get("lyrics"))
    if timed:
        bar_seconds = (60 / bpm) * beats_per_bar
        per_scene = distribute_timed_lines(timed, plan, bar_seconds)
        if not lines:
            lines = [row["text"] for row in timed]
        placed = sum(1 for group in per_scene if group)
        logger.info(f"  時刻付きの歌詞 {len(timed)} 行を、{placed} カットへ実時間で割り当てました")
    else:
        per_scene = _distribute_lines(lines, plan)

    logger.info("  構成")
    for index, item in enumerate(plan):
        logger.info(f'    {str(index + 1).rjust(2)}. {item["scene"].ljust(10)} {str(item["bars"]).rjust(3)} 小節')

    # ── 3. ムービースキルへ流し込む ────────────────────────────
    registry = SkillRegistry().load(project_root=os.getcwd())
    style_name = options.get("style") or "lyric-mv"
    entry = registry.movie(style_name)
    if entry is None:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'ムービースキル "{style_name}" が見つかりません',
            hint="一覧は movo skill list --movies",
        )

    given = parse_input_assignments(options.get("set"))
    inputs = {
        **given,
        "title": options.get("title") or absolute.stem,
        "bpm": bpm,
    }
    if lines:
        inputs["lines"] = "\n".join(lines)
        # **歌詞を受け取る入力は、スキルによって名前が違います**
        # （`lines` / `hypeLines` / `verseLines` / `chorusLines` …）。`lines` だけ
        # 埋めると、他の名前はスキルの既定のまま残り、渡した歌詞ではなく
        # «ぐるぐる まわる» が出ます（実際に出ました）。カットごとの指定が
        # 勝つので、ここは «行き場のない既定を潰す» ためのものです。
        for key, definition in (entry["definition"].get("inputs") or {}).items():
            if key in given or not isinstance(definition, dict):
                continue
            if definition.get("type") == "textList":
                inputs[key] = "\n".join(lines)
        # 決め文句は «いちばん多く繰り返される行»。サビはそれゆえサビです。
        if "hook" not in given:
            counted = Counter(lines)
            top, times = counted.most_common(1)[0]
            if times > 1:
                inputs["hook"] = top

    sequence = build_sequence(
        plan, entry["definition"].get("sequence") or [], per_scene, authoritative=bool(timed)
    )

    # スキル定義を «この曲用» に差し替えた一時的な写しを作る。
    # 元の定義には触らない（ほかの呼び出しに影響しないように）。
    tailored = {
        **entry["definition"],
        "sequence": sequence,
        "project": {**(entry["definition"].get("project") or {}), "bpm": bpm},
        "assets": {
            **(entry["definition"].get("assets") or {}),
            **parse_asset_assignments(options.get("asset")),
            "_track": {"type": "audio", "path": str(absolute)},
        },
        "audio": [{"asset": "_track", "volume": 0.9, "fadeOut": 2}],
    }
    registry.register(tailored, kind="movie", source="make-mv")

    built = build_movie_project(
        registry,
        style_name,
        inputs,
        {
            "width": options.get("width"),
            "height": options.get("height"),
            "fps": options.get("fps"),
            "quality": options.get("quality"),
            "name": inputs["title"],
        },
    )

    output = options.get("output") or str(Path("tmp") / f"{absolute.stem}.mp4")
    session = create_session(
        output,
        {
            "inline_project": built["project"],
            "project_root": os.getcwd(),
            "quality": options.get("quality"),
            "skill_registry": registry,
        },
    )
    # MV は長くなりがちなので、**ここでこそ `--jobs` が効きます**。
    jobs = resolve_job_count(options.get("jobs"))
    render = render_video_parallel if jobs > 1 else render_video
    result = render(
        session, {"output": output, "format": options.get("format"), "quiet": options.get("quiet"), "jobs": jobs, "cli_options": options}
    )
    logger.success(f'{result["frames"]} フレーム → {result["path"]}')
    return result
