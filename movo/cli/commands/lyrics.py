"""`movo lyrics align <音声ファイル>` — 時刻の無い歌詞に «下書きの時刻» を付ける。

**完全自動を名乗りません。** 出すのは下書きで、直す前提です。そのかわり
«どこが怪しいか» を必ず一緒に出します（`--json` の `needsCheck`）。

直し方は 1 つだけ覚えれば済みます。**`--anchor <行番号>=<秒>` で数点留める**と、
そのあいだが配分し直されます。28 行を全部打つのに比べて、3〜5 点で済みます。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError

SUBCOMMANDS = ("align",)


def lyrics_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    sub = positional[0] if positional else None
    if sub not in SUBCOMMANDS:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            f'movo lyrics <{" | ".join(SUBCOMMANDS)}> <音声ファイル> [オプション]',
            hint="例: movo lyrics align song.mp3 --text lyrics.txt -o song.lrc",
        )
    return _align(positional[1:], options)


def _align(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    audio_path = positional[0] if positional else None
    if not audio_path:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "movo lyrics align <音声ファイル> --text <歌詞ファイル>",
            hint="例: movo lyrics align song.mp3 --text lyrics.txt -o song.lrc",
        )
    absolute = Path(audio_path).resolve()
    if not absolute.exists():
        raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"音声ファイルが見つかりません: {audio_path}")

    text = _read_lyrics(options)

    decode_audio_file = bridge.pick("movo.audio", "decode_audio_file", "decodeAudioFile")
    align_lyrics = bridge.pick("movo.audio", "align_lyrics", "alignLyrics")
    to_lrc = bridge.pick("movo.audio", "to_lrc", "toLrc")
    to_scenario = bridge.pick("movo.audio", "to_scenario", "toScenario")

    logger.info(f"{style.bold(absolute.name)} と歌詞を突き合わせています…")
    audio = decode_audio_file(str(absolute))
    result = align_lyrics(audio, text, {
        "anchors": options.get("anchor"),
        "beatsPerBar": options.get("beatsPerBar"),
        "minBpm": options.get("minBpm"),
        "maxBpm": options.get("maxBpm"),
        "snap": options.get("snap", True),
        "gaps": options.get("gaps", True),
        "start": options.get("start"),
        "end": options.get("end"),
    })

    scenario = to_scenario(result)
    if options.get("json"):
        say(json.dumps(scenario, ensure_ascii=False, indent=2, default=str))
    else:
        _report(result, scenario)

    output = options.get("output")
    if output:
        lrc = to_lrc(result["lines"], meta={"ti": options.get("title") or absolute.stem})
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(lrc, encoding="utf-8")
        logger.success(f'{len(result["lines"])} 行 → {target}')

    scenario_path = options.get("scenario")
    if scenario_path:
        target = Path(scenario_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success(f"シナリオの下書き → {target}")

    return result


def _read_lyrics(options: dict[str, Any]) -> str:
    """`--text <ファイル>` か `--lines "…"` を受ける。

    ファイルを既定にしているのは、歌詞は改行が意味を持つからです
    （空行がブロックの区切りになり、そこからサビを見つけています）。
    """
    file = options.get("text")
    if file:
        path = Path(file)
        if not path.exists():
            raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"歌詞ファイルが見つかりません: {file}")
        return path.read_text(encoding="utf-8")
    inline = options.get("lines")
    if inline:
        return str(inline).replace("\\n", "\n")
    raise MovoError(
        ErrorCodes.MOVO_CLI_USAGE,
        "歌詞がありません",
        hint='--text lyrics.txt か --lines "1 行目\\n2 行目" を渡してください',
    )


def _report(result: dict[str, Any], scenario: dict[str, Any]) -> None:
    logger.info("")
    logger.info(
        f'  BPM {result["bpm"]:.2f}（確からしさ {result["confidence"]:.2f}） / '
        f'{result["duration"]:.1f} 秒 / 1 小節 {scenario["barSeconds"]:.2f} 秒'
    )
    logger.info("")

    logger.info(style.bold(f'  ブロック {len(scenario["blocks"])} 個'))
    for block in scenario["blocks"]:
        kind = "サビ" if block["kind"] == "chorus" else "A メロ"
        repeat = f'（{block["repeatOf"] + 1} 番目の繰り返し）' if block["repeatOf"] is not None else ""
        logger.info(
            f'    {block["start"]:7.2f} 〜 {block["end"]:7.2f} 秒  {kind.ljust(6)}'
            f'{str(block["bars"]).rjust(5)} 小節  確からしさ {block["confidence"]:.2f}{repeat}'
        )
        logger.info(f'      {style.gray(block["lines"][0][:34])}')

    if scenario["instrumental"]:
        logger.info("")
        logger.info(style.bold("  歌の無いところ"))
        for gap in scenario["instrumental"]:
            logger.info(f'    {gap["start"]:7.2f} 〜 {gap["end"]:7.2f} 秒  {gap["kind"]}')

    if scenario["needsCheck"]:
        logger.info("")
        logger.info(style.bold(f'  要確認 {len(scenario["needsCheck"])} 行'))
        for row in scenario["needsCheck"]:
            logger.info(f'    {str(row["line"]).rjust(3)} 行目  {row["at"]:7.2f} 秒  {row["text"][:28]}')

    logger.info("")
    for warning in result["warnings"]:
        logger.warn("  " + warning)
    logger.info("")
    logger.info("  " + style.gray("直すときは --anchor <行番号>=<秒> を 2〜3 点。あいだは配分し直されます"))
