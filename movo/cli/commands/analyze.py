"""`movo analyze <音声ファイル>` — BPM・拍・小節・区間を調べる。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError

# 人が読む表示で並べる拍の数。全部出すと数千行になって端末が流れてしまいます。
PREVIEW_BEATS = 8


def analyze_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    file = positional[0] if positional else None
    if not file:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "movo analyze には音声ファイルが要ります",
            hint="使い方: movo analyze track.wav [--json]",
        )

    decode_audio_file = bridge.pick("movo.audio", "decode_audio_file", "decodeAudioFile")
    analyze_audio = bridge.pick("movo.audio", "analyze_audio", "analyzeAudio")

    audio = decode_audio_file(str(Path(file).resolve()))
    # 設定は **JS 版と同じ綴りの辞書** で渡します。None は «指定なし» なので落とします
    # （渡すと既定値が打ち消され、探索範囲が消えます）。
    settings = {
        key: value
        for key, value in (
            ("minBpm", _positive(options.get("minBpm"))),
            ("maxBpm", _positive(options.get("maxBpm"))),
            ("beatsPerBar", _positive(options.get("beatsPerBar"))),
        )
        if value is not None
    }
    result = analyze_audio(audio, settings)

    if options.get("json"):
        say(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return result

    logger.info(style.bold(Path(file).name))
    logger.info("")
    logger.info(f'  長さ       : {result["duration"]:.2f} 秒 / {result["sampleRate"]} Hz')
    if result["bpm"] <= 0:
        logger.warn("  拍が見つかりませんでした（無音か、拍のない音源かもしれません）")
        return result
    bpm_text = style.bold(f'{result["bpm"]:.2f}')
    logger.info(f'  BPM        : {bpm_text}  （確からしさ {result["confidence"]:.2f}）')
    logger.info(f'  1 拍目     : {result["firstBeat"]:.3f} 秒')
    logger.info(
        f'  拍 / 小節  : {len(result["beats"])} 拍 / {len(result["bars"])} 小節（{result["beatsPerBar"]} 拍子）'
    )
    head = ", ".join(f"{t:.3f}" for t in result["beats"][:PREVIEW_BEATS])
    more = ", …" if len(result["beats"]) > PREVIEW_BEATS else ""
    logger.info(f"  最初の拍   : {head}{more}")
    logger.info("")
    logger.info(style.bold(f'  区間 {len(result["sections"])} 件'))
    for section in result["sections"]:
        length = section["end"] - section["start"]
        logger.info(
            f'    {section["start"]:7.2f} 〜 {section["end"]:7.2f} 秒  '
            f'{str(section["label"]).ljust(7)} 勢い {section["energy"]:.2f}  '
            f'({length:.1f} 秒 / {section["bars"]} 小節)'
        )
    logger.info("")
    if result["confidence"] < 0.4:
        logger.warn("  確からしさが低めです。--min-bpm / --max-bpm で範囲を絞ると当たることがあります。")
    usage = 'プロジェクトから使うには: "project": { "bpm": { "fromAudio": "<素材名>" } }'
    logger.info("  " + style.gray(usage))
    return result


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
