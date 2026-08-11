"""`movo profile` — 映像を数値にする。
`movo compare` — 測った数値を目標と突き合わせ、直し方まで出す。

真似て作った映像が «どこがどう足りないか» を、目で見て探すのをやめるための
コマンドです。プロジェクト JSON はその場で描いて測り、mp4 などは ffmpeg で
生フレームに開いて測ります。

**結果は `say`（stdout）から出します。** `logger.info` はログ水準で止まるので、
`--quiet` を付けると結果まで消えます。`--quiet` が黙らせたいのは進捗だけです。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import create_session


def _profile_project(file: str, options: dict[str, Any]) -> dict:
    """プロジェクト JSON を描きながら測る。

    書き出し済みの動画が無くても測れるので、作り込みの途中で回せます。
    """
    session = create_session(file, {"quality": options.get("quality") or "draft"})
    timeline = session["timeline"]
    fps = timeline["fps"]
    total = max(1, round(timeline["duration"] * fps))
    profiler = bridge.pick("movo.core.video_profile", "VideoProfiler")(
        fps=fps, width=timeline["width"], height=timeline["height"]
    )
    progress = None if options.get("quiet") else logger.progress(total, "profile")
    for frame in range(total):
        profiler.push(session["renderer"].render_frame(frame))
        if progress is not None:
            progress.update(frame + 1)
    if progress is not None:
        progress.done(f"{total} フレームを測りました")
    return profiler.report()


def _profile_video(file: str, options: dict[str, Any]) -> dict:
    """書き出し済みの動画を測る。

    ffmpeg に生の RGBA を吐かせて 1 フレームずつ読みます。自前のデコーダを書く
    ほどの用途ではないので、ここは ffmpeg に任せます。
    """
    ffmpeg = bridge.find_ffmpeg()
    if not ffmpeg:
        raise MovoError(
            ErrorCodes.MOVO_FFMPEG_NOT_FOUND,
            "動画を読むには ffmpeg が必要です",
            hint="プロジェクト JSON を渡せば ffmpeg なしで測れます",
        )
    # 測るのに原寸は要らない。横 320 に落とすと速く、指標はほとんど変わりません。
    width = int(options.get("width") or 320)
    fps = int(options.get("fps") or 24)
    # 高さは «起動する前に» 決めます。あとから決めると、動画全体が 1 チャンクで
    # 届いたときに 1 フレームも取り込めないまま終わります。取れなければ 16:9 と仮定。
    height = int(options.get("height") or 0) or _resolve_height(file, width)
    frame_bytes = width * height * 4

    profiler = bridge.pick("movo.core.video_profile", "VideoProfiler")(fps=fps, width=width, height=height)

    args = [
        ffmpeg["path"],
        "-v", "error",
        "-i", file,
        "-vf", f"fps={fps},scale={width}:-2",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "pipe:1",
    ]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = 0
    pending = b""
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(frame_bytes)
        if not chunk:
            break
        pending += chunk
        while len(pending) >= frame_bytes:
            profiler.push(bridge.to_bitmap(width, height, pending[:frame_bytes]))
            pending = pending[frame_bytes:]
            frames += 1
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    process.wait()
    if frames == 0:
        tail = " ".join(stderr.strip().split("\n")[-2:])
        raise MovoError(
            ErrorCodes.MOVO_INTERNAL,
            "動画からフレームを読めませんでした" if process.returncode == 0 else f"ffmpeg が失敗しました: {tail}",
        )
    return profiler.report()


def _resolve_height(file: str, width: int) -> int:
    """出力する高さ（幅を固定したときの偶数丸め）を求める。"""
    probe = bridge.find_ffprobe()
    if probe:
        completed = subprocess.run(
            [
                probe["path"],
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                file,
            ],
            capture_output=True,
            text=True,
        )
        try:
            w, h = (int(float(v)) for v in (completed.stdout or "").strip().split(",")[:2])
            if w and h:
                return round(width * h / w / 2) * 2
        except (TypeError, ValueError):
            pass
    return round(width * 9 / 16 / 2) * 2


def profile_of(file: str, options: dict[str, Any] | None = None) -> dict:
    """プロジェクトでも動画でも測れるようにする窓口。"""
    options = options or {}
    if not Path(file).exists():
        raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"ファイルが見つかりません: {file}")
    return _profile_project(file, options) if file.lower().endswith(".json") else _profile_video(file, options)


def profile_command(positional: list[str], options: dict[str, Any]) -> dict:
    file = positional[0] if positional else None
    if not file:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "movo profile <動画 または プロジェクト>",
            hint="例: movo profile tmp/mv/01.mp4 --json",
        )
    profile = profile_of(file, options)
    if options.get("json"):
        say(json.dumps(profile, ensure_ascii=False, indent=2, default=str))
        return profile

    say("")
    say(
        f'{style.bold(Path(file).name)}  {profile["width"]}x{profile["height"]} @ {profile["fps"]}fps  '
        f'{profile["seconds"]:.2f} 秒'
    )
    say("")
    cuts, motion, palette, detail = profile["cuts"], profile["motion"], profile["palette"], profile["detail"]
    say(f'  カット      {cuts["count"]} 本 / 中央値 {cuts["medianSeconds"]} 秒 / 毎分 {cuts["perMinute"]} 本')
    say(
        f'  動きの量    {motion["mean"]}（最大 {motion["peak"]} / '
        f'止まっている割合 {motion["stillRatio"] * 100:.1f}%）'
    )
    say(
        f'  色          実質 {palette["effectiveColors"]} 色 / 彩度 {palette["saturation"]} / '
        f'明度 {palette["brightness"]} / コントラスト {palette["contrast"]}'
    )
    say(f'  支配色      {" ".join(palette["dominant"][:5])}')
    say(f'  細かさ      {detail["edgeDensity"]}（文字や模様の多さ）')
    say("")
    return profile


def compare_command(positional: list[str], options: dict[str, Any]) -> dict:
    mine = positional[0] if positional else None
    if not mine:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "movo compare <自分の映像> [相手の映像] [--target <目標.json>]",
            hint="例: movo compare tmp/mv/01.mp4 --target profiles/handdrawn-punk.json",
        )
    other = positional[1] if len(positional) > 1 else None
    if not other and not options.get("target"):
        raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "相手の映像か --target のどちらかが要ります")

    profile = profile_of(mine, {**options, "quiet": True})
    if options.get("target"):
        # 名前（同梱のスタイル）でもファイルパスでも受ける
        load_target = bridge.pick("movo.core.profile_library", "load_profile_target", "loadProfileTarget")
        compare_profile = bridge.pick("movo.core.video_compare", "compare_profile", "compareProfile")
        entry = load_target(options["target"], str(Path(mine).resolve().parent))
        comparison = compare_profile(profile, entry["target"])
        label = f'{entry["name"]}（{entry["label"]}）' if entry.get("label") else entry["name"]
    else:
        compare_to_reference = bridge.pick("movo.core.video_compare", "compare_to_reference", "compareToReference")
        reference = profile_of(other, {**options, "quiet": True})
        comparison = compare_to_reference(profile, reference, options.get("tolerance"))
        label = Path(other).name

    if options.get("json"):
        say(json.dumps({"profile": profile, "comparison": comparison}, ensure_ascii=False, indent=2, default=str))
        return comparison

    describe = bridge.pick("movo.core.video_compare", "describe_comparison", "describeComparison")
    say("")
    say(f"{style.bold(Path(mine).name)}  ←→  {label}")
    say("")
    for line in describe(comparison):
        say(line)
    say("")
    if comparison.get("ok"):
        logger.success("目標の範囲に収まっています")
    else:
        off = len([r for r in comparison.get("results", []) if r.get("status") != "ok"])
        logger.warn(f"{off} 項目が目標から外れています")
    return comparison
