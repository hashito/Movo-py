"""動画レイヤーの «素材» を読むところ。

移植元: ``packages/renderer/src/video.js``

フレームは ffmpeg で **要るぶんだけ** 取り出して覚えます。1 フレームだけ見たい
（`movo frame`）ときに動画全体をデコードしないためです。ffmpeg が無い環境では
**動画が出ないことを 1 回だけ言って飛ばします**（仕様の原則 9 — 一部が
描けなくても、描けるところまでは描く）。

## Python 版で変えたところ

- ``spawnSync`` → ``subprocess.run``。Windows で黒い窓が出ないよう
  ``creationflags`` を渡します（並列レンダリングでは子プロセスの数だけ
  窓が開いてしまうため）。
- キャッシュの鍵は JS 版と同じ «パス + 大きさ + 更新時刻 + fps + 番号» です。
  **更新時刻を入れておかないと、素材を差し替えたのに古いフレームが出ます。**
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from movo.cli.console import logger
from movo.core.hash import short_hash
from movo.core.math import js_round
from movo.core.platform import find_ffmpeg, find_ffprobe
from movo.core.png import decode_png

#: 覚えておくフレーム数。1920x1080 で 1 枚 8MB なので、48 枚で 400MB 程度。
MEMORY_LIMIT = 48

# Windows で子プロセスの窓を出さない。他の OS では 0（無効）になります。
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class VideoSource:
    """1 本の動画ファイルから «その時刻のフレーム» を取り出す。

    :param file_path: 動画ファイル
    :param cache: ``movo.core.cache.Cache``（あれば取り出したフレームを残す）
    :param fps: 出力の fps。時刻をフレーム番号に直すのに使います
    """

    def __init__(self, file_path: str, cache=None, fps: float = 30) -> None:
        self.file_path = str(file_path)
        self.cache = cache
        self.fps = fps or 30
        self.frames: dict[int, object] = {}
        self.order: list[int] = []
        self.available = bool(find_ffmpeg())
        self.duration = _probe_duration(self.file_path) if self.available else None
        self._warned = False

    def frame_at(self, time: float):
        """`time` 秒のフレーム。取り出せなければ ``None``。"""
        if not self.available:
            if not self._warned:
                self._warned = True
                logger.warn(
                    f"動画レイヤーを読むには ffmpeg が要ります。"
                    f'"{Path(self.file_path).name}" は飛ばします'
                )
            return None
        index = max(0, js_round(time * self.fps))
        if index in self.frames:
            return self.frames[index]

        key = short_hash(f"{self.file_path}:{_file_stamp(self.file_path)}:{self.fps}:{index}")
        buffer = self.cache.read_buffer("video-frames", key, ".png") if self.cache else None
        if not buffer:
            buffer = _extract_frame(self.file_path, index / self.fps)
            if buffer and self.cache:
                self.cache.write_buffer("video-frames", key, buffer, ".png")
        if not buffer:
            return None
        bitmap = decode_png(buffer)
        self.frames[index] = bitmap
        self.order.append(index)
        while len(self.order) > MEMORY_LIMIT:
            self.frames.pop(self.order.pop(0), None)
        return bitmap


def _file_stamp(file_path: str) -> str:
    """素材の «版»。差し替えたら別のフレームだと分かるようにするためのもの。"""
    try:
        stat = os.stat(file_path)
        return f"{stat.st_size}:{js_round(stat.st_mtime * 1000)}"
    except OSError:
        return "0"


def _probe_duration(file_path: str) -> float | None:
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe["path"], "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True, text=True, creationflags=_NO_WINDOW,
        )
        value = float((result.stdout or "").strip())
        return value if value > 0 else None
    except (OSError, ValueError):
        return None


def _extract_frame(file_path: str, time: float) -> bytes | None:
    """`time` 秒の 1 枚を PNG で取り出す。

    ``-ss`` を ``-i`` の **前** に置くのは、そこまでのフレームをデコードせずに
    飛ばすためです（後ろに置くと頭から全部読むので、長い素材で桁違いに遅い）。
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    directory = tempfile.mkdtemp(prefix="movo-vf-")
    output = Path(directory) / "frame.png"
    try:
        result = subprocess.run(
            [
                ffmpeg["path"], "-y", "-loglevel", "error",
                "-ss", str(max(0.0, time)),
                "-i", file_path,
                "-frames:v", "1", str(output),
            ],
            capture_output=True, creationflags=_NO_WINDOW,
        )
        if result.returncode == 0 and output.is_file():
            return output.read_bytes()
        return None
    except OSError:
        return None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


__all__ = ["MEMORY_LIMIT", "VideoSource"]
