"""movo-exporters — 描いたフレームをファイルにする。

書き出し口はどれも同じ小さな面を持ちます。

    begin() → write_frame(bitmap, index) → end()

`mp4` / `webm` / `mov` は ffmpeg が要ります。`png-sequence` と `gif` は
自前実装なので、**ffmpeg が無くても «見られるもの» は必ず出せます**。

## JS 版の不具合をここで直しています（Movo issue #78）

JS 版は **ffmpeg が落ちても終了コード 0 で静かに終わる**ことがありました。
原因は 3 つあって、Python 版では最初から潰してあります。

1. **`close` の code が `null` のとき 0 として扱っていた。**
   シグナルで殺されたとき（メモリ不足で落とされる、など）code は `null` に
   なります。JS 版は `code ?? 0` と書いていたので «正常終了» になり、
   **0 バイトの mp4 ができて成功と表示されました**。Python 版は
   `returncode` が 0 でなければ必ず例外にします。負の値（シグナル）も
   «落ちた» として扱います。

2. **パイプの `EPIPE` を握りつぶしていた。** ffmpeg が先に死ぬと
   `stdin.write` が EPIPE になります。JS 版はそれを `debug` に落として
   捨てていたので、**残りのフレームを «書いたつもり» で捨て続けました**。
   Python 版は `BrokenPipeError` を捕まえたらすぐ ffmpeg の標準エラーを読み、
   何が起きたかを添えて例外にします。

3. **出来たファイルを確かめていなかった。** 書き出しの最後に
   **実在・大きさ・尺**を検査します（`verify_output`）。ffprobe があれば
   尺も見て、**要求した長さと 0.5 秒以上ずれていたら失敗**にします。
   «途中で切れた動画» が «成功» として残るのがいちばん困ります。

標準エラーは **別スレッドで読み続けます**。ffmpeg は進捗を標準エラーに書くので、
読まずに放っておくとパイプのバッファが詰まり、こちらの `write` と ffmpeg の
`write` が互いを待って**固まります**（JS 版はイベントで読めていたので
起きませんでした。Python では自分で読む必要があります）。
"""

from __future__ import annotations

import os
import re
import subprocess
import threading

from movo.cli.console import logger
from movo.core.errors import ErrorCodes, MovoError
from movo.core.platform import find_ffmpeg, find_ffprobe
from movo.core.png import encode_png
from movo.core.wav import encode_wav

from .gif import build_palette, encode_gif

# **ログは `movo.cli.console` のものを使います。** `movo.core.logger` にも同じ形の
# logger がありますが、`--verbose` / `--quiet` が水準を変えているのは console 側
# だけです（console は core があればそちらへ乗り換えるつもりで書かれていますが、
# `movo.core` が logger を «実体» として再輸出しているため乗り換えは起きません）。
# core 側を使うと、書き出しの警告だけが `--quiet` で消えなくなります。

EXPORT_FORMATS = ["mp4", "webm", "mov", "gif", "png-sequence", "wav"]

FFMPEG_FORMATS = ("mp4", "webm", "mov")

# GIF は全フレームを覚えてから 1 枚にします。上限を切らないと 3 分の MV で
# メモリが数 GB になります。
GIF_FRAME_LIMIT = 900

# 書き出した尺がこれ以上ずれていたら «途中で切れた» とみなします（秒）。
DURATION_TOLERANCE = 0.5

_NO_WINDOW = {"creationflags": 0x08000000} if os.name == "nt" else {}


def _verbose_enabled() -> bool:
    """`--verbose` が効いているか（ffmpeg の出力を見せるかどうか）。

    **水準の «数の向き» が実装によって逆**なので、数を直に比べません
    （`movo.core.logger` は debug が大きく、`movo.cli.console` は小さい）。
    自分で判定を持っているならそれに従い、無ければ console の表で比べます。
    """
    should = getattr(logger, "_should", None)
    if callable(should):
        try:
            return bool(should("verbose"))
        except Exception:
            return False
    from movo.cli.console import LEVELS

    return getattr(logger, "level", LEVELS["info"]) <= LEVELS["verbose"]


class BaseSink:
    """書き出し口の共通部分。"""

    def __init__(self, options: dict) -> None:
        self.options = options
        self.width = options.get("width")
        self.height = options.get("height")
        self.fps = options.get("fps")
        self.output_path = options.get("outputPath")
        self.frame_count = 0

    def begin(self) -> None:
        pass

    def write_frame(self, bitmap, index: int | None = None) -> None:
        self.frame_count += 1

    def end(self) -> dict:
        return {"path": self.output_path, "frames": self.frame_count}


class FfmpegSink(BaseSink):
    """生の RGBA を ffmpeg に流し込む。"""

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.ffmpeg = find_ffmpeg()
        if not self.ffmpeg:
            raise MovoError(
                ErrorCodes.MOVO_FFMPEG_NOT_FOUND,
                "ffmpeg がこの環境にありません",
                hint="ffmpeg を入れて PATH に通すか、MOVO_FFMPEG=/path/to/ffmpeg を設定するか、"
                     "--format gif / png-sequence で書き出してください",
            )
        self.process = None
        self.stderr_text = ""
        self._stderr_thread = None

    def _args(self) -> list[str]:
        output = self.options.get("output") or {}
        fmt = self.options.get("format", "mp4")
        args = [
            "-y",
            # CLI は verbose をここへ渡してこないので、ログの水準から拾います
            "-loglevel", "info" if (self.options.get("verbose") or _verbose_enabled()) else "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
        ]
        if self.options.get("audioPath"):
            args += ["-i", self.options["audioPath"]]

        codec = output.get("codec") or ("vp9" if fmt == "webm" else "h264")
        video_codec = {
            "h264": "libx264", "h265": "libx265", "hevc": "libx265",
            "vp9": "libvpx-vp9", "vp8": "libvpx", "prores": "prores_ks",
        }.get(codec, codec)
        args += ["-c:v", video_codec]
        if output.get("pixelFormat"):
            args += ["-pix_fmt", output["pixelFormat"]]
        elif video_codec in ("libx264", "libx265"):
            args += ["-pix_fmt", "yuv420p"]
        if output.get("crf") is not None:
            args += ["-crf", str(output["crf"])]
        elif video_codec in ("libx264", "libx265"):
            args += ["-crf", "18"]
        if output.get("bitrate"):
            args += ["-b:v", output["bitrate"]]
        if output.get("preset"):
            args += ["-preset", output["preset"]]
        if video_codec in ("libx264", "libx265"):
            args += ["-movflags", "+faststart"]

        if self.options.get("audioPath"):
            args += ["-c:a", output.get("audioCodec") or ("libopus" if fmt == "webm" else "aac")]
            if output.get("audioBitrate"):
                args += ["-b:a", output["audioBitrate"]]
            args += ["-shortest"]
        args.append(self.output_path)
        return args

    def begin(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)) or ".", exist_ok=True)
        args = self._args()
        logger.verbose(f"ffmpeg {' '.join(args)}")
        try:
            self.process = subprocess.Popen(
                [self.ffmpeg["path"], *args],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                **_NO_WINDOW,
            )
        except OSError as error:
            # **起動に失敗したら、そこで止めます。**
            raise MovoError(
                ErrorCodes.MOVO_RENDERER_UNAVAILABLE,
                f"ffmpeg を起動できませんでした: {error}",
                hint=f"実行ファイル: {self.ffmpeg['path']}",
                cause=error,
            ) from error

        # 標準エラーは «読み続けないと詰まる»。別スレッドで吸い出します。
        def drain() -> None:
            chunks: list[bytes] = []
            size = 0
            for line in self.process.stderr:
                chunks.append(line)
                size += len(line)
                while size > 20000 and len(chunks) > 1:
                    size -= len(chunks.pop(0))
            self.stderr_text = b"".join(chunks).decode("utf-8", "replace")

        self._stderr_thread = threading.Thread(target=drain, daemon=True)
        self._stderr_thread.start()

    def write_frame(self, bitmap, index: int | None = None) -> None:
        if self.process is None:
            raise MovoError(ErrorCodes.MOVO_INTERNAL, "begin() を呼ぶ前に write_frame() が呼ばれました")
        self.frame_count += 1
        try:
            self.process.stdin.write(bitmap.data.tobytes())
        except (BrokenPipeError, OSError) as error:
            # **握りつぶしません。** ffmpeg が先に死んでいます。
            self._fail_from_pipe(error)

    def _fail_from_pipe(self, error: BaseException) -> None:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        code = self.process.wait()
        if self._stderr_thread:
            self._stderr_thread.join(timeout=5)
        raise MovoError(
            ErrorCodes.MOVO_RENDERER_UNAVAILABLE,
            f"ffmpeg が {self.frame_count} フレーム目で終了しました（終了コード {code}）",
            hint=self._stderr_tail() or str(error),
            cause=error,
        ) from error

    def _stderr_tail(self) -> str:
        lines = [line for line in re.split(r"\r?\n", self.stderr_text) if line]
        return "\n".join(lines[-6:])

    def end(self) -> dict:
        if self.process is None:
            raise MovoError(ErrorCodes.MOVO_INTERNAL, "begin() を呼ばずに end() が呼ばれました")
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        code = self.process.wait()
        if self._stderr_thread:
            self._stderr_thread.join(timeout=5)
        if code != 0:
            # 負の値は «シグナルで殺された»。JS 版はここを 0 として扱っていました。
            reason = (
                f"ffmpeg はシグナル {-code} で落ちました"
                if code < 0
                else f"ffmpeg が終了コード {code} で終わりました"
            )
            raise MovoError(
                ErrorCodes.MOVO_RENDERER_UNAVAILABLE, reason,
                file=self.output_path,
                hint=self._stderr_tail() or "ffmpeg の出力がありません",
            )
        verify_output(
            self.output_path,
            expected_duration=self.options.get("expectedDuration"),
            frames=self.frame_count,
            fps=self.fps,
        )
        return {"path": self.output_path, "frames": self.frame_count}


class PngSequenceSink(BaseSink):
    """`frame_00001.png` をディレクトリに並べる。"""

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.directory = options.get("outputPath")
        self.pattern = options.get("pattern") or "frame_%05d.png"
        self.start_index = options.get("startIndex") or 0

    def begin(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def write_frame(self, bitmap, index: int | None = None) -> None:
        frame_number = self.start_index + self.frame_count if index is None else index

        def expand(match: re.Match) -> str:
            width = int(match.group(1) or 0) or 1
            return str(frame_number).rjust(width, "0")

        name = re.sub(r"%(\d*)d", expand, self.pattern, count=1)
        with open(os.path.join(self.directory, name), "wb") as handle:
            handle.write(encode_png(bitmap))
        self.frame_count += 1

    def end(self) -> dict:
        if self.frame_count == 0:
            raise MovoError(ErrorCodes.MOVO_INTERNAL, "PNG 連番に 1 枚も書き出されませんでした")
        # 書いたはずの枚数が本当にあるか（ディスクが一杯だった、など）。
        written = len([n for n in os.listdir(self.directory) if n.lower().endswith(".png")])
        if written < self.frame_count:
            raise MovoError(
                ErrorCodes.MOVO_INTERNAL,
                f"PNG が {self.frame_count} 枚のはずが {written} 枚しかありません",
                file=self.directory,
                hint="ディスクの空きと書き込み権限を確かめてください",
            )
        return {"path": self.directory, "frames": self.frame_count}


class GifSink(BaseSink):
    """フレームを覚えておいて 1 本の GIF にする。"""

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.frames: list = []
        self.skipped = 0
        # **`or` で既定値に落とすのは、プロジェクト JSON の `null` を «指定なし» と
        # 読むためです。** `dict.get(key, 既定値)` はキーがあれば `None` を返すので、
        # `"maxWidth": null` と書かれた途端に `bitmap.width > None` で落ちます
        # （JS 版の `??` はここを既定値に落としていました）。
        self.stride = max(1, round(options.get("stride") or 1))
        # GIF は逃げ道の形式です。フル HD のままだと巨大で減色も遅いので、
        # 明示されない限り縮めます。
        self.max_width = (options.get("output") or {}).get("maxWidth") or 960
        self.scaled_down = False

    def write_frame(self, bitmap, index: int | None = None) -> None:
        if self.frame_count % self.stride == 0:
            if len(self.frames) < GIF_FRAME_LIMIT:
                if bitmap.width > self.max_width:
                    height = max(1, round((bitmap.height * self.max_width) / bitmap.width))
                    self.frames.append(bitmap.resize(self.max_width, height))
                    self.scaled_down = True
                else:
                    self.frames.append(bitmap.copy())
            else:
                self.skipped += 1
        self.frame_count += 1

    def end(self) -> dict:
        if self.skipped > 0:
            logger.warn(
                f"GIF は先頭 {GIF_FRAME_LIMIT} フレームだけ残し、"
                f"メモリのために {self.skipped} フレームを捨てました"
            )
        if not self.frames:
            raise MovoError(ErrorCodes.MOVO_INTERNAL, "GIF に入れるフレームが 1 枚もありません")
        if self.scaled_down:
            logger.warn(
                f"GIF は {self.frames[0].width}x{self.frames[0].height} に縮めました"
                "（output.maxWidth で変えられます）"
            )
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)) or ".", exist_ok=True)
        output = self.options.get("output") or {}
        buffer = encode_gif(self.frames, {
            "fps": self.fps / self.stride,
            "colors": output.get("colors") or 128,
            "loop": output.get("loop") or 0,
        })
        with open(self.output_path, "wb") as handle:
            handle.write(buffer)
        verify_output(self.output_path, frames=len(self.frames))
        return {"path": self.output_path, "frames": len(self.frames)}


class WavSink(BaseSink):
    """音だけの書き出し。"""

    def end(self) -> dict:
        if not self.options.get("audio"):
            raise MovoError(ErrorCodes.MOVO_INTERNAL, "wav の書き出しにはミックス済みの音が要ります")
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)) or ".", exist_ok=True)
        with open(self.output_path, "wb") as handle:
            handle.write(encode_wav(self.options["audio"]))
        verify_output(self.output_path, expected_duration=self.options.get("expectedDuration"))
        return {"path": self.output_path, "frames": 0}


# ── 書き出したものを確かめる（#78） ────────────────────────────────


def probe_duration(path: str) -> float | None:
    """ffprobe で尺（秒）を読む。ffprobe が無ければ `None`。"""
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe["path"], "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30, **_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float((result.stdout or "").strip())
    except ValueError:
        return None


def verify_output(path: str, expected_duration: float | None = None,
                  frames: int | None = None, fps: float | None = None) -> dict:
    """**書き出したものが本当にできているか**を確かめる（#78）。

    ここを省くと «0 バイトの mp4» や «途中で切れた動画» が «成功» として
    残ります。書き出しは 1 本 10 分かかることもあるので、あとで気付くのと
    その場で気付くのとでは意味が違います。

    :param expected_duration: 秒。`None` なら `frames` と `fps` から出します
    :raises MovoError: 実在しない・空・尺が合わない
    """
    if not os.path.exists(path):
        raise MovoError(
            ErrorCodes.MOVO_INTERNAL, "書き出したはずのファイルがありません", file=path,
            hint="書き込み先の権限とディスクの空きを確かめてください",
        )
    size = os.path.getsize(path)
    if size == 0:
        raise MovoError(
            ErrorCodes.MOVO_INTERNAL, "書き出したファイルが 0 バイトです", file=path,
            hint="ffmpeg が途中で落ちた可能性があります。--verbose で出力を確かめてください",
        )

    if expected_duration is None and frames and fps:
        expected_duration = frames / fps
    if not expected_duration:
        return {"path": path, "size": size, "duration": None}

    duration = probe_duration(path)
    if duration is None:
        # ffprobe が無い環境では «実在と大きさ» までしか見られません。
        logger.verbose(f"ffprobe が無いので尺の検査は飛ばしました: {path}")
        return {"path": path, "size": size, "duration": None}
    if abs(duration - expected_duration) > DURATION_TOLERANCE:
        raise MovoError(
            ErrorCodes.MOVO_INTERNAL,
            f"書き出した尺が合いません: {duration:.3f} 秒（期待 {expected_duration:.3f} 秒）",
            file=path,
            hint="ffmpeg が途中で終わったか、入力のフレーム数が足りていません",
        )
    return {"path": path, "size": size, "duration": duration}


# ── 選ぶ ─────────────────────────────────────────────────────────


#: 呼ぶ側が使う名前（Python 版の綴り）と、この中で使う名前（JS 版の綴り）の対応。
#:
#: **ここが要点です。** `movo.cli.pipeline.render_video()` は
#: `create_exporter(format, width=…, output_path=…, audio_path=…, start_index=…)`
#: と **キーワード引数** で呼びます。JS 版のまま «第 2 引数に辞書» にしておくと
#: `TypeError` になり、`--format` も `-o` も含めて書き出しがまるごと動きません。
#: 中身は JS 版の綴り（`outputPath`）のままにして、入口だけで直しています。
#: 移植したコードを JS と並べて読めるほうが、後から追いやすいためです。
_OPTION_ALIASES = {
    "output_path": "outputPath",
    "audio_path": "audioPath",
    "start_index": "startIndex",
    "expected_duration": "expectedDuration",
    "max_width": "maxWidth",
}


def _normalise_options(options: dict | None, extra: dict) -> dict:
    """辞書でもキーワードでも受け、綴りを 1 つに揃える。"""
    merged = {**(options or {}), **extra}
    for snake, camel in _OPTION_ALIASES.items():
        if snake in merged:
            # camel が既にあればそちらを優先（両方来ることは無いはずですが、
            # 黙って上書きするより «明示された JS 綴り» を残すほうが安全です）
            merged.setdefault(camel, merged.pop(snake))
    return merged


def create_exporter(fmt: str, options: dict | None = None, plugins: dict | None = None, **kwargs):
    """形式に合った書き出し口を作る。

    :param fmt: `mp4` / `webm` / `mov` / `gif` / `png-sequence` / `wav`
    :param options: 設定をまとめた辞書（JS 版と同じ渡し方）
    :param kwargs: `width` / `height` / `fps` / `output_path` / `output` /
        `audio_path` / `audio` / `start_index` / `stride`（CLI はこちらで渡します）
    :returns: `begin()` → `write_frame(bitmap, index)` → `end()` を持つ書き出し口。
        **JS 版と違って同期です**（Python 側に描画の非同期が無いため）。
    """
    settings = _normalise_options(options, kwargs)
    custom = plugins.get("exporter")(fmt) if plugins and plugins.get("exporter") else None
    if custom:
        return custom(settings)
    options = settings
    if fmt == "png-sequence":
        return PngSequenceSink(options)
    if fmt == "gif":
        return GifSink(options)
    if fmt == "wav":
        return WavSink(options)
    if fmt in FFMPEG_FORMATS:
        return FfmpegSink({**options, "format": fmt})
    raise MovoError(
        ErrorCodes.MOVO_UNSUPPORTED, f'知らない出力形式です: "{fmt}"',
        hint=f"使える形式: {', '.join(EXPORT_FORMATS)}",
    )


def negotiate_format(requested: str | None) -> dict:
    """使える形式に落とす。ffmpeg が無ければ GIF へ降ります。"""
    fmt = requested or "mp4"
    if fmt in FFMPEG_FORMATS and not find_ffmpeg():
        return {
            "format": "gif",
            "downgraded": True,
            "reason": f'ffmpeg が無いので "{fmt}" は出せません。代わりにアニメーション GIF を書き出します',
        }
    # `reason` は降格しなくても **キーごと** 返します。呼ぶ側が
    # `negotiated["reason"]` と書いても落ちないようにするためです。
    return {"format": fmt, "downgraded": False, "reason": None}


def default_extension_for(fmt: str) -> str:
    return {
        "png-sequence": "",
        "gif": ".gif",
        "wav": ".wav",
        "webm": ".webm",
        "mov": ".mov",
    }.get(fmt, ".mp4")


def list_exporters() -> list[dict]:
    return [{"format": fmt, "requiresFfmpeg": fmt in FFMPEG_FORMATS} for fmt in EXPORT_FORMATS]


__all__ = [
    "EXPORT_FORMATS", "BaseSink", "FfmpegSink", "PngSequenceSink", "GifSink", "WavSink",
    "create_exporter", "negotiate_format", "default_extension_for", "list_exporters",
    "encode_gif", "build_palette", "verify_output", "probe_duration",
]
