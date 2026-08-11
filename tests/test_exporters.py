"""書き出しの検査。

大きく 2 つを見ています。

1. **GIF が JS 版と 1 バイトも変わらないこと。** パレットの選び方（メディアン
   カット）も LZW も自前実装なので、少しでも違うと «同じ JSON から違う絵» に
   なります。基準は

       node tests/data/parity_gif.mjs > tests/data/parity_gif.json

2. **JS 版の不具合（Movo issue #78）が持ち込まれていないこと。**
   ffmpeg が落ちたとき・パイプが切れたとき・出来たものが空のときに、
   **必ず例外になる**ことを確かめます。ここが «静かに成功» に戻ると、
   0 バイトの mp4 が «書き出し完了» として残ります。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.core.errors import ErrorCodes, MovoError
from movo.core.wav import create_silence
from movo.exporters import (
    EXPORT_FORMATS, FfmpegSink, GifSink, PngSequenceSink, WavSink,
    build_palette, create_exporter, default_extension_for, encode_gif,
    list_exporters, negotiate_format, verify_output,
)

GOLDEN = json.loads((Path(__file__).parent / "data" / "parity_gif.json").read_text("utf-8"))


def frame(width: int, height: int, k: int) -> Bitmap:
    """JS 版の parity_gif.mjs と同じ絵。"""
    bitmap = Bitmap(width, height)
    x = np.arange(width)[None, :]
    y = np.arange(height)[:, None]
    bitmap.data[..., 0] = (x * 8 + k * 40) % 256
    bitmap.data[..., 1] = (y * 10) % 256
    bitmap.data[..., 2] = (x * y + k * 17) % 256
    bitmap.data[..., 3] = 255
    bitmap.data[0:4, 0:4, 3] = 0
    return bitmap


# ── GIF が JS 版と一致するか ──────────────────────────────────────


def test_palette_matches_js():
    frames = [frame(32, 24, k) for k in range(3)]
    palette = build_palette(frames, 64)
    assert palette.tolist() == GOLDEN["palette64"]


def test_gif_bytes_match_js():
    """**1 バイトも変えない。** 透明色つき・3 フレーム。"""
    frames = [frame(32, 24, k) for k in range(3)]
    got = encode_gif(frames, {"fps": 10, "colors": 64})
    assert list(got) == GOLDEN["small"], f"長さ {len(got)} 対 {len(GOLDEN['small'])}"


def test_gif_opaque_and_loop_match_js():
    frames = [frame(32, 24, k) for k in range(3)]
    got = encode_gif(frames, {"fps": 12, "colors": 16, "transparent": False, "loop": 3})
    assert list(got) == GOLDEN["smallOpaque"]


def test_gif_large_matches_js():
    """符号長が伸びる経路と、255 バイトごとのサブブロック分割を通す。"""
    frames = [frame(120, 90, 0), frame(120, 90, 5)]
    got = encode_gif(frames, {"fps": 24, "colors": 200})
    assert list(got) == GOLDEN["large"]


def test_gif_needs_at_least_one_frame():
    with pytest.raises(ValueError):
        encode_gif([], {})


# ── 書き出し口 ────────────────────────────────────────────────────


def test_png_sequence_writes_files(tmp_path):
    sink = create_exporter("png-sequence", {"outputPath": str(tmp_path / "seq"), "width": 8, "height": 8, "fps": 10})
    sink.begin()
    for i in range(3):
        sink.write_frame(frame(8, 8, i))
    result = sink.end()
    assert result["frames"] == 3
    names = sorted(os.listdir(result["path"]))
    assert names == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert all((Path(result["path"]) / n).stat().st_size > 0 for n in names)


def test_png_sequence_refuses_to_report_success_when_empty(tmp_path):
    """**1 枚も書けていないのに «成功» を返さない**（#78 と同じ考え方）。"""
    sink = PngSequenceSink({"outputPath": str(tmp_path / "empty"), "fps": 10})
    sink.begin()
    with pytest.raises(MovoError) as error:
        sink.end()
    assert error.value.code == ErrorCodes.MOVO_INTERNAL


def test_gif_sink_writes_and_verifies(tmp_path):
    path = str(tmp_path / "out.gif")
    sink = create_exporter("gif", {"outputPath": path, "width": 32, "height": 24, "fps": 10,
                                   "output": {"colors": 64}})
    sink.begin()
    for i in range(3):
        sink.write_frame(frame(32, 24, i))
    result = sink.end()
    assert result["frames"] == 3
    assert Path(path).read_bytes()[:6] == b"GIF89a"


def test_gif_sink_refuses_when_no_frames(tmp_path):
    sink = GifSink({"outputPath": str(tmp_path / "none.gif"), "fps": 10})
    sink.begin()
    with pytest.raises(MovoError):
        sink.end()


def test_wav_sink_writes(tmp_path):
    path = str(tmp_path / "out.wav")
    sink = create_exporter("wav", {"outputPath": path, "audio": create_silence(0.25, 48000, 2), "fps": 30})
    sink.begin()
    result = sink.end()
    assert result["frames"] == 0
    assert Path(path).read_bytes()[:4] == b"RIFF"


def test_wav_sink_needs_audio(tmp_path):
    sink = WavSink({"outputPath": str(tmp_path / "x.wav")})
    with pytest.raises(MovoError):
        sink.end()


def test_unknown_format_is_rejected():
    with pytest.raises(MovoError) as error:
        create_exporter("avi", {"outputPath": "x.avi"})
    assert error.value.code == ErrorCodes.MOVO_UNSUPPORTED


def test_create_exporter_accepts_the_pipeline_call(tmp_path):
    """**呼ぶ側の書き方をそのまま試します。**

    `movo.cli.pipeline.render_video()` は書き出し口を «キーワード引数» で作ります
    （`output_path=` / `audio_path=` / `start_index=`）。JS 版のまま «第 2 引数に
    辞書» だけを受けると、ここで `TypeError` になって **`--format` も `-o` も
    含めて書き出しがまるごと動かなくなります**。実際に一度そうなっていたので、
    呼び出し方そのものを検査に残します。
    """
    sink = create_exporter(
        "png-sequence",
        width=8,
        height=6,
        fps=10,
        output_path=str(tmp_path / "seq"),
        output={},
        audio_path=None,
        audio=None,
        start_index=7,
        stride=1,
    )
    sink.begin()
    sink.write_frame(frame(8, 6, 0), 7)
    result = sink.end()
    assert result == {"path": str(tmp_path / "seq"), "frames": 1}
    # start_index / index が «何番のファイルか» に効いていること
    assert os.listdir(tmp_path / "seq") == ["frame_00007.png"]


def test_create_exporter_still_accepts_a_plain_dict(tmp_path):
    """辞書で渡す JS 版の書き方も残っていること（移植したコードが使います）。"""
    sink = create_exporter("gif", {"outputPath": str(tmp_path / "a.gif"), "width": 8, "height": 6, "fps": 10})
    sink.begin()
    sink.write_frame(frame(8, 6, 0), 0)
    assert sink.end()["frames"] == 1


def test_negotiate_format_always_has_a_reason_key():
    """降格しなくても `reason` を返すこと（呼ぶ側が添字で読んでも落ちないように）。"""
    assert negotiate_format("gif") == {"format": "gif", "downgraded": False, "reason": None}


def test_format_helpers():
    assert default_extension_for("gif") == ".gif"
    assert default_extension_for("png-sequence") == ""
    assert default_extension_for("なにか") == ".mp4"
    assert {e["format"] for e in list_exporters()} == set(EXPORT_FORMATS)
    assert negotiate_format(None)["format"] in ("mp4", "gif")


# ── #78: 静かに成功しないこと ─────────────────────────────────────


def test_verify_output_rejects_a_missing_file(tmp_path):
    with pytest.raises(MovoError) as error:
        verify_output(str(tmp_path / "ない.mp4"))
    assert "ありません" in error.value.reason


def test_verify_output_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    with pytest.raises(MovoError) as error:
        verify_output(str(path))
    assert "0 バイト" in error.value.reason


def test_verify_output_rejects_a_truncated_video(tmp_path, monkeypatch):
    """**尺が足りない動画を «成功» にしない。**

    ffprobe が無い環境でも検査そのものを確かめられるよう、尺を読む関数だけ
    差し替えます（ffmpeg が途中で終わった状況の再現）。
    """
    import movo.exporters as exporters

    path = tmp_path / "short.mp4"
    path.write_bytes(b"x" * 1024)
    monkeypatch.setattr(exporters, "probe_duration", lambda _: 1.0)
    with pytest.raises(MovoError) as error:
        exporters.verify_output(str(path), expected_duration=10.0)
    assert "尺が合いません" in error.value.reason
    # 誤差の範囲なら通ること
    monkeypatch.setattr(exporters, "probe_duration", lambda _: 9.9)
    assert exporters.verify_output(str(path), expected_duration=10.0)["duration"] == 9.9


def test_ffmpeg_failure_raises_instead_of_reporting_success(tmp_path, monkeypatch):
    """**ffmpeg が 0 以外で終わったら例外。** JS 版はここを見落としていました。"""
    import movo.exporters as exporters

    monkeypatch.setattr(exporters, "find_ffmpeg", lambda: {"path": sys.executable})
    path = str(tmp_path / "out.mp4")
    sink = FfmpegSink({"outputPath": path, "width": 4, "height": 4, "fps": 10, "format": "mp4"})
    # ffmpeg の代わりに «必ず終了コード 3 で落ちる» Python を走らせます。
    sink._args = lambda: ["-c", "import sys; sys.exit(3)"]
    sink.begin()
    with pytest.raises(MovoError) as error:
        # 相手はすぐ死ぬので、書いている途中か end() のどちらかで失敗します。
        for i in range(50):
            sink.write_frame(frame(4, 4, i))
        sink.end()
    assert error.value.code == ErrorCodes.MOVO_RENDERER_UNAVAILABLE
    assert not os.path.exists(path), "落ちたのにファイルが «成功» として残っています"


def test_ffmpeg_broken_pipe_is_not_swallowed(tmp_path, monkeypatch):
    """**EPIPE を握りつぶさない。** 相手が読まずに終わる状況を作ります。"""
    import movo.exporters as exporters

    monkeypatch.setattr(exporters, "find_ffmpeg", lambda: {"path": sys.executable})
    sink = FfmpegSink({"outputPath": str(tmp_path / "out.mp4"), "width": 64, "height": 64,
                       "fps": 10, "format": "mp4"})
    # 標準入力を読まずに «すぐ» 終わるので、書き込み側は必ずパイプ切れになります。
    sink._args = lambda: ["-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(1)"]
    sink.begin()
    with pytest.raises(MovoError) as error:
        for i in range(2000):
            sink.write_frame(frame(64, 64, i % 3))
        sink.end()
    assert error.value.code == ErrorCodes.MOVO_RENDERER_UNAVAILABLE


def test_ffmpeg_signal_death_is_not_reported_as_success(tmp_path, monkeypatch):
    """**シグナルで死んだら «終了コード 0» にしない。** これが #78 の本体です。"""
    import movo.exporters as exporters

    monkeypatch.setattr(exporters, "find_ffmpeg", lambda: {"path": sys.executable})
    path = str(tmp_path / "signal.mp4")
    sink = FfmpegSink({"outputPath": path, "width": 4, "height": 4, "fps": 10, "format": "mp4"})
    # 自分を強制終了する（POSIX では SIGKILL、Windows では異常終了コード）。
    sink._args = lambda: [
        "-c",
        "import os, sys, signal;"
        "sys.stdin.buffer.read(16);"
        "os.kill(os.getpid(), signal.SIGTERM) if os.name != 'nt' else os._exit(0xC000013A - (1 << 32))",
    ]
    sink.begin()
    with pytest.raises(MovoError) as error:
        for i in range(500):
            sink.write_frame(frame(4, 4, i % 3))
        sink.end()
    assert error.value.code == ErrorCodes.MOVO_RENDERER_UNAVAILABLE
    assert not os.path.exists(path)
