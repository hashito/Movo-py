"""コーデックの «縁» のテスト — ビット深度・透明色・壊れた入力。

:mod:`test_core_js_parity` が «普通の画像で JS 版と同じ結果か» を見るのに対し、
ここは **JS 版のテストが持っていない縁**を埋めます。移植のときに落としやすい
のは «1 ビット PNG» や «16 ビット PNG» のような滅多に来ない経路で、
そこはたいてい素材をもらった当日に初めて踏みます。

PNG は手で組み立てています。ffmpeg に作らせると «その ffmpeg が出す形» しか
試せず、深度 1/2/4 の PNG は普通は出てこないためです。
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import movo.core as core
from movo.core.png import SIGNATURE, _chunk


def _build_png(width: int, height: int, depth: int, color_type: int, rows: bytes, extra: bytes = b"") -> bytes:
    """フィルタ 0（無変換）だけの PNG を組み立てる。

    ``rows`` は 1 行ぶんのバイト列を並べたもの（フィルタバイトは付けない）。
    """
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    stride = -(-(channels * depth * width) // 8)
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # フィルタ 0
        raw += rows[y * stride : (y + 1) * stride]
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([depth, color_type, 0, 0, 0])
    return SIGNATURE + _chunk(b"IHDR", ihdr) + extra + _chunk(b"IDAT", zlib.compress(bytes(raw))) + _chunk(b"IEND", b"")


def test_png_1bit_grayscale():
    """深度 1。倍率 255 はちょうど整数なので、黒と白がぴったり出ること。"""
    # 8 画素 x 2 行、10101010 / 01010101
    png = _build_png(8, 2, 1, 0, bytes([0b10101010, 0b01010101]))
    bitmap = core.decode_png(png)
    assert bitmap.data[0, 0, 0] == 255
    assert bitmap.data[0, 1, 0] == 0
    assert bitmap.data[1, 0, 0] == 0
    assert (bitmap.data[..., 3] == 255).all()


def test_png_2bit_and_4bit_scale_exactly():
    """深度 2 / 4 の倍率（85 と 17）が整数であること。

    ここが浮動小数点になると «白が 254» のような 1 ずれが出て、
    往復テストが通らなくなります。
    """
    two = core.decode_png(_build_png(4, 1, 2, 0, bytes([0b00011011])))
    assert [int(v) for v in two.data[0, :, 0]] == [0, 85, 170, 255]
    four = core.decode_png(_build_png(2, 1, 4, 0, bytes([0x0F])))
    assert [int(v) for v in four.data[0, :, 0]] == [0, 255]


def test_png_16bit_takes_the_high_byte():
    """深度 16 は上位バイトだけ見ること（JS 版と同じ割り切り）。"""
    rows = bytes([0x12, 0x34, 0xAB, 0xCD])  # 2 画素ぶん
    bitmap = core.decode_png(_build_png(2, 1, 16, 0, rows))
    assert [int(v) for v in bitmap.data[0, :, 0]] == [0x12, 0xAB]


def test_png_grayscale_transparency_key():
    """tRNS の «透明にする色» が効くこと。"""
    trns = _chunk(b"tRNS", (0x80).to_bytes(2, "big"))
    bitmap = core.decode_png(_build_png(2, 1, 8, 0, bytes([0x80, 0x40]), extra=trns))
    assert int(bitmap.data[0, 0, 3]) == 0
    assert int(bitmap.data[0, 1, 3]) == 255


def test_png_palette_with_alpha():
    plte = _chunk(b"PLTE", bytes([255, 0, 0, 0, 255, 0, 0, 0, 255]))
    trns = _chunk(b"tRNS", bytes([0, 128]))  # 0 番は透明、1 番は半透明、2 番は指定なし
    bitmap = core.decode_png(_build_png(3, 1, 8, 3, bytes([0, 1, 2]), extra=plte + trns))
    assert [int(v) for v in bitmap.data[0, 0]] == [255, 0, 0, 0]
    assert [int(v) for v in bitmap.data[0, 1]] == [0, 255, 0, 128]
    assert [int(v) for v in bitmap.data[0, 2]] == [0, 0, 255, 255]


def test_png_gray_alpha():
    bitmap = core.decode_png(_build_png(2, 1, 8, 4, bytes([100, 200, 50, 25])))
    assert [int(v) for v in bitmap.data[0, 0]] == [100, 100, 100, 200]
    assert [int(v) for v in bitmap.data[0, 1]] == [50, 50, 50, 25]


def test_png_interlaced_is_refused_with_a_hint():
    """**Adam7 は «対応しない» とはっきり言うこと。**

    黙って壊れた絵を返すより、直し方を出すほうが親切です。
    """
    ihdr = (2).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 6, 0, 0, 1])
    png = SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(b"\0" * 9)) + _chunk(b"IEND", b"")
    with pytest.raises(core.MovoError) as info:
        core.decode_png(png)
    assert info.value.code == core.ErrorCodes.MOVO_ASSET_DECODE_FAILED
    assert info.value.hint  # 直し方が付いていること


def test_png_rejects_non_png_and_short_data():
    with pytest.raises(core.MovoError):
        core.decode_png(b"not a png at all")
    truncated = _build_png(4, 4, 8, 6, bytes(4 * 4 * 4))[:-40]
    with pytest.raises(core.MovoError):
        core.decode_png(truncated)


def test_png_round_trip_for_every_filter_choice():
    """5 種類のフィルタが選ばれうる絵で往復すること。

    左右に同じ値が続く行（フィルタ 1 が有利）、上下に同じ値が続く行
    （フィルタ 2 が有利）、勾配（3 / 4 が有利）を混ぜてあります。
    """
    bitmap = core.Bitmap(32, 12)
    bitmap.data[0:4] = 200  # 一様 → 0/1/2 のどれでも
    bitmap.data[4:8, :, 0] = np.arange(32)[None, :]  # 横の勾配
    bitmap.data[8:12, :, 1] = (np.arange(12) - 8)[8:12, None] * 20  # 縦の勾配
    bitmap.data[..., 3] = 255
    assert np.array_equal(core.decode_png(core.encode_png(bitmap)).data, bitmap.data)


def test_encode_png_refuses_an_empty_bitmap():
    with pytest.raises(core.MovoError):
        core.encode_png(core.Bitmap(0, 0))


# ── JPEG ────────────────────────────────────────────────────


def test_jpeg_progressive_is_refused_with_a_hint():
    """プログレッシブ JPEG は **ffmpeg に回す** と分かる文面で断ること。"""
    # SOI + SOF2（プログレッシブのフレームヘッダ）
    data = bytes([0xFF, 0xD8, 0xFF, 0xC2, 0x00, 0x0B, 0x08, 0, 8, 0, 8, 1, 1, 0x11, 0])
    with pytest.raises(core.MovoError) as info:
        core.decode_jpeg(data)
    assert "ffmpeg" in (info.value.hint or "")


def test_jpeg_rejects_non_jpeg():
    with pytest.raises(core.MovoError):
        core.decode_jpeg(b"\x00\x01\x02\x03")


def test_is_jpeg_and_is_png_look_at_the_content_not_the_name():
    assert core.is_png(core.encode_png(core.Bitmap.create(2, 2, "#fff")))
    assert not core.is_jpeg(core.encode_png(core.Bitmap.create(2, 2, "#fff")))


# ── BMP ─────────────────────────────────────────────────────


def _build_bmp(width: int, height: int, bpp: int, pixels: bytes, bottom_up: bool = True) -> bytes:
    stride = ((bpp * width + 31) // 32) * 4
    body = bytearray()
    for y in range(height):
        row = pixels[y * width * (bpp // 8) : (y + 1) * width * (bpp // 8)]
        body += row + bytes(stride - len(row))
    offset = 14 + 40
    header = bytearray(b"BM")
    header += (offset + len(body)).to_bytes(4, "little") + bytes(4) + offset.to_bytes(4, "little")
    header += (40).to_bytes(4, "little")
    header += width.to_bytes(4, "little", signed=True)
    header += (height if bottom_up else -height).to_bytes(4, "little", signed=True)
    header += (1).to_bytes(2, "little") + bpp.to_bytes(2, "little") + bytes(24)
    return bytes(header) + bytes(body)


def test_bmp_bottom_up_rows_are_flipped():
    """**正の高さは «下から上»** に並んでいること（BMP の伝統）。"""
    pixels = bytes([1, 2, 3, 4, 5, 6])  # 1 画素 x 2 行（BGR）
    bitmap = core.decode_bmp(_build_bmp(1, 2, 24, pixels, bottom_up=True))
    assert [int(v) for v in bitmap.data[0, 0, :3]] == [6, 5, 4]  # 最後の行が上に来る
    top_down = core.decode_bmp(_build_bmp(1, 2, 24, pixels, bottom_up=False))
    assert [int(v) for v in top_down.data[0, 0, :3]] == [3, 2, 1]


def test_bmp_32bit_keeps_alpha():
    pixels = bytes([10, 20, 30, 40])
    bitmap = core.decode_bmp(_build_bmp(1, 1, 32, pixels))
    assert [int(v) for v in bitmap.data[0, 0]] == [30, 20, 10, 40]


def test_bmp_refuses_unsupported_depth():
    with pytest.raises(core.MovoError):
        core.decode_bmp(_build_bmp(1, 1, 8, bytes([1])))


def test_decode_image_reports_an_unknown_format_clearly():
    with pytest.raises(core.MovoError) as info:
        core.decode_image(b"RIFFxxxxWEBPVP8 ", allow_ffmpeg=False)
    assert info.value.code == core.ErrorCodes.MOVO_ASSET_DECODE_FAILED


# ── WAV ─────────────────────────────────────────────────────


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_wav_round_trip_for_every_integer_depth(bits: int):
    audio = core.create_silence(0.005, 16000, 2)
    index = np.arange(audio.length)
    audio.channels[0][:] = np.sin(index / 7).astype(np.float32)
    audio.channels[1][:] = np.cos(index / 11).astype(np.float32)
    decoded = core.decode_wav(core.encode_wav(audio, bits_per_sample=bits))
    assert decoded.length == audio.length
    assert decoded.sample_rate == 16000
    # 許容値が分解能そのもの（16 ビットなら 1/32768）より少し大きいのは、
    # **書き出しが 32767 倍・読み込みが 32768 で割る** という非対称のためです
    # （そうしないと -1.0 が表現できません）。JS 版も同じ非対称です。
    tolerance = {16: 5e-5, 24: 2e-7, 32: 2e-7}[bits]
    assert np.allclose(decoded.channels[0], audio.channels[0], atol=tolerance)
    assert np.allclose(decoded.channels[1], audio.channels[1], atol=tolerance)


def test_wav_8bit_is_offset_binary():
    """8 ビットだけは **符号なし（128 が無音）** であること。RIFF の決まりです。"""
    header = bytearray(b"RIFF")
    payload = bytes([128, 255, 0])
    header += (36 + len(payload)).to_bytes(4, "little") + b"WAVEfmt "
    header += (16).to_bytes(4, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
    header += (8000).to_bytes(4, "little") + (8000).to_bytes(4, "little")
    header += (1).to_bytes(2, "little") + (8).to_bytes(2, "little")
    header += b"data" + len(payload).to_bytes(4, "little")
    decoded = core.decode_wav(bytes(header) + payload)
    assert decoded.channels[0].tolist() == pytest.approx([0.0, 0.9921875, -1.0])


def test_wav_extensible_format_is_unwrapped():
    """``WAVE_FORMAT_EXTENSIBLE``（0xFFFE）の «本当の形式» を拡張部から読むこと。

    録音ソフトが出す WAV はたいていこの形なので、ここを落とすと
    «自分で録った音が読めない» になります。
    """
    payload = np.array([0.5, -0.5], np.float32).tobytes()
    fmt = bytearray()
    fmt += (0xFFFE).to_bytes(2, "little") + (1).to_bytes(2, "little")
    fmt += (8000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
    fmt += (4).to_bytes(2, "little") + (32).to_bytes(2, "little")
    fmt += (22).to_bytes(2, "little") + (32).to_bytes(2, "little") + (4).to_bytes(4, "little")
    fmt += (3).to_bytes(2, "little") + bytes(14)  # 本当の形式は IEEE float
    body = bytearray(b"RIFF")
    body += (36 + len(fmt) - 16 + len(payload)).to_bytes(4, "little") + b"WAVEfmt "
    body += len(fmt).to_bytes(4, "little") + fmt
    body += b"data" + len(payload).to_bytes(4, "little") + payload
    decoded = core.decode_wav(bytes(body))
    assert decoded.channels[0].tolist() == pytest.approx([0.5, -0.5])


def test_wav_rejects_garbage():
    with pytest.raises(core.MovoError):
        core.decode_wav(b"not riff")
    with pytest.raises(core.MovoError):
        core.decode_wav(b"RIFF" + (4).to_bytes(4, "little") + b"WAVE")


def test_encode_wav_clips_out_of_range_samples():
    """**1 を超える標本は切り詰めること。** 巻き戻ると耳に痛いノイズになります。"""
    audio = core.create_silence(0.001, 8000, 1)
    audio.channels[0][:] = np.linspace(-3, 3, audio.length, dtype=np.float32)
    decoded = core.decode_wav(core.encode_wav(audio))
    assert decoded.channels[0].min() >= -1.0
    assert decoded.channels[0].max() <= 1.0


def test_resample_is_a_no_op_at_the_same_rate():
    audio = core.create_silence(0.01, 8000, 1)
    assert core.resample(audio, 8000) is audio
