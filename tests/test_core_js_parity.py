"""**JS 版と同じ結果が出ることを確かめるテスト。**

ここが core の移植でいちばん大事なテストです。Movo の約束は
«同じ JSON からは同じ動画が出る» ことで、それは JS 版と Python 版の間でも
成り立たなければ意味がありません（JS 版で作った MV を Python 版で
作り直したら色が違う、では移植したことになりません）。

``core_js_reference.json`` は **JS 版を実際に走らせて取った値**です。
作り直すときは ``packages/core/src`` の各関数を呼んで JSON に落としてください
（乱数の系列、色の変換、PNG のバイト列、JPEG の復号結果、歌詞の解析、
SVG のパス、LUT、WAV、行列、閃光検査、映像プロファイル）。

画像のフィクスチャ（``core_fixtures/``）は ffmpeg で作った小さな
97x61 のテストパターンです。**JPEG は 4:2:0 / 4:4:4 / グレースケールの
3 種類**を置いてあります。間引き（クロマサブサンプリング）の扱いが
デコーダのいちばん間違えやすいところだからです。
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

import movo.core as core

HERE = Path(__file__).parent
REFERENCE = json.loads((HERE / "core_js_reference.json").read_text(encoding="utf-8"))
FIXTURES = HERE / "core_fixtures"


# ── 乱数（決定性） ──────────────────────────────────────────


def test_mulberry32_matches_js():
    """同じ種から **同じ系列**が出ること。ここがずれると全部ずれます。"""
    rng = core.create_random(99)
    assert [rng() for _ in range(8)] == REFERENCE["random99"]
    # 種 0 は «指定なし» として黄金比に置き換わる（JS の `|| 0x9e3779b9`）
    zero = core.create_random(0)
    assert [zero() for _ in range(5)] == REFERENCE["random0"]


def test_random_helpers_match_js():
    rng = core.create_random(7)
    assert [rng.int(0, 10) for _ in range(6)] == REFERENCE["random7_int"]
    gauss = core.create_random(42)
    assert [gauss.gaussian() for _ in range(4)] == pytest.approx(REFERENCE["gaussian42"], abs=1e-12)


def test_hash_string_matches_js_including_non_bmp():
    """絵文字（サロゲートペア）でも一致すること。

    JS の ``charCodeAt`` は **UTF-16 のコード単位**を返すので、Python 側で
    ``ord()`` を使うと絵文字を含む名前で値がずれます。
    """
    names = ["hero", "x", "y", "z", "あ", "🎵"]
    assert [core.hash_string(n) for n in names] == REFERENCE["hashString"]


def test_random_streams_are_independent_and_reproducible():
    assert core.RandomSource(5).stream("x")() == REFERENCE["stream"]["seed5x"]
    assert core.RandomSource(5).stream("y")() == REFERENCE["stream"]["seed5y"]
    source = core.RandomSource(5)
    assert source.stream("y")() != source.stream("z")()


def test_value_noise_matches_js():
    assert [core.value_noise_1d(x, 7) for x in (0, 0.5, 3.25, -2.75, 100.125)] == REFERENCE["noise1"]
    assert [core.value_noise_2d(x, y, 3) for x, y in ((0.5, 0.5), (3.25, -1.75), (10.1, 20.2))] == REFERENCE["noise2"]
    assert [core.value_noise_3d(*p, 11) for p in ((0.5, 0.5, 0.5), (3.25, -1.75, 2.5))] == REFERENCE["noise3"]


def test_fbm_matches_js():
    assert [core.fbm1d(x, 5) for x in (0.5, 2.25)] == REFERENCE["fbm1"]
    assert [
        core.fbm2d(1.5, 2.5, seed=9),
        core.fbm2d(1.5, 2.5, seed=9, z=1.25, octaves=5, type="turbulent"),
        core.fbm2d(1.5, 2.5, seed=9, z=1.25, octaves=3, type="ridged", gain=0.6, lacunarity=2.5),
    ] == REFERENCE["fbm2"]


def test_numba_noise_grid_equals_scalar_version():
    """**Numba 版と素の Python 版が 1 ビットも違わないこと。**

    :mod:`movo.core.rng` は同じ式を 2 回書いています（読みやすさのための
    スカラ版と、画素ごとに呼ぶための Numba 版）。片方だけ直して食い違うと、
    «プレビューと書き出しで模様が違う» という追いにくい壊れ方をします。
    """
    grid = core.fbm2d_grid(
        6, 4, x0=0.3, y0=-1.2, dx=0.7, dy=0.5, seed=13, z=0.25, octaves=3, type="ridged", gain=0.6, lacunarity=2.5
    )
    scalar = np.array(
        [
            [
                core.fbm2d(
                    0.3 + 0.7 * i, -1.2 + 0.5 * j, seed=13, z=0.25, octaves=3, type="ridged", gain=0.6, lacunarity=2.5
                )
                for i in range(6)
            ]
            for j in range(4)
        ]
    )
    assert np.array_equal(grid, scalar)


# ── 色 ──────────────────────────────────────────────────────


def test_parse_color_matches_js_for_every_notation():
    for text, expected in REFERENCE["colors"].items():
        assert core.parse_color(text) == expected, text


def test_hsl_conversions_match_js():
    assert [list(core.hsl_to_rgb(*x)) for x in ((0, 0, 0.5), (0.5, 1, 0.5), (0.1234, 0.77, 0.31))] == REFERENCE["hslToRgb"]
    assert [list(core.rgb_to_hsl(*x)) for x in ((255, 0, 0), (10, 200, 50), (128, 128, 128))] == REFERENCE["rgbToHsl"]


def test_mix_color_matches_js():
    black = {"r": 0, "g": 0, "b": 0, "a": 0}
    grey = {"r": 100, "g": 100, "b": 100, "a": 1}
    assert core.mix_color(black, grey, 0.5) == REFERENCE["mixColor"]


# ── ハッシュ（キャッシュ鍵の互換） ──────────────────────────


def test_stable_stringify_matches_js_exactly():
    """**キャッシュ鍵が JS 版と共有できること。**

    ``json.dumps`` は ``1.0`` を ``"1.0"`` と書き、非 ASCII を ``\\uXXXX`` に
    開くので、そのまま使うと同じプロジェクトでも別の鍵になります。
    """
    assert core.stable_stringify({"b": 1, "a": [3, {"d": 1, "c": 2}]}) == REFERENCE["stableStringify"]
    assert (
        core.stable_stringify({"fps": 30.0, "name": "あ", "flag": True, "nil": None, "arr": [1.5, -0.25]})
        == REFERENCE["stableStringify2"]
    )
    assert core.hash_json({"a": 1, "b": 2}) == REFERENCE["hashJson"]
    assert core.sha256("movo") == REFERENCE["sha256"]


# ── PNG ─────────────────────────────────────────────────────


def _sample_bitmap() -> core.Bitmap:
    bitmap = core.Bitmap(17, 9)
    bitmap.data.reshape(-1)[:] = [(i * 7) % 256 for i in range(17 * 9 * 4)]
    return bitmap


def _idat(buffer: bytes) -> bytes:
    """PNG から IDAT を取り出して展開する（フィルタをかけた «生の行»）。"""
    offset = 8
    raw = b""
    while offset + 8 <= len(buffer):
        length = int.from_bytes(buffer[offset : offset + 4], "big")
        if buffer[offset + 4 : offset + 8] == b"IDAT":
            raw += buffer[offset + 8 : offset + 8 + length]
        offset += 8 + length + 4
    return zlib.decompress(raw)


def test_png_filtered_rows_match_js_byte_for_byte():
    """**フィルタのかけ方が JS 版と 1 バイトも違わないこと。**

    ファイル全体を比べていないのは、``zlib`` の実装が Node と Python で
    違うためです（同じ入力から違う長さの圧縮結果が出ますが、どちらも
    正しい PNG で、展開すれば同じバイト列になります）。**Movo が決めているのは
    «フィルタの選び方» までなので、そこを比べます。**
    """
    mine = core.encode_png(_sample_bitmap())
    assert _idat(mine) == _idat(bytes(REFERENCE["pngBytes"]))


def test_png_round_trip_preserves_every_pixel():
    bitmap = _sample_bitmap()
    buffer = core.encode_png(bitmap)
    assert core.is_png(buffer)
    decoded = core.decode_png(buffer)
    assert (decoded.width, decoded.height) == (17, 9)
    assert np.array_equal(decoded.data, bitmap.data)


def test_png_round_trip_with_a_gradient_and_varying_alpha():
    """**フィルタの 5 種類が実際に選び分けられる絵**で往復すること。

    べた塗りだとどのフィルタでも同じ結果になり、選択のバグが見つかりません。
    """
    bitmap = core.Bitmap(64, 40)
    xs = np.arange(64)
    ys = np.arange(40)
    bitmap.data[..., 0] = (xs[None, :] * 4) % 256
    bitmap.data[..., 1] = (ys[:, None] * 6) % 256
    bitmap.data[..., 2] = (xs[None, :] * ys[:, None]) % 256
    bitmap.data[..., 3] = 255 - ((xs[None, :] + ys[:, None]) % 64)
    encoded = core.encode_png(bitmap)
    assert np.array_equal(core.decode_png(encoded).data, bitmap.data)
    # フィルタの選び方まで JS 版と同じであること（行ごとの選択も含めて）
    raw = _idat(encoded)
    assert core.sha256(raw) == REFERENCE["png2RawSha"]
    stride = 64 * 4 + 1
    assert [raw[y * stride] for y in range(40)] == REFERENCE["png2RawFilters"]


@pytest.mark.parametrize("key,file", [("pngRGB", "src.png"), ("pngPal", "pal.png"), ("pngGray", "gray.png")])
def test_png_decoding_matches_js(key: str, file: str):
    """RGB / パレット / グレースケールの PNG が JS 版と同じ画素になること。"""
    bitmap = core.decode_image((FIXTURES / file).read_bytes(), allow_ffmpeg=False)
    expected = REFERENCE[key]
    assert (bitmap.width, bitmap.height) == (expected["w"], expected["h"])
    assert core.sha256(bitmap.data.tobytes()) == expected["sha"]


# ── JPEG / BMP ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "key,file", [("jpeg420", "test420.jpg"), ("jpeg444", "test444.jpg"), ("jpegGray", "gray.jpg")]
)
def test_jpeg_decoding_matches_js(key: str, file: str):
    """**JPEG の復号結果が 1 画素も違わないこと。**

    逆 DCT の途中経過を float32 に落としているのはこれを通すためです
    （float64 のまま通すと、最下位ビットの違う画素が数十個出ます）。
    """
    bitmap = core.decode_jpeg((FIXTURES / file).read_bytes())
    expected = REFERENCE[key]
    assert (bitmap.width, bitmap.height) == (expected["w"], expected["h"])
    assert core.sha256(bitmap.data.tobytes()) == expected["sha"]


def test_bmp_decoding_matches_js():
    bitmap = core.decode_image((FIXTURES / "test.bmp").read_bytes(), allow_ffmpeg=False)
    assert core.sha256(bitmap.data.tobytes()) == REFERENCE["bmp24"]["sha"]


# ── WAV ─────────────────────────────────────────────────────


def _sine_audio(seconds: float = 0.01, rate: int = 8000) -> core.AudioBuffer:
    audio = core.create_silence(seconds, rate, 2)
    index = np.arange(audio.length)
    audio.channels[0][:] = (np.sin(index / 5) * 0.5).astype(np.float32)
    audio.channels[1][:] = -audio.channels[0]
    return audio


def test_wav_bytes_match_js_exactly():
    """**書き出した WAV がバイト単位で同じであること。**

    ここは «自前実装» なので、ヘッダの 1 バイトのずれが再生できない
    ファイルになります。ハッシュで丸ごと比べます。
    """
    assert core.sha256(core.encode_wav(_sine_audio())) == REFERENCE["wavSha"]


def test_wav_24bit_matches_js():
    """24 ビットは **掛ける前に float64 へ上げないと** 最下位が 1 ずれます。"""
    audio = core.create_silence(0.002, 8000, 1)
    audio.channels[0][:] = (np.cos(np.arange(audio.length) / 3) * 0.9).astype(np.float32)
    encoded = core.encode_wav(audio, bits_per_sample=24)
    assert core.sha256(encoded) == REFERENCE["wav24Sha"]
    assert core.decode_wav(encoded).channels[0].tolist() == REFERENCE["wav24Decoded"]


def test_wav_float32_round_trip_matches_js():
    audio = core.create_silence(0.002, 8000, 1)
    audio.channels[0][:] = (np.cos(np.arange(audio.length) / 3) * 0.9).astype(np.float32)
    decoded = core.decode_wav(core.encode_wav(audio, float32=True))
    assert decoded.channels[0].tolist() == REFERENCE["wavFloatDecoded"]


def test_wav_round_trip_stays_within_16bit_precision():
    audio = _sine_audio()
    decoded = core.decode_wav(core.encode_wav(audio))
    assert decoded.sample_rate == 8000
    assert decoded.length == audio.length
    assert np.allclose(decoded.channels[0], audio.channels[0], atol=1e-3)


def test_resample_matches_js():
    resampled = core.resample(core.create_silence(1, 8000, 1), 16000)
    assert resampled.sample_rate == REFERENCE["resample"]["rate"]
    assert resampled.length == REFERENCE["resample"]["length"]


# ── 行列 ────────────────────────────────────────────────────


def test_matrix_helpers_match_js():
    m = core.Mat2D.identity()
    m = core.Mat2D.translate(m, 10, 20)
    m = core.Mat2D.rotate(m, 0.7)
    m = core.Mat2D.scale(m, 2, 3)
    assert list(m) == REFERENCE["mat"]
    assert list(core.Mat2D.apply(m, 1, 1)) == REFERENCE["matApply"]
    assert list(core.Mat2D.invert(m)) == REFERENCE["matInv"]
    assert list(
        core.Mat2D.from_transform({"x": 5, "y": 6, "rotation": 30, "scaleX": 2, "scaleY": 0.5, "skewX": 0.2})
    ) == REFERENCE["matFromTransform"]


# ── 歌詞 ────────────────────────────────────────────────────

_LRC = "[ti:test]\n[00:01.00]hello\n[00:03.50]world\n[00:05.00]<00:05.00>ね<00:05.62>え\n"
_SRT = "1\n00:00:01,000 --> 00:00:03,500\nfirst line\nsecond\n\n2\n00:00:04,000 --> 00:00:05,000\nnext\n"


def test_lyrics_parsing_matches_js():
    assert core.parse_lrc(_LRC) == REFERENCE["lrc"]
    assert core.parse_subtitles(_SRT) == REFERENCE["srt"]
    assert core.parse_lyrics('[{"text":"a","at":1,"for":0.5},{"text":"b","at":0.25}]') == REFERENCE["lyricsJson"]


def test_slice_lyrics_matches_js():
    lines = core.parse_lrc(_LRC)
    assert core.slice_lyrics(lines, 2, 6, overlap=True) == REFERENCE["slice"]
    assert core.slice_lyrics(lines, 2, 6) == REFERENCE["sliceStrict"]


# ── SVG ─────────────────────────────────────────────────────

_SVG = (
    '<svg viewBox="0 0 100 50" width="100" height="50">'
    '<defs><path d="M0 0 L1 1"/></defs>'
    '<g transform="translate(5 5)"><rect x="0" y="0" width="10" height="10" rx="2"/>'
    '<circle cx="20" cy="20" r="5"/></g>'
    "<script>evil()</script>"
    '<polygon points="1,2 3,4 5,6"/></svg>'
)


def test_path_flattening_matches_js():
    assert core.path_to_subpaths("M10 10 L20 20 H30 V5 Z") == REFERENCE["path1"]
    assert core.path_to_subpaths("M0 0 C10 0 10 10 0 10 S -10 20 0 20 Q 5 25 10 20 T 20 20 Z") == REFERENCE["path2"]


def test_arc_flags_without_separators_match_js():
    """``a5 5 0 011 1`` の «011» を «0, 1, 1» と読めること。

    フラグを数値として読むと ``011`` を 11 と解釈し、**円弧が明後日の方向に
    飛びます。** 実際のロゴの SVG によくある書き方です。
    """
    assert core.path_to_subpaths("M10 10 a5 5 0 011 1") == REFERENCE["path3"]


def test_arc_to_cubics_matches_js():
    """円弧をベジェに直した制御点が JS 版と一致すること。

    **ここだけは完全一致ではなく 1e-14 の相対誤差で比べます。** V8 と CPython は
    ``tan`` / ``atan2`` の最下位ビットが環境によって 1 だけ違うことがあり、
    Movo 側で埋められません。折れ線に落とした結果
    （:func:`test_arc_flags_without_separators_match_js`）は完全に一致するので、
    絵には出ない差です。
    """
    got = core.arc_to_cubics(0, 0, 10, 5, 30, True, False, 8, 6)
    want = REFERENCE["arc"]
    assert len(got) == len(want)
    for curve, expected in zip(got, want):
        assert curve == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_implicit_lineto_after_moveto_matches_js():
    """``M`` の 2 組目以降が暗黙の ``L`` になること。"""
    assert core.parse_path_data("M0 0 l1 1 2 2 3 3") == REFERENCE["parsed"]


def test_transform_matches_js():
    assert core.parse_transform("translate(10 20) rotate(30 5 5) scale(2) skewX(15)") == REFERENCE["transform"]


def test_svg_extraction_matches_js():
    parsed = core.extract_svg_shapes(_SVG)
    assert parsed["stats"] == REFERENCE["svg"]["stats"]
    assert parsed["viewBox"] == REFERENCE["svg"]["viewBox"]
    assert parsed["subpaths"] == REFERENCE["svg"]["subpaths"]


def test_trim_matches_js():
    square = core.path_to_subpaths("M0 0 L10 0 L10 10 L0 10 Z")
    assert core.trim_subpaths(square, {"start": 0.1, "end": 0.6}) == REFERENCE["trim"]
    two = core.path_to_subpaths("M0 0 L10 0 M20 0 L30 0")
    assert core.trim_subpaths(two, {"start": 0, "end": 0.5, "mode": "sequential"}) == REFERENCE["trimSeq"]


# ── LUT ─────────────────────────────────────────────────────

_CUBE = 'TITLE "t"\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n'


def test_cube_lut_layout_matches_js():
    """**格子の並び順が «赤が一番速く回る» であること。**

    ここを取り違えると色が «斜めに» 転びます。値の並びそのものを比べます。
    """
    lut = core.parse_cube_lut(_CUBE)
    assert lut.size == REFERENCE["lut"]["size"]
    assert lut.title == REFERENCE["lut"]["title"]
    assert lut.flat.tolist() == REFERENCE["lut"]["data"]
    assert core.identity_lut(3).flat.tolist() == REFERENCE["identityLut3"]


def test_lut_sampling_matches_js():
    lut = core.parse_cube_lut(_CUBE)
    samples = [core.sample_lut(lut, *rgb) for rgb in ([0, 0, 0], [1, 1, 1], [0.25, 0.5, 0.75], [0.5, 0.5, 0.5])]
    assert samples == REFERENCE["lutSamples"]


# ── 仮の絵（決定性） ────────────────────────────────────────


def test_placeholder_is_deterministic_per_asset_name():
    """素材名から決まる絵になること。**JS 版と 1 画素も違わないこと。**"""
    hero = core.create_placeholder({"width": 16, "height": 16}, "hero", 1)
    assert hero.data.reshape(-1).tolist() == REFERENCE["placeholderHero"]
    again = core.create_placeholder({"width": 16, "height": 16}, "hero", 1)
    assert np.array_equal(hero.data, again.data)
    other = core.create_placeholder({"width": 16, "height": 16}, "other", 1)
    assert not np.array_equal(hero.data, other.data)
    assert core.sha256(other.data.tobytes()) == REFERENCE["placeholderOtherSha"]


# ── 閃光検査 / 映像プロファイル ─────────────────────────────


def _strobe_frames(width: int = 160, height: int = 90, count: int = 60):
    """4 フレームに 1 回、全画面が白く飛ぶ映像。赤いブロックが横に流れる。"""
    for i in range(count):
        frame = core.Bitmap.create(width, height, "#ffffff" if i % 4 == 0 else "#101018")
        x0 = (i * 3) % 120
        frame.data[10:40, x0 : x0 + 30] = (240, 20, 20, 255)
        yield frame


def test_flash_guard_matches_js():
    """閃光の数え方が JS 版と同じであること（1 往復＝1 回）。"""
    guard = core.FlashGuard(width=160, height=90, fps=30)
    for frame in _strobe_frames():
        guard.push(frame)
    assert guard.report() == REFERENCE["flash"]
    assert core.describe_flash_report(guard.report()) == REFERENCE["flashText"]


def test_video_profile_matches_js():
    """測った «型» の全項目が JS 版と一致すること（丸め方まで含めて）。"""
    profiler = core.VideoProfiler(width=160, height=90, fps=30)
    for frame in _strobe_frames():
        profiler.push(frame)
    assert profiler.report() == REFERENCE["profile"]


def test_profile_comparison_matches_js():
    profiler = core.VideoProfiler(width=160, height=90, fps=30)
    for frame in _strobe_frames():
        profiler.push(frame)
    target = {"cutSeconds": [1, 5], "colors": [2, 10], "saturation": [0.1, 0.4]}
    assert core.compare_profile(profiler.report(), target) == REFERENCE["compare"]
