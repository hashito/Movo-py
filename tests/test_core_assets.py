"""素材の解決・LUT・歌詞のテスト。

ここの主題は **«壊れていても止まらないこと»** です。素材は外から来るので、
無い・壊れている・大きすぎるが普通に起きます。それでプレビューが出ないのは
割に合わないので、警告して仮の絵で進みます。

**ただし ``strict`` のときは例外にします。** ``movo validate`` と CI は
「見逃さない」ほうが仕事だからです。この 2 つの振る舞いが逆になっていないかを
ここで押さえます。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import movo.core as core


@pytest.fixture()
def project(tmp_path):
    """小さなプロジェクトを 1 つ作る。"""
    (tmp_path / "assets").mkdir()
    core.save_image(core.Bitmap.create(8, 6, "#ff8800"), tmp_path / "assets" / "logo.png")
    (tmp_path / "assets" / "look.cube").write_text(
        "LUT_3D_SIZE 2\n" + "".join(f"{r} {g} {b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1)),
        encoding="utf-8",
    )
    (tmp_path / "assets" / "mark.svg").write_text(
        '<svg viewBox="0 0 10 10"><path d="M0 0 L10 10"/></svg>', encoding="utf-8"
    )
    (tmp_path / "assets" / "words.lrc").write_text("[00:01.00]hello\n[00:02.50]world\n", encoding="utf-8")
    (tmp_path / "assets" / "numbers.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (tmp_path / "assets" / "beep.wav").write_bytes(core.encode_wav(core.create_silence(0.01, 8000, 1)))
    (tmp_path / "assets" / "shape.obj").write_text("v 0 0 0\n", encoding="utf-8")
    return tmp_path


def _store(project, assets, **kwargs):
    return core.AssetStore(project_root=str(project), assets=assets, **kwargs)


# ── 種類の推測 ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "path,kind",
    [
        ("a/b.png", "image"), ("a/b.JPG", "image"), ("a/b.webp", "image"),
        ("a/b.wav", "audio"), ("a/b.mp3", "audio"),
        ("a/b.mp4", "video"), ("a/b.mov", "video"),
        ("a/b.json", "data"), ("a/b.ttf", "font"),
        ("a/b.obj", "mesh"), ("a/b.mtl", "mesh"),
        ("a/b.cube", "lut"), ("a/b.svg", "svg"),
    ],
)
def test_extension_decides_the_kind(path: str, kind: str):
    """**拡張子で先に振り分けること。**

    ``.obj`` や ``.cube`` は中身がテキストなので、画像として復号しようとすると
    «形式が分かりません» という的外れな警告が出ます。
    """
    assert core.infer_spec(path)["type"] == kind


def test_urls_are_recognised():
    spec = core.infer_spec("https://example.invalid/a.png")
    assert spec["type"] == "image" and spec["url"].startswith("https://")
    assert "path" not in spec


# ── 読み込み ────────────────────────────────────────────────


def test_loading_every_kind(project):
    store = _store(
        project,
        {
            "logo": "assets/logo.png",
            "look": "assets/look.cube",
            "mark": "assets/mark.svg",
            "words": {"type": "lyrics", "path": "assets/words.lrc"},
            "numbers": "assets/numbers.json",
            "beep": "assets/beep.wav",
            "shape": "assets/shape.obj",
        },
    )
    assert store.resolve_all(strict=True) == []
    assert store.get("logo").width == 8
    assert store.get_lut("look").size == 2
    assert len(store.get_svg("mark")["subpaths"]) == 1
    assert [line["text"] for line in store.describe("words")["lines"]] == ["hello", "world"]
    assert store.describe("numbers")["value"] == {"a": 1}
    assert store.get_audio("beep").sample_rate == 8000
    assert store.describe("shape")["type"] == "mesh"
    assert store.text("shape").startswith("v ")
    assert store.stats() == {"images": 1, "audio": 1, "placeholders": 0}


def test_a_missing_image_becomes_a_placeholder(project):
    """**素材 1 つが無くても絵は出る**こと（プレビューを止めない）。"""
    store = _store(project, {"hero": "assets/gone.png"})
    errors = store.resolve_all()
    assert errors == []
    assert "hero" in store.missing
    assert store.get("hero") is not None
    assert store.describe("hero")["placeholder"] is True


def test_strict_mode_refuses_a_missing_non_image(project):
    """``strict`` では見逃さないこと（``movo validate`` / CI の仕事）。"""
    store = _store(project, {"song": "assets/gone.wav"})
    with pytest.raises(core.MovoError) as info:
        store.resolve_all(strict=True)
    assert info.value.code == core.ErrorCodes.MOVO_ASSET_NOT_FOUND


def test_resolve_all_collects_errors_instead_of_stopping(project):
    store = _store(project, {"ok": "assets/logo.png", "bad": "assets/gone.wav"})
    errors = store.resolve_all()
    assert len(errors) == 1
    assert store.get("ok") is not None  # 壊れた 1 つで他が巻き添えにならない


def test_fallback_asset_is_used(project):
    store = _store(project, {"hero": {"path": "assets/gone.png", "fallback": "logo"}, "logo": "assets/logo.png"})
    store.load("hero")
    assert store.get("logo").width == 8


def test_undeclared_asset_is_an_error(project):
    store = _store(project, {})
    with pytest.raises(core.MovoError) as info:
        store.load("nothing")
    assert info.value.path == "assets.nothing"


def test_network_is_blocked_when_the_project_says_so(project):
    """``security.allowNetwork: false`` なら **取りに行かないこと**。"""
    store = _store(
        project,
        {"remote": {"type": "image", "url": "https://example.invalid/a.png"}},
        security={"allowNetwork": False, "maxDownloadSizeMB": 100},
    )
    with pytest.raises(core.MovoError) as info:
        store.load("remote")
    assert info.value.code == core.ErrorCodes.MOVO_NETWORK_DENIED


def test_oversized_lut_and_svg_are_refused_before_reading(project):
    """**読む前に**大きさで弾くこと。読んでから弾いてもメモリは食われています。"""
    store = _store(project, {"look": "assets/look.cube"}, security={"maxDownloadSizeMB": 0})
    with pytest.raises(core.MovoError) as info:
        store.load("look")
    assert info.value.code == core.ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE

    svg_store = _store(project, {"mark": "assets/mark.svg"}, security={"maxSvgSizeMB": 0})
    # maxSvgSizeMB が 0（未指定扱い）なら既定の 2 MB が効いて通ること
    assert svg_store.load("mark")["type"] == "svg"


def test_a_broken_lut_warns_instead_of_stopping_the_render(project):
    """エフェクトから引くときは **警告して None**（色が付かないだけで済ませる）。"""
    (project / "assets" / "broken.cube").write_text("LUT_3D_SIZE 4\n1 2 3\n", encoding="utf-8")
    store = _store(project, {"look": "assets/broken.cube"})
    assert store.get_lut("look") is None
    # ただし load() は例外にする（validate が見逃さないように）
    with pytest.raises(core.MovoError):
        store.load("look")


def test_ai_asset_without_a_generator_falls_back_to_a_placeholder(project):
    store = _store(project, {"hero": {"type": "ai-image", "prompt": "a cat"}})
    meta = store.load("hero")
    assert meta["placeholder"] is True and meta["generated"] is False
    assert store.get("hero") is not None


def test_generator_callback_is_used_when_present(project):
    def generator(*, name, spec, store, generate):
        return {"bitmap": core.Bitmap.create(4, 4, "#00ff00"), "provider": "test"}

    store = _store(project, {"hero": {"type": "ai-image"}}, generator=generator)
    meta = store.load("hero")
    assert meta["generated"] is True and meta["provider"] == "test"
    assert store.get("hero").width == 4


# ── LUT ─────────────────────────────────────────────────────


def test_lut_size_limit_prevents_huge_allocations():
    """``LUT_3D_SIZE`` **そのもの**に上限があること。

    行数の上限だけだと ``LUT_3D_SIZE 2000`` の 1 行で 2000³ × 3 個を先に
    確保してしまい、そこで落ちます。
    """
    with pytest.raises(core.MovoError) as info:
        core.parse_cube_lut("LUT_3D_SIZE 2000\n")
    assert info.value.code == core.ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE


def test_lut_1d_is_refused_with_a_pointer_to_curves():
    with pytest.raises(core.MovoError) as info:
        core.parse_cube_lut("LUT_1D_SIZE 16\n")
    assert info.value.code == core.ErrorCodes.MOVO_UNSUPPORTED
    assert "curves" in info.value.hint


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1 2 3\n",  # LUT_3D_SIZE より前に数値
        "LUT_3D_SIZE 2\n0 0 0\n",  # 行が足りない
        "LUT_3D_SIZE 2\n" + "0 0 0\n" * 9,  # 行が多い
        "LUT_3D_SIZE 2\n" + "a b c\n" * 8,  # 数値でない
        "LUT_3D_SIZE 2\n" + "0 0\n" * 8,  # 値が 3 つない
        "TITLE x\n",  # LUT_3D_SIZE が無い
    ],
)
def test_broken_cube_files_are_refused(text: str):
    with pytest.raises(core.MovoError):
        core.parse_cube_lut(text)


def test_lut_comments_and_unknown_keywords_are_skipped():
    text = "# 注釈\nTITLE \"look\"\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\nWHATEVER 3\nLUT_3D_SIZE 2\n"
    text += "".join(f"{r} {g} {b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1))
    lut = core.parse_cube_lut(text)
    assert lut.title == "look"
    assert lut.size == 2


def test_zero_width_domain_is_repaired():
    """幅 0 の定義域は割り算で壊れるので直しておくこと。"""
    text = "LUT_3D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 1 1 1\n"
    text += "".join(f"{r} {g} {b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1))
    lut = core.parse_cube_lut(text)
    assert lut.domain_min == [0.0, 0.0, 0.0]
    assert lut.domain_max == [1.0, 1.0, 1.0]


def test_identity_lut_leaves_colours_alone():
    """**何もしない LUT が本当に何もしないこと。** 並び順の取り違えの番人です。"""
    image = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    graded = core.apply_lut(image, core.identity_lut(33))
    assert np.allclose(graded, image, atol=2e-3)


def test_lut_amount_blends_towards_the_original():
    image = np.full((2, 2, 3), 0.25, np.float32)
    inverted = core.identity_lut(2)
    inverted.data[:] = 1.0 - inverted.data
    full = core.apply_lut(image, inverted, 1.0)
    half = core.apply_lut(image, inverted, 0.5)
    assert np.allclose(half, image + (full - image) * 0.5, atol=1e-6)
    assert np.array_equal(core.apply_lut(image, inverted, 0.0), image)


def test_lut_grid_corners_are_where_we_think():
    """**赤が一番速く回る**こと。取り違えると色が «斜めに» 転びます。"""
    lut = core.identity_lut(2)
    assert core.sample_lut(lut, 1.0, 0.0, 0.0) == pytest.approx([1, 0, 0])
    assert core.sample_lut(lut, 0.0, 1.0, 0.0) == pytest.approx([0, 1, 0])
    assert core.sample_lut(lut, 0.0, 0.0, 1.0) == pytest.approx([0, 0, 1])


# ── 歌詞 ────────────────────────────────────────────────────


def test_lyrics_format_is_detected_from_the_content_not_the_name():
    """**拡張子は当てになりません**（``.txt`` に LRC が入っていることがよくあります）。"""
    assert core.detect_lyrics_format("[00:01.00]hello") == "lrc"
    assert core.detect_lyrics_format("00:00:01,000 --> 00:00:02,000") == "subtitle"
    assert core.detect_lyrics_format("just words") == "unknown"


def test_plain_text_lyrics_are_refused_with_a_pointer_to_the_other_way():
    """時刻の無いただの行は **受けないこと**（等分したいなら別の道があります）。"""
    with pytest.raises(core.MovoError) as info:
        core.parse_lyrics("hello\nworld\n")
    assert "配列" in info.value.hint


def test_the_last_line_has_no_duration():
    """最後の 1 行にだけ ``for`` を付けないこと（«次» が無いので決められません）。"""
    lines = core.parse_lrc("[00:01.00]a\n[00:02.00]b\n")
    assert lines[0]["for"] == pytest.approx(1.0)
    assert "for" not in lines[1]


def test_overlapping_slice_keeps_lines_that_cross_the_boundary():
    """**シーンの境目をまたぐ行が消えないこと。**

    大サビを 6.5 秒ごとに割ったとき、4.86 秒間隔の歌詞が境目をまたいで
    何行も消えたのがこの機能の由来です。
    """
    lines = [{"text": "a", "at": 1.0, "for": 4.0}]
    assert core.slice_lyrics(lines, 2.0, 8.0) == []  # 既定は «その範囲で始まる行» だけ
    kept = core.slice_lyrics(lines, 2.0, 8.0, overlap=True)
    assert len(kept) == 1
    assert kept[0]["at"] == 0.0  # 途中から始まる行は 0 に丸める
    assert kept[0]["for"] == pytest.approx(3.0)  # 見えている時間は変わらない


def test_tiny_fragments_are_dropped():
    """一瞬しか見えない断片は捨てること（読めないうえフェードの尺が取れません）。"""
    lines = [{"text": "a", "at": 0.0, "for": 5.0}]
    assert core.slice_lyrics(lines, 4.95, 10.0, overlap=True) == []
    assert len(core.slice_lyrics(lines, 4.95, 10.0, overlap=True, min_span=0)) == 1


def test_syllable_timings_are_shifted_too():
    lines = core.parse_lrc("[00:05.00]<00:05.00>ね<00:05.62>え\n")
    shifted = core.slice_lyrics(lines, 4.0, 10.0)
    assert shifted[0]["syllables"][0]["at"] == pytest.approx(1.0)
    assert shifted[0]["syllables"][1]["at"] == pytest.approx(1.62)


def test_vtt_header_is_ignored():
    text = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhello\n"
    assert [line["text"] for line in core.parse_lyrics(text)] == ["hello"]
