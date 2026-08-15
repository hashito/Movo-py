"""TrueType パーサ（``movo.renderer.font``）のテスト。

**実機のシステムフォントを使います**。合成のテスト用フォントを作ると、
「自分で書いた形を自分で読めた」だけになり、実際のフォントで起きる
（loca の短い形式・合成グリフ・cmap format 12 のような）事情を踏めないためです。

その代わり、フォントが 1 つも見つからない環境では **skip** します。CI の
コンテナにはフォントが入っていないことがあるので、そこで赤くなっても
何も分からないからです。
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from movo.renderer import font as font_module
from movo.renderer.font import (
    FONT_EXTENSIONS,
    Font,
    FontManager,
    Glyph,
    covers_cjk,
    is_cjk,
    list_font_files,
    to_sfnt,
)


# --------------------------------------------------------------------------
# 共通の下ごしらえ
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def font_files() -> list[str]:
    """このマシンにあるフォントファイル。1 つも無ければ以降を skip します。"""
    files = list_font_files()
    if not files:
        pytest.skip("このマシンにはシステムフォントが見つかりません")
    return files


@pytest.fixture(scope="module")
def latin_font(font_files: list[str]) -> Font:
    """``A`` を持っている、実際に開けたフォントを 1 つ返す。

    CFF（OpenType/PostScript）のフォントは仕様どおり例外になるので、
    読めるものが出るまで順に試します。
    """
    for path in font_files[:200]:
        try:
            font = Font.load(path)
        except Exception:
            continue
        if font.glyph_index_for(ord("A")) != 0:
            return font
    pytest.skip("ラテン文字を持つ TrueType フォントが見つかりません")


@pytest.fixture(scope="module")
def manager() -> FontManager:
    return FontManager()


# --------------------------------------------------------------------------
# ファイル探索
# --------------------------------------------------------------------------


def test_list_font_files_finds_something(font_files: list[str]) -> None:
    assert len(font_files) >= 1
    for path in font_files:
        assert os.path.isabs(path)
        assert path.lower().endswith((".ttf", ".otf", ".ttc", ".otc"))


def test_list_font_files_respects_limit() -> None:
    files = list_font_files(limit=3)
    assert len(files) <= 3


def test_list_font_files_has_no_duplicates(font_files: list[str]) -> None:
    lowered = [p.lower() for p in font_files]
    assert len(lowered) == len(set(lowered))


def test_font_extensions_pattern() -> None:
    for name in ("a.ttf", "a.TTF", "a.otf", "a.ttc", "a.woff", "a.woff2"):
        assert FONT_EXTENSIONS.search(name), name
    for name in ("a.png", "a.ttf.txt", "ttf"):
        assert not FONT_EXTENSIONS.search(name), name


# --------------------------------------------------------------------------
# フォントを開く
# --------------------------------------------------------------------------


def test_font_loads_with_sane_metrics(latin_font: Font) -> None:
    assert latin_font.units_per_em > 0
    assert latin_font.num_glyphs > 0
    assert latin_font.ascender > latin_font.descender
    assert isinstance(latin_font.family_name, str) and latin_font.family_name
    assert isinstance(latin_font.subfamily_name, str) and latin_font.subfamily_name
    assert isinstance(latin_font.full_name, str) and latin_font.full_name
    assert latin_font.file_path is not None


def test_metrics_property_keys(latin_font: Font) -> None:
    metrics = latin_font.metrics
    assert set(metrics) == {"units_per_em", "ascender", "descender", "line_gap"}
    assert metrics["units_per_em"] == latin_font.units_per_em


def test_glyph_index_for_letter_a(latin_font: Font) -> None:
    assert latin_font.glyph_index_for(ord("A")) != 0
    assert latin_font.has_glyph(ord("A"))


def test_unmapped_code_point_is_zero(latin_font: Font) -> None:
    # 私用領域の最後の方は、まず割り当てられていません。
    assert latin_font.glyph_index_for(0x10FFFD) == 0
    assert not latin_font.has_glyph(0x10FFFD)


def test_character_count_is_positive(latin_font: Font) -> None:
    assert latin_font.character_count() > 0


def test_advance_width_is_positive(latin_font: Font) -> None:
    gi = latin_font.glyph_index_for(ord("A"))
    assert latin_font.advance_width(gi) > 0


# --------------------------------------------------------------------------
# 字形の輪郭
# --------------------------------------------------------------------------


def test_glyph_contours_shape(latin_font: Font) -> None:
    gi = latin_font.glyph_index_for(ord("A"))
    glyph = latin_font.glyph(gi)
    assert isinstance(glyph, Glyph)
    assert len(glyph.contours) >= 1
    for contour in glyph.contours:
        assert isinstance(contour, np.ndarray)
        assert contour.dtype == np.float64
        assert contour.ndim == 2
        assert contour.shape[1] == 3
        assert contour.shape[0] > 0  # 空の輪郭は入れない決まり
        # on_curve の列は 0.0 か 1.0 だけ
        assert set(np.unique(contour[:, 2]).tolist()) <= {0.0, 1.0}


def test_glyph_bounds_and_advance(latin_font: Font) -> None:
    gi = latin_font.glyph_index_for(ord("A"))
    glyph = latin_font.glyph(gi)
    assert glyph.x_max >= glyph.x_min
    assert glyph.advance > 0
    xs = np.concatenate([c[:, 0] for c in glyph.contours])
    # 実測の座標が xMin/xMax の外に大きくはみ出していないこと（1 単位の丸めは許す）。
    assert xs.min() >= glyph.x_min - 1
    assert xs.max() <= glyph.x_max + 1


def test_glyph_is_cached(latin_font: Font) -> None:
    gi = latin_font.glyph_index_for(ord("A"))
    assert latin_font.glyph(gi) is latin_font.glyph(gi)


def test_glyph_out_of_range_is_empty(latin_font: Font) -> None:
    glyph = latin_font.glyph(latin_font.num_glyphs + 10)
    assert glyph.contours == []


def test_space_glyph_has_no_contours(latin_font: Font) -> None:
    gi = latin_font.glyph_index_for(ord(" "))
    if gi == 0:
        pytest.skip("空白の字形がありません")
    assert latin_font.glyph(gi).contours == []


def test_many_glyphs_parse_without_error(latin_font: Font) -> None:
    """先頭 200 字形をひととおり読んで、途中で壊れないことを見ます。

    合成グリフ（アクセント付きの文字など）はこの範囲に必ず紛れ込むので、
    合成の変換が壊れているとここで落ちます。
    """
    for gi in range(min(200, latin_font.num_glyphs)):
        glyph = latin_font.glyph(gi)
        for contour in glyph.contours:
            assert contour.shape[1] == 3
            assert np.isfinite(contour).all()


def test_composite_glyph_matches_component(latin_font: Font) -> None:
    """合成グリフ（例: Ä）が、部品（A）より多くの輪郭を持つこと。"""
    base = latin_font.glyph_index_for(ord("A"))
    accented = latin_font.glyph_index_for(0x00C4)  # Ä
    if base == 0 or accented == 0:
        pytest.skip("Ä か A の字形がありません")
    assert len(latin_font.glyph(accented).contours) >= len(latin_font.glyph(base).contours)


# --------------------------------------------------------------------------
# to_sfnt
# --------------------------------------------------------------------------


def test_to_sfnt_passes_through_plain_truetype(latin_font: Font) -> None:
    assert latin_font.file_path is not None
    with open(latin_font.file_path, "rb") as handle:
        raw = handle.read(4096)
    assert to_sfnt(raw) is raw


def test_to_sfnt_ignores_short_buffers() -> None:
    assert to_sfnt(b"") == b""
    assert to_sfnt(b"ab") == b"ab"


def test_woff2_without_brotli_raises_readable_error() -> None:
    """brotli が無い環境では «brotli が要ります» と分かる例外になること。"""
    try:
        import brotli  # noqa: F401

        pytest.skip("brotli が入っているので、この分岐は通りません")
    except ImportError:
        pass
    try:
        import brotlicffi  # noqa: F401

        pytest.skip("brotlicffi が入っているので、この分岐は通りません")
    except ImportError:
        pass

    # 'wOF2' + それらしいヘッダ（テーブル 0 個）。展開に入る前に例外になります。
    buf = bytearray(48)
    buf[0:4] = b"wOF2"
    buf[4:8] = (0x00010000).to_bytes(4, "big")
    buf[12:14] = (0).to_bytes(2, "big")
    with pytest.raises(Exception) as info:
        to_sfnt(bytes(buf))
    assert "brotli" in str(info.value).lower()


# --------------------------------------------------------------------------
# is_cjk
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code_point,expected",
    [
        (0x2FFF, False),
        (0x3000, True),
        (0x30FF, True),
        (0x3100, False),
        (0x3400, True),
        (0x4DBF, True),
        (0x4DC0, False),
        (0x4E00, True),
        (0x9FFF, True),
        (0xA000, False),
        (0xF900, True),
        (0xFAFF, True),
        (0xFF00, True),
        (0xFFEF, True),
        (0xFFF0, False),
        (ord("A"), False),
        (ord("あ"), True),
        (ord("漢"), True),
    ],
)
def test_is_cjk(code_point: int, expected: bool) -> None:
    assert is_cjk(code_point) is expected


# --------------------------------------------------------------------------
# FontManager
# --------------------------------------------------------------------------


def test_resolve_by_family(manager: FontManager) -> None:
    font = manager.resolve("Arial")
    assert isinstance(font, Font)
    assert font.units_per_em > 0


def test_resolve_unknown_family_falls_back(manager: FontManager) -> None:
    font = manager.resolve("この名前のフォントは存在しません 12345")
    assert isinstance(font, Font)


def test_resolve_generic_family_is_default(manager: FontManager) -> None:
    assert manager.resolve("sans-serif") is manager.default_font()


def test_resolve_none_is_default(manager: FontManager) -> None:
    assert manager.resolve() is manager.default_font()


def test_default_font_is_cached(manager: FontManager) -> None:
    assert manager.default_font() is manager.default_font()


def test_resolve_by_path(manager: FontManager, latin_font: Font) -> None:
    assert latin_font.file_path is not None
    font = manager.resolve(latin_font.file_path)
    assert font.family_name == latin_font.family_name


def test_resolve_declared_family(latin_font: Font) -> None:
    assert latin_font.file_path is not None
    local = FontManager(fonts={"Main": latin_font.file_path})
    assert local.resolve("Main").family_name == latin_font.family_name


def test_resolve_declared_weight_map(latin_font: Font) -> None:
    """ウェイトごとにファイルを書いた形も受けること。"""
    assert latin_font.file_path is not None
    local = FontManager(fonts={"Main": {"regular": latin_font.file_path}})
    assert local.resolve("Main").family_name == latin_font.family_name
    # bold を頼んでも、無ければ regular に落ちます。
    assert local.resolve("Main", bold=True).family_name == latin_font.family_name


def test_resolve_list_is_explicit_fallback(manager: FontManager, latin_font: Font) -> None:
    assert latin_font.file_path is not None
    font = manager.resolve(["存在しない家族名 999", latin_font.file_path])
    assert font.family_name == latin_font.family_name


def test_fallback_chain_is_stable(manager: FontManager) -> None:
    chain = manager.fallback_chain()
    assert chain is manager.fallback_chain()
    assert all(isinstance(f, Font) for f in chain)
    # 同じフォントが二度入っていないこと
    assert len({id(f) for f in chain}) == len(chain)


def test_font_for_code_point_keeps_primary_when_possible(
    manager: FontManager, latin_font: Font
) -> None:
    assert manager.font_for_code_point(latin_font, ord("A")) is latin_font


def test_font_for_code_point_falls_back_to_cjk(manager: FontManager) -> None:
    """日本語の字が主フォントに無ければ、CJK の面へ落ちること。"""
    primary = manager.resolve("Arial")
    code = ord("あ")
    if primary.has_glyph(code):
        pytest.skip("主フォントがすでに «あ» を持っています（落ちる先を試せません）")
    if not any(f.has_glyph(code) for f in manager.fallback_chain()):
        pytest.skip("このマシンには日本語フォントがありません")
    picked = manager.font_for_code_point(primary, code)
    assert picked is not None
    assert picked is not primary
    assert picked.has_glyph(code)


def test_font_for_code_point_with_none_primary(manager: FontManager) -> None:
    picked = manager.font_for_code_point(None, ord("A"))
    if picked is not None:
        assert picked.has_glyph(ord("A"))


def test_check_font_ok(manager: FontManager, latin_font: Font) -> None:
    assert latin_font.file_path is not None
    info = manager.check_font(latin_font.file_path)
    assert info["ok"] is True
    assert info["family"] == latin_font.family_name
    assert info["glyphs"] > 0
    assert info["characters"] > 0
    assert info["units_per_em"] == latin_font.units_per_em


def test_check_font_missing_file(manager: FontManager, tmp_path) -> None:
    info = manager.check_font(str(tmp_path / "no-such-font.ttf"))
    assert info["ok"] is False
    assert info["error"]


def test_missing_glyphs_for_ascii(manager: FontManager) -> None:
    assert manager.missing_glyphs("Hello, Movo!", "Arial") == []


def test_missing_glyphs_ignores_whitespace(manager: FontManager) -> None:
    assert manager.missing_glyphs(" \n\t", "Arial") == []


def test_missing_glyphs_reports_unmappable(manager: FontManager) -> None:
    # 未割り当ての私用領域。どのフォントにも入っていないはずです。
    missing = manager.missing_glyphs("A\U000FFFFDA", "Arial")
    assert missing == ["\U000FFFFD"]


def test_describe(manager: FontManager) -> None:
    info = manager.describe()
    assert info["font_file_count"] >= 1
    assert info["font_file_count"] == info["fontFileCount"]
    assert isinstance(info["fallbacks"], list)


# --------------------------------------------------------------------------
# 太さの見分けと、CJK の面選び
#
# ここだけは **実機のフォントに頼りません**。「太字の CJK 面が無いマシン」を
# 実機で用意できない（あるいは逆に、あるマシンでは再現しない）ので、面の並びを
# 作り物で固定します。実機で起きた不具合は
# 「weight:"bold" のとき日本語が豆腐になる」でした。
# --------------------------------------------------------------------------


def _os2_font(weight: int | None, subfamily: str = "Regular", file_path: str = "Stub.ttf") -> Font:
    """OS/2 の usWeightClass だけを持つ、解析済みのふりをした Font。"""
    font = Font.__new__(Font)
    font.tables = {}
    font.buffer = b""
    font.file_path = file_path
    font.subfamily_name = subfamily
    if weight is not None:
        buffer = bytearray(64)
        struct.pack_into(">H", buffer, 4, weight)
        # fsSelection の BOLD ビット（32）も立てて «ビットだけ太字» を再現します。
        struct.pack_into(">H", buffer, 62, 32)
        font.buffer = bytes(buffer)
        font.tables = {"OS/2": (0, len(buffer))}
    return font


def test_weight_class_defaults_to_regular_without_os2() -> None:
    assert _os2_font(None).weight_class == 400


def test_weight_class_reads_os2() -> None:
    assert _os2_font(700).weight_class == 700


def test_is_bold_face_from_weight_class() -> None:
    assert _os2_font(700).is_bold_face is True
    assert _os2_font(400).is_bold_face is False


def test_is_bold_face_ignores_fs_selection_bold_bit() -> None:
    """usWeightClass が 400 のままの «Medium» を太字と数えないこと。

    Apple の STHeiti Medium がこれで、太字として選ぶと «太くならない上に
    書体まで中国語のものに変わる» という一番分かりにくい結果になります。
    """
    assert _os2_font(400, subfamily="Medium", file_path="STHeiti Medium.ttc").is_bold_face is False


def test_is_bold_face_from_japanese_weight_number() -> None:
    """OS/2 が無くても «W6» 以上のファイル名は太字と見なすこと。"""
    assert _os2_font(None, file_path="/x/ヒラギノ角ゴシック W6.ttc").is_bold_face is True
    assert _os2_font(None, file_path="/x/ヒラギノ角ゴシック W3.ttc").is_bold_face is False


class _FakeFace:
    """面の «並び» だけを試すための最小のフォント。"""

    def __init__(
        self,
        name: str,
        *,
        bold: bool = False,
        cjk: bool = True,
        latin: bool = True,
        family: str | None = None,
    ) -> None:
        self.full_name = name
        self.family_name = family if family is not None else name
        self.subfamily_name = "Bold" if bold else "Regular"
        self.weight_class = 700 if bold else 400
        self.is_bold_face = bold
        self.file_path = f"/fake/{name}.ttf"
        self._cjk = cjk
        self._latin = latin

    def glyph_index_for(self, code_point: int) -> int:
        covered = self._cjk if is_cjk(code_point) else self._latin
        return 1 if covered else 0

    def glyph(self, glyph_index: int, depth: int = 0) -> Glyph:
        return Glyph([np.zeros((3, 3))] if glyph_index else [])

    def has_glyph(self, code_point: int) -> bool:
        return self.glyph_index_for(code_point) != 0


def _fake_manager(monkeypatch, faces: dict[str, _FakeFace]) -> FontManager:
    """指定した面«だけ»が入っているマシンを作る。"""
    manager = FontManager()
    monkeypatch.setattr(manager, "_font_files", lambda: list(faces))
    monkeypatch.setattr(manager, "_load_file", lambda file, face_index=0: faces.get(file))
    # 汎用のフォールバック鎖は «CJK が 1 つも入らないことがある» のが今回の不具合の
    # 半分なので、空にして «CJK の面選び» だけを見ます。
    monkeypatch.setattr(manager, "fallback_chain", lambda: [])
    return manager


@pytest.fixture()
def warnings(monkeypatch) -> list[str]:
    """``_log_warn`` に渡された文字列を集める。"""
    collected: list[str] = []
    monkeypatch.setattr(font_module, "_log_warn", collected.append)
    return collected


_JP = ord("決")


def test_covers_cjk_needs_kanji_and_kana() -> None:
    assert covers_cjk(_FakeFace("Full")) is True
    assert covers_cjk(_FakeFace("LatinOnly", cjk=False)) is False


def test_covers_cjk_rejects_empty_outlines() -> None:
    """cmap には載っているが輪郭が空の面は «描ける» と数えないこと。"""

    class _Hollow(_FakeFace):
        def glyph(self, glyph_index: int, depth: int = 0) -> Glyph:
            return Glyph([])

    assert covers_cjk(_Hollow("Hollow")) is False


def test_cjk_faces_keeps_candidates_when_the_first_file_cannot_be_read(monkeypatch) -> None:
    """名前が当たった 1 本目が開けなくても、次の候補を捨てないこと。

    macOS の «Hiragino Sans GB.ttc» は CFF なので読めません。そこで打ち切って
    いたために、CJK の候補が 1 つも残っていませんでした。
    """
    good = _FakeFace("NotoSansJP")
    faces = {"/fake/NotoSansCJK-Broken.ttc": None, "/fake/NotoSansJP-Regular.ttf": good}
    manager = FontManager()
    monkeypatch.setattr(manager, "_font_files", lambda: list(faces))
    monkeypatch.setattr(manager, "_load_file", lambda file, face_index=0: faces.get(file))
    assert manager.cjk_faces() == [good]


def test_cjk_face_prefers_the_bold_face(monkeypatch, warnings: list[str]) -> None:
    regular = _FakeFace("NotoSansJP Regular")
    bold = _FakeFace("NotoSansJP Bold", bold=True)
    manager = _fake_manager(
        monkeypatch,
        {"/fake/NotoSansJP-Regular.ttf": regular, "/fake/NotoSansJP-Bold.ttf": bold},
    )
    assert manager.cjk_face(bold=True) is bold
    assert warnings == []


def test_cjk_face_drops_to_regular_weight_and_warns(monkeypatch, warnings: list[str]) -> None:
    """太字の CJK 面が無い環境では、豆腐を出さずに通常ウェイトへ落ちること。"""
    regular = _FakeFace("NotoSansJP Regular")
    manager = _fake_manager(monkeypatch, {"/fake/NotoSansJP-Regular.ttf": regular})
    assert manager.cjk_face(bold=True) is regular
    assert len(warnings) == 1
    assert "bold" in warnings[0].lower()
    assert "NotoSansJP Regular" in warnings[0]
    # 1 文字ごとに呼ばれるので、2 度目からは黙ること。
    assert manager.cjk_face(bold=True) is regular
    assert len(warnings) == 1


def test_cjk_face_does_not_warn_without_bold(monkeypatch, warnings: list[str]) -> None:
    regular = _FakeFace("NotoSansJP Regular")
    manager = _fake_manager(monkeypatch, {"/fake/NotoSansJP-Regular.ttf": regular})
    assert manager.cjk_face() is regular
    assert warnings == []


def test_cjk_face_finds_faces_whose_file_name_is_not_ascii(monkeypatch) -> None:
    """ファイル名が日本語のフォントも拾えること。

    macOS の «ヒラギノ角ゴシック W6.ttc» はどの英字名にも当たらないので、
    ファイル名で見当がつかないときは総当たりで探します。
    """
    face = _FakeFace("Hiragino Sans W6", bold=True)
    manager = _fake_manager(monkeypatch, {"/fake/ヒラギノ角ゴシック W6.ttc": face})
    assert manager.cjk_faces() == []
    assert manager.cjk_face(bold=True) is face


def test_cjk_face_skips_private_system_faces(monkeypatch) -> None:
    """先頭が . の面（Apple の «.Aqua かな» など）は本文に使わないこと。"""
    private = _FakeFace(".Aqua Kana", family=".Aqua かな")
    manager = _fake_manager(monkeypatch, {"/fake/AquaKana.ttc": private})
    assert manager.cjk_face() is None


def test_font_for_code_point_keeps_cjk_when_bold_is_requested(
    monkeypatch, warnings: list[str]
) -> None:
    """weight:"bold" でも CJK が候補に残ること（この Issue の本体）。

    太字のラテン面（Arial Bold のように日本語を持たない面）が主フォントでも、
    日本語は CJK の面で描かれます。
    """
    latin_bold = _FakeFace("Arial Bold", bold=True, cjk=False)
    cjk = _FakeFace("NotoSansJP Regular")
    manager = _fake_manager(monkeypatch, {"/fake/NotoSansJP-Regular.ttf": cjk})
    manager._remember_weight(latin_bold, True)
    assert manager.font_for_code_point(latin_bold, _JP) is cjk
    assert len(warnings) == 1


def test_font_for_code_point_leaves_latin_alone(monkeypatch, warnings: list[str]) -> None:
    """ラテン文字の道筋は変わっていないこと（他のスキルを壊さない）。"""
    latin_bold = _FakeFace("Arial Bold", bold=True, cjk=False)
    cjk = _FakeFace("NotoSansJP Regular")
    manager = _fake_manager(monkeypatch, {"/fake/NotoSansJP-Regular.ttf": cjk})
    manager._remember_weight(latin_bold, True)
    assert manager.font_for_code_point(latin_bold, ord("A")) is latin_bold
    assert warnings == []


def test_font_for_code_point_prefers_bold_cjk_over_the_generic_chain(monkeypatch) -> None:
    """鎖に通常ウェイトの CJK が居ても、太字の CJK 面があればそちらを使うこと。"""
    latin_bold = _FakeFace("Arial Bold", bold=True, cjk=False)
    regular = _FakeFace("NotoSansJP Regular")
    bold = _FakeFace("NotoSansJP Bold", bold=True)
    manager = FontManager()
    faces = {"/fake/NotoSansJP-Regular.ttf": regular, "/fake/NotoSansJP-Bold.ttf": bold}
    monkeypatch.setattr(manager, "_font_files", lambda: list(faces))
    monkeypatch.setattr(manager, "_load_file", lambda file, face_index=0: faces.get(file))
    monkeypatch.setattr(manager, "fallback_chain", lambda: [regular])
    manager._remember_weight(latin_bold, True)
    assert manager.font_for_code_point(latin_bold, _JP) is bold


def test_resolve_records_the_requested_weight(manager: FontManager) -> None:
    """resolve() で頼んだ太さが、欠字で落ちるときまで残っていること。"""
    bold = manager.resolve(None, bold=True)
    assert manager._requested_bold.get(bold) is True
    regular = manager.resolve(None, bold=False)
    assert manager._requested_bold.get(regular) is False
