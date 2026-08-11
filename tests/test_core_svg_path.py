"""SVG の «安全な取り込み» と、パスの細かい規則のテスト。

SVG は **外から持ち込まれるデータ**です。ロゴを 1 枚もらうたびに、その中身が
何をしてくるか分からない前提で扱います。ここのテストの半分は
«取り込まないこと» を確かめるもので、機能ではなく **境界**の確認です。

もう半分はパスの規則です。``d`` 文字列の書き方には «区切りを省ける» とか
«コマンドを省ける» といった省略が多く、実際のロゴはたいていその省略を
使っています。省略を読み違えると **形が明後日の方向に飛びます。**
"""

from __future__ import annotations

import math

import pytest

import movo.core as core


# ── 取り込まないもの（安全側） ──────────────────────────────


def test_script_and_event_handlers_are_ignored():
    """``<script>`` と ``on*=`` が **形として取り込まれない**こと。"""
    svg = (
        '<svg width="10" height="10">'
        "<script>alert(1)</script>"
        '<path d="M0 0 L10 10" onclick="alert(2)" onload="x()"/>'
        "</svg>"
    )
    parsed = core.extract_svg_shapes(svg)
    assert parsed["stats"]["paths"] == 1
    assert parsed["stats"]["skipped"] >= 1
    assert len(parsed["subpaths"]) == 1


def test_external_references_are_never_followed():
    """``href`` / ``xlink:href`` / ``<image>`` / ``<use>`` を **たどらない**こと。

    たどる処理が 1 つも無いので、SVG がどこから来ても «数字を読むだけ» で済みます。
    """
    svg = (
        '<svg width="10" height="10">'
        '<image href="http://example.invalid/x.png" x="0" y="0" width="10" height="10"/>'
        '<use xlink:href="#gone"/>'
        '<path d="M0 0 L1 1" href="http://example.invalid/y"/>'
        "</svg>"
    )
    parsed = core.extract_svg_shapes(svg)
    assert parsed["stats"]["paths"] == 1
    assert parsed["stats"]["shapes"] == 0


def test_definition_elements_are_skipped_with_their_contents():
    """``<defs>`` の中身は «描かれないもの» なので取り込まないこと。

    取り込むと «見えないはずの形» が画面に出ます。
    """
    svg = (
        '<svg width="10" height="10">'
        '<defs><path d="M0 0 L9 9"/><rect x="0" y="0" width="5" height="5"/></defs>'
        '<clipPath id="c"><circle cx="5" cy="5" r="4"/></clipPath>'
        '<path d="M1 1 L2 2"/>'
        "</svg>"
    )
    parsed = core.extract_svg_shapes(svg)
    assert parsed["stats"]["paths"] == 1
    assert parsed["stats"]["shapes"] == 0


def test_comments_cannot_smuggle_shapes():
    """コメントや CDATA の中の ``<path>`` を拾わないこと。"""
    svg = '<svg width="4" height="4"><!-- <path d="M0 0 L4 4"/> --><![CDATA[<path d="M0 0 L1 1"/>]]></svg>'
    parsed = core.extract_svg_shapes(svg)
    assert parsed["subpaths"] == []


def test_oversized_svg_is_refused_before_parsing():
    """**読む前に**大きさで弾くこと（読んでから弾いてもメモリは食われています）。"""
    big = '<svg width="1" height="1">' + " " * 5000 + "</svg>"
    with pytest.raises(ValueError):
        core.extract_svg_shapes(big, max_bytes=1024)


def test_segment_count_is_capped():
    """壊れた（あるいは意地の悪い）``d`` で無限に確保しないこと。"""
    d = "M0 0" + "".join(f"L{i} {i}" for i in range(500))
    parsed = core.parse_path_data(d, max_segments=50)
    assert parsed["truncated"] is True
    assert len(parsed["segments"]) == 50


def test_element_count_is_capped():
    svg = '<svg width="10" height="10">' + '<path d="M0 0 L1 1"/>' * 50 + "</svg>"
    parsed = core.extract_svg_shapes(svg, max_elements=5)
    assert parsed["stats"]["truncated"] is True
    assert parsed["stats"]["paths"] == 5


# ── パスの規則 ──────────────────────────────────────────────


def test_broken_path_returns_what_it_could_read():
    """壊れた ``d`` は **読めたところまで**返し、壊れていることを伝えること。

    例外にしないのは、ロゴが 1 つ欠けるより «途中まで出る» ほうがましだからです。
    """
    parsed = core.parse_path_data("M0 0 L10 10 L20")
    assert parsed["invalid"] is True
    assert [s["op"] for s in parsed["segments"]] == ["M", "L"]


def test_path_without_a_leading_command_is_invalid():
    parsed = core.parse_path_data("10 10 20 20")
    assert parsed["invalid"] is True
    assert parsed["segments"] == []


def test_z_reopens_the_subpath_at_the_closing_point():
    """``Z`` のあとに ``M`` なしで続く命令は、閉じた点から再開すること。"""
    subpaths = core.path_to_subpaths("M0 0 L10 0 L10 10 Z L20 20")
    assert len(subpaths) == 2
    assert subpaths[0]["closed"] is True
    assert subpaths[1]["points"][:2] == [0.0, 0.0]


def test_quadratic_is_converted_exactly_to_cubic():
    """2 次ベジェ → 3 次ベジェの変換が **厳密** であること（制御点を 2/3 寄せる）。"""
    parsed = core.parse_path_data("M0 0 Q 10 0 10 10")
    cubic = parsed["segments"][1]
    assert cubic["op"] == "C"
    assert cubic["values"] == pytest.approx([20 / 3, 0, 10, 10 / 3, 10, 10])


def test_smooth_curve_mirrors_the_previous_control_point():
    """``S`` は直前の制御点を現在点で鏡映すること。直前が C/S でなければ現在点。"""
    mirrored = core.parse_path_data("M0 0 C 1 1 2 2 3 3 S 5 5 6 6")["segments"][2]
    assert mirrored["values"][:2] == pytest.approx([4.0, 4.0])  # (3,3) で (2,2) を鏡映
    plain = core.parse_path_data("M0 0 S 5 5 6 6")["segments"][1]
    assert plain["values"][:2] == pytest.approx([0.0, 0.0])  # 直前が C/S でない → 現在点


def test_arc_with_too_small_radii_is_enlarged():
    """半径が届かないときは «届く最小の大きさ» まで広げること（仕様どおり）。"""
    curves = core.arc_to_cubics(0, 0, 1, 1, 0, False, True, 10, 0)
    assert curves  # 直線に落ちず、弧として出る
    assert curves[-1][4] == pytest.approx(10)
    assert curves[-1][5] == pytest.approx(0)


def test_arc_with_zero_radius_becomes_a_line():
    assert core.arc_to_cubics(0, 0, 0, 5, 0, False, True, 10, 0) == []
    assert core.path_to_subpaths("M0 0 A0 0 0 0 1 10 0")[0]["points"] == [0.0, 0.0, 10.0, 0.0]


def test_curve_flattening_is_deterministic():
    """同じ入力からは **必ず同じ点列**が出ること（決定性）。"""
    a = core.path_to_subpaths("M0 0 C 30 0 30 30 0 30")
    b = core.path_to_subpaths("M0 0 C 30 0 30 30 0 30")
    assert a == b


def test_transform_is_applied_before_flattening():
    """**行列は «折れ線にする前» に掛かること。**

    折れ線にしてから掛けると、拡大したときに曲線が角張ります。ここでは
    «拡大しても点の数が変わらない» ことでその順番を確かめます
    （後から掛けていれば、拡大前の粗い点列がそのまま拡大されます）。
    """
    plain = core.path_to_subpaths("M0 0 C 10 0 10 10 0 10")
    scaled = core.path_to_subpaths("M0 0 C 10 0 10 10 0 10", transform=[10, 0, 0, 10, 0, 0])
    assert len(scaled[0]["points"]) > len(plain[0]["points"])


def test_basic_shapes_become_paths():
    for svg, expected in [
        ('<svg><rect x="0" y="0" width="4" height="4"/></svg>', 1),
        ('<svg><circle cx="5" cy="5" r="3"/></svg>', 1),
        ('<svg><ellipse cx="5" cy="5" rx="3" ry="2"/></svg>', 1),
        ('<svg><line x1="0" y1="0" x2="5" y2="5"/></svg>', 1),
        ('<svg><polyline points="0,0 1,1 2,0"/></svg>', 1),
        ('<svg><polygon points="0,0 1,1 2,0"/></svg>', 1),
    ]:
        parsed = core.extract_svg_shapes(svg)
        assert len(parsed["subpaths"]) == expected, svg


def test_degenerate_shapes_are_dropped():
    for svg in [
        '<svg><rect x="0" y="0" width="0" height="4"/></svg>',
        '<svg><circle cx="5" cy="5" r="0"/></svg>',
        '<svg><polygon points="0,0"/></svg>',
    ]:
        assert core.extract_svg_shapes(svg)["subpaths"] == []


def test_rounded_rect_uses_arcs():
    square = core.extract_svg_shapes('<svg><rect x="0" y="0" width="10" height="10"/></svg>')
    rounded = core.extract_svg_shapes('<svg><rect x="0" y="0" width="10" height="10" rx="3"/></svg>')
    assert len(rounded["subpaths"][0]["points"]) > len(square["subpaths"][0]["points"])


def test_view_box_falls_back_to_width_and_height():
    parsed = core.extract_svg_shapes('<svg width="120px" height="60px"><path d="M0 0 L1 1"/></svg>')
    assert parsed["viewBox"] == [0.0, 0.0, 120.0, 60.0]
    # 百分率は «分からない» ので 0 扱い（囲む矩形から決める）
    percent = core.extract_svg_shapes('<svg width="100%" height="100%"><path d="M0 0 L4 8"/></svg>')
    assert percent["width"] > 0 and percent["height"] > 0


# ── トリムパス ──────────────────────────────────────────────


def test_trim_is_a_no_op_when_it_removes_nothing():
    subpaths = core.path_to_subpaths("M0 0 L10 0")
    assert core.trim_subpaths(subpaths, {"start": 0, "end": 1}) is subpaths
    assert core.trim_subpaths(subpaths, {"start": 0.2, "end": 0.8, "enabled": False}) is subpaths
    assert core.is_trim_active({"start": 0, "end": 1, "offset": 0}) is False
    assert core.is_trim_active({"start": 0.1}) is True


def test_trim_cuts_by_length_not_by_point_count():
    """**長さの割合**で切ること。点の数で切ると、曲線の «描かれ方» が不均等になります。"""
    line = [{"points": [0.0, 0.0, 100.0, 0.0], "closed": False}]
    half = core.trim_subpaths(line, {"start": 0, "end": 0.5})
    assert half[0]["points"] == pytest.approx([0.0, 0.0, 50.0, 0.0])


def test_trim_wraps_around_a_closed_path():
    """閉じたパスでは ``offset`` が一周して回り込み、またぐと 2 本に割れること。"""
    square = core.path_to_subpaths("M0 0 L10 0 L10 10 L0 10 Z")
    pieces = core.trim_subpaths(square, {"start": 0, "end": 0.5, "offset": 0.8})
    assert len(pieces) == 2
    assert all(piece["closed"] is False for piece in pieces)


def test_trim_result_is_always_open():
    """切った結果は **開いた折れ線**であること（線が描かれていく演出のため）。"""
    square = core.path_to_subpaths("M0 0 L10 0 L10 10 L0 10 Z")
    for piece in core.trim_subpaths(square, {"start": 0.1, "end": 0.6}):
        assert piece["closed"] is False


def test_trim_empty_range_gives_nothing():
    square = core.path_to_subpaths("M0 0 L10 0 L10 10 L0 10 Z")
    assert core.trim_subpaths(square, {"start": 0.5, "end": 0.5}) == []


# ── 行列 ────────────────────────────────────────────────────


def test_transform_functions():
    assert core.identity_matrix() == [1, 0, 0, 1, 0, 0]
    assert core.parse_transform("translate(5 7)") == [1, 0, 0, 1, 5, 7]
    assert core.parse_transform("scale(2)") == [2, 0, 0, 2, 0, 0]
    rotated = core.parse_transform("rotate(90)")
    assert rotated[0] == pytest.approx(0, abs=1e-12)
    assert rotated[1] == pytest.approx(1)
    assert core.parse_transform("skewX(45)")[2] == pytest.approx(math.tan(math.pi / 4))
    assert core.parse_transform("nonsense(1 2)") == core.identity_matrix()
    assert core.parse_transform(None) == core.identity_matrix()


def test_subpaths_bounds_never_returns_zero_size():
    assert core.subpaths_bounds([]) == {"minX": 0, "minY": 0, "maxX": 1, "maxY": 1, "width": 1, "height": 1}
    flat = core.subpaths_bounds([{"points": [0.0, 5.0, 10.0, 5.0], "closed": False}])
    assert flat["height"] > 0
