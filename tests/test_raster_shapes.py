"""図形レイヤー・SVG パス・トリムパスの試験。

移植の «落とし穴» を狙って書いてあります。とくに次の 3 つは、直したときに
壊しやすいので必ず残しておいてください。

  1. 円弧のフラグが詰めて書かれた `a1 1 0 011 1`（数値として読むと `011` を 11 と読む）
  2. トリムは **最後に** 掛ける（先に切ると囲む矩形が育って図形が動く）
  3. 線の穴（issue #74）— 細かく折れた線を nonzero で塗ると打ち消し合う
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from movo.renderer.shapes import (
    SHAPE_KINDS,
    arc_to_cubics,
    extract_svg_shapes,
    flatten_segments,
    is_trim_active,
    multiply,
    identity_matrix,
    parse_path_data,
    parse_transform,
    path_to_subpaths,
    render_shape,
    shape_contours,
    subpaths_bounds,
    trim_subpaths,
)


def ops(d: str) -> list[str]:
    """`d` 文字列を «命令の並び» だけにして見る。"""
    return [segment["op"] for segment in parse_path_data(d)["segments"]]


# ══════════════════════════════════════════════════════════════════
# `d` 文字列のパーサ
# ══════════════════════════════════════════════════════════════════


def test_parse_path_data_basic():
    parsed = parse_path_data("M10 10 L90 10 L90 90 Z")
    assert [s["op"] for s in parsed["segments"]] == ["M", "L", "L", "Z"]
    assert parsed["truncated"] is False
    assert parsed["invalid"] is False
    assert parsed["segments"][0]["values"] == [10, 10]
    assert parsed["segments"][2]["values"] == [90, 90]


def test_h_v_s_t_q_collapse_to_l_and_c():
    """`H` `V` は `L` に、`Q` `T` `S` は `C` に潰れます（後段を単純にするため）。"""
    assert ops("M0 0 H10") == ["M", "L"]
    assert ops("M0 0 V10") == ["M", "L"]
    assert ops("M0 0 Q5 5 10 0") == ["M", "C"]
    assert ops("M0 0 Q5 5 10 0 T20 0") == ["M", "C", "C"]
    assert ops("M0 0 C1 1 2 2 3 3 S4 4 5 5") == ["M", "C", "C"]
    assert ops("M0 0 H10 V10 Q20 20 30 30 T40 40 C1 1 2 2 3 3 S4 4 5 5") == [
        "M", "L", "L", "C", "C", "C", "C",
    ]


def test_h_and_v_keep_the_other_axis():
    segments = parse_path_data("M10 20 H50 V60")["segments"]
    assert segments[1]["values"] == [50, 20]
    assert segments[2]["values"] == [50, 60]


def test_quadratic_becomes_the_exact_same_cubic():
    """2 次 → 3 次は «厳密» な変換なので、端点と 2/3 の制御点が一致します。"""
    values = parse_path_data("M0 0 Q30 60 60 0")["segments"][1]["values"]
    assert values == pytest.approx([20.0, 40.0, 40.0, 40.0, 60.0, 0.0])


def test_implicit_lineto_after_moveto():
    """`M` の 2 組目以降は暗黙の `L`（`m` なら `l`）。"""
    parsed = parse_path_data("M10 10 20 20 30 30")
    assert [s["op"] for s in parsed["segments"]] == ["M", "L", "L"]
    assert parsed["segments"][2]["values"] == [30, 30]

    relative = parse_path_data("m10 10 5 5")
    assert [s["op"] for s in relative["segments"]] == ["M", "L"]
    assert relative["segments"][1]["values"] == [15, 15]


def test_restart_after_close_without_moveto():
    """`Z` の後に `M` なしで続く命令は «閉じた点» から再開します。"""
    subpaths = path_to_subpaths("M10 10 L50 10 Z L90 90")
    assert len(subpaths) == 2
    assert subpaths[0]["closed"] is True
    # 2 本目は閉じた点（10, 10）から始まる
    assert subpaths[1]["points"][:2] == [10.0, 10.0]
    assert subpaths[1]["points"][-2:] == [90.0, 90.0]


def test_broken_path_is_reported_not_raised():
    """壊れた `d` でも «読める分だけ» 返します（例外は投げません）。"""
    parsed = parse_path_data("10 10 L20 20")
    assert parsed["invalid"] is True
    assert parsed["segments"] == []

    partial = parse_path_data("M10 10 L20")
    assert partial["invalid"] is True
    assert [s["op"] for s in partial["segments"]] == ["M"]


def test_max_segments_truncates():
    d = "M0 0" + "L1 1" * 50
    parsed = parse_path_data(d, max_segments=10)
    assert parsed["truncated"] is True
    assert len(parsed["segments"]) == 10


def test_read_number_does_not_delegate_to_float():
    """`1.5.5` は «2 つの数値»、`10-20` も «2 つ» です。`float()` には任せられません。"""
    assert parse_path_data("M1.5.5")["segments"][0]["values"] == [1.5, 0.5]
    assert parse_path_data("M10-20")["segments"][0]["values"] == [10.0, -20.0]
    assert parse_path_data("M1e2 1E-2")["segments"][0]["values"] == [100.0, 0.01]


# ══════════════════════════════════════════════════════════════════
# 円弧
# ══════════════════════════════════════════════════════════════════


def test_arc_becomes_cubics_with_exact_endpoint():
    """円弧は 3 次ベジェになり、**終点は指定どおりぴったり**になります。"""
    parsed = parse_path_data("M0 0 A50 50 0 0 1 100 0")
    assert [s["op"] for s in parsed["segments"]] == ["M", "C", "C"]
    last = parsed["segments"][-1]["values"]
    assert last[4] == 100.0
    assert last[5] == 0.0


def test_arc_to_cubics_endpoint_is_exact():
    curves = arc_to_cubics(0, 0, 30, 20, 25, True, False, 40, 10)
    assert len(curves) >= 1
    assert curves[-1][4] == pytest.approx(40.0, abs=1e-12)
    assert curves[-1][5] == pytest.approx(10.0, abs=1e-12)


def test_arc_approximation_stays_on_the_circle():
    """90 度ごとに割るので、誤差は半径の 0.03 % 未満に収まります。"""
    points = path_to_subpaths("M100 0 A100 100 0 1 1 -100 0", tolerance=0.05)[0]["points"]
    radii = [math.hypot(points[i], points[i + 1]) for i in range(0, len(points), 2)]
    assert max(abs(r - 100) for r in radii) < 100 * 0.0003


def test_degenerate_arc_becomes_a_line():
    """半径 0 の円弧は «直線» です（SVG の仕様どおり）。"""
    assert arc_to_cubics(0, 0, 0, 0, 0, False, True, 10, 10) == []
    assert ops("M0 0 A0 0 0 0 1 10 10") == ["M", "L"]


def test_arc_flags_may_be_packed_together():
    """**`a1 1 0 011 1`** — フラグは «1 文字» ずつ読みます。

    数値として読むと `011` を 11 と読んでしまい、円弧が壊れます。
    """
    parsed = parse_path_data("M0 0 a1 1 0 011 1")
    assert parsed["invalid"] is False
    assert [s["op"] for s in parsed["segments"]] == ["M", "C"]
    last = parsed["segments"][-1]["values"]
    assert (last[4], last[5]) == (1.0, 1.0)

    # 詰めない書き方（`a1 1 0 0 1 1 1`）と同じ結果になること
    spaced = parse_path_data("M0 0 a1 1 0 0 1 1 1")
    assert parsed["segments"][-1]["values"] == pytest.approx(spaced["segments"][-1]["values"])


def test_arc_flags_zero_zero_packed():
    """`a1 1 0 001 1` は largeArc=0 / sweep=0。"""
    packed = parse_path_data("M0 0 a1 1 0 001 1")["segments"]
    spaced = parse_path_data("M0 0 a1 1 0 0 0 1 1")["segments"]
    assert len(packed) == len(spaced)
    assert packed[-1]["values"] == pytest.approx(spaced[-1]["values"])


# ══════════════════════════════════════════════════════════════════
# 変換行列
# ══════════════════════════════════════════════════════════════════


def test_parse_transform_translate_then_rotate():
    """`translate(10 20) rotate(90)` — 右のものが «先に» 点に効きます。"""
    matrix = parse_transform("translate(10 20) rotate(90)")
    assert matrix == pytest.approx([0, 1, -1, 0, 10, 20], abs=1e-12)

    # (1, 0) は回転で (0, 1) になり、そこから (10, 20) ずれる
    x, y = matrix[0] * 1 + matrix[2] * 0 + matrix[4], matrix[1] * 1 + matrix[3] * 0 + matrix[5]
    assert (x, y) == pytest.approx((10.0, 21.0), abs=1e-12)


def test_parse_transform_pieces():
    assert parse_transform("") == identity_matrix()
    assert parse_transform("scale(2)") == pytest.approx([2, 0, 0, 2, 0, 0])
    assert parse_transform("scale(2 3)") == pytest.approx([2, 0, 0, 3, 0, 0])
    assert parse_transform("matrix(1 2 3 4 5 6)") == pytest.approx([1, 2, 3, 4, 5, 6])
    # 中心つきの回転は «移動 → 回転 → 戻す»。中心は動きません
    m = parse_transform("rotate(90 10 10)")
    assert (m[0] * 10 + m[2] * 10 + m[4], m[1] * 10 + m[3] * 10 + m[5]) == pytest.approx((10.0, 10.0))


def test_multiply_is_the_svg_order():
    a = [1, 0, 0, 1, 5, 0]
    b = [2, 0, 0, 2, 0, 0]
    assert multiply(a, b) == pytest.approx([2, 0, 0, 2, 5, 0])


def test_transform_is_applied_before_flattening():
    """行列は «折れ線にする前» に掛かるので、拡大しても曲線が角張りません。"""
    plain = path_to_subpaths("M0 0 C0 50 100 50 100 0")[0]["points"]
    scaled = path_to_subpaths("M0 0 C0 50 100 50 100 0", transform=[10, 0, 0, 10, 0, 0])[0]["points"]
    # 分割数は «変換後の長さ» から決まるので、点の数は増えます
    assert len(scaled) > len(plain)
    assert scaled[-2:] == pytest.approx([1000.0, 0.0])


# ══════════════════════════════════════════════════════════════════
# トリムパス
# ══════════════════════════════════════════════════════════════════


def polyline_length(points, closed=False) -> float:
    arr = np.asarray(points, dtype=float).ravel()
    xs, ys = arr[0::2], arr[1::2]
    total = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
    if closed:
        total += float(math.hypot(xs[0] - xs[-1], ys[0] - ys[-1]))
    return total


def test_is_trim_active():
    assert is_trim_active(None) is False
    assert is_trim_active({}) is False
    assert is_trim_active({"start": 0, "end": 1}) is False
    assert is_trim_active({"start": 0, "end": 1, "enabled": False}) is False
    assert is_trim_active({"end": 0.5}) is True
    assert is_trim_active({"start": 0.5}) is True
    assert is_trim_active({"offset": 0.25}) is True


def test_trim_half_of_a_straight_line():
    subpaths = [{"points": [0, 0, 100, 0], "closed": False}]
    trimmed = trim_subpaths(subpaths, {"start": 0, "end": 0.5})
    assert len(trimmed) == 1
    assert polyline_length(trimmed[0]["points"]) == pytest.approx(50.0)
    assert trimmed[0]["closed"] is False


def test_trim_half_of_a_curve_keeps_half_the_length():
    subpaths = path_to_subpaths("M0 0 C0 100 200 100 200 0")
    full = polyline_length(subpaths[0]["points"])
    trimmed = trim_subpaths(subpaths, {"start": 0, "end": 0.5})
    assert polyline_length(trimmed[0]["points"]) == pytest.approx(full / 2, rel=1e-6)


def test_trim_empty_range_yields_nothing():
    subpaths = [{"points": [0, 0, 100, 0], "closed": False}]
    assert trim_subpaths(subpaths, {"start": 0.5, "end": 0.5}) == []


def test_trim_offset_wraps_around_a_closed_path():
    """閉じたパスでは `offset` が «一周して» 回り込み、2 本に割れます。"""
    square = [{"points": [0, 0, 100, 0, 100, 100, 0, 100], "closed": True}]
    trimmed = trim_subpaths(square, {"start": 0, "end": 0.5, "offset": 0.75})
    assert len(trimmed) == 2
    total = sum(polyline_length(piece["points"]) for piece in trimmed)
    assert total == pytest.approx(400 * 0.5, rel=1e-6)
    # 切った断片は «開いた線» です
    assert all(piece["closed"] is False for piece in trimmed)


def test_trim_offset_without_wrap_on_an_open_path():
    """開いたパスでは、はみ出した分は «落とします»（一周する先がないので）。"""
    line = [{"points": [0, 0, 100, 0], "closed": False}]
    trimmed = trim_subpaths(line, {"start": 0, "end": 0.5, "offset": 0.75})
    assert len(trimmed) == 1
    # [0.75, 1.25] のうち [0.75, 1] だけが残る
    assert polyline_length(trimmed[0]["points"]) == pytest.approx(25.0)


def test_trim_sequential_draws_one_subpath_at_a_time():
    """`mode: "sequential"` は全部を 1 本の長さとみなして «順に» 切ります。"""
    subpaths = [
        {"points": [0, 0, 100, 0], "closed": False},
        {"points": [0, 50, 100, 50], "closed": False},
    ]
    half = trim_subpaths(subpaths, {"start": 0, "end": 0.5, "mode": "sequential"})
    # 前半 50 % ＝ 1 本目だけが丸ごと出る
    assert len(half) == 1
    assert polyline_length(half[0]["points"]) == pytest.approx(100.0)

    quarter = trim_subpaths(subpaths, {"start": 0, "end": 0.75, "mode": "sequential"})
    assert len(quarter) == 2
    assert polyline_length(quarter[1]["points"]) == pytest.approx(50.0)


def test_trim_is_applied_last_so_the_box_does_not_grow():
    """**トリムは «最後に» 掛けます。**

    先に切ってしまうと、トリムを 0 → 1 に動かしたときに «囲む矩形が育って»
    図形が動いて見えます。大きさは «切る前の形» から決めるのが正解です。
    """
    sizes = []
    for end in (0.1, 0.5, 0.9, 1.0):
        geometry = shape_contours({"type": "circle", "radius": 50, "trim": {"start": 0, "end": end}})
        sizes.append((geometry["width"], geometry["height"]))
    assert sizes == [(100.0, 100.0)] * 4

    # 描いたビットマップの大きさも変わらないこと
    widths = {
        render_shape({"type": "circle", "radius": 50, "strokeWidth": 4, "trim": {"start": 0, "end": end}})["width"]
        for end in (0.1, 0.5, 0.9, 1.0)
    }
    assert len(widths) == 1


def test_trimmed_geometry_is_open():
    geometry = shape_contours({"type": "circle", "radius": 50, "trim": {"start": 0, "end": 0.5}})
    assert geometry["closed"] is False
    assert geometry["trimmed"] is True


# ══════════════════════════════════════════════════════════════════
# `.svg` の取り込み
# ══════════════════════════════════════════════════════════════════


SAMPLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <!-- <path d="M0 0 L9 9"/> このコメントの中は拾いません -->
  <script>var evil = 1;</script>
  <style>.a { fill: red }</style>
  <defs><path d="M0 0 L5 5"/></defs>
  <image xlink:href="http://example.com/x.png" onload="boom()"/>
  <use href="#nope"/>
  <text x="0" y="0">見出し</text>
  <g transform="translate(10 10)"><rect x="0" y="0" width="10" height="10"/></g>
</svg>"""


def test_extract_svg_shapes_reads_only_geometry():
    result = extract_svg_shapes(SAMPLE_SVG)
    assert result["view_box"] == [0, 0, 100, 100]
    assert result["width"] == 100
    assert result["height"] == 100
    # 拾うのは `<g>` の中の `<rect>` 1 つだけ
    assert len(result["subpaths"]) == 1
    assert result["stats"]["shapes"] == 1
    assert result["stats"]["paths"] == 0
    # script / style / defs / image / use / text の 6 つを «中身ごと» 飛ばしています
    assert result["stats"]["skipped"] == 6


def test_extract_svg_shapes_applies_the_parent_transform():
    points = extract_svg_shapes(SAMPLE_SVG)["subpaths"][0]["points"]
    bounds = subpaths_bounds([{"points": points}])
    assert (bounds["min_x"], bounds["min_y"]) == pytest.approx((10.0, 10.0))
    assert (bounds["max_x"], bounds["max_y"]) == pytest.approx((20.0, 20.0))


def test_extract_svg_shapes_never_follows_references():
    """`href` / `xlink:href` / `on*=` は «読みもしません»。"""
    svg = '<svg viewBox="0 0 10 10"><a href="http://x/"><circle cx="5" cy="5" r="1" onclick="x()"/></a></svg>'
    result = extract_svg_shapes(svg)
    assert len(result["subpaths"]) == 1  # circle だけ。`<a>` の href は無視


def test_extract_svg_shapes_size_limit_is_in_bytes():
    """上限は **バイト数** で見ます（日本語の入った SVG でもファイルサイズと同じ尺度）。"""
    # 3 バイトの文字を 10 個 ＝ 30 バイト。文字数（10）では超えません
    result = extract_svg_shapes("あ" * 10 + '<svg><rect width="1" height="1"/></svg>', max_bytes=20)
    assert result["subpaths"] == []
    assert result["stats"]["invalid"] is True


def test_extract_svg_shapes_does_not_raise_on_garbage():
    """読めなくても «警告して空»。例外を投げると動画全体が出なくなります。"""
    for source in (None, 123, "", "<svg", "<svg><path d='???'/></svg>"):
        result = extract_svg_shapes(source)
        assert isinstance(result["subpaths"], list)


def test_extract_svg_shapes_max_elements():
    body = "".join(f'<rect x="{i}" y="0" width="1" height="1"/>' for i in range(20))
    result = extract_svg_shapes(f'<svg viewBox="0 0 40 2">{body}</svg>', max_elements=5)
    assert result["stats"]["truncated"] is True
    assert len(result["subpaths"]) == 5


def test_extract_svg_shapes_basic_elements():
    svg = (
        '<svg viewBox="0 0 100 100">'
        '<line x1="0" y1="0" x2="10" y2="10"/>'
        '<polyline points="0,0 5,5 10,0"/>'
        '<polygon points="0,0 5,5 10,0"/>'
        '<ellipse cx="50" cy="50" rx="20" ry="10"/>'
        '<rect x="0" y="0" width="20" height="20" rx="4"/>'
        "</svg>"
    )
    result = extract_svg_shapes(svg)
    assert len(result["subpaths"]) == 5
    closed = [subpath["closed"] for subpath in result["subpaths"]]
    assert closed == [False, False, True, True, True]


# ══════════════════════════════════════════════════════════════════
# 図形の輪郭
# ══════════════════════════════════════════════════════════════════


def test_shape_kinds_are_all_understood():
    for kind in SHAPE_KINDS:
        geometry = shape_contours({"type": kind})
        assert geometry["width"] > 0
        assert geometry["height"] > 0


def test_shape_contours_from_a_d_string():
    geometry = shape_contours({"type": "path", "d": "M10 10 L90 10 L90 90 Z"})
    assert geometry["closed"] is True
    assert geometry["width"] == pytest.approx(80.0)
    assert geometry["height"] == pytest.approx(80.0)
    # 囲む矩形の左上が原点に来ていること
    points = np.asarray(geometry["contours"][0], dtype=float)
    assert points.min() == pytest.approx(0.0)


def test_shape_contours_from_an_asset_store():
    class Assets:
        def get_svg(self, name):
            assert name == "logo"
            return {"subpaths": path_to_subpaths("M0 0 L40 0 L40 20 Z")}

    geometry = shape_contours({"type": "svg", "svgAsset": "logo"}, {"assets": Assets()})
    assert geometry["width"] == pytest.approx(40.0)
    assert geometry["height"] == pytest.approx(20.0)


def test_missing_asset_is_empty_not_an_error():
    geometry = shape_contours({"type": "svg", "svgAsset": "nope"})
    assert geometry["contours"] == []
    assert geometry["width"] == 1


def test_shape_contours_from_the_array_form():
    geometry = shape_contours(
        {"type": "path", "path": [{"m": [0, 0]}, {"l": [50, 0]}, {"l": [50, 30]}, {"z": True}]}
    )
    assert geometry["width"] == pytest.approx(50.0)
    assert geometry["height"] == pytest.approx(30.0)


# ══════════════════════════════════════════════════════════════════
# 描画
# ══════════════════════════════════════════════════════════════════


def alpha_of(result: dict, x: int, y: int) -> int:
    return int(result["bitmap"].data[y, x, 3])


def test_render_circle_is_opaque_in_the_middle_and_clear_at_the_corners():
    result = render_shape({"type": "circle", "radius": 50, "fill": "#ffffff"})
    width, height = result["width"], result["height"]
    assert alpha_of(result, width // 2, height // 2) == 255
    middle = result["bitmap"].data[height // 2, width // 2]
    assert tuple(middle[:3]) == (255, 255, 255)
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        assert alpha_of(result, x, y) == 0


def test_render_rectangle_has_the_requested_size():
    result = render_shape({"type": "rectangle", "width": 120, "height": 40, "fill": "#ff0000"})
    assert result["box_width"] == pytest.approx(120.0)
    assert result["box_height"] == pytest.approx(40.0)
    # ビットマップは線の分だけ外へ広げてあります（`origin_*` がそのずれ）
    pad = result["origin_x"]
    assert result["width"] == 120 + int(pad) * 2
    assert result["height"] == 40 + int(pad) * 2
    assert alpha_of(result, result["width"] // 2, result["height"] // 2) == 255


def test_render_scale_multiplies_the_bitmap_but_not_the_box():
    result = render_shape({"type": "rectangle", "width": 100, "height": 50, "fill": "#fff"}, 2.0)
    assert result["box_width"] == pytest.approx(100.0)
    assert result["width"] == pytest.approx(200 + math.ceil(2.0) * 2)


def test_render_shape_never_raises_on_a_broken_svg():
    result = render_shape({"type": "path", "svg": "<svg><path d='!!!'/></svg>", "fill": "#fff"})
    assert result["bitmap"].data.shape[2] == 4


# ── issue #74 の回帰試験 ─────────────────────────────────────────


def stroke_gaps(shape: dict, scale: float = 1.0) -> list[tuple[int, int]]:
    """線の «上» を歩いて、透明になっている画素を集めます。

    issue #74 は «細かく折れた線を nonzero で塗ると、辺の四角形と継ぎ目の円が
    逆回りになって打ち消し合い、点線のように穴が開く» という不具合でした。
    輪郭の点列（＝線の芯）をたどって、そこが 1 つでも透明なら穴です。
    """
    geometry = shape_contours(shape)
    result = render_shape(shape, scale)
    data = result["bitmap"].data
    pad = result["origin_x"] * scale
    gaps: list[tuple[int, int]] = []
    for contour in geometry["contours"]:
        points = np.asarray(contour, dtype=float).ravel() * scale + pad
        count = points.size // 2
        if count < 2:
            continue
        indices = range(count) if geometry["closed"] else range(count - 1)
        for i in indices:
            j = (i + 1) % count
            x0, y0 = points[2 * i], points[2 * i + 1]
            x1, y1 = points[2 * j], points[2 * j + 1]
            steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.25))
            for k in range(steps + 1):
                t = k / steps
                px = int(round(x0 + (x1 - x0) * t))
                py = int(round(y0 + (y1 - y0) * t))
                if 0 <= py < data.shape[0] and 0 <= px < data.shape[1]:
                    if data[py, px, 3] == 0:
                        gaps.append((px, py))
    return gaps


def test_issue_74_stroked_circle_has_no_holes():
    assert stroke_gaps({"type": "circle", "radius": 40, "strokeWidth": 3}) == []


def test_issue_74_stroked_arc_path_has_no_holes():
    assert stroke_gaps({"type": "path", "d": "M0 0 A50 50 0 1 1 100 0", "strokeWidth": 3}) == []


def test_issue_74_trimmed_stroke_has_no_holes():
    assert stroke_gaps({"type": "circle", "radius": 40, "strokeWidth": 3, "trim": {"start": 0.1, "end": 0.6}}) == []


def test_issue_74_the_stroke_is_continuous_along_a_scanline():
    """円の «真ん中の行» を見ると、線は左右に 1 本ずつだけ現れます。

    穴が開いていると、ここが «途切れた塊» に分かれて数が増えます。
    """
    result = render_shape({"type": "circle", "radius": 40, "strokeWidth": 3})
    row = result["bitmap"].data[result["height"] // 2, :, 3]
    runs = 0
    previous = 0
    for value in row:
        if value > 0 and previous == 0:
            runs += 1
        previous = value
    assert runs == 2


def test_trimmed_shape_without_stroke_still_draws_a_line():
    """**トリムしたのに線幅が 0** のときの救済（既定 2 px、`fill` の色を流用）。"""
    result = render_shape({"type": "circle", "radius": 40, "fill": "#00ff00", "trim": {"start": 0, "end": 0.5}})
    data = result["bitmap"].data
    painted = data[..., 3] > 0
    assert painted.any()
    # 塗りではなく «線» なので、色は fill の緑が流用されます
    ys, xs = np.nonzero(data[..., 3] == 255)
    assert tuple(data[ys[0], xs[0], :3]) == (0, 255, 0)
    # 塗りつぶしていないこと（円の中心は空いている）
    assert data[result["height"] // 2, result["width"] // 2, 3] == 0


# ── グラデーション ───────────────────────────────────────────────


def test_linear_gradient_changes_from_left_to_right():
    result = render_shape(
        {
            "type": "rectangle",
            "width": 100,
            "height": 50,
            "fill": {
                "type": "linear",
                "angle": 0,
                "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}],
            },
        }
    )
    data = result["bitmap"].data
    y = result["height"] // 2
    left = int(data[y, 2, 0])
    right = int(data[y, result["width"] - 3, 0])
    assert data[y, 2, 3] == 255
    assert data[y, result["width"] - 3, 3] == 255
    assert left < 40
    assert right > 215
    # 途中が単調に増えていること（1 画素ずつ Python を呼ばずに作れている証拠）
    middle = int(data[y, result["width"] // 2, 0])
    assert left < middle < right


def test_radial_gradient_changes_from_the_centre_outwards():
    result = render_shape(
        {
            "type": "rectangle",
            "width": 100,
            "height": 100,
            "fill": {
                "type": "radial",
                "radius": 0.5,
                "stops": [{"offset": 0, "color": "#ffffff"}, {"offset": 1, "color": "#000000"}],
            },
        }
    )
    data = result["bitmap"].data
    centre = int(data[result["height"] // 2, result["width"] // 2, 0])
    edge = int(data[result["height"] // 2, 2, 0])
    assert centre > 240
    assert edge < 30


def test_gradient_shader_is_vectorised():
    """シェーダは `(h, w)` の配列を受けて `(h, w, 4)` を返す約束です。"""
    from movo.renderer.shapes import gradient_shader

    shader = gradient_shader(
        {"type": "linear", "angle": 0, "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}]},
        100,
        10,
    )
    ys, xs = np.mgrid[0:10, 0:100]
    colors = shader(xs.astype(float), ys.astype(float))
    assert colors.shape == (10, 100, 4)
    assert colors[0, 0, 0] == pytest.approx(0.0)
    assert colors[0, -1, 0] > 240
    assert colors[..., 3].min() == pytest.approx(1.0)


def test_gradient_without_stops_is_ignored():
    result = render_shape({"type": "rectangle", "width": 20, "height": 20, "fill": {"type": "linear", "stops": []}})
    assert result["bitmap"].data[..., 3].max() == 0


# ── 変換の掛かった `flatten_segments` ────────────────────────────


def test_flatten_segments_without_a_moveto_is_empty():
    assert flatten_segments([{"op": "L", "values": [1, 1]}]) == []


def test_flatten_segments_drops_one_point_subpaths():
    assert flatten_segments([{"op": "M", "values": [1, 1]}]) == []


def test_subpaths_bounds_floor_is_tiny():
    """`subpaths_bounds` の下限は `1e-6`（`_bounds_of` の `1` とは違います）。"""
    bounds = subpaths_bounds([{"points": [0, 0, 0, 0]}])
    assert bounds["width"] == pytest.approx(1e-6)
    assert bounds["height"] == pytest.approx(1e-6)
    assert subpaths_bounds([])["width"] == 1.0
