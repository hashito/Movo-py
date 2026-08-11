"""カラーグレーディングの «意味» を守るテスト。

画素の一致は `test_effects_parity.py` が見ています。**こちらは «値の読み方» を
固定する**ためのものです。ここが崩れると、絵は出るのに «指定した通りに
ならない» という一番たちの悪い壊れ方をします。
"""

from __future__ import annotations

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.core.lut import MAX_LUT_3D_SIZE, identity_lut
from movo.renderer.effects import color_adjust, effects, gradient_map, gradient_overlay
from movo.renderer.effects_grade import build_curve_table, color_wheels, curves, hsl_secondary, lut


def flat(r=120, g=90, b=200, a=255, width=8, height=6) -> Bitmap:
    bitmap = Bitmap(width, height)
    bitmap.data[...] = (r, g, b, a)
    return bitmap


# ── 「0 が変化なし」の約束 ────────────────────────────────────────


def test_color_adjust_zero_is_no_change():
    """**`colorAdjust` の値は «増減量» で 0 が «変化なし» です。**

    倍率ではありません。ここを取り違えて 1.0 を «等倍» のつもりで入れると
    真っ白になります（下のテストがその意味を固定します）。
    """
    src = flat()
    same = color_adjust(src, {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0, "gamma": 1})
    assert np.array_equal(same.data, src.data)


def test_color_adjust_brightness_one_blows_out():
    """`brightness: 1.0` は «+255» の意味なので、白飛びするのが正しい挙動です。"""
    out = color_adjust(flat(), {"brightness": 1.0})
    assert np.all(out.data[..., :3] == 255)


def test_color_adjust_brightness_is_additive():
    """0.1 なら 0.1 × 255 ぶん明るくなる（掛け算ではない）。"""
    out = color_adjust(flat(100, 100, 100), {"brightness": 0.1})
    assert out.data[0, 0, 0] == pytest.approx(100 + 25.5, abs=1)


def test_curves_diagonal_is_no_change():
    """対角線のカーブは «変化なし»。"""
    src = flat()
    assert np.array_equal(curves(src, {"rgb": [[0, 0], [1, 1]]}).data, src.data)


def test_color_wheels_neutral_is_no_change():
    """`lift 0 / gamma 1 / gain 1` が «変化なし»。**gamma と gain だけ 1 起点です。**"""
    src = flat()
    out = color_wheels(src, {"lift": 0, "gamma": 1, "gain": 1})
    assert np.array_equal(out.data, src.data)


def test_hsl_secondary_without_shift_is_no_change():
    src = flat()
    out = hsl_secondary(src, {"select": {"hue": [0, 360]}, "shift": {}})
    assert np.array_equal(out.data, src.data)


def test_hsl_secondary_only_touches_the_selected_hue():
    """**選んだ色相の外は 1 も動かないこと。**"""
    bitmap = Bitmap(2, 1)
    bitmap.data[0, 0] = (30, 120, 220, 255)  # 青寄り（色相 およそ 212 度）
    bitmap.data[0, 1] = (220, 60, 40, 255)  # 赤寄り（色相 およそ 7 度）
    out = hsl_secondary(bitmap, {"select": {"hue": [180, 260]}, "shift": {"sat": -1}})
    assert not np.array_equal(out.data[0, 0], bitmap.data[0, 0])
    assert np.array_equal(out.data[0, 1], bitmap.data[0, 1])


# ── グラデーションの書き方 ────────────────────────────────────────


def test_gradient_overlay_uses_stops_not_colors():
    """**`gradientOverlay` は `colors` ではなく `stops` です。**

    `colors` を渡しても既定のグラデーションのままになる、という «黙って
    効かない» 挙動をここで固定します（JS 版と同じ）。
    """
    src = flat()
    with_colors = gradient_overlay(src, {"colors": ["#ff0000", "#00ff00"], "opacity": 1})
    default = gradient_overlay(src, {"opacity": 1})
    assert np.array_equal(with_colors.data, default.data)

    with_stops = gradient_overlay(
        src, {"stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#00ff00"}], "opacity": 1}
    )
    assert not np.array_equal(with_stops.data, default.data)


def test_gradient_map_uses_stops_too():
    src = flat()
    mapped = gradient_map(src, {"stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ff0000"}]})
    assert mapped.data[0, 0, 1] == 0
    assert mapped.data[0, 0, 2] == 0


# ── トーンカーブの作り ────────────────────────────────────────────


def test_curve_table_is_monotonic():
    """**制御点の間で «行き過ぎ» ないこと。**

    ふつうの 3 次スプラインだと暗部を持ち上げただけで途中がへこみ、絵に
    «縞» が出ます。Fritsch–Carlson は接線を縮めて単調性を守ります。
    """
    table = build_curve_table([[0, 0], [0.25, 0.45], [0.75, 0.55], [1, 1]])
    assert table is not None
    assert np.all(np.diff(table) >= -1e-9)


def test_curve_table_needs_two_points():
    assert build_curve_table([[0.5, 0.5]]) is None
    assert build_curve_table(None) is None


# ── `.cube` の守り ────────────────────────────────────────────────


def test_identity_lut_changes_nothing():
    src = flat()
    assert np.array_equal(lut(src, {"lut": identity_lut(9), "amount": 1}).data, src.data)


def test_lut_amount_zero_changes_nothing():
    src = flat()
    table = identity_lut(2)
    table.data[..., :] = 0.0  # 真っ黒に潰す LUT
    assert np.array_equal(lut(src, {"lut": table, "amount": 0}).data, src.data)


def test_missing_lut_asset_is_a_no_op():
    """素材が読めないときは «何もしない»。色が付かないだけで、絵は出ます。"""
    src = flat()
    assert lut(src, {"asset": "there-is-no-such-look"}, {}) is src


def test_cube_size_is_checked_before_allocating():
    """**`LUT_3D_SIZE` は配列を確保する «前» に弾くこと。**

    行数の上限だけでは守れません。`LUT_3D_SIZE 2000` の 1 行で 2000³ × 3 個の
    配列を先に確保してしまい、そこでメモリを食い潰されます。数値の行を
    1 行も書いていないのに落ちる、という形でそれを確かめます。
    """
    from movo.core.errors import MovoError
    from movo.core.lut import parse_cube_lut

    with pytest.raises(MovoError):
        parse_cube_lut("LUT_3D_SIZE 2000\n")
    with pytest.raises(MovoError):
        parse_cube_lut(f"LUT_3D_SIZE {MAX_LUT_3D_SIZE + 1}\n")


def test_cube_round_trip():
    """書き出した `.cube` を読み直すと同じ格子になること（並び順の確認）。"""
    from movo.core.lut import parse_cube_lut

    size = 3
    lines = ["TITLE \"test\"", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r / 2:.6f} {g / 2:.6f} {b / 2:.6f}")
    parsed = parse_cube_lut("\n".join(lines))
    assert parsed.size == size
    # 添字は [b, g, r]。ここを取り違えると色が «斜めに» 転びます。
    assert parsed.data[0, 0, size - 1].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert parsed.data[size - 1, 0, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_grade_effects_are_in_the_registry():
    for name in ("curves", "colorWheels", "hslSecondary", "lut"):
        assert name in effects
