"""画素バッファのテスト。

**速度の作法が守られているかは測れません**（テストで «速いこと» を確かめると
機械の速さに左右されて不安定になります）。代わりにここでは
**«NumPy で書き直したせいで結果が変わっていないか»** を見ます。
とくに ``resize`` は累積和を使う書き方に変えてあるので、素直な二重ループと
同じ結果になることを直接くらべます。
"""

from __future__ import annotations

import numpy as np
import pytest

import movo.core as core
from movo.core.bitmap import blend_over, to_float, to_u8


def test_new_bitmap_is_transparent_black():
    bitmap = core.Bitmap(4, 3)
    assert bitmap.data.shape == (3, 4, 4)
    assert not bitmap.data.any()
    assert bitmap.is_empty


def test_create_parses_colour_strings():
    assert [int(v) for v in core.Bitmap.create(2, 2, "#ff0000").data[0, 0]] == [255, 0, 0, 255]
    assert [int(v) for v in core.Bitmap.create(2, 2, "rgba(0,0,255,0.5)").data[0, 0]] == [0, 0, 255, 128]
    assert core.Bitmap.create(2, 2).is_empty


def test_get_and_set_pixel_ignore_out_of_range():
    bitmap = core.Bitmap(3, 3)
    bitmap.set_pixel(1, 2, 10, 20, 30, 255)
    assert bitmap.get_pixel(1, 2) == {"r": 10, "g": 20, "b": 30, "a": 1.0}
    bitmap.set_pixel(-1, 0, 1, 2, 3, 4)  # 黙って捨てる
    bitmap.set_pixel(99, 0, 1, 2, 3, 4)
    assert bitmap.get_pixel(-1, 0) == {"r": 0, "g": 0, "b": 0, "a": 0}


def test_blend_over_does_not_wrap_around_at_255():
    """**``uint8`` のまま計算していないこと。**

    ``200 * 2`` が 144 に巻き戻る事故を防ぐための番人です。不透明な白を
    不透明な赤の上に重ねたら、素直に白にならなければいけません。
    """
    dst = np.full((2, 2, 4), 255, np.uint8)
    dst[..., 1:3] = 0  # 不透明な赤
    src = np.full((2, 2, 4), 255, np.uint8)  # 不透明な白
    blend_over(dst, src)
    assert [int(v) for v in dst[0, 0]] == [255, 255, 255, 255]


def test_blend_over_matches_the_alpha_compositing_formula():
    dst = np.zeros((1, 1, 4), np.uint8)
    dst[0, 0] = (0, 0, 0, 255)
    src = np.zeros((1, 1, 4), np.uint8)
    src[0, 0] = (255, 255, 255, 128)
    blend_over(dst, src)
    # 128/255 の白を黒に重ねる → だいたい半分の灰色
    assert 126 <= int(dst[0, 0, 0]) <= 130
    assert int(dst[0, 0, 3]) == 255


def test_draw_clips_at_the_edges():
    """はみ出しを呼ぶ側で確かめずに済むこと。"""
    canvas = core.Bitmap(4, 4)
    stamp = core.Bitmap.create(4, 4, "#ffffff")
    canvas.draw(stamp, -2, -2)
    assert int(canvas.data[0, 0, 3]) == 255
    assert int(canvas.data[3, 3, 3]) == 0
    canvas.draw(stamp, 100, 100)  # 完全に外 → 何も起きない
    assert int(canvas.data[3, 3, 3]) == 0


def test_alpha_bounds_finds_the_opaque_area():
    bitmap = core.Bitmap(5, 5)
    bitmap.set_pixel(2, 3, 255, 255, 255, 255)
    assert bitmap.alpha_bounds() == {"x": 2, "y": 3, "width": 1, "height": 1}
    assert core.Bitmap(3, 3).alpha_bounds() is None


def test_crop_clips_to_the_source():
    bitmap = core.Bitmap.create(8, 8, "#00ff00")
    assert (bitmap.crop(2, 2, 4, 4).width, bitmap.crop(2, 2, 4, 4).height) == (4, 4)
    edge = bitmap.crop(6, 6, 10, 10)
    assert (edge.width, edge.height) == (2, 2)
    assert core.Bitmap(4, 4).crop(10, 10, 2, 2).width == 0


def _resize_reference(bitmap: core.Bitmap, w: int, h: int) -> np.ndarray:
    """JS 版そのままの二重ループ。**遅いのでテスト専用**です。"""
    out = np.zeros((h, w, 4), np.uint8)
    sx = bitmap.width / w
    sy = bitmap.height / h
    for y in range(h):
        y0 = int(np.floor(y * sy))
        y1 = max(y0 + 1, min(bitmap.height, int(np.ceil((y + 1) * sy))))
        for x in range(w):
            x0 = int(np.floor(x * sx))
            x1 = max(x0 + 1, min(bitmap.width, int(np.ceil((x + 1) * sx))))
            block = bitmap.data[y0:y1, x0:x1].astype(np.float64)
            alpha = block[..., 3]
            total_a = alpha.sum()
            count = block.shape[0] * block.shape[1]
            if total_a > 0:
                out[y, x, :3] = np.clip(np.rint((block[..., :3] * alpha[..., None]).sum((0, 1)) / total_a), 0, 255)
                out[y, x, 3] = np.clip(np.rint(total_a / count), 0, 255)
    return out


@pytest.mark.parametrize("size", [(4, 4), (5, 3), (16, 16), (7, 11)])
def test_resize_matches_the_straightforward_loop(size):
    """**累積和を使う速い書き方が、素直なループと同じ答えを出すこと。**

    ここを外すと «縮小したときだけ色がずれる» という、目で見て気付きにくい
    壊れ方をします。透明な画素を混ぜて、アルファの重み付けも試します。
    """
    source = core.Bitmap(12, 9)
    xs = np.arange(12)
    ys = np.arange(9)
    source.data[..., 0] = (xs[None, :] * 21) % 256
    source.data[..., 1] = (ys[:, None] * 28) % 256
    source.data[..., 2] = (xs[None, :] * ys[:, None] * 3) % 256
    source.data[..., 3] = ((xs[None, :] + ys[:, None]) * 17) % 256
    assert np.array_equal(source.resize(*size).data, _resize_reference(source, *size))


def test_resize_to_the_same_size_copies():
    bitmap = core.Bitmap.create(4, 4, "#123456")
    same = bitmap.resize(4, 4)
    assert same is not bitmap
    assert np.array_equal(same.data, bitmap.data)


def test_flatten_drops_alpha_onto_a_background():
    bitmap = core.Bitmap(2, 2)
    bitmap.data[...] = (255, 255, 255, 128)
    flat = bitmap.flatten("#000000")
    assert (flat.data[..., 3] == 255).all()
    assert 126 <= int(flat.data[0, 0, 0]) <= 129


def test_float_helpers_round_trip():
    values = np.array([[[0, 1, 127, 128], [254, 255, 3, 200]]], np.uint8)
    assert np.array_equal(to_u8(to_float(values)), values)


def test_wrapping_an_array_checks_the_shape():
    with pytest.raises(ValueError):
        core.Bitmap(4, 4, np.zeros((3, 3, 4), np.uint8))
