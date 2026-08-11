"""ラスタライザ・図形・合成モードのテスト。

**JS 版と同じ絵が出ること**が第一の目的です。ここには JS 版で書き出した画素と
突き合わせて確かめた «性質» を、外部ファイル無しで再現できる形で置いています
（画素そのものを固定値で持つと、フォントや環境が変わるたびに壊れるため）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.renderer import kernels, raster as R


# ══════════════════════════════════════════════════════════════════
# 走査線ラスタライザ
# ══════════════════════════════════════════════════════════════════


def test_矩形は指定どおりの範囲を塗る():
    region = R.rasterize_contours([[10, 10, 30, 10, 30, 25, 10, 25]], 40, 40)
    cov = region.coverage
    assert cov[15, 20] == pytest.approx(1.0)
    assert cov[5, 20] == 0.0
    assert cov[15, 35] == 0.0
    # 囲む矩形は «整数に丸めて» 返る
    assert (region.min_x, region.min_y, region.max_x, region.max_y) == (10, 10, 30, 25)


def test_辺が画素の途中にあると被覆率は端数になる():
    # 左辺が x=10.25、右辺が x=20.75。x=10 の画素は 0.75 だけ覆われる。
    region = R.rasterize_contours([[10.25, 5, 20.75, 5, 20.75, 15, 10.25, 15]], 30, 20)
    assert region.coverage[10, 10] == pytest.approx(0.75, abs=1e-5)
    assert region.coverage[10, 20] == pytest.approx(0.75, abs=1e-5)
    assert region.coverage[10, 15] == pytest.approx(1.0)


def test_縦は4倍サンプリングで刻む():
    # 上辺が y=5.5 の矩形。y=5 の行は下半分だけ覆われるので 0.5。
    region = R.rasterize_contours([[2, 5.5, 18, 5.5, 18, 15, 2, 15]], 20, 20)
    assert region.coverage[5, 10] == pytest.approx(0.5, abs=1e-5)
    # 4 分割なので、8 分の 1 だけずらすと 0.25 刻みになる
    region = R.rasterize_contours([[2, 5.25, 18, 5.25, 18, 15, 2, 15]], 20, 20)
    assert region.coverage[5, 10] == pytest.approx(0.75, abs=1e-5)


def test_nonzero_と_evenodd_で自己交差の中身が変わる():
    penta = []
    for i in range(5):
        angle = -math.pi / 2 + ((i * 2) / 5) * math.pi * 2
        penta += [50 + math.cos(angle) * 40, 50 + math.sin(angle) * 40]
    nonzero = R.rasterize_contours([penta], 100, 100, "nonzero")
    evenodd = R.rasterize_contours([penta], 100, 100, "evenodd")
    # 五芒星の «真ん中の五角形» は nonzero では塗られ、evenodd では抜ける
    assert nonzero.coverage[50, 50] == pytest.approx(1.0)
    assert evenodd.coverage[50, 50] == pytest.approx(0.0)
    # 星の «腕» はどちらでも塗られる
    assert nonzero.coverage[20, 50] > 0.9
    assert evenodd.coverage[20, 50] > 0.9


def test_空の輪郭は空の領域を返す():
    region = R.rasterize_contours([], 10, 10)
    assert region.is_empty
    region = R.rasterize_contours([[1, 2]], 10, 10)  # 点が 1 つでは辺が張れない
    assert region.is_empty


def test_画面の外へ出ても落ちない():
    region = R.rasterize_contours([[-50, -50, 500, -50, 500, 500, -50, 500]], 16, 16)
    assert region.coverage.min() == pytest.approx(1.0)
    assert (region.min_x, region.min_y, region.max_x, region.max_y) == (0, 0, 15, 15)


def test_水平な辺は交点を作らない():
    # 水平な辺だけの «つぶれた» 輪郭は何も塗らない
    region = R.rasterize_contours([[0, 5, 10, 5, 20, 5]], 30, 10)
    assert region.is_empty


# ══════════════════════════════════════════════════════════════════
# 線の回り方（Movo の issue #74 の回帰テスト）
# ══════════════════════════════════════════════════════════════════


def test_細かく折れた線に穴が開かない():
    """«辺の四角形» と «継ぎ目の円» の回り方がそろっていることを確かめる。

    向きが逆だと nonzero 塗りで打ち消し合い、円弧やトリムした線が
    **点線のように見えます**。円周をたどって被覆率が落ちないことを見ます。
    """
    cx = cy = 80.0
    radius = 60.0
    points = R.circle_contour(cx, cy, radius, 96)
    region = R.rasterize_contours(R.stroke_to_contours(points, 2.5, True), 160, 160)

    holes = []
    for i in range(720):
        angle = i / 720 * math.tau
        x = int(round(cx + math.cos(angle) * radius))
        y = int(round(cy + math.sin(angle) * radius))
        # 線の «真上» を丸めた画素は縁に当たることがあるので、隣も見て
        # «そのあたりに墨があるか» で判定します。穴が開いていれば 3x3 が丸ごと空です。
        near = float(region.coverage[y - 1 : y + 2, x - 1 : x + 2].max())
        if near < 0.9:
            holes.append((i, near))
    assert not holes, f"線に穴が開いています: {holes[:8]}"

    # 逆向きの輪郭を混ぜると、まさに #74 の «打ち消し合い» が起きることも見ておく
    broken = []
    for index, contour in enumerate(R.stroke_to_contours(points, 2.5, True)):
        broken.append(contour if index % 3 == 0 else contour.reshape(-1, 2)[::-1].ravel())
    broken_region = R.rasterize_contours(broken, 160, 160)
    assert broken_region.coverage.sum() < region.coverage.sum(), "この比較が成り立たないとテストの意味がありません"


def test_線の輪郭はすべて同じ向きに回る():
    """符号付き面積の符号がそろっていること（穴が開かないことの直接の条件）。"""
    points = R.circle_contour(50, 50, 30, 48)
    for contour in R.stroke_to_contours(points, 4, True):
        xs = contour[0::2]
        ys = contour[1::2]
        area = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert area > 0, "回り方が逆の輪郭があります（#74 の再発）"


def test_折れ線の端には丸い帽子が付く():
    contours = R.stroke_to_contours([10.0, 50.0, 90.0, 50.0], 8, False)
    region = R.rasterize_contours(contours, 100, 100)
    # 端の «外側» にも半径ぶん伸びている
    assert region.coverage[50, 8] > 0.5
    assert region.coverage[50, 92] > 0.5


# ══════════════════════════════════════════════════════════════════
# 塗りと合成モード
# ══════════════════════════════════════════════════════════════════


def test_不透明な塗りは色をそのまま置く():
    bmp = Bitmap(20, 20)
    R.fill_coverage(bmp, R.rasterize_contours([[2, 2, 18, 2, 18, 18, 2, 18]], 20, 20), "#39c5bb")
    assert list(bmp.data[10, 10]) == [57, 197, 187, 255]
    assert list(bmp.data[0, 0]) == [0, 0, 0, 0]


def test_半透明の塗りを重ねると_source_over_になる():
    bmp = Bitmap(8, 8)
    contour = [0, 0, 8, 0, 8, 8, 0, 8]
    R.fill_coverage(bmp, R.rasterize_contours([contour], 8, 8), "rgba(255,0,0,0.5)")
    assert list(bmp.data[4, 4]) == [255, 0, 0, 128]
    R.fill_coverage(bmp, R.rasterize_contours([contour], 8, 8), "rgba(0,0,255,0.5)")
    # 1 回目で 127.5 -> 128 になっているので、outA = 0.5 + (128/255)*0.5 = 0.75098
    # -> 191.5 -> 五捨五入で 192。JS の Uint8ClampedArray と同じ値です。
    assert bmp.data[4, 4, 3] == 192


def test_合成モードは22種そろっている():
    assert len(R.BLEND_MODES) == 22
    assert R.BLEND_MODES[0] == "normal"
    # 番号は kernels 側の定数と 1 対 1
    assert R.blend_id("multiply") == kernels.BLEND_MULTIPLY
    assert R.blend_id("luminosity") == kernels.BLEND_LUMINOSITY
    # 綴りを間違えると黙って normal（JS 版と同じ）
    assert R.blend_id("multipy") == 0


@pytest.mark.parametrize("mode", R.BLEND_MODES)
def test_全ての合成モードで塗れる(mode):
    bmp = Bitmap(16, 16)
    bmp.data[..., :3] = 100
    bmp.data[..., 3] = 255
    R.fill_coverage(bmp, R.rasterize_contours([[0, 0, 16, 0, 16, 16, 0, 16]], 16, 16), "#ff8020", 1.0, mode)
    assert bmp.data[8, 8, 3] == 255


def test_乗算と加算の値():
    def blended(mode, base, src):
        bmp = Bitmap(4, 4)
        bmp.data[..., :3] = base
        bmp.data[..., 3] = 255
        R.fill_coverage(bmp, R.rasterize_contours([[0, 0, 4, 0, 4, 4, 0, 4]], 4, 4), src, 1.0, mode)
        return list(bmp.data[2, 2, :3])

    assert blended("multiply", 128, "#808080") == [64, 64, 64]  # 128*128/255 = 64.25 -> 64
    assert blended("add", 200, "#646464") == [255, 255, 255]  # 200+100 で頭打ち
    assert blended("difference", 200, "#323232") == [150, 150, 150]
    assert blended("darken", 200, "#323232") == [50, 50, 50]
    assert blended("lighten", 200, "#323232") == [200, 200, 200]


def test_非分離の合成モードは3チャンネルまとめて決まる():
    # luminosity は «下の色相・彩度» に «上の明るさ» を移す
    bmp = Bitmap(4, 4)
    bmp.data[..., 0] = 200
    bmp.data[..., 1] = 50
    bmp.data[..., 2] = 50
    bmp.data[..., 3] = 255
    R.fill_coverage(bmp, R.rasterize_contours([[0, 0, 4, 0, 4, 4, 0, 4]], 4, 4), "#ffffff", 1.0, "luminosity")
    r, g, b = (int(v) for v in bmp.data[2, 2, :3])
    # 白の明るさ（255）に合わせるので、全チャンネルが押し上げられて白に近づく
    assert r == g == b == 255


def test_numpy版の合成モードがカーネルと一致する():
    """全画面ぶんは NumPy、画素ごとは Numba。**両方が同じ答えを出すこと。**"""
    rng = np.random.default_rng(7)
    cb = rng.integers(0, 256, (8, 8, 3)).astype(np.float64)
    cs = rng.integers(0, 256, (8, 8, 3)).astype(np.float64)
    for mode in R.BLEND_MODES:
        got = R.blend_rgb(cb, cs, mode)
        expected = np.empty_like(got)
        mode_id = R.blend_id(mode)
        for y in range(8):
            for x in range(8):
                if mode in R.NON_SEPARABLE:
                    expected[y, x] = kernels.blend_non_separable(mode_id, *cb[y, x], *cs[y, x])
                elif mode == "normal":
                    expected[y, x] = cs[y, x]
                else:
                    for c in range(3):
                        expected[y, x, c] = kernels.blend_channel(mode_id, cb[y, x, c], cs[y, x, c])
        assert np.allclose(got, expected, atol=1e-9), mode


def test_画素の丸めはJSのUint8ClampedArrayと同じ():
    """0.5 ちょうどは **偶数側** へ丸まる（切り捨てでも四捨五入でもない）。"""
    assert int(kernels._u8(0.5)) == 0
    assert int(kernels._u8(1.5)) == 2
    assert int(kernels._u8(2.5)) == 2
    assert int(kernels._u8(3.5)) == 4
    assert int(kernels._u8(-3)) == 0
    assert int(kernels._u8(999)) == 255


# ══════════════════════════════════════════════════════════════════
# グラデーション（画素ごとに色が変わる塗り）
# ══════════════════════════════════════════════════════════════════


def test_シェーダで画素ごとに色を変えられる():
    bmp = Bitmap(32, 8)

    def shader(xs, ys):
        out = np.zeros(xs.shape + (4,), np.float64)
        out[..., 0] = xs * 8
        out[..., 3] = 1.0
        return out

    R.fill_coverage_with(bmp, R.rasterize_contours([[0, 0, 32, 0, 32, 8, 0, 8]], 32, 8), shader)
    assert bmp.data[4, 0, 0] == 0
    assert bmp.data[4, 20, 0] == 160
    assert bmp.data[4, 31, 0] == 248


# ══════════════════════════════════════════════════════════════════
# テクスチャ付き三角形
# ══════════════════════════════════════════════════════════════════


def _checker(size=16):
    tex = Bitmap(size, size)
    xs = np.arange(size)
    tex.data[..., 0] = np.clip(xs * 16, 0, 255)[None, :]
    tex.data[..., 1] = np.clip(xs * 16, 0, 255)[:, None]
    tex.data[..., 2] = 200
    tex.data[..., 3] = 255
    return tex


def test_三角形にテクスチャが貼られる():
    tex = _checker()
    dst = Bitmap(32, 32)
    R.draw_textured_triangle(dst, tex, (0, 0, 0, 0), (32, 0, 16, 0), (0, 32, 0, 16))
    assert dst.data[2, 2, 3] == 255
    # 対角線の «向こう側» は塗られない
    assert dst.data[30, 30, 3] == 0
    # 右へ行くほど R が増える（u が増える）
    assert dst.data[4, 20, 0] > dst.data[4, 4, 0]


def test_隣り合う三角形が二重に塗られない():
    """左上規則。共有辺の画素はちょうど 1 回だけ塗られます。"""
    tex = Bitmap(4, 4)
    tex.data[...] = [255, 255, 255, 128]
    dst = Bitmap(16, 16)
    R.draw_textured_triangle(dst, tex, (0, 0, 0, 0), (16, 0, 4, 0), (0, 16, 0, 4), clamp_edge=True)
    R.draw_textured_triangle(dst, tex, (16, 0, 4, 0), (16, 16, 4, 4), (0, 16, 0, 4), clamp_edge=True)
    # 半透明を 1 回塗ると 128、二重に塗ると 191 になる
    assert set(np.unique(dst.data[..., 3])) == {128}


def test_深度バッファで前後関係が決まる():
    tex_far = Bitmap(4, 4)
    tex_far.data[...] = [255, 0, 0, 255]
    tex_near = Bitmap(4, 4)
    tex_near.data[...] = [0, 255, 0, 255]
    dst = Bitmap(16, 16)
    buffer = np.full((16, 16), 1e9, np.float32)
    # 先に «手前»、あとから «奥» を描いても、奥は隠れる
    R.draw_textured_triangle(dst, tex_near, (0, 0, 0, 0), (16, 0, 4, 0), (0, 16, 0, 4),
                             depth={"buffer": buffer, "z": [0.5, 0.5, 0.5]})
    R.draw_textured_triangle(dst, tex_far, (0, 0, 0, 0), (16, 0, 4, 0), (0, 16, 0, 4),
                             depth={"buffer": buffer, "z": [2.0, 2.0, 2.0]})
    assert list(dst.data[2, 2, :3]) == [0, 255, 0]


def test_tint_で色を被せられる():
    tex = Bitmap(4, 4)
    tex.data[...] = [0, 0, 0, 255]
    dst = Bitmap(16, 16)
    R.draw_textured_triangle(dst, tex, (0, 0, 0, 0), (16, 0, 4, 0), (0, 16, 0, 4), tint=(255, 0, 0, 0.5))
    assert list(dst.data[2, 2, :3]) == [128, 0, 0]


# ══════════════════════════════════════════════════════════════════
# 形を作る道具
# ══════════════════════════════════════════════════════════════════


def test_円の輪郭は指定した点数で閉じる():
    points = R.circle_contour(0, 0, 10, 8)
    assert points.size == 16
    radii = np.hypot(points[0::2], points[1::2])
    assert np.allclose(radii, 10)


def test_角丸矩形は半径が辺の半分で頭打ちになる():
    small = R.rect_contour(0, 0, 10, 10, 50)
    # r = min(50, 5, 5) = 5 なので、角が完全な円弧になっても外へ出ない
    xs = small[0::2]
    ys = small[1::2]
    assert min(xs) >= -1e-9 and max(xs) <= 10 + 1e-9
    assert min(ys) >= -1e-9 and max(ys) <= 10 + 1e-9


def test_半径0の角丸矩形はただの矩形():
    assert R.rect_contour(1, 2, 3, 4) == [1, 2, 4, 2, 4, 6, 1, 6]


def test_ベジェの分割数は入力から決まる_決定的():
    a: list[float] = []
    b: list[float] = []
    R.flatten_cubic(a, 0, 0, 10, 40, 30, 40, 40, 0)
    R.flatten_cubic(b, 0, 0, 10, 40, 30, 40, 40, 0)
    assert a == b
    assert len(a) >= 6
    # 終点は指定どおり
    assert a[-2] == pytest.approx(40)
    assert a[-1] == pytest.approx(0)


# ══════════════════════════════════════════════════════════════════
# 領域拡張
# ══════════════════════════════════════════════════════════════════


def _gradient_bitmap(w=8, h=6):
    bmp = Bitmap(w, h)
    bmp.data[..., 0] = np.arange(w)[None, :] * 10
    bmp.data[..., 1] = np.arange(h)[:, None] * 10
    bmp.data[..., 2] = 128
    bmp.data[..., 3] = 255
    return bmp


def test_領域拡張は中身をそのまま真ん中に置く():
    src = _gradient_bitmap()
    out = R.expand_region_with(Bitmap, src, {"all": 3}, 1)
    assert out["bitmap"].width == 14 and out["bitmap"].height == 12
    assert np.array_equal(out["bitmap"].data[3:9, 3:11], src.data)
    assert out["bitmap"].data[0, 0, 3] == 0  # 既定は透明


def test_領域拡張のedgeは端を外へ伸ばす():
    src = _gradient_bitmap()
    out = R.expand_region_with(Bitmap, src, {"all": 2, "fill": "edge"}, 1)["bitmap"]
    assert list(out.data[0, 0]) == list(src.data[0, 0])
    assert list(out.data[0, 5]) == list(src.data[0, 3])
    assert out.data[0, 0, 3] == 255


def test_領域拡張は色でも塗れる_倍率も効く():
    src = _gradient_bitmap()
    out = R.expand_region_with(Bitmap, src, {"all": 2, "fill": "#ff0000"}, 1)["bitmap"]
    assert list(out.data[0, 0]) == [255, 0, 0, 255]
    scaled = R.expand_region_with(Bitmap, src, {"all": 2}, 2)["bitmap"]
    assert scaled.width == 8 + 4 * 2 and scaled.height == 6 + 4 * 2


def test_領域拡張は0なら何もしない_大きすぎても断る():
    assert R.expand_region_with(Bitmap, _gradient_bitmap(), {"all": 0}, 1) is None
    assert R.expand_region_with(Bitmap, _gradient_bitmap(), {"all": 99999}, 1) is None


# ══════════════════════════════════════════════════════════════════
# 色の読み取り
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#39c5bb", (57, 197, 187, 1.0)),
        ("#abc", (170, 187, 204, 1.0)),
        ("#11223344", (17, 34, 51, 68 / 255)),
        ("rgb(1, 2, 3)", (1, 2, 3, 1.0)),
        ("rgba(1,2,3,0.25)", (1, 2, 3, 0.25)),
        ("white", (255, 255, 255, 1.0)),
        ("transparent", (0, 0, 0, 0.0)),
        ("hsl(120, 100%, 50%)", (0, 255, 0, 1.0)),
        ("なにこれ", (0, 0, 0, 1.0)),
    ],
)
def test_色の書き方(text, expected):
    got = R.parse_color(text)
    assert got[:3] == expected[:3]
    assert got[3] == pytest.approx(expected[3])


def test_数の文字列化はJSと同じ():
    # JS の `${1}` は "1"。Python の str(1.0) は "1.0" なので合わせている
    assert R.js_number(1.0) == "1"
    assert R.js_number(0.5) == "0.5"
    assert R.js_number(255) == "255"


# ══════════════════════════════════════════════════════════════════
# 重ね合わせ
# ══════════════════════════════════════════════════════════════════


def test_draw_bitmap_は画面外を刈る():
    dst = Bitmap(10, 10)
    src = Bitmap(6, 6)
    src.data[...] = [255, 255, 255, 255]
    R.draw_bitmap(dst, src, -3, -3)
    assert dst.data[0, 0, 3] == 255
    assert dst.data[3, 3, 3] == 0
    R.draw_bitmap(dst, src, 100, 100)  # 完全に外でも落ちない
    assert dst.data[9, 9, 3] == 0


def test_draw_bitmap_のずらし量はJSのMathroundで丸まる():
    """`Math.round` は 0.5 を **常に上へ**。Python の `round` は偶数丸めです。

    `round(2.5)` は Python では 2、JS では 3 です。1 画素ずれると、影や枠が
    JS 版と食い違います。
    """
    # 負の 0.5 は «上へ»＝ゼロ側へ寄る（JS の Math.round(-2.5) は -2）
    assert [R._js_round(v) for v in (2.4, 2.5, 2.6, 3.5, -2.5, -2.6)] == [2, 3, 3, 4, -2, -3]

    for dx, expected in [(2.4, 2), (2.5, 3), (2.6, 3), (3.5, 4)]:
        dst = Bitmap(12, 4)
        src = Bitmap(2, 4)
        src.data[...] = [255, 255, 255, 255]
        R.draw_bitmap(dst, src, dx, 0)
        assert int(np.nonzero(dst.data[0, :, 3])[0][0]) == expected, dx


def test_draw_bitmap_で効く合成モードは9種だけ():
    """⚠ **`fill_coverage` の 22 種とは違います。**

    JS 版の `compositeBitmap` は `core/src/bitmap.js` にある短い一覧でしか
    照合しないので、`softLight` などは黙って `normal` になります。直したく
    なりますが、直すと **同じ JSON から違う動画が出ます**。
    """
    assert len(R.COMPOSITE_BLEND_MODES) == 9
    assert R.COMPOSITE_BLEND_MODES < set(R.BLEND_MODES)

    def overlay_with(mode):
        dst = Bitmap(4, 4)
        dst.data[...] = [200, 200, 200, 255]
        src = Bitmap(4, 4)
        src.data[...] = [50, 50, 50, 255]
        R.draw_bitmap(dst, src, 0, 0, 1.0, mode)
        return int(dst.data[0, 0, 0])

    # 一覧にあるモードは効く
    assert overlay_with("multiply") == 39  # 200 * 50 / 255 = 39.2
    assert overlay_with("darken") == 50
    assert overlay_with("lighten") == 200
    # 一覧に無いモードは normal に落ちる（＝上の色がそのまま出る）
    for mode in ("softLight", "colorDodge", "linearBurn", "vividLight", "pinLight", "exclusion", "hue"):
        assert overlay_with(mode) == 50, f"{mode} が効いてしまっています（JS 版では normal です）"


def test_draw_bitmap_は透明な画素に触らない():
    dst = Bitmap(4, 4)
    dst.data[...] = [10, 20, 30, 255]
    src = Bitmap(4, 4)  # 全透明
    R.draw_bitmap(dst, src, 0, 0, 1.0)
    assert list(dst.data[2, 2]) == [10, 20, 30, 255]
    # alpha 0 でも下の色はそのまま
    src.data[...] = [255, 255, 255, 255]
    R.draw_bitmap(dst, src, 0, 0, 0.0)
    assert list(dst.data[2, 2]) == [10, 20, 30, 255]


def test_draw_bitmap_は大きい画像でも速い():
    """**NumPy で書いていたころは 94 ms かかっていました。** カーネル化の回帰よけです。

    速度そのものより «一時配列を作っていないこと» を見たいので、しきい値は
    うんと緩めにしてあります（環境差で落ちないように）。
    """
    import time

    dst = Bitmap(960, 540)
    dst.data[...] = [20, 26, 34, 255]
    src = Bitmap(466, 466)
    src.data[..., 3] = 128
    R.draw_bitmap(dst, src, 100, 50, 0.55)  # コンパイルを済ませておく
    start = time.perf_counter()
    for _ in range(5):
        R.draw_bitmap(dst, src, 100, 50, 0.55)
    elapsed = (time.perf_counter() - start) / 5 * 1000
    assert elapsed < 20, f"draw_bitmap が遅くなっています: {elapsed:.1f} ms"


def test_composite_は全画面を一括で重ねる():
    dst = Bitmap(4, 4)
    dst.data[...] = [100, 100, 100, 255]
    src = Bitmap(4, 4)
    src.data[...] = [128, 128, 128, 255]
    R.composite(dst.data, src.data, 1.0, "multiply")
    # 100 * 128 / 255 = 50.196 -> 50
    assert list(dst.data[0, 0]) == [50, 50, 50, 255]
