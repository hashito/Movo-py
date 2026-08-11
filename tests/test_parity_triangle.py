"""JS 版と «画素そのもの» を突き合わせる（テクスチャ付き三角形）。

基準の作り直しかた:

    node tests/data/parity_triangle.mjs > tests/data/parity_triangle.json

見ているもの:

  - `drawTexturedTriangle` を 2 枚重ねた絵の **全画素・全チャンネル**

## なぜこの検査が要るか

合成の式 `outC = (cs * sa + cb * da * (1 - sa)) / outA` は、**掛ける順を変えると
最後の 1 ビットが変わります。** JS は左から順に `(cb * da) * (1 - sa)` と評価します。

Numba 側では `fastmath` の扱いに落とし穴があります。`fastmath=True` を付けると
LLVM が式を勝手に括り直す（`reassoc`）ので、**Python 側で JS と同じ並びに書いても
その並びが消えます。** そのため合成のカーネルには
{@link movo.renderer.kernels.PARITY_FASTMATH} を付けています。

この検査は下地のアルファを場所ごとに変えてあります。下地が透明だと
`cb * da * (1 - sa)` の項が丸ごと消えて、**違いが出ないまま通ってしまいます。**
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.renderer.raster import draw_textured_triangle

GOLDEN = json.loads((Path(__file__).parent / "data" / "parity_triangle.json").read_text("utf-8"))

#: プレーンの 4 隅（TL, TR, BR, BL）。`.mjs` と同じ値であること。
TL = (4.3, 3.7, 0, 0)
TR = (57.1, 9.2, 40, 0)
BR = (52.6, 42.4, 40, 28)
BL = (7.8, 38.1, 0, 28)


def make_image(width: int, height: int, tweak: int = 0) -> Bitmap:
    """`tests/test_effects_parity.py` と同じ試験用の画像。"""
    ys, xs = np.mgrid[0:height, 0:width]
    data = np.zeros((height, width, 4), np.uint8)
    data[..., 0] = (xs * 7 + ys * 13 + tweak * 31) % 256
    data[..., 1] = (xs * xs + ys * 3 + 40 + tweak * 17) % 256
    data[..., 2] = ((xs * 5) ^ (ys * 11)) % 256
    edge = (xs < 3) | (ys < 2) | (xs > width - 4) | (ys > height - 3)
    data[..., 3] = np.where(edge, 0, np.where((xs + ys) % 5 == 0, 90, 255))
    return Bitmap(width, height, data)


def make_dest(width: int, height: int) -> Bitmap:
    """下地。**アルファを 40..255 で散らします**（透明にはしない）。"""
    ys, xs = np.mgrid[0:height, 0:width]
    data = np.zeros((height, width, 4), np.uint8)
    data[..., 0] = (xs * 3 + ys * 5 + 17) % 256
    data[..., 1] = (xs * 11 + ys * 2 + 90) % 256
    data[..., 2] = ((xs * 13) ^ (ys * 7)) % 256
    data[..., 3] = 40 + ((xs * 9 + ys * 6) % 216)
    return Bitmap(width, height, data)


def _draw_plane() -> Bitmap:
    dst = make_dest(GOLDEN["width"], GOLDEN["height"])
    texture = make_image(40, 28)
    # 2 枚目は 1 枚目が書いた画素の上にも乗ります（da > 0 の経路を通すため）
    draw_textured_triangle(dst, texture, TL, TR, BR, alpha=0.9, blend="normal", clamp_edge=True)
    draw_textured_triangle(dst, texture, TL, BR, BL, alpha=0.9, blend="normal", clamp_edge=True)
    return dst


def test_matches_js():
    """JS 版と **1 画素も** 違わないこと。"""
    got = _draw_plane().data
    want = np.array(GOLDEN["data"], np.uint8).reshape(GOLDEN["height"], GOLDEN["width"], 4)
    diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
    bad = int((diff != 0).sum())
    if bad:
        ys, xs = np.nonzero(diff.any(axis=2))
        first = [(int(x), int(y), want[y, x].tolist(), got[y, x].tolist()) for y, x in list(zip(ys, xs))[:5]]
        pytest.fail(
            f"{bad} 個のチャンネルが JS 版と違います（最大 {int(diff.max())}）。"
            f" 先頭: {first}（x, y, js, py）"
        )


def test_actually_covers_the_blend():
    """**下地も三角形も «効いている» こと。**

    下地が透明だったり三角形が画面外だったりすると、上の検査は
    «何も起きていない» まま通ってしまいます。その取りこぼしを防ぎます。
    """
    before = make_dest(GOLDEN["width"], GOLDEN["height"])
    after = _draw_plane()
    assert (before.data[..., 3] > 0).all(), "下地が透明では合成の項が消えます"
    changed = int((before.data != after.data).any(axis=2).sum())
    total = GOLDEN["width"] * GOLDEN["height"]
    # 四隅は画面いっぱいではなく、素材の縁も透明なので 4 割ほどが掛かります
    assert changed > total // 4, f"三角形が {changed}/{total} 画素にしか掛かっていません"
