"""キャラの «一枚絵 → 切り出し»（`movo/core/ai_image.py`）。

**外部 API は呼びません。** 呼ぶのは `generate_sheet` だけで、そこは «指示文を
組み立てる» ところと «返ってきた絵を処理する» ところに分けてあります。
ここで見ているのは後者です。

見ているもの:

  - 指示文に格子の数値が入ること（ここが崩れると切り出しが当たらない）
  - 地の緑を抜き、**格子線も抜く**こと
  - 縁がなめらかになること（0/255 の二値だとギザギザになる）
  - 縁の色を外へ染み出させること（緑のまま縮小すると輪郭に緑が滲む）
  - 格子どおりに切れること、返ってきた絵が指定と違う大きさでも合わせること
"""

from __future__ import annotations

import numpy as np
import pytest

from movo.core.ai_image import (
    CHROMA, GRID_LINE, bleed_edges, build_prompt, cut_out_chroma, resolve_api_key, slice_sheet,
)
from movo.core.bitmap import Bitmap
from movo.core.errors import MovoError

SPEC = {
    "theme": "けんさよう",
    "grid": {"cols": 2, "rows": 2, "cellSize": [10, 20]},
    "parts": [
        {"name": "a", "cell": [0, 0], "description": "ひだりうえ"},
        {"name": "b", "cell": [1, 1], "description": "みぎした"},
    ],
}


def green_sheet(width: int, height: int) -> Bitmap:
    """地が緑で埋まった絵。"""
    sheet = Bitmap(width, height)
    sheet.data[..., 0] = CHROMA[0]
    sheet.data[..., 1] = CHROMA[1]
    sheet.data[..., 2] = CHROMA[2]
    sheet.data[..., 3] = 255
    return sheet


# ── 指示文 ────────────────────────────────────────────────────────


def test_build_prompt_states_the_grid_in_numbers():
    """«適当に並べて» では配置が毎回変わり、切り出しの座標が当たりません。"""
    text = build_prompt(SPEC)
    assert "20x40" in text          # 2x10 x 2x20
    assert "2 列 x 2 行" in text
    assert "10x20" in text
    assert "ひだりうえ" in text and "みぎした" in text


def test_build_prompt_forbids_text_and_demands_one_person():
    text = build_prompt(SPEC)
    assert "同一人物" in text
    assert "文字" in text


def test_build_prompt_falls_back_to_the_part_name():
    spec = {**SPEC, "parts": [{"name": "なまえだけ", "cell": [0, 0]}]}
    assert "なまえだけ" in build_prompt(spec)


# ── 地と格子線を抜く ──────────────────────────────────────────────


def test_cut_out_chroma_removes_the_green_ground():
    sheet = green_sheet(8, 8)
    sheet.data[2:6, 2:6] = [200, 40, 40, 255]  # 中身
    cut = cut_out_chroma(sheet)
    assert cut.data[0, 0, 3] == 0
    assert cut.data[4, 4, 3] == 255


def test_cut_out_chroma_also_removes_the_grid_lines():
    """線が残ると «不透明な画素» になり、余白の刈り取りが効きません。"""
    sheet = green_sheet(8, 8)
    sheet.data[4, :] = [*GRID_LINE, 255]
    cut = cut_out_chroma(sheet)
    assert cut.data[4, 0, 3] == 0


def test_cut_out_chroma_removes_blended_grid_lines():
    """線は細くて地と混ざるので、純マゼンタでは来ません（実測 [203, 67, 150]）。"""
    sheet = green_sheet(8, 8)
    sheet.data[4, :] = [203, 67, 150, 255]
    cut = cut_out_chroma(sheet)
    assert cut.data[4, 0, 3] == 0


def test_cut_out_chroma_keeps_reddish_colours():
    """マゼンタの許容差を広げすぎると «赤っぽい色» が半透明になります。

    実際に [200, 40, 40] の赤が alpha 166 まで削られました。唇・頬・赤い服が
    薄くなる、という気付きにくい壊れ方をします。
    """
    sheet = green_sheet(4, 4)
    sheet.data[1, 1] = [200, 40, 40, 255]
    assert cut_out_chroma(sheet).data[1, 1, 3] == 255


def test_cut_out_chroma_keeps_skin_and_hair():
    """許容差を広げすぎて肌や髪まで抜けては困ります。"""
    sheet = green_sheet(8, 8)
    sheet.data[1, 1] = [250, 235, 175, 255]  # 肌
    sheet.data[2, 2] = [150, 90, 60, 255]    # 髪
    sheet.data[3, 3] = [80, 140, 70, 255]    # 緑の服
    cut = cut_out_chroma(sheet)
    assert cut.data[1, 1, 3] == 255
    assert cut.data[2, 2, 3] == 255
    assert cut.data[3, 3, 3] == 255


def test_cut_out_chroma_makes_a_soft_edge():
    """二値で抜くと輪郭がギザギザになり、縮小したとき階段が目立ちます。"""
    sheet = green_sheet(8, 8)
    # 地の緑から «少しだけ» 離れた色を置くと、途中の alpha になるはず
    sheet.data[3, 3] = [70, 200, 40, 255]
    cut = cut_out_chroma(sheet)
    assert 0 < int(cut.data[3, 3, 3]) < 255


def test_cut_out_chroma_does_not_overflow():
    """色距離を int16 で計算すると 255^2 x 3 が上限を超えて負になります。"""
    sheet = green_sheet(4, 4)
    sheet.data[0, 0] = [255, 0, 255, 255]   # 最も遠い組み合わせ
    with np.errstate(invalid="raise"):
        cut_out_chroma(sheet)


# ── 縁の染み出し ──────────────────────────────────────────────────


def test_bleed_edges_pushes_colour_outwards():
    """アルファを 0 にしても RGB は緑のまま。そのまま縮小すると緑が滲みます。"""
    sheet = green_sheet(8, 8)
    sheet.data[3:5, 3:5] = [200, 40, 40, 255]
    cut = cut_out_chroma(sheet)
    bled = bleed_edges(cut, passes=2)
    # 中身の隣（透明）の色が «緑» ではなく «中身の色» に寄っている
    assert bled.data[2, 3, 1] < 200
    assert bled.data[2, 3, 3] == 0      # 透明のままであること


def test_bleed_edges_keeps_the_visible_pixels():
    sheet = green_sheet(8, 8)
    sheet.data[3:5, 3:5] = [200, 40, 40, 255]
    cut = cut_out_chroma(sheet)
    bled = bleed_edges(cut, passes=3)
    assert list(bled.data[3, 3]) == [200, 40, 40, 255]


def test_bleed_edges_survives_a_fully_transparent_image():
    assert bleed_edges(Bitmap(4, 4), passes=2).width == 4


# ── 切り出し ──────────────────────────────────────────────────────


def test_slice_sheet_cuts_by_the_grid(tmp_path):
    sheet = Bitmap(20, 40)
    sheet.data[..., 3] = 255
    sheet.data[0:20, 0:10] = [10, 20, 30, 255]     # 左上
    sheet.data[20:40, 10:20] = [90, 80, 70, 255]   # 右下
    written = slice_sheet(sheet, SPEC, tmp_path, trim=False)
    assert set(written) == {"a", "b"}

    from movo.core.png import decode_png
    from pathlib import Path

    a = decode_png(Path(written["a"]).read_bytes())
    assert (a.width, a.height) == (10, 20)
    assert list(a.data[0, 0]) == [10, 20, 30, 255]


def test_slice_sheet_scales_when_the_image_comes_back_bigger(tmp_path):
    """返ってくる絵が指定どおりの大きさとは限りません。"""
    sheet = Bitmap(40, 80)  # 指定の 2 倍
    sheet.data[..., 3] = 255
    written = slice_sheet(sheet, SPEC, tmp_path, trim=False)

    from movo.core.png import decode_png
    from pathlib import Path

    a = decode_png(Path(written["a"]).read_bytes())
    assert (a.width, a.height) == (20, 40)


def test_slice_sheet_trims_the_transparent_margin(tmp_path):
    sheet = Bitmap(20, 40)
    sheet.data[4:12, 2:8] = [200, 40, 40, 255]
    written = slice_sheet(sheet, SPEC, tmp_path, trim=True)

    from movo.core.png import decode_png
    from pathlib import Path

    a = decode_png(Path(written["a"]).read_bytes())
    assert a.height < 20


# ── API キー ──────────────────────────────────────────────────────


def test_resolve_api_key_prefers_the_argument():
    assert resolve_api_key("あたえた鍵") == "あたえた鍵"


def test_resolve_api_key_reads_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "かんきょうへんすう")
    assert resolve_api_key() == "かんきょうへんすう"


def test_resolve_api_key_explains_how_to_set_it(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GPT_API_KEY", raising=False)
    monkeypatch.setattr("movo.cli.config_store.get_config_value", lambda key: None)
    with pytest.raises(MovoError) as caught:
        resolve_api_key()
    assert "movo config set" in (caught.value.hint or "")


# ── 線と断片を落とす ──────────────────────────────────────────────


def test_remove_thin_marks_drops_a_thin_line_and_keeps_a_blob():
    from movo.core.ai_image import remove_thin_marks

    sheet = Bitmap(40, 40)
    sheet.data[10:30, 10:30] = [200, 40, 40, 255]   # 太い塊（キャラ）
    sheet.data[35, :] = [255, 0, 255, 255]          # 細い線（格子線）
    out = remove_thin_marks(sheet, radius=2)
    assert out.data[20, 20, 3] == 255               # 塊は残る
    assert out.data[35, 5, 3] == 0                  # 線は消える


def test_remove_thin_marks_keeps_a_thin_feature_that_is_attached():
    """アホ毛や指のような «細いが本体に繋がっている» ところは残したい。"""
    from movo.core.ai_image import remove_thin_marks

    sheet = Bitmap(40, 40)
    sheet.data[15:35, 10:30] = [200, 40, 40, 255]
    sheet.data[5:15, 18:24] = [200, 40, 40, 255]    # 6px 幅の «アホ毛»
    out = remove_thin_marks(sheet, radius=2)
    assert out.data[10, 21, 3] == 255


def test_keep_largest_blob_removes_detached_fragments():
    """実際に踏んだ壊れ方: 頭上に 8px の破片が残り、細さでは落とせなかった。"""
    from movo.core.ai_image import keep_largest_blob

    sheet = Bitmap(40, 40)
    sheet.data[12:32, 12:32] = [200, 40, 40, 255]   # 本体
    sheet.data[2:6, 4:16] = [200, 40, 40, 255]      # 離れた破片
    out = keep_largest_blob(sheet)
    assert out.data[20, 20, 3] == 255
    assert out.data[4, 10, 3] == 0


def test_keep_largest_blob_survives_an_empty_image():
    from movo.core.ai_image import keep_largest_blob

    assert keep_largest_blob(Bitmap(8, 8)).width == 8


def test_slice_sheet_applies_the_blob_filter_per_part(tmp_path):
    """⚠ シート全体に掛けると «いちばん大きな 1 体» 以外が全部消えます。

    実際に 4 ポーズを全滅させました。切り出した «後» に、パーツごとに掛けます。
    """
    sheet = Bitmap(20, 40)
    sheet.data[2:18, 2:8] = [200, 40, 40, 255]      # 左上のセルの中身
    sheet.data[22:38, 12:18] = [40, 200, 40, 255]   # 右下のセルの中身（別の塊）
    written = slice_sheet(sheet, SPEC, tmp_path, trim=False)

    from movo.core.png import decode_png
    from pathlib import Path

    for name in ("a", "b"):
        img = decode_png(Path(written[name]).read_bytes())
        assert (img.data[..., 3] > 8).any(), f"{name} が消えている"
