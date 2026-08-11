"""他の担当が移植中のモジュールへの «繋ぎ口»。

守りたいのは 2 つです。

1. 未接続を **黙って通さない**（`None` が奥まで流れて `AttributeError` になるのが
   いちばん困る壊れ方です）
2. 未接続を **見えるようにする**（`movo doctor` が «全部繋がっています» と
   嘘をつかない）
"""

import pytest

from movo.cli import bridge


def test_目印まで見て繋がりを判定する():
    # import が通っただけでは «中身が空» のことがあります（移植中は必ず通る状態）。
    assert bridge.is_connected("movo.core.bitmap") is True
    assert bridge.is_connected("movo.そんなもの") is False


def test_未接続を使うと名指しで止まる():
    function = bridge.pick("movo.まだない", "何か")
    with pytest.raises(bridge.NotConnectedError) as caught:
        function()
    assert "後で繋ぐ" in str(caught.value)


def test_繋がっていれば普通に呼べる():
    encode = bridge.pick("movo.core.png", "encode_png")
    assert callable(encode)
    assert getattr(encode, "movo_not_connected", False) is False


def test_一覧は関数でも定数でも受ける():
    # `list_effects()` のような関数と、`BLEND_MODES` のような定数が混在します。
    # 関数として呼ぶ前提で書くと、定数のほうが黙って «0 件» になります。
    assert bridge.listing("movo.renderer.effects", "list_effects") != []
    assert bridge.listing("movo.renderer.raster", "BLEND_MODES") != []


def test_一覧は未接続でも止まらない():
    assert bridge.listing("movo.まだない", "list_x") == []


def test_繋がり具合の一覧には理由が付く():
    rows = bridge.module_status()
    assert rows
    for row in rows:
        assert set(row) == {"module", "label", "connected", "reason"}
        # 繋がっていないものには «なぜ» が必ず入っていること
        assert row["connected"] or row["reason"]


def test_生の_RGBA_を_Bitmap_にできる():
    # ffmpeg から受けたフレームを閃光検査へ渡すのに使います。
    bitmap = bridge.to_bitmap(2, 2, bytes([255, 0, 0, 255] * 4))
    assert bitmap.width == 2 and bitmap.height == 2
    assert bitmap.data.shape == (2, 2, 4)
    assert int(bitmap.data[0, 0, 0]) == 255
