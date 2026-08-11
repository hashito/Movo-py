"""歌詞と曲を合わせる（`movo/audio/align.py`）。

見ているもの:

  - モーラの数え方（小書き・促音・撥音・長音・漢字）
  - 空行でブロックに割れること、繰り返すブロックをサビと見なすこと
  - 歌唱区間の拾い方（短い切れ目を埋めてから短い塊を捨てる、の順序）
  - 割り付けが «時刻順・曲の中・間奏を跨がない» こと
  - **アンカーで留めた行がその時刻ちょうどに来ること**（この仕組みの要）
  - `.lrc` に書いて `parse_lrc` で読み返せること

音そのものの正しさは `test_parity_audio.py` が見ています。ここは «歌詞を
どう置くか» だけです。
"""

from __future__ import annotations

import numpy as np
import pytest

from movo.audio import create_silence
from movo.audio.align import (
    align_lyrics, count_morae, label_blocks, parse_anchors, singing_windows,
    split_blocks, to_lrc, to_scenario,
)
from movo.core.lyrics import parse_lrc

RATE = 24000


def click_track(bpm: float, bars: int, *, rate: int = RATE, beats_per_bar: int = 4):
    """拍の頭でだけ鳴る音。BPM がはっきり出るので、割り付けの検査に向きます。"""
    beats = bars * beats_per_bar
    seconds = beats * 60 / bpm
    audio = create_silence(seconds, rate, 1)
    channel = audio.channels[0]
    period = int(rate * 60 / bpm)
    decay = np.exp(-np.arange(600) / 90.0).astype(np.float32)
    for beat in range(beats):
        start = beat * period
        stop = min(channel.size, start + decay.size)
        if stop > start:
            channel[start:stop] += decay[: stop - start] * 0.6
    return audio


# ── モーラ ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,want",
    [
        ("あいうえお", 5),
        ("きゃ", 1),          # 拗音は前の字と合わせて 1
        ("がっこう", 4),       # 促音は 1 モーラ
        ("ほん", 2),          # 撥音も 1 モーラ
        ("ケーキ", 3),         # 長音も 1 モーラ
        ("シャッター", 4),      # 拗音 + 促音 + 長音
    ],
)
def test_count_morae_counts_japanese_units(text, want):
    assert count_morae(text) == pytest.approx(want)


def test_count_morae_ignores_punctuation_and_spaces():
    assert count_morae("あい、うえ お！") == count_morae("あいうえお")


def test_count_morae_never_returns_zero():
    """0 を返すと配分の重みが消えて、その行の尺が無くなります。"""
    assert count_morae("！？…") > 0
    assert count_morae("") > 0


def test_count_morae_beats_character_count_for_kanji():
    """文字数で割ると漢字の行が短く出る、という動機そのものを固定する。"""
    kanji = "風に揺れる葉っぱの音"
    assert count_morae(kanji) > len(kanji)


# ── 歌詞の構造 ────────────────────────────────────────────────────


def test_split_blocks_splits_on_blank_lines():
    blocks = split_blocks("あ\nい\n\nう\n\n\nえ\n")
    assert [b["lines"] for b in blocks] == [["あ", "い"], ["う"], ["え"]]


def test_label_blocks_marks_repeats_as_chorus():
    text = "あさ\nひる\n\nさびさび\nもえる\n\nよる\nねる\n\nさびさび\nもえる"
    blocks = label_blocks(split_blocks(text))
    assert [b["kind"] for b in blocks] == ["verse", "chorus", "verse", "chorus"]
    # 1 回目のサビは «誰かの繰り返し» ではないので repeatOf は None
    assert blocks[1]["repeatOf"] is None
    assert blocks[3]["repeatOf"] == 1


def test_label_blocks_tolerates_one_word_differences():
    """2 番のサビが 1 語だけ違うのはよくあります。完全一致だと取り逃します。"""
    text = "さびの歌詞がここにながく続いている\nつぎの行\n\nさびの歌詞がここにながく続いていた\nつぎの行"
    blocks = label_blocks(split_blocks(text))
    assert [b["kind"] for b in blocks] == ["chorus", "chorus"]


def test_label_blocks_leaves_unique_blocks_alone():
    blocks = label_blocks(split_blocks("まったく違う内容の一段目\n\nぜんぜん似ていない二段目"))
    assert [b["kind"] for b in blocks] == ["verse", "verse"]


# ── 歌唱区間 ──────────────────────────────────────────────────────


def _presence(pattern: list[tuple[float, float]], hop: float, length: float) -> np.ndarray:
    curve = np.zeros(int(length / hop), np.float64)
    for start, end in pattern:
        curve[int(start / hop): int(end / hop)] = 1.0
    return curve


def test_singing_windows_fills_short_gaps_then_drops_short_runs():
    """息継ぎ（短い切れ目）は埋め、間奏中の一言（短い塊）は捨てる。"""
    hop = 0.01
    # 0〜8 秒に息継ぎ 1 つ、20 秒に 0.5 秒だけの塊、25〜33 秒に本物
    curve = _presence([(0, 4), (4.5, 8), (20, 20.5), (25, 33)], hop, 40)
    windows = singing_windows(curve, hop, 40)
    assert windows[0] == pytest.approx((0.0, 8.0), abs=0.05)
    assert len(windows) == 2
    assert windows[1] == pytest.approx((25.0, 33.0), abs=0.05)


def test_singing_windows_falls_back_to_whole_song():
    """何も拾えないときに «歌う場所が無い» を返すと、全部の歌詞が消えます。"""
    hop = 0.01
    assert singing_windows(np.zeros(1000), hop, 10.0) == [(0.0, 10.0)]


# ── アンカーの読み方 ──────────────────────────────────────────────


def test_parse_anchors_accepts_line_numbers_and_text():
    lines = ["ひとつめ", "ふたつめ", "みっつめ"]
    assert parse_anchors(["1=4.5", "みっつめ=30"], lines) == [(0, 4.5), (2, 30.0)]


def test_parse_anchors_drops_nonsense():
    lines = ["ひとつめ"]
    assert parse_anchors(["9=1", "1=abc", "こわれた", None], lines) == []


# ── 割り付け ──────────────────────────────────────────────────────


LYRICS = "いちぎょうめ\nにぎょうめ\n\nさびのぎょう\nさびのつぎ\n\nさんぎょうめ\nよんぎょうめ"


@pytest.fixture(scope="module")
def aligned():
    return align_lyrics(click_track(120, 16), LYRICS, {"gaps": False})


def test_align_returns_one_entry_per_line(aligned):
    assert [line["text"] for line in aligned["lines"]] == [
        "いちぎょうめ", "にぎょうめ", "さびのぎょう", "さびのつぎ", "さんぎょうめ", "よんぎょうめ",
    ]


def test_align_times_are_ordered_and_inside_the_song(aligned):
    times = [line["at"] for line in aligned["lines"]]
    assert times == sorted(times)
    assert all(t != times[i - 1] for i, t in enumerate(times) if i)
    assert times[0] >= 0
    assert times[-1] <= aligned["duration"]


def test_align_gives_longer_lines_more_time(aligned):
    """モーラ数で割る、の効き目そのもの。"""
    spans = {line["text"]: line["for"] for line in aligned["lines"]}
    assert spans["いちぎょうめ"] > spans["にぎょうめ"]


def test_align_marks_the_repeated_block_as_chorus():
    text = "あさのうた\nひるのうた\n\nさびのぎょう\nさびのつぎ\n\nさびのぎょう\nさびのつぎ"
    result = align_lyrics(click_track(120, 12), text, {"gaps": False})
    kinds = {line["text"]: line["kind"] for line in result["lines"]}
    assert kinds["さびのぎょう"] == "chorus"
    assert kinds["あさのうた"] == "verse"


def test_align_warns_when_there_are_no_anchors(aligned):
    assert any("アンカー" in warning for warning in aligned["warnings"])


# ── アンカー（この仕組みの要） ────────────────────────────────────


def test_anchored_lines_land_exactly_on_the_given_time():
    result = align_lyrics(
        click_track(120, 16), LYRICS,
        {"gaps": False, "snap": False, "anchors": ["3=12.0", "5=20.0"]},
    )
    lines = result["lines"]
    assert lines[2]["at"] == pytest.approx(12.0, abs=0.01)
    assert lines[4]["at"] == pytest.approx(20.0, abs=0.01)
    assert lines[2]["anchored"] is True
    assert lines[2]["confidence"] == 1.0


def test_anchors_redistribute_the_lines_between_them():
    """留めた点のあいだが配分し直される、が «補助» の本体です。"""
    plain = align_lyrics(click_track(120, 16), LYRICS, {"gaps": False, "snap": False})
    pinned = align_lyrics(
        click_track(120, 16), LYRICS, {"gaps": False, "snap": False, "anchors": ["3=12.0"]}
    )
    assert pinned["lines"][2]["at"] != pytest.approx(plain["lines"][2]["at"], abs=0.01)
    # 留めた行の前後も動く（前だけ／後だけ、ではない）
    assert pinned["lines"][1]["at"] != pytest.approx(plain["lines"][1]["at"], abs=0.01)
    assert pinned["lines"][3]["at"] != pytest.approx(plain["lines"][3]["at"], abs=0.01)
    # 順番は保たれる
    times = [line["at"] for line in pinned["lines"]]
    assert times == sorted(times)


def test_anchors_raise_confidence_nearby():
    plain = align_lyrics(click_track(120, 16), LYRICS, {"gaps": False})
    pinned = align_lyrics(click_track(120, 16), LYRICS, {"gaps": False, "anchors": ["1=2.0"]})
    assert pinned["lines"][1]["confidence"] > plain["lines"][1]["confidence"]


# ── 間奏を跨がない ────────────────────────────────────────────────


def test_lines_never_start_inside_a_gap():
    """歌っていないところに歌詞を置かない。"""
    result = align_lyrics(click_track(120, 16), LYRICS, {"gaps": False, "start": 4.0, "end": 24.0})
    for line in result["lines"]:
        assert 4.0 - 0.6 <= line["at"] <= 24.0


def test_line_span_is_clipped_at_the_end_of_its_window():
    """間奏のあいだ «直前の 1 行» が画面に残らないこと。"""
    result = align_lyrics(click_track(120, 16), LYRICS, {"gaps": False, "start": 2.0, "end": 10.0})
    last = result["lines"][-1]
    assert last["at"] + last["for"] <= 10.0 + 1e-6


# ── 書き出し ──────────────────────────────────────────────────────


def test_to_lrc_round_trips_through_parse_lrc(aligned):
    text = to_lrc(aligned["lines"], meta={"ti": "けんさ"})
    assert text.startswith("[ti:けんさ]")
    back = parse_lrc(text)
    assert [line["text"] for line in back] == [line["text"] for line in aligned["lines"]]
    for got, want in zip(back, aligned["lines"]):
        assert got["at"] == pytest.approx(want["at"], abs=0.01)


def test_to_scenario_summarises_by_block(aligned):
    scenario = to_scenario(aligned)
    assert len(scenario["blocks"]) == 3
    assert [b["kind"] for b in scenario["blocks"]] == ["verse", "verse", "verse"]
    assert scenario["barSeconds"] == pytest.approx(2.0, abs=0.01)  # 120BPM / 4 拍
    for block in scenario["blocks"]:
        assert block["end"] > block["start"]


def test_align_handles_empty_lyrics():
    result = align_lyrics(click_track(120, 8), "\n\n  \n", {"gaps": False})
    assert result["lines"] == []
    assert result["warnings"]
