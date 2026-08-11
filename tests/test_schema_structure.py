"""曲の構造（区間）をシーンに貼る（JS 版 tests/structure.test.js の移植）。

解析は前からできていたのに **読む側が一人もいなかった** ので、
`movo batch` で 10 曲流しても «どの曲もイントロ 4 小節» になっていました。
ここでは «曲を差し替えると構成が付いてくる» ことを確かめます。
"""

import copy

import pytest

from movo.expression._compat import MovoError
from movo.schema import find_section, normalize_project, resolve_structure, structure_of

SECTIONS = [
    {"start": 0, "end": 4, "energy": 0.2, "label": "intro"},
    {"start": 4, "end": 12, "energy": 0.9, "label": "chorus"},
    {"start": 12, "end": 18, "energy": 0.5, "label": "verse"},
    {"start": 18, "end": 24, "energy": 0.95, "label": "chorus"},
    {"start": 24, "end": 28, "energy": 0.3, "label": "outro"},
]


def project_with(scenes):
    return {
        "project": {"name": "t", "bpm": 120},
        "video": {"width": 320, "height": 180, "fps": 12},
        "structure": {"sections": copy.deepcopy(SECTIONS)},
        "scenes": scenes,
    }


def test_indexing_a_section_gives_its_length_and_start():
    out = resolve_structure(project_with([{"id": "a", "from": {"section": 1}}]))
    assert out["scenes"][0]["start"] == 4
    assert out["scenes"][0]["duration"] == 8
    assert out["scenes"][0]["_section"]["label"] == "chorus"
    # 畳んだあとに from は残さない（正規化の後段が知らなくてよい）
    assert "from" not in out["scenes"][0]


def test_label_and_nth_select_the_second_chorus():
    out = resolve_structure(
        project_with(
            [
                {"id": "a", "from": {"section": "chorus"}},
                {"id": "b", "from": {"section": "chorus", "nth": 2}},
            ]
        )
    )
    assert out["scenes"][0]["start"] == 4, "nth を省いたら 1 番目"
    assert out["scenes"][1]["start"] == 18
    assert out["scenes"][1]["duration"] == 6


def test_negative_index_counts_from_the_end():
    out = resolve_structure(project_with([{"id": "a", "from": {"section": -1}}]))
    assert out["scenes"][0]["_section"]["label"] == "outro"
    assert out["scenes"][0]["start"] == 24


def test_a_declared_duration_wins():
    out = resolve_structure(project_with([{"id": "a", "from": {"section": 1}, "duration": 3}]))
    assert out["scenes"][0]["duration"] == 3, "書いた尺が勝つ"
    assert out["scenes"][0]["start"] == 4, "開始位置は区間から来る"


def test_start_bar_negative_counts_from_the_end_of_the_song():
    # 120 BPM / 4 拍 = 1 小節 2 秒。曲は 28 秒なので 28 - 8 = 20 秒。
    out = resolve_structure(project_with([{"id": "a", "start": {"bar": -4}, "duration": 8}]))
    assert out["scenes"][0]["start"] == 20


def test_start_bar_positive_counts_from_the_beginning():
    out = resolve_structure(project_with([{"id": "a", "start": {"bar": 2}, "duration": 4}]))
    assert out["scenes"][0]["start"] == 4


def test_pointing_at_a_missing_section_says_what_exists():
    with pytest.raises(MovoError) as info:
        resolve_structure(project_with([{"id": "a", "from": {"section": "ラララ"}}]))
    assert info.value.code == "MOVO_SCHEMA_INVALID"
    assert "intro / chorus / verse / outro" in (info.value.hint or "")


def test_pointing_at_a_section_without_a_song_says_how_to_pass_one():
    with pytest.raises(MovoError) as info:
        resolve_structure(
            {"project": {"bpm": 120}, "video": {}, "scenes": [{"id": "a", "from": {"section": 0}}]}
        )
    assert info.value.code == "MOVO_SCHEMA_INVALID"
    assert "fromAudio" in (info.value.hint or "")


def test_a_hand_written_structure_beats_the_analysis():
    project = {
        "structure": {"sections": [{"start": 0, "end": 9, "energy": 1, "label": "written"}]},
        "_audioAnalysis": {"sections": SECTIONS},
    }
    assert structure_of(project)["sections"][0]["label"] == "written"


def test_the_analysis_is_used_when_nothing_is_written():
    project = {"_audioAnalysis": {"sections": SECTIONS}}
    assert len(structure_of(project)["sections"]) == 5


def test_find_section_returns_none_rather_than_raising():
    assert find_section(SECTIONS, {"section": "ラララ"}) is None
    assert find_section(SECTIONS, {"section": 99}) is None


def test_normalising_makes_the_video_as_long_as_the_song():
    normalized = normalize_project(
        project_with(
            [
                {"id": "a", "from": {"section": 0}, "layers": []},
                {"id": "b", "from": {"section": 1}, "layers": []},
                {"id": "c", "from": {"section": -1}, "layers": []},
            ]
        )
    )
    assert [[s["start"], s["duration"]] for s in normalized["scenes"]] == [[0, 4], [4, 8], [24, 4]]


def test_json_without_sections_is_left_alone():
    before = {"project": {"bpm": 120}, "video": {}, "scenes": [{"id": "a", "start": 1, "duration": 2}]}
    after = resolve_structure(copy.deepcopy(before))
    assert after["scenes"] == before["scenes"]
