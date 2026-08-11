"""拍・小節による時間指定（JS 版 tests/musical-time.test.js の移植）。

MV ではカット尺を «小節» で決めるのが基本なので、秒に手で直さずに書けること。
既存の JSON（数値だけ）が 1 つも変わらないことも合わせて見る。
"""

import copy

import pytest

from movo.expression._compat import MovoError
from movo.schema import is_musical_time, musical_units, normalize_project, to_seconds


def near(actual, expected, tolerance=1e-9):
    assert abs(actual - expected) < tolerance, f"{actual} ≈ {expected} ではありません"


def test_musical_units_from_bpm_and_time_signature():
    u = musical_units({"bpm": 120})
    near(u["beat"], 0.5)
    near(u["bar"], 2)
    three = musical_units({"bpm": 120, "timeSignature": [3, 4]})
    near(three["bar"], 1.5)
    as_string = musical_units({"bpm": 120, "timeSignature": "6/8"})
    near(as_string["bar"], 3)
    assert musical_units({}) is None


def test_to_seconds_accepts_bar_beat_fraction_and_seconds():
    u = musical_units({"bpm": 205})
    near(to_seconds("4bar", u), (60 / 205) * 4 * 4)
    near(to_seconds("8beat", u), (60 / 205) * 8)
    near(to_seconds("1/2bar", u), (60 / 205) * 4 * 0.5)
    near(to_seconds("0.75s", u), 0.75)
    assert to_seconds(1.5, u) == 1.5
    # 対象外の文字列はそのまま返す（式や色を壊さない）
    assert to_seconds("${duration}", u) == "${duration}"
    assert to_seconds("#ff0000", u) == "#ff0000"


def test_is_musical_time_only_matches_the_beat_notation():
    assert is_musical_time("4bar")
    assert is_musical_time("1/2bar")
    assert is_musical_time(" 16 beats ")
    assert not is_musical_time("4")
    assert not is_musical_time("bar")
    assert not is_musical_time("#e8382d")
    assert not is_musical_time("${bpm}")
    # 改行を挟んだものを通すと、歌詞の途中が時間になってしまう
    assert not is_musical_time("4bar\nx")


def test_normalize_converts_scene_layer_and_keyframe_times_to_seconds():
    bar = (60 / 205) * 4
    project = normalize_project(
        {
            "project": {"bpm": 205},
            "video": {"width": 64, "height": 64, "duration": "18bar"},
            "scenes": [
                {
                    "id": "a",
                    "duration": "4bar",
                    "transition": {"type": "fade", "in": "1beat", "out": "1beat"},
                    "layers": [
                        {
                            "id": "x",
                            "type": "text",
                            "start": "2beat",
                            "end": "3bar",
                            "animations": [
                                {
                                    "property": "transform.x",
                                    "keyframes": [
                                        {"time": "0bar", "value": 0},
                                        {"time": "2bar", "value": 10},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    near(project["video"]["duration"], bar * 18)
    near(project["scenes"][0]["duration"], bar * 4)
    near(project["scenes"][0]["transition"]["in"], 60 / 205)
    layer = project["scenes"][0]["layers"][0]
    near(layer["start"], (60 / 205) * 2)
    near(layer["end"], bar * 3)
    near(layer["animations"][0]["keyframes"][1]["time"], bar * 2)


def test_normalize_leaves_numeric_json_untouched():
    raw = {
        "project": {"bpm": 150},
        "video": {"width": 64, "height": 64, "duration": 4},
        "scenes": [
            {"id": "a", "duration": 4, "layers": [{"id": "x", "type": "shape", "start": 0.5, "end": 3.5}]}
        ],
    }
    project = normalize_project(copy.deepcopy(raw))
    assert project["video"]["duration"] == 4
    assert project["scenes"][0]["layers"][0]["start"] == 0.5
    assert project["scenes"][0]["layers"][0]["end"] == 3.5


def test_normalize_does_not_break_colours_or_expressions():
    project = normalize_project(
        {
            "project": {"bpm": 150},
            "video": {"width": 64, "height": 64, "duration": 4},
            "scenes": [
                {
                    "id": "a",
                    "duration": 4,
                    "layers": [
                        {
                            "id": "x",
                            "type": "shape",
                            "shape": {"type": "rectangle", "width": 10, "height": 10, "fill": "#e8382d"},
                            "transform": {"x": {"expression": "sin(time) * 4"}, "opacity": 1},
                        }
                    ],
                }
            ],
        }
    )
    layer = project["scenes"][0]["layers"][0]
    assert layer["shape"]["fill"] == "#e8382d"
    assert layer["transform"]["x"]["expression"] == "sin(time) * 4"


def test_writing_beats_without_a_bpm_gives_a_readable_error():
    with pytest.raises(MovoError) as info:
        normalize_project(
            {
                "video": {"width": 64, "height": 64, "duration": 4},
                "scenes": [{"id": "a", "duration": "4bar", "layers": []}],
            }
        )
    assert "bpm" in info.value.reason
