"""プリセット（エイリアス）の解決（JS 版 tests/presets.test.js の移植）。"""

import copy
import re

import pytest

from movo.expression._compat import MovoError
from movo.schema import describe_presets, normalize_project, resolve_presets, validate_project


def base(extra):
    return {
        "movoVersion": "1.0",
        "project": {"name": "p", "seed": 1},
        "video": {"width": 320, "height": 180, "fps": 30, "duration": 1},
        **extra,
    }


def layers_of(raw):
    return normalize_project(raw)["scenes"][0]["layers"]


def test_the_same_preset_on_two_layers_gives_the_same_result():
    a, b = layers_of(
        base(
            {
                "presets": {
                    "crt": {
                        "effects": [{"type": "scanlines", "amount": 0.3}],
                        "style": {"color": "#fff"},
                    }
                },
                "scenes": [
                    {
                        "id": "m",
                        "layers": [
                            {"id": "a", "type": "text", "preset": "crt"},
                            {"id": "b", "type": "text", "preset": "crt"},
                        ],
                    }
                ],
            }
        )
    )
    assert a["effects"] == b["effects"]
    assert a["style"] == b["style"]
    assert "preset" not in a, "解決後に preset は残らない"


def test_the_layer_beats_the_preset():
    (layer,) = layers_of(
        base(
            {
                "presets": {"brand": {"style": {"color": "#ffffff", "size": 40}}},
                "scenes": [
                    {
                        "id": "m",
                        "layers": [
                            {"id": "a", "type": "text", "preset": "brand", "style": {"color": "#ffd166"}}
                        ],
                    }
                ],
            }
        )
    )
    assert layer["style"]["color"] == "#ffd166", "レイヤーの色が採用される"
    assert layer["style"]["size"] == 40, "書いていないキーはプリセットから来る"


def test_arrays_are_concatenated_with_the_preset_first():
    (layer,) = layers_of(
        base(
            {
                "presets": {"crt": {"effects": [{"type": "scanlines"}, {"type": "vignette"}]}},
                "scenes": [
                    {
                        "id": "m",
                        "layers": [
                            {"id": "a", "type": "text", "preset": "crt", "effects": [{"type": "bloom"}]}
                        ],
                    }
                ],
            }
        )
    )
    assert [e["type"] for e in layer["effects"]] == ["scanlines", "vignette", "bloom"]


def test_preset_merge_replace_swaps_arrays_too():
    (layer,) = layers_of(
        base(
            {
                "presets": {"crt": {"effects": [{"type": "scanlines"}]}},
                "scenes": [
                    {
                        "id": "m",
                        "layers": [
                            {
                                "id": "a",
                                "type": "text",
                                "preset": "crt",
                                "presetMerge": "replace",
                                "effects": [{"type": "bloom"}],
                            }
                        ],
                    }
                ],
            }
        )
    )
    assert [e["type"] for e in layer["effects"]] == ["bloom"]
    assert "presetMerge" not in layer


def test_multiple_presets_stack_in_written_order():
    (layer,) = layers_of(
        base(
            {
                "presets": {
                    "first": {"style": {"color": "#111111", "size": 20}, "effects": [{"type": "a"}]},
                    "second": {"style": {"color": "#222222"}, "effects": [{"type": "b"}]},
                },
                "scenes": [
                    {"id": "m", "layers": [{"id": "x", "type": "text", "preset": ["first", "second"]}]}
                ],
            }
        )
    )
    assert layer["style"]["color"] == "#222222", "後のプリセットが勝つ"
    assert layer["style"]["size"] == 20
    assert [e["type"] for e in layer["effects"]] == ["a", "b"]


def test_extends_pulls_in_another_preset():
    (layer,) = layers_of(
        base(
            {
                "presets": {
                    "crt": {"effects": [{"type": "scanlines"}]},
                    "broken": {"extends": "crt", "effects": [{"type": "glitch"}]},
                },
                "scenes": [{"id": "m", "layers": [{"id": "x", "type": "image", "preset": "broken"}]}],
            }
        )
    )
    assert [e["type"] for e in layer["effects"]] == ["scanlines", "glitch"]


def test_extends_can_be_an_array():
    (layer,) = layers_of(
        base(
            {
                "presets": {
                    "a": {"style": {"size": 10}},
                    "b": {"style": {"color": "#fff"}},
                    "both": {"extends": ["a", "b"]},
                },
                "scenes": [{"id": "m", "layers": [{"id": "x", "type": "text", "preset": "both"}]}],
            }
        )
    )
    assert layer["style"] == {"size": 10, "color": "#fff"}


def test_nested_layers_get_presets_too():
    (group,) = layers_of(
        base(
            {
                "presets": {"fade": {"animations": [{"property": "transform.opacity", "value": 0.5}]}},
                "scenes": [
                    {
                        "id": "m",
                        "layers": [
                            {
                                "id": "g",
                                "type": "group",
                                "layers": [{"id": "child", "type": "text", "preset": "fade"}],
                            }
                        ],
                    }
                ],
            }
        )
    )
    assert group["layers"][0]["animations"][0]["value"] == 0.5


def test_circular_references_are_an_error():
    with pytest.raises(MovoError) as info:
        normalize_project(
            base(
                {
                    "presets": {"a": {"extends": "b"}, "b": {"extends": "a"}},
                    "scenes": [{"id": "m", "layers": []}],
                }
            )
        )
    assert info.value.code == "MOVO_SCHEMA_INVALID"
    assert re.search("循環", info.value.reason)


def test_extending_an_undefined_preset_is_an_error():
    with pytest.raises(MovoError) as info:
        normalize_project(
            base({"presets": {"a": {"extends": "nope"}}, "scenes": [{"id": "m", "layers": []}]})
        )
    assert info.value.code == "MOVO_SCHEMA_INVALID"
    assert re.search("未定義", info.value.reason)


def test_referencing_an_undefined_preset_fails_validation():
    result = validate_project(
        base(
            {
                "presets": {"a": {"style": {}}},
                "scenes": [{"id": "m", "layers": [{"id": "x", "type": "text", "preset": "nope"}]}],
            }
        )
    )
    assert result["valid"] is False
    assert re.search('preset "nope"', result["issues"][0]["message"])
    assert re.search("preset", result["issues"][0]["path"])


def test_preset_metadata_does_not_leak_into_the_layer():
    (layer,) = layers_of(
        base(
            {
                "presets": {"a": {"title": "見出し", "description": "説明", "style": {"size": 10}}},
                "scenes": [{"id": "m", "layers": [{"id": "x", "type": "text", "preset": "a"}]}],
            }
        )
    )
    assert "title" not in layer
    assert "description" not in layer
    assert layer["style"]["size"] == 10


def test_presets_do_not_mutate_the_original_json():
    raw = base(
        {
            "presets": {"a": {"effects": [{"type": "bloom"}]}},
            "scenes": [{"id": "m", "layers": [{"id": "x", "type": "text", "preset": "a"}]}],
        }
    )
    frozen = copy.deepcopy(raw)
    normalize_project(raw)
    assert raw == frozen


def test_a_project_without_presets_still_works():
    stats = resolve_presets({"scenes": [{"id": "m", "layers": [{"id": "x", "type": "text"}]}]})
    assert stats["applied"] == 0


def test_describe_presets_summarises_the_list():
    listed = describe_presets(
        {
            "presets": {
                "crt": {"description": "ブラウン管", "effects": [{"type": "a"}, {"type": "b"}]},
                "broken": {"extends": "crt", "animations": [{"property": "x"}]},
            }
        }
    )
    assert [p["name"] for p in listed] == ["broken", "crt"]
    broken = next(p for p in listed if p["name"] == "broken")
    assert broken["extends"] == ["crt"]
    assert "effects×2" in broken["provides"], broken["provides"]


def test_composition_layers_get_presets_too():
    project = normalize_project(
        base(
            {
                "presets": {"a": {"style": {"size": 11}}},
                "compositions": {
                    "inner": {
                        "width": 100,
                        "height": 100,
                        "layers": [{"id": "c", "type": "text", "preset": "a"}],
                    }
                },
                "scenes": [{"id": "m", "layers": []}],
            }
        )
    )
    assert project["compositions"]["inner"]["layers"][0]["style"]["size"] == 11
