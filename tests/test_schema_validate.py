"""検証と正規化（JS 版 tests/schema.test.js の移植）。

タイムラインに関わる 1 件は timeline パッケージ側の担当なので外してあります。
"""

import copy
import re

from movo.schema import SchemaValidator, normalize_project, validate_project

minimal = {
    "movoVersion": "1.0",
    "video": {"width": 640, "height": 360, "fps": 24, "duration": 2},
    "assets": {"logo": "assets/images/logo.png"},
    "scenes": [
        {
            "id": "main",
            "start": 0,
            "duration": 2,
            "layers": [{"id": "a", "type": "image", "asset": "logo"}],
        }
    ],
}


def test_a_minimal_project_validates():
    result = validate_project(minimal)
    assert result["valid"] is True, result["issues"]


def test_missing_video_is_rejected():
    result = validate_project({"movoVersion": "1.0"})
    assert result["valid"] is False
    assert result["issues"][0]["path"] == "video"


def test_modulator_frequency_of_zero_is_rejected_with_an_authoring_path():
    project = copy.deepcopy(minimal)
    project["scenes"][0]["layers"][0]["modifiers"] = [{"type": "wave", "frequency": 0}]
    result = validate_project(project)
    assert result["valid"] is False
    issue = next((i for i in result["issues"] if "modifiers[0].frequency" in i["path"]), None)
    assert issue, result["issues"]
    assert re.search("greater than 0", issue["message"])


def test_references_to_undeclared_assets_are_reported():
    project = copy.deepcopy(minimal)
    project["scenes"][0]["layers"][0]["asset"] = "nope"
    result = validate_project(project)
    assert result["valid"] is False
    assert re.search("not declared", result["issues"][0]["message"])


def test_duplicate_layer_ids_are_reported():
    project = copy.deepcopy(minimal)
    project["scenes"][0]["layers"].append({"id": "a", "type": "text", "text": "hi"})
    result = validate_project(project)
    assert result["valid"] is False
    assert any(re.search("duplicate layer id", i["message"]) for i in result["issues"])


def test_broken_expressions_are_reported():
    project = copy.deepcopy(minimal)
    project["scenes"][0]["layers"][0]["animations"] = [
        {"property": "transform.x", "expression": "sin(time"}
    ]
    result = validate_project(project)
    assert result["valid"] is False
    assert any("expression" in i["path"] for i in result["issues"])


def test_unknown_deformer_types_are_warnings_not_errors():
    project = copy.deepcopy(minimal)
    project["scenes"][0]["layers"][0]["modifiers"] = [{"type": "notAThing"}]
    result = validate_project(project, known_deformers={"bend"}, known_effects={"blur"})
    assert result["valid"] is True
    assert len(result["warnings"]) == 1


def test_rig_parents_and_ik_chains_are_checked():
    project = copy.deepcopy(minimal)
    project["characters"] = {
        "p": {
            "parts": [{"id": "body"}, {"id": "arm", "parent": "ghost"}],
            "ik": [{"chain": ["body", "missing"]}],
        }
    }
    result = validate_project(project)
    assert result["valid"] is False
    assert any(re.search("unknown parent part", i["message"]) for i in result["issues"])
    assert any(re.search('unknown part "missing"', i["message"]) for i in result["issues"])


def test_normalize_fills_defaults_and_converts_top_level_layers_into_a_scene():
    project = normalize_project(
        {"video": {"width": 100, "height": 50}, "layers": [{"type": "text", "text": "x"}]}
    )
    assert project["video"]["fps"] == 30
    assert project["render"]["quality"] == "standard"
    assert len(project["scenes"]) == 1
    assert len(project["scenes"][0]["layers"]) == 1
    assert project["physicsWorld"]["gravity"]["y"] == 980
    assert project["deterministic"]["seed"] == project["project"]["seed"]
    assert "layers" not in project


def test_quality_presets_change_mesh_resolution_and_supersampling():
    draft = normalize_project({"video": {"width": 10, "height": 10}, "render": {"quality": "draft"}})
    ultra = normalize_project({"video": {"width": 10, "height": 10}, "render": {"quality": "ultra"}})
    assert ultra["render"]["deformation"]["meshResolution"] > draft["render"]["deformation"]["meshResolution"]
    assert ultra["render"]["superSample"] > draft["render"]["superSample"]


def test_the_standalone_validator_supports_one_of_refs_and_bounds():
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {"a": {"$ref": "#/definitions/num"}},
            "definitions": {"num": {"oneOf": [{"type": "number", "minimum": 1}, {"type": "string"}]}},
        }
    )
    assert validator.validate({"a": 5})["valid"] is True
    assert validator.validate({"a": "x"})["valid"] is True
    assert validator.validate({"a": 0})["valid"] is False
    # JSON の true は number ではない（Python の bool を数と見ない）
    assert validator.validate({"a": True})["valid"] is False


def test_keyframes_outside_the_visible_range_are_warned():
    # カラオケ塗りが 3 秒までかかるのに、レイヤーは 2.5 秒で消える場合
    result = validate_project(
        {
            "video": {"width": 100, "height": 100, "duration": 3},
            "scenes": [
                {
                    "id": "m",
                    "duration": 2.5,
                    "layers": [
                        {
                            "id": "a",
                            "type": "text",
                            "text": "x",
                            "karaoke": {
                                "progress": {
                                    "keyframes": [
                                        {"time": 0.3, "value": 0},
                                        {"time": 3, "value": 1},
                                    ]
                                }
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert result["valid"] is True, "エラーではなく警告"
    warning = next((w for w in result["warnings"] if "途中で止まった" in w["message"]), None)
    assert warning, "警告が出る"
    assert re.search(r"karaoke\.progress", warning["path"])


def test_keyframes_inside_the_visible_range_are_not_warned():
    result = validate_project(
        {
            "video": {"width": 100, "height": 100, "duration": 3},
            "scenes": [
                {
                    "id": "m",
                    "duration": 2.5,
                    "layers": [
                        {
                            "id": "a",
                            "type": "text",
                            "text": "x",
                            "karaoke": {
                                "progress": {
                                    "keyframes": [
                                        {"time": 0.3, "value": 0},
                                        {"time": 2, "value": 1},
                                    ]
                                }
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert len([w for w in result["warnings"] if "途中で止まった" in w["message"]]) == 0


def test_line_growth_counter_and_box_reveal_are_checked_the_same_way():
    def check(layer):
        result = validate_project(
            {
                "video": {"width": 100, "height": 100, "duration": 5},
                "scenes": [{"id": "m", "duration": 1, "layers": [layer]}],
            }
        )
        return len([w for w in result["warnings"] if "途中で止まった" in w["message"]])

    assert (
        check(
            {
                "id": "a",
                "type": "linePath",
                "linePath": {"end": {"keyframes": [{"time": 0, "value": 0}, {"time": 3, "value": 1}]}},
            }
        )
        == 1
    )
    assert (
        check(
            {
                "id": "b",
                "type": "text",
                "counter": {"progress": {"keyframes": [{"time": 0, "value": 0}, {"time": 4, "value": 1}]}},
            }
        )
        == 1
    )
    assert (
        check(
            {
                "id": "c",
                "type": "text",
                "textBox": {
                    "reveal": {
                        "progress": {"keyframes": [{"time": 0, "value": 0}, {"time": 2, "value": 1}]}
                    }
                },
            }
        )
        == 1
    )
