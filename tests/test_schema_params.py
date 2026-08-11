"""params（差し替え可能な入力値）と extends（継承）。

JS 版 tests/params-recipe.test.js のうち、schema パッケージが持っている部分を
移植したものです（CLI とレシピの入出力は cli 側の担当）。
"""

import json

import pytest

from movo.expression._compat import MovoError
from movo.schema import (
    apply_extends,
    expand_params,
    list_params,
    merge_deep,
    normalize_project,
    prepare_project,
    resolve_params,
    strip_json_comments,
    validate_project,
)


def template():
    return {
        "movoVersion": "1.0",
        "params": {
            "art": {"type": "asset", "default": "assets/singer.png"},
            "title": {"type": "text", "default": "入れ子の街"},
            "bpm": {"type": "number", "default": 205, "min": 40, "max": 300},
            "showLines": {"type": "boolean", "default": True},
        },
        "project": {"name": "${title}", "bpm": "${bpm}"},
        "video": {"width": 1920, "height": 1080, "fps": 30, "duration": 4},
        "assets": {"singer": "${art}"},
        "scenes": [
            {
                "id": "m",
                "duration": 4,
                "layers": [
                    {"id": "a", "type": "image", "asset": "singer"},
                    {"id": "t", "type": "text", "text": "${title}", "transform": {"y": "${height * 0.4}"}},
                ],
            }
        ],
    }


def test_defaults_are_filled_in_and_the_declaration_disappears():
    project = resolve_params(template())
    assert "params" not in project
    assert project["project"]["name"] == "入れ子の街"
    # 全体が 1 個の式なら型を保つ（"205" ではなく 205）
    assert project["project"]["bpm"] == 205
    assert project["assets"]["singer"] == "assets/singer.png"


def test_ambient_values_are_visible_to_the_expressions():
    project = resolve_params(template())
    assert project["scenes"][0]["layers"][1]["transform"]["y"] == 1080 * 0.4


def test_set_overrides_the_default():
    project = resolve_params(template(), set_values=["title=別の街", "bpm=174"])
    assert project["project"]["name"] == "別の街"
    assert project["project"]["bpm"] == 174


def test_values_include_every_declared_item():
    result = expand_params(template(), set_values=["bpm=174"])
    assert result["values"]["bpm"] == 174
    assert result["values"]["title"] == "入れ子の街"
    assert result["values"]["showLines"] is True


def test_a_number_outside_the_range_is_rejected():
    with pytest.raises(MovoError) as info:
        resolve_params(template(), set_values=["bpm=1000"])
    assert info.value.code == "MOVO_SCHEMA_INVALID"
    assert "300" in info.value.reason


def test_an_unknown_key_is_rejected():
    with pytest.raises(MovoError) as info:
        resolve_params(template(), set_values=["nope=1"])
    assert "nope" in info.value.reason


def test_a_project_without_params_is_untouched():
    raw = {"video": {"width": 10, "height": 10}, "scenes": []}
    assert resolve_params(raw) is raw


def test_passing_values_to_a_project_without_params_is_an_error():
    with pytest.raises(MovoError) as info:
        resolve_params({"video": {"width": 10, "height": 10}}, set_values=["a=1"])
    assert "params" in info.value.reason


def test_list_params_describes_the_declarations():
    listed = list_params(template())
    by_key = {entry["key"]: entry for entry in listed}
    assert by_key["bpm"]["type"] == "number"
    assert by_key["bpm"]["min"] == 40
    assert by_key["art"]["default"] == "assets/singer.png"
    assert by_key["title"]["required"] is False


def test_normalize_expands_params_too():
    project = normalize_project(template())
    assert project["project"]["bpm"] == 205
    assert "params" not in project


def test_validate_sees_the_expanded_project():
    result = validate_project(template())
    assert result["valid"] is True, result["issues"]


def test_set_must_be_key_equals_value():
    with pytest.raises(MovoError) as info:
        resolve_params(template(), set_values=["title"])
    assert info.value.code == "MOVO_CLI_USAGE"


# ── 継承 ────────────────────────────────────────────────────


def test_extends_merges_deeply_and_the_child_wins(tmp_path):
    base = tmp_path / "_base.json"
    base.write_text(
        json.dumps(
            {
                "video": {"width": 1920, "height": 1080, "fps": 30},
                "project": {"name": "土台", "seed": 1},
                "presets": {"a": {"style": {"size": 10}}},
            }
        ),
        encoding="utf-8",
    )
    child = tmp_path / "mv.json"
    project = apply_extends(
        {"extends": "_base.json", "project": {"name": "01"}, "scenes": []}, file=str(child)
    )
    assert project["video"]["width"] == 1920
    assert project["project"]["name"] == "01", "自分自身がいちばん強い"
    assert project["project"]["seed"] == 1, "書いていないキーは土台から来る"
    assert "extends" not in project


def test_extends_replaces_arrays_rather_than_concatenating(tmp_path):
    base = tmp_path / "_base.json"
    base.write_text(json.dumps({"video": {"width": 1, "height": 1}, "scenes": [{"id": "x"}]}), encoding="utf-8")
    project = apply_extends(
        {"extends": "_base.json", "scenes": [{"id": "y"}]}, file=str(tmp_path / "mv.json")
    )
    assert [s["id"] for s in project["scenes"]] == ["y"]


def test_extends_accepts_comments_in_the_base_file(tmp_path):
    base = tmp_path / "_base.json"
    base.write_text(
        '{\n  // 土台の解像度\n  "video": { "width": 640, "height": 360 } /* ここまで */\n}',
        encoding="utf-8",
    )
    project = apply_extends({"extends": "_base.json"}, file=str(tmp_path / "mv.json"))
    assert project["video"]["width"] == 640


def test_a_circular_extends_says_how_it_looped(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"extends": "b.json"}), encoding="utf-8")
    b.write_text(json.dumps({"extends": "a.json"}), encoding="utf-8")
    with pytest.raises(MovoError) as info:
        apply_extends({"extends": "a.json"}, file=str(tmp_path / "mv.json"))
    assert "循環" in info.value.reason


def test_a_missing_base_file_is_reported(tmp_path):
    with pytest.raises(MovoError) as info:
        apply_extends({"extends": "nope.json"}, file=str(tmp_path / "mv.json"))
    assert info.value.code == "MOVO_ASSET_NOT_FOUND"


def test_json_without_extends_is_returned_as_is():
    raw = {"video": {"width": 1, "height": 1}}
    assert apply_extends(raw) is raw


def test_merge_deep_and_strip_json_comments():
    assert merge_deep({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
    # 配列は置き換え
    assert merge_deep({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}
    # 文字列の中の // は落とさない
    assert json.loads(strip_json_comments('{"url": "http://x" /* c */ }')) == {"url": "http://x"}


# ── 継承 → バリアント → params の順 ──────────────────────────


def test_prepare_project_applies_extends_then_variant_then_params(tmp_path):
    """順番が結果を変えるところ。バリアント後の解像度が params の式に見える。"""
    base = tmp_path / "_base.json"
    base.write_text(
        json.dumps({"video": {"width": 1920, "height": 1080, "fps": 30}}), encoding="utf-8"
    )
    raw = {
        "extends": "_base.json",
        "params": {"y": {"type": "number", "default": 0}},
        "variants": {"shorts": {"video": {"width": 1080, "height": 1920}}},
        "scenes": [{"id": "m", "layers": [{"id": "a", "type": "text", "transform": {"y": "${height / 2}"}}]}],
    }
    project = prepare_project(raw, file=str(tmp_path / "mv.json"), variant="shorts")
    assert project["video"]["width"] == 1080
    # バリアントを畳んだ «あと» の高さが式から見える
    assert project["scenes"][0]["layers"][0]["transform"]["y"] == 960
    assert "variants" not in project
    assert "params" not in project
