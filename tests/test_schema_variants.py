"""相対単位・セーフエリア・アスペクト比バリアント（JS 版 tests/variants.test.js の移植）。

いちばん大事なのは «既存の JSON が 1 文字も変わらないこと» なので、
「書いていなければ何も起きない」を各段で確かめています。
CLI に関わる 4 件は cli パッケージ側の担当なので外してあります。
"""

import json
import re

import pytest

from movo.expression._compat import MovoError
from movo.schema import (
    apply_variant,
    axis_of_key,
    expand_all_variants,
    find_unresolved_relative_units,
    is_relative_unit,
    list_variants,
    normalize_project,
    relative_to_pixels,
    resolve_relative_units,
    validate_project,
    variant_names,
)


def base():
    return {
        "movoVersion": "1.0",
        "project": {"name": "variants-test", "seed": 7},
        "video": {"width": 1920, "height": 1080, "fps": 30, "duration": 2},
        "scenes": [
            {
                "id": "main",
                "start": 0,
                "duration": 2,
                "layers": [
                    {
                        "id": "title",
                        "type": "text",
                        "text": "こんにちは",
                        "transform": {"x": "50%", "y": "38%"},
                        "style": {"size": "7vh"},
                    },
                    {
                        "id": "bg-wide",
                        "type": "shape",
                        "shape": {"type": "rectangle", "width": "80%", "height": 100},
                        "transform": {"x": "50%", "y": "50%"},
                    },
                ],
            }
        ],
    }


# ── 相対単位 ────────────────────────────────────────────────


def test_percent_uses_width_for_x_keys_and_height_for_y_keys():
    project = resolve_relative_units(base())
    title = project["scenes"][0]["layers"][0]
    assert title["transform"]["x"] == 960  # 1920 の 50%
    assert title["transform"]["y"] == 410.4  # 1080 の 38%


def test_viewport_units_are_one_hundredth_of_the_screen():
    box = {"width": 1920, "height": 1080}
    assert relative_to_pixels("7vh", box, None) == 75.6
    assert relative_to_pixels("10vw", box, None) == 192
    assert relative_to_pixels("10vmin", box, None) == 108
    assert relative_to_pixels("10vmax", box, None) == 192


def test_style_size_is_height_based():
    project = resolve_relative_units(base())
    assert project["scenes"][0]["layers"][0]["style"]["size"] == 75.6
    assert axis_of_key("size") == "y"
    assert axis_of_key("maxWidth") == "x"
    assert axis_of_key("offsetY") == "y"


def test_keyframe_values_read_the_axis_from_property():
    project = base()
    project["scenes"][0]["layers"][0]["animations"] = [
        {
            "property": "transform.x",
            "keyframes": [{"time": 0, "value": "10%"}, {"time": 1, "value": "90%"}],
        }
    ]
    resolve_relative_units(project)
    keyframes = project["scenes"][0]["layers"][0]["animations"][0]["keyframes"]
    assert keyframes[0]["value"] == 192
    assert keyframes[1]["value"] == 1728


def test_compositions_use_their_own_canvas_as_the_basis():
    project = base()
    project["compositions"] = {
        "card": {
            "width": 400,
            "height": 200,
            "layers": [{"id": "inner", "type": "shape", "transform": {"x": "50%", "y": "50%"}}],
        }
    }
    resolve_relative_units(project)
    assert project["compositions"]["card"]["layers"][0]["transform"]["x"] == 200
    assert project["compositions"]["card"]["layers"][0]["transform"]["y"] == 100


def test_numeric_json_is_left_byte_for_byte():
    project = {
        "movoVersion": "1.0",
        "video": {"width": 960, "height": 540, "duration": 1},
        "scenes": [
            {"id": "main", "duration": 1, "layers": [{"id": "a", "type": "shape", "transform": {"x": 480, "y": 270}}]}
        ],
    }
    before = json.dumps(project, ensure_ascii=False)
    resolve_relative_units(project)
    assert json.dumps(project, ensure_ascii=False) == before


def test_a_percent_inside_prose_is_not_converted():
    project = base()
    project["scenes"][0]["layers"][0]["text"] = "100%"
    resolve_relative_units(project)
    assert project["scenes"][0]["layers"][0]["text"] == "100%"
    assert len(find_unresolved_relative_units(project)) == 0


def test_a_percent_without_an_axis_is_warned_not_converted():
    project = base()
    project["scenes"][0]["layers"][0]["style"]["blurriness"] = "50%"
    result = validate_project(project)
    assert result["valid"] is True, result["issues"]
    warning = next((w for w in result["warnings"] if "50%" in w["message"]), None)
    assert warning, result["warnings"]
    assert re.search("幅と高さのどちら", warning["message"])


def test_relative_units_in_a_ratio_slot_are_warned():
    project = base()
    project["scenes"][0]["layers"][0]["transform"]["anchorX"] = "50%"
    result = validate_project(project)
    assert project["scenes"][0]["layers"][0]["transform"]["anchorX"] == "50%"
    warning = next((w for w in result["warnings"] if "anchorX" in w["path"]), None)
    assert warning, result["warnings"]
    assert re.search("0〜1 の割合", warning["message"])


def test_is_relative_unit_only_matches_the_whole_string():
    assert is_relative_unit("50%") is True
    assert is_relative_unit(" 7vh ") is True
    assert is_relative_unit("達成率 100%") is False
    assert is_relative_unit("4bar") is False
    assert is_relative_unit(50) is False


def test_normalize_also_turns_relative_units_into_pixels():
    project = normalize_project(base())
    assert project["scenes"][0]["layers"][0]["transform"]["x"] == 960
    assert project["scenes"][0]["layers"][1]["shape"]["width"] == 1536


# ── セーフエリア ────────────────────────────────────────────


def test_no_safe_area_means_no_warning():
    project = base()
    project["scenes"][0]["layers"].append(
        {
            "id": "edge",
            "type": "shape",
            "shape": {"type": "rectangle", "width": 200, "height": 100},
            "transform": {"x": 20, "y": 540},
        }
    )
    result = validate_project(project)
    assert len([w for w in result["warnings"] if "セーフエリア" in w["message"]]) == 0


def test_layers_outside_the_safe_area_are_warnings_not_errors():
    project = base()
    project["video"]["safeArea"] = {"x": 0.05, "y": 0.05}
    project["scenes"][0]["layers"].append(
        {
            "id": "edge",
            "type": "shape",
            "shape": {"type": "rectangle", "width": 200, "height": 100},
            "transform": {"x": 20, "y": 540},
        }
    )
    result = validate_project(project)
    assert result["valid"] is True, result["issues"]
    warning = next((w for w in result["warnings"] if '"edge"' in w["message"]), None)
    assert warning, result["warnings"]
    assert re.search("はみ出しています", warning["message"])


def test_a_full_screen_background_is_not_treated_as_overflowing():
    project = base()
    project["video"]["safeArea"] = {"x": 0.05, "y": 0.05}
    project["scenes"][0]["layers"].append(
        {
            "id": "bg",
            "type": "shape",
            "shape": {"type": "rectangle", "width": "100%", "height": "100%"},
            "transform": {"x": 0, "y": 0, "anchorX": 0, "anchorY": 0},
        }
    )
    result = validate_project(project)
    assert len([w for w in result["warnings"] if '"bg"' in w["message"]]) == 0


def test_layer_safe_area_false_opts_out():
    project = base()
    project["video"]["safeArea"] = {"x": 0.05, "y": 0.05}
    project["scenes"][0]["layers"].append(
        {
            "id": "stream",
            "type": "shape",
            "safeArea": False,
            "shape": {"type": "rectangle", "width": 200, "height": 100},
            "transform": {"x": 20, "y": 540},
        }
    )
    result = validate_project(project)
    assert len([w for w in result["warnings"] if '"stream"' in w["message"]]) == 0


# ── バリアント ──────────────────────────────────────────────


def with_variants():
    project = base()
    project["variants"] = {
        "shorts": {
            "video": {"width": 1080, "height": 1920},
            "layers": {"title": {"transform": {"y": "30%"}}, "bg-wide": {"enabled": False}},
        }
    }
    return project


def test_list_variants_and_variant_names_put_base_first():
    assert list_variants(with_variants()) == ["shorts"]
    assert variant_names(with_variants()) == ["base", "shorts"]
    assert list_variants(base()) == []


def test_variants_merge_deeply_and_layers_are_patched_by_id():
    source = with_variants()
    applied = apply_variant(source, "shorts")
    assert applied["video"]["width"] == 1080
    assert applied["video"]["height"] == 1920
    # 書き換えていないキーは残る
    assert applied["video"]["fps"] == 30
    assert applied["scenes"][0]["layers"][0]["transform"]["y"] == "30%"
    # 同じ transform の x は消えない（深いマージ）
    assert applied["scenes"][0]["layers"][0]["transform"]["x"] == "50%"
    assert applied["scenes"][0]["layers"][1]["enabled"] is False
    # 畳んだあとに variants は残さない
    assert "variants" not in applied
    # 元の JSON は触らない
    assert source["video"]["width"] == 1920
    assert source["scenes"][0]["layers"][0]["transform"]["y"] == "38%"


def test_percentages_resolve_against_the_variant_resolution():
    project = normalize_project(apply_variant(with_variants(), "shorts"))
    assert project["scenes"][0]["layers"][0]["transform"]["x"] == 540  # 1080 の 50%
    assert project["scenes"][0]["layers"][0]["transform"]["y"] == 576  # 1920 の 30%
    assert project["scenes"][0]["layers"][0]["style"]["size"] == 134.4  # 1920 の 7%


def test_an_unknown_variant_says_what_exists():
    with pytest.raises(MovoError) as info:
        apply_variant(with_variants(), "square")
    assert "square" in info.value.reason
    assert re.search("shorts", info.value.hint or "")


def test_layer_ids_that_never_match_are_warned():
    project = with_variants()
    project["variants"]["shorts"]["layers"]["nope"] = {"enabled": False}
    warnings = []
    apply_variant(project, "shorts", on_warn=warnings.append)
    assert len(warnings) == 1
    assert re.search("nope", warnings[0])


def test_no_variant_name_leaves_the_project_alone():
    source = with_variants()
    assert apply_variant(source, None) is source
    assert apply_variant(source, "") is source


def test_expand_all_variants_puts_the_bare_project_first():
    everything = expand_all_variants(with_variants())
    assert [entry["name"] for entry in everything] == ["base", "shorts"]
    assert everything[0]["project"]["video"]["width"] == 1920
    assert "variants" not in everything[0]["project"]
    assert everything[1]["project"]["video"]["width"] == 1080


def test_json_declaring_variants_still_passes_structural_validation():
    result = validate_project(with_variants())
    assert result["valid"] is True, result["issues"]
