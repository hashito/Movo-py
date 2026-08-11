"""キーフレーム・モジュレーター・値の解決（JS 版 tests/animation.test.js の移植）。"""

import math
import re

from movo.animation import (
    EASINGS,
    apply_animations,
    combine_values,
    evaluate_modulator,
    get_easing,
    get_path,
    resolve_animated,
    sample_keyframes,
)
from movo.expression import ExpressionEngine

engine = ExpressionEngine(seed=1)


def ctx(time):
    return {"time": time, "engine": engine, "scope": {"time": time}, "seed": 1}


def test_keyframes_interpolate_linearly_and_hold_outside_the_range():
    keyframes = [{"time": 0, "value": 0}, {"time": 2, "value": 800}]
    assert sample_keyframes(keyframes, -1) == 0
    assert sample_keyframes(keyframes, 0) == 0
    assert sample_keyframes(keyframes, 1) == 400
    assert sample_keyframes(keyframes, 2) == 800
    assert sample_keyframes(keyframes, 5) == 800


def test_easing_applies_to_the_keyframe_being_approached():
    keyframes = [{"time": 0, "value": 0}, {"time": 1, "value": 100, "easing": "easeInQuad"}]
    assert sample_keyframes(keyframes, 0.5) == 25


def test_keyframes_interpolate_arrays_and_colours():
    arrays = sample_keyframes([{"time": 0, "value": [0, 10]}, {"time": 1, "value": [10, 20]}], 0.5)
    assert arrays == [5, 15]
    color = sample_keyframes([{"time": 0, "value": "#000000"}, {"time": 1, "value": "#ffffff"}], 0.5)
    assert re.match(r"^rgba\(128, 128, 128", color)
    # 透明度は JS の String() と同じ書き方（"1.0" ではなく "1"）
    assert color == "rgba(128, 128, 128, 1)"


def test_extrapolate_loop_wraps_time():
    keyframes = [{"time": 0, "value": 0}, {"time": 1, "value": 10}]
    assert sample_keyframes(keyframes, 2.5, extrapolate="loop") == 5


def test_sine_modulator_matches_offset_plus_amplitude_times_sin():
    spec = {"type": "sine", "frequency": 1.5, "amplitude": 30, "offset": 500}
    assert round(evaluate_modulator(spec, 0)) == 500
    quarter = 1 / (1.5 * 4)
    assert round(evaluate_modulator(spec, quarter)) == 530


def test_every_modulator_type_returns_a_finite_number():
    for kind in [
        "sine",
        "cosine",
        "triangle",
        "square",
        "sawtooth",
        "pulse",
        "noise",
        "random-step",
    ]:
        value = evaluate_modulator(
            {"type": kind, "frequency": 2, "amplitude": 1}, 0.37, {"seed": 5}
        )
        assert math.isfinite(value), f"{kind} produced {value}"
    curve = evaluate_modulator(
        {"type": "custom-curve", "curve": [0, 1, 0], "frequency": 1, "amplitude": 2}, 0.25
    )
    assert math.isfinite(curve)
    reactive = evaluate_modulator(
        {"type": "audio-reactive", "amplitude": 10}, 0, {"audio": {"level": 0.5, "bands": [0.2]}}
    )
    assert reactive == 5


def test_shake_beat_and_snap_stay_finite():
    for kind in ["shake", "beat", "snap"]:
        value = evaluate_modulator({"type": kind, "frequency": 2, "amplitude": 1}, 0.37, {"seed": 5})
        assert math.isfinite(value), f"{kind} produced {value}"


def test_combine_modes():
    assert combine_values([1, 2, 3], "add", 10) == 16
    assert combine_values([2, 3], "multiply", 2) == 12
    assert combine_values([1, 9], "average") == 5
    assert combine_values([1, 9], "replace") == 9
    assert combine_values([1, 9], "max", 4) == 9


def test_resolve_animated_handles_the_four_value_forms():
    assert resolve_animated(500, ctx(0)) == 500
    assert resolve_animated({"keyframes": [{"time": 0, "value": 1}, {"time": 1, "value": 3}]}, ctx(0.5)) == 2
    assert resolve_animated({"expression": "time * 10"}, ctx(2)) == 20
    assert (
        round(
            resolve_animated(
                {"modulator": {"type": "sine", "frequency": 1, "amplitude": 15, "offset": 0}}, ctx(0.25)
            )
        )
        == 15
    )


def test_modulators_without_a_base_use_the_modulator_value_directly():
    value = resolve_animated(
        {"modulator": {"type": "sine", "frequency": 1.5, "amplitude": 30, "offset": 500}},
        ctx(0),
        12345,  # 無視されるべき既定値
    )
    assert round(value) == 500


def test_multiple_modulators_combine():
    value = resolve_animated(
        {
            "modulators": [
                {"type": "sine", "frequency": 1, "amplitude": 10},
                {"type": "noise", "frequency": 8, "amplitude": 2},
            ],
            "combine": "add",
        },
        ctx(0.25),
    )
    assert math.isfinite(value)


def test_nested_records_resolve_recursively():
    resolved = resolve_animated({"center": {"x": {"expression": "time"}, "y": 0.5}}, ctx(0.25))
    assert resolved == {"center": {"x": 0.25, "y": 0.5}}


def test_apply_animations_writes_dotted_property_paths():
    target = {"transform": {"x": 0, "y": 0}, "modifiers": {"flagWave": {"amplitude": 0}}}
    apply_animations(
        target,
        [
            {"property": "transform.x", "keyframes": [{"time": 0, "value": 0}, {"time": 1, "value": 100}]},
            {"property": "modifiers.flagWave.amplitude", "expression": "clamp(time * 40, 0, 40)"},
        ],
        ctx(0.5),
    )
    assert get_path(target, "transform.x") == 50
    assert get_path(target, "modifiers.flagWave.amplitude") == 20


def test_a_plain_value_animation_writes_the_value_itself():
    """`{"property": ..., "value": 9}` は 9 を書く（`{"value": 9}` ではない）。

    JS 版は spec をオブジェクトリテラルで組むので `keyframes` などのキーが
    «値は undefined でも存在する» 状態になり、必ず «動く値» と判定されます。
    Python で «無いキーは入れない» と書くと «ただの入れ物» と誤判定され、
    辞書がまるごと書き込まれてしまいました。
    """
    target = {}
    apply_animations(target, [{"property": "deep.nested.value", "value": 9}], ctx(0.5))
    assert target == {"deep": {"nested": {"value": 9}}}


def test_disabled_animations_are_skipped():
    target = {"transform": {"y": 0}}
    apply_animations(
        target, [{"property": "transform.y", "expression": "3", "enabled": False}], ctx(0.5)
    )
    assert target["transform"]["y"] == 0


def test_relative_animations_add_to_the_existing_value():
    target = {"transform": {"rotation": 10}}
    apply_animations(
        target, [{"property": "transform.rotation", "relative": True, "expression": "5"}], ctx(0)
    )
    assert target["transform"]["rotation"] == 15


def test_easing_lookup_falls_back_safely():
    assert get_easing("easeInOut") is EASINGS["easeInOut"]
    assert get_easing("does-not-exist") is EASINGS["linear"]
    bezier = get_easing([0.25, 0.1, 0.25, 1])
    assert 0 < bezier(0.5) < 1
    assert bezier(0) == 0
    assert bezier(1) == 1


def test_easing_aliases_and_cubic_bezier_strings():
    assert get_easing("ease-in") is EASINGS["easeIn"]
    # 区切り文字を落とした名前でも引ける（大文字小文字はそのまま）
    assert get_easing("ease In Out") is EASINGS["easeInOut"]
    assert get_easing("ease_in_out") is EASINGS["linear"]
    fn = get_easing("cubic-bezier(0.25, 0.1, 0.25, 1)")
    assert 0 < fn(0.5) < 1
