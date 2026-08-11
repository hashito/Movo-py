"""JS 版と «ビット単位で同じ数» が出ることを固定する。

ここに並ぶ数値は **JS 版を実際に走らせて写したもの** です。乱数とノイズは
32 ビットの折り返し（`>>> 0` と `Math.imul`）でできているので、Python に
素直に書き直すと静かにずれます。ずれると «同じ JSON から同じ動画» が崩れ、
しかも絵は出てしまうので気付けません。だから数で釘を刺しておきます。

値を更新するときは、必ず JS 版を走らせた出力を貼ってください。
"""

import math

from movo.animation import evaluate_modulator
from movo.expression import ExpressionEngine
from movo.expression._compat import (
    create_random,
    fbm_1d,
    hash_string,
    value_noise_1d,
    value_noise_2d,
)


def near(actual, expected, tolerance=1e-12):
    assert abs(actual - expected) <= tolerance * max(1.0, abs(expected)), f"{actual} != {expected}"


def test_hash_string_matches_js_including_non_bmp_characters():
    # JS の charCodeAt は UTF-16 のコード単位。絵文字（BMP 外）で
    # Python の ord() と食い違うので、そこも見ておく。
    assert [
        hash_string(s) for s in ["", "a", "particles", "ふつうの日本語", "🎵"]
    ] == [2166136261, 3826002220, 3740252708, 2883966805, 1143895498]


def test_mulberry32_produces_the_same_sequence():
    rng = create_random(7)
    got = [rng(), rng(), rng(), rng()]
    for a, b in zip(
        got,
        [0.011704753153026104, 0.06195825757458806, 0.97690763277933, 0.6990287057124078],
    ):
        near(a, b)


def test_value_noise_matches_js():
    near(value_noise_1d(1.5, 42), 0.2722607273608446)
    near(value_noise_1d(-3.25, 42), -0.7239261418717433)
    near(value_noise_2d(1.5, 2.5, 42), -0.1597878517350182)
    near(fbm_1d(1.5, 42, 4), -0.14187175659462808)


def test_expression_random_helpers_match_js():
    engine = ExpressionEngine(seed=42)
    near(engine.evaluate("noise(1.5)"), 0.2722607273608446)
    near(engine.evaluate("fbm(1.5, 6)"), -0.1676303183083378)
    near(engine.evaluate("random('particles')"), 0.34909343416802585)
    near(engine.evaluate("hashRandom('abc')"), 0.1026597695890814)
    near(engine.evaluate("wiggleAt(1.5, 2, 10)"), -1.342575988466186)
    near(engine.evaluate("beatAt(0.7, 174, 1)"), 0.9398180695167597)


def test_modulators_match_js():
    cases = [
        ({"type": "noise", "frequency": 3, "amplitude": 4}, 0.77, -2.4706920261383667),
        ({"type": "random-step", "frequency": 4}, 0.77, -0.48754127649590373),
        (
            {"type": "shake", "frequency": 8, "amplitude": 20, "decay": 3, "start": 0.1},
            0.5,
            0.16651802186900594,
        ),
        ({"type": "snap", "step": 15, "rate": 4}, 2.333, -15),
        ({"type": "beat", "bpm": 174, "division": 0.5, "decay": 6}, 0.77, 0.6175044737807127),
    ]
    for spec, time, expected in cases:
        near(evaluate_modulator(spec, time, {"seed": 5}), expected, 1e-12)


def test_scalar_semantics_match_js():
    """Python に素直に書くと変わってしまう «数» の一覧。"""
    engine = ExpressionEngine(seed=42)
    # Math.round は 0.5 を大きいほうへ（Python の round は偶数へ寄る）
    assert engine.evaluate("round(2.5)") == 3
    assert engine.evaluate("round(-2.5)") == -2
    # % の符号は «割られる数»
    assert engine.evaluate("-7 % 3") == -1
    assert engine.evaluate("7 % -3") == 1
    # 0 除算は Infinity ではなく 0
    assert engine.evaluate("1 / 0") == 0
    assert engine.evaluate("0 / 0") == 0
    # 定義域の外は既定値に落とす（例外にしない）
    assert engine.evaluate("sqrt(-4)") == 0
    assert engine.evaluate("asin(2)") == math.pi / 2
    near(engine.evaluate("cbrt(-27)"), -3)
    # 引数が無い min / max は ±Infinity
    assert engine.evaluate("min()") == math.inf
    assert engine.evaluate("max()") == -math.inf
    assert engine.evaluate("hypot()") == 0
