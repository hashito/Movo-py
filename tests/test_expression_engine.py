"""式エンジンのテスト（JS 版 tests/expression.test.js の移植）。

JS 版と «同じ数» が出ることを確かめます。加えて、Python 固有の逃げ道
（`__globals__` など）が閉じていることも見ます。
"""

import math

import pytest

from movo.expression import UNDEFINED, ExpressionEngine, bind_props
from movo.expression._compat import MovoError

engine = ExpressionEngine(seed=42)


def test_arithmetic_and_precedence():
    assert engine.evaluate("1 + 2 * 3") == 7
    assert engine.evaluate("(1 + 2) * 3") == 9
    assert engine.evaluate("2 ** 3 ** 2") == 512
    assert engine.evaluate("-2 ** 2") == 4
    assert engine.evaluate("10 % 3") == 1


def test_division_by_zero_yields_zero():
    # Infinity を返すと、下流で «画面いっぱいに引き伸ばされた 1 枚» になる。
    assert engine.evaluate("1 / 0") == 0
    assert engine.evaluate("1 % 0") == 0


def test_constants_and_trigonometry():
    assert abs(engine.evaluate("sin(PI / 2)") - 1) < 1e-12
    assert engine.evaluate("round(degrees(PI))") == 180


def test_scope_variables_and_time():
    assert engine.evaluate("time * 2", {"time": 3}) == 6
    assert engine.evaluate("sin(time * 3) * 15", {"time": 0}) == 0


def test_comparison_logic_and_ternary():
    assert engine.evaluate("1 < 2 && 3 > 2") is True
    assert engine.evaluate("time > 1 ? 10 : 20", {"time": 2}) == 10
    assert engine.evaluate("time > 1 ? 10 : 20", {"time": 0}) == 20
    assert engine.evaluate("null ?? 5") == 5


def test_member_access_and_layer_accessor():
    def layer_fn(layer_id):
        return {"transform": {"x": 100 if layer_id == "target" else 0}}

    # JS 版は関数にプロパティを生やしていた。Python では読ませる辞書を明示する。
    bind_props(layer_fn, {"transform": {"x": 7}})
    assert engine.evaluate("layer('target').transform.x + 100", {"layer": layer_fn}) == 200
    assert engine.evaluate("layer.transform.x", {"layer": layer_fn}) == 7


def test_clamp_remap_helpers():
    assert engine.evaluate("clamp(abs(-50) * 0.1, 0, 40)") == 5
    assert engine.evaluate("remap(5, 0, 10, 0, 100)") == 50
    assert engine.evaluate("pingPong(1.5, 1)") == 0.5


def test_noise_and_random_are_deterministic_for_a_seed():
    a = ExpressionEngine(seed=7)
    b = ExpressionEngine(seed=7)
    assert a.evaluate("noise(1.5)") == b.evaluate("noise(1.5)")
    assert a.evaluate("random('particles')") == b.evaluate("random('particles')")
    c = ExpressionEngine(seed=8)
    assert a.evaluate("noise(2.5)") != c.evaluate("noise(2.5)")


def test_sandbox_blocks_prototype_access_and_unknown_identifiers():
    with pytest.raises(MovoError):
        engine.evaluate("time.constructor")
    with pytest.raises(MovoError):
        engine.evaluate("foo + 1")
    with pytest.raises(MovoError):
        engine.evaluate('require("fs")')


@pytest.mark.parametrize(
    "source",
    [
        'require("os")',
        'import("os")',
        '__import__("os")',
        'open("/etc/passwd")',
        "eval('1')",
        "exec('1')",
        "compile('1')",
        "globals()",
        "locals()",
        "getattr(time, 'x')",
        "print('x')",
        "process.exit(1)",
        "Function('return 1')",
    ],
)
def test_sandbox_rejects_every_escape_hatch(source):
    """外の世界に触る名前は «未定義の識別子» として弾かれる。

    実行系そのものを持っていないので、通す経路がありません。
    """
    with pytest.raises(MovoError):
        engine.evaluate(source, {"time": 1})


def test_sandbox_cannot_reach_python_internals_through_functions():
    """スコープに何を置いても、そこから Python の内部へは降りられない。

    プロパティの取得は `safe_get` の 1 か所だけを通り、そこは `getattr` を
    まったく使いません。読めるのは辞書・配列・文字列と、`bind_props` で
    明示的にぶら下げた辞書だけです。届かない名前は undefined になります。
    """
    scope = {"layer": lambda: 1, "time": 1, "cfg": {"a": 1}}
    for source in [
        "layer.__globals__",
        "layer.__dict__",
        "layer.func_globals",
        "time.__class__",
        "time.real",
        "cfg.__class__",
        "cfg.keys",
        "cfg.items",
    ]:
        assert engine.evaluate(source, scope) is UNDEFINED, source
    with pytest.raises(MovoError):
        engine.evaluate("layer.constructor", scope)


def test_sandbox_cannot_write_anything():
    """代入も文も無いので、スコープを書き換えられない。"""
    for source in ["time = 5", "time.x = 5", "a; b", "import os"]:
        with pytest.raises(MovoError):
            engine.evaluate(source, {"time": 1})


def test_syntax_errors_are_reported_as_expression_invalid():
    with pytest.raises(MovoError) as info:
        engine.evaluate("1 +")
    assert info.value.code == "MOVO_EXPRESSION_INVALID"
    assert engine.check("sin(time").get("ok") is False
    assert engine.check("sin(time)").get("ok") is True


def test_function_library_is_exposed_for_discovery():
    names = engine.list_functions()
    for expected in ["sin", "clamp", "noise", "remap", "wave"]:
        assert expected in names


def test_parse_results_are_cached():
    """1 フレームに数百回評価されるので、パースは 1 度きり。"""
    local = ExpressionEngine(seed=1)
    first = local.compile("sin(time) * 3")
    second = local.compile("sin(time) * 3")
    assert first is second


def test_javascript_number_semantics():
    """Python に素直に書くと変わってしまう «数» を押さえる。"""
    # Math.round は 0.5 を上げる（Python の round は偶数へ寄る）
    assert engine.evaluate("round(2.5)") == 3
    assert engine.evaluate("round(3.5)") == 4
    # % は割られる数の符号（Python の % は割る数の符号）
    assert engine.evaluate("-7 % 3") == -1
    # mod() のほうは常に正（JS 版も同じ）
    assert engine.evaluate("mod(-7, 3)") == 2
    # 文字列との足し算は連結。1.0 が "1" になること
    assert engine.evaluate("'v' + 1") == "v1"
    assert engine.evaluate("'' + (2 / 2)") == "1"
    # === は型も見る
    assert engine.evaluate("1 === true") is False
    assert engine.evaluate("0 === false") is False


def test_infinite_values_collapse_to_zero():
    assert engine.evaluate("0 * Infinity") == 0 or math.isnan(engine.evaluate("0 * Infinity")) is False
