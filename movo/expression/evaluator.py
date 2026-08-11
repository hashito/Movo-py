"""木をたどって値を出す。

## サンドボックスの要点

プロパティの取得は `safe_get` だけを通します。触れるのは **辞書・配列・文字列**、
それに «スコープに置いた関数へ明示的にぶら下げた辞書» だけです。

Python では `getattr` を無制限に許すと、関数 1 つから
`f.__globals__["__builtins__"]["__import__"]("os")` まで一直線です。
そのため **`getattr` は一切使いません**。JS 版では関数のプロパティを読めましたが、
そこは `bind_props()` で «読ませてよい辞書» を明示する形に置き換えています。
"""

from __future__ import annotations

import math

from ._compat import (
    UNDEFINED,
    ErrorCodes,
    MovoError,
    is_finite_number,
    is_nullish,
    js_mod,
    js_number,
    js_string,
    js_truthy,
)
from .functions import CONSTANTS
from .parser import (
    KIND_ARRAY,
    KIND_BINARY,
    KIND_CALL,
    KIND_CONDITIONAL,
    KIND_IDENTIFIER,
    KIND_LITERAL,
    KIND_MEMBER,
    KIND_UNARY,
)

# JS 版と同じ «たどらせない» 名前。Python 側の危ない名前は safe_get の作りで
# そもそも届きません（`getattr` を使わないため）。
BLOCKED_KEYS = frozenset({"__proto__", "prototype", "constructor"})

#: `bind_props` が関数にぶら下げる辞書の名前。
PROPS_ATTRIBUTE = "movo_props"


def bind_props(function, props: dict):
    """スコープに置く関数へ «式から読んでよい» プロパティを付ける。

    JS 版では `layer` が関数でありながら `layer.transform.x` も読めました
    （`Object.assign(layerFn, {...})`）。Python で同じことを `getattr` で
    実現するとサンドボックスに穴が開くので、読ませる辞書を明示させます。
    """
    setattr(function, PROPS_ATTRIBUTE, props)
    return function


def to_number(value) -> float:
    """式の値を «有限の数» に寄せる。NaN と Infinity は 0 になる。"""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else 0.0
    if isinstance(value, str):
        n = js_number(value)
        return n if math.isfinite(n) else 0.0
    if isinstance(value, (list, tuple)):
        return to_number(value[0]) if len(value) else 0.0
    return 0.0


def truthy(value) -> bool:
    """式の真偽判定。JS の `Boolean` と同じ（空配列は真）。"""
    return js_truthy(value)


def js_strict_equals(a, b) -> bool:
    """JS の `===`。`1 === true` は偽、`NaN === NaN` も偽。

    Python の `==` は `1 == True` を真にするので、そのままでは使えません。
    """
    a_bool = isinstance(a, bool)
    b_bool = isinstance(b, bool)
    if a_bool or b_bool:
        return a_bool and b_bool and a is b
    a_num = isinstance(a, (int, float))
    b_num = isinstance(b, (int, float))
    if a_num and b_num:
        return a == b
    if a_num != b_num:
        return False
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if a is None or b is None:
        return a is None and b is None
    if a is UNDEFINED or b is UNDEFINED:
        return a is UNDEFINED and b is UNDEFINED
    return a is b


def safe_get(obj, key):
    """プロパティを 1 段たどる。たどれないものは `undefined`。"""
    if is_nullish(obj):
        return UNDEFINED
    name = js_string(key)
    if name in BLOCKED_KEYS:
        raise MovoError(
            ErrorCodes.MOVO_EXPRESSION_INVALID,
            f'access to "{name}" is not allowed in expressions',
        )
    if isinstance(obj, str):
        if name == "length":
            return len(obj)
        index = js_number(name)
        if math.isfinite(index) and index == int(index) and 0 <= int(index) < len(obj):
            return obj[int(index)]
        return UNDEFINED
    if isinstance(obj, (list, tuple)):
        if name == "length":
            return len(obj)
        index = js_number(name)
        if math.isfinite(index) and index == int(index) and 0 <= int(index) < len(obj):
            return obj[int(index)]
        return UNDEFINED
    if isinstance(obj, dict):
        return obj[name] if name in obj else UNDEFINED
    if callable(obj):
        # スコープに置いた関数（`layer` など）に明示的にぶら下げた辞書だけ読む。
        props = obj.__dict__.get(PROPS_ATTRIBUTE) if hasattr(obj, "__dict__") else None
        if isinstance(props, dict) and name in props:
            return props[name]
        return UNDEFINED
    return UNDEFINED


def evaluate(node, scope, functions):
    """木を評価する。`scope` は辞書、`functions` は名前 → 呼べるもの。"""
    # ホットパス。1 フレームに数百回来るので、辞書引きより先に «よく来る節» を並べる。
    kind = node[0]

    if kind == KIND_LITERAL:
        return node[1]

    if kind == KIND_IDENTIFIER:
        name = node[1]
        if name in scope:
            return scope[name]
        if name in CONSTANTS:
            return CONSTANTS[name]
        fn = functions.get(name)
        if fn is not None:
            return fn
        raise MovoError(ErrorCodes.MOVO_EXPRESSION_INVALID, f'unknown identifier "{name}"')

    if kind == KIND_ARRAY:
        return [evaluate(element, scope, functions) for element in node[1]]

    if kind == KIND_UNARY:
        operator = node[1]
        value = evaluate(node[2], scope, functions)
        if operator == "-":
            return -to_number(value)
        if operator == "+":
            return to_number(value)
        return not truthy(value)

    if kind == KIND_BINARY:
        op = node[1]
        # 短絡する 3 つは、右辺を評価する前に判断する。
        if op == "&&":
            left = evaluate(node[2], scope, functions)
            return evaluate(node[3], scope, functions) if truthy(left) else False
        if op == "||":
            left = evaluate(node[2], scope, functions)
            return left if truthy(left) else evaluate(node[3], scope, functions)
        if op == "??":
            left = evaluate(node[2], scope, functions)
            return evaluate(node[3], scope, functions) if is_nullish(left) else left

        a = evaluate(node[2], scope, functions)
        b = evaluate(node[3], scope, functions)
        if op == "+":
            if isinstance(a, str) or isinstance(b, str):
                return js_string(a) + js_string(b)
            return to_number(a) + to_number(b)
        if op == "-":
            return to_number(a) - to_number(b)
        if op == "*":
            return to_number(a) * to_number(b)
        if op == "/":
            divisor = to_number(b)
            # 0 除算は Infinity ではなく 0。絵が消し飛ぶより «動かない» ほうがまし。
            return 0.0 if divisor == 0 else to_number(a) / divisor
        if op == "%":
            m = to_number(b)
            if m == 0:
                return 0.0
            result = js_mod(to_number(a), m)
            return 0.0 if math.isnan(result) else result
        if op == "**":
            try:
                return float(to_number(a) ** to_number(b))
            except (OverflowError, ZeroDivisionError, ValueError):
                return float("inf")
        if op == "<":
            return to_number(a) < to_number(b)
        if op == ">":
            return to_number(a) > to_number(b)
        if op == "<=":
            return to_number(a) <= to_number(b)
        if op == ">=":
            return to_number(a) >= to_number(b)
        if op in ("==", "==="):
            return js_strict_equals(a, b)
        if op in ("!=", "!=="):
            return not js_strict_equals(a, b)
        raise MovoError(ErrorCodes.MOVO_EXPRESSION_INVALID, f'unsupported operator "{op}"')

    if kind == KIND_CONDITIONAL:
        if truthy(evaluate(node[1], scope, functions)):
            return evaluate(node[2], scope, functions)
        return evaluate(node[3], scope, functions)

    if kind == KIND_MEMBER:
        obj = evaluate(node[1], scope, functions)
        key = evaluate(node[2], scope, functions) if node[3] else node[2][1]
        return safe_get(obj, key)

    if kind == KIND_CALL:
        callee_node = node[1]
        args = [evaluate(arg, scope, functions) for arg in node[2]]
        if callee_node[0] == KIND_IDENTIFIER:
            name = callee_node[1]
            callee = scope[name] if name in scope else functions.get(name)
            if not callable(callee):
                raise MovoError(ErrorCodes.MOVO_EXPRESSION_INVALID, f'"{name}" is not a function')
            return callee(*args)
        callee = evaluate(callee_node, scope, functions)
        if not callable(callee):
            raise MovoError(
                ErrorCodes.MOVO_EXPRESSION_INVALID, "attempted to call a non-function value"
            )
        return callee(*args)

    raise MovoError(ErrorCodes.MOVO_EXPRESSION_INVALID, f'unknown AST node "{kind}"')


__all__ = [
    "BLOCKED_KEYS",
    "bind_props",
    "evaluate",
    "is_finite_number",
    "js_strict_equals",
    "safe_get",
    "to_number",
    "truthy",
]
