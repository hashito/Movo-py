"""movo.expression — 安全な式エンジン。

    engine = ExpressionEngine(seed=12345)
    engine.evaluate("sin(time * 3) * 15", {"time": 1.2})

## サンドボックスであること

**`eval` / `exec` / `compile` は使いません。** 字句解析・構文解析・評価を
すべて自前で書いています（`lexer.py` / `parser.py` / `evaluator.py`）。
式から呼べるのは `functions.py` に並べた純粋な数学関数だけで、
ファイル・ネットワーク・プロセス・時刻には手が届きません。

`import`・`require`・`open` といった名前は «未定義の識別子» として弾かれます。
これはサンドボックスの副作用ではなく、**そもそも実行系を持っていない**からです。

## 速さ

式は 1 フレームに数百回評価されます。`compile()` が構文木を辞書にためるので、
2 回目以降は木をたどるだけです。木はタプルで持っており、
辞書引きより速く、うっかり書き換わることもありません。
"""

from __future__ import annotations

from ._compat import UNDEFINED, ErrorCodes, MovoError
from .evaluator import bind_props, evaluate, safe_get, to_number, truthy
from .functions import CONSTANTS, create_function_library
from .lexer import tokenize
from .parser import parse

__all__ = [
    "CONSTANTS",
    "ExpressionEngine",
    "UNDEFINED",
    "bind_props",
    "create_function_library",
    "evaluate",
    "parse",
    "safe_get",
    "to_number",
    "tokenize",
    "truthy",
]


class ExpressionEngine:
    """式をパースして評価する。パース結果は使い回す。"""

    def __init__(self, seed: int = 0, extra_functions: dict | None = None):
        self.functions = create_function_library(seed if seed is not None else 0)
        if extra_functions:
            self.functions.update(extra_functions)
        self._cache: dict[str, object] = {}

    def compile(self, source):
        """構文木を返す（キャッシュ付き）。構文が壊れていれば MOVO_EXPRESSION_INVALID。"""
        key = source if isinstance(source, str) else str(source)
        ast = self._cache.get(key)
        if ast is None:
            ast = parse(key)
            self._cache[key] = ast
        return ast

    def evaluate(self, source, scope: dict | None = None, path=None, file=None):
        """式を評価する。`path` / `file` はエラーの «どこで» に使う。"""
        try:
            return evaluate(self.compile(source), scope if scope is not None else {}, self.functions)
        except MovoError as err:
            raise MovoError(err.code, err.reason, path=path, file=file, cause=err) from err
        except Exception as err:  # 想定外も式のエラーとして包む
            raise MovoError(
                ErrorCodes.MOVO_EXPRESSION_INVALID,
                f'{err} (in "{source}")',
                path=path,
                file=file,
                cause=err,
            ) from err

    def evaluate_number(self, source, scope: dict | None = None, path=None, file=None) -> float:
        """評価して «有限の数» に寄せる。"""
        return to_number(self.evaluate(source, scope, path=path, file=file))

    def check(self, source) -> dict:
        """`movo validate` が使う構文チェック。例外を投げずに結果を返す。"""
        try:
            self.compile(source)
            return {"ok": True}
        except MovoError as err:
            return {"ok": False, "message": f"{err.code}: {err.reason}"}
        except Exception as err:  # pragma: no cover
            return {"ok": False, "message": str(err)}

    def list_functions(self) -> list[str]:
        """`movo list functions` の中身。"""
        return sorted(self.functions.keys())

    def list_constants(self) -> list[str]:
        return sorted(CONSTANTS.keys())
