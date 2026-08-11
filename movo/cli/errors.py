"""CLI が投げるエラーと、その終了コード。

**ここは «`movo.core.errors` が来るまでの仮置き» ではありません。**
`movo.core.errors` が用意できたらそちらを使い、無ければこの実装を使います
（下の `_adopt_core` を参照）。CLI は片方だけを見ればよいように、
インポート先をここ 1 か所にまとめています。

JS 版（`packages/core/src/errors.js`）と **コードの文字列を揃えてあります**。
終了コードで分岐しているスクリプトが JS 版と Python 版で違う値を返すと、
«動くけれど分岐が壊れる» という、いちばん気付きにくい形で壊れるためです。
"""

from __future__ import annotations

from typing import Any


class ErrorCodes:
    """JS 版と同じ文字列。並び順も揃えてあります。"""

    MOVO_CLI_USAGE = "MOVO_CLI_USAGE"
    MOVO_SCHEMA_INVALID = "MOVO_SCHEMA_INVALID"
    MOVO_ASSET_NOT_FOUND = "MOVO_ASSET_NOT_FOUND"
    MOVO_EXPRESSION_INVALID = "MOVO_EXPRESSION_INVALID"
    MOVO_PROVIDER_AUTH_FAILED = "MOVO_PROVIDER_AUTH_FAILED"
    MOVO_PLUGIN_NOT_FOUND = "MOVO_PLUGIN_NOT_FOUND"
    MOVO_PLUGIN_DENIED = "MOVO_PLUGIN_DENIED"
    MOVO_FFMPEG_NOT_FOUND = "MOVO_FFMPEG_NOT_FOUND"
    MOVO_RENDERER_UNAVAILABLE = "MOVO_RENDERER_UNAVAILABLE"
    MOVO_OUT_OF_MEMORY = "MOVO_OUT_OF_MEMORY"
    MOVO_INTERNAL = "MOVO_INTERNAL"


class MovoError(Exception):
    """人が読んで «次に何をすればよいか» が分かる形のエラー。

    `hint` を必ず書くようにしているのは、CLI のエラーが «原因» だけ言って
    «直し方» を言わないと、結局ドキュメントを探すことになるからです。
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        path: str | None = None,
        file: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.path = path
        self.file = file
        self.__cause__ = cause

    def format(self) -> str:
        lines = [f"{self.code}", f"  Reason: {self.message}"]
        if self.file:
            lines.insert(1, f"  File:   {self.file}")
        if self.path:
            lines.append(f"  Path:   {self.path}")
        if self.hint:
            lines.append(f"  Hint:   {self.hint}")
        return "\n".join(lines)


class MovoValidationError(MovoError):
    """検証で見つかった問題をまとめて運ぶ。"""

    def __init__(self, issues: list[dict[str, Any]], file: str | None = None) -> None:
        super().__init__(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f"{len(issues)} 件の問題が見つかりました",
            file=file,
        )
        self.issues = issues

    def format(self) -> str:
        lines = [self.code]
        if self.file:
            lines.append(f"  File:   {self.file}")
        for issue in self.issues:
            path = issue.get("path") or "(root)"
            lines.append(f"  {path}: {issue.get('message', '')}")
        return "\n".join(lines)


def to_movo_error(error: BaseException) -> MovoError:
    """素の例外を MovoError に包む。`format()` を必ず呼べるようにするため。"""
    if isinstance(error, MovoError):
        return error
    if isinstance(error, MemoryError):
        return MovoError(
            ErrorCodes.MOVO_OUT_OF_MEMORY,
            "メモリが足りません",
            hint="解像度を下げるか、--jobs の値を小さくしてください",
            cause=error,
        )
    if isinstance(error, FileNotFoundError):
        return MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND, str(error), cause=error
        )
    return MovoError(ErrorCodes.MOVO_INTERNAL, str(error) or type(error).__name__, cause=error)


# コマンド名 → プロセスの終了コード。JS 版（packages/cli/src/index.js）と同じ値です。
EXIT_CODES: dict[str, int] = {
    ErrorCodes.MOVO_CLI_USAGE: 2,
    ErrorCodes.MOVO_SCHEMA_INVALID: 3,
    ErrorCodes.MOVO_ASSET_NOT_FOUND: 4,
    ErrorCodes.MOVO_EXPRESSION_INVALID: 5,
    ErrorCodes.MOVO_PROVIDER_AUTH_FAILED: 6,
    ErrorCodes.MOVO_PLUGIN_NOT_FOUND: 7,
    ErrorCodes.MOVO_PLUGIN_DENIED: 7,
    ErrorCodes.MOVO_FFMPEG_NOT_FOUND: 8,
    ErrorCodes.MOVO_OUT_OF_MEMORY: 9,
}


def reason_of(error: BaseException) -> str:
    """エラーの «理由» を取り出す。

    `movo.core.errors.MovoError` は `reason`、この場の実装は `message` で
    持っています（JS 版がその両方の名前を使っていた名残です）。**どちらが
    採用されても呼ぶ側が書き分けずに済むよう**、取り出し方をここに集めます。
    """
    for attribute in ("reason", "message"):
        value = getattr(error, attribute, None)
        if isinstance(value, str) and value:
            return value
    return str(error)


def _adopt_core() -> None:
    """`movo.core.errors` が用意できていたら、そちらの定義に乗り換える。

    core は別の担当が移植中です。**先に揃ったほうを使う** ことで、どちらの
    順番で仕上がっても CLI が動きます。

    乗り換えは «同じ例外クラスにする» ためでもあります。core が投げた
    `MovoError` を CLI が別のクラスとして持っていると、`except MovoError` に
    引っかからず、**丁寧に書いたエラー文が «内部エラー» として潰れます**。
    """
    try:  # pragma: no cover - core が来たときだけ通る
        from movo.core import errors as core_errors  # type: ignore
    except Exception:
        return
    globals()["MovoError"] = getattr(core_errors, "MovoError", MovoError)
    globals()["MovoValidationError"] = getattr(core_errors, "MovoValidationError", MovoValidationError)
    globals()["ErrorCodes"] = getattr(core_errors, "ErrorCodes", ErrorCodes)
    core_convert = getattr(core_errors, "to_movo_error", None)
    if core_convert is not None:
        globals()["to_movo_error"] = core_convert


_adopt_core()
