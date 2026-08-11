"""Movo のエラーコードと構造化されたエラー。

利用者に見せる失敗はすべて **機械可読なコード** を持ちます。加えて、原因になった
ファイルと JSON ポインタを添えます（仕様 33 節）。

**エラーコードの文字列は JS 版と 1 文字も変えていません。** ログを機械で読む
仕組みや、CI の突き合わせが JS 版と Python 版で共通に使えるようにするためです。
"""

from __future__ import annotations

from typing import Any


class ErrorCodes:
    """エラーコードの定数。

    Python なら `enum.Enum` が自然ですが、**あえて素の文字列**にしています。
    JSON へ書き出すとき・ログに出すときに `.value` を書き忘れて
    `ErrorCodes.MOVO_INTERNAL` のような表示になるのを避けるためです。
    """

    MOVO_SCHEMA_INVALID = "MOVO_SCHEMA_INVALID"
    MOVO_ASSET_NOT_FOUND = "MOVO_ASSET_NOT_FOUND"
    MOVO_ASSET_DECODE_FAILED = "MOVO_ASSET_DECODE_FAILED"
    MOVO_PROVIDER_AUTH_FAILED = "MOVO_PROVIDER_AUTH_FAILED"
    MOVO_PROVIDER_FAILED = "MOVO_PROVIDER_FAILED"
    MOVO_PLUGIN_NOT_FOUND = "MOVO_PLUGIN_NOT_FOUND"
    MOVO_PLUGIN_DENIED = "MOVO_PLUGIN_DENIED"
    MOVO_PLUGIN_INVALID = "MOVO_PLUGIN_INVALID"
    MOVO_RENDERER_UNAVAILABLE = "MOVO_RENDERER_UNAVAILABLE"
    MOVO_PHYSICS_UNSTABLE = "MOVO_PHYSICS_UNSTABLE"
    MOVO_EXPRESSION_INVALID = "MOVO_EXPRESSION_INVALID"
    MOVO_FFMPEG_NOT_FOUND = "MOVO_FFMPEG_NOT_FOUND"
    MOVO_OUT_OF_MEMORY = "MOVO_OUT_OF_MEMORY"
    MOVO_NETWORK_DENIED = "MOVO_NETWORK_DENIED"
    MOVO_DOWNLOAD_TOO_LARGE = "MOVO_DOWNLOAD_TOO_LARGE"
    MOVO_FONT_NOT_FOUND = "MOVO_FONT_NOT_FOUND"
    MOVO_UNSUPPORTED = "MOVO_UNSUPPORTED"
    MOVO_CLI_USAGE = "MOVO_CLI_USAGE"
    MOVO_INTERNAL = "MOVO_INTERNAL"


class MovoError(Exception):
    """Movo が投げる唯一の例外の基底。

    :param code: :class:`ErrorCodes` のいずれか
    :param reason: 人が読む説明
    :param file: 原因になったファイル
    :param path: 原因になった JSON ポインタ（``layers.0.text``）
    :param hint: どう直せばよいか
    :param cause: 元になった例外
    """

    def __init__(
        self,
        code: str,
        reason: str,
        *,
        file: str | None = None,
        path: str | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason
        self.file = file
        self.path = path
        self.hint = hint
        if cause is not None:
            self.__cause__ = cause

    def format(self) -> str:
        """仕様どおりのブロック表示にする。JS 版の ``format()`` と同じ並びです。"""
        lines: list[str] = [self.code, ""]
        if self.file:
            lines += ["File:", self.file, ""]
        if self.path:
            lines += ["Path:", self.path, ""]
        lines += ["Reason:", self.reason]
        if self.hint:
            lines += ["", "Hint:", self.hint]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """JSON へ落とす。**キー名は JS 版のまま**（互換のため snake_case にしません）。"""
        return {
            "code": self.code,
            "reason": self.reason,
            "file": self.file,
            "path": self.path,
            "hint": self.hint,
        }


class MovoValidationError(MovoError):
    """検証の指摘をまとめて 1 つの例外にする。

    1 件ずつ投げると «直しては再実行» を何度も繰り返すことになるので、
    まとめて出します。
    """

    def __init__(self, issues: list[dict[str, str]], file: str | None = None) -> None:
        first = issues[0] if issues else None
        super().__init__(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            first["message"] if first else "invalid project",
            file=file,
            path=first.get("path") if first else None,
        )
        self.issues = issues

    def format(self) -> str:
        out = []
        for issue in self.issues:
            out.append(
                "\n".join(
                    [
                        self.code,
                        "",
                        "File:",
                        self.file or "(inline)",
                        "",
                        "Path:",
                        issue.get("path") or "(root)",
                        "",
                        "Reason:",
                        issue.get("message", ""),
                    ]
                )
            )
        return "\n\n---\n\n".join(out)


def to_movo_error(err: BaseException | str, code: str = ErrorCodes.MOVO_INTERNAL) -> MovoError:
    """何が飛んできても :class:`MovoError` に包む。

    メモリ不足だけは専用のコードに振り替えます。**利用者に見せる文面が
    «内部エラー» のままだと、解像度を下げれば通ると気付けない**からです。
    """
    if isinstance(err, MovoError):
        return err
    message = str(err)
    if isinstance(err, MemoryError) or "out of memory" in message.lower():
        return MovoError(
            ErrorCodes.MOVO_OUT_OF_MEMORY,
            message or "out of memory",
            cause=err if isinstance(err, BaseException) else None,
        )
    return MovoError(code, message, cause=err if isinstance(err, BaseException) else None)
