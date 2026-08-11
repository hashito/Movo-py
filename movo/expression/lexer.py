"""式の字句解析（仕様 23 章）。

**Python の `eval` / `exec` / `compile` は使いません。** ここから評価器まで、
文字列を自前で読んで自前の木にします。`eval` を通した瞬間に、プロジェクト JSON に
`__import__("os").system(...)` と書けば何でもできることになり、
«式はファイル・ネットワーク・プロセスに触れない» という約束が消えます。
速さのためでも、この一線は越えません（JS 版も同じ理由で自前実装です）。
"""

from __future__ import annotations

import re
from typing import NamedTuple

from ._compat import ErrorCodes, MovoError


class TokenType:
    NUMBER = "number"
    STRING = "string"
    IDENTIFIER = "identifier"
    OPERATOR = "operator"
    PUNCTUATION = "punctuation"
    END = "end"


class Token(NamedTuple):
    type: str
    value: str
    start: int


# 長いものから並べる。`==` より先に `===` を見ないと切り出しを間違える。
OPERATORS = (
    "===",
    "!==",
    "**",
    "&&",
    "||",
    "??",
    "<=",
    ">=",
    "==",
    "!=",
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "!",
    "?",
    ":",
)

PUNCTUATION = ("(", ")", "[", "]", ",", ".")

_DIGIT = re.compile(r"[0-9]")
_IDENT_START = re.compile(r"[A-Za-z_$]")
_IDENT_PART = re.compile(r"[A-Za-z0-9_$]")


def tokenize(source: str) -> list[Token]:
    """式の文字列をトークンの列にする。"""
    tokens: list[Token] = []
    i = 0
    length = len(source)

    def fail(message: str, at: int):
        raise MovoError(
            ErrorCodes.MOVO_EXPRESSION_INVALID,
            f'{message} at position {at} in "{source}"',
        )

    def at(index: int) -> str:
        return source[index] if 0 <= index < length else ""

    while i < length:
        ch = source[i]
        if ch in " \t\n\r":
            i += 1
            continue
        # `#` と `//` から行末まではコメント
        if ch == "#" or (ch == "/" and at(i + 1) == "/"):
            while i < length and source[i] != "\n":
                i += 1
            continue
        if "0" <= ch <= "9":
            j = i
            while j < length and _DIGIT.match(source[j]):
                j += 1
            if at(j) == ".":
                j += 1
                while j < length and _DIGIT.match(source[j]):
                    j += 1
            if at(j) in ("e", "E"):
                k = j + 1
                if at(k) in ("+", "-"):
                    k += 1
                if _DIGIT.match(at(k) or " "):
                    k += 1
                    while k < length and _DIGIT.match(source[k]):
                        k += 1
                    j = k
            tokens.append(Token(TokenType.NUMBER, source[i:j], i))
            i = j
            continue
        if ch == "." and _DIGIT.match(at(i + 1) or " "):
            j = i + 1
            while j < length and _DIGIT.match(source[j]):
                j += 1
            tokens.append(Token(TokenType.NUMBER, source[i:j], i))
            i = j
            continue
        if ch in ('"', "'"):
            j = i + 1
            out = []
            while j < length and source[j] != ch:
                if source[j] == "\\":
                    j += 1
                    esc = at(j)
                    out.append("\n" if esc == "n" else "\t" if esc == "t" else esc)
                else:
                    out.append(source[j])
                j += 1
            if j >= length:
                fail("unterminated string", i)
            tokens.append(Token(TokenType.STRING, "".join(out), i))
            i = j + 1
            continue
        if _IDENT_START.match(ch):
            j = i
            while j < length and _IDENT_PART.match(source[j]):
                j += 1
            tokens.append(Token(TokenType.IDENTIFIER, source[i:j], i))
            i = j
            continue
        op = next((candidate for candidate in OPERATORS if source.startswith(candidate, i)), None)
        if op:
            tokens.append(Token(TokenType.OPERATOR, op, i))
            i += len(op)
            continue
        if ch in PUNCTUATION:
            tokens.append(Token(TokenType.PUNCTUATION, ch, i))
            i += 1
            continue
        fail(f'unexpected character "{ch}"', i)

    tokens.append(Token(TokenType.END, "", length))
    return tokens
