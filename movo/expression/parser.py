"""小さな抽象構文木を作る Pratt パーサ。

**書けることをわざと絞っています。** 代入も文も関数定義もプロパティへの書き込みも
ありません。式はサンドボックスのスコープから «読む» ことしかできません。
新しい構文を足すときは、それが «読むだけ» で済むかを先に確かめてください。
"""

from __future__ import annotations

from ._compat import ErrorCodes, MovoError
from .lexer import TokenType, tokenize

BINARY_PRECEDENCE = {
    "??": 1,
    "||": 2,
    "&&": 3,
    "==": 4,
    "!=": 4,
    "===": 4,
    "!==": 4,
    "<": 5,
    ">": 5,
    "<=": 5,
    ">=": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "%": 7,
    "**": 8,
}

RIGHT_ASSOCIATIVE = {"**"}

# 木の節はタプルにします。辞書より軽く、評価器の分岐が速くなります。
# 形は ("kind", ...) で、kind ごとの中身は evaluator.py を参照。
KIND_LITERAL = "literal"
KIND_IDENTIFIER = "identifier"
KIND_ARRAY = "array"
KIND_UNARY = "unary"
KIND_BINARY = "binary"
KIND_CONDITIONAL = "conditional"
KIND_MEMBER = "member"
KIND_CALL = "call"


def parse(source: str):
    """式の文字列から木を作る。構文が壊れていれば MOVO_EXPRESSION_INVALID。"""
    tokens = tokenize(source)
    position = 0

    def peek():
        return tokens[position]

    def advance():
        nonlocal position
        token = tokens[position]
        position += 1
        return token

    def fail(message: str):
        token = peek()
        raise MovoError(
            ErrorCodes.MOVO_EXPRESSION_INVALID,
            f'{message} near "{token.value or "<end>"}" in "{source}"',
        )

    def expect(token_type: str, value=None):
        token = peek()
        if token.type != token_type or (value is not None and token.value != value):
            fail(f"expected {value if value is not None else token_type}")
        return advance()

    def parse_primary():
        token = peek()
        if token.type == TokenType.NUMBER:
            advance()
            return (KIND_LITERAL, float(token.value))
        if token.type == TokenType.STRING:
            advance()
            return (KIND_LITERAL, token.value)
        if token.type == TokenType.IDENTIFIER:
            advance()
            if token.value == "true":
                return (KIND_LITERAL, True)
            if token.value == "false":
                return (KIND_LITERAL, False)
            if token.value == "null":
                return (KIND_LITERAL, None)
            return (KIND_IDENTIFIER, token.value)
        if token.type == TokenType.OPERATOR and token.value in ("-", "+", "!"):
            advance()
            return (KIND_UNARY, token.value, parse_unary())
        if token.type == TokenType.PUNCTUATION and token.value == "(":
            advance()
            expr = parse_expression(0)
            expect(TokenType.PUNCTUATION, ")")
            return expr
        if token.type == TokenType.PUNCTUATION and token.value == "[":
            advance()
            elements = []
            while not (peek().type == TokenType.PUNCTUATION and peek().value == "]"):
                elements.append(parse_expression(0))
                if peek().type == TokenType.PUNCTUATION and peek().value == ",":
                    advance()
                else:
                    break
            expect(TokenType.PUNCTUATION, "]")
            return (KIND_ARRAY, tuple(elements))
        return fail("unexpected token")

    def parse_unary():
        return parse_accessors(parse_primary())

    def parse_accessors(node):
        current = node
        while True:
            token = peek()
            if token.type == TokenType.PUNCTUATION and token.value == ".":
                advance()
                name = expect(TokenType.IDENTIFIER).value
                current = (KIND_MEMBER, current, (KIND_LITERAL, name), False)
            elif token.type == TokenType.PUNCTUATION and token.value == "[":
                advance()
                prop = parse_expression(0)
                expect(TokenType.PUNCTUATION, "]")
                current = (KIND_MEMBER, current, prop, True)
            elif token.type == TokenType.PUNCTUATION and token.value == "(":
                advance()
                args = []
                while not (peek().type == TokenType.PUNCTUATION and peek().value == ")"):
                    args.append(parse_expression(0))
                    if peek().type == TokenType.PUNCTUATION and peek().value == ",":
                        advance()
                    else:
                        break
                expect(TokenType.PUNCTUATION, ")")
                current = (KIND_CALL, current, tuple(args))
            else:
                return current

    def parse_expression(min_precedence: int):
        left = parse_unary()
        while True:
            token = peek()
            if token.type == TokenType.OPERATOR and token.value == "?":
                if min_precedence > 0:
                    return left
                advance()
                consequent = parse_expression(0)
                expect(TokenType.OPERATOR, ":")
                alternate = parse_expression(0)
                left = (KIND_CONDITIONAL, left, consequent, alternate)
                continue
            if token.type != TokenType.OPERATOR:
                return left
            precedence = BINARY_PRECEDENCE.get(token.value)
            if precedence is None or precedence < min_precedence:
                return left
            advance()
            next_min = precedence if token.value in RIGHT_ASSOCIATIVE else precedence + 1
            right = parse_expression(next_min)
            left = (KIND_BINARY, token.value, left, right)

    ast = parse_expression(0)
    if peek().type != TokenType.END:
        fail("unexpected trailing token")
    return ast
