"""キャッシュ鍵とロックファイルのためのハッシュ。

**JS 版と同じ鍵が出ること**が要点です。同じプロジェクトを JS 版と Python 版で
レンダリングしたとき、キャッシュを共有できるようにしてあります。そのため
JSON の文字列化は Python の ``json.dumps`` ではなく **JS の ``JSON.stringify``
に合わせた自前実装**にしています（``json.dumps`` は ``1.0`` を ``"1.0"`` と書き、
非 ASCII を ``\\uXXXX`` に開くので、そのままでは鍵がずれます）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def short_hash(data: bytes | str, length: int = 16) -> str:
    return sha256(data)[:length]


def hash_file(file_path: str | os.PathLike[str]) -> str:
    """ファイルの中身のハッシュ。**まとめ読みせず 1 MB ずつ**流します。

    素材には数百 MB の動画が混ざるので、全部をメモリに載せると素材が
    増えたときに落ちます。
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _js_number(value: float | int) -> str:
    """JS の ``String(number)`` と同じ書き方にする。

    JS には整数と小数の区別がないので、``1.0`` は ``"1"`` と書かれます。
    ここを合わせないと ``{"fps": 30}`` と ``{"fps": 30.0}`` で鍵が変わります。
    """
    if isinstance(value, bool):  # bool は int の子なので先に弾く
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "null"  # JSON.stringify(NaN) は null
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))
    return repr(value)


def stable_stringify(value: Any) -> str:
    """キーを並べ替えてから JSON にする。

    辞書の «書いた順» は意味を持たないので、順番が違うだけで別の鍵になるのは
    困ります。並べ替えてから文字列にすることで、意味が同じ設定からは
    必ず同じ鍵が出ます。
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _js_number(value)
    if isinstance(value, str):
        # ensure_ascii=False は必須です。JS は非 ASCII をそのまま書きます。
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(stable_stringify(v) for v in value) + "]"
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value.keys())
        body = ",".join(f"{stable_stringify(k)}:{stable_stringify(value[k])}" for k in keys)
        return "{" + body + "}"
    # 想定外の型は文字列として扱う（JS 側でも toString されるため）
    return stable_stringify(str(value))


def hash_json(value: Any) -> str:
    """JSON にできる値のハッシュ。キーの順番に依らず同じ値になります。"""
    return sha256(stable_stringify(value))
