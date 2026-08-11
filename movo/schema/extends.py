"""プロジェクトの継承（extends）。

シリーズものを作ると、`video` / `output` / `presets` / `camera` は 10 本とも
同じで、違うのは中身だけ、ということがよくあります。共通ブロックを 10 か所に
書き写すと、あとで解像度を変えたくなったときに «直し漏れた 1 本» が必ず出ます。

    { "extends": "../_base.json", "project": { "name": "01" }, "scenes": [ ... ] }

決めごとはこの 4 つです。

  1. 深いマージ。オブジェクトは «キーごとに» 重ねる。
  2. 配列は «置き換え»。連結にすると、土台のレイヤーを消せなくなるため。
  3. `extends` は配列も可。左から順に重ねるので、右にあるほうが強い。
     いちばん強いのは «自分自身» です。
  4. 相対パスは «その JSON からの» 相対。土台が更に extends していれば、
     その土台のある場所を基準に辿ります。

循環参照はここで «どう回ったか» まで出して止めます。10 本の JSON が
互いを継承し始めると、スタックを溢れさせてから気付くのでは遅いためです。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from movo.expression._compat import ErrorCodes, MovoError

#: 継承の深さの上限。ここまで辿って終わらないなら書き方が間違っている。
MAX_DEPTH = 16


def _is_plain_object(value) -> bool:
    return isinstance(value, dict)


def _to_array(value):
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def apply_extends(project, file=None, base_dir=None):
    """トップレベルの `extends` を畳んで «継承済みの 1 枚の JSON» を返す。

    `extends` を書いていない JSON は複製もせずそのまま返します（従来どおり）。
    """
    if not _is_plain_object(project) or "extends" not in project:
        return project
    resolved_file = Path(file).resolve() if file else None
    if base_dir:
        directory = Path(base_dir).resolve()
    elif resolved_file:
        directory = resolved_file.parent
    else:
        directory = Path(os.getcwd())
    return _resolve_node(project, directory, [resolved_file] if resolved_file else [], "extends")


def _resolve_node(node, base_dir: Path, stack, at):
    """継承の連鎖を «土台 → 自分» の順に畳む。"""
    parents = _to_array(node.get("extends"))
    own = {key: value for key, value in node.items() if key != "extends"}
    if not parents:
        return own
    if len(stack) > MAX_DEPTH:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f"extends が {MAX_DEPTH} 段を超えました",
            path=at,
            hint="継承をたどり直してください（循環しているかもしれません）",
        )

    merged: dict = {}
    for index, reference in enumerate(parents):
        if not isinstance(reference, str) or reference.strip() == "":
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                "extends にはファイルのパスを書いてください",
                path=f"{at}[{index}]",
                hint='例: "extends": "../_base.json"',
            )
        target = (base_dir / reference).resolve()
        if target in stack:
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                f"extends が循環参照しています: {_describe_cycle([*stack, target])}",
                path=f"{at}[{index}]",
                hint="どこかで自分自身に戻っています。継承の向きを一方通行にしてください",
            )
        parent = _resolve_node(
            _read_jsonc(target, f"{at}[{index}]"),
            target.parent,
            [*stack, target],
            f"{at}[{index}]",
        )
        # 同じ列に並ぶ土台どうしは «右が強い»
        merged = merge_deep(merged, parent)
    # 自分自身は最後に重ねる＝いちばん強い
    return merge_deep(merged, own)


def merge_deep(under, over):
    """2 つの JSON を深く重ねる。後から来たほう（`over`）が勝つ。

    配列は置き換え（連結ではない）。
    """
    if over is None:
        return under
    if _is_plain_object(under) and _is_plain_object(over):
        out = dict(under)
        for key, value in over.items():
            out[key] = merge_deep(under.get(key), value)
        return out
    # 配列も «そのまま置き換え»。土台の要素を消したいときに連結だと消せない。
    return over


def _describe_cycle(files) -> str:
    """循環をそのまま並べる（どこで折り返したかが読めるように短い名前で出す）。"""
    return " → ".join(Path(f).name for f in files)


def _read_jsonc(file: Path, at):
    """コメント付きの JSON を読む。

    プロジェクト JSON は `//` を書ける約束なので、土台の JSON にも当然コメントが
    入っています。素の `json.loads` では «土台にコメントを書いた瞬間に壊れる» ので、
    ここでも同じ書式を受けます。
    """
    if not file.exists():
        raise MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND,
            f"extends の継承元が見つかりません: {file}",
            path=at,
            hint="パスは «その JSON からの» 相対です",
        )
    text = file.read_text(encoding="utf-8")
    try:
        return json.loads(strip_json_comments(text))
    except ValueError as err:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f"{file.name} を JSON として読めません: {err}",
            path=at,
            file=str(file),
        ) from err


def strip_json_comments(text: str) -> str:
    """`//` と `/* */` を落とす（文字列の中は触らない）。"""
    out = []
    in_string = False
    in_line = False
    in_block = False
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < length else ""
        if in_line:
            if ch == "\n":
                in_line = False
                out.append(ch)
            i += 1
            continue
        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                i += 1
            i += 1
            continue
        if in_string:
            out.append(ch)
            if ch == "\\":
                out.append(nxt)
                i += 1
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


__all__ = ["MAX_DEPTH", "apply_extends", "merge_deep", "strip_json_comments"]
