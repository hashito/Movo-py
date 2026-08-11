"""`${...}` のテンプレート展開と入力値の検証への «繋ぎ»。

実体は `movo/skill/template.py`（スキル側）にあります。ここはその薄い包みです。

**なぜ包むのか。** JS 版の `packages/schema/src/params.js` は
`packages/skill/src/template.js` をそのまま呼んでいます。テンプレート展開を
2 つ持つと «スキルでは書けるのに params では書けない» という食い違いが必ず出るので、
実装は 1 つに保ちます。ここが担うのは 2 つだけです。

  1. 呼び出し形の違いを吸収する（skill 側は `options` 辞書と «タプル» を使う）
  2. **読み込みを遅らせる。** skill は `movo.cli` の橋渡しを経由するので、
     モジュールの読み込み時に掴むと schema → skill → cli → schema の輪ができます。
     関数の中で読むことで、輪になる前にほどけます。
"""

from __future__ import annotations


def expand_template(node, scope, engine=None, path="", file=None):
    """テンプレート（JSON から読んだ任意の値）の `${...}` を展開する。"""
    from movo.skill.template import expand_template as _expand

    return _expand(node, scope, {"engine": engine, "path": path or "", "file": file})


def resolve_inputs(definitions, given=None, name=None, file=None, scale=1):
    """入力定義に沿って値を検証・既定値補完する。

    skill 側はタプル `(values, issues)` を返すので、ここで辞書に直します。
    """
    from movo.skill.template import resolve_inputs as _resolve

    values, issues = _resolve(definitions, given or {}, {"name": name, "file": file, "scale": scale})
    return {"values": values, "issues": issues}


def parse_input_assignments(assignments):
    """`--set key=value` を辞書にする。"""
    from movo.skill import parse_input_assignments as _parse

    return _parse(assignments)


__all__ = ["expand_template", "parse_input_assignments", "resolve_inputs"]
