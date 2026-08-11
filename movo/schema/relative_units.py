"""画面の大きさに «割合» で位置と大きさを書けるようにする。

作例 10 本はどれも絶対 px の直書きでした。1920x1080 で組んだものを
1080x1920 に入れると、レターボックスか切り落としにしかなりません。
«画面の真ん中» と書きたいのに «960» と書いているのが原因です。

    "transform": { "x": "50%", "y": "38%" }      画面の幅の 50% / 高さの 38%
    "style":     { "size": "7vh", "maxWidth": "80%" }

生の文字列は今まで数値化のときに NaN になり、既定値に落ちるだけでした。
つまり **この構文空間は空いていた** ので、後方互換を壊さずに意味を与えられます。

変換は «`"4bar"` を秒に直しているのと同じ正規化段» で済ませます。
レンダラ以降は px しか見ません。数値で書かれていたものは 1 文字も変わりません。

## 何を基準にするか

`%` は «そのレイヤーが載っているキャンバス» に対する割合です。画面か、
コンポジションの中ならそのコンポジションの大きさ。**親レイヤーの大きさは
使いません。** 親の実寸はテキストの行送りや画像の原寸で決まり、正規化の
時点では確定していないためです。ここで «たぶんこのくらい» を混ぜると、
同じ JSON から同じ動画が出るという約束（決定性）が崩れます。

`vw` / `vh` は常に画面（またはコンポジション）の幅・高さの 1/100 です。
"""

from __future__ import annotations

import re

from movo.expression._compat import is_finite_number, js_number, js_round, js_string

#: `"50%"` `"7vh"` `"1.5vmin"` を受ける。全体がこの形のときだけ相対単位とみなす。
PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(%|vw|vh|vmin|vmax)\s*\Z", re.IGNORECASE)

#: `%` を «幅» で解くキー。
X_KEYS = frozenset(
    {
        "x", "x1", "x2", "left", "right", "width", "maxwidth", "minwidth",
        "dx", "cx", "radiusx", "letterspacing", "indent", "columnwidth",
    }
)

#: `%` を «高さ» で解くキー。文字の大きさ（size）は高さ基準にする。
Y_KEYS = frozenset(
    {
        "y", "y1", "y2", "top", "bottom", "height", "maxheight", "minheight",
        "dy", "cy", "radiusy", "size", "fontsize", "lineheight", "baseline",
    }
)

#: すでに «0〜1 の割合» を意味しているキー。
#:
#: `anchorX` に `"50%"` と書きたくなりますが、あそこは最初から 0.5 と書く
#: 場所です。px に直すと 1 桁違う値が入るので、触らずに «解決できなかった»
#: として報告します。
RATIO_KEYS = frozenset(
    {
        "anchorx", "anchory", "centerx", "centery", "pivotx", "pivoty",
        "scale", "scalex", "scaley", "opacity", "progress", "amount", "ratio",
    }
)

#: 文字列を入れる場所。`"100%"` という «文» を数値に直してはいけない。
TEXT_KEYS = frozenset(
    {
        "text", "label", "title", "content", "format", "prefix", "suffix",
        "unit", "id", "name", "asset", "font", "fontfamily", "value2",
    }
)


def is_relative_unit(value) -> bool:
    """その文字列が相対単位の指定に見えるか。"""
    return isinstance(value, str) and PATTERN.match(value) is not None


def axis_of_key(key):
    """キーの名前から «どちらの軸か» を決める。None は «分からない»。"""
    if not isinstance(key, str) or key == "":
        return None
    lower = key.lower()
    if lower in TEXT_KEYS:
        return "text"
    if lower in RATIO_KEYS:
        return "ratio"
    if lower in X_KEYS:
        return "x"
    if lower in Y_KEYS:
        return "y"
    # `offsetX` `radiusY` のような «末尾が大文字の X / Y» はその軸として扱う。
    # 大文字で見るのは、`index` や `max` を軸と読み違えないためです。
    if key.endswith("X"):
        return "x"
    if key.endswith("Y"):
        return "y"
    if "width" in lower:
        return "x"
    if "height" in lower:
        return "y"
    return None


def _axis_of_property(prop):
    """`"transform.x"` のような property 指定から軸を読む。"""
    return axis_of_key(js_string(prop).split(".")[-1])


def relative_to_pixels(value, box, axis):
    """相対単位を px に直す。解決できなければ None。"""
    match = PATTERN.match(js_string(value))
    if not match:
        return None
    amount = js_number(match.group(1))
    unit = match.group(2).lower()
    width = js_number(box.get("width")) if isinstance(box, dict) else float("nan")
    height = js_number(box.get("height")) if isinstance(box, dict) else float("nan")
    if not (is_finite_number(amount) and is_finite_number(width) and is_finite_number(height)):
        return None

    if unit == "%":
        if axis == "x":
            basis = width
        elif axis == "y":
            basis = height
        else:
            # 軸が分からない `%` は解かない（黙って半分ずれるより良い）
            return None
    elif unit == "vw":
        basis = width
    elif unit == "vh":
        basis = height
    elif unit == "vmin":
        basis = min(width, height)
    elif unit == "vmax":
        basis = max(width, height)
    else:
        return None

    # 小数 4 桁で止める。同じ入力からは必ず同じ数になるので決定性は保たれます。
    return js_round((amount / 100) * basis * 1e4) / 1e4


def _for_each_tree(project, visit_tree):
    """相対単位が書かれていそうな枝だけを歩く。

    拍の解決（musical_time.py）と同じで、対象を «全部» にせず列挙します。
    `assets` や `params` の中の文字列まで触る理由がないからです。
    """
    video = project.get("video") if isinstance(project, dict) else None
    video = video if isinstance(video, dict) else {}
    screen_width = js_number(video.get("width"))
    screen_height = js_number(video.get("height"))
    screen = {
        "width": screen_width if is_finite_number(screen_width) and screen_width else 1920,
        "height": screen_height if is_finite_number(screen_height) and screen_height else 1080,
    }
    visit_tree(project.get("scenes"), screen, "scenes")
    visit_tree(project.get("layers"), screen, "layers")
    visit_tree(project.get("camera"), screen, "camera")
    visit_tree(project.get("presets"), screen, "presets")
    visit_tree(project.get("transitions"), screen, "transitions")
    compositions = project.get("compositions")
    if isinstance(compositions, dict):
        for name, composition in compositions.items():
            if not isinstance(composition, dict):
                continue
            # コンポジションは自前の画布を持てる。中の `%` はそちらを基準にする。
            width = js_number(composition.get("width"))
            height = js_number(composition.get("height"))
            box = {
                "width": width if is_finite_number(width) and width else screen["width"],
                "height": height if is_finite_number(height) and height else screen["height"],
            }
            visit_tree(composition.get("layers"), box, f"compositions.{name}.layers")
            visit_tree(composition.get("scenes"), box, f"compositions.{name}.scenes")


def _walk(node, box, key, hint, visit, path):
    """木を歩いて、文字列に出会うたび `visit` を呼ぶ。

    `visit` が値を返したらその場で置き換えます（None なら据え置き）。
    """
    if isinstance(node, list):
        for index, child in enumerate(node):
            child_path = f"{path}[{index}]"
            if isinstance(child, str):
                nxt = visit(child, key, hint, box, child_path)
                if nxt is not None:
                    node[index] = nxt
            else:
                _walk(child, box, key, hint, visit, child_path)
        return
    if not isinstance(node, dict):
        return
    # `{ "property": "transform.x", "keyframes": [{ "value": "50%" }] }` のように、
    # 軸が «その場のキー» ではなく property 側に書いてあることがあります。
    prop = node.get("property")
    local_hint = _axis_of_property(prop) if isinstance(prop, str) else hint
    for child_key, value in list(node.items()):
        child_path = f"{path}.{child_key}" if path else child_key
        if isinstance(value, str):
            nxt = visit(value, child_key, local_hint, box, child_path)
            if nxt is not None:
                node[child_key] = nxt
            continue
        _walk(value, box, child_key, local_hint, visit, child_path)


def resolve_relative_units(project):
    """プロジェクト全体の相対単位を px に直す（破壊的）。

    相対単位を 1 つも書いていない JSON は 1 文字も変わりません。
    """
    if not isinstance(project, dict):
        return project

    def visit(value, key, hint, box, path):
        if not is_relative_unit(value):
            return None
        axis = axis_of_key(key)
        if axis is None:
            axis = hint
        if axis in ("text", "ratio"):
            return None
        return relative_to_pixels(value, box, axis if axis in ("x", "y") else None)

    _for_each_tree(project, lambda tree, box, label: _walk(tree, box, "", None, visit, label))
    return project


def find_unresolved_relative_units(project):
    """直せなかった相対単位を集める（意味検証の警告に使う）。

    `"50%"` を «軸の分からないキー» に書くと、今までどおり NaN になって既定値に
    落ちます。黙って落ちるのがいちばん困るので、書いた場所を伝えます。
    """
    if not isinstance(project, dict):
        return []
    found: list[dict] = []

    def visit(value, key, hint, box, path):
        if not is_relative_unit(value):
            return None
        axis = axis_of_key(key)
        if axis is None:
            axis = hint
        if axis == "text":
            return None
        found.append(
            {
                "path": path,
                "value": value,
                "reason": (
                    f'"{key}" は 0〜1 の割合を書く場所なので相対単位は使えません'
                    if axis == "ratio"
                    else f'"{key}" は幅と高さのどちらを基準にするか決められません'
                ),
            }
        )
        return None

    _for_each_tree(project, lambda tree, box, label: _walk(tree, box, "", None, visit, label))
    return found


__all__ = [
    "PATTERN",
    "RATIO_KEYS",
    "TEXT_KEYS",
    "X_KEYS",
    "Y_KEYS",
    "axis_of_key",
    "find_unresolved_relative_units",
    "is_relative_unit",
    "relative_to_pixels",
    "resolve_relative_units",
]
