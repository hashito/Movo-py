"""1 つの JSON から 16:9 / 9:16 / 1:1 を出す。

同じ MV を YouTube と Shorts と TikTok に出すたびに、JSON を丸ごと複製して
`video.width` と数か所の座標を書き換える——という作業が発生していました。
3 本に増えた JSON は、あとから 1 行直すたびに 3 か所直すことになります。

そこで «違うところだけ» に名前を付けて、同じファイルに同居させます。

    "variants": {
      "shorts": {
        "video": { "width": 1080, "height": 1920 },
        "layers": {
          "title":   { "transform": { "y": "30%" } },
          "bg-wide": { "enabled": false }
        }
      }
    }

    movo render mv.json --variant shorts

`layers` だけは «id をキーにした部分上書き» という特別扱いです。レイヤーは
配列なので、素直に深いマージをすると «3 番目のレイヤー» のような順序依存の
指定になります。順番を入れ替えた瞬間に別のレイヤーが書き換わるのは事故の
もとなので、id で名指しさせます。

## いつ畳むか

`prepare_project`（継承のあと、params の展開の前）です。

- 継承より «あと»: 土台から受け継いだ値もバリアントで上書きできます。
- params より «前»: バリアントの中にも `${...}` を書けます。また params の
  式から見える `width` / `height` がバリアント後の解像度になります。

その代わり **スキル（`use`）が作るレイヤーには当たりません**。スキルの展開は
もっと後ろだからです。当たらなかった id は警告します（黙って効かないのが最悪）。
"""

from __future__ import annotations

import copy

from movo.expression._compat import ErrorCodes, MovoError


def _is_plain_object(value) -> bool:
    return isinstance(value, dict)


def list_variants(project):
    """宣言されているバリアント名の一覧。書いていなければ空リスト。"""
    variants = project.get("variants") if isinstance(project, dict) else None
    if not _is_plain_object(variants):
        return []
    return list(variants.keys())


def apply_variant(project, name, file=None, on_warn=None):
    """バリアントを 1 つ選んで畳む。元の JSON は変更しません。"""
    if name is None or name == "":
        return project
    declared = list_variants(project)
    variants = project.get("variants") if isinstance(project, dict) else None
    patch = variants.get(name) if _is_plain_object(variants) else None
    if not _is_plain_object(patch):
        hint = (
            f"宣言されているのは: {', '.join(declared)}"
            if declared
            else 'variants に { "shorts": { "video": { ... } } } の形で足してください'
        )
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'バリアント "{name}" がこのプロジェクトにありません',
            path="variants",
            file=file,
            hint=hint,
        )

    out = copy.deepcopy(project)
    # 畳んだあとは残さない。以降の段（検証・正規化・レンダラ）は variants を
    # 知らなくてよく、二重適用も起こりません。
    out.pop("variants", None)

    layer_patches = patch.get("layers")
    rest = {key: value for key, value in patch.items() if key != "layers"}
    _merge_into(out, rest)
    if _is_plain_object(layer_patches):
        for missing in _apply_layer_patches(out, layer_patches):
            if on_warn:
                on_warn(
                    f'バリアント "{name}" の layers."{missing}" に当たるレイヤーがありません'
                    "（スキルが作るレイヤーには当たりません）"
                )
    return out


def variant_names(project, base_name="base"):
    """`--all-variants` で回す名前の一覧。

    先頭は «素のまま»（`base`）です。3 アスペクト出したいときに欲しいのはたいてい
    «元も含めて全部» で、元だけ別のコマンドで出すのは面倒だからです。
    `variants` に `base` という名前があれば、そちらを尊重して重ねません。
    """
    names = list_variants(project)
    return names if base_name in names else [base_name, *names]


def expand_all_variants(project, file=None, on_warn=None, base_name="base"):
    """すべてのバリアントを畳んだものを順に返す。"""
    declared = set(list_variants(project))
    out = []
    for name in variant_names(project, base_name):
        if name not in declared:
            bare = copy.deepcopy(project)
            bare.pop("variants", None)
            out.append({"name": name, "project": bare})
        else:
            out.append({"name": name, "project": apply_variant(project, name, file, on_warn)})
    return out


def _apply_layer_patches(project, patches):
    """id で名指ししたレイヤーに部分上書きをかける。当たらなかった id を返す。"""
    remaining = set(patches.keys())

    def visit(layers):
        for layer in layers or []:
            if not _is_plain_object(layer):
                continue
            layer_id = layer.get("id")
            patch = patches.get(layer_id) if isinstance(layer_id, str) else None
            if _is_plain_object(patch):
                _merge_into(layer, patch)
                remaining.discard(layer_id)
            # 入れ子（group / composition の中身）も同じ id 空間として扱う。
            # レイヤー id はプロジェクト全体で一意（意味検証がそれを保証している）ので、
            # «どの階層にいるか» を書かずに名指しできます。
            if isinstance(layer.get("layers"), list):
                visit(layer["layers"])

    visit(project.get("layers"))
    for scene in project.get("scenes") or []:
        if _is_plain_object(scene):
            visit(scene.get("layers"))
    for composition in (project.get("compositions") or {}).values():
        if not _is_plain_object(composition):
            continue
        visit(composition.get("layers"))
        for scene in composition.get("scenes") or []:
            if _is_plain_object(scene):
                visit(scene.get("layers"))
    return list(remaining)


def _merge_into(target, patch):
    """深いマージ。オブジェクトどうしは畳み、配列とそれ以外は差し替えます。

    配列を «要素ごとに» 畳まないのは、レイヤーの並びと同じ理由です。添字で
    指すマージは、並びを変えた瞬間に別の場所へ当たります。
    """
    for key, value in (patch or {}).items():
        if _is_plain_object(value) and _is_plain_object(target.get(key)):
            _merge_into(target[key], value)
            continue
        target[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    return target


__all__ = ["apply_variant", "expand_all_variants", "list_variants", "variant_names"]
