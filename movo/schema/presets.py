"""プリセット（エイリアス）の解決。

`project.presets` に「レイヤーに書く内容の断片」に名前を付けておき、
`layer.preset` から参照します。AviUtl のエイリアスに相当します。

  "presets": {
    "crtLook": { "effects": [{ "type": "lensDistortion", "strength": 0.3 }] },
    "popIn":   { "textAnimator": { "unit": "character", "stagger": 0.06 } }
  }
  "layers": [{ "type": "text", "preset": ["popIn", "crtLook"], "effects": [{ "type": "bloom" }] }]

マージ順は「プリセット（配列の順） → レイヤー自身」。
配列は連結、オブジェクトは再帰マージ、スカラーはレイヤー側が勝ちます。
`presetMerge: "replace"` を指定すると、配列も置き換えになります。

プリセットは他のプリセットを `extends` で取り込めます。循環したらエラーです。
"""

from __future__ import annotations

from movo.expression._compat import ErrorCodes, MovoError

#: プリセット自身のメタ情報。レイヤーには渡さない。
PRESET_META = frozenset({"extends", "description", "title", "presetMerge"})

#: レイヤーに書いても意味がないので、プリセットからは無視するキー。
NEVER_FROM_PRESET = frozenset({"id", "preset", "use"})


def _is_plain_object(value) -> bool:
    return isinstance(value, dict)


def _to_array(value):
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def resolve_presets(project):
    """プロジェクト全体のプリセット参照を解決する。

    戻り値は `{"applied": 適用したレイヤー数}`。
    """
    definitions = project.get("presets") if isinstance(project, dict) else None
    stats = {"applied": 0}
    if not isinstance(definitions, dict):
        return stats

    # extends を先に畳んでおく（同じプリセットを何度も辿らないため）
    flattened: dict[str, dict] = {}
    for name in list(definitions.keys()):
        flattened[name] = _flatten_preset(name, definitions, flattened, [])

    def walk_layers(layers, path):
        if not isinstance(layers, list):
            return layers
        return [
            _apply_presets(layer, flattened, f"{path}[{index}]", stats)
            for index, layer in enumerate(layers)
        ]

    for index, scene in enumerate(project.get("scenes") or []):
        if isinstance(scene, dict):
            scene["layers"] = walk_layers(scene.get("layers"), f"scenes[{index}].layers")
    if isinstance(project.get("layers"), list):
        project["layers"] = walk_layers(project["layers"], "layers")
    for name, composition in (project.get("compositions") or {}).items():
        if not isinstance(composition, dict):
            continue
        composition["layers"] = walk_layers(composition.get("layers"), f"compositions.{name}.layers")
        for index, scene in enumerate(composition.get("scenes") or []):
            if isinstance(scene, dict):
                scene["layers"] = walk_layers(
                    scene.get("layers"), f"compositions.{name}.scenes[{index}].layers"
                )
    return stats


def _flatten_preset(name, definitions, cache, stack):
    """`extends` を辿ってプリセット 1 件を平らにする。"""
    if name in cache:
        return cache[name]
    if name in stack:
        chain = " → ".join([*stack, name])
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f"プリセットが循環参照しています: {chain}",
            path=f"presets.{name}",
            hint="extends の連鎖をたどり直してください",
        )
    definition = definitions.get(name)
    if not isinstance(definition, dict):
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'プリセット "{name}" の定義がオブジェクトではありません',
            path=f"presets.{name}",
        )

    result: dict = {}
    for parent in _to_array(definition.get("extends")):
        if parent not in definitions:
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                f'プリセット "{name}" が未定義の "{parent}" を extends しています',
                path=f"presets.{name}.extends",
                hint=f"定義済み: {', '.join(definitions.keys())}",
            )
        result = _merge_preset(
            result,
            _flatten_preset(parent, definitions, cache, [*stack, name]),
            definition.get("presetMerge"),
        )
    result = _merge_preset(result, _strip(definition, PRESET_META), definition.get("presetMerge"))
    cache[name] = result
    return result


def _apply_presets(layer, presets, path, stats):
    """レイヤー 1 枚にプリセットを適用する（入れ子のレイヤーも辿る）。"""
    if not isinstance(layer, dict):
        return layer

    names = _to_array(layer.get("preset"))
    merged = layer

    if names:
        mode = layer.get("presetMerge") or "concat"
        base: dict = {}
        for name in names:
            if name not in presets:
                hint = (
                    f"定義済み: {', '.join(presets.keys())}"
                    if presets
                    else "project.presets に定義してください"
                )
                raise MovoError(
                    ErrorCodes.MOVO_SCHEMA_INVALID,
                    f'プリセット "{name}" は定義されていません',
                    path=f"{path}.preset",
                    hint=hint,
                )
            base = _merge_preset(base, presets[name], mode)
        # レイヤー自身の指定が最後に来る＝レイヤーが勝つ
        merged = _merge_preset(base, _strip(layer, {"preset", "presetMerge"}), mode)
        stats["applied"] += 1

    if isinstance(merged.get("layers"), list):
        merged = {
            **merged,
            "layers": [
                _apply_presets(child, presets, f"{path}.layers[{index}]", stats)
                for index, child in enumerate(merged["layers"])
            ],
        }
    return merged


def _merge_preset(under, over, mode):
    """2 つの断片を重ねる。後から来たほう（`over`）が勝つ。

    配列は既定で連結、`mode == "replace"` なら置き換え。
    """
    if over is None:
        return under
    if under is None:
        return over
    if isinstance(under, list) and isinstance(over, list):
        return list(over) if mode == "replace" else [*under, *over]
    if _is_plain_object(under) and _is_plain_object(over):
        out = dict(under)
        for key, value in over.items():
            out[key] = _merge_preset(under.get(key), value, mode)
        return out
    return over


def _strip(obj, keys):
    return {key: value for key, value in obj.items() if key not in keys}


def describe_presets(project):
    """`movo list presets` 用の要約。"""
    definitions = (project or {}).get("presets") or {}
    cache: dict[str, dict] = {}
    out = []
    for name in sorted(definitions.keys()):
        flat = _flatten_preset(name, definitions, cache, [])
        out.append(
            {
                "name": name,
                "title": definitions[name].get("title") or name,
                "description": definitions[name].get("description") or "",
                "extends": _to_array(definitions[name].get("extends")),
                "provides": [
                    f"{key}×{len(value)}" if isinstance(value, list) else key
                    for key, value in flat.items()
                ],
            }
        )
    return out


__all__ = ["NEVER_FROM_PRESET", "PRESET_META", "describe_presets", "resolve_presets"]
