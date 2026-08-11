"""動く値の解決 — 仕様 7 章の中心。

プロジェクト JSON のどの値も、次のどれか（またはその組み合わせ）で書けます。

  1. 定数              42            "#ff0000"      [1, 2]
  2. キーフレーム      {"keyframes": [...]}
  3. 式                {"expression": "sin(time)"}
  4. モジュレーター    {"modulator": {...}}
                       {"modulators": [...], "combine": "add"}

どれでもない辞書は «中身を 1 つずつ» 解決します。`center: {x, y}` のような
入れ子の値も、成分ごとに動かせるようにするためです。
"""

from __future__ import annotations

import math

from movo.expression import to_number
from movo.expression._compat import UNDEFINED, clamp, is_nullish, js_round

from .keyframes import sample_keyframes
from .modulators import combine_values, evaluate_modulator

ANIMATED_KEYS = ("keyframes", "expression", "modulator", "modulators")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_animated_spec(value) -> bool:
    return isinstance(value, dict) and any(k in value for k in ANIMATED_KEYS)


def local_time_for(spec, time):
    """`delay` / `timeScale` / `loop` を当てて «その指定の中の時刻» を出す。"""
    t = time
    if _is_number(spec.get("delay")):
        t -= spec["delay"]
    if _is_number(spec.get("timeScale")) and spec["timeScale"] != 0:
        t *= spec["timeScale"]
    if _is_number(spec.get("timeOffset")):
        t += spec["timeOffset"]
    if spec.get("loop"):
        loop = spec.get("loop")
        length = loop if _is_number(loop) else (spec.get("loopDuration") or 0)
        if length > 0:
            t = math.fmod(math.fmod(t, length) + length, length)
    return t


def resolve_animated(spec, ctx, fallback=UNDEFINED):
    """値 1 つを «その時刻の値» にする。"""
    if is_nullish(spec):
        return fallback
    if isinstance(spec, (int, float, str, bool)):
        return spec
    if isinstance(spec, (list, tuple)):
        base_path = ctx.get("path") or ""
        return [
            resolve_animated(item, {**ctx, "path": f"{base_path}[{index}]"}, UNDEFINED)
            for index, item in enumerate(spec)
        ]
    if not isinstance(spec, dict):
        return fallback

    if not is_animated_spec(spec):
        # ただの入れ物。中身を解決して «成分ごとに動く» を成り立たせる。
        out = {}
        for key, value in spec.items():
            child_path = f"{ctx['path']}.{key}" if ctx.get("path") else key
            out[key] = resolve_animated(value, {**ctx, "path": child_path}, UNDEFINED)
        return out

    time = local_time_for(spec, ctx.get("time"))
    raw_value = spec.get("value", UNDEFINED)
    raw_base = spec.get("base", UNDEFINED)
    has_base = raw_value is not UNDEFINED or raw_base is not UNDEFINED
    if has_base:
        source = raw_value if raw_value is not UNDEFINED else raw_base
        value = resolve_animated(source, {**ctx, "time": time}, fallback)
    else:
        value = fallback

    if spec.get("keyframes"):
        sampled = sample_keyframes(spec["keyframes"], time, extrapolate=spec.get("extrapolate"))
        if sampled is not UNDEFINED:
            value = sampled

    if spec.get("expression"):
        scope = dict(ctx.get("scope") or {})
        scope["time"] = time
        scope["value"] = 0 if is_nullish(value) else value
        scope["base"] = scope["value"]
        value = ctx["engine"].evaluate(
            spec["expression"], scope, path=ctx.get("path"), file=ctx.get("file")
        )

    modulators = spec.get("modulators")
    if is_nullish(modulators):
        modulators = [spec["modulator"]] if spec.get("modulator") else None
    if modulators:
        combine = spec.get("combine") or "add"
        # 土台が無いときは «モジュレーターの出力そのもの» が値になる。
        # 単位元は combine の種類で変わる。
        if not is_nullish(value) and (has_base or spec.get("keyframes") or spec.get("expression")):
            base = to_number(value)
        else:
            base = 1 if combine == "multiply" else 0

        def make_resolve_number(local_time):
            def resolve_number_(v, d):
                if is_nullish(v):
                    return d
                if _is_number(v):
                    return v
                return to_number(resolve_animated(v, {**ctx, "time": local_time}, d))

            return resolve_number_

        results = [
            evaluate_modulator(
                modulator,
                time,
                {
                    "seed": (ctx.get("seed") or 0) + index * 7919,
                    "audio": ctx.get("audio"),
                    "bpm": ctx.get("bpm"),
                    "fps": ctx.get("fps"),
                    "resolve_number": make_resolve_number(time),
                },
            )
            for index, modulator in enumerate(modulators)
        ]
        value = combine_values(results, combine, base)

    if value is UNDEFINED:
        value = fallback
    if _is_number(value):
        clamp_spec = spec.get("clamp")
        if isinstance(clamp_spec, (list, tuple)) and len(clamp_spec) == 2:
            value = clamp(value, clamp_spec[0], clamp_spec[1])
        if spec.get("round"):
            value = js_round(value)
    return value


def resolve_number(spec, ctx, fallback=0):
    """必ず «有限の数» を返す薄い包み。"""
    value = resolve_animated(spec, ctx, fallback)
    n = to_number(value)
    return n if math.isfinite(n) else fallback


def get_path(target, path):
    """`"transform.x"` のような点つなぎの道をたどる。"""
    node = target
    for part in str(path).split("."):
        if is_nullish(node):
            return UNDEFINED
        if isinstance(node, dict):
            node = node.get(part, UNDEFINED)
        else:
            return UNDEFINED
    return node


def set_path(target, path, value):
    """点つなぎの道に値を書く。途中の入れ物は作る。"""
    parts = str(path).split(".")
    node = target
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return target


def apply_animations(target, animations, ctx):
    """`animations` を «解決済みのレイヤーの状態» に当てる。

    各項目は `property`（`transform.x` のような点つなぎの道）を狙います。
    `relative: True` にすると、置き換えではなく «今の値に足す» になります。
    """
    if not isinstance(animations, (list, tuple)):
        return target
    for animation in animations:
        if not animation or not animation.get("property"):
            continue
        if animation.get("enabled") is False:
            continue
        if _is_number(animation.get("startTime")) and ctx.get("time") < animation["startTime"]:
            continue
        if _is_number(animation.get("endTime")) and ctx.get("time") > animation["endTime"]:
            continue

        path = animation["property"]
        current = get_path(target, path)
        # **キーは常に全部そろえます**（無い項目は UNDEFINED）。
        # JS 版はオブジェクトリテラルで組むので `keyframes` などのキーが
        # «値は undefined でも存在する» 状態になり、`isAnimatedSpec` が必ず真に
        # なります。Python で «無いキーは入れない» と書くと
        # `{"property": "a.b", "value": 9}` が «ただの入れ物» と判定され、
        # 9 ではなく {"value": 9} が書き込まれます（実際にそうなりました）。
        spec = {
            "value": UNDEFINED if animation.get("relative") else animation.get("value", UNDEFINED),
            "keyframes": animation.get("keyframes", UNDEFINED),
            "expression": animation.get("expression", UNDEFINED),
            "modulator": animation.get("modulator", UNDEFINED),
            "modulators": animation.get("modulators", UNDEFINED),
            "combine": animation.get("combine", UNDEFINED),
            "delay": animation.get("delay", UNDEFINED),
            "timeScale": animation.get("timeScale", UNDEFINED),
            "loop": animation.get("loop", UNDEFINED),
            "loopDuration": animation.get("loopDuration", UNDEFINED),
            "extrapolate": animation.get("extrapolate", UNDEFINED),
            "clamp": animation.get("clamp", UNDEFINED),
            "round": animation.get("round", UNDEFINED),
        }

        resolved = resolve_animated(
            spec, {**ctx, "path": path}, 0 if animation.get("relative") else current
        )

        if resolved is UNDEFINED:
            continue
        if animation.get("relative") and _is_number(current):
            set_path(target, path, current + to_number(resolved))
        else:
            set_path(target, path, resolved)
    return target


__all__ = [
    "ANIMATED_KEYS",
    "apply_animations",
    "get_path",
    "is_animated_spec",
    "local_time_for",
    "resolve_animated",
    "resolve_number",
    "set_path",
]
