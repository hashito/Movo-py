"""尺追従の保護区間（Responsive Design – Time）。

スキルは尺（`duration`）を変えて使えます。短い尺に入れたときに動きが終わり切ら
ないと、塗りが途中で止まったまま終わってしまうので、展開時に «レイヤーごと一律の
比率で» キーフレームを畳んできました。

ところが一律に畳むと、**«登場の勢い» まで一緒に縮みます**。6 小節ぶんのサビを
3 小節で使うと、跳ねて出るはずの決め文句がぬるっと出てきて別物になります。
逆に伸ばせば、間延びした登場になります。

Movo ではこれが BPM でも起きます。尺を小節で書くからです。

    同じ «8 小節»   92 BPM → 20.87 秒 ／ 205 BPM → 9.37 秒

秒で書いた 0.3 秒の登場が、**曲を変えただけで半分以下の速さ**になります。
After Effects の MOGRT が保護区間を持つのと同じ理由で、Movo にはもっと強く
必要でした。

そこで «頭» と «尻» の決まった長さは畳まず、**真ん中だけを比率で伸縮**します。

    元   |<-- in -->|<--------- 真ん中 --------->|<- out ->|
    先   |<-- in -->|<-- 真ん中（縮む） -->|<- out ->|

保護区間は秒でも小節でも書けます（`0.8` / `"1beat"` / `"1/2bar"`）。
拍で書けば «決め文句は 1 拍で出る» が BPM を変えても保たれます。

## 保護区間の合計が尺より長いとき — «縮める» と «警告» の両方

保護を守り切ろうとすると、保護区間だけで尺をはみ出します。それは «動きが途中で
止まったまま終わる» という、畳みがそもそも防ごうとしていた壊れ方に戻ることです。
だから最後は «収まること» を優先して、保護区間ごと一律に畳みます。

ただし黙って畳むと、**«保護区間を書いたのに効いていない» ことが誰にも見えません**。
そこで警告を 1 行出します。「尺を延ばす」「保護区間を短くする」のどちらを選ぶかは
作る人にしか決められないので、直し方まで書いておきます。
"""

from __future__ import annotations

import copy
import re
from typing import Any

from movo.cli.bridge import optional_module
from movo.cli.console import logger

# フレーム 1 枚ぶんの余裕。最後のキーフレームを «消える少し手前» に置くための値です。
FRAME_MARGIN = 0.03


def _musical_seconds(value: Any, bpm: float | None, time_signature: Any, label: str) -> Any:
    """`"1beat"` `"1/2bar"` `0.8` を秒に直す。

    **`movo.schema.musical_time` が用意できたらそちらを使います。** ここに置いて
    あるのは «schema がまだ来ていない間もスキルを試せるようにする» ための最小実装
    で、`bar` / `beat` / 分数・小数の係数だけを扱います。schema 側は同じ書式に
    加えて `frame` などの単位も持つので、揃ったらこの関数は素通しになります。
    """
    module = optional_module("movo.schema.musical_time")
    if module is not None:
        units = getattr(module, "musical_units", None) or getattr(module, "musicalUnits", None)
        to_seconds = getattr(module, "to_seconds", None) or getattr(module, "toSeconds", None)
        if units is not None and to_seconds is not None:
            # 単位表は `project` の設定（`bpm` / `timeSignature`）から作ります。
            # 綴りはプロジェクト JSON と同じです。
            settings = {"bpm": bpm}
            if time_signature is not None:
                settings["timeSignature"] = time_signature
            return to_seconds(value, units(settings), label)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+(?:/[0-9]*\.?[0-9]+)?)\s*(bar|beat|b)?", text)
    if not match:
        try:
            return float(text)
        except ValueError:
            return None
    amount_text, unit = match.group(1), match.group(2)
    if "/" in amount_text:
        numerator, _, denominator = amount_text.partition("/")
        try:
            amount = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return None
    else:
        amount = float(amount_text)
    if unit is None:
        return amount
    if bpm is None or bpm <= 0:
        return None
    beats_per_bar = 4
    if isinstance(time_signature, (list, tuple)) and time_signature:
        beats_per_bar = float(time_signature[0]) or 4
    elif isinstance(time_signature, str) and "/" in time_signature:
        beats_per_bar = float(time_signature.split("/")[0] or 4)
    beat_seconds = 60.0 / float(bpm)
    return amount * (beat_seconds * beats_per_bar if unit == "bar" else beat_seconds)


def resolve_protect(spec: Any, context: dict[str, Any] | None = None) -> dict[str, float | None] | None:
    """`{ in, out }` の指定を秒に直す。

    書いていない側は `None` のまま返します。**0 と «書いていない» を区別しないと、
    レイヤーの `timeProtect` でどちらか片側だけを上書きできない**ためです。
    """
    context = context or {}
    if not isinstance(spec, dict):
        return None
    label = context.get("path", "responsiveTime")

    def side(value: Any, name: str) -> float | None:
        if value is None:
            return None
        seconds = _musical_seconds(value, context.get("bpm"), context.get("timeSignature"), f"{label}.{name}")
        try:
            number = float(seconds)
        except (TypeError, ValueError):
            number = float("nan")
        if number != number or number < 0:
            # 打ち間違い（"1bea" など）を «保護区間 0» として黙って流すと、
            # 効かない理由が分からないまま終わります。
            logger.warn(f"{label}.{name}: 保護区間として読めない値です（{value!r}）。無視します")
            return None
        return number

    inside = side(spec.get("in"), "in")
    outside = side(spec.get("out"), "out")
    if inside is None and outside is None:
        return None
    return {"in": inside, "out": outside}


def merge_protect(outer: dict | None, inner: dict | None) -> dict[str, float] | None:
    """スキル側の `responsiveTime` とレイヤー側の `timeProtect` を重ねる。

    片側ずつ重ねるので、スキルで `{in, out}` を決めておいて、あるレイヤーだけ
    `{"in": "1beat"}` と頭だけ変える、という書き方ができます。
    """
    if not outer and not inner:
        return None
    inner = inner or {}
    outer = outer or {}
    return {
        "in": inner.get("in") if inner.get("in") is not None else (outer.get("in") or 0),
        "out": inner.get("out") if inner.get("out") is not None else (outer.get("out") or 0),
    }


def _collect_keyframe_groups(layer: Any) -> list[list]:
    """レイヤーが持つキーフレームの並びをすべて集める。"""
    groups: list[list] = []

    def collect(node: Any, depth: int = 0) -> None:
        if depth > 6 or not isinstance(node, (dict, list)):
            return
        if isinstance(node, list):
            for item in node:
                collect(item, depth + 1)
            return
        if isinstance(node.get("keyframes"), list):
            groups.append(node["keyframes"])
        for key, value in node.items():
            # 入れ子のレイヤーは、そのレイヤー自身の区間で別途そろえる
            if key == "layers":
                continue
            collect(value, depth + 1)

    collect(layer)
    return groups


def _round(value: float) -> float:
    return round(value * 1000) / 1000


def _time_map_for(latest: float, target: float, origin: float, protect: dict | None, label: str):
    """時刻 → 時刻の写像を作る。

    `protect` が無いときは «全体を同じ比率で» になり、保護区間を入れる前と
    同じ結果になります（既存のプロジェクトの見た目を変えないため）。
    """
    head = origin + max(0.0, (protect or {}).get("in") or 0.0)
    tail = max(0.0, (protect or {}).get("out") or 0.0)
    movable_source = latest - head - tail
    movable_target = target - head - tail

    if movable_source <= 0 or movable_target <= 0:
        # 保護区間だけで尺を使い切ってしまう場合。«収まること» を優先して
        # 一律に畳み、代わりに警告を出す。
        if protect and ((protect.get("in") or 0) > 0 or (protect.get("out") or 0) > 0):
            logger.warn(
                f"{label}: 保護区間（頭 {_round(protect.get('in') or 0)}s ＋ 尻 "
                f"{_round(protect.get('out') or 0)}s）が尺 {_round(target)}s に収まらないので、"
                "保護をあきらめて全体を一律に畳みます"
                "（尺を延ばすか、responsiveTime / timeProtect を短くしてください）"
            )
        factor = max(0.01, target / latest)
        return lambda time: time * factor

    factor = movable_target / movable_source

    def mapping(time: float) -> float:
        if time <= head:
            return time  # 頭の保護区間はそのまま
        if time >= latest - tail:
            return target - (latest - time)  # 尻は «終わりからの距離» を保つ
        return head + (time - head) * factor

    return mapping


def fit_animations_to_clip(layer: Any, options: dict[str, Any] | None = None) -> Any:
    """キーフレームの時刻を «レイヤーが見えている間» に収める。

    はみ出していないときは何もしません。**足りないぶんを引き伸ばしはしません。**
    0.9 秒で出るタイトルを 32 秒のシーンに置いたら 32 秒かけて出てくる、では
    誰も望まない結果になるからです。保護区間は «縮めるときに何を守るか» の指定です。
    """
    options = options or {}
    if not isinstance(layer, dict):
        return layer
    label = options.get("path") or layer.get("id") or "layer"

    declared = resolve_protect(
        layer.get("timeProtect"),
        {"bpm": options.get("bpm"), "timeSignature": options.get("timeSignature"), "path": f"{label}.timeProtect"},
    )
    # `timeProtect` は «作る人向けの指定» なので、展開したあとの JSON には残しません
    # （レンダラは秒しか見ないという約束を保つため）。
    current = layer
    if "timeProtect" in layer:
        current = {k: v for k, v in layer.items() if k != "timeProtect"}
    protect = merge_protect(options.get("protect"), declared)

    if isinstance(current.get("end"), (int, float)) and not isinstance(current.get("end"), bool):
        visible_end = float(current["end"])
    elif isinstance(current.get("duration"), (int, float)) and not isinstance(current.get("duration"), bool):
        visible_end = float(current.get("start") or 0) + float(current["duration"])
    elif isinstance(options.get("clipEnd"), (int, float)):
        visible_end = float(options["clipEnd"])
    else:
        return current
    if visible_end <= 0:
        return current

    groups = _collect_keyframe_groups(current)
    if not groups:
        return current

    latest = 0.0
    for keyframes in groups:
        for keyframe in keyframes:
            time = keyframe.get("time") if isinstance(keyframe, dict) else None
            if isinstance(time, (int, float)) and not isinstance(time, bool) and time > latest:
                latest = float(time)
    target = visible_end - FRAME_MARGIN
    if latest <= target or latest <= 0:
        return current

    # 保護区間はレイヤーの «頭» から測ります。start がずれているレイヤーでも
    # «出てきてから 1 拍» が同じ意味になるからです。保護区間を書いていない
    # レイヤーは 0 起点のまま（＝従来と同じ計算）にしてあります。**同じ JSON から
    # 同じ動画が出るという約束を、この変更で崩さないため**です。
    origin = max(0.0, float(current.get("start") or 0)) if protect else 0.0
    mapping = _time_map_for(latest, target, origin, protect, label)

    shrunk = copy.deepcopy(current)

    def rescale(node: Any, depth: int = 0) -> None:
        if depth > 6 or not isinstance(node, (dict, list)):
            return
        if isinstance(node, list):
            for item in node:
                rescale(item, depth + 1)
            return
        if isinstance(node.get("keyframes"), list):
            for keyframe in node["keyframes"]:
                if isinstance(keyframe, dict) and isinstance(keyframe.get("time"), (int, float)):
                    if not isinstance(keyframe["time"], bool):
                        keyframe["time"] = _round(mapping(float(keyframe["time"])))
        for key, value in node.items():
            if key == "layers":
                continue
            rescale(value, depth + 1)

    rescale(shrunk)
    return shrunk
