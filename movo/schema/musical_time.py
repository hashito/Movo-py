"""拍・小節で時間を書けるようにする。

MV を 10 本作って、いちばん効いた設計判断が «カット尺を秒ではなく小節で決める»
ことでした。拍同期の演出を足さなくても、それだけで音に合って見えます。

ところが素の JSON では毎回こちらで計算することになります。

    205 BPM → 1 小節 = 60 / 205 * 4 = 1.17073... 秒
    イントロ 4 小節 = 4.68293 秒     ← これを手で書いていた

そこで、時間を書く場所では次の書き方を許します。

    "4bar"      4 小節
    "8beat"     8 拍
    "1/2bar"    半小節
    "0.5s"      0.5 秒（明示）
    0.5         0.5 秒（従来どおり）

変換はここ（正規化の段階）で済ませ、レンダラ以降は秒しか見ません。
"""

from __future__ import annotations

import re

from movo.expression._compat import ErrorCodes, MovoError, is_finite_number, js_number

#: `"4bar"` `"1/2bar"` `"8beat"` `"0.5s"` を受ける。
PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*(bar|bars|beat|beats|s|sec|secs)\s*\Z",
    re.IGNORECASE,
)


def musical_units(project_settings=None):
    """BPM と拍子から «1 拍・1 小節の長さ» を出す。BPM が無ければ None。"""
    project_settings = project_settings or {}
    bpm = js_number(project_settings.get("bpm"))
    if not is_finite_number(bpm) or bpm <= 0:
        return None
    # 拍子。[4, 4] でも "4/4" でも受ける。分子が «1 小節あたりの拍数»。
    beats_per_bar = 4.0
    signature = project_settings.get("timeSignature")
    if isinstance(signature, (list, tuple)) and len(signature) and js_number(signature[0]) > 0:
        beats_per_bar = js_number(signature[0])
    elif isinstance(signature, str) and re.match(r"^\d+\s*/\s*\d+$", signature):
        beats_per_bar = js_number(signature.split("/")[0])
    beat = 60 / bpm
    return {"beat": beat, "bar": beat * beats_per_bar, "bpm": bpm, "beatsPerBar": beats_per_bar}


def to_seconds(value, units, path=""):
    """1 つの値を秒に直す。数値と «対象外の文字列» はそのまま返す。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    match = PATTERN.match(value)
    if not match:
        return value
    numerator, denominator, raw_unit = match.group(1), match.group(2), match.group(3)
    amount = js_number(numerator) / js_number(denominator) if denominator else js_number(numerator)
    unit = raw_unit.lower()
    if unit.startswith("s"):
        return amount
    if not units:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'"{value}" は拍の単位ですが project.bpm がありません',
            path=path,
            hint="project.bpm を設定するか、秒で書いてください",
        )
    return amount * (units["bar"] if unit.startswith("bar") else units["beat"])


def is_musical_time(value) -> bool:
    """その文字列が拍・小節の指定に見えるか。"""
    return isinstance(value, str) and PATTERN.match(value) is not None


#: 時間として扱うキーの一覧。
#:
#: «時間» と «それ以外の数値» を取り違えると、たとえば `duration` という名前の
#: 別物（エフェクトの強さなど）まで拍で解釈してしまいます。そこで «この位置に
#: あるこのキー» を明示的に列挙します。
TIME_KEYS = frozenset(
    {
        "start",
        "end",
        "duration",
        "time",
        "delay",
        "at",
        "startAt",
        "endAt",
        "fadeIn",
        "fadeOut",
        "in",
        "out",
        "offset",
        "prewarm",
        "lifetime",
        "from",
        "to",
        "startTime",
        "endTime",
        "loopDuration",
    }
)

#: `from` / `to` / `offset` のように «時間とは限らない» キーは、
#: 拍の書き方（"4bar"）をしているときだけ変換します。数値はそのまま。
ALWAYS_NUMERIC_CONTEXT = frozenset(
    {"shape", "style", "textBox", "emitter", "linePath", "neonPath", "speedLines", "starfield"}
)


def resolve_musical_time(project):
    """プロジェクト全体を歩いて、拍・小節の書き方を秒に直す（破壊的）。

    文字列が拍の形をしているときだけ触るので、既存の JSON は 1 文字も変わりません。
    """
    units = musical_units(project.get("project") or {})

    def walk(node, path, parent_key):
        if isinstance(node, list):
            for i, child in enumerate(node):
                if is_musical_time(child) and parent_key in TIME_KEYS:
                    node[i] = to_seconds(child, units, f"{path}[{i}]")
                else:
                    walk(child, f"{path}[{i}]", parent_key)
            return
        if not isinstance(node, dict):
            return
        for key, value in list(node.items()):
            child_path = f"{path}.{key}" if path else key
            if isinstance(value, str):
                # 拍の書き方をしていて、かつ «時間のキー» のときだけ直す。
                if is_musical_time(value) and (
                    key in TIME_KEYS or parent_key not in ALWAYS_NUMERIC_CONTEXT
                ):
                    node[key] = to_seconds(value, units, child_path)
                continue
            walk(value, child_path, key)

    walk(project.get("scenes"), "scenes", "scenes")
    walk(project.get("compositions"), "compositions", "compositions")
    walk(project.get("audio"), "audio", "audio")
    walk(project.get("camera"), "camera", "camera")
    walk(project.get("presets"), "presets", "presets")

    # video.duration も拍で書けるようにする（"32bar" で 1 曲ぶん）
    video = project.get("video")
    if isinstance(video, dict) and is_musical_time(video.get("duration")):
        video["duration"] = to_seconds(video["duration"], units, "video.duration")
    return project


__all__ = [
    "ALWAYS_NUMERIC_CONTEXT",
    "PATTERN",
    "TIME_KEYS",
    "is_musical_time",
    "musical_units",
    "resolve_musical_time",
    "to_seconds",
]
