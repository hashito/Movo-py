"""曲の構造（区間）をシーンに貼る。

`analyze_audio` は区間・拍・小節をすべて出しており、BPM 解決がそれを
`project["_audioAnalysis"]` に残しています。ところが **読む側が長らく
一人もいませんでした**。JSON から出られるのは `bpm` だけだったので、
`movo batch` で 10 曲流しても «どの曲もイントロ 4 小節» という決め打ちに
なっていました。曲ごとに構成が違うのに、構成だけが曲を無視している状態です。

ここでその配線をつなぎます。シーンに «何小節» ではなく **«どの区間»** を
書けるようにすると、曲を差し替えるだけで構成が付いてきます。

  { "id": "intro", "from": { "section": 0 } }
  { "id": "hook",  "from": { "section": "chorus", "nth": 1 } }
"""

from __future__ import annotations

import json

from movo.expression._compat import ErrorCodes, MovoError, is_finite_number, js_number, js_round


def structure_of(project):
    """解析結果から «公開用» の構造を作る。

    `_audioAnalysis` は内部の置き場なので、そのまま参照させると名前を変えられなく
    なります。`project["structure"]` を正とします。
    """
    if not isinstance(project, dict):
        return None
    # 利用者が自分で書いた構造が最優先。解析より «人が書いたもの» を信じます。
    declared = project.get("structure")
    if isinstance(declared, dict):
        sections = declared.get("sections")
        if isinstance(sections, list) and len(sections):
            return {"sections": sections}
    analysis = project.get("_audioAnalysis")
    analysed = analysis.get("sections") if isinstance(analysis, dict) else None
    if isinstance(analysed, list) and len(analysed):
        return {"sections": analysed}
    return None


def find_section(sections, spec):
    """`from: { section }` の指す区間を探す。見つからなければ None。"""
    key = spec.get("section")
    if isinstance(key, (int, float)) and not isinstance(key, bool):
        # 負の添字は «後ろから»。`start: { bar: -4 }` を «終わりから 4 小節» と
        # 読ませている以上、ここだけ別の意味にすると混乱します。
        index = int(len(sections) + key) if key < 0 else int(key)
        return sections[index] if 0 <= index < len(sections) else None
    # ラベル指定。`nth` は 1 始まり（«1 番目のサビ» と数えるのが自然なので）。
    matched = [section for section in sections if section.get("label") == key]
    if not matched:
        return None
    nth = max(1, js_round(js_number(spec.get("nth", 1)) if spec.get("nth") is not None else 1))
    nth = int(nth)
    return matched[nth - 1] if 0 < nth <= len(matched) else None


def resolve_structure(project):
    """シーンの `from` / `start` を、実際の秒に畳む（破壊的）。

    **正規化の早い段階で呼びます。** 小節の解決（`"4bar"` → 秒）より前に
    数値へ潰しておかないと、区間から来た尺が二重に変換されます。
    """
    structure = structure_of(project)
    scenes = project.get("scenes") if isinstance(project, dict) else None
    if not isinstance(scenes, list):
        return project

    bar_seconds = _bar_seconds_of(project)
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        # `start: { bar: -4 }` — 曲の «終わりから 4 小節» のような書き方。
        start = scene.get("start")
        if isinstance(start, dict) and "bar" in start:
            bars = js_number(start.get("bar"))
            if not is_finite_number(bars):
                raise MovoError(
                    ErrorCodes.MOVO_SCHEMA_INVALID,
                    f'scene "{scene.get("id", "?")}" の start.bar が数値ではありません',
                )
            total = _total_seconds(project, structure)
            scene["start"] = bars * bar_seconds if bars >= 0 else max(0, total + bars * bar_seconds)

        source = scene.get("from")
        if not isinstance(source, dict) or source.get("section") is None:
            continue
        if not structure:
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                f'scene "{scene.get("id", "?")}" が区間を指していますが、曲の構造がありません',
                hint="project.structure.fromAudio か project.bpm.fromAudio で曲を渡してください",
            )
        section = find_section(structure["sections"], source)
        if not section:
            labels = " / ".join(dict.fromkeys(str(s.get("label")) for s in structure["sections"]))
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                f'scene "{scene.get("id", "?")}" の指す区間が見つかりません: '
                f"{json.dumps(source, ensure_ascii=False, separators=(',', ':'))}",
                hint=f"この曲の区間は {len(structure['sections'])} 個（ラベル: {labels}）です",
            )
        # 尺を «書いてあれば» 尊重します。区間に貼りたいのは主に開始位置で、
        # 尺だけ手で決めたいことがあるためです。
        if scene.get("duration") is None:
            scene["duration"] = section["end"] - section["start"]
        if scene.get("start") is None:
            scene["start"] = section["start"]
        scene["_section"] = {
            "start": section.get("start"),
            "end": section.get("end"),
            "label": section.get("label"),
            "energy": section.get("energy"),
        }
        del scene["from"]
    return project


def _bar_seconds_of(project) -> float:
    """1 小節の秒数。BPM が無ければ 120 とみなします。"""
    settings = project.get("project") if isinstance(project, dict) else None
    settings = settings if isinstance(settings, dict) else {}
    bpm = js_number(settings.get("bpm"))
    if not is_finite_number(bpm) or bpm == 0:
        bpm = 120
    signature = settings.get("timeSignature")
    beats = None
    if isinstance(signature, dict):
        beats = signature.get("beatsPerBar")
    if beats is None:
        beats = settings.get("beatsPerBar")
    beats_value = js_number(beats) if beats is not None else 4
    if not is_finite_number(beats_value) or beats_value == 0:
        beats_value = 4
    return (60 / bpm) * beats_value


def _total_seconds(project, structure) -> float:
    """曲全体の長さ。区間の終わりか、`video.duration` を使います。"""
    if structure and structure.get("sections"):
        return structure["sections"][-1]["end"]
    video = project.get("video") if isinstance(project, dict) else None
    declared = js_number(video.get("duration")) if isinstance(video, dict) else float("nan")
    return declared if is_finite_number(declared) else 0


__all__ = ["find_section", "resolve_structure", "structure_of"]
