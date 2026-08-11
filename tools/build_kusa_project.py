"""「木みたいな草だった」の MV プロジェクト JSON を組み立てて書き出す。

`make-mv` と同じ手順を踏みますが、**描かずに JSON で止めます**。1 本 12 分の
レンダリングを回す前に «文字が収まっているか» を 1 フレームで確かめられるように
するためです（`movo frame <json> --time 34`）。

    python tools/build_kusa_project.py hype-lyric-mv tmp/kusa-hype.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movo.audio import analyze_audio, decode_audio_file  # noqa: E402
from movo.cli.commands.make_mv import (  # noqa: E402
    build_sequence, distribute_timed_lines, parse_asset_assignments, plan_sequence,
)
from movo.core import parse_lyrics  # noqa: E402
from movo.skill import SkillRegistry, build_movie_project  # noqa: E402

# 曲のファイル。手元の場所は人によって違うので、環境変数で渡せるようにしてある。
#   MOVO_DEMO_SONG=/path/to/song.mp3 python tools/build_kusa_project.py
SONG = os.environ.get("MOVO_DEMO_SONG", "tmp/song.mp3")
LRC = "tmp/kusa.lrc"


def build(style: str, out: str, *, intensity=0.0, max_bars=8, assets=(), sets=None, size=(1280, 720)):
    root = os.getcwd()
    registry = SkillRegistry().load(project_root=root)
    entry = registry.movie(style)

    analysis = analyze_audio(decode_audio_file(SONG), {"beatsPerBar": 4, "maxBeats": 0})
    bpm = round(analysis["bpm"] * 100) / 100
    plan = plan_sequence(analysis["sections"], bpm, 4, {"maxBars": max_bars, "intensity": intensity})

    timed = parse_lyrics(Path(LRC).read_text(encoding="utf-8"))
    lines = [row["text"] for row in timed]
    per_scene = distribute_timed_lines(timed, plan, (60 / bpm) * 4)
    sequence = build_sequence(plan, entry["definition"].get("sequence") or [], per_scene, authoritative=True)

    tailored = {
        **entry["definition"],
        "sequence": sequence,
        "project": {**(entry["definition"].get("project") or {}), "bpm": bpm},
        "assets": {
            **(entry["definition"].get("assets") or {}),
            **parse_asset_assignments(list(assets)),
            "_track": {"type": "audio", "path": SONG},
        },
        "audio": [{"asset": "_track", "volume": 0.9, "fadeOut": 2}],
    }
    registry.register(tailored, kind="movie", source="build_kusa_project")

    inputs = {"title": "木みたいな草だった", "bpm": bpm, "lines": "\n".join(lines), **(sets or {})}
    for key, definition in (entry["definition"].get("inputs") or {}).items():
        if key in (sets or {}) or not isinstance(definition, dict):
            continue
        if definition.get("type") == "textList":
            inputs[key] = "\n".join(lines)
    counted = Counter(lines)
    top, times = counted.most_common(1)[0]
    if times > 1 and "hook" not in (sets or {}):
        inputs["hook"] = top

    built = build_movie_project(registry, style, inputs, {"width": size[0], "height": size[1], "name": "kusa"})

    # **CRF を上げます。** 既定は 18（ほぼ可逆）で、rich-* のシーンは全面に
    # グレインを乗せるため、x264 がその粒を全部符号化して **154 秒で 651MB
    # （35Mbps）** になりました。21 なら見た目はほぼ変わらず 1/5 以下になります。
    project = built["project"]
    output = dict(project.get("output") or {})
    output.setdefault("crf", 21)
    output.setdefault("preset", "medium")
    project["output"] = output

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(built["project"], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f'{out} を書きました（シーン {len(built["project"]["scenes"])} 個 / BPM {bpm}）')


if __name__ == "__main__":
    style = sys.argv[1] if len(sys.argv) > 1 else "hype-lyric-mv"
    out = sys.argv[2] if len(sys.argv) > 2 else "tmp/kusa-hype.json"
    build(style, out, intensity=0.6)
