"""JS 版と «数値そのもの» を突き合わせる（タイムライン）。

基準の作り直しかた:

    node tests/data/parity_timeline.mjs > tests/data/parity_timeline.json

タイムラインは «どのシーンがいつ映るか» を決める土台です。ここが 1 フレーム
ずれると、**映像も音も歌詞も全部ずれます**。しかも «少しずれている» のは
目で見て気付きにくいので、JS 版の解決結果と 1 つずつ突き合わせています。

見ているもの:

  - シーンを順に並べる／`start` を書いたシーン／`enabled: false` の飛ばし
  - 長さを書いていないシーンが «中身から» 決まる（キーフレームの終わりまで）
  - 長さを書いていないシーンがプロジェクトの長さまで伸びる
  - レイヤーの `zIndex` と «書いた順» による並び替え（安定ソート）
  - 入れ子のレイヤーとその時刻
  - `scenes_at` / `is_layer_active` / `time_to_frame` / `resolve_range`
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from movo.timeline import (
    all_layers, build_timeline, frame_to_time, is_layer_active,
    resolve_range, scenes_at, time_to_frame,
)

GOLDEN = json.loads((Path(__file__).parent / "data" / "parity_timeline.json").read_text("utf-8"))

PROJECTS = {
    "sequential": {
        "video": {"width": 640, "height": 360, "fps": 24, "duration": 12},
        "scenes": [
            {"id": "a", "duration": 3, "layers": [{"type": "text", "text": "x"}, {"type": "shape", "zIndex": -1}]},
            {"id": "b", "duration": 4, "layers": [{"type": "image", "start": 1, "duration": 2}]},
            {"id": "c", "start": 9, "duration": 2, "layers": []},
        ],
    },
    "intrinsic": {
        "video": {"width": 320, "height": 180, "fps": 30},
        "scenes": [
            {"id": "anim", "layers": [
                {"type": "text", "animations": [{"delay": 0.5, "keyframes": [{"time": 0}, {"time": 2.5}]}]}
            ]},
            {"id": "ends", "layers": [{"type": "shape", "end": 4}]},
        ],
    },
    "stretch": {
        "video": {"width": 320, "height": 180, "fps": 12, "duration": 10},
        "scenes": [{"id": "only", "layers": [{"type": "shape"}]}],
    },
    "nested": {
        "video": {"width": 320, "height": 180, "fps": 30, "duration": 6},
        "scenes": [{
            "id": "root", "duration": 6,
            "layers": [
                {"id": "group", "type": "group", "start": 1, "duration": 4, "layers": [
                    {"type": "text", "zIndex": 5}, {"type": "shape", "start": 0.5, "duration": 1},
                ]},
                {"type": "image", "zIndex": 5},
                {"type": "image", "zIndex": 5},
                {"type": "shape", "enabled": False},
            ],
        }],
    },
    "disabled": {
        "video": {"width": 320, "height": 180, "fps": 25, "duration": 8},
        "scenes": [
            {"id": "skip", "enabled": False, "duration": 3, "layers": []},
            {"id": "keep", "duration": 3, "layers": []},
        ],
    },
    "empty": {"video": {"width": 320, "height": 180, "fps": 30}, "scenes": []},
}


def summarise(timeline: dict) -> dict:
    return {
        "fps": timeline["fps"],
        "duration": timeline["duration"],
        "frameCount": timeline["frameCount"],
        "width": timeline["width"],
        "height": timeline["height"],
        "scenes": [
            {
                "id": s["id"], "index": s["index"], "start": s["start"], "end": s["end"],
                "duration": s["duration"],
                "layers": [
                    {
                        "id": l["id"], "type": l["type"], "order": l["order"], "zIndex": l["zIndex"],
                        "localStart": l["localStart"], "localEnd": l["localEnd"],
                        "children": [
                            {"id": c["id"], "localStart": c["localStart"], "localEnd": c["localEnd"],
                             "zIndex": c["zIndex"]}
                            for c in (l["children"] or [])
                        ],
                    }
                    for l in s["layers"]
                ],
            }
            for s in timeline["scenes"]
        ],
        "allLayerIds": [l["id"] for l in all_layers(timeline)],
    }


@pytest.mark.parametrize("label", list(PROJECTS))
def test_timeline_matches_js(label):
    timeline = build_timeline(PROJECTS[label])
    want = GOLDEN[label]
    got = summarise(timeline)
    for key in ("fps", "duration", "frameCount", "width", "height", "allLayerIds"):
        assert got[key] == want[key], f"{label}.{key}: {got[key]} ≠ {want[key]}"
    assert got["scenes"] == want["scenes"], f"{label}: シーンの解決が違います"


@pytest.mark.parametrize("label", list(PROJECTS))
def test_scenes_at_matches_js(label):
    """**0.5 秒刻みで «どのシーンが映るか» を全部見ます。** 境界がずれると
    シーンの継ぎ目で 1 フレーム黒くなります。"""
    timeline = build_timeline(PROJECTS[label])
    want = GOLDEN[label]["scenesAt"]
    # JS 版と同じ «0.5 を足し込む» ループ（`arange` だと点の数がずれます）。
    got = []
    t = 0.0
    while t <= timeline["duration"] + 0.001:
        got.append([round(t, 3), [s["id"] for s in scenes_at(timeline, t)]])
        t += 0.5
    assert got == want


@pytest.mark.parametrize("label", list(PROJECTS))
def test_frames_and_ranges_match_js(label):
    timeline = build_timeline(PROJECTS[label])
    want = GOLDEN[label]
    got = [[t, time_to_frame(timeline, t), frame_to_time(timeline, time_to_frame(timeline, t))]
           for t in (0, 0.5, 1.0001, 3.49, 5)]
    assert got == want["frames"]

    assert resolve_range(timeline, {}) == want["ranges"]["all"]
    assert resolve_range(timeline, {"from": 1, "to": 3}) == want["ranges"]["partial"]
    assert resolve_range(timeline, {"from": -5, "to": 1e6}) == want["ranges"]["clamped"]
    if "scene" in want["ranges"]:
        assert resolve_range(timeline, {"scene": timeline["scenes"][0]["id"]}) == want["ranges"]["scene"]


@pytest.mark.parametrize("label", [k for k in PROJECTS if "layerActive" in GOLDEN[k]])
def test_layer_active_matches_js(label):
    timeline = build_timeline(PROJECTS[label])
    layers = timeline["scenes"][0]["layers"]
    got = [[t, [is_layer_active(l, t) for l in layers]] for t in (0, 0.5, 1, 2, 3, 4)]
    assert got == GOLDEN[label]["layerActive"]


def test_unknown_scene_is_rejected():
    """存在しないシーン名は **黙って全体を描かず**、名前を挙げて失敗すること。"""
    timeline = build_timeline(PROJECTS["sequential"])
    with pytest.raises(ValueError) as error:
        resolve_range(timeline, {"scene": "ない"})
    assert "a" in str(error.value) and "b" in str(error.value)


# ── group の入れ子（JSON の綴りと内部の綴りの食い違い）──────────


def _group_scene(key: str) -> dict:
    """`layers` / `children` どちらの綴りでも同じ内容の group を作る。"""
    return {
        "id": "s",
        "duration": 1,
        "layers": [
            {
                "id": "grp",
                "type": "group",
                key: [{"id": "child", "type": "shape", "shape": {"type": "ellipse", "width": 8, "height": 8}}],
            }
        ],
    }


def _prepared_group(key: str):
    from movo.timeline import build_timeline

    project = {
        "project": {"name": "t"},
        "video": {"width": 32, "height": 32, "fps": 10, "duration": 1},
        "scenes": [_group_scene(key)],
    }
    timeline = build_timeline(project)
    return timeline["scenes"][0]["layers"][0]


def test_group_children_survive_the_layers_spelling():
    assert len(_prepared_group("layers")["children"]) == 1


def test_group_children_survive_the_children_spelling():
    """実際に踏んだ壊れ方: JSON に `children` と書くと中身が消える。

    JSON の綴りは `layers`、レンダラーが読む綴りは `children` です。
    `children` を無条件に上書きしていたため、**検証は通るのに何も描かれない**
    という、いちばん気付きにくい形で壊れていました。
    """
    assert len(_prepared_group("children")["children"]) == 1


def test_group_without_children_is_none():
    from movo.timeline import build_timeline

    project = {
        "project": {"name": "t"},
        "video": {"width": 32, "height": 32, "fps": 10, "duration": 1},
        "scenes": [{"id": "s", "duration": 1, "layers": [{"id": "grp", "type": "group"}]}],
    }
    prepared = build_timeline(project)["scenes"][0]["layers"][0]
    assert prepared["children"] is None
