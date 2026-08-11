"""movo-timeline — 宣言的なプロジェクトを «時刻で引ける形» に開く。

シーンは `start` を書かないかぎり順番に並びます。レイヤーの時刻は
**シーン相対**です（4 秒から始まるシーンの中の `start: 1` は、全体では 5 秒）。
作者はそのつもりで書くので、ここを絶対時刻にすると «シーンを 1 つ足したら
全部書き直し» になります。

移植元: ``packages/timeline/src/index.js``（167 行）

## 辞書のキーについて

戻り値のキーは **JS 版と同じ camelCase** です（``frameCount`` ``localStart``
``zIndex``…）。理由は 2 つあります。

1. レイヤーの辞書は **プロジェクト JSON をそのまま広げたもの**で、
   ``zIndex`` や ``timeRemap`` は JSON の綴りです。ここだけ snake_case に
   直すと「JSON の綴り」と「内部の綴り」の 2 つを覚えることになります。
2. 呼ぶ側（``movo/cli/pipeline.py``）が既に ``timeline["frameCount"]`` /
   ``rng["startFrame"]`` で読んでいます。**呼ぶ側に合わせました。**
"""

from __future__ import annotations

from movo.cli.console import logger
from movo.core.math import js_round

# **`round()` ではなく `js_round()` を使います。** Python の `round` は «偶数へ»
# 丸めるので、fps 25 の 0.5 秒が `round(12.5) == 12`、JS の `Math.round(12.5)` は
# 13 になります。フレーム番号が 1 つずれると、そこから先の絵と音がまるごと
# 1 フレームずれます（実際に fps 25 の検査で JS 版と食い違いました）。

#: `video.duration` もシーンの長さも決まらなかったときの秒数。
DEFAULT_DURATION = 5

# 時刻の比較に使う «ごく小さい値»。フレーム時刻は割り算で作るので、
# 3.0 のつもりが 2.9999999999999996 になります。境界のシーン・レイヤーが
# 1 フレームだけ消えるのを防ぐためのものです（JS 版と同じ 1e-9）。
_EPSILON = 1e-9


def build_timeline(project: dict) -> dict:
    """正規化済みのプロジェクトを時間軸に開く。

    :returns: ``{"fps","duration","frameCount","width","height","background","scenes"}``
    """
    video = project.get("video") or {}
    fps = video.get("fps")
    fps = 30 if fps is None else fps
    scenes: list[dict] = []
    cursor = 0.0

    raw_scenes = project.get("scenes") or []
    for index, scene in enumerate(raw_scenes):
        if scene.get("enabled") is False:
            continue
        start = scene.get("start")
        start = cursor if start is None else start
        duration = scene.get("duration")
        if duration is None and scene.get("end") is not None:
            duration = scene["end"] - start
        if duration is None:
            duration = _intrinsic_scene_duration(scene, project)
        duration = max(0.0, duration)
        end = start + duration
        scenes.append(
            {
                "id": scene.get("id") or f"scene-{index}",
                "index": index,
                "start": start,
                "end": end,
                "duration": duration,
                "background": scene.get("background"),
                "transition": scene.get("transition"),
                "physicsWorld": scene.get("physicsWorld"),
                "layers": _prepare_layers(scene.get("layers") or [], duration, project),
                "raw": scene,
                # `from: { section }` を書いたシーンに貼られる曲の区間。
                # 正規化の段階で入っていれば持ち回す（式の `section.*` が読む）。
                "_section": scene.get("_section"),
            }
        )
        cursor = max(cursor, end)

    intrinsic = max((s["end"] for s in scenes), default=DEFAULT_DURATION)
    duration = video.get("duration")
    if duration is None:
        duration = intrinsic if intrinsic is not None else DEFAULT_DURATION

    # 長さを書いていないシーンは、プロジェクトの長さまで伸びる。
    for scene in scenes:
        raw = scene["raw"]
        if scene["duration"] == 0 or raw.get("duration") is None:
            if raw.get("duration") is None and raw.get("end") is None:
                scene["end"] = max(scene["end"], duration)
                scene["duration"] = scene["end"] - scene["start"]
                scene["layers"] = _prepare_layers(raw.get("layers") or [], scene["duration"], project)

    frame_count = max(1, js_round(duration * fps))
    return {
        "fps": fps,
        "duration": duration,
        "frameCount": frame_count,
        "width": video.get("width"),
        "height": video.get("height"),
        "background": video.get("background"),
        "scenes": scenes,
    }


def _intrinsic_scene_duration(scene: dict, project: dict) -> float:
    """長さを書いていないシーンの «中身から決まる長さ»。"""
    longest = 0.0
    for layer in scene.get("layers") or []:
        start = layer.get("start") or 0
        end = layer.get("end")
        if end is None and layer.get("duration") is not None:
            end = start + layer["duration"]
        if end is not None:
            longest = max(longest, end)
        else:
            for animation in layer.get("animations") or []:
                delay = animation.get("delay") or 0
                for keyframe in animation.get("keyframes") or []:
                    longest = max(longest, (keyframe.get("time") or 0) + delay)
    if longest > 0:
        return longest
    fallback = (project.get("video") or {}).get("duration")
    return DEFAULT_DURATION if fallback is None else fallback


def _prepare_layers(layers: list, scene_duration: float, project: dict, depth: int = 0) -> list[dict]:
    """レイヤーに «シーン相対の始まり／終わり» と描画順を付ける。

    入れ子は 16 段で打ち切ります。JSON は手でも AI でも書かれるので、
    自分を含む合成を作られたときに «止まらない» のがいちばん困る壊れ方です。
    """
    if depth > 16:
        logger.warn("レイヤーの入れ子が 16 段を超えたので、そこから先は切り捨てました")
        return []
    prepared: list[dict] = []
    for index, layer in enumerate(layers):
        if not layer or layer.get("enabled") is False:
            continue
        start = layer.get("start") or 0
        end = layer.get("end")
        if end is None:
            end = start + layer["duration"] if layer.get("duration") is not None else scene_duration
        entry = dict(layer)  # 元の JSON は書き換えない（同じ project を 2 回開けるように）
        entry["id"] = layer.get("id") or f'{layer.get("type")}-{index}'
        entry["order"] = index
        entry["zIndex"] = index if layer.get("zIndex") is None else layer["zIndex"]
        entry["localStart"] = start
        entry["localEnd"] = end
        # ⚠ **`layers` と `children` の両方を受けます。**
        #
        # JSON に書く綴りは `layers`、レンダラーが読む綴りは `children` です
        # （`renderer/index.py` の `kind == "group"` の分岐）。ここで `children` を
        # 無条件に上書きしていたため、**JSON に `children` と書くと中身が消えます**。
        # しかも検証は通り、エラーも警告も出ません。«group を書いたのに何も
        # 描かれない» という、いちばん気付きにくい壊れ方をしていました
        # （最小の再現: 図形 1 枚を group の中に入れると消える）。
        nested = layer.get("layers") or layer.get("children")
        entry["children"] = (
            _prepare_layers(nested, end - start, project, depth + 1) if nested else None
        )
        prepared.append(entry)
    # zIndex が同じなら «書いた順»。安定ソートなので order を第 2 キーにするだけで足ります。
    prepared.sort(key=lambda entry: (entry["zIndex"], entry["order"]))
    return prepared


def scenes_at(timeline: dict, time: float) -> list[dict]:
    """`time` に重なっているシーンを、描く順に返す。"""
    return [
        scene
        for scene in timeline["scenes"]
        if time >= scene["start"] - _EPSILON and time < scene["end"] + _EPSILON
    ]


def is_layer_active(layer: dict, scene_time: float) -> bool:
    """そのレイヤーを `scene_time`（シーン相対）で描くかどうか。"""
    return (
        scene_time >= (layer.get("localStart") or 0) - _EPSILON
        and scene_time <= (layer.get("localEnd") or 0) + _EPSILON
    )


def time_to_frame(timeline: dict, time: float) -> int:
    """絶対時刻をフレーム番号に。キャッシュの鍵に使います。"""
    return js_round(time * timeline["fps"])


def frame_to_time(timeline: dict, frame: int) -> float:
    return frame / timeline["fps"]


def resolve_range(
    timeline: dict,
    options: dict | None = None,
    *,
    from_: float | None = None,
    to: float | None = None,
    scene: str | None = None,
) -> dict:
    """描く範囲（``--from`` / ``--to`` / ``--scene``）を決める。

    **キーワード引数でも辞書でも受けます。** JS 版はオプション辞書 1 つでしたが、
    ``movo/cli/pipeline.py`` は ``resolve_range(timeline, from_=…, to=…, scene=…)``
    と呼びます。`from` は Python の予約語なので `from_` になるぶん、JS の綴りと
    機械的には揃いません。**呼ぶ側に合わせて両方受ける** ことで、どちらで書いても
    «黙って範囲が無視される» ことが起きないようにしました。

    :returns: ``{"startFrame","endFrame","startTime","endTime"}``
    """
    options = options or {}
    if from_ is None:
        from_ = options.get("from", options.get("from_"))
    if to is None:
        to = options.get("to")
    if scene is None:
        scene = options.get("scene")

    start_time = 0.0 if from_ is None else float(from_)
    end_time = timeline["duration"] if to is None else float(to)
    if scene:
        found = next((s for s in timeline["scenes"] if s["id"] == scene), None)
        if found is None:
            available = ", ".join(s["id"] for s in timeline["scenes"])
            raise ValueError(f'scene "{scene}" was not found (available: {available})')
        start_time = found["start"]
        end_time = found["end"]
    start_time = max(0.0, start_time)
    end_time = min(timeline["duration"], max(start_time, end_time))
    start_frame = js_round(start_time * timeline["fps"])
    end_frame = max(start_frame, js_round(end_time * timeline["fps"]) - 1)
    return {
        "startFrame": start_frame,
        "endFrame": end_frame,
        "startTime": start_time,
        "endTime": end_time,
    }


def all_layers(timeline: dict) -> list[dict]:
    """入れ子も含めた «全レイヤー» の平らな一覧（検査ツール用）。"""
    out: list[dict] = []

    def walk(layers: list) -> None:
        for layer in layers:
            out.append(layer)
            if layer.get("children"):
                walk(layer["children"])

    for scene in timeline["scenes"]:
        walk(scene["layers"])
    return out


def find_layer(layers: list, layer_id: str) -> dict | None:
    """入れ子のレイヤーから id で探す。

    JS 版は ``renderer/helpers.js`` に置いてありますが、Python では
    **タイムラインが作った構造（``children``）を歩く関数**なので、その形を
    決めているこちら側に置きました。レンダラーと物理の両方から呼ばれます。
    """
    for layer in layers:
        if layer.get("id") == layer_id:
            return layer
        if layer.get("children"):
            found = find_layer(layer["children"], layer_id)
            if found is not None:
                return found
    return None


__all__ = [
    "DEFAULT_DURATION",
    "all_layers",
    "build_timeline",
    "find_layer",
    "frame_to_time",
    "is_layer_active",
    "resolve_range",
    "scenes_at",
    "time_to_frame",
]
