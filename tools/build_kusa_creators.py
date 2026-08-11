"""同じ曲を «別々の制作者» の作風で 3 パターン組み立てる。

作風は `profiles/creator-*.json` に数値で書いてあります。書き出したあと
`movo compare <映像> --target profiles/creator-xxx.json` で、狙った範囲に
入っているかを数値で確かめられます。

**既存の MV の映像は使いません。** 真似るのは «カット尺・色数・彩度・動きの量»
といった作り方の傾向だけです（`movo/core/video_compare.py` の設計と同じ考え方）。

    python tools/build_kusa_creators.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_kusa_chibi_mv import ASSETS, POSES, SONG, LRC, LYRICS  # noqa: E402,F401

from movo.audio import analyze_audio, decode_audio_file  # noqa: E402
from movo.audio.align import label_blocks, split_blocks  # noqa: E402
from movo.core import parse_lyrics, slice_lyrics  # noqa: E402

# ── 制作者 3 人ぶんの «作り方» ────────────────────────────────────
#
# それぞれ profiles/creator-*.json の目標値に対応します。
# 「何を変えるか」を 1 か所にまとめて、同じ組み立て関数へ流します。

CREATORS = {
    "yohaku": {
        "profile": "creator-yohaku",
        "label": "制作者 A「余白」",
        # 切らない。1 ブロックを 1 カットのまま流す。色は 2 色。文字は小さく端に。
        "split": 1,
        "size": 44,
        "energy": 1.15,
        "charHeight": 0.5,
        "charX": 0.16,
        "textX": 0.62,
        "textY": 0.5,
        "tints": [""],
        "tintAmount": 0.0,
        "grade": {"saturation": -0.55, "contrast": -0.1},
        "grounds": ["#17171a"],
        "cloths": ["#6f6f66"],
        "color": "#d8d6cf",
        "backgrounds": ["susuki", "grass"],
        "scrim": 0.8,
        "scrimColor": "#0d0d10",
        "alternateScrim": True,
        # 低彩度の作風は全体が中間の灰に寄り、カットの差が出ません
        # （実測の最大差 0.127 / 判定は 0.18）。振れ幅を広く取ります。
        "scrimSteps": (1.0, 0.06, 0.9, 0.03),
        # 長回しの作風でも «止まって見える» のは別問題。寄せで動きを作ります。
        "fade": 0.0, "zoom": 1.9,
    },
    "senkou": {
        "profile": "creator-senkou",
        "label": "制作者 B「閃光」",
        # 半小節で切る。蛍光色。文字は大きく全画面。写真は使わず «色面» で押す。
        # 1 ブロック（約 19 秒）を 10 に割ると 1 カット 1.9 秒。目標は 0.4〜2.5 秒。
        "split": 10,
        # 文字を大きくすると «白と黒» が同じ画面に増え、コントラストが上がります。
        "size": 104,
        "energy": 3.0,
        "charHeight": 0.78,
        "charX": 0.5,
        "textX": 0.5,
        # キャラの真上に文字を置くと «どちらも読めない»。文字は下三分の一へ。
        "textY": 0.8,
        # **カットごとに色を変えます。** 同じ絵が続くと、いくら切っても
        # «カット 1 本» としか測れません（実測でそうなりました）。
        "tints": ["#ff2e6d", "#3ad6ff", "#ffd166"],
        # 歌詞の後ろに明るい帯を敷いて «明暗の差» を作ります。色面だけだと
        # コントラストが 0.144 までしか出ません（目標は 0.200 以上）。
        "band": 0.22, "bandColor": "#fdfdf5",
        "tintAmount": 0.6,
        "grade": {"saturation": 0.5, "contrast": 0.3},
        # **どれもほぼ黒だと «切った» と分かりません。** カット検出は画面全体の
        # 差が 0.18 を超えたら 1 カットと数えるので、地の色を振り切ります。
        # **暗と明を交互に**並べます。似た明るさの色が続くと、シーンは変わって
        # いるのにフレーム間の差が 0.16〜0.17 にしかならず、カット判定の
        # 0.18 に届きません（実測しました）。
        "grounds": ["#07050c", "#ffd166", "#0a0f1e", "#3ad6ff", "#120418", "#ff5c8a"],
        "cloths": ["#ff2e6d", "#3ad6ff", "#ffd166"],
        "color": "#0a0710",
        # **写真は敷きません。** この作風は «少ない色数・高彩度» が身上で、
        # 写真を入れると実質の色数が 102 色、彩度が 0.41 まで落ちました。
        "backgrounds": [""],
        "scrim": 0.85,
        # 地の色とは別の色を重ねないと明滅が出ません。
        "scrimColor": "#050308",
        "alternateScrim": True,
        # 断絶が身上。フェードを 0 にして «切り» ます。
        "fade": 0.0, "zoom": 1.12,
    },
    "ryushi": {
        "profile": "creator-ryushi",
        "label": "制作者 C「粒子」",
        # 写真を大きく敷いてゆっくり寄る。中庸のカット。文字は写真の空いた側へ。
        # 1 ブロック（約 19 秒）を 4 に割ると 1 カット 4.7 秒。目標は 2.5〜8 秒。
        "split": 5,
        "size": 62,
        "energy": 1.25,
        "charHeight": 0.72,
        "charX": 0.24,
        "textX": 0.66,
        "textY": 0.46,
        "tints": ["#b9a26a", "#7e8c7a", "#a8785a"],
        "tintAmount": 0.18,
        "grade": {"saturation": -0.3, "contrast": 0.32},
        "grounds": ["#14171a", "#0c1410", "#150f0a"],
        "cloths": ["#b9a26a", "#7e8c7a", "#a8785a"],
        "color": "#e9e6df",
        "backgrounds": ["susuki", "grass", "tree", "sky"],
        "scrim": 0.5,
        "scrimColor": "",
        "alternateScrim": True,
        # 0.03 秒でも 2〜3 フレームに変化が散り、1 フレームの差が判定に届きません。
        "fade": 0.0, "zoom": 1.5,
    },
}


def build(name: str, out: str, *, width: int = 1280, height: int = 720, fps: int = 30) -> None:
    creator = CREATORS[name]
    analysis = analyze_audio(decode_audio_file(SONG), {"beatsPerBar": 4, "maxBeats": 0})
    bpm = round(analysis["bpm"] * 100) / 100
    bar = (60 / bpm) * 4
    duration = analysis["duration"]

    timed = parse_lyrics(Path(LRC).read_text(encoding="utf-8"))
    blocks = label_blocks(split_blocks(Path(LYRICS).read_text(encoding="utf-8")))

    spans: list[tuple[float, float, dict]] = []
    cursor = 0
    for block in blocks:
        rows = timed[cursor: cursor + len(block["lines"])]
        cursor += len(block["lines"])
        if rows:
            spans.append((rows[0]["at"], rows[-1]["at"] + rows[-1].get("for", 2.0), block))

    scenes: list[dict] = []
    #: 丸めの誤差を «次のカット» で打ち消すための、ここまでに使った小節数。
    #:
    #: 1 カットずつ独立に `round(尺 / 1 小節)` すると、誤差が後ろへ積もります。
    #: 実測で B（1 ブロックを 10 に割る作風）が **162.2 秒**になりました
    #: （曲は 153.6 秒）。8.6 秒ぶん歌詞が後ろへずれます。
    #: «曲頭からの絶対位置» を基準に取り直せば、誤差は 1 カットぶんで止まります。
    used_bars = 0.0

    def add(start: float, end: float, block: dict | None) -> None:
        nonlocal used_bars
        """1 ブロックを `split` 個のカットへ割る。**ここが作風の要**です。

        «同じ内容をいくつのカットで見せるか» がカット尺そのものなので、
        制作者ごとの違いがいちばん出ます（A は 1、B は 4）。
        """
        pieces = max(1, int(creator["split"]))
        step = (end - start) / pieces
        for piece in range(pieces):
            index = len(scenes)
            piece_start = start + piece * step
            piece_end = piece_start + step
            # 絶対位置から取り直す（誤差を積み上げない）
            bars = max(0.5, round((piece_end / bar) - used_bars, 0) * 1.0)
            bars = max(0.5, round(((piece_end / bar) - used_bars) * 2) / 2)
            used_bars += bars
            chorus = bool(block and block["kind"] == "chorus")
            pose = ("sing", "surprise", "sing", "think")[index % 4] if chorus else ("idle", "think", "surprise")[index % 3]
            backgrounds = creator["backgrounds"]
            lines = slice_lyrics(timed, piece_start, piece_end, overlap=True) if block else []
            scenes.append({
                "id": f"{name}{index:02d}",
                "use": "chibi-stage",
                "with": {
                    "lines": lines,
                    "bars": bars,
                    "bgAsset": backgrounds[index % len(backgrounds)],
                    "background": creator["grounds"][index % len(creator["grounds"])],
                    "cloth": creator["cloths"][index % len(creator["cloths"])],
                    "hair": "#5b3f2e",
                    "color": creator["color"],
                    "size": creator["size"] * (1.15 if chorus else 1.0),
                    "energy": creator["energy"],
                    "charAsset": f"char_{pose}",
                    "charHeight": creator["charHeight"] * (1.0, 0.86, 1.12, 0.94)[index % 4],
                    "charX": (creator["charX"], 1 - creator["charX"], 0.5, 1 - creator["charX"] * 0.6)[index % 4],
                    "textX": creator["textX"],
                    "textY": creator["textY"],
                    "charTint": creator["tints"][index % len(creator["tints"])],
                    "charTintAmount": creator["tintAmount"],
                    "scrim": creator["scrim"] * tuple(creator.get("scrimSteps", (1.0, 0.45, 0.85, 0.35)))[index % 4]
                    if creator.get("alternateScrim") else creator["scrim"],
                    "scrimColor": creator["scrimColor"],
                    "band": creator.get("band", 0),
                    "bandColor": creator.get("bandColor", "#ffffff"),
                    "bgSaturation": creator["grade"]["saturation"],
                    "bgContrast": creator["grade"]["contrast"],
                    "fade": creator["fade"],
                    "zoom": creator["zoom"],
                },
            })

    if spans and spans[0][0] > 1.0:
        add(0.0, spans[0][0], None)
    for start, end, block in spans:
        add(start, end, block)
    if spans and duration - spans[-1][1] > 1.0:
        add(spans[-1][1], duration, None)

    project = {
        "movoVersion": "1.0",
        "project": {"name": f'木みたいな草だった（{creator["label"]}）', "bpm": bpm, "seed": 20260801, "root": ".."},
        "video": {"width": width, "height": height, "fps": fps, "background": creator["grounds"][0]},
        "assets": {
            **{key: {"type": "image", "path": str(Path(path).resolve())} for key, path in ASSETS.items()},
            "_track": {"type": "audio", "path": SONG},
        },
        "audio": [{"asset": "_track", "volume": 0.95, "fadeOut": 2}],
        "scenes": scenes,
        "render": {"quality": "standard"},
        "output": {"format": "mp4", "codec": "h264", "crf": 21, "preset": "medium"},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f'{creator["label"]}: {out}（{len(scenes)} カット / 目標 {creator["profile"]}）')


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    for key in CREATORS:
        build(key, f"tmp/kusa-{key}.json")
