"""「木みたいな草だった」の MV を `chibi-stage` で組み立てる。

`make-mv` は «区間 → ムービースキルの並び» に落とす道具で、シーンの名前を
役割（`intro` / `verse` / `chorus` …）で引きます。ここでやりたいのは
**歌詞ブロック 1 つ ＝ シーン 1 つ**という別の割り方なので、プロジェクトを
直接組み立てます。

こうする利点が 2 つあります。

  1. **歌詞の切れ目とカットの切れ目が必ず一致する。** 区間から割ると
     «サビの途中でカットが変わる» が起きます
  2. 背景の写真とキャラの色をブロックごとに変えられるので、**同じ絵が
     15 カット続く**（rich-mv でそうなりました）のを避けられます

    python tools/build_kusa_chibi_mv.py tmp/kusa-chibi.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movo.audio import analyze_audio, decode_audio_file  # noqa: E402
from movo.audio.align import label_blocks, split_blocks  # noqa: E402
from movo.core import parse_lyrics, slice_lyrics  # noqa: E402

# 曲のファイル。手元の場所は人によって違うので、環境変数で渡せるようにしてある。
#   MOVO_DEMO_SONG=/path/to/song.mp3 python tools/build_kusa_chibi_mv.py
SONG = os.environ.get("MOVO_DEMO_SONG", "tmp/song.mp3")
LRC = "tmp/kusa.lrc"
LYRICS = "tmp/kusa-lyrics.txt"

#: ブロックごとに変える «見た目»。同じ絵が続くのがいちばん退屈なので、
#: 背景・服・髪をブロック単位で回します。サビは動きを強め、文字も大きく。
LOOKS = [
    {"bg": "sky",    "cloth": "#3f6fd8", "hair": "#5b3f2e", "accentBg": "#101512"},
    {"bg": "susuki", "cloth": "#d8543f", "hair": "#2e2a3f", "accentBg": "#141008"},
    {"bg": "grass",  "cloth": "#3fb27a", "hair": "#5b3f2e", "accentBg": "#0c1410"},
    {"bg": "tree",   "cloth": "#c9a227", "hair": "#3f2e2e", "accentBg": "#150f0a"},
]

#: 背景の «並び»。`index % 4` で回すと 4 つおきに同じ絵が来ます（実際に
#: 20 秒と 105 秒がまったく同じ画になりました）。**離れたカットほど違う絵に
#: したい**ので、周期にせず並びを書き下します。足りなければ折り返します。
BG_ORDER = ["sky", "susuki", "grass", "tree", "susuki", "sky", "tree", "grass", "sky", "tree"]

#: 服の色も同じ理由で並びにします。
CLOTH_ORDER = ["#3f6fd8", "#3fb27a", "#d8543f", "#c9a227", "#7a5fd8", "#3fb2a8", "#d88a3f", "#5f8fd8", "#3f6fd8", "#c9a227"]

#: 地の色（暗幕）。背景写真の明るさに合わせます。
GROUND_ORDER = ["#101512", "#141008", "#0c1410", "#150f0a", "#141008", "#101512", "#150f0a", "#0c1410", "#101512", "#150f0a"]

#: 生成したキャラのポーズ。ブロックの «役割» で使い分けます。
POSES = {"verse": "idle", "chorus": "sing", "bridge": "think", "intro": "surprise"}

ASSETS = {
    "sky": "assets/free/sky.png",
    "susuki": "assets/free/susuki.png",
    "grass": "assets/free/grass.png",
    "tree": "assets/free/tree.png",
    "char_idle": "assets/character/idle.png",
    "char_sing": "assets/character/sing.png",
    "char_think": "assets/character/think.png",
    "char_surprise": "assets/character/surprise.png",
}


def build(out: str, *, width: int = 1280, height: int = 720, fps: int = 30) -> None:
    analysis = analyze_audio(decode_audio_file(SONG), {"beatsPerBar": 4, "maxBeats": 0})
    bpm = round(analysis["bpm"] * 100) / 100
    bar = (60 / bpm) * 4
    duration = analysis["duration"]

    timed = parse_lyrics(Path(LRC).read_text(encoding="utf-8"))
    blocks = label_blocks(split_blocks(Path(LYRICS).read_text(encoding="utf-8")))

    # 行 → ブロックの対応（`.lrc` は行の並びが歌詞と同じなので順に配れます）
    spans: list[tuple[float, float, dict]] = []
    cursor = 0
    for block in blocks:
        count = len(block["lines"])
        rows = timed[cursor: cursor + count]
        cursor += count
        if not rows:
            continue
        start = rows[0]["at"]
        end = rows[-1]["at"] + rows[-1].get("for", 2.0)
        spans.append((start, end, block))

    scenes: list[dict] = []
    #: 丸めの誤差を «次のカット» で打ち消すための、ここまでに使った小節数。
    #:
    #: 1 カットずつ独立に `round(尺 / 1 小節)` すると、誤差が後ろへ積もります。
    #: 実測で B（1 ブロックを 10 に割る作風）が **162.2 秒**になりました
    #: （曲は 153.6 秒）。8.6 秒ぶん歌詞が後ろへずれます。
    #: «曲頭からの絶対位置» を基準に取り直せば、誤差は 1 カットぶんで止まります。
    used_bars = 0.0

    def add(start: float, end: float, block: dict | None, index: int) -> None:
        nonlocal used_bars
        bars = max(0.5, round(((end / bar) - used_bars) * 2) / 2)
        used_bars += bars
        slot = index % len(BG_ORDER)
        chorus = bool(block and block["kind"] == "chorus")
        # ポーズはブロックの役割で選ぶ。サビは «歌っている» 絵にする。
        if block is None:
            pose = "surprise"
        elif chorus:
            pose = "sing"
        else:
            pose = ("idle", "think", "surprise", "idle", "think")[index % 5]
        char_x = (0.22, 0.78, 0.5, 0.78, 0.22)[index % 5]
        lines = slice_lyrics(timed, start, end, overlap=True) if block else []
        scenes.append({
            "id": f"s{index:02d}",
            "use": "chibi-stage",
            "with": {
                "lines": lines,
                "bars": bars,
                "bgAsset": BG_ORDER[slot],
                "background": GROUND_ORDER[slot],
                "cloth": CLOTH_ORDER[slot],
                "hair": "#5b3f2e",
                "color": "#fdfbf4",
                "size": 76 if chorus else 62,
                "energy": 1.7 if chorus else 1.0,
                "charAsset": f"char_{pose}",
                "charHeight": 0.78 if chorus else 0.70,
                "fade": 0.1,
                "zoom": 1.2,

                # 左右も 3 通りに散らす（2 通りだと «左・右・左» で単調になります）
                "charX": char_x,
                # **歌詞はキャラの反対側へ。** 位置を別々に決めていたため、
                # キャラが中央（0.5）のカットで文字が体の上に重なり、どちらも
                # 読めなくなっていました（130 秒地点で実際にそうなりました）。
                "textX": 0.72 if char_x < 0.5 else (0.28 if char_x > 0.5 else 0.5),
                # 中央に立つカットだけは横に逃がせないので、下へ降ろします。
                "textY": 0.5 if char_x != 0.5 else 0.84,
            },
        })

    # イントロ（歌が始まるまで）
    if spans and spans[0][0] > 1.0:
        add(0.0, spans[0][0], None, 0)
    for index, (start, end, block) in enumerate(spans, start=1):
        # 前のシーンとの間（間奏）は、直前のシーンに足さず «歌詞なし» のカットにする
        previous_end = scenes[-1].get("_end") if scenes else None
        add(start, end, block, index)
        scenes[-1]["_end"] = end
    # アウトロ
    if spans and duration - spans[-1][1] > 1.0:
        add(spans[-1][1], duration, None, len(spans) + 1)

    for scene in scenes:
        scene.pop("_end", None)

    project = {
        "movoVersion": "1.0",
        # `root` を書かないと «プロジェクトの置き場所 ＝ JSON のあるディレクトリ»
        # になり、`tmp/` に置いた JSON からは `scenes/chibi-stage.json` が見つかりません
        # （"スキル chibi-stage が見つかりません" になりました）。
        "project": {"name": "木みたいな草だった", "bpm": bpm, "seed": 20260801, "root": ".."},
        "video": {"width": width, "height": height, "fps": fps, "background": "#0b0f0c"},
        "assets": {
            **{name: {"type": "image", "path": str(Path(path).resolve())} for name, path in ASSETS.items()},
            "_track": {"type": "audio", "path": SONG},
        },
        "audio": [{"asset": "_track", "volume": 0.95, "fadeOut": 2}],
        "scenes": scenes,
        "render": {"quality": "standard"},
        # CRF 既定の 18 はほぼ可逆で、グレインを全部符号化して 650MB になります。
        "output": {"format": "mp4", "codec": "h264", "crf": 21, "preset": "medium"},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(s["with"]["bars"] for s in scenes) * bar
    print(f"{out} を書きました（シーン {len(scenes)} 個 / 合計 {total:.1f} 秒 / 曲 {duration:.1f} 秒）")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    build(sys.argv[1] if len(sys.argv) > 1 else "tmp/kusa-chibi.json")
