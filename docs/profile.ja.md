# 作風を数値で見る — `profile` と `compare`

「なんとなく違う」を **数字にして、直し方まで出す**ための仕組みです。

```bash
movo profile tmp/mv.mp4
movo compare tmp/mv.mp4 --target profiles/creator-yohaku.json
```

---

## 測る 8 つの指標

| 指標 | 何を見ているか | どこから |
| --- | --- | --- |
| `cutSeconds` | カット尺（中央値）| `cuts.medianSeconds` |
| `cutsPerMinute` | 毎分のカット数 | `cuts.perMinute` |
| `motion` | 動きの量 | `motion.mean` |
| `stillRatio` | 止まっている割合 | `motion.stillRatio` |
| `colors` | 実質の色数（全体の 90% を占めるのに要る色数）| `palette.effectiveColors` |
| `saturation` | 彩度 | `palette.saturation` |
| `contrast` | 明暗の広がり | `palette.contrast` |
| `detail` | 細かさ（文字や模様の多さ）| `detail.edgeDensity` |

**キー名は JS 版と揃えてあります**（`profiles/*.json` を共有できるように）。

---

## ⚠ カット検出の癖

**画面全体の明るさが変わらないと、カットとして数えません。**

- 差を測るのは **24 x 13 に縮めた輝度の格子**
- そこでの平均差が **0.18 以上**で 1 カット

つまり写真から別の写真へ切り替えても、どちらも同じくらいの明るさなら **0 本**です。
実際に «30 秒で 5 回切っているのに 1 本» と出ました。

対処は 2 つ。

1. **暗幕（スクリム）の濃さをカットごとに変える。** 画面全体の明るさそのものを
   上下させれば数えられます
2. **繋ぎのフェードを詰める。** 0.2 秒のフェードは変化を 10 フレーム前後に散らし、
   1 フレームぶんの差が判定に届きません

```python
# カットごとに暗幕を変える例
"scrim": base * (1.0, 0.06, 0.9, 0.03)[index % 4]
```

低彩度の作風では特に効きます（画面が中間の灰に寄って差が出ないため）。

---

## 目標値（プロファイル）

作風を «数値の範囲» で書いたものです。

```json
{
  "name": "creator-yohaku",
  "label": "制作者 A「余白」— 切らない・低彩度・文字は小さく整列",
  "note": "切らずに横へ流す。色は 2 色。……",
  "target": {
    "cutSeconds": [6, 30],
    "cutsPerMinute": [2, 10],
    "motion": [0.003, 0.022],
    "stillRatio": [0, 0.8],
    "colors": [8, 48],
    "saturation": [0.05, 0.28],
    "contrast": [0.1, 0.35],
    "detail": [0.012, 0.045]
  }
}
```

置き場は **組み込み → プロジェクト固有**の順です。

1. `movo/library/profiles/*.json`（同梱 10 種）
2. `<プロジェクト>/profiles/*.json`（自作）

```bash
movo list profiles                                   # 使えるもの
movo compare tmp/mv.mp4 --target creator-yohaku      # 名前で
movo compare tmp/mv.mp4 --target profiles/x.json     # パスで
```

⚠ `--target` を **名前**で指定したときの探索は «その動画のあるディレクトリ» が
基準です。`tmp/` に置いた動画からは `profiles/` が見えないので、**パスで
渡すほうが確実**です。

**参考にした映像そのものは同梱も配布もしません**（著作物のため）。真似るのは
カット尺・色数・彩度・動きの量といった «作り方の傾向» だけです。手元に権利の
ある映像があれば、それを `movo profile` で測って目標にできます。

---

## 出力の読み方

```
! 1 項目が目標から外れています

  ✔ カット尺　　　　　　 1.62秒（目標 0.400〜2.50秒）
  ✔ 毎分のカット数　　　 30.15本（目標 24〜150本）
  ✖ コントラスト　　　　 0.144（目標 0.200〜0.500）
      明暗の差が足りません。……
```

外れた項目には **Movo での直し方**が付きます。ここがこの機能の値打ちです。

---

## 数値と目視は別物

`movo compare` が 8 / 8 でも «読めるか» は測っていません。実際に、全項目が
目標内なのに **歌詞がキャラの体に重なって両方読めない**カットが残りました。

**1 本につき 3 枚ほどフレームを抜いて見る**手順を必ず入れてください。

```bash
ffmpeg -v error -y -ss 72 -i tmp/mv.mp4 -frames:v 1 tmp/check.png
```

描く前なら `movo frame` のほうが速いです（1 本 5〜12 分に対して数秒）。
