# スキル — テンプレートで動画を組む

スキルは **«入力値だけで絵になる» ひとまとまり**です。JSON をゼロから書かずに、
名前と値を渡すだけで動画になります。

```bash
movo skill list
movo skill render lyric-line --set text=夜のとばりが降りて -o tmp/lyric.mp4
```

---

## 4 つの種類

大きさの違いだけです。**下にいくほど «丸ごと» になります。**

| 種類 | 置き場 | 何が入っているか | 同梱数 |
| --- | --- | --- | --- |
| 基礎アニメーション | `animations/` | **1 レイヤーぶんの動き**（登場・退場・持続）| 28 |
| スキル | `skills/` | **レイヤー群**（文字＋飾り、天気の演出など）| 15 |
| シーン | `scenes/` | **1 カットぶん**（尺・背景・レイヤー一式）| 11 |
| ムービー | `movies/` | **1 本ぶん**（シーンの並び）| 3 |

```
ムービー（lyric-mv）
 └ シーン（mv-intro / mv-verse / mv-chorus / mv-outro）
    └ スキル（lyric-line …）
       └ 基礎アニメーション（pop-in / beat-bounce …）
```

---

## 読み込む順番

**組み込み → プロジェクト固有**の順に読み、後が優先されます。
同じ名前を置けば上書きできます。

1. `movo/skill/library/{animations,skills,scenes,movies}/*.json`（同梱）
2. `<プロジェクト>/{animations,skills,scenes,movies}/*.json`（自作）

⚠ **«プロジェクト» はどこか。** スキルはプロジェクトの根から探されます。
根は **JSON のあるディレクトリ**です。`tmp/` に置いた JSON からは
`scenes/` が見えず「スキルが見つかりません」になります。根をずらすには:

```json
{ "project": { "root": ".." } }
```

---

## 使う

### コマンドから

```bash
movo skill list --scenes            # シーンだけ一覧
movo skill show lyric-line          # 入力値と中身
movo skill render weather --set kind=sakura --preset 720p
movo skill expand title-card --set title=Movo -o project.json
movo skill new my-title --scene     # 雛形を作る
```

入力値の渡し方は 3 通り。**`--set` がいちばん強い**です。

| | |
| --- | --- |
| `--set key=value` | 何度でも。数値・`true`/`false`・JSON 配列も解釈します |
| `--with '{"text":"あ"}'` | まとめて JSON で |
| `--inputs <file.json>` | ファイルから |

### プロジェクト JSON から

```json
"scenes": [{ "use": [{ "skill": "title-card", "with": { "title": "Movo" } }] }]
"scenes": [{ "use": "mv-intro", "with": { "title": "入れ子の街", "bars": 4 } }]
"movie":  { "use": "lyric-mv", "with": { "title": "夜明けまで", "bpm": 92 } }
"layers": [{ "type": "text", "text": "あ", "use": [{ "animation": "pop-in" }] }]
```

---

## 書く

`movo skill new <名前> --scene` で雛形が出ます。中身はこの形です。

```json
{
  "skill": { "name": "chibi-stage", "kind": "scene", "title": "…", "description": "…" },
  "inputs": {
    "lines": { "type": "textList", "label": "歌詞", "default": ["…"] },
    "bars":  { "type": "number", "label": "尺（小節）", "default": 8, "min": 0.5, "max": 64 },
    "color": { "type": "color", "default": "#f4f2ea" }
  },
  "scene": {
    "duration": "${bars}bar",
    "layers": [ … ]
  }
}
```

### 入力の型

| `type` | 受けるもの |
| --- | --- |
| `text` | 文字列 |
| `textList` | 行の配列。**時刻つきの歌詞（`{text, at, for}`）もそのまま通ります** |
| `number` | 数値（`min` / `max` で範囲）|
| `color` | `#rrggbb` |
| `asset` | 素材名 |
| `enum` | `options` のどれか |
| `list` | 配列（`separator` で区切り）|

### 式（`${…}`）

**スキルを展開するときに 1 度だけ**評価されます。入力値・`width` / `height` /
`centerX` / `centerY` / `bpm` / `_id` / `_duration` が使えます。

```json
"size": "${titleSize * 0.75}",
"color": "${i % 3 == 0 ? color : accent}"
```

**実行時（毎フレーム）の式は `{"expression": "…"}`** です。こちらでは
`time` / `bpm` / `beatPulse(...)` / `wiggle(...)` / `audio` が使えます。

```json
"y": { "expression": "${height * 0.6} - beatPulse(bpm, 1, 9) * 14" }
```

⚠ **2 つは別物です。** `{"expression": "height * 0.6"}` と書くと
`unknown identifier "height"` で落ちます（`height` は展開時の変数）。
展開時の値を実行時の式に埋めたいときは、上のように `${…}` を **中に**書きます。

### 繰り返し

```json
{ "type": "text", "text": "${lyricText(line)}",
  "repeat": { "over": "${lines}", "as": "line", "indexAs": "i" } }
```

歌詞は文字列でも時刻つきの辞書でも来るので、`lyricText` / `lyricAt` /
`lyricFor` / `lyricTimed` を通してください。**書き分けると片方の形でだけ
壊れます。**

---

## 気をつけること

- **レイヤー ID は全体で一意**にしてください。`${_id}-…` を付ける約束です
  （固定の ID を書くと、同じスキルを 2 回使ったときに «重複 ID» で検証に落ちます）
- **`group` レイヤーの子は `layers`** です。`children` と書いても検証は通りますが、
  かつては中身が消えていました（現在は両方受けます）
- **文字は `fit` を付けてください。** 歌詞は曲ごとに長さが変わるので、
  «この大きさなら入る» と決め打ちできません

  ```json
  "fit": { "mode": "shrink", "maxWidth": "${width * 0.92}", "minSize": 0.3 }
  ```

---

## 一覧

```bash
movo skill list                    # 全部
movo skill list --movies           # 種類で絞る
movo skill list --category lyric   # 分類で絞る
movo skill list --grep 歌詞         # 語で探す
movo skill list --json             # 機械向け
```
