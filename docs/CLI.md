# movo コマンドリファレンス

JSON から動画を作る CLI です。**22 個のコマンド**があります。

```bash
movo <コマンド> [オプション]
```

各コマンドは `movo <コマンド> --help` でも同じ内容が読めます。

---

## やりたいことから引く

| やりたいこと | コマンド |
| --- | --- |
| とりあえず作り始める | [`init`](#init) |
| JSON が正しいか確かめる | [`validate`](#validate) |
| 動画にする | [`render`](#render) |
| **描く前に 1 枚だけ確かめる** | [`frame`](#frame) |
| 曲の BPM・拍・区間を調べる | [`analyze`](#analyze) |
| **時刻の無い歌詞に時刻を付ける** | [`lyrics align`](#lyrics-align) |
| 曲を渡して MV を 1 本作る | [`make-mv`](#make-mv) |
| 出来上がった映像を数値で見る | [`profile`](#profile) |
| 狙った作風に寄っているか確かめる | [`compare`](#compare) |
| 同じ型で何本も作る | [`batch`](#batch) |
| 使える効果・レイヤーの一覧 | [`list`](#list) |
| 環境が整っているか確かめる | [`doctor`](#doctor) |

**時間のかかることを始める前に `frame` で 1 枚見る**のがいちばん効きます。
1 本 5〜12 分かかるので、色や文字の大きさは 1 フレームで詰めてください。

---

## 共通オプション

どのコマンドでも使えます。

| | |
| --- | --- |
| `-h, --help` | ヘルプを表示 |
| `-v, --version` | バージョンを表示 |
| `--quiet` | 進捗表示を抑制 |
| `-V, --verbose` | 詳細ログ |
| `--debug` | デバッグログ（例外の追跡も出す） |
| `--json` | 結果を JSON で出力（対応コマンドのみ） |

---

## 作る・描く

### `init`

```bash
movo init <名前> [オプション]
```

新しいプロジェクトを作ります。`movo.json`・`assets/`・`output/` を用意し、
すぐ `render` できるサンプル素材も生成します。

| | |
| --- | --- |
| `--template <名前>` | `basic` \| `text` \| `physics` \| `character` \| `showcase`（既定 `basic`）|
| `--force` | 既存のディレクトリでも上書きする |
| `--width` / `--height` | 解像度（既定 1920x1080）|
| `--fps` | フレームレート（既定 30）|

### `validate`

```bash
movo validate <file> [--json] [--strict]
```

スキーマ検証と意味検証（重複 ID・未宣言の素材・式の構文）を行います。
`--strict` は警告もエラー扱いにします。

**レンダリング前に必ず通してください。** 1 本 5 分描いたあとで
「素材名が違う」と分かるのがいちばん惜しい失敗です。

### `render`

```bash
movo render <file> [オプション]
```

| | |
| --- | --- |
| `-o, --output <path>` | 出力先（既定 `output/<name>.mp4`）|
| `-f, --format <fmt>` | `mp4` \| `webm` \| `mov` \| `gif` \| `png-sequence` \| `wav` |
| `-q, --quality <名前>` | `draft` \| `preview` \| `standard` \| `high` \| `ultra` |
| `--from <秒>` / `--to <秒>` | 時間範囲 |
| `-s, --scene <id>` | 指定シーンのみ |
| `--seed <整数>` | 乱数シードを上書き |
| `--super-sample <n>` | 1〜4（アンチエイリアス倍率）|
| `--renderer <名前>` | `cpu` \| `numba` |
| `--no-cache` | キャッシュを使わない |
| `--no-generate` | AI 素材を生成しない（仮の絵を使う）|
| `--lock` | `movo.lock.json` を書き出す |
| `--variant <名前>` / `--all-variants` | アスペクト比バリアント |
| `--jobs <N\|auto>` | **区間に割って同時に描く**（`auto` はコア数 − 1、既定 1）|
| `--keep-parts` | 並列時の中間ファイルを消さない |
| `--no-audio` | 音を動画に入れない |
| `--no-check-flash` | 光過敏性発作の検査をこの 1 回だけ止める |

**`--jobs` は長い動画ほど効きます。** «同じ JSON からは同じ動画» を頼りに
フレーム境界で割り、別プロセスで描いて ffmpeg で繋ぎます（再エンコードしない
ので画質は落ちません）。割れないときは理由を言って 1 本で描きます。

```bash
movo render mv.json --jobs 5 -o tmp/mv.mp4
```

⚠ **検証に `timeout` コマンドを使わないでください。** 強制終了と
「子プロセスが落ちた」が同じメッセージになり、原因を取り違えます。

### `frame` / `frames`

```bash
movo frame  <file> [-t 秒 | --frame 番号] [-o out.png] [-q 品質]
movo frames <file> [-o dir] [--from 秒] [--to 秒] [--pattern frame_%05d.png]
```

`frame` は 1 枚だけ。**色・文字の大きさ・重なりの確認はここでやってください。**

### `preview`

```bash
movo preview <file> [-p ポート] [-q 品質] [--open]
```

ローカルサーバーを立ててブラウザでタイムラインを見ます（既定ポート 7777）。

---

## 音と歌詞

### `analyze`

```bash
movo analyze <音声ファイル> [--json]
```

BPM・1 拍目・小節・区間を推定します。WAV はそのまま、mp3 などは ffmpeg が
あれば WAV に変換して読みます。

| | |
| --- | --- |
| `--min-bpm` / `--max-bpm` | 探す範囲（既定 60〜240）|
| `--beats-per-bar <数>` | 1 小節の拍数（既定 4）|

プロジェクトから使うときは `"project": { "bpm": { "fromAudio": "<素材名>" } }`。

### `lyrics align`

```bash
movo lyrics align <音声ファイル> --text <歌詞ファイル> [オプション]
```

**時刻の無い歌詞に «下書きの時刻» を付けて `.lrc` にします。**

| | |
| --- | --- |
| `--text <ファイル>` | 歌詞（空行でブロックに割る。繰り返すブロック＝サビ）|
| `--lines "…"` | 歌詞を直接（`\n` 区切り）|
| `--anchor <行=秒>` | **この行はこの時刻、と留める**（何点でも。行番号は 1 始まり）|
| `--start` / `--end` | 歌う範囲を秒で直接指定 |
| `--no-snap` | 拍に寄せない |
| `--no-gaps` | 間奏を避けず、曲全体に均等に配る |
| `--scenario <path>` | シナリオの下書き（JSON）の出力先 |
| `-o, --output <path>` | `.lrc` の出力先 |

**完全自動ではありません。** 出るのは下書きです。ずれていたら `--anchor` で
2〜3 点だけ留めてください。留めた点のあいだが配分し直されます。

```bash
movo lyrics align song.mp3 --text lyrics.txt -o song.lrc
movo lyrics align song.mp3 --text lyrics.txt --anchor 1=5.4 --anchor 17=88.0
```

仕組みは [歌詞と曲を合わせる.md](歌詞と曲を合わせる.md) に書いてあります。

### `make-mv`

```bash
movo make-mv <音声ファイル> [オプション]
```

曲を渡すと、その曲に合わせた MV を 1 本作ります。BPM・1 拍目・区間を解析し、
カット尺を **小節**で決めるので、曲を差し替えればカット割りが追従します。

| | |
| --- | --- |
| `--title <曲名>` | 曲名 |
| `--lines <歌詞>` | 歌詞（改行区切り）|
| `--lyrics <ファイル>` | **時刻つきの歌詞**（`.lrc` / `.srt` / `.vtt` / JSON）|
| `--asset 名前=パス` | 画像などを素材として渡す（何度でも）|
| `--style <スキル名>` | ムービースキル（既定 `lyric-mv`／激しいのは `hype-lyric-mv`）|
| `--intensity <0〜1>` | 勢いの強い A メロを激しいシーンに寄せる（既定 0）|
| `--max-bars <数>` | 1 カットの上限（既定 8 小節）|
| `--beats-per-bar <数>` | 1 小節の拍数（既定 4）|
| `--min-bpm` / `--max-bpm` | BPM の探索範囲 |
| `-o, --output <path>` | 出力先 |

`--lyrics` を渡すと、小節数で機械的に配るのをやめ、**実際に歌われる時刻**で
カットへ割り当てます。`--intensity` は決め（サビ）と入り・終わりを変えません
（ずっと激しいと «うるさいだけ» になるので、落差を残すのが要点です）。

---

## 測る・寄せる

### `profile`

```bash
movo profile <動画 または プロジェクト> [--json] [--width 320] [--fps 24]
```

映像を数値にします — カット尺・動きの量・実質の色数・彩度・コントラスト・
細かさ。プロジェクト JSON はその場で描いて測るので、書き出し前でも回せます。

### `compare`

```bash
movo compare <自分の映像> [相手の映像] [--target <名前 または ファイル>]
```

測った数値を目標と突き合わせ、外れた項目の **直し方**まで出します。

| | |
| --- | --- |
| `--target <名前>` | 同梱の目標値（`movo list profiles`）またはファイルパス |
| `--tolerance <数>` | 相手の映像とくらべるときの許容幅（既定 0.25）|

```bash
movo compare tmp/mv.mp4 --target profiles/creator-yohaku.json
```

自作の目標値は `<プロジェクト>/profiles/*.json` に置くと名前で呼べます。
詳細は [profile.ja.md](profile.ja.md)。

---

## スキル（テンプレート）

### `skill`

```bash
movo skill <list|show|render|expand|new> [名前] [オプション]
```

| サブコマンド | |
| --- | --- |
| `list` | 一覧。`--animations` / `--skills` / `--scenes` / `--movies` / `--category` / `--tag` / `--grep` |
| `show <名前>` | 入力値・生成物・学習元を表示 |
| `render <名前>` | スキル単体で動画にする |
| `expand <名前>` | 素の Movo JSON に展開する |
| `new <名前>` | 雛形を作る（`--animation` / `--scene` / `--movie`）|

入力値は `--set key=value`（何度でも）・`--with '{"text":"あ"}'`・
`--inputs <file.json>` で渡します。出力は `-o` と `--preset`
（`1080p` / `720p` / `480p` / `square` / `vertical` / `shorts` / `thumb`）。

```bash
movo skill render lyric-line --set text=夜のとばりが降りて -o tmp/lyric.mp4
movo skill expand title-card --set title=Movo -o project.json
```

定義は **組み込み → プロジェクト固有**の順に読み、後が優先されます。

1. `movo/skill/library/{animations,skills,scenes,movies}/*.json`
2. `<プロジェクト>/{animations,skills,scenes,movies}/*.json`

⚠ **プロジェクト側のスキルを使うときは、JSON の置き場所に注意してください。**
スキルは «プロジェクトの根» から探されます。根は JSON のあるディレクトリなので、
`tmp/` に置いた JSON から `scenes/` は見えません。`"project": { "root": ".." }`
と書けば根をずらせます。

詳細は [skills.ja.md](skills.ja.md)。

---

## まとめて作る

### `batch`

```bash
movo batch <テンプレート> --input <表> --out <パターン> [--jobs N] [--continue]
```

«1 つのテンプレート × N 通りの入力値» を連番で書き出します。表は JSON の配列か
CSV で、1 行 1 本です。`--out` には `{name}` `{index}` `{basename}` と表の
列名が使えます。`--continue` は書き出し済みを飛ばします（中断からの再開）。

### `make` / `params`

```bash
movo make <recipe> [--set key=value] [-o out.mp4]
movo params <file> [--json]
```

`make` は `--save-recipe` で保存した «作り方» からもう一度作ります。
`params` はそのプロジェクトで差し替えられる項目を一覧します。

---

## 素材・環境

### `assets`

```bash
movo assets <generate|plan> <file> [--json] [--force]
```

`generate` は宣言された AI 素材を生成して `assets/generated/` へ、
`plan` は生成内容（プロバイダー・プロンプト・出力先・キャッシュ状況）を
**API を呼ばずに**表示します。

### `list`

```bash
movo list <種類> [--json]
```

種類: `effects` / `deformers` / `physics` / `modulators` / `easings` /
`functions` / `layers` / `formats` / `masks` / `renderers` / `particles` /
`presets` / `profiles` / `blends`

### `plugin`

```bash
movo plugin list
movo plugin create <名前>
```

`plugins` / `skills` と複数形でも書けます（指が勝手に s を付けるため、
JS 版から両方受けています）。

### `config`

```bash
movo config <set|get|list|unset|path> [key] [value]
```

API キーは `~/.movo/config.json` に 600 で保存され、プロジェクト JSON には
入りません。環境変数 `MOVO_OPENAI_API_KEY` / `MOVO_GEMINI_API_KEY` があれば
そちらが優先されます。

```bash
movo config set openai.apiKey <キー>
movo config set gemini.apiKey <キー>
```

### `doctor`

```bash
movo doctor [--json]
```

Python・OS・CPU・メモリ・ffmpeg・フォント・NumPy / Numba・キャッシュ、
そして «移植がどこまで繋がっているか» を診断します。

---

## 終了コード

**エラーの種類ごとに違う値を返します。** «失敗した» ことだけでなく
«何で失敗したか» をスクリプト側で分けられるようにするためです
（`movo/cli/errors.py` の `EXIT_CODES`）。

| | |
| --- | --- |
| 0 | 成功 |
| 1 | 上記以外の失敗 |
| 2 | 使い方の誤り（`MOVO_CLI_USAGE`）|
| 3 | JSON が不正（`MOVO_SCHEMA_INVALID`）|
| 4 | 素材が見つからない（`MOVO_ASSET_NOT_FOUND`）|
| 5 | 式が不正（`MOVO_EXPRESSION_INVALID`）|
| 6 | API キーの認証に失敗（`MOVO_PROVIDER_AUTH_FAILED`）|
| 7 | プラグインが無い／拒否（`MOVO_PLUGIN_NOT_FOUND` / `MOVO_PLUGIN_DENIED`）|
| 8 | ffmpeg が無い（`MOVO_FFMPEG_NOT_FOUND`）|
| 9 | メモリ不足（`MOVO_OUT_OF_MEMORY`）|
| 130 | 中断（Ctrl-C）|
