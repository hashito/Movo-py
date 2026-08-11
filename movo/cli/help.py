"""ヘルプ本文（日本語）。JS 版の `packages/cli/src/help.js` と同じ内容です。"""

from __future__ import annotations

from movo import __version__

from .console import style

MAIN_HELP = f"""{style.bold('movo')} {__version__} — JSON から動画を生成する CLI（Python 版）

{style.bold('使い方')}
  movo <コマンド> [オプション]

{style.bold('コマンド')}
  init <名前>              新しいプロジェクトを作成する
  validate <file>          JSON を検証する
  render <file>            動画を書き出す
  frame <file>             指定時刻の 1 フレームを PNG で書き出す
  frames <file>            連番 PNG を書き出す
  preview <file>           ブラウザでプレビューする
  assets generate <file>   AI 素材を生成する
  assets plan <file>       生成される素材を確認する（API 呼び出しなし）
  analyze <音声>           曲の BPM・拍・小節・区間を調べる
  lyrics align <音声>      時刻の無い歌詞に、曲に合わせた時刻を付ける（.lrc）
  make-mv <音声>           曲を渡すと、その曲に合わせた MV を 1 本作る
  profile <映像>           映像を数値にする（カット尺・動き・色数・細かさ）
  compare <映像>           目標や別の映像とくらべ、直し方まで出す
  skill list               スキル・シーン・ムービー・基礎アニメーションの一覧
  skill show <名前>        入力値と中身を表示する
  skill render <スキル名>   スキル単体で動画を書き出す
  skill expand <スキル名>   スキルを素の JSON に展開する
  skill new <名前>         雛形を作る（--animation / --scene / --movie）
  params <file>            差し替えられる項目（params）の一覧
  make <recipe>            保存した «作り方» からもう一度作る
  batch <テンプレート>      表形式の入力値から連番で書き出す
  list <種類>              effects / deformers / physics / modulators / easings / functions / layers / formats / particles / profiles
  plugin list              読み込まれるプラグインを表示する
  plugin create <名前>     プラグインの雛形を作る
  config set <key> [値]    設定（API キーなど）を保存する
  config get <key>         設定を表示する
  config list              設定一覧を表示する
  doctor                   実行環境を診断する

{style.bold('共通オプション')}
  -h, --help               ヘルプを表示
  -v, --version            バージョンを表示
      --quiet              進捗表示を抑制
  -V, --verbose            詳細ログ
      --debug              デバッグログ
      --json               結果を JSON で出力（対応コマンドのみ）

{style.bold('例')}
  movo init my-video
  movo render movo.json --output output.mp4
  movo render movo.json --from 3 --to 8 --quality high
  movo render movo.json --scene opening
  movo render movo.json --jobs auto        # 区間に割って同時に描く（長い曲向け）
  movo frame movo.json --time 3.5 --output frame.png
  movo frames movo.json --output ./frames
  movo list deformers
  movo skill list
  movo skill render cutin-title --set text=サビ --set bpm=174 -o tmp/cutin.mp4
  movo skill render title-card --set title=Movo --preset 720p

詳しいドキュメント: docs/ 以下（日本語）
"""

COMMAND_HELP: dict[str, str] = {
    "skill": f"""movo skill <サブコマンド> [オプション]

  スキル（レイヤー群のテンプレート）と基礎アニメーション（1 レイヤー分の動き）を扱います。
  スキル用 JSON と入力値だけで動画が作れます。定義は次の順で読み込まれ、後が優先されます。
    1. movo/skill/library/{{animations,skills,scenes,movies}}/*.json（組み込み）
    2. <プロジェクト>/{{animations,skills,scenes,movies}}/*.json（自作・上書き）

  {style.bold('サブコマンド')}
    list                   一覧。--animations / --skills / --scenes / --movies / --category <名前> / --tag <名前> / --grep <語> / --json
    show <名前>             入力値・生成物・学習元を表示。--json で定義そのまま
    render <スキル名>        スキル単体で動画にする
    expand <スキル名>        素の Movo JSON に展開する（-o でファイルへ）
    new <名前>              雛形を作る。--animation / --scene / --movie で種類を選ぶ

  {style.bold('入力値の渡し方')}
    --set key=value         何度でも指定できる（数値・true/false・JSON 配列も解釈）
    --with '{{"text":"あ"}}'   まとめて JSON で渡す
    --inputs <file.json>    ファイルから読む（--set が優先）

  {style.bold('出力とサイズ')}
    -o, --output <path>     既定は tmp/<スキル名>.mp4
    -f, --format <fmt>      mp4 | webm | gif | png-sequence
        --preset <名前>      1080p | 720p | 480p | square | vertical | shorts | thumb
        --width / --height / --fps / --duration / --bpm / --seed / --quality / --background

  {style.bold('例')}
    movo skill list --skills
    movo skill show lyric-line
    movo skill render lyric-line --set text=夜のとばりが降りて -o tmp/lyric.mp4
    movo skill render weather --set kind=sakura --preset 720p
    movo skill expand title-card --set title=Movo -o project.json
    movo skill new my-title

  {style.bold('プロジェクト JSON から使う')}
    "scenes": [{{ "use": [{{ "skill": "title-card", "with": {{ "title": "Movo" }} }}] }}]
    "scenes": [{{ "use": "mv-intro", "with": {{ "title": "入れ子の街", "bars": 4 }} }}]
    "movie":  {{ "use": "lyric-mv", "with": {{ "title": "夜明けまで", "bpm": 92 }} }}
    "layers": [{{ "type": "text", "text": "あ", "use": [{{ "animation": "pop-in" }}] }}]

  詳細: docs/skills.ja.md
""",
    "init": """movo init <名前> [オプション]

  新しい Movo プロジェクトを作成します。movo.json、assets/、output/ などを用意し、
  すぐに render できるサンプル素材も生成します。

  --template <名前>   basic | text | physics | character | showcase（既定: basic）
  --force             既存のディレクトリでも上書きする
  --width, --height   解像度（既定 1920x1080）
  --fps               フレームレート（既定 30）
""",
    "validate": """movo validate <file> [オプション]

  スキーマ検証と意味検証（重複 ID、未宣言素材、式の構文など）を行います。

  --json              結果を JSON で出力
  --strict            警告もエラー扱いにする
""",
    "render": """movo render <file> [オプション]

  -o, --output <path>   出力先（既定: output/<name>.mp4）
  -f, --format <fmt>    mp4 | webm | mov | gif | png-sequence | wav
  -q, --quality <名前>  draft | preview | standard | high | ultra
      --from <秒>       開始時刻
      --to <秒>         終了時刻
  -s, --scene <id>      指定シーンのみ出力
      --seed <整数>     乱数シードを上書き
      --super-sample <n> 1..4（アンチエイリアス倍率）
      --renderer <名前> cpu | numba（未対応時は自動フォールバック）
      --no-cache        キャッシュを使わない
      --no-generate     AI 素材の生成を行わない（プレースホルダを使う）
      --lock            movo.lock.json を書き出す
      --variant <名前>  アスペクト比バリアントで出す（project.variants）
      --all-variants    全バリアントを出す（-o "tmp/{name}-{variant}.mp4"）
      --jobs <N|auto>   区間に割って同時に描く（auto はコア数 - 1、既定は 1）
      --keep-parts      並列時の区間ごとの中間ファイルを消さない
      --no-audio        音を動画に入れない（音に反応する動きはそのまま）
      --no-check-flash  光過敏性発作の検査をこの 1 回だけ止める

  --jobs は «同じ JSON からは同じ動画» を頼りに、フレーム境界で区間に割って
  別々のプロセス（multiprocessing）で描き、ffmpeg で繋ぎます（再エンコードしない
  ので画質は落ちません）。長い曲ほど効きます。gif や ffmpeg の無い環境など、
  割れないときは理由を言って 1 本で描きます。閃光の検査は、繋いだあとの 1 本を
  まとめて 1 回だけ回します（区間ごとに検査すると繋ぎ目の明滅を見落とすため）。

  movo render mv.json --jobs 12         # 12 並列
  movo render mv.json --jobs auto       # コア数 - 1
""",
    "frame": """movo frame <file> [オプション]

  -t, --time <秒>       書き出す時刻（既定 0）
      --frame <番号>    フレーム番号で指定（--time より優先）
  -o, --output <path>   出力 PNG（既定: output/frame-<番号>.png）
  -q, --quality <名前>  品質プリセット
""",
    "frames": """movo frames <file> [オプション]

  -o, --output <dir>    出力ディレクトリ（既定: output/frames）
      --from / --to     時間範囲
      --pattern <名前>  ファイル名（既定 frame_%05d.png）
""",
    "preview": """movo preview <file> [オプション]

  ローカルサーバーを立ち上げ、ブラウザでタイムラインを確認できます。

  -p, --port <番号>     ポート（既定 7777）
  -q, --quality <名前>  既定は preview
      --open            起動後にブラウザを開く
""",
    "assets": """movo assets <generate|plan> <file>

  generate  宣言された AI 素材を生成し assets/generated/ に保存する
  plan      生成内容（プロバイダー、プロンプト、出力先、キャッシュ状況）を表示する

  --json    JSON で出力
  --force   キャッシュを無視して再生成する
""",
    "list": """movo list <effects|deformers|physics|modulators|easings|functions|layers|formats|masks|renderers|particles|presets|profiles|blends>

  --json    JSON で出力
""",
    "plugin": """movo plugin <list|create> [名前]

  list             プロジェクトが読み込むプラグインを一覧表示
  create <名前>    plugins/<名前>/__init__.py に雛形を作成
""",
    "config": """movo config <set|get|list|unset|path> [key] [value]

  例:
    movo config set openai.apiKey            （値は対話入力ではなく引数で渡します）
    movo config set gemini.apiKey sk-...
    movo config list

  API キーは ~/.movo/config.json に 600 で保存され、プロジェクト JSON には保存されません。
  環境変数 MOVO_OPENAI_API_KEY / MOVO_GEMINI_API_KEY が設定されていればそちらが優先されます。
""",
    "profile": """movo profile <動画 または プロジェクト>

  映像を数値にします。カット尺・動きの量・実質の色数・彩度・コントラスト・
  細かさを出します。プロジェクト JSON はその場で描いて測るので、書き出し前でも回せます。

  --json               結果を JSON で出力
  --width <数>         測る幅（既定 320。小さいほど速く、指標はほぼ変わらない）
  --fps <数>           測るフレームレート（既定 24）
  --quality <名前>     プロジェクトを測るときの品質（既定 draft）

  動画を読むには ffmpeg が必要です（プロジェクト JSON なら不要）。
""",
    "compare": """movo compare <自分の映像> [相手の映像] [--target <名前 または ファイル>]

  測った数値を目標と突き合わせ、外れた項目の «直し方» まで出します。

  --target <名前>      同梱の目標値（movo list profiles で一覧）またはファイルパス
  --tolerance <数>     相手の映像とくらべるときの許容幅（既定 0.25 = ±25%）
  --json               結果を JSON で出力

  例:
    movo compare tmp/mv/03.mp4 --target geometric
    movo compare examples/vocaloid-mv/03-geometric.json --target geometric
    movo compare tmp/mine.mp4 tmp/reference.mp4

  自作の目標値は <プロジェクト>/profiles/*.json に置くと名前で呼べます。
  詳細: docs/profile.ja.md
""",
    "make-mv": """movo make-mv <音声ファイル> [オプション]

  曲を渡すと、その曲に合わせた MV を 1 本作ります。BPM・1 拍目・区間を解析し、
  カット尺を «小節» で決めるので、曲を差し替えればカット割りが勝手に追従します。

  --title <曲名>       曲名
  --lines <歌詞>       歌詞（改行区切り）
  --lyrics <ファイル>  **時刻付きの歌詞**（.lrc / .srt / .vtt / JSON）。これを渡すと
                       小節数で機械的に配るのをやめ、実際に歌われる時刻で割り当てます
                       （時刻が無ければ movo lyrics align で下書きを作れます）
  --style <スキル名>   ムービースキル（既定 lyric-mv / 激しいのは hype-lyric-mv）
  --asset 名前=パス    画像などを素材として渡す（何度でも）。rich-mv なら
                       --asset art=写真.png --set artAsset=art で背景に敷けます
  --intensity <0〜1>   勢いの強い A メロを激しいシーンに寄せる（既定 0 = 従来どおり）
  --max-bars <数>      1 カットの上限（既定 8 小節）
  --beats-per-bar <数> 1 小節の拍数（既定 4）
  --min-bpm / --max-bpm  BPM の探索範囲
  -o, --output <path>  出力先

  例:
    movo make-mv song.wav --title 夜明けまで -o tmp/mv.mp4
    movo make-mv song.wav --style hype-lyric-mv --intensity 0.7 -o tmp/hot.mp4

  --intensity は **決め（サビ）と入り・終わりを変えません**。ずっと激しいと
  «うるさいだけ» になるので、落差を残すのがこの指定の要点です。
""",
    "analyze": """movo analyze <音声ファイル>

  曲の BPM・1 拍目・小節・区間を推定します。WAV はそのまま、mp3 などは
  ffmpeg があれば WAV に変換して読みます。

  --json               解析結果を JSON で出力
  --min-bpm <数>       探す BPM の下限（既定 60）
  --max-bpm <数>       探す BPM の上限（既定 240）
  --beats-per-bar <数> 1 小節の拍数（既定 4）

  プロジェクトから使うには: "project": { "bpm": { "fromAudio": "<素材名>" } }
""",
    "lyrics": """movo lyrics align <音声ファイル> --text <歌詞ファイル>

  時刻の無い歌詞に、曲に合わせた «下書きの時刻» を付けて .lrc にします。

  --text <ファイル>    歌詞（空行でブロックに割ります。繰り返すブロック＝サビ）
  --lines "…"          歌詞を直接（\\n 区切り）
  --anchor <行=秒>     この行はこの時刻、と留める（何点でも。1 始まりの行番号）
  --start / --end      歌う範囲を秒で直接指定する
  --no-snap            拍に寄せない
  --no-gaps            間奏を避けず、曲全体に均等に配る
  --scenario <path>    シナリオの下書き（JSON）の出力先
  --json               結果を JSON で出力
  -o, --output <path>  .lrc の出力先

  **完全自動ではありません。** 出るのは下書きです。ずれていたら
  --anchor で 2〜3 点だけ留めてください。留めた点のあいだが配分し直されます。

  例:
    movo lyrics align song.mp3 --text lyrics.txt -o song.lrc
    movo lyrics align song.mp3 --text lyrics.txt --anchor 1=5.4 --anchor 17=88.0
    movo make-mv song.mp3 --lyrics song.lrc --title 曲名 -o tmp/mv.mp4
""",
    "batch": """movo batch <テンプレート> [オプション]

  «1 つのテンプレート × N 通りの入力値» を連番で書き出します。

  --input <file>       表（JSON の配列 / CSV）。1 行 = 1 本
  --out <パターン>     書き出し先。{name} {index} {basename} と表の列名が使えます
  --jobs <数>          同時に走らせる本数（既定 コア数 - 2）
  --continue           既に書き出し済みのものを飛ばす（中断からの再開）
  --json               結果を JSON で出力

  例:
    movo batch lyric-mv.json --input songs.json --out tmp/mv/{name}.mp4 --jobs 8
    movo batch "examples/*.json" --out tmp/mv/{basename}.mp4
""",
    "make": """movo make <recipe> [オプション]

  保存した «作り方»（--save-recipe で書き出した JSON）からもう一度作ります。

  --set key=value      レシピの値をさらに上書きする
  -o, --output <path>  書き出し先を変える
""",
    "params": """movo params <file>

  そのプロジェクトで «差し替えられる項目»（params の宣言）を一覧します。

  --json    JSON で出力
""",
    "doctor": """movo doctor

  Python、OS、CPU、メモリ、ffmpeg、フォント、NumPy / Numba、キャッシュ、
  そして «移植がどこまで繋がっているか» を診断します。

  --json    JSON で出力
""",
}
