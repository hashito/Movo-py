# Movo-py

JSON から動画を作る CLI の Python 版。[Movo](../Movo)（JavaScript 版）の全機能を移植し、
**実行速度を最優先**に組み直したものです。

---

## なぜ作り直したか — 速度の設計判断

JS 版は `TypedArray` が C 実装なので «依存ゼロ» と速度を両立できていました。
Python の標準ライブラリには配列演算が無いので、同じ作りにすると壊滅的に遅くなります。

**実測してから決めました。** 1280×720 の 1 フレームで測った値です。

### 判断 1: 全画面の演算は NumPy

| | 1 パス |
| --- | --- |
| 純 Python のループ | **720 ms** |
| NumPy | **13 ms**（54 倍） |

全画面エフェクトは 1 フレームに 10 個前後乗るので、純 Python だと 1 フレーム 7 秒、
153 秒の MV で **7 時間**かかります。ここは NumPy 一択です。

### 判断 2: ラスタライザは NumPy では**遅くなる**

ここが直感に反するところです。「NumPy でベクトル化すれば速い」は
**多角形の塗りには当てはまりません。**

| 多角形 1 枚の塗り | |
| --- | --- |
| NumPy で «囲む矩形の全画素を一括判定» | **30.4 ms** |
| Numba でコンパイルした走査線 | **0.296 ms**（103 倍） |

NumPy 版は **O(囲む矩形の面積 × 辺の数)** で、辺ごとに矩形いっぱいの一時配列を作ります。
走査線は **O(辺 × 行)** で、塗る必要のある画素にしか触りません。
**アルゴリズムの差を、ベクトル化では埋められません。**

239 レイヤーのフレームで比べると 7,261 ms 対 **71 ms** です。

### 結論

| 対象 | 使うもの | 理由 |
| --- | --- | --- |
| 全画面の演算（エフェクト・合成・ブレンド） | **NumPy** | メモリ帯域が支配的。C の一括処理が効く |
| 画素ごとのループ（ラスタライザ・文字・メッシュ変形・粒子） | **Numba**（JIT） | アルゴリズムを保ったまま C 並みの速度になる |
| フレーム単位の並列 | **multiprocessing** | JS 版で 6.3 倍の実績。決定性があるので割れる |

JS 版の実測 1,500 ms/フレーム（1280×720・239 レイヤー）に対し、
この構成なら **70〜100 ms/フレーム** を狙えます。

---

## 依存関係とライセンス

**単体の実行ファイル（EXE）に同梱して配布する**前提です。すべて再配布可能な
寛容ライセンスであることを確認しています。

| | ライセンス | 再配布 |
| --- | --- | --- |
| Python 本体 | PSF License | ✅ |
| NumPy | BSD-3-Clause（他 0BSD / MIT / Zlib / CC0） | ✅ 著作権表示の同梱が必要 |
| Numba | BSD-2-Clause | ✅ |
| llvmlite | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | ✅ |
| PyInstaller | GPLv2+ **ただし «ビルドした成果物は任意のライセンスで配布可» という例外条項付き** | ✅ |

**PyInstaller の例外条項が要点です。** GPL の伝播を受けずに成果物を配布できます。

同梱する著作権表示は `THIRD-PARTY-NOTICES.md` にまとめ、ビルド時に EXE へ入れます。

**PNG / JPEG / WAV のコーデック、TrueType のパーサ、式エンジン、物理演算、
BPM 検出は JS 版と同じく自前実装です。** そこは依存を増やしません。

---

## JS 版から変えないこと

- **同じ JSON からは同じ動画が出る**（決定性）。並列にしても変わらない
- **JSON の書き方は完全に互換**。JS 版のプロジェクトがそのまま動く
- 式エンジンはファイル・ネットワーク・プロセスに触れない
- プラグインは許可リストに載っているものだけ読む
- 光過敏性発作（PSE）の検査を書き出し時に回す

---

## 構成

```
movo/
  core/        画像・PNG/JPEG・WAV・色・数学・乱数・キャッシュ・設定
  schema/      検証と正規化（拍/小節・params・継承・バリアント・相対単位）
  expression/  サンドボックス式エンジン
  animation/   キーフレーム・イージング・モジュレーター
  renderer/    ラスタライザ・文字・図形・エフェクト・3D・粒子
  deformer/    メッシュ変形・マスク
  physics/     剛体・拘束・IK
  audio/       WAV・BPM 検出・ミックス・ラウドネス
  timeline/    シーンとレイヤーの時間解決
  exporters/   mp4 / webm / gif / png 連番
  skill/       スキル（テンプレート）
  cli/         コマンド
```

---

## 開発

```bash
pip install -e ".[dev]"
pytest                      # テスト
python tools/bench.py       # 速度の実測（回帰を見る）
python tools/build_exe.py   # 単体 EXE を作る
```

---

## ドキュメント

| | |
| --- | --- |
| [docs/CLI.md](docs/CLI.md) | **コマンドリファレンス**（22 コマンド）|
| [docs/skills.ja.md](docs/skills.ja.md) | スキル — テンプレートで動画を組む |
| [docs/profile.ja.md](docs/profile.ja.md) | 作風を数値で見る（`profile` / `compare`）|
| [docs/歌詞と曲を合わせる.md](docs/歌詞と曲を合わせる.md) | 時刻の無い歌詞に時刻を付ける |
| [docs/MV改善ログ.md](docs/MV改善ログ.md) | 1 本作るまでに踏んだ不具合と直し方の記録 |

```bash
movo doctor                                               # 環境の診断
movo init my-video                                        # 雛形
movo analyze song.mp3                                     # BPM・拍・区間
movo lyrics align song.mp3 --text lyrics.txt -o song.lrc  # 歌詞に時刻を付ける
movo make-mv song.mp3 --lyrics song.lrc --jobs 5 -o mv.mp4
```
