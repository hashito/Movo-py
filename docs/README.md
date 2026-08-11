# ドキュメント

| | |
| --- | --- |
| [CLI.md](CLI.md) | **コマンドリファレンス**（22 コマンド）。まずここ |
| [skills.ja.md](skills.ja.md) | スキル — テンプレートで動画を組む。書き方も |
| [profile.ja.md](profile.ja.md) | 作風を数値で見る（`profile` / `compare`）|
| [歌詞と曲を合わせる.md](歌詞と曲を合わせる.md) | 時刻の無い歌詞に時刻を付ける（`lyrics align`）|
| [MV改善ログ.md](MV改善ログ.md) | 実際に 1 本作るまでに踏んだ不具合と直し方の記録 |

プロジェクト全体の説明は [../README.ja.md](../README.ja.md) にあります。

---

## はじめての 5 分

```bash
pip install -e ".[dev]"
movo doctor                       # 環境が整っているか
movo init my-video                # 雛形を作る
movo frame my-video/movo.json -t 1 -o tmp/check.png   # 1 枚だけ見る
movo render my-video/movo.json -o tmp/out.mp4
```

## 曲から MV を作る 3 分

```bash
movo analyze song.mp3                                     # BPM・拍・区間
movo lyrics align song.mp3 --text lyrics.txt -o song.lrc  # 歌詞に時刻を付ける
movo make-mv song.mp3 --lyrics song.lrc --jobs 5 -o tmp/mv.mp4
```

## 覚えておくと損をしないこと

- **描く前に `movo frame` で 1 枚見る。** 1 本 5〜12 分かかります
- **`movo validate` を通してから描く。** 素材名の打ち間違いは検証で出ます
- **並列レンダリングの検証に `timeout` を使わない。** 強制終了と
  「子プロセスが落ちた」が同じメッセージになり、原因を取り違えます
- **`movo compare` は «読めるか» を測らない。** 数値が揃っても目視は必要です
