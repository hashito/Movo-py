#!/usr/bin/env bash
# 「木みたいな草だった」の MV を 4 本まとめて書き出す。
#
#   tmp/mv-v5.mp4         本編（背景写真 ＋ 生成キャラ ＋ 時刻つき歌詞）
#   tmp/mv-a-yohaku.mp4   制作者 A「余白」
#   tmp/mv-b-senkou.mp4   制作者 B「閃光」
#   tmp/mv-c-ryushi.mp4   制作者 C「粒子」
#
# 先に歌詞の時刻が要ります（無ければ作ってください）。
#
#   movo lyrics align <曲>.mp3 --text tmp/kusa-lyrics.txt -o tmp/kusa.lrc
#
# ⚠ **`timeout` コマンドで囲まないでください。** 強制終了と «子プロセスが
#    落ちた» が同じメッセージになり、原因を取り違えます（実際に取り違えました）。
set -u

cd "$(dirname "$0")/.."

echo "── プロジェクトを組み立てます ──"
python tools/build_kusa_chibi_mv.py tmp/kusa-chibi.json
python tools/build_kusa_creators.py

render() {
  local name="$1" project="$2"
  echo "── $name ──"
  rm -rf "tmp/.movo-parallel-$name" "tmp/$name.mp4"
  python -m movo.cli.main render "$project" --jobs 5 -o "tmp/$name.mp4" > "tmp/$name.log" 2>&1
  echo "  exit=$? → tmp/$name.mp4"
  tail -1 "tmp/$name.log"
}

render mv-v5        tmp/kusa-chibi.json
render mv-a-yohaku  tmp/kusa-yohaku.json
render mv-b-senkou  tmp/kusa-senkou.json
render mv-c-ryushi  tmp/kusa-ryushi.json

echo "── 作風の一致度 ──"
for pair in "mv-a-yohaku creator-yohaku" "mv-b-senkou creator-senkou" "mv-c-ryushi creator-ryushi"; do
  set -- $pair
  echo "  $1"
  python -m movo.cli.main compare "tmp/$1.mp4" --target "profiles/$2.json" 2>&1 | grep -E "✔|✖"
done

echo "── 目視の確認用に 3 枚ずつ抜きます ──"
# 数値が揃っていても «読めるか» は測れません。必ず目で見てください。
for f in mv-v5 mv-a-yohaku mv-b-senkou mv-c-ryushi; do
  for t in 25 72 130; do
    ffmpeg -v error -y -ss "$t" -i "tmp/$f.mp4" -frames:v 1 "tmp/chk-$f-$t.png"
  done
done
echo "  tmp/chk-*.png"
