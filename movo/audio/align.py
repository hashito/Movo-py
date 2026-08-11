"""歌詞と曲を合わせる — 時刻の無い歌詞に «下書きの時刻» を付ける。

## なぜ要るのか

歌詞 MV を作るときに手元にあるのは、たいてい **時刻の付いていない歌詞**です。
`movo/core/lyrics.py` は `.lrc` / `.srt` を読めますが、**読むだけ**です。
時刻が無ければ行を等分するしかなく、実際の歌は行ごとに長さが違うので必ずずれます。

`make-mv` はこれまで **小節数で機械的に配って**いました
（`_distribute_lines` が `lines[cursor % len(lines)]`）。曲の構成には追従しますが、
«その行がいつ歌われるか» は一切見ていません。ここを埋めるのがこのモジュールです。

## 何をしないか

**音声認識はしません。** 外部 API に出すと «依存ゼロ» と «同じ JSON からは同じ
動画が出る» の両方が壊れます（`core/lyrics.py` が音声認識を入れなかったのと
同じ理由です）。ここがやるのは **«下書きを作って、直す手間を 28 か所から
数か所に減らす»** ことです。完全自動を名乗りません。

## 4 段構え

  1. **曲の構造**   … `analyze_audio` の BPM・拍・区間
  2. **歌詞の構造** … 空行でブロックに割り、繰り返すブロックをサビとみなす
  3. **歌う範囲**   … 中域の比率から «声が乗っていそうな» ところを拾い、
                      イントロ・間奏・アウトロを配分の対象から外す
  4. **ブロックの中** … 行を **モーラ数**で割り付け、頭を拍にスナップする

## いちばん効いたのは «アンカー»

自動の下書きだけでは実測でずれます（この曲では区間 15 個に対して歌詞ブロックが
7 個で、そもそも 1 対 1 に対応しません）。そこで **数点だけ «この行はここ» と
留めれば、残りがそのあいだで配分し直される** ようにしました。28 行を全部打つのと、
3〜5 点だけ留めるのとでは手間が桁で違います。**これが «補助の仕組み» の本体**で、
自動の下書きはその出発点にすぎません。

## モーラ数で割る理由

文字数で割ると漢字の行が短く出ます（「風に揺れる葉っぱの音」は 10 文字ですが
12 モーラ）。日本語の歌はモーラがほぼ等間隔に乗るので、**モーラ数は文字数より
はっきり良い近似**です。読みが分からない漢字は定数で近似します
（`KANJI_MORAE`。当てずっぽうではありますが、文字数 1.0 とみなすよりは近い）。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from ._compat import as_audio
from .analyze import analyze_audio, onset_envelope

# ── モーラの数え方 ────────────────────────────────────────────────

#: 前の字にくっついて 1 モーラになる小書き。**促音「っ」と撥音「ん」は入れません**
#: （どちらも独立した 1 モーラです。「がっこう」は 4 モーラ）。
NON_MORA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")

#: 読みが分からない漢字 1 字を何モーラとみなすか。
#:
#: 常用漢字の音読みは 1〜3 モーラ（平均およそ 2）、訓読みはもっと長いものの
#: 送り仮名が別に数えられるので、実質はこのあたりです。**辞書を持たない以上
#: これは近似**で、正確にしたければアンカーで留めてください。
KANJI_MORAE = 1.8

#: ラテン文字 1 字ぶん。英単語は «字数 ÷ 2» くらいのモーラで歌われます。
LATIN_MORAE = 0.5

#: 数字 1 字ぶん（「1」＝「いち」で 2 モーラ、「5」＝「ご」で 1 モーラ、平均 2 前後）。
DIGIT_MORAE = 1.8

#: 数えない字。**長音符「ー」を入れてはいけません**（1 モーラです）。
#: 見た目が似ているので実際にここへ書いてしまい、「ケーキ」が 2 モーラに
#: なりました。ダッシュ「—」「-」とは別の字です。
_PUNCTUATION = re.compile(r"[\s、。，．・…！？!?,.\-—「」『』（）()\[\]{}〜~:;：；\"'’”]")
_HIRAGANA = (0x3041, 0x309F)
_KATAKANA = (0x30A0, 0x30FF)
_CJK = (0x4E00, 0x9FFF)
_CJK_EXT = (0x3400, 0x4DBF)

# ── 歌う範囲の見立て ──────────────────────────────────────────────

#: 声が乗っているかの目安に使う «中域の比率» を均す幅（秒）。
#: 1 音ごとの上下ではなく «フレーズ単位» で見たいので、やや長めです。
VOCAL_SMOOTH_SECONDS = 0.45

#: これより短い «歌っていない» 区間は間奏とみなしません（息継ぎ・行間です）。
MIN_GAP_SECONDS = 1.6

#: これより短い «歌っている» 区間は拾いません（間奏中のシャウトなど）。
MIN_RUN_SECONDS = 1.2

#: **フレーズ単位**で切るときの目安。区間（`MIN_GAP_SECONDS`）より細かく見ます。
#:
#: 区間だけで割ると «1 つの長い塊に 12 行を等分» になり、行ごとのズレが積もります。
#: 歌は 1 行ごとに息を継ぐので、0.3 秒級の切れ目を拾えば **行の切れ目そのもの** が
#: 見えます。ここを使って «行 → フレーズ» を対応づけるのが、モーラ配分より
#: はっきり効きます。
PHRASE_GAP_SECONDS = 0.32
PHRASE_MIN_SECONDS = 0.55

#: 中域比率をこの分位で正規化します。曲全体の «高いほう» を 1 とみなす目安。
VOCAL_HIGH_PERCENTILE = 88
VOCAL_LOW_PERCENTILE = 12

#: 正規化した中域比率がこれを超えていれば «歌っていそう»。
#: 低くすると間奏を歌だと見て、高くすると歌を間奏だと見ます。実測でこのあたりが
#: いちばん «間奏だけ» を落としました。
VOCAL_THRESHOLD = 0.5


def count_morae(text: str) -> float:
    """日本語混じりの 1 行が何モーラかを見積もる。

    ``0`` は返しません（0 だと配分の重みが消えて、その行の尺が無くなります）。
    """
    total = 0.0
    for char in str(text):
        if char in NON_MORA or _PUNCTUATION.match(char):
            continue
        code = ord(char)
        if _HIRAGANA[0] <= code <= _HIRAGANA[1] or _KATAKANA[0] <= code <= _KATAKANA[1]:
            total += 1.0
        elif _CJK[0] <= code <= _CJK[1] or _CJK_EXT[0] <= code <= _CJK_EXT[1]:
            total += KANJI_MORAE
        elif char.isdigit():
            total += DIGIT_MORAE
        elif char.isascii() and char.isalpha():
            total += LATIN_MORAE
        else:
            total += 1.0
    return max(0.5, total)


# ================================================================== #
# 歌詞の構造                                                          #
# ================================================================== #


def split_blocks(text: str) -> list[dict[str, Any]]:
    """歌詞を «空行区切りのブロック» に割る。

    歌詞は普通 1 行空けて A メロ・サビと書き分けられているので、**書式が
    そのまま構造の情報**になっています。ここを捨てて 1 行ずつ均等に扱うと、
    サビの繰り返しが見えなくなります。
    """
    blocks: list[dict[str, Any]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append({"index": len(blocks), "lines": list(current)})
            current.clear()

    for raw in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if line:
            current.append(line)
        else:
            flush()
    flush()
    return blocks


def _normalise_for_compare(lines: list[str]) -> str:
    return _PUNCTUATION.sub("", "".join(lines))


def label_blocks(blocks: list[dict[str, Any]], *, similarity: float = 0.82) -> list[dict[str, Any]]:
    """繰り返すブロックを見つけて ``kind`` を付ける。

    **繰り返し ＝ サビ**とみなします。歌詞だけを見て «ここがサビ» と言える
    数少ない手掛かりで、しかもよく当たります（サビは 2 回以上出るからサビです）。

    完全一致ではなく類似度で見るのは、2 番のサビが 1 語だけ違うことがよくある
    ためです。閾値 0.82 は «1〜2 語違い» までを同じと見る値です。
    """
    out: list[dict[str, Any]] = []
    signatures: list[str] = []
    for block in blocks:
        signature = _normalise_for_compare(block["lines"])
        repeat_of: int | None = None
        for earlier, seen in enumerate(signatures):
            if not seen or not signature:
                continue
            if SequenceMatcher(None, seen, signature).ratio() >= similarity:
                repeat_of = out[earlier].get("repeatOf")
                if repeat_of is None:
                    repeat_of = earlier
                break
        signatures.append(signature)
        out.append({**block, "repeatOf": repeat_of})

    # «あとから誰かに真似された» ブロックもサビです（1 回目のサビは repeatOf が None）
    repeated = {block["repeatOf"] for block in out if block["repeatOf"] is not None}
    for block in out:
        is_chorus = block["repeatOf"] is not None or block["index"] in repeated
        block["kind"] = "chorus" if is_chorus else "verse"
        block["morae"] = sum(count_morae(line) for line in block["lines"])
    return out


# ================================================================== #
# 歌う範囲                                                            #
# ================================================================== #


def vocal_presence(audio, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """«声が乗っていそうか» の目安を 0〜1 で作る。

    :returns: ``{"hop", "presence", "rms"}``

    中域（200〜2000Hz）が全体に占める比率を見ます。歌の基音とその倍音の下のほうが
    ここに集まる一方、間奏はキック（低域）とシンバル・ギターの倍音（高域）に
    寄るためです。

    ⚠ **音源分離ではありません。** 中域の濃いシンセやピアノが鳴っていれば
    «歌っている» と出ます。ここで狙っているのは «イントロ・間奏・アウトロを
    配分の対象から外す» ことだけで、それ以上の精度は求めていません
    （足りなければアンカーで留めてください）。
    """
    options = options or {}
    audio = as_audio(audio)
    envelope = onset_envelope(audio, options)
    hop = envelope["hop"]
    bands = envelope.get("bands")
    if bands is None:  # 古い戻り値でも動くように（rms しか無ければ一様に «歌っている»）
        frames = envelope["rms"].size
        return {"hop": hop, "presence": np.ones(frames, np.float64), "rms": envelope["rms"]}

    low, mid, high, full = bands
    ratio = mid / (low + mid + high + 1e-12)

    # 無音のところは比率が暴れる（0/0 に近い）ので、音量で重みを付けて潰します
    loud = full / (float(full.max()) or 1.0)
    ratio = ratio * np.minimum(1.0, loud * 4.0)

    # **均す幅は用途で変えます。** 区間（歌っているひとかたまり）を見るときは
    # 0.45 秒で均して «フレーズ単位» にしますが、その幅だと 0.3 秒の息継ぎが
    # 消えてしまい、フレーズの切れ目が拾えません（実測で 28 行に対して
    # フレーズが 10 個しか出ませんでした）。細かく見たいときは短く渡します。
    smooth = options.get("smooth")
    radius = max(1, round((VOCAL_SMOOTH_SECONDS if smooth is None else float(smooth)) / hop))
    smoothed = _smooth(ratio, radius)

    low_value = float(np.percentile(smoothed, VOCAL_LOW_PERCENTILE))
    high_value = float(np.percentile(smoothed, VOCAL_HIGH_PERCENTILE))
    span = high_value - low_value
    presence = np.clip((smoothed - low_value) / span, 0.0, 1.0) if span > 1e-9 else np.zeros_like(smoothed)
    return {"hop": hop, "presence": presence, "rms": full}


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    """端では «届いた枠だけ» で平均する移動平均（`analyze.py` と同じ数え方）。"""
    n = values.size
    padded = np.concatenate([[0.0], np.cumsum(values)])
    index = np.arange(n)
    start = np.maximum(0, index - radius)
    stop = np.minimum(n, index + radius + 1)
    return (padded[stop] - padded[start]) / (stop - start)


def singing_windows(
    presence: np.ndarray,
    hop: float,
    duration: float,
    *,
    threshold: float = VOCAL_THRESHOLD,
    min_gap: float = MIN_GAP_SECONDS,
    min_run: float = MIN_RUN_SECONDS,
) -> list[tuple[float, float]]:
    """歌っていそうな時間帯を ``[(始まり, 終わり), …]`` で返す。

    短い切れ目を埋めてから短い塊を捨てる、という順番が要ります。逆にすると
    息継ぎで割れた «同じフレーズ» が両方とも短すぎて消えます。
    """
    if presence.size == 0:
        return [(0.0, duration)]
    active = presence >= threshold
    windows: list[list[float]] = []
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            windows.append([start * hop, index * hop])
            start = None
    if start is not None:
        windows.append([start * hop, presence.size * hop])
    if not windows:
        return [(0.0, duration)]

    merged: list[list[float]] = [windows[0]]
    for window in windows[1:]:
        if window[0] - merged[-1][1] < min_gap:
            merged[-1][1] = window[1]
        else:
            merged.append(window)

    kept = [(max(0.0, s), min(duration, e)) for s, e in merged if e - s >= min_run]
    return kept or [(0.0, duration)]


def singing_phrases(
    presence: np.ndarray,
    hop: float,
    windows: list[tuple[float, float]],
    *,
    threshold: float = VOCAL_THRESHOLD,
) -> list[tuple[float, float]]:
    """歌唱区間の **中を** さらにフレーズへ割る。

    `singing_windows` は «歌っているひとかたまり» を返します。そこへ行を
    モーラ数で等分すると、1 行ぶんのズレが後ろへ積もっていきます（実際に
    «全体的にズレている» と指摘されました）。

    歌は 1 行ごとに息を継ぐので、**0.3 秒級の切れ目を拾えば行の切れ目が見えます。**
    ここで拾った切れ目に行の頭を合わせるほうが、配分をどれだけ工夫するより効きます。
    """
    phrases: list[tuple[float, float]] = []
    for start, end in windows:
        first = max(0, int(start / hop))
        last = min(presence.size, int(end / hop))
        if last <= first:
            continue
        segment = presence[first:last]
        runs = _runs_above(segment, threshold, hop, PHRASE_GAP_SECONDS, PHRASE_MIN_SECONDS)
        phrases.extend((start + a, start + b) for a, b in runs)
    return phrases


def _runs_above(
    values: np.ndarray, threshold: float, hop: float, min_gap: float, min_run: float
) -> list[tuple[float, float]]:
    """しきい値を超えている «塊» を返す。短い切れ目を埋めてから短い塊を捨てます。"""
    active = values >= threshold
    runs: list[list[float]] = []
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append([start * hop, index * hop])
            start = None
    if start is not None:
        runs.append([start * hop, values.size * hop])
    if not runs:
        return []
    merged = [runs[0]]
    for run in runs[1:]:
        if run[0] - merged[-1][1] < min_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(a, b) for a, b in merged if b - a >= min_run]


def assign_lines_to_phrases(morae: list[float], phrases: list[tuple[float, float]]) -> list[float] | None:
    """行をフレーズへ «順番を保ったまま» 割り当て、各行の開始時刻を返す。

    動的計画法で «行 i までをフレーズ j までに収める» 最小費用を解きます。
    費用は **モーラ数の比と、フレーズの長さの比のズレ**です。長い行には長い
    フレーズが当たるべき、という以上のことは仮定していません。

    1 つのフレーズに複数行が入ることも（早口）、1 行が複数フレーズにまたがる
    ことも（伸ばす歌い方）許します。**行数よりフレーズが極端に少ないときは
    ``None``** を返して、呼ぶ側がモーラ配分へ落とせるようにします。
    """
    n, m = len(morae), len(phrases)
    if n == 0 or m == 0:
        return None
    # フレーズが行数の半分も無いなら «行の切れ目が見えていない»。素直に配分へ落とす。
    if m < max(2, n // 2):
        return None

    total_morae = sum(morae) or 1.0
    total_time = sum(end - start for start, end in phrases) or 1.0
    want = [value / total_morae for value in morae]
    have = [(end - start) / total_time for start, end in phrases]

    INF = float("inf")
    # cost[i][j] = 行 i..(次) をフレーズ j..(次) に割り当てるまでの最小費用
    best = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0

    # 1 手で «行を a 本、フレーズを b 本» まとめて消費する。a も b も 1〜3 まで。
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == INF:
                continue
            for a in range(1, 4):
                if i + a > n:
                    break
                for b in range(1, 4):
                    if j + b > m:
                        break
                    share_lines = sum(want[i:i + a])
                    share_time = sum(have[j:j + b])
                    # 1 対 1 から離れるほど罰を足す（早口・伸ばしを «少し» 許す）
                    penalty = abs(share_lines - share_time) + 0.04 * (a - 1 + b - 1)
                    value = best[i][j] + penalty
                    if value < best[i + a][j + b]:
                        best[i + a][j + b] = value
                        back[i + a][j + b] = (i, j)

    # 最後は «行を全部使い切る»。フレーズは余ってよい（間奏の歌以外の音）。
    end_j = min(range(m + 1), key=lambda j: best[n][j])
    if best[n][end_j] == INF:
        return None

    # 逆に辿って、行の頭を «割り当てたフレーズの頭» に置く
    starts = [0.0] * n
    i, j = n, end_j
    while i > 0:
        previous = back[i][j]
        if previous is None:
            return None
        pi, pj = previous
        span_start = phrases[pj][0]
        span_end = phrases[min(m, j) - 1][1]
        chunk = morae[pi:i]
        total = sum(chunk) or 1.0
        used = 0.0
        for offset, value in enumerate(chunk):
            starts[pi + offset] = span_start + (used / total) * (span_end - span_start)
            used += value
        i, j = pi, pj
    return starts


# ================================================================== #
# 割り付け                                                            #
# ================================================================== #


def _total_span(windows: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in windows)


def _regularise_within_blocks(
    times: list[float], flat: list[dict[str, Any]], bar_seconds: float, anchored: set[int]
) -> list[float]:
    """同じブロックの行を «一定の小節間隔» に均す。

    間隔は «実際に取れた平均間隔を、半小節単位に丸めたもの»。最短 0.5 小節、
    最長 8 小節に収めます。**留めた行（アンカー）があるブロックは触りません** —
    人が «ここ» と言った位置を、平均で上書きしては意味がありません。
    """
    out = list(times)
    step_unit = bar_seconds / 2
    start = 0
    while start < len(flat):
        stop = start
        while stop + 1 < len(flat) and flat[stop + 1]["block"] == flat[start]["block"]:
            stop += 1
        count = stop - start + 1
        if count >= 3 and not any(index in anchored for index in range(start, stop + 1)):
            span = times[stop] - times[start]
            raw = span / (count - 1)
            step = max(step_unit, min(8 * bar_seconds, round(raw / step_unit) * step_unit))
            for offset in range(count):
                out[start + offset] = times[start] + offset * step
        start = stop + 1
    return out


def _half_bar_grid(bars: list[float], bar_seconds: float, duration: float) -> list[float]:
    """小節の頭とその中間（半小節）を並べた «歌詞が切り替わってよい場所»。

    `analyze_audio` が返す小節の一覧は曲の頭から等間隔なので、間を 1 つ足すだけで
    半小節のグリッドになります。小節が取れなかったときは空を返します
    （呼ぶ側が拍へ落とします）。
    """
    if not bars or bar_seconds <= 0:
        return []
    grid: list[float] = []
    for index, bar in enumerate(bars):
        grid.append(bar)
        middle = bar + bar_seconds / 2
        if middle < duration and (index + 1 >= len(bars) or middle < bars[index + 1]):
            grid.append(middle)
    return sorted(grid)


def _offset_of(windows: list[tuple[float, float]], time: float) -> float:
    """実時間を «歌っている時間» の先頭からの秒数に直す（`_time_at_offset` の逆）。"""
    used = 0.0
    for start, end in windows:
        if time < start:
            return used
        if time <= end:
            return used + (time - start)
        used += end - start
    return used


def _time_at_offset(windows: list[tuple[float, float]], offset: float) -> float:
    """«歌っている時間» の先頭から数えて offset 秒の地点を、実時間に直す。

    間奏を跨いだぶんは足し飛ばします。ここが «間奏に歌詞を置かない» の実体です。
    """
    remaining = max(0.0, offset)
    for start, end in windows:
        length = end - start
        if remaining <= length:
            return start + remaining
        remaining -= length
    return windows[-1][1] if windows else 0.0


def _window_end_after(windows: list[tuple[float, float]], time: float) -> float:
    """``time`` が入っている（か、次に来る）歌唱区間の終わりを返す。"""
    for start, end in windows:
        if time < end:
            return end if time >= start else start
    return windows[-1][1] if windows else time


def _snap_to_beat(time: float, beats: list[float], tolerance: float) -> tuple[float, bool]:
    """いちばん近い拍へ寄せる。離れすぎているときは動かしません。

    無理に寄せると、拍の裏から入る歌い出し（アウフタクト）まで表に引っ張られて
    かえってずれます。``tolerance`` は普通 «拍の半分» を渡します。
    """
    if not beats:
        return time, False
    position = np.searchsorted(beats, time)
    candidates = []
    if position > 0:
        candidates.append(beats[position - 1])
    if position < len(beats):
        candidates.append(beats[position])
    if not candidates:
        return time, False
    nearest = min(candidates, key=lambda beat: abs(beat - time))
    if abs(nearest - time) <= tolerance:
        return float(nearest), True
    return time, False


def _apply_anchors(offsets: list[float], anchors: list[tuple[int, float]], windows) -> list[float]:
    """アンカーで «モーラ軸» を区分線形に伸縮する。

    留めた行がぴったりその時刻に来て、あいだの行はそのあいだで配り直されます。
    時間そのものを動かすのではなく **モーラ軸のほうを伸縮する** のが要点で、
    こうすると «長い行は長く» という性質が留めたあとも保たれます。
    """
    if not anchors:
        return offsets

    total = _total_span(windows)
    # アンカーの時刻を «歌っている時間» 上の位置へ直す
    def to_offset(time: float) -> float:
        used = 0.0
        for start, end in windows:
            if time < start:
                return used
            if time <= end:
                return used + (time - start)
            used += end - start
        return total

    points = sorted({index: to_offset(time) for index, time in anchors}.items())
    known = [(0, 0.0)] + [p for p in points if 0 < p[0] < len(offsets)] + [(len(offsets) - 1, offsets[-1])]
    # 先頭・末尾がアンカーされていれば、そちらを優先する
    if points and points[0][0] == 0:
        known[0] = points[0]
    if points and points[-1][0] == len(offsets) - 1:
        known[-1] = points[-1]

    out = list(offsets)
    for (left_index, left_time), (right_index, right_time) in zip(known, known[1:]):
        if right_index <= left_index:
            continue
        source_span = offsets[right_index] - offsets[left_index]
        target_span = right_time - left_time
        for index in range(left_index, right_index + 1):
            if source_span <= 1e-9:
                ratio = (index - left_index) / max(1, right_index - left_index)
            else:
                ratio = (offsets[index] - offsets[left_index]) / source_span
            out[index] = left_time + ratio * target_span
    return out


def parse_anchors(raw: Any, lines: list[str]) -> list[tuple[int, float]]:
    """``"12=27.5"`` / ``"木だと思ったけど=27.5"`` を ``(行番号, 秒)`` に直す。

    行番号は **1 始まり**です（人が数えるときの番号と揃えます）。文字で書いた
    ときは «その文字で始まる最初の行» を探します。
    """
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[tuple[int, float]] = []
    for item in items:
        text = str(item)
        if "=" not in text:
            continue
        key, _, value = text.rpartition("=")
        try:
            at = float(value)
        except ValueError:
            continue
        key = key.strip()
        index: int | None = None
        if key.isdigit():
            number = int(key)
            if 1 <= number <= len(lines):
                index = number - 1
        else:
            needle = _PUNCTUATION.sub("", key)
            for position, line in enumerate(lines):
                if needle and _PUNCTUATION.sub("", line).startswith(needle):
                    index = position
                    break
        if index is not None:
            out.append((index, at))
    return sorted(set(out))


# ================================================================== #
# 本体                                                                #
# ================================================================== #


def align_lyrics(audio, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """時刻の無い歌詞に、曲に合わせた «下書きの時刻» を付ける。

    :param audio: 音源（`decode_audio_file` の戻り値）
    :param text: 歌詞（空行でブロックに割られます）
    :param options: ``anchors`` / ``beatsPerBar`` / ``minBpm`` / ``maxBpm`` /
        ``snap``（拍に寄せるか。既定 True）/ ``gaps``（間奏を避けるか。既定 True）/
        ``start`` / ``end``（歌う範囲を秒で直接指定）
    :returns: ``{"bpm", "duration", "blocks", "lines", "windows", "warnings", …}``
    """
    options = options or {}
    audio = as_audio(audio)

    settings = {"beatsPerBar": options.get("beatsPerBar") or 4}
    for key in ("minBpm", "maxBpm"):
        if options.get(key):
            settings[key] = options[key]
    analysis = analyze_audio(audio, settings)
    duration = analysis["duration"]
    beats = list(analysis["beats"])
    bpm = analysis["bpm"]

    blocks = label_blocks(split_blocks(text))
    flat: list[dict[str, Any]] = []
    for block in blocks:
        for position, line in enumerate(block["lines"]):
            flat.append({
                "text": line,
                "block": block["index"],
                "kind": block["kind"],
                "firstOfBlock": position == 0,
                "morae": count_morae(line),
            })

    warnings: list[str] = []
    if not flat:
        return {
            "bpm": bpm, "duration": duration, "confidence": 0.0,
            "blocks": [], "lines": [], "windows": [], "sections": analysis["sections"],
            "warnings": ["歌詞が空です"],
        }

    # ── 歌う範囲を決める ────────────────────────────────────────
    presence: dict[str, Any] | None = None
    if options.get("gaps", True):
        presence = vocal_presence(audio, {})
        windows = singing_windows(presence["presence"], presence["hop"], duration)
    else:
        windows = [(0.0, duration)]

    start_override = options.get("start")
    end_override = options.get("end")
    if start_override is not None:
        windows = [(max(float(start_override), s), e) for s, e in windows if e > float(start_override)]
    if end_override is not None:
        windows = [(s, min(float(end_override), e)) for s, e in windows if s < float(end_override)]
    windows = [(s, e) for s, e in windows if e - s > 1e-3] or [(0.0, duration)]

    total = _total_span(windows)
    if total <= 0:
        warnings.append("歌う範囲が取れませんでした。曲全体に配ります")
        windows = [(0.0, duration)]
        total = duration

    # ── まずフレーズに合わせる。駄目ならモーラ数で割り付ける ────
    #
    # **フレーズに合わせるほうがはっきり効きます。** 区間へモーラ数で等分すると
    # 1 行ぶんのズレが後ろへ積もり、«全体的にズレている» という出方をします。
    # 息継ぎの切れ目に頭を合わせれば、そもそも積もりません。
    morae = [row["morae"] for row in flat]
    method = "morae"
    offsets: list[float] | None = None
    phrases: list[tuple[float, float]] = []
    if presence is not None and options.get("phrases", True):
        # フレーズ用は «細かく» 見た曲線を使います（息継ぎを潰さない幅）。
        fine = vocal_presence(audio, {"smooth": options.get("phraseSmooth", 0.12)})
        phrases = singing_phrases(fine["presence"], fine["hop"], windows)
        starts = assign_lines_to_phrases(morae, phrases)
        if starts is not None:
            offsets = [_offset_of(windows, value) for value in starts]
            method = "phrase"

    if offsets is None:
        morae_total = sum(morae) or 1.0
        offsets = []
        used = 0.0
        for value in morae:
            offsets.append(used)
            used += value / morae_total * total

    anchors = parse_anchors(options.get("anchors"), [row["text"] for row in flat])
    if anchors:
        offsets = _apply_anchors(offsets, anchors, windows)

    anchored = {index for index, _ in anchors}
    times = [_time_at_offset(windows, offset) for offset in offsets]

    # ── 音楽のグリッドに寄せる ──────────────────────────────────
    #
    # **拍ではなく «半小節» に寄せます。** 148BPM だと 1 拍は 0.405 秒しかなく、
    # 拍に寄せてもほとんど動きません（ズレは 1〜2 秒の単位で出ます）。歌詞は
    # 小節の頭か半小節で切り替わるので、そこに寄せると **ズレの積み上がりが
    # 消え、切り替わりが音楽と揃って «合っている» ように見えます。**
    beat_seconds = (60.0 / bpm) if bpm > 0 else 0.0
    bar_seconds = beat_seconds * analysis["beatsPerBar"]
    grid = _half_bar_grid(analysis["bars"], bar_seconds, duration) if bar_seconds > 0 else []
    if not grid:
        grid = beats
    # 半小節の «半分» まで動かします。これ以上動かすと隣の行と入れ替わります。
    tolerance = (bar_seconds / 4) if bar_seconds > 0 else 0.0
    snap = options.get("snap", True) and bool(grid)
    beats = grid
    snapped: list[bool] = []
    for index, time in enumerate(times):
        if not snap or index in anchored:
            snapped.append(False)
            continue
        # 留めた行は動かしません（留めた意味が無くなります）
        moved, hit = _snap_to_beat(time, beats, tolerance)
        times[index] = moved
        snapped.append(hit)

    # ── ブロックの中は等間隔にする ──────────────────────────────
    #
    # 歌詞は «同じ節の中では同じ間隔» で出ます（4 行のサビなら 2 小節ごと、など）。
    # モーラ配分のままだと 2.0・3.5・3.0・3.5 小節とばらつき、1 行ごとに
    # «早い / 遅い» が交互に出て «ズレている» と感じます。**ブロック単位で
    # 一定の小節間隔に均す**と、多少頭がずれていても «拍に乗っている» と読めます。
    if options.get("regular", True) and bar_seconds > 0:
        times = _regularise_within_blocks(times, flat, bar_seconds, anchored)

    # 単調にする（グリッドへ寄せた結果、前の行を追い越すことがあります）
    for index in range(1, len(times)):
        if times[index] <= times[index - 1]:
            times[index] = times[index - 1] + max(0.05, beat_seconds / 4)

    # ── 尺と確からしさ ──────────────────────────────────────────
    presence_curve = presence["presence"] if presence is not None else None
    presence_hop = presence["hop"] if presence is not None else 0.0

    lines: list[dict[str, Any]] = []
    for index, row in enumerate(flat):
        at = times[index]
        nxt = times[index + 1] if index + 1 < len(times) else None
        span = (nxt - at) if nxt is not None else max(1.0, duration - at)
        # **間奏をまたいで残さない。** 次の行まで表示し続けると、9 秒の間奏の
        # あいだ «直前の 1 行» が画面に残ります。歌っていないところで歌詞が
        # 出ているのは、ずれているより目に付きます。
        span = min(span, _window_end_after(windows, at) - at)
        confidence = _line_confidence(
            analysis["confidence"], index, anchored, len(flat), presence_curve, presence_hop, at
        )
        entry = {
            "text": row["text"],
            "at": round(at, 3),
            "for": round(max(0.2, span), 3),
            "block": row["block"],
            "kind": row["kind"],
            "confidence": round(confidence, 3),
            "snapped": snapped[index],
            "anchored": index in anchored,
        }
        lines.append(entry)

    if analysis["confidence"] < 0.4:
        warnings.append("BPM の確からしさが低めです。拍へのスナップがずれている可能性があります")
    if not anchors:
        warnings.append(
            "アンカーがありません。歌い出しとサビの頭を 2〜3 点だけ留めると精度が大きく上がります"
            "（--anchor 1=4.9 --anchor 5=20.1）"
        )
    weak = [line for line in lines if line["confidence"] < 0.5]
    if weak:
        warnings.append(f"確からしさが低い行が {len(weak)} 行あります（要確認）")

    if method == "morae":
        warnings.append(
            "息継ぎの切れ目が拾えなかったので、モーラ数で等分しました"
            "（フレーズに合わせるより精度が落ちます）"
        )

    return {
        "bpm": bpm,
        "duration": duration,
        "beatsPerBar": analysis["beatsPerBar"],
        "confidence": analysis["confidence"],
        "sections": analysis["sections"],
        "blocks": blocks,
        "windows": [(round(s, 3), round(e, 3)) for s, e in windows],
        "phrases": [(round(s, 3), round(e, 3)) for s, e in phrases],
        "method": method,
        "lines": lines,
        "warnings": warnings,
    }


def _line_confidence(
    tempo_confidence: float,
    index: int,
    anchored: set[int],
    count: int,
    presence: np.ndarray | None,
    hop: float,
    at: float,
) -> float:
    """1 行ぶんの «どれくらい信じてよいか»。

    留めた行は 1.0。留めた行から離れるほど下がり、その時刻に声が乗って
    いなさそうなら更に下がります。**表示のための目安**で、これ自体を
    別の計算に使うことは想定していません。
    """
    if index in anchored:
        return 1.0
    if anchored:
        distance = min(abs(index - a) for a in anchored)
        # 留めた行から 8 行離れると «アンカー無し» と同じ扱いまで落ちます
        nearness = max(0.0, 1.0 - distance / 8)
    else:
        nearness = 0.0
    base = 0.35 + 0.35 * tempo_confidence + 0.3 * nearness

    if presence is not None and presence.size and hop > 0:
        frame = min(presence.size - 1, max(0, int(at / hop)))
        base *= 0.55 + 0.45 * float(presence[frame])
    return min(1.0, max(0.0, base))


# ================================================================== #
# 書き出し                                                            #
# ================================================================== #


def to_lrc(lines: list[dict[str, Any]], *, meta: dict[str, str] | None = None) -> str:
    """`.lrc` の文字列にする。`core.lyrics.parse_lrc` がそのまま読み返せます。

    小数は 2 桁です。LRC の慣習であり、10ms 刻みの解析より細かく書いても
    «あるように見えるだけ» だからです。
    """
    out: list[str] = []
    for key, value in (meta or {}).items():
        if value:
            out.append(f"[{key}:{value}]")
    for line in lines:
        at = max(0.0, float(line["at"]))
        minutes = int(at // 60)
        seconds = at - minutes * 60
        out.append(f"[{minutes:02d}:{seconds:05.2f}]{line['text']}")
    return "\n".join(out) + "\n"


def to_scenario(result: dict[str, Any]) -> dict[str, Any]:
    """AI がシナリオを書くための «下ごしらえ» を作る。

    生の解析結果は拍が数千個あって読ませられません。ここでは **構成を決めるのに
    要るものだけ** に絞ります（区間・ブロック・各ブロックの時刻・要確認の行）。
    シナリオを書く側が «どこが何秒から何秒で、そこで何を歌っているか» だけ
    分かればよい、という切り方です。
    """
    by_block: dict[int, list[dict[str, Any]]] = {}
    for line in result["lines"]:
        by_block.setdefault(line["block"], []).append(line)

    blocks = []
    for block in result["blocks"]:
        rows = by_block.get(block["index"], [])
        if not rows:
            continue
        start = rows[0]["at"]
        end = rows[-1]["at"] + rows[-1]["for"]
        blocks.append({
            "index": block["index"],
            "kind": block["kind"],
            "repeatOf": block["repeatOf"],
            "start": round(start, 2),
            "end": round(end, 2),
            "bars": round((end - start) / (60 / result["bpm"] * result["beatsPerBar"]), 1) if result["bpm"] else 0,
            "lines": [row["text"] for row in rows],
            "confidence": round(sum(row["confidence"] for row in rows) / len(rows), 2),
        })

    gaps = []
    windows = result["windows"]
    for (_, end), (start, _) in zip(windows, windows[1:]):
        gaps.append({"start": round(end, 2), "end": round(start, 2), "kind": "間奏"})
    if windows and windows[0][0] > 0.5:
        gaps.insert(0, {"start": 0.0, "end": round(windows[0][0], 2), "kind": "イントロ"})
    if windows and result["duration"] - windows[-1][1] > 0.5:
        gaps.append({"start": round(windows[-1][1], 2), "end": round(result["duration"], 2), "kind": "アウトロ"})

    return {
        "duration": result["duration"],
        "bpm": result["bpm"],
        "beatsPerBar": result["beatsPerBar"],
        "barSeconds": round(60 / result["bpm"] * result["beatsPerBar"], 3) if result["bpm"] else 0,
        "blocks": blocks,
        "instrumental": gaps,
        "needsCheck": [
            {"line": index + 1, "text": line["text"], "at": line["at"], "confidence": line["confidence"]}
            for index, line in enumerate(result["lines"])
            if line["confidence"] < 0.5
        ],
        "warnings": result["warnings"],
    }


__all__ = [
    "KANJI_MORAE",
    "align_lyrics",
    "assign_lines_to_phrases",
    "count_morae",
    "label_blocks",
    "parse_anchors",
    "singing_phrases",
    "singing_windows",
    "split_blocks",
    "to_lrc",
    "to_scenario",
    "vocal_presence",
]
