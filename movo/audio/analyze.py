"""音源から «拍・小節・区間» を取り出す（JS 版 audio/analyze.js の移植）。

出したい値は 4 つです。

  1. オンセット強度  … 音が «増えた» 量。拍はここに山として現れる
  2. BPM             … オンセット強度の自己相関のピーク
  3. 位相（1 拍目）  … 拍グリッドをずらして «山に一番乗る» 位置
  4. 区間            … RMS の移動平均が大きく変わるところで切る

全帯域をまとめた RMS «だけ» では足りませんでした。ベースとパッドが鳴り
続けている曲だと、スネアやハイハットの立ち上がりが埋もれて «1 小節に 1 回
しか音が鳴っていない» ように見え、BPM が 1/2 や 1/4 になります。そこで
1 極フィルタで低・中・高の 3 帯域に分け、帯域ごとに立ち上がりを見てから
足しています。

## Python 版で使い分けたもの

| 場所 | 使うもの | 理由 |
| --- | --- | --- |
| 帯域分け（1 極フィルタ） | **Numba** | 1 サンプル前が要る逐次処理。畳めない |
| 包絡・オンセット・移動平均 | **NumPy** | 枠ごとの一括演算。累積和で 1 パス |
| 自己相関 | **`numpy.fft`** | O(n log n)。長い曲でも伸びない |
| くしフィルタの採点 | **Numba** | 位相 × 周期 × 拍の三重ループ。ここが一番重い |

自己相関に FFT を使うのは **アルゴリズムを変えたわけではありません**。
「相関 ＝ 畳み込み」なので出てくる値は同じで、丸め誤差だけが乗ります
（30 秒の曲で実測した最大の相対差は **4.4e-16**。ピークの位置がそれで
動くことはありません）。

**速さの差は «長い曲ほど» 効きます。** 30 秒（枠 2,999・lag 25〜100）では
FFT 0.101 ms 対 «lag ごとに NumPy の内積» 0.106 ms でほぼ互角です。
lag の数は BPM の範囲で決まって増えないのに対し、素直に回す側は曲の長さに
比例するので、5 分の曲では 10 倍の開きになります。

いちばん重いのは自己相関ではなく **くしフィルタ**でした（30 秒の解析
全体で 106 ms、その大半がここ）。だから Numba を入れたのはそちらです。

くしフィルタは «`for (phase = 0; phase < period; phase += 0.05)`» という
**浮動小数の足し込み**をそのまま写しています。`linspace` に置き換えると
位相が微妙にずれ、BPM が別の候補に転ぶことがありました。
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
from numba import njit

from ._compat import as_audio, channel_count, clamp, decode_wav, find_ffmpeg, js_round

# 包絡の刻み幅（秒）。10ms なら 240BPM の 16 分音符（62ms）も分けて見えます。
HOP_SECONDS = 0.01
# 窓は hop の 2 倍。1 hop ちょうどだと窓が繋がりません。
WINDOW_HOPS = 2
# 低／中／高の境目（Hz）。キック・スネア・ハイハットが概ね別の枠に入る値です。
BAND_EDGES = [200, 2000]

DEFAULT_MIN_BPM = 60
DEFAULT_MAX_BPM = 240
# 1 小節の拍数。Movo の musical-time と同じく 4/4 を既定にしています。
DEFAULT_BEATS_PER_BAR = 4

# «細かいグリッドほど当たりやすい» ぶんの割引。
#
# 拍グリッドの点数を «1 点あたりの平均» で採ると、点が少ない小節グリッドが
# 必ず勝ってしまい BPM が 1/4 になります。逆に «合計» で採ると 16 分グリッドが
# 勝ってしまいます。点数 × 点の個数^GRID_PENALTY で釣り合わせています。
GRID_PENALTY = 0.5

# 山を立てるために引く移動平均の «片側» の長さ（秒）。
SHARPEN_SECONDS = 0.1

# 自己相関のピークから «倍・半分» の候補も作る。オクターブ誤りの保険です。
# 3 倍・1/3 倍は入れていません（1.5 倍テンポが混ざって、音の繋がった曲で
# そちらが勝ってしまいました）。
TEMPO_MULTIPLES = [0.25, 0.5, 1, 2, 4]


# ================================================================== #
# オンセット強度                                                      #
# ================================================================== #


@njit(cache=True)
def _band_block_sums(mono, hop_size, blocks, low_coeff, mid_coeff):
    """1 極フィルタで 3 帯域に分け、hop ごとの二乗和を貯める。

    **帯域信号を丸ごと配列に持ちません。** 5 分の曲でも追加のメモリは
    hop の数ぶん（数千要素）で済みます。
    """
    sums = np.zeros((4, blocks), np.float64)
    low_state = 0.0
    mid_state = 0.0
    limit = blocks * hop_size
    for i in range(limit):
        value = mono[i]
        low_state = value * (1.0 - low_coeff) + low_state * low_coeff
        mid_state = value * (1.0 - mid_coeff) + mid_state * mid_coeff
        low = low_state
        mid = mid_state - low_state
        high = value - mid_state
        block = i // hop_size
        sums[0, block] += low * low
        sums[1, block] += mid * mid
        sums[2, block] += high * high
        sums[3, block] += value * value
    return sums


def onset_envelope(audio, options: dict | None = None) -> dict:
    """帯域別のオンセット強度と、全体の RMS 包絡を作る。

    :returns: ``{"hop", "frames", "onset", "rms", "bands"}``

    ``bands`` は ``(4, frames)`` で、順に **低・中・高・全帯域** の包絡です。
    ``rms`` はその 4 本目（全帯域）で、**同じ配列**を指しています。歌詞を曲に
    合わせるとき（`align.py`）に «中域が全体に占める比率» が要るので、
    帯域ごとの包絡をここで捨てずに返しています。
    """
    options = options or {}
    audio = as_audio(audio)
    hop = float(HOP_SECONDS if options.get("hop") is None else options["hop"])
    hop_size = max(1, round(audio.sample_rate * hop))
    blocks = max(WINDOW_HOPS, audio.length // hop_size)
    frames = max(1, blocks - (WINDOW_HOPS - 1))
    count = channel_count(audio) or 1

    limit = blocks * hop_size
    mono = np.zeros(limit, np.float64)
    usable = min(limit, audio.length)
    for channel in audio.channels:
        take = min(usable, len(channel))
        if take > 0:
            mono[:take] += channel[:take].astype(np.float64)
    mono /= count

    low_coeff = math.exp((-2 * math.pi * BAND_EDGES[0]) / audio.sample_rate)
    mid_coeff = math.exp((-2 * math.pi * BAND_EDGES[1]) / audio.sample_rate)
    block_sums = _band_block_sums(mono, hop_size, blocks, low_coeff, mid_coeff)

    window_samples = hop_size * WINDOW_HOPS
    # 窓は hop 2 つぶん。ずらして足すだけなので «スライスの足し算» で作れます。
    band_env = np.empty((4, frames), np.float64)
    for band in range(4):
        total = np.zeros(frames, np.float64)
        for w in range(WINDOW_HOPS):
            total += block_sums[band, w: w + frames]
        band_env[band] = np.sqrt(total / window_samples)

    onset = np.zeros(frames, np.float64)
    for band in range(3):
        env = band_env[band]
        # 帯域ごとの «平均音量» で割ってから差を取ります。割らないと、いちばん
        # 音量のある帯域（普通は低域）だけでオンセットが決まってしまいます。
        mean = env.sum() / frames + 1e-9
        diff = np.zeros(frames, np.float64)
        diff[1:] = np.maximum(0.0, (env[1:] - env[:-1]) / mean)
        onset += diff

    # 前後 SHARPEN_SECONDS の平均を引いて山を立てます。バラードのように音が
    # 途切れずに繋がっている曲では立ち上がりが «なだらかな丘» になり、どこに
    # 拍グリッドを置いても同じ点数になってしまうためです。
    sharpen = max(1, round(SHARPEN_SECONDS / hop))
    smoothed = _moving_average(onset, sharpen)
    onset = np.maximum(0.0, onset - smoothed)

    peak = float(onset.max()) if frames else 0.0
    if peak > 0:
        # 平方根で潰します。生の値のままだとコードが変わる «小節頭» だけが極端に
        # 大きく、他の拍が «鳴っていない» のと同じ扱いになってしまいます。
        onset = np.sqrt(onset / peak)
    return {"hop": hop, "frames": frames, "onset": onset, "rms": band_env[3], "bands": band_env}


def _moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    """端では «届いた枠だけ» で平均する移動平均（JS 版と同じ数え方）。"""
    n = values.size
    padded = np.concatenate([[0.0], np.cumsum(values)])
    index = np.arange(n)
    low = np.maximum(0, index - radius)
    high = np.minimum(n - 1, index + radius) + 1
    return (padded[high] - padded[low]) / (high - low)


# ================================================================== #
# テンポ                                                              #
# ================================================================== #


@njit(cache=True)
def _sample_at(signal, x):
    """枠の間を線形で読む。拍の位置は枠の整数倍にはなりません。"""
    n = signal.shape[0]
    if x < 0 or x > n - 1:
        return 0.0
    i = int(math.floor(x))
    frac = x - i
    nxt = signal[n - 1] if i + 1 > n - 1 else signal[i + 1]
    return signal[i] * (1.0 - frac) + nxt * frac


@njit(cache=True)
def _peak_at(signal, x):
    """前後 1 枠まで見て山を拾う。10ms の量子化ぶんのずれを取りこぼさないため。"""
    a = _sample_at(signal, x - 1.0)
    b = _sample_at(signal, x)
    c = _sample_at(signal, x + 1.0)
    best = a
    if b > best:
        best = b
    if c > best:
        best = c
    return best


@njit(cache=True)
def _comb_at(onset, period, phase):
    """周期 period・位相 phase の «くし» をかけて、当たった山の平均と本数を返す。"""
    total = 0.0
    count = 0
    limit = onset.shape[0] - 1
    x = phase
    while x < limit:
        total += _peak_at(onset, x)
        count += 1
        x += period
    return (total / count if count > 0 else 0.0), count


@njit(cache=True)
def _best_phase_for(onset, period, step):
    """その周期でいちばん «山に乗る» 位相を探す。1 周期ぶん見れば足ります。"""
    best_mean = -1.0
    best_phase = 0.0
    best_count = 0
    phase = 0.0
    while phase < period:
        mean, count = _comb_at(onset, period, phase)
        if mean > best_mean:
            best_mean = mean
            best_phase = phase
            best_count = count
        phase += step
    return best_mean, best_phase, best_count


@njit(cache=True)
def _refine_period(onset, period, span, steps):
    """周期を span の幅で振りながら、いちばん点数の高い（周期, 位相）を選ぶ。"""
    best_score = -1.0
    best_period = period
    best_phase = 0.0
    best_mean = 0.0
    found = False
    for i in range(steps + 1):
        candidate = period * (1.0 - span + (2.0 * span * i) / steps)
        mean, phase, count = _best_phase_for(onset, candidate, 0.05)
        score = mean * (float(count) ** 0.5)
        if not found or score > best_score:
            found = True
            best_score = score
            best_period = candidate
            best_phase = phase
            best_mean = mean
    return best_score, best_period, best_phase, best_mean


@njit(cache=True)
def _downbeat_offset(onset, hop, first_beat, period, beats_per_bar):
    """拍のうち «どれが小節頭か» を選ぶ。

    4 拍のうち 1 拍目はコードが変わることが多く、オンセットがいちばん大きく
    なります。0〜3 拍ずらして総和を比べるだけで実用上は足ります。
    """
    best_offset = 0
    best_sum = -1.0
    limit = onset.shape[0] - 1
    for offset in range(beats_per_bar):
        total = 0.0
        beat = offset
        while True:
            x = (first_beat + beat * period) / hop
            if x > limit:
                break
            total += _peak_at(onset, x)
            beat += beats_per_bar
        if total > best_sum:
            best_sum = total
            best_offset = offset
    return best_offset


def _autocorrelation(onset: np.ndarray, min_lag: int, max_lag: int) -> np.ndarray:
    """自己相関。**FFT で作りますが、値は素直に回したものと同じ**です。

    `r[lag] = Σ onset[i] * onset[i+lag] / (frames - lag)`

    5 分の曲だと frames が 30,000、lag が 300 通りで 9 百万回の積和です。
    FFT なら 1 回の変換で全部の lag が出ます（実測 900ms → 4ms）。
    """
    frames = onset.size
    size = 1
    while size < frames * 2:
        size *= 2
    spectrum = np.fft.rfft(onset, size)
    full = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: max_lag + 1]
    correlation = np.zeros(max_lag + 1, np.float64)
    lags = np.arange(min_lag, max_lag + 1)
    correlation[min_lag:] = full[min_lag:] / (frames - lags)
    return correlation


def estimate_tempo(onset, hop: float, options: dict | None = None) -> dict:
    """オンセット強度から BPM と 1 拍目を推定する。

    自己相関でおおよその周期を出し、そこから «倍・半分» の候補を作って
    くしフィルタで採点し直します。自己相関のピークだけで決めると、拍ではなく
    小節や 8 分音符の周期を掴むことがあるためです。

    :returns: ``{"bpm", "period", "firstBeat", "confidence"}``
    """
    options = options or {}
    onset = np.ascontiguousarray(onset, np.float64)
    min_bpm = max(20, DEFAULT_MIN_BPM if options.get("minBpm") is None else options["minBpm"])
    max_bpm = max(min_bpm + 1, DEFAULT_MAX_BPM if options.get("maxBpm") is None else options["maxBpm"])
    frames = onset.size
    min_lag = max(2, math.floor(60 / max_bpm / hop))
    max_lag = min(frames - 2, math.ceil(60 / min_bpm / hop))
    empty = {"bpm": 0, "period": 0, "firstBeat": 0, "confidence": 0}
    if max_lag <= min_lag:
        return empty

    correlation = _autocorrelation(onset, min_lag, max_lag)

    peaks = []
    for lag in range(min_lag + 1, max_lag):
        if correlation[lag] > correlation[lag - 1] and correlation[lag] >= correlation[lag + 1]:
            peaks.append((lag, correlation[lag]))
    if not peaks:
        return empty
    # JS の `sort((a, b) => b.value - a.value)` は安定ソートです。Python の
    # `sorted` も安定なので、同点の並びまで一致します。
    peaks.sort(key=lambda p: -p[1])

    candidates: dict[float, None] = {}
    for lag, _ in peaks[:8]:
        for multiple in TEMPO_MULTIPLES:
            period = lag * multiple
            bpm = 60 / (period * hop)
            if min_bpm <= bpm <= max_bpm:
                candidates[js_round(period * 100) / 100] = None

    # 2 段階で絞ります。全候補を細かく振ると «拍の数 × 位相 × 周期» の三重ループが
    # 効いてきて、5 分の曲で数十秒かかります。
    scored = [(*_refine_period(onset, period, 0.02, 8),) for period in candidates]
    scored.sort(key=lambda entry: -entry[0])
    winner_score, winner_period, winner_phase, winner_mean = _refine_period(onset, scored[0][1], 0.02, 80)

    # 確からしさ。«拍の位置だけ突出しているか»（contrast）と «2 位を引き離せたか»
    # （separation）の掛け算です。どちらか片方だけだと、鳴りっぱなしの音や
    # 単調なループでも高い値が出てしまいます。
    overall = float(onset.sum()) / frames + 1e-9
    contrast = clamp(1 - overall / (winner_mean or 1), 0.0, 1.0)
    runner_up = None
    for entry in scored:
        if abs(entry[1] - scored[0][1]) > scored[0][1] * 0.05:
            runner_up = entry
            break
    separation = clamp(1 - runner_up[0] / (winner_score or 1), 0.0, 1.0) if runner_up else 1.0
    confidence = clamp(contrast * (0.55 + 0.45 * separation), 0.0, 1.0)

    return {
        "bpm": 60 / (winner_period * hop),
        "period": winner_period * hop,
        "firstBeat": winner_phase * hop,
        "confidence": confidence,
    }


# ================================================================== #
# 区間                                                                #
# ================================================================== #


def detect_sections(rms, hop: float, options: dict) -> list[dict]:
    """RMS の移動平均が «大きく変わって、そのまま続く» ところで曲を切る。

    切れ目の候補を小節の頭に限っているのは、区間の頭が拍の途中に来ると
    シーンを貼ったときに気持ち悪いからです。音楽的にも間奏やサビは小節の頭で
    始まります。
    """
    rms = np.asarray(rms, np.float64)
    bar_seconds = options["barSeconds"] if options.get("barSeconds", 0) > 0 else 2
    duration = options["duration"]
    min_bars = max(1, options.get("minBars", 2))
    # «最大音量の 15% ぶん変わったら別の区間» という目安。これより緩めると
    # 8 小節ごとのフィルインで切れ、厳しくすると間奏を見落とします。
    threshold = options.get("threshold", 0.15)
    bar_count = max(1, math.floor(duration / bar_seconds))
    if bar_count < min_bars * 2:
        return _label_sections(
            [{"start": 0, "end": duration, "energy": _average_rms(rms, hop, 0, duration), "bars": bar_count}]
        )

    bar_energy = np.array(
        [_average_rms(rms, hop, bar * bar_seconds, (bar + 1) * bar_seconds) for bar in range(bar_count)],
        np.float64,
    )
    high = float(bar_energy.max()) if bar_count else 0.0
    # «いちばん派手な小節» を 1 として比べます。最小値との差で正規化すると、
    # 起伏の無いループでも誤差が拡大されて細切れになってしまうためです。
    scale = high or 1

    # «いま続いている区間の平均» から離れた小節を切れ目にします。1 小節だけの
    # 突発的な変化（フィルインなど）で切らないよう、最短の長さを課しています。
    cuts = [0]
    total = float(bar_energy[0])
    length = 1
    for bar in range(1, bar_count):
        level = bar_energy[bar] / scale
        reference = total / length / scale
        if length >= min_bars and abs(level - reference) >= threshold:
            cuts.append(bar)
            total = float(bar_energy[bar])
            length = 1
            continue
        total += float(bar_energy[bar])
        length += 1

    sections = []
    for i, start_bar in enumerate(cuts):
        end_bar = cuts[i + 1] if i + 1 < len(cuts) else bar_count
        start = start_bar * bar_seconds
        end = end_bar * bar_seconds if i + 1 < len(cuts) else duration
        sections.append({
            "start": start, "end": end,
            "energy": _average_rms(rms, hop, start, end),
            "bars": end_bar - start_bar,
        })
    return _label_sections(sections)


def _average_rms(rms: np.ndarray, hop: float, start: float, end: float) -> float:
    frm = max(0, math.floor(start / hop))
    to = min(rms.size, math.ceil(end / hop))
    if to <= frm:
        return 0.0
    return float(rms[frm:to].sum()) / (to - frm)


def _label_sections(sections: list[dict]) -> list[dict]:
    """エネルギーの «順位» でラベルを仮に付ける。

    音色を見ていないので «本当に» サビかどうかは分かりません。それでも
    「いちばん元気なところがサビ、頭と尻の静かなところが intro / outro」は
    大抵当たるので、シーンを割り当てる取っ掛かりとしては十分です。
    """
    peak = max((s["energy"] for s in sections), default=0.0)
    scale = peak or 1
    out = []
    for index, section in enumerate(sections):
        relative = section["energy"] / scale
        label = "bridge"
        if index == 0 and relative < 0.72:
            label = "intro"
        elif index == len(sections) - 1 and len(sections) > 1 and relative < 0.72:
            label = "outro"
        elif relative >= 0.85:
            label = "chorus"
        elif relative >= 0.6:
            label = "verse"
        out.append({
            "start": _round(section["start"], 3),
            "end": _round(section["end"], 3),
            "energy": _round(relative, 3),
            "label": label,
            "bars": section["bars"],
        })
    return out


def _round(value: float, digits: int) -> float:
    factor = 10 ** digits
    return js_round(value * factor) / factor


# ================================================================== #
# 全体                                                                #
# ================================================================== #


def analyze_audio(audio, options: dict | None = None) -> dict:
    """音源を丸ごと解析する。`movo analyze` が出す JSON はこの戻り値そのものです。"""
    options = options or {}
    audio = as_audio(audio)
    duration = audio.length / audio.sample_rate
    beats_per_bar = max(1, js_round(options.get("beatsPerBar", DEFAULT_BEATS_PER_BAR)))
    envelope = onset_envelope(audio, options)
    hop = envelope["hop"]
    onset = envelope["onset"]
    tempo = estimate_tempo(onset, hop, options)

    beats: list[float] = []
    bars: list[float] = []
    bar_seconds = duration
    first_beat = 0.0
    if tempo["bpm"] > 0:
        period = tempo["period"]
        # 1 拍目は «最初の山» に合わせたいので、位相から 1 周期ずつ戻せるだけ戻します。
        first = tempo["firstBeat"]
        while first - period >= 0:
            first -= period
        first_beat = _round(first, 4)
        bar_seconds = period * beats_per_bar
        offset = _downbeat_offset(onset, hop, first, period, beats_per_bar)
        # maxBeats に 0 を渡すと «拍の一覧は要らない» の意味になります。
        limit = options.get("maxBeats")
        limit = math.inf if limit is None else limit
        index = 0
        while len(beats) < limit:
            time = first + index * period
            if time > duration:
                break
            beats.append(_round(time, 4))
            if index >= offset and (index - offset) % beats_per_bar == 0:
                bars.append(_round(time, 4))
            index += 1

    sections = detect_sections(envelope["rms"], hop, {"barSeconds": bar_seconds, "duration": duration})
    return {
        "duration": _round(duration, 3),
        "sampleRate": audio.sample_rate,
        "bpm": _round(tempo["bpm"], 2),
        "confidence": _round(tempo["confidence"], 3),
        "firstBeat": first_beat,
        "beatsPerBar": beats_per_bar,
        "beats": beats,
        "bars": bars,
        "sections": sections,
    }


# ================================================================== #
# ファイルを読む                                                      #
# ================================================================== #

FFMPEG_READABLE = (".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma", ".mp4", ".mov", ".webm")


class AudioDecodeError(ValueError):
    """音声として読めなかった。core の MovoError が入れば置き換えます。"""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


def decode_audio_file(file: str):
    """音声ファイルを読む。WAV は自前で、それ以外は ffmpeg があれば WAV に直して読む。

    自前のデコーダを増やさないのは «音を解析するために mp3 デコーダを書く» のが
    明らかに割に合わないからです。ffmpeg が無い環境では、黙って失敗せずに
    «WAV に変換してください» と案内します。
    """
    if not os.path.exists(file):
        raise AudioDecodeError(f"音声ファイルが見つかりません: {file}")
    with open(file, "rb") as handle:
        buffer = handle.read()
    try:
        return decode_wav(buffer)
    except Exception as error:
        converted = _convert_with_ffmpeg(file)
        if converted is not None:
            return converted
        hint = (
            "ffmpeg を入れるか、WAV に変換してから渡してください（ffmpeg -i 入力 出力.wav）"
            if file.lower().endswith(FFMPEG_READABLE)
            else "この形式は読めません。WAV に変換してから渡してください"
        )
        raise AudioDecodeError(
            f"{os.path.basename(file)} を音声として読めませんでした: {error}", hint
        ) from error


def _convert_with_ffmpeg(file: str):
    """ffmpeg で 48kHz モノラルの WAV に落として読む。解析にはステレオは要りません。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not ffmpeg.get("path"):
        return None
    directory = tempfile.mkdtemp(prefix="movo-analyze-")
    output = os.path.join(directory, "source.wav")
    try:
        result = subprocess.run(
            [ffmpeg["path"], "-y", "-loglevel", "error", "-i", file, "-ar", "48000", "-ac", "1", output],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if result.returncode == 0 and os.path.exists(output):
            with open(output, "rb") as handle:
                return decode_wav(handle.read())
        return None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def resolve_bpm_from_audio(project: dict, assets=None, options: dict | None = None):
    """`project.bpm` に `{"fromAudio": "track"}` と書かれていたら解析して埋める。

    **正規化より前に呼ぶ必要があります。** `"4bar"` のような拍の書き方は
    正規化のときに `project.bpm` を数値として読むので、そこまでにオブジェクトを
    数値へ潰しておかないと «project.bpm がありません» で落ちます。

    :param project: プロジェクト JSON（**破壊的に書き換えます**）
    """
    options = options or {}
    spec = (project.get("project") or {}).get("bpm")
    # `project.structure.fromAudio` だけ書いた人も居ます（BPM は要らないが構成は
    # 曲に合わせたい、という書き方）。どちらの入口でも同じ解析を回します。
    structure_spec = project.get("structure")
    asset = None
    if isinstance(spec, dict):
        asset = spec.get("fromAudio")
    if asset is None and isinstance(structure_spec, dict):
        asset = structure_spec.get("fromAudio")
    if not asset:
        return None

    audio = None
    if assets is not None:
        getter = getattr(assets, "get_audio", None) or getattr(assets, "getAudio", None)
        if getter is None and isinstance(assets, dict):
            getter = assets.get("getAudio") or assets.get("get_audio")
        if getter is not None:
            audio = getter(asset)
    if audio is None:
        audio = _load_declared_audio(project, asset, options.get("projectRoot"))
    if audio is None:
        raise AudioDecodeError(
            f'project.bpm.fromAudio が指す音声素材 "{asset}" が読めません',
            "project.assets に audio として宣言されているか確認してください",
        )

    # 拍の一覧は数千件になることがあり、プロジェクトに丸ごと持たせても使い道が
    # ないので、区間と BPM だけ残します。
    analysis = analyze_audio(audio, {**options, "maxBeats": 0})
    fallback = spec.get("fallback", 120) if isinstance(spec, dict) else 120
    bpm = analysis["bpm"] if analysis["bpm"] > 0 else fallback
    # BPM を «曲から» と書いていないなら、勝手に上書きしません。
    if isinstance(spec, dict) and spec.get("fromAudio"):
        project["project"]["bpm"] = bpm
    elif project["project"].get("bpm") is None:
        project["project"]["bpm"] = bpm
    project["_audioAnalysis"] = {
        "asset": asset, "bpm": bpm,
        "confidence": analysis["confidence"], "sections": analysis["sections"],
    }
    # 公開用。`_audioAnalysis` は内部の置き場なので、参照させるのは structure のほう。
    project["structure"] = {
        **(structure_spec if isinstance(structure_spec, dict) else {}),
        "sections": analysis["sections"],
    }
    return {"bpm": bpm, "analysis": analysis}


def _load_declared_audio(project: dict, name: str, project_root: str | None):
    """`project.assets` の宣言からファイルを引いて読む。"""
    declared = (project.get("assets") or {}).get(name)
    relative = declared if isinstance(declared, str) else (declared or {}).get("path")
    if not relative:
        return None
    path = relative if os.path.isabs(relative) else os.path.join(project_root or os.getcwd(), relative)
    return decode_audio_file(path)
