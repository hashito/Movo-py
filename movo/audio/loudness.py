"""ラウドネス（ITU-R BS.1770 / EBU R128）の測定と正規化、そしてトゥルーピーク。

これまでのミックスは «ピークが 1 を超えたら割る» だけでした。ピークは
«いちばん大きい 1 サンプル» の話なので、**人が感じる音量とはほとんど関係が
ありません**。ピークだけ揃えた MV を 10 本作って続けて再生すると、曲ごとに
体感音量がばらばらのまま出てきます。

やっていることは 3 つです。

  1. **K 特性フィルタ** … «高い音ほど大きく聞こえる» 耳の傾向を 2 段の
     双二次フィルタで模す（BS.1770 の中身はほぼこれだけです）
  2. **ゲーティング** … 400ms のブロックに切って静かなブロックを捨てる。
     **絶対閾値 −70 LUFS と相対閾値 −10 LU の 2 段**です。ここを省くと、
     間奏や無音の長い曲が «静かな曲» と判定されて音量を上げすぎます
  3. **トゥルーピーク** … 4 倍オーバーサンプリングしてから最大値を見る。
     サンプルとサンプルの «間» に立つ山を拾うためです

## なぜ Numba か

**K 特性は IIR（無限インパルス応答）で、1 サンプル前の出力が要ります。**
NumPy では «1 本の掛け算» に畳めません（`lfilter` のような C 実装は
SciPy にしかなく、依存を増やせません）。

| 30 秒・48kHz を 1 チャンネル | |
| --- | --- |
| 純 Python の逐次ループ | 1,237 ms |
| **Numba** | **4.76 ms**（260 倍） |

`measure_loudness` 全体（30 秒ステレオ）で **17.2 ms** です。ラウドネス
正規化は測り直しで最大 4 回ここを通るので、純 Python だと 5 分の曲 1 本で
**3 分以上**かかる計算になります。トゥルーピークの補間とリミッターの
«なまし» も同じ理由で Numba です。

外部依存はゼロです。フィルタ係数も補間フィルタも、この場で設計しています。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._compat import as_audio, channels_2d, scale_channels, verbose, warn, write_channels

# BS.1770 のラウドネス式に付く定数。K 特性のエネルギーを LUFS に直す下駄です。
LOUDNESS_OFFSET = -0.691

# 解析ブロックの長さ（秒）。BS.1770 が 400ms と決めています。
BLOCK_SECONDS = 0.4

# ブロックの重なり。75% 重ねる（＝ 100ms 刻み）のも規格どおりです。
BLOCK_OVERLAP = 0.75

# ゲーティングの 2 段。**片方だけでは足りません。** 絶対閾値だけだと静かな
# 部分の暗騒音まで数えて平均が下がり、相対閾値だけだと完全な無音が平均を壊します。
ABSOLUTE_GATE = -70
RELATIVE_GATE = -10

# チャンネルごとの重み（BS.1770 の G_i）。L/R/C は 1、サラウンドだけ 1.41。
CHANNEL_WEIGHTS = [1.0, 1.0, 1.0, 1.41, 1.41]

# トゥルーピークのオーバーサンプリング倍率。BS.1770-4 が最低 4 倍と定めています。
OVERSAMPLE = 4

# 補間フィルタの «1 位相あたりの» タップ数。規格の参考実装は 12 ですが、
# 8 に落としてあります。実測との差は 0.1dB 未満で、リミッターの天井を
# 決めるには十分でした。
OVERSAMPLE_TAPS = 8

# 使える規格の名前。**測定そのものはどれも BS.1770 で同じ**です。
LOUDNESS_STANDARDS = {
    "ebu-r128": {"target": -23, "truePeak": -1},
    "itu-bs1770": {"target": -23, "truePeak": -1},
    "atsc-a85": {"target": -24, "truePeak": -2},
    "streaming": {"target": -14, "truePeak": -1},
}

# 何も書かなかったときの既定。配信先（YouTube / Spotify）に合わせて −14 LUFS。
DEFAULT_TARGET = -14
DEFAULT_TRUE_PEAK = -1


# ================================================================== #
# K 特性フィルタ                                                      #
# ================================================================== #


def k_weighting_stages(sample_rate: float):
    """K 特性の 2 段を «その標本化周波数向けに» 設計する。

    BS.1770 の本文には 48kHz の係数表しか載っていません。表の数字をそのまま
    貼ると 44.1kHz の素材で特性がずれるので、表の元になっているアナログ原型
    （高域シェルビング＋ハイパス）から双一次変換で作り直しています。
    下の定数を入れて 48kHz で計算すると、**規格の表と小数 10 桁まで一致します**。

    :returns: 2 段ぶんの ``{"b": [...], "a": [...]}``（`a[0]` は 1 に正規化済み）
    """
    # 1 段目：高域シェルビング。頭による «高い音の持ち上がり» を模します。
    shelf_freq = 1681.974450955533
    shelf_gain = 3.999843853973347  # dB
    shelf_q = 0.7071752369554196
    k1 = math.tan((math.pi * shelf_freq) / sample_rate)
    vh = 10 ** (shelf_gain / 20)
    vb = vh ** 0.4996667741545416
    shelf_a0 = 1 + k1 / shelf_q + k1 * k1
    shelf = {
        "b": [
            (vh + (vb * k1) / shelf_q + k1 * k1) / shelf_a0,
            (2 * (k1 * k1 - vh)) / shelf_a0,
            (vh - (vb * k1) / shelf_q + k1 * k1) / shelf_a0,
        ],
        "a": [1.0, (2 * (k1 * k1 - 1)) / shelf_a0, (1 - k1 / shelf_q + k1 * k1) / shelf_a0],
    }

    # 2 段目：38Hz のハイパス（RLB 特性）。超低域はエネルギーの割に聞こえません。
    hp_freq = 38.13547087602444
    hp_q = 0.5003270373238773
    k2 = math.tan((math.pi * hp_freq) / sample_rate)
    hp_a0 = 1 + k2 / hp_q + k2 * k2
    high_pass = {
        "b": [1.0, -2.0, 1.0],
        "a": [1.0, (2 * (k2 * k2 - 1)) / hp_a0, (1 - k2 / hp_q + k2 * k2) / hp_a0],
    }
    return [shelf, high_pass]


@njit(cache=True)
def _k_weighted_hop_sums(channel, coefficients, hop, hop_count):
    """K 特性を掛けた波形の «hop ごとの二乗和» を作る。

    **波形そのものは配列に持ちません。** 5 分の曲を 1 本持つだけで 60MB
    増え、ラウドネス正規化はここを何度も通ります。1 サンプルずつ流しながら
    hop の枠に足していけば、余分なメモリは hop の数（数千）で済みます。
    """
    s1b0, s1b1, s1b2, s1a1, s1a2, s2b0, s2b1, s2b2, s2a1, s2a2 = (
        coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4],
        coefficients[5], coefficients[6], coefficients[7], coefficients[8], coefficients[9],
    )
    hop_sum = np.zeros(hop_count, np.float64)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    u1 = 0.0
    u2 = 0.0
    v1 = 0.0
    v2 = 0.0
    for i in range(channel.shape[0]):
        x0 = channel[i]
        y0 = s1b0 * x0 + s1b1 * x1 + s1b2 * x2 - s1a1 * y1 - s1a2 * y2
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
        v0 = s2b0 * y0 + s2b1 * u1 + s2b2 * u2 - s2a1 * v1 - s2a2 * v2
        u2 = u1
        u1 = y0
        v2 = v1
        v1 = v0
        hop_sum[i // hop] += v0 * v0
    return hop_sum


def measure_loudness(audio) -> dict:
    """積分ラウドネス（LUFS）を測る。

    :returns: ``{"lufs", "blocks", "gatedBlocks", "threshold"}``
        完全な無音では `lufs` が `-inf` になります。0 を返すと «普通の音量» と
        区別が付かず、正規化で無限に増幅してしまうためです。
    """
    audio = as_audio(audio)
    sample_rate = audio.sample_rate
    block_size = max(1, round(BLOCK_SECONDS * sample_rate))
    hop = max(1, round(block_size * (1 - BLOCK_OVERLAP)))
    length = audio.length
    blocks = (length - block_size) // hop + 1 if length >= block_size else 0
    if blocks == 0:
        return {"lufs": -math.inf, "blocks": 0, "gatedBlocks": 0, "threshold": -math.inf}

    stages = k_weighting_stages(sample_rate)
    coefficients = np.array(
        [stages[0]["b"][0], stages[0]["b"][1], stages[0]["b"][2], stages[0]["a"][1], stages[0]["a"][2],
         stages[1]["b"][0], stages[1]["b"][1], stages[1]["b"][2], stages[1]["a"][1], stages[1]["a"][2]],
        np.float64,
    )
    hops_per_block = max(1, round(block_size / hop))
    hop_count = math.ceil(length / hop)
    power = np.zeros(blocks, np.float64)

    for c, channel in enumerate(audio.channels):
        hop_sum = _k_weighted_hop_sums(
            np.ascontiguousarray(channel[: length], np.float64), coefficients, hop, hop_count
        )
        weight = CHANNEL_WEIGHTS[c] if c < len(CHANNEL_WEIGHTS) else 1.0
        # ブロックは hop の枠を hops_per_block 本ぶん足したもの（75% 重なり）。
        total = np.zeros(blocks, np.float64)
        for h in range(hops_per_block):
            usable = min(blocks, hop_count - h)
            if usable <= 0:
                break
            total[:usable] += hop_sum[h: h + usable]
        power += (weight * total) / block_size

    with np.errstate(divide="ignore"):
        block_loudness = np.where(power > 0, LOUDNESS_OFFSET + 10 * np.log10(np.maximum(power, 1e-300)), -np.inf)

    # 1 段目：絶対閾値 −70 LUFS。**ラウドネスではなくパワーの平均**から相対閾値を
    # 作るのが肝で、dB のまま平均すると小さいブロックの影響が強く出すぎます。
    passed = block_loudness > ABSOLUTE_GATE
    count = int(passed.sum())
    if count == 0:
        return {"lufs": -math.inf, "blocks": int(blocks), "gatedBlocks": 0, "threshold": -math.inf}
    threshold = LOUDNESS_OFFSET + 10 * math.log10(float(power[passed].sum()) / count) + RELATIVE_GATE

    # 2 段目：相対閾値。
    gated = passed & (block_loudness > threshold)
    gated_count = int(gated.sum())
    if gated_count == 0:
        return {"lufs": -math.inf, "blocks": int(blocks), "gatedBlocks": 0, "threshold": threshold}

    return {
        "lufs": LOUDNESS_OFFSET + 10 * math.log10(float(power[gated].sum()) / gated_count),
        "blocks": int(blocks),
        "gatedBlocks": gated_count,
        "threshold": threshold,
    }


# ================================================================== #
# トゥルーピーク                                                      #
# ================================================================== #

_cached_interpolator = None


def _interpolator():
    """4 倍オーバーサンプリング用のポリフェーズ補間フィルタを組む。

    窓関数付き sinc（Blackman 窓）です。中心を «倍率の整数倍» の位置に置いて
    あるので、**位相 0 はちょうど恒等**（元のサンプルそのもの）になります。
    つまり «元のサンプルのピーク» は必ず候補に入ります。

    位相ごとに係数の和を 1 に揃えているのも意図的です。揃えないと直流に対する
    利得が 1 からずれ、ピークをわずかに大きく（あるいは小さく）見積もります。

    `gain` は «係数の絶対値の和» の最大値です。補間値は必ず
    «窓の中の最大絶対値 × gain» 以下なので、これを使うと
    «ここは調べなくても天井を超えない» を証明付きで飛ばせます。
    """
    global _cached_interpolator
    if _cached_interpolator is not None:
        return _cached_interpolator
    length = OVERSAMPLE_TAPS * OVERSAMPLE + 1
    center = (OVERSAMPLE_TAPS / 2) * OVERSAMPLE
    n = np.arange(length, dtype=np.float64)
    x = (n - center) / OVERSAMPLE
    sinc = np.where(x == 0, 1.0, np.sin(np.pi * np.where(x == 0, 1.0, x)) / (np.pi * np.where(x == 0, 1.0, x)))
    w = n / (length - 1)
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * w) + 0.08 * np.cos(4 * np.pi * w)
    prototype = sinc * blackman

    taps = OVERSAMPLE_TAPS + 1
    phases = np.zeros((OVERSAMPLE - 1, taps), np.float64)
    gain = 1.0
    for p in range(1, OVERSAMPLE):
        row = np.zeros(taps, np.float64)
        for k in range(taps):
            index = k * OVERSAMPLE + p
            row[k] = prototype[index] if index < length else 0.0
        row /= row.sum()
        absolute = float(np.abs(row).sum())
        if absolute > gain:
            gain = absolute
        phases[p - 1] = row
    _cached_interpolator = (phases, taps, gain)
    return _cached_interpolator


def _sample_magnitude(audio) -> np.ndarray:
    """サンプルごとの «全チャンネル中の最大絶対値»。float32 なのは JS 版と同じ。"""
    magnitude = np.zeros(audio.length, np.float32)
    for channel in audio.channels:
        usable = min(audio.length, len(channel))
        np.maximum(magnitude[:usable], np.abs(channel[:usable]), out=magnitude[:usable])
    return magnitude


def _candidates(magnitude: np.ndarray, gate: float, taps: int) -> np.ndarray:
    """`gate` を超えるサンプルが補間の窓に入る位置だけ True になる配列。

    JS 版は «直近で gate を超えたのは何サンプル前か» を持ち回る 1 周のループ
    でした。NumPy では `maximum.accumulate` で同じものが作れます。
    """
    index = np.arange(magnitude.size)
    hit = np.where(magnitude >= gate, index, -taps)
    last = np.maximum.accumulate(hit)
    return (index - last) < taps


@njit(cache=True)
def _interpolated_peak_at(channels, m, phases, taps):
    """サンプル位置 m の «間» に立つ補間値の最大絶対値（位相 0 は含めない）。"""
    peak = 0.0
    from_index = m - taps + 1
    for c in range(channels.shape[0]):
        for p in range(phases.shape[0]):
            value = 0.0
            if from_index >= 0:
                for k in range(taps):
                    value += phases[p, k] * channels[c, m - k]
            else:
                for k in range(m + 1):
                    value += phases[p, k] * channels[c, m - k]
            magnitude = -value if value < 0.0 else value
            if magnitude > peak:
                peak = magnitude
    return peak


@njit(cache=True)
def _scan_true_peak(channels, flags, phases, taps, raw):
    """候補の位置だけ補間して最大値を返す。"""
    peak = raw
    for m in range(flags.shape[0]):
        if not flags[m]:
            continue
        value = _interpolated_peak_at(channels, m, phases, taps)
        if value > peak:
            peak = value
    return peak


def measure_true_peak(audio) -> float:
    """トゥルーピーク（dBTP）。無音では `-inf`。"""
    linear = true_peak_linear(audio)
    return 20 * math.log10(linear) if linear > 0 else -math.inf


def true_peak_linear(audio) -> float:
    """dB に直す前のトゥルーピーク（線形）。

    全サンプルで補間を回すと 5 分の曲で数億回の積和になります。補間値は
    «窓の中の最大絶対値 × フィルタ利得» を超えられないので、生のピークを
    その利得で割った線より小さいところは **調べなくても超えないと分かります**。
    ここを飛ばすだけで、普通のミックスなら数%のサンプルしか触りません。
    """
    audio = as_audio(audio)
    if audio.length == 0:
        return 0.0
    magnitude = _sample_magnitude(audio)
    raw = float(magnitude.max())
    if raw == 0:
        return 0.0
    phases, taps, gain = _interpolator()
    flags = _candidates(magnitude, raw / gain, taps)
    return float(_scan_true_peak(channels_2d(audio), flags, phases, taps, raw))


# ================================================================== #
# トゥルーピーク・リミッター                                          #
# ================================================================== #

# ゲインを下げ始める早さ（秒）。短すぎると «プツッ» と鳴ります。
LIMITER_ATTACK = 0.0015

# ゲインを戻す遅さ（秒）。短いとポンピング（音量が波打つ）が耳に付きます。
LIMITER_RELEASE = 0.05


@njit(cache=True)
def _build_need(channels, magnitude, flags, phases, taps, ceiling):
    """天井を超えた所に «必要な倍率» を書き込む。戻り値は (need, 動いたか)。"""
    n = magnitude.shape[0]
    need = np.ones(n, np.float32)
    touched = False
    for m in range(n):
        value = float(magnitude[m])
        if flags[m]:
            interpolated = _interpolated_peak_at(channels, m, phases, taps)
            if interpolated > value:
                value = interpolated
        if value <= ceiling:
            continue
        touched = True
        ratio = ceiling / value
        start = m - taps + 1
        if start < 0:
            start = 0
        for k in range(start, m + 1):
            if ratio < need[k]:
                need[k] = ratio
    return need, touched


@njit(cache=True)
def _smooth_and_apply(channels, need, attack, release):
    """後ろ向き（アタック）→ 前向き（リリース）の 1 極フィルタを掛けて適用する。

    どちらも «必要な倍率との最小» を取りながら進むので、**なました結果が
    必要量を上回ることはありません**（＝天井を割りません）。
    """
    n = need.shape[0]
    state = 1.0
    for i in range(n - 1, -1, -1):
        candidate = state * attack + need[i] * (1.0 - attack)
        state = need[i] if need[i] < candidate else candidate
        need[i] = state
    state = 1.0
    lowest = 1.0
    for i in range(n):
        candidate = state * release + need[i] * (1.0 - release)
        state = need[i] if need[i] < candidate else candidate
        need[i] = state
        if state < lowest:
            lowest = state
        for c in range(channels.shape[0]):
            channels[c, i] = channels[c, i] * state
    return lowest


def limit_true_peak(audio, ceiling_db: float) -> dict:
    """トゥルーピークが天井を超えないところまで «超えている所だけ» 下げる。

    全体を一律に下げるやり方もあります（3 行で済みます）。やめたのは、1 発の
    キックのために曲全体が 3dB 小さくなり、**せっかく合わせたラウドネスが
    目標から外れてしまう**からです。

    :returns: ``{"reduction": 最大の減衰量（dB。0 なら何もしていない）}``
    """
    audio = as_audio(audio)
    if audio.length == 0:
        return {"reduction": 0.0}
    ceiling = 10 ** (ceiling_db / 20)
    magnitude = _sample_magnitude(audio)
    phases, taps, gain = _interpolator()
    flags = _candidates(magnitude, ceiling / gain, taps)

    channels = channels_2d(audio)
    need, touched = _build_need(channels, magnitude, flags, phases, taps, ceiling)
    if not touched:
        return {"reduction": 0.0}

    attack = math.exp(-1 / max(1, LIMITER_ATTACK * audio.sample_rate))
    release = math.exp(-1 / max(1, LIMITER_RELEASE * audio.sample_rate))
    lowest = _smooth_and_apply(channels, need, attack, release)
    write_channels(audio, channels)

    # 念のため実測。補間の相互作用でまだ超えていたら一律に詰めます。
    after = true_peak_linear(audio)
    if after > ceiling:
        scale = ceiling / after
        scale_channels(audio, scale)
        lowest *= scale
    return {"reduction": 20 * math.log10(lowest) if lowest > 0 else -math.inf}


# ================================================================== #
# 正規化                                                              #
# ================================================================== #

# 何回まで «測って掛け直す» か。リミッターが噛むとラウドネスが少し下がります。
MAX_PASSES = 4

# これ以下のずれは «合った» とみなす（LU）。完了条件の ±0.5 に対して十分な余裕。
TOLERANCE = 0.05


def resolve_loudness_spec(spec):
    """`output.loudness` の指定を、既定値込みの形に整える。

    :returns: ``{"target", "truePeak", "standard"}``。無効なら `None`
    """
    if spec is None or spec is False:
        return None
    options = spec if isinstance(spec, dict) else {}
    if options.get("enabled") is False:
        return None
    name = options.get("standard") if isinstance(options.get("standard"), str) else "ebu-r128"
    preset = LOUDNESS_STANDARDS.get(name)
    if preset is None:
        warn(f'output.loudness.standard "{name}" は知らない規格です。ebu-r128 として扱います')
    target = options.get("target")
    true_peak = options.get("truePeak")
    return {
        "target": float(target) if isinstance(target, (int, float)) and math.isfinite(target) else DEFAULT_TARGET,
        "truePeak": float(true_peak) if isinstance(true_peak, (int, float)) and math.isfinite(true_peak)
        else (preset["truePeak"] if preset else DEFAULT_TRUE_PEAK),
        "standard": name if preset else "ebu-r128",
    }


def normalize_loudness(audio, spec):
    """ラウドネスを目標へ合わせ、トゥルーピークの天井も守らせる。

    «測って掛けて終わり» にしていないのは、リミッターが噛むとラウドネスが
    少しだけ下がるからです。もう一度測って足りないぶんを足す、を最大 4 回
    繰り返します。リミッターは «減らす» 方向にしか働かないので発散しません。
    それでも届かない素材は、実測値を警告に出したうえでそこで止めます。
    **黙って目標を名乗らない**のが大事だと考えています。

    :param audio: 破壊的に書き換えます
    """
    settings = resolve_loudness_spec(spec)
    if settings is None:
        return None
    audio = as_audio(audio)

    input_lufs = measure_loudness(audio)["lufs"]
    if not math.isfinite(input_lufs):
        warn("ラウドネス正規化: 音が無音（もしくは −70 LUFS 未満）なので何もしません")
        return {**settings, "input": input_lufs, "output": input_lufs,
                "outputTruePeak": -math.inf, "gain": 0.0, "passes": 0, "limited": False}

    total_gain = 1.0
    measured = input_lufs
    limited = False
    passes = 0
    while passes < MAX_PASSES:
        delta = settings["target"] - measured
        if abs(delta) <= TOLERANCE:
            break
        step = 10 ** (delta / 20)
        scale_channels(audio, step)
        total_gain *= step
        passes += 1
        if limit_true_peak(audio, settings["truePeak"])["reduction"] < 0:
            limited = True
        nxt = measure_loudness(audio)["lufs"]
        if not math.isfinite(nxt):
            measured = nxt
            break
        # リミッターが噛んで «上げたのに上がらない» ときは、これ以上は追えません。
        stalled = delta > 0 and nxt <= measured + TOLERANCE
        measured = nxt
        if stalled:
            break

    # 一度も掛けていなくても天井は守らせます（元から超えている素材があるため）。
    if passes == 0 and limit_true_peak(audio, settings["truePeak"])["reduction"] < 0:
        limited = True
        measured = measure_loudness(audio)["lufs"]

    output_true_peak = measure_true_peak(audio)
    summary = (
        f"ラウドネス正規化: {_format(input_lufs)} → {_format(measured)} LUFS "
        f"(目標 {settings['target']}, ゲイン {_format(20 * math.log10(total_gain))}dB, "
        f"トゥルーピーク {_format(output_true_peak)}dBTP{'、リミッター作動' if limited else ''})"
    )
    if abs(measured - settings["target"]) > 0.5:
        warn(f"{summary} — 目標に届いていません。トゥルーピークの天井が先に当たっています")
    else:
        verbose(summary)

    return {
        **settings,
        "input": input_lufs,
        "output": measured,
        "outputTruePeak": output_true_peak,
        "gain": 20 * math.log10(total_gain),
        "passes": passes,
        "limited": limited,
    }


def _format(value: float) -> str:
    """ログ用。`-inf` を並べても読めないので置き換えます。"""
    return f"{value:.2f}" if math.isfinite(value) else "-inf"
