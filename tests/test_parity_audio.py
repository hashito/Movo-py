"""JS 版と «数値そのもの» を突き合わせる（音）。

基準の作り直しかた:

    node tests/data/parity_audio.mjs > tests/data/parity_audio.json

見ているもの:

  - WAV の読み書き（16/24/32 ビット）と再標本化 — **1 バイトまで同じか**
  - K 特性の係数（32k / 44.1k / 48k / 96kHz）
  - ラウドネス・ゲーティング・トゥルーピーク・リミッター・正規化
  - ダッキングの包絡とゲインカーブ
  - ミックス（フェード・ループ・パン・トリム・ダッキング）
  - `audio-reactive` の元になる包絡
  - オンセット強度・BPM・拍・小節・区間
  - **BPM が既知の自作音源 10 本**（`../Movo/examples/assets/audio`）

BPM は «当たっているか» だけでなく **JS 版と同じ値か** を見ています。
拍のグリッドはカット尺の基準になるので、0.02 BPM ずれるだけで 5 分後には
1 拍ぶんずれます。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from movo.audio import (
    analyze_audio, analyze_envelope, create_silence, decode_audio_file, decode_wav,
    detect_sections, detector_envelope, duck_gain_curve, encode_wav, estimate_tempo,
    k_weighting_stages, limit_true_peak, measure_loudness, measure_true_peak,
    mix_project_audio, normalize_loudness, onset_envelope, resample, resolve_duck_spec,
)
from movo.core.rng import create_random
from movo.core.wav import AudioBuffer

GOLDEN = json.loads((Path(__file__).parent / "data" / "parity_audio.json").read_text("utf-8"))
RATE = 48000
JS_AUDIO_DIR = Path(__file__).resolve().parents[2] / "Movo" / "examples" / "assets" / "audio"


# ── 検査用の音（JS 版の parity_audio.mjs と同じ作り） ─────────────────


def sine(seconds, amplitude=0.1, frequency=1000, phase=0.0, rate=RATE):
    audio = create_silence(seconds, rate, 2)
    i = np.arange(audio.length, dtype=np.float64)
    value = (amplitude * np.sin((2 * np.pi * frequency * i) / rate + phase)).astype(np.float32)
    # **必ず別の配列にします。** 同じ配列を 2 本のチャンネルに入れると、
    # 「全チャンネルに掛ける」処理が同じ配列を 2 度掛けてしまいます。
    audio.channels[0] = value
    audio.channels[1] = value.copy()
    return audio


def noise(seconds, amplitude=0.1, seed=7, rate=RATE):
    audio = create_silence(seconds, rate, 2)
    random = create_random(seed)
    # **1 サンプルずつ引きます。** まとめて引くと乱数の «順番» が変わり、
    # JS 版と違う波形になります（左・右・左・右の順）。
    for i in range(audio.length):
        audio.channels[0][i] = np.float32((random() * 2 - 1) * amplitude)
        audio.channels[1][i] = np.float32((random() * 2 - 1) * amplitude)
    return audio


def click_track(bpm, seconds, sample_rate=32000, offset=0.25):
    audio = create_silence(seconds, sample_rate, 1)
    random = create_random(1234)
    beat = 60 / bpm
    channel = audio.channels[0]
    time = offset
    while time < seconds:
        start = round(time * sample_rate)
        length = round(0.03 * sample_rate)
        for i in range(length):
            if start + i >= channel.size:
                break
            channel[start + i] = np.float32((random() * 2 - 1) * math.exp(-(i / length) * 6) * 0.8)
        time += beat
    return audio


def three_part_track(sample_rate=32000):
    seconds = 36
    audio = create_silence(seconds, sample_rate, 1)
    random = create_random(99)
    channel = audio.channels[0]
    for i in range(channel.size):
        time = i / sample_rate
        level = 0.06 if time < 12 else (0.8 if time < 24 else 0.06)
        channel[i] = np.float32((random() * 2 - 1) * level)
    return audio


def concat(*parts):
    length = sum(p.length for p in parts)
    out = AudioBuffer(sample_rate=parts[0].sample_rate,
                      channels=[np.zeros(length, np.float32) for _ in range(2)],
                      length=length)
    cursor = 0
    for part in parts:
        for c in range(2):
            out.channels[c][cursor: cursor + part.length] = part.channels[c]
        cursor += part.length
    return out


def num(value):
    """基準 JSON の "inf" / "-inf" を float に戻す。"""
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return value


def close(got, want, atol=1e-8):
    got = num(got)
    want = num(want)
    if not math.isfinite(want) or not math.isfinite(got):
        return got == want
    return abs(got - want) <= atol + abs(want) * 1e-9


# ── WAV ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_wav_bytes_match_js(bits):
    """**書き出した WAV が 1 バイトも変わらない**こと。"""
    audio = noise(0.05, amplitude=0.6, seed=3)
    encoded = encode_wav(audio, bits_per_sample=bits)
    want = GOLDEN["wav"][f"bits{bits}"]
    assert list(encoded[:64]) == want[:64]
    assert list(encoded[-32:]) == want[64:]
    decoded = decode_wav(encoded)
    assert np.allclose(decoded.channels[0][:40], GOLDEN["wav"][f"roundtrip{bits}"], rtol=0, atol=1e-9)


def test_resample_matches_js():
    audio = noise(0.05, amplitude=0.6, seed=3)
    resampled = resample(audio, 32000)
    assert resampled.length == GOLDEN["wav"]["resampleLength"]
    assert np.allclose(resampled.channels[1][:40], GOLDEN["wav"]["resample"], rtol=0, atol=1e-9)


# ── K 特性 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("rate", [32000, 44100, 48000, 96000])
def test_k_weighting_matches_js(rate):
    stages = k_weighting_stages(rate)
    want = GOLDEN["kWeighting"][str(rate)]
    for stage, expected in zip(stages, want):
        for a, b in zip(stage["b"], expected["b"]):
            assert abs(a - b) < 1e-14
        for a, b in zip(stage["a"], expected["a"]):
            assert abs(a - b) < 1e-14


def test_k_weighting_matches_the_standard_table():
    """規格（ITU-R BS.1770-4）の 48kHz 係数表そのものと照合する。"""
    shelf, high_pass = k_weighting_stages(48000)
    expected = {
        "shelfB": [1.53512485958697, -2.69169618940638, 1.19839281085285],
        "shelfA": [1, -1.69065929318241, 0.73248077421585],
        "highPassB": [1, -2, 1],
        "highPassA": [1, -1.99004745483398, 0.99007225036621],
    }
    for i in range(3):
        assert abs(shelf["b"][i] - expected["shelfB"][i]) < 1e-12
        assert abs(shelf["a"][i] - expected["shelfA"][i]) < 1e-12
        assert abs(high_pass["b"][i] - expected["highPassB"][i]) < 1e-12
        assert abs(high_pass["a"][i] - expected["highPassA"][i]) < 1e-12


# ── ラウドネス ────────────────────────────────────────────────────


def loudness_cases():
    return {
        "sine-23": sine(10, amplitude=10 ** (-23 / 20)),
        "sine-14": sine(6, amplitude=10 ** (-14 / 20)),
        "sine-100hz": sine(6, amplitude=0.2, frequency=100),
        "noise-6": noise(6, amplitude=0.5),
        "noise-26": noise(6, amplitude=0.05),
        "noise-44100": noise(6, amplitude=0.1, rate=44100),
        "gated": concat(noise(6, amplitude=0.2), noise(20, amplitude=0.2 * 10 ** (-26 / 20), seed=11)),
        "with-silence": concat(noise(6, amplitude=0.1), create_silence(20, RATE, 2)),
    }


@pytest.mark.parametrize("label", list(loudness_cases()))
def test_measure_loudness_matches_js(label):
    audio = loudness_cases()[label]
    measured = measure_loudness(audio)
    want = GOLDEN["loudness"][label]
    assert measured["blocks"] == want["blocks"]
    assert measured["gatedBlocks"] == want["gatedBlocks"]
    assert close(measured["lufs"], want["lufs"], atol=1e-6)
    assert close(measured["threshold"], want["threshold"], atol=1e-6)
    assert close(measure_true_peak(audio), want["truePeak"], atol=1e-6)


def test_ebu_tech_3341_test_signal():
    """EBU Tech 3341 の試験信号。**規格の許容は ±0.1 LU** です。"""
    for level in (-23, -20, -14, -6):
        measured = measure_loudness(sine(10, amplitude=10 ** (level / 20)))["lufs"]
        assert abs(measured - level) <= 0.1, f"{level} dBFS の正弦波が {measured:.3f} LUFS になりました"


def test_silence_is_negative_infinity():
    assert measure_loudness(create_silence(3, RATE, 2))["lufs"] == -math.inf


# ── トゥルーピークとリミッター ─────────────────────────────────────


def test_true_peak_matches_js():
    audio = sine(2, amplitude=0.5, frequency=RATE / 4, phase=math.pi / 4)
    assert close(measure_true_peak(audio), GOLDEN["truePeak"]["intersample"], atol=1e-6)
    # サンプル間のピークは «サンプルピーク» より約 3dB 上に立ちます。
    assert measure_true_peak(audio) - 20 * math.log10(0.5 / math.sqrt(2)) > 2.5
    assert close(measure_true_peak(sine(2, amplitude=0.5, frequency=100)), GOLDEN["truePeak"]["low"], atol=1e-6)


def test_limiter_matches_js():
    audio = noise(6, amplitude=0.3)
    spike = round(3 * RATE)
    audio.channels[0][spike: spike + 64] = np.float32(0.99)
    audio.channels[1][spike: spike + 64] = np.float32(0.99)
    before = measure_loudness(audio)["lufs"]
    result = limit_true_peak(audio, -6)
    want = GOLDEN["limiter"]
    assert close(before, want["before"], atol=1e-6)
    assert close(result["reduction"], want["reduction"], atol=1e-5)
    assert close(measure_loudness(audio)["lufs"], want["after"], atol=1e-5)
    assert close(measure_true_peak(audio), want["truePeak"], atol=1e-5)
    got = audio.channels[0][spike - 8: spike + 80]
    assert np.allclose(got, want["samples"], rtol=0, atol=1e-7)
    # 天井は守れているか（ここが本題）
    assert measure_true_peak(audio) <= -6 + 0.01


# ── 正規化 ────────────────────────────────────────────────────────


def normalize_cases():
    return {
        "quiet": lambda: sine(8, amplitude=10 ** (-30 / 20)),
        "loud": lambda: sine(8, amplitude=10 ** (-6 / 20)),
        "noise": lambda: noise(8, amplitude=0.05),
        "noise-loud": lambda: noise(8, amplitude=0.5),
        "low": lambda: sine(8, amplitude=0.2, frequency=100),
        "gapped": lambda: concat(noise(6, amplitude=0.1), create_silence(10, RATE, 2), noise(6, amplitude=0.1, seed=3)),
        "44100": lambda: noise(8, amplitude=0.1, rate=44100),
    }


@pytest.mark.parametrize("label", list(normalize_cases()))
def test_normalize_loudness_matches_js(label):
    audio = normalize_cases()[label]()
    info = normalize_loudness(audio, {"target": -14, "truePeak": -1, "standard": "ebu-r128"})
    want = GOLDEN["normalize"][label]
    assert info["passes"] == want["passes"]
    assert info["limited"] == want["limited"]
    assert close(info["input"], want["input"], atol=1e-6)
    assert close(info["output"], want["output"], atol=1e-5)
    assert close(info["gain"], want["gain"], atol=1e-5)
    assert np.allclose(audio.channels[0][1000:1040], want["samples"], rtol=0, atol=1e-7)
    # 完了条件そのもの: **−14 LUFS ±0.5** に入り、天井も超えない。
    verified = measure_loudness(audio)["lufs"]
    assert abs(verified - -14) <= 0.5, f"{label} が {verified:.3f} LUFS になりました"
    assert measure_true_peak(audio) <= -1 + 0.05


# ── ダッキング ────────────────────────────────────────────────────


def test_duck_curve_matches_js():
    narration = create_silence(8, RATE, 2)
    i = np.arange(round(2 * RATE), round(4 * RATE))
    value = (0.4 * np.sin((2 * np.pi * 300 * i) / RATE)).astype(np.float32)
    narration.channels[0][i] = value
    narration.channels[1][i] = value

    spec = resolve_duck_spec({"target": "track", "amount": -12, "attack": 0.08,
                              "release": 0.4, "threshold": -30, "hold": 0})
    assert spec == GOLDEN["duck"]["spec"]
    envelope = detector_envelope(narration, RATE)
    start = round(2 * RATE)
    assert np.allclose(envelope[start: start + 40], GOLDEN["duck"]["envelope"], rtol=0, atol=1e-8)

    curve = duck_gain_curve(envelope, RATE, spec)
    probe = [float(curve[min(curve.size - 1, round(s * RATE))]) for s in np.arange(0, 8.0001, 0.05)]
    assert np.allclose(probe, GOLDEN["duck"]["curve"], rtol=0, atol=1e-7)


# ── ミックス ──────────────────────────────────────────────────────


def test_mix_with_ducking_matches_js():
    bgm = create_silence(8, RATE, 2)
    i = np.arange(bgm.length)
    bgm.channels[0] = (0.3 * np.sin((2 * np.pi * 220 * i) / RATE)).astype(np.float32)
    bgm.channels[1] = bgm.channels[0].copy()
    narration = create_silence(8, RATE, 2)
    j = np.arange(round(3 * RATE), round(5 * RATE))
    value = (0.4 * np.sin((2 * np.pi * 900 * j) / RATE)).astype(np.float32)
    narration.channels[0][j] = value
    narration.channels[1][j] = value

    assets = {"getAudio": lambda name: {"track": bgm, "narration": narration}.get(name)}
    project = {
        "audio": [
            {"asset": "track", "volume": 1},
            {"asset": "narration",
             "ducks": [{"target": "track", "amount": -12, "attack": 0.08, "release": 0.4, "threshold": -30}]},
        ],
    }
    mixed = mix_project_audio(project, assets, {"duration": 8, "fps": 30})
    want = GOLDEN["mix"]
    assert mixed["tracks"] == want["tracks"]
    assert mixed["ducked"] == want["ducked"]
    assert mixed["loudness"] is None, "output.loudness を書いていないのに掛かっています"
    # **JS 版と同じ «0.1 を足し込む» ループ**にします。`np.arange` だと
    # 浮動小数の誤差の出方が違い、点の数が 1 つずれます。
    probe = []
    expected = []
    s = 0.0
    index = 0
    while s < 8:
        at = round(s * RATE)
        # 最後の 1 点はバッファの外を指します（JS 版では `undefined` ＝ null）。
        # そこは «どちらも読めない» ことだけ確かめて飛ばします。
        if at < mixed["audio"].length:
            probe.append([float(mixed["audio"].channels[0][at]), float(mixed["audio"].channels[1][at])])
            expected.append(want["probe"][index])
        else:
            assert want["probe"][index] == [None, None]
        s += 0.1
        index += 1
    assert np.allclose(probe, expected, rtol=0, atol=1e-7)


def test_mix_fades_loop_pan_trim_match_js():
    source = noise(3, amplitude=0.4, seed=17)
    assets = {"getAudio": lambda name: source if name == "track" else None}
    project = {
        "audio": [
            {"asset": "track", "start": 0.5, "offset": 0.25, "duration": 6, "volume": 0.8,
             "pan": -0.4, "fadeIn": 0.7, "fadeOut": 1.2, "loop": True},
            {"id": "second", "asset": "track", "start": 4, "volume": 1.4, "pan": 0.6},
        ],
    }
    mixed = mix_project_audio(project, assets, {"duration": 10, "fps": 30})
    assert mixed["tracks"] == GOLDEN["mixShapes"]["tracks"]
    probe = [
        [float(mixed["audio"].channels[0][i * 1201]), float(mixed["audio"].channels[1][i * 1201])]
        for i in range(400)
    ]
    assert np.allclose(probe, GOLDEN["mixShapes"]["probe"], rtol=0, atol=1e-7)


def test_analyze_envelope_matches_js():
    audio = concat(noise(2, amplitude=0.5, seed=5), sine(2, amplitude=0.4, frequency=80),
                   sine(2, amplitude=0.3, frequency=6000))
    result = analyze_envelope(audio, 30, 180)
    assert np.allclose(result["levels"], GOLDEN["envelope"]["levels"], rtol=0, atol=1e-6)
    for band in range(3):
        assert np.allclose(result["bands"][band], GOLDEN["envelope"]["bands"][band], rtol=0, atol=1e-6)


# ── オンセットと BPM ──────────────────────────────────────────────


def test_onset_envelope_matches_js():
    audio = click_track(120, 12)
    result = onset_envelope(audio)
    want = GOLDEN["onset"]["click120"]
    assert result["frames"] == want["frames"]
    assert abs(result["hop"] - want["hop"]) < 1e-12
    assert np.allclose(result["onset"], want["onset"], rtol=0, atol=1e-8)
    assert np.allclose(result["rms"], want["rms"], rtol=0, atol=1e-8)

    tempo = estimate_tempo(result["onset"], result["hop"])
    for key in ("bpm", "period", "firstBeat", "confidence"):
        assert close(tempo[key], want["tempo"][key], atol=1e-7), f"{key}: {tempo[key]} ≠ {want['tempo'][key]}"


@pytest.mark.parametrize("label", ["click120", "click150", "click128", "flat", "silence"])
def test_analyze_audio_matches_js(label):
    audio = {
        "click120": lambda: click_track(120, 12),
        "click150": lambda: click_track(150, 12),
        "click128": lambda: click_track(128, 8),
        "flat": lambda: click_track(120, 24),
        "silence": lambda: create_silence(4, 32000, 1),
    }[label]()
    got = analyze_audio(audio)
    want = GOLDEN["analyze"][label]
    assert got["bpm"] == want["bpm"]
    assert got["confidence"] == want["confidence"]
    assert got["firstBeat"] == want["firstBeat"]
    assert got["beatsPerBar"] == want["beatsPerBar"]
    assert got["duration"] == want["duration"]
    assert got["beats"] == want["beats"]
    assert got["bars"] == want["bars"]
    assert got["sections"] == want["sections"]


def test_detect_sections_matches_js():
    audio = three_part_track()
    envelope = onset_envelope(audio)
    sections = detect_sections(envelope["rms"], envelope["hop"],
                               {"barSeconds": 2, "duration": audio.length / audio.sample_rate})
    assert sections == GOLDEN["sections"]
    # 中身の «意味» も確かめる（静か → 派手 → 静か）
    assert sections[0]["label"] == "intro"
    assert sections[-1]["label"] == "outro"
    assert any(s["label"] == "chorus" for s in sections)


# ── 自作音源 10 本 ────────────────────────────────────────────────


@pytest.mark.skipif(not JS_AUDIO_DIR.exists(), reason="JS 版の検査用音源が見つかりません")
@pytest.mark.parametrize("name", sorted(GOLDEN["tracks"]))
def test_ground_truth_tracks_match_js(name):
    """**JS 版と同じ BPM が出るか。** 正解との差も同時に見ます。"""
    want = GOLDEN["tracks"][name]
    analysis = analyze_audio(decode_audio_file(str(JS_AUDIO_DIR / name)), {"maxBeats": 12})
    assert analysis["bpm"] == want["bpm"], f"{name}: {analysis['bpm']} ≠ {want['bpm']}（JS 版）"
    assert analysis["confidence"] == want["confidence"]
    assert analysis["firstBeat"] == want["firstBeat"]
    assert analysis["beats"] == want["beats"]
    assert analysis["bars"] == want["bars"]
    assert analysis["sections"] == want["sections"]
    # ファイル名に入っている «正解» との差
    assert abs(analysis["bpm"] - want["truth"]) <= 2, f"{name}: 正解 {want['truth']} に対し {analysis['bpm']}"


@pytest.mark.skipif(not JS_AUDIO_DIR.exists(), reason="JS 版の検査用音源が見つかりません")
def test_ground_truth_hit_rate():
    """JS 版と同じ «10 本中 8 本以上を ±2 BPM» の基準を満たすこと。"""
    rows = []
    hits = 0
    for name, want in sorted(GOLDEN["tracks"].items()):
        analysis = analyze_audio(decode_audio_file(str(JS_AUDIO_DIR / name)), {"maxBeats": 0})
        error = analysis["bpm"] - want["truth"]
        hit = abs(error) <= 2
        hits += hit
        rows.append(f"{'ok' if hit else 'NG'} {name:<24} 正解 {want['truth']:>4} → {analysis['bpm']:>7.2f}"
                    f" (誤差 {error:+.2f}, 確からしさ {analysis['confidence']:.2f})")
    assert len(rows) >= 10, f"検査用の音源が足りません: {len(rows)} 本"
    assert hits >= 8, f"{hits}/{len(rows)} 本しか当たっていません\n" + "\n".join(rows)
