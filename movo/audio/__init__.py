"""movo-audio — ミックスと解析（JS 版 packages/audio/src/index.js の移植）。

トラックは素材ストアが復号し、ここでゲイン・パン・ループ・トリム・
フェードを掛けながら 1 本の float バッファへ足し込みます。同じバッファが
`audio-reactive` モジュレーターの元にもなります（`analyze_envelope`）。

仕上げとして 2 つ:

  - **オートダッキング**（`audio[].ducks`）… ナレーションが鳴っている間だけ
    BGM を下げる。実装は `duck.py`
  - **ラウドネス正規化**（`output.loudness`）… EBU R128 / ITU-R BS.1770 の
    LUFS で測って目標へ合わせる。実装は `loudness.py`

## ラウドネス正規化は «書いたときだけ»

**`output.loudness` を書いたときだけ**動きます。既定を「入り」にすると、
既存の JSON から出てくる音が黙って変わってしまうためです。しかも
`audio-reactive` は音量の包絡を見て映像を動かすので、**音だけでなく
映像まで変わります**。書いていないプロジェクトは今までどおり
«割れないためのピーク正規化» だけです。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._compat import (
    AudioBuffer, as_audio, channel_count, clamp, create_silence, decode_wav, encode_wav,
    is_animated, js_round, peak_absolute, resample, resolve_animated, scale_channels,
    verbose, warn,
)
from .align import (
    align_lyrics, count_morae, label_blocks, parse_anchors, singing_windows,
    split_blocks, to_lrc, to_scenario, vocal_presence,
)
from .analyze import (
    AudioDecodeError, analyze_audio, decode_audio_file, detect_sections,
    estimate_tempo, onset_envelope, resolve_bpm_from_audio,
)
from .duck import (
    build_duck_curve, combine_duck_curves, detector_envelope, duck_gain_curve,
    mix_ducked, resolve_duck_spec,
)
from .loudness import (
    LOUDNESS_STANDARDS, k_weighting_stages, limit_true_peak, measure_loudness,
    measure_true_peak, normalize_loudness, resolve_loudness_spec, true_peak_linear,
)

DEFAULT_SAMPLE_RATE = 48000


@njit(cache=True)
def _split_bands(mono, low_coeff, high_coeff):
    """1 極フィルタで低域と高域を作る。

    1 サンプル前の出力が要る逐次処理なので、ここだけ Numba です
    （30 秒で純 Python 245ms → **4.7ms**、52 倍）。
    """
    n = mono.shape[0]
    low = np.empty(n, np.float64)
    high = np.empty(n, np.float64)
    low_state = 0.0
    high_state = 0.0
    for i in range(n):
        low_state = mono[i] * (1.0 - low_coeff) + low_state * low_coeff
        low[i] = low_state
        high_state = mono[i] * (1.0 - high_coeff) + high_state * high_coeff
        high[i] = mono[i] - high_state
    return low, high


def mix_project_audio(project: dict, assets, options: dict) -> dict | None:
    """プロジェクトの音を 1 本にまとめる。

    :param project: 正規化済みのプロジェクト
    :param assets: `get_audio(name)`（または `getAudio`）を持つもの
    :param options: ``duration`` / ``sampleRate`` / ``fps`` / ``seed``
    :returns: ``{"audio", "tracks", "loudness", "ducked"}``。音が 1 本も無ければ `None`
    """
    tracks = [t for t in (project.get("audio") or []) if t.get("enabled") is not False]
    layer_tracks = _collect_audio_layers(project)
    everything = tracks + layer_tracks
    if not everything:
        return None

    sample_rate = options.get("sampleRate") or DEFAULT_SAMPLE_RATE
    output = create_silence(options["duration"], sample_rate, 2)

    # ダッキングに関わるトラックだけ «自分専用のバッファ» に書きます。全部を
    # 個別に持つと 5 分の曲 1 本で 100MB 単位のメモリを食うので、関係の無い
    # トラックは今までどおり直接 output へ足します。
    plans = _collect_duck_plans(everything)
    isolated_keys = set()
    for plan in plans:
        isolated_keys.add(plan["source"])
        isolated_keys.add(plan["target"])
    isolated: dict[str, AudioBuffer] = {}

    mixed = 0
    for track in everything:
        key = _track_key(track)
        destination = output
        if key in isolated_keys:
            if key not in isolated:
                isolated[key] = create_silence(options["duration"], sample_rate, 2)
            destination = isolated[key]
        if _render_track(track, assets, destination, {**options, "sampleRate": sample_rate}):
            mixed += 1

    if mixed == 0:
        return None

    ducked = _apply_ducking(plans, isolated, output, sample_rate)

    loudness_spec = (project.get("output") or {}).get("loudness")
    loudness = None
    if resolve_loudness_spec(loudness_spec):
        loudness = normalize_loudness(output, loudness_spec)
    else:
        # 従来どおりのピーク正規化。«割れない» ことしか保証しませんが、
        # output.loudness を書いていない既存の JSON の音は変えられません。
        peak = peak_absolute(output)
        if peak > 1:
            scale = 1 / peak
            verbose(f"audio peak was {peak:.2f}; normalising by {scale:.3f}")
            scale_channels(output, scale)

    return {"audio": output, "tracks": mixed, "loudness": loudness, "ducked": ducked}


def _track_key(track: dict) -> str:
    """ダッキングの `target` が指す名前。id を付けていればそれ、無ければ素材名。"""
    return track.get("id") or track.get("asset") or track.get("path") or ""


def _collect_duck_plans(tracks: list[dict]) -> list[dict]:
    """`ducks` を «どのトラックがどのトラックを下げるか» の一覧に開く。"""
    keys = {_track_key(t) for t in tracks}
    plans = []
    for track in tracks:
        for spec in track.get("ducks") or []:
            settings = resolve_duck_spec(spec)
            if settings is None:
                warn(f"audio ducks: target が無い指定は無視します（{_track_key(track)}）")
                continue
            if settings["target"] not in keys:
                warn(f'audio ducks: target "{settings["target"]}" というトラックがありません')
                continue
            plans.append({"source": _track_key(track), "target": settings["target"], "spec": settings})
    return plans


def _apply_ducking(plans, isolated, output, sample_rate: int) -> int:
    """個別バッファへ書いたトラックを、ゲインカーブを掛けながら output へ足す。

    カーブは **掛ける前に全部作る** のが要点です。ナレーション同士が互いを
    下げ合う指定でも、サイドチェーンには常に «下げる前の» 音が使われます。

    :returns: 実際に下げたトラックの本数
    """
    if not isolated:
        return 0
    envelopes: dict[str, np.ndarray] = {}
    curves: dict[str, list] = {}
    for plan in plans:
        sidechain = isolated.get(plan["source"])
        target = isolated.get(plan["target"])
        if sidechain is None or target is None:
            continue
        if plan["source"] not in envelopes:
            envelopes[plan["source"]] = detector_envelope(sidechain, sample_rate)
        curve = duck_gain_curve(envelopes[plan["source"]], sample_rate, plan["spec"])
        curves.setdefault(plan["target"], []).append(curve)
    ducked = 0
    for key, buffer in isolated.items():
        curve = combine_duck_curves(curves.get(key, []))
        if curve is not None:
            ducked += 1
        mix_ducked(buffer, output, curve)
    return ducked


def _render_track(track: dict, assets, output, options: dict) -> bool:
    """トラック 1 本を出力バッファへ足す（ゲイン・パン・ループ・トリム・フェード）。

    ## 速い道と遅い道

    JS 版は **1 サンプルごとに** `resolveAnimated` を呼んでいました。Python で
    同じことをすると 48kHz × 5 分で 1,440 万回の関数呼び出しになり、それだけで
    30 秒かかります。

    音量とパンが «ただの数» のとき（ほとんどの JSON がそうです）は
    **NumPy の一括演算**にしました。キーフレームや式が書かれているときだけ、
    JS 版と同じくサンプルごとに解決します（結果は同じで、速度だけ違います）。
    """
    sample_rate = options["sampleRate"]
    name = track.get("asset") or track.get("path")
    source = None
    if track.get("asset") and assets is not None:
        getter = getattr(assets, "get_audio", None) or getattr(assets, "getAudio", None)
        if getter is None and isinstance(assets, dict):
            getter = assets.get("getAudio") or assets.get("get_audio")
        if getter is not None:
            source = getter(track["asset"])
    if source is None and track.get("path"):
        warn(f'audio track "{name}" uses "path"; declare it in project.assets so it can be decoded')
        return False
    if source is None:
        warn(f'audio track "{name}" could not be resolved and was skipped')
        return False
    source = resample(as_audio(source), sample_rate)
    output = as_audio(output)

    start = max(0.0, float(track.get("start") or 0))
    offset = max(0.0, float(track.get("offset") or 0))
    source_duration = source.length / sample_rate - offset
    track_duration = track.get("duration")
    track_duration = source_duration if track_duration is None else float(track_duration)
    duration = max(0.0, min(track_duration, options["duration"] - start))
    if duration <= 0:
        return False

    start_sample = js_round(start * sample_rate)
    offset_sample = js_round(offset * sample_rate)
    total_samples = js_round(duration * sample_rate)
    fade_in_samples = js_round(float(track.get("fadeIn") or 0) * sample_rate)
    fade_out_samples = js_round(float(track.get("fadeOut") or 0) * sample_rate)
    usable_source = max(1, source.length - offset_sample)

    # 出力に収まる範囲だけを扱う（JS 版の `if (destIndex >= output.length) break`）
    total_samples = min(total_samples, max(0, output.length - start_sample))
    if total_samples <= 0:
        return True

    i = np.arange(total_samples)
    source_index = offset_sample + i
    if track.get("loop"):
        wrapped = np.where(source_index >= source.length, offset_sample + (i % usable_source), source_index)
        source_index = wrapped
        valid = np.ones(total_samples, bool)
    else:
        valid = source_index < source.length
        if not valid.all():
            # ループしないので、素材が尽きたところで打ち切ります。
            total_samples = int(valid.argmin())
            if total_samples <= 0:
                return True
            i = i[:total_samples]
            source_index = source_index[:total_samples]
            valid = valid[:total_samples]
    source_index = np.clip(source_index, 0, max(0, source.length - 1))

    volume_spec = track.get("volume", 1)
    pan_spec = track.get("pan", 0)
    if is_animated(volume_spec) or is_animated(pan_spec):
        # 遅い道：JS 版と同じくサンプルごとに解決します。
        times = (start_sample + i) / sample_rate
        volume = np.empty(total_samples, np.float64)
        pan = np.empty(total_samples, np.float64)
        for k in range(total_samples):
            time = float(times[k])
            ctx = {"time": time, "scope": {"time": time, "index": int(i[k])}, "seed": options.get("seed", 0)}
            volume[k] = clamp(float(resolve_animated(volume_spec, ctx, 1) or 1), 0, 4)
            pan[k] = clamp(float(resolve_animated(pan_spec, ctx, 0) or 0), -1, 1)
    else:
        volume = np.full(total_samples, clamp(float(volume_spec if volume_spec is not None else 1), 0, 4))
        pan = np.full(total_samples, clamp(float(pan_spec if pan_spec is not None else 0), -1, 1))

    gain = volume.copy()
    if fade_in_samples > 0:
        rising = i < fade_in_samples
        gain[rising] *= i[rising] / fade_in_samples
    if fade_out_samples > 0:
        # JS は «残り時間 ÷ fadeOut» を掛けます。総サンプル数が出力で切られても
        # 基準は «トラックの» 総サンプル数のままです。
        falling = i > total_samples - fade_out_samples
        gain[falling] *= np.maximum(0.0, (total_samples - i[falling]) / fade_out_samples)

    left_gain = gain * np.where(pan <= 0, 1.0, 1 - pan)
    right_gain = gain * np.where(pan >= 0, 1.0, 1 + pan)

    left = source.channels[0][source_index].astype(np.float64)
    right = (source.channels[1][source_index].astype(np.float64)
             if channel_count(source) > 1 else left)
    left = np.where(valid, left, 0.0)
    right = np.where(valid, right, 0.0)

    end = start_sample + total_samples
    output.channels[0][start_sample:end] += (left * left_gain).astype(np.float32)
    output.channels[1][start_sample:end] += (right * right_gain).astype(np.float32)
    return True


def _collect_audio_layers(project: dict) -> list[dict]:
    """音声レイヤーもトラックとして扱う。時間の指定を映像の隣に書けるように。"""
    out = []

    def walk(layers, scene_start):
        for layer in layers or []:
            if layer.get("type") == "audio" and layer.get("enabled") is not False:
                out.append({
                    "id": layer.get("id"),
                    "asset": layer.get("asset"),
                    "ducks": layer.get("ducks"),
                    "start": (layer.get("start") or 0) + scene_start,
                    "offset": layer.get("offset") or 0,
                    "duration": layer.get("duration"),
                    "volume": layer.get("volume", 1),
                    "pan": layer.get("pan", 0),
                    "loop": layer.get("loop"),
                    "fadeIn": layer.get("fadeIn"),
                    "fadeOut": layer.get("fadeOut"),
                })
            if layer.get("layers"):
                walk(layer["layers"], scene_start)

    cursor = 0
    for scene in project.get("scenes") or []:
        start = scene.get("start")
        start = cursor if start is None else start
        walk(scene.get("layers"), start)
        cursor = start + (scene.get("duration") or 0)
    return out


def analyze_envelope(audio, fps: float, frame_count: int | None = None) -> dict:
    """フレームごとの音量包絡と 3 帯域。`audio-reactive` モジュレーターの元です。

    JS 版はサンプルごとの二重ループでした。ここは **NumPy 向きの形**
    （1 次元の一括演算＋枠ごとの合計）なので全部畳んであります。1 極フィルタ
    だけは逐次なので Numba に回します。

    :returns: ``{"levels": (frames,), "bands": (3, frames)}``
    """
    audio = as_audio(audio)
    frames = frame_count if frame_count else math.ceil((audio.length / audio.sample_rate) * fps)
    frames = max(1, int(frames))
    levels = np.zeros(frames, np.float32)
    bands = np.zeros((3, frames), np.float32)
    samples_per_frame = max(1, int(audio.sample_rate // fps))
    if audio.length == 0:
        return {"levels": levels, "bands": bands}

    count = channel_count(audio) or 1
    mono = np.zeros(audio.length, np.float64)
    for channel in audio.channels:
        take = min(audio.length, len(channel))
        mono[:take] += channel[:take].astype(np.float64)
    mono /= count

    low_coeff = math.exp((-2 * math.pi * 200) / audio.sample_rate)
    high_coeff = math.exp((-2 * math.pi * 3000) / audio.sample_rate)
    low, high = _split_bands(mono, low_coeff, high_coeff)

    # 枠ごとの二乗和は累積和で 1 パス。
    mono_sq = np.concatenate([[0.0], np.cumsum(mono * mono)])
    low_sq = np.concatenate([[0.0], np.cumsum(low * low)])
    high_sq = np.concatenate([[0.0], np.cumsum(high * high)])
    starts = np.minimum(np.arange(frames) * samples_per_frame, audio.length)
    ends = np.minimum(starts + samples_per_frame, audio.length)
    counts = (ends - starts).astype(np.float64)
    lit = counts > 0
    safe = np.where(lit, counts, 1.0)

    rms = np.sqrt((mono_sq[ends] - mono_sq[starts]) / safe)
    low_rms = np.sqrt((low_sq[ends] - low_sq[starts]) / safe)
    high_rms = np.sqrt((high_sq[ends] - high_sq[starts]) / safe)

    levels_f = np.where(lit, np.clip(rms * 2.5, 0, 1), 0.0)
    band0 = np.where(lit, np.clip(low_rms * 3, 0, 1), 0.0)
    band2 = np.where(lit, np.clip(high_rms * 4, 0, 1), 0.0)
    band1 = np.where(lit, np.clip(np.maximum(0.0, levels_f - band0 * 0.5 - band2 * 0.3), 0, 1), 0.0)

    levels[:] = levels_f
    bands[0] = band0
    bands[1] = band1
    bands[2] = band2
    return {"levels": levels, "bands": bands}


__all__ = [
    "AudioBuffer", "mix_project_audio", "analyze_envelope",
    "encode_wav", "decode_wav", "create_silence", "resample",
    "analyze_audio", "estimate_tempo", "onset_envelope", "detect_sections",
    "decode_audio_file", "resolve_bpm_from_audio", "AudioDecodeError",
    "measure_loudness", "measure_true_peak", "true_peak_linear", "limit_true_peak",
    "normalize_loudness", "resolve_loudness_spec", "k_weighting_stages", "LOUDNESS_STANDARDS",
    "resolve_duck_spec", "detector_envelope", "duck_gain_curve", "build_duck_curve",
    "combine_duck_curves", "mix_ducked",
    "align_lyrics", "count_morae", "label_blocks", "parse_anchors", "singing_windows",
    "split_blocks", "to_lrc", "to_scenario", "vocal_presence",
]
