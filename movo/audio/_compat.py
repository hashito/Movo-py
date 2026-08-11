"""音の «入れ物» と、まわりのモジュールへの繋ぎ。

## 音の持ち方は core に合わせます

`movo.core.wav.AudioBuffer` が Movo 全体の共通の形です。

    audio.sample_rate    標本化周波数
    audio.channels       チャンネルごとの float32 配列（**リスト**）
    audio.length         フレーム数

**チャンネルがリストなのはわざとです。** 素材ごとにチャンネル数が違う
（モノラルの効果音とステレオの BGM が混ざる）ので、2 次元配列で持つと
足すたびに詰め直しが要ります。

ただし **測る側は 2 次元配列のほうが速い**です。ラウドネスもトゥルーピークも
«全チャンネルを同じ位置で舐める» ので、`channels_2d()` で 1 度だけ
`(チャンネル数, 長さ)` に積み直してから Numba へ渡します。積み直しは
5 分ステレオで実測 **81 ms**。同じ素材のラウドネス正規化は 4 回測り直して
数秒かかるので、ここは飲める代償です。

## 掛け算は float64 で

`audio.channels` は float32 ですが、**倍率を掛けるときは一度 float64 に
上げます**。JS はすべて double で計算してから Float32Array に格納するので、
float32 のまま掛けると最下位ビットがずれます。ラウドネス正規化は
«天井ぎりぎり» の判定を繰り返すので、そこが反転すると結果が変わります。
"""

from __future__ import annotations

import math
import sys

import numpy as np

from movo.core.wav import AudioBuffer, create_silence, decode_wav, encode_wav, resample


def as_audio(obj) -> AudioBuffer:
    """`AudioBuffer` でも «JS 版と同じ形の辞書» でも受け取れるようにする。"""
    if isinstance(obj, AudioBuffer):
        return obj
    if isinstance(obj, dict):
        channels = [np.asarray(c, np.float32) for c in obj.get("channels", [])]
        length = obj.get("length")
        if length is None:
            length = len(channels[0]) if channels else 0
        return AudioBuffer(
            sample_rate=obj.get("sampleRate", obj.get("sample_rate", 48000)),
            channels=channels,
            length=int(length),
        )
    channels = getattr(obj, "channels", None)
    if channels is None:
        raise TypeError("音声バッファではありません")
    rate = getattr(obj, "sample_rate", None) or getattr(obj, "sampleRate", 48000)
    channels = [np.asarray(c, np.float32) for c in channels]
    length = getattr(obj, "length", None)
    if length is None:
        length = len(channels[0]) if channels else 0
    return AudioBuffer(sample_rate=int(rate), channels=channels, length=int(length))


def channel_count(audio) -> int:
    return len(audio.channels)


def channels_2d(audio, dtype=np.float64) -> np.ndarray:
    """`(チャンネル数, 長さ)` に積み直した **写し**を返す（Numba へ渡す用）。"""
    n = audio.length
    if not audio.channels:
        return np.zeros((0, n), dtype)
    out = np.empty((len(audio.channels), n), dtype)
    for c, channel in enumerate(audio.channels):
        usable = min(n, len(channel))
        out[c, :usable] = channel[:usable]
        if usable < n:
            out[c, usable:] = 0
    return out


def write_channels(audio, matrix: np.ndarray) -> None:
    """`channels_2d` で取り出した写しを書き戻す。"""
    for c, channel in enumerate(audio.channels):
        usable = min(len(channel), matrix.shape[1])
        channel[:usable] = matrix[c, :usable].astype(np.float32)


def scale_channels(audio, factor: float) -> None:
    """全チャンネルに倍率を掛ける（その場で書き換え）。

    **float64 で掛けてから float32 に戻します。** 倍率を先に float32 へ
    落とすと JS 版と 1e-7 ずれ、リミッターの «天井ぎりぎり» の判定が
    反転することがあります。
    """
    value = float(factor)
    for channel in audio.channels:
        channel[:] = (channel.astype(np.float64) * value).astype(np.float32)


def peak_absolute(audio) -> float:
    """全チャンネルを通した最大絶対値。"""
    peak = 0.0
    for channel in audio.channels:
        if channel.size:
            peak = max(peak, float(np.abs(channel).max()))
    return peak


# ── その他の小物 ─────────────────────────────────────────────────


def clamp(value, low, high):
    """`low` と `high` の間に丸める。**配列でも通ります**（core のは走査型）。"""
    if isinstance(value, np.ndarray):
        return np.clip(value, low, high)
    return low if value < low else (high if value > high else value)


def js_round(value) -> int:
    """JS の `Math.round`（0.5 は上へ）。Python の `round` は偶数へ丸めます。"""
    return math.floor(float(value) + 0.5)


def warn(message: str) -> None:
    try:
        from movo.core.logger import logger

        logger.warn(message)
    except Exception:  # pragma: no cover
        print(f"⚠ {message}", file=sys.stderr)


def verbose(message: str) -> None:
    try:
        from movo.core.logger import logger

        logger.verbose(message)
    except Exception:  # pragma: no cover
        pass


def find_ffmpeg():
    """ffmpeg を探す。"""
    try:
        from movo.core.platform import find_ffmpeg as impl

        return impl()
    except Exception:  # pragma: no cover
        import os
        import shutil

        override = os.environ.get("MOVO_FFMPEG")
        if override and os.path.exists(override):
            return {"path": override, "source": "MOVO_FFMPEG"}
        found = shutil.which("ffmpeg")
        return {"path": found, "source": "PATH"} if found else None


def resolve_animated(spec, ctx=None, default=None):
    """«動く値» を今の時刻で 1 つの数に潰す（animation が入るまでの後退実装）。"""
    try:
        from movo.animation.resolver import resolve_animated as impl  # type: ignore

        return impl(spec, ctx, default)
    except Exception:
        if spec is None:
            return default
        if isinstance(spec, (int, float)):
            return spec
        return default


def is_animated(spec) -> bool:
    """時刻で変わる指定か。定数なら «全サンプル同じ» の速い道を通れます。"""
    return isinstance(spec, dict)


__all__ = [
    "AudioBuffer", "as_audio", "create_silence", "decode_wav", "encode_wav", "resample",
    "channel_count", "channels_2d", "write_channels", "scale_channels", "peak_absolute",
    "clamp", "js_round", "warn", "verbose", "find_ffmpeg", "resolve_animated", "is_animated",
]
