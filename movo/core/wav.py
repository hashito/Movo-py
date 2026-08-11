"""RIFF/WAVE の読み書き。**自前実装です**（依存を足しません）。

Movo の中では音は «プロジェクトのサンプリング周波数に揃えた float32 の
チャンネル別配列» で持ちます。ディスク上のインターリーブされた PCM との
往復をここが受け持ちます。

## 速度の作法

JS 版は 1 サンプルずつ ``readInt16LE`` を呼んでいます。3 分の 48 kHz ステレオで
1,700 万回の呼び出しになり、Python でそれをやると **11 秒**かかります。
ここは **``np.frombuffer`` で型ごと読み替えて一括で割る**だけにしました
（同じデータで **31 ミリ秒**、355 倍）。24 ビットだけは Python にも NumPy にも
型が無いので、3 バイトを 4 バイトに詰め替えてから読みます。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .errors import ErrorCodes, MovoError


@dataclass
class AudioBuffer:
    """音のかたまり。

    :param sample_rate: 標本化周波数
    :param channels: チャンネルごとの float32 配列（-1..1）
    :param length: フレーム数（チャンネルあたりの標本数）
    """

    sample_rate: int
    channels: list[np.ndarray] = field(default_factory=list)
    length: int = 0

    @property
    def duration(self) -> float:
        return self.length / self.sample_rate if self.sample_rate else 0.0


def decode_wav(buffer: bytes) -> AudioBuffer:
    """WAVE を読む。8/16/24/32 ビット整数と 32 ビット float に対応します。"""
    data = bytes(buffer)
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "RIFF/WAVE のファイルではありません")

    offset = 12
    fmt = None
    chunk: bytes | None = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = {
                "format": int.from_bytes(body[0:2], "little"),
                "channels": int.from_bytes(body[2:4], "little"),
                "sampleRate": int.from_bytes(body[4:8], "little"),
                "bitsPerSample": int.from_bytes(body[14:16], "little"),
            }
            # WAVE_FORMAT_EXTENSIBLE。本当の形式は拡張部の先頭に入っています。
            if fmt["format"] == 0xFFFE and size >= 26:
                fmt["format"] = int.from_bytes(body[24:26], "little")
        elif chunk_id == b"data":
            chunk = body
        offset += 8 + size + (size % 2)  # チャンクは偶数バイト境界に揃う

    if fmt is None or chunk is None:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "WAVE に fmt か data のチャンクがありません")

    num_channels = max(1, fmt["channels"])
    bits = fmt["bitsPerSample"]
    bytes_per_sample = bits // 8
    if bytes_per_sample <= 0:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"対応していない WAVE のビット深度 {bits}")
    frames = len(chunk) // (bytes_per_sample * num_channels)
    usable = frames * bytes_per_sample * num_channels
    body = chunk[:usable]

    if fmt["format"] == 3 and bits == 32:
        flat = np.frombuffer(body, "<f4").astype(np.float32)
    elif bits == 8:
        flat = (np.frombuffer(body, np.uint8).astype(np.float32) - 128) / 128
    elif bits == 16:
        flat = np.frombuffer(body, "<i2").astype(np.float32) / 32768
    elif bits == 24:
        # 24 ビットには型が無いので、3 バイトを 4 バイトに詰め替えて int32 として読みます。
        # 最下位に 0 を足して «上位 24 ビット» に置き、あとで 256 で割ります。
        packed = np.frombuffer(body, np.uint8).reshape(-1, 3)
        wide = np.zeros((packed.shape[0], 4), np.uint8)
        wide[:, 1:] = packed
        flat = wide.view("<i4").reshape(-1).astype(np.float32) / (256.0 * 8388608.0)
    elif bits == 32:
        flat = np.frombuffer(body, "<i4").astype(np.float32) / 2147483648
    else:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"対応していない WAVE のビット深度 {bits}")

    planar = flat.reshape(frames, num_channels)
    channels = [np.ascontiguousarray(planar[:, c]) for c in range(num_channels)]
    return AudioBuffer(sample_rate=fmt["sampleRate"], channels=channels, length=frames)


def encode_wav(audio: AudioBuffer, bits_per_sample: int = 16, float32: bool = False) -> bytes:
    """WAVE のバイト列にする。既定は 16 ビット整数。"""
    bits = 32 if float32 else bits_per_sample
    fmt_code = 3 if float32 else 1
    num_channels = max(1, len(audio.channels))
    frames = audio.length
    bytes_per_sample = bits // 8

    if audio.channels:
        stacked = np.empty((frames, num_channels), np.float32)
        for c in range(num_channels):
            src = audio.channels[c]
            n = min(frames, len(src))
            stacked[:n, c] = src[:n]
            if n < frames:
                stacked[n:, c] = 0
    else:
        stacked = np.zeros((frames, num_channels), np.float32)
    clipped = np.clip(stacked, -1.0, 1.0)
    # **掛ける前に float64 へ上げます。** float32 のまま 8388607 を掛けると
    # 有効桁が足りず、24 ビットの最下位が JS 版と 1 ずれる標本が出ます
    # （JS の数値はすべて double なので、掛け算は double で行われます）。
    wide = clipped.astype(np.float64)

    if fmt_code == 3:
        payload = clipped.astype("<f4").tobytes()
    elif bits == 16:
        payload = _js_round_array(wide * 32767).astype("<i2").tobytes()
    elif bits == 24:
        values = _js_round_array(wide * 8388607).astype("<i4")
        payload = values.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        payload = _js_round_array(wide * 2147483647).astype("<i4").tobytes()

    data_size = len(payload)
    header = bytearray()
    header += b"RIFF"
    header += (36 + data_size).to_bytes(4, "little")
    header += b"WAVE"
    header += b"fmt "
    header += (16).to_bytes(4, "little")
    header += fmt_code.to_bytes(2, "little")
    header += num_channels.to_bytes(2, "little")
    header += int(audio.sample_rate).to_bytes(4, "little")
    header += (audio.sample_rate * num_channels * bytes_per_sample).to_bytes(4, "little")
    header += (num_channels * bytes_per_sample).to_bytes(2, "little")
    header += bits.to_bytes(2, "little")
    header += b"data"
    header += data_size.to_bytes(4, "little")
    return bytes(header) + payload


def _js_round_array(values: np.ndarray) -> np.ndarray:
    """JS の ``Math.round`` と同じ丸め（常に +∞ 方向）を配列にかける。

    NumPy の ``rint`` は «偶数丸め» なので、0.5 のところで JS と 1 ずれます。
    量子化の境目にちょうど乗る合成音（テスト用の矩形波など）で差が出ます。
    """
    return np.floor(values.astype(np.float64) + 0.5)


def create_silence(seconds: float, sample_rate: int = 48000, channel_count: int = 2) -> AudioBuffer:
    """``seconds`` 秒の無音を作る。"""
    length = max(0, int(np.ceil(seconds * sample_rate)))
    return AudioBuffer(
        sample_rate=sample_rate,
        channels=[np.zeros(length, np.float32) for _ in range(channel_count)],
        length=length,
    )


def resample(audio: AudioBuffer, target_rate: int) -> AudioBuffer:
    """線形補間で標本化周波数を変える。

    高級な変換（sinc 補間）にしていないのは、**素材はほぼ 44.1k か 48k で、
    その間の変換なら線形でも聴いて分からない**からです。JS 版と同じ式です。
    """
    if audio.sample_rate == target_rate:
        return audio
    ratio = target_rate / audio.sample_rate
    length = max(1, int(np.floor(audio.length * ratio + 0.5)))
    pos = np.arange(length, dtype=np.float64) / ratio
    i0 = np.floor(pos).astype(np.int64)
    frac = (pos - i0).astype(np.float32)

    channels = []
    for src in audio.channels:
        n = len(src)
        if n == 0:
            channels.append(np.zeros(length, np.float32))
            continue
        a = np.where(i0 < n, src[np.clip(i0, 0, n - 1)], 0).astype(np.float32)
        i1 = np.minimum(n - 1, i0 + 1)
        b = src[i1].astype(np.float32)
        channels.append((a * (1 - frac) + b * frac).astype(np.float32))
    return AudioBuffer(sample_rate=target_rate, channels=channels, length=length)
