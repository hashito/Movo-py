"""オートダッキング — «ナレーションが鳴っている間だけ BGM を下げる»。

MV に語りやセリフを乗せると、BGM に埋もれて何を言っているか分からなく
なります。手で音量のキーフレームを打てば直りますが、歌詞やセリフを 1 行
足すたびに打ち直しになります。「この素材が鳴っている間、あの素材を
12dB 下げて」と 1 行書けば済むようにしたのがここです。

```jsonc
{ "asset": "narration",
  "ducks": [{ "target": "track", "amount": -12, "attack": 0.08, "release": 0.4, "threshold": -30 }] }
```

中身は «サイドチェーン付きのゲート» です。

  1. 下げる «きっかけ» になる音（ナレーション）の包絡を取る
  2. 閾値を超えている間は `amount` dB、それ以外は 0dB を目標にする
  3. 目標へ向かって attack / release の速さで滑らかに動かす

平滑化を **dB のまま**やっているのが要点です。線形の倍率で滑らかにすると、
下がり始めだけ急に落ちて戻りがだらだら続く、耳に付く動きになります。

## なぜ Numba か

包絡もゲインカーブも **1 サンプル前の状態が要る逐次処理**です。しかも
「上がるときは待たない」「下げるときと戻すときで係数が違う」という分岐が
入るので、NumPy の一括演算には畳めません。

| 30 秒・48kHz ステレオ（包絡＋ゲインカーブ） | |
| --- | --- |
| 純 Python | 472 ms |
| **Numba** | **30 ms**（16 倍） |

決定性は保たれます。入力サンプルだけから決まる純粋な計算で、乱数も時刻も
使っていません。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._compat import as_audio, channels_2d

# 包絡検出の時定数（秒）。これより速いと語尾の子音のたびにゲインが揺れます。
DETECTOR_SECONDS = 0.01

# 既定値。無指定でも «喋っている間だけ 12dB 下がる» が成り立つ値です。
DEFAULTS = {"amount": -12, "attack": 0.08, "release": 0.4, "threshold": -30, "hold": 0.1}


def _number(value, fallback):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else fallback


def resolve_duck_spec(spec):
    """ダッキング指定を既定値込みの形に整える。

    :returns: ``{"target", "amount", "attack", "release", "threshold", "hold"}``。
        `target` が無ければ `None`
    """
    if not isinstance(spec, dict):
        return None
    target = spec.get("target")
    if not isinstance(target, str):
        return None
    amount = _number(spec.get("amount"), DEFAULTS["amount"])
    return {
        "target": target,
        # 正の値で書かれても «下げる» と解釈します。−12 と 12 で逆に動くほうが事故です。
        "amount": -abs(amount),
        "attack": max(0.0, _number(spec.get("attack"), DEFAULTS["attack"])),
        "release": max(0.0, _number(spec.get("release"), DEFAULTS["release"])),
        "threshold": _number(spec.get("threshold"), DEFAULTS["threshold"]),
        "hold": max(0.0, _number(spec.get("hold"), DEFAULTS["hold"])),
    }


@njit(cache=True)
def _detector(channels, coefficient):
    """二乗値に 1 極フィルタを掛けてから平方根を取る。

    振幅そのものを平滑化すると、波形が 0 を横切るたびに谷ができて閾値を
    またぎ続けます。**立ち上がりは «待たない»** のも要点で、喋り出しの頭が
    BGM に潰されるのがいちばん困ります。
    """
    count = channels.shape[0]
    n = channels.shape[1]
    envelope = np.empty(n, np.float32)
    state = 0.0
    for i in range(n):
        total = 0.0
        for c in range(count):
            value = channels[c, i]
            total += value * value
        power = total / count
        state = power if power > state else power + (state - power) * coefficient
        envelope[i] = np.float32(math.sqrt(state))
    return envelope


def detector_envelope(audio, sample_rate: int | None = None) -> np.ndarray:
    """サイドチェーン用の包絡（RMS 相当）を作る。"""
    audio = as_audio(audio)
    rate = sample_rate or audio.sample_rate
    coefficient = math.exp(-1 / max(1, DETECTOR_SECONDS * rate))
    if not audio.channels:
        return np.zeros(audio.length, np.float32)
    return _detector(channels_2d(audio), coefficient)


@njit(cache=True)
def _gain_curve(envelope, threshold_linear, hold_samples, amount, attack, release):
    n = envelope.shape[0]
    gains = np.empty(n, np.float32)
    current = 0.0  # いまのゲイン（dB）
    hold_left = 0
    for i in range(n):
        if envelope[i] >= threshold_linear:
            hold_left = hold_samples
        elif hold_left > 0:
            hold_left -= 1
        wanted = amount if (envelope[i] >= threshold_linear or hold_left > 0) else 0.0
        # 下げるときは attack、戻すときは release。ここを取り違えると
        # «喋り終わってから下がる» という間抜けな動きになります。
        coefficient = attack if wanted < current else release
        current = wanted + (current - wanted) * coefficient
        gains[i] = np.float32(10.0 ** (current / 20.0))
    return gains


def duck_gain_curve(envelope, sample_rate: int, spec: dict) -> np.ndarray:
    """包絡からゲインカーブ（線形倍率）を作る。"""
    threshold_linear = 10 ** (spec["threshold"] / 20)
    hold_samples = round(spec["hold"] * sample_rate)
    # 時定数 0（＝瞬時）も書けます。効果音を «パッと» 避ける用途があります。
    attack = math.exp(-1 / (spec["attack"] * sample_rate)) if spec["attack"] > 0 else 0.0
    release = math.exp(-1 / (spec["release"] * sample_rate)) if spec["release"] > 0 else 0.0
    return _gain_curve(np.asarray(envelope, np.float32), threshold_linear, int(hold_samples),
                       spec["amount"], attack, release)


def build_duck_curve(sidechain, spec):
    """トラック 1 本ぶんのゲインカーブを作る（包絡 → カーブ）。"""
    settings = resolve_duck_spec(spec)
    if settings is None:
        return None
    sidechain = as_audio(sidechain)
    envelope = detector_envelope(sidechain, sidechain.sample_rate)
    return duck_gain_curve(envelope, sidechain.sample_rate, settings)


def combine_duck_curves(curves):
    """複数のゲインカーブを重ねる。**いちばん深く下げているものを採ります。**

    掛け算にしていないのは、ナレーションとセリフが同時に鳴ったときに
    −12dB が 2 本で −24dB になり、BGM が消えてしまうからです。
    """
    usable = [c for c in curves if c is not None]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]
    out = usable[0]
    for other in usable[1:]:
        np.minimum(out, other, out=out)
    return out


def mix_ducked(source, destination, curve) -> None:
    """ゲインカーブを掛けながら、バッファを出力へ足し込む。"""
    source = as_audio(source)
    destination = as_audio(destination)
    length = min(source.length, destination.length)
    if length <= 0:
        return
    for c in range(len(destination.channels)):
        frm = source.channels[c] if c < len(source.channels) else (
            source.channels[0] if source.channels else None
        )
        if frm is None:
            continue
        if curve is not None:
            gain = np.clip(np.asarray(curve[:length], np.float32), 0.0, 1.0)
            destination.channels[c][:length] += frm[:length] * gain
        else:
            destination.channels[c][:length] += frm[:length]
