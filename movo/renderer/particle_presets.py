"""パーティクルのプリセット。

MV 制作ブログでよく紹介されている «舞い物» の定番設定です。個数や速度の目安は
SIGA BLOG の雨（個数 400〜2000・拡散 20〜60・方向 ±100）や、kotsu x kotsu /
わたしの教科書 の花びら・雪の解説を参考にしています。

    { "type": "particle", "emitter": { "preset": "sakura", "rate": 40 } }

プリセットの値は個別指定で上書きできます。**画面サイズを受け取る形**にして
あるのは、画面幅いっぱいから降らせるといった調整が要るからです。

寸法と速度は 1080p 基準で書いてあります。**呼ぶ側（`create_particle_system`）で
解像度に合わせて掛け直します。** 利用者が emitter に直接書いた値はそのプロジェクトの
座標系なので、そちらは触りません。
"""

from __future__ import annotations

from typing import Callable


def _rain(width: float, height: float) -> dict:
    """雨。細長い粒を斜めに速く落とす。"""
    return {
        "rate": 260,
        "lifetime": 1.1,
        "lifetimeVariance": 0.2,
        "speed": 1500,
        "speedVariance": 0.25,
        "direction": 100,
        "spread": 6,
        "size": 3,
        "sizeVariance": 0.4,
        "sizeOverLife": 1,
        "color": "#9fc7ff",
        "endColor": "#5f86c4",
        "gravityScale": 0.6,
        "drag": 0,
        "fadeIn": 0.02,
        "fadeOut": 0.15,
        "maxParticles": 900,
        "width": width,
        "height": 0,
        "y": -height * 0.1,
    }


def _snow(width: float, height: float) -> dict:
    """雪。ゆっくり揺れながら落ちる。"""
    return {
        "rate": 90,
        "lifetime": 6,
        "lifetimeVariance": 0.35,
        "speed": 70,
        "speedVariance": 0.6,
        "direction": 96,
        "spread": 40,
        "size": 9,
        "sizeVariance": 0.6,
        "sizeOverLife": 0.9,
        "color": "#ffffff",
        "gravityScale": 0.05,
        "drag": 1.4,
        "spin": 40,
        "fadeIn": 0.08,
        "fadeOut": 0.25,
        "maxParticles": 700,
        "width": width,
        "height": 0,
        "y": -height * 0.08,
    }


def _sakura(width: float, height: float) -> dict:
    """桜吹雪。横に流されながらひらひら落ちる。"""
    return {
        "rate": 55,
        "lifetime": 5.5,
        "lifetimeVariance": 0.3,
        "speed": 130,
        "speedVariance": 0.5,
        "direction": 70,
        "spread": 55,
        "size": 16,
        "sizeVariance": 0.5,
        "sizeOverLife": 0.95,
        "color": "#ffc0d4",
        "endColor": "#ff8fb3",
        "gravityScale": 0.1,
        "drag": 1.1,
        "spin": 180,
        "fadeIn": 0.06,
        "fadeOut": 0.3,
        "maxParticles": 500,
        "width": width,
        "height": 0,
        "y": -height * 0.06,
    }


def _confetti(width: float, height: float) -> dict:
    """紙吹雪。上から勢いよく、回転しながら。"""
    return {
        "rate": 120,
        "lifetime": 3.2,
        "lifetimeVariance": 0.35,
        "speed": 320,
        "speedVariance": 0.7,
        "direction": 80,
        "spread": 70,
        "size": 14,
        "sizeVariance": 0.5,
        "sizeOverLife": 1,
        "color": "#ffd166",
        "endColor": "#f72585",
        "gravityScale": 0.7,
        "drag": 0.9,
        "spin": 520,
        "fadeIn": 0.03,
        "fadeOut": 0.2,
        "maxParticles": 700,
        "width": width,
        "height": 0,
        "y": -height * 0.05,
    }


def _bubble(width: float, height: float) -> dict:
    """泡。下から上へゆっくり。水中表現と相性が良い。"""
    return {
        "rate": 40,
        "lifetime": 4.5,
        "lifetimeVariance": 0.4,
        "speed": 90,
        "speedVariance": 0.6,
        "direction": -90,
        "spread": 25,
        "size": 18,
        "sizeVariance": 0.7,
        "sizeOverLife": 1.4,
        "color": "rgba(200, 235, 255, 0.75)",
        "gravityScale": -0.08,
        "drag": 1.6,
        "fadeIn": 0.15,
        "fadeOut": 0.35,
        "maxParticles": 300,
        "width": width,
        "height": 0,
        "y": height * 1.05,
    }


#: 火花。衝撃点から放射状に飛ぶ（画面サイズに依らないので辞書のまま）。
_SPARK = {
    "rate": 120,
    "lifetime": 0.9,
    "lifetimeVariance": 0.4,
    "speed": 420,
    "speedVariance": 0.6,
    "direction": -90,
    "spread": 120,
    "size": 10,
    "sizeVariance": 0.5,
    "sizeOverLife": 0.2,
    "color": "#fff3c4",
    "endColor": "#ff6b35",
    "gravityScale": 0.9,
    "drag": 1.2,
    "fadeIn": 0.02,
    "fadeOut": 0.5,
    "maxParticles": 500,
}

#: 煙。大きくゆっくり広がって消える。
_SMOKE = {
    "rate": 26,
    "lifetime": 3.4,
    "lifetimeVariance": 0.4,
    "speed": 60,
    "speedVariance": 0.7,
    "direction": -90,
    "spread": 45,
    "size": 60,
    "sizeVariance": 0.5,
    "sizeOverLife": 2.4,
    "color": "rgba(190, 195, 210, 0.5)",
    "gravityScale": -0.05,
    "drag": 1.8,
    "spin": 30,
    "fadeIn": 0.2,
    "fadeOut": 0.6,
    "maxParticles": 220,
}

#: プリセット名 → 設定（関数のものは画面サイズを受け取ります）。
PARTICLE_PRESETS: dict[str, Callable[[float, float], dict] | dict] = {
    "rain": _rain,
    "snow": _snow,
    "sakura": _sakura,
    "confetti": _confetti,
    "bubble": _bubble,
    "spark": _SPARK,
    "smoke": _SMOKE,
}


def resolve_preset(name: str, width: float, height: float) -> dict | None:
    """プリセットを画面サイズに当てはめて辞書で返す。知らない名前なら `None`。"""
    preset = PARTICLE_PRESETS.get(name)
    if preset is None:
        return None
    return dict(preset(width, height)) if callable(preset) else dict(preset)


def list_particle_presets() -> list[str]:
    return sorted(PARTICLE_PRESETS.keys())


__all__ = ["PARTICLE_PRESETS", "list_particle_presets", "resolve_preset"]
