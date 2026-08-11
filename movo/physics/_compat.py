"""物理演算がまわりのパッケージから借りているもの。

**中身は持ちません。** `movo.core` と `movo.animation` にあるものをそのまま
使い、この 1 か所に «どこから借りているか» を集めています。

移植を並列に進めていたとき、ここには mulberry32 や `Mat2D` の写しが置いて
ありました。core が入った今それを残すと、**同じ乱数列の実装が 2 つ**あって
片方だけ直る、という事故のもとになります。粒の軌跡は乱数列で決まるので、
そこが 2 系統あるのはいちばん危ない形でした。

`resolve_animated` / `apply_animations` だけは «無くても動く» 形にしてあります。
リグ（`rig.py`）はキーフレームを読みますが、物理そのものは読まないので、
`movo.animation` が無い状態でも剛体と粒は動かせるようにしておくためです。
"""

from __future__ import annotations

from movo.core.math import Mat2D, clamp
from movo.core.rng import create_random


def resolve_animated(spec, ctx=None, default=None):
    """«動く値» を今の時刻で 1 つの数に潰す。"""
    try:
        from movo.animation.resolver import resolve_animated as impl

        return impl(spec, ctx, default)
    except ImportError:  # pragma: no cover - animation が無い環境向け
        if spec is None:
            return default
        if isinstance(spec, (int, float)):
            return spec
        return default


def apply_animations(state: dict, animations, ctx=None) -> None:
    """`animations` の結果を `state` に書き込む。"""
    try:
        from movo.animation.resolver import apply_animations as impl

        impl(state, animations, ctx)
    except ImportError:  # pragma: no cover
        return


__all__ = ["clamp", "create_random", "Mat2D", "resolve_animated", "apply_animations"]
