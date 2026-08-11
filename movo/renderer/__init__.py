"""movo-renderer — フレームを描くところ。

``movo/cli/bridge.py`` はこのパッケージの ``Renderer`` を «繋がった目印» に
しているので、ここで公開します。

**重い読み込みはしません。** ``movo list effects`` のように «一覧を見るだけ» の
コマンドでもここが読まれるので、``Renderer`` を import した時点でフォントの
走査や Numba のコンパイルが走らないようにしてあります（実体は ``index.py``）。
"""

from __future__ import annotations

from movo.renderer.effect_graph import apply_effect_graph
from movo.renderer.effects import apply_effect, has_effect, list_effects
from movo.renderer.font import FontManager
from movo.renderer.index import (
    DEFAULT_TRANSFORM,
    RENDERER_KINDS,
    Renderer,
    apply_scene_transition,
)
from movo.renderer.kernels import warmup as warm_up_kernels
from movo.renderer.shapes import render_shape
from movo.renderer.text import render_text

# ``movo/cli/parallel.py`` は **`warm_up_kernels`** という名前でここを探します
# （`bridge.pick("movo.renderer", "warm_up_kernels", "warmUpKernels")`）。
# 実体は `kernels.warmup()` で、名前だけが食い違っていました。見つからないと
# «並列で描く前に親で 1 回暖機する» が黙って飛ばされ、子プロセスの数だけ
# JIT のコンパイルが走ります（実測で 1 フレーム目に 10.6 秒）。**呼ぶ側の
# 名前に合わせて別名を貼ります** — カーネル側の `warmup` は
# `movo.renderer.kernels` の中で完結した名前として自然なためです。

__all__ = [
    "DEFAULT_TRANSFORM",
    "RENDERER_KINDS",
    "FontManager",
    "Renderer",
    "apply_effect",
    "apply_effect_graph",
    "apply_scene_transition",
    "has_effect",
    "list_effects",
    "render_shape",
    "render_text",
    "warm_up_kernels",
]
