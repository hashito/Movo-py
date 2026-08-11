"""レンダラー内で共有する小さな道具。

移植元: ``packages/renderer/src/helpers.js``

どれも状態を持たない純粋な関数です。``index.py`` と ``layers_builtin.py`` /
``layers_generator.py`` の両方から使うので、ここに置いています。

**JS 版から移した先が 2 か所に分かれたもの**があります。

- ``findLayer`` は ``movo.timeline.find_layer`` に置きました。歩く相手が
  «タイムラインが作った ``children``» なので、その形を決めている側に置くのが
  自然だからです。ここからも再輸出しておくので、JS 版の地図のまま探しても
  見つかります。
- ``particleContour`` は ``movo.renderer.particles.particle_contour`` に既に
  移植済みです。**ただし引数の形が違います**（JS は粒オブジェクトを渡す、
  Python は粒の値をばらして渡す）。Python 版の粒は NumPy の «配列の束» で、
  «粒 1 個のオブジェクト» が存在しないためです。
"""

from __future__ import annotations

import math

from movo.core.math import TAU
from movo.renderer.effects import hash_string
from movo.renderer.raster import parse_color
from movo.timeline import find_layer  # noqa: F401  （JS 版の地図に合わせた再輸出）


def hash_code(value) -> int:
    """文字列から 32bit のハッシュ（FNV-1a）。レイヤー ID から乱数の種を作る。

    JS 版の ``hashCode`` そのものですが、**同じものが
    ``movo.renderer.effects.hash_string`` に既に移植されていました。**
    2 つ持つと «片方だけ 32bit の丸めを直した» という食い違いが起きるので、
    こちらは名前だけ残して中身を委ねます（JS 版の地図のまま探しても届きます）。
    """
    return hash_string("" if value is None else value)


def mix_css(a, b, t: float) -> tuple[float, float, float, float]:
    """2 つの CSS 色を混ぜる。粒の色が寿命で変わるときに使う。

    ``movo.renderer.raster.parse_color`` はタプル ``(r, g, b, a)`` を返すので、
    こちらもタプルで返します（``Bitmap.fill`` や ``fill_coverage`` がタプルを
    取るため、途中で形を変えない）。
    """
    fr, fg, fb, fa = parse_color(a)
    tr, tg, tb, ta = parse_color(b)
    k = min(1.0, max(0.0, t))
    return (fr + (tr - fr) * k, fg + (tg - fg) * k, fb + (tb - fb) * k, fa + (ta - fa) * k)


def rect_contour_flat(x: float, y: float, width: float, height: float, radius: float = 0) -> list[float]:
    """角丸の矩形を «平坦な数値の並び» で返す（波形のバーなどに使う）。

    ``raster.rect_contour`` と同じ形ですが、呼ぶ側の都合でこちらは
    リストをそのまま返します。
    """
    w = max(0.5, width)
    h = max(0.5, height)
    if radius <= 0:
        return [x, y, x + w, y, x + w, y + h, x, y + h]
    r = min(radius, w / 2, h / 2)
    points: list[float] = []

    def corner(cx: float, cy: float, start_angle: float) -> None:
        for i in range(7):
            angle = start_angle + (i / 6) * (math.pi / 2)
            points.append(cx + math.cos(angle) * r)
            points.append(cy + math.sin(angle) * r)

    corner(x + w - r, y + r, -math.pi / 2)
    corner(x + w - r, y + h - r, 0)
    corner(x + r, y + h - r, math.pi / 2)
    corner(x + r, y + r, math.pi)
    return points


def within_range(control: dict, time: float) -> bool:
    """``physicsControl`` の from/to の範囲内かどうか。"""
    control = control or {}
    start = control.get("from")
    end = control.get("to")
    if start is not None and time < start:
        return False
    if end is not None and time > end:
        return False
    return True


__all__ = ["TAU", "find_layer", "hash_code", "mix_css", "rect_contour_flat", "within_range"]
