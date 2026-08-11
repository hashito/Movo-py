"""文字まわりの追加機能。

    textBox      文字の実寸に追従する枠・帯
    textPath     円周や折れ線に沿って文字を並べる
    strokeOrder  一画ずつ書いていく（近似／外部データ）
    randomFont   文字ごとに別のフォントを割り当てる
    antialias / pixelGrid  ビットマップフォント風に «ドット» で描く

どれも `layout_text` の結果を土台にしています。文字の実寸が分かってから枠を描く、
輪郭をパス上へ移す、といった «レイアウト後の加工» です。

## レイアウトの持ち方

JS 版がオブジェクトリテラルで持っていたところは **`dict`（キーは snake_case）**
にしています。JSON から来る «設定» の側（`spec` や `style`）は
**camelCase のまま**です。JSON の書き方は JS 版と完全に互換なので、
入力のキーを勝手に直すと動かなくなります。
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

from . import kernels
from .raster import (
    Region,
    circle_contour,
    clamp,
    draw_bitmap,
    fill_coverage,
    rasterize_contours,
    rect_contour,
    stroke_to_contours,
)

try:  # pragma: no cover - core が入ったらそちらを使う
    from movo.core.bitmap import Bitmap
except Exception:  # pragma: no cover
    Bitmap = None

try:  # pragma: no cover
    from movo.core.logger import logger
except Exception:  # pragma: no cover
    logger = logging.getLogger("movo")

TAU = math.pi * 2


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ══════════════════════════════════════════════════════════════════
# 32 ビット整数の演算（JS と同じ乱数を出すため）
# ══════════════════════════════════════════════════════════════════
#
# JS の `>>>` `|` `^` `Math.imul` は **32 ビット**で回ります。Python の int は
# 無限精度なので、そのまま書くと違う数列になります。同じ JSON から同じ動画が
# 出る（決定性）ことを JS 版と共有するために、ここだけは丁寧に真似します。


def _to_uint32(x) -> int:
    return int(x) & 0xFFFFFFFF


def _to_int32(x) -> int:
    v = int(x) & 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _imul(a, b) -> int:
    """JS の `Math.imul`（32 ビットの符号付き積）。"""
    return _to_int32(_to_uint32(a) * _to_uint32(b))


def create_random(seed: int = 0) -> Callable[[], float]:
    """mulberry32。小さくて速く、絵の用途には十分な品質です。

    JS 版の `createRandom` と **1 ビットも違わない数列**を出します。
    """
    state = _to_uint32(seed) or 0x9E3779B9

    def next_value() -> float:
        nonlocal state
        state = _to_uint32(state + 0x6D2B79F5)
        t = state
        t = _imul(_to_int32(t) ^ (_to_uint32(t) >> 15), _to_int32(t) | 1)
        t = _to_int32(_to_int32(t) ^ _to_int32(t + _imul(_to_int32(t) ^ (_to_uint32(t) >> 7), _to_int32(t) | 61)))
        return _to_uint32(_to_int32(t) ^ (_to_uint32(t) >> 14)) / 4294967296

    return next_value


# ══════════════════════════════════════════════════════════════════
# ぼかし（影に使う）
# ══════════════════════════════════════════════════════════════════

try:  # pragma: no cover - core が入ったらそちらを使う
    from movo.core.blur import box_blur as _core_box_blur
except Exception:  # pragma: no cover
    _core_box_blur = None


def blur_axis(bitmap, radius: float, horizontal: bool):
    """1 軸だけボックスぼかし（新しいビットマップを返す）。"""
    r = max(0, round(radius))
    if r == 0:
        return bitmap.copy()
    out = type(bitmap)(bitmap.width, bitmap.height)
    kernels.blur_axis_kernel(bitmap.data, out.data, int(r), 1 if horizontal else 0)
    return out


def separable_blur(bitmap, radius_x: float, radius_y: float, passes: int = 3):
    """縦横で別々の半径をかけてぼかす（方向性ブラーの下地）。"""
    rx = max(0, round(radius_x))
    ry = max(0, round(radius_y))
    if rx == 0 and ry == 0:
        return bitmap.copy()
    current = bitmap
    for _ in range(passes):
        if rx > 0:
            current = blur_axis(current, rx, True)
        if ry > 0:
            current = blur_axis(current, ry, False)
    return current


def box_blur(bitmap, radius: float, passes: int = 3):
    """縦横に同じ半径でぼかす。

    ガウスではなくボックスを重ねる方式です。半径に対して線形時間で済み、
    3 回重ねればガウスとほぼ見分けが付きません。
    """
    if _core_box_blur is not None:  # pragma: no cover
        return _core_box_blur(bitmap, radius, passes)
    return separable_blur(bitmap, radius, radius, passes)


# ══════════════════════════════════════════════════════════════════
# カバレッジを «ドット» に丸める（ビットマップフォント風）
# ══════════════════════════════════════════════════════════════════


def quantize_coverage(region: Region, width: int, height: int, style: dict | None) -> Region:
    """カバレッジを «ドット» に丸める。

    ドット絵の文字は 8x8 / 8x16 のビットマップフォントで、アンチエイリアスが
    ありません。なめらかな文字を描いてから画面全体を `pixelate` すると、
    「あとから粗くした」ムラが出てしまうので、描く段階で潰します。

      `pixelGrid` … グリッド内の被覆率を平均して字形そのものを量子化する
      `antialias: False` … 被覆率を 0/1 に丸めて中間調を無くす

    グリッドは **ビットマップの原点からの絶対座標**で切ります。文字ごとに位相が
    ずれると、同じ字なのに違うドットの並びになってしまうためです。

    入力の `region` を書き換えて、そのまま返します（JS 版と同じ）。
    """
    style = style or {}
    grid = max(0, round(style.get("pixelGrid") or 0))
    hard = style.get("antialias") is False
    if grid < 2 and not hard:
        return region
    if region.max_x < region.min_x or region.max_y < region.min_y:
        return region

    if grid >= 2:
        x0 = math.floor(region.min_x / grid) * grid
        y0 = math.floor(region.min_y / grid) * grid
        x1 = min(width - 1, math.ceil((region.max_x + 1) / grid) * grid - 1)
        y1 = min(height - 1, math.ceil((region.max_y + 1) / grid) * grid - 1)
        kernels.quantize_grid(region.coverage, width, height, x0, y0, x1, y1, int(grid), 1 if hard else 0)
        # グリッドに合わせて広がったぶん、描画範囲も広げる
        region.min_x = max(0, x0)
        region.min_y = max(0, y0)
        region.max_x = x1
        region.max_y = y1
        return region

    kernels.quantize_hard(region.coverage, region.min_x, region.min_y, region.max_x, region.max_y)
    return region


# ══════════════════════════════════════════════════════════════════
# 文字に追従する枠（textBox）
# ══════════════════════════════════════════════════════════════════


def resolve_padding(value) -> list[float]:
    """padding の書き方を `[上, 右, 下, 左]` に揃える。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [value, value, value, value]
    if not isinstance(value, (list, tuple)):
        return [0, 0, 0, 0]
    if len(value) == 2:
        return [value[0], value[1], value[0], value[1]]
    if len(value) == 4:
        return list(value)
    if len(value) == 1:
        return [value[0], value[0], value[0], value[0]]
    return [0, 0, 0, 0]


def draw_text_box(text_bitmap, rendered: dict, spec: dict) -> dict:
    """文字に追従する枠を描く。

    文字を描いたビットマップと、そのレイアウト結果を受け取り、枠を足した新しい
    ビットマップを返します。文字数が変われば枠も自動で変わります。

    :param rendered: `render_text` の戻り（`offset_x` / `offset_y` / `layout`）
    :param spec: textBox の設定（キーは JSON のまま camelCase）
    """
    top, right, bottom, left = resolve_padding(spec.get("padding", 16))
    radius = spec.get("radius", 0) or 0
    skew_x = spec.get("skewX", 0) or 0
    stroke = spec.get("stroke") or {}
    stroke_width = stroke.get("width", 0) or 0
    shadow = spec.get("shadow")
    shadow_offset_x = (shadow or {}).get("offsetX", 0) or 0
    shadow_offset_y = (shadow or {}).get("offsetY", 0) or 0
    shadow_blur = (shadow or {}).get("blur", 0) or 0

    # 文字の実寸（padding を除いた部分）
    layout = rendered["layout"]
    content_width = layout["width"]
    content_height = layout["height"]
    box_width = content_width + left + right
    box_height = content_height + top + bottom

    # 傾けたぶんと縁取り・影のぶん、外側に余白を取る
    skew_pad = abs(skew_x) * box_height
    pad_outer = math.ceil(stroke_width + max(0, shadow_blur) + abs(shadow_offset_x) + abs(shadow_offset_y) + 2)
    width = math.ceil(box_width + skew_pad + pad_outer * 2)
    height = math.ceil(box_height + pad_outer * 2)
    out = Bitmap(max(1, width), max(1, height))

    # 枠の左上（傾けた分だけ右へ寄せる）
    box_x = pad_outer + skew_pad / 2
    box_y = pad_outer

    # reveal は枠が伸びながら現れる演出。0 で幅ゼロ、1 で全幅。
    reveal = spec.get("reveal")
    progress = clamp(reveal.get("progress", 1) if reveal else 1, 0, 1)
    direction = (reveal or {}).get("direction", "left")
    draw_width = box_width
    draw_x = box_x
    if progress < 1:
        draw_width = box_width * progress
        if direction == "right":
            draw_x = box_x + (box_width - draw_width)
        elif direction == "center":
            draw_x = box_x + (box_width - draw_width) / 2

    def shape_at(x, y, w, h) -> list[float]:
        base = rect_contour(x, y, w, h, radius)
        if skew_x == 0:
            return list(base)
        # 上辺ほど右へ（skewX が負なら左へ）ずらす平行四辺形にする
        skewed = list(base)
        for i in range(0, len(base), 2):
            local_y = (base[i + 1] - y) / max(1e-6, h)
            skewed[i] = base[i] + (0.5 - local_y) * skew_x * h
            skewed[i + 1] = base[i + 1]
        return skewed

    if draw_width > 0.5:
        contour = shape_at(draw_x, box_y, draw_width, box_height)

        if shadow:
            shadow_layer = Bitmap(out.width, out.height)
            shifted = list(contour)
            for i in range(0, len(contour), 2):
                shifted[i] = contour[i] + shadow_offset_x
                shifted[i + 1] = contour[i + 1] + shadow_offset_y
            region = rasterize_contours([shifted], out.width, out.height)
            fill_coverage(shadow_layer, region, shadow.get("color", "rgba(0,0,0,0.4)"), 1)
            draw_bitmap(out, box_blur(shadow_layer, shadow_blur, 2) if shadow_blur > 0 else shadow_layer, 0, 0, 1)

        if spec.get("fill"):
            region = rasterize_contours([contour], out.width, out.height)
            fill_coverage(out, region, spec["fill"], 1)
        if stroke_width > 0:
            region = rasterize_contours(stroke_to_contours(contour, stroke_width, True), out.width, out.height)
            fill_coverage(out, region, stroke.get("color", "#000000"), 1)

    # 文字は枠の中央に置く。reveal 中は文字を出さない指定もできる。
    text_delay = (reveal or {}).get("textAfter", 0) or 0
    if not reveal or progress >= text_delay:
        text_x = round(box_x + left - rendered["offset_x"])
        text_y = round(box_y + top - rendered["offset_y"])
        draw_bitmap(out, text_bitmap, text_x, text_y, 1)

    # アンカーが枠基準になるよう、原点を枠の左上に合わせて返す
    return {"bitmap": out, "offset_x": box_x, "offset_y": box_y, "box_width": box_width, "box_height": box_height}


# ══════════════════════════════════════════════════════════════════
# パス上に文字を並べる（textPath）
# ══════════════════════════════════════════════════════════════════


def layout_text_on_path(layout: dict, spec: dict, contours_for: Callable) -> dict | None:
    """パス上に文字を並べる。

    各グリフの «送り» を弧長に読み替え、その位置と接線を求めて輪郭を移します。
    `firstMargin` を動かすと文字がパスの上を流れます。

    :param contours_for: `(glyph, glyph_index) -> list[list[float]]`。
        «原点にベースライン、字の左端が x=0» で返す約束です
    """
    shape = spec.get("shape", "circle")
    flip = spec.get("flip") is True
    perpendicular = spec.get("perpendicular") is not False
    first_margin = spec.get("firstMargin", 0) or 0

    # パスを「弧長 → 位置と接線」に変換できる形にする
    if shape in ("circle", "arc"):
        radius = max(1, spec.get("radius", 200))
        start_angle = (spec.get("startAngle", -90)) * math.pi / 180
        sweep = (spec.get("sweep", 360)) * math.pi / 180
        total_length = abs(sweep) * radius
        sign = 1.0 if (sweep or 1) > 0 else -1.0

        def sample(distance: float) -> tuple[float, float, float]:
            angle = start_angle + (distance / radius) * sign
            # 接線の向き。flip で内側／外側が入れ替わる
            return (
                math.cos(angle) * radius,
                math.sin(angle) * radius,
                angle + (-math.pi / 2 if flip else math.pi / 2),
            )
    else:
        raw = spec.get("points") or []
        points = [list(p) if isinstance(p, (list, tuple)) else [p.get("x", 0), p.get("y", 0)] for p in raw]
        if len(points) < 2:
            return None
        lengths = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            lengths.append(total)
        total_length = total

        def sample(distance: float) -> tuple[float, float, float]:
            clamped = clamp(distance, 0, total)
            for i in range(1, len(lengths)):
                if clamped <= lengths[i] or i == len(lengths) - 1:
                    span = (lengths[i] - lengths[i - 1]) or 1
                    t = clamp((clamped - lengths[i - 1]) / span, 0, 1)
                    angle = math.atan2(points[i][1] - points[i - 1][1], points[i][0] - points[i - 1][0])
                    return (
                        lerp(points[i - 1][0], points[i][0], t),
                        lerp(points[i - 1][1], points[i][1], t),
                        angle + (math.pi if flip else 0),
                    )
            return (points[0][0], points[0][1], 0.0)

    # 全グリフを 1 列に並べ、送り量の累積を弧長として使う
    entries = []
    advance = 0.0
    for line in layout["lines"]:
        for glyph in line["glyphs"]:
            entries.append((glyph, advance + glyph["advance"] / 2))
            advance += glyph["advance"]
    text_length = advance

    start = first_margin
    align = spec.get("align", "start")
    if align == "center":
        start += (total_length - text_length) / 2
    elif align == "end":
        start += total_length - text_length

    contours: list[list[float]] = []
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for index, (glyph, offset) in enumerate(entries):
        ax, ay, tangent = sample(start + offset)
        raw_contours = contours_for(glyph, index)
        if not raw_contours:
            continue
        # グリフは「ベースライン上・原点が字の左」で来るので、中心に寄せて回す。
        # tangent は «ベースラインを向けたい方向» なので、そのまま回転角にする。
        angle = tangent if perpendicular else 0.0
        cos = math.cos(angle)
        sin = math.sin(angle)
        shift_x = glyph["advance"] / 2
        for contour in raw_contours:
            moved = [0.0] * len(contour)
            for i in range(0, len(contour), 2):
                lx = contour[i] - shift_x
                ly = contour[i + 1]
                x = ax + lx * cos - ly * sin
                y = ay + lx * sin + ly * cos
                moved[i] = x
                moved[i + 1] = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
            contours.append(moved)

    if not contours:
        return None
    return {"contours": contours, "min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}


# ══════════════════════════════════════════════════════════════════
# 書き順アニメーション（近似）
# ══════════════════════════════════════════════════════════════════


def apply_stroke_order(contours: list, spec: dict, time: float) -> list:
    """一画ずつ現れるように見せる。

    正確な書き順にはデータが要ります。データが無いときは輪郭を
    「左上から右下へ」の順に並べて、一画ずつ現れるように見せます。
    漢字の «正しい» 書き順とは限らないので、そこは割り切りです。
    """
    if not contours:
        return contours
    stagger = spec.get("stagger", 0.12)
    duration = max(1e-4, spec.get("duration", 0.25))
    delay = spec.get("delay", 0) or 0

    # 輪郭の重心で「左上から右下へ」の順を作る
    ranked = []
    for index, contour in enumerate(contours):
        count = len(contour) / 2
        sx = float(np.sum(np.asarray(contour, dtype=np.float64)[0::2]))
        sy = float(np.sum(np.asarray(contour, dtype=np.float64)[1::2]))
        ranked.append((sy / count + sx / count * 0.4, index, contour))
    # Python の sorted は安定なので、rank が同じなら元の順のまま（JS と同じ）
    ranked.sort(key=lambda entry: entry[0])

    out = []
    for order, (_rank, _index, contour) in enumerate(ranked):
        start_at = delay + order * stagger
        progress = clamp((time - start_at) / duration, 0, 1)
        if progress <= 0:
            continue
        if progress >= 1:
            out.append(contour)
            continue
        # 書きかけの画は、輪郭の «先» を切り落として «伸びている» ように見せる
        out.append(_truncate_contour(contour, progress))
    return out


def _truncate_contour(contour, progress: float) -> list[float]:
    """輪郭を頭から `progress` の割合だけ残す。"""
    count = len(contour) // 2
    keep = max(2, round(count * progress))
    if keep >= count:
        return contour
    out = []
    for i in range(keep):
        out.append(contour[i * 2])
        out.append(contour[i * 2 + 1])
    # 閉じるために始点へ戻す
    out.append(contour[0])
    out.append(contour[1])
    return out


# ══════════════════════════════════════════════════════════════════
# ランダムフォント
# ══════════════════════════════════════════════════════════════════


def random_font_for(index: int, spec: dict | None, time: float = 0) -> str | None:
    """文字ごとに使うフォント名を決める。

    `interval > 0` なら、その秒数ごとに割り当てが変わります（ちらつき演出）。
    """
    families = (spec or {}).get("families") or []
    if not families:
        return None
    interval = spec.get("interval", 0) or 0
    step = math.floor(time / interval) if interval > 0 else 0
    seed = _to_uint32(
        _to_int32(spec.get("seed", 3)) ^ _to_int32(index * 2654435761) ^ _to_int32(step * 40503)
    )
    random = create_random(seed)
    return families[math.floor(random() * len(families)) % len(families)]


__all__ = [
    "TAU",
    "apply_stroke_order",
    "blur_axis",
    "box_blur",
    "circle_contour",
    "create_random",
    "draw_text_box",
    "fill_coverage",
    "layout_text_on_path",
    "quantize_coverage",
    "random_font_for",
    "rasterize_contours",
    "resolve_padding",
    "separable_blur",
]
