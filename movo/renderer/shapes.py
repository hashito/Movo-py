"""図形レイヤーのラスタライズと、SVG のパス（`d` 文字列・`.svg` 素材）の取り込み。

長方形・角丸・円・楕円・多角形・星・線・矢印・三角形、そして自由なパスを
**それぞれ専用のビットマップ**に描きます。こうしておくと、変形や効果の段は
図形を «ただの画像» として扱えます。

パスは «独自の配列表現» のほかに、SVG の `d` 文字列と `.svg` 素材からも作れます。
SVG は外から持ち込まれるデータなので、**«実行しない・取りに行かない»** を守ります。

  - `<script>` や `on*=` のイベント属性は «読みも実行もしません»
  - `href` / `xlink:href` / `url(...)` などの外部参照は «一切たどりません»
  - `<image>` `<use>` `<style>` `<filter>` `<defs>` などは «丸ごと無視» します
  - 入力サイズに上限を設けます（既定 2 MB・**バイト数**で判定）

つまり «描画命令の座標» 以外は捨てます。ロゴを持ち込むには十分で、
攻撃面はほぼ «数字の列» だけになります。

SVG の解釈は 2 段構えです。

  1. :func:`parse_path_data` — `d` 文字列を «直線と 3 次ベジェだけ» の命令列に直す
  2. :func:`flatten_segments` — 命令列を折れ線（サブパス）に落とす

分けている理由は «変換行列を正確に掛けるため» です。折れ線にしてから行列を
掛けると、拡大したときに «曲線が角張ります»。制御点のまま掛けてから折れ線に
すれば、拡大しても滑らかなままです。

どの形にも `trim`（トリムパス）を掛けられます。線が «描かれていく» 演出は
これが基本になります。

## JSON のキーはそのまま

`cornerRadius` `innerRadius` `headSize` `fillRule` `strokeWidth` `svgAsset` …
**入力の辞書のキーは JS 版のまま camelCase です。** プロジェクト JSON に
そのまま書かれる名前なので、Python 側で snake_case に直すわけにはいきません。
Python の関数名と «戻り値» のキーだけ snake_case にしてあります。
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Sequence

import numpy as np

from .raster import (
    circle_contour,
    clamp,
    ellipse_contour,
    fill_coverage,
    fill_coverage_with,
    flatten_cubic,
    flatten_quadratic,
    parse_color,
    rasterize_contours,
    rect_contour,
    stroke_to_contours,
)

# ── 依存の解き方 ────────────────────────────────────────────────
#
# `movo.core` は別の担当が並行して移植しています。**まだ無いときでも
# このファイルだけで動くように**、最小限の代用を用意しておきます。
try:  # pragma: no cover - core が入ったらそちらを使う
    from movo.core.bitmap import Bitmap
except Exception:  # pragma: no cover
    Bitmap = None  # type: ignore[assignment]

try:  # pragma: no cover
    from movo.core.logger import logger as _logger
except Exception:  # pragma: no cover
    import logging

    _logger = logging.getLogger("movo")


def _warn(message: str) -> None:
    """警告を 1 行出す。**ここから例外を投げてはいけません。**

    ロゴが 1 つ壊れているだけで動画全体が出なくなるのを避けたいので、
    読めなかったものは «警告して空» にします。
    """
    warn = getattr(_logger, "warning", None) or getattr(_logger, "warn", None)
    if warn is not None:
        try:
            warn(message)
        except Exception:  # pragma: no cover - ログで落ちるのが一番まずい
            pass


#: 使える図形の種類。JS 版と同じ並びです。
SHAPE_KINDS = [
    "rectangle",
    "rect",
    "roundedRectangle",
    "circle",
    "ellipse",
    "polygon",
    "star",
    "line",
    "arrow",
    "triangle",
    "path",
    "svg",
]

#: 既定の入力サイズ上限（バイト）。外から来る SVG を «読む前に» 弾くための値です。
DEFAULT_SVG_MAX_BYTES = 2 * 1024 * 1024

#: `d` 文字列 1 本あたりに許す命令数。壊れた（あるいは意地の悪い）入力への保険です。
DEFAULT_MAX_SEGMENTS = 20000

#: 1 つの SVG から取り込む図形要素の数の上限。
DEFAULT_MAX_ELEMENTS = 2000

TAU = math.pi * 2

# 「指定されていない」を表す番兵。JS の `undefined` と `null` を区別したい
# ところ（`shape.closed === undefined`）で使います。
_MISSING = object()

#: 中身を «見ない» と決めた要素です。この名前が出てきたら閉じタグまで飛ばします。
#:
#: `defs` `clipPath` `mask` `marker` `symbol` `pattern` は «定義» であって
#: 描かれるものではないので、取り込むと «見えないはずの形» が出てしまいます。
#: `script` `style` `filter` `image` `use` `text` `foreignObject` は
#: 外部参照・実行・フォントが絡むので、そもそも触りません。
SKIPPED_ELEMENTS = frozenset(
    {
        "defs",
        "clippath",
        "mask",
        "marker",
        "symbol",
        "pattern",
        "script",
        "style",
        "filter",
        "image",
        "use",
        "text",
        "textpath",
        "tspan",
        "foreignobject",
        "metadata",
        "title",
        "desc",
        "switch",
        "animate",
        "animatemotion",
        "animatetransform",
        "set",
    }
)


# ══════════════════════════════════════════════════════════════════
# 小道具
# ══════════════════════════════════════════════════════════════════


def _is_number(value: Any) -> bool:
    """JS の `Number.isFinite` 相当。`True` / `False` は数値として扱いません。"""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _number(value: Any, default: float) -> float:
    """数値でなければ既定値。JS の `?? 既定値` を «数値限定» にしたものです。"""
    return float(value) if _is_number(value) else float(default)


def _coalesce(*values: Any) -> Any:
    """JS の `??` の連鎖。`None` でない最初の値を返します。"""
    for value in values:
        if value is not None:
            return value
    return None


def _points_array(points: Any) -> np.ndarray:
    """`[x0, y0, x1, y1, ...]`（list でも ndarray でも）を float64 の 1 次元にする。"""
    return np.asarray(points, dtype=np.float64).ravel()


def _points_list(points: Any) -> list[float]:
    """平らな点列を Python の list にする（JS の配列と同じ触り心地にするため）。"""
    if isinstance(points, list):
        return points
    return [float(v) for v in _points_array(points)]


def _subpath_points(subpath: Any) -> Any:
    """`{"points": [...]}` でも «点列そのもの» でも受け取れるようにする。"""
    if isinstance(subpath, dict):
        return subpath.get("points", [])
    return subpath


def _subpath_closed(subpath: Any) -> bool:
    return bool(subpath.get("closed")) if isinstance(subpath, dict) else False


def _js_number_text(value: float) -> str:
    """数値を JS のテンプレート文字列と同じ見た目にする。

    基本図形を `d` 文字列に直すときに使います。`10.0` ではなく `10` と書けば、
    あとで :func:`parse_path_data` が読み直したときの見た目が JS 版と揃います
    （どちらにせよ数値としては同じですが、デバッグのときに読みやすい）。
    """
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))
    return repr(float(value))


_JS_PARSE_FLOAT_RE = re.compile(r"^\s*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _parse_float(value: Any) -> float | None:
    """JS の `Number.parseFloat`。«先頭から読める分だけ» 読みます（`"100px"` → 100）。"""
    if _is_number(value):
        return float(value)
    if not isinstance(value, str):
        return None
    match = _JS_PARSE_FLOAT_RE.match(value)
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:  # pragma: no cover - 正規表現が通れば必ず読めます
        return None
    return number if math.isfinite(number) else None


# ══════════════════════════════════════════════════════════════════
# 1. `d` 文字列のパーサ
# ══════════════════════════════════════════════════════════════════


def parse_path_data(d: Any, max_segments: int | None = None) -> dict:
    """SVG の `d` 文字列を命令列に直します。

    返る命令は «3 種類だけ» です。後段を単純にするため、`H` `V` `Q` `T` `S` `A` は
    すべて `L` か `C` に潰してしまいます。

      ``{"op": "M", "values": [x, y]}``                        サブパスの開始
      ``{"op": "L", "values": [x, y]}``                        直線
      ``{"op": "C", "values": [c1x, c1y, c2x, c2y, x, y]}``    3 次ベジェ
      ``{"op": "Z"}``                                          閉じる

    :returns: ``{"segments": [...], "truncated": bool, "invalid": bool}``
    """
    text = d if isinstance(d, str) else ""
    limit = DEFAULT_MAX_SEGMENTS if max_segments is None else int(max_segments)
    segments: list[dict] = []
    truncated = False
    invalid = False

    index = 0
    length = len(text)

    separators = " \t\n\r\f,"

    def skip_separators() -> None:
        nonlocal index
        while index < length and text[index] in separators:
            index += 1

    def at_end() -> bool:
        skip_separators()
        return index >= length

    def read_number() -> float | None:
        """数値を 1 つ読みます。読めなければ `None`（＝そこで打ち切り）。

        **`float()` に丸投げしません。** どこまでが 1 つの数値かは «自分で»
        決める必要があります（`1.5.5` は 2 つの数値、`1-2` も 2 つ）。
        """
        nonlocal index
        skip_separators()
        start = index
        if index < length and (text[index] == "+" or text[index] == "-"):
            index += 1
        digits = 0
        while index < length and "0" <= text[index] <= "9":
            index += 1
            digits += 1
        if index < length and text[index] == ".":
            index += 1
            while index < length and "0" <= text[index] <= "9":
                index += 1
                digits += 1
        if digits == 0:
            index = start
            return None
        if index < length and (text[index] == "e" or text[index] == "E"):
            save = index
            index += 1
            if index < length and (text[index] == "+" or text[index] == "-"):
                index += 1
            exp_digits = 0
            while index < length and "0" <= text[index] <= "9":
                index += 1
                exp_digits += 1
            if exp_digits == 0:
                index = save
        try:
            value = float(text[start:index])
        except ValueError:  # pragma: no cover - 上の読み方なら必ず数値になります
            return None
        return value if math.isfinite(value) else None

    def read_flag() -> bool | None:
        """円弧のフラグを 1 つ読みます。

        フラグは **«1 文字»** です。`a1 1 0 011 1` のように区切りなしで詰められる
        ことがあるので、数値として読んではいけません（`011` を 11 と読みます）。
        """
        nonlocal index
        skip_separators()
        ch = text[index] if index < length else None
        if ch == "0" or ch == "1":
            index += 1
            return ch == "1"
        return None

    current_x = 0.0
    current_y = 0.0
    start_x = 0.0
    start_y = 0.0
    # 直前が C/S だったときの 2 つ目の制御点（S の鏡映に使う）
    last_cubic_x: float | None = None
    last_cubic_y: float | None = None
    # 直前が Q/T だったときの制御点（T の鏡映に使う）
    last_quad_x: float | None = None
    last_quad_y: float | None = None

    def push(segment: dict) -> bool:
        nonlocal truncated
        if len(segments) >= limit:
            truncated = True
            return False
        segments.append(segment)
        return True

    def move_to(x: float, y: float) -> bool:
        nonlocal current_x, current_y, start_x, start_y
        current_x = x
        current_y = y
        start_x = x
        start_y = y
        return push({"op": "M", "values": [x, y]})

    def line_to(x: float, y: float) -> bool:
        nonlocal current_x, current_y
        ok = push({"op": "L", "values": [x, y]})
        current_x = x
        current_y = y
        return ok

    def cubic_to(c1x: float, c1y: float, c2x: float, c2y: float, x: float, y: float) -> bool:
        nonlocal current_x, current_y
        ok = push({"op": "C", "values": [c1x, c1y, c2x, c2y, x, y]})
        current_x = x
        current_y = y
        return ok

    def quad_to(cx: float, cy: float, x: float, y: float) -> bool:
        # 2 次ベジェは 3 次に «厳密に» 直せます（制御点を 2/3 だけ寄せる）。
        return cubic_to(
            current_x + (2 / 3) * (cx - current_x),
            current_y + (2 / 3) * (cy - current_y),
            x + (2 / 3) * (cx - x),
            y + (2 / 3) * (cy - y),
            x,
            y,
        )

    command: str | None = None
    started = False

    while not at_end():
        ch = text[index]
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            command = ch
            index += 1
        elif command is None:
            # 先頭がコマンドでない＝壊れた `d`。読める分だけ返します。
            invalid = True
            break
        elif command == "Z" or command == "z":
            # `Z` に引数はありません。数字が続いていたら壊れています。
            invalid = True
            break
        elif command == "M":
            # «繰り返し引数»: `M` の 2 組目以降は暗黙の `L` です（`m` なら `l`）。
            command = "L"
        elif command == "m":
            command = "l"

        relative = "a" <= command <= "z"
        upper = command.upper()
        base_x = current_x if relative else 0.0
        base_y = current_y if relative else 0.0
        ok = True

        if upper == "M":
            x = read_number()
            y = read_number()
            if x is None or y is None:
                invalid = True
                ok = False
            else:
                ok = move_to(base_x + x, base_y + y)
                started = True
                last_cubic_x = None
                last_quad_x = None
        elif upper == "L":
            x = read_number()
            y = read_number()
            if x is None or y is None:
                invalid = True
                ok = False
            else:
                if not started:
                    ok = move_to(base_x + x, base_y + y)
                    started = True
                else:
                    ok = line_to(base_x + x, base_y + y)
                last_cubic_x = None
                last_quad_x = None
        elif upper == "H":
            x = read_number()
            if x is None:
                invalid = True
                ok = False
            else:
                ok = line_to(base_x + x, current_y)
                last_cubic_x = None
                last_quad_x = None
        elif upper == "V":
            y = read_number()
            if y is None:
                invalid = True
                ok = False
            else:
                ok = line_to(current_x, base_y + y)
                last_cubic_x = None
                last_quad_x = None
        elif upper == "C":
            values = [read_number() for _ in range(6)]
            if any(v is None for v in values):
                invalid = True
                ok = False
            else:
                c2x = base_x + values[2]
                c2y = base_y + values[3]
                ok = cubic_to(base_x + values[0], base_y + values[1], c2x, c2y, base_x + values[4], base_y + values[5])
                last_cubic_x = c2x
                last_cubic_y = c2y
                last_quad_x = None
        elif upper == "S":
            values = [read_number() for _ in range(4)]
            if any(v is None for v in values):
                invalid = True
                ok = False
            else:
                # 直前が C/S なら制御点を «現在点で鏡映»、そうでなければ現在点そのもの。
                c1x = current_x if last_cubic_x is None else current_x * 2 - last_cubic_x
                c1y = current_y if last_cubic_x is None else current_y * 2 - last_cubic_y
                c2x = base_x + values[0]
                c2y = base_y + values[1]
                ok = cubic_to(c1x, c1y, c2x, c2y, base_x + values[2], base_y + values[3])
                last_cubic_x = c2x
                last_cubic_y = c2y
                last_quad_x = None
        elif upper == "Q":
            values = [read_number() for _ in range(4)]
            if any(v is None for v in values):
                invalid = True
                ok = False
            else:
                cx = base_x + values[0]
                cy = base_y + values[1]
                ok = quad_to(cx, cy, base_x + values[2], base_y + values[3])
                last_quad_x = cx
                last_quad_y = cy
                last_cubic_x = None
        elif upper == "T":
            x = read_number()
            y = read_number()
            if x is None or y is None:
                invalid = True
                ok = False
            else:
                cx = current_x if last_quad_x is None else current_x * 2 - last_quad_x
                cy = current_y if last_quad_x is None else current_y * 2 - last_quad_y
                ok = quad_to(cx, cy, base_x + x, base_y + y)
                last_quad_x = cx
                last_quad_y = cy
                last_cubic_x = None
        elif upper == "A":
            rx = read_number()
            ry = read_number()
            rotation = read_number()
            large_arc = read_flag()
            sweep = read_flag()
            x = read_number()
            y = read_number()
            if rx is None or ry is None or rotation is None or large_arc is None or sweep is None or x is None or y is None:
                invalid = True
                ok = False
            else:
                end_x = base_x + x
                end_y = base_y + y
                curves = arc_to_cubics(current_x, current_y, rx, ry, rotation, large_arc, sweep, end_x, end_y)
                if not curves:
                    ok = line_to(end_x, end_y)
                else:
                    for curve in curves:
                        ok = cubic_to(curve[0], curve[1], curve[2], curve[3], curve[4], curve[5])
                        if not ok:
                            break
                last_cubic_x = None
                last_quad_x = None
        elif upper == "Z":
            ok = push({"op": "Z"})
            current_x = start_x
            current_y = start_y
            last_cubic_x = None
            last_quad_x = None
        else:
            # 未知のコマンド文字。ここから先は信用できないので打ち切ります。
            invalid = True
            ok = False

        if not ok:
            break

    return {"segments": segments, "truncated": truncated, "invalid": invalid}


def arc_to_cubics(
    x0: float,
    y0: float,
    rx_in: float,
    ry_in: float,
    rotation_deg: float,
    large_arc: bool,
    sweep: bool,
    x1: float,
    y1: float,
) -> list[list[float]]:
    """円弧（`A`）を 3 次ベジェの列に直します。

    手順は SVG 仕様の付録 F.6.5（端点表現 → 中心表現）そのままです。
    90 度を超える弧は «そのままでは近似できない» ので分割します。
    誤差は半径の 0.03 % 未満で、ロゴの角丸には十分な精度です。

    :returns: ``[c1x, c1y, c2x, c2y, x, y]`` の配列
    """
    rx = abs(rx_in)
    ry = abs(ry_in)
    # 半径 0、あるいは始点と終点が同じ弧は «直線» です（仕様どおり）。
    if rx < 1e-12 or ry < 1e-12:
        return []
    if abs(x1 - x0) < 1e-12 and abs(y1 - y0) < 1e-12:
        return []

    phi = (rotation_deg * math.pi) / 180
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx2 = (x0 - x1) / 2
    dy2 = (y0 - y1) / 2
    xp = cos_phi * dx2 + sin_phi * dy2
    yp = -sin_phi * dx2 + cos_phi * dy2

    # 半径が足りないときは «届く最小の大きさ» まで広げます（仕様どおり）。
    lam = (xp * xp) / (rx * rx) + (yp * yp) / (ry * ry)
    if lam > 1:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    rx2 = rx * rx
    ry2 = ry * ry
    numerator = max(0.0, rx2 * ry2 - rx2 * yp * yp - ry2 * xp * xp)
    denominator = rx2 * yp * yp + ry2 * xp * xp
    coefficient = (-1 if bool(large_arc) == bool(sweep) else 1) * math.sqrt(
        0.0 if denominator == 0 else numerator / denominator
    )
    cxp = (coefficient * (rx * yp)) / ry
    cyp = (coefficient * -(ry * xp)) / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x0 + x1) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y0 + y1) / 2

    start_vector_x = (xp - cxp) / rx
    start_vector_y = (yp - cyp) / ry
    end_vector_x = (-xp - cxp) / rx
    end_vector_y = (-yp - cyp) / ry

    theta1 = math.atan2(start_vector_y, start_vector_x)
    delta = math.atan2(end_vector_y, end_vector_x) - theta1
    if not sweep and delta > 0:
        delta -= TAU
    elif sweep and delta < 0:
        delta += TAU

    count = max(1, math.ceil(abs(delta) / (math.pi / 2)))
    step = delta / count
    # 4/3 tan(θ/4) は «円弧を 3 次ベジェで近似する» ときの標準の係数です。
    k = (4 / 3) * math.tan(step / 4)

    def point_at(theta: float) -> tuple[float, float]:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return (
            cx + rx * cos_t * cos_phi - ry * sin_t * sin_phi,
            cy + rx * cos_t * sin_phi + ry * sin_t * cos_phi,
        )

    def tangent_at(theta: float) -> tuple[float, float]:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return (
            -rx * sin_t * cos_phi - ry * cos_t * sin_phi,
            -rx * sin_t * sin_phi + ry * cos_t * cos_phi,
        )

    curves: list[list[float]] = []
    for i in range(count):
        t1 = theta1 + step * i
        t2 = t1 + step
        p1 = point_at(t1)
        p2 = point_at(t2)
        d1 = tangent_at(t1)
        d2 = tangent_at(t2)
        curves.append(
            [p1[0] + k * d1[0], p1[1] + k * d1[1], p2[0] - k * d2[0], p2[1] - k * d2[1], p2[0], p2[1]]
        )
    # 端点は «指定された値» に戻します。三角関数の誤差で継ぎ目が開くのを防ぎます。
    curves[-1][4] = x1
    curves[-1][5] = y1
    return curves


# ══════════════════════════════════════════════════════════════════
# 2. 命令列 → 折れ線（サブパス）
# ══════════════════════════════════════════════════════════════════


def _flatten_cubic_into(
    out: list[float],
    x0: float,
    y0: float,
    c1x: float,
    c1y: float,
    c2x: float,
    c2y: float,
    x1: float,
    y1: float,
    tolerance: float,
) -> None:
    """3 次ベジェを折れ線にします。

    分割数は制御多角形の «長さ» から決めるので、同じ入力からは必ず同じ点列が
    出ます（決定性）。分割の上限が :func:`movo.renderer.raster.flatten_cubic` と
    違う（96 対 64）のは JS 版どおりです。SVG のパスは 1 本が長くなりがちなので、
    こちらは少し細かく割ります。
    """
    distance = math.hypot(c1x - x0, c1y - y0) + math.hypot(c2x - c1x, c2y - c1y) + math.hypot(x1 - c2x, y1 - c2y)
    steps = min(96, max(3, math.ceil(math.sqrt(distance / max(1e-6, tolerance)) * 1.5)))
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        out.append(mt**3 * x0 + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t**3 * x1)
        out.append(mt**3 * y0 + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t**3 * y1)


def flatten_segments(
    segments: Sequence[dict] | None,
    transform: Sequence[float] | None = None,
    tolerance: float = 0.25,
) -> list[dict]:
    """命令列を折れ線に落とします。

    :param segments: :func:`parse_path_data` の結果の ``"segments"``
    :param transform: ``[a, b, c, d, e, f]``（SVG の matrix と同じ並び）。
        **«折れ線にする前» に掛ける**ので、拡大しても曲線が角張りません
    :returns: ``[{"points": [x0, y0, x1, y1, ...], "closed": bool}, ...]``
    """
    matrix = transform
    tol = 0.25 if tolerance is None else float(tolerance)

    if matrix is None:
        def map_point(x: float, y: float) -> tuple[float, float]:
            return (x, y)
    else:
        m = [float(v) for v in matrix]

        def map_point(x: float, y: float) -> tuple[float, float]:
            return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])

    subpaths: list[dict] = []
    points: list[float] | None = None
    closed = False
    cursor_x = 0.0
    cursor_y = 0.0

    def flush() -> None:
        nonlocal points, closed
        if points is not None and len(points) >= 4:
            subpaths.append({"points": points, "closed": closed})
        points = None
        closed = False

    for segment in segments or []:
        op = segment.get("op")
        values = segment.get("values") or []
        if op == "M":
            flush()
            x, y = map_point(values[0], values[1])
            points = [x, y]
            cursor_x = x
            cursor_y = y
        elif op == "L":
            if points is None:
                continue
            x, y = map_point(values[0], values[1])
            points.append(x)
            points.append(y)
            cursor_x = x
            cursor_y = y
        elif op == "C":
            if points is None:
                continue
            c1x, c1y = map_point(values[0], values[1])
            c2x, c2y = map_point(values[2], values[3])
            x, y = map_point(values[4], values[5])
            _flatten_cubic_into(points, cursor_x, cursor_y, c1x, c1y, c2x, c2y, x, y, tol)
            cursor_x = x
            cursor_y = y
        elif op == "Z":
            if points is not None and len(points) >= 4:
                closed = True
                cursor_x = points[0]
                cursor_y = points[1]
                restart_x = cursor_x
                restart_y = cursor_y
                flush()
                # `Z` の «後» に M なしで続く命令は、閉じた点から再開します。
                points = [restart_x, restart_y]
    flush()
    return subpaths


def path_to_subpaths(
    d: Any,
    transform: Sequence[float] | None = None,
    tolerance: float = 0.25,
    max_segments: int | None = None,
) -> list[dict]:
    """`d` 文字列を折れ線に落とすところまでを一息でやる入口です。"""
    parsed = parse_path_data(d, max_segments)
    return flatten_segments(parsed["segments"], transform, tolerance)


def subpaths_bounds(subpaths: Sequence[Any] | None) -> dict:
    """サブパス群を囲む矩形。

    **`width` / `height` の下限は `1e-6`** です（:func:`_bounds_of` の `1` とは
    違います。取り違えないこと）。SVG は座標系が小さいことがあるので、
    ここで 1 に切り上げると «拡大したときに位置がずれます»。
    """
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for subpath in subpaths or []:
        points = _points_array(_subpath_points(subpath))
        if points.size == 0:
            continue
        xs = points[0::2]
        ys = points[1::2]
        min_x = min(min_x, float(xs.min()))
        max_x = max(max_x, float(xs.max()))
        min_y = min(min_y, float(ys.min()))
        max_y = max(max_y, float(ys.max()))
    if min_x == math.inf:
        return {"min_x": 0.0, "min_y": 0.0, "max_x": 1.0, "max_y": 1.0, "width": 1.0, "height": 1.0}
    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": max(1e-6, max_x - min_x),
        "height": max(1e-6, max_y - min_y),
    }


# ══════════════════════════════════════════════════════════════════
# 3. トリムパス
# ══════════════════════════════════════════════════════════════════


def is_trim_active(trim: Any) -> bool:
    """トリムが «何かを削る» 指定かどうか。

    既定値のままなら触らずに返したいので使います。
    """
    if not isinstance(trim, dict):
        return False
    if trim.get("enabled") is False:
        return False
    start = trim["start"] if _is_number(trim.get("start")) else 0
    end = trim["end"] if _is_number(trim.get("end")) else 1
    offset = trim["offset"] if _is_number(trim.get("offset")) else 0
    return start > 1e-9 or end < 1 - 1e-9 or abs(offset) > 1e-9


def _subpath_length(subpath: Any) -> float:
    """折れ線の全長。閉じているときは «最後から最初へ» の 1 辺も数えます。"""
    points = _points_array(_subpath_points(subpath))
    if points.size < 4:
        return 0.0
    xs = points[0::2]
    ys = points[1::2]
    total = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
    if _subpath_closed(subpath):
        total += float(math.hypot(xs[0] - xs[-1], ys[0] - ys[-1]))
    return total


def _extract_range(subpath: Any, a: float, b: float) -> list[float] | None:
    """累積長で `[a, b]`（0〜1）の区間を切り出します。端点は線形補間で作ります。"""
    source = _points_array(_subpath_points(subpath))
    # 閉じたパスは «最初の点をもう一度» 足して 1 本の折れ線として扱います。
    if _subpath_closed(subpath):
        points = np.concatenate([source, source[0:2]])
    else:
        points = source
    count = points.size // 2
    if count < 2:
        return None

    xs = points[0::2]
    ys = points[1::2]
    steps = np.hypot(np.diff(xs), np.diff(ys))
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    if total <= 0:
        return None

    from_distance = a * total
    to_distance = b * total

    def point_at(distance: float) -> tuple[float, float]:
        for i in range(1, count):
            if distance <= cumulative[i] or i == count - 1:
                span = cumulative[i] - cumulative[i - 1]
                t = 0.0 if span <= 0 else clamp((distance - cumulative[i - 1]) / span, 0.0, 1.0)
                return (
                    float(xs[i - 1] + (xs[i] - xs[i - 1]) * t),
                    float(ys[i - 1] + (ys[i] - ys[i - 1]) * t),
                )
        return (float(xs[count - 1]), float(ys[count - 1]))  # pragma: no cover

    head = point_at(from_distance)
    out: list[float] = [head[0], head[1]]
    for i in range(1, count):
        if cumulative[i] > from_distance + 1e-9 and cumulative[i] < to_distance - 1e-9:
            out.append(float(xs[i]))
            out.append(float(ys[i]))
    tail = point_at(to_distance)
    out.append(tail[0])
    out.append(tail[1])
    return out


def _wrap_slices(subpath: Any, ranges: Sequence[Sequence[float]]) -> list[dict]:
    out: list[dict] = []
    for a, b in ranges:
        piece = _extract_range(subpath, a, b)
        if piece and len(piece) >= 4:
            out.append({"points": piece, "closed": False})
    return out


def _slice_subpath(subpath: Any, from_ratio: float, to_ratio: float, wrap: bool = True) -> list[dict]:
    """1 本のサブパスを `[from, to]`（0〜1）で切ります。

    閉じたパスは «回り込む» ので、結果が 2 本に分かれることがあります。
    """
    points = _points_array(_subpath_points(subpath))
    if points.size < 4:
        return []
    closed = _subpath_closed(subpath)
    if closed and wrap:
        full = to_ratio - from_ratio >= 1 - 1e-9
    else:
        full = from_ratio <= 1e-9 and to_ratio >= 1 - 1e-9
    if full:
        return [{"points": _points_list(points.copy()), "closed": closed}]

    if closed and wrap:
        # 回り込ませて [0, 1) に収めます。またぐときだけ 2 本に割ります。
        span = min(1.0, to_ratio - from_ratio)
        head = math.fmod(from_ratio, 1.0)
        if head < 0:
            head += 1
        tail = head + span
        if tail <= 1 + 1e-9:
            return _wrap_slices(subpath, [[head, min(1.0, tail)]])
        return _wrap_slices(subpath, [[head, 1.0], [0.0, tail - 1]])

    low = max(0.0, from_ratio)
    high = min(1.0, to_ratio)
    if high - low <= 1e-9:
        return []
    return _wrap_slices(subpath, [[low, high]])


def trim_subpaths(subpaths: Sequence[Any] | None, trim: Any) -> list[dict]:
    """サブパスを «長さの割合» で切り出します（トリムパス）。

    `start` / `end` は 0〜1、`offset` は «開始位置をずらす量»（同じく 0〜1）です。
    閉じたパスでは `offset` は «一周して» 回り込みます。開いたパスでははみ出した
    分は落とします（一周する «先» がないからです）。

    `mode` は 2 つあります。

      ``"each"``（既定）  サブパスごとに同じ割合で切る。全部が同時に伸びます
      ``"sequential"``    全サブパスを 1 本の長さとみなして順に切る。1 つずつ描かれます

    切った結果は **«開いた折れ線»** です。線が描かれていく演出が目的なので、
    閉じ直して塗ってしまうと «途中経過» が見えなくなるためです。
    """
    items = list(subpaths or [])
    if not trim or not is_trim_active(trim):
        return items
    raw_start = trim["start"] if _is_number(trim.get("start")) else 0.0
    raw_end = trim["end"] if _is_number(trim.get("end")) else 1.0
    offset = trim["offset"] if _is_number(trim.get("offset")) else 0.0
    from_ratio = min(raw_start, raw_end) + offset
    to_ratio = max(raw_start, raw_end) + offset
    if to_ratio - from_ratio <= 1e-9:
        return []

    if trim.get("mode") == "sequential":
        lengths = [_subpath_length(subpath) for subpath in items]
        total = sum(lengths)
        if total <= 0:
            return []
        out: list[dict] = []
        accumulated = 0.0
        for i, subpath in enumerate(items):
            share = lengths[i]
            if share <= 0:
                accumulated += share
                continue
            # 全体の割合を «このサブパスの中での割合» に読み替えます。
            local_from = (from_ratio * total - accumulated) / share
            local_to = (to_ratio * total - accumulated) / share
            # 順番に描く指定では回り込ませません。回り込むと «最後のサブパスが
            # 最初に戻ってくる» ことになり、順番という指定と矛盾するからです。
            out.extend(_slice_subpath(subpath, local_from, local_to, False))
            accumulated += share
        return out

    out = []
    for subpath in items:
        out.extend(_slice_subpath(subpath, from_ratio, to_ratio))
    return out


# ══════════════════════════════════════════════════════════════════
# 4. `.svg` からの取り込み（安全な範囲だけ）
# ══════════════════════════════════════════════════════════════════

# JS の正規表現と «同じ構造» にしてあります。`\w` は JS だと ASCII だけなので、
# Python では取り違えないように文字を並べて書きます（`re.UNICODE` が既定のため）。
_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_CDATA_RE = re.compile(r"<!\[CDATA\[[\s\S]*?\]\]>")
_PI_RE = re.compile(r"<\?[\s\S]*?\?>")
_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>\[]*(\[[\s\S]*?\])?[^>]*>", re.I)
_TAG_RE = re.compile(r"""<(/)?([A-Za-z_][A-Za-z0-9_.:-]*)((?:"[^"]*"|'[^']*'|[^>])*?)(/)?>""")
_ATTR_RE = re.compile(r"""([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")
_NUMBERS_RE = re.compile(r"-?[0-9.]+(?:[eE][+-]?[0-9]+)?")
_TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
_ENTITY_NUM_RE = re.compile(r"&#([0-9]+);")
_PERCENT_RE = re.compile(r"%\s*$")


def _local_name(name: str) -> str:
    """`xlink:href` のような接頭辞を落として要素名を小文字で返します。"""
    colon = name.find(":")
    return (name[colon + 1 :] if colon >= 0 else name).lower()


def _decode_entities(value: str) -> str:
    def numeric(match: re.Match) -> str:
        try:
            code = int(match.group(1))
        except ValueError:  # pragma: no cover
            return match.group(0)
        return chr(code) if 0 <= code <= 0x10FFFF else match.group(0)

    out = value.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
    out = _ENTITY_NUM_RE.sub(numeric, out)
    return out.replace("&amp;", "&")


def _parse_attributes(text: str) -> dict:
    """属性を読みます。

    **`on*=`（イベント）は «名前ごと捨てます»。`href` も読みません。**
    使わない値をわざわざ持ち歩かないのが一番安全だからです。
    """
    out: dict[str, str] = {}
    for match in _ATTR_RE.finditer(text or ""):
        raw_key = match.group(1)
        if raw_key[:2].lower() == "on":
            continue
        key = _local_name(raw_key)
        if key == "href":
            continue
        value = _coalesce(match.group(2), match.group(3), match.group(4)) or ""
        out["viewBox" if key == "viewbox" else key] = _decode_entities(value)
    return out


def _parse_length(value: Any) -> float:
    """`width="100px"` のような単位付きの長さを数値にします（% は 0 扱い）。"""
    if not value:
        return 0.0
    if isinstance(value, str) and _PERCENT_RE.search(value):
        return 0.0
    number = _parse_float(value)
    return number if number is not None else 0.0


def _find_numbers(text: Any) -> list[str]:
    return _NUMBERS_RE.findall(text) if isinstance(text, str) else []


def _element_to_path_data(name: str, attributes: dict) -> str | None:
    """基本図形を `d` 文字列に直します。

    ここを通すと、後段は «パスだけ» を相手にすれば済みます。
    """

    def number(key: str, fallback: float = 0.0) -> float:
        value = _parse_float(attributes.get(key))
        return value if value is not None else fallback

    def n(value: float) -> str:
        return _js_number_text(value)

    if name == "path":
        return attributes["d"] if attributes.get("d") else None
    if name == "rect":
        x = number("x")
        y = number("y")
        w = number("width")
        h = number("height")
        if w <= 0 or h <= 0:
            return None
        rx = number("rx") if "rx" in attributes else math.nan
        ry = number("ry") if "ry" in attributes else math.nan
        if math.isnan(rx) and math.isnan(ry):
            return f"M{n(x)} {n(y)}H{n(x + w)}V{n(y + h)}H{n(x)}Z"
        if math.isnan(rx):
            rx = ry
        if math.isnan(ry):
            ry = rx
        rx = min(max(0.0, rx), w / 2)
        ry = min(max(0.0, ry), h / 2)
        if rx == 0 or ry == 0:
            return f"M{n(x)} {n(y)}H{n(x + w)}V{n(y + h)}H{n(x)}Z"
        return (
            f"M{n(x + rx)} {n(y)}H{n(x + w - rx)}A{n(rx)} {n(ry)} 0 0 1 {n(x + w)} {n(y + ry)}"
            f"V{n(y + h - ry)}A{n(rx)} {n(ry)} 0 0 1 {n(x + w - rx)} {n(y + h)}"
            f"H{n(x + rx)}A{n(rx)} {n(ry)} 0 0 1 {n(x)} {n(y + h - ry)}"
            f"V{n(y + ry)}A{n(rx)} {n(ry)} 0 0 1 {n(x + rx)} {n(y)}Z"
        )
    if name == "circle":
        r = number("r")
        if r <= 0:
            return None
        cx = number("cx")
        cy = number("cy")
        return f"M{n(cx - r)} {n(cy)}A{n(r)} {n(r)} 0 1 0 {n(cx + r)} {n(cy)}A{n(r)} {n(r)} 0 1 0 {n(cx - r)} {n(cy)}Z"
    if name == "ellipse":
        rx = number("rx")
        ry = number("ry")
        if rx <= 0 or ry <= 0:
            return None
        cx = number("cx")
        cy = number("cy")
        return (
            f"M{n(cx - rx)} {n(cy)}A{n(rx)} {n(ry)} 0 1 0 {n(cx + rx)} {n(cy)}"
            f"A{n(rx)} {n(ry)} 0 1 0 {n(cx - rx)} {n(cy)}Z"
        )
    if name == "line":
        return f"M{n(number('x1'))} {n(number('y1'))}L{n(number('x2'))} {n(number('y2'))}"
    if name in ("polyline", "polygon"):
        numbers = _find_numbers(attributes.get("points", ""))
        if len(numbers) < 4:
            return None
        pairs = [f"{numbers[i]} {numbers[i + 1]}" for i in range(0, len(numbers) - 1, 2)]
        return "M" + "L".join(pairs) + ("Z" if name == "polygon" else "")
    return None


def extract_svg_shapes(
    source: Any,
    max_bytes: int | None = None,
    max_elements: int | None = None,
    tolerance: float = 0.25,
    max_segments: int | None = None,
) -> dict:
    """SVG の中身から «形» だけを取り出します。

    **扱うもの**（これだけです）

      - ``<path d="...">``
      - ``<rect>`` ``<circle>`` ``<ellipse>`` ``<line>`` ``<polyline>`` ``<polygon>``
      - それらの `transform`（`matrix` `translate` `scale` `rotate` `skewX` `skewY`）
      - ``<svg>`` の `viewBox` / `width` / `height`

    **無視するもの**

      - ``<script>`` ``<style>`` ``<filter>`` ``<image>`` ``<use>`` ``<text>``
        ``<foreignObject>`` — 実行・外部参照・フォントが絡むため、中身ごと飛ばします
      - ``<defs>`` ``<clipPath>`` ``<mask>`` ``<marker>`` ``<symbol>`` ``<pattern>``
        — «定義» であって描かれるものではないため
      - `fill` `stroke` `opacity` などの見た目、`id` `class`、`on*=` のイベント属性
      - `href` / `xlink:href` / `url(...)` — **«たどりません»**

    取りに行く処理が 1 つも無いので、SVG がどこから来ても «数字を読むだけ» で済みます。

    :returns: ``{"subpaths": [...], "view_box": [...]|None, "width": float,
        "height": float, "stats": {...}}``

    **例外は投げません。** 読めなかったら «警告して空» にします（JS 版は
    大きすぎる入力で `throw` していましたが、呼ぶ側で必ず捕まえていたので、
    Python では最初から «空を返す» ことにしました）。
    """
    limit_bytes = DEFAULT_SVG_MAX_BYTES if max_bytes is None else int(max_bytes)
    text = source if isinstance(source, str) else ""
    stats = {"paths": 0, "shapes": 0, "skipped": 0, "truncated": False, "invalid": False}

    # 文字数ではなく «バイト数» で見ます。日本語のコメントが入った SVG でも
    # ファイルサイズと同じ尺度になるからです。
    size = len(text.encode("utf-8"))
    if size > limit_bytes:
        _warn(f"SVG が大きすぎます（{size / 1024:.0f} KB > {limit_bytes / 1024:.0f} KB）。この図形は空になります")
        stats["invalid"] = True
        stats["oversize"] = True
        return {"subpaths": [], "view_box": None, "width": 0.0, "height": 0.0, "stats": stats}

    limit_elements = DEFAULT_MAX_ELEMENTS if max_elements is None else int(max_elements)
    # コメント・CDATA・DOCTYPE・処理命令は «先に» 落とします。中に `<path>` に
    # 見える文字列が入っていても拾わないようにするためです。
    cleaned = _COMMENT_RE.sub("", text)
    cleaned = _CDATA_RE.sub("", cleaned)
    cleaned = _PI_RE.sub("", cleaned)
    cleaned = _DOCTYPE_RE.sub("", cleaned)

    stack: list[list[float]] = [identity_matrix()]
    skip_depth = 0
    view_box: list[float] | None = None
    width = 0.0
    height = 0.0
    subpaths: list[dict] = []

    for match in _TAG_RE.finditer(cleaned):
        closing = bool(match.group(1))
        name = _local_name(match.group(2))
        attributes_text = match.group(3) or ""
        self_closing = bool(match.group(4))

        if skip_depth > 0:
            # 飛ばしている最中。開始タグで深くなり、閉じタグで戻ります。
            if not closing and not self_closing:
                skip_depth += 1
            elif closing:
                skip_depth -= 1
            continue

        if closing:
            if len(stack) > 1:
                stack.pop()
            continue

        if name in SKIPPED_ELEMENTS:
            stats["skipped"] += 1
            if not self_closing:
                skip_depth = 1
            continue

        attributes = _parse_attributes(attributes_text)
        parent = stack[-1]
        local = multiply(parent, parse_transform(attributes["transform"])) if attributes.get("transform") else parent

        if name == "svg":
            if not view_box and attributes.get("viewBox"):
                numbers = []
                for token in _find_numbers(attributes["viewBox"]):
                    try:
                        numbers.append(float(token))
                    except ValueError:
                        numbers.append(math.nan)
                if len(numbers) == 4 and all(_is_number(v) for v in numbers):
                    view_box = numbers
            if not width:
                width = _parse_length(attributes.get("width"))
            if not height:
                height = _parse_length(attributes.get("height"))

        d = _element_to_path_data(name, attributes)
        if d:
            if stats["paths"] + stats["shapes"] >= limit_elements:
                stats["truncated"] = True
                break
            if name == "path":
                stats["paths"] += 1
            else:
                stats["shapes"] += 1
            parsed = parse_path_data(d, max_segments)
            if parsed["truncated"]:
                stats["truncated"] = True
            if parsed["invalid"]:
                stats["invalid"] = True
            subpaths.extend(flatten_segments(parsed["segments"], local, tolerance))

        if not self_closing:
            stack.append(local)

    if not view_box and width > 0 and height > 0:
        view_box = [0.0, 0.0, width, height]
    if not width or not height:
        bounds = subpaths_bounds(subpaths)
        width = width or (view_box[2] if view_box else bounds["width"])
        height = height or (view_box[3] if view_box else bounds["height"])
    return {"subpaths": subpaths, "view_box": view_box, "width": width, "height": height, "stats": stats}


# ══════════════════════════════════════════════════════════════════
# 5. 変換行列
# ══════════════════════════════════════════════════════════════════


def identity_matrix() -> list[float]:
    return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """SVG の `[a, b, c, d, e, f]` 同士の積（`a` を先に、`b` を後に掛ける並び）。"""
    return [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]


def parse_transform(text: Any) -> list[float]:
    """`transform="translate(10 20) rotate(30)"` を 1 つの行列にまとめます。

    左から順に掛けます（SVG の規定どおり、右のものが «先に» 点に効きます）。
    """
    matrix = identity_matrix()
    if not isinstance(text, str):
        return matrix
    for match in _TRANSFORM_RE.finditer(text):
        name = match.group(1).lower()
        args: list[float] = []
        for token in _find_numbers(match.group(2)):
            try:
                value = float(token)
            except ValueError:
                continue
            if math.isfinite(value):
                args.append(value)
        step: list[float] | None = None
        if name == "matrix":
            if len(args) >= 6:
                step = list(args[:6])
        elif name == "translate":
            step = [1.0, 0.0, 0.0, 1.0, args[0] if len(args) > 0 else 0.0, args[1] if len(args) > 1 else 0.0]
        elif name == "scale":
            sx = args[0] if len(args) > 0 else 1.0
            sy = args[1] if len(args) > 1 else sx
            step = [sx, 0.0, 0.0, sy, 0.0, 0.0]
        elif name == "rotate":
            angle = ((args[0] if len(args) > 0 else 0.0) * math.pi) / 180
            cos = math.cos(angle)
            sin = math.sin(angle)
            rotation = [cos, sin, -sin, cos, 0.0, 0.0]
            if len(args) >= 3:
                # 中心指定は «移動 → 回転 → 戻す» に展開します。
                step = multiply(
                    multiply([1.0, 0.0, 0.0, 1.0, args[1], args[2]], rotation),
                    [1.0, 0.0, 0.0, 1.0, -args[1], -args[2]],
                )
            else:
                step = rotation
        elif name == "skewx":
            step = [1.0, 0.0, math.tan(((args[0] if args else 0.0) * math.pi) / 180), 1.0, 0.0, 0.0]
        elif name == "skewy":
            step = [1.0, math.tan(((args[0] if args else 0.0) * math.pi) / 180), 0.0, 1.0, 0.0, 0.0]
        if step:
            matrix = multiply(matrix, step)
    return matrix


# ══════════════════════════════════════════════════════════════════
# 6. 図形の輪郭
# ══════════════════════════════════════════════════════════════════


def _bounds_of(flat: Any) -> dict:
    """平らな点列を囲む矩形。

    **`width` / `height` の下限は `1`** です（:func:`subpaths_bounds` の
    `1e-6` とは違います。**値が違うので取り違えないこと**）。
    こちらは «図形の論理サイズ» なので、0 幅の線でも 1px は確保します。
    """
    points = _points_array(flat)
    if points.size == 0:
        return {"min_x": 0.0, "min_y": 0.0, "width": 1.0, "height": 1.0}
    xs = points[0::2]
    ys = points[1::2]
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    return {
        "min_x": min_x,
        "min_y": min_y,
        "width": max(1.0, max_x - min_x),
        "height": max(1.0, max_y - min_y),
    }


def _shift_to_origin(flat: Any, bounds: dict) -> np.ndarray:
    """囲む矩形の左上が原点に来るようにずらす。"""
    out = _points_array(flat).copy()
    if out.size:
        out[0::2] -= bounds["min_x"]
        out[1::2] -= bounds["min_y"]
    return out


def shape_contours(shape: dict, ctx: dict | None = None) -> dict:
    """図形の輪郭を «その図形の座標系» で組み立てます。

    `shape["trim"]` があるときは **«最後に»** 切ります。先に切ってしまうと、
    トリムを 0 → 1 に動かしたときに «囲む矩形が育って» 図形が動いて見えます。
    大きさは **«切る前の形»** から決めるのが正解です。

    :returns: ``{"contours": [...], "width": …, "height": …, "closed": bool,
        "trimmed": bool（切ったときだけ）}``
    """
    geometry = _base_shape_contours(shape, ctx or {})
    trim = shape.get("trim")
    if not is_trim_active(trim):
        return geometry
    subpaths = [{"points": points, "closed": geometry["closed"]} for points in geometry["contours"]]
    trimmed = trim_subpaths(subpaths, trim)
    out = dict(geometry)
    out["contours"] = [subpath["points"] for subpath in trimmed]
    # 切った断片は «開いた線» です。閉じて塗ってしまうと途中経過が見えません。
    out["closed"] = False
    out["trimmed"] = True
    return out


def _base_shape_contours(shape: dict, ctx: dict) -> dict:
    kind = _coalesce(shape.get("type"), shape.get("kind"), "rectangle")

    if kind == "circle":
        radius = max(0.5, _number(shape.get("radius"), 50))
        return {
            "contours": [circle_contour(radius, radius, radius, 64)],
            "width": radius * 2,
            "height": radius * 2,
            "closed": True,
        }

    if kind == "ellipse":
        rx = max(0.5, _number(shape.get("width"), 100) / 2)
        ry = max(0.5, _number(shape.get("height"), 60) / 2)
        return {
            "contours": [ellipse_contour(rx, ry, rx, ry, 72)],
            "width": rx * 2,
            "height": ry * 2,
            "closed": True,
        }

    if kind == "polygon":
        points = shape.get("points") or []
        if isinstance(points, (list, tuple)) and len(points) >= 3:
            flat: list[float] = []
            for p in points:
                if isinstance(p, dict):
                    flat.append(_number(_coalesce(p.get("x")), 0))
                    flat.append(_number(_coalesce(p.get("y")), 0))
                else:
                    flat.append(_number(p[0] if len(p) > 0 else None, 0))
                    flat.append(_number(p[1] if len(p) > 1 else None, 0))
            bounds = _bounds_of(flat)
            return {
                "contours": [_shift_to_origin(flat, bounds)],
                "width": bounds["width"],
                "height": bounds["height"],
                "closed": True,
            }
        sides = max(3, round(_number(shape.get("sides"), 6)))
        radius = max(0.5, _number(shape.get("radius"), 50))
        rotation = (_number(shape.get("rotation"), -90) * math.pi) / 180
        flat = []
        for i in range(sides):
            angle = rotation + (i / sides) * math.pi * 2
            flat.append(radius + math.cos(angle) * radius)
            flat.append(radius + math.sin(angle) * radius)
        return {"contours": [flat], "width": radius * 2, "height": radius * 2, "closed": True}

    if kind == "star":
        count = max(3, round(_number(_coalesce(shape.get("points"), shape.get("sides")), 5)))
        outer = max(0.5, _number(_coalesce(shape.get("radius"), shape.get("outerRadius")), 50))
        inner = max(0.1, _number(shape.get("innerRadius"), outer * 0.5))
        rotation = (_number(shape.get("rotation"), -90) * math.pi) / 180
        flat = []
        for i in range(count * 2):
            radius = outer if i % 2 == 0 else inner
            angle = rotation + (i / (count * 2)) * math.pi * 2
            flat.append(outer + math.cos(angle) * radius)
            flat.append(outer + math.sin(angle) * radius)
        return {"contours": [flat], "width": outer * 2, "height": outer * 2, "closed": True}

    if kind == "line":
        frm = shape.get("from") or [0, 0]
        to = shape.get("to") or [100, 0]
        flat = [
            _number(frm[0] if len(frm) > 0 else None, 0),
            _number(frm[1] if len(frm) > 1 else None, 0),
            _number(to[0] if len(to) > 0 else None, 0),
            _number(to[1] if len(to) > 1 else None, 0),
        ]
        bounds = _bounds_of(flat)
        return {
            "contours": [_shift_to_origin(flat, bounds)],
            "width": bounds["width"],
            "height": bounds["height"],
            "closed": False,
        }

    if kind == "arrow":
        length = _number(shape.get("length"), 120)
        head = _number(shape.get("headSize"), 24)
        thickness = _number(shape.get("thickness"), 8)
        half = thickness / 2
        flat = [
            0.0, head / 2 - half,
            length - head, head / 2 - half,
            length - head, 0.0,
            length, head / 2,
            length - head, head,
            length - head, head / 2 + half,
            0.0, head / 2 + half,
        ]
        return {"contours": [flat], "width": length, "height": head, "closed": True}

    if kind == "triangle":
        width = _number(shape.get("width"), 100)
        height = _number(shape.get("height"), 100)
        return {
            "contours": [[width / 2, 0.0, width, height, 0.0, height]],
            "width": width,
            "height": height,
            "closed": True,
        }

    if kind in ("svg", "path"):
        # SVG 由来（`d` 文字列 / インライン SVG / `.svg` 素材）が指定されていれば
        # そちらを優先します。無ければ従来の配列表現です。
        svg_subpaths = _resolve_svg_subpaths(shape, ctx)
        if svg_subpaths is not None:
            if len(svg_subpaths) == 0:
                return {"contours": [], "width": 1.0, "height": 1.0, "closed": False}
            bounds = subpaths_bounds(svg_subpaths)
            contours = [_shift_to_origin(subpath["points"], bounds) for subpath in svg_subpaths]
            # 閉じたサブパスが 1 本でもあれば «塗れる形» とみなします。
            # `closed` を明示されていればそれが優先です。
            declared = shape.get("closed", _MISSING)
            if declared is _MISSING:
                closed = any(subpath.get("closed") for subpath in svg_subpaths)
            else:
                closed = declared is not False
            return {"contours": contours, "width": bounds["width"], "height": bounds["height"], "closed": closed}
        parsed = _parse_path(_coalesce(shape.get("path"), shape.get("commands"), shape.get("points"), []))
        bounds = parsed["bounds"]
        shifted = [_shift_to_origin(contour, bounds) for contour in parsed["contours"]]
        return {
            "contours": shifted,
            "width": bounds["width"],
            "height": bounds["height"],
            "closed": shape.get("closed") is not False,
        }

    # roundedRectangle / rectangle / rect / それ以外
    width = max(0.5, _number(shape.get("width"), 100))
    height = max(0.5, _number(shape.get("height"), 100))
    radius = _number(_coalesce(shape.get("radius"), shape.get("cornerRadius")), 0)
    return {
        "contours": [rect_contour(0, 0, width, height, radius)],
        "width": width,
        "height": height,
        "closed": True,
    }


def _resolve_svg_subpaths(shape: dict, ctx: dict) -> list[dict] | None:
    """SVG 由来のサブパスを取り出します。無関係なら `None`（＝従来の配列表現）。

    受け付ける書き方は 3 つです。

      ``{"type": "path", "d": "M10 10 L90 10 Z"}``       SVG のパス文字列
      ``{"type": "path", "svg": "<svg ...>…</svg>"}``    SVG をそのまま貼る
      ``{"type": "path", "asset": "logo"}``              `.svg` 素材（`ctx["assets"]` が要る）

    `d` は文字列の配列でも書けます（サブパスを別々に管理したいとき用）。
    読めなかったときは **«警告して空»** にします。ここで例外を投げると、
    ロゴが 1 つ壊れているだけで動画全体が出なくなってしまうからです。
    """
    tolerance = shape.get("tolerance")
    tolerance = 0.25 if tolerance is None else tolerance

    inline_d = shape.get("d")
    if inline_d is None and isinstance(shape.get("path"), str):
        inline_d = shape["path"]
    if isinstance(inline_d, str) or isinstance(inline_d, (list, tuple)):
        entries = [inline_d] if isinstance(inline_d, str) else list(inline_d)
        out: list[dict] = []
        for entry in entries:
            if not isinstance(entry, str):
                continue
            out.extend(path_to_subpaths(entry, None, tolerance))
        return out

    if isinstance(shape.get("svg"), str):
        try:
            return extract_svg_shapes(shape["svg"], tolerance=tolerance)["subpaths"]
        except Exception as err:  # pragma: no cover - 保険（上は例外を投げない作り）
            _warn(f"inline svg shape could not be read: {err}")
            return []

    asset_name = _coalesce(shape.get("asset"), shape.get("svgAsset"))
    if asset_name:
        assets = ctx.get("assets") if isinstance(ctx, dict) else None
        get_svg = getattr(assets, "get_svg", None) if assets is not None else None
        if callable(get_svg):
            try:
                parsed = get_svg(asset_name)
            except Exception as err:
                _warn(f'svg asset "{asset_name}" could not be read: {err}')
                return []
            if not parsed:
                _warn(f'svg asset "{asset_name}" is unavailable; the shape is empty')
                return []
            if isinstance(parsed, dict):
                return parsed.get("subpaths") or []
            return getattr(parsed, "subpaths", None) or []
        _warn(f'svg asset "{asset_name}" cannot be resolved here (no asset store was passed to render_shape)')
        return []
    return None


def _parse_path(path: Any) -> dict:
    """独自のパス表現を読みます。

    受け付けるのは点の並び ``[[x, y], ...]`` か、命令の並び
    ``[{"m": [x, y]}, {"l": [x, y]}, {"q": [cx, cy, x, y]}, {"c": [...]}, {"z": True}]``
    です。
    """
    contours: list[list[float]] = []
    current: list[float] = []
    cursor_x = 0.0
    cursor_y = 0.0

    def push(x: float, y: float) -> None:
        nonlocal cursor_x, cursor_y
        current.append(x)
        current.append(y)
        cursor_x = x
        cursor_y = y

    for entry in path or []:
        if isinstance(entry, (list, tuple)):
            push(_number(entry[0] if len(entry) > 0 else None, 0), _number(entry[1] if len(entry) > 1 else None, 0))
            continue
        if not isinstance(entry, dict):
            continue
        move = _coalesce(entry.get("m"), entry.get("moveTo"))
        line = _coalesce(entry.get("l"), entry.get("lineTo"))
        quad = _coalesce(entry.get("q"), entry.get("quadTo"))
        cubic = _coalesce(entry.get("c"), entry.get("cubicTo"))
        if move is not None:
            if len(current) >= 4:
                contours.append(current)
            current = []
            push(move[0], move[1])
        elif line is not None:
            push(line[0], line[1])
        elif quad is not None:
            flatten_quadratic(current, cursor_x, cursor_y, quad[0], quad[1], quad[2], quad[3])
            cursor_x = quad[2]
            cursor_y = quad[3]
        elif cubic is not None:
            flatten_cubic(current, cursor_x, cursor_y, cubic[0], cubic[1], cubic[2], cubic[3], cubic[4], cubic[5])
            cursor_x = cubic[4]
            cursor_y = cubic[5]
        elif entry.get("z") or entry.get("close"):
            if len(current) >= 4:
                contours.append(current)
            current = []
    if len(current) >= 4:
        contours.append(current)
    flushed: list[float] = []
    for contour in contours:
        flushed.extend(contour)
    return {"contours": contours, "bounds": _bounds_of(flushed)}


# ══════════════════════════════════════════════════════════════════
# 7. グラデーション
# ══════════════════════════════════════════════════════════════════


def gradient_shader(fill: dict, width: float, height: float) -> Callable | None:
    """グラデーションのシェーダを作ります。

    **`fill_coverage_with` のシェーダは «ベクトル化» の約束です。**
    `shader(xs, ys)`（どちらも `(h, w)` の float64 配列）を受け取り、
    `(h, w, 4)` の色（RGB 0..255・A 0..1）を返します。1 画素ずつ Python の
    関数を呼ぶと、そこだけで数百ミリ秒かかってしまいます。

    色の補間は :func:`numpy.interp` に任せます。`stops` の外側は端の色で
    止まる（外挿しない）ので、JS 版の `sample()` と同じ振る舞いです。
    """
    raw_stops = fill.get("stops") or []
    stops = []
    for stop in raw_stops:
        if not isinstance(stop, dict):
            continue
        offset = clamp(_number(_coalesce(stop.get("offset"), stop.get("position")), 0), 0, 1)
        stops.append((offset, parse_color(_coalesce(stop.get("color"), "#ffffff"))))
    if not stops:
        return None
    stops.sort(key=lambda item: item[0])

    offsets = np.array([offset for offset, _ in stops], dtype=np.float64)
    colors = np.array([[c[0], c[1], c[2], c[3]] for _, c in stops], dtype=np.float64)

    kind = _coalesce(fill.get("type"), "linear")
    angle = (_number(fill.get("angle"), 90) * math.pi) / 180
    dx = math.cos(angle)
    dy = math.sin(angle)
    cx = _number(fill.get("centerX"), 0.5) * width
    cy = _number(fill.get("centerY"), 0.5) * height
    radius = _number(fill.get("radius"), 0.5) * max(width, height)
    projection_length = abs(dx * width) + abs(dy * height)

    def shader(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if kind == "radial":
            t = np.hypot(xs - cx, ys - cy) / (radius if radius else 1.0)
        else:
            px = xs - (width if dx < 0 else 0.0)
            py = ys - (height if dy < 0 else 0.0)
            t = (px * dx + py * dy) / (projection_length if projection_length else 1.0)
        p = np.clip(t, 0.0, 1.0)
        out = np.empty(p.shape + (4,), dtype=np.float64)
        for channel in range(4):
            out[..., channel] = np.interp(p, offsets, colors[:, channel])
        # RGB は JS の `mixColor` と同じく «丸めた整数» にそろえます。
        out[..., :3] = np.floor(out[..., :3] + 0.5)
        return out

    return shader


# ══════════════════════════════════════════════════════════════════
# 8. 描画
# ══════════════════════════════════════════════════════════════════


def render_shape(shape: dict, scale: float = 1.0, ctx: dict | None = None) -> dict:
    """図形を «その図形専用» のビットマップに描きます。

    :param scale: スーパーサンプリング倍率
    :param ctx: ``{"assets": <get_svg(name) を持つ何か>}``（`.svg` 素材を引くため）
    :returns: ``{"bitmap": Bitmap, "width": int, "height": int,
        "box_width": float, "box_height": float, "origin_x": float, "origin_y": float}``
    """
    geometry = shape_contours(shape, ctx or {})
    stroke = shape.get("stroke") if isinstance(shape.get("stroke"), dict) else None
    stroke_width = _number(
        _coalesce(stroke.get("width") if stroke else None, shape.get("strokeWidth")), 0
    ) * scale
    stroke_color = _coalesce(
        stroke.get("color") if stroke else None, shape.get("strokeColor"), "#000000"
    )
    explicit_fill = _coalesce(shape.get("fill"), shape.get("color"))

    if geometry.get("trimmed") and stroke_width <= 0:
        # トリムは «線» を切る指定なので、線幅が無いと 1 ドットも描けません。
        # 真っ黒なフレームを出すより、塗りの色で細い線を引くほうが親切です。
        trim = shape.get("trim") if isinstance(shape.get("trim"), dict) else {}
        stroke_width = max(0.5, _number(trim.get("width"), 2)) * scale
        if isinstance(explicit_fill, str):
            stroke_color = explicit_fill

    fill = explicit_fill if explicit_fill is not None else (None if stroke_width > 0 else "#ffffff")
    pad = math.ceil(stroke_width / 2 + scale)
    width = max(1, math.ceil(geometry["width"] * scale) + pad * 2)
    height = max(1, math.ceil(geometry["height"] * scale) + pad * 2)
    bitmap = Bitmap(width, height)

    # 図形の座標系 → ビットマップの画素座標。x も y も同じ式なので一括で掛けます。
    shifted = [_points_array(contour) * scale + pad for contour in geometry["contours"]]

    if fill and geometry["closed"]:
        region = rasterize_contours(shifted, width, height, _coalesce(shape.get("fillRule"), "nonzero"))
        # `stops` が **空の配列でも** グラデーション扱いです（JS の `[]` は真なので、
        # `if fill.get("stops")` と書くと空のときだけ «単色» に落ちて挙動が変わります）。
        if isinstance(fill, dict) and fill.get("stops") is not None:
            shader = gradient_shader(fill, width, height)
            if shader:
                fill_coverage_with(bitmap, region, shader, 1)
        else:
            fill_coverage(bitmap, region, fill, 1)

    if stroke_width > 0:
        stroke_contours: list[np.ndarray] = []
        for contour in shifted:
            stroke_contours.extend(stroke_to_contours(contour, stroke_width, geometry["closed"]))
        # 線の «向き» をそろえてから塗ります。
        #
        # 線は «1 辺ごとの四角形 ＋ 継ぎ目の円» に分けて作られます。四角形の
        # 回り方は線の進行方向で決まるので、円と逆向きになることがあります。
        # nonzero で塗ると逆向き同士は «打ち消し合って» 穴が開き、細かく折れた
        # 曲線（円弧やトリムした線）が点線のように見えてしまいます（issue #74）。
        # `stroke_to_contours` が全部を同じ向きにそろえてくれているので、
        # **ここでは fillRule を渡しません**（nonzero のままで重なりは «足される» だけ）。
        region = rasterize_contours(stroke_contours, width, height)
        fill_coverage(bitmap, region, stroke_color, 1)

    return {
        "bitmap": bitmap,
        "width": width,
        "height": height,
        # 図形の論理サイズと、その原点がビットマップのどこに座っているか。
        "box_width": geometry["width"],
        "box_height": geometry["height"],
        "origin_x": pad / scale,
        "origin_y": pad / scale,
    }


__all__ = [
    "DEFAULT_MAX_ELEMENTS",
    "DEFAULT_MAX_SEGMENTS",
    "DEFAULT_SVG_MAX_BYTES",
    "SHAPE_KINDS",
    "SKIPPED_ELEMENTS",
    "arc_to_cubics",
    "extract_svg_shapes",
    "flatten_segments",
    "gradient_shader",
    "identity_matrix",
    "is_trim_active",
    "multiply",
    "parse_path_data",
    "parse_transform",
    "path_to_subpaths",
    "render_shape",
    "shape_contours",
    "subpaths_bounds",
    "trim_subpaths",
]
