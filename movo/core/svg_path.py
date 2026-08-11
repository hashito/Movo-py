"""SVG のパス（``d`` 文字列）と ``.svg`` ファイルを «形» として取り込みます。

ここが扱うのは **«パスの形» だけ**です。SVG は外から持ち込まれるデータなので、
«実行しない・取りに行かない» を守ります。具体的には次のとおりです。

- ``<script>`` や ``on*=`` のイベント属性は «読みも実行もしません»
- ``href`` / ``xlink:href`` / ``url(...)`` などの外部参照は **一切たどりません**
- ``<image>`` ``<use>`` ``<style>`` ``<filter>`` などは «丸ごと無視» します
- 入力サイズに上限を設けます（既定 2 MB）。大きすぎるものは読みません

つまり «描画命令の座標» 以外は捨てます。ロゴを持ち込むには十分で、攻撃面は
ほぼ «数字の列» だけになります。

**XML パーサ（``xml.etree``）を使っていません。** 標準の XML パーサは実体参照の
展開を通じて外部ファイルを読みに行く経路（XXE）を持ちます。ここでは «タグの
形をした文字列» を正規表現で拾うだけにして、その経路そのものを無くしています。

実装は 2 段構えです。

1. :func:`parse_path_data` — ``d`` 文字列を «直線と 3 次ベジェだけ» の命令列に直す
2. :func:`flatten_segments` — 命令列を折れ線（サブパス）に落とす

分けている理由は **«変換行列を正確に掛けるため»** です。折れ線にしてから行列を
掛けると、拡大したときに «曲線が角張ります»。制御点のまま掛けてから折れ線に
すれば、拡大しても滑らかなままです。

円弧（``A``）は 3 次ベジェに直します。90 度ごとに分割して近似するので、誤差は
半径の 0.03% 未満です。ロゴの角丸には十分な精度です。
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

#: 既定の入力サイズ上限（バイト）。外から来る SVG を «読む前に» 弾くための値です。
DEFAULT_SVG_MAX_BYTES = 2 * 1024 * 1024

#: ``d`` 文字列 1 本あたりに許す命令数。壊れた（あるいは意地の悪い）入力への保険です。
DEFAULT_MAX_SEGMENTS = 20000

#: 1 つの SVG から取り込む図形要素の数の上限。
DEFAULT_MAX_ELEMENTS = 2000

_TAU = math.pi * 2

#: 中身を «見ない» と決めた要素。この名前が出てきたら閉じタグまで飛ばします。
#:
#: ``defs`` ``clipPath`` ``mask`` ``marker`` ``symbol`` ``pattern`` は «定義» で
#: あって描かれるものではないので、取り込むと «見えないはずの形» が出てしまいます。
#: ``script`` ``style`` ``filter`` ``image`` ``use`` ``text`` ``foreignObject`` は
#: 外部参照・実行・フォントが絡むので、そもそも触りません。
SKIPPED_ELEMENTS = frozenset(
    {
        "defs", "clippath", "mask", "marker", "symbol", "pattern",
        "script", "style", "filter", "image", "use", "text", "textpath", "tspan",
        "foreignobject", "metadata", "title", "desc", "switch",
        "animate", "animatemotion", "animatetransform", "set",
    }
)

Matrix = list[float]
Segment = dict[str, Any]
Subpath = dict[str, Any]


# ------------------------------------------------------------------ #
# 1. `d` 文字列のパーサ
# ------------------------------------------------------------------ #

_SEPARATORS = " \t\n\r\f,"


class _PathReader:
    """``d`` 文字列を 1 文字ずつ読む小さな読み手。

    正規表現で «数値をまとめて取る» ことをしていません。SVG の ``d`` は
    ``a1 1 0 011 1`` のように **区切りなしで詰められる**ことがあり、
    フラグと数値を区別できるのは «1 文字ずつ読む» ときだけだからです。
    """

    __slots__ = ("text", "index", "length")

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.length = len(text)

    def skip_separators(self) -> None:
        while self.index < self.length and self.text[self.index] in _SEPARATORS:
            self.index += 1

    def at_end(self) -> bool:
        self.skip_separators()
        return self.index >= self.length

    def read_number(self) -> float | None:
        """数値を 1 つ読む。読めなければ None（＝そこで打ち切り）。"""
        self.skip_separators()
        start = self.index
        text = self.text
        if self.index < self.length and text[self.index] in "+-":
            self.index += 1
        digits = 0
        while self.index < self.length and text[self.index].isdigit() and text[self.index].isascii():
            self.index += 1
            digits += 1
        if self.index < self.length and text[self.index] == ".":
            self.index += 1
            while self.index < self.length and text[self.index].isdigit() and text[self.index].isascii():
                self.index += 1
                digits += 1
        if digits == 0:
            self.index = start
            return None
        if self.index < self.length and text[self.index] in "eE":
            save = self.index
            self.index += 1
            if self.index < self.length and text[self.index] in "+-":
                self.index += 1
            exp_digits = 0
            while self.index < self.length and text[self.index].isdigit() and text[self.index].isascii():
                self.index += 1
                exp_digits += 1
            if exp_digits == 0:
                self.index = save
        try:
            value = float(text[start : self.index])
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    def read_flag(self) -> bool | None:
        """円弧のフラグを 1 つ読む。

        フラグは **«1 文字»** です。``a1 1 0 011 1`` のように区切りなしで
        詰められることがあるので、数値として読んではいけません
        （``011`` を 11 と読んでしまいます）。
        """
        self.skip_separators()
        if self.index >= self.length:
            return None
        ch = self.text[self.index]
        if ch in "01":
            self.index += 1
            return ch == "1"
        return None


def parse_path_data(d: str, *, max_segments: int = DEFAULT_MAX_SEGMENTS, **_ignored) -> dict[str, Any]:
    """SVG の ``d`` 文字列を命令列に直す。

    返る命令は **3 種類だけ**です。後段を単純にするため、``H`` ``V`` ``Q`` ``T``
    ``S`` ``A`` はすべて ``L`` か ``C`` に潰してしまいます。

    - ``{"op": "M", "values": [x, y]}`` サブパスの開始
    - ``{"op": "L", "values": [x, y]}`` 直線
    - ``{"op": "C", "values": [c1x, c1y, c2x, c2y, x, y]}`` 3 次ベジェ
    - ``{"op": "Z"}`` 閉じる

    :returns: ``{"segments": [...], "truncated": bool, "invalid": bool}``
    """
    reader = _PathReader(d if isinstance(d, str) else "")
    segments: list[Segment] = []
    state = {"truncated": False, "invalid": False}

    current = [0.0, 0.0]
    start = [0.0, 0.0]
    last_cubic: list[float] | None = None
    last_quad: list[float] | None = None

    def push(segment: Segment) -> bool:
        if len(segments) >= max_segments:
            state["truncated"] = True
            return False
        segments.append(segment)
        return True

    def move_to(x: float, y: float) -> bool:
        current[0] = start[0] = x
        current[1] = start[1] = y
        return push({"op": "M", "values": [x, y]})

    def line_to(x: float, y: float) -> bool:
        ok = push({"op": "L", "values": [x, y]})
        current[0] = x
        current[1] = y
        return ok

    def cubic_to(c1x, c1y, c2x, c2y, x, y) -> bool:
        ok = push({"op": "C", "values": [c1x, c1y, c2x, c2y, x, y]})
        current[0] = x
        current[1] = y
        return ok

    def quad_to(cx, cy, x, y) -> bool:
        # 2 次ベジェは 3 次に «厳密に» 直せます（制御点を 2/3 だけ寄せる）。
        return cubic_to(
            current[0] + (2 / 3) * (cx - current[0]),
            current[1] + (2 / 3) * (cy - current[1]),
            x + (2 / 3) * (cx - x),
            y + (2 / 3) * (cy - y),
            x,
            y,
        )

    command: str | None = None
    started = False

    while not reader.at_end():
        ch = reader.text[reader.index]
        if ch.isascii() and ch.isalpha():
            command = ch
            reader.index += 1
        elif command is None:
            # 先頭がコマンドでない＝壊れた `d`。読める分だけ返します。
            state["invalid"] = True
            break
        elif command in ("Z", "z"):
            # `Z` に引数はありません。数字が続いていたら壊れています。
            state["invalid"] = True
            break
        elif command == "M":
            # «繰り返し引数»: `M` の 2 組目以降は暗黙の `L` です（`m` なら `l`）。
            command = "L"
        elif command == "m":
            command = "l"

        relative = command.islower()
        upper = command.upper()
        base_x = current[0] if relative else 0.0
        base_y = current[1] if relative else 0.0
        ok = True

        if upper == "M":
            x = reader.read_number()
            y = reader.read_number()
            if x is None or y is None:
                state["invalid"] = True
                break
            ok = move_to(base_x + x, base_y + y)
            started = True
            last_cubic = None
            last_quad = None
        elif upper == "L":
            x = reader.read_number()
            y = reader.read_number()
            if x is None or y is None:
                state["invalid"] = True
                break
            if not started:
                ok = move_to(base_x + x, base_y + y)
                started = True
            else:
                ok = line_to(base_x + x, base_y + y)
            last_cubic = None
            last_quad = None
        elif upper == "H":
            x = reader.read_number()
            if x is None:
                state["invalid"] = True
                break
            ok = line_to(base_x + x, current[1])
            last_cubic = None
            last_quad = None
        elif upper == "V":
            y = reader.read_number()
            if y is None:
                state["invalid"] = True
                break
            ok = line_to(current[0], base_y + y)
            last_cubic = None
            last_quad = None
        elif upper == "C":
            values = [reader.read_number() for _ in range(6)]
            if any(v is None for v in values):
                state["invalid"] = True
                break
            c2x = base_x + values[2]
            c2y = base_y + values[3]
            ok = cubic_to(base_x + values[0], base_y + values[1], c2x, c2y, base_x + values[4], base_y + values[5])
            last_cubic = [c2x, c2y]
            last_quad = None
        elif upper == "S":
            values = [reader.read_number() for _ in range(4)]
            if any(v is None for v in values):
                state["invalid"] = True
                break
            # 直前が C/S なら制御点を «現在点で鏡映»、そうでなければ現在点そのもの。
            c1x = current[0] if last_cubic is None else current[0] * 2 - last_cubic[0]
            c1y = current[1] if last_cubic is None else current[1] * 2 - last_cubic[1]
            c2x = base_x + values[0]
            c2y = base_y + values[1]
            ok = cubic_to(c1x, c1y, c2x, c2y, base_x + values[2], base_y + values[3])
            last_cubic = [c2x, c2y]
            last_quad = None
        elif upper == "Q":
            values = [reader.read_number() for _ in range(4)]
            if any(v is None for v in values):
                state["invalid"] = True
                break
            cx = base_x + values[0]
            cy = base_y + values[1]
            ok = quad_to(cx, cy, base_x + values[2], base_y + values[3])
            last_quad = [cx, cy]
            last_cubic = None
        elif upper == "T":
            x = reader.read_number()
            y = reader.read_number()
            if x is None or y is None:
                state["invalid"] = True
                break
            cx = current[0] if last_quad is None else current[0] * 2 - last_quad[0]
            cy = current[1] if last_quad is None else current[1] * 2 - last_quad[1]
            ok = quad_to(cx, cy, base_x + x, base_y + y)
            last_quad = [cx, cy]
            last_cubic = None
        elif upper == "A":
            rx = reader.read_number()
            ry = reader.read_number()
            rotation = reader.read_number()
            large_arc = reader.read_flag()
            sweep = reader.read_flag()
            x = reader.read_number()
            y = reader.read_number()
            if None in (rx, ry, rotation, large_arc, sweep, x, y):
                state["invalid"] = True
                break
            end_x = base_x + x
            end_y = base_y + y
            curves = arc_to_cubics(current[0], current[1], rx, ry, rotation, large_arc, sweep, end_x, end_y)
            if not curves:
                ok = line_to(end_x, end_y)
            else:
                for curve in curves:
                    ok = cubic_to(*curve)
                    if not ok:
                        break
            last_cubic = None
            last_quad = None
        elif upper == "Z":
            ok = push({"op": "Z"})
            current[0] = start[0]
            current[1] = start[1]
            last_cubic = None
            last_quad = None
        else:
            # 未知のコマンド文字。ここから先は信用できないので打ち切ります。
            state["invalid"] = True
            break

        if not ok:
            break

    return {"segments": segments, "truncated": state["truncated"], "invalid": state["invalid"]}


def arc_to_cubics(
    x0: float, y0: float, rx_in: float, ry_in: float,
    rotation_deg: float, large_arc: bool, sweep: bool, x1: float, y1: float,
) -> list[list[float]]:
    """円弧（``A``）を 3 次ベジェの列に直す。

    手順は SVG 仕様の付録 F.6.5（端点表現 → 中心表現）そのままです。
    90 度を超える弧は «そのままでは近似できない» ので分割します。

    :returns: ``[c1x, c1y, c2x, c2y, x, y]`` の配列
    """
    rx = abs(rx_in)
    ry = abs(ry_in)
    # 半径 0、あるいは始点と終点が同じ弧は «直線» です（仕様どおり）。
    if rx < 1e-12 or ry < 1e-12:
        return []
    if abs(x1 - x0) < 1e-12 and abs(y1 - y0) < 1e-12:
        return []

    phi = rotation_deg * math.pi / 180
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
    cxp = coefficient * (rx * yp) / ry
    cyp = coefficient * -(ry * xp) / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x0 + x1) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y0 + y1) / 2

    start_x = (xp - cxp) / rx
    start_y = (yp - cyp) / ry
    end_x = (-xp - cxp) / rx
    end_y = (-yp - cyp) / ry

    theta1 = math.atan2(start_y, start_x)
    delta = math.atan2(end_y, end_x) - theta1
    if not sweep and delta > 0:
        delta -= _TAU
    elif sweep and delta < 0:
        delta += _TAU

    count = max(1, math.ceil(abs(delta) / (math.pi / 2)))
    step = delta / count
    # 4/3 tan(θ/4) は «円弧を 3 次ベジェで近似する» ときの標準の係数です。
    k = (4 / 3) * math.tan(step / 4)

    def point_at(theta: float) -> tuple[float, float]:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return (cx + rx * cos_t * cos_phi - ry * sin_t * sin_phi, cy + rx * cos_t * sin_phi + ry * sin_t * cos_phi)

    def tangent_at(theta: float) -> tuple[float, float]:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return (-rx * sin_t * cos_phi - ry * cos_t * sin_phi, -rx * sin_t * sin_phi + ry * cos_t * cos_phi)

    curves: list[list[float]] = []
    for i in range(count):
        t1 = theta1 + step * i
        t2 = t1 + step
        p1 = point_at(t1)
        p2 = point_at(t2)
        d1 = tangent_at(t1)
        d2 = tangent_at(t2)
        curves.append([p1[0] + k * d1[0], p1[1] + k * d1[1], p2[0] - k * d2[0], p2[1] - k * d2[1], p2[0], p2[1]])
    # 端点は «指定された値» に戻します。三角関数の誤差で継ぎ目が開くのを防ぎます。
    curves[-1][4] = x1
    curves[-1][5] = y1
    return curves


# ------------------------------------------------------------------ #
# 2. 命令列 → 折れ線（サブパス）
# ------------------------------------------------------------------ #


def flatten_segments(
    segments: Sequence[Segment] | None,
    *,
    transform: Sequence[float] | None = None,
    tolerance: float = 0.25,
    **_ignored,
) -> list[Subpath]:
    """命令列を折れ線に落とす。

    :param transform: ``[a, b, c, d, e, f]``（SVG の matrix と同じ並び）。
        **«折れ線にする前» に掛ける**ので、拡大しても曲線が角張りません。
    :returns: ``[{"points": [x0, y0, x1, y1, ...], "closed": bool}, ...]``
    """
    if transform is not None:
        m = transform

        def apply(x: float, y: float) -> tuple[float, float]:
            return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
    else:

        def apply(x: float, y: float) -> tuple[float, float]:
            return (x, y)

    subpaths: list[Subpath] = []
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

    for segment in segments or ():
        op = segment.get("op")
        if op == "M":
            flush()
            x, y = apply(segment["values"][0], segment["values"][1])
            points = [x, y]
            cursor_x, cursor_y = x, y
        elif op == "L":
            if points is None:
                continue
            x, y = apply(segment["values"][0], segment["values"][1])
            points.extend((x, y))
            cursor_x, cursor_y = x, y
        elif op == "C":
            if points is None:
                continue
            c1x, c1y = apply(segment["values"][0], segment["values"][1])
            c2x, c2y = apply(segment["values"][2], segment["values"][3])
            x, y = apply(segment["values"][4], segment["values"][5])
            _flatten_cubic_into(points, cursor_x, cursor_y, c1x, c1y, c2x, c2y, x, y, tolerance)
            cursor_x, cursor_y = x, y
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


def _flatten_cubic_into(out, x0, y0, c1x, c1y, c2x, c2y, x1, y1, tolerance) -> None:
    """3 次ベジェを折れ線にする。

    分割数は制御多角形の «長さ» から決めるので、**同じ入力からは必ず同じ
    点列が出ます**（決定性）。曲率から適応的に決める方式にすると、浮動小数点の
    わずかな差で分割数が変わり、別の機械では違う絵になります。
    """
    distance = (
        math.hypot(c1x - x0, c1y - y0) + math.hypot(c2x - c1x, c2y - c1y) + math.hypot(x1 - c2x, y1 - c2y)
    )
    steps = min(96, max(3, math.ceil(math.sqrt(distance / max(1e-6, tolerance)) * 1.5)))
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        out.append(mt**3 * x0 + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t**3 * x1)
        out.append(mt**3 * y0 + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t**3 * y1)


def path_to_subpaths(d: str, **options) -> list[Subpath]:
    """``d`` 文字列を折れ線に落とすところまでを一息でやる入口。"""
    parsed = parse_path_data(d, **options)
    return flatten_segments(parsed["segments"], **options)


def subpaths_bounds(subpaths: Sequence[Subpath] | None) -> dict[str, float]:
    """サブパス群を囲む矩形。``width`` / ``height`` は **0 になりません**

    （0 だと «収める» ときの割り算で壊れるためです）。
    """
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for subpath in subpaths or ():
        points = subpath["points"] if isinstance(subpath, dict) else subpath
        for i in range(0, len(points), 2):
            min_x = min(min_x, points[i])
            max_x = max(max_x, points[i])
            min_y = min(min_y, points[i + 1])
            max_y = max(max_y, points[i + 1])
    if min_x == math.inf:
        return {"minX": 0, "minY": 0, "maxX": 1, "maxY": 1, "width": 1, "height": 1}
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": max(1e-6, max_x - min_x),
        "height": max(1e-6, max_y - min_y),
    }


# ------------------------------------------------------------------ #
# 3. トリムパス
# ------------------------------------------------------------------ #


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_trim_active(trim: dict | None) -> bool:
    """トリムが «何かを削る» 指定かどうか。既定値のままなら触らずに返したいので使います。"""
    if not isinstance(trim, dict):
        return False
    if trim.get("enabled") is False:
        return False
    start = trim["start"] if _finite(trim.get("start")) else 0
    end = trim["end"] if _finite(trim.get("end")) else 1
    offset = trim["offset"] if _finite(trim.get("offset")) else 0
    return start > 1e-9 or end < 1 - 1e-9 or abs(offset) > 1e-9


def trim_subpaths(subpaths: Sequence[Subpath] | None, trim: dict | None) -> list[Subpath]:
    """サブパスを «長さの割合» で切り出す（トリムパス）。

    ``start`` / ``end`` は 0〜1、``offset`` は «開始位置をずらす量»（同じく 0〜1）です。
    閉じたパスでは ``offset`` は «一周して» 回り込みます。開いたパスでは
    はみ出した分は落とします（一周する «先» がないからです）。

    ``mode`` は 2 つあります。

    - ``"each"``（既定） サブパスごとに同じ割合で切る。全部が同時に伸びます
    - ``"sequential"`` 全サブパスを 1 本の長さとみなして順に切る。1 つずつ描かれます

    切った結果は **«開いた折れ線»** です。線が描かれていく演出が目的なので、
    閉じ直して塗ってしまうと «途中経過» が見えなくなるためです。
    """
    # 何も削らない指定なら **受け取ったものをそのまま返します**（複製しません）。
    # トリムは既定値のまま置かれることが多く、フレームごとに折れ線を丸ごと
    # 複製すると、使っていない機能のためにメモリ帯域を食うことになります。
    if not trim or not is_trim_active(trim):
        return subpaths if subpaths is not None else []
    items = list(subpaths)
    raw_start = trim["start"] if _finite(trim.get("start")) else 0.0
    raw_end = trim["end"] if _finite(trim.get("end")) else 1.0
    offset = trim["offset"] if _finite(trim.get("offset")) else 0.0
    frm = min(raw_start, raw_end) + offset
    to = max(raw_start, raw_end) + offset
    if to - frm <= 1e-9:
        return []

    if trim.get("mode") == "sequential":
        lengths = [_subpath_length(s) for s in items]
        total = sum(lengths)
        if total <= 0:
            return []
        out: list[Subpath] = []
        accumulated = 0.0
        for i, item in enumerate(items):
            share = lengths[i]
            if share <= 0:
                accumulated += share
                continue
            # 全体の割合を «このサブパスの中での割合» に読み替えます。
            local_from = (frm * total - accumulated) / share
            local_to = (to * total - accumulated) / share
            # 順番に描く指定では回り込ませません。回り込むと «最後のサブパスが
            # 最初に戻ってくる» ことになり、順番という指定と矛盾するからです。
            out.extend(_slice_subpath(item, local_from, local_to, wrap=False))
            accumulated += share
        return out

    out = []
    for subpath in items:
        out.extend(_slice_subpath(subpath, frm, to))
    return out


def _subpath_length(subpath: Subpath) -> float:
    """折れ線の全長。閉じているときは «最後から最初へ» の 1 辺も数えます。"""
    points = subpath["points"] if isinstance(subpath, dict) else subpath
    total = 0.0
    for i in range(2, len(points), 2):
        total += math.hypot(points[i] - points[i - 2], points[i + 1] - points[i - 1])
    if isinstance(subpath, dict) and subpath.get("closed") and len(points) >= 4:
        total += math.hypot(points[0] - points[-2], points[1] - points[-1])
    return total


def _slice_subpath(subpath: Subpath, frm: float, to: float, wrap: bool = True) -> list[Subpath]:
    """1 本のサブパスを ``[frm, to]``（0〜1）で切る。

    閉じたパスは回り込むので、結果が 2 本に分かれることがあります。
    """
    points = subpath["points"] if isinstance(subpath, dict) else subpath
    if len(points) < 4:
        return []
    is_closed = isinstance(subpath, dict) and subpath.get("closed") is True
    full = (to - frm >= 1 - 1e-9) if (is_closed and wrap) else (frm <= 1e-9 and to >= 1 - 1e-9)
    if full:
        return [{"points": list(points), "closed": is_closed}]

    if is_closed and wrap:
        # 回り込ませて [0, 1) に収めます。またぐときだけ 2 本に割ります。
        span = min(1.0, to - frm)
        head = math.fmod(frm, 1.0)
        if head < 0:
            head += 1
        tail = head + span
        if tail <= 1 + 1e-9:
            return _wrap_slices(subpath, [(head, min(1.0, tail))])
        return _wrap_slices(subpath, [(head, 1.0), (0.0, tail - 1)])

    low = max(0.0, frm)
    high = min(1.0, to)
    if high - low <= 1e-9:
        return []
    return _wrap_slices(subpath, [(low, high)])


def _wrap_slices(subpath: Subpath, ranges: Sequence[tuple[float, float]]) -> list[Subpath]:
    out: list[Subpath] = []
    for a, b in ranges:
        piece = _extract_range(subpath, a, b)
        if piece and len(piece) >= 4:
            out.append({"points": piece, "closed": False})
    return out


def _extract_range(subpath: Subpath, a: float, b: float) -> list[float] | None:
    """累積長で ``[a, b]``（0〜1）の区間を切り出す。端点は線形補間で作ります。"""
    source = subpath["points"] if isinstance(subpath, dict) else subpath
    # 閉じたパスは «最初の点をもう一度» 足して 1 本の折れ線として扱います。
    points = list(source) + [source[0], source[1]] if (isinstance(subpath, dict) and subpath.get("closed")) else list(source)
    count = len(points) // 2
    if count < 2:
        return None

    cumulative = [0.0] * count
    for i in range(1, count):
        cumulative[i] = cumulative[i - 1] + math.hypot(
            points[i * 2] - points[(i - 1) * 2], points[i * 2 + 1] - points[(i - 1) * 2 + 1]
        )
    total = cumulative[count - 1]
    if total <= 0:
        return None

    from_distance = a * total
    to_distance = b * total

    def point_at(distance: float) -> tuple[float, float]:
        for i in range(1, count):
            if distance <= cumulative[i] or i == count - 1:
                span = cumulative[i] - cumulative[i - 1]
                t = 0.0 if span <= 0 else min(1.0, max(0.0, (distance - cumulative[i - 1]) / span))
                return (
                    points[(i - 1) * 2] + (points[i * 2] - points[(i - 1) * 2]) * t,
                    points[(i - 1) * 2 + 1] + (points[i * 2 + 1] - points[(i - 1) * 2 + 1]) * t,
                )
        return (points[(count - 1) * 2], points[(count - 1) * 2 + 1])

    head = point_at(from_distance)
    out = [head[0], head[1]]
    for i in range(1, count):
        if cumulative[i] > from_distance + 1e-9 and cumulative[i] < to_distance - 1e-9:
            out.extend((points[i * 2], points[i * 2 + 1]))
    tail = point_at(to_distance)
    out.extend(tail)
    return out


# ------------------------------------------------------------------ #
# 4. `.svg` からの取り込み（安全な範囲だけ）
# ------------------------------------------------------------------ #

_TAG_RE = re.compile(r"""<(/)?([A-Za-z_][\w.:-]*)((?:"[^"]*"|'[^']*'|[^>])*?)(/)?>""")
_ATTR_RE = re.compile(r"""([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")
_NUMBER_RE = re.compile(r"-?[\d.]+(?:e[+-]?\d+)?", re.IGNORECASE)
_TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")


def extract_svg_shapes(
    source: str,
    *,
    max_bytes: int | None = None,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    tolerance: float = 0.25,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    **_ignored,
) -> dict[str, Any]:
    """SVG の中身から «形» だけを取り出す。

    **扱うもの**（これだけです）

    - ``<path d="...">``
    - ``<rect>`` ``<circle>`` ``<ellipse>`` ``<line>`` ``<polyline>`` ``<polygon>``
    - それらの ``transform``（matrix / translate / scale / rotate / skewX / skewY）
    - ``<svg>`` の ``viewBox`` / ``width`` / ``height``

    **無視するもの**

    - ``<script>`` ``<style>`` ``<filter>`` ``<image>`` ``<use>`` ``<text>``
      ``<foreignObject>`` — 実行・外部参照・フォントが絡むため、中身ごと飛ばします
    - ``<defs>`` ``<clipPath>`` ``<mask>`` ``<marker>`` ``<symbol>`` ``<pattern>``
      — «定義» であって描かれるものではないため
    - ``fill`` ``stroke`` ``opacity`` などの見た目、``id`` ``class``、``on*=``
    - ``href`` / ``xlink:href`` / ``url(...)`` — **たどりません**

    取りに行く処理が 1 つも無いので、SVG がどこから来ても «数字を読むだけ» で済みます。
    """
    limit = DEFAULT_SVG_MAX_BYTES if max_bytes is None else max_bytes
    text = source if isinstance(source, str) else ""
    # 文字数ではなく **«バイト数»** で見ます。日本語のコメントが入った SVG でも
    # ファイルサイズと同じ尺度になるからです。
    byte_length = len(text.encode("utf-8"))
    if byte_length > limit:
        raise ValueError(f"SVG が大きすぎます（{byte_length / 1024:.0f} KB > {limit / 1024:.0f} KB）")

    # コメント・CDATA・DOCTYPE・処理命令は «先に» 落とします。中に `<path>` に
    # 見える文字列が入っていても拾わないようにするためです。
    cleaned = re.sub(r"<!--[\s\S]*?-->", "", text)
    cleaned = re.sub(r"<!\[CDATA\[[\s\S]*?\]\]>", "", cleaned)
    cleaned = re.sub(r"<\?[\s\S]*?\?>", "", cleaned)
    cleaned = re.sub(r"<!DOCTYPE[^>\[]*(\[[\s\S]*?\])?[^>]*>", "", cleaned, flags=re.IGNORECASE)

    stack: list[Sequence[float]] = [identity_matrix()]
    skip_depth = 0
    view_box: list[float] | None = None
    width = 0.0
    height = 0.0
    subpaths: list[Subpath] = []
    stats = {"paths": 0, "shapes": 0, "skipped": 0, "truncated": False, "invalid": False}

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
        local = multiply(parent, parse_transform(attributes["transform"])) if "transform" in attributes else parent

        if name == "svg":
            if view_box is None and attributes.get("viewBox"):
                numbers = [float(n) for n in _NUMBER_RE.findall(attributes["viewBox"]) if _is_number(n)]
                if len(numbers) == 4:
                    view_box = numbers
            if not width:
                width = _parse_length(attributes.get("width"))
            if not height:
                height = _parse_length(attributes.get("height"))

        d = _element_to_path_data(name, attributes)
        if d:
            if stats["paths"] + stats["shapes"] >= max_elements:
                stats["truncated"] = True
                break
            if name == "path":
                stats["paths"] += 1
            else:
                stats["shapes"] += 1
            parsed = parse_path_data(d, max_segments=max_segments)
            if parsed["truncated"]:
                stats["truncated"] = True
            if parsed["invalid"]:
                stats["invalid"] = True
            subpaths.extend(flatten_segments(parsed["segments"], transform=local, tolerance=tolerance))

        if not self_closing:
            stack.append(local)

    if view_box is None and width > 0 and height > 0:
        view_box = [0.0, 0.0, width, height]
    if not width or not height:
        bounds = subpaths_bounds(subpaths)
        width = width or (view_box[2] if view_box else bounds["width"])
        height = height or (view_box[3] if view_box else bounds["height"])
    return {"subpaths": subpaths, "viewBox": view_box, "width": width, "height": height, "stats": stats}


def _local_name(name: str) -> str:
    """``xlink:href`` のような接頭辞を落として要素名を小文字で返す。"""
    colon = name.find(":")
    return (name[colon + 1 :] if colon >= 0 else name).lower()


def _parse_attributes(text: str) -> dict[str, str]:
    """属性を読む。``on*=``（イベント）と ``href`` は **名前ごと捨てます**。

    使わない値をわざわざ持ち歩かないのが一番安全だからです。
    """
    out: dict[str, str] = {}
    for match in _ATTR_RE.finditer(text):
        raw_key = match.group(1)
        if raw_key[:2].lower() == "on":
            continue
        key = _local_name(raw_key)
        if key == "href":
            continue
        value = match.group(2) or match.group(3) or match.group(4) or ""
        out["viewBox" if key == "viewbox" else key] = _decode_entities(value)
    return out


def _decode_entities(value: str) -> str:
    """よく使う実体参照だけ戻す。**外部実体は展開しません**（それが XXE の入口です）。"""
    value = value.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
    value = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), value)
    return value.replace("&amp;", "&")


def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _parse_length(value: str | None) -> float:
    """``width="100px"`` のような単位付きの長さを数値にする（``%`` は 0 扱い）。"""
    if not value:
        return 0.0
    if re.search(r"%\s*$", value):
        return 0.0
    match = re.match(r"\s*[-+]?[\d.]+(?:[eE][-+]?\d+)?", value)
    if not match:
        return 0.0
    try:
        number = float(match.group(0))
    except ValueError:
        return 0.0
    return number if math.isfinite(number) else 0.0


def _element_to_path_data(name: str, attributes: dict[str, str]) -> str | None:
    """基本図形を ``d`` 文字列に直す。

    ここを通すと、後段は **«パスだけ»** を相手にすれば済みます。図形ごとに
    別の折れ線化を書くと、``transform`` の扱いが 6 か所に散ります。
    """

    def number(key: str, fallback: float = 0.0) -> float:
        raw = attributes.get(key)
        if raw is None:
            return fallback
        match = re.match(r"\s*[-+]?[\d.]+(?:[eE][-+]?\d+)?", raw)
        if not match:
            return fallback
        try:
            value = float(match.group(0))
        except ValueError:
            return fallback
        return value if math.isfinite(value) else fallback

    if name == "path":
        return attributes.get("d") or None
    if name == "rect":
        x = number("x")
        y = number("y")
        w = number("width")
        h = number("height")
        if w <= 0 or h <= 0:
            return None
        rx = number("rx", math.nan) if "rx" in attributes else math.nan
        ry = number("ry", math.nan) if "ry" in attributes else math.nan
        if math.isnan(rx) and math.isnan(ry):
            return f"M{_n(x)} {_n(y)}H{_n(x + w)}V{_n(y + h)}H{_n(x)}Z"
        if math.isnan(rx):
            rx = ry
        if math.isnan(ry):
            ry = rx
        rx = min(max(0.0, rx), w / 2)
        ry = min(max(0.0, ry), h / 2)
        if rx == 0 or ry == 0:
            return f"M{_n(x)} {_n(y)}H{_n(x + w)}V{_n(y + h)}H{_n(x)}Z"
        return (
            f"M{_n(x + rx)} {_n(y)}H{_n(x + w - rx)}A{_n(rx)} {_n(ry)} 0 0 1 {_n(x + w)} {_n(y + ry)}"
            f"V{_n(y + h - ry)}A{_n(rx)} {_n(ry)} 0 0 1 {_n(x + w - rx)} {_n(y + h)}"
            f"H{_n(x + rx)}A{_n(rx)} {_n(ry)} 0 0 1 {_n(x)} {_n(y + h - ry)}"
            f"V{_n(y + ry)}A{_n(rx)} {_n(ry)} 0 0 1 {_n(x + rx)} {_n(y)}Z"
        )
    if name == "circle":
        r = number("r")
        if r <= 0:
            return None
        cx = number("cx")
        cy = number("cy")
        return f"M{_n(cx - r)} {_n(cy)}A{_n(r)} {_n(r)} 0 1 0 {_n(cx + r)} {_n(cy)}A{_n(r)} {_n(r)} 0 1 0 {_n(cx - r)} {_n(cy)}Z"
    if name == "ellipse":
        rx = number("rx")
        ry = number("ry")
        if rx <= 0 or ry <= 0:
            return None
        cx = number("cx")
        cy = number("cy")
        return (
            f"M{_n(cx - rx)} {_n(cy)}A{_n(rx)} {_n(ry)} 0 1 0 {_n(cx + rx)} {_n(cy)}"
            f"A{_n(rx)} {_n(ry)} 0 1 0 {_n(cx - rx)} {_n(cy)}Z"
        )
    if name == "line":
        return f"M{_n(number('x1'))} {_n(number('y1'))}L{_n(number('x2'))} {_n(number('y2'))}"
    if name in ("polyline", "polygon"):
        numbers = _NUMBER_RE.findall(attributes.get("points", ""))
        numbers = [n for n in numbers if _is_number(n)]
        if len(numbers) < 4:
            return None
        pairs = [f"{numbers[i]} {numbers[i + 1]}" for i in range(0, len(numbers) - 1, 2)]
        return "M" + "L".join(pairs) + ("Z" if name == "polygon" else "")
    return None


def _n(value: float) -> str:
    """数値を ``d`` 文字列に埋める書き方。整数は ``.0`` を付けません。

    ``M10.0 20.0`` でも読めますが、`d` を目で追うときに邪魔なのと、
    JS 版が出す文字列と揃えるためです。
    """
    if float(value).is_integer() and abs(value) < 1e16:
        return str(int(value))
    return repr(float(value))


# ------------------------------------------------------------------ #
# 5. 変換行列
# ------------------------------------------------------------------ #


def identity_matrix() -> Matrix:
    return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def multiply(a: Sequence[float], b: Sequence[float]) -> Matrix:
    """SVG の ``[a, b, c, d, e, f]`` 同士の積（``a`` を先に、``b`` を後に掛ける並び）。"""
    return [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]


def parse_transform(text: str | None) -> Matrix:
    """``transform="translate(10 20) rotate(30)"`` を 1 つの行列にまとめる。

    左から順に掛けます（SVG の規定どおり、**右のものが «先に» 点に効きます**）。
    """
    matrix = identity_matrix()
    if not isinstance(text, str):
        return matrix
    for match in _TRANSFORM_RE.finditer(text):
        name = match.group(1).lower()
        args = [float(n) for n in _NUMBER_RE.findall(match.group(2)) if _is_number(n)]
        step: Matrix | None = None
        if name == "matrix":
            if len(args) >= 6:
                step = list(args[:6])
        elif name == "translate":
            step = [1, 0, 0, 1, args[0] if len(args) > 0 else 0, args[1] if len(args) > 1 else 0]
        elif name == "scale":
            sx = args[0] if len(args) > 0 else 1
            sy = args[1] if len(args) > 1 else sx
            step = [sx, 0, 0, sy, 0, 0]
        elif name == "rotate":
            angle = (args[0] if args else 0) * math.pi / 180
            cos = math.cos(angle)
            sin = math.sin(angle)
            rotation = [cos, sin, -sin, cos, 0, 0]
            if len(args) >= 3:
                # 中心指定は «移動 → 回転 → 戻す» に展開します。
                step = multiply(multiply([1, 0, 0, 1, args[1], args[2]], rotation), [1, 0, 0, 1, -args[1], -args[2]])
            else:
                step = rotation
        elif name == "skewx":
            step = [1, 0, math.tan((args[0] if args else 0) * math.pi / 180), 1, 0, 0]
        elif name == "skewy":
            step = [1, math.tan((args[0] if args else 0) * math.pi / 180), 0, 1, 0, 0]
        if step:
            matrix = multiply(matrix, step)
    return matrix
