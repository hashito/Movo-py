"""生成レイヤー — 素材を用意せずに絵を作るレイヤー種別。

  starfield    星空（瞬き・奥行き）
  linePath     パスに沿った線（伸びる・破線・矢印）
  neonPath     パスに沿った発光（芯 ＋ グロー）
  metaball     くっついたり離れたりする液体状の形
  waterSurface 波打つ水面（下のレイヤーを映す）
  primitive3d  立方体・球体にテクスチャを貼って回す
  mesh         OBJ を読んでそのまま置く
  spotlight    円錐状に広がって減衰する照明（ボリュメトリック）
  shapeAnim    定番の図形アニメーション（8 種）

どれも `(spec, gctx)` を受け取り
`{"bitmap", "box_width", "box_height", "origin_x", "origin_y", "scale"}`
を返します。あとは通常のレイヤーと同じ経路（変形・エフェクト・合成）に載ります。
乱数はすべてシードから決まるので、同じ時刻を描けば必ず同じ絵になります。

## Movo-py での作り替え（JS 版との違い）

生成レイヤーは **毎フレーム画面 1 枚を作り直す** ので、JS 版の «二重 for で
画素を舐める» 書き方をそのまま持ってくると 1 パスあたり 720 ミリ秒かかり、
数分の MV が現実的な時間で焼けなくなります。そこで

* **メタボール** … 場の計算を «粗い格子の一括演算 ＋ 拡大» に置き換え（NumPy）
* **水面・スポットライト・星の点** … 画素ごとに分岐が要るので `@njit`
* **星の乱数** … mulberry32 の状態更新が単なる足し算なので «まとめて» 発生

としています。**式と丸めは JS 版のままです。**（`Uint8ClampedArray` への代入は
偶数丸め、`Math.round` は +∞ 方向。取り違えると 1 画素だけ 1 ずれます。
`Math.hypot` も «素直な平方根» とは最後の 1 ビットが違うので `math.hypot` を使います。）

JS 版と 1 画素ずつ突き合わせて、`primitive3d` の球以外は **完全に一致** します。
球だけ 9 万画素中 140 画素ほどが 1 だけずれますが、これは V8 の `Math.sin` と
CPython の `math.sin` が 1 ULP 違う角度があるためで、移植の取りこぼしではありません
（緯度経度の三角関数がそのまま面の法線 ＝ 陰影の明るさになるので露見します）。

依存は NumPy と Numba だけです。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from movo.animation.resolver import resolve_animated
from movo.core.bitmap import Bitmap
from movo.core.math import TAU, clamp, js_round, lerp, to_radians
from movo.expression._compat import is_nullish
from movo.renderer.effects import box_blur
from movo.renderer.effects import parse_color as parse_color_object
from movo.renderer.mesh3d import draw_floor_shadow, draw_mesh, normalize_bounds, parse_obj
from movo.renderer.plane3d import camera_basis
from movo.renderer.raster import (
    circle_contour,
    draw_bitmap,
    draw_textured_triangle,
    fill_coverage,
    parse_color,
    rasterize_contours,
    stroke_to_contours,
)

# ══════════════════════════════════════════════════════════════════
# JSON から来た値を読む小道具
# ══════════════════════════════════════════════════════════════════


def _n(value, default):
    """JS の `??`。**None のときだけ**既定値（`0` や `False` はそのまま）。

    `or` で代用すると `0` や `""` まで既定値に化けます。`opacity: 0` が
    «見えない» ではなく «既定の 1» になる、という形で必ず事故ります。
    """
    return default if is_nullish(value) else value


def _num(value, default=0.0):
    """数として読む。読めなければ既定値。

    JSON には `"12"` のような文字列が紛れ込むことがあり、JS は算術のたびに
    黙って数へ直します。同じ寛容さをここで 1 か所にまとめています。
    """
    if is_nullish(value):
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _jsmod(a: float, b: float) -> float:
    """JS の `%`（余りの符号は «割られる数» に従う）。

    Python の `%` は «割る数» の符号に従うので、時刻が負のときだけ結果が
    ずれます。ループ演出の進行度で使うため、JS と同じ `fmod` にそろえます。
    """
    return math.fmod(a, b)


def _u8_array(value: np.ndarray) -> np.ndarray:
    """`Uint8ClampedArray` への代入と同じ丸め（切り詰め → 偶数丸め）の一括版。"""
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# 折れ線の道具
# ══════════════════════════════════════════════════════════════════


def _to_points(raw) -> list[list[float]]:
    """パスの点列を `[x, y]` の並びにそろえる（`[x,y]` でも `{x,y}` でも受ける）。"""
    out: list[list[float]] = []
    for p in _n(raw, []) or []:
        if isinstance(p, dict):
            out.append([_num(p.get("x")), _num(p.get("y"))])
        elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
            out.append([_num(p[0]), _num(p[1])])
    return out


def _cumulative_lengths(points, closed: bool):
    """折れ線の累積長。区間の切り出しに使う。"""
    lengths = [0.0]
    listed = [*points, points[0]] if closed else list(points)
    total = 0.0
    for i in range(1, len(listed)):
        total += math.hypot(listed[i][0] - listed[i - 1][0], listed[i][1] - listed[i - 1][1])
        lengths.append(total)
    return lengths, total, listed


def _slice_polyline(points, closed: bool, start: float, end: float):
    """折れ線の `start`〜`end`（0〜1）の区間を切り出す。

    線が «伸びる» アニメーションはこれで作ります。
    """
    if len(points) < 2:
        return []
    lengths, total, listed = _cumulative_lengths(points, closed)
    if total <= 0:
        return []
    frm = clamp(min(start, end), 0, 1) * total
    to = clamp(max(start, end), 0, 1) * total
    if to - frm < 1e-6:
        return []

    def point_at(distance: float):
        for i in range(1, len(lengths)):
            if distance <= lengths[i] or i == len(lengths) - 1:
                span = (lengths[i] - lengths[i - 1]) or 1
                t = clamp((distance - lengths[i - 1]) / span, 0, 1)
                return [
                    lerp(listed[i - 1][0], listed[i][0], t),
                    lerp(listed[i - 1][1], listed[i][1], t),
                ]
        return listed[-1]

    out = [point_at(frm)]
    for i in range(1, len(lengths)):
        if frm < lengths[i] < to:
            out.append(listed[i])
    out.append(point_at(to))
    return out


def _dash_polyline(points, dash):
    """破線に分ける。`dash` は [線, 空き] の繰り返し。"""
    if not dash:
        return [points]
    pattern = [d for d in dash if d > 0]
    if not pattern:
        return [points]
    segments = []
    current = [points[0]]
    pattern_index = 0
    remaining = pattern[0]
    drawing = True

    for i in range(1, len(points)):
        ax = points[i - 1][0]
        ay = points[i - 1][1]
        bx = points[i][0]
        by = points[i][1]
        segment_length = math.hypot(bx - ax, by - ay)
        while segment_length > remaining:
            t = remaining / segment_length
            nx = ax + (bx - ax) * t
            ny = ay + (by - ay) * t
            if drawing:
                current.append([nx, ny])
                if len(current) >= 2:
                    segments.append(current)
                current = []
            else:
                current = [[nx, ny]]
            drawing = not drawing
            ax = nx
            ay = ny
            segment_length -= remaining
            pattern_index = (pattern_index + 1) % len(pattern)
            remaining = pattern[pattern_index]
        remaining -= segment_length
        if drawing:
            current.append([bx, by])
    if drawing and len(current) >= 2:
        segments.append(current)
    return segments


def _flatten(points) -> np.ndarray:
    """`[[x,y], ...]` を `[x,y,x,y,...]` にする（ラスタライザは平坦配列を取る）。"""
    arr = np.asarray(points, dtype=np.float64)
    return arr.reshape(-1)


def _reverse_contour(flat) -> np.ndarray:
    """平坦な輪郭の «向き» を反転する。穴を開けるときに使う（nonzero 規則）。"""
    return np.asarray(flat, dtype=np.float64).reshape(-1, 2)[::-1].reshape(-1)


def _stroke_polyline(points, thickness):
    """折れ線を太さのある帯にする（平坦配列で渡す）。"""
    if len(points) < 2:
        return []
    return stroke_to_contours(_flatten(points), thickness, False)


def _arrow_contour(points, size: float):
    """先端に矢じりを足す。"""
    if len(points) < 2 or size <= 0:
        return None
    tip = points[-1]
    previous = points[-2]
    angle = math.atan2(tip[1] - previous[1], tip[0] - previous[0])
    spread = 0.42
    return [
        tip[0],
        tip[1],
        tip[0] - math.cos(angle - spread) * size,
        tip[1] - math.sin(angle - spread) * size,
        tip[0] - math.cos(angle + spread) * size,
        tip[1] - math.sin(angle + spread) * size,
    ]


def _ray_exit_distance(ox: float, oy: float, dx: float, dy: float, width: float, height: float) -> float:
    """光源から向き (dx,dy) へ進んだとき、レイヤーの矩形から «出ていく» までの距離。

    スポットライトの `length` をピクセルで書かせると、解像度を変えたときに
    届く先がずれてしまうので、「1 でレイヤーの端まで」を基準にしています。
    光源が枠の外（`originY: -0.05` など）にあっても、前方の交点だけを見るので
    そのまま使えます。
    """
    near = -math.inf
    far = math.inf

    def slab(origin, direction, size):
        nonlocal near, far
        if abs(direction) < 1e-9:
            # 軸に平行 ＝ その軸では «外にいるなら永遠に入らない»
            if origin < 0 or origin > size:
                far = -math.inf
            return
        t0 = -origin / direction
        t1 = (size - origin) / direction
        near = max(near, min(t0, t1))
        far = min(far, max(t0, t1))

    slab(ox, dx, width)
    slab(oy, dy, height)
    if far < max(near, 0):
        return max(width, height)
    return far


# ══════════════════════════════════════════════════════════════════
# 乱数（mulberry32 の «まとめ発生»）
# ══════════════════════════════════════════════════════════════════


def _mulberry_stream(seed: int, count: int) -> np.ndarray:
    """mulberry32 を `count` 個まとめて出す。`effects.Random` と同じ数列です。

    **なぜ一括にできるか**: mulberry32 の状態更新は `a += 0x6D2B79F5` の
    足し算だけで、出力は状態だけから決まります。つまり i 番目の状態は
    `seed + (i+1) * 0x6D2B79F5 (mod 2^32)` で直に求まり、«前から順に» 回す
    必要がありません。星を 20000 個置くと `Random()` の呼び出しだけで
    12 万回になるので、ここは配列演算に落とす価値があります。
    """
    if count <= 0:
        return np.empty(0, np.float64)
    a0 = int(seed) & 0xFFFFFFFF
    if a0 == 0:
        a0 = 0x9E3779B9
    index = np.arange(1, count + 1, dtype=np.uint64)
    a = ((np.uint64(a0) + index * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    t = a
    t = (t ^ (t >> np.uint32(15))) * (t | np.uint32(1))
    t = t ^ (t + ((t ^ (t >> np.uint32(7))) * (t | np.uint32(61))))
    t = t ^ (t >> np.uint32(14))
    return t.astype(np.float64) / 4294967296.0


# ══════════════════════════════════════════════════════════════════
# 画素ごとの核（Numba）
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, inline="always")
def _u8_round(v):
    """`Uint8ClampedArray` への代入と同じ丸め（切り詰め → **偶数**丸め）。

    `math.floor(v + 0.5)`（＝ `Math.round`）で代用すると `200.5` が `201` に
    なり、JS 版と 1 ずれた画素がまだらに出ます。
    """
    if v != v:
        return np.uint8(0)
    if v <= 0.0:
        return np.uint8(0)
    if v >= 255.0:
        return np.uint8(255)
    f = math.floor(v)
    d = v - f
    if d > 0.5:
        f += 1.0
    elif d == 0.5 and (int(f) & 1) == 1:
        f += 1.0
    return np.uint8(f)


@njit(cache=True, inline="always")
def _hash_unit(i, seed):
    """JS の `hashToUnit`。32 ビットの巻き戻りまで含めて同じにします。"""
    h = np.uint32(i & 0xFFFFFFFF) ^ np.uint32(seed & 0xFFFFFFFF)
    h = np.uint32(h * np.uint32(0x27D4EB2D))
    h = np.uint32(h ^ (h >> np.uint32(15)))
    h = np.uint32(h * np.uint32(0x85EBCA6B))
    h = np.uint32(h ^ (h >> np.uint32(13)))
    return np.float64(h) / 4294967296.0


@njit(cache=True, inline="always")
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, inline="always")
def _noise2(x, y, seed):
    """決定的な 2 次元値ノイズ（-1..1）。`effects.value_noise_2d` のスカラー版。

    `fbm2D` は 3 次元ノイズを呼びますが、生成レイヤーはどこも `z = 0` なので
    格子ハッシュの第 3 項が `imul(0, ...) = 0` で消え、`zf = fade(0) = 0` で
    奥の面も使われません。**2 次元ノイズと完全に同じ値**になるため、こちらで
    代用しています。
    """
    xi = np.int64(math.floor(x))
    yi = np.int64(math.floor(y))
    xf = _fade(x - math.floor(x))
    yf = _fade(y - math.floor(y))
    gy0 = np.uint32(np.uint32(yi & 0xFFFFFFFF) * np.uint32(19349663))
    gy1 = np.uint32(np.uint32((yi + 1) & 0xFFFFFFFF) * np.uint32(19349663))
    gx0 = np.uint32(np.uint32(xi & 0xFFFFFFFF) * np.uint32(73856093))
    gx1 = np.uint32(np.uint32((xi + 1) & 0xFFFFFFFF) * np.uint32(73856093))
    v00 = _hash_unit(np.uint32(gx0 ^ gy0), seed)
    v10 = _hash_unit(np.uint32(gx1 ^ gy0), seed)
    v01 = _hash_unit(np.uint32(gx0 ^ gy1), seed)
    v11 = _hash_unit(np.uint32(gx1 ^ gy1), seed)
    top = v00 + (v10 - v00) * xf
    bottom = v01 + (v11 - v01) * xf
    return (top + (bottom - top) * yf) * 2.0 - 1.0


@njit(cache=True, inline="always")
def _fbm3(x, y, seed):
    """`fbm2D(x, y, {octaves: 3})` と同じ値（lacunarity 2・gain 0.5・type fbm）。

    格子ノイズは軸に沿った模様が出やすいので、JS 版と同じくオクターブごとに
    座標を 0.5 ラジアンだけ回しています。
    """
    cos_a = 0.8775825618903728  # cos(0.5)
    sin_a = 0.479425538604203  # sin(0.5)
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    px = x
    py = y
    for i in range(3):
        sample = _noise2(px * freq, py * freq, seed + i * 1013)
        rx = px * cos_a - py * sin_a
        py = px * sin_a + py * cos_a
        px = rx
        total += sample * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0
    if norm == 0.0:
        return 0.0
    return total / norm


@njit(cache=True)
def _k_plot_stars(data, cx, cy, radius, tint):
    """星を «小さな円» として直接置く。

    星は数が多いので全面ラスタライズは回さず、距離から被覆率を出すだけに
    しています（半径 1〜4px なのでこれで十分きれい）。重なっても飛ばないよう
    スクリーン合成です。**星ごとの外接矩形しか触らない**ので、星の数に比例した
    仕事しかしません。

    :param tint: `(星の数, 4)` の `r, g, b, a`
    """
    height = data.shape[0]
    width = data.shape[1]
    for k in range(cx.shape[0]):
        r = radius[k]
        px = cx[k]
        py = cy[k]
        ta = tint[k, 3]
        min_x = int(math.floor(px - r - 1.0))
        max_x = int(math.ceil(px + r + 1.0))
        min_y = int(math.floor(py - r - 1.0))
        max_y = int(math.ceil(py + r + 1.0))
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > width - 1:
            max_x = width - 1
        if max_y > height - 1:
            max_y = height - 1
        for y in range(min_y, max_y + 1):
            dy = y + 0.5 - py
            for x in range(min_x, max_x + 1):
                dx = x + 0.5 - px
                # JS は `Math.hypot`。素直な平方根と最後の 1 ビットが違うことがあるので、
                # 被覆率が変わらないようこちらも `hypot` にそろえます。
                distance = math.hypot(dx, dy)
                coverage = r - distance + 0.5
                if coverage < 0.0:
                    coverage = 0.0
                elif coverage > 1.0:
                    coverage = 1.0
                coverage *= ta
                if coverage <= 0.0:
                    continue
                for c in range(3):
                    base = np.float64(data[y, x, c])
                    data[y, x, c] = _u8_round(
                        255.0 - ((255.0 - base) * (255.0 - tint[k, c] * coverage)) / 255.0
                    )
                a = np.float64(data[y, x, 3])
                v = coverage * 255.0
                if v > a:
                    a = v
                data[y, x, 3] = _u8_round(a)


@njit(cache=True)
def _k_water_surface(
    data,
    horizon_y,
    wave_scale,
    wave_speed,
    amplitude,
    refraction,
    tint_r,
    tint_g,
    tint_b,
    specular,
    perspective,
    time,
    seed,
    source,
    use_source,
):
    """水面 1 枚。**JS の二重 for をそのまま `@njit` に落としたもの。**

    NumPy の一括演算でも書けますが、3 オクターブぶんの格子ハッシュで画面
    1 枚ぶんの一時配列が何本も出るうえ、下の絵を «画素ごとにずれた位置から»
    拾う処理は gather になり、まとめて書くと逆に遅くなります。ここは
    1 画素ずつ回して一時配列を作らないほうが速い場所です。
    """
    height = data.shape[0]
    width = data.shape[1]
    source_h = source.shape[0]
    source_w = source.shape[1]
    span = height - horizon_y
    if span < 1:
        span = 1
    for y in range(horizon_y, height):
        if y < 0:
            continue
        # 水平線から遠いほど手前 ＝ 波が大きく粗くなる
        depth = (y - horizon_y) / span
        detail = 1.0 + (0.25 - 1.0) * (perspective * (1.0 - depth))
        local_amplitude = amplitude * (0.15 + (1.0 - 0.15) * depth)
        reflect = 0.35 + depth * 0.5
        if reflect < 0.0:
            reflect = 0.0
        elif reflect > 1.0:
            reflect = 1.0
        mirrored = horizon_y - (y - horizon_y) * (1.0 - refraction * 0.5)
        for x in range(width):
            wave = _fbm3(
                x * wave_scale * detail,
                (y * wave_scale * detail + time * wave_speed) * 2.0,
                seed,
            )
            r = tint_r
            g = tint_g
            b = tint_b
            if use_source == 1:
                # 水平線を軸に折り返した位置を、波のぶんだけずらして拾う
                sx = int(math.floor(x + wave * local_amplitude + 0.5))
                sy = int(math.floor(mirrored + wave * local_amplitude * 0.5 + 0.5))
                if sx < 0:
                    sx = 0
                elif sx > source_w - 1:
                    sx = source_w - 1
                if sy < 0:
                    sy = 0
                elif sy > source_h - 1:
                    sy = source_h - 1
                r = tint_r + (np.float64(source[sy, sx, 0]) - tint_r) * reflect
                g = tint_g + (np.float64(source[sy, sx, 1]) - tint_g) * reflect
                b = tint_b + (np.float64(source[sy, sx, 2]) - tint_b) * reflect
            if specular > 0.0:
                # 波の «山» を白く光らせる
                peak = wave
                if peak < 0.0:
                    peak = 0.0
                highlight = peak * peak * peak * specular * 255.0
                r += highlight
                g += highlight
                b += highlight
            data[y, x, 0] = _u8_round(r)
            data[y, x, 1] = _u8_round(g)
            data[y, x, 2] = _u8_round(b)
            data[y, x, 3] = np.uint8(255)


@njit(cache=True)
def _k_spotlight(
    data,
    min_x,
    min_y,
    max_x,
    max_y,
    ox,
    oy,
    dx,
    dy,
    nx,
    ny,
    tan_spread,
    beam,
    inner,
    lightness,
    haze,
    haze_freq,
    time,
    seed,
    color_r,
    color_g,
    color_b,
    color_a,
):
    """円錐状の光 1 本。**画素ごとに «円錐の内か外か» を判定するので `@njit`。**

    円錐の外は必ず 0 なので、呼ぶ側で外接矩形に絞ってから渡します。生成レイヤーは
    毎フレーム width×height を作るため、ここを絞るかどうかで «1 フレームの重さ»
    が桁で変わります。
    """
    for y in range(min_y, max_y + 1):
        ry = y + 0.5 - oy
        for x in range(min_x, max_x + 1):
            rx = x + 0.5 - ox
            along = rx * dx + ry * dy
            if along <= 0.0 or along >= beam:
                continue
            side = rx * nx + ry * ny
            half_at = along * tan_spread
            radial = abs(side) / half_at
            if radial >= 1.0:
                continue
            # 側面のぼけ ×「length で消えきる」減衰（1.5 乗。pow より sqrt が速い）
            remain = 1.0 - along / beam
            # smoothstep(1, inner, radial)
            t = (radial - 1.0) / (inner - 1.0)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            value = t * t * (3.0 - 2.0 * t) * remain * math.sqrt(remain) * lightness
            # haze は明るさを «減らす» 方向にしか効かないので、ここで消える画素は
            # ノイズを引くまでもない。1 フレームの走査コストがこれで目に見えて減る。
            if value <= 0.002:
                continue
            if haze > 0.0:
                # 空気中の粒に散った光。ビームに沿って «筋» が流れるよう、
                # 進行方向の座標だけを時間でずらしている。
                n = _noise2(side * haze_freq, (along * 0.3) * haze_freq - time * 0.7, seed)
                value *= 1.0 - haze + haze * (0.5 + 0.5 * n)
            alpha = value
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0
            alpha *= color_a
            if alpha <= 0.002:
                continue
            data[y, x, 0] = _u8_round(color_r)
            data[y, x, 1] = _u8_round(color_g)
            data[y, x, 2] = _u8_round(color_b)
            data[y, x, 3] = _u8_round(alpha * 255.0)


# ══════════════════════════════════════════════════════════════════
# 戻り値
# ══════════════════════════════════════════════════════════════════

#: OBJ の解析結果を «中身の文字列» で覚えておく。毎フレーム parse し直さないため。
MESH_CACHE: dict[str, dict] = {}


def _result(bitmap: Bitmap, width: float, height: float, scale: float) -> dict:
    """生成レイヤーの戻り値を組み立てる（`_render_content` と同じ形）。"""
    return {
        "bitmap": bitmap,
        "box_width": width,
        "box_height": height,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "scale": scale,
    }


def _canvas(gctx) -> tuple[Bitmap, float, float, float]:
    """レイヤーの大きさから «描く紙» を用意する（どのレイヤーも先頭でこれ）。"""
    width = gctx["width"]
    height = gctx["height"]
    scale = _n(gctx.get("scale"), 1)
    return Bitmap(js_round(width * scale), js_round(height * scale)), width, height, scale


# ══════════════════════════════════════════════════════════════════
# starfield
# ══════════════════════════════════════════════════════════════════


def _star_alpha(d0, d4, layers, twinkle, twinkle_speed, time):
    """星の明るさ。**配列でもスカラーでも同じ式で動くよう `np.*` で書いています。**

    JS はこの値が 0.01 以下だと «tintShift を引く前に» 星を捨てるので、
    乱数の読み位置がそこで 1 つずれます。速い経路と正確な経路の両方で
    同じ式が要るため、ここに切り出しました。
    """
    layer = np.floor(d0 * layers)
    depth = layer / (layers - 1) if layers > 1 else layer * 0 + 1.0
    phase = d4 * TAU
    # 明滅は星ごとに位相をずらす
    flicker = 1 - twinkle * 0.5 * (1 - np.sin(time * twinkle_speed * TAU * 0.25 + phase))
    return np.clip((0.35 + 0.65 * depth) * flicker, 0, 1), depth


def _starfield(spec: dict, gctx: dict) -> dict:
    """星空。奥行きの違う層を重ねると、カメラを振ったときに視差が出ます。"""
    bitmap, width, height, scale = _canvas(gctx)
    count = int(clamp(js_round(_num(spec.get("count"), 400)), 0, 20000))
    if count == 0:
        return _result(bitmap, width, height, scale)

    layers = int(clamp(js_round(_num(spec.get("layers"), 3)), 1, 8))
    parallax = _num(spec.get("parallax"), 0.4)
    size_min = _num(spec.get("sizeMin"), 1)
    size_max = _num(spec.get("sizeMax"), 3.5)
    base_color = parse_color(_n(spec.get("color"), "#ffffff"))
    variation = clamp(_num(spec.get("colorVariation"), 0.3), 0, 1)
    twinkle = clamp(_num(spec.get("twinkle"), 0.6), 0, 1)
    twinkle_speed = _num(spec.get("twinkleSpeed"), 1.4)
    scroll_x = _num(spec.get("scrollX"), 0)
    scroll_y = _num(spec.get("scrollY"), 0)
    time = _num(gctx.get("time"), 0)
    seed = js_round(_num(spec.get("seed"), 9))

    # 1 星あたり «6 個» の乱数（層・x・y・半径・位相・色ずれ）を使う前提で
    # まとめて発生させる。捨てられる星があると読み位置がずれるので、下で確かめる。
    draws = _mulberry_stream(seed, count * 6).reshape(count, 6)
    alpha, depth = _star_alpha(draws[:, 0], draws[:, 4], layers, twinkle, twinkle_speed, time)
    if bool(np.any(alpha <= 0.01)):
        # «捨てられる星» が 1 つでもあると、そこから先の乱数の読み位置が 1 つ
        # ずれます（JS は色ずれを引く前に continue するため）。珍しい場合なので、
        # ここだけ星ごとに読み位置を進め直して JS と同じ数列に合わせます。
        draws = _starfield_realign(draws, count, layers, twinkle, twinkle_speed, time)
        alpha, depth = _star_alpha(draws[:, 0], draws[:, 4], layers, twinkle, twinkle_speed, time)

    keep = alpha > 0.01
    if not bool(np.any(keep)):
        return _result(bitmap, width, height, scale)

    # 手前の層ほど大きく、速く流れる
    speed = 1 - parallax + parallax * (depth + 0.2)
    base_x = draws[:, 1] * width
    base_y = draws[:, 2] * height
    # JS の `((v % w) + w) % w`。fmod → + w → floor 剰余の順まで合わせる。
    x = np.mod(np.fmod(base_x + scroll_x * width * time * speed, width) + width, width)
    y = np.mod(np.fmod(base_y + scroll_y * height * time * speed, height) + height, height)
    radius = size_min + (size_max - size_min) * (draws[:, 3] * (0.4 + 0.6 * depth))
    tint_shift = (draws[:, 5] - 0.5) * 2 * variation

    tint = np.empty((count, 4), np.float64)
    tint[:, 0] = np.clip(base_color[0] * (1 - np.maximum(0, -tint_shift) * 0.4), 0, 255)
    tint[:, 1] = np.clip(base_color[1] * (1 - np.abs(tint_shift) * 0.15), 0, 255)
    tint[:, 2] = np.clip(base_color[2] * (1 - np.maximum(0, tint_shift) * 0.3), 0, 255)
    tint[:, 3] = alpha

    _k_plot_stars(
        bitmap.data,
        np.ascontiguousarray(x[keep] * scale),
        np.ascontiguousarray(y[keep] * scale),
        np.ascontiguousarray(radius[keep] * scale),
        np.ascontiguousarray(tint[keep]),
    )
    return _result(bitmap, width, height, scale)


def _starfield_realign(draws, count, layers, twinkle, twinkle_speed, time):
    """捨てられる星があるときだけ、乱数の読み位置を星ごとに進め直す。

    速い経路（1 星 6 個ずつ）が使えないときの «正確な» 経路です。ここに来るのは
    `twinkle` がほぼ 1 で、なおかつ位相がちょうど合った星があるときだけなので、
    星ごとの Python ループでも実用上の重さになりません。
    """
    values = draws.reshape(-1)
    out = np.zeros((count, 6), np.float64)
    cursor = 0
    for k in range(count):
        if cursor + 5 > values.size:
            break
        block = values[cursor : cursor + 5]
        out[k, :5] = block
        alpha, _ = _star_alpha(block[0], block[4], layers, twinkle, twinkle_speed, time)
        if alpha <= 0.01:
            cursor += 5
            continue
        if cursor + 5 < values.size:
            out[k, 5] = values[cursor + 5]
        cursor += 6
    return out


# ══════════════════════════════════════════════════════════════════
# linePath / neonPath
# ══════════════════════════════════════════════════════════════════


def _line_path(spec: dict, gctx: dict) -> dict:
    """パスに沿った線。`end` を動かすと線が伸びます。破線・矢印・グローに対応。"""
    bitmap, width, height, scale = _canvas(gctx)
    points = _to_points(spec.get("points"))
    if len(points) < 2:
        return _result(bitmap, width, height, scale)

    closed = spec.get("closed") is True
    thickness = max(0.5, _num(spec.get("thickness"), 6))
    color = parse_color(_n(spec.get("color"), "#ffffff"))
    start = _num(spec.get("start"), 0)
    end = _num(spec.get("end"), 1)
    visible = _slice_polyline(points, closed, start, end)
    if len(visible) < 2:
        return _result(bitmap, width, height, scale)

    scaled = [[p[0] * scale, p[1] * scale] for p in visible]
    dash = spec.get("dash")
    if isinstance(dash, (list, tuple)) and len(dash) > 0:
        pieces = _dash_polyline(scaled, [_num(d) * scale for d in dash])
    else:
        pieces = [scaled]

    contours: list = []
    for piece in pieces:
        if len(piece) < 2:
            continue
        contours.extend(_stroke_polyline(piece, thickness * scale))
        if spec.get("cap") == "round":
            contours.append(circle_contour(piece[0][0], piece[0][1], (thickness * scale) / 2, 12))
            last = piece[-1]
            contours.append(circle_contour(last[0], last[1], (thickness * scale) / 2, 12))
    if spec.get("arrowEnd"):
        head = _arrow_contour(scaled, thickness * scale * 2.4)
        if head:
            contours.append(head)
    if spec.get("arrowStart"):
        head = _arrow_contour(list(reversed(scaled)), thickness * scale * 2.4)
        if head:
            contours.append(head)

    glow = _num(spec.get("glow"), 0)
    if glow > 0:
        glow_map = Bitmap(bitmap.width, bitmap.height)
        region = rasterize_contours(contours, glow_map.width, glow_map.height)
        fill_coverage(glow_map, region, _n(spec.get("glowColor"), color), 1)
        blurred = box_blur(glow_map, glow * scale, 1)
        draw_bitmap(bitmap, blurred, 0, 0, clamp(_num(spec.get("glowOpacity"), 0.8), 0, 1), "screen")

    region = rasterize_contours(contours, bitmap.width, bitmap.height)
    fill_coverage(bitmap, region, color, clamp(_num(spec.get("opacity"), 1), 0, 1))
    return _result(bitmap, width, height, scale)


def _neon_path(spec: dict, gctx: dict) -> dict:
    """ネオン。芯とグローの二層で描きます。`flicker` で不安定に明滅します。"""
    bitmap, width, height, scale = _canvas(gctx)
    points = _to_points(spec.get("points"))
    if len(points) < 2:
        return _result(bitmap, width, height, scale)

    end = _num(spec.get("end"), 1)
    visible = _slice_polyline(points, spec.get("closed") is True, _num(spec.get("start"), 0), end)
    if len(visible) < 2:
        return _result(bitmap, width, height, scale)
    scaled = [[p[0] * scale, p[1] * scale] for p in visible]

    intensity = _num(spec.get("intensity"), 1.2)
    flicker = clamp(_num(spec.get("flicker"), 0), 0, 1)
    # 明滅は時刻から決まるノイズ。乱数を回さないので巻き戻しても同じ。
    if flicker > 0:
        # `effects.value_noise_2d` は «一括版» で、スカラーを渡すと 1 要素の配列が
        # 返ります。ここは 1 個しか要らないので、同じ値を出す `_noise2` を使います。
        noise = _noise2(_num(gctx.get("time"), 0) * 9, 0.5, js_round(_num(spec.get("seed"), 1)))
        wobble = 1 - flicker * (0.5 + 0.5 * noise)
    else:
        wobble = 1.0

    glow_width = max(1, _num(spec.get("glowWidth"), 24))
    core_width = max(0.5, _num(spec.get("coreWidth"), 4))
    glow_color = parse_color(_n(spec.get("glowColor"), "#ff2bd1"))
    core_color = parse_color(_n(spec.get("coreColor"), "#ffffff"))

    # 外側のグローを 2 段階（広く薄く → 狭く濃く）
    for factor, opacity in ((1.0, 0.35), (0.55, 0.55)):
        # JS はここのビットマップを `layer` と呼んでいますが、この移植では
        # `layer` は «プロジェクト JSON のレイヤー（辞書）» の名前なので、
        # 取り違えないよう別名にしています。
        glow_layer = Bitmap(bitmap.width, bitmap.height)
        contours = _stroke_polyline(scaled, glow_width * factor * scale)
        region = rasterize_contours(contours, glow_layer.width, glow_layer.height)
        fill_coverage(glow_layer, region, glow_color, 1)
        blurred = box_blur(glow_layer, glow_width * factor * scale * 0.4, 1)
        draw_bitmap(bitmap, blurred, 0, 0, clamp(opacity * intensity * wobble, 0, 1), "screen")

    core_contours = list(_stroke_polyline(scaled, core_width * scale))
    core_contours.append(circle_contour(scaled[0][0], scaled[0][1], (core_width * scale) / 2, 12))
    last = scaled[-1]
    core_contours.append(circle_contour(last[0], last[1], (core_width * scale) / 2, 12))
    core_region = rasterize_contours(core_contours, bitmap.width, bitmap.height)
    fill_coverage(bitmap, core_region, core_color, clamp(wobble, 0, 1))
    return _result(bitmap, width, height, scale)


# ══════════════════════════════════════════════════════════════════
# metaball
# ══════════════════════════════════════════════════════════════════


def _metaball(spec: dict, gctx: dict) -> dict:
    """メタボール。点の «場» が閾値を超えた範囲を塗ります。点が近づくと融合します。

    **JS の二重 for を NumPy の一括演算に置き換えています。** JS も «粗い格子で
    場を求めて拡大» していたので、格子の上で計算して `np.repeat` で伸ばすだけで
    同じ絵になります（拡大は最近傍 ＝ `floor(x / stepX)` と同じ意味）。
    """
    bitmap, width, height, scale = _canvas(gctx)
    raw_points = _n(spec.get("points"), []) or []
    points = []
    for p in raw_points:
        # JS は `p.x ?? 0` なので、辞書でないものが混じっても «原点の点» になります。
        # 黙って落とすと «点が 1 つ消えた» と気付きにくいので、同じ扱いにそろえます。
        item = p if isinstance(p, dict) else {}
        points.append(
            (
                _num(item.get("x")) * scale,
                _num(item.get("y")) * scale,
                max(1, _num(item.get("radius"), 60) * scale),
            )
        )
    if not points or bitmap.width == 0 or bitmap.height == 0:
        return _result(bitmap, width, height, scale)

    threshold = _num(spec.get("threshold"), 0.5)
    smoothness = max(0.05, _num(spec.get("smoothness"), 1.2))
    fill = parse_color(_n(spec.get("fill"), "#4cc9f0"))
    outline = spec.get("outline") if isinstance(spec.get("outline"), dict) else None
    outline_color = parse_color(_n(outline.get("color"), "#ffffff")) if outline else None
    outline_width = max(0.5, _num(outline.get("width"), 3) * scale) if outline else 0.0
    # 内部解像度を落として場を計算し、拡大して使う（重さ対策）
    resolution = int(clamp(js_round(_num(spec.get("resolution"), 200)), 16, 1024))
    step_x = max(1, js_round(bitmap.width / resolution))
    step_y = max(1, js_round(bitmap.height / resolution))

    blocks_x = (bitmap.width - 1) // step_x + 1
    blocks_y = (bitmap.height - 1) // step_y + 1
    fx = (np.arange(blocks_x, dtype=np.float64) * step_x + step_x / 2)[None, :]
    fy = (np.arange(blocks_y, dtype=np.float64) * step_y + step_y / 2)[:, None]

    total = np.zeros((blocks_y, blocks_x), np.float64)
    inside = np.zeros((blocks_y, blocks_x), np.bool_)
    for px, py, radius in points:
        distance_sq = (fx - px) ** 2 + (fy - py) ** 2
        # 逆二乗の «場»。半径で正規化しているので radius がそのまま効く。
        # 中心にぴったり乗った画素は JS が «場 = 10» で打ち切るので同じにする。
        near = distance_sq <= 1e-6
        inside |= near
        total += (radius * radius) / np.where(near, 1.0, distance_sq)
    value = np.where(inside, 10.0, total)

    # 閾値の周りをなだらかにして輪郭のギザギザを抑える
    denominator = threshold * smoothness
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.clip((value - threshold) / denominator, 0, 1)
    # threshold が 0 だと JS も 0/0 → NaN → 代入時に 0 になる。同じ結果にそろえる。
    coverage = np.nan_to_num(coverage, nan=0.0, posinf=1.0, neginf=0.0)

    painted = coverage > 0
    if outline_color is not None:
        # JS は «被覆率が薄いところ» を輪郭とみなして不透明に塗る。同じ判定にする。
        is_outline = painted & (coverage < outline_width / 20)
        channels = [np.where(is_outline, outline_color[c], fill[c]) for c in range(3)]
        channels.append(np.where(is_outline, 255.0, coverage * 255 * fill[3]))
    else:
        channels = [np.full(coverage.shape, fill[c], np.float64) for c in range(3)]
        channels.append(coverage * 255 * fill[3])

    # 格子 → 画素。`floor(x / stepX)` は «step 個ずつ同じ値» なので repeat で足ります。
    # **色を出してから広げます。** 先に画素まで広げると (H, W) の float64 が何本も
    # 出て、1280x720 では場の計算より «広げたあとの後始末» のほうが重くなります。
    for c in range(4):
        block = _u8_array(channels[c]) * painted
        expanded = np.repeat(np.repeat(block, step_y, axis=0), step_x, axis=1)
        bitmap.data[..., c] = expanded[: bitmap.height, : bitmap.width]
    return _result(bitmap, width, height, scale)


# ══════════════════════════════════════════════════════════════════
# waterSurface
# ══════════════════════════════════════════════════════════════════


def _water_surface(spec: dict, gctx: dict) -> dict:
    """水面。`reflect: "below"` のときは呼び出し側が渡してくれた «下の絵» を
    上下反転して波で歪ませます。遠景ほど波を細かくして奥行きを出します。
    """
    bitmap, width, height, scale = _canvas(gctx)
    if bitmap.width == 0 or bitmap.height == 0:
        return _result(bitmap, width, height, scale)
    horizon = clamp(_num(spec.get("horizon"), 0.62), 0, 1)
    tint = parse_color(_n(spec.get("colorTint"), "#1b3a5c"))
    source = gctx.get("reflection_source")
    source_data = source.data if source is not None else np.zeros((1, 1, 4), np.uint8)

    _k_water_surface(
        bitmap.data,
        js_round(bitmap.height * horizon),
        _num(spec.get("waveScale"), 0.02),
        _num(spec.get("waveSpeed"), 0.6),
        _num(spec.get("amplitude"), 8) * scale,
        _num(spec.get("refraction"), 0.3),
        float(tint[0]),
        float(tint[1]),
        float(tint[2]),
        clamp(_num(spec.get("specular"), 0.4), 0, 1),
        clamp(_num(spec.get("perspective"), 0.7), 0, 1),
        _num(gctx.get("time"), 0),
        js_round(_num(spec.get("seed"), 17)),
        source_data,
        1 if source is not None else 0,
    )
    return _result(bitmap, width, height, scale)


# ══════════════════════════════════════════════════════════════════
# mesh（OBJ）
# ══════════════════════════════════════════════════════════════════


def _mesh(spec: dict, gctx: dict) -> dict:
    """OBJ メッシュ。立体そのものを置きたいときに使う。

    `primitive3d` が «立方体と球体しか出せない» のに対し、こちらは読み込んだ
    モデルをそのまま描きます。面ごとに奥行きを持たせているので、へこんだ形でも
    自分自身の前後関係が正しく出ます。

    カメラはレイヤー内で完結させています（プロジェクトの camera とは別）。
    他のレイヤーと深度を共有しないので «メッシュの向こう側に別のレイヤーが
    差し込む» ことはできません。
    """
    bitmap, width, height, scale = _canvas(gctx)
    assets = gctx.get("assets")
    source = spec.get("source")
    if not source and spec.get("asset") and assets is not None:
        source = assets.text(spec["asset"])
    if not source:
        return _result(bitmap, width, height, scale)

    # 同じ文字列からは同じモデルになるので、解析結果をレイヤー間で使い回す。
    # 毎フレーム parse すると、頂点数の多いモデルでそこが支配的になる。
    model = MESH_CACHE.get(source)
    if model is None:
        model = parse_obj(source)
        model["bounds"] = normalize_bounds(model["positions"])
        MESH_CACHE[source] = model
    texture = assets.get(spec["texture"]) if spec.get("texture") and assets is not None else None

    centre_x = bitmap.width / 2
    centre_y = bitmap.height / 2
    # レイヤーの中で完結したカメラ。距離は «大きさ 3 倍» を既定にして、
    # 遠近が付きすぎず、かつ立体に見える程度にしてある。
    size = _num(spec.get("size"), 260) * scale
    distance = _num(spec.get("distance"), size * 3)
    eye = {"x": centre_x, "y": centre_y, "z": -distance}
    camera = {
        "eye": eye,
        "basis": camera_basis(eye, None, None),
        "referenceDistance": distance,
        "centreX": centre_x,
        "centreY": centre_y,
    }
    depth = np.full((bitmap.height, bitmap.width), np.inf, np.float32)
    mesh_transform = {
        "x": centre_x,
        "y": centre_y,
        "z": 0,
        "rotation": _num(spec.get("rotationZ"), 0),
        "rotationX": _num(spec.get("rotationX"), 0),
        "rotationY": _num(spec.get("rotationY"), 0),
        "scaleX": 1,
        "scaleY": 1,
        "meshSize": size,
    }
    spec_light = spec.get("light")
    light = None
    if isinstance(spec_light, dict):
        light = [
            _num(spec_light.get("x"), -0.4),
            _num(spec_light.get("y"), -0.7),
            _num(spec_light.get("z"), 0.6),
        ]

    # 影は本体より «先に» 描く。あとから描くと本体の上に乗ってしまう。
    shadow = spec.get("shadow")
    if isinstance(shadow, dict) and shadow.get("enabled") is not False:
        draw_floor_shadow(
            bitmap,
            model,
            mesh_transform,
            camera,
            {
                "floorY": centre_y + _num(shadow.get("floorY"), size * 0.55),
                "light": light,
                "lightPosition": _n(shadow.get("from"), None),
                "opacity": _num(shadow.get("opacity"), 0.3),
                "color": parse_color_object(shadow["color"]) if shadow.get("color") else None,
                "depth": depth,
            },
        )

    point_lights = spec.get("pointLights")
    draw_mesh(
        bitmap,
        model,
        texture,
        mesh_transform,
        camera,
        {
            "alpha": 1,
            "color": parse_color_object(_n(spec.get("color"), "#c8c8d2")),
            "shading": clamp(_num(spec.get("shading"), 0.55), 0, 1),
            "light": light,
            "pointLights": point_lights if isinstance(point_lights, list) else [],
            "doubleSided": spec.get("doubleSided") is not False,
            "depth": depth,
        },
    )
    return _result(bitmap, width, height, scale)


# ══════════════════════════════════════════════════════════════════
# primitive3d
# ══════════════════════════════════════════════════════════════════


def _fill_quad(bitmap: Bitmap, xs, ys, color) -> None:
    """四角形 1 枚を «その外接矩形だけ» で塗る。

    JS は面ごとに画面いっぱいのカバレッジ配列を作り直しています。球は既定でも
    288 面あるので、そのまま移すと 1 フレームに 288 枚ぶんの確保になり、
    Python では確保だけで支配的になります。外接矩形へ切り出してから塗っても
    **触る画素と結果は同じ**なので、こちらにしています。
    """
    min_x = max(0, int(math.floor(min(xs))) - 1)
    max_x = min(bitmap.width - 1, int(math.ceil(max(xs))) + 1)
    min_y = max(0, int(math.floor(min(ys))) - 1)
    max_y = min(bitmap.height - 1, int(math.ceil(max(ys))) + 1)
    if max_x < min_x or max_y < min_y:
        return
    local = np.empty(len(xs) * 2, np.float64)
    local[0::2] = np.asarray(xs, np.float64) - min_x
    local[1::2] = np.asarray(ys, np.float64) - min_y
    region = rasterize_contours([local], max_x - min_x + 1, max_y - min_y + 1)
    view = Bitmap(max_x - min_x + 1, max_y - min_y + 1, bitmap.data[min_y : max_y + 1, min_x : max_x + 1])
    fill_coverage(view, region, color, 1)


def _primitive3d(spec: dict, gctx: dict) -> dict:
    """3D プリミティブ。立方体と球体にテクスチャを貼って回します。背面は描きません。"""
    bitmap, width, height, scale = _canvas(gctx)
    assets = gctx.get("assets")
    texture = assets.get(spec["asset"]) if spec.get("asset") and assets is not None else None
    size = _num(spec.get("size"), 300) * scale
    rx = to_radians(_num(spec.get("rotationX"), 0))
    ry = to_radians(_num(spec.get("rotationY"), 0))
    rz = to_radians(_num(spec.get("rotationZ"), 0))
    shading = clamp(_num(spec.get("shading"), 0.5), 0, 1)
    centre_x = bitmap.width / 2
    centre_y = bitmap.height / 2
    distance = size * 3
    fallback = parse_color(_n(spec.get("color"), "#4cc9f0"))

    cos_x, sin_x = math.cos(rx), math.sin(rx)
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    cos_z, sin_z = math.cos(rz), math.sin(rz)

    def rotate(p):
        # X → Y → Z の順に回す
        x, y, z = p
        t = y * cos_x - z * sin_x
        z = y * sin_x + z * cos_x
        y = t
        t = x * cos_y + z * sin_y
        z = -x * sin_y + z * cos_y
        x = t
        t = x * cos_z - y * sin_z
        y = x * sin_z + y * cos_z
        x = t
        return (x, y, z)

    def project(p):
        k = distance / (distance + p[2]) if (distance + p[2]) != 0 else math.inf
        return (centre_x + p[0] * k, centre_y + p[1] * k, p[2])

    def make_face(corners, uvs):
        """面を 1 枚組み立てる。法線が手前を向いていなければ捨てる。"""
        rotated = [rotate(c) for c in corners]
        projected = [project(r) for r in rotated]
        # 面の向き（画面上での符号付き面積）で表裏を判定
        area = (projected[1][0] - projected[0][0]) * (projected[2][1] - projected[0][1]) - (
            projected[2][0] - projected[0][0]
        ) * (projected[1][1] - projected[0][1])
        if area <= 0:
            return None
        # 法線の z 成分から簡単な陰影を付ける
        ax = rotated[1][0] - rotated[0][0]
        ay = rotated[1][1] - rotated[0][1]
        az = rotated[1][2] - rotated[0][2]
        bx = rotated[2][0] - rotated[0][0]
        by = rotated[2][1] - rotated[0][1]
        bz = rotated[2][2] - rotated[0][2]
        nz = ax * by - ay * bx
        # ★ `sqrt(a² + b² + c²)` ではなく `hypot` を使うこと。JS の `Math.hypot` は
        # 桁あふれを避ける計算をしていて、素直な平方根とは最後の 1 ビットが違います。
        # ここは陰影の明るさになるので、その差がそのまま «1 だけ色の違う画素» で出ます。
        length = math.hypot(ay * bz - az * by, az * bx - ax * bz, nz) or 1
        light = 1 - shading + shading * clamp(abs(nz / length), 0, 1)
        depth = (rotated[0][2] + rotated[1][2] + rotated[2][2] + rotated[3][2]) / 4
        return {"projected": projected, "uvs": uvs, "light": light, "depth": depth}

    faces = []
    if _n(spec.get("shape"), "cube") == "sphere":
        # 球はテクスチャを緯度経度で貼る
        segments = int(clamp(js_round(_num(spec.get("segments"), 24)), 6, 64))
        rings = js_round(segments / 2)
        radius = size / 2
        for ring in range(rings):
            phi0 = (ring / rings) * math.pi
            phi1 = ((ring + 1) / rings) * math.pi
            for segment in range(segments):
                theta0 = (segment / segments) * TAU
                theta1 = ((segment + 1) / segments) * TAU

                def point(phi, theta):
                    return (
                        radius * math.sin(phi) * math.cos(theta),
                        radius * math.cos(phi),
                        radius * math.sin(phi) * math.sin(theta),
                    )

                face = make_face(
                    [point(phi0, theta0), point(phi0, theta1), point(phi1, theta1), point(phi1, theta0)],
                    [
                        (segment / segments, ring / rings),
                        ((segment + 1) / segments, ring / rings),
                        ((segment + 1) / segments, (ring + 1) / rings),
                        (segment / segments, (ring + 1) / rings),
                    ],
                )
                if face:
                    faces.append(face)
    else:
        h = size / 2
        corners = [
            (-h, -h, -h),
            (h, -h, -h),
            (h, h, -h),
            (-h, h, -h),
            (-h, -h, h),
            (h, -h, h),
            (h, h, h),
            (-h, h, h),
        ]
        quads = [
            (0, 1, 2, 3),
            (5, 4, 7, 6),
            (4, 0, 3, 7),
            (1, 5, 6, 2),
            (4, 5, 1, 0),
            (3, 2, 6, 7),
        ]
        uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for quad in quads:
            face = make_face([corners[i] for i in quad], uv)
            if face:
                faces.append(face)

    # 奥の面から描く（Z ソート）。Python の sort も JS と同じく安定なので、
    # 深度が同じ面の前後関係は作った順のまま変わりません。
    faces.sort(key=lambda f: -f["depth"])
    for face in faces:
        light = face["light"]
        # tint は「その色へ寄せる」ので、陰影は黒へ寄せる量として渡す
        tint = None if light >= 1 else (0, 0, 0, clamp(1 - light, 0, 1))
        if texture is not None:
            def vertex(index, face=face):
                return (
                    face["projected"][index][0],
                    face["projected"][index][1],
                    face["uvs"][index][0] * texture.width,
                    face["uvs"][index][1] * texture.height,
                )

            draw_textured_triangle(
                bitmap, texture, vertex(0), vertex(1), vertex(2), clamp_edge=True, tint=tint
            )
            draw_textured_triangle(
                bitmap, texture, vertex(0), vertex(2), vertex(3), clamp_edge=True, tint=tint
            )
        else:
            xs = [p[0] for p in face["projected"]]
            ys = [p[1] for p in face["projected"]]
            _fill_quad(
                bitmap,
                xs,
                ys,
                (fallback[0] * light, fallback[1] * light, fallback[2] * light, 1),
            )
    return _result(bitmap, width, height, scale)


# ══════════════════════════════════════════════════════════════════
# spotlight
# ══════════════════════════════════════════════════════════════════


def _spotlight(spec: dict, gctx: dict) -> dict:
    """スポットライト（円錐・ボリュメトリック）。

    舞台照明は «円錐状に広がって、床で切れる» のがそれらしさの正体です。
    縦長の矩形をぼかしたものだと上下が同じ幅の «帯» にしか見えず、光が降ってくる
    感じが出ないので、光源から扇形に広げて描いています。

    明るさはアルファに載せるだけで色は一定です。`blend: "screen"` で重ねる «光»
    なので、そのほうが下の絵と素直に馴染みます。
    """
    bitmap, width, height, scale = _canvas(gctx)
    intensity = _num(spec.get("intensity"), 0.7)
    if intensity <= 0 or bitmap.width == 0 or bitmap.height == 0:
        return _result(bitmap, width, height, scale)

    color = parse_color(_n(spec.get("color"), "#fff4c2"))
    ox = _num(spec.get("originX"), 0.5) * bitmap.width
    oy = _num(spec.get("originY"), -0.05) * bitmap.height
    # angle は度。90 が真下（パーティクルの direction と同じ向きの取り方）。
    angle = to_radians(_num(spec.get("angle"), 90))
    dx = math.cos(angle)
    dy = math.sin(angle)
    # ビームの «横» 方向。ここまでの距離で円錐の内外を判定する。
    nx = -dy
    ny = dx
    # spread は半頂角。90 度に近いと円錐が平面に潰れて意味を失うので上限を切る。
    tan_spread = math.tan(to_radians(clamp(_num(spec.get("spread"), 14), 0.1, 88)))
    reach = _ray_exit_distance(ox, oy, dx, dy, bitmap.width, bitmap.height)
    beam = max(1, _num(spec.get("length"), 0.95) * reach)
    # softness 0 でも端を 1 段階だけぼかす。完全な直線の境界はギザギザが目立つため。
    inner = clamp(1 - _num(spec.get("softness"), 0.5), 0, 0.985)
    haze = clamp(_num(spec.get("haze"), 0), 0, 1)
    haze_freq = _num(spec.get("hazeScale"), 0.012) / scale
    time = _num(gctx.get("time"), 0)
    seed = js_round(_num(spec.get("seed"), 23))
    # 明滅は時刻から決まるノイズ。乱数を回さないので巻き戻しても同じ絵になる。
    flicker = clamp(_num(spec.get("flicker"), 0), 0, 1)
    if flicker > 0:
        wobble = 1 - flicker * (0.5 + 0.5 * _noise2(time * 7.0, 0.5, seed))
    else:
        wobble = 1.0
    # 明るさは画素ごとに変わらないので、ループの外で 1 つにまとめておく
    lightness = intensity * wobble

    # 円錐の外は必ず 0 なので、三角形の外接矩形しか走査しない。
    half_width = beam * tan_spread
    tip_x = ox + dx * beam
    tip_y = oy + dy * beam
    xs = (ox, tip_x + nx * half_width, tip_x - nx * half_width)
    ys = (oy, tip_y + ny * half_width, tip_y - ny * half_width)
    min_x = int(clamp(math.floor(min(xs)) - 1, 0, bitmap.width - 1))
    max_x = int(clamp(math.ceil(max(xs)) + 1, 0, bitmap.width - 1))
    min_y = int(clamp(math.floor(min(ys)) - 1, 0, bitmap.height - 1))
    max_y = int(clamp(math.ceil(max(ys)) + 1, 0, bitmap.height - 1))

    _k_spotlight(
        bitmap.data,
        min_x,
        min_y,
        max_x,
        max_y,
        ox,
        oy,
        dx,
        dy,
        nx,
        ny,
        tan_spread,
        beam,
        inner,
        lightness,
        haze,
        haze_freq,
        time,
        seed,
        float(color[0]),
        float(color[1]),
        float(color[2]),
        float(color[3]),
    )
    return _result(bitmap, width, height, scale)


# ══════════════════════════════════════════════════════════════════
# shapeAnim
# ══════════════════════════════════════════════════════════════════


def _shape_anim(spec: dict, gctx: dict) -> dict:
    """図形アニメーションのプリセット集。素材なしで «画面を彩る動き» を出します。"""
    bitmap, width, height, scale = _canvas(gctx)
    preset = _n(spec.get("preset"), "circleBurst")
    color = parse_color(_n(spec.get("color"), "#ffd166"))
    speed = _num(spec.get("speed"), 1)
    amount = _num(spec.get("scale"), 1)
    time = _num(gctx.get("time"), 0) * speed
    seed = js_round(_num(spec.get("seed"), 2))
    cx = bitmap.width / 2
    cy = bitmap.height / 2
    unit = min(bitmap.width, bitmap.height)
    contours: list = []

    # 乱数は «使う個数が決まらない» ので、多めに出しておいて前から取る。
    # 呼ぶ順は JS と同じなので、同じ絵が出ます。
    draws = _mulberry_stream(seed, max(1, js_round(300 * max(1.0, abs(amount)))) + 64)
    cursor = 0

    def random() -> float:
        nonlocal cursor, draws
        if cursor >= draws.size:
            # 足りなくなったら «続き» を出す（数列は位置だけで決まる）
            more = _mulberry_stream(seed, cursor * 2 + 64)
            draws = more
        value = float(draws[cursor])
        cursor += 1
        return value

    def push(*items):
        for item in items:
            if item is not None:
                contours.append(item)

    def loop(period: float) -> float:
        """0..1 を繰り返す進行度。ループする演出はこれを使う。"""
        return _jsmod(time, period) / period

    if preset == "circleBurst":
        # 輪が広がって消える。3 本を時間差で。
        for i in range(3):
            t = _jsmod(loop(1.2) + i / 3, 1)
            radius = t * unit * 0.5 * amount
            thickness = max(1, unit * 0.02 * amount * (1 - t))
            if thickness <= 0.5:
                continue
            push(circle_contour(cx, cy, radius + thickness / 2, 64))
            push(_reverse_contour(circle_contour(cx, cy, max(0, radius - thickness / 2), 64)))
    elif preset == "ringPulse":
        t = loop(1)
        radius = unit * 0.3 * amount * (1 + math.sin(t * TAU) * 0.08)
        thickness = max(1, unit * 0.015 * amount)
        push(circle_contour(cx, cy, radius + thickness / 2, 72))
        push(_reverse_contour(circle_contour(cx, cy, max(0, radius - thickness / 2), 72)))
    elif preset == "lineSweep":
        # 線が左から右へ走る
        count = js_round(6 * amount)
        for i in range(count):
            t = _jsmod(loop(1.4) + i / count, 1)
            x = t * bitmap.width
            w = max(1, unit * 0.006 * amount)
            top = (i / count) * bitmap.height
            push([x, top, x + w, top, x + w, top + bitmap.height / count, x, top + bitmap.height / count])
    elif preset == "speedStreaks":
        count = js_round(24 * amount)
        for _ in range(count):
            y = random() * bitmap.height
            length = (0.1 + random() * 0.3) * bitmap.width * amount
            t = _jsmod(loop(0.9) + random(), 1)
            x = t * (bitmap.width + length) - length
            w = max(1, unit * 0.004 * (0.5 + random()))
            push([x, y, x + length, y, x + length, y + w, x, y + w])
    elif preset == "squareGrid":
        columns = max(2, js_round(6 * amount))
        rows = max(2, js_round((columns * bitmap.height) / bitmap.width)) if bitmap.width else 2
        cell_w = bitmap.width / columns
        cell_h = bitmap.height / rows
        for row in range(rows):
            for column in range(columns):
                # 中心からの距離で時間差を付けて «波» のように点く
                distance = math.hypot(column - columns / 2, row - rows / 2) / math.hypot(
                    columns / 2, rows / 2
                )
                t = _jsmod(loop(1.6) - distance * 0.4 + 1, 1)
                size = max(0, math.sin(t * math.pi)) * min(cell_w, cell_h) * 0.55
                if size <= 0.5:
                    continue
                x = column * cell_w + cell_w / 2
                y = row * cell_h + cell_h / 2
                push([
                    x - size / 2, y - size / 2,
                    x + size / 2, y - size / 2,
                    x + size / 2, y + size / 2,
                    x - size / 2, y + size / 2,
                ])
    elif preset == "radialTicks":
        count = js_round(36 * amount)
        inner = unit * 0.28
        outer = unit * 0.38
        for i in range(count):
            angle = (i / count) * TAU + time * 0.4
            pulse = 0.6 + 0.4 * math.sin(time * TAU + i * 0.7)
            w = max(1, unit * 0.006)
            x0 = cx + math.cos(angle) * inner
            y0 = cy + math.sin(angle) * inner
            x1 = cx + math.cos(angle) * (inner + (outer - inner) * pulse)
            y1 = cy + math.sin(angle) * (inner + (outer - inner) * pulse)
            push(*_stroke_polyline([[x0, y0], [x1, y1]], w))
    elif preset == "zigzag":
        points = []
        # JS は steps が 0 でも 0/0 = NaN の点を 1 つ作って «何も出ない» で済みますが、
        # Python は ZeroDivisionError になるので、ここで «描かない» に倒します。
        steps = max(0, js_round(12 * amount))
        for i in range(steps + 1 if steps > 0 else 0):
            t = i / steps
            points.append([
                t * bitmap.width,
                cy + (-1 if i % 2 == 0 else 1) * unit * 0.12 * math.sin(time * TAU * 0.5 + t * 4),
            ])
        push(*_stroke_polyline(points, max(1, unit * 0.012 * amount)))
    elif preset == "confettiShapes":
        count = js_round(40 * amount)
        for _ in range(count):
            x = random() * bitmap.width
            fall_speed = 0.4 + random() * 0.8
            y = _jsmod(random() + time * fall_speed * 0.3, 1) * bitmap.height
            size = unit * (0.01 + random() * 0.02) * amount
            angle = random() * TAU + time * 2
            kind = math.floor(random() * 3)
            if kind == 0:
                push(circle_contour(x, y, size, 10))
            else:
                points = []
                sides = 3 if kind == 1 else 4
                for s in range(sides):
                    a = angle + (s / sides) * TAU
                    points.append(x + math.cos(a) * size)
                    points.append(y + math.sin(a) * size)
                push(points)

    if contours:
        region = rasterize_contours(contours, bitmap.width, bitmap.height)
        fill_coverage(bitmap, region, color, clamp(_num(spec.get("opacity"), 1), 0, 1))
    return _result(bitmap, width, height, scale)


# ══════════════════════════════════════════════════════════════════
# 登録と入口
# ══════════════════════════════════════════════════════════════════

#: JS 版の `generatorLayers` と同じ並び。キーは JSON の綴り（camelCase）のまま。
GENERATOR_LAYERS = {
    "starfield": _starfield,
    "linePath": _line_path,
    "neonPath": _neon_path,
    "metaball": _metaball,
    "waterSurface": _water_surface,
    "mesh": _mesh,
    "primitive3d": _primitive3d,
    "spotlight": _spotlight,
    "shapeAnim": _shape_anim,
}

GENERATOR_LAYER_TYPES = list(GENERATOR_LAYERS)


def render_generator(renderer, layer: dict, ctx: dict, transform: dict, scene_time: float, target):
    """生成レイヤー（星空・線・ネオン・メタボール・水面・3D・照明・図形アニメ）の入口。

    どれも「素材なしで絵を作る」ものなので、共通の入口でまとめて扱います。

    :param renderer: `movo.renderer.index.Renderer`（属性は `render_scale` などの snake_case）
    :param layer: プロジェクト JSON のレイヤー（キーは camelCase のまま）
    :param ctx: `renderer._context_for()` の結果
    :param transform: `renderer._resolve_transform()` の結果
    :param target: このレイヤーを描き込む先（水面が «下の絵» を映すのに使う）
    :returns: `_render_content` と同じ形の辞書。描かないときは `None`
    """
    layer_type = layer.get("type")
    render = GENERATOR_LAYERS.get(layer_type)
    if render is None:
        return None
    spec = resolve_animated(_n(layer.get(layer_type), {}) or {}, ctx, {})
    if not isinstance(spec, dict):
        spec = {}

    # followLayer: 他レイヤーが通った跡を points にする（軌跡の描画）
    follow = spec.get("followLayer")
    if follow:
        trails = getattr(renderer, "motion_trails", None) or {}
        track = trails.get(follow)
        if not track or len(track) < 2:
            return None
        spec = {**spec, "points": [[_num(p.get("x")), _num(p.get("y"))] for p in track]}

    # 水面だけは「下に描かれている絵」を映すので、描画先を渡す
    reflection_source = (
        target if layer_type == "waterSurface" and _n(spec.get("reflect"), "below") == "below" else None
    )
    return render(
        spec,
        {
            "width": max(1, js_round(_num(transform.get("width"), renderer.width))),
            "height": max(1, js_round(_num(transform.get("height"), renderer.height))),
            "scale": renderer.render_scale,
            "time": scene_time,
            # ★ `renderer.timeline` は «辞書» です（`build_timeline()` の戻り値。
            # キーは JSON と同じ camelCase）。JS の `renderer.timeline.fps` を
            # そのまま写すと属性アクセスになって落ちます。
            "fps": renderer.timeline["fps"],
            "seed": renderer.seed,
            "assets": renderer.assets,
            "reflection_source": reflection_source,
        },
    )


__all__ = ["GENERATOR_LAYERS", "GENERATOR_LAYER_TYPES", "MESH_CACHE", "render_generator"]
