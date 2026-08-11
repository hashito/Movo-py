"""Numba でコンパイルする «画素ごとのループ» を集めた場所。

## なぜここに集めるか

Python の `for` を 1 画素ずつ回すと 1280x720 の 1 パスで **720 ミリ秒**かかります。
NumPy で書き直せる処理（全画面に一様なもの）は NumPy にしますが、
**ラスタライザは NumPy にすると «遅くなります»**。

| 多角形 1 枚の塗り（1280x720） | |
| --- | --- |
| NumPy で «囲む矩形を一括判定» | 30.4 ms |
| Numba でコンパイルした走査線 | **0.296 ms**（103 倍） |

NumPy 版は **O(囲む矩形の面積 x 辺の数)** で、辺ごとに矩形いっぱいの一時配列を
作ります。走査線は **O(辺 x 行)** で、塗る必要のある画素にしか触りません。
**アルゴリズムの差はベクトル化では埋まりません。**

## 書くときの決まり

- `@njit(cache=True, fastmath=True)` を付ける。`cache=True` があると
  2 回目以降の起動で JIT のコンパイル（初回 1 秒ほど）が消えます。
  ただし **合成（source-over）の式を含むカーネルは `fastmath=PARITY_FASTMATH`**
  にします。`True` のままだと LLVM が式を括り直して JS と 1 ビットずれます
- **渡してよいのは NumPy 配列とスカラーだけ**です。辞書・クラス・文字列は渡せません。
  合成モードは文字列ではなく **番号**（`BLEND_IDS`）で渡します
- 呼ぶ側（`raster.py` / `text.py`）が «形の組み立て» を担当し、ここは
  «出来上がった配列を舐める» だけにします

## JS 版と同じ絵を出すために

- **画素の丸め方**。JS の `Uint8ClampedArray` は «五捨五入»（round half to even）
  です。Python の `astype(np.uint8)` は切り捨てなので、そのままだと 1 ずれます。
  `_u8()` で JS と同じ丸め方をしています
- **交点の並べ替えは挿入ソート**。JS の `Array.prototype.sort` は安定なので、
  x が同じ交点の順が入れ替わると nonzero 塗りの結果が変わります
- **カバレッジは float32**。JS も `Float32Array` なので、足すたびに float32 へ
  丸められます。float64 で持つと «ほんの少し» 違う絵になります
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# ── 合成モードの番号 ──────────────────────────────────────────────
#
# Numba には文字列を渡せないので番号にします。**並び順が ID そのもの**なので、
# 途中に足さないでください（`raster.BLEND_MODES` と 1 対 1 で対応します）。
BLEND_NORMAL = 0
BLEND_ADD = 1
BLEND_SUBTRACT = 2
BLEND_SCREEN = 3
BLEND_MULTIPLY = 4
BLEND_OVERLAY = 5
BLEND_HARD_LIGHT = 6
BLEND_SOFT_LIGHT = 7
BLEND_COLOR_DODGE = 8
BLEND_COLOR_BURN = 9
BLEND_LINEAR_BURN = 10
BLEND_LINEAR_LIGHT = 11
BLEND_VIVID_LIGHT = 12
BLEND_PIN_LIGHT = 13
BLEND_DARKEN = 14
BLEND_LIGHTEN = 15
BLEND_DIFFERENCE = 16
BLEND_EXCLUSION = 17
BLEND_HUE = 18
BLEND_SATURATION = 19
BLEND_COLOR = 20
BLEND_LUMINOSITY = 21

# 4 倍の «縦» スーパーサンプリング。横は被覆率を面積で正確に出すので、
# 縦だけ細かく刻めば足ります（JS 版と同じ）。
SUBSAMPLES = 4

#: **JS 版と 1 ビットまで同じ絵**が要る合成のカーネルに付ける `fastmath`。
#:
#: `fastmath=True` は LLVM に «速いなら式を書き換えてよい» と伝えます。そのうち
#: 4 つが合成の式を壊すので、**残り 3 つだけ**を渡します。
#:
#: - `reassoc` — 掛け算・足し算を括り直す。`cb * da * (1 - sa)` と
#:   `cb * (da * (1 - sa))` が同じ命令に潰れるので、**JS の並びに寄せて書いても
#:   効きません**（この旗が付いている限り、並べ替えは «書いても無駄» です）
#: - `contract` — `sa + da * (1 - sa)` を FMA 1 命令に畳む。途中の丸めが 1 回減ります
#: - `arcp` — `x / out_a` を `x * (1 / out_a)` に変える。割り算の丸めが変わります
#: - `afn` — 数学関数を近似版に差し替える
#:
#: 残る `nnan` `ninf` `nsz`（NaN・無限大・符号付きゼロを «来ない» と仮定する）は
#: 有限の値どうしの計算を変えないので、絵は 1 ビットも動きません。
#:
#: 1280x720 のプレーン 1 枚（三角形 2 枚）の実測:
#:
#: | `fastmath` | 時間 | JS との差 |
#: | --- | --- | --- |
#: | `True` | 64.5 ms | **3072 画素中 4 画素ずれる** |
#: | `PARITY_FASTMATH` | 83.9 ms | 0 画素 |
#: | `False` | 93.0 ms | 0 画素 |
PARITY_FASTMATH = {"nnan", "ninf", "nsz"}


# ══════════════════════════════════════════════════════════════════
# 画素の丸め
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True, inline="always")
def _u8(v):
    """JS の `Uint8ClampedArray` への代入と同じ丸め方（五捨五入・0..255 で頭打ち）。

    **`int(v)` や `astype(np.uint8)` では合いません。** 0.5 ちょうどのときに
    «偶数側へ» 丸めるのが JS の仕様で、輪郭の縁の画素がしばしばここに当たります。
    """
    if v <= 0.0:
        return np.uint8(0)
    if v >= 255.0:
        return np.uint8(255)
    f = math.floor(v)
    d = v - f
    if d > 0.5:
        f += 1.0
    elif d == 0.5:
        # 端数がちょうど半分なら偶数側へ
        if int(f) % 2 == 1:
            f += 1.0
    return np.uint8(int(f))


# ══════════════════════════════════════════════════════════════════
# 合成モード
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True, inline="always")
def blend_channel(mode, cb, cs):
    """1 チャンネルずつ計算できる合成モード（0..255 で受けて 0..255 を返す）。

    式は W3C の Compositing and Blending に合わせています。自前で «それらしい»
    式を書くと、他のツールで作った絵と並べたときに色が合わなくなるためです。
    """
    if mode == BLEND_ADD:
        return min(255.0, cb + cs)
    if mode == BLEND_SUBTRACT:
        return max(0.0, cb - cs)
    if mode == BLEND_SCREEN:
        return 255.0 - ((255.0 - cb) * (255.0 - cs)) / 255.0
    if mode == BLEND_MULTIPLY:
        return (cb * cs) / 255.0
    if mode == BLEND_OVERLAY:
        if cb < 128.0:
            return (2.0 * cb * cs) / 255.0
        return 255.0 - (2.0 * (255.0 - cb) * (255.0 - cs)) / 255.0
    if mode == BLEND_HARD_LIGHT:
        # overlay の «下と上» を入れ替えたもの
        if cs < 128.0:
            return (2.0 * cb * cs) / 255.0
        return 255.0 - (2.0 * (255.0 - cb) * (255.0 - cs)) / 255.0
    if mode == BLEND_SOFT_LIGHT:
        b = cb / 255.0
        sv = cs / 255.0
        if b <= 0.25:
            d = ((16.0 * b - 12.0) * b + 4.0) * b
        else:
            d = math.sqrt(b)
        if sv <= 0.5:
            out = b - (1.0 - 2.0 * sv) * b * (1.0 - b)
        else:
            out = b + (2.0 * sv - 1.0) * (d - b)
        return out * 255.0
    if mode == BLEND_COLOR_DODGE:
        if cs >= 255.0:
            return 255.0
        return min(255.0, (cb * 255.0) / (255.0 - cs))
    if mode == BLEND_COLOR_BURN:
        if cs <= 0.0:
            return 0.0
        return 255.0 - min(255.0, ((255.0 - cb) * 255.0) / cs)
    if mode == BLEND_LINEAR_BURN:
        return max(0.0, cb + cs - 255.0)
    if mode == BLEND_LINEAR_LIGHT:
        return max(0.0, min(255.0, cb + 2.0 * cs - 255.0))
    if mode == BLEND_VIVID_LIGHT:
        if cs < 128.0:
            if cs <= 0.0:
                return 0.0
            return 255.0 - min(255.0, ((255.0 - cb) * 255.0) / (2.0 * cs))
        if cs >= 255.0:
            return 255.0
        return min(255.0, (cb * 255.0) / (2.0 * (255.0 - cs)))
    if mode == BLEND_PIN_LIGHT:
        if cs < 128.0:
            return min(cb, 2.0 * cs)
        return max(cb, 2.0 * cs - 255.0)
    if mode == BLEND_DARKEN:
        return min(cb, cs)
    if mode == BLEND_LIGHTEN:
        return max(cb, cs)
    if mode == BLEND_DIFFERENCE:
        return abs(cb - cs)
    if mode == BLEND_EXCLUSION:
        return cb + cs - (2.0 * cb * cs) / 255.0
    return cs


@njit(cache=True, fastmath=True, inline="always")
def _lum(r, g, b):
    return 0.3 * r + 0.59 * g + 0.11 * b


@njit(cache=True, fastmath=True)
def _clip_color(r, g, b):
    """はみ出した色を «明るさを保ったまま» 0..255 へ押し戻す（W3C の ClipColor）。

    `l` `mn` `mx` は **入ってきた色**から求めます。JS 版は 1 つ目の変換の結果に
    2 つ目の変換を掛けるので、その順序もそのままにしています。
    """
    l = _lum(r, g, b)
    mn = min(r, min(g, b))
    mx = max(r, max(g, b))
    if mn < 0.0:
        denom = l - mn
        if denom == 0.0:
            denom = 1.0
        r = l + ((r - l) * l) / denom
        g = l + ((g - l) * l) / denom
        b = l + ((b - l) * l) / denom
    if mx > 255.0:
        denom = mx - l
        if denom == 0.0:
            denom = 1.0
        r = l + ((r - l) * (255.0 - l)) / denom
        g = l + ((g - l) * (255.0 - l)) / denom
        b = l + ((b - l) * (255.0 - l)) / denom
    return r, g, b


@njit(cache=True, fastmath=True)
def _set_lum(r, g, b, l):
    d = l - _lum(r, g, b)
    return _clip_color(r + d, g + d, b + d)


@njit(cache=True, fastmath=True, inline="always")
def _sat(r, g, b):
    return max(r, max(g, b)) - min(r, min(g, b))


@njit(cache=True, fastmath=True)
def _set_sat(r, g, b, s):
    """最大・中間・最小の «位置» を保ったまま、幅だけ `s` に合わせ直す。

    JS 版は `[0,1,2].sort(...)` で並べます。V8 の小さい配列のソートは安定なので、
    同じ値が並んだときの順は元のまま（R→G→B）です。ここも挿入ソートで
    同じ挙動にしています。
    """
    c = np.empty(3, np.float64)
    c[0] = r
    c[1] = g
    c[2] = b
    order = np.empty(3, np.int64)
    order[0] = 0
    order[1] = 1
    order[2] = 2
    for i in range(1, 3):
        key = order[i]
        j = i - 1
        while j >= 0 and c[order[j]] > c[key]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    min_i = order[0]
    mid_i = order[1]
    max_i = order[2]
    out = np.zeros(3, np.float64)
    if c[max_i] > c[min_i]:
        out[mid_i] = ((c[mid_i] - c[min_i]) * s) / (c[max_i] - c[min_i])
        out[max_i] = s
    return out[0], out[1], out[2]


@njit(cache=True, fastmath=True)
def blend_non_separable(mode, br, bg, bb, sr, sg, sb):
    """3 チャンネルまとめて計算する合成モード（hue / saturation / color / luminosity）。

    «色相だけ» «明るさだけ» を取り替えるものなので、チャンネルごとに独立して
    計算できません。W3C の非分離ブレンドの手順そのままです。
    """
    if mode == BLEND_HUE:
        r, g, b = _set_sat(sr, sg, sb, _sat(br, bg, bb))
        return _set_lum(r, g, b, _lum(br, bg, bb))
    if mode == BLEND_SATURATION:
        r, g, b = _set_sat(br, bg, bb, _sat(sr, sg, sb))
        return _set_lum(r, g, b, _lum(br, bg, bb))
    if mode == BLEND_COLOR:
        return _set_lum(sr, sg, sb, _lum(br, bg, bb))
    # luminosity
    return _set_lum(br, bg, bb, _lum(sr, sg, sb))


@njit(cache=True, fastmath=PARITY_FASTMATH)
def blend_pixel(data, y, x, r, g, b, sa, mode):
    """1 画素に source-over で色を乗せる（`data` は (h, w, 4) の uint8）。

        outA = sa + da * (1 - sa)
        outC = (cs * sa + cb * da * (1 - sa)) / outA

    **`fastmath=True` にしないでください**（`PARITY_FASTMATH` の説明を参照）。
    掛け算の並びを LLVM に括り直されると、下の «JS と同じ並び» が消えます。
    """
    da = data[y, x, 3] / 255.0
    out_a = sa + da * (1.0 - sa)
    if out_a <= 0.0:
        data[y, x, 3] = 0
        return
    cb_r = np.float64(data[y, x, 0])
    cb_g = np.float64(data[y, x, 1])
    cb_b = np.float64(data[y, x, 2])
    if mode >= BLEND_HUE:
        mr, mg, mb = blend_non_separable(mode, cb_r, cb_g, cb_b, r, g, b)
    elif mode == BLEND_NORMAL:
        mr = r
        mg = g
        mb = b
    else:
        mr = blend_channel(mode, cb_r, r)
        mg = blend_channel(mode, cb_g, g)
        mb = blend_channel(mode, cb_b, b)
    # JS は `cb * da * (1 - sa)` を左から順に掛けます。`da * (1 - sa)` を先に
    # まとめると float64 の丸めが変わり、画素が 1 ずれます。ここは崩さないこと。
    inv = 1.0 - sa
    data[y, x, 0] = _u8((mr * sa + cb_r * da * inv) / out_a)
    data[y, x, 1] = _u8((mg * sa + cb_g * da * inv) / out_a)
    data[y, x, 2] = _u8((mb * sa + cb_b * da * inv) / out_a)
    data[y, x, 3] = _u8(out_a * 255.0)


# ══════════════════════════════════════════════════════════════════
# 走査線ラスタライザ — **ここが Movo でいちばん速度に効きます**
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=PARITY_FASTMATH)
def composite_bitmap_kernel(dst, src, ox, oy, alpha, mode):
    """ビットマップ 1 枚を `dst` の (ox, oy) へ source-over で重ねる。

    **NumPy で書くと «一時配列の山» になって遅くなります。** 1 画素あたり
    `astype` で float64 の中間を 10 本ほど作るので、960x540 の 1 枚で 94 ms
    かかっていました。ここは «重なる範囲を 1 回舐めるだけ» で足ります。

    | 466x466 を 960x540 へ 1 枚重ねる | |
    | --- | --- |
    | NumPy（float64 の中間を作る） | 94.3 ms |
    | このカーネル | **1.5 ms** |

    エフェクト側が «gather が支配的な処理はベクトル化しないほうが速い»
    （双一次補間 470 → 20 ms）と同じ結論に至っているのと同じ話です。
    帯域ではなく «中間配列を作る手間» が支配的なときは Numba が勝ちます。

    式と丸め方は JS 版の `compositeBitmap` そのままです。掛ける並びを
    崩さないでください（`PARITY_FASTMATH` は `reassoc` を渡さないので、
    **ここに書いた並びがそのまま命令になります**）。

    :param mode: 合成モードの番号。**非分離のモード（hue など）は呼ぶ側で 0 に
        してから渡します。** JS の `compositeBitmap` は分離できるモードしか
        見ないので、`hue` を渡すと `normal` になる — その挙動もそのままです
    """
    dst_h = dst.shape[0]
    dst_w = dst.shape[1]
    src_h = src.shape[0]
    src_w = src.shape[1]
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(dst_w, ox + src_w)
    y1 = min(dst_h, oy + src_h)
    if x1 <= x0 or y1 <= y0:
        return

    for y in range(y0, y1):
        sy = y - oy
        # 行の先を 1 回だけ引いておく（毎回 y のストライドを掛け直さない）
        d_row = dst[y]
        s_row = src[sy]
        for x in range(x0, x1):
            sx = x - ox
            sa = (s_row[sx, 3] / 255.0) * alpha
            # 透明な画素は «触らない»。下の色をそのまま残します
            if sa <= 0.0:
                continue

            # ── 不透明なときの近道 ────────────────────────────
            #
            # `sa == 1.0` なら out_a = 1 + da*0 = 1、inv = 0 になるので、式は
            # `(cs * 1 + cb * da * 0) / 1` = `cs` に **厳密に** 潰れます。
            # float64 でも丸めが 1 度も起きないので、**絵は 1 ビットも変わりません**
            # （`>=` ではなく `==` で見るのは、alpha に 1 より大きい値を渡された
            # ときに out_a が 1 でなくなり、この等式が崩れるためです）。
            # 不透明な背景板やレイヤーはこれで一気に速くなります。
            if sa == 1.0:
                if mode == BLEND_NORMAL:
                    d_row[x, 0] = s_row[sx, 0]
                    d_row[x, 1] = s_row[sx, 1]
                    d_row[x, 2] = s_row[sx, 2]
                else:
                    d_row[x, 0] = _u8(blend_channel(mode, np.float64(d_row[x, 0]), np.float64(s_row[sx, 0])))
                    d_row[x, 1] = _u8(blend_channel(mode, np.float64(d_row[x, 1]), np.float64(s_row[sx, 1])))
                    d_row[x, 2] = _u8(blend_channel(mode, np.float64(d_row[x, 2]), np.float64(s_row[sx, 2])))
                d_row[x, 3] = 255
                continue

            da = d_row[x, 3] / 255.0
            out_a = sa + da * (1.0 - sa)
            if out_a <= 0.0:
                d_row[x, 3] = 0
                continue
            inv = 1.0 - sa
            cb_r = np.float64(d_row[x, 0])
            cb_g = np.float64(d_row[x, 1])
            cb_b = np.float64(d_row[x, 2])
            cs_r = np.float64(s_row[sx, 0])
            cs_g = np.float64(s_row[sx, 1])
            cs_b = np.float64(s_row[sx, 2])
            # `normal` が圧倒的に多いので、そこだけ分岐を通しません
            if mode != BLEND_NORMAL:
                cs_r = blend_channel(mode, cb_r, cs_r)
                cs_g = blend_channel(mode, cb_g, cs_g)
                cs_b = blend_channel(mode, cb_b, cs_b)
            d_row[x, 0] = _u8((cs_r * sa + cb_r * da * inv) / out_a)
            d_row[x, 1] = _u8((cs_g * sa + cb_g * da * inv) / out_a)
            d_row[x, 2] = _u8((cs_b * sa + cb_b * da * inv) / out_a)
            d_row[x, 3] = _u8(out_a * 255.0)


@njit(cache=True, fastmath=True, inline="always")
def _add_span(coverage, y, width, xa, xb, weight):
    """1 本の «塗る区間» をカバレッジに足す。両端は面積で按分します。

    横方向は «画素にどれだけ掛かったか» を面積で正確に出すので、
    縦の 4 倍サンプリングと合わせて滑らかな縁になります。
    """
    x0 = xa
    x1 = xb
    if x1 <= x0:
        return
    if x1 <= 0.0 or x0 >= width:
        return
    if x0 < 0.0:
        x0 = 0.0
    if x1 > width:
        x1 = np.float64(width)
    first = int(math.floor(x0))
    last = int(math.floor(x1 - 1e-9))
    if first == last:
        coverage[y, first] += (x1 - x0) * weight
        return
    coverage[y, first] += (first + 1.0 - x0) * weight
    for x in range(first + 1, last):
        coverage[y, x] += weight
    coverage[y, last] += (x1 - last) * weight


@njit(cache=True, fastmath=True)
def rasterize_contours_kernel(verts, offsets, width, height, evenodd, coverage, bbox):
    """輪郭の集まりからカバレッジ（被覆率）を作る。

    :param verts: 全輪郭をつないだ `[x0, y0, x1, y1, ...]`（float64）
    :param offsets: 輪郭の切れ目。`verts` の «float 単位» の添字（長さは輪郭数 + 1）
    :param evenodd: 1 なら even-odd、0 なら nonzero
    :param coverage: `(height, width)` の float32。**呼ぶ側で 0 埋めしておくこと**
    :param bbox: 長さ 4 の int64。`[minX, minY, maxX, maxY]` が返ります

    輪郭は常に «閉じている» ものとして扱います（最後の点から最初の点へも辺を張る）。
    水平な辺は交点を作らないので捨てます。
    """
    n_contours = offsets.shape[0] - 1

    # ── 1. 辺を数える（配列を 1 回で確保するため） ──────────────────
    total = 0
    for c in range(n_contours):
        count = (offsets[c + 1] - offsets[c]) // 2
        if count < 2:
            continue
        total += count
    if total == 0:
        bbox[0] = 0
        bbox[1] = 0
        bbox[2] = -1
        bbox[3] = -1
        return

    ex0 = np.empty(total, np.float64)
    ey0 = np.empty(total, np.float64)
    ex1 = np.empty(total, np.float64)
    ey1 = np.empty(total, np.float64)
    edir = np.empty(total, np.int64)
    elo = np.empty(total, np.float64)
    ehi = np.empty(total, np.float64)

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    n_edges = 0

    for c in range(n_contours):
        base = offsets[c]
        count = (offsets[c + 1] - base) // 2
        if count < 2:
            continue
        for i in range(count):
            x0 = verts[base + i * 2]
            y0 = verts[base + i * 2 + 1]
            j = (i + 1) % count
            x1 = verts[base + j * 2]
            y1 = verts[base + j * 2 + 1]
            if y0 == y1:
                continue
            ex0[n_edges] = x0
            ey0[n_edges] = y0
            ex1[n_edges] = x1
            ey1[n_edges] = y1
            edir[n_edges] = 1 if y1 > y0 else -1
            if y0 < y1:
                elo[n_edges] = y0
                ehi[n_edges] = y1
            else:
                elo[n_edges] = y1
                ehi[n_edges] = y0
            n_edges += 1
            if elo[n_edges - 1] < min_y:
                min_y = elo[n_edges - 1]
            if ehi[n_edges - 1] > max_y:
                max_y = ehi[n_edges - 1]
            if min(x0, x1) < min_x:
                min_x = min(x0, x1)
            if max(x0, x1) > max_x:
                max_x = max(x0, x1)

    if n_edges == 0:
        bbox[0] = 0
        bbox[1] = 0
        bbox[2] = -1
        bbox[3] = -1
        return

    y_start = max(0, int(math.floor(min_y)))
    y_end = min(height - 1, int(math.ceil(max_y)))
    weight = 1.0 / SUBSAMPLES

    xs = np.empty(n_edges, np.float64)
    ds = np.empty(n_edges, np.int64)

    # ── 2. 走査線 ────────────────────────────────────────────────
    for y in range(y_start, y_end + 1):
        for s in range(SUBSAMPLES):
            sy = y + (s + 0.5) / SUBSAMPLES
            m = 0
            for e in range(n_edges):
                if sy < elo[e] or sy >= ehi[e]:
                    continue
                t = (sy - ey0[e]) / (ey1[e] - ey0[e])
                xs[m] = ex0[e] + (ex1[e] - ex0[e]) * t
                ds[m] = edir[e]
                m += 1
            if m < 2:
                continue
            # **安定な**挿入ソート。JS の Array.sort は安定なので、x が同じ交点の
            # 順が入れ替わると nonzero の巻き数が変わって絵が変わります。
            for a in range(1, m):
                kx = xs[a]
                kd = ds[a]
                b = a - 1
                while b >= 0 and xs[b] > kx:
                    xs[b + 1] = xs[b]
                    ds[b + 1] = ds[b]
                    b -= 1
                xs[b + 1] = kx
                ds[b + 1] = kd

            if evenodd != 0:
                a = 0
                while a + 1 < m:
                    _add_span(coverage, y, width, xs[a], xs[a + 1], weight)
                    a += 2
            else:
                winding = 0
                for a in range(m - 1):
                    winding += ds[a]
                    if winding != 0:
                        _add_span(coverage, y, width, xs[a], xs[a + 1], weight)

    bbox[0] = max(0, int(math.floor(min_x)))
    bbox[1] = y_start
    bbox[2] = min(width - 1, int(math.ceil(max_x)))
    bbox[3] = y_end


# ══════════════════════════════════════════════════════════════════
# カバレッジを通して色を乗せる
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True)
def fill_coverage_solid(data, coverage, min_x, min_y, max_x, max_y, r, g, b, a, alpha, mode):
    """カバレッジを通して 1 色を乗せる。`a` と `alpha` は 0..1。"""
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cov = coverage[y, x]
            if cov <= 0.0005:
                continue
            if cov > 1.0:
                cov = np.float32(1.0)
            sa = cov * a * alpha
            if sa <= 0.0:
                continue
            blend_pixel(data, y, x, r, g, b, sa, mode)


@njit(cache=True, fastmath=True)
def fill_coverage_rgba(data, coverage, min_x, min_y, max_x, max_y, colors, alpha, mode):
    """カバレッジを通して «画素ごとに違う色» を乗せる（グラデーション用）。

    :param colors: `(max_y - min_y + 1, max_x - min_x + 1, 4)` の float64。
        RGB は 0..255、A は 0..1。囲む矩形のぶんだけ渡します
    """
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cov = coverage[y, x]
            if cov <= 0.0005:
                continue
            if cov > 1.0:
                cov = np.float32(1.0)
            ly = y - min_y
            lx = x - min_x
            sa = cov * colors[ly, lx, 3] * alpha
            if sa <= 0.0:
                continue
            blend_pixel(data, y, x, colors[ly, lx, 0], colors[ly, lx, 1], colors[ly, lx, 2], sa, mode)


# ══════════════════════════════════════════════════════════════════
# カバレッジの量子化（ドット絵風の文字）
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True)
def quantize_grid(coverage, width, height, x0, y0, x1, y1, grid, hard):
    """グリッド内の被覆率を平均して字形そのものを «ドット» に丸める。

    平均が半分を超えたドットだけが «点く» ので、面積で決まります。線の細い字も
    消えずに残ります。
    """
    cell_y = y0
    while cell_y <= y1:
        cell_x = x0
        while cell_x <= x1:
            end_y = min(height - 1, cell_y + grid - 1)
            end_x = min(width - 1, cell_x + grid - 1)
            total = 0.0
            cells = 0
            for y in range(cell_y, end_y + 1):
                for x in range(cell_x, end_x + 1):
                    total += coverage[y, x]
                    cells += 1
            value = 0.0
            if cells > 0:
                value = total / cells
                if value < 0.0:
                    value = 0.0
                elif value > 1.0:
                    value = 1.0
            if hard != 0:
                filled = 1.0 if value >= 0.5 else 0.0
            else:
                filled = value
            for y in range(cell_y, end_y + 1):
                for x in range(cell_x, end_x + 1):
                    coverage[y, x] = filled
            cell_x += grid
        cell_y += grid


@njit(cache=True, fastmath=True)
def quantize_hard(coverage, min_x, min_y, max_x, max_y):
    """被覆率を 0/1 に丸めて中間調を無くす（`antialias: false`）。"""
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            coverage[y, x] = np.float32(1.0) if coverage[y, x] >= 0.5 else np.float32(0.0)


# ══════════════════════════════════════════════════════════════════
# テクスチャ付き三角形（メッシュ変形・3D）
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True, inline="always")
def _sample_bilinear(src, x, y, clamp_edge, out):
    """双一次補間。**アルファで重み付けしてから割り戻します。**

    そのまま補間すると、透明な画素に入っている «見えない色» が縁に滲み出します。
    """
    w = src.shape[1]
    h = src.shape[0]
    px = x - 0.5
    py = y - 0.5
    x0 = int(math.floor(px))
    y0 = int(math.floor(py))
    fx = px - x0
    fy = py - y0
    r = 0.0
    g = 0.0
    b = 0.0
    a = 0.0
    for dy in range(2):
        for dx in range(2):
            sx = x0 + dx
            sy = y0 + dy
            wx = fx if dx else 1.0 - fx
            wy = fy if dy else 1.0 - fy
            weight = wx * wy
            if weight == 0.0:
                continue
            if sx < 0 or sy < 0 or sx >= w or sy >= h:
                if clamp_edge == 0:
                    continue
                sx = min(max(sx, 0), w - 1)
                sy = min(max(sy, 0), h - 1)
            pa = np.float64(src[sy, sx, 3])
            r += src[sy, sx, 0] * pa * weight
            g += src[sy, sx, 1] * pa * weight
            b += src[sy, sx, 2] * pa * weight
            a += pa * weight
    if a > 0.0001:
        out[0] = r / a
        out[1] = g / a
        out[2] = b / a
        out[3] = a
    else:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0


@njit(cache=True, fastmath=PARITY_FASTMATH)
def draw_textured_triangle_kernel(
    dst,
    src,
    vx,
    vy,
    vu,
    vv,
    alpha,
    mode,
    clamp_edge,
    tint_r,
    tint_g,
    tint_b,
    tint_a,
    use_tint,
    depth_buffer,
    vz,
    use_depth,
    depth_test,
    depth_write,
):
    """テクスチャを貼った三角形を描く（`vx` `vy` `vu` `vv` `vz` は長さ 3）。

    **左上規則**（top-left fill rule）で塗るので、隣り合う三角形の共有辺が
    二重に塗られたり隙間が空いたりしません。
    """
    area = (vx[1] - vx[0]) * (vy[2] - vy[0]) - (vx[2] - vx[0]) * (vy[1] - vy[0])
    if abs(area) < 1e-9:
        return

    # 辺の関数の符号をそろえるため、反時計回りに並べ直す
    i0 = 0
    if area < 0:
        i1 = 2
        i2 = 1
    else:
        i1 = 1
        i2 = 2

    x0 = vx[i0]
    y0 = vy[i0]
    x1 = vx[i1]
    y1 = vy[i1]
    x2 = vx[i2]
    y2 = vy[i2]
    u0 = vu[i0]
    uu1 = vu[i1]
    u2 = vu[i2]
    v0 = vv[i0]
    vv1 = vv[i1]
    v2 = vv[i2]

    double_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    inv_area = 1.0 / double_area

    width = dst.shape[1]
    height = dst.shape[0]
    min_x = max(0, int(math.floor(min(x0, min(x1, x2)))))
    max_x = min(width - 1, int(math.ceil(max(x0, max(x1, x2)))))
    min_y = max(0, int(math.floor(min(y0, min(y1, y2)))))
    max_y = min(height - 1, int(math.ceil(max(y0, max(y1, y2)))))
    if min_x > max_x or min_y > max_y:
        return

    # 左上規則の «ずらし幅»。辺の関数の誤差（面積に比例）より大きく、
    # 1 画素の被覆率よりずっと小さくないと、隣り合う四角形の間に髪の毛のような
    # 隙間が残ります。
    epsilon = (abs(double_area) + 1.0) * 1e-9
    top_left_01 = (y0 == y1 and x1 < x0) or y1 < y0
    top_left_12 = (y1 == y2 and x2 < x1) or y2 < y1
    top_left_20 = (y2 == y0 and x0 < x2) or y0 < y2
    bias01 = epsilon if top_left_01 else -epsilon
    bias12 = epsilon if top_left_12 else -epsilon
    bias20 = epsilon if top_left_20 else -epsilon

    dz0 = 0.0
    dz1 = 0.0
    dz2 = 0.0
    if use_depth != 0:
        dz0 = vz[0]
        dz1 = vz[2] if area < 0 else vz[1]
        dz2 = vz[1] if area < 0 else vz[2]

    sample = np.empty(4, np.float64)

    for y in range(min_y, max_y + 1):
        py = y + 0.5
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            w0 = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0) + bias01
            w1 = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) + bias12
            w2 = (x0 - x2) * (py - y2) - (px - x2) * (y0 - y2) + bias20
            if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                continue
            # 重心座標。w1 が v0、w2 が v1、w0 が v2 に対応します
            l0 = w1 * inv_area
            l1 = w2 * inv_area
            l2 = w0 * inv_area
            # 深度テストは «色を作る前» に。奥だと分かった画素でテクスチャを
            # 引くのは無駄なので。
            z = 0.0
            if use_depth != 0:
                z = dz0 * l0 + dz1 * l1 + dz2 * l2
                if depth_test != 0 and z >= depth_buffer[y, x]:
                    continue
            u = u0 * l0 + uu1 * l1 + u2 * l2
            v = v0 * l0 + vv1 * l1 + v2 * l2
            _sample_bilinear(src, u, v, clamp_edge, sample)
            sa = (sample[3] / 255.0) * alpha
            if sa <= 0.0005:
                continue
            r = sample[0]
            g = sample[1]
            b = sample[2]
            if use_tint != 0:
                r = r * (1.0 - tint_a) + tint_r * tint_a
                g = g * (1.0 - tint_a) + tint_g * tint_a
                b = b * (1.0 - tint_a) + tint_b * tint_a
            if sa > 1.0:
                sa = 1.0
            blend_pixel(dst, y, x, r, g, b, sa, mode)
            # 半透明の画素で深度を書くと «向こう側が永久に描けない» ので、
            # ほぼ不透明なときだけ記録する。
            if depth_write != 0 and sa > 0.98:
                depth_buffer[y, x] = z


# ══════════════════════════════════════════════════════════════════
# 線を «塗れる形» に変える
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True)
def stroke_to_contours_kernel(points, half, closed, out_verts, out_offsets):
    """折れ線を «辺の四角形 + 継ぎ目の円» の輪郭に変える。

    **回り方（winding）を全部そろえるのが要点です。**（Movo の issue #74）
    素直に (p0+n) → (p1+n) → (p1-n) → (p0-n) と並べると時計回りになり、
    反時計回りの継ぎ目の円と重なったところで nonzero 塗りが打ち消し合って
    **穴が開きます**。折れが粗いうちは目立ちませんが、円弧やトリムした線のように
    細かく折れると点線状に見えます。

    :param out_verts: 呼ぶ側が確保する受け皿（`stroke_capacity()` で寸法を出す）
    :param out_offsets: 輪郭の切れ目
    :returns: 実際に作った輪郭の数
    """
    count = points.shape[0] // 2
    n_out = 0
    cursor = 0
    out_offsets[0] = 0
    if count < 2:
        return 0
    segments = count if closed != 0 else count - 1
    for i in range(segments):
        j = (i + 1) % count
        x0 = points[i * 2]
        y0 = points[i * 2 + 1]
        x1 = points[j * 2]
        y1 = points[j * 2 + 1]
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            continue
        nx = (-dy / length) * half
        ny = (dx / length) * half
        out_verts[cursor + 0] = x0 + nx
        out_verts[cursor + 1] = y0 + ny
        out_verts[cursor + 2] = x0 - nx
        out_verts[cursor + 3] = y0 - ny
        out_verts[cursor + 4] = x1 - nx
        out_verts[cursor + 5] = y1 - ny
        out_verts[cursor + 6] = x1 + nx
        out_verts[cursor + 7] = y1 + ny
        cursor += 8
        n_out += 1
        out_offsets[n_out] = cursor
        # 継ぎ目と端の丸みは小さな多角形で近似する
        for k in range(8):
            angle = (k / 8.0) * 2.0 * math.pi
            out_verts[cursor + k * 2] = x1 + math.cos(angle) * half
            out_verts[cursor + k * 2 + 1] = y1 + math.sin(angle) * half
        cursor += 16
        n_out += 1
        out_offsets[n_out] = cursor
        if i == 0:
            for k in range(8):
                angle = (k / 8.0) * 2.0 * math.pi
                out_verts[cursor + k * 2] = x0 + math.cos(angle) * half
                out_verts[cursor + k * 2 + 1] = y0 + math.sin(angle) * half
            cursor += 16
            n_out += 1
            out_offsets[n_out] = cursor
    return n_out


# ══════════════════════════════════════════════════════════════════
# ぼかし（影・グロー）
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True)
def blur_axis_kernel(data, out, radius, horizontal):
    """1 軸だけボックスぼかし。**アルファで重み付けしてから割り戻します。**

    ガウスではなくボックスを重ねる方式なのは、半径に対して線形時間で済み、
    3 回重ねればガウスとほぼ見分けが付かないからです。
    """
    height = data.shape[0]
    width = data.shape[1]
    window = radius * 2 + 1
    for y in range(height):
        for x in range(width):
            sum_r = 0.0
            sum_g = 0.0
            sum_b = 0.0
            sum_a = 0.0
            for k in range(-radius, radius + 1):
                if horizontal != 0:
                    sx = min(max(x + k, 0), width - 1)
                    sy = y
                else:
                    sx = x
                    sy = min(max(y + k, 0), height - 1)
                alpha = np.float64(data[sy, sx, 3])
                sum_r += data[sy, sx, 0] * alpha
                sum_g += data[sy, sx, 1] * alpha
                sum_b += data[sy, sx, 2] * alpha
                sum_a += alpha
            if sum_a > 0.0:
                out[y, x, 0] = _u8(sum_r / sum_a)
                out[y, x, 1] = _u8(sum_g / sum_a)
                out[y, x, 2] = _u8(sum_b / sum_a)
                out[y, x, 3] = _u8(sum_a / window)


# ══════════════════════════════════════════════════════════════════
# カラオケ塗り
# ══════════════════════════════════════════════════════════════════


@njit(cache=True, fastmath=True)
def karaoke_fill_kernel(src, out, base_r, base_g, base_b, fill_r, fill_g, fill_b, edge, softness):
    """左から順に «塗り色» へ入れ替えていく（カラオケ）。

    縁取り（塗り色から遠い色）は塗り替えません。`base_*` は塗り替えてよい
    «地の色» の一覧で、リッチテキストで色を変えた語のぶんだけ増えます。
    """
    height = src.shape[0]
    width = src.shape[1]
    n_base = base_r.shape[0]
    for y in range(height):
        for x in range(width):
            if src[y, x, 3] <= 4:
                continue
            if softness > 0.0:
                weight = (edge - x) / softness
                if weight < 0.0:
                    weight = 0.0
                elif weight > 1.0:
                    weight = 1.0
            else:
                weight = 1.0 if x <= edge else 0.0
            if weight <= 0.0:
                continue
            sr = np.float64(src[y, x, 0])
            sg = np.float64(src[y, x, 1])
            sb = np.float64(src[y, x, 2])
            near = False
            for i in range(n_base):
                dr = sr - base_r[i]
                dg = sg - base_g[i]
                db = sb - base_b[i]
                if math.sqrt(dr * dr + dg * dg + db * db) <= 90.0:
                    near = True
                    break
            if not near:
                continue
            out[y, x, 0] = _u8(sr + (fill_r - sr) * weight)
            out[y, x, 1] = _u8(sg + (fill_g - sg) * weight)
            out[y, x, 2] = _u8(sb + (fill_b - sb) * weight)


# ══════════════════════════════════════════════════════════════════
# 準備運動
# ══════════════════════════════════════════════════════════════════


def warmup() -> None:
    """よく使うカーネルを 1 度だけ小さな入力で走らせてコンパイルさせる。

    `cache=True` を付けてあるので、**一度走らせればディスクに残り**、
    次のプロセスでは待ち時間が消えます。並列レンダリングの前に呼んでおくと、
    全ワーカーが同時に 1 秒待つのを避けられます。
    """
    verts = np.array([0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0], np.float64)
    offsets = np.array([0, 8], np.int64)
    cov = np.zeros((8, 8), np.float32)
    bbox = np.zeros(4, np.int64)
    rasterize_contours_kernel(verts, offsets, 8, 8, 0, cov, bbox)
    data = np.zeros((8, 8, 4), np.uint8)
    fill_coverage_solid(data, cov, 0, 0, 7, 7, 255.0, 255.0, 255.0, 1.0, 1.0, 0)
    colors = np.ones((8, 8, 4), np.float64)
    fill_coverage_rgba(data, cov, 0, 0, 7, 7, colors, 1.0, 0)
    quantize_hard(cov, 0, 0, 7, 7)
    quantize_grid(cov, 8, 8, 0, 0, 7, 7, 2, 0)
    out = np.zeros((8, 8, 4), np.uint8)
    blur_axis_kernel(data, out, 1, 1)
    composite_bitmap_kernel(data, out, 1, 1, 1.0, 0)
    pts = np.array([0.0, 0.0, 4.0, 4.0], np.float64)
    ov = np.zeros(stroke_capacity(2, False), np.float64)
    oo = np.zeros(stroke_offsets_size(2, False), np.int64)
    stroke_to_contours_kernel(pts, 1.0, 0, ov, oo)
    vx = np.array([0.0, 4.0, 0.0], np.float64)
    vy = np.array([0.0, 0.0, 4.0], np.float64)
    vu = np.array([0.0, 4.0, 0.0], np.float64)
    vz = np.zeros(3, np.float64)
    depth = np.zeros((8, 8), np.float32)
    draw_textured_triangle_kernel(
        data, data, vx, vy, vu, vy, 1.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, depth, vz, 0, 0, 0
    )
    br = np.zeros(1, np.float64)
    karaoke_fill_kernel(data, out, br, br, br, 0.0, 0.0, 0.0, 1.0, 0.0)


def stroke_capacity(point_count: int, closed: bool) -> int:
    """`stroke_to_contours_kernel` の `out_verts` に要る長さ（float の個数）。"""
    segments = point_count if closed else max(0, point_count - 1)
    # 1 辺につき 四角形(8) + 円(16)、最初の 1 辺だけもう 1 つ円(16)
    return segments * 24 + 16


def stroke_offsets_size(point_count: int, closed: bool) -> int:
    """`stroke_to_contours_kernel` の `out_offsets` に要る長さ。"""
    segments = point_count if closed else max(0, point_count - 1)
    return segments * 3 + 2
