"""カラーグレーディング。

`colorAdjust` は «絵全体を同じだけ» 動かすので、「シャドウだけ青に寄せる」
「ハイライトだけ落とす」「空の水色だけ転ばす」ができません。ここでその 4 つを
足します。

    curves        … 明るさの «どこを» どう動かすかを曲線で描く
    colorWheels   … リフト（暗部）／ガンマ（中間）／ゲイン（明部）で三分割
    hslSecondary  … 色・彩度・明るさで «選んで» からそこだけ動かす
    lut           … `.cube` のルックを 1 枚当てる

## 値の単位（既存とちぐはぐにしないための決めごと）

`colorAdjust` の値は «増減量» で **0 が «変化なし»** です。ここでも

- `curves` の制御点は 0..1 の «入力 → 出力»（対角線が変化なし）
- `lift` は足し算（0 が変化なし）
- `hslSecondary.shift.hue` は度の足し算、`shift.sat` は `s × (1 + shift.sat)`

と «0 が変化なし» で揃えました。**そろえなかったのは `gamma` と `gain`** で、
こちらは 1 が «変化なし» です。リフト/ガンマ/ゲインは色屋の共通語で、
`gain: 1.05` を «+5%» と読む人に `gain: 0.05` を強いると事故になるからです。

## 速さ

`curves` と `colorWheels` は **チャンネルの値«だけ» で答えが決まります。**
8bit の入力は 256 通りしかないので、先に 256 段の表へ焼いてから 1 パスで引きます。
`hslSecondary` と `lut` は画素ごとに違う答えになりますが、どちらも NumPy の
一括演算に落ちます（画素ごとのループは書いていません）。

## `.cube` の安全について（大事）

LUT は «外からもらう» 素材です。他人の書いた `.cube` を読む前提なので、

- テキストの大きさに上限（{@link DEFAULT_MAX_LUT_BYTES}）
- **`LUT_3D_SIZE` そのものに上限**（{@link MAX_LUT_3D_SIZE}）

の二段構えにしています。**行数の上限だけでは守れません。** `LUT_3D_SIZE 2000`
の 1 行で 2000³ × 3 個の配列を先に確保してしまい、そこで落ちるからです。
つまり **配列を確保する «前» に大きさを見ます**（JS 版と同じ守り）。
"""

from __future__ import annotations

import math

import numpy as np

from movo.core.bitmap import Bitmap

# `.cube` の読み込みと当てはめは core にあります。**ここで作り直しません。**
from movo.core.lut import (
    DEFAULT_MAX_LUT_BYTES,
    MAX_LUT_3D_SIZE,
    Lut3D,
    apply_lut,
    identity_lut,
    parse_cube_lut,
    sample_lut,
)
from movo.renderer.effects import _u8, clamp, hsl_to_rgb, rgb_to_hsl

#: 8bit の入力は 256 通り。対応表の段数はこれで足ります。
TABLE_SIZE = 256


# ── 256 段の対応表 ────────────────────────────────────────────────

def _apply_channel_tables(bitmap: Bitmap, tables, amount: float) -> Bitmap:
    """チャンネルごとの 256 段の表を 1 パスで当てる。

    `amount` は原画との混ぜ量です（0 で無変化、1 で焼き切り）。
    表引きは «添字にそのまま uint8 を使う» だけなので、全画面でも一瞬です。
    """
    mix = clamp(amount, 0, 1)
    out = bitmap.copy()
    if mix <= 0:
        return out
    source = bitmap.data[..., :3].astype(np.float64)
    mapped = np.stack([tables[c][bitmap.data[..., c]] for c in range(3)], axis=-1)
    out.data[..., :3] = _u8(source + (mapped - source) * mix)
    return out


def _lookup(table: np.ndarray, value: float) -> float:
    """表を «整数でない位置» でも引く（表を重ねがけするとき用）。"""
    x = clamp(value, 0, TABLE_SIZE - 1)
    i = math.floor(x)
    if i >= TABLE_SIZE - 1:
        return float(table[TABLE_SIZE - 1])
    return float(table[i] + (table[i + 1] - table[i]) * (x - i))


def _channel_triple(value, fallback: float):
    """`{r,g,b}` / 数値 / `[r,g,b]` のどれで書かれていても 3 要素に開く。

    数値ひとつなら «3 チャンネルとも同じ» という意味にします。
    """
    out = [fallback, fallback, fallback]
    if value is None:
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)] * 3
    if isinstance(value, (list, tuple)):
        for c in range(3):
            if c < len(value) and isinstance(value[c], (int, float)):
                out[c] = float(value[c])
        return out
    if isinstance(value, dict):
        for c, keys in enumerate((("r", "red"), ("g", "green"), ("b", "blue"))):
            found = value.get(keys[0], value.get(keys[1]))
            if isinstance(found, (int, float)):
                out[c] = float(found)
    return out


def build_curve_table(points) -> np.ndarray | None:
    """単調 3 次スプライン（Fritsch–Carlson）を 256 段の表に焼く。

    **ふつうの 3 次スプラインだと制御点の間で «行き過ぎ» が出ます。** トーン
    カーブでそれが起きると、暗部を持ち上げただけなのに途中がへこんで «縞» に
    見えます。Fritsch–Carlson は接線を縮めて単調性を守るので、そうなりません。

    :param points: `[[x, y], ...]`（0..1）。2 点未満なら `None`（＝何もしない）
    """
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return None
    parsed = []
    for p in points:
        try:
            if isinstance(p, (list, tuple)):
                x, y = float(p[0]), float(p[1])
            elif isinstance(p, dict):
                x, y = float(p.get("x")), float(p.get("y"))
            else:
                continue
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            parsed.append((x, y))
    parsed.sort(key=lambda p: p[0])

    xs: list[float] = []
    ys: list[float] = []
    for x, y in parsed:
        # x が重なった点は後に書いたほうを残す（同じ x が 2 つあると傾きが無限大）
        if xs and abs(x - xs[-1]) < 1e-6:
            ys[-1] = y
            continue
        xs.append(x)
        ys.append(y)
    n = len(xs)
    if n < 2:
        return None

    slope = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    tangent = [0.0] * n
    tangent[0] = slope[0]
    tangent[n - 1] = slope[n - 2]
    for i in range(1, n - 1):
        # 向きが変わるところは接線 0（山や谷を作らない）
        tangent[i] = 0.0 if slope[i - 1] * slope[i] <= 0 else (slope[i - 1] + slope[i]) / 2
    for i in range(n - 1):
        if slope[i] == 0:
            tangent[i] = 0.0
            tangent[i + 1] = 0.0
            continue
        a = tangent[i] / slope[i]
        b = tangent[i + 1] / slope[i]
        magnitude = a * a + b * b
        if magnitude > 9:
            # 半径 3 の円に押し込むのが Fritsch–Carlson の条件です
            scale = 3 / math.sqrt(magnitude)
            tangent[i] = scale * a * slope[i]
            tangent[i + 1] = scale * b * slope[i]

    table = np.empty(TABLE_SIZE, np.float64)
    segment = 0
    for i in range(TABLE_SIZE):
        x = i / (TABLE_SIZE - 1)
        if x <= xs[0]:
            y = ys[0]
        elif x >= xs[n - 1]:
            y = ys[n - 1]
        else:
            while segment < n - 2 and x > xs[segment + 1]:
                segment += 1
            hstep = xs[segment + 1] - xs[segment]
            t = (x - xs[segment]) / hstep
            t2 = t * t
            t3 = t2 * t
            y = (
                (2 * t3 - 3 * t2 + 1) * ys[segment]
                + (t3 - 2 * t2 + t) * hstep * tangent[segment]
                + (-2 * t3 + 3 * t2) * ys[segment + 1]
                + (t3 - t2) * hstep * tangent[segment + 1]
            )
        table[i] = clamp(y, 0, 1) * (TABLE_SIZE - 1)
    return table


# ── 選択の重み（hslSecondary） ────────────────────────────────────

def _forward_angle(a: float, b: float) -> float:
    """角度 a から b へ «前向きに» 何度か（0..360）。"""
    return ((b - a) % 360 + 360) % 360


def _hue_weight(hue_deg: np.ndarray, rng, feather: float) -> np.ndarray:
    """色相の帯に入っているかを 0..1 で返す。**色相は輪なので `[340, 20]` も書けます。**"""
    if rng is None:
        return np.ones_like(hue_deg)
    width = _forward_angle(rng[0], rng[1])
    offset = ((hue_deg - rng[0]) % 360 + 360) % 360
    distance = np.minimum(offset - width, 360 - offset)
    outside = np.clip(1 - distance / feather, 0, 1) if feather > 0 else np.zeros_like(hue_deg)
    return np.where(offset <= width, 1.0, outside)


def _range_weight(value: np.ndarray, rng, feather: float) -> np.ndarray:
    """値の帯に入っているかを 0..1 で返す。"""
    if rng is None:
        return np.ones_like(value)
    low, high = rng
    distance = np.where(value < low, low - value, value - high)
    outside = np.clip(1 - distance / feather, 0, 1) if feather > 0 else np.zeros_like(value)
    return np.where((value >= low) & (value <= high), 1.0, outside)


def _pair_range(value):
    """`[a, b]` の形だけ受け取る（数値が入っていなければ «指定なし»）。"""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return (low, high)


# ── `.cube`（3D LUT） ─────────────────────────────────────────────
#
# 読み込みと当てはめは `movo.core.lut` にあります（**ここで作り直しません**）。
# あちらは `LUT_3D_SIZE` を «配列を確保する前に» 上限と突き合わせる守りが
# 入っていて、当てはめも Numba の 1 画素ずつ引く形なので、NumPy で 8 隅を
# gather するより 45 倍速い（410 ms → 9 ms）実装になっています。
#
# 名前だけここから引けるようにしておきます（`from ... import Lut3D` で
# エフェクト側の import 元が 1 つで済むように）。


def _resolve_lut(params: dict, ctx: dict):
    """lut エフェクトが使う LUT を決める。素材が無ければ `None`（＝何もしない）。"""
    given = params.get("lut")
    if isinstance(given, Lut3D):
        return given
    name = params.get("asset") or params.get("lutAsset") or params.get("name")
    if not name:
        return None
    assets = (ctx or {}).get("assets")
    getter = getattr(assets, "get_lut", None) if assets is not None else None
    return getter(name) if callable(getter) else None


# ── エフェクト本体 ────────────────────────────────────────────────

def curves(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """トーンカーブ。

        { "type": "curves",
          "rgb":  [[0, 0], [0.25, 0.18], [0.75, 0.82], [1, 1]],
          "blue": [[0, 0.04], [1, 0.94]] }

    上の `blue` は «暗部に青を足してハイライトの青を抜く»、フィルム調の定番です。
    掛ける順は **チャンネル → 全体**。全体のカーブは «最後の仕上げ» と読むほうが
    直感に合うからです。対角線（`[[0,0],[1,1]]`）が «変化なし» です。
    """
    master = build_curve_table(params.get("rgb", params.get("master")))
    channels = [
        build_curve_table(params.get("red", params.get("r"))),
        build_curve_table(params.get("green", params.get("g"))),
        build_curve_table(params.get("blue", params.get("b"))),
    ]
    if master is None and not any(c is not None for c in channels):
        return bitmap
    tables = []
    for c in range(3):
        table = np.empty(TABLE_SIZE, np.float64)
        for v in range(TABLE_SIZE):
            after = channels[c][v] if channels[c] is not None else v
            table[v] = _lookup(master, after) if master is not None else after
        tables.append(table)
    amount = params.get("amount")
    return _apply_channel_tables(bitmap, tables, 1 if amount is None else amount)


def color_wheels(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """リフト／ガンマ／ゲイン（色屋の «3 つのホイール»）。

        lift  … 暗部。**0 が変化なし。**足し算なので «黒を浮かせる» ときに使う
        gamma … 中間。**1 が変化なし。**1 より大きいと明るく
        gain  … 明部。**1 が変化なし。**倍率

    式は `out = lift + in × (gain - lift)` を当ててから `out ^ (1 / gamma)` です。
    前半は «黒を lift に、白を gain に置き直す» 直線なので、lift は暗部、gain は
    明部にだけ効きます。この形にしたのは、2 つを別々に動かしても白飛び・黒潰れが
    増えないからです。`offset` は最後に全体を足し引きします（3 つとは別枠）。
    """
    lift = _channel_triple(params.get("lift"), 0)
    gamma = _channel_triple(params.get("gamma"), 1)
    gain = _channel_triple(params.get("gain"), 1)
    offset = _channel_triple(params.get("offset"), 0)
    tables = []
    grid = np.arange(TABLE_SIZE, dtype=np.float64) / (TABLE_SIZE - 1)
    for c in range(3):
        # ガンマ 0 以下は 1 / gamma が壊れるので下限を置きます
        g = gamma[c] if gamma[c] > 0.01 else 0.01
        y = lift[c] + grid * (gain[c] - lift[c]) + offset[c]
        if g != 1:
            y = np.power(np.maximum(0.0, y), 1.0 / g)
        tables.append(np.clip(y, 0, 1) * (TABLE_SIZE - 1))
    amount = params.get("amount")
    return _apply_channel_tables(bitmap, tables, 1 if amount is None else amount)


def hsl_secondary(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """HSL セカンダリ。**色で選んでから、そこだけ動かします。**

        { "type": "hslSecondary",
          "select":   { "hue": [180, 220], "sat": [0.2, 1] },
          "softness": 0.15,
          "shift":    { "hue": -12, "sat": 0.3 } }

    `select` の 3 つ（`hue` は度、`sat` と `lum` は 0..1）はすべて «かつ» で、
    選び具合は一番きつい条件に合わせます。書かなかった条件は «全部» です。

    `softness` は羽根の «幅そのもの» ではなく **«選択範囲の幅に対する割合»** です。
    範囲 40 度に 0.15 なら 6 度。こうしておくと、狭く選んだときに羽根だけが
    大きく残って «選んだつもりが全体に効く» という事故が起きません。
    """
    select = params.get("select") or {}
    shift = params.get("shift") or {}
    amount = clamp(params.get("amount", 1) if params.get("amount") is not None else 1, 0, 1)
    if amount <= 0:
        return bitmap
    hue_range = _pair_range(select.get("hue"))
    sat_range = _pair_range(select.get("sat", select.get("saturation")))
    lum_range = _pair_range(select.get("lum", select.get("lightness")))
    softness = max(0, params.get("softness", 0) or 0)
    hue_feather = _forward_angle(hue_range[0], hue_range[1]) * softness if hue_range else 0
    sat_feather = (sat_range[1] - sat_range[0]) * softness if sat_range else 0
    lum_feather = (lum_range[1] - lum_range[0]) * softness if lum_range else 0
    hue_shift = (shift.get("hue", 0) or 0) / 360
    sat_shift = shift.get("sat", shift.get("saturation", 0)) or 0
    lum_shift = shift.get("lum", shift.get("lightness", 0)) or 0
    if hue_shift == 0 and sat_shift == 0 and lum_shift == 0:
        return bitmap

    base = bitmap.data[..., :3].astype(np.float64)
    hsl = rgb_to_hsl(base)
    h, s, lightness = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    weight = _hue_weight(h * 360, hue_range, hue_feather)
    weight = np.minimum(weight, _range_weight(s, sat_range, sat_feather))
    weight = np.minimum(weight, _range_weight(lightness, lum_range, lum_feather))
    weight = weight * amount

    moved = np.empty_like(hsl)
    moved[..., 0] = (h + hue_shift + 1) % 1
    moved[..., 1] = np.clip(s * (1 + sat_shift), 0, 1)
    moved[..., 2] = np.clip(lightness + lum_shift, 0, 1)
    target = hsl_to_rgb(moved)

    out = bitmap.copy()
    touched = weight > 0
    out.data[..., :3] = np.where(
        touched[..., None], _u8(base + (target - base) * weight[..., None]), bitmap.data[..., :3]
    )
    return out


def lut(bitmap: Bitmap, params: dict, ctx: dict | None = None) -> Bitmap:
    """3D LUT（`.cube`）を当てる。

        "assets": { "teal-orange": { "type": "lut", "path": "looks/teal-orange.cube" } }
        "effects": [{ "type": "lut", "asset": "teal-orange", "amount": 0.8 }]

    `movo batch` で 10 本のシリーズを作るとき、**ルック 1 枚で全部の色を揃えられる**
    のがこのエフェクトの値打ちです。素材が読めないときは «何もしない» で通します
    （色が付かないだけで、絵は出ます）。

    当てはめは `movo.core.lut.apply_lut`（Numba）に任せます。NumPy で 8 隅を
    gather すると 1280x720 で 410 ミリ秒、1 画素ずつ引くと 9 ミリ秒です。
    """
    table = _resolve_lut(params, ctx or {})
    if table is None:
        return bitmap
    amount = params.get("amount")
    mix = clamp(1 if amount is None else amount, 0, 1)
    out = bitmap.copy()
    if mix <= 0:
        return out
    base = bitmap.data[..., :3].astype(np.float64)
    mapped = apply_lut(base / 255.0, table).astype(np.float64) * 255.0
    out.data[..., :3] = _u8(base + (mapped - base) * mix)
    return out


#: `type` からグレーディング関数を引く表。`effects.py` の末尾で取り込まれます。
grade_effects = {
    "curves": curves,
    "colorWheels": color_wheels,
    "hslSecondary": hsl_secondary,
    "lut": lut,
}


def list_grade_effects() -> list[str]:
    """一覧（テストと `movo list effects` の確認用）。"""
    return sorted(grade_effects.keys())


__all__ = [
    "DEFAULT_MAX_LUT_BYTES",
    "MAX_LUT_3D_SIZE",
    "Lut3D",
    "apply_lut",
    "build_curve_table",
    "color_wheels",
    "curves",
    "grade_effects",
    "hsl_secondary",
    "identity_lut",
    "list_grade_effects",
    "lut",
    "parse_cube_lut",
    "sample_lut",
]
