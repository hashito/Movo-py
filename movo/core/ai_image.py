"""キャラクターの «一枚絵» を作って、パーツに切り出す。

## なぜ «一枚絵 → 切り出し» なのか

ポーズごとに別々に生成すると、**同じキャラに見えません**（髪の色も目の形も
毎回変わります）。1 枚の絵の中に並べて描かせれば、モデルは «同じキャラの
別のポーズ» として描くので、見た目が揃います。切り出しは格子の算術だけです。

この作りは `line-stamp-generator` プラグイン（`scripts/generate_image.py` /
`slice_image.py`）と同じ考え方で、そちらで実績のある «クロマキー緑の地 ＋
マゼンタの格子線» という指示をそのまま借りています。

## 依存

`openai` パッケージと API キーが要るときだけ読み込みます。**モジュールの
import では読みません** — キーが無い環境で `movo` 全体が動かなくなるのは
割に合わないためです。

## 決定性について

**生成そのものは決定的ではありません。** 同じ指示から毎回同じ絵は出ません。
そのため «生成は 1 回だけ、結果を PNG として保存して使い回す» 前提です
（`movo` の «同じ JSON からは同じ動画» は、保存済みの PNG を読む時点から
成り立ちます）。
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bitmap import Bitmap
from .errors import ErrorCodes, MovoError
from .logger import logger
from numba import njit

from .png import decode_png, encode_png

#: 地の色（クロマキー）。この色に近い画素を透明にします。
CHROMA = (0, 255, 0)

#: 地とみなす許容差（0〜255）。大きくすると髪の緑がかった影まで抜けます。
CHROMA_TOLERANCE = 78

#: 格子線の色。**これも抜きます。** 地の緑だけ抜くと、マゼンタの線が残って
#: «不透明な画素» になり、余白の刈り取り（`_trim`）がセル全体を «中身あり» と
#: 判定します（実際に 512x768 のまま刈られませんでした）。
GRID_LINE = (255, 0, 255)

#: 格子線用の許容差。**地の緑より大きく取ります。** 線は細いので周りと混ざり、
#: 実測では純マゼンタ #FF00FF ではなく [203, 67, 150] のような «くすんだ桃» に
#: なっていました（距離 135）。肌・髪・服はいずれもマゼンタから 220 以上
#: 離れているので、200 まで広げても巻き込みません（肌 248・髪 238・服 292）。
#: 150 では **地の緑と混ざって薄くなった線**（距離 190 前後）が残り、キャラの
#: 周りに赤い横線・縦線として出ました。
GRID_LINE_TOLERANCE = 200

#: 格子線のなだらかさ。**地の緑よりずっと狭く取ります。** 広いと «赤っぽい色»
#: （唇・頬・赤い服）がマゼンタの近くにあるせいで半透明になります。実測で
#: [200, 40, 40] の赤が alpha 166 まで削られました（マゼンタからの距離 223）。
#: 12 なら 212 以上の色はまるごと残ります。
GRID_LINE_SOFTNESS = 12

#: 既定のモデル。画像生成のモデル名は変わりやすいので、指定で上書きできます。
DEFAULT_MODEL = "gpt-image-1"


def resolve_api_key(explicit: str | None = None) -> str:
    """API キーを見つける。**値はログに出しません。**

    順に `引数` → `OPENAI_API_KEY` → `GPT_API_KEY` → `movo config` を見ます。
    """
    if explicit:
        return explicit
    for name in ("OPENAI_API_KEY", "GPT_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        from movo.cli.config_store import get_config_value  # 遅延読み込み（CLI に依存しないため）

        value = get_config_value("openai.apiKey")
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 - 設定が読めなくても «無い» として扱う
        pass
    raise MovoError(
        ErrorCodes.MOVO_ASSET_NOT_FOUND,
        "OpenAI の API キーが見つかりません",
        hint="movo config set openai.apiKey <キー> か、環境変数 OPENAI_API_KEY を設定してください",
    )


def build_prompt(spec: dict[str, Any]) -> str:
    """仕様から生成の指示文を作る。

    **格子を明示するのが要点です。** «適当に並べて» と言うと配置が毎回変わり、
    切り出しの座標が当たりません。行と列、セルの大きさ、地の色、格子線の色を
    数値で指定します。
    """
    grid = spec["grid"]
    cols, rows = grid["cols"], grid["rows"]
    cell_w, cell_h = grid["cellSize"]
    width, height = cols * cell_w, rows * cell_h

    lines = [
        spec["theme"].strip(),
        "",
        f"キャンバスは {width}x{height} ピクセル。{cols} 列 x {rows} 行の格子に分けて描く。",
        f"1 マスは {cell_w}x{cell_h} ピクセル。",
        "",
        "**厳守すること**",
        "- 地（背景）は完全に均一な純緑 #00FF00 で塗る。グラデーション・影・模様は禁止。",
        "- 格子の境界線は太さ 4px の純マゼンタ #FF00FF でまっすぐ引く。",
        "- 各マスの中身は境界線から 24px 以上離す（切り出しで欠けるため）。",
        "- **全てのマスで同一人物**。髪型・髪色・目の形・服の色を変えない。",
        "- 文字・ロゴ・透かしは一切入れない。",
        "",
        "**各マスの中身**",
    ]
    for index, part in enumerate(spec["parts"], start=1):
        col, row = part["cell"]
        text = part.get("description") or part["name"]
        lines.append(f"{index}. 列 {col + 1} 行 {row + 1}: {text}")
    return "\n".join(lines)


def generate_sheet(spec: dict[str, Any], *, api_key: str | None = None, model: str | None = None) -> Bitmap:
    """指示を投げて «一枚絵» を受け取る。

    ⚠ **外部の有料 API を呼びます。** 1 回で 1 枚です。
    """
    try:
        from openai import OpenAI
    except ImportError as error:
        raise MovoError(
            ErrorCodes.MOVO_RENDERER_UNAVAILABLE,
            "openai パッケージが入っていません",
            hint="pip install openai",
            cause=error,
        ) from error

    grid = spec["grid"]
    width = grid["cols"] * grid["cellSize"][0]
    height = grid["rows"] * grid["cellSize"][1]

    client = OpenAI(api_key=resolve_api_key(api_key))
    logger.info(f"画像を生成しています（{width}x{height}）… 外部 API を呼びます")
    response = client.images.generate(
        model=model or spec.get("model") or DEFAULT_MODEL,
        prompt=build_prompt(spec),
        size=f"{width}x{height}",
        n=1,
    )
    payload = response.data[0]
    raw = base64.b64decode(payload.b64_json) if getattr(payload, "b64_json", None) else None
    if raw is None:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "画像が返ってきませんでした")
    return decode_png(raw) if raw[:8] == b"\x89PNG\r\n\x1a\n" else _decode_any(raw)


def _decode_any(raw: bytes) -> Bitmap:
    from .jpeg import decode_jpeg

    return decode_jpeg(raw)


def cut_out_chroma(
    bitmap: Bitmap, *, tolerance: int = CHROMA_TOLERANCE, extra: tuple = (GRID_LINE,)
) -> Bitmap:
    """地の緑（と格子線）を透明にする。

    «近い色» の判定は 3 チャンネルの距離です。緑の明度だけを見ると、髪や服の
    緑まで抜けます。
    """
    out = bitmap.clone()
    # **int16 では溢れます。** (255-0)^2 * 3 = 195,075 で int16 の上限 32,767 を
    # 超え、負の値の平方根になって «判定が全部おかしくなる» という壊れ方をします
    # （実際に RuntimeWarning が出ました）。int32 で計算します。
    rgb = out.data[..., :3].astype(np.int32)
    distance = np.sqrt(((rgb - np.array(CHROMA, np.int32)) ** 2).sum(axis=2))
    # **境目をなだらかにします。** 0 か 255 かで切ると輪郭がギザギザになり、
    # 縮小したときに «階段» が目立ちます。地の色からの距離で 0〜1 に渡します。
    softness = max(1.0, tolerance * 0.5)
    ramp = np.clip((distance - tolerance) / softness, 0.0, 1.0)
    for color in extra:
        other = np.sqrt(((rgb - np.array(color, np.int32)) ** 2).sum(axis=2))
        ramp = np.minimum(ramp, np.clip((other - GRID_LINE_TOLERANCE) / GRID_LINE_SOFTNESS, 0.0, 1.0))
    out.data[..., 3] = (out.data[..., 3].astype(np.float64) * ramp).astype(np.uint8)
    return out


def bleed_edges(bitmap: Bitmap, *, passes: int = 3) -> Bitmap:
    """透明な画素の色を «隣の不透明な色» で埋める（縁の染み出し）。

    アルファを 0 にしても **RGB は緑のまま**です。そのまま縮小すると、縁の
    画素が «緑 x 透明» と «肌 x 不透明» の平均になり、**輪郭に緑が滲みます**。
    先に色だけ外へ広げておけば、混ざっても肌の色同士になります。
    """
    out = bitmap.clone()
    for _ in range(max(0, passes)):
        alpha = out.data[..., 3]
        solid = alpha > 8
        if solid.all() or not solid.any():
            break
        rgb = out.data[..., :3].astype(np.int32)
        weight = solid.astype(np.int32)
        total = np.zeros_like(rgb)
        count = np.zeros_like(weight)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            total += np.roll(np.roll(rgb * weight[..., None], dy, axis=0), dx, axis=1)
            count += np.roll(np.roll(weight, dy, axis=0), dx, axis=1)
        fill = (~solid) & (count > 0)
        if not fill.any():
            break
        averaged = np.zeros_like(rgb)
        safe = np.maximum(count, 1)[..., None]
        averaged = total // safe
        out.data[..., :3] = np.where(fill[..., None], averaged.astype(np.uint8), out.data[..., :3])
        # 埋めた画素は «次の回では隣» として使えるよう、ごく薄い alpha を与える
        out.data[..., 3] = np.where(fill, 9, out.data[..., 3])
    # 最後に染み出し用の薄い alpha を戻す（見た目には出さない）
    out.data[..., 3] = np.where(out.data[..., 3] == 9, 0, out.data[..., 3])
    return out


@njit(cache=True)
def _flood(mask, sy, sx):
    """種から 4 近傍で塗り広げる。**戻り値は塗れた画素の bool 配列**。

    連結成分のラベル付けは «1 つ前の結果が要る» 逐次処理なので、NumPy では
    素直に書けません（`scipy.ndimage.label` があれば 1 行ですが、この計画は
    numpy と numba だけで通す方針です）。ここは Numba の担当です。
    """
    height, width = mask.shape
    filled = np.zeros((height, width), np.bool_)
    stack_y = np.empty(height * width, np.int32)
    stack_x = np.empty(height * width, np.int32)
    top = 0
    stack_y[0] = sy
    stack_x[0] = sx
    filled[sy, sx] = True
    while top >= 0:
        y = stack_y[top]
        x = stack_x[top]
        top -= 1
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not filled[ny, nx]:
                filled[ny, nx] = True
                top += 1
                stack_y[top] = ny
                stack_x[top] = nx
    return filled


def keep_largest_blob(bitmap: Bitmap) -> Bitmap:
    """いちばん大きな «ひとつながり» だけを残す。

    細さで狙う開処理（`remove_thin_marks`）は、太めの線の断片を落とせません
    （実測で頭上に 8px の破片が残りました）。**キャラは 1 つの塊**なので、
    «中心にいちばん近い不透明な画素からつながっているところ» だけ残せば、
    離れた断片は形や太さに関係なく消えます。
    """
    mask = bitmap.data[..., 3] > 8
    if not mask.any():
        return bitmap
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    # 重心そのものが透明なこともあるので、いちばん近い不透明画素を種にします
    nearest = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    filled = _flood(mask, int(ys[nearest]), int(xs[nearest]))
    out = bitmap.clone()
    out.data[..., 3] = np.where(filled, out.data[..., 3], 0)
    return out


def remove_thin_marks(bitmap: Bitmap, *, radius: int = 2) -> Bitmap:
    """**細い線だけ**を消す（モルフォロジーの開処理）。

    格子線は色で狙っても取り切れません。細いぶん周りと混ざって、指定した色から
    離れた «くすんだ» 画素になるためです（実測で純マゼンタから距離 190 前後）。
    許容差を広げれば取れますが、今度は唇や頬の赤まで巻き込みます。

    そこで **形で見ます。** 線は «細くてまっすぐ»、キャラは «太い塊» なので、
    1 度縮めてから同じだけ膨らませる（開処理）と、縮めた時点で消えた細いものだけが
    戻ってきません。半径 2（5x5）はアホ毛や指（6〜10px）を残しつつ、
    3〜4px の線を落とせる大きさです。
    """
    alpha = bitmap.data[..., 3] > 8
    eroded = alpha.copy()
    for _ in range(radius):
        shrunk = eroded.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shrunk &= np.roll(np.roll(eroded, dy, axis=0), dx, axis=1)
        eroded = shrunk
    grown = eroded.copy()
    for _ in range(radius):
        spread = grown.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            spread |= np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
        grown = spread
    out = bitmap.clone()
    out.data[..., 3] = np.where(grown, out.data[..., 3], 0)
    return out


def _trim(bitmap: Bitmap, *, margin: int = 4) -> Bitmap:
    """透明な余白を落とす。パーツを «実物大» で扱えるようにするため。"""
    bounds = bitmap.alpha_bounds(8)
    if bounds is None:
        return bitmap
    left = max(0, bounds["x"] - margin)
    top = max(0, bounds["y"] - margin)
    right = min(bitmap.width, bounds["x"] + bounds["width"] + margin)
    bottom = min(bitmap.height, bounds["y"] + bounds["height"] + margin)
    return bitmap.crop(left, top, right - left, bottom - top)


def slice_sheet(
    bitmap: Bitmap, spec: dict[str, Any], out_dir: str | os.PathLike[str], *,
    trim: bool = True, keep_blob: bool = True,
) -> dict[str, str]:
    """格子どおりに切り出して PNG で保存する。

    :returns: パーツ名 → 保存したパス
    """
    grid = spec["grid"]
    cell_w, cell_h = grid["cellSize"]
    # 返ってくる絵が指定どおりの大きさとは限らないので、実寸から倍率を出します
    scale_x = bitmap.width / (grid["cols"] * cell_w)
    scale_y = bitmap.height / (grid["rows"] * cell_h)

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for part in spec["parts"]:
        col, row = part["cell"]
        span_c, span_r = part.get("span", [1, 1])
        left = int(round(col * cell_w * scale_x))
        top = int(round(row * cell_h * scale_y))
        right = int(round((col + span_c) * cell_w * scale_x))
        bottom = int(round((row + span_r) * cell_h * scale_y))
        piece = bitmap.crop(left, top, right - left, bottom - top)
        # ⚠ **切り出した «後» に掛けます。** シート全体に掛けると、いちばん大きな
        # 1 体だけが残って他のポーズが全部消えます（実際に全滅させました）。
        if keep_blob:
            piece = keep_largest_blob(piece)
        if trim:
            piece = _trim(piece)
        name = part["name"]
        path = directory / f"{name}.png"
        path.write_bytes(encode_png(piece))
        written[name] = str(path)
        logger.verbose(f"  {name}: {piece.width}x{piece.height} → {path}")
    return written


def generate_character_parts(
    spec_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    api_key: str | None = None,
    sheet_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """仕様 JSON から «一枚絵 → 透過 → 切り出し» までを通しでやる。

    `sheet_path` に既存の PNG を渡すと **生成せずに切り出しだけ**します。
    切り出しの座標を直したいだけのときに、無駄な課金をしないためです。
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if sheet_path and Path(sheet_path).exists():
        logger.info(f"既にある一枚絵を使います: {sheet_path}")
        sheet = decode_png(Path(sheet_path).read_bytes())
    else:
        sheet = generate_sheet(spec, api_key=api_key)
        if sheet_path:
            Path(sheet_path).parent.mkdir(parents=True, exist_ok=True)
            Path(sheet_path).write_bytes(encode_png(sheet))
            logger.info(f"一枚絵を保存しました: {sheet_path}")
    cut = cut_out_chroma(sheet, tolerance=spec.get("chromaTolerance", CHROMA_TOLERANCE))
    cut = remove_thin_marks(cut, radius=spec.get("thinRadius", 2))
    cut = bleed_edges(cut)
    return slice_sheet(cut, spec, out_dir)


__all__ = [
    "CHROMA",
    "CHROMA_TOLERANCE",
    "bleed_edges",
    "build_prompt",
    "cut_out_chroma",
    "generate_character_parts",
    "generate_sheet",
    "keep_largest_blob",
    "remove_thin_marks",
    "resolve_api_key",
    "slice_sheet",
]
