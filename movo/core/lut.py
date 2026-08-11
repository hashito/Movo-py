"""3D LUT（``.cube``）の読み込みと当てはめ。

``.cube`` は行区切りのテキストです。``LUT_3D_SIZE n`` のあとに n³ 行の RGB が
並ぶだけなので、依存を足さずに 1 ファイルで完結します。

並び順は **«赤が一番速く回る»**（r → g → b）と決まっています。つまり
``index = ((b * size + g) * size + r) * 3`` です。ここを取り違えると色が
«斜めに» 転ぶので、テストで格子の角を直接確かめています。

## なぜ 3D LUT を入れるか

``movo batch`` で 10 本のシリーズを作るとき、ルック 1 枚で全部の色を揃えられる
のが値打ちです。カーブやリフト/ガンマ/ゲインでは «この色だけこう転ばす» が
書けません。

## 安全について（大事）

LUT は **外からもらう素材**です。他人の書いた ``.cube`` を読む前提なので、

- 読み込むテキストの大きさに上限
- ``LUT_3D_SIZE`` そのものに上限（:data:`MAX_LUT_3D_SIZE`）

の二段構えにしています。行数の上限だけだと ``LUT_3D_SIZE 2000`` の 1 行で
2000³ × 3 個の配列を先に確保してしまい、そこで落ちるからです。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .errors import ErrorCodes, MovoError

#: 許す格子の細かさ。配られている ``.cube`` は 17 / 32 / 33 / 64 なので 64 で足ります。
#: （65 以上を弾くのは «性能» ではなく **«メモリを食い潰させない»** ためです）
MAX_LUT_3D_SIZE = 64

#: テキストの大きさの既定の上限。6 桁の数値で 64³ 行を書くと 7 MB ほどです。
DEFAULT_MAX_LUT_BYTES = 8 * 1024 * 1024


@dataclass
class Lut3D:
    """読み込んだ 3D LUT。

    ``data`` は ``(size, size, size, 3)`` の float32 で、軸の順は **[b, g, r]** です。
    JS 版は 1 次元の Float32Array ですが、当てはめを NumPy の索引 1 回で
    済ませるためにここでは 4 次元にしています。並び順の意味は同じです。
    """

    size: int
    title: str = ""
    domain_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    domain_max: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    data: np.ndarray = field(default_factory=lambda: np.zeros((2, 2, 2, 3), np.float32))

    @property
    def flat(self) -> np.ndarray:
        """JS 版と同じ 1 次元の並び（``((b*size+g)*size+r)*3``）。"""
        return self.data.reshape(-1)


_KEYWORD_RE = re.compile(r"^([A-Z_0-9]+)\s*(.*)$", re.IGNORECASE)


def parse_cube_lut(
    text: str, *, max_bytes: int | None = None, max_size: int | None = None, source: str | None = None
) -> Lut3D:
    """``.cube`` のテキストを読む。

    :param max_bytes: テキストの大きさの上限（既定 :data:`DEFAULT_MAX_LUT_BYTES`）
    :param max_size: 格子の細かさの上限（既定 :data:`MAX_LUT_3D_SIZE`）
    """
    limit_bytes = max(1, DEFAULT_MAX_LUT_BYTES if max_bytes is None else max_bytes)
    limit_size = max(2, min(MAX_LUT_3D_SIZE if max_size is None else max_size, MAX_LUT_3D_SIZE))
    where = f": {source}" if source else ""

    if not isinstance(text, str) or len(text) == 0:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"LUT が空です{where}")
    if len(text) > limit_bytes:
        raise MovoError(
            ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE,
            f"LUT が大きすぎます（{len(text) / 1024 / 1024:.1f} MB > {limit_bytes / 1024 / 1024:.1f} MB）{where}",
            hint="security.maxDownloadSizeMB を上げるか、格子の粗い .cube を使ってください",
        )

    size = 0
    title = ""
    values: np.ndarray | None = None
    count = 0
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]

    for raw in text.splitlines():
        line = re.sub(r"#.*$", "", raw).strip()  # `#` から後ろは注釈
        if not line:
            continue

        keyword = _KEYWORD_RE.match(line)
        name = keyword.group(1).upper() if keyword else ""
        if name == "TITLE":
            title = keyword.group(2).strip().strip('"')
            continue
        if name == "LUT_1D_SIZE":
            # 1D LUT は «チャンネルごとの曲線» でしかないので curves エフェクトの仕事です。
            # 中途半端に対応するより、はっきり断ったほうが親切です。
            raise MovoError(
                ErrorCodes.MOVO_UNSUPPORTED,
                f"1D LUT には対応していません{where}",
                hint="curves エフェクトで同じことができます",
            )
        if name == "LUT_3D_SIZE":
            try:
                size = int(keyword.group(2).strip().split()[0])
            except (ValueError, IndexError):
                raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"LUT_3D_SIZE が読めません{where}") from None
            if size < 2:
                raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"LUT_3D_SIZE が読めません{where}")
            if size > limit_size:
                raise MovoError(
                    ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE,
                    f"LUT_3D_SIZE が大きすぎます（{size} > {limit_size}）{where}",
                    hint=f"{limit_size} 以下の .cube に焼き直してください",
                )
            values = np.zeros(size * size * size * 3, np.float32)
            continue
        if name in ("DOMAIN_MIN", "DOMAIN_MAX"):
            target = domain_min if name == "DOMAIN_MIN" else domain_max
            parts = keyword.group(2).strip().split()
            for c in range(min(3, len(parts))):
                try:
                    target[c] = float(parts[c])
                except ValueError:
                    pass
            continue
        if name and not re.match(r"^[-+.\d]", line):
            continue  # 知らないキーワードは読み飛ばす

        if values is None:
            raise MovoError(
                ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"LUT_3D_SIZE より前に数値の行があります{where}"
            )
        parts = line.split()
        if len(parts) < 3:
            raise MovoError(
                ErrorCodes.MOVO_ASSET_DECODE_FAILED, f'LUT の行に値が 3 つありません: "{line}"{where}'
            )
        if count + 3 > values.size:
            raise MovoError(
                ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"LUT の行数が LUT_3D_SIZE（{size}）より多いです{where}"
            )
        for c in range(3):
            try:
                value = float(parts[c])
            except ValueError:
                raise MovoError(
                    ErrorCodes.MOVO_ASSET_DECODE_FAILED, f'LUT に数値でない値があります: "{line}"{where}'
                ) from None
            if value != value or value in (float("inf"), float("-inf")):
                raise MovoError(
                    ErrorCodes.MOVO_ASSET_DECODE_FAILED, f'LUT に数値でない値があります: "{line}"{where}'
                )
            values[count + c] = value
        count += 3

    if values is None:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"LUT_3D_SIZE がありません{where}")
    if count != values.size:
        raise MovoError(
            ErrorCodes.MOVO_ASSET_DECODE_FAILED,
            f"LUT の行数が足りません（{count // 3} / {values.size // 3}）{where}",
        )
    for c in range(3):
        # 幅 0 の定義域は割り算で壊れるので、ここで直しておきます。
        if not domain_max[c] > domain_min[c]:
            domain_min[c] = 0.0
            domain_max[c] = 1.0
    return Lut3D(
        size=size,
        title=title,
        domain_min=domain_min,
        domain_max=domain_max,
        data=values.reshape(size, size, size, 3),
    )


def identity_lut(size: int = 2) -> Lut3D:
    """何もしない LUT。テストと «混ぜ量 0 と同じか» の確認に使います。"""
    n = max(2, min(int(round(size)), MAX_LUT_3D_SIZE))
    last = n - 1
    axis = np.arange(n, dtype=np.float32) / last
    data = np.empty((n, n, n, 3), np.float32)
    data[..., 0] = axis[None, None, :]  # r が一番速く回る
    data[..., 1] = axis[None, :, None]
    data[..., 2] = axis[:, None, None]
    return Lut3D(size=n, title="identity", data=data)


def sample_lut(lut: Lut3D, r: float, g: float, b: float) -> list[float]:
    """三線形補間で 1 色引く。入出力とも 0..1。

    格子の «間» はまっすぐ繋ぐだけ（四面体分割はしない）です。粗い LUT だと
    わずかに差が出ますが、絵で見て分かる差ではないので短さを取りました。

    **画面全体に当てるときは :func:`apply_lut` を使ってください。**
    この関数を画素ごとに呼ぶと 1280x720 で 30 秒かかります。
    """
    out = apply_lut(np.array([[[r, g, b]]], np.float32), lut)
    return [float(out[0, 0, 0]), float(out[0, 0, 1]), float(out[0, 0, 2])]


@njit(cache=True)
def _apply_lut_kernel(src, grid, size, dmin, dmax, amount, out):
    """三線形補間で LUT を当てる（画素ごと）。

    **ここは Numba でないと話になりません。** NumPy で書くと «8 隅を引く»
    たびに画面 1 枚ぶんの一時配列ができます。1280x720・33³ の LUT で実測すると

    - NumPy の索引配列（4 次元に 3 本渡す）: 411 ミリ秒
    - NumPy の索引配列（1 次元に潰して 8 回 gather）: 300 ミリ秒
    - **この Numba のカーネル: 32 ミリ秒**

    1 画素ずつ引けば一時配列が要らず、同じ格子点が L1 に載ったままになります。
    """
    n = src.shape[0]
    last = size - 1
    row = size
    slab = size * size
    for i in range(n):
        base0 = 0
        base1 = 0
        base2 = 0
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        for ch in range(3):
            t = (src[i, ch] - dmin[ch]) / (dmax[ch] - dmin[ch])
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            scaled = t * last
            index = int(math.floor(scaled))
            if index > last - 1:
                index = last - 1
            if index < 0:
                index = 0
            frac = scaled - index
            if ch == 0:
                base0 = index
                f0 = frac
            elif ch == 1:
                base1 = index
                f1 = frac
            else:
                base2 = index
                f2 = frac

        origin = (base2 * size + base1) * size + base0
        for ch in range(3):
            c000 = grid[origin, ch]
            c100 = grid[origin + 1, ch]
            c010 = grid[origin + row, ch]
            c110 = grid[origin + row + 1, ch]
            c001 = grid[origin + slab, ch]
            c101 = grid[origin + slab + 1, ch]
            c011 = grid[origin + slab + row, ch]
            c111 = grid[origin + slab + row + 1, ch]
            x00 = c000 + (c100 - c000) * f0
            x10 = c010 + (c110 - c010) * f0
            x01 = c001 + (c101 - c001) * f0
            x11 = c011 + (c111 - c011) * f0
            y0 = x00 + (x10 - x00) * f1
            y1 = x01 + (x11 - x01) * f1
            value = y0 + (y1 - y0) * f2
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
            if amount < 1.0:
                value = src[i, ch] + (value - src[i, ch]) * amount
            out[i, ch] = value


def apply_lut(rgb: np.ndarray, lut: Lut3D, amount: float = 1.0) -> np.ndarray:
    """0..1 の RGB 配列に LUT を当てる（**全画面を一括で**）。

    :param rgb: ``(..., 3)`` の float 配列（0..1）
    :param amount: 0 で無変化、1 で LUT そのまま

    **画素ごとに :func:`sample_lut` を呼ばないでください**（あちらは
    この関数を 1 画素だけ呼ぶ薄い皮なので、呼び出しの手間だけが積み上がります）。
    1280x720・33³ の LUT で **32 ミリ秒**、同じコードを Numba なしで走らせると
    512 ミリ秒です（320x180 での比較で 320 倍）。
    """
    source = np.ascontiguousarray(rgb, np.float32)
    if amount <= 0.0:
        return source
    shape = source.shape
    flat = source.reshape(-1, 3)
    out = np.empty_like(flat)
    _apply_lut_kernel(
        flat,
        lut.data.reshape(-1, 3),
        lut.size,
        np.asarray(lut.domain_min, np.float32),
        np.asarray(lut.domain_max, np.float32),
        np.float32(min(1.0, amount)),
        out,
    )
    return out.reshape(shape)
