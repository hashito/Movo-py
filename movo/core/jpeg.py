"""ベースライン（逐次 DCT・ハフマン）の JPEG デコーダ。**自前実装です。**

ffmpeg の入っていない機械でも、普通の写真素材が使えるようにするためです。
プログレッシブ JPEG はここでは扱いません（素材ローダが ffmpeg に回し、
ffmpeg も無ければはっきりしたエラーを出します）。

## ここが Numba の効きどころです

JPEG の復号は **ビット単位で前から順に読む**処理です。ハフマン符号は
可変長なので «次のビットがどこから始まるか» が直前の結果で決まります。
**並べて計算できないので NumPy では書けません。**

逆 DCT も同様です。8x8 ブロックごとに 1,024 回の積和があり、フル HD だと
4 万ブロックあります。ここも同じ関数の中でコンパイルしています。

実測です（同じコードを ``NUMBA_DISABLE_JIT=1`` で走らせて比べたもの）。

| 320x180 の写真 1 枚 | |
| --- | --- |
| 素の Python | 1,250 ms |
| Numba（`@njit`） | **3.4 ms**（368 倍） |

1280x720 で 84 ミリ秒です。

## JS 版と «同じ絵» が出るようにしていること

逆 DCT の途中経過を **わざと float32 に落としています**（`_rows` / `_tmp`）。
JS 版が `Float32Array` を使っているためで、float64 のまま通すと最下位ビットが
ずれた画素が出ます。積和そのものは JS の数値型に合わせて float64 で行います。
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .bitmap import Bitmap
from .errors import ErrorCodes, MovoError

ZIGZAG = np.array(
    [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ],
    np.int32,
)

# 逆 DCT の余弦表と正規化係数。**float32 なのは JS 版に合わせるため**です。
_COS = np.array(
    [math.cos((2 * x + 1) * u * math.pi / 16) for u in range(8) for x in range(8)], np.float32
)
_CU = np.array([1 / math.sqrt(2) if u == 0 else 1.0 for u in range(8)], np.float32)


def is_jpeg(buffer: bytes) -> bool:
    return len(buffer) > 3 and buffer[0] == 0xFF and buffer[1] == 0xD8


# ── ハフマン表 ──────────────────────────────────────────────


def _build_huffman(counts: list[int], values: list[int]):
    """正準ハフマン表を «長さごとの範囲» の形に組む。

    JS 版は ``"長さ:符号"`` をキーにした Map を引いていますが、Numba の中では
    辞書が使えません。**同じ符号割り当てを配列 3 本で表す** 標準の形
    （JPEG 仕様 F.2.2.3 の MINCODE / MAXCODE / VALPTR）に直しています。
    符号の割り当て順は同じなので、結果は 1 ビットも変わりません。
    """
    mincode = np.zeros(17, np.int32)
    maxcode = np.full(17, -1, np.int32)
    valptr = np.zeros(17, np.int32)
    code = 0
    k = 0
    for length in range(1, 17):
        n = counts[length - 1]
        mincode[length] = code
        valptr[length] = k
        if n > 0:
            maxcode[length] = code + n - 1
            k += n
            code += n
        code <<= 1
    table_values = np.zeros(256, np.int32)
    table_values[: len(values)] = np.array(values, np.int32)
    return mincode, maxcode, valptr, table_values


# ── ビット読み ──────────────────────────────────────────────
#
# 状態は int64 の配列 1 本で持ちます（[位置, ビットバッファ, 残ビット数, マーカー]）。
# Numba の中で «参照渡しの可変な状態» を持つ一番素直なやり方です。


@njit(cache=True, inline="always")
def _read_bit(data, state):
    """1 ビット読む。読めなければ -1（ストリームの終わり、またはマーカー）。"""
    if state[2] == 0:
        if state[0] >= data.shape[0]:
            return -1
        byte = np.int64(data[state[0]])
        state[0] += 1
        if byte == 0xFF:
            nxt = np.int64(data[state[0]]) if state[0] < data.shape[0] else np.int64(-1)
            if nxt == 0x00:
                # 詰め物（0xFF00）。0xFF そのものを表す
                state[0] += 1
            elif 0xD0 <= nxt <= 0xD7:
                # リスタートマーカーはまたいで次のバイトへ
                state[0] += 1
                if state[0] >= data.shape[0]:
                    return -1
                byte = np.int64(data[state[0]])
                state[0] += 1
                if byte == 0xFF and state[0] < data.shape[0] and data[state[0]] == 0x00:
                    state[0] += 1
            else:
                state[3] = nxt
                return -1
        state[1] = byte
        state[2] = 8
    state[2] -= 1
    return np.int64((state[1] >> state[2]) & 1)


@njit(cache=True, inline="always")
def _decode_huffman(data, state, mincode, maxcode, valptr, values, t):
    """1 符号ぶん読む。-1 は打ち切り、-2 は «そんな符号は無い»。"""
    code = np.int64(0)
    for length in range(1, 17):
        bit = _read_bit(data, state)
        if bit < 0:
            return np.int64(-1)
        code = (code << 1) | bit
        if maxcode[t, length] >= 0 and mincode[t, length] <= code <= maxcode[t, length]:
            return np.int64(values[t, valptr[t, length] + code - mincode[t, length]])
    return np.int64(-2)


@njit(cache=True, inline="always")
def _receive(data, state, length):
    value = np.int64(0)
    for i in range(length):
        bit = _read_bit(data, state)
        if bit < 0:
            return value << (length - i - 1)
        value = (value << 1) | bit
    return value


@njit(cache=True, inline="always")
def _receive_and_extend(data, state, length):
    if length == 0:
        return np.int64(0)
    value = _receive(data, state, length)
    if value < (np.int64(1) << (length - 1)):
        return value - (np.int64(1) << length) + 1
    return value


@njit(cache=True)
def _idct(block, out, rows, tmp):
    """8x8 の逆 DCT（行 → 列の 2 段）。

    途中の ``rows`` / ``tmp`` は **float32** です（JS 版の Float32Array に合わせています）。
    積和そのものは float64 で行います。JS の数値はすべて double だからです。
    """
    for v in range(8):
        for x in range(8):
            total = 0.0
            for u in range(8):
                coeff = block[v * 8 + u]
                if coeff != 0:
                    total += np.float64(_CU[u]) * np.float64(coeff) * np.float64(_COS[u * 8 + x])
            rows[v * 8 + x] = np.float32(total / 2.0)
    for x in range(8):
        for y in range(8):
            total = 0.0
            for v in range(8):
                total += np.float64(_CU[v]) * np.float64(rows[v * 8 + x]) * np.float64(_COS[v * 8 + y])
            tmp[y * 8 + x] = np.float32(total / 2.0)
    for i in range(64):
        v = np.int64(math.floor(np.float64(tmp[i]) + 0.5)) + 128
        out[i] = np.uint8(0 if v < 0 else (255 if v > 255 else v))


@njit(cache=True)
def _decode_scan(
    data,
    start,
    mcus_x,
    mcus_y,
    comp_h,
    comp_v,
    comp_tq,
    comp_line_width,
    comp_offset,
    pixels,
    sc_comp,
    sc_dc,
    sc_ac,
    dc_min,
    dc_max,
    dc_ptr,
    dc_val,
    ac_min,
    ac_max,
    ac_ptr,
    ac_val,
    quant,
    restart_interval,
):
    """走査（SOS）を丸ごと復号して ``pixels`` を埋める。

    返り値は ``(次に読むべき位置, エラーコード)``。エラーコードは
    0=正常 / 1=ハフマン符号が壊れている。
    """
    state = np.zeros(4, np.int64)
    state[0] = start
    n_scan = sc_comp.shape[0]
    n_comp = comp_h.shape[0]
    pred = np.zeros(n_comp, np.int64)

    block = np.zeros(64, np.int64)
    dequant = np.zeros(64, np.int64)
    out = np.zeros(64, np.uint8)
    rows = np.zeros(64, np.float32)
    tmp = np.zeros(64, np.float32)

    mcu_count = 0
    truncated = False
    for my in range(mcus_y):
        if truncated:
            break
        for mx in range(mcus_x):
            if restart_interval > 0 and mcu_count > 0 and mcu_count % restart_interval == 0:
                state[2] = 0  # ビットバッファを捨てる（リスタートはバイト境界から）
                while state[0] < data.shape[0] - 1:
                    if data[state[0]] == 0xFF and 0xD0 <= data[state[0] + 1] <= 0xD7:
                        state[0] += 2
                        break
                    if data[state[0]] == 0xFF and data[state[0] + 1] != 0x00:
                        break
                    state[0] += 1
                for i in range(n_comp):
                    pred[i] = 0
            mcu_count += 1

            for s in range(n_scan):
                ci = sc_comp[s]
                h = comp_h[ci]
                v = comp_v[ci]
                line_width = comp_line_width[ci]
                base_offset = comp_offset[ci]
                for by in range(v):
                    for bx in range(h):
                        for i in range(64):
                            block[i] = 0
                        t = _decode_huffman(data, state, dc_min, dc_max, dc_ptr, dc_val, sc_dc[s])
                        if t == -1:
                            truncated = True
                            break
                        if t == -2:
                            return state[0], 1
                        diff = 0 if t == 0 else _receive_and_extend(data, state, t)
                        pred[ci] += diff
                        block[0] = pred[ci]
                        k = 1
                        while k < 64:
                            rs = _decode_huffman(data, state, ac_min, ac_max, ac_ptr, ac_val, sc_ac[s])
                            if rs < 0:
                                break
                            ssss = rs & 15
                            r = rs >> 4
                            if ssss == 0:
                                if r < 15:
                                    break
                                k += 16
                                continue
                            k += r
                            if k > 63:
                                break
                            block[ZIGZAG[k]] = _receive_and_extend(data, state, ssss)
                            k += 1
                        tq = comp_tq[ci]
                        for i in range(64):
                            dequant[i] = block[i] * quant[tq, i]
                        _idct(dequant, out, rows, tmp)
                        base_x = (mx * h + bx) * 8
                        base_y = (my * v + by) * 8
                        for y in range(8):
                            row = base_offset + (base_y + y) * line_width + base_x
                            for x in range(8):
                                pixels[row + x] = out[y * 8 + x]
                    if truncated:
                        break
                if truncated:
                    break
            if truncated:
                break

    # 次のマーカーを探す。外側の解析がここから続けられるように。
    p = state[0]
    while p < data.shape[0] - 1:
        if data[p] == 0xFF and data[p + 1] != 0x00 and not (0xD0 <= data[p + 1] <= 0xD7):
            break
        p += 1
    return p, 0


# ── 全体の流れ ──────────────────────────────────────────────


def decode_jpeg(buffer: bytes) -> Bitmap:
    """ベースライン JPEG を :class:`Bitmap` にする。"""
    if not is_jpeg(buffer):
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "JPEG ではありません")

    raw = np.frombuffer(bytes(buffer), np.uint8)
    view = memoryview(bytes(buffer))
    quant_tables: dict[int, np.ndarray] = {}
    huff_dc: dict[int, tuple] = {}
    huff_ac: dict[int, tuple] = {}
    frame = None
    restart_interval = 0
    adobe_transform = -1
    offset = 2

    while offset < len(view):
        if view[offset] != 0xFF:
            offset += 1
            continue
        marker = view[offset + 1] if offset + 1 < len(view) else 0xD9
        offset += 2
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            break
        if offset + 2 > len(view):
            break
        length = int.from_bytes(view[offset : offset + 2], "big")
        segment = bytes(view[offset + 2 : offset + length])

        if marker == 0xDB:  # 量子化表
            p = 0
            while p < len(segment):
                pq = segment[p] >> 4
                tq = segment[p] & 15
                p += 1
                table = np.zeros(64, np.int32)
                for i in range(64):
                    table[ZIGZAG[i]] = (
                        int.from_bytes(segment[p + i * 2 : p + i * 2 + 2], "big") if pq else segment[p + i]
                    )
                p += 128 if pq else 64
                quant_tables[tq] = table
        elif marker in (0xC0, 0xC1):  # ベースラインのフレームヘッダ
            frame = {
                "height": int.from_bytes(segment[1:3], "big"),
                "width": int.from_bytes(segment[3:5], "big"),
                "components": [],
            }
            count = segment[5]
            for i in range(count):
                p = 6 + i * 3
                frame["components"].append(
                    {"id": segment[p], "h": segment[p + 1] >> 4, "v": segment[p + 1] & 15, "tq": segment[p + 2]}
                )
        elif marker == 0xC2:
            raise MovoError(
                ErrorCodes.MOVO_ASSET_DECODE_FAILED,
                "プログレッシブ JPEG には自前のデコーダは対応していません",
                hint="ffmpeg を入れて変換させるか、ベースライン JPEG か PNG で保存し直してください",
            )
        elif marker == 0xC4:  # ハフマン表
            p = 0
            while p < len(segment):
                tc = segment[p] >> 4
                th = segment[p] & 15
                p += 1
                counts = list(segment[p : p + 16])
                p += 16
                total = sum(counts)
                values = list(segment[p : p + total])
                p += total
                table = _build_huffman(counts, values)
                if tc == 0:
                    huff_dc[th] = table
                else:
                    huff_ac[th] = table
        elif marker == 0xDD:
            restart_interval = int.from_bytes(segment[0:2], "big")
        elif marker == 0xEE:
            if segment[:5] == b"Adobe":
                adobe_transform = segment[-1]
        elif marker == 0xDA:  # 走査の開始
            if frame is None:
                raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "フレームヘッダの前に走査が来ました")
            count = segment[0]
            scan = []
            for i in range(count):
                cid = segment[1 + i * 2]
                tables = segment[2 + i * 2]
                index = next((j for j, c in enumerate(frame["components"]) if c["id"] == cid), 0)
                scan.append({"index": index, "dc": tables >> 4, "ac": tables & 15})
            offset = _run_scan(
                raw, offset + length, frame, scan, quant_tables, huff_dc, huff_ac, restart_interval
            )
            continue
        offset += length

    if frame is None:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "JPEG にフレームヘッダがありません")
    return _assemble(frame, adobe_transform)


def _run_scan(raw, start, frame, scan, quant_tables, huff_dc, huff_ac, restart_interval) -> int:
    """走査 1 本を復号する。Numba の関数へ渡せる «配列だけ» の形に直します。"""
    components = frame["components"]
    max_h = max(c["h"] for c in components)
    max_v = max(c["v"] for c in components)
    mcus_x = -(-frame["width"] // (8 * max_h))
    mcus_y = -(-frame["height"] // (8 * max_v))

    n = len(components)
    comp_h = np.array([c["h"] for c in components], np.int32)
    comp_v = np.array([c["v"] for c in components], np.int32)
    comp_tq = np.array([c["tq"] for c in components], np.int32)
    comp_line_width = np.zeros(n, np.int64)
    comp_offset = np.zeros(n, np.int64)

    cursor = 0
    for i, c in enumerate(components):
        c["blocksPerLine"] = mcus_x * c["h"]
        c["blocksPerColumn"] = mcus_y * c["v"]
        c["lineWidth"] = c["blocksPerLine"] * 8
        c["pixelHeight"] = c["blocksPerColumn"] * 8
        c["maxH"] = max_h
        c["maxV"] = max_v
        comp_line_width[i] = c["lineWidth"]
        comp_offset[i] = cursor
        cursor += c["lineWidth"] * c["pixelHeight"]

    pixels = np.zeros(max(1, cursor), np.uint8)

    sc_comp = np.array([s["index"] for s in scan], np.int32)
    sc_dc = np.array([s["dc"] for s in scan], np.int32)
    sc_ac = np.array([s["ac"] for s in scan], np.int32)

    dc_min, dc_max, dc_ptr, dc_val = _pack_tables(huff_dc)
    ac_min, ac_max, ac_ptr, ac_val = _pack_tables(huff_ac)
    for s in scan:
        if s["dc"] not in huff_dc or s["ac"] not in huff_ac:
            raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "JPEG が存在しないハフマン表を指しています")

    quant = np.ones((4, 64), np.int32)
    for tq, table in quant_tables.items():
        if 0 <= tq < 4:
            quant[tq] = table

    next_offset, status = _decode_scan(
        raw, start, mcus_x, mcus_y, comp_h, comp_v, comp_tq, comp_line_width, comp_offset,
        pixels, sc_comp, sc_dc, sc_ac,
        dc_min, dc_max, dc_ptr, dc_val, ac_min, ac_max, ac_ptr, ac_val,
        quant, restart_interval,
    )
    if status == 1:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "JPEG のハフマン符号が壊れています")

    for i, c in enumerate(components):
        o = int(comp_offset[i])
        size = c["lineWidth"] * c["pixelHeight"]
        c["pixels"] = pixels[o : o + size].reshape(c["pixelHeight"], c["lineWidth"])
    return int(next_offset)


def _pack_tables(tables: dict[int, tuple]):
    """0..3 番のハフマン表を 1 つの配列にまとめる（Numba へ渡すため）。"""
    mincode = np.zeros((4, 17), np.int32)
    maxcode = np.full((4, 17), -1, np.int32)
    valptr = np.zeros((4, 17), np.int32)
    values = np.zeros((4, 256), np.int32)
    for th, (mn, mx, vp, vals) in tables.items():
        if 0 <= th < 4:
            mincode[th] = mn
            maxcode[th] = mx
            valptr[th] = vp
            values[th] = vals
    return mincode, maxcode, valptr, values


def _upsample(component: dict, width: int, height: int, max_h: int, max_v: int) -> np.ndarray:
    """成分の «間引かれた» 画素を画面いっぱいに広げる。

    最近傍です（JS 版と同じ）。**NumPy の索引配列 1 回**で済むので、
    画素ごとのループは要りません。
    """
    xs = np.minimum(component["lineWidth"] - 1, (np.arange(width) * component["h"]) // max_h)
    ys = np.minimum(component["pixelHeight"] - 1, (np.arange(height) * component["v"]) // max_v)
    return component["pixels"][np.ix_(ys, xs)].astype(np.float64)


def _clamped(values: np.ndarray) -> np.ndarray:
    """JS の ``Uint8ClampedArray`` への代入と同じ丸め（**偶数丸め**）＋切り詰め。"""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _assemble(frame: dict, adobe_transform: int) -> Bitmap:
    width = frame["width"]
    height = frame["height"]
    components = frame["components"]
    bitmap = Bitmap(width, height)
    if not components or "pixels" not in components[0]:
        return bitmap  # 走査が無かった（壊れたファイル）。透明のまま返す

    max_h = max(c["h"] for c in components)
    max_v = max(c["v"] for c in components)
    planes = [_upsample(c, width, height, max_h, max_v) for c in components]
    out = bitmap.data

    if len(components) == 1:
        g = _clamped(planes[0])
        out[..., 0] = out[..., 1] = out[..., 2] = g
    elif len(components) == 3:
        y = planes[0]
        cb = planes[1] - 128
        cr = planes[2] - 128
        out[..., 0] = _clamped(y + 1.402 * cr)
        out[..., 1] = _clamped(y - 0.344136 * cb - 0.714136 * cr)
        out[..., 2] = _clamped(y + 1.772 * cb)
    elif len(components) == 4:
        c0, c1, c2 = planes[0], planes[1], planes[2]
        k = planes[3]
        if adobe_transform != 0:
            y = c0
            cb = c1 - 128
            cr = c2 - 128
            c0 = y + 1.402 * cr
            c1 = y - 0.344136 * cb - 0.714136 * cr
            c2 = y + 1.772 * cb
        out[..., 0] = _clamped(c0 * k / 255)
        out[..., 1] = _clamped(c1 * k / 255)
        out[..., 2] = _clamped(c2 * k / 255)
    out[..., 3] = 255
    return bitmap
