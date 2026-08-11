"""アニメーション GIF の書き出し（GIF89a・LZW）。自前実装です。

ffmpeg が無い環境でも «動くもの» を出せるようにするための逃げ道です
（仕様の原則 8）。PNG 連番しか出せないのと、GIF が出せるのとでは
«確認できるか» が違います。

## 速度の作法

GIF は 1 画素ずつ «パレットの何番か» を決めて、その並びを LZW で縮めます。
**どちらも画素ごとのループ**で、NumPy では畳めません（LZW は前の画素の
結果に依存し、色の最近傍探索は分岐が深い）。

| 960x540 の 1 フレーム（減色＋LZW） | |
| --- | --- |
| 純 Python | 3,952 ms |
| **Numba** | **10.8 ms**（367 倍） |

900 フレームの GIF で **59 分 対 10 秒**です。ここは Numba 一択でした。

LZW の辞書は Python の `dict` ではなく **開番地法のハッシュ表**（配列 2 本）に
してあります。Numba の `typed.Dict` は要素ごとにロックを取るので、この用途
では素の配列より 6 倍遅くなりました。**出てくるバイト列は同じ**です。
"""

from __future__ import annotations

import numpy as np
from numba import njit

# 辞書の容量。LZW の符号は最大 4096 なので、その 4 倍を取って «詰まらない»
# ようにしています（開番地法は 7 割を超えると急に遅くなります）。
_TABLE_SIZE = 16384


def build_palette(frames, max_colors: int = 256) -> np.ndarray:
    """メディアンカットで色を選ぶ。

    **標本の取り方も分割の順番も JS 版のままです。** ここを変えると同じ
    プロジェクトから «色の違う GIF» が出ます。

    :param frames: `Bitmap` の並び
    :param max_colors: 2〜256
    :returns: ``(色数, 3)`` の uint8
    """
    stride = max(1, int((frames[0].width * frames[0].height) // 6000))
    chunks = []
    for frame in frames:
        flat = frame.data.reshape(-1, 4)
        picked = flat[::stride]
        chunks.append(picked[picked[:, 3] >= 128][:, :3])
    samples = np.concatenate(chunks) if chunks else np.zeros((0, 3), np.uint8)
    if len(samples) == 0:
        samples = np.zeros((1, 3), np.uint8)
    samples = samples.astype(np.int32)

    buckets = [samples]
    limit = max(2, min(256, max_colors))
    while len(buckets) < limit:
        # いちばん «色の幅» が広い入れ物を割る。
        target_index = -1
        best_range = -1
        best_channel = 0
        for index, bucket in enumerate(buckets):
            if len(bucket) < 2:
                continue
            for channel in range(3):
                span = int(bucket[:, channel].max()) - int(bucket[:, channel].min())
                if span > best_range:
                    best_range = span
                    target_index = index
                    best_channel = channel
        if target_index < 0 or best_range <= 0:
            break
        bucket = buckets[target_index]
        # **安定ソートです。** 同じ値の並びが変わると割れる位置が変わり、
        # パレットの色が変わってしまいます（JS の `Array.sort` も安定です）。
        order = np.argsort(bucket[:, best_channel], kind="stable")
        bucket = bucket[order]
        middle = len(bucket) // 2
        buckets[target_index: target_index + 1] = [bucket[:middle], bucket[middle:]]

    # **`np.round` は «偶数へ» 丸めます。** JS の `Math.round` は 0.5 を上へ
    # 丸めるので、平均がちょうど .5 になった入れ物で色が 1 ずれます
    # （実際に緑が 82 と 83 に割れました）。
    palette = [np.floor(b.mean(axis=0) + 0.5).astype(np.int64) for b in buckets if len(b)]
    while len(palette) < 2:
        palette.append(np.zeros(3, np.int64))
    return np.clip(np.array(palette), 0, 255).astype(np.uint8)


@njit(cache=True)
def _quantize(data, palette, cache, transparent, transparent_index):
    """画素をパレットの番号に置き換える。

    5 ビット × 3 の表に覚えさせて、同じ色を 2 度探さないようにしています
    （JS 版と同じ 32,768 要素の表）。
    """
    n = data.shape[0]
    indices = np.empty(n, np.uint8)
    for p in range(n):
        if transparent and data[p, 3] < 128:
            indices[p] = transparent_index
            continue
        r = data[p, 0]
        g = data[p, 1]
        b = data[p, 2]
        key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
        cached = cache[key]
        if cached >= 0:
            indices[p] = cached
            continue
        best = 0
        best_distance = 1 << 30
        for i in range(palette.shape[0]):
            dr = int(palette[i, 0]) - int(r)
            dg = int(palette[i, 1]) - int(g)
            db = int(palette[i, 2]) - int(b)
            distance = dr * dr + dg * dg + db * db
            if distance < best_distance:
                best_distance = distance
                best = i
        cache[key] = best
        indices[p] = best
    return indices


@njit(cache=True)
def _lzw_encode(indices, min_code_size, out):
    """GIF の仕様どおりの LZW。戻り値は書いたバイト数。

    JS 版の «文字列を鍵にした Map» と同じ辞書を、開番地法のハッシュ表で
    作っています。鍵は ``(前の符号 << 8) | 次の画素``。前の符号が同じで
    次の画素も同じなら同じ並び、なので文字列と 1 対 1 に対応します。
    """
    clear_code = 1 << min_code_size
    end_code = clear_code + 1

    keys = np.full(_TABLE_SIZE, -1, np.int64)
    values = np.zeros(_TABLE_SIZE, np.int32)

    code_size = min_code_size + 1
    next_code = end_code + 1
    written = 0
    current = 0
    bit_count = 0

    # 最初のクリア符号。ビットは «下位から» 詰めます（GIF の決まり）。
    for b in range(code_size):
        current |= ((clear_code >> b) & 1) << bit_count
        bit_count += 1
        if bit_count == 8:
            out[written] = current
            written += 1
            current = 0
            bit_count = 0

    prefix = np.int64(indices[0])
    for i in range(1, indices.shape[0]):
        k = np.int64(indices[i])
        combined = (prefix << 8) | k
        # 探す
        slot = np.int64((combined * 2654435761) & (_TABLE_SIZE - 1))
        found = -1
        while keys[slot] != -1:
            if keys[slot] == combined:
                found = values[slot]
                break
            slot = (slot + 1) & (_TABLE_SIZE - 1)
        if found >= 0:
            prefix = np.int64(found)
            continue

        # 前の並びを出力
        code = prefix
        for b in range(code_size):
            current |= ((code >> b) & 1) << bit_count
            bit_count += 1
            if bit_count == 8:
                out[written] = current
                written += 1
                current = 0
                bit_count = 0

        keys[slot] = combined
        values[slot] = next_code
        next_code += 1
        # **JS 版と同じ判定です。** 標準的な LZW は `>=` ですが、JS 版は `>` で
        # 書かれていて、符号長が 1 画素ぶん遅れて伸びます。ここを «直す» と
        # 既存の GIF とバイト列が変わるので、そのままにしてあります。
        if next_code > (1 << code_size) and code_size < 12:
            code_size += 1
        elif next_code > 4095:
            code = clear_code
            for b in range(code_size):
                current |= ((code >> b) & 1) << bit_count
                bit_count += 1
                if bit_count == 8:
                    out[written] = current
                    written += 1
                    current = 0
                    bit_count = 0
            keys[:] = -1
            next_code = end_code + 1
            code_size = min_code_size + 1
        prefix = k

    for b in range(code_size):
        current |= ((prefix >> b) & 1) << bit_count
        bit_count += 1
        if bit_count == 8:
            out[written] = current
            written += 1
            current = 0
            bit_count = 0
    for b in range(code_size):
        current |= ((end_code >> b) & 1) << bit_count
        bit_count += 1
        if bit_count == 8:
            out[written] = current
            written += 1
            current = 0
            bit_count = 0
    if bit_count > 0:
        out[written] = current
        written += 1
    return written


def _push_blocks(target: bytearray, data: np.ndarray, length: int) -> None:
    """LZW の出力を GIF の «サブブロック»（先頭 1 バイトが長さ）に割り直す。"""
    i = 0
    while i < length:
        chunk = min(255, length - i)
        target.append(chunk)
        target += data[i: i + chunk].tobytes()
        i += chunk
    target.append(0)


def encode_gif(frames, options: dict | None = None) -> bytes:
    """フレームの並びをアニメーション GIF にする。

    :param frames: `Bitmap` の並び（すべて同じ大きさ）
    :param options: ``fps`` / ``colors`` / ``loop`` / ``transparent``
    """
    options = options or {}
    if not frames:
        raise ValueError("encode_gif には最低 1 フレーム要ります")
    # `or` で既定値に落とすのは、プロジェクト JSON の `null` を «指定なし» と
    # 読むためです（JS 版の `??` と同じ扱い）。`get(key, 既定値)` だとキーがあれば
    # `None` が返り、その先の割り算で落ちます。
    fps = options.get("fps") or 15
    # **組み込みの `round` は使えません。** Python は «偶数へ» 丸めるので
    # 100/8 = 12.5 が 12 になりますが、JS の `Math.round` は 13 です。
    # 1/100 秒の差でも、GIF の再生速度が JS 版とわずかにずれます。
    delay = max(2, int(np.floor(100 / fps + 0.5)))
    transparent = options.get("transparent") is not False
    max_colors = min(255 if transparent else 256, options.get("colors") or 256)

    palette = build_palette(frames, max_colors)
    transparent_index = len(palette) if transparent else -1
    if transparent:
        palette = np.vstack([palette, np.zeros((1, 3), np.uint8)])

    palette_bits = 1
    while (1 << palette_bits) < len(palette):
        palette_bits += 1
    palette_bits = max(1, min(8, palette_bits))
    palette_size = 1 << palette_bits

    width = frames[0].width
    height = frames[0].height
    out = bytearray()

    def push_u16(value: int) -> None:
        out.append(value & 0xFF)
        out.append((value >> 8) & 0xFF)

    out += b"GIF89a"
    push_u16(width)
    push_u16(height)
    out.append(0x80 | (palette_bits - 1))
    out += bytes((0, 0))
    for i in range(palette_size):
        color = palette[i] if i < len(palette) else (0, 0, 0)
        out += bytes((int(color[0]), int(color[1]), int(color[2])))

    # Netscape のループ拡張
    out += bytes((0x21, 0xFF, 0x0B))
    out += b"NETSCAPE2.0"
    out += bytes((0x03, 0x01))
    push_u16(int(options.get("loop") or 0))
    out.append(0x00)

    cache = np.full(32768, -1, np.int16)
    min_code_size = max(2, palette_bits)
    # LZW の出力は «最悪でも画素数 × 2 バイト» に収まります（符号は 12 ビット以下、
    # 1 画素で 1 符号が上限）。毎フレーム確保し直さないよう 1 度だけ取ります。
    scratch = np.empty(width * height * 2 + 64, np.uint8)

    for frame in frames:
        flat = np.ascontiguousarray(frame.data.reshape(-1, 4))
        indices = _quantize(flat, palette, cache, transparent, transparent_index if transparent else 0)

        # グラフィック制御拡張（表示時間と透明色）
        out += bytes((0x21, 0xF9, 0x04))
        out.append(0x09 if transparent else 0x08)
        push_u16(delay)
        out.append(transparent_index if transparent else 0)
        out.append(0x00)

        # 画像記述子
        out.append(0x2C)
        push_u16(0)
        push_u16(0)
        push_u16(width)
        push_u16(height)
        out.append(0x00)

        out.append(min_code_size)
        written = _lzw_encode(indices, min_code_size, scratch)
        _push_blocks(out, scratch, written)

    out.append(0x3B)
    return bytes(out)
