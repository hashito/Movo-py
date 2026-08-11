"""TrueType / OpenType(glyf) の読み込みと字形（グリフ）輪郭の取り出し。

Movo は文字を **自前でラスタライズ** します。そうすると OS が変わっても
同じ絵が出て、ネイティブ依存（freetype など）も要らないためです。その代わり
PostScript 輪郭（CFF）のフォントはここでは描けません。フォントマネージャは
そういうファイルを黙って読み飛ばし、TrueType の面へ落とします。

JS 版（``packages/renderer/src/font.js``）からの移植です。**選び方・点数付け・
オフセットの移動量は 1 バイトも変えていません**。JS 版と Python 版で同じ
プロジェクトを描いたときに、同じフォントが選ばれないと意味がないためです。

外部のフォントライブラリ（fontTools / freetype）は使いません。標準ライブラリの
zlib と numpy だけで完結しています。
"""

from __future__ import annotations

import logging
import os
import re
import struct
import sys
import weakref
import zlib
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# 依存の «あとで書かれるかもしれない» モジュールたち。
#
# movo.core.errors / movo.core.logger / movo.core.platform は別々に作られる
# ので、無くてもこのファイル単体で動くようにフォールバックを置いています。
# --------------------------------------------------------------------------

try:  # pragma: no cover - 本番では必ず通る側
    from movo.core.errors import ErrorCodes, MovoError
except Exception:  # pragma: no cover - errors.py がまだ無いときの最小実装

    class ErrorCodes:  # type: ignore[no-redef]
        """movo.core.errors が無いときの最小の代用。"""

        MOVO_ASSET_DECODE_FAILED = "MOVO_ASSET_DECODE_FAILED"
        MOVO_FONT_NOT_FOUND = "MOVO_FONT_NOT_FOUND"
        MOVO_UNSUPPORTED = "MOVO_UNSUPPORTED"

    class MovoError(Exception):  # type: ignore[no-redef]
        """movo.core.errors が無いときの最小の代用。"""

        def __init__(
            self,
            code: str,
            reason: str,
            *,
            file: str | None = None,
            path: str | None = None,
            hint: str | None = None,
            cause: BaseException | None = None,
        ) -> None:
            super().__init__(f"{code}: {reason}")
            self.code = code
            self.reason = reason
            self.file = file
            self.path = path
            self.hint = hint
            if cause is not None:
                self.__cause__ = cause


try:  # pragma: no cover
    from movo.core.logger import logger  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    logger = logging.getLogger("movo")


def _log_debug(message: str) -> None:
    """logger の実体が何であっても debug を出す。"""
    try:
        logger.debug(message)
    except Exception:
        pass


def _log_warn(message: str) -> None:
    """JS 版の ``logger.warn`` 相当。

    標準 logging の ``Logger.warn`` は Python 3.13 で消えたので、
    ``warning`` を先に試します。
    """
    fn = getattr(logger, "warning", None) or getattr(logger, "warn", None)
    if fn is None:
        return
    try:
        fn(message)
    except Exception:
        pass


#: 読み込めるフォントの拡張子。woff / woff2 は «その場で SFNT に戻して» から読みます。
FONT_EXTENSIONS = re.compile(r"\.(ttf|otf|ttc|woff2?)$", re.I)


# ==========================================================================
# バイト列を読む道具
#
# **フォントのバイト順は必ずビッグエンディアン** です。i16 は符号付き、
# u16 は符号なし。取り違えると座標が突然 65000 台に飛びます。
# ==========================================================================


class Reader:
    """ビッグエンディアンの読み取りカーソル。JS 版の ``Reader`` と同じ。"""

    __slots__ = ("buffer", "offset")

    def __init__(self, buffer: bytes, offset: int = 0) -> None:
        self.buffer = buffer
        self.offset = offset

    def u8(self) -> int:
        value = self.buffer[self.offset]
        self.offset += 1
        return value

    def i8(self) -> int:
        value = self.buffer[self.offset]
        self.offset += 1
        return value - 256 if value >= 128 else value

    def u16(self) -> int:
        value = struct.unpack_from(">H", self.buffer, self.offset)[0]
        self.offset += 2
        return value

    def i16(self) -> int:
        value = struct.unpack_from(">h", self.buffer, self.offset)[0]
        self.offset += 2
        return value

    def u32(self) -> int:
        value = struct.unpack_from(">I", self.buffer, self.offset)[0]
        self.offset += 4
        return value

    def seek(self, offset: int) -> "Reader":
        self.offset = offset
        return self


def _u16(buffer: bytes, offset: int) -> int:
    return struct.unpack_from(">H", buffer, offset)[0]


def _i16(buffer: bytes, offset: int) -> int:
    return struct.unpack_from(">h", buffer, offset)[0]


def _u32(buffer: bytes, offset: int) -> int:
    return struct.unpack_from(">I", buffer, offset)[0]


def _ascii(buffer: bytes, start: int, end: int) -> str:
    return bytes(buffer[start:end]).decode("ascii", errors="replace")


# ==========================================================================
# WOFF / WOFF2 → SFNT
# ==========================================================================


def to_sfnt(buf: bytes) -> bytes:
    """Web 配布のフォントを «普通の TrueType» に戻す。

    Web で拾えるフォントは woff / woff2 であることが多く、そのままだと
    「ダウンロードしたのに使えない」で止まってしまいます。woff は標準ライブラリの
    zlib で戻せます。woff2 は Brotli が要りますが Python の標準ライブラリには
    無いので、``brotli`` / ``brotlicffi`` があれば使い、無ければ分かりやすい
    例外にします。

    戻すのは «テーブルの中身» だけで、チェックサムは作り直しません。Movo の
    パーサは字形しか見ないので実害がなく、正しく作ろうとすると head の
    checkSumAdjustment まで巻き込むためです。

    :param buf: フォントのバイト列
    :returns: SFNT（ttf/otf/ttc）そのままなら入力をそのまま返す
    """
    if len(buf) < 4:
        return buf
    tag = _u32(buf, 0)
    if tag == 0x774F4646:  # 'wOFF'
        return _sfnt_from_woff(buf)
    if tag == 0x774F4632:  # 'wOF2'
        return _sfnt_from_woff2(buf)
    return buf


def _build_sfnt(flavor: int, tables: Sequence[tuple[str, bytes]]) -> bytes:
    """テーブルを 4 バイト境界に並べた SFNT を組み立てる。woff / woff2 で共通。"""
    count = len(tables)
    entry_selector = 0
    while (1 << (entry_selector + 1)) <= count:
        entry_selector += 1
    search_range = (1 << entry_selector) * 16
    header = bytearray(12 + count * 16)
    struct.pack_into(">I", header, 0, flavor)
    struct.pack_into(">H", header, 4, count)
    struct.pack_into(">H", header, 6, search_range & 0xFFFF)
    struct.pack_into(">H", header, 8, entry_selector)
    struct.pack_into(">H", header, 10, (count * 16 - search_range) & 0xFFFF)

    # テーブルはタグ順に並べる決まりです（一部のパーサが二分探索を前提にするため）。
    sorted_tables = sorted(tables, key=lambda t: t[0])
    parts: list[bytes] = []
    offset = len(header)
    for index, (tag, data) in enumerate(sorted_tables):
        record = 12 + index * 16
        raw_tag = tag.encode("ascii", errors="replace")[:4].ljust(4, b" ")
        header[record : record + 4] = raw_tag
        struct.pack_into(">I", header, record + 4, 0)  # checksum は作り直さない
        struct.pack_into(">I", header, record + 8, offset)
        struct.pack_into(">I", header, record + 12, len(data))
        parts.append(bytes(data))
        offset += len(data)
        padding = (4 - (len(data) % 4)) % 4
        if padding:
            parts.append(b"\x00" * padding)
            offset += padding
    return bytes(header) + b"".join(parts)


def _sfnt_from_woff(buf: bytes) -> bytes:
    """woff（zlib 圧縮）を SFNT に戻す。"""
    flavor = _u32(buf, 4)
    num_tables = _u16(buf, 12)
    tables: list[tuple[str, bytes]] = []
    for i in range(num_tables):
        entry = 44 + i * 20
        tag = _ascii(buf, entry, entry + 4)
        offset = _u32(buf, entry + 4)
        compressed = _u32(buf, entry + 8)
        original = _u32(buf, entry + 12)
        raw = buf[offset : offset + compressed]
        # compLength === origLength は «圧縮しない方が小さかった» という意味です。
        tables.append((tag, bytes(raw) if compressed == original else zlib.decompress(raw)))
    return _build_sfnt(flavor, tables)


#: woff2 のテーブル名は 63 種を番号で持つ。並び順が仕様そのものなので変えられません。
WOFF2_KNOWN_TAGS = [
    "cmap", "head", "hhea", "hmtx", "maxp", "name", "OS/2", "post", "cvt ", "fpgm", "glyf", "loca",
    "prep", "CFF ", "VORG", "EBDT", "EBLC", "gasp", "hdmx", "kern", "LTSH", "PCLT", "VDMX", "vhea",
    "vmtx", "BASE", "GDEF", "GPOS", "GSUB", "EBSC", "JSTF", "MATH", "CBDT", "CBLC", "COLR", "CPAL",
    "SVG ", "sbix", "acnt", "avar", "bdat", "bloc", "bsln", "cvar", "fdsc", "feat", "fmtx", "fvar",
    "gvar", "hsty", "just", "lcar", "mort", "morx", "opbd", "prop", "trak", "Zapf", "Silf", "Glat",
    "Gloc", "Feat", "Sill",
]


def _brotli_decompress(data: bytes) -> bytes:
    """Brotli を展開する。無ければ «何を入れればよいか» が分かる例外にする。"""
    try:
        import brotli  # type: ignore[import-not-found]

        return brotli.decompress(data)
    except ImportError:
        pass
    try:
        import brotlicffi  # type: ignore[import-not-found]

        return brotlicffi.decompress(data)
    except ImportError:
        pass
    raise MovoError(
        ErrorCodes.MOVO_UNSUPPORTED,
        "woff2 を読むには brotli が要ります（Python の標準ライブラリには入っていません）",
        hint='pip install brotli（または brotlicffi）を入れるか、woff2 を ttf / otf に変換してから使ってください',
    )


def _read_base128(reader: Reader) -> int:
    """woff2 が使う可変長整数（7 ビットずつ、最上位ビットが継続フラグ）。"""
    value = 0
    for _ in range(5):
        byte = reader.u8()
        value = value * 128 + (byte & 0x7F)
        if (byte & 0x80) == 0:
            return value
    raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "woff2: UIntBase128 が長すぎます")


def _read_255_ushort(reader: Reader) -> int:
    """同じく可変長。1 バイトで 0〜252、2〜3 バイトで 253 以上を表す。"""
    code = reader.u8()
    if code == 253:
        return reader.u16()
    if code == 254:
        return reader.u8() + 253 * 2
    if code == 255:
        return reader.u8() + 253
    return code


class _Woff2Entry:
    """woff2 のテーブルディレクトリの 1 件。"""

    __slots__ = ("tag", "transform", "transformed", "original_length", "transform_length")

    def __init__(
        self, tag: str, transform: int, transformed: bool, original_length: int, transform_length: int
    ) -> None:
        self.tag = tag
        self.transform = transform
        self.transformed = transformed
        self.original_length = original_length
        self.transform_length = transform_length


def _sfnt_from_woff2(buf: bytes) -> bytes:
    """woff2（Brotli 圧縮 + glyf/hmtx 変換）を SFNT に戻す。"""
    flavor = _u32(buf, 4)
    num_tables = _u16(buf, 12)
    reader = Reader(buf, 48)
    directory: list[_Woff2Entry] = []
    for _ in range(num_tables):
        flags = reader.u8()
        index = flags & 0x3F
        if index == 0x3F:
            reader.offset += 4
            tag = _ascii(buf, reader.offset - 4, reader.offset)
        else:
            tag = WOFF2_KNOWN_TAGS[index]
        transform = (flags >> 6) & 0x03
        original_length = _read_base128(reader)
        # glyf / loca だけ «変換されているのが既定»（3 が無変換）。他は 0 が無変換。
        if tag in ("glyf", "loca"):
            transformed = transform != 3
        else:
            transformed = transform != 0
        transform_length = _read_base128(reader) if transformed else original_length
        directory.append(_Woff2Entry(tag, transform, transformed, original_length, transform_length))

    stream = _brotli_decompress(bytes(buf[reader.offset :]))
    cursor = 0
    raw: dict[str, tuple[_Woff2Entry, bytes]] = {}
    for entry in directory:
        raw[entry.tag] = (entry, stream[cursor : cursor + entry.transform_length])
        cursor += entry.transform_length

    tables: list[list[Any]] = []
    rebuilt_loca: bytes | None = None
    for entry in directory:
        data = raw[entry.tag][1]
        if entry.tag == "loca":
            continue  # glyf を戻すときに一緒に作る
        if entry.tag == "glyf" and entry.transformed:
            glyf_data, loca_data = _rebuild_glyf(data)
            rebuilt_loca = loca_data
            tables.append(["glyf", glyf_data])
            continue
        if entry.tag == "hmtx" and entry.transformed:
            tables.append(["hmtx", _rebuild_hmtx(data, raw)])
            continue
        tables.append([entry.tag, bytes(data)])
    if rebuilt_loca is not None:
        tables.append(["loca", rebuilt_loca])
    elif "loca" in raw:
        tables.append(["loca", bytes(raw["loca"][1])])

    # loca は必ず «4 バイト形式» で作り直しているので、head にもそう書いておきます。
    if rebuilt_loca is not None:
        for table in tables:
            if table[0] == "head" and len(table[1]) >= 52:
                head_data = bytearray(table[1])
                struct.pack_into(">h", head_data, 50, 1)
                table[1] = bytes(head_data)
                break
    return _build_sfnt(flavor, [(t[0], t[1]) for t in tables])


def _rebuild_glyf(data: bytes) -> tuple[bytes, bytes]:
    """woff2 の «変換された glyf» を普通の glyf + loca に戻す。

    変換後は、輪郭数・点数・フラグ・座標・命令が別々の列に分かれて入っています
    （同じ種類の数字を隣り合わせにすると Brotli がよく縮むため）。ここでは
    字形ごとに縫い直します。座標は «3 つ組符号化» で 1〜4 バイトに詰められて
    いるので、fontTools と同じ表で解きます。

    書き戻す側は最短形を狙わず、座標を常に 16 ビット差分で書いています。
    ファイルサイズは増えますが、これは «読むためだけ» の一時的な形なので
    短くする意味がありません。

    :returns: ``(glyf のバイト列, loca のバイト列)``
    """
    header = Reader(data, 0)
    header.u16()  # reserved
    header.u16()  # optionFlags
    num_glyphs = header.u16()
    header.u16()  # indexFormat（loca は常に 4 バイト形式で作り直す）
    sizes = [header.u32() for _ in range(7)]
    offset = header.offset
    streams: list[Reader] = []
    for size in sizes:
        streams.append(Reader(data, offset))
        offset += size
    (
        n_contour_stream,
        n_points_stream,
        flag_stream,
        glyph_stream,
        composite_stream,
        bbox_stream,
        instruction_stream,
    ) = streams

    bbox_bitmap_bytes = -(-num_glyphs // 8)  # ceil
    bbox_bitmap = data[bbox_stream.offset : bbox_stream.offset + bbox_bitmap_bytes]
    bbox_stream.offset += bbox_bitmap_bytes

    glyphs: list[bytes] = []
    for g in range(num_glyphs):
        number_of_contours = n_contour_stream.i16()
        has_bbox = (bbox_bitmap[g >> 3] & (0x80 >> (g & 7))) != 0
        if number_of_contours == 0:
            glyphs.append(b"")
            continue
        if number_of_contours < 0:
            # 合成グリフ。中身はそのまま写して、命令だけ別の列から拾います。
            start = composite_stream.offset
            more = True
            while more:
                flags = composite_stream.u16()
                composite_stream.u16()  # glyphIndex
                composite_stream.offset += 4 if (flags & 1) else 2
                if flags & 8:
                    composite_stream.offset += 2
                elif flags & 0x40:
                    composite_stream.offset += 4
                elif flags & 0x80:
                    composite_stream.offset += 8
                more = (flags & 0x20) != 0
            body = data[start : composite_stream.offset]
            instruction_length = _read_255_ushort(glyph_stream)
            instructions = data[instruction_stream.offset : instruction_stream.offset + instruction_length]
            instruction_stream.offset += instruction_length
            box = _read_bbox(bbox_stream) if has_bbox else [0, 0, 0, 0]
            out = bytearray(10 + len(body) + 2 + instruction_length)
            struct.pack_into(">h", out, 0, number_of_contours)
            _write_bbox(out, box)
            out[10 : 10 + len(body)] = body
            struct.pack_into(">H", out, 10 + len(body), instruction_length)
            out[12 + len(body) : 12 + len(body) + instruction_length] = instructions
            glyphs.append(bytes(out))
            continue

        end_points: list[int] = []
        total = 0
        for _ in range(number_of_contours):
            total += _read_255_ushort(n_points_stream)
            end_points.append(total - 1)
        flags_arr = bytearray(total)
        for i in range(total):
            flags_arr[i] = flag_stream.u8()
        xs = [0] * total
        ys = [0] * total
        x = 0
        y = 0
        for i in range(total):
            dx, dy = _read_triplet(flags_arr[i], glyph_stream)
            x += dx
            y += dy
            xs[i] = x
            ys[i] = y
        instruction_length = _read_255_ushort(glyph_stream)
        instructions = data[instruction_stream.offset : instruction_stream.offset + instruction_length]
        instruction_stream.offset += instruction_length
        box = _read_bbox(bbox_stream) if has_bbox else _bounds_of(xs, ys)
        glyphs.append(_write_simple_glyph(number_of_contours, end_points, flags_arr, xs, ys, box, instructions))

    loca = bytearray((num_glyphs + 1) * 4)
    parts: list[bytes] = []
    position = 0
    for g in range(num_glyphs):
        struct.pack_into(">I", loca, g * 4, position)
        parts.append(glyphs[g])
        position += len(glyphs[g])
        padding = (4 - (len(glyphs[g]) % 4)) % 4
        if padding:
            parts.append(b"\x00" * padding)
            position += padding
    struct.pack_into(">I", loca, num_glyphs * 4, position)
    return b"".join(parts), bytes(loca)


def _read_bbox(reader: Reader) -> list[int]:
    return [reader.i16(), reader.i16(), reader.i16(), reader.i16()]


def _write_bbox(target: bytearray, box: Sequence[int]) -> None:
    struct.pack_into(">h", target, 2, _clamp_int16(box[0]))
    struct.pack_into(">h", target, 4, _clamp_int16(box[1]))
    struct.pack_into(">h", target, 6, _clamp_int16(box[2]))
    struct.pack_into(">h", target, 8, _clamp_int16(box[3]))


def _bounds_of(xs: Sequence[int], ys: Sequence[int]) -> list[int]:
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [min(xs), min(ys), max(xs), max(ys)]


def _read_triplet(flag: int, stream: Reader) -> tuple[int, int]:
    """3 つ組符号化の 1 点ぶん。表の区切り（10 / 20 / 84 / 120 / 124）は仕様どおりです。"""
    code = flag & 0x7F

    def sign(value: int, bit: int) -> int:
        return value if (bit & 1) else -value

    dx = 0
    dy = 0
    if code < 10:
        dy = sign(((code & 14) << 7) + stream.u8(), code)
    elif code < 20:
        dx = sign((((code - 10) & 14) << 7) + stream.u8(), code)
    elif code < 84:
        b0 = code - 20
        b1 = stream.u8()
        dx = sign(1 + (b0 & 0x30) + (b1 >> 4), code)
        dy = sign(1 + ((b0 & 0x0C) << 2) + (b1 & 0x0F), code >> 1)
    elif code < 120:
        b0 = code - 84
        # JS 版は dx 用の 1 バイトを先に読みます。読む順を変えないこと。
        dx = sign(1 + ((b0 // 12) << 8) + stream.u8(), code)
        dy = sign(1 + (((b0 % 12) >> 2) << 8) + stream.u8(), code >> 1)
    elif code < 124:
        b0 = stream.u8()
        b1 = stream.u8()
        b2 = stream.u8()
        dx = sign((b0 << 4) + (b1 >> 4), code)
        dy = sign(((b1 & 0x0F) << 8) + b2, code >> 1)
    else:
        b0 = stream.u8()
        b1 = stream.u8()
        dx = sign((b0 << 8) + b1, code)
        b2 = stream.u8()
        b3 = stream.u8()
        dy = sign((b2 << 8) + b3, code >> 1)
    return dx, dy


def _write_simple_glyph(
    number_of_contours: int,
    end_points: Sequence[int],
    flags: Sequence[int],
    xs: Sequence[int],
    ys: Sequence[int],
    box: Sequence[int],
    instructions: bytes,
) -> bytes:
    points = len(xs)
    size = 10 + number_of_contours * 2 + 2 + len(instructions) + points * 5
    out = bytearray(size)
    struct.pack_into(">h", out, 0, number_of_contours)
    _write_bbox(out, box)
    p = 10
    for end in end_points:
        struct.pack_into(">H", out, p, end & 0xFFFF)
        p += 2
    struct.pack_into(">H", out, p, len(instructions))
    p += 2
    out[p : p + len(instructions)] = instructions
    p += len(instructions)
    # フラグは «曲線上か» だけ。短縮ビットを立てないので座標は常に 16 ビット差分。
    for i in range(points):
        out[p] = 1 if (flags[i] & 0x80) == 0 else 0
        p += 1
    previous = 0
    for i in range(points):
        struct.pack_into(">h", out, p, _clamp_int16(xs[i] - previous))
        previous = xs[i]
        p += 2
    previous = 0
    for i in range(points):
        struct.pack_into(">h", out, p, _clamp_int16(ys[i] - previous))
        previous = ys[i]
        p += 2
    return bytes(out[:p])


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def _rebuild_hmtx(data: bytes, raw: dict[str, tuple[Any, bytes]]) -> bytes:
    """hmtx の変換を戻す。

    変換後は «左サイドベアリングが省かれている» だけで、advanceWidth はそのまま
    入っています。Movo は字送りにしか hmtx を使わないので、省かれた lsb は 0 で
    埋めています（本来は字形の xMin と同じ値です）。
    """
    hhea = raw["hhea"][1] if "hhea" in raw else None
    maxp = raw["maxp"][1] if "maxp" in raw else None
    if not hhea or not maxp:
        return bytes(data)
    number_of_h_metrics = _u16(hhea, 34)
    num_glyphs = _u16(maxp, 4)
    flags = data[0]
    cursor = 1
    out = bytearray(number_of_h_metrics * 4 + max(0, num_glyphs - number_of_h_metrics) * 2)
    for i in range(number_of_h_metrics):
        struct.pack_into(">H", out, i * 4, _u16(data, cursor))
        cursor += 2
    if (flags & 1) == 0:
        for i in range(number_of_h_metrics):
            struct.pack_into(">h", out, i * 4 + 2, _i16(data, cursor))
            cursor += 2
    if (flags & 2) == 0:
        for i in range(number_of_h_metrics, num_glyphs):
            struct.pack_into(
                ">h", out, number_of_h_metrics * 4 + (i - number_of_h_metrics) * 2, _i16(data, cursor)
            )
            cursor += 2
    return bytes(out)


# ==========================================================================
# システムのフォント探索（movo.core.platform が無いときの代用込み）
# ==========================================================================

#: 探索する拡張子。woff / woff2 はシステムには置かれないので入れません。
_SYSTEM_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")


def _system_font_dirs() -> list[str]:
    """この OS でシステムフォントが置かれているディレクトリ。"""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR") or "C:\\Windows"
        local = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        dirs = [
            os.path.join(windir, "Fonts"),
            os.path.join(local, "Microsoft", "Windows", "Fonts"),
        ]
    elif sys.platform == "darwin":
        dirs = [
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
            "/Library/Fonts",
            os.path.join(home, "Library", "Fonts"),
        ]
    else:
        dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.join(home, ".fonts"),
            os.path.join(home, ".local", "share", "fonts"),
        ]
    return [d for d in dirs if os.path.isdir(d)]


def _list_font_files_fallback(extra_dirs: Iterable[str] = (), limit: int = 4000) -> list[str]:
    """movo.core.platform が無いときの ``listFontFiles``。

    深さ 4 まで、件数は ``limit`` で頭打ち。フォントが数万個あるマシンでも
    起動が重くならないようにするためです。
    """
    found: list[str] = []
    seen: set[str] = set()

    def walk(directory: str, depth: int) -> None:
        if len(found) >= limit or depth > 4:
            return
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if len(found) >= limit:
                return
            full = os.path.join(directory, entry.name)
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if is_dir:
                walk(full, depth + 1)
            elif entry.name.lower().endswith(_SYSTEM_FONT_EXTENSIONS) and full.lower() not in seen:
                seen.add(full.lower())
                found.append(full)

    for directory in [*extra_dirs, *_system_font_dirs()]:
        walk(directory, 0)
    return found


try:  # pragma: no cover - platform.py が先にできていればそちらを使う
    from movo.core.platform import list_font_files as _platform_list_font_files  # type: ignore
except Exception:  # pragma: no cover
    _platform_list_font_files = None  # type: ignore[assignment]


def list_font_files(extra_dirs: Iterable[str] = (), limit: int = 4000) -> list[str]:
    """システム（と追加ディレクトリ）にあるフォントファイルを列挙する。"""
    if _platform_list_font_files is not None:
        try:
            return list(_platform_list_font_files(extra_dirs, limit))
        except Exception:
            pass
    return _list_font_files_fallback(extra_dirs, limit)


# ==========================================================================
# フォント本体
# ==========================================================================


class Glyph:
    """字形の輪郭（フォント単位）。

    :ivar contours: 各要素が ``(n, 3)`` の float64 配列。列は x, y, on_curve(0.0/1.0)。
        **点が 0 個の輪郭は入りません。**
    :ivar advance: 字送り幅（フォント単位）
    :ivar x_min: 字形の左端
    :ivar x_max: 字形の右端
    """

    __slots__ = ("contours", "advance", "x_min", "x_max")

    def __init__(
        self,
        contours: list[np.ndarray] | None = None,
        advance: float = 0.0,
        x_min: float = 0.0,
        x_max: float = 0.0,
    ) -> None:
        self.contours: list[np.ndarray] = contours if contours is not None else []
        self.advance = advance
        self.x_min = x_min
        self.x_max = x_max

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"Glyph(contours={len(self.contours)}, advance={self.advance})"


class Font:
    """読み込み済みのフォント 1 面。"""

    def __init__(self, buf: bytes, file_path: str | None = None, face_index: int = 0) -> None:
        self.buffer = to_sfnt(bytes(buf))
        self.file_path = file_path
        self.face_index = face_index
        self.tables: dict[str, tuple[int, int]] = {}
        self._glyph_cache: dict[int, Glyph] = {}
        self._cmap: dict[int, int] | None = None
        self.is_cff = False
        self.names: dict[int, str] = {}
        self._parse()

    @classmethod
    def load(cls, file_path: str, face_index: int = 0) -> "Font":
        """ファイルから読む。"""
        with open(file_path, "rb") as handle:
            buf = handle.read()
        return cls(buf, file_path=file_path, face_index=face_index)

    # ---------------------------------------------------------------- parse

    def _parse(self) -> None:
        reader = Reader(self.buffer)
        tag = reader.u32()
        if tag == 0x74746366:
            # 'ttcf' — font collection
            reader.u32()  # version
            count = reader.u32()
            index = min(self.face_index, count - 1)
            reader.seek(12 + index * 4)
            offset = reader.u32()
            reader.seek(offset)
            tag = reader.u32()
        if tag not in (0x00010000, 0x74727565, 0x4F54544F):
            raise MovoError(
                ErrorCodes.MOVO_FONT_NOT_FOUND,
                f"unsupported font format in {self.file_path or 'buffer'}",
            )
        self.is_cff = tag == 0x4F54544F
        num_tables = reader.u16()
        reader.offset += 6
        for _ in range(num_tables):
            name = _ascii(self.buffer, reader.offset, reader.offset + 4)
            reader.offset += 4
            reader.u32()  # checksum
            offset = reader.u32()
            length = reader.u32()
            self.tables[name] = (offset, length)
        if "glyf" not in self.tables or "loca" not in self.tables:
            base = os.path.basename(self.file_path or "font")
            raise MovoError(
                ErrorCodes.MOVO_FONT_NOT_FOUND,
                f"{base} uses PostScript outlines (CFF), which the built-in rasteriser cannot draw",
            )

        # head: unitsPerEm は 18 バイト目、indexToLocFormat は 50 バイト目。
        head = Reader(self.buffer, self.tables["head"][0])
        head.offset += 18
        self.units_per_em = head.u16() or 1000
        head.offset += 30
        self.index_to_loc_format = head.i16()

        # maxp: numGlyphs は 4 バイト目。
        maxp = Reader(self.buffer, self.tables["maxp"][0])
        maxp.offset += 4
        self.num_glyphs = maxp.u16()

        # hhea: ascender は 4 バイト目、numberOfHMetrics は 34 バイト目。
        hhea = Reader(self.buffer, self.tables["hhea"][0])
        hhea.offset += 4
        self.ascender = hhea.i16()
        self.descender = hhea.i16()
        self.line_gap = hhea.i16()
        # advanceWidthMax, bearings, xMaxExtent, caret*, 4 reserved, metricDataFormat
        hhea.offset += 24
        self.number_of_h_metrics = max(1, hhea.u16())

        self._parse_loca()
        self._parse_name()

    def _parse_loca(self) -> None:
        offset, length = self.tables["loca"]
        count = self.num_glyphs + 1
        self.loca = np.zeros(count, dtype=np.uint32)
        if self.index_to_loc_format == 0:
            # 短い形式は «2 バイト単位» なので 2 倍します。
            for i in range(count):
                if not (i * 2 + 1 < length + 2):
                    break
                if offset + i * 2 + 2 > len(self.buffer):
                    break
                self.loca[i] = _u16(self.buffer, offset + i * 2) * 2
        else:
            for i in range(count):
                if offset + i * 4 + 4 > len(self.buffer):
                    break
                self.loca[i] = _u32(self.buffer, offset + i * 4)

    def _parse_name(self) -> None:
        """name テーブルを読む。

        platformId 3（Windows）と 0（Unicode）は UTF-16BE、それ以外は latin1。
        同じ nameId が複数あるときは **Windows のものを優先** します（日本語
        フォントで «MS ゴシック» 側が入っている方が読みやすいため）。
        """
        self.names = {}
        table = self.tables.get("name")
        if table is None:
            self.family_name = os.path.basename(self.file_path or "font")
            self.subfamily_name = "Regular"
            self.full_name = f"{self.family_name} {self.subfamily_name}"
            return
        table_offset, _table_length = table
        reader = Reader(self.buffer, table_offset)
        reader.u16()  # format
        count = reader.u16()
        string_offset = reader.u16()
        for _ in range(count):
            platform_id = reader.u16()
            encoding_id = reader.u16()
            reader.u16()  # language
            name_id = reader.u16()
            length = reader.u16()
            offset = reader.u16()
            start = table_offset + string_offset + offset
            if start + length > len(self.buffer):
                continue
            chunk = self.buffer[start : start + length]
            if platform_id == 3 or (platform_id == 0 and encoding_id != 0):
                value = chunk.decode("utf-16-be", errors="replace")
            else:
                value = chunk.decode("latin-1", errors="replace")
            if not self.names.get(name_id) or platform_id == 3:
                self.names[name_id] = value
        self.family_name = (
            self.names.get(16) or self.names.get(1) or os.path.basename(self.file_path or "font")
        )
        self.subfamily_name = self.names.get(17) or self.names.get(2) or "Regular"
        self.full_name = self.names.get(4) or f"{self.family_name} {self.subfamily_name}"

    # ----------------------------------------------------------------- cmap

    def _ensure_cmap(self) -> dict[int, int]:
        """cmap を «文字 → 字形番号» の dict に展開する（初回のみ）。

        サブテーブルの点数付けは JS 版のまま: 3/10 → 5、3/1 → 4、0/* → 3、他 1。
        """
        if self._cmap is not None:
            return self._cmap
        mapping: dict[int, int] = {}
        table = self.tables.get("cmap")
        if table is None:
            self._cmap = mapping
            return mapping
        table_offset, _ = table
        reader = Reader(self.buffer, table_offset)
        reader.u16()  # version
        num_tables = reader.u16()
        best: tuple[int, int] | None = None  # (score, offset)
        for _ in range(num_tables):
            platform_id = reader.u16()
            encoding_id = reader.u16()
            offset = reader.u32()
            if platform_id == 3 and encoding_id == 10:
                score = 5
            elif platform_id == 3 and encoding_id == 1:
                score = 4
            elif platform_id == 0:
                score = 3
            else:
                score = 1
            if best is None or score > best[0]:
                best = (score, table_offset + offset)
        if best is None:
            self._cmap = mapping
            return mapping

        sub = Reader(self.buffer, best[1])
        fmt = sub.u16()
        if fmt == 4:
            sub.u16()  # length
            sub.u16()  # language
            seg_count_x2 = sub.u16()
            seg_count = seg_count_x2 // 2
            sub.offset += 6  # searchRange, entrySelector, rangeShift
            end_codes = [sub.u16() for _ in range(seg_count)]
            sub.u16()  # reservedPad
            start_codes = [sub.u16() for _ in range(seg_count)]
            id_deltas = [sub.i16() for _ in range(seg_count)]
            id_range_offset_pos = sub.offset
            id_range_offsets = [sub.u16() for _ in range(seg_count)]
            buffer_length = len(self.buffer)
            for i in range(seg_count):
                start = start_codes[i]
                end = end_codes[i]
                if start == 0xFFFF:
                    continue
                delta = id_deltas[i]
                range_offset = id_range_offsets[i]
                code = start
                while code <= end and code != 0x10000:
                    if range_offset == 0:
                        glyph = (code + delta) & 0xFFFF
                    else:
                        address = id_range_offset_pos + i * 2 + range_offset + (code - start) * 2
                        if address + 1 >= buffer_length:
                            code += 1
                            continue
                        glyph = _u16(self.buffer, address)
                        if glyph != 0:
                            glyph = (glyph + delta) & 0xFFFF
                    if glyph:
                        mapping[code] = glyph
                    code += 1
        elif fmt == 12:
            sub.u16()  # reserved
            sub.u32()  # length
            sub.u32()  # language
            n_groups = sub.u32()
            for _ in range(n_groups):
                start_char = sub.u32()
                end_char = sub.u32()
                start_glyph = sub.u32()
                span = min(end_char - start_char, 0x10000)
                for c in range(span + 1):
                    mapping[start_char + c] = start_glyph + c
        elif fmt == 6:
            sub.u16()  # length
            sub.u16()  # language
            first = sub.u16()
            count = sub.u16()
            for i in range(count):
                mapping[first + i] = sub.u16()
        elif fmt == 0:
            sub.u16()  # length
            sub.u16()  # language
            for i in range(256):
                mapping[i] = sub.u8()

        self._cmap = mapping
        return mapping

    def glyph_index_for(self, code_point: int) -> int:
        """文字コードから字形番号を引く。無ければ 0（.notdef）。"""
        return self._ensure_cmap().get(code_point, 0)

    def has_glyph(self, code_point: int) -> bool:
        """その文字の字形を持っているか。"""
        return self.glyph_index_for(code_point) != 0

    def character_count(self) -> int:
        """cmap に載っている文字数。«字数» は収録範囲の目安として字形数より当てになります。"""
        return len(self._ensure_cmap())

    def advance_width(self, glyph_index: int) -> float:
        """字送り幅（フォント単位）。"""
        table = self.tables.get("hmtx")
        if table is None:
            return self.units_per_em / 2
        index = min(glyph_index, self.number_of_h_metrics - 1)
        offset = table[0] + index * 4
        if offset + 1 >= len(self.buffer):
            return self.units_per_em / 2
        return float(_u16(self.buffer, offset))

    # ---------------------------------------------------------------- glyph

    def glyph(self, glyph_index: int, depth: int = 0) -> Glyph:
        """字形の輪郭をフォント単位で返す。

        :param glyph_index: 字形番号
        :param depth: 合成グリフの入れ子の深さ（内部用。5 を超えたら打ち切り）
        """
        cached = self._glyph_cache.get(glyph_index)
        if cached is not None:
            return cached
        result = Glyph([], self.advance_width(glyph_index), 0.0, 0.0)
        if glyph_index >= self.num_glyphs or depth > 5:
            self._glyph_cache[glyph_index] = result
            return result
        glyf_offset = self.tables["glyf"][0]
        start = glyf_offset + int(self.loca[glyph_index])
        end = glyf_offset + int(self.loca[glyph_index + 1])
        if end <= start or end > len(self.buffer):
            self._glyph_cache[glyph_index] = result
            return result

        reader = Reader(self.buffer, start)
        number_of_contours = reader.i16()
        result.x_min = float(reader.i16())
        reader.i16()  # yMin
        result.x_max = float(reader.i16())
        reader.i16()  # yMax

        if number_of_contours >= 0:
            end_pts = [reader.u16() for _ in range(number_of_contours)]
            num_points = (end_pts[-1] + 1) if number_of_contours else 0
            instruction_length = reader.u16()
            reader.offset += instruction_length

            flags = bytearray(num_points)
            i = 0
            while i < num_points:
                flag = reader.u8()
                flags[i] = flag
                i += 1
                if flag & 8:
                    repeat = reader.u8()
                    while repeat > 0 and i < num_points:
                        flags[i] = flag
                        i += 1
                        repeat -= 1

            xs = np.zeros(num_points, dtype=np.float64)
            x = 0
            for i in range(num_points):
                flag = flags[i]
                if flag & 2:
                    delta = reader.u8()
                    x += delta if (flag & 16) else -delta
                elif not (flag & 16):
                    x += reader.i16()
                # 16 ビットに丸めるのは JS の Int16Array と同じ挙動にするためです。
                xs[i] = _wrap_int16(x)
            ys = np.zeros(num_points, dtype=np.float64)
            y = 0
            for i in range(num_points):
                flag = flags[i]
                if flag & 4:
                    delta = reader.u8()
                    y += delta if (flag & 32) else -delta
                elif not (flag & 32):
                    y += reader.i16()
                ys[i] = _wrap_int16(y)

            point_index = 0
            for c in range(number_of_contours):
                last = end_pts[c]
                count = last - point_index + 1
                if count > 0:
                    contour = np.empty((count, 3), dtype=np.float64)
                    contour[:, 0] = xs[point_index : last + 1]
                    contour[:, 1] = ys[point_index : last + 1]
                    for k in range(count):
                        contour[k, 2] = 1.0 if (flags[point_index + k] & 1) else 0.0
                    # «点 0 個の輪郭» は入れません（後段が空配列を踏まないように）。
                    result.contours.append(contour)
                # endPts が逆順に壊れているフォントでも巻き戻らないようにします。
                point_index = max(last + 1, point_index)
        else:
            # 合成グリフ。部品の輪郭に 2x2 の行列と平行移動をかけて足していきます。
            while True:
                flags16 = reader.u16()
                component_index = reader.u16()
                if flags16 & 1:
                    dx = reader.i16()
                    dy = reader.i16()
                else:
                    dx = reader.i8()
                    dy = reader.i8()
                a = 1.0
                b = 0.0
                c = 0.0
                d = 1.0
                if flags16 & 8:
                    a = d = reader.i16() / 16384
                elif flags16 & 0x40:
                    a = reader.i16() / 16384
                    d = reader.i16() / 16384
                elif flags16 & 0x80:
                    a = reader.i16() / 16384
                    b = reader.i16() / 16384
                    c = reader.i16() / 16384
                    d = reader.i16() / 16384
                component = self.glyph(component_index, depth + 1)
                for contour in component.contours:
                    moved = np.empty_like(contour)
                    moved[:, 0] = a * contour[:, 0] + c * contour[:, 1] + dx
                    moved[:, 1] = b * contour[:, 0] + d * contour[:, 1] + dy
                    moved[:, 2] = contour[:, 2]
                    result.contours.append(moved)
                if not (flags16 & 0x20):
                    break

        self._glyph_cache[glyph_index] = result
        return result

    @property
    def metrics(self) -> dict[str, int]:
        """行の高さを決めるための数値。"""
        return {
            "units_per_em": self.units_per_em,
            "ascender": self.ascender,
            "descender": self.descender,
            "line_gap": self.line_gap,
        }

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"Font({self.full_name!r}, upem={self.units_per_em})"


def _wrap_int16(value: int) -> int:
    """JS の ``Int16Array`` と同じ折り返し（座標が壊れたフォントで暴れないように）。"""
    value &= 0xFFFF
    return value - 0x10000 if value >= 0x8000 else value


# ==========================================================================
# 文字の種類と、フォントの選び方
# ==========================================================================

CJK_RANGES = (
    (0x3000, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)


def is_cjk(code_point: int) -> bool:
    """CJK（漢字・かな・全角記号）の範囲か。"""
    return any(lo <= code_point <= hi for lo, hi in CJK_RANGES)


PREFERRED_LATIN = [
    "arial", "helvetica", "dejavusans", "liberationsans", "segoeui", "verdana", "roboto", "notosans",
]
PREFERRED_CJK = [
    "yugothm",
    "yugothic",
    "meiryo",
    "msgothic",
    "msmincho",
    "notosanscjk",
    "notosansjp",
    "hiraginosans",
    "hiragino",
    "sourcehansans",
    "droidsansfallback",
    "wqy",
]

#: CSS 由来の総称名。«どのフォントか» ではなく «既定でよい» の意味なので、
#: フォールバック配列の末尾に書けるようにここで受け止めます。
GENERIC_FAMILIES = frozenset(
    {"sans-serif", "sans", "serif", "monospace", "system-ui", "cursive", "fantasy"}
)

_MISSING = object()


def _normalise(text: str) -> str:
    """比較用に «英数字だけの小文字» にする。"""
    return re.sub(r"[^a-z0-9]", "", text.lower())


class FontManager:
    """フォントを読み、家族名から面を選び、欠字のときに別の面へ落とす。"""

    def __init__(
        self,
        project_root: str | None = None,
        font_dirs: Iterable[str] = (),
        fonts: dict[str, Any] | None = None,
    ) -> None:
        self.project_root = project_root if project_root is not None else os.getcwd()
        self.declared: dict[str, Any] = fonts or {}
        self.extra_dirs = list(font_dirs)
        self._by_key: dict[str, Font | None] = {}
        self._files: list[str] | None = None
        self._fallback_chain: list[Font] | None = None
        self._warned: set[str] = set()
        # «明示フォールバック» の並び。style.family に配列を書いたときだけ入ります。
        # 先頭のフォントを鍵にしているのは、text.py が resolve() の戻り値しか
        # 持ち回らないためです。resolve() → font_for_code_point() は同じレイアウトの
        # 中で続けて呼ばれるので、この持ち方で «そのレイヤーの並び» になります。
        self._explicit_chains: "weakref.WeakKeyDictionary[Font, list[Font]]" = (
            weakref.WeakKeyDictionary()
        )

    # ------------------------------------------------------------ 読み込み

    def _font_files(self) -> list[str]:
        if self._files is None:
            self._files = list_font_files(self.extra_dirs)
        return self._files

    def _load_file(self, file_path: str, face_index: int = 0) -> Font | None:
        """1 ファイル読む。読めなければ None（CFF などは «次を試す» ため）。"""
        key = f"{file_path}#{face_index}"
        if key in self._by_key:
            return self._by_key[key]
        font: Font | None
        try:
            font = Font.load(file_path, face_index)
        except Exception as err:  # CFF・壊れたファイル・権限なし、どれも同じ扱い
            if file_path not in self._warned:
                self._warned.add(file_path)
                _log_debug(f"font skipped: {file_path} ({err})")
            font = None
        self._by_key[key] = font
        return font

    def _absolute(self, file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path
        return os.path.abspath(os.path.join(self.project_root, file_path))

    # -------------------------------------------------------------- 面選び

    def _declared_face(
        self, declared: Any, bold: bool = False, italic: bool = False
    ) -> dict[str, Any] | None:
        """``project.fonts`` の 1 件から «いま欲しい太さ・斜体» のファイルを選ぶ。

        1 家族 = 1 ファイルのままだと weight:"bold" が合成ボールド（線を太らせる
        だけ）になり、本物のボールド字形が使えません。ウェイトごとにファイルを
        分けて書けるようにしています。

        :returns: ``{"file": ..., "index": ...}`` または None
        """
        if not declared:
            return None
        if isinstance(declared, str):
            return {"file": declared, "index": 0}
        if not isinstance(declared, dict):
            return None
        if declared.get("file"):
            return {"file": declared["file"], "index": declared.get("index", 0)}

        bold = bool(bold)
        italic = bool(italic)

        def pick(*keys: str) -> dict[str, Any] | None:
            for key in keys:
                value = declared.get(key)
                if isinstance(value, str):
                    return {"file": value, "index": 0}
                if isinstance(value, dict) and value.get("file"):
                    return {"file": value["file"], "index": value.get("index", 0)}
            return None

        # 欲しい面が無ければ «近い方» へ降ります。bold-italic → bold → italic →
        # regular の順に落とすのは、太さの方が字面の印象を左右するためです。
        order: list[str] = []
        if bold and italic:
            order += ["boldItalic", "bolditalic", "italicBold", "700i"]
        if bold:
            order += ["bold", "700", "600"]
        if italic:
            order += ["italic", "oblique", "400i"]
        order += ["regular", "normal", "400", "book"]
        chosen = pick(*order)
        if chosen:
            return chosen
        # 名前が想定外でも «最初に書いてあるファイル» で描けた方が親切です。
        return pick(*declared.keys())

    def resolve(
        self, family: str | Sequence[str] | None = None, bold: bool = False, italic: bool = False
    ) -> Font:
        """家族名（または ``project.fonts`` に書いたパス）から Font を引く。

        ``family`` にリストを渡すと «明示フォールバック» になります。暗黙のシステム
        検索は環境によって別の字形になるので、並べて書けるようにしています。
        リストのときは先頭の読めたフォントを返しつつ、残りを欠字時の並びとして
        覚えておきます。
        """
        if isinstance(family, (list, tuple)):
            chain: list[Font] = []
            for entry in family:
                font = self._resolve_one(entry, bold, italic, quiet=True)
                if font is not None and not any(font is existing for existing in chain):
                    chain.append(font)
            if not chain:
                return self.default_font(bold, italic)
            # 明示した並びで足りなかったときのために、既定の並びも後ろに足します。
            self._explicit_chains[chain[0]] = [*chain[1:], *self.fallback_chain()]
            return chain[0]
        font = self._resolve_one(family, bold, italic, quiet=False)
        return font if font is not None else self.default_font(bold, italic)

    def _resolve_one(
        self, family: Any, bold: bool = False, italic: bool = False, quiet: bool = False
    ) -> Font | None:
        """1 つの家族名を解決する。

        :param quiet: リスト指定のときは «見つからない» を警告しない（次を試すため）
        """
        wanted = (family or "").strip() if isinstance(family, str) else ""
        if not wanted:
            return None
        if wanted.lower() in GENERIC_FAMILIES:
            return self.default_font(bold, italic)

        declared = self._declared_face(self.declared.get(wanted), bold, italic)
        if declared:
            absolute = self._absolute(declared["file"])
            font = self._load_file(absolute, declared["index"])
            if font is not None:
                return font
        if re.search(r"[\\/]", wanted) or FONT_EXTENSIONS.search(wanted):
            absolute = self._absolute(wanted)
            if os.path.exists(absolute):
                font = self._load_file(absolute)
                if font is not None:
                    return font
        match = self._find_by_family(wanted, bold, italic)
        if match is not None:
            return match
        if not quiet and f"family:{wanted}" not in self._warned:
            self._warned.add(f"family:{wanted}")
            _log_warn(f'font family "{wanted}" was not found; using the default face instead')
        return None

    def _find_by_family(self, family: str, bold: bool = False, italic: bool = False) -> Font | None:
        """ファイル名と name テーブルの両方を見て、一番それらしい面を選ぶ。

        点数付けは JS 版そのまま: 家族名が完全一致 +10 / 部分一致 +5、
        太さが合っていれば +3、斜体が合っていれば +3。
        """
        target = _normalise(family)
        want_bold = bool(bold)
        want_italic = bool(italic)
        # 「メイリオ」のように英数字が 1 文字も無い家族名だと target が空文字になり、
        # includes('') がどのファイルにも当たって «最初に見つかったフォント» が
        # 返ってしまいます。その場合はファイル名ではなく家族名そのもので突き合わせます。
        raw = str(family or "").strip().lower()
        best: tuple[int, Font] | None = None
        for file in self._font_files():
            base = _normalise(os.path.splitext(os.path.basename(file))[0])
            if target:
                if target not in base and base not in target:
                    continue
            else:
                font = self._load_file(file)
                if font is None or (font.family_name or "").lower() != raw:
                    continue
                return font
            font = self._load_file(file)
            if font is None:
                continue
            font_family = _normalise(font.family_name or "")
            sub = (font.subfamily_name or "").lower()
            is_bold = bool(re.search(r"bold", sub)) or bool(re.search(r"bold", base))
            is_italic = bool(re.search(r"italic|oblique", sub)) or bool(
                re.search(r"italic|oblique", base)
            )
            score = 0
            if font_family == target:
                score += 10
            elif target in font_family:
                score += 5
            if is_bold == want_bold:
                score += 3
            if is_italic == want_italic:
                score += 3
            if best is None or score > best[0]:
                best = (score, font)
        return best[1] if best is not None else None

    def default_font(self, bold: bool = False, italic: bool = False) -> Font:
        """何も指定されなかったときの面。Arial 系を先に探します。"""
        key = f"default:{'b' if bold else ''}{'i' if italic else ''}"
        if key in self._by_key:
            cached = self._by_key[key]
            if cached is not None:
                return cached
        files = self._font_files()
        chosen: Font | None = None
        for preferred in PREFERRED_LATIN:
            for file in files:
                base = _normalise(os.path.splitext(os.path.basename(file))[0])
                if not base.startswith(preferred):
                    continue
                is_bold = bool(re.search(r"bold|bd$", base))
                is_italic = bool(re.search(r"italic|oblique|i$", base))
                if bool(bold) != is_bold or bool(italic) != is_italic:
                    continue
                font = self._load_file(file)
                if font is not None:
                    chosen = font
                    break
            if chosen is not None:
                break
        if chosen is None:
            # 好みの面が 1 つも無いマシンでも «とにかく描ける» ようにします。
            for file in files:
                font = self._load_file(file)
                if font is not None:
                    chosen = font
                    break
        if chosen is None:
            raise MovoError(
                ErrorCodes.MOVO_FONT_NOT_FOUND,
                "no usable TrueType font was found on this system",
                hint='declare one in project.fonts, e.g. {"fonts": {"Main": "assets/fonts/MyFont.ttf"}}',
            )
        self._by_key[key] = chosen
        return chosen

    def fallback_chain(self) -> list[Font]:
        """主フォントに字形が無いときに順に試す面の並び。"""
        if self._fallback_chain is not None:
            return self._fallback_chain
        chain: list[Font] = []
        files = self._font_files()
        for preferred in [*PREFERRED_CJK, *PREFERRED_LATIN]:
            for file in files:
                base = _normalise(os.path.splitext(os.path.basename(file))[0])
                if not base.startswith(preferred):
                    continue
                font = self._load_file(file)
                if font is not None and not any(font is existing for existing in chain):
                    chain.append(font)
                break
        self._fallback_chain = chain
        return chain

    def font_for_code_point(self, primary: Font | None, code_point: int) -> Font | None:
        """``code_point`` を実際に描ける面を選ぶ。

        family をリストで書いた場合は «そこに並べた順» を先に見ます。書いた
        とおりの字形で出したいから並べているので、システムから拾った並びが
        割り込んではいけません。
        """
        if primary is not None and primary.has_glyph(code_point):
            return primary
        explicit = self._explicit_chains.get(primary) if primary is not None else None
        for font in (explicit if explicit is not None else self.fallback_chain()):
            if font.has_glyph(code_point):
                return font
        return primary

    # ------------------------------------------------------------ 診断まわり

    def check_font(self, file: str) -> dict[str, Any]:
        """フォントファイルを 1 つ検査する（``movo fonts --check`` 用）。

        読めるかどうかだけでなく «収録字数» を返すのは、日本語フォントで
        «この字が入っていない» が起きたときに、まず疑うのがここだからです。
        """
        absolute = self._absolute(file)
        try:
            font = Font.load(absolute)
        except Exception as error:
            return {"file": absolute, "ok": False, "error": str(error)}
        return {
            "file": absolute,
            "ok": True,
            "family": font.family_name,
            "subfamily": font.subfamily_name,
            "glyphs": font.num_glyphs,
            "characters": font.character_count(),
            "units_per_em": font.units_per_em,
            # JS 版の JSON 出力と突き合わせられるよう、camelCase の別名も残します。
            "unitsPerEm": font.units_per_em,
        }

    def missing_glyphs(
        self,
        text: Any,
        family: str | Sequence[str] | None = None,
        bold: bool = False,
        italic: bool = False,
    ) -> list[str]:
        """その文字列のうち «どのフォントでも描けない字» を返す（``movo fonts --missing`` 用）。

        書き出してから «豆腐» に気付くと手戻りが大きいので、先に分かるようにします。
        """
        source = "" if text is None else str(text)
        try:
            primary = self.resolve(family, bold, italic)
        except Exception:
            # そもそも 1 つも読めないなら «全部欠字» です。
            seen: list[str] = []
            for character in source:
                if character not in seen:
                    seen.append(character)
            return seen
        missing: list[str] = []
        for character in source:
            code = ord(character)
            # 改行・空白は字形が無くて当たり前なので数えません。
            if code < 0x21:
                continue
            font = self.font_for_code_point(primary, code)
            if font is None or not font.has_glyph(code):
                if character not in missing:
                    missing.append(character)
        return missing

    def describe(self) -> dict[str, Any]:
        """``movo doctor`` 用の情報。"""
        files = self._font_files()
        try:
            default_name = self.default_font().full_name
        except Exception:
            default_name = None
        return {
            "font_file_count": len(files),
            "default_font": default_name,
            "fallbacks": [f.full_name for f in self.fallback_chain()],
            # JS 版の JSON 出力と突き合わせられるよう、camelCase の別名も残します。
            "fontFileCount": len(files),
            "defaultFont": default_name,
        }


__all__ = [
    "FONT_EXTENSIONS",
    "CJK_RANGES",
    "GENERIC_FAMILIES",
    "PREFERRED_CJK",
    "PREFERRED_LATIN",
    "WOFF2_KNOWN_TAGS",
    "Font",
    "FontManager",
    "Glyph",
    "Reader",
    "is_cjk",
    "list_font_files",
    "to_sfnt",
]
