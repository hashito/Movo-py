"""画像の復号の入口。

PNG・ベースライン JPEG・BMP は **自前で**読みます。それ以外（WebP・TIFF・
プログレッシブ JPEG など）は ffmpeg があれば通し、無ければはっきりした
``MOVO_ASSET_DECODE_FAILED`` を出します。

**Pillow を使わないのは依存を増やさない方針**だからです。単体 EXE に固めて
配るので、同梱するライブラリは «再配布できるか» を 1 つずつ確かめています。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .bitmap import Bitmap
from .errors import ErrorCodes, MovoError
from .jpeg import decode_jpeg, is_jpeg
from .platform import find_ffmpeg
from .png import decode_png, encode_png, is_png


def is_bmp(buffer: bytes) -> bool:
    return len(buffer) > 2 and buffer[0] == 0x42 and buffer[1] == 0x4D


def decode_bmp(buffer: bytes) -> Bitmap:
    """無圧縮の 24/32 ビット BMP を読む（書き出し経由でよく来る形式です）。

    行ごとの詰め物（4 バイト境界）を外したあとは **NumPy のスライスだけ**で
    BGR → RGB に並べ替えます。画素ごとのループは要りません。
    """
    data = bytes(buffer)
    data_offset = int.from_bytes(data[10:14], "little")
    header_size = int.from_bytes(data[14:18], "little")
    width = int.from_bytes(data[18:22], "little", signed=True)
    height_raw = int.from_bytes(data[22:26], "little", signed=True)
    height = abs(height_raw)
    bpp = int.from_bytes(data[28:30], "little")
    compression = int.from_bytes(data[30:34], "little") if header_size >= 40 else 0

    if compression not in (0, 3):
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "圧縮された BMP には対応していません")
    if bpp not in (24, 32):
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, f"対応していない BMP のビット深度 {bpp}")

    stride = ((bpp * width + 31) // 32) * 4
    needed = data_offset + stride * height
    if len(data) < needed:
        raise MovoError(ErrorCodes.MOVO_ASSET_DECODE_FAILED, "BMP のデータが足りません")

    rows = np.frombuffer(data, np.uint8, count=stride * height, offset=data_offset).reshape(height, stride)
    per_pixel = bpp // 8
    pixels = rows[:, : width * per_pixel].reshape(height, width, per_pixel)
    # 正の高さは «下から上» に並んでいます（BMP の伝統）。
    if height_raw > 0:
        pixels = pixels[::-1]

    bitmap = Bitmap(width, height)
    bitmap.data[..., 0] = pixels[..., 2]  # BGR → RGB
    bitmap.data[..., 1] = pixels[..., 1]
    bitmap.data[..., 2] = pixels[..., 0]
    bitmap.data[..., 3] = pixels[..., 3] if per_pixel == 4 else 255
    return bitmap


def decode_image(buffer: bytes, *, source_path: str | None = None, allow_ffmpeg: bool = True) -> Bitmap:
    """バイト列を :class:`Bitmap` にする。形式は **中身を見て**決めます。

    拡張子で判断しないのは、``.png`` という名前の JPEG が普通に流通しているためです。
    """
    if is_png(buffer):
        return decode_png(buffer)
    if is_jpeg(buffer):
        try:
            return decode_jpeg(buffer)
        except MovoError:
            if not allow_ffmpeg:
                raise
            converted = _convert_with_ffmpeg(buffer, source_path)
            if converted is not None:
                return converted
            raise
    if is_bmp(buffer):
        return decode_bmp(buffer)
    if allow_ffmpeg:
        converted = _convert_with_ffmpeg(buffer, source_path)
        if converted is not None:
            return converted
    where = f"（{source_path}）" if source_path else ""
    raise MovoError(
        ErrorCodes.MOVO_ASSET_DECODE_FAILED,
        f"画像の形式が分かりません{where}",
        hint="ffmpeg なしで読めるのは PNG / ベースライン JPEG / BMP です。ほかは ffmpeg を入れてください",
    )


def _convert_with_ffmpeg(buffer: bytes, source_path: str | None) -> Bitmap | None:
    """ffmpeg で PNG に変換してから読む。**失敗しても例外にしません**（呼ぶ側が判断します）。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    directory = tempfile.mkdtemp(prefix="movo-img-")
    try:
        if source_path and os.path.exists(source_path):
            source = source_path
        else:
            source = os.path.join(directory, "input.bin")
            with open(source, "wb") as handle:
                handle.write(bytes(buffer))
        output = os.path.join(directory, "out.png")
        result = subprocess.run(
            [ffmpeg["path"], "-y", "-loglevel", "error", "-i", source, output],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(output):
            with open(output, "rb") as handle:
                return decode_png(handle.read())
        return None
    except (OSError, subprocess.SubprocessError, MovoError):
        return None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def load_image(file_path: str | os.PathLike[str]) -> Bitmap:
    """ディスクから読んで復号する。"""
    try:
        with open(file_path, "rb") as handle:
            buffer = handle.read()
    except OSError as error:
        raise MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND, f"画像ファイルを読めません: {file_path}", cause=error
        ) from error
    return decode_image(buffer, source_path=str(file_path))


def save_image(bitmap: Bitmap, file_path: str | os.PathLike[str]) -> str:
    """PNG として書き出す。親ディレクトリが無ければ作ります。"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(bitmap))
    return str(path)
