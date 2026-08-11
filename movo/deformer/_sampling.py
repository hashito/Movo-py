"""画像の «間» を読む（双一次補間）。JS 版 `Bitmap.sampleBilinear` の配列版。

`movo.core.bitmap.Bitmap` に同じ名前のメソッドが生えたら、そちらを使うよう
`sample_bilinear` を差し替えてください。ここに置いてあるのは、変形とマスクが
core の移植を待たずに動くようにするためです。

**半透明の縁で色が濁らないよう、アルファを掛けてから混ぜて最後に割ります。**
これをしないと «透明な黒» が色として混ざり、輪郭が暗くなります。
"""

from __future__ import annotations

import numpy as np


def sample_bilinear(bitmap, x, y, clamp_edge: bool = True) -> np.ndarray:
    """`(..., 4)` の float64 で色を返す。`x`/`y` は元画像の画素座標。"""
    w = bitmap.width
    h = bitmap.height
    data = bitmap.data
    px = np.asarray(x, np.float64) - 0.5
    py = np.asarray(y, np.float64) - 0.5
    x0 = np.floor(px)
    y0 = np.floor(py)
    fx = px - x0
    fy = py - y0
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)

    shape = np.broadcast(px, py).shape
    acc = np.zeros(shape + (3,), np.float64)
    alpha = np.zeros(shape, np.float64)

    for dy in (0, 1):
        for dx in (0, 1):
            wx = fx if dx else 1 - fx
            wy = fy if dy else 1 - fy
            weight = wx * wy
            sx = x0 + dx
            sy = y0 + dy
            inside = (sx >= 0) & (sy >= 0) & (sx < w) & (sy < h)
            if clamp_edge:
                cx = np.clip(sx, 0, w - 1)
                cy = np.clip(sy, 0, h - 1)
                valid = weight != 0
            else:
                cx = np.clip(sx, 0, w - 1)
                cy = np.clip(sy, 0, h - 1)
                valid = inside & (weight != 0)
            texel = data[cy, cx].astype(np.float64)
            pa = texel[..., 3] * np.where(valid, 1.0, 0.0) * weight
            acc += texel[..., :3] * pa[..., None]
            alpha += pa

    out = np.zeros(shape + (4,), np.float64)
    lit = alpha > 0.0001
    safe = np.where(lit, alpha, 1.0)
    out[..., :3] = np.where(lit[..., None], acc / safe[..., None], 0.0)
    out[..., 3] = np.where(lit, alpha, 0.0)
    return out


def channel_value(sample: np.ndarray, channel: str) -> np.ndarray:
    """チャンネル 1 本を取り出す。既定は輝度（BT.601）。"""
    if channel == "red":
        return sample[..., 0]
    if channel == "green":
        return sample[..., 1]
    if channel == "blue":
        return sample[..., 2]
    if channel == "alpha":
        return sample[..., 3]
    return 0.299 * sample[..., 0] + 0.587 * sample[..., 1] + 0.114 * sample[..., 2]
