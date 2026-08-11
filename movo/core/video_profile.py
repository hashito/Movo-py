"""映像を «数値» にする。

ボカロ MV のスタイルを真似て 10 本作ったとき、似ているかどうかを毎回目で見て
判断していました。「なんとなく足りない」を言語化するのに時間がかかり、しかも
見落とします（``04-pixel-retro`` の «色の境目がベタ» は 3 回見返して気付いた）。

ここでは映像を «型» として測ります。カット尺・動きの量・色数・文字の占有率と
いった、**スタイルを決めている数字**だけを見ます。画質の良し悪しではなく、
«その映像がどういう作りか» を出すのが目的です。

目標値と突き合わせるのは :mod:`movo.core.video_compare` の仕事です。ここは測るだけ。

## 速度

書き出しの後段で **全フレーム**を通します。1280x720 の 1 フレームで
**6.6 ミリ秒**（3 分の MV で 36 秒）。画素ごとの Python ループにすると
1 フレーム 240 ミリ秒、3 分で 22 分かかります。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from .bitmap import Bitmap

#: sRGB の 0..255 を線形の 0..1 へ。相対輝度の計算に使う（JS 版と同じ float32）。
_LINEAR = np.array(
    [(c / 255 / 12.92) if (c / 255) <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4 for c in range(256)],
    np.float32,
)


def _fixed(value: float, digits: int) -> float:
    """JS の ``Number(x.toFixed(n))`` と同じ丸め。

    Python の組み込み ``round`` は «偶数丸め» なので、``round(0.125, 2)`` が
    0.12 になります。JS は 0.13 です。**プロファイルの数値は JS 版と
    突き合わせる**ので、ここがずれると «同じ映像なのに違う型» に見えます。

    2 進の実際の値から丸めるのが要点です（``1.005`` は本当は
    1.00499999... なので、JS も Python もここでは切り下げになります）。
    """
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(float(value)).quantize(quantum, rounding=ROUND_HALF_UP))


class _FrameFeatures:
    """1 フレームぶんの特徴。**フレームは溜めません。**

    3 分の MV は 5,400 フレームあり、1280x720 の RGBA を全部持つと 20 GB です。
    受け取るたびに «格子の輝度 + 色のヒストグラム + 数個の平均» まで小さくして捨てます。
    """

    __slots__ = ("luma", "cols", "rows", "flat_keys", "palette_counts", "saturation", "brightness", "edge")

    def __init__(self, luma, cols, rows, flat_keys, palette_counts, saturation, brightness, edge):
        self.luma = luma
        self.cols = cols
        self.rows = rows
        self.flat_keys = flat_keys
        self.palette_counts = palette_counts
        self.saturation = saturation
        self.brightness = brightness
        self.edge = edge


def _frame_features(bitmap: Bitmap, cells: int = 24) -> _FrameFeatures:
    width = bitmap.width
    height = bitmap.height
    cols = cells
    rows = max(3, int(np.floor(cells * height / width + 0.5)))

    step_x = max(1, width // 240)
    step_y = max(1, height // 240)
    xs = np.arange(0, width, step_x)
    ys = np.arange(0, height, step_y)
    sub = bitmap.data[np.ix_(ys, xs)]
    r = sub[..., 0].astype(np.int32)
    g = sub[..., 1].astype(np.int32)
    b = sub[..., 2].astype(np.int32)

    col_idx = np.minimum(cols - 1, (xs / width * cols).astype(np.int64))
    row_idx = np.minimum(rows - 1, (ys / height * rows).astype(np.int64))
    cell = (row_idx[:, None] * cols + col_idx[None, :]).ravel()

    lum = (
        0.2126 * _LINEAR[r].astype(np.float64)
        + 0.7152 * _LINEAR[g].astype(np.float64)
        + 0.0722 * _LINEAR[b].astype(np.float64)
    )
    counts = np.bincount(cell, minlength=cols * rows).astype(np.float64)
    counts[counts == 0] = 1.0
    luma = np.bincount(cell, weights=lum.ravel(), minlength=cols * rows) / counts

    # 色は 4bit × 3 に量子化して数える。厳密な色数ではなく «見た目の色数»。
    keys = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
    flat_keys = keys.ravel()
    palette_counts = np.bincount(flat_keys, minlength=4096)

    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    saturation = np.where(high == 0, 0.0, (high - low) / np.where(high == 0, 1, high))
    brightness = high / 255.0
    samples = max(1, r.size)

    # 右隣（ステップぶん離れた画素）との差。文字や細かい模様が多いほど大きくなる。
    # ``xs`` は等間隔なので、``x + step_x`` はそのまま «次の要素» です。
    # 最後の列だけは右隣が画面の外なので数えません（JS 版の `x + stepX < width`）。
    if len(xs) >= 2:
        left = sub[:, :-1, :3].astype(np.int32)
        right = sub[:, 1:, :3].astype(np.int32)
        edge_sum = float(np.sum(np.abs(right - left).sum(axis=2) / 765.0))
    else:
        edge_sum = 0.0

    return _FrameFeatures(
        luma=luma,
        cols=cols,
        rows=rows,
        flat_keys=flat_keys,
        palette_counts=palette_counts,
        saturation=float(np.sum(saturation)) / samples,
        brightness=float(np.sum(brightness)) / samples,
        edge=edge_sum / samples,
    )


class VideoProfiler:
    """映像の «型» を測る。フレームは 1 枚ずつ渡します。"""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float = 30,
        cut_threshold: float = 0.18,
        still_threshold: float = 0.004,
    ) -> None:
        self.fps = fps or 30
        self.width = width
        self.height = height
        # カットとみなす «フレーム間の変わり方»。0.18 は経験値で、ディゾルブは
        # 拾わずハードカットだけ拾う程度。
        self.cut_threshold = cut_threshold
        # 止まっているとみなす差。ノイズやグレインで 0 にはならないので少し上。
        self.still_threshold = still_threshold

        self._previous: np.ndarray | None = None
        # 色のヒストグラムは **配列で溜めます**（4bit×3 なので必ず 4096 個）。
        # 辞書に足していくと 1 フレームあたり最大 4,096 回の Python の演算になり、
        # 1280x720 で 1 フレーム 9.3 ミリ秒 → 3 分の MV で 50 秒かかりました。
        # 配列なら 1 回の加算で済みます（1 フレーム 1.9 ミリ秒）。
        self._palette_counts = np.zeros(4096, np.int64)
        # 並び順だけは «初めて出てきた順» を保ちます（JS 版の Map と同じにするため）。
        self._palette_seen = np.zeros(4096, bool)
        self._palette_order: list[int] = []
        self._saturation = 0.0
        self._brightness = 0.0
        self._edge = 0.0
        self._luma_sum = 0.0
        self._luma_square = 0.0
        self._diffs: list[float] = []
        self.cut_frames: list[int] = [0]
        self.frame_index = 0

    def push(self, bitmap: Bitmap) -> None:
        features = _frame_features(bitmap)
        self._saturation += features.saturation
        self._brightness += features.brightness
        self._edge += features.edge
        self._palette_counts += features.palette_counts
        # 初出の色だけ «出てきた順» に控えます。2 フレーム目からはたいてい
        # 新しい色が無いので、そのときは並べ替えを一切しません（走査順を
        # 求める np.unique は 1 フレームあたり 1.1 ミリ秒かかります）。
        present = np.flatnonzero(features.palette_counts)
        if not self._palette_seen[present].all():
            _keys, first = np.unique(features.flat_keys, return_index=True)
            for key in features.flat_keys[np.sort(first)]:
                if not self._palette_seen[key]:
                    self._palette_seen[key] = True
                    self._palette_order.append(int(key))
        # 明暗の広がり（コントラスト）は格子の輝度のばらつきで見る
        self._luma_sum += float(features.luma.sum())
        self._luma_square += float((features.luma * features.luma).sum())

        if self._previous is not None:
            diff = float(np.abs(features.luma - self._previous).sum()) / features.luma.size
            self._diffs.append(diff)
            if diff >= self.cut_threshold:
                self.cut_frames.append(self.frame_index)
        self._previous = features.luma
        self.frame_index += 1

    def report(self) -> dict:
        """測った結果。**キー名は JS 版のまま**（``profiles/*.json`` と突き合わせるため）。"""
        frames = max(1, self.frame_index)
        luma_count = max(1, frames * 24 * int(np.floor(24 * self.height / self.width + 0.5)))
        mean = self._luma_sum / luma_count
        variance = max(0.0, self._luma_square / luma_count - mean * mean)

        # カット尺。最後のカットから終わりまでも 1 本として数える。
        cut_lengths: list[float] = []
        for i, frm in enumerate(self.cut_frames):
            to = self.cut_frames[i + 1] if i + 1 < len(self.cut_frames) else frames
            cut_lengths.append((to - frm) / self.fps)
        ordered = sorted(cut_lengths)
        median = ordered[len(ordered) // 2] if ordered else 0.0

        motion = sum(self._diffs) / len(self._diffs) if self._diffs else 0.0
        peak = max(self._diffs) if self._diffs else 0.0
        still = (
            len([d for d in self._diffs if d < self.still_threshold]) / max(1, len(self._diffs))
            if self._diffs
            else 0.0
        )

        # 支配色。上位を «その映像の色» とみなす。量子化した鍵から色に戻す。
        total = int(self._palette_counts.sum()) or 1
        # 同数のときは **«先に出てきた色» が勝ちます**（JS 版の Map と同じ）。
        # `sorted` は安定なので、初出順に並べておけば同数の順序が保たれます。
        ranked = sorted(
            ((key, int(self._palette_counts[key])) for key in self._palette_order),
            key=lambda kv: -kv[1],
        )
        dominant = [
            {
                "color": "#" + "".join(f"{((v << 4) | v):02x}" for v in ((key >> 8) & 0xF, (key >> 4) & 0xF, key & 0xF)),
                "ratio": count / total,
            }
            for key, count in ranked[:6]
        ]
        # «実質の色数» は、全体の 90% を占めるのに何色要るか。
        cumulative = 0.0
        effective = 0
        for _key, count in ranked:
            cumulative += count / total
            effective += 1
            if cumulative >= 0.9:
                break

        return {
            "frames": frames,
            "seconds": frames / self.fps,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "cuts": {
                "count": len(cut_lengths),
                "medianSeconds": _fixed(median, 3),
                "perMinute": _fixed((len(cut_lengths) / (frames / self.fps)) * 60, 2),
                "lengths": [_fixed(v, 2) for v in cut_lengths],
            },
            "motion": {
                "mean": _fixed(motion, 4),
                "peak": _fixed(peak, 4),
                "stillRatio": _fixed(still, 3),
            },
            "palette": {
                "dominant": [d["color"] for d in dominant],
                "dominantRatio": [_fixed(d["ratio"], 3) for d in dominant],
                "effectiveColors": effective,
                "saturation": _fixed(self._saturation / frames, 3),
                "brightness": _fixed(self._brightness / frames, 3),
                "contrast": _fixed(float(np.sqrt(variance)), 3),
            },
            # エッジ密度は «文字や細かい模様がどれだけあるか»。文字主体の映像で高い。
            "detail": {"edgeDensity": _fixed(self._edge / frames, 4)},
        }
