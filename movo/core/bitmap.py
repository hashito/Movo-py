"""画素の入れ物。**ここの作法が全体の速度を決めます。**

## 持ち方

`(高さ, 幅, 4)` の `uint8`、RGBA、**アルファは乗算済みではありません**（JS 版と同じ）。

形を 3 次元にしているのは、`bitmap.data[..., :3]` で色だけ、`[..., 3]` でアルファだけを
一息で触れるからです。1 次元で持っても速度は変わりませんでした（メモリ帯域が
支配的なので）が、書くときの読みやすさが違います。

## 書くときの決まり

**画素ごとの `for` を Python で書かないでください。** 1280x720 の 1 パスで
720 ミリ秒かかります（NumPy なら 13 ミリ秒、Numba なら 0.3 ミリ秒）。

| やりたいこと | 使うもの |
| --- | --- |
| 全画面に一様な処理（明度・色調・合成） | NumPy の一括演算 |
| 画素ごとに分岐する処理（塗り・文字・変形） | `movo.renderer.kernels` の Numba 関数 |

## 飽和に注意

`uint8` のまま掛け算すると 255 を超えた時点で巻き戻ります（`200 * 2 = 144`）。
**必ず一度 `uint16` か `float32` に上げてから戻してください。** `blend_over` などの
共通処理はそれをやっているので、まずはそちらを使ってください。
"""

from __future__ import annotations

import numpy as np

# 画素の型。ここを float32 にすると精度は上がりますが、メモリが 4 倍になり
# 帯域が支配的な処理では遅くなります。実測して uint8 のままにしています。
DTYPE = np.uint8


class Bitmap:
    """RGBA の画素バッファ。

    :param width: 幅（画素）
    :param height: 高さ（画素）
    :param data: 既存の配列を包むとき。`(h, w, 4)` の uint8
    """

    __slots__ = ("width", "height", "data")

    def __init__(self, width: int, height: int, data: np.ndarray | None = None) -> None:
        self.width = int(width)
        self.height = int(height)
        if data is None:
            # zeros は «透明な黒» です。JS 版の new Bitmap と同じ初期状態。
            self.data = np.zeros((self.height, self.width, 4), DTYPE)
        else:
            if data.shape != (self.height, self.width, 4):
                raise ValueError(f"配列の形が合いません: {data.shape} ではなく {(self.height, self.width, 4)} が要ります")
            self.data = data

    # ── 生成 ────────────────────────────────────────────────

    @classmethod
    def create(cls, width: int, height: int, fill: object | None = None) -> "Bitmap":
        """色を指定して作る。JS 版の ``Bitmap.create()`` と同じ入口です。

        色は ``"#ff0000"`` でも ``{"r":255,...}`` でも受けます（:mod:`movo.core.color`）。
        """
        bmp = cls(width, height)
        if fill is not None:
            from .color import color_to_rgba8, parse_color

            bmp.fill(color_to_rgba8(parse_color(fill)))
        return bmp

    # ── 基本 ────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """1 画素も不透明でないか。**描く前の早期打ち切りに使います。**

        `any()` は最初の非ゼロで止まるので、空のバッファでは速く返ります。
        """
        return not self.data[..., 3].any()

    def copy(self) -> "Bitmap":
        return Bitmap(self.width, self.height, self.data.copy())

    #: JS 版の名前。移植したコードをそのまま読めるように別名を置いています。
    clone = copy

    # ── 画素の出し入れ ──────────────────────────────────────
    #
    # **ここを画素ごとのループから呼ばないでください。** 1 回の呼び出しが
    # 数マイクロ秒なので、1280x720 で 3 秒以上かかります。1 画素だけ見たいとき
    # （テスト、当たり判定）のための入口です。

    def get_pixel(self, x: int, y: int) -> dict[str, float]:
        """1 画素を ``{"r","g","b","a"}`` で返す。範囲外は透明な黒。"""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return {"r": 0, "g": 0, "b": 0, "a": 0.0}
        px = self.data[y, x]
        return {"r": int(px[0]), "g": int(px[1]), "b": int(px[2]), "a": int(px[3]) / 255}

    def set_pixel(self, x: int, y: int, r: int, g: int, b: int, a: int) -> None:
        """1 画素を書く。範囲外は黙って捨てます（呼ぶ側で確かめずに済むように）。"""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        self.data[y, x] = (r, g, b, a)

    # ── 切り出し・拡縮 ──────────────────────────────────────

    def crop(self, x: int, y: int, w: int, h: int) -> "Bitmap":
        """矩形を切り出す（画面の外にはみ出した分は刈ります）。"""
        x0 = int(np.clip(np.floor(x), 0, self.width))
        y0 = int(np.clip(np.floor(y), 0, self.height))
        x1 = int(np.clip(np.floor(x + w), 0, self.width))
        y1 = int(np.clip(np.floor(y + h), 0, self.height))
        out = Bitmap(max(0, x1 - x0), max(0, y1 - y0))
        if out.width and out.height:
            out.data[...] = self.data[y0:y1, x0:x1]
        return out

    def resize(self, width: int, height: int) -> "Bitmap":
        """箱フィルタで拡縮する（スーパーサンプリングの縮小と縮小画像に使います）。

        **アルファで重みを付けてから色を平均します。** そうしないと、透明な
        画素の «見えない色» が混ざって縁が黒ずみます。JS 版と同じ計算です。

        整数倍の縮小は NumPy の一括平均で済ませ、それ以外だけ行ごとに畳みます。
        画素ごとの Python ループは避けています。
        """
        w = max(1, int(np.floor(width + 0.5)))
        h = max(1, int(np.floor(height + 0.5)))
        if w == self.width and h == self.height:
            return self.copy()

        src = self.data.astype(np.float64)
        alpha = src[..., 3]
        # 色はアルファで重み付けしてから足す（乗算済みにしてから平均する）
        weighted = np.empty((self.height, self.width, 4), np.float64)
        weighted[..., :3] = src[..., :3] * alpha[..., None]
        weighted[..., 3] = alpha

        sx = self.width / w
        sy = self.height / h
        # 出力の各行・各列が読む «元の範囲» を先に作っておく（JS 版と同じ丸め）
        yb = [(int(np.floor(j * sy)), max(int(np.floor(j * sy)) + 1, min(self.height, int(np.ceil((j + 1) * sy))))) for j in range(h)]
        xb = [(int(np.floor(i * sx)), max(int(np.floor(i * sx)) + 1, min(self.width, int(np.ceil((i + 1) * sx))))) for i in range(w)]

        # 累積和にしておくと «任意の矩形の合計» が引き算 1 回で出ます。
        # 出力画素ごとに元画素を舐めると O(元の画素数) の Python ループになります。
        cum = np.zeros((self.height + 1, self.width + 1, 4), np.float64)
        np.cumsum(np.cumsum(weighted, axis=0), axis=1, out=cum[1:, 1:])

        y0 = np.array([b[0] for b in yb])
        y1 = np.array([b[1] for b in yb])
        x0 = np.array([b[0] for b in xb])
        x1 = np.array([b[1] for b in xb])
        total = (
            cum[np.ix_(y1, x1)] - cum[np.ix_(y0, x1)] - cum[np.ix_(y1, x0)] + cum[np.ix_(y0, x0)]
        )
        count = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float64)

        out = Bitmap(w, h)
        acc_a = total[..., 3]
        safe = np.where(acc_a > 0, acc_a, 1.0)
        rgb = total[..., :3] / safe[..., None]
        out_a = acc_a / count
        out.data[..., :3] = np.where(acc_a[..., None] > 0, np.clip(np.rint(rgb), 0, 255), 0).astype(DTYPE)
        out.data[..., 3] = np.where(acc_a > 0, np.clip(np.rint(out_a), 0, 255), 0).astype(DTYPE)
        return out

    # ── 調べる ──────────────────────────────────────────────

    def alpha_bounds(self, threshold: int = 0) -> dict[str, int] | None:
        """不透明な画素を囲む矩形。1 つも無ければ None。

        **描く前に «そもそも見えるか» を確かめる**のに使います。空のレイヤーに
        エフェクトを何枚も掛けるのは丸損なので、ここで打ち切ります。
        """
        mask = self.data[..., 3] > threshold
        if not mask.any():
            return None
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        return {
            "x": int(cols[0]),
            "y": int(rows[0]),
            "width": int(cols[-1] - cols[0] + 1),
            "height": int(rows[-1] - rows[0] + 1),
        }

    def flatten(self, background: object = "#000000") -> "Bitmap":
        """不透明な背景色の上に落として、アルファを 255 にする。

        書き出しの直前に通します。mp4 にはアルファが無いので、透明なまま
        渡すと «黒 v.s. 白» が実装まかせになるためです。
        """
        from .color import color_to_rgba8, parse_color

        bg = color_to_rgba8(parse_color(background))
        a = self.data[..., 3].astype(np.float32)[..., None] / 255.0
        src = self.data[..., :3].astype(np.float32)
        bg_arr = np.array(bg[:3], np.float32)
        out = Bitmap(self.width, self.height)
        out.data[..., :3] = np.clip(np.rint(src * a + bg_arr * (1.0 - a)), 0, 255).astype(DTYPE)
        out.data[..., 3] = 255
        return out

    def fill(self, rgba: tuple[int, int, int, int]) -> "Bitmap":
        """一色で塗りつぶす。"""
        self.data[...] = rgba
        return self

    def clear(self) -> "Bitmap":
        """透明に戻す。`data[:] = 0` は新しい配列を作らないので速いです。"""
        self.data[...] = 0
        return self

    # ── 合成 ────────────────────────────────────────────────

    def draw(self, src: "Bitmap", x: int = 0, y: int = 0, alpha: float = 1.0) -> "Bitmap":
        """別のバッファを重ねる（source-over）。

        **はみ出しはここで刈ります。** 呼ぶ側で座標を確かめずに済むよう、
        画面の外はそのまま捨てます。
        """
        if alpha <= 0:
            return self
        sx0 = max(0, -x)
        sy0 = max(0, -y)
        sx1 = min(src.width, self.width - x)
        sy1 = min(src.height, self.height - y)
        if sx1 <= sx0 or sy1 <= sy0:
            return self

        dx0 = x + sx0
        dy0 = y + sy0
        dst = self.data[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)]
        cut = src.data[sy0:sy1, sx0:sx1]
        blend_over(dst, cut, alpha)
        return self


def blend_over(dst: np.ndarray, src: np.ndarray, alpha: float = 1.0) -> None:
    """`dst` の上に `src` を source-over で重ねる（その場で書き換えます）。

    アルファ合成の式そのままです。**`float32` に上げてから戻します。**
    `uint8` のまま掛けると 255 を超えたところで巻き戻ります。

        outA = sa + da * (1 - sa)
        outC = (sc * sa + dc * da * (1 - sa)) / outA
    """
    sa = src[..., 3].astype(np.float32) * (alpha / 255.0)
    if not sa.any():
        return
    da = dst[..., 3].astype(np.float32) / 255.0
    out_a = sa + da * (1.0 - sa)

    # 出力が透明なところは色を求めても意味がないので 1 で割って捨てます
    # （0 除算の警告を出さないため）。最後にアルファで潰れます。
    safe = np.where(out_a > 0, out_a, 1.0)
    sc = src[..., :3].astype(np.float32)
    dc = dst[..., :3].astype(np.float32)
    mixed = (sc * sa[..., None] + dc * da[..., None] * (1.0 - sa[..., None])) / safe[..., None]

    # **`astype` は切り捨てです。** JS の `Uint8ClampedArray` は «最も近い整数へ»
    # 丸めるので、そのまま真似ると全面で 1 ずれます（実際に textBox の合成で
    # 最大差 1 が出ました）。`rint` を挟んで JS と同じ絵にします。
    dst[..., :3] = np.clip(np.rint(mixed), 0, 255).astype(DTYPE)
    dst[..., 3] = np.clip(np.rint(out_a * 255.0), 0, 255).astype(DTYPE)


def to_float(data: np.ndarray) -> np.ndarray:
    """0..1 の float32 にする。エフェクトを書くときの入口。"""
    return data.astype(np.float32) / 255.0


def to_u8(data: np.ndarray) -> np.ndarray:
    """0..1 の float32 を uint8 に戻す。**必ず clip してから**丸めます。"""
    return np.clip(data * 255.0 + 0.5, 0, 255).astype(DTYPE)
