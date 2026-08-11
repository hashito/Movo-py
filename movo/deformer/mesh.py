"""変形用のメッシュ（JS 版 packages/deformer/src/mesh.js の移植）。

レイヤーの中身を «局所画素座標の格子» にします。頂点は自分がどこから
来たか（`u0`/`v0`、0〜1）と、どのテクセルを読むか（`u`/`v`、元画像の画素）を
覚えています。変形は `x`/`y` を動かすだけで、描画側が三角形として貼ります。

**幾何的な変形をすべて 1 枚のメッシュに載せる**ので、曲げ・ねじり・波・
自由変形・パス沿わせ・柔体物理が、**書いた順のまま**自然に重なります。

## Python 版で変えたところ

`Float64Array` を 7 本並べていたところを NumPy の 1 次元配列にしています。
これで変形が «頂点ごとの for» ではなく **配列 1 本の演算**になります。

| `twist` を 1 回 | 頂点ごとの Python | NumPy の一括演算 |
| --- | --- | --- |
| 777 頂点（21x37） | 0.262 ms | **0.031 ms**（8 倍） |
| 6,588 頂点（61x108） | 1.986 ms | **0.119 ms**（17 倍） |

メッシュは 1 レイヤーに 10 個前後の変形が乗り、頂点を増やすほど差が開きます。
"""

from __future__ import annotations

import numpy as np

from ._compat import js_round, warn


class Mesh:
    """変形用の格子。

    :param columns: 横の分割数
    :param rows: 縦の分割数
    :param width: 中身の幅（画素）
    :param height: 中身の高さ（画素）
    :param source_width: テクスチャの幅（画素）
    :param source_height: テクスチャの高さ（画素）
    """

    __slots__ = ("columns", "rows", "width", "height", "source_width", "source_height",
                 "x", "y", "u", "v", "u0", "v0", "alpha")

    def __init__(self, columns, rows, width, height, source_width, source_height) -> None:
        self.columns = max(1, int(columns))
        self.rows = max(1, int(rows))
        self.width = float(width)
        self.height = float(height)
        self.source_width = float(source_width)
        self.source_height = float(source_height)

        # 頂点の並びは «行優先»（row * (columns+1) + column）。JS 版と同じです。
        nu = np.arange(self.columns + 1, dtype=np.float64) / self.columns
        nv = np.arange(self.rows + 1, dtype=np.float64) / self.rows
        grid_u, grid_v = np.meshgrid(nu, nv)
        self.u0 = grid_u.ravel()
        self.v0 = grid_v.ravel()
        self.x = self.u0 * self.width
        self.y = self.v0 * self.height
        self.u = self.u0 * self.source_width
        self.v = self.v0 * self.source_height
        self.alpha = np.ones(self.u0.size, np.float64)

    @staticmethod
    def grid(width, height, resolution, source_width=None, source_height=None) -> "Mesh":
        """縦横比に合わせて «横長なら横を細かく» 割った格子を作る。"""
        if source_width is None:
            source_width = width
        if source_height is None:
            source_height = height
        aspect = width / max(1, height)
        columns = max(1, js_round(resolution * min(4, max(0.25, aspect))))
        rows = max(1, js_round(resolution))
        return Mesh(columns, rows, width, height, source_width, source_height)

    @property
    def vertex_count(self) -> int:
        return int(self.x.size)

    def index(self, column: int, row: int) -> int:
        return row * (self.columns + 1) + column

    def clone(self) -> "Mesh":
        copy = Mesh(self.columns, self.rows, self.width, self.height, self.source_width, self.source_height)
        copy.x = self.x.copy()
        copy.y = self.y.copy()
        copy.u = self.u.copy()
        copy.v = self.v.copy()
        copy.u0 = self.u0.copy()
        copy.v0 = self.v0.copy()
        copy.alpha = self.alpha.copy()
        return copy

    def bounds(self) -> dict:
        """変形後の外接矩形（局所座標）。"""
        min_x = float(self.x.min())
        max_x = float(self.x.max())
        min_y = float(self.y.min())
        max_y = float(self.y.max())
        return {
            "minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y,
            "width": max_x - min_x, "height": max_y - min_y,
        }

    def is_identity(self) -> bool:
        """1 つの頂点も «元の位置» から動いていないか。

        動いていなければメッシュを焼かずに済むので、ここが早く返ることは
        そのまま速度に効きます（`np.any` は最初の 1 つで止まりません
        が、比較 1 本ぶんなので誤差の範囲です）。
        """
        if np.any(np.abs(self.x - self.u0 * self.width) > 1e-6):
            return False
        return not np.any(np.abs(self.y - self.v0 * self.height) > 1e-6)

    def draw(self, dst, src, matrix, options: dict | None = None):
        """`dst` へメッシュを貼る。局所座標は `matrix` で写します。

        三角形の塗りは **`movo.renderer` の Numba カーネル**が受け持ちます
        （README の「多角形の塗りは Numba が 103 倍」の通り）。renderer が
        まだ入っていない環境では、その旨を言って何もしません。

        ⚠ **`options` は «JSON と同じ camelCase の辞書»、`draw_textured_triangle`
        は «キーワード引数»** です。移植が並列に進んだせいで綴りが分かれました。
        JS 版は options オブジェクトを素通ししていたので、ここで «そのまま渡す»
        と書かれていて、`alpha` の位置に辞書が入り `TypeError` になっていました。
        画素一致を確かめてある `raster.py` は動かさず、**呼ぶ側（ここ）で
        ばらして渡します。**
        """
        options = options or {}
        try:
            from movo.renderer.raster import draw_textured_triangle  # type: ignore
        except Exception:
            warn("メッシュの描画には movo.renderer が要ります（まだ移植されていません）")
            return dst

        blend = options.get("blend", "normal") or "normal"
        clamp_edge = bool(options.get("clampEdge", options.get("clamp_edge", False)))
        tint = options.get("tint")
        depth = options.get("depth")

        a, b, c, d, e, f = matrix
        px = a * self.x + c * self.y + e
        py = b * self.x + d * self.y + f
        alpha = float(options.get("alpha", 1.0))

        def quad(i00, i10, i01, i11) -> None:
            quad_alpha = alpha * (
                (self.alpha[i00] + self.alpha[i10] + self.alpha[i01] + self.alpha[i11]) / 4
            )
            if quad_alpha <= 0.002:
                return
            v00 = (px[i00], py[i00], self.u[i00], self.v[i00])
            v10 = (px[i10], py[i10], self.u[i10], self.v[i10])
            v01 = (px[i01], py[i01], self.u[i01], self.v[i01])
            v11 = (px[i11], py[i11], self.u[i11], self.v[i11])
            draw_textured_triangle(
                dst, src, v00, v10, v11, alpha=quad_alpha, blend=blend,
                clamp_edge=clamp_edge, tint=tint, depth=depth,
            )
            draw_textured_triangle(
                dst, src, v00, v11, v01, alpha=quad_alpha, blend=blend,
                clamp_edge=clamp_edge, tint=tint, depth=depth,
            )

        # **変形していない格子は 4 隅だけで描きます。**
        # 変換は全体で 1 つのアフィン行列なので、20x20 に割っても «大きな 1 枚» と
        # 同じ絵になります。割ったままだと 1 レイヤーあたり 800 回カーネルを呼ぶ
        # ことになり、そこがレンダリング時間の大半を占めていました。頂点ごとの
        # アルファが一様でないとき（deformer が付けたフェード）は割ったままにします。
        if self.is_identity() and float(self.alpha.min()) == float(self.alpha.max()):
            quad(
                self.index(0, 0),
                self.index(self.columns, 0),
                self.index(0, self.rows),
                self.index(self.columns, self.rows),
            )
            return dst

        for row in range(self.rows):
            for column in range(self.columns):
                quad(
                    self.index(column, row),
                    self.index(column + 1, row),
                    self.index(column, row + 1),
                    self.index(column + 1, row + 1),
                )
        return dst


def bake_mesh(mesh: Mesh, src, bitmap_class, scale: float = 1.0) -> dict:
    """変形したメッシュを «平らな画像» に焼き戻す。

    画素のエフェクトが幾何変形の後ろに来たときに要ります。エフェクトは画素を
    見るので、先にメッシュを解決しておかないといけません。戻り値には
    «元の箱と位置を合わせるためのずれ» も入れてあります。
    """
    bounds = mesh.bounds()
    pad = 1
    offset_x = int(np.floor(bounds["minX"])) - pad
    offset_y = int(np.floor(bounds["minY"])) - pad
    logical_width = max(1, int(np.ceil(bounds["maxX"])) - offset_x + pad)
    logical_height = max(1, int(np.ceil(bounds["maxY"])) - offset_y + pad)
    out = bitmap_class(max(1, js_round(logical_width * scale)), max(1, js_round(logical_height * scale)))
    mesh.draw(out, src, [scale, 0, 0, scale, -offset_x * scale, -offset_y * scale], {"clampEdge": False})
    return {
        "bitmap": out,
        "offsetX": offset_x,
        "offsetY": offset_y,
        "width": logical_width,
        "height": logical_height,
    }
