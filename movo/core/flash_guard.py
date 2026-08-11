"""光過敏性発作（PSE）の検査。

MV では拍ごとの白フラッシュ・ストロボ・グリッチを普通に使います。166 BPM で
「拍のアタマに全画面を白く飛ばす」と **毎秒 2.8 回**の閃光になり、これは放送の
ガイドラインが危険とする水準のすぐ手前です。作った本人が気付かないまま公開
できてしまうのは良くないので、書き出しながら実際のフレームを測って警告します。

判定は ITU-R BT.1702 / ARIB / WCAG 2.3.1 の «ハーディング試験» を、画面の画素値
だけで近似したものです。

1. 相対輝度が 10% 以上変化した領域が画面の 25% 以上を占めたら «遷移» とする
   （暗い側が明るすぎるときは数えない。白から白への変化は «閃光» として
   知覚されないため）
2. 向きが逆の遷移が対になったら «閃光» 1 回
3. 1 秒の窓に閃光が 3 回を超えたら危険
4. 彩度の高い赤の反転は、輝度差が小さくても別枠で数える

**実測なので、式で書こうがキーフレームで書こうがプラグインが出そうが
同じように引っ掛かります。**

## 速度

書き出し中に **全フレーム**を通します。ここが遅いとレンダリング全体が遅く
なるので、画素は間引いたうえで NumPy の一括演算にしています。1280x720 の
1 フレームで **3.7 ミリ秒**。画素ごとの Python ループなら 210 ミリ秒かかり、
3 分の MV で 19 分が検査だけに消えます。
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from .bitmap import Bitmap

#: sRGB の 0..255 を線形の 0..1 へ直す表。毎画素 pow() を呼ぶのは重い。
#: **float32 なのは JS 版の Float32Array に合わせるため**です。
_LINEAR = np.array(
    [(c / 255 / 12.92) if (c / 255) <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4 for c in range(256)],
    np.float32,
)

#: ガイドラインの既定値。**キー名は JS 版のまま**（``project.render.flashGuard`` に書く名前）。
FLASH_DEFAULTS = MappingProxyType(
    {
        "maxPerSecond": 3,       # 1 秒あたりに許す閃光の回数
        "luminanceDelta": 0.1,   # 閃光とみなす相対輝度の差
        "areaRatio": 0.25,       # 閃光とみなす面積（画面に対する比）
        "darkerCeiling": 0.8,    # これより明るい同士の変化は数えない
        "redRatio": 0.6,         # 彩度の高い赤とみなす下限（線形 R の比率）
    }
)


class FlashGuard:
    """書き出しながらフレームを測る。

    画面をそのまま比べると重いので、格子（既定 32x18）に落としてから比べます。
    ハーディング試験も «視野 10 度ぶんの面積» を見るので、画素単位の精度は要りません。
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float = 30,
        cells: int | None = None,
        **overrides,
    ) -> None:
        self.fps = fps or 30
        self.width = width
        self.height = height
        # **``None`` は «指定なし» として捨てます。**
        # 単純に辞書を上書きすると、呼び出し側が `maxPerSecond=None` を渡した
        # だけで既定値が打ち消され、閾値が None になります（`project.render.flashGuard`
        # を書いていないプロジェクトでは実際にそうなっていました）。
        #
        # そうなると比較がすべて False になるので、**閃光を 1 つも検出しないのに
        # 毎回«危険»と警告する**という、いちばん困る壊れ方をします。検査が
        # 素通しになっていることに、警告が出ているせいで気付けません。
        self.settings = dict(FLASH_DEFAULTS)
        for key, value in overrides.items():
            if value is not None and key in FLASH_DEFAULTS:
                self.settings[key] = value

        self.cols = max(4, min(64, 32 if cells is None else cells))
        self.rows = max(3, int(np.floor(self.cols * self.height / self.width + 0.5)))
        self.cell_count = self.cols * self.rows

        self._previous_luma: np.ndarray | None = None
        self._previous_red: np.ndarray | None = None
        #: 直前の «遷移» の向き（+1 明るく / -1 暗く）。対になったら閃光 1 回。
        self._pending_direction = 0
        self._pending_red_direction = 0
        self.frame_index = 0
        #: 閃光が起きたフレーム番号
        self.flashes: list[int] = []
        self.red_flashes: list[int] = []

        # 画素を間引いて舐める。格子 1 マスあたり 100 点も見れば十分。
        step_x = max(1, self.width // (self.cols * 10))
        step_y = max(1, self.height // (self.rows * 10))
        self._xs = np.arange(0, self.width, step_x)
        self._ys = np.arange(0, self.height, step_y)
        cols_idx = np.minimum(self.cols - 1, (self._xs / self.width * self.cols).astype(np.int64))
        rows_idx = np.minimum(self.rows - 1, (self._ys / self.height * self.rows).astype(np.int64))
        self._cells = (rows_idx[:, None] * self.cols + cols_idx[None, :]).ravel()
        self._counts = np.bincount(self._cells, minlength=self.cell_count).astype(np.float64)
        self._counts[self._counts == 0] = 1.0

    def push(self, bitmap: Bitmap) -> None:
        """1 フレーム測る。"""
        luma, red = self._grid(bitmap)
        if self._previous_luma is not None:
            self._pending_direction = self._accumulate(
                self._pending_direction, self._compare_luma(self._previous_luma, luma), self.flashes
            )
            self._pending_red_direction = self._accumulate(
                self._pending_red_direction, self._compare_red(self._previous_red, red), self.red_flashes
            )
        self._previous_luma = luma
        self._previous_red = red
        self.frame_index += 1

    def _accumulate(self, pending: int, direction: int, sink: list[int]) -> int:
        """遷移を «対» にまとめて閃光を数える。

        ガイドラインの «閃光» は「向きが逆の遷移の対」1 つです。1 フレームだけ
        白く飛ばす演出は、明→暗の 1 往復なので **1 回**。向きが変わるたびに
        数えると 1 往復を 2 回と数えてしまい、実際の倍の危険度に見えます。
        対が成立したら向きを空に戻し、次の対を新しく数え始めます。
        """
        if direction == 0:
            return pending
        if pending != 0 and direction != pending:
            sink.append(self.frame_index)
            return 0
        return direction

    def _grid(self, bitmap: Bitmap) -> tuple[np.ndarray, np.ndarray]:
        """画面を格子に落として、格子ごとの相対輝度と «赤さ» を出す。"""
        sub = bitmap.data[np.ix_(self._ys, self._xs)]
        r = _LINEAR[sub[..., 0]].astype(np.float64)
        g = _LINEAR[sub[..., 1]].astype(np.float64)
        b = _LINEAR[sub[..., 2]].astype(np.float64)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # 彩度の高い赤かどうか。R が全体に対して突出しているか見る。
        total = r + g + b
        safe = np.where(total > 0, total, 1.0)
        red_mask = (total > 0.02) & (r / safe > self.settings["redRatio"])
        red_value = np.where(red_mask, r, 0.0)

        luma = np.bincount(self._cells, weights=lum.ravel(), minlength=self.cell_count) / self._counts
        red = np.bincount(self._cells, weights=red_value.ravel(), minlength=self.cell_count) / self._counts
        return luma, red

    def _compare_luma(self, before: np.ndarray, after: np.ndarray) -> int:
        """前後のフレームを比べて «遷移» の向きを返す。0 なら遷移なし。

        **明るくなったマスと暗くなったマスは別々に数えます。** 画面の半分が
        明るく、半分が暗くなるような «入れ替わり» を 1 つの閃光と数えないためです。
        """
        delta = after - before
        big = np.abs(delta) >= self.settings["luminanceDelta"]
        # 暗い側が十分に暗いときだけ «閃光» として数える
        dark_enough = np.minimum(before, after) <= self.settings["darkerCeiling"]
        counted = big & dark_enough
        brighter = int(np.count_nonzero(counted & (delta > 0)))
        darker = int(np.count_nonzero(counted & (delta <= 0)))
        threshold = self.cell_count * self.settings["areaRatio"]
        if brighter >= threshold and brighter >= darker:
            return 1
        if darker >= threshold and darker > brighter:
            return -1
        return 0

    def _compare_red(self, before: np.ndarray, after: np.ndarray) -> int:
        """赤の反転。輝度が変わらなくても «赤 ⇄ 別の色» は危険とされます。"""
        delta = after - before
        big = np.abs(delta) >= 0.2
        appeared = int(np.count_nonzero(big & (delta > 0)))
        vanished = int(np.count_nonzero(big & (delta <= 0)))
        threshold = self.cell_count * self.settings["areaRatio"]
        if appeared >= threshold and appeared >= vanished:
            return 1
        if vanished >= threshold and vanished > appeared:
            return -1
        return 0

    def _worst_window(self, frames: list[int]) -> dict[str, int]:
        """1 秒の窓を滑らせて «いちばん多かったところ» を探す。"""
        if not frames:
            return {"count": 0, "startFrame": 0, "endFrame": 0}
        best = {"count": 0, "startFrame": 0, "endFrame": 0}
        for i in range(len(frames)):
            j = i
            while j + 1 < len(frames) and frames[j + 1] - frames[i] < self.fps:
                j += 1
            count = j - i + 1
            if count > best["count"]:
                best = {"count": count, "startFrame": frames[i], "endFrame": frames[j]}
        return best

    def report(self) -> dict:
        """判定結果。**キー名は JS 版のまま**（JSON にそのまま出すため）。"""
        worst = self._worst_window(self.flashes)
        worst_red = self._worst_window(self.red_flashes)

        def to_seconds(window: dict[str, int]) -> dict | None:
            if window["count"] == 0:
                return None
            return {
                "count": window["count"],
                "startSeconds": window["startFrame"] / self.fps,
                "endSeconds": window["endFrame"] / self.fps,
            }

        limit = self.settings["maxPerSecond"]
        return {
            "ok": worst["count"] <= limit and worst_red["count"] <= limit,
            "limit": limit,
            "flashesPerSecond": worst["count"],
            "redFlashesPerSecond": worst_red["count"],
            "worst": to_seconds(worst),
            "worstRed": to_seconds(worst_red),
            "totalFlashes": len(self.flashes),
            "totalRedFlashes": len(self.red_flashes),
            "frames": self.frame_index,
        }


def describe_flash_report(report: dict) -> list[str]:
    """判定結果を人が読める行にする。"""
    if report["ok"]:
        worst = f"（最大 {report['worst']['count']} 回/秒）" if report["worst"] else "（閃光なし）"
        return [f"光過敏性発作の検査: 問題なし {worst}"]
    lines: list[str] = []
    if report["worst"] and report["worst"]["count"] > report["limit"]:
        lines.append(
            f"光過敏性発作の危険: {report['worst']['startSeconds']:.2f}〜{report['worst']['endSeconds']:.2f} 秒で"
            f" 毎秒 {report['worst']['count']} 回の閃光（上限 {report['limit']} 回）"
        )
    if report["worstRed"] and report["worstRed"]["count"] > report["limit"]:
        lines.append(
            f"光過敏性発作の危険: {report['worstRed']['startSeconds']:.2f}〜{report['worstRed']['endSeconds']:.2f} 秒で"
            f" 毎秒 {report['worstRed']['count']} 回の «彩度の高い赤» の反転（上限 {report['limit']} 回）"
        )
    lines.append("対策: 閃光の面積を画面の 1/4 未満にする / 頻度を下げる / 明暗の差を小さくする")
    return lines
