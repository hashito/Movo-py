"""速度の実測。**回帰を見るための道具です。**

この移植の目的は速度なので、«速いはずだったのに遅くなっていた» を後から
気付くのがいちばん困ります。README.ja.md に載せた判断の根拠と同じ 4 つを、
いつでも測り直せるようにしてあります。

    python tools/bench.py                  4 つ全部
    python tools/bench.py --only raster    1 つだけ
    python tools/bench.py --json           数値だけ（回帰の比較に使う）
    python tools/bench.py --baseline b.json  前回と比べて遅くなった項目を出す
    python tools/bench.py --compare-js      JS 版と同じ JSON を描いて秒を比べる

## 測るもの

| 名前 | 何を見るか | 使うもの |
| --- | --- | --- |
| `fullscreen` | 全画面 1 パス | NumPy |
| `raster` | 多角形 1 枚の塗り | Numba |
| `frame` | 1 フレームの合成（239 レイヤー相当） | 両方 |
| `export` | 短い動画の書き出し | 全部 |

## 測り方の決めごと

- **JIT のコンパイルは測らない。** 1 回空回ししてから測ります。混ぜると
  «初回だけ 3 秒» という数字になり、回帰かどうかが分からなくなります。
- **中央値を出す。** 平均だと、たまたま走った GC や OS の割り込みに引きずられます。
- 1 回が速すぎるものは繰り返して割ります（時計の分解能に負けないため）。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

# `python tools/bench.py` を直接叩けるようにします（pip install しなくても回る）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from movo.core.bitmap import Bitmap  # noqa: E402

# 実測の基準にする 1 フレームの大きさ。README の表と揃えてあります。
WIDTH, HEIGHT = 1280, 720

# 比べる相手（同じ 1280x720）。README.ja.md に載せた実測です。
#
# **どれと比べているかを取り違えないでください。** `raster-kernel` は «走査線
# そのもの»、`raster` は «塗って合成するところまで» です。前者を後者と比べると
# 「10 倍以上遅い」という嘘の回帰報告になります（実際に一度そう出しました）。
REFERENCE = {
    "fullscreen": ("純 Python のループ", 720.0),
    "raster-kernel": ("NumPy の一括判定", 30.4),
    "frame": ("JS 版の 1 フレーム", 1500.0),
}


def measure(function: Callable[[], Any], repeat: int = 30, inner: int = 1) -> float:
    """1 回あたりのミリ秒（中央値）。

    `inner` は «1 回が速すぎて時計の分解能に負ける» ときに、まとめて回して
    割るための回数です。
    """
    function()  # JIT のコンパイルと初回のページフォルトはここで済ませる
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        for _ in range(inner):
            function()
        samples.append((time.perf_counter() - started) / inner * 1000)
    return statistics.median(samples)


# ── 1. 全画面 1 パス ────────────────────────────────────────


def bench_fullscreen() -> dict[str, Any]:
    """全画面に一様な処理を **1 パス** かける。

    エフェクトも合成もこの形なので、**1 フレームに 10 個前後乗ります**。
    ここが 1 ミリ秒違うと、153 秒の MV では 45 秒違います。

    測るのは «256 個の対応表で置き換える»（明度・カーブ・LUT はどれもこの形）
    にしています。**足し算や掛け算で書くと、そちらが 1 パスで済まないことが
    あります** — `uint8` のまま掛けると 255 で巻き戻るので `uint16` へ上げて
    戻す必要があり、それだけで 3 パス増えて «NumPy が遅い» という誤った数字に
    なります（実際に 25 ms と出ました）。
    """
    bitmap = Bitmap(WIDTH, HEIGHT)
    bitmap.data[...] = np.random.default_rng(1234).integers(0, 255, bitmap.data.shape, dtype=np.uint8)
    table = np.minimum(np.arange(256, dtype=np.uint16) * 5 // 4, 255).astype(np.uint8)

    def run() -> None:
        np.take(table, bitmap.data, out=bitmap.data)

    milliseconds = measure(run)
    return {
        "name": "fullscreen",
        "label": "全画面 1 パス（NumPy）",
        "ms": milliseconds,
        "note": f"1 フレームに 10 個乗ると {milliseconds * 10:.1f} ms",
    }


# ── 2. 多角形の塗り ────────────────────────────────────────


# README の実測と同じ形（画面の半分ほどを覆う四角）
SHAPE = [200.0, 100.0, 900.0, 180.0, 1000.0, 600.0, 300.0, 650.0]


def bench_raster_kernel() -> dict[str, Any]:
    """多角形 1 枚の **走査線だけ** を測る。

    **ここが NumPy では遅くなる場所です**（囲む矩形の全画素を辺ごとに判定する
    ことになり、O(面積 × 辺) になります）。Numba の走査線は O(辺 × 行) で、
    塗る必要のある画素にしか触りません。README の «30.4 ms 対 0.296 ms» は
    この 2 つを比べたものなので、比べる相手もここに合わせてあります。

    被覆率の配列は使い回します。**毎回確保すると «確保の時間» を走査線の時間
    として数えることになります**（それは下の `raster` で別に測ります）。
    """
    from movo.renderer.raster import rasterize_contours

    coverage = np.zeros((HEIGHT, WIDTH), np.float32)

    def run() -> None:
        rasterize_contours([SHAPE], WIDTH, HEIGHT, "nonzero", coverage)

    milliseconds = measure(run, repeat=20, inner=5)
    return {
        "name": "raster-kernel",
        "label": "多角形 1 枚の走査線（Numba）",
        "ms": milliseconds,
        "note": f"239 レイヤーで {milliseconds * 239:.0f} ms/フレーム",
    }


def bench_raster() -> dict[str, Any]:
    """多角形 1 枚を «塗って合成する» ところまで測る。

    走査線そのものより重くなります。塗った範囲へ色を書き、下の絵と重ねる分が
    乗るからです。**実際に効いてくるのはこちらの数字** なので、両方を並べます。
    """
    from movo.renderer.raster import fill_contours

    bitmap = Bitmap(WIDTH, HEIGHT)

    def run() -> None:
        fill_contours(bitmap, [SHAPE], "#ff8844")

    milliseconds = measure(run, repeat=20, inner=3)
    return {
        "name": "raster",
        "label": "多角形 1 枚の塗り＋合成",
        "ms": milliseconds,
        # **この形は画面の半分ほどを覆います。** 239 倍しても «1 フレーム» には
        # なりません（実際のフレームは小さい図形が大半なので、下の `frame` を
        # 見てください）。ここは «大きい 1 枚» の上限を知るための数字です。
        "note": f"画面の 4 割ほどを覆う 1 枚。塗る面積に比例します",
    }


def bench_raster_numpy() -> dict[str, Any]:
    """**比較のため** に、同じ多角形を NumPy の一括判定で塗る。

    これは実装ではありません。«NumPy でベクトル化すれば速い» が多角形の塗りには
    当てはまらないことを、いつでも測り直せるようにするためのものです。
    """
    xs = np.arange(WIDTH, dtype=np.float32) + 0.5
    ys = np.arange(HEIGHT, dtype=np.float32) + 0.5
    px = np.array([200, 900, 1000, 300], np.float32)
    py = np.array([100, 180, 600, 650], np.float32)

    def run() -> None:
        gx, gy = np.meshgrid(xs, ys)
        inside = np.zeros((HEIGHT, WIDTH), bool)
        for i in range(len(px)):
            j = (i + 1) % len(px)
            # 辺ごとに «囲む矩形いっぱいの一時配列» を作るのが効かない理由です
            crossing = (py[i] > gy) != (py[j] > gy)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at = (px[j] - px[i]) * (gy - py[i]) / (py[j] - py[i]) + px[i]
            inside ^= crossing & (gx < x_at)
        inside.sum()

    return {"name": "raster-numpy", "label": "多角形 1 枚の塗り（NumPy 一括判定・比較用）", "ms": measure(run, repeat=5)}


# ── 3. 1 フレームの合成 ────────────────────────────────────


def bench_frame() -> dict[str, Any]:
    """239 レイヤーぶんの塗りと合成を 1 フレーム分やる。

    JS 版の «1280x720 / 239 レイヤー / 1,500 ms» と同じ土俵にしてあります。

    **レイヤーごとに全画面のバッファを取ってはいけません。** 最初そう書いたら
    1 フレーム 43 秒になりました（239 回の 3.7MB 確保と全画面合成が乗るためです）。
    実際のレンダラーは «塗った範囲» にしか触らないので、こちらもそう測ります。
    """
    from movo.renderer.raster import fill_contours

    canvas = Bitmap(WIDTH, HEIGHT)
    rng = np.random.default_rng(4242)
    # 実際のフレームに近づけて、大小の図形を混ぜます
    shapes = []
    for _ in range(239):
        cx, cy = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
        size = rng.uniform(20, 260)
        shapes.append(
            [cx, cy, cx + size, cy + size * 0.4, cx + size * 0.7, cy + size, cx - size * 0.2, cy + size * 0.6]
        )

    def run() -> None:
        canvas.data[...] = 0
        for contour in shapes:
            fill_contours(canvas, [contour], "#88aaff", alpha=0.8)

    milliseconds = measure(run, repeat=5)
    return {
        "name": "frame",
        "label": "1 フレームの合成（239 レイヤー）",
        "ms": milliseconds,
        "note": f"153 秒の MV（30fps）なら {milliseconds * 153 * 30 / 1000 / 60:.1f} 分",
    }


# ── 4. 短い動画の書き出し ──────────────────────────────────


def bench_export() -> dict[str, Any]:
    """短い動画を実際に書き出す。

    **繋がっていない部分があれば、ここで «測れない» と正直に返します。**
    測れなかったことを 0 秒として混ぜると、回帰の比較が壊れます。
    """
    from movo.cli import bridge

    missing = [row["module"] for row in bridge.module_status() if not row["connected"]]
    if missing:
        return {
            "name": "export",
            "label": "短い動画の書き出し",
            "ms": None,
            "skipped": f'まだ繋がっていないものがあります: {", ".join(missing)}（後で繋ぐ）',
        }

    from movo.cli.pipeline import create_session, render_video

    project = {
        "movoVersion": "1.0",
        "project": {"name": "bench", "seed": 1},
        "video": {"width": WIDTH, "height": HEIGHT, "fps": 30, "duration": 2, "background": "#101020"},
        "scenes": [
            {
                "id": "main",
                "start": 0,
                "duration": 2,
                "layers": [
                    {
                        "id": f"s{i}",
                        "type": "shape",
                        "shape": {"type": "rectangle", "width": 200, "height": 120, "fill": "#88aaff"},
                        "transform": {"x": 100 + i * 90, "y": 300, "rotation": {"expression": "time * 40"}},
                    }
                    for i in range(12)
                ],
            }
        ],
        "output": {"format": "png-sequence"},
    }
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "bench.json"
        path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
        started = time.perf_counter()
        session = create_session(str(path), {"quality": "draft"})
        result = render_video(session, {"output": str(Path(work) / "frames"), "format": "png-sequence", "quiet": True})
        elapsed = time.perf_counter() - started
    frames = result["frames"]
    return {
        "name": "export",
        "label": "短い動画の書き出し（60 フレーム）",
        "ms": elapsed * 1000,
        "note": f"{frames} フレーム / {frames / elapsed:.1f} fps",
    }


BENCHES = {
    "fullscreen": bench_fullscreen,
    "raster-kernel": bench_raster_kernel,
    "raster": bench_raster,
    "raster-numpy": bench_raster_numpy,
    "frame": bench_frame,
    "export": bench_export,
}

# 既定で回すもの。`raster-numpy` は «比較用» なので既定には入れません
# （実装ではないものの数字が回帰の表に並ぶと紛らわしいためです）。
DEFAULT_BENCHES = ["fullscreen", "raster-kernel", "raster", "frame", "export"]


# ── JS 版との比較 ──────────────────────────────────────────


def compare_with_js(project_file: str, js_root: str | None = None) -> dict[str, Any] | None:
    """**同じ JSON を両方で描いて秒を比べます。**

    これが «速くなった» を言える唯一の測り方です。個別の関数がいくら速くても、
    1 本書き出す時間が変わっていなければ意味がありません。

    JS 版が見つからないときは None を返します（無い環境で止めないため）。
    """
    root = Path(js_root or Path(__file__).resolve().parent.parent.parent / "Movo")
    entry = root / "packages" / "cli" / "bin" / "movo.js"
    if not entry.is_file():
        return None
    node = shutil.which("node")
    if not node:
        return None

    with tempfile.TemporaryDirectory() as work:
        results = {}
        for label, command in (
            ("js", [node, str(entry), "render", project_file, "-o", str(Path(work) / "js.mp4"), "--quiet"]),
            ("py", [sys.executable, "-m", "movo.cli.main", "render", project_file, "-o", str(Path(work) / "py.mp4"), "--quiet"]),
        ):
            started = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
            elapsed = time.perf_counter() - started
            results[label] = {"seconds": elapsed, "ok": completed.returncode == 0, "stderr": (completed.stderr or "")[-400:]}
        if results["js"]["ok"] and results["py"]["ok"]:
            results["speedup"] = results["js"]["seconds"] / max(0.001, results["py"]["seconds"])
        return results


# ── 表示 ────────────────────────────────────────────────────


def describe(rows: list[dict[str, Any]], baseline: dict[str, float] | None = None) -> None:
    print(f"movo bench — {platform.python_version()} / {platform.machine()} / {os.cpu_count()} コア")
    print(f"  {WIDTH}x{HEIGHT}\n")
    for row in rows:
        if row.get("ms") is None:
            print(f'  {row["label"]}')
            print(f'      測れませんでした: {row.get("skipped", "")}')
            continue
        line = f'  {row["label"]:<34} {row["ms"]:>9.3f} ms'
        reference = REFERENCE.get(row["name"])
        if reference:
            what, milliseconds = reference
            line += f"   （{what} {milliseconds} ms の {milliseconds / row['ms']:.1f} 倍）"
        print(line)
        if row.get("note"):
            print(f'      {row["note"]}')
        if baseline and row["name"] in baseline:
            before = baseline[row["name"]]
            ratio = row["ms"] / before
            # **10% 以上遅くなったら «回帰» と呼びます。** それ以下は測定の揺れです。
            if ratio > 1.1:
                print(f"      ! 前回より {(ratio - 1) * 100:.0f}% 遅くなっています（{before:.3f} ms → {row['ms']:.3f} ms）")
            elif ratio < 0.9:
                print(f"      速くなりました（{before:.3f} ms → {row['ms']:.3f} ms）")


def main() -> int:
    parser = argparse.ArgumentParser(description="Movo の速度を測る")
    parser.add_argument("--only", action="append", choices=list(BENCHES), help="測るものを絞る（何度でも指定できます）")
    parser.add_argument("--json", action="store_true", help="数値だけを JSON で出す")
    parser.add_argument("--baseline", help="前回の --json の結果。遅くなった項目を指摘します")
    parser.add_argument("--compare-js", metavar="PROJECT", help="同じ JSON を JS 版と両方で描いて秒を比べる")
    parser.add_argument("--js-root", help="JS 版（Movo）の置き場。既定は ../Movo")
    args = parser.parse_args()

    names = args.only or DEFAULT_BENCHES
    rows = [BENCHES[name]() for name in names]

    baseline = None
    if args.baseline:
        loaded = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline = {row["name"]: row["ms"] for row in loaded.get("benches", []) if row.get("ms") is not None}

    comparison = None
    if args.compare_js:
        comparison = compare_with_js(args.compare_js, args.js_root)

    if args.json:
        print(json.dumps({"benches": rows, "js": comparison}, ensure_ascii=False, indent=2))
        return 0

    describe(rows, baseline)
    if comparison is None and args.compare_js:
        print("\n  JS 版（../Movo）か node が見つからないので、比較はできませんでした")
    elif comparison:
        print("\n  同じ JSON を両方で描いた結果")
        for label in ("js", "py"):
            state = "" if comparison[label]["ok"] else "（失敗）"
            print(f'    {label:<3} {comparison[label]["seconds"]:>7.2f} 秒 {state}')
        if "speedup" in comparison:
            print(f'    Python 版は JS 版の {comparison["speedup"]:.2f} 倍')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
