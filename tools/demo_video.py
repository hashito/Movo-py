#!/usr/bin/env python
"""移植できたところだけで動画を 1 本作る（動作確認用）。

`movo render` の一式はまだ繋がっていませんが、**ラスタライザ・図形・文字・
エフェクトは JS 版と画素一致まで確認済み**なので、そこを直接呼んで動画にします。

  python tools/demo_video.py -o tmp/demo.mp4

ここで測った 1 フレームの秒数が、そのまま «Python 版の素の速さ» です。
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movo.core.bitmap import Bitmap  # noqa: E402
from movo.core.png import encode_png  # noqa: E402
from movo.renderer import effects, shapes, text  # noqa: E402
from movo.renderer.font import FontManager  # noqa: E402
from movo.renderer.raster import draw_bitmap  # noqa: E402

W, H, FPS, SECONDS = 960, 540, 24, 6

# 文字を描くにはフォントを解決する係が要ります。1 度だけ作って使い回します
# （毎フレーム作ると、フォントを読み直すぶんだけ丸損です）。
FONTS = None
BPM = 120


def beat_pulse(t: float, bpm: float = BPM, decay: float = 9.0) -> float:
    """拍のたびに 1 → 0 へ落ちる値。JS 版の `beatPulse` と同じ形です。"""
    length = 60.0 / bpm
    phase = (t % length) / length
    return math.exp(-phase * decay)


def draw_frame(index: int) -> Bitmap:
    t = index / FPS
    canvas = Bitmap(W, H)
    # 背景。上から下へ暗くする縦グラデーションを NumPy で一息に作ります。
    top = np.array([18, 26, 34], np.float32)
    bottom = np.array([8, 12, 18], np.float32)
    ramp = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    canvas.data[..., :3] = (top * (1 - ramp) + bottom * ramp).astype(np.uint8)
    canvas.data[..., 3] = 255

    # 拍で脈打つ輪を 3 本。位相をずらして «追いかける» 見た目にします。
    for ring in range(3):
        pulse = beat_pulse(t - ring * 0.08)
        radius = 90 + ring * 46 + pulse * 26
        stroke = shapes.render_shape(
            {
                "type": "circle",
                "radius": radius,
                "fill": None,
                "stroke": {"color": ["#39c5bb", "#8fd14f", "#f2b705"][ring], "width": 3 + pulse * 5},
            },
            1,
        )
        draw_bitmap(canvas, stroke["bitmap"], W // 2 - stroke["bitmap"].width // 2, H // 2 - stroke["bitmap"].height // 2, 0.55)

    # 円運動する玉。等速で回し、拍で少しだけ大きくします。
    angle = t * 0.9
    ball = shapes.render_shape({"type": "circle", "radius": 22 + beat_pulse(t) * 10, "fill": "#f2f4ee"}, 1)
    bx = int(W / 2 + math.cos(angle) * 210 - ball["bitmap"].width / 2)
    by = int(H / 2 + math.sin(angle) * 118 - ball["bitmap"].height / 2)
    draw_bitmap(canvas, ball["bitmap"], bx, by, 1.0)

    # 文字。組版・縁取り・影は移植済みのものをそのまま使います。
    title = text.render_text(
        "Movo-py",
        {"size": 76, "color": "#f2f4ee", "align": "center", "weight": "bold", "letterSpacing": 2,
         "stroke": {"color": "#0e1512", "width": 5}},
        FONTS,
    )
    draw_bitmap(canvas, title["bitmap"], W // 2 - title["bitmap"].width // 2, 52, 1.0)

    caption = text.render_text(
        f"NumPy + Numba   {index + 1:3d} / {FPS * SECONDS}",
        {"size": 24, "color": "#8fd14f", "align": "center"},
        FONTS,
    )
    draw_bitmap(canvas, caption["bitmap"], W // 2 - caption["bitmap"].width // 2, H - 74, 0.9)

    # 仕上げ。全画面のエフェクトは NumPy の独壇場です。
    for spec in ({"type": "bloom", "threshold": 0.65, "amount": 0.35, "radius": 18},
                 {"type": "vignette", "amount": 0.38},
                 {"type": "noise", "amount": 0.05}):
        canvas = effects.apply_effect(canvas, spec, {"time": t, "frame": index, "fps": FPS})
    return canvas


def main() -> int:
    global FONTS
    FONTS = FontManager()
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="tmp/demo.mp4")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / "_demo_frames"
    frames_dir.mkdir(exist_ok=True)

    total = FPS * SECONDS
    draw_frame(0)  # Numba の JIT をここで済ませ、計測から外す
    started = time.perf_counter()
    for i in range(total):
        frame = draw_frame(i)
        (frames_dir / f"f{i:04d}.png").write_bytes(encode_png(frame))
        if (i + 1) % 24 == 0:
            done = time.perf_counter() - started
            print(f"  {i + 1}/{total} フレーム  {done / (i + 1) * 1000:.0f} ms/フレーム")
    elapsed = time.perf_counter() - started
    print(f"\n描画 {total} フレーム / {elapsed:.1f} 秒  →  {elapsed / total * 1000:.0f} ms/フレーム")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg が無いので PNG 連番のままにします:", frames_dir)
        return 0
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-framerate", str(FPS), "-i", str(frames_dir / "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
        check=True,
    )
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"→ {out}  （{out.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
