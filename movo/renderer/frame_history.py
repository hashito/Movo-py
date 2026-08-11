"""フレーム履歴（frameEcho）。

移植元: ``packages/renderer/src/frame-history.js``

過去に描いたフレームのビットマップとメッシュを覚えておき、残像として重ねます。
``echo`` が «過去のトランスフォームで今の絵を描き直す» のに対し、こちらは
«実際に描いた絵そのもの» を使うので、変形やエフェクトの結果まで残像に写ります。

メモリを食うので上限（``render.frameHistory``）を守り、近づいたら警告します。
"""

from __future__ import annotations

from movo.cli.console import logger
from movo.core.math import Mat2D, clamp, js_round
from movo.renderer.raster import parse_color


def draw_frame_echo(renderer, destination, layer: dict, spec: dict, draw_options: dict, frame_index: int) -> None:
    """過去フレームの描画結果を古い順に重ねる（新しいものが上に来る）。"""
    history = renderer.layer_history.get(layer["id"])
    if not history:
        return
    count = int(clamp(js_round(spec.get("count", 6) or 6), 1, 32))
    delay_frames = max(1, js_round(spec.get("delayFrames", 2) or 2))
    decay = clamp(spec.get("decay", 0.7) if spec.get("decay") is not None else 0.7, 0, 1)
    tint = None
    if spec.get("tint"):
        r, g, b, _ = parse_color(spec["tint"])
        amount = clamp(spec.get("tintAmount", 0.5) if spec.get("tintAmount") is not None else 0.5, 0, 1)
        tint = (r, g, b, amount)
    blend = spec.get("blend") or draw_options.get("blend")

    for step in range(count, 0, -1):
        wanted = frame_index - step * delay_frames
        entry = next((item for item in history if item["frame"] == wanted), None)
        if entry is None:
            continue
        alpha = entry["opacity"] * (decay**step)
        if alpha <= 0.002:
            continue
        matrix = entry["matrix"]
        if spec.get("scale") and spec["scale"] != 1:
            # 残像を少し縮める／広げる（軌跡に奥行きが出る）
            factor = spec["scale"] ** step
            matrix = Mat2D.multiply(matrix, [factor, 0, 0, factor, 0, 0])
        options = dict(draw_options)
        options.update({"blend": blend, "alpha": alpha, "tint": tint})
        entry["mesh"].draw(destination, entry["bitmap"], matrix, options)


def push_frame_history(renderer, layer: dict, spec: dict, entry: dict) -> None:
    """フレーム履歴に 1 件積み、必要な長さとメモリ上限で切り詰める。"""
    count = int(clamp(js_round(spec.get("count", 6) or 6), 1, 32))
    delay_frames = max(1, js_round(spec.get("delayFrames", 2) or 2))
    needed = count * delay_frames + 1
    limit = min(needed, renderer.frame_history_limit)
    history = renderer.layer_history.setdefault(layer["id"], [])
    history.append(entry)
    while len(history) > limit:
        history.pop(0)

    if needed > renderer.frame_history_limit and layer["id"] not in renderer._history_warned:
        renderer._history_warned.add(layer["id"])
        logger.warn(
            f'frameEcho on "{layer["id"]}" wants {needed} frames of history but render.frameHistory '
            f"is {renderer.frame_history_limit}; the oldest echoes are dropped "
            "(raise render.frameHistory to keep them)"
        )
    check_history_memory(renderer)


def check_history_memory(renderer) -> None:
    """履歴のメモリ使用量を見積もり、大きすぎる場合に一度だけ警告する。"""
    total = 0
    for history in renderer.layer_history.values():
        for entry in history:
            total += entry["bitmap"].data.nbytes
    renderer.history_bytes = total
    if total > 512 * 1024 * 1024 and not renderer._memory_warned:
        renderer._memory_warned = True
        logger.warn(
            f"frameEcho のフレーム履歴が {total / 1024 / 1024:.0f}MB になっています。"
            "render.frameHistory を下げるか、count / delayFrames を小さくしてください。"
        )


__all__ = ["check_history_memory", "draw_frame_echo", "push_frame_history"]
