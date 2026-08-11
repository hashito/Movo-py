"""測った «型» を、目標の型と突き合わせる。

差分だけ出しても «で、どうすれば» が分からないので、指標ごとに
**Movo での直し方**を持たせています。**ここがこの機能の値打ちです。**

目標値（``profiles/*.json``）は人が書きます。参考にした映像を見て数値に
言語化したものです。**参考映像そのものは同梱も配布もしません**（著作物なので）。
手元に権利のある映像があれば、それを測って目標にできます。
"""

from __future__ import annotations

import math
from typing import Any

#: 見る指標の定義。
#:
#: ``path`` は profile の中の場所、``range`` は目標の幅、``low`` / ``high`` は
#: 外れたときの助言です。**助言は «低すぎるとき» と «高すぎるとき» で別々**に
#: 持ちます。方向が分からないと直しようがないためです。
METRICS: list[dict[str, Any]] = [
    {
        "key": "cutSeconds",
        "label": "カット尺",
        "path": "cuts.medianSeconds",
        "unit": "秒",
        "low": 'カットが短すぎます。落ち着かせたいところは尺を伸ばしてください（小節で書くなら "4bar" 以上）',
        "high": 'カットが長すぎます。尺を小節で刻んでください（"4bar" など）。make-mv なら --max-bars で上限を下げられます',
    },
    {
        "key": "cutsPerMinute",
        "label": "毎分のカット数",
        "path": "cuts.perMinute",
        "unit": "本",
        "low": "カットが足りません。区間の変わり目でシーンを割ってください",
        # 毎分 180 本を超えると «毎秒 3 回» に達します。光過敏性発作の目安と同じ線です。
        "high": "カットが速すぎます。毎分 180 本で «毎秒 3 回» に達し、光過敏性発作の危険域に入ります。尺を伸ばすか、明暗差の小さい繋ぎにしてください",
    },
    {
        "key": "motion",
        "label": "動きの量",
        "path": "motion.mean",
        "low": "画が止まって見えます。カットの中でも動かしてください（ken-burns / float / カメラ移動）",
        "high": "動きが多すぎて何を見ればよいか分かりません。動かす対象を絞ってください",
    },
    {
        "key": "stillRatio",
        "label": "静止の割合",
        "path": "motion.stillRatio",
        "low": None,
        "high": "止まっている時間が長すぎます。呼吸や浮遊など、ゆっくりした動きを足してください",
    },
    {
        "key": "colors",
        "label": "実質の色数",
        "path": "palette.effectiveColors",
        "unit": "色",
        "low": "色数が少なすぎます。差し色を足すか、階調を戻してください",
        "high": "中間調が残っています。posterize の levels を下げるか、dither にパレットを渡してください",
    },
    {
        "key": "saturation",
        "label": "彩度",
        "path": "palette.saturation",
        "low": "くすんでいます。colorAdjust の saturation を上げるか、原色を使ってください",
        "high": "派手すぎます。colorAdjust の saturation を下げてください（値は «増減量» で、0 が変化なし）",
    },
    {
        "key": "contrast",
        "label": "コントラスト",
        "path": "palette.contrast",
        "low": "明暗の差が小さく、のっぺり見えます。背景を暗く、文字を明るくしてください",
        "high": "明暗が強すぎます。中間の明るさを持つ面を足してください",
    },
    {
        "key": "detail",
        "label": "細かさ（文字・模様）",
        "path": "detail.edgeDensity",
        "low": "画面が寂しく見えます。文字を大きくするか、模様・粒子を足してください",
        "high": "要素が多すぎて読みにくいかもしれません。文字か背景のどちらかを整理してください",
    },
]


def _pick(obj: Any, path: str) -> Any:
    """``"a.b.c"`` を辿って値を取る。"""
    node = obj
    for key in path.split("."):
        if node is None or not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def compare_profile(profile: dict, target: dict) -> dict:
    """目標と突き合わせる。

    :param profile: :meth:`VideoProfiler.report` の結果
    :param target: 目標。``{"cutSeconds": [4.7, 7.0], "colors": [0, 5], ...}``
    """
    results = []
    for metric in METRICS:
        rng = target.get(metric["key"])
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        value = _pick(profile, metric["path"])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        low, high = rng
        status = "ok"
        advice = None
        if value < low:
            status = "low"
            advice = metric["low"]
        elif value > high:
            status = "high"
            advice = metric["high"]
        results.append(
            {
                "key": metric["key"],
                "label": metric["label"],
                "unit": metric.get("unit", ""),
                "value": value,
                "min": low,
                "max": high,
                "status": status,
                "advice": advice,
            }
        )
    return {"ok": all(r["status"] == "ok" for r in results), "results": results}


def compare_to_reference(mine: dict, reference: dict, tolerance: float = 0.25) -> dict:
    """2 つの profile を直接くらべる。

    目標が «幅» ではなく «もう 1 本の映像» のときに使います。相手の値の ±25% を
    目標の幅とみなします。**厳密に一致させても意味が無い**ので、«だいたい同じ
    作りか» を見る程度の幅です。
    """
    target: dict[str, list[float]] = {}
    for metric in METRICS:
        value = _pick(reference, metric["path"])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        span = abs(value) * tolerance
        target[metric["key"]] = [value - span, value + span]
    return compare_profile(mine, target)


def describe_comparison(comparison: dict) -> list[str]:
    """人が読める行にする。"""
    lines: list[str] = []
    for result in comparison["results"]:
        mark = "✔" if result["status"] == "ok" else "✖"
        rng = f"{_format_number(result['min'])}〜{_format_number(result['max'])}"
        # 全角スペースで詰めるのは、日本語のラベルの幅を揃えるためです
        label = result["label"].ljust(10, "　")
        lines.append(
            f"  {mark} {label} {_format_number(result['value'])}{result['unit']}（目標 {rng}{result['unit']}）"
        )
        if result["advice"]:
            lines.append(f"      {result['advice']}")
    if not comparison["results"]:
        lines.append("  くらべられる指標がありませんでした（目標の書き方を確認してください）")
    return lines


def _format_number(value: float) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
