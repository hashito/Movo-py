"""文字の組版とラスタライズ。

複数行・明示的な改行・折返し（日本語も見る）・左右と上下のそろえ・字間・塗り・
縁取り・影に対応します。字形の輪郭は `movo.renderer.font` から取ります。

ここには «1 行の中で色や大きさを変える»（リッチテキスト）と «日本語の組み方»
（禁則処理・ルビ・枠に収める自動縮小）も入っています。どちらも `layout_text` が
«1 文字 = 1 セル» に分解する段で吸収していて、`render_text` /
`render_animated_text` は出来上がったセルに書いてある色と大きさをそのまま使う
だけです。**組み方を 1 か所に集めないと**、縦書き・textAnimator・カラオケ塗りの
それぞれで同じ計算を書き直すことになり、必ずどれかがずれます。

## データの持ち方

レイアウトの結果は **`dict`（キーは snake_case）**です。

    layout = {
      "lines": [ {"text":…, "glyphs":[…], "width":…, "ascent":…, "descent":…,
                  "ruby_glyphs":[…]} , … ],
      "width":…, "height":…, "ascent":…, "descent":…,
      "line_height":…, "natural_line_height":…, "size":…, "font":…, "vertical": bool,
      "fit_scale": float,   # style.fit で縮めたときの倍率
    }
    glyph = {"font":…, "glyph_index":…, "x":…, "advance":…, "scale":…, "char":…,
             "size":…, "color":…, "group":…, "pad_before":…, "pad_after":…,
             "dy":…, "baseline":…, "ruby": bool, "parent_index": int}

いっぽう **`style` と `spec`（JSON から来る設定）は camelCase のまま**です。
JSON の書き方は JS 版と完全に互換なので、入力のキーは直しません。
"""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable

import numpy as np

from . import kernels
from .font import is_cjk
from .raster import (
    clamp,
    draw_bitmap,
    fill_coverage,
    flatten_quadratic,
    js_number,
    parse_color,
    rasterize_contours,
    stroke_to_contours,
)
from .text_extras import (
    apply_stroke_order,
    box_blur,
    draw_text_box,
    layout_text_on_path,
    quantize_coverage,
    random_font_for,
)

try:  # pragma: no cover
    from movo.core.bitmap import Bitmap
except Exception:  # pragma: no cover
    Bitmap = None


# ── イージング ──────────────────────────────────────────────────
#
# `movo.animation` は別の担当が移植中です。まだ無いときのために最小限を
# 持っておきます（textAnimator で使う分だけ）。
try:  # pragma: no cover
    from movo.animation.easing import get_easing as _core_get_easing
except Exception:  # pragma: no cover
    _core_get_easing = None

_FALLBACK_EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda x: x,
    "hold": lambda x: 0.0,
    "step": lambda x: 1.0 if x >= 1 else 0.0,
    "easeIn": lambda x: x * x,
    "easeOut": lambda x: 1 - (1 - x) * (1 - x),
    "easeInOut": lambda x: 2 * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 2) / 2,
    "easeInQuad": lambda x: x * x,
    "easeOutQuad": lambda x: 1 - (1 - x) * (1 - x),
    "easeInOutQuad": lambda x: 2 * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 2) / 2,
    "easeInCubic": lambda x: x**3,
    "easeOutCubic": lambda x: 1 - (1 - x) ** 3,
    "easeInOutCubic": lambda x: 4 * x**3 if x < 0.5 else 1 - ((-2 * x + 2) ** 3) / 2,
    "easeInQuart": lambda x: x**4,
    "easeOutQuart": lambda x: 1 - (1 - x) ** 4,
    "easeInQuint": lambda x: x**5,
    "easeOutQuint": lambda x: 1 - (1 - x) ** 5,
    "easeInSine": lambda x: 1 - math.cos((x * math.pi) / 2),
    "easeOutSine": lambda x: math.sin((x * math.pi) / 2),
    "easeInOutSine": lambda x: -(math.cos(math.pi * x) - 1) / 2,
    "easeOutExpo": lambda x: 1.0 if x == 1 else 1 - 2 ** (-10 * x),
    "easeInExpo": lambda x: 0.0 if x == 0 else 2 ** (10 * x - 10),
    "easeOutBack": lambda x: 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2,
}


def _get_easing(name: str | None) -> Callable[[float], float]:
    if _core_get_easing is not None:  # pragma: no cover
        return _core_get_easing(name)
    return _FALLBACK_EASINGS.get(name or "linear", _FALLBACK_EASINGS["linear"])


# ══════════════════════════════════════════════════════════════════
# 字形 → 画素座標の輪郭
# ══════════════════════════════════════════════════════════════════


def _coverage_for(contours, width, height, style, fill_rule: str = "nonzero"):
    """字形をラスタライズして、必要なら «ドット» に丸める。

    塗りと縁取りで同じ丸め方をしないと、縁だけなめらかなままになって
    ビットマップフォント風にならないので、ここに一本化しています。
    """
    return quantize_coverage(rasterize_contours(contours, width, height, fill_rule), width, height, style)


def glyph_contours(outline, scale: float, origin_x: float, baseline_y: float) -> list[list[float]]:
    """フォント単位の字形を画素座標の輪郭に直す。

    TrueType の輪郭は «曲線上の点» と «制御点» が混ざった列です。制御点が続く
    ところには中点を挟んで 2 次ベジェに割り、最後は始点へ戻して閉じます。

    :param outline: `Font.glyph()` の戻り。各輪郭は `(n, 3)` の配列（x, y, on_curve）
    """
    contours: list[list[float]] = []
    for points in outline.contours:
        n = points.shape[0]
        if n == 0:
            continue
        on_curve = points[:, 2]
        found = np.nonzero(on_curve)[0]
        if found.size == 0:
            # 曲線上の点が 1 つも無い輪郭。両端の中点を «仮の始点» にする
            start_x = (points[0, 0] + points[n - 1, 0]) / 2
            start_y = (points[0, 1] + points[n - 1, 1]) / 2
            start_index = 0
        else:
            start_index = int(found[0])
            start_x = points[start_index, 0]
            start_y = points[start_index, 1]

        def dx(x: float) -> float:
            return origin_x + x * scale

        def dy(y: float) -> float:
            # フォントの y は上向き、画面の y は下向きなので反転する
            return baseline_y - y * scale

        out: list[float] = [dx(start_x), dy(start_y)]
        current_x = start_x
        current_y = start_y
        control: tuple[float, float] | None = None
        for k in range(1, n + 1):
            point = points[(start_index + k) % n]
            px = point[0]
            py = point[1]
            if point[2]:
                if control is not None:
                    flatten_quadratic(out, dx(current_x), dy(current_y), dx(control[0]), dy(control[1]), dx(px), dy(py))
                    control = None
                else:
                    out.append(dx(px))
                    out.append(dy(py))
                current_x = px
                current_y = py
            elif control is not None:
                # 制御点が 2 つ続いたら、その中点が «曲線上の点» になる約束
                mid_x = (control[0] + px) / 2
                mid_y = (control[1] + py) / 2
                flatten_quadratic(
                    out, dx(current_x), dy(current_y), dx(control[0]), dy(control[1]), dx(mid_x), dy(mid_y)
                )
                current_x = mid_x
                current_y = mid_y
                control = (px, py)
            else:
                control = (px, py)
        if control is not None:
            flatten_quadratic(
                out, dx(current_x), dy(current_y), dx(control[0]), dy(control[1]), dx(start_x), dy(start_y)
            )
        contours.append(out)
    return contours


# ══════════════════════════════════════════════════════════════════
# インラインのリッチテキスト
# ══════════════════════════════════════════════════════════════════

#: 簡易記法の開きタグ。`<c:#39c5bb>` `<s:96>` `<s:1.5x>` `<f:Serif>` `<b>` `<i>`
MARKUP_OPEN = re.compile(r"^<(c|s|f|b|i)(?::([^<>]*))?>")
#: 簡易記法の閉じタグ。`</c>` `</s>` `</f>` `</b>` `</i>`
MARKUP_CLOSE = re.compile(r"^</(c|s|f|b|i)>")


def parse_text_markup(source) -> list[dict]:
    """簡易記法をラン（同じ見た目が続くひとかたまり）に分解する。

    AviUtl の制御文字にあたるもので、歌詞のように «1 語だけ色を変える» ときに
    `runs` を手で書くより速い、という一点のためにあります。

    **認識できない `<…>` は «ただの文字» として残します。** `<` を含む普通の
    文章を記法の書き間違いとして壊してしまうより、そのまま出すほうが安全です。
    """
    text = "" if source is None else str(source)
    runs: list[dict] = []
    stack: list[tuple[str, str]] = []
    buffer = ""

    def current() -> dict:
        style: dict = {}
        for tag, value in stack:
            if tag == "c" and value:
                style["color"] = value
            elif tag == "s" and value:
                # `1.5x` は «基準の何倍» の意味。倍率で書いておくと高品質出力
                # （superSample）でも比が崩れません。
                if re.search(r"x$", value, re.I):
                    style["sizeScale"] = _parse_float(value)
                else:
                    style["size"] = _parse_float(value)
            elif tag == "f" and value:
                style["family"] = value
            elif tag == "b":
                style["bold"] = True
            elif tag == "i":
                style["italic"] = True
        return style

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        run = {"t": buffer}
        run.update(current())
        runs.append(run)
        buffer = ""

    index = 0
    while index < len(text):
        if text[index] == "<":
            rest = text[index:]
            close = MARKUP_CLOSE.match(rest)
            if close:
                # いちばん内側の同じタグを閉じる（対応が無ければ黙って無視する）
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == close.group(1):
                        flush()
                        del stack[i]
                        break
                index += len(close.group(0))
                continue
            open_tag = MARKUP_OPEN.match(rest)
            if open_tag:
                flush()
                stack.append((open_tag.group(1), (open_tag.group(2) or "").strip()))
                index += len(open_tag.group(0))
                continue
        buffer += text[index]
        index += 1
    flush()
    return runs


def _parse_float(value) -> float:
    """JS の `Number.parseFloat` 相当（先頭から読めるだけ読む）。"""
    m = re.match(r"\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", str(value))
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except ValueError:
        return float("nan")


def _normalize_run(run: dict | None, base_size: float) -> dict:
    """ランの書き方の揺れ（`t` / `text` / `content`）を 1 つに揃える。"""
    source = run or {}
    text = str(source.get("t") or source.get("text") or source.get("content") or "")
    normalized: dict = {"text": text}
    # 大きさは «基準サイズの何倍» で持ちます。レンダラーは高品質出力のときに
    # `style.size` だけを倍にするので、ランに実寸を残すと比が崩れます。
    #
    # `sizeScale`（JSON の書き方）と `size_scale`（この関数が返す書き方）の
    # **両方を読みます**。`resolve_text_style` が正規化したランがもう一度ここへ
    # 来ることがあり、片方しか見ないと «倍率が消えて» 高品質出力で比が崩れます。
    size_scale = source.get("sizeScale")
    if size_scale is None:
        size_scale = source.get("size_scale")
    size = source.get("size")
    if isinstance(size_scale, (int, float)) and size_scale > 0:
        normalized["size_scale"] = float(size_scale)
    elif isinstance(size, (int, float)) and size > 0 and base_size > 0:
        normalized["size_scale"] = size / base_size
    if source.get("color"):
        r, g, b, a = parse_color(source["color"])
        normalized["color"] = f"rgba({js_number(r)}, {js_number(g)}, {js_number(b)}, {js_number(a)})"
    family = source.get("family") or source.get("fontFamily")
    if family:
        normalized["family"] = family
    if "bold" in source:
        normalized["bold"] = bool(source["bold"])
    if "italic" in source:
        normalized["italic"] = bool(source["italic"])
    return normalized


def _resolve_runs(text, style: dict, base_size: float) -> list[dict]:
    """本文とスタイルから «ランの列» を作る。

    `style.runs` が本文と食い違っているときは捨てます。カウンター（数字が動く
    演出）のように本文だけ差し替わることがあり、そのまま色分けを当てると
    «別の文字列の色» を塗ってしまうためです。
    """
    plain = "" if text is None else str(text)
    raw_runs = style.get("runs")
    if isinstance(raw_runs, list) and raw_runs:
        runs = [r for r in (_normalize_run(run, base_size) for run in raw_runs) if r["text"]]
        joined = "".join(run["text"] for run in runs)
        if runs and (plain == "" or joined == plain):
            return runs
    if style.get("markup") and "<" in plain:
        parsed = [r for r in (_normalize_run(run, base_size) for run in parse_text_markup(plain)) if r["text"]]
        if parsed:
            return parsed
    return [{"text": plain}]


# ══════════════════════════════════════════════════════════════════
# 日本語組版 — 禁則処理・ルビ・枠に収める自動縮小
# ══════════════════════════════════════════════════════════════════

#: 行頭に来てはいけない文字（終わり括弧・句読点・繰り返し記号など）。
#: JIS X 4051 と一般的なワープロの «標準» 禁則に合わせています。
KINSOKU_LEADING_NORMAL = frozenset(
    "、。，．：；？！‼⁇⁈⁉"
    ")]}）〕］｝〉》」』】〙〗〟’”｠»"
    "ー〜～"
    "・:;?!"
    "゛゜ヽヾゝゞ々"
    "%‰℃℉°′″"
    "‐–—…‥,."
)

#: «強い» 禁則で足す文字。小書きの仮名を行頭に出さないぶん、
#: 1 行あたりの文字数が減って «追い出し» が起きやすくなります。
KINSOKU_LEADING_STRICT = KINSOKU_LEADING_NORMAL | frozenset(
    "ぁぃぅぇぉっゃゅょゎゕゖ"
    "ァィゥェォッャュョヮヵヶ"
    "ㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
)

#: 行末に来てはいけない文字（始め括弧・前置き記号）。
KINSOKU_TRAILING = frozenset("([{（〔［｛〈《「『【〘〖〝‘“｟«" "￥＄£＃$#")


def _kinsoku_table(mode):
    if not mode or mode == "off" or mode is False:
        return None
    return {
        "leading": KINSOKU_LEADING_STRICT if mode == "strict" else KINSOKU_LEADING_NORMAL,
        "trailing": KINSOKU_TRAILING,
    }


def _is_kanji(code: int) -> bool:
    """ルビの親文字にできる漢字か（`夜明[よあ]` の «夜明» を拾うのに使う）。"""
    return (0x3400 <= code <= 0x4DBF) or (0x4E00 <= code <= 0x9FFF) or (0xF900 <= code <= 0xFAFF)


def _is_word_char(code: int) -> bool:
    return (0x30 <= code <= 0x39) or (0x41 <= code <= 0x5A) or (0x61 <= code <= 0x7A)


def _parent_start(plain: str) -> int:
    """ルビの親文字の «始まり» を、直前の文字の種類から決める。"""
    if not plain:
        return 0
    last = ord(plain[-1])
    if _is_kanji(last):
        start = len(plain)
        while start > 0 and _is_kanji(ord(plain[start - 1])):
            start -= 1
        return start
    if _is_word_char(last):
        start = len(plain)
        while start > 0 and _is_word_char(ord(plain[start - 1])):
            start -= 1
        return start
    return len(plain) - 1


def split_ruby(text: str, enabled) -> list[dict]:
    """ルビ記法を «親文字＋ルビ» に分解する。

        夜明[よあ]けまで      … 直前の漢字の並びが親文字
        ｜1[いち]番            … `|` `｜` で親文字の範囲を明示する

    親文字が決まらないときは記法として扱わず、そのままの文字として残します。
    括弧を «文字として» 書きたい文章を壊さないためです。
    """
    if not enabled or not re.search(r"[\[［]", text):
        return [{"text": text}]
    segments: list[dict] = []
    plain = ""
    marker_at = -1
    index = 0

    def push_plain(value: str) -> None:
        if not value:
            return
        if segments and "ruby" not in segments[-1]:
            segments[-1]["text"] += value
        else:
            segments.append({"text": value})

    while index < len(text):
        char = text[index]
        if char in ("|", "｜"):
            marker_at = len(plain)
            index += 1
            continue
        if char in ("[", "［"):
            close = text.find("]" if char == "[" else "］", index + 1)
            if close > index + 1:
                ruby = text[index + 1 : close]
                start = marker_at if marker_at >= 0 else _parent_start(plain)
                if start < len(plain):
                    push_plain(plain[:start])
                    segments.append({"text": plain[start:], "ruby": ruby})
                    plain = ""
                    marker_at = -1
                    index = close + 1
                    continue
        plain += char
        index += 1
    push_plain(plain)
    return segments if segments else [{"text": text}]


def _resolve_fit_length(value, em, base_size: float, basis):
    """数値か「80%」かを解いて実寸（px）にする。基準が無い % は «指定なし» 扱い。"""
    if isinstance(em, (int, float)) and not isinstance(em, bool) and em > 0:
        return em * base_size
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    if isinstance(value, str):
        percent = re.match(r"^(-?[\d.]+)\s*%$", value.strip())
        if percent and basis and basis > 0:
            return (float(percent.group(1)) / 100) * basis
        number = _parse_float(value)
        if math.isfinite(number) and number > 0 and not percent:
            return number
    return math.inf


def _normalize_fit(style: dict, base_size: float):
    """`style.fit` を «収めたい寸法» に正規化する。

    `movo batch` では «人が見て直す» 工程が無いので、曲や歌詞を差し替えたとたんに
    文字が枠からはみ出すのを機械側で防ぐ必要があります。
    """
    fit = style.get("fit")
    if not isinstance(fit, dict):
        return None
    if fit.get("enabled") is False:
        return None
    mode = fit.get("mode", "shrink")
    if mode not in ("shrink", "wrap"):
        return None
    basis = None
    if isinstance(fit.get("basisEm"), (int, float)):
        basis = fit["basisEm"] * base_size
    elif isinstance(fit.get("basis"), (int, float)):
        basis = fit["basis"]
    elif isinstance(style.get("maxWidth"), (int, float)) and style["maxWidth"] > 0:
        basis = style["maxWidth"]
    width = _resolve_fit_length(fit.get("maxWidth"), fit.get("maxWidthEm"), base_size, basis)
    height = _resolve_fit_length(fit.get("maxHeight"), fit.get("maxHeightEm"), base_size, None)
    max_lines = fit["maxLines"] if isinstance(fit.get("maxLines"), (int, float)) and fit["maxLines"] > 0 else math.inf
    if width == math.inf and height == math.inf and max_lines == math.inf:
        return None
    return {
        "mode": mode,
        "width": width,
        "height": height,
        "max_lines": max_lines,
        "min_size": clamp(fit.get("minSize", 0.5), 0.05, 1),
    }


# ══════════════════════════════════════════════════════════════════
# 組版の本体
# ══════════════════════════════════════════════════════════════════


def layout_text(text, style: dict, font_manager) -> dict:
    """文字を測って 1 字ずつの置き場所を決める。

    :param style: JSON から来た «見た目の設定»（キーは camelCase のまま）
    """
    style = style or {}
    base_size = max(1, style.get("size", 48) or 48)
    base_letter_spacing = style.get("letterSpacing", 0) or 0
    runs = _resolve_runs(text, style, base_size)
    fit = _normalize_fit(style, base_size)

    max_width = style.get("maxWidth")
    wrap_width = max_width if isinstance(max_width, (int, float)) and max_width > 0 else math.inf
    # «折返す幅» と «収めたい幅» の両方があるときは狭いほうで折ります。
    # 折返し幅のまま縮めても行の幅は縮まらないので、いつまでも収まりません。
    if fit and (fit["mode"] == "wrap" or wrap_width < math.inf):
        wrap_width = min(wrap_width, fit["width"])

    layout = _layout_runs(runs, style, font_manager, base_size, base_letter_spacing, wrap_width)
    if not fit or fit["mode"] != "shrink":
        return layout

    # 収まるまで «少しずつ» ではなく «はみ出した比» で一気に縮めます。
    # 折返しがあると幅は線形に減らないので、2〜3 回で追い込みます。
    ratio = 1.0
    for _attempt in range(3):
        over = max(
            layout["width"] / fit["width"],
            layout["height"] / fit["height"],
            len(layout["lines"]) / fit["max_lines"],
        )
        if not over > 1.002:
            break
        if ratio <= fit["min_size"] + 1e-6:
            break
        ratio = clamp(ratio / over, fit["min_size"], 1)
        layout = _layout_runs(
            runs, style, font_manager, base_size * ratio, base_letter_spacing * ratio, wrap_width
        )
    layout["fit_scale"] = ratio
    return layout


def _layout_runs(runs, style, font_manager, size, letter_spacing, max_width) -> dict:
    """ラン → セル → 行 → グリフ、と 1 方向に組む本体。

    `layout_text` はここを «大きさを変えて» 何度か呼ぶことがあります（fit）。
    そのため副作用を持たせず、同じ引数からは必ず同じ結果が出るようにします。
    """
    primary = font_manager.resolve(style.get("family"), bold=style.get("bold"), italic=style.get("italic"))
    upem = primary.units_per_em
    ascender = primary.ascender if primary.ascender is not None else upem * 0.8
    descender = primary.descender if primary.descender is not None else upem * 0.2
    ascent = ascender * (size / upem)
    descent = abs(descender) * (size / upem)
    natural_line_height = ascent + descent + (primary.line_gap or 0) * (size / upem)
    line_height = (style.get("lineHeight", 1.2) if style.get("lineHeight") is not None else 1.2) * size
    vertical = style.get("direction") == "vertical"
    char_spacing = style.get("charSpacing", 1) if style.get("charSpacing") is not None else 1
    kinsoku = _kinsoku_table(style.get("kinsoku", "normal"))
    ruby_spec = style.get("ruby")
    ruby = None if (ruby_spec or {}).get("enabled") is False else ruby_spec
    ruby_ratio = clamp((ruby or {}).get("size", 0.45) or 0.45, 0.1, 1)
    if isinstance((ruby or {}).get("offsetEm"), (int, float)):
        ruby_offset = ruby["offsetEm"] * size
    else:
        ruby_offset = (ruby or {}).get("offset", 2) if ruby else 2
        if ruby_offset is None:
            ruby_offset = 2

    # 家族名を何度も解決すると毎回ファイルを探しに行くので、この組版のあいだ覚えておきます。
    faces: dict[tuple, object] = {}

    def face_for(family, bold, italic):
        key = (family or "", bool(bold), bool(italic))
        if key not in faces:
            faces[key] = font_manager.resolve(
                family if family is not None else style.get("family"), bold=bold, italic=italic
            )
        return faces[key]

    # 文字ごとに別のフォントを割り当てる（randomFont）。ここで解決しておくと、
    # 幅の計測と描画で同じフォントが使われる。
    random_font = style.get("randomFont")
    random_index = [0]

    def font_for(font, code):
        if not (random_font or {}).get("families"):
            return font_manager.font_for_code_point(font, code)
        family = random_font_for(random_index[0], random_font, style.get("time", 0) or 0)
        picked = (
            font_manager.resolve(family, bold=style.get("bold"), italic=style.get("italic")) if family else font
        )
        return font_manager.font_for_code_point(picked, code)

    # ── 1. セルに分解する ─────────────────────────────────────
    paragraphs: list[list[dict]] = [[]]
    cells = paragraphs[0]
    for run in runs:
        run_size = max(1, size * run.get("size_scale", 1))
        run_face = face_for(
            run.get("family"),
            run.get("bold", style.get("bold")),
            run.get("italic", style.get("italic")),
        )
        for segment in split_ruby(run["text"], ruby):
            group = (
                {
                    "text": segment["ruby"],
                    "size": run_size * ruby_ratio,
                    "face": run_face,
                    "color": run.get("color"),
                    "cells": [],
                }
                if segment.get("ruby")
                else None
            )
            for char in segment["text"]:
                if char == "\n":
                    cells = []
                    paragraphs.append(cells)
                    continue
                # CRLF の CR は改行の一部。字として組むと «見えない箱» が挟まります。
                if char == "\r":
                    continue
                code = ord(char)
                font = font_manager.font_for_code_point(run_face, code)
                advance = font.advance_width(font.glyph_index_for(code)) * (run_size / font.units_per_em)
                cell = {
                    "char": char,
                    "code": code,
                    "face": run_face,
                    "size": run_size,
                    "color": run.get("color"),
                    "advance": advance,
                    # 縦書きは «字の高さ» で送るので、折返しの物差しも送り量に合わせます。
                    "step": (run_size * char_spacing if vertical else advance) + letter_spacing,
                    "pad_before": 0.0,
                    "pad_after": 0.0,
                    "space": char in (" ", "\t"),
                    "cjk": is_cjk(code),
                    "group": group,
                }
                cells.append(cell)
                if group:
                    group["cells"].append(cell)
            if group and group["cells"]:
                _measure_ruby_group(group, font_manager, letter_spacing, vertical, char_spacing)

    # ── 2. 折返す（幅 + 禁則） ────────────────────────────────
    lines: list[list[dict]] = []
    for paragraph in paragraphs:
        if max_width == math.inf or not paragraph:
            lines.append(paragraph)
            continue
        start = 0
        width = 0.0
        last_break = -1
        for i, cell in enumerate(paragraph):
            if i > start and _can_break_before(paragraph, i, kinsoku):
                last_break = i
            if not cell["space"] and i > start and width + _cell_width(cell) > max_width and last_break > start:
                lines.append(_trim_trailing_spaces(paragraph[start:last_break]))
                start = last_break
                width = 0.0
                for k in range(start, i):
                    width += _cell_width(paragraph[k])
                last_break = -1
            width += _cell_width(cell)
        lines.append(_trim_trailing_spaces(paragraph[start:]))

    # ── 3. グリフを置く ──────────────────────────────────────
    laid_out: list[dict] = []
    for line_cells in lines:
        glyphs: list[dict] = []
        pen_x = 0.0
        line_ascent = 0.0
        line_descent = 0.0
        for cell in line_cells:
            pen_x += cell["pad_before"]
            font = font_for(cell["face"], cell["code"])
            random_index[0] += 1
            glyph_scale = cell["size"] / font.units_per_em
            glyph_index = font.glyph_index_for(cell["code"])
            advance = font.advance_width(glyph_index) * glyph_scale
            ratio = cell["size"] / size
            glyphs.append(
                {
                    "font": font,
                    "glyph_index": glyph_index,
                    "x": pen_x,
                    "advance": advance,
                    "scale": glyph_scale,
                    "char": cell["char"],
                    "size": cell["size"],
                    "color": cell["color"],
                    "group": cell["group"],
                    "pad_before": cell["pad_before"],
                    "pad_after": cell["pad_after"],
                }
            )
            pen_x += advance + letter_spacing + cell["pad_after"]
            line_ascent = max(
                line_ascent,
                ascent * ratio + (ruby_offset + ascent * ratio * ruby_ratio if cell["group"] else 0),
            )
            line_descent = max(line_descent, descent * ratio)
        width = max(0.0, pen_x - (letter_spacing if glyphs else 0))
        laid_out.append(
            {
                "text": "".join(cell["char"] for cell in line_cells),
                "glyphs": glyphs,
                "width": width,
                "ascent": line_ascent if glyphs else ascent,
                "descent": line_descent if glyphs else descent,
            }
        )

    max_ascent = ascent
    max_descent = descent
    for line in laid_out:
        max_ascent = max(max_ascent, line["ascent"])
        max_descent = max(max_descent, line["descent"])

    if vertical:
        return _layout_vertical(
            laid_out,
            style=style,
            size=size,
            letter_spacing=letter_spacing,
            char_spacing=char_spacing,
            ascent=max_ascent,
            descent=max_descent,
            line_height=line_height,
            natural_line_height=natural_line_height,
            primary=primary,
            ruby_offset=ruby_offset,
        )

    # 横書きのルビは «親文字の上» に、親の並びの中心へ寄せて置きます。
    for line in laid_out:
        ruby_glyphs: list[dict] = []
        for group in _collect_groups(line):
            _place_ruby_horizontal(group, line, ruby_glyphs, ascent, size, ruby_offset)
        if ruby_glyphs:
            line["ruby_glyphs"] = ruby_glyphs

    width = max((line["width"] for line in laid_out), default=0.0)
    height = line_height * (len(laid_out) - 1) + max_ascent + max_descent if laid_out else 0.0

    return {
        "lines": laid_out,
        "width": width,
        "height": height,
        "ascent": max_ascent,
        "descent": max_descent,
        "line_height": line_height,
        "natural_line_height": natural_line_height,
        "size": size,
        "font": primary,
        "vertical": False,
    }


def _cell_width(cell: dict) -> float:
    """折返しの物差し。ルビのはみ出し分（pad_before / pad_after）も込みで測ります。"""
    return cell["step"] + cell["pad_before"] + cell["pad_after"]


def _trim_trailing_spaces(cells: list[dict]) -> list[dict]:
    end = len(cells)
    while end > 0 and cells[end - 1]["space"]:
        end -= 1
    return cells if end == len(cells) else cells[:end]


def _can_break_before(cells: list[dict], index: int, kinsoku) -> bool:
    """ここで折ってよいか。

    «ラテン語の途中では折らない»（従来どおり）に加えて、禁則の 2 条件
    （行頭に来てはいけない字・行末に来てはいけない字）を見ます。前に戻って
    折り直す «追い出し» は、条件を満たす位置まで `last_break` が下がることで
    自然に起こります。
    """
    cell = cells[index]
    prev = cells[index - 1] if index >= 1 else None
    if prev is None or cell["space"]:
        return False
    # ルビの親文字は途中で切らない（ルビだけ次の行に残ってしまうため）
    if cell["group"] and prev["group"] is cell["group"]:
        return False
    if not (prev["space"] or prev["cjk"] or cell["cjk"]):
        return False
    if kinsoku:
        if cell["char"] in kinsoku["leading"]:
            return False
        if prev["char"] in kinsoku["trailing"]:
            return False
    return True


def _measure_ruby_group(group, font_manager, letter_spacing, vertical, char_spacing) -> None:
    """ルビ 1 組の寸法を測り、親文字より長ければ親文字の前後に余白を足す。

    測り終えたら `group["cells"]` は捨てます。セル → 組 → セルの循環参照が
    残ると、レイアウト結果を JSON にしようとしたところで落ちるためです。
    """
    glyphs = []
    length = 0.0
    for char in group["text"]:
        code = ord(char)
        font = font_manager.font_for_code_point(group["face"], code)
        glyph_index = font.glyph_index_for(code)
        scale = group["size"] / font.units_per_em
        advance = font.advance_width(glyph_index) * scale
        glyphs.append(
            {
                "font": font,
                "glyph_index": glyph_index,
                "scale": scale,
                "advance": advance,
                "char": char,
                "size": group["size"],
                "color": group.get("color"),
            }
        )
        length += group["size"] * char_spacing if vertical else advance
    group["glyphs"] = glyphs
    group["length"] = length
    parent = sum(cell["step"] for cell in group["cells"]) - letter_spacing
    extra = length - parent
    if extra > 0:
        # 親文字よりルビが長いときは «親文字を広げる»。隣の語のルビと
        # ぶつからせないための、日本語組版では普通のやり方です。
        group["cells"][0]["pad_before"] += extra / 2
        group["cells"][-1]["pad_after"] += extra / 2
    group["cells"] = None


def _index_of(items: list, target) -> int:
    """`list.index` の代わり。**同じ «もの» かどうか**（`is`）で探します。

    グリフは `dict` なので `==` だと «中身が同じ別のグリフ» に当たり得ます。
    親文字の位置を取り違えるとルビが別の字の上に乗ってしまいます。
    """
    for index, item in enumerate(items):
        if item is target:
            return index
    return -1


def _collect_groups(line: dict) -> list:
    """行の中に現れるルビの組を、出てきた順に集める。"""
    groups = []
    for glyph in line["glyphs"]:
        group = glyph.get("group")
        if group is not None and not any(group is existing for existing in groups):
            groups.append(group)
    return groups


def _place_ruby_horizontal(group, line, out, ascent, size, ruby_offset) -> None:
    """横書きのルビ：親文字の上に、中心をそろえて置く。"""
    own = [glyph for glyph in line["glyphs"] if glyph.get("group") is group]
    if not own or not group.get("glyphs"):
        return
    parent_index = _index_of(line["glyphs"], own[0])
    start = own[0]["x"]
    end = own[-1]["x"] + own[-1]["advance"]
    ratio = own[0]["size"] / size
    pen_x = (start + end - group["length"]) / 2
    dy = -(ascent * ratio + ruby_offset)
    for glyph in group["glyphs"]:
        placed = dict(glyph)
        placed.update({"x": pen_x, "dy": dy, "parent_index": parent_index, "ruby": True})
        out.append(placed)
        pen_x += glyph["advance"]


def _layout_vertical(
    laid_out, *, style, size, letter_spacing, char_spacing, ascent, descent,
    line_height, natural_line_height, primary, ruby_offset,
) -> dict:
    """縦書き：1 行が 1 列になり、列は右から左へ並ぶ（日本語の組み方）。

    ルビは列の «右側» に添えるので、その幅だけ列を広く取ります。
    """
    has_ruby = any(glyph.get("group") for line in laid_out for glyph in line["glyphs"])
    if has_ruby:
        ruby_sizes = [
            glyph["group"]["size"] for line in laid_out for glyph in line["glyphs"] if glyph.get("group")
        ]
        ruby_slot = max(ruby_sizes) + ruby_offset
    else:
        ruby_slot = 0.0
    body_width = max(
        (max([size] + [glyph["size"] for glyph in line["glyphs"]]) for line in laid_out), default=size
    ) * (style.get("lineHeight", 1.2) if style.get("lineHeight") is not None else 1.2)
    column_width = body_width + ruby_slot
    total_width = column_width * len(laid_out)
    height = 0.0

    for column_index, line in enumerate(laid_out):
        column_left = total_width - (column_index + 1) * column_width
        pen_y = ascent
        for glyph in line["glyphs"]:
            # ルビのはみ出し分（pad_before / pad_after）は縦書きでは «送り» に効きます
            pen_y += glyph.get("pad_before") or 0
            glyph["x"] = column_left + (body_width - glyph["advance"]) / 2
            glyph["baseline"] = pen_y
            pen_y += glyph["size"] * char_spacing + letter_spacing + (glyph.get("pad_after") or 0)
        line["width"] = column_width
        last = line["glyphs"][-1] if line["glyphs"] else None
        height = max(height, last["baseline"] + descent if last else ascent + descent)

        if not has_ruby:
            continue
        ruby_glyphs: list[dict] = []
        for group in _collect_groups(line):
            own = [glyph for glyph in line["glyphs"] if glyph.get("group") is group]
            if not own or not group.get("glyphs"):
                continue
            parent_index = _index_of(line["glyphs"], own[0])
            top = own[0]["baseline"] - ascent
            bottom = own[-1]["baseline"] + descent
            pen_ruby = (top + bottom - group["length"]) / 2
            for glyph in group["glyphs"]:
                placed = dict(glyph)
                placed.update(
                    {
                        "x": column_left + body_width + (ruby_slot - ruby_offset - glyph["advance"]) / 2 + ruby_offset,
                        "baseline": pen_ruby + group["size"] * 0.8,
                        "parent_index": parent_index,
                        "ruby": True,
                    }
                )
                ruby_glyphs.append(placed)
                pen_ruby += group["size"]
        if ruby_glyphs:
            line["ruby_glyphs"] = ruby_glyphs

    return {
        "lines": laid_out,
        "width": total_width,
        "height": height if laid_out else 0.0,
        "ascent": ascent,
        "descent": descent,
        "line_height": line_height,
        "natural_line_height": natural_line_height,
        "size": size,
        "font": primary,
        "vertical": True,
    }


def for_each_placed_glyph(layout: dict, align: str, offset_x: float, offset_y: float, fn: Callable) -> None:
    """組み上がったレイアウトの «置き場所が決まったグリフ» を順に渡す。

    通常の描画・アニメーション・カラオケ塗りで «同じ位置» を使うために、
    行そろえとベースラインの計算はここだけに置いています。
    """
    for index, line in enumerate(layout["lines"]):
        origin_x = offset_x
        if not layout["vertical"]:
            if align == "center":
                origin_x += (layout["width"] - line["width"]) / 2
            elif align == "right":
                origin_x += layout["width"] - line["width"]
        baseline_y = offset_y + layout["ascent"] + index * layout["line_height"]

        def emit(glyph):
            gx = offset_x + glyph["x"] if layout["vertical"] else origin_x + glyph["x"]
            gy = offset_y + glyph["baseline"] if layout["vertical"] else baseline_y + (glyph.get("dy") or 0)
            fn(glyph, gx, gy, index)

        for glyph in line["glyphs"]:
            emit(glyph)
        for glyph in line.get("ruby_glyphs") or []:
            emit(glyph)


class _Buckets:
    """塗り色ごとに輪郭をまとめる入れ物。

    1 レイヤー 1 色だったころは «全部まとめて 1 回» で塗れましたが、ランで色が
    変わるようになったので色ごとに塗り分けます。**色が 1 つのときは今までと
    同じ «1 回の塗り»** になるようにしてあります（見た目を変えないため）。
    """

    __slots__ = ("_buckets", "_keys", "_default")

    def __init__(self, default_color) -> None:
        self._buckets: dict[str, dict] = {}
        self._keys: dict = {}
        self._default = default_color

    def _key_for(self, color) -> str:
        raw = color if color is not None else (self._default if self._default is not None else "#ffffff")
        if raw not in self._keys:
            r, g, b, a = parse_color(raw)
            self._keys[raw] = f"{r},{g},{b},{a}"
        return self._keys[raw]

    def add(self, color, contours) -> None:
        key = self._key_for(color)
        bucket = self._buckets.get(key)
        if bucket:
            bucket["contours"].extend(contours)
        else:
            self._buckets[key] = {
                "color": color if color is not None else (self._default if self._default is not None else "#ffffff"),
                "contours": list(contours),
            }

    def list(self) -> list[dict]:
        return list(self._buckets.values())


# ══════════════════════════════════════════════════════════════════
# 描画
# ══════════════════════════════════════════════════════════════════


def render_text(text, style: dict, font_manager) -> dict:
    """文字を専用のビットマップへ描く。

    :returns: `{"bitmap", "width", "height", "offset_x", "offset_y", "layout"}`
    """
    style = style or {}
    layout = layout_text(text, style, font_manager)
    stroke = style.get("stroke") or {}
    stroke_width = stroke.get("width", 0) or 0
    shadow = style.get("shadow")
    shadow_blur = (shadow or {}).get("blur", 0) or 0
    shadow_offset_x = (shadow or {}).get("offsetX", 0) or 0
    shadow_offset_y = (shadow or {}).get("offsetY", 0) or 0

    pad_left = math.ceil(stroke_width + max(0, -shadow_offset_x + shadow_blur) + 2)
    pad_top = math.ceil(stroke_width + max(0, -shadow_offset_y + shadow_blur) + 2)
    pad_right = math.ceil(stroke_width + max(0, shadow_offset_x + shadow_blur) + 2)
    pad_bottom = math.ceil(stroke_width + max(0, shadow_offset_y + shadow_blur) + 2)

    width = max(1, math.ceil(layout["width"]) + pad_left + pad_right)
    height = max(1, math.ceil(layout["height"]) + pad_top + pad_bottom)
    bitmap = Bitmap(width, height)

    align = style.get("align", "left")
    contours: list[list[float]] = []
    buckets = _Buckets(style.get("color"))

    def place(glyph, gx, gy, _line_index):
        if not glyph["glyph_index"] and glyph["char"].strip() == "":
            return
        outline = glyph["font"].glyph(glyph["glyph_index"])
        if not outline.contours:
            return
        built = glyph_contours(outline, glyph["scale"], gx, gy)
        contours.extend(built)
        buckets.add(glyph.get("color"), built)

    for_each_placed_glyph(layout, align, pad_left, pad_top, place)

    result = {
        "bitmap": bitmap,
        "width": width,
        "height": height,
        "offset_x": pad_left,
        "offset_y": pad_top,
        "layout": layout,
    }
    if not contours:
        return result

    # 書き順アニメーション。データが無いときは «左上から右下へ» の近似。
    painted = buckets.list()
    if style.get("strokeOrder"):
        # 色ごとに «書けたところまで» を切り出します。色が 1 つなら今までと同じです。
        painted = [
            {
                "color": bucket["color"],
                "contours": apply_stroke_order(bucket["contours"], style["strokeOrder"], style.get("time", 0) or 0),
            }
            for bucket in painted
        ]
        painted = [bucket for bucket in painted if bucket["contours"]]
        contours = []
        for bucket in painted:
            contours.extend(bucket["contours"])
        if not contours:
            return result

    # 影 → 縁取り → 塗り の順に描く
    if shadow:
        shadow_layer = Bitmap(width, height)
        shifted = []
        for contour in contours:
            copy = list(contour)
            for i in range(0, len(contour), 2):
                copy[i] = contour[i] + shadow_offset_x
                copy[i + 1] = contour[i + 1] + shadow_offset_y
            shifted.append(copy)
        region = rasterize_contours(shifted, width, height)
        fill_coverage(shadow_layer, region, shadow.get("color", "rgba(0,0,0,0.5)"), 1)
        blurred = box_blur(shadow_layer, shadow_blur, 2) if shadow_blur > 0 else shadow_layer
        draw_bitmap(bitmap, blurred, 0, 0, 1)

    if stroke_width > 0:
        stroke_contours: list = []
        for contour in contours:
            stroke_contours.extend(stroke_to_contours(contour, stroke_width, True))
        fill_coverage(
            bitmap, _coverage_for(stroke_contours, width, height, style), stroke.get("color", "#000000"), 1
        )

    for bucket in painted:
        region = _coverage_for(bucket["contours"], width, height, style, "nonzero")
        fill_coverage(bitmap, region, bucket["color"] if bucket["color"] is not None else "#ffffff", 1)

    return result


def render_animated_text(text, style: dict, font_manager, animator: dict, time: float) -> dict:
    """1 文字（語・行）ずつ現れる文字アニメーション。

    文字・語・行が少しずつずれて現れ、それぞれが `from`（位置・拡大・回転・不透明度）
    から «落ち着く姿» へ補間されます。タイプライター・ポップイン・ウェーブインが
    JSON 数行で書けるのはこれのおかげです。
    """
    style = style or {}
    layout = layout_text(text, style, font_manager)
    unit = animator.get("unit", "character")
    stagger = animator.get("stagger", 0.04)
    duration = max(1e-4, animator.get("duration", 0.5))
    delay = animator.get("delay", 0) or 0
    easing = _get_easing(animator.get("easing", "easeOutCubic"))
    from_spec = animator.get("from") or {"opacity": 0, "y": 30, "scale": 0.8}
    from_opacity = from_spec.get("opacity", 0) or 0
    from_x = from_spec.get("x", 0) or 0
    from_y = from_spec.get("y", 0) or 0
    from_scale = from_spec.get("scale", 1) if from_spec.get("scale") is not None else 1
    from_rotation = (from_spec.get("rotation", 0) or 0) * math.pi / 180

    units = group_units(layout, unit)
    order = order_indices(len(units), animator.get("order", "forward"), animator.get("seed", 12345))

    travel = max(abs(from_x), abs(from_y), layout["size"] * abs(1 - from_scale) + layout["size"] * 0.4)
    stroke = style.get("stroke") or {}
    stroke_width = stroke.get("width", 0) or 0
    pad = math.ceil(travel + stroke_width + 4)
    width = max(1, math.ceil(layout["width"]) + pad * 2)
    height = max(1, math.ceil(layout["height"]) + pad * 2)
    bitmap = Bitmap(width, height)
    align = style.get("align", "left")
    seed = animator.get("seed", 12345)

    for index, group in enumerate(units):
        step = order[index]
        local = time - delay - step * stagger
        progress = clamp(local / duration, 0, 1)
        if animator.get("loop") and animator.get("loopDuration"):
            loop_duration = animator["loopDuration"]
            cycle = ((time - delay) % loop_duration + loop_duration) % loop_duration
            progress = clamp((cycle - step * stagger) / duration, 0, 1)
        eased = easing(progress)
        opacity = from_opacity + (1 - from_opacity) * eased
        if opacity <= 0.002:
            continue
        # from.random があると文字ごとに散らばった状態から集まってくる（分解・集合）。
        spread = from_spec.get("random")
        if spread:
            jitter_x = (hash_unit(index, seed, 1) * 2 - 1) * (spread.get("x", 0) or 0)
            jitter_y = (hash_unit(index, seed, 2) * 2 - 1) * (spread.get("y", 0) or 0)
            jitter_rotation = (hash_unit(index, seed, 3) * 2 - 1) * (spread.get("rotation", 0) or 0)
            jitter_scale = spread.get("scale", 1) if spread.get("scale") is not None else 1
        else:
            jitter_x = jitter_y = jitter_rotation = 0.0
            jitter_scale = None

        offset_x = (from_x + jitter_x) * (1 - eased)
        offset_y = (from_y + jitter_y) * (1 - eased)
        base_scale = from_scale * jitter_scale if jitter_scale is not None else from_scale
        scale = base_scale + (1 - base_scale) * eased
        rotation = (from_rotation + (jitter_rotation * math.pi / 180 if spread else 0)) * (1 - eased)

        contours: list[list[float]] = []
        buckets = _Buckets(style.get("color"))
        min_x = math.inf
        max_x = -math.inf
        baseline = 0.0
        for item in group:
            glyph = item["glyph"]
            line_index = item["line_index"]
            line = layout["lines"][line_index]
            origin_x = pad
            if not layout["vertical"]:
                if align == "center":
                    origin_x += (layout["width"] - line["width"]) / 2
                elif align == "right":
                    origin_x += layout["width"] - line["width"]
            glyph_baseline = (
                pad + glyph["baseline"]
                if layout["vertical"]
                else pad + layout["ascent"] + line_index * layout["line_height"] + (glyph.get("dy") or 0)
            )
            # 回転・拡大の中心は «親文字» で決める（ルビだけ違う所を回らないように）
            if not glyph.get("ruby"):
                baseline = glyph_baseline
            elif baseline == 0:
                baseline = glyph_baseline
            if not glyph["glyph_index"] and glyph["char"].strip() == "":
                continue
            outline = glyph["font"].glyph(glyph["glyph_index"])
            if not outline.contours:
                continue
            gx = pad + glyph["x"] if layout["vertical"] else origin_x + glyph["x"]
            built = glyph_contours(outline, glyph["scale"], gx, glyph_baseline)
            for contour in built:
                for i in range(0, len(contour), 2):
                    if contour[i] < min_x:
                        min_x = contour[i]
                    if contour[i] > max_x:
                        max_x = contour[i]
                contours.append(contour)
            buckets.add(glyph.get("color"), built)
        if not contours:
            continue

        # 単位ごとに «自分の中心» で回して拡大する。そうしないと歪んで見えます。
        cx = (min_x + max_x) / 2
        cy = baseline - layout["size"] * 0.35
        cos = math.cos(rotation)
        sin = math.sin(rotation)
        for contour in contours:
            for i in range(0, len(contour), 2):
                dx = (contour[i] - cx) * scale
                dy = (contour[i + 1] - cy) * scale
                contour[i] = cx + dx * cos - dy * sin + offset_x
                contour[i + 1] = cy + dx * sin + dy * cos + offset_y

        if stroke_width > 0:
            stroke_contours: list = []
            for contour in contours:
                stroke_contours.extend(stroke_to_contours(contour, stroke_width, True))
            fill_coverage(
                bitmap,
                _coverage_for(stroke_contours, width, height, style),
                stroke.get("color", "#000000"),
                opacity,
            )
        for bucket in buckets.list():
            fill_coverage(
                bitmap,
                _coverage_for(bucket["contours"], width, height, style, "nonzero"),
                bucket["color"] if bucket["color"] is not None else "#ffffff",
                opacity,
            )

    return {
        "bitmap": bitmap,
        "width": width,
        "height": height,
        "offset_x": pad,
        "offset_y": pad,
        "layout": layout,
        "units": len(units),
    }


def group_units(layout: dict, unit: str) -> list[list[dict]]:
    """アニメーションの «ひとかたまり» に分ける（文字・語・行）。"""
    groups: list[list[dict]] = []
    for line_index, line in enumerate(layout["lines"]):
        # ルビは親文字と «同じ組» で動かす。別々に動くと振り仮名だけ取り残されます。
        ruby_at: dict[int, list[dict]] = {}
        for glyph in line.get("ruby_glyphs") or []:
            ruby_at.setdefault(glyph.get("parent_index"), []).append({"glyph": glyph, "line_index": line_index})

        def with_ruby(glyph, index):
            return [{"glyph": glyph, "line_index": line_index}] + ruby_at.get(index, [])

        if unit == "line":
            merged: list[dict] = []
            for index, glyph in enumerate(line["glyphs"]):
                merged.extend(with_ruby(glyph, index))
            groups.append(merged)
            continue
        if unit == "word":
            current: list[dict] = []
            for index, glyph in enumerate(line["glyphs"]):
                if glyph["char"] == " ":
                    if current:
                        groups.append(current)
                    current = []
                    continue
                current.extend(with_ruby(glyph, index))
            if current:
                groups.append(current)
            continue
        for index, glyph in enumerate(line["glyphs"]):
            if glyph["char"] == " ":
                continue
            groups.append(with_ruby(glyph, index))
    return groups


def _to_uint32(x) -> int:
    return int(x) & 0xFFFFFFFF


def _to_int32(x) -> int:
    v = int(x) & 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _imul(a, b) -> int:
    return _to_int32(_to_uint32(a) * _to_uint32(b))


def hash_unit(index: int, seed: int, salt: int) -> float:
    """決定的な 0..1 の疑似乱数（文字ごとのばらつきに使う）。

    JS の 32 ビット演算をそのまま真似ています。**同じ JSON から同じ動画が出る**
    ことを JS 版と共有するためです。
    """
    h = _to_uint32(_imul(index * 2654435761 + seed + salt * 40503, 0x27D4EB2D))
    h = _to_int32(_to_int32(h) ^ (_to_uint32(h) >> 15))
    h = _to_uint32(_imul(h, 0x85EBCA6B))
    h = _to_int32(_to_int32(h) ^ (_to_uint32(h) >> 13))
    return _to_uint32(h) / 4294967296


def order_indices(count: int, order: str, seed: int) -> list[int]:
    """現れる順番を作る（forward / reverse / center / random）。"""
    indices = list(range(count))
    if order == "reverse":
        return [count - 1 - i for i in indices]
    if order == "center":
        middle = (count - 1) / 2
        return [round(abs(i - middle)) for i in indices]
    if order == "random":
        # 決定的なシャッフル。同じ seed からは必ず同じ並びになります。
        shuffled = list(indices)
        state = _to_uint32(seed) or 1
        for i in range(len(shuffled) - 1, 0, -1):
            state = _to_uint32(_imul(state, 1664525) + 1013904223)
            j = state % (i + 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        return shuffled
    return indices


# ══════════════════════════════════════════════════════════════════
# カラオケ塗り・カウンター・スタイルの正規化
# ══════════════════════════════════════════════════════════════════


def apply_karaoke_fill(bitmap, karaoke: dict, box: dict):
    """左から順に «塗り色» へ入れ替えていく（カラオケ）。

    :param box: `offset_x`（または `offsetX`）と `width`、
        `base_colors`（または `baseColors` / `baseColor`）
    """
    progress = clamp(karaoke.get("progress", 0) or 0, 0, 1)
    if progress <= 0:
        return bitmap
    fill = parse_color(karaoke.get("color", "#ffd166"))
    # 塗り替えてよい «地の色» は複数あり得ます（リッチテキストで色を変えた語）。
    raw_bases = box.get("base_colors") or box.get("baseColors")
    if not raw_bases:
        raw_bases = [box.get("base_color") or box.get("baseColor") or "#ffffff"]
    bases = [parse_color(color) for color in raw_bases]
    offset_x = box.get("offset_x", box.get("offsetX", 0)) or 0
    box_width = box.get("width", 0) or 0
    softness = max(0.0, (karaoke.get("softness", 0.02) if karaoke.get("softness") is not None else 0.02) * box_width)
    edge = offset_x + box_width * progress
    out = bitmap.copy()
    base_r = np.array([b[0] for b in bases], np.float64)
    base_g = np.array([b[1] for b in bases], np.float64)
    base_b = np.array([b[2] for b in bases], np.float64)
    kernels.karaoke_fill_kernel(
        bitmap.data, out.data, base_r, base_g, base_b,
        float(fill[0]), float(fill[1]), float(fill[2]), float(edge), float(softness),
    )
    return out


def _to_fixed(value: float, decimals: int) -> str:
    """JS の `Number.prototype.toFixed`（0.5 は **必ず上へ**）。

    Python の `format` は偶数丸めなので、`0.125` を 2 桁にすると `0.12` に
    なります。JS は `0.13` です。歌詞に数字を出す演出でずれると気付きにくいので、
    ここで合わせます。
    """
    negative = value < 0
    quantum = Decimal(1).scaleb(-decimals)
    # `Decimal(float)` は **2 進数の正確な値**から作られます。`repr` 経由にすると
    # `1.005` が本当は 1.00499999… であることを見失い、JS と答えが変わります。
    d = Decimal(abs(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    text = f"{d:.{decimals}f}"
    # `-0` になるのも JS の `toFixed` どおりです（先に符号を外して丸めるため）
    return ("-" + text) if negative else text


def format_counter(counter: dict, progress: float) -> str:
    """カウンターの表示文字列を組み立てる。"""
    from_value = counter.get("from", 0) or 0
    to_value = counter.get("to", 100) if counter.get("to") is not None else 100
    value = from_value + (to_value - from_value) * clamp(progress, 0, 1)
    decimals = max(0, round(counter.get("decimals", 0) or 0))
    text = _to_fixed(value, decimals)
    if counter.get("separator"):
        parts = text.split(".")
        integer = parts[0]
        frac = parts[1] if len(parts) > 1 else None
        integer = re.sub(r"\B(?=(\d{3})+(?!\d))", ",", integer)
        text = integer + (f".{frac}" if frac else "")
    pad = max(0, round(counter.get("pad", 0) or 0))
    if pad > 0:
        negative = text.startswith("-")
        body = text[1:] if negative else text
        text = ("-" if negative else "") + body.rjust(pad, "0")
    return f"{counter.get('prefix', '') or ''}{text}{counter.get('suffix', '') or ''}"


def resolve_text_style(layer: dict, resolved: dict) -> dict:
    """文字レイヤーの «いろいろな書き方» を 1 つの style にそろえる。

    :returns: `{"content": str, "style": dict}`。style のキーは **camelCase のまま**
    """
    source = resolved.get("text") if isinstance(resolved.get("text"), dict) else {}
    if isinstance(resolved.get("text"), str):
        content = resolved["text"]
    else:
        content = source.get("content") or source.get("value") or ""
    style = {}
    style.update(resolved.get("style") or {})
    style.update(resolved.get("font") or {})
    style.update(source)
    color = parse_color(style.get("color") or style.get("fill") or "#ffffff")
    size = style.get("size") if style.get("size") is not None else style.get("fontSize", 48)
    if size is None:
        size = 48

    # ランはここで «倍率» に直しておきます。レンダラーは高品質出力で `style.size`
    # だけを倍にするので、実寸のまま持ち回ると比が崩れます。
    runs = None
    raw_runs = style.get("runs")
    if isinstance(raw_runs, list) and raw_runs:
        runs = [r for r in (_normalize_run(run, size) for run in raw_runs) if r["text"]]
        if not content:
            content = "".join(run["text"] for run in runs)
    elif style.get("markup") and isinstance(content, str) and "<" in content:
        runs = [r for r in (_normalize_run(run, size) for run in parse_text_markup(content)) if r["text"]]
        content = "".join(run["text"] for run in runs)

    weight = str(style.get("weight", "") or "")
    return {
        "content": str(content or ""),
        "style": {
            "family": style.get("family") or style.get("fontFamily") or layer.get("fontFamily"),
            "size": size,
            "bold": style["bold"] if "bold" in style else bool(re.search(r"bold|700|800|900", weight)),
            "italic": style["italic"] if "italic" in style else style.get("style") == "italic",
            "color": f"rgba({js_number(color[0])}, {js_number(color[1])}, {js_number(color[2])}, {js_number(color[3])})",
            "align": style.get("align", "left"),
            "verticalAlign": style.get("verticalAlign", "top"),
            "lineHeight": style.get("lineHeight", 1.2) if style.get("lineHeight") is not None else 1.2,
            "letterSpacing": style.get("letterSpacing", 0) or 0,
            # ⚠ **`width` は «そのまま» `maxWidth` に流します。**
            # 片方だけ書いても比率は補われません（JS 版と同じ挙動を守るため）。
            "maxWidth": style.get("maxWidth") if style.get("maxWidth") is not None else style.get("width"),
            "direction": style.get("direction", "horizontal"),
            "charSpacing": style.get("charSpacing", 1) if style.get("charSpacing") is not None else 1,
            "stroke": style.get("stroke"),
            "shadow": style.get("shadow"),
            # ドット絵風。既定は «今まで通り»（アンチエイリアスあり・量子化なし）。
            "antialias": style.get("antialias"),
            "pixelGrid": style.get("pixelGrid"),
            # 1 行の中で色・大きさ・書体を変える
            "runs": runs,
            # 日本語組版。既定は «標準» の禁則で、`off` で従来どおりになります。
            "kinsoku": style.get("kinsoku", "normal"),
            "fit": _normalize_fit_spec(style.get("fit"), style, size, layer),
            "ruby": _normalize_ruby_spec(style.get("ruby"), size),
        },
    }


def _normalize_fit_spec(fit, style: dict, size, layer: dict):
    """`fit` の «px で書かれた寸法» を基準サイズとの比に直す。

    高品質出力（superSample）では `style.size` だけが倍になるので、px のまま
    渡すと «縮める目標» だけが元の大きさに取り残されてしまいます。
    `80%` の基準は `maxWidth`（無ければ `transform.width` か `fit.basis`）です。
    """
    if not isinstance(fit, dict):
        return None
    out = dict(fit)
    basis = None
    for candidate in (
        fit.get("basis"),
        style.get("maxWidth"),
        style.get("width"),
        ((layer or {}).get("transform") or {}).get("width"),
    ):
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            basis = candidate
            break
    if basis and size > 0:
        out["basisEm"] = basis / size
    if isinstance(fit.get("maxWidth"), (int, float)) and size > 0:
        out["maxWidthEm"] = fit["maxWidth"] / size
    if isinstance(fit.get("maxHeight"), (int, float)) and size > 0:
        out["maxHeightEm"] = fit["maxHeight"] / size
    return out


def _normalize_ruby_spec(ruby, size):
    """ルビの `offset`（px）も基準サイズとの比に直す（fit と同じ理由）。"""
    if not isinstance(ruby, dict):
        return None
    out = dict(ruby)
    if isinstance(ruby.get("offset"), (int, float)) and size > 0:
        out["offsetEm"] = ruby["offset"] / size
    return out


def render_text_on_path(text, style: dict, font_manager, spec: dict):
    """パス（円周・弧・折れ線）に沿って文字を描く。

    `layout_text` で送り量を出し、それを弧長としてパス上に置き直します。
    `firstMargin` を動かすと文字がパスの上を流れます。
    """
    style = style or {}
    layout = layout_text(text, style, font_manager)

    def contours_for(glyph, _index):
        outline = glyph["font"].glyph(glyph["glyph_index"])
        if not outline.contours:
            return None
        # 原点にベースライン、字の左端が x=0 になるように出す
        return glyph_contours(outline, glyph["scale"], 0, 0)

    placed = layout_text_on_path(layout, spec, contours_for)
    if not placed:
        return None

    stroke = style.get("stroke") or {}
    stroke_width = stroke.get("width", 0) or 0
    pad = math.ceil(stroke_width + 2)
    width = max(1, math.ceil(placed["max_x"] - placed["min_x"]) + pad * 2)
    height = max(1, math.ceil(placed["max_y"] - placed["min_y"]) + pad * 2)
    bitmap = Bitmap(width, height)

    # 原点を左上へ寄せる
    shifted = []
    for contour in placed["contours"]:
        copy = list(contour)
        for i in range(0, len(contour), 2):
            copy[i] = contour[i] - placed["min_x"] + pad
            copy[i + 1] = contour[i + 1] - placed["min_y"] + pad
        shifted.append(copy)

    if stroke_width > 0:
        stroke_contours: list = []
        for contour in shifted:
            stroke_contours.extend(stroke_to_contours(contour, stroke_width, True))
        fill_coverage(
            bitmap, _coverage_for(stroke_contours, width, height, style), stroke.get("color", "#000000"), 1
        )
    region = _coverage_for(shifted, width, height, style, "nonzero")
    fill_coverage(bitmap, region, style.get("color", "#ffffff"), 1)

    return {
        "bitmap": bitmap,
        "width": width,
        "height": height,
        "offset_x": pad,
        "offset_y": pad,
        "layout": layout,
    }


__all__ = [
    "KINSOKU_LEADING_NORMAL",
    "KINSOKU_LEADING_STRICT",
    "KINSOKU_TRAILING",
    "apply_karaoke_fill",
    "apply_stroke_order",
    "box_blur",
    "draw_text_box",
    "for_each_placed_glyph",
    "format_counter",
    "glyph_contours",
    "group_units",
    "hash_unit",
    "layout_text",
    "order_indices",
    "parse_text_markup",
    "quantize_coverage",
    "random_font_for",
    "render_animated_text",
    "render_text",
    "render_text_on_path",
    "resolve_text_style",
    "split_ruby",
]
