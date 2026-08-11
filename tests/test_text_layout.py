"""組版と文字の描画のテスト。

**JS 版と同じ組み方・同じ絵になること**が目的です。システムのフォントを使うので、
フォントが見つからない環境では skip します（他の環境でも落ちないように）。
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.renderer import text as T
from movo.renderer import text_extras as TX
from movo.renderer.font import FontManager, list_font_files

# ── フォントの用意 ──────────────────────────────────────────
#
# 字形そのものを当てにしないよう、**幅や行の並び**で確かめます。
# 日本語のフォントが無い環境では、日本語のテストだけ skip します。


def _find(*names):
    for path in list_font_files():
        base = os.path.basename(path).lower()
        if any(base.startswith(name) for name in names):
            return path
    return None


LATIN = _find("arial", "dejavusans", "liberationsans", "segoeui", "verdana", "roboto")
CJK = _find("meiryo", "msgothic", "yugoth", "notosanscjk", "notosansjp", "hiragino", "sourcehansans")

pytestmark = pytest.mark.skipif(LATIN is None, reason="使える TrueType フォントがこの環境にありません")
needs_cjk = pytest.mark.skipif(CJK is None, reason="日本語のフォントがこの環境にありません")


@pytest.fixture(scope="module")
def fm():
    fonts = {"Main": LATIN}
    if CJK:
        fonts["JP"] = CJK
    return FontManager(fonts=fonts)


# ══════════════════════════════════════════════════════════════════
# 基本の組版
# ══════════════════════════════════════════════════════════════════


def test_1行の幅と高さが取れる(fm):
    layout = T.layout_text("Hello", {"family": "Main", "size": 40}, fm)
    assert len(layout["lines"]) == 1
    assert layout["lines"][0]["text"] == "Hello"
    assert layout["width"] > 0
    assert layout["height"] == pytest.approx(layout["ascent"] + layout["descent"])


def test_改行で行が分かれる_CRは字にならない(fm):
    layout = T.layout_text("a\r\nb\nc", {"family": "Main", "size": 20}, fm)
    assert [line["text"] for line in layout["lines"]] == ["a", "b", "c"]


def test_字間を空けると幅が増える(fm):
    a = T.layout_text("ABCDE", {"family": "Main", "size": 30}, fm)
    b = T.layout_text("ABCDE", {"family": "Main", "size": 30, "letterSpacing": 5}, fm)
    # 5 字なので «字間 4 つ分» 増える（末尾の字間は幅に含めない）
    assert b["width"] == pytest.approx(a["width"] + 20)


def test_行間は基準サイズの倍率で決まる(fm):
    layout = T.layout_text("a\nb\nc", {"family": "Main", "size": 40, "lineHeight": 1.5}, fm)
    assert layout["line_height"] == pytest.approx(60)
    assert layout["height"] == pytest.approx(120 + layout["ascent"] + layout["descent"])


def test_折返しは語の切れ目で起きる(fm):
    layout = T.layout_text("aaa bbb ccc ddd eee", {"family": "Main", "size": 24, "maxWidth": 90}, fm)
    assert len(layout["lines"]) > 1
    for line in layout["lines"]:
        # 語の途中では切らない
        assert not line["text"].startswith(" ")
        assert not line["text"].endswith(" ")


def test_そろえを変えても幅は変わらない(fm):
    style = {"family": "Main", "size": 24}
    left = T.layout_text("one\ntwo three", style, fm)
    center = T.layout_text("one\ntwo three", {**style, "align": "center"}, fm)
    assert left["width"] == pytest.approx(center["width"])


# ══════════════════════════════════════════════════════════════════
# 日本語組版 — 禁則処理
# ══════════════════════════════════════════════════════════════════


@needs_cjk
def test_句読点は行頭に来ない(fm):
    style = {"family": "JP", "size": 28, "maxWidth": 130, "kinsoku": "normal"}
    layout = T.layout_text("あいうえ、おかきくけこ。さしす", style, fm)
    for line in layout["lines"][1:]:
        assert line["text"][0] not in "、。", f"行頭に句読点が来ています: {line['text']}"


@needs_cjk
def test_始め括弧は行末に来ない(fm):
    style = {"family": "JP", "size": 28, "maxWidth": 120, "kinsoku": "normal"}
    layout = T.layout_text("あいうえお「かきくけこ」さしす", style, fm)
    for line in layout["lines"][:-1]:
        assert line["text"][-1] not in "「（【", f"行末に始め括弧が来ています: {line['text']}"


@needs_cjk
def test_強い禁則では小書きの仮名も行頭に来ない(fm):
    style = {"family": "JP", "size": 28, "maxWidth": 130}
    strict = T.layout_text("きゃっとぁいうぇおっかきく", {**style, "kinsoku": "strict"}, fm)
    for line in strict["lines"][1:]:
        assert line["text"][0] not in "ぁぃぅぇぉっゃゅょ", f"行頭に小書きの仮名: {line['text']}"


@needs_cjk
def test_禁則をoffにすると幅だけで折る(fm):
    style = {"family": "JP", "size": 28, "maxWidth": 130}
    off = T.layout_text("あいうえ、おかきくけこ", {**style, "kinsoku": "off"}, fm)
    normal = T.layout_text("あいうえ、おかきくけこ", {**style, "kinsoku": "normal"}, fm)
    # off のほうが 1 行に詰められるので、行数は増えない
    assert len(off["lines"]) <= len(normal["lines"])
    assert off["lines"][0].__contains__("text")


def test_禁則の表に必要な文字が入っている():
    for char in "、。）」』！？ー・…":
        assert char in T.KINSOKU_LEADING_NORMAL
    for char in "（「『【":
        assert char in T.KINSOKU_TRAILING
    for char in "ぁっゃゅょ":
        assert char in T.KINSOKU_LEADING_STRICT
        assert char not in T.KINSOKU_LEADING_NORMAL


# ══════════════════════════════════════════════════════════════════
# ルビ
# ══════════════════════════════════════════════════════════════════


def test_ルビ記法は親文字とルビに分かれる():
    assert T.split_ruby("夜明[よあ]けまで", True) == [
        {"text": "夜明", "ruby": "よあ"},
        {"text": "けまで"},
    ]
    # 縦棒で親文字の範囲を明示できる
    assert T.split_ruby("｜1[いち]番", True) == [{"text": "1", "ruby": "いち"}, {"text": "番"}]
    # 英数字の並びも親文字になる
    assert T.split_ruby("abc[えー]def", True) == [{"text": "abc", "ruby": "えー"}, {"text": "def"}]


def test_ルビが無効なら記法として扱わない():
    assert T.split_ruby("夜明[よあ]け", False) == [{"text": "夜明[よあ]け"}]
    # 親文字が決まらないときは «ただの文字» として残す
    assert T.split_ruby("[よあ]け", True) == [{"text": "[よあ]け"}]


@needs_cjk
def test_ルビが親文字の上に並ぶ(fm):
    layout = T.layout_text("夜明[よあ]けまで", {"family": "JP", "size": 40, "ruby": {"enabled": True}}, fm)
    ruby = layout["lines"][0].get("ruby_glyphs")
    assert ruby and [g["char"] for g in ruby] == ["よ", "あ"]
    # ルビはベースラインより上（dy が負）
    assert all(g["dy"] < 0 for g in ruby)
    # ルビの中心が親文字（先頭 2 字）の中心にそろう
    parents = layout["lines"][0]["glyphs"][:2]
    parent_center = (parents[0]["x"] + parents[-1]["x"] + parents[-1]["advance"]) / 2
    ruby_center = (ruby[0]["x"] + ruby[-1]["x"] + ruby[-1]["advance"]) / 2
    assert ruby_center == pytest.approx(parent_center, abs=0.5)


@needs_cjk
def test_ルビが長いと親文字の前後に余白が入る(fm):
    style = {"family": "JP", "size": 40, "ruby": {"enabled": True}}
    plain = T.layout_text("空へ", style, fm)
    with_ruby = T.layout_text("空[おおぞら]へ", style, fm)
    # ルビ 5 字 > 親 1 字 なので、親文字が広げられて全体が長くなる
    assert with_ruby["width"] > plain["width"]


@needs_cjk
def test_ルビの親文字は途中で折らない(fm):
    style = {"family": "JP", "size": 28, "maxWidth": 120, "ruby": {"enabled": True}}
    layout = T.layout_text("あいうえお夜明[よあ]けまで", style, fm)
    joined = "".join(line["text"] for line in layout["lines"])
    assert "夜明" in joined
    for line in layout["lines"]:
        assert not (line["text"].endswith("夜") and "明" not in line["text"])


# ══════════════════════════════════════════════════════════════════
# 枠に収める自動縮小（fit）
# ══════════════════════════════════════════════════════════════════


def test_fitは枠に収まるまで縮める(fm):
    style = {"family": "Main", "size": 60, "fit": {"mode": "shrink", "maxWidth": 200, "minSize": 0.2}}
    layout = T.layout_text("A very long line of text", style, fm)
    assert layout["fit_scale"] < 1
    assert layout["width"] <= 200 * 1.01


def test_fitはminSizeより小さくしない(fm):
    style = {"family": "Main", "size": 80, "fit": {"mode": "shrink", "maxWidth": 20, "minSize": 0.5}}
    layout = T.layout_text("This will never fit", style, fm)
    assert layout["fit_scale"] == pytest.approx(0.5)


def test_fitは行数でも効く(fm):
    style = {"family": "Main", "size": 30, "maxWidth": 120, "fit": {"mode": "shrink", "maxLines": 2, "minSize": 0.2}}
    layout = T.layout_text("a b c d e f g h i j k l", style, fm)
    assert layout["fit_scale"] < 1


def test_fitが無効なら何もしない(fm):
    base = T.layout_text("Hello", {"family": "Main", "size": 40}, fm)
    off = T.layout_text("Hello", {"family": "Main", "size": 40, "fit": {"enabled": False, "maxWidth": 5}}, fm)
    assert off["width"] == pytest.approx(base["width"])
    assert "fit_scale" not in off


def test_fitのパーセント指定は基準から解ける(fm):
    style = {"family": "Main", "size": 40, "maxWidth": 400, "fit": {"mode": "shrink", "maxWidth": "50%", "minSize": 0.1}}
    layout = T.layout_text("Long enough to need shrinking here", style, fm)
    assert layout["width"] <= 200 * 1.01


# ══════════════════════════════════════════════════════════════════
# リッチテキスト（runs / markup）
# ══════════════════════════════════════════════════════════════════


def test_簡易記法をランに分解する():
    assert T.parse_text_markup("a<c:#f00>b</c>c") == [
        {"t": "a"},
        {"t": "b", "color": "#f00"},
        {"t": "c"},
    ]
    assert T.parse_text_markup("<s:1.5x>big</s>") == [{"t": "big", "sizeScale": 1.5}]
    assert T.parse_text_markup("<s:96>px</s>") == [{"t": "px", "size": 96.0}]
    assert T.parse_text_markup("<b><i>bi</i></b>") == [{"t": "bi", "bold": True, "italic": True}]


def test_知らないタグはただの文字として残る():
    # `<` を含む普通の文章を «記法の書き間違い» として壊さない
    assert T.parse_text_markup("a < b and <span>x</span>") == [{"t": "a < b and <span>x</span>"}]


def test_閉じ忘れても落ちない():
    assert T.parse_text_markup("<c:#f00>red") == [{"t": "red", "color": "#f00"}]
    assert T.parse_text_markup("red</c>") == [{"t": "red"}]


def test_ランで大きさを変えると幅が変わる(fm):
    style = {"family": "Main", "size": 40}
    plain = T.layout_text("abcabc", style, fm)
    big = T.layout_text("abcabc", {**style, "runs": [{"t": "abc"}, {"t": "abc", "size": 80}]}, fm)
    assert big["width"] > plain["width"]
    # 大きいランのグリフだけ size が上がっている
    sizes = [glyph["size"] for glyph in big["lines"][0]["glyphs"]]
    assert sizes[:3] == [40, 40, 40] and sizes[3:] == [80, 80, 80]


def test_ランが本文と食い違うときは捨てる(fm):
    # カウンターのように «本文だけ» 差し替わることがあるので、そのときは色分けを当てない
    style = {"family": "Main", "size": 30, "runs": [{"t": "old", "color": "#ff0000"}]}
    layout = T.layout_text("new", style, fm)
    assert layout["lines"][0]["text"] == "new"
    assert all(glyph["color"] is None for glyph in layout["lines"][0]["glyphs"])


def test_ランごとに色が塗り分けられる(fm):
    out = T.render_text(
        "AB",
        {"family": "Main", "size": 60, "runs": [{"t": "A", "color": "#ff0000"}, {"t": "B", "color": "#0000ff"}]},
        fm,
    )
    pixels = out["bitmap"].data.reshape(-1, 4)
    opaque = pixels[pixels[:, 3] > 200]
    assert (opaque[:, 0] > 200).any(), "赤い画素がありません"
    assert (opaque[:, 2] > 200).any(), "青い画素がありません"


# ══════════════════════════════════════════════════════════════════
# 縦書き
# ══════════════════════════════════════════════════════════════════


@needs_cjk
def test_縦書きは列が右から左へ並ぶ(fm):
    layout = T.layout_text("あい\nうえ", {"family": "JP", "size": 34, "direction": "vertical"}, fm)
    assert layout["vertical"] is True
    first = layout["lines"][0]["glyphs"][0]["x"]
    second = layout["lines"][1]["glyphs"][0]["x"]
    assert first > second, "1 行目が右に来ていません"
    # 同じ列の中は上から下へ
    baselines = [glyph["baseline"] for glyph in layout["lines"][0]["glyphs"]]
    assert baselines == sorted(baselines)


@needs_cjk
def test_縦書きのルビは列の右側に付く(fm):
    layout = T.layout_text("夜明[よあ]け", {"family": "JP", "size": 34, "direction": "vertical", "ruby": {"enabled": True}}, fm)
    ruby = layout["lines"][0].get("ruby_glyphs")
    assert ruby
    body_x = layout["lines"][0]["glyphs"][0]["x"]
    assert ruby[0]["x"] > body_x


# ══════════════════════════════════════════════════════════════════
# 描画
# ══════════════════════════════════════════════════════════════════


def test_文字が描かれる_余白は透明(fm):
    out = T.render_text("Ag", {"family": "Main", "size": 48, "color": "#ffffff"}, fm)
    data = out["bitmap"].data
    assert (data[..., 3] > 0).sum() > 50
    assert data[0, 0, 3] == 0


def test_空文字でも落ちない(fm):
    out = T.render_text("", {"family": "Main", "size": 40}, fm)
    assert out["bitmap"].width >= 1
    assert not out["bitmap"].data[..., 3].any()


def test_縁取りは塗りの外側に付く(fm):
    plain = T.render_text("O", {"family": "Main", "size": 60, "color": "#ffffff"}, fm)
    stroked = T.render_text(
        "O", {"family": "Main", "size": 60, "color": "#ffffff", "stroke": {"width": 6, "color": "#ff0000"}}, fm
    )
    assert stroked["bitmap"].width > plain["bitmap"].width
    pixels = stroked["bitmap"].data.reshape(-1, 4)
    opaque = pixels[pixels[:, 3] > 200]
    assert ((opaque[:, 0] > 200) & (opaque[:, 1] < 60)).any(), "赤い縁がありません"


def test_影はずらしてぼかされる(fm):
    out = T.render_text(
        "S",
        {"family": "Main", "size": 48, "color": "#ffffff", "shadow": {"color": "#000000", "blur": 4, "offsetX": 6, "offsetY": 6}},
        fm,
    )
    # 影は右下へ出るので、余白は «右» に増える（左は blur - offset で 0 のまま）
    plain = T.render_text("S", {"family": "Main", "size": 48, "color": "#ffffff"}, fm)
    assert out["bitmap"].width > plain["bitmap"].width
    assert out["offset_x"] == plain["offset_x"]
    pixels = out["bitmap"].data.reshape(-1, 4)
    # 半透明（ぼけた影の縁）の画素があること
    assert ((pixels[:, 3] > 10) & (pixels[:, 3] < 200)).any()


def test_pixelGridでドット絵風になる(fm):
    out = T.render_text("A", {"family": "Main", "size": 48, "color": "#ffffff", "pixelGrid": 6}, fm)
    alpha = out["bitmap"].data[..., 3]
    # 6x6 のマスの中はすべて同じ値になる
    ys, xs = np.nonzero(alpha)
    if ys.size:
        y0 = (ys[0] // 6) * 6
        x0 = (xs[0] // 6) * 6
        cell = alpha[y0 : y0 + 6, x0 : x0 + 6]
        assert len(np.unique(cell)) == 1


def test_antialias_falseで中間調が消える(fm):
    out = T.render_text("A", {"family": "Main", "size": 48, "color": "#ffffff", "antialias": False}, fm)
    alpha = np.unique(out["bitmap"].data[..., 3])
    assert set(alpha.tolist()) <= {0, 255}


# ══════════════════════════════════════════════════════════════════
# 文字アニメーション
# ══════════════════════════════════════════════════════════════════


def test_時間0では何も出ない_時間が進むと出る(fm):
    style = {"family": "Main", "size": 40, "color": "#ffffff"}
    animator = {"unit": "character", "stagger": 0.1, "duration": 0.3, "from": {"opacity": 0, "y": 20}}
    early = T.render_animated_text("ABC", style, fm, animator, 0.0)
    late = T.render_animated_text("ABC", style, fm, animator, 5.0)
    assert not early["bitmap"].data[..., 3].any()
    assert late["bitmap"].data[..., 3].any()
    assert late["units"] == 3


def test_単位を語や行にすると数が変わる(fm):
    style = {"family": "Main", "size": 30, "color": "#ffffff"}
    base = {"stagger": 0.05, "duration": 0.2}
    chars = T.render_animated_text("ab cd", style, fm, {**base, "unit": "character"}, 9)
    words = T.render_animated_text("ab cd", style, fm, {**base, "unit": "word"}, 9)
    lines = T.render_animated_text("ab cd", style, fm, {**base, "unit": "line"}, 9)
    assert chars["units"] == 4  # 空白は数えない
    assert words["units"] == 2
    assert lines["units"] == 1


def test_順番の指定():
    assert T.order_indices(4, "forward", 1) == [0, 1, 2, 3]
    assert T.order_indices(4, "reverse", 1) == [3, 2, 1, 0]
    assert T.order_indices(5, "center", 1) == [2, 1, 0, 1, 2]
    shuffled = T.order_indices(8, "random", 12345)
    assert sorted(shuffled) == list(range(8))
    # 同じ seed からは必ず同じ並び（決定性）
    assert shuffled == T.order_indices(8, "random", 12345)


def test_乱数はJSと同じ数列を出す():
    # JS の `hashUnit(0, 12345, 1)` などと突き合わせた値
    values = [T.hash_unit(i, 12345, 1) for i in range(4)]
    assert all(0 <= v < 1 for v in values)
    assert len(set(values)) == 4
    # 32 ビットで回っていること（Python の無限精度のままだと違う値になる）
    assert T.hash_unit(1000000, 999999, 3) == T.hash_unit(1000000, 999999, 3)


def test_mulberry32はJSと同じ数列を出す():
    # JS 版の `createRandom(12345)` を実際に走らせて取った値
    random = TX.create_random(12345)
    assert [random() for _ in range(3)] == [0.9797282677609473, 0.3067522644996643, 0.484205421525985]
    # seed 0 は 0x9e3779b9 に読み替えられる（JS 版と同じ）
    zero = TX.create_random(0)
    assert [zero() for _ in range(2)] == [0.3588899802416563, 0.10590326134115458]


# ══════════════════════════════════════════════════════════════════
# textBox / textPath / 書き順 / ランダムフォント
# ══════════════════════════════════════════════════════════════════


def test_paddingの書き方(fm):
    assert TX.resolve_padding(8) == [8, 8, 8, 8]
    assert TX.resolve_padding([4, 10]) == [4, 10, 4, 10]
    assert TX.resolve_padding([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert TX.resolve_padding(None) == [0, 0, 0, 0]


def test_枠は文字の実寸に追従する(fm):
    short = T.render_text("A", {"family": "Main", "size": 30, "color": "#ffffff"}, fm)
    long = T.render_text("AAAAAAAA", {"family": "Main", "size": 30, "color": "#ffffff"}, fm)
    box_s = T.draw_text_box(short["bitmap"], short, {"padding": 10, "fill": "#334455"})
    box_l = T.draw_text_box(long["bitmap"], long, {"padding": 10, "fill": "#334455"})
    assert box_l["box_width"] > box_s["box_width"]
    assert box_s["box_width"] == pytest.approx(short["layout"]["width"] + 20)


def test_枠のrevealは幅を狭める(fm):
    rendered = T.render_text("Reveal", {"family": "Main", "size": 30, "color": "#ffffff"}, fm)
    full = T.draw_text_box(rendered["bitmap"], rendered, {"padding": 8, "fill": "#ff8800"})
    half = T.draw_text_box(rendered["bitmap"], rendered, {"padding": 8, "fill": "#ff8800", "reveal": {"progress": 0.5}})
    # 枠の «塗られた» 面積が半分ほどになる
    orange = lambda b: int(((b.data[..., 0] > 200) & (b.data[..., 1] > 100) & (b.data[..., 2] < 60)).sum())  # noqa: E731
    assert orange(half["bitmap"]) < orange(full["bitmap"]) * 0.7


def test_パス上に文字を並べる(fm):
    out = T.render_text_on_path(
        "CIRCLE", {"family": "Main", "size": 26, "color": "#ffffff"}, fm,
        {"shape": "circle", "radius": 80, "startAngle": -90, "sweep": 360},
    )
    assert out is not None
    assert out["bitmap"].data[..., 3].any()
    # 円周に沿うので、真ん中は空く
    h, w = out["bitmap"].data.shape[:2]
    assert out["bitmap"].data[h // 2, w // 2, 3] == 0


def test_折れ線が2点未満なら描かない(fm):
    assert T.render_text_on_path("X", {"family": "Main", "size": 20}, fm, {"shape": "polyline", "points": [[0, 0]]}) is None


def test_書き順は時間とともに増える():
    contours = [
        [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
        [20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0],
        [40.0, 40.0, 50.0, 40.0, 50.0, 50.0, 40.0, 50.0],
    ]
    spec = {"stagger": 0.1, "duration": 0.05}
    assert TX.apply_stroke_order(contours, spec, 0.0) == []
    assert len(TX.apply_stroke_order(contours, spec, 0.12)) == 2
    assert len(TX.apply_stroke_order(contours, spec, 1.0)) == 3


def test_ランダムフォントは決定的():
    spec = {"families": ["A", "B", "C"], "seed": 42}
    first = [TX.random_font_for(i, spec) for i in range(10)]
    assert first == [TX.random_font_for(i, spec) for i in range(10)]
    assert set(first) <= {"A", "B", "C"}
    # interval を付けると時間で切り替わる
    moving = {"families": ["A", "B", "C"], "seed": 42, "interval": 0.1}
    assert TX.random_font_for(0, moving, 0.0) == TX.random_font_for(0, moving, 0.05)
    assert {TX.random_font_for(0, moving, t / 10) for t in range(20)} != {TX.random_font_for(0, moving, 0)}
    assert TX.random_font_for(0, {"families": []}) is None


# ══════════════════════════════════════════════════════════════════
# カラオケ塗り・カウンター
# ══════════════════════════════════════════════════════════════════


def test_カラオケは左から塗り替わる():
    bmp = Bitmap(20, 4)
    bmp.data[...] = [255, 255, 255, 255]
    out = T.apply_karaoke_fill(bmp, {"progress": 0.5, "color": "#ff0000", "softness": 0},
                               {"offset_x": 0, "width": 20, "base_color": "#ffffff"})
    assert list(out.data[2, 2]) == [255, 0, 0, 255]
    assert list(out.data[2, 18]) == [255, 255, 255, 255]


def test_カラオケは縁取りを塗り替えない():
    bmp = Bitmap(20, 4)
    bmp.data[...] = [255, 255, 255, 255]
    bmp.data[:, :5] = [0, 0, 0, 255]  # 縁取りのつもりの黒
    out = T.apply_karaoke_fill(bmp, {"progress": 1.0, "color": "#ff0000", "softness": 0},
                               {"offset_x": 0, "width": 20, "base_color": "#ffffff"})
    assert list(out.data[2, 2]) == [0, 0, 0, 255], "地の色から遠い縁取りが塗り替わっています"
    assert list(out.data[2, 10]) == [255, 0, 0, 255]


def test_カラオケのprogress0は何もしない():
    bmp = Bitmap(8, 4)
    bmp.data[...] = [255, 255, 255, 255]
    out = T.apply_karaoke_fill(bmp, {"progress": 0}, {"offset_x": 0, "width": 8})
    assert out is bmp


@pytest.mark.parametrize(
    "counter,progress,expected",
    [
        ({"from": 0, "to": 1234567, "separator": True}, 1, "1,234,567"),
        ({"from": 0, "to": 100, "decimals": 2}, 0.3333, "33.33"),
        ({"from": 0, "to": 42, "pad": 5, "prefix": "#"}, 1, "#00042"),
        ({"from": -50, "to": 50, "decimals": 1, "suffix": "%"}, 0.25, "-25.0%"),
        # JS の toFixed は «2 進数の正確な値» を丸めるので 1.005 は 1.00 になる
        ({"from": 0, "to": 1.005, "decimals": 2}, 1, "1.00"),
        # 0.125 は 2 進数で正確なので «半分は上へ» が効いて 0.13
        ({"from": 0, "to": 0.125, "decimals": 2}, 1, "0.13"),
        ({"from": 0, "to": 9876.5432, "decimals": 3, "separator": True}, 1, "9,876.543"),
    ],
)
def test_カウンターの書式(counter, progress, expected):
    assert T.format_counter(counter, progress) == expected


# ══════════════════════════════════════════════════════════════════
# スタイルの正規化
# ══════════════════════════════════════════════════════════════════


def test_スタイルのいろいろな書き方が1つにまとまる():
    out = T.resolve_text_style({}, {"text": "hi", "style": {"fontSize": 32, "weight": "700", "fill": "#39c5bb"}})
    assert out["content"] == "hi"
    assert out["style"]["size"] == 32
    assert out["style"]["bold"] is True
    assert out["style"]["color"] == "rgba(57, 197, 187, 1)"


def test_widthはそのままmaxWidthになる():
    """⚠ **片方だけ書いても比率は補われません。** JS 版のこの挙動をそのまま守ります。

    `width` を書くと折返し幅になりますが、`height` を書いても文字の大きさは
    変わりません。比率を保ちたいときは `scale` を使ってください
    （`docs/json-reference.ja.md` に理由が書いてあります）。
    """
    out = T.resolve_text_style({}, {"text": "x", "style": {"width": 320}})
    assert out["style"]["maxWidth"] == 320
    # maxWidth を明示したらそちらが勝つ
    out = T.resolve_text_style({}, {"text": "x", "style": {"width": 320, "maxWidth": 200}})
    assert out["style"]["maxWidth"] == 200


def test_fitとrubyのpxは基準サイズとの比に直る():
    out = T.resolve_text_style(
        {"transform": {"width": 400}},
        {"text": "x", "style": {"size": 40, "fit": {"mode": "shrink", "maxWidth": 200}, "ruby": {"offset": 6}}},
    )
    assert out["style"]["fit"]["maxWidthEm"] == pytest.approx(5.0)
    assert out["style"]["fit"]["basisEm"] == pytest.approx(10.0)
    assert out["style"]["ruby"]["offsetEm"] == pytest.approx(0.15)


def test_markupは本文からランを作る():
    out = T.resolve_text_style({}, {"text": "a<c:#ff0000>b</c>", "style": {"markup": True, "size": 40}})
    assert out["content"] == "ab"
    assert [run["text"] for run in out["style"]["runs"]] == ["a", "b"]
    assert out["style"]["runs"][1]["color"] == "rgba(255, 0, 0, 1)"


# ══════════════════════════════════════════════════════════════════
# 高品質出力（superSample）
# ══════════════════════════════════════════════════════════════════


def test_サイズを倍にすると絵も倍になる(fm):
    """`--super-sample` は `style.size` を倍にするだけ。比が崩れないことを見ます。"""
    one = T.layout_text("Hamburg", {"family": "Main", "size": 40}, fm)
    two = T.layout_text("Hamburg", {"family": "Main", "size": 80}, fm)
    assert two["width"] == pytest.approx(one["width"] * 2, rel=1e-9)
    assert two["height"] == pytest.approx(one["height"] * 2, rel=1e-9)


def test_ランの倍率は高品質出力でも崩れない(fm):
    """ランの大きさは **«実寸» ではなく «倍率»** で持ちます。

    レンダラーは高品質出力（`--super-sample`）で `style.size` **だけ**を倍にします。
    ランに px の実寸を残すと、そこだけ元の大きさに取り残されて比が崩れます。
    `resolve_text_style` が入口で 1 度だけ倍率に直すのはそのためです。
    """
    resolved = T.resolve_text_style({}, {"text": "abcd", "style": {"size": 40, "runs": [{"t": "ab"}, {"t": "cd", "size": 60}]}})
    runs = resolved["style"]["runs"]
    assert runs[1]["size_scale"] == pytest.approx(1.5)

    # 正規化したランをもう一度組版に渡しても倍率が消えないこと（往復の回帰テスト）
    one = T.layout_text("abcd", {"family": "Main", "size": 40, "runs": runs}, fm)
    assert [g["size"] for g in one["lines"][0]["glyphs"]] == [40, 40, 60, 60]

    # 倍率で持っていれば、size を倍にしたときに全体がきれいに倍になる
    two = T.layout_text("abcd", {"family": "Main", "size": 80, "runs": runs}, fm)
    assert two["width"] == pytest.approx(one["width"] * 2, rel=1e-9)
