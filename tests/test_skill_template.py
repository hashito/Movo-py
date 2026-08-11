"""スキルのテンプレート展開（`${...}` / `when` / `repeat`）と入力値の解決。"""

import pytest

from movo.cli.errors import MovoError, reason_of
from movo.skill import SkillRegistry, build_skill_project, expand_template, resolve_inputs
from movo.skill.template import create_skill_engine, is_timed_line

pytestmark = pytest.mark.skipif(
    create_skill_engine(0) is None, reason="式エンジン（movo.expression）がまだ繋がっていません"
)


def expand(node, scope=None):
    return expand_template(node, scope or {})


# ── 式の埋め込み ────────────────────────────────────────────


def test_全体が式なら型を保つ():
    assert expand("${bars}", {"bars": 4}) == 4
    assert expand("${lines}", {"lines": ["あ", "い"]}) == ["あ", "い"]


def test_文字列の中にも埋め込める():
    assert expand("BPM ${bpm} で同期", {"bpm": 174}) == "BPM 174 で同期"


def test_計算できる():
    assert expand("${height * 0.4}", {"height": 1080}) == pytest.approx(432)


def test_2_つ並べたら文字列になる():
    # `[^{}]` にしているのは "${a}-${b}" を 1 個と誤認しないためです。
    assert expand("${a}-${b}", {"a": 1, "b": 2}) == "1-2"


# ── when ────────────────────────────────────────────────────


def test_when_が偽なら要素ごと消える():
    assert expand([{"when": "${show}", "id": "a"}, {"id": "b"}], {"show": False}) == [{"id": "b"}]


def test_when_が真なら残る():
    assert expand([{"when": "${show}", "id": "a"}], {"show": True}) == [{"id": "a"}]


def test_値に付けた_when_はキーごと消える():
    result = expand({"assets": {"when": "${use}", "art": "a.png"}, "id": "x"}, {"use": False})
    assert result == {"id": "x"}


# ── repeat ──────────────────────────────────────────────────


def test_count_で数だけ繰り返す():
    result = expand([{"repeat": {"count": "${n}", "as": "i"}, "id": "l${i}"}], {"n": 3})
    assert [item["id"] for item in result] == ["l0", "l1", "l2"]


def test_over_で配列の要素ごとに繰り返す():
    # «行ごとに 1 レイヤー» が MV でいちばん書きたい形です。
    result = expand(
        [{"repeat": {"over": "${lines}", "as": "line", "indexAs": "i"}, "text": "${line}", "n": "${i}"}],
        {"lines": ["あ", "い"]},
    )
    assert result == [{"text": "あ", "n": 0}, {"text": "い", "n": 1}]


def test_over_に配列でないものを渡すと止まる():
    with pytest.raises(MovoError) as caught:
        expand([{"repeat": {"over": "${x}"}, "id": "a"}], {"x": 5})
    assert "配列" in reason_of(caught.value)


def test_多すぎる繰り返しは止める():
    with pytest.raises(MovoError) as caught:
        expand([{"repeat": {"count": 900}, "id": "a"}])
    assert "too large" in reason_of(caught.value)


# ── 入力値の解決 ────────────────────────────────────────────


def test_既定値で埋まる():
    values, issues = resolve_inputs({"size": {"type": "number", "default": 64}}, {})
    assert values["size"] == 64
    assert issues == []


def test_必須を書かないと問題として返る():
    _, issues = resolve_inputs({"text": {"type": "text", "required": True}}, {})
    assert issues and "必須" in issues[0]["message"]


def test_範囲の外は問題として返る():
    _, issues = resolve_inputs({"bpm": {"type": "number", "min": 40, "max": 300}}, {"bpm": 999})
    assert issues and "300 以下" in issues[0]["message"]


def test_px_の寸法は解像度に追従する():
    # ライブラリの寸法は 1080p 基準なので、720p では 2/3 になります。
    values, _ = resolve_inputs({"size": {"type": "number", "unit": "px", "default": 96}}, {}, {"scale": 720 / 1080})
    assert values["size"] == pytest.approx(64)


def test_範囲の判定はスケール前の値で行う():
    # スケール後で判定すると、解像度を下げただけで «最小値を割った» と言われます。
    _, issues = resolve_inputs(
        {"size": {"type": "number", "unit": "px", "min": 80, "default": 96}}, {}, {"scale": 0.5}
    )
    assert issues == []


def test_textList_は行で切る():
    values, _ = resolve_inputs({"lines": {"type": "textList"}}, {"lines": "あ\nい\n\nう"})
    assert values["lines"] == ["あ", "い", "う"]


def test_textList_の既定は空配列():
    # None にすると «行の数だけ繰り返す» テンプレートが落ちます。
    values, _ = resolve_inputs({"lines": {"type": "textList"}}, {})
    assert values["lines"] == []


def test_時刻つきの歌詞はそのまま通す():
    # str() に掛けると歌詞は出るのに時刻だけ消え、気付きにくい壊れ方をします。
    line = {"text": "あ", "at": 12.4, "for": 2.7}
    values, _ = resolve_inputs({"lines": {"type": "textList"}}, {"lines": [line]})
    assert values["lines"] == [line]
    assert is_timed_line(line)


def test_選択肢の外は問題として返る():
    _, issues = resolve_inputs({"kind": {"type": "choice", "options": ["a", "b"]}}, {"kind": "c"})
    assert issues and "いずれか" in issues[0]["message"]


def test_定義に無い入力もそのまま渡す():
    values, issues = resolve_inputs({}, {"未知": 1})
    assert values["未知"] == 1
    assert issues == []


# ── 単体で動画にする（プロジェクト JSON の組み立てまで）──────


def test_スキル単体からプロジェクト_JSON_ができる():
    registry = SkillRegistry().load()
    built = build_skill_project(registry, "title-card", {"title": "Movo"})
    project = built["project"]
    assert project["video"]["width"] == 1920
    assert project["scenes"] and project["scenes"][0]["layers"]
    # 基礎アニメーションの `use` は展開済みで、素の JSON になっていること
    assert "use" not in project["scenes"][0]["layers"][0]


def test_ムービースキルはシーンの並びになる():
    registry = SkillRegistry().load()
    built = build_skill_project(registry, "lyric-mv", {"title": "夜明けまで", "bpm": 92})
    assert len(built["project"]["scenes"]) >= 4
    assert built["project"]["project"]["bpm"] == 92


def test_入力値はスキルの既定より強い():
    # 以前はスキルの既定が入力値より強く、--set bpm=90 が効きませんでした。
    registry = SkillRegistry().load()
    built = build_skill_project(registry, "lyric-mv", {"bpm": 90})
    assert built["project"]["project"]["bpm"] == 90


def test_解像度を変えても同じ絵の作りになる():
    registry = SkillRegistry().load()
    small = build_skill_project(registry, "title-card", {"title": "Movo"}, {"width": 1280, "height": 720})
    assert small["project"]["video"]["height"] == 720
    assert small["project"]["scenes"][0]["layers"]
