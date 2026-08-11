"""同梱のスキル定義（`movo/skill/library/`）。

定義は JS 版から **JSON のまま持ってきた** ので、移植で «取りこぼした» ものが
無いことをここで確かめます。数と名前を書いてあるのは、静かに 1 件消えるのが
いちばん気付きにくいからです。
"""

import pytest

from movo.skill import KINDS, SkillRegistry, builtin_library_root, find_dead_inputs

REGISTRY = SkillRegistry().load()


def test_組み込みの置き場が見つかる():
    assert builtin_library_root().is_dir()


def test_4_種類すべてが読み込まれている():
    for kind in KINDS:
        assert REGISTRY.list(kind), f"{kind} が 1 件も読み込まれていません"


def test_定義の総数が_JS_版と同じ():
    # 28 アニメーション / 15 スキル / 11 シーン / 3 ムービー = 57 件
    assert len(REGISTRY.list()) == 57


@pytest.mark.parametrize("name", ["rich-intro", "rich-verse", "rich-chorus", "rich-bridge", "rich-burst", "rich-outro"])
def test_rich_シーンが_6_種そろっている(name):
    assert REGISTRY.scene_skill(name) is not None


@pytest.mark.parametrize("name", ["rich-mv", "lyric-mv", "hype-lyric-mv"])
def test_ムービースキルが_3_種そろっている(name):
    entry = REGISTRY.movie(name)
    assert entry is not None
    assert entry["definition"].get("sequence"), f"{name} に sequence がありません"


@pytest.mark.parametrize("name", ["mv-intro", "mv-verse", "mv-chorus", "mv-outro", "mv-hype"])
def test_mv_シーンがそろっている(name):
    assert REGISTRY.scene_skill(name) is not None


def test_名前空間は種類ごとに分かれている():
    # 同じ "intro" でも «レイヤー群» と «シーンまるごと» では使う場所が違います。
    names = REGISTRY.names()
    assert set(names) == {"animations", "skills", "scenes", "movies"}


def test_見出しはすべて揃っている():
    for entry in REGISTRY.list():
        assert entry["name"]
        assert entry["title"]
        assert entry["kind"] in KINDS


def test_どの定義にも死んだ入力が無い():
    # 宣言だけ残って本文から参照されていない入力は、使う人からは
    # «変えても何も起きない壊れた項目» に見えます。
    dead = {entry["name"]: find_dead_inputs(REGISTRY, entry["name"]) for entry in REGISTRY.list()}
    assert {name: keys for name, keys in dead.items() if keys} == {}


def test_プロジェクト側の定義が組み込みを上書きする(tmp_path):
    directory = tmp_path / "skills"
    directory.mkdir()
    (directory / "title-card.json").write_text(
        '{"skill":{"name":"title-card","title":"じぶんの"},"layers":[]}', encoding="utf-8"
    )
    registry = SkillRegistry().load(project_root=str(tmp_path))
    assert registry.skill("title-card")["title"] == "じぶんの"
    assert registry.skill("title-card")["source"] == "project"


def test_コメント付きの_JSON_も読める(tmp_path):
    directory = tmp_path / "animations"
    directory.mkdir()
    (directory / "x.json").write_text(
        '{\n // 見出し\n "animation": {"name": "x"},\n "produces": {} /* 何も作らない */\n}',
        encoding="utf-8",
    )
    registry = SkillRegistry().load(project_root=str(tmp_path), builtin=False)
    assert registry.animation("x") is not None
