"""CLI の入口とテンプレート。**必ず起動すること** を守るための試験です。

移植の途中でも `movo --version` や `movo skill list` は動く、という約束を
ここで固定しています（繋がっていない部分に触れたときだけ «後で繋ぐ» で止まる）。
"""

import json

import pytest

from movo.cli.main import run
from movo.cli.templates import TEMPLATES, build_template


def test_バージョンを出す(capsys):
    assert run(["--version"]) == 0
    assert capsys.readouterr().out.strip()


def test_引数なしならヘルプ(capsys):
    assert run([]) == 0
    assert "使い方" in capsys.readouterr().out


def test_コマンドごとのヘルプ(capsys):
    assert run(["help", "render"]) == 0
    assert "--jobs" in capsys.readouterr().out


def test_不明なコマンドは終了コード_2(capsys):
    assert run(["renderr"]) == 2


def test_打ち間違いに提案を出す(capsys):
    run(["rende"])
    assert "もしかして" in capsys.readouterr().err


def test_スキル一覧は繋がっていなくても動く(capsys):
    assert run(["skill", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["skills"] and payload["movies"]


def test_一覧は未接続でも止まらない(capsys):
    # «何が使えるか» を調べるコマンドなので、0 件と出すほうが情報量が多いです。
    assert run(["list", "easings"]) == 0


def test_不明な一覧の種類は終了コード_2():
    assert run(["list", "そんなの"]) == 2


def test_doctor_は繋がり具合を出す(capsys):
    assert run(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bridge"]
    assert all("connected" in row for row in payload["bridge"])


def test_無いファイルを渡すと終了コード_4(capsys):
    assert run(["validate", "存在しない.json"]) == 4


@pytest.mark.parametrize("template", TEMPLATES)
def test_どのテンプレートも組み立てられる(template):
    project = build_template(template, {"name": "t", "width": 1920, "height": 1080, "fps": 30})
    assert project["scenes"]
    assert project["video"]["width"] == 1920
    # そのまま JSON にできること（NumPy の値などが紛れていないこと）
    json.dumps(project, ensure_ascii=False)


def test_init_はプロジェクト一式を作る(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert run(["init", "demo"]) == 0
    assert (tmp_path / "demo" / "movo.json").is_file()
    assert (tmp_path / "demo" / ".gitignore").is_file()
    assert (tmp_path / "demo" / "output").is_dir()


def test_init_は空でない場所に上書きしない(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "x").write_text("", encoding="utf-8")
    assert run(["init", "demo"]) == 2


def test_skill_new_は雛形を作る(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run(["skill", "new", "my-title"]) == 0
    written = json.loads((tmp_path / "skills" / "my-title.json").read_text(encoding="utf-8"))
    assert written["skill"]["name"] == "my-title"


def test_skill_new_scene_は_scenes_に置く(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run(["skill", "new", "my-intro", "--scene"]) == 0
    written = json.loads((tmp_path / "scenes" / "my-intro.json").read_text(encoding="utf-8"))
    assert written["skill"]["kind"] == "scene"


def test_config_は保存して読み戻せる(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MOVO_HOME", str(tmp_path / ".movo"))
    assert run(["config", "set", "openai.apiKey", "sk-0123456789abcdef"]) == 0
    capsys.readouterr()
    assert run(["config", "get", "openai.apiKey", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # 秘密は伏せて返すこと（一覧を出しただけで端末の履歴に残らないように）
    assert payload["value"].startswith("sk-0")
    assert "*" in payload["value"]
