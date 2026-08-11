"""`movo batch` — 表から連番で書き出す。

**走る前に全部決める** のがこのコマンドの肝なので、そこを重点的に見ます。
20 分回してから «--out の書き方が違う» と言われるのがいちばん困ります。
"""

import json

import pytest

from movo.cli.commands import batch
from movo.cli.errors import MovoError, reason_of


# ── 表を読む ────────────────────────────────────────────────


def test_json_の配列を読む():
    rows = batch.parse_table('[{"name":"01","bpm":205}]', "songs.json")
    assert rows == [{"name": "01", "bpm": 205}]


def test_rows_を包んだ形も読む():
    rows = batch.parse_table('{"rows":[{"name":"a"}]}', "songs.json")
    assert rows == [{"name": "a"}]


def test_配列でなければ書き方を教える():
    with pytest.raises(MovoError) as caught:
        batch.parse_table('{"name":"a"}', "songs.json")
    assert "1 行 = 1 本" in reason_of(caught.value)


def test_csv_の型寄せは_set_と同じ規則():
    rows = batch.parse_table("name,bpm,loud\n01,205,true\n", "songs.csv")
    assert rows == [{"name": "01", "bpm": 205, "loud": True}]


def test_連番の名前は文字列のまま残す():
    # "01" を 1 にすると 01.mp4 が 1.mp4 になり、並び順が崩れます。
    rows = batch.parse_table("name\n01\n02\n", "songs.csv")
    assert [row["name"] for row in rows] == ["01", "02"]


def test_引用符の中のカンマは守る():
    rows = batch.parse_table('name,title\na,"入れ子の街, 夜"\n', "songs.csv")
    assert rows[0]["title"] == "入れ子の街, 夜"


# ── 書き出し先の組み立て ────────────────────────────────────


def test_out_のパターンに列を差し込む():
    assert batch.format_output("tmp/{name}.mp4", {"name": "01"}) == "tmp/01.mp4"


def test_値に区切り文字があっても掘らせない():
    # 表の値でフォルダを掘られると、書き出し先が散らばります。
    assert batch.format_output("tmp/{name}.mp4", {"name": "a/b"}) == "tmp/a-b.mp4"


def test_入れる値が無ければ使える名前を教える():
    with pytest.raises(MovoError) as caught:
        batch.format_output("tmp/{nope}.mp4", {"name": "01"})
    assert "使えるのは" in (caught.value.hint or "")


# ── 走る前に全部決める ──────────────────────────────────────


def _plan(tmp_path, rows, out, **extra):
    template = tmp_path / "mv.json"
    template.write_text("{}", encoding="utf-8")
    spec = {"target": str(template), "rows": rows, "out": out, "cwd": str(tmp_path)}
    spec.update(extra)
    return batch.build_batch_plan(spec)


def test_行ごとに_1_本の仕事になる(tmp_path):
    plan = _plan(tmp_path, [{"name": "01"}, {"name": "02"}], "tmp/{name}.mp4")
    assert [job["name"] for job in plan["jobs"]] == ["01", "02"]


def test_書き出し先がぶつかると走る前に止める(tmp_path):
    with pytest.raises(MovoError) as caught:
        _plan(tmp_path, [{"name": "01"}, {"name": "01"}], "tmp/{name}.mp4")
    assert "書き出し先が同じ" in reason_of(caught.value)


def test_params_に無い列は警告して無視する(tmp_path):
    plan = _plan(tmp_path, [{"name": "01", "bqm": 205}], "tmp/{name}.mp4", declared=["bpm"])
    assert plan["warnings"] and "bqm" in plan["warnings"][0]
    assert "bqm" not in plan["jobs"][0]["params"]


def test_出力名にしか使わない列は黙って通す(tmp_path):
    plan = _plan(tmp_path, [{"name": "01"}], "tmp/{name}.mp4", declared=["bpm"])
    assert plan["warnings"] == []


def test_continue_は書き出し済みを飛ばす(tmp_path):
    done = tmp_path / "tmp" / "01.mp4"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"already")
    plan = _plan(tmp_path, [{"name": "01"}, {"name": "02"}], "tmp/{name}.mp4", continue_existing=True)
    assert [job["name"] for job in plan["jobs"]] == ["02"]
    assert [job["name"] for job in plan["skipped"]] == ["01"]


def test_中身が空の出力は書き出し済みと見なさない(tmp_path):
    # 途中で落ちて 0 バイトのファイルが残ることがあります。
    done = tmp_path / "tmp" / "01.mp4"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"")
    plan = _plan(tmp_path, [{"name": "01"}], "tmp/{name}.mp4", continue_existing=True)
    assert len(plan["jobs"]) == 1


def test_out_を書き忘れたら書き方を教える(tmp_path):
    with pytest.raises(MovoError) as caught:
        _plan(tmp_path, [{"name": "01"}], None)
    assert "--out" in reason_of(caught.value)


# ── ワイルドカードの展開 ────────────────────────────────────


def test_ファイル名のワイルドカードを展開する(tmp_path):
    for name in ("b.json", "a.json", "c.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    found = batch.expand_targets("*.json", str(tmp_path))
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in found] == ["a.json", "b.json"]


def test_フォルダ側にワイルドカードは書けない(tmp_path):
    with pytest.raises(MovoError) as caught:
        batch.expand_targets("*/x.json", str(tmp_path))
    assert "ファイル名の部分" in reason_of(caught.value)


# ── 並べて流す ──────────────────────────────────────────────


def test_1_本失敗しても残りは続ける():
    jobs = [{"name": str(i)} for i in range(4)]

    def run(job):
        if job["name"] == "1":
            raise RuntimeError("落ちた")
        return {"code": 0}

    results = batch.run_jobs(jobs, 2, run)
    assert len(results) == 4
    assert sum(1 for r in results if r["code"] != 0) == 1
    assert any("落ちた" in (r.get("error") or "") for r in results)
