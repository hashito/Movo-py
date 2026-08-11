"""並列レンダリング。**速度の要なので、割り方の性質を数で押さえます。**

ここで守っているのは 4 つです。

1. フレーム境界で割る（重複も欠けも出さない）
2. 助走が要るときだけ助走する
3. 割れないときは «理由を言って» 1 本に落とす
4. Numba のキャッシュ先が子プロセスと共有できる場所を向く
"""

import json
import os

import pytest

from movo.cli import parallel


# ── 区間の割り方 ────────────────────────────────────────────


@pytest.mark.parametrize("total,jobs", [(100, 4), (101, 4), (99, 7), (1000, 12), (16, 2), (37, 5)])
def test_区間は重複も欠けもなく全フレームを覆う(total, jobs):
    chunks = parallel.plan_chunks(0, total - 1, jobs)
    covered = []
    for chunk in chunks:
        covered.extend(range(chunk["startFrame"], chunk["endFrame"] + 1))
    assert covered == list(range(total)), "秒で割ると端数のフレームが重複・欠落します"


def test_区間の長さは_1_フレームまでしか違わない():
    # 最後の区間だけ極端に短いと、そこだけ早く終わって 1 コアが遊びます。
    chunks = parallel.plan_chunks(0, 100, 7)
    sizes = [chunk["frames"] for chunk in chunks]
    assert max(sizes) - min(sizes) <= 1


def test_短い動画は細切れにしない():
    # 1 区間 8 枚を下回るほど刻んでも、起動の分だけ損をします。
    chunks = parallel.plan_chunks(0, 19, 12)
    assert len(chunks) == 2
    assert all(chunk["frames"] >= parallel.MIN_FRAMES_PER_CHUNK for chunk in chunks)


def test_開始フレームがずれていても覆う():
    chunks = parallel.plan_chunks(300, 399, 3)
    assert chunks[0]["startFrame"] == 300
    assert chunks[-1]["endFrame"] == 399


# ── --jobs の読み方 ────────────────────────────────────────


def test_jobs_の既定は_1_本():
    assert parallel.resolve_job_count(None) == 1
    assert parallel.resolve_job_count(False) == 1


def test_auto_はコア数から_1_を引く():
    cores = max(1, os.cpu_count() or 1)
    assert parallel.resolve_job_count("auto") == max(1, cores - 1)
    # 値を書かずに `--jobs` だけ渡すと True になります。auto と同じ扱いです。
    assert parallel.resolve_job_count(True) == max(1, cores - 1)


def test_読めない値は_1_本に落とす():
    assert parallel.resolve_job_count("たくさん") == 1
    assert parallel.resolve_job_count(0) == 1
    assert parallel.resolve_job_count(-3) == 1


def test_上限は_64():
    assert parallel.resolve_job_count(500) == 64


# ── 助走（warmup）──────────────────────────────────────────


def test_残像を使わないなら助走は要らない():
    project = {"scenes": [{"layers": [{"type": "text"}]}]}
    assert parallel.warmup_frames_for(project) == 0


def test_frameEcho_を使うと助走が要る():
    project = {"scenes": [{"layers": [{"frameEcho": {"count": 4}}]}], "render": {"frameHistory": 16}}
    assert parallel.warmup_frames_for(project) == 16


def test_slitScan_も助走が要る():
    project = {"scenes": [{"layers": [{"effects": [{"type": "slitScan"}]}]}]}
    assert parallel.warmup_frames_for(project) == 16


def test_助走は_240_フレームで頭打ち():
    project = {"scenes": [{"layers": [{"frameEcho": {}}]}], "render": {"frameHistory": 9999}}
    assert parallel.warmup_frames_for(project) == 240


# ── 割れない理由 ────────────────────────────────────────────


class _FakeSession(dict):
    def __init__(self, scenes=None, file=__file__):
        super().__init__(
            file=file,
            projectRoot=os.path.dirname(file),
            project={"scenes": scenes or []},
            timeline={"width": 1280, "height": 720, "fps": 30},
            assets=None,
            audio=None,
            cache=None,
        )


def _blockers(session, **context):
    base = {"format": "mp4", "jobs": 8, "frames": 1000, "outputs": 1}
    base.update(context)
    return parallel.parallel_blockers(session, base)


def test_gif_は割れない():
    reasons = _blockers(_FakeSession(), format="gif")
    assert any("gif" in reason for reason in reasons)


def test_追跡線がある場合は割らない():
    # 軌跡は «描き始めからずっと» 積み上がるので、助走では再現できません。
    scenes = [{"layers": [{"type": "linePath", "followLayer": "ball"}]}]
    reasons = _blockers(_FakeSession(scenes))
    assert any("followLayer" in reason for reason in reasons)


def test_フレームが少なすぎると割らない():
    reasons = _blockers(_FakeSession(), frames=10)
    assert any("フレームが" in reason for reason in reasons)


def test_何通りも書き出す指定には未対応():
    reasons = _blockers(_FakeSession(), outputs=2)
    assert any("output の配列" in reason for reason in reasons)


def test_jobs_が_1_なら理由に挙がる():
    reasons = _blockers(_FakeSession(), jobs=1)
    assert any("--jobs" in reason for reason in reasons)


def test_プロジェクトファイルが無いと子に渡せない():
    reasons = _blockers(_FakeSession(file="存在しない.json"))
    assert any("プロジェクトファイル" in reason for reason in reasons)


def test_割れる条件がそろえば理由は空():
    assert _blockers(_FakeSession()) == []


# ── メモリの見積もり ────────────────────────────────────────


def test_見積もりは解像度に応じて増える():
    small = parallel.estimate_bytes_per_job(_FakeSession())
    big = _FakeSession()
    big["timeline"] = {"width": 3840, "height": 2160, "fps": 30}
    assert parallel.estimate_bytes_per_job(big) > small


def test_空きが測れない環境では並列数を減らさない(monkeypatch):
    # 測れないことを理由に «勝手に遅くする» のはいちばん困ります。
    monkeypatch.setattr(parallel, "available_memory_bytes", lambda: 0)
    assert parallel.limit_jobs_by_memory(_FakeSession(), 12)["jobs"] == 12


def test_空きが足りなければ理由つきで減らす(monkeypatch):
    monkeypatch.setattr(parallel, "available_memory_bytes", lambda: 600 * 1024 * 1024)
    result = parallel.limit_jobs_by_memory(_FakeSession(), 12)
    assert result["jobs"] < 12
    assert "空きメモリ" in result["warning"]


# ── 子プロセスに引き継ぐ指定 ────────────────────────────────


def test_絵が変わる指定は子に引き継ぐ():
    # ここに挙げ忘れた指定は子に伝わらず、**繋ぎ目で絵が変わります**。
    options = parallel.child_render_options(
        {"quality": "high", "seed": 42, "superSample": 2, "set": ["art=b.png"], "noCache": True}
    )
    assert options["quality"] == "high"
    assert options["seed"] == 42
    assert options["super_sample"] == 2
    assert options["set"] == ["art=b.png"]
    assert options["no_cache"] is True


# ── Numba のキャッシュ ──────────────────────────────────────


def test_キャッシュ先は書ける場所を向く(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVO_NUMBA_CACHE_DIR", str(tmp_path / "jit"))
    assert parallel.numba_cache_dir() == tmp_path / "jit"


def test_キャッシュ先を用意すると環境変数に入る(tmp_path, monkeypatch):
    # 子プロセスは環境変数を引き継ぐので、ここで揃えれば 1 回のコンパイルを
    # 全員で使い回せます（12 並列 × 1 秒が丸ごと消えます）。
    monkeypatch.setenv("MOVO_NUMBA_CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    directory = parallel.prepare_numba_cache()
    assert directory == str(tmp_path / "jit")
    assert os.environ["NUMBA_CACHE_DIR"] == directory


def test_区間の書き出しは大きさで裏を取る(tmp_path):
    # 終了コードだけを信じると «エラーも出ないのに 1/4 の長さの動画» ができます。
    empty = tmp_path / "part-000.mp4"
    empty.write_bytes(b"")
    assert parallel.chunk_output_looks_sane(str(empty)) is False
    filled = tmp_path / "part-001.mp4"
    filled.write_bytes(b"\x00" * 10)
    assert parallel.chunk_output_looks_sane(str(filled)) is True
    assert parallel.chunk_output_looks_sane(str(tmp_path / "ない.mp4")) is False


def test_割れないときは理由を言って_1_本に落とす(tmp_path, monkeypatch, capsys):
    """**黙って壊れず、黙って遅くもならない** のがこの分岐の役目です。

    使えないと分かったときに例外を投げると «gif だと必ず失敗する CLI» になり、
    黙って 1 本にすると «--jobs を付けたのに速くならない» 理由が分かりません。
    理由を読み上げてから 1 本に落とします。
    """
    from movo.cli import pipeline

    project_file = tmp_path / "movo.json"
    project_file.write_text("{}", encoding="utf-8")
    session = _FakeSession(file=str(project_file))
    session["project"]["output"] = {"format": "gif"}

    calls = []
    monkeypatch.setattr(pipeline, "render_video", lambda s, o: calls.append(o) or {"path": "x"})
    monkeypatch.setattr(
        pipeline, "resolve_range", lambda timeline, options: {"startFrame": 0, "endFrame": 299}
    )
    monkeypatch.setattr(
        pipeline, "resolve_output_path", lambda **kwargs: str(tmp_path / "out.gif")
    )
    monkeypatch.setattr(parallel.bridge, "pick", lambda *a, **k: (lambda fmt: {"format": "gif"}))

    result = parallel.render_video_parallel(session, {"jobs": 8, "format": "gif"})
    assert result == {"path": "x"}, "1 本で描いた結果をそのまま返すこと"
    assert calls, "render_video に落ちていること"
    assert "gif" in capsys.readouterr().err, "理由を読み上げていること"


def test_区間の秒への直し方が全フレームを覆う():
    # 子には秒で渡すので、«次のフレームの頭» を終わりにしないと 1 枚落ちます。
    fps = 30
    chunks = parallel.plan_chunks(0, 89, 3)
    edges = [(c["startFrame"] / fps, (c["endFrame"] + 1) / fps) for c in chunks]
    for (_, end), (next_start, _) in zip(edges, edges[1:]):
        assert end == pytest.approx(next_start)
    assert edges[0][0] == 0
    assert edges[-1][1] == pytest.approx(90 / fps)


# ── インラインのプロジェクト（make-mv / skill render）────────────


def test_プロジェクトファイルが無ければ割らない():
    session = _FakeSession(file=os.path.join(os.path.dirname(__file__), "存在しない.json"))
    reasons = _blockers(session)
    assert any("プロジェクトファイル" in reason for reason in reasons)


def test_インラインのプロジェクトは割れる():
    """`make-mv` はファイルではなく組み立てた JSON を描きます。

    ファイルが無いことを理由に 1 本へ落とすと、いちばん長い MV でこそ
    並列が効かない、という逆の結果になります。
    """
    session = _FakeSession(file=os.path.join(os.path.dirname(__file__), "存在しない.json"))
    session["inlineProject"] = {"scenes": []}
    assert _blockers(session) == []


def test_出力が残っていてもインラインなら誤判定しない():
    """実際に踏んだ壊れ方の再現。

    インラインのとき `session["file"]` には **出力先（.mp4）** が入ります。
    «存在するか» だけで判定していたため、前に書き出した動画が残っていると
    判定をすり抜け、子プロセスが .mp4 を JSON として読んで
    `'utf-8' codec can't decode byte 0xaf` で落ちました。
    """
    # このテストファイル自身を «残っている出力» の代わりにする（確かに存在する）
    session = _FakeSession(file=__file__)
    session["inlineProject"] = {"scenes": []}
    reasons = _blockers(session)
    assert not any("プロジェクトファイル" in reason for reason in reasons)
