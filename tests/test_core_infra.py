"""土台まわりのテスト — エラー・キャッシュ・設定・環境・映像プロファイル。

見た目に地味ですが、**ここが緩いと «なぜか前と違う動画が出る»** に直結します。
とくにキャッシュの鍵は、入力を 1 つ取りこぼしただけで «前に作った絵が返ってくる»
という一番気付きにくい壊れ方をします。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import movo.core as core


# ── エラー ──────────────────────────────────────────────────


def test_error_carries_a_machine_readable_code_and_a_place():
    error = core.MovoError(
        core.ErrorCodes.MOVO_SCHEMA_INVALID, "だめです", file="p.json", path="layers.0", hint="こう直す"
    )
    assert error.code == "MOVO_SCHEMA_INVALID"
    block = error.format()
    assert "p.json" in block and "layers.0" in block and "こう直す" in block
    assert error.to_json()["reason"] == "だめです"


def test_validation_error_shows_every_issue():
    """**指摘はまとめて出すこと。** 1 件ずつだと «直しては再実行» の繰り返しになります。"""
    error = core.MovoValidationError(
        [{"path": "a", "message": "one"}, {"path": "b", "message": "two"}], file="p.json"
    )
    assert "one" in error.format() and "two" in error.format()
    assert error.code == core.ErrorCodes.MOVO_SCHEMA_INVALID


def test_to_movo_error_keeps_movo_errors_and_maps_memory_errors():
    original = core.MovoError(core.ErrorCodes.MOVO_CLI_USAGE, "x")
    assert core.to_movo_error(original) is original
    assert core.to_movo_error(MemoryError("out of memory")).code == core.ErrorCodes.MOVO_OUT_OF_MEMORY
    assert core.to_movo_error(ValueError("boom")).code == core.ErrorCodes.MOVO_INTERNAL


# ── ハッシュ ────────────────────────────────────────────────


def test_hashing_ignores_key_order_but_not_values():
    assert core.hash_json({"a": 1, "b": 2}) == core.hash_json({"b": 2, "a": 1})
    assert core.hash_json({"a": 1}) != core.hash_json({"a": 2})
    # 整数と小数の «見た目» の違いで鍵が変わらないこと（JS には区別がありません）
    assert core.hash_json({"fps": 30}) == core.hash_json({"fps": 30.0})


def test_hash_file_streams_instead_of_reading_everything(tmp_path):
    target = tmp_path / "big.bin"
    target.write_bytes(b"movo" * 100000)
    assert core.hash_file(target) == core.sha256(b"movo" * 100000)


# ── キャッシュ ──────────────────────────────────────────────


def test_cache_key_changes_with_every_input(tmp_path):
    cache = core.Cache(tmp_path, namespace_salt={"renderer": "1.4.0"})
    base = cache.key("frames", {"scene": 1})
    assert base != cache.key("frames", {"scene": 2})
    assert base != cache.key("masks", {"scene": 1})
    # 実装の版を変えたら一斉に失効すること
    newer = core.Cache(tmp_path, namespace_salt={"renderer": "1.5.0"})
    assert base != newer.key("frames", {"scene": 1})


def test_cache_round_trip_and_stats(tmp_path):
    cache = core.Cache(tmp_path)
    key = cache.key("frames", {"i": 0})
    assert cache.read_buffer("frames", key) is None
    cache.write_buffer("frames", key, b"pixels")
    assert cache.has("frames", key)
    assert cache.read_buffer("frames", key) == b"pixels"
    cache.write_json("meta", key, {"a": 1})
    assert cache.read_json("meta", key) == {"a": 1}
    assert cache.stats["writes"] == 2 and cache.stats["hits"] == 2
    assert cache.size() > 0
    cache.clear()
    assert cache.read_buffer("frames", key) is None


def test_disabled_cache_touches_nothing(tmp_path):
    cache = core.Cache(tmp_path, enabled=False)
    assert cache.write_buffer("frames", "k", b"x") is None
    assert cache.read_buffer("frames", "k") is None
    assert cache.has("frames", "k") is False
    assert not any(tmp_path.iterdir())


def test_cache_writes_are_atomic(tmp_path):
    """**書きかけのファイルを他のプロセスに読ませないこと。**

    並列レンダリング中に半端なフレームを読むと、原因の分からない
    «たまに壊れる» になります。一時ファイルに書いてから差し替えます。
    """
    cache = core.Cache(tmp_path)
    cache.write_buffer("frames", "k", b"x" * 1000)
    files = [p.name for p in (tmp_path / "frames").iterdir()]
    assert files == ["k.bin"]  # .tmp が残っていないこと


def test_memo_keeps_values_in_process(tmp_path):
    cache = core.Cache(tmp_path)
    calls = []

    def factory():
        calls.append(1)
        return 42

    assert cache.memo("k", factory) == 42
    assert cache.memo("k", factory) == 42
    assert len(calls) == 1


# ── 設定 ────────────────────────────────────────────────────


@pytest.fixture()
def movo_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVO_HOME", str(tmp_path))
    return tmp_path


def test_config_round_trip(movo_home):
    core.set_config_value("openai.apiKey", "sk-secret-value-1234")
    assert core.get_config_value("openai.apiKey") == "sk-secret-value-1234"
    assert core.get_config_value("missing.key", "fallback") == "fallback"
    core.unset_config_value("openai.apiKey")
    assert core.get_config_value("openai.apiKey") is None


def test_environment_variables_win_over_the_file(movo_home, monkeypatch):
    """CI やコンテナで **ファイルを置かずに** 渡せること。"""
    core.set_config_value("openai.apiKey", "from-file")
    monkeypatch.setenv("MOVO_OPENAI_API_KEY", "from-env")
    assert core.get_config_value("openai.apiKey") == "from-env"


def test_secrets_are_masked_when_listed(movo_home):
    """**秘密をそのまま画面に出さないこと。** 両端だけ残して見分けはつくようにします。"""
    core.set_config_value("openai.apiKey", "sk-abcdefghijklmnop")
    core.set_config_value("render.fps", 30)
    listed = {row["key"]: row["value"] for row in core.list_config()}
    assert listed["render.fps"] == 30
    assert listed["openai.apiKey"].startswith("sk-a")
    assert "*" in listed["openai.apiKey"]
    assert "cdefghijklm" not in listed["openai.apiKey"]


def test_is_secret_key_matches_the_usual_names():
    for key in ("openai.apiKey", "x.token", "y.SECRET", "z.password"):
        assert core.is_secret_key(key), key
    assert not core.is_secret_key("render.fps")


def test_config_file_is_written_with_tight_permissions(movo_home):
    """**書いてから権限を絞らないこと。** その一瞬だけ他人に読める状態ができます。"""
    path = core.set_config_value("openai.apiKey", "x")
    assert os.path.exists(path)
    if os.name != "nt":  # Windows の権限は別物なので見ません
        assert oct(os.stat(path).st_mode)[-3:] == "600"


def test_broken_config_is_treated_as_empty(movo_home):
    (movo_home / "config.json").write_text("{ broken", encoding="utf-8")
    assert core.load_config() == {}


# ── 環境 ────────────────────────────────────────────────────


def test_project_paths_are_written_with_forward_slashes(tmp_path):
    """**JSON には常に ``/`` で書くこと。** Windows で作った作品が macOS でも動くように。"""
    absolute = core.resolve_project_path(tmp_path, "assets\\logo.png")
    assert os.path.isabs(absolute)
    assert core.to_project_relative(tmp_path, absolute) == "assets/logo.png"
    assert core.resolve_project_path(tmp_path, None) == str(tmp_path)


def test_absolute_paths_pass_through(tmp_path):
    absolute = str(tmp_path / "x.png")
    assert core.resolve_project_path("/elsewhere", absolute) == os.path.normpath(absolute)


def test_temp_dirs_do_not_collide():
    a = core.temp_dir("test")
    b = core.temp_dir("test")
    assert a != b
    assert os.path.isdir(a) and os.path.isdir(b)
    os.rmdir(a)
    os.rmdir(b)


def test_environment_summary_has_the_expected_shape():
    described = core.describe_environment()
    assert set(described) == {"movoPython", "os", "platform", "arch", "cpus", "memoryGB", "home"}
    assert described["cpus"] >= 1


def test_cpu_count_is_at_least_one():
    assert core.cpu_count() >= 1


# ── 版 ──────────────────────────────────────────────────────


def test_json_dialect_compatibility():
    assert core.is_compatible_json_version("1.0")
    assert core.is_compatible_json_version("1.4.2")
    assert core.is_compatible_json_version(None)
    assert not core.is_compatible_json_version("2.0")


def test_component_versions_cannot_be_edited_by_accident():
    """**キャッシュ鍵の素を書き換えられないこと。**

    うっかり書き換えると «前に作った動画と違う絵が出る» という、
    いちばん気付きにくい壊れ方をします。
    """
    with pytest.raises(TypeError):
        core.COMPONENT_VERSIONS["renderer"] = "9.9.9"


# ── ログ ────────────────────────────────────────────────────


def test_warnings_are_collected_even_when_not_shown():
    """**流れた警告を最後にまとめて出せること。** レンダリングは数分かかります。"""
    logger = core.Logger("silent")
    logger.warn("これは見えないが溜まる")
    assert logger.warnings == ["これは見えないが溜まる"]


def test_log_levels_filter():
    logger = core.Logger("error")
    assert logger._should("error")
    assert not logger._should("info")
    logger.set_level("debug")
    assert logger._should("debug")


# ── 映像プロファイル ────────────────────────────────────────


def test_profiler_counts_cuts_at_hard_changes():
    profiler = core.VideoProfiler(width=64, height=36, fps=30)
    for i in range(30):
        profiler.push(core.Bitmap.create(64, 36, "#000000" if i < 15 else "#ffffff"))
    report = profiler.report()
    assert report["cuts"]["count"] == 2  # 最初の 1 本 + 切り替わり
    assert report["frames"] == 30


def test_profiler_sees_a_still_image_as_still():
    profiler = core.VideoProfiler(width=64, height=36, fps=30)
    frame = core.Bitmap.create(64, 36, "#336699")
    for _ in range(20):
        profiler.push(frame)
    report = profiler.report()
    assert report["motion"]["stillRatio"] == 1.0
    assert report["cuts"]["count"] == 1


def test_comparison_gives_a_direction_and_advice():
    """**外れたときに «どちらへ» 直すかが分かること。** 差分だけでは動けません。"""
    profile = {"cuts": {"medianSeconds": 0.2}, "palette": {"effectiveColors": 400}}
    result = core.compare_profile(profile, {"cutSeconds": [3, 8], "colors": [2, 10]})
    assert result["ok"] is False
    by_key = {r["key"]: r for r in result["results"]}
    assert by_key["cutSeconds"]["status"] == "low"
    assert by_key["colors"]["status"] == "high"
    assert "posterize" in by_key["colors"]["advice"]
    assert core.describe_comparison(result)


def test_comparison_ignores_metrics_the_target_does_not_mention():
    profile = {"cuts": {"medianSeconds": 5}}
    assert core.compare_profile(profile, {})["results"] == []
    assert core.compare_profile(profile, {"cutSeconds": "not a range"})["results"] == []


def test_compare_to_reference_builds_a_band_around_the_other_video():
    reference = {"cuts": {"medianSeconds": 4.0}, "palette": {"effectiveColors": 20}}
    mine = {"cuts": {"medianSeconds": 4.2}, "palette": {"effectiveColors": 60}}
    result = core.compare_to_reference(mine, reference)
    by_key = {r["key"]: r for r in result["results"]}
    assert by_key["cutSeconds"]["status"] == "ok"  # ±25% の中
    assert by_key["colors"]["status"] == "high"


# ── 閃光検査 ────────────────────────────────────────────────


def test_a_calm_video_passes():
    guard = core.FlashGuard(width=64, height=36, fps=30)
    for i in range(60):
        guard.push(core.Bitmap.create(64, 36, f"#{i:02x}{i:02x}{i:02x}"))
    report = guard.report()
    assert report["ok"] is True
    assert report["totalFlashes"] == 0


def test_a_strobe_is_caught():
    """毎フレーム白黒が入れ替わる映像は **危険と判定されること**。"""
    guard = core.FlashGuard(width=64, height=36, fps=30)
    for i in range(60):
        guard.push(core.Bitmap.create(64, 36, "#ffffff" if i % 2 == 0 else "#000000"))
    report = guard.report()
    assert report["ok"] is False
    assert report["flashesPerSecond"] > report["limit"]
    assert any("危険" in line for line in core.describe_flash_report(report))


def test_white_to_white_is_not_a_flash():
    """**明るい同士の変化は数えないこと。** 白から白は «閃光» として見えません。"""
    guard = core.FlashGuard(width=64, height=36, fps=30)
    for i in range(60):
        guard.push(core.Bitmap.create(64, 36, "#ffffff" if i % 2 == 0 else "#f0f0f0"))
    assert guard.report()["totalFlashes"] == 0


def test_none_overrides_do_not_wipe_out_the_defaults():
    """**指定なし（``None``）で閾値を潰さないこと。**

    潰れると比較がすべて False になり、«閃光を 1 つも検出しないのに毎回
    危険と警告する» という、いちばん困る壊れ方をします。
    """
    guard = core.FlashGuard(width=64, height=36, fps=30, maxPerSecond=None, luminanceDelta=None)
    assert guard.settings == dict(core.FLASH_DEFAULTS)
    explicit = core.FlashGuard(width=64, height=36, fps=30, maxPerSecond=1)
    assert explicit.settings["maxPerSecond"] == 1


def test_one_white_frame_counts_as_one_flash_not_two():
    """1 往復（明→暗）は **1 回**。向きが変わるたびに数えると倍に見えます。"""
    guard = core.FlashGuard(width=64, height=36, fps=30)
    frames = ["#000000", "#ffffff", "#000000", "#000000", "#000000"]
    for colour in frames:
        guard.push(core.Bitmap.create(64, 36, colour))
    assert guard.report()["totalFlashes"] == 1


# ── プロファイルの置き場 ────────────────────────────────────


def test_project_profiles_override_the_builtin_ones(tmp_path):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "mine.json").write_text(
        json.dumps({"label": "自作", "target": {"cutSeconds": [1, 2]}}), encoding="utf-8"
    )
    names = [entry["name"] for entry in core.list_profiles(tmp_path)]
    assert "mine" in names
    loaded = core.load_profile_target("mine", tmp_path)
    assert loaded["target"] == {"cutSeconds": [1, 2]}


def test_a_file_path_wins_over_a_name(tmp_path):
    """``./x.json`` と名前 ``x`` がぶつかったら **手元のファイルを優先**すること。"""
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"target": {"colors": [1, 2]}}), encoding="utf-8")
    assert core.load_profile_target(str(path))["target"] == {"colors": [1, 2]}


def test_an_unknown_profile_lists_the_available_ones(tmp_path):
    with pytest.raises(core.MovoError) as info:
        core.load_profile_target("does-not-exist", tmp_path)
    assert info.value.code == core.ErrorCodes.MOVO_ASSET_NOT_FOUND
    assert info.value.hint is not None


def test_a_broken_profile_file_is_reported_with_its_path(tmp_path):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "bad.json").write_text("{ nope", encoding="utf-8")
    with pytest.raises(core.MovoError) as info:
        core.list_profiles(tmp_path)
    assert "bad.json" in info.value.reason


# ── 数学 ────────────────────────────────────────────────────


def test_js_round_goes_away_from_zero_on_ties():
    """**Python の ``round`` は «偶数丸め»** なので、JS と揃えるために別に持ちます。"""
    assert core.js_round(0.5) == 1
    assert core.js_round(1.5) == 2
    assert core.js_round(2.5) == 3  # Python の round(2.5) は 2
    assert core.js_round(-0.5) == 0


def test_small_math_helpers():
    assert core.clamp(5, 0, 1) == 1
    assert core.lerp(0, 10, 0.25) == 2.5
    assert core.inverse_lerp(0, 10, 2.5) == 0.25
    assert core.inverse_lerp(3, 3, 5) == 0
    assert core.smoothstep(0, 1, 0.5) == 0.5
    assert core.approximately(0.1 + 0.2, 0.3)
    assert core.to_degrees(core.to_radians(90)) == pytest.approx(90)


def test_singular_matrices_return_none_instead_of_raising():
    """``scaleX: 0`` のレイヤーは普通にあるので、例外ではなく None を返します。"""
    assert core.Mat2D.invert([0, 0, 0, 0, 0, 0]) is None
    assert core.solve2x2(1, 1, 1, 1, 1, 1) is None
    assert core.solve2x2(1, 0, 0, 1, 3, 4) == (3, 4)


def test_polyline_sampling_stays_inside_the_points():
    points = [(0, 0), (10, 0), (10, 10)]
    assert core.sample_polyline(points, 0) == [0, 0]
    assert core.sample_polyline(points, 1) == pytest.approx([10, 10])
    assert core.sample_polyline([], 0.5) == [0, 0]
    assert core.sample_polyline([(3, 4)], 0.5) == [3, 4]


# ── 画像の入出力 ────────────────────────────────────────────


def test_save_image_creates_parent_folders(tmp_path):
    target = tmp_path / "a" / "b" / "out.png"
    core.save_image(core.Bitmap.create(4, 4, "#123456"), target)
    assert target.exists()
    assert np.array_equal(core.load_image(target).data, core.Bitmap.create(4, 4, "#123456").data)


def test_load_image_reports_a_missing_file_clearly(tmp_path):
    with pytest.raises(core.MovoError) as info:
        core.load_image(tmp_path / "nope.png")
    assert info.value.code == core.ErrorCodes.MOVO_ASSET_NOT_FOUND
