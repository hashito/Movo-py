"""`movo make-mv` の «構成の組み立て» を押さえる（描画はしません）。

見ているもの:

  - 区間をムービースキルの並びへ割り当てる（`plan_sequence`）
  - **時刻つきの歌詞を、実時間でカットへ配る**（`distribute_timed_lines`）
  - 歌詞の無いカットに «歌詞なし» を明示する（`build_sequence` の authoritative）

3 つ目は実際に踏んだ壊れ方です。空を «指定なし» と見なしていたため、
テンプレートの `${lines}` が生き残り、2 小節の間奏に曲の全 28 行が落ちました。
"""

from __future__ import annotations

import pytest

from movo.cli.commands.make_mv import (
    build_sequence, distribute_timed_lines, plan_sequence,
)

BAR = 60 / 120 * 4  # 120BPM・4 拍子 → 1 小節 2 秒

TEMPLATE = [
    {"scene": "mv-intro", "with": {"title": "${title}", "bars": "${introBars}"}},
    {"scene": "mv-verse", "with": {"lines": "${lines}", "bars": "${verseBars}"}},
    {"scene": "mv-chorus", "with": {"hook": "${hook}", "bars": "${chorusBars}"}},
    {"scene": "mv-outro", "with": {"title": "${title}", "bars": "${outroBars}"}},
]


# ── 構成 ──────────────────────────────────────────────────────────


def test_plan_sequence_uses_a_fixed_shape_when_the_song_is_flat():
    """区間が 2 つ以下 ＝ 起伏が取れなかった、ということ。"""
    sections = [{"start": 0, "end": 48, "energy": 0.5, "label": "verse"}]
    plan = plan_sequence(sections, 120, 4)
    assert [item["scene"] for item in plan] == ["mv-intro", "mv-verse", "mv-chorus", "mv-outro"]
    assert sum(item["bars"] for item in plan) == 24


def test_plan_sequence_puts_the_peak_at_the_chorus():
    sections = [
        {"start": 0, "end": 8, "energy": 0.2, "label": "intro"},
        {"start": 8, "end": 24, "energy": 0.6, "label": "verse"},
        {"start": 24, "end": 40, "energy": 1.0, "label": "chorus"},
        {"start": 40, "end": 48, "energy": 0.3, "label": "outro"},
    ]
    scenes = [item["scene"] for item in plan_sequence(sections, 120, 4)]
    assert scenes[0] == "mv-intro"
    assert "mv-chorus" in scenes
    assert scenes[-1] == "mv-outro"


def test_plan_sequence_splits_long_sections_into_cuts():
    """96 小節を 1 カットで通す MV はありません。"""
    sections = [
        {"start": 0, "end": 4, "energy": 0.2, "label": "intro"},
        {"start": 4, "end": 68, "energy": 1.0, "label": "chorus"},
        {"start": 68, "end": 72, "energy": 0.2, "label": "outro"},
    ]
    plan = plan_sequence(sections, 120, 4, {"maxBars": 4})
    chorus = [item for item in plan if item["scene"] == "mv-chorus"]
    assert len(chorus) > 1
    assert all(item["bars"] <= 5 for item in chorus)


# ── 時刻つきの歌詞を配る ──────────────────────────────────────────


TIMED = [
    {"text": "いちぎょうめ", "at": 4.0, "for": 2.0},
    {"text": "にぎょうめ", "at": 6.0, "for": 2.0},
    {"text": "さんぎょうめ", "at": 20.0, "for": 2.0},
]
PLAN = [
    {"scene": "mv-intro", "bars": 2, "start": 0.0},    # 0〜4 秒（歌なし）
    {"scene": "mv-verse", "bars": 4, "start": 4.0},    # 4〜12 秒（2 行）
    {"scene": "mv-verse", "bars": 4, "start": 12.0},   # 12〜20 秒（歌なし）
    {"scene": "mv-chorus", "bars": 2, "start": 20.0},  # 20〜24 秒（1 行）
]


def test_distribute_timed_lines_puts_each_line_in_the_cut_it_is_sung_in():
    groups = distribute_timed_lines(TIMED, PLAN, BAR)
    assert [[row["text"] for row in group] for group in groups] == [
        [], ["いちぎょうめ", "にぎょうめ"], [], ["さんぎょうめ"],
    ]


def test_distribute_timed_lines_keeps_the_timing():
    """**文字列にしない。** ここで時刻を捨てると、使える唯一の場所の直前で消えます。"""
    groups = distribute_timed_lines(TIMED, PLAN, BAR)
    assert all(isinstance(row, dict) for row in groups[1])
    # 時刻は «そのシーンの中の秒» に直っている（曲頭からの 4.0 秒 → 0.0 秒）
    assert groups[1][0]["at"] == pytest.approx(0.0)
    assert groups[1][1]["at"] == pytest.approx(2.0)


def test_distribute_timed_lines_keeps_lines_that_straddle_a_cut():
    """歌はカットの切れ目とは無関係に続きます。またぐ行を落とすと 1 行消えます。"""
    straddling = [{"text": "またぐ", "at": 10.0, "for": 4.0}]  # 10〜14 秒
    groups = distribute_timed_lines(straddling, PLAN, BAR)
    assert [row["text"] for row in groups[1]] == ["またぐ"]  # 4〜12 秒のカット
    assert [row["text"] for row in groups[2]] == ["またぐ"]  # 12〜20 秒のカット


# ── 歌詞の無いカット ──────────────────────────────────────────────


def test_build_sequence_states_no_lyrics_for_silent_cuts_when_authoritative():
    """実際に踏んだ壊れ方: 空を «指定なし» と見なすと全歌詞が落ちてくる。"""
    groups = distribute_timed_lines(TIMED, PLAN, BAR)
    sequence = build_sequence(PLAN, TEMPLATE, groups, authoritative=True)
    assert sequence[0]["with"]["lines"] == []
    assert sequence[2]["with"]["lines"] == []
    assert [row["text"] for row in sequence[1]["with"]["lines"]] == ["いちぎょうめ", "にぎょうめ"]


def test_build_sequence_leaves_the_template_alone_without_timed_lyrics():
    """時刻が無いときは今までどおり。テンプレートの既定に任せます。"""
    sequence = build_sequence(PLAN, TEMPLATE, [[], [], [], []])
    # mv-intro はそもそも lines を持たない。mv-verse は «テンプレートのまま» 残る。
    assert "lines" not in sequence[0]["with"]
    assert sequence[1]["with"]["lines"] == "${lines}"


def test_build_sequence_borrows_the_template_values_per_scene():
    """借りずに作り直すと «無題» とスキルの既定の歌詞が並ぶ MV になります。"""
    sequence = build_sequence(PLAN, TEMPLATE, [[], [], [], []])
    assert sequence[0]["with"]["title"] == "${title}"      # mv-intro
    assert sequence[3]["with"]["hook"] == "${hook}"        # mv-chorus
    # bars は実際の構成で上書きされる
    assert sequence[0]["with"]["bars"] == 2
    assert sequence[1]["with"]["bars"] == 4


def test_build_sequence_joins_plain_string_lines():
    sequence = build_sequence(PLAN, TEMPLATE, [[], ["あ", "い"], [], []])
    assert sequence[1]["with"]["lines"] == "あ\nい"


# ── 画像などの素材（--asset）──────────────────────────────────────


def test_parse_asset_assignments_guesses_the_kind_from_the_extension(tmp_path):
    from movo.cli.commands.make_mv import parse_asset_assignments

    png = tmp_path / "art.png"
    png.write_bytes(b"x")
    wav = tmp_path / "track.wav"
    wav.write_bytes(b"x")
    got = parse_asset_assignments([f"art={png}", f"bgm={wav}"])
    assert got["art"]["type"] == "image"
    assert got["bgm"]["type"] == "audio"
    assert got["art"]["path"].endswith("art.png")


def test_parse_asset_assignments_rejects_missing_files(tmp_path):
    from movo.cli.commands.make_mv import parse_asset_assignments
    from movo.cli.errors import MovoError

    with pytest.raises(MovoError):
        parse_asset_assignments([f"art={tmp_path / 'ない.png'}"])


def test_parse_asset_assignments_needs_an_equals_sign():
    from movo.cli.commands.make_mv import parse_asset_assignments
    from movo.cli.errors import MovoError

    with pytest.raises(MovoError):
        parse_asset_assignments(["art"])


# ── スキルごとにシーン名が違う ────────────────────────────────────


RICH_TEMPLATE = [
    {"scene": "rich-intro", "with": {"title": "${title}", "artAsset": "${artAsset}"}},
    {"scene": "rich-verse", "with": {"lines": "${verseLines}", "artAsset": "${artAsset}"}},
    {"scene": "rich-chorus", "with": {"lines": "${chorusLines}", "artAsset": "${artAsset}"}},
    {"scene": "rich-burst", "with": {"lines": "${chorusLines}", "artAsset": "${artAsset}"}},
    {"scene": "rich-outro", "with": {"title": "${title}", "artAsset": "${artAsset}"}},
]


def test_scene_roles_reads_the_role_from_the_suffix():
    from movo.cli.commands.make_mv import scene_roles

    assert scene_roles(RICH_TEMPLATE) == {
        "intro": "rich-intro", "verse": "rich-verse",
        "chorus": "rich-chorus", "burst": "rich-burst", "outro": "rich-outro",
    }


def test_resolve_scene_maps_mv_names_onto_the_skills_own_names():
    from movo.cli.commands.make_mv import resolve_scene, scene_roles

    roles = scene_roles(RICH_TEMPLATE)
    assert resolve_scene("mv-chorus", roles) == "rich-chorus"
    assert resolve_scene("mv-intro", roles) == "rich-intro"
    # hype が無いスキルでは **chorus** を先に選ぶ。burst は «1 回だけ» 差す崩しなので、
    # ここへ落とすと盛り上がりのカットが全部 duotone + invert の紫になる。
    assert resolve_scene("mv-hype", roles) == "rich-chorus"


def test_resolve_scene_falls_back_to_chorus_without_a_hype_scene():
    from movo.cli.commands.make_mv import resolve_scene, scene_roles

    roles = scene_roles(TEMPLATE)  # mv-intro / mv-verse / mv-chorus / mv-outro
    assert resolve_scene("mv-hype", roles) == "mv-chorus"


def test_build_sequence_keeps_every_input_for_a_skill_with_other_scene_names():
    """実際に踏んだ壊れ方: rich-mv を指定すると 8 カット中 5 カットの with が空になった。"""
    plan = [
        {"scene": "mv-intro", "bars": 2, "start": 0.0},
        {"scene": "mv-verse", "bars": 4, "start": 4.0},
        {"scene": "mv-chorus", "bars": 4, "start": 12.0},
        {"scene": "mv-hype", "bars": 4, "start": 20.0},
        {"scene": "mv-outro", "bars": 2, "start": 28.0},
    ]
    sequence = build_sequence(plan, RICH_TEMPLATE)
    assert [item["scene"] for item in sequence] == [
        "rich-intro", "rich-verse", "rich-chorus", "rich-chorus", "rich-outro",
    ]
    # どのカットにも «画像の指定» が残っている（既定値に戻っていない）
    assert all(item["with"].get("artAsset") == "${artAsset}" for item in sequence)


# ── 決め文句（hook）──────────────────────────────────────────────


HOOK_TEMPLATE = [
    {"scene": "mv-intro", "with": {"title": "${title}", "bars": "${introBars}"}},
    {"scene": "mv-chorus", "with": {"hook": "${hook != '' ? hook : lines[0]}", "bars": "${chorusBars}"}},
]


def test_build_sequence_sets_the_hook_from_the_lines_actually_sung():
    """実際に踏んだ壊れ方: サビが 8 小節ずっとスキル既定の «ぐるぐる» のままだった。

    `hook` を使うシーンは `lines` を見ないので、時刻つきの歌詞を渡しても
    決め文句だけ既定に取り残されます。
    """
    plan = [
        {"scene": "mv-intro", "bars": 2, "start": 0.0},
        {"scene": "mv-chorus", "bars": 4, "start": 4.0},
    ]
    groups = [[], [{"text": "木だと思ったけど 草だった", "at": 0.0, "for": 2.0}]]
    sequence = build_sequence(plan, HOOK_TEMPLATE, groups, authoritative=True)
    assert sequence[1]["with"]["hook"] == "木だと思ったけど 草だった"


def test_build_sequence_leaves_the_hook_alone_for_cuts_without_lyrics():
    plan = [
        {"scene": "mv-intro", "bars": 2, "start": 0.0},
        {"scene": "mv-chorus", "bars": 4, "start": 4.0},
    ]
    sequence = build_sequence(plan, HOOK_TEMPLATE, [[], []], authoritative=True)
    assert sequence[1]["with"]["hook"] == "${hook != '' ? hook : lines[0]}"
