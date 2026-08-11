"""尺追従の保護区間（Responsive Design – Time）。

守りたいのは «短い尺でも動きが終わり切る» ことと、«登場の勢いは縮めない» ことの
両立です。片方だけなら簡単なので、両方が同時に成り立つことを見ます。
"""

import pytest

from movo.skill.responsive_time import fit_animations_to_clip, merge_protect, resolve_protect


def layer_with(times, **extra):
    return {
        "id": "x",
        "animations": [{"property": "transform.opacity", "keyframes": [{"time": t, "value": 1} for t in times]}],
        **extra,
    }


def times_of(layer):
    return [k["time"] for k in layer["animations"][0]["keyframes"]]


# ── 畳む・畳まない ──────────────────────────────────────────


def test_収まっているものは触らない():
    layer = layer_with([0, 1, 2], end=4)
    assert times_of(fit_animations_to_clip(layer)) == [0, 1, 2]


def test_足りない分を引き伸ばしはしない():
    # 0.9 秒で出るタイトルが 32 秒かけて出てきては誰も望みません。
    layer = layer_with([0, 0.9], end=32)
    assert times_of(fit_animations_to_clip(layer)) == [0, 0.9]


def test_はみ出したら畳む():
    layer = layer_with([0, 4, 8], end=4)
    result = times_of(fit_animations_to_clip(layer))
    assert result[-1] <= 4
    assert result[0] == 0


def test_キーフレームが無ければ何もしない():
    assert fit_animations_to_clip({"id": "x", "end": 1}) == {"id": "x", "end": 1}


# ── 保護区間 ────────────────────────────────────────────────


def test_頭の保護区間は縮まない():
    # «跳ねて出る決め文句» が、尺を半分にしただけでぬるっと出るのを防ぎます。
    layer = layer_with([0, 0.4, 4, 8], end=4)
    result = times_of(fit_animations_to_clip(layer, {"protect": {"in": 0.4, "out": 0}}))
    assert result[1] == pytest.approx(0.4)
    assert result[-1] <= 4


def test_尻の保護区間は終わりからの距離を保つ():
    layer = layer_with([0, 4, 7.5, 8], end=4)
    result = times_of(fit_animations_to_clip(layer, {"protect": {"in": 0, "out": 0.5}}))
    # 元の «終わりの 0.5 秒前» が、畳んだあとも «終わりの 0.5 秒前» に来ること
    assert result[-2] == pytest.approx(result[-1] - 0.5, abs=0.01)


def test_保護区間が尺に収まらなければ一律に畳む():
    # 守り切ろうとすると «動きが途中で止まったまま終わる» に戻ります。
    layer = layer_with([0, 5, 10], end=1)
    result = times_of(fit_animations_to_clip(layer, {"protect": {"in": 2, "out": 2}}))
    assert result[-1] <= 1


def test_保護区間は拍でも書ける():
    # 拍で書けば «決め文句は 1 拍で出る» が BPM を変えても保たれます。
    protect = resolve_protect({"in": "1beat"}, {"bpm": 120})
    assert protect["in"] == pytest.approx(0.5)


def test_保護区間は小節でも書ける():
    protect = resolve_protect({"in": "1/2bar"}, {"bpm": 120})
    assert protect["in"] == pytest.approx(1.0)


def test_読めない値は無視して先へ進む():
    # 打ち間違い（"1bea"）を «保護区間 0» として黙って流すと、効かない理由が
    # 分からないまま終わります。警告を出して None にします。
    assert resolve_protect({"in": "1bea"}, {"bpm": 120}) is None


def test_書いていない側は_None_のまま返す():
    # 0 と «書いていない» を区別しないと、片側だけの上書きができません。
    protect = resolve_protect({"in": 0.4}, {})
    assert protect["in"] == pytest.approx(0.4)
    assert protect["out"] is None


def test_レイヤー側の指定がスキル側より強い():
    merged = merge_protect({"in": 0.4, "out": 0.4}, {"in": 0.1, "out": None})
    assert merged == {"in": 0.1, "out": 0.4}


def test_どちらも無ければ_None():
    assert merge_protect(None, None) is None


def test_timeProtect_は出力から消える():
    # レンダラは秒しか見ない、という約束を保つためです。
    layer = layer_with([0, 8], end=4, timeProtect={"in": 0.2})
    assert "timeProtect" not in fit_animations_to_clip(layer, {"bpm": 120})
