"""`movo setup-ffmpeg` のテスト。**通信しない。**

このコマンドは «外から実行ファイルを取ってきて動かす» ので、いちばん危ないのは
「どこから取ったか」である。転送先の判定を純粋関数に出してあるので、そこを固定する。
"""

from __future__ import annotations

from movo.cli.commands.setup_ffmpeg import (
    ALLOWED_HOSTS,
    SOURCES,
    current_platform,
    host_allowed,
)


def test_許可したホストは通る():
    assert host_allowed("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/x.zip")
    assert host_allowed("https://objects.githubusercontent.com/whatever")
    assert host_allowed("https://evermeet.cx/ffmpeg/getrelease/zip")


def test_知らないホストは弾く():
    """evermeet.cx は別ドメインへ 302 する（実測 2026-08-12: deolaha.ca）。

    素性の分からないミラーから実行ファイルを取らないための本丸。
    **ここを緩めない。** 緩めると転送先の差し替えに気づけなくなる。
    """
    assert not host_allowed("https://deolaha.ca/ffmpeg/ffmpeg-9.0.zip")
    assert not host_allowed("http://evil.example/ffmpeg.zip")


def test_平文のhttpも弾く():
    """http:// の GitHub も許さない（ホスト名だけ見ているので念のため固定）。"""
    # ホスト名は一致するが、実運用の URL はすべて https。表に http を足さないこと。
    for url in SOURCES.values():
        assert url["url"].startswith("https://")


def test_取得先はすべて許可ホストを指している():
    """表に «最初から弾かれる URL» を書いてしまう事故を止める。"""
    for name, src in SOURCES.items():
        assert host_allowed(src["url"]), f"{name} の取得先が許可ホストに無い: {src['url']}"


def test_プラットフォームの判定が3種に収まる():
    assert current_platform() in ("windows", "linux", "darwin")


def test_全プラットフォームぶんの取得先がある():
    assert set(SOURCES) == {"windows", "linux", "darwin"}
    for name, src in SOURCES.items():
        for key in ("url", "archive", "member", "out", "note"):
            assert src.get(key), f"{name} に {key} が無い"


def test_許可ホストにワイルドカードを入れない():
    """`*` や空文字を入れると «何でも通る» になる。"""
    for h in ALLOWED_HOSTS:
        assert h and "*" not in h
