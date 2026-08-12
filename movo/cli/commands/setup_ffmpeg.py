"""`movo setup-ffmpeg` — ffmpeg を ``~/.movo/bin`` に取ってくる。

## なぜ «同梱» ではなく «取ってくる» のか

mp4 / webm / mov を繋ぐのに ffmpeg が要ります。実行ファイルに同梱するのが
利用者にはいちばん楽ですが、**配布されている ffmpeg のビルドはほぼ GPL** です
（mp4 の H.264 に使う libx264 が GPL なので ``--enable-gpl`` 付きで作られている）。

このプロジェクトは ``THIRD-PARTY-NOTICES.md`` に

    いずれも再配布が許諾されている寛容なライセンスです。

と書いて、PyInstaller の例外条項まで確認したうえで **同梱物を寛容ライセンスだけで
揃える** 判断をしています。GPL のバイナリを配布物に入れると、その部分のソース提供
義務が生じ、この方針が崩れます。

**取ってくるだけなら再配布に当たらない** ので、方針を保ったまま「利用者は1回
コマンドを叩くだけ」にできます。配布物も太りません（静的ビルドは 1 つ 100MB 超）。

## 使い方

    movo setup-ffmpeg            確認してから取ってくる
    movo setup-ffmpeg --yes      確認せずに取ってくる（CI 用）
    movo setup-ffmpeg --dry-run  取得先とサイズだけ見る
    movo setup-ffmpeg --where    置き場所を出す

環境変数 ``MOVO_SETUP_YES=1`` でも確認を飛ばせます。

## 勝手に通信しません

Movo は «式エンジンはファイル・ネットワーク・プロセスに触れない» を設計の約束に
しています。ここはその外側（利用者が明示的に叩くコマンド）ですが、それでも
**既定では確認を取ります**。取得先とサイズを見せてから聞きます。
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ..config_store import movo_home
from ..console import say, style

#: 取得先。**実測で存在を確認したものだけを載せる**（2026-08-12）。
#:   BtbN/FFmpeg-Builds … GitHub Releases。win64 / linux64 の静的ビルド
#:   evermeet.cx        … macOS の定番。arm64 / x86_64 の universal
SOURCES = {
    "windows": {
        "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "archive": "zip",
        "member": "bin/ffmpeg.exe",
        "out": "ffmpeg.exe",
        "note": "BtbN/FFmpeg-Builds（GPL ビルド）",
    },
    "linux": {
        "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
        "archive": "tar.xz",
        "member": "bin/ffmpeg",
        "out": "ffmpeg",
        "note": "BtbN/FFmpeg-Builds（GPL ビルド）",
    },
    "darwin": {
        "url": "https://evermeet.cx/ffmpeg/getrelease/zip",
        "archive": "zip",
        "member": "ffmpeg",
        "out": "ffmpeg",
        "note": "evermeet.cx（GPL ビルド）",
        # **evermeet.cx は別ドメインへ 302 する**（実測 2026-08-12:
        # https://evermeet.cx/ffmpeg/getrelease/zip → https://deolaha.ca/ffmpeg/ffmpeg-9.0.zip）。
        # 素性の分からないミラーから実行ファイルを取ることになるので、
        # macOS では Homebrew を勧め、それでも取るなら明示の同意を求める。
        "prefer": "brew install ffmpeg",
        "mirror_warning": True,
    },
}

#: 実行ファイルを受け取ってよいホスト。ここに無いところへ飛んだら止める。
#: 転送先を確かめずに落とすと、途中で差し替えられても気づけない。
ALLOWED_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "evermeet.cx",
}


def host_allowed(url: str) -> bool:
    """その URL から実行ファイルを受け取ってよいか。

    転送先を確かめずに落とすと、途中で差し替えられても気づけない。
    純粋関数にしてあるのは、通信せずにテストできるようにするため。
    """
    import urllib.parse

    return (urllib.parse.urlparse(url).hostname or "") in ALLOWED_HOSTS


def bin_dir() -> Path:
    """取ってきた ffmpeg の置き場。``MOVO_HOME`` があればその下。"""
    return movo_home() / "bin"


def current_platform() -> str:
    s = sys.platform
    if s.startswith("win"):
        return "windows"
    if s == "darwin":
        return "darwin"
    return "linux"


def installed_path() -> Path | None:
    """既に取ってきてあれば、その場所。無ければ None。"""
    name = SOURCES[current_platform()]["out"]
    p = bin_dir() / name
    return p if p.exists() else None


def _extract(archive_path: Path, kind: str, member_suffix: str, dest: Path) -> None:
    """書庫から ffmpeg 本体1つだけを取り出す。

    **メンバ名を完全一致で探さない。** 書庫の中身は
    ``ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe`` のように版名のフォルダが
    先頭に付き、その名前は更新のたびに変わる。末尾一致で拾う。
    """
    if kind == "zip":
        with zipfile.ZipFile(archive_path) as z:
            names = [n for n in z.namelist() if n.endswith(member_suffix) and not n.endswith("/")]
            if not names:
                raise RuntimeError(f"書庫に {member_suffix} が見つかりません")
            with z.open(names[0]) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    else:
        import tarfile

        with tarfile.open(archive_path, "r:xz") as t:
            names = [m for m in t.getmembers() if m.name.endswith(member_suffix) and m.isfile()]
            if not names:
                raise RuntimeError(f"書庫に {member_suffix} が見つかりません")
            src = t.extractfile(names[0])
            if src is None:
                raise RuntimeError("書庫からファイルを取り出せません")
            with src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)


def setup_ffmpeg_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    plat = current_platform()
    src = SOURCES[plat]
    target = bin_dir() / src["out"]

    if options.get("where"):
        say(str(target))
        return {"path": str(target), "exists": target.exists()}

    if target.exists() and not options.get("force"):
        say(f"すでにあります: {target}")
        say("入れ直すなら --force を付けてください。")
        return {"path": str(target), "installed": False, "reason": "already-present"}

    if src.get("prefer") and not options.get("allow_mirror"):
        say(f"macOS は {src['prefer']} で入れるのがいちばん確実です。")
        say("公式の配布元が別ドメインへ転送するため、こちらからの取得は既定で止まります。")
        say("")

    say(f"取得先 : {src['url']}")
    say(f"提供元 : {src['note']}")
    say(f"置き場 : {target}")
    say("")
    say(style.bold("ffmpeg は GPL のソフトウェアです。") if hasattr(style, "bold") else "ffmpeg は GPL のソフトウェアです。")
    say("Movo 本体（MIT）とは別のプログラムとして、別ファイルで置きます。")
    say("同梱して再配布はしません。ライセンスの全文は取得先を参照してください。")
    say("")

    if options.get("dry_run") or options.get("dry-run"):
        return {"path": str(target), "installed": False, "reason": "dry-run", "url": src["url"]}

    if not (options.get("yes") or os.environ.get("MOVO_SETUP_YES")):
        try:
            answer = input("取得しますか？ [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            say("やめました。")
            return {"path": str(target), "installed": False, "reason": "declined"}

    import urllib.error
    import urllib.parse
    import urllib.request

    bin_dir().mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if src["archive"] == "zip" else ".tar.xz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        say("取得中…")
        try:
            response = urllib.request.urlopen(src["url"], timeout=180)
        except urllib.error.URLError as e:
            reason = str(getattr(e, "reason", e))
            say(f"取得できませんでした: {reason}")
            if "CERTIFICATE_VERIFY_FAILED" in reason:
                # macOS の python.org 版は CA 証明書を自分で入れる必要がある。
                # 素の SSL エラーだけ出されても何を直せばよいか分からない。
                say("")
                say("Python に CA 証明書が入っていないようです。macOS で python.org 版を")
                say("使っている場合は、次を1度だけ実行してください。")
                say('  /Applications/Python\\ 3.11/Install\\ Certificates.command')
                say("それでも直らなければ、ffmpeg を直接入れてください。")
                if src.get("prefer"):
                    say(f"  {src['prefer']}")
            return {"path": str(target), "installed": False, "reason": "download-failed"}

        with response:
            final = response.geturl()
            host = urllib.parse.urlparse(final).hostname or ""
            if not host_allowed(final) and not options.get("allow_mirror"):
                say("")
                say(f"転送先が想定外です: {host}")
                say(f"  {src['url']}")
                say(f"  → {final}")
                say("素性の分からない場所から実行ファイルを取るのは危ないので止めました。")
                if src.get("prefer"):
                    say(f"こちらを使ってください: {src['prefer']}")
                say("それでも取るなら --allow-mirror を付けてください。")
                return {"path": str(target), "installed": False, "reason": "unexpected-host", "host": host}
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(response, out)

        digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        say(f"取得しました（{tmp_path.stat().st_size / 1048576:.1f} MB / sha256 {digest[:16]}…）")

        _extract(tmp_path, src["archive"], src["member"], target)
        if plat != "windows":
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        # 取ってきたものの出所を残す。あとから «どこの何を入れたか» を辿れるように。
        (bin_dir() / "SOURCE.txt").write_text(
            f"url: {src['url']}\nsha256(archive): {digest}\nnote: {src['note']}\n",
            encoding="utf-8",
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # **動くことまで確かめる。** 置けただけで «入った» と言うと、壊れた書庫を
    # 掴んだときに render の途中まで気づけない。
    ok, version = _verify(target)
    if not ok:
        return {"path": str(target), "installed": False, "reason": "verify-failed"}

    say(f"使えます: {version}")
    say(f"置き場所: {target}")
    return {"path": str(target), "installed": True, "version": version}


def _verify(path: Path) -> tuple[bool, str]:
    import subprocess

    try:
        out = subprocess.run(
            [str(path), "-version"], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError as e:
        say(f"取ってきましたが動きません: {e}")
        return False, ""
    if out.returncode != 0:
        say("取ってきましたが動きません（-version が失敗）。")
        return False, ""
    first = (out.stdout or out.stderr or "").splitlines()
    return True, first[0] if first else "ffmpeg"
