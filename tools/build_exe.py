"""単体の実行ファイル（EXE）を作る。

    python tools/build_exe.py              作って動作確認までやる
    python tools/build_exe.py --no-verify  作るだけ
    python tools/build_exe.py --onedir     1 ファイルではなくフォルダ形式で作る

## ライセンスの確認（済み）

PyInstaller は GPLv2+ ですが、**«ビルドした成果物は任意のライセンスで配布可»
という例外条項** が付いています。したがって成果物に GPL は伝播しません。

同梱するものは NumPy（BSD-3）/ Numba（BSD-2）/ llvmlite（BSD-2 と
Apache-2.0 WITH LLVM-exception）/ Python 本体（PSF）で、いずれも再配布できます。
**NumPy は BSD なので著作権表示の同梱が要ります。** `THIRD-PARTY-NOTICES.md` を
EXE に入れ、`movo doctor` が «同梱の著作権表示» としてその場所を案内します
（入れただけで場所を言えないと、義務を果たした形になりません）。

## 気をつけたところ

1. **遅延インポートは PyInstaller から «見えません»。**
   `movo/cli/main.py` はコマンドを `importlib` で読みます（`movo --version` に
   1 秒かけないため）。静的解析では追えないので、`--hidden-import` で全部
   名指しします。**書き忘れると «そのコマンドだけ動かない EXE» ができます。**

2. **Numba のキャッシュは EXE の中に書けません。**
   PyInstaller が展開する `_MEIPASS` は実行のたびに作り直される一時フォルダです。
   そこへ書いてもキャッシュは毎回捨てられ、しかも **並列レンダリングでは
   プロセスの数だけコンパイルが走ります**。実行時のキャッシュ先を
   `%LOCALAPPDATA%` に向けます（`movo/cli/parallel.py` の `numba_cache_dir`）。

3. **子プロセスが EXE を無限に増やしません。**
   `multiprocessing.freeze_support()` を `main()` の先頭で呼んでいます。
   これが無いと、並列レンダリングの子が «また CLI として» 起動します。

   ただし **1 ファイル形式では並列が割に合わないことがあります。**
   子プロセスは EXE を起こし直すので、そのたびに 75MB を展開します。実測で
   10 秒の動画が「ソースから 13 秒／1 ファイル EXE から 42 秒」でした。
   **長い動画を並列で書き出すなら `--onedir` で作ってください**（展開が
   要らないぶん、子の起動が段違いに速くなります）。

4. **スキル定義とプロファイルは JSON なので同梱します。**
   `movo/skill/library/**/*.json`（57 件）と `movo/library/profiles/*.json` です。
   同じ相対位置に入れるので、`builtin_library_root()` がそのまま見つけます。

5. **入口は `movo/cli/main.py` を直接渡してはいけません。**
   そうすると PyInstaller はそれを `__main__` として実行するので、
   `from .args import ...` のような相対 import が
   «attempted relative import with no known parent package» で全部落ちます
   （**ビルドは成功して、起動だけが必ず失敗します**）。パッケージ経由で呼ぶ
   1 行の起動口を作って、そちらを渡します。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
NAME = "movo"

# **`movo` は丸ごと集めます。** CLI はコマンドを `importlib` で、他のモジュールを
# `movo/cli/bridge.py` で «名前を文字列で» 読みます。どちらも静的解析からは
# 追えないので、名指しの一覧に頼ると必ず取りこぼします（実際に
# `movo.schema.params` が入らず、**起動はするのに render だけ落ちる EXE** が
# できました）。丸ごと集めるほうが確実で、増える大きさも数百 KB です。
COLLECT_PACKAGES = ["movo"]

HIDDEN_IMPORTS = [
    # 並列レンダリングの子プロセスが読むもの
    "multiprocessing.pool",
    "multiprocessing.managers",
    "multiprocessing.spawn",
    # Numba は実行時に自分のモジュールを読みます
    "numba",
    "numba.core.typing.builtins",
]

# EXE に入れるデータ。`(元, EXE の中での置き場)`。
# **置き場は import したときと同じ相対位置** にします。そうすれば
# `builtin_library_root()` のような «自分の隣を見る» 書き方がそのまま通ります。
def data_entries() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    library = ROOT / "movo" / "skill" / "library"
    if library.is_dir():
        entries.append((library, "movo/skill/library"))
    profiles = ROOT / "movo" / "library" / "profiles"
    if profiles.is_dir():
        entries.append((profiles, "movo/library/profiles"))
    notices = ROOT / "THIRD-PARTY-NOTICES.md"
    if notices.is_file():
        # NumPy が BSD なので、**著作権表示の同梱が要ります**。
        entries.append((notices, "."))
    return entries


# PyInstaller に渡す «起動口»。`movo/cli/main.py` をそのまま渡すと `__main__` として
# 実行され、相対 import が全部落ちます。パッケージ経由で呼ぶ 1 行だけを置きます。
LAUNCHER = '''"""movo の実行ファイルの起動口（tools/build_exe.py が生成します）。"""

from movo.cli.main import main

if __name__ == "__main__":
    main()
'''


def write_launcher() -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    launcher = BUILD / "movo_entry.py"
    launcher.write_text(LAUNCHER, encoding="utf-8")
    return launcher


def build(onefile: bool = True, clean: bool = True) -> Path:
    if shutil.which("pyinstaller") is None and not _has_module("PyInstaller"):
        raise SystemExit(
            "PyInstaller が見つかりません。\n"
            '  pip install -e ".[dev]"   もしくは   pip install pyinstaller'
        )

    if clean:
        shutil.rmtree(BUILD, ignore_errors=True)
        shutil.rmtree(DIST, ignore_errors=True)

    launcher = write_launcher()
    separator = ";" if os.name == "nt" else ":"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile" if onefile else "--onedir",
        "--console",
        "--name",
        NAME,
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD),
        "--specpath",
        str(BUILD),
        # ライセンス表示と定義ファイルを同梱する
        *[arg for source, target in data_entries() for arg in ("--add-data", f"{source}{separator}{target}")],
        *[arg for name in COLLECT_PACKAGES for arg in ("--collect-submodules", name)],
        *[arg for name in HIDDEN_IMPORTS for arg in ("--hidden-import", name)],
        # tkinter と matplotlib は使いません。入れると 30MB 増えます。
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "pytest",
        # `movo` パッケージそのものは «探せる場所» に無いと拾われません
        "--paths",
        str(ROOT),
        str(launcher),
    ]

    print("PyInstaller で固めています…")
    print("  " + " ".join(command[3:12]) + " …")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(f"ビルドに失敗しました（終了コード {completed.returncode}）")

    exe = DIST / (f"{NAME}.exe" if os.name == "nt" else NAME)
    if not onefile:
        exe = DIST / NAME / (f"{NAME}.exe" if os.name == "nt" else NAME)
    if not exe.is_file():
        raise SystemExit(f"できたはずの実行ファイルが見つかりません: {exe}")
    size_mb = exe.stat().st_size / 1024 / 1024
    print(f"\nできました: {exe}")
    print(f"  大きさ  : {size_mb:.1f} MB")
    print(f"  かかった時間: {time.perf_counter() - started:.0f} 秒")
    return exe


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def verify(exe: Path) -> bool:
    """**できた EXE で実際にコマンドを走らせます。**

    «ビルドは通ったのに動かない» のが PyInstaller のいちばん多い失敗です。
    データの同梱漏れも遅延インポートの書き忘れも、走らせないと分かりません。
    """
    print("\n動作確認")
    checks: list[tuple[str, list[str], callable]] = [
        ("movo --version", ["--version"], lambda out: out.strip() != ""),
        ("movo list effects", ["list", "effects"], lambda out: "件" in out),
        ("movo skill list（定義の同梱）", ["skill", "list", "--movies"], lambda out: "lyric-mv" in out),
        ("movo doctor", ["doctor"], lambda out: "movo" in out),
        # 著作権表示が EXE の中から «見つけられる» ことまで確かめます
        ("THIRD-PARTY-NOTICES の同梱", ["doctor"], lambda out: "同梱の著作権表示" in out),
    ]
    ok = True
    for label, args, check in checks:
        completed = subprocess.run(
            [str(exe), *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        passed = completed.returncode == 0 and check(output)
        print(f'  {"v" if passed else "x"} {label}')
        if not passed:
            ok = False
            print("      " + "\n      ".join(output.strip().splitlines()[-6:]))

    # `movo render` は実際に 1 本書き出します。**ここまでやらないと «同梱した
    # つもりのものが入っていない» に気付けません。**
    ok = _verify_render(exe) and ok
    return ok


def _verify_render(exe: Path) -> bool:
    import json

    project = {
        "movoVersion": "1.0",
        "project": {"name": "smoke", "seed": 1},
        "video": {"width": 320, "height": 180, "fps": 12, "duration": 1, "background": "#101020"},
        "scenes": [
            {
                "id": "main",
                "start": 0,
                "duration": 1,
                "layers": [
                    {
                        "id": "box",
                        "type": "shape",
                        "shape": {"type": "rectangle", "width": 120, "height": 80, "fill": "#88aaff"},
                        "transform": {"x": 160, "y": 90, "anchorX": 0.5, "anchorY": 0.5},
                    }
                ],
            }
        ],
        "output": {"format": "png-sequence"},
    }
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "smoke.json"
        path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
        out = Path(work) / "frames"
        completed = subprocess.run(
            [str(exe), "render", str(path), "-o", str(out), "--format", "png-sequence", "--quiet"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        written = len(list(out.glob("*.png"))) if out.is_dir() else 0
        passed = completed.returncode == 0 and written >= 12
        print(f'  {"v" if passed else "x"} movo render（{written} 枚）')
        if not passed:
            tail = ((completed.stdout or "") + (completed.stderr or "")).strip().splitlines()[-8:]
            print("      " + "\n      ".join(tail))
            if any("後で繋ぐ" in line for line in tail):
                print("      （まだ移植されていないモジュールがあります。EXE 側の問題ではありません）")
        return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Movo の単体 EXE を作る")
    parser.add_argument("--onedir", action="store_true", help="1 ファイルではなくフォルダ形式で作る（起動が速い）")
    parser.add_argument("--no-verify", action="store_true", help="作るだけで動作確認をしない")
    parser.add_argument("--no-clean", action="store_true", help="前回の中間ファイルを消さない")
    args = parser.parse_args()

    exe = build(onefile=not args.onedir, clean=not args.no_clean)
    if args.no_verify:
        return 0
    ok = verify(exe)
    print("\n同梱したもの")
    for source, target in data_entries():
        print(f"  {source.name} → {target}")
    print("\n配布時の注意")
    print("  THIRD-PARTY-NOTICES.md を EXE に同梱しています（NumPy が BSD のため）。")
    print("  PyInstaller の例外条項により、成果物に GPL は伝播しません。")
    if not args.onedir:
        print("\n  1 ファイル形式は起動のたびに中身を展開します。並列レンダリング")
        print("  （--jobs）を主に使うなら --onedir で作ると子プロセスの起動が速くなります。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
