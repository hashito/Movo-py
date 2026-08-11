"""OS ごとの違いを吸収する層（仕様 31 節）。

パス・フォントの置き場・ffmpeg 探し・GPU 検出・一時ディレクトリ・プロセス起動を
**すべてここに集めます。** 他のモジュールが ``sys.platform`` を見ないようにする
のが目的です。分散すると «Windows でだけ動かない» が発見しにくくなります。
"""

from __future__ import annotations

import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


class _Platform:
    """今動いている環境。"""

    os = sys.platform
    machine = os_arch = ""
    is_windows = sys.platform == "win32"
    is_mac = sys.platform == "darwin"
    is_linux = sys.platform.startswith("linux")


platform = _Platform()
platform.os_arch = platform.machine = os.uname().machine if hasattr(os, "uname") else os.environ.get("PROCESSOR_ARCHITECTURE", "")

#: 子プロセスのコンソール窓を出さない（Windows のみ意味があります）。
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if platform.is_windows else {}


def movo_home() -> str:
    """利用者ごとの Movo の置き場（設定・全体キャッシュ・資格情報）。"""
    env = os.environ.get("MOVO_HOME")
    if env:
        return env
    return str(Path.home() / ".movo")


_counter = 0


def temp_dir(name: str = "movo") -> str:
    """使い捨てのディレクトリを作る。

    プロセス ID と連番を入れるのは、**並列レンダリングで複数のプロセスが
    同じ名前を掴まないようにする**ためです。
    """
    global _counter
    _counter += 1
    base = os.environ.get("MOVO_TMPDIR") or tempfile.gettempdir()
    path = Path(base) / f"{name}-{os.getpid()}-{_counter}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_project_path(project_root: str | os.PathLike[str], relative: str | None) -> str:
    """プロジェクト相対のパスを、この OS の区切りに直して絶対パスにする。

    **JSON には常に ``/`` で書きます。** Windows で作ったプロジェクトが
    macOS でも動くようにするためで、``\\`` は読むときにここで直します。
    """
    if not relative:
        return str(project_root)
    normalised = str(relative).replace("\\", "/")
    if os.path.isabs(normalised) or re.match(r"^[a-zA-Z]:/", normalised):
        return os.path.normpath(normalised)
    return os.path.abspath(os.path.join(str(project_root), normalised))


def to_project_relative(project_root: str | os.PathLike[str], absolute: str | os.PathLike[str]) -> str:
    """絶対パスを «持ち運べる» プロジェクト相対のパスに戻す（区切りは ``/``）。"""
    rel = os.path.relpath(str(absolute), str(project_root))
    return rel.replace(os.sep, "/")


_FONT_DIRS = {
    "win32": [
        Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts",
        Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
    ],
    "darwin": [
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
    ],
    "linux": [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ],
}


def system_font_dirs() -> list[str]:
    """この OS でフォントを探す場所。存在するものだけ返します。"""
    key = "win32" if platform.is_windows else "darwin" if platform.is_mac else "linux"
    return [str(d) for d in _FONT_DIRS[key] if d.is_dir()]


def list_font_files(extra_dirs: Iterable[str] = (), limit: int = 4000) -> list[str]:
    """フォントファイルを列挙する。

    **上限と深さで打ち切ります。** フォントを 2 万本入れている機械があり、
    無制限に走査すると ``movo doctor`` が 40 秒かかったためです。
    """
    found: list[str] = []
    seen: set[str] = set()

    def walk(directory: str, depth: int) -> None:
        if len(found) >= limit or depth > 4:
            return
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for entry in entries:
            if len(found) >= limit:
                return
            if entry.is_dir(follow_symlinks=False):
                walk(entry.path, depth + 1)
            elif entry.name.lower().endswith((".ttf", ".otf", ".ttc", ".otc")):
                key = entry.path.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(entry.path)

    for directory in [*extra_dirs, *system_font_dirs()]:
        walk(directory, 0)
    return found


_FFMPEG_CANDIDATES = {
    "win32": ["ffmpeg.exe", "C:\\ffmpeg\\bin\\ffmpeg.exe", "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"],
    "darwin": ["ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"],
    "linux": ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/snap/bin/ffmpeg"],
}

#: «まだ探していない» を表す番人。``None`` は «探したが無かった» の意味なので、
#: 素直に ``None`` を初期値にすると毎回探し直してしまいます。
_UNSET = object()
_ffmpeg_cache: object = _UNSET


def find_ffmpeg(force: bool = False) -> dict | None:
    """ffmpeg を探す。順番は ``MOVO_FFMPEG`` → PATH → よくある場所。

    **結果を覚えます。** 探索は 1 回あたり 100 ミリ秒ほどかかり、
    書き出しのたびに呼ばれるためです。
    """
    global _ffmpeg_cache
    if not force and _ffmpeg_cache is not _UNSET:
        return _ffmpeg_cache  # type: ignore[return-value]
    candidates: list[str] = []
    env = os.environ.get("MOVO_FFMPEG")
    if env:
        candidates.append(env)
    candidates.append("ffmpeg")
    key = "win32" if platform.is_windows else "darwin" if platform.is_mac else "linux"
    candidates += _FFMPEG_CANDIDATES[key]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-version"], capture_output=True, text=True, timeout=15, **_NO_WINDOW
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
            _ffmpeg_cache = {"path": candidate, "version": first_line.strip()}
            return _ffmpeg_cache  # type: ignore[return-value]
    _ffmpeg_cache = None
    return None


def find_ffprobe() -> dict | None:
    """ffmpeg の隣にある ffprobe を探す（無くても動きます）。"""
    names = ["ffprobe.exe", "ffprobe"] if platform.is_windows else ["ffprobe"]
    env = os.environ.get("MOVO_FFPROBE")
    if env:
        names.insert(0, env)
    for name in names:
        try:
            result = subprocess.run(
                [name, "-version"], capture_output=True, text=True, timeout=15, **_NO_WINDOW
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return {"path": name}
    return None


def detect_gpu() -> dict:
    """GPU の «あるなし» を軽く見る。

    Movo のソフトウェアレンダラは GPU を使いません。``movo doctor`` の
    報告と、将来の切り替え判断のためだけの情報です。**失敗しても黙って
    «無い» と答えます。**
    """
    try:
        if platform.is_windows:
            r = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                capture_output=True, text=True, timeout=8, **_NO_WINDOW,
            )
            if r.returncode == 0 and r.stdout.strip():
                names = [s.strip() for s in r.stdout.strip().splitlines() if s.strip()]
                return {"available": True, "devices": names}
        elif platform.is_mac:
            r = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=8
            )
            if r.returncode == 0:
                names = [m.strip() for m in re.findall(r"Chipset Model:\s*(.+)", r.stdout)]
                if names:
                    return {"available": True, "devices": names}
        else:
            r = subprocess.run(
                ["sh", "-c", 'lspci 2>/dev/null | grep -i "vga\\|3d\\|display"'],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode == 0 and r.stdout.strip():
                return {"available": True, "devices": [s.strip() for s in r.stdout.strip().split("\n")]}
    except Exception:
        pass  # 調べられなくても困らない
    return {"available": False, "devices": []}


def run(command: str, args: list[str], **options) -> dict:
    """子プロセスを回して終了状態を返す。

    JS 版は Promise を返しますが、Python 側は **同期**にしました。
    レンダリングは multiprocessing で並列にするので、ここを非同期にする
    意味がありません（async の伝染だけが残ります）。
    """
    result = subprocess.run(
        [command, *args], capture_output=True, text=True, **{**_NO_WINDOW, **options}
    )
    return {"code": result.returncode, "stdout": result.stdout or "", "stderr": result.stderr or ""}


def cpu_count() -> int:
    """使える CPU の数。ワーカー数を決めるのに使います。"""
    try:
        # Linux の cgroup / taskset で «使ってよい» 数が絞られていることがあるので、
        # 実際に使える数を優先します（コンテナで 64 個見えて 2 個しか使えない例）。
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
        return max(1, multiprocessing.cpu_count())
    except (OSError, NotImplementedError):
        return 1


def describe_environment() -> dict:
    """``movo doctor`` が出す環境の要約。**キー名は JS 版のまま**です。"""
    try:
        import platform as _pyplatform

        os_name = f"{_pyplatform.system()} {_pyplatform.release()}"
    except Exception:
        os_name = sys.platform
    total_gb = 0.0
    try:
        total_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3 * 10) / 10
    except (ValueError, OSError, AttributeError):
        try:  # Windows
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total_gb = round(status.ullTotalPhys / 1024**3 * 10) / 10
        except Exception:
            total_gb = 0.0
    return {
        "movoPython": sys.version.split()[0],
        "os": os_name,
        "platform": platform.os,
        "arch": platform.machine,
        "cpus": cpu_count(),
        "memoryGB": total_gb,
        "home": movo_home(),
    }


def which(name: str) -> str | None:
    """PATH から実行ファイルを探す小さな入口。"""
    return shutil.which(name)
