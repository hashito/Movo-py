"""`movo doctor` — 実行環境の診断。

JS 版に **1 節足しています**: «移植がどこまで繋がっているか»。core / schema /
renderer などは別の担当が並行して移植中なので、«動かない» と思ったときに
最初に見る場所が要ります。
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from movo import __version__

from .. import bridge
from ..config_store import list_config, movo_home
from ..console import logger, say, style


def doctor_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    ffmpeg = bridge.find_ffmpeg(refresh=True)
    # 無いときに «どうすればよいか» まで言う。「見つかりません」だけだと、
    # mp4 が書き出せない理由が分かっても直し方が分からない。
    ffmpeg_hint = None if ffmpeg else "movo setup-ffmpeg で取ってこられます（mp4 / webm / mov に必要）"
    ffprobe = bridge.find_ffprobe()
    config = list_config()
    python_ok = sys.version_info >= (3, 11)

    numpy_version = _version_of("numpy")
    numba_version = _version_of("numba")

    fonts = _describe_fonts()
    bridge_rows = bridge.module_status()
    connected = sum(1 for row in bridge_rows if row["connected"])

    checks = [
        {
            "name": "Python",
            "ok": python_ok,
            "value": platform.python_version(),
            "hint": "Python 3.11 以上が必要です",
        },
        {"name": "OS", "ok": True, "value": f"{platform.platform()} ({platform.machine()})"},
        {
            "name": "CPU / メモリ",
            "ok": True,
            "value": f"{os.cpu_count() or '?'} コア / {_memory_gb()} GB",
        },
        {
            "name": "NumPy",
            "ok": numpy_version is not None,
            "value": numpy_version or "見つかりません",
            "hint": "全画面のエフェクトと合成に必須です（pip install numpy）",
        },
        {
            "name": "Numba",
            "ok": numba_version is not None,
            "value": numba_version or "見つかりません（ラスタライザが 100 倍遅くなります）",
            "hint": "画素ごとの処理を C 並みの速度にします（pip install numba）",
        },
        {
            "name": "ffmpeg",
            "ok": bool(ffmpeg),
            "value": (ffmpeg or {}).get("path") or "見つかりません",
            "hint": ffmpeg_hint or "MP4/WebM 出力、動画・音声素材、並列レンダリングの連結に必要です",
        },
        {"name": "ffprobe", "ok": bool(ffprobe), "value": "利用可能" if ffprobe else "見つかりません（任意）"},
        {
            "name": "フォント",
            "ok": (fonts.get("fontFileCount") or 0) > 0,
            "value": fonts.get("error") or f'{fonts.get("fontFileCount", 0)} 個 / 既定 {fonts.get("defaultFont") or "不明"}',
            "hint": "テキストレイヤーには TrueType フォントが必要です。project.fonts で明示できます。",
        },
        # 日本語を別行にするのは、«ラテン文字は出るのに日本語だけ豆腐» と
        # «太字を頼んだのに太くならない» が、どちらもここを見れば分かるためです。
        {
            "name": "日本語フォント",
            "ok": bool(fonts.get("cjk_font")),
            "value": _cjk_font_value(fonts),
            "hint": "日本語を描ける TrueType フォント（CFF ではないもの）が要ります。project.fonts で明示できます。",
        },
        {
            "name": "API キー",
            "ok": True,
            "value": ", ".join(c["key"] for c in config) if config else "未設定（AI 素材はプレースホルダになります）",
        },
        {
            "name": "移植の接続",
            "ok": connected == len(bridge_rows),
            "value": f"{connected}/{len(bridge_rows)} モジュール",
            "hint": "未接続のものは «後で繋ぐ»。--json で内訳が出ます",
        },
    ]

    cache_directory = Path.cwd() / "cache"
    cache_size = _directory_size(cache_directory) if cache_directory.exists() else 0
    from ..parallel import numba_cache_dir

    notices = _notices_path()

    payload = {
        "movo": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "numpy": numpy_version,
        "numba": numba_version,
        "numbaCacheDir": str(numba_cache_dir()),
        "ffmpeg": ffmpeg,
        "ffprobe": bool(ffprobe),
        "fonts": fonts,
        "config": [c["key"] for c in config],
        "home": str(movo_home()),
        "cache": {"path": str(cache_directory), "exists": cache_directory.exists(), "bytes": cache_size},
        "frozen": bool(getattr(sys, "frozen", False)),
        "thirdPartyNotices": str(notices) if notices else None,
        "bridge": bridge_rows,
        "checks": [{"name": c["name"], "ok": c["ok"], "value": c["value"]} for c in checks],
    }

    if options.get("json"):
        say(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return payload

    logger.info(style.bold(f"movo {__version__}（Python 版）"))
    logger.info("")
    for check in checks:
        mark = style.green("v") if check["ok"] else style.yellow("!")
        logger.info(f'  {mark} {check["name"].ljust(14)} {check["value"]}')
        if not check["ok"] and check.get("hint"):
            logger.info(f'      {style.gray(check["hint"])}')

    missing = [row for row in bridge_rows if not row["connected"]]
    if missing:
        logger.info("")
        logger.info(style.bold(f"  まだ繋がっていないもの {len(missing)} 件（後で繋ぐ）"))
        for row in missing:
            logger.info(f'    {row["module"].ljust(28)} {row["label"]}')

    logger.info("")
    logger.info(f"  設定ファイル: {movo_home()}")
    logger.info(f"  JIT キャッシュ: {numba_cache_dir()}")
    if notices:
        # NumPy が BSD なので、**著作権表示を同梱する義務があります**。
        # 同梱したものが «どこにあるか» を言えないと、義務を果たした形になりません。
        logger.info(f"  同梱の著作権表示: {notices}")
    if payload["cache"]["exists"]:
        logger.info(f"  キャッシュ: {cache_directory} ({cache_size / 1024 / 1024:.1f} MB)")
    problems = [c for c in checks if not c["ok"]]
    logger.info("")
    if not problems:
        logger.success("問題は見つかりませんでした。")
    else:
        logger.warn(f"{len(problems)} 件の注意があります（動作はしますが機能が制限されます）。")
    return payload


def _notices_path() -> Path | None:
    """同梱した著作権表示（`THIRD-PARTY-NOTICES.md`）の場所。

    単体 EXE では展開先（`sys._MEIPASS`）に入っています。開発中はリポジトリの
    直下です。どちらでも «見つけて案内できる» ようにします。
    """
    candidates = []
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        candidates.append(Path(bundled) / "THIRD-PARTY-NOTICES.md")
    candidates.append(Path(__file__).resolve().parents[3] / "THIRD-PARTY-NOTICES.md")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _version_of(name: str) -> str | None:
    module = bridge.optional_module(name)
    if module is None:
        return None
    return getattr(module, "__version__", "不明")


def _memory_gb() -> str:
    from ..parallel import available_memory_bytes

    free = available_memory_bytes()
    if free <= 0:
        return "?"
    return f"{free / 1024 / 1024 / 1024:.1f}（空き）"


def _describe_fonts() -> dict[str, Any]:
    manager = bridge.pick("movo.renderer.font", "FontManager")
    if getattr(manager, "movo_not_connected", False):
        return {"error": "movo.renderer.font が未接続です（後で繋ぐ）", "fontFileCount": 0}
    try:
        return manager(project_root=os.getcwd()).describe()
    except Exception as error:  # noqa: BLE001
        return {"error": str(error), "fontFileCount": 0}


def _cjk_font_value(fonts: dict[str, Any]) -> str:
    """日本語の行に出す 1 文。«太字が無い» までここで言い切ります。"""
    if fonts.get("error"):
        return str(fonts["error"])
    regular = fonts.get("cjk_font")
    if not regular:
        return "見つかりません（日本語が豆腐 □ になります）"
    bold = fonts.get("cjk_bold_font")
    if bold:
        return f"{regular} / 太字 {bold}"
    return f'{regular} / 太字の面なし（weight:"bold" は通常ウェイトで描かれます）'


def _directory_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total
