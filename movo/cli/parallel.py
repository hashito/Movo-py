"""並列レンダリング（`movo render --jobs N`）。

**なぜ要るのか。** レンダリングは 1 プロセス 1 スレッドなので、長い曲を 1 本で
書き出すとコアが 1 つしか働きません。JS 版では 1280x720 の 1 フレームに
1.5〜2.5 秒かかり、153 秒の曲は **2 時間近く** かかっていました。
JS 版の実測は **4.8〜6.3 倍** です。

**Python 版では倍率が下がります（実測 2.6 倍）。** 1 フレームが 1.5 秒から
45 ミリ秒になったぶん、**時間の中身が «計算» から «メモリ帯域» に変わった**
からです。全画面のエフェクトは 1 枚 3.7MB を何度も往復するので、コアを増やしても
帯域が先に尽きます。12 スレッドの機械で 4 並列が 2.6 倍、11 並列は 1.4 倍まで
落ちました。**倍率は下がりましたが、1 本あたりの時間は JS 版より速いままです**
（60 秒の動画で JS 版 1 本 ≒ 30 分に対し、Python 版は 1 本 87 秒 / 4 並列 34 秒）。

**なぜ割れるのか。** Movo は «同じ JSON からは同じ動画» を保証しています。
乱数はすべてシードから作り、物理とパーティクルは 0 フレーム目から追いつき直すので、
**フレーム 1000 番だけを描いても、1 本で描いたときの 1000 番と同じ絵になります**。
だから区間に割って別々のプロセスで描き、あとから繋いでも中身は変わりません。

**例外が 2 つあります。**
  - frameEcho / slitScan は «直前の何フレームか» を覚えていて初めて正しい絵に
    なります → 区間の頭に助走（warmup）を足して履歴を作り直します
  - linePath.followLayer の軌跡は «描き始めからずっと» 積み上がるので、助走では
    再現できません → 並列にせず 1 本で描きます

**繋ぎ方。** 全部の区間を同じ設定で描いているので、concat デマルチプレクサに
`-c copy` を渡すだけで繋がります。再エンコードしないので画質は落ちません。
音は最後にまとめて多重化します（区間ごとに入れると、どの区間も曲の頭から
鳴ってしまうためです）。

## JS 版との違い — プロセスの起こし方

JS 版は `spawn(process.execPath, [CLI, ...])` で **CLI をもう 1 回起動** して
いました。Python には `multiprocessing` があるので、`ProcessPoolExecutor` で
素直に書けます。考え方は同じ（プロセスを分ける）ですが、

  - 引数を文字列に直して渡し直す必要がない（**渡し忘れ＝絵が変わる、を避けられる**）
  - 進捗は Queue で受け取れる（標準エラーに目印を流す必要がない）
  - 例外がそのまま親に返る

ぶんだけ、間違いの入りようが減っています。

## Numba の JIT をどう扱うか

**子プロセスごとに JIT のコンパイルが走ります。** 12 並列なら 12 回です。
1 回 1 秒でも 12 秒が丸ごと無駄になり、短い動画では並列にした意味が消えます。

対策は 2 つで、両方要ります。

1. `NUMBA_CACHE_DIR` を **子プロセス間で共有できる書ける場所** に向ける
   （`prepare_numba_cache`）。子は環境変数を引き継ぐので、1 つ目の子が
   コンパイルした結果を残りの子が読みます。
2. 親で **1 回だけ暖機** する（`warm_up_jit`）。子を起こす前にキャッシュを
   作っておけば、子はどれも読むだけで済みます。

`cache=True` はカーネル側（`movo.renderer.kernels`）に付ける必要があります。
そこは別の担当の持ち場なので、**`prepare_numba_cache` は «付いていれば効く»
形にしてあります**（付いていなければ、単に暖機の効果だけが残ります）。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

from . import bridge
from .console import logger
from .errors import ErrorCodes, MovoError

# 区間を割るのに最低これだけのフレームは欲しい。細切れにしても起動の分だけ損をする。
MIN_FRAMES_PER_CHUNK = 8

# 上限は 64。これ以上はメモリのほうが先に尽きます。
MAX_JOBS = 64


# ══════════════════════════════════════════════════════════════════
#  Numba の JIT を子プロセスで無駄に走らせない
# ══════════════════════════════════════════════════════════════════


def numba_cache_dir() -> Path:
    """JIT のキャッシュ置き場。

    **単体 EXE の中には書けません。** PyInstaller が展開する `_MEIPASS` は
    実行のたびに作り直される一時フォルダなので、そこに置くとキャッシュが
    毎回捨てられます。書けて、かつ残る場所（Windows なら `%LOCALAPPDATA%`）に
    向けます。`tools/build_exe.py` の説明も参照してください。
    """
    override = os.environ.get("MOVO_NUMBA_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        return Path(base) / "movo" / "numba-cache"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "movo" / "numba-cache"


def prepare_numba_cache() -> str | None:
    """`NUMBA_CACHE_DIR` を «子プロセスが共有できる場所» に向ける。

    子は環境変数を引き継ぐので、ここで 1 回設定すれば全員が同じ場所を見ます。
    設定できなかった（書けない）ときは None を返して、黙って先へ進みます
    （キャッシュが効かないだけで、絵は変わりません）。
    """
    directory = numba_cache_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".writable"
        probe.write_text("1", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as error:
        logger.verbose(f"Numba のキャッシュ置き場を用意できませんでした（{error}）。JIT は毎回コンパイルされます")
        return None
    os.environ["NUMBA_CACHE_DIR"] = str(directory)
    return str(directory)


def cache_looks_warm() -> bool:
    """JIT のキャッシュが既にできているか。

    Numba は `.nbi`（索引）と `.nbc`（本体）を書きます。1 つでもあれば
    «一度は通った» と見なします。**中身の正しさは見ません** — 版が変われば
    Numba 自身が作り直すので、ここで見るべきは «作り直しが要りそうか» だけです。
    """
    directory = numba_cache_dir()
    try:
        return any(directory.glob("**/*.nbi"))
    except OSError:
        return False


def warm_up_jit(session: Any = None) -> float:
    """親で 1 回だけ JIT を走らせて、キャッシュを作っておく。

    **子を起こす «前» に呼びます。** これをしないと 12 プロセスが同時に同じ
    関数をコンパイルし始めます。実測で **1 フレーム目に 10.6 秒** かかったので、
    12 並列なら 2 分がまるごと «コンパイルの待ち» に消えます。

    暖機の仕方は 2 通りで、上から順に試します。

    1. `movo.renderer.warm_up_kernels()`（あればこれがいちばん確実）
    2. **親のセッションで 1 フレームだけ描く。** カーネルは実際に呼ばれた形で
       しかコンパイルされないので、そのプロジェクトで «本当に使う» ものだけを
       暖められます。用意された暖機関数より無駄がありません。

    キャッシュが既にできているときは何もしません。**暖機そのものにも
    3 秒前後かかる**ので、要らないときにやると純粋な足し算になります。

    @returns かかった秒数（0 なら暖機しなかった）
    """
    if cache_looks_warm():
        return 0.0

    warm = bridge.pick("movo.renderer", "warm_up_kernels", "warmUpKernels")
    if not getattr(warm, "movo_not_connected", False):
        started = time.perf_counter()
        try:
            warm()
            return time.perf_counter() - started
        except Exception as error:  # noqa: BLE001
            logger.verbose(f"JIT の暖機に失敗しました（{error}）")

    if session is None:
        return 0.0
    started = time.perf_counter()
    try:
        session["renderer"].render_frame(0)
    except Exception as error:  # noqa: BLE001 - 暖機に失敗しても描画は続けられる
        logger.verbose(f"1 フレーム描いての暖機に失敗しました（{error}）。各プロセスでコンパイルされます")
        return 0.0
    return time.perf_counter() - started


# ══════════════════════════════════════════════════════════════════
#  «割れるか» と «どう割るか»
# ══════════════════════════════════════════════════════════════════


def resolve_job_count(value: Any) -> int:
    """`--jobs` の値を «同時に走らせる数» に直す。

    `auto` はコア数 - 1 です。1 つ空けておくのは、描いている間に他のことが
    できるようにするためです（全部使うと操作までもたつきます）。

    @returns 1 なら «1 本で描く»
    """
    if value is None or value is False:
        return 1
    cores = max(1, os.cpu_count() or 1)
    # 値を書かずに `--jobs` だけ渡すと True になります。auto と同じ扱いにします。
    if value is True or value == "auto":
        return max(1, cores - 1)
    try:
        jobs = int(float(value))
    except (TypeError, ValueError):
        logger.warn(f'--jobs の値 "{value}" が読めません。1 本で描きます（数字か auto を指定してください）')
        return 1
    if jobs < 1:
        logger.warn(f'--jobs の値 "{value}" が読めません。1 本で描きます（数字か auto を指定してください）')
        return 1
    return min(MAX_JOBS, jobs)


def plan_chunks(start_frame: int, end_frame: int, jobs: int) -> list[dict[str, int]]:
    """フレーム番号の範囲を «なるべく均等に» 区間へ割る。

    **秒ではなくフレーム境界で割ります。** 秒で割ると、端数のフレームが重複したり
    抜けたりして、繋いだときに尺がずれます。
    """
    total = end_frame - start_frame + 1
    count = max(1, min(int(jobs), max(1, total // MIN_FRAMES_PER_CHUNK)))
    # 割り切れない分は先頭の区間から 1 フレームずつ配ります。最後の区間だけが
    # 極端に短い（＝そこだけ早く終わって 1 コアが遊ぶ）のを避けるためです。
    base = total // count
    remainder = total % count
    chunks: list[dict[str, int]] = []
    cursor = start_frame
    for i in range(count):
        frames = base + (1 if i < remainder else 0)
        if frames <= 0:
            continue
        chunks.append(
            {"index": len(chunks), "startFrame": cursor, "endFrame": cursor + frames - 1, "frames": frames}
        )
        cursor += frames
    return chunks


def warmup_frames_for(project: dict) -> int:
    """区間の頭に必要な助走フレーム数。

    frameEcho と slitScan は «直前の何フレームか» を覚えています。覚えていられる
    上限は `render.frameHistory`（既定 16）なので、それだけ助走すれば
    **1 本で描いたときとまったく同じ状態** から区間を描き始められます。
    どちらも使っていなければ 0 です（＝助走の分の無駄は一切ありません）。
    """
    text = json.dumps(project.get("scenes") or [], ensure_ascii=False)
    if '"frameEcho"' not in text and '"slitScan"' not in text:
        return 0
    history = (project.get("render") or {}).get("frameHistory")
    return min(240, max(1, round(float(history if history is not None else 16))))


def parallel_blockers(session: Any, context: dict[str, Any]) -> list[str]:
    """並列にできない理由を挙げる。空の一覧なら並列にできる。

    **黙って壊れるのがいちばん困る** ので、判定はここに全部集めて、呼ぶ側は
    «理由を読み上げて 1 本に落とす» だけにしています。
    """
    reasons: list[str] = []
    output_format = context["format"]
    if context["jobs"] < 2:
        reasons.append("--jobs が 1 です")
    if output_format == "gif":
        reasons.append("gif は区間に割って繋げません（1 つのファイルに全フレームを詰める形式のため）")
    if output_format == "wav":
        reasons.append("wav は音だけの書き出しなので、割っても速くなりません")
    if output_format in ("mp4", "webm", "mov") and not bridge.find_ffmpeg():
        reasons.append("繋ぐのに ffmpeg が要ります（見つかりませんでした）")
    if context["outputs"] > 1:
        reasons.append("1 回で何通りも書き出す指定（output の配列）には未対応です")
    if context["frames"] < MIN_FRAMES_PER_CHUNK * 2:
        reasons.append(
            f'フレームが {context["frames"]} 枚しかありません'
            f"（割るなら 1 区間 {MIN_FRAMES_PER_CHUNK} 枚以上は欲しい）"
        )
    # **インラインのプロジェクト（`make-mv` / `skill render`）も割れます。**
    # 元になった JSON を子へそのまま送って、同じ手順で組み立て直させます。
    #
    # ⚠ `session["file"]` の «存在» だけで判定してはいけません。インラインのとき
    # `file` には出力先（.mp4）が入るので、**前に書き出した動画が残っていると
    # 判定をすり抜け**、子が .mp4 を JSON として読んで
    # `'utf-8' codec can't decode byte 0xaf` で落ちます。実際にそうなりました。
    inline = session.get("inlineProject") if hasattr(session, "get") else None
    file = session.get("file") if hasattr(session, "get") else None
    if inline is None and (not file or not Path(file).exists()):
        reasons.append("子プロセスに渡せるプロジェクトファイルがありません")
    # 軌跡（linePath.followLayer）は «描き始めからずっと» 積み上がるので、区間の頭を
    # 助走しても再現できません。繋ぎ目で軌跡が飛ぶくらいなら、時間がかかっても
    # 1 本で描くほうがましです。
    scenes = json.dumps((session["project"] or {}).get("scenes") or [], ensure_ascii=False)
    if '"followLayer"' in scenes:
        reasons.append(
            "linePath.followLayer の軌跡は描き始めから積み上がるので、区間に割ると繋ぎ目で軌跡が飛びます"
        )
    return reasons


def estimate_bytes_per_job(session: Any) -> int:
    """1 プロセスが使うメモリのざっくりした見積もり（バイト）。

    12 プロセスが同時に素材を読むので、大きな素材だと «全部立ち上げた瞬間に OOM»
    になり得ます。桁が合っていれば «この並列数は無理» の判断はできるので、
    精度より «安全側に倒すこと» を優先しています。
    """
    timeline = session["timeline"]
    assets_bytes = 0
    assets = session.get("assets")
    for attribute in ("images", "audio"):
        store = getattr(assets, attribute, None) if assets is not None else None
        if not store:
            continue
        try:
            for item in store.values():
                data = getattr(item, "data", None)
                if data is not None:
                    assets_bytes += getattr(data, "nbytes", 0) or len(data)
                for channel in getattr(item, "channels", None) or []:
                    assets_bytes += getattr(channel, "nbytes", 0) or 0
        except Exception:  # noqa: BLE001 - 見積もりなので、取れなければ 0 で構わない
            pass
    # 合成の途中でフレームバッファが何枚も要ります（シーン用・レイヤー用・
    # エフェクトの前後）。実測に近づけて 8 枚ぶんとしています。
    frames = timeline["width"] * timeline["height"] * 4 * 8
    # Python + NumPy + Numba の常駐分。JS 版（96MB）より重いので多めに見ます。
    return frames + assets_bytes + 220 * 1024 * 1024


def available_memory_bytes() -> int:
    """空きメモリ（バイト）。取れなければ 0。

    `psutil` は依存に足したくない（EXE に同梱するものを増やしたくない）ので、
    OS ごとに標準の口だけを使います。取れない環境では «見積もらない» を選び、
    利用者の指定した並列数をそのまま使います。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:  # noqa: BLE001
            return 0
        return 0
    try:
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def limit_jobs_by_memory(session: Any, jobs: int) -> dict[str, Any]:
    """空きメモリを見て並列数を抑える。"""
    per_job = estimate_bytes_per_job(session)
    free = available_memory_bytes()
    if free <= 0:
        # 測れない環境では «測れないこと» を理由に減らしません。指定どおり並べます。
        return {"jobs": jobs}
    # 空きを全部使い切るとページングで遅くなるので 70% までにします。
    affordable = max(1, int((free * 0.7) // per_job))
    if jobs <= affordable:
        return {"jobs": jobs}

    def mb(value: int) -> str:
        return f"{value / 1024 / 1024:.0f}MB"

    return {
        "jobs": affordable,
        "warning": (
            f"1 プロセスあたり約 {mb(per_job)} 使う見込みで、空きメモリは {mb(free)} です。"
            f"{jobs} 並列だと足りないので {affordable} に減らします"
            "（他のアプリを閉じるともっと並べられます）"
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  子プロセス側
# ══════════════════════════════════════════════════════════════════


def child_render_options(cli_options: dict[str, Any] | None) -> dict[str, Any]:
    """子プロセスに引き継ぐ «絵が変わる指定»。

    **ここに挙げ忘れた指定は子に伝わりません＝絵が変わります。** 品質・シード・
    レンダラー・素材の差し替えは必ず並べてください。JS 版は引数の文字列に
    直して渡していたので «並べ忘れ» が起きやすい形でしたが、Python では
    辞書のまま渡せるので、ここを 1 か所読めば全部分かります。
    """
    cli = cli_options or {}
    return {
        "quality": cli.get("quality"),
        "renderer": cli.get("renderer"),
        "super_sample": cli.get("superSample"),
        "seed": cli.get("seed"),
        "variant": cli.get("variant"),
        "params": cli.get("params"),
        "set": cli.get("set"),
        "no_cache": cli.get("noCache") is True or cli.get("cache") is False,
        "generate_assets": cli.get("generate") is not False,
        "strict_plugins": cli.get("strict") is not False,
        "dry_run_ai": cli.get("dryRun") is True,
    }


_PROGRESS_QUEUE = None


def _init_worker(queue, numba_cache: str | None) -> None:
    """子プロセスの初期設定。

    - 進捗の報告口（Queue）を覚える
    - `NUMBA_CACHE_DIR` を親と同じ場所に向ける（**親が設定した環境変数は
      `spawn` でも引き継がれますが、明示しておくほうが読み手に分かります**）
    - 子の標準出力は親と混ざるので、ログ水準を下げて黙らせる
    """
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = queue
    if numba_cache:
        os.environ["NUMBA_CACHE_DIR"] = numba_cache
    logger.set_level("error")


def render_chunk(spec: dict[str, Any]) -> dict[str, Any]:
    """区間を 1 つ描く（子プロセスで動きます）。

    親が持っているセッションは «描くための一式» なので、そのまま送れません
    （レンダラーもキャッシュも送れません）。**プロジェクトファイルから
    組み立て直します。** 同じ JSON からは同じ状態になるので、これで
    1 本で描いたときと同じ絵になります。
    """
    from .pipeline import create_session, render_video

    index = spec["index"]

    def report(done: int) -> None:
        if _PROGRESS_QUEUE is not None:
            try:
                _PROGRESS_QUEUE.put_nowait((index, done))
            except Exception:  # noqa: BLE001 - 進捗が届かなくても描画は続ける
                pass

    session = create_session(spec["file"], spec["sessionOptions"])
    fps = session["timeline"]["fps"]
    result = render_video(
        session,
        {
            "output": spec["output"],
            "format": spec["format"],
            # resolve_range は秒で受けるので、フレーム境界を秒に直して渡します。
            # 終わりは «次のフレームの頭» にします（endFrame = round(to*fps)-1 のため）。
            "from": spec["startFrame"] / fps,
            "to": (spec["endFrame"] + 1) / fps,
            # 音は親が最後にまとめて載せます。
            "audio": False,
            # 閃光検査も親が最後に 1 回だけ回します（理由は _scan_flash を参照）。
            "check_flash": False,
            "quiet": True,
            "warmup": spec["warmup"],
            "report_progress": report,
        },
    )
    return {"index": index, "path": result["path"], "frames": result["frames"]}


def chunk_output_looks_sane(file: str) -> bool:
    """区間の書き出しが «本当にできたか» を確かめる。

    **終了コードだけを信じてはいけません。** 実際に、書き出し先が書けない場所
    だったときに ffmpeg が落ちても、子プロセスが正常終了することがありました
    （ffmpeg が死んだあとの書き込みが待ちっぱなしになり、そのまま何もせず
    終わるため）。そのまま繋ぐと «エラーも出ないのに 1/4 の長さの動画» が
    できます。いちばん困る壊れ方なので、ファイルの有無と大きさで裏を取ります。
    """
    try:
        stat = Path(file).stat()
    except OSError:
        return False
    return stat.st_size > 0


# ══════════════════════════════════════════════════════════════════
#  ffmpeg まわり
# ══════════════════════════════════════════════════════════════════


def _run_ffmpeg(args: list[str], what: str) -> None:
    """ffmpeg を 1 回叩く。失敗したら stderr を添えて投げる。"""
    ffmpeg = bridge.find_ffmpeg()
    if not ffmpeg:
        raise MovoError(ErrorCodes.MOVO_FFMPEG_NOT_FOUND, f"{what}に ffmpeg が要ります")
    completed = subprocess.run(
        [ffmpeg["path"], *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        raise MovoError(
            ErrorCodes.MOVO_RENDERER_UNAVAILABLE, f"{what}に失敗しました", hint=(completed.stderr or "").strip()
        )


def measure_duration_seconds(file: str) -> float | None:
    """書き出した動画の尺を測る（秒）。ffprobe が無ければ None。

    **繋いだ結果が «短くなっていないか» を確かめるため** に使います。concat は
    読めないファイルが並びに混ざっていても 0 で終わることがあり、そのときは
    気付かないまま短い動画ができます。尺を測れば一発で分かります。
    """
    ffprobe = bridge.find_ffprobe()
    if not ffprobe:
        return None
    completed = subprocess.run(
        [
            ffprobe["path"],
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float((completed.stdout or "").strip())
    except ValueError:
        return None


def assert_duration_matches(file: str, expected_seconds: float, chunk_count: int) -> None:
    """繋いだ結果が «要求した尺» になっているか確かめる。

    区間が 1 つでも抜けると、ここで «6 秒のはずが 1.5 秒» のように現れます。
    ffprobe が無い環境では測れないので、その旨だけ残して先へ進みます
    （測れないことを理由に書き出しを止めるほどではありません）。
    """
    measured = measure_duration_seconds(file)
    if measured is None:
        logger.verbose("ffprobe が無いので、繋いだ結果の尺は確かめられませんでした")
        return
    # 容器の丸めがあるので 0.2 秒までは許します。区間 1 つぶんの抜けはこれより
    # ずっと大きいので、見逃す心配はありません。
    if abs(measured - expected_seconds) <= 0.2:
        return
    raise MovoError(
        ErrorCodes.MOVO_INTERNAL,
        f"繋いだ結果が {measured:.2f} 秒で、要求した {expected_seconds:.2f} 秒と合いません",
        hint=(
            f"{chunk_count} 区間のうち、書き出せていないものがあるようです。\n"
            "--keep-parts を付けて描き直すと、区間ごとのファイルが残るので確かめられます"
        ),
    )


def _sequence_files_in_range(directory: str, rng: dict) -> list[str]:
    """PNG 連番のうち «今回描いた範囲» のファイルを番号順に返す。

    ファイル名の数字は絶対フレーム番号です。同じフォルダに前回の連番が残って
    いることがあるので、範囲で絞ります（絞らないと、描いていないフレームまで
    検査に混ざって判定が変わります）。
    """
    import re

    found: list[tuple[float, str]] = []
    try:
        entries = sorted(Path(directory).iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.suffix.lower() != ".png":
            continue
        match = re.search(r"(\d+)\D*$", entry.name)
        frame = int(match.group(1)) if match else None
        if frame is not None and not (rng["startFrame"] <= frame <= rng["endFrame"]):
            continue
        found.append((float(frame) if frame is not None else float("inf"), entry.name))
    return [name for _, name in sorted(found)]


def _scan_flash_of_video(file: str, project: dict, timeline: dict, options: dict):
    """書き出した動画を読み直して閃光検査を回す。

    **並列にすると閃光検査が区間ごとに分断されます。** 各プロセスは自分の区間しか
    見ないので、区間をまたぐ明滅（区間の最後で白く飛んで、次の区間の頭で戻る）を
    見落とします。1 秒に何回光ったかを数える検査なので、境界で数え直しが起きるのは
    そのまま «検査が甘くなる» ことを意味します。

    そこで **並列のときは、繋ぎ終わった 1 本をまとめて 1 回だけ検査します。**
    区間ごとの結果を足し合わせる案もありましたが、境界の 1 フレーム差を復元できず、
    «足し算は合っているのに見落とす» ことになるのでやめました。実際に配る 1 本を
    測るのがいちばん確かです。

    読み直しは ffmpeg で生の RGBA に戻します。FlashGuard はもともと画面を
    32x18 の格子に落として測るので、**縮めてから渡しても結果は変わりません**。
    原寸のまま流すと 153 秒で 4.7GB 流れることになるので、幅 320 に縮めます。

    ⚠ **検査係は «縮めたあとの寸法» で作ります。**
    `FlashGuard` は作るときの寸法から «どの画素を見るか» の索引を先に計算して
    持ちます。原寸（640x360）で作った検査係に縮めた 320x180 のフレームを渡すと
    索引が配列からはみ出して落ちます（実際に `--jobs 8` で落ちていました）。
    ここは «呼ぶ側» を直しました — 縮めるのはこちらの都合なので、その寸法を
    知っているこちらが検査係を作るのが筋だからです。
    """
    from .pipeline import create_flash_guard

    ffmpeg = bridge.find_ffmpeg()
    if not ffmpeg:
        return None
    width = min(timeline["width"], 320)
    height = max(1, round(timeline["height"] * width / timeline["width"]))
    frame_bytes = width * height * 4
    if not bridge.is_connected("movo.core.bitmap"):
        return None
    guard = create_flash_guard(project, {**timeline, "width": width, "height": height}, options)
    if guard is None:
        return None

    args = [
        ffmpeg["path"],
        "-v", "error",
        "-i", file,
        "-vf", f"scale={width}:{height}:flags=area",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "pipe:1",
    ]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pending = b""
    frames = 0
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(frame_bytes)
        if not chunk:
            break
        pending += chunk
        while len(pending) >= frame_bytes:
            guard.push(bridge.to_bitmap(width, height, pending[:frame_bytes]))
            pending = pending[frame_bytes:]
            frames += 1
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    process.wait()
    if process.returncode != 0 or frames == 0:
        logger.verbose(f"閃光検査のための読み直しに失敗しました: {stderr.strip()}")
        return None
    return guard.report()


def _scan_flash_of_png_sequence(directory: str, guard, rng: dict):
    """書き出した PNG 連番を読み直して閃光検査を回す（ffmpeg が要りません）。"""
    files = _sequence_files_in_range(directory, rng)
    if not files:
        return None
    for name in files:
        guard.push(bridge.decode_png(Path(directory, name).read_bytes()))
    return guard.report()


# ══════════════════════════════════════════════════════════════════
#  本体
# ══════════════════════════════════════════════════════════════════


def render_video_parallel(session: Any, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """区間に割って同時に描き、繋いで 1 本にする（`movo render --jobs N`）。

    並列にできないと判断したときは **黙って壊れず、理由を言って 1 本で描きます**。
    戻り値は `render_video` と同じ形なので、呼ぶ側はどちらで描かれたか気にせずに
    済みます（`parallel` と `jobs` だけ増えます）。
    """
    from .pipeline import create_flash_guard, render_video, resolve_output_path, resolve_range

    options = options or {}
    project = session["project"]
    timeline = session["timeline"]
    project_root = session["projectRoot"]
    jobs_requested = max(1, int(options.get("jobs") or 1))
    rng = resolve_range(timeline, options)

    # 書き出し先と形式は render_video と同じ決め方をします（1 本目だけを見ます）。
    declared = project.get("output")
    declarations = declared if isinstance(declared, list) else [declared or {}]
    declaration = declarations[0] if declarations else {}
    negotiate_format = bridge.pick("movo.exporters", "negotiate_format", "negotiateFormat")
    negotiated = negotiate_format(options.get("format") or declaration.get("format") or "mp4")
    if negotiated.get("downgraded"):
        logger.warn(negotiated.get("reason", ""))
    output_format = negotiated["format"]
    output_path = resolve_output_path(
        explicit=options.get("output"),
        project_output=declaration.get("path"),
        project_root=project_root,
        name=(project.get("project") or {}).get("name") or "output",
        format=output_format,
    )

    total_frames = rng["endFrame"] - rng["startFrame"] + 1
    blockers = parallel_blockers(
        session,
        {"format": output_format, "jobs": jobs_requested, "frames": total_frames, "outputs": len(declarations)},
    )
    if blockers:
        for reason in blockers:
            logger.warn(f"並列レンダリングは使えません: {reason}")
        logger.info("  1 本で描きます")
        return render_video(session, options)

    memory = limit_jobs_by_memory(session, jobs_requested)
    if memory.get("warning"):
        logger.warn(memory["warning"])
    jobs = memory["jobs"]
    if jobs < 2:
        logger.info("  メモリが足りないので 1 本で描きます")
        return render_video(session, options)
    cores = max(1, os.cpu_count() or 1)
    if jobs > cores:
        logger.warn(f"--jobs {jobs} はコア数 {cores} より多いので、これ以上は速くなりません")
    elif jobs > max(2, cores // 2):
        # **コアの数だけ並べても、そのぶん速くはなりません。** Python 版の
        # 描画は全画面のエフェクトが多く、**メモリ帯域が先に尽きます**
        # （README.ja.md の «判断 1» と同じ理由）。実測では 12 スレッドの機械で
        # 4 並列が 2.6 倍、11 並列は 1.4 倍まで落ちました。JS 版で 6.3 倍出たのは
        # 1 フレーム 1.5〜2.5 秒という «計算が支配的» な作りだったからです。
        logger.info(
            f"  --jobs {jobs} はこの機械（{cores} スレッド）では効きが鈍るかもしれません。"
            f"速くならないときは --jobs {max(2, cores // 2)} も試してください"
        )

    chunks = plan_chunks(rng["startFrame"], rng["endFrame"], jobs)
    jobs = min(jobs, len(chunks))
    warmup = warmup_frames_for(project)
    if getattr(sys, "frozen", False) and "_MEI" in str(getattr(sys, "_MEIPASS", "")):
        # **1 ファイル形式の EXE は、子プロセスを起こすたびに中身を展開します。**
        # 実測で 10 秒の動画が「ソースから 13 秒／1 ファイル EXE から 42 秒」でした。
        # 黙って遅いと «並列にしたのに遅くなった» としか見えないので、理由を出します。
        logger.info("  1 ファイル形式の EXE では、子プロセスの起動に展開の時間がかかります")
        logger.info("  （--onedir で作り直すと速くなります）")

    logger.step(
        f"{len(chunks)} 区間に割って {jobs} 並列で描きます"
        f'（全 {total_frames} フレーム / {total_frames / timeline["fps"]:.1f} 秒'
        f'{f" / 助走 {warmup} フレーム" if warmup > 0 else ""}）'
    )

    # ── JIT のキャッシュをそろえてから子を起こす ────────────
    cache_dir = prepare_numba_cache()
    warm_seconds = warm_up_jit(session)
    if warm_seconds > 0:
        where = f"（{cache_dir}）" if cache_dir else ""
        logger.info(
            f"  JIT を親で 1 回暖機しました（{warm_seconds:.1f} 秒）。"
            f"子プロセスはキャッシュを読むだけになります{where}"
        )

    is_sequence = output_format == "png-sequence"
    # PNG 連番は繋ぐ必要がありません。**各区間が絶対フレーム番号で書くので、
    # 同じフォルダに書かせるだけで連番が続きます。** 中間ファイルも要りません。
    work_dir = (
        None
        if is_sequence
        else Path(output_path).parent / f".movo-parallel-{Path(output_path).stem}"
    )
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    def part_of(chunk: dict) -> str:
        if is_sequence:
            return output_path
        return str(work_dir / f'part-{chunk["index"]:03d}{Path(output_path).suffix}')

    session_options = child_render_options(options.get("cli_options"))
    # インラインのプロジェクトは、ファイルではなく **JSON そのもの** を子へ渡します。
    # 素の辞書なので pickle でそのまま送れます。
    if session.get("inlineProject") is not None:
        session_options = {
            **session_options,
            "inline_project": session["inlineProject"],
            "project_root": session["projectRoot"],
        }
    specs = [
        {
            "index": chunk["index"],
            "file": session["file"],
            "sessionOptions": session_options,
            "startFrame": chunk["startFrame"],
            "endFrame": chunk["endFrame"],
            "output": part_of(chunk),
            "format": output_format,
            "warmup": warmup,
        }
        for chunk in chunks
    ]

    # ── 描く ────────────────────────────────────────────
    started = time.perf_counter()
    done_by_chunk = [0] * len(chunks)
    progress = None if options.get("quiet") else logger.progress(total_frames, "render")

    # `spawn` を明示します。Windows は元々 spawn ですが、Linux の既定（fork）だと
    # 親が抱えている NumPy / Numba の状態を引き継いで、**親のスレッド数の設定が
    # 子に効かない**などの差が出ます。どの OS でも同じ動きにしたいので揃えます。
    context = mp.get_context("spawn")
    # **Manager は挟みません。** `context.Manager().Queue()` はプロセスをもう 1 つ
    # 起こし、子はその «代理» 越しに進捗を送ります。この代理の接続が途中で閉じると
    # 子の送信スレッドが `OSError: handle is closed` を出し、**進捗の報告が失敗した
    # だけなのにプール全体が壊れます**（12 分のレンダリングが最後まで行かずに
    # 落ちました。絵とは何の関係もない失敗です）。
    #
    # 素の `Queue` なら代理も余分なプロセスも要りません。子の生成時に渡るので
    # spawn でもそのまま使えます。
    queue = context.Queue()

    failure: dict[str, Any] | None = None
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=context, initializer=_init_worker, initargs=(queue, cache_dir)
    ) as pool:
        futures = {pool.submit(render_chunk, spec): spec for spec in specs}
        pending = set(futures)
        retried: set[int] = set()
        while pending:
            finished, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            # 進捗を吸い出す（待っている間に溜まっているぶんも含めて）
            while not queue.empty():
                try:
                    index, count = queue.get_nowait()
                except Exception:  # noqa: BLE001
                    break
                done_by_chunk[index] = count
            if progress is not None:
                progress.update(min(total_frames, sum(done_by_chunk)))

            for future in finished:
                spec = futures[future]
                error = future.exception()
                if error is None and (is_sequence or chunk_output_looks_sane(spec["output"])):
                    continue
                if error is None:
                    error = MovoError(
                        ErrorCodes.MOVO_INTERNAL,
                        f'区間 {spec["index"]} は正常に終わったのに、書き出したファイルがありません',
                    )
                if spec["index"] not in retried:
                    # **1 回だけやり直す。** ファイルの掴み合いのような一過性の失敗は
                    # やり直せば通ります。それでも駄目なら、その区間だけを名指しで
                    # 止めます（成功した区間は残すので、直したあとに描き直す量が減ります）。
                    logger.verbose(f'区間 {spec["index"]} をやり直します: {error}')
                    retried.add(spec["index"])
                    done_by_chunk[spec["index"]] = 0
                    retry = pool.submit(render_chunk, spec)
                    futures[retry] = spec
                    pending.add(retry)
                    continue
                failure = {"chunk": chunks[spec["index"]], "error": error}
                # 走っている途中の子は止めます（放っておくと親が終わってもコアを
                # 食い続けます）。
                for other in pending:
                    other.cancel()
                pending = set()
                break

    if failure is not None:
        if progress is not None:
            progress.done("")
        chunk = failure["chunk"]
        fps = timeline["fps"]
        raise MovoError(
            ErrorCodes.MOVO_INTERNAL,
            str(failure["error"]),
            hint=(
                "この区間だけ描き直すには:\n"
                f'  movo render {Path(session["file"]).name} '
                f'--from {chunk["startFrame"] / fps:.3f} --to {(chunk["endFrame"] + 1) / fps:.3f}\n'
                "うまくいかないときは --jobs を外して 1 本で描いてください"
                + (f"\n描けた区間は {work_dir} に残してあります" if work_dir else "")
            ),
        )

    # ── 繋ぐ ────────────────────────────────────────────
    audio_path = None
    if is_sequence:
        # 連番は繋ぎませんが、**枚数だけは数えます**。子プロセスが «成功したのに
        # 書けていない» ことが起こり得るので、黙って歯抜けの連番を渡さないためです。
        written = len(_sequence_files_in_range(output_path, rng))
        if written < total_frames:
            raise MovoError(
                ErrorCodes.MOVO_INTERNAL,
                f"PNG が {written} 枚しかありません（{total_frames} 枚のはずです）",
                hint="--jobs を外して 1 本で描き直すと、どこで失敗しているか分かります",
            )
    else:
        list_file = work_dir / "parts.txt"
        # concat デマルチプレクサは «file '...'» の並びを読みます。Windows の
        # 区切りはそのままだとエスケープ扱いになるので / に直します。
        lines = "\n".join(f"file '{part_of(chunk).replace(chr(92), '/')}'" for chunk in chunks)
        list_file.write_text(lines + "\n", encoding="utf-8")

        wants_audio = bool(session.get("audio")) and options.get("audio") is not False
        merged = str(work_dir / f"merged{Path(output_path).suffix}") if wants_audio else output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.verbose("区間を繋いでいます（再エンコードはしません）")
        # `-c copy` なので繋ぎ目で画質は落ちません。全部同じ設定で描いているから
        # できることです。
        _run_ffmpeg(
            ["-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", merged],
            "区間の連結",
        )
        assert_duration_matches(merged, total_frames / timeline["fps"], len(chunks))

        if wants_audio:
            import hashlib

            encoded = bytes(bridge.encode_wav(session["audio"]))
            digest = hashlib.sha256(encoded).hexdigest()[:12]
            cache_root = getattr(session["cache"], "root", None) or str(Path(project_root) / "cache")
            audio_file = Path(cache_root) / "audio" / f"mix-{digest}.wav"
            audio_file.parent.mkdir(parents=True, exist_ok=True)
            audio_file.write_bytes(encoded)
            audio_path = str(audio_file)
            logger.verbose("音を載せています")
            audio_codec = declaration.get("audioCodec") or ("libopus" if output_format == "webm" else "aac")
            args = ["-y", "-v", "error", "-i", merged, "-i", audio_path, "-c:v", "copy", "-c:a", audio_codec]
            if declaration.get("audioBitrate"):
                args += ["-b:a", str(declaration["audioBitrate"])]
            # 曲のほうが長いことがあるので、映像の長さに合わせて切ります。
            args += ["-shortest", output_path]
            _run_ffmpeg(args, "音の多重化")

    elapsed = time.perf_counter() - started
    if progress is not None:
        progress.done(f"{total_frames} frames -> {output_path}")

    # ── 閃光検査（まとめて 1 回）────────────────────────
    guard = create_flash_guard(project, timeline, options)
    flash_report = None
    if guard is not None:
        # 長い動画だと数十秒かかるので、黙って止まって見えないように一言出します。
        logger.info("  書き出した動画を読み直して、光過敏性発作の検査をします")
        flash_report = (
            _scan_flash_of_png_sequence(output_path, guard, rng)
            if is_sequence
            else _scan_flash_of_video(output_path, project, timeline, options)
        )
        if not flash_report:
            # **黙って検査を飛ばすのがいちばん困る** ので、はっきり言います。
            logger.warn("光過敏性発作の検査ができませんでした（書き出した動画を読み直せませんでした）")
            logger.warn("  気になる場合は --jobs を外して 1 本で描き直すと、描きながら検査します")
        elif not flash_report.get("ok", True):
            describe = bridge.pick("movo.core.flash_guard", "describe_flash_report", "describeFlashReport")
            if not getattr(describe, "movo_not_connected", False):
                for line in describe(flash_report):
                    logger.warn(line)

    if work_dir is not None and options.get("keep_parts") is not True:
        shutil.rmtree(work_dir, ignore_errors=True)
    elif work_dir is not None:
        logger.info(f"  区間ごとのファイルを残しました: {work_dir}")

    return {
        "path": output_path,
        "frames": total_frames,
        "format": output_format,
        "outputs": [{"path": output_path, "format": output_format, "frames": total_frames}],
        "flashReport": flash_report,
        "elapsedSeconds": elapsed,
        "fpsAchieved": total_frames / max(0.001, elapsed),
        "range": rng,
        "audioPath": audio_path,
        "parallel": True,
        "jobs": jobs,
        "chunks": len(chunks),
    }
