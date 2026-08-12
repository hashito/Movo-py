"""movo CLI の入口。引数を読み、コマンドへ振り分け、エラーを人が読める形で出す。

**コマンドは遅延インポートしています。** `movo --version` に 1 秒かかるのは
（Numba や NumPy を読むと実際にそうなります）道具として使いづらいので、
使うコマンドのモジュールだけを読みます。
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Callable

from movo import __version__

from .args import COMMON_SCHEMA, parse_args
from .console import logger, say
from .errors import EXIT_CODES, MovoError, MovoValidationError, to_movo_error
from .help import COMMAND_HELP, MAIN_HELP

# コマンド名 → (モジュール, 関数名)。`skills` と `plugins` は複数形でも書けます
# （JS 版がそうなっており、指が勝手に s を付けるので）。
COMMANDS: dict[str, tuple[str, str]] = {
    "init": (".commands.init_cmd", "init_command"),
    "validate": (".commands.validate", "validate_command"),
    "render": (".commands.render", "render_command"),
    "frame": (".commands.render", "frame_command"),
    "frames": (".commands.render", "frames_command"),
    "preview": (".commands.preview", "preview_command"),
    "assets": (".commands.assets", "assets_command"),
    "analyze": (".commands.analyze", "analyze_command"),
    "lyrics": (".commands.lyrics", "lyrics_command"),
    "make-mv": (".commands.make_mv", "make_mv_command"),
    "profile": (".commands.profile", "profile_command"),
    "compare": (".commands.profile", "compare_command"),
    "list": (".commands.list_cmd", "list_command"),
    "skill": (".commands.skill", "skill_command"),
    "skills": (".commands.skill", "skill_command"),
    "make": (".commands.make", "make_command"),
    "params": (".commands.make", "params_command"),
    "batch": (".commands.batch", "batch_command"),
    "plugin": (".commands.plugin", "plugin_command"),
    "plugins": (".commands.plugin", "plugin_command"),
    "config": (".commands.config", "config_command"),
    "doctor": (".commands.doctor", "doctor_command"),
    "setup-ffmpeg": (".commands.setup_ffmpeg", "setup_ffmpeg_command"),
}


def _load(name: str) -> Callable[..., Any]:
    module_name, function_name = COMMANDS[name]
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, function_name)


def run(argv: list[str]) -> int:
    """@returns プロセスの終了コード"""
    options, positional = parse_args(argv, COMMON_SCHEMA)

    # **JIT のキャッシュ先は、並列でなくても毎回そろえます。**
    # 1 フレーム目のコンパイルに実測 10.6 秒かかるので、置き場が揃っていないと
    # «毎回 10 秒待つ» ことになります。並列レンダリングのときは、ここで作られた
    # キャッシュを子プロセスがそのまま読みます（`movo/cli/parallel.py`）。
    from .parallel import prepare_numba_cache

    prepare_numba_cache()

    if options.get("debug"):
        logger.set_level("debug")
    elif options.get("verbose"):
        logger.set_level("verbose")
    elif options.get("quiet"):
        logger.set_level("warn")

    if options.get("version") or (positional and positional[0] == "version"):
        say(__version__)
        return 0

    command_name = positional[0] if positional else None
    if not command_name or command_name == "help":
        topic = positional[1] if len(positional) > 1 else None
        sys.stdout.write(COMMAND_HELP.get(topic) if topic and topic in COMMAND_HELP else MAIN_HELP)
        return 0

    if command_name not in COMMANDS:
        logger.error(f"不明なコマンド: {command_name}")
        suggestion = _closest(command_name, list(COMMANDS))
        if suggestion:
            logger.info(f"  もしかして: movo {suggestion}")
        sys.stdout.write(f"\n{MAIN_HELP}")
        return 2

    if options.get("help"):
        sys.stdout.write(COMMAND_HELP.get(command_name, MAIN_HELP))
        return 0

    try:
        command = _load(command_name)
        command(positional[1:], options)
        return 0
    except SystemExit as exit_request:
        # コマンドが «終了コードだけ» を指定して抜けたとき（検証の失敗など）
        return int(exit_request.code or 0)
    except KeyboardInterrupt:
        logger.warn("中断しました")
        return 130
    except BaseException as error:  # noqa: BLE001 - 最後の砦。必ず読める形にして返す
        movo_error = error if isinstance(error, MovoError) else to_movo_error(error)
        if isinstance(movo_error, MovoValidationError):
            logger.error("プロジェクト JSON に問題があります")
        sys.stderr.write(f"\n{movo_error.format()}\n\n")
        if options.get("debug"):
            import traceback

            traceback.print_exc()
        return EXIT_CODES.get(movo_error.code, 1)


def _closest(text: str, candidates: list[str]) -> str | None:
    """打ち間違いの救済（「もしかして」）。"""
    best = None
    best_distance = float("inf")
    for candidate in candidates:
        distance = _levenshtein(text, candidate)
        if distance < best_distance:
            best_distance = distance
            best = candidate
    return best if best_distance <= max(2, len(text) // 3) else None


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[len(b)]


def _force_utf8_output() -> None:
    """端末が cp932 でも日本語を出せるようにする。

    Movo の表示はすべて日本語なので、既定の cp932 のままだと «‹› や ✔ を
    書こうとした瞬間に UnicodeEncodeError で落ちる» ことがあります。
    **エラーメッセージを出す途中で落ちるのがいちばん困る** ので、入口で
    UTF-8 に寄せ、書けない文字は置き換えて捨てます。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> None:
    # Windows で «子プロセスが親をもう一度起動する» のを止めるおまじない。
    # PyInstaller で固めた EXE では **これが無いと EXE が無限に増えます**
    # （並列レンダリングの子が、また CLI として起動してしまうため）。
    import multiprocessing

    multiprocessing.freeze_support()
    _force_utf8_output()
    # JIT のキャッシュ先は `run()` の頭でそろえます（単体 EXE の中には書けないので、
    # %LOCALAPPDATA% のような «書けて残る場所» へ向け直す必要があります）。
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
