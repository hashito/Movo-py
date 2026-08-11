"""端末への出力（色・ログ水準・進捗）。

JS 版の `packages/core/src/logger.js` に当たるものです。`movo.core.logger` が
用意できたらそちらを使い、無ければこの実装を使います。

## 決めごと

- **結果は stdout、進捗とログは stderr。** `movo skill expand ... | jq` のように
  «結果だけを配管に流す» 使い方をしたときに、進捗が混ざらないようにするためです。
- `--quiet` は **進捗だけ**を黙らせます。結果まで消えると、パイプで受けている側が
  «成功したのに何も来ない» ことになります（JS 版の profile.js が同じ理由で
  `logger.info` を避けて直接書いています）。
- 色は端末のときだけ。`NO_COLOR` があれば無条件で切ります。
"""

from __future__ import annotations

import os
import sys
import time

LEVELS = {"debug": 10, "verbose": 20, "info": 30, "warn": 40, "error": 50, "silent": 99}


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("MOVO_FORCE_COLOR") == "1":
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class Style:
    """色付け。端末でなければ素通しします。"""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text) -> str:
        if not self.enabled:
            return str(text)
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text) -> str:
        return self._wrap("1", text)

    def dim(self, text) -> str:
        return self._wrap("2", text)

    def gray(self, text) -> str:
        return self._wrap("90", text)

    def red(self, text) -> str:
        return self._wrap("31", text)

    def green(self, text) -> str:
        return self._wrap("32", text)

    def yellow(self, text) -> str:
        return self._wrap("33", text)

    def cyan(self, text) -> str:
        return self._wrap("36", text)


style = Style(_supports_color(sys.stderr))


class Progress:
    """1 行を書き換え続ける進捗表示。

    端末でないとき（ログにリダイレクトしたとき、並列レンダリングの子プロセス）は
    **1 行も出しません**。書き換え用の制御文字がそのまま残ると、ログが読めなく
    なるためです。
    """

    def __init__(self, total: int, label: str, enabled: bool) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.enabled = enabled
        self.started = time.time()
        self._last_draw = 0.0
        self.current = 0

    def update(self, done: int) -> None:
        self.current = max(0, min(self.total, int(done)))
        if not self.enabled:
            return
        now = time.time()
        # 毎フレーム描くと、描画のほうが重くなることがあります（1 フレーム 0.07 秒
        # を狙う設計なので、実際に無視できない差になります）。
        if now - self._last_draw < 0.1 and self.current < self.total:
            return
        self._last_draw = now
        ratio = self.current / self.total
        width = 24
        filled = int(width * ratio)
        elapsed = now - self.started
        eta = (elapsed / self.current) * (self.total - self.current) if self.current else 0.0
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r  {self.label} [{bar}] {self.current}/{self.total}"
            f"  {elapsed:5.1f}s 残り {eta:5.1f}s "
        )
        sys.stderr.flush()

    def done(self, message: str = "") -> None:
        if self.enabled:
            sys.stderr.write("\r" + " " * 78 + "\r")
            sys.stderr.flush()
        if message:
            logger.success(message)


class Logger:
    def __init__(self) -> None:
        self.level = LEVELS["info"]

    def set_level(self, name: str) -> None:
        self.level = LEVELS.get(name, LEVELS["info"])

    def _write(self, level: int, text: str) -> None:
        if level < self.level:
            return
        sys.stderr.write(f"{text}\n")

    def debug(self, message: str) -> None:
        self._write(LEVELS["debug"], style.gray(f"  {message}"))

    def verbose(self, message: str) -> None:
        self._write(LEVELS["verbose"], style.gray(message))

    def info(self, message: str = "") -> None:
        self._write(LEVELS["info"], message)

    def step(self, message: str) -> None:
        self._write(LEVELS["info"], f"{style.cyan('>')} {message}")

    def success(self, message: str) -> None:
        self._write(LEVELS["info"], f"{style.green('OK')} {message}")

    def warn(self, message: str) -> None:
        self._write(LEVELS["warn"], f"{style.yellow('!')} {message}")

    def error(self, message: str) -> None:
        self._write(LEVELS["error"], f"{style.red('x')} {message}")

    def progress(self, total: int, label: str) -> Progress:
        enabled = self.level <= LEVELS["info"] and _supports_color(sys.stderr)
        return Progress(total, label, enabled)


logger = Logger()


def say(line: str = "") -> None:
    """**結果**を stdout に書く。

    `--quiet` でも消えません。`movo profile` の数値や `movo skill expand` の
    JSON のように «後ろに繋いで使うもの» はここから出します。
    """
    sys.stdout.write(f"{line}\n")


def _adopt_core() -> None:
    """`movo.core.logger` があればそちらを使う（core は別担当が移植中）。"""
    try:  # pragma: no cover - core が来たときだけ通る
        from movo.core import logger as core_logger  # type: ignore
    except Exception:
        return
    if hasattr(core_logger, "logger"):
        globals()["logger"] = core_logger.logger
    if hasattr(core_logger, "style"):
        globals()["style"] = core_logger.style


_adopt_core()
