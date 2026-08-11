"""段階つきのログと進捗バー。

**警告は捨てずに溜めます。** レンダリングは数分かかるので、途中で流れた警告を
最後にもう一度まとめて出さないと «気付かれない» ためです。
"""

from __future__ import annotations

import os
import sys
import time

LEVELS = {"silent": 0, "error": 1, "warn": 2, "info": 3, "verbose": 4, "debug": 5}


def _supports_color() -> bool:
    """色を付けてよいか。

    ``NO_COLOR`` は事実上の標準なので尊重します。パイプに流したときに
    エスケープが混ざると、ログを機械で読む側が困ります。
    """
    if os.environ.get("FORCE_COLOR") == "0":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") == "1":
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


_COLOR = _supports_color()


def _paint(code: int):
    def apply(text: object) -> str:
        return f"[{code}m{text}[0m" if _COLOR else str(text)

    return apply


class style:
    """文字の飾り。色が使えない環境では素通しになります。"""

    bold = staticmethod(_paint(1))
    dim = staticmethod(_paint(2))
    red = staticmethod(_paint(31))
    green = staticmethod(_paint(32))
    yellow = staticmethod(_paint(33))
    blue = staticmethod(_paint(34))
    magenta = staticmethod(_paint(35))
    cyan = staticmethod(_paint(36))
    gray = staticmethod(_paint(90))


class Progress:
    """1 行で書き換わる進捗バー。

    パイプに流されているときは **10% ごとに 1 行**出すだけに落とします。
    ``\\r`` で書き換える表示をファイルに落とすと、巨大な 1 行になるためです。
    """

    def __init__(self, total: int, label: str = "", enabled: bool = True) -> None:
        self.total = total
        self.label = label
        self.enabled = enabled
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.width = 28
        self.start = time.time()
        self._last = -1

    def update(self, current: int) -> None:
        if not self.enabled or self.total <= 0:
            return
        ratio = min(1.0, current / self.total)
        pct = int(ratio * 100)
        if pct == self._last and current < self.total:
            return
        self._last = pct
        filled = int(ratio * self.width + 0.5)
        elapsed = time.time() - self.start
        eta = elapsed / ratio - elapsed if ratio > 0 else 0.0
        bar = "█" * filled + "░" * (self.width - filled)
        line = f"{self.label} {bar} {pct:>3}%  {current}/{self.total}  ETA {eta:.1f}s"
        if self.tty:
            sys.stdout.write(f"\r{line}   ")
            sys.stdout.flush()
        elif pct % 10 == 0:
            print(line)

    def done(self, message: str = "") -> None:
        if not self.enabled:
            return
        if self.tty:
            sys.stdout.write("\r" + " " * 90 + "\r")
            sys.stdout.flush()
        if message:
            print(style.green("✔"), message)


class Logger:
    def __init__(self, level: str = "info") -> None:
        self.level = LEVELS.get(level, LEVELS["info"])
        self.warnings: list[str] = []

    def set_level(self, level: str) -> None:
        self.level = LEVELS.get(level, self.level)

    def _should(self, level: str) -> bool:
        return self.level >= LEVELS[level]

    def error(self, *args: object) -> None:
        if self._should("error"):
            print(style.red("✖"), *args, file=sys.stderr)

    def warn(self, *args: object) -> None:
        """警告。**表示するかどうかに関わらず溜めます**（最後にまとめて出すため）。"""
        self.warnings.append(" ".join(str(a) for a in args))
        if self._should("warn"):
            print(style.yellow("⚠"), *args, file=sys.stderr)

    def info(self, *args: object) -> None:
        if self._should("info"):
            print(*args)

    def success(self, *args: object) -> None:
        if self._should("info"):
            print(style.green("✔"), *args)

    def step(self, *args: object) -> None:
        if self._should("info"):
            print(style.cyan("›"), *args)

    def verbose(self, *args: object) -> None:
        if self._should("verbose"):
            print(style.gray("·"), *args)

    def debug(self, *args: object) -> None:
        if self._should("debug"):
            print(style.gray("debug"), *args)

    def progress(self, total: int, label: str = "") -> Progress:
        return Progress(total, label, enabled=self._should("info"))


logger = Logger(os.environ.get("MOVO_LOG_LEVEL", "info"))
