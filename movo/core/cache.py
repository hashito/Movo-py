"""内容で番地が決まるキャッシュ（仕様 30 節）。

**高くつくが必ず同じ結果になるもの**をここに置きます。AI が作った素材、
復号した画像、メッシュ、マスク、物理の軌跡、音の波形、描き終えたフレーム。

鍵は «結果を変えうる入力すべて» から作ります。1 つでも取りこぼすと、
**前に作った絵が返ってくる**という一番気付きにくい壊れ方をします。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from .hash import hash_json


class Cache:
    """ディスクとメモリの二段構えのキャッシュ。

    :param root: 置き場（普通は ``<プロジェクト>/cache``）
    :param enabled: False にすると読みも書きもしません（``--no-cache``）
    :param namespace_salt: 鍵に混ぜる値。実装の版などを入れて一斉に失効させます
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        enabled: bool = True,
        namespace_salt: dict[str, Any] | None = None,
    ) -> None:
        self.root = str(root)
        self.enabled = enabled
        self.namespace_salt = namespace_salt or {}
        self.stats = {"hits": 0, "misses": 0, "writes": 0}
        self._memory: dict[Any, Any] = {}

    def key(self, namespace: str, parts: Any) -> str:
        return hash_json({"namespace": namespace, "parts": parts, "salt": self.namespace_salt})

    def path_for(self, namespace: str, key: str, extension: str = ".bin") -> str:
        return os.path.join(self.root, namespace, f"{key}{extension}")

    def has(self, namespace: str, key: str, extension: str = ".bin") -> bool:
        if not self.enabled:
            return False
        return os.path.exists(self.path_for(namespace, key, extension))

    def read_buffer(self, namespace: str, key: str, extension: str = ".bin") -> bytes | None:
        if not self.enabled:
            return None
        try:
            with open(self.path_for(namespace, key, extension), "rb") as handle:
                data = handle.read()
        except OSError:
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        return data

    def write_buffer(self, namespace: str, key: str, buffer: bytes, extension: str = ".bin") -> str | None:
        if not self.enabled:
            return None
        path = Path(self.path_for(namespace, key, extension))
        path.parent.mkdir(parents=True, exist_ok=True)
        # 一時ファイルに書いてから差し替えます。**並列レンダリング中に
        # 書きかけのファイルを別のプロセスが読むのを防ぐ**ためです。
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as handle:
            handle.write(buffer)
        os.replace(tmp, path)
        self.stats["writes"] += 1
        return str(path)

    def read_json(self, namespace: str, key: str) -> Any:
        raw = self.read_buffer(namespace, key, ".json")
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return None

    def write_json(self, namespace: str, key: str, value: Any) -> str | None:
        text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        return self.write_buffer(namespace, key, text.encode("utf-8"), ".json")

    def memo(self, key: Any, factory: Callable[[], Any]) -> Any:
        """プロセス内だけの記憶。ディスクには触りません。

        同じフレームの中で同じ計算が何度も要求される（マスクの参照など）ときに
        効きます。ディスクに出すには小さすぎるものの置き場です。
        """
        if key in self._memory:
            self.stats["hits"] += 1
            return self._memory[key]
        self.stats["misses"] += 1
        value = factory()
        self._memory[key] = value
        return value

    def clear(self, namespace: str | None = None) -> None:
        target = os.path.join(self.root, namespace) if namespace else self.root
        shutil.rmtree(target, ignore_errors=True)
        self._memory.clear()

    def size(self) -> int:
        """ディスク上の合計バイト数。"""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total
