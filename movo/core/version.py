"""版番号。ロックファイルと doctor が同じ値を見るよう 1 か所にまとめます。"""

from __future__ import annotations

from types import MappingProxyType

MOVO_VERSION = "1.5.0"

#: 読める JSON 方言の版。
MOVO_JSON_VERSION = "1.0"

#: キャッシュ鍵に混ぜる実装の版。ここを上げると該当部分のキャッシュが失効します。
#: **辞書を読み取り専用にしている**のは、キャッシュ鍵の素をうっかり書き換えると
#: «前に作った動画と違う絵が出る» という一番気付きにくい壊れ方をするためです。
COMPONENT_VERSIONS = MappingProxyType(
    {
        "renderer": "1.4.0",
        "deformer": "1.2.0",
        "physics": "1.1.0",
        "audio": "1.1.0",
        "ai": "1.0.0",
        "skill": "1.1.0",
    }
)


def is_compatible_json_version(version: str | None) -> bool:
    """``"1.0"`` も ``"1.0.3"`` も通し、メジャーが違えば False。"""
    if not version:
        return True
    major = str(version).split(".")[0]
    return major == MOVO_JSON_VERSION.split(".")[0]
