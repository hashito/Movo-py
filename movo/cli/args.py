"""小さな引数パーサ。`--key value` / `--key=value` / `-abc` / `--` に対応します。

**`argparse` を使っていません。** 理由は 3 つあります。

1. JS 版と «書き方» を 1 文字も変えないため。`--no-audio` や `--jobs auto` の
   ような癖のある指定を argparse で再現すると、結局この量のコードになります。
2. `movo render --all-variants movo.json` のように **値を取らない指定の直後に
   位置引数が来る**書き方を、JS 版と同じ挙動で通すため。
3. サブコマンドごとに別の表を持たずに済むため（JS 版と同じ 1 枚の表です）。
"""

from __future__ import annotations

import re
from typing import Any

_CAMEL = re.compile(r"-+([a-z0-9])")


def _camel(key: str) -> str:
    """`--save-recipe` を `saveRecipe` にする。

    ハイフンのままだと呼ぶ側が `options["save-recipe"]` と書くことになり、
    JS 版では実際に `--super-sample` と `--dry-run` を **そう書き忘れていて
    効いていませんでした**。ここで一律に直します。
    """
    return _CAMEL.sub(lambda m: m.group(1).upper(), key)


def parse_args(argv: list[str], schema: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    """@returns (options, positional)"""
    schema = schema or {}
    booleans = set(schema.get("booleans", []))
    numbers = set(schema.get("numbers", []))
    multiples = set(schema.get("multiples", []))
    aliases: dict[str, str] = schema.get("aliases", {})

    options: dict[str, Any] = {}
    positional: list[str] = []
    passthrough = False

    def normalise(key: str) -> str:
        return aliases.get(key, _camel(key))

    def assign(key: str, value: Any) -> None:
        # multiples のキーは上書きせず積み上げる（--set a=1 --set b=2）
        if key not in multiples:
            options[key] = value
            return
        previous = options.get(key)
        if previous is None:
            options[key] = [value]
        elif isinstance(previous, list):
            options[key] = [*previous, value]
        else:
            options[key] = [previous, value]

    def coerce(key: str, value: str) -> Any:
        if key in numbers:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return value
            return int(number) if number.is_integer() else number
        return value

    index = 0
    while index < len(argv):
        argument = argv[index]
        if passthrough:
            positional.append(argument)
            index += 1
            continue
        if argument == "--":
            passthrough = True
            index += 1
            continue

        if argument.startswith("--"):
            body = argument[2:]
            if "=" in body:
                name, _, value = body.partition("=")
                key = normalise(name)
                assign(key, coerce(key, value))
                index += 1
                continue
            negated = body.startswith("no-")
            key = normalise(body[3:] if negated else body)
            if negated:
                options[key] = False
                index += 1
                continue
            if key in booleans:
                options[key] = True
                index += 1
                continue
            nxt = argv[index + 1] if index + 1 < len(argv) else None
            if nxt is None or nxt.startswith("-"):
                options[key] = True
                index += 1
            else:
                assign(key, coerce(key, nxt))
                index += 2
            continue

        if argument.startswith("-") and len(argument) > 1:
            flags = argument[1:]
            consumed_next = False
            for position, flag in enumerate(flags):
                key = normalise(flag)
                is_last = position == len(flags) - 1
                if key in booleans or not is_last:
                    options[key] = True
                    continue
                nxt = argv[index + 1] if index + 1 < len(argv) else None
                if nxt is None or nxt.startswith("-"):
                    options[key] = True
                else:
                    assign(key, coerce(key, nxt))
                    consumed_next = True
            index += 2 if consumed_next else 1
            continue

        positional.append(argument)
        index += 1

    return options, positional


COMMON_SCHEMA: dict[str, Any] = {
    "booleans": [
        "help",
        "version",
        "quiet",
        "verbose",
        "debug",
        "json",
        "force",
        "noCache",
        "dryRun",
        "open",
        "watch",
        "lock",
        "strict",
        "generate",
        "yes",
        "animations",
        "animation",
        "skills",
        "skill",
        "scenes",
        "scene",
        "movies",
        "movie",
        # 一括レンダリングの «中断からの再開»（movo batch --continue）
        "continue",
        # アスペクト比バリアントを全部出す（movo render --all-variants movo.json）。
        # ここに書かないと «次の語» を値として飲み込み、ファイル名が消えます。
        "allVariants",
        # 並列レンダリング（movo render --jobs N）で、区間ごとの中間ファイルを消さない。
        # 繋ぎ目を目で確かめたいときや、失敗した区間を調べたいときに使います。
        "keepParts",
    ],
    # 繰り返し指定できるキー（movo skill render --set a=1 --set b=2）
    # `anchor` は歌詞と曲を合わせるときの «この行はここ» の留め（movo lyrics align）。
    # 何点でも打てないと意味が無いので multiples です。
    # `asset` は make-mv へ画像などを渡す口（--asset art=path）。
    "multiples": ["set", "anchor", "asset"],
    # `jobs` は数値ですが `--jobs auto` とも書けます。数値に直せない値はそのまま
    # 文字列で入るので、受け取る側（resolve_job_count）で auto を解釈しています。
    # `warmup` は並列レンダリングの親が子に渡す助走フレーム数です。
    "numbers": [
        "from",
        "to",
        "time",
        "fps",
        "port",
        "seed",
        "superSample",
        "threads",
        "width",
        "height",
        "frame",
        "duration",
        "bpm",
        "jobs",
        "warmup",
        "intensity",
        "maxBars",
        "beatsPerBar",
        "minBpm",
        "maxBpm",
        "tolerance",
        # 歌う範囲を秒で直接指定する（movo lyrics align --start 5.4 --end 146）
        "start",
        "end",
    ],
    "aliases": {
        "h": "help",
        "v": "version",
        "o": "output",
        "q": "quality",
        "s": "scene",
        "t": "time",
        "f": "format",
        "p": "port",
        "V": "verbose",
    },
}


def param_options_from(options: dict[str, Any] | None = None) -> dict[str, Any]:
    """params 関係の指定だけを取り出す。

    `create_session` は «プロジェクトを読み込む唯一の入口» なので、素材の
    差し替え（`--set` / `--params`）と «作り方の保存»（`--save-recipe`）も
    そこで畳みます。どのコマンドからでも同じ書き方になるよう、拾い方を
    ここに 1 か所だけ置きます。
    """
    options = options or {}
    return {
        "set": options.get("set"),
        "params": options.get("params"),
        "save_recipe": options.get("saveRecipe"),
        "output": options.get("output"),
        "format": options.get("format"),
    }
