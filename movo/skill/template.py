"""スキル／基礎アニメーションのテンプレート展開。

JSON の中に `${...}` を書くと、入力値を参照した式として評価されます。式は
`movo.expression` のサンドボックスなので、ファイルやネットワークには触れません。

    "text": "${title}"                 → 入力値そのまま（型も保たれる）
    "y": "${height * 0.4}"             → 数値計算
    "text": "BPM ${bpm} で同期"          → 文字列の中に埋め込む
    { "when": "${showLines}", ... }     → 偽なら要素ごと消える
    { "repeat": { "count": "${n}", "as": "i" }, ... } → n 個に展開（i が使える）
    { "repeat": { "over": "${lines}", "as": "line", "indexAs": "i" }, ... }
                                        → 配列の要素ごとに展開（line と i が使える）

`when` と `repeat` は配列の要素・オブジェクトのどちらにも書けます。

**式エンジン（`movo.expression`）は別の担当が移植中です。** 揃うまでは
`${...}` を含むテンプレートの展開だけが «後で繋ぐ» で止まります。式を含まない
テンプレート（定数だけのスキル）と、`when` / `repeat` の構造そのものは
このファイルだけで動きます。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from movo.cli.bridge import NotConnectedError, optional_module
from movo.cli.errors import ErrorCodes, MovoError


def is_timed_line(value: Any) -> bool:
    """時刻つきの歌詞の行か（`{ text, at }` の形か）。"""
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("text"), str):
        return False
    try:
        at = float(value.get("at"))
    except (TypeError, ValueError):
        return False
    return at == at  # NaN 除け


def _lyric_text(line: Any) -> str:
    if isinstance(line, dict):
        return line.get("text", "") or ""
    return "" if line is None else str(line)


def _lyric_at(line: Any, fallback: float = 0) -> float:
    if isinstance(line, dict):
        try:
            return float(line["at"])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return 0.0


def _lyric_for(line: Any, fallback: float = 1) -> float:
    if isinstance(line, dict):
        try:
            return float(line["for"])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return 0.0


def _lyric_timed(line: Any) -> bool:
    if not isinstance(line, dict):
        return False
    try:
        float(line["at"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


# 歌詞の «行» を、文字列でもオブジェクトでも同じように扱うための関数。
#
# `lines` には 2 通りが来ます。
#   ["眠らない街の音が", "少しずつ遠ざかって"]              ← 時刻なし。等分する
#   [{"text": "...", "at": 12.4, "for": 2.7}, ...]        ← 時刻つき（.lrc など）
#
# スキルの中で書き分けると、書き忘れたときに **片方の形でだけ壊れる** ので
# 関数にしました。秒は «そのシーンの中の秒» です（曲頭からの時刻ではありません）。
LYRIC_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "lyricText": _lyric_text,
    "lyricAt": _lyric_at,
    "lyricFor": _lyric_for,
    "lyricTimed": _lyric_timed,
}

# 文字列全体が 1 個の `${...}` のときだけ「型を保つ」扱いにする。
# `[^{}]` にしているのは "${a}-${b}" を 1 個と誤認しないため。
FULL_EXPRESSION = re.compile(r"^\$\{([^{}]+)\}$")
INLINE_EXPRESSION = re.compile(r"\$\{([^{}]+)\}")


def create_skill_engine(seed: int = 0):
    """スキル展開で使う式エンジンを作る。

    **必ずこれを通してください。** 直接エンジンを作ると歌詞用の関数が入らず、
    «ある経路では動くのに別の経路では lyricText が無い» という分かりにくい
    壊れ方をします（JS 版では 5 か所で個別に作っていて、1 か所直しただけでは
    動きませんでした）。
    """
    module = optional_module("movo.expression")
    if module is None:
        return None
    engine_class = getattr(module, "ExpressionEngine", None)
    if engine_class is None:
        return None
    try:
        return engine_class(seed=seed, extra_functions=dict(LYRIC_FUNCTIONS))
    except TypeError:
        # 相手の引数名がまだ決まっていない場合の逃げ道。名前が固まったら
        # この分岐は消せます。
        engine = engine_class(seed)
        setter = getattr(engine, "add_functions", None) or getattr(engine, "register", None)
        if setter is not None:
            setter(dict(LYRIC_FUNCTIONS))
        return engine


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def expand_template(node: Any, scope: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
    options = options or {}
    engine = options.get("engine") or create_skill_engine(options.get("seed", 0))
    return _walk(node, scope, engine, options.get("path", ""), options.get("file"))


def _walk(node, scope, engine, path, file):
    if isinstance(node, str):
        return _expand_string(node, scope, engine, path, file)
    if isinstance(node, list):
        out = []
        for index, item in enumerate(node):
            out.extend(_expand_item(item, scope, engine, f"{path}[{index}]", file))
        return out
    if isinstance(node, dict):
        results = _expand_item(node, scope, engine, path, file)
        # オブジェクト単体に repeat が付いていた場合は配列になる
        if len(results) == 1:
            return results[0]
        return results
    return node


def _expand_item(node, scope, engine, path, file) -> list:
    """1 要素を展開する。`when` で 0 個、`repeat` で n 個になりうるので配列を返す。"""
    if not isinstance(node, dict):
        return [_walk(node, scope, engine, path, file)]

    if "when" in node:
        if not _truthy(_evaluate(node["when"], scope, engine, f"{path}.when", file)):
            return []

    if node.get("repeat"):
        repeat = node["repeat"]

        # `over` に配列を渡すと «要素ごとに 1 つ» に展開します。
        #
        # これが無いと、歌詞 MV でいちばん書きたい «行ごとに 1 レイヤー» を
        #   { "repeat": { "count": "${lines.length}", "as": "i" }, "text": "${lines[i]}" }
        # と書くことになり、**`lines.length` と `lines[i]` の対応を書き手が守る**
        # 必要が出ます。行を足したときに壊れやすい形でした。
        has_over = "over" in repeat
        items = _evaluate(repeat["over"], scope, engine, f"{path}.repeat.over", file) if has_over else None
        if has_over and not isinstance(items, list):
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                f"repeat.over には配列を渡してください（{type(items).__name__}）",
                path=path,
                file=file,
            )

        if has_over:
            count = len(items)
        else:
            count = max(
                0,
                round(_to_number(_evaluate(repeat.get("count"), scope, engine, f"{path}.repeat.count", file))),
            )
        if count > 500:
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                f"repeat count {count} is too large (max 500)",
                path=path,
                file=file,
            )

        # `over` のときは «要素» の名前、`count` のときは «数» の名前。既定を分けて
        # あるのは、`over` で `index` という名前に要素が入ると紛らわしいためです。
        name = repeat.get("as") or ("item" if has_over else "index")
        index_name = repeat.get("indexAs") or f"{name}Index"
        start = _to_number(_evaluate(repeat.get("from", 0), scope, engine, f"{path}.repeat.from", file))
        step = _to_number(_evaluate(repeat.get("step", 1), scope, engine, f"{path}.repeat.step", file)) or 1

        out = []
        for i in range(count):
            value = items[i] if has_over else start + i * step
            child_scope = {
                **scope,
                name: value,
                index_name: i,
                f"{name}Index": i,
                f"{name}Count": count,
            }
            body = {k: v for k, v in node.items() if k not in ("repeat", "when")}
            out.extend(_expand_item(body, child_scope, engine, f"{path}[{i}]", file))
        return out

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("when", "repeat"):
            continue
        # 値のほうに when が付いていたら、偽ならキーごと消す。
        # assets のような「配列ではない入れ物」を条件付きにするための書き方。
        if isinstance(value, dict) and "when" in value:
            if not _truthy(_evaluate(value["when"], scope, engine, f"{path}.{key}.when", file)):
                continue
        # キー自体にも式を書ける（"${name}Layer" のような使い方）
        expanded_key = (
            str(_expand_string(key, scope, engine, path, file)) if "${" in key else key
        )
        out[expanded_key] = _walk(value, scope, engine, f"{path}.{key}" if path else key, file)
    return [out]


def _expand_string(text: str, scope, engine, path, file):
    """文字列中の `${...}` を解決する。全体が式なら型を保ったまま返す。"""
    full = FULL_EXPRESSION.match(text)
    if full:
        return _evaluate_source(full.group(1), scope, engine, path, file)
    if "${" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        value = _evaluate_source(match.group(1), scope, engine, path, file)
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return INLINE_EXPRESSION.sub(replace, text)


def _evaluate(value, scope, engine, path, file):
    if isinstance(value, str):
        return _expand_string(value, scope, engine, path, file)
    return value


def _evaluate_source(source: str, scope, engine, path, file):
    if engine is None:
        # **ここが «後で繋ぐ» の境目です。** 式エンジンは別担当が移植中なので、
        # 揃うまでは式を含むスキルだけが展開できません。名指しで止めます。
        raise NotConnectedError("movo.expression", "式エンジン")
    try:
        return engine.evaluate(source, scope, path=path, file=file)
    except NotConnectedError:
        raise
    except TypeError:
        # 相手の引数名がまだ決まっていない場合の逃げ道
        return engine.evaluate(source, scope)
    except MovoError:
        raise
    except Exception as error:  # noqa: BLE001
        raise MovoError(
            ErrorCodes.MOVO_EXPRESSION_INVALID,
            f'テンプレートの式 "${{{source.strip()}}}" を評価できませんでした: {error}',
            path=path,
            file=file,
        ) from error


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0 and value == value
    if isinstance(value, str):
        return value not in ("", "false", "0")
    return bool(value)


def resolve_inputs(
    definitions: dict[str, Any] | None,
    given: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """入力定義に沿って値を検証・既定値補完する。

    unit が "px" の数値は `scale` 倍されます。ライブラリの寸法は 1080p を基準に
    書いてあるので、これで解像度を変えても同じ絵になります。範囲チェックは
    **スケール前**の値に対して行うので、入力の目安（min/max）はそのまま使えます。

    @returns (values, issues)
    """
    definitions = definitions or {}
    given = given or {}
    options = options or {}
    values: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    label = f"{options['name']}: " if options.get("name") else ""
    raw_scale = options.get("scale", 1)
    scale = raw_scale if isinstance(raw_scale, (int, float)) and raw_scale > 0 else 1

    for key, definition in definitions.items():
        kind = definition.get("type", "text")
        value = given.get(key)

        if value is None or value == "":
            if definition.get("required") and "default" not in definition:
                issues.append(
                    {
                        "path": f"inputs.{key}",
                        "message": f'{label}入力 "{key}" は必須です（{definition.get("label", kind)}）',
                    }
                )
                continue
            value = definition.get("default")
        if value is None:
            # 既定も指定も無いとき。textList だけ None ではなく空配列にするのは、
            # 歌詞を «行の数だけ繰り返す» テンプレートが None で落ちるためです。
            if kind == "textList":
                values[key] = []
            elif kind == "boolean":
                values[key] = False
            elif kind == "number":
                values[key] = 0
            else:
                values[key] = None
            continue

        if kind == "number":
            try:
                number = float(value) if not isinstance(value, bool) else float(int(value))
            except (TypeError, ValueError):
                issues.append(
                    {
                        "path": f"inputs.{key}",
                        "message": f'{label}"{key}" には数値が必要です（受け取った値: {value}）',
                    }
                )
                continue
            if definition.get("min") is not None and number < definition["min"]:
                issues.append(
                    {"path": f"inputs.{key}", "message": f'{label}"{key}" は {definition["min"]} 以上にしてください'}
                )
                continue
            if definition.get("max") is not None and number > definition["max"]:
                issues.append(
                    {"path": f"inputs.{key}", "message": f'{label}"{key}" は {definition["max"]} 以下にしてください'}
                )
                continue
            scaled = number * scale if definition.get("unit") == "px" else number
            values[key] = int(scaled) if float(scaled).is_integer() else scaled
        elif kind == "boolean":
            if isinstance(value, bool):
                values[key] = value
            else:
                values[key] = str(value).lower() in ("true", "1", "yes", "on")
        elif kind == "choice":
            allowed = definition.get("options") or []
            if allowed and value not in allowed:
                issues.append(
                    {
                        "path": f"inputs.{key}",
                        "message": f'{label}"{key}" は次のいずれかにしてください: {", ".join(map(str, allowed))}',
                    }
                )
                continue
            values[key] = value
        elif kind == "list":
            if isinstance(value, list):
                values[key] = value
            else:
                separator = definition.get("separator", ",")
                values[key] = [s.strip() for s in str(value).split(separator) if s.strip()]
        elif kind == "textList":
            # 歌詞を渡すための型。カンマは歌詞の «中身» に出るので、区切りは行に
            # します。--set lines='["ア","イ"]' のような JSON もそのまま通ります。
            # 時刻つきの行（`.lrc` などから来る `{text, at, for}`）は **そのまま
            # 通します**。str() に掛けると "[object Object]" 相当になり、しかも
            # 歌詞は表示されるので «時刻だけ無視された» ことに気付きにくい壊れ方
            # をします。
            if isinstance(value, list):
                values[key] = [item if is_timed_line(item) else str(item) for item in value]
            else:
                separator = definition.get("separator")
                if separator:
                    parts = str(value).split(separator)
                else:
                    parts = re.split(r"\r?\n|\\n", str(value))
                values[key] = [s.strip() for s in parts if s.strip()]
        else:
            # color / asset / text とそれ以外
            values[key] = value if isinstance(value, str) else str(value)

    # 定義に無い入力は警告ではなくそのまま渡す（前方互換のため）
    for key, value in given.items():
        if key not in values and key not in definitions:
            values[key] = value

    return values, issues
