"""差し替え可能な入力値（params）と、制作指示（レシピ）。

「素材だけ差し替えて、同じ動画をもう一度作る」ための仕組みです。

同じ構成で絵だけ違う動画を作ろうとすると、プロジェクト JSON を丸ごと複製して
`assets` の 1 行を書き換えることになります。10 本作れば «ほぼ同じ JSON» が
10 個残り、あとから共通部分を直すのが破綻します。そこで、差し替えたいところ
だけに名前を付けて、外から与えられるようにします。

    {
      "params": {
        "art":   { "type": "asset",  "default": "../assets/singer.png" },
        "title": { "type": "text",   "default": "入れ子の街" },
        "bpm":   { "type": "number", "default": 205, "min": 40, "max": 300 }
      },
      "project": { "bpm": "${bpm}" },
      "assets":  { "singer": "${art}" }
    }

    movo render mv.json --set art=other.png -o tmp/b.mp4

`${...}` はスキルと «同じ» 式エンジンで評価します。式エンジンはサンドボックス
なので、ファイルにもネットワークにも触れません。params の値は CLI と
レシピからしか入りません。

`params` を «書いていない» JSON には一切触りません。既存のプロジェクトが
従来どおり動くことのほうが、この機能より大事だからです。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from movo.expression._compat import (
    MOVO_JSON_VERSION,
    MOVO_VERSION,
    ErrorCodes,
    MovoError,
    is_finite_number,
    js_number,
)

from ._template import expand_template, parse_input_assignments, resolve_inputs
from .extends import apply_extends
from .variants import apply_variant

#: レシピ JSON の $schema。«これはレシピだ» と一目で分かるようにする。
RECIPE_SCHEMA = "https://movo.dev/schema/recipe-v1.json"

parse_param_assignments = parse_input_assignments


def _is_plain_object(value) -> bool:
    return isinstance(value, dict)


def _to_finite_number(value):
    """数として読めれば返し、読めなければ None。

    書かれていないキー（None）は None のままにします。JS では `undefined` が
    来るところなので、`Number(null) === 0` の側に寄せてはいけません。
    """
    if value is None:
        return None
    n = js_number(value)
    return n if is_finite_number(n) else None


def list_params(project):
    """宣言されている params を一覧にする（`movo params <file>` の中身）。"""
    declarations = project.get("params") if _is_plain_object(project) else None
    declarations = declarations if _is_plain_object(declarations) else {}
    out = []
    for key, definition in declarations.items():
        definition = definition or {}
        entry = {
            "key": key,
            "type": definition.get("type") or "text",
            "label": definition.get("label") or "",
            "default": definition.get("default"),
            "required": bool(definition.get("required")),
        }
        if definition.get("min") is not None:
            entry["min"] = definition["min"]
        if definition.get("max") is not None:
            entry["max"] = definition["max"]
        if definition.get("options"):
            entry["options"] = definition["options"]
        out.append(entry)
    return out


def expand_params(project, file=None, set_values=None, params=None):
    """`params` を解決して `${...}` を展開する。

    返す `values` は «解決後の全項目»（既定値も含む）です。レシピにはこれを
    そのまま書きます。あとから既定値が変わっても、レシピを再実行すれば当時と
    同じ絵が出るようにするためです。
    """
    declarations = project.get("params") if _is_plain_object(project) else None
    declarations = declarations if _is_plain_object(declarations) else None
    given = param_overrides_from(set_values=set_values, params=params)
    # params を書いていない JSON は 1 文字も変えずに返す（従来どおり動くこと優先）
    if not declarations:
        if given:
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                "このプロジェクトには params が宣言されていないので値を差し替えられません",
                path="params",
                file=file,
                hint='差し替えたい箇所を "params" に宣言して、値を "${名前}" で参照してください',
            )
        return {"project": project, "values": {}, "declarations": {}}

    resolved = resolve_inputs(declarations, given, name="params")
    values = resolved["values"]
    if resolved["issues"]:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            " / ".join(issue["message"] for issue in resolved["issues"]),
            path="params",
            file=file,
            hint="差し替えられる項目は movo params <file> で見られます",
        )
    # 宣言に無いキーを渡したときは «打ち間違い» のことが多いので気付けるようにする
    for key in given:
        if key in declarations:
            continue
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f'"{key}" はこのプロジェクトの params にありません',
            path="params",
            file=file,
            hint=f"宣言されているのは: {', '.join(declarations.keys()) or '（なし）'}",
        )

    body = dict(project)
    # 宣言そのものは残さない。ここから先（検証・レンダラ）は params を知らなくてよい。
    body.pop("params", None)
    from movo.expression import ExpressionEngine

    seed = _to_finite_number((project.get("project") or {}).get("seed"))
    engine = ExpressionEngine(seed=int(seed) if seed is not None else 0)
    expanded = expand_template(
        body, {**_ambient_scope(project), **values}, engine=engine, path="", file=file
    )
    return {"project": expanded, "values": values, "declarations": declarations}


def resolve_params(project, file=None, set_values=None, params=None):
    """`expand_params` の薄い包み。展開後のプロジェクトだけが要るとき用。"""
    return expand_params(project, file=file, set_values=set_values, params=params)["project"]


def _ambient_scope(project):
    """式から見える «params 以外» の値。

    解像度や BPM は params から計算したくなることが多い（`${height * 0.4}`）ので
    添えます。同じ名前の params があればそちらが勝ちます。
    """
    video = project.get("video") or {}
    settings = project.get("project") or {}
    width = _to_finite_number(video.get("width"))
    height = _to_finite_number(video.get("height"))
    width = 1920 if width is None else width
    height = 1080 if height is None else height
    fps = _to_finite_number(video.get("fps"))
    bpm = _to_finite_number(settings.get("bpm"))
    seed = _to_finite_number(settings.get("seed"))
    return {
        "width": width,
        "height": height,
        "fps": 30 if fps is None else fps,
        "bpm": 120 if bpm is None else bpm,
        "seed": 0 if seed is None else seed,
        "centerX": width / 2,
        "centerY": height / 2,
    }


def param_overrides_from(set_values=None, params=None):
    """与えられた値を集める。

    重ねる順は「`--params <file>` → `--set key=値`」で、手で打ったほうが強い。
    """
    out: dict = {}
    if isinstance(params, str):
        out.update(_read_params_file(params))
    elif _is_plain_object(params):
        out.update(params)
    out.update(parse_input_assignments(set_values))
    return out


def _read_params_file(file):
    """`--params <file>`。素の `{key: 値}` でも、レシピそのものでも受ける。"""
    absolute = Path(file).resolve()
    if not absolute.exists():
        raise MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND, f"--params のファイルが見つかりません: {file}"
        )
    try:
        parsed = json.loads(absolute.read_text(encoding="utf-8"))
    except ValueError as err:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f"--params の JSON を読めません: {err}",
            file=str(absolute),
        ) from err
    if _is_plain_object(parsed) and _is_plain_object(parsed.get("params")):
        return parsed["params"]
    if not _is_plain_object(parsed):
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            "--params のファイルはオブジェクトにしてください",
            file=str(absolute),
        )
    return parsed


def prepare_project(
    raw,
    file=None,
    set_values=None,
    params=None,
    save_recipe=None,
    output=None,
    output_format=None,
    variant=None,
    on_variant_warning=None,
):
    """読み込んだ JSON を «素の Movo JSON» にする。

    継承 → バリアント → params 展開 → レシピ保存 の順です。
    バリアントを継承のあと・params の前に置く理由は variants.py に書いています。
    """
    inherited = apply_variant(
        apply_extends(raw, file=file), variant, file=file, on_warn=on_variant_warning
    )
    expanded = expand_params(inherited, file=file, set_values=set_values, params=params)
    project = expanded["project"]
    if isinstance(save_recipe, str) and save_recipe != "":
        write_recipe(
            save_recipe,
            build_recipe(
                template=file,
                recipe_file=save_recipe,
                params=expanded["values"],
                declarations=expanded["declarations"],
                output=output if output is not None else (project.get("output") or {}).get("path"),
                output_format=(
                    output_format
                    if output_format is not None
                    else (project.get("output") or {}).get("format")
                ),
            ),
        )
    return project


def build_recipe(template, recipe_file, params=None, declarations=None, output=None, output_format=None):
    """レシピ（制作指示）を組み立てる。

    `params` には «解決後の全項目» を書きます。素材のハッシュも残すので、
    再実行したときに «素材が差し替わっている» ことに気付けます。
    """
    recipe_dir = Path(recipe_file).resolve().parent
    template_path = Path(template or "movo.json").resolve()
    params = params or {}
    assets = _hash_param_assets(params, declarations or {}, template_path.parent)
    recipe = {
        "$schema": RECIPE_SCHEMA,
        "movoVersion": MOVO_JSON_VERSION,
        "recipe": {
            "template": _to_posix(_relative(template_path, recipe_dir)),
            "createdWith": f"movo {MOVO_VERSION}",
        },
        "params": params,
    }
    if assets:
        recipe["assets"] = assets
    out: dict = {}
    if output:
        out["path"] = _to_posix(_relative(Path(output).resolve(), recipe_dir))
    # JSON のキーは JS 版のまま `format`。引数名だけ Python の組み込みを避けている。
    out["format"] = output_format or "mp4"
    recipe["output"] = out
    return recipe


def _hash_param_assets(params, declarations, project_dir: Path):
    """`type: "asset"` の params だけ中身のハッシュを控える。"""
    out: dict = {}
    for key, value in params.items():
        definition = declarations.get(key) or {}
        if definition.get("type") != "asset" or not isinstance(value, str) or value == "":
            continue
        file = (project_dir / value).resolve()
        if not file.is_file():
            continue
        out[key] = {"path": _to_posix(value), "hash": hash_file(file)}
    return out


def check_recipe_assets(recipe, template_file):
    """レシピを «当時と同じ素材か» の観点で見直す。"""
    project_dir = Path(template_file).resolve().parent
    issues = []
    for key, entry in ((recipe or {}).get("assets") or {}).items():
        current = ((recipe or {}).get("params") or {}).get(key) or entry.get("path")
        file = (project_dir / current).resolve()
        if not file.exists():
            issues.append({"key": key, "path": current, "reason": "素材が見つかりません"})
            continue
        if entry.get("hash") and hash_file(file) != entry["hash"]:
            issues.append(
                {"key": key, "path": current, "reason": "レシピを作ったときと中身が違います"}
            )
    return issues


def write_recipe(file, recipe):
    """レシピを書き出す。"""
    target = Path(file).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return str(target)


def read_recipe(file):
    """レシピを読む。`template` はレシピからの相対パスなので絶対パスに直す。"""
    absolute = Path(file).resolve()
    if not absolute.exists():
        raise MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND,
            f"レシピが見つかりません: {file}",
            hint="movo render <project> --save-recipe <file> で作れます",
        )
    try:
        recipe = json.loads(absolute.read_text(encoding="utf-8"))
    except ValueError as err:
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            f"レシピを JSON として読めません: {err}",
            file=str(absolute),
        ) from err
    template = ((recipe or {}).get("recipe") or {}).get("template")
    if not isinstance(template, str) or template == "":
        raise MovoError(
            ErrorCodes.MOVO_SCHEMA_INVALID,
            "レシピに recipe.template がありません",
            file=str(absolute),
            hint='{ "recipe": { "template": "../mv.json" }, "params": { ... } } の形です',
        )
    recipe_dir = absolute.parent
    template_path = (recipe_dir / template).resolve()
    if not template_path.exists():
        raise MovoError(
            ErrorCodes.MOVO_ASSET_NOT_FOUND,
            f"レシピの元になるプロジェクトが見つかりません: {template}",
            file=str(absolute),
            hint="template はレシピからの相対パスです",
        )
    output = dict(recipe.get("output") or {})
    if isinstance(output.get("path"), str):
        output["path"] = str((recipe_dir / output["path"]).resolve())
    return {
        "file": str(absolute),
        "recipe": recipe,
        "template": str(template_path),
        "params": recipe["params"] if _is_plain_object(recipe.get("params")) else {},
        "output": output,
    }


def hash_file(path) -> str:
    """ファイルの中身の SHA-256（レシピが素材の差し替えに気付くため）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(target: Path, base: Path) -> str:
    """`base` から見た `target` の相対パス（別ドライブなら絶対パス）。"""
    import os

    try:
        return os.path.relpath(target, base)
    except ValueError:  # Windows で別ドライブのとき
        return str(target)


def _to_posix(value) -> str:
    """Windows でも «レシピの中身» は / で書く（別の OS へ持って行けるように）。"""
    return str(value).replace("\\", "/")


__all__ = [
    "RECIPE_SCHEMA",
    "build_recipe",
    "check_recipe_assets",
    "expand_params",
    "hash_file",
    "list_params",
    "param_overrides_from",
    "parse_param_assignments",
    "prepare_project",
    "read_recipe",
    "resolve_params",
    "write_recipe",
]
