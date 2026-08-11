"""`movo make <recipe>` — 保存した «作り方» からもう一度作る。
`movo params <project>` — 差し替えられる項目の一覧。

  movo render mv.json --set art=other.png --save-recipe tmp/b.recipe.json
  movo make tmp/b.recipe.json                      まったく同じものをもう一度
  movo make tmp/b.recipe.json --set art=third.png  絵だけ変えてもう一度

レシピには «解決後の全項目» が入っているので、あとからプロジェクト側の既定値が
変わっても、レシピを再実行すれば当時と同じ絵が出ます。素材のハッシュも控えて
あるので、同じ名前のまま中身が差し替わっていれば気付けます。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import create_session, read_project_file, render_video


def plan_recipe(file: str | None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """レシピ 1 件から «描き出す材料» をそろえる。

    実際に描く前に確かめられるよう、レンダリングとは切り離してあります。
    """
    options = options or {}
    if not file:
        raise MovoError(ErrorCodes.MOVO_CLI_USAGE, "レシピのファイルが必要です", hint="movo make tmp/b.recipe.json")
    read_recipe = bridge.pick("movo.schema.params", "read_recipe", "readRecipe")
    param_overrides_from = bridge.pick("movo.schema.params", "param_overrides_from", "paramOverridesFrom")
    check_recipe_assets = bridge.pick("movo.schema.params", "check_recipe_assets", "checkRecipeAssets")

    loaded = read_recipe(file)
    # レシピの値を土台に、その場の --set / --params をさらに重ねる（絵だけ差し替える）
    params = {**loaded["params"], **param_overrides_from(options)}
    output = str(Path(options["output"]).resolve()) if isinstance(options.get("output"), str) else loaded["output"].get("path")
    return {
        "recipe": loaded["recipe"],
        "file": loaded["file"],
        "template": loaded["template"],
        "params": params,
        "output": output,
        "format": options.get("format") or loaded["output"].get("format") or "mp4",
        "warnings": check_recipe_assets(loaded["recipe"], loaded["template"]),
    }


def make_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    plan = plan_recipe(positional[0] if positional else None, options)

    logger.step(f'レシピ {style.bold(Path(plan["file"]).name)} から作り直します → {plan["template"]}')
    for warning in plan["warnings"]:
        logger.warn(f'params.{warning["key"]}（{warning["path"]}）: {warning["reason"]}')

    session = create_session(
        plan["template"],
        {
            "quality": options.get("quality"),
            "renderer": options.get("renderer"),
            "super_sample": options.get("superSample"),
            "seed": options.get("seed"),
            "no_cache": options.get("noCache") is True or options.get("cache") is False,
            "generate_assets": options.get("generate") is not False,
            "strict_plugins": options.get("strict") is not False,
            "dry_run_ai": options.get("dryRun") is True,
            # params はレシピと --set を畳んだあとの «解決済みの値»
            "params": plan["params"],
            "save_recipe": options.get("saveRecipe"),
            "output": plan["output"],
            "format": plan["format"],
        },
    )

    result = render_video(
        session,
        {"output": plan["output"], "format": plan["format"], "quiet": options.get("quiet")},
    )
    logger.info(
        f'  {result["frames"]} フレーム / {result["elapsedSeconds"]:.1f} 秒 '
        f'({result["fpsAchieved"]:.1f} fps) / 形式 {result["format"]}'
    )
    if options.get("json"):
        say(json.dumps({**result, "params": plan["params"]}, ensure_ascii=False, indent=2, default=str))
    return result


def params_command(positional: list[str], options: dict[str, Any]) -> Any:
    """`movo params <project>` — 差し替えられる項目を読む。

    «この JSON のどこを差し替えられるのか» が分からないと --set の名前を
    当てずっぽうで打つことになるので、宣言をそのまま見せます。
    """
    file = (positional[0] if positional else None) or "movo.json"
    read = read_project_file(file)
    # 継承した先に params が書いてあることもあるので、先に畳んでから見る
    apply_extends = bridge.pick("movo.schema", "apply_extends", "applyExtends")
    list_params = bridge.pick("movo.schema.params", "list_params", "listParams")
    project = apply_extends(read["raw"], file=read["file"])
    declared = list_params(project)

    if options.get("json"):
        say(json.dumps({"file": read["file"], "params": declared}, ensure_ascii=False, indent=2, default=str))
        return declared

    short = os.path.relpath(read["file"], os.getcwd())
    logger.info("")
    logger.info(f"{style.bold(short)}  差し替えられる項目 {len(declared)} 件")
    if not declared:
        logger.info('  （なし）"params" を宣言すると素材や文字を外から差し替えられます')
        logger.info('  例: "params": { "art": { "type": "asset", "default": "assets/a.png" } }')
        return declared
    for entry in declared:
        bits = [entry.get("type")]
        if entry.get("required"):
            bits.append("必須")
        if entry.get("options"):
            bits.append("|".join(map(str, entry["options"])))
        if entry.get("min") is not None or entry.get("max") is not None:
            bits.append(f'{entry.get("min", "-∞")}..{entry.get("max", "∞")}')
        fallback = "" if "default" not in entry else " 既定=" + json.dumps(entry["default"], ensure_ascii=False)
        kinds = style.dim("(" + ", ".join(map(str, bits)) + ")")
        logger.info(f'  {str(entry["key"]).ljust(14)} {kinds}{fallback}')
        if entry.get("label"):
            logger.info(f'    {entry["label"]}')
    logger.info("")
    example = " ".join(f'--set {e["key"]}=値' for e in declared[:2])
    logger.info(f"  差し替えて書き出す: movo render {short} {example}")
    logger.info("  作り方を残す:       同じ行に --save-recipe tmp/b.recipe.json を足す")
    logger.info("")
    return declared
