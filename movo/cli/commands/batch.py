"""`movo batch` — 表形式の入力値から連番で書き出す。

  movo batch lyric-mv --input songs.json --out tmp/mv/{name}.mp4 --jobs 8
  movo batch "examples/*.json" --out tmp/mv/{basename}.mp4

連続制作の基本形は «1 つのテンプレート × N 通りの入力値» です。10 本の MV を
書き出すのに使い捨てのスクリプトを書くのは CLI の仕事なので、ここに置きました。

設計の決めごと:

  - 1 本のレンダリングは単一スレッドなので、«本数» を並べて速くする。
    プロセスを分けるのは、1 本が落ちても他を巻き込まないためでもある。
  - 途中で失敗しても残りは続ける。20 分回して 3 本目で止まっているのが
    いちばん困るため。失敗は最後にまとめて出し、終了コードで判別できる。
  - `--continue` は «既にある出力を飛ばす»。中断したところから再開できる。

**`movo render --jobs` との違い。** あちらは «1 本を区間に割る»、こちらは
«何本かを並べる» です。1 本しかないときは前者、10 本あるときは後者が効きます。
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .. import bridge
from ..console import logger, say, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import read_project_file

# 出力名にだけ使う列。params に無くても «打ち間違い» とは言わない。
PATTERN_ONLY_KEYS = {"name", "basename", "index", "out", "output"}


def parse_table(text: str, file: str = "") -> list[dict[str, Any]]:
    """表（JSON / CSV）を 1 行 = 1 本の入力値として読む。

    値の型寄せ（数値・真偽・JSON）は `--set` と «同じ» 規則にしてあります。
    CSV で 205 と書いたときと `--set bpm=205` が別物になると混乱するためです。
    """
    from movo.skill import coerce_assignment

    trimmed = text.lstrip("﻿").strip()
    looks_json = bool(re.search(r"\.jsonc?$", file, re.IGNORECASE)) or trimmed[:1] in ("[", "{")
    if looks_json:
        try:
            parsed = json.loads(trimmed)
        except ValueError as error:
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID, f"--input の JSON を読めません: {error}", file=file
            ) from error
        rows = parsed if isinstance(parsed, list) else parsed.get("rows") if isinstance(parsed, dict) else None
        if not isinstance(rows, list):
            raise MovoError(
                ErrorCodes.MOVO_SCHEMA_INVALID,
                "--input は「1 行 = 1 本」の配列にしてください",
                file=file,
                hint='例: [{ "name": "01", "title": "入れ子の街", "bpm": 205 }]',
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise MovoError(
                    ErrorCodes.MOVO_SCHEMA_INVALID, f"--input の {index + 1} 行目がオブジェクトではありません", file=file
                )
        return rows

    lines = [line for line in re.split(r"\r?\n", trimmed) if line.strip()]
    if len(lines) < 2:
        raise MovoError(ErrorCodes.MOVO_SCHEMA_INVALID, "CSV には見出し行と 1 行以上の中身が必要です", file=file)
    headers = [cell.strip() for cell in _split_csv_line(lines[0])]
    rows = []
    for line in lines[1:]:
        cells = _split_csv_line(line)
        row: dict[str, Any] = {}
        for index, header in enumerate(headers):
            raw = cells[index] if index < len(cells) else ""
            value = coerce_assignment(raw)
            # "01" を 1 にしてしまうと連番の名前が崩れる（01.mp4 が 1.mp4 になる）。
            # 数値に直して «書いたとおりに戻らない» ものは文字列のままにする。
            if isinstance(value, (int, float)) and not isinstance(value, bool) and str(value) != raw.strip():
                value = raw.strip()
            row[header] = value
        rows.append(row)
    return rows


def _split_csv_line(line: str) -> list[str]:
    """ダブルクォートで括られた値の中のカンマを守る、最小限の CSV 分解。"""
    cells: list[str] = []
    current = ""
    quoted = False
    i = 0
    while i < len(line):
        ch = line[i]
        if quoted:
            if ch == '"' and i + 1 < len(line) and line[i + 1] == '"':
                current += '"'
                i += 2
                continue
            if ch == '"':
                quoted = False
            else:
                current += ch
            i += 1
            continue
        if ch == '"':
            quoted = True
        elif ch == ",":
            cells.append(current)
            current = ""
        else:
            current += ch
        i += 1
    cells.append(current)
    return [cell.strip() for cell in cells]


def expand_targets(pattern: str, cwd: str | None = None) -> list[str]:
    """`examples/*.json` のような書き方をファイルの一覧にする。

    シェルが展開してくれない環境（Windows のコマンドプロンプト）でも同じように
    書けるように、ここでも展開します。`*` と `?` はファイル名の部分にだけ書けます。
    """
    cwd = cwd or os.getcwd()
    absolute = Path(cwd) / pattern
    directory = absolute.parent
    base = absolute.name
    if re.search(r"[*?]", str(directory)):
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "ワイルドカードはファイル名の部分にだけ書けます",
            hint='例: movo batch "examples/*.json" --out tmp/mv/{basename}.mp4',
        )
    if not re.search(r"[*?]", base):
        return [str(absolute.resolve())]
    if not directory.exists():
        raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"フォルダが見つかりません: {directory}")
    return sorted(
        str(entry.resolve()) for entry in directory.iterdir() if entry.is_file() and fnmatch.fnmatch(entry.name, base)
    )


def format_output(pattern: str, context: dict[str, Any]) -> str:
    """`tmp/mv/{name}.mp4` のような書き方を 1 本ぶんの出力先にする。

    差し込める名前は «その行の全列» と `index` / `basename` です。値がそのまま
    ファイル名になるので、区切り文字だけは潰します（表の値でフォルダを掘られると
    書き出し先が散らばるため）。
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if name not in context or context[name] is None:
            raise MovoError(
                ErrorCodes.MOVO_CLI_USAGE,
                f"--out の {{{name}}} に入れる値がありません",
                hint=f'使えるのは: {", ".join(context)}',
            )
        return re.sub(r'[\\/:*?"<>|]', "-", str(context[name])).strip()

    return re.sub(r"\{([^{}]+)\}", replace, pattern)


def build_batch_plan(spec: dict[str, Any]) -> dict[str, Any]:
    """何をどこへ書き出すかを «先に» 全部決める。

    20 分回してから «--out の書き方が違う» と言われないよう、走らせる前に
    すべての行を組み立てて重複や欠けを見ます。
    """
    cwd = spec.get("cwd") or os.getcwd()
    out = spec.get("out")
    if not isinstance(out, str) or out == "":
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE, "--out で書き出し先を指定してください", hint="例: --out tmp/mv/{name}.mp4"
        )

    jobs: list[dict[str, Any]] = []
    warnings: list[str] = []
    mode = "table" if spec.get("rows") is not None else "files"

    if mode == "table":
        template = _resolve_template(spec["target"], cwd)
        declared = spec.get("declared")
        for index, row in enumerate(spec["rows"]):
            context = {**row, "index": index + 1, "basename": Path(template).stem}
            params = {}
            for key, value in row.items():
                if declared is not None and key not in declared:
                    # 出力名にしか使わない列（name など）は黙って通す。
                    # それ以外は打ち間違いを疑う。
                    if key not in PATTERN_ONLY_KEYS:
                        warnings.append(f'{index + 1} 行目の "{key}" はテンプレートの params にありません（無視します）')
                    continue
                params[key] = value
            jobs.append(
                {
                    "index": index + 1,
                    "name": str(row.get("name") or context["basename"] or index + 1),
                    "template": template,
                    "params": params,
                    "output": str((Path(cwd) / format_output(out, context)).resolve()),
                }
            )
    else:
        files = expand_targets(spec["target"], cwd)
        if not files:
            raise MovoError(
                ErrorCodes.MOVO_ASSET_NOT_FOUND,
                f'対象のプロジェクトがありません: {spec["target"]}',
                hint="--input を付けると «1 つのテンプレート × N 通りの入力値» で回せます",
            )
        for index, file in enumerate(files):
            basename = Path(file).stem
            context = {"name": basename, "basename": basename, "index": index + 1}
            jobs.append(
                {
                    "index": index + 1,
                    "name": basename,
                    "template": file,
                    "params": {},
                    "output": str((Path(cwd) / format_output(out, context)).resolve()),
                }
            )

    # 同じ行き先に 2 本書くと «先に終わったほうが消える»。走る前に気付けるようにする。
    seen: dict[str, int] = {}
    for job in jobs:
        if job["output"] in seen:
            raise MovoError(
                ErrorCodes.MOVO_CLI_USAGE,
                f'{job["index"]} 本目と {seen[job["output"]]} 本目の書き出し先が同じです: {job["output"]}',
                hint="--out に {name} や {index} を入れて 1 本ずつ違う名前にしてください",
            )
        seen[job["output"]] = job["index"]

    # --continue は «既にある出力を飛ばす»。中断からの再開に使う。
    skipped = []
    remaining = []
    for job in jobs:
        path = Path(job["output"])
        done = bool(spec.get("continue_existing")) and path.exists() and path.stat().st_size > 0
        (skipped if done else remaining).append(job)
    return {"mode": mode, "jobs": remaining, "skipped": skipped, "warnings": warnings}


def _resolve_template(target: str | None, cwd: str) -> str:
    """テンプレートは拡張子を省いても書ける（`movo batch lyric-mv`）。"""
    if not target:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE,
            "テンプレートのプロジェクトを指定してください",
            hint="movo batch lyric-mv.json --input songs.json --out tmp/mv/{name}.mp4",
        )
    for candidate in (target, f"{target}.json"):
        absolute = Path(cwd) / candidate
        if absolute.is_file():
            return str(absolute.resolve())
    raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"テンプレートが見つかりません: {target}")


def default_job_count() -> int:
    """既定の並列数。OS と ffmpeg のぶんに 2 コア空けておく。"""
    return max(1, (os.cpu_count() or 4) - 2)


def run_jobs(jobs: list[dict], limit: int, run: Callable, on_done: Callable | None = None) -> list[dict]:
    """決めた本数を並べて流す。

    1 本が失敗しても残りは続けます。`run` を差し替えられるようにしてあるのは、
    テストから «実際に描かずに» 順序と再開の挙動を確かめるためです。

    ここが **スレッド** プールなのは、`run` が «子プロセスを起こして待つ» だけで、
    Python 側では何も計算しないからです（GIL は待ち時間には効きません）。
    描画そのものは子プロセスの中なので、コアはきちんと全部使われます。
    """
    limit = max(1, min(len(jobs) or 1, round(limit)))
    results: list[dict] = []
    started = time.perf_counter()

    def worker(job: dict) -> dict:
        began = time.perf_counter()
        try:
            outcome = run(job) or {"code": 0}
        except Exception as error:  # noqa: BLE001
            outcome = {"code": 1, "error": str(error)}
        return {
            "job": job,
            "code": outcome.get("code", 0),
            "seconds": time.perf_counter() - began,
            "error": outcome.get("error"),
        }

    with ThreadPoolExecutor(max_workers=limit) as pool:
        for result in pool.map(worker, jobs):
            results.append(result)
            if on_done:
                on_done(
                    result,
                    {"done": len(results), "total": len(jobs), "elapsed": time.perf_counter() - started},
                )
    return results


def _render_in_child(job: dict, options: dict[str, Any]) -> dict[str, Any]:
    """子プロセスで 1 本描く。1 本が落ちても他を巻き込まないので子にしている。"""
    args = [*_movo_command(), "render", job["template"], "--output", job["output"], "--quiet"]
    for key, value in job["params"].items():
        args += ["--set", _assignment_for(key, value)]
    if isinstance(options.get("quality"), str):
        args += ["--quality", options["quality"]]
    if isinstance(options.get("format"), str):
        args += ["--format", options["format"]]
    if isinstance(options.get("seed"), (int, float)):
        args += ["--seed", str(options["seed"])]
    if options.get("noCache") is True or options.get("cache") is False:
        args.append("--no-cache")
    if options.get("generate") is False:
        args.append("--no-generate")

    completed = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode == 0:
        return {"code": 0}
    return {"code": completed.returncode, "error": _last_lines(completed.stderr or "")}


def _movo_command() -> list[str]:
    """自分自身を起こすためのコマンド。

    **単体 EXE（PyInstaller）では `sys.executable` が movo.exe そのもの** です。
    `-m movo.cli.main` を付けると «movo に -m という引数» として渡ってしまうので、
    ここで分けます。
    """
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "movo.cli.main"]


def _assignment_for(key: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        return f"{key}={json.dumps(value, ensure_ascii=False)}"
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f'{key}={"true" if value else "false"}'
    return f"{key}={value}"


def _last_lines(text: str, count: int = 3) -> str:
    return " / ".join(text.strip().split("\n")[-count:])


def _declared_param_names(template: str) -> list[str] | None:
    """テンプレートが宣言している params の名前（表の列を照らし合わせるのに使う）。"""
    try:
        read = read_project_file(template)
        apply_extends = bridge.pick("movo.schema", "apply_extends", "applyExtends")
        list_params = bridge.pick("movo.schema.params", "list_params", "listParams")
        return [entry["key"] for entry in list_params(apply_extends(read["raw"], file=read["file"]))]
    except Exception:  # noqa: BLE001
        # 読めなくても «一括で回す» こと自体は止めない（各本のエラーで分かる）
        return None


def batch_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    target = positional[0] if positional else None
    out = options.get("out") or options.get("output")
    rows = parse_table(_read_input(options["input"]), options["input"]) if isinstance(options.get("input"), str) else None
    template = _resolve_template(target, os.getcwd()) if rows is not None else None

    plan = build_batch_plan(
        {
            "target": template or target,
            "rows": rows,
            "out": out,
            "cwd": os.getcwd(),
            "declared": _declared_param_names(template) if template else None,
            "continue_existing": options.get("continue") is True,
        }
    )
    for warning in plan["warnings"]:
        logger.warn(warning)

    total = len(plan["jobs"])
    if plan["skipped"]:
        logger.info(f'{len(plan["skipped"])} 本は書き出し済みなので飛ばします（--continue）')
    if total == 0:
        logger.success("書き出すものはありません")
        return {"results": [], "skipped": plan["skipped"]}

    requested = options.get("jobs") if isinstance(options.get("jobs"), (int, float)) else default_job_count()
    limit = max(1, min(total, round(requested)))
    logger.step(f"{total} 本を同時 {limit} 本で書き出します")

    for job in plan["jobs"]:
        Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)

    def on_done(result: dict, progress: dict) -> None:
        # 残り見込みは «終わった本数あたりの平均» から。並列数が同じなら十分当たる。
        eta = (progress["elapsed"] / progress["done"]) * (progress["total"] - progress["done"])
        mark = style.green("v") if result["code"] == 0 else style.red("x")
        logger.info(
            f'  [{progress["done"]}/{progress["total"]}] {mark} {result["job"]["name"]}  {result["seconds"]:.1f}s'
            f'  経過 {progress["elapsed"]:.0f}s / 残り見込み {eta:.0f}s'
        )
        if result.get("error"):
            logger.warn(f'      {result["error"]}')

    results = run_jobs(plan["jobs"], limit, lambda job: _render_in_child(job, options), on_done)

    failed = [r for r in results if r["code"] != 0]
    if failed:
        logger.error(f"{len(failed)} 本が失敗しました")
        for result in failed:
            logger.error(f'  {result["job"]["name"]}  → {result["job"]["output"]}')
    else:
        logger.success(f"{len(results)} 本を書き出しました")
    if options.get("json"):
        say(
            json.dumps(
                {
                    "total": len(results),
                    "failed": len(failed),
                    "skipped": [job["output"] for job in plan["skipped"]],
                    "results": [
                        {
                            "name": r["job"]["name"],
                            "output": r["job"]["output"],
                            "code": r["code"],
                            "seconds": r["seconds"],
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if failed:
        raise SystemExit(1)
    return {"results": results, "skipped": plan["skipped"]}


def _read_input(file: str) -> str:
    path = Path(file).resolve()
    if not path.exists():
        raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f"--input のファイルが見つかりません: {file}")
    return path.read_text(encoding="utf-8")
