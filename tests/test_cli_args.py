"""引数パーサ。**JS 版と «書き方» が 1 文字も変わらないこと** を確かめます。

ここが崩れると、JS 版で書いた手順書やスクリプトがそのままでは動かなくなります。
"""

from movo.cli.args import COMMON_SCHEMA, param_options_from, parse_args


def parse(argv):
    return parse_args(argv, COMMON_SCHEMA)


def test_値を取る指定と位置引数を分けて読む():
    options, positional = parse(["render", "movo.json", "-o", "out.mp4"])
    assert positional == ["render", "movo.json"]
    assert options["output"] == "out.mp4"


def test_イコールでも書ける():
    options, _ = parse(["--quality=high"])
    assert options["quality"] == "high"


def test_ハイフンはキャメルに直す():
    # `options["save-recipe"]` と書くのを呼ぶ側に強いると、書き忘れて
    # «指定が黙って効かない» ことが実際に起きました。
    options, _ = parse(["--save-recipe", "tmp/b.json", "--super-sample", "2"])
    assert options["saveRecipe"] == "tmp/b.json"
    assert options["superSample"] == 2


def test_no_を付けると偽になる():
    options, _ = parse(["--no-audio", "--no-check-flash"])
    assert options["audio"] is False
    assert options["checkFlash"] is False


def test_真偽値の指定は次の語を飲み込まない():
    # `--all-variants` を «値を取る指定» と読むと、ファイル名が消えます。
    options, positional = parse(["render", "--all-variants", "movo.json"])
    assert options["allVariants"] is True
    assert positional == ["render", "movo.json"]


def test_set_は繰り返せる():
    options, _ = parse(["--set", "a=1", "--set", "b=2"])
    assert options["set"] == ["a=1", "b=2"]


def test_数値の指定は数になる():
    options, _ = parse(["--from", "3", "--to", "8.5", "--jobs", "12"])
    assert options["from"] == 3
    assert options["to"] == 8.5
    assert options["jobs"] == 12


def test_jobs_は_auto_とも書ける():
    # 数値に直せない値は文字列のまま入り、resolve_job_count が解釈します。
    options, _ = parse(["--jobs", "auto"])
    assert options["jobs"] == "auto"


def test_短い指定をまとめて書ける():
    options, _ = parse(["-Vq"])
    assert options["verbose"] is True
    assert options["quality"] is True


def test_二重ハイフンから後ろは位置引数():
    _, positional = parse(["render", "--", "--not-an-option"])
    assert positional == ["render", "--not-an-option"]


def test_params_の指定だけを取り出す():
    options, _ = parse(["--set", "a=1", "--params", "p.json", "--save-recipe", "r.json"])
    picked = param_options_from(options)
    # `--set` は繰り返せるので、1 回でも配列で入ります（受け取る側が
    # «1 回のときだけ文字列» を書き分けずに済むように）。
    assert picked["set"] == ["a=1"]
    assert picked["params"] == "p.json"
    assert picked["save_recipe"] == "r.json"
