"""JS 版の作例 JSON が、そのまま Python 版の検証を通ること。

**これがいちばん大事な受け入れ条件です。** JSON の書き方が完全に互換であること
（README.ja.md「JS 版から変えないこと」）を、実物 48 本で確かめます。

**このテストだけは隣に JS 版（`../Movo`）がある前提です。**

Movo-py は JS 版から独立した単体のプロジェクトで、実行にも配布にも
JS 版は要りません。ここは «移植が忠実か» を確かめるためだけの検証で、
**JS 版が無ければ丸ごと飛ばします**（CI でも手元でも落ちません）。
"""

import json
from pathlib import Path

import pytest

from movo.schema import normalize_project, validate_project
from movo.schema.extends import strip_json_comments

EXAMPLES = Path(__file__).resolve().parents[2] / "Movo" / "examples"


def _project_files():
    if not EXAMPLES.is_dir():
        return []
    files = []
    for path in sorted(EXAMPLES.rglob("*.json")):
        parts = set(path.parts)
        if "cache" in parts or "assets" in parts:
            continue
        try:
            project = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
        # プロジェクト JSON かどうかは video があるかで見る（歌詞やレシピを外す）
        if isinstance(project, dict) and "video" in project:
            files.append(path)
    return files


FILES = _project_files()

pytestmark = pytest.mark.skipif(
    not FILES, reason=f"JS 版の作例が見つかりません（{EXAMPLES}）"
)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_js_example_projects_validate(path):
    project = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    result = validate_project(project, file=str(path))
    assert result["valid"] is True, result["issues"][:5]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_js_example_projects_normalize(path):
    project = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    normalized = normalize_project(project, file=str(path))
    # 正規化は «後段が完全な記録として扱える» ことが目的なので、
    # 既定値が入っていることだけ確かめる（値の一致は JS との突き合わせで見ている）
    assert normalized["video"]["width"] >= 1
    assert normalized["render"]["quality"] in ("draft", "preview", "standard", "high", "ultra")
    assert isinstance(normalized["scenes"], list)
    assert "layers" not in normalized
