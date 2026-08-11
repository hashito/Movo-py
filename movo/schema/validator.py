"""小さな JSON Schema（draft-07 の部分集合）検証器。

自前で持っているのは 2 つの理由からです。

1. `movo validate` を «依存ゼロ» で動かすため
2. エラーの場所を **書き手の書き方** で出すため
   （`scenes[1].layers[2].modifiers[0].frequency`）

対応キーワード: `$ref`（ローカル）, type, enum, const, required, properties,
patternProperties, additionalProperties, items, additionalItems, minItems,
maxItems, uniqueItems, minimum, maximum, exclusiveMinimum, exclusiveMaximum,
multipleOf, minLength, maxLength, pattern, format（情報のみ）, oneOf, anyOf,
allOf, not, dependencies。
"""

from __future__ import annotations

import json
import math
import re


def _is_number(value) -> bool:
    """JSON の数。Python の真偽値は数に数えない（JS の typeof に合わせる）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_integer(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float):
        return math.isfinite(value) and value.is_integer()
    return True


TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": _is_number,
    "integer": _is_integer,
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, (list, tuple)),
    "null": lambda v: v is None,
}


def join_path(parent, key) -> str:
    """親の場所とキーを «書き手の書き方» でつなぐ。"""
    if isinstance(key, int) and not isinstance(key, bool):
        return f"{parent}[{key}]"
    if not parent:
        return str(key)
    return f"{parent}.{key}"


def _type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    return "undefined"


def _deep_equal(a, b) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys()) and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return False
    if isinstance(a, dict) or isinstance(b, dict):
        return False
    return a == b


def _json_dumps(value) -> str:
    """メッセージに値を埋めるときの書き方（JS の JSON.stringify に寄せる）。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _stable_key(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, default=str)


class SchemaValidator:
    """スキーマ 1 本を持ち、値を検証する。"""

    def __init__(self, schema: dict):
        self.schema = schema
        # `pattern` と `patternProperties` の正規表現は使い回す。
        # 1 プロジェクトで数万回走るので、毎回コンパイルすると効いてきます。
        self._regex_cache: dict[str, re.Pattern] = {}

    def _regex(self, pattern: str) -> re.Pattern:
        compiled = self._regex_cache.get(pattern)
        if compiled is None:
            compiled = re.compile(pattern)
            self._regex_cache[pattern] = compiled
        return compiled

    def resolve(self, node):
        """`$ref` をたどる。ローカル参照（`#/definitions/...`）だけ。"""
        current = node
        guard = 0
        while isinstance(current, dict) and current.get("$ref") and guard < 32:
            guard += 1
            ref = current["$ref"]
            if not ref.startswith("#/"):
                break
            target = self.schema
            for part in ref[2:].split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                target = target.get(key) if isinstance(target, dict) else None
                if target is None:
                    break
            if not target:
                break
            rest = {k: v for k, v in current.items() if k != "$ref"}
            current = {**target, **rest} if rest else target
        return current

    def validate(self, value) -> dict:
        issues: list[dict] = []
        self._validate(value, self.schema, "", issues)
        return {"valid": len(issues) == 0, "issues": issues}

    def _validate(self, value, raw_schema, path, issues):
        schema = self.resolve(raw_schema)
        if schema is None or schema is True:
            return
        if schema is False:
            issues.append({"path": path, "message": "value is not allowed here"})
            return
        if not isinstance(schema, dict):
            return

        if schema.get("type"):
            types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not any(TYPE_CHECKS[t](value) for t in types if t in TYPE_CHECKS):
                issues.append(
                    {
                        "path": path,
                        "message": f"expected {' or '.join(types)} but found {_type_name(value)}",
                    }
                )
                return

        if "const" in schema and not _deep_equal(value, schema["const"]):
            issues.append({"path": path, "message": f"must equal {_json_dumps(schema['const'])}"})

        if schema.get("enum") and not any(_deep_equal(option, value) for option in schema["enum"]):
            options = ", ".join(_json_dumps(e) for e in schema["enum"])
            issues.append({"path": path, "message": f"must be one of: {options}"})

        if _is_number(value):
            self._validate_number(value, schema, path, issues)
        if isinstance(value, str):
            self._validate_string(value, schema, path, issues)
        if isinstance(value, (list, tuple)):
            self._validate_array(value, schema, path, issues)
        elif isinstance(value, dict):
            self._validate_object(value, schema, path, issues)

        if schema.get("allOf"):
            for sub in schema["allOf"]:
                self._validate(value, sub, path, issues)

        if schema.get("anyOf"):
            branches = []
            for sub in schema["anyOf"]:
                local: list[dict] = []
                self._validate(value, sub, path, local)
                branches.append(local)
            if all(len(b) > 0 for b in branches):
                issues.append({"path": path, "message": _describe_branch_failure(schema, branches, "anyOf")})

        if schema.get("oneOf"):
            branches = []
            for sub in schema["oneOf"]:
                local = []
                self._validate(value, sub, path, local)
                branches.append(local)
            passing = sum(1 for b in branches if len(b) == 0)
            if passing == 0:
                issues.append({"path": path, "message": _describe_branch_failure(schema, branches, "oneOf")})
            elif passing > 1 and schema.get("strictOneOf"):
                issues.append({"path": path, "message": "matches more than one allowed form"})

        if schema.get("not") is not None:
            local = []
            self._validate(value, schema["not"], path, local)
            if len(local) == 0:
                issues.append({"path": path, "message": "value matches a forbidden schema"})

    def _validate_number(self, value, schema, path, issues):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append({"path": path, "message": f"must be greater than or equal to {_num_text(schema['minimum'])}"})
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            issues.append({"path": path, "message": f"must be greater than {_num_text(schema['exclusiveMinimum'])}"})
        if "maximum" in schema and value > schema["maximum"]:
            issues.append({"path": path, "message": f"must be less than or equal to {_num_text(schema['maximum'])}"})
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            issues.append({"path": path, "message": f"must be less than {_num_text(schema['exclusiveMaximum'])}"})
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None and multiple_of > 0:
            ratio = value / multiple_of
            if abs(ratio - round(ratio)) > 1e-9:
                issues.append({"path": path, "message": f"must be a multiple of {_num_text(multiple_of)}"})

    def _validate_string(self, value, schema, path, issues):
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append({"path": path, "message": f"must be at least {schema['minLength']} characters"})
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            issues.append({"path": path, "message": f"must be at most {schema['maxLength']} characters"})
        if schema.get("pattern") and not self._regex(schema["pattern"]).search(value):
            issues.append({"path": path, "message": f"must match {schema['pattern']}"})

    def _validate_array(self, value, schema, path, issues):
        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append({"path": path, "message": f"must contain at least {schema['minItems']} item(s)"})
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append({"path": path, "message": f"must contain at most {schema['maxItems']} item(s)"})
        if schema.get("uniqueItems"):
            seen = set()
            for item in value:
                key = _stable_key(item)
                if key in seen:
                    issues.append({"path": path, "message": "items must be unique"})
                    break
                seen.add(key)
        items = schema.get("items")
        if isinstance(items, list):
            for index, item in enumerate(value):
                sub = items[index] if index < len(items) else schema.get("additionalItems")
                if sub:
                    self._validate(item, sub, join_path(path, index), issues)
        elif items:
            for index, item in enumerate(value):
                self._validate(item, items, join_path(path, index), issues)

    def _validate_object(self, value, schema, path, issues):
        if isinstance(schema.get("required"), list):
            for key in schema["required"]:
                if key not in value:
                    issues.append({"path": join_path(path, key), "message": f'"{key}" is required'})
        properties = schema.get("properties") or {}
        for key, sub in properties.items():
            if key in value:
                self._validate(value[key], sub, join_path(path, key), issues)
        pattern_props = schema.get("patternProperties") or {}
        pattern_entries = [(self._regex(p), s) for p, s in pattern_props.items()]
        additional = schema.get("additionalProperties", None)
        has_additional = "additionalProperties" in schema
        for key, entry in value.items():
            matched = key in properties
            for regex, sub in pattern_entries:
                if regex.search(key):
                    matched = True
                    self._validate(entry, sub, join_path(path, key), issues)
            if not matched and has_additional and additional is False:
                issues.append({"path": join_path(path, key), "message": f'unknown property "{key}"'})
            elif not matched and isinstance(additional, dict):
                self._validate(entry, additional, join_path(path, key), issues)
        dependencies = schema.get("dependencies")
        if dependencies:
            for key, dep in dependencies.items():
                if key not in value:
                    continue
                if isinstance(dep, list):
                    for required in dep:
                        if required not in value:
                            issues.append(
                                {
                                    "path": join_path(path, required),
                                    "message": f'"{required}" is required when "{key}" is present',
                                }
                            )
                else:
                    self._validate(value, dep, path, issues)


def _num_text(value) -> str:
    """メッセージに出す数。`1.0` ではなく `1` と書く（JS の書き方）。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _describe_branch_failure(schema, branches, keyword) -> str:
    if schema.get("errorMessage"):
        return schema["errorMessage"]
    reasons = []
    for branch in branches:
        if branch and branch[0].get("message") and branch[0]["message"] not in reasons:
            reasons.append(branch[0]["message"])
    listed = "; ".join(reasons[:3])
    return f"does not match any allowed form ({keyword})" + (f": {listed}" if listed else "")


__all__ = ["SchemaValidator", "join_path"]
