"""エフェクトグラフの評価（仕様書 22 章）。

エフェクトを «並べる» だけだと、「ぼかした絵と元の絵を掛け合わせる」のような
**枝分かれして合流する** 加工が書けません。ここではノードを依存順にたどり、
`source` から `output` までを 1 回ずつ評価します。

    { "nodes": [ { "id": "src",  "type": "source" },
                 { "id": "soft", "type": "blur", "radius": 20 },
                 { "id": "mix",  "type": "blend", "blend": "screen" } ],
      "connections": [ { "from": "src",  "to": "soft" },
                       { "from": "src",  "to": "mix" },
                       { "from": "soft", "to": "mix" } ],
      "output": "mix" }

入力が 1 つのノードは最初の接続だけを使い、`blend` / `composite` は 2 つ受け取り
ます（マスクや多重合成をこれで書きます）。**評価結果はノードごとに覚えておく**
ので、同じ枝を 2 回計算することはありません。上の例なら `src` は 1 回だけです。

`output` を書かなければ «最後に並べたノード» が出口になります。
"""

from __future__ import annotations

from movo.core.bitmap import Bitmap
from movo.renderer.effects import apply_effect, composite


class EffectGraphError(ValueError):
    """輪ができている・出口が無いなど、グラフとして成り立たないとき。"""


def apply_effect_graph(source: Bitmap, graph: dict | None, ctx: dict | None = None) -> Bitmap:
    """グラフを評価して 1 枚のビットマップにする。

    :param source: `source` ノードが返す絵
    :param graph: `{"nodes": [...], "connections": [...], "output": "id"}`
    :param ctx: `apply_effect` にそのまま渡ります
    """
    ctx = ctx or {}
    if not graph or not isinstance(graph.get("nodes"), list) or not graph["nodes"]:
        return source

    nodes = {node["id"]: node for node in graph["nodes"]}
    incoming: dict[str, list[str]] = {node["id"]: [] for node in graph["nodes"]}
    for connection in graph.get("connections") or []:
        if connection.get("from") not in nodes or connection.get("to") not in nodes:
            # 綴り間違いで «黙って絵が変わる» のを避けたいので、飛ばしつつ報せます
            _warn(f"エフェクトグラフの接続 {connection.get('from')} → {connection.get('to')} は知らないノードを指しています。飛ばします")
            continue
        incoming[connection["to"]].append(connection["from"])

    output_id = graph.get("output") or graph["nodes"][-1]["id"]
    if output_id not in nodes:
        raise EffectGraphError(f'エフェクトグラフの出口 "{output_id}" が宣言されていません')

    results: dict[str, Bitmap] = {}
    visiting: set[str] = set()

    def evaluate(node_id: str) -> Bitmap:
        if node_id in results:
            return results[node_id]
        if node_id in visiting:
            raise EffectGraphError(f'エフェクトグラフがノード "{node_id}" で輪になっています')
        visiting.add(node_id)
        node = nodes[node_id]
        inputs = [evaluate(source_id) for source_id in incoming.get(node_id, [])]

        kind = node.get("type")
        if kind == "source":
            result = source
        elif kind in ("blend", "composite"):
            base = inputs[0] if inputs else source
            overlay = inputs[1] if len(inputs) > 1 else None
            result = base.copy()
            if overlay is not None:
                composite(
                    result,
                    overlay,
                    node.get("offsetX", 0) or 0,
                    node.get("offsetY", 0) or 0,
                    node.get("opacity", 1) if node.get("opacity") is not None else 1,
                    node.get("blend", "normal") or "normal",
                )
        elif kind == "output":
            result = inputs[0] if inputs else source
        else:
            result = apply_effect(inputs[0] if inputs else source, node, ctx)

        visiting.discard(node_id)
        results[node_id] = result
        return result

    return evaluate(output_id)


def _warn(message: str) -> None:
    """ロガーが移植されるまでの繋ぎ（`movo.core.logger` があればそちらを使います）。"""
    try:
        from movo.core.logger import logger

        logger.warn(message)
    except ImportError:
        import warnings

        warnings.warn(message, stacklevel=3)


__all__ = ["EffectGraphError", "apply_effect_graph"]
