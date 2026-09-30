from __future__ import annotations

from typing import Any

import networkx as nx

NODE_VALUED = ("node", "node_set", "path", "edge_set")


def relabel_answer(value: Any, answer_type: str, mapping: dict[int, int]) -> Any:
    """Map node labels inside an answer through `mapping` (labels not in it are kept)."""
    if value is None or answer_type not in NODE_VALUED:
        return value
    m = lambda x: mapping.get(x, x)
    if answer_type == "node":
        return m(value)
    if answer_type in ("node_set", "path"):
        return [m(x) for x in value]
    return [[m(e[0]), m(e[1])] for e in value]


def normalize_answer(value: Any, answer_type: str, directed: bool = False) -> Any:
    """Hashable canonical form used for invariance comparisons."""
    if value is None:
        return None
    if answer_type == "bool":
        return bool(value)
    if answer_type in ("int", "node"):
        return int(value)
    if answer_type == "float":
        return round(float(value), 4)
    if answer_type == "node_set":
        return tuple(sorted(int(x) for x in value))
    if answer_type == "path":
        return tuple(int(x) for x in value)
    if answer_type == "edge_set":
        pairs = [(int(e[0]), int(e[1])) for e in value]
        if not directed:
            pairs = [tuple(sorted(p)) for p in pairs]
        return tuple(sorted(set(pairs)))
    raise ValueError(f"unknown answer_type {answer_type}")


def compare(pred: Any, gt: Any, answer_type: str, *, G: nx.Graph | None = None) -> bool:
    if pred is None:
        return False
    if answer_type == "float":
        a, b = float(pred), float(gt)
        return abs(a - b) <= max(1e-2, 1e-2 * abs(b))
    if answer_type == "path":
        if G is None:
            return list(pred) == list(gt)
        if not pred or pred[0] != gt[0] or pred[-1] != gt[-1]:
            return False
        if not all(G.has_edge(u, v) for u, v in zip(pred, pred[1:])):
            return False
        if nx.is_weighted(G):
            return abs(nx.path_weight(G, pred, "weight") - nx.path_weight(G, gt, "weight")) < 1e-9
        return len(pred) == len(gt)
    directed = G.is_directed() if G is not None else False
    return normalize_answer(pred, answer_type, directed) == normalize_answer(gt, answer_type, directed)
