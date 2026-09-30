"""Erdős suite (Guo et al., NeurIPS 2025) from HuggingFace `PKU-ML/Erdos`, test split.

Only tasks with a unique, checkable answer are exposed; ordering-ambiguous tasks
(topological sort, MST, BFS/DFS orders, TSP, ...) are left out.
"""
from __future__ import annotations

import ast
import random
import re

from gsi.data.base import GraphInstance
from gsi.score.parse_answer import parse_value

# task -> our answer_type
TASKS: dict[str, str] = {
    "shortest_path": "path",
    "weighted_shortest_path": "path",
    "triangles": "int",
    "degree": "int",
    "edge_number": "int",
    "node_number": "int",
    "connected_component_number": "int",
    "strongly_connected_number": "int",
    "diameter": "float",
    "radius": "float",
    "density": "float",
    "clustering_coefficient": "float",
    "jaccard_coefficient": "float",
    "maximum_flow": "float",
    "edge_existence": "bool",
    "has_cycle": "bool",
    "is_bipartite": "bool",
    "is_regular": "bool",
    "local_connectivity": "bool",
    "neighbor": "node_set",
    "common_neighbor": "node_set",
    "center": "node_set",
    "periphery": "node_set",
    "bridges": "edge_set",
}

DEFAULT_TASKS = ("shortest_path", "triangles", "degree", "edge_number", "connected_component_number", "diameter",
                 "density", "edge_existence", "has_cycle", "neighbor", "common_neighbor", "bridges")

_NODE = re.compile(r"\bnode (\d+)\b")
_FORMAT_LINE = re.compile(r"\n+((?:You need to format your answer|Your answer should be)[^\n]*)$", re.DOTALL)
_EDGE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([-+]?\d+(?:\.\d+)?))?\s*\)")


def _edges_from_prompt(prompt: str) -> list[list] | None:
    """The `edges` column drops capacities/weights for some tasks; the prompt is what the model sees."""
    if "The edges are:" not in prompt:
        return None
    body = prompt.split("The edges are:", 1)[1].split("\n", 1)[0]
    edges = []
    for u, v, w in _EDGE.findall(body):
        e = [int(u), int(v)]
        if w:
            e.append(float(w) if "." in w else int(w))
        edges.append(e)
    return edges


def _split_prompt(prompt: str) -> tuple[str, str, str]:
    """-> (preamble before the graph, question without the format sentence, the format sentence)."""
    pre, rest = prompt.split("Here is ", 1)
    q = rest.split("Question:", 1)[1] if "Question:" in rest else rest
    m = _FORMAT_LINE.search(q)
    hint = m.group(1).strip() if m else ""
    q = _FORMAT_LINE.sub("", q).strip()
    return pre.strip(), q, hint


def _query_args(question: str) -> dict:
    nodes = [int(x) for x in _NODE.findall(question)]
    if len(nodes) == 1:
        return {"node": nodes[0]}
    if len(nodes) == 2:
        return {"u": nodes[0], "v": nodes[1]}
    return {}


def row_to_instance(row: dict, idx: int) -> GraphInstance | None:
    task = row["task"]
    if task not in TASKS:
        return None
    t = TASKS[task]
    lo, hi = ast.literal_eval(row["nodes"])
    edges = _edges_from_prompt(row["prompt"])
    if edges is None:
        return None
    column = [list(e) for e in ast.literal_eval(row["edges"])]
    if len(edges) != len(column) or any(e[:2] != c[:2] for e, c in zip(edges, column)):
        return None
    weighted = bool(edges) and len(edges[0]) == 3
    gt = parse_value(row["answer"], t)
    if gt is None:
        return None
    preamble, question, hint = _split_prompt(row["prompt"])
    return GraphInstance(
        id=f"erdos-{task}-{idx:04d}",
        dataset="erdos",
        task=task,
        directed=row["direction"] == "directed",
        weighted=weighted,
        nodes=list(range(lo, hi + 1)),
        edges=edges,
        query_args=_query_args(question),
        answer_type=t,
        ground_truth=gt,
        native_prompt=row["prompt"],
        meta={"graph_id": f"erdos-{task}-{idx:04d}", "preamble": preamble, "question": question,
              "format_hint": hint, "n": hi - lo + 1, "m": len(edges), "source_file": row.get("source_file")},
    )


# The dataset revision the committed instances were sampled from (rebuilding from it is byte-identical).
ERDOS_REVISION = "786f85015eec845e821ec45533087e8c7955c025"


def build_erdos(tasks=DEFAULT_TASKS, n_per_task: int = 100, seed: int = 0, split: str = "test",
                max_prompt_chars: int | None = None) -> list[GraphInstance]:
    from datasets import load_dataset

    ds = load_dataset("PKU-ML/Erdos", revision=ERDOS_REVISION)[split]
    rng = random.Random(seed)
    out: list[GraphInstance] = []
    for task in tasks:
        rows = [r for r in ds if r["task"] == task]
        if max_prompt_chars:
            rows = [r for r in rows if len(r["prompt"]) <= max_prompt_chars]
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        for i in sorted(idxs[:n_per_task]):
            inst = row_to_instance(rows[i], i)
            if inst is not None:
                out.append(inst)
    return out


def question(inst: GraphInstance, label_map: dict[int, int] | None = None) -> str:
    q = inst.meta["question"]
    if label_map:
        q = _NODE.sub(lambda m: f"node {label_map.get(int(m.group(1)), int(m.group(1)))}", q)
    return q
