"""Test helpers: parse rendered graph text back into a normalized edge set, and build a variant's graph."""
from __future__ import annotations

import json
import re

import networkx as nx

from gsi.serial.variant import Variant, normalize_edges

_PAIR = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)(?:\s*,\s*[-\d.]+)?\s*\)")
_INT = re.compile(r"-?\d+")


def _adj_lines(text: str, pattern: str) -> list[tuple[int, int]]:
    edges = []
    for m in re.finditer(pattern, text, re.MULTILINE):
        u = int(m.group(1))
        body = re.sub(r"\(weight [^)]*\)", "", m.group(2))
        edges += [(u, int(w)) for w in _INT.findall(body)]
    return edges


def parse(text: str, syntax: str, structure: str, directed: bool) -> frozenset[str]:
    if syntax == "json":
        obj = json.loads(text)
        if "edges" in obj:
            edges = [(e["source"], e["target"]) if isinstance(e, dict) else (e[0], e[1]) for e in obj["edges"]]
        else:
            edges = [(int(u), (w["node"] if isinstance(w, dict) else w)) for u, ws in obj["adjacency"].items() for w in ws]
    elif syntax == "networkx_code":
        ns: dict = {}
        exec(text, ns)
        edges = list(ns["G"].edges())
    elif syntax == "plain":
        if structure == "edge_list":
            body = text.split("The edges are:", 1)[1]
            edges = [(int(u), int(v)) for u, v in _PAIR.findall(body)]
        else:
            edges = _adj_lines(text, r"^(\d+): (.*)$")
    elif syntax == "graphqa_nl":
        if structure == "edge_list":
            body = text.split("The edges in G are:", 1)[1]
            edges = [(int(u), int(v)) for u, v in _PAIR.findall(body)]
        else:
            edges = _adj_lines(text, r"^Node (\d+) is connected to nodes? (.*)\.$")  # "node 2" when only one
    elif syntax == "erdos_nl":
        if structure == "edge_list":
            body = text.split("The edges are:", 1)[1]
            edges = [(int(u), int(v)) for u, v in _PAIR.findall(body)]
        else:
            edges = _adj_lines(text, r"^(\d+): (.*)$")
    else:
        raise ValueError(syntax)
    return normalize_edges(edges, directed)


def to_nx(v: Variant) -> nx.Graph:
    G = nx.DiGraph() if v.directed else nx.Graph()
    G.add_nodes_from(v.node_seq)
    if v.weighted:
        G.add_weighted_edges_from((e[0], e[1], e[2]) for e in v.edge_seq)
    else:
        G.add_edges_from((e[0], e[1]) for e in v.edge_seq)
    return G
