"""Render a Variant into the graph-description block shown to the model.

Dispatch is on (syntax, structure). Edge/neighbour order is taken verbatim
from the variant's node_seq/edge_seq so ordering perturbations survive rendering.
"""
from __future__ import annotations

import json

from gsi.serial.variant import Variant


def _edges_for_render(v: Variant) -> list[list]:
    edges = [list(e) for e in v.edge_seq]
    if v.params.get("replicated") and not v.directed:
        out = []
        for e in edges:
            out.append(e)
            out.append([e[1], e[0], *e[2:]])
        edges = out
    return edges


def adjacency(v: Variant) -> list[tuple[int, list]]:
    adj: dict[int, list] = {u: [] for u in v.node_seq}
    for e in v.edge_seq:
        u, w, *rest = e
        adj[u].append([w, *rest])
        if not v.directed:
            adj[w].append([u, *rest])
    return [(u, adj[u]) for u in v.node_seq]


def _kind(v: Variant) -> str:
    return "directed" if v.directed else "undirected"


def _edge_str(e: list) -> str:
    return "(" + ", ".join(str(x) for x in e) + ")"


def _nbr_str(n: list) -> str:
    return str(n[0]) if len(n) == 1 else f"{n[0]} (weight {n[1]})"


def _nodes_nl(nodes: list[int]) -> str:
    s = [str(n) for n in nodes]
    if len(s) == 1:
        return s[0]
    return ", ".join(s[:-1]) + ", and " + s[-1]


# ---------- plain ----------

def plain_edge_list(v: Variant) -> str:
    head = f"This is an {_kind(v)} graph with nodes {', '.join(map(str, v.node_seq))}."
    if v.directed:
        head += " An edge (u, v) goes from node u to node v."
    if v.weighted:
        head += " Each edge is written as (u, v, w) where w is its weight."
    edges = ", ".join(_edge_str(e) for e in _edges_for_render(v))
    return f"{head}\nThe edges are: {edges if edges else '(none)'}."


def plain_adj_list(v: Variant) -> str:
    head = f"This is an {_kind(v)} graph with nodes {', '.join(map(str, v.node_seq))}."
    if v.directed:
        head += " Each line lists a node followed by the nodes it has an edge to."
    else:
        head += " Each line lists a node followed by its neighbours."
    lines = [f"{u}: {', '.join(_nbr_str(n) for n in nbrs) if nbrs else '(none)'}" for u, nbrs in adjacency(v)]
    return head + "\n" + "\n".join(lines)


# ---------- json ----------

def json_edge_list(v: Variant) -> str:
    obj = {"directed": v.directed, "nodes": list(v.node_seq), "edges": _edges_for_render(v)}
    if v.weighted:
        obj["edges"] = [{"source": e[0], "target": e[1], "weight": e[2]} for e in obj["edges"]]
    return json.dumps(obj)


def json_adj_list(v: Variant) -> str:
    adj = {str(u): ([n[0] for n in nbrs] if not v.weighted else [{"node": n[0], "weight": n[1]} for n in nbrs])
           for u, nbrs in adjacency(v)}
    return json.dumps({"directed": v.directed, "adjacency": adj})


# ---------- networkx code ----------

def networkx_edge_list(v: Variant) -> str:
    cls = "DiGraph" if v.directed else "Graph"
    edges = _edges_for_render(v)
    adder = "add_weighted_edges_from" if v.weighted else "add_edges_from"
    edge_lit = "[" + ", ".join(_edge_str(e) for e in edges) + "]"
    return (f"import networkx as nx\nG = nx.{cls}()\nG.add_nodes_from({list(v.node_seq)})\n"
            f"G.{adder}({edge_lit})")


def networkx_adj_list(v: Variant) -> str:
    cls = "DiGraph" if v.directed else "Graph"
    if v.weighted:
        body = ", ".join(f"{u}: {{" + ", ".join(f"{n[0]}: {{'weight': {n[1]}}}" for n in nbrs) + "}"
                         for u, nbrs in adjacency(v))
    else:
        body = ", ".join(f"{u}: {[n[0] for n in nbrs]}" for u, nbrs in adjacency(v))
    return f"import networkx as nx\nadjacency = {{{body}}}\nG = nx.{cls}(adjacency)"


# ---------- GraphQA natural-language encoders (Fatemi et al.) ----------

def graphqa_edge_list(v: Variant) -> str:
    if v.directed:
        head = "In a directed graph, (i,j) means that node i and node j are connected with a directed edge from i to j."
    else:
        head = "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge."
    edges = " ".join(_edge_str(e[:2]) for e in _edges_for_render(v))
    return f"{head} G describes a graph among nodes {_nodes_nl(v.node_seq)}.\nThe edges in G are: {edges}."


def graphqa_adj_list(v: Variant) -> str:
    """Fatemi's "incident" encoder, verbatim (Appendix A.1). Two details differ from the
    "adjacency" encoder above and are easy to get wrong: there is no "nodes" after "among",
    and the neighbour noun is singular for a node with exactly one neighbour."""
    kind = "directed graph" if v.directed else "graph"
    head = f"G describes a {kind} among {_nodes_nl(v.node_seq)}.\nIn this graph:"
    lines = []
    for u, nbrs in adjacency(v):
        if not nbrs:
            lines.append(f"Node {u} is not connected to any node.")
        else:
            noun = "node" if len(nbrs) == 1 else "nodes"
            lines.append(f"Node {u} is connected to {noun} {', '.join(str(n[0]) for n in nbrs)}.")
    return head + "\n" + "\n".join(lines)


# ---------- Erdős native wording (Guo et al.); node range is invariant under relabeling ----------

def _erdos_head(v: Variant) -> str:
    article = "a directed" if v.directed else "an undirected"
    return f"Here is {article} graph containing nodes from {min(v.node_seq)} to {max(v.node_seq)}."


def erdos_edge_list(v: Variant) -> str:
    edges = ", ".join(_edge_str(e) for e in _edges_for_render(v))
    return f"{_erdos_head(v)} The edges are: {edges if edges else '(none)'}."


def erdos_adj_list(v: Variant) -> str:
    lines = [f"{u}: {', '.join(_nbr_str(n) for n in nbrs) if nbrs else '(none)'}" for u, nbrs in adjacency(v)]
    what = "the nodes it has an edge to" if v.directed else "its neighbours"
    return f"{_erdos_head(v)} Each line below lists a node followed by {what}:\n" + "\n".join(lines)


RENDERERS = {
    ("erdos_nl", "edge_list"): erdos_edge_list,
    ("erdos_nl", "adj_list"): erdos_adj_list,
    ("plain", "edge_list"): plain_edge_list,
    ("plain", "adj_list"): plain_adj_list,
    ("json", "edge_list"): json_edge_list,
    ("json", "adj_list"): json_adj_list,
    ("networkx_code", "edge_list"): networkx_edge_list,
    ("networkx_code", "adj_list"): networkx_adj_list,
    ("graphqa_nl", "edge_list"): graphqa_edge_list,
    ("graphqa_nl", "adj_list"): graphqa_adj_list,
}


def render(v: Variant) -> str:
    key = (v.params["syntax"], v.params["structure"])
    if key not in RENDERERS:
        raise ValueError(f"no renderer for syntax={key[0]} structure={key[1]}")
    return RENDERERS[key](v)
