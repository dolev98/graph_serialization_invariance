"""GraphQA-style graphs and tasks (Fatemi et al., ICLR 2024), regenerated locally.

The generator follows Fatemi Appendix A.4: seven families, 5-20 nodes, ER edge probability
drawn from [0, 1], SBM communities drawn from 2 to 10, and the 500/500/500/500/100/100/100
sampling proportions (the last three families "have less variety"). A.4 leaves the rest open;
our choices are 1-4 attachment edges for BA and equal-sized SBM blocks with edge probability
0.7 inside a block and 0.1 between blocks. Ground truth is computed with networkx, as they did.
Node labels are 0..n-1; canonical edge order is sorted (u < v).

networkx 3.7 changed `stochastic_block_model`, so the SBM graphs of data/processed/graphqa were
drawn with networkx 3.6 and are not reproduced by later versions; the committed instances are
the data.
"""
from __future__ import annotations

import random

import networkx as nx

from gsi.data.base import GraphInstance

GENERATORS = ("er", "ba", "sfn", "sbm", "star", "path", "complete")
NODE_RANGE = (5, 20)
# Fatemi A.4: 500 graphs each for ER/BA/SFN/SBM, 100 each for path/complete/star
GENERATOR_WEIGHTS = {"er": 500, "ba": 500, "sfn": 500, "sbm": 500, "star": 100, "path": 100, "complete": 100}

# Fatemi A.2's seven tasks and their answer types
TASKS: dict[str, str] = {
    "edge_existence": "bool",
    "node_degree": "int",
    "node_count": "int",
    "edge_count": "int",
    "connected_nodes": "node_set",
    "disconnected_nodes": "node_set",
    "cycle_check": "bool",
}

QUESTIONS: dict[str, str] = {
    "edge_existence": "Is node {u} connected to node {v}?",
    "node_degree": "What is the degree of node {node}?",
    "node_count": "How many nodes are in this graph?",
    "edge_count": "How many edges are in this graph?",
    "connected_nodes": "List all the nodes connected to node {node} in ascending order.",
    "disconnected_nodes": "List all the nodes that are not connected to node {node} (excluding node {node} itself).",
    "cycle_check": "Is there a cycle in this graph?",
}


def generate_graph(kind: str, n: int, rng: random.Random) -> nx.Graph:
    seed = rng.randrange(2**31)
    if kind == "er":
        G = nx.gnp_random_graph(n, rng.uniform(0.0, 1.0), seed=seed)   # Fatemi: p sampled from [0, 1]
    elif kind == "ba":
        G = nx.barabasi_albert_graph(n, rng.randint(1, min(4, n - 1)), seed=seed)
    elif kind == "sfn":
        # scale-free network (Barabasi & Albert 1999); nx returns a MultiDiGraph, simplified below
        G = nx.scale_free_graph(n, seed=seed)
    elif kind == "sbm":
        k = rng.randint(2, min(10, n))                                  # Fatemi: communities from 2 to 10
        sizes = [n // k] * k
        sizes[0] += n - sum(sizes)
        p = [[0.7 if i == j else 0.1 for j in range(k)] for i in range(k)]
        G = nx.stochastic_block_model(sizes, p, seed=seed)
    elif kind == "complete":
        G = nx.complete_graph(n)
    elif kind == "star":
        G = nx.star_graph(n - 1)
    elif kind == "path":
        G = nx.path_graph(n)
    else:
        raise ValueError(f"unknown generator {kind}")
    G = nx.Graph(G)                       # collapses direction and parallel edges (sfn)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.add_nodes_from(range(n))            # sfn/sbm may emit fewer nodes than requested
    return nx.convert_node_labels_to_integers(G)


def sample_query(task: str, G: nx.Graph, rng: random.Random) -> dict | None:
    nodes = sorted(G.nodes)
    edges = sorted(tuple(sorted(e)) for e in G.edges)
    if task in ("node_degree", "connected_nodes", "disconnected_nodes"):
        return {"node": rng.choice(nodes)}
    if task == "edge_existence":
        non_edges = [(u, v) for u in nodes for v in nodes if u < v and not G.has_edge(u, v)]
        pool = edges if (rng.random() < 0.5 and edges) or not non_edges else non_edges
        u, v = rng.choice(pool)
        return {"u": u, "v": v} if rng.random() < 0.5 else {"u": v, "v": u}
    return {}


def ground_truth(task: str, G: nx.Graph, q: dict):
    if task == "edge_existence":
        return G.has_edge(q["u"], q["v"])
    if task == "node_degree":
        return G.degree(q["node"])
    if task == "node_count":
        return G.number_of_nodes()
    if task == "edge_count":
        return G.number_of_edges()
    if task == "connected_nodes":
        return sorted(G.neighbors(q["node"]))
    if task == "disconnected_nodes":
        nbrs = set(G.neighbors(q["node"])) | {q["node"]}
        return sorted(n for n in G.nodes if n not in nbrs)
    if task == "cycle_check":
        return not nx.is_forest(G)
    raise ValueError(f"unknown task {task}")


def build_graphqa(
    n_graphs: int,
    seed: int = 0,
    generators: tuple[str, ...] = GENERATORS,
    tasks: tuple[str, ...] = tuple(TASKS),
) -> list[GraphInstance]:
    rng = random.Random(seed)
    # sample families in Fatemi's proportions rather than cycling them evenly
    weights = [GENERATOR_WEIGHTS.get(g, 1) for g in generators]
    instances: list[GraphInstance] = []
    for i in range(n_graphs):
        kind = rng.choices(generators, weights=weights, k=1)[0]
        n = rng.randint(*NODE_RANGE)
        G = generate_graph(kind, n, rng)
        graph_id = f"graphqa-{kind}-{i:04d}"
        nodes = sorted(G.nodes)
        edges = [list(e) for e in sorted(tuple(sorted(e)) for e in G.edges)]
        for task in tasks:
            q = sample_query(task, G, rng)
            if q is None:
                continue
            instances.append(
                GraphInstance(
                    id=f"{graph_id}-{task}",
                    dataset="graphqa",
                    task=task,
                    directed=False,
                    weighted=False,
                    nodes=nodes,
                    edges=edges,
                    query_args=q,
                    answer_type=TASKS[task],
                    ground_truth=ground_truth(task, G, q),
                    meta={"graph_id": graph_id, "generator": kind, "n": len(nodes), "m": len(edges)},
                )
            )
    return instances


def question(inst: GraphInstance, label_map: dict[int, int] | None = None) -> str:
    args = inst.query_args
    if label_map:
        args = {k: label_map.get(v, v) for k, v in args.items()}
    return QUESTIONS[inst.task].format(**args)
