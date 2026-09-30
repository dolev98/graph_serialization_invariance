"""Reference solution expressions, one per task, over `G` (networkx) and `q` (query args).

Used only by the stub model, to emit a program that really computes the right answer
rather than printing a literal. Nothing in the prompting path reads this: the models are
never shown a solution. (Previously these lived in `prompts/exemplars.py`, which the
one-shot modes used; the fixed template replaced worked examples.)
"""
from __future__ import annotations

from gsi.data.base import GraphInstance

GRAPHQA_SOLUTIONS = {
    "edge_existence": "G.has_edge(q['u'], q['v'])",
    "node_degree": "G.degree(q['node'])",
    "node_count": "G.number_of_nodes()",
    "edge_count": "G.number_of_edges()",
    "connected_nodes": "sorted(G.neighbors(q['node']))",
    "disconnected_nodes": "sorted(n for n in G.nodes if n != q['node'] and not G.has_edge(q['node'], n))",
    "cycle_check": "not nx.is_forest(G)",
}

ERDOS_SOLUTIONS = {
    "shortest_path": "nx.shortest_path(G, q['u'], q['v'])",
    "weighted_shortest_path": "nx.shortest_path(G, q['u'], q['v'], weight='weight')",
    "triangles": "nx.triangles(G, q['node'])",
    "degree": "G.degree(q['node'])",
    "edge_number": "G.number_of_edges()",
    "node_number": "G.number_of_nodes()",
    "connected_component_number": "nx.number_connected_components(G)",
    "strongly_connected_number": "nx.number_strongly_connected_components(G)",
    "diameter": "float(nx.diameter(G))",
    "radius": "float(nx.radius(G))",
    "density": "nx.density(G)",
    "clustering_coefficient": "nx.clustering(G, q['node'])",
    "jaccard_coefficient": "next(iter(nx.jaccard_coefficient(G, [(q['u'], q['v'])])))[2]",
    "maximum_flow": "float(nx.maximum_flow_value(G, q['u'], q['v'], capacity='weight'))",
    "edge_existence": "G.has_edge(q['u'], q['v'])",
    "has_cycle": "not nx.is_forest(G)",
    "is_bipartite": "nx.is_bipartite(G)",
    "is_regular": "nx.is_regular(G)",
    "local_connectivity": "nx.has_path(G, q['u'], q['v'])",
    "neighbor": "sorted(G.neighbors(q['node']))",
    "common_neighbor": "sorted(nx.common_neighbors(G, q['u'], q['v']))",
    "center": "sorted(nx.center(G))",
    "periphery": "sorted(nx.periphery(G))",
    "bridges": "sorted(tuple(sorted(e)) for e in nx.bridges(G))",
}

SOLUTIONS = {"graphqa": GRAPHQA_SOLUTIONS, "erdos": ERDOS_SOLUTIONS}


def solution_expr(inst: GraphInstance) -> str | None:
    return SOLUTIONS.get(inst.dataset, {}).get(inst.task)
