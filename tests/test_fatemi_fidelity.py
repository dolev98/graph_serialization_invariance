"""Pins our GraphQA encoders and generator to Fatemi et al. (ICLR 2024).

The encoder strings are quoted from Appendix A.1 for the Figure 5 running example, except
that the incident encoder lists each node's neighbours in ascending order (A.1 prints nodes
6 and 7 as "7, 5" and "2, 8, 6"); the generator parameters are from Appendix A.4. If these drift, our claim that
GraphQA numbers are comparable to theirs stops holding.
"""
from pathlib import Path

import networkx as nx
import pytest
import yaml

from gsi.data.base import GraphInstance
from gsi.data.graphqa import GENERATOR_WEIGHTS, GENERATORS, NODE_RANGE, TASKS, build_graphqa, generate_graph
from gsi.serial.render import render
from gsi.serial.variant import canonical, restructure

REPO = Path(__file__).resolve().parents[1]

# Figure 5 running example
FIG5 = GraphInstance(id="fig5", dataset="graphqa", task="node_degree", directed=False, weighted=False,
                     nodes=list(range(9)),
                     edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [2, 5], [2, 7], [3, 8], [5, 6], [6, 7], [7, 8]],
                     query_args={"node": 2}, answer_type="int", ground_truth=6, meta={"graph_id": "fig5"})

ADJACENCY = ("In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. "
             "G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, 7, and 8.\n"
             "The edges in G are: (0, 1) (0, 2) (1, 2) (2, 3) (2, 4) (2, 5) (2, 7) (3, 8) (5, 6) (6, 7) (7, 8).")

INCIDENT = """G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
In this graph:
Node 0 is connected to nodes 1, 2.
Node 1 is connected to nodes 0, 2.
Node 2 is connected to nodes 0, 1, 3, 4, 5, 7.
Node 3 is connected to nodes 2, 8.
Node 4 is connected to node 2.
Node 5 is connected to nodes 2, 6.
Node 6 is connected to nodes 5, 7.
Node 7 is connected to nodes 2, 6, 8.
Node 8 is connected to nodes 3, 7."""


def test_adjacency_encoder_matches_fatemi_verbatim():
    v = canonical(FIG5, structure="edge_list", syntax="graphqa_nl")
    assert render(v) == ADJACENCY


def test_incident_encoder_matches_fatemi_verbatim():
    v = restructure(canonical(FIG5, structure="edge_list", syntax="graphqa_nl"), structure="adj_list")
    assert render(v) == INCIDENT


def test_incident_uses_singular_noun_for_a_single_neighbour():
    v = restructure(canonical(FIG5, structure="edge_list", syntax="graphqa_nl"), structure="adj_list")
    text = render(v)
    assert "Node 4 is connected to node 2." in text      # one neighbour -> singular
    assert "Node 0 is connected to nodes 1, 2." in text  # two -> plural


def test_generator_families_and_weights_follow_appendix_a4():
    assert set(GENERATORS) == {"er", "ba", "sfn", "sbm", "star", "path", "complete"}
    assert [GENERATOR_WEIGHTS[g] for g in ("er", "ba", "sfn", "sbm")] == [500] * 4
    assert [GENERATOR_WEIGHTS[g] for g in ("star", "path", "complete")] == [100] * 3
    assert NODE_RANGE == (5, 20)


def test_fatemi_task_set():
    fatemi = ["edge_existence", "node_degree", "node_count", "edge_count", "connected_nodes", "disconnected_nodes",
              "cycle_check"]
    assert list(TASKS) == fatemi
    assert yaml.safe_load((REPO / "configs" / "graphqa.yaml").read_text())["datasets"]["graphqa"]["tasks"] == fatemi


@pytest.mark.parametrize("kind", GENERATORS)
def test_every_family_generates_a_usable_simple_graph(kind):
    import random
    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(*NODE_RANGE)
        G = generate_graph(kind, n, rng)
        assert G.number_of_nodes() == n, kind
        assert not isinstance(G, (nx.MultiGraph, nx.DiGraph))
        assert not list(nx.selfloop_edges(G))


def test_sampling_hits_the_rare_families_in_roughly_the_right_proportion():
    insts = build_graphqa(400, seed=0, tasks=("node_degree",))
    fams = {}
    for i in insts:
        fams[i.meta["generator"]] = fams.get(i.meta["generator"], 0) + 1
    assert set(fams) == set(GENERATORS)          # all seven appear
    common = sum(fams[g] for g in ("er", "ba", "sfn", "sbm"))
    rare = sum(fams[g] for g in ("star", "path", "complete"))
    assert common > 3 * rare                      # 2000:300 in the paper
