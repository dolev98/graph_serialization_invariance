import pytest

from gsi.data.graphqa import build_graphqa
from gsi.serial.render import render
from gsi.serial.variant import (ADJ_LIST_ORDERINGS, EDGE_LIST_ORDERINGS, IDENTITY, SYNTAXES, axes_differing,
                                canonical, make_variants, normalize_edges, relabel, relabel_identity, reorder, restructure,
                                same_serialization, true_edge_set)

from graph_text import parse, to_nx

INSTANCES = build_graphqa(6, seed=3, generators=("er", "ba", "star", "path", "complete", "sbm"), tasks=("node_degree",))

CANONS = [
    {"structure": "edge_list", "syntax": "graphqa_nl"},
    {"structure": "adj_list", "syntax": "graphqa_nl"},
    {"structure": "edge_list", "syntax": "plain"},
]


def full_spec(structure: str) -> dict:
    kinds = list(ADJ_LIST_ORDERINGS if structure == "adj_list" else EDGE_LIST_ORDERINGS)
    return {"relabel": {"seeds": 3}, "order": {"kinds": kinds, "seeds": 2},
            "structure": {"toggle": True, "replicated": True}, "syntax": {"kinds": list(SYNTAXES)}}


def reference_for(v, canon, variants):
    if v.axis == "relabel":
        return next(x for x in variants if x.params.get("seed") == IDENTITY)
    return canon


@pytest.mark.parametrize("inst", INSTANCES, ids=[i.id for i in INSTANCES])
@pytest.mark.parametrize("canon_cfg", CANONS, ids=["edge/graphqa", "adj/graphqa", "edge/plain"])
def test_round_trip_and_single_axis(inst, canon_cfg):
    canon = canonical(inst, **canon_cfg)
    variants = make_variants(inst, canon, full_spec(canon_cfg["structure"]))
    assert len(variants) > 8
    truth_canon = true_edge_set(canon)
    for v in variants:
        text = render(v)
        assert parse(text, v.params["syntax"], v.params["structure"], v.directed) == true_edge_set(v), v.variant_id
        assert set(to_nx(v).nodes) == set(v.node_seq)
        assert normalize_edges(to_nx(v).edges, v.directed) == true_edge_set(v)
        if v.axis == "canonical" or v.params.get("seed") == IDENTITY:
            continue
        ref = reference_for(v, canon, variants)
        assert axes_differing(v, ref) == {v.axis}, v.variant_id
        inv = v.inverse_label_map
        back = normalize_edges([[inv[e[0]], inv[e[1]]] for e in v.edge_seq], v.directed)
        assert back == truth_canon


def test_relabel_is_a_permutation_and_deterministic():
    inst = INSTANCES[0]
    canon = canonical(inst, structure="edge_list", syntax="plain")
    a, b = relabel(canon, 0), relabel(canon, 0)
    assert a.label_map == b.label_map and a.edge_seq == b.edge_seq
    assert sorted(a.label_map.values()) == sorted(inst.nodes)
    assert relabel(canon, 1).label_map != a.label_map
    assert a.label_map != canon.label_map


def test_sorted_relabel_follows_herbst():
    inst = INSTANCES[1]
    canon = canonical(inst, structure="edge_list", syntax="plain")
    ident = relabel_identity(canon)
    v = relabel(ident, 0)
    assert v.node_seq == sorted(v.node_seq)
    assert v.edge_seq == sorted(v.edge_seq, key=lambda e: (e[0], e[1]))
    assert v.params["ordering"] == "sorted_st" and ident.params["seed"] == IDENTITY
    assert axes_differing(v, ident) == {"relabel"}


def test_positional_relabel_keeps_edge_positions():
    inst = INSTANCES[0]
    canon = canonical(inst, structure="edge_list", syntax="plain")
    v = relabel(canon, 0, sort=False)
    for e_new, e_old in zip(v.edge_seq, canon.edge_seq):
        assert e_new[0] == v.label_map[e_old[0]] and e_new[1] == v.label_map[e_old[1]]
    assert axes_differing(v, canon) == {"relabel"}


def test_make_variants_emits_identity_reference_even_when_canonical_is_sorted():
    inst = INSTANCES[2]
    canon = canonical(inst, structure="edge_list", syntax="graphqa_nl")  # graphqa canonical is already sorted
    vs = make_variants(inst, canon, {"relabel": {"seeds": 2}})
    ident = [v for v in vs if v.params.get("seed") == IDENTITY]
    assert len(ident) == 1 and ident[0].axis == "relabel"
    assert same_serialization(ident[0], canon)
    assert sum(v.axis == "relabel" for v in vs) == 3


def test_no_op_sorted_st_is_dropped_on_sorted_canonical():
    inst = INSTANCES[2]
    canon = canonical(inst, structure="edge_list", syntax="graphqa_nl")
    vs = make_variants(inst, canon, {"order": {"kinds": ["sorted_st", "shuffle_all"]}})
    assert [v.params["ordering"] for v in vs if v.axis == "order"] == ["shuffle_all"]


def test_edge_list_orderings():
    inst = INSTANCES[1]
    canon = canonical(inst, structure="edge_list", syntax="plain")
    v = reorder(canon, "sorted_st")
    assert v.edge_seq == sorted(v.edge_seq)
    v = reorder(canon, "s_sorted_t_shuffled")
    assert [e[0] for e in v.edge_seq] == sorted(e[0] for e in v.edge_seq)
    v = reorder(canon, "t_sorted_s_shuffled")
    assert [e[1] for e in v.edge_seq] == sorted(e[1] for e in v.edge_seq)
    v = reorder(canon, "shuffle_all", seed=0)
    assert sorted(map(tuple, v.edge_seq)) == sorted(map(tuple, canon.edge_seq))
    assert v.edge_seq != canon.edge_seq
    with pytest.raises(ValueError):
        reorder(canon, "lines_shuffled")


def test_adj_list_orderings_change_rendering_only():
    inst = INSTANCES[1]
    canon = canonical(inst, structure="adj_list", syntax="graphqa_nl")
    for kind in ADJ_LIST_ORDERINGS:
        v = reorder(canon, kind)
        assert render(v) != render(canon)
        assert true_edge_set(v) == true_edge_set(canon)


def test_replicated_edge_list_lists_both_directions():
    inst = INSTANCES[1]
    canon = canonical(inst, structure="edge_list", syntax="plain")
    v = make_variants(inst, canon, {"structure": {"replicated": True}})[1]
    assert v.params["replicated"] and v.axis == "structure"
    assert render(v).count("(") == 2 * len(canon.edge_seq)


def test_adjacency_list_round_trip_with_single_neighbour_lines():
    star = build_graphqa(1, seed=0, generators=("star",), tasks=("node_degree",))[0]
    v = restructure(canonical(star, structure="edge_list", syntax="graphqa_nl"), structure="adj_list")
    text = render(v)
    assert "is connected to node 0." in text  # a leaf has one neighbour
    assert parse(text, "graphqa_nl", "adj_list", False) == true_edge_set(v)
