"""A serialization variant is the 4-tuple (label_map, order, structure, syntax).

Every non-canonical variant is produced from a reference by exactly one transform,
each mutating exactly one of the four fields (Herbst et al., 2026). Relabelings
follow Herbst's convention - relabel, then sort lexicographically - so the relabel
axis is compared against a sorted identity reference rather than against positions.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

from gsi.data.base import GraphInstance

SYNTAXES = ("plain", "json", "networkx_code", "graphqa_nl", "erdos_nl")
EDGE_LIST_ORDERINGS = ("sorted_st", "s_sorted_t_shuffled", "t_sorted_s_shuffled", "shuffle_all")
ADJ_LIST_ORDERINGS = ("lines_shuffled", "neighbors_shuffled", "shuffle_all")
IDENTITY = "identity"


@dataclass
class Variant:
    variant_id: str
    instance_id: str
    axis: str
    params: dict[str, Any]
    label_map: dict[int, int]
    node_seq: list[int]
    edge_seq: list[list]
    directed: bool
    weighted: bool
    text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label_map"] = {str(k): v for k, v in self.label_map.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Variant":
        d = dict(d)
        d["label_map"] = {int(k): v for k, v in d["label_map"].items()}
        return cls(**d)

    @property
    def inverse_label_map(self) -> dict[int, int]:
        return {v: k for k, v in self.label_map.items()}


def _rng(*parts: Any) -> random.Random:
    key = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def canonical(inst: GraphInstance, *, structure: str, syntax: str, replicated: bool = False) -> Variant:
    return Variant(
        variant_id=f"{inst.id}::canonical",
        instance_id=inst.id,
        axis="canonical",
        params={"structure": structure, "replicated": replicated, "syntax": syntax, "ordering": None, "seed": None},
        label_map={u: u for u in inst.nodes},
        node_seq=list(inst.nodes),
        edge_seq=[list(e) for e in inst.edges],
        directed=inst.directed,
        weighted=inst.weighted,
    )


def _derive(base: Variant, axis: str, tag: str, **overrides) -> Variant:
    params = dict(base.params)
    params.update(overrides.pop("params", {}))
    return Variant(
        variant_id=f"{base.instance_id}::{axis}:{tag}",
        instance_id=base.instance_id,
        axis=axis,
        params=params,
        label_map=overrides.get("label_map", dict(base.label_map)),
        node_seq=overrides.get("node_seq", list(base.node_seq)),
        edge_seq=overrides.get("edge_seq", [list(e) for e in base.edge_seq]),
        directed=base.directed,
        weighted=base.weighted,
    )


def _sort_in_place(node_seq: list, edge_seq: list) -> None:
    node_seq.sort()
    edge_seq.sort(key=lambda e: (e[0], e[1]))


def relabel_identity(base: Variant) -> Variant:
    """The sorted, identity-labelled reference every relabeling is compared against."""
    node_seq = list(base.node_seq)
    edge_seq = [list(e) for e in base.edge_seq]
    _sort_in_place(node_seq, edge_seq)
    return _derive(base, "relabel", IDENTITY, params={"seed": IDENTITY, "ordering": "sorted_st"},
                   node_seq=node_seq, edge_seq=edge_seq)


def relabel(base: Variant, seed: int, sort: bool = True) -> Variant:
    rng = _rng(base.instance_id, "relabel", seed)
    labels = [base.label_map[u] for u in sorted(base.label_map)]
    perm = list(labels)
    rng.shuffle(perm)
    new_of_old = dict(zip(labels, perm))
    label_map = {c: new_of_old[cur] for c, cur in base.label_map.items()}
    node_seq = [new_of_old[u] for u in base.node_seq]
    edge_seq = [[new_of_old[e[0]], new_of_old[e[1]], *e[2:]] for e in base.edge_seq]
    params: dict[str, Any] = {"seed": seed}
    if sort:
        _sort_in_place(node_seq, edge_seq)
        params["ordering"] = "sorted_st"
    return _derive(base, "relabel", f"s{seed}", params=params,
                   label_map=label_map, node_seq=node_seq, edge_seq=edge_seq)


def reorder(base: Variant, kind: str, seed: int = 0) -> Variant:
    rng = _rng(base.instance_id, "order", kind, seed)
    node_seq = list(base.node_seq)
    edge_seq = [list(e) for e in base.edge_seq]
    structure = base.params["structure"]
    if structure == "edge_list":
        if kind not in EDGE_LIST_ORDERINGS:
            raise ValueError(f"{kind} is not an edge-list ordering")
        if kind == "sorted_st":
            edge_seq.sort(key=lambda e: (e[0], e[1]))
        elif kind == "s_sorted_t_shuffled":
            rng.shuffle(edge_seq)
            edge_seq.sort(key=lambda e: e[0])
        elif kind == "t_sorted_s_shuffled":
            rng.shuffle(edge_seq)
            edge_seq.sort(key=lambda e: e[1])
        else:
            rng.shuffle(edge_seq)
    else:
        if kind not in ADJ_LIST_ORDERINGS:
            raise ValueError(f"{kind} is not an adjacency-list ordering")
        if kind in ("lines_shuffled", "shuffle_all"):
            rng.shuffle(node_seq)
        if kind in ("neighbors_shuffled", "shuffle_all"):
            rng.shuffle(edge_seq)
    return _derive(base, "order", f"{kind}:s{seed}", params={"ordering": kind, "seed": seed},
                   node_seq=node_seq, edge_seq=edge_seq)


def restructure(base: Variant, structure: str | None = None, replicated: bool | None = None) -> Variant:
    params = {}
    tag = []
    if structure is not None:
        params["structure"] = structure
        tag.append(structure)
    if replicated is not None:
        params["replicated"] = replicated
        tag.append("replicated" if replicated else "unreplicated")
    return _derive(base, "structure", "+".join(tag), params=params)


def resyntax(base: Variant, syntax: str) -> Variant:
    return _derive(base, "syntax", syntax, params={"syntax": syntax})


def _canonical_order(v: Variant) -> tuple[list, list]:
    inv = v.inverse_label_map
    return ([inv[u] for u in v.node_seq], [[inv[e[0]], inv[e[1]], *e[2:]] for e in v.edge_seq])


def _order_sig(v: Variant):
    """Order is a *rule* when one is named (e.g. sorted_st, shuffle_all), else the positions."""
    rule = v.params.get("ordering")
    return (rule, None) if rule else (None, _canonical_order(v))


def axes_differing(v: Variant, ref: Variant) -> set[str]:
    diff = set()
    if v.label_map != ref.label_map:
        diff.add("relabel")
    if _order_sig(v) != _order_sig(ref):
        diff.add("order")
    if (v.params["structure"], v.params["replicated"]) != (ref.params["structure"], ref.params["replicated"]):
        diff.add("structure")
    if v.params["syntax"] != ref.params["syntax"]:
        diff.add("syntax")
    return diff


def same_serialization(v: Variant, ref: Variant) -> bool:
    return (v.label_map == ref.label_map and v.node_seq == ref.node_seq and v.edge_seq == ref.edge_seq
            and v.params["structure"] == ref.params["structure"]
            and v.params["replicated"] == ref.params["replicated"]
            and v.params["syntax"] == ref.params["syntax"])


def norm_label(x: Any) -> str:
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(x)


def normalize_edges(edges, directed: bool) -> frozenset[str]:
    out = set()
    for e in edges:
        u, v = norm_label(e[0]), norm_label(e[1])
        if u == v:
            continue
        if not directed and u > v:
            u, v = v, u
        out.add(f"{u}|{v}")
    return frozenset(out)


def true_edge_set(v: Variant) -> frozenset[str]:
    return normalize_edges(v.edge_seq, v.directed)


def make_variants(inst: GraphInstance, canon: Variant, spec: dict) -> list[Variant]:
    """Expand a config block like
    {relabel: {seeds: 3, sort: true}, order: {kinds: [shuffle_all], seeds: 1},
     structure: {toggle: true, replicated: true}, syntax: {kinds: [plain, json]}}
    into canonical + single-axis variants.

    Relabelings (sort=true, the default) are compared against a sorted identity
    reference that is emitted as a member of the relabel axis; every other axis is
    compared against canonical. The single-axis invariant is asserted per reference."""
    out = [canon]
    refs: dict[str, Variant] = {}
    if "relabel" in spec:
        sort = spec["relabel"].get("sort", True)
        base = canon
        if sort:
            base = relabel_identity(canon)
            refs["relabel"] = base
            out.append(base)
        for s in range(spec["relabel"].get("seeds", 1)):
            out.append(relabel(base, s, sort=sort))
    if "order" in spec:
        for kind in spec["order"].get("kinds", []):
            for s in range(spec["order"].get("seeds", 1)):
                out.append(reorder(canon, kind, s))
    if "structure" in spec:
        if spec["structure"].get("toggle"):
            other = "adj_list" if canon.params["structure"] == "edge_list" else "edge_list"
            out.append(restructure(canon, structure=other))
        if spec["structure"].get("replicated") and not canon.directed and canon.params["structure"] == "edge_list":
            out.append(restructure(canon, replicated=True))
    if "syntax" in spec:
        for syn in spec["syntax"].get("kinds", []):
            if syn != canon.params["syntax"]:
                out.append(resyntax(canon, syn))
    kept = [canon]
    for v in out[1:]:
        ref = refs.get(v.axis, canon)
        if v is ref:
            kept.append(v)  # the relabel reference is kept even when it coincides with canonical
            continue
        if same_serialization(v, ref):
            continue  # e.g. sorted_st on an already-sorted canonical: identical text, nothing to measure
        diff = axes_differing(v, ref)
        assert diff == {v.axis}, f"{v.variant_id}: differs in {diff}"
        kept.append(v)
    return kept
