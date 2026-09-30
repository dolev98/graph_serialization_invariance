"""Repeat-floor statistics (scripts/floor_stats.py, loaded by path)."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gsi.data.base import GraphInstance

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "results" / "review" / "floor"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = load("floor_stats")


def test_midp_known_values():
    # all discordant pairs on one side: mid-p = P(X = 0) = 2 * 0.5 * 2^-m
    assert fs.mcnemar_midp(4, 0) == 0.0625
    assert fs.mcnemar_midp(5, 0) == 0.03125
    assert fs.mcnemar_midp(0, 0) == 1.0
    assert fs.mcnemar_midp(3, 3) == 1.0
    # mid-p is never above the exact conditional p
    for b, c in [(4, 0), (8, 1), (11, 1), (21, 3), (4, 4)]:
        assert fs.mcnemar_midp(b, c) <= fs.mcnemar_exact(b, c)
    assert fs.mcnemar_exact(4, 0) == 0.125


def test_clustered_mcnemar_reduces_to_asymptotic_with_singleton_clusters():
    # one instance per cluster: sum(d)^2 / sum(d^2) = (b - c)^2 / (b + c)
    b, c = 11, 1
    p = fs.mcnemar_clustered([1] * b + [0] * c + [0] * 10, [0] * b + [1] * c + [0] * 10)
    z2 = (b - c) ** 2 / (b + c)
    assert math.isclose(p, math.erfc(math.sqrt(z2 / 2)))
    # the same discordant pairs inside few clusters give a larger p-value
    assert fs.mcnemar_clustered([4, 4, 3], [0, 1, 0]) > p


def test_pairwise_estimator():
    same = ["1", "1", "1", "1"]
    assert fs.pairwise(same, ["1", "1"]) == (0.0, 0.0)
    assert fs.pairwise(same, ["2", "3"]) == (0.0, 1.0)
    # an unparsable answer disagrees with everything, itself included
    within, across = fs.pairwise([None, "1", "1", "1"], ["1"])
    assert within == 0.5 and across == 0.25
    # known value: half of the cross pairs differ, four of the six repeat pairs do
    within, across = fs.pairwise(["1", "2", "1", "2"], ["1", "2", "1", "2"])
    assert across - within == 0.5 - 4 / 6
    # under the flip rule a missing answer is one shared value, so two of them agree
    assert fs.pairwise(["NA", "NA", "1", "1"], ["1"], fs.same_value) == (4 / 6, 0.5)


def test_bh_and_holm():
    p = [0.001, 0.02, 0.03, 0.5]
    q = fs.bh(p)
    assert np.allclose(q, [0.004, 0.04, 0.04, 0.5])
    assert list(fs.holm(p)) == [True, False, False, False]


def _row(ds, iid, graph, repeats, forms, ref_key):
    agree = lambda ks: all(k not in (None, "null") for k in ks) and len(set(ks)) == 1  # noqa: E731
    return {"dataset": ds, "model": "m", "mode": "code", "library": "native", "ref": "canonical",
            "axis": "order", "instance_id": iid, "task": "t", "graph_id": graph, "source_file": graph,
            "repeats": repeats, "ref_key": ref_key,
            "forms": [{"variant_id": f"{iid}::f{i}", "key": k, "same_prompt": False} for i, k in enumerate(forms)],
            "flip": int(not agree(forms + [ref_key])), "floor": int(not agree(repeats))}


def test_load_and_analyse(tmp_path):
    rows = []
    for ds in fs.DATASETS:
        for i in range(20):
            flip = i < 6
            rows.append(_row(ds, f"{ds}-{i}", f"g{i // 2}", ["1"] * 4, ["2" if flip else "1", "1", "1"], "1"))
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    df = fs.load(path)
    assert df.flip.sum() == 12 and df["floor"].sum() == 0
    cells, pooled, modes = fs.analyse(df, reps=500, seed=1)
    assert list(cells.b) == [6, 6] and list(cells.c) == [0, 0]
    assert math.isclose(cells.excess.iloc[0], 0.3)
    # one flipped form of three against four stable repeats: u = 1/3 on flipped instances
    assert math.isclose(cells.u.iloc[0], 6 / 20 * (4 / 12))
    allrow = pooled[pooled.scope == "all"].iloc[0]
    assert allrow.boot_lo > 0 and allrow.cells == 2
    # one model, one arm: per dataset x arm, the model's row and the all-models row are the same numbers
    assert list(modes.model) == ["m", "all", "m", "all"] and list(modes.n) == [20] * 4
    assert modes.excess.tolist() == [0.3] * 4


# A 4-cycle: 0 -> 2 has two shortest paths, [0, 1, 2] and [0, 3, 2].
SQUARE = GraphInstance(id="sq", dataset="erdos", task="shortest_path", directed=False, weighted=False,
                       nodes=[0, 1, 2, 3], edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                       query_args={"source": 0, "target": 2}, answer_type="path", ground_truth=[0, 1, 2])


def test_correctness_uses_compare_on_the_answer_key():
    is_correct = fs.Correctness([SQUARE])
    assert is_correct("sq", "[0, 1, 2]") and is_correct("sq", "[0, 3, 2]")
    assert not is_correct("sq", "[0, 1, 3]") and not is_correct("sq", "null") and not is_correct("sq", None)
    assert fs.flip_value("sq", "[0, 3, 2]", is_correct) == fs.CORRECT
    assert fs.flip_value("sq", "null", is_correct) == fs.NA
    assert fs.flip_value("sq", "[0, 1, 3]", is_correct) == "[0, 1, 3]"


def test_flip_rule_on_a_tiny_case(tmp_path):
    """Literal vs flip rule on one instance each:
    a  two different correct paths among the forms        -> literal flip, no flip-rule flip
    b  the same crash on every repeat                      -> literal floor, no flip-rule floor
    c  a wrong answer and a crash among the forms           -> a flip under both rules
    d  a correct answer and a crash among the repeats       -> a floor under both rules
    e  the same wrong answer throughout                     -> nothing under either rule"""
    ok1, ok2, bad = "[0, 1, 2]", "[0, 3, 2]", "[0, 1, 3]"
    rows = [_row("erdos", "a", "ga", [ok1] * 4, [ok2, ok1, ok1], ok1),
            _row("erdos", "b", "gb", ["null"] * 4, [ok1, ok1, ok1], ok1),
            _row("erdos", "c", "gc", [ok1] * 4, [bad, "null", ok1], ok1),
            _row("erdos", "d", "gd", [ok1, "null", ok1, ok1], [ok1] * 3, ok1),
            _row("erdos", "e", "ge", [bad] * 4, [bad] * 3, bad)]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    literal = fs.load(path).set_index("instance_id")
    assert literal.flip.to_dict() == {"a": 1, "b": 0, "c": 1, "d": 0, "e": 0}
    assert literal["floor"].to_dict() == {"a": 0, "b": 1, "c": 0, "d": 1, "e": 0}
    instances = [GraphInstance(**{**SQUARE.__dict__, "id": i}) for i in "abcde"]
    flip = fs.load(path, "flip", fs.Correctness(instances)).set_index("instance_id")
    assert flip.flip.to_dict() == {"a": 0, "b": 0, "c": 1, "d": 0, "e": 0}
    assert flip["floor"].to_dict() == {"a": 0, "b": 0, "c": 0, "d": 1, "e": 0}
    # the pairwise estimator follows the rule too: the two correct paths no longer disagree
    assert literal.loc["a", "d_across"] == 1 / 3 and flip.loc["a", "d_across"] == 0.0
    assert literal.loc["b", "d_within"] == 1.0 and flip.loc["b", "d_within"] == 0.0


def test_literal_rule_reproduces_noise_floor_json():
    df = pd.concat([fs.load(FLOOR / ds / "noise_floor_instances.jsonl") for ds in fs.DATASETS])
    cells, _, _ = fs.analyse(df, reps=50, seed=0)
    assert len(cells) == 48
    fs.check_against_json(cells)  # asserts n, floor, flip rate, excess and the Wald interval of every cell


def test_committed_literal_cells_match_a_rerun():
    """The committed cell table is what the script computes with its default bootstrap settings."""
    df = pd.concat([fs.load(FLOOR / ds / "noise_floor_instances.jsonl") for ds in fs.DATASETS])
    cells, pooled, _ = fs.analyse(df, reps=10000, seed=20260928)
    committed = pd.read_csv(FLOOR / "floor_stats_literal_cells.csv")
    assert committed[["dataset", "model", "arm", "axis", "n", "b", "c"]].equals(
        cells[["dataset", "model", "arm", "axis", "n", "b", "c"]])
    assert np.allclose(committed[["floor", "flip", "excess", "boot_lo", "boot_hi"]],
                       cells[["floor", "flip", "excess", "boot_lo", "boot_hi"]], atol=1e-5)


def test_committed_flip_rule_table_matches_a_rerun():
    """The flip-rule table per dataset x arm is what the script computes."""
    is_correct = fs.Correctness.from_configs()
    df = pd.concat([fs.load(FLOOR / ds / "noise_floor_instances.jsonl", "flip", is_correct)
                    for ds in fs.DATASETS])
    _, _, modes = fs.analyse(df, reps=10000, seed=20260928)
    committed = pd.read_csv(FLOOR / "floor_stats_flip_modes.csv")
    assert committed[["dataset", "arm", "model", "n", "b", "c"]].equals(modes[["dataset", "arm", "model", "n", "b", "c"]])
    assert np.allclose(committed[["floor", "flip", "excess", "boot_lo", "boot_hi"]],
                       modes[["floor", "flip", "excess", "boot_lo", "boot_hi"]], atol=1e-5)
    pooled = committed[committed.model == "all"].set_index(["dataset", "arm"])
    # four models counting equally: GraphQA M1 floor 5.25 -> flip 12.625, Erdos M1 excess 7.5 points
    assert math.isclose(pooled.loc[("graphqa", "direct/-"), "floor"], 0.0525)
    assert math.isclose(pooled.loc[("graphqa", "direct/-"), "flip"], 0.12625)
    assert round(100 * pooled.loc[("erdos", "direct/-"), "excess"], 1) == 7.5


def test_every_floor_cell_covers_every_sampled_instance():
    """Relabel is judged against the sorted identity where an instance has one and against canonical where it
    does not; either way every sampled instance is in the relabel cell, as in the order cell."""
    for ds in fs.DATASETS:
        rows = [json.loads(line) for line in (FLOOR / ds / "noise_floor_instances.jsonl").open()]
        cells: dict = {}
        for r in rows:
            cells.setdefault((r["model"], r["mode"], r["library"], r["axis"]), set()).add(r["instance_id"])
        for (model, mode, lib, axis), ids in cells.items():
            assert ids == cells[(model, mode, lib, "order")], (ds, model, mode, lib, axis, len(ids))
