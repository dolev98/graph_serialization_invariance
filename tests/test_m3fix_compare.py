"""scripts/m3fix_compare.py (loaded by path): the base/fixed comparison must compare the same
records, must not load models nobody asked for, and must report a delta its columns explain."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m3fix_compare", ROOT / "scripts" / "m3fix_compare.py")
m3fix_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3fix_compare)


def rec(iid, form, axis, key, ok, mode="graph_as_code"):
    return {"dataset": "erdos", "task": "t", "instance_id": iid, "variant_id": f"{iid}::{form}",
            "record_id": f"{iid}::{form}::{mode}::native::m", "axis": axis, "mode": mode, "library": "native",
            "model": "m", "answer_key": key, "parsed": key, "parsed_num": None, "correct": ok,
            "params": {"seed": None}, "prompt_hash": f"{iid}{form}", "ground_truth": 1, "answer_type": "int"}


BASE = pd.DataFrame([rec("A", "canonical", "canonical", "1", True), rec("A", "order:x", "order", "1", True),
                     rec("B", "canonical", "canonical", "1", True), rec("B", "order:x", "order", "2", False)])


def test_an_unfinished_rerun_stops_the_comparison():
    fixed = BASE[BASE["instance_id"] == "A"]                   # B not rerun yet
    with pytest.raises(SystemExit):
        m3fix_compare.match(BASE, fixed, allow_partial=False)


def test_a_partial_comparison_uses_only_shared_records_and_shows_no_false_gain():
    fixed = BASE[BASE["instance_id"] == "A"]
    b, f = m3fix_compare.match(BASE, fixed, allow_partial=True)
    inv = m3fix_compare.invariance_by_axis(b, f, ["m"])
    assert inv["I_base"].tolist() == inv["I_fixed"].tolist() == [1.0]   # unmatched: 0.5 -> 1.0


def test_other_modes_of_the_base_run_are_kept():
    base = pd.concat([BASE, pd.DataFrame([rec("A", "canonical", "canonical", "1", True, mode="code")])])
    b, _ = m3fix_compare.match(base, BASE, allow_partial=False)
    assert (b["mode"] == "code").sum() == 1


def test_no_requested_model_on_disk_is_an_error_not_every_model(tmp_path, monkeypatch):
    (tmp_path / "results" / "run" / "scored").mkdir(parents=True)
    other = rec("A", "canonical", "canonical", "1", True) | {"model": "other"}
    (tmp_path / "results" / "run" / "scored" / "other.jsonl").write_text(json.dumps(other) + "\n")
    monkeypatch.setattr(m3fix_compare, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        m3fix_compare.load("run", ["wanted"])


def test_delta_is_the_difference_of_the_instance_averaged_columns():
    # instance A has one form, B has three: record- and instance-weighted accuracies differ
    rows = []
    for iid, forms, base_ok, fixed_ok in (("A", 1, [False], [True]), ("B", 3, [True, True, False], [True] * 3)):
        for k in range(forms):
            rows.append({"model": "m", "arm": "graph_as_code/native", "instance_id": iid, "directed": False,
                         "correct_base": base_ok[k], "correct_fixed": fixed_ok[k]})
    out = m3fix_compare.accuracy_overall(pd.DataFrame(rows), repeats=200, seed=0).iloc[0]
    assert out["delta"] == pytest.approx(out["acc_fixed_instance"] - out["acc_base_instance"])
    assert out["acc_fixed"] - out["acc_base"] != pytest.approx(out["delta"])


def test_the_rerun_config_rebuilds_the_base_run_variants():
    """variants.jsonl is Git-ignored; the `variants` stage rebuilds it from the tracked instances,
    which is only the base run's set if both configs describe the dataset identically."""
    base = yaml.safe_load((ROOT / "configs" / "erdos.yaml").read_text())
    fixed = yaml.safe_load((ROOT / "configs" / "erdos_m3fix.yaml").read_text())
    assert fixed["datasets"] == base["datasets"]
    assert (ROOT / "data/processed/erdos_m3fix/instances.jsonl").read_bytes() == \
           (ROOT / "data/processed/erdos/instances.jsonl").read_bytes()


def test_a_model_missing_from_one_run_is_set_aside_not_an_error():
    other = BASE.assign(model="n", record_id=BASE["record_id"].str.replace("::m", "::n"))
    b, f = m3fix_compare.match(BASE, pd.concat([BASE, other]), allow_partial=False)
    assert set(b["model"]) == set(f["model"]) == {"m"}


def test_direction_shares_count_only_recovered_graphs():
    """directedness_mismatch is stored False where no graph was recovered; such a record says nothing."""
    row = lambda nb, mb, nf, mf, lib="networkx": {"model": "m", "library": lib, "directed": False,
                                                 "n_graphs_base": nb, "directedness_mismatch_base": mb,
                                                 "n_graphs_fixed": nf, "directedness_mismatch_fixed": mf}
    df = pd.DataFrame([row(1, True, 1, False), row(0, False, 0, False), row(1, True, 1, True, lib="native")])
    d = m3fix_compare.directedness(df).iloc[0]
    assert d["n"] == 2 and d["n_recovered_base"] == 1 and d["n_recovered_fixed"] == 1
    assert d["mismatch_base"] == 1.0 and d["mismatch_fixed"] == 0.0
