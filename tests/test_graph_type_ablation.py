"""scripts/graph_type_ablation.py (loaded by path): the omitted/stated comparison must use the same
canonical records on both sides."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("graph_type_ablation", ROOT / "scripts" / "graph_type_ablation.py")
graph_type_ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_type_ablation)


def rec(iid, ok):
    return {"record_id": f"{iid}::canonical::graph_as_code::native::m", "instance_id": iid, "task": "neighbor",
            "mode": "graph_as_code", "library": "native", "axis": "canonical", "correct": ok,
            "directedness_mismatch": None}


def setup(tmp_path, monkeypatch, base, fixed):
    (tmp_path / "data/processed/erdos").mkdir(parents=True)
    (tmp_path / "data/processed/erdos/instances.jsonl").write_text(
        "".join(json.dumps({"id": i, "directed": False}) + "\n" for i in ("A", "B")))
    for run, rows in (("erdos", base), ("erdos_m3fix", fixed)):
        (tmp_path / f"results/{run}/scored").mkdir(parents=True)
        (tmp_path / f"results/{run}/scored/m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(graph_type_ablation, "ROOT", tmp_path)
    monkeypatch.setattr(graph_type_ablation, "MODELS", ["m"])


def test_an_unfinished_rerun_stops_the_task_comparison(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, [rec("A", True), rec("B", False)], [rec("A", True)])
    with pytest.raises(SystemExit):
        graph_type_ablation.task_sensitivity()


def test_matched_runs_compare(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, [rec("A", False), rec("B", False)], [rec("A", True), rec("B", True)])
    t = graph_type_ablation.task_sensitivity()
    assert t["delta"].tolist() == [1.0] and t["base_source"].tolist() == ["records"]


def test_the_table_fallback_refuses_an_incomplete_rerun(tmp_path, monkeypatch):
    """No base records: the base side comes from the committed per-model table (2 instances, 50%).
    A rerun holding only the instance that was already right must not read as a 50-point gain."""
    setup(tmp_path, monkeypatch, [], [rec("A", True)])
    (tmp_path / "results/erdos/scored/m.jsonl").unlink()
    (tmp_path / "results/erdos/tables/m").mkdir(parents=True)
    (tmp_path / "results/erdos/tables/m/accuracy.csv").write_text(
        "mode,axis,task,library,accuracy,n\ngraph_as_code,canonical,neighbor,native,0.5,2\n")
    with pytest.raises(SystemExit):
        graph_type_ablation.task_sensitivity()


def test_the_table_fallback_accepts_a_complete_rerun(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, [], [rec("A", True), rec("B", True)])
    (tmp_path / "results/erdos/scored/m.jsonl").unlink()
    (tmp_path / "results/erdos/tables/m").mkdir(parents=True)
    (tmp_path / "results/erdos/tables/m/accuracy.csv").write_text(
        "mode,axis,task,library,accuracy,n\ngraph_as_code,canonical,neighbor,native,0.5,2\n")
    t = graph_type_ablation.task_sensitivity()
    assert t["delta"].tolist() == [0.5] and t["base_source"].tolist() == ["tables"]


def test_no_recovered_graph_is_not_an_observed_direction(tmp_path, monkeypatch):
    rows = [rec("A", True) | {"library": "networkx", "record_id": "A::nx", "n_graphs": 0, "directedness_mismatch": False},
            rec("B", True) | {"library": "networkx", "record_id": "B::nx", "n_graphs": 1, "directedness_mismatch": True},
            rec("A", True) | {"n_graphs": 0, "directedness_mismatch": False}]           # native arm
    setup(tmp_path, monkeypatch, rows, rows)
    # A's unrecovered graph is stored as "no mismatch"; counting it would halve the share to 0.5
    lat = graph_type_ablation.latent_misreading().iloc[0]
    assert lat["n_recovered"] == 1 and lat["built_digraph"] == 1.0
    rec_ = graph_type_ablation.recovery().set_index("library")
    assert rec_.loc["networkx", "n_recovered"] == 1 and rec_.loc["networkx", "built_digraph_after"] == 1.0
    assert pd.isna(rec_.loc["native", "built_digraph_after"]) and rec_.loc["native", "n_recovered"] == 0
