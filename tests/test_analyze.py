"""Regression coverage for sequential per-model analysis overwriting shared CSVs."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def test_filtered_analysis_preserves_other_models_and_combined_tables(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"
    spec = importlib.util.spec_from_file_location("analyze_script", script)
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    monkeypatch.setattr(analyze, "all_figures", lambda *args: None)

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "name": "audit", "root": str(tmp_path), "datasets": {},
        "modes": ["direct", "code", "graph_as_code"], "models": ["model-a", "model-b"],
    }))
    results = tmp_path / "results" / "audit"
    scored = results / "scored"
    scored.mkdir(parents=True)
    arms = [("direct", None), ("code", "native"), ("graph_as_code", "native")]
    for model in ("model-a", "model-b"):
        rows = []
        for mode, library in arms:
            for axis in ("canonical", "order"):
                correct = model == "model-a" or axis == "canonical"
                rows.append({
                    "dataset": "example", "task": "edge_count", "instance_id": "one",
                    "variant_id": axis, "axis": axis, "params": {}, "model": model,
                    "mode": mode, "library": library, "answer_type": "int", "ground_truth": 1,
                    "parsed": int(correct), "parsed_canonical": int(correct),
                    "answer_key": str(int(correct)), "correct": correct,
                    "failure_class": "ok" if correct else "logic",
                    "declared_ok": True, "graph_match": True,
                    "prompt_hash": f"{model}-{mode}-{axis}",
                })
        (scored / f"{model}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    def run(*models):
        argv = [str(script), "--config", str(config)]
        if models:
            argv += ["--models", *models]
        monkeypatch.setattr(sys, "argv", argv)
        analyze.main()

    def snapshot(folder):
        return {p.name: p.read_bytes() for p in folder.glob("*.csv")}

    tables = results / "tables"
    run()
    combined = snapshot(tables)
    assert set(pd.read_csv(tables / "accuracy.csv")["model"]) == {"model-a", "model-b"}
    run("model-a")
    first = snapshot(tables / "model-a")
    # The perfect model has no failure table; every other export must be present.
    assert first.keys() == combined.keys() - {"failures.csv"}
    run("model-b")
    assert snapshot(tables / "model-a") == first
    assert snapshot(tables) == combined
    second = snapshot(tables / "model-b")
    assert second.keys() == combined.keys()
    for model, accuracy in (("model-a", 1.0), ("model-b", 0.5)):
        frame = pd.read_csv(tables / model / "accuracy.csv")
        assert set(frame["model"]) == {model}
        assert frame["accuracy"].mean() == accuracy

    run("model-a", "model-b")
    assert snapshot(tables / "model-a") == first
    assert snapshot(tables / "model-b") == second
    assert snapshot(tables) == combined


def test_failure_figure_is_the_same_from_the_combined_and_the_per_model_run(tmp_path):
    """Both runs write figs/failures_<dataset>_<model>.png; a class another model has must not change it."""
    from gsi.analysis.figures import failure_stacks
    one = pd.DataFrame({"dataset": ["d", "d"], "mode": ["code", "code"], "library": ["native", "native"],
                        "axis": ["relabel", "order"], "model": ["a", "a"], "logic": [0.6, 1.0],
                        "execution": [0.4, 0.0], "n_failures": [5, 3]})
    both = pd.concat([one.assign(construction=0.0),
                      one.assign(model="b", logic=[0.5, 0.5], execution=[0.0, 0.0], construction=[0.5, 0.5])])
    for d in ("per_model", "combined"):
        (tmp_path / d).mkdir()
    failure_stacks(one, tmp_path / "per_model")
    failure_stacks(both, tmp_path / "combined")
    assert (tmp_path / "per_model" / "failures_d_a.png").read_bytes() == (tmp_path / "combined" / "failures_d_a.png").read_bytes()
