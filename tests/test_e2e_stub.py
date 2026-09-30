import os
from pathlib import Path

import yaml

from gsi.analysis.tables import all_tables, load_scored
from gsi.data.base import read_jsonl
from gsi.experiment.run import main

REPO = Path(__file__).resolve().parent.parent


def write_config(tmp_path: Path) -> Path:
    cfg = {
        "name": "e2e", "root": str(tmp_path),
        "datasets": {"graphqa": {"n_graphs": 4, "seed": 0, "generators": ["er", "ba"],
                                 "tasks": ["node_degree", "cycle_check", "connected_nodes"]}},
        "variants": {"relabel": {"seeds": 1}, "order": {"kinds": ["shuffle_all"]}, "syntax": {"kinds": ["json"]}},
        "modes": ["direct", "code", "graph_as_code"],
        "libraries": ["networkx", "native"],
        "models": ["stub"],
        # Concurrent Python/NetworkX startup on Windows can exceed three seconds.
        # Use the experiment allowance here; test_sandbox covers timeout enforcement.
        "exec": {"timeout": 20 if os.name == "nt" else 3, "workers": 4},
        "paths": {"models": str(REPO / "configs" / "models.yaml")},
    }
    p = tmp_path / "e2e.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_pipeline_runs_and_resumes(tmp_path):
    cfg = write_config(tmp_path)
    main(["--config", str(cfg)])
    results = tmp_path / "results" / "e2e"
    scored = list(read_jsonl(results / "scored" / "stub.jsonl"))
    n_instances = 4 * 3
    n_variants = n_instances * 5  # canonical, relabel identity, relabel s0, shuffle_all, json
    n_arms = 5                    # direct + {code, graph_as_code} x {networkx, native}
    assert len(scored) == n_variants * n_arms
    assert {r["library"] for r in scored} == {None, "networkx", "native"}
    assert all((r["library"] is None) == (r["mode"] == "direct") for r in scored)
    assert {r["failure_class"] for r in scored} >= {"ok", "logic", "format"}
    assert all(r["axis"] in ("canonical", "relabel", "order", "syntax") for r in scored)
    code_rows = [r for r in scored if r["mode"] != "direct" and r["stub_behaviour"] == "correct"]
    assert code_rows and all(r["correct"] for r in code_rows)

    exec_cache = tmp_path / "results" / "cache" / "exec"
    n_cache = len(list(exec_cache.glob("*.json")))
    responses_before = (results / "responses" / "stub.jsonl").read_text()

    main(["--config", str(cfg), "--stages", "llm,exec,score"])
    assert (results / "responses" / "stub.jsonl").read_text() == responses_before
    assert len(list(exec_cache.glob("*.json"))) == n_cache
    assert len(list(read_jsonl(results / "scored" / "stub.jsonl"))) == len(scored)

    tables = all_tables(load_scored(results))
    assert not tables["accuracy"].empty and not tables["invariance"].empty
    assert set(tables["invariance"]["axis"]) == {"relabel", "order", "syntax"}
    assert "gap_code_minus_graph_as_code" in tables["gap"].columns
    # the gap is computed inside a library arm, never across arms
    assert set(tables["gap"]["library"]) == {"networkx", "native"}
    assert set(tables["accuracy"]["library"]) == {"-", "networkx", "native"}
    assert set(tables["ladder"]["rung"]) == {"prose", "json", "injected"}
