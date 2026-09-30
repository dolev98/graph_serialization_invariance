"""Re-running the pipeline: committed data is kept, stale replies are re-asked, nothing is silently lost."""
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gsi.data.base import read_jsonl, trim_partial_line
from gsi.experiment import run
from gsi.experiment.config import load_config
from gsi.experiment.run import load_dotenv, main

REPO = Path(__file__).resolve().parent.parent


def write_config(tmp_path: Path, **extra) -> Path:
    cfg = {
        "name": "pipe", "root": str(tmp_path),
        "datasets": {"graphqa": {"n_graphs": 2, "seed": 0, "generators": ["er"], "tasks": ["node_degree"]}},
        "variants": {"relabel": {"seeds": 1}},
        "modes": ["direct", "graph_as_code"], "libraries": ["native"], "models": ["stub"],
        "exec": {"timeout": 20, "workers": 2},
        "paths": {"models": str(REPO / "configs" / "models.yaml")},
        **extra,
    }
    p = tmp_path / "pipe.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_data_stage_keeps_existing_instances(tmp_path):
    cfg = write_config(tmp_path)
    main(["--config", str(cfg), "--stages", "data,variants"])
    path = tmp_path / "data" / "processed" / "pipe" / "instances.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["ground_truth"] = 999  # stands in for data that a rebuild would not reproduce
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pinned = path.read_text()
    main(["--config", str(cfg), "--stages", "data,variants"])
    main(["--config", str(cfg), "--estimate"])
    assert path.read_text() == pinned
    main(["--config", str(cfg), "--stages", "data", "--rebuild-data"])
    assert path.read_text() != pinned


def test_bad_arguments_are_rejected(tmp_path):
    cfg = write_config(tmp_path)
    for argv in (["--stages", "variants,lmm"], ["--models"], ["--models", "no-such-model"]):
        with pytest.raises(SystemExit):
            main(["--config", str(cfg), *argv])
    with pytest.raises(ValueError, match="inject_graph_types"):
        load_config(write_config(tmp_path, inject_graph_types=True))


def test_changed_prompts_are_asked_again(tmp_path):
    cfg = write_config(tmp_path)
    main(["--config", str(cfg)])
    responses = tmp_path / "results" / "pipe" / "responses" / "stub.jsonl"
    before = {r["record_id"]: r["prompt_hash"] for r in read_jsonl(responses)}
    # stating the graph type changes every undirected M3 prompt and nothing else
    main(["--config", str(write_config(tmp_path, inject_graph_type=True)), "--stages", "llm,exec,score"])
    after = {r["record_id"]: r["prompt_hash"] for r in read_jsonl(responses)}
    assert after.keys() == before.keys()
    changed = {k for k in after if after[k] != before[k]}
    assert changed and all("::graph_as_code::" in k for k in changed)
    scored = {r["record_id"]: r["prompt_hash"] for r in read_jsonl(tmp_path / "results" / "pipe" / "scored" / "stub.jsonl")}
    assert scored == after
    # the changed records' programs ran again: their executions were dropped and appended anew
    execs = [r["record_id"] for r in read_jsonl(tmp_path / "results" / "pipe" / "exec" / "stub.jsonl")]
    assert set(execs[-len(changed):]) == changed and len(execs) == len(set(execs))


def test_a_changed_graph_is_executed_again_under_the_same_prompt(tmp_path):
    cfg = write_config(tmp_path)
    main(["--config", str(cfg)])
    variants_path = tmp_path / "data" / "processed" / "pipe" / "variants.jsonl"
    rows = list(read_jsonl(variants_path))
    target = next(r for r in rows if r["axis"] == "relabel" and len(r["edge_seq"]) > 1)
    target["edge_seq"] = target["edge_seq"][::-1]  # same graph text for M3 (none), different data
    variants_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    responses = (tmp_path / "results" / "pipe" / "responses" / "stub.jsonl").read_text()
    main(["--config", str(cfg), "--stages", "llm,exec,score"])
    assert (tmp_path / "results" / "pipe" / "responses" / "stub.jsonl").read_text() == responses  # no new call
    execs = [r["record_id"] for r in read_jsonl(tmp_path / "results" / "pipe" / "exec" / "stub.jsonl")]
    assert execs[-1].startswith(target["variant_id"] + "::graph_as_code::")


def test_score_stage_does_not_wipe_restored_records(tmp_path):
    cfg = write_config(tmp_path)
    main(["--config", str(cfg), "--stages", "data,variants"])
    scored = tmp_path / "results" / "pipe" / "scored" / "stub.jsonl"
    scored.parent.mkdir(parents=True)
    scored.write_text('{"record_id": "restored"}\n')
    with pytest.raises(SystemExit):
        main(["--config", str(cfg), "--stages", "score"])
    assert scored.read_text() == '{"record_id": "restored"}\n'


class FlakyClient:
    """Answers the first `ok` calls, then fails every call."""

    def __init__(self, ok):
        self.ok, self.calls = ok, 0

    def complete(self, prompt, context=None):
        time.sleep(0.005)  # a real call takes time; an instant one would let the workers outrun the checks
        self.calls += 1
        if self.calls > self.ok:
            raise RuntimeError("402 payment required")
        return run.StubLLM(run.load_models(REPO / "configs" / "models.yaml")["stub"]).complete(prompt, context=context)


def test_llm_stage_aborts_on_a_recent_run_of_errors(tmp_path, monkeypatch):
    cfg_path = write_config(tmp_path, datasets={"graphqa": {"n_graphs": 80, "seed": 0, "generators": ["er"],
                                                            "tasks": ["node_degree"]}}, modes=["direct"])
    main(["--config", str(cfg_path), "--stages", "data,variants"])
    cfg = load_config(cfg_path)
    spec = replace(run.load_models(cfg.models_file)["stub"], max_concurrency=2)
    client = FlakyClient(ok=100)
    monkeypatch.setattr(run, "make_client", lambda spec, cache_dir: client)
    with pytest.raises(RuntimeError, match="aborting"):
        run.stage_llm(cfg, spec, run.load_instances(cfg), run.load_variants(cfg))
    assert client.calls - 100 < 60  # stops soon after the failures start, not after as many as succeeded


def test_load_dotenv_strips_inline_comments(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("A_KEY=abc   # a comment\nB_KEY='x # y'\nC_KEY=   # only a comment\nD_KEY=#abc\n")
    for k in ("A_KEY", "B_KEY", "C_KEY", "D_KEY"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(tmp_path)
    import os
    assert [os.environ[k] for k in ("A_KEY", "B_KEY", "C_KEY", "D_KEY")] == ["abc", "x # y", "", "#abc"]


def test_a_truncated_last_line_does_not_block_resuming(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"record_id": "a"}\n{"record_id": "b"}\n{"record_id": "c", "raw')
    with pytest.raises(json.JSONDecodeError):  # readers stay strict; only a writer repairs the file
        list(read_jsonl(p))
    run.Sink(p).write({"record_id": "c"})
    assert [r["record_id"] for r in read_jsonl(p)] == ["a", "b", "c"]
    p.write_text('{"record_id": "a"}\n{"record_id": "b"}')  # complete, only the newline is missing
    assert not trim_partial_line(p)
    run.Sink(p).write({"record_id": "c"})
    assert [r["record_id"] for r in read_jsonl(p)] == ["a", "b", "c"]
    p.write_text('{"record_id": "a"}\n{"broken\n{"record_id": "c"}\n')
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(p))


class FailsOnce(FlakyClient):
    """Fails the first call only."""

    def complete(self, prompt, context=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("503 service unavailable")
        return run.StubLLM(run.load_models(REPO / "configs" / "models.yaml")["stub"]).complete(prompt, context=context)


def test_a_failed_shared_prompt_skips_its_records_without_aborting(tmp_path, monkeypatch):
    # node_count names no node, so every Graph-as-Code record of the task shares one prompt
    cfg_path = write_config(tmp_path, datasets={"graphqa": {"n_graphs": 30, "seed": 0, "generators": ["er"],
                                                            "tasks": ["node_count"]}}, modes=["graph_as_code"])
    main(["--config", str(cfg_path), "--stages", "data,variants"])
    cfg = load_config(cfg_path)
    spec = replace(run.load_models(cfg.models_file)["stub"], max_concurrency=1)
    monkeypatch.setattr(run, "make_client", lambda spec, cache_dir: FailsOnce(ok=0))
    missing = run.stage_llm(cfg, spec, run.load_instances(cfg), run.load_variants(cfg))
    assert missing == 90  # one failed call, 89 records skipped rather than 89 errors and an abort
    monkeypatch.setattr(run, "make_client", lambda spec, cache_dir: FlakyClient(ok=10**6))
    assert run.stage_llm(cfg, spec, run.load_instances(cfg), run.load_variants(cfg)) == 0
