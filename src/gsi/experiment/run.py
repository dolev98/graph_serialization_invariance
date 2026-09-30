"""Stages: data -> variants -> llm -> exec -> score. Every stage can be re-run and resumes where it stopped:
LLM responses are cached by prompt hash, executions by code hash, and each JSONL sink skips the record_ids it
already holds. A stored response whose prompt has changed since is dropped and asked again.

The data stage writes data/processed/<config>/instances.jsonl only when it does not exist: the committed
instances pin the sampled graphs (GraphQA's SBM generator draws different graphs under networkx 3.7).
--rebuild-data regenerates them on purpose. An execution is redone when its program or the graph it runs on
has changed since."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gsi.data.base import GraphInstance, append_jsonl, read_jsonl, trim_partial_line, write_jsonl
from gsi.exec import sandbox
from gsi.exec.sandbox import atomic_write_text
from gsi.experiment.config import ExperimentConfig, load_config, load_models
from gsi.llm.client import LLMClient, ModelSpec
from gsi.llm.stub import StubLLM
from gsi.prompts.modes import CODE_MODES, build_prompt
from gsi.score.classify import classify
from gsi.score.compare import compare, normalize_answer, relabel_answer
from gsi.score.parse_answer import find_boxed_answer, parse_value
from gsi.serial.render import render
from gsi.serial.variant import Variant, canonical, make_variants

STAGES = ("data", "variants", "llm", "exec", "score")
# Abort the llm stage once at least ERROR_ABORT_MIN calls have failed and at least ERROR_ABORT_FRACTION of the
# last ERROR_WINDOW completions failed.
ERROR_ABORT_MIN, ERROR_ABORT_FRACTION, ERROR_WINDOW = 25, 0.5, 50


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_dotenv(root: Path) -> None:
    p = root / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, raw = line.split("=", 1)
        v = raw.strip()
        if v[:1] in ("'", '"'):
            v = v[1:v.index(v[0], 1)] if v[0] in v[1:] else v[1:]
        else:
            # as in a shell, "#" starts a comment only after whitespace: KEY=abc  # note -> "abc", KEY=#abc -> "#abc"
            v = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        os.environ.setdefault(k.strip(), v)


def arm_label(mode: str, library: str | None) -> str:
    return mode if library is None else f"{mode}/{library}"


def iter_arms(cfg: ExperimentConfig, inst: GraphInstance):
    """(mode, library) pairs to run for one instance. `library` is crossed with the code
    modes rather than baked into the mode name; `direct` yields library None."""
    for mode in cfg.modes:
        for library in cfg.arms_for(mode, inst.task):
            yield mode, library


def estimate(cfg: ExperimentConfig, instances, variants, model_names: list[str]) -> None:
    """Prompt counts and rough token volume (chars/4) per arm; M3 prompts dedupe by hash."""
    by_arm: dict[str, dict] = {}
    for inst in instances:
        for v in variants[inst.id]:
            for mode, library in iter_arms(cfg, inst):
                p = build_prompt(mode, inst, v, library, inject_graph_type=cfg.inject_graph_type)
                d = by_arm.setdefault(arm_label(mode, library), {"prompts": 0, "unique": set(), "chars": 0})
                d["prompts"] += 1
                if p.prompt_hash not in d["unique"]:
                    d["unique"].add(p.prompt_hash)
                    d["chars"] += len(p.system) + len(p.user)
    total_unique = sum(len(d["unique"]) for d in by_arm.values())
    total_tokens = sum(d["chars"] for d in by_arm.values()) / 4
    print(f"config {cfg.name}: {len(instances)} instances, {sum(len(v) for v in variants.values())} variants")
    for arm, d in by_arm.items():
        print(f"  {arm:24s} {d['prompts']:7d} prompts, {len(d['unique']):7d} unique, ~{d['chars'] / 4 / 1e6:.2f}M input tokens")
    print(f"  per model: {total_unique} API calls, ~{total_tokens / 1e6:.2f}M input tokens (+ outputs); "
          f"x {len(model_names)} models = {total_unique * len(model_names)} calls")


# ---------------- data & variants ----------------

def build_instances(cfg: ExperimentConfig) -> list[GraphInstance]:
    out: list[GraphInstance] = []
    for name, ds in cfg.datasets.items():
        if name == "graphqa":
            from gsi.data.graphqa import GENERATORS, TASKS, build_graphqa
            out += build_graphqa(ds["n_graphs"], seed=ds.get("seed", 0),
                                 generators=tuple(ds.get("generators", GENERATORS)),
                                 tasks=tuple(ds.get("tasks", TASKS)))
        elif name == "erdos":
            from gsi.data.erdos import build_erdos
            out += build_erdos(**{k: v for k, v in ds.items() if k not in ("canonical", "variants")})
        else:
            raise ValueError(f"unknown dataset {name}")
    write_jsonl(cfg.data_dir / "instances.jsonl", (i.to_dict() for i in out))
    return out


def instances_for(cfg: ExperimentConfig, rebuild: bool) -> list[GraphInstance]:
    """The data stage: the committed instances when they exist, built (and written) otherwise."""
    path = cfg.data_dir / "instances.jsonl"
    if path.exists() and not rebuild:
        instances = load_instances(cfg)
        log(f"[data] {len(instances)} instances from {path} (kept; --rebuild-data regenerates them)")
        return instances
    instances = build_instances(cfg)
    log(f"[data] {len(instances)} instances built and written to {path}")
    return instances


def build_variants(cfg: ExperimentConfig, instances: list[GraphInstance]) -> dict[str, list[Variant]]:
    out: dict[str, list[Variant]] = {}
    rows = []
    for inst in instances:
        canon = canonical(inst, **cfg.canonical_for(inst.dataset))
        vs = make_variants(inst, canon, cfg.variants_for(inst.dataset))
        for v in vs:
            v.text = render(v)
            rows.append(v.to_dict())
        out[inst.id] = vs
    write_jsonl(cfg.data_dir / "variants.jsonl", rows)
    return out


def load_instances(cfg: ExperimentConfig) -> list[GraphInstance]:
    return [GraphInstance.from_dict(d) for d in read_jsonl(cfg.data_dir / "instances.jsonl")]


def load_variants(cfg: ExperimentConfig) -> dict[str, list[Variant]]:
    out: dict[str, list[Variant]] = {}
    for d in read_jsonl(cfg.data_dir / "variants.jsonl"):
        v = Variant.from_dict(d)
        out.setdefault(v.instance_id, []).append(v)
    return out


# ---------------- sinks ----------------

class Sink:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        trim_partial_line(path)
        self.ids = {r["record_id"] for r in read_jsonl(path)}

    def has(self, record_id: str) -> bool:
        return record_id in self.ids

    def write(self, row: dict) -> None:
        with self.lock:
            if row["record_id"] in self.ids:
                return
            append_jsonl(self.path, row)
            self.ids.add(row["record_id"])


def make_client(spec: ModelSpec, cache_dir: Path):
    return StubLLM(spec) if spec.provider == "stub" else LLMClient(spec, cache_dir)


# ---------------- llm stage ----------------

class Skipped(Exception):
    """A record whose prompt already failed in this run."""


def drop_records(path: Path, record_ids: set[str]) -> int:
    """Remove the rows with these record_ids from a JSONL sink; returns how many were removed."""
    if not path.exists() or not record_ids:
        return 0
    rows = list(read_jsonl(path))
    kept = [r for r in rows if r["record_id"] not in record_ids]
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in kept))
    return len(rows) - len(kept)


def stage_llm(cfg: ExperimentConfig, spec: ModelSpec, instances, variants, limit: int | None = None) -> int:
    """Ask every prompt not answered yet; returns the number of records left unanswered."""
    client = make_client(spec, cfg.cache_dir)
    out = cfg.results_dir / "responses" / f"{spec.name}.jsonl"
    trim_partial_line(out)  # a killed run can leave half a row
    stored = {r["record_id"]: r.get("prompt_hash") for r in read_jsonl(out)}
    todo, stale = [], set()
    for inst in instances:
        for v in variants[inst.id]:
            for mode, library in iter_arms(cfg, inst):
                p = build_prompt(mode, inst, v, library, inject_graph_type=cfg.inject_graph_type)
                rid = f"{p.prompt_id}::{spec.name}"
                if rid in stored and stored[rid] != p.prompt_hash:
                    stale.add(rid)
                todo.append((rid, p, inst, v))
    if stale:
        # The record_id names the instance, form, arm and model, not the prompt text, so a prompt that
        # changed since (new instances, a template edit) must not be answered by the old reply.
        # (their executions are redone by the exec stage, whose signature covers the program)
        log(f"[llm:{spec.name}] {drop_records(out, stale)} stored responses answer an older prompt; asking again")
    sink = Sink(out)
    jobs = [j for j in todo if not sink.has(j[0])]
    if limit:
        jobs = jobs[:limit]
    log(f"[llm:{spec.name}] {len(jobs)} prompts to run ({len(sink.ids)} already done)")
    errors = skipped = 0
    # A prompt shared by many records (up to ~1,300 for one Graph-as-Code prompt) that has just failed is not
    # sent again for each of them in this run: those records are skipped, not counted as errors.
    failed: set[str] = set()

    def work(job):
        rid, p, inst, v = job
        if p.prompt_hash in failed:
            raise Skipped(p.prompt_hash)
        try:
            r = client.complete(p, context={"inst": inst, "variant": v})
        except Exception:
            failed.add(p.prompt_hash)
            raise
        failed.discard(p.prompt_hash)  # a later attempt of the same prompt succeeded
        row = {"record_id": rid, "instance_id": inst.id, "variant_id": v.variant_id, "mode": p.mode,
               "library": p.library, **r.to_dict()}
        sink.write(row)
        return r.cached

    with ThreadPoolExecutor(max_workers=max(1, spec.max_concurrency)) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        done = cached = 0
        recent: deque[bool] = deque(maxlen=ERROR_WINDOW)
        for fut in as_completed(futs):
            try:
                cached += bool(fut.result())
                recent.append(False)
            except Skipped:
                skipped += 1
            except Exception as e:
                errors += 1
                recent.append(True)
                log(f"[llm:{spec.name}] error on {futs[fut][0]}: {e!r}")
            done += 1
            if done % 50 == 0 or done == len(jobs):
                log(f"[llm:{spec.name}] {done}/{len(jobs)} done, {cached} from cache, {errors} errors, "
                    f"{skipped} skipped after their prompt failed")
            # A sustained error rate means something systemic -- exhausted credits (402), a dead
            # endpoint, a bad key -- and spinning through the remaining jobs only fills the log.
            # Stop; the sink has everything that succeeded and a re-run resumes from there.
            if errors >= ERROR_ABORT_MIN and sum(recent) >= ERROR_ABORT_FRACTION * len(recent):
                cancelled = sum(f.cancel() for f in futs)
                pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"[llm:{spec.name}] aborting: {sum(recent)} of the last {len(recent)} calls "
                                   f"failed ({errors} in all, {cancelled} jobs cancelled); fix the cause and "
                                   f"re-run to resume")
    return errors + skipped


# ---------------- exec stage ----------------

def exec_signature(code: str, v: Variant, mode: str) -> str:
    """What an execution depends on: the program, the mode and the graph it runs on. A Graph-as-Code prompt does
    not contain the graph, so a changed graph leaves the prompt, and the reply, as they were."""
    payload = json.dumps([code, mode, v.node_seq, v.edge_seq, v.directed, v.weighted])
    return hashlib.sha256(payload.encode()).hexdigest()


def stage_exec(cfg: ExperimentConfig, spec: ModelSpec, variants) -> None:
    by_id = {v.variant_id: v for vs in variants.values() for v in vs}
    responses = [r for r in read_jsonl(cfg.results_dir / "responses" / f"{spec.name}.jsonl")
                 if r["mode"] in CODE_MODES and r.get("code")]
    sig = {r["record_id"]: exec_signature(r["code"], by_id[r["variant_id"]], r["mode"]) for r in responses}
    out = cfg.results_dir / "exec" / f"{spec.name}.jsonl"
    trim_partial_line(out)
    stale = {r["record_id"] for r in read_jsonl(out) if r["record_id"] in sig and r.get("exec_sig") != sig[r["record_id"]]}
    if stale:
        log(f"[exec:{spec.name}] {drop_records(out, stale)} executions ran another program or graph; running again")
    sink = Sink(out)
    jobs = [r for r in responses if not sink.has(r["record_id"])]
    log(f"[exec:{spec.name}] {len(jobs)} programs to run ({len(sink.ids)} already done)")
    timeout, mem = cfg.exec.get("timeout", 20), cfg.exec.get("mem_mb", 1024)

    def work(r):
        ex = sandbox.run(r["code"], by_id[r["variant_id"]], r["mode"], timeout=timeout, mem_mb=mem,
                         cache_dir=cfg.cache_dir / "exec")
        sink.write({"record_id": r["record_id"], "exec_sig": sig[r["record_id"]], **ex.to_dict()})

    with ThreadPoolExecutor(max_workers=cfg.exec.get("workers", os.cpu_count() or 2)) as pool:
        futs = [pool.submit(work, r) for r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 50 == 0 or i == len(jobs):
                log(f"[exec:{spec.name}] {i}/{len(jobs)} done")


# ---------------- score stage ----------------

def score_record(inst: GraphInstance, v: Variant, resp: dict, ex: dict | None) -> dict:
    mode, t = resp["mode"], inst.answer_type
    if mode in CODE_MODES:
        # CodeGraph contract: the answer is the variable `ans`; nothing else is consulted.
        line = ex.get("ans") if (ex and resp.get("code")) else None
        parsed = parse_value(line, t)
    else:
        line = find_boxed_answer(resp["raw_text"])
        parsed = parse_value(line, t)
    pred = relabel_answer(parsed, t, v.inverse_label_map)
    correct = compare(pred, inst.ground_truth, t, G=inst.to_nx()) if parsed is not None else False
    if mode in CODE_MODES and not resp.get("code"):
        fclass = "format"
    else:
        fclass = classify(mode, correct, parsed, ex)
    key = normalize_answer(pred, t, inst.directed) if parsed is not None else None
    return {
        "record_id": resp["record_id"], "dataset": inst.dataset, "task": inst.task,
        "graph_id": inst.meta.get("graph_id", inst.id), "instance_id": inst.id, "variant_id": v.variant_id,
        "axis": v.axis, "params": v.params, "mode": mode, "library": resp.get("library"),
        "model": resp["model"], "answer_type": t,
        # Rows sharing a prompt_hash came from ONE model call. M3 prompts for a task whose
        # question names no node are identical across every graph and every variant, so a
        # cell can hold 100 rows backed by a single response; the tables report both counts.
        "prompt_hash": resp.get("prompt_hash"),
        # "length" = the reply was cut off at max_tokens. For a degenerate reply (Qwen3-8B in
        # non-thinking mode loops on some prompts at temperature 0) this is a model defect, not a
        # serialization effect; the analysis needs to be able to see it to say so.
        "finish_reason": (resp.get("usage") or {}).get("finish_reason"),
        "ground_truth": inst.ground_truth, "answer_line": line, "parsed": parsed, "parsed_canonical": pred,
        "answer_key": json.dumps(key), "correct": bool(correct), "failure_class": fclass,
        "graph_match": ex.get("graph_match") if ex else None, "n_graphs": len(ex["graphs"]) if ex else None,
        "declared_ok": ex.get("declared_ok") if ex else None,
        "construction_ok": ex.get("construction_ok") if ex else None,
        "ans_is_literal": ex.get("ans_is_literal") if ex else None,
        "reads_graph_vars": ex.get("reads_graph_vars") if ex else None,
        "timed_out": ex.get("timed_out") if ex else None, "exit_code": ex.get("exit_code") if ex else None,
        "directedness_mismatch": ex.get("directedness_mismatch") if ex else None,
        "stub_behaviour": (resp.get("usage") or {}).get("stub_behaviour"),
    }


def stage_score(cfg: ExperimentConfig, spec: ModelSpec, instances, variants) -> None:
    inst_by_id = {i.id: i for i in instances}
    var_by_id = {v.variant_id: v for vs in variants.values() for v in vs}
    execs = {r["record_id"]: r for r in read_jsonl(cfg.results_dir / "exec" / f"{spec.name}.jsonl")}
    # one row per record: the last one written, if two processes ever appended the same record
    responses = {r["record_id"]: r for r in read_jsonl(cfg.results_dir / "responses" / f"{spec.name}.jsonl")}
    if not responses:
        # scored/<model>.jsonl may have been restored from the published dataset without its responses
        raise SystemExit(f"[score:{spec.name}] no responses in {cfg.results_dir / 'responses'}; "
                         f"not overwriting {cfg.results_dir / 'scored' / (spec.name + '.jsonl')}")
    rows = []
    for resp in responses.values():
        inst, v = inst_by_id.get(resp["instance_id"]), var_by_id.get(resp["variant_id"])
        if inst is None or v is None:
            continue
        ex = execs.get(resp["record_id"])
        if resp["mode"] in CODE_MODES and resp.get("code") and ex is None:
            continue
        rows.append(score_record(inst, v, resp, ex))
    write_jsonl(cfg.results_dir / "scored" / f"{spec.name}.jsonl", rows)
    n_ok = sum(r["correct"] for r in rows)
    log(f"[score:{spec.name}] {len(rows)} records scored, accuracy {n_ok / max(1, len(rows)):.3f}")


# ---------------- main ----------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="override the model list from the config")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--limit", type=int, help="max new prompts per model (debugging)")
    ap.add_argument("--estimate", action="store_true", help="build variants, print prompt counts, and exit")
    ap.add_argument("--rebuild-data", action="store_true",
                    help="regenerate instances.jsonl even though it exists (changes the committed data)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    load_dotenv(cfg.root)
    stages = [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; stages are {','.join(STAGES)}")
    if args.models is not None and not args.models:
        ap.error("--models needs at least one model name")
    specs = load_models(cfg.models_file)
    model_names = args.models or cfg.models
    missing = [m for m in model_names if m not in specs]
    if missing:
        ap.error(f"model(s) {missing} not in {cfg.models_file}")

    if "data" in stages or args.estimate or not (cfg.data_dir / "instances.jsonl").exists():
        instances = instances_for(cfg, args.rebuild_data)
    else:
        instances = load_instances(cfg)
    if "variants" in stages or args.estimate:
        variants = build_variants(cfg, instances)
        log(f"[variants] {sum(len(v) for v in variants.values())} variants")
    else:
        variants = load_variants(cfg)

    if args.estimate:
        estimate(cfg, instances, variants, model_names)
        return

    errors = 0
    for name in model_names:
        spec = specs[name]
        if "llm" in stages:
            errors += stage_llm(cfg, spec, instances, variants, limit=args.limit)
        if "exec" in stages:
            stage_exec(cfg, spec, variants)
        if "score" in stages:
            stage_score(cfg, spec, instances, variants)
    if errors:
        # the records that failed are missing; a re-run asks only them
        raise SystemExit(f"{errors} records have no reply (failed calls); re-run the same command to retry them")


if __name__ == "__main__":
    main()
