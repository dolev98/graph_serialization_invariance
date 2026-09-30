"""The nondeterminism (repeat) floor, measured with the invariance metric's own definitions.

How often does an IDENTICAL prompt give a different answer? Every "the answer changed under a
permutation" number is only evidence about the permutation to the extent it exceeds this.

    python scripts/nondeterminism_floor.py --config configs/graphqa.yaml \
        --models m1 m2 --instances 100 --repeats 4 --out results/review/floor/graphqa/noise_floor.json \
        --per-instance results/review/floor/graphqa/noise_floor_instances.jsonl

The committed files come from `--models hf-qwen3-8b-nscale hf-deepseek-v3.1-novita hf-gemma-4-31b
hf-deepseek-v4-flash` (that order). `--per-instance` was added later and run cache-only, with the
API keys blanked; the JSON it wrote alongside was byte-identical to the committed one, so the rows
are exactly what the JSON summarizes. `scripts/floor_stats.py` tests those rows.

Definitions match `gsi.analysis.tables.invariance_table`, which the first version of this script
did not:
  * k identical asks are compared the way an axis's forms are: they agree only if every answer
    parsed and all answer keys are equal. An unparsable reply counts as disagreement.
  * default k = 4, the number of forms `relabel` has per instance (identity + 3 seeds); `order`
    has 3. A 3-ask floor was *less* strict than either axis.
  * sampling is stratified (GraphQA by generator family, Erdos by task) so every family is
    represented -- a strided sample over sorted ids missed every star graph.
  * two reference prompts are re-asked where they differ: the canonical form (the reference for
    order/structure/syntax) and the sorted-identity relabeling (the reference for relabel). On
    GraphQA they are byte-identical; on Erdos they are not.
  * for the same instances, the observed flip rate on each permutation axis is reported next to
    the floor with a paired 95% interval on the difference. That is the number to quote; the
    floor is not subtracted from anything.

Answers are compared literally here: two different correct answers (two shortest paths) disagree,
and a reply that fails to parse disagrees even with an identical failure. The paper's flip rule
(correct answers count as one value, missing answers as another) is applied afterwards, from the
per-instance rows, by `floor_stats.py --rule flip`. The interval above is a Wald interval per cell,
with no correction for the 48 cells or for instances sharing a graph; `floor_stats.py` adds
mid-p McNemar with Benjamini-Hochberg, a clustered McNemar and a bootstrap over graphs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from gsi.data.base import read_jsonl
from gsi.exec import sandbox
from gsi.experiment.config import load_config, load_models
from gsi.experiment.run import load_dotenv, load_instances, load_variants, make_client, score_record
from gsi.prompts.modes import build_prompt

ARMS = [("direct", None), ("code", "networkx"), ("code", "native")]
NULL = (None, "null")


def stratified(instances, n):
    """Round-robin over strata (GraphQA: generator family; Erdos: task) until n are picked."""
    by = defaultdict(list)
    for i in instances:
        by[i.meta.get("generator") or i.task].append(i)
    out = []
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop(0))
    return out


def agree(keys: list) -> bool:
    return all(k not in NULL for k in keys) and len(set(keys)) == 1


def record_id(v, mode, lib, model):
    return f"{v.variant_id}::{mode}" + (f"::{lib}" if lib else "") + f"::{model}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-instance", default=None,
                    help="also write one JSONL row per (model, arm, reference, axis, instance) with the repeat "
                         "answers and the form answers, for scripts/floor_stats.py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    load_dotenv(cfg.root)
    specs = load_models(cfg.models_file)
    instances = {i.id: i for i in load_instances(cfg)}
    variants = load_variants(cfg)
    picked = stratified(list(instances.values()), args.instances)
    refs: dict[str, dict] = {}
    for inst in picked:
        vs = variants[inst.id]
        canon = next(v for v in vs if v.axis == "canonical")
        ident = next((v for v in vs if v.axis == "relabel" and v.params.get("seed") == "identity"), None)
        refs[inst.id] = {"canonical": canon}
        if ident is not None and ident.text != canon.text:
            refs[inst.id]["identity"] = ident
    n_ref = sum(len(r) for r in refs.values())
    print(f"{len(picked)} instances (stratified) x {len(ARMS)} arms x {args.repeats} asks x "
          f"{n_ref / len(picked):.2f} reference prompts = {n_ref * len(ARMS) * args.repeats} calls per model")

    report = {}
    inst_rows: list[dict] = []
    for name in args.models:
        spec = specs[name]
        client = make_client(spec, cfg.cache_dir)
        scored = {r["record_id"]: r for r in read_jsonl(cfg.results_dir / "scored" / f"{name}.jsonl")}
        jobs = [(iid, ref, mode, lib, r) for iid in refs for ref in refs[iid] for mode, lib in ARMS
                for r in range(args.repeats)]
        answers: dict[tuple, dict] = defaultdict(dict)

        def work(job):
            iid, ref, mode, lib, r = job
            inst, v = instances[iid], refs[iid][ref]
            p = build_prompt(mode, inst, v, lib, repeat=r)
            resp = client.complete(p, context={"inst": inst, "variant": v})
            ex = None
            if mode != "direct" and resp.code:
                ex = sandbox.run(resp.code, v, mode, timeout=cfg.exec.get("timeout", 20),
                                 cache_dir=cfg.cache_dir / "exec").to_dict()
            row = {"record_id": p.prompt_id, "mode": mode, "library": lib, "model": name, "raw_text": resp.raw_text,
                   "code": resp.code, "usage": resp.usage, "prompt_hash": resp.prompt_hash}
            return (iid, ref, mode, lib), r, score_record(inst, v, row, ex)["answer_key"]

        failed = 0
        with ThreadPoolExecutor(max_workers=max(1, spec.max_concurrency)) as pool:
            for n, f in enumerate(as_completed([pool.submit(work, j) for j in jobs]), 1):
                try:
                    key, r, ans = f.result()
                    answers[key][r] = ans
                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print(f"  [{name}] call failed ({type(e).__name__}); instance dropped", flush=True)
                if n % 200 == 0:
                    print(f"  [{name}] {n}/{len(jobs)} ({failed} failed)", flush=True)

        print(f"\n### {name}: {args.repeats} identical asks -- floor, next to the observed flip rate on the "
              f"same instances (paired)")
        rows = {}
        for mode, lib in ARMS:
            arm = f"{mode}/{lib or '-'}"
            for ref in ("canonical", "identity"):
                # sorted: answers fill in completion order, and summing the paired differences in a
                # different order changes the interval in its last bits from run to run
                keys = sorted(k for k in answers if k[1] == ref and k[2] == mode and k[3] == lib
                              and len(answers[k]) == args.repeats)
                if not keys:
                    continue
                floor_by_inst = {k[0]: (0 if agree(list(answers[k].values())) else 1) for k in keys}
                obs = {}
                for axis in ("relabel", "order"):
                    # relabel is judged against the sorted-identity reference where it exists;
                    # order/structure/syntax against canonical -- mirror invariance_table
                    if axis == "order" and ref == "identity":
                        continue
                    flips, floors = [], []
                    for k in keys:
                        iid = k[0]; ref_v = refs[iid][ref]
                        # per instance: an instance whose identity relabeling reads exactly like canonical has
                        # no separate identity reference, and its relabelings are judged against canonical
                        if axis == "relabel" and ref == "canonical" and "identity" in refs[iid]:
                            continue
                        forms = [v for v in variants[iid] if v.axis == axis and v.variant_id != ref_v.variant_id]
                        got = [scored[record_id(v, mode, lib, name)]["answer_key"] for v in forms + [ref_v]
                               if record_id(v, mode, lib, name) in scored]
                        if len(got) < 2:
                            continue
                        flips.append(0 if agree(got) else 1)
                        floors.append(floor_by_inst[iid])
                        if args.per_instance:
                            inst = instances[iid]
                            ref_rec = scored.get(record_id(ref_v, mode, lib, name))
                            inst_rows.append({
                                "dataset": cfg.name, "model": name, "mode": mode, "library": lib,
                                "ref": ref, "axis": axis, "instance_id": iid, "task": inst.task,
                                "graph_id": inst.meta.get("graph_id"), "source_file": inst.meta.get("source_file"),
                                "repeats": [answers[k][r] for r in range(args.repeats)],
                                "ref_key": ref_rec["answer_key"] if ref_rec else None,
                                # same_prompt: the form's prompt is byte-identical to the reference's
                                # (GraphQA's identity relabeling), so it reuses the reference's reply
                                "forms": [{"variant_id": v.variant_id,
                                           "key": scored[record_id(v, mode, lib, name)]["answer_key"],
                                           "same_prompt": v.text == ref_v.text}
                                          for v in forms if record_id(v, mode, lib, name) in scored],
                                "flip": flips[-1], "floor": floors[-1]})
                    if flips:
                        d = np.array(flips, float) - np.array(floors, float)
                        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                        obs[axis] = {"n": len(flips), "flip_rate": float(np.mean(flips)),
                                     "excess_over_floor": float(d.mean()),
                                     "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}
                floor = float(np.mean(list(floor_by_inst.values())))
                # An unparsable reply counts as disagreement (the metric's rule), so a program that
                # crashes on every repeat raises the floor without any answer changing. Report how
                # much of the floor that is, and how many instances lost a repeat to a failed call.
                rows[f"{arm}|{ref}"] = {"n": len(keys), "floor": floor, "disagreed": int(sum(floor_by_inst.values())),
                                        "disagreed_with_unparsable": sum(
                                            1 for k in keys if floor_by_inst[k[0]]
                                            and any(a in NULL for a in answers[k].values())),
                                        "dropped": sum(1 for k in answers if k[1] == ref and k[2] == mode
                                                       and k[3] == lib and len(answers[k]) != args.repeats),
                                        **{f"obs_{a}": o for a, o in obs.items()}}
                line = f"  {arm:15s} ref={ref:9s} n={len(keys):4d} floor={floor:6.1%}"
                for a, o in obs.items():
                    lo, hi = o["ci95"]
                    tag = "ABOVE floor" if lo > 0 else ("BELOW" if hi < 0 else "not distinguishable")
                    line += (f" | {a}: flips {o['flip_rate']:5.1%} excess {o['excess_over_floor']:+5.1%} "
                             f"[{lo:+.1%},{hi:+.1%}] {tag}")
                print(line)
        report[name] = rows
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": cfg.name, "repeats": args.repeats, "instances": len(picked),
                       "stratified": True, "report": report}, f, indent=1)
        print(f"\nwritten to {args.out}")
    if args.per_instance:
        with open(args.per_instance, "w") as f:
            for row in inst_rows:
                f.write(json.dumps(row) + "\n")
        print(f"{len(inst_rows)} per-instance rows written to {args.per_instance}")


if __name__ == "__main__":
    main()
