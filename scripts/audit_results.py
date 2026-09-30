"""Recompute the descriptive audit and paired graph-cluster bootstrap, without inference.

    python scripts/audit_results.py

Reads restored scored JSONL; writes diagnostics under results/review/.
Intervals describe graph-sampling uncertainty conditional on these cached model responses.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gsi.data.base import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PERMUTATIONS = ["relabel", "order"]
ARMS = ["direct", "code/native", "code/networkx"]
METRICS = ["identical", "all_correct", "identical_correct", "stable_wrong", "valid_different",
           "varying_error", "missing", "all_unparsed", "ref_correct", "ref_failure"]


def per_instance(rows, instances):
    groups = defaultdict(list)
    for row in rows:
        arm = row["mode"] + ("/" + row["library"] if row["library"] else "")
        groups[(row["instance_id"], arm)].append(row)
    output = []
    for (iid, arm), records in groups.items():
        inst = instances[iid]
        canonical = [r for r in records if r["axis"] == "canonical"]
        for axis in ["relabel", "order", "structure", "syntax"]:
            variants = [r for r in records if r["axis"] == axis]
            if axis == "relabel":
                refs = [r for r in variants if r["params"].get("seed") == "identity"]
                sub = variants if refs else canonical + variants
                refs = refs or canonical
            else:
                refs, sub = canonical, canonical + variants
            if not refs or len({r["variant_id"] for r in sub}) < 2:
                continue
            assert len(refs) == 1, (iid, arm, axis)
            parsed = [r["parsed"] is not None for r in sub]
            identical = all(parsed) and len({r["answer_key"] for r in sub}) == 1
            all_correct = all(r["correct"] for r in sub)
            output.append({
                "dataset": inst["dataset"], "task": inst["task"], "instance_id": iid,
                "cluster": inst["meta"].get("source_file") or inst["meta"]["graph_id"],
                "arm": arm, "axis": axis, "n_variants": len(sub),
                "n_prompts": len({r["prompt_hash"] for r in sub}),
                "identical": identical, "all_correct": all_correct,
                "identical_correct": identical and all_correct,
                "stable_wrong": identical and not any(r["correct"] for r in sub),
                "valid_different": all_correct and not identical,
                "varying_error": all(parsed) and not identical and not all_correct,
                "missing": not all(parsed), "all_unparsed": not any(parsed),
                "ref_correct": refs[0]["correct"],
                "ref_failure": refs[0]["correct"] and not all_correct,
            })
    frame = pd.DataFrame(output)
    categories = ["identical_correct", "stable_wrong", "valid_different", "varying_error", "missing"]
    assert (frame[categories].sum(axis=1) == 1).all(), "Categories do not partition the instance groups"
    return frame


def bootstrap(per, repeats, seed):
    """Resample graph clusters jointly across arms, tasks and permutation variants.

    Every draw estimates the mean of task x axis rates (the plotted estimand).
    Erdős source_file identities cluster the 21 sources reused across tasks.
    """
    sub = per[per.axis.isin(PERMUTATIONS) & per.arm.isin(ARMS)].copy()
    expected = set(zip(sub[sub.arm == ARMS[0]].instance_id, sub[sub.arm == ARMS[0]].axis))
    for arm in ARMS[1:]:
        selected = sub[sub.arm == arm]
        assert set(zip(selected.instance_id, selected.axis)) == expected
    clusters = sorted(sub.cluster.unique())
    cells = sorted(set(zip(sub.task, sub.axis)))
    ci = {c: i for i, c in enumerate(clusters)}
    ti = {t: i for i, t in enumerate(cells)}
    values = np.zeros((len(clusters), len(cells), len(ARMS), 3))
    for row in sub.itertuples():
        values[ci[row.cluster], ti[(row.task, row.axis)], ARMS.index(row.arm)] += [
            1, row.identical, row.all_correct]
    original = values.sum(axis=0)
    estimates = (original[:, :, 1:] / original[:, :, :1]).mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repeats, 250):
        weights = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)),
                                  size=min(250, repeats - start))
        total = (weights @ values.reshape(len(clusters), -1)).reshape(-1, *values.shape[1:])
        assert (total[:, :, :, 0] > 0).all(), "A bootstrap draw omitted a task/axis cell"
        draws.append((total[:, :, :, 1:] / total[:, :, :, :1]).mean(axis=1))
    draws = np.concatenate(draws)
    results = []
    for ai, arm in enumerate(ARMS[1:], 1):
        for mi, metric in enumerate(["identical", "all_correct"]):
            delta = draws[:, ai, mi] - draws[:, 0, mi]
            low, high = np.quantile(delta, [0.025, 0.975])
            results.append({"arm": arm, "metric": metric, "n_clusters": len(clusters),
                            "difference": float(estimates[ai, mi] - estimates[0, mi]),
                            "ci_low": float(low), "ci_high": float(high)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    out = ROOT / "results" / "review"
    out.mkdir(parents=True, exist_ok=True)
    report, instances_all, intervals = [], [], []
    for dataset in ["graphqa", "erdos"]:
        instances = {r["id"]: r for r in read_jsonl(ROOT / "data" / "processed" / dataset / "instances.jsonl")}
        for path in sorted((ROOT / "results" / dataset / "scored").glob("*.jsonl")):
            rows = list(read_jsonl(path))
            ids = [r["record_id"] for r in rows]
            assert len(ids) == len(set(ids)), f"Duplicate scored records in {path}"
            model = path.stem
            per = per_instance(rows, instances)
            per["model"] = model
            instances_all.append(per)
            task_rates = per.groupby(["arm", "axis", "task"])[METRICS].mean().reset_index()
            rates = task_rates.groupby(["arm", "axis"])[METRICS].mean().reset_index()
            perm = task_rates[task_rates.axis.isin(PERMUTATIONS)].groupby("arm")[METRICS].mean().reset_index()
            records = pd.DataFrame([{
                "arm": r["mode"] + ("/" + r["library"] if r["library"] else ""),
                "correct": r["correct"], "task": r["task"], "axis": r["axis"],
            } for r in rows])
            acc = records.groupby("arm").agg(accuracy=("correct", "mean"), n=("correct", "size")).reset_index()
            record = {"dataset": dataset, "model": model, "n_records": len(rows),
                      "accuracy": acc.to_dict("records"), "axis_rates": rates.to_dict("records"),
                      "permutation_rates": perm.to_dict("records")}
            report.append(record)
            for row in bootstrap(per, args.bootstrap, args.seed):
                intervals.append({"dataset": dataset, "model": model, **row})
            print(dataset, model, "records", len(rows), flush=True)
            print(perm[["arm", "identical", "all_correct", "stable_wrong", "missing", "all_unparsed"]].to_string(index=False), flush=True)
    pd.concat(instances_all, ignore_index=True).to_csv(out / "per_instance.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"bootstrap_repeats": args.bootstrap,
        "seed": args.seed, "results": report, "paired_intervals": intervals}, indent=2), encoding="utf-8")
    print(pd.DataFrame(intervals).to_string(index=False))
    print("Audit written to", out)


if __name__ == "__main__":
    main()
