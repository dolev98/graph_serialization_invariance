"""Before/after tables for the Erdős graph-as-code (M3) rerun with the graph type stated explicitly.

    python scripts/m3fix_compare.py [--fixed-run erdos_m3fix] [--models ...]

The rerun changes one sentence in undirected M3 prompts and nothing else, and `prompt_id` does not
include the sentence, so every fixed record pairs one-to-one with the base run on `record_id`:
same instance, form, arm and model. No inference; reads scored files only.

Writes CSVs under results/<fixed-run>/tables/compare/:
  accuracy_by_task.csv    canonical accuracy per model x arm x task, split by graph direction
  accuracy_overall.csv    accuracy per model x arm, both ways: record-weighted (acc_base, acc_fixed)
                          and instance-averaged (acc_base_instance, acc_fixed_instance: each
                          instance's forms averaged first). `delta` and its paired bootstrap interval
                          are instance-averaged: delta = acc_fixed_instance - acc_base_instance
  invariance_by_axis.csv  task-averaged I (all forms identical) and R (all forms correct) per axis,
                          computed with the project's own invariance_table on each run
  directedness.csv        share of NetworkX-arm recovered graphs with the wrong direction (n_recovered_*)
  m2_vs_m3.csv            the repaired localization comparison: code (M2, base run) vs M3 before/after
  failures.csv            failure-class shares among wrong M3 answers, before/after
Intervals are 95% percentile bootstrap over instances, conditional on the collected responses: they
do not cover a different generation of the same prompt, and a task whose M3 prompt names no node is
decided by one or two generated programs.

The two runs must hold the same M3 records (same instances, forms, arms, models). The script stops
if they do not -- a partial rerun would otherwise compare different instance sets -- unless
--allow-partial is given, in which case every table uses only the records both runs have.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gsi.analysis.tables import invariance_table, load_scored

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["hf-qwen3-8b-nscale", "hf-deepseek-v4-flash", "hf-gemma-4-31b", "hf-deepseek-v3.1-novita"]
M3 = "graph_as_code"


def directed_map() -> dict[str, bool]:
    with (ROOT / "data" / "processed" / "erdos" / "instances.jsonl").open(encoding="utf-8") as f:
        return {r["id"]: bool(r["directed"]) for r in map(json.loads, f)}


def load(run: str, models: list[str]) -> pd.DataFrame:
    d = ROOT / "results" / run
    have = [m for m in models if (d / "scored" / f"{m}.jsonl").exists()]
    if not have:
        # load_scored treats an empty list as "every model on disk"; never fall through to that
        raise SystemExit(f"none of the requested models has scored records under {d}: {models}")
    if len(have) < len(models):
        print(f"note: no scored records under {d} for {sorted(set(models) - set(have))}; left out")
    df = load_scored(d, have)
    if df.empty:
        raise SystemExit(f"no scored records under {d}")
    df["arm"] = df["mode"] + "/" + df["library"]
    return df


def ci(diffs: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    if len(diffs) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = diffs[rng.integers(0, len(diffs), (repeats, len(diffs)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def match(base: pd.DataFrame, fixed: pd.DataFrame, allow_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both runs to the M3 records they share, after checking that they share all of them.
    Every table then compares the same instances, forms and arms. A model present in only one run
    cannot be compared and is set aside with a note; for the others the record sets must be equal.
    Base rows of other modes (M2, used by m2_vs_m3) are kept for the compared models."""
    b3 = base[base["mode"] == M3]
    f3 = fixed[fixed["mode"] == M3]
    both = set(b3["model"]) & set(f3["model"])
    alone = sorted((set(b3["model"]) | set(f3["model"])) - both)
    if not both:
        raise SystemExit("no model has M3 records in both runs")
    if alone:
        print(f"note: {alone} have M3 records in only one run and are not compared")
    base, b3, f3 = base[base["model"].isin(both)], b3[b3["model"].isin(both)], f3[f3["model"].isin(both)]
    only_base = set(b3["record_id"]) - set(f3["record_id"])
    only_fixed = set(f3["record_id"]) - set(b3["record_id"])
    if only_base or only_fixed:
        msg = (f"the runs do not hold the same M3 records: {len(only_base)} only in the base run, "
               f"{len(only_fixed)} only in the fixed run")
        if not allow_partial:
            raise SystemExit(msg + "; finish the rerun, or pass --allow-partial to compare the shared records")
        print(f"note: {msg}; every table uses the shared records only")
    shared = set(b3["record_id"]) & set(f3["record_id"])
    base = pd.concat([base[base["mode"] != M3], b3[b3["record_id"].isin(shared)]])
    return base, f3[f3["record_id"].isin(shared)]


def paired(base: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    base = base[base["mode"] == M3]
    fixed = fixed[fixed["mode"] == M3]
    cols = ["record_id", "correct", "parsed", "answer_key", "failure_class", "directedness_mismatch", "n_graphs",
            "prompt_hash"]
    df = fixed.merge(base[cols], on="record_id", suffixes=("_fixed", "_base"), validate="one_to_one")
    missing = len(fixed) - len(df)
    if missing:
        print(f"note: {missing} fixed records have no base counterpart and are dropped from paired tables")
    df["directed"] = df["instance_id"].map(directed_map())
    return df


def accuracy_by_task(df: pd.DataFrame) -> pd.DataFrame:
    can = df[df["axis"] == "canonical"]
    rows = []
    for (model, arm, task), g in can.groupby(["model", "arm", "task"]):
        row = {"model": model, "arm": arm, "task": task, "n": len(g),
               "n_undirected": int((~g["directed"]).sum()),
               "acc_base": g["correct_base"].mean(), "acc_fixed": g["correct_fixed"].mean()}
        for name, sub in (("undirected", g[~g["directed"]]), ("directed", g[g["directed"]])):
            row[f"{name}_base"] = sub["correct_base"].mean() if len(sub) else np.nan
            row[f"{name}_fixed"] = sub["correct_fixed"].mean() if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def accuracy_overall(df: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    for (model, arm), g in df.groupby(["model", "arm"]):
        per = g.groupby("instance_id")[["correct_base", "correct_fixed"]].mean()
        diffs = (per["correct_fixed"] - per["correct_base"]).to_numpy()
        lo, hi = ci(diffs, repeats, seed)
        rows.append({"model": model, "arm": arm, "n_records": len(g), "n_instances": len(per),
                     "acc_base": g["correct_base"].mean(), "acc_fixed": g["correct_fixed"].mean(),
                     "acc_base_instance": per["correct_base"].mean(),
                     "acc_fixed_instance": per["correct_fixed"].mean(),
                     "delta": diffs.mean(), "ci_low": lo, "ci_high": hi,
                     "undirected_base": g.loc[~g["directed"], "correct_base"].mean(),
                     "undirected_fixed": g.loc[~g["directed"], "correct_fixed"].mean(),
                     "directed_base": g.loc[g["directed"], "correct_base"].mean(),
                     "directed_fixed": g.loc[g["directed"], "correct_fixed"].mean()})
    return pd.DataFrame(rows)


def invariance_by_axis(base: pd.DataFrame, fixed: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    keys = ["model", "library", "axis"]
    b = invariance_table(base[(base["mode"] == M3) & base["model"].isin(models)])
    f = invariance_table(fixed[fixed["mode"] == M3])
    # task-averaged: every task weighs the same
    b = b.groupby(keys)[["frac_identical", "frac_all_correct"]].mean().rename(
        columns={"frac_identical": "I_base", "frac_all_correct": "R_base"})
    f = f.groupby(keys)[["frac_identical", "frac_all_correct"]].mean().rename(
        columns={"frac_identical": "I_fixed", "frac_all_correct": "R_fixed"})
    out = b.join(f, how="inner").reset_index()
    out["I_delta"] = out["I_fixed"] - out["I_base"]
    out["R_delta"] = out["R_fixed"] - out["R_base"]
    return out[keys + ["I_base", "I_fixed", "I_delta", "R_base", "R_fixed", "R_delta"]]


def directedness(df: pd.DataFrame) -> pd.DataFrame:
    nx_arm = df[df["library"] == "networkx"]
    rows = []
    for (model, directed), g in nx_arm.groupby(["model", "directed"]):
        row = {"model": model, "graph": "directed" if directed else "undirected", "n": len(g)}
        # direction is observed only where a graph object was recovered; directedness_mismatch is
        # stored as False where none was, so shares are taken over recovered graphs only
        for side in ("base", "fixed"):
            seen = g[g[f"n_graphs_{side}"].fillna(0) > 0]
            row[f"n_recovered_{side}"] = len(seen)
            row[f"mismatch_{side}"] = (seen[f"directedness_mismatch_{side}"].fillna(False).astype(bool).mean()
                                       if len(seen) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def m2_vs_m3(base: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Canonical accuracy of code-from-text (M2, base run) next to M3 before and after the fix."""
    m2 = base[(base["mode"] == "code") & (base["axis"] == "canonical")]
    m2 = m2.groupby(["model", "library", "task"])["correct"].mean().rename("m2_code")
    m3 = df[df["axis"] == "canonical"].groupby(["model", "library", "task"]).agg(
        m3_base=("correct_base", "mean"), m3_fixed=("correct_fixed", "mean"))
    out = m3.join(m2, how="left").reset_index()
    out["gap_base"] = out["m2_code"] - out["m3_base"]
    out["gap_fixed"] = out["m2_code"] - out["m3_fixed"]
    return out


def failures(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, arm), g in df.groupby(["model", "arm"]):
        row = {"model": model, "arm": arm, "n": len(g),
               "wrong_base": int((~g["correct_base"]).sum()), "wrong_fixed": int((~g["correct_fixed"]).sum())}
        for side in ("base", "fixed"):
            wrong = g[~g[f"correct_{side}"]]
            for cls, share in wrong[f"failure_class_{side}"].value_counts(normalize=True).items():
                row[f"{cls}_{side}"] = share
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", default="erdos")
    ap.add_argument("--fixed-run", default="erdos_m3fix")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--repeats", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--allow-partial", action="store_true",
                    help="compare only the M3 records both runs hold instead of stopping")
    args = ap.parse_args()

    base, fixed = load(args.base_run, args.models), load(args.fixed_run, args.models)
    models = sorted(set(fixed["model"]))
    base = base[base["model"].isin(models)]
    base, fixed = match(base, fixed, args.allow_partial)
    df = paired(base, fixed)
    out = ROOT / "results" / args.fixed_run / "tables" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "accuracy_by_task": accuracy_by_task(df),
        "accuracy_overall": accuracy_overall(df, args.repeats, args.seed),
        "invariance_by_axis": invariance_by_axis(base, fixed, models),
        "directedness": directedness(df),
        "m2_vs_m3": m2_vs_m3(base, df),
        "failures": failures(df),
    }
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)
    for name, t in tables.items():
        print(f"\n=== {name} ===")
        print(t.round(3).to_string(index=False))
        t.to_csv(out / f"{name}.csv", index=False)
    print(f"\ntables written to {out}")


if __name__ == "__main__":
    main()
