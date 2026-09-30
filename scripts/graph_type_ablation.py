"""Ablation: what does a model assume when the serialization omits the graph type?

For an undirected graph the Erdos graph-as-code (M3) prompt stated the graph type only if the task
preamble did (paper, Section 5.4 and Appendix D). The rerun that states it for
every undirected graph (`configs/erdos_m3fix.yaml`) turns that defect into a controlled
one-sentence ablation: the same model, task, graph and template, with one fact present or absent.

Three questions, three tables under results/erdos_m3fix/tables/ablation/:

  task_sensitivity.csv  Which tasks reveal the misreading? Per model x library x task, canonical
                        accuracy with the type omitted vs stated, from records (base_source =
                        "records"); a model whose base records are absent falls back to the base
                        run's committed per-model tables (base_source = "tables").
  latent_misreading.csv Did the model misread the input even where the answer was right? In the
                        NetworkX arm the sandbox recovers the graph the program built, so
                        `directedness_mismatch` is direct evidence of a DiGraph on undirected
                        input. Cross-tabulated with correctness. Needs the base run's records
                        (results/erdos/scored/, restored by scripts/restore_from_hf.py).
  recovery.csv          After the sentence is added: mismatch rate and accuracy, all four models.

Graph direction is observed only where the sandbox recovered a graph object (`n_graphs` > 0): never
in the native arm, and not for a NetworkX-arm program that builds no graph. `directedness_mismatch`
is stored as False in those records too, so every DiGraph share here is taken over recovered
graphs only, next to their count (`n_recovered`), and is empty when none was recovered.

No inference and no program execution; reads scored records and committed tables.

    python scripts/graph_type_ablation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["hf-qwen3-8b-nscale", "hf-deepseek-v4-flash", "hf-gemma-4-31b", "hf-deepseek-v3.1-novita"]
M3 = "graph_as_code"


def directed_map() -> dict[str, bool]:
    with (ROOT / "data" / "processed" / "erdos" / "instances.jsonl").open(encoding="utf-8") as f:
        return {r["id"]: bool(r["directed"]) for r in map(json.loads, f)}


def read_records(run: str, model: str) -> pd.DataFrame | None:
    p = ROOT / "results" / run / "scored" / f"{model}.jsonl"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        df = pd.DataFrame(map(json.loads, f))
    df = df[(df["mode"] == M3) & (df["axis"] == "canonical")].copy()
    df["directed"] = df["instance_id"].map(directed_map())
    return df


def recovered(df: pd.DataFrame) -> pd.Series:
    """Records whose program left a graph object to inspect."""
    return df["n_graphs"].fillna(0) > 0


def base_accuracy_table(model: str) -> pd.DataFrame | None:
    """Canonical M3 accuracy per task/library from the base run's committed tables. A fallback for
    a model whose base records are absent; the published records cover all four models."""
    p = ROOT / "results" / "erdos" / "tables" / model / "accuracy.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p)
    a = a[(a["mode"] == M3) & (a["axis"] == "canonical")]
    return a[["task", "library", "accuracy", "n"]].rename(columns={"accuracy": "acc_omitted"})


def task_sensitivity() -> pd.DataFrame:
    """Canonical accuracy with the graph type omitted vs stated, per model x library x task.
    The undirected columns are the clean contrast: directed prompts are byte-identical between
    the runs, so only undirected instances actually saw a changed prompt."""
    rows = []
    for model in MODELS:
        fixed = read_records("erdos_m3fix", model)
        if fixed is None:
            continue
        stated = fixed.groupby(["task", "library"]).agg(acc_stated=("correct", "mean")).reset_index()
        und = (fixed[~fixed["directed"]].groupby(["task", "library"])
               .agg(acc_stated_undirected=("correct", "mean"), n_undirected=("correct", "size")).reset_index())
        stated = stated.merge(und, on=["task", "library"], how="left")

        base_rec = read_records("erdos", model)
        if base_rec is not None:
            # compare the same canonical records on both sides, or not at all: an unfinished
            # rerun would otherwise set different instance sets against each other
            gap = set(base_rec["record_id"]) ^ set(fixed["record_id"])
            if gap:
                raise SystemExit(f"{model}: {len(gap)} canonical M3 records are in only one run; "
                                 "finish the rerun before comparing")
            omitted = base_rec.groupby(["task", "library"]).agg(acc_omitted=("correct", "mean")).reset_index()
            ou = (base_rec[~base_rec["directed"]].groupby(["task", "library"])
                  .agg(acc_omitted_undirected=("correct", "mean")).reset_index())
            omitted = omitted.merge(ou, on=["task", "library"], how="left")
            src = "records"
        else:
            omitted = base_accuracy_table(model)
            if omitted is None:
                continue
            # the table's n is the base run's count of canonical records per task/library; the rerun
            # must hold exactly as many, or an unfinished rerun is compared with a complete base
            have = fixed.groupby(["task", "library"]).size().rename("n_fixed").reset_index()
            chk = omitted.merge(have, on=["task", "library"], how="outer")
            if (chk["n"] != chk["n_fixed"]).any():
                raise SystemExit(f"{model}: the rerun's canonical record counts differ from the base "
                                 "run's tables; finish the rerun before comparing")
            omitted = omitted.drop(columns=["n"])
            omitted["acc_omitted_undirected"] = float("nan")
            src = "tables"

        t = stated.merge(omitted, on=["task", "library"], how="inner")
        t["model"], t["base_source"] = model, src
        t["delta"] = t["acc_stated"] - t["acc_omitted"]
        t["delta_undirected"] = t["acc_stated_undirected"] - t["acc_omitted_undirected"]
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    # An accuracy change does not say why it happened: single-program flips and re-asked prompts
    # move tasks too (paper, Section 5.4). Flag the size only.
    out["accuracy_gain_gt_5pp"] = out["delta"] > 0.05
    cols = ["model", "library", "task", "base_source", "n_undirected", "acc_omitted", "acc_stated",
            "delta", "acc_omitted_undirected", "acc_stated_undirected", "delta_undirected", "accuracy_gain_gt_5pp"]
    return out[cols].sort_values(["model", "library", "task"]).reset_index(drop=True)


def latent_misreading() -> pd.DataFrame:
    """Among undirected instances in the NetworkX arm, how often did the program build a DiGraph,
    and how often did that still produce the right answer? A correct answer from a wrongly
    constructed graph is a latent error: the model misread the input and the task did not notice."""
    rows = []
    for model in MODELS:
        base = read_records("erdos", model)
        if base is None:
            continue
        nx_arm = base[(base["library"] == "networkx") & (~base["directed"])].copy()
        nx_arm["mismatch"] = nx_arm["directedness_mismatch"].fillna(False).astype(bool)
        nx_arm["recovered"] = recovered(nx_arm)
        for task, g in nx_arm.groupby("task"):
            mism = g[g["mismatch"]]
            seen = g[g["recovered"]]
            rows.append({
                "model": model, "task": task, "n_undirected": len(g), "n_recovered": len(seen),
                "built_digraph": seen["mismatch"].mean() if len(seen) else float("nan"),
                "correct_overall": g["correct"].mean(),
                "correct_given_digraph": mism["correct"].mean() if len(mism) else float("nan"),
                "latent_errors": int((mism["correct"]).sum()),
                "surfaced_errors": int((~mism["correct"]).sum()),
            })
    return pd.DataFrame(rows).sort_values(["model", "task"]).reset_index(drop=True)


def recovery() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        fixed = read_records("erdos_m3fix", model)
        if fixed is None:
            continue
        for lib, g in fixed[~fixed["directed"]].groupby("library"):
            seen = g[recovered(g)]
            mism = seen["directedness_mismatch"].fillna(False).astype(bool)
            rows.append({"model": model, "library": lib, "n_undirected": len(g), "n_recovered": len(seen),
                         "built_digraph_after": mism.mean() if len(seen) else float("nan"),
                         "accuracy_after": g["correct"].mean(),
                         "note": "" if lib == "networkx" else "direction not observable in native arm"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/erdos_m3fix/tables/ablation")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 30)
    for name, t in (("task_sensitivity", task_sensitivity()),
                    ("latent_misreading", latent_misreading()),
                    ("recovery", recovery())):
        print(f"\n=== {name} ===")
        print(t.round(3).to_string(index=False))
        t.to_csv(out / f"{name}.csv", index=False)
    print(f"\ntables written to {out}")


if __name__ == "__main__":
    main()
