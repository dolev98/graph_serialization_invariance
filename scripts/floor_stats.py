"""Repeat-floor statistics: is each permutation flip rate above the repeat-noise floor? No inference.

    python scripts/floor_stats.py --rule literal
    python scripts/floor_stats.py --rule flip

Reads the per-instance rows `nondeterminism_floor.py --per-instance` writes
(results/review/floor/<dataset>/noise_floor_instances.jsonl, both datasets) and writes, for the chosen
rule, results/review/floor/floor_stats_<rule>_cells.csv (one row per dataset x model x arm x axis),
floor_stats_<rule>_pooled.csv (per model: all cells, per dataset, per arm) and
floor_stats_<rule>_modes.csv (per dataset x arm, for each model and for the four models together).
Section 5.1 of the paper rests on the flip rule's cells: Qwen3-8B's code clearly exceeds its floor,
and the other models' code rarely does.

Rules. When do k answers "agree"?
  * literal -- the rule of nondeterminism_floor.py and noise_floor.json: every answer parsed and all
    answer keys equal. Two different correct answers (two shortest paths) disagree, and an
    unparsable reply disagrees even with an identical failure. The script asserts that it
    reproduces noise_floor.json under this rule.
  * flip -- the paper's flip rule (Section 4.2), the one scripts/paper_metrics.py applies to the main
    run: the answers agree when they are all the same, where every correct answer counts as the
    same answer and every missing answer counts as one shared value. Correctness is
    gsi.score.compare of the answer key against the instance's ground truth, the comparison that
    sets the scored records' `correct` field; where the scored records are on disk
    (scripts/restore_from_hf.py) the script asserts that it agrees with that field.
Both rules are applied to the repeats (the floor) and to the forms (the flip) alike. An answer
group that agrees literally also agrees under the flip rule, so the flip rule can only lower both.

Why this exists. noise_floor.json puts a normal-approximation (Wald) interval on the paired excess, on
about 100 instances with rates near zero, cell by cell, 48 cells, without clustering. The
evaluation-statistics literature says that is too optimistic in exactly this regime (Bowyer et al.,
ICML 2025; Fagerland et al., BMC Med Res Methodol 2013), and the GraphQA sample is 100 instances from
only 16 graphs (the stratified picker takes all seven tasks of a graph). So, per cell:

  * mid-p McNemar on the discordant pairs (b = flipped but repeats agreed, c = the reverse), the test
    Fagerland et al. recommend for matched binary data; BH over all cells, Holm for reference;
  * Durkalski's clustered McNemar (Stat Med 2003), which lets instances of one graph be correlated;
  * a percentile interval from a bootstrap that resamples graphs, not instances;
  * the pairwise estimator: disagreement between a repeat and a perturbed form, minus disagreement
    between two repeats (Feng et al., 2026). It uses every pair of answers, does not depend on how many
    forms or repeats there are, and leaves out forms whose prompt is byte-identical to the reference.

and, pooled over cells with the same graph bootstrap, per model (datasets x arms x axes, and per
dataset or arm) and per dataset x arm (both axes; per model and over the four models): the answer to
"does this model flip more than its floor at all", which 48 small tests cannot give. A pooled value
is the mean over its cells, so every cell (and every model) counts equally.

Bootstrap. `--bootstrap` resamples (default 10,000) with `--seed` (default 20260928); the committed
tables use the defaults. Graphs (GraphQA graph_id, Erdos source file) are drawn once per dataset and
the same draws serve every cell, every pooled row and both rules, so pooled intervals resample graphs
jointly across arms, axes and models.

Per-cell decisions use mid-p + BH, not the bootstrap: in a cell where no instance goes the other way
(c = 0) the percentile interval's lower bound is just the smallest positive value a resample can take
(1/n), the same optimism as Wald's. The bootstrap is for the pooled estimates, which rest on many more
discordant pairs.

The floor is measured on the reference prompt only, for M1 and M2 (no M3 arm); both estimators assume
the perturbed prompts are no noisier than it. One draw per perturbed form cannot test that assumption.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gsi.analysis.tuples import durkalski

REPO = Path(__file__).resolve().parents[1]
FLOOR = REPO / "results" / "review" / "floor"
DATASETS = ["graphqa", "erdos"]
RULES = ("literal", "flip")
ARMS = ["direct/-", "code/native", "code/networkx"]   # the paper's column order: M1, plain Python, NetworkX
NULL = (None, "null")
NA, CORRECT = "NA", "CORRECT"                          # the flip rule's shared values


def key(k):
    """Answer keys as comparable strings; None for an unparsable answer."""
    if k in NULL:
        return None
    return k if isinstance(k, str) else json.dumps(k)


def agree_all(keys) -> bool:
    """The invariance metric's rule: every answer parsed and all are equal."""
    return all(k is not None for k in keys) and len(set(keys)) == 1


def same(a, b) -> bool:
    return a is not None and a == b


def same_value(a, b) -> bool:
    """Flip rule: values are already mapped to CORRECT / NA / the key, so equality is enough."""
    return a == b


def pairwise(repeats, forms, same=same):
    """(disagreement within repeats, disagreement between a repeat and a form); pairs, not all-agree."""
    within = [not same(a, b) for a, b in itertools.combinations(repeats, 2)]
    across = [not same(a, f) for a in repeats for f in forms]
    return float(np.mean(within)), float(np.mean(across))


def mcnemar_midp(b: int, c: int) -> float:
    """Two-sided mid-p McNemar (Fagerland, Lydersen & Laake 2013)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    below = sum(math.comb(m, i) for i in range(k)) / 2 ** m
    return min(1.0, 2 * (below + 0.5 * math.comb(m, k) / 2 ** m))


def mcnemar_exact(b: int, c: int) -> float:
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m)


def mcnemar_clustered(b_by_cluster, c_by_cluster) -> float:
    """Durkalski et al. (2003): X = (sum_k (b_k - c_k))^2 / sum_k (b_k - c_k)^2, chi-square(1)."""
    return durkalski(list(b_by_cluster), list(c_by_cluster))[1]


def bh(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def holm(pvalues, alpha=0.05) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    reject = np.zeros(len(p), bool)
    for j, i in enumerate(order):
        if p[i] > alpha / (len(p) - j):
            break
        reject[i] = True
    return reject


class Correctness:
    """is_correct(instance_id, answer_key) by gsi.score.compare against the instance's ground truth.

    score_record (gsi.experiment.run) sets a scored record's `correct` with the same call on the parsed
    answer; the answer key is that answer normalised, and check_against_scored confirms the two agree
    on these data. Memoised per (instance, key)."""

    def __init__(self, instances):
        self.instances = {i.id: i for i in instances}
        self._graphs, self._memo = {}, {}

    @classmethod
    def from_configs(cls, datasets=DATASETS):
        from gsi.experiment.config import load_config
        from gsi.experiment.run import load_instances
        return cls([i for ds in datasets for i in load_instances(load_config(REPO / "configs" / f"{ds}.yaml"))])

    def __call__(self, instance_id: str, k) -> bool:
        if key(k) is None:
            return False
        memo = (instance_id, key(k))
        if memo not in self._memo:
            from gsi.score.compare import compare
            inst = self.instances[instance_id]
            g = self._graphs.setdefault(instance_id, inst.to_nx())
            self._memo[memo] = bool(compare(json.loads(key(k)), inst.ground_truth, inst.answer_type, G=g))
        return self._memo[memo]


def flip_value(instance_id: str, k, is_correct) -> str:
    """The value an answer takes under the flip rule: NA if missing, CORRECT if correct, else its key."""
    if key(k) is None:
        return NA
    return CORRECT if is_correct(instance_id, k) else key(k)


def load(path: Path, rule: str = "literal", is_correct=None) -> pd.DataFrame:
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        reps = [key(k) for k in r["repeats"]]
        forms = [key(f["key"]) for f in r["forms"]]
        distinct = [key(f["key"]) for f in r["forms"] if not f["same_prompt"]]
        # the dump must reproduce the floor script's own indicators
        assert int(not agree_all(reps)) == r["floor"], r["instance_id"]
        assert int(not agree_all(forms + [key(r["ref_key"])])) == r["flip"], r["instance_id"]
        if rule == "literal":
            floor, flip, eq = r["floor"], r["flip"], same
        elif rule == "flip":
            v = lambda k: flip_value(r["instance_id"], k, is_correct)  # noqa: E731
            reps, distinct = [v(k) for k in r["repeats"]], [v(f["key"]) for f in r["forms"] if not f["same_prompt"]]
            floor = int(len(set(reps)) > 1)
            flip = int(len({v(f["key"]) for f in r["forms"]} | {v(r["ref_key"])}) > 1)
            # agreeing literally implies agreeing under the flip rule, never the reverse
            assert floor <= r["floor"] and flip <= r["flip"], r["instance_id"]
            eq = same_value
        else:
            raise ValueError(rule)
        within, across = pairwise(reps, distinct, eq) if distinct else (np.nan, np.nan)
        rows.append({"dataset": r["dataset"], "model": r["model"],
                     "arm": r["mode"] + "/" + (r["library"] or "-"), "axis": r["axis"],
                     "instance_id": r["instance_id"], "task": r["task"],
                     "cluster": r["graph_id"] if r["dataset"] == "graphqa" else r["source_file"],
                     "flip": flip, "floor": floor,
                     "d_within": within, "d_across": across, "u": across - within})
    return pd.DataFrame(rows)


CELL = ["dataset", "model", "arm", "axis"]


def cluster_sums(df: pd.DataFrame, clusters: list, value: str) -> np.ndarray:
    """(clusters x 2) matrix: count and sum of `value` per cluster (NaN rows dropped)."""
    idx = {c: i for i, c in enumerate(clusters)}
    out = np.zeros((len(clusters), 2))
    for c, v in zip(df.cluster, df[value]):
        if not np.isnan(v):
            out[idx[c]] += (1, v)
    return out


def boot_weights(n_clusters: int, reps: int, rng) -> np.ndarray:
    return rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=reps)


def ratio(w: np.ndarray, sums: np.ndarray) -> np.ndarray:
    cnt, tot = w @ sums[:, 0], w @ sums[:, 1]
    return np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), np.nan)


def pool(g: pd.DataFrame, draws: dict) -> dict:
    """Mean over the cells in `g` (rows of the cell table), with the graph-bootstrap interval of that mean."""
    sel = [tuple(r) for r in g[CELL].itertuples(index=False)]
    ex = np.nanmean(np.stack([draws[s][0] for s in sel]), axis=0)
    u = np.nanmean(np.stack([draws[s][1] for s in sel]), axis=0)
    lo, hi = np.nanquantile(ex, [0.025, 0.975])
    ulo, uhi = np.nanquantile(u, [0.025, 0.975])
    return {"cells": len(sel), "floor": g["floor"].mean(), "flip": g.flip.mean(),
            "excess": g.excess.mean(), "boot_lo": lo, "boot_hi": hi,
            "p_boot": float(min(1.0, 2 * min((ex <= 0).mean(), (ex >= 0).mean()))),
            "u": g.u.mean(), "u_lo": ulo, "u_hi": uhi}


def analyse(df: pd.DataFrame, reps: int, seed: int):
    """(cells, pooled per model, per dataset x arm table)."""
    df = df.assign(excess=df.flip - df["floor"])
    rng = np.random.default_rng(seed)
    clusters = {ds: sorted(df[df.dataset == ds].cluster.unique()) for ds in DATASETS}
    # one set of cluster draws per dataset, shared by every cell and every pooled statistic,
    # so pooled estimates resample graphs jointly across arms and axes
    weights = {ds: boot_weights(len(clusters[ds]), reps, rng) for ds in DATASETS}

    cells, draws = [], {}
    for cell, g in df.groupby(CELL, sort=False):
        ds = cell[0]
        b = int(((g.flip == 1) & (g["floor"] == 0)).sum())
        c = int(((g.flip == 0) & (g["floor"] == 1)).sum())
        d = g.excess.to_numpy(float)
        se = d.std(ddof=1) / math.sqrt(len(d))
        per_cluster = g.groupby("cluster").agg(b=("excess", lambda s: int((s == 1).sum())),
                                               c=("excess", lambda s: int((s == -1).sum())))
        ex_draws = ratio(weights[ds], cluster_sums(g, clusters[ds], "excess"))
        u_draws = ratio(weights[ds], cluster_sums(g, clusters[ds], "u"))
        draws[cell] = (ex_draws, u_draws)
        lo, hi = np.nanquantile(ex_draws, [0.025, 0.975])
        ulo, uhi = np.nanquantile(u_draws, [0.025, 0.975])
        cells.append({**dict(zip(CELL, cell)), "n": len(g), "n_clusters": g.cluster.nunique(),
                      "floor": g["floor"].mean(), "flip": g.flip.mean(), "excess": d.mean(),
                      "b": b, "c": c,
                      "wald_lo": d.mean() - 1.96 * se, "wald_hi": d.mean() + 1.96 * se,
                      "p_exact": mcnemar_exact(b, c), "p_midp": mcnemar_midp(b, c),
                      "p_clustered": mcnemar_clustered(per_cluster.b, per_cluster.c),
                      "boot_lo": lo, "boot_hi": hi,
                      "d_within": g.d_within.mean(), "d_across": g.d_across.mean(), "u": g.u.mean(),
                      "u_lo": ulo, "u_hi": uhi})
    cells = pd.DataFrame(cells)
    cells["q_bh"] = bh(cells.p_midp)
    cells["holm"] = holm(cells.p_midp)
    cells["wald_sig"] = cells.wald_lo > 0
    cells["midp_sig"] = cells.p_midp < 0.05
    cells["bh_sig"] = cells.q_bh < 0.05
    cells["boot_sig"] = cells.boot_lo > 0

    pooled = []
    groupings = {"all": [], "dataset": ["dataset"], "arm": ["arm"]}
    for scope, extra in groupings.items():
        for (model, *rest), g in cells.groupby(["model"] + extra, sort=False):
            pooled.append({"model": model, "scope": scope, "level": rest[0] if rest else "all", **pool(g, draws)})

    # per dataset x arm, both axes: each model, then the four models counting equally
    modes = []
    for ds in DATASETS:
        for arm in ARMS:
            g = cells[(cells.dataset == ds) & (cells.arm == arm)]
            if g.empty:
                continue
            for model, gm in [*g.groupby("model", sort=False), ("all", g)]:
                modes.append({"dataset": ds, "arm": arm, "model": model, "n": int(gm.n.sum()),
                              "b": int(gm.b.sum()), "c": int(gm.c.sum()), **pool(gm, draws)})
    return cells, pd.DataFrame(pooled), pd.DataFrame(modes)


def check_against_json(cells: pd.DataFrame) -> None:
    """Literal rule: the recomputed floor, flip rate, excess and Wald interval must match
    results/review/floor/<dataset>/noise_floor.json."""
    checked = 0
    for ds in DATASETS:
        report = json.loads((FLOOR / ds / "noise_floor.json").read_text())["report"]
        for model, arms in report.items():
            # a cell's instances can come from both references: an Erdos relabel cell holds the instances with a
            # separate sorted-identity reference and those whose identity relabeling reads like canonical
            parts: dict = {}
            for arm_ref, r in arms.items():
                for axis in ("relabel", "order"):
                    if r.get(f"obs_{axis}"):
                        parts.setdefault((arm_ref.split("|")[0], axis), []).append((r, r[f"obs_{axis}"]))
            for (arm, axis), got in parts.items():
                row = cells[(cells.dataset == ds) & (cells.model == model) & (cells.arm == arm)
                            & (cells.axis == axis)].iloc[0]
                n = sum(o["n"] for _, o in got)
                assert row.n == n, (ds, model, arm, axis)
                assert math.isclose(row.flip, sum(o["n"] * o["flip_rate"] for _, o in got) / n), (ds, model, arm, axis)
                assert math.isclose(row.excess, sum(o["n"] * o["excess_over_floor"] for _, o in got) / n, abs_tol=1e-12)
                if len(got) == 1:  # one reference: its Wald interval and floor are the cell's
                    r, o = got[0]
                    assert math.isclose(row.wald_lo, o["ci95"][0], abs_tol=1e-9)
                    assert math.isclose(row.wald_hi, o["ci95"][1], abs_tol=1e-9)
                    if o["n"] == r["n"]:  # the cell covers every instance the floor was measured on
                        assert math.isclose(row["floor"], r["floor"]), (ds, model, arm, axis)
                checked += 1
    assert checked == len(cells), (checked, len(cells))
    print(f"literal rule: all {checked} cells reproduce noise_floor.json")


def check_against_scored(is_correct: Correctness) -> None:
    """Flip rule: the recomputed correctness must equal the scored records' `correct` field. The
    reference and form answers are checked against their own records; a repeat answer is checked
    wherever the same instance got the same answer key in some scored record (any model, arm or
    variant). Skipped, with a note, when the scored records are not on disk."""
    checked = records = unmatched = 0
    for ds in DATASETS:
        rows = [json.loads(line) for line in (FLOOR / ds / "noise_floor_instances.jsonl").open()]
        ids = {r["instance_id"] for r in rows}
        scored_dir = REPO / "results" / ds / "scored"
        files = [scored_dir / f"{m}.jsonl" for m in dict.fromkeys(r["model"] for r in rows)]
        if not all(p.exists() for p in files):
            print(f"flip rule: no scored records under {scored_dir}; correctness NOT checked against their "
                  f"`correct` field (scripts/restore_from_hf.py restores them)")
            continue
        by_id, by_key = {}, {}
        for p in files:
            for line in p.open(encoding="utf-8"):
                s = json.loads(line)
                if s["instance_id"] not in ids:
                    continue
                by_id[s["record_id"]] = s
                prev = by_key.setdefault((s["instance_id"], s["answer_key"]), s["correct"])
                assert prev == s["correct"], ("one answer both right and wrong", s["record_id"])
        for r in rows:
            iid, tail = r["instance_id"], f"::{r['mode']}" + (f"::{r['library']}" if r["library"] else "") + f"::{r['model']}"
            ref_id = iid + ("::canonical" if r["ref"] == "canonical" else "::relabel:identity") + tail
            own = [(ref_id, r["ref_key"])] + [(f["variant_id"] + tail, f["key"]) for f in r["forms"]]
            for rid, k in own:
                s = by_id[rid]
                assert s["answer_key"] == k and s["correct"] == is_correct(iid, k), rid
                records += 1
            for k in r["repeats"]:
                if (iid, k) in by_key:
                    assert by_key[(iid, k)] == is_correct(iid, k), (iid, k)
                    checked += 1
                else:
                    unmatched += 1
    if records:
        print(f"flip rule: correctness agrees with the scored `correct` on all {records} reference and form "
              f"answers and on {checked} repeat answers ({unmatched} repeat answers occur in no scored record)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", choices=RULES, default="literal",
                    help="literal: noise_floor.json's agreement rule; flip: the paper's flip rule (see above)")
    ap.add_argument("--bootstrap", type=int, default=10000, help="graph-bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=20260928, help="bootstrap seed (default 20260928, as committed)")
    ap.add_argument("--out-dir", default=str(FLOOR))
    args = ap.parse_args()
    is_correct = Correctness.from_configs() if args.rule == "flip" else None
    df = pd.concat([load(FLOOR / ds / "noise_floor_instances.jsonl", args.rule, is_correct)
                    for ds in DATASETS])
    cells, pooled, modes = analyse(df, args.bootstrap, args.seed)
    if args.rule == "literal":
        check_against_json(cells)
    else:
        check_against_scored(is_correct)
    out = Path(args.out_dir)
    paths = {name: out / f"floor_stats_{args.rule}_{name}.csv" for name in ("cells", "pooled", "modes")}
    cells.to_csv(paths["cells"], index=False, float_format="%.6g")
    pooled.to_csv(paths["pooled"], index=False, float_format="%.6g")
    modes.to_csv(paths["modes"], index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    print(f"\n[{args.rule} rule] {len(cells)} cells | Wald CI excludes 0: {cells.wald_sig.sum()}"
          f" | mid-p < .05: {cells.midp_sig.sum()} | mid-p BH q < .05: {cells.bh_sig.sum()}"
          f" | mid-p Holm: {cells.holm.sum()} | graph bootstrap CI excludes 0: {cells.boot_sig.sum()}"
          f" | clustered McNemar < .05: {(cells.p_clustered < .05).sum()}")
    print("\nby model (cells significant of 12): Wald / mid-p / BH / graph bootstrap")
    for m, g in cells.groupby("model", sort=False):
        print(f"  {m:26s} {g.wald_sig.sum():2d} / {g.midp_sig.sum():2d} / {g.bh_sig.sum():2d} / {g.boot_sig.sum():2d}")
    print("\npooled per model (mean excess over cells, graph-bootstrap 95% CI; pairwise u beside it):")
    cols = ["model", "scope", "level", "cells", "floor", "flip", "excess", "boot_lo", "boot_hi", "p_boot",
            "u", "u_lo", "u_hi"]
    print(pooled[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nper dataset x arm, both axes (%; floor -> flip rate on the same instances -> excess [graph-bootstrap 95% CI]):")
    for r in modes.itertuples(index=False):
        print(f"  {r.dataset:8s} {r.arm:14s} {r.model:24s} {100 * r.floor:6.2f} -> {100 * r.flip:6.2f} -> "
              f"{100 * r.excess:+6.2f} [{100 * r.boot_lo:+5.1f}, {100 * r.boot_hi:+5.1f}]")
    print("\nwritten " + ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    main()
