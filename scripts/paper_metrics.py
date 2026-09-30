"""The paper's metric set, computed from the scored records of all four models.

    python scripts/paper_metrics.py [--out PATH]   ->  results/review/paper_metrics.json

Inputs
  scored records, results/<dataset>/scored/<model>.jsonl (git-ignored; scripts/restore_from_hf.py writes them).
  data/processed/<dataset>/instances.jsonl gives the true answers, edge counts, graphs and graph ids.

Units
  answer = one serialization of one instance, one model, one mode
  tuple  = the reference serialization + one factor's variants (as in scripts/audit_results.py):
           relabel -> the sorted identity + the relabelings; other factors -> canonical + that factor's variants
Same-answer rules: node answers are compared after mapping back (answer_key); two correct answers are the
same answer; "no answer" (crash, timeout, unparseable) is an answer of its own.

Metrics
  performance  accuracy (no answer = wrong), robust accuracy (every serialization correct), drop vs reference
  invariance   six buckets; flip rate (D > 0); D = share of serialization pairs that differ; severity = mean D
               among flips; weighted flip rate = mean D; S = (max - min) / |truth| over a flip's answered values
               (flipped tuples of numeric tasks)
  localization (a) wrong M2 answers by the first step that failed (copy -> solve -> run), as shares among
                   the wrong answers, with their rate
               (b) flipped M2 tuples by where they diverge: copy / crash / program
               (c) flip rate of M2 vs M3 on the same tuples, relabel and order only
  also         flip rate by graph size, and whether flipped node-list answers take the other reading
Statistics: McNemar (flip, code vs direct; M2 vs M3), Wilcoxon signed-rank (D, code vs direct), cluster
bootstrap over graphs (headline rates, and the M1 / pooled-M2 rates per model drawn in the figures), logistic
regressions clustered by graph, one per dataset. Datasets are never pooled. McNemar and Wilcoxon treat a
model's tuples as independent pairs; tuples of one graph are not.
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests

from gsi.analysis.tuples import (READING_TASKS, crash_only_flip, divergence_site, durkalski, failure_step, is_double,
                                 like_for_like, reading_tag, spread)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["graphqa", "erdos"]
MODELS = ["hf-qwen3-8b-nscale", "hf-deepseek-v3.1-novita", "hf-gemma-4-31b", "hf-deepseek-v4-flash"]
COLS = ["dataset", "task", "instance_id", "variant_id", "axis", "params", "mode", "library", "model",
        "answer_key", "parsed", "correct", "failure_class", "declared_ok", "prompt_hash"]
RNG = np.random.default_rng(0)
ARMS_M12 = ["direct", "code/native", "code/networkx"]
FACTORS = ["relabel", "order", "structure", "syntax"]
NUMERIC = {"node_degree", "node_count", "edge_count", "triangles", "degree", "edge_number",
           "connected_component_number", "diameter", "density"}

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", default=str(ROOT / "results/review/paper_metrics.json"))
args = ap.parse_args()

miss = lambda v: v is None or (isinstance(v, float) and math.isnan(v))


def read_scored(run):
    """The four models' scored records of one run (results/<run>/scored), restricted to COLS."""
    rows = []
    for m in MODELS:
        p = ROOT / "results" / run / "scored" / f"{m}.jsonl"
        if not p.exists():
            raise SystemExit(f"no scored records at {p}; run scripts/restore_from_hf.py")
        with open(p) as fh:
            rows += [{k: r.get(k) for k in COLS} for r in map(json.loads, fh)]
    return rows


def answer_frame(rows):
    j = pd.DataFrame(rows)
    j["arm"] = np.where(j["mode"] == "direct", "direct", j["mode"] + "/" + j["library"].astype(str))
    j["answered"] = ~j["parsed"].map(miss)
    j["correct"] = j["correct"].astype(bool)
    j["is_identity"] = j["params"].map(lambda p: (p or {}).get("seed") == "identity")
    return j


j = answer_frame([r for ds in DATASETS for r in read_scored(ds)])

inst = {}
for ds in DATASETS:
    for line in open(ROOT / "data/processed" / ds / "instances.jsonl"):
        r = json.loads(line)
        inst[r["id"]] = r
truth = {i: r["ground_truth"] for i, r in inst.items()}
cluster = {i: r["meta"].get("source_file") or r["meta"]["graph_id"] for i, r in inst.items()}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- tuples
def build_tuples(j):
    """One row per tuple, and the flipped code tuples of the node-list tasks (the "other reading" analysis)."""
    rows, reading = [], []
    for (iid, model, arm), g in j.groupby(["instance_id", "model", "arm"], sort=False):
        canon = g[g["axis"] == "canonical"]
        ds, task = g["dataset"].iat[0], g["task"].iat[0]
        for axis in FACTORS:
            if arm.startswith("graph_as_code") and axis not in ("relabel", "order"):
                continue
            var = g[g["axis"] == axis]
            sub = var if (axis == "relabel" and var["is_identity"].any()) else pd.concat([canon, var])
            if sub["variant_id"].nunique() < 2:
                continue
            keys = [k if a else "NA" for k, a in zip(sub["answer_key"], sub["answered"])]
            corr = sub["correct"].tolist()
            vals = ["CORRECT" if c else k for k, c in zip(keys, corr)]
            n = len(vals)
            pairs = list(combinations(vals, 2))
            D = sum(a != b for a, b in pairs) / len(pairs)
            flip = D > 0
            all_answered = "NA" not in keys
            literal_same = all_answered and len(set(keys)) == 1
            if literal_same and all(corr):
                bucket = "same_correct"
            elif literal_same:
                bucket = "same_wrong"
            elif all(corr):
                bucket = "different_all_correct"
            elif all(k == "NA" for k in keys):
                bucket = "no_answer_all"
            elif flip and all_answered:
                bucket = "flip_all_answered"
            else:
                bucket = "flip_with_crash"
            assert flip == (bucket in ("flip_all_answered", "flip_with_crash")), (iid, arm, axis, vals)
            # S: how far apart a flip's answers are; flipped tuples of numeric tasks, answered values only
            S = doubled = None
            if flip and task in NUMERIC:
                xs = [x for x in (num(k) for k in keys if k != "NA") if x is not None]
                S, doubled = spread(xs, num(truth.get(iid)))
            # (b) where a flipped M2 tuple diverges
            where = None
            if flip and arm.startswith("code/"):
                bad = (sub["declared_ok"] == False).tolist()  # noqa: E712  (None = not known to be wrong)
                where = divergence_site(vals, bad, all_answered)
            if flip and arm.startswith("code/") and axis in ("relabel", "order") and task in READING_TASKS:
                reading.append(dict(dataset=ds, task=task, model=model, instance_id=iid, with_crash=not all_answered,
                                    answers=[json.loads(k) for k in keys if k != "NA"]))
            # a NetworkX syntax flip that only the NetworkX-code serialization's crash makes (Section 5.2)
            nx_crash = None
            if flip and arm == "code/networkx" and axis == "syntax":
                nx_crash = crash_only_flip(keys, sub["variant_id"].tolist(), "::syntax:networkx_code")
            rows.append(dict(dataset=ds, task=task, instance_id=iid, cluster=cluster.get(iid, iid), model=model,
                             arm=arm, axis=axis, n=n, bucket=bucket, flip=flip, D=D, robust=all(corr),
                             S=S, doubled=doubled, where=where, nx_crash=nx_crash))
    return pd.DataFrame(rows), reading


T, reading = build_tuples(j)
M12 = T[T.arm.isin(ARMS_M12)]
PERM = ["relabel", "order"]


def fam(df):
    """Rows for each factor plus the relabel+order pool."""
    return pd.concat([df.assign(factor=df.axis), df[df.axis.isin(PERM)].assign(factor="relabel+order")])


out = {"notes": __doc__}

# ---------------------------------------------------------------- invariance
F = fam(M12)
inv = {}
for (ds, arm, fac), d in F.groupby(["dataset", "arm", "factor"]):
    flips = d[d.flip]
    b = d.bucket.value_counts(normalize=True).mul(100).round(1).to_dict()
    s = d[d.S.notna()]
    inv[f"{ds}|{arm}|{fac}"] = dict(
        tuples=int(len(d)), flip_rate=round(100 * d.flip.mean(), 1), severity=round(flips.D.mean(), 2) if len(flips) else None,
        weighted_flip=round(100 * d.D.mean(), 1), robust_accuracy=round(100 * d.robust.mean(), 1), buckets=b,
        S_n=int(len(s)), S_median=round(s.S.median(), 2) if len(s) else None,
        S_doubling_pct=round(100 * s.doubled.mean(), 1) if len(s) else None,
        one_off_pct_of_4member_flips=round(100 * ((flips.n == 4) & (flips.D == 0.5)).sum() / max((flips.n == 4).sum(), 1), 1),
        all_different_pct_of_4member_flips=round(100 * ((flips.n == 4) & (flips.D == 1)).sum() / max((flips.n == 4).sum(), 1), 1))
out["invariance"] = inv

# Section 5.2: NetworkX code's syntax flips caused by the two crashes of the NetworkX-code serialization (Qwen3-8B
# leaves out the import, DeepSeek-V4-Flash uses a G it never defines): that serialization gives no answer and every
# other one is answered. The rate without them keeps the four models weighted equally.
CRASH_MODELS = ("hf-qwen3-8b-nscale", "hf-deepseek-v4-flash")
nx_crash_out = {}
for ds in DATASETS:
    sx = T[(T.dataset == ds) & (T.arm == "code/networkx") & (T.axis == "syntax")]
    crash = sx.flip & (sx.nx_crash == True) & sx.model.isin(CRASH_MODELS)  # noqa: E712
    nx_crash_out[ds] = dict(
        tuples=int(len(sx)), flips=int(sx.flip.sum()), crash_flips=int(crash.sum()),
        crash_flips_by_model={m: int(crash[sx.model == m].sum()) for m in CRASH_MODELS},
        crash_share_of_flips_pct=round(100 * crash.sum() / sx.flip.sum(), 1),
        flip_rate_pct=round(100 * sx.flip.groupby(sx.model).mean().mean(), 1),
        flip_rate_without_crashes_pct=round(100 * (sx.flip & ~crash).groupby(sx.model).mean().mean(), 1))
out["networkx_syntax_crash_flips"] = nx_crash_out

# per model flip rates (the spread across models), relabel+order
pm = F[F.factor == "relabel+order"].groupby(["dataset", "arm", "model"]).flip.mean().mul(100).round(1)
out["flip_rate_per_model_relabel+order"] = {f"{a}|{b}|{c}": v for (a, b, c), v in pm.items()}

# ---------------------------------------------------------------- performance
R = j[j["mode"].isin(["direct", "code"])]
perf = {}
for (ds, arm), g in R.groupby(["dataset", "arm"]):
    ref_canon = g[g.axis == "canonical"].correct.mean()
    ref_ident = g[(g.axis == "relabel") & g.is_identity].correct.mean()
    for fac in FACTORS:
        if fac == "relabel":
            var, ref = g[(g.axis == "relabel") & ~g.is_identity], ref_ident
        else:
            var, ref = g[g.axis == fac], ref_canon
        perf[f"{ds}|{arm}|{fac}"] = dict(answers=int(len(var)), accuracy=round(100 * var.correct.mean(), 1),
                                         reference_accuracy=round(100 * ref, 1),
                                         drop=round(100 * (var.correct.mean() - ref), 1))
    perf[f"{ds}|{arm}|all"] = dict(answers=int(len(g)), accuracy=round(100 * g.correct.mean(), 1))
out["performance"] = perf

# ---------------------------------------------------------------- localization (a)
CODE = j[j["mode"] == "code"].copy()
# A failure is a wrong answer (as in accuracy), assigned the first step that failed (copy -> solve -> run)
CODE["step"] = [failure_step(fc, dok, c) for fc, dok, c in zip(CODE.failure_class, CODE.declared_ok, CODE.correct)]
CODE["factor"] = CODE.axis.replace({"canonical": "reference", "relabel": "relabel+order", "order": "relabel+order"})
loc_a = {}
for (ds, lib, fac), g in CODE.groupby(["dataset", "library", "factor"]):
    fails = g[g.step.notna()]
    shares = fails.step.value_counts(normalize=True).mul(100).round(1).to_dict()
    loc_a[f"{ds}|{lib}|{fac}"] = dict(answers=int(len(g)), failures=int(len(fails)),
                                      failure_rate=round(100 * len(fails) / len(g), 1),
                                      shares_among_failures={k: shares.get(k, 0.0) for k in ["copy", "solve", "run", "other"]},
                                      counts={k: int((fails.step == k).sum()) for k in ["copy", "solve", "run", "other"]})
out["localization_a"] = loc_a

# ---------------------------------------------------------------- localization (b) (Figure B4)
FB = fam(T[T.arm.str.startswith("code/") & T.flip])
loc_b = {}
for (ds, arm, fac), g in FB.groupby(["dataset", "arm", "factor"]):
    loc_b[f"{ds}|{arm}|{fac}"] = dict(flips=int(len(g)), **{k: round(100 * (g["where"] == k).mean(), 1)
                                                         for k in ["program", "crash", "copy"]})
bm = FB[FB.factor == "relabel+order"].groupby(["dataset", "arm", "model"])["where"].value_counts().unstack(fill_value=0)
out["localization_b_by_model_relabel+order"] = {f"{a}|{b}|{c}": {k: int(v) for k, v in r.items()} for (a, b, c), r in bm.iterrows()}
out["localization_b"] = loc_b

# ---------------------------------------------------------------- localization (c)
# mcnemar_m2_vs_m3 pools the relabel and order tuples; mcnemar_m2_vs_m3_by_axis (at the end) tests each factor.
loc_c, mc_rows = {}, []
for lib in ["native", "networkx"]:
    a = T[T.arm == f"code/{lib}"].set_index(["dataset", "model", "instance_id", "axis"]).flip
    b = T[T.arm == f"graph_as_code/{lib}"].set_index(["dataset", "model", "instance_id", "axis"]).flip
    P = pd.concat([a.rename("m2"), b.rename("m3")], axis=1, join="inner").reset_index()
    for (ds, ax), g in P.groupby(["dataset", "axis"]):
        loc_c[f"{ds}|{lib}|{ax}"] = dict(tuples=int(len(g)), m2_flip=round(100 * g.m2.mean(), 1), m3_flip=round(100 * g.m3.mean(), 1))
    for (ds, model), g in P.groupby(["dataset", "model"]):
        m2o, m3o = int((g.m2 & ~g.m3).sum()), int((~g.m2 & g.m3).sum())
        p = binomtest(m2o, m2o + m3o, 0.5).pvalue if m2o + m3o else 1.0
        mc_rows.append(dict(dataset=ds, model=model, library=lib, only_m2=m2o, only_m3=m3o, p=p))
MC = pd.DataFrame(mc_rows)
MC["p_holm"] = multipletests(MC.p, method="holm")[1]
out["localization_c"] = loc_c
out["mcnemar_m2_vs_m3"] = MC.round(4).to_dict("records")

# ---------------------------------------------------------------- statistics: code vs direct
RO = M12[M12.axis.isin(PERM)]
key = ["dataset", "model", "instance_id", "axis"]
dirx = RO[RO.arm == "direct"].set_index(key)[["flip", "D"]]
mcn, wil = [], []
for lib in ["native", "networkx"]:
    code = RO[RO.arm == f"code/{lib}"].set_index(key)[["flip", "D"]]
    P = dirx.join(code, lsuffix="_dir", rsuffix="_code", how="inner").reset_index()
    for (ds, model), g in P.groupby(["dataset", "model"]):
        od, oc = int((g.flip_dir & ~g.flip_code).sum()), int((~g.flip_dir & g.flip_code).sum())
        mcn.append(dict(dataset=ds, model=model, library=lib, only_direct=od, only_code=oc,
                        p=binomtest(od, od + oc, 0.5).pvalue if od + oc else 1.0))
        diff = g.D_dir - g.D_code
        nz = diff[diff != 0]
        p = wilcoxon(g.D_dir, g.D_code, zero_method="wilcox").pvalue if len(nz) else 1.0
        wil.append(dict(dataset=ds, model=model, library=lib, pairs=int(len(g)), direct_worse=int((nz > 0).sum()),
                        code_worse=int((nz < 0).sum()), p=p))
    for ds, g in P.groupby("dataset"):
        diff = g.D_dir - g.D_code
        out[f"wilcoxon_pooled_models|{ds}|{lib}"] = dict(pairs=int(len(g)), direct_worse=int((diff > 0).sum()),
                                                         code_worse=int((diff < 0).sum()),
                                                         p=float(wilcoxon(g.D_dir, g.D_code, zero_method="wilcox").pvalue))
for name, rows_ in [("mcnemar_code_vs_direct", mcn), ("wilcoxon_D_code_vs_direct", wil)]:
    df = pd.DataFrame(rows_)
    df["p_holm"] = multipletests(df.p, method="holm")[1]
    better = "only_direct" if name.startswith("mcnemar") else "direct_worse"
    worse = "only_code" if name.startswith("mcnemar") else "code_worse"
    out[name] = df.round(4).to_dict("records")
    out[name + "_summary"] = dict(cells=len(df), code_better_sig=int(((df.p_holm < 0.05) & (df[better] > df[worse])).sum()),
                                  direct_better_sig=int(((df.p_holm < 0.05) & (df[better] < df[worse])).sum()))

# ---------------------------------------------------------------- bootstrap CIs (headline, relabel+order)
boot = {}
for ds, g in RO.groupby("dataset"):
    cl = g.cluster.unique()
    agg = g.groupby(["cluster", "arm"]).agg(flips=("flip", "sum"), dsum=("D", "sum"), n=("flip", "size")).unstack("arm")
    reps = {a: {"flip": [], "weighted": []} for a in ARMS_M12}
    reps.update({f"diff|{a}": {"flip": [], "weighted": []} for a in ARMS_M12[1:]})
    for _ in range(2000):
        s = agg.loc[RNG.choice(cl, len(cl), replace=True)].sum()
        vals = {a: (s[("flips", a)] / s[("n", a)], s[("dsum", a)] / s[("n", a)]) for a in ARMS_M12}
        for a in ARMS_M12:
            reps[a]["flip"].append(vals[a][0]); reps[a]["weighted"].append(vals[a][1])
        for a in ARMS_M12[1:]:
            reps[f"diff|{a}"]["flip"].append(vals["direct"][0] - vals[a][0])
            reps[f"diff|{a}"]["weighted"].append(vals["direct"][1] - vals[a][1])
    for a, r in reps.items():
        boot[f"{ds}|{a}"] = {m: [round(100 * float(np.percentile(v, 2.5)), 1), round(100 * float(np.percentile(v, 97.5)), 1)]
                             for m, v in r.items()}
out["bootstrap_95ci_relabel+order"] = boot

# ---------------------------------------------------------------- logistic regression (flip), clustered by graph
# One model per dataset (datasets are never pooled). Main effects only: a factor's odds ratio is averaged
# over the modes, and a mode's over the factors.
out["logit_flip"], out["logit_flip_n"] = {}, {}
for ds, L in M12.assign(y=M12.flip.astype(int)).groupby("dataset", sort=False):
    fit = smf.glm("y ~ C(arm, Treatment('direct')) + C(axis, Treatment('relabel')) + C(model) + C(task)", data=L,
                  family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(L.cluster)[0]})
    ci = fit.conf_int()
    out["logit_flip"][ds] = {k: [round(float(np.exp(fit.params[k])), 2), round(float(np.exp(ci.loc[k, 0])), 2),
                                 round(float(np.exp(ci.loc[k, 1])), 2)] for k in fit.params.index if "arm" in k or "axis" in k}
    out["logit_flip_n"][ds] = dict(tuples=int(len(L)), clusters=int(L.cluster.nunique()))

# ---------------------------------------------------------------- charts: how flips split (D) and how far apart (S)
prof = {}
F4 = M12[M12.axis.isin(PERM) & M12.flip & (M12.n == 4)]
for (ds, arm), g in F4.groupby(["dataset", "arm"]):
    vc = g.D.round(2).value_counts(normalize=True).mul(100)
    prof[f"{ds}|{arm}"] = dict(flips=int(len(g)), mean_D=round(float(g.D.mean()), 2),
                               **{f"D={k:.2f}": round(float(vc.get(k, 0.0)), 1) for k in [0.5, 0.67, 0.83, 1.0]})
out["severity_profile_4member_relabel+order"] = prof
sq = {}
for (ds, arm, fac), g in fam(M12).groupby(["dataset", "arm", "factor"]):
    s = g.S.dropna()
    if len(s):
        sq[f"{ds}|{arm}|{fac}"] = dict(n=int(len(s)), q1=round(float(s.quantile(.25)), 2),
                                       median=round(float(s.median()), 2), q3=round(float(s.quantile(.75)), 2))
out["S_quartiles"] = sq

# ---------------------------------------------------------------- graph size (Figure 2)
P = M12[M12.axis.isin(PERM)].copy()
P["m"] = P.instance_id.map(lambda i: len(inst[i]["edges"]))
P["bin"] = pd.cut(P.m, bins=[-1, 10, 20, 40, 10**6], labels=["up to 10", "11 to 20", "21 to 40", "over 40"])
G_ = P.groupby(["dataset", "bin", "arm"], observed=True).flip.agg(["sum", "size"]).reset_index()
out["flip_rate_by_edges_relabel+order"] = {f"{r.dataset}|{r.bin}|{r.arm}": dict(flips=int(r["sum"]), tuples=int(r["size"]),
                                            flip_rate=round(100 * r["sum"] / r["size"], 1)) for _, r in G_.iterrows()}
P["log2m"] = np.log2(P.m + 1)  # log2(edges + 1): one unit = a doubling of (edges + 1)
P["y"] = P.flip.astype(int)
P["arm"] = pd.Categorical(P.arm, ARMS_M12)
size_or = {}
for ds, g in P.groupby("dataset", sort=False):  # one model per dataset
    fit = smf.glm("y ~ C(arm) * log2m + C(task) + C(model)", data=g, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(g.cluster)[0]})
    size_or[ds] = {}
    for arm, terms in [("direct", ["log2m"]), ("code/native", ["log2m", "C(arm)[T.code/native]:log2m"]),
                       ("code/networkx", ["log2m", "C(arm)[T.code/networkx]:log2m"])]:
        v = pd.Series(0.0, index=fit.params.index)
        v[terms] = 1.0
        est, se = float(v @ fit.params), float(np.sqrt(v @ fit.cov_params() @ v))
        size_or[ds][arm] = [round(math.exp(est), 2), round(math.exp(est - 1.96 * se), 2), round(math.exp(est + 1.96 * se), 2)]
out["flip_odds_ratio_per_doubling_of_edges"] = size_or
wt = []
for (ds, task), g in P.groupby(["dataset", "task"]):
    if g.m.nunique() < 3:
        continue
    lo, hi = g.m.quantile(1 / 3), g.m.quantile(2 / 3)
    for arm in ["direct", "code/native"]:
        a = g[g.arm == arm]
        wt.append(dict(arm=arm, rise=100 * (a[a.m >= hi].flip.mean() - a[a.m <= lo].flip.mean())))
W = pd.DataFrame(wt)
out["within_task_rise_largest_vs_smallest_third"] = {a: dict(tasks=int((W.arm == a).sum()), rising=int((W[W.arm == a].rise > 0).sum()),
                                                            median_points=round(float(W[W.arm == a].rise.median()), 1))
                                                     for a in ["direct", "code/native"]}

# ---------------------------------------------------------------- other reading (Figure B5)
# Flipped M2 tuples (relabel and order, both libraries) of the node-list questions: does some answer in the
# tuple equal the other reading's answer? GraphQA's "connected to" questions (undirected graphs) read as either
# adjacency or reachability; Erdős `neighbor` means the successors on a directed graph and the neighbours on an
# undirected one. Tags use the answers that exist; `with_crash` counts the flips in which some serialization
# gave no answer (they are tagged on the rest); the paper counts only the others (`..._all_answered`).
alt = {}
for iid, r in inst.items():
    if r["task"] in READING_TASKS and r["query_args"].get("node") is not None:
        Gx = nx.DiGraph() if r["directed"] else nx.Graph()
        Gx.add_nodes_from(r["nodes"])
        Gx.add_edges_from(tuple(e[:2]) for e in r["edges"])
        q = r["query_args"]["node"]
        N, V = set(Gx.successors(q) if r["directed"] else Gx.neighbors(q)), set(Gx.nodes)
        R = None if r["directed"] else nx.node_connected_component(Gx, q) - {q}
        alt[iid] = {"connected_nodes": (N, R), "disconnected_nodes": (V - N - {q}, V - R - {q}) if R is not None else None,
                    "neighbor": (N, None)}[r["task"]]
TG = pd.DataFrame([dict(t, tag=reading_tag(t["task"], [set(a) for a in t["answers"] if isinstance(a, list)],
                                             *alt[t["instance_id"]])) for t in reading])
reading_out = {}
for (d, t, m), g in TG.groupby(["dataset", "task", "model"]):
    reading_out[f"{d}|{t}|{m}"] = dict(flips=int(len(g)), with_crash=int(g.with_crash.sum()),
                                       **{k: int(v) for k, v in g.tag.value_counts().items()})
out["other_reading_code_flips"] = reading_out
# The paper (Section 5.3, Figure B5) counts only the flips in which every serialization is answered.
TA = TG[~TG.with_crash]
out["other_reading_code_flips_all_answered"] = {
    f"{d}|{t}|{m}": dict(flips=int(len(g)), **{k: int(v) for k, v in g.tag.value_counts().items()})
    for (d, t, m), g in TA.groupby(["dataset", "task", "model"])}
QA = TA[(TA.dataset == "graphqa") & TA.task.isin(["connected_nodes", "disconnected_nodes"])]
out["other_reading_graphqa_all_answered"] = dict(flips=int(len(QA)),
                                                 takes_the_other_reading=int((QA.tag == "takes the other reading").sum()))

# ---------------------------------------------------------------- bootstrap CIs for the figures (relabel+order)
# Figures 1 and B1 pool M2 over its two libraries and the models; Figures B2-B3 keep the models apart. Same
# resampling as the headline CIs (a dataset's graphs with replacement, 2,000 times, a graph's tuples kept
# together), with its own generator so that every number above stays as it was.
RNG_FIG = np.random.default_rng(1)


def fig_rates(s):
    """Flip rate and weighted flip rate per (model or "all", mode) from summed cluster counts."""
    res = {}
    for md in ("M1", "M2"):
        f, d, n = (sum(s[(c, m, md)] for m in MODELS) for c in ("flips", "dsum", "n"))
        res[("all", md)] = (f / n, d / n)
        for m in MODELS:
            n_m = s[("n", m, md)]
            res[(m, md)] = (s[("flips", m, md)] / n_m, s[("dsum", m, md)] / n_m) if n_m else (np.nan, np.nan)
    return res


fig_ci = {}
for ds, g in RO.assign(mode2=np.where(RO.arm == "direct", "M1", "M2")).groupby("dataset"):
    cl = g.cluster.unique()
    agg = (g.groupby(["cluster", "model", "mode2"]).agg(flips=("flip", "sum"), dsum=("D", "sum"), n=("flip", "size"))
           .unstack(["model", "mode2"]).fillna(0))
    point = fig_rates(agg.sum())
    reps = {k: ([], []) for k in point}
    for _ in range(2000):
        for k, (fl, wt) in fig_rates(agg.loc[RNG_FIG.choice(cl, len(cl), replace=True)].sum()).items():
            reps[k][0].append(fl)
            reps[k][1].append(wt)
    for (who, md), (fl, wt) in reps.items():
        fig_ci[f"{ds}|{who}|{md}"] = {
            name: [round(100 * float(point[(who, md)][i]), 1), round(100 * float(np.nanpercentile(v, 2.5)), 1),
                   round(100 * float(np.nanpercentile(v, 97.5)), 1)]
            for i, (name, v) in enumerate([("flip", fl), ("weighted", wt)])}
out["figure_intervals_relabel+order"] = fig_ci  # {dataset|model or all|M1 or M2: {flip|weighted: [rate, lo, hi]}}

# ---------------------------------------------------------------- the Erdős M3 rerun
# Nothing from here on draws random numbers. results/erdos_m3fix reruns M3 on Erdős with the graph type stated in
# its prompts; M1, M2 and GraphQA have one run each. RUNS holds each run's answers and tuples: "erdos" is the main
# run (its M3 is the first M3 run), "erdos_rerun" holds M3 only.
j3 = answer_frame([r for r in read_scored("erdos_m3fix") if r["mode"] == "graph_as_code"])
T3 = build_tuples(j3)[0]
RUNS = {"graphqa": (j[j.dataset == "graphqa"], T[T.dataset == "graphqa"]),
        "erdos": (j[j.dataset == "erdos"], T[T.dataset == "erdos"]), "erdos_rerun": (j3, T3)}
ARMS = ARMS_M12 + ["graph_as_code/native", "graph_as_code/networkx"]


def sig_counts(df, a, b):
    """Cells in which a is significantly larger than b, and b than a (Holm-adjusted p < 0.05)."""
    s = df.p_holm < 0.05
    return int((s & (df[a] > df[b])).sum()), int((s & (df[a] < df[b])).sum())


# ---------------------------------------------------------------- localization (c) by axis
# mcnemar_m2_vs_m3 pools relabel and order. Here each factor is its own family of 16 cells (dataset x model x
# library, Holm within the family), once with the first Erdős M3 run and once with the rerun (GraphQA's cells are
# the same in both). first_run|relabel+order repeats mcnemar_m2_vs_m3.
def m2_vs_m3(t_m3, axes):
    res = []
    for lib in ["native", "networkx"]:
        a = T[(T.arm == f"code/{lib}") & T.axis.isin(axes)].set_index(key).flip
        b = t_m3[(t_m3.arm == f"graph_as_code/{lib}") & t_m3.axis.isin(axes)].set_index(key).flip
        pairs = pd.concat([a.rename("m2"), b.rename("m3")], axis=1, join="inner").reset_index()
        for (ds, model), g in pairs.groupby(["dataset", "model"]):
            m2o, m3o = int((g.m2 & ~g.m3).sum()), int((~g.m2 & g.m3).sum())
            res.append(dict(dataset=ds, model=model, library=lib, tuples=int(len(g)), only_m2=m2o, only_m3=m3o,
                            p=binomtest(m2o, m2o + m3o, 0.5).pvalue if m2o + m3o else 1.0))
    df = pd.DataFrame(res)
    df["p_holm"] = multipletests(df.p, method="holm")[1]
    return df


mc_axis, mc_axis_sum = {}, {}
for run, t_m3 in [("first_run", T), ("rerun", pd.concat([T[T.dataset == "graphqa"], T3]))]:
    for fac, axes in [("relabel", ["relabel"]), ("order", ["order"]), ("relabel+order", PERM)]:
        df = m2_vs_m3(t_m3, axes)
        if (run, fac) == ("first_run", "relabel+order"):
            assert df.drop(columns="tuples").equals(MC), "the pooled family must repeat mcnemar_m2_vs_m3"
        mc_axis[f"{run}|{fac}"] = df.round(4).to_dict("records")
        m2_more, m3_more = sig_counts(df, "only_m2", "only_m3")
        mc_axis_sum[f"{run}|{fac}"] = dict(cells=len(df), m2_more_sig=m2_more, m3_more_sig=m3_more)
out["mcnemar_m2_vs_m3_by_axis"] = mc_axis  # {first_run or rerun|factor: [cells]}
out["mcnemar_m2_vs_m3_by_axis_summary"] = mc_axis_sum


# ---------------------------------------------------------------- performance of every mode (Tables 3 and D1)
# accuracy: every serialization the mode answered; like_for_like_accuracy: the reference, relabelings and edge
# orders only (tuples.like_for_like), the serializations every mode is compared on; robust_accuracy and flip_rate:
# relabel+order tuples. Table 3 takes Erdős M3 from erdos_rerun, Table D1 from erdos.
def mode_summary(a, t):
    lfl, ro = a[a.axis.map(like_for_like)], t[t.axis.isin(PERM)]
    return dict(answers=int(len(a)), accuracy=round(100 * a.correct.mean(), 1), like_for_like_answers=int(len(lfl)),
                like_for_like_accuracy=round(100 * lfl.correct.mean(), 1), tuples=int(len(ro)),
                robust_accuracy=round(100 * ro.robust.mean(), 1), flip_rate=round(100 * ro.flip.mean(), 1))


out["performance_by_mode"] = {f"{run}|{arm}": mode_summary(a[a.arm == arm], t[t.arm == arm])
                              for run, (a, t) in RUNS.items() for arm in ARMS if (a.arm == arm).any()}
# GraphQA without edge_existence ("Is node u connected to node v?", which also reads as reachability)
qa, qt = (x[x.task != "edge_existence"] for x in RUNS["graphqa"])
out["performance_by_mode_graphqa_without_edge_existence"] = {arm: mode_summary(qa[qa.arm == arm], qt[qt.arm == arm])
                                                             for arm in ARMS}

# ---------------------------------------------------------------- M3 flip rate by factor
m3_flip = {}
for run, (_, t) in RUNS.items():
    for (arm, fac), d in fam(t[t.arm.str.startswith("graph_as_code")]).groupby(["arm", "factor"]):
        m3_flip[f"{run}|{arm.split('/')[1]}|{fac}"] = dict(tuples=int(len(d)), flips=int(d.flip.sum()),
                                                          flip_rate=round(100 * d.flip.mean(), 1))
out["m3_flip_rate"] = m3_flip  # {run|library|factor}

# ---------------------------------------------------------------- invariance per model (Tables B1 and B2)
BUCKETS = ["same_correct", "same_wrong", "different_all_correct", "flip_all_answered", "flip_with_crash",
           "no_answer_all"]
out["invariance_per_model"] = {
    f"{ds}|{arm}|{fac}|{model}": dict(tuples=int(len(d)), flip_rate=round(100 * d.flip.mean(), 1),
                                      buckets={b: round(100 * (d.bucket == b).mean(), 1) for b in BUCKETS})
    for (ds, arm, fac, model), d in fam(M12).groupby(["dataset", "arm", "factor", "model"])}
# The daggers of Tables 2 and B2: cells in which a model's code flips at least 1 point more often than its own
# direct answers (unrounded rates)
fr = M12.groupby(["dataset", "axis", "model", "arm"]).flip.mean().mul(100)
out["dagger_cells"] = [dict(dataset=ds, factor=fac, model=model, arm=arm, code_flip=round(v, 1),
                            direct_flip=round(fr[(ds, fac, model, "direct")], 1))
                       for (ds, fac, model, arm), v in fr.items()
                       if arm != "direct" and v - fr[(ds, fac, model, "direct")] >= 1]

# ---------------------------------------------------------------- tuple sizes (Table C1)
# Answers per tuple follow from the serializations alone, so every model and mode has the same sizes.
sizes = {}
for (ds, ax), d in M12.groupby(["dataset", "axis"]):
    prof = d.groupby(["model", "arm"]).n.apply(lambda s: tuple(sorted(s.value_counts().items(), reverse=True)))
    assert prof.nunique() == 1, (ds, ax, prof.unique())
    sizes[f"{ds}|{ax}"] = {str(n): int(c) for n, c in prof.iat[0]}
out["tuple_sizes"] = sizes  # {dataset|factor: {answers per tuple: tuples}}, per model and mode

# ---------------------------------------------------------------- McNemar code vs direct, clustered by graph
# mcnemar_code_vs_direct treats a model's tuples as independent; a graph's tuples are not. Durkalski's test
# sums each graph's discordant pairs (tuples.durkalski); same 16 cells, Holm over them.
dir_flip = RO[RO.arm == "direct"].set_index(key)[["flip", "cluster"]]
cl_rows = []
for lib in ["native", "networkx"]:
    pairs = dir_flip.join(RO[RO.arm == f"code/{lib}"].set_index(key)[["flip"]], rsuffix="_code", how="inner")
    pairs = pairs.assign(b=(pairs.flip & ~pairs.flip_code).astype(int), c=(~pairs.flip & pairs.flip_code).astype(int))
    for (ds, model), g in pairs.reset_index().groupby(["dataset", "model"]):
        per = g.groupby("cluster")[["b", "c"]].sum()
        x, p = durkalski(per.b.tolist(), per.c.tolist())
        cl_rows.append(dict(dataset=ds, model=model, library=lib, graphs=int(len(per)), only_direct=int(per.b.sum()),
                            only_code=int(per.c.sum()), statistic=x, p=p))
CL = pd.DataFrame(cl_rows)
CL["p_holm"] = multipletests(CL.p, method="holm")[1]
out["mcnemar_code_vs_direct_clustered"] = CL.round(4).to_dict("records")
code_better, direct_better = sig_counts(CL, "only_direct", "only_code")
out["mcnemar_code_vs_direct_clustered_summary"] = dict(cells=len(CL), code_better_sig=code_better,
                                                       direct_better_sig=direct_better)

# ---------------------------------------------------------------- the declared graph (M2)
# Share of M2 answers (both libraries) whose program declared the graph it was given (declared_ok True; an
# unknown counts as no match), and how many answers are correct although the declared graph is wrong.
M2 = j[j["mode"] == "code"]
out["declared_graph_match"] = {
    f"{ds}|{model}": dict(answers=int(len(g)), match_pct=round(100 * g.declared_ok.eq(True).mean(), 2))
    for (ds, model), g in M2.groupby(["dataset", "model"])}
out["correct_despite_wrong_declared_graph"] = {
    ds: dict(wrong_declared=int(len(g)), correct=int(g.correct.sum()), correct_pct=round(100 * g.correct.mean(), 1))
    for ds, g in M2[M2.declared_ok.eq(False)].groupby("dataset")}

# ---------------------------------------------------------------- doubled degrees under structure
# GraphQA node_degree answers on the two structure serializations that are exactly twice the true degree
# (tuples.is_double): the adjacency list, and the edge list with every edge in both directions.
nd = j[(j.task == "node_degree") & (j.axis == "structure") & j.arm.isin(ARMS_M12)]
nd = nd.assign(
    kind=np.where(nd.params.map(lambda p: bool((p or {}).get("replicated"))), "both_directions", "adjacency_list"),
    dbl=[is_double(ak if a else None, truth[i]) for ak, a, i in zip(nd.answer_key, nd.answered, nd.instance_id)])
out["doubled_node_degree_structure"] = {
    f"{model}|{arm}": dict(answers=int(len(g)), adjacency_list=int(g[g.kind == "adjacency_list"].dbl.sum()),
                           both_directions=int(g[g.kind == "both_directions"].dbl.sum()), doubled=int(g.dbl.sum()))
    for (model, arm), g in nd.groupby(["model", "arm"])}

# ---------------------------------------------------------------- flips and same-wrong tuples by task (relabel+order)
out["by_task_relabel+order"] = {
    f"{ds}|{arm}|{task}": dict(tuples=int(len(d)), flips=int(d.flip.sum()), flip_rate=round(100 * d.flip.mean(), 1),
                               same_wrong=int((d.bucket == "same_wrong").sum()))
    for (ds, arm, task), d in RO.groupby(["dataset", "arm", "task"])}
# GraphQA's "connected to" questions: edge_existence and the two node-list tasks
NODE_LIST = ["connected_nodes", "disconnected_nodes"]
CONNECTED = ["edge_existence"] + NODE_LIST
groups = {}
for arm, d in RO[RO.dataset == "graphqa"].groupby("arm"):
    sw, rest = d[d.bucket == "same_wrong"], d[~d.task.isin(NODE_LIST)]
    groups[arm] = dict(same_wrong=int(len(sw)), same_wrong_connected_tasks=int(sw.task.isin(CONNECTED).sum()),
                       flips=int(d.flip.sum()), flips_node_list_tasks=int(d[d.task.isin(NODE_LIST)].flip.sum()),
                       other_tasks_tuples=int(len(rest)), other_tasks_flip_rate=round(100 * rest.flip.mean(), 1))
out["graphqa_task_groups_relabel+order"] = groups

# ---------------------------------------------------------------- distinct prompts
# A repeated prompt is sent once per model and its reply reused: model_calls = distinct (model, prompt) pairs.
# In M3 a call is a program, and one program can answer many graphs.
# The Erdős rerun's count includes the directed graphs' prompts, which did not change and were answered from
# the first run's cache (150 per model).
dp = {}
for run, (a, _) in RUNS.items():
    m3 = a[a["mode"] == "graph_as_code"]
    calls = int(a.groupby("model").prompt_hash.nunique().sum())
    progs = int(m3.groupby("model").prompt_hash.nunique().sum())
    dp[run] = dict(records=int(len(a)), model_calls=calls, model_calls_pct=round(100 * calls / len(a), 1),
                   m3_records=int(len(m3)), m3_programs=progs, m3_programs_pct=round(100 * progs / len(m3), 1),
                   m3_answers_per_program=round(len(m3) / progs, 1))
out["distinct_prompts"] = dp

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1, default=float)
print(f"{len(T)} tuples ({len(M12)} direct and code) -> {args.out}")
