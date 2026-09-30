from __future__ import annotations

from pathlib import Path

import pandas as pd

from gsi.data.base import read_jsonl
from gsi.prompts.modes import CODE_MODES, INJECT_MODES

NUMERIC = ("int", "float")
# `library` is part of the key, not part of the mode name: it is crossed with the code
# modes. Drop it from a groupby to pool the arms, keep it to separate them.
KEYS = ["dataset", "task", "mode", "library", "axis", "model"]
NO_LIBRARY = "-"
RUNG_ORDER = ["prose", "plain", "json", "networkx_code", "injected"]
CLASS_ORDER = ["transcription", "construction", "logic", "no_computation", "execution", "format", "unverifiable"]
PROSE_SYNTAXES = ("graphqa_nl", "erdos_nl")


def _is_identity(params: dict) -> bool:
    return isinstance(params, dict) and params.get("seed") == "identity"


def load_scored(results_dir: str | Path, models: list[str] | None = None) -> pd.DataFrame:
    d = Path(results_dir) / "scored"
    files = [d / f"{m}.jsonl" for m in models] if models else sorted(d.glob("*.jsonl"))
    rows = [r for f in files for r in read_jsonl(f)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["parsed_num"] = pd.to_numeric(df["parsed_canonical"].where(df["answer_type"].isin(NUMERIC)), errors="coerce")
    # direct mode writes no code, so it has no library arm; a sentinel keeps it out of the
    # NaN-dropping behaviour of groupby rather than out of the tables.
    if "library" not in df.columns:
        df["library"] = NO_LIBRARY
    df["library"] = df["library"].fillna(NO_LIBRARY)
    return df


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """`n` counts scored records; `n_calls` counts the distinct model responses behind them.

    They differ wherever the response cache collapsed prompts -- most sharply for M3 on a
    task whose question names no node, where one reply is executed against every graph and
    every variant. A cell with n=100 and n_calls=1 carries one model decision, not a
    hundred, and must not be read as a hundred independent trials."""
    agg = {"accuracy": ("correct", "mean"), "n": ("correct", "size")}
    if "prompt_hash" in df.columns:
        agg["n_calls"] = ("prompt_hash", "nunique")
    return (df.groupby(KEYS).agg(**agg).reset_index()
              .sort_values(KEYS).reset_index(drop=True))


def _task_range(df: pd.DataFrame) -> pd.Series:
    gt = pd.to_numeric(df["ground_truth"].where(df["answer_type"].isin(NUMERIC)), errors="coerce")
    rng = gt.groupby([df["dataset"], df["task"]]).agg(lambda s: max(1.0, s.max() - s.min()))
    return rng


def invariance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-instance invariance: for each perturbation axis, is the answer identical on the
    canonical form and every variant of that axis? Numeric spread is (max-min)/task range."""
    rng = _task_range(df)
    identity = df["params"].apply(_is_identity)
    out = []
    for axis in sorted(a for a in df["axis"].unique() if a != "canonical"):
        if axis == "relabel" and identity.any():
            # Herbst convention: relabelings are compared against the sorted identity reference
            sub = df[df["axis"] == "relabel"].assign(is_ref=identity[df["axis"] == "relabel"])
        else:
            sub = df[df["axis"].isin(["canonical", axis])].assign(is_ref=lambda d: d["axis"] == "canonical")
        grp = sub.groupby(["dataset", "task", "instance_id", "mode", "library", "model"])
        per = grp.agg(
            n_variants=("variant_id", "nunique"),
            n_keys=("answer_key", "nunique"),
            has_ref=("is_ref", "any"),
            vmax=("parsed_num", "max"), vmin=("parsed_num", "min"),
            all_parsed=("parsed", lambda s: s.notna().all()),
            all_correct=("correct", "all"),
            # 1 means every variant in this cell came from the same cached reply, so the
            # instance was never actually re-asked and its invariance is trivially perfect
            n_calls=("prompt_hash", "nunique") if "prompt_hash" in sub.columns else ("variant_id", "nunique"),
        ).reset_index()
        per = per[(per["n_variants"] > 1) & per["has_ref"]]
        per["identical"] = (per["n_keys"] == 1) & per["all_parsed"]
        per["spread"] = (per["vmax"] - per["vmin"]) / per.set_index(["dataset", "task"]).index.map(rng).values
        per["axis"] = axis
        out.append(per)
    if not out:
        return pd.DataFrame(columns=KEYS + ["frac_identical", "mean_spread", "frac_all_correct",
                                            "n_instances", "n_calls"])
    per = pd.concat(out)
    return (per.groupby(KEYS).agg(frac_identical=("identical", "mean"), mean_spread=("spread", "mean"),
                                  frac_all_correct=("all_correct", "mean"), n_instances=("instance_id", "size"),
                                  n_calls=("n_calls", "sum"))
              .reset_index().sort_values(KEYS).reset_index(drop=True))


def _rung(row) -> str:
    if row["mode"] in INJECT_MODES:
        return "injected"
    syntax = row["params"].get("syntax") if isinstance(row["params"], dict) else None
    return "prose" if syntax in PROSE_SYNTAXES else (syntax or "prose")


def ladder_table(df: pd.DataFrame) -> pd.DataFrame:
    """The transcription ladder: the same code prompt with the graph as prose -> json ->
    networkx code -> injected data (no transcription). Only canonical labels/order are used,
    so each rung differs from the previous one in the graph text alone."""
    sub = df[df["mode"].isin(CODE_MODES) & df["axis"].isin(["canonical", "syntax"])].copy()
    sub = sub[~(sub["mode"].isin(INJECT_MODES) & (sub["axis"] == "syntax"))]
    if sub.empty:
        return pd.DataFrame()
    sub["rung"] = sub.apply(_rung, axis=1)
    agg = {"accuracy": ("correct", "mean"), "n": ("correct", "size")}
    for c in CLASS_ORDER:
        agg[c] = ("failure_class", lambda s, c=c: (s == c).mean())
    out = sub.groupby(["dataset", "mode", "library", "rung", "model"]).agg(**agg).reset_index()
    out["rung_order"] = out["rung"].map({r: i for i, r in enumerate(RUNG_ORDER)}).fillna(99)
    return (out.sort_values(["dataset", "model", "library", "mode", "rung_order"])
               .drop(columns="rung_order").reset_index(drop=True))


def failure_table(df: pd.DataFrame) -> pd.DataFrame:
    """Among code-mode failures, the share of each failure class per (mode, library, axis, model).

    Never pool across `library`: `construction` is only separable in the networkx arm, so
    the native arm reports those failures as `logic` and the two columns do not mean the
    same thing."""
    sub = df[df["mode"].isin(CODE_MODES) & (df["failure_class"] != "ok")]
    if sub.empty:
        return pd.DataFrame()
    ct = pd.crosstab([sub["dataset"], sub["mode"], sub["library"], sub["axis"], sub["model"]],
                     sub["failure_class"])
    ct["n_failures"] = ct.sum(axis=1)
    frac = ct.drop(columns="n_failures").div(ct["n_failures"], axis=0).round(3)
    frac["n_failures"] = ct["n_failures"]
    return frac.reset_index()


def gap_table(inv: pd.DataFrame) -> pd.DataFrame:
    """Invariance gap between the two code modes, within a library arm: M2 minus M3.

    The pivot is taken inside `library` so the two modes being compared differ in
    transcription alone -- which is the whole point of M3 being a minimal pair of M2."""
    if inv.empty:
        return inv
    piv = inv.pivot_table(index=["dataset", "task", "axis", "library", "model"],
                          columns="mode", values="frac_identical")
    if "code" not in piv or "graph_as_code" not in piv:
        return pd.DataFrame()
    piv["gap_code_minus_graph_as_code"] = piv["code"] - piv["graph_as_code"]
    # `direct` has no library arm and no M3 counterpart, so its rows carry no gap
    piv = piv.dropna(subset=["gap_code_minus_graph_as_code"])
    return piv.reset_index()


def silent_transcription_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of programs that transcribed the graph wrong, split by whether the answer still
    came out right. The right-hand side of that split is the silent failure: a wrong graph
    that produced a correct answer, which no accuracy number can see.

    Uses `declared_ok` (the template slots) and falls back to `graph_match` (the recovered
    networkx graph) for programs that ignored the template."""
    sub = df[df["mode"] == "code"].copy()
    if sub.empty:
        return pd.DataFrame()
    declared = sub["declared_ok"] if "declared_ok" in sub.columns else pd.Series(index=sub.index, dtype="object")
    signal = declared.where(declared.notna(), sub["graph_match"])
    sub = sub[signal.notna()].assign(wrong_graph=~signal[signal.notna()].astype(bool))
    if sub.empty:
        return pd.DataFrame()
    return (sub.groupby(["dataset", "mode", "library", "axis", "model", "correct"])["wrong_graph"]
              .agg(rate="mean", n="size").reset_index())


def all_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    inv = invariance_table(df)
    return {
        "accuracy": accuracy_table(df),
        "invariance": inv,
        "ladder": ladder_table(df),
        "failures": failure_table(df),
        "gap": gap_table(inv),
        "silent_transcription": silent_transcription_table(df),
    }
