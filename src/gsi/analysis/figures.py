from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gsi.analysis.tables import CLASS_ORDER, RUNG_ORDER

MODE_ORDER = ["direct", "code", "graph_as_code"]
AXIS_ORDER = ["canonical", "relabel", "order", "structure", "syntax"]


def _order(values, order):
    return [x for x in order if x in set(values)]


def _arm(df: pd.DataFrame) -> pd.Series:
    """Plot label for a (mode, library) cell. The two are separate columns in the data;
    they are joined only here, for an axis tick."""
    if "library" not in df.columns:
        return df["mode"]
    lib = df["library"].fillna("-")
    return df["mode"].where(lib.isin(["-", ""]), df["mode"] + "/" + lib)


def _arm_order(values) -> list:
    seen = list(dict.fromkeys(values))
    return sorted(seen, key=lambda a: (MODE_ORDER.index(a.split("/")[0]) if a.split("/")[0] in MODE_ORDER else 99, a))


def accuracy_heatmaps(acc: pd.DataFrame, out: Path) -> None:
    for (dataset, model), sub in acc.groupby(["dataset", "model"]):
        sub = sub.assign(arm=_arm(sub))
        piv = sub.pivot_table(index=["task", "arm"], columns="axis", values="accuracy")
        piv = piv[_order(piv.columns, AXIS_ORDER)]
        fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 3, 0.35 * len(piv) + 1.5))
        im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv)), [f"{t} / {m}" for t, m in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v < 0.5 else "black")
        fig.colorbar(im, ax=ax, label="accuracy")
        ax.set_title(f"Accuracy by task x arm x axis - {dataset}, {model}")
        fig.tight_layout()
        fig.savefig(out / f"accuracy_{dataset}_{model}.png", dpi=150)
        plt.close(fig)


def invariance_bars(inv: pd.DataFrame, out: Path) -> None:
    for (dataset, model), sub in inv.groupby(["dataset", "model"]):
        sub = sub.assign(arm=_arm(sub))
        piv = sub.groupby(["axis", "arm"])["frac_identical"].mean().unstack("arm")
        piv = piv.loc[_order(piv.index, AXIS_ORDER), _arm_order(piv.columns)]
        ax = piv.plot.bar(figsize=(1.6 * len(piv) + 3, 4), ylim=(0, 1.05), rot=0)
        ax.set_ylabel("fraction of graphs answered identically")
        ax.set_title(f"Per-instance invariance by perturbation axis - {dataset}, {model}")
        ax.legend(title="mode / library", fontsize=8)
        ax.figure.tight_layout()
        ax.figure.savefig(out / f"invariance_{dataset}_{model}.png", dpi=150)
        plt.close(ax.figure)


def failure_stacks(fail: pd.DataFrame, out: Path) -> None:
    if fail.empty:
        return
    classes = [c for c in CLASS_ORDER if c in fail.columns]
    for (dataset, model), sub in fail.groupby(["dataset", "model"]):
        # only the classes this model has: the combined and the per-model runs write the same file
        cols = [c for c in classes if sub[c].fillna(0).any()]
        sub = sub.assign(arm=_arm(sub)).set_index(["arm", "axis"])[cols]
        ax = sub.plot.bar(stacked=True, figsize=(0.9 * len(sub) + 3, 4), rot=45)
        ax.set_ylabel("share of failures")
        ax.set_title(f"Failure decomposition among code-mode failures - {dataset}, {model}")
        ax.legend(fontsize=8)
        ax.figure.tight_layout()
        ax.figure.savefig(out / f"failures_{dataset}_{model}.png", dpi=150)
        plt.close(ax.figure)


def gap_bars(gap: pd.DataFrame, out: Path) -> None:
    if gap.empty:
        return
    gap_cols = [c for c in gap.columns if c.startswith("gap_")]
    for (dataset, model), sub in gap.groupby(["dataset", "model"]):
        for col in gap_cols:
            piv = sub.pivot_table(index=["task", "library"] if "library" in sub.columns else "task",
                                  columns="axis", values=col)
            if piv.empty:
                continue
            ax = piv.plot.bar(figsize=(0.8 * len(piv) + 3, 4), rot=45)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("M2 - M3 invariance gap")
            ax.set_title(f"Transcription cost ({col[4:]}) - {dataset}, {model}")
            ax.figure.tight_layout()
            ax.figure.savefig(out / f"{col}_{dataset}_{model}.png", dpi=150)
            plt.close(ax.figure)


def ladder_bars(ladder: pd.DataFrame, out: Path) -> None:
    if ladder.empty:
        return
    classes = [c for c in CLASS_ORDER if c in ladder.columns]
    for (dataset, model), sub in ladder.groupby(["dataset", "model"]):
        sub = sub.copy()
        sub["arm"] = _arm(sub)
        sub["rung"] = pd.Categorical(sub["rung"], [r for r in RUNG_ORDER if r in set(sub["rung"])], ordered=True)
        sub = sub.sort_values(["arm", "rung"]).set_index(["arm", "rung"])
        ax = sub[classes].plot.bar(stacked=True, figsize=(0.9 * len(sub) + 3, 4), rot=45)
        ax.plot(range(len(sub)), sub["accuracy"].values, "k.-", label="accuracy")
        ax.set_ylabel("share of records")
        ax.set_title(f"Transcription ladder: prose -> json -> networkx -> injected - {dataset}, {model}")
        ax.legend(fontsize=8)
        ax.figure.tight_layout()
        ax.figure.savefig(out / f"ladder_{dataset}_{model}.png", dpi=150)
        plt.close(ax.figure)


def all_figures(tables: dict[str, pd.DataFrame], out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not tables["accuracy"].empty:
        accuracy_heatmaps(tables["accuracy"], out)
    if not tables["invariance"].empty:
        invariance_bars(tables["invariance"], out)
    failure_stacks(tables["failures"], out)
    gap_bars(tables["gap"], out)
    ladder_bars(tables.get("ladder", pd.DataFrame()), out)
