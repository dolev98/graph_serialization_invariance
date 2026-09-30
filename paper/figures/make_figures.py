"""Draw the paper's figures as vector PDFs next to this file.

Every figure but the one-tuple example (fig1_tuple, whose answers are typed in) reads its numbers from
results/review/paper_metrics.json (scripts/paper_metrics.py), including every 95% interval:
graphs resampled with replacement 2,000 times, a graph's tuples kept together.

    python paper/figures/make_figures.py      # needs matplotlib
"""
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import json  # noqa: E402

OUT = Path(__file__).resolve().parent
METRICS = json.load(open(OUT.parents[1] / "results/review/paper_metrics.json"))
FIG_CI = METRICS["figure_intervals_relabel+order"]  # dataset|model or all|M1 or M2 -> {flip, weighted: [v, lo, hi]}
DS_KEY = {"GraphQA": "graphqa", "Erdős": "erdos"}
MODEL_KEY = {"Qwen3-8B": "hf-qwen3-8b-nscale", "Gemma 4 31B": "hf-gemma-4-31b",
             "DeepSeek-V4-Flash": "hf-deepseek-v4-flash", "DeepSeek-V3.1": "hf-deepseek-v3.1-novita"}
COL, FULL = 3.03, 6.3  # ACL column and page width, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42,
})
BLUE, ORANGE, GREEN, INK, QUIET = "#2a78d6", "#e8702a", "#1f9e89", "#222222", "#666666"
CI_GRAY = "#8c8c8c"
DS_COLOR = {"GraphQA": BLUE, "Erdős": ORANGE}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def point_ci(ax, x, v, lo, hi, color, marker, ms=4.5, capsize=3.2):
    """A filled marker with its 95% interval drawn in gray behind it.

    The caps are wider than the marker, so an interval shorter than the marker still shows its two ends.
    """
    eb = ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt=marker, ms=ms, mfc=color, mec=color, mew=0.8,
                     ecolor=CI_GRAY, elinewidth=0.9, capsize=capsize, capthick=0.9, zorder=3)
    for artist in eb.get_children():  # a rate near 0 must not be cut off by the axis
        artist.set_clip_on(False)


def ci_handle(ax, label="95% interval"):
    """A legend entry drawn like the intervals themselves: a gray bar with caps."""
    h = ax.errorbar([float("nan")], [float("nan")], yerr=[[1], [1]], fmt="none", ecolor=CI_GRAY, elinewidth=0.9,
                    capsize=3.2, capthick=0.9)
    h.set_label(label)
    return h


# ---------------------------------------------------------------- Figure C1: one tuple
TUPLE_ANSWERS = [9, 9, 4, 10]  # Qwen3-8B, Erdős triangles-0000, direct: baseline and three relabelings
TRUE_ANSWER = 9
# A schematic graph (two triangles sharing a node), drawn identically in every panel; only the labels move.
G_POS = {0: (0.0, 1.0), 1: (0.0, 0.0), 2: (0.5, 0.5), 3: (1.0, 1.0), 4: (1.0, 0.0)}
G_EDGES = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
G_LABELS = [[0, 1, 2, 3, 4], [3, 0, 4, 1, 2], [2, 4, 1, 0, 3], [4, 2, 0, 3, 1]]
RIGHT, WRONG, SAME, DIFF = "#1f9e89", "#d1495b", "#ececec", "#e8702a"


def fig_tuple():
    """What a tuple is and how D grades a flip, on one real tuple.

    Left: the same graph under the baseline labeling and three relabelings, each with the answer the model
    gave (green: correct, red: wrong); the four answers form one tuple. Right: every pair of answers, shaded
    when the two differ; D is the shaded share. Coordinates are points on a 218 x 96 pt canvas (one column
    wide), y pointing down.
    """
    fig, ax = plt.subplots(figsize=(COL, 1.34))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 218); ax.set_ylim(96, 0); ax.axis("off"); ax.set_aspect("equal")

    # --- left: four serializations of one instance, one answer each
    titles = ["Baseline", "Relabeling 1", "Relabeling 2", "Relabeling 3"]
    x0, pw, gap, top = 2, 31, 4, 9
    right_edge = x0 + 4 * pw + 3 * gap
    ax.text(x0, 4, "Same graph, new node names", fontsize=6.2, weight="bold", color=INK, va="center")
    for p, (title, labels, ans) in enumerate(zip(titles, G_LABELS, TUPLE_ANSWERS)):
        left = x0 + p * (pw + gap)
        ax.add_patch(FancyBboxPatch((left, top), pw, 60, boxstyle="round,pad=0,rounding_size=2.5",
                                    fc="white", ec="#c9c9c9", lw=0.6))
        ax.text(left + pw / 2, top + 6, title, ha="center", va="center", fontsize=5.8, color=QUIET)
        gx, gy, s = left + 6.5, top + 13, 18  # graph box: 18 x 18 pt
        pts = {n: (gx + s * u, gy + s * (1 - v)) for n, (u, v) in G_POS.items()}
        for a, b in G_EDGES:
            ax.plot(*zip(pts[a], pts[b]), color="#9a9a9a", lw=0.6, zorder=1)
        for n, (px, py) in pts.items():
            ax.add_patch(plt.Circle((px, py), 3.6, fc="white", ec=INK, lw=0.5, zorder=2))
            ax.text(px, py + 0.2, str(labels[n]), ha="center", va="center", fontsize=5.4, color=INK, zorder=3)
        ok = ans == TRUE_ANSWER
        ax.add_patch(FancyBboxPatch((left + 6, top + 40), pw - 12, 14, boxstyle="round,pad=0,rounding_size=2",
                                    fc=RIGHT if ok else WRONG, ec="none"))
        ax.text(left + pw / 2, top + 47.3, str(ans), ha="center", va="center", fontsize=8, weight="bold",
                color="white")
    # bracket under the four answers: they form one tuple
    by = top + 64
    ax.plot([x0 + 3, x0 + 3, right_edge - 3, right_edge - 3], [by - 2.5, by, by, by - 2.5], color=QUIET, lw=0.6)
    ax.plot([(x0 + right_edge) / 2] * 2, [by, by + 2.5], color=QUIET, lw=0.6)
    ax.text((x0 + right_edge) / 2, by + 8, f"one tuple: one model, one mode (true answer {TRUE_ANSWER})",
            ha="center", va="center", fontsize=6.2, color=INK)
    kx = (x0 + right_edge) / 2 - 24  # colour key under the bracket
    for k, (lab, col) in enumerate([("correct", RIGHT), ("wrong", WRONG)]):
        ax.add_patch(FancyBboxPatch((kx + 28 * k, by + 13.5), 5, 4, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=col, ec="none"))
        ax.text(kx + 28 * k + 6.5, by + 15.5, lab, va="center", fontsize=5.8, color=QUIET)

    # --- right: the six answer pairs, shaded where the two answers differ (upper triangle only)
    k = len(TUPLE_ANSWERS)
    c = 12  # cell size
    gx0, gy0 = 216 - (k - 1) * c, 21  # grid origin: column 1, row 0
    mid = gx0 + (k - 1) * c / 2
    ax.text(mid - 4, 4, "Answer pairs", fontsize=6.2, weight="bold", color=INK, ha="center", va="center")
    for i, a in enumerate(TUPLE_ANSWERS):
        col = RIGHT if a == TRUE_ANSWER else WRONG
        if i < k - 1:
            ax.text(gx0 - 2.5, gy0 + (i + 0.5) * c, str(a), ha="right", va="center", fontsize=6, color=col,
                    weight="bold")
        if i > 0:
            ax.text(gx0 + (i - 0.5) * c, gy0 - 2, str(a), ha="center", va="bottom", fontsize=6, color=col,
                    weight="bold")
        for j, b in enumerate(TUPLE_ANSWERS):
            if j > i:
                ax.add_patch(plt.Rectangle((gx0 + (j - 1) * c, gy0 + i * c), c, c,
                                           fc=DIFF if a != b else SAME, ec="white", lw=0.8))
    n_pairs = k * (k - 1) // 2
    n_diff = sum(a != b for i, a in enumerate(TUPLE_ANSWERS) for b in TUPLE_ANSWERS[i + 1:])
    ky = gy0 + 3 * c + 5  # key under the grid
    for kk, (lab, fc) in enumerate([("differ", DIFF), ("same", SAME)]):
        ax.add_patch(plt.Rectangle((gx0 - 12 + 22 * kk, ky), 4, 4, fc=fc, ec="#cfcfcf" if fc == SAME else "none",
                                   lw=0.4))
        ax.text(gx0 - 12 + 22 * kk + 5.5, ky + 2, lab, va="center", fontsize=5.8, color=QUIET)
    ax.text(mid - 4, by + 8, f"$D$ = {n_diff}/{n_pairs} = {n_diff / n_pairs:.2f}", ha="center", va="center",
            fontsize=7.2, color=INK)
    ax.text(mid - 4, by + 15.5, "$D > 0$: the tuple flips", ha="center", va="center", fontsize=6.0, color=INK)
    save(fig, "fig1_tuple")


# ---------------------------------------------------------------- Figures 1 and B1: M1 vs M2, pooled
POOLED = {ds: {md: tuple(FIG_CI[f"{DS_KEY[ds]}|all|{md}"]["flip"] + FIG_CI[f"{DS_KEY[ds]}|all|{md}"]["weighted"])
              for md in ("M1", "M2")} for ds in ("GraphQA", "Erdős")}  # (flip, lo, hi, weighted, lo, hi)
MODE_LABEL = {"M1": "M1 direct", "M2": "M2 code from text"}
DS_MARKER = {"GraphQA": "^", "Erdős": "s"}


def fig_pooled(k, name, ylabel, ymax, pct):
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    unit = "%" if pct else ""
    for i, mode in enumerate(["M1", "M2"]):
        for ds, off in [("GraphQA", -0.1), ("Erdős", 0.1)]:
            v, lo, hi = POOLED[ds][mode][3 * k:3 * k + 3]
            x = i + off
            point_ci(ax, x, v, lo, hi, DS_COLOR[ds], DS_MARKER[ds])
            left = ds == "GraphQA"
            ax.text(x + (-0.07 if left else 0.07), v, f"{v:.1f}{unit}\n[{lo:.1f}, {hi:.1f}]",
                    ha="right" if left else "left", va="center", fontsize=5.8, color=QUIET, linespacing=1.1)
    ax.set_xticks([0, 1], [MODE_LABEL["M1"], MODE_LABEL["M2"]])
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    handles = [Line2D([], [], ls="", marker=DS_MARKER[ds], color=DS_COLOR[ds], ms=4.5, label=ds)
               for ds in ("GraphQA", "Erdős")] + [ci_handle(ax)]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.03, 1.04), handlelength=1.2,
              borderaxespad=0.2, labelspacing=0.35)
    save(fig, name)


# ---------------------------------------------------------------- Figure 2: flip rate by graph size
SIZE = ["up to 10", "11 to 20", "21 to 40", "over 40"]  # bins of scripts/paper_metrics.py
SIZE_TICKS = ["≤10", "11–20", "21–40", ">40"]
ARM_KEY = {"Direct (M1)": "direct", "Code, plain Python (M2)": "code/native", "Code, NetworkX (M2)": "code/networkx"}
SIZE_FLIP = {ds: {arm: [METRICS["flip_rate_by_edges_relabel+order"][f"{DS_KEY[ds]}|{b}|{key}"]["flip_rate"]
                        for b in SIZE] for arm, key in ARM_KEY.items()} for ds in ("GraphQA", "Erdős")}
ARM_STYLE = {"Direct (M1)": (BLUE, "o"), "Code, plain Python (M2)": (ORANGE, "s"), "Code, NetworkX (M2)": (GREEN, "D")}


def fig_size():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85), sharey=True)
    for ax, ds in zip(axes, ["GraphQA", "Erdős"]):
        for arm, ys in SIZE_FLIP[ds].items():
            c, m = ARM_STYLE[arm]
            ax.plot(range(4), ys, color=c, marker=m, ms=3.2, lw=1.1, label=arm)
        ax.set_title(ds, fontsize=8)
        ax.set_xticks(range(4), SIZE_TICKS, fontsize=6.8)
        ax.tick_params(axis="x", length=2, pad=2)
        ax.set_xlim(-0.35, 3.35)
        ax.set_xlabel("Edges in the graph", fontsize=7)
    axes[0].set_ylabel("Tuples that flip (%)")
    axes[0].set_ylim(0, 40)
    axes[0].yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08),
               fontsize=6.3, handlelength=1.6, columnspacing=0.8)
    save(fig, "fig4_graph_size")


# ---------------------------------------------------------------- Figure B4: where plain-Python flips start
LOC_B = METRICS["localization_b"]  # dataset|arm|factor -> {flips, program, crash, copy}
START = {  # condition: (program, crash, copy), % of flipped plain-Python tuples
    f"{ds} {factor}": tuple(LOC_B[f"{DS_KEY[ds]}|code/native|{factor}"][k] for k in ("program", "crash", "copy"))
    for ds in ("GraphQA", "Erdős") for factor in ("relabel+order", "structure")
}


def fig_start():
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    w = 0.26
    for j, (lab, c) in enumerate([("Program", BLUE), ("Crash", ORANGE), ("Copy", GREEN)]):
        xs = [i + (j - 1) * w for i in range(4)]
        ax.bar(xs, [v[j] for v in START.values()], width=w - 0.03, color=c, label=lab)
    ax.set_xticks(range(4), [c.replace(" ", "\n") for c in START])
    ax.set_ylim(0, 100); ax.set_ylabel("Flipped tuples (%)")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig5_where_flips_start")


# ---------------------------------------------------------------- Figure B5: flips that take the other reading
OTHER = METRICS["other_reading_code_flips_all_answered"]  # dataset|task|model -> counts of flipped code tuples
# in which every serialization is answered, as Section 5.3 counts them


def reading_share(key):
    """Flips whose answers include the other reading (or, for Erdős neighbor, a strict subset of the neighbours)."""
    c = OTHER[key]
    hits = c.get("takes the other reading", 0) + c.get("strict subset of true neighbours", 0)
    return c["flips"], 100 * hits / c["flips"]


READING = [(f"{label} ({n})", share) for label, (n, share) in [
    ("Gemma 4 31B, disconnected nodes", reading_share("graphqa|disconnected_nodes|hf-gemma-4-31b")),
    ("DeepSeek-V4-Flash, connected nodes", reading_share("graphqa|connected_nodes|hf-deepseek-v4-flash")),
    ("DeepSeek-V3.1, connected nodes", reading_share("graphqa|connected_nodes|hf-deepseek-v3.1-novita")),
    ("Qwen3-8B, disconnected nodes", reading_share("graphqa|disconnected_nodes|hf-qwen3-8b-nscale")),
    ("Qwen3-8B, Erdős neighbor", reading_share("erdos|neighbor|hf-qwen3-8b-nscale")),
]]


def fig_reading():
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    labels, vals = zip(*READING)
    ys = range(len(vals))[::-1]
    ax.barh(list(ys), vals, height=0.62, color=BLUE)
    for y, v in zip(ys, vals):
        ax.text(v - 2, y, f"{v:.0f}%", ha="right", va="center", fontsize=6.5, color="white")
    ax.set_yticks(list(ys), labels, fontsize=6.8)
    ax.set_xlim(0, 100); ax.set_xlabel("Flips taking the other reading (%)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.grid(axis="x", color="#e3e3e3", lw=0.5); ax.grid(axis="y", visible=False)
    save(fig, "fig6_other_reading")


# ---------------------------------------------------------------- Figures B2 and B3: per model
MODELS = ["Qwen3-8B", "Gemma 4 31B", "DeepSeek-V4-Flash", "DeepSeek-V3.1"]
MODEL_MARKER = {"Qwen3-8B": "o", "Gemma 4 31B": "^", "DeepSeek-V4-Flash": "s", "DeepSeek-V3.1": "D"}
PER_MODEL = {(ds, m, md): tuple(FIG_CI[f"{DS_KEY[ds]}|{MODEL_KEY[m]}|{md}"]["flip"]
                                + FIG_CI[f"{DS_KEY[ds]}|{MODEL_KEY[m]}|{md}"]["weighted"])
             for ds in ("GraphQA", "Erdős") for m in MODELS for md in ("M1", "M2")}  # (flip, lo, hi, weighted, lo, hi)


def fig_per_model(k, name, ylabel, ymax, pct):
    fig, ax = plt.subplots(figsize=(FULL - 1.3, 2.1))
    step = 0.09
    for i, mode in enumerate(["M1", "M2"]):
        for ds, sign in [("GraphQA", -1), ("Erdős", 1)]:
            order = MODELS[::-1] if sign < 0 else MODELS
            for j, m in enumerate(order):
                v, lo, hi = PER_MODEL[(ds, m, mode)][3 * k:3 * k + 3]
                x = i + sign * (0.08 + step * j)
                point_ci(ax, x, v, lo, hi, DS_COLOR[ds], MODEL_MARKER[m], ms=4.2, capsize=2.6)
            ax.text(i + sign * (0.08 + 1.5 * step), -0.035 * ymax, ds, ha="center", va="top", fontsize=7,
                    color=QUIET, transform=ax.transData, clip_on=False)
    ax.set_xticks([0, 1], [MODE_LABEL["M1"], MODE_LABEL["M2"]])
    ax.tick_params(axis="x", pad=15, length=0)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, ymax); ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    handles = [Line2D([], [], ls="", marker="s", color=BLUE, ms=5, label="GraphQA"),
               Line2D([], [], ls="", marker="s", color=ORANGE, ms=5, label="Erdős")]
    handles += [Line2D([], [], ls="", marker=MODEL_MARKER[m], color=QUIET, ms=4.2, label=m)
                for m in MODELS]
    handles.append(ci_handle(ax))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, name)


if __name__ == "__main__":
    fig_tuple()
    fig_pooled(0, "fig2_flip_by_mode", "Tuples that flip (%)", 25, True)
    fig_pooled(1, "fig3_weighted_by_mode", "Weighted flip rate (×100)", 20, False)
    fig_size()
    fig_start()
    fig_reading()
    fig_per_model(0, "figB1_flip_by_model", "Tuples that flip (%)", 60, True)
    fig_per_model(1, "figB2_weighted_by_model", "Weighted flip rate (×100)", 40, False)
    print("wrote", sorted(p.name for p in OUT.glob("fig*.pdf")))
