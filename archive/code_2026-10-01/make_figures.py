"""
Figures for the Section 5.2 re-run.

Figure: per-subject forest plot (peer_review_round3.md C6). For each WESAD
subject, reduced-condition (EDA missing) within-subject AUROC under LOSO,
for three models, next to that subject's heart-rate response to the
stressor (stress-minus-baseline change in mean NN interval). Subjects are
ordered by HR response, weakest at the top.

Palette: the dataviz reference categorical slots 1-3 (validated all-pairs,
light mode). Aqua is below 3:1 contrast on the surface, so every series also
has its own marker shape, and the numbers are in per_subject_loso.csv (table view).

Output: results/fig_per_subject_forest.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parent / "results"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = [  # (model key, label, colour, marker)
    ("gated_dropout_p0.5", "Gated + dropout", "#2a78d6", "o"),
    ("lgbm_nan_training_p0.5", "LightGBM, NaN-trained (ii)", "#eb6834", "s"),
    ("hr_only", "Heart rate only", "#1baf7a", "^"),
]


def forest(scheme: str = "loso") -> Path:
    d = pd.read_csv(RESULTS / f"per_subject_{scheme}.csv")
    d = d[d.tier == "reduced"]
    hr = (d.drop_duplicates("subject_id")
            .set_index("subject_id")["delta_mean_nn_ms_stress_minus_baseline"])
    order = hr.sort_values(ascending=False).index.tolist()   # weakest response first
    ypos = {s: i for i, s in enumerate(order)}
    labels = [s.replace("WESAD_", "") for s in order]

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
                         "font.size": 9})
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(7.2, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.08})
    fig.patch.set_facecolor(SURFACE)

    offsets = np.linspace(-0.22, 0.22, len(SERIES))
    rho_txt = []
    for (key, lab, col, mk), off in zip(SERIES, offsets):
        g = d[d.model == key].set_index("subject_id").reindex(order)
        y = np.array([ypos[s] for s in order]) + off
        ax.scatter(g["auroc_raw"], y, s=34, marker=mk, color=col, label=lab,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        rho, p = spearmanr(-hr.reindex(order), g["auroc_raw"])
        rho_txt.append(f"{lab}: ρ = {rho:+.2f}")

    ax.axvline(0.5, color=BASELINE, lw=1, zorder=1)
    ax.text(0.5, -0.9, "chance", color=MUTED, fontsize=8, ha="center", va="bottom")
    ax.set_xlim(0.15, 1.03)
    ax.set_xlabel("Within-subject AUROC with EDA missing (LOSO)", color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)

    axh.barh(range(len(order)), -hr.reindex(order).values, height=0.6,
             color="#86b6ef", edgecolor=SURFACE, linewidth=2, zorder=2)
    axh.set_xlabel("HR response\n(drop in mean NN, ms)", color=INK_2)
    axh.grid(axis="x", color=GRID, lw=0.6, zorder=0)

    for a in (ax, axh):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(BASELINE)
        a.tick_params(colors=INK_2, length=0)

    ax.set_ylabel("Subject (ordered by HR response, weakest at top)", color=INK_2)
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Discrimination without EDA tracks each subject's heart-rate response",
                 x=0.02, ha="left", fontsize=10.5, color=INK)
    fig.text(0.02, 0.02,
             "Spearman, HR-response magnitude vs AUROC (n = 15): " + "; ".join(rho_txt),
             fontsize=7.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.2)

    out = RESULTS / "fig_per_subject_forest"
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    return out.with_suffix(".png")


RC_SERIES = [  # (model key, label, colour, dash) - categorical slots 1-4; dashes
    # are the secondary cue for the two sub-3:1-contrast slots (print/CVD).
    ("gated_dropout_p0.5", "Gated + dropout", "#2a78d6", "-"),
    ("lgbm_nan_training_p0.5", "LightGBM, NaN-trained (ii)", "#eb6834", "-"),
    ("lgbm_ecg_only_per_config", "LightGBM, ECG-only (iii)", "#1baf7a", (0, (6, 2))),
    ("lgbm_nan_at_test", "LightGBM, complete-trained (i)", "#eda100", (0, (2, 2))),
]


def risk_coverage_fig(scheme: str = "loso", pcol: str = "p_raw") -> Path:
    """Selective-prediction curves with EDA missing (outline §5.5/§6).

    Windows are retained in order of confidence max(p, 1-p); the y-axis is
    the error rate among retained windows. Pooled over held-out subjects.
    """
    from run_metrics import aurc, risk_coverage
    pred = pd.read_csv(RESULTS / f"predictions_{scheme}.csv")
    red = pred[pred.tier == "reduced"]

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
                         "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ends = []
    for key, lab, col, dash in RC_SERIES:
        g = red[red.model == key]
        cov, risk = risk_coverage(g["y"].values, g[pcol].values)
        ax.plot(cov, risk, color=col, lw=2, ls=dash, label=lab, solid_capstyle="round")
        ends.append((dash, f"{lab}  (AURC {aurc(g['y'].values, g[pcol].values):.3f})", col))
    base = red[red.model == RC_SERIES[0][0]]["y"].mean()
    ax.axhline(min(base, 1 - base), color=BASELINE, lw=1, ls=(0, (3, 3)))
    ax.text(0.99, min(base, 1 - base) - 0.012, "error if always predicting 'baseline'",
            color=MUTED, fontsize=8, ha="right", va="top")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0, 0.56)
    ax.set_xlabel("Coverage (fraction of windows the model does not abstain on)", color=INK_2)
    ax.set_ylabel("Error rate on retained windows", color=INK_2)
    ax.grid(color=GRID, lw=0.6)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_2, length=0)
    ax.legend([plt.Line2D([], [], color=c, lw=2, ls=d) for d, _, c in ends],
              [t for _, t, _ in ends], loc="upper left", frameon=False, fontsize=8,
              labelcolor=INK)
    fig.suptitle("Abstaining on low-confidence windows when EDA is missing (LOSO)",
                 x=0.02, ha="left", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.13)
    out = RESULTS / "fig_risk_coverage_reduced"
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    return out.with_suffix(".png")


if __name__ == "__main__":
    print(forest())
    print(risk_coverage_fig())
