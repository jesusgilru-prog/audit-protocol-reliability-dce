"""Figure for the fourth pass: detection against matched effect size.

Left panel: threshold rule. Right panel: permutation rule. One line per
mechanism, plotted against the effect-size bin, with Wilson 95% intervals.
The point of the figure is that the four mechanisms, which are 73 points apart
when indexed by the raw heterogeneity parameter, lie within a few points of one
another once the effect size is matched.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

LABELS = {
    "gaussian_linear": "Gaussian parameter drift",
    "outlier_contaminated": "Drift + outlier contamination",
    "nonlinear_misspecified": "Nonlinear misspecification",
    "clustered": "Clustered (partial)",
}
COLORS = {
    "gaussian_linear": "#1f77b4",
    "outlier_contaminated": "#ff7f0e",
    "nonlinear_misspecified": "#d62728",
    "clustered": "#2ca02c",
}
MARKERS = {"gaussian_linear": "o", "outlier_contaminated": "s",
           "nonlinear_misspecified": "^", "clustered": "D"}


def main():
    with open(os.path.join(RESULTS, "effect_size_calibration.json")) as f:
        d = json.load(f)
    bins = list(d["detection_by_delta_bin"].keys())
    x = np.arange(len(bins))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, rule, title in (
            (axes[0], "threshold_rule", r"Threshold rule ($R^2_{\rm LOFO}>0$)"),
            (axes[1], "permutation_rule", "Facility-label permutation test")):
        for ctype in LABELS:
            y, lo, hi = [], [], []
            for b in bins:
                cell = d["detection_by_delta_bin"][b][ctype]
                y.append(cell[rule])
                ci = cell[rule + "_ci95"]
                lo.append(cell[rule] - ci[0])
                hi.append(ci[1] - cell[rule])
            ax.errorbar(x, y, yerr=[lo, hi], marker=MARKERS[ctype],
                        color=COLORS[ctype], label=LABELS[ctype],
                        capsize=2.5, lw=1.6, ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(r"effect size $\delta$ (unabsorbed structure / noise)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("detection rate (confounded arm)")
    axes[0].legend(fontsize=8, loc="upper left", frameon=True)
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig5_matched_effect_size.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
