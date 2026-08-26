"""
Additional figures: the conceptual schematic, the real corpus's operating
windows, a 3-D detection surface, and the decision rule.

Added 2026-08-26. The manuscript carried its central mechanism, its
decision rule and the real corpus's geometry entirely in prose, which for
a journal whose articles are typically figure-led made a 21-page paper
read as a wall of text. None of these is decorative: each renders
something the text otherwise asks the reader to hold in their head.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

BLUE, RED, GREEN, PURPLE, GREY = "#4C72B0", "#C44E52", "#55A868", "#8172B3", "#8a8f98"


# ---------------------------------------------------------------- Figure A
# Why separated support breaks the permutation null: LOFO becomes
# extrapolation while the permuted pseudo-folds stay interpolation.
def fig_schematic():
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.6))
    rng = np.random.default_rng(3)

    def panel(ax, centres, width, title, held=0, permuted=False):
        cols = [BLUE, GREEN, PURPLE, "#d1a34a"]
        for k, c in enumerate(centres):
            x = rng.uniform(c - width / 2, c + width / 2, 26)
            y = 2.0 - 0.45 * x + rng.normal(0, 0.10, x.size)
            if permuted:
                # labels are shuffled ACROSS points, so each cluster carries a
                # mix of pseudo-facilities: that is precisely why a permuted
                # fold spans the whole range and stays an interpolation task
                cs = [cols[i] for i in rng.integers(0, len(centres), x.size)]
                ax.scatter(x, y, s=7, c=cs, alpha=.75, linewidths=0)
            else:
                is_held = (k == held)
                ax.scatter(x, y, s=7,
                           c=(RED if is_held else cols[k]),
                           alpha=.95 if is_held else .55, linewidths=0,
                           marker="o")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.4, 4.4)
        for s in ax.spines.values():
            s.set_color("#ccd0d6")

    wide = [1.0, 1.8, 2.4, 3.2]
    narrow = [0.5, 1.5, 2.6, 3.7]

    panel(axes[0, 0], wide, 2.4, "Shared support: held-out facility (red)\nsits inside the others' range")
    panel(axes[0, 1], wide, 2.4, "Permuted labels: pseudo-folds are\nstatistically the same problem", permuted=True)
    panel(axes[1, 0], narrow, 0.7, "Separated support: predicting the\nheld-out facility is extrapolation")
    panel(axes[1, 1], narrow, 0.7, "Permuted labels: pseudo-folds are\nstill interpolation — the null no longer matches", permuted=True)

    for ax in axes[1]:
        ax.set_xlabel(r"$\log Re$", fontsize=9)
    axes[0, 0].set_ylabel(r"$\log C_p$", fontsize=9)
    axes[1, 0].set_ylabel(r"$\log C_p$", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figA_schematic_support.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure B
# The companion corpus's actual operating windows, which is where the
# one-shared-point situation becomes visible at a glance.
def fig_corpus_windows():
    p = os.path.join(RESULTS, "companion_overlap.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    w = d.get("per_facility_log10_Re_window")
    if not w:
        return
    order = sorted(w, key=lambda k: w[k]["log10_re_min"])
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    cols = {order[i]: c for i, c in enumerate([BLUE, GREEN, PURPLE, RED])}
    for i, k in enumerate(order):
        lo, hi, n = w[k]["log10_re_min"], w[k]["log10_re_max"], w[k]["n"]
        ax.plot([lo, hi], [i, i], lw=9, solid_capstyle="butt",
                color=cols[k], alpha=.85)
        ax.text(hi + 0.06, i, f"  {k}  (n={n})", va="center", fontsize=8.5)
    # shared region for the worst fold
    wf = d.get("points_in_shared_region_worst_fold")
    if wf:
        lo, hi = wf["shared_region_log10"]
        ax.axvspan(lo, hi, color="#d62728", alpha=.18, zorder=0)
        ax.annotate(f"shared region for the {wf['facility']} fold:\n"
                    f"{wf['held_out_points_inside']} of {wf['held_out_points_total']} held-out points, "
                    f"{wf['other_points_inside']} of {wf['other_points_total']} others",
                    xy=((lo + hi) / 2, len(order) - 0.62),
                    xytext=((lo + hi) / 2 - 1.75, len(order) + 0.28),
                    fontsize=8, color="#8b1a1a",
                    arrowprops=dict(arrowstyle="->", color="#8b1a1a", lw=1))
    ax.set_yticks([]); ax.set_ylim(-0.7, len(order) + 0.95)
    ax.set_xlim(4.7, 8.6)
    ax.set_xlabel(r"$\log_{10} Re_\Omega$")
    ax.set_title("Operating windows of the four facilities in the companion corpus",
                 fontsize=10, pad=10)
    ax.grid(axis="x", alpha=.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figB_corpus_windows.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure C
# 3-D detection surface over the two factors that jointly govern it.
def fig_surface():
    d = pd.read_csv(os.path.join(RESULTS, "synth_scenarios_v3.csv"))
    c = d[~d.is_universal].copy()
    nb = 6
    ov_e = np.linspace(0, c.obs_min_fold_overlap.max(), nb + 1)
    he_e = np.linspace(c.heterogeneity.min(), c.heterogeneity.max(), nb + 1)
    c["oi"] = np.clip(np.digitize(c.obs_min_fold_overlap, ov_e[1:-1]), 0, nb - 1)
    c["hi"] = np.clip(np.digitize(c.heterogeneity, he_e[1:-1]), 0, nb - 1)
    ovc = (ov_e[:-1] + ov_e[1:]) / 2
    hec = (he_e[:-1] + he_e[1:]) / 2
    X, Y = np.meshgrid(ovc, hec, indexing="ij")

    fig = plt.figure(figsize=(9.6, 4.2))
    for idx, (col, name, cmap) in enumerate(
            [("strat_correct", "Stratified permutation", "viridis"),
             ("thr_correct", "Threshold rule", "viridis")]):
        Z = np.full((nb, nb), np.nan)
        for i in range(nb):
            for j in range(nb):
                sub = c[(c.oi == i) & (c.hi == j)]
                if len(sub) >= 15:
                    Z[i, j] = sub[col].mean()
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cmap, vmin=0, vmax=1,
                        edgecolor="k", linewidth=.25, alpha=.93,
                        rstride=1, cstride=1)
        ax.set_xlabel("min fold overlap", fontsize=8, labelpad=-2)
        ax.set_ylabel("heterogeneity $h$", fontsize=8, labelpad=-2)
        ax.set_zlabel("detection rate", fontsize=8, labelpad=-4)
        ax.set_zlim(0, 1)
        ax.set_title(name, fontsize=10, pad=0)
        ax.tick_params(labelsize=7, pad=-1)
        ax.view_init(elev=22, azim=-128)
    fig.suptitle("Detection of genuine confounding over covariate-support overlap "
                 "and confounding strength", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figC_surface_detection.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure D
# The decision rule, as a rule.
def fig_decision():
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.14",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#6b7280", lw=1.1))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                    fontsize=8, color="#374151", ha="center")

    box(3.1, 4.2, 3.8, .8, "Fit the pooled model and run LOFO", "#eef2f7", "#9aa3b2")
    box(2.4, 2.85, 5.2, .85,
        "On the worst fold, count the held-out points\n"
        r"inside the others' covariate range  ($m_{\min}$)",
        "#e7effb", BLUE)
    arrow(5, 4.2, 5, 3.7)

    box(.15, 1.15, 4.3, .95,
        r"$m_{\min}\geq 1$" "\nStratified permutation test\npower $\\approx$1.00",
        "#e8f4ec", GREEN)
    box(5.5, 1.15, 4.3, .95,
        r"$m_{\min}=0$" "\nPlain LOFO$>0$ threshold\npower 0.80, no false alarms",
        "#fdeceb", RED)
    arrow(4.3, 2.85, 2.3, 2.1)
    arrow(5.7, 2.85, 7.6, 2.1)

    ax.text(2.3, .45, "calibration only approximately nominal\nwhen $m_{\\min}$ is 1 or 2",
            ha="center", fontsize=7.4, color="#4b5563", style="italic")
    ax.text(7.65, .45, "a stratified test loses ~39 points\nof power in this regime",
            ha="center", fontsize=7.4, color="#4b5563", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figD_decision_rule.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_schematic(); fig_corpus_windows(); fig_surface(); fig_decision()
    print("nuevas figuras:")
    for f in sorted(os.listdir(FIGS)):
        if f.startswith("fig") and f[3].isupper():
            print("  ", f)
