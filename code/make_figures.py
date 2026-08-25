import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(HERE, "..", "figures")

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ---- Figure 1: accuracy by confound type (v2) ----
df2 = pd.read_csv(os.path.join(RESULTS, "synth_scenarios_v2.csv"))
types = ["gaussian_linear", "outlier_contaminated", "nonlinear_misspecified", "clustered"]
labels = ["Gaussian\nparameter drift", "Drift with\noutlier contamination", "Nonlinear\nmisspecification", "Clustered\n(partial universality)"]

acc_universal = [df2[(df2.confound_type == t) & (df2.is_universal)].correct.mean() for t in types]
acc_confounded = [df2[(df2.confound_type == t) & (~df2.is_universal)].correct.mean() for t in types]
acc_universal_perm = [df2[(df2.confound_type == t) & (df2.is_universal)].perm_correct.mean() for t in types]
acc_confounded_perm = [df2[(df2.confound_type == t) & (~df2.is_universal)].perm_correct.mean() for t in types]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(types))
w = 0.2
ax.bar(x - 1.5*w, acc_universal, w, label="Universal (threshold rule)", color="#4C72B0")
ax.bar(x - 0.5*w, acc_confounded, w, label="Confounded (threshold rule)", color="#C44E52")
ax.bar(x + 0.5*w, acc_universal_perm, w, label="Universal (permutation rule)", color="#8172B3")
ax.bar(x + 1.5*w, acc_confounded_perm, w, label="Confounded (permutation rule)", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Protocol accuracy\n(correct verdict vs. ground truth)")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="grey", ls="--", lw=0.8)
ax.legend(loc="lower left", fontsize=7, ncol=2)
ax.set_title("LOFO-verdict accuracy by confounding mechanism:\nthreshold rule vs. permutation rule")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_accuracy_by_confound_type.png"), bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: permutation importance (meta-model v2, OBSERVABLE-ONLY variant) ----
with open(os.path.join(RESULTS, "meta_model_v2_results.json")) as f:
    mv2 = json.load(f)
imp = mv2["observable_only"]["permutation_importance"]
items = sorted(imp.items(), key=lambda kv: kv[1])
names = [k.replace("ct_", "type: ").replace("_", " ") for k, v in items]
vals = [v for k, v in items]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.barh(names, vals, color="#55A868")
ax.set_xlabel("Permutation importance\n(drop in accuracy when shuffled)")
ax.set_title("Observable-only meta-model:\nwhat predicts protocol correctness?")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_permutation_importance.png"), bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: accuracy vs K in the "hard" heterogeneity band (v1),
# CONFOUNDED ARM ONLY (F7 fix, external review 2026-08-20): the pooled
# version dilutes the real K-trend with the near-ceiling universal arm.
with open(os.path.join(RESULTS, "practical_guideline.json")) as f:
    pg = json.load(f)
by_k_thr = pd.DataFrame(pg["accuracy_by_K_confounded_only"])
by_k_perm = pd.DataFrame(pg["accuracy_by_K_confounded_only_permutation"])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(by_k_thr.K_bin, by_k_thr.accuracy, marker="o", color="#C44E52", label="Threshold rule")
ax.plot(by_k_perm.K_bin, by_k_perm.accuracy, marker="s", color="#55A868", label="Permutation rule")
ax.set_xlabel("Number of facilities/domains (K)")
ax.set_ylabel("Protocol accuracy\n(confounded arm only)")
ax.set_ylim(0.0, 1.05)
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Accuracy vs. K in the ambiguous heterogeneity band\n(0.5-1.5), gaussian_linear confounding, confounded arm only")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig3_accuracy_vs_K_hard_band.png"), bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: false alarms and power vs covariate-support overlap (v3) ----
# The headline of the third pass: the permutation rule's validity is
# conditional on facilities sharing covariate support, and no label
# permutation scheme survives full separation.
v3_path = os.path.join(RESULTS, "synth_scenarios_v3_summary.json")
if os.path.exists(v3_path):
    with open(v3_path) as f:
        v3 = json.load(f)
    b = pd.DataFrame(v3["by_overlap_bin"])
    x = np.arange(len(b))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)

    ax = axes[0]
    ax.plot(x, b.threshold_false_alarm, marker="o", color="#C44E52",
            label="Threshold")
    ax.plot(x, b.permutation_false_alarm, marker="s", color="#55A868",
            label="Permutation")
    ax.plot(x, b.permutation_stratified_false_alarm, marker="^",
            color="#8172B3", label="Permutation (stratified)")
    ax.axhline(0.05, color="grey", ls="--", lw=1)
    ax.text(0.06, 0.028, r"nominal $\alpha$=0.05", fontsize=7.5,
            color="grey", ha="left")
    ax.set_ylabel("False-alarm rate\n(universal arm)")
    ax.set_title("Calibration", fontsize=11)
    ax.set_ylim(-0.02, max(0.32, float(b.permutation_false_alarm.max()) + 0.05))
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[1]
    ax.plot(x, b.threshold_power, marker="o", color="#C44E52", label="Threshold")
    ax.plot(x, b.permutation_power, marker="s", color="#55A868",
            label="Permutation")
    ax.plot(x, b.permutation_stratified_power, marker="^", color="#8172B3",
            label="Permutation (stratified)")
    ax.set_ylabel("Power\n(confounded arm)")
    ax.set_title("Detection", fontsize=11)
    ax.set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(b.overlap_bin, fontsize=8)
        ax.set_xlabel("Covariate-support overlap between facilities")
        ax.grid(alpha=0.15)
    fig.suptitle("Validity of each declaration rule vs. how much the facilities' "
                 "covariate windows overlap", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_overlap_calibration_power.png"),
                bbox_inches="tight")
    plt.close(fig)

print("figures saved to", FIGS)
for fn in sorted(os.listdir(FIGS)):
    print(" ", fn)
