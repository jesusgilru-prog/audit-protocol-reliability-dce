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
acc_confounded_perm = [df2[(df2.confound_type == t) & (~df2.is_universal)].perm_correct.mean() for t in types]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(types))
w = 0.27
ax.bar(x - w, acc_universal, w, label="Universal (threshold rule)", color="#4C72B0")
ax.bar(x, acc_confounded, w, label="Confounded (threshold rule)", color="#C44E52")
ax.bar(x + w, acc_confounded_perm, w, label="Confounded (permutation rule)", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Protocol accuracy\n(correct verdict vs. ground truth)")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="grey", ls="--", lw=0.8)
ax.legend(loc="lower left", fontsize=7.5)
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

print("figures saved to", FIGS)
for fn in os.listdir(FIGS):
    print(" ", fn)
