"""
Practical guideline: within the "hard" (ambiguous) heterogeneity band --
where the ground truth (universal vs confounded) is genuinely difficult
to tell apart from noise -- how does protocol accuracy depend on K
(number of facilities/domains) and n (points per facility)?

CORRECTED after external review (multi-model debate, 2026-08-20, finding
independently verified before fixing): the first version pooled the
universal and confounded arms together. Since the universal arm scores
~100% regardless of K by construction (there is no heterogeneity to
detect), pooling both arms diluted and obscured the real K-dependence,
which only exists in the confounded arm. This version reports the
confounded-only breakdown as the primary result, with the pooled
version kept alongside for continuity/comparison. It also reports the
permutation-rule accuracy by K for the same band, and removes the
`min_K_for_threshold` field flagged as misleading (it only checked
the first K-bin clearing each threshold in bin order, not that every
larger K bin also clears it -- not a real "minimum" in the presence of
non-monotonic accuracy).
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def by_k_table(df, target_col):
    df = df.copy()
    df["K_bin"] = pd.cut(df.K, bins=[0, 3, 6, 9, 12, 15, 20],
                          labels=["2-3", "4-6", "7-9", "10-12", "13-15", "16-20"])
    return df.groupby("K_bin", observed=True).agg(
        n_scenarios=(target_col, "size"),
        accuracy=(target_col, "mean"),
    ).reset_index()


def main():
    df = pd.read_csv(os.path.join(RESULTS, "synth_scenarios.csv"))

    # "hard" band: heterogeneity between 0.5 and 1.5 (ambiguous regime,
    # picked by inspecting the full distribution: below ~0.5 the protocol
    # is already near-ceiling, above ~1.5 it saturates too -- see
    # results/synth_scenarios.csv for the raw distribution used to pick this)
    hard = df[(df.heterogeneity >= 0.5) & (df.heterogeneity <= 1.5)].copy()
    hard_conf = hard[~hard.is_universal].copy()
    hard_univ = hard[hard.is_universal].copy()
    print(f"hard-band scenarios: {len(hard)} / {len(df)} "
          f"({len(hard_conf)} confounded, {len(hard_univ)} universal)")

    by_k_pooled = by_k_table(hard, "correct")
    by_k_conf = by_k_table(hard_conf, "correct")
    by_k_conf_perm = by_k_table(hard_conf.dropna(subset=["perm_correct"]), "perm_correct")
    by_k_univ = by_k_table(hard_univ, "correct")

    print("\nAccuracy by K, POOLED (both arms, diluted -- kept for comparison only):")
    print(by_k_pooled.to_string(index=False))
    print("\nAccuracy by K, CONFOUNDED ARM ONLY (the real signal, threshold rule):")
    print(by_k_conf.to_string(index=False))
    print("\nAccuracy by K, CONFOUNDED ARM ONLY (permutation rule):")
    print(by_k_conf_perm.to_string(index=False))
    print("\nAccuracy by K, UNIVERSAL ARM ONLY (near-ceiling by construction, sanity check):")
    print(by_k_univ.to_string(index=False))

    hard["n_bin"] = pd.cut(hard.n_per_facility, bins=[0, 15, 30, 50, 75, 100],
                            labels=["5-15", "16-30", "31-50", "51-75", "76-100"])
    by_n = hard.groupby("n_bin", observed=True).agg(
        n_scenarios=("correct", "size"),
        accuracy=("correct", "mean"),
    ).reset_index()
    print("\nAccuracy by n (points per facility), hard heterogeneity band, pooled:")
    print(by_n.to_string(index=False))

    out = dict(
        hard_band_heterogeneity=[0.5, 1.5],
        n_hard_scenarios=int(len(hard)),
        n_hard_confounded=int(len(hard_conf)),
        n_hard_universal=int(len(hard_univ)),
        accuracy_by_K_pooled=by_k_pooled.to_dict(orient="records"),
        accuracy_by_K_confounded_only=by_k_conf.to_dict(orient="records"),
        accuracy_by_K_confounded_only_permutation=by_k_conf_perm.to_dict(orient="records"),
        accuracy_by_K_universal_only=by_k_univ.to_dict(orient="records"),
        accuracy_by_n=by_n.to_dict(orient="records"),
    )
    with open(os.path.join(RESULTS, "practical_guideline.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nsaved results/practical_guideline.json")


if __name__ == "__main__":
    main()
