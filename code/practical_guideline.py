"""
Practical guideline: within the "hard" (ambiguous) heterogeneity band --
where the ground truth (universal vs confounded) is genuinely difficult
to tell apart from noise -- how does protocol accuracy depend on K
(number of facilities/domains) and n (points per facility)?

At the extremes (heterogeneity near 0 or near 3) the earlier meta-model
found K barely matters (accuracy saturates near 1.0 either way). This
script isolates the regime where K actually should matter, and reports
the minimum K at which the LOFO+bootstrap protocol crosses reliability
thresholds (80%, 90%, 95%) -- the concrete, quantitative answer to
"how many domains do you need" that motivated this study (BRIEF.md
section 8, point 3).
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def main():
    df = pd.read_csv(os.path.join(RESULTS, "synth_scenarios.csv"))

    # "hard" band: heterogeneity between 0.5 and 1.5 (ambiguous regime,
    # picked by inspecting the full distribution: below ~0.5 the protocol
    # is already near-ceiling, above ~1.5 it saturates too -- see
    # results/synth_scenarios.csv for the raw distribution used to pick this)
    hard = df[(df.heterogeneity >= 0.5) & (df.heterogeneity <= 1.5)].copy()
    print(f"hard-band scenarios: {len(hard)} / {len(df)}")

    hard["K_bin"] = pd.cut(hard.K, bins=[0, 3, 6, 9, 12, 15, 20],
                            labels=["2-3", "4-6", "7-9", "10-12", "13-15", "16-20"])
    by_k = hard.groupby("K_bin", observed=True).agg(
        n_scenarios=("correct", "size"),
        accuracy=("correct", "mean"),
    ).reset_index()
    print("\nAccuracy by K (number of facilities/domains), hard heterogeneity band:")
    print(by_k.to_string(index=False))

    hard["n_bin"] = pd.cut(hard.n_per_facility, bins=[0, 15, 30, 50, 75, 100],
                            labels=["5-15", "16-30", "31-50", "51-75", "76-100"])
    by_n = hard.groupby("n_bin", observed=True).agg(
        n_scenarios=("correct", "size"),
        accuracy=("correct", "mean"),
    ).reset_index()
    print("\nAccuracy by n (points per facility), hard heterogeneity band:")
    print(by_n.to_string(index=False))

    # minimum K crossing reliability thresholds (monotonic-ish trend expected)
    thresholds = [0.80, 0.90, 0.95]
    crossing = {}
    by_k_sorted = by_k.dropna(subset=["accuracy"])
    for t in thresholds:
        above = by_k_sorted[by_k_sorted.accuracy >= t]
        crossing[t] = str(above.K_bin.iloc[0]) if len(above) else "not reached in tested range (K<=20)"
    print("\nMinimum K bin reaching each reliability threshold (hard band):")
    for t, k in crossing.items():
        print(f"  {int(t*100)}%: K in {k}")

    out = dict(
        hard_band_heterogeneity=[0.5, 1.5],
        n_hard_scenarios=int(len(hard)),
        accuracy_by_K=by_k.to_dict(orient="records"),
        accuracy_by_n=by_n.to_dict(orient="records"),
        min_K_for_threshold=crossing,
    )
    with open(os.path.join(RESULTS, "practical_guideline.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nsaved results/practical_guideline.json")


if __name__ == "__main__":
    main()
