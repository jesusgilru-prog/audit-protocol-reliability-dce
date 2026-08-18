"""
Sensitivity of the headline accuracy to the LOFO R^2 > 0 threshold used to
declare a scenario "universal" (raised in external review, ronda 1: the
threshold is a somewhat arbitrary choice and the whole result depends on it).

Recomputes protocol accuracy (verdict vs declared ground truth) at several
alternative thresholds using the raw r2_lofo values already stored per
scenario in results/synth_scenarios.csv (v1, Gaussian-drift pass) -- no
regeneration needed, this is a pure re-derivation from existing numbers.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def accuracy_at_threshold(df, thresh):
    says_universal = df.r2_lofo > thresh
    correct = says_universal == df.is_universal
    return dict(
        threshold=thresh,
        overall=float(correct.mean()),
        when_universal=float(correct[df.is_universal].mean()),
        when_confounded=float(correct[~df.is_universal].mean()),
    )


def main():
    df = pd.read_csv(os.path.join(RESULTS, "synth_scenarios.csv"))
    df = df.dropna(subset=["r2_lofo"])
    thresholds = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    rows = [accuracy_at_threshold(df, t) for t in thresholds]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    with open(os.path.join(RESULTS, "threshold_sensitivity.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\nsaved results/threshold_sensitivity.json")


if __name__ == "__main__":
    main()
