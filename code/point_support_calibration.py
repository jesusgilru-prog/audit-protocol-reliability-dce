"""
Calibrates the stratified permutation test against the NUMBER OF SHARED
POINTS on the worst leave-one-facility-out fold.

Why this exists (Codex, ronda 14, 2026-08-25). The manuscript concludes
that the companion corpus falls back to the threshold rule because the fold
holding out its largest facility has one point of shared support. That step
-- from "one point" to "no usable permutation" -- was an inference, not a
measurement. This script measures it: it bins scenarios by how many points
of the held-out facility actually fall inside the range spanned by the
others on the worst fold, and reports the stratified test's power and
false-alarm rate per bin.

The output is what turns the manuscript's operational threshold into a
measured quantity rather than a judgement call.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth_audit_protocol_v3 import (  # noqa: E402
    gen_scenario_overlap, quantile_strata, perm_pvalue, ALPHA, wilson,
)

RESULTS = os.path.join(HERE, "..", "results")
RNG_SEED = 20260828


def worst_fold_shared_points(df, col="log_re"):
    """(count, total) of held-out points inside the others' range, for the
    fold where that count is smallest."""
    best = None
    for k in df.facility.unique():
        held = df.loc[df.facility == k, col]
        rest = df.loc[df.facility != k, col]
        lo, hi = rest.min(), rest.max()
        c = int(((held >= lo) & (held <= hi)).sum())
        if best is None or c < best[0]:
            best = (c, int(len(held)))
    return best


def main(target_per_bin=140, nperm=99, max_draws=60000):
    rng = np.random.default_rng(RNG_SEED)
    # bins on the raw count of shared points on the worst fold
    edges = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 10**6)]
    labels = ["0", "1", "2", "3-4", "5-9", ">=10"]
    buckets = {l: [] for l in labels}

    def which(c):
        for (lo, hi), l in zip(edges, labels):
            if lo <= c <= hi:
                return l
        return None

    draws = 0
    while draws < max_draws and any(
            sum(1 for r in buckets[l] if r["is_universal"]) < target_per_bin
            or sum(1 for r in buckets[l] if not r["is_universal"]) < target_per_bin
            for l in labels):
        draws += 1
        K = int(rng.integers(3, 9))
        n = int(rng.integers(20, 50))
        sigma = float(rng.uniform(0.05, 0.6))
        het = float(rng.uniform(0.5, 1.5))
        is_universal = bool(rng.integers(0, 2))
        overlap = float(rng.uniform(0.0, 0.30))
        df = gen_scenario_overlap(K, n, sigma, het, is_universal, overlap, rng)
        c, tot = worst_fold_shared_points(df)
        lab = which(c)
        if lab is None:
            continue
        arm = [r for r in buckets[lab] if r["is_universal"] == is_universal]
        if len(arm) >= target_per_bin:
            continue
        X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
        y = df.log_cp.values
        g = df.facility.values
        r2, p_naive = perm_pvalue(X, y, g, rng, nperm=nperm)
        strata = quantile_strata(df.log_re.values, nbins=max(2, K))
        _, p_strat = perm_pvalue(X, y, g, rng, nperm=nperm, strata=strata)
        buckets[lab].append(dict(
            is_universal=is_universal, shared_points=c, held_out_n=tot,
            thr_correct=(bool(np.isfinite(r2) and r2 > 0)) == is_universal,
            perm_correct=(bool(np.isfinite(p_naive) and p_naive > ALPHA)) == is_universal,
            strat_correct=(bool(np.isfinite(p_strat) and p_strat > ALPHA)) == is_universal,
        ))
        if draws % 2000 == 0:
            filled = {l: len(buckets[l]) for l in labels}
            print(f"draws={draws} filled={filled}", flush=True)

    rows = [r | {"bin": l} for l in labels for r in buckets[l]]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "point_support_calibration.csv"), index=False)

    out = {"rng_seed": RNG_SEED, "nperm": nperm, "alpha": ALPHA,
           "binning_statistic": "shared_points_on_worst_fold",
           "total_draws": draws, "bins": []}
    print(f"\n{'shared pts':>11} {'n_u':>5} {'n_c':>5} {'strat FA':>9} "
          f"{'strat power':>12} {'power 95% CI':>18} {'thr power':>10}")
    for l in labels:
        b = df[df.bin == l]
        u, c = b[b.is_universal], b[~b.is_universal]
        if len(u) < 20 or len(c) < 20:
            continue
        fa = float(1 - u.strat_correct.mean())
        pw = float(c.strat_correct.mean())
        lo, hi = wilson(int(c.strat_correct.sum()), len(c))
        print(f"{l:>11} {len(u):>5} {len(c):>5} {fa:>9.3f} {pw:>12.3f} "
              f"  [{lo:.3f}, {hi:.3f}] {c.thr_correct.mean():>10.3f}")
        out["bins"].append(dict(
            shared_points=l, n_universal=int(len(u)), n_confounded=int(len(c)),
            stratified_false_alarm=fa, stratified_power=pw,
            stratified_power_ci95=[lo, hi],
            naive_false_alarm=float(1 - u.perm_correct.mean()),
            threshold_power=float(c.thr_correct.mean()),
            threshold_false_alarm=float(1 - u.thr_correct.mean()),
        ))
    path = os.path.join(RESULTS, "point_support_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}  ({draws} draws)")


if __name__ == "__main__":
    main()
