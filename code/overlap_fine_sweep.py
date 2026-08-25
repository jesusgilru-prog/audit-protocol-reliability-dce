"""
Fine sweep of the low-overlap regime, to locate empirically where the
stratified permutation test stops working.

Why this exists (Codex, ronda 13, 2026-08-25). The third pass shows the
stratified test well calibrated down to the whole 0.0-0.2 bin of Table 5,
and a dedicated batch shows it completely inert at overlap EXACTLY zero.
The manuscript then applies the second fact to the companion corpus, whose
minimum fold overlap is 0.008. That is an extrapolation between two
measured points, not a measured result, and it drives the manuscript's
recommendation for that corpus. This script measures the transition
directly instead.

Scenarios are drawn with narrow windows and then binned by the REALIZED
observable statistic (minimum fold overlap), so the bins are on the
quantity a practitioner computes, not on the latent design knob.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth_audit_protocol_v3 import (  # noqa: E402
    gen_scenario_overlap, min_fold_overlap, quantile_strata, perm_pvalue,
    ALPHA, wilson,
)

RESULTS = os.path.join(HERE, "..", "results")
RNG_SEED = 20260826

EDGES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
LABELS = ["0-0.005", "0.005-0.01", "0.01-0.02", "0.02-0.05",
          "0.05-0.10", "0.10-0.20"]


def main(n_scenarios=4000, nperm=99):
    rng = np.random.default_rng(RNG_SEED)
    recs = []
    for i in range(n_scenarios):
        K = int(rng.integers(3, 9))
        n = int(rng.integers(15, 60))
        sigma = float(rng.uniform(0.05, 0.6))
        het = float(rng.uniform(0.5, 1.5))
        is_universal = bool(rng.integers(0, 2))
        # narrow band of the design knob: this is the regime of interest
        overlap = float(rng.uniform(0.0, 0.18))
        df = gen_scenario_overlap(K, n, sigma, het, is_universal, overlap, rng)
        X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
        y = df.log_cp.values
        g = df.facility.values
        r2, p_naive = perm_pvalue(X, y, g, rng, nperm=nperm)
        strata = quantile_strata(df.log_re.values, nbins=max(2, K))
        _, p_strat = perm_pvalue(X, y, g, rng, nperm=nperm, strata=strata)
        thr_u = bool(np.isfinite(r2) and r2 > 0)
        perm_u = bool(np.isfinite(p_naive) and p_naive > ALPHA)
        strat_u = bool(np.isfinite(p_strat) and p_strat > ALPHA)
        recs.append(dict(
            is_universal=is_universal,
            obs_min_fold_overlap=min_fold_overlap(df),
            thr_correct=thr_u == is_universal,
            perm_correct=perm_u == is_universal,
            strat_correct=strat_u == is_universal,
        ))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{n_scenarios}", flush=True)

    df = pd.DataFrame(recs)
    df["bin"] = pd.cut(df.obs_min_fold_overlap, bins=EDGES, labels=LABELS,
                        include_lowest=True)
    df.to_csv(os.path.join(RESULTS, "overlap_fine_sweep.csv"), index=False)

    out = {"rng_seed": RNG_SEED, "nperm": nperm, "alpha": ALPHA,
           "binning_statistic": "obs_min_fold_overlap", "bins": []}
    univ, conf = df[df.is_universal], df[~df.is_universal]
    print(f"\n{'min fold overlap':>18} {'n_u':>5} {'n_c':>5} "
          f"{'strat FA':>9} {'strat FA 95% CI':>18} {'strat power':>12} "
          f"{'perm FA':>8} {'thr power':>10}")
    for lab in LABELS:
        u, c = univ[univ.bin == lab], conf[conf.bin == lab]
        if len(u) == 0 or len(c) == 0:
            continue
        fa = 1 - u.strat_correct.mean()
        lo, hi = wilson(int((~u.strat_correct).sum()), len(u))
        pw = c.strat_correct.mean()
        print(f"{lab:>18} {len(u):>5} {len(c):>5} {fa:>9.3f} "
              f"  [{lo:.3f}, {hi:.3f}] {pw:>12.3f} "
              f"{1 - u.perm_correct.mean():>8.3f} {c.thr_correct.mean():>10.3f}")
        out["bins"].append(dict(
            bin=lab, n_universal=int(len(u)), n_confounded=int(len(c)),
            stratified_false_alarm=float(fa),
            stratified_false_alarm_ci95=[lo, hi],
            stratified_power=float(pw),
            stratified_power_ci95=list(wilson(int(c.strat_correct.sum()), len(c))),
            naive_false_alarm=float(1 - u.perm_correct.mean()),
            threshold_power=float(c.thr_correct.mean()),
            threshold_false_alarm=float(1 - u.thr_correct.mean()),
        ))
    # The exactly-zero split of the lowest bin, reported in the manuscript:
    # scenarios whose worst fold has zero RANGE overlap behave differently
    # from those with a sliver of it, and the lowest bin mixes both.
    zc = conf[conf.obs_min_fold_overlap == 0.0]
    zu = univ[univ.obs_min_fold_overlap == 0.0]
    pc = conf[(conf.obs_min_fold_overlap > 0) & (conf.obs_min_fold_overlap <= 0.005)]
    pu = univ[(univ.obs_min_fold_overlap > 0) & (univ.obs_min_fold_overlap <= 0.005)]
    out["lowest_bin_split_at_exact_zero"] = {
        "min_fold_overlap_exactly_zero": {
            "n_confounded": int(len(zc)),
            "stratified_power": float(zc.strat_correct.mean()) if len(zc) else None,
            "n_universal": int(len(zu)),
            "stratified_false_alarm": float(1 - zu.strat_correct.mean()) if len(zu) else None,
            "threshold_power": float(zc.thr_correct.mean()) if len(zc) else None,
        },
        "min_fold_overlap_in_0_to_0.005_exclusive": {
            "n_confounded": int(len(pc)),
            "stratified_power": float(pc.strat_correct.mean()) if len(pc) else None,
            "n_universal": int(len(pu)),
            "stratified_false_alarm": float(1 - pu.strat_correct.mean()) if len(pu) else None,
        },
    }
    print(f"\nlowest bin split: min-fold exactly 0 -> stratified power "
          f"{out['lowest_bin_split_at_exact_zero']['min_fold_overlap_exactly_zero']['stratified_power']:.3f} "
          f"(n={len(zc)}); strictly positive but <=0.005 -> "
          f"{out['lowest_bin_split_at_exact_zero']['min_fold_overlap_in_0_to_0.005_exclusive']['stratified_power']:.3f} "
          f"(n={len(pc)})")

    path = os.path.join(RESULTS, "overlap_fine_sweep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
