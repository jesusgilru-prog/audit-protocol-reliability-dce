"""Sensitivity check for the fourth pass: how the contaminated mechanism is
normalised.

The effect size delta divides the unabsorbed systematic structure by the RMS of
the scenario's error term. For the outlier-contaminated mechanism that RMS
includes the contamination sub-population, so sigma_eff = 2.145 sigma, and
matching delta there requires a deviation roughly 2.1x larger in units of the
base sigma than in the uncontaminated mechanisms. That is a defensible reading
(contamination is noise, and delta is signal-to-noise), but it is a choice, and
it is the one that puts the contaminated mechanism at the top of the matched
comparison.

This script re-runs the contaminated arm on the same paired design points with
the alternative normalisation, delta divided by the base sigma instead, and
reports both side by side. Everything else is identical.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "..", "results")

import effect_size_calibration as esc  # noqa: E402


def run_one_altnorm(rep):
    """Same as esc.run_one for the contaminated mechanism, except that the
    deviation is scaled against the base sigma rather than the RMS of the
    contaminated error term."""
    rng_design = np.random.default_rng(esc.RNG_SEED + rep)
    K = int(rng_design.integers(2, 21))
    n_per_facility = int(rng_design.integers(5, 101))
    sigma = float(rng_design.uniform(0.05, 0.6))
    target_delta = float(np.exp(rng_design.uniform(np.log(esc.DELTA_MIN),
                                                   np.log(esc.DELTA_MAX))))
    n_mode = "equal" if rng_design.uniform() < 0.5 else "unequal"

    # ctype_idx 1 == outlier_contaminated in esc.CONFOUND_TYPES, so the
    # generation stream is the same one the main pass used
    rng = np.random.default_rng(esc.RNG_SEED + 7_000_000 * 2 + rep)

    # target_delta is interpreted against the BASE sigma here: asking the
    # generator for target_delta * (sigma / sigma_eff) reproduces exactly that
    inflate = np.sqrt((1 - esc.CONTAM_FRAC)
                      + esc.CONTAM_FRAC * esc.CONTAM_SCALE ** 2)
    df, scale, rms_unabsorbed, sigma_eff = esc.gen_confounded_scenario(
        K, n_per_facility, sigma, target_delta / inflate,
        "outlier_contaminated", n_mode, rng)
    if df is None:
        return None

    delta_base_sigma = rms_unabsorbed / sigma
    r2_lofo = esc.lofo_pooled_r2(df)
    thr_says_universal = bool(np.isfinite(r2_lofo) and r2_lofo > 0)
    _, perm_p, perm_says_universal = esc.permutation_test(df, rng, nperm=99)
    return dict(rep=rep, K=K, n_per_facility=n_per_facility, sigma=sigma,
                target_delta=target_delta,
                delta_base_sigma=float(delta_base_sigma),
                n_mode=n_mode, r2_lofo=r2_lofo,
                thr_detects=(not thr_says_universal),
                perm_detects=(not perm_says_universal))


def main():
    from multiprocessing import Pool
    with Pool(processes=8) as pool:
        records = pool.map(run_one_altnorm, range(esc.N_PER_TYPE),
                           chunksize=25)
    df = pd.DataFrame.from_records([r for r in records if r is not None])
    df["delta_bin"] = pd.cut(df.target_delta,
                             bins=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
                                   16.0, np.inf],
                             labels=["<=0.1", "0.1-0.25", "0.25-0.5", "0.5-1",
                                     "1-2", "2-4", "4-8", "8-16", ">16"])

    main_pass = pd.read_csv(os.path.join(RESULTS,
                                         "effect_size_calibration.csv"))
    contaminated = main_pass[main_pass.confound_type == "outlier_contaminated"]
    gaussian = main_pass[main_pass.confound_type == "gaussian_linear"]

    summary = {
        "what": "outlier-contaminated mechanism, effect size normalised by the "
                "base sigma instead of the RMS of the contaminated error term",
        "n": int(len(df)),
        "pooled_detection_threshold_rule": {
            "normalised_by_contaminated_rms (main pass)":
                float(contaminated.thr_detects.mean()),
            "normalised_by_base_sigma (this check)":
                float(df.thr_detects.mean()),
            "gaussian_linear_for_reference":
                float(gaussian.thr_detects.mean()),
        },
        "pooled_detection_permutation_rule": {
            "normalised_by_contaminated_rms (main pass)":
                float(contaminated.perm_detects.mean()),
            "normalised_by_base_sigma (this check)":
                float(df.perm_detects.mean()),
            "gaussian_linear_for_reference":
                float(gaussian.perm_detects.mean()),
        },
        "by_delta_bin_threshold_rule": {},
    }
    for label, sub in df.groupby("delta_bin", observed=True):
        n = len(sub)
        if n:
            summary["by_delta_bin_threshold_rule"][str(label)] = {
                "n": int(n),
                "threshold_rule": float(sub.thr_detects.mean()),
                "permutation_rule": float(sub.perm_detects.mean()),
            }

    df.to_csv(os.path.join(RESULTS, "effect_size_altnorm.csv"), index=False)
    with open(os.path.join(RESULTS, "effect_size_altnorm.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("threshold rule, contaminated mechanism:")
    for k, v in summary["pooled_detection_threshold_rule"].items():
        print(f"  {k:48s} {v:.3f}")
    print("permutation rule, contaminated mechanism:")
    for k, v in summary["pooled_detection_permutation_rule"].items():
        print(f"  {k:48s} {v:.3f}")


if __name__ == "__main__":
    main()
