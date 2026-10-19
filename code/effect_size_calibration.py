"""Fourth pass: are the four confounding mechanisms comparable at a matched
effect size?

Motivation (raised as a limitation by the manuscript itself, and the most
likely referee objection to its headline result): the heterogeneity parameter
`h` enters each mechanism through a different fixed multiplier -- the intercept
at h, the slopes at 0.3h, the nonlinear quadratic coefficient at 0.15h -- so at
equal h the four mechanisms do not perturb the fitted model by equal amounts.
The reported ordering of mechanisms by detection difficulty could therefore be
an artifact of the generator's parameterisation rather than a property of the
mechanisms.

This pass fixes that by measuring, for every scenario, a mechanism-independent
effect size and then comparing detection rates *within* bins of it.

Effect size (delta): the systematic structure the pooled log-linear model
cannot absorb, in units of noise.

    mu_i        true noiseless mean at point i (facility-specific)
    r_i         residual of mu after projection onto the pooled design span
                [1, log_re, log_pg] -- i.e. the part of the facility-specific
                structure that no pooled linear fit can represent
    sigma_eff   RMS of the error term actually used in the scenario
                (contaminated arms: sqrt(0.85 + 0.15*25) = 2.145 sigma)

    delta = RMS(r) / sigma_eff

delta is 0 by construction on a universal scenario, and it is defined the same
way for all four mechanisms, which is the whole point. Only the confounded arm
is generated here: it is where the mechanisms differ and where the disputed
ordering lives.

Everything about the audit protocol itself (LOFO, the permutation test, the
observable proxies) is imported from the second-pass module, not reimplemented,
so this pass scores scenarios with exactly the same code as the results it is
meant to qualify.

Seed 20260922, independent of the first three passes.
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

from synth_audit_protocol_v2 import (  # noqa: E402
    A_TRUE, B_TRUE, B0_TRUE,
    facility_sizes, fit_ols, lofo_pooled_r2, observable_stats,
    permutation_test,
)

RNG_SEED = 20260922
CONFOUND_TYPES = ["gaussian_linear", "outlier_contaminated",
                  "nonlinear_misspecified", "clustered"]
N_PER_TYPE = 3000
DELTA_MIN = 0.05
DELTA_MAX = 40.0
CONTAM_FRAC = 0.15
CONTAM_SCALE = 5.0


def gen_points_with_mu(k, n, b0_k, a_k, b_k, sigma, rng, contaminate=False,
                       nonlinear_coef=0.0):
    """Identical generating process to v2's gen_points, but also returns the
    noiseless mean mu of every point. The draws are made in the same order and
    from the same distributions; the only addition is that mu is kept."""
    log_re = rng.uniform(np.log(1e4), np.log(1e7), size=n)
    log_pg = rng.uniform(np.log(0.01), np.log(1.0), size=n)
    if contaminate:
        is_out = rng.uniform(size=n) < CONTAM_FRAC
        eps = np.where(is_out, rng.normal(0, sigma * CONTAM_SCALE, size=n),
                       rng.normal(0, sigma, size=n))
    else:
        eps = rng.normal(0, sigma, size=n)
    nonlin = nonlinear_coef * (log_re - log_re.mean()) ** 2
    mu = b0_k + a_k * log_re + b_k * log_pg + nonlin
    return pd.DataFrame(dict(facility=k, log_re=log_re, log_pg=log_pg,
                             log_cp=mu + eps, mu=mu))


def gen_confounded_scenario(K, n_per_facility, sigma, target_delta,
                            confound_type, n_mode, rng):
    """Confounded arm with the effect size fixed by construction.

    Every mechanism's facility-specific deviation is linear in its scale
    parameter, so a scenario is built in two pieces: the universal mean, and
    the mechanism's deviation drawn at unit scale. The deviation is then
    multiplied by whatever factor makes the realised effect size equal
    `target_delta`. That replaces `h` -- whose meaning differs across
    mechanisms -- with a quantity that means the same thing in all four, and
    it does so without touching the *shape* of each mechanism's deviation,
    which is what distinguishes them.

    The clustered mechanism keeps v2's rejection sampling (at least two
    populated clusters)."""
    sizes = facility_sizes(K, n_per_facility, n_mode, rng)
    contaminate = confound_type == "outlier_contaminated"

    if confound_type == "clustered":
        cluster_of = rng.integers(0, 2, size=K)
        while len(set(cluster_of)) < 2:
            cluster_of = rng.integers(0, 2, size=K)
        cluster_dev = {0: (0.0, 0.0, 0.0),
                       1: (rng.normal(0, 1.0), rng.normal(0, 0.3),
                           rng.normal(0, 0.3))}

    # unit-scale deviations per facility: (d_b0, d_a, d_b, d_quad)
    devs = []
    for k in range(K):
        if confound_type in ("gaussian_linear", "outlier_contaminated"):
            devs.append((rng.normal(0, 1.0), rng.normal(0, 0.3),
                         rng.normal(0, 0.3), 0.0))
        elif confound_type == "nonlinear_misspecified":
            devs.append((0.0, 0.0, 0.0, rng.normal(0, 0.15)))
        elif confound_type == "clustered":
            d_b0, d_a, d_b = cluster_dev[cluster_of[k]]
            devs.append((d_b0, d_a, d_b, 0.0))
        else:
            raise ValueError(confound_type)

    frames = []
    for k in range(K):
        n = sizes[k]
        log_re = rng.uniform(np.log(1e4), np.log(1e7), size=n)
        log_pg = rng.uniform(np.log(0.01), np.log(1.0), size=n)
        if contaminate:
            is_out = rng.uniform(size=n) < CONTAM_FRAC
            eps = np.where(is_out, rng.normal(0, sigma * CONTAM_SCALE, size=n),
                           rng.normal(0, sigma, size=n))
        else:
            eps = rng.normal(0, sigma, size=n)
        d_b0, d_a, d_b, d_quad = devs[k]
        dev = (d_b0 + d_a * log_re + d_b * log_pg
               + d_quad * (log_re - log_re.mean()) ** 2)
        univ = B0_TRUE + A_TRUE * log_re + B_TRUE * log_pg
        frames.append(pd.DataFrame(dict(facility=k, log_re=log_re,
                                        log_pg=log_pg, univ=univ, dev=dev,
                                        eps=eps)))
    df = pd.concat(frames, ignore_index=True)

    # scale the deviation so the realised effect size hits the target
    sigma_eff = sigma * np.sqrt((1 - CONTAM_FRAC)
                                + CONTAM_FRAC * CONTAM_SCALE ** 2) \
        if contaminate else sigma
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    coef, *_ = np.linalg.lstsq(X, df.dev.values, rcond=None)
    unabsorbed = df.dev.values - X @ coef
    rms_unit = float(np.sqrt(np.mean(unabsorbed ** 2)))
    if rms_unit <= 0 or not np.isfinite(rms_unit):
        return None, np.nan, np.nan, np.nan
    scale = target_delta * sigma_eff / rms_unit

    df["mu"] = df.univ.values + scale * df.dev.values
    df["log_cp"] = df.mu.values + df.eps.values
    return (df[["facility", "log_re", "log_pg", "log_cp", "mu"]],
            float(scale), float(rms_unit * scale), float(sigma_eff))


def effect_size(df, sigma, contaminate):
    """Non-absorbable systematic structure, in units of noise."""
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    mu = df.mu.values
    coef, *_ = np.linalg.lstsq(X, mu, rcond=None)
    resid = mu - X @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if contaminate:
        sigma_eff = sigma * np.sqrt(
            (1 - CONTAM_FRAC) + CONTAM_FRAC * CONTAM_SCALE ** 2)
    else:
        sigma_eff = sigma
    return rms / sigma_eff, rms, float(sigma_eff)


def run_one(args):
    """Paired design: the scenario's design point -- K, n, sigma, the target
    effect size and the facility-size mode -- is drawn from a stream keyed only
    by the replicate index, so all four mechanisms are evaluated at the *same*
    design points. Only the shape of the facility-specific deviation differs
    between them, which is exactly the contrast of interest."""
    ctype, ctype_idx, rep = args
    rng_design = np.random.default_rng(RNG_SEED + rep)
    K = int(rng_design.integers(2, 21))
    n_per_facility = int(rng_design.integers(5, 101))
    sigma = float(rng_design.uniform(0.05, 0.6))
    # effect size drawn log-uniformly on a common range for all mechanisms
    target_delta = float(np.exp(rng_design.uniform(np.log(DELTA_MIN),
                                                   np.log(DELTA_MAX))))
    n_mode = "equal" if rng_design.uniform() < 0.5 else "unequal"

    rng = np.random.default_rng(RNG_SEED + 7_000_000 * (ctype_idx + 1) + rep)
    df, scale, rms_unabsorbed, sigma_eff = gen_confounded_scenario(
        K, n_per_facility, sigma, target_delta, ctype, n_mode, rng)
    if df is None:
        return None
    # realised effect size, recomputed from the generated mu (a check on the
    # construction, not a second definition of it)
    delta, _, _ = effect_size(df, sigma, ctype == "outlier_contaminated")

    r2_lofo = lofo_pooled_r2(df)
    thr_says_universal = bool(np.isfinite(r2_lofo) and r2_lofo > 0)
    _, perm_p, perm_says_universal = permutation_test(df, rng, nperm=99)
    obs_sigma_hat, obs_hetero_hat = observable_stats(df)

    # ground truth is "confounded" throughout this pass, so a correct verdict
    # is a verdict of NOT universal
    return dict(confound_type=ctype, rep=rep, K=K,
                n_per_facility=n_per_facility, sigma=sigma,
                target_delta=target_delta, dev_scale=scale, n_mode=n_mode,
                delta=delta, rms_unabsorbed=rms_unabsorbed,
                sigma_eff=sigma_eff,
                r2_lofo=r2_lofo,
                thr_detects=(not thr_says_universal),
                perm_pvalue=perm_p,
                perm_detects=(not perm_says_universal),
                obs_sigma_hat=obs_sigma_hat, obs_hetero_hat=obs_hetero_hat)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return float((c - s) / d), float((c + s) / d)


def main():
    from multiprocessing import Pool
    jobs = []
    for i, ctype in enumerate(CONFOUND_TYPES):
        for j in range(N_PER_TYPE):
            jobs.append((ctype, i, j))

    with Pool(processes=8) as pool:
        records = pool.map(run_one, jobs, chunksize=25)
    records = [r for r in records if r is not None]

    df = pd.DataFrame.from_records(records)
    out_csv = os.path.join(RESULTS, "effect_size_calibration.csv")
    df.to_csv(out_csv, index=False)

    # delta bins shared across mechanisms, on a log-ish scale because delta
    # spans orders of magnitude
    edges = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf]
    labels = ["<=0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", "4-8",
              "8-16", ">16"]
    df["delta_bin"] = pd.cut(df.delta, bins=edges, labels=labels,
                             include_lowest=True)

    summary = {
        "seed": RNG_SEED,
        "n_scenarios": int(len(df)),
        "n_per_type": N_PER_TYPE,
        "arm": "confounded only (ground truth = not universal)",
        "effect_size_definition":
            "RMS of the true mean's residual after projection on the pooled "
            "design span [1, log_re, log_pg], divided by the RMS of the "
            "scenario's error term",
        "delta_distribution_by_mechanism": {},
        "detection_by_delta_bin": {},
        "pooled_detection": {},
    }

    for ctype, sub in df.groupby("confound_type"):
        summary["delta_distribution_by_mechanism"][ctype] = {
            "median": float(sub.delta.median()),
            "q25": float(sub.delta.quantile(0.25)),
            "q75": float(sub.delta.quantile(0.75)),
            "min": float(sub.delta.min()),
            "max": float(sub.delta.max()),
        }
        summary["pooled_detection"][ctype] = {
            "n": int(len(sub)),
            "threshold_rule": float(sub.thr_detects.mean()),
            "permutation_rule": float(sub.perm_detects.mean()),
        }

    for label in labels:
        cell = {}
        for ctype in CONFOUND_TYPES:
            sub = df[(df.confound_type == ctype) & (df.delta_bin == label)]
            n = len(sub)
            if n == 0:
                continue
            k_thr = int(sub.thr_detects.sum())
            k_perm = int(sub.perm_detects.sum())
            cell[ctype] = {
                "n": n,
                "threshold_rule": k_thr / n,
                "threshold_rule_ci95": wilson(k_thr, n),
                "permutation_rule": k_perm / n,
                "permutation_rule_ci95": wilson(k_perm, n),
            }
        if cell:
            summary["detection_by_delta_bin"][label] = cell

    # paired comparison against the mechanism the manuscript reports as the
    # threshold rule's best case, on the identical design points
    wide = df.pivot_table(index="rep", columns="confound_type",
                          values="thr_detects")
    wide = wide.dropna()
    summary["paired_vs_gaussian_linear"] = {}
    for ctype in CONFOUND_TYPES:
        if ctype == "gaussian_linear" or ctype not in wide.columns:
            continue
        a = wide["gaussian_linear"].astype(bool).values
        b = wide[ctype].astype(bool).values
        n01 = int(np.sum(a & ~b))   # gaussian detects, this one does not
        n10 = int(np.sum(~a & b))
        n_disc = n01 + n10
        # exact McNemar (binomial) two-sided p
        from scipy.stats import binomtest
        pval = float(binomtest(n01, n_disc, 0.5).pvalue) if n_disc else 1.0
        summary["paired_vs_gaussian_linear"][ctype] = {
            "n_pairs": int(len(wide)),
            "gaussian_detects_other_misses": n01,
            "other_detects_gaussian_misses": n10,
            "difference_in_detection_rate": float(b.mean() - a.mean()),
            "mcnemar_exact_p": pval,
        }

    out_json = os.path.join(RESULTS, "effect_size_calibration.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"scenarios: {len(df)}  ->  {out_csv}")
    print("\npooled detection (confounded arm):")
    for ctype, d in summary["pooled_detection"].items():
        print(f"  {ctype:26s} thr={d['threshold_rule']:.3f}  "
              f"perm={d['permutation_rule']:.3f}  (n={d['n']})")
    print("\ndelta (effect size) distribution:")
    for ctype, d in summary["delta_distribution_by_mechanism"].items():
        print(f"  {ctype:26s} median={d['median']:.3f}  "
              f"IQR=[{d['q25']:.3f}, {d['q75']:.3f}]")
    print("\nthreshold-rule detection at MATCHED effect size:")
    hdr = f"  {'delta bin':12s}" + "".join(f"{c[:12]:>14s}" for c in CONFOUND_TYPES)
    print(hdr)
    for label, cell in summary["detection_by_delta_bin"].items():
        row = f"  {label:12s}"
        for ctype in CONFOUND_TYPES:
            if ctype in cell:
                row += f"{cell[ctype]['threshold_rule']:>9.3f}" \
                       f" ({cell[ctype]['n']:>3d})"
            else:
                row += f"{'-':>14s}"
        print(row)


if __name__ == "__main__":
    main()
