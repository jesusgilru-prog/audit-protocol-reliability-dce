"""
Extension of synth_audit_protocol.py: the first pass (results/synth_scenarios.csv)
used a single confounding mechanism (Gaussian heterogeneity in intercept+slopes).
This adds three more realistic/adversarial confounding mechanisms, plus unequal
per-facility sample sizes (matching the real windage case's pattern of very
unequal facility sizes, 45/41/20/8), and reruns the full sweep + meta-model.

Confound types (categorical, in addition to "universal" ground truth):
  - gaussian_linear   : same as v1 (intercept+slope drawn from a Normal spread)
  - outlier_contaminated : universal law, but each facility has a chance of a
    contamination sub-population with much larger noise (heavy-tailed, not
    parameter drift) -- tests whether the protocol confuses noise/outliers
    with genuine non-generalization.
  - nonlinear_misspecified : universal LINEAR coefficients but each facility
    adds its own quadratic term in log_re that the fitted model (log-linear)
    can't capture -- tests model misspecification, not parameter drift.
  - clustered : facilities belong to one of 2 latent clusters; within a
    cluster the law is IDENTICAL, across clusters it differs by
    `heterogeneity` -- tests whether the protocol can detect "partial"
    universality (a subgroup shares structure) vs the all-or-nothing framing
    of v1.

Unequal n: `n_mode` in {"equal", "unequal"} -- "unequal" draws facility sizes
from a lognormal spread around n_per_facility, echoing the real case's
45/41/20/8 pattern (mean ~28.5, high dispersion), instead of assuming every
facility has the same n.
"""
import json
import os

import numpy as np
import pandas as pd


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion k/n -- used to report
    uncertainty on the per-mechanism accuracy figures (raised in external
    review, ronda 1: 'ninguna cifra lleva incertidumbre')."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (float(center - half), float(center + half))

RNG_SEED = 20260818
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

B0_TRUE, A_TRUE, B_TRUE = 2.0, -0.45, 0.30


def facility_sizes(K, n_mean, n_mode, rng):
    if n_mode == "equal":
        return [n_mean] * K
    # lognormal spread around n_mean, clipped to [5, 150], matching the real
    # case's wide facility-size dispersion (45/41/20/8)
    raw = rng.lognormal(mean=np.log(max(n_mean, 5)), sigma=0.7, size=K)
    return [int(np.clip(round(v), 5, 150)) for v in raw]


def gen_points(k, n, b0_k, a_k, b_k, sigma, rng, contaminate=False,
                nonlinear_coef=0.0):
    log_re = rng.uniform(np.log(1e4), np.log(1e7), size=n)
    log_pg = rng.uniform(np.log(0.01), np.log(1.0), size=n)
    if contaminate:
        # 15% of points per facility come from a heavy-noise contamination
        # sub-population (5x the base sigma) -- outliers, not parameter drift
        is_out = rng.uniform(size=n) < 0.15
        eps = np.where(is_out, rng.normal(0, sigma * 5, size=n),
                       rng.normal(0, sigma, size=n))
    else:
        eps = rng.normal(0, sigma, size=n)
    nonlin = nonlinear_coef * (log_re - log_re.mean()) ** 2
    log_cp = b0_k + a_k * log_re + b_k * log_pg + nonlin + eps
    return pd.DataFrame(dict(facility=k, log_re=log_re, log_pg=log_pg, log_cp=log_cp))


def gen_scenario(K, n_per_facility, sigma, heterogeneity, is_universal,
                  confound_type, n_mode, rng):
    sizes = facility_sizes(K, n_per_facility, n_mode, rng)
    frames = []

    if confound_type == "clustered" and not is_universal:
        # 2 latent clusters; within-cluster identical law, across differs.
        #
        # BUG FIXED after external review (multi-model debate, 2026-08-20,
        # finding independently verified before fixing): rng.integers(0,2,K)
        # can put every facility in the same cluster by chance
        # (P = 2*0.5**K, e.g. 12.5% at K=4 up to 50% at K=2), which produces
        # a scenario with IDENTICAL law across all K facilities -- i.e.
        # genuinely universal -- while still labeled is_universal=False.
        # This silently biased the reported "when confounded" accuracy for
        # this mechanism downward. Fixed by resampling until both clusters
        # are populated (rejection sampling, K>=2 always makes this
        # terminate quickly).
        cluster_of = rng.integers(0, 2, size=K)
        while len(set(cluster_of)) < 2:
            cluster_of = rng.integers(0, 2, size=K)
        cluster_params = {
            0: (B0_TRUE, A_TRUE, B_TRUE),
            1: (B0_TRUE + rng.normal(0, heterogeneity),
                A_TRUE + rng.normal(0, heterogeneity * 0.3),
                B_TRUE + rng.normal(0, heterogeneity * 0.3)),
        }

    for k in range(K):
        n = sizes[k]
        contaminate = False
        nonlinear_coef = 0.0

        if is_universal:
            b0_k, a_k, b_k = B0_TRUE, A_TRUE, B_TRUE
            if confound_type == "outlier_contaminated":
                contaminate = True  # universal law + heavy noise, even in the "universal" arm
        else:
            if confound_type == "gaussian_linear":
                b0_k = rng.normal(B0_TRUE, heterogeneity)
                a_k = rng.normal(A_TRUE, heterogeneity * 0.3)
                b_k = rng.normal(B_TRUE, heterogeneity * 0.3)
            elif confound_type == "outlier_contaminated":
                # true non-generalization from parameter drift; contamination
                # noise on top makes it harder to tell apart from pure noise
                b0_k = rng.normal(B0_TRUE, heterogeneity)
                a_k = rng.normal(A_TRUE, heterogeneity * 0.3)
                b_k = rng.normal(B_TRUE, heterogeneity * 0.3)
                contaminate = True
            elif confound_type == "nonlinear_misspecified":
                b0_k, a_k, b_k = B0_TRUE, A_TRUE, B_TRUE  # linear coefs universal
                nonlinear_coef = rng.normal(0, heterogeneity * 0.15)
            elif confound_type == "clustered":
                b0_k, a_k, b_k = cluster_params[cluster_of[k]]
            else:
                raise ValueError(confound_type)

        frames.append(gen_points(k, n, b0_k, a_k, b_k, sigma, rng,
                                   contaminate=contaminate,
                                   nonlinear_coef=nonlinear_coef))
    return pd.concat(frames, ignore_index=True)


def fit_ols(df):
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    y = df.log_cp.values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def lofo_pooled_r2(df):
    facilities = df.facility.unique()
    if len(facilities) < 2:
        return np.nan
    preds, actuals = [], []
    for held_out in facilities:
        train = df[df.facility != held_out]
        test = df[df.facility == held_out]
        if len(train) < 3 or len(test) == 0:
            continue
        coef = fit_ols(train)
        X_test = np.column_stack([np.ones(len(test)), test.log_re.values, test.log_pg.values])
        pred = X_test @ coef
        preds.append(pred)
        actuals.append(test.log_cp.values)
    if not preds:
        return np.nan
    pred = np.concatenate(preds)
    actual = np.concatenate(actuals)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def bootstrap_reynolds_ci(df, n_boot, rng):
    """Same facility-level bootstrap as synth_audit_protocol.py (v1) --
    duplicated here (not imported) to keep this module self-contained and
    independently runnable, but implementing the identical method.
    Numpy-only inner loop (no per-draw pandas filtering) for speed -- see
    v1's docstring for why the naive pandas version was too slow."""
    facility_ids = df.facility.unique()
    if len(facility_ids) < 2:
        return np.nan, np.nan, False
    groups = {}
    for f in facility_ids:
        sub = df[df.facility == f]
        X = np.column_stack([np.ones(len(sub)), sub.log_re.values, sub.log_pg.values])
        groups[f] = (X, sub.log_cp.values)

    boot_a = []
    n_facilities = len(facility_ids)
    for _ in range(n_boot):
        sampled = rng.choice(facility_ids, size=n_facilities, replace=True)
        Xs = np.concatenate([groups[f][0] for f in sampled], axis=0)
        ys = np.concatenate([groups[f][1] for f in sampled], axis=0)
        if len(ys) < 4:
            continue
        coef, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        boot_a.append(coef[1])
    if len(boot_a) < 10:
        return np.nan, np.nan, False
    ci_lo, ci_hi = np.percentile(boot_a, [2.5, 97.5])
    excludes_zero = bool(ci_lo > 0 or ci_hi < 0)
    return float(ci_lo), float(ci_hi), excludes_zero


def observable_stats(df):
    """Same observable proxies as v1 (see its docstring) -- duplicated
    here to keep this module self-contained."""
    coef_pooled = fit_ols(df)
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    resid = df.log_cp.values - X @ coef_pooled
    obs_sigma_hat = float(np.std(resid))

    per_facility_a = []
    for f in df.facility.unique():
        sub = df[df.facility == f]
        if len(sub) < 4:
            continue
        coef_f = fit_ols(sub)
        per_facility_a.append(coef_f[1])
    obs_hetero_hat = float(np.std(per_facility_a)) if len(per_facility_a) >= 2 else np.nan
    return obs_sigma_hat, obs_hetero_hat


def _lofo_r2_fast(X, y, g):
    """Same as v1's _lofo_r2_fast -- see its docstring."""
    ug = np.unique(g)
    if len(ug) < 2:
        return np.nan
    Stot = X.T @ X
    btot = X.T @ y
    ss_res = 0.0
    for k in ug:
        m = g == k
        Xk, yk = X[m], y[m]
        S = Stot - Xk.T @ Xk
        b = btot - Xk.T @ yk
        try:
            coef = np.linalg.solve(S, b)
        except np.linalg.LinAlgError:
            return np.nan
        ss_res += float(np.sum((yk - Xk @ coef) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def permutation_test(df, rng, nperm=99):
    """Same facility-label permutation test as v1 -- see its docstring."""
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    y = df.log_cp.values
    g = df.facility.values
    obs = _lofo_r2_fast(X, y, g)
    if not np.isfinite(obs):
        return obs, np.nan, False
    cnt, n_ok = 0, 0
    for _ in range(nperm):
        gp = rng.permutation(g)
        r = _lofo_r2_fast(X, y, gp)
        if np.isfinite(r):
            n_ok += 1
            if r <= obs:
                cnt += 1
    if n_ok == 0:
        return obs, np.nan, False
    pval = (1 + cnt) / (1 + n_ok)
    perm_says_universal = bool(pval > 0.05)
    return obs, float(pval), perm_says_universal


def run_scenario(K, n_per_facility, sigma, heterogeneity, is_universal,
                  confound_type, n_mode, rng, n_boot=200, nperm=99):
    df = gen_scenario(K, n_per_facility, sigma, heterogeneity, is_universal,
                       confound_type, n_mode, rng)
    r2_lofo = lofo_pooled_r2(df)
    protocol_says_universal = bool(r2_lofo is not None and not np.isnan(r2_lofo) and r2_lofo > 0)
    boot_lo, boot_hi, boot_stable = bootstrap_reynolds_ci(df, n_boot, rng)
    obs_sigma_hat, obs_hetero_hat = observable_stats(df)
    _, perm_pvalue, perm_says_universal = permutation_test(df, rng, nperm=nperm)
    correct = protocol_says_universal == is_universal
    perm_correct = perm_says_universal == is_universal
    return dict(K=K, n_per_facility=n_per_facility, sigma=sigma,
                heterogeneity=heterogeneity, is_universal=is_universal,
                confound_type=confound_type, n_mode=n_mode,
                r2_lofo=r2_lofo, protocol_says_universal=protocol_says_universal,
                bootstrap_a_ci_lo=boot_lo, bootstrap_a_ci_hi=boot_hi,
                bootstrap_a_stable=boot_stable,
                obs_sigma_hat=obs_sigma_hat, obs_hetero_hat=obs_hetero_hat,
                perm_pvalue=perm_pvalue, perm_says_universal=perm_says_universal,
                perm_correct=perm_correct,
                correct=correct)


def main():
    rng = np.random.default_rng(RNG_SEED + 1)
    CONFOUND_TYPES = ["gaussian_linear", "outlier_contaminated",
                       "nonlinear_misspecified", "clustered"]
    N_PER_TYPE = 2000
    records = []
    for ctype in CONFOUND_TYPES:
        for i in range(N_PER_TYPE):
            K = int(rng.integers(2, 21))
            n_per_facility = int(rng.integers(5, 101))
            sigma = float(rng.uniform(0.05, 0.6))
            heterogeneity = float(rng.uniform(0.0, 3.0))
            is_universal = bool(rng.integers(0, 2))
            n_mode = "unequal" if rng.uniform() < 0.5 else "equal"
            rec = run_scenario(K, n_per_facility, sigma, heterogeneity,
                                is_universal, ctype, n_mode, rng)
            rec["scenario_id"] = f"{ctype}_{i}"
            records.append(rec)
        print(f"{ctype}: {N_PER_TYPE} scenarios done")

    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS, "synth_scenarios_v2.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {len(df)} scenarios to {out_csv}")

    summary = {}
    for ctype in CONFOUND_TYPES:
        sub = df[df.confound_type == ctype]
        boot_sub = sub.dropna(subset=["bootstrap_a_stable"])
        perm_sub = sub.dropna(subset=["perm_correct"])
        n_conf = int((~sub.is_universal).sum())
        k_conf = int((sub[~sub.is_universal].correct).sum())
        ci_conf = wilson_ci(k_conf, n_conf)
        n_univ = int(sub.is_universal.sum())
        k_univ = int((sub[sub.is_universal].correct).sum())
        ci_univ = wilson_ci(k_univ, n_univ)

        n_perm_conf = int((~perm_sub.is_universal).sum())
        k_perm_conf = int((perm_sub[~perm_sub.is_universal].perm_correct).sum())
        ci_perm_conf = wilson_ci(k_perm_conf, n_perm_conf)
        n_perm_univ = int(perm_sub.is_universal.sum())
        k_perm_univ = int((perm_sub[perm_sub.is_universal].perm_correct).sum())
        ci_perm_univ = wilson_ci(k_perm_univ, n_perm_univ)

        summary[ctype] = dict(
            n=len(sub),
            overall_accuracy=float(sub.correct.mean()),
            accuracy_when_universal=float(sub[sub.is_universal].correct.mean()),
            accuracy_when_universal_ci95=ci_univ,
            accuracy_when_confounded=float(sub[~sub.is_universal].correct.mean()),
            accuracy_when_confounded_ci95=ci_conf,
            bootstrap_a_stable_when_universal=float(boot_sub[boot_sub.is_universal].bootstrap_a_stable.mean()),
            bootstrap_a_stable_when_confounded=float(boot_sub[~boot_sub.is_universal].bootstrap_a_stable.mean()),
            permutation_overall_accuracy=float(perm_sub.perm_correct.mean()),
            permutation_accuracy_when_universal=float(perm_sub[perm_sub.is_universal].perm_correct.mean()),
            permutation_accuracy_when_universal_ci95=ci_perm_univ,
            permutation_accuracy_when_confounded=float(perm_sub[~perm_sub.is_universal].perm_correct.mean()),
            permutation_accuracy_when_confounded_ci95=ci_perm_conf,
        )
    for n_mode in ["equal", "unequal"]:
        sub = df[df.n_mode == n_mode]
        summary[f"n_mode_{n_mode}"] = dict(
            n=len(sub), overall_accuracy=float(sub.correct.mean()))

    # F3 (external review, 2026-08-20): majority-class trivial baseline,
    # pooled across all 8000 scenarios.
    summary["majority_class_baseline_accuracy"] = float(
        max(df.is_universal.mean(), 1 - df.is_universal.mean()))

    print("\n=== Summary by confound type ===")
    for ctype, s in summary.items():
        print(f"{ctype}: {json.dumps(s)}")

    with open(os.path.join(RESULTS, "synth_scenarios_v2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved results/synth_scenarios_v2_summary.json")


if __name__ == "__main__":
    main()
