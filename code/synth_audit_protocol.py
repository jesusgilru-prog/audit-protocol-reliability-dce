"""
Synthetic study: when does a LOFO+bootstrap scaling-law audit protocol
(the same class of method validated empirically in Papers 1-2 on the
real 114-point windage dataset) correctly tell universal structure
apart from facility-specific confounding?

Ground truth is synthetic and known by construction (declared, not
inferred) -- this is NOT a re-analysis of the real windage data. The
real case (K=4 facilities, n=[45,41,20,8]) is used only as a single
anchor point at the end, for external validity, citing Papers 1-2's
already-published numbers, not re-deriving them.

Two ground-truth regimes per scenario:
  - "universal":  one shared (a, b, intercept) law across all K facilities,
                  only iid noise (sigma) differs per point.
  - "confounded": each facility draws its OWN (a_k, b_k, intercept_k)
                  independently from a spread controlled by
                  `heterogeneity`, i.e. genuinely no shared law.

For each scenario we fit ONE pooled log-linear model (OLS in log space,
same functional form Papers 1-2 use: log(Cp) = b0 + a*log(Re) + b*log(Pg)),
run leave-one-facility-out (LOFO) cross-validation and report the pooled
R^2 in log space (same metric already used and validated in Papers 1-2).
"protocol says universal" := pooled R^2_LOFO > 0.

We then generate thousands of scenarios varying (K, n_per_facility, sigma,
heterogeneity) and train a small ML meta-model (gradient boosting
classifier) to predict, from those four design parameters, whether the
audit protocol reaches the CORRECT verdict (matches the declared ground
truth). This is real ML applied to a genuinely large synthetic sample
(no small-N overfitting risk, unlike applying deep ML directly to the
114-point windage dataset -- see BRIEF.md section 6/8 for why that path
was rejected).
"""
import json
import os

import numpy as np
import pandas as pd

RNG_SEED = 20260818
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def gen_scenario(K, n_per_facility, sigma, heterogeneity, is_universal, rng):
    """Generate one synthetic multi-facility dataset in log space.

    log(Cp) = b0_k + a_k*log(Re) + b_k*log(Pg) + eps,  eps ~ N(0, sigma^2)

    If is_universal: (b0_k, a_k, b_k) = (b0, a, b) for every facility k
    (shared law, only noise differs).
    If not: (b0_k, a_k, b_k) drawn independently per facility from
    N((b0,a,b), heterogeneity^2) -- genuinely no shared law by construction.
    """
    b0_true, a_true, b_true = 2.0, -0.45, 0.30  # loosely matches the real
    # windage fit (Re exponent around -0.45, per paper_windage_power bootstrap)
    rows = []
    for k in range(K):
        if is_universal:
            b0_k, a_k, b_k = b0_true, a_true, b_true
        else:
            b0_k = rng.normal(b0_true, heterogeneity)
            a_k = rng.normal(a_true, heterogeneity * 0.3)
            b_k = rng.normal(b_true, heterogeneity * 0.3)
        log_re = rng.uniform(np.log(1e4), np.log(1e7), size=n_per_facility)
        log_pg = rng.uniform(np.log(0.01), np.log(1.0), size=n_per_facility)
        eps = rng.normal(0, sigma, size=n_per_facility)
        log_cp = b0_k + a_k * log_re + b_k * log_pg + eps
        for i in range(n_per_facility):
            rows.append(dict(facility=k, log_re=log_re[i], log_pg=log_pg[i],
                              log_cp=log_cp[i]))
    return pd.DataFrame(rows)


def fit_ols(df):
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    y = df.log_cp.values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef  # [b0, a, b]


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
    """Bootstrap by resampling FACILITIES with replacement (not individual
    points), matching the companion windage study's own protocol
    ("bootstrap por instalacion"). Refits the pooled OLS model on each
    resample and returns a percentile 95% CI for the Reynolds-like
    exponent `a` (the 2nd coefficient). Returns (ci_lo, ci_hi,
    excludes_zero) -- `excludes_zero` is the stability/significance flag
    Papers 1-2 use to say a coefficient "survives" bootstrap.

    This was MISSING from the first submitted draft of this study, which
    claimed a "LOFO+bootstrap" protocol while only implementing LOFO --
    caught in external review (Codex, ronda 1, 2026-08-18) and fixed here.

    Performance note: the first implementation re-filtered the pandas
    DataFrame per facility on every bootstrap draw (~n_boot x K pandas
    boolean-mask scans per scenario), which made the full 4000/8000-scenario
    sweeps impractically slow (still running after 10+ minutes, killed).
    This version groups each facility's design matrix and target into plain
    numpy arrays ONCE per scenario, then does pure numpy concatenation
    inside the bootstrap loop -- same statistical method, ~2 orders of
    magnitude faster.
    """
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
    """Statistics a real practitioner CAN compute from their own data,
    without knowing the synthetic ground-truth parameters `sigma` and
    `heterogeneity` (which are latent generator settings, not observable
    quantities -- flagged correctly in external review, ronda 2, as a
    contradiction with the claim that the meta-model uses "only
    information a practitioner actually has"). Two observable proxies:

    - obs_sigma_hat: residual std of the single pooled OLS fit (a real
      practitioner always has this).
    - obs_hetero_hat: std, across facilities, of each facility's OWN
      individually-fit Reynolds-like coefficient `a` -- computable
      without knowing whether the underlying law is truly universal,
      and correlated with (but not identical to) the latent
      `heterogeneity` parameter.
    """
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


def run_scenario(K, n_per_facility, sigma, heterogeneity, is_universal, rng,
                  n_boot=200):
    df = gen_scenario(K, n_per_facility, sigma, heterogeneity, is_universal, rng)
    r2_lofo = lofo_pooled_r2(df)
    protocol_says_universal = bool(r2_lofo is not None and not np.isnan(r2_lofo) and r2_lofo > 0)
    boot_lo, boot_hi, boot_stable = bootstrap_reynolds_ci(df, n_boot, rng)
    obs_sigma_hat, obs_hetero_hat = observable_stats(df)
    correct = protocol_says_universal == is_universal
    return dict(K=K, n_per_facility=n_per_facility, sigma=sigma,
                heterogeneity=heterogeneity, is_universal=is_universal,
                r2_lofo=r2_lofo, protocol_says_universal=protocol_says_universal,
                bootstrap_a_ci_lo=boot_lo, bootstrap_a_ci_hi=boot_hi,
                bootstrap_a_stable=boot_stable,
                obs_sigma_hat=obs_sigma_hat, obs_hetero_hat=obs_hetero_hat,
                correct=correct)


def main():
    rng = np.random.default_rng(RNG_SEED)
    N_SCENARIOS = 4000
    records = []
    for i in range(N_SCENARIOS):
        K = int(rng.integers(2, 21))
        n_per_facility = int(rng.integers(5, 101))
        sigma = float(rng.uniform(0.05, 0.6))
        heterogeneity = float(rng.uniform(0.0, 3.0))
        is_universal = bool(rng.integers(0, 2))
        rec = run_scenario(K, n_per_facility, sigma, heterogeneity, is_universal, rng)
        rec["scenario_id"] = i
        records.append(rec)
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{N_SCENARIOS} scenarios done")

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "synth_scenarios.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {len(df)} scenarios to {out_csv}")

    overall_acc = df.correct.mean()
    print(f"\noverall protocol accuracy (correct verdict vs ground truth): {overall_acc:.3f}")
    print(f"  accuracy when truly universal: {df[df.is_universal].correct.mean():.3f}  "
          f"(n={df.is_universal.sum()})")
    print(f"  accuracy when truly confounded: {df[~df.is_universal].correct.mean():.3f}  "
          f"(n={(~df.is_universal).sum()})")

    boot_valid = df.dropna(subset=["bootstrap_a_stable"])
    boot_stable_when_universal = float(boot_valid[boot_valid.is_universal].bootstrap_a_stable.mean())
    boot_stable_when_confounded = float(boot_valid[~boot_valid.is_universal].bootstrap_a_stable.mean())
    print(f"\nbootstrap (facility-resample, n_boot=200) stability of the "
          f"Reynolds-like exponent (CI excludes zero):")
    print(f"  when truly universal:  {boot_stable_when_universal:.3f}")
    print(f"  when truly confounded: {boot_stable_when_confounded:.3f}")

    summary = dict(
        n_scenarios=N_SCENARIOS,
        rng_seed=RNG_SEED,
        overall_accuracy=float(overall_acc),
        accuracy_when_universal=float(df[df.is_universal].correct.mean()),
        accuracy_when_confounded=float(df[~df.is_universal].correct.mean()),
        bootstrap_a_stable_when_universal=boot_stable_when_universal,
        bootstrap_a_stable_when_confounded=boot_stable_when_confounded,
        bootstrap_n_boot_per_scenario=200,
        bootstrap_method="facility-level resampling with replacement, percentile 95% CI on the Reynolds-like exponent (2nd OLS coefficient)",
        ground_truth_law="log(Cp) = 2.0 - 0.45*log(Re) + 0.30*log(Pg) (+ per-facility deviation if confounded)",
    )
    with open(os.path.join(OUT_DIR, "synth_scenarios_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary to results/synth_scenarios_summary.json")


if __name__ == "__main__":
    main()
