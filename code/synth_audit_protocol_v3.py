"""
THIRD PASS: covariate-support overlap as a design dimension.

Motivation (author's own check, 2026-08-25, after the ronda-10 review had
already converged). Both earlier passes draw every facility's covariates
from the SAME uniform range (synth_audit_protocol.py L65-66), i.e. the
facilities always share full covariate support. That is a silent and
consequential assumption, because the facility-label permutation test
recommended in this manuscript is only valid under exchangeability of
points across facility labels.

When facilities occupy DIFFERENT operating windows -- the normal situation
in real multi-source engineering data, and demonstrably the situation in
the companion manuscripts' own windage corpus -- leave-one-facility-out is
an EXTRAPOLATION task while the permuted pseudo-folds are INTERPOLATION.
The observed LOFO R^2 then sits in the lower tail of the permutation null
even under a genuinely universal law, and the test rejects. This pass
quantifies that, and evaluates a stratified permutation variant that
permutes facility labels only WITHIN covariate strata.

Three declaration rules are scored on every scenario:
  1. threshold          : R^2_LOFO > 0            (the companion studies' rule)
  2. permutation        : naive facility-label permutation
  3. permutation-strat  : permutation restricted to within-stratum swaps

Ground truth is declared by construction, so on the universal arm every
rejection is a false alarm and on the confounded arm every rejection is a
correct detection.
"""
import json
import os

import numpy as np
import pandas as pd

RNG_SEED = 20260825
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

B0_TRUE, A_TRUE, B_TRUE = 2.0, -0.45, 0.30
L_RE, U_RE = np.log(1e4), np.log(1e7)
L_PG, U_PG = np.log(0.01), np.log(1.0)
ALPHA = 0.05


def gen_scenario_overlap(K, n_per_facility, sigma, heterogeneity, is_universal,
                          overlap, rng):
    """As gen_scenario() in the first pass, but each facility's log_Re window
    is controlled by `overlap` in [0, 1]:

      overlap = 1 -> every facility spans the full range (identical to the
                     first two passes; complete shared support)
      overlap = 0 -> facilities tile the range in disjoint windows
                     (complete separation; facility identity is a
                     deterministic function of log_Re)

    The window width interpolates between W/K (disjoint tiling) and W
    (full range). log_Pg is left on the common range throughout, so
    `overlap` isolates separation along a single covariate -- the
    situation in the companion corpus, where facilities differ mainly in
    Reynolds window.
    """
    W = U_RE - L_RE
    h = (W / K) + overlap * (W - W / K)
    rows = []
    for k in range(K):
        if is_universal:
            b0_k, a_k, b_k = B0_TRUE, A_TRUE, B_TRUE
        else:
            b0_k = rng.normal(B0_TRUE, heterogeneity)
            a_k = rng.normal(A_TRUE, heterogeneity * 0.3)
            b_k = rng.normal(B_TRUE, heterogeneity * 0.3)
        centre = L_RE + (k + 0.5) * (W / K)
        lo, hi = centre - h / 2, centre + h / 2
        if lo < L_RE:                       # shift, do not shrink, at the edges
            lo, hi = L_RE, L_RE + h
        if hi > U_RE:
            lo, hi = U_RE - h, U_RE
        log_re = rng.uniform(lo, hi, size=n_per_facility)
        log_pg = rng.uniform(L_PG, U_PG, size=n_per_facility)
        eps = rng.normal(0, sigma, size=n_per_facility)
        log_cp = b0_k + a_k * log_re + b_k * log_pg + eps
        for i in range(n_per_facility):
            rows.append((k, log_re[i], log_pg[i], log_cp[i]))
    return pd.DataFrame(rows, columns=["facility", "log_re", "log_pg", "log_cp"])


def mean_pairwise_overlap(df, col="log_re"):
    """OBSERVABLE support-overlap statistic: mean over facility pairs of
    |intersection| / |union| of their covariate ranges.

    Deliberately computable from real data with no knowledge of the
    generating process -- this is the quantity a practitioner should
    measure on their own dataset before choosing a declaration rule.
    Reported for the companion windage corpus in the manuscript.
    """
    fac = df.facility.unique()
    if len(fac) < 2:
        return np.nan
    vals = []
    for i in range(len(fac)):
        for j in range(i + 1, len(fac)):
            a = df.loc[df.facility == fac[i], col]
            b = df.loc[df.facility == fac[j], col]
            inter = max(0.0, min(a.max(), b.max()) - max(a.min(), b.min()))
            union = max(a.max(), b.max()) - min(a.min(), b.min())
            vals.append(inter / union if union > 0 else 0.0)
    return float(np.mean(vals))


def _lofo_r2_fast(X, y, g):
    """Pooled LOFO R^2 via closed-form downdating of the normal equations.
    Identical definition to the first two passes."""
    ug = np.unique(g)
    if len(ug) < 2:
        return np.nan
    Stot, btot = X.T @ X, X.T @ y
    ss_res = 0.0
    for k in ug:
        m = g == k
        Xk, yk = X[m], y[m]
        try:
            coef = np.linalg.solve(Stot - Xk.T @ Xk, btot - Xk.T @ yk)
        except np.linalg.LinAlgError:
            return np.nan
        ss_res += float(np.sum((yk - Xk @ coef) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def quantile_strata(v, nbins):
    """Equal-count bins of a covariate, used to restrict permutations."""
    edges = np.quantile(v, np.linspace(0, 1, nbins + 1))
    return np.clip(np.digitize(v, edges[1:-1]), 0, nbins - 1)


def perm_pvalue(X, y, g, rng, nperm=99, strata=None):
    """p-value for 'the observed LOFO R^2 is worse than chance relabelling'.

    strata=None reproduces the naive test used in the first two passes.
    With strata, facility labels are permuted only within each stratum, so
    the null preserves the covariate-facility association and tests only
    for a coefficient difference beyond design separation.
    """
    obs = _lofo_r2_fast(X, y, g)
    if not np.isfinite(obs):
        return obs, np.nan
    cnt, n_ok = 0, 0
    for _ in range(nperm):
        if strata is None:
            gp = rng.permutation(g)
        else:
            gp = g.copy()
            for s in np.unique(strata):
                idx = np.where(strata == s)[0]
                gp[idx] = rng.permutation(gp[idx])
        r = _lofo_r2_fast(X, y, gp)
        if np.isfinite(r):
            n_ok += 1
            if r <= obs:
                cnt += 1
    if n_ok == 0:
        return obs, np.nan
    return obs, (1 + cnt) / (1 + n_ok)


def run_scenario(K, n_per_facility, sigma, heterogeneity, is_universal,
                  overlap, rng, nperm=99):
    df = gen_scenario_overlap(K, n_per_facility, sigma, heterogeneity,
                               is_universal, overlap, rng)
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    y = df.log_cp.values
    g = df.facility.values

    r2_lofo, p_naive = perm_pvalue(X, y, g, rng, nperm=nperm)
    strata = quantile_strata(df.log_re.values, nbins=max(2, K))
    _, p_strat = perm_pvalue(X, y, g, rng, nperm=nperm, strata=strata)

    thr_says_universal = bool(np.isfinite(r2_lofo) and r2_lofo > 0)
    perm_says_universal = bool(np.isfinite(p_naive) and p_naive > ALPHA)
    strat_says_universal = bool(np.isfinite(p_strat) and p_strat > ALPHA)

    return dict(
        K=K, n_per_facility=n_per_facility, sigma=sigma,
        heterogeneity=heterogeneity, is_universal=is_universal,
        overlap=overlap,
        obs_overlap=mean_pairwise_overlap(df),
        r2_lofo=r2_lofo, p_naive=p_naive, p_strat=p_strat,
        thr_says_universal=thr_says_universal,
        perm_says_universal=perm_says_universal,
        strat_says_universal=strat_says_universal,
        thr_correct=thr_says_universal == is_universal,
        perm_correct=perm_says_universal == is_universal,
        strat_correct=strat_says_universal == is_universal,
    )


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return float((c - m) / den), float((c + m) / den)


def main(n_scenarios=6000, nperm=99):
    rng = np.random.default_rng(RNG_SEED)
    recs = []
    for i in range(n_scenarios):
        K = int(rng.integers(3, 13))
        n = int(rng.integers(10, 80))
        sigma = float(rng.uniform(0.05, 0.6))
        het = float(rng.uniform(0.3, 2.0))
        is_universal = bool(rng.integers(0, 2))
        overlap = float(rng.uniform(0.0, 1.0))
        recs.append(run_scenario(K, n, sigma, het, is_universal, overlap,
                                  rng, nperm=nperm))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{n_scenarios} scenarios done", flush=True)

    df = pd.DataFrame(recs)
    out_csv = os.path.join(OUT_DIR, "synth_scenarios_v3.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {len(df)} scenarios to {out_csv}")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df["ov_bin"] = pd.cut(df.overlap, bins=bins, labels=labels,
                           include_lowest=True)

    summary = {"n_scenarios": int(len(df)), "rng_seed": RNG_SEED,
               "nperm": nperm, "alpha": ALPHA, "by_overlap_bin": []}

    print("\n=== FALSE ALARM RATE (universal arm; rejection = error) ===")
    print(f"{'overlap':>10} {'n':>5} {'threshold':>10} {'permutation':>12} "
          f"{'perm-strat':>11}")
    univ = df[df.is_universal]
    for lab in labels:
        s = univ[univ.ov_bin == lab]
        if len(s) == 0:
            continue
        fa_t = 1 - s.thr_correct.mean()
        fa_p = 1 - s.perm_correct.mean()
        fa_s = 1 - s.strat_correct.mean()
        print(f"{lab:>10} {len(s):>5} {fa_t:>10.3f} {fa_p:>12.3f} {fa_s:>11.3f}")

    print("\n=== POWER (confounded arm; rejection = correct detection) ===")
    print(f"{'overlap':>10} {'n':>5} {'threshold':>10} {'permutation':>12} "
          f"{'perm-strat':>11}")
    conf = df[~df.is_universal]
    for lab in labels:
        s = conf[conf.ov_bin == lab]
        if len(s) == 0:
            continue
        pw_t = s.thr_correct.mean()
        pw_p = s.perm_correct.mean()
        pw_s = s.strat_correct.mean()
        print(f"{lab:>10} {len(s):>5} {pw_t:>10.3f} {pw_p:>12.3f} {pw_s:>11.3f}")

    for lab in labels:
        u, c = univ[univ.ov_bin == lab], conf[conf.ov_bin == lab]
        if len(u) == 0 or len(c) == 0:
            continue
        entry = {"overlap_bin": lab, "n_universal": int(len(u)),
                 "n_confounded": int(len(c))}
        for rule, col in [("threshold", "thr_correct"),
                          ("permutation", "perm_correct"),
                          ("permutation_stratified", "strat_correct")]:
            fa = float(1 - u[col].mean())
            pw = float(c[col].mean())
            entry[f"{rule}_false_alarm"] = fa
            entry[f"{rule}_false_alarm_ci95"] = wilson(int((~u[col]).sum()), len(u))
            entry[f"{rule}_power"] = pw
            entry[f"{rule}_power_ci95"] = wilson(int(c[col].sum()), len(c))
        summary["by_overlap_bin"].append(entry)

    # Degenerate corner: overlap EXACTLY 0 (perfectly disjoint tiling). Kept
    # as its own small batch rather than a bin, because it is a measure-zero
    # corner of the uniform sweep above and the manuscript makes a specific
    # claim about it: with facility identity a deterministic function of the
    # covariates, each stratum holds a single facility, no label swap is
    # possible, and the stratified test has neither false alarms nor power.
    print("\n=== DEGENERATE CORNER: overlap exactly 0 ===")
    corner = {}
    for arm_universal in (True, False):
        rec = []
        for _ in range(150):
            K = int(rng.integers(3, 9))
            n = int(rng.integers(15, 60))
            sigma = float(rng.uniform(0.05, 0.6))
            het = float(rng.uniform(0.5, 1.5))
            rec.append(run_scenario(K, n, sigma, het, arm_universal, 0.0,
                                     rng, nperm=nperm))
        c = pd.DataFrame(rec)
        arm = "universal" if arm_universal else "confounded"
        corner[arm] = {
            "n": int(len(c)),
            "threshold": float(1 - c.thr_correct.mean() if arm_universal
                               else c.thr_correct.mean()),
            "permutation": float(1 - c.perm_correct.mean() if arm_universal
                                 else c.perm_correct.mean()),
            "permutation_stratified": float(1 - c.strat_correct.mean()
                                            if arm_universal
                                            else c.strat_correct.mean()),
        }
        metric = "false alarm" if arm_universal else "power"
        print(f"  {arm:<11} ({metric:<11}) thr={corner[arm]['threshold']:.3f} "
              f"perm={corner[arm]['permutation']:.3f} "
              f"strat={corner[arm]['permutation_stratified']:.3f}")
    summary["degenerate_corner_overlap_zero"] = corner

    summary["overall"] = {
        rule: {"accuracy": float(df[col].mean()),
               "false_alarm": float(1 - univ[col].mean()),
               "power": float(conf[col].mean())}
        for rule, col in [("threshold", "thr_correct"),
                          ("permutation", "perm_correct"),
                          ("permutation_stratified", "strat_correct")]
    }

    out_json = os.path.join(OUT_DIR, "synth_scenarios_v3_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
