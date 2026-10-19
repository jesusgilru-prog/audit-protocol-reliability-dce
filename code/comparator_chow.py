"""Auxiliary batch: direct equality-of-regressions F test (Chow-type) against
the LOFO threshold and the facility-label permutation test, on the second-pass
generator. Seed 20261019. Also stores realised facility sizes."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
import synth_audit_protocol_v2 as v2

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

def chow_f(df):
    X = np.column_stack([np.ones(len(df)), df.log_re.values, df.log_pg.values])
    y = df.log_cp.values; g = df.facility.values
    rss_r = np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)
    rss_u, p_u = 0.0, 0
    for k in np.unique(g):
        m = g == k
        if m.sum() <= 3:
            return np.nan
        b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        rss_u += np.sum((y[m] - X[m] @ b) ** 2); p_u += 3
    K = len(np.unique(g)); df1 = p_u - 3; df2 = len(y) - p_u
    if df2 <= 0 or rss_u <= 0:
        return np.nan
    F = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return float(stats.f.sf(F, df1, df2))

def main(n_per=1000):
    rng = np.random.default_rng(20261019)
    types = ["gaussian_linear", "outlier_contaminated", "nonlinear_misspecified", "clustered"]
    recs = []
    for ct in types:
        for i in range(n_per):
            K = int(rng.integers(2, 21)); n = int(rng.integers(5, 101))
            sigma = float(rng.uniform(0.05, 0.6)); h = float(rng.uniform(0, 3))
            uni = bool(rng.integers(0, 2)); mode = "unequal" if rng.uniform() < 0.5 else "equal"
            df = v2.gen_scenario(K, n, sigma, h, uni, ct, mode, rng)
            sizes = df.groupby("facility").size().tolist()
            r2 = v2.lofo_pooled_r2(df)
            thr = bool(r2 is not None and not np.isnan(r2) and r2 > 0)
            _, pp, pu = v2.permutation_test(df, rng, nperm=99)
            pf = chow_f(df)
            recs.append(dict(confound_type=ct, K=K, n_nominal=n, sigma=sigma, heterogeneity=h,
                             is_universal=uni, n_mode=mode, realised_sizes=json.dumps(sizes),
                             r2_lofo=r2, thr_says_universal=thr, perm_pvalue=pp,
                             perm_says_universal=pu, chow_pvalue=pf,
                             chow_says_universal=bool(np.isfinite(pf) and pf > 0.05)))
        print(ct, "done", flush=True)
    d = pd.DataFrame(recs); d.to_csv(os.path.join(RESULTS, "comparator_chow.csv"), index=False)
    out = {"seed": 20261019, "n": len(d)}
    for name, col in [("threshold", "thr_says_universal"), ("permutation", "perm_says_universal"), ("chow_F", "chow_says_universal")]:
        c = d[~d.is_universal]; u = d[d.is_universal]
        out[name] = dict(power=float((~c[col]).mean()), false_alarm=float((~u[col]).mean()),
                         accuracy=float((d[col] == d.is_universal).mean()))
        for ct in types:
            cc = c[c.confound_type == ct]; uu = u[u.confound_type == ct]
            out[name][ct] = dict(power=float((~cc[col]).mean()), false_alarm=float((~uu[col]).mean()))
    json.dump(out, open(os.path.join(RESULTS, "comparator_chow.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))
main()
