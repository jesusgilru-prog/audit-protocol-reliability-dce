"""
Advanced-ML piece of the synthetic study: train a meta-model on the
4000 synthetic scenarios (results/synth_scenarios.csv) to predict, from
information a REAL practitioner actually has -- experimental-design
counts (K facilities, n points per facility), two statistics computable
from their own data (obs_sigma_hat: residual std of the pooled fit;
obs_hetero_hat: std across facilities of each facility's own
individually-fit Reynolds-like coefficient -- see observable_stats() in
synth_audit_protocol.py), PLUS the protocol's own observed verdict
(protocol_says_universal, i.e. whether their own LOFO run came back
positive or negative) -- whether that verdict is likely correct.

CORRECTED TWICE after external review:
  - Ronda 1 (Codex, 2026-08-18): an earlier version included
    `is_universal` (the declared GROUND TRUTH label) as an input
    feature. Since the target `correct` is defined as
    `protocol_says_universal == is_universal`, using `is_universal` as a
    predictor leaks the answer directly into the model -- a practitioner
    never knows the ground truth in advance.
  - Ronda 2 (Codex, 2026-08-18): the fix above still used the synthetic
    generator's LATENT `sigma`/`heterogeneity` settings directly, which
    are not observable from real data either (they are generator
    parameters, not measured statistics). Replaced with obs_sigma_hat /
    obs_hetero_hat, which a real practitioner can compute.

This version predicts from (K, n, obs_sigma_hat, obs_hetero_hat,
protocol_says_universal) only: everything a practitioner running this
protocol on their own data actually observes or can compute.

Two models trained:
  1. Gradient Boosting classifier -> P(verdict correct | K, n,
     obs_sigma_hat, obs_hetero_hat, protocol_says_universal)
  2. Gaussian Process classifier (smaller subsample) as a probabilistic
     cross-check.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
RNG_SEED = 20260818


def main():
    df = pd.read_csv(os.path.join(RESULTS, "synth_scenarios.csv"))
    # is_universal (ground truth) is DELIBERATELY EXCLUDED -- see module
    # docstring. protocol_says_universal is the practitioner's own
    # observed LOFO verdict, always available in practice.
    #
    # CORRECTED after external review (Codex, ronda 2, 2026-08-18): the
    # previous version of this script used `sigma` and `heterogeneity`
    # directly -- both are LATENT generator settings, not observable from
    # real data, contradicting the claim that this meta-model uses "only
    # information a practitioner actually has". Replaced with
    # obs_sigma_hat (residual std of the pooled fit) and obs_hetero_hat
    # (std across facilities of each facility's own individually-fit
    # Reynolds-like coefficient) -- both computable from real data alone,
    # see observable_stats() in synth_audit_protocol.py.
    features = ["K", "n_per_facility", "obs_sigma_hat", "obs_hetero_hat", "protocol_says_universal"]
    df = df.dropna(subset=features)
    X = df[features].astype(float).values
    y = df["correct"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, stratify=y)

    gbm = GradientBoostingClassifier(random_state=RNG_SEED, n_estimators=200, max_depth=3)
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]
    gbm_acc = accuracy_score(y_test, gbm_pred)
    gbm_auc = roc_auc_score(y_test, gbm_proba)

    imp = permutation_importance(gbm, X_test, y_test, n_repeats=20, random_state=RNG_SEED)
    importances = {f: float(m) for f, m in zip(features, imp.importances_mean)}

    rng = np.random.default_rng(RNG_SEED)
    sub_idx = rng.choice(len(X_train), size=min(600, len(X_train)), replace=False)
    Xs, ys = X_train[sub_idx], y_train[sub_idx]
    Xs_mean, Xs_std = Xs.mean(0), Xs.std(0) + 1e-9
    Xs_n = (Xs - Xs_mean) / Xs_std
    Xtest_n = (X_test - Xs_mean) / Xs_std

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1]))
    gp = GaussianProcessClassifier(kernel=kernel, random_state=RNG_SEED, n_jobs=2)
    gp.fit(Xs_n, ys)
    gp_pred = gp.predict(Xtest_n)
    gp_proba = gp.predict_proba(Xtest_n)[:, 1]
    gp_acc = accuracy_score(y_test, gp_pred)
    gp_auc = roc_auc_score(y_test, gp_proba)

    print(f"GBM  test accuracy={gbm_acc:.3f}  AUC={gbm_auc:.3f}")
    print(f"GP   test accuracy={gp_acc:.3f}  AUC={gp_auc:.3f}  (trained on {len(sub_idx)} pts)")
    print("\nPermutation importance (GBM), predicting verdict correctness:")
    for f, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {f:24s} {v:.4f}")

    # ---- external anchor: the real windage case (Papers 1-2) ----
    # K=4 facilities, n = [45, 41, 20, 8] -> mean ~28.5.
    # protocol_says_universal=False is not an assumption -- it is the
    # OBSERVED fact from the companion study's own reported LOFO result
    # (pooled R^2_log = -0.885 < 0), i.e. the real verdict their own
    # protocol run produced. obs_sigma_hat and obs_hetero_hat WOULD be
    # computable from the real 114-point dataset, but doing so here would
    # mean re-analyzing the companion study's raw data, which this paper
    # deliberately avoids (see Introduction, scope). Both are therefore
    # SWEPT over a plausible range spanning this study's own training
    # distribution, rather than fixed at one asserted value -- this
    # anchor is an illustrative sensitivity range, not a point estimate.
    sigma_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    hetero_grid = [0.1, 0.3, 0.5, 0.8, 1.2]
    anchor_sweep = []
    for s in sigma_grid:
        for h in hetero_grid:
            real_case = dict(K=4, n_per_facility=28.5, obs_sigma_hat=s,
                              obs_hetero_hat=h, protocol_says_universal=0.0)
            x_real = np.array([[real_case[f] for f in features]])
            p_gbm = float(gbm.predict_proba(x_real)[0, 1])
            x_real_n = (x_real - Xs_mean) / Xs_std
            p_gp = float(gp.predict_proba(x_real_n)[0, 1])
            anchor_sweep.append(dict(obs_sigma_hat=s, obs_hetero_hat=h,
                                       gbm_p_correct=p_gbm, gp_p_correct=p_gp))

    gbm_ps = [r["gbm_p_correct"] for r in anchor_sweep]
    gp_ps = [r["gp_p_correct"] for r in anchor_sweep]
    print(f"\nReal windage case anchor (K=4, n~28.5, protocol_says_universal=False "
          f"-- the OBSERVED verdict -- swept over plausible obs_sigma_hat x "
          f"obs_hetero_hat, {len(anchor_sweep)} grid points):")
    print(f"  GBM P(verdict correct): min={min(gbm_ps):.3f} max={max(gbm_ps):.3f} "
          f"mean={np.mean(gbm_ps):.3f}")
    print(f"  GP  P(verdict correct): min={min(gp_ps):.3f} max={max(gp_ps):.3f} "
          f"mean={np.mean(gp_ps):.3f}")

    out = dict(
        gbm=dict(test_accuracy=float(gbm_acc), test_auc=float(gbm_auc),
                  permutation_importance=importances),
        gp=dict(test_accuracy=float(gp_acc), test_auc=float(gp_auc), n_train=len(sub_idx)),
        real_windage_case_anchor_sweep=anchor_sweep,
        real_windage_case_anchor_summary=dict(
            gbm_p_correct_min=float(min(gbm_ps)), gbm_p_correct_max=float(max(gbm_ps)),
            gbm_p_correct_mean=float(np.mean(gbm_ps)),
            gp_p_correct_min=float(min(gp_ps)), gp_p_correct_max=float(max(gp_ps)),
            gp_p_correct_mean=float(np.mean(gp_ps)),
        ),
        anchor_note="obs_sigma_hat and obs_hetero_hat are swept over a plausible "
                    "range rather than fixed, because computing their real values "
                    "for the windage case would require re-analyzing the companion "
                    "study's raw data, which this paper deliberately avoids. "
                    "protocol_says_universal=False is the OBSERVED verdict "
                    "(pooled LOFO R2=-0.885<0) from the companion study, not an "
                    "assumption.",
        rng_seed=RNG_SEED,
        n_scenarios_total=len(df),
        features_used=features,
    )
    with open(os.path.join(RESULTS, "meta_model_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved results/meta_model_results.json")


if __name__ == "__main__":
    main()
