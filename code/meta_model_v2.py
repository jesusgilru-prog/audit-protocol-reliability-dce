"""
Meta-model v2: extends meta_model.py to the 8000-scenario, 4-confound-type
sweep (results/synth_scenarios_v2.csv). Adds `confound_type` (one-hot) and
`n_mode` as features, since v2's headline finding is that protocol
reliability depends heavily on the TYPE of confounding, not just its
strength -- a single meta-model across all four types, told which type it's
looking at, should reveal whether that dependence is learnable/predictable
in principle (relevant for a practitioner who suspects, but doesn't know for
certain, which confound type applies to their own data).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
RNG_SEED = 20260818

CONFOUND_TYPES = ["gaussian_linear", "outlier_contaminated",
                   "nonlinear_misspecified", "clustered"]


def main():
    df = pd.read_csv(os.path.join(RESULTS, "synth_scenarios_v2.csv"))
    df = df.dropna(subset=["obs_sigma_hat", "obs_hetero_hat"])
    for ct in CONFOUND_TYPES:
        df[f"ct_{ct}"] = (df.confound_type == ct).astype(float)
    df["n_mode_unequal"] = (df.n_mode == "unequal").astype(float)

    # NOTE: is_universal (ground truth) deliberately excluded -- see v1's
    # meta_model.py docstring for why an earlier version's inclusion of it
    # was a leakage bug caught in external review (ronda 1). protocol_says_universal
    # (the practitioner's own observed LOFO output) is used instead.
    # sigma/heterogeneity (LATENT generator settings, not observable --
    # second leakage-adjacent bug caught in external review, ronda 2)
    # replaced with obs_sigma_hat/obs_hetero_hat, computable from real
    # data alone (see observable_stats() in synth_audit_protocol.py).
    # confound_type is kept as an explicit "what-if I suspect this
    # mechanism" input, not something assumed known with certainty.
    features = ["K", "n_per_facility", "obs_sigma_hat", "obs_hetero_hat",
                "protocol_says_universal", "n_mode_unequal"] + [f"ct_{ct}" for ct in CONFOUND_TYPES]
    X = df[features].astype(float).values
    y = df["correct"].astype(int).values

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=RNG_SEED, stratify=df.confound_type)

    gbm = GradientBoostingClassifier(random_state=RNG_SEED, n_estimators=300, max_depth=4)
    gbm.fit(X_train, y_train)
    proba = gbm.predict_proba(X_test)[:, 1]
    pred = gbm.predict(X_test)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    print(f"Overall (all confound types pooled): accuracy={acc:.3f} AUC={auc:.3f}")

    imp = permutation_importance(gbm, X_test, y_test, n_repeats=20, random_state=RNG_SEED)
    importances = {f: float(m) for f, m in zip(features, imp.importances_mean)}
    print("\nPermutation importance:")
    for f, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {f:20s} {v:.4f}")

    # per-confound-type test accuracy, using the SAME pooled model (checks
    # whether one meta-model, given the confound type as input, correctly
    # tracks the very different reliability regimes found per type)
    per_type = {}
    for ct in CONFOUND_TYPES:
        mask = df_test.confound_type.values == ct
        if mask.sum() == 0:
            continue
        a = accuracy_score(y_test[mask], pred[mask])
        au = roc_auc_score(y_test[mask], proba[mask]) if len(set(y_test[mask])) > 1 else float("nan")
        per_type[ct] = dict(n=int(mask.sum()), accuracy=float(a), auc=float(au))
        print(f"  [{ct}] meta-model accuracy={a:.3f} auc={au:.3f}")

    out = dict(
        pooled_accuracy=float(acc),
        pooled_auc=float(auc),
        permutation_importance=importances,
        per_confound_type_meta_accuracy=per_type,
        raw_summary_by_type_and_ground_truth={
            ct: dict(
                overall=float(df[df.confound_type == ct].correct.mean()),
                when_universal=float(df[(df.confound_type == ct) & (df.is_universal)].correct.mean()),
                when_confounded=float(df[(df.confound_type == ct) & (~df.is_universal)].correct.mean()),
            ) for ct in CONFOUND_TYPES
        },
        rng_seed=RNG_SEED,
        n_scenarios_total=len(df),
    )
    with open(os.path.join(RESULTS, "meta_model_v2_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/meta_model_v2_results.json")


if __name__ == "__main__":
    main()
