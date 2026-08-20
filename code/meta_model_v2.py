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
    #
    # CORRECTED AGAIN after external review (Codex, ronda 5, 2026-08-18):
    # `confound_type` is NOT an observable quantity -- it is a hypothesis
    # the analyst brings, not something computable from data the way
    # obs_sigma_hat/obs_hetero_hat are. Folding it into a single "pooled"
    # accuracy number and calling the whole thing observable-only broke
    # the paper's own stated promise. This version trains and reports
    # TWO separate models: `observable_only` (K, n, obs_sigma_hat,
    # obs_hetero_hat, protocol_says_universal, n_mode_unequal -- nothing
    # a real practitioner couldn't compute) and `with_confound_type_oracle`
    # (the same features PLUS confound_type, explicitly labeled as a
    # what-if/oracle variant, not a claim about practical performance).
    observable_features = ["K", "n_per_facility", "obs_sigma_hat", "obs_hetero_hat",
                            "protocol_says_universal", "n_mode_unequal"]
    oracle_features = observable_features + [f"ct_{ct}" for ct in CONFOUND_TYPES]

    def fit_and_eval(features, label):
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
        # F3 (external review, 2026-08-20): same-target majority-class
        # baseline (predict the majority `correct` value on this test
        # split), not the unrelated is_universal majority-class rate.
        baseline_acc = float(max(y_test.mean(), 1 - y_test.mean()))
        print(f"[{label}] pooled accuracy={acc:.3f} AUC={auc:.3f}  "
              f"(baseline={baseline_acc:.3f})")

        imp = permutation_importance(gbm, X_test, y_test, n_repeats=20, random_state=RNG_SEED)
        importances = {f: float(m) for f, m in zip(features, imp.importances_mean)}
        print(f"[{label}] permutation importance:")
        for f, v in sorted(importances.items(), key=lambda kv: -kv[1]):
            print(f"    {f:20s} {v:.4f}")

        per_type = {}
        for ct in CONFOUND_TYPES:
            mask = df_test.confound_type.values == ct
            if mask.sum() == 0:
                continue
            a = accuracy_score(y_test[mask], pred[mask])
            au = roc_auc_score(y_test[mask], proba[mask]) if len(set(y_test[mask])) > 1 else float("nan")
            per_type[ct] = dict(n=int(mask.sum()), accuracy=float(a), auc=float(au))
            print(f"    [{ct}] accuracy={a:.3f} auc={au:.3f}")

        return dict(pooled_accuracy=float(acc), pooled_auc=float(auc),
                    majority_correct_baseline_accuracy=baseline_acc,
                    permutation_importance=importances,
                    per_confound_type_meta_accuracy=per_type, features=features)

    observable_result = fit_and_eval(observable_features, "observable_only")
    oracle_result = fit_and_eval(oracle_features, "with_confound_type_oracle")

    out = dict(
        observable_only=observable_result,
        with_confound_type_oracle=oracle_result,
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
