"""
Measures the covariate-support overlap of the companion manuscripts'
windage corpus, using the SAME observable statistic the third pass uses
(mean_pairwise_overlap in synth_audit_protocol_v3.py).

Scope note. This manuscript deliberately does not re-analyse the companion
studies' raw data, and this script does not either: it computes a
DESIGN statistic (where each facility sits in Reynolds space), not a
re-estimate of the windage relationship. It is reported for the same
reason the facility sizes 45/41/20/8 already are -- the reader needs it to
judge which declaration rule applies to that case. Per-facility Reynolds
windows are in any case readable from the four original published sources.

The raw dataset is NOT copied into this repository; only the aggregate
statistic is written to results/companion_overlap.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth_audit_protocol_v3 import (  # noqa: E402
    mean_pairwise_overlap, min_fold_overlap, min_fold_point_overlap)

RESULTS = os.path.join(HERE, "..", "results")
COMPANION_CSV = "/home/jesus/paper_windage_power/data/cross_rotor_dataset_v3.csv"


# The four log10(Re) windows exactly as printed in the manuscript's
# Appendix A. Both overlap statistics are determined by these eight numbers
# alone, so a reader who has only the paper can reproduce the corpus-level
# claim without the dataset (Codex, ronda 13, raised the dependence on a
# non-redistributed file). Recomputing from these rounded windows agrees
# with the full-data values to within the rounding.
PUBLISHED_LOG10_RE_WINDOWS = {
    "Guo2024": (6.80, 7.91),
    "Liu2024": (5.47, 6.82),
    "Vrancik1968": (4.96, 6.72),
    "Zheng2024": (5.57, 6.51),
}


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def from_published_windows(windows=None):
    """Both statistics from the published windows only, no dataset needed."""
    w = {k: (lo * np.log(10), hi * np.log(10))
         for k, (lo, hi) in (windows or PUBLISHED_LOG10_RE_WINDOWS).items()}
    ks = list(w)
    pairwise = [_iou(w[ks[i]], w[ks[j]])
                for i in range(len(ks)) for j in range(i + 1, len(ks))]
    folds = []
    for k in ks:
        rest = (min(w[o][0] for o in ks if o != k),
                max(w[o][1] for o in ks if o != k))
        folds.append(_iou(w[k], rest))
    return {"mean_pairwise_overlap": float(np.mean(pairwise)),
            "min_fold_overlap": float(np.min(folds))}


def _worst_fold_counts(d, col="log_re"):
    """Point counts inside the shared region for the worst fold, so the
    range-vs-count discrepancy is visible in the artefact itself."""
    worst, best = None, np.inf
    for k in d.facility.unique():
        held = d.loc[d.facility == k, col]
        rest = d.loc[d.facility != k, col]
        lo, hi = rest.min(), rest.max()
        frac = float(((held >= lo) & (held <= hi)).mean())
        if frac < best:
            best, worst = frac, k
    held = d.loc[d.facility == worst, col]
    rest = d.loc[d.facility != worst, col]
    lo = max(held.min(), rest.min()); hi = min(held.max(), rest.max())
    return dict(
        facility=str(worst),
        shared_region_log10=[float(lo / np.log(10)), float(hi / np.log(10))],
        held_out_points_inside=int(((held >= lo) & (held <= hi)).sum()),
        held_out_points_total=int(len(held)),
        other_points_inside=int(((rest >= lo) & (rest <= hi)).sum()),
        other_points_total=int(len(rest)),
    )


def main():
    pub = from_published_windows()
    print("From the four published windows in Appendix A alone "
          "(no dataset required):")
    print(f"  mean pairwise overlap = {pub['mean_pairwise_overlap']:.3f}")
    print(f"  min fold overlap      = {pub['min_fold_overlap']:.4f}\n")

    if not os.path.exists(COMPANION_CSV):
        print(f"companion dataset not available at {COMPANION_CSV}; "
              f"reporting the published-window values only")
        with open(os.path.join(RESULTS, "companion_overlap.json"), "w") as f:
            json.dump({"from_published_windows": pub,
                       "note": "dataset not present; published-window "
                               "recomputation only"}, f, indent=2)
        return
    d = pd.read_csv(COMPANION_CSV).dropna(subset=["Re_Omega", "Cp"])
    d = d.rename(columns={"source": "facility"})
    d["log_re"] = np.log(d.Re_Omega.values)

    windows = {}
    for f, sub in d.groupby("facility"):
        windows[f] = dict(
            n=int(len(sub)),
            log10_re_min=float(np.log10(sub.Re_Omega.min())),
            log10_re_max=float(np.log10(sub.Re_Omega.max())),
        )

    pairwise = {}
    facs = sorted(d.facility.unique())
    for i in range(len(facs)):
        for j in range(i + 1, len(facs)):
            a = np.log10(d.loc[d.facility == facs[i], "Re_Omega"])
            b = np.log10(d.loc[d.facility == facs[j], "Re_Omega"])
            inter = max(0.0, min(a.max(), b.max()) - max(a.min(), b.min()))
            union = max(a.max(), b.max()) - min(a.min(), b.min())
            pairwise[f"{facs[i]}|{facs[j]}"] = float(inter / union) if union > 0 else 0.0

    # Per-fold overlap: held-out facility vs. the union of the others. This
    # is the quantity the declaration rule keys on (see min_fold_overlap).
    fold = {}
    for k in facs:
        held = np.log(d.loc[d.facility == k, "Re_Omega"])
        rest = np.log(d.loc[d.facility != k, "Re_Omega"])
        inter = max(0.0, min(held.max(), rest.max()) - max(held.min(), rest.min()))
        union = max(held.max(), rest.max()) - min(held.min(), rest.min())
        fold[k] = float(inter / union) if union > 0 else 0.0

    out = dict(
        n_points=int(len(d)),
        n_facilities=len(facs),
        per_facility_log10_Re_window=windows,
        pairwise_overlap=pairwise,
        mean_pairwise_overlap=float(np.mean(list(pairwise.values()))),
        mean_pairwise_overlap_via_shared_function=mean_pairwise_overlap(d),
        per_fold_overlap=fold,
        min_fold_overlap=min_fold_overlap(d),
        min_fold_point_overlap=min_fold_point_overlap(d),
        points_in_shared_region_worst_fold=_worst_fold_counts(d),
        source_file=COMPANION_CSV,
        from_published_windows=pub,
        note=("Design statistic only (where each facility sits in Reynolds "
              "space), not a re-analysis of the windage relationship. Raw "
              "data not redistributed here."),
    )

    print(f"N = {out['n_points']} across {out['n_facilities']} facilities")
    print("\nlog10(Re) window per facility:")
    for f, w in windows.items():
        print(f"  {f:<14} n={w['n']:>3}  [{w['log10_re_min']:.2f}, {w['log10_re_max']:.2f}]")
    print("\npairwise support overlap (intersection/union):")
    for k, v in pairwise.items():
        print(f"  {k:<32} {v:.3f}")
    print(f"\nmean pairwise overlap = {out['mean_pairwise_overlap']:.3f}")
    print("\nper-fold overlap (held-out facility vs. union of the rest):")
    for k, v in fold.items():
        print(f"  {k:<14} {v:.4f}")
    print(f"\nMIN fold overlap (range)  = {out['min_fold_overlap']:.4f}")
    print(f"MIN fold overlap (points) = {out['min_fold_point_overlap']:.4f}")
    w = out["points_in_shared_region_worst_fold"]
    print(f"  worst fold: {w['facility']}, shared region log10Re "
          f"[{w['shared_region_log10'][0]:.3f}, {w['shared_region_log10'][1]:.3f}] "
          f"contains {w['held_out_points_inside']}/{w['held_out_points_total']} "
          f"held-out points and {w['other_points_inside']}/{w['other_points_total']} others")

    path = os.path.join(RESULTS, "companion_overlap.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
