# When Does a Cross-Domain Audit Protocol Actually Detect Universality?

Code, synthetic data, and results for the manuscript *"When Does a
Cross-Domain Audit Protocol Actually Detect Universality? A
Synthetic-Data Study of Scaling-Law Discovery Under Domain
Confounding"* (Jesús Gil Ruiz, Roberto A. Pava-Díaz, Carlos Enrique
Montenegro Marín; submitted to Data-Centric Engineering, Cambridge
University Press).

## What this is

A large-scale synthetic study of when a leave-one-facility-out (LOFO)
cross-validation + bootstrap audit protocol — used in two companion
manuscripts to detect whether a data-driven scaling law generalizes
across experimental facilities or is a facility-specific artifact —
itself reaches the correct verdict. Ground truth is synthetic and
declared by construction; no real experimental data is used or required
to reproduce any result here.

## Structure

- `code/synth_audit_protocol.py` — first-pass generator (4,000 scenarios,
  Gaussian parameter drift only) + LOFO + facility-level bootstrap.
- `code/synth_audit_protocol_v2.py` — second-pass generator (8,000
  scenarios, four confounding mechanisms × equal/unequal facility
  sizes).
- `code/synth_audit_protocol_v3.py` — third-pass generator (6,000
  scenarios) adding covariate-support overlap between facilities as a
  design dimension, plus the two observable overlap statistics
  (`mean_pairwise_overlap`, `min_fold_overlap`) and the stratified
  permutation variant. Includes the dedicated overlap-exactly-zero batch
  reported as the identifiability limit.
- `code/companion_overlap.py` — measures those same overlap statistics on
  the companion manuscripts' windage corpus. This is a description of
  that corpus's *design* (where each facility sits in Reynolds space),
  not a re-analysis of its windage relationship; the raw dataset is not
  redistributed here, only the aggregate statistics in
  `results/companion_overlap.json`. Skips cleanly if the dataset is not
  present.
- `code/meta_model.py`, `code/meta_model_v2.py` — gradient-boosting /
  Gaussian-process meta-models predicting verdict reliability from
  observable statistics only (see manuscript Method).
- `code/practical_guideline.py`, `code/threshold_sensitivity.py` —
  secondary analyses reported in Results.
- `code/effect_size_calibration.py` — fourth pass (12,000 scenarios): a
  paired design that runs all four confounding mechanisms at identical design
  points with the effect size matched by construction, where the effect size
  is the systematic structure the fitted model cannot absorb, in units of
  noise. This is the pass that qualifies the second pass's ordering of
  mechanisms.
- `code/effect_size_altnorm.py` — sensitivity check on how the contaminated
  mechanism is normalised in that comparison.
- `code/make_figure_effect_size.py` — the fourth pass's figure.
- `code/verify_manuscript_numbers.py` — checks every number the manuscript
  reports for the fourth pass against the result files; exits non-zero on any
  mismatch.
- `code/make_figures.py` — regenerates all manuscript figures from
  `results/`.
- `results/` — all raw scenario tables (`*.csv`) and summary statistics
  (`*.json`) referenced by number in the manuscript.
- `figures/` — manuscript figures, generated from `results/`.
- `manuscript.tex`, `refs.bib` — manuscript source.

## Reproducing

All scenario generation uses fixed random seeds, hardcoded as
`RNG_SEED` at the top of each generation script and reported in the
corresponding summary JSON: `20260818` for the first pass and the meta-models, `20260819` for the second
(the base seed plus one), `20260825` for the third and `20260922` for the
fourth. Requires Python 3.12
with `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

```
python code/synth_audit_protocol.py      # -> results/synth_scenarios.csv
python code/synth_audit_protocol_v2.py   # -> results/synth_scenarios_v2.csv
python code/synth_audit_protocol_v3.py   # -> results/synth_scenarios_v3.csv
python code/companion_overlap.py         # -> results/companion_overlap.json
python code/meta_model.py
python code/meta_model_v2.py
python code/practical_guideline.py
python code/threshold_sensitivity.py
python code/make_figures.py
```

## License

Code and data: MIT. See manuscript for text license upon publication.
