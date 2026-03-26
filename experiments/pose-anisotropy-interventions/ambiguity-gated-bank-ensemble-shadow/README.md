# Ambiguity-Gated Bank Ensemble Shadow

## Purpose

The current working focused solver uses an entropy gate.

The ambiguity-width diagnostic established a sharper control signal:

- instability concentrates in `alpha`
- the ambiguity-width ratio predicts that instability better than entropy

This shadow experiment asks the lightest decisive next question:

- if the current focused ensemble solver keeps the same cached candidates, the
  same four-way chooser, and the same calibration-only thresholding protocol,
  does swapping the gate signal from entropy to ambiguity width improve fresh
  evaluation blocks?

## What Stays Fixed

This experiment is intentionally narrow.

It keeps fixed:

- the cached candidate families
- the four-way logistic chooser
- the default fallback candidate `dense_support`
- the calibration blocks
- the holdout and confirmation evaluation blocks

It changes only one thing:

- the scalar gate signal used to decide when to trust the chooser

## Policies Compared

The comparison is fully paired on the same trials:

1. `dense_support` default only
2. entropy gate
3. ambiguity gate

The entropy gate uses the current working solver feature:

- `d_joint_entropy`

The ambiguity gate uses the diagnostic signal:

- `mean_ambiguity_ratio`

Both thresholds are selected on calibration only with the same objective:

- minimize calibration mean `alpha` error
- tie-break on worst calibration block
- tie-break toward the median threshold candidate

## Decisive Criterion

The ambiguity gate counts as a benefit only if it beats the entropy gate on:

- holdout mean `alpha` error
- confirmation mean `alpha` error
- combined paired fresh-block delta

If it misses on either fresh block under this frozen protocol, the benefit is
not established.

## Artifacts

Data:

- [ambiguity_gated_bank_ensemble_shadow_trials.csv](outputs/ambiguity_gated_bank_ensemble_shadow_trials.csv)
- [ambiguity_gated_bank_ensemble_shadow_summary.json](outputs/ambiguity_gated_bank_ensemble_shadow_summary.json)

Code:

- [run.py](run.py)
