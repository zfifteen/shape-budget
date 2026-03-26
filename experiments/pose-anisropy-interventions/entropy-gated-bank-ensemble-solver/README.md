# Entropy-Gated Bank Ensemble Solver

## Purpose

This experiment resolves the focused solver bottleneck by combining the two
bank-adaptive candidate generators that remained complementary under fresh-bank
evaluation:

- `baseline` bank-adaptive candidates
- `density_ablation` bank-adaptive candidates

The solver keeps the forward model and latent control object fixed.
It changes only the inverse policy:

1. fit a frozen four-way logistic chooser on calibration blocks only
2. default to the stable `dense_support` candidate
3. open an observability gate only when `dense_joint_entropy` is high enough
4. when the gate opens, trust the frozen four-way chooser

## Why This Solver

The single-variant bank-adaptive chooser still split:

- the density branch was stronger on one evaluation block
- the baseline branch was stronger in several fresh-bank confirmation cells

The merged candidate set exposes the real solver opportunity:

- the four-candidate oracle is much lower than any single path
- the missing piece is a calibration-only rule for when to trust the richer
  chooser instead of the stable default

The selected gate is:

- feature: `dense_joint_entropy`
- default candidate: `dense_support`
- gate condition: `dense_joint_entropy >= 0.3655148794`

That threshold is selected from calibration only.
It is not tuned on holdout or confirmation.

## Main Result

From
[entropy_gated_bank_ensemble_solver_summary.json](outputs/entropy_gated_bank_ensemble_solver_summary.json):

- holdout solver mean `alpha` error: `0.1050`
- holdout best single cached candidate: `0.1091`
- confirmation solver mean `alpha` error: `0.1064`
- confirmation best single cached candidate: `0.1104`

So this solver clears both evaluation blocks:

- holdout improvement: about `0.0040`
- confirmation improvement: about `0.0040`

The repo now has a working focused solver on the tested slice.

## Plain-Language Read

The dense-support candidate is the safest default.
It keeps the low-observability cells from blowing up.

The extra gain comes from only trusting the richer four-way chooser when the
dense-joint entropy says the observation has enough structure to justify it.

Plainly:

- low observability: stay with `dense_support`
- higher observability: let the chooser pick among all four cached candidates

That is exactly the solver-design resolution the earlier bottleneck work was
pointing toward.

## BGP Impact

This strengthens BGP.

The result shows the remaining failure was in solver policy, not in the latent
control object:

- the control backbone stayed usable
- the solver bottleneck yielded to an observability-gated ensemble
- no larger latent object or theory rewrite was needed

## Artifacts

Data:

- [entropy_gated_bank_ensemble_solver_summary.json](outputs/entropy_gated_bank_ensemble_solver_summary.json)
- [entropy_gated_bank_ensemble_solver_trials.csv](outputs/entropy_gated_bank_ensemble_solver_trials.csv)
- [entropy_gated_bank_ensemble_solver_model.json](outputs/entropy_gated_bank_ensemble_solver_model.json)

Code:

- [run.py](run.py)
