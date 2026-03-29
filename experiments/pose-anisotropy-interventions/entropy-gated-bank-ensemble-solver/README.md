# Entropy-Gated Bank Ensemble Solver

## Purpose

This experiment resolves the focused solver bottleneck in the tested regime by
combining the two bank-adaptive candidate generators that remained
complementary under fresh-bank evaluation:

- `baseline` bank-adaptive candidates
- `density_ablation` bank-adaptive candidates

The solved slice is:

- `sparse_full_noisy`
- `sparse_partial_high_noise`
- moderate anisotropy
- `low_skew`, `mid_skew`, `high_skew`

The solver keeps the forward model and latent control object fixed. It changes
only the inverse policy:

1. fit a frozen four-way logistic chooser on calibration blocks only
2. default to the stable `dense_support` candidate
3. open an observability gate only when `d_joint_entropy` is high enough
4. when the gate opens, trust the frozen four-way chooser

## Why This Solver

The single-variant bank-adaptive chooser still split:

- the density branch was stronger on one evaluation block
- the baseline branch was stronger in several fresh-bank confirmation cells

The merged candidate set exposed the real solver opportunity:

- the four-candidate oracle was much lower than any single path
- the missing piece was a calibration-only rule for when to trust the richer
  chooser instead of the stable default

The selected gate is:

- feature: `d_joint_entropy`
- default candidate: `dense_support`
- gate condition: `d_joint_entropy >= 0.3655148794`

That threshold is selected from calibration only. It is not tuned on holdout or
confirmation.

## Main Result

From [entropy_gated_bank_ensemble_solver_summary.json](outputs/entropy_gated_bank_ensemble_solver_summary.json):

- holdout solver mean `alpha` error: `0.1050`
- holdout best single cached candidate: `0.1091`
- confirmation solver mean `alpha` error: `0.1064`
- confirmation best single cached candidate: `0.1104`

So this solver clears both evaluation blocks:

- holdout improvement: about `0.0040`
- confirmation improvement: about `0.0040`

The experiments show a solver-policy result on the tested slice.

The gate rule is fixed from calibration only:

- open the chooser when `d_joint_entropy >= 0.3655148794`
- otherwise return `dense_support`

The solver selects one cached candidate when the gate opens. It does not
average bank outputs.

Under the matched frozen shadow protocol, the ambiguity-gated alternative is
worse on fresh combined data:

- entropy gate fresh combined mean `alpha` error: `0.105721`
- ambiguity gate fresh combined mean `alpha` error: `0.116399`

## Plain-Language Read

The dense-support candidate is the safest default. It keeps the low-observability
cells from blowing up.

The extra gain comes from trusting the richer four-way chooser only when
`d_joint_entropy` is high enough to justify leaving that default.

Plainly:

- low observability: stay with `dense_support`
- higher observability: let the chooser pick among all four cached candidates

Per-cell results remain mixed, but the aggregate gain holds on both independent
fresh evaluation blocks.

## BGP Impact

This strengthens BGP.

The result shows the remaining failure in the focused slice was in solver policy,
not in the latent control object. This is a practical observability gate in the
tested regime, not a new control law:

- the control backbone stayed usable
- the solver bottleneck yielded to a frozen observability-gated bank policy
- no larger latent object or theory rewrite was needed

## Remaining Limits

This is a focused-slice result, not a blanket claim that the entire anisotropic
solver stack is finished.

The remaining open work is:

- broader regime coverage outside the solved slice
- broader fresh-bank validation outside the solved slice
- unknown anisotropy-axis orientation
- richer media
- outward extension to harder families

## Artifacts

Data:

- [entropy_gated_bank_ensemble_solver_summary.json](outputs/entropy_gated_bank_ensemble_solver_summary.json)
- [entropy_gated_bank_ensemble_solver_trials.csv](outputs/entropy_gated_bank_ensemble_solver_trials.csv)
- [entropy_gated_bank_ensemble_solver_model.json](outputs/entropy_gated_bank_ensemble_solver_model.json)

Code:

- [run.py](run.py)
