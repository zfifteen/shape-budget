# Pose-Free Weighted Inverse Experiment

## Purpose

This experiment extends the weighted three-source inverse from canonical pose to the setting where rotation is also unknown.

The forward family is still

\[
w_1\|x-p_1\| + w_2\|x-p_2\| + w_3\|x-p_3\| = S
\]

with positive normalized weights.

The recovery target is still the normalized control object:

- the normalized source triangle relative to budget
- the normalized weight vector in the simplex

What changes is that the observed boundary is no longer given in a shared orientation.

Rotation is now a nuisance variable.

## Research Question

Does the weighted multi-source control object remain recoverable when rotation is unknown too, and does a weighted inverse still beat an equal-weight shortcut in that harder setting?

## Scope Of This Pose-Free Test

This experiment is pose-free in rotation, but it is still not the hardest possible inverse.

The observations are:

- boundary-only
- centered and scale-normalized from the boundary itself
- randomly rotated on the signature grid

So:

- translation is removed
- scale is removed
- rotation is unknown and must be handled by the inverse

This is exactly the next step we wanted.

## Inverse Method

The experiment script is [run_pose_free_weighted_inverse_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_pose_free_weighted_inverse_experiment.py#L1).

The method stays deliberately simple:

1. build a weighted reference bank of forward models
2. encode each boundary as a centroid-centered, mean-radius-normalized radial signature
3. randomly rotate the observed signature
4. recover the best bank element under cyclic-shift-aware masked L2 matching

So the inverse is not handed the orientation.

It has to find the latent control object while also absorbing the unknown rotation through cyclic alignment.

That makes this a much stronger test than the canonical-pose inverse.

## Baseline

The baseline remains an equal-weight reference bank with the same geometric ranges.

This keeps the comparison clean:

- if weighted participation is really part of the latent state, the weighted bank should still outperform the equal-weight bank even when pose is unknown

## Parameter Sweep

Reference banks:

- weighted bank size: `300`
- equal-weight baseline bank size: `150`

Test set:

- `40` weighted test cases per observation regime

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

The random rotation is sampled as a uniform cyclic shift on the `64`-bin signature grid.

## Main Result

The pose-free inverse still works.

> In the rotation-unknown boundary-only setting, the weighted three-source control object remains recoverable to a useful degree, and the weighted inverse continues to outperform the equal-weight baseline across all tested regimes.

The summary file is [pose_free_weighted_inverse_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/pose_free_weighted_inverse_summary.json).

Global summary:

- reference bank size: `300`
- equal-weight baseline bank size: `150`
- trials per regime: `40`
- best mean geometry MAE: `5.5297e-02`
- worst mean geometry MAE: `1.0776e-01`
- best mean weight MAE: `1.2554e-01`
- worst mean weight MAE: `1.6937e-01`
- best mean weighted-fit RMSE: `1.1065e-03`
- worst mean weighted-fit RMSE: `1.4554e-02`
- smallest mean improvement factor over the equal-weight baseline: `1.1240`
- largest mean improvement factor: `5.4417`

By regime:

- `full_clean`
  - geometry MAE mean: `5.5297e-02`
  - weight MAE mean: `1.4503e-01`
  - weighted-fit RMSE mean: `1.1065e-03`
  - equal-weight baseline RMSE mean: `6.3471e-03`
  - mean improvement factor: `5.4417`

- `full_noisy`
  - geometry MAE mean: `5.7871e-02`
  - weight MAE mean: `1.6937e-01`
  - weighted-fit RMSE mean: `2.0846e-03`
  - equal-weight baseline RMSE mean: `4.0144e-03`
  - mean improvement factor: `2.4632`

- `partial_arc_noisy`
  - geometry MAE mean: `8.6826e-02`
  - weight MAE mean: `1.4782e-01`
  - weighted-fit RMSE mean: `4.3168e-03`
  - equal-weight baseline RMSE mean: `8.3046e-03`
  - mean improvement factor: `2.3937`

- `sparse_full_noisy`
  - geometry MAE mean: `1.0776e-01`
  - weight MAE mean: `1.2554e-01`
  - weighted-fit RMSE mean: `8.7527e-03`
  - equal-weight baseline RMSE mean: `1.0741e-02`
  - mean improvement factor: `1.3466`

- `sparse_partial_high_noise`
  - geometry MAE mean: `7.7682e-02`
  - weight MAE mean: `1.4868e-01`
  - weighted-fit RMSE mean: `1.4554e-02`
  - equal-weight baseline RMSE mean: `1.4557e-02`
  - mean improvement factor: `1.1240`

These numbers say something important:

- the latent object survives unknown rotation
- geometry recovery stays clearly usable
- weight recovery gets harder than in the canonical-pose case
- the weighted bank still carries real information that the equal-weight shortcut throws away

## Comparison To The Canonical-Pose Inverse

This experiment also records direct penalty factors relative to the earlier canonical-pose weighted inverse.

The comparison is in [pose_free_weighted_inverse_penalties.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/pose_free_weighted_inverse_penalties.csv).

The high-level picture is:

- geometry MAE stays in the same overall range
- weight MAE worsens by about `1.19x` to `1.82x`
- weighted-fit RMSE is comparable to canonical in some regimes and modestly worse in the hardest ones
- the weighted-versus-equal-weight advantage shrinks, but it does not disappear

Because both experiments use finite reference banks and nearest-neighbor recovery, penalty factors below `1` in a few easier regimes should be read as finite-bank variation, not as evidence that unknown rotation makes inversion easier.

The structural takeaway is:

- unknown rotation is a real nuisance
- it hurts weight recovery more than geometry recovery
- it does not destroy the operational status of the weighted control object

## Interpretation

This is a strong result.

The weighted multi-source inverse no longer depends on being given the right orientation.

A simple cyclic-shift-aware boundary inverse still finds the right kind of hidden state.

That means the project has now crossed another threshold:

- not just forward structure
- not just canonical-pose inversion
- but pose-robust latent recovery

The strongest reading is:

> normalized placement plus normalized participation remains an operational latent variable even when orientation is unknown, at least in the rotation-discrete pose-free setting tested here.

That is a serious strengthening of the project.

## Scope Of The Result

This experiment does show:

- rotation-unknown boundary-only recovery of the weighted control object
- consistent improvement over an equal-weight baseline across all tested regimes
- a clear nuisance-variable cost map from canonical pose to pose-free recovery

This experiment does not yet show:

- fully continuous arbitrary-angle rotation handling beyond the signature grid
- recovery when the source count exceeds three
- recovery when the medium is anisotropic or otherwise warped
- recovery of absolute scale or absolute position

## Figures

- [pose_free_weighted_inverse_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/figures/pose_free_weighted_inverse_heatmap.png)
- [pose_free_weighted_inverse_baseline_and_penalty.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/figures/pose_free_weighted_inverse_baseline_and_penalty.png)
- [pose_free_weighted_inverse_examples.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/figures/pose_free_weighted_inverse_examples.png)

The clearest figures are:

- [pose_free_weighted_inverse_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/figures/pose_free_weighted_inverse_heatmap.png) for the actual pose-free recovery error scale
- [pose_free_weighted_inverse_baseline_and_penalty.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/figures/pose_free_weighted_inverse_baseline_and_penalty.png) for the weighted-bank advantage and the rotation penalty

## Artifacts

Data:

- [pose_free_weighted_inverse_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/pose_free_weighted_inverse_trials.csv)
- [pose_free_weighted_inverse_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/pose_free_weighted_inverse_summary.csv)
- [pose_free_weighted_inverse_penalties.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/pose_free_weighted_inverse_penalties.csv)
- [pose_free_weighted_inverse_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/pose_free_weighted_inverse_outputs/pose_free_weighted_inverse_summary.json)

Code:

- [run_pose_free_weighted_inverse_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_pose_free_weighted_inverse_experiment.py#L1)

## Recommended Next Step

The cleanest next step is weighted inversion in warped media.

At this point the project has:

- weighted forward structure
- weighted canonical-pose inversion
- weighted pose-free inversion

The next meaningful question is whether the same latent-object story survives once the medium itself adds an anisotropic or warped nuisance structure on top of the unknown pose.
