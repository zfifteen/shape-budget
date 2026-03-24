# Weighted Anisotropic Inverse Experiment

## Purpose

This experiment extends the weighted three-source inverse into a controlled warped-medium setting.

The forward family is now

\[
w_1 d_\alpha(x, p_1) + w_2 d_\alpha(x, p_2) + w_3 d_\alpha(x, p_3) = S
\]

with positive normalized weights and the axis-aligned anisotropic metric

\[
d_\alpha((x, y), (u, v)) = \sqrt{(x-u)^2 + \alpha^2 (y-v)^2}.
\]

The recovery target is the joint latent object:

- the normalized source triangle relative to budget
- the normalized weight vector in the simplex
- the anisotropy parameter `alpha`

This is the first inverse test where the medium itself is part of the hidden state.

## Research Question

Does the weighted multi-source control object remain recoverable when the medium adds an unknown anisotropy parameter, and does an anisotropy-aware inverse beat a Euclidean weighted shortcut in that harder setting?

## Scope Of This Test

This is a controlled anisotropic inverse, not the fully general warped-medium problem.

The assumptions are:

- three sources
- weighted constant-sum boundary
- canonical source orientation retained
- translation removed from the boundary observation
- scale removed from the boundary observation
- one unknown axis-aligned anisotropy parameter `alpha`

So the inverse is not asked to solve arbitrary pose, arbitrary anisotropy orientation, or arbitrary non-quadratic media.

It is asked to recover geometry, weights, and `alpha` jointly from the boundary alone.

## Inverse Method

The experiment script is [run_weighted_anisotropic_inverse_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_weighted_anisotropic_inverse_experiment.py#L1).

The method stays deliberately simple:

1. build an anisotropy-aware reference bank over geometry, weights, and `alpha`
2. generate each forward boundary in whitened coordinates
3. unwhiten back to raw coordinates and encode the observed boundary as a centroid-centered, mean-radius-normalized radial signature
4. recover the best bank element under masked L2 matching

So the inverse is explicitly trying to recover the medium parameter together with the shape budget state.

## Baseline

The baseline is a Euclidean weighted reference bank with the same geometric and weight ranges but fixed `alpha = 1`.

This keeps the question sharp:

- if anisotropy is truly part of the latent state, an anisotropy-aware bank should beat a Euclidean weighted shortcut by a large margin

## Parameter Sweep

Reference banks:

- anisotropy-aware bank size: `300`
- Euclidean weighted baseline bank size: `150`

Test set:

- `40` weighted anisotropic test cases per observation regime

Anisotropy range:

- `alpha in [0.60, 1.80]`

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

## Main Result

The anisotropic inverse works.

> In the canonical-pose boundary-only setting, the weighted three-source control object remains jointly recoverable when the medium adds an unknown anisotropy parameter, and the anisotropy-aware inverse decisively outperforms a Euclidean weighted shortcut across all tested regimes.

The summary file is [weighted_anisotropic_inverse_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/weighted_anisotropic_inverse_summary.json).

Global summary:

- anisotropy-aware bank size: `300`
- Euclidean baseline bank size: `150`
- trials per regime: `40`
- best mean geometry MAE: `7.7290e-02`
- worst mean geometry MAE: `8.8645e-02`
- best mean weight MAE: `8.1259e-02`
- worst mean weight MAE: `1.3339e-01`
- best mean alpha absolute error: `1.8350e-02`
- worst mean alpha absolute error: `6.3789e-02`
- best mean anisotropy-aware fit RMSE: `9.9540e-03`
- worst mean anisotropy-aware fit RMSE: `3.1271e-02`
- smallest mean improvement factor over the Euclidean baseline: `7.6104`
- largest mean improvement factor: `1.4036e+01`

By regime:

- `full_clean`
  - geometry MAE mean: `7.7290e-02`
  - weight MAE mean: `1.1399e-01`
  - alpha absolute error mean: `1.8350e-02`
  - anisotropy-aware fit RMSE mean: `9.9540e-03`
  - Euclidean baseline fit RMSE mean: `1.1466e-01`
  - mean improvement factor: `1.4036e+01`

- `full_noisy`
  - geometry MAE mean: `8.6085e-02`
  - weight MAE mean: `8.1259e-02`
  - alpha absolute error mean: `1.8586e-02`
  - anisotropy-aware fit RMSE mean: `1.1077e-02`
  - Euclidean baseline fit RMSE mean: `1.1079e-01`
  - mean improvement factor: `1.3862e+01`

- `partial_arc_noisy`
  - geometry MAE mean: `8.8645e-02`
  - weight MAE mean: `1.0579e-01`
  - alpha absolute error mean: `2.5984e-02`
  - anisotropy-aware fit RMSE mean: `1.3933e-02`
  - Euclidean baseline fit RMSE mean: `1.0921e-01`
  - mean improvement factor: `1.0780e+01`

- `sparse_full_noisy`
  - geometry MAE mean: `8.7692e-02`
  - weight MAE mean: `1.0281e-01`
  - alpha absolute error mean: `2.4774e-02`
  - anisotropy-aware fit RMSE mean: `1.3720e-02`
  - Euclidean baseline fit RMSE mean: `1.0874e-01`
  - mean improvement factor: `9.7603`

- `sparse_partial_high_noise`
  - geometry MAE mean: `8.4095e-02`
  - weight MAE mean: `1.3339e-01`
  - alpha absolute error mean: `6.3789e-02`
  - anisotropy-aware fit RMSE mean: `3.1271e-02`
  - Euclidean baseline fit RMSE mean: `1.2777e-01`
  - mean improvement factor: `7.6104`

These numbers say something important:

- the latent state can absorb medium anisotropy as well as geometry and weights
- `alpha` is recoverable with useful accuracy, not just fit away implicitly
- a Euclidean weighted model is not an adequate substitute once the medium is warped
- the anisotropy-aware latent object carries much more explanatory power than the Euclidean shortcut

## Interpretation

This is a strong result.

The project now has evidence for three inverse layers:

- canonical-pose weighted inversion in Euclidean space
- pose-free weighted inversion with unknown rotation
- canonical-pose weighted inversion with unknown medium anisotropy

That means the control object has moved even further away from being a descriptive convenience and further toward being an operational latent variable.

The strongest reading is:

> normalized geometry plus normalized participation plus controlled medium anisotropy remains jointly recoverable from the boundary, at least in the canonical-pose axis-aligned setting tested here.

The most striking part is the baseline gap.

Improvement factors between about `7.6x` and `14.0x` are much larger than the earlier weighted-versus-equal-weight gaps, which is exactly what we should expect if medium mismatch is a major source of model error.

## Scope Of The Result

This experiment does show:

- boundary-only recovery of geometry, weights, and `alpha`
- strong inferential advantage for an anisotropy-aware latent model over a Euclidean weighted shortcut
- a controlled demonstration that the hidden state can include medium structure, not just placement and participation

This experiment does not yet show:

- recovery when rotation is unknown as well as anisotropy
- recovery when the anisotropy axis is itself unknown
- recovery under a full positive-definite metric matrix rather than a single scalar `alpha`
- recovery in non-quadratic or spatially varying media

## Figures

- [weighted_anisotropic_inverse_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/figures/weighted_anisotropic_inverse_heatmap.png)
- [weighted_anisotropic_inverse_baselines.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/figures/weighted_anisotropic_inverse_baselines.png)
- [weighted_anisotropic_inverse_examples.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/figures/weighted_anisotropic_inverse_examples.png)

The clearest figures are:

- [weighted_anisotropic_inverse_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/figures/weighted_anisotropic_inverse_heatmap.png) for the joint recovery scale across geometry, weights, and `alpha`
- [weighted_anisotropic_inverse_baselines.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/figures/weighted_anisotropic_inverse_baselines.png) for the large anisotropy-aware versus Euclidean baseline gap

## Artifacts

Data:

- [weighted_anisotropic_inverse_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/weighted_anisotropic_inverse_trials.csv)
- [weighted_anisotropic_inverse_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/weighted_anisotropic_inverse_summary.csv)
- [weighted_anisotropic_inverse_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_anisotropic_inverse_outputs/weighted_anisotropic_inverse_summary.json)

Code:

- [run_weighted_anisotropic_inverse_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_weighted_anisotropic_inverse_experiment.py#L1)

## Recommended Next Step

The cleanest next step is pose-free weighted anisotropic inversion.

At this point the project has:

- Euclidean weighted inversion in canonical pose
- Euclidean weighted inversion with unknown rotation
- anisotropic weighted inversion in canonical pose

The next meaningful question is whether the same latent-object story survives when unknown rotation and unknown medium anisotropy appear together in the same inverse problem.
