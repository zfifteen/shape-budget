# Pose-Free Weighted Anisotropic Inverse Experiment

## Purpose

This experiment combines the two nuisance variables that were separated in the previous inverse extensions:

- unknown rotation of the observed boundary
- unknown medium anisotropy `alpha`

The forward family is still

\[
w_1 d_\alpha(x, p_1) + w_2 d_\alpha(x, p_2) + w_3 d_\alpha(x, p_3) = S
\]

with positive normalized weights and the axis-aligned anisotropic metric

\[
d_\alpha((x, y), (u, v)) = \sqrt{(x-u)^2 + \alpha^2 (y-v)^2}.
\]

The recovery target remains the joint latent object:

- the normalized source triangle relative to budget
- the normalized weight vector in the simplex
- the anisotropy parameter `alpha`

This is the first inverse test where the boundary is pose-free and the medium is hidden at the same time.

## Research Question

Does the weighted multi-source control object remain recoverable when unknown rotation and unknown medium anisotropy appear together, and does an anisotropy-aware inverse still beat a Euclidean weighted shortcut in that combined-nuisance setting?

## Scope Of This Test

This is a pose-free anisotropic inverse, but it is still not the most general warped-medium problem.

The assumptions are:

- three sources
- weighted constant-sum boundary
- translation removed from the boundary observation
- scale removed from the boundary observation
- one unknown axis-aligned anisotropy parameter `alpha`
- rotation treated as an observational pose nuisance on the radial-signature grid

So this experiment does not solve unknown anisotropy-axis orientation.

It solves the combined problem where:

- the observed signature can be rotated
- the medium strength parameter `alpha` is unknown
- geometry and weights are also unknown

## Inverse Method

The experiment script is [run_pose_free_weighted_anisotropic_inverse_experiment.py](run.py#L1).

The method stays deliberately simple:

1. build an anisotropy-aware reference bank over geometry, weights, and `alpha`
2. encode each forward boundary as a centroid-centered, mean-radius-normalized radial signature
3. randomly rotate the observed signature on the `64`-bin grid
4. recover the best bank element under cyclic-shift-aware masked L2 matching

So the inverse is not handed either the orientation or the medium parameter.

## Baseline

The baseline is a Euclidean weighted reference bank with the same geometric and weight ranges but fixed `alpha = 1`, evaluated under the same pose-free cyclic-shift matching.

This keeps the comparison clean:

- if medium anisotropy is truly part of the latent state, the anisotropy-aware bank should still beat a Euclidean weighted shortcut even when rotation is unknown

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

The combined-nuisance inverse works, but unevenly across the latent variables.

> In the pose-free boundary-only setting, the weighted three-source latent object remains operational when unknown rotation and unknown anisotropy are present together: geometry and weights stay recoverable to a useful degree, and the anisotropy-aware inverse still decisively outperforms a Euclidean weighted shortcut, but `alpha` becomes much more weakly identified.

The summary file is [pose_free_weighted_anisotropic_inverse_summary.json](outputs/pose_free_weighted_anisotropic_inverse_summary.json).

Global summary:

- anisotropy-aware bank size: `300`
- Euclidean baseline bank size: `150`
- trials per regime: `40`
- best mean geometry MAE: `7.1066e-02`
- worst mean geometry MAE: `1.0414e-01`
- best mean weight MAE: `1.3793e-01`
- worst mean weight MAE: `1.6562e-01`
- best mean alpha absolute error: `1.4557e-01`
- worst mean alpha absolute error: `3.0421e-01`
- best mean anisotropy-aware fit RMSE: `7.8338e-03`
- worst mean anisotropy-aware fit RMSE: `4.6687e-02`
- smallest mean improvement factor over the Euclidean baseline: `3.7804`
- largest mean improvement factor: `1.5925e+01`

By regime:

- `full_clean`
  - geometry MAE mean: `7.1066e-02`
  - weight MAE mean: `1.6163e-01`
  - alpha absolute error mean: `1.9084e-01`
  - anisotropy-aware fit RMSE mean: `8.0384e-03`
  - Euclidean baseline fit RMSE mean: `9.8257e-02`
  - mean improvement factor: `1.4075e+01`

- `full_noisy`
  - geometry MAE mean: `7.8980e-02`
  - weight MAE mean: `1.3793e-01`
  - alpha absolute error mean: `2.6749e-01`
  - anisotropy-aware fit RMSE mean: `7.8338e-03`
  - Euclidean baseline fit RMSE mean: `1.0664e-01`
  - mean improvement factor: `1.5925e+01`

- `partial_arc_noisy`
  - geometry MAE mean: `8.7021e-02`
  - weight MAE mean: `1.5521e-01`
  - alpha absolute error mean: `2.3414e-01`
  - anisotropy-aware fit RMSE mean: `1.5793e-02`
  - Euclidean baseline fit RMSE mean: `1.2223e-01`
  - mean improvement factor: `1.2703e+01`

- `sparse_full_noisy`
  - geometry MAE mean: `8.6516e-02`
  - weight MAE mean: `1.5850e-01`
  - alpha absolute error mean: `1.4557e-01`
  - anisotropy-aware fit RMSE mean: `1.5624e-02`
  - Euclidean baseline fit RMSE mean: `1.1404e-01`
  - mean improvement factor: `9.4438`

- `sparse_partial_high_noise`
  - geometry MAE mean: `1.0414e-01`
  - weight MAE mean: `1.6562e-01`
  - alpha absolute error mean: `3.0421e-01`
  - anisotropy-aware fit RMSE mean: `4.6687e-02`
  - Euclidean baseline fit RMSE mean: `1.1085e-01`
  - mean improvement factor: `3.7804`

These numbers say something important:

- the anisotropy-aware latent model still carries much more explanatory power than the Euclidean shortcut
- geometry recovery remains fairly stable even under the combined nuisance
- weight recovery degrades, but not catastrophically
- `alpha` recovery degrades sharply once rotation is hidden too

## Comparison To The Canonical Anisotropic Inverse

This experiment also records direct penalties relative to the earlier canonical-pose anisotropic inverse.

The comparison is in [pose_free_weighted_anisotropic_inverse_penalties.csv](outputs/pose_free_weighted_anisotropic_inverse_penalties.csv).

The high-level picture is:

- geometry MAE stays in roughly the same range, with penalty factors from `0.917` to `1.238`
- weight MAE worsens by about `1.24x` to `1.70x`
- alpha MAE worsens much more sharply, by about `4.77x` to `14.39x`
- anisotropy-aware fit RMSE is comparable to canonical in some regimes and modestly worse in the harder ones

Because both experiments use finite reference banks and nearest-neighbor recovery, penalty factors below `1` in a few geometry and fit entries should be read as finite-bank variation, not as evidence that unknown rotation makes anisotropic inversion easier.

The structural takeaway is:

- unknown rotation does not destroy the latent-object result
- it specifically damages anisotropy identifiability much more than geometry identifiability
- the combined nuisance creates a real conditioning problem for `alpha`

## Interpretation

This is still a strong result, but it is strong in a more specific way than the previous inverse extensions.

The project now has evidence that:

- geometry and participation remain operational latent variables under combined pose and medium nuisance
- medium-aware modeling still matters a great deal for fit quality
- `alpha` is not equally well-conditioned under the current signature representation

So the right reading is not “everything survived equally well.”

The right reading is:

> the latent-object framework survives the combined nuisance well enough to stay operational, but the medium parameter becomes the weak link under pose-free observation.

That is scientifically useful. It tells us where the framework is strong and where the inverse needs a better representation or better matching strategy.

One plausible mechanism is partial aliasing between cyclic pose variation and anisotropic deformation in the current radial-signature encoding.

## Scope Of The Result

This experiment does show:

- pose-free boundary-only recovery of geometry, weights, and `alpha`
- continued strong advantage for an anisotropy-aware latent model over a Euclidean weighted shortcut
- a clear conditioning map showing that `alpha` is much more fragile than geometry under the combined nuisance

This experiment does not address:

- good `alpha` identifiability under pose-free observation
- recovery when the anisotropy axis itself is unknown
- recovery under a full positive-definite metric matrix rather than a single scalar `alpha`
- recovery in non-quadratic or spatially varying media

## Figures

- [pose_free_weighted_anisotropic_inverse_heatmap.png](outputs/figures/pose_free_weighted_anisotropic_inverse_heatmap.png)
- [pose_free_weighted_anisotropic_inverse_baseline_and_penalty.png](outputs/figures/pose_free_weighted_anisotropic_inverse_baseline_and_penalty.png)
- [pose_free_weighted_anisotropic_inverse_examples.png](outputs/figures/pose_free_weighted_anisotropic_inverse_examples.png)

The clearest figures are:

- [pose_free_weighted_anisotropic_inverse_heatmap.png](outputs/figures/pose_free_weighted_anisotropic_inverse_heatmap.png) for the uneven recovery profile across geometry, weights, and `alpha`
- [pose_free_weighted_anisotropic_inverse_baseline_and_penalty.png](outputs/figures/pose_free_weighted_anisotropic_inverse_baseline_and_penalty.png) for the continued baseline gap and the rotation penalty relative to canonical anisotropic inversion

## Artifacts

Data:

- [pose_free_weighted_anisotropic_inverse_trials.csv](outputs/pose_free_weighted_anisotropic_inverse_trials.csv)
- [pose_free_weighted_anisotropic_inverse_summary.csv](outputs/pose_free_weighted_anisotropic_inverse_summary.csv)
- [pose_free_weighted_anisotropic_inverse_penalties.csv](outputs/pose_free_weighted_anisotropic_inverse_penalties.csv)
- [pose_free_weighted_anisotropic_inverse_summary.json](outputs/pose_free_weighted_anisotropic_inverse_summary.json)

Code:

- [run_pose_free_weighted_anisotropic_inverse_experiment.py](run.py#L1)

## Recommended Next Step

The cleanest next step is not a broader medium yet.

It is to improve `alpha` identifiability under the same combined nuisance.

At this point the project has shown that the latent-object result survives, but it has also revealed a bottleneck:

- geometry and weights are relatively robust
- `alpha` is much more weakly identified once pose is hidden

So the highest-leverage next experiment is to test a richer inverse representation or alignment scheme that tries to separate anisotropy from pose rather than broadening immediately to unknown anisotropy axes or richer warped media.
