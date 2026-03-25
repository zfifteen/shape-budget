# Pose-Free Alpha Refinement Experiment

## Purpose

This pilot tests a narrower hypothesis about the pose-free anisotropic inverse.

The previous result showed that:

- geometry and weights remained recoverable under unknown rotation plus unknown anisotropy
- the anisotropy-aware inverse still beat a Euclidean shortcut by a large margin
- `alpha` became the weakly identified part of the latent state

One possible explanation was simple search resolution:

- maybe the latent state is still there
- maybe the bank match is landing near the right geometry and weights
- maybe `alpha` just needs a more direct local optimization step

That is what this experiment probes.

## Research Question

If we hold candidate geometry and weights fixed and directly re-optimize `alpha` and rotation shift after the initial bank match, does `alpha` recover more reliably under the same pose-free anisotropic inverse setting?

## Method

The experiment script is [run_pose_free_alpha_refinement_experiment.py](run.py#L1).

The method is:

1. run the same pose-free anisotropic inverse as before
2. take the top `2` geometry-plus-weight bank candidates
3. for each candidate, hold geometry and weights fixed
4. sweep `alpha` on a coarse grid of `17` values over `[0.60, 1.80]`
5. refine around the best coarse `alpha` on a local grid of `9` values
6. jointly choose the best `alpha` and rotation shift by masked fit

This is a pilot, not a full production rerun.

The refinement implementation explicitly includes the starting candidate's bank
`alpha` in the search grids, so the local search does not accidentally discard
its own starting point.

The test budget here is:

- `10` trials per observation regime
- `300` anisotropy-aware bank elements
- `150` Euclidean weighted baseline elements

So this artifact is about direction, not final benchmark precision.

## Main Result

The result is mixed, but informative.

> Direct alpha refinement often improves fit and improves `alpha` recovery in some regimes, but it does not rescue `alpha` uniformly. That means the solver challenge is not only coarse search resolution.

The summary file is [pose_free_alpha_refinement_summary.json](outputs/pose_free_alpha_refinement_summary.json).

Global pilot summary:

- trials per regime: `10`
- top geometry-weight candidates refined: `2`
- coarse alpha grid size: `17`
- fine alpha grid size: `9`
- baseline alpha MAE mean range: `1.0212e-01` to `2.9075e-01`
- refined alpha MAE mean range: `1.0676e-01` to `2.3135e-01`
- smallest alpha improvement factor: `0.8627`
- largest alpha improvement factor: `1.6449`
- smallest fit-improvement factor over baseline: `0.8123`
- largest fit-improvement factor over baseline: `1.5584`

By regime:

- `full_clean`
  - alpha MAE mean: `0.1875 -> 0.2150`
  - alpha improvement factor: `0.8718`
  - fit RMSE mean: `0.00701 -> 0.00450`

- `full_noisy`
  - alpha MAE mean: `0.2907 -> 0.1768`
  - alpha improvement factor: `1.6449`
  - fit RMSE mean: `0.00825 -> 0.00614`

- `partial_arc_noisy`
  - alpha MAE mean: `0.1021 -> 0.1068`
  - alpha improvement factor: `0.9565`
  - fit RMSE mean: `0.01149 -> 0.01414`

- `sparse_full_noisy`
  - alpha MAE mean: `0.1996 -> 0.2314`
  - alpha improvement factor: `0.8627`
  - fit RMSE mean: `0.01302 -> 0.01228`

- `sparse_partial_high_noise`
  - alpha MAE mean: `0.1562 -> 0.1329`
  - alpha improvement factor: `1.1751`
  - fit RMSE mean: `0.05002 -> 0.04263`

These numbers say something precise:

- direct alpha re-optimization is not useless
- it can materially help in noisy full-view settings
- it can help modestly in the hardest sparse partial setting
- but it can also make `alpha` worse in other regimes while still improving fit

## Interpretation

This is the strongest read I take from the pilot:

The alpha recovery challenge is only partly a search problem.

If the issue were mostly that the bank did not search alpha finely enough, then direct refinement should have helped almost everywhere.

Instead, what we see is:

- fit often improves
- geometry sometimes improves and sometimes worsens a bit
- `alpha` improvement is regime-dependent rather than universal

That pattern suggests partial non-identifiability rather than just coarse quantization.

In plain language:

- the inverse can often find a shape that fits the observed boundary better
- but a better-fitting shape does not always imply a more correct `alpha`

That is strong evidence that the current pose-free representation still mixes medium anisotropy with other latent effects in a way that local fit optimization alone cannot cleanly untangle.

## What This Rules Out

This pilot weakens the hypothesis that the `alpha` problem is mostly caused by:

- too-coarse alpha sampling in the bank
- failure to locally optimize the weak parameter after retrieval

Those things matter some of the time, but they are not the whole mechanism.

## What It Suggests Instead

The project now has a sharper next question:

> what representation or inference scheme can separate medium anisotropy from pose and geometry more cleanly, rather than merely fitting the observed boundary a little better?

The likely next directions are:

- a representation that makes anisotropy more directly observable under unknown rotation
- joint local refinement of geometry and `alpha`, not just `alpha` alone
- explicit uncertainty or multimodality analysis for `alpha` under pose-free observation

## Scope Of The Result

This pilot does show:

- direct local refinement of `alpha` is feasible
- local refinement can improve fit without uniformly improving `alpha`
- the combined-nuisance solver challenge is deeper than a coarse alpha grid alone

This pilot does not address:

- that alpha refinement is ineffective in principle
- that the latent-object framework fails
- that joint geometry-plus-alpha refinement would behave the same way

## Figures

- [pose_free_alpha_refinement_heatmap.png](outputs/figures/pose_free_alpha_refinement_heatmap.png)
- [pose_free_alpha_refinement_method_comparison.png](outputs/figures/pose_free_alpha_refinement_method_comparison.png)
- [pose_free_alpha_refinement_examples.png](outputs/figures/pose_free_alpha_refinement_examples.png)

The clearest figures are:

- [pose_free_alpha_refinement_heatmap.png](outputs/figures/pose_free_alpha_refinement_heatmap.png) for the regime-by-regime mixed effect on `alpha`
- [pose_free_alpha_refinement_method_comparison.png](outputs/figures/pose_free_alpha_refinement_method_comparison.png) for the contrast between better fit and non-uniform `alpha` improvement

## Artifacts

Data:

- [pose_free_alpha_refinement_trials.csv](outputs/pose_free_alpha_refinement_trials.csv)
- [pose_free_alpha_refinement_baseline_summary.csv](outputs/pose_free_alpha_refinement_baseline_summary.csv)
- [pose_free_alpha_refinement_refined_summary.csv](outputs/pose_free_alpha_refinement_refined_summary.csv)
- [pose_free_alpha_refinement_comparison.csv](outputs/pose_free_alpha_refinement_comparison.csv)
- [pose_free_alpha_refinement_summary.json](outputs/pose_free_alpha_refinement_summary.json)

Code:

- [run_pose_free_alpha_refinement_experiment.py](run.py#L1)

## Recommended Next Step

The next step should not be a larger brute-force alpha sweep.

This pilot already tells us that better fit does not automatically mean better `alpha`.

The highest-leverage next experiment is to test a representation or uncertainty analysis that can expose whether multiple distinct `(geometry, alpha, pose)` combinations are producing nearly indistinguishable observed boundaries.
