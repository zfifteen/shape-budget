# Rotation-Invariant Spectral Experiment

## Purpose

This experiment tests a direct version of the question raised by the previous artifact:

Is the pose-free `alpha` bottleneck mostly an artifact of the current shift-search radial-signature representation?

The previous pose-free anisotropic inverse and the latent ambiguity experiment showed that:

- geometry remains fairly stable under hidden rotation
- `alpha` broadens much more than geometry
- the broadening is real in the current inverse setup

But that still left open an important possibility:

- maybe the bottleneck comes mostly from how the inverse handles pose
- maybe a cleaner pose-invariant representation would make `alpha` much easier to read

This experiment probes that possibility directly.

## Research Question

If we replace the current cyclic-shift-aware full-signature matcher with a low-order rotation-invariant spectral representation, does `alpha` become easier to recover under pose-free observation?

## Pre-Benchmark Logic Audit

Before running the benchmark, the new representation was checked in two ways.

First, the script compiled cleanly.

Second, the spectral feature itself was tested for pose invariance.

The representation uses low-order harmonic magnitudes estimated from the observed boundary. Because cyclic shifts only rotate harmonic phase, the magnitudes should stay fixed if the signal and observation mask rotate together.

The audit confirmed that:

- full-signature invariance held to `4.89e-16`
- matched masked-observation invariance held to `4.11e-09`

That matters because it means this experiment really is testing a cleaner pose-invariant encoding, not just a different shift-search implementation.

The experiment script is [run_rotation_invariant_spectral_experiment.py](run.py#L1).

## Method

The comparison is between two inverse methods on the same pose-free weighted anisotropic three-source family.

Baseline:

- the current cyclic-shift-aware full-signature matcher

Spectral method:

- estimate a low-order harmonic model from the observed bins only
- use the magnitudes of harmonics `1` through `4` as a rotation-invariant feature
- match that feature against a bank of full-boundary spectral features
- after selecting the best candidate, recover the best pose only for fit evaluation

So the spectral method removes pose by construction rather than by searching over all `64` shifts.

This is a deliberately compressed representation. It is not supposed to preserve every detail of the boundary. It is supposed to test whether a cleaner pose-invariant encoding helps `alpha`.

## Parameter Sweep

Reference bank:

- anisotropy-aware bank size: `300`

Test set:

- `40` trials per observation regime

Spectral representation:

- maximum harmonic: `4`

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

## Main Result

The result is clear.

> A cleaner pose-invariant spectral encoding does not generally rescue `alpha`. In four of the five tested regimes, `alpha` recovery gets worse, and in the hardest partial regimes the compressed spectral representation becomes strongly ill-conditioned.

The summary file is [rotation_invariant_spectral_summary.json](outputs/rotation_invariant_spectral_summary.json).

At the regime level:

- `alpha` improves only once:
  - `sparse_full_noisy`: `0.2734 -> 0.2342`
- `alpha` gets worse in the other four regimes:
  - `full_clean`: `0.2500 -> 0.3004`
  - `full_noisy`: `0.2647 -> 0.3932`
  - `partial_arc_noisy`: `0.1294 -> 0.4909`
  - `sparse_partial_high_noise`: `0.2871 -> 0.5824`

The ratio-of-means view is the cleanest summary:

- best baseline-over-spectral `alpha` ratio: `1.1676`
- worst baseline-over-spectral `alpha` ratio: `0.2636`

That means the spectral method wins only modestly in its best regime, while failing badly in its worst one.

## By Regime

- `full_clean`
  - alpha error: `0.2500 -> 0.3004`
  - alpha-span top-`10`: `0.5922 -> 0.5724`
  - geometry MAE: `0.0829 -> 0.0790`
  - fit RMSE: `0.00773 -> 0.02015`
  - spectral near-tie diverse fraction: `0.55`

- `full_noisy`
  - alpha error: `0.2647 -> 0.3932`
  - alpha-span top-`10`: `0.5275 -> 0.5759`
  - geometry MAE: `0.0843 -> 0.0858`
  - fit RMSE: `0.00858 -> 0.02081`
  - spectral near-tie diverse fraction: `0.625`

- `partial_arc_noisy`
  - alpha error: `0.1294 -> 0.4909`
  - alpha-span top-`10`: `0.5220 -> 1.1703`
  - geometry MAE: `0.0802 -> 0.1032`
  - fit RMSE: `0.01693 -> 0.12539`
  - spectral design condition mean: `6.95e3`

- `sparse_full_noisy`
  - alpha error: `0.2734 -> 0.2342`
  - alpha-span top-`10`: `0.5527 -> 0.5792`
  - geometry MAE: `0.0835 -> 0.0922`
  - fit RMSE: `0.01329 -> 0.04871`
  - spectral design condition mean: `4.59e1`

- `sparse_partial_high_noise`
  - alpha error: `0.2871 -> 0.5824`
  - alpha-span top-`10`: `0.5410 -> 1.1711`
  - geometry MAE: `0.0970 -> 0.0947`
  - fit RMSE: `0.05042 -> 0.13630`
  - spectral design condition mean: `3.66e6`

Three things stand out:

- the spectral method is truly pose-invariant, but that does not translate into broadly better `alpha`
- partial-support regimes become badly conditioned in the low-order spectral fit
- even where `alpha` improves a bit, overall boundary fit usually gets much worse

## Interpretation

This result answers the earlier question in a useful way.

The pose-free `alpha` bottleneck is not just a trivial artifact of the current shift-search representation.

If that were the main problem, then a cleaner pose-invariant encoding should have made `alpha` broadly easier to recover.

Instead, what we see is:

- the invariant encoding is mathematically sound
- it removes pose exactly in the feature itself
- but it usually makes `alpha` worse, not better

That means two things are true at once:

- the earlier bottleneck was partly representation-dependent in the sense that representation always matters
- but it was not merely a bug of using full-signature shift search

In plain language:

- simply throwing away pose by compressing the boundary into a few invariant magnitudes is too lossy
- the `alpha` problem survives that cleanup
- so the project needs a richer pose-aware or pose-equivariant representation, not just a more aggressively invariant one

The partial-regime condition numbers support that interpretation.

The spectral method becomes numerically unstable exactly where support is restricted:

- `partial_arc_noisy`: `6.95e3`
- `sparse_partial_high_noise`: `3.66e6`

So the worst failures are not random. They line up with the regimes where the low-order invariant fit is least trustworthy.

## What This Establishes

This experiment does show:

- a genuinely pose-invariant boundary encoding can be built and audited cleanly
- that encoding does not generally improve `alpha` recovery
- the earlier pose-free `alpha` bottleneck is not explained away by the current shift-search representation alone
- low-order invariant compression becomes strongly ill-conditioned under partial support

This experiment does not address:

- that no better invariant or equivariant representation exists
- that the current full-signature matcher is already optimal
- whether a hybrid method can keep the pose-cleaning benefits without throwing away too much shape information

## Figures

- [rotation_invariant_spectral_overview.png](outputs/figures/rotation_invariant_spectral_overview.png)
- [rotation_invariant_spectral_trial_scatter.png](outputs/figures/rotation_invariant_spectral_trial_scatter.png)

The clearest figure is [rotation_invariant_spectral_overview.png](outputs/figures/rotation_invariant_spectral_overview.png), because it shows the main asymmetry immediately:

- the spectral method really is different
- but that difference mostly does not help `alpha`

## Artifacts

Data:

- [rotation_invariant_spectral_trials.csv](outputs/rotation_invariant_spectral_trials.csv)
- [rotation_invariant_spectral_summary.csv](outputs/rotation_invariant_spectral_summary.csv)
- [rotation_invariant_spectral_summary.json](outputs/rotation_invariant_spectral_summary.json)

Code:

- [run_rotation_invariant_spectral_experiment.py](run.py#L1)

## Recommended Next Step

The next step should not be an even smaller invariant feature.

This experiment says that blunt pose-invariant compression loses too much.

The better next target is a hybrid representation that keeps more structure while handling pose more intelligently, for example:

- a candidate-conditioned shift-marginalized representation rather than pure invariant magnitudes
- a richer equivariant harmonic representation that keeps phase relationships instead of discarding them
- a two-stage inverse where invariant features shortlist candidates but full-signature evidence resolves `alpha`
