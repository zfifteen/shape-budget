# Latent Ambiguity Experiment

## Purpose

This experiment asks a narrower question than the previous inverse artifacts.

The pose-free weighted anisotropic inverse and the alpha-refinement pilot already showed that:

- geometry and weights remain operational under unknown rotation plus unknown anisotropy
- `alpha` is the weakly identified part of the latent state
- better local fit does not uniformly translate into better `alpha`

The next question is not whether the inverse still works at all.

It is:

> what kind of ambiguity is hiding inside the inverse once rotation is unknown, and does that ambiguity broaden mostly in `alpha`, mostly in geometry, or across the whole latent object?

This experiment targets that question directly.

## Research Question

If we compare matched canonical-pose and pose-free observations of the same true latent state, does hiding rotation broaden the near-optimal latent family, and is that broadening concentrated in `alpha` rather than in geometry or weights?

## Pre-Benchmark Logic Audit

Before running the benchmark, the script was checked in three ways.

First, the script compiled cleanly.

Second, the matched-observation construction was tested directly. The canonical and pose-free observations were built from the same true latent state and the same relative mask-plus-noise pattern on the underlying boundary, with only the pose nuisance changed.

Third, the scoring logic was checked with a forced-shift identity test. When the pose-free scorer is forced to use the true observation shift rather than minimizing over all cyclic shifts, its per-candidate scores agree with the canonical scorer down to floating-point noise:

- forced-shift max score difference: `8.33e-17`

That matters because it means the benchmark is comparing matched inference conditions rather than silently mixing in a different observation pattern.

The experiment script is [run_latent_ambiguity_experiment.py](run.py#L1).

## Method

The setup uses the same anisotropic weighted three-source forward family as the recent inverse artifacts.

For each trial:

1. sample one true latent state
2. generate one clean radial-signature boundary
3. create a matched pair of observations from that same boundary:
   - one canonical observation
   - one pose-free observation with unknown cyclic rotation
4. use the same relative observation regime on both views
5. score the same anisotropy-aware reference bank against both observations
6. measure not only the best candidate, but the top-`10` near-optimal candidate envelope

The ambiguity metrics are:

- best-candidate `alpha` error
- top-`10` `alpha` span
- top-`10` geometry dispersion
- top-`10` weight dispersion
- score-gap width of the top-`10` envelope
- a near-tie-and-alpha-diverse flag for cases where the top-`10` family stays both close in score and broad in `alpha`

This is still a bank-based ambiguity profile, not a continuous posterior.

## Parameter Sweep

Reference bank:

- anisotropy-aware bank size: `300`

Test set:

- `40` matched trials per observation regime

Envelope:

- top-`10` candidates per trial

Near-tie rule:

- a trial is marked near-tie-and-alpha-diverse when the top-`10` score spread is at most `max(noise_sigma^2, 5e-5)` and the top-`10` `alpha` span is at least `0.20`

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

## Main Result

The result is sharp.

> Hiding rotation does not broaden the near-optimal latent family uniformly. It broadens the `alpha` envelope dramatically while leaving geometry dispersion nearly unchanged and weight dispersion only modestly larger.

The summary file is [latent_ambiguity_summary.json](outputs/latent_ambiguity_summary.json).

Global summary:

- bank size: `300`
- trials per regime: `40`
- envelope size: `10`
- smallest mean pose-over-canonical `alpha` span ratio: `3.7419`
- largest mean pose-over-canonical `alpha` span ratio: `5.5697`
- smallest mean pose-over-canonical best-`alpha` error ratio: `11.3755`
- largest mean pose-over-canonical best-`alpha` error ratio: `30.9833`
- largest increase in near-tie-and-alpha-diverse fraction: `0.725`

These ratios are much larger than the corresponding geometry and weight broadening factors:

- geometry-dispersion ratio range: `0.9286` to `1.0054`
- weight-dispersion ratio range: `1.0891` to `1.1631`

That is the core result.

## By Regime

- `full_clean`
  - mean top-`10` `alpha` span: `0.1231 -> 0.5922`
  - mean best-candidate `alpha` error: `0.0158 -> 0.2500`
  - mean top-`10` geometry dispersion: `0.0711 -> 0.0681`
  - mean top-`10` weight dispersion: `0.1472 -> 0.1650`
  - near-tie-and-alpha-diverse fraction: `0.000 -> 0.000`

- `full_noisy`
  - mean top-`10` `alpha` span: `0.1327 -> 0.5275`
  - mean best-candidate `alpha` error: `0.0199 -> 0.2647`
  - mean top-`10` geometry dispersion: `0.0738 -> 0.0676`
  - mean top-`10` weight dispersion: `0.1556 -> 0.1723`
  - near-tie-and-alpha-diverse fraction: `0.000 -> 0.000`

- `partial_arc_noisy`
  - mean top-`10` `alpha` span: `0.1572 -> 0.5220`
  - mean best-candidate `alpha` error: `0.0316 -> 0.1294`
  - mean top-`10` geometry dispersion: `0.0724 -> 0.0670`
  - mean top-`10` weight dispersion: `0.1580 -> 0.1715`
  - near-tie-and-alpha-diverse fraction: `0.000 -> 0.500`

- `sparse_full_noisy`
  - mean top-`10` `alpha` span: `0.1323 -> 0.5527`
  - mean best-candidate `alpha` error: `0.0249 -> 0.2734`
  - mean top-`10` geometry dispersion: `0.0746 -> 0.0685`
  - mean top-`10` weight dispersion: `0.1527 -> 0.1668`
  - near-tie-and-alpha-diverse fraction: `0.000 -> 0.625`

- `sparse_partial_high_noise`
  - mean top-`10` `alpha` span: `0.1407 -> 0.5410`
  - mean best-candidate `alpha` error: `0.0384 -> 0.2871`
  - mean top-`10` geometry dispersion: `0.0753 -> 0.0737`
  - mean top-`10` weight dispersion: `0.1544 -> 0.1641`
  - near-tie-and-alpha-diverse fraction: `0.125 -> 0.850`

Two features stand out:

- the `alpha` envelope expands by about fourfold to sixfold in every regime
- the hardest partial and sparse regimes frequently produce genuinely near-tied candidate families that differ materially in `alpha`

## Interpretation

This result sharpens the earlier pose-free anisotropic findings.

The key issue is not just that `alpha` gets noisier.

It is that hidden rotation creates a materially broader family of near-optimal latent explanations, and that broadening concentrates much more in `alpha` than in geometry.

That is why local alpha refinement helped fit but did not rescue `alpha` uniformly:

- the inverse was often navigating a real ambiguity envelope
- not just a slightly too-coarse alpha grid

In plain language:

- the shape is still telling us a lot about normalized geometry
- it is telling us somewhat less about weights
- but once rotation is hidden, it can support several materially different anisotropy values that all score nearly as well

That is a more precise diagnosis of the bottleneck than the earlier inverse artifacts alone.

## What This Establishes

This experiment does show:

- matched evidence that the pose nuisance broadens the latent candidate family
- that the broadening is strongly concentrated in `alpha`
- that geometry dispersion stays comparatively stable under the same nuisance
- that the hardest regimes often contain near-tied but `alpha`-diverse candidate families

This experiment does not address:

- whether that `alpha` ambiguity is fundamentally irreducible
- whether a better representation or alignment scheme can collapse the `alpha` envelope again
- how much of the remaining ambiguity comes from pose-anisotropy aliasing versus finite-bank discretization

## Figures

- [latent_ambiguity_overview.png](outputs/figures/latent_ambiguity_overview.png)
- [latent_ambiguity_spectra.png](outputs/figures/latent_ambiguity_spectra.png)

The clearest figure is [latent_ambiguity_overview.png](outputs/figures/latent_ambiguity_overview.png), because it shows the asymmetry directly:

- `alpha` broadens a lot
- geometry barely broadens
- near-tie diverse families emerge mainly under pose-free partial and sparse observation

## Artifacts

Data:

- [latent_ambiguity_trials.csv](outputs/latent_ambiguity_trials.csv)
- [latent_ambiguity_summary.csv](outputs/latent_ambiguity_summary.csv)
- [latent_ambiguity_summary.json](outputs/latent_ambiguity_summary.json)

Code:

- [run_latent_ambiguity_experiment.py](run.py#L1)

## Recommended Next Step

The highest-leverage next step is not a larger alpha sweep.

This experiment already shows that the pose-free anisotropic bottleneck is an ambiguity-structure problem.

The next experiment should try to reduce that ambiguity structure directly, for example by:

- a pose-alignment representation that separates cyclic orientation from anisotropic deformation more cleanly
- an uncertainty or multimodality map over `(geometry, weights, alpha, pose)`
- a representation that exposes anisotropy with rotationally stable features rather than relying only on the current radial-signature encoding
