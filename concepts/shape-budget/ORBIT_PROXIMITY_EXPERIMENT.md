# Orbit Proximity Experiment

## Purpose

This experiment is the first direct test of the symmetry-orbit mechanism proposed for the pose-free anisotropic bottleneck.

The working hypothesis was:

- in the current radial-signature representation, `alpha` sits closer to the rotation orbit than normalized geometry does
- if so, small `alpha` perturbations should be more easily absorbed by cyclic rotation than matched geometry perturbations are

This note tests that claim directly in the same signature space used by the inverse artifacts.

## Research Question

Are `alpha` perturbations more rotation-orbit-close than geometry perturbations in the current centroid-centered, mean-radius-normalized radial-signature representation?

## Pre-Benchmark Logic Audit

Before the benchmark, the script was checked in two ways.

First, it compiled cleanly.

Second, the orbit metric itself was audited before any sweep was run.

The audit confirmed that:

- pure cyclic shifts lie exactly on the rotation orbit, with max shift-identity RMSE `0.0`
- the orbit-minimized distance never exceeded the raw distance, with max violation `0.0`
- the absorption statistic stayed in the expected range over random perturbation tests

The audit summary was:

- random audit cases: `40`
- max shift-identity RMSE: `0.0`
- max orbit-minus-raw violation: `0.0`
- min absorption fraction: `0.0`
- max absorption fraction in the audit sample: `4.53e-3`

So the metric behaved exactly as intended.

The experiment script is [run_orbit_proximity_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_orbit_proximity_experiment.py#L1).

## Method

This is a representation-level diagnostic, not a noisy inverse benchmark.

For each sampled latent state:

1. generate the clean anisotropic boundary signature
2. apply matched `+/-` perturbations of several types
3. measure:
   - raw signature RMSE
   - best-rotation-orbit RMSE
   - orbit absorption fraction `1 - orbit/raw`
   - whether the best shift is nonzero

Perturbation types:

- `alpha`
- `geometry_random`
- `rho`
- `t`
- `h`

The `geometry_random` perturbation is the main comparator.

It uses a random unit direction in bound-normalized geometry space `(rho, t, h)` at the same normalized step fraction as the `alpha` perturbation.

Weights are held fixed throughout so the experiment isolates geometry-versus-anisotropy behavior.

## Parameter Sweep

Base states:

- `96`

Step fractions:

- `0.03`
- `0.06`
- `0.10`
- `0.14`

Per-step perturbations:

- all five perturbation types
- both signs

This produces `192` perturbation instances per type and step.

## Main Result

The result supports the careful version of the symmetry-orbit hypothesis.

> In clean full-signature space, `alpha` perturbations are systematically more rotation-absorbable than matched random geometry perturbations, especially at medium and larger step sizes.

The summary file is [orbit_proximity_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/orbit_proximity_summary.json).

The clearest comparisons are against `geometry_random`.

Mean orbit absorption:

- step `0.03`: `alpha 0.0011`, geometry random `0.0000`
- step `0.06`: `alpha 0.0118`, geometry random `0.0037`
- step `0.10`: `alpha 0.0330`, geometry random `0.0031`
- step `0.14`: `alpha 0.0615`, geometry random `0.0059`

Nontrivial best-shift fraction:

- step `0.03`: `alpha 0.0260`, geometry random `0.0000`
- step `0.06`: `alpha 0.0677`, geometry random `0.0052`
- step `0.10`: `alpha 0.1563`, geometry random `0.0156`
- step `0.14`: `alpha 0.2292`, geometry random `0.0208`

So the same pattern shows up in two different ways:

- `alpha` loses more of its raw effect under orbit minimization
- `alpha` much more often prefers a nonzero compensating rotation

That is real support for the orbit-proximity story.

## Important Nuance

The effect is real, but it is not the strongest possible version of the hypothesis.

Two things are true at once:

- `alpha` is more orbit-close than matched random geometry
- the absolute mean absorption is still moderate in the clean full-signature setting

At the largest tested step:

- `alpha` mean absorption is `0.0615`
- geometry-random mean absorption is `0.0059`

So `alpha` is about ten times more absorbed than matched random geometry there, but it is not becoming indistinguishable from a pure rotation.

That matters for interpretation.

This experiment supports:

- `alpha` is more vulnerable to rotation aliasing in this representation

This experiment does not support:

- the strongest claim that full clean boundary information by itself makes `alpha` fundamentally unreadable

The cleaner reading is:

- orbit proximity is one real contributor to the pose-free anisotropic bottleneck
- but by itself it probably does not explain the full `11x` to `31x` inverse penalty seen earlier

## By Perturbation Type

The scalar geometry directions help sharpen the picture.

- `rho` is effectively not orbit-absorbed at any tested step
- `t` shows some orbit absorption at larger steps
- `h` shows some orbit absorption only at the larger steps
- `alpha` is the most consistently orbit-absorbed direction overall

So geometry is not uniformly “far from the orbit” in every coordinate direction.

But the main comparison still holds:

- the anisotropy direction is the most rotation-sensitive one in the tested family

## Interpretation

This result strengthens the BGP program in a useful way.

It says the pose-free anisotropic bottleneck is not just an arbitrary inverse failure.

There is real structure behind it.

In plain language:

- the hidden budget-governed geometry is still comparatively stable under rotation
- the medium-anisotropy variable is more easily blended with pose in the current encoding
- that makes `alpha` harder to read once orientation is unknown

At the same time, the result also narrows the story.

It suggests the right claim is not:

- `alpha` is doomed because it sits almost on the orbit

It is:

- `alpha` is measurably closer to the orbit than geometry is, and that makes it especially susceptible once noise, masking, and bank discretization are added

That is a stronger and more accurate mechanism.

## What This Establishes

This experiment does show:

- direct clean-signature evidence that `alpha` is more rotation-aliased than matched random geometry
- that the effect grows with perturbation size
- that the same asymmetry appears both in orbit absorption and in nontrivial best-shift frequency
- that `rho` is especially stable, while `t` and `h` show smaller but nonzero orbit effects

This experiment does not yet show:

- that orbit proximity alone explains the full pose-free inverse penalty
- how much practical symmetry-breaking alignment can recover
- whether the same effect persists under partial support and noise after alignment

## Figures

- [orbit_proximity_absorption_curves.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/figures/orbit_proximity_absorption_curves.png)
- [orbit_proximity_distance_scatter.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/figures/orbit_proximity_distance_scatter.png)
- [orbit_proximity_shift_fraction.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/figures/orbit_proximity_shift_fraction.png)

The most important figure is [orbit_proximity_absorption_curves.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/figures/orbit_proximity_absorption_curves.png), because it shows the core mechanism directly:

- `alpha` stays above the geometry curves across the sweep
- the gap widens at medium and larger perturbation sizes

## Artifacts

Data:

- [orbit_proximity_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/orbit_proximity_summary.json)
- [orbit_proximity_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/orbit_proximity_summary.csv)
- [orbit_proximity_rows.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/orbit_proximity_outputs/orbit_proximity_rows.csv)

Code:

- [run_orbit_proximity_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_orbit_proximity_experiment.py#L1)
