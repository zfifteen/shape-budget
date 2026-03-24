# Controlled Anisotropy Experiment

## Purpose

This experiment asks whether the Shape Budget control-knob result survives a simple directional warp of distance.

Instead of standard Euclidean distance, it uses the axis-aligned quadratic metric

\[
d_{\alpha}\big((x, y), (u, v)\big) = \sqrt{(x-u)^2 + \alpha^2 (y-v)^2}
\]

with the two sources still placed on the x-axis.

This is a controlled anisotropy test, not the fully general warped-medium case.

## Research Question

When distance is directionally stretched by one anisotropy parameter `alpha`, does the Shape Budget family stay low-dimensional, and can the original one-knob collapse be recovered by whitening?

## Why This Setup Matters

The key feature of this metric is that it is exactly whitenable:

\[
(x, y) \mapsto (x, \alpha y)
\]

Under that transform, the anisotropic constant-sum locus becomes the standard Euclidean ellipse again.

So this experiment is testing a precise claim:

- in raw Euclidean coordinates, the family should depend on both `e` and `alpha`
- after whitening, the family should collapse back to the original one-knob `e` result

That is exactly the kind of controlled universality result we want before moving to harder media.

## Experiment Design

The experiment script is [run_anisotropy_experiment.py](run.py#L1).

It performs four linked tests:

1. anisotropic process reconstruction
2. raw one-knob failure versus whitening recovery
3. response curves over `e` for multiple `alpha`
4. two-parameter raw parameter map in `(e, alpha)`

## Parameter Sweep

- `e` values: 17 values from `0.10` to `0.90`
- `alpha` values: `0.50, 0.75, 1.00, 1.50, 2.00`
- scales `a`: `0.75, 1.0, 1.5, 2.5, 4.0`

## Main Result

The result is clean and exactly interpretable:

> Controlled anisotropy does not destroy the Shape Budget program. It promotes raw geometry from a one-parameter family to a two-parameter family `(e, alpha)`, and whitening restores the original one-knob collapse.

The summary file is [anisotropy_summary.json](outputs/anisotropy_summary.json).

Key numerical results:

- maximum anisotropic-equation residual: `6.2172e-15`
- maximum RMS residual: `1.2391e-15`
- maximum raw scale-collapse error at fixed `(e, alpha)`: `7.9473e-08`
- mean raw scale-collapse error: `1.9925e-11`
- minimum raw family distance at fixed `e` across varying `alpha`: `5.6938e-02`
- maximum raw family distance: `1.4925`
- maximum whitened collapse error across varying `alpha` and scale at fixed `e`: `3.9736e-08`
- mean whitened collapse error: `1.5095e-11`

Those numbers say exactly what the theory predicts:

- raw one-knob sufficiency fails when anisotropy varies
- raw geometry remains low-dimensional and stable under `(e, alpha)`
- whitening recovers the original one-knob family to numerical precision

## Interpretation

This is a strong controlled extension.

It means:

- `e` is still the right control variable once distance is expressed in the whitened coordinates
- the extra anisotropy parameter is a geometric warp descriptor, not uncontrolled variation
- the Shape Budget principle survives a simple directional distortion in a structured way

The raw geometry is no longer organized by `e` alone.

For example:

- raw vertical residue becomes `sqrt(1 - e^2) / alpha`
- raw major-tip response becomes `alpha^2 / (1 - e^2)`

So in raw Euclidean coordinates, anisotropy multiplies or divides the geometric consequences of `e`.

After whitening, those extra `alpha` factors disappear and the original `e`-only collapse returns.

That is a meaningful universality result.

## What This Changes

Before this experiment, the strongest extension result was:

- asymmetry upgrades the family from one parameter to two
- the hyperbola twin gives a second one-knob family

After this experiment, there is now a controlled warped-distance result:

- one extra directional descriptor is enough in raw space
- whitening restores the original one-knob Euclidean result exactly

That is much better than either of the two simpler outcomes:

- “anisotropy destroys the idea”
- “anisotropy changes nothing”

## Scope Of The Result

This experiment does show:

- exact process reconstruction under one quadratic anisotropic metric
- clean raw organization by `(e, alpha)`
- exact recovery of the `e`-only family after whitening

This experiment does not address:

- that arbitrary anisotropic media behave the same way
- that non-quadratic directional costs reduce to one extra descriptor
- that the same recovery result survives when the sources are not aligned with the principal anisotropy axis

So this is a controlled universality test, not the final word on warped spaces.

## Figures

- [anisotropy_process_reconstruction.png](outputs/figures/anisotropy_process_reconstruction.png)
- [anisotropy_whitening_recovery.png](outputs/figures/anisotropy_whitening_recovery.png)
- [anisotropy_response_curves.png](outputs/figures/anisotropy_response_curves.png)
- [anisotropy_parameter_map.png](outputs/figures/anisotropy_parameter_map.png)

The clearest figures are:

- [anisotropy_whitening_recovery.png](outputs/figures/anisotropy_whitening_recovery.png) for the raw-failure / whitening-recovery result
- [anisotropy_response_curves.png](outputs/figures/anisotropy_response_curves.png) for the exact way `alpha` enters the raw observables

## Artifacts

Data:

- [anisotropy_metrics.csv](outputs/anisotropy_metrics.csv)
- [anisotropy_scale_collapse.csv](outputs/anisotropy_scale_collapse.csv)
- [anisotropy_raw_family_distances.csv](outputs/anisotropy_raw_family_distances.csv)
- [anisotropy_whitened_collapse.csv](outputs/anisotropy_whitened_collapse.csv)

Code:

- [run_anisotropy_experiment.py](run.py#L1)

## Recommended Next Step

The next natural step is multi-source generalization.

At this point the project has:

- a symmetric one-knob closed family
- a symmetric one-knob open twin
- a structured asymmetry result
- a structured anisotropy result

The next clean question is whether the two-source control-knob idea lifts into a low-dimensional allocation simplex once three or more centers share the budget.
