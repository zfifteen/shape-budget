# Asymmetry Experiment

## Purpose

This is the first robustness test after the symmetric control-knob experiment.

The original result established a careful claim:

> Under the symmetric constant-sum two-source Euclidean process, `e = c/a` behaves like a sufficient control variable for normalized geometry.

This experiment asks what happens when that symmetry is broken in the smallest possible way.

## Pilot Model

Instead of the symmetric rule

\[
r_1 + r_2 = 2a
\]

we use a weighted budget rule

\[
w r_1 + (1-w) r_2 = a
\]

with `0 < w < 1`.

Interpretation:

- `w = 0.5` recovers the symmetric ellipse experiment
- `w < 0.5` makes radius from the left source cheaper in the budget accounting
- `w > 0.5` makes radius from the right source cheaper

For `w != 0.5`, the locus is no longer an ellipse. Numerically, it is a Cartesian-oval family.

The separation parameter is still normalized as

\[
e = \frac{c}{a}
\]

but the question is now whether `e` alone is still sufficient, or whether the geometry becomes a two-parameter family `(e, w)`.

## Research Question

Does symmetry breaking destroy one-knob sufficiency while preserving a clean two-knob normalized collapse?

## Experiment Design

The experiment script is [run_asymmetry_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_asymmetry_experiment.py#L1).

It performs four tests:

1. asymmetric family gallery
2. one-knob failure versus two-knob collapse
3. response surfaces over `(e, w)`
4. error summary across the full parameter sweep

### Parameter sweep

- `e` values: 17 values from `0.10` to `0.90`
- `w` values: `0.30, 0.40, 0.50, 0.60, 0.70`
- budget scales `a`: `0.75, 1.0, 1.5, 2.5, 4.0`

## Main Result

The pilot gives a very clear answer:

- one-knob sufficiency fails once asymmetry is introduced
- a two-parameter family `(e, w)` collapses cleanly across scale

The summary file is [asymmetry_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/asymmetry_summary.json).

Key numbers:

- max two-knob scale-collapse error: `3.7196e-08`
- mean two-knob scale-collapse error: `4.6682e-09`
- minimum one-knob family distance across differing `w` at fixed `e`: `0.0200`
- maximum one-knob family distance: `1.1450`

These numbers support the interpretation that once symmetry is broken, `e` alone is no longer sufficient, but the normalized geometry is still low-dimensional.

## Interpretation

This is a strong outcome for the overall Shape Budget program.

It means:

- the original one-knob claim was not fake or trivial
- it was genuinely tied to the symmetric process model
- when symmetry is broken, the failure is structured rather than chaotic
- the geometry appears to upgrade from a one-parameter family to a two-parameter family

That is exactly the kind of result we would hope for if the “control knob” idea is tracking a real process variable rather than just a convenient rephrasing of the ellipse equation.

## Figures

- [asymmetry_family_gallery.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/figures/asymmetry_family_gallery.png)
- [asymmetry_collapse.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/figures/asymmetry_collapse.png)
- [asymmetry_response_surfaces.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/figures/asymmetry_response_surfaces.png)
- [asymmetry_error_summary.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/figures/asymmetry_error_summary.png)

The most important figure is [asymmetry_collapse.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/figures/asymmetry_collapse.png):

- left panel: fixed `e`, varying `w` gives visibly different normalized shapes
- right panel: fixed `(e, w)`, varying scale `a` gives collapse

That is the cleanest summary of the result.

## Artifacts

Data:

- [asymmetry_metrics.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/asymmetry_metrics.csv)
- [asymmetry_scale_collapse.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/asymmetry_scale_collapse.csv)
- [asymmetry_family_distances.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/asymmetry_experiment_outputs/asymmetry_family_distances.csv)

Code:

- [run_asymmetry_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_asymmetry_experiment.py#L1)

## What This Changes

Before this experiment, the best case was:

> `e` is sufficient in the symmetric ellipse case.

After this experiment, the stronger statement is:

> Shape Budget appears to be a low-dimensional allocation geometry.
> In the symmetric case, the control space collapses to one parameter.
> In the first asymmetric pilot, it expands to two parameters.

That is a much more interesting research direction than either a trivial failure or a vague “maybe it generalizes.”

## Recommended Next Step

Run Experiment 2 from the roadmap:

- identifiability of `e` in the symmetric case
- baseline comparison to show that `e` is not only elegant, but practically useful

The asymmetry pilot already suggests that the next important question is not “does the idea survive more complexity?” but “how recoverable and operational is the control variable in the case we already understand best?”
