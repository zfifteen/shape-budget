# Multi-Source Generalization Experiment

## Purpose

This experiment tests the first genuinely higher-dimensional extension of the Shape Budget idea.

The two-source case produced a one-knob family because, after normalization, the source arrangement relative to budget had only one free degree of freedom.

With three equal-weight sources, that is no longer true.

The constant-sum boundary is now

\[
\|x-p_1\| + \|x-p_2\| + \|x-p_3\| = S
\]

and the natural control object is no longer a single ratio.

It is the normalized source triangle relative to budget.

## Research Question

When three equal-weight sources share a fixed total budget, does the Shape Budget idea lift into a structured low-dimensional control space, or does the geometry become too unconstrained to organize cleanly?

## Control Object Versus Simplex Readout

This distinction matters.

The control object is:

- the source triangle relative to the budget `S`

After translation and rotation are factored out, that control object has three degrees of freedom.

One clean invariant representation is the normalized edge-length triple

\[
\left(\frac{d_{12}}{S}, \frac{d_{13}}{S}, \frac{d_{23}}{S}\right)
\]

subject to triangle inequalities and the existence of a closed constant-sum boundary.

The allocation simplex is something different.

For each boundary point `x`, define the distance-share vector

\[
w(x) = \left(\frac{\|x-p_1\|}{S}, \frac{\|x-p_2\|}{S}, \frac{\|x-p_3\|}{S}\right)
\]

Then:

- each coordinate is nonnegative
- the coordinates sum to `1`
- the boundary traces a loop inside the 2-simplex

So the simplex loop is an invariant readout of how the budget is distributed along the boundary.
It is not, by itself, the full control space.

## Why This Setup Matters

This is the first place where the Shape Budget program has to grow up.

If the idea only works when everything collapses to one scalar, it is much narrower than it first seemed.

If instead the two-source control knob lifts into a low-dimensional structured control object in the three-source case, that is a real extension.

This experiment is aimed at exactly that question.

## Experiment Design

The experiment script is [run_multisource_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_multisource_experiment.py#L1).

It performs four linked tests:

1. three-source family reconstruction by direct ray solving
2. scale collapse for fixed normalized source triangles
3. scale collapse of the induced allocation-share loop in the simplex
4. dimensionality probes for an equilateral slice and for a broader random normalized family

Unlike the earlier contour-based scratch work, the final experiment solves the boundary directly.

For each angle `theta`, it shoots a ray from the geometric median of the source triangle and finds the unique radius where the constant-sum equation is satisfied.

That gives a stable, reproducible boundary parameterization without depending on plotting heuristics.

## Parameter Sweep

Scale-collapse families:

- four representative normalized source triangles
- scales `S`: `0.75, 1.00, 1.50, 2.50, 4.00`
- boundary samples per curve: `360`

Equilateral slice:

- `rho` values: 28 values from `0.05` to `0.26`
- boundary samples per curve: `240`

Random normalized family:

- 180 random normalized source triangles
- canonical parameter ranges:
  - `rho in [0.05, 0.24]`
  - `t in [-0.8, 0.8]`
  - `h in [0.45, 1.6]`
- only configurations with a genuine closed constant-sum boundary were kept

## Main Result

The extension works, but it does not stay one-knob.

> In the equal-weight three-source constant-sum case, normalized geometry is controlled by the normalized source triangle relative to budget, and both the boundary shape and the induced allocation-share loop collapse exactly across absolute scale.

The summary file is [multisource_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/multisource_summary.json).

Key numerical results:

- maximum constant-sum equation residual: `1.7764e-15`
- mean RMS residual: `3.7716e-16`
- maximum pairwise normalized boundary-collapse error: `2.3333e-13`
- mean pairwise normalized boundary-collapse error: `5.8904e-14`
- maximum pairwise simplex-loop collapse error: `1.5454e-13`
- mean pairwise simplex-loop collapse error: `2.9071e-14`

Dimensionality results:

- equilateral slice PC1 explained variance ratio: `0.9957`
- equilateral slice first 2 PCs cumulative ratio: `0.9994`
- random normalized family PC1 explained variance ratio: `0.5396`
- random normalized family first 3 PCs cumulative ratio: `0.9901`
- random normalized family first 5 PCs cumulative ratio: `0.9993`

Those numbers say something precise:

- the three-source family is not secretly one-dimensional
- the broad normalized family is still strongly low-dimensional
- the equilateral slice behaves like a near one-parameter subfamily

That is the structured outcome we wanted.

## Interpretation

This is a meaningful extension of the Shape Budget idea.

The two-source case gave:

- one scalar control ratio
- one normalized boundary family

The three-source case gives:

- a normalized source-triangle control object
- a low-dimensional family of normalized boundaries
- an induced allocation-share loop in the simplex

So the right language here is not “the control knob survives unchanged.”

The right language is:

> the two-source control knob lifts into a low-dimensional multi-source control manifold, with the allocation simplex providing a boundary-level readout of budget sharing.

That is stronger than a loose analogy, but narrower than claiming that the simplex itself is the full parameter space.

## What This Changes

Before this experiment, the strongest higher-dimensional hope was that multi-source geometry might still be organized by some compact budget object.

After this experiment, there is direct evidence for a careful version of that story:

- scale still factors out cleanly
- normalized source placement relative to budget is the right control object
- the allocation-share loop is a stable invariant readout
- the equal-weight three-source family remains low-dimensional rather than exploding in complexity

That is exactly the sort of structure that makes the Shape Budget idea feel like a research program rather than a one-off reinterpretation.

## Scope Of The Result

This experiment does show:

- exact numerical reconstruction of the equal-weight three-source constant-sum boundary
- exact scale collapse for fixed normalized source triangles
- exact scale collapse of the induced allocation-share loop
- a near one-parameter equilateral slice
- a broader normalized family that behaves like a low-dimensional, roughly three-parameter manifold

This experiment does not yet show:

- that the simplex loop alone determines the whole geometry
- that unequal source weights preserve the same dimensional story
- that the same control structure survives four or more sources without additional complications
- that inverse recovery remains easy when the source locations are unknown

## Figures

- [multisource_family_gallery.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/figures/multisource_family_gallery.png)
- [multisource_scale_collapse.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/figures/multisource_scale_collapse.png)
- [multisource_dimension.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/figures/multisource_dimension.png)

The clearest figures are:

- [multisource_scale_collapse.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/figures/multisource_scale_collapse.png) for the exact collapse of both the normalized boundary and the simplex loop
- [multisource_dimension.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/figures/multisource_dimension.png) for the “equilateral slice is near one-parameter, random family is low-dimensional but not one-knob” story

## Artifacts

Data:

- [multisource_residuals.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/multisource_residuals.csv)
- [multisource_scale_collapse.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/multisource_scale_collapse.csv)
- [multisource_spectra.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/multisource_spectra.csv)
- [multisource_random_parameters.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/multisource_random_parameters.csv)
- [multisource_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/multisource_outputs/multisource_summary.json)

Code:

- [run_multisource_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_multisource_experiment.py#L1)

## Recommended Next Step

The cleanest next step is weighted multi-source generalization.

At this point the project has evidence for:

- a one-knob two-source symmetric family
- a two-parameter asymmetric two-source family
- a one-knob hyperbola twin
- a two-parameter anisotropic family recoverable by whitening
- a low-dimensional equal-weight three-source family

The next natural question is whether unequal source weights produce a similarly structured lift of the three-source control manifold, or whether the geometry becomes materially harder in the genuinely weighted case.
