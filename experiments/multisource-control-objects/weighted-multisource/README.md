# Weighted Multi-Source Experiment

## Purpose

This experiment extends the equal-weight three-source Shape Budget result into the genuinely weighted case.

The equal-weight three-source boundary was

\[
\|x-p_1\| + \|x-p_2\| + \|x-p_3\| = S
\]

This experiment replaces it with the weighted rule

\[
w_1\|x-p_1\| + w_2\|x-p_2\| + w_3\|x-p_3\| = S
\]

with positive weights normalized so that

\[
w_1 + w_2 + w_3 = 1.
\]

The question is whether the equal-weight three-source control result survives in a structured way once the sources are allowed to matter unequally.

## Research Question

When three sources have unequal positive weights, does normalized geometry still collapse cleanly under a compact control object, or do the weights make the family materially harder and less organized?

## Control Object And Readout

In the equal-weight three-source experiment, the control object was the normalized source triangle relative to budget.

In the weighted case, that is no longer enough.

The natural control object becomes:

- the normalized source triangle relative to budget
- the weight vector `(w1, w2, w3)` in the simplex

That means:

- 3 geometric degrees of freedom from the normalized source triangle
- 2 additional degrees of freedom from the normalized positive weight vector

So the weighted three-source family should behave like a roughly five-parameter normalized family, not a one-knob family and not an uncontrolled mess.

The corresponding boundary-level readout is the weighted allocation-share vector

\[
u(x) = \left(\frac{w_1\|x-p_1\|}{S}, \frac{w_2\|x-p_2\|}{S}, \frac{w_3\|x-p_3\|}{S}\right)
\]

which lies in the 2-simplex because the coordinates are nonnegative and sum to `1`.

That weighted share loop is not the full control space, but it is a natural invariant readout of how the total budget is distributed along the boundary.

## Why This Matters

This is the cleanest way to test whether the multi-source Shape Budget idea is robust to unequal participation.

If weights simply destroyed the structure, then the equal-weight three-source result would be much less meaningful.

If instead weights add a small number of new control dimensions while scale still factors out exactly, that is a strong sign that the Shape Budget program is capturing real process structure rather than just a fragile special case.

## Experiment Design

The experiment script is [run_weighted_multisource_experiment.py](run.py#L1).

It performs four linked tests:

1. weighted three-source reconstruction by direct ray solving
2. scale collapse at fixed normalized geometry and fixed weights
3. fixed-geometry family divergence when weights vary
4. dimensionality probes for an equilateral weighted slice and for a broader random weighted family

As in the equal-weight three-source experiment, the final implementation solves the boundary directly.

For each angle `theta`, it shoots a ray from the weighted geometric median and finds the unique radius where the weighted constant-sum equation is satisfied.

That gives a stable parameterization without depending on contour extraction heuristics.

## Parameter Sweep

Scale-collapse test:

- four representative normalized source triangles
- fixed weights `(0.20, 0.35, 0.45)`
- scales `S`: `0.75, 1.00, 1.50, 2.50, 4.00`
- boundary samples per curve: `360`

Fixed-geometry weight-variation test:

- one representative normalized source triangle
- four weight vectors:
  - `(0.33, 0.33, 0.33)`
  - `(0.20, 0.35, 0.45)`
  - `(0.10, 0.20, 0.70)`
  - `(0.55, 0.25, 0.20)`

Equilateral weighted slice:

- fixed equilateral geometry with `rho = 0.16`
- weight-simplex grid with all weights between `0.15` and `0.70`

Random weighted family:

- 180 random normalized source triangles
- random positive normalized weights from a symmetric Dirichlet distribution
- only configurations with a genuine closed weighted constant-sum boundary were kept

## Main Result

The weighted extension works cleanly.

> In the weighted three-source case, normalized geometry is controlled by the normalized source triangle plus the weight simplex, and both the boundary shape and the weighted allocation-share loop collapse exactly across absolute scale when those controls are fixed.

The summary file is [weighted_multisource_summary.json](outputs/weighted_multisource_summary.json).

Key numerical results:

- maximum weighted constant-sum residual: `1.7764e-15`
- mean RMS residual: `4.0434e-16`
- maximum pairwise normalized boundary-collapse error: `2.4476e-12`
- mean pairwise normalized boundary-collapse error: `5.4252e-13`
- maximum pairwise weighted-simplex-loop collapse error: `2.0458e-13`
- mean pairwise weighted-simplex-loop collapse error: `3.1878e-14`

Fixed-geometry variation results:

- minimum mean boundary-family distance across differing weights: `1.9482e-02`
- maximum mean boundary-family distance across differing weights: `1.0095e-01`
- minimum mean weighted-simplex-loop family distance across differing weights: `1.7685e-01`
- maximum mean weighted-simplex-loop family distance across differing weights: `6.6895e-01`

Dimensionality results:

- equilateral weighted slice PC1 explained variance ratio: `0.4994`
- equilateral weighted slice first 2 PCs cumulative ratio: `0.9988`
- equilateral weighted slice first 3 PCs cumulative ratio: `0.9993`
- random weighted family PC1 explained variance ratio: `0.6924`
- random weighted family first 3 PCs cumulative ratio: `0.9975`
- random weighted family first 5 PCs cumulative ratio: `0.99996`

Those numbers say something precise:

- fixing geometry without fixing weights is not enough
- fixing geometry plus weights is enough for exact scale collapse
- the equilateral weighted slice forms a near two-parameter family
- the broader weighted family forms a low-dimensional roughly five-parameter family

That is the strongest structured outcome available in this design.

## Interpretation

This is not a failure of the Shape Budget program.

It is a successful lift.

The equal-weight three-source case already showed that the one-knob result grows into a higher-dimensional control object.

This experiment shows that unequal weights add exactly the kind of extra structure you would expect:

- they do not destroy scale collapse
- they do not leave geometry unchanged
- they do not make the family explode into something unstructured

Instead, they add a compact two-dimensional weight component to the existing multi-source control object.

The cleanest way to say it is:

> weighted multi-source Shape Budget geometry is organized by normalized placement plus normalized participation.

That is a stronger and more useful statement than “weights matter,” because it says how they matter.

## What This Changes

Before this experiment, the multi-source program had a good equal-weight result but no evidence yet that unequal source importance would remain tractable.

After this experiment, there is direct evidence that:

- scale still factors out exactly
- weights add independent control structure
- the weighted share loop is a stable boundary-level invariant
- dimensionality grows in a controlled way that matches the count of added degrees of freedom

That makes the broader Shape Budget result set much more mature.

## Scope Of The Result

This experiment does show:

- exact numerical reconstruction of the weighted three-source boundary
- exact scale collapse for fixed normalized geometry and fixed weights
- exact scale collapse of the weighted share loop
- clear failure of equal-weight sufficiency when weights vary at fixed geometry
- a near two-parameter equilateral weighted slice
- a broader weighted family that forms a low-dimensional roughly five-parameter manifold

This experiment does not address:

- that four or more weighted sources preserve the same compact control logic
- that the weighted inverse problem remains easy when both positions and weights are unknown
- that arbitrary non-Euclidean or anisotropic multi-source systems reduce to the same control object

## Figures

- [weighted_multisource_scale_collapse.png](outputs/figures/weighted_multisource_scale_collapse.png)
- [weighted_multisource_weight_variation.png](outputs/figures/weighted_multisource_weight_variation.png)
- [weighted_multisource_dimension.png](outputs/figures/weighted_multisource_dimension.png)

The clearest figures are:

- [weighted_multisource_weight_variation.png](outputs/figures/weighted_multisource_weight_variation.png) for the exact point that geometry is not controlled by placement alone once weights vary
- [weighted_multisource_dimension.png](outputs/figures/weighted_multisource_dimension.png) for the “near two-parameter slice inside a roughly five-parameter family” result

## Artifacts

Data:

- [weighted_multisource_residuals.csv](outputs/weighted_multisource_residuals.csv)
- [weighted_multisource_scale_collapse.csv](outputs/weighted_multisource_scale_collapse.csv)
- [weighted_multisource_weight_variation.csv](outputs/weighted_multisource_weight_variation.csv)
- [weighted_multisource_spectra.csv](outputs/weighted_multisource_spectra.csv)
- [weighted_multisource_random_parameters.csv](outputs/weighted_multisource_random_parameters.csv)
- [weighted_multisource_summary.json](outputs/weighted_multisource_summary.json)

Code:

- [run_weighted_multisource_experiment.py](run.py#L1)

## Recommended Next Step

The cleanest next step is the inverse weighted multi-source problem.

At this point the project has strong forward evidence for:

- two-source symmetric control
- asymmetric and anisotropic lifts
- equal-weight multi-source control
- weighted multi-source control

The next question is whether the same compact control objects are recoverable when the source positions and, in the weighted case, the source weights are not given in advance.
