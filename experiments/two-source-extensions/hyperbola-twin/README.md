# Hyperbola Twin Experiment

## Purpose

This experiment tests the cleanest nearby extension of the Shape Budget idea.

The ellipse experiment used the constant-sum rule:

\[
r_1 + r_2 = 2a
\]

This experiment flips the rule to a constant difference:

\[
|r_1 - r_2| = 2a
\]

with focal separation `2c` and `c > a`.

The question is whether the same budget logic survives on the fixed-difference side, producing a hyperbola family with its own clean control knob.

## Research Question

Does the fixed-difference two-source process produce a hyperbola family governed by a single bounded control ratio, with the same kind of normalized scale collapse seen in the ellipse case?

## Control Ratio

For the hyperbola twin, the clean bounded control ratio is not the standard hyperbolic eccentricity.

It is:

\[
\lambda = \frac{a}{c}
\]

which has two direct interpretations:

- fixed difference / focal separation
- deficit allowance / structural separation

This lives in `(0, 1)`.

Standard hyperbolic eccentricity is just its reciprocal:

\[
e_h = \frac{c}{a} = \frac{1}{\lambda}
\]

That turns out to matter.

If we normalize by `c`, not by `a`, the transverse residue law becomes:

\[
\frac{b}{c} = \sqrt{1 - \lambda^2}
\]

So the hyperbola twin has a residue relation with exactly the same algebraic shape as the ellipse case, but with the roles of `a` and `c` swapped.

## Process Construction

For the right branch, the experiment uses the fixed-difference process

\[
r_{left} = s + a,\quad r_{right} = s - a
\]

with `s >= c`.

The full hyperbola is then recovered by symmetry.

This is the difference-process twin of the constant-sum circle construction.

Because the hyperbola is open, the experiment compares branches on a shared truncated hyperbolic-parameter window rather than over a closed curve.

That truncation is explicit:

- branch window: `u <= 1.8`

## Experiment Design

The experiment script is [run_hyperbola_twin_experiment.py](run.py#L1).

It performs four tests:

1. process reconstruction
2. scale collapse under normalization by `c`
3. response curves for open-family observables
4. phase map on the `(D = 2a, d = 2c)` plane

Because the family is open, the response curves use intrinsic local or asymptotic quantities rather than finite area or perimeter:

- openness residue `b/c`
- asymptote slope `b/a`
- normalized vertex curvature `c kappa_vertex`

## Parameter Sweep

- `lambda` values: 19 values from `0.05` to `0.95`
- scales `c`: `0.75, 1.0, 1.5, 2.5, 4.0`
- process samples per branch half: 500
- branch truncation window: `u <= 1.8`

## Main Result

The extension works cleanly.

> The fixed-difference process produces a hyperbola family with a one-knob normalized collapse under `lambda = a/c`.

The summary file is [hyperbola_twin_summary.json](outputs/hyperbola_twin_summary.json).

Key numerical results:

- maximum hyperbola-equation residual: `1.0658e-13`
- maximum RMS residual: `2.5084e-14`
- maximum pairwise scale-collapse error after normalization by `c`: `3.9736e-08`
- mean pairwise scale-collapse error: `3.7655e-12`

Across fixed-`lambda` groups, the spread over scale was numerically negligible:

- `normalized_openness`: `2.2204e-16`
- `asymptote_slope`: `3.5527e-15`
- `normalized_vertex_curvature`: `1.4211e-14`

So under this process model, the normalized open family behaves as if `lambda = a/c` is sufficient.

## Interpretation

This gives the Shape Budget program a genuine twin, not just a nearby comparison.

The ellipse side says:

- surplus-style rule: fixed sum
- closed family
- control ratio `c/a`

The hyperbola side says:

- deficit-style rule: fixed difference
- open family
- control ratio `a/c`

That is a satisfying dual structure.

The strongest way to say it is:

> The constant-sum and constant-difference two-source processes form a matched pair of one-knob normalized families, with ellipse and hyperbola as their canonical Euclidean realizations.

That is a broader result set than “the ellipse case happened to work.”

## What The Control Curves Say

Three responses are especially informative:

- `b/c = sqrt(1 - lambda^2)`
  This is the normalized openness residue. It falls smoothly as the allowed difference consumes more of the focal separation.

- `b/a = sqrt(1 - lambda^2) / lambda`
  This is the asymptote slope. It is very large for small `lambda` and shrinks toward zero as `lambda -> 1`.

- `c kappa_vertex = lambda / (1 - lambda^2)`
  This is the local sharpness at the vertex. It rises strongly as `lambda -> 1`.

So the same one-knob control law survives, but the downstream observables are the open-family versions rather than the closed-family ones.

## Scope Of The Result

This experiment does show:

- exact process reconstruction of the analytic hyperbola on the shared branch window
- clean normalized collapse across scale under `lambda = a/c`
- a well-behaved phase map in the fixed-difference regime

This experiment does not address:

- that `lambda = a/c` is the only useful parameter outside Euclidean distance
- that the same control law survives anisotropy or multi-source generalization
- that hyperbola is the only meaningful deficit-style twin in more general media

## Figures

- [hyperbola_process_reconstruction.png](outputs/figures/hyperbola_process_reconstruction.png)
- [hyperbola_scale_collapse.png](outputs/figures/hyperbola_scale_collapse.png)
- [hyperbola_response_curves.png](outputs/figures/hyperbola_response_curves.png)
- [hyperbola_phase_map.png](outputs/figures/hyperbola_phase_map.png)

The clearest figures are:

- [hyperbola_scale_collapse.png](outputs/figures/hyperbola_scale_collapse.png) for the one-knob normalized family
- [hyperbola_phase_map.png](outputs/figures/hyperbola_phase_map.png) for the deficit-spending geometry on the `(D, d)` plane

## Artifacts

Data:

- [hyperbola_twin_metrics.csv](outputs/hyperbola_twin_metrics.csv)
- [hyperbola_twin_scale_collapse.csv](outputs/hyperbola_twin_scale_collapse.csv)

Code:

- [run_hyperbola_twin_experiment.py](run.py#L1)

## Recommended Next Step

The next natural extension is controlled anisotropy.

At this point the project has:

- a closed one-knob surplus family
- an open one-knob deficit family
- a first asymmetric generalization
- inverse and conditioning evidence in the symmetric case

The next clean question is whether the same low-dimensional budget logic survives when distance itself is directionally warped.
