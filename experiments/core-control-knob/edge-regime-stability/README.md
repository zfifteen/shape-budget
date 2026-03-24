# Edge-Regime Stability

## Purpose

This experiment asks how stable the control-knob consequences are near the two extreme regimes:

- `e -> 0`, where the normalized family approaches a circle
- `e -> 1`, where the family approaches degeneration

The core question is not just “what do the observables equal there?” It is:

> how much does a small change in `e` actually matter near each edge, and how much uncertainty in `e` would be induced by small measurement error in an observable?

## Research Question

How well-conditioned are the main normalized Shape Budget observables near the circular edge and the degenerate edge?

## Observables

The experiment tracks three distinct normalized observables:

- width residue `b/a = sqrt(1 - e^2)`
- exact normalized perimeter `P / (2 pi a) = (2 / pi) E(e)` with `E` the complete elliptic integral of the second kind with modulus `e`
- normalized major-tip response `a kappa_major = 1 / (1 - e^2)`

The normalized minor-tip response is not plotted separately because under the current normalization it exactly matches width residue:

\[
a \kappa_{minor} = \frac{b}{a}
\]

## Stability Metrics

The experiment uses two complementary notions of stability:

1. forward sensitivity

\[
\kappa(e) = \left| \frac{e}{q(e)} \frac{dq}{de} \right|
\]

This is a relative condition number. It measures how strongly a relative change in `e` is amplified into a relative change in the observable.

2. inverse recoverability

For a fixed 1 percent relative measurement error in an observable, the experiment estimates the implied local uncertainty in `e`:

\[
\delta e \approx 0.01 \left| \frac{q(e)}{dq/de} \right|
\]

This makes the result directly operational.

## Experiment Design

The experiment script is [run_edge_regime_stability_experiment.py](run.py#L1).

It uses exact formulas and exact derivatives for all three observables, then evaluates them on a dense grid concentrated near both edges.

The main output is not a single score. It is a conditioning map.

## Main Result

The result has a very clean shape:

> The circular edge is the poorly conditioned end for shape-only inference, while the degenerate edge splits observables into “very sensitive” and “moderately sensitive” families.

That is a stronger and more useful statement than a generic “things get unstable near the edge.”

## Key Findings

### 1. Near `e = 0`, all tested observables are first-order flat

Numerically and analytically, the circular end is locally insensitive.

The leading expansions are:

\[
\frac{b}{a} = 1 - \frac{e^2}{2} + O(e^4)
\]

\[
\frac{P}{2 \pi a} = 1 - \frac{e^2}{4} + O(e^4)
\]

\[
a \kappa_{major} = 1 + e^2 + O(e^4)
\]

So every one of these observables changes only at second order in `e` near zero.

That means small nonzero eccentricities are hard to resolve from shape alone in the near-circular regime.

With 1 percent relative measurement noise, the implied local uncertainty in `e` is:

- at `e = 0.01`
  - width residue: `9.9990e-01`
  - normalized perimeter: `1.9999`
  - major-tip response: `4.9995e-01`

- at `e = 0.1`
  - width residue: `9.9000e-02`
  - normalized perimeter: `1.9875e-01`
  - major-tip response: `4.9500e-02`

So the circular end is genuinely low-information if the measurement comes only from these geometric summaries.

### 2. Near `e = 1`, width and major-tip response become extremely sensitive

At the degenerate edge, the result splits.

The width residue and major-tip response become sharply sensitive to `e`, while the normalized perimeter remains much more moderate.

With 1 percent relative measurement noise:

- at `e = 0.99`
  - width residue: `2.0101e-04`
  - normalized perimeter: `4.3734e-03`
  - major-tip response: `1.0051e-04`

- at `e = 0.999`
  - width residue: `2.0010e-05`
  - normalized perimeter: `2.8726e-03`
  - major-tip response: `1.0005e-05`

So perimeter remains the most edge-stable observable, while width and especially major-tip response become much sharper probes of very high eccentricity.

### 3. The observables enter the “more than linear” sensitivity regime at different `e`

The relative-condition crossover `kappa(e) = 1` occurs at:

- major-tip response: `e = 0.5773502692`
- width residue: `e = 0.7071067812`
- normalized perimeter: `e = 0.9089085575`

This means perimeter stays relatively mild much farther into the high-`e` regime, while major-tip response becomes strongly sensitive earliest.

## Interpretation

This experiment gives the control-knob result a conditioning map.

It says:

- the circular regime is not “easy” from the point of view of shape-only inference; it is actually the flattest regime
- the degenerate regime is not uniformly unstable; different observables split into different sensitivity classes
- perimeter is the smoothest high-`e` observable
- major-tip response is the sharpest high-`e` observable
- width residue sits between them and shares its exact conditioning with normalized minor-tip curvature

That establishes:

> The control knob is globally meaningful, but the observables it governs are not equally informative everywhere. Near `e = 0`, they are all weakly informative. Near `e = 1`, they separate into coarse and sharp probes.

## What This Changes

Before this experiment, the project knew that `e` organized geometry and was recoverable in the known-source setting.

After this experiment, the project also knows where different geometric summaries are well-conditioned and poorly conditioned as measurements of `e`.

That is a practical upgrade, not just a conceptual one.

## Scope Of The Result

This experiment does show:

- exact conditioning behavior for the symmetric Euclidean normalized observables tested here
- strong low-information behavior near the circular end
- differentiated high-`e` behavior near the degenerate end

This experiment does not address:

- conditioning under asymmetry or anisotropy
- conditioning when source positions are unknown
- how these same conclusions transfer to non-elliptic family members

## Figures

- [edge_conditioning_overview.png](outputs/figures/edge_conditioning_overview.png)
- [edge_zoom_panels.png](outputs/figures/edge_zoom_panels.png)
- [edge_reference_heatmap.png](outputs/figures/edge_reference_heatmap.png)

The clearest figures are:

- [edge_conditioning_overview.png](outputs/figures/edge_conditioning_overview.png) for the full-range result
- [edge_zoom_panels.png](outputs/figures/edge_zoom_panels.png) for the asymptotic edge behavior

## Artifacts

Data:

- [edge_regime_metrics.csv](outputs/edge_regime_metrics.csv)
- [edge_regime_summary.json](outputs/edge_regime_summary.json)

Code:

- [run_edge_regime_stability_experiment.py](run.py#L1)

## Recommended Next Step

The hardening side of the roadmap is now in good shape.

The next natural move is to broaden the family and test whether the same budget logic has a clean twin on the fixed-difference side.

That means the next experiment should be the hyperbola flip.
