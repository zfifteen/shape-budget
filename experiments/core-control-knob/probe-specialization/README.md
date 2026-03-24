# Probe Specialization Experiment

## Purpose

This experiment tests the next theory-hardening question after representation independence:

> does BGP tell us which measurement strategy to trust, or does it improve inverse measurement strategy directly?

The target is operational usefulness.

The experiment script is [run_probe_specialization_experiment.py](run.py#L1).

## Pre-Benchmark Logic Audit

Before the benchmark, the new script was checked explicitly.

Code sanity:

- script compiled cleanly

Exact inverse audit:

- max width inverse error: `5.34e-16`
- max perimeter inverse error: `3.32e-15`
- max major-tip inverse error: `5.34e-16`

Clean strategy audit:

- max perimeter abs error: `6.26e-05`
- max width abs error: `5.34e-16`
- max major-tip abs error: `3.24e-03`

That matters because the benchmark only means anything if the probe-specific estimators are clean in the noiseless limit. They are.

Router sanity audit:

- toy threshold `tau1`: `0.4`
- toy threshold `tau2`: `0.75`
- toy order: `perimeter -> width -> major_tip`
- toy router mean abs error: `0.0133`

So the routing logic itself behaves correctly before the real benchmark starts.

## Method

The experiment has two parts.

### Part A: Ideal direct-measurement benchmark

Each probe is measured directly with matched `1 percent` relative scalar noise.

Probes:

- width residue `b / a`
- normalized perimeter `P / (2 pi a)`
- major-tip response `a kappa_major`

This is a control benchmark.

It asks which probe would win if all three were equally easy to measure as single scalars.

### Part B: Practical equal-budget benchmark

Each probe gets the same total point budget, but the sampling plan is specialized to that probe.

Perimeter strategy:

- sample the full boundary uniformly

Width strategy:

- sample near the four extremal directions
- estimate `a` and `b` from the focused extremum cloud

Major-tip strategy:

- sample near the two major-axis tips
- estimate local curvature from tip-local circle fits

This is the practical part of the experiment:

> not “which scalar is sharpest in principle,” but “which dedicated measurement strategy is best under the same noisy observation budget?”

### Adaptive policy

The adaptive policy uses a small perimeter pilot first and then routes the remaining budget to one specialized strategy.

Pilot fraction:

- `25 percent`

The router is trained on alternating `e` values and evaluated on the held-out values.

It is allowed to discover the actual phase ordering rather than forcing a hand-picked order.

## Main Result

The result is real but nuanced.

> BGP now has partial operational support as a probe-selection principle. Under equal-budget dedicated sampling, the best inverse probe really does change with depletion phase, but in this measurement model the practical competition is mostly between width and perimeter, not between perimeter and major-tip curvature. A simple perimeter-pilot router modestly beats the best fixed probe in three of the four tested regimes, but not by a large margin.

That is still a meaningful hardening result.

It means BGP is no longer only saying “the geometry is organized by the budget ratio.”

It is also saying:

> the budget phase changes which measurement strategy is best.

## Part A: Ideal Direct Measurements

Under matched direct scalar noise, the result is simple:

> major-tip response wins at every tested `e`.

Examples:

- `e = 0.05`
  - width: `0.0629`
  - perimeter: `0.0861`
  - major-tip: `0.0457`

- `e = 0.60`
  - width: `0.00848`
  - perimeter: `0.02096`
  - major-tip: `0.00430`

- `e = 0.95`
  - width: `8.28e-04`
  - perimeter: `5.76e-03`
  - major-tip: `4.11e-04`

So in the ideal scalar world, curvature is the strongest probe everywhere.

That makes the practical result below more informative, not less.

## Part B: Practical Equal-Budget Sampling

Under dedicated noisy sampling strategies, the result changes sharply.

Major-tip curvature becomes too fragile to win.

Across all four tested conditions:

- perimeter wins in part of the `e` range
- width wins in part of the `e` range
- major-tip never wins as the best fixed practical strategy

Best-probe counts across the tested `e` grid:

- `dense_low_noise`
  - width: `11`
  - perimeter: `8`

- `dense_medium_noise`
  - width: `13`
  - perimeter: `6`

- `sparse_medium_noise`
  - perimeter: `10`
  - width: `9`

- `sparse_high_noise`
  - width: `13`
  - perimeter: `6`

So the practical specialization is real, but it is a width-perimeter split rather than a perimeter-curvature split.

## The Phase Split

The learned phase policies are simple and interpretable.

For three of the four regimes, the discovered order is:

- `width -> perimeter -> major_tip`

with a single effective switch around:

- `tau1 = 0.6` in `dense_low_noise`
- `tau1 = 0.7` in `dense_medium_noise`
- `tau1 = 0.7` in `sparse_medium_noise`

That means:

- lower-to-middle depletion favors the dedicated width strategy
- higher depletion favors the dedicated perimeter strategy
- major-tip curvature remains too noisy to become competitive

The hardest regime is `sparse_high_noise`.

There:

- the practical router learns `width -> perimeter -> major_tip` with `tau1 = 0.8`
- the phase oracle prefers `perimeter -> width -> major_tip` with `tau1 = 0.1`

So the hardest sparse regime still has a routing gap.

That is useful to know.

## Router Performance

The router is not a dramatic win, but it is a real one in most regimes.

Mean absolute error on held-out `e` values:

- `dense_low_noise`
  - best fixed: `0.00351`
  - router: `0.00338`
  - improvement: `1.037x`

- `dense_medium_noise`
  - best fixed: `0.01285`
  - router: `0.01233`
  - improvement: `1.042x`

- `sparse_medium_noise`
  - best fixed: `0.01564`
  - router: `0.01586`
  - change: `0.987x`

- `sparse_high_noise`
  - best fixed: `0.02653`
  - router: `0.02629`
  - improvement: `1.009x`

So the practical router:

- beats the best fixed probe in `3` of `4` regimes
- loses slightly in `1`
- never closes the full gap to the trial oracle

This is exactly the kind of result that strengthens BGP while keeping the remaining routing gap explicit.

## Interpretation

This experiment changes the read in three useful ways.

What it establishes:

- BGP now has real operational content beyond shape description
- depletion phase really does predict probe specialization under practical measurement constraints
- the best practical probe is not universal across the `e` range
- even a simple pilot-and-route policy can recover a measurable part of that gain

What it does not establish:

- the claim that a single simple router is already near-optimal
- the claim that major-tip curvature is the practical late-phase winner under this measurement model

The established reading is:

> BGP now functions as an experimental-control principle. It can guide which probe family to trust, but the practical winning split depends on how the probes are actually measured, and curvature loses badly once local estimation noise is accounted for.

That is still a substantial hardening step.

## Why This Matters For BGP

This strengthens BGP.

Not because every initial expectation survived.

Because the central operational claim survived in direct form:

- the budget phase changes which probe is best
- the winning probe depends on the observation model
- a phase-aware policy can outperform a fixed probe

That is exactly the kind of evidence a control principle needs.

It also gives the next priority more clearly:

- the immediate next theory-hardening task should now move to scope boundaries and falsification structure
- not because operational usefulness failed
- but because it has now been established strongly enough to justify the next step

## Figures

Key figures:

- [probe_specialization_ideal.png](outputs/figures/probe_specialization_ideal.png)
- [probe_specialization_empirical_curves.png](outputs/figures/probe_specialization_empirical_curves.png)
- [probe_specialization_best_probe_map.png](outputs/figures/probe_specialization_best_probe_map.png)
- [probe_specialization_router_comparison.png](outputs/figures/probe_specialization_router_comparison.png)

The most important contrast is between the first two figures.

The ideal benchmark says curvature would win if all probes were equally easy to measure.

The practical benchmark says they are not.

That gap is the key message:

> BGP does not just rank abstract observables. It helps organize real measurement strategy once probe difficulty is folded back into the problem.
