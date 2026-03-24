# Identifiability and Baseline Comparison

## Purpose

This experiment tests whether the Shape Budget control knob is operational, not just elegant.

The earlier control-knob experiment established that, under the symmetric constant-sum two-source Euclidean process, `e = c/a` is sufficient to organize normalized geometry across scale. This experiment asks two follow-up questions:

1. can `e` be recovered from noisy boundary data when the source positions are known?
2. does `e` outperform raw alternatives such as `d`, `S`, or the pair `(d, S)` as a predictive summary variable?

## Research Questions

### Part A: Identifiability

If the two source positions are known, how accurately can `e` be recovered from noisy, partial, or sparse observations of the boundary?

### Part B: Baseline comparison

Under a scale-held-out split, does `e` preserve predictive power for normalized observables better than:

- raw separation `d`
- raw budget `S`
- the unnormalized pair `(d, S)`

## Experiment Design

The experiment script is [run_identifiability_and_baselines_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_identifiability_and_baselines_experiment.py#L1).

### Part A

For each true `e` and budget scale `a`, the script samples ellipse boundary points, adds isotropic Gaussian noise, and estimates `a` from the median focal-sum value:

\[
\hat{a} = \frac{1}{2} \operatorname{median}\left(\|P - F_1\| + \|P - F_2\|\right)
\]

with

\[
\hat{e} = \frac{c}{\hat{a}}
\]

The source positions are treated as known. This is a deliberate first identifiability test, not the fully unknown inverse problem.

### Part B

The script builds a synthetic dataset over varying `e` and varying absolute scale `a`, then predicts four normalized observables:

- width residue `b/a`
- exact normalized perimeter `P / (2\pi a) = (2/\pi) E(e)` where `E` is the complete elliptic integral of the second kind with modulus `e`
- normalized major-tip response `a \kappa_major = 1 / (1 - e^2)`
- normalized minor-tip response `a \kappa_minor = \sqrt{1 - e^2}`

Each target is predicted using low-capacity polynomial regressions from:

- `e` alone
- `d` alone
- `S` alone
- `(d, S)` together

The train/test split is deliberately scale-held-out:

- train on `a <= 2.5`
- test on `a > 2.5`

That makes the comparison about generalization across scale rather than interpolation at a single scale.

## Parameter Sweep

### Identifiability

- `e` values: 17 values from `0.10` to `0.90`
- budget scales `a`: `0.75, 1.0, 1.5, 2.5, 4.0`
- replicates per condition: 120

Observation conditions:

- `full_low_noise`: 200 boundary points, full boundary, noise `0.005 a`
- `full_medium_noise`: 200 boundary points, full boundary, noise `0.02 a`
- `partial_arc_medium_noise`: 80 points, first-quadrant arc, noise `0.02 a`
- `sparse_full_medium_noise`: 16 boundary points, full boundary, noise `0.02 a`
- `sparse_partial_high_noise`: 12 points, first-quadrant arc, noise `0.03 a`

### Baseline comparison

- synthetic samples: 5000
- `e` sampled uniformly in `[0.05, 0.95]`
- `a` sampled uniformly in `[0.5, 5.0]`

## Main Result

This experiment supports two concrete statements:

1. `e` is recoverable with high accuracy from noisy boundary data when the source positions are known.
2. `e` is a much stronger scale-generalizing summary variable than raw `d`, raw `S`, or a low-capacity model on `(d, S)`.

The summary file is [experiment_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/experiment_summary.json).

### Identifiability summary

Mean absolute error in recovered `e`:

- `full_low_noise`: `1.4642e-04`
- `full_medium_noise`: `5.9549e-04`
- `partial_arc_medium_noise`: `9.2215e-04`
- `sparse_full_medium_noise`: `1.9801e-03`
- `sparse_partial_high_noise`: `3.4627e-03`

Worst 95th-percentile absolute error across the `e` sweep:

- `full_low_noise`: `5.2771e-04`
- `full_medium_noise`: `2.1715e-03`
- `partial_arc_medium_noise`: `3.1544e-03`
- `sparse_full_medium_noise`: `6.8930e-03`
- `sparse_partial_high_noise`: `1.2621e-02`

Even in the harshest tested condition, the 95th-percentile absolute error remained about `1.26e-02`.

### Baseline comparison summary

Test RMSE under the scale-held-out split:

- `width_residue`
  `e_only`: `1.0469e-03`
  `d_only`: `8.0707`
  `S_only`: `12.8062`
  `d_and_S`: `1.2695`

- `normalized_perimeter`
  `e_only`: `1.8962e-04`
  `d_only`: `3.6377`
  `S_only`: `5.6473`
  `d_and_S`: `5.5897e-01`

- `major_tip_response`
  `e_only`: `1.5885e-01`
  `d_only`: `36.2331`
  `S_only`: `116.7075`
  `d_and_S`: `13.2530`

- `minor_tip_response`
  `e_only`: `1.0469e-03`
  `d_only`: `8.0707`
  `S_only`: `12.8062`
  `d_and_S`: `1.2695`

The direction is unambiguous: `e` is dramatically better than the raw alternatives under scale shift.

## Interpretation

This is the first experiment in the project that directly supports the claim that the control knob is operational.

It shows three things:

- the variable is not just available from the closed-form ellipse equation; it can be recovered from noisy observations in a stable way when the source positions are known
- the normalization matters; raw `d` and raw `S` alias together geometrically different cases
- even giving the model both raw variables `(d, S)` is still much weaker than using the normalized ratio `e` directly, at least under the low-capacity scale-held-out setup used here

That supports a careful statement:

> In the symmetric known-source setting, `e = c/a` is not only a valid descriptive parameter. It is a practically useful summary variable for recovery and prediction.

## Scope Of What This Does And Does Not Show

This experiment does show:

- known-source identifiability
- scale-generalizing predictive usefulness

This experiment does not yet show:

- recovery when the source positions are unknown
- recovery under model mismatch or asymmetry
- that `e` remains sufficient outside the symmetric Euclidean constant-sum setting

Those remain separate research questions.

## Figures

- [identifiability_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/figures/identifiability_heatmap.png)
- [identifiability_recovery.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/figures/identifiability_recovery.png)
- [baseline_collapse.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/figures/baseline_collapse.png)
- [baseline_rmse.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/figures/baseline_rmse.png)

The clearest figures are:

- [identifiability_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/figures/identifiability_heatmap.png) for robustness across observation regimes
- [baseline_rmse.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/figures/baseline_rmse.png) for the scale-generalization comparison

## Artifacts

Data:

- [identifiability_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/identifiability_trials.csv)
- [identifiability_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/identifiability_summary.csv)
- [baseline_rmse.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/identifiability_baselines_outputs/baseline_rmse.csv)

Code:

- [run_identifiability_and_baselines_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_identifiability_and_baselines_experiment.py#L1)

## Recommended Next Step

The most useful next hardening step is now the manifold-dimension test.

The current evidence says:

- one-knob collapse holds in the symmetric case
- one-knob collapse fails in the first asymmetric pilot
- `e` is recoverable and operational in the symmetric known-source case

The next clean question is whether the symmetric shape family really behaves like a one-dimensional manifold in boundary space, not just in a handful of chosen observables.
