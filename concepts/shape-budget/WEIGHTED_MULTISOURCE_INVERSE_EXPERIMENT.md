# Weighted Multi-Source Inverse Experiment

## Purpose

This experiment asks whether the weighted three-source Shape Budget control object is recoverable from boundary data when the source positions and weights are not given in advance.

The forward weighted family was:

\[
w_1\|x-p_1\| + w_2\|x-p_2\| + w_3\|x-p_3\| = S
\]

with positive normalized weights.

The recovery target is not the absolute placement or absolute scale.

It is the normalized control object:

- the normalized source triangle relative to budget
- the normalized weight vector in the simplex

## Research Question

Can a boundary-only inverse recover the compact weighted multi-source control object well enough to be useful, and does it outperform an equal-weight baseline that ignores the weight degrees of freedom?

## Scope Of This Inverse Test

This is a deliberate first inverse test, not the hardest possible one.

The observations are:

- boundary-only
- centered and scale-normalized from the boundary itself
- kept in canonical pose

So this experiment removes translation and scale from the observed boundary, but it does not randomize orientation.

That is intentional.

The question here is whether the weighted control object is operational at all under a simple boundary-only encoding, not whether the fully arbitrary-pose inverse is already solved.

## Inverse Method

The experiment script is [run_weighted_multisource_inverse_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_weighted_multisource_inverse_experiment.py#L1).

The inverse is deliberately simple:

1. build a weighted reference bank of forward models
2. encode each boundary as a centroid-centered, mean-radius-normalized radial signature
3. recover the nearest weighted reference signature under masked L2 distance

There is no custom optimizer here.

That is part of the point.

If a simple reference-bank inverse works, that is strong evidence that the weighted control object is genuinely recoverable and not just elegant on paper.

## Baseline

The baseline is an equal-weight reference bank:

- same geometric ranges
- weights fixed to `(1/3, 1/3, 1/3)`

This asks a very clean question:

- if the true family is weighted, how much do we lose by forcing the inverse to pretend that all sources matter equally?

## Parameter Sweep

Reference banks:

- weighted bank size: `300`
- equal-weight baseline bank size: `150`

Test set:

- `40` weighted test cases per observation regime

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

Signature encoding:

- boundary samples per forward curve: `96`
- radial-signature bins: `64`

## Recovery Metrics

The experiment records three kinds of quantities.

Control-object recovery:

- geometry MAE on the normalized edge-length triple
- weight MAE on the normalized weight vector

Fit quality:

- clean-signature RMSE of the recovered weighted-bank model
- clean-signature RMSE of the equal-weight baseline model

Because the canonical three-source setup has a left-right reflection symmetry, the geometry and weight errors are evaluated with a symmetry-aware source-1/source-2 swap when that gives the smaller error.

## Main Result

The inverse works meaningfully well.

> In the canonical-pose boundary-only setting, a simple weighted reference-bank inverse can recover the normalized three-source geometry and normalized weights to a useful degree, and it consistently outperforms an equal-weight baseline.

The summary file is [weighted_multisource_inverse_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/weighted_multisource_inverse_summary.json).

Global summary:

- reference bank size: `300`
- equal-weight baseline bank size: `150`
- trials per regime: `40`
- best mean geometry MAE: `6.2280e-02`
- worst mean geometry MAE: `8.5950e-02`
- best mean weight MAE: `8.6077e-02`
- worst mean weight MAE: `1.2484e-01`
- best mean weighted-fit RMSE: `2.9443e-03`
- worst mean weighted-fit RMSE: `1.5177e-02`
- smallest mean fit-improvement factor over the equal-weight baseline: `1.8771`
- largest mean fit-improvement factor: `5.5255`

By regime:

- `full_clean`
  - geometry MAE mean: `6.6360e-02`
  - weight MAE mean: `9.9616e-02`
  - weighted-fit RMSE mean: `2.9443e-03`
  - equal-weight baseline RMSE mean: `1.3205e-02`
  - mean improvement factor: `5.5255`

- `full_noisy`
  - geometry MAE mean: `7.5033e-02`
  - weight MAE mean: `9.3243e-02`
  - weighted-fit RMSE mean: `3.1231e-03`
  - equal-weight baseline RMSE mean: `1.2232e-02`
  - mean improvement factor: `4.8779`

- `partial_arc_noisy`
  - geometry MAE mean: `6.2280e-02`
  - weight MAE mean: `8.6077e-02`
  - weighted-fit RMSE mean: `4.2799e-03`
  - equal-weight baseline RMSE mean: `1.5237e-02`
  - mean improvement factor: `4.2167`

- `sparse_full_noisy`
  - geometry MAE mean: `7.7342e-02`
  - weight MAE mean: `9.8146e-02`
  - weighted-fit RMSE mean: `6.4296e-03`
  - equal-weight baseline RMSE mean: `1.4043e-02`
  - mean improvement factor: `2.9858`

- `sparse_partial_high_noise`
  - geometry MAE mean: `8.5950e-02`
  - weight MAE mean: `1.2484e-01`
  - weighted-fit RMSE mean: `1.5177e-02`
  - equal-weight baseline RMSE mean: `1.8457e-02`
  - mean improvement factor: `1.8771`

These numbers support a careful statement:

- the inverse is not perfect
- the control object is still recoverable in a meaningful way
- the weight degrees of freedom are operational, not decorative

## Interpretation

This is a real step forward for the project.

The weighted multi-source forward experiment showed that normalized placement plus normalized participation organizes the family.

This inverse experiment shows that the same compact object is not just a forward description.

It is recoverable from boundary data by a simple method.

That matters because it turns the weighted multi-source control object into something you can infer, not just something you can write down after the fact.

The strongest reading is:

> normalized placement plus normalized participation is an operational state variable for the weighted three-source family, at least in the canonical-pose boundary-only setting tested here.

## What This Changes

Before this experiment, the multi-source program had strong forward structure but only a hypothesis about inverse usefulness.

After this experiment, there is direct evidence that:

- the weighted control object can be recovered from boundary-only data
- the recovery remains useful under noise and partial observation
- forcing equal weights leaves measurable predictive performance on the table

That moves the project closer to a genuine inferential framework.

## Scope Of The Result

This experiment does show:

- boundary-only recovery of the normalized weighted control object in canonical pose
- useful geometry and weight recovery across multiple observation regimes
- consistent improvement over an equal-weight inverse baseline

This experiment does not yet show:

- full arbitrary-pose recovery
- recovery of absolute scale
- recovery when the source count exceeds three
- recovery in anisotropic or otherwise non-Euclidean media

## Figures

- [weighted_multisource_inverse_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/figures/weighted_multisource_inverse_heatmap.png)
- [weighted_multisource_inverse_baseline.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/figures/weighted_multisource_inverse_baseline.png)
- [weighted_multisource_inverse_examples.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/figures/weighted_multisource_inverse_examples.png)

The clearest figures are:

- [weighted_multisource_inverse_heatmap.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/figures/weighted_multisource_inverse_heatmap.png) for the actual recovery error scale
- [weighted_multisource_inverse_baseline.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/figures/weighted_multisource_inverse_baseline.png) for the evidence that the weighted control object beats the equal-weight shortcut

## Artifacts

Data:

- [weighted_multisource_inverse_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/weighted_multisource_inverse_trials.csv)
- [weighted_multisource_inverse_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/weighted_multisource_inverse_summary.csv)
- [weighted_multisource_inverse_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/weighted_multisource_inverse_outputs/weighted_multisource_inverse_summary.json)

Code:

- [run_weighted_multisource_inverse_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_weighted_multisource_inverse_experiment.py#L1)

## Recommended Next Step

The cleanest next step is pose-free weighted inversion.

At this point the project has:

- strong forward structure for weighted multi-source geometry
- a first boundary-only inverse in canonical pose
- a clear signal that the weight simplex is inferentially useful

The next question is whether the same recovery story survives once rotation is also unknown, and then whether the inverse remains stable when the medium itself is anisotropic or otherwise warped.
