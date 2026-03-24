# Shift-Marginalized Pose Experiment

## Purpose

This experiment tests a narrower hypothesis about the pose-free anisotropic bottleneck.

The earlier results showed that:

- hidden rotation broadens the near-optimal latent family mainly in `alpha`
- a low-order pose-invariant spectral encoding does not rescue `alpha`
- blunt invariant compression throws away too much boundary structure

That leaves a more targeted possibility:

- maybe the current inverse is still too brittle because it trusts only the single best shift
- maybe some of the `alpha` ambiguity comes from candidates winning on one lucky pose alignment

This experiment probes that possibility directly.

## Research Question

If we keep the full radial signature but replace the hard best-shift score with a soft shift-marginalized score, does `alpha` become more stable under pose-free observation?

## Pre-Benchmark Logic Audit

Before running the benchmark, the new scoring rule was checked in two ways.

First, the script compiled cleanly.

Second, the marginalized score was tested for exact pose consistency.

The audit confirmed that:

- under arbitrary clean full rotations, the marginalized method still recovers the exact bank candidate
- the marginalized score is rotation-invariant when the observation and mask rotate together, with max score discrepancy `1.73e-17`

The clean exact-match score is not zero because the soft score integrates over all `64` shifts. Its clean value is the expected softmin constant:

- max clean exact-match marginalized score: `4.16e-4`

That is expected behavior, not an error.

The experiment script is [run_shift_marginalized_pose_experiment.py](run.py#L1).

## Method

The comparison is between two pose-free inverse rules on the same weighted anisotropic three-source family.

Baseline:

- for each candidate, evaluate masked fit over all cyclic shifts
- keep only the single best shift
- rank candidates by that minimum score

Shift-marginalized method:

- evaluate masked fit over all cyclic shifts
- compute a soft shift-marginalized score instead of a hard minimum
- still recover the best shift of the selected candidate for final fit evaluation

So the new method does not discard pose and does not discard boundary detail.

It changes only one thing:

- pose is treated as distributed evidence rather than a winner-take-all alignment

The softmin temperature is tied to the regime noise floor:

- `tau = max(noise_sigma^2, 1e-4)`

## Parameter Sweep

Reference bank:

- anisotropy-aware bank size: `300`

Test set:

- `40` trials per observation regime

Envelope:

- top-`10` candidates per trial

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

## Main Result

The result is encouraging but not dramatic.

> Soft shift-marginalization modestly improves `alpha` recovery in most regimes, especially in the harder sparse settings, and it consistently tightens the top-`10` `alpha` envelope, but the gains are moderate rather than transformative.

The summary file is [shift_marginalized_pose_summary.json](outputs/shift_marginalized_pose_summary.json).

At the regime level:

- `alpha` improves in three regimes:
  - `full_noisy`: `0.2647 -> 0.2548`
  - `sparse_full_noisy`: `0.2734 -> 0.2360`
  - `sparse_partial_high_noise`: `0.2871 -> 0.2602`
- `alpha` is essentially unchanged in one regime:
  - `full_clean`: `0.2500 -> 0.2500`
- `alpha` worsens slightly in one regime:
  - `partial_arc_noisy`: `0.1294 -> 0.1325`

The ratio-of-means summary is the cleanest read:

- best baseline-over-marginalized `alpha` ratio: `1.1586`
- worst baseline-over-marginalized `alpha` ratio: `0.9763`

So the new score helps, but modestly.

## By Regime

- `full_clean`
  - alpha error: `0.2500 -> 0.2500`
  - alpha-span top-`10`: `0.5922 -> 0.5900`
  - fit RMSE: `0.00773 -> 0.00774`
  - geometry MAE: `0.0829 -> 0.0859`

- `full_noisy`
  - alpha error: `0.2647 -> 0.2548`
  - alpha-span top-`10`: `0.5275 -> 0.5251`
  - fit RMSE: `0.00858 -> 0.00859`
  - geometry MAE: `0.0843 -> 0.0859`

- `partial_arc_noisy`
  - alpha error: `0.1294 -> 0.1325`
  - alpha-span top-`10`: `0.5220 -> 0.5136`
  - near-tie diverse fraction: `0.500 -> 0.425`
  - fit RMSE: `0.01693 -> 0.01837`

- `sparse_full_noisy`
  - alpha error: `0.2734 -> 0.2360`
  - alpha-span top-`10`: `0.5527 -> 0.5398`
  - near-tie diverse fraction: `0.625 -> 0.600`
  - fit RMSE: `0.01329 -> 0.01493`

- `sparse_partial_high_noise`
  - alpha error: `0.2871 -> 0.2602`
  - alpha-span top-`10`: `0.5410 -> 0.4255`
  - near-tie diverse fraction: `0.850 -> 0.600`
  - fit RMSE: `0.05042 -> 0.04357`
  - geometry MAE: `0.0970 -> 0.0903`

The clearest positive regime is the hardest one:

- in `sparse_partial_high_noise`, `alpha` improves
- the `alpha` envelope tightens substantially
- near-tie diverse families drop materially
- fit and geometry both improve

That is the strongest signal in the whole artifact.

## Interpretation

This result says something precise.

The hard best-shift rule was part of the bottleneck.

Not all of it, but part of it.

If the best-shift rule were irrelevant, then soft shift-marginalization would not consistently tighten the `alpha` envelope.

Instead, what we see is:

- the top-`10` `alpha` span shrinks in every regime
- near-tie diverse families shrink in the hardest ambiguous cases
- `alpha` itself improves in most noisy regimes

So the pose logic matters.

At the same time, the improvement is not large enough to claim the problem is solved.

In plain language:

- letting candidates win on one lucky shift was helping some wrong `alpha` explanations survive
- treating pose as softer evidence cleans that up a bit
- but it does not collapse the ambiguity enough to make `alpha` easy

That is exactly the kind of intermediate result that helps the program:

- it identifies a real mechanism
- it improves on the baseline
- and it also shows that more work is still needed

## What This Establishes

This experiment does show:

- the hard best-shift pose rule contributes to the pose-free `alpha` bottleneck
- soft shift-marginalization modestly improves `alpha` in most noisy regimes
- the `alpha` ambiguity envelope tightens in every regime
- the strongest gains appear in the hardest sparse partial setting

This experiment does not address:

- that the `alpha` bottleneck is solved
- that soft marginalization is the best pose treatment available
- that weight recovery improves too; in fact, weight MAE often gets a bit worse

## Figures

- [shift_marginalized_pose_overview.png](outputs/figures/shift_marginalized_pose_overview.png)
- [shift_marginalized_pose_trial_scatter.png](outputs/figures/shift_marginalized_pose_trial_scatter.png)

The clearest figure is [shift_marginalized_pose_overview.png](outputs/figures/shift_marginalized_pose_overview.png), because it shows the pattern directly:

- small changes in easy regimes
- meaningful tightening in the hard sparse regime

## Artifacts

Data:

- [shift_marginalized_pose_trials.csv](outputs/shift_marginalized_pose_trials.csv)
- [shift_marginalized_pose_summary.csv](outputs/shift_marginalized_pose_summary.csv)
- [shift_marginalized_pose_summary.json](outputs/shift_marginalized_pose_summary.json)

Code:

- [run_shift_marginalized_pose_experiment.py](run.py#L1)

## Recommended Next Step

The natural next step is a richer version of the same idea:

- candidate-conditioned shift marginalization with learned or adaptive temperature
- local geometry-plus-alpha refinement on top of the marginalized score
- or a hybrid two-stage inverse where the marginalized score narrows the family and a second-stage local search resolves the remaining `alpha` ambiguity
