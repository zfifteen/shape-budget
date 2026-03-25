# Oracle Alignment Ceiling Experiment

## Purpose

This experiment asks the cleanest next question after the failed practical locking trial.

The orientation-locking experiment showed that:

- simple observation-only locking can be exact in ideal conditions
- simple locking gives only small gains in full observations
- simple locking degrades badly under partial and sparse support

That still leaves one crucial question open:

> is the lost `alpha` signal actually there to be recovered if pose were handled perfectly, or does most of the gap remain even with ideal orientation information?

This note answers that directly.

## Research Question

If the inverse is given the true observation pose, how much of the pose-free anisotropic `alpha` penalty disappears?

## Pre-Benchmark Logic Audit

Before the benchmark, all three alignment paths were audited.

Practical methods rechecked:

- harmonic lock rotation invariance: max aligned RMSE `0.0`
- principal-axis lock rotation invariance: max aligned RMSE `0.0`
- harmonic clean exact-bank recovery: `1.0`
- principal-axis clean exact-bank recovery: `1.0`

Oracle path audited directly:

- oracle identity audit cases: `30`
- max oracle identity RMSE: `0.0`
- oracle clean exact-bank recovery: `1.0`
- oracle clean max fit RMSE: `0.0`

So the ceiling path is exact in the idealized setting.

That matters because the benchmark result can then be read as a true upper-bound statement about alignment headroom, not an artifact of a sloppy oracle construction.

The experiment script is [run_oracle_alignment_ceiling_experiment.py](run.py#L1).

## Method

The same pose-free weighted anisotropic inverse trial is evaluated four ways:

- shift-aware baseline
- harmonic orientation lock
- principal-axis orientation lock
- oracle alignment

The oracle method is simple:

1. generate the usual pose-free observed boundary
2. use the true observation shift to roll that boundary back to canonical orientation
3. run the same nearest-neighbor inverse on the same anisotropic reference bank

So the oracle changes only one thing:

- pose is given perfectly

It does not give the inverse any help with geometry, weights, or `alpha`.

That makes it the right ceiling for the current representation and bank family.

## Parameter Sweep

Reference bank:

- anisotropy-aware bank size: `300`

Test set:

- `40` trials per observation regime

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

## Main Result

The result is very strong.

> Oracle alignment removes most of the pose-free `alpha` penalty across every tested regime, while geometry changes only modestly. The lost `alpha` signal is genuinely present; the main problem is practical alignment stability, not absence of recoverable information.

The summary file is [oracle_alignment_ceiling_summary.json](outputs/oracle_alignment_ceiling_summary.json).

Oracle baseline-over-ceiling `alpha` improvement factor:

- best regime: `13.21x`
- worst regime: `5.65x`

That is the clean headline.

## By Regime

- `full_clean`
  - baseline alpha: `0.2750`
  - oracle alpha: `0.0218`
  - improvement: `12.62x`
  - geometry ratio oracle/baseline: `1.028`

- `full_noisy`
  - baseline alpha: `0.1251`
  - oracle alpha: `0.0174`
  - improvement: `7.17x`
  - geometry ratio oracle/baseline: `0.977`

- `partial_arc_noisy`
  - baseline alpha: `0.3038`
  - oracle alpha: `0.0230`
  - improvement: `13.21x`
  - geometry ratio oracle/baseline: `0.898`

- `sparse_full_noisy`
  - baseline alpha: `0.1929`
  - oracle alpha: `0.0188`
  - improvement: `10.28x`
  - geometry ratio oracle/baseline: `0.910`

- `sparse_partial_high_noise`
  - baseline alpha: `0.2540`
  - oracle alpha: `0.0450`
  - improvement: `5.65x`
  - geometry ratio oracle/baseline: `1.043`

So the ceiling is not just better in the easy cases.

It is dramatically better in the hard partial and sparse cases too.

That is the key result.

## Practical Locks Versus Oracle

This comparison is what really sharpens the conclusion.

In the full regimes, the practical locks recover only a small fraction of the available oracle gain:

- `full_clean`
  - harmonic captures about `0.192`
  - principal-axis captures about `0.196`

- `full_noisy`
  - harmonic captures about `-0.010`
  - principal-axis captures about `0.192`

In the sparse regimes, the practical locks often move in the wrong direction relative to the available oracle headroom:

- `sparse_full_noisy`
  - harmonic fraction of oracle gain: `-0.504`
  - principal-axis fraction of oracle gain: `-0.290`

- `sparse_partial_high_noise`
  - harmonic fraction of oracle gain: `-0.186`
  - principal-axis fraction of oracle gain: `-0.213`

So the oracle gain is real, but the naive locks capture little of it and sometimes actively push away from it.

## Interpretation

This experiment changes the posture of the project in a very helpful way.

It rules out one important pessimistic reading:

- that the pose-free `alpha` gap mostly remains even if pose were handled perfectly

That is not what happens.

Instead:

- once pose is restored correctly, `alpha` improves dramatically
- geometry stays roughly where it already was
- the information was there all along

In plain language:

- the budget-governed geometry was already readable
- the anisotropy signal was being scrambled mainly by pose handling
- perfect pose largely unscrumbles it

That means the current solver challenge is now much clearer:

- not “there is no `alpha` signal in the boundary”
- but “our practical pose-handling methods are too unstable under incomplete observations to extract that signal”

That is a major narrowing of the problem.

## What This Establishes

This experiment does show:

- the pose-free `alpha` penalty is mostly alignment headroom, not missing signal
- perfect pose information recovers large `alpha` gains in every regime
- geometry changes only modestly under the same oracle intervention
- the current practical locks capture only a small fraction of the available gain and often fail under sparse or partial support

This experiment does not address:

- which practical alignment method can approach the oracle ceiling
- whether the remaining gap after oracle is due mainly to observation noise, bank discretization, or both
- exactly where alignment becomes ill-posed as a function of anisotropy strength, support fraction, and source geometry

## Why The Result Matters

For the BGP program, this is a big deal.

It says the hidden state really is recoverable to a much greater extent than the current practical pipeline suggests.

So the project should not react by weakening the latent-variable claim.

It should react by focusing harder on:

- robust pose handling under incomplete support
- failure maps for alignment stability
- representations that preserve the recoverable `alpha` signal while staying stable when observations are sparse

## Figures

- [oracle_alignment_ceiling_alpha_methods.png](outputs/figures/oracle_alignment_ceiling_alpha_methods.png)
- [oracle_alignment_ceiling_gap.png](outputs/figures/oracle_alignment_ceiling_gap.png)

The clearest figure is [oracle_alignment_ceiling_alpha_methods.png](outputs/figures/oracle_alignment_ceiling_alpha_methods.png), because it shows the whole point in one glance:

- baseline high
- practical locks inconsistent
- oracle dramatically lower across every regime

## Artifacts

Data:

- [oracle_alignment_ceiling_summary.json](outputs/oracle_alignment_ceiling_summary.json)
- [oracle_alignment_ceiling_summary.csv](outputs/oracle_alignment_ceiling_summary.csv)
- [oracle_alignment_ceiling_trials.csv](outputs/oracle_alignment_ceiling_trials.csv)

Code:

- [run_oracle_alignment_ceiling_experiment.py](run.py#L1)
