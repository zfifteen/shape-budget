# Orientation Locking Experiment

## Purpose

This experiment is the first direct intervention test implied by the symmetry-orbit hypothesis.

The orbit-proximity diagnostic suggested that:

- `alpha` is more rotation-absorbable than geometry in the current radial-signature space
- so a symmetry-breaking pre-alignment step should recover `alpha` much more than geometry

This note tests that idea directly.

## Research Question

If we break rotational symmetry before inverse matching, does `alpha` recover substantially more than geometry in the pose-free weighted anisotropic inverse?

## Pre-Benchmark Logic Audit

Before the benchmark, both alignment rules were audited in two ways.

Methods:

- harmonic lock
- principal-axis lock

First, both methods were tested for clean full-signature rotation invariance.

Result:

- harmonic max aligned rotation RMSE: `0.0`
- principal-axis max aligned rotation RMSE: `0.0`

Second, both methods were tested for clean exact-bank recovery after locking.

Result:

- harmonic exact recovery fraction: `1.0`
- principal-axis exact recovery fraction: `1.0`
- max aligned fit RMSE for both: `0.0`

So the locking logic itself is correct in the idealized clean full-observation setting.

That matters because any failure in the benchmark is then a failure of practical alignment under masking and noise, not a coding bug in the lock itself.

The experiment script is [run_orientation_locking_experiment.py](run.py#L1).

## Method

The comparison uses three pose-free anisotropic inverse rules on the same trials:

- the existing shift-aware baseline
- harmonic orientation locking
- principal-axis orientation locking

For the locking methods:

1. estimate a canonical orientation from the observed boundary alone
2. roll the observed signature and mask into that locked orientation
3. compare against a bank that has been canonicalized by the same locking rule

So the lock is observation-only.

It does not look at the candidate bank during orientation selection.

This is a real symmetry-breaking pre-alignment, not a disguised shift search over candidates.

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

The result is a clear negative on the strong version of the orientation-locking intervention.

> Simple observation-only orientation locking does not rescue `alpha` overall. It gives only small gains in fully observed conditions and degrades badly once support becomes partial or sparse.

The summary file is [orientation_locking_summary.json](outputs/orientation_locking_summary.json).

Best improvements:

- harmonic lock best alpha gain over baseline: `1.039x`
- principal-axis lock best alpha gain over baseline: `1.113x`

Worst degradations:

- harmonic lock worst alpha factor: `0.670x`
- principal-axis lock worst alpha factor: `0.784x`

So neither method comes close to a dramatic `alpha` rescue.

## By Regime

### Full observation regimes

There are small positive signs here.

- `full_clean`
  - baseline alpha: `0.2251`
  - harmonic alpha: `0.2167`
  - principal-axis alpha: `0.2338`

- `full_noisy`
  - baseline alpha: `0.2386`
  - harmonic alpha: `0.2361`
  - principal-axis alpha: `0.2144`

So in the fully observed cases:

- harmonic lock helps slightly or is nearly neutral
- principal-axis lock helps modestly in `full_noisy`

Geometry remains fairly stable in those same regimes.

That part is consistent with the orbit-proximity idea.

### Partial and sparse regimes

This is where the main result pattern changes.

- `partial_arc_noisy`
  - baseline alpha: `0.2408`
  - harmonic alpha: `0.3595`
  - principal-axis alpha: `0.3070`
  - baseline fit RMSE: `0.0140`
  - harmonic fit RMSE: `0.0795`
  - principal-axis fit RMSE: `0.0613`

- `sparse_full_noisy`
  - baseline alpha: `0.2153`
  - harmonic alpha: `0.3013`
  - principal-axis alpha: `0.2732`

- `sparse_partial_high_noise`
  - baseline alpha: `0.2483`
  - harmonic alpha: `0.3128`
  - principal-axis alpha: `0.2827`

So once support is incomplete, the simple locks stop helping and often hurt a lot.

That is the key result.

## Interpretation

This experiment narrows the symmetry-orbit mechanism in an important way.

It does not say the orbit idea was wrong.

It says:

- the locking intervention was too brittle in the practically hard regimes

The clean audits and the full-observation behavior matter here.

They tell us:

- the lock itself is mathematically coherent
- the lock can help a little when orientation information is truly available in the observation
- but partial support and sparse noisy sampling make naive pre-alignment unstable enough to overwhelm the potential gain

So the right reading is:

- orbit proximity is probably real
- but simple observation-only locking is not robust enough to exploit it under the regimes that actually matter most

That is a valuable result, because it tells us exactly what not to do next:

- do not assume that any symmetry-breaking pre-alignment will automatically rescue `alpha`

## What This Establishes

This experiment does show:

- both locking rules are exact under clean full-signature rotation tests
- simple orientation locking can give small `alpha` gains in fully observed settings
- simple orientation locking degrades sharply under partial and sparse support
- the main challenge is now practical alignment stability, not logical impossibility of locking

This experiment does not address:

- whether better alignment could still recover most of the lost `alpha`
- whether the remaining failure is mainly due to missing support, noise, or both
- whether oracle alignment would close the pose-free gap

## Why The Result Matters

This is a stronger result than a mere “didn’t work.”

It separates three possibilities:

- coding failure: ruled out by the clean audits
- wrong mechanism: ruled out, because full-observation cases still show the expected direction
- brittle intervention under incomplete observations: established

That makes the next step much clearer.

The right next experiment is not:

- more naive locking variants

It is:

- oracle alignment ceiling

That is now the cleanest way to tell how much headroom alignment really has.

## Figures

- [orientation_locking_alpha_geometry.png](outputs/figures/orientation_locking_alpha_geometry.png)
- [orientation_locking_penalties.png](outputs/figures/orientation_locking_penalties.png)

The clearest figure is [orientation_locking_alpha_geometry.png](outputs/figures/orientation_locking_alpha_geometry.png), because it shows the split immediately:

- small or modest gains in the full regimes
- strong deterioration in the partial and sparse regimes

## Artifacts

Data:

- [orientation_locking_summary.json](outputs/orientation_locking_summary.json)
- [orientation_locking_summary.csv](outputs/orientation_locking_summary.csv)
- [orientation_locking_penalties.csv](outputs/orientation_locking_penalties.csv)
- [orientation_locking_trials.csv](outputs/orientation_locking_trials.csv)

Code:

- [run_orientation_locking_experiment.py](run.py#L1)
