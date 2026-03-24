# Shift-Marginalized Local Refinement Experiment

## Purpose

This experiment follows directly from the shift-marginalized pose result.

That earlier artifact showed that:

- hard best-shift scoring was part of the pose-free `alpha` bottleneck
- soft shift-marginalization tightened the `alpha` envelope in every regime
- the gains were real but only moderate

That leaves a narrower next question:

> once pose is handled more softly, is the remaining `alpha` blur mostly a local-fit problem inside the chosen basin?

This experiment tests that question directly.

## Research Question

If we start from the shift-marginalized best bank candidate and then locally refine geometry plus `alpha` under the same marginalized score, does `alpha` become materially more recoverable?

## Pre-Benchmark Logic Audit

Before the benchmark, the script was checked in three ways.

First, it compiled cleanly.

Second, a clean full-observation sanity check confirmed that the current micro-pilot settings still behave correctly when the observation comes from the bank itself:

- in `10/10` clean full-observation bank trials, the exact seed candidate was recovered
- the local refiner never worsened its own marginalized objective
- max refined fit RMSE against the rotated clean boundary was `8.42e-4`

Third, a direct single-case sanity check with the current micro-pilot settings recovered the exact seed and slightly improved the marginalized score:

- seed index: `72`
- true index: `72`
- seed marginalized score: `2.9835e-4`
- refined marginalized score: `2.9835e-4`
- refined fit RMSE: `1.16e-5`

So the local loop is behaving consistently under the audited objective.

The experiment script is [run_shift_marginalized_local_refinement_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_shift_marginalized_local_refinement_experiment.py#L1).

## Method

This is a small extension of the shift-marginalized pose inverse.

For each trial:

1. score the anisotropy-aware reference bank with the soft shift-marginalized score
2. take the top bank seed
3. locally refine `rho`, `t`, `h`, and `alpha` around that seed under the same marginalized score
4. compare the refined result against the marginalized bank baseline

Important design choice:

- weights are held fixed during refinement

So this experiment is specifically about whether local cleanup of geometry plus `alpha` helps once pose has already been softened.

Important interpretation note:

- the refined ambiguity envelope is local, not global
- it is built from all deduplicated states explored around the chosen top seed
- so a tighter refined top-`10` `alpha` span means the selected basin was locally tightened
- it does not mean the full global ambiguity across multiple basins has disappeared

## Pilot Settings

This artifact is intentionally a micro-pilot so the method could be audited interactively before any larger run.

Reference bank:

- anisotropy-aware bank size: `300`

Trial count:

- `2` trials per observation regime

Refinement:

- top seeds refined per trial: `1`
- grid points per local axis sweep: `3`
- refinement rounds: `1`

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

## Main Result

The result is useful and fairly sharp.

> In this micro-pilot, shift-marginalized local refinement clearly tightens the selected local basin, and it sometimes improves fit or geometry, but it does not materially improve `alpha` recovery.

The summary file is [shift_marginalized_local_refinement_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/shift_marginalized_local_refinement_outputs/shift_marginalized_local_refinement_summary.json).

The cleanest summary is:

- baseline-over-refined mean `alpha` error ratio ranges from `0.8111` to `1.0000`
- baseline-over-refined mean top-`10` `alpha` span ratio ranges from `1.5669` to `4.4188`
- refined near-tie-and-alpha-diverse fraction ranges from `0.0` to `0.5`

So the local basin gets tighter almost everywhere, but `alpha` itself mostly does not improve.

## By Regime

- `full_clean`
  - alpha error: `0.4683 -> 0.4683`
  - alpha-span local top-`10`: `0.5266 -> 0.2000`
  - fit RMSE: `0.00764 -> 0.00471`
  - geometry MAE: `0.1265 -> 0.1362`

- `full_noisy`
  - alpha error: `0.0314 -> 0.0387`
  - alpha-span local top-`10`: `0.3256 -> 0.0737`
  - fit RMSE: `0.01330 -> 0.01094`
  - geometry MAE: `0.0931 -> 0.0789`

- `partial_arc_noisy`
  - alpha error: `0.0178 -> 0.0178`
  - alpha-span local top-`10`: `0.6851 -> 0.2000`
  - fit RMSE: `0.01261 -> 0.01471`
  - geometry MAE: `0.0451 -> 0.0326`
  - near-tie diverse fraction: `0.500 -> 0.000`

- `sparse_full_noisy`
  - alpha error: `0.0876 -> 0.0876`
  - alpha-span local top-`10`: `0.2350 -> 0.1500`
  - fit RMSE: `0.01208 -> 0.01559`
  - geometry MAE: `0.1439 -> 0.1766`
  - near-tie diverse fraction: `1.000 -> 0.000`

- `sparse_partial_high_noise`
  - alpha error: `0.0846 -> 0.0846`
  - alpha-span local top-`10`: `0.3507 -> 0.2000`
  - fit RMSE: `0.04800 -> 0.04801`
  - geometry MAE: `0.1423 -> 0.1249`
  - near-tie diverse fraction: `0.500 -> 0.500`

Two features stand out:

- the local envelope gets much tighter in every regime
- `alpha` itself is unchanged in four regimes and worse in one

That is the core read.

## Interpretation

This micro-pilot weakens one specific hope.

It suggests that the remaining pose-free anisotropic `alpha` bottleneck is not mainly a matter of cleaning up the chosen local basin.

In plain language:

- once the method commits to a basin, local continuous cleanup can make that basin look tidier
- but tidier local fit does not usually reveal a better `alpha`

That matters because it narrows the next move.

The result does not say local refinement is useless.

It does say:

- local refinement can improve fit
- local refinement can improve geometry in some regimes
- local refinement can shrink the local ambiguity cloud
- but local refinement alone is not enough to recover the missing `alpha` signal

So the remaining bottleneck looks more like a basin-selection or representation problem than a simple within-basin optimization problem.

## What This Establishes

This experiment does show:

- the audited local refinement loop behaves correctly under the marginalized objective
- local refinement can tighten the selected ambiguity basin substantially
- local refinement can improve fit or geometry without improving `alpha`
- the remaining `alpha` bottleneck is not solved by a one-seed one-round local cleanup

This experiment does not yet show:

- what happens with broader multi-seed or multi-round refinement
- whether joint refinement of weights would help
- whether a better representation can convert the tighter local basin into better `alpha` recovery
- whether the same conclusion survives a larger benchmark than this micro-pilot

## Figures

- [shift_marginalized_local_refinement_overview.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/shift_marginalized_local_refinement_outputs/figures/shift_marginalized_local_refinement_overview.png)
- [shift_marginalized_local_refinement_trial_scatter.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/shift_marginalized_local_refinement_outputs/figures/shift_marginalized_local_refinement_trial_scatter.png)

The clearest figure is [shift_marginalized_local_refinement_overview.png](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/shift_marginalized_local_refinement_outputs/figures/shift_marginalized_local_refinement_overview.png), because it separates two different stories:

- the local top-`10` basin usually tightens
- the actual `alpha` estimate usually does not improve

## Artifacts

Data:

- [shift_marginalized_local_refinement_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/shift_marginalized_local_refinement_outputs/shift_marginalized_local_refinement_summary.json)
- [shift_marginalized_local_refinement_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/shift_marginalized_local_refinement_outputs/shift_marginalized_local_refinement_summary.csv)
- [shift_marginalized_local_refinement_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/shift_marginalized_local_refinement_outputs/shift_marginalized_local_refinement_trials.csv)

Code:

- [run_shift_marginalized_local_refinement_experiment.py](/Users/velocityworks/IdeaProjects/shape-budget/concepts/shape-budget/run_shift_marginalized_local_refinement_experiment.py#L1)
