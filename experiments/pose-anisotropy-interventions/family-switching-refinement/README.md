# Family-Switching Refinement Experiment

## Purpose

This experiment tests the stronger intervention suggested by the last failure pattern.

The previous candidate-conditioned method already used multiple top seeds, but it still held geometry fixed inside each seed family and only searched locally over `alpha`.

The sharper question here is:

> if the sparse-full moderate failure slice is really being driven by wrong-family selection or by geometry-level ambiguity upstream of local alpha cleanup, does a true multi-seed family-switching refinement help more than the alpha-only family search?

So this experiment keeps the same top marginalized seed shortlist and compares:

1. shift-marginalized baseline
2. candidate-conditioned alpha-only family search
3. geometry-plus-alpha family-switching local refinement
4. oracle-aligned bank prediction

The experiment script is [run_family_switching_refinement_experiment.py](run.py#L1).

## Pre-Benchmark Logic Audit

Before the benchmark, the new script was checked explicitly.

Code sanity:

- script compiled cleanly

Seed non-degradation:

- audit cases: `30`
- max refined-minus-seed score: `0.0`

That matters because the family-switching local search always includes the seed state, so it should never score worse than the seed under its own marginalized objective.

Nearby clean recovery:

- audit cases: `30`
- max geometry MAE after nearby refinement: `6.3393e-02`
- max alpha error after nearby refinement: `3.3580e-02`
- exact rotated-signature recovery fraction: `0.0`

This second audit is a nearby-seed sanity check, not an exact discrete-bank recovery test. The local family-switcher searches a small continuous neighborhood around the seed, so the right question is whether it stays close to truth under clean full observations when started near the right family. It does.

## Method

This is a deliberately focused pilot rather than a broad sweep.

Scope:

- observation regimes:
  - `sparse_full_noisy`
  - `sparse_partial_high_noise`
- anisotropy band:
  - `moderate`
- geometry-skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

Local family-switching refinement:

- keep the top `3` marginalized seeds
- for each seed, hold weights fixed
- locally refine:
  - `rho`
  - `t`
  - `h`
  - `alpha`
- use `2` refinement rounds
- use a `3`-point local grid per coordinate per round

Trials:

- `4` trials per cell
- `24` benchmark trials total

The goal here is not exhaustive coverage. It is to test the exact mechanism claim on the cells that matter most.

## Main Result

The result is not a blanket win, but it is a very strong clue.

> Geometry-plus-alpha family switching helps the sparse-full moderate branch overall and modestly improves the original sparse-full moderate mid-skew failure slice, but it hurts the sparse-partial moderate branch overall. The new method therefore does not solve the whole solver challenge. Instead, it splits the solver challenge by support type.

Focused summary:

- sparse-full family minus conditioned gain mean: `+0.4343`
- sparse-partial family minus conditioned gain mean: `-0.2351`
- sparse-full mid-skew extra gain over alpha-only search: `+0.1559`
- sparse-full mid-skew alpha improvement factor over alpha-only search: `1.0759`
- largest family-vs-conditioned alpha improvement factor: `7.9933`
- largest non-top1 winner fraction: `1.0`

Those numbers already say a lot:

- the sparse-full branch does benefit from geometry-level family switching
- the sparse-partial branch does not
- the original sparse-full mid-skew miss gets better, but only modestly

## Cell-Level Pattern

### Sparse full, moderate

This is where the new method helps.

- `low_skew`
  - marginalized alpha: `0.1152`
  - alpha-only family search: `0.1073`
  - geometry-plus-alpha family switch: `0.1043`
  - extra gain over alpha-only: `+0.0406`

- `mid_skew`
  - marginalized alpha: `0.1139`
  - alpha-only family search: `0.2247`
  - geometry-plus-alpha family switch: `0.2088`
  - extra gain over alpha-only: `+0.1559`
  - non-top1 winner fraction: `1.0`
  - mean winning seed rank: `2.5`

- `high_skew`
  - marginalized alpha: `0.1440`
  - alpha-only family search: `0.1376`
  - geometry-plus-alpha family switch: `0.0172`
  - extra gain over alpha-only: `+1.1064`
  - improvement factor over alpha-only: `7.9933`

The mid-skew cell is the original target slice, so the important point is:

> the family-switcher does improve the targeted sparse-full moderate mid-skew miss, but it does not repair it all the way back to the marginalized baseline, let alone to the oracle-aligned result.

That means the “wrong family” mechanism is real, but incomplete.

### Sparse partial, moderate

This is where the new method gives back ground.

- `low_skew`
  - alpha-only family search: `0.2011`
  - geometry-plus-alpha family switch: `0.2474`
  - extra gain over alpha-only: `-0.2075`

- `mid_skew`
  - alpha-only family search: `0.2022`
  - geometry-plus-alpha family switch: `0.2163`
  - extra gain over alpha-only: `-0.0697`

- `high_skew`
  - alpha-only family search: `0.1159`
  - geometry-plus-alpha family switch: `0.2124`
  - extra gain over alpha-only: `-0.4279`

So the same geometry movement that helps the sparse-full branch is not generally the right move for the sparse-partial branch.

That is the central new clue from this experiment.

## The Strongest Clue

The key pattern is not just who wins. It is what actually moves.

Across almost every cell:

- `family_delta_alpha_mean` is `0.0`
- but the geometry coordinates move materially:
  - `rho`
  - `t`
  - `h`

Examples:

- `sparse_full_noisy`, `moderate`, `mid_skew`
  - mean `delta_rho`: `0.0228`
  - mean `delta_t`: `0.2100`
  - mean `delta_h`: `0.1400`
  - mean `delta_alpha`: `0.0`

- `sparse_full_noisy`, `moderate`, `high_skew`
  - mean `delta_rho`: `0.0214`
  - mean `delta_t`: `0.1236`
  - mean `delta_h`: `0.1575`
  - mean `delta_alpha`: `0.0`

That means the new gains are coming mainly from geometry switching, not from better local alpha tuning.

This is a strong mechanism clue.

It suggests:

> in the sparse-full moderate branch, a meaningful part of the solver challenge really is upstream family choice or geometry-level ambiguity, not local alpha cleanup inside the already-chosen family.

## Interpretation

This result sharpens the solver challenge diagnosis.

What it establishes:

- the sparse-full failure slice is not just a local alpha-orbit alias problem
- geometry-level family switching can recover real signal there
- non-top1 seeds matter a lot in the sparse-full branch
- the active correction is usually geometric, not anisotropic

What it does not support:

- a single practical method that fixes every hard pose-free anisotropic cell

The strongest reading is:

> the pose-free anisotropic solver challenge has at least two support-dependent subproblems. Sparse-full moderate cells need better family selection or geometry switching before local alpha cleanup. Sparse-partial moderate cells benefit more from fixed-family local alpha disentanglement and can be harmed by freer geometry motion.

That is more informative than a simple “worked” or “failed” result.

## Note on the Oracle Comparison

The oracle-aligned comparator here is still a discrete-bank method with the true pose supplied.

The family-switching method is allowed to move locally off-bank in geometry and `alpha`.

Because of that, a few cells can show family-switching gain fractions above `1.0`. That does not mean the method exceeded a true continuous-information ceiling. It means the local family switcher beat the discrete oracle bank in that cell.

## Figure

Key figures:

- [family_switching_refinement_focus.png](outputs/figures/family_switching_refinement_focus.png)
- [family_switching_refinement_method_bars.png](outputs/figures/family_switching_refinement_method_bars.png)

The first figure is the most important one. It shows:

- extra oracle-gap capture from geometry-plus-alpha switching over alpha-only family search
- alpha improvement factor by cell
- how often the winning family came from a non-top1 seed

Taken together, those panels make the new clue visible: the sparse-full branch is genuinely family-selection-sensitive, while the sparse-partial branch is not helped by the same freedom.

