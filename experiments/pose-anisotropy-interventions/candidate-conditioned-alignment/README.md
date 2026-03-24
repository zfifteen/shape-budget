# Candidate-Conditioned Alignment Experiment

## Purpose

This experiment tests a sharper intervention suggested by the symmetry-orbit diagnosis.

The current pose-free anisotropic inverse already does per-candidate best-shift matching. The stronger idea tested here is narrower:

> maybe the practical bottleneck is not lack of candidate conditioning in the minimal sense, but lack of a candidate-conditioned local model of the `shift <-> alpha` tradeoff.

So instead of only choosing the best bank candidate under an observation-only pose rule, this experiment:

1. scores the bank with the existing shift-marginalized observation-only method
2. keeps the top candidate seeds
3. locally searches `alpha` inside each seed family while re-solving pose against that candidate family
4. chooses the best refined candidate

That makes this the first direct test of candidate-conditioned local `shift + alpha` disentanglement.

## Research Question

Can a richer candidate-conditioned local `shift + alpha` search recover more of the pose-free anisotropic `alpha` headroom than observation-only alignment, especially in the sparse moderate-anisotropy cells where orbit aliasing was expected to be strongest?

## Pre-Benchmark Logic Audit

Before the benchmark, the new script was checked explicitly.

Code sanity:

- script compiled cleanly

Candidate-family non-degradation:

- audit cases: `30`
- max refined-minus-seed score: `0.0`

That matters because the local search always includes the seed `alpha`, so the refined candidate family should never score worse than the seed family under its own objective.

Clean family audit:

- audit cases: `30`
- max alpha error after candidate-family clean refinement: `1.8108e-02`

This second audit is not an exact-recovery audit in the old bank-item sense, because the local alpha search uses continuous local grids rather than a requirement that the exact true `alpha` be present as a bank item. It is a sanity check that, with clean full observations and the correct geometry-weight family, the local search stays very close to the true anisotropy.

The experiment script is [run_candidate_conditioned_alignment_experiment.py](run.py#L1).

## Method

The experiment compares four methods on the same pose-free weighted anisotropic observations:

- hard best-shift baseline
- observation-only shift-marginalized baseline
- candidate-conditioned local `shift + alpha` search
- oracle alignment

The candidate-conditioned method works like this:

1. use the observation-only shift-marginalized score to rank the bank
2. keep the top `3` seeds
3. for each seed, hold geometry and weights fixed
4. search locally over `alpha` with:
   - a coarse radius `0.22`
   - a fine radius `0.07`
5. for each local `alpha`, recompute pose by the same marginalized-over-shifts rule
6. keep the best refined seed

So the new method is not a generic “stronger aligner.” It is a targeted candidate-family search over the local alpha-orbit alias direction.

## Balanced Parameter Sweep

Reference bank:

- anisotropy-aware bank size: `300`

Trials:

- `4` trials per cell
- `5` observation regimes
- `3` anisotropy-strength bins
- `3` geometry-skew bins

Total benchmark trials:

- `180`

Observation regimes:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

Latent bins:

- anisotropy strength from `|log(alpha)|`
  - `weak`
  - `moderate`
  - `strong`
- geometry skew from `|t|`
  - `low_skew`
  - `mid_skew`
  - `high_skew`

This balanced design matters because the question is specifically about the hard cells, not only the regime averages.

## Main Result

The result is mixed, but meaningfully so.

> Candidate-conditioned local `shift + alpha` search improves `alpha` recovery over observation-only shift marginalization in `4` of `5` regimes overall, with its strongest gains in the partial and sparse-partial regimes. But it does not uniformly rescue the targeted moderate sparse band. In particular, the sparse-full moderate mid-skew slice gets much worse, and the neighboring sparse-full moderate high-skew slice also fails sharply. So this intervention establishes the orbit-alias mechanism as one part of the bottleneck, not the whole bottleneck.

The summary file is [candidate_conditioned_alignment_summary.json](outputs/candidate_conditioned_alignment_summary.json).

Global regime-level summary:

- best conditioned-vs-marginalized alpha improvement ratio: `1.3740`
- worst conditioned-vs-marginalized alpha improvement ratio: `1.0250`

By regime:

- `full_clean`
  - marginalized alpha: `0.2110`
  - conditioned alpha: `0.2059`
  - oracle-gain capture: `0.0000 -> 0.0263`

- `full_noisy`
  - marginalized alpha: `0.1376`
  - conditioned alpha: `0.1310`
  - oracle-gain capture: `-0.0711 -> -0.0118`

- `partial_arc_noisy`
  - marginalized alpha: `0.1555`
  - conditioned alpha: `0.1197`
  - oracle-gain capture: `0.0930 -> 0.3511`

- `sparse_full_noisy`
  - marginalized alpha: `0.1588`
  - conditioned alpha: `0.1523`
  - oracle-gain capture: `0.1924 -> 0.2299`

- `sparse_partial_high_noise`
  - marginalized alpha: `0.2279`
  - conditioned alpha: `0.1658`
  - oracle-gain capture: `0.0425 -> 0.3902`

Those regime averages are encouraging.

The sparse-partial regime is the clearest success.

## The Hard-Cell Test

The stronger test was the one built into the mechanism claim:

- sparse support
- moderate anisotropy
- low-skew or mid-skew geometry

On that target band, the result is not uniformly positive.

Combined hard-cell summary:

- baseline alpha: `0.1209`
- marginalized alpha: `0.1325`
- conditioned alpha: `0.1408`
- oracle alpha: `0.0359`
- marginalized oracle-gain capture: `-0.2305`
- conditioned oracle-gain capture: `-0.4788`

If that were the whole mechanism, the intervention would look like a miss.

But the combined average hides a sharp split.

### Sparse partial, moderate, low-to-mid skew

This branch behaves the way the candidate-conditioned mechanism predicted.

- marginalized alpha: `0.2185`
- conditioned alpha: `0.1805`
- oracle-gain capture: `-0.9325 -> -0.4140`
- average extra oracle-gain capture: `+0.5185`

At the cell level:

- `sparse_partial_high_noise`, `moderate`, `low_skew`
  - extra oracle-gain capture: `+0.9173`
  - alpha improvement factor: `1.3686`

- `sparse_partial_high_noise`, `moderate`, `mid_skew`
  - extra oracle-gain capture: `+0.1198`
  - alpha improvement factor: `1.0715`

### Sparse full, moderate, low-to-mid skew

This branch splits.

- `sparse_full_noisy`, `moderate`, `low_skew`
  - extra oracle-gain capture: `+0.0516`
  - alpha improvement factor: `1.3826`

- `sparse_full_noisy`, `moderate`, `mid_skew`
  - extra oracle-gain capture: `-2.0818`
  - alpha improvement factor: `0.3925`

The neighboring `high_skew` moderate sparse-full cell also fails badly:

- `sparse_full_noisy`, `moderate`, `high_skew`
  - extra oracle-gain capture: `-2.1331`
  - alpha improvement factor: `0.4503`

So the candidate-conditioned intervention helps in `3` of the `4` sparse moderate low-to-mid-skew target cells, but it fails badly in one of them, and the sparse-full moderate family is worse than the original targeted slice alone suggested.

That is the central outcome of the experiment.

## Interpretation

This result strengthens the symmetry-orbit mechanism, but only in a partial way.

What it establishes:

- the bottleneck is not purely “generic inverse noise”
- a candidate-family local disentanglement step can recover real alpha headroom
- the gains are especially visible in the partial-support branch, including the hardest sparse-partial regime

What it does not establish:

- a simple claim that candidate-conditioned local `shift + alpha` search is enough to rescue the whole moderate sparse band

So the established reading is:

> candidate-conditioned local disentanglement does target a real part of the alpha bottleneck, but the orbit-proximity mechanism is incomplete. There is at least one additional difficulty inside the sparse-full moderate region, especially around the mid-skew slice and its nearby family members, that this intervention does not reach.

One plausible interpretation is:

- in partial-support regimes, the candidate family provides useful local orientation information and reduces alpha-orbit aliasing
- in sparse-full moderate mid-skew regimes, the ambiguity is happening at a more global family-selection level rather than inside a single candidate family

That is not proven by this experiment, but it fits the pattern.

## What This Establishes

This experiment does show:

- candidate-conditioned local `shift + alpha` search is a real lever, not a null intervention
- the lever is strongest in `partial_arc_noisy` and `sparse_partial_high_noise`
- the intervention can recover substantially more oracle headroom than observation-only marginalization in several hard cells
- the moderate sparse band is not uniform; some cells improve strongly and the sparse-full moderate family still contains sharply failing cells

This experiment does not address:

- a general solution to the pose-free alpha bottleneck
- that candidate-conditioned local search is the right final mechanism for sparse-full ambiguity
- whether the remaining failure is due mainly to wrong seed-family selection, insufficient local alpha search, or a deeper representation-level alias

## Why The Result Matters

This is a useful result even though it is not a clean universal win.

If the candidate-conditioned method had failed everywhere, the orbit-alias mechanism would have weakened sharply.

If it had succeeded everywhere, we would have had a near-immediate path forward.

Instead, the result is more informative:

- the hypothesis has teeth
- but it only explains part of the bottleneck

That means the next experiment should not abandon the orbit diagnosis.

It should sharpen it.

The most natural next move is:

- candidate-conditioned multi-seed or family-switching refinement

because the current method holds geometry and weights fixed inside each seed family. The sparse-full moderate mid-skew failure is exactly the sort of case where that restriction is too rigid.

## Figures

- [candidate_conditioned_alignment_overview.png](outputs/figures/candidate_conditioned_alignment_overview.png)
- [candidate_conditioned_alignment_hard_cells.png](outputs/figures/candidate_conditioned_alignment_hard_cells.png)

The clearest figure is [candidate_conditioned_alignment_hard_cells.png](outputs/figures/candidate_conditioned_alignment_hard_cells.png), because it shows the key thing the regime averages hide:

- the sparse-partial moderate cells improve
- the sparse-full moderate mid/high cells can still fail badly

## Artifacts

Data:

- [candidate_conditioned_alignment_summary.json](outputs/candidate_conditioned_alignment_summary.json)
- [candidate_conditioned_alignment_summary.csv](outputs/candidate_conditioned_alignment_summary.csv)
- [candidate_conditioned_alignment_cells.csv](outputs/candidate_conditioned_alignment_cells.csv)
- [candidate_conditioned_alignment_trials.csv](outputs/candidate_conditioned_alignment_trials.csv)

Code:

- [run_candidate_conditioned_alignment_experiment.py](run.py#L1)
