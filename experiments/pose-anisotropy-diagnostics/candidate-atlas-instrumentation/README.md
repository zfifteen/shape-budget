# Candidate Atlas Instrumentation

## Purpose

This experiment captures the internal candidate family structure of the current backbone-first solver stack.

The repo already established:

- a stable Layer 1 backbone
- a valid Layer 2 observability gate
- a hidden-ratio Layer 3 program

The next missing piece is candidate visibility.

This experiment does not change the solver.
It instruments the candidate family so the repo can see:

- which candidates survive scoring
- how they cluster
- how much compression was needed to recover the backbone
- what an informed bank should sample more densely, less densely, or preserve by quota

## Research Question

If the current Layer 1 and Layer 2 path is instrumented directly, does the near-best family expose recurring structure strong enough to justify an informed bank instead of a global random bank?

## Position

This is a diagnostic experiment for the next bank-design step described in:

- [Intelligent Bank Design](../../../docs/INTELLIGENT-BANK-DESIGN.md)
- [Candidate Atlas Instrumentation](../../../docs/CANDIDATE-ATLAS-INSTRUMENTATION.md)

It is not a new solver layer.

## Method

The experiment stays on the active focused slice:

- `alpha_strength_bin = moderate`
- conditions:
  - `sparse_full_noisy`
  - `sparse_partial_high_noise`
- geometry skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

Observation seeds reuse the same calibration, holdout, and confirmation blocks as the current solver ladder.

Each observation is scored against the current five-bank setup:

- bank seeds: `20260324`, `20260325`, `20260326`, `20260327`, `20260328`
- bank size: `300`

For each trial and bank, the instrumentation captures:

1. all near-best band survivors
2. a fixed frontier shell of the next `16` candidates by score
3. Layer 1 consensus weights
4. Layer 2 anchored weights
5. geometry-first candidate clusters
6. candidate influence metrics such as pull toward or away from the Layer 1 backbone

The executable artifact is [run.py](run.py).

## Main Result

The candidate family is not random clutter.
It has strong recurring structure, and the hard branch carries much larger internal families than the easier branch.

The summary file is [candidate_atlas_summary.json](outputs/candidate_atlas_summary.json).

Global result:

- trial count: `72`
- candidate rows: `27013`
- cluster rows: `5440`
- mean band candidate count: `59.04`
- mean frontier candidate count: `16.00`
- mean geometry-cluster count: `15.11`
- mean compression load: `10.82`
- mean mode persistence rate: `0.9353`
- compression-load vs mode-persistence correlation: `0.4802`

That is the core result.

The internal family is rich enough to justify an informed bank.
The hard branch is not just noisier.
It is carrying a larger, more persistent candidate atlas.

## By Split

- calibration:
  - mean band candidate count: `59.74`
  - mean cluster count: `15.08`
  - mean compression load: `10.46`
  - mean mode persistence rate: `0.9342`

- holdout:
  - mean band candidate count: `58.57`
  - mean cluster count: `15.24`
  - mean compression load: `11.37`
  - mean mode persistence rate: `0.9529`

- confirmation:
  - mean band candidate count: `58.10`
  - mean cluster count: `15.04`
  - mean compression load: `11.00`
  - mean mode persistence rate: `0.9197`

The atlas remains large and structured on the fresh blocks.

## By Condition

The main separation is by support branch.

- `sparse_full_noisy`
  - calibration mean band candidate count: `26.64`
  - holdout mean band candidate count: `27.38`
  - confirmation mean band candidate count: `20.38`
  - calibration mean compression load: `8.42`
  - holdout mean compression load: `7.86`
  - confirmation mean compression load: `7.18`
  - mean cluster count stays around `11.67` to `12.69`

- `sparse_partial_high_noise`
  - calibration mean band candidate count: `92.83`
  - holdout mean band candidate count: `89.76`
  - confirmation mean band candidate count: `95.82`
  - calibration mean compression load: `12.50`
  - holdout mean compression load: `14.87`
  - confirmation mean compression load: `14.83`
  - mean cluster count stays around `17.80` to `18.42`

This is the most important result in the file.

The hard sparse-partial branch is carrying:

- about three to four times as many band survivors
- about five to six more geometry clusters per trial
- roughly twice the Layer 1 compression load

That is exactly the kind of structure an informed bank should use.

## Interpretation

The current random bank is exposing real candidate structure, but it is not organizing that structure well.

The atlas says:

- the hard cases are not just single-candidate misses
- they contain large recurring families
- those families are clusterable
- those families remain persistent across bank seeds
- the branch split is visible directly in candidate-family size and compression load

This is strong support for the next design move:

- keep candidate visibility first-class
- then replace the global random bank with a backbone-conditioned informed bank

## Important Nuance

The first poison-candidate heuristic is still weak.

Mean poison pull is small across all splits and conditions, and it goes to zero in the sparse-partial branch under the first-pass rule.
That should not be read as proof that poisoning is absent.
It means the first heuristic is conservative and needs refinement.

The atlas still succeeded at its main job:

- candidate-family size
- clustering
- compression load
- mode persistence

all came through clearly.

## What This Establishes

This experiment does show:

- the current near-best family is large enough and structured enough to justify candidate-atlas instrumentation
- the hard branch carries far more internal candidate diversity than the easier branch
- compression load is visible directly from the instrumented family
- recurring candidate modes survive across bank seeds strongly enough to support informed-bank design

This experiment does not show:

- that the informed bank is already built
- that the first poison-candidate heuristic is finished
- that the full solver challenge is solved

## Figures

- [candidate_atlas_compression_load.png](outputs/figures/candidate_atlas_compression_load.png)
- [candidate_atlas_cluster_count.png](outputs/figures/candidate_atlas_cluster_count.png)

The clearest figure is [candidate_atlas_compression_load.png](outputs/figures/candidate_atlas_compression_load.png), because it shows the branch split directly:

- `sparse_partial_high_noise` carries much higher compression load than `sparse_full_noisy`

## Artifacts

Data:

- [candidate_atlas_rows.csv](outputs/candidate_atlas_rows.csv)
- [candidate_atlas_cluster_rows.csv](outputs/candidate_atlas_cluster_rows.csv)
- [candidate_atlas_trial_summary.csv](outputs/candidate_atlas_trial_summary.csv)
- [candidate_atlas_split_summary.csv](outputs/candidate_atlas_split_summary.csv)
- [candidate_atlas_condition_summary.csv](outputs/candidate_atlas_condition_summary.csv)
- [candidate_atlas_summary.json](outputs/candidate_atlas_summary.json)

Code:

- [run.py](run.py)

## Next Step

The next step should not be another random-bank solver tweak.

It should be:

1. inspect the atlas for recurring candidate modes and ambiguity directions
2. sharpen the poison-candidate heuristic
3. build the first backbone-conditioned informed bank from those patterns
