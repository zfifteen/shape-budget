# Persistent-Mode Bank Candidate Atlas

## Purpose

This experiment adds fine-grained candidate instrumentation to the first informed-bank pilot.

The repo already has:

- a candidate atlas for the earlier random-bank solver family
- a first informed-bank pilot with strong confirmation behavior

The missing piece was direct candidate visibility inside the informed bank itself.

This experiment fills that gap on the hard branch:

- `sparse_partial_high_noise`
- holdout
- confirmation

## What It Captures

For both:

- `one_shot_random`
- `persistent_mode_informed`

the experiment logs:

- candidate rows
- cluster rows
- per-bank summaries
- per-trial summaries
- per-cell summaries

It preserves the earlier candidate-atlas fields and adds informed-bank provenance:

- `candidate_source`
- `source_cluster_id`
- `source_cluster_rank`
- `source_archetype`

The informed-bank provenance distinguishes:

- `carryover`
- `local_expansion`
- `local_fill`
- `exploration`

## Research Questions

This instrumentation is meant to answer:

- what the informed bank actually changes inside the surviving candidate family
- whether it reduces broad-fan burden or just moves it around
- how holdout and confirmation differ at candidate level under the new bank
- which candidate-source types are helping or poisoning the Layer 1 family

## Main Result

The informed bank changes the candidate family in a very specific way.

It does not reduce the surviving family by brute-force pruning.
It produces a larger band family, but that larger family is more structurally organized.

Across the full hard-branch run:

- `one_shot_random`
  - mean band candidate count: `90.41`
  - mean compression load: `27.54`
  - mean mode persistence rate: `0.8664`

- `persistent_mode_informed`
  - mean band candidate count: `209.93`
  - mean compression load: `18.80`
  - mean mode persistence rate: `0.8411`

The key point is:

- more surviving candidates
- fewer geometry clusters
- lower compression load

So the informed bank is not simply narrowing the family.
It is making the family more coherent.

The confirmation block is the clearest case:

- random confirmation compression load: `31.77`
- informed confirmation compression load: `16.13`

By cell, the strongest old failure remained confirmation `mid_skew`, and that is exactly where the informed bank cuts load the most:

- random confirmation `mid_skew`: `39.20`
- informed confirmation `mid_skew`: `14.47`

The provenance fields are also useful already.
Among informed-bank captured candidates, most band survivors come from local expansion rather than exploration:

- `local_expansion`: `7980`
- `local_fill`: `762`
- `carryover`: `2026`
- `exploration`: `568`

That means the informed bank is winning mainly by spending density around discovered modes, not by random reserve.

## Candidate Classification for Bank Decisions

The rows and cluster data contain clear signals for distinguishing candidate types.

- Reinforce mode: pull_toward_consensus > 0.25, in_band_flag = 1, poison_flag = 0
- Clutter: high local_density_geometry > 30 with low pull_toward_consensus < 0.1
- Poison: poison_candidate_flag = 1 or pull_away_from_consensus > 0.2
- Dense sample: rank_by_score <= 5 and candidate_source in local_expansion or carryover
- Avoid: high distance_to_anchored_alpha > 0.08 or fringe archetype

These rules can be used to set quotas in the bank: increase local_expansion budget for reinforce and dense sample candidates, reduce for avoid and clutter.

The informed bank already supplies more reinforce candidates with tighter control parameters.

## Artifacts

Code:

- [run.py](run.py)

Outputs:

- [persistent_mode_bank_candidate_atlas_rows.csv](outputs/persistent_mode_bank_candidate_atlas_rows.csv)
- [persistent_mode_bank_candidate_atlas_cluster_rows.csv](outputs/persistent_mode_bank_candidate_atlas_cluster_rows.csv)
- [persistent_mode_bank_candidate_atlas_bank_summary.csv](outputs/persistent_mode_bank_candidate_atlas_bank_summary.csv)
- [persistent_mode_bank_candidate_atlas_trial_summary.csv](outputs/persistent_mode_bank_candidate_atlas_trial_summary.csv)
- [persistent_mode_bank_candidate_atlas_split_summary.csv](outputs/persistent_mode_bank_candidate_atlas_split_summary.csv)
- [persistent_mode_bank_candidate_atlas_cell_summary.csv](outputs/persistent_mode_bank_candidate_atlas_cell_summary.csv)
- [persistent_mode_bank_candidate_atlas_summary.json](outputs/persistent_mode_bank_candidate_atlas_summary.json)
