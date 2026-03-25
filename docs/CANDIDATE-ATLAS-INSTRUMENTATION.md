# Candidate Atlas Instrumentation

The next solver improvement depends on candidate visibility.

The repo already has enough evidence to justify a backbone-first solver.
The current missing piece is candidate observability inside each bank.

This spec defines the minimum instrumentation needed to turn the current near-best family into a usable candidate atlas.

## Purpose

The instrumentation must answer five questions directly.

1. Which candidate attributes survive scoring repeatedly?
2. Which candidates are near-duplicates and which are genuinely different modes?
3. Which candidates pull the Layer 1 consensus in useful directions and which poison it?
4. Which directions of variation are real ambiguity directions?
5. What structure should an informed bank sample more densely, less densely, or preserve by quota?

## Scope

The first pass should stay on the current focused slice and reuse the same observation blocks:

- `sparse_full_noisy`
- `sparse_partial_high_noise`
- moderate anisotropy
- `low_skew`, `mid_skew`, `high_skew`
- calibration, holdout, and confirmation blocks
- the current five-bank setup

This keeps the atlas directly comparable to the existing Layer 1, Layer 2, and Layer 3 artifacts.

## Capture Policy

The instrumentation should not log only the final winner.

For each trial and bank, capture:

- all near-best family survivors inside the current score band
- plus a small frontier shell outside the band, such as the next top `16` candidates by score

This gives visibility into:

- the active family
- the edge of the family
- what almost survived but did not

Each captured row should carry a tier label:

- `band`
- `frontier`

## Output Files

The first instrumentation pass should write these files.

### 1. `candidate_atlas_rows.csv`

One row per captured candidate per trial per bank.

This is the main raw table.

### 2. `candidate_atlas_cluster_rows.csv`

One row per candidate cluster per trial per bank.

This table summarizes discovered local modes.

### 3. `candidate_atlas_trial_summary.csv`

One row per trial.

This table aggregates candidate-family structure across banks.

### 4. `candidate_atlas_summary.json`

Top-level configuration and summary metrics.

This should record:

- observation blocks
- bank seeds
- bank size
- capture rules
- clustering settings
- count summaries

## Required Candidate Columns

The raw candidate table should contain at least the following columns.

### Identity

- `split`
- `observation_seed`
- `condition`
- `geometry_skew_bin`
- `bank_seed`
- `candidate_index`
- `capture_tier`
- `rank_by_score`

### Score And Shift

- `marginalized_score`
- `score_gap_from_best`
- `best_shift`
- `best_shift_score_gap`

If full shift-profile logging is too heavy for the first pass, store:

- the best shift
- the score gap between the best and second-best shifts

### Latent Parameters

- `rho`
- `t`
- `h`
- `w1`
- `w2`
- `w3`
- `alpha`
- `log_alpha`

### Canonical Invariants

- `rho12`
- `rho13`
- `rho23`
- `geometry_vector_norm`
- `weight_entropy`

### Family Relation

- `in_band_flag`
- `consensus_weight_layer1`
- `anchored_weight_layer2`
- `distance_to_best_geometry`
- `distance_to_consensus_geometry`
- `distance_to_backbone_geometry`
- `distance_to_anchored_alpha`

These are the columns that let us see whether a candidate is:

- near the winner
- near the backbone
- near the anchored Layer 2 state
- or only surviving by score despite pulling away structurally

### Local Structure

- `nearest_neighbor_geometry_distance`
- `nearest_neighbor_alpha_distance`
- `local_density_geometry`
- `local_density_joint`
- `cluster_id`
- `cluster_rank_by_mass`
- `cluster_size`

### Influence

- `pull_toward_consensus`
- `pull_away_from_consensus`
- `pull_toward_anchored_alpha`
- `pull_away_from_anchored_alpha`

The first implementation can define these as signed projections on the line from:

- best candidate to consensus geometry
- best candidate to anchored `alpha`

The exact formula can be simple at first.
What matters is that the table records directional influence, not only static distance.

## Required Cluster Columns

Each cluster row should contain at least:

- `split`
- `observation_seed`
- `condition`
- `geometry_skew_bin`
- `bank_seed`
- `cluster_id`
- `cluster_size`
- `cluster_mass_layer1`
- `cluster_mass_layer2`
- `cluster_best_score`
- `cluster_mean_score_gap`
- `cluster_center_rho12`
- `cluster_center_rho13`
- `cluster_center_rho23`
- `cluster_center_alpha`
- `cluster_geometry_span`
- `cluster_alpha_span`

## Required Trial Summary Columns

Each trial summary row should contain at least:

- `split`
- `observation_seed`
- `condition`
- `geometry_skew_bin`
- `mean_band_candidate_count`
- `mean_frontier_candidate_count`
- `mean_cluster_count`
- `mean_cluster_entropy`
- `mean_geometry_span_norm_set`
- `consensus_geometry_bank_span_norm`
- `compression_load`
- `mean_alpha_log_span_set`
- `mean_ambiguity_ratio`
- `mean_poison_pull`
- `mean_supportive_pull`
- `mode_persistence_rate`

`compression_load` should be written explicitly:

- `compression_load = mean_geometry_span_norm_set / consensus_geometry_bank_span_norm`

## Clustering Rule

The first pass does not need a perfect clustering method.

Use a simple geometry-first clustering rule on the captured family, such as:

- cluster in canonical geometry space
- then record alpha spread inside each geometry cluster

This is enough to detect:

- duplicate clouds
- multiple geometry modes
- alpha-only ambiguity within one geometry mode

## Poison-Candidate Heuristic

The first pass should explicitly tag possible consensus-poisoning candidates.

A candidate should be flagged as a provisional poison candidate when all are true:

- it survives the score frontier
- it has non-trivial consensus weight
- it sits structurally far from the final consensus
- its direction of pull increases cross-bank disagreement or downstream correction demand

This is only a diagnostic label.
It is not a training target.

## Instrumentation Hook Points

The current stack already has clean hook points.

### Hook A. After bank scoring

Capture:

- score
- rank
- shift information
- top shell and band survivors

### Hook B. After Layer 1 consensus weighting

Capture:

- Layer 1 consensus weight
- distance to consensus geometry
- cluster structure
- compression-load ingredients

### Hook C. After Layer 2 anchoring

Capture:

- anchored weight
- distance to anchored `alpha`
- whether the candidate remains aligned with the anchored posterior

### Hook D. Optional Layer 3 posthoc labels

For diagnostic joins only, attach:

- whether refined beat anchored
- correction flux
- correction pressure
- correction transmission

This should be posthoc metadata, not used during capture-time selection.

## What This Instrumentation Must Enable

Once the atlas exists, the repo should be able to answer concrete bank-design questions such as:

- Do the same candidate clusters recur across bank seeds?
- Are the hard cases dominated by duplicate families or by multiple real modes?
- Which candidate attributes predict consensus poisoning?
- Which ambiguity directions deserve dense informed-bank sampling?
- Where should the informed bank keep quota diversity instead of raw density?

## Acceptance Criteria

The instrumentation pass is useful only if it produces enough visibility to support an informed bank design.

The first pass counts as successful if it can show:

- the main recurring cluster types
- the recurring poison-candidate patterns
- the main ambiguity directions
- the main compression-load regimes

The first completed follow-up mining pass now lives at [Candidate Atlas Pattern Mining](../experiments/pose-anisotropy-diagnostics/candidate-atlas-pattern-mining/README.md).

## Immediate Build Rule

Do not wait for a perfect theory of candidate structure.

The first pass should be simple, explicit, and lossless enough to preserve the candidate family for later analysis.

The output should be rich enough that future selection rules can be derived from the logged attributes instead of guessed in advance.

The next solver-facing use of the atlas is not another blind heuristic. It is an informed-bank design that reacts to recurring cluster archetypes, `fan_vs_core`, and residual-shell burden.
