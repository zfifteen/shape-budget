# Persistent-Mode Informed Bank

## Purpose

This experiment is the first direct informed-bank test for the backbone-first solver program.

The current atlas work established two facts:

- the hard branch carries persistent candidate modes rather than random clutter
- compression load is materially higher there

The question here is narrower:

- can a scout-conditioned, mode-aware bank reduce Layer 1 compression load
- without breaking backbone recovery

## Design

The experiment compares two bank constructions on the hard branch of the current focused moderate-anisotropy slice:

- `sparse_partial_high_noise`

1. `one_shot_random`
   - the current baseline
   - one random bank of size `300`

2. `persistent_mode_informed`
   - the informed-bank candidate
   - a scout bank of size `120`
   - preserve `72` carryover anchors chosen across scout clusters
   - add `192` local proposals around scout modes
   - keep `36` random exploration entries

This first pass uses `3` independent bank seeds so it can run as a focused pilot rather than a full production benchmark.

The informed stage uses only the scout-family structure:

- geometry clusters
- cluster mass
- cluster size
- cluster geometry span
- cluster alpha span

The cluster archetypes are the same first-pass vocabulary from the atlas mining work:

- `dominant_core`
- `broad_fan`
- `alpha_fan`
- `compact_minor`
- `fringe_singleton`

## Acceptance Rule

The first informed bank counts as successful only if it does both:

- cuts hard-branch compression load by at least `20%` relative to `one_shot_random`
- preserves hard-branch backbone geometry MAE within `5%` of `one_shot_random`

The hard branch here is:

- `sparse_partial_high_noise`

The experiment uses only the fresh evaluation blocks because this first bank design does not tune any calibration-side thresholds.

The primary checks are:

- holdout
- confirmation

## Result

The first pilot did not clear the precommitted `20%` rule on both fresh blocks, but it did produce a strong confirmation-block win while preserving backbone quality.

Fresh-block compression load:

- holdout
  - `one_shot_random`: `23.3228`
  - `persistent_mode_informed`: `21.4700`
  - reduction: `7.94%`

- confirmation
  - `one_shot_random`: `31.7663`
  - `persistent_mode_informed`: `16.1310`
  - reduction: `49.22%`

Fresh-block backbone geometry MAE:

- holdout
  - `one_shot_random`: `0.07409`
  - `persistent_mode_informed`: `0.07442`
  - ratio: `1.0043`

- confirmation
  - `one_shot_random`: `0.07573`
  - `persistent_mode_informed`: `0.07361`
  - ratio: `0.9721`

So the pilot result is:

- `20%` target: failed because holdout stayed below threshold
- geometry-preservation rule: passed
- confirmation behavior: very strong

This means the informed-bank idea is real, but the first allocation rule is not yet stable enough across fresh blocks to count as solved.

## Artifacts

Code:

- [run.py](run.py)

Outputs:

- [persistent_mode_informed_bank_summary.json](outputs/persistent_mode_informed_bank_summary.json)
- [persistent_mode_informed_bank_trial_summary.csv](outputs/persistent_mode_informed_bank_trial_summary.csv)
- [persistent_mode_informed_bank_split_summary.csv](outputs/persistent_mode_informed_bank_split_summary.csv)
- [persistent_mode_informed_bank_condition_summary.csv](outputs/persistent_mode_informed_bank_condition_summary.csv)
- [persistent_mode_informed_bank_bank_rows.csv](outputs/persistent_mode_informed_bank_bank_rows.csv)

Figures:

- [persistent_mode_informed_bank_compression_load.png](outputs/figures/persistent_mode_informed_bank_compression_load.png)
- [persistent_mode_informed_bank_geometry_mae.png](outputs/figures/persistent_mode_informed_bank_geometry_mae.png)
