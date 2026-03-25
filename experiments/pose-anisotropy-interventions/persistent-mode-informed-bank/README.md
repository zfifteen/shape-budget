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

The experiment compares three bank constructions on the current focused moderate-anisotropy slice:

1. `one_shot_random`
   - the current baseline
   - one random bank of size `300`

2. `scout_random_fill`
   - a two-stage control
   - a scout bank of size `120`
   - carry over `72` scout representatives
   - fill the rest of the final bank randomly

3. `persistent_mode_informed`
   - the informed-bank candidate
   - a scout bank of size `120`
   - preserve `72` carryover anchors chosen across scout clusters
   - add `192` local proposals around scout modes
   - keep `36` random exploration entries

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

The primary fresh-block checks are:

- holdout
- confirmation

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
