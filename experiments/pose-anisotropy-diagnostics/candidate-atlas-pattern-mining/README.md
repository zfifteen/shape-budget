# Candidate Atlas Pattern Mining

## Purpose

This experiment mines the instrumented candidate atlas for recurring cluster types and trial-level burden ratios.

The atlas instrumentation already showed that the current near-best family is large, structured, and branch-dependent.
This follow-up asks the next question:

- what recurring candidate modes are actually present
- which trial-level ratios extracted from those modes carry solver-relevant signal
- whether the first poison-candidate idea should move from single rows to cluster-level burden

## Research Question

If the candidate atlas is reduced to recurring cluster archetypes, does that reveal a simpler trial-level structure that can guide an informed bank and improve downstream solver decisions?

## Method

This experiment reads the existing atlas outputs from [Candidate Atlas Instrumentation](../candidate-atlas-instrumentation/README.md) and joins them to the current Layer 3 pressure-trigger outputs.

The first-pass archetype rules are empirical and explicit:

- `dominant_core`
  - `cluster_mass_layer1 >= 0.18`
  - `cluster_size >= 8`
- `broad_fan`
  - `cluster_alpha_span >= 0.35`
  - `cluster_geometry_span >= 0.12`
- `alpha_fan`
  - `cluster_alpha_span >= 0.25`
- `fringe_singleton`
  - `cluster_size <= 2`
  - `cluster_mass_layer1 <= 0.02`
- otherwise `compact_minor`

From those archetypes, the experiment builds trial-level burden metrics such as:

- `fan_vs_core`
- `useful_structure_ratio`
- `residual_shell_alpha_mass`

The executable artifact is [run.py](run.py).

## Main Result

The atlas is not just a large family.
It contains recurring archetypes, and the strongest new trial-level ratio is a competition between `broad_fan` mass and `dominant_core` mass.

The summary file is [candidate_atlas_pattern_summary.json](outputs/candidate_atlas_pattern_summary.json).

Global result:

- trial count: `72`
- gate-open trial count: `53`
- archetype counts:
  - `fringe_singleton`: `1551`
  - `compact_minor`: `1288`
  - `alpha_fan`: `1182`
  - `broad_fan`: `1036`
  - `dominant_core`: `383`

On gate-open trials:

- `fan_vs_core` vs improvement correlation: `0.2828`
- `fan_vs_core` vs correction-pressure correlation: `0.1803`
- `useful_structure_ratio` vs improvement correlation: `0.2797`
- `useful_structure_ratio` vs refined-bank-span correlation: `0.3657`

That is the core result.

The atlas is expressing a real structural competition:

- a stable core mass
- versus a broad fan of still-active alternative structure

## Archetype Read

Mean archetype shape:

| Archetype | Size | Mass | Alpha Span | Geometry Span |
| --- | ---: | ---: | ---: | ---: |
| `dominant_core` | `17.44` | `0.2476` | `0.4701` | `0.1959` |
| `broad_fan` | `9.94` | `0.0977` | `0.5357` | `0.1743` |
| `alpha_fan` | `4.14` | `0.0648` | `0.3938` | `0.0939` |
| `compact_minor` | `2.56` | `0.0595` | `0.0849` | `0.0574` |
| `fringe_singleton` | `1.19` | `0.0069` | `0.0184` | `0.0096` |

The most important split is:

- `broad_fan` is heavily concentrated in `sparse_partial_high_noise`
- `compact_minor` is much more common in `sparse_full_noisy`

That means the hard branch is not just “larger.”
It is organized around a stronger broad-fan structure.

## By Condition

The branch split is sharp.

- `sparse_full_noisy`
  - holdout mean `fan_vs_core`: `0.3824`
  - confirmation mean `fan_vs_core`: `0.4371`
  - holdout mean `useful_structure_ratio`: `0.8417`
  - confirmation mean `useful_structure_ratio`: `0.9825`

- `sparse_partial_high_noise`
  - holdout mean `fan_vs_core`: `0.4829`
  - confirmation mean `fan_vs_core`: `0.6612`
  - holdout mean `useful_structure_ratio`: `1.3192`
  - confirmation mean `useful_structure_ratio`: `1.6637`

The confirmation block matters most.

The hardest branch is carrying much more broad-fan mass relative to the dominant core on confirmation.

## Atlas-Only Trigger Probe

The strongest simple atlas-only trigger in this pass is:

- refine if `fan_vs_core >= threshold`

Calibration-frozen threshold:

- `0.4107`

Gate-open errors:

- calibration: `0.1278`
- holdout: `0.1587`
- confirmation: `0.1182`

This is not better than the current pressure trigger.
But it is close enough to matter.

The atlas alone is carrying a real part of the Layer 3 activation signal.

## Poison Heuristic Shift

The first candidate-level poison score did not tell us much:

- old candidate poison score vs improvement: `0.0245`
- old candidate poison score vs correction pressure: `-0.0009`

The better replacement in this pass is not a single bad candidate score.
It is a cluster-level residual-shell burden:

- `residual_shell_alpha_mass`

This measures the mass sitting in the higher-score shell inside alpha-wide clusters.

It performs materially better:

- `residual_shell_alpha_mass` vs improvement: `0.1394`
- `residual_shell_alpha_mass` vs correction pressure: `0.2664`

That is the important shift.

The atlas is saying:

- the harmful or unresolved structure is not well described as one poison candidate
- it is better described as residual shell mass in alpha-wide clusters

## Interpretation

This is the strongest evidence yet that the bank should be informed by cluster structure rather than by raw sample density.

The current random bank is surfacing:

- dominant cores
- broad fans
- alpha fans
- compact minors
- fringe singletons

The next informed bank should use that structure directly.

The most actionable design signal is:

- high `fan_vs_core` means the observation is carrying more broad active structure than stable core structure

That is exactly the kind of state variable a backbone-conditioned informed bank should react to.

## What This Establishes

This experiment does show:

- the candidate atlas contains recurring cluster archetypes
- `broad_fan` mass is a defining structural feature of the hard branch
- `fan_vs_core` is a real trial-level ratio with downstream solver relevance
- the first poison-candidate framing should move from single rows to cluster-level residual-shell burden

This experiment does not show:

- that the archetype thresholds are final
- that `fan_vs_core` should replace the current pressure trigger outright
- that the informed bank is already solved

## Figures

- [candidate_atlas_archetype_mass_by_condition.png](outputs/figures/candidate_atlas_archetype_mass_by_condition.png)
- [candidate_atlas_fan_vs_core_scatter.png](outputs/figures/candidate_atlas_fan_vs_core_scatter.png)

The clearest figure is [candidate_atlas_archetype_mass_by_condition.png](outputs/figures/candidate_atlas_archetype_mass_by_condition.png), because it shows the broad-fan expansion in the hard branch directly.

## Artifacts

Data:

- [candidate_atlas_archetype_rows.csv](outputs/candidate_atlas_archetype_rows.csv)
- [candidate_atlas_trial_patterns.csv](outputs/candidate_atlas_trial_patterns.csv)
- [candidate_atlas_pattern_split_summary.csv](outputs/candidate_atlas_pattern_split_summary.csv)
- [candidate_atlas_pattern_condition_summary.csv](outputs/candidate_atlas_pattern_condition_summary.csv)
- [candidate_atlas_fan_rule_rows.csv](outputs/candidate_atlas_fan_rule_rows.csv)
- [candidate_atlas_pattern_summary.json](outputs/candidate_atlas_pattern_summary.json)

Code:

- [run.py](run.py)

## Next Step

The next informed-bank pass should not sample globally.

It should use the atlas pattern state directly:

1. preserve `dominant_core` coverage
2. allocate extra density along `broad_fan` structure
3. treat residual shell alpha mass as unresolved burden, not as a one-row poison event
