# Backbone Observability Gate On Top Of The Informed Bank

This experiment asks the next solver-stack question after the Layer 1 informed-bank win:

- once Layer 1 produces a better candidate family, what happens to the Layer 2 `alpha` observability gate?

The comparison stays on the hard branch:

- `sparse_partial_high_noise`

And it compares three views:

- `legacy_random_5seed`: the original Layer 2 hard-branch result
- `one_shot_random`: the same 3-seed bank regime used by the informed-bank atlas
- `persistent_mode_informed`: the informed-bank Layer 1 output

## Design

This is a derived experiment, not a full rerun of the expensive bank build.

The script reconstructs Layer 2 trial rows directly from the informed-bank candidate atlas:

- [persistent_mode_bank_candidate_atlas_rows.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-diagnostics/persistent-mode-bank-candidate-atlas/outputs/persistent_mode_bank_candidate_atlas_rows.csv)
- [persistent_mode_bank_candidate_atlas_bank_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-diagnostics/persistent-mode-bank-candidate-atlas/outputs/persistent_mode_bank_candidate_atlas_bank_summary.csv)

For each trial it recomputes:

- anchored `alpha` mean per bank from `anchored_weight_layer2`
- anchored cross-bank `alpha` span
- best-candidate and anchored `alpha` error
- the Layer 2 recoverable / unrecoverable label

Then it compares those rows against the legacy Layer 2 hard-branch outputs:

- [backbone_observability_gate_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate/outputs/backbone_observability_gate_trials.csv)

Two views matter:

1. `Frozen legacy gate`
Use the old calibration-frozen Layer 2 rule exactly as-is:

- metric: `mean_anchored_alpha_log_std`
- threshold: `0.1890273561`
- direction: higher means more unrecoverable

2. `Threshold-free separation`
Ask whether the Layer 2 signal still separates recoverable from unrecoverable trials after Layer 1 changed the candidate family.

## Main Result

Layer 1 improved the hard-branch target itself, but the old Layer 2 gate does not transfer cleanly.

What improved under the informed bank:

- holdout recoverable rate: `0.2222 -> 0.3333`
- confirmation recoverable rate: `0.3333 -> 0.5556`
- holdout anchored `alpha` error: `0.2013 -> 0.1931`
- confirmation anchored `alpha` error: `0.1643 -> 0.1536`

So the informed bank is helping the Layer 2 target.

But the frozen old Layer 2 rule gets worse:

- legacy random holdout balanced accuracy: `0.6429`
- legacy random confirmation balanced accuracy: `0.7500`
- informed-bank holdout balanced accuracy with the same frozen rule: `0.4167`
- informed-bank confirmation balanced accuracy with the same frozen rule: `0.2000`

So the old gate no longer matches the new Layer 1 regime.

## What Changed

The important part is not just threshold drift.

On the same hard branch:

- `one_shot_random` still behaves the old way:
  - larger `mean_anchored_alpha_log_std` means more unrecoverable
  - pooled best balanced accuracy reaches `0.8000`

- `persistent_mode_informed` changes the structure:
  - holdout still roughly follows the old direction
  - confirmation flips direction
  - lower `mean_anchored_alpha_log_std` now tracks unrecoverability better on confirmation

For the informed bank:

- pooled best balanced accuracy for `mean_anchored_alpha_log_std`: `0.7500`
- best direction: `<= threshold`
- pooled best threshold: `0.1899`

Split by split:

- informed holdout best `mean_anchored_alpha_log_std` rule:
  - balanced accuracy `0.6667`
  - direction `>=`
  - threshold `0.0547`

- informed confirmation best `mean_anchored_alpha_log_std` rule:
  - balanced accuracy `0.8000`
  - direction `<=`
  - threshold `0.1650`

That split-direction change is the real result.

## Interpretation

The informed bank made the Layer 1 family better, but it changed what Layer 2 uncertainty means.

Under the random bank:

- higher anchored `alpha` spread behaves like ordinary unrecoverability

Under the informed bank:

- holdout still mostly looks like that
- confirmation does not
- the informed bank can produce lower within-bank anchored spread while the trial is still unrecoverable

So Layer 2 now looks structurally mixed:

- the target improved
- the old gate stopped transferring
- the signal is not gone, but the confirmation regime changed orientation

## Solver Implication

This is enough to move the solver discussion forward.

The Layer 1 informed-bank win is real.

But Layer 2 is not “done but with a new threshold.”
It needs redesign on top of the new Layer 1 regime.

The next Layer 2 question is:

- what normalized observability ratio survives the informed-bank confirmation regime without flipping direction?

The current best answer from the follow-up ratio sweep is:

- `mean_candidate_count * mean_anchored_alpha_log_span / mean_anchored_alpha_log_std`

That candidate is now emitted directly in this experiment’s summary as the proposed informed-bank Layer 2 gate rule.

## Outputs

- [backbone_observability_gate_informed_bank_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/backbone_observability_gate_informed_bank_trials.csv)
- [backbone_observability_gate_informed_bank_split_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/backbone_observability_gate_informed_bank_split_summary.csv)
- [backbone_observability_gate_informed_bank_cell_summary.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/backbone_observability_gate_informed_bank_cell_summary.csv)
- [backbone_observability_gate_informed_bank_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/backbone_observability_gate_informed_bank_summary.json)
- [backbone_observability_gate_informed_bank_std_scatter.png](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/figures/backbone_observability_gate_informed_bank_std_scatter.png)
- [backbone_observability_gate_informed_bank_threshold_bars.png](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/figures/backbone_observability_gate_informed_bank_threshold_bars.png)
