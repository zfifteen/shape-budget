# Backbone Correction-Flux-Triggered Alpha Solver

## Purpose

This experiment is a bounded Layer 3 follow-up in the backbone-first solver program.

Layer 1 established a stable geometry backbone across fresh banks.
Layer 2 established a trustworthy gate for when point `alpha` recovery should even be attempted.
The first Layer 3 attempt showed that always-refine improves some gate-open trials and harms others.

This variant tests one narrow claim:

- should Layer 3 refine only when the post-anchor correction event is strong enough to be real?

## Research Question

If the existing Layer 3 anchored and refined candidates are held fixed, can a calibration-frozen correction-flux trigger beat the anchored output on holdout and confirmation while keeping most of the anchor stability?

## Layer Position

This is still `Layer 3` of the staged solver plan in [docs/SOLVER_CHALLENGES.md](../../../docs/SOLVER_CHALLENGES.md).

Layer target:

1. stable backbone recovery
2. extension-coordinate observability gate
3. conditional `alpha` recovery inside the gate-open region

Not attempted here:

4. full confirmation-stable solver policy across the whole focused slice

## Method

This experiment reuses the saved output bundle from [Backbone Conditional Alpha Solver](../backbone-conditional-alpha-solver/README.md) as the fixed candidate generator.
That is the cleanest form of this test because the trigger changes only the final Layer 3 selection rule.

The setup stays on the same focused slice:

- `alpha_strength_bin = moderate`
- conditions:
  - `sparse_full_noisy`
  - `sparse_partial_high_noise`
- geometry skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

The layer order is:

1. keep the validated Layer 2 gate exactly where it was:
   - metric: `mean_anchored_alpha_log_std`
   - threshold: `0.1890`
2. inside the gate-open region, reuse the same two Layer 3 candidates:
   - anchored output
   - always-refine output
3. compute trial-level correction flux from the five banks:
   - `F = mean_i |log(refined_alpha_i) - log(anchored_alpha_i)|`
4. choose one calibration-frozen threshold on the calibration gate-open trials only:
   - objective: minimize mean open-trial `alpha` output error
   - tie-break: prefer sparser switching
5. output:
   - refined if `F >= threshold`
   - anchored otherwise

The executable artifact is [run.py](run.py).

## Main Result

The trigger confirms the over-activation diagnosis, but it does not clear the stricter stop-go rule for this branch.

The summary file is [backbone_correction_flux_triggered_alpha_solver_summary.json](outputs/backbone_correction_flux_triggered_alpha_solver_summary.json).

Global result on the gate-open region:

- trial count: `72`
- point-output count: `53`
- point-output rate: `0.7361`
- calibration-frozen correction-flux threshold: `0.0738`
- trigger fire rate on gate-open trials: `0.1887`
- anchored ensemble `alpha` error: `0.1432`
- always-refine ensemble `alpha` error: `0.1364`
- flux-triggered ensemble `alpha` error: `0.1358`
- anchored ensemble bank log-span: `0.0684`
- always-refine ensemble bank log-span: `0.1738`
- flux-triggered ensemble bank log-span: `0.1031`

That is the core result.

The trigger improves on both earlier Layer 3 outputs overall.
It keeps most of the anchor stability while recovering part of the refinement gain.

## By Split

- calibration:
  - trigger fire rate: `0.2800`
  - anchored output error: `0.1382`
  - always-refine output error: `0.1316`
  - flux-triggered output error: `0.1281`

- holdout:
  - trigger fire rate: `0.0714`
  - best output error: `0.1794`
  - anchored output error: `0.1598`
  - always-refine output error: `0.1650`
  - flux-triggered output error: `0.1592`

- confirmation:
  - trigger fire rate: `0.1429`
  - best output error: `0.1266`
  - anchored output error: `0.1353`
  - always-refine output error: `0.1163`
  - flux-triggered output error: `0.1262`

The split read is precise:

- on holdout, the trigger beats both anchored and always-refine
- on confirmation, the trigger beats anchored but not always-refine
- overall, the trigger is the best average gate-open output among the three Layer 3 policies

## Stability Tradeoff

The new trigger keeps most of the anchor stability.

Open-trial bank log-span:

- anchored output: `0.0684`
- flux-triggered output: `0.1031`
- always-refine output: `0.1738`

So the trigger recovers a large part of the spread that always-refine had reopened.

## What The Trigger Fixed

The holdout blocker from the first Layer 3 attempt is suppressed cleanly.

The clearest example remains the holdout cell:

- `sparse_full_noisy`
- `moderate`
- `mid_skew`

On that cell:

- anchored output error: `0.1418`
- always-refine output error: `0.1697`
- flux-triggered output error: `0.1418`
- trigger fire rate: `0.0000`

That is exactly the behavior the core insight predicted.
The harmful weak-correction case no longer fires refinement.

## Interpretation

This experiment shows that the Layer 3 miss was not random.

The correction-flux signal is real:

- the trigger fires rarely
- it fixes the key holdout over-activation failure
- it preserves most of the anchored stability

But the result also gives a clean boundary for this branch.

The flux-only trigger does not beat always-refine on confirmation.
So it does not clear the stricter stop-go rule that this branch must beat both anchored and always-refine on fresh blocks.

## What This Establishes

This experiment does show:

- the correction-flux idea is a real Layer 3 control signal
- the first Layer 3 failure was genuinely an over-activation problem
- a sparse trigger can beat anchored on both holdout and confirmation
- a sparse trigger can beat always-refine on holdout while keeping much better bank stability

This experiment does not show:

- that flux-only triggering fully solves Layer 3
- that this branch has earned more Layer 3 patching under the precommitted stop rule
- that the solver is ready to advance to Layer 4

## Figures

- [backbone_correction_flux_triggered_alpha_solver_alpha_error.png](outputs/figures/backbone_correction_flux_triggered_alpha_solver_alpha_error.png)
- [backbone_correction_flux_triggered_alpha_solver_alpha_span.png](outputs/figures/backbone_correction_flux_triggered_alpha_solver_alpha_span.png)

The clearest figure is [backbone_correction_flux_triggered_alpha_solver_alpha_error.png](outputs/figures/backbone_correction_flux_triggered_alpha_solver_alpha_error.png), because it shows the exact layer-3 trade:

- holdout now prefers the sparse trigger
- confirmation still prefers always-refine
- the anchored baseline is no longer the best global compromise

## Artifacts

Data:

- [backbone_correction_flux_triggered_alpha_solver_bank_rows.csv](outputs/backbone_correction_flux_triggered_alpha_solver_bank_rows.csv)
- [backbone_correction_flux_triggered_alpha_solver_trials.csv](outputs/backbone_correction_flux_triggered_alpha_solver_trials.csv)
- [backbone_correction_flux_triggered_alpha_solver_split_summary.csv](outputs/backbone_correction_flux_triggered_alpha_solver_split_summary.csv)
- [backbone_correction_flux_triggered_alpha_solver_condition_summary.csv](outputs/backbone_correction_flux_triggered_alpha_solver_condition_summary.csv)
- [backbone_correction_flux_triggered_alpha_solver_cell_summary.csv](outputs/backbone_correction_flux_triggered_alpha_solver_cell_summary.csv)
- [backbone_correction_flux_triggered_alpha_solver_summary.json](outputs/backbone_correction_flux_triggered_alpha_solver_summary.json)

Code:

- [run.py](run.py)

## Outcome

This result is strong enough to validate the correction-event diagnosis.
It is not strong enough to justify more ad hoc Layer 3 elaboration under the branch stop rule.

The next move in this branch is now the ratio-based follow-up:

- [backbone correction-pressure-triggered alpha solver](../backbone-correction-pressure-triggered-alpha-solver/README.md)
