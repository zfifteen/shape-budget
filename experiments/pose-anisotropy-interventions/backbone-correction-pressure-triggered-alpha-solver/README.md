# Backbone Correction-Pressure-Triggered Alpha Solver

## Purpose

This experiment is a bounded Layer 3 follow-up in the backbone-first solver program.

The earlier [Backbone Correction-Flux-Triggered Alpha Solver](../backbone-correction-flux-triggered-alpha-solver/README.md) showed that raw flux is a real signal, but also suggested that flux alone is too primitive.

This variant tests one simpler claim:

- the higher Layer 3 activation is governed by a hidden ratio, not by raw flux alone

## Research Question

If the existing Layer 3 anchored and refined candidates are held fixed, can a simple correction-pressure ratio beat the raw-flux trigger on both fresh blocks?

## Layer Position

This is still `Layer 3` of the staged solver plan in [docs/SOLVER_CHALLENGES.md](../../../docs/SOLVER_CHALLENGES.md).

Layer target:

1. stable backbone recovery
2. extension-coordinate observability gate
3. conditional `alpha` recovery inside the gate-open region

Not attempted here:

4. full confirmation-stable solver policy across the whole focused slice

## Method

This experiment reuses the cached output bundle from [Backbone Correction-Flux-Triggered Alpha Solver](../backbone-correction-flux-triggered-alpha-solver/README.md).
That keeps the candidate generator fixed and changes only the final Layer 3 control quantity.

The pressure metric is:

- `pressure = (correction_flux * correction_sign_majority) / mean_anchored_alpha_log_std`

where:

- `correction_flux` is the mean absolute anchored-to-refined bank move
- `correction_sign_majority` is the fraction of banks agreeing on the dominant correction direction
- `mean_anchored_alpha_log_std` is the Layer 2 gate uncertainty already measured by the anchored posterior

The trigger rule is:

- refine if `pressure >= threshold`
- otherwise keep the anchored answer

The threshold is selected on calibration gate-open trials only, with the same objective as the raw-flux variant:

- minimize mean open-trial `alpha` output error
- tie-break toward sparser switching

The executable artifact is [run.py](run.py).

## Main Result

The pressure ratio is materially stronger than raw flux.

The summary file is [backbone_correction_pressure_triggered_alpha_solver_summary.json](outputs/backbone_correction_pressure_triggered_alpha_solver_summary.json).

Global result on the gate-open region:

- trial count: `72`
- point-output count: `53`
- point-output rate: `0.7361`
- calibration-frozen pressure threshold: `0.2427`
- trigger fire rate on gate-open trials: `0.5849`
- anchored ensemble `alpha` error: `0.1432`
- always-refine ensemble `alpha` error: `0.1364`
- raw-flux-triggered ensemble `alpha` error: `0.1358`
- pressure-triggered ensemble `alpha` error: `0.1331`
- anchored ensemble bank log-span: `0.0684`
- raw-flux-triggered bank log-span: `0.1031`
- pressure-triggered bank log-span: `0.1408`
- always-refine bank log-span: `0.1738`

That is the core result.

The pressure ratio is now the strongest Layer 3 trigger tested so far.
It improves on the raw-flux trigger on both fresh blocks and overall.

## By Split

- calibration:
  - trigger fire rate: `0.6000`
  - raw-flux-triggered output error: `0.1281`
  - pressure-triggered output error: `0.1279`

- holdout:
  - trigger fire rate: `0.6429`
  - anchored output error: `0.1598`
  - always-refine output error: `0.1650`
  - raw-flux-triggered output error: `0.1592`
  - pressure-triggered output error: `0.1584`

- confirmation:
  - trigger fire rate: `0.5000`
  - anchored output error: `0.1353`
  - always-refine output error: `0.1163`
  - raw-flux-triggered output error: `0.1262`
  - pressure-triggered output error: `0.1172`

The split read is precise:

- on holdout, the pressure ratio is the best of all tested Layer 3 policies
- on confirmation, it nearly matches always-refine and is much stronger than the raw-flux trigger
- on both fresh blocks, it beats the raw-flux trigger

## Stability Tradeoff

The pressure ratio spends more correction capacity than raw flux, but still stays meaningfully below always-refine.

Open-trial bank log-span:

- anchored output: `0.0684`
- raw-flux-triggered output: `0.1031`
- pressure-triggered output: `0.1408`
- always-refine output: `0.1738`

So the new metric does pay some stability to recover the missed confirmation gains, but not all the way back to always-refine.

## Interpretation

This experiment strengthens the hidden-ratio diagnosis.

Raw flux was real, but incomplete.
The more natural control quantity is closer to:

- correction size
- times directional agreement
- scaled by the anchored gate uncertainty

That simple ratio changes the Layer 3 behavior in exactly the direction we wanted:

- more active than the raw-flux trigger
- less reckless than always-refine
- better on holdout
- much better on confirmation

## What This Establishes

This experiment does show:

- a simple ratio-governed trigger is stronger than raw flux
- directional coherence matters
- the Layer 3 control quantity is not just move size
- the hidden-ratio framing is consistent with the current data

This experiment does not show:

- that Layer 3 is fully solved
- that the branch has already cleared the strict fresh-block stop rule
- that the solver is ready to advance to Layer 4

## Figures

- [backbone_correction_pressure_triggered_alpha_solver_alpha_error.png](outputs/figures/backbone_correction_pressure_triggered_alpha_solver_alpha_error.png)
- [backbone_correction_pressure_triggered_alpha_solver_alpha_span.png](outputs/figures/backbone_correction_pressure_triggered_alpha_solver_alpha_span.png)

The clearest figure is [backbone_correction_pressure_triggered_alpha_solver_alpha_error.png](outputs/figures/backbone_correction_pressure_triggered_alpha_solver_alpha_error.png), because it shows the ratio-based trigger sitting between raw flux and always-refine in exactly the way the hidden-ratio idea predicts.

## Artifacts

Data:

- [backbone_correction_pressure_triggered_alpha_solver_trials.csv](outputs/backbone_correction_pressure_triggered_alpha_solver_trials.csv)
- [backbone_correction_pressure_triggered_alpha_solver_split_summary.csv](outputs/backbone_correction_pressure_triggered_alpha_solver_split_summary.csv)
- [backbone_correction_pressure_triggered_alpha_solver_condition_summary.csv](outputs/backbone_correction_pressure_triggered_alpha_solver_condition_summary.csv)
- [backbone_correction_pressure_triggered_alpha_solver_cell_summary.csv](outputs/backbone_correction_pressure_triggered_alpha_solver_cell_summary.csv)
- [backbone_correction_pressure_triggered_alpha_solver_summary.json](outputs/backbone_correction_pressure_triggered_alpha_solver_summary.json)

Code:

- [run.py](run.py)

## Outcome

This result is significant.

The correction-event branch did not collapse into ad hoc patching.
It simplified into a cleaner ratio law that beats the raw-flux trigger on both fresh blocks.
