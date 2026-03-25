# Backbone Conditional Alpha Solver

## Purpose

This experiment is the third capability layer in the backbone-first solver program.

Layer 1 established a stable geometry backbone across fresh bank seeds.
Layer 2 established a trustworthy gate for when point `alpha` recovery should even be attempted.

Layer 3 asks the next question:

- once the backbone is fixed and the gate opens, can a dedicated conditional `alpha` solver improve the point estimate enough to count as a real new capability layer?

This is still not a full solver.
It is the first gated `alpha` recovery layer built on top of the validated backbone and gate.

## Research Question

If the solver recovers the backbone first and only emits a point `alpha` estimate inside the layer-2 gate-open region, can backbone-anchored local `alpha` refinement improve the gate-open point estimate on holdout and confirmation?

## Layer Position

This is `Layer 3` of the staged solver plan in [docs/SOLVER_CHALLENGES.md](../../../docs/SOLVER_CHALLENGES.md).

Layer target:

1. stable backbone recovery
2. extension-coordinate observability gate
3. conditional `alpha` recovery inside the gate-open region

Not attempted here:

4. full confirmation-stable solver policy across the whole focused slice

## Method

The experiment stays on the same focused slice:

- `alpha_strength_bin = moderate`
- conditions:
  - `sparse_full_noisy`
  - `sparse_partial_high_noise`
- geometry skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

Observation blocks reuse the same calibration, holdout, and confirmation seeds as the current solver ladder.

Each observation is scored against the same fixed five-bank ensemble:

- bank seeds: `20260324`, `20260325`, `20260326`, `20260327`, `20260328`
- bank size: `300`

The layer order is explicit:

1. recover a bank-local near-family geometry consensus in each bank
2. average those bank-local consensuses into a trial backbone
3. keep the Layer 2 gate exactly where it was validated:
   - metric: `mean_anchored_alpha_log_std`
   - calibration-frozen threshold: `0.1890`
4. only when the gate opens:
   - reweight each bank’s near-best family against the trial backbone
   - keep the top `3` backbone-consistent seeds
   - run candidate-conditioned local `alpha` search inside each seed family
   - combine the refined bank outputs into one ensemble point estimate
5. when the gate stays closed:
   - abstain

The emitted point estimate is the geometric-mean ensemble over the bank-wise refined `alpha` values.

## Main Result

Layer 3 is directionally useful, but it does not validate yet.

The summary file is [backbone_conditional_alpha_solver_summary.json](outputs/backbone_conditional_alpha_solver_summary.json).

Global result on the gate-open region:

- trial count: `72`
- point-output count: `53`
- point-output rate: `0.7361`
- gate precision: `0.5849`
- best-bank ensemble `alpha` error: `0.1471`
- anchored ensemble `alpha` error: `0.1432`
- conditional refined ensemble `alpha` error: `0.1364`

That is the positive part:

- the conditional refinement does improve the open-region point estimate overall
- it stays much more stable than the raw best-bank output

But the validation bar for this layer was stricter:

- beat the anchored point estimate on holdout
- beat it again on confirmation

This experiment clears the second half and misses the first.

## By Split

- calibration:
  - point-output rate: `0.6944`
  - gate precision: `0.6400`
  - anchored output error: `0.1382`
  - refined output error: `0.1316`

- holdout:
  - point-output rate: `0.7778`
  - gate precision: `0.5000`
  - best output error: `0.1794`
  - anchored output error: `0.1598`
  - refined output error: `0.1650`

- confirmation:
  - point-output rate: `0.7778`
  - gate precision: `0.5714`
  - best output error: `0.1266`
  - anchored output error: `0.1353`
  - refined output error: `0.1163`

So the central read is precise:

- the new layer beats both comparison outputs on confirmation
- it beats the raw best-bank output on holdout
- it does not beat the anchored point estimate on holdout

That means Layer 3 is promising, but not yet validated as a correct next capability layer.

## Stability Tradeoff

The new refinement improves error overall, but it partially reopens cross-bank spread.

Open-trial bank log-span:

- best-bank output: `0.2509`
- anchored output: `0.0684`
- refined output: `0.1738`

That is still better than the raw best-bank spread, but clearly worse than the anchored layer.

This is the current layer-3 trade:

- recover some `alpha` bias
- give back some bank stability

## Holdout Failure Mode

The holdout miss is not diffuse.

The clearest blocking cell is:

- `sparse_full_noisy`
- `moderate`
- `mid_skew`

On that holdout cell:

- best output error: `0.0695`
- anchored output error: `0.1418`
- refined output error: `0.1697`

So the gated local search is still pushing the solver the wrong way in one of the important sparse-full cells.

That is why this layer does not validate yet.

## Interpretation

This experiment does show something real.

The backbone-first structure is still the right direction:

- a gated conditional solve can improve the point estimate
- the improvement is strongest on confirmation
- the method is not just reproducing the anchored estimate

But the current local refinement is not yet selective enough.

It helps often enough to matter and hurts often enough to block validation.

That means the next change should stay inside Layer 3, not jump ahead to Layer 4.

## What This Establishes

This experiment does show:

- Layer 3 should remain a gated conditional solver, not a full-slice point estimator
- candidate-conditioned local `alpha` refinement is a real lever inside the gate-open region
- confirmation performance can beat both the anchored and raw best-bank outputs

This experiment does not show:

- that the current Layer 3 refinement is correct
- that the holdout gate-open region is solved
- that the solver is ready to advance to the final confirmation-stable policy layer

## Figures

- [backbone_conditional_alpha_solver_alpha_error.png](outputs/figures/backbone_conditional_alpha_solver_alpha_error.png)
- [backbone_conditional_alpha_solver_alpha_span.png](outputs/figures/backbone_conditional_alpha_solver_alpha_span.png)

The clearest figure is [backbone_conditional_alpha_solver_alpha_error.png](outputs/figures/backbone_conditional_alpha_solver_alpha_error.png), because it shows the real layer-3 outcome directly:

- the gated refinement improves the point estimate overall
- the confirmation block is genuinely strong
- the holdout block still misses the anchored baseline

## Artifacts

Data:

- [backbone_conditional_alpha_solver_bank_rows.csv](outputs/backbone_conditional_alpha_solver_bank_rows.csv)
- [backbone_conditional_alpha_solver_trials.csv](outputs/backbone_conditional_alpha_solver_trials.csv)
- [backbone_conditional_alpha_solver_split_summary.csv](outputs/backbone_conditional_alpha_solver_split_summary.csv)
- [backbone_conditional_alpha_solver_condition_summary.csv](outputs/backbone_conditional_alpha_solver_condition_summary.csv)
- [backbone_conditional_alpha_solver_cell_summary.csv](outputs/backbone_conditional_alpha_solver_cell_summary.csv)
- [backbone_conditional_alpha_solver_summary.json](outputs/backbone_conditional_alpha_solver_summary.json)

Code:

- [run.py](run.py)

## Next Layer

Do not advance to Layer 4 yet.

The next move should stay inside Layer 3 and target the holdout miss directly.

The most useful immediate variants are:

1. tighten the gate so the mid-skew sparse-full failures do not open
2. make the local refinement more conservative when the refined bank spread grows too fast
3. test whether only one or two backbone-consistent seed families should be allowed in the sparse-full branch
