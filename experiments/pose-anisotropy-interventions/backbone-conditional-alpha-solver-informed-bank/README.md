# Backbone Conditional Alpha Solver With Informed Bank

## Purpose

This experiment is the integrated Layer 3 pass on top of the new upstream stack:

- Layer 1: persistent-mode informed bank
- Layer 2: specialized informed-bank gate
- Layer 3: candidate-conditioned conditional `alpha` refinement

The question is whether the informed bank plus a tighter Layer 2 rule can make
the Layer 3 point solver behave cleanly on the hard `sparse_partial_high_noise`
branch.

## Scope

This run stays on the hard fresh branch that the informed-bank atlas already
covers:

- condition: `sparse_partial_high_noise`
- alpha strength: `moderate`
- skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`
- splits:
  - `holdout`
  - `confirmation`

It is a cached integrated pass:

- Layer 1 candidate families come from the informed-bank atlas
- Layer 2 metrics come from the informed-bank trial table
- Layer 3 reruns only the local candidate-conditioned refinement

## Layer 2 Rule

This version uses the specialized informed-bank Layer 2 rule from
[backbone observability gate informed-bank specialized ratio sweep](../backbone-observability-gate-informed-bank-specialized-ratio-sweep/README.md):

- metric: `mean_candidate_count * mean_anchored_alpha_log_span / mean_anchored_effective_count`
- threshold: `0.700453`
- direction: `ge`

The gate opens when that ratio rises above the threshold.

This run also restores the intended Layer 3 refinement weighting:

- refined seed weights use `exp(-score_offset / band)`
- `band` comes from the observation regime, matching the earlier conditional
  solver math
- the Layer 2 rule is loaded from the serialized specialized sweep output
  rather than from a hardcoded override

## Main Result

The math-corrected integrated pass is more conservative than the earlier
optimistic version.

From [backbone_conditional_alpha_solver_informed_bank_summary.json](outputs/backbone_conditional_alpha_solver_informed_bank_summary.json):

- nominal final bank size: `300`
- mean band candidate count: `209.9259`
- point-output count: `5 / 18`
- point-output rate: `0.2778`
- gate precision: `0.8000`
- gate reject-unrecoverable rate: `0.9000`
- best open-trial `alpha` error: `0.2075`
- anchored open-trial `alpha` error: `0.1356`
- refined open-trial `alpha` error: `0.1807`

So the corrected stack still gives a reasonably selective gate, but the current
Layer 3 refinement does not beat the anchored answer overall on the opened
trials.

## By Split

### Holdout

- point-output count: `2 / 9`
- point-output rate: `0.2222`
- gate balanced accuracy: `0.5833`
- gate precision: `0.5000`
- best open-trial error: `0.1484`
- anchored open-trial error: `0.0406`
- refined open-trial error: `0.1499`

### Confirmation

- point-output count: `3 / 9`
- point-output rate: `0.3333`
- gate balanced accuracy: `0.8000`
- gate precision: `1.0000`
- best open-trial error: `0.2468`
- anchored open-trial error: `0.1989`
- refined open-trial error: `0.2013`

The confirmation side remains close, but holdout is clearly not there yet.

## Cell Read

The gate opens in five fresh cells:

- holdout `low_skew`
- holdout `high_skew`
- confirmation `low_skew`
- confirmation `mid_skew`
- confirmation `high_skew`

The cell picture is mixed, not uniformly good:

- holdout `high_skew`: refined beats best but not anchored
- confirmation `low_skew`: refined beats both anchored and best
- confirmation `mid_skew`: refined loses to both anchored and best
- confirmation `high_skew`: refined beats both anchored and best

## Interpretation

This is still a useful integration result, but it is a corrective one.

What it now shows is:

- Layer 1 improves the candidate family
- Layer 2 can be serialized cleanly and consumed without hardcoded drift
- the restored Layer 3 math is less flattering than the earlier cached result

The current open problem is not gate serialization anymore.
That part is fixed.
The remaining question is whether Layer 3 can produce a net-improving refined
answer under the corrected weighting math.

So the next implementation move is:

1. keep this Layer 2 rule as the current informed-bank source of truth
2. revisit Layer 3 correction behavior under the restored band-scaled weights
3. only widen coverage after refinement is net-positive again

## Artifacts

Data:

- [backbone_conditional_alpha_solver_informed_bank_bank_rows.csv](outputs/backbone_conditional_alpha_solver_informed_bank_bank_rows.csv)
- [backbone_conditional_alpha_solver_informed_bank_trials.csv](outputs/backbone_conditional_alpha_solver_informed_bank_trials.csv)
- [backbone_conditional_alpha_solver_informed_bank_all_refine_trials.csv](outputs/backbone_conditional_alpha_solver_informed_bank_all_refine_trials.csv)
- [backbone_conditional_alpha_solver_informed_bank_split_summary.csv](outputs/backbone_conditional_alpha_solver_informed_bank_split_summary.csv)
- [backbone_conditional_alpha_solver_informed_bank_condition_summary.csv](outputs/backbone_conditional_alpha_solver_informed_bank_condition_summary.csv)
- [backbone_conditional_alpha_solver_informed_bank_cell_summary.csv](outputs/backbone_conditional_alpha_solver_informed_bank_cell_summary.csv)
- [backbone_conditional_alpha_solver_informed_bank_summary.json](outputs/backbone_conditional_alpha_solver_informed_bank_summary.json)

Figures:

- [backbone_conditional_alpha_solver_informed_bank_alpha_error.png](outputs/figures/backbone_conditional_alpha_solver_informed_bank_alpha_error.png)
- [backbone_conditional_alpha_solver_informed_bank_alpha_span.png](outputs/figures/backbone_conditional_alpha_solver_informed_bank_alpha_span.png)

Code:

- [run.py](run.py)
