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

- metric: `mean_anchored_alpha_log_std / mean_alpha_log_span_set`
- threshold: `0.215678`
- direction: `ge`

The gate opens when that ratio falls below the threshold.

## Main Result

This is the first integrated informed-bank Layer 3 pass that behaves cleanly on
both fresh splits inside the trials it opens.

From [backbone_conditional_alpha_solver_informed_bank_summary.json](outputs/backbone_conditional_alpha_solver_informed_bank_summary.json):

- point-output count: `4 / 18`
- point-output rate: `0.2222`
- gate precision: `0.7500`
- gate reject-unrecoverable rate: `0.9000`
- best open-trial `alpha` error: `0.1266`
- anchored open-trial `alpha` error: `0.1264`
- refined open-trial `alpha` error: `0.0514`

So the open-trial refinement is now doing something much sharper:

- it opens rarely
- it opens mostly in the right places
- and when it opens, refined beats both anchored and best-bank

## By Split

### Holdout

- point-output count: `2 / 9`
- point-output rate: `0.2222`
- gate balanced accuracy: `0.5833`
- gate precision: `0.5000`
- best open-trial error: `0.1535`
- anchored open-trial error: `0.1417`
- refined open-trial error: `0.0647`

### Confirmation

- point-output count: `2 / 9`
- point-output rate: `0.2222`
- gate balanced accuracy: `0.7000`
- gate precision: `1.0000`
- best open-trial error: `0.0996`
- anchored open-trial error: `0.1111`
- refined open-trial error: `0.0381`

That is the important change.
The earlier informed-bank integration was still over-opening.
This version is selective enough that the opened trials are genuinely good.

## Cell Read

The gate opens in four cells:

- holdout `low_skew`
- holdout `mid_skew`
- confirmation `low_skew`
- confirmation `mid_skew`

It stays closed in both `high_skew` fresh cells.

When it opens, refinement is uniformly good:

- `refined_beats_anchored_rate_open = 1.0`
- `refined_beats_best_rate_open = 1.0`

The strongest confirmation gains are:

- `confirmation + low_skew`: `0.1641 -> 0.0504`
- `confirmation + mid_skew`: `0.0582 -> 0.0257`

The holdout gains are real too:

- `holdout + low_skew`: `0.1643 -> 0.0399`
- `holdout + mid_skew`: `0.1190 -> 0.0895`

## Interpretation

This is a meaningful solver step.

What it now shows is:

- Layer 1 improves the candidate family
- Layer 2 can be tightened into a selective informed-bank gate
- Layer 3 refinement can then produce very strong open-trial `alpha` recovery

What remains open is not whether the integrated stack can work.
It can.
The remaining question is coverage:

- can we safely expand beyond this very selective `4 / 18` open-trial regime?

So the next implementation move is:

1. preserve this specialized Layer 2 rule as the current best selective gate
2. look for a controlled way to widen coverage without losing the clean open-trial win

## Artifacts

Data:

- [backbone_conditional_alpha_solver_informed_bank_bank_rows.csv](outputs/backbone_conditional_alpha_solver_informed_bank_bank_rows.csv)
- [backbone_conditional_alpha_solver_informed_bank_trials.csv](outputs/backbone_conditional_alpha_solver_informed_bank_trials.csv)
- [backbone_conditional_alpha_solver_informed_bank_split_summary.csv](outputs/backbone_conditional_alpha_solver_informed_bank_split_summary.csv)
- [backbone_conditional_alpha_solver_informed_bank_condition_summary.csv](outputs/backbone_conditional_alpha_solver_informed_bank_condition_summary.csv)
- [backbone_conditional_alpha_solver_informed_bank_cell_summary.csv](outputs/backbone_conditional_alpha_solver_informed_bank_cell_summary.csv)
- [backbone_conditional_alpha_solver_informed_bank_summary.json](outputs/backbone_conditional_alpha_solver_informed_bank_summary.json)

Figures:

- [backbone_conditional_alpha_solver_informed_bank_alpha_error.png](outputs/figures/backbone_conditional_alpha_solver_informed_bank_alpha_error.png)
- [backbone_conditional_alpha_solver_informed_bank_alpha_span.png](outputs/figures/backbone_conditional_alpha_solver_informed_bank_alpha_span.png)

Code:

- [run.py](run.py)
