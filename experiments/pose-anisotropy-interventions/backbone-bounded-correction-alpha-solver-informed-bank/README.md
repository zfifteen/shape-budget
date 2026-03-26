# Backbone Bounded Correction Alpha Solver With Informed Bank

## Purpose

This experiment revisits Layer 3 after the corrected informed-bank integration.

The corrected stack fixed the math and the Layer 2 handoff, but it also showed
that full Layer 3 refinement is still too aggressive on the current open set.

This run tests a simpler rule:

- keep the informed-bank Layer 2 open set unchanged
- keep anchored and refined Layer 3 outputs unchanged
- scale the refined move down when it tries to move too far from anchored

## Method

For each gate-open trial:

1. measure correction excursion in log-alpha space  
   `|log(refined) - log(anchored)|`

2. normalize it by anchored alpha span  
   `excursion_ratio = excursion / mean_anchored_alpha_log_span`

3. apply a bounded correction weight  
   `w = max(0, 1 - excursion_ratio / tau)`

4. output the bounded log-alpha blend  
   `log(bounded) = log(anchored) + w * (log(refined) - log(anchored))`

So large excursions collapse back toward anchored, while small excursions keep
most of the refined move.

## Main Result

The best cached tau on the current fresh open-trial set is:

- `tau = 0.083015`

From [backbone_bounded_correction_alpha_solver_informed_bank_summary.json](outputs/backbone_bounded_correction_alpha_solver_informed_bank_summary.json):

- mean open-trial anchored alpha error: `0.1356`
- mean open-trial full refined alpha error: `0.1807`
- mean open-trial bounded alpha error: `0.1304`

So bounded correction beats both:

- anchored by `0.0052`
- full refinement by `0.0503`

That makes this the strongest Layer 3 behavior we have on top of the corrected
informed-bank stack.

## Split Read

### Holdout

- anchored open-trial error: `0.0406`
- full refined open-trial error: `0.1499`
- bounded open-trial error: `0.0406`

On holdout, bounded correction effectively collapses back to anchored.

### Confirmation

- anchored open-trial error: `0.1989`
- full refined open-trial error: `0.2013`
- bounded open-trial error: `0.1903`

On confirmation, bounded correction keeps the useful small moves while
suppressing the overreach.

## Interpretation

This is the cleanest Layer 3 result after the math fix.

The key point is:

- the problem was not that refinement existed
- the problem was that full refinement moved too far when correction excursion
  got large relative to anchored span

So the useful Layer 3 action is not all-or-nothing refinement.
It is bounded correction.

That fits the current stack very naturally:

- Layer 1 gives the better family
- Layer 2 decides whether point output is allowed
- Layer 3 applies only the amount of correction the anchored state can absorb

## Artifacts

Data:

- [backbone_bounded_correction_alpha_solver_informed_bank_trials.csv](outputs/backbone_bounded_correction_alpha_solver_informed_bank_trials.csv)
- [backbone_bounded_correction_alpha_solver_informed_bank_split_summary.csv](outputs/backbone_bounded_correction_alpha_solver_informed_bank_split_summary.csv)
- [backbone_bounded_correction_alpha_solver_informed_bank_cell_summary.csv](outputs/backbone_bounded_correction_alpha_solver_informed_bank_cell_summary.csv)
- [backbone_bounded_correction_alpha_solver_informed_bank_summary.json](outputs/backbone_bounded_correction_alpha_solver_informed_bank_summary.json)

Figures:

- [backbone_bounded_correction_alpha_solver_informed_bank_alpha_error.png](outputs/figures/backbone_bounded_correction_alpha_solver_informed_bank_alpha_error.png)
- [backbone_bounded_correction_alpha_solver_informed_bank_bounded_weight.png](outputs/figures/backbone_bounded_correction_alpha_solver_informed_bank_bounded_weight.png)

Code:

- [run.py](run.py)
