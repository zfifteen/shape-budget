# Backbone Consensus Solver

## Purpose

This experiment is the first capability layer in the backbone-first solver program.

The construction rule is:

- mirror the control structure first
- validate one layer at a time
- do not force full-latent point recovery before the stable backbone is established

The first layer is therefore narrow:

- recover the normalized-geometry backbone
- do not try to solve `alpha`
- test whether geometry can be made bank-stable by taking a near-family consensus instead of a winner-take-all best candidate

## Research Question

If the same focused pose-free observation is scored against several independent banks, does a geometry consensus over the near-best family recover the backbone more stably than the usual best candidate?

## Layer Position

This is `Layer 1` of the staged solver plan in [docs/SOLVER_CHALLENGES.md](../../../docs/SOLVER_CHALLENGES.md).

Layer target:

1. stable backbone recovery

Not attempted here:

2. extension-coordinate observability gating
3. conditional `alpha` recovery
4. full confirmation-stable solver policy

## Method

The experiment stays on the active focused slice:

- `alpha_strength_bin = moderate`
- conditions:
  - `sparse_full_noisy`
  - `sparse_partial_high_noise`
- geometry skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

Observation seeds reuse the same calibration, holdout, and confirmation blocks as the current bank-adaptive solver ladder.

Each fixed observation is scored against five independent anisotropic banks:

- bank seeds: `20260324`, `20260325`, `20260326`, `20260327`, `20260328`
- bank size: `300`

For each bank:

1. score the bank with the existing pose-marginalized bank scorer
2. keep the near-best family within `best_score + max(noise_sigma^2, 5e-5)`
3. form a geometry consensus by averaging canonicalized geometry invariants across that family with score-decay weights
4. compare that consensus geometry to the ordinary best-candidate geometry

For each fixed observation across banks, record:

- best-candidate geometry error
- consensus-geometry error
- best-candidate geometry bank span
- consensus-geometry bank span
- best-candidate `alpha` bank span for context

The executable artifact is [run.py](run.py).

## Main Result

The first backbone layer validates strongly.

The summary file is [backbone_consensus_solver_summary.json](outputs/backbone_consensus_solver_summary.json).

Global result:

- trial count: `72`
- mean best-candidate geometry bank span: `0.4972`
- mean consensus-geometry bank span: `0.0945`
- mean best-candidate geometry MAE: `0.0952`
- mean consensus-geometry MAE: `0.0675`
- mean geometry MAE gain from consensus: `0.0277`
- mean geometry bank-span gain from consensus: `0.4026`
- consensus beats best on geometry MAE in `0.7639` of trials
- consensus is tighter across banks in `0.9722` of trials

That is the core result.

The raw winner-take-all best candidate does not expose a stable backbone across bank seeds.
The near-family geometry consensus does.

## By Split

- calibration:
  - best geometry MAE: `0.0879`
  - consensus geometry MAE: `0.0674`
  - best geometry bank span: `0.4641`
  - consensus geometry bank span: `0.0966`
  - consensus beats best rate: `0.6667`
  - consensus tighter rate: `0.9444`

- holdout:
  - best geometry MAE: `0.0982`
  - consensus geometry MAE: `0.0655`
  - best geometry bank span: `0.5409`
  - consensus geometry bank span: `0.0899`
  - consensus beats best rate: `0.8333`
  - consensus tighter rate: `1.0000`

- confirmation:
  - best geometry MAE: `0.1067`
  - consensus geometry MAE: `0.0696`
  - best geometry bank span: `0.5195`
  - consensus geometry bank span: `0.0950`
  - consensus beats best rate: `0.8889`
  - consensus tighter rate: `1.0000`

The confirmation block matters most.

This layer stays strong on the fresh-bank confirmation block without any calibrated chooser.

## By Condition

- `sparse_full_noisy`
  - holdout best geometry MAE: `0.0930`
  - holdout consensus geometry MAE: `0.0578`
  - confirmation best geometry MAE: `0.0956`
  - confirmation consensus geometry MAE: `0.0623`

- `sparse_partial_high_noise`
  - holdout best geometry MAE: `0.1035`
  - holdout consensus geometry MAE: `0.0731`
  - confirmation best geometry MAE: `0.1178`
  - confirmation consensus geometry MAE: `0.0769`

The geometry layer helps in both support branches.

## Important Nuance

The result is strong, but it is not perfectly uniform.

The main cautionary cell is calibration `sparse_full_noisy + mid_skew`, where consensus still tightens bank span but only beats the best-candidate geometry error in `1` of `6` cases.

So this is a validated first layer, not a claim that every backbone detail is already solved in every cell.

## Interpretation

This is the first direct evidence that the solver should be built layer by layer rather than as one full-latent estimator.

The near-best family contains a geometry signal that the single best candidate does not preserve across bank resampling.

That means:

- the backbone is not best represented by one winning candidate
- the backbone can be recovered more stably from the family consensus
- the remaining hard problem is now pushed downstream toward extension-coordinate handling rather than upstream backbone collapse

This is exactly the kind of result the layered construction was meant to test.

## What This Establishes

This experiment does show:

- a backbone-first geometry layer can be made fresh-bank stable on the focused slice
- near-family geometry consensus is materially more stable than winner-take-all best-candidate geometry
- geometry MAE also improves, not only stability
- the backbone-first approach is now experimentally justified as the correct next solver program

This experiment does not show:

- that weights should already be included in the backbone layer
- that `alpha` is point-recoverable after the backbone is recovered
- that the full solver challenge is solved

## Figures

- [backbone_consensus_solver_geometry_error.png](outputs/figures/backbone_consensus_solver_geometry_error.png)
- [backbone_consensus_solver_geometry_span.png](outputs/figures/backbone_consensus_solver_geometry_span.png)

The clearest figure is [backbone_consensus_solver_geometry_error.png](outputs/figures/backbone_consensus_solver_geometry_error.png), because it shows that the first-layer backbone estimate improves on all three solver blocks.

## Artifacts

Data:

- [backbone_consensus_solver_bank_rows.csv](outputs/backbone_consensus_solver_bank_rows.csv)
- [backbone_consensus_solver_trials.csv](outputs/backbone_consensus_solver_trials.csv)
- [backbone_consensus_solver_split_summary.csv](outputs/backbone_consensus_solver_split_summary.csv)
- [backbone_consensus_solver_condition_summary.csv](outputs/backbone_consensus_solver_condition_summary.csv)
- [backbone_consensus_solver_cell_summary.csv](outputs/backbone_consensus_solver_cell_summary.csv)
- [backbone_consensus_solver_summary.json](outputs/backbone_consensus_solver_summary.json)

Code:

- [run.py](run.py)

## Next Layer

The next capability layer should not jump to full `alpha` recovery.

It should add an extension observability gate on top of the backbone layer:

1. keep the geometry backbone fixed or tightly anchored
2. diagnose whether the observation supports point `alpha` recovery
3. only then attempt conditional `alpha` recovery or abstention
