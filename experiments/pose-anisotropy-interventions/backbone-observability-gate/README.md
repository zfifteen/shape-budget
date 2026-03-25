# Backbone Observability Gate

## Purpose

This experiment is the second capability layer in the backbone-first solver program.

Layer 1 established that a near-family geometry consensus recovers a stable backbone across fresh banks.

Layer 2 asks the next question:

- once the backbone is anchored, when should the solver trust a point estimate for `alpha` at all?

This is a gate experiment, not a full solver.

## Research Question

If the geometry backbone is anchored first, can a backbone-anchored `alpha` uncertainty metric identify the cases where a point `alpha` estimate is trustworthy better than the older ambiguity metrics?

## Layer Position

This is `Layer 2` of the staged solver plan in [docs/SOLVER_CHALLENGES.md](../../../docs/SOLVER_CHALLENGES.md).

Layer target:

1. stable backbone recovery
2. extension-coordinate observability gate

Not attempted here:

3. full conditional `alpha` recovery
4. full confirmation-stable solver policy

## Method

The setup stays on the focused slice:

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
3. recover the geometry backbone as a score-weighted geometry consensus
4. build a backbone-anchored `alpha` posterior by reweighting the near-best family with:
   - score decay
   - geometry distance to the recovered backbone
5. record:
   - anchored `alpha` mean
   - anchored `alpha` log standard deviation
   - anchored weighted `10%` to `90%` log-span

The geometry-anchor scale is fixed at `0.10` in normalized geometry units.

## Trustworthiness Label

This layer evaluates whether the anchored point estimate is trustworthy.

A trial is marked `alpha`-point-recoverable only when both are true across banks:

- anchored `alpha` bank log-span `< 0.20`
- anchored mean absolute `alpha` error `< 0.15`

That label is intentionally narrow.
It evaluates the trustworthiness of the layer-2 anchored estimate, not a universal claim about final `alpha` recoverability under every possible future method.

## Main Result

The layer-2 gate validates, but the anchored point estimate is not yet a finished solver.

The summary file is [backbone_observability_gate_summary.json](outputs/backbone_observability_gate_summary.json).

Global result:

- trial count: `72`
- `alpha`-point-recoverable rate: `0.4444`
- mean anchored `alpha` bank log-span: `0.0757`
- mean best-candidate `alpha` bank log-span: `0.2993`
- mean anchored `alpha` absolute error: `0.1610`
- mean best-candidate `alpha` absolute error: `0.1641`
- anchored estimate beats best-candidate `alpha` in `0.5000` of trials

That is the core result.

Anchoring to the backbone removes most of the fresh-bank `alpha` volatility.
It does not, by itself, remove enough `alpha` bias to count as a solved conditional estimator.

## Frozen Gate Test

Thresholds were selected on calibration only and then frozen.

The strongest unseen-block gate is `mean_anchored_alpha_log_std`:

- threshold: `0.1890`
- calibration balanced accuracy: `0.7337`
- holdout balanced accuracy: `0.6818`
- confirmation balanced accuracy: `0.7000`
- overall balanced accuracy: `0.7094`

The weighted anchored-span metric matches it on holdout and confirmation:

- threshold: `0.4639`
- calibration balanced accuracy: `0.7337`
- holdout balanced accuracy: `0.6818`
- confirmation balanced accuracy: `0.7000`
- overall balanced accuracy: `0.7094`

The older ambiguity ratio is still strong on calibration, but generalizes worse on confirmation for this layer-2 target:

- threshold: `0.5972`
- calibration balanced accuracy: `0.7802`
- holdout balanced accuracy: `0.6558`
- confirmation balanced accuracy: `0.5625`
- overall balanced accuracy: `0.6906`

Entropy remains weak:

- overall balanced accuracy: `0.5469`

So the important distinction is:

- ambiguity ratio is still a good structural diagnostic
- backbone-anchored `alpha` uncertainty is the better gate for the trustworthiness of the anchored point estimate

## By Split

- calibration:
  - recoverable rate: `0.4722`
  - anchored `alpha` bank span: `0.0854`
  - best `alpha` bank span: `0.3097`
  - anchored `alpha` error: `0.1575`
  - best `alpha` error: `0.1661`

- holdout:
  - recoverable rate: `0.3889`
  - anchored `alpha` bank span: `0.0619`
  - best `alpha` bank span: `0.2592`
  - anchored `alpha` error: `0.1770`
  - best `alpha` error: `0.1915`

- confirmation:
  - recoverable rate: `0.4444`
  - anchored `alpha` bank span: `0.0701`
  - best `alpha` bank span: `0.3186`
  - anchored `alpha` error: `0.1521`
  - best `alpha` error: `0.1326`

The confirmation block is the key nuance.

The gate holds up on confirmation, but the anchored point estimate itself is still not uniformly better than the raw best candidate there.

## By Condition

- `sparse_full_noisy`
  - holdout recoverable rate: `0.5556`
  - confirmation recoverable rate: `0.5556`
  - holdout anchored `alpha` error: `0.1528`
  - confirmation anchored `alpha` error: `0.1399`

- `sparse_partial_high_noise`
  - holdout recoverable rate: `0.2222`
  - confirmation recoverable rate: `0.3333`
  - holdout anchored `alpha` error: `0.2013`
  - confirmation anchored `alpha` error: `0.1643`

The gate says something structurally useful:

- `sparse_full_noisy` contains a much larger point-recoverable region
- `sparse_partial_high_noise` remains mostly outside that region even after the backbone is anchored

## Interpretation

Layer 2 changes the read of the solver challenge in a useful way.

The first layer showed that the backbone can be stabilized.
This layer shows that stabilizing the backbone is not the same thing as already solving `alpha`.

What the gate is capturing is:

- some observations support a trustworthy anchored `alpha` estimate
- others remain unrecoverable even after the backbone is fixed
- that boundary is better measured by anchored `alpha` uncertainty than by the older generic ambiguity metrics

This is exactly the role a layer-2 gate should play.

## What This Establishes

This experiment does show:

- a layer-2 observability gate can be built on top of the validated backbone layer
- backbone-anchored `alpha` uncertainty generalizes better than the older ambiguity ratio for the trustworthiness of the anchored estimate
- backbone anchoring collapses most cross-bank `alpha` volatility
- the remaining difficulty is now mainly conditional `alpha` bias, not raw bank instability

This experiment does not show:

- that the anchored `alpha` mean is already a good final estimator
- that the full solver challenge is solved
- that the gate should replace a proper conditional `alpha` recovery layer

## Figures

- [backbone_observability_gate_scatter.png](outputs/figures/backbone_observability_gate_scatter.png)
- [backbone_observability_gate_alpha_error.png](outputs/figures/backbone_observability_gate_alpha_error.png)
- [backbone_observability_gate_thresholds.png](outputs/figures/backbone_observability_gate_thresholds.png)

The clearest figure is [backbone_observability_gate_thresholds.png](outputs/figures/backbone_observability_gate_thresholds.png), because it shows the main layer-2 outcome directly:

- anchored uncertainty metrics are the best unseen-block gate for anchored-point unrecoverability
- the old ambiguity ratio is no longer the best metric once the backbone has already been recovered

## Artifacts

Data:

- [backbone_observability_gate_bank_rows.csv](outputs/backbone_observability_gate_bank_rows.csv)
- [backbone_observability_gate_trials.csv](outputs/backbone_observability_gate_trials.csv)
- [backbone_observability_gate_split_summary.csv](outputs/backbone_observability_gate_split_summary.csv)
- [backbone_observability_gate_condition_summary.csv](outputs/backbone_observability_gate_condition_summary.csv)
- [backbone_observability_gate_cell_summary.csv](outputs/backbone_observability_gate_cell_summary.csv)
- [backbone_observability_gate_summary.json](outputs/backbone_observability_gate_summary.json)

Code:

- [run.py](run.py)

## Next Layer

Layer 3 is a conditional `alpha` recovery method, not a new gate.

The first Layer 3 attempt is:

- [Backbone Conditional Alpha Solver](../backbone-conditional-alpha-solver/README.md)

The next move inside Layer 3 is:

1. keep the geometry backbone anchored
2. open the gate only on the layer-2 point-recoverable region
3. perform a dedicated conditional `alpha` solve inside that region
4. abstain or return an ambiguity object when the gate stays closed
5. tighten the conditional solver until it clears both holdout and confirmation
