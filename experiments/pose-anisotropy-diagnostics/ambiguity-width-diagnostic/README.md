# Ambiguity-Width Diagnostic

## Purpose

The current bottleneck is whether the focused solver failures come from weak routing inside one bank or from observation-conditioned ambiguity that survives across independently sampled banks.

This experiment tests that distinction directly.

## Research Question

If the same focused observation is scored against several independent reference banks, does the width of the near-optimal candidate family predict cross-bank `alpha` instability better than a simple entropy signal?

## Method

The setup stays inside the active solver challenge slice:

- `alpha_strength_bin = moderate`
- conditions:
  - `sparse_full_noisy`
  - `sparse_partial_high_noise`
- geometry skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

The observation seeds reuse the same calibration, holdout, and confirmation blocks as the current bank-adaptive solver ladder.

Each fixed observation is then evaluated against five independently sampled anisotropic reference banks:

- bank seeds: `20260324`, `20260325`, `20260326`, `20260327`, `20260328`
- bank size: `300`

For each bank:

1. score the observation with the existing marginalized-bank solver
2. keep the near-optimal candidate family within `best_score + max(noise_sigma^2, 5e-5)`
3. measure the `log(alpha)` span of that family
4. measure the normalized geometry span and weight span of that family
5. define the ambiguity ratio as:
   - `alpha span / max(combined geometry-weight span, eps)`

For each fixed observation across banks:

- measure cross-bank `log(alpha)` span from the best candidates
- mark the observation as `alpha`-unstable when that span is at least `0.20`
- compare two frozen threshold rules:
  - `mean_ambiguity_ratio`
  - `mean_best_entropy`

The executable artifact is [run.py](run.py).

## Main Result

The experiment shows that ambiguity width is a real diagnostic signal in the focused challenge slice.

The summary file is [ambiguity_width_diagnostic_summary.json](outputs/ambiguity_width_diagnostic_summary.json).

Global result:

- trial count: `72`
- mean ambiguity ratio: `0.6381`
- mean cross-bank `log(alpha)` span: `0.2993`
- mean cross-bank geometry span: `0.4972`
- ambiguity-ratio vs cross-bank `alpha` span correlation: `0.6539`
- plain `alpha`-set-span vs cross-bank `alpha` span correlation: `0.5636`
- ambiguity-ratio vs cross-bank geometry span correlation: `0.2063`
- entropy vs cross-bank `alpha` span correlation: `-0.5436`

That is the core result.

The ambiguity metric tracks the unstable part of the latent state much better than the entropy baseline, and entropy points in the wrong direction in this experiment.

## Frozen Threshold Test

Thresholds were selected on calibration only and then frozen.

`mean_ambiguity_ratio`:

- threshold: `0.5649`
- calibration balanced accuracy: `0.7500`
- holdout balanced accuracy: `0.6250`
- confirmation balanced accuracy: `0.8667`
- overall balanced accuracy: `0.7369`

`mean_alpha_log_span_set`:

- threshold: `0.4236`
- calibration balanced accuracy: `0.7321`
- holdout balanced accuracy: `0.6250`
- confirmation balanced accuracy: `0.8333`
- overall balanced accuracy: `0.7187`

`mean_best_entropy`:

- threshold: `0.3047`
- calibration balanced accuracy: `0.5000`
- holdout balanced accuracy: `0.4583`
- confirmation balanced accuracy: `0.5000`
- overall balanced accuracy: `0.4909`

The ambiguity-width rule is the strongest frozen classifier.
Plain `alpha`-set width is a meaningful baseline, but the ratio still adds predictive value by comparing `alpha` width to the accompanying geometry-weight spread.
Entropy does not generalize.

## Stable vs Unstable Observations

The split between stable and unstable observations is sharp.

- stable observations: `17`
- unstable observations: `55`

Stable observations:

- mean ambiguity ratio: `0.5122`
- mean best entropy: `0.7416`
- mean cross-bank `log(alpha)` span: `0.0647`
- mean cross-bank geometry span: `0.4531`

Unstable observations:

- mean ambiguity ratio: `0.6770`
- mean best entropy: `0.5398`
- mean cross-bank `log(alpha)` span: `0.3718`
- mean cross-bank geometry span: `0.5108`

This pattern matters.

The unstable cases do not show a comparable blow-up in cross-bank geometry spread.
The geometry term does move, but the strongest separation is in `alpha`, not in the whole latent object.

## By Split

- calibration:
  - ambiguity-ratio vs `alpha`-span correlation: `0.5841`
  - plain `alpha`-set-span vs `alpha`-span correlation: `0.4353`
  - entropy vs `alpha`-span correlation: `-0.5657`
  - instability rate: `0.7778`

- holdout:
  - ambiguity-ratio vs `alpha`-span correlation: `0.6296`
  - plain `alpha`-set-span vs `alpha`-span correlation: `0.6054`
  - entropy vs `alpha`-span correlation: `-0.6569`
  - instability rate: `0.6667`

- confirmation:
  - ambiguity-ratio vs `alpha`-span correlation: `0.9142`
  - plain `alpha`-set-span vs `alpha`-span correlation: `0.8465`
  - entropy vs `alpha`-span correlation: `-0.3341`
  - instability rate: `0.8333`

The confirmation block is the cleanest evidence that the metric is not just fitting calibration noise.

## By Condition

Condition means:

- `sparse_full_noisy`
  - holdout mean ambiguity ratio: `0.6205`
  - holdout mean cross-bank `alpha` span: `0.2884`
  - confirmation mean ambiguity ratio: `0.5857`
  - confirmation mean cross-bank `alpha` span: `0.2893`

- `sparse_partial_high_noise`
  - holdout mean ambiguity ratio: `0.5696`
  - holdout mean cross-bank `alpha` span: `0.2301`
  - confirmation mean ambiguity ratio: `0.6649`
  - confirmation mean cross-bank `alpha` span: `0.3479`

The hardest confirmation cell is `sparse_partial_high_noise + high_skew`:

- mean ambiguity ratio: `0.7638`
- mean cross-bank `alpha` span: `0.4516`
- mean cross-bank geometry span: `0.4933`
- instability rate: `1.0000`

## Interpretation

This experiment sharpens the current symmetry diagnosis.

The useful variable is not just whether one bank has a wide near-best family.
It is whether the observation induces a wide `alpha` family that stays unstable when the bank is resampled.

That is a different failure mode from ordinary confidence loss.
Entropy rises on many stable observations and falls on many unstable ones, so it is not measuring the right object here.

The ambiguity-width ratio is closer to the mechanism we care about:

- a fixed observation
- several near-best explanations
- comparatively modest geometry drift
- materially different `alpha` answers across independent banks

## What This Establishes

This experiment does show:

- a frozen ambiguity-width threshold beats entropy as a predictor of cross-bank `alpha` instability
- the ambiguity-width ratio is also stronger than plain within-set `alpha` width
- the signal survives calibration, holdout, and confirmation
- the instability concentrates more in `alpha` than in geometry
- the hardest confirmation failures sit in the high-ambiguity part of the focused slice

This experiment does not show:

- that the challenge is formally impossible under the current observation model
- that ambiguity width by itself solves the solver challenge
- which remedy is best after a trial is classified as ambiguity-dominated

## Figures

- [ambiguity_width_diagnostic_alpha_scatter.png](outputs/figures/ambiguity_width_diagnostic_alpha_scatter.png)
- [ambiguity_width_diagnostic_geometry_scatter.png](outputs/figures/ambiguity_width_diagnostic_geometry_scatter.png)
- [ambiguity_width_diagnostic_threshold_bars.png](outputs/figures/ambiguity_width_diagnostic_threshold_bars.png)

The clearest figure is [ambiguity_width_diagnostic_alpha_scatter.png](outputs/figures/ambiguity_width_diagnostic_alpha_scatter.png), because it shows the main structural claim directly:

- higher ambiguity ratio tracks larger cross-bank `alpha` spread
- the instability threshold cuts through a visibly different region of the scatter

## Artifacts

Data:

- [ambiguity_width_diagnostic_bank_rows.csv](outputs/ambiguity_width_diagnostic_bank_rows.csv)
- [ambiguity_width_diagnostic_trials.csv](outputs/ambiguity_width_diagnostic_trials.csv)
- [ambiguity_width_diagnostic_split_summary.csv](outputs/ambiguity_width_diagnostic_split_summary.csv)
- [ambiguity_width_diagnostic_condition_summary.csv](outputs/ambiguity_width_diagnostic_condition_summary.csv)
- [ambiguity_width_diagnostic_cell_summary.csv](outputs/ambiguity_width_diagnostic_cell_summary.csv)
- [ambiguity_width_diagnostic_summary.json](outputs/ambiguity_width_diagnostic_summary.json)

Code:

- [run.py](run.py)

## Recommended Next Step

The next intervention should use ambiguity width as a gate, not as a final solver.

Two follow-ups are now justified:

1. add a frozen ambiguity-width gate in front of the current solver ladder and measure whether abstention improves the accepted slice
2. condition a symmetry-breaking alignment step only on the high-ambiguity observations and test whether that collapses cross-bank `alpha` spread
