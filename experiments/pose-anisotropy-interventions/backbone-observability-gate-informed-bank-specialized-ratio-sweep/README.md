# Backbone Observability Gate Informed-Bank Specialized Ratio Sweep

## Purpose

This experiment specializes Layer 2 for the informed-bank regime on the hard
`sparse_partial_high_noise` branch.

The earlier informed-bank gate comparison had two problems for this job:

- the informed-bank fresh data was narrow
- the best reported rule was stale relative to the current trial table

So this sweep rebuilds the missing informed-bank calibration block for the
informed method only, then compares candidate-aware Layer 2 ratios on:

- calibration
- holdout
- confirmation

## Method

Calibration rows are generated directly with the persistent-mode informed bank:

- calibration seeds: `20260410`, `20260411`, `20260412`, `20260416`, `20260417`, `20260418`
- condition: `sparse_partial_high_noise`
- skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

That gives:

- calibration count: `18`
- holdout count: `9`
- confirmation count: `9`

Each metric is calibrated on the informed-bank calibration block, then frozen
and scored on holdout and confirmation.

The sweep now serializes two selections:

- `best_metric`: the strongest pure Layer 2 unrecoverable classifier
- `selected_for_layer3`: the rule Layer 3 should actually consume

Those are not forced to be the same object. `selected_for_layer3` is stored as
an open-rule, not as an unrecoverable classifier.

## Main Result

The best Layer 2 rule by out-of-sample balanced accuracy is:

- metric: `mean_candidate_count * mean_anchored_alpha_log_span / mean_anchored_effective_count`
- threshold: `0.700453`
- direction: `le`
- calibration balanced accuracy: `0.6667`
- holdout balanced accuracy: `0.5833`
- confirmation balanced accuracy: `0.8000`
- out-of-sample mean balanced accuracy: `0.6917`

That is the strongest pure gate rule in this specialized sweep.

## Layer 3 Selection

After restoring the intended Layer 3 weighting math, no downstream candidate
rule cleared the current Layer 3 selection criteria on both fresh splits.

So the serialized Layer 3-facing rule falls back to the open-side form of the
best pure Layer 2 classifier:

- metric: `mean_candidate_count * mean_anchored_alpha_log_span / mean_anchored_effective_count`
- threshold: `0.700453`
- direction: `ge`
- selection rule: `fallback_best_balanced_accuracy`

That distinction matters:

- `best_metric` remains the best unrecoverable classifier
- `selected_for_layer3` is the rule Layer 3 actually consumes
- when the downstream search finds no better open-rule, `selected_for_layer3`
  becomes the direct open-side version of `best_metric`

So this sweep now produces a single source of truth for both:

1. the best pure Layer 2 classifier
2. the current Layer 3-facing gate rule

## Top Metrics

From [backbone_observability_gate_informed_bank_specialized_ratio_sweep_summary.json](outputs/backbone_observability_gate_informed_bank_specialized_ratio_sweep_summary.json):

1. `ratio_candidate_times_span_over_effective`
   - OOS mean balanced accuracy: `0.6917`
2. `metric_anchored_span`
   - OOS mean balanced accuracy: `0.6917`
3. `ratio_geomspan_over_std`
   - OOS mean balanced accuracy: `0.6917`
4. `ratio_std_over_set_span`
   - OOS mean balanced accuracy: `0.6417`
5. `ratio_ambiguity_over_std`
   - OOS mean balanced accuracy: `0.6417`

## Artifacts

Data:

- [backbone_observability_gate_informed_bank_specialized_ratio_sweep_leaderboard.csv](outputs/backbone_observability_gate_informed_bank_specialized_ratio_sweep_leaderboard.csv)
- [backbone_observability_gate_informed_bank_specialized_ratio_sweep_summary.json](outputs/backbone_observability_gate_informed_bank_specialized_ratio_sweep_summary.json)

Code:

- [run.py](run.py)
