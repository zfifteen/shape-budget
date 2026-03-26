# Backbone Observability Gate Ratio Sweep

This experiment compares multiple Layer 2 metric families on the hard branch:

- `legacy_random_5seed`
- `one_shot_random`
- `persistent_mode_informed`

It uses the Layer 2 trial table from:

- [backbone_observability_gate_informed_bank_trials.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/backbone_observability_gate_informed_bank_trials.csv)

For each method:

1. choose threshold and direction on `calibration`
2. freeze that rule
3. score `holdout`
4. score `confirmation`
5. rank metrics by out-of-sample mean balanced accuracy

The compared metric families include:

- raw Layer 2 signals
- spread-normalized ratios
- support-normalized ratios
- mixed spread-and-support ratios

This is a comparative test harness, not a claim that every ratio here is a valid control law.

## Main Result

For the informed-bank Layer 2 regime, the strongest tested candidate in this sweep is:

- `ratio_candidate_times_anchored_span_over_std`

That is:

- `mean_candidate_count * mean_anchored_alpha_log_span / mean_anchored_alpha_log_std`

With calibration-frozen thresholding on `persistent_mode_informed`, it reached:

- calibration balanced accuracy: `0.6818`
- holdout balanced accuracy: `0.7500`
- confirmation balanced accuracy: `0.6667`
- out-of-sample mean balanced accuracy: `0.7083`

That makes it the strongest current Layer 2 implementation candidate for the informed-bank regime in this repo.

## Outputs

- [backbone_observability_gate_ratio_sweep_leaderboard.csv](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-ratio-sweep/outputs/backbone_observability_gate_ratio_sweep_leaderboard.csv)
- [backbone_observability_gate_ratio_sweep_summary.json](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-ratio-sweep/outputs/backbone_observability_gate_ratio_sweep_summary.json)
- [backbone_observability_gate_ratio_sweep_top_metrics.png](/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-interventions/backbone-observability-gate-ratio-sweep/outputs/figures/backbone_observability_gate_ratio_sweep_top_metrics.png)
