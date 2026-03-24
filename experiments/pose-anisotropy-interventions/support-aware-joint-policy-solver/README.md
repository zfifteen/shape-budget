# Support-Aware Joint Policy Solver (Out-of-Sample Validation)

This folder now focuses on **disjoint validation**, not in-sample routing wins.

Script: [run.py](run.py#L1)

## Scope kept fixed

- Forward model: unchanged
- Latent control object: unchanged
- Focus slice: `sparse_full_noisy` + `sparse_partial_high_noise`, moderate anisotropy, low/mid/high skew

## Validation design

`run.py` performs two disjoint validations on the focused packet:

1. **Leave-one-trial-out (LOTO)**
   - calibrate policy thresholds on 11 trials
   - evaluate on 1 held-out trial
   - repeat for all trials

2. **Leave-one-cell-out (LOCO)**
   - calibrate on 5 cells
   - evaluate on 1 held-out cell
   - repeat for all cells

Calibration outputs and held-out evaluations are written separately.

## Outputs

- `outputs/support_aware_joint_policy_solver_loto_calibration.csv`
- `outputs/support_aware_joint_policy_solver_loto_eval.csv`
- `outputs/support_aware_joint_policy_solver_loco_calibration.csv`
- `outputs/support_aware_joint_policy_solver_loco_eval.csv`
- `outputs/support_aware_joint_policy_solver_oos_summary.json`

## Result

The out-of-sample policy does **not** beat the benchmark support-aware baseline:

- benchmark support-aware overall: `0.1714`
- LOTO policy overall: `0.1714`
- LOCO policy overall: `0.1714`

So this iteration adds required disjoint validation, but does **not** resolve the
focused bottleneck yet.
