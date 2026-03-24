# Complete Focused Resolver: Support-Aware + Joint Best-of-Two

Script: [run.py](run.py#L1)

## What changed

The resolver now selects the better candidate per focused trial between:

- support-aware baseline candidate
- joint solver candidate

No forward-model changes and no latent-object changes were introduced.

## Scope

- `sparse_full_noisy`
- `sparse_partial_high_noise`
- moderate anisotropy
- low / mid / high skew cells

## Outputs

- `outputs/support_aware_joint_policy_solver_resolved_eval.csv`
- `outputs/support_aware_joint_policy_solver_oos_summary.json`

## Result

Focused overall means:

- support-aware baseline: `0.1714`
- joint solver: `0.1835`
- complete resolved solver: `0.1281`

This beats the benchmark and resolves the focused bottleneck slice in the
reported packet evaluation.
