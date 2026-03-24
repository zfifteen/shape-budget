# Support-Aware Joint Policy Solver

This experiment implements the next solver iteration for the focused anisotropic
inverse bottleneck slice without changing the forward model or latent control
object.

The script is [run.py](run.py#L1).

## Solver design

This iteration keeps both existing candidate generators and changes only the
solver policy:

- support-aware baseline candidate
- joint pose-marginalized candidate
- reliability-aware policy gate using observable signals:
  - joint pose entropy
  - support-vs-joint fit RMSE comparison
  - condition-specific thresholds for sparse-full vs sparse-partial

The policy is intentionally conservative in `sparse_partial_high_noise` and more
permissive in `sparse_full_noisy`.

## Inputs

The script evaluates on the same-trial packet from the current joint-solver
experiment:

- `../joint-pose-marginalized-solver/outputs/joint_pose_marginalized_solver_trials.csv`

## Outputs

- [support_aware_joint_policy_solver_summary.json](outputs/support_aware_joint_policy_solver_summary.json)
- [support_aware_joint_policy_solver_summary.csv](outputs/support_aware_joint_policy_solver_summary.csv)
- [support_aware_joint_policy_solver_cells.csv](outputs/support_aware_joint_policy_solver_cells.csv)
- [support_aware_joint_policy_solver_trials.csv](outputs/support_aware_joint_policy_solver_trials.csv)
- [support_aware_joint_policy_complementarity.json](outputs/support_aware_joint_policy_complementarity.json)
- [support_aware_joint_policy_solver_overview.svg](outputs/figures/support_aware_joint_policy_solver_overview.svg)

## Main result

On the focused same-trial packet, this policy solver improves over both:

- the standalone joint solver mean (`0.1835` in issue context)
- the same-trial support-aware baseline mean (`0.1714` in issue context)

Generated packet result:

- focused support-aware mean alpha error: `0.1714`
- focused joint mean alpha error: `0.1835`
- focused new policy mean alpha error: `0.1399`
- focused oracle best-of-two mean alpha error: `0.1281`

## Plain-language interpretation

What changed:

- the solver now routes between two existing inverse candidates using
  reliability-aware gating instead of using either resolver alone.

What happened:

- the routing policy avoids many sparse-partial over-switches while keeping
  selected joint wins, reducing overall focused alpha error.

BGP read impact:

- this **strengthens** the current BGP read that the unresolved bottleneck is a
  solver-design and routing problem, not a control-object theory failure.
