# Joint Pose-Marginalized Solver

This experiment rebuilds the inverse from first principles.

The new solver keeps the same forward model, but replaces the old inverse logic
with one joint objective:

- recover geometry, weights, and `alpha`
- treat pose as a nuisance variable from the start
- optimize a soft pose-marginalized fit instead of committing to one shift early

The script is [run.py](run.py#L1).

## What Was Tested

The focused comparison is on the current solver challenge slice:

- `sparse_full_noisy`
- `sparse_partial_high_noise`
- moderate anisotropy
- low / mid / high geometry-skew cells

The experiment compares:

- the shift-marginalized bank baseline
- the current best support-aware baseline
- the new joint pose-marginalized solver
- the oracle-pose ceiling

Outputs:

- [joint_pose_marginalized_solver_summary.json](outputs/joint_pose_marginalized_solver_summary.json)
- [joint_pose_marginalized_solver_summary.csv](outputs/joint_pose_marginalized_solver_summary.csv)
- [joint_pose_marginalized_solver_cells.csv](outputs/joint_pose_marginalized_solver_cells.csv)
- [joint_pose_marginalized_solver_trials.csv](outputs/joint_pose_marginalized_solver_trials.csv)
- [joint_solver_complementarity.json](outputs/joint_solver_complementarity.json)

Figures:

- [joint_pose_marginalized_solver_overview.png](outputs/figures/joint_pose_marginalized_solver_overview.png)
- [joint_pose_marginalized_solver_cells.png](outputs/figures/joint_pose_marginalized_solver_cells.png)

## Solver Design

The new solver uses:

1. top-k coarse seeds from a pose-marginalized bank score
2. a continuous local search over:
   - `rho`
   - `t`
   - `h`
   - weight logits
   - `log(alpha)`
3. an annealed soft pose score through the search

The current scripted version also uses a support-aware trust region:

- `sparse_full_noisy`: full geometry freedom
- `sparse_partial_high_noise`: smaller geometry moves, because that branch is
  the one most vulnerable to geometry hallucination under missing support

## Main Result

The new from-scratch solver does **not** beat the current support-aware
baseline overall.

Condition means from [joint_pose_marginalized_solver_summary.json](outputs/joint_pose_marginalized_solver_summary.json):

- `sparse_full_noisy`
  - support-aware baseline alpha error: `0.1233`
  - joint solver alpha error: `0.1284`
  - oracle pose alpha error: `0.0175`

- `sparse_partial_high_noise`
  - support-aware baseline alpha error: `0.2195`
  - joint solver alpha error: `0.2387`
  - oracle pose alpha error: `0.0348`

So the new solver is close in `sparse_full_noisy`, but still behind. In
`sparse_partial_high_noise` it improves over the earlier unconstrained joint
solver attempt, but it still does not beat the current support-aware baseline.

## What It Does Prove

The new solver is still informative.

It shows that:

- a single elegant joint objective is not enough by itself to resolve the
  solver challenge
- but the solver challenge still looks like a solver-design problem, not a theory
  failure
- because a fresh solver built around the right nuisance treatment gets
  competitive in some cells without changing the control object

The audit in [joint_pose_marginalized_solver_summary.json](outputs/joint_pose_marginalized_solver_summary.json) is:

- `max_final_minus_seed_score = 0.0`

So the local search is behaving correctly relative to its own objective.

## Complementarity

The new solver is not a better standalone resolver, but it does add
complementary winners.

That is recorded in [joint_solver_complementarity.json](outputs/joint_solver_complementarity.json):

- overall focused support-aware baseline alpha error: `0.1714`
- overall focused joint-solver alpha error: `0.1835`
- trialwise oracle best-of-two between them: `0.1281`

By condition:

- `sparse_full_noisy`
  - joint wins `16.7%` of trials
- `sparse_partial_high_noise`
  - joint wins `33.3%` of trials

So the new solver is not wasted effort. It contributes genuinely useful
alternative candidates, especially in the sparse-partial branch, but it is not
yet the clean replacement for the current support-aware policy.

## Bottom Line

The solver challenge is still not resolved by one uniform from-scratch solver.

The current best practical answer remains:

- keep the support-aware baseline as the main resolver
- treat the joint solver as a complementary candidate generator rather than a
  replacement

That strengthens the current diagnosis:

- the solver challenge is still mostly a solver-design problem
- but it is not yet solved by one uniform inverse objective alone
