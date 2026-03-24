# Support-Calibrated Competitive Solver

This experiment implements the next solver iteration for issue #1.

It stays within the issue constraints:

- same latent control object
- same forward model
- solver-design change only
- focused on the actual bottleneck slice

The new solver is a **support-calibrated competitive resolver**.

It still runs the two strongest existing refinement paths:

- fixed-family candidate-conditioned cleanup
- geometry-plus-alpha family switching

But it does **not** let raw score competition decide every case the same way.
Instead it applies a small support-sensitive family penalty in the sparse-partial branch.

That is the whole point of the change:

- keep the family path where geometry freedom helps
- suppress it where degraded support makes that freedom misroute the solve

## Files

- `run.py`
- `outputs/support_calibrated_competitive_solver_summary.json`
- `outputs/support_calibrated_competitive_solver_summary.csv`
- `outputs/support_calibrated_competitive_solver_cells.csv`
- `outputs/support_calibrated_competitive_solver_trials.csv`

## Execution

Syntax check:

```bash
python3 -m py_compile experiments/pose-anisotropy-interventions/support-calibrated-competitive-solver/run.py
```

Run:

```bash
python3 experiments/pose-anisotropy-interventions/support-calibrated-competitive-solver/run.py
```

## Focused same-trial result

From `support_calibrated_competitive_solver_summary.json`:

- support-aware baseline mean alpha error: `0.1397`
- direct competitive hybrid mean alpha error: `0.1367`
- calibrated solver mean alpha error: `0.1216`
- oracle best-of-two mean alpha error: `0.1174`

So this focused run:

- beats the issue minimum target of `0.1835`
- beats the issue primary target of `0.1714`
- closes most of the remaining gap toward the local oracle packet

## What changed

The new resolver keeps the two-path competition, but changes the selection rule.

In plain language:

- `sparse_full_noisy`: keep geometry freedom alive
- `sparse_partial_high_noise`: make family switching pay a penalty before it can beat the fixed-family path

That gives the solver a cleaner answer to the bottleneck the issue called out:

- sparse-full tends to want the family path
- sparse-partial tends to want the fixed-family conditioned path
- the remaining problem is mostly path selection under degraded support

## Plain-language read

This result strengthens the current BGP read.

It does **not** look like a failure of the control object.
It still looks like a solver-design bottleneck, specifically a support-sensitive routing problem between two already useful refinement paths.

## Note on execution scale

This local run uses a reduced focused packet for tractable execution inside the container:

- smaller reference bank
- smaller per-cell trial count
- same focused regimes and same solver-design question

So the exact numbers here are from a compact packet, but the design change itself is isolated and ready for fuller reruns in the repo environment.