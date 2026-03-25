# Bank-Invariant Solver

## Goal

Build a solver that stays reliable across fresh bank seeds on the focused
pose-free anisotropic slice, by replacing bank-dependent absolute features
with bank-invariant differential features in the ridge chooser.

## Motivation

The [bank-adaptive solver](../bank-adaptive-solver/README.md) cleared holdout
under the density fallback but failed fresh-bank confirmation.  The root cause
is that the ridge chooser's features include absolute bank-dependent scores
(`support_score`, `joint_score`, `support_entropy`, `joint_entropy`, etc.)
that shift when the reference bank changes.  The frozen chooser weights,
calibrated to one bank distribution, produce incorrect routing decisions on a
fresh bank.

## Key Design Change

This experiment replaces all absolute score features with **differential**
features (support minus joint, or joint minus support).  Differentials are
first-order invariant to additive bank-induced shifts: when the bank changes,
both candidates' scores shift together, but their difference is preserved.

The routing decision only needs to know which candidate is better, not the
absolute quality level.  Differential features retain all the information
needed for routing while discarding the bank-specific baseline that causes
instability.

### Feature Vector

| Feature | Definition | Bank invariance |
| --- | --- | --- |
| Cell one-hot (×6) | Condition × skew bin | Deterministic |
| `log_alpha_diff` | `log(joint_α) − log(support_α)` | Parameter-based |
| `t_diff` | `joint_t − support_t` | Parameter-based |
| `rho_diff` | `joint_ρ − support_ρ` | Parameter-based |
| `h_diff` | `joint_h − support_h` | Parameter-based |
| `w1_diff` | `joint_w₁ − support_w₁` | Parameter-based |
| `w2_diff` | `joint_w₂ − support_w₂` | Parameter-based |
| `score_diff` | `joint_score − support_score` | First-order invariant |
| `entropy_diff` | `joint_entropy − support_entropy` | First-order invariant |
| `cv_score_diff` | `joint_cv − support_cv` | First-order invariant |
| `abs_log_alpha_diff` | `|log(joint_α) − log(support_α)|` | Magnitude of disagreement |
| `abs_t_diff` | `|joint_t − support_t|` | Magnitude of disagreement |
| `score_sign` | `sign(joint_score − support_score)` | Rank-based (fully invariant) |

Compare to the bank-adaptive solver which uses absolute features like
`support_score`, `joint_score`, `support_entropy`, `joint_entropy`, etc.

## Evaluation Ladder

The evaluation ladder is identical to the bank-adaptive solver:

1. Generate one cache table per block × variant.
2. Fit one ridge chooser on calibration blocks only.
3. Freeze the chooser before any holdout evaluation.
4. Validate on holdout block 1.
5. If holdout passes, validate on a fresh-bank confirmation block.
6. If baseline fails holdout, run one density fallback branch.

## Hard Constraints Respected

- Disjoint calibration vs holdout.
- All chooser weights frozen before holdout.
- No true latent errors used at evaluation time.
- No routing by regime label alone (cell one-hot is combined with observable features).
- No copied outputs from older experiments.
- One fallback branch only.

## How To Run

```bash
# Full plan (baseline ladder + density fallback if needed)
python experiments/pose-anisotropy-interventions/bank-invariant-solver/run.py run-full

# Single variant ladder
python experiments/pose-anisotropy-interventions/bank-invariant-solver/run.py run-ladder --variant baseline

# Generate one block cache
python experiments/pose-anisotropy-interventions/bank-invariant-solver/run.py generate-block --block calibration_block_1 --variant baseline
```

## Results

### Baseline Variant (bank_size=300, TOP_K_SEEDS=3)

| Split | Support | Joint | Chooser | Outcome |
| --- | ---: | ---: | ---: | --- |
| Calibration | `0.1862` | `0.1596` | `0.1403` | beats both |
| Holdout block 1 | `0.1273` | `0.1180` | `0.1152` | beats both |
| Confirmation block | `0.1319` | `0.1773` | `0.1674` | beats joint, loses to support |

The bank-invariant chooser cleared holdout (beating both candidates) but
failed the fresh-bank confirmation block: it beat the joint candidate
(`0.1674 < 0.1773`) but not the support-aware baseline (`0.1674 > 0.1319`).

Compared to the bank-adaptive solver:

- The bank-adaptive baseline failed holdout entirely (beat support but lost
  to joint).
- The bank-invariant baseline **cleared holdout** by beating both candidates —
  a stronger result than the bank-adaptive baseline.
- Both approaches still fail fresh-bank confirmation.

### Interpretation

The differential feature redesign improved bank stability enough to clear
holdout, which the bank-adaptive solver's baseline could not do.  But the
confirmation failure shows that bank sensitivity remains, albeit reduced.

The result narrows the remaining solver bottleneck further: differential
features help, but are not sufficient alone for confirmation-stable routing.

Relevant reports:

- [ladder summary](outputs/reports/baseline__ladder_summary.json)
- [full plan result](outputs/reports/baseline__full_plan_result.json)

## Outputs

- `outputs/cache/` — cached trial tables per block × variant
- `outputs/models/` — frozen ridge chooser artifacts
- `outputs/reports/` — ladder summaries, prediction CSVs, full plan results
