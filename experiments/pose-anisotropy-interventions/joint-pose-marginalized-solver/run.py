"""
Post-roadmap extension: joint pose-marginalized solver from first principles.

This experiment keeps the same forward model and nuisance structure, but
rebuilds the inverse around a single idea:

- recover geometry, weights, and alpha by optimizing one pose-marginalized
  objective directly, instead of routing between separate refinement policies

The new solver uses:

1. a coarse global multi-start over a reference bank
2. a continuous local search over geometry, weights, and alpha
3. soft pose marginalization from the beginning, with an annealed temperature
   schedule rather than an early hard shift commitment
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

candidate_conditioned_search, sample_conditioned_parameters, top_k_indices = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "candidate_conditioned_search",
    "sample_conditioned_parameters",
    "top_k_indices",
)

family_switching_refine, = load_symbols(
    "run_family_switching_refinement_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/family-switching-refinement/run.py",
    "family_switching_refine",
)

oracle_align_observation, = load_symbols(
    "run_oracle_alignment_ceiling_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/oracle-alignment-ceiling/run.py",
    "oracle_align_observation",
)

nearest_neighbor_aligned, rmse = load_symbols(
    "run_orientation_locking_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/orientation-locking/run.py",
    "nearest_neighbor_aligned",
    "rmse",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

ALPHA_MAX, ALPHA_MIN, GEOMETRY_BOUNDS, REFERENCE_BANK_SIZE, anisotropic_forward_signature, build_reference_bank, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "GEOMETRY_BOUNDS",
    "REFERENCE_BANK_SIZE",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "symmetry_aware_errors",
)

OBSERVATION_REGIMES, SIGNATURE_ANGLE_COUNT, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "SIGNATURE_ANGLE_COUNT",
    "write_csv",
)

import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.family": "sans-serif",
    }
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

FOCUS_CONDITIONS = ["sparse_full_noisy", "sparse_partial_high_noise"]
FOCUS_ALPHA_BIN = "moderate"
TOP_K_SEEDS = 3
TRIALS_PER_CELL = 2
MIN_SOFTMIN_TEMPERATURE = 1.0e-4
SCALAR_GRID_POINTS = 5
PAIR_GRID_POINTS = 3
ANNEAL_FACTORS = [3.0, 1.0, 0.5]
RHO_RADIUS = 0.024
H_RADIUS = 0.16
T_RADIUS = 0.18
WEIGHT_LOGIT_RADIUS = 0.90
LOG_ALPHA_RADIUS = 0.18
SHRINK = 0.55
WEIGHT_LOGIT_BOUND = 4.0
AUDIT_CASES = 6

GEOMETRY_SKEW_BIN_LABELS = ["low_skew", "mid_skew", "high_skew"]


@dataclass
class TrialRow:
    condition: str
    geometry_skew_bin: str
    trial_in_cell: int
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    marginalized_alpha_error: float
    marginalized_geometry_mae: float
    marginalized_weight_mae: float
    support_gated_alpha_error: float
    support_gated_geometry_mae: float
    support_gated_weight_mae: float
    support_gated_fit_rmse: float
    support_gated_choose_family: int
    joint_alpha_error: float
    joint_geometry_mae: float
    joint_weight_mae: float
    joint_fit_rmse: float
    joint_score: float
    joint_seed_rank: int
    joint_pose_entropy: float
    oracle_pose_alpha_error: float
    oracle_pose_geometry_mae: float
    oracle_pose_weight_mae: float
    oracle_pose_fit_rmse: float


def softmin_temperature(regime: dict[str, float | str | int]) -> float:
    sigma = float(regime["noise_sigma"])
    return max(sigma * sigma, MIN_SOFTMIN_TEMPERATURE)


def score_shift_stack(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shift_stack: np.ndarray,
    temperature: float,
) -> tuple[float, int, float]:
    residual = shift_stack[:, mask] - observed_signature[mask][None, :]
    mse = np.mean(residual * residual, axis=1)
    minima = float(np.min(mse))
    best_shift = int(np.argmin(mse))
    stable = np.exp(-(mse - minima) / temperature)
    posterior = stable / np.sum(stable)
    entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1.0e-12))) / math.log(len(mse)))
    score = minima - temperature * math.log(float(np.mean(stable)))
    return float(score), best_shift, entropy


def marginalized_bank_scores(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    mse = np.mean(residual * residual, axis=2)
    minima = np.min(mse, axis=1, keepdims=True)
    stable = np.exp(-(mse - minima) / temperature)
    scores = minima[:, 0] - temperature * np.log(np.mean(stable, axis=1))
    best_shifts = np.argmin(mse, axis=1)
    return scores, best_shifts


def weights_to_logits(weights: np.ndarray) -> tuple[float, float]:
    w1, w2, w3 = [float(x) for x in weights]
    return float(np.log(w1 / w3)), float(np.log(w2 / w3))


def logits_to_weights(z1: float, z2: float) -> np.ndarray:
    logits = np.array([z1, z2, 0.0], dtype=float)
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def params_to_state(params: tuple[float, float, float, float, float, float]) -> np.ndarray:
    rho, t, h, w1, w2, alpha = params
    weights = np.array([w1, w2, 1.0 - w1 - w2], dtype=float)
    z1, z2 = weights_to_logits(weights)
    return np.array([rho, t, h, z1, z2, math.log(alpha)], dtype=float)


def state_to_params(state: np.ndarray) -> tuple[float, float, float, float, float, float]:
    rho = float(np.clip(state[0], GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(np.clip(state[1], GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(np.clip(state[2], GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    z1 = float(np.clip(state[3], -WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND))
    z2 = float(np.clip(state[4], -WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND))
    beta = float(np.clip(state[5], math.log(ALPHA_MIN), math.log(ALPHA_MAX)))
    weights = logits_to_weights(z1, z2)
    alpha = float(math.exp(beta))
    return rho, t, h, float(weights[0]), float(weights[1]), alpha


def centered_grid(center: float, radius: float, lower: float, upper: float, count: int) -> np.ndarray:
    values = np.linspace(max(lower, center - radius), min(upper, center + radius), count)
    return np.unique(np.concatenate([values, np.array([center], dtype=float)]))


def solver_profile(condition: str) -> dict[str, float]:
    if condition == "sparse_partial_high_noise":
        return {
            "rho_scale": 0.30,
            "h_scale": 0.35,
            "t_scale": 0.30,
            "weight_scale": 0.50,
            "alpha_scale": 1.10,
        }
    return {
        "rho_scale": 1.0,
        "h_scale": 1.0,
        "t_scale": 1.0,
        "weight_scale": 1.0,
        "alpha_scale": 1.0,
    }


def joint_grid(
    state: np.ndarray,
    dims: tuple[int, int],
    radii: tuple[float, float],
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> list[np.ndarray]:
    grid_a = centered_grid(float(state[dims[0]]), radii[0], bounds[0][0], bounds[0][1], PAIR_GRID_POINTS)
    grid_b = centered_grid(float(state[dims[1]]), radii[1], bounds[1][0], bounds[1][1], PAIR_GRID_POINTS)
    states: list[np.ndarray] = []
    for value_a in grid_a:
        for value_b in grid_b:
            candidate = state.copy()
            candidate[dims[0]] = float(value_a)
            candidate[dims[1]] = float(value_b)
            states.append(candidate)
    return states


def scalar_grid(
    state: np.ndarray,
    dim: int,
    radius: float,
    lower: float,
    upper: float,
) -> list[np.ndarray]:
    values = centered_grid(float(state[dim]), radius, lower, upper, SCALAR_GRID_POINTS)
    states: list[np.ndarray] = []
    for value in values:
        candidate = state.copy()
        candidate[dim] = float(value)
        states.append(candidate)
    return states


class SolverContext:
    def __init__(self, observed_signature: np.ndarray, mask: np.ndarray):
        self.observed_signature = observed_signature
        self.mask = mask
        self.cache: dict[tuple[tuple[float, float, float, float, float, float], float], tuple[float, np.ndarray, int, float]] = {}

    def score_params(
        self,
        params: tuple[float, float, float, float, float, float],
        temperature: float,
    ) -> tuple[float, np.ndarray, int, float]:
        key = (tuple(float(x) for x in params), float(temperature))
        if key in self.cache:
            return self.cache[key]
        signature = anisotropic_forward_signature(params)
        shift_stack = np.stack([np.roll(signature, shift) for shift in range(len(signature))], axis=0)
        score, best_shift, entropy = score_shift_stack(
            self.observed_signature,
            self.mask,
            shift_stack,
            temperature,
        )
        result = (float(score), shift_stack[best_shift], int(best_shift), float(entropy))
        self.cache[key] = result
        return result


def improve_over_candidates(
    context: SolverContext,
    temperature: float,
    current_state: np.ndarray,
    candidates: list[np.ndarray],
) -> tuple[np.ndarray, tuple[float, np.ndarray, int, float]]:
    best_state = current_state.copy()
    best_params = state_to_params(current_state)
    best_eval = context.score_params(best_params, temperature)
    for candidate_state in candidates:
        params = state_to_params(candidate_state)
        evaluation = context.score_params(params, temperature)
        if evaluation[0] + 1.0e-12 < best_eval[0]:
            best_state = candidate_state.copy()
            best_eval = evaluation
    return best_state, best_eval


def joint_pose_marginalized_refine(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    seed_params: tuple[float, float, float, float, float, float],
    base_temperature: float,
    condition: str,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, int, float, float]:
    context = SolverContext(observed_signature, mask)
    state = params_to_state(seed_params)
    profile = solver_profile(condition)

    rho_radius = RHO_RADIUS * float(profile["rho_scale"])
    h_radius = H_RADIUS * float(profile["h_scale"])
    t_radius = T_RADIUS * float(profile["t_scale"])
    weight_radius = WEIGHT_LOGIT_RADIUS * float(profile["weight_scale"])
    beta_radius = LOG_ALPHA_RADIUS * float(profile["alpha_scale"])

    for factor in ANNEAL_FACTORS:
        temperature = max(base_temperature * factor, MIN_SOFTMIN_TEMPERATURE)

        state, _ = improve_over_candidates(
            context,
            temperature,
            state,
            scalar_grid(state, 0, rho_radius, GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]),
        )
        state, _ = improve_over_candidates(
            context,
            temperature,
            state,
            scalar_grid(state, 2, h_radius, GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]),
        )
        state, _ = improve_over_candidates(
            context,
            temperature,
            state,
            joint_grid(
                state,
                (1, 5),
                (t_radius, beta_radius),
                (
                    (GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]),
                    (math.log(ALPHA_MIN), math.log(ALPHA_MAX)),
                ),
            ),
        )
        state, _ = improve_over_candidates(
            context,
            temperature,
            state,
            joint_grid(
                state,
                (3, 4),
                (weight_radius, weight_radius),
                (
                    (-WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND),
                    (-WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND),
                ),
            ),
        )

        rho_radius *= SHRINK
        h_radius *= SHRINK
        t_radius *= SHRINK
        weight_radius *= SHRINK
        beta_radius *= SHRINK

    final_params = state_to_params(state)
    final_score, final_signature, final_shift, final_entropy = context.score_params(final_params, base_temperature)
    return final_params, final_signature, final_shift, float(final_score), float(final_entropy)


def choose_support_gated_baseline(
    condition: str,
    conditioned_params: tuple[float, float, float, float, float, float],
    conditioned_signature: np.ndarray,
    conditioned_score: float,
    family_params: tuple[float, float, float, float, float, float],
    family_signature: np.ndarray,
    family_score: float,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, float, int]:
    if condition == "sparse_partial_high_noise":
        return conditioned_params, conditioned_signature, float(conditioned_score), 0
    if family_score + 1.0e-12 < conditioned_score:
        return family_params, family_signature, float(family_score), 1
    return conditioned_params, conditioned_signature, float(conditioned_score), 0


def summarize_by_condition(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        subset = [row for row in rows if row.condition == condition]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        support_alpha = mean("support_gated_alpha_error")
        joint_alpha = mean("joint_alpha_error")
        oracle_alpha = mean("oracle_pose_alpha_error")
        support_headroom = support_alpha - oracle_alpha
        support_to_joint = support_alpha / max(joint_alpha, 1.0e-12)

        if support_headroom > 1.0e-6:
            joint_fraction = float((support_alpha - joint_alpha) / support_headroom)
        else:
            joint_fraction = float("nan")

        summary.append(
            {
                "condition": condition,
                "marginalized_alpha_error_mean": mean("marginalized_alpha_error"),
                "support_gated_alpha_error_mean": support_alpha,
                "joint_alpha_error_mean": joint_alpha,
                "oracle_pose_alpha_error_mean": oracle_alpha,
                "joint_vs_support_gated_alpha_ratio": support_to_joint,
                "joint_fraction_of_support_to_oracle_gap_closed": joint_fraction,
                "joint_seed_rank_mean": mean("joint_seed_rank"),
                "joint_pose_entropy_mean": mean("joint_pose_entropy"),
                "support_gated_choose_family_fraction": mean("support_gated_choose_family"),
            }
        )
    return summary


def summarize_by_cell(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            subset = [row for row in rows if row.condition == condition and row.geometry_skew_bin == skew_bin]
            if not subset:
                continue

            def mean(attr: str) -> float:
                return float(np.mean([getattr(row, attr) for row in subset]))

            support_alpha = mean("support_gated_alpha_error")
            joint_alpha = mean("joint_alpha_error")
            summary.append(
                {
                    "condition": condition,
                    "alpha_strength_bin": FOCUS_ALPHA_BIN,
                    "geometry_skew_bin": skew_bin,
                    "count": len(subset),
                    "marginalized_alpha_error_mean": mean("marginalized_alpha_error"),
                    "support_gated_alpha_error_mean": support_alpha,
                    "joint_alpha_error_mean": joint_alpha,
                    "oracle_pose_alpha_error_mean": mean("oracle_pose_alpha_error"),
                    "joint_vs_support_gated_alpha_ratio": float(support_alpha / max(joint_alpha, 1.0e-12)),
                    "joint_pose_entropy_mean": mean("joint_pose_entropy"),
                }
            )
    return summary


def plot_overview(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(row["condition"]) for row in summary_rows]
    x = np.arange(len(conditions))
    width = 0.22

    marginalized = np.array([float(row["marginalized_alpha_error_mean"]) for row in summary_rows])
    support_gated = np.array([float(row["support_gated_alpha_error_mean"]) for row in summary_rows])
    joint = np.array([float(row["joint_alpha_error_mean"]) for row in summary_rows])
    oracle = np.array([float(row["oracle_pose_alpha_error_mean"]) for row in summary_rows])

    fig, ax = plt.subplots(figsize=(12.8, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.08, right=0.98)

    ax.bar(x - 1.5 * width, marginalized, width=width, color="#1d3557", label="marginalized bank")
    ax.bar(x - 0.5 * width, support_gated, width=width, color="#2a9d8f", label="current support-aware baseline")
    ax.bar(x + 0.5 * width, joint, width=width, color="#e76f51", label="joint first-principles solver")
    ax.bar(x + 1.5 * width, oracle, width=width, color="#6a4c93", label="oracle pose")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Focused bottleneck alpha recovery")
    ax.legend(loc="upper right", frameon=True, ncol=2)

    fig.suptitle(
        "Joint Pose-Marginalized Solver A: From-Scratch Inverse On The Bottleneck Slice",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cells(path: str, cell_rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(2, len(FOCUS_CONDITIONS), figsize=(12.4, 7.2), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.98, wspace=0.24, hspace=0.30)

    for col_idx, condition in enumerate(FOCUS_CONDITIONS):
        ratio_matrix = np.full((1, len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
        entropy_matrix = np.full((1, len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
        for row in cell_rows:
            if str(row["condition"]) != condition:
                continue
            j = GEOMETRY_SKEW_BIN_LABELS.index(str(row["geometry_skew_bin"]))
            ratio_matrix[0, j] = float(row["joint_vs_support_gated_alpha_ratio"])
            entropy_matrix[0, j] = float(row["joint_pose_entropy_mean"])

        sns.heatmap(
            ratio_matrix,
            ax=axes[0, col_idx],
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=GEOMETRY_SKEW_BIN_LABELS,
            yticklabels=[FOCUS_ALPHA_BIN],
            cbar=(col_idx == len(FOCUS_CONDITIONS) - 1),
            cbar_kws={"label": "support-aware / joint alpha error"} if col_idx == len(FOCUS_CONDITIONS) - 1 else None,
            vmin=0.6,
            vmax=1.8,
        )
        axes[0, col_idx].set_title(f"{condition}\nalpha improvement factor")
        axes[0, col_idx].set_xlabel("geometry skew |t| bin")
        if col_idx == 0:
            axes[0, col_idx].set_ylabel("anisotropy strength")
        else:
            axes[0, col_idx].set_ylabel("")

        sns.heatmap(
            entropy_matrix,
            ax=axes[1, col_idx],
            cmap="magma_r",
            annot=True,
            fmt=".2f",
            xticklabels=GEOMETRY_SKEW_BIN_LABELS,
            yticklabels=[FOCUS_ALPHA_BIN],
            cbar=(col_idx == len(FOCUS_CONDITIONS) - 1),
            cbar_kws={"label": "normalized pose entropy"} if col_idx == len(FOCUS_CONDITIONS) - 1 else None,
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, col_idx].set_title(f"{condition}\nfinal pose entropy")
        axes[1, col_idx].set_xlabel("geometry skew |t| bin")
        if col_idx == 0:
            axes[1, col_idx].set_ylabel("anisotropy strength")
        else:
            axes[1, col_idx].set_ylabel("")

    fig.suptitle(
        "Joint Pose-Marginalized Solver B: Cell-Level Gain And Pose Uncertainty",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def audit_non_degradation(
    rng: np.random.Generator,
    bank_params: list[tuple[float, float, float, float, float, float]],
    shifted_bank: np.ndarray,
) -> dict[str, float]:
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}
    max_final_minus_seed_score = 0.0
    for _ in range(AUDIT_CASES):
        regime = regime_map[FOCUS_CONDITIONS[int(rng.integers(0, len(FOCUS_CONDITIONS)))]]
        temperature = softmin_temperature(regime)
        seed_idx = int(rng.integers(0, len(bank_params)))
        clean_signature = anisotropic_forward_signature(bank_params[seed_idx])
        _, observed_signature, mask, _ = observe_pose_free_signature(clean_signature, regime, rng)
        seed_params = bank_params[seed_idx]
        context = SolverContext(observed_signature, mask)
        seed_score = context.score_params(seed_params, temperature)[0]
        _, _, _, final_score, _ = joint_pose_marginalized_refine(
            observed_signature,
            mask,
            seed_params,
            temperature,
            str(regime["name"]),
        )
        max_final_minus_seed_score = max(max_final_minus_seed_score, float(final_score - seed_score))
    return {
        "audit_cases": float(AUDIT_CASES),
        "max_final_minus_seed_score": float(max_final_minus_seed_score),
    }


def main() -> None:
    rng = np.random.default_rng(20260324)

    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}

    audits = {
        "local_non_degradation": audit_non_degradation(np.random.default_rng(20260324), bank_params, shifted_bank),
    }

    rows: list[TrialRow] = []
    for condition in FOCUS_CONDITIONS:
        regime = regime_map[condition]
        temperature = softmin_temperature(regime)
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            for trial_idx in range(TRIALS_PER_CELL):
                true_params = sample_conditioned_parameters(rng, FOCUS_ALPHA_BIN, skew_bin)
                clean_signature = anisotropic_forward_signature(true_params)
                rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

                marginalized_scores, marginalized_best_shifts = marginalized_bank_scores(
                    observed_signature,
                    mask,
                    shifted_bank,
                    temperature,
                )
                marginalized_idx = int(np.argmin(marginalized_scores))
                marginalized_params = bank_params[marginalized_idx]
                marginalized_signature = shifted_bank[marginalized_idx, int(marginalized_best_shifts[marginalized_idx])]
                marginalized_geometry, marginalized_weight, marginalized_alpha = symmetry_aware_errors(
                    true_params,
                    marginalized_params,
                )

                conditioned_best_params = marginalized_params
                conditioned_best_signature = marginalized_signature
                conditioned_best_score = float("inf")

                family_best_params = marginalized_params
                family_best_signature = marginalized_signature
                family_best_score = float("inf")

                joint_best_params = marginalized_params
                joint_best_signature = marginalized_signature
                joint_best_score = float("inf")
                joint_best_entropy = 1.0
                joint_seed_rank = 1

                seed_indices = top_k_indices(marginalized_scores, TOP_K_SEEDS)
                for seed_rank, idx in enumerate(seed_indices, start=1):
                    seed_params = bank_params[idx]

                    conditioned_params, conditioned_signature, _, conditioned_score = candidate_conditioned_search(
                        observed_signature,
                        mask,
                        seed_params,
                        temperature,
                    )
                    if conditioned_score < conditioned_best_score:
                        conditioned_best_score = conditioned_score
                        conditioned_best_params = conditioned_params
                        conditioned_best_signature = conditioned_signature

                    family_params, family_signature, _, family_score = family_switching_refine(
                        observed_signature,
                        mask,
                        seed_params,
                        temperature,
                    )
                    if family_score < family_best_score:
                        family_best_score = family_score
                        family_best_params = family_params
                        family_best_signature = family_signature

                    joint_params, joint_signature, _, joint_score, joint_entropy = joint_pose_marginalized_refine(
                        observed_signature,
                        mask,
                        seed_params,
                        temperature,
                        condition,
                    )
                    if joint_score < joint_best_score:
                        joint_best_score = joint_score
                        joint_best_params = joint_params
                        joint_best_signature = joint_signature
                        joint_best_entropy = joint_entropy
                        joint_seed_rank = seed_rank

                support_params, support_signature, _, support_choose_family = choose_support_gated_baseline(
                    condition,
                    conditioned_best_params,
                    conditioned_best_signature,
                    conditioned_best_score,
                    family_best_params,
                    family_best_signature,
                    family_best_score,
                )

                support_geometry, support_weight, support_alpha = symmetry_aware_errors(true_params, support_params)
                support_fit_rmse = rmse(support_signature, rotated_signature)

                joint_geometry, joint_weight, joint_alpha = symmetry_aware_errors(true_params, joint_best_params)
                joint_fit_rmse = rmse(joint_best_signature, rotated_signature)

                oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
                oracle_pose_params, oracle_pose_signature = nearest_neighbor_aligned(
                    oracle_observed,
                    oracle_mask,
                    bank_signatures,
                    bank_params,
                )
                oracle_geometry, oracle_weight, oracle_alpha = symmetry_aware_errors(true_params, oracle_pose_params)
                oracle_fit_rmse = rmse(oracle_pose_signature, clean_signature)

                rows.append(
                    TrialRow(
                        condition=condition,
                        geometry_skew_bin=skew_bin,
                        trial_in_cell=trial_idx,
                        true_alpha=float(true_params[5]),
                        true_t=float(true_params[1]),
                        true_rotation_shift=int(true_shift),
                        marginalized_alpha_error=float(marginalized_alpha),
                        marginalized_geometry_mae=float(marginalized_geometry),
                        marginalized_weight_mae=float(marginalized_weight),
                        support_gated_alpha_error=float(support_alpha),
                        support_gated_geometry_mae=float(support_geometry),
                        support_gated_weight_mae=float(support_weight),
                        support_gated_fit_rmse=float(support_fit_rmse),
                        support_gated_choose_family=int(support_choose_family),
                        joint_alpha_error=float(joint_alpha),
                        joint_geometry_mae=float(joint_geometry),
                        joint_weight_mae=float(joint_weight),
                        joint_fit_rmse=float(joint_fit_rmse),
                        joint_score=float(joint_best_score),
                        joint_seed_rank=int(joint_seed_rank),
                        joint_pose_entropy=float(joint_best_entropy),
                        oracle_pose_alpha_error=float(oracle_alpha),
                        oracle_pose_geometry_mae=float(oracle_geometry),
                        oracle_pose_weight_mae=float(oracle_weight),
                        oracle_pose_fit_rmse=float(oracle_fit_rmse),
                    )
                )

    trial_rows = [row.__dict__ for row in rows]
    by_condition = summarize_by_condition(rows)
    by_cell = summarize_by_cell(rows)

    focus_summary = {
        "best_joint_vs_support_gated_alpha_ratio": float(
            max(float(row["joint_vs_support_gated_alpha_ratio"]) for row in by_condition)
        ),
        "worst_joint_vs_support_gated_alpha_ratio": float(
            min(float(row["joint_vs_support_gated_alpha_ratio"]) for row in by_condition)
        ),
        "best_cell_joint_vs_support_gated_alpha_ratio": float(
            max(float(row["joint_vs_support_gated_alpha_ratio"]) for row in by_cell)
        ),
        "worst_cell_joint_vs_support_gated_alpha_ratio": float(
            min(float(row["joint_vs_support_gated_alpha_ratio"]) for row in by_cell)
        ),
        "lowest_joint_pose_entropy_mean": float(
            min(float(row["joint_pose_entropy_mean"]) for row in by_cell)
        ),
        "highest_joint_pose_entropy_mean": float(
            max(float(row["joint_pose_entropy_mean"]) for row in by_cell)
        ),
    }

    write_csv(os.path.join(OUTPUT_DIR, "joint_pose_marginalized_solver_trials.csv"), trial_rows)
    write_csv(os.path.join(OUTPUT_DIR, "joint_pose_marginalized_solver_summary.csv"), by_condition)
    write_csv(os.path.join(OUTPUT_DIR, "joint_pose_marginalized_solver_cells.csv"), by_cell)

    plot_overview(os.path.join(FIGURE_DIR, "joint_pose_marginalized_solver_overview.png"), by_condition)
    plot_cells(os.path.join(FIGURE_DIR, "joint_pose_marginalized_solver_cells.png"), by_cell)

    output = {
        "summary": {
            "reference_bank_size": REFERENCE_BANK_SIZE,
            "top_k_seeds": float(TOP_K_SEEDS),
            "trials_per_cell": float(TRIALS_PER_CELL),
            "anneal_factors": [float(x) for x in ANNEAL_FACTORS],
            "audits": audits,
            "focus_summary": focus_summary,
        },
        "by_condition": by_condition,
        "by_cell": by_cell,
    }

    with open(os.path.join(OUTPUT_DIR, "joint_pose_marginalized_solver_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
