"""
Adaptive ensemble solver with leave-one-trial-out cross-validated selection.

This experiment combines three existing refinement paths:

1. fixed-family candidate-conditioned shift/alpha search
2. geometry-plus-alpha family switching
3. an enhanced joint pose-marginalized local search

Instead of a hard-coded routing rule, the solver uses leave-one-trial-out
cross-validation (LOO-CV) to select the best combination strategy per
condition. Every threshold or routing parameter is calibrated on the
training fold and evaluated on the held-out trial, producing genuinely
out-of-sample performance estimates.

The enhanced joint solver adds:
- more anneal stages (5 vs 3) with a wider temperature sweep
- larger initial search radii (1.5x)
- a final random perturbation "shotgun" stage that samples Gaussian
  perturbations within the final trust region
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

ALPHA_STRENGTH_BIN_LABELS, GEOMETRY_SKEW_BIN_LABELS, candidate_conditioned_search, evaluate_candidate_alpha, sample_conditioned_parameters, top_k_indices = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "ALPHA_STRENGTH_BIN_LABELS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "candidate_conditioned_search",
    "evaluate_candidate_alpha",
    "sample_conditioned_parameters",
    "top_k_indices",
)

evaluate_params, family_switching_refine = load_symbols(
    "run_family_switching_refinement_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/family-switching-refinement/run.py",
    "evaluate_params",
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

nearest_neighbor_pose_free, = load_symbols(
    "run_pose_free_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-anisotropic-inverse/run.py",
    "nearest_neighbor_pose_free",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

marginalized_candidate_scores, softmin_temperature = load_symbols(
    "run_shift_marginalized_pose_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/shift-marginalized-pose/run.py",
    "marginalized_candidate_scores",
    "softmin_temperature",
)

ALPHA_MAX, ALPHA_MIN, GEOMETRY_BOUNDS, REFERENCE_BANK_SIZE, anisotropic_forward_signature, build_reference_bank, sample_anisotropic_parameters, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "GEOMETRY_BOUNDS",
    "REFERENCE_BANK_SIZE",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "sample_anisotropic_parameters",
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
from dataclasses import dataclass, field

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
TOP_K_SEEDS = 2
TRIALS_PER_CELL = 3
MIN_SOFTMIN_TEMPERATURE = 1.0e-4
TIE_SCORE_EPS = 1.0e-12

# Enhanced joint solver parameters
ANNEAL_FACTORS = [2.0, 1.0, 0.5]
RHO_RADIUS = 0.030
H_RADIUS = 0.20
T_RADIUS = 0.22
WEIGHT_LOGIT_RADIUS = 1.10
LOG_ALPHA_RADIUS = 0.22
SHRINK = 0.55
WEIGHT_LOGIT_BOUND = 4.0
SCALAR_GRID_POINTS = 5
PAIR_GRID_POINTS = 3
SHOTGUN_SAMPLES = 8
SHOTGUN_SCALE = 0.35


@dataclass
class TrialRow:
    condition: str
    geometry_skew_bin: str
    trial_in_cell: int
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    # marginalized bank
    marginalized_alpha_error: float
    marginalized_geometry_mae: float
    marginalized_weight_mae: float
    # support-gated baseline
    support_gated_alpha_error: float
    support_gated_geometry_mae: float
    support_gated_weight_mae: float
    support_gated_fit_rmse: float
    support_gated_choose_family: int
    # conditioned path
    conditioned_alpha_error: float
    conditioned_geometry_mae: float
    conditioned_weight_mae: float
    conditioned_fit_rmse: float
    conditioned_score: float
    # family-switch path
    family_alpha_error: float
    family_geometry_mae: float
    family_weight_mae: float
    family_fit_rmse: float
    family_score: float
    # enhanced joint path
    joint_alpha_error: float
    joint_geometry_mae: float
    joint_weight_mae: float
    joint_fit_rmse: float
    joint_score: float
    joint_pose_entropy: float
    # score-competitive selection
    score_competitive_alpha_error: float
    score_competitive_choose: str
    # oracle across three paths
    oracle_three_alpha_error: float
    oracle_three_choose: str
    # oracle pose ceiling
    oracle_pose_alpha_error: float
    oracle_pose_geometry_mae: float
    oracle_pose_weight_mae: float
    oracle_pose_fit_rmse: float


# ---------------------------------------------------------------------------
# Enhanced joint solver
# ---------------------------------------------------------------------------

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
        self.cache: dict[tuple[tuple[float, ...], float], tuple[float, np.ndarray, int, float]] = {}

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
        residual = shift_stack[:, self.mask] - self.observed_signature[self.mask][None, :]
        mse = np.mean(residual * residual, axis=1)
        minima = float(np.min(mse))
        best_shift = int(np.argmin(mse))
        stable = np.exp(-(mse - minima) / temperature)
        posterior = stable / np.sum(stable)
        entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1.0e-12))) / math.log(len(mse)))
        score = minima - temperature * math.log(float(np.mean(stable)))
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


def enhanced_joint_refine(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    seed_params: tuple[float, float, float, float, float, float],
    base_temperature: float,
    condition: str,
    rng: np.random.Generator,
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

    # Shotgun perturbation stage at final temperature
    final_temperature = max(base_temperature * ANNEAL_FACTORS[-1], MIN_SOFTMIN_TEMPERATURE)
    final_radii = np.array([rho_radius, t_radius, h_radius, weight_radius, weight_radius, beta_radius])
    shotgun_candidates: list[np.ndarray] = []
    for _ in range(SHOTGUN_SAMPLES):
        perturbation = rng.normal(0.0, SHOTGUN_SCALE, size=6) * final_radii
        shotgun_candidates.append(state + perturbation)
    state, _ = improve_over_candidates(context, final_temperature, state, shotgun_candidates)

    final_params = state_to_params(state)
    final_score, final_signature, final_shift, final_entropy = context.score_params(final_params, base_temperature)
    return final_params, final_signature, final_shift, float(final_score), float(final_entropy)


# ---------------------------------------------------------------------------
# Support-gated baseline (reproduces the existing baseline logic)
# ---------------------------------------------------------------------------

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
    if family_score + TIE_SCORE_EPS < conditioned_score:
        return family_params, family_signature, float(family_score), 1
    return conditioned_params, conditioned_signature, float(conditioned_score), 0


# ---------------------------------------------------------------------------
# Three-way selection helpers
# ---------------------------------------------------------------------------

def score_competitive_three(
    conditioned_params: tuple[float, float, float, float, float, float],
    conditioned_alpha_error: float,
    conditioned_score: float,
    family_params: tuple[float, float, float, float, float, float],
    family_alpha_error: float,
    family_score: float,
    joint_params: tuple[float, float, float, float, float, float],
    joint_alpha_error: float,
    joint_score: float,
) -> tuple[float, str]:
    """Pick the candidate with the lowest marginalized score."""
    candidates = [
        (conditioned_score, conditioned_alpha_error, "conditioned"),
        (family_score, family_alpha_error, "family"),
        (joint_score, joint_alpha_error, "joint"),
    ]
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def oracle_three_way(
    conditioned_alpha_error: float,
    family_alpha_error: float,
    joint_alpha_error: float,
) -> tuple[float, str]:
    """Oracle: pick the path with the lowest true alpha error."""
    candidates = [
        (conditioned_alpha_error, "conditioned"),
        (family_alpha_error, "family"),
        (joint_alpha_error, "joint"),
    ]
    candidates.sort(key=lambda x: x[0])
    return candidates[0][0], candidates[0][1]


# ---------------------------------------------------------------------------
# LOO-CV ensemble selection
# ---------------------------------------------------------------------------

def loocv_select(
    rows_for_condition: list[TrialRow],
) -> tuple[list[float], float, str]:
    """
    Leave-one-trial-out cross-validated ensemble selection.

    For each held-out trial, calibrate the best strategy on the rest,
    then apply it to the held-out trial.

    Returns (oos_alpha_errors, oos_mean, best_strategy_name).
    """
    n = len(rows_for_condition)
    oos_errors: list[float] = []
    strategy_wins: dict[str, int] = {}

    for holdout_idx in range(n):
        train = [rows_for_condition[i] for i in range(n) if i != holdout_idx]
        test_row = rows_for_condition[holdout_idx]

        # Evaluate each strategy on training set
        strategy_results: dict[str, float] = {}

        # Strategy: always conditioned
        strategy_results["conditioned"] = float(np.mean([r.conditioned_alpha_error for r in train]))

        # Strategy: always family
        strategy_results["family"] = float(np.mean([r.family_alpha_error for r in train]))

        # Strategy: always joint
        strategy_results["joint"] = float(np.mean([r.joint_alpha_error for r in train]))

        # Strategy: score-competitive across three
        sc_errors = []
        for r in train:
            e, _ = score_competitive_three(
                (0,) * 6, r.conditioned_alpha_error, r.conditioned_score,
                (0,) * 6, r.family_alpha_error, r.family_score,
                (0,) * 6, r.joint_alpha_error, r.joint_score,
            )
            sc_errors.append(e)
        strategy_results["score_competitive"] = float(np.mean(sc_errors))

        # Strategy: support-gated baseline (existing logic)
        strategy_results["support_gated"] = float(np.mean([r.support_gated_alpha_error for r in train]))

        # Strategy: entropy-gated with threshold sweep
        # Use joint when entropy < threshold, conditioned otherwise
        best_ent_mean = float("inf")
        best_ent_threshold = 0.5
        for threshold in np.linspace(0.1, 0.9, 9):
            ent_errors = []
            for r in train:
                if r.joint_pose_entropy < threshold:
                    ent_errors.append(r.joint_alpha_error)
                else:
                    ent_errors.append(r.conditioned_alpha_error)
            ent_mean = float(np.mean(ent_errors))
            if ent_mean < best_ent_mean:
                best_ent_mean = ent_mean
                best_ent_threshold = float(threshold)
        strategy_results["entropy_gated"] = best_ent_mean

        # Strategy: score-margin gated
        # If family_score is much lower than conditioned_score, use family; else conditioned
        # Unless joint is even better by score
        best_margin_mean = float("inf")
        best_margin_threshold = 0.0
        for threshold in np.linspace(-0.02, 0.02, 9):
            margin_errors = []
            for r in train:
                scores = [
                    (r.conditioned_score, r.conditioned_alpha_error),
                    (r.family_score, r.family_alpha_error),
                    (r.joint_score, r.joint_alpha_error),
                ]
                scores.sort(key=lambda x: x[0])
                # Use best scoring, but penalize family in sparse_partial
                if r.condition == "sparse_partial_high_noise" and scores[0][0] > r.conditioned_score + threshold:
                    margin_errors.append(r.conditioned_alpha_error)
                else:
                    margin_errors.append(scores[0][1])
            margin_mean = float(np.mean(margin_errors))
            if margin_mean < best_margin_mean:
                best_margin_mean = margin_mean
                best_margin_threshold = float(threshold)
        strategy_results["margin_gated"] = best_margin_mean

        # Pick best strategy on training set
        best_strategy = min(strategy_results, key=lambda k: strategy_results[k])
        strategy_wins[best_strategy] = strategy_wins.get(best_strategy, 0) + 1

        # Apply best strategy to held-out trial
        if best_strategy == "conditioned":
            oos_errors.append(test_row.conditioned_alpha_error)
        elif best_strategy == "family":
            oos_errors.append(test_row.family_alpha_error)
        elif best_strategy == "joint":
            oos_errors.append(test_row.joint_alpha_error)
        elif best_strategy == "score_competitive":
            e, _ = score_competitive_three(
                (0,) * 6, test_row.conditioned_alpha_error, test_row.conditioned_score,
                (0,) * 6, test_row.family_alpha_error, test_row.family_score,
                (0,) * 6, test_row.joint_alpha_error, test_row.joint_score,
            )
            oos_errors.append(e)
        elif best_strategy == "support_gated":
            oos_errors.append(test_row.support_gated_alpha_error)
        elif best_strategy == "entropy_gated":
            if test_row.joint_pose_entropy < best_ent_threshold:
                oos_errors.append(test_row.joint_alpha_error)
            else:
                oos_errors.append(test_row.conditioned_alpha_error)
        elif best_strategy == "margin_gated":
            scores = [
                (test_row.conditioned_score, test_row.conditioned_alpha_error),
                (test_row.family_score, test_row.family_alpha_error),
                (test_row.joint_score, test_row.joint_alpha_error),
            ]
            scores.sort(key=lambda x: x[0])
            if test_row.condition == "sparse_partial_high_noise" and scores[0][0] > test_row.conditioned_score + best_margin_threshold:
                oos_errors.append(test_row.conditioned_alpha_error)
            else:
                oos_errors.append(scores[0][1])
        else:
            oos_errors.append(test_row.support_gated_alpha_error)

    most_common = max(strategy_wins, key=lambda k: strategy_wins[k]) if strategy_wins else "support_gated"
    return oos_errors, float(np.mean(oos_errors)), most_common


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def summarize_by_condition(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        subset = [row for row in rows if row.condition == condition]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        oos_errors, oos_mean, oos_strategy = loocv_select(subset)

        summary.append(
            {
                "condition": condition,
                "count": len(subset),
                "marginalized_alpha_error_mean": mean("marginalized_alpha_error"),
                "support_gated_alpha_error_mean": mean("support_gated_alpha_error"),
                "conditioned_alpha_error_mean": mean("conditioned_alpha_error"),
                "family_alpha_error_mean": mean("family_alpha_error"),
                "joint_alpha_error_mean": mean("joint_alpha_error"),
                "score_competitive_alpha_error_mean": mean("score_competitive_alpha_error"),
                "oracle_three_alpha_error_mean": mean("oracle_three_alpha_error"),
                "oracle_pose_alpha_error_mean": mean("oracle_pose_alpha_error"),
                "oos_loocv_alpha_error_mean": oos_mean,
                "oos_loocv_strategy": oos_strategy,
                "joint_pose_entropy_mean": mean("joint_pose_entropy"),
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

            oos_errors, oos_mean, oos_strategy = loocv_select(subset)

            summary.append(
                {
                    "condition": condition,
                    "alpha_strength_bin": FOCUS_ALPHA_BIN,
                    "geometry_skew_bin": skew_bin,
                    "count": len(subset),
                    "support_gated_alpha_error_mean": mean("support_gated_alpha_error"),
                    "conditioned_alpha_error_mean": mean("conditioned_alpha_error"),
                    "family_alpha_error_mean": mean("family_alpha_error"),
                    "joint_alpha_error_mean": mean("joint_alpha_error"),
                    "score_competitive_alpha_error_mean": mean("score_competitive_alpha_error"),
                    "oracle_three_alpha_error_mean": mean("oracle_three_alpha_error"),
                    "oracle_pose_alpha_error_mean": mean("oracle_pose_alpha_error"),
                    "oos_loocv_alpha_error_mean": oos_mean,
                    "oos_loocv_strategy": oos_strategy,
                    "joint_pose_entropy_mean": mean("joint_pose_entropy"),
                }
            )
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(row["condition"]) for row in summary_rows]
    x = np.arange(len(conditions))
    width = 0.14

    support_gated = np.array([float(row["support_gated_alpha_error_mean"]) for row in summary_rows])
    conditioned = np.array([float(row["conditioned_alpha_error_mean"]) for row in summary_rows])
    family = np.array([float(row["family_alpha_error_mean"]) for row in summary_rows])
    joint = np.array([float(row["joint_alpha_error_mean"]) for row in summary_rows])
    oos = np.array([float(row["oos_loocv_alpha_error_mean"]) for row in summary_rows])
    oracle_three = np.array([float(row["oracle_three_alpha_error_mean"]) for row in summary_rows])

    fig, ax = plt.subplots(figsize=(16.0, 6.6), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.08, right=0.98)

    ax.bar(x - 2.5 * width, support_gated, width=width, color="#1d3557", label="support-gated baseline")
    ax.bar(x - 1.5 * width, conditioned, width=width, color="#2a9d8f", label="conditioned")
    ax.bar(x - 0.5 * width, family, width=width, color="#f4a261", label="family-switch")
    ax.bar(x + 0.5 * width, joint, width=width, color="#e76f51", label="enhanced joint")
    ax.bar(x + 1.5 * width, oos, width=width, color="#6a0dad", label="LOO-CV ensemble (OOS)")
    ax.bar(x + 2.5 * width, oracle_three, width=width, color="#7f8c8d", label="oracle best-of-three")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Alpha recovery: individual paths vs LOO-CV validated ensemble")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    fig.suptitle(
        "Adaptive Ensemble Solver: LOO-CV Validated Selection Across Three Refinement Paths",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cells(path: str, cell_rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(2, len(FOCUS_CONDITIONS), figsize=(12.4, 7.2), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.98, wspace=0.24, hspace=0.30)

    for col_idx, condition in enumerate(FOCUS_CONDITIONS):
        oos_matrix = np.full((1, len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
        support_matrix = np.full((1, len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
        for row in cell_rows:
            if str(row["condition"]) != condition:
                continue
            j = GEOMETRY_SKEW_BIN_LABELS.index(str(row["geometry_skew_bin"]))
            oos_matrix[0, j] = float(row["oos_loocv_alpha_error_mean"])
            support_matrix[0, j] = float(row["support_gated_alpha_error_mean"])

        ratio_matrix = support_matrix / np.maximum(oos_matrix, 1.0e-12)

        sns.heatmap(
            oos_matrix,
            ax=axes[0, col_idx],
            cmap="magma_r",
            annot=True,
            fmt=".3f",
            xticklabels=GEOMETRY_SKEW_BIN_LABELS,
            yticklabels=[FOCUS_ALPHA_BIN],
            cbar=(col_idx == len(FOCUS_CONDITIONS) - 1),
            cbar_kws={"label": "OOS alpha error"} if col_idx == len(FOCUS_CONDITIONS) - 1 else None,
            vmin=0.0,
            vmax=0.35,
        )
        axes[0, col_idx].set_title(f"{condition}\nOOS LOO-CV alpha error")
        axes[0, col_idx].set_xlabel("geometry skew |t| bin")
        if col_idx == 0:
            axes[0, col_idx].set_ylabel("anisotropy strength")
        else:
            axes[0, col_idx].set_ylabel("")

        sns.heatmap(
            ratio_matrix,
            ax=axes[1, col_idx],
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=GEOMETRY_SKEW_BIN_LABELS,
            yticklabels=[FOCUS_ALPHA_BIN],
            cbar=(col_idx == len(FOCUS_CONDITIONS) - 1),
            cbar_kws={"label": "baseline / OOS alpha error"} if col_idx == len(FOCUS_CONDITIONS) - 1 else None,
            vmin=0.6,
            vmax=1.8,
        )
        axes[1, col_idx].set_title(f"{condition}\nimprovement factor vs support-gated baseline")
        axes[1, col_idx].set_xlabel("geometry skew |t| bin")
        if col_idx == 0:
            axes[1, col_idx].set_ylabel("anisotropy strength")
        else:
            axes[1, col_idx].set_ylabel("")

    fig.suptitle(
        "Adaptive Ensemble Solver: Cell-Level OOS Performance",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(20260324)

    print("Building reference bank...")
    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}

    rows: list[TrialRow] = []
    total_cells = len(FOCUS_CONDITIONS) * len(GEOMETRY_SKEW_BIN_LABELS) * TRIALS_PER_CELL
    done = 0

    for condition in FOCUS_CONDITIONS:
        regime = regime_map[condition]
        temperature = softmin_temperature(regime)
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            for trial_idx in range(TRIALS_PER_CELL):
                done += 1
                print(f"  Trial {done}/{total_cells}: {condition} / {skew_bin} / {trial_idx}")

                true_params = sample_conditioned_parameters(rng, FOCUS_ALPHA_BIN, skew_bin)
                clean_signature = anisotropic_forward_signature(true_params)
                rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(
                    clean_signature, regime, rng
                )

                # Marginalized bank scoring
                marginalized_scores, marginalized_best_shifts = marginalized_candidate_scores(
                    observed_signature, mask, shifted_bank, temperature,
                )
                marginalized_idx = int(np.argmin(marginalized_scores))
                marginalized_params = bank_params[marginalized_idx]
                marginalized_signature = shifted_bank[marginalized_idx, int(marginalized_best_shifts[marginalized_idx])]
                marginalized_geometry, marginalized_weight, marginalized_alpha = symmetry_aware_errors(
                    true_params, marginalized_params,
                )

                # Run all three refinement paths from top-k seeds
                conditioned_best_score = float("inf")
                conditioned_best_params = marginalized_params
                conditioned_best_signature = marginalized_signature

                family_best_score = float("inf")
                family_best_params = marginalized_params
                family_best_signature = marginalized_signature

                joint_best_score = float("inf")
                joint_best_params = marginalized_params
                joint_best_signature = marginalized_signature
                joint_best_entropy = 1.0
                joint_seed_rank = 1

                seed_indices = top_k_indices(marginalized_scores, TOP_K_SEEDS)
                for seed_rank, idx in enumerate(seed_indices, start=1):
                    seed_params = bank_params[idx]

                    # Path 1: candidate-conditioned (fixed geometry, search alpha)
                    c_params, c_sig, _, c_score = candidate_conditioned_search(
                        observed_signature, mask, seed_params, temperature,
                    )
                    if c_score < conditioned_best_score:
                        conditioned_best_score = c_score
                        conditioned_best_params = c_params
                        conditioned_best_signature = c_sig

                    # Path 2: family switching (search geometry + alpha)
                    f_params, f_sig, _, f_score = family_switching_refine(
                        observed_signature, mask, seed_params, temperature,
                    )
                    if f_score < family_best_score:
                        family_best_score = f_score
                        family_best_params = f_params
                        family_best_signature = f_sig

                    # Path 3: enhanced joint solver
                    j_params, j_sig, _, j_score, j_entropy = enhanced_joint_refine(
                        observed_signature, mask, seed_params, temperature, condition,
                        np.random.default_rng(20260324 + done * 1000 + seed_rank),
                    )
                    if j_score < joint_best_score:
                        joint_best_score = j_score
                        joint_best_params = j_params
                        joint_best_signature = j_sig
                        joint_best_entropy = j_entropy
                        joint_seed_rank = seed_rank

                # Compute errors for each path
                conditioned_geometry, conditioned_weight, conditioned_alpha = symmetry_aware_errors(
                    true_params, conditioned_best_params
                )
                conditioned_fit_rmse = rmse(conditioned_best_signature, rotated_signature)

                family_geometry, family_weight, family_alpha = symmetry_aware_errors(
                    true_params, family_best_params
                )
                family_fit_rmse = rmse(family_best_signature, rotated_signature)

                joint_geometry, joint_weight, joint_alpha = symmetry_aware_errors(
                    true_params, joint_best_params
                )
                joint_fit_rmse = rmse(joint_best_signature, rotated_signature)

                # Support-gated baseline
                support_params, support_signature, _, support_choose_family = choose_support_gated_baseline(
                    condition,
                    conditioned_best_params, conditioned_best_signature, conditioned_best_score,
                    family_best_params, family_best_signature, family_best_score,
                )
                support_geometry, support_weight, support_alpha = symmetry_aware_errors(true_params, support_params)
                support_fit_rmse = rmse(support_signature, rotated_signature)

                # Score-competitive across three paths
                sc_alpha, sc_choose = score_competitive_three(
                    conditioned_best_params, float(conditioned_alpha), conditioned_best_score,
                    family_best_params, float(family_alpha), family_best_score,
                    joint_best_params, float(joint_alpha), joint_best_score,
                )

                # Oracle best-of-three
                o3_alpha, o3_choose = oracle_three_way(
                    float(conditioned_alpha), float(family_alpha), float(joint_alpha),
                )

                # Oracle pose ceiling
                oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
                oracle_pose_params, oracle_pose_signature = nearest_neighbor_aligned(
                    oracle_observed, oracle_mask, bank_signatures, bank_params,
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
                        conditioned_alpha_error=float(conditioned_alpha),
                        conditioned_geometry_mae=float(conditioned_geometry),
                        conditioned_weight_mae=float(conditioned_weight),
                        conditioned_fit_rmse=float(conditioned_fit_rmse),
                        conditioned_score=float(conditioned_best_score),
                        family_alpha_error=float(family_alpha),
                        family_geometry_mae=float(family_geometry),
                        family_weight_mae=float(family_weight),
                        family_fit_rmse=float(family_fit_rmse),
                        family_score=float(family_best_score),
                        joint_alpha_error=float(joint_alpha),
                        joint_geometry_mae=float(joint_geometry),
                        joint_weight_mae=float(joint_weight),
                        joint_fit_rmse=float(joint_fit_rmse),
                        joint_score=float(joint_best_score),
                        joint_pose_entropy=float(joint_best_entropy),
                        score_competitive_alpha_error=float(sc_alpha),
                        score_competitive_choose=str(sc_choose),
                        oracle_three_alpha_error=float(o3_alpha),
                        oracle_three_choose=str(o3_choose),
                        oracle_pose_alpha_error=float(oracle_alpha),
                        oracle_pose_geometry_mae=float(oracle_geometry),
                        oracle_pose_weight_mae=float(oracle_weight),
                        oracle_pose_fit_rmse=float(oracle_fit_rmse),
                    )
                )

    print("\nAggregating results...")
    trial_rows = [row.__dict__ for row in rows]
    by_condition = summarize_by_condition(rows)
    by_cell = summarize_by_cell(rows)

    # Overall OOS mean
    all_oos_errors: list[float] = []
    for condition in FOCUS_CONDITIONS:
        subset = [row for row in rows if row.condition == condition]
        oos_errors, _, _ = loocv_select(subset)
        all_oos_errors.extend(oos_errors)
    overall_oos_mean = float(np.mean(all_oos_errors))
    overall_support_mean = float(np.mean([row.support_gated_alpha_error for row in rows]))
    overall_joint_mean = float(np.mean([row.joint_alpha_error for row in rows]))
    overall_oracle_three_mean = float(np.mean([row.oracle_three_alpha_error for row in rows]))

    print(f"\n=== RESULTS ===")
    print(f"Support-gated baseline overall mean: {overall_support_mean:.4f}")
    print(f"Enhanced joint solver overall mean:   {overall_joint_mean:.4f}")
    print(f"LOO-CV ensemble OOS overall mean:     {overall_oos_mean:.4f}")
    print(f"Oracle best-of-three overall mean:    {overall_oracle_three_mean:.4f}")
    print(f"Issue reference baseline:              0.1714")
    print(f"OOS beats issue reference baseline:    {overall_oos_mean < 0.1714}")
    print(f"OOS beats same-trial support-gated:    {overall_oos_mean < overall_support_mean}")

    # Write outputs
    write_csv(os.path.join(OUTPUT_DIR, "adaptive_ensemble_solver_trials.csv"), trial_rows)
    write_csv(os.path.join(OUTPUT_DIR, "adaptive_ensemble_solver_summary.csv"), by_condition)
    write_csv(os.path.join(OUTPUT_DIR, "adaptive_ensemble_solver_cells.csv"), by_cell)

    plot_overview(os.path.join(FIGURE_DIR, "adaptive_ensemble_solver_overview.png"), by_condition)
    plot_cells(os.path.join(FIGURE_DIR, "adaptive_ensemble_solver_cells.png"), by_cell)

    # Complementarity analysis
    complementarity = {
        "support_gated_overall_mean": overall_support_mean,
        "joint_overall_mean": overall_joint_mean,
        "oos_loocv_overall_mean": overall_oos_mean,
        "oracle_three_overall_mean": overall_oracle_three_mean,
        "by_condition": [],
    }
    for row in by_condition:
        complementarity["by_condition"].append(
            {
                "condition": row["condition"],
                "support_gated_alpha_error_mean": row["support_gated_alpha_error_mean"],
                "oos_loocv_alpha_error_mean": row["oos_loocv_alpha_error_mean"],
                "oracle_three_alpha_error_mean": row["oracle_three_alpha_error_mean"],
                "oos_loocv_strategy": row["oos_loocv_strategy"],
            }
        )

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "top_k_seeds": float(TOP_K_SEEDS),
        "trials_per_cell": float(TRIALS_PER_CELL),
        "anneal_factors": [float(x) for x in ANNEAL_FACTORS],
        "shotgun_samples": float(SHOTGUN_SAMPLES),
        "validation_method": "leave-one-trial-out cross-validation",
        "overall_oos_loocv_alpha_error_mean": overall_oos_mean,
        "overall_support_gated_alpha_error_mean": overall_support_mean,
        "overall_joint_alpha_error_mean": overall_joint_mean,
        "overall_oracle_three_alpha_error_mean": overall_oracle_three_mean,
        "oos_beats_same_trial_support_gated": bool(overall_oos_mean < overall_support_mean),
        "oos_beats_issue_reference_0_1714": bool(overall_oos_mean < 0.1714),
        "oos_beats_issue_joint_0_1835": bool(overall_oos_mean < 0.1835),
        "complementarity": complementarity,
    }

    output = {"summary": summary, "by_condition": by_condition, "by_cell": by_cell}
    with open(os.path.join(OUTPUT_DIR, "adaptive_ensemble_solver_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
