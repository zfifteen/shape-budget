"""
Post-roadmap extension: shift-marginalized local refinement experiment.

This experiment builds directly on the soft shift-marginalized pose score.

It asks whether the remaining pose-free alpha bottleneck is partly a local-fit
problem once pose is handled more softly:

1. retrieve top candidates using the shift-marginalized score
2. locally refine geometry and alpha under that same score
3. test whether alpha and geometry become more recoverable
"""

from __future__ import annotations

import sys
from pathlib import Path

_COMPAT_MODULES = Path(__file__).resolve().parents[3] / ".experiment_modules"
if str(_COMPAT_MODULES) not in sys.path:
    sys.path.insert(0, str(_COMPAT_MODULES))

import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from run_pose_free_weighted_inverse_experiment import observe_pose_free_signature
from run_shift_marginalized_pose_experiment import (
    MIN_SOFTMIN_TEMPERATURE,
    marginalized_candidate_scores,
    shift_error_matrix,
    softmin_temperature,
)
from run_weighted_anisotropic_inverse_experiment import (
    ALPHA_MAX,
    ALPHA_MIN,
    GEOMETRY_BOUNDS,
    REFERENCE_BANK_SIZE,
    anisotropic_forward_signature,
    build_reference_bank,
    sample_anisotropic_parameters,
    symmetry_aware_errors,
)
from run_weighted_multisource_inverse_experiment import OBSERVATION_REGIMES, write_csv


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


TOP_K_CANDIDATES = 1
GRID_POINTS = 3
REFINEMENT_ROUNDS = 1
PILOT_TRIALS_PER_REGIME = 2
TOP_K_ENVELOPE = 10
ALPHA_DIVERSE_THRESHOLD = 0.20
MIN_NEAR_TIE_DELTA = 5.0e-5

INITIAL_RHO_RADIUS = 0.018
INITIAL_T_RADIUS = 0.12
INITIAL_H_RADIUS = 0.12
INITIAL_ALPHA_RADIUS = 0.10


@dataclass
class TrialRow:
    condition: str
    trial: int
    true_rho: float
    true_t: float
    true_h: float
    true_w1: float
    true_w2: float
    true_w3: float
    true_alpha: float
    true_rotation_shift: int
    softmin_temperature: float
    baseline_geometry_mae: float
    baseline_weight_mae: float
    baseline_alpha_error: float
    baseline_fit_rmse: float
    baseline_alpha_span_topk: float
    baseline_geometry_dispersion_topk: float
    baseline_weight_dispersion_topk: float
    baseline_near_tie_diverse: int
    refined_geometry_mae: float
    refined_weight_mae: float
    refined_alpha_error: float
    refined_fit_rmse: float
    refined_alpha_span_topk: float
    refined_geometry_dispersion_topk: float
    refined_weight_dispersion_topk: float
    refined_near_tie_diverse: int
    alpha_error_improvement_factor: float
    alpha_span_reduction_factor: float
    geometry_ratio_refined_over_baseline: float
    weight_ratio_refined_over_baseline: float


def near_tie_gap_threshold(regime: dict[str, float | str | int]) -> float:
    sigma = float(regime["noise_sigma"])
    return max(sigma * sigma, MIN_NEAR_TIE_DELTA)


def control_invariants(
    params: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    from run_weighted_anisotropic_inverse_experiment import control_invariants as base_control_invariants

    return base_control_invariants(params)


def canonicalize_candidate(
    params: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    geometry, weights, alpha = control_invariants(params)
    swapped_geometry = np.array([geometry[0], geometry[2], geometry[1]])
    swapped_weights = np.array([weights[1], weights[0], weights[2]])

    direct_tuple = tuple(np.concatenate([geometry, weights]))
    swapped_tuple = tuple(np.concatenate([swapped_geometry, swapped_weights]))
    if swapped_tuple < direct_tuple:
        return swapped_geometry, swapped_weights, alpha
    return geometry, weights, alpha


def ambiguity_metrics(
    scores: np.ndarray,
    params_list: list[tuple[float, float, float, float, float, float]],
    true_params: tuple[float, float, float, float, float, float],
    regime: dict[str, float | str | int],
) -> dict[str, float]:
    order = np.argsort(scores)
    top_k = min(TOP_K_ENVELOPE, len(order))
    top_indices = order[:top_k]
    top_scores = scores[top_indices]

    geometries = []
    weights = []
    alphas = []
    for idx in top_indices:
        geometry, weight, alpha = canonicalize_candidate(params_list[int(idx)])
        geometries.append(geometry)
        weights.append(weight)
        alphas.append(alpha)

    geometry_matrix = np.array(geometries)
    weight_matrix = np.array(weights)
    alpha_vec = np.array(alphas)

    best_idx = int(order[0])
    best_params = params_list[best_idx]
    best_geometry_mae, best_weight_mae, best_alpha_error = symmetry_aware_errors(true_params, best_params)

    gap_topk = float(top_scores[-1] - top_scores[0])
    near_tie_diverse = int(
        gap_topk <= near_tie_gap_threshold(regime)
        and float(np.max(alpha_vec) - np.min(alpha_vec)) >= ALPHA_DIVERSE_THRESHOLD
    )

    return {
        "best_idx": float(best_idx),
        "alpha_error": float(best_alpha_error),
        "geometry_mae": float(best_geometry_mae),
        "weight_mae": float(best_weight_mae),
        "alpha_span_topk": float(np.max(alpha_vec) - np.min(alpha_vec)),
        "geometry_dispersion_topk": float(np.mean(np.std(geometry_matrix, axis=0))),
        "weight_dispersion_topk": float(np.mean(np.std(weight_matrix, axis=0))),
        "near_tie_diverse": near_tie_diverse,
    }


def evaluate_params(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    params: tuple[float, float, float, float, float, float],
    temperature: float,
) -> tuple[float, np.ndarray, int]:
    signature = anisotropic_forward_signature(params)
    shift_stack = np.stack([np.roll(signature, shift) for shift in range(len(signature))], axis=0)
    mse = shift_error_matrix(observed_signature, mask, shift_stack[None, :, :])[0]
    best_shift = int(np.argmin(mse))
    minima = float(np.min(mse))
    stable = np.exp(-(mse - minima) / temperature)
    marginalized = minima - temperature * math.log(float(np.mean(stable)))
    return float(marginalized), shift_stack[best_shift], best_shift


def unique_centered_grid(center: float, radius: float, lower: float, upper: float, count: int) -> np.ndarray:
    grid = np.linspace(max(lower, center - radius), min(upper, center + radius), count)
    return np.unique(np.concatenate([grid, np.array([center], dtype=float)]))


def top_k_unique_candidates(
    scores: np.ndarray,
    best_shifts: np.ndarray,
    k: int,
) -> list[tuple[int, int, float]]:
    order = np.argsort(scores)
    selected = []
    for idx in order[:k]:
        selected.append((int(idx), int(best_shifts[int(idx)]), float(scores[int(idx)])))
    return selected


def refine_candidate(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    seed_params: tuple[float, float, float, float, float, float],
    temperature: float,
) -> tuple[
    tuple[float, float, float, float, float, float],
    np.ndarray,
    int,
    float,
    list[tuple[tuple[float, float, float, float, float, float], float]],
]:
    rho, t, h, w1, w2, alpha = seed_params
    current = {
        "rho": float(rho),
        "t": float(t),
        "h": float(h),
        "alpha": float(alpha),
    }

    best_params = seed_params
    best_score, best_signature, best_shift = evaluate_params(observed_signature, mask, best_params, temperature)
    explored_states: list[tuple[tuple[float, float, float, float, float, float], float]] = [(best_params, best_score)]

    rho_radius = INITIAL_RHO_RADIUS
    t_radius = INITIAL_T_RADIUS
    h_radius = INITIAL_H_RADIUS
    alpha_radius = INITIAL_ALPHA_RADIUS

    for _ in range(REFINEMENT_ROUNDS):
        for name, radius, lower, upper in [
            ("rho", rho_radius, GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]),
            ("t", t_radius, GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]),
            ("h", h_radius, GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]),
            ("alpha", alpha_radius, ALPHA_MIN, ALPHA_MAX),
        ]:
            grid = unique_centered_grid(current[name], radius, lower, upper, GRID_POINTS)
            local_best_score = best_score
            local_best_params = best_params
            local_best_signature = best_signature
            local_best_shift = best_shift

            for value in grid:
                candidate = (
                    float(value) if name == "rho" else current["rho"],
                    float(value) if name == "t" else current["t"],
                    float(value) if name == "h" else current["h"],
                    w1,
                    w2,
                    float(value) if name == "alpha" else current["alpha"],
                )
                score, signature, shift = evaluate_params(observed_signature, mask, candidate, temperature)
                explored_states.append((candidate, score))
                if score < local_best_score:
                    local_best_score = score
                    local_best_params = candidate
                    local_best_signature = signature
                    local_best_shift = shift

            best_score = local_best_score
            best_params = local_best_params
            best_signature = local_best_signature
            best_shift = local_best_shift
            current["rho"], current["t"], current["h"], _, _, current["alpha"] = best_params

        rho_radius *= 0.5
        t_radius *= 0.5
        h_radius *= 0.5
        alpha_radius *= 0.5

    return best_params, best_signature, best_shift, float(best_score), explored_states


def summarize_trials(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        baseline_alpha_error_mean = mean("baseline_alpha_error")
        refined_alpha_error_mean = mean("refined_alpha_error")
        baseline_alpha_span_mean = mean("baseline_alpha_span_topk")
        refined_alpha_span_mean = mean("refined_alpha_span_topk")
        baseline_geometry_mae_mean = mean("baseline_geometry_mae")
        refined_geometry_mae_mean = mean("refined_geometry_mae")
        baseline_weight_mae_mean = mean("baseline_weight_mae")
        refined_weight_mae_mean = mean("refined_weight_mae")

        summary.append(
            {
                "condition": name,
                "baseline_geometry_mae_mean": baseline_geometry_mae_mean,
                "refined_geometry_mae_mean": refined_geometry_mae_mean,
                "baseline_weight_mae_mean": baseline_weight_mae_mean,
                "refined_weight_mae_mean": refined_weight_mae_mean,
                "baseline_alpha_error_mean": baseline_alpha_error_mean,
                "refined_alpha_error_mean": refined_alpha_error_mean,
                "baseline_fit_rmse_mean": mean("baseline_fit_rmse"),
                "refined_fit_rmse_mean": mean("refined_fit_rmse"),
                "baseline_alpha_span_topk_mean": baseline_alpha_span_mean,
                "refined_alpha_span_topk_mean": refined_alpha_span_mean,
                "baseline_geometry_dispersion_topk_mean": mean("baseline_geometry_dispersion_topk"),
                "refined_geometry_dispersion_topk_mean": mean("refined_geometry_dispersion_topk"),
                "baseline_weight_dispersion_topk_mean": mean("baseline_weight_dispersion_topk"),
                "refined_weight_dispersion_topk_mean": mean("refined_weight_dispersion_topk"),
                "baseline_near_tie_diverse_fraction": mean("baseline_near_tie_diverse"),
                "refined_near_tie_diverse_fraction": mean("refined_near_tie_diverse"),
                "alpha_error_improvement_factor_mean": mean("alpha_error_improvement_factor"),
                "alpha_span_reduction_factor_mean": mean("alpha_span_reduction_factor"),
                "alpha_error_ratio_of_means_baseline_over_refined": float(
                    baseline_alpha_error_mean / max(refined_alpha_error_mean, 1.0e-12)
                ),
                "alpha_span_ratio_of_means_baseline_over_refined": float(
                    baseline_alpha_span_mean / max(refined_alpha_span_mean, 1.0e-12)
                ),
                "geometry_ratio_of_means_refined_over_baseline": float(
                    refined_geometry_mae_mean / max(baseline_geometry_mae_mean, 1.0e-12)
                ),
                "weight_ratio_of_means_refined_over_baseline": float(
                    refined_weight_mae_mean / max(baseline_weight_mae_mean, 1.0e-12)
                ),
                "softmin_temperature_mean": mean("softmin_temperature"),
            }
        )
    return summary


def plot_overview(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    x = np.arange(len(conditions))
    width = 0.36

    baseline_alpha = np.array([float(item["baseline_alpha_error_mean"]) for item in summary_rows])
    refined_alpha = np.array([float(item["refined_alpha_error_mean"]) for item in summary_rows])
    baseline_span = np.array([float(item["baseline_alpha_span_topk_mean"]) for item in summary_rows])
    refined_span = np.array([float(item["refined_alpha_span_topk_mean"]) for item in summary_rows])
    baseline_geom = np.array([float(item["baseline_geometry_mae_mean"]) for item in summary_rows])
    refined_geom = np.array([float(item["refined_geometry_mae_mean"]) for item in summary_rows])
    baseline_diverse = np.array([float(item["baseline_near_tie_diverse_fraction"]) for item in summary_rows])
    refined_diverse = np.array([float(item["refined_near_tie_diverse_fraction"]) for item in summary_rows])

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.98, wspace=0.24, hspace=0.34)

    for ax, baseline_vals, refined_vals, ylabel, title in [
        (axes[0, 0], baseline_alpha, refined_alpha, "mean alpha absolute error", "Alpha recovery under marginalized pose handling"),
        (axes[0, 1], baseline_span, refined_span, "mean top-10 alpha span", "Alpha ambiguity envelope after local refinement"),
        (axes[1, 0], baseline_geom, refined_geom, "mean geometry MAE", "Geometry recovery under marginalized pose handling"),
        (axes[1, 1], baseline_diverse, refined_diverse, "fraction of trials", "Near-tie and alpha-diverse trials"),
    ]:
        ax.bar(x - width / 2.0, baseline_vals, width=width, color="#e76f51", label="marginalized baseline")
        ax.bar(x + width / 2.0, refined_vals, width=width, color="#2a9d8f", label="local refined")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0, 0].legend(loc="upper left", frameon=True)

    fig.suptitle("Shift-Marginalized Local Refinement A: Baseline Versus Local Refinement", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_trial_scatter(path: str, rows: list[TrialRow]) -> None:
    chosen_conditions = ["full_noisy", "sparse_full_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(1, len(chosen_conditions), figsize=(15.4, 5.0), constrained_layout=False)
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.28)

    for ax, condition in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == condition]
        baseline = np.array([row.baseline_alpha_error for row in subset])
        refined = np.array([row.refined_alpha_error for row in subset])
        temperature = np.array([row.softmin_temperature for row in subset])

        scatter = ax.scatter(
            baseline,
            refined,
            c=np.log10(np.maximum(temperature, 1.0e-12)),
            cmap="viridis",
            s=42,
            alpha=0.82,
            edgecolors="none",
        )
        max_val = max(float(np.max(baseline)), float(np.max(refined)), 1.0e-6)
        ax.plot([0.0, max_val], [0.0, max_val], color="#444444", linestyle="--", lw=1.4)
        ax.set_xlabel("marginalized baseline alpha error")
        ax.set_ylabel("refined alpha error")
        ax.set_title(condition)

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.86)
    cbar.set_label("log10 softmin temperature")
    fig.suptitle("Shift-Marginalized Local Refinement B: Trial-Level Alpha Error", fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = np.stack([np.roll(bank_signatures, shift, axis=1) for shift in range(bank_signatures.shape[1])], axis=1)

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        temperature = softmin_temperature(regime)
        for trial in range(PILOT_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            baseline_scores, baseline_best_shift = marginalized_candidate_scores(
                observed_signature,
                mask,
                shifted_bank,
                temperature,
            )
            baseline_metrics = ambiguity_metrics(baseline_scores, params_list, true_params, regime)
            baseline_best_idx = int(baseline_metrics["best_idx"])
            baseline_best_signature = shifted_bank[baseline_best_idx, int(baseline_best_shift[baseline_best_idx])]
            baseline_fit_rmse = float(np.sqrt(np.mean((baseline_best_signature - rotated_signature) ** 2)))

            top_candidates = top_k_unique_candidates(baseline_scores, baseline_best_shift, TOP_K_CANDIDATES)
            refined_best_score = float("inf")
            refined_best_params: tuple[float, float, float, float, float, float] | None = None
            refined_best_signature: np.ndarray | None = None
            refined_best_shift = 0
            refined_family_map: dict[tuple[float, float, float, float, float, float], float] = {}

            for bank_idx, _, _ in top_candidates:
                seed_params = params_list[bank_idx]
                refined_params, refined_signature, refined_shift, refined_score, explored_states = refine_candidate(
                    observed_signature,
                    mask,
                    seed_params,
                    temperature,
                )
                for candidate_params, candidate_score in explored_states:
                    key = tuple(float(x) for x in candidate_params)
                    if key not in refined_family_map or candidate_score < refined_family_map[key]:
                        refined_family_map[key] = candidate_score
                if refined_score < refined_best_score:
                    refined_best_score = refined_score
                    refined_best_params = refined_params
                    refined_best_signature = refined_signature
                    refined_best_shift = refined_shift

            assert refined_best_params is not None
            assert refined_best_signature is not None

            baseline_geom, baseline_weight, baseline_alpha = symmetry_aware_errors(true_params, params_list[baseline_best_idx])
            refined_geom, refined_weight, refined_alpha = symmetry_aware_errors(true_params, refined_best_params)
            refined_fit_rmse = float(np.sqrt(np.mean((refined_best_signature - rotated_signature) ** 2)))

            # Build a small refined family around the top candidates for the ambiguity envelope.
            refined_family_params = list(refined_family_map.keys())
            refined_family_scores = [refined_family_map[key] for key in refined_family_params]
            refined_family_metrics = ambiguity_metrics(
                np.array(refined_family_scores, dtype=float),
                refined_family_params,
                true_params,
                regime,
            )

            rows.append(
                TrialRow(
                    condition=str(regime["name"]),
                    trial=trial,
                    true_rho=float(true_params[0]),
                    true_t=float(true_params[1]),
                    true_h=float(true_params[2]),
                    true_w1=float(true_params[3]),
                    true_w2=float(true_params[4]),
                    true_w3=float(1.0 - true_params[3] - true_params[4]),
                    true_alpha=float(true_params[5]),
                    true_rotation_shift=int(true_shift),
                    softmin_temperature=float(temperature),
                    baseline_geometry_mae=float(baseline_geom),
                    baseline_weight_mae=float(baseline_weight),
                    baseline_alpha_error=float(baseline_alpha),
                    baseline_fit_rmse=float(baseline_fit_rmse),
                    baseline_alpha_span_topk=float(baseline_metrics["alpha_span_topk"]),
                    baseline_geometry_dispersion_topk=float(baseline_metrics["geometry_dispersion_topk"]),
                    baseline_weight_dispersion_topk=float(baseline_metrics["weight_dispersion_topk"]),
                    baseline_near_tie_diverse=int(baseline_metrics["near_tie_diverse"]),
                    refined_geometry_mae=float(refined_geom),
                    refined_weight_mae=float(refined_weight),
                    refined_alpha_error=float(refined_alpha),
                    refined_fit_rmse=float(refined_fit_rmse),
                    refined_alpha_span_topk=float(refined_family_metrics["alpha_span_topk"]),
                    refined_geometry_dispersion_topk=float(refined_family_metrics["geometry_dispersion_topk"]),
                    refined_weight_dispersion_topk=float(refined_family_metrics["weight_dispersion_topk"]),
                    refined_near_tie_diverse=int(refined_family_metrics["near_tie_diverse"]),
                    alpha_error_improvement_factor=float(baseline_alpha / max(refined_alpha, 1.0e-12)),
                    alpha_span_reduction_factor=float(
                        baseline_metrics["alpha_span_topk"] / max(refined_family_metrics["alpha_span_topk"], 1.0e-12)
                    ),
                    geometry_ratio_refined_over_baseline=float(refined_geom / max(baseline_geom, 1.0e-12)),
                    weight_ratio_refined_over_baseline=float(refined_weight / max(baseline_weight, 1.0e-12)),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = summarize_trials(rows)
    write_csv(os.path.join(OUTPUT_DIR, "shift_marginalized_local_refinement_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "shift_marginalized_local_refinement_summary.csv"), summary_rows)

    plot_overview(os.path.join(FIGURE_DIR, "shift_marginalized_local_refinement_overview.png"), summary_rows)
    plot_trial_scatter(os.path.join(FIGURE_DIR, "shift_marginalized_local_refinement_trial_scatter.png"), rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "pilot_trials_per_regime": PILOT_TRIALS_PER_REGIME,
        "top_k_candidates": TOP_K_CANDIDATES,
        "grid_points": GRID_POINTS,
        "refinement_rounds": REFINEMENT_ROUNDS,
        "largest_alpha_error_ratio_of_means_baseline_over_refined": float(
            max(item["alpha_error_ratio_of_means_baseline_over_refined"] for item in summary_rows)
        ),
        "smallest_alpha_error_ratio_of_means_baseline_over_refined": float(
            min(item["alpha_error_ratio_of_means_baseline_over_refined"] for item in summary_rows)
        ),
        "largest_alpha_span_ratio_of_means_baseline_over_refined": float(
            max(item["alpha_span_ratio_of_means_baseline_over_refined"] for item in summary_rows)
        ),
        "smallest_alpha_span_ratio_of_means_baseline_over_refined": float(
            min(item["alpha_span_ratio_of_means_baseline_over_refined"] for item in summary_rows)
        ),
        "largest_refined_near_tie_diverse_fraction": float(
            max(item["refined_near_tie_diverse_fraction"] for item in summary_rows)
        ),
        "smallest_refined_near_tie_diverse_fraction": float(
            min(item["refined_near_tie_diverse_fraction"] for item in summary_rows)
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "shift_marginalized_local_refinement_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
