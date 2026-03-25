"""
Post-roadmap extension: shift-marginalized pose experiment.

This experiment tests whether the pose-free alpha recovery challenge is partly caused
by the hard winner-take-all pose rule used in the current inverse.

The comparison is between:

1. the current hard min-over-shifts score
2. a soft shift-marginalized score that keeps full-signature information but
   integrates evidence over all shifts instead of trusting only the single best
   one
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

REFERENCE_BANK_SIZE, TEST_TRIALS_PER_REGIME, anisotropic_forward_signature, build_reference_bank, control_invariants, sample_anisotropic_parameters, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "REFERENCE_BANK_SIZE",
    "TEST_TRIALS_PER_REGIME",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "control_invariants",
    "sample_anisotropic_parameters",
    "symmetry_aware_errors",
)

OBSERVATION_REGIMES, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
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

TOP_K_ENVELOPE = 10
ALPHA_DIVERSE_THRESHOLD = 0.20
MIN_NEAR_TIE_DELTA = 5.0e-5
MIN_SOFTMIN_TEMPERATURE = 1.0e-4

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
    marginalized_geometry_mae: float
    marginalized_weight_mae: float
    marginalized_alpha_error: float
    marginalized_fit_rmse: float
    marginalized_alpha_span_topk: float
    marginalized_geometry_dispersion_topk: float
    marginalized_weight_dispersion_topk: float
    marginalized_near_tie_diverse: int
    alpha_error_improvement_factor: float
    alpha_span_reduction_factor: float
    geometry_ratio_marginalized_over_baseline: float
    weight_ratio_marginalized_over_baseline: float

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

def near_tie_gap_threshold(regime: dict[str, float | str | int]) -> float:
    sigma = float(regime["noise_sigma"])
    return max(sigma * sigma, MIN_NEAR_TIE_DELTA)

def softmin_temperature(regime: dict[str, float | str | int]) -> float:
    sigma = float(regime["noise_sigma"])
    return max(sigma * sigma, MIN_SOFTMIN_TEMPERATURE)

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

def shift_error_matrix(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
) -> np.ndarray:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    return np.mean(residual * residual, axis=2)

def baseline_candidate_scores(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mse = shift_error_matrix(observed_signature, mask, shifted_bank)
    best_shift = np.argmin(mse, axis=1)
    best_score = np.min(mse, axis=1)
    return best_score, best_shift

def marginalized_candidate_scores(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    mse = shift_error_matrix(observed_signature, mask, shifted_bank)
    best_shift = np.argmin(mse, axis=1)
    minima = np.min(mse, axis=1, keepdims=True)
    stable = np.exp(-(mse - minima) / temperature)
    marginalized = minima[:, 0] - temperature * np.log(np.mean(stable, axis=1))
    return marginalized, best_shift

def summarize_trials(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        baseline_alpha_error_mean = mean("baseline_alpha_error")
        marginalized_alpha_error_mean = mean("marginalized_alpha_error")
        baseline_alpha_span_mean = mean("baseline_alpha_span_topk")
        marginalized_alpha_span_mean = mean("marginalized_alpha_span_topk")
        baseline_geometry_mae_mean = mean("baseline_geometry_mae")
        marginalized_geometry_mae_mean = mean("marginalized_geometry_mae")
        baseline_weight_mae_mean = mean("baseline_weight_mae")
        marginalized_weight_mae_mean = mean("marginalized_weight_mae")

        summary.append(
            {
                "condition": name,
                "baseline_geometry_mae_mean": baseline_geometry_mae_mean,
                "marginalized_geometry_mae_mean": marginalized_geometry_mae_mean,
                "baseline_weight_mae_mean": baseline_weight_mae_mean,
                "marginalized_weight_mae_mean": marginalized_weight_mae_mean,
                "baseline_alpha_error_mean": baseline_alpha_error_mean,
                "marginalized_alpha_error_mean": marginalized_alpha_error_mean,
                "baseline_fit_rmse_mean": mean("baseline_fit_rmse"),
                "marginalized_fit_rmse_mean": mean("marginalized_fit_rmse"),
                "baseline_alpha_span_topk_mean": baseline_alpha_span_mean,
                "marginalized_alpha_span_topk_mean": marginalized_alpha_span_mean,
                "baseline_geometry_dispersion_topk_mean": mean("baseline_geometry_dispersion_topk"),
                "marginalized_geometry_dispersion_topk_mean": mean("marginalized_geometry_dispersion_topk"),
                "baseline_weight_dispersion_topk_mean": mean("baseline_weight_dispersion_topk"),
                "marginalized_weight_dispersion_topk_mean": mean("marginalized_weight_dispersion_topk"),
                "baseline_near_tie_diverse_fraction": mean("baseline_near_tie_diverse"),
                "marginalized_near_tie_diverse_fraction": mean("marginalized_near_tie_diverse"),
                "alpha_error_improvement_factor_mean": mean("alpha_error_improvement_factor"),
                "alpha_span_reduction_factor_mean": mean("alpha_span_reduction_factor"),
                "alpha_error_ratio_of_means_baseline_over_marginalized": float(
                    baseline_alpha_error_mean / max(marginalized_alpha_error_mean, 1.0e-12)
                ),
                "alpha_span_ratio_of_means_baseline_over_marginalized": float(
                    baseline_alpha_span_mean / max(marginalized_alpha_span_mean, 1.0e-12)
                ),
                "geometry_ratio_of_means_marginalized_over_baseline": float(
                    marginalized_geometry_mae_mean / max(baseline_geometry_mae_mean, 1.0e-12)
                ),
                "weight_ratio_of_means_marginalized_over_baseline": float(
                    marginalized_weight_mae_mean / max(baseline_weight_mae_mean, 1.0e-12)
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
    marginalized_alpha = np.array([float(item["marginalized_alpha_error_mean"]) for item in summary_rows])
    baseline_span = np.array([float(item["baseline_alpha_span_topk_mean"]) for item in summary_rows])
    marginalized_span = np.array([float(item["marginalized_alpha_span_topk_mean"]) for item in summary_rows])
    baseline_geom = np.array([float(item["baseline_geometry_mae_mean"]) for item in summary_rows])
    marginalized_geom = np.array([float(item["marginalized_geometry_mae_mean"]) for item in summary_rows])
    baseline_diverse = np.array([float(item["baseline_near_tie_diverse_fraction"]) for item in summary_rows])
    marginalized_diverse = np.array([float(item["marginalized_near_tie_diverse_fraction"]) for item in summary_rows])

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.98, wspace=0.24, hspace=0.34)

    for ax, baseline_vals, marginalized_vals, ylabel, title in [
        (axes[0, 0], baseline_alpha, marginalized_alpha, "mean alpha absolute error", "Alpha recovery under pose-free inversion"),
        (axes[0, 1], baseline_span, marginalized_span, "mean top-10 alpha span", "Alpha ambiguity envelope under pose-free inversion"),
        (axes[1, 0], baseline_geom, marginalized_geom, "mean geometry MAE", "Geometry recovery under pose-free inversion"),
        (axes[1, 1], baseline_diverse, marginalized_diverse, "fraction of trials", "Near-tie and alpha-diverse trials"),
    ]:
        ax.bar(x - width / 2.0, baseline_vals, width=width, color="#e76f51", label="baseline best-shift")
        ax.bar(x + width / 2.0, marginalized_vals, width=width, color="#2a9d8f", label="shift-marginalized")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0, 0].legend(loc="upper left", frameon=True)

    fig.suptitle("Shift-Marginalized Pose Experiment A: Best-Shift Versus Soft Shift Marginalization", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_trial_scatter(path: str, rows: list[TrialRow]) -> None:
    chosen_conditions = ["full_noisy", "partial_arc_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(1, len(chosen_conditions), figsize=(15.4, 5.0), constrained_layout=False)
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.28)

    for ax, condition in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == condition]
        baseline = np.array([row.baseline_alpha_error for row in subset])
        marginalized = np.array([row.marginalized_alpha_error for row in subset])
        temperature = np.array([row.softmin_temperature for row in subset])

        scatter = ax.scatter(
            baseline,
            marginalized,
            c=np.log10(np.maximum(temperature, 1.0e-12)),
            cmap="viridis",
            s=42,
            alpha=0.82,
            edgecolors="none",
        )
        max_val = max(float(np.max(baseline)), float(np.max(marginalized)), 1.0e-6)
        ax.plot([0.0, max_val], [0.0, max_val], color="#444444", linestyle="--", lw=1.4)
        ax.set_xlabel("baseline alpha error")
        ax.set_ylabel("marginalized alpha error")
        ax.set_title(condition)

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.86)
    cbar.set_label("log10 softmin temperature")
    fig.suptitle("Shift-Marginalized Pose Experiment B: Trial-Level Alpha Error", fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    rng = np.random.default_rng(20260324)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        temperature = softmin_temperature(regime)
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            baseline_scores, baseline_best_shift = baseline_candidate_scores(observed_signature, mask, shifted_bank)
            baseline_metrics = ambiguity_metrics(baseline_scores, params_list, true_params, regime)
            baseline_best_idx = int(baseline_metrics["best_idx"])
            baseline_best_signature = shifted_bank[baseline_best_idx, int(baseline_best_shift[baseline_best_idx])]
            baseline_fit_rmse = float(np.sqrt(np.mean((baseline_best_signature - rotated_signature) ** 2)))

            marginalized_scores, marginalized_best_shift = marginalized_candidate_scores(
                observed_signature,
                mask,
                shifted_bank,
                temperature,
            )
            marginalized_metrics = ambiguity_metrics(marginalized_scores, params_list, true_params, regime)
            marginalized_best_idx = int(marginalized_metrics["best_idx"])
            marginalized_best_signature = shifted_bank[marginalized_best_idx, int(marginalized_best_shift[marginalized_best_idx])]
            marginalized_fit_rmse = float(np.sqrt(np.mean((marginalized_best_signature - rotated_signature) ** 2)))

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
                    baseline_geometry_mae=float(baseline_metrics["geometry_mae"]),
                    baseline_weight_mae=float(baseline_metrics["weight_mae"]),
                    baseline_alpha_error=float(baseline_metrics["alpha_error"]),
                    baseline_fit_rmse=float(baseline_fit_rmse),
                    baseline_alpha_span_topk=float(baseline_metrics["alpha_span_topk"]),
                    baseline_geometry_dispersion_topk=float(baseline_metrics["geometry_dispersion_topk"]),
                    baseline_weight_dispersion_topk=float(baseline_metrics["weight_dispersion_topk"]),
                    baseline_near_tie_diverse=int(baseline_metrics["near_tie_diverse"]),
                    marginalized_geometry_mae=float(marginalized_metrics["geometry_mae"]),
                    marginalized_weight_mae=float(marginalized_metrics["weight_mae"]),
                    marginalized_alpha_error=float(marginalized_metrics["alpha_error"]),
                    marginalized_fit_rmse=float(marginalized_fit_rmse),
                    marginalized_alpha_span_topk=float(marginalized_metrics["alpha_span_topk"]),
                    marginalized_geometry_dispersion_topk=float(marginalized_metrics["geometry_dispersion_topk"]),
                    marginalized_weight_dispersion_topk=float(marginalized_metrics["weight_dispersion_topk"]),
                    marginalized_near_tie_diverse=int(marginalized_metrics["near_tie_diverse"]),
                    alpha_error_improvement_factor=float(
                        baseline_metrics["alpha_error"] / max(marginalized_metrics["alpha_error"], 1.0e-12)
                    ),
                    alpha_span_reduction_factor=float(
                        baseline_metrics["alpha_span_topk"] / max(marginalized_metrics["alpha_span_topk"], 1.0e-12)
                    ),
                    geometry_ratio_marginalized_over_baseline=float(
                        marginalized_metrics["geometry_mae"] / max(baseline_metrics["geometry_mae"], 1.0e-12)
                    ),
                    weight_ratio_marginalized_over_baseline=float(
                        marginalized_metrics["weight_mae"] / max(baseline_metrics["weight_mae"], 1.0e-12)
                    ),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = summarize_trials(rows)
    write_csv(os.path.join(OUTPUT_DIR, "shift_marginalized_pose_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "shift_marginalized_pose_summary.csv"), summary_rows)

    plot_overview(os.path.join(FIGURE_DIR, "shift_marginalized_pose_overview.png"), summary_rows)
    plot_trial_scatter(os.path.join(FIGURE_DIR, "shift_marginalized_pose_trial_scatter.png"), rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "top_k_envelope": TOP_K_ENVELOPE,
        "min_softmin_temperature": MIN_SOFTMIN_TEMPERATURE,
        "largest_alpha_error_ratio_of_means_baseline_over_marginalized": float(
            max(item["alpha_error_ratio_of_means_baseline_over_marginalized"] for item in summary_rows)
        ),
        "smallest_alpha_error_ratio_of_means_baseline_over_marginalized": float(
            min(item["alpha_error_ratio_of_means_baseline_over_marginalized"] for item in summary_rows)
        ),
        "largest_alpha_span_ratio_of_means_baseline_over_marginalized": float(
            max(item["alpha_span_ratio_of_means_baseline_over_marginalized"] for item in summary_rows)
        ),
        "smallest_alpha_span_ratio_of_means_baseline_over_marginalized": float(
            min(item["alpha_span_ratio_of_means_baseline_over_marginalized"] for item in summary_rows)
        ),
        "largest_marginalized_near_tie_diverse_fraction": float(
            max(item["marginalized_near_tie_diverse_fraction"] for item in summary_rows)
        ),
        "smallest_marginalized_near_tie_diverse_fraction": float(
            min(item["marginalized_near_tie_diverse_fraction"] for item in summary_rows)
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "shift_marginalized_pose_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))

if __name__ == "__main__":
    main()
