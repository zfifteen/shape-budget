"""
Post-roadmap extension: rotation-invariant spectral representation experiment.

This experiment tests whether the pose-free alpha recovery challenge is partly caused
by the current shift-search radial-signature representation.

The comparison is between:

1. the current pose-free cyclic-shift-aware matcher on full signatures
2. a compact rotation-invariant spectral matcher using low-order harmonic
   magnitudes estimated from the observed boundary alone

The spectral representation removes pose by construction rather than by
searching over shifts.
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
MAX_HARMONIC = 4

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
    spectral_design_condition: float
    baseline_geometry_mae: float
    baseline_weight_mae: float
    baseline_alpha_error: float
    baseline_fit_rmse: float
    baseline_alpha_span_topk: float
    baseline_geometry_dispersion_topk: float
    baseline_weight_dispersion_topk: float
    baseline_near_tie_diverse: int
    spectral_geometry_mae: float
    spectral_weight_mae: float
    spectral_alpha_error: float
    spectral_fit_rmse: float
    spectral_alpha_span_topk: float
    spectral_geometry_dispersion_topk: float
    spectral_weight_dispersion_topk: float
    spectral_near_tie_diverse: int
    alpha_error_improvement_factor: float
    alpha_span_reduction_factor: float
    geometry_error_ratio_spectral_over_baseline: float
    weight_error_ratio_spectral_over_baseline: float

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

def harmonic_design(indices: np.ndarray, angle_count: int, max_harmonic: int) -> np.ndarray:
    theta = 2.0 * math.pi * indices / angle_count
    columns = [np.ones(len(indices), dtype=float)]
    for harmonic in range(1, max_harmonic + 1):
        columns.append(np.cos(harmonic * theta))
        columns.append(np.sin(harmonic * theta))
    return np.column_stack(columns)

def spectral_feature_from_observation(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    max_harmonic: int,
) -> tuple[np.ndarray, float]:
    observed = np.flatnonzero(mask)
    angle_count = len(observed_signature)
    if len(observed) == 0:
        return np.zeros(max_harmonic, dtype=float), float("inf")

    design = harmonic_design(observed, angle_count, max_harmonic)
    coeffs, *_ = np.linalg.lstsq(design, observed_signature[observed], rcond=None)
    amps = []
    for harmonic in range(max_harmonic):
        cos_coeff = coeffs[1 + 2 * harmonic]
        sin_coeff = coeffs[2 + 2 * harmonic]
        amps.append(float(math.hypot(float(cos_coeff), float(sin_coeff))))
    return np.array(amps, dtype=float), float(np.linalg.cond(design))

def build_spectral_bank(
    bank_signatures: np.ndarray,
    max_harmonic: int,
) -> np.ndarray:
    full_mask = np.ones(bank_signatures.shape[1], dtype=bool)
    return np.array(
        [
            spectral_feature_from_observation(signature, full_mask, max_harmonic)[0]
            for signature in bank_signatures
        ]
    )

def pose_free_candidate_scores(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    mse = np.mean(residual * residual, axis=2)
    best_shift = np.argmin(mse, axis=1)
    best_score = np.min(mse, axis=1)
    return best_score, best_shift

def spectral_candidate_scores(
    observed_feature: np.ndarray,
    bank_features: np.ndarray,
) -> np.ndarray:
    residual = bank_features - observed_feature[None, :]
    return np.mean(residual * residual, axis=1)

def best_shift_for_candidate(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    candidate_shift_stack: np.ndarray,
) -> tuple[np.ndarray, int]:
    residual = candidate_shift_stack[:, mask] - observed_signature[mask][None, :]
    mse = np.mean(residual * residual, axis=1)
    best_shift = int(np.argmin(mse))
    return candidate_shift_stack[best_shift], best_shift

def summarize_trials(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        baseline_alpha_error_mean = mean("baseline_alpha_error")
        spectral_alpha_error_mean = mean("spectral_alpha_error")
        baseline_alpha_span_mean = mean("baseline_alpha_span_topk")
        spectral_alpha_span_mean = mean("spectral_alpha_span_topk")
        baseline_geometry_mae_mean = mean("baseline_geometry_mae")
        spectral_geometry_mae_mean = mean("spectral_geometry_mae")
        baseline_weight_mae_mean = mean("baseline_weight_mae")
        spectral_weight_mae_mean = mean("spectral_weight_mae")

        summary.append(
            {
                "condition": name,
                "baseline_geometry_mae_mean": baseline_geometry_mae_mean,
                "spectral_geometry_mae_mean": spectral_geometry_mae_mean,
                "baseline_weight_mae_mean": baseline_weight_mae_mean,
                "spectral_weight_mae_mean": spectral_weight_mae_mean,
                "baseline_alpha_error_mean": baseline_alpha_error_mean,
                "spectral_alpha_error_mean": spectral_alpha_error_mean,
                "baseline_fit_rmse_mean": mean("baseline_fit_rmse"),
                "spectral_fit_rmse_mean": mean("spectral_fit_rmse"),
                "baseline_alpha_span_topk_mean": baseline_alpha_span_mean,
                "spectral_alpha_span_topk_mean": spectral_alpha_span_mean,
                "baseline_geometry_dispersion_topk_mean": mean("baseline_geometry_dispersion_topk"),
                "spectral_geometry_dispersion_topk_mean": mean("spectral_geometry_dispersion_topk"),
                "baseline_weight_dispersion_topk_mean": mean("baseline_weight_dispersion_topk"),
                "spectral_weight_dispersion_topk_mean": mean("spectral_weight_dispersion_topk"),
                "baseline_near_tie_diverse_fraction": mean("baseline_near_tie_diverse"),
                "spectral_near_tie_diverse_fraction": mean("spectral_near_tie_diverse"),
                "alpha_error_improvement_factor_mean": mean("alpha_error_improvement_factor"),
                "alpha_span_reduction_factor_mean": mean("alpha_span_reduction_factor"),
                "alpha_error_ratio_of_means_baseline_over_spectral": float(
                    baseline_alpha_error_mean / max(spectral_alpha_error_mean, 1.0e-12)
                ),
                "alpha_span_ratio_of_means_baseline_over_spectral": float(
                    baseline_alpha_span_mean / max(spectral_alpha_span_mean, 1.0e-12)
                ),
                "geometry_ratio_of_means_spectral_over_baseline": float(
                    spectral_geometry_mae_mean / max(baseline_geometry_mae_mean, 1.0e-12)
                ),
                "weight_ratio_of_means_spectral_over_baseline": float(
                    spectral_weight_mae_mean / max(baseline_weight_mae_mean, 1.0e-12)
                ),
                "geometry_error_ratio_spectral_over_baseline_mean": mean("geometry_error_ratio_spectral_over_baseline"),
                "weight_error_ratio_spectral_over_baseline_mean": mean("weight_error_ratio_spectral_over_baseline"),
                "spectral_design_condition_mean": mean("spectral_design_condition"),
            }
        )
    return summary

def plot_overview(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    x = np.arange(len(conditions))
    width = 0.36

    baseline_alpha = np.array([float(item["baseline_alpha_error_mean"]) for item in summary_rows])
    spectral_alpha = np.array([float(item["spectral_alpha_error_mean"]) for item in summary_rows])
    baseline_span = np.array([float(item["baseline_alpha_span_topk_mean"]) for item in summary_rows])
    spectral_span = np.array([float(item["spectral_alpha_span_topk_mean"]) for item in summary_rows])
    baseline_geom = np.array([float(item["baseline_geometry_mae_mean"]) for item in summary_rows])
    spectral_geom = np.array([float(item["spectral_geometry_mae_mean"]) for item in summary_rows])
    baseline_diverse = np.array([float(item["baseline_near_tie_diverse_fraction"]) for item in summary_rows])
    spectral_diverse = np.array([float(item["spectral_near_tie_diverse_fraction"]) for item in summary_rows])

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.98, wspace=0.24, hspace=0.34)

    for ax, baseline_vals, spectral_vals, ylabel, title in [
        (axes[0, 0], baseline_alpha, spectral_alpha, "mean alpha absolute error", "Alpha recovery under pose-free inversion"),
        (axes[0, 1], baseline_span, spectral_span, "mean top-10 alpha span", "Alpha ambiguity envelope under pose-free inversion"),
        (axes[1, 0], baseline_geom, spectral_geom, "mean geometry MAE", "Geometry recovery under pose-free inversion"),
        (axes[1, 1], baseline_diverse, spectral_diverse, "fraction of trials", "Near-tie and alpha-diverse trials"),
    ]:
        ax.bar(x - width / 2.0, baseline_vals, width=width, color="#e76f51", label="baseline shift search")
        ax.bar(x + width / 2.0, spectral_vals, width=width, color="#2a9d8f", label="spectral invariant")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0, 0].legend(loc="upper left", frameon=True)

    fig.suptitle("Rotation-Invariant Spectral Experiment A: Baseline Versus Spectral Invariant", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_trial_scatter(path: str, rows: list[TrialRow]) -> None:
    chosen_conditions = ["full_noisy", "partial_arc_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(1, len(chosen_conditions), figsize=(15.4, 5.0), constrained_layout=False)
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.28)

    for ax, condition in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == condition]
        baseline = np.array([row.baseline_alpha_error for row in subset])
        spectral = np.array([row.spectral_alpha_error for row in subset])
        design_cond = np.array([row.spectral_design_condition for row in subset])

        scatter = ax.scatter(
            baseline,
            spectral,
            c=np.log10(np.maximum(design_cond, 1.0)),
            cmap="viridis",
            s=42,
            alpha=0.82,
            edgecolors="none",
        )
        max_val = max(float(np.max(baseline)), float(np.max(spectral)), 1.0e-6)
        ax.plot([0.0, max_val], [0.0, max_val], color="#444444", linestyle="--", lw=1.4)
        ax.set_xlabel("baseline alpha error")
        ax.set_ylabel("spectral alpha error")
        ax.set_title(condition)

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.86)
    cbar.set_label("log10 design condition")
    fig.suptitle("Rotation-Invariant Spectral Experiment B: Trial-Level Alpha Error", fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    rng = np.random.default_rng(20260324)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    spectral_bank = build_spectral_bank(bank_signatures, MAX_HARMONIC)

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            baseline_scores, baseline_best_shift = pose_free_candidate_scores(observed_signature, mask, shifted_bank)
            baseline_metrics = ambiguity_metrics(baseline_scores, params_list, true_params, regime)
            baseline_best_idx = int(baseline_metrics["best_idx"])
            baseline_best_signature = shifted_bank[baseline_best_idx, int(baseline_best_shift[baseline_best_idx])]
            baseline_fit_rmse = float(np.sqrt(np.mean((baseline_best_signature - rotated_signature) ** 2)))

            spectral_feature, spectral_design_condition = spectral_feature_from_observation(
                observed_signature,
                mask,
                MAX_HARMONIC,
            )
            spectral_scores = spectral_candidate_scores(spectral_feature, spectral_bank)
            spectral_metrics = ambiguity_metrics(spectral_scores, params_list, true_params, regime)
            spectral_best_idx = int(spectral_metrics["best_idx"])
            spectral_best_signature, _ = best_shift_for_candidate(
                observed_signature,
                mask,
                shifted_bank[spectral_best_idx],
            )
            spectral_fit_rmse = float(np.sqrt(np.mean((spectral_best_signature - rotated_signature) ** 2)))

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
                    spectral_design_condition=float(spectral_design_condition),
                    baseline_geometry_mae=float(baseline_metrics["geometry_mae"]),
                    baseline_weight_mae=float(baseline_metrics["weight_mae"]),
                    baseline_alpha_error=float(baseline_metrics["alpha_error"]),
                    baseline_fit_rmse=float(baseline_fit_rmse),
                    baseline_alpha_span_topk=float(baseline_metrics["alpha_span_topk"]),
                    baseline_geometry_dispersion_topk=float(baseline_metrics["geometry_dispersion_topk"]),
                    baseline_weight_dispersion_topk=float(baseline_metrics["weight_dispersion_topk"]),
                    baseline_near_tie_diverse=int(baseline_metrics["near_tie_diverse"]),
                    spectral_geometry_mae=float(spectral_metrics["geometry_mae"]),
                    spectral_weight_mae=float(spectral_metrics["weight_mae"]),
                    spectral_alpha_error=float(spectral_metrics["alpha_error"]),
                    spectral_fit_rmse=float(spectral_fit_rmse),
                    spectral_alpha_span_topk=float(spectral_metrics["alpha_span_topk"]),
                    spectral_geometry_dispersion_topk=float(spectral_metrics["geometry_dispersion_topk"]),
                    spectral_weight_dispersion_topk=float(spectral_metrics["weight_dispersion_topk"]),
                    spectral_near_tie_diverse=int(spectral_metrics["near_tie_diverse"]),
                    alpha_error_improvement_factor=float(
                        baseline_metrics["alpha_error"] / max(spectral_metrics["alpha_error"], 1.0e-12)
                    ),
                    alpha_span_reduction_factor=float(
                        baseline_metrics["alpha_span_topk"] / max(spectral_metrics["alpha_span_topk"], 1.0e-12)
                    ),
                    geometry_error_ratio_spectral_over_baseline=float(
                        spectral_metrics["geometry_mae"] / max(baseline_metrics["geometry_mae"], 1.0e-12)
                    ),
                    weight_error_ratio_spectral_over_baseline=float(
                        spectral_metrics["weight_mae"] / max(baseline_metrics["weight_mae"], 1.0e-12)
                    ),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = summarize_trials(rows)
    write_csv(os.path.join(OUTPUT_DIR, "rotation_invariant_spectral_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "rotation_invariant_spectral_summary.csv"), summary_rows)

    plot_overview(os.path.join(FIGURE_DIR, "rotation_invariant_spectral_overview.png"), summary_rows)
    plot_trial_scatter(os.path.join(FIGURE_DIR, "rotation_invariant_spectral_trial_scatter.png"), rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "max_harmonic": MAX_HARMONIC,
        "top_k_envelope": TOP_K_ENVELOPE,
        "largest_alpha_error_improvement_factor_mean": float(
            max(item["alpha_error_improvement_factor_mean"] for item in summary_rows)
        ),
        "smallest_alpha_error_improvement_factor_mean": float(
            min(item["alpha_error_improvement_factor_mean"] for item in summary_rows)
        ),
        "largest_alpha_error_ratio_of_means_baseline_over_spectral": float(
            max(item["alpha_error_ratio_of_means_baseline_over_spectral"] for item in summary_rows)
        ),
        "smallest_alpha_error_ratio_of_means_baseline_over_spectral": float(
            min(item["alpha_error_ratio_of_means_baseline_over_spectral"] for item in summary_rows)
        ),
        "largest_alpha_span_reduction_factor_mean": float(
            max(item["alpha_span_reduction_factor_mean"] for item in summary_rows)
        ),
        "smallest_alpha_span_reduction_factor_mean": float(
            min(item["alpha_span_reduction_factor_mean"] for item in summary_rows)
        ),
        "largest_alpha_span_ratio_of_means_baseline_over_spectral": float(
            max(item["alpha_span_ratio_of_means_baseline_over_spectral"] for item in summary_rows)
        ),
        "smallest_alpha_span_ratio_of_means_baseline_over_spectral": float(
            min(item["alpha_span_ratio_of_means_baseline_over_spectral"] for item in summary_rows)
        ),
        "smallest_spectral_near_tie_diverse_fraction": float(
            min(item["spectral_near_tie_diverse_fraction"] for item in summary_rows)
        ),
        "largest_spectral_near_tie_diverse_fraction": float(
            max(item["spectral_near_tie_diverse_fraction"] for item in summary_rows)
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "rotation_invariant_spectral_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))

if __name__ == "__main__":
    main()
