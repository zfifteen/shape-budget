"""
Post-roadmap extension: targeted alpha refinement under combined nuisance.

This experiment keeps the same pose-free weighted anisotropic inverse problem,
but upgrades the inverse by directly refining the weak parameter alpha after an
initial bank match.

Method:
- use the existing pose-free anisotropic inverse to retrieve top-K candidate
  geometry-plus-weight states
- hold each candidate geometry and weight vector fixed
- jointly re-optimize alpha and rotation shift on a dense alpha grid
- keep the refined candidate with the best masked fit

This tests whether the alpha recovery challenge is partly a search-resolution problem
rather than a failure of the latent-object framework itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

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

ALPHA_MAX, ALPHA_MIN, EUCLIDEAN_BASELINE_BANK_SIZE, REFERENCE_BANK_SIZE, TEST_TRIALS_PER_REGIME, aggregate_trials, anisotropic_forward_signature, build_reference_bank, sample_anisotropic_parameters, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "EUCLIDEAN_BASELINE_BANK_SIZE",
    "REFERENCE_BANK_SIZE",
    "TEST_TRIALS_PER_REGIME",
    "aggregate_trials",
    "anisotropic_forward_signature",
    "build_reference_bank",
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

TOP_K_CANDIDATES = 2
COARSE_ALPHA_GRID = np.linspace(ALPHA_MIN, ALPHA_MAX, 17)
FINE_ALPHA_RADIUS = 0.04
FINE_ALPHA_POINTS = 9
PILOT_TRIALS_PER_REGIME = 10

@dataclass
class AlphaRefinementTrialRow:
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
    baseline_pred_rho: float
    baseline_pred_t: float
    baseline_pred_h: float
    baseline_pred_w1: float
    baseline_pred_w2: float
    baseline_pred_w3: float
    baseline_pred_alpha: float
    baseline_pred_rotation_shift: int
    refined_pred_rho: float
    refined_pred_t: float
    refined_pred_h: float
    refined_pred_w1: float
    refined_pred_w2: float
    refined_pred_w3: float
    refined_pred_alpha: float
    refined_pred_rotation_shift: int
    euclidean_pred_rho: float
    euclidean_pred_t: float
    euclidean_pred_h: float
    euclidean_pred_w1: float
    euclidean_pred_w2: float
    euclidean_pred_rotation_shift: int
    baseline_geometry_mae: float
    baseline_weight_mae: float
    baseline_alpha_abs_error: float
    baseline_fit_rmse: float
    refined_geometry_mae: float
    refined_weight_mae: float
    refined_alpha_abs_error: float
    refined_fit_rmse: float
    euclidean_fit_rmse: float
    alpha_improvement_factor: float
    fit_improvement_over_baseline: float
    fit_improvement_over_euclidean: float

def shifted_signature_mse(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
) -> np.ndarray:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    return np.mean(residual * residual, axis=2)

def top_k_unique_bank_candidates(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
    bank_params: list[tuple[float, float, float, float, float, float]],
    k: int,
) -> list[tuple[int, int, float]]:
    mse = shifted_signature_mse(observed_signature, mask, shifted_bank)
    flat_order = np.argsort(mse, axis=None)
    selected: list[tuple[int, int, float]] = []
    used_bank_indices: set[int] = set()

    for flat_idx in flat_order:
        bank_idx, shift_idx = np.unravel_index(int(flat_idx), mse.shape)
        if int(bank_idx) in used_bank_indices:
            continue
        used_bank_indices.add(int(bank_idx))
        selected.append((int(bank_idx), int(shift_idx), float(mse[bank_idx, shift_idx])))
        if len(selected) >= k:
            break
    return selected

def evaluate_alpha_and_shift(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    geometry_weight_params: tuple[float, float, float, float, float],
    alpha_grid: np.ndarray,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, int, float]:
    rho, t, h, w1, w2 = geometry_weight_params

    best_score = float("inf")
    best_params: tuple[float, float, float, float, float, float] | None = None
    best_signature: np.ndarray | None = None
    best_shift = 0

    for alpha in alpha_grid:
        params = (rho, t, h, w1, w2, float(alpha))
        signature = anisotropic_forward_signature(params)
        shift_stack = np.stack([np.roll(signature, shift) for shift in range(len(signature))], axis=0)
        residual = shift_stack[:, mask] - observed_signature[mask][None, :]
        mse = np.mean(residual * residual, axis=1)
        shift_idx = int(np.argmin(mse))
        score = float(mse[shift_idx])
        if score < best_score:
            best_score = score
            best_params = params
            best_signature = shift_stack[shift_idx]
            best_shift = shift_idx

    assert best_params is not None
    assert best_signature is not None
    return best_params, best_signature, best_shift, best_score

def refine_alpha_for_candidate(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    geometry_weight_params: tuple[float, float, float, float, float],
    seed_alpha: float,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, int, float]:
    coarse_grid = np.unique(np.concatenate([COARSE_ALPHA_GRID, np.array([seed_alpha], dtype=float)]))
    coarse_params, _, _, _ = evaluate_alpha_and_shift(observed_signature, mask, geometry_weight_params, coarse_grid)
    coarse_alpha = coarse_params[5]
    fine_grid = np.unique(
        np.concatenate(
            [
                np.linspace(
                    max(ALPHA_MIN, coarse_alpha - FINE_ALPHA_RADIUS),
                    min(ALPHA_MAX, coarse_alpha + FINE_ALPHA_RADIUS),
                    FINE_ALPHA_POINTS,
                ),
                np.array([coarse_alpha, seed_alpha], dtype=float),
            ]
        )
    )
    return evaluate_alpha_and_shift(observed_signature, mask, geometry_weight_params, fine_grid)

def summarize_method(rows: list[AlphaRefinementTrialRow], prefix: str) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]
        geometry = np.array([getattr(row, f"{prefix}_geometry_mae") for row in subset])
        weight = np.array([getattr(row, f"{prefix}_weight_mae") for row in subset])
        alpha = np.array([getattr(row, f"{prefix}_alpha_abs_error") for row in subset])
        fit = np.array([getattr(row, f"{prefix}_fit_rmse") for row in subset])
        summary.append(
            {
                "condition": name,
                "geometry_mae_mean": float(np.mean(geometry)),
                "geometry_mae_p95": float(np.quantile(geometry, 0.95)),
                "weight_mae_mean": float(np.mean(weight)),
                "weight_mae_p95": float(np.quantile(weight, 0.95)),
                "alpha_mae_mean": float(np.mean(alpha)),
                "alpha_mae_p95": float(np.quantile(alpha, 0.95)),
                "fit_rmse_mean": float(np.mean(fit)),
            }
        )
    return summary

def compare_methods(rows: list[AlphaRefinementTrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]
        base_g = np.array([row.baseline_geometry_mae for row in subset])
        base_w = np.array([row.baseline_weight_mae for row in subset])
        base_a = np.array([row.baseline_alpha_abs_error for row in subset])
        base_f = np.array([row.baseline_fit_rmse for row in subset])
        ref_g = np.array([row.refined_geometry_mae for row in subset])
        ref_w = np.array([row.refined_weight_mae for row in subset])
        ref_a = np.array([row.refined_alpha_abs_error for row in subset])
        ref_f = np.array([row.refined_fit_rmse for row in subset])
        eu_f = np.array([row.euclidean_fit_rmse for row in subset])
        summary.append(
            {
                "condition": name,
                "geometry_improvement_factor": float(np.mean(base_g) / max(float(np.mean(ref_g)), 1.0e-12)),
                "weight_improvement_factor": float(np.mean(base_w) / max(float(np.mean(ref_w)), 1.0e-12)),
                "alpha_improvement_factor": float(np.mean(base_a) / max(float(np.mean(ref_a)), 1.0e-12)),
                "fit_improvement_factor_vs_baseline": float(np.mean(base_f) / max(float(np.mean(ref_f)), 1.0e-12)),
                "fit_improvement_factor_vs_euclidean": float(np.mean(eu_f) / max(float(np.mean(ref_f)), 1.0e-12)),
            }
        )
    return summary

def plot_alpha_refinement_heatmap(
    path: str,
    baseline_rows: list[dict[str, float | str]],
    refined_rows: list[dict[str, float | str]],
    comparison_rows: list[dict[str, float | str]],
) -> None:
    conditions = [str(item["condition"]) for item in baseline_rows]
    baseline_alpha = np.array([[float(item["alpha_mae_mean"]) for item in baseline_rows]])
    refined_alpha = np.array([[float(item["alpha_mae_mean"]) for item in refined_rows]])
    improve = np.array([[float(item["alpha_improvement_factor"]) for item in comparison_rows]])

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.8), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.19, right=0.98, hspace=0.42)

    for ax, matrix, title, cmap, fmt, cbar in [
        (axes[0], baseline_alpha, "Baseline alpha error", "magma", ".3f", "mean absolute error"),
        (axes[1], refined_alpha, "Refined alpha error", "viridis", ".3f", "mean absolute error"),
        (axes[2], improve, "Alpha improvement factor (baseline / refined)", "crest", ".2f", "improvement factor"),
    ]:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            xticklabels=conditions,
            yticklabels=[title],
            cbar_kws={"label": cbar},
        )
        ax.set_title(title)
    axes[2].set_xlabel("observation regime")

    fig.suptitle(
        "Pose-Free Alpha Refinement A: Recovering Anisotropy By Direct Search",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_method_comparison(
    path: str,
    baseline_rows: list[dict[str, float | str]],
    refined_rows: list[dict[str, float | str]],
    comparison_rows: list[dict[str, float | str]],
) -> None:
    conditions = [str(item["condition"]) for item in baseline_rows]
    baseline_fit = np.array([float(item["fit_rmse_mean"]) for item in baseline_rows])
    refined_fit = np.array([float(item["fit_rmse_mean"]) for item in refined_rows])
    alpha_improve = np.array([float(item["alpha_improvement_factor"]) for item in comparison_rows])

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.20, wspace=0.30)

    x = np.arange(len(conditions))
    width = 0.36
    axes[0].bar(x - width / 2.0, baseline_fit, width=width, color="#e76f51", label="baseline")
    axes[0].bar(x + width / 2.0, refined_fit, width=width, color="#2a9d8f", label="alpha refined")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=20, ha="right")
    axes[0].set_ylabel("mean fit RMSE")
    axes[0].set_title("Fit quality under the same combined nuisance")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(x, alpha_improve, color="#1d3557", lw=2.4, marker="o")
    axes[1].axhline(1.0, color="#444444", linestyle="--", lw=1.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[1].set_ylabel("baseline / refined alpha error")
    axes[1].set_title("Alpha-identifiability gain")

    geometry_improve = np.array([float(item["geometry_improvement_factor"]) for item in comparison_rows])
    weight_improve = np.array([float(item["weight_improvement_factor"]) for item in comparison_rows])
    axes[2].plot(x, geometry_improve, color="#457b9d", lw=2.0, marker="s", label="geometry")
    axes[2].plot(x, weight_improve, color="#f4a261", lw=2.0, marker="^", label="weights")
    axes[2].axhline(1.0, color="#444444", linestyle="--", lw=1.4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(conditions, rotation=20, ha="right")
    axes[2].set_ylabel("baseline / refined error")
    axes[2].set_title("Collateral effect on geometry and weights")
    axes[2].legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Pose-Free Alpha Refinement B: Method Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_example_recoveries(path: str, rows: list[AlphaRefinementTrialRow]) -> None:
    chosen_conditions = ["full_noisy", "partial_arc_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(len(chosen_conditions), 1, figsize=(10.2, 9.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.40)

    for ax, name in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == name]
        exemplar = max(subset, key=lambda row: row.alpha_improvement_factor)
        true_sig = np.roll(
            anisotropic_forward_signature(
                (exemplar.true_rho, exemplar.true_t, exemplar.true_h, exemplar.true_w1, exemplar.true_w2, exemplar.true_alpha)
            ),
            exemplar.true_rotation_shift,
        )
        baseline_sig = np.roll(
            anisotropic_forward_signature(
                (
                    exemplar.baseline_pred_rho,
                    exemplar.baseline_pred_t,
                    exemplar.baseline_pred_h,
                    exemplar.baseline_pred_w1,
                    exemplar.baseline_pred_w2,
                    exemplar.baseline_pred_alpha,
                )
            ),
            exemplar.baseline_pred_rotation_shift,
        )
        refined_sig = np.roll(
            anisotropic_forward_signature(
                (
                    exemplar.refined_pred_rho,
                    exemplar.refined_pred_t,
                    exemplar.refined_pred_h,
                    exemplar.refined_pred_w1,
                    exemplar.refined_pred_w2,
                    exemplar.refined_pred_alpha,
                )
            ),
            exemplar.refined_pred_rotation_shift,
        )
        angle_grid = np.linspace(0.0, 2.0 * math.pi, len(true_sig), endpoint=False)

        ax.plot(angle_grid, true_sig, color="#222222", lw=2.4, label="true rotated signature")
        ax.plot(angle_grid, baseline_sig, color="#e76f51", lw=1.8, linestyle="--", label="baseline")
        ax.plot(angle_grid, refined_sig, color="#2a9d8f", lw=2.0, label="alpha refined")
        ax.set_title(
            f"{name}: alpha error {exemplar.baseline_alpha_abs_error:.3f} -> {exemplar.refined_alpha_abs_error:.3f}"
        )
        ax.set_xlabel("angle")
        ax.set_ylabel("normalized radius")
        if ax is axes[0]:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Pose-Free Alpha Refinement C: Representative Alpha-Recovery Improvements",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    rng = np.random.default_rng(20260324)
    anisotropic_params, anisotropic_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    euclidean_params, euclidean_signatures = build_reference_bank(EUCLIDEAN_BASELINE_BANK_SIZE, rng, anisotropic=False)
    anisotropic_shift_stack = build_shift_stack(anisotropic_signatures)
    euclidean_shift_stack = build_shift_stack(euclidean_signatures)

    rows: list[AlphaRefinementTrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(PILOT_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            baseline_pred_params, baseline_pred_signature, baseline_pred_shift = nearest_neighbor_pose_free(
                observed_signature,
                mask,
                anisotropic_shift_stack,
                anisotropic_params,
            )
            euclidean_pred_params, euclidean_pred_signature, euclidean_pred_shift = nearest_neighbor_pose_free(
                observed_signature,
                mask,
                euclidean_shift_stack,
                euclidean_params,
            )

            top_candidates = top_k_unique_bank_candidates(
                observed_signature,
                mask,
                anisotropic_shift_stack,
                anisotropic_params,
                TOP_K_CANDIDATES,
            )

            refined_best_score = float("inf")
            refined_best_params: tuple[float, float, float, float, float, float] | None = None
            refined_best_signature: np.ndarray | None = None
            refined_best_shift = 0

            for bank_idx, _, _ in top_candidates:
                rho, t, h, w1, w2, seed_alpha = anisotropic_params[bank_idx]
                refined_params, refined_signature, refined_shift, refined_score = refine_alpha_for_candidate(
                    observed_signature,
                    mask,
                    (rho, t, h, w1, w2),
                    float(seed_alpha),
                )
                if refined_score < refined_best_score:
                    refined_best_score = refined_score
                    refined_best_params = refined_params
                    refined_best_signature = refined_signature
                    refined_best_shift = refined_shift

            assert refined_best_params is not None
            assert refined_best_signature is not None

            baseline_geom, baseline_weight, baseline_alpha = symmetry_aware_errors(true_params, baseline_pred_params)
            refined_geom, refined_weight, refined_alpha = symmetry_aware_errors(true_params, refined_best_params)
            baseline_fit = float(np.sqrt(np.mean((baseline_pred_signature - rotated_signature) ** 2)))
            refined_fit = float(np.sqrt(np.mean((refined_best_signature - rotated_signature) ** 2)))
            euclidean_fit = float(np.sqrt(np.mean((euclidean_pred_signature - rotated_signature) ** 2)))

            rows.append(
                AlphaRefinementTrialRow(
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
                    baseline_pred_rho=float(baseline_pred_params[0]),
                    baseline_pred_t=float(baseline_pred_params[1]),
                    baseline_pred_h=float(baseline_pred_params[2]),
                    baseline_pred_w1=float(baseline_pred_params[3]),
                    baseline_pred_w2=float(baseline_pred_params[4]),
                    baseline_pred_w3=float(1.0 - baseline_pred_params[3] - baseline_pred_params[4]),
                    baseline_pred_alpha=float(baseline_pred_params[5]),
                    baseline_pred_rotation_shift=int(baseline_pred_shift),
                    refined_pred_rho=float(refined_best_params[0]),
                    refined_pred_t=float(refined_best_params[1]),
                    refined_pred_h=float(refined_best_params[2]),
                    refined_pred_w1=float(refined_best_params[3]),
                    refined_pred_w2=float(refined_best_params[4]),
                    refined_pred_w3=float(1.0 - refined_best_params[3] - refined_best_params[4]),
                    refined_pred_alpha=float(refined_best_params[5]),
                    refined_pred_rotation_shift=int(refined_best_shift),
                    euclidean_pred_rho=float(euclidean_pred_params[0]),
                    euclidean_pred_t=float(euclidean_pred_params[1]),
                    euclidean_pred_h=float(euclidean_pred_params[2]),
                    euclidean_pred_w1=float(euclidean_pred_params[3]),
                    euclidean_pred_w2=float(euclidean_pred_params[4]),
                    euclidean_pred_rotation_shift=int(euclidean_pred_shift),
                    baseline_geometry_mae=baseline_geom,
                    baseline_weight_mae=baseline_weight,
                    baseline_alpha_abs_error=baseline_alpha,
                    baseline_fit_rmse=baseline_fit,
                    refined_geometry_mae=refined_geom,
                    refined_weight_mae=refined_weight,
                    refined_alpha_abs_error=refined_alpha,
                    refined_fit_rmse=refined_fit,
                    euclidean_fit_rmse=euclidean_fit,
                    alpha_improvement_factor=float(baseline_alpha / max(refined_alpha, 1.0e-12)),
                    fit_improvement_over_baseline=float(baseline_fit / max(refined_fit, 1.0e-12)),
                    fit_improvement_over_euclidean=float(euclidean_fit / max(refined_fit, 1.0e-12)),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    baseline_summary = summarize_method(rows, "baseline")
    refined_summary = summarize_method(rows, "refined")
    comparison_rows = compare_methods(rows)

    write_csv(os.path.join(OUTPUT_DIR, "pose_free_alpha_refinement_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "pose_free_alpha_refinement_baseline_summary.csv"), baseline_summary)
    write_csv(os.path.join(OUTPUT_DIR, "pose_free_alpha_refinement_refined_summary.csv"), refined_summary)
    write_csv(os.path.join(OUTPUT_DIR, "pose_free_alpha_refinement_comparison.csv"), comparison_rows)

    plot_alpha_refinement_heatmap(
        os.path.join(FIGURE_DIR, "pose_free_alpha_refinement_heatmap.png"),
        baseline_summary,
        refined_summary,
        comparison_rows,
    )
    plot_method_comparison(
        os.path.join(FIGURE_DIR, "pose_free_alpha_refinement_method_comparison.png"),
        baseline_summary,
        refined_summary,
        comparison_rows,
    )
    plot_example_recoveries(
        os.path.join(FIGURE_DIR, "pose_free_alpha_refinement_examples.png"),
        rows,
    )

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "euclidean_baseline_bank_size": EUCLIDEAN_BASELINE_BANK_SIZE,
        "test_trials_per_regime": int(PILOT_TRIALS_PER_REGIME),
        "top_k_candidates": TOP_K_CANDIDATES,
        "coarse_alpha_grid_size": int(len(COARSE_ALPHA_GRID)),
        "fine_alpha_grid_size": int(FINE_ALPHA_POINTS),
        "best_baseline_alpha_mae_mean": float(min(item["alpha_mae_mean"] for item in baseline_summary)),
        "worst_baseline_alpha_mae_mean": float(max(item["alpha_mae_mean"] for item in baseline_summary)),
        "best_refined_alpha_mae_mean": float(min(item["alpha_mae_mean"] for item in refined_summary)),
        "worst_refined_alpha_mae_mean": float(max(item["alpha_mae_mean"] for item in refined_summary)),
        "smallest_alpha_improvement_factor": float(min(item["alpha_improvement_factor"] for item in comparison_rows)),
        "largest_alpha_improvement_factor": float(max(item["alpha_improvement_factor"] for item in comparison_rows)),
        "smallest_fit_improvement_over_baseline": float(min(item["fit_improvement_factor_vs_baseline"] for item in comparison_rows)),
        "largest_fit_improvement_over_baseline": float(max(item["fit_improvement_factor_vs_baseline"] for item in comparison_rows)),
        "smallest_fit_improvement_over_euclidean": float(min(item["fit_improvement_factor_vs_euclidean"] for item in comparison_rows)),
        "largest_fit_improvement_over_euclidean": float(max(item["fit_improvement_factor_vs_euclidean"] for item in comparison_rows)),
    }

    with open(os.path.join(OUTPUT_DIR, "pose_free_alpha_refinement_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "baseline_by_condition": baseline_summary,
                "refined_by_condition": refined_summary,
                "comparison_by_condition": comparison_rows,
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "summary": summary,
                "baseline_by_condition": baseline_summary,
                "refined_by_condition": refined_summary,
                "comparison_by_condition": comparison_rows,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
