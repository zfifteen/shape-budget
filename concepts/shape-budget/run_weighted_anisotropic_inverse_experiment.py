"""
Post-roadmap extension: weighted anisotropic inverse experiment for Shape Budget.

This experiment asks whether the weighted three-source control object remains
recoverable when the medium itself introduces a single axis-aligned anisotropy
parameter alpha:

    d_alpha((x, y), (u, v)) = sqrt((x-u)^2 + alpha^2 (y-v)^2)

The recovery target is the joint latent object:

- normalized raw source-triangle geometry relative to budget
- normalized weight vector in the simplex
- anisotropy parameter alpha

This is a controlled warped-medium inverse:

- canonical source orientation is retained
- translation and scale are removed from the boundary observation
- alpha is unknown and must be inferred jointly with geometry and weights
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from run_weighted_multisource_experiment import canonical_sources, normalize_weights, weighted_boundary_curve
from run_weighted_multisource_inverse_experiment import (
    GEOMETRY_BOUNDS,
    OBSERVATION_REGIMES,
    boundary_signature_from_curve,
    observe_signature,
    write_csv,
)


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
OUTPUT_DIR = os.path.join(BASE_DIR, "weighted_anisotropic_inverse_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


REFERENCE_BANK_SIZE = 300
EUCLIDEAN_BASELINE_BANK_SIZE = 150
TEST_TRIALS_PER_REGIME = 40
CURVE_SAMPLE_COUNT = 96
SIGNATURE_ANGLE_COUNT = 64
ALPHA_MIN = 0.60
ALPHA_MAX = 1.80


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
    pred_rho: float
    pred_t: float
    pred_h: float
    pred_w1: float
    pred_w2: float
    pred_w3: float
    pred_alpha: float
    euclidean_pred_rho: float
    euclidean_pred_t: float
    euclidean_pred_h: float
    euclidean_pred_w1: float
    euclidean_pred_w2: float
    geometry_mae: float
    weight_mae: float
    alpha_abs_error: float
    anisotropic_fit_rmse: float
    euclidean_baseline_fit_rmse: float
    fit_improvement_factor: float


def sample_anisotropic_parameters(rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
    rho = float(rng.uniform(GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(rng.uniform(GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(rng.uniform(GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    weights = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
    alpha = float(rng.uniform(ALPHA_MIN, ALPHA_MAX))
    return rho, t, h, float(weights[0]), float(weights[1]), alpha


def sample_euclidean_parameters(rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
    rho = float(rng.uniform(GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(rng.uniform(GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(rng.uniform(GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    weights = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
    return rho, t, h, float(weights[0]), float(weights[1]), 1.0


def anisotropic_forward_signature(params: tuple[float, float, float, float, float, float]) -> np.ndarray:
    rho, t, h, w1, w2, alpha = params
    weights = normalize_weights(np.array([w1, w2, 1.0 - w1 - w2], dtype=float))

    points_raw = canonical_sources(rho, t, h, S=1.0)
    points_white = points_raw.copy()
    points_white[:, 1] *= alpha

    _, _, curve_white = weighted_boundary_curve(points_white, weights, 1.0, angle_count=CURVE_SAMPLE_COUNT)
    curve_raw = curve_white.copy()
    curve_raw[:, 1] /= alpha
    return boundary_signature_from_curve(curve_raw, angle_count=SIGNATURE_ANGLE_COUNT)


def control_invariants(params: tuple[float, float, float, float, float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    rho, t, h, w1, w2, alpha = params
    points = canonical_sources(rho, t, h, S=1.0)
    d12 = np.linalg.norm(points[0] - points[1])
    d13 = np.linalg.norm(points[0] - points[2])
    d23 = np.linalg.norm(points[1] - points[2])
    return np.array([d12, d13, d23]), np.array([w1, w2, 1.0 - w1 - w2]), float(alpha)


def symmetry_aware_errors(
    true_params: tuple[float, float, float, float, float, float],
    pred_params: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float]:
    true_geom, true_weights, true_alpha = control_invariants(true_params)
    pred_geom, pred_weights, pred_alpha = control_invariants(pred_params)

    direct_geom_mae = float(np.mean(np.abs(true_geom - pred_geom)))
    direct_weight_mae = float(np.mean(np.abs(true_weights - pred_weights)))

    swapped_geom = np.array([pred_geom[0], pred_geom[2], pred_geom[1]])
    swapped_weights = np.array([pred_weights[1], pred_weights[0], pred_weights[2]])
    swapped_geom_mae = float(np.mean(np.abs(true_geom - swapped_geom)))
    swapped_weight_mae = float(np.mean(np.abs(true_weights - swapped_weights)))

    if direct_geom_mae + direct_weight_mae <= swapped_geom_mae + swapped_weight_mae:
        geom_mae = direct_geom_mae
        weight_mae = direct_weight_mae
    else:
        geom_mae = swapped_geom_mae
        weight_mae = swapped_weight_mae

    return geom_mae, weight_mae, float(abs(true_alpha - pred_alpha))


def build_reference_bank(
    sample_size: int,
    rng: np.random.Generator,
    anisotropic: bool,
) -> tuple[list[tuple[float, float, float, float, float, float]], np.ndarray]:
    params_list: list[tuple[float, float, float, float, float, float]] = []
    signatures: list[np.ndarray] = []
    sampler = sample_anisotropic_parameters if anisotropic else sample_euclidean_parameters

    while len(params_list) < sample_size:
        params = sampler(rng)
        params_list.append(params)
        signatures.append(anisotropic_forward_signature(params))
    return params_list, np.array(signatures)


def nearest_neighbor_prediction(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    bank_signatures: np.ndarray,
    bank_params: list[tuple[float, float, float, float, float, float]],
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray]:
    residual = bank_signatures[:, mask] - observed_signature[mask]
    mse = np.mean(residual * residual, axis=1)
    idx = int(np.argmin(mse))
    return bank_params[idx], bank_signatures[idx]


def aggregate_trials(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]
        geometry = np.array([row.geometry_mae for row in subset])
        weight = np.array([row.weight_mae for row in subset])
        alpha = np.array([row.alpha_abs_error for row in subset])
        fit = np.array([row.anisotropic_fit_rmse for row in subset])
        baseline = np.array([row.euclidean_baseline_fit_rmse for row in subset])
        improve = np.array([row.fit_improvement_factor for row in subset])
        summary.append(
            {
                "condition": name,
                "geometry_mae_mean": float(np.mean(geometry)),
                "geometry_mae_p95": float(np.quantile(geometry, 0.95)),
                "weight_mae_mean": float(np.mean(weight)),
                "weight_mae_p95": float(np.quantile(weight, 0.95)),
                "alpha_mae_mean": float(np.mean(alpha)),
                "alpha_mae_p95": float(np.quantile(alpha, 0.95)),
                "anisotropic_fit_rmse_mean": float(np.mean(fit)),
                "euclidean_baseline_fit_rmse_mean": float(np.mean(baseline)),
                "fit_improvement_factor_mean": float(np.mean(improve)),
            }
        )
    return summary


def plot_error_heatmap(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    geometry = np.array([[float(item["geometry_mae_mean"]) for item in summary_rows]])
    weights = np.array([[float(item["weight_mae_mean"]) for item in summary_rows]])
    alpha = np.array([[float(item["alpha_mae_mean"]) for item in summary_rows]])

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.8), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.19, right=0.98, hspace=0.42)

    for ax, matrix, title, cmap, label in [
        (axes[0], geometry, "Recovery of normalized geometry", "viridis", "mean absolute error"),
        (axes[1], weights, "Recovery of normalized weights", "magma", "mean absolute error"),
        (axes[2], alpha, "Recovery of anisotropy alpha", "crest", "mean absolute error"),
    ]:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".3f",
            xticklabels=conditions,
            yticklabels=[title.split()[-1] + " MAE" if title.startswith("Recovery of anisotropy") else title.split()[-1] + " MAE"],
            cbar_kws={"label": label},
        )
        ax.set_title(title)
    axes[2].set_xlabel("observation regime")

    fig.suptitle("Weighted Anisotropic Inverse A: Recovery Error Across Regimes", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_baselines_and_alpha(
    path: str,
    summary_rows: list[dict[str, float | str]],
    rows: list[TrialRow],
) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    weighted_fit = np.array([float(item["anisotropic_fit_rmse_mean"]) for item in summary_rows])
    euclidean_fit = np.array([float(item["euclidean_baseline_fit_rmse_mean"]) for item in summary_rows])
    improve = np.array([float(item["fit_improvement_factor_mean"]) for item in summary_rows])

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.20, wspace=0.28)

    x = np.arange(len(conditions))
    width = 0.36
    axes[0].bar(x - width / 2.0, weighted_fit, width=width, color="#2a9d8f", label="anisotropy-aware bank")
    axes[0].bar(x + width / 2.0, euclidean_fit, width=width, color="#e76f51", label="euclidean baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=20, ha="right")
    axes[0].set_ylabel("mean clean-signature RMSE")
    axes[0].set_title("Anisotropy-aware bank beats the Euclidean shortcut")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(x, improve, color="#1d3557", lw=2.4, marker="o")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[1].set_ylabel("euclidean / anisotropy-aware fit ratio")
    axes[1].set_title("Average fit-improvement factor")

    condition_to_color = {item["name"]: color for item, color in zip(OBSERVATION_REGIMES, sns.color_palette("viridis", n_colors=len(OBSERVATION_REGIMES)))}
    for name in [item["name"] for item in OBSERVATION_REGIMES]:
        subset = [row for row in rows if row.condition == name]
        axes[2].scatter(
            [row.true_alpha for row in subset],
            [row.pred_alpha for row in subset],
            s=20,
            alpha=0.72,
            color=condition_to_color[name],
            label=name,
        )
    axes[2].plot([ALPHA_MIN, ALPHA_MAX], [ALPHA_MIN, ALPHA_MAX], color="#333333", linestyle="--", lw=1.4)
    axes[2].set_xlim(ALPHA_MIN - 0.03, ALPHA_MAX + 0.03)
    axes[2].set_ylim(ALPHA_MIN - 0.03, ALPHA_MAX + 0.03)
    axes[2].set_xlabel("true alpha")
    axes[2].set_ylabel("predicted alpha")
    axes[2].set_title("Joint recovery of anisotropy")
    axes[2].legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle("Weighted Anisotropic Inverse B: Baselines And Alpha Recovery", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_example_recoveries(path: str, rows: list[TrialRow]) -> None:
    chosen_conditions = ["full_clean", "partial_arc_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(len(chosen_conditions), 1, figsize=(10.2, 9.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.40)

    for ax, name in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == name]
        exemplar = min(subset, key=lambda row: row.anisotropic_fit_rmse)
        true_sig = anisotropic_forward_signature(
            (exemplar.true_rho, exemplar.true_t, exemplar.true_h, exemplar.true_w1, exemplar.true_w2, exemplar.true_alpha)
        )
        pred_sig = anisotropic_forward_signature(
            (exemplar.pred_rho, exemplar.pred_t, exemplar.pred_h, exemplar.pred_w1, exemplar.pred_w2, exemplar.pred_alpha)
        )
        euclidean_sig = anisotropic_forward_signature(
            (exemplar.euclidean_pred_rho, exemplar.euclidean_pred_t, exemplar.euclidean_pred_h, exemplar.euclidean_pred_w1, exemplar.euclidean_pred_w2, 1.0)
        )
        angle_grid = np.linspace(0.0, 2.0 * np.pi, len(true_sig), endpoint=False)

        ax.plot(angle_grid, true_sig, color="#222222", lw=2.4, label="true raw signature")
        ax.plot(angle_grid, pred_sig, color="#2a9d8f", lw=2.0, label="anisotropy-aware recovery")
        ax.plot(angle_grid, euclidean_sig, color="#e76f51", lw=1.8, linestyle="--", label="euclidean baseline")
        ax.set_title(
            f"{name}: geometry MAE = {exemplar.geometry_mae:.3f}, weight MAE = {exemplar.weight_mae:.3f}, alpha error = {exemplar.alpha_abs_error:.3f}"
        )
        ax.set_xlabel("angle")
        ax.set_ylabel("normalized radius")
        if ax is axes[0]:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Weighted Anisotropic Inverse C: Representative Signature Recoveries", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)
    anisotropic_params, anisotropic_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    euclidean_params, euclidean_signatures = build_reference_bank(EUCLIDEAN_BASELINE_BANK_SIZE, rng, anisotropic=False)

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            observed_signature, mask = observe_signature(clean_signature, regime, rng)

            pred_params, pred_signature = nearest_neighbor_prediction(
                observed_signature,
                mask,
                anisotropic_signatures,
                anisotropic_params,
            )
            euclidean_pred_params, euclidean_signature = nearest_neighbor_prediction(
                observed_signature,
                mask,
                euclidean_signatures,
                euclidean_params,
            )

            geometry_mae, weight_mae, alpha_abs_error = symmetry_aware_errors(true_params, pred_params)
            anisotropic_fit_rmse = float(np.sqrt(np.mean((pred_signature - clean_signature) ** 2)))
            euclidean_fit_rmse = float(np.sqrt(np.mean((euclidean_signature - clean_signature) ** 2)))

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
                    pred_rho=float(pred_params[0]),
                    pred_t=float(pred_params[1]),
                    pred_h=float(pred_params[2]),
                    pred_w1=float(pred_params[3]),
                    pred_w2=float(pred_params[4]),
                    pred_w3=float(1.0 - pred_params[3] - pred_params[4]),
                    pred_alpha=float(pred_params[5]),
                    euclidean_pred_rho=float(euclidean_pred_params[0]),
                    euclidean_pred_t=float(euclidean_pred_params[1]),
                    euclidean_pred_h=float(euclidean_pred_params[2]),
                    euclidean_pred_w1=float(euclidean_pred_params[3]),
                    euclidean_pred_w2=float(euclidean_pred_params[4]),
                    geometry_mae=geometry_mae,
                    weight_mae=weight_mae,
                    alpha_abs_error=alpha_abs_error,
                    anisotropic_fit_rmse=anisotropic_fit_rmse,
                    euclidean_baseline_fit_rmse=euclidean_fit_rmse,
                    fit_improvement_factor=float(euclidean_fit_rmse / max(anisotropic_fit_rmse, 1.0e-12)),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = aggregate_trials(rows)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_anisotropic_inverse_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_anisotropic_inverse_summary.csv"), summary_rows)

    plot_error_heatmap(os.path.join(FIGURE_DIR, "weighted_anisotropic_inverse_heatmap.png"), summary_rows)
    plot_baselines_and_alpha(os.path.join(FIGURE_DIR, "weighted_anisotropic_inverse_baselines.png"), summary_rows, rows)
    plot_example_recoveries(os.path.join(FIGURE_DIR, "weighted_anisotropic_inverse_examples.png"), rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "euclidean_baseline_bank_size": EUCLIDEAN_BASELINE_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "best_geometry_mae_mean": float(min(item["geometry_mae_mean"] for item in summary_rows)),
        "worst_geometry_mae_mean": float(max(item["geometry_mae_mean"] for item in summary_rows)),
        "best_weight_mae_mean": float(min(item["weight_mae_mean"] for item in summary_rows)),
        "worst_weight_mae_mean": float(max(item["weight_mae_mean"] for item in summary_rows)),
        "best_alpha_mae_mean": float(min(item["alpha_mae_mean"] for item in summary_rows)),
        "worst_alpha_mae_mean": float(max(item["alpha_mae_mean"] for item in summary_rows)),
        "best_anisotropic_fit_rmse_mean": float(min(item["anisotropic_fit_rmse_mean"] for item in summary_rows)),
        "worst_anisotropic_fit_rmse_mean": float(max(item["anisotropic_fit_rmse_mean"] for item in summary_rows)),
        "smallest_fit_improvement_factor_mean": float(min(item["fit_improvement_factor_mean"] for item in summary_rows)),
        "largest_fit_improvement_factor_mean": float(max(item["fit_improvement_factor_mean"] for item in summary_rows)),
    }

    with open(os.path.join(OUTPUT_DIR, "weighted_anisotropic_inverse_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
