"""
Post-roadmap extension: pose-free weighted anisotropic inverse experiment.

This experiment combines the two nuisance variables isolated in the earlier
inverse extensions:

- unknown rotation of the observed boundary signature
- unknown axis-aligned anisotropy parameter alpha

The forward family is still the weighted three-source constant-sum boundary
under the controlled quadratic anisotropic metric

    d_alpha((x, y), (u, v)) = sqrt((x-u)^2 + alpha^2 (y-v)^2)

The latent recovery target is:

- normalized source-triangle geometry relative to budget
- normalized weight vector in the simplex
- anisotropy parameter alpha

Scope note:
- the anisotropy axis itself is still not unknown
- the rotation here is an observational pose nuisance handled as a cyclic shift
  on the radial signature grid
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

ALPHA_MAX, ALPHA_MIN, EUCLIDEAN_BASELINE_BANK_SIZE, CANONICAL_ANISO_OUTPUT_DIR, REFERENCE_BANK_SIZE, TEST_TRIALS_PER_REGIME, aggregate_trials, anisotropic_forward_signature, build_reference_bank, sample_anisotropic_parameters, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "EUCLIDEAN_BASELINE_BANK_SIZE",
    "OUTPUT_DIR",
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

@dataclass
class PoseFreeAnisotropicTrialRow:
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
    pred_rho: float
    pred_t: float
    pred_h: float
    pred_w1: float
    pred_w2: float
    pred_w3: float
    pred_alpha: float
    pred_rotation_shift: int
    euclidean_pred_rho: float
    euclidean_pred_t: float
    euclidean_pred_h: float
    euclidean_pred_w1: float
    euclidean_pred_w2: float
    euclidean_pred_rotation_shift: int
    geometry_mae: float
    weight_mae: float
    alpha_abs_error: float
    anisotropic_fit_rmse: float
    euclidean_baseline_fit_rmse: float
    fit_improvement_factor: float

def nearest_neighbor_pose_free(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
    bank_params: list[tuple[float, float, float, float, float, float]],
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, int]:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    mse = np.mean(residual * residual, axis=2)
    best_flat = int(np.argmin(mse))
    bank_idx, shift_idx = np.unravel_index(best_flat, mse.shape)
    return bank_params[bank_idx], shifted_bank[bank_idx, shift_idx], int(shift_idx)

def load_canonical_anisotropic_summary() -> dict[str, dict[str, float]]:
    path = os.path.join(CANONICAL_ANISO_OUTPUT_DIR, "weighted_anisotropic_inverse_summary.json")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {item["condition"]: item for item in data["by_condition"]}

def compare_to_canonical_anisotropic(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    canonical = load_canonical_anisotropic_summary()
    rows: list[dict[str, float | str]] = []
    for row in summary_rows:
        name = str(row["condition"])
        base = canonical[name]
        rows.append(
            {
                "condition": name,
                "geometry_mae_penalty_factor": float(row["geometry_mae_mean"]) / float(base["geometry_mae_mean"]),
                "weight_mae_penalty_factor": float(row["weight_mae_mean"]) / float(base["weight_mae_mean"]),
                "alpha_mae_penalty_factor": float(row["alpha_mae_mean"]) / float(base["alpha_mae_mean"]),
                "anisotropic_fit_rmse_penalty_factor": float(row["anisotropic_fit_rmse_mean"]) / float(base["anisotropic_fit_rmse_mean"]),
                "fit_improvement_ratio_vs_canonical": float(row["fit_improvement_factor_mean"]) / float(base["fit_improvement_factor_mean"]),
            }
        )
    return rows

def plot_error_heatmap(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    geometry = np.array([[float(item["geometry_mae_mean"]) for item in summary_rows]])
    weights = np.array([[float(item["weight_mae_mean"]) for item in summary_rows]])
    alpha = np.array([[float(item["alpha_mae_mean"]) for item in summary_rows]])

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.8), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.19, right=0.98, hspace=0.42)

    for ax, matrix, title, ylabel, cmap in [
        (axes[0], geometry, "Pose-free recovery of normalized geometry", "geometry MAE", "viridis"),
        (axes[1], weights, "Pose-free recovery of normalized weights", "weight MAE", "magma"),
        (axes[2], alpha, "Pose-free recovery of anisotropy alpha", "alpha MAE", "crest"),
    ]:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".3f",
            xticklabels=conditions,
            yticklabels=[ylabel],
            cbar_kws={"label": "mean absolute error"},
        )
        ax.set_title(title)
    axes[2].set_xlabel("observation regime")

    fig.suptitle(
        "Pose-Free Weighted Anisotropic Inverse A: Recovery Error Across Regimes",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_baseline_and_penalty(
    path: str,
    summary_rows: list[dict[str, float | str]],
    penalty_rows: list[dict[str, float | str]],
) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    anisotropic_fit = np.array([float(item["anisotropic_fit_rmse_mean"]) for item in summary_rows])
    euclidean_fit = np.array([float(item["euclidean_baseline_fit_rmse_mean"]) for item in summary_rows])
    improve = np.array([float(item["fit_improvement_factor_mean"]) for item in summary_rows])
    rotation_penalty = np.array([float(item["anisotropic_fit_rmse_penalty_factor"]) for item in penalty_rows])

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.20, wspace=0.28)

    x = np.arange(len(conditions))
    width = 0.36
    axes[0].bar(x - width / 2.0, anisotropic_fit, width=width, color="#2a9d8f", label="anisotropy-aware bank")
    axes[0].bar(x + width / 2.0, euclidean_fit, width=width, color="#e76f51", label="euclidean baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=20, ha="right")
    axes[0].set_ylabel("mean rotated-signature RMSE")
    axes[0].set_title("Anisotropy-aware bank still beats the Euclidean shortcut")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(x, improve, color="#1d3557", lw=2.4, marker="o")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[1].set_ylabel("euclidean / anisotropy-aware fit ratio")
    axes[1].set_title("Average fit-improvement factor")

    axes[2].plot(x, rotation_penalty, color="#6a4c93", lw=2.4, marker="s")
    axes[2].axhline(1.0, color="#444444", linestyle="--", lw=1.4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(conditions, rotation=20, ha="right")
    axes[2].set_ylabel("pose-free / canonical fit RMSE")
    axes[2].set_title("Rotation penalty relative to canonical anisotropic inverse")

    fig.suptitle(
        "Pose-Free Weighted Anisotropic Inverse B: Baseline And Rotation Penalty",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_example_recoveries(path: str, rows: list[PoseFreeAnisotropicTrialRow]) -> None:
    chosen_conditions = ["full_clean", "partial_arc_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(len(chosen_conditions), 1, figsize=(10.2, 9.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.40)

    for ax, name in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == name]
        exemplar = min(subset, key=lambda row: row.anisotropic_fit_rmse)
        true_sig = np.roll(
            anisotropic_forward_signature(
                (exemplar.true_rho, exemplar.true_t, exemplar.true_h, exemplar.true_w1, exemplar.true_w2, exemplar.true_alpha)
            ),
            exemplar.true_rotation_shift,
        )
        pred_sig = np.roll(
            anisotropic_forward_signature(
                (exemplar.pred_rho, exemplar.pred_t, exemplar.pred_h, exemplar.pred_w1, exemplar.pred_w2, exemplar.pred_alpha)
            ),
            exemplar.pred_rotation_shift,
        )
        euclidean_sig = np.roll(
            anisotropic_forward_signature(
                (exemplar.euclidean_pred_rho, exemplar.euclidean_pred_t, exemplar.euclidean_pred_h, exemplar.euclidean_pred_w1, exemplar.euclidean_pred_w2, 1.0)
            ),
            exemplar.euclidean_pred_rotation_shift,
        )
        angle_grid = np.linspace(0.0, 2.0 * math.pi, len(true_sig), endpoint=False)

        ax.plot(angle_grid, true_sig, color="#222222", lw=2.4, label="true rotated signature")
        ax.plot(angle_grid, pred_sig, color="#2a9d8f", lw=2.0, label="anisotropy-aware recovery")
        ax.plot(angle_grid, euclidean_sig, color="#e76f51", lw=1.8, linestyle="--", label="euclidean baseline")
        ax.set_title(
            f"{name}: geometry MAE = {exemplar.geometry_mae:.3f}, weight MAE = {exemplar.weight_mae:.3f}, alpha error = {exemplar.alpha_abs_error:.3f}"
        )
        ax.set_xlabel("angle")
        ax.set_ylabel("normalized radius")
        if ax is axes[0]:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Pose-Free Weighted Anisotropic Inverse C: Representative Rotated Signature Recoveries",
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

    rows: list[PoseFreeAnisotropicTrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            pred_params, pred_signature, pred_shift = nearest_neighbor_pose_free(
                observed_signature,
                mask,
                anisotropic_shift_stack,
                anisotropic_params,
            )
            euclidean_pred_params, euclidean_signature, euclidean_pred_shift = nearest_neighbor_pose_free(
                observed_signature,
                mask,
                euclidean_shift_stack,
                euclidean_params,
            )

            geometry_mae, weight_mae, alpha_abs_error = symmetry_aware_errors(true_params, pred_params)
            anisotropic_fit_rmse = float(np.sqrt(np.mean((pred_signature - rotated_signature) ** 2)))
            euclidean_fit_rmse = float(np.sqrt(np.mean((euclidean_signature - rotated_signature) ** 2)))

            rows.append(
                PoseFreeAnisotropicTrialRow(
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
                    pred_rho=float(pred_params[0]),
                    pred_t=float(pred_params[1]),
                    pred_h=float(pred_params[2]),
                    pred_w1=float(pred_params[3]),
                    pred_w2=float(pred_params[4]),
                    pred_w3=float(1.0 - pred_params[3] - pred_params[4]),
                    pred_alpha=float(pred_params[5]),
                    pred_rotation_shift=int(pred_shift),
                    euclidean_pred_rho=float(euclidean_pred_params[0]),
                    euclidean_pred_t=float(euclidean_pred_params[1]),
                    euclidean_pred_h=float(euclidean_pred_params[2]),
                    euclidean_pred_w1=float(euclidean_pred_params[3]),
                    euclidean_pred_w2=float(euclidean_pred_params[4]),
                    euclidean_pred_rotation_shift=int(euclidean_pred_shift),
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
    penalty_rows = compare_to_canonical_anisotropic(summary_rows)

    write_csv(os.path.join(OUTPUT_DIR, "pose_free_weighted_anisotropic_inverse_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "pose_free_weighted_anisotropic_inverse_summary.csv"), summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "pose_free_weighted_anisotropic_inverse_penalties.csv"), penalty_rows)

    plot_error_heatmap(os.path.join(FIGURE_DIR, "pose_free_weighted_anisotropic_inverse_heatmap.png"), summary_rows)
    plot_baseline_and_penalty(
        os.path.join(FIGURE_DIR, "pose_free_weighted_anisotropic_inverse_baseline_and_penalty.png"),
        summary_rows,
        penalty_rows,
    )
    plot_example_recoveries(os.path.join(FIGURE_DIR, "pose_free_weighted_anisotropic_inverse_examples.png"), rows)

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
        "smallest_geometry_penalty_factor": float(min(item["geometry_mae_penalty_factor"] for item in penalty_rows)),
        "largest_geometry_penalty_factor": float(max(item["geometry_mae_penalty_factor"] for item in penalty_rows)),
        "smallest_weight_penalty_factor": float(min(item["weight_mae_penalty_factor"] for item in penalty_rows)),
        "largest_weight_penalty_factor": float(max(item["weight_mae_penalty_factor"] for item in penalty_rows)),
        "smallest_alpha_penalty_factor": float(min(item["alpha_mae_penalty_factor"] for item in penalty_rows)),
        "largest_alpha_penalty_factor": float(max(item["alpha_mae_penalty_factor"] for item in penalty_rows)),
        "smallest_fit_penalty_factor": float(min(item["anisotropic_fit_rmse_penalty_factor"] for item in penalty_rows)),
        "largest_fit_penalty_factor": float(max(item["anisotropic_fit_rmse_penalty_factor"] for item in penalty_rows)),
    }

    with open(os.path.join(OUTPUT_DIR, "pose_free_weighted_anisotropic_inverse_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows, "penalties_vs_canonical_anisotropic": penalty_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows, "penalties_vs_canonical_anisotropic": penalty_rows}, indent=2))

if __name__ == "__main__":
    main()
