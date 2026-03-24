"""
Post-roadmap extension: pose-free weighted inverse experiment for Shape Budget.

This experiment extends the weighted three-source inverse from canonical pose
to the setting where rotation is also unknown.

The recovery target remains the normalized control object:

- normalized source-triangle geometry relative to budget
- normalized weight vector in the simplex

Translation and scale are still removed from the observed boundary via the
centroid-centered mean-radius-normalized signature used in the earlier inverse
test. Rotation is now treated as a nuisance variable and is jointly handled by
cyclic-shift matching against a reference bank.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from run_weighted_multisource_inverse_experiment import (
    EQUAL_WEIGHT_BANK_SIZE,
    GEOMETRY_BOUNDS,
    OBSERVATION_REGIMES,
    REFERENCE_BANK_SIZE,
    SIGNATURE_ANGLE_COUNT,
    TEST_TRIALS_PER_REGIME,
    TrialRow,
    aggregate_trials,
    build_reference_bank,
    forward_signature,
    observe_signature,
    symmetry_aware_errors,
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
OUTPUT_DIR = os.path.join(BASE_DIR, "pose_free_weighted_inverse_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


@dataclass
class PoseFreeTrialRow:
    condition: str
    trial: int
    true_rho: float
    true_t: float
    true_h: float
    true_w1: float
    true_w2: float
    true_w3: float
    true_rotation_shift: int
    pred_rho: float
    pred_t: float
    pred_h: float
    pred_w1: float
    pred_w2: float
    pred_w3: float
    pred_rotation_shift: int
    equal_pred_rho: float
    equal_pred_t: float
    equal_pred_h: float
    equal_pred_rotation_shift: int
    geometry_mae: float
    weight_mae: float
    weighted_fit_rmse: float
    equal_weight_baseline_fit_rmse: float
    weighted_fit_improvement_factor: float


def sample_weighted_parameters(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    rho = float(rng.uniform(GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(rng.uniform(GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(rng.uniform(GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    weights = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
    return rho, t, h, float(weights[0]), float(weights[1])


def build_shift_stack(bank_signatures: np.ndarray) -> np.ndarray:
    angle_count = bank_signatures.shape[1]
    return np.stack([np.roll(bank_signatures, shift, axis=1) for shift in range(angle_count)], axis=1)


def observe_pose_free_signature(
    clean_signature: np.ndarray,
    regime: dict[str, float | str | int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    shift = int(rng.integers(0, len(clean_signature)))
    rotated = np.roll(clean_signature, shift)
    observed, mask = observe_signature(rotated, regime, rng)
    return rotated, observed, mask, shift


def nearest_neighbor_pose_free(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
    bank_params: list[tuple[float, float, float, float, float]],
) -> tuple[tuple[float, float, float, float, float], np.ndarray, int]:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    mse = np.mean(residual * residual, axis=2)
    best_flat = int(np.argmin(mse))
    bank_idx, shift_idx = np.unravel_index(best_flat, mse.shape)
    return bank_params[bank_idx], shifted_bank[bank_idx, shift_idx], int(shift_idx)


def load_canonical_inverse_summary() -> dict[str, dict[str, float]]:
    path = os.path.join(BASE_DIR, "weighted_multisource_inverse_outputs", "weighted_multisource_inverse_summary.json")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {item["condition"]: item for item in data["by_condition"]}


def compare_to_canonical(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    canonical = load_canonical_inverse_summary()
    rows: list[dict[str, float | str]] = []
    for row in summary_rows:
        name = str(row["condition"])
        base = canonical[name]
        rows.append(
            {
                "condition": name,
                "geometry_mae_penalty_factor": float(row["geometry_mae_mean"]) / float(base["geometry_mae_mean"]),
                "weight_mae_penalty_factor": float(row["weight_mae_mean"]) / float(base["weight_mae_mean"]),
                "weighted_fit_rmse_penalty_factor": float(row["weighted_fit_rmse_mean"]) / float(base["weighted_fit_rmse_mean"]),
                "fit_improvement_ratio_vs_canonical": float(row["fit_improvement_factor_mean"]) / float(base["fit_improvement_factor_mean"]),
            }
        )
    return rows


def plot_error_heatmap(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    geometry = np.array([[float(item["geometry_mae_mean"]) for item in summary_rows]])
    weights = np.array([[float(item["weight_mae_mean"]) for item in summary_rows]])

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.19, right=0.98, hspace=0.35)

    sns.heatmap(
        geometry,
        ax=axes[0],
        cmap="viridis",
        annot=True,
        fmt=".3f",
        xticklabels=conditions,
        yticklabels=["geometry MAE"],
        cbar_kws={"label": "mean absolute error"},
    )
    axes[0].set_title("Pose-free boundary-only recovery of normalized geometry")

    sns.heatmap(
        weights,
        ax=axes[1],
        cmap="magma",
        annot=True,
        fmt=".3f",
        xticklabels=conditions,
        yticklabels=["weight MAE"],
        cbar_kws={"label": "mean absolute error"},
    )
    axes[1].set_title("Pose-free boundary-only recovery of normalized weights")
    axes[1].set_xlabel("observation regime")

    fig.suptitle("Pose-Free Weighted Inverse A: Recovery Error Across Regimes", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_and_penalty(
    path: str,
    summary_rows: list[dict[str, float | str]],
    penalty_rows: list[dict[str, float | str]],
) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    weighted_fit = np.array([float(item["weighted_fit_rmse_mean"]) for item in summary_rows])
    equal_fit = np.array([float(item["equal_weight_baseline_fit_rmse_mean"]) for item in summary_rows])
    improve = np.array([float(item["fit_improvement_factor_mean"]) for item in summary_rows])
    fit_penalty = np.array([float(item["weighted_fit_rmse_penalty_factor"]) for item in penalty_rows])

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.28)

    x = np.arange(len(conditions))
    width = 0.36
    axes[0].bar(x - width / 2.0, weighted_fit, width=width, color="#2a9d8f", label="weighted bank")
    axes[0].bar(x + width / 2.0, equal_fit, width=width, color="#e76f51", label="equal-weight baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=20, ha="right")
    axes[0].set_ylabel("mean rotated-signature RMSE")
    axes[0].set_title("Weighted bank still beats the equal-weight baseline")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(x, improve, color="#1d3557", lw=2.4, marker="o")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[1].set_ylabel("baseline / weighted fit ratio")
    axes[1].set_title("Average fit-improvement factor")

    axes[2].plot(x, fit_penalty, color="#6a4c93", lw=2.4, marker="s")
    axes[2].axhline(1.0, color="#444444", linestyle="--", lw=1.4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(conditions, rotation=20, ha="right")
    axes[2].set_ylabel("pose-free / canonical fit RMSE")
    axes[2].set_title("Rotation penalty relative to canonical-pose inverse")

    fig.suptitle("Pose-Free Weighted Inverse B: Baseline And Rotation Penalty", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_example_recoveries(path: str, rows: list[PoseFreeTrialRow]) -> None:
    chosen_conditions = ["full_clean", "partial_arc_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(len(chosen_conditions), 1, figsize=(10.2, 9.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.40)

    for ax, name in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == name]
        exemplar = min(subset, key=lambda row: row.weighted_fit_rmse)
        true_sig = np.roll(
            forward_signature((exemplar.true_rho, exemplar.true_t, exemplar.true_h, exemplar.true_w1, exemplar.true_w2)),
            exemplar.true_rotation_shift,
        )
        pred_sig = np.roll(
            forward_signature((exemplar.pred_rho, exemplar.pred_t, exemplar.pred_h, exemplar.pred_w1, exemplar.pred_w2)),
            exemplar.pred_rotation_shift,
        )
        equal_sig = np.roll(
            forward_signature((exemplar.equal_pred_rho, exemplar.equal_pred_t, exemplar.equal_pred_h, 1.0 / 3.0, 1.0 / 3.0)),
            exemplar.equal_pred_rotation_shift,
        )
        angle_grid = np.linspace(0.0, 2.0 * math.pi, len(true_sig), endpoint=False)

        ax.plot(angle_grid, true_sig, color="#222222", lw=2.4, label="true rotated signature")
        ax.plot(angle_grid, pred_sig, color="#2a9d8f", lw=2.0, label="weighted-bank recovery")
        ax.plot(angle_grid, equal_sig, color="#e76f51", lw=1.8, linestyle="--", label="equal-weight baseline")
        ax.set_title(
            f"{name}: geometry MAE = {exemplar.geometry_mae:.3f}, weight MAE = {exemplar.weight_mae:.3f}, shift = {exemplar.true_rotation_shift}"
        )
        ax.set_xlabel("angle")
        ax.set_ylabel("normalized radius")
        if ax is axes[0]:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Pose-Free Weighted Inverse C: Representative Rotated Signature Recoveries", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)
    weighted_params, weighted_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, weighted=True)
    equal_params, equal_signatures = build_reference_bank(EQUAL_WEIGHT_BANK_SIZE, rng, weighted=False)
    weighted_shift_stack = build_shift_stack(weighted_signatures)
    equal_shift_stack = build_shift_stack(equal_signatures)

    rows: list[PoseFreeTrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_weighted_parameters(rng)
            clean_signature = forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            pred_params, pred_signature, pred_shift = nearest_neighbor_pose_free(
                observed_signature,
                mask,
                weighted_shift_stack,
                weighted_params,
            )
            equal_pred_params, equal_signature, equal_pred_shift = nearest_neighbor_pose_free(
                observed_signature,
                mask,
                equal_shift_stack,
                equal_params,
            )

            geometry_mae, weight_mae = symmetry_aware_errors(true_params, pred_params)
            weighted_fit_rmse = float(np.sqrt(np.mean((pred_signature - rotated_signature) ** 2)))
            equal_fit_rmse = float(np.sqrt(np.mean((equal_signature - rotated_signature) ** 2)))

            rows.append(
                PoseFreeTrialRow(
                    condition=str(regime["name"]),
                    trial=trial,
                    true_rho=float(true_params[0]),
                    true_t=float(true_params[1]),
                    true_h=float(true_params[2]),
                    true_w1=float(true_params[3]),
                    true_w2=float(true_params[4]),
                    true_w3=float(1.0 - true_params[3] - true_params[4]),
                    true_rotation_shift=int(true_shift),
                    pred_rho=float(pred_params[0]),
                    pred_t=float(pred_params[1]),
                    pred_h=float(pred_params[2]),
                    pred_w1=float(pred_params[3]),
                    pred_w2=float(pred_params[4]),
                    pred_w3=float(1.0 - pred_params[3] - pred_params[4]),
                    pred_rotation_shift=int(pred_shift),
                    equal_pred_rho=float(equal_pred_params[0]),
                    equal_pred_t=float(equal_pred_params[1]),
                    equal_pred_h=float(equal_pred_params[2]),
                    equal_pred_rotation_shift=int(equal_pred_shift),
                    geometry_mae=geometry_mae,
                    weight_mae=weight_mae,
                    weighted_fit_rmse=weighted_fit_rmse,
                    equal_weight_baseline_fit_rmse=equal_fit_rmse,
                    weighted_fit_improvement_factor=float(equal_fit_rmse / max(weighted_fit_rmse, 1.0e-12)),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = aggregate_trials(
        [
            TrialRow(
                condition=row.condition,
                trial=row.trial,
                true_rho=row.true_rho,
                true_t=row.true_t,
                true_h=row.true_h,
                true_w1=row.true_w1,
                true_w2=row.true_w2,
                true_w3=row.true_w3,
                pred_rho=row.pred_rho,
                pred_t=row.pred_t,
                pred_h=row.pred_h,
                pred_w1=row.pred_w1,
                pred_w2=row.pred_w2,
                pred_w3=row.pred_w3,
                equal_pred_rho=row.equal_pred_rho,
                equal_pred_t=row.equal_pred_t,
                equal_pred_h=row.equal_pred_h,
                geometry_mae=row.geometry_mae,
                weight_mae=row.weight_mae,
                weighted_fit_rmse=row.weighted_fit_rmse,
                equal_weight_baseline_fit_rmse=row.equal_weight_baseline_fit_rmse,
                weighted_fit_improvement_factor=row.weighted_fit_improvement_factor,
            )
            for row in rows
        ]
    )
    penalty_rows = compare_to_canonical(summary_rows)

    write_csv(os.path.join(OUTPUT_DIR, "pose_free_weighted_inverse_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "pose_free_weighted_inverse_summary.csv"), summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "pose_free_weighted_inverse_penalties.csv"), penalty_rows)

    plot_error_heatmap(os.path.join(FIGURE_DIR, "pose_free_weighted_inverse_heatmap.png"), summary_rows)
    plot_baseline_and_penalty(
        os.path.join(FIGURE_DIR, "pose_free_weighted_inverse_baseline_and_penalty.png"),
        summary_rows,
        penalty_rows,
    )
    plot_example_recoveries(os.path.join(FIGURE_DIR, "pose_free_weighted_inverse_examples.png"), rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "equal_weight_bank_size": EQUAL_WEIGHT_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "best_geometry_mae_mean": float(min(item["geometry_mae_mean"] for item in summary_rows)),
        "worst_geometry_mae_mean": float(max(item["geometry_mae_mean"] for item in summary_rows)),
        "best_weight_mae_mean": float(min(item["weight_mae_mean"] for item in summary_rows)),
        "worst_weight_mae_mean": float(max(item["weight_mae_mean"] for item in summary_rows)),
        "best_weighted_fit_rmse_mean": float(min(item["weighted_fit_rmse_mean"] for item in summary_rows)),
        "worst_weighted_fit_rmse_mean": float(max(item["weighted_fit_rmse_mean"] for item in summary_rows)),
        "smallest_fit_improvement_factor_mean": float(min(item["fit_improvement_factor_mean"] for item in summary_rows)),
        "largest_fit_improvement_factor_mean": float(max(item["fit_improvement_factor_mean"] for item in summary_rows)),
        "smallest_geometry_penalty_factor": float(min(item["geometry_mae_penalty_factor"] for item in penalty_rows)),
        "largest_geometry_penalty_factor": float(max(item["geometry_mae_penalty_factor"] for item in penalty_rows)),
        "smallest_weight_penalty_factor": float(min(item["weight_mae_penalty_factor"] for item in penalty_rows)),
        "largest_weight_penalty_factor": float(max(item["weight_mae_penalty_factor"] for item in penalty_rows)),
        "smallest_fit_penalty_factor": float(min(item["weighted_fit_rmse_penalty_factor"] for item in penalty_rows)),
        "largest_fit_penalty_factor": float(max(item["weighted_fit_rmse_penalty_factor"] for item in penalty_rows)),
    }

    with open(os.path.join(OUTPUT_DIR, "pose_free_weighted_inverse_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows, "penalties_vs_canonical": penalty_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows, "penalties_vs_canonical": penalty_rows}, indent=2))


if __name__ == "__main__":
    main()
