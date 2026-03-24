"""
Theory-hardening experiment: representation independence for weighted
anisotropic inverse Shape Budget.

This experiment asks whether the main inferential story survives a meaningful
swap of the boundary encoding while keeping the forward family fixed.

The two encodings are:

- the existing centroid-normalized radial signature
- a centroid-normalized support-function profile

The main questions are:

1. does the anisotropy-aware inverse still recover the control object under the
   alternative encoding in canonical pose?
2. does it still decisively beat a Euclidean baseline?
3. under pose-free observation, does alpha still degrade much more than
   geometry?
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from run_pose_free_weighted_anisotropic_inverse_experiment import nearest_neighbor_pose_free
from run_pose_free_weighted_inverse_experiment import build_shift_stack
from run_weighted_anisotropic_inverse_experiment import (
    ALPHA_MAX,
    ALPHA_MIN,
    GEOMETRY_BOUNDS,
    sample_anisotropic_parameters,
    sample_euclidean_parameters,
    symmetry_aware_errors,
)
from run_weighted_multisource_experiment import canonical_sources, normalize_weights, weighted_boundary_curve
from run_weighted_multisource_inverse_experiment import (
    OBSERVATION_REGIMES,
    SIGNATURE_ANGLE_COUNT,
    boundary_signature_from_curve,
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
OUTPUT_DIR = os.path.join(BASE_DIR, "representation_independence_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


ANISOTROPIC_BANK_SIZE = 220
EUCLIDEAN_BANK_SIZE = 120
TEST_TRIALS_PER_REGIME = 20
CURVE_SAMPLE_COUNT = 160
SUPPORT_AUDIT_CASES = 30

REPRESENTATIONS = ["radial", "support"]
MODES = ["canonical", "pose_free"]


@dataclass
class TrialRow:
    condition: str
    mode: str
    representation: str
    trial: int
    true_alpha: float
    true_rotation_shift: int
    geometry_mae: float
    weight_mae: float
    alpha_abs_error: float
    fit_rmse: float
    euclidean_fit_rmse: float
    fit_improvement_factor: float


def anisotropic_forward_curve(params: tuple[float, float, float, float, float, float]) -> np.ndarray:
    rho, t, h, w1, w2, alpha = params
    weights = normalize_weights(np.array([w1, w2, 1.0 - w1 - w2], dtype=float))

    points_raw = canonical_sources(rho, t, h, S=1.0)
    points_white = points_raw.copy()
    points_white[:, 1] *= alpha

    _, _, curve_white = weighted_boundary_curve(points_white, weights, 1.0, angle_count=CURVE_SAMPLE_COUNT)
    curve_raw = curve_white.copy()
    curve_raw[:, 1] /= alpha
    return curve_raw


def support_signature_from_curve(curve: np.ndarray, angle_count: int = SIGNATURE_ANGLE_COUNT) -> np.ndarray:
    center = np.mean(curve, axis=0)
    shifted = curve - center
    grid = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    directions = np.column_stack([np.cos(grid), np.sin(grid)])
    support = np.max(shifted @ directions.T, axis=0)
    return support / np.mean(support)


def representation_signature(curve: np.ndarray, representation: str) -> np.ndarray:
    if representation == "radial":
        return boundary_signature_from_curve(curve, angle_count=SIGNATURE_ANGLE_COUNT)
    if representation == "support":
        return support_signature_from_curve(curve, angle_count=SIGNATURE_ANGLE_COUNT)
    raise ValueError(f"Unknown representation: {representation}")


def build_representation_bank(
    sample_size: int,
    rng: np.random.Generator,
    anisotropic: bool,
) -> tuple[list[tuple[float, float, float, float, float, float]], dict[str, np.ndarray]]:
    params_list: list[tuple[float, float, float, float, float, float]] = []
    signatures = {name: [] for name in REPRESENTATIONS}
    sampler = sample_anisotropic_parameters if anisotropic else sample_euclidean_parameters

    while len(params_list) < sample_size:
        params = sampler(rng)
        curve = anisotropic_forward_curve(params)
        params_list.append(params)
        for representation in REPRESENTATIONS:
            signatures[representation].append(representation_signature(curve, representation))

    return params_list, {key: np.array(value) for key, value in signatures.items()}


def sample_mask_and_noise(
    regime: dict[str, float | str | int],
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(length, dtype=bool)
    mode = str(regime["mode"])

    if mode == "full":
        mask[:] = True
    elif mode == "contiguous":
        span = int(float(regime["observed_fraction"]) * length)
        start = int(rng.integers(0, length))
        mask[(np.arange(span) + start) % length] = True
    elif mode == "random":
        count = int(regime["observed_count"])
        mask[rng.choice(length, size=count, replace=False)] = True
    elif mode == "sparse_contiguous":
        span = int(float(regime["arc_fraction"]) * length)
        start = int(rng.integers(0, length))
        pool = (np.arange(span) + start) % length
        count = min(int(regime["observed_count"]), len(pool))
        mask[rng.choice(pool, size=count, replace=False)] = True
    else:
        raise ValueError(f"Unknown observation mode: {mode}")

    noise = np.zeros(length, dtype=float)
    sigma = float(regime["noise_sigma"])
    if sigma > 0.0:
        noise[mask] = rng.normal(scale=sigma, size=int(np.sum(mask)))
    return mask, noise


def apply_observation(clean_signature: np.ndarray, mask: np.ndarray, noise: np.ndarray) -> np.ndarray:
    observed = clean_signature.copy()
    observed[mask] += noise[mask]
    return observed


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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def audit_support_identity(
    bank_signatures: np.ndarray,
    bank_params: list[tuple[float, float, float, float, float, float]],
    rng: np.random.Generator,
) -> dict[str, float]:
    full_clean = next(regime for regime in OBSERVATION_REGIMES if str(regime["name"]) == "full_clean")
    max_canonical_fit = 0.0
    max_pose_fit = 0.0
    exact_canonical = 0
    exact_pose = 0

    shift_stack = build_shift_stack(bank_signatures)
    for _ in range(SUPPORT_AUDIT_CASES):
        idx = int(rng.integers(0, len(bank_signatures)))
        signature = bank_signatures[idx]

        mask, noise = sample_mask_and_noise(full_clean, len(signature), rng)
        observed = apply_observation(signature, mask, noise)
        pred_params, pred_signature = nearest_neighbor_prediction(observed, mask, bank_signatures, bank_params)
        if pred_params == bank_params[idx]:
            exact_canonical += 1
        max_canonical_fit = max(max_canonical_fit, rmse(pred_signature, signature))

        shift = int(rng.integers(0, len(signature)))
        rotated = np.roll(signature, shift)
        pose_observed = apply_observation(rotated, mask, noise)
        pose_params, pose_signature, _ = nearest_neighbor_pose_free(pose_observed, mask, shift_stack, bank_params)
        if pose_params == bank_params[idx]:
            exact_pose += 1
        max_pose_fit = max(max_pose_fit, rmse(pose_signature, rotated))

    return {
        "audit_cases": float(SUPPORT_AUDIT_CASES),
        "canonical_exact_recovery_fraction": float(exact_canonical / SUPPORT_AUDIT_CASES),
        "pose_free_exact_recovery_fraction": float(exact_pose / SUPPORT_AUDIT_CASES),
        "max_canonical_fit_rmse": float(max_canonical_fit),
        "max_pose_free_fit_rmse": float(max_pose_fit),
    }


def aggregate(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for representation in REPRESENTATIONS:
        for mode in MODES:
            for regime in OBSERVATION_REGIMES:
                name = str(regime["name"])
                subset = [
                    row for row in rows if row.representation == representation and row.mode == mode and row.condition == name
                ]
                geometry = np.array([row.geometry_mae for row in subset])
                weight = np.array([row.weight_mae for row in subset])
                alpha = np.array([row.alpha_abs_error for row in subset])
                fit = np.array([row.fit_rmse for row in subset])
                baseline = np.array([row.euclidean_fit_rmse for row in subset])
                improve = np.array([row.fit_improvement_factor for row in subset])
                summary.append(
                    {
                        "representation": representation,
                        "mode": mode,
                        "condition": name,
                        "geometry_mae_mean": float(np.mean(geometry)),
                        "weight_mae_mean": float(np.mean(weight)),
                        "alpha_mae_mean": float(np.mean(alpha)),
                        "fit_rmse_mean": float(np.mean(fit)),
                        "euclidean_baseline_fit_rmse_mean": float(np.mean(baseline)),
                        "fit_improvement_factor_mean": float(np.mean(improve)),
                    }
                )
    return summary


def pose_penalty_rows(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for representation in REPRESENTATIONS:
        for regime in OBSERVATION_REGIMES:
            name = str(regime["name"])
            canonical = next(
                row
                for row in summary_rows
                if str(row["representation"]) == representation and str(row["mode"]) == "canonical" and str(row["condition"]) == name
            )
            pose = next(
                row
                for row in summary_rows
                if str(row["representation"]) == representation and str(row["mode"]) == "pose_free" and str(row["condition"]) == name
            )
            geometry_penalty = float(pose["geometry_mae_mean"]) / max(float(canonical["geometry_mae_mean"]), 1.0e-12)
            weight_penalty = float(pose["weight_mae_mean"]) / max(float(canonical["weight_mae_mean"]), 1.0e-12)
            alpha_penalty = float(pose["alpha_mae_mean"]) / max(float(canonical["alpha_mae_mean"]), 1.0e-12)
            output.append(
                {
                    "representation": representation,
                    "condition": name,
                    "geometry_penalty_factor": geometry_penalty,
                    "weight_penalty_factor": weight_penalty,
                    "alpha_penalty_factor": alpha_penalty,
                    "alpha_over_geometry_selectivity": alpha_penalty / max(geometry_penalty, 1.0e-12),
                }
            )
    return output


def representation_gap_rows(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for mode in MODES:
        for regime in OBSERVATION_REGIMES:
            name = str(regime["name"])
            radial = next(
                row
                for row in summary_rows
                if str(row["representation"]) == "radial" and str(row["mode"]) == mode and str(row["condition"]) == name
            )
            support = next(
                row
                for row in summary_rows
                if str(row["representation"]) == "support" and str(row["mode"]) == mode and str(row["condition"]) == name
            )
            output.append(
                {
                    "mode": mode,
                    "condition": name,
                    "support_over_radial_geometry_mae": float(support["geometry_mae_mean"]) / max(float(radial["geometry_mae_mean"]), 1.0e-12),
                    "support_over_radial_weight_mae": float(support["weight_mae_mean"]) / max(float(radial["weight_mae_mean"]), 1.0e-12),
                    "support_over_radial_alpha_mae": float(support["alpha_mae_mean"]) / max(float(radial["alpha_mae_mean"]), 1.0e-12),
                    "support_over_radial_fit_improvement": float(support["fit_improvement_factor_mean"]) / max(
                        float(radial["fit_improvement_factor_mean"]), 1.0e-12
                    ),
                }
            )
    return output


def plot_alpha_comparison(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(regime["name"]) for regime in OBSERVATION_REGIMES]
    x = np.arange(len(conditions))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.28)

    for ax, mode, title in [
        (axes[0], "canonical", "Canonical pose alpha recovery"),
        (axes[1], "pose_free", "Pose-free alpha recovery"),
    ]:
        radial = np.array(
            [
                float(
                    next(
                        row["alpha_mae_mean"]
                        for row in summary_rows
                        if str(row["representation"]) == "radial"
                        and str(row["mode"]) == mode
                        and str(row["condition"]) == condition
                    )
                )
                for condition in conditions
            ]
        )
        support = np.array(
            [
                float(
                    next(
                        row["alpha_mae_mean"]
                        for row in summary_rows
                        if str(row["representation"]) == "support"
                        and str(row["mode"]) == mode
                        and str(row["condition"]) == condition
                    )
                )
                for condition in conditions
            ]
        )
        ax.bar(x - width / 2.0, radial, width=width, color="#1d3557", label="radial")
        ax.bar(x + width / 2.0, support, width=width, color="#2a9d8f", label="support")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel("mean alpha absolute error")
        ax.set_title(title)
    axes[0].legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Representation Independence A: Alpha Recovery Under Two Encodings",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_fit_improvement_and_selectivity(
    path: str,
    summary_rows: list[dict[str, float | str]],
    penalty_rows: list[dict[str, float | str]],
) -> None:
    conditions = [str(regime["name"]) for regime in OBSERVATION_REGIMES]
    x = np.arange(len(conditions))

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.4), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.28)

    for representation, color in [("radial", "#1d3557"), ("support", "#2a9d8f")]:
        canonical_improve = np.array(
            [
                float(
                    next(
                        row["fit_improvement_factor_mean"]
                        for row in summary_rows
                        if str(row["representation"]) == representation
                        and str(row["mode"]) == "canonical"
                        and str(row["condition"]) == condition
                    )
                )
                for condition in conditions
            ]
        )
        selectivity = np.array(
            [
                float(
                    next(
                        row["alpha_over_geometry_selectivity"]
                        for row in penalty_rows
                        if str(row["representation"]) == representation and str(row["condition"]) == condition
                    )
                )
                for condition in conditions
            ]
        )
        axes[0].plot(x, canonical_improve, color=color, lw=2.4, marker="o", label=representation)
        axes[1].plot(x, selectivity, color=color, lw=2.4, marker="s", label=representation)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=20, ha="right")
    axes[0].set_ylabel("Euclidean / anisotropy-aware fit ratio")
    axes[0].set_title("Canonical anisotropy-aware improvement over Euclidean baseline")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[1].set_ylabel("alpha pose penalty / geometry pose penalty")
    axes[1].set_title("Selective pose penalty under each representation")
    axes[1].axhline(1.0, color="#444444", linestyle="--", lw=1.3)

    fig.suptitle(
        "Representation Independence B: Baseline Advantage And Pose-Penalty Selectivity",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)

    anisotropic_params, anisotropic_signatures = build_representation_bank(ANISOTROPIC_BANK_SIZE, rng, anisotropic=True)
    euclidean_params, euclidean_signatures = build_representation_bank(EUCLIDEAN_BANK_SIZE, rng, anisotropic=False)

    shift_stacks = {representation: build_shift_stack(anisotropic_signatures[representation]) for representation in REPRESENTATIONS}
    euclidean_shift_stacks = {
        representation: build_shift_stack(euclidean_signatures[representation]) for representation in REPRESENTATIONS
    }

    audits = {
        "support_identity": audit_support_identity(
            anisotropic_signatures["support"],
            anisotropic_params,
            np.random.default_rng(20260324),
        )
    }

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            curve = anisotropic_forward_curve(true_params)
            clean_signatures = {representation: representation_signature(curve, representation) for representation in REPRESENTATIONS}

            canonical_mask, canonical_noise = sample_mask_and_noise(regime, SIGNATURE_ANGLE_COUNT, rng)
            pose_shift = int(rng.integers(0, SIGNATURE_ANGLE_COUNT))
            pose_mask, pose_noise = sample_mask_and_noise(regime, SIGNATURE_ANGLE_COUNT, rng)

            for representation in REPRESENTATIONS:
                clean_signature = clean_signatures[representation]

                observed = apply_observation(clean_signature, canonical_mask, canonical_noise)
                pred_params, pred_signature = nearest_neighbor_prediction(
                    observed,
                    canonical_mask,
                    anisotropic_signatures[representation],
                    anisotropic_params,
                )
                euclidean_params_pred, euclidean_signature = nearest_neighbor_prediction(
                    observed,
                    canonical_mask,
                    euclidean_signatures[representation],
                    euclidean_params,
                )
                geometry_mae, weight_mae, alpha_error = symmetry_aware_errors(true_params, pred_params)
                fit_rmse = rmse(pred_signature, clean_signature)
                euclidean_fit = rmse(euclidean_signature, clean_signature)
                rows.append(
                    TrialRow(
                        condition=str(regime["name"]),
                        mode="canonical",
                        representation=representation,
                        trial=trial,
                        true_alpha=float(true_params[5]),
                        true_rotation_shift=0,
                        geometry_mae=float(geometry_mae),
                        weight_mae=float(weight_mae),
                        alpha_abs_error=float(alpha_error),
                        fit_rmse=float(fit_rmse),
                        euclidean_fit_rmse=float(euclidean_fit),
                        fit_improvement_factor=float(euclidean_fit / max(fit_rmse, 1.0e-12)),
                    )
                )

                rotated_signature = np.roll(clean_signature, pose_shift)
                pose_observed = apply_observation(rotated_signature, pose_mask, pose_noise)
                pose_pred_params, pose_pred_signature, _ = nearest_neighbor_pose_free(
                    pose_observed,
                    pose_mask,
                    shift_stacks[representation],
                    anisotropic_params,
                )
                euclidean_pose_pred_params, euclidean_pose_signature, _ = nearest_neighbor_pose_free(
                    pose_observed,
                    pose_mask,
                    euclidean_shift_stacks[representation],
                    euclidean_params,
                )
                pose_geometry_mae, pose_weight_mae, pose_alpha_error = symmetry_aware_errors(true_params, pose_pred_params)
                pose_fit_rmse = rmse(pose_pred_signature, rotated_signature)
                euclidean_pose_fit = rmse(euclidean_pose_signature, rotated_signature)
                rows.append(
                    TrialRow(
                        condition=str(regime["name"]),
                        mode="pose_free",
                        representation=representation,
                        trial=trial,
                        true_alpha=float(true_params[5]),
                        true_rotation_shift=pose_shift,
                        geometry_mae=float(pose_geometry_mae),
                        weight_mae=float(pose_weight_mae),
                        alpha_abs_error=float(pose_alpha_error),
                        fit_rmse=float(pose_fit_rmse),
                        euclidean_fit_rmse=float(euclidean_pose_fit),
                        fit_improvement_factor=float(euclidean_pose_fit / max(pose_fit_rmse, 1.0e-12)),
                    )
                )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = aggregate(rows)
    penalty_rows = pose_penalty_rows(summary_rows)
    gap_rows = representation_gap_rows(summary_rows)

    write_csv(os.path.join(OUTPUT_DIR, "representation_independence_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "representation_independence_summary.csv"), summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "representation_independence_pose_penalties.csv"), penalty_rows)
    write_csv(os.path.join(OUTPUT_DIR, "representation_independence_representation_gaps.csv"), gap_rows)

    plot_alpha_comparison(os.path.join(FIGURE_DIR, "representation_independence_alpha.png"), summary_rows)
    plot_fit_improvement_and_selectivity(
        os.path.join(FIGURE_DIR, "representation_independence_selectivity.png"),
        summary_rows,
        penalty_rows,
    )

    summary = {
        "anisotropic_bank_size": float(ANISOTROPIC_BANK_SIZE),
        "euclidean_bank_size": float(EUCLIDEAN_BANK_SIZE),
        "test_trials_per_regime": float(TEST_TRIALS_PER_REGIME),
        "audits": audits,
        "best_support_over_radial_alpha_ratio": float(
            min(float(row["support_over_radial_alpha_mae"]) for row in gap_rows)
        ),
        "worst_support_over_radial_alpha_ratio": float(
            max(float(row["support_over_radial_alpha_mae"]) for row in gap_rows)
        ),
        "best_support_selective_pose_penalty": float(
            min(float(row["alpha_over_geometry_selectivity"]) for row in penalty_rows if str(row["representation"]) == "support")
        ),
        "worst_support_selective_pose_penalty": float(
            max(float(row["alpha_over_geometry_selectivity"]) for row in penalty_rows if str(row["representation"]) == "support")
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "representation_independence_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "by_condition_mode_representation": summary_rows,
                "pose_penalties": penalty_rows,
                "representation_gaps": gap_rows,
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "summary": summary,
                "by_condition_mode_representation": summary_rows,
                "pose_penalties": penalty_rows,
                "representation_gaps": gap_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
