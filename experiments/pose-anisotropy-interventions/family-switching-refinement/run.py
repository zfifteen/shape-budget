"""
Post-roadmap extension: family-switching refinement experiment.

This experiment targets the remaining sparse moderate failure slice after the
candidate-conditioned local shift-alpha search.

The sharper question is:

1. keep the same top marginalized candidate seeds as the prior experiment
2. compare an alpha-only local search inside each fixed seed family against a
   true family-switching local refinement that lets geometry and alpha move
3. test whether that stronger intervention specifically repairs the
   sparse-full moderate mid-skew failure slice
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

from run_candidate_conditioned_alignment_experiment import (
    GEOMETRY_SKEW_BIN_LABELS,
    candidate_conditioned_search,
    sample_conditioned_parameters,
    top_k_indices,
)
from run_oracle_alignment_ceiling_experiment import oracle_align_observation
from run_orientation_locking_experiment import nearest_neighbor_aligned, rmse
from run_shift_marginalized_pose_experiment import (
    marginalized_candidate_scores,
    shift_error_matrix,
    softmin_temperature,
)
from run_pose_free_weighted_inverse_experiment import build_shift_stack, observe_pose_free_signature
from run_weighted_anisotropic_inverse_experiment import (
    ALPHA_MAX,
    ALPHA_MIN,
    GEOMETRY_BOUNDS,
    REFERENCE_BANK_SIZE,
    anisotropic_forward_signature,
    build_reference_bank,
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


FOCUS_CONDITIONS = ["sparse_full_noisy", "sparse_partial_high_noise"]
FOCUS_ALPHA_BIN = "moderate"
TOP_K_SEEDS = 3
TRIALS_PER_CELL = 4
GRID_POINTS = 3
REFINEMENT_ROUNDS = 2
SEED_AUDIT_CASES = 30
NEARBY_AUDIT_CASES = 30

INITIAL_RHO_RADIUS = 0.024
INITIAL_T_RADIUS = 0.14
INITIAL_H_RADIUS = 0.14
INITIAL_ALPHA_RADIUS = 0.12


@dataclass
class TrialRow:
    condition: str
    geometry_skew_bin: str
    trial_in_cell: int
    true_rho: float
    true_t: float
    true_h: float
    true_w1: float
    true_w2: float
    true_w3: float
    true_alpha: float
    true_rotation_shift: int
    marginalized_geometry_mae: float
    marginalized_weight_mae: float
    marginalized_alpha_error: float
    marginalized_fit_rmse: float
    conditioned_geometry_mae: float
    conditioned_weight_mae: float
    conditioned_alpha_error: float
    conditioned_fit_rmse: float
    conditioned_seed_rank: int
    family_geometry_mae: float
    family_weight_mae: float
    family_alpha_error: float
    family_fit_rmse: float
    family_winner_seed_rank: int
    family_switched_seed: int
    family_score_improvement_over_seed: float
    family_delta_rho: float
    family_delta_t: float
    family_delta_h: float
    family_delta_alpha: float
    oracle_geometry_mae: float
    oracle_weight_mae: float
    oracle_alpha_error: float
    oracle_fit_rmse: float


def unique_centered_grid(
    center: float,
    radius: float,
    lower: float,
    upper: float,
    count: int,
    extra_values: list[float] | None = None,
) -> np.ndarray:
    values = np.linspace(max(lower, center - radius), min(upper, center + radius), count)
    if extra_values:
        values = np.concatenate([values, np.array(extra_values, dtype=float)])
    values = np.clip(values, lower, upper)
    return np.unique(values)


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


def family_switching_refine(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    seed_params: tuple[float, float, float, float, float, float],
    temperature: float,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, int, float]:
    rho, t, h, w1, w2, alpha = seed_params
    current = {
        "rho": float(rho),
        "t": float(t),
        "h": float(h),
        "alpha": float(alpha),
    }

    best_params = seed_params
    best_score, best_signature, best_shift = evaluate_params(observed_signature, mask, best_params, temperature)

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
            grid = unique_centered_grid(
                current[name],
                radius,
                lower,
                upper,
                GRID_POINTS,
                extra_values=[current[name]],
            )
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

    return best_params, best_signature, best_shift, float(best_score)


def oracle_prediction(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    true_shift: int,
    bank_signatures: np.ndarray,
    bank_params: list[tuple[float, float, float, float, float, float]],
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray]:
    oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
    return nearest_neighbor_aligned(oracle_observed, oracle_mask, bank_signatures, bank_params)


def summarize_cells(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            subset = [
                row
                for row in rows
                if row.condition == condition and row.geometry_skew_bin == skew_bin
            ]
            if not subset:
                continue

            def mean(attr: str) -> float:
                return float(np.mean([getattr(row, attr) for row in subset]))

            marginalized_alpha = mean("marginalized_alpha_error")
            conditioned_alpha = mean("conditioned_alpha_error")
            family_alpha = mean("family_alpha_error")
            oracle_alpha = mean("oracle_alpha_error")
            oracle_gap = marginalized_alpha - oracle_alpha
            conditioned_fraction = (
                float((marginalized_alpha - conditioned_alpha) / oracle_gap)
                if oracle_gap > 1.0e-6
                else float("nan")
            )
            family_fraction = (
                float((marginalized_alpha - family_alpha) / oracle_gap)
                if oracle_gap > 1.0e-6
                else float("nan")
            )

            summary.append(
                {
                    "condition": condition,
                    "alpha_strength_bin": FOCUS_ALPHA_BIN,
                    "geometry_skew_bin": skew_bin,
                    "count": len(subset),
                    "marginalized_alpha_error_mean": marginalized_alpha,
                    "conditioned_alpha_error_mean": conditioned_alpha,
                    "family_alpha_error_mean": family_alpha,
                    "oracle_alpha_error_mean": oracle_alpha,
                    "conditioned_fraction_of_marginalized_oracle_gap_mean": conditioned_fraction,
                    "family_fraction_of_marginalized_oracle_gap_mean": family_fraction,
                    "family_minus_conditioned_oracle_gap_capture": float(family_fraction - conditioned_fraction)
                    if np.isfinite(conditioned_fraction) and np.isfinite(family_fraction)
                    else float("nan"),
                    "family_vs_conditioned_alpha_ratio": float(
                        conditioned_alpha / max(family_alpha, 1.0e-12)
                    ),
                    "marginalized_geometry_mae_mean": mean("marginalized_geometry_mae"),
                    "conditioned_geometry_mae_mean": mean("conditioned_geometry_mae"),
                    "family_geometry_mae_mean": mean("family_geometry_mae"),
                    "oracle_geometry_mae_mean": mean("oracle_geometry_mae"),
                    "family_winner_seed_rank_mean": mean("family_winner_seed_rank"),
                    "family_non_top1_seed_fraction": mean("family_switched_seed"),
                    "family_score_improvement_over_seed_mean": mean("family_score_improvement_over_seed"),
                    "family_delta_rho_mean": mean("family_delta_rho"),
                    "family_delta_t_mean": mean("family_delta_t"),
                    "family_delta_h_mean": mean("family_delta_h"),
                    "family_delta_alpha_mean": mean("family_delta_alpha"),
                }
            )
    return summary


def build_focus_summary(cell_rows: list[dict[str, float | str]]) -> dict[str, float]:
    sparse_full_rows = [row for row in cell_rows if str(row["condition"]) == "sparse_full_noisy"]
    sparse_partial_rows = [row for row in cell_rows if str(row["condition"]) == "sparse_partial_high_noise"]
    target_mid = [
        row
        for row in cell_rows
        if str(row["condition"]) == "sparse_full_noisy"
        and str(row["geometry_skew_bin"]) == "mid_skew"
    ]

    def mean(subset: list[dict[str, float | str]], metric: str) -> float:
        vals = [float(row[metric]) for row in subset if np.isfinite(float(row[metric]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "sparse_full_family_minus_conditioned_gain_mean": mean(
            sparse_full_rows,
            "family_minus_conditioned_oracle_gap_capture",
        ),
        "sparse_partial_family_minus_conditioned_gain_mean": mean(
            sparse_partial_rows,
            "family_minus_conditioned_oracle_gap_capture",
        ),
        "sparse_full_mid_skew_family_minus_conditioned_gain_mean": mean(
            target_mid,
            "family_minus_conditioned_oracle_gap_capture",
        ),
        "sparse_full_mid_skew_family_vs_conditioned_alpha_ratio_mean": mean(
            target_mid,
            "family_vs_conditioned_alpha_ratio",
        ),
        "largest_family_vs_conditioned_alpha_ratio": float(
            max(float(row["family_vs_conditioned_alpha_ratio"]) for row in cell_rows)
        ),
        "smallest_family_vs_conditioned_alpha_ratio": float(
            min(float(row["family_vs_conditioned_alpha_ratio"]) for row in cell_rows)
        ),
        "largest_family_non_top1_seed_fraction": float(
            max(float(row["family_non_top1_seed_fraction"]) for row in cell_rows)
        ),
    }


def plot_focus_heatmaps(path: str, cell_rows: list[dict[str, float | str]]) -> None:
    metrics = [
        (
            "family_minus_conditioned_oracle_gap_capture",
            "extra oracle-gap capture",
            "coolwarm",
            -1.0,
            1.0,
        ),
        (
            "family_vs_conditioned_alpha_ratio",
            "conditioned / family alpha error",
            "viridis",
            0.5,
            2.0,
        ),
        (
            "family_non_top1_seed_fraction",
            "fraction of non-top1 winners",
            "magma",
            0.0,
            1.0,
        ),
    ]
    fig, axes = plt.subplots(len(metrics), len(FOCUS_CONDITIONS), figsize=(12.8, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.98, wspace=0.28, hspace=0.36)

    for col_idx, condition in enumerate(FOCUS_CONDITIONS):
        for row_idx, (metric, label, cmap, vmin, vmax) in enumerate(metrics):
            matrix = np.full((1, len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
            for cell in cell_rows:
                if str(cell["condition"]) != condition:
                    continue
                j = GEOMETRY_SKEW_BIN_LABELS.index(str(cell["geometry_skew_bin"]))
                matrix[0, j] = float(cell[metric])
            sns.heatmap(
                matrix,
                ax=axes[row_idx, col_idx],
                cmap=cmap,
                annot=True,
                fmt=".2f",
                xticklabels=GEOMETRY_SKEW_BIN_LABELS,
                yticklabels=[FOCUS_ALPHA_BIN],
                cbar=(col_idx == len(FOCUS_CONDITIONS) - 1),
                cbar_kws={"label": label} if col_idx == len(FOCUS_CONDITIONS) - 1 else None,
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_idx, col_idx].set_title(f"{condition}\n{label}")
            axes[row_idx, col_idx].set_xlabel("geometry skew |t| bin")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("anisotropy strength")
            else:
                axes[row_idx, col_idx].set_ylabel("")

    fig.suptitle(
        "Family-Switching Refinement A: Where Geometry+Alpha Switching Helps",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_method_bars(path: str, cell_rows: list[dict[str, float | str]]) -> None:
    labels = [f'{row["condition"]}\n{row["geometry_skew_bin"]}' for row in cell_rows]
    x = np.arange(len(labels))
    width = 0.18

    marginalized = np.array([float(row["marginalized_alpha_error_mean"]) for row in cell_rows])
    conditioned = np.array([float(row["conditioned_alpha_error_mean"]) for row in cell_rows])
    family = np.array([float(row["family_alpha_error_mean"]) for row in cell_rows])
    oracle = np.array([float(row["oracle_alpha_error_mean"]) for row in cell_rows])

    fig, ax = plt.subplots(figsize=(14.4, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.24, left=0.08, right=0.98)

    ax.bar(x - 1.5 * width, marginalized, width=width, color="#1d3557", label="shift-marginalized")
    ax.bar(x - 0.5 * width, conditioned, width=width, color="#2a9d8f", label="alpha-only family search")
    ax.bar(x + 0.5 * width, family, width=width, color="#f4a261", label="geometry+alpha family switch")
    ax.bar(x + 1.5 * width, oracle, width=width, color="#6a4c93", label="oracle")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Targeted moderate-band alpha recovery by cell")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    fig.suptitle(
        "Family-Switching Refinement B: Moderate Sparse Cells",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def audit_seed_non_degradation(
    rng: np.random.Generator,
    bank_params: list[tuple[float, float, float, float, float, float]],
) -> dict[str, float]:
    max_refined_minus_seed_score = 0.0
    regimes = [regime for regime in OBSERVATION_REGIMES if str(regime["name"]) in FOCUS_CONDITIONS]
    for _ in range(SEED_AUDIT_CASES):
        seed = bank_params[int(rng.integers(0, len(bank_params)))]
        clean_signature = anisotropic_forward_signature(seed)
        regime = regimes[int(rng.integers(0, len(regimes)))]
        _, observed_signature, mask, _ = observe_pose_free_signature(clean_signature, regime, rng)
        temperature = softmin_temperature(regime)
        seed_score, _, _ = evaluate_params(observed_signature, mask, seed, temperature)
        _, _, _, refined_score = family_switching_refine(observed_signature, mask, seed, temperature)
        max_refined_minus_seed_score = max(max_refined_minus_seed_score, refined_score - seed_score)
    return {
        "audit_cases": float(SEED_AUDIT_CASES),
        "max_refined_minus_seed_score": float(max_refined_minus_seed_score),
    }


def audit_nearby_clean_recovery(
    rng: np.random.Generator,
) -> dict[str, float]:
    full_clean = next(regime for regime in OBSERVATION_REGIMES if str(regime["name"]) == "full_clean")
    max_alpha_error = 0.0
    max_geometry_mae = 0.0
    exact_signature_fraction = 0

    for _ in range(NEARBY_AUDIT_CASES):
        skew_bin = GEOMETRY_SKEW_BIN_LABELS[int(rng.integers(0, len(GEOMETRY_SKEW_BIN_LABELS)))]
        true_params = sample_conditioned_parameters(rng, FOCUS_ALPHA_BIN, skew_bin)
        clean_signature = anisotropic_forward_signature(true_params)
        rotated_signature, observed_signature, mask, _ = observe_pose_free_signature(clean_signature, full_clean, rng)

        seed_params = (
            float(np.clip(true_params[0] + rng.uniform(-0.012, 0.012), GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"])),
            float(np.clip(true_params[1] + rng.uniform(-0.06, 0.06), GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"])),
            float(np.clip(true_params[2] + rng.uniform(-0.06, 0.06), GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"])),
            float(true_params[3]),
            float(true_params[4]),
            float(np.clip(true_params[5] + rng.uniform(-0.06, 0.06), ALPHA_MIN, ALPHA_MAX)),
        )
        refined_params, refined_signature, _, _ = family_switching_refine(
            observed_signature,
            mask,
            seed_params,
            softmin_temperature(full_clean),
        )
        geometry_mae, _, alpha_error = symmetry_aware_errors(true_params, refined_params)
        max_geometry_mae = max(max_geometry_mae, geometry_mae)
        max_alpha_error = max(max_alpha_error, alpha_error)
        if rmse(refined_signature, rotated_signature) <= 1.0e-12:
            exact_signature_fraction += 1

    return {
        "audit_cases": float(NEARBY_AUDIT_CASES),
        "max_geometry_mae_after_nearby_refinement": float(max_geometry_mae),
        "max_alpha_error_after_nearby_refinement": float(max_alpha_error),
        "exact_rotated_signature_recovery_fraction": float(exact_signature_fraction / NEARBY_AUDIT_CASES),
    }


def main() -> None:
    rng = np.random.default_rng(20260324)

    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}

    audits = {
        "seed_non_degradation": audit_seed_non_degradation(np.random.default_rng(20260324), bank_params),
        "nearby_clean_recovery": audit_nearby_clean_recovery(np.random.default_rng(20260325)),
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

                marginalized_scores, marginalized_best_shifts = marginalized_candidate_scores(
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
                marginalized_fit_rmse = rmse(marginalized_signature, rotated_signature)

                top_seed_indices = top_k_indices(marginalized_scores, TOP_K_SEEDS)

                conditioned_best_params = None
                conditioned_best_signature = None
                conditioned_best_score = float("inf")
                conditioned_seed_rank = 1
                for seed_rank, bank_idx in enumerate(top_seed_indices, start=1):
                    refined_params, refined_signature, _, refined_score = candidate_conditioned_search(
                        observed_signature,
                        mask,
                        bank_params[bank_idx],
                        temperature,
                    )
                    if refined_score < conditioned_best_score:
                        conditioned_best_score = refined_score
                        conditioned_best_params = refined_params
                        conditioned_best_signature = refined_signature
                        conditioned_seed_rank = seed_rank

                assert conditioned_best_params is not None
                assert conditioned_best_signature is not None
                conditioned_geometry, conditioned_weight, conditioned_alpha = symmetry_aware_errors(
                    true_params,
                    conditioned_best_params,
                )
                conditioned_fit_rmse = rmse(conditioned_best_signature, rotated_signature)

                family_best_params = None
                family_best_signature = None
                family_best_score = float("inf")
                family_best_seed_rank = 1
                family_best_seed_score = float("inf")
                family_best_seed_params = None
                for seed_rank, bank_idx in enumerate(top_seed_indices, start=1):
                    seed_params = bank_params[bank_idx]
                    seed_score = float(marginalized_scores[bank_idx])
                    refined_params, refined_signature, _, refined_score = family_switching_refine(
                        observed_signature,
                        mask,
                        seed_params,
                        temperature,
                    )
                    if refined_score < family_best_score:
                        family_best_score = refined_score
                        family_best_params = refined_params
                        family_best_signature = refined_signature
                        family_best_seed_rank = seed_rank
                        family_best_seed_score = seed_score
                        family_best_seed_params = seed_params

                assert family_best_params is not None
                assert family_best_signature is not None
                assert family_best_seed_params is not None
                family_geometry, family_weight, family_alpha = symmetry_aware_errors(true_params, family_best_params)
                family_fit_rmse = rmse(family_best_signature, rotated_signature)

                oracle_params, oracle_signature = oracle_prediction(
                    observed_signature,
                    mask,
                    true_shift,
                    bank_signatures,
                    bank_params,
                )
                oracle_geometry, oracle_weight, oracle_alpha = symmetry_aware_errors(true_params, oracle_params)
                oracle_fit_rmse = rmse(oracle_signature, clean_signature)

                rows.append(
                    TrialRow(
                        condition=condition,
                        geometry_skew_bin=skew_bin,
                        trial_in_cell=trial_idx,
                        true_rho=float(true_params[0]),
                        true_t=float(true_params[1]),
                        true_h=float(true_params[2]),
                        true_w1=float(true_params[3]),
                        true_w2=float(true_params[4]),
                        true_w3=float(1.0 - true_params[3] - true_params[4]),
                        true_alpha=float(true_params[5]),
                        true_rotation_shift=int(true_shift),
                        marginalized_geometry_mae=float(marginalized_geometry),
                        marginalized_weight_mae=float(marginalized_weight),
                        marginalized_alpha_error=float(marginalized_alpha),
                        marginalized_fit_rmse=float(marginalized_fit_rmse),
                        conditioned_geometry_mae=float(conditioned_geometry),
                        conditioned_weight_mae=float(conditioned_weight),
                        conditioned_alpha_error=float(conditioned_alpha),
                        conditioned_fit_rmse=float(conditioned_fit_rmse),
                        conditioned_seed_rank=int(conditioned_seed_rank),
                        family_geometry_mae=float(family_geometry),
                        family_weight_mae=float(family_weight),
                        family_alpha_error=float(family_alpha),
                        family_fit_rmse=float(family_fit_rmse),
                        family_winner_seed_rank=int(family_best_seed_rank),
                        family_switched_seed=int(family_best_seed_rank > 1),
                        family_score_improvement_over_seed=float(family_best_seed_score - family_best_score),
                        family_delta_rho=float(abs(family_best_params[0] - family_best_seed_params[0])),
                        family_delta_t=float(abs(family_best_params[1] - family_best_seed_params[1])),
                        family_delta_h=float(abs(family_best_params[2] - family_best_seed_params[2])),
                        family_delta_alpha=float(abs(family_best_params[5] - family_best_seed_params[5])),
                        oracle_geometry_mae=float(oracle_geometry),
                        oracle_weight_mae=float(oracle_weight),
                        oracle_alpha_error=float(oracle_alpha),
                        oracle_fit_rmse=float(oracle_fit_rmse),
                    )
                )

    trial_dicts = [row.__dict__ for row in rows]
    cell_rows = summarize_cells(rows)
    focus_summary = build_focus_summary(cell_rows)

    write_csv(os.path.join(OUTPUT_DIR, "family_switching_refinement_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "family_switching_refinement_summary.csv"), cell_rows)

    plot_focus_heatmaps(os.path.join(FIGURE_DIR, "family_switching_refinement_focus.png"), cell_rows)
    plot_method_bars(os.path.join(FIGURE_DIR, "family_switching_refinement_method_bars.png"), cell_rows)

    summary = {
        "focus_conditions": FOCUS_CONDITIONS,
        "focus_alpha_bin": FOCUS_ALPHA_BIN,
        "trials_per_cell": float(TRIALS_PER_CELL),
        "top_k_seeds": float(TOP_K_SEEDS),
        "grid_points": float(GRID_POINTS),
        "refinement_rounds": float(REFINEMENT_ROUNDS),
        "audits": audits,
        "focus_summary": focus_summary,
    }

    with open(os.path.join(OUTPUT_DIR, "family_switching_refinement_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_cell": cell_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_cell": cell_rows}, indent=2))


if __name__ == "__main__":
    main()
