"""
Post-roadmap extension: candidate-conditioned alignment experiment.

This experiment tests a sharper version of the symmetry-orbit hypothesis.

The current pose-free anisotropic inverse already does per-candidate best-shift
matching. The new question is whether a richer candidate-conditioned local
disentanglement step can recover more alpha headroom in the hard cells:

1. score the bank with an observation-only shift-marginalized pose rule
2. keep the top candidate seeds
3. for each seed, hold geometry and weights fixed and locally search alpha
   while re-solving pose against that candidate family
4. select the best refined candidate

This is meant to test whether local alpha-orbit aliasing can be reduced once
the candidate family itself provides the reference frame.
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

from run_oracle_alignment_ceiling_experiment import oracle_align_observation
from run_orientation_locking_experiment import nearest_neighbor_aligned, rmse
from run_pose_free_weighted_anisotropic_inverse_experiment import nearest_neighbor_pose_free
from run_pose_free_weighted_inverse_experiment import build_shift_stack, observe_pose_free_signature
from run_shift_marginalized_pose_experiment import (
    marginalized_candidate_scores,
    shift_error_matrix,
    softmin_temperature,
)
from run_weighted_anisotropic_inverse_experiment import (
    ALPHA_MAX,
    ALPHA_MIN,
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


TOP_K_SEEDS = 3
TRIALS_PER_CELL = 4
SEED_AUDIT_CASES = 30
FAMILY_AUDIT_CASES = 30

ALPHA_STRENGTH_BIN_EDGES = [0.0, 0.10, 0.25, float("inf")]
ALPHA_STRENGTH_BIN_LABELS = ["weak", "moderate", "strong"]

GEOMETRY_SKEW_BIN_EDGES = [0.0, 0.20, 0.45, float("inf")]
GEOMETRY_SKEW_BIN_LABELS = ["low_skew", "mid_skew", "high_skew"]

COARSE_ALPHA_RADIUS = 0.22
COARSE_ALPHA_POINTS = 9
FINE_ALPHA_RADIUS = 0.07
FINE_ALPHA_POINTS = 7


@dataclass
class TrialRow:
    condition: str
    alpha_strength_bin: str
    geometry_skew_bin: str
    trial_in_cell: int
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    baseline_geometry_mae: float
    baseline_weight_mae: float
    baseline_alpha_error: float
    baseline_fit_rmse: float
    marginalized_geometry_mae: float
    marginalized_weight_mae: float
    marginalized_alpha_error: float
    marginalized_fit_rmse: float
    conditioned_geometry_mae: float
    conditioned_weight_mae: float
    conditioned_alpha_error: float
    conditioned_fit_rmse: float
    conditioned_seed_rank: int
    conditioned_alpha_shift: float
    oracle_geometry_mae: float
    oracle_weight_mae: float
    oracle_alpha_error: float
    oracle_fit_rmse: float


def alpha_strength(alpha: float) -> float:
    return float(abs(math.log(alpha)))


def geometry_skew_from_t(t: float) -> float:
    return float(abs(t))


def assign_bin(value: float, edges: list[float], labels: list[str]) -> str:
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        if lo <= value < hi:
            return label
    return labels[-1]


def sample_conditioned_parameters(
    rng: np.random.Generator,
    alpha_bin: str,
    skew_bin: str,
) -> tuple[float, float, float, float, float, float]:
    for _ in range(10000):
        params = sample_anisotropic_parameters(rng)
        if assign_bin(alpha_strength(float(params[5])), ALPHA_STRENGTH_BIN_EDGES, ALPHA_STRENGTH_BIN_LABELS) != alpha_bin:
            continue
        if assign_bin(geometry_skew_from_t(float(params[1])), GEOMETRY_SKEW_BIN_EDGES, GEOMETRY_SKEW_BIN_LABELS) != skew_bin:
            continue
        return params
    raise RuntimeError(f"Failed to sample parameters for bins alpha={alpha_bin}, skew={skew_bin}")


def unique_centered_grid(center: float, radius: float, lower: float, upper: float, count: int, extra_values: list[float] | None = None) -> np.ndarray:
    values = np.linspace(max(lower, center - radius), min(upper, center + radius), count)
    if extra_values:
        values = np.concatenate([values, np.array(extra_values, dtype=float)])
    values = np.clip(values, lower, upper)
    return np.unique(values)


def evaluate_candidate_alpha(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    geometry_weight_seed: tuple[float, float, float, float, float],
    alpha: float,
    temperature: float,
) -> tuple[float, np.ndarray, int]:
    rho, t, h, w1, w2 = geometry_weight_seed
    signature = anisotropic_forward_signature((rho, t, h, w1, w2, alpha))
    shift_stack = np.stack([np.roll(signature, shift) for shift in range(len(signature))], axis=0)
    mse = shift_error_matrix(observed_signature, mask, shift_stack[None, :, :])[0]
    best_shift = int(np.argmin(mse))
    minima = float(np.min(mse))
    stable = np.exp(-(mse - minima) / temperature)
    marginalized = minima - temperature * math.log(float(np.mean(stable)))
    return float(marginalized), shift_stack[best_shift], best_shift


def candidate_conditioned_search(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    seed_params: tuple[float, float, float, float, float, float],
    temperature: float,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, int, float]:
    rho, t, h, w1, w2, seed_alpha = seed_params
    geometry_weight_seed = (rho, t, h, w1, w2)

    best_params = seed_params
    best_score, best_signature, best_shift = evaluate_candidate_alpha(
        observed_signature,
        mask,
        geometry_weight_seed,
        seed_alpha,
        temperature,
    )

    coarse_grid = unique_centered_grid(
        seed_alpha,
        COARSE_ALPHA_RADIUS,
        ALPHA_MIN,
        ALPHA_MAX,
        COARSE_ALPHA_POINTS,
        extra_values=[seed_alpha],
    )
    coarse_best_alpha = seed_alpha
    for alpha in coarse_grid:
        score, signature, shift = evaluate_candidate_alpha(observed_signature, mask, geometry_weight_seed, float(alpha), temperature)
        if score < best_score:
            best_score = score
            best_signature = signature
            best_shift = shift
            coarse_best_alpha = float(alpha)
            best_params = (rho, t, h, w1, w2, float(alpha))

    fine_grid = unique_centered_grid(
        coarse_best_alpha,
        FINE_ALPHA_RADIUS,
        ALPHA_MIN,
        ALPHA_MAX,
        FINE_ALPHA_POINTS,
        extra_values=[seed_alpha, coarse_best_alpha],
    )
    for alpha in fine_grid:
        score, signature, shift = evaluate_candidate_alpha(observed_signature, mask, geometry_weight_seed, float(alpha), temperature)
        if score < best_score:
            best_score = score
            best_signature = signature
            best_shift = shift
            best_params = (rho, t, h, w1, w2, float(alpha))

    return best_params, best_signature, best_shift, float(best_score)


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    order = np.argsort(scores)
    return [int(idx) for idx in order[:k]]


def aggregate_by_condition(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        baseline_alpha_mean = mean("baseline_alpha_error")
        marginalized_alpha_mean = mean("marginalized_alpha_error")
        conditioned_alpha_mean = mean("conditioned_alpha_error")
        oracle_alpha_mean = mean("oracle_alpha_error")
        oracle_headroom_mean = baseline_alpha_mean - oracle_alpha_mean

        if oracle_headroom_mean > 1.0e-6:
            marginalized_fraction = float((baseline_alpha_mean - marginalized_alpha_mean) / oracle_headroom_mean)
            conditioned_fraction = float((baseline_alpha_mean - conditioned_alpha_mean) / oracle_headroom_mean)
        else:
            marginalized_fraction = float("nan")
            conditioned_fraction = float("nan")

        summary.append(
            {
                "condition": name,
                "baseline_alpha_error_mean": baseline_alpha_mean,
                "marginalized_alpha_error_mean": marginalized_alpha_mean,
                "conditioned_alpha_error_mean": conditioned_alpha_mean,
                "oracle_alpha_error_mean": oracle_alpha_mean,
                "baseline_geometry_mae_mean": mean("baseline_geometry_mae"),
                "marginalized_geometry_mae_mean": mean("marginalized_geometry_mae"),
                "conditioned_geometry_mae_mean": mean("conditioned_geometry_mae"),
                "oracle_geometry_mae_mean": mean("oracle_geometry_mae"),
                "baseline_weight_mae_mean": mean("baseline_weight_mae"),
                "marginalized_weight_mae_mean": mean("marginalized_weight_mae"),
                "conditioned_weight_mae_mean": mean("conditioned_weight_mae"),
                "oracle_weight_mae_mean": mean("oracle_weight_mae"),
                "baseline_fit_rmse_mean": mean("baseline_fit_rmse"),
                "marginalized_fit_rmse_mean": mean("marginalized_fit_rmse"),
                "conditioned_fit_rmse_mean": mean("conditioned_fit_rmse"),
                "oracle_fit_rmse_mean": mean("oracle_fit_rmse"),
                "oracle_headroom_mean": oracle_headroom_mean,
                "marginalized_fraction_of_oracle_gain_mean": marginalized_fraction,
                "conditioned_fraction_of_oracle_gain_mean": conditioned_fraction,
                "conditioned_vs_marginalized_alpha_ratio": float(
                    marginalized_alpha_mean / max(conditioned_alpha_mean, 1.0e-12)
                ),
                "conditioned_seed_rank_mean": mean("conditioned_seed_rank"),
                "conditioned_alpha_shift_mean": mean("conditioned_alpha_shift"),
            }
        )
    return summary


def aggregate_by_cell(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        for alpha_bin in ALPHA_STRENGTH_BIN_LABELS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                subset = [
                    row
                    for row in rows
                    if row.condition == name
                    and row.alpha_strength_bin == alpha_bin
                    and row.geometry_skew_bin == skew_bin
                ]
                if not subset:
                    continue

                def mean(attr: str) -> float:
                    return float(np.mean([getattr(row, attr) for row in subset]))

                baseline_alpha_mean = mean("baseline_alpha_error")
                marginalized_alpha_mean = mean("marginalized_alpha_error")
                conditioned_alpha_mean = mean("conditioned_alpha_error")
                oracle_alpha_mean = mean("oracle_alpha_error")
                oracle_headroom_mean = baseline_alpha_mean - oracle_alpha_mean

                if oracle_headroom_mean > 1.0e-6:
                    marginalized_fraction = float((baseline_alpha_mean - marginalized_alpha_mean) / oracle_headroom_mean)
                    conditioned_fraction = float((baseline_alpha_mean - conditioned_alpha_mean) / oracle_headroom_mean)
                else:
                    marginalized_fraction = float("nan")
                    conditioned_fraction = float("nan")

                summary.append(
                    {
                        "condition": name,
                        "alpha_strength_bin": alpha_bin,
                        "geometry_skew_bin": skew_bin,
                        "count": len(subset),
                        "baseline_alpha_error_mean": baseline_alpha_mean,
                        "marginalized_alpha_error_mean": marginalized_alpha_mean,
                        "conditioned_alpha_error_mean": conditioned_alpha_mean,
                        "oracle_alpha_error_mean": oracle_alpha_mean,
                        "marginalized_fraction_of_oracle_gain_mean": marginalized_fraction,
                        "conditioned_fraction_of_oracle_gain_mean": conditioned_fraction,
                        "conditioned_minus_marginalized_oracle_gain": float(conditioned_fraction - marginalized_fraction)
                        if np.isfinite(marginalized_fraction) and np.isfinite(conditioned_fraction)
                        else float("nan"),
                        "conditioned_vs_marginalized_alpha_ratio": float(
                            marginalized_alpha_mean / max(conditioned_alpha_mean, 1.0e-12)
                        ),
                    }
                )
    return summary


def summarize_hard_cells(cell_rows: list[dict[str, float | str]]) -> dict[str, float]:
    subset = [
        row
        for row in cell_rows
        if str(row["condition"]) in {"sparse_full_noisy", "sparse_partial_high_noise"}
        and str(row["alpha_strength_bin"]) == "moderate"
        and str(row["geometry_skew_bin"]) in {"low_skew", "mid_skew"}
        and int(row["count"]) >= 1
    ]

    def mean(metric: str) -> float:
        vals = [float(row[metric]) for row in subset if np.isfinite(float(row[metric]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "cell_count": float(len(subset)),
        "baseline_alpha_error_mean": mean("baseline_alpha_error_mean"),
        "marginalized_alpha_error_mean": mean("marginalized_alpha_error_mean"),
        "conditioned_alpha_error_mean": mean("conditioned_alpha_error_mean"),
        "oracle_alpha_error_mean": mean("oracle_alpha_error_mean"),
        "marginalized_fraction_of_oracle_gain_mean": mean("marginalized_fraction_of_oracle_gain_mean"),
        "conditioned_fraction_of_oracle_gain_mean": mean("conditioned_fraction_of_oracle_gain_mean"),
        "conditioned_minus_marginalized_oracle_gain_mean": mean("conditioned_minus_marginalized_oracle_gain"),
        "conditioned_vs_marginalized_alpha_ratio_mean": mean("conditioned_vs_marginalized_alpha_ratio"),
    }


def plot_overview(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    x = np.arange(len(conditions))
    width = 0.20

    baseline_alpha = np.array([float(item["baseline_alpha_error_mean"]) for item in summary_rows])
    marginalized_alpha = np.array([float(item["marginalized_alpha_error_mean"]) for item in summary_rows])
    conditioned_alpha = np.array([float(item["conditioned_alpha_error_mean"]) for item in summary_rows])
    oracle_alpha = np.array([float(item["oracle_alpha_error_mean"]) for item in summary_rows])

    fig, ax = plt.subplots(figsize=(14.0, 6.2), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.98)

    ax.bar(x - 1.5 * width, baseline_alpha, width=width, color="#e76f51", label="hard best-shift")
    ax.bar(x - 0.5 * width, marginalized_alpha, width=width, color="#1d3557", label="shift-marginalized")
    ax.bar(x + 0.5 * width, conditioned_alpha, width=width, color="#2a9d8f", label="candidate-conditioned")
    ax.bar(x + 1.5 * width, oracle_alpha, width=width, color="#6a4c93", label="oracle")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Alpha recovery across practical and oracle methods")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    fig.suptitle(
        "Candidate-Conditioned Alignment A: Does Local Shift-Alpha Search Help?",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def cell_matrix(
    cell_rows: list[dict[str, float | str]],
    condition: str,
    metric: str,
) -> np.ndarray:
    matrix = np.full((len(ALPHA_STRENGTH_BIN_LABELS), len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
    for row in cell_rows:
        if str(row["condition"]) != condition:
            continue
        i = ALPHA_STRENGTH_BIN_LABELS.index(str(row["alpha_strength_bin"]))
        j = GEOMETRY_SKEW_BIN_LABELS.index(str(row["geometry_skew_bin"]))
        matrix[i, j] = float(row[metric])
    return matrix


def plot_hard_cell_maps(path: str, cell_rows: list[dict[str, float | str]]) -> None:
    conditions = ["sparse_full_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(2, len(conditions), figsize=(12.8, 7.6), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.98, wspace=0.24, hspace=0.34)

    for col_idx, condition in enumerate(conditions):
        gain_matrix = cell_matrix(cell_rows, condition, "conditioned_minus_marginalized_oracle_gain")
        ratio_matrix = cell_matrix(cell_rows, condition, "conditioned_vs_marginalized_alpha_ratio")

        sns.heatmap(
            gain_matrix,
            ax=axes[0, col_idx],
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            xticklabels=GEOMETRY_SKEW_BIN_LABELS,
            yticklabels=ALPHA_STRENGTH_BIN_LABELS,
            cbar=(col_idx == len(conditions) - 1),
            cbar_kws={"label": "extra oracle-gain capture"} if col_idx == len(conditions) - 1 else None,
            vmin=-1.0,
            vmax=1.0,
        )
        axes[0, col_idx].set_title(f"{condition}\nconditioned minus marginalized oracle gain")
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
            yticklabels=ALPHA_STRENGTH_BIN_LABELS,
            cbar=(col_idx == len(conditions) - 1),
            cbar_kws={"label": "marginalized / conditioned alpha error"} if col_idx == len(conditions) - 1 else None,
            vmin=0.5,
            vmax=2.0,
        )
        axes[1, col_idx].set_title(f"{condition}\nconditioned alpha improvement factor")
        axes[1, col_idx].set_xlabel("geometry skew |t| bin")
        if col_idx == 0:
            axes[1, col_idx].set_ylabel("anisotropy strength")
        else:
            axes[1, col_idx].set_ylabel("")

    fig.suptitle(
        "Candidate-Conditioned Alignment B: Where The New Method Helps",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def audit_seed_preservation(
    rng: np.random.Generator,
) -> dict[str, float]:
    max_score_delta = 0.0
    bank_params, _ = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    for _ in range(SEED_AUDIT_CASES):
        seed = bank_params[int(rng.integers(0, len(bank_params)))]
        clean_signature = anisotropic_forward_signature(seed)
        regime = OBSERVATION_REGIMES[int(rng.integers(0, len(OBSERVATION_REGIMES)))]
        _, observed_signature, mask, _ = observe_pose_free_signature(clean_signature, regime, rng)
        temperature = softmin_temperature(regime)
        seed_score, _, _ = evaluate_candidate_alpha(observed_signature, mask, seed[:5], float(seed[5]), temperature)
        _, _, _, refined_score = candidate_conditioned_search(observed_signature, mask, seed, temperature)
        max_score_delta = max(max_score_delta, refined_score - seed_score)
    return {
        "audit_cases": float(SEED_AUDIT_CASES),
        "max_refined_minus_seed_score": float(max_score_delta),
    }


def audit_candidate_family_clean_recovery(
    rng: np.random.Generator,
) -> dict[str, float]:
    full_regime = next(regime for regime in OBSERVATION_REGIMES if str(regime["name"]) == "full_clean")
    max_alpha_error = 0.0
    exact_shift_fraction = 0
    for _ in range(FAMILY_AUDIT_CASES):
        true_params = sample_anisotropic_parameters(rng)
        clean_signature = anisotropic_forward_signature(true_params)
        rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, full_regime, rng)

        seed_alpha = float(np.clip(true_params[5] + rng.uniform(-0.10, 0.10), ALPHA_MIN, ALPHA_MAX))
        seed_params = (true_params[0], true_params[1], true_params[2], true_params[3], true_params[4], seed_alpha)
        refined_params, refined_signature, refined_shift, _ = candidate_conditioned_search(
            observed_signature,
            mask,
            seed_params,
            softmin_temperature(full_regime),
        )
        max_alpha_error = max(max_alpha_error, abs(float(refined_params[5]) - float(true_params[5])))
        if rmse(refined_signature, rotated_signature) <= 1.0e-12:
            exact_shift_fraction += 1
    return {
        "audit_cases": float(FAMILY_AUDIT_CASES),
        "max_alpha_error_after_family_refinement": float(max_alpha_error),
        "exact_rotated_signature_recovery_fraction": float(exact_shift_fraction / FAMILY_AUDIT_CASES),
    }


def main() -> None:
    rng = np.random.default_rng(20260324)

    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)

    audits = {
        "seed_preservation": audit_seed_preservation(np.random.default_rng(20260324)),
        "candidate_family_clean_recovery": audit_candidate_family_clean_recovery(np.random.default_rng(20260325)),
    }

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for alpha_bin in ALPHA_STRENGTH_BIN_LABELS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                for trial_idx in range(TRIALS_PER_CELL):
                    true_params = sample_conditioned_parameters(rng, alpha_bin, skew_bin)
                    clean_signature = anisotropic_forward_signature(true_params)
                    rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)
                    temperature = softmin_temperature(regime)

                    baseline_params, baseline_signature, _ = nearest_neighbor_pose_free(
                        observed_signature,
                        mask,
                        shifted_bank,
                        bank_params,
                    )
                    baseline_geometry, baseline_weight, baseline_alpha = symmetry_aware_errors(true_params, baseline_params)
                    baseline_fit_rmse = rmse(baseline_signature, rotated_signature)

                    marginalized_scores, marginalized_best_shifts = marginalized_candidate_scores(
                        observed_signature,
                        mask,
                        shifted_bank,
                        temperature,
                    )
                    marginalized_idx = int(np.argmin(marginalized_scores))
                    marginalized_params = bank_params[marginalized_idx]
                    marginalized_signature = shifted_bank[marginalized_idx, int(marginalized_best_shifts[marginalized_idx])]
                    marginalized_geometry, marginalized_weight, marginalized_alpha = symmetry_aware_errors(true_params, marginalized_params)
                    marginalized_fit_rmse = rmse(marginalized_signature, rotated_signature)

                    conditioned_best_score = float("inf")
                    conditioned_best_params = marginalized_params
                    conditioned_best_signature = marginalized_signature
                    conditioned_seed_rank = 0

                    for seed_rank, idx in enumerate(top_k_indices(marginalized_scores, TOP_K_SEEDS), start=1):
                        seed_params = bank_params[idx]
                        refined_params, refined_signature, _, refined_score = candidate_conditioned_search(
                            observed_signature,
                            mask,
                            seed_params,
                            temperature,
                        )
                        if refined_score < conditioned_best_score:
                            conditioned_best_score = refined_score
                            conditioned_best_params = refined_params
                            conditioned_best_signature = refined_signature
                            conditioned_seed_rank = seed_rank

                    conditioned_geometry, conditioned_weight, conditioned_alpha = symmetry_aware_errors(true_params, conditioned_best_params)
                    conditioned_fit_rmse = rmse(conditioned_best_signature, rotated_signature)

                    oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
                    oracle_params, oracle_signature = nearest_neighbor_aligned(oracle_observed, oracle_mask, bank_signatures, bank_params)
                    oracle_geometry, oracle_weight, oracle_alpha = symmetry_aware_errors(true_params, oracle_params)
                    oracle_fit_rmse = rmse(oracle_signature, clean_signature)

                    rows.append(
                        TrialRow(
                            condition=str(regime["name"]),
                            alpha_strength_bin=alpha_bin,
                            geometry_skew_bin=skew_bin,
                            trial_in_cell=trial_idx,
                            true_alpha=float(true_params[5]),
                            true_t=float(true_params[1]),
                            true_rotation_shift=int(true_shift),
                            baseline_geometry_mae=float(baseline_geometry),
                            baseline_weight_mae=float(baseline_weight),
                            baseline_alpha_error=float(baseline_alpha),
                            baseline_fit_rmse=float(baseline_fit_rmse),
                            marginalized_geometry_mae=float(marginalized_geometry),
                            marginalized_weight_mae=float(marginalized_weight),
                            marginalized_alpha_error=float(marginalized_alpha),
                            marginalized_fit_rmse=float(marginalized_fit_rmse),
                            conditioned_geometry_mae=float(conditioned_geometry),
                            conditioned_weight_mae=float(conditioned_weight),
                            conditioned_alpha_error=float(conditioned_alpha),
                            conditioned_fit_rmse=float(conditioned_fit_rmse),
                            conditioned_seed_rank=int(conditioned_seed_rank),
                            conditioned_alpha_shift=float(abs(float(conditioned_best_params[5]) - float(marginalized_params[5]))),
                            oracle_geometry_mae=float(oracle_geometry),
                            oracle_weight_mae=float(oracle_weight),
                            oracle_alpha_error=float(oracle_alpha),
                            oracle_fit_rmse=float(oracle_fit_rmse),
                        )
                    )

    trial_dicts = [row.__dict__ for row in rows]
    by_condition = aggregate_by_condition(rows)
    by_cell = aggregate_by_cell(rows)
    hard_cell_summary = summarize_hard_cells(by_cell)

    write_csv(os.path.join(OUTPUT_DIR, "candidate_conditioned_alignment_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "candidate_conditioned_alignment_summary.csv"), by_condition)
    write_csv(os.path.join(OUTPUT_DIR, "candidate_conditioned_alignment_cells.csv"), by_cell)

    plot_overview(os.path.join(FIGURE_DIR, "candidate_conditioned_alignment_overview.png"), by_condition)
    plot_hard_cell_maps(os.path.join(FIGURE_DIR, "candidate_conditioned_alignment_hard_cells.png"), by_cell)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "top_k_seeds": float(TOP_K_SEEDS),
        "trials_per_cell": float(TRIALS_PER_CELL),
        "audits": audits,
        "best_conditioned_vs_marginalized_alpha_ratio": float(
            max(float(row["conditioned_vs_marginalized_alpha_ratio"]) for row in by_condition)
        ),
        "worst_conditioned_vs_marginalized_alpha_ratio": float(
            min(float(row["conditioned_vs_marginalized_alpha_ratio"]) for row in by_condition)
        ),
        "best_conditioned_fraction_of_oracle_gain": float(
            max(float(row["conditioned_fraction_of_oracle_gain_mean"]) for row in by_condition if np.isfinite(float(row["conditioned_fraction_of_oracle_gain_mean"])))
        ),
        "worst_conditioned_fraction_of_oracle_gain": float(
            min(float(row["conditioned_fraction_of_oracle_gain_mean"]) for row in by_condition if np.isfinite(float(row["conditioned_fraction_of_oracle_gain_mean"])))
        ),
        "hard_cell_summary": hard_cell_summary,
    }

    with open(os.path.join(OUTPUT_DIR, "candidate_conditioned_alignment_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": by_condition, "by_cell": by_cell}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": by_condition, "by_cell": by_cell}, indent=2))


if __name__ == "__main__":
    main()
