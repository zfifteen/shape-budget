"""
Post-roadmap extension: competitive hybrid resolver for the pose-free
anisotropic bottleneck.

This experiment stops trying to predict the right refinement family in advance.
Instead it runs the two strongest existing refiners from the same top
marginalized seeds:

1. fixed-family candidate-conditioned shift/alpha search
2. geometry-plus-alpha family switching

It then chooses the refined candidate with the lower marginalized score, using
the fixed-family path as the tie-break because that path is empirically safer in
the sparse-partial branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

ALPHA_STRENGTH_BIN_LABELS, GEOMETRY_SKEW_BIN_LABELS, candidate_conditioned_search, evaluate_candidate_alpha, sample_conditioned_parameters, top_k_indices = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "ALPHA_STRENGTH_BIN_LABELS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "candidate_conditioned_search",
    "evaluate_candidate_alpha",
    "sample_conditioned_parameters",
    "top_k_indices",
)

evaluate_params, family_switching_refine = load_symbols(
    "run_family_switching_refinement_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/family-switching-refinement/run.py",
    "evaluate_params",
    "family_switching_refine",
)

oracle_align_observation, = load_symbols(
    "run_oracle_alignment_ceiling_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/oracle-alignment-ceiling/run.py",
    "oracle_align_observation",
)

nearest_neighbor_aligned, rmse = load_symbols(
    "run_orientation_locking_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/orientation-locking/run.py",
    "nearest_neighbor_aligned",
    "rmse",
)

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

marginalized_candidate_scores, softmin_temperature = load_symbols(
    "run_shift_marginalized_pose_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/shift-marginalized-pose/run.py",
    "marginalized_candidate_scores",
    "softmin_temperature",
)

REFERENCE_BANK_SIZE, anisotropic_forward_signature, build_reference_bank, sample_anisotropic_parameters, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "REFERENCE_BANK_SIZE",
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

TOP_K_SEEDS = 3
TRIALS_PER_CELL = 4
AUDIT_CASES = 12
FOCUS_CONDITIONS = ["sparse_full_noisy", "sparse_partial_high_noise"]
FOCUS_ALPHA_BINS = ["moderate"]
TIE_SCORE_EPS = 1.0e-12


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
    conditioned_score: float
    conditioned_seed_rank: int
    family_geometry_mae: float
    family_weight_mae: float
    family_alpha_error: float
    family_fit_rmse: float
    family_score: float
    family_seed_rank: int
    hybrid_geometry_mae: float
    hybrid_weight_mae: float
    hybrid_alpha_error: float
    hybrid_fit_rmse: float
    hybrid_score: float
    hybrid_choose_family: int
    hybrid_score_margin: float
    oracle_combo_alpha_error: float
    oracle_combo_choose_family: int
    hybrid_matches_oracle_combo: int
    oracle_pose_geometry_mae: float
    oracle_pose_weight_mae: float
    oracle_pose_alpha_error: float
    oracle_pose_fit_rmse: float


def choose_hybrid_path(
    conditioned_params: tuple[float, float, float, float, float, float],
    conditioned_signature: np.ndarray,
    conditioned_score: float,
    family_params: tuple[float, float, float, float, float, float],
    family_signature: np.ndarray,
    family_score: float,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, float, int, float]:
    score_margin = float(conditioned_score - family_score)
    if family_score + TIE_SCORE_EPS < conditioned_score:
        return family_params, family_signature, float(family_score), 1, score_margin
    return conditioned_params, conditioned_signature, float(conditioned_score), 0, score_margin


def oracle_combo_path(
    conditioned_params: tuple[float, float, float, float, float, float],
    conditioned_signature: np.ndarray,
    conditioned_alpha_error: float,
    family_params: tuple[float, float, float, float, float, float],
    family_signature: np.ndarray,
    family_alpha_error: float,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, int]:
    if family_alpha_error + 1.0e-12 < conditioned_alpha_error:
        return family_params, family_signature, 1
    return conditioned_params, conditioned_signature, 0


def aggregate_by_condition(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for name in FOCUS_CONDITIONS:
        subset = [row for row in rows if row.condition == name]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        baseline_alpha = mean("baseline_alpha_error")
        marginalized_alpha = mean("marginalized_alpha_error")
        conditioned_alpha = mean("conditioned_alpha_error")
        family_alpha = mean("family_alpha_error")
        hybrid_alpha = mean("hybrid_alpha_error")
        oracle_combo_alpha = mean("oracle_combo_alpha_error")
        oracle_pose_alpha = mean("oracle_pose_alpha_error")
        oracle_pose_headroom = baseline_alpha - oracle_pose_alpha

        def fraction_of_oracle_gain(method_alpha: float) -> float:
            if oracle_pose_headroom <= 1.0e-6:
                return float("nan")
            return float((baseline_alpha - method_alpha) / oracle_pose_headroom)

        summary.append(
            {
                "condition": name,
                "baseline_alpha_error_mean": baseline_alpha,
                "marginalized_alpha_error_mean": marginalized_alpha,
                "conditioned_alpha_error_mean": conditioned_alpha,
                "family_alpha_error_mean": family_alpha,
                "hybrid_alpha_error_mean": hybrid_alpha,
                "oracle_combo_alpha_error_mean": oracle_combo_alpha,
                "oracle_pose_alpha_error_mean": oracle_pose_alpha,
                "conditioned_fraction_of_oracle_gain_mean": fraction_of_oracle_gain(conditioned_alpha),
                "family_fraction_of_oracle_gain_mean": fraction_of_oracle_gain(family_alpha),
                "hybrid_fraction_of_oracle_gain_mean": fraction_of_oracle_gain(hybrid_alpha),
                "hybrid_vs_marginalized_alpha_ratio": float(marginalized_alpha / max(hybrid_alpha, 1.0e-12)),
                "hybrid_vs_conditioned_alpha_ratio": float(conditioned_alpha / max(hybrid_alpha, 1.0e-12)),
                "hybrid_vs_family_alpha_ratio": float(family_alpha / max(hybrid_alpha, 1.0e-12)),
                "hybrid_over_oracle_combo_alpha_mean": float(hybrid_alpha - oracle_combo_alpha),
                "hybrid_choose_family_fraction": mean("hybrid_choose_family"),
                "oracle_combo_choose_family_fraction": mean("oracle_combo_choose_family"),
                "hybrid_matches_oracle_combo_fraction": mean("hybrid_matches_oracle_combo"),
                "hybrid_geometry_mae_mean": mean("hybrid_geometry_mae"),
                "hybrid_weight_mae_mean": mean("hybrid_weight_mae"),
                "hybrid_fit_rmse_mean": mean("hybrid_fit_rmse"),
            }
        )
    return summary


def aggregate_by_cell(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for name in FOCUS_CONDITIONS:
        for alpha_bin in FOCUS_ALPHA_BINS:
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

                conditioned_alpha = mean("conditioned_alpha_error")
                family_alpha = mean("family_alpha_error")
                hybrid_alpha = mean("hybrid_alpha_error")
                oracle_combo_alpha = mean("oracle_combo_alpha_error")

                summary.append(
                    {
                        "condition": name,
                        "alpha_strength_bin": alpha_bin,
                        "geometry_skew_bin": skew_bin,
                        "count": len(subset),
                        "marginalized_alpha_error_mean": mean("marginalized_alpha_error"),
                        "conditioned_alpha_error_mean": conditioned_alpha,
                        "family_alpha_error_mean": family_alpha,
                        "hybrid_alpha_error_mean": hybrid_alpha,
                        "oracle_combo_alpha_error_mean": oracle_combo_alpha,
                        "hybrid_vs_marginalized_alpha_ratio": float(
                            mean("marginalized_alpha_error") / max(hybrid_alpha, 1.0e-12)
                        ),
                        "hybrid_over_oracle_combo_alpha_mean": float(hybrid_alpha - oracle_combo_alpha),
                        "hybrid_choose_family_fraction": mean("hybrid_choose_family"),
                        "oracle_combo_choose_family_fraction": mean("oracle_combo_choose_family"),
                        "hybrid_matches_oracle_combo_fraction": mean("hybrid_matches_oracle_combo"),
                        "mean_hybrid_score_margin": mean("hybrid_score_margin"),
                    }
                )
    return summary


def build_focus_summary(by_condition: list[dict[str, float | str]], by_cell: list[dict[str, float | str]]) -> dict[str, float]:
    focus_rows = [row for row in by_cell if str(row["condition"]) in set(FOCUS_CONDITIONS) and str(row["alpha_strength_bin"]) in set(FOCUS_ALPHA_BINS)]

    def extrema(metric: str, fn) -> float:
        values = [float(row[metric]) for row in focus_rows]
        return float(fn(values))

    sparse_full = [row for row in by_condition if str(row["condition"]) == "sparse_full_noisy"][0]
    sparse_partial = [row for row in by_condition if str(row["condition"]) == "sparse_partial_high_noise"][0]

    return {
        "sparse_full_hybrid_vs_marginalized_alpha_ratio": float(sparse_full["hybrid_vs_marginalized_alpha_ratio"]),
        "sparse_partial_hybrid_vs_marginalized_alpha_ratio": float(sparse_partial["hybrid_vs_marginalized_alpha_ratio"]),
        "sparse_full_hybrid_match_oracle_fraction": float(sparse_full["hybrid_matches_oracle_combo_fraction"]),
        "sparse_partial_hybrid_match_oracle_fraction": float(sparse_partial["hybrid_matches_oracle_combo_fraction"]),
        "best_focus_hybrid_vs_marginalized_alpha_ratio": extrema("hybrid_vs_marginalized_alpha_ratio", max),
        "worst_focus_hybrid_vs_marginalized_alpha_ratio": extrema("hybrid_vs_marginalized_alpha_ratio", min),
        "largest_focus_hybrid_over_oracle_combo_alpha": extrema("hybrid_over_oracle_combo_alpha_mean", max),
        "largest_focus_hybrid_choose_family_fraction": extrema("hybrid_choose_family_fraction", max),
    }


def plot_overview(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(row["condition"]) for row in summary_rows]
    x = np.arange(len(conditions))
    width = 0.16

    marginalized = np.array([float(row["marginalized_alpha_error_mean"]) for row in summary_rows])
    conditioned = np.array([float(row["conditioned_alpha_error_mean"]) for row in summary_rows])
    family = np.array([float(row["family_alpha_error_mean"]) for row in summary_rows])
    hybrid = np.array([float(row["hybrid_alpha_error_mean"]) for row in summary_rows])
    oracle_combo = np.array([float(row["oracle_combo_alpha_error_mean"]) for row in summary_rows])

    fig, ax = plt.subplots(figsize=(15.0, 6.4), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.08, right=0.98)

    ax.bar(x - 2 * width, marginalized, width=width, color="#1d3557", label="shift-marginalized")
    ax.bar(x - width, conditioned, width=width, color="#2a9d8f", label="fixed-family")
    ax.bar(x, family, width=width, color="#f4a261", label="family-switch")
    ax.bar(x + width, hybrid, width=width, color="#6a4c93", label="competitive hybrid")
    ax.bar(x + 2 * width, oracle_combo, width=width, color="#7f8c8d", label="oracle best-of-two")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Alpha recovery after direct competition between the two best refiners")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    fig.suptitle(
        "Competitive Hybrid Resolver A: Let The Two Strongest Paths Compete",
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
    matrix = np.full((len(FOCUS_ALPHA_BINS), len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
    for row in cell_rows:
        if str(row["condition"]) != condition:
            continue
        i = FOCUS_ALPHA_BINS.index(str(row["alpha_strength_bin"]))
        j = GEOMETRY_SKEW_BIN_LABELS.index(str(row["geometry_skew_bin"]))
        matrix[i, j] = float(row[metric])
    return matrix


def plot_focus_maps(path: str, cell_rows: list[dict[str, float | str]]) -> None:
    conditions = FOCUS_CONDITIONS
    fig, axes = plt.subplots(3, len(conditions), figsize=(12.8, 9.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.98, wspace=0.24, hspace=0.34)

    metric_specs = [
        (
            "hybrid_vs_marginalized_alpha_ratio",
            "marginalized / hybrid alpha error",
            "viridis",
            0.8,
            2.2,
        ),
        (
            "hybrid_over_oracle_combo_alpha_mean",
            "hybrid minus oracle-best alpha",
            "magma_r",
            0.0,
            0.12,
        ),
        (
            "hybrid_choose_family_fraction",
            "hybrid choose-family fraction",
            "coolwarm",
            0.0,
            1.0,
        ),
    ]

    for col_idx, condition in enumerate(conditions):
        for row_idx, (metric, label, cmap, vmin, vmax) in enumerate(metric_specs):
            matrix = cell_matrix(cell_rows, condition, metric)
            sns.heatmap(
                matrix,
                ax=axes[row_idx, col_idx],
                cmap=cmap,
                annot=True,
                fmt=".2f",
                xticklabels=GEOMETRY_SKEW_BIN_LABELS,
                yticklabels=FOCUS_ALPHA_BINS,
                cbar=(col_idx == len(conditions) - 1),
                cbar_kws={"label": label} if col_idx == len(conditions) - 1 else None,
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
        "Competitive Hybrid Resolver B: Focus Cells And Remaining Selection Loss",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def audit_score_compatibility(
    rng: np.random.Generator,
    bank_params: list[tuple[float, float, float, float, float, float]],
) -> dict[str, float]:
    max_delta = 0.0
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}
    for _ in range(AUDIT_CASES):
        regime = regime_map[FOCUS_CONDITIONS[int(rng.integers(0, len(FOCUS_CONDITIONS)))]]
        temperature = softmin_temperature(regime)
        params = bank_params[int(rng.integers(0, len(bank_params)))]
        clean_signature = anisotropic_forward_signature(params)
        _, observed_signature, mask, _ = observe_pose_free_signature(clean_signature, regime, rng)

        conditioned_score, _, _ = evaluate_candidate_alpha(
            observed_signature,
            mask,
            params[:5],
            float(params[5]),
            temperature,
        )
        family_score, _, _ = evaluate_params(
            observed_signature,
            mask,
            params,
            temperature,
        )
        max_delta = max(max_delta, abs(float(conditioned_score) - float(family_score)))

    return {
        "audit_cases": float(AUDIT_CASES),
        "max_conditioned_vs_family_score_delta_same_params": float(max_delta),
    }


def main() -> None:
    rng = np.random.default_rng(20260324)

    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)

    audits = {
        "score_compatibility": audit_score_compatibility(np.random.default_rng(20260324), bank_params),
    }

    rows: list[TrialRow] = []
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}
    for condition in FOCUS_CONDITIONS:
        regime = regime_map[condition]
        temperature = softmin_temperature(regime)
        for alpha_bin in FOCUS_ALPHA_BINS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                for trial_idx in range(TRIALS_PER_CELL):
                    true_params = sample_conditioned_parameters(rng, alpha_bin, skew_bin)
                    clean_signature = anisotropic_forward_signature(true_params)
                    rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

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
                    marginalized_geometry, marginalized_weight, marginalized_alpha = symmetry_aware_errors(
                        true_params,
                        marginalized_params,
                    )
                    marginalized_fit_rmse = rmse(marginalized_signature, rotated_signature)

                    conditioned_best_score = float("inf")
                    conditioned_best_params = marginalized_params
                    conditioned_best_signature = marginalized_signature
                    conditioned_seed_rank = 0

                    family_best_score = float("inf")
                    family_best_params = marginalized_params
                    family_best_signature = marginalized_signature
                    family_seed_rank = 0

                    for seed_rank, idx in enumerate(top_k_indices(marginalized_scores, TOP_K_SEEDS), start=1):
                        seed_params = bank_params[idx]

                        conditioned_params, conditioned_signature, _, conditioned_score = candidate_conditioned_search(
                            observed_signature,
                            mask,
                            seed_params,
                            temperature,
                        )
                        if conditioned_score < conditioned_best_score:
                            conditioned_best_score = conditioned_score
                            conditioned_best_params = conditioned_params
                            conditioned_best_signature = conditioned_signature
                            conditioned_seed_rank = seed_rank

                        family_params, family_signature, _, family_score = family_switching_refine(
                            observed_signature,
                            mask,
                            seed_params,
                            temperature,
                        )
                        if family_score < family_best_score:
                            family_best_score = family_score
                            family_best_params = family_params
                            family_best_signature = family_signature
                            family_seed_rank = seed_rank

                    conditioned_geometry, conditioned_weight, conditioned_alpha = symmetry_aware_errors(
                        true_params,
                        conditioned_best_params,
                    )
                    conditioned_fit_rmse = rmse(conditioned_best_signature, rotated_signature)

                    family_geometry, family_weight, family_alpha = symmetry_aware_errors(
                        true_params,
                        family_best_params,
                    )
                    family_fit_rmse = rmse(family_best_signature, rotated_signature)

                    hybrid_params, hybrid_signature, hybrid_score, hybrid_choose_family, hybrid_score_margin = choose_hybrid_path(
                        conditioned_best_params,
                        conditioned_best_signature,
                        conditioned_best_score,
                        family_best_params,
                        family_best_signature,
                        family_best_score,
                    )
                    hybrid_geometry, hybrid_weight, hybrid_alpha = symmetry_aware_errors(true_params, hybrid_params)
                    hybrid_fit_rmse = rmse(hybrid_signature, rotated_signature)

                    _, _, oracle_combo_choose_family = oracle_combo_path(
                        conditioned_best_params,
                        conditioned_best_signature,
                        float(conditioned_alpha),
                        family_best_params,
                        family_best_signature,
                        float(family_alpha),
                    )
                    oracle_combo_alpha = float(min(conditioned_alpha, family_alpha))

                    oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
                    oracle_pose_params, oracle_pose_signature = nearest_neighbor_aligned(
                        oracle_observed,
                        oracle_mask,
                        bank_signatures,
                        bank_params,
                    )
                    oracle_pose_geometry, oracle_pose_weight, oracle_pose_alpha = symmetry_aware_errors(
                        true_params,
                        oracle_pose_params,
                    )
                    oracle_pose_fit_rmse = rmse(oracle_pose_signature, clean_signature)

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
                            conditioned_score=float(conditioned_best_score),
                            conditioned_seed_rank=int(conditioned_seed_rank),
                            family_geometry_mae=float(family_geometry),
                            family_weight_mae=float(family_weight),
                            family_alpha_error=float(family_alpha),
                            family_fit_rmse=float(family_fit_rmse),
                            family_score=float(family_best_score),
                            family_seed_rank=int(family_seed_rank),
                            hybrid_geometry_mae=float(hybrid_geometry),
                            hybrid_weight_mae=float(hybrid_weight),
                            hybrid_alpha_error=float(hybrid_alpha),
                            hybrid_fit_rmse=float(hybrid_fit_rmse),
                            hybrid_score=float(hybrid_score),
                            hybrid_choose_family=int(hybrid_choose_family),
                            hybrid_score_margin=float(hybrid_score_margin),
                            oracle_combo_alpha_error=float(oracle_combo_alpha),
                            oracle_combo_choose_family=int(oracle_combo_choose_family),
                            hybrid_matches_oracle_combo=int(hybrid_choose_family == oracle_combo_choose_family),
                            oracle_pose_geometry_mae=float(oracle_pose_geometry),
                            oracle_pose_weight_mae=float(oracle_pose_weight),
                            oracle_pose_alpha_error=float(oracle_pose_alpha),
                            oracle_pose_fit_rmse=float(oracle_pose_fit_rmse),
                        )
                    )

    trial_rows = [row.__dict__ for row in rows]
    by_condition = aggregate_by_condition(rows)
    by_cell = aggregate_by_cell(rows)
    focus_summary = build_focus_summary(by_condition, by_cell)

    write_csv(os.path.join(OUTPUT_DIR, "competitive_hybrid_resolver_trials.csv"), trial_rows)
    write_csv(os.path.join(OUTPUT_DIR, "competitive_hybrid_resolver_summary.csv"), by_condition)
    write_csv(os.path.join(OUTPUT_DIR, "competitive_hybrid_resolver_cells.csv"), by_cell)

    plot_overview(os.path.join(FIGURE_DIR, "competitive_hybrid_resolver_overview.png"), by_condition)
    plot_focus_maps(os.path.join(FIGURE_DIR, "competitive_hybrid_resolver_focus.png"), by_cell)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "top_k_seeds": float(TOP_K_SEEDS),
        "trials_per_cell": float(TRIALS_PER_CELL),
        "tie_break": "conditioned_if_scores_equal",
        "audits": audits,
        "best_hybrid_vs_marginalized_alpha_ratio": float(
            max(float(row["hybrid_vs_marginalized_alpha_ratio"]) for row in by_condition)
        ),
        "worst_hybrid_vs_marginalized_alpha_ratio": float(
            min(float(row["hybrid_vs_marginalized_alpha_ratio"]) for row in by_condition)
        ),
        "best_hybrid_fraction_of_oracle_gain": float(
            max(
                float(row["hybrid_fraction_of_oracle_gain_mean"])
                for row in by_condition
                if np.isfinite(float(row["hybrid_fraction_of_oracle_gain_mean"]))
            )
        ),
        "worst_hybrid_fraction_of_oracle_gain": float(
            min(
                float(row["hybrid_fraction_of_oracle_gain_mean"])
                for row in by_condition
                if np.isfinite(float(row["hybrid_fraction_of_oracle_gain_mean"]))
            )
        ),
        "largest_hybrid_over_oracle_combo_alpha_mean": float(
            max(float(row["hybrid_over_oracle_combo_alpha_mean"]) for row in by_condition)
        ),
        "focus_summary": focus_summary,
    }

    output = {"summary": summary, "by_condition": by_condition, "by_cell": by_cell}
    with open(os.path.join(OUTPUT_DIR, "competitive_hybrid_resolver_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
