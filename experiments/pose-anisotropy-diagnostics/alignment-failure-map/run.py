"""
Post-roadmap extension: alignment failure map for pose-free anisotropic inversion.

This experiment maps where practical orientation locking helps and where it
becomes unstable, using anisotropy strength, source-geometry skew, and
observation regime as the main control axes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

oracle_align_observation, = load_symbols(
    "run_oracle_alignment_ceiling_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/oracle-alignment-ceiling/run.py",
    "oracle_align_observation",
)

align_by_shift_search, audit_alignment_invariance, audit_clean_recovery, build_aligned_bank, harmonic_alignment_score, nearest_neighbor_aligned, principal_axis_score, rmse = load_symbols(
    "run_orientation_locking_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/orientation-locking/run.py",
    "align_by_shift_search",
    "audit_alignment_invariance",
    "audit_clean_recovery",
    "build_aligned_bank",
    "harmonic_alignment_score",
    "nearest_neighbor_aligned",
    "principal_axis_score",
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

REFERENCE_BANK_SIZE, anisotropic_forward_signature, build_reference_bank, sample_anisotropic_parameters, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "REFERENCE_BANK_SIZE",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "sample_anisotropic_parameters",
    "symmetry_aware_errors",
)

OBSERVATION_REGIMES, SIGNATURE_ANGLE_COUNT, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "SIGNATURE_ANGLE_COUNT",
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

MAP_TRIALS_PER_REGIME = 60

ALPHA_STRENGTH_BIN_EDGES = [0.0, 0.10, 0.25, float("inf")]
ALPHA_STRENGTH_BIN_LABELS = ["weak", "moderate", "strong"]

GEOMETRY_SKEW_BIN_EDGES = [0.0, 0.20, 0.45, float("inf")]
GEOMETRY_SKEW_BIN_LABELS = ["low_skew", "mid_skew", "high_skew"]

@dataclass
class TrialRow:
    condition: str
    trial: int
    true_alpha: float
    alpha_strength: float
    alpha_strength_bin: str
    true_t: float
    geometry_skew: float
    geometry_skew_bin: str
    baseline_geometry_mae: float
    baseline_weight_mae: float
    baseline_alpha_error: float
    baseline_fit_rmse: float
    harmonic_geometry_mae: float
    harmonic_weight_mae: float
    harmonic_alpha_error: float
    harmonic_fit_rmse: float
    harmonic_alignment_clean_rmse: float
    harmonic_improves_alpha: int
    pca_geometry_mae: float
    pca_weight_mae: float
    pca_alpha_error: float
    pca_fit_rmse: float
    pca_alignment_clean_rmse: float
    pca_improves_alpha: int
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

def aggregate(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        condition = str(regime["name"])
        for alpha_bin in ALPHA_STRENGTH_BIN_LABELS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                subset = [
                    row
                    for row in rows
                    if row.condition == condition
                    and row.alpha_strength_bin == alpha_bin
                    and row.geometry_skew_bin == skew_bin
                ]
                if not subset:
                    continue

                def mean(attr: str) -> float:
                    return float(np.mean([getattr(row, attr) for row in subset]))

                baseline_alpha_mean = mean("baseline_alpha_error")
                oracle_alpha_mean = mean("oracle_alpha_error")
                harmonic_alpha_mean = mean("harmonic_alpha_error")
                pca_alpha_mean = mean("pca_alpha_error")
                oracle_headroom_mean = baseline_alpha_mean - oracle_alpha_mean

                if oracle_headroom_mean > 1.0e-6:
                    harmonic_fraction = float((baseline_alpha_mean - harmonic_alpha_mean) / oracle_headroom_mean)
                    pca_fraction = float((baseline_alpha_mean - pca_alpha_mean) / oracle_headroom_mean)
                else:
                    harmonic_fraction = float("nan")
                    pca_fraction = float("nan")

                summary.append(
                    {
                        "condition": condition,
                        "alpha_strength_bin": alpha_bin,
                        "geometry_skew_bin": skew_bin,
                        "count": len(subset),
                        "baseline_alpha_error_mean": baseline_alpha_mean,
                        "oracle_alpha_error_mean": oracle_alpha_mean,
                        "harmonic_alpha_error_mean": harmonic_alpha_mean,
                        "pca_alpha_error_mean": pca_alpha_mean,
                        "harmonic_alignment_clean_rmse_mean": mean("harmonic_alignment_clean_rmse"),
                        "pca_alignment_clean_rmse_mean": mean("pca_alignment_clean_rmse"),
                        "oracle_headroom_mean": oracle_headroom_mean,
                        "harmonic_fraction_of_oracle_gain_mean": harmonic_fraction,
                        "pca_fraction_of_oracle_gain_mean": pca_fraction,
                        "harmonic_improves_alpha_fraction": mean("harmonic_improves_alpha"),
                        "pca_improves_alpha_fraction": mean("pca_improves_alpha"),
                    }
                )
    return summary

def grid_matrix(
    summary_rows: list[dict[str, float | str]],
    condition: str,
    metric: str,
) -> np.ndarray:
    matrix = np.full((len(ALPHA_STRENGTH_BIN_LABELS), len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
    for row in summary_rows:
        if row["condition"] != condition:
            continue
        i = ALPHA_STRENGTH_BIN_LABELS.index(str(row["alpha_strength_bin"]))
        j = GEOMETRY_SKEW_BIN_LABELS.index(str(row["geometry_skew_bin"]))
        matrix[i, j] = float(row[metric])
    return matrix

def plot_capture_maps(path: str, summary_rows: list[dict[str, float | str]], metric: str, title: str, cmap: str, cbar_label: str) -> None:
    conditions = [str(regime["name"]) for regime in OBSERVATION_REGIMES]
    fig, axes = plt.subplots(2, len(conditions), figsize=(18.0, 7.4), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.98, wspace=0.28, hspace=0.34)

    for row_idx, method_prefix in enumerate(["harmonic", "pca"]):
        for col_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            matrix = grid_matrix(summary_rows, condition, f"{method_prefix}_{metric}")
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                annot=True,
                fmt=".2f",
                xticklabels=GEOMETRY_SKEW_BIN_LABELS,
                yticklabels=ALPHA_STRENGTH_BIN_LABELS,
                cbar=(col_idx == len(conditions) - 1),
                cbar_kws={"label": cbar_label} if col_idx == len(conditions) - 1 else None,
                vmin=(-0.5 if "fraction_of_oracle_gain" in metric else (0.0 if "improves_alpha" in metric else None)),
                vmax=(1.0 if "fraction_of_oracle_gain" in metric or "improves_alpha" in metric else None),
            )
            ax.set_title(f"{condition}\n{method_prefix}")
            if col_idx == 0:
                ax.set_ylabel("anisotropy strength")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("geometry skew |t| bin")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    rng = np.random.default_rng(20260324)
    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    full_mask = np.ones(SIGNATURE_ANGLE_COUNT, dtype=bool)

    harmonic_bank = build_aligned_bank(bank_signatures, harmonic_alignment_score)
    pca_bank = build_aligned_bank(bank_signatures, principal_axis_score)

    audits = {
        "harmonic_invariance": audit_alignment_invariance(bank_signatures, harmonic_alignment_score, rng),
        "pca_invariance": audit_alignment_invariance(bank_signatures, principal_axis_score, rng),
        "harmonic_clean_recovery": audit_clean_recovery(bank_params, bank_signatures, harmonic_bank, harmonic_alignment_score, rng),
        "pca_clean_recovery": audit_clean_recovery(bank_params, bank_signatures, pca_bank, principal_axis_score, rng),
    }

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(MAP_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            baseline_params, baseline_sig, _ = nearest_neighbor_pose_free(
                observed_signature,
                mask,
                shifted_bank,
                bank_params,
            )
            baseline_geom, baseline_weight, baseline_alpha = symmetry_aware_errors(true_params, baseline_params)
            baseline_fit_rmse = rmse(baseline_sig, rotated_signature)

            harmonic_observed, harmonic_mask, harmonic_shift, _ = align_by_shift_search(
                observed_signature,
                mask,
                harmonic_alignment_score,
            )
            harmonic_true_clean = np.roll(rotated_signature, -harmonic_shift)
            harmonic_params, harmonic_sig = nearest_neighbor_aligned(harmonic_observed, harmonic_mask, harmonic_bank, bank_params)
            harmonic_geom, harmonic_weight, harmonic_alpha = symmetry_aware_errors(true_params, harmonic_params)
            harmonic_true_ref, _, _, _ = align_by_shift_search(rotated_signature, full_mask, harmonic_alignment_score)
            harmonic_fit_rmse = rmse(harmonic_sig, harmonic_true_ref)
            harmonic_alignment_clean_rmse = rmse(harmonic_true_clean, clean_signature)

            pca_observed, pca_mask, pca_shift, _ = align_by_shift_search(
                observed_signature,
                mask,
                principal_axis_score,
            )
            pca_true_clean = np.roll(rotated_signature, -pca_shift)
            pca_params, pca_sig = nearest_neighbor_aligned(pca_observed, pca_mask, pca_bank, bank_params)
            pca_geom, pca_weight, pca_alpha = symmetry_aware_errors(true_params, pca_params)
            pca_true_ref, _, _, _ = align_by_shift_search(rotated_signature, full_mask, principal_axis_score)
            pca_fit_rmse = rmse(pca_sig, pca_true_ref)
            pca_alignment_clean_rmse = rmse(pca_true_clean, clean_signature)

            oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
            oracle_params, oracle_sig = nearest_neighbor_aligned(oracle_observed, oracle_mask, bank_signatures, bank_params)
            oracle_geom, oracle_weight, oracle_alpha = symmetry_aware_errors(true_params, oracle_params)
            oracle_fit_rmse = rmse(oracle_sig, clean_signature)

            strength = alpha_strength(float(true_params[5]))
            skew = geometry_skew_from_t(float(true_params[1]))

            rows.append(
                TrialRow(
                    condition=str(regime["name"]),
                    trial=trial,
                    true_alpha=float(true_params[5]),
                    alpha_strength=float(strength),
                    alpha_strength_bin=assign_bin(strength, ALPHA_STRENGTH_BIN_EDGES, ALPHA_STRENGTH_BIN_LABELS),
                    true_t=float(true_params[1]),
                    geometry_skew=float(skew),
                    geometry_skew_bin=assign_bin(skew, GEOMETRY_SKEW_BIN_EDGES, GEOMETRY_SKEW_BIN_LABELS),
                    baseline_geometry_mae=float(baseline_geom),
                    baseline_weight_mae=float(baseline_weight),
                    baseline_alpha_error=float(baseline_alpha),
                    baseline_fit_rmse=float(baseline_fit_rmse),
                    harmonic_geometry_mae=float(harmonic_geom),
                    harmonic_weight_mae=float(harmonic_weight),
                    harmonic_alpha_error=float(harmonic_alpha),
                    harmonic_fit_rmse=float(harmonic_fit_rmse),
                    harmonic_alignment_clean_rmse=float(harmonic_alignment_clean_rmse),
                    harmonic_improves_alpha=int(harmonic_alpha < baseline_alpha),
                    pca_geometry_mae=float(pca_geom),
                    pca_weight_mae=float(pca_weight),
                    pca_alpha_error=float(pca_alpha),
                    pca_fit_rmse=float(pca_fit_rmse),
                    pca_alignment_clean_rmse=float(pca_alignment_clean_rmse),
                    pca_improves_alpha=int(pca_alpha < baseline_alpha),
                    oracle_alpha_error=float(oracle_alpha),
                    oracle_fit_rmse=float(oracle_fit_rmse),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = aggregate(rows)

    write_csv(os.path.join(OUTPUT_DIR, "alignment_failure_map_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "alignment_failure_map_summary.csv"), summary_rows)

    plot_capture_maps(
        os.path.join(FIGURE_DIR, "alignment_failure_map_capture.png"),
        summary_rows,
        "fraction_of_oracle_gain_mean",
        "Alignment Failure Map A: Fraction Of Oracle Alpha Gain Captured",
        "coolwarm",
        "fraction of oracle alpha gain",
    )
    plot_capture_maps(
        os.path.join(FIGURE_DIR, "alignment_failure_map_alignment_rmse.png"),
        summary_rows,
        "alignment_clean_rmse_mean",
        "Alignment Failure Map B: Clean Alignment Error From Observation-Derived Shift",
        "mako",
        "mean clean alignment RMSE",
    )
    plot_capture_maps(
        os.path.join(FIGURE_DIR, "alignment_failure_map_improvement_rate.png"),
        summary_rows,
        "improves_alpha_fraction",
        "Alignment Failure Map C: How Often The Lock Beats The Baseline On Alpha",
        "viridis",
        "fraction of trials with alpha improvement",
    )

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "map_trials_per_regime": MAP_TRIALS_PER_REGIME,
        "audits": audits,
        "alpha_strength_bin_edges": ALPHA_STRENGTH_BIN_EDGES[:-1],
        "geometry_skew_bin_edges": GEOMETRY_SKEW_BIN_EDGES[:-1],
        "largest_harmonic_fraction_of_oracle_gain": float(
            max(float(row["harmonic_fraction_of_oracle_gain_mean"]) for row in summary_rows if np.isfinite(float(row["harmonic_fraction_of_oracle_gain_mean"])))
        ),
        "smallest_harmonic_fraction_of_oracle_gain": float(
            min(float(row["harmonic_fraction_of_oracle_gain_mean"]) for row in summary_rows if np.isfinite(float(row["harmonic_fraction_of_oracle_gain_mean"])))
        ),
        "largest_pca_fraction_of_oracle_gain": float(
            max(float(row["pca_fraction_of_oracle_gain_mean"]) for row in summary_rows if np.isfinite(float(row["pca_fraction_of_oracle_gain_mean"])))
        ),
        "smallest_pca_fraction_of_oracle_gain": float(
            min(float(row["pca_fraction_of_oracle_gain_mean"]) for row in summary_rows if np.isfinite(float(row["pca_fraction_of_oracle_gain_mean"])))
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "alignment_failure_map_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))

if __name__ == "__main__":
    main()
