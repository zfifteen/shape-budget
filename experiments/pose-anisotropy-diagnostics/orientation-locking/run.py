"""
Post-roadmap extension: orientation-locking experiment for pose-free anisotropic inversion.

This experiment tests whether observation-only symmetry-breaking pre-alignment
can recover alpha much more than geometry in the pose-free weighted anisotropic
inverse setting.
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

from run_pose_free_weighted_anisotropic_inverse_experiment import nearest_neighbor_pose_free
from run_pose_free_weighted_inverse_experiment import build_shift_stack, observe_pose_free_signature
from run_weighted_anisotropic_inverse_experiment import (
    OUTPUT_DIR as CANONICAL_ANISO_OUTPUT_DIR,
    REFERENCE_BANK_SIZE,
    TEST_TRIALS_PER_REGIME,
    anisotropic_forward_signature,
    build_reference_bank,
    sample_anisotropic_parameters,
    symmetry_aware_errors,
)
from run_weighted_multisource_inverse_experiment import OBSERVATION_REGIMES, SIGNATURE_ANGLE_COUNT, write_csv


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


HARMONIC_TIEBREAK_WEIGHT = 0.25
PCA_TIEBREAK_WEIGHT = 0.12
AUDIT_ROTATION_CASES = 30
AUDIT_RECOVERY_CASES = 30


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
    baseline_geometry_mae: float
    baseline_weight_mae: float
    baseline_alpha_error: float
    baseline_fit_rmse: float
    harmonic_geometry_mae: float
    harmonic_weight_mae: float
    harmonic_alpha_error: float
    harmonic_fit_rmse: float
    harmonic_alignment_shift: int
    pca_geometry_mae: float
    pca_weight_mae: float
    pca_alpha_error: float
    pca_fit_rmse: float
    pca_alignment_shift: int


def load_canonical_anisotropic_summary() -> dict[str, dict[str, float]]:
    path = os.path.join(CANONICAL_ANISO_OUTPUT_DIR, "weighted_anisotropic_inverse_summary.json")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {item["condition"]: item for item in data["by_condition"]}


def masked_fourier_coeff(signature: np.ndarray, mask: np.ndarray, order: int) -> complex:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.0 + 0.0j
    angles = 2.0 * math.pi * idx / len(signature)
    values = signature[idx]
    return complex(np.mean(values * np.exp(-1j * order * angles)))


def harmonic_alignment_score(signature: np.ndarray, mask: np.ndarray) -> float:
    c2 = masked_fourier_coeff(signature, mask, 2)
    c3 = masked_fourier_coeff(signature, mask, 3)
    return float(np.real(c2) + HARMONIC_TIEBREAK_WEIGHT * np.real(c3))


def principal_axis_score(signature: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return float("-inf")

    angles = 2.0 * math.pi * idx / len(signature)
    radii = signature[idx]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    energy = float(np.mean(x * x + y * y))
    if energy <= 1.0e-12:
        return float("-inf")

    axis_anisotropy = float(np.mean(x * x - y * y) / energy)
    x_std = float(np.sqrt(np.mean(x * x)))
    if x_std <= 1.0e-12:
        skew_x = 0.0
    else:
        skew_x = float(np.mean(x**3) / (x_std**3))
    return axis_anisotropy + PCA_TIEBREAK_WEIGHT * skew_x


def align_by_shift_search(
    signature: np.ndarray,
    mask: np.ndarray,
    scorer,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    best_score = float("-inf")
    best_shift = 0
    best_signature = signature
    best_mask = mask

    for shift in range(len(signature)):
        candidate_signature = np.roll(signature, -shift)
        candidate_mask = np.roll(mask, -shift)
        score = scorer(candidate_signature, candidate_mask)
        if score > best_score + 1.0e-15:
            best_score = float(score)
            best_shift = int(shift)
            best_signature = candidate_signature
            best_mask = candidate_mask

    return best_signature, best_mask, best_shift, best_score


def nearest_neighbor_aligned(
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


def build_aligned_bank(
    bank_signatures: np.ndarray,
    scorer,
) -> np.ndarray:
    full_mask = np.ones(bank_signatures.shape[1], dtype=bool)
    aligned = []
    for signature in bank_signatures:
        aligned_signature, _, _, _ = align_by_shift_search(signature, full_mask, scorer)
        aligned.append(aligned_signature)
    return np.array(aligned)


def audit_alignment_invariance(
    bank_signatures: np.ndarray,
    scorer,
    rng: np.random.Generator,
) -> dict[str, float]:
    full_mask = np.ones(bank_signatures.shape[1], dtype=bool)
    max_rmse = 0.0
    for _ in range(AUDIT_ROTATION_CASES):
        idx = int(rng.integers(0, len(bank_signatures)))
        base = bank_signatures[idx]
        shift = int(rng.integers(0, len(base)))
        rotated = np.roll(base, shift)
        aligned_base, _, _, _ = align_by_shift_search(base, full_mask, scorer)
        aligned_rot, _, _, _ = align_by_shift_search(rotated, full_mask, scorer)
        max_rmse = max(max_rmse, rmse(aligned_base, aligned_rot))
    return {"max_aligned_rotation_rmse": float(max_rmse)}


def audit_clean_recovery(
    bank_params: list[tuple[float, float, float, float, float, float]],
    bank_signatures: np.ndarray,
    aligned_bank: np.ndarray,
    scorer,
    rng: np.random.Generator,
) -> dict[str, float]:
    exact_count = 0
    max_fit_rmse = 0.0
    full_clean = OBSERVATION_REGIMES[0]
    for _ in range(AUDIT_RECOVERY_CASES):
        idx = int(rng.integers(0, len(bank_signatures)))
        clean = bank_signatures[idx]
        rotated, observed, mask, _ = observe_pose_free_signature(clean, full_clean, rng)
        aligned_observed, aligned_mask, _, _ = align_by_shift_search(observed, mask, scorer)
        pred_params, pred_sig = nearest_neighbor_aligned(aligned_observed, aligned_mask, aligned_bank, bank_params)
        if pred_params == bank_params[idx]:
            exact_count += 1
        aligned_true, _, _, _ = align_by_shift_search(rotated, mask, scorer)
        max_fit_rmse = max(max_fit_rmse, rmse(pred_sig, aligned_true))
    return {
        "audit_cases": AUDIT_RECOVERY_CASES,
        "exact_recovery_fraction": float(exact_count / AUDIT_RECOVERY_CASES),
        "max_aligned_fit_rmse": float(max_fit_rmse),
    }


def aggregate(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        summary.append(
            {
                "condition": name,
                "baseline_geometry_mae_mean": mean("baseline_geometry_mae"),
                "baseline_weight_mae_mean": mean("baseline_weight_mae"),
                "baseline_alpha_error_mean": mean("baseline_alpha_error"),
                "baseline_fit_rmse_mean": mean("baseline_fit_rmse"),
                "harmonic_geometry_mae_mean": mean("harmonic_geometry_mae"),
                "harmonic_weight_mae_mean": mean("harmonic_weight_mae"),
                "harmonic_alpha_error_mean": mean("harmonic_alpha_error"),
                "harmonic_fit_rmse_mean": mean("harmonic_fit_rmse"),
                "pca_geometry_mae_mean": mean("pca_geometry_mae"),
                "pca_weight_mae_mean": mean("pca_weight_mae"),
                "pca_alpha_error_mean": mean("pca_alpha_error"),
                "pca_fit_rmse_mean": mean("pca_fit_rmse"),
                "harmonic_alpha_improvement_vs_baseline": float(
                    mean("baseline_alpha_error") / max(mean("harmonic_alpha_error"), 1.0e-12)
                ),
                "pca_alpha_improvement_vs_baseline": float(
                    mean("baseline_alpha_error") / max(mean("pca_alpha_error"), 1.0e-12)
                ),
                "harmonic_geometry_ratio_vs_baseline": float(
                    mean("harmonic_geometry_mae") / max(mean("baseline_geometry_mae"), 1.0e-12)
                ),
                "pca_geometry_ratio_vs_baseline": float(
                    mean("pca_geometry_mae") / max(mean("baseline_geometry_mae"), 1.0e-12)
                ),
            }
        )
    return summary


def compare_to_canonical(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    canonical = load_canonical_anisotropic_summary()
    rows: list[dict[str, float | str]] = []
    for row in summary_rows:
        name = str(row["condition"])
        base = canonical[name]
        rows.append(
            {
                "condition": name,
                "baseline_alpha_penalty_vs_canonical": float(row["baseline_alpha_error_mean"]) / float(base["alpha_mae_mean"]),
                "harmonic_alpha_penalty_vs_canonical": float(row["harmonic_alpha_error_mean"]) / float(base["alpha_mae_mean"]),
                "pca_alpha_penalty_vs_canonical": float(row["pca_alpha_error_mean"]) / float(base["alpha_mae_mean"]),
                "baseline_geometry_penalty_vs_canonical": float(row["baseline_geometry_mae_mean"]) / float(base["geometry_mae_mean"]),
                "harmonic_geometry_penalty_vs_canonical": float(row["harmonic_geometry_mae_mean"]) / float(base["geometry_mae_mean"]),
                "pca_geometry_penalty_vs_canonical": float(row["pca_geometry_mae_mean"]) / float(base["geometry_mae_mean"]),
            }
        )
    return rows


def plot_alpha_and_geometry(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    x = np.arange(len(conditions))
    width = 0.25

    baseline_alpha = np.array([float(item["baseline_alpha_error_mean"]) for item in summary_rows])
    harmonic_alpha = np.array([float(item["harmonic_alpha_error_mean"]) for item in summary_rows])
    pca_alpha = np.array([float(item["pca_alpha_error_mean"]) for item in summary_rows])

    baseline_geom = np.array([float(item["baseline_geometry_mae_mean"]) for item in summary_rows])
    harmonic_geom = np.array([float(item["harmonic_geometry_mae_mean"]) for item in summary_rows])
    pca_geom = np.array([float(item["pca_geometry_mae_mean"]) for item in summary_rows])

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.98, hspace=0.34)

    for ax, baseline, harmonic, pca, ylabel, title in [
        (axes[0], baseline_alpha, harmonic_alpha, pca_alpha, "mean alpha absolute error", "Alpha recovery under orientation locking"),
        (axes[1], baseline_geom, harmonic_geom, pca_geom, "mean geometry MAE", "Geometry recovery under orientation locking"),
    ]:
        ax.bar(x - width, baseline, width=width, color="#e76f51", label="shift-aware baseline")
        ax.bar(x, harmonic, width=width, color="#2a9d8f", label="harmonic lock")
        ax.bar(x + width, pca, width=width, color="#1d3557", label="principal-axis lock")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("Orientation Locking A: Alpha Versus Geometry", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_penalties(path: str, penalty_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in penalty_rows]
    x = np.arange(len(conditions))

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.08, right=0.98, wspace=0.28)

    for ax, baseline_key, harmonic_key, pca_key, ylabel, title in [
        (
            axes[0],
            "baseline_alpha_penalty_vs_canonical",
            "harmonic_alpha_penalty_vs_canonical",
            "pca_alpha_penalty_vs_canonical",
            "method / canonical alpha MAE",
            "How much alpha penalty remains after locking",
        ),
        (
            axes[1],
            "baseline_geometry_penalty_vs_canonical",
            "harmonic_geometry_penalty_vs_canonical",
            "pca_geometry_penalty_vs_canonical",
            "method / canonical geometry MAE",
            "Geometry penalty relative to canonical",
        ),
    ]:
        ax.plot(x, [float(item[baseline_key]) for item in penalty_rows], color="#e76f51", lw=2.2, marker="o", label="shift-aware baseline")
        ax.plot(x, [float(item[harmonic_key]) for item in penalty_rows], color="#2a9d8f", lw=2.2, marker="s", label="harmonic lock")
        ax.plot(x, [float(item[pca_key]) for item in penalty_rows], color="#1d3557", lw=2.2, marker="^", label="principal-axis lock")
        ax.axhline(1.0, color="#444444", linestyle="--", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("Orientation Locking B: Penalty Relative To Canonical", fontsize=16, fontweight="bold", y=0.95)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)
    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)

    full_mask = np.ones(SIGNATURE_ANGLE_COUNT, dtype=bool)
    harmonic_bank = build_aligned_bank(bank_signatures, harmonic_alignment_score)
    pca_bank = build_aligned_bank(bank_signatures, principal_axis_score)

    harmonic_invariance = audit_alignment_invariance(bank_signatures, harmonic_alignment_score, rng)
    pca_invariance = audit_alignment_invariance(bank_signatures, principal_axis_score, rng)
    harmonic_recovery = audit_clean_recovery(bank_params, bank_signatures, harmonic_bank, harmonic_alignment_score, rng)
    pca_recovery = audit_clean_recovery(bank_params, bank_signatures, pca_bank, principal_axis_score, rng)

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, rng)

            baseline_params, baseline_sig, baseline_shift = nearest_neighbor_pose_free(
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
            harmonic_true, _, _, _ = align_by_shift_search(rotated_signature, full_mask, harmonic_alignment_score)
            harmonic_params, harmonic_sig = nearest_neighbor_aligned(harmonic_observed, harmonic_mask, harmonic_bank, bank_params)
            harmonic_geom, harmonic_weight, harmonic_alpha = symmetry_aware_errors(true_params, harmonic_params)
            harmonic_fit_rmse = rmse(harmonic_sig, harmonic_true)

            pca_observed, pca_mask, pca_shift, _ = align_by_shift_search(
                observed_signature,
                mask,
                principal_axis_score,
            )
            pca_true, _, _, _ = align_by_shift_search(rotated_signature, full_mask, principal_axis_score)
            pca_params, pca_sig = nearest_neighbor_aligned(pca_observed, pca_mask, pca_bank, bank_params)
            pca_geom, pca_weight, pca_alpha = symmetry_aware_errors(true_params, pca_params)
            pca_fit_rmse = rmse(pca_sig, pca_true)

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
                    baseline_geometry_mae=float(baseline_geom),
                    baseline_weight_mae=float(baseline_weight),
                    baseline_alpha_error=float(baseline_alpha),
                    baseline_fit_rmse=float(baseline_fit_rmse),
                    harmonic_geometry_mae=float(harmonic_geom),
                    harmonic_weight_mae=float(harmonic_weight),
                    harmonic_alpha_error=float(harmonic_alpha),
                    harmonic_fit_rmse=float(harmonic_fit_rmse),
                    harmonic_alignment_shift=int(harmonic_shift),
                    pca_geometry_mae=float(pca_geom),
                    pca_weight_mae=float(pca_weight),
                    pca_alpha_error=float(pca_alpha),
                    pca_fit_rmse=float(pca_fit_rmse),
                    pca_alignment_shift=int(pca_shift),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = aggregate(rows)
    penalty_rows = compare_to_canonical(summary_rows)

    write_csv(os.path.join(OUTPUT_DIR, "orientation_locking_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "orientation_locking_summary.csv"), summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "orientation_locking_penalties.csv"), penalty_rows)

    plot_alpha_and_geometry(os.path.join(FIGURE_DIR, "orientation_locking_alpha_geometry.png"), summary_rows)
    plot_penalties(os.path.join(FIGURE_DIR, "orientation_locking_penalties.png"), penalty_rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "harmonic_invariance": harmonic_invariance,
        "pca_invariance": pca_invariance,
        "harmonic_clean_recovery": harmonic_recovery,
        "pca_clean_recovery": pca_recovery,
        "largest_harmonic_alpha_improvement_vs_baseline": float(
            max(item["harmonic_alpha_improvement_vs_baseline"] for item in summary_rows)
        ),
        "smallest_harmonic_alpha_improvement_vs_baseline": float(
            min(item["harmonic_alpha_improvement_vs_baseline"] for item in summary_rows)
        ),
        "largest_pca_alpha_improvement_vs_baseline": float(
            max(item["pca_alpha_improvement_vs_baseline"] for item in summary_rows)
        ),
        "smallest_pca_alpha_improvement_vs_baseline": float(
            min(item["pca_alpha_improvement_vs_baseline"] for item in summary_rows)
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "orientation_locking_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "by_condition": summary_rows,
                "penalties_vs_canonical": penalty_rows,
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "summary": summary,
                "by_condition": summary_rows,
                "penalties_vs_canonical": penalty_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
