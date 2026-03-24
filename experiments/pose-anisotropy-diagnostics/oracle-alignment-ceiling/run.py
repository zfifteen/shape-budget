"""
Post-roadmap extension: oracle alignment ceiling experiment for pose-free
anisotropic inversion.

This experiment asks how much of the pose-free alpha penalty disappears if the
true rotation is given to the inverse, while everything else stays the same.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

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

REFERENCE_BANK_SIZE, TEST_TRIALS_PER_REGIME, anisotropic_forward_signature, build_reference_bank, sample_anisotropic_parameters, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "REFERENCE_BANK_SIZE",
    "TEST_TRIALS_PER_REGIME",
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

ORACLE_AUDIT_CASES = 30

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
    pca_geometry_mae: float
    pca_weight_mae: float
    pca_alpha_error: float
    pca_fit_rmse: float
    oracle_geometry_mae: float
    oracle_weight_mae: float
    oracle_alpha_error: float
    oracle_fit_rmse: float

def oracle_align_observation(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    true_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    return np.roll(observed_signature, -true_shift), np.roll(mask, -true_shift)

def audit_oracle_identity(
    bank_signatures: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    max_oracle_identity_rmse = 0.0
    full_mask = np.ones(SIGNATURE_ANGLE_COUNT, dtype=bool)
    regime = OBSERVATION_REGIMES[0]
    for _ in range(ORACLE_AUDIT_CASES):
        idx = int(rng.integers(0, len(bank_signatures)))
        clean = bank_signatures[idx]
        rotated, observed, mask, true_shift = observe_pose_free_signature(clean, regime, rng)
        oracle_observed, oracle_mask = oracle_align_observation(observed, mask, true_shift)
        oracle_true, oracle_true_mask = oracle_align_observation(rotated, full_mask, true_shift)
        max_oracle_identity_rmse = max(max_oracle_identity_rmse, rmse(oracle_observed[oracle_mask], oracle_true[oracle_true_mask]))
    return {
        "audit_cases": ORACLE_AUDIT_CASES,
        "max_oracle_identity_rmse": float(max_oracle_identity_rmse),
    }

def audit_oracle_clean_recovery(
    bank_params: list[tuple[float, float, float, float, float, float]],
    bank_signatures: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    exact_count = 0
    max_fit_rmse = 0.0
    regime = OBSERVATION_REGIMES[0]
    for _ in range(ORACLE_AUDIT_CASES):
        idx = int(rng.integers(0, len(bank_signatures)))
        clean = bank_signatures[idx]
        rotated, observed, mask, true_shift = observe_pose_free_signature(clean, regime, rng)
        oracle_observed, oracle_mask = oracle_align_observation(observed, mask, true_shift)
        pred_params, pred_sig = nearest_neighbor_aligned(oracle_observed, oracle_mask, bank_signatures, bank_params)
        if pred_params == bank_params[idx]:
            exact_count += 1
        max_fit_rmse = max(max_fit_rmse, rmse(pred_sig, clean))
    return {
        "audit_cases": ORACLE_AUDIT_CASES,
        "exact_recovery_fraction": float(exact_count / ORACLE_AUDIT_CASES),
        "max_fit_rmse": float(max_fit_rmse),
    }

def aggregate(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
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
                "oracle_geometry_mae_mean": mean("oracle_geometry_mae"),
                "oracle_weight_mae_mean": mean("oracle_weight_mae"),
                "oracle_alpha_error_mean": mean("oracle_alpha_error"),
                "oracle_fit_rmse_mean": mean("oracle_fit_rmse"),
                "oracle_alpha_improvement_vs_baseline": float(
                    mean("baseline_alpha_error") / max(mean("oracle_alpha_error"), 1.0e-12)
                ),
                "oracle_geometry_ratio_vs_baseline": float(
                    mean("oracle_geometry_mae") / max(mean("baseline_geometry_mae"), 1.0e-12)
                ),
                "harmonic_fraction_of_oracle_alpha_gain": float(
                    (mean("baseline_alpha_error") - mean("harmonic_alpha_error"))
                    / max(mean("baseline_alpha_error") - mean("oracle_alpha_error"), 1.0e-12)
                ),
                "pca_fraction_of_oracle_alpha_gain": float(
                    (mean("baseline_alpha_error") - mean("pca_alpha_error"))
                    / max(mean("baseline_alpha_error") - mean("oracle_alpha_error"), 1.0e-12)
                ),
            }
        )
    return summary

def plot_alpha_methods(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    x = np.arange(len(conditions))
    width = 0.20

    baseline = np.array([float(item["baseline_alpha_error_mean"]) for item in summary_rows])
    harmonic = np.array([float(item["harmonic_alpha_error_mean"]) for item in summary_rows])
    pca = np.array([float(item["pca_alpha_error_mean"]) for item in summary_rows])
    oracle = np.array([float(item["oracle_alpha_error_mean"]) for item in summary_rows])

    fig, ax = plt.subplots(figsize=(13.8, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.98)

    ax.bar(x - 1.5 * width, baseline, width=width, color="#e76f51", label="shift-aware baseline")
    ax.bar(x - 0.5 * width, harmonic, width=width, color="#2a9d8f", label="harmonic lock")
    ax.bar(x + 0.5 * width, pca, width=width, color="#1d3557", label="principal-axis lock")
    ax.bar(x + 1.5 * width, oracle, width=width, color="#6a4c93", label="oracle alignment")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Alpha recovery across baseline, practical locks, and oracle alignment")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    fig.suptitle("Oracle Alignment Ceiling A: How Much Alpha Headroom Remains", fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_oracle_gap(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    x = np.arange(len(conditions))

    oracle_improve = np.array([float(item["oracle_alpha_improvement_vs_baseline"]) for item in summary_rows])
    harmonic_fraction = np.array([float(item["harmonic_fraction_of_oracle_alpha_gain"]) for item in summary_rows])
    pca_fraction = np.array([float(item["pca_fraction_of_oracle_alpha_gain"]) for item in summary_rows])

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.20, left=0.08, right=0.98, wspace=0.28)

    axes[0].plot(x, oracle_improve, color="#6a4c93", lw=2.4, marker="o")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=20, ha="right")
    axes[0].set_ylabel("baseline / oracle alpha error")
    axes[0].set_title("Oracle alignment improvement factor")

    axes[1].plot(x, harmonic_fraction, color="#2a9d8f", lw=2.4, marker="s", label="harmonic")
    axes[1].plot(x, pca_fraction, color="#1d3557", lw=2.4, marker="^", label="principal-axis")
    axes[1].axhline(1.0, color="#444444", linestyle="--", lw=1.2)
    axes[1].axhline(0.0, color="#888888", linestyle=":", lw=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[1].set_ylabel("fraction of oracle alpha gain captured")
    axes[1].set_title("How much of the oracle headroom practical locks capture")
    axes[1].legend(loc="upper right", frameon=True)

    fig.suptitle("Oracle Alignment Ceiling B: Oracle Headroom Versus Practical Locks", fontsize=16, fontweight="bold", y=0.95)
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
        "oracle_identity": audit_oracle_identity(bank_signatures, rng),
        "oracle_clean_recovery": audit_oracle_clean_recovery(bank_params, bank_signatures, rng),
    }

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
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

            harmonic_observed, harmonic_mask, _, _ = align_by_shift_search(
                observed_signature,
                mask,
                harmonic_alignment_score,
            )
            harmonic_true, _, _, _ = align_by_shift_search(rotated_signature, full_mask, harmonic_alignment_score)
            harmonic_params, harmonic_sig = nearest_neighbor_aligned(harmonic_observed, harmonic_mask, harmonic_bank, bank_params)
            harmonic_geom, harmonic_weight, harmonic_alpha = symmetry_aware_errors(true_params, harmonic_params)
            harmonic_fit_rmse = rmse(harmonic_sig, harmonic_true)

            pca_observed, pca_mask, _, _ = align_by_shift_search(
                observed_signature,
                mask,
                principal_axis_score,
            )
            pca_true, _, _, _ = align_by_shift_search(rotated_signature, full_mask, principal_axis_score)
            pca_params, pca_sig = nearest_neighbor_aligned(pca_observed, pca_mask, pca_bank, bank_params)
            pca_geom, pca_weight, pca_alpha = symmetry_aware_errors(true_params, pca_params)
            pca_fit_rmse = rmse(pca_sig, pca_true)

            oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
            oracle_params, oracle_sig = nearest_neighbor_aligned(oracle_observed, oracle_mask, bank_signatures, bank_params)
            oracle_geom, oracle_weight, oracle_alpha = symmetry_aware_errors(true_params, oracle_params)
            oracle_fit_rmse = rmse(oracle_sig, clean_signature)

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
                    pca_geometry_mae=float(pca_geom),
                    pca_weight_mae=float(pca_weight),
                    pca_alpha_error=float(pca_alpha),
                    pca_fit_rmse=float(pca_fit_rmse),
                    oracle_geometry_mae=float(oracle_geom),
                    oracle_weight_mae=float(oracle_weight),
                    oracle_alpha_error=float(oracle_alpha),
                    oracle_fit_rmse=float(oracle_fit_rmse),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = aggregate(rows)

    write_csv(os.path.join(OUTPUT_DIR, "oracle_alignment_ceiling_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "oracle_alignment_ceiling_summary.csv"), summary_rows)

    plot_alpha_methods(os.path.join(FIGURE_DIR, "oracle_alignment_ceiling_alpha_methods.png"), summary_rows)
    plot_oracle_gap(os.path.join(FIGURE_DIR, "oracle_alignment_ceiling_gap.png"), summary_rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "audits": audits,
        "largest_oracle_alpha_improvement_vs_baseline": float(
            max(item["oracle_alpha_improvement_vs_baseline"] for item in summary_rows)
        ),
        "smallest_oracle_alpha_improvement_vs_baseline": float(
            min(item["oracle_alpha_improvement_vs_baseline"] for item in summary_rows)
        ),
        "largest_harmonic_fraction_of_oracle_alpha_gain": float(
            max(item["harmonic_fraction_of_oracle_alpha_gain"] for item in summary_rows)
        ),
        "smallest_harmonic_fraction_of_oracle_alpha_gain": float(
            min(item["harmonic_fraction_of_oracle_alpha_gain"] for item in summary_rows)
        ),
        "largest_pca_fraction_of_oracle_alpha_gain": float(
            max(item["pca_fraction_of_oracle_alpha_gain"] for item in summary_rows)
        ),
        "smallest_pca_fraction_of_oracle_alpha_gain": float(
            min(item["pca_fraction_of_oracle_alpha_gain"] for item in summary_rows)
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "oracle_alignment_ceiling_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))

if __name__ == "__main__":
    main()
