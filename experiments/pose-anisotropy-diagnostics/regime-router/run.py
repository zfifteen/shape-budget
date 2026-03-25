"""
Post-roadmap extension: regime-router experiment for the pose-free anisotropic
solver challenge.

This experiment tests whether the remaining sparse moderate solver challenge can be
handled better by routing each trial to one of two refinement policies:

1. fixed-family alpha-only search
2. geometry-plus-alpha family switching

The routing signals compared here are:

- a raw anisotropy-to-skew ratio computed from the top seed
- a support-aware orbit-alias index computed from the top seed plus the mask
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

GEOMETRY_SKEW_BIN_LABELS, candidate_conditioned_search, sample_conditioned_parameters, top_k_indices = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "GEOMETRY_SKEW_BIN_LABELS",
    "candidate_conditioned_search",
    "sample_conditioned_parameters",
    "top_k_indices",
)

family_switching_refine, = load_symbols(
    "run_family_switching_refinement_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/family-switching-refinement/run.py",
    "family_switching_refine",
)

marginalized_candidate_scores, softmin_temperature = load_symbols(
    "run_shift_marginalized_pose_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/shift-marginalized-pose/run.py",
    "marginalized_candidate_scores",
    "softmin_temperature",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

ALPHA_MAX, ALPHA_MIN, GEOMETRY_BOUNDS, REFERENCE_BANK_SIZE, anisotropic_forward_signature, build_reference_bank, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "GEOMETRY_BOUNDS",
    "REFERENCE_BANK_SIZE",
    "anisotropic_forward_signature",
    "build_reference_bank",
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
from dataclasses import asdict, dataclass

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

FOCUS_CONDITIONS = ["sparse_full_noisy", "sparse_partial_high_noise"]
FOCUS_ALPHA_BIN = "moderate"
TOP_K_SEEDS = 3
TRIALS_PER_CELL = 4

ROUTER_AUDIT_CASES = 20
ALPHA_PROBE_STEP = 0.08
T_PROBE_STEP = 0.10
RAW_RATIO_EPS = 0.03
ALIAS_INDEX_EPS = 1.0e-6

@dataclass
class TrialRow:
    condition: str
    geometry_skew_bin: str
    trial_in_cell: int
    true_alpha: float
    true_t: float
    top_seed_rank: int
    top_seed_alpha: float
    top_seed_t: float
    top_seed_alpha_strength: float
    raw_ratio_index: float
    support_alias_index: float
    alpha_orbit_absorbable_change: float
    visible_skew_anchor: float
    conditioned_alpha_error: float
    family_alpha_error: float
    conditioned_geometry_mae: float
    family_geometry_mae: float
    family_better: int
    family_minus_conditioned_alpha_gain: float
    raw_router_choose_family: int
    raw_router_alpha_error: float
    raw_router_correct: int
    support_router_choose_family: int
    support_router_alpha_error: float
    support_router_correct: int
    oracle_router_alpha_error: float

def masked_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))

def orbit_min_rmse(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    shifts = np.stack([np.roll(candidate, shift) for shift in range(len(candidate))], axis=0)
    residual = shifts[:, mask] - reference[mask][None, :]
    mse = np.mean(residual * residual, axis=1)
    return float(np.sqrt(np.min(mse)))

def perturb_values(center: float, step: float, lower: float, upper: float) -> list[float]:
    values: list[float] = []
    if center - step >= lower:
        values.append(float(center - step))
    if center + step <= upper:
        values.append(float(center + step))
    if not values:
        values.append(float(np.clip(center, lower, upper)))
    return values

def alpha_strength(alpha: float) -> float:
    return float(abs(math.log(alpha)))

def support_aware_alias_index(
    seed_params: tuple[float, float, float, float, float, float],
    mask: np.ndarray,
) -> tuple[float, float, float]:
    seed_signature = anisotropic_forward_signature(seed_params)
    rho, t, h, w1, w2, alpha = seed_params

    alpha_absorbable: list[float] = []
    for probe_alpha in perturb_values(float(alpha), ALPHA_PROBE_STEP, ALPHA_MIN, ALPHA_MAX):
        probe_signature = anisotropic_forward_signature((rho, t, h, w1, w2, probe_alpha))
        raw = masked_rmse(seed_signature, probe_signature, mask)
        orbit = orbit_min_rmse(seed_signature, probe_signature, mask)
        alpha_absorbable.append(max(raw - orbit, 0.0))

    skew_anchor: list[float] = []
    for probe_t in perturb_values(float(t), T_PROBE_STEP, GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]):
        probe_signature = anisotropic_forward_signature((rho, probe_t, h, w1, w2, alpha))
        skew_anchor.append(orbit_min_rmse(seed_signature, probe_signature, mask))

    absorbable_change = float(np.mean(alpha_absorbable))
    visible_anchor = float(np.mean(skew_anchor))
    index = absorbable_change / max(visible_anchor, ALIAS_INDEX_EPS)
    return index, absorbable_change, visible_anchor

def audit_alias_index_joint_shift_invariance(
    rng: np.random.Generator,
    bank_params: list[tuple[float, float, float, float, float, float]],
) -> dict[str, float]:
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}
    max_delta = 0.0
    for _ in range(ROUTER_AUDIT_CASES):
        params = bank_params[int(rng.integers(0, len(bank_params)))]
        clean_signature = anisotropic_forward_signature(params)
        regime = regime_map[FOCUS_CONDITIONS[int(rng.integers(0, len(FOCUS_CONDITIONS)))]]
        _, _, mask, _ = observe_pose_free_signature(clean_signature, regime, rng)
        base_index, base_absorbable, base_anchor = support_aware_alias_index(params, mask)
        shift = int(rng.integers(0, len(clean_signature)))
        rotated_params_signature = np.roll(clean_signature, shift)
        rotated_mask = np.roll(mask, shift)

        def shifted_index(signature: np.ndarray, rotated_mask_local: np.ndarray) -> tuple[float, float, float]:
            rho, t, h, w1, w2, alpha = params
            alpha_absorbable: list[float] = []
            for probe_alpha in perturb_values(float(alpha), ALPHA_PROBE_STEP, ALPHA_MIN, ALPHA_MAX):
                probe_signature = np.roll(anisotropic_forward_signature((rho, t, h, w1, w2, probe_alpha)), shift)
                raw = masked_rmse(signature, probe_signature, rotated_mask_local)
                orbit = orbit_min_rmse(signature, probe_signature, rotated_mask_local)
                alpha_absorbable.append(max(raw - orbit, 0.0))

            skew_anchor: list[float] = []
            for probe_t in perturb_values(float(t), T_PROBE_STEP, GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]):
                probe_signature = np.roll(anisotropic_forward_signature((rho, probe_t, h, w1, w2, alpha)), shift)
                skew_anchor.append(orbit_min_rmse(signature, probe_signature, rotated_mask_local))

            absorbable_change = float(np.mean(alpha_absorbable))
            visible_anchor = float(np.mean(skew_anchor))
            return (
                absorbable_change / max(visible_anchor, ALIAS_INDEX_EPS),
                absorbable_change,
                visible_anchor,
            )

        test_index, test_absorbable, test_anchor = shifted_index(rotated_params_signature, rotated_mask)
        max_delta = max(
            max_delta,
            abs(test_index - base_index),
            abs(test_absorbable - base_absorbable),
            abs(test_anchor - base_anchor),
        )
    return {
        "audit_cases": float(ROUTER_AUDIT_CASES),
        "max_joint_shift_delta": float(max_delta),
    }

def audit_leave_one_out_router_logic() -> dict[str, float]:
    toy_values = np.array([0.10, 0.20, 0.90, 1.00], dtype=float)
    toy_rows = [
        {
            "value": toy_values[0],
            "conditioned_alpha_error": 0.20,
            "family_alpha_error": 0.10,
            "family_better": 1,
        },
        {
            "value": toy_values[1],
            "conditioned_alpha_error": 0.30,
            "family_alpha_error": 0.10,
            "family_better": 1,
        },
        {
            "value": toy_values[2],
            "conditioned_alpha_error": 0.10,
            "family_alpha_error": 0.30,
            "family_better": 0,
        },
        {
            "value": toy_values[3],
            "conditioned_alpha_error": 0.10,
            "family_alpha_error": 0.20,
            "family_better": 0,
        },
    ]
    routed = leave_one_out_router(toy_rows, "value")
    return {
        "toy_accuracy": float(np.mean([row["correct"] for row in routed])),
        "toy_mean_routed_alpha_error": float(np.mean([row["alpha_error"] for row in routed])),
    }

def threshold_candidates(values: np.ndarray) -> list[float]:
    unique = np.unique(values)
    if len(unique) == 1:
        return [float(unique[0])]
    candidates = [float(unique[0] - 1.0e-9)]
    for left, right in zip(unique[:-1], unique[1:]):
        candidates.append(float(0.5 * (left + right)))
    candidates.append(float(unique[-1] + 1.0e-9))
    return candidates

def choose_router_rule(train_rows: list[dict[str, float | int]], key: str) -> tuple[float, str]:
    values = np.array([float(row[key]) for row in train_rows], dtype=float)
    best_threshold = float(values[0])
    best_direction = "family_if_low"
    best_error = float("inf")
    best_accuracy = -1.0

    for threshold in threshold_candidates(values):
        for direction in ["family_if_low", "family_if_high"]:
            routed_errors = []
            correct = []
            for row in train_rows:
                value = float(row[key])
                choose_family = int(value <= threshold) if direction == "family_if_low" else int(value > threshold)
                alpha_error = (
                    float(row["family_alpha_error"]) if choose_family else float(row["conditioned_alpha_error"])
                )
                routed_errors.append(alpha_error)
                correct.append(int(choose_family == int(row["family_better"])))

            mean_error = float(np.mean(routed_errors))
            accuracy = float(np.mean(correct))
            if mean_error < best_error - 1.0e-12 or (
                abs(mean_error - best_error) <= 1.0e-12 and accuracy > best_accuracy + 1.0e-12
            ):
                best_threshold = float(threshold)
                best_direction = direction
                best_error = mean_error
                best_accuracy = accuracy

    return best_threshold, best_direction

def apply_router_rule(value: float, threshold: float, direction: str) -> int:
    if direction == "family_if_low":
        return int(value <= threshold)
    return int(value > threshold)

def leave_one_out_router(rows: list[dict[str, float | int]], key: str) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for holdout_idx, row in enumerate(rows):
        train_rows = [item for idx, item in enumerate(rows) if idx != holdout_idx]
        threshold, direction = choose_router_rule(train_rows, key)
        choose_family = apply_router_rule(float(row[key]), threshold, direction)
        alpha_error = float(row["family_alpha_error"]) if choose_family else float(row["conditioned_alpha_error"])
        results.append(
            {
                "choose_family": choose_family,
                "alpha_error": alpha_error,
                "correct": int(choose_family == int(row["family_better"])),
                "threshold": float(threshold),
                "direction_family_if_low": int(direction == "family_if_low"),
            }
        )
    return results

def aggregate(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        subset = [row for row in rows if row.condition == condition]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        summary.append(
            {
                "condition": condition,
                "conditioned_alpha_error_mean": mean("conditioned_alpha_error"),
                "family_alpha_error_mean": mean("family_alpha_error"),
                "raw_router_alpha_error_mean": mean("raw_router_alpha_error"),
                "support_router_alpha_error_mean": mean("support_router_alpha_error"),
                "oracle_router_alpha_error_mean": mean("oracle_router_alpha_error"),
                "raw_router_accuracy_mean": mean("raw_router_correct"),
                "support_router_accuracy_mean": mean("support_router_correct"),
                "family_better_fraction": mean("family_better"),
                "raw_router_choose_family_fraction": mean("raw_router_choose_family"),
                "support_router_choose_family_fraction": mean("support_router_choose_family"),
            }
        )
    return summary

def aggregate_by_cell(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            subset = [row for row in rows if row.condition == condition and row.geometry_skew_bin == skew_bin]
            if not subset:
                continue

            def mean(attr: str) -> float:
                return float(np.mean([getattr(row, attr) for row in subset]))

            summary.append(
                {
                    "condition": condition,
                    "geometry_skew_bin": skew_bin,
                    "count": len(subset),
                    "conditioned_alpha_error_mean": mean("conditioned_alpha_error"),
                    "family_alpha_error_mean": mean("family_alpha_error"),
                    "raw_router_alpha_error_mean": mean("raw_router_alpha_error"),
                    "support_router_alpha_error_mean": mean("support_router_alpha_error"),
                    "oracle_router_alpha_error_mean": mean("oracle_router_alpha_error"),
                    "raw_router_accuracy_mean": mean("raw_router_correct"),
                    "support_router_accuracy_mean": mean("support_router_correct"),
                    "family_better_fraction": mean("family_better"),
                    "raw_ratio_index_mean": mean("raw_ratio_index"),
                    "support_alias_index_mean": mean("support_alias_index"),
                }
            )
    return summary

def plot_scatter(path: str, rows: list[TrialRow]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.08, right=0.98, wspace=0.26)

    x_raw = np.array([row.raw_ratio_index for row in rows])
    x_support = np.array([row.support_alias_index for row in rows])
    y = np.array([row.family_minus_conditioned_alpha_gain for row in rows])
    colors = ["#1d3557" if row.condition == "sparse_full_noisy" else "#e76f51" for row in rows]
    markers = {"low_skew": "o", "mid_skew": "s", "high_skew": "^"}

    for ax, xvals, title, xlabel in [
        (axes[0], x_raw, "Raw anisotropy-to-skew ratio", "raw ratio index"),
        (axes[1], x_support, "Support-aware orbit-alias index", "support-aware alias index"),
    ]:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            idx = [i for i, row in enumerate(rows) if row.geometry_skew_bin == skew_bin]
            ax.scatter(
                xvals[idx],
                y[idx],
                c=[colors[i] for i in idx],
                marker=markers[skew_bin],
                s=58,
                alpha=0.86,
                edgecolors="none",
                label=skew_bin if ax is axes[0] else None,
            )
        ax.axhline(0.0, color="#444444", linestyle="--", lw=1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("family minus alpha-only gain")
        ax.set_title(title)
    axes[0].legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Regime Router A: Do Router Signals Predict When Geometry Freedom Helps?",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_method_bars(path: str, cell_rows: list[dict[str, float | str]]) -> None:
    labels = [f'{row["condition"]}\n{row["geometry_skew_bin"]}' for row in cell_rows]
    x = np.arange(len(labels))
    width = 0.16

    conditioned = np.array([float(row["conditioned_alpha_error_mean"]) for row in cell_rows])
    family = np.array([float(row["family_alpha_error_mean"]) for row in cell_rows])
    raw_router = np.array([float(row["raw_router_alpha_error_mean"]) for row in cell_rows])
    support_router = np.array([float(row["support_router_alpha_error_mean"]) for row in cell_rows])
    oracle = np.array([float(row["oracle_router_alpha_error_mean"]) for row in cell_rows])

    fig, ax = plt.subplots(figsize=(15.0, 6.2), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.24, left=0.08, right=0.98)

    ax.bar(x - 2 * width, conditioned, width=width, color="#2a9d8f", label="alpha-only")
    ax.bar(x - width, family, width=width, color="#f4a261", label="geometry+alpha")
    ax.bar(x, raw_router, width=width, color="#577590", label="raw-ratio router")
    ax.bar(x + width, support_router, width=width, color="#264653", label="support-aware router")
    ax.bar(x + 2 * width, oracle, width=width, color="#6a4c93", label="oracle router")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Routing policies on the focused sparse moderate cells")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    fig.suptitle(
        "Regime Router B: Which Policy Wins By Cell?",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    rng = np.random.default_rng(20260324)

    bank_params, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}

    audits = {
        "alias_index_joint_shift_invariance": audit_alias_index_joint_shift_invariance(
            np.random.default_rng(20260324),
            bank_params,
        ),
        "leave_one_out_router_logic": audit_leave_one_out_router_logic(),
    }

    base_rows: list[dict[str, float | int | str]] = []
    for condition in FOCUS_CONDITIONS:
        regime = regime_map[condition]
        temperature = softmin_temperature(regime)
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            for trial_idx in range(TRIALS_PER_CELL):
                true_params = sample_conditioned_parameters(rng, FOCUS_ALPHA_BIN, skew_bin)
                clean_signature = anisotropic_forward_signature(true_params)
                _, observed_signature, mask, _ = observe_pose_free_signature(clean_signature, regime, rng)

                marginalized_scores, _ = marginalized_candidate_scores(
                    observed_signature,
                    mask,
                    shifted_bank,
                    temperature,
                )
                top_seed_indices = top_k_indices(marginalized_scores, TOP_K_SEEDS)
                top_seed_rank = 1
                top_seed_params = bank_params[top_seed_indices[0]]

                raw_ratio = alpha_strength(float(top_seed_params[5])) / max(abs(float(top_seed_params[1])), RAW_RATIO_EPS)
                support_index, absorbable_change, visible_anchor = support_aware_alias_index(top_seed_params, mask)

                conditioned_best_params = None
                conditioned_best_score = float("inf")
                for seed_rank, bank_idx in enumerate(top_seed_indices, start=1):
                    refined_params, _, _, refined_score = candidate_conditioned_search(
                        observed_signature,
                        mask,
                        bank_params[bank_idx],
                        temperature,
                    )
                    if refined_score < conditioned_best_score:
                        conditioned_best_score = refined_score
                        conditioned_best_params = refined_params
                        top_seed_rank = seed_rank if seed_rank == 1 else top_seed_rank

                family_best_params = None
                family_best_score = float("inf")
                for bank_idx in top_seed_indices:
                    refined_params, _, _, refined_score = family_switching_refine(
                        observed_signature,
                        mask,
                        bank_params[bank_idx],
                        temperature,
                    )
                    if refined_score < family_best_score:
                        family_best_score = refined_score
                        family_best_params = refined_params

                assert conditioned_best_params is not None
                assert family_best_params is not None

                conditioned_geometry, _, conditioned_alpha = symmetry_aware_errors(true_params, conditioned_best_params)
                family_geometry, _, family_alpha = symmetry_aware_errors(true_params, family_best_params)

                base_rows.append(
                    {
                        "condition": condition,
                        "geometry_skew_bin": skew_bin,
                        "trial_in_cell": trial_idx,
                        "true_alpha": float(true_params[5]),
                        "true_t": float(true_params[1]),
                        "top_seed_rank": int(top_seed_rank),
                        "top_seed_alpha": float(top_seed_params[5]),
                        "top_seed_t": float(top_seed_params[1]),
                        "top_seed_alpha_strength": alpha_strength(float(top_seed_params[5])),
                        "raw_ratio_index": float(raw_ratio),
                        "support_alias_index": float(support_index),
                        "alpha_orbit_absorbable_change": float(absorbable_change),
                        "visible_skew_anchor": float(visible_anchor),
                        "conditioned_alpha_error": float(conditioned_alpha),
                        "family_alpha_error": float(family_alpha),
                        "conditioned_geometry_mae": float(conditioned_geometry),
                        "family_geometry_mae": float(family_geometry),
                        "family_better": int(family_alpha < conditioned_alpha),
                        "family_minus_conditioned_alpha_gain": float(conditioned_alpha - family_alpha),
                        "oracle_router_alpha_error": float(min(conditioned_alpha, family_alpha)),
                    }
                )

    raw_results = leave_one_out_router(base_rows, "raw_ratio_index")
    support_results = leave_one_out_router(base_rows, "support_alias_index")

    rows: list[TrialRow] = []
    for base_row, raw_result, support_result in zip(base_rows, raw_results, support_results):
        rows.append(
            TrialRow(
                condition=str(base_row["condition"]),
                geometry_skew_bin=str(base_row["geometry_skew_bin"]),
                trial_in_cell=int(base_row["trial_in_cell"]),
                true_alpha=float(base_row["true_alpha"]),
                true_t=float(base_row["true_t"]),
                top_seed_rank=int(base_row["top_seed_rank"]),
                top_seed_alpha=float(base_row["top_seed_alpha"]),
                top_seed_t=float(base_row["top_seed_t"]),
                top_seed_alpha_strength=float(base_row["top_seed_alpha_strength"]),
                raw_ratio_index=float(base_row["raw_ratio_index"]),
                support_alias_index=float(base_row["support_alias_index"]),
                alpha_orbit_absorbable_change=float(base_row["alpha_orbit_absorbable_change"]),
                visible_skew_anchor=float(base_row["visible_skew_anchor"]),
                conditioned_alpha_error=float(base_row["conditioned_alpha_error"]),
                family_alpha_error=float(base_row["family_alpha_error"]),
                conditioned_geometry_mae=float(base_row["conditioned_geometry_mae"]),
                family_geometry_mae=float(base_row["family_geometry_mae"]),
                family_better=int(base_row["family_better"]),
                family_minus_conditioned_alpha_gain=float(base_row["family_minus_conditioned_alpha_gain"]),
                raw_router_choose_family=int(raw_result["choose_family"]),
                raw_router_alpha_error=float(raw_result["alpha_error"]),
                raw_router_correct=int(raw_result["correct"]),
                support_router_choose_family=int(support_result["choose_family"]),
                support_router_alpha_error=float(support_result["alpha_error"]),
                support_router_correct=int(support_result["correct"]),
                oracle_router_alpha_error=float(base_row["oracle_router_alpha_error"]),
            )
        )

    trial_dicts = [asdict(row) for row in rows]
    summary_rows = aggregate(rows)
    cell_rows = aggregate_by_cell(rows)

    write_csv(os.path.join(OUTPUT_DIR, "regime_router_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "regime_router_summary.csv"), summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "regime_router_by_cell.csv"), cell_rows)

    plot_scatter(os.path.join(FIGURE_DIR, "regime_router_scatter.png"), rows)
    plot_method_bars(os.path.join(FIGURE_DIR, "regime_router_method_bars.png"), cell_rows)

    summary = {
        "focus_conditions": FOCUS_CONDITIONS,
        "focus_alpha_bin": FOCUS_ALPHA_BIN,
        "trials_per_cell": float(TRIALS_PER_CELL),
        "top_k_seeds": float(TOP_K_SEEDS),
        "audits": audits,
        "overall_raw_router_alpha_error_mean": float(np.mean([row.raw_router_alpha_error for row in rows])),
        "overall_support_router_alpha_error_mean": float(np.mean([row.support_router_alpha_error for row in rows])),
        "overall_conditioned_alpha_error_mean": float(np.mean([row.conditioned_alpha_error for row in rows])),
        "overall_family_alpha_error_mean": float(np.mean([row.family_alpha_error for row in rows])),
        "overall_oracle_router_alpha_error_mean": float(np.mean([row.oracle_router_alpha_error for row in rows])),
        "overall_raw_router_accuracy_mean": float(np.mean([row.raw_router_correct for row in rows])),
        "overall_support_router_accuracy_mean": float(np.mean([row.support_router_correct for row in rows])),
        "overall_family_better_fraction": float(np.mean([row.family_better for row in rows])),
    }

    with open(os.path.join(OUTPUT_DIR, "regime_router_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows, "by_cell": cell_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows, "by_cell": cell_rows}, indent=2))

if __name__ == "__main__":
    main()
