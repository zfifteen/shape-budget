"""
Post-roadmap extension: latent ambiguity experiment for the anisotropic inverse.

This experiment asks a sharper question than best-fit recovery:

- when the latent state is inferred from boundary data, how broad is the
  near-optimal candidate family?
- does that near-optimal family broaden substantially once rotation is hidden?
- is the broadening concentrated more in alpha than in geometry or weights?

The comparison is matched:

- the same true latent state is used for both settings
- the same relative observation pattern is used on the underlying shape
- only the pose nuisance differs between the canonical and pose-free views

This is a bank-based ambiguity profile rather than a continuous posterior.
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

from run_pose_free_weighted_inverse_experiment import build_shift_stack
from run_weighted_anisotropic_inverse_experiment import (
    REFERENCE_BANK_SIZE,
    TEST_TRIALS_PER_REGIME,
    aggregate_trials,  # imported for consistency with earlier experiments
    anisotropic_forward_signature,
    build_reference_bank,
    control_invariants,
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


TOP_K_ENVELOPE = 10
ALPHA_DIVERSE_THRESHOLD = 0.20
MIN_NEAR_TIE_DELTA = 5.0e-5


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
    rotation_shift: int
    canonical_best_score: float
    canonical_gap_top2: float
    canonical_gap_topk: float
    canonical_alpha_best_error: float
    canonical_alpha_span_topk: float
    canonical_alpha_std_topk: float
    canonical_geometry_dispersion_topk: float
    canonical_weight_dispersion_topk: float
    canonical_near_tie_diverse: int
    pose_best_score: float
    pose_gap_top2: float
    pose_gap_topk: float
    pose_alpha_best_error: float
    pose_alpha_span_topk: float
    pose_alpha_std_topk: float
    pose_geometry_dispersion_topk: float
    pose_weight_dispersion_topk: float
    pose_near_tie_diverse: int
    alpha_span_ratio_pose_over_canonical: float
    alpha_error_ratio_pose_over_canonical: float
    geometry_dispersion_ratio_pose_over_canonical: float
    weight_dispersion_ratio_pose_over_canonical: float


def sample_observation_pattern(
    regime: dict[str, float | str | int],
    angle_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(angle_count, dtype=bool)
    noise = np.zeros(angle_count, dtype=float)
    mode = str(regime["mode"])

    if mode == "full":
        mask[:] = True
    elif mode == "contiguous":
        span = int(float(regime["observed_fraction"]) * angle_count)
        start = int(rng.integers(0, angle_count))
        mask[(np.arange(span) + start) % angle_count] = True
    elif mode == "random":
        count = int(regime["observed_count"])
        mask[rng.choice(angle_count, size=count, replace=False)] = True
    elif mode == "sparse_contiguous":
        span = int(float(regime["arc_fraction"]) * angle_count)
        start = int(rng.integers(0, angle_count))
        pool = (np.arange(span) + start) % angle_count
        count = min(int(regime["observed_count"]), len(pool))
        mask[rng.choice(pool, size=count, replace=False)] = True
    else:
        raise ValueError(f"Unknown observation mode: {mode}")

    sigma = float(regime["noise_sigma"])
    if sigma > 0.0:
        noise_vals = rng.normal(scale=sigma, size=int(np.sum(mask)))
        noise[mask] = noise_vals
    return mask, noise


def matched_observation_pair(
    clean_signature: np.ndarray,
    regime: dict[str, float | str | int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    angle_count = len(clean_signature)
    shift = int(rng.integers(0, angle_count))
    rotated_signature = np.roll(clean_signature, shift)

    pose_mask, pose_noise = sample_observation_pattern(regime, angle_count, rng)
    pose_observed = rotated_signature.copy()
    pose_observed[pose_mask] += pose_noise[pose_mask]

    canonical_mask = np.roll(pose_mask, -shift)
    canonical_noise = np.roll(pose_noise, -shift)
    canonical_observed = clean_signature.copy()
    canonical_observed[canonical_mask] += canonical_noise[canonical_mask]

    return canonical_observed, canonical_mask, pose_observed, pose_mask, shift, rotated_signature


def canonical_candidate_scores(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    bank_signatures: np.ndarray,
) -> np.ndarray:
    residual = bank_signatures[:, mask] - observed_signature[mask]
    return np.mean(residual * residual, axis=1)


def pose_free_candidate_scores(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    mse = np.mean(residual * residual, axis=2)
    best_shift = np.argmin(mse, axis=1)
    best_score = np.min(mse, axis=1)
    return best_score, best_shift


def canonicalize_candidate(
    params: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    geometry, weights, alpha = control_invariants(params)
    swapped_geometry = np.array([geometry[0], geometry[2], geometry[1]])
    swapped_weights = np.array([weights[1], weights[0], weights[2]])

    direct_tuple = tuple(np.concatenate([geometry, weights]))
    swapped_tuple = tuple(np.concatenate([swapped_geometry, swapped_weights]))
    if swapped_tuple < direct_tuple:
        return swapped_geometry, swapped_weights, alpha
    return geometry, weights, alpha


def near_tie_gap_threshold(regime: dict[str, float | str | int]) -> float:
    sigma = float(regime["noise_sigma"])
    return max(sigma * sigma, MIN_NEAR_TIE_DELTA)


def ambiguity_metrics(
    scores: np.ndarray,
    params_list: list[tuple[float, float, float, float, float, float]],
    true_params: tuple[float, float, float, float, float, float],
    regime: dict[str, float | str | int],
) -> dict[str, float]:
    order = np.argsort(scores)
    top_k = min(TOP_K_ENVELOPE, len(order))
    top_indices = order[:top_k]
    top_scores = scores[top_indices]

    geometries = []
    weights = []
    alphas = []
    for idx in top_indices:
        geometry, weight, alpha = canonicalize_candidate(params_list[int(idx)])
        geometries.append(geometry)
        weights.append(weight)
        alphas.append(alpha)

    geometry_matrix = np.array(geometries)
    weight_matrix = np.array(weights)
    alpha_vec = np.array(alphas)

    best_idx = int(order[0])
    best_params = params_list[best_idx]
    best_geometry_mae, best_weight_mae, best_alpha_error = symmetry_aware_errors(true_params, best_params)

    gap_top2 = float(scores[order[1]] - scores[order[0]]) if len(order) > 1 else 0.0
    gap_topk = float(top_scores[-1] - top_scores[0])
    near_tie_diverse = int(gap_topk <= near_tie_gap_threshold(regime) and (float(np.max(alpha_vec) - np.min(alpha_vec)) >= ALPHA_DIVERSE_THRESHOLD))

    return {
        "best_score": float(scores[order[0]]),
        "gap_top2": gap_top2,
        "gap_topk": gap_topk,
        "alpha_best_error": float(best_alpha_error),
        "alpha_span_topk": float(np.max(alpha_vec) - np.min(alpha_vec)),
        "alpha_std_topk": float(np.std(alpha_vec)),
        "geometry_dispersion_topk": float(np.mean(np.std(geometry_matrix, axis=0))),
        "weight_dispersion_topk": float(np.mean(np.std(weight_matrix, axis=0))),
        "near_tie_diverse": near_tie_diverse,
        "best_geometry_mae": float(best_geometry_mae),
        "best_weight_mae": float(best_weight_mae),
    }


def summarize_trials(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in subset]))

        summary.append(
            {
                "condition": name,
                "canonical_alpha_span_topk_mean": mean("canonical_alpha_span_topk"),
                "pose_alpha_span_topk_mean": mean("pose_alpha_span_topk"),
                "canonical_alpha_best_error_mean": mean("canonical_alpha_best_error"),
                "pose_alpha_best_error_mean": mean("pose_alpha_best_error"),
                "canonical_geometry_dispersion_topk_mean": mean("canonical_geometry_dispersion_topk"),
                "pose_geometry_dispersion_topk_mean": mean("pose_geometry_dispersion_topk"),
                "canonical_weight_dispersion_topk_mean": mean("canonical_weight_dispersion_topk"),
                "pose_weight_dispersion_topk_mean": mean("pose_weight_dispersion_topk"),
                "canonical_gap_topk_mean": mean("canonical_gap_topk"),
                "pose_gap_topk_mean": mean("pose_gap_topk"),
                "canonical_near_tie_diverse_fraction": mean("canonical_near_tie_diverse"),
                "pose_near_tie_diverse_fraction": mean("pose_near_tie_diverse"),
                "alpha_span_ratio_pose_over_canonical_mean": mean("alpha_span_ratio_pose_over_canonical"),
                "alpha_error_ratio_pose_over_canonical_mean": mean("alpha_error_ratio_pose_over_canonical"),
                "geometry_dispersion_ratio_pose_over_canonical_mean": mean("geometry_dispersion_ratio_pose_over_canonical"),
                "weight_dispersion_ratio_pose_over_canonical_mean": mean("weight_dispersion_ratio_pose_over_canonical"),
            }
        )
    return summary


def plot_ambiguity_overview(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    x = np.arange(len(conditions))
    width = 0.36

    canonical_alpha_span = np.array([float(item["canonical_alpha_span_topk_mean"]) for item in summary_rows])
    pose_alpha_span = np.array([float(item["pose_alpha_span_topk_mean"]) for item in summary_rows])
    canonical_alpha_error = np.array([float(item["canonical_alpha_best_error_mean"]) for item in summary_rows])
    pose_alpha_error = np.array([float(item["pose_alpha_best_error_mean"]) for item in summary_rows])
    canonical_geom = np.array([float(item["canonical_geometry_dispersion_topk_mean"]) for item in summary_rows])
    pose_geom = np.array([float(item["pose_geometry_dispersion_topk_mean"]) for item in summary_rows])
    canonical_flag = np.array([float(item["canonical_near_tie_diverse_fraction"]) for item in summary_rows])
    pose_flag = np.array([float(item["pose_near_tie_diverse_fraction"]) for item in summary_rows])

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.98, wspace=0.22, hspace=0.34)

    for ax, canonical_vals, pose_vals, ylabel, title in [
        (axes[0, 0], canonical_alpha_span, pose_alpha_span, "mean top-10 alpha span", "Ambiguity in alpha broadens under hidden rotation"),
        (axes[0, 1], canonical_alpha_error, pose_alpha_error, "mean best-candidate alpha error", "Best alpha error versus observation setting"),
        (axes[1, 0], canonical_geom, pose_geom, "mean top-10 geometry dispersion", "Geometry ambiguity broadening is smaller"),
        (axes[1, 1], canonical_flag, pose_flag, "fraction of trials", "Near-tie and alpha-diverse trials"),
    ]:
        ax.bar(x - width / 2.0, canonical_vals, width=width, color="#457b9d", label="canonical")
        ax.bar(x + width / 2.0, pose_vals, width=width, color="#e76f51", label="pose-free")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0, 0].legend(loc="upper left", frameon=True)

    fig.suptitle("Latent Ambiguity A: Matched Canonical Versus Pose-Free Ambiguity Profile", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_example_spectra(
    path: str,
    rows: list[TrialRow],
    canonical_scores: dict[tuple[str, int], np.ndarray],
    pose_scores: dict[tuple[str, int], np.ndarray],
    params_list: list[tuple[float, float, float, float, float, float]],
) -> None:
    chosen_conditions = ["full_noisy", "sparse_full_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(len(chosen_conditions), 2, figsize=(13.8, 10.8), constrained_layout=False)
    fig.subplots_adjust(top=0.92, hspace=0.42, wspace=0.18)

    for row_idx, condition in enumerate(chosen_conditions):
        subset = [row for row in rows if row.condition == condition]
        exemplar = max(subset, key=lambda row: row.alpha_span_ratio_pose_over_canonical)
        key = (condition, exemplar.trial)
        true_alpha = exemplar.true_alpha

        for col_idx, (label, score_map, alpha_span_attr) in enumerate(
            [
                ("canonical", canonical_scores, "canonical_alpha_span_topk"),
                ("pose-free", pose_scores, "pose_alpha_span_topk"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            scores = score_map[key]
            order = np.argsort(scores)
            top_indices = order[:20]
            top_scores = scores[top_indices]
            alphas = np.array([params_list[int(idx)][5] for idx in top_indices])
            rel_scores = top_scores - top_scores[0]

            ax.scatter(alphas, rel_scores, s=34, color="#1d3557", alpha=0.78)
            top10 = top_indices[: min(TOP_K_ENVELOPE, len(top_indices))]
            top10_alpha = np.array([params_list[int(idx)][5] for idx in top10])
            top10_rel = scores[top10] - scores[top10[0]]
            ax.scatter(top10_alpha, top10_rel, s=48, color="#e76f51", alpha=0.9, label="top-10 envelope")
            ax.axvline(true_alpha, color="#2a9d8f", linestyle="--", lw=1.6, label="true alpha")
            ax.set_xlabel("candidate alpha")
            ax.set_ylabel("score minus best score")
            ax.set_title(
                f"{condition} / {label}: alpha span = {getattr(exemplar, alpha_span_attr):.3f}"
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Latent Ambiguity B: Near-Optimal Alpha Spectra For Matched Cases", fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)

    rows: list[TrialRow] = []
    canonical_scores_store: dict[tuple[str, int], np.ndarray] = {}
    pose_scores_store: dict[tuple[str, int], np.ndarray] = {}

    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_anisotropic_parameters(rng)
            clean_signature = anisotropic_forward_signature(true_params)
            canonical_observed, canonical_mask, pose_observed, pose_mask, shift, _ = matched_observation_pair(clean_signature, regime, rng)

            canonical_scores = canonical_candidate_scores(canonical_observed, canonical_mask, bank_signatures)
            pose_scores, _ = pose_free_candidate_scores(pose_observed, pose_mask, shifted_bank)

            canonical_metrics = ambiguity_metrics(canonical_scores, params_list, true_params, regime)
            pose_metrics = ambiguity_metrics(pose_scores, params_list, true_params, regime)

            key = (str(regime["name"]), trial)
            canonical_scores_store[key] = canonical_scores
            pose_scores_store[key] = pose_scores

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
                    rotation_shift=int(shift),
                    canonical_best_score=float(canonical_metrics["best_score"]),
                    canonical_gap_top2=float(canonical_metrics["gap_top2"]),
                    canonical_gap_topk=float(canonical_metrics["gap_topk"]),
                    canonical_alpha_best_error=float(canonical_metrics["alpha_best_error"]),
                    canonical_alpha_span_topk=float(canonical_metrics["alpha_span_topk"]),
                    canonical_alpha_std_topk=float(canonical_metrics["alpha_std_topk"]),
                    canonical_geometry_dispersion_topk=float(canonical_metrics["geometry_dispersion_topk"]),
                    canonical_weight_dispersion_topk=float(canonical_metrics["weight_dispersion_topk"]),
                    canonical_near_tie_diverse=int(canonical_metrics["near_tie_diverse"]),
                    pose_best_score=float(pose_metrics["best_score"]),
                    pose_gap_top2=float(pose_metrics["gap_top2"]),
                    pose_gap_topk=float(pose_metrics["gap_topk"]),
                    pose_alpha_best_error=float(pose_metrics["alpha_best_error"]),
                    pose_alpha_span_topk=float(pose_metrics["alpha_span_topk"]),
                    pose_alpha_std_topk=float(pose_metrics["alpha_std_topk"]),
                    pose_geometry_dispersion_topk=float(pose_metrics["geometry_dispersion_topk"]),
                    pose_weight_dispersion_topk=float(pose_metrics["weight_dispersion_topk"]),
                    pose_near_tie_diverse=int(pose_metrics["near_tie_diverse"]),
                    alpha_span_ratio_pose_over_canonical=float(pose_metrics["alpha_span_topk"] / max(canonical_metrics["alpha_span_topk"], 1.0e-12)),
                    alpha_error_ratio_pose_over_canonical=float(pose_metrics["alpha_best_error"] / max(canonical_metrics["alpha_best_error"], 1.0e-12)),
                    geometry_dispersion_ratio_pose_over_canonical=float(
                        pose_metrics["geometry_dispersion_topk"] / max(canonical_metrics["geometry_dispersion_topk"], 1.0e-12)
                    ),
                    weight_dispersion_ratio_pose_over_canonical=float(
                        pose_metrics["weight_dispersion_topk"] / max(canonical_metrics["weight_dispersion_topk"], 1.0e-12)
                    ),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = summarize_trials(rows)
    write_csv(os.path.join(OUTPUT_DIR, "latent_ambiguity_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "latent_ambiguity_summary.csv"), summary_rows)

    plot_ambiguity_overview(os.path.join(FIGURE_DIR, "latent_ambiguity_overview.png"), summary_rows)
    plot_example_spectra(
        os.path.join(FIGURE_DIR, "latent_ambiguity_spectra.png"),
        rows,
        canonical_scores_store,
        pose_scores_store,
        params_list,
    )

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "top_k_envelope": TOP_K_ENVELOPE,
        "alpha_diverse_threshold": ALPHA_DIVERSE_THRESHOLD,
        "min_near_tie_delta": MIN_NEAR_TIE_DELTA,
        "smallest_alpha_span_ratio_pose_over_canonical_mean": float(
            min(item["alpha_span_ratio_pose_over_canonical_mean"] for item in summary_rows)
        ),
        "largest_alpha_span_ratio_pose_over_canonical_mean": float(
            max(item["alpha_span_ratio_pose_over_canonical_mean"] for item in summary_rows)
        ),
        "smallest_alpha_error_ratio_pose_over_canonical_mean": float(
            min(item["alpha_error_ratio_pose_over_canonical_mean"] for item in summary_rows)
        ),
        "largest_alpha_error_ratio_pose_over_canonical_mean": float(
            max(item["alpha_error_ratio_pose_over_canonical_mean"] for item in summary_rows)
        ),
        "smallest_near_tie_diverse_fraction_gap": float(
            min(
                float(item["pose_near_tie_diverse_fraction"]) - float(item["canonical_near_tie_diverse_fraction"])
                for item in summary_rows
            )
        ),
        "largest_near_tie_diverse_fraction_gap": float(
            max(
                float(item["pose_near_tie_diverse_fraction"]) - float(item["canonical_near_tie_diverse_fraction"])
                for item in summary_rows
            )
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "latent_ambiguity_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
