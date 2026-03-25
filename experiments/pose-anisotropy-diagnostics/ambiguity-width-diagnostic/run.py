"""
Focused ambiguity-width diagnostic for the pose-free anisotropic solver challenge.

This experiment tests a sharper version of the current symmetry diagnosis.

Question:

- when the same focused observation is evaluated against several independent
  reference banks, does a wide near-optimal alpha family predict cross-bank
  alpha instability while geometry stays comparatively tight?

The core idea is to measure an observation-conditioned ambiguity ratio from the
near-best candidate set before any router is trained, then test whether that
ratio predicts cross-bank disagreement on alpha more strongly than geometry
spread or simple entropy-style signals do.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

GEOMETRY_SKEW_BIN_LABELS, sample_conditioned_parameters = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "GEOMETRY_SKEW_BIN_LABELS",
    "sample_conditioned_parameters",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

ALPHA_MAX, ALPHA_MIN, GEOMETRY_BOUNDS, anisotropic_forward_signature, build_reference_bank, control_invariants, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "GEOMETRY_BOUNDS",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "control_invariants",
    "symmetry_aware_errors",
)

OBSERVATION_REGIMES, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "write_csv",
)

FOCUS_ALPHA_BIN, FOCUS_CONDITIONS, SolverContext, marginalized_bank_scores, softmin_temperature = load_symbols(
    "run_joint_pose_marginalized_solver_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py",
    "FOCUS_ALPHA_BIN",
    "FOCUS_CONDITIONS",
    "SolverContext",
    "marginalized_bank_scores",
    "softmin_temperature",
)

import json
import math
import os
from dataclasses import dataclass, asdict

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

BLOCK_SPECS = {
    "calibration": (20260410, 20260411, 20260412, 20260416, 20260417, 20260418),
    "holdout": (20260422, 20260423, 20260424),
    "confirmation": (20260425, 20260426, 20260427),
}

BANK_SEEDS = (20260324, 20260325, 20260326, 20260327, 20260328)
REFERENCE_BANK_SIZE = 300
MIN_SCORE_BAND = 5.0e-5
ALPHA_UNSTABLE_LOG_SPAN_THRESHOLD = 0.20
NUMERIC_EPS = 1.0e-9


@dataclass
class BankRow:
    split: str
    observation_seed: int
    bank_seed: int
    condition: str
    geometry_skew_bin: str
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    best_score: float
    best_entropy: float
    candidate_count: int
    alpha_log_span_set: float
    geometry_span_norm_set: float
    weight_span_norm_set: float
    ambiguity_ratio: float
    best_alpha: float
    best_alpha_error: float
    best_geometry_error: float
    best_weight_error: float


@dataclass
class TrialRow:
    split: str
    observation_seed: int
    condition: str
    geometry_skew_bin: str
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    mean_best_entropy: float
    mean_candidate_count: float
    mean_alpha_log_span_set: float
    mean_geometry_span_norm_set: float
    mean_weight_span_norm_set: float
    mean_ambiguity_ratio: float
    max_ambiguity_ratio: float
    alpha_bank_log_span: float
    geometry_bank_span_norm: float
    weight_bank_span_norm: float
    alpha_unstable_flag: int
    best_alpha_error_mean: float
    best_alpha_error_std: float
    best_geometry_error_mean: float
    best_weight_error_mean: float


@dataclass(frozen=True)
class BankContext:
    seed: int
    params_list: list[tuple[float, float, float, float, float, float]]
    shifted_bank: np.ndarray
    geometries: np.ndarray
    weights: np.ndarray
    alpha_logs: np.ndarray


def condition_index(condition: str) -> int:
    return FOCUS_CONDITIONS.index(condition)


def skew_index(skew_bin: str) -> int:
    return GEOMETRY_SKEW_BIN_LABELS.index(skew_bin)


def make_trial_rng(observation_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(observation_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


def canonicalize_candidate(
    params: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    geometry, weights, alpha = control_invariants(params)
    swapped_geometry = np.array([geometry[0], geometry[2], geometry[1]])
    swapped_weights = np.array([weights[1], weights[0], weights[2]])
    direct_tuple = tuple(np.concatenate([geometry, weights]))
    swapped_tuple = tuple(np.concatenate([swapped_geometry, swapped_weights]))
    if swapped_tuple < direct_tuple:
        return swapped_geometry, swapped_weights, float(alpha)
    return geometry, weights, float(alpha)


def build_bank_context(bank_seed: int) -> BankContext:
    bank_rng = np.random.default_rng(bank_seed)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, bank_rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    geometries = []
    weights = []
    alpha_logs = []
    for params in params_list:
        geometry, weight, alpha = canonicalize_candidate(params)
        geometries.append(geometry)
        weights.append(weight)
        alpha_logs.append(math.log(alpha))
    return BankContext(
        seed=bank_seed,
        params_list=params_list,
        shifted_bank=shifted_bank,
        geometries=np.array(geometries),
        weights=np.array(weights),
        alpha_logs=np.array(alpha_logs),
    )


def empirical_geometry_ranges(bank_contexts: list[BankContext]) -> np.ndarray:
    all_geometries = np.concatenate([context.geometries for context in bank_contexts], axis=0)
    span = np.max(all_geometries, axis=0) - np.min(all_geometries, axis=0)
    return np.maximum(span, NUMERIC_EPS)


def score_band(regime: dict[str, float | str | int]) -> float:
    return max(float(regime["noise_sigma"]) ** 2, MIN_SCORE_BAND)


def normalized_spans(
    geometries: np.ndarray,
    weights: np.ndarray,
    geometry_ranges: np.ndarray,
) -> tuple[float, float, float]:
    geometry_span = (np.max(geometries, axis=0) - np.min(geometries, axis=0)) / geometry_ranges
    weight_span = np.max(weights, axis=0) - np.min(weights, axis=0)
    geometry_mean = float(np.mean(geometry_span))
    weight_mean = float(np.mean(weight_span))
    combined = float(np.mean(np.concatenate([geometry_span, weight_span])))
    return geometry_mean, weight_mean, combined


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def balanced_accuracy(rows: list[TrialRow], metric_name: str, threshold: float) -> float:
    labels = np.array([row.alpha_unstable_flag for row in rows], dtype=int)
    preds = np.array([int(getattr(row, metric_name) >= threshold) for row in rows], dtype=int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = float(np.sum((preds == 1) & (labels == 1)) / positives)
    tnr = float(np.sum((preds == 0) & (labels == 0)) / negatives)
    return 0.5 * (tpr + tnr)


def choose_threshold(calibration_rows: list[TrialRow], metric_name: str) -> tuple[float, float]:
    values = sorted({float(getattr(row, metric_name)) for row in calibration_rows})
    if not values:
        return 0.0, float("nan")
    candidates = [values[0] - 1.0e-6]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(values[:-1], values[1:]))
    candidates.append(values[-1] + 1.0e-6)
    scored = [(balanced_accuracy(calibration_rows, metric_name, threshold), threshold) for threshold in candidates]
    scored.sort(key=lambda item: (item[0], -item[1]))
    best_score, best_threshold = scored[-1]
    return float(best_threshold), float(best_score)


def select_candidate_indices(scores: np.ndarray, band: float) -> np.ndarray:
    order = np.argsort(scores)
    best_score = float(scores[order[0]])
    keep = order[scores[order] <= best_score + band]
    if len(keep) == 0:
        return order[:1]
    return keep


def evaluate_trial(
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    regime: dict[str, float | str | int],
    bank_contexts: list[BankContext],
    geometry_ranges: np.ndarray,
) -> tuple[list[BankRow], TrialRow]:
    trial_rng = make_trial_rng(observation_seed, condition, skew_bin)
    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
    clean_signature = anisotropic_forward_signature(true_params)
    _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
    temperature = softmin_temperature(regime)
    band = score_band(regime)

    bank_rows: list[BankRow] = []
    best_geometries = []
    best_weights = []
    best_alpha_logs = []
    best_alpha_errors = []
    best_geometry_errors = []
    best_weight_errors = []
    ambiguity_ratios = []
    best_entropies = []
    candidate_counts = []
    alpha_log_spans = []
    geometry_set_spans = []
    weight_set_spans = []

    for context in bank_contexts:
        scores, _ = marginalized_bank_scores(observed_signature, mask, context.shifted_bank, temperature)
        candidate_indices = select_candidate_indices(scores, band)
        geometry_set = context.geometries[candidate_indices]
        weight_set = context.weights[candidate_indices]
        alpha_log_set = context.alpha_logs[candidate_indices]
        geometry_span_norm, weight_span_norm, combined_span_norm = normalized_spans(geometry_set, weight_set, geometry_ranges)
        alpha_log_span = float(np.max(alpha_log_set) - np.min(alpha_log_set))
        ambiguity_ratio = float(alpha_log_span / max(combined_span_norm, NUMERIC_EPS))

        best_idx = int(np.argmin(scores))
        best_params = context.params_list[best_idx]
        solver_context = SolverContext(observed_signature, mask)
        best_score, _, _, best_entropy = solver_context.score_params(best_params, temperature)
        best_geometry_error, best_weight_error, best_alpha_error = symmetry_aware_errors(true_params, best_params)

        best_geometry, best_weight, best_alpha = canonicalize_candidate(best_params)
        best_geometries.append(best_geometry)
        best_weights.append(best_weight)
        best_alpha_logs.append(math.log(best_alpha))
        best_alpha_errors.append(float(best_alpha_error))
        best_geometry_errors.append(float(best_geometry_error))
        best_weight_errors.append(float(best_weight_error))
        ambiguity_ratios.append(ambiguity_ratio)
        best_entropies.append(float(best_entropy))
        candidate_counts.append(int(len(candidate_indices)))
        alpha_log_spans.append(alpha_log_span)
        geometry_set_spans.append(geometry_span_norm)
        weight_set_spans.append(weight_span_norm)

        bank_rows.append(
            BankRow(
                split=split,
                observation_seed=int(observation_seed),
                bank_seed=int(context.seed),
                condition=condition,
                geometry_skew_bin=skew_bin,
                true_alpha=float(true_params[5]),
                true_t=float(true_params[1]),
                true_rotation_shift=int(true_shift),
                best_score=float(best_score),
                best_entropy=float(best_entropy),
                candidate_count=int(len(candidate_indices)),
                alpha_log_span_set=alpha_log_span,
                geometry_span_norm_set=geometry_span_norm,
                weight_span_norm_set=weight_span_norm,
                ambiguity_ratio=ambiguity_ratio,
                best_alpha=float(best_alpha),
                best_alpha_error=float(best_alpha_error),
                best_geometry_error=float(best_geometry_error),
                best_weight_error=float(best_weight_error),
            )
        )

    bank_geometry_span_norm, bank_weight_span_norm, _ = normalized_spans(
        np.array(best_geometries),
        np.array(best_weights),
        geometry_ranges,
    )
    alpha_bank_log_span = float(np.max(best_alpha_logs) - np.min(best_alpha_logs))

    trial_row = TrialRow(
        split=split,
        observation_seed=int(observation_seed),
        condition=condition,
        geometry_skew_bin=skew_bin,
        true_alpha=float(true_params[5]),
        true_t=float(true_params[1]),
        true_rotation_shift=int(true_shift),
        mean_best_entropy=float(np.mean(best_entropies)),
        mean_candidate_count=float(np.mean(candidate_counts)),
        mean_alpha_log_span_set=float(np.mean(alpha_log_spans)),
        mean_geometry_span_norm_set=float(np.mean(geometry_set_spans)),
        mean_weight_span_norm_set=float(np.mean(weight_set_spans)),
        mean_ambiguity_ratio=float(np.mean(ambiguity_ratios)),
        max_ambiguity_ratio=float(np.max(ambiguity_ratios)),
        alpha_bank_log_span=alpha_bank_log_span,
        geometry_bank_span_norm=bank_geometry_span_norm,
        weight_bank_span_norm=bank_weight_span_norm,
        alpha_unstable_flag=int(alpha_bank_log_span >= ALPHA_UNSTABLE_LOG_SPAN_THRESHOLD),
        best_alpha_error_mean=float(np.mean(best_alpha_errors)),
        best_alpha_error_std=safe_std(best_alpha_errors),
        best_geometry_error_mean=float(np.mean(best_geometry_errors)),
        best_weight_error_mean=float(np.mean(best_weight_errors)),
    )
    return bank_rows, trial_row


def summarize_by_split(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        subset = [row for row in rows if row.split == split]
        if not subset:
            continue
        summary.append(
            {
                "split": split,
                "count": len(subset),
                "mean_ambiguity_ratio": float(np.mean([row.mean_ambiguity_ratio for row in subset])),
                "mean_best_entropy": float(np.mean([row.mean_best_entropy for row in subset])),
                "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in subset])),
                "mean_geometry_bank_span_norm": float(np.mean([row.geometry_bank_span_norm for row in subset])),
                "mean_weight_bank_span_norm": float(np.mean([row.weight_bank_span_norm for row in subset])),
                "alpha_instability_rate": float(np.mean([row.alpha_unstable_flag for row in subset])),
                "ambiguity_ratio_vs_alpha_span_corr": safe_corr(
                    [row.mean_ambiguity_ratio for row in subset],
                    [row.alpha_bank_log_span for row in subset],
                ),
                "alpha_set_span_vs_alpha_span_corr": safe_corr(
                    [row.mean_alpha_log_span_set for row in subset],
                    [row.alpha_bank_log_span for row in subset],
                ),
                "ambiguity_ratio_vs_geometry_span_corr": safe_corr(
                    [row.mean_ambiguity_ratio for row in subset],
                    [row.geometry_bank_span_norm for row in subset],
                ),
                "entropy_vs_alpha_span_corr": safe_corr(
                    [row.mean_best_entropy for row in subset],
                    [row.alpha_bank_log_span for row in subset],
                ),
            }
        )
    return summary


def summarize_by_condition(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            subset = [row for row in rows if row.split == split and row.condition == condition]
            if not subset:
                continue
            summary.append(
                {
                    "split": split,
                    "condition": condition,
                    "count": len(subset),
                    "mean_ambiguity_ratio": float(np.mean([row.mean_ambiguity_ratio for row in subset])),
                    "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in subset])),
                    "mean_geometry_bank_span_norm": float(np.mean([row.geometry_bank_span_norm for row in subset])),
                    "alpha_instability_rate": float(np.mean([row.alpha_unstable_flag for row in subset])),
                }
            )
    return summary


def summarize_by_cell(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                subset = [
                    row
                    for row in rows
                    if row.split == split
                    and row.condition == condition
                    and row.geometry_skew_bin == skew_bin
                ]
                if not subset:
                    continue
                summary.append(
                    {
                        "split": split,
                        "condition": condition,
                        "alpha_strength_bin": FOCUS_ALPHA_BIN,
                        "geometry_skew_bin": skew_bin,
                        "count": len(subset),
                        "mean_ambiguity_ratio": float(np.mean([row.mean_ambiguity_ratio for row in subset])),
                        "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in subset])),
                        "mean_geometry_bank_span_norm": float(np.mean([row.geometry_bank_span_norm for row in subset])),
                        "alpha_instability_rate": float(np.mean([row.alpha_unstable_flag for row in subset])),
                    }
                )
    return summary


def evaluate_threshold_rules(rows: list[TrialRow]) -> dict[str, object]:
    calibration_rows = [row for row in rows if row.split == "calibration"]
    holdout_rows = [row for row in rows if row.split == "holdout"]
    confirmation_rows = [row for row in rows if row.split == "confirmation"]
    metrics = ["mean_ambiguity_ratio", "mean_alpha_log_span_set", "mean_best_entropy"]
    results = []
    for metric in metrics:
        threshold, calibration_score = choose_threshold(calibration_rows, metric)
        results.append(
            {
                "metric": metric,
                "threshold": threshold,
                "calibration_balanced_accuracy": calibration_score,
                "holdout_balanced_accuracy": balanced_accuracy(holdout_rows, metric, threshold),
                "confirmation_balanced_accuracy": balanced_accuracy(confirmation_rows, metric, threshold),
                "overall_balanced_accuracy": balanced_accuracy(rows, metric, threshold),
            }
        )
    best_metric = max(results, key=lambda item: item["calibration_balanced_accuracy"])
    return {"metrics": results, "best_by_calibration": best_metric}


def plot_alpha_scatter(path: str, rows: list[TrialRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.97)
    palette = {"calibration": "#1d3557", "holdout": "#2a9d8f", "confirmation": "#e76f51"}
    for split in BLOCK_SPECS:
        subset = [row for row in rows if row.split == split]
        ax.scatter(
            [row.mean_ambiguity_ratio for row in subset],
            [row.alpha_bank_log_span for row in subset],
            s=56,
            alpha=0.82,
            label=split,
            color=palette[split],
        )
    xs = np.array([row.mean_ambiguity_ratio for row in rows], dtype=float)
    ys = np.array([row.alpha_bank_log_span for row in rows], dtype=float)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
    ax.plot(grid, slope * grid + intercept, color="#111111", lw=1.6, linestyle=":")
    ax.axhline(ALPHA_UNSTABLE_LOG_SPAN_THRESHOLD, color="#444444", lw=1.2, linestyle="--")
    ax.set_xlabel("mean ambiguity ratio")
    ax.set_ylabel("cross-bank log(alpha) span")
    ax.set_title("Ambiguity ratio vs cross-bank alpha instability")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_scatter(path: str, rows: list[TrialRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.97)
    palette = {"sparse_full_noisy": "#457b9d", "sparse_partial_high_noise": "#e63946"}
    for condition in FOCUS_CONDITIONS:
        subset = [row for row in rows if row.condition == condition]
        ax.scatter(
            [row.mean_ambiguity_ratio for row in subset],
            [row.geometry_bank_span_norm for row in subset],
            s=56,
            alpha=0.82,
            label=condition,
            color=palette[condition],
        )
    xs = np.array([row.mean_ambiguity_ratio for row in rows], dtype=float)
    ys = np.array([row.geometry_bank_span_norm for row in rows], dtype=float)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
    ax.plot(grid, slope * grid + intercept, color="#111111", lw=1.6, linestyle=":")
    ax.set_xlabel("mean ambiguity ratio")
    ax.set_ylabel("cross-bank geometry span (normalized)")
    ax.set_title("Geometry stays tighter than alpha across the ambiguity sweep")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_bars(path: str, threshold_payload: dict[str, object]) -> None:
    metrics = [item["metric"] for item in threshold_payload["metrics"]]
    calibration = np.array([item["calibration_balanced_accuracy"] for item in threshold_payload["metrics"]], dtype=float)
    holdout = np.array([item["holdout_balanced_accuracy"] for item in threshold_payload["metrics"]], dtype=float)
    confirmation = np.array([item["confirmation_balanced_accuracy"] for item in threshold_payload["metrics"]], dtype=float)
    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.10, right=0.97)
    ax.bar(x - width, calibration, width=width, color="#1d3557", label="calibration")
    ax.bar(x, holdout, width=width, color="#2a9d8f", label="holdout")
    ax.bar(x + width, confirmation, width=width, color="#e76f51", label="confirmation")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=12, ha="right")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Ambiguity ratio vs entropy as instability classifiers")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}
    bank_contexts = [build_bank_context(bank_seed) for bank_seed in BANK_SEEDS]
    geometry_ranges = empirical_geometry_ranges(bank_contexts)

    bank_rows: list[BankRow] = []
    trial_rows: list[TrialRow] = []

    for split, observation_seeds in BLOCK_SPECS.items():
        for observation_seed in observation_seeds:
            for condition in FOCUS_CONDITIONS:
                regime = regime_map[condition]
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    new_bank_rows, trial_row = evaluate_trial(
                        split=split,
                        observation_seed=int(observation_seed),
                        condition=condition,
                        skew_bin=skew_bin,
                        regime=regime,
                        bank_contexts=bank_contexts,
                        geometry_ranges=geometry_ranges,
                    )
                    bank_rows.extend(new_bank_rows)
                    trial_rows.append(trial_row)

    split_summary = summarize_by_split(trial_rows)
    condition_summary = summarize_by_condition(trial_rows)
    cell_summary = summarize_by_cell(trial_rows)
    threshold_payload = evaluate_threshold_rules(trial_rows)

    global_summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "bank_seeds": list(BANK_SEEDS),
        "score_band_rule": "best_score + max(noise_sigma^2, 5e-5)",
        "alpha_unstable_log_span_threshold": ALPHA_UNSTABLE_LOG_SPAN_THRESHOLD,
        "trial_count": len(trial_rows),
        "mean_ambiguity_ratio": float(np.mean([row.mean_ambiguity_ratio for row in trial_rows])),
        "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in trial_rows])),
        "mean_geometry_bank_span_norm": float(np.mean([row.geometry_bank_span_norm for row in trial_rows])),
        "ambiguity_ratio_vs_alpha_span_corr": safe_corr(
            [row.mean_ambiguity_ratio for row in trial_rows],
            [row.alpha_bank_log_span for row in trial_rows],
        ),
        "alpha_set_span_vs_alpha_span_corr": safe_corr(
            [row.mean_alpha_log_span_set for row in trial_rows],
            [row.alpha_bank_log_span for row in trial_rows],
        ),
        "ambiguity_ratio_vs_geometry_span_corr": safe_corr(
            [row.mean_ambiguity_ratio for row in trial_rows],
            [row.geometry_bank_span_norm for row in trial_rows],
        ),
        "entropy_vs_alpha_span_corr": safe_corr(
            [row.mean_best_entropy for row in trial_rows],
            [row.alpha_bank_log_span for row in trial_rows],
        ),
        "threshold_rule": threshold_payload,
    }

    output_payload = {
        "summary": global_summary,
        "by_split": split_summary,
        "by_condition": condition_summary,
        "by_cell": cell_summary,
    }

    write_csv(os.path.join(OUTPUT_DIR, "ambiguity_width_diagnostic_bank_rows.csv"), [asdict(row) for row in bank_rows])
    write_csv(os.path.join(OUTPUT_DIR, "ambiguity_width_diagnostic_trials.csv"), [asdict(row) for row in trial_rows])
    write_csv(os.path.join(OUTPUT_DIR, "ambiguity_width_diagnostic_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, "ambiguity_width_diagnostic_condition_summary.csv"), condition_summary)
    write_csv(os.path.join(OUTPUT_DIR, "ambiguity_width_diagnostic_cell_summary.csv"), cell_summary)
    with open(os.path.join(OUTPUT_DIR, "ambiguity_width_diagnostic_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    plot_alpha_scatter(
        os.path.join(FIGURE_DIR, "ambiguity_width_diagnostic_alpha_scatter.png"),
        trial_rows,
    )
    plot_geometry_scatter(
        os.path.join(FIGURE_DIR, "ambiguity_width_diagnostic_geometry_scatter.png"),
        trial_rows,
    )
    plot_threshold_bars(
        os.path.join(FIGURE_DIR, "ambiguity_width_diagnostic_threshold_bars.png"),
        threshold_payload,
    )

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
