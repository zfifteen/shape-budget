"""
Backbone-anchored alpha observability gate for the pose-free anisotropic solver challenge.

This is the second capability layer in the layered solver program.

Layer 1 established that a near-family geometry consensus recovers a stable
control backbone across fresh bank seeds. This layer asks the next question:

- once that backbone is anchored, does the observation support a trustworthy
  point estimate for alpha?

The experiment does not attempt a full solver. It builds an observability gate
on top of the validated geometry backbone and tests whether that gate can
identify the cases where a backbone-anchored alpha estimate is trustworthy.
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

ALPHA_MAX, ALPHA_MIN, anisotropic_forward_signature, build_reference_bank, control_invariants = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "control_invariants",
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
from dataclasses import asdict, dataclass

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
NUMERIC_EPS = 1.0e-9
GEOMETRY_ANCHOR_SCALE = 0.10
ALPHA_STABLE_LOG_SPAN_THRESHOLD = 0.20
ALPHA_POINT_ABS_ERROR_THRESHOLD = 0.15


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
    best_entropy: float
    candidate_count: int
    alpha_log_span_set: float
    geometry_span_norm_set: float
    ambiguity_ratio: float
    anchored_alpha: float
    anchored_alpha_abs_error: float
    anchored_alpha_log_std: float
    anchored_alpha_log_span: float
    anchored_effective_count: float
    best_alpha: float
    best_alpha_abs_error: float


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
    mean_ambiguity_ratio: float
    mean_anchored_alpha_log_std: float
    mean_anchored_alpha_log_span: float
    mean_anchored_effective_count: float
    best_alpha_bank_log_span: float
    anchored_alpha_bank_log_span: float
    best_alpha_abs_error_mean: float
    anchored_alpha_abs_error_mean: float
    alpha_abs_error_gain: float
    anchored_beats_best_flag: int
    alpha_point_recoverable_flag: int
    alpha_point_unrecoverable_flag: int


@dataclass(frozen=True)
class BankContext:
    seed: int
    params_list: list[tuple[float, float, float, float, float, float]]
    shifted_bank: np.ndarray
    geometries: np.ndarray
    alpha_logs: np.ndarray


def condition_index(condition: str) -> int:
    return FOCUS_CONDITIONS.index(condition)


def skew_index(skew_bin: str) -> int:
    return GEOMETRY_SKEW_BIN_LABELS.index(skew_bin)


def make_trial_rng(observation_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(observation_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


def canonicalize_geometry(params: tuple[float, float, float, float, float, float]) -> tuple[np.ndarray, float]:
    geometry, _, alpha = control_invariants(params)
    swapped = np.array([geometry[0], geometry[2], geometry[1]])
    if tuple(swapped) < tuple(geometry):
        return swapped, float(alpha)
    return geometry, float(alpha)


def build_bank_context(bank_seed: int) -> BankContext:
    bank_rng = np.random.default_rng(bank_seed)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, bank_rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    geometries = []
    alpha_logs = []
    for params in params_list:
        geometry, alpha = canonicalize_geometry(params)
        geometries.append(geometry)
        alpha_logs.append(math.log(alpha))
    return BankContext(
        seed=bank_seed,
        params_list=params_list,
        shifted_bank=shifted_bank,
        geometries=np.array(geometries),
        alpha_logs=np.array(alpha_logs),
    )


def empirical_geometry_ranges(bank_contexts: list[BankContext]) -> np.ndarray:
    all_geometries = np.concatenate([context.geometries for context in bank_contexts], axis=0)
    span = np.max(all_geometries, axis=0) - np.min(all_geometries, axis=0)
    return np.maximum(span, NUMERIC_EPS)


def score_band(regime: dict[str, float | str | int]) -> float:
    return max(float(regime["noise_sigma"]) ** 2, MIN_SCORE_BAND)


def select_candidate_indices(scores: np.ndarray, band: float) -> np.ndarray:
    order = np.argsort(scores)
    best_score = float(scores[order[0]])
    keep = order[scores[order] <= best_score + band]
    if len(keep) == 0:
        return order[:1]
    return keep


def geometry_span_norm(geometries: np.ndarray, geometry_ranges: np.ndarray) -> float:
    span = (np.max(geometries, axis=0) - np.min(geometries, axis=0)) / geometry_ranges
    return float(np.mean(span))


def score_weights(scores: np.ndarray, scale: float) -> np.ndarray:
    offsets = scores - float(np.min(scores))
    weights = np.exp(-offsets / max(scale, NUMERIC_EPS))
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(scores), 1.0 / len(scores))
    return weights / total


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0.0:
        return float(sorted_values[len(sorted_values) // 2])
    cumulative = cumulative / cumulative[-1]
    idx = int(np.searchsorted(cumulative, quantile, side="left"))
    idx = min(max(idx, 0), len(sorted_values) - 1)
    return float(sorted_values[idx])


def anchored_alpha_posterior(
    candidate_scores: np.ndarray,
    candidate_geometries: np.ndarray,
    candidate_alpha_logs: np.ndarray,
    geometry_consensus: np.ndarray,
    geometry_ranges: np.ndarray,
    score_scale: float,
) -> tuple[float, float, float, float]:
    base_weights = score_weights(candidate_scores, score_scale)
    geometry_distance = np.mean(
        np.abs(candidate_geometries - geometry_consensus[None, :]) / geometry_ranges[None, :],
        axis=1,
    )
    geometry_penalty = np.exp(-geometry_distance / GEOMETRY_ANCHOR_SCALE)
    anchored_weights = base_weights * geometry_penalty
    total = float(np.sum(anchored_weights))
    if total <= 0.0:
        anchored_weights = np.full(len(candidate_scores), 1.0 / len(candidate_scores))
    else:
        anchored_weights = anchored_weights / total
    anchored_mean_log = float(np.sum(anchored_weights * candidate_alpha_logs))
    anchored_var_log = float(np.sum(anchored_weights * (candidate_alpha_logs - anchored_mean_log) ** 2))
    anchored_std_log = float(math.sqrt(max(anchored_var_log, 0.0)))
    anchored_span_log = float(
        weighted_quantile(candidate_alpha_logs, anchored_weights, 0.90)
        - weighted_quantile(candidate_alpha_logs, anchored_weights, 0.10)
    )
    effective_count = float(1.0 / np.sum(anchored_weights * anchored_weights))
    return anchored_mean_log, anchored_std_log, anchored_span_log, effective_count


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def balanced_accuracy(rows: list[TrialRow], metric_name: str, threshold: float) -> float:
    labels = np.array([row.alpha_point_unrecoverable_flag for row in rows], dtype=int)
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
    true_alpha = float(true_params[5])

    bank_rows: list[BankRow] = []
    best_entropies = []
    candidate_counts = []
    alpha_log_spans = []
    geometry_set_spans = []
    ambiguity_ratios = []
    best_alpha_logs = []
    anchored_alpha_logs = []
    best_alpha_errors = []
    anchored_alpha_errors = []
    anchored_alpha_log_stds = []
    anchored_alpha_log_spans = []
    anchored_effective_counts = []

    for context in bank_contexts:
        scores, _ = marginalized_bank_scores(observed_signature, mask, context.shifted_bank, temperature)
        best_idx = int(np.argmin(scores))
        candidate_indices = select_candidate_indices(scores, band)
        candidate_scores = scores[candidate_indices]
        candidate_geometries = context.geometries[candidate_indices]
        candidate_alpha_logs = context.alpha_logs[candidate_indices]

        geometry_weights = score_weights(candidate_scores, band)
        geometry_consensus = np.sum(candidate_geometries * geometry_weights[:, None], axis=0)
        geometry_span = geometry_span_norm(candidate_geometries, geometry_ranges)
        alpha_log_span = float(np.max(candidate_alpha_logs) - np.min(candidate_alpha_logs))
        ambiguity_ratio = float(alpha_log_span / max(geometry_span, NUMERIC_EPS))

        anchored_mean_log, anchored_std_log, anchored_span_log, anchored_effective_count = anchored_alpha_posterior(
            candidate_scores,
            candidate_geometries,
            candidate_alpha_logs,
            geometry_consensus,
            geometry_ranges,
            band,
        )

        best_alpha_log = float(context.alpha_logs[best_idx])
        best_alpha = float(math.exp(best_alpha_log))
        anchored_alpha = float(math.exp(anchored_mean_log))
        best_alpha_error = float(abs(best_alpha - true_alpha))
        anchored_alpha_error = float(abs(anchored_alpha - true_alpha))

        solver_context = SolverContext(observed_signature, mask)
        _, _, _, best_entropy = solver_context.score_params(context.params_list[best_idx], temperature)

        best_entropies.append(float(best_entropy))
        candidate_counts.append(int(len(candidate_indices)))
        alpha_log_spans.append(alpha_log_span)
        geometry_set_spans.append(geometry_span)
        ambiguity_ratios.append(ambiguity_ratio)
        best_alpha_logs.append(best_alpha_log)
        anchored_alpha_logs.append(anchored_mean_log)
        best_alpha_errors.append(best_alpha_error)
        anchored_alpha_errors.append(anchored_alpha_error)
        anchored_alpha_log_stds.append(anchored_std_log)
        anchored_alpha_log_spans.append(anchored_span_log)
        anchored_effective_counts.append(anchored_effective_count)

        bank_rows.append(
            BankRow(
                split=split,
                observation_seed=int(observation_seed),
                bank_seed=int(context.seed),
                condition=condition,
                geometry_skew_bin=skew_bin,
                true_alpha=true_alpha,
                true_t=float(true_params[1]),
                true_rotation_shift=int(true_shift),
                best_entropy=float(best_entropy),
                candidate_count=int(len(candidate_indices)),
                alpha_log_span_set=alpha_log_span,
                geometry_span_norm_set=geometry_span,
                ambiguity_ratio=ambiguity_ratio,
                anchored_alpha=anchored_alpha,
                anchored_alpha_abs_error=anchored_alpha_error,
                anchored_alpha_log_std=anchored_std_log,
                anchored_alpha_log_span=anchored_span_log,
                anchored_effective_count=anchored_effective_count,
                best_alpha=best_alpha,
                best_alpha_abs_error=best_alpha_error,
            )
        )

    anchored_alpha_bank_log_span = float(np.max(anchored_alpha_logs) - np.min(anchored_alpha_logs))
    anchored_alpha_abs_error_mean = float(np.mean(anchored_alpha_errors))
    alpha_point_recoverable_flag = int(
        anchored_alpha_bank_log_span < ALPHA_STABLE_LOG_SPAN_THRESHOLD
        and anchored_alpha_abs_error_mean < ALPHA_POINT_ABS_ERROR_THRESHOLD
    )

    trial_row = TrialRow(
        split=split,
        observation_seed=int(observation_seed),
        condition=condition,
        geometry_skew_bin=skew_bin,
        true_alpha=true_alpha,
        true_t=float(true_params[1]),
        true_rotation_shift=int(true_shift),
        mean_best_entropy=float(np.mean(best_entropies)),
        mean_candidate_count=float(np.mean(candidate_counts)),
        mean_alpha_log_span_set=float(np.mean(alpha_log_spans)),
        mean_geometry_span_norm_set=float(np.mean(geometry_set_spans)),
        mean_ambiguity_ratio=float(np.mean(ambiguity_ratios)),
        mean_anchored_alpha_log_std=float(np.mean(anchored_alpha_log_stds)),
        mean_anchored_alpha_log_span=float(np.mean(anchored_alpha_log_spans)),
        mean_anchored_effective_count=float(np.mean(anchored_effective_counts)),
        best_alpha_bank_log_span=float(np.max(best_alpha_logs) - np.min(best_alpha_logs)),
        anchored_alpha_bank_log_span=anchored_alpha_bank_log_span,
        best_alpha_abs_error_mean=float(np.mean(best_alpha_errors)),
        anchored_alpha_abs_error_mean=anchored_alpha_abs_error_mean,
        alpha_abs_error_gain=float(np.mean(best_alpha_errors) - anchored_alpha_abs_error_mean),
        anchored_beats_best_flag=int(anchored_alpha_abs_error_mean <= float(np.mean(best_alpha_errors))),
        alpha_point_recoverable_flag=alpha_point_recoverable_flag,
        alpha_point_unrecoverable_flag=int(1 - alpha_point_recoverable_flag),
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
                "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in subset])),
                "mean_anchored_alpha_log_std": float(np.mean([row.mean_anchored_alpha_log_std for row in subset])),
                "mean_anchored_alpha_bank_log_span": float(np.mean([row.anchored_alpha_bank_log_span for row in subset])),
                "mean_best_alpha_bank_log_span": float(np.mean([row.best_alpha_bank_log_span for row in subset])),
                "mean_anchored_alpha_abs_error": float(np.mean([row.anchored_alpha_abs_error_mean for row in subset])),
                "mean_best_alpha_abs_error": float(np.mean([row.best_alpha_abs_error_mean for row in subset])),
                "mean_alpha_abs_error_gain": float(np.mean([row.alpha_abs_error_gain for row in subset])),
                "anchored_beats_best_rate": float(np.mean([row.anchored_beats_best_flag for row in subset])),
                "anchored_std_vs_unrecoverable_corr": safe_corr(
                    [row.mean_anchored_alpha_log_std for row in subset],
                    [row.alpha_point_unrecoverable_flag for row in subset],
                ),
                "ambiguity_ratio_vs_unrecoverable_corr": safe_corr(
                    [row.mean_ambiguity_ratio for row in subset],
                    [row.alpha_point_unrecoverable_flag for row in subset],
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
                    "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in subset])),
                    "mean_anchored_alpha_log_std": float(np.mean([row.mean_anchored_alpha_log_std for row in subset])),
                    "mean_anchored_alpha_bank_log_span": float(np.mean([row.anchored_alpha_bank_log_span for row in subset])),
                    "mean_anchored_alpha_abs_error": float(np.mean([row.anchored_alpha_abs_error_mean for row in subset])),
                    "anchored_beats_best_rate": float(np.mean([row.anchored_beats_best_flag for row in subset])),
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
                        "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in subset])),
                        "mean_anchored_alpha_log_std": float(np.mean([row.mean_anchored_alpha_log_std for row in subset])),
                        "mean_anchored_alpha_bank_log_span": float(np.mean([row.anchored_alpha_bank_log_span for row in subset])),
                        "mean_anchored_alpha_abs_error": float(np.mean([row.anchored_alpha_abs_error_mean for row in subset])),
                        "anchored_beats_best_rate": float(np.mean([row.anchored_beats_best_flag for row in subset])),
                    }
                )
    return summary


def evaluate_threshold_rules(rows: list[TrialRow]) -> dict[str, object]:
    calibration_rows = [row for row in rows if row.split == "calibration"]
    holdout_rows = [row for row in rows if row.split == "holdout"]
    confirmation_rows = [row for row in rows if row.split == "confirmation"]
    metrics = [
        "mean_anchored_alpha_log_std",
        "mean_anchored_alpha_log_span",
        "mean_ambiguity_ratio",
        "mean_alpha_log_span_set",
        "mean_best_entropy",
    ]
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
    best_oos_metric = max(
        results,
        key=lambda item: (
            0.5 * (item["holdout_balanced_accuracy"] + item["confirmation_balanced_accuracy"]),
            item["confirmation_balanced_accuracy"],
        ),
    )
    return {"metrics": results, "best_by_calibration": best_metric, "best_by_oos_mean": best_oos_metric}


def plot_gate_scatter(path: str, rows: list[TrialRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.97)
    palette = {"calibration": "#1d3557", "holdout": "#2a9d8f", "confirmation": "#e76f51"}
    for split in BLOCK_SPECS:
        subset = [row for row in rows if row.split == split]
        ax.scatter(
            [row.mean_anchored_alpha_log_std for row in subset],
            [row.anchored_alpha_bank_log_span for row in subset],
            s=56,
            alpha=0.82,
            label=split,
            color=palette[split],
        )
    xs = np.array([row.mean_anchored_alpha_log_std for row in rows], dtype=float)
    ys = np.array([row.anchored_alpha_bank_log_span for row in rows], dtype=float)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
    ax.plot(grid, slope * grid + intercept, color="#111111", lw=1.6, linestyle=":")
    ax.axhline(ALPHA_STABLE_LOG_SPAN_THRESHOLD, color="#444444", lw=1.2, linestyle="--")
    ax.set_xlabel("mean anchored alpha log std")
    ax.set_ylabel("anchored alpha bank log span")
    ax.set_title("Backbone-anchored alpha uncertainty vs cross-bank alpha stability")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_error_bars(path: str, split_summary: list[dict[str, float | str]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([float(item["mean_best_alpha_abs_error"]) for item in split_summary], dtype=float)
    anchored = np.array([float(item["mean_anchored_alpha_abs_error"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - width / 2.0, best, width=width, color="#9c6644", label="best candidate alpha")
    ax.bar(x + width / 2.0, anchored, width=width, color="#2a9d8f", label="anchored alpha mean")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Anchored alpha estimate vs best-candidate alpha")
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

    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.24, left=0.10, right=0.97)
    ax.bar(x - width, calibration, width=width, color="#1d3557", label="calibration")
    ax.bar(x, holdout, width=width, color="#2a9d8f", label="holdout")
    ax.bar(x + width, confirmation, width=width, color="#e76f51", label="confirmation")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=16, ha="right")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Layer-2 gate metrics for alpha point unrecoverability")
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
        "geometry_anchor_scale": GEOMETRY_ANCHOR_SCALE,
        "alpha_stable_log_span_threshold": ALPHA_STABLE_LOG_SPAN_THRESHOLD,
        "alpha_point_abs_error_threshold": ALPHA_POINT_ABS_ERROR_THRESHOLD,
        "trial_count": len(trial_rows),
        "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in trial_rows])),
        "mean_anchored_alpha_log_std": float(np.mean([row.mean_anchored_alpha_log_std for row in trial_rows])),
        "mean_anchored_alpha_bank_log_span": float(np.mean([row.anchored_alpha_bank_log_span for row in trial_rows])),
        "mean_best_alpha_bank_log_span": float(np.mean([row.best_alpha_bank_log_span for row in trial_rows])),
        "mean_anchored_alpha_abs_error": float(np.mean([row.anchored_alpha_abs_error_mean for row in trial_rows])),
        "mean_best_alpha_abs_error": float(np.mean([row.best_alpha_abs_error_mean for row in trial_rows])),
        "mean_alpha_abs_error_gain": float(np.mean([row.alpha_abs_error_gain for row in trial_rows])),
        "anchored_beats_best_rate": float(np.mean([row.anchored_beats_best_flag for row in trial_rows])),
        "anchored_std_vs_unrecoverable_corr": safe_corr(
            [row.mean_anchored_alpha_log_std for row in trial_rows],
            [row.alpha_point_unrecoverable_flag for row in trial_rows],
        ),
        "ambiguity_ratio_vs_unrecoverable_corr": safe_corr(
            [row.mean_ambiguity_ratio for row in trial_rows],
            [row.alpha_point_unrecoverable_flag for row in trial_rows],
        ),
        "threshold_rule": threshold_payload,
    }

    output_payload = {
        "summary": global_summary,
        "by_split": split_summary,
        "by_condition": condition_summary,
        "by_cell": cell_summary,
    }

    write_csv(os.path.join(OUTPUT_DIR, "backbone_observability_gate_bank_rows.csv"), [asdict(row) for row in bank_rows])
    write_csv(os.path.join(OUTPUT_DIR, "backbone_observability_gate_trials.csv"), [asdict(row) for row in trial_rows])
    write_csv(os.path.join(OUTPUT_DIR, "backbone_observability_gate_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, "backbone_observability_gate_condition_summary.csv"), condition_summary)
    write_csv(os.path.join(OUTPUT_DIR, "backbone_observability_gate_cell_summary.csv"), cell_summary)
    with open(os.path.join(OUTPUT_DIR, "backbone_observability_gate_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    plot_gate_scatter(
        os.path.join(FIGURE_DIR, "backbone_observability_gate_scatter.png"),
        trial_rows,
    )
    plot_alpha_error_bars(
        os.path.join(FIGURE_DIR, "backbone_observability_gate_alpha_error.png"),
        split_summary,
    )
    plot_threshold_bars(
        os.path.join(FIGURE_DIR, "backbone_observability_gate_thresholds.png"),
        threshold_payload,
    )

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
