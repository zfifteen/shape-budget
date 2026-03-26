"""
Specialized Layer 2 ratio sweep for the informed-bank regime.

This script fills the calibration gap for the informed bank only, then sweeps
candidate-aware Layer 2 ratios against the completed informed-bank trial table.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

(
    BANK_SEEDS,
    GEOMETRY_SKEW_BIN_LABELS,
    FOCUS_ALPHA_BIN,
    sample_conditioned_parameters,
    observe_pose_free_signature,
    anisotropic_forward_signature,
    OBSERVATION_REGIMES,
    marginalized_bank_scores,
    softmin_temperature,
    make_trial_rng,
    score_band,
    geometry_span_norm,
    global_geometry_ranges,
    build_informed_method_context,
    capture_candidate_indices,
    anchored_alpha_posterior,
) = load_symbols(
    "run_informed_gate_specialized_sweep",
    ROOT / "experiments/pose-anisotropy-diagnostics/persistent-mode-bank-candidate-atlas/run.py",
    "BANK_SEEDS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "FOCUS_ALPHA_BIN",
    "sample_conditioned_parameters",
    "observe_pose_free_signature",
    "anisotropic_forward_signature",
    "OBSERVATION_REGIMES",
    "marginalized_bank_scores",
    "softmin_temperature",
    "make_trial_rng",
    "score_band",
    "geometry_span_norm",
    "global_geometry_ranges",
    "build_informed_method_context",
    "capture_candidate_indices",
    "anchored_alpha_posterior",
)

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EXISTING_TRIALS_PATH = (
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/"
    "backbone_observability_gate_informed_bank_trials.csv"
)
LEADERBOARD_CSV = OUTPUT_DIR / "backbone_observability_gate_informed_bank_specialized_ratio_sweep_leaderboard.csv"
SUMMARY_JSON = OUTPUT_DIR / "backbone_observability_gate_informed_bank_specialized_ratio_sweep_summary.json"

METHOD = "persistent_mode_informed"
TARGET_CONDITION = "sparse_partial_high_noise"
CALIBRATION_SEEDS = (20260410, 20260411, 20260412, 20260416, 20260417, 20260418)
ALPHA_STABLE_LOG_SPAN_THRESHOLD = 0.20
ALPHA_POINT_ABS_ERROR_THRESHOLD = 0.15
EPS = 1.0e-9


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def calibration_rows() -> pd.DataFrame:
    geometry_ranges = global_geometry_ranges()
    regime = next(item for item in OBSERVATION_REGIMES if item["name"] == TARGET_CONDITION)
    rows: list[dict[str, object]] = []

    for observation_seed in CALIBRATION_SEEDS:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            trial_rng = make_trial_rng(observation_seed, TARGET_CONDITION, skew_bin)
            true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
            true_alpha = float(true_params[5])
            clean_signature = anisotropic_forward_signature(true_params)
            _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
            temperature = softmin_temperature(regime)
            band = score_band(regime)

            best_alpha_logs: list[float] = []
            anchored_alpha_logs: list[float] = []
            best_alpha_errors: list[float] = []
            anchored_alpha_errors: list[float] = []
            anchored_alpha_stds: list[float] = []
            anchored_alpha_spans: list[float] = []
            anchored_effective_counts: list[float] = []
            candidate_counts: list[int] = []
            alpha_log_span_sets: list[float] = []
            geometry_span_sets: list[float] = []
            ambiguity_ratios: list[float] = []

            for bank_seed_value in BANK_SEEDS:
                context, _ = build_informed_method_context(
                    int(bank_seed_value),
                    int(observation_seed),
                    TARGET_CONDITION,
                    skew_bin,
                    observed_signature,
                    mask,
                    band,
                    temperature,
                    geometry_ranges,
                )
                scores, _ = marginalized_bank_scores(observed_signature, mask, context.shifted_bank, temperature)
                _captured_indices, band_set, _frontier_set, order = capture_candidate_indices(scores, band)
                rank_lookup = {int(idx): rank for rank, idx in enumerate(order)}
                band_indices = np.array(sorted(band_set, key=lambda idx: rank_lookup[int(idx)]), dtype=int)
                band_scores = scores[band_indices]
                band_geometries = context.geometries[band_indices]
                band_alpha_logs = context.alpha_logs[band_indices]

                geometry_weights = np.exp(-(band_scores - float(np.min(band_scores))) / max(band, EPS))
                geometry_weights = geometry_weights / max(float(np.sum(geometry_weights)), EPS)
                geometry_consensus = np.sum(band_geometries * geometry_weights[:, None], axis=0)
                anchored_weights, anchored_mean_log, anchored_std_log, anchored_effective_count = anchored_alpha_posterior(
                    band_scores,
                    band_geometries,
                    band_alpha_logs,
                    geometry_consensus,
                    geometry_ranges,
                    band,
                )
                anchored_span_log = float(
                    weighted_quantile(band_alpha_logs, anchored_weights, 0.90)
                    - weighted_quantile(band_alpha_logs, anchored_weights, 0.10)
                )
                best_idx = int(order[0])
                best_alpha_log = float(context.alpha_logs[best_idx])
                best_alpha = float(math.exp(best_alpha_log))
                anchored_alpha = float(math.exp(anchored_mean_log))
                geometry_span = float(geometry_span_norm(band_geometries, geometry_ranges))
                alpha_log_span = float(np.max(band_alpha_logs) - np.min(band_alpha_logs))
                ambiguity_ratio = float(alpha_log_span / max(geometry_span, EPS))

                best_alpha_logs.append(best_alpha_log)
                anchored_alpha_logs.append(anchored_mean_log)
                best_alpha_errors.append(float(abs(best_alpha - true_alpha)))
                anchored_alpha_errors.append(float(abs(anchored_alpha - true_alpha)))
                anchored_alpha_stds.append(float(anchored_std_log))
                anchored_alpha_spans.append(anchored_span_log)
                anchored_effective_counts.append(float(anchored_effective_count))
                candidate_counts.append(int(len(band_indices)))
                alpha_log_span_sets.append(alpha_log_span)
                geometry_span_sets.append(geometry_span)
                ambiguity_ratios.append(ambiguity_ratio)

            best_alpha_bank_log_span = float(np.max(best_alpha_logs) - np.min(best_alpha_logs))
            anchored_alpha_bank_log_span = float(np.max(anchored_alpha_logs) - np.min(anchored_alpha_logs))
            best_alpha_abs_error_mean = float(np.mean(best_alpha_errors))
            anchored_alpha_abs_error_mean = float(np.mean(anchored_alpha_errors))
            alpha_point_recoverable_flag = int(
                anchored_alpha_bank_log_span < ALPHA_STABLE_LOG_SPAN_THRESHOLD
                and anchored_alpha_abs_error_mean < ALPHA_POINT_ABS_ERROR_THRESHOLD
            )

            row = {
                "method": METHOD,
                "split": "calibration",
                "observation_seed": int(observation_seed),
                "condition": TARGET_CONDITION,
                "geometry_skew_bin": skew_bin,
                "true_alpha": true_alpha,
                "true_t": float(true_params[1]),
                "true_rotation_shift": int(true_shift),
                "mean_best_entropy": float("nan"),
                "mean_candidate_count": float(np.mean(candidate_counts)),
                "mean_alpha_log_span_set": float(np.mean(alpha_log_span_sets)),
                "mean_geometry_span_norm_set": float(np.mean(geometry_span_sets)),
                "mean_ambiguity_ratio": float(np.mean(ambiguity_ratios)),
                "mean_anchored_alpha_log_std": float(np.mean(anchored_alpha_stds)),
                "mean_anchored_alpha_log_span": float(np.mean(anchored_alpha_spans)),
                "mean_anchored_effective_count": float(np.mean(anchored_effective_counts)),
                "best_alpha_bank_log_span": best_alpha_bank_log_span,
                "anchored_alpha_bank_log_span": anchored_alpha_bank_log_span,
                "best_alpha_abs_error_mean": best_alpha_abs_error_mean,
                "anchored_alpha_abs_error_mean": anchored_alpha_abs_error_mean,
            }
            row["alpha_abs_error_gain"] = float(best_alpha_abs_error_mean - anchored_alpha_abs_error_mean)
            row["anchored_beats_best_flag"] = int(anchored_alpha_abs_error_mean <= best_alpha_abs_error_mean)
            row["alpha_point_recoverable_flag"] = alpha_point_recoverable_flag
            row["alpha_point_unrecoverable_flag"] = int(1 - alpha_point_recoverable_flag)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["ratio_anchored_span_over_std"] = df["mean_anchored_alpha_log_span"] / (df["mean_anchored_alpha_log_std"] + EPS)
    df["ratio_candidate_times_anchored_span_over_std"] = (
        df["mean_candidate_count"] * df["mean_anchored_alpha_log_span"] / (df["mean_anchored_alpha_log_std"] + EPS)
    )
    return df


def build_ratio_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["metric_std"] = out["mean_anchored_alpha_log_std"]
    out["metric_ambiguity"] = out["mean_ambiguity_ratio"]
    out["metric_set_span"] = out["mean_alpha_log_span_set"]
    out["metric_anchored_span"] = out["mean_anchored_alpha_log_span"]
    out["ratio_std_over_set_span"] = out["mean_anchored_alpha_log_std"] / (out["mean_alpha_log_span_set"] + EPS)
    out["ratio_std_over_ambiguity"] = out["mean_anchored_alpha_log_std"] / (out["mean_ambiguity_ratio"] + EPS)
    out["ratio_ambiguity_over_std"] = out["mean_ambiguity_ratio"] / (out["mean_anchored_alpha_log_std"] + EPS)
    out["ratio_effective_over_count"] = out["mean_anchored_effective_count"] / (out["mean_candidate_count"] + EPS)
    out["ratio_count_over_effective"] = out["mean_candidate_count"] / (out["mean_anchored_effective_count"] + EPS)
    out["ratio_std_times_effective_over_count"] = (
        out["mean_anchored_alpha_log_std"] * out["mean_anchored_effective_count"] / (out["mean_candidate_count"] + EPS)
    )
    out["ratio_anchored_span_over_std"] = out["mean_anchored_alpha_log_span"] / (out["mean_anchored_alpha_log_std"] + EPS)
    out["ratio_candidate_times_anchored_span_over_std"] = (
        out["mean_candidate_count"] * out["mean_anchored_alpha_log_span"] / (out["mean_anchored_alpha_log_std"] + EPS)
    )
    out["ratio_effective_times_anchored_span_over_std"] = (
        out["mean_anchored_effective_count"] * out["mean_anchored_alpha_log_span"] / (out["mean_anchored_alpha_log_std"] + EPS)
    )
    out["ratio_candidate_times_std_over_effective"] = (
        out["mean_candidate_count"] * out["mean_anchored_alpha_log_std"] / (out["mean_anchored_effective_count"] + EPS)
    )
    out["ratio_candidate_times_ambiguity_over_effective"] = (
        out["mean_candidate_count"] * out["mean_ambiguity_ratio"] / (out["mean_anchored_effective_count"] + EPS)
    )
    out["ratio_candidate_times_std_over_ambiguity"] = (
        out["mean_candidate_count"] * out["mean_anchored_alpha_log_std"] / (out["mean_ambiguity_ratio"] + EPS)
    )
    out["ratio_candidate_times_span_over_effective"] = (
        out["mean_candidate_count"] * out["mean_anchored_alpha_log_span"] / (out["mean_anchored_effective_count"] + EPS)
    )
    out["ratio_geomspan_over_std"] = out["mean_geometry_span_norm_set"] / (out["mean_anchored_alpha_log_std"] + EPS)
    out["ratio_geomspan_times_count_over_effective"] = (
        out["mean_geometry_span_norm_set"] * out["mean_candidate_count"] / (out["mean_anchored_effective_count"] + EPS)
    )
    return out


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("metric_") or c.startswith("ratio_")]


def balanced_accuracy(labels: np.ndarray, values: np.ndarray, threshold: float, direction: str) -> float:
    preds = (values >= threshold).astype(int) if direction == "ge" else (values <= threshold).astype(int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = float(np.sum((preds == 1) & (labels == 1)) / positives)
    tnr = float(np.sum((preds == 0) & (labels == 0)) / negatives)
    return 0.5 * (tpr + tnr)


def choose_threshold(values: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    unique = sorted({float(v) for v in values})
    candidates = [unique[0] - 1.0e-6]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(unique[:-1], unique[1:]))
    candidates.append(unique[-1] + 1.0e-6)
    best = {"threshold": 0.0, "direction": "ge", "calibration_balanced_accuracy": -1.0}
    for direction in ("ge", "le"):
        for threshold in candidates:
            score = balanced_accuracy(labels, values, threshold, direction)
            if score > best["calibration_balanced_accuracy"]:
                best = {
                    "threshold": float(threshold),
                    "direction": direction,
                    "calibration_balanced_accuracy": float(score),
                }
    return best


def main() -> None:
    existing = pd.read_csv(EXISTING_TRIALS_PATH)
    existing = existing[(existing["method"] == METHOD) & (existing["split"].isin(["holdout", "confirmation"]))].copy()
    calibration = calibration_rows()
    df = pd.concat([calibration, existing], ignore_index=True)
    ratio_df = build_ratio_frame(df)

    calibration_df = ratio_df[ratio_df["split"] == "calibration"]
    holdout_df = ratio_df[ratio_df["split"] == "holdout"]
    confirmation_df = ratio_df[ratio_df["split"] == "confirmation"]

    leaderboard: list[dict[str, object]] = []
    labels_cal = calibration_df["alpha_point_unrecoverable_flag"].to_numpy(dtype=int)
    labels_holdout = holdout_df["alpha_point_unrecoverable_flag"].to_numpy(dtype=int)
    labels_confirmation = confirmation_df["alpha_point_unrecoverable_flag"].to_numpy(dtype=int)

    for metric in metric_columns(ratio_df):
        values_cal = calibration_df[metric].to_numpy(dtype=float)
        chosen = choose_threshold(values_cal, labels_cal)
        threshold = float(chosen["threshold"])
        direction = str(chosen["direction"])
        holdout_score = balanced_accuracy(labels_holdout, holdout_df[metric].to_numpy(dtype=float), threshold, direction)
        confirmation_score = balanced_accuracy(
            labels_confirmation,
            confirmation_df[metric].to_numpy(dtype=float),
            threshold,
            direction,
        )
        leaderboard.append(
            {
                "metric": metric,
                "threshold": threshold,
                "direction": direction,
                "calibration_balanced_accuracy": float(chosen["calibration_balanced_accuracy"]),
                "holdout_balanced_accuracy": float(holdout_score),
                "confirmation_balanced_accuracy": float(confirmation_score),
                "oos_mean_balanced_accuracy": float(0.5 * (holdout_score + confirmation_score)),
            }
        )

    leaderboard.sort(
        key=lambda item: (
            round(float(item["oos_mean_balanced_accuracy"]), 6),
            float(item["confirmation_balanced_accuracy"]),
            float(item["calibration_balanced_accuracy"]),
        ),
        reverse=True,
    )
    write_csv(LEADERBOARD_CSV, leaderboard)

    summary = {
        "experiment": "backbone-observability-gate-informed-bank-specialized-ratio-sweep",
        "method": METHOD,
        "target_condition": TARGET_CONDITION,
        "calibration_count": int(len(calibration_df)),
        "holdout_count": int(len(holdout_df)),
        "confirmation_count": int(len(confirmation_df)),
        "metric_count": int(len(metric_columns(ratio_df))),
        "best_metric": leaderboard[0],
        "top5": leaderboard[:5],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
