"""
Layer 2 informed-bank comparison derived from the persistent-mode bank atlas.

This experiment answers the next concrete question in the layered solver stack:

- after Layer 1 improved the candidate family with the informed bank,
  what happens to the Layer 2 observability gate?

Instead of rerunning the full expensive bank build, this script derives the
Layer 2 trial rows directly from the informed-bank candidate atlas and compares
them against:

- the same-run random baseline from the atlas
- the legacy 5-seed Layer 2 hard-branch run

The key comparison uses the frozen legacy Layer 2 gate threshold on
holdout/confirmation, then follows with threshold-free separation probes to
check whether the signal merely shifted or changed direction.
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FC_CACHEDIR", "/tmp/fontconfig")

import matplotlib
import numpy as np
import pandas as pd
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

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

(
    BANK_SEEDS,
    GEOMETRY_SKEW_BIN_LABELS,
    FOCUS_ALPHA_BIN,
    sample_conditioned_parameters,
    observe_pose_free_signature,
    anisotropic_forward_signature,
    OBSERVATION_REGIMES_FULL,
    marginalized_bank_scores,
    softmin_temperature,
    make_trial_rng,
    score_band,
    geometry_span_norm,
    global_geometry_ranges,
    build_random_method_context,
    build_informed_method_context,
    capture_candidate_indices,
    anchored_alpha_posterior,
) = load_symbols(
    "run_persistent_mode_bank_candidate_atlas_for_gate_calibration",
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
    "build_random_method_context",
    "build_informed_method_context",
    "capture_candidate_indices",
    "anchored_alpha_posterior",
)

ATLAS_ROWS_PATH = (
    BASE_DIR.parents[1]
    / "pose-anisotropy-diagnostics/persistent-mode-bank-candidate-atlas/outputs"
    / "persistent_mode_bank_candidate_atlas_rows.csv"
)
ATLAS_BANK_PATH = (
    BASE_DIR.parents[1]
    / "pose-anisotropy-diagnostics/persistent-mode-bank-candidate-atlas/outputs"
    / "persistent_mode_bank_candidate_atlas_bank_summary.csv"
)
LEGACY_TRIALS_PATH = (
    BASE_DIR.parent / "backbone-observability-gate/outputs/backbone_observability_gate_trials.csv"
)
LEGACY_SUMMARY_PATH = (
    BASE_DIR.parent / "backbone-observability-gate/outputs/backbone_observability_gate_summary.json"
)
RATIO_SWEEP_SUMMARY_PATH = (
    BASE_DIR.parent
    / "backbone-observability-gate-ratio-sweep/outputs/backbone_observability_gate_ratio_sweep_summary.json"
)

METHOD_RANDOM = "one_shot_random"
METHOD_INFORMED = "persistent_mode_informed"
LEGACY_METHOD = "legacy_random_5seed"
TARGET_CONDITION = "sparse_partial_high_noise"
CALIBRATION_SEEDS = (20260410, 20260411, 20260412, 20260416, 20260417, 20260418)

ALPHA_STABLE_LOG_SPAN_THRESHOLD = 0.20
ALPHA_POINT_ABS_ERROR_THRESHOLD = 0.15

TRIALS_CSV = OUTPUT_DIR / "backbone_observability_gate_informed_bank_trials.csv"
SPLIT_CSV = OUTPUT_DIR / "backbone_observability_gate_informed_bank_split_summary.csv"
CELL_CSV = OUTPUT_DIR / "backbone_observability_gate_informed_bank_cell_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "backbone_observability_gate_informed_bank_summary.json"
STD_SCATTER_PNG = FIGURE_DIR / "backbone_observability_gate_informed_bank_std_scatter.png"
THRESHOLD_BAR_PNG = FIGURE_DIR / "backbone_observability_gate_informed_bank_threshold_bars.png"


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


def derive_calibration_rows() -> pd.DataFrame:
    geometry_ranges = global_geometry_ranges()
    regime = next(item for item in OBSERVATION_REGIMES_FULL if item["name"] == TARGET_CONDITION)
    trial_rows: list[dict[str, object]] = []

    for method in (METHOD_RANDOM, METHOD_INFORMED):
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
                    if method == METHOD_RANDOM:
                        context, _ = build_random_method_context(
                            int(bank_seed_value),
                            int(observation_seed),
                            TARGET_CONDITION,
                            skew_bin,
                        )
                    else:
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
                    if len(band_indices) == 0:
                        continue
                    band_scores = scores[band_indices]
                    band_geometries = context.geometries[band_indices]
                    band_alpha_logs = context.alpha_logs[band_indices]
                    geometry_weights = np.exp(-(band_scores - float(np.min(band_scores))) / max(band, 1.0e-9))
                    geometry_weights = geometry_weights / max(float(np.sum(geometry_weights)), 1.0e-12)
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
                    ambiguity_ratio = float(alpha_log_span / max(geometry_span, 1.0e-9))

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
                    "method": method,
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
                row["ratio_anchored_span_over_std"] = row["mean_anchored_alpha_log_span"] / (row["mean_anchored_alpha_log_std"] + 1.0e-9)
                row["ratio_candidate_times_anchored_span_over_std"] = (
                    row["mean_candidate_count"] * row["mean_anchored_alpha_log_span"] / (row["mean_anchored_alpha_log_std"] + 1.0e-9)
                )
                trial_rows.append(row)

    return pd.DataFrame(trial_rows)


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def balanced_accuracy(df: pd.DataFrame, metric: str, threshold: float, direction: str = "ge") -> float:
    labels = df["alpha_point_unrecoverable_flag"].to_numpy(dtype=int)
    values = df[metric].to_numpy(dtype=float)
    if direction == "ge":
        preds = (values >= threshold).astype(int)
    else:
        preds = (values <= threshold).astype(int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = float(np.sum((preds == 1) & (labels == 1)) / positives)
    tnr = float(np.sum((preds == 0) & (labels == 0)) / negatives)
    return 0.5 * (tpr + tnr)


def best_threshold(df: pd.DataFrame, metric: str) -> dict[str, object]:
    values = sorted({float(item) for item in df[metric]})
    candidates = [values[0] - 1.0e-6]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(values[:-1], values[1:]))
    candidates.append(values[-1] + 1.0e-6)
    best_score = -1.0
    best_payload: dict[str, object] = {}
    for direction in ("ge", "le"):
        for threshold in candidates:
            score = balanced_accuracy(df, metric, threshold, direction)
            if score > best_score:
                best_score = score
                best_payload = {
                    "metric": metric,
                    "threshold": float(threshold),
                    "direction": direction,
                    "balanced_accuracy": float(score),
                }
    return best_payload


def derive_trial_rows() -> pd.DataFrame:
    atlas_rows = pd.read_csv(ATLAS_ROWS_PATH)
    atlas_rows = atlas_rows[
        (atlas_rows["condition"] == TARGET_CONDITION) & (atlas_rows["in_band_flag"] == 1)
    ].copy()
    atlas_bank = pd.read_csv(ATLAS_BANK_PATH)
    atlas_bank = atlas_bank[atlas_bank["condition"] == TARGET_CONDITION].copy()

    group_keys = ["method", "split", "observation_seed", "condition", "geometry_skew_bin"]
    bank_keys = group_keys + ["bank_seed"]

    per_bank_rows: list[dict[str, object]] = []
    for key, subset in atlas_rows.groupby(bank_keys, sort=False):
        best = subset.sort_values("rank_by_score").iloc[0]
        weights = subset["anchored_weight_layer2"].to_numpy(dtype=float)
        alpha_logs = subset["log_alpha"].to_numpy(dtype=float)
        anchored_mean_log = float(np.sum(weights * alpha_logs))
        anchored_span_log = float(
            weighted_quantile(alpha_logs, weights, 0.90) - weighted_quantile(alpha_logs, weights, 0.10)
        )
        per_bank_rows.append(
            {
                "method": key[0],
                "split": key[1],
                "observation_seed": int(key[2]),
                "condition": key[3],
                "geometry_skew_bin": key[4],
                "bank_seed": int(key[5]),
                "true_alpha": float(subset["true_alpha"].iloc[0]),
                "best_alpha_log": float(best["log_alpha"]),
                "best_alpha_abs_error": float(abs(float(best["alpha"]) - float(subset["true_alpha"].iloc[0]))),
                "anchored_alpha_log": anchored_mean_log,
                "anchored_alpha_abs_error": float(abs(math.exp(anchored_mean_log) - float(subset["true_alpha"].iloc[0]))),
                "anchored_alpha_log_span": anchored_span_log,
                "anchored_effective_count": float(1.0 / np.sum(weights * weights)),
            }
        )
    per_bank = pd.DataFrame(per_bank_rows)
    merged = atlas_bank.merge(per_bank, on=bank_keys + ["true_alpha"], how="left")

    trial_rows: list[dict[str, object]] = []
    for key, subset in merged.groupby(group_keys, sort=False):
        best_alpha_logs = subset["best_alpha_log"].to_numpy(dtype=float)
        anchored_alpha_logs = subset["anchored_alpha_log"].to_numpy(dtype=float)
        best_alpha_errors = subset["best_alpha_abs_error"].to_numpy(dtype=float)
        anchored_alpha_errors = subset["anchored_alpha_abs_error"].to_numpy(dtype=float)
        row = {
            "method": key[0],
            "split": key[1],
            "observation_seed": int(key[2]),
            "condition": key[3],
            "geometry_skew_bin": key[4],
            "true_alpha": float(subset["true_alpha"].iloc[0]),
            "true_t": float(subset["true_t"].iloc[0]),
            "true_rotation_shift": int(subset["true_rotation_shift"].iloc[0]),
            "mean_best_entropy": float("nan"),
            "mean_candidate_count": float(subset["mean_band_candidate_count"].mean()),
            "mean_alpha_log_span_set": float(subset["mean_alpha_log_span_set"].mean()),
            "mean_geometry_span_norm_set": float(subset["mean_geometry_span_norm_set"].mean()),
            "mean_ambiguity_ratio": float(subset["mean_ambiguity_ratio"].mean()),
            "mean_anchored_alpha_log_std": float(subset["anchored_alpha_log_std"].mean()),
            "mean_anchored_alpha_log_span": float(subset["anchored_alpha_log_span"].mean()),
            "mean_anchored_effective_count": float(subset["anchored_effective_count"].mean()),
            "best_alpha_bank_log_span": float(np.max(best_alpha_logs) - np.min(best_alpha_logs)),
            "anchored_alpha_bank_log_span": float(np.max(anchored_alpha_logs) - np.min(anchored_alpha_logs)),
            "best_alpha_abs_error_mean": float(np.mean(best_alpha_errors)),
            "anchored_alpha_abs_error_mean": float(np.mean(anchored_alpha_errors)),
        }
        row["alpha_abs_error_gain"] = float(row["best_alpha_abs_error_mean"] - row["anchored_alpha_abs_error_mean"])
        row["anchored_beats_best_flag"] = int(row["anchored_alpha_abs_error_mean"] <= row["best_alpha_abs_error_mean"])
        row["alpha_point_recoverable_flag"] = int(
            row["anchored_alpha_bank_log_span"] < ALPHA_STABLE_LOG_SPAN_THRESHOLD
            and row["anchored_alpha_abs_error_mean"] < ALPHA_POINT_ABS_ERROR_THRESHOLD
        )
        row["alpha_point_unrecoverable_flag"] = int(1 - row["alpha_point_recoverable_flag"])
        trial_rows.append(row)

    derived = pd.DataFrame(trial_rows)
    derived["ratio_anchored_span_over_std"] = (
        derived["mean_anchored_alpha_log_span"] / (derived["mean_anchored_alpha_log_std"] + 1.0e-9)
    )
    derived["ratio_candidate_times_anchored_span_over_std"] = (
        derived["mean_candidate_count"]
        * derived["mean_anchored_alpha_log_span"]
        / (derived["mean_anchored_alpha_log_std"] + 1.0e-9)
    )

    legacy = pd.read_csv(LEGACY_TRIALS_PATH)
    legacy = legacy[legacy["condition"] == TARGET_CONDITION].copy()
    legacy.insert(0, "method", LEGACY_METHOD)
    legacy["ratio_anchored_span_over_std"] = (
        legacy["mean_anchored_alpha_log_span"] / (legacy["mean_anchored_alpha_log_std"] + 1.0e-9)
    )
    legacy["ratio_candidate_times_anchored_span_over_std"] = (
        legacy["mean_candidate_count"]
        * legacy["mean_anchored_alpha_log_span"]
        / (legacy["mean_anchored_alpha_log_std"] + 1.0e-9)
    )

    columns = [
        "method",
        "split",
        "observation_seed",
        "condition",
        "geometry_skew_bin",
        "true_alpha",
        "true_t",
        "true_rotation_shift",
        "mean_best_entropy",
        "mean_candidate_count",
        "mean_alpha_log_span_set",
        "mean_geometry_span_norm_set",
        "mean_ambiguity_ratio",
        "mean_anchored_alpha_log_std",
        "mean_anchored_alpha_log_span",
        "mean_anchored_effective_count",
        "best_alpha_bank_log_span",
        "anchored_alpha_bank_log_span",
        "best_alpha_abs_error_mean",
        "anchored_alpha_abs_error_mean",
        "alpha_abs_error_gain",
        "anchored_beats_best_flag",
        "alpha_point_recoverable_flag",
        "alpha_point_unrecoverable_flag",
        "ratio_anchored_span_over_std",
        "ratio_candidate_times_anchored_span_over_std",
    ]
    derived = derived[columns]
    calibration = derive_calibration_rows()[columns]
    legacy = legacy[columns]
    return pd.concat([calibration, derived, legacy], ignore_index=True)


def summarize_by_split(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in (METHOD_RANDOM, METHOD_INFORMED, LEGACY_METHOD):
        for split in ("holdout", "confirmation"):
            subset = df[(df["method"] == method) & (df["split"] == split)]
            if subset.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "count": int(len(subset)),
                    "alpha_point_recoverable_rate": float(subset["alpha_point_recoverable_flag"].mean()),
                    "mean_anchored_alpha_log_std": float(subset["mean_anchored_alpha_log_std"].mean()),
                    "mean_anchored_alpha_bank_log_span": float(subset["anchored_alpha_bank_log_span"].mean()),
                    "mean_best_alpha_bank_log_span": float(subset["best_alpha_bank_log_span"].mean()),
                    "mean_anchored_alpha_abs_error": float(subset["anchored_alpha_abs_error_mean"].mean()),
                    "mean_best_alpha_abs_error": float(subset["best_alpha_abs_error_mean"].mean()),
                    "mean_alpha_abs_error_gain": float(subset["alpha_abs_error_gain"].mean()),
                    "anchored_beats_best_rate": float(subset["anchored_beats_best_flag"].mean()),
                    "anchored_std_vs_unrecoverable_corr": safe_corr(
                        subset["mean_anchored_alpha_log_std"].tolist(),
                        subset["alpha_point_unrecoverable_flag"].tolist(),
                    ),
                    "ambiguity_ratio_vs_unrecoverable_corr": safe_corr(
                        subset["mean_ambiguity_ratio"].tolist(),
                        subset["alpha_point_unrecoverable_flag"].tolist(),
                    ),
                }
            )
    return rows


def summarize_by_cell(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in (METHOD_RANDOM, METHOD_INFORMED, LEGACY_METHOD):
        for split in ("holdout", "confirmation"):
            for skew_bin in sorted(df["geometry_skew_bin"].unique()):
                subset = df[
                    (df["method"] == method)
                    & (df["split"] == split)
                    & (df["geometry_skew_bin"] == skew_bin)
                ]
                if subset.empty:
                    continue
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "condition": TARGET_CONDITION,
                        "geometry_skew_bin": skew_bin,
                        "count": int(len(subset)),
                        "alpha_point_recoverable_rate": float(subset["alpha_point_recoverable_flag"].mean()),
                        "mean_anchored_alpha_log_std": float(subset["mean_anchored_alpha_log_std"].mean()),
                        "mean_anchored_alpha_bank_log_span": float(subset["anchored_alpha_bank_log_span"].mean()),
                        "mean_anchored_alpha_abs_error": float(subset["anchored_alpha_abs_error_mean"].mean()),
                        "anchored_beats_best_rate": float(subset["anchored_beats_best_flag"].mean()),
                    }
                )
    return rows


def split_deltas(split_rows: list[dict[str, object]], lhs: str, rhs: str) -> list[dict[str, object]]:
    index = {(row["method"], row["split"]): row for row in split_rows}
    metrics = [
        "alpha_point_recoverable_rate",
        "mean_anchored_alpha_log_std",
        "mean_anchored_alpha_bank_log_span",
        "mean_anchored_alpha_abs_error",
        "mean_best_alpha_abs_error",
        "mean_alpha_abs_error_gain",
        "anchored_beats_best_rate",
    ]
    rows: list[dict[str, object]] = []
    for split in ("holdout", "confirmation"):
        left = index.get((lhs, split))
        right = index.get((rhs, split))
        if left is None or right is None:
            continue
        row = {"lhs_method": lhs, "rhs_method": rhs, "split": split}
        for metric in metrics:
            row[f"{metric}_delta"] = float(left[metric] - right[metric])
        rows.append(row)
    return rows


def load_legacy_gate_rule() -> dict[str, object]:
    payload = json.loads(LEGACY_SUMMARY_PATH.read_text(encoding="utf-8"))
    return dict(payload["summary"]["threshold_rule"]["best_by_oos_mean"])


def fixed_gate_scores(df: pd.DataFrame, metric: str, threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in (METHOD_RANDOM, METHOD_INFORMED, LEGACY_METHOD):
        for split in ("holdout", "confirmation"):
            subset = df[(df["method"] == method) & (df["split"] == split)]
            if subset.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "metric": metric,
                    "threshold": float(threshold),
                    "direction": "ge",
                    "balanced_accuracy": balanced_accuracy(subset, metric, threshold, direction="ge"),
                }
            )
    return rows


def pooled_threshold_probes(df: pd.DataFrame) -> list[dict[str, object]]:
    metrics = [
        "mean_anchored_alpha_log_std",
        "mean_anchored_alpha_log_span",
        "mean_ambiguity_ratio",
        "mean_alpha_log_span_set",
    ]
    rows: list[dict[str, object]] = []
    for method in (METHOD_RANDOM, METHOD_INFORMED, LEGACY_METHOD):
        subset = df[df["method"] == method]
        for metric in metrics:
            payload = {"method": method}
            payload.update(best_threshold(subset, metric))
            rows.append(payload)
    return rows


def split_threshold_probes(df: pd.DataFrame) -> list[dict[str, object]]:
    metrics = ["mean_anchored_alpha_log_std", "mean_ambiguity_ratio"]
    rows: list[dict[str, object]] = []
    for method in (METHOD_RANDOM, METHOD_INFORMED, LEGACY_METHOD):
        for split in ("holdout", "confirmation"):
            subset = df[(df["method"] == method) & (df["split"] == split)]
            for metric in metrics:
                payload = {"method": method, "split": split}
                payload.update(best_threshold(subset, metric))
                rows.append(payload)
    return rows


def plot_std_scatter(df: pd.DataFrame) -> None:
    plot_df = df[df["method"].isin([METHOD_RANDOM, METHOD_INFORMED])].copy()
    plot_df["unrecoverable"] = plot_df["alpha_point_unrecoverable_flag"].map({0: "recoverable", 1: "unrecoverable"})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=False, sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.08, right=0.98, wspace=0.18)
    palette = {"recoverable": "#2a9d8f", "unrecoverable": "#e76f51"}
    for ax, split in zip(axes, ("holdout", "confirmation")):
        subset = plot_df[plot_df["split"] == split]
        sns.scatterplot(
            data=subset,
            x="mean_anchored_alpha_log_std",
            y="anchored_alpha_bank_log_span",
            hue="unrecoverable",
            style="method",
            s=72,
            palette=palette,
            ax=ax,
        )
        ax.axhline(ALPHA_STABLE_LOG_SPAN_THRESHOLD, color="#444444", lw=1.1, linestyle="--")
        ax.set_title(split)
        ax.set_xlabel("mean anchored alpha log std")
        ax.set_ylabel("anchored alpha bank log span")
        ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.suptitle("Layer 2 signal after informed-bank Layer 1")
    fig.savefig(STD_SCATTER_PNG, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_bars(fixed_scores: list[dict[str, object]]) -> None:
    df = pd.DataFrame(fixed_scores)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.11, right=0.97)
    sns.barplot(
        data=df,
        x="method",
        y="balanced_accuracy",
        hue="split",
        palette={"holdout": "#2a9d8f", "confirmation": "#e76f51"},
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("balanced accuracy")
    ax.set_xlabel("method")
    ax.set_title("Frozen legacy Layer 2 gate on hard-branch holdout/confirmation")
    fig.savefig(THRESHOLD_BAR_PNG, bbox_inches="tight")
    plt.close(fig)


def proposed_informed_gate_rule() -> dict[str, object]:
    if not RATIO_SWEEP_SUMMARY_PATH.exists():
        return {}
    payload = json.loads(RATIO_SWEEP_SUMMARY_PATH.read_text(encoding="utf-8"))
    return dict(payload.get("best_by_method", {}).get(METHOD_INFORMED, {}))


def main() -> None:
    trial_df = derive_trial_rows()
    split_rows = summarize_by_split(trial_df)
    cell_rows = summarize_by_cell(trial_df)

    legacy_gate = load_legacy_gate_rule()
    fixed_scores = fixed_gate_scores(trial_df, str(legacy_gate["metric"]), float(legacy_gate["threshold"]))
    pooled_probes = pooled_threshold_probes(trial_df)
    split_probes = split_threshold_probes(trial_df)
    proposed_rule = proposed_informed_gate_rule()

    write_csv(TRIALS_CSV, trial_df.to_dict(orient="records"))
    write_csv(SPLIT_CSV, split_rows)
    write_csv(CELL_CSV, cell_rows)
    plot_std_scatter(trial_df)
    plot_threshold_bars(fixed_scores)

    summary = {
        "experiment": "backbone-observability-gate-informed-bank",
        "source": "derived from persistent-mode bank candidate atlas plus legacy Layer 2 outputs",
        "target_condition": TARGET_CONDITION,
        "trial_count": int(len(trial_df)),
        "methods": [METHOD_RANDOM, METHOD_INFORMED, LEGACY_METHOD],
        "alpha_stable_log_span_threshold": ALPHA_STABLE_LOG_SPAN_THRESHOLD,
        "alpha_point_abs_error_threshold": ALPHA_POINT_ABS_ERROR_THRESHOLD,
        "legacy_frozen_gate_rule": legacy_gate,
        "fixed_legacy_gate_scores": fixed_scores,
        "pooled_threshold_probes": pooled_probes,
        "split_threshold_probes": split_probes,
        "proposed_informed_gate_rule": proposed_rule,
        "split_summary": split_rows,
        "cell_summary": cell_rows,
        "informed_vs_same_run_random": split_deltas(split_rows, METHOD_INFORMED, METHOD_RANDOM),
        "informed_vs_legacy_random": split_deltas(split_rows, METHOD_INFORMED, LEGACY_METHOD),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
