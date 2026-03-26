"""
Comparative ratio sweep for Layer 2 observability signals.

This experiment compares multiple Layer 2 metric families on the same hard-branch
trial table:

- raw Layer 2 metrics
- spread-normalized ratios
- support-normalized ratios
- simple mixed ratios

For each method, it freezes threshold and direction on calibration, then scores
holdout and confirmation to see which ratio family transfers best.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

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

TRIALS_PATH = (
    BASE_DIR.parent
    / "backbone-observability-gate-informed-bank/outputs/backbone_observability_gate_informed_bank_trials.csv"
)

SUMMARY_JSON = OUTPUT_DIR / "backbone_observability_gate_ratio_sweep_summary.json"
LEADERBOARD_CSV = OUTPUT_DIR / "backbone_observability_gate_ratio_sweep_leaderboard.csv"
TOP_PLOT = FIGURE_DIR / "backbone_observability_gate_ratio_sweep_top_metrics.png"

METHODS = ("legacy_random_5seed", "one_shot_random", "persistent_mode_informed")
SPLITS = ("calibration", "holdout", "confirmation")
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
    unique = sorted({float(item) for item in values})
    candidates = [unique[0] - 1.0e-6]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(unique[:-1], unique[1:]))
    candidates.append(unique[-1] + 1.0e-6)
    best_score = -1.0
    best_threshold = 0.0
    best_direction = "ge"
    for direction in ("ge", "le"):
        for threshold in candidates:
            score = balanced_accuracy(labels, values, threshold, direction)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_direction = direction
    return {
        "threshold": best_threshold,
        "direction": best_direction,
        "calibration_balanced_accuracy": float(best_score),
    }


def build_ratio_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["metric_std"] = out["mean_anchored_alpha_log_std"]
    out["metric_ambiguity"] = out["mean_ambiguity_ratio"]
    out["metric_set_span"] = out["mean_alpha_log_span_set"]
    out["metric_anchored_span"] = out["mean_anchored_alpha_log_span"]
    out["ratio_std_over_set_span"] = out["mean_anchored_alpha_log_std"] / (out["mean_alpha_log_span_set"] + EPS)
    out["ratio_std_over_ambiguity"] = out["mean_anchored_alpha_log_std"] / (out["mean_ambiguity_ratio"] + EPS)
    out["ratio_ambiguity_over_std"] = out["mean_ambiguity_ratio"] / (out["mean_anchored_alpha_log_std"] + EPS)
    out["ratio_std_over_geomspan"] = out["mean_anchored_alpha_log_std"] / (out["mean_geometry_span_norm_set"] + EPS)
    out["ratio_effective_over_count"] = out["mean_anchored_effective_count"] / (out["mean_candidate_count"] + EPS)
    out["ratio_count_over_effective"] = out["mean_candidate_count"] / (out["mean_anchored_effective_count"] + EPS)
    out["ratio_ambiguity_over_effective"] = out["mean_ambiguity_ratio"] / (out["mean_anchored_effective_count"] + EPS)
    out["ratio_effective_over_ambiguity"] = out["mean_anchored_effective_count"] / (out["mean_ambiguity_ratio"] + EPS)
    out["ratio_std_times_effective_over_count"] = (
        out["mean_anchored_alpha_log_std"] * out["mean_anchored_effective_count"] / (out["mean_candidate_count"] + EPS)
    )
    out["ratio_anchored_span_over_std"] = out["mean_anchored_alpha_log_span"] / (
        out["mean_anchored_alpha_log_std"] + EPS
    )
    out["ratio_candidate_times_anchored_span_over_std"] = (
        out["mean_candidate_count"] * out["mean_anchored_alpha_log_span"] / (out["mean_anchored_alpha_log_std"] + EPS)
    )
    out["ratio_effective_times_anchored_span_over_std"] = (
        out["mean_anchored_effective_count"] * out["mean_anchored_alpha_log_span"]
        / (out["mean_anchored_alpha_log_std"] + EPS)
    )
    out["ratio_geometry_over_candidate_effective"] = out["mean_geometry_span_norm_set"] / (
        (out["mean_candidate_count"] * out["mean_anchored_effective_count"]) + EPS
    )
    out["ratio_candidate_times_geometry"] = out["mean_candidate_count"] * out["mean_geometry_span_norm_set"]
    return out


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("metric_") or column.startswith("ratio_")]


def evaluate_method(df: pd.DataFrame, method: str) -> list[dict[str, object]]:
    subset = df[df["method"] == method].copy()
    calibration = subset[subset["split"] == "calibration"]
    holdout = subset[subset["split"] == "holdout"]
    confirmation = subset[subset["split"] == "confirmation"]
    rows: list[dict[str, object]] = []
    for metric in metric_columns(subset):
        cal_values = calibration[metric].to_numpy(dtype=float)
        cal_labels = calibration["alpha_point_unrecoverable_flag"].to_numpy(dtype=int)
        threshold_payload = choose_threshold(cal_values, cal_labels)
        threshold = float(threshold_payload["threshold"])
        direction = str(threshold_payload["direction"])
        holdout_score = balanced_accuracy(
            holdout["alpha_point_unrecoverable_flag"].to_numpy(dtype=int),
            holdout[metric].to_numpy(dtype=float),
            threshold,
            direction,
        )
        confirmation_score = balanced_accuracy(
            confirmation["alpha_point_unrecoverable_flag"].to_numpy(dtype=int),
            confirmation[metric].to_numpy(dtype=float),
            threshold,
            direction,
        )
        rows.append(
            {
                "method": method,
                "metric": metric,
                "threshold": threshold,
                "direction": direction,
                "calibration_balanced_accuracy": float(threshold_payload["calibration_balanced_accuracy"]),
                "holdout_balanced_accuracy": float(holdout_score),
                "confirmation_balanced_accuracy": float(confirmation_score),
                "oos_mean_balanced_accuracy": float(0.5 * (holdout_score + confirmation_score)),
            }
        )
    rows.sort(
        key=lambda item: (
            round(item["oos_mean_balanced_accuracy"], 6),
            item["confirmation_balanced_accuracy"],
            item["calibration_balanced_accuracy"],
        ),
        reverse=True,
    )
    return rows


def plot_top_metrics(df: pd.DataFrame) -> None:
    top = (
        df.groupby("method", sort=False)
        .head(5)[
            [
                "method",
                "metric",
                "holdout_balanced_accuracy",
                "confirmation_balanced_accuracy",
                "oos_mean_balanced_accuracy",
            ]
        ]
        .melt(id_vars=["method", "metric"], var_name="split_metric", value_name="balanced_accuracy")
    )
    rename = {
        "holdout_balanced_accuracy": "holdout",
        "confirmation_balanced_accuracy": "confirmation",
        "oos_mean_balanced_accuracy": "oos_mean",
    }
    top["split_metric"] = top["split_metric"].map(rename)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.28, left=0.05, right=0.99, wspace=0.16)
    palette = {"holdout": "#2a9d8f", "confirmation": "#e76f51", "oos_mean": "#264653"}
    for ax, method in zip(axes, METHODS):
        subset = top[top["method"] == method]
        sns.barplot(
            data=subset,
            x="metric",
            y="balanced_accuracy",
            hue="split_metric",
            palette=palette,
            ax=ax,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(method)
        ax.set_xlabel("")
        ax.set_ylabel("balanced accuracy")
        ax.tick_params(axis="x", rotation=72)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.suptitle("Layer 2 ratio sweep: top calibration-frozen metrics by method")
    fig.savefig(TOP_PLOT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    trials = pd.read_csv(TRIALS_PATH)
    trials = trials[trials["condition"] == "sparse_partial_high_noise"].copy()
    ratio_df = build_ratio_frame(trials)

    leaderboard_rows: list[dict[str, object]] = []
    best_by_method: dict[str, dict[str, object]] = {}
    for method in METHODS:
        rows = evaluate_method(ratio_df, method)
        leaderboard_rows.extend(rows)
        best_by_method[method] = rows[0]

    leaderboard = pd.DataFrame(leaderboard_rows)
    write_csv(LEADERBOARD_CSV, leaderboard_rows)
    plot_top_metrics(leaderboard)

    summary = {
        "experiment": "backbone-observability-gate-ratio-sweep",
        "source_trials": str(TRIALS_PATH.relative_to(BASE_DIR.parents[2])),
        "method_counts": {
            method: int(len(ratio_df[ratio_df["method"] == method])) for method in METHODS
        },
        "metric_count": int(len(metric_columns(ratio_df))),
        "best_by_method": best_by_method,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
