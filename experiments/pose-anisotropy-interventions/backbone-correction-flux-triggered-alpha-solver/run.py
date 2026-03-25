"""
Correction-flux-triggered Layer 3 for the backbone-first pose-anisotropy solver.

This experiment reuses the existing backbone-conditional Layer 3 output bundle
as the fixed candidate generator, then changes only the final decision rule:

- keep the validated Layer 2 gate unchanged
- keep the same anchored and always-refine Layer 3 candidates
- compute post-anchor correction flux from bank-wise anchored/refined outputs
- refine only when that correction flux exceeds a calibration-frozen threshold
- otherwise keep the anchored answer
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

BLOCK_SPECS, GEOMETRY_SKEW_BIN_LABELS, FOCUS_ALPHA_BIN, FOCUS_CONDITIONS = load_symbols(
    "run_backbone_conditional_alpha_solver_experiment_constants",
    ROOT / "experiments/pose-anisotropy-interventions/backbone-conditional-alpha-solver/run.py",
    "BLOCK_SPECS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "FOCUS_ALPHA_BIN",
    "FOCUS_CONDITIONS",
)

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

SOURCE_DIR = ROOT / "experiments/pose-anisotropy-interventions/backbone-conditional-alpha-solver/outputs"
SOURCE_TRIALS = SOURCE_DIR / "backbone_conditional_alpha_solver_trials.csv"
SOURCE_BANK_ROWS = SOURCE_DIR / "backbone_conditional_alpha_solver_bank_rows.csv"
SOURCE_SUMMARY = SOURCE_DIR / "backbone_conditional_alpha_solver_summary.json"

TRIGGER_METRIC_NAME = "correction_flux"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(value: object) -> float:
    return float(value)


def i(value: object) -> int:
    return int(float(value))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def rate_or_nan(flags: list[int]) -> float:
    if not flags:
        return float("nan")
    return float(np.mean(flags))


def trial_key(split: object, observation_seed: object, condition: object, geometry_skew_bin: object) -> tuple[str, int, str, str]:
    return str(split), i(observation_seed), str(condition), str(geometry_skew_bin)


def base_summary_payload() -> dict[str, object]:
    with SOURCE_SUMMARY.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def enrich_bank_rows(bank_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[tuple[str, int, str, str], dict[str, object]]]:
    enriched_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int, str, str], list[dict[str, object]]] = defaultdict(list)

    for row in bank_rows:
        row_dict: dict[str, object] = dict(row)
        gate_open_flag = i(row["gate_open_flag"])
        if gate_open_flag == 1:
            anchored_alpha = f(row["anchored_alpha"])
            refined_alpha = f(row["refined_alpha"])
            correction_log = float(math.log(refined_alpha) - math.log(anchored_alpha))
            correction_abs_log = float(abs(correction_log))
            correction_sign = int(np.sign(correction_log))
        else:
            correction_log = float("nan")
            correction_abs_log = float("nan")
            correction_sign = 0

        row_dict["correction_log"] = correction_log
        row_dict["correction_abs_log"] = correction_abs_log
        row_dict["correction_sign"] = correction_sign
        enriched_rows.append(row_dict)
        grouped[trial_key(row["split"], row["observation_seed"], row["condition"], row["geometry_skew_bin"])].append(row_dict)

    trial_features: dict[tuple[str, int, str, str], dict[str, object]] = {}
    for key, rows in grouped.items():
        if i(rows[0]["gate_open_flag"]) != 1:
            trial_features[key] = {
                "correction_flux": float("nan"),
                "correction_sign_majority": float("nan"),
                "correction_signed_mean": float("nan"),
                "correction_bank_count": 0,
            }
            continue

        logs = np.array([f(row["correction_log"]) for row in rows], dtype=float)
        pos = int(np.sum(logs > 0.0))
        neg = int(np.sum(logs < 0.0))
        trial_features[key] = {
            "correction_flux": float(np.mean(np.abs(logs))),
            "correction_sign_majority": float(max(pos, neg) / len(logs)),
            "correction_signed_mean": float(np.mean(logs)),
            "correction_bank_count": int(len(logs)),
        }
    return enriched_rows, trial_features


def triggered_error(row: dict[str, object], flux_threshold: float) -> float:
    if i(row["gate_open_flag"]) == 1 and f(row["correction_flux"]) >= flux_threshold:
        return f(row["refined_alpha_output_abs_error"])
    return f(row["anchored_alpha_output_abs_error"])


def choose_flux_threshold(calibration_rows: list[dict[str, object]]) -> tuple[float, float, float]:
    values = sorted({f(row["correction_flux"]) for row in calibration_rows})
    candidates = [values[0] - 1.0e-6]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(values[:-1], values[1:]))
    candidates.append(values[-1] + 1.0e-6)

    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        errors = [triggered_error(row, threshold) for row in calibration_rows]
        switch_rate = float(np.mean([int(f(row["correction_flux"]) >= threshold) for row in calibration_rows]))
        scored.append((float(np.mean(errors)), switch_rate, float(threshold)))

    scored.sort(key=lambda item: (item[0], item[1], -item[2]))
    best_error, best_switch_rate, best_threshold = scored[0]
    return float(best_threshold), float(best_error), float(best_switch_rate)


def build_triggered_trial_rows(
    trial_rows: list[dict[str, str]],
    trial_features: dict[tuple[str, int, str, str], dict[str, object]],
    flux_threshold: float,
) -> list[dict[str, object]]:
    triggered_rows: list[dict[str, object]] = []

    for row in trial_rows:
        row_dict: dict[str, object] = dict(row)
        key = trial_key(row["split"], row["observation_seed"], row["condition"], row["geometry_skew_bin"])
        row_dict.update(
            trial_features.get(
                key,
                {
                    "correction_flux": float("nan"),
                    "correction_sign_majority": float("nan"),
                    "correction_signed_mean": float("nan"),
                    "correction_bank_count": 0,
                },
            )
        )

        gate_open_flag = i(row["gate_open_flag"])
        trigger_fire_flag = int(gate_open_flag == 1 and f(row_dict["correction_flux"]) >= flux_threshold)

        if gate_open_flag == 1:
            if trigger_fire_flag == 1:
                trigger_source = "refined"
                triggered_alpha_output = f(row["refined_alpha_output"])
                triggered_alpha_output_abs_error = f(row["refined_alpha_output_abs_error"])
                triggered_alpha_bank_log_span = f(row["refined_alpha_bank_log_span"])
                triggered_alpha_abs_error_mean = f(row["refined_alpha_abs_error_mean"])
            else:
                trigger_source = "anchored"
                triggered_alpha_output = f(row["anchored_alpha_output"])
                triggered_alpha_output_abs_error = f(row["anchored_alpha_output_abs_error"])
                triggered_alpha_bank_log_span = f(row["anchored_alpha_bank_log_span"])
                triggered_alpha_abs_error_mean = f(row["anchored_alpha_abs_error_mean"])
        else:
            trigger_source = "abstain"
            triggered_alpha_output = float("nan")
            triggered_alpha_output_abs_error = float("nan")
            triggered_alpha_bank_log_span = float("nan")
            triggered_alpha_abs_error_mean = float("nan")

        row_dict["flux_threshold"] = float(flux_threshold)
        row_dict["trigger_fire_flag"] = int(trigger_fire_flag)
        row_dict["trigger_keep_anchored_flag"] = int(gate_open_flag == 1 and trigger_fire_flag == 0)
        row_dict["trigger_source"] = trigger_source
        row_dict["triggered_alpha_output"] = float(triggered_alpha_output)
        row_dict["triggered_alpha_output_abs_error"] = float(triggered_alpha_output_abs_error)
        row_dict["triggered_alpha_bank_log_span"] = float(triggered_alpha_bank_log_span)
        row_dict["triggered_alpha_abs_error_mean"] = float(triggered_alpha_abs_error_mean)
        row_dict["triggered_beats_anchored_flag"] = int(
            gate_open_flag == 1 and triggered_alpha_output_abs_error <= f(row["anchored_alpha_output_abs_error"])
        )
        row_dict["triggered_beats_best_flag"] = int(
            gate_open_flag == 1 and triggered_alpha_output_abs_error <= f(row["best_alpha_output_abs_error"])
        )
        row_dict["triggered_beats_refined_flag"] = int(
            gate_open_flag == 1 and triggered_alpha_output_abs_error <= f(row["refined_alpha_output_abs_error"])
        )
        triggered_rows.append(row_dict)
    return triggered_rows


def gate_balanced_accuracy(subset: list[dict[str, object]]) -> float:
    labels = np.array([i(row["alpha_point_unrecoverable_flag"]) for row in subset], dtype=int)
    preds = np.array([1 - i(row["gate_open_flag"]) for row in subset], dtype=int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = float(np.sum((preds == 1) & (labels == 1)) / positives)
    tnr = float(np.sum((preds == 0) & (labels == 0)) / negatives)
    return 0.5 * (tpr + tnr)


def summarize_by_split(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for split in BLOCK_SPECS:
        subset = [row for row in rows if str(row["split"]) == split]
        if not subset:
            continue
        open_subset = [row for row in subset if i(row["gate_open_flag"]) == 1]
        unrecoverable_subset = [row for row in subset if i(row["alpha_point_unrecoverable_flag"]) == 1]
        summary.append(
            {
                "split": split,
                "count": len(subset),
                "point_output_count": int(sum(i(row["point_output_flag"]) for row in subset)),
                "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in subset])),
                "alpha_point_recoverable_rate": float(np.mean([i(row["alpha_point_recoverable_flag"]) for row in subset])),
                "gate_balanced_accuracy": gate_balanced_accuracy(subset),
                "gate_precision": rate_or_nan([i(row["alpha_point_recoverable_flag"]) for row in open_subset]),
                "gate_reject_unrecoverable_rate": rate_or_nan([i(row["gate_closed_and_unrecoverable_flag"]) for row in unrecoverable_subset]),
                "trigger_fire_rate_open": rate_or_nan([i(row["trigger_fire_flag"]) for row in open_subset]),
                "trigger_keep_anchored_rate_open": rate_or_nan([i(row["trigger_keep_anchored_flag"]) for row in open_subset]),
                "mean_correction_flux_open": mean_or_nan([f(row["correction_flux"]) for row in open_subset]),
                "mean_correction_sign_majority_open": mean_or_nan([f(row["correction_sign_majority"]) for row in open_subset]),
                "mean_best_alpha_output_abs_error_open": mean_or_nan([f(row["best_alpha_output_abs_error"]) for row in open_subset]),
                "mean_anchored_alpha_output_abs_error_open": mean_or_nan([f(row["anchored_alpha_output_abs_error"]) for row in open_subset]),
                "mean_refined_alpha_output_abs_error_open": mean_or_nan([f(row["refined_alpha_output_abs_error"]) for row in open_subset]),
                "mean_triggered_alpha_output_abs_error_open": mean_or_nan([f(row["triggered_alpha_output_abs_error"]) for row in open_subset]),
                "mean_best_alpha_bank_log_span_open": mean_or_nan([f(row["best_alpha_bank_log_span"]) for row in open_subset]),
                "mean_anchored_alpha_bank_log_span_open": mean_or_nan([f(row["anchored_alpha_bank_log_span"]) for row in open_subset]),
                "mean_refined_alpha_bank_log_span_open": mean_or_nan([f(row["refined_alpha_bank_log_span"]) for row in open_subset]),
                "mean_triggered_alpha_bank_log_span_open": mean_or_nan([f(row["triggered_alpha_bank_log_span"]) for row in open_subset]),
                "triggered_beats_anchored_rate_open": rate_or_nan([i(row["triggered_beats_anchored_flag"]) for row in open_subset]),
                "triggered_beats_best_rate_open": rate_or_nan([i(row["triggered_beats_best_flag"]) for row in open_subset]),
                "triggered_beats_refined_rate_open": rate_or_nan([i(row["triggered_beats_refined_flag"]) for row in open_subset]),
            }
        )
    return summary


def summarize_by_condition(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            subset = [row for row in rows if str(row["split"]) == split and str(row["condition"]) == condition]
            if not subset:
                continue
            open_subset = [row for row in subset if i(row["gate_open_flag"]) == 1]
            summary.append(
                {
                    "split": split,
                    "condition": condition,
                    "count": len(subset),
                    "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in subset])),
                    "alpha_point_recoverable_rate": float(np.mean([i(row["alpha_point_recoverable_flag"]) for row in subset])),
                    "gate_precision": rate_or_nan([i(row["alpha_point_recoverable_flag"]) for row in open_subset]),
                    "trigger_fire_rate_open": rate_or_nan([i(row["trigger_fire_flag"]) for row in open_subset]),
                    "mean_correction_flux_open": mean_or_nan([f(row["correction_flux"]) for row in open_subset]),
                    "mean_triggered_alpha_output_abs_error_open": mean_or_nan([f(row["triggered_alpha_output_abs_error"]) for row in open_subset]),
                    "mean_triggered_alpha_bank_log_span_open": mean_or_nan([f(row["triggered_alpha_bank_log_span"]) for row in open_subset]),
                    "triggered_beats_anchored_rate_open": rate_or_nan([i(row["triggered_beats_anchored_flag"]) for row in open_subset]),
                    "triggered_beats_best_rate_open": rate_or_nan([i(row["triggered_beats_best_flag"]) for row in open_subset]),
                }
            )
    return summary


def summarize_by_cell(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                subset = [
                    row
                    for row in rows
                    if str(row["split"]) == split
                    and str(row["condition"]) == condition
                    and str(row["geometry_skew_bin"]) == skew_bin
                ]
                if not subset:
                    continue
                open_subset = [row for row in subset if i(row["gate_open_flag"]) == 1]
                summary.append(
                    {
                        "split": split,
                        "condition": condition,
                        "alpha_strength_bin": FOCUS_ALPHA_BIN,
                        "geometry_skew_bin": skew_bin,
                        "count": len(subset),
                        "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in subset])),
                        "alpha_point_recoverable_rate": float(np.mean([i(row["alpha_point_recoverable_flag"]) for row in subset])),
                        "gate_precision": rate_or_nan([i(row["alpha_point_recoverable_flag"]) for row in open_subset]),
                        "trigger_fire_rate_open": rate_or_nan([i(row["trigger_fire_flag"]) for row in open_subset]),
                        "mean_correction_flux_open": mean_or_nan([f(row["correction_flux"]) for row in open_subset]),
                        "mean_best_alpha_output_abs_error_open": mean_or_nan([f(row["best_alpha_output_abs_error"]) for row in open_subset]),
                        "mean_anchored_alpha_output_abs_error_open": mean_or_nan([f(row["anchored_alpha_output_abs_error"]) for row in open_subset]),
                        "mean_refined_alpha_output_abs_error_open": mean_or_nan([f(row["refined_alpha_output_abs_error"]) for row in open_subset]),
                        "mean_triggered_alpha_output_abs_error_open": mean_or_nan([f(row["triggered_alpha_output_abs_error"]) for row in open_subset]),
                        "mean_triggered_alpha_bank_log_span_open": mean_or_nan([f(row["triggered_alpha_bank_log_span"]) for row in open_subset]),
                        "triggered_beats_anchored_rate_open": rate_or_nan([i(row["triggered_beats_anchored_flag"]) for row in open_subset]),
                        "triggered_beats_best_rate_open": rate_or_nan([i(row["triggered_beats_best_flag"]) for row in open_subset]),
                    }
                )
    return summary


def plot_open_alpha_errors(path: str, split_summary: list[dict[str, object]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([f(item["mean_best_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    anchored = np.array([f(item["mean_anchored_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    refined = np.array([f(item["mean_refined_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    triggered = np.array([f(item["mean_triggered_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.19

    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - 1.5 * width, best, width=width, color="#9c6644", label="best-bank ensemble")
    ax.bar(x - 0.5 * width, anchored, width=width, color="#577590", label="anchored ensemble")
    ax.bar(x + 0.5 * width, refined, width=width, color="#e76f51", label="always-refine ensemble")
    ax.bar(x + 1.5 * width, triggered, width=width, color="#2a9d8f", label="flux-triggered ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean open-trial alpha abs error")
    ax.set_title("Flux-triggered Layer 3 alpha output error on gate-open trials")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_open_alpha_spans(path: str, split_summary: list[dict[str, object]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([f(item["mean_best_alpha_bank_log_span_open"]) for item in split_summary], dtype=float)
    anchored = np.array([f(item["mean_anchored_alpha_bank_log_span_open"]) for item in split_summary], dtype=float)
    refined = np.array([f(item["mean_refined_alpha_bank_log_span_open"]) for item in split_summary], dtype=float)
    triggered = np.array([f(item["mean_triggered_alpha_bank_log_span_open"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.19

    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - 1.5 * width, best, width=width, color="#8d99ae", label="best-bank span")
    ax.bar(x - 0.5 * width, anchored, width=width, color="#577590", label="anchored span")
    ax.bar(x + 0.5 * width, refined, width=width, color="#e76f51", label="always-refine span")
    ax.bar(x + 1.5 * width, triggered, width=width, color="#2a9d8f", label="flux-triggered span")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean open-trial alpha bank log-span")
    ax.set_title("Flux-triggered Layer 3 keeps most anchor stability")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_payload = base_summary_payload()
    base_trials = read_csv(SOURCE_TRIALS)
    base_bank_rows = read_csv(SOURCE_BANK_ROWS)

    bank_rows, trial_features = enrich_bank_rows(base_bank_rows)
    base_trial_rows: list[dict[str, object]] = []
    for row in base_trials:
        row_dict: dict[str, object] = dict(row)
        row_dict.update(
            trial_features.get(
                trial_key(row["split"], row["observation_seed"], row["condition"], row["geometry_skew_bin"]),
                {
                    "correction_flux": float("nan"),
                    "correction_sign_majority": float("nan"),
                    "correction_signed_mean": float("nan"),
                    "correction_bank_count": 0,
                },
            )
        )
        base_trial_rows.append(row_dict)

    calibration_open_rows = [
        row for row in base_trial_rows if str(row["split"]) == "calibration" and i(row["gate_open_flag"]) == 1
    ]
    flux_threshold, calibration_trigger_error, calibration_trigger_rate = choose_flux_threshold(calibration_open_rows)
    trial_rows = build_triggered_trial_rows(base_trials, trial_features, flux_threshold)

    split_summary = summarize_by_split(trial_rows)
    condition_summary = summarize_by_condition(trial_rows)
    cell_summary = summarize_by_cell(trial_rows)

    open_rows = [row for row in trial_rows if i(row["gate_open_flag"]) == 1]
    unrecoverable_rows = [row for row in trial_rows if i(row["alpha_point_unrecoverable_flag"]) == 1]

    flux_summary = {
        "metric": TRIGGER_METRIC_NAME,
        "threshold": float(flux_threshold),
        "calibration_mean_triggered_alpha_output_abs_error_open": float(calibration_trigger_error),
        "calibration_trigger_fire_rate_open": float(calibration_trigger_rate),
        "holdout_mean_triggered_alpha_output_abs_error_open": mean_or_nan(
            [f(row["triggered_alpha_output_abs_error"]) for row in open_rows if str(row["split"]) == "holdout"]
        ),
        "holdout_trigger_fire_rate_open": rate_or_nan(
            [i(row["trigger_fire_flag"]) for row in open_rows if str(row["split"]) == "holdout"]
        ),
        "confirmation_mean_triggered_alpha_output_abs_error_open": mean_or_nan(
            [f(row["triggered_alpha_output_abs_error"]) for row in open_rows if str(row["split"]) == "confirmation"]
        ),
        "confirmation_trigger_fire_rate_open": rate_or_nan(
            [i(row["trigger_fire_flag"]) for row in open_rows if str(row["split"]) == "confirmation"]
        ),
        "overall_mean_triggered_alpha_output_abs_error_open": mean_or_nan(
            [f(row["triggered_alpha_output_abs_error"]) for row in open_rows]
        ),
        "overall_trigger_fire_rate_open": rate_or_nan([i(row["trigger_fire_flag"]) for row in open_rows]),
    }

    global_summary = {
        "reference_bank_size": int(base_payload["summary"]["reference_bank_size"]),
        "bank_seeds": list(base_payload["summary"]["bank_seeds"]),
        "score_band_rule": base_payload["summary"]["score_band_rule"],
        "top_k_refinement_seeds": int(base_payload["summary"]["top_k_refinement_seeds"]),
        "gate_metric": base_payload["summary"]["gate_metric"],
        "trigger_metric": TRIGGER_METRIC_NAME,
        "trial_count": len(trial_rows),
        "point_output_count": int(sum(i(row["point_output_flag"]) for row in trial_rows)),
        "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in trial_rows])),
        "alpha_point_recoverable_rate": float(np.mean([i(row["alpha_point_recoverable_flag"]) for row in trial_rows])),
        "gate_precision": rate_or_nan([i(row["alpha_point_recoverable_flag"]) for row in open_rows]),
        "gate_reject_unrecoverable_rate": rate_or_nan([i(row["gate_closed_and_unrecoverable_flag"]) for row in unrecoverable_rows]),
        "mean_correction_flux_open": mean_or_nan([f(row["correction_flux"]) for row in open_rows]),
        "mean_correction_sign_majority_open": mean_or_nan([f(row["correction_sign_majority"]) for row in open_rows]),
        "trigger_fire_rate_open": rate_or_nan([i(row["trigger_fire_flag"]) for row in open_rows]),
        "trigger_keep_anchored_rate_open": rate_or_nan([i(row["trigger_keep_anchored_flag"]) for row in open_rows]),
        "mean_best_alpha_output_abs_error_open": mean_or_nan([f(row["best_alpha_output_abs_error"]) for row in open_rows]),
        "mean_anchored_alpha_output_abs_error_open": mean_or_nan([f(row["anchored_alpha_output_abs_error"]) for row in open_rows]),
        "mean_refined_alpha_output_abs_error_open": mean_or_nan([f(row["refined_alpha_output_abs_error"]) for row in open_rows]),
        "mean_triggered_alpha_output_abs_error_open": mean_or_nan([f(row["triggered_alpha_output_abs_error"]) for row in open_rows]),
        "mean_best_alpha_bank_log_span_open": mean_or_nan([f(row["best_alpha_bank_log_span"]) for row in open_rows]),
        "mean_anchored_alpha_bank_log_span_open": mean_or_nan([f(row["anchored_alpha_bank_log_span"]) for row in open_rows]),
        "mean_refined_alpha_bank_log_span_open": mean_or_nan([f(row["refined_alpha_bank_log_span"]) for row in open_rows]),
        "mean_triggered_alpha_bank_log_span_open": mean_or_nan([f(row["triggered_alpha_bank_log_span"]) for row in open_rows]),
        "triggered_beats_anchored_rate_open": rate_or_nan([i(row["triggered_beats_anchored_flag"]) for row in open_rows]),
        "triggered_beats_best_rate_open": rate_or_nan([i(row["triggered_beats_best_flag"]) for row in open_rows]),
        "triggered_beats_refined_rate_open": rate_or_nan([i(row["triggered_beats_refined_flag"]) for row in open_rows]),
    }

    output_payload = {
        "summary": global_summary,
        "gate_threshold": base_payload["gate_threshold"],
        "flux_threshold": flux_summary,
        "by_split": split_summary,
        "by_condition": condition_summary,
        "by_cell": cell_summary,
    }

    prefix = "backbone_correction_flux_triggered_alpha_solver"
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_bank_rows.csv"), bank_rows)
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_trials.csv"), trial_rows)
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_condition_summary.csv"), condition_summary)
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_cell_summary.csv"), cell_summary)
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    plot_open_alpha_errors(
        os.path.join(FIGURE_DIR, f"{prefix}_alpha_error.png"),
        split_summary,
    )
    plot_open_alpha_spans(
        os.path.join(FIGURE_DIR, f"{prefix}_alpha_span.png"),
        split_summary,
    )

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
