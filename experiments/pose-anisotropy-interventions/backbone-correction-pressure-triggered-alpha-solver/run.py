"""
Correction-pressure-triggered Layer 3 for the backbone-first pose-anisotropy solver.

This cached-output experiment promotes a simple ratio-controlled trigger:

- keep the validated Layer 2 gate unchanged
- keep the same anchored and always-refine Layer 3 candidates
- compute coherent correction pressure as:
  (correction_flux * correction_sign_majority) / anchored_gate_std
- refine only when that pressure exceeds a calibration-frozen threshold
- otherwise keep the anchored answer
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

BLOCK_SPECS, GEOMETRY_SKEW_BIN_LABELS, FOCUS_ALPHA_BIN, FOCUS_CONDITIONS = load_symbols(
    "run_backbone_conditional_alpha_solver_experiment_constants_for_pressure",
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

SOURCE_DIR = ROOT / "experiments/pose-anisotropy-interventions/backbone-correction-flux-triggered-alpha-solver/outputs"
SOURCE_TRIALS = SOURCE_DIR / "backbone_correction_flux_triggered_alpha_solver_trials.csv"
SOURCE_SUMMARY = SOURCE_DIR / "backbone_correction_flux_triggered_alpha_solver_summary.json"

TRIGGER_METRIC_NAME = "correction_pressure"
NUMERIC_EPS = 1.0e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
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


def base_summary_payload() -> dict[str, object]:
    with SOURCE_SUMMARY.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def correction_pressure(row: dict[str, object]) -> float:
    coherent_flux = f(row["correction_flux"]) * f(row["correction_sign_majority"])
    return coherent_flux / max(f(row["mean_anchored_alpha_log_std"]), NUMERIC_EPS)


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


def choose_threshold(calibration_rows: list[dict[str, object]]) -> tuple[float, float, float]:
    values = sorted({correction_pressure(row) for row in calibration_rows})
    candidates = [values[0] - 1.0e-6]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(values[:-1], values[1:]))
    candidates.append(values[-1] + 1.0e-6)

    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        errors = [
            f(row["refined_alpha_output_abs_error"]) if correction_pressure(row) >= threshold else f(row["anchored_alpha_output_abs_error"])
            for row in calibration_rows
        ]
        switch_rate = float(np.mean([int(correction_pressure(row) >= threshold) for row in calibration_rows]))
        scored.append((float(np.mean(errors)), switch_rate, float(threshold)))

    scored.sort(key=lambda item: (item[0], item[1], -item[2]))
    best_error, best_switch_rate, best_threshold = scored[0]
    return float(best_threshold), float(best_error), float(best_switch_rate)


def build_triggered_trial_rows(trial_rows: list[dict[str, str]], pressure_threshold: float) -> list[dict[str, object]]:
    triggered_rows: list[dict[str, object]] = []
    for row in trial_rows:
        row_dict: dict[str, object] = dict(row)
        gate_open_flag = i(row["gate_open_flag"])
        pressure = correction_pressure(row_dict)
        coherent_flux = f(row["correction_flux"]) * f(row["correction_sign_majority"])
        trigger_fire_flag = int(gate_open_flag == 1 and pressure >= pressure_threshold)

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

        row_dict["coherent_correction_flux"] = float(coherent_flux)
        row_dict["correction_pressure"] = float(pressure)
        row_dict["pressure_threshold"] = float(pressure_threshold)
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
                "mean_coherent_correction_flux_open": mean_or_nan([f(row["coherent_correction_flux"]) for row in open_subset]),
                "mean_correction_pressure_open": mean_or_nan([f(row["correction_pressure"]) for row in open_subset]),
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
                    "mean_correction_pressure_open": mean_or_nan([f(row["correction_pressure"]) for row in open_subset]),
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
                        "mean_correction_pressure_open": mean_or_nan([f(row["correction_pressure"]) for row in open_subset]),
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
    ax.bar(x + 1.5 * width, triggered, width=width, color="#2a9d8f", label="pressure-triggered ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean open-trial alpha abs error")
    ax.set_title("Pressure-triggered Layer 3 alpha output error on gate-open trials")
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
    ax.bar(x + 1.5 * width, triggered, width=width, color="#2a9d8f", label="pressure-triggered span")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean open-trial alpha bank log-span")
    ax.set_title("Pressure-triggered Layer 3 improves accuracy without reopening full spread")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_payload = base_summary_payload()
    base_trials = read_csv(SOURCE_TRIALS)
    trial_rows_open = [row for row in base_trials if i(row["gate_open_flag"]) == 1]
    calibration_open_rows = [row for row in trial_rows_open if str(row["split"]) == "calibration"]
    pressure_threshold, calibration_error, calibration_switch_rate = choose_threshold(calibration_open_rows)

    trial_rows = build_triggered_trial_rows(base_trials, pressure_threshold)
    split_summary = summarize_by_split(trial_rows)
    condition_summary = summarize_by_condition(trial_rows)
    cell_summary = summarize_by_cell(trial_rows)

    open_rows = [row for row in trial_rows if i(row["gate_open_flag"]) == 1]
    unrecoverable_rows = [row for row in trial_rows if i(row["alpha_point_unrecoverable_flag"]) == 1]
    pressure_summary = {
        "metric": TRIGGER_METRIC_NAME,
        "definition": "(correction_flux * correction_sign_majority) / mean_anchored_alpha_log_std",
        "threshold": float(pressure_threshold),
        "calibration_mean_triggered_alpha_output_abs_error_open": float(calibration_error),
        "calibration_trigger_fire_rate_open": float(calibration_switch_rate),
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
        "mean_coherent_correction_flux_open": mean_or_nan([f(row["coherent_correction_flux"]) for row in open_rows]),
        "mean_correction_pressure_open": mean_or_nan([f(row["correction_pressure"]) for row in open_rows]),
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
        "pressure_threshold": pressure_summary,
        "flux_baseline": base_payload["flux_threshold"],
        "by_split": split_summary,
        "by_condition": condition_summary,
        "by_cell": cell_summary,
    }

    prefix = "backbone_correction_pressure_triggered_alpha_solver"
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
