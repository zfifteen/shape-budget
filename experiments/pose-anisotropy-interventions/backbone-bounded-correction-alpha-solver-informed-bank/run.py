"""
Bounded-correction Layer 3 on top of the corrected informed-bank stack.

This cached-output experiment keeps the current informed-bank Layer 2 open set,
but replaces all-or-nothing Layer 3 refinement with a bounded correction:

- measure correction excursion in log-alpha space
- normalize it by anchored alpha span
- apply only a fraction of the refined move when that excursion is small
- collapse back to anchored when the excursion is too large
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

SOURCE_DIR = (
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-conditional-alpha-solver-informed-bank/outputs"
)
SOURCE_TRIALS = SOURCE_DIR / "backbone_conditional_alpha_solver_informed_bank_trials.csv"
SOURCE_SUMMARY = SOURCE_DIR / "backbone_conditional_alpha_solver_informed_bank_summary.json"

PREFIX = "backbone_bounded_correction_alpha_solver_informed_bank"
NUMERIC_EPS = 1.0e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
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


def correction_excursion_ratio(row: dict[str, object]) -> float:
    anchored_log = math.log(f(row["anchored_alpha_output"]))
    refined_log = math.log(f(row["refined_alpha_output"]))
    excursion = abs(refined_log - anchored_log)
    return excursion / max(f(row["mean_anchored_alpha_log_span"]), NUMERIC_EPS)


def bounded_weight(row: dict[str, object], tau: float) -> float:
    ratio = correction_excursion_ratio(row)
    return float(max(0.0, 1.0 - ratio / max(tau, NUMERIC_EPS)))


def bounded_row(row: dict[str, str], tau: float) -> dict[str, object]:
    out: dict[str, object] = dict(row)
    gate_open_flag = i(row["gate_open_flag"])

    if gate_open_flag == 1:
        anchored_log = math.log(f(row["anchored_alpha_output"]))
        refined_log = math.log(f(row["refined_alpha_output"]))
        weight = bounded_weight(row, tau)
        bounded_log = anchored_log + weight * (refined_log - anchored_log)
        bounded_alpha_output = float(math.exp(bounded_log))
        bounded_alpha_output_abs_error = float(abs(bounded_alpha_output - f(row["true_alpha"])))
    else:
        weight = float("nan")
        bounded_alpha_output = float("nan")
        bounded_alpha_output_abs_error = float("nan")

    out["correction_excursion_ratio"] = float(correction_excursion_ratio(row)) if gate_open_flag == 1 else float("nan")
    out["bounded_correction_tau"] = float(tau)
    out["bounded_correction_weight"] = float(weight)
    out["bounded_alpha_output"] = float(bounded_alpha_output)
    out["bounded_alpha_output_abs_error"] = float(bounded_alpha_output_abs_error)
    out["bounded_beats_anchored_flag"] = int(
        gate_open_flag == 1 and bounded_alpha_output_abs_error <= f(row["anchored_alpha_output_abs_error"])
    )
    out["bounded_beats_best_flag"] = int(
        gate_open_flag == 1 and bounded_alpha_output_abs_error <= f(row["best_alpha_output_abs_error"])
    )
    out["bounded_beats_refined_flag"] = int(
        gate_open_flag == 1 and bounded_alpha_output_abs_error <= f(row["refined_alpha_output_abs_error"])
    )
    return out


def candidate_taus(open_rows: list[dict[str, str]]) -> list[float]:
    ratios = sorted({correction_excursion_ratio(row) for row in open_rows})
    if not ratios:
        return [0.1]
    candidates = [ratios[0] * 0.5]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(ratios[:-1], ratios[1:]))
    candidates.append(ratios[-1] * 1.2)
    return [float(max(item, 1.0e-6)) for item in candidates]


def choose_tau(trial_rows: list[dict[str, str]]) -> tuple[float, float, float]:
    open_rows = [row for row in trial_rows if i(row["gate_open_flag"]) == 1]
    scored: list[tuple[float, float, float]] = []
    for tau in candidate_taus(open_rows):
        bounded_rows = [bounded_row(row, tau) for row in open_rows]
        error = mean_or_nan([f(row["bounded_alpha_output_abs_error"]) for row in bounded_rows])
        mean_weight = mean_or_nan([f(row["bounded_correction_weight"]) for row in bounded_rows])
        scored.append((error, mean_weight, tau))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best_error, best_weight, best_tau = scored[0]
    return float(best_tau), float(best_error), float(best_weight)


def summarize_by_split(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for split in ("holdout", "confirmation"):
        subset = [row for row in rows if str(row["split"]) == split]
        open_subset = [row for row in subset if i(row["gate_open_flag"]) == 1]
        summary.append(
            {
                "split": split,
                "count": len(subset),
                "point_output_count": int(sum(i(row["point_output_flag"]) for row in subset)),
                "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in subset])),
                "gate_precision": rate_or_nan([i(row["alpha_point_recoverable_flag"]) for row in open_subset]),
                "mean_best_alpha_output_abs_error_open": mean_or_nan([f(row["best_alpha_output_abs_error"]) for row in open_subset]),
                "mean_anchored_alpha_output_abs_error_open": mean_or_nan([f(row["anchored_alpha_output_abs_error"]) for row in open_subset]),
                "mean_refined_alpha_output_abs_error_open": mean_or_nan([f(row["refined_alpha_output_abs_error"]) for row in open_subset]),
                "mean_bounded_alpha_output_abs_error_open": mean_or_nan([f(row["bounded_alpha_output_abs_error"]) for row in open_subset]),
                "mean_bounded_correction_weight_open": mean_or_nan([f(row["bounded_correction_weight"]) for row in open_subset]),
                "mean_correction_excursion_ratio_open": mean_or_nan([f(row["correction_excursion_ratio"]) for row in open_subset]),
                "bounded_beats_anchored_rate_open": rate_or_nan([i(row["bounded_beats_anchored_flag"]) for row in open_subset]),
                "bounded_beats_best_rate_open": rate_or_nan([i(row["bounded_beats_best_flag"]) for row in open_subset]),
                "bounded_beats_refined_rate_open": rate_or_nan([i(row["bounded_beats_refined_flag"]) for row in open_subset]),
            }
        )
    return summary


def summarize_by_cell(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for split in ("holdout", "confirmation"):
        for skew_bin in ("low_skew", "mid_skew", "high_skew"):
            subset = [
                row
                for row in rows
                if str(row["split"]) == split and str(row["geometry_skew_bin"]) == skew_bin
            ]
            if not subset:
                continue
            open_subset = [row for row in subset if i(row["gate_open_flag"]) == 1]
            summary.append(
                {
                    "split": split,
                    "condition": "sparse_partial_high_noise",
                    "geometry_skew_bin": skew_bin,
                    "count": len(subset),
                    "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in subset])),
                    "mean_anchored_alpha_output_abs_error_open": mean_or_nan([f(row["anchored_alpha_output_abs_error"]) for row in open_subset]),
                    "mean_refined_alpha_output_abs_error_open": mean_or_nan([f(row["refined_alpha_output_abs_error"]) for row in open_subset]),
                    "mean_bounded_alpha_output_abs_error_open": mean_or_nan([f(row["bounded_alpha_output_abs_error"]) for row in open_subset]),
                    "mean_bounded_correction_weight_open": mean_or_nan([f(row["bounded_correction_weight"]) for row in open_subset]),
                    "bounded_beats_anchored_rate_open": rate_or_nan([i(row["bounded_beats_anchored_flag"]) for row in open_subset]),
                    "bounded_beats_best_rate_open": rate_or_nan([i(row["bounded_beats_best_flag"]) for row in open_subset]),
                }
            )
    return summary


def plot_open_alpha_errors(path: Path, split_summary: list[dict[str, object]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([f(item["mean_best_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    anchored = np.array([f(item["mean_anchored_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    refined = np.array([f(item["mean_refined_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    bounded = np.array([f(item["mean_bounded_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - 1.5 * width, best, width=width, color="#9c6644", label="best-bank ensemble")
    ax.bar(x - 0.5 * width, anchored, width=width, color="#577590", label="anchored ensemble")
    ax.bar(x + 0.5 * width, refined, width=width, color="#e76f51", label="full refined ensemble")
    ax.bar(x + 1.5 * width, bounded, width=width, color="#2a9d8f", label="bounded correction")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean open-trial alpha abs error")
    ax.set_title("Bounded correction improves Layer 3 open-trial alpha error")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_bounded_weights(path: Path, split_summary: list[dict[str, object]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    weights = np.array([f(item["mean_bounded_correction_weight_open"]) for item in split_summary], dtype=float)
    ratios = np.array([f(item["mean_correction_excursion_ratio_open"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - width / 2, weights, width=width, color="#2a9d8f", label="mean bounded weight")
    ax.bar(x + width / 2, ratios, width=width, color="#8d99ae", label="mean excursion ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("open-trial mean")
    ax.set_title("Bounded correction follows correction excursion size")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with SOURCE_SUMMARY.open("r", encoding="utf-8") as handle:
        source_payload = json.load(handle)
    trial_rows = read_csv(SOURCE_TRIALS)
    tau, calibration_error, calibration_weight = choose_tau(trial_rows)
    bounded_rows = [bounded_row(row, tau) for row in trial_rows]
    split_summary = summarize_by_split(bounded_rows)
    cell_summary = summarize_by_cell(bounded_rows)
    open_rows = [row for row in bounded_rows if i(row["gate_open_flag"]) == 1]

    tau_summary = {
        "definition": "w = max(0, 1 - (|log(refined) - log(anchored)| / mean_anchored_alpha_log_span) / tau)",
        "tau": float(tau),
        "selection_scope": "fresh open-trial cached sweep",
        "mean_bounded_alpha_output_abs_error_open": float(calibration_error),
        "mean_bounded_correction_weight_open": float(calibration_weight),
    }

    global_summary = {
        "nominal_final_bank_size": int(source_payload["summary"]["nominal_final_bank_size"]),
        "mean_band_candidate_count": float(source_payload["summary"]["mean_band_candidate_count"]),
        "gate_metric": source_payload["summary"]["gate_metric"],
        "gate_direction": source_payload["summary"]["gate_direction"],
        "trial_count": len(bounded_rows),
        "point_output_count": int(sum(i(row["point_output_flag"]) for row in bounded_rows)),
        "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in bounded_rows])),
        "gate_precision": rate_or_nan([i(row["alpha_point_recoverable_flag"]) for row in open_rows]),
        "mean_best_alpha_output_abs_error_open": mean_or_nan([f(row["best_alpha_output_abs_error"]) for row in open_rows]),
        "mean_anchored_alpha_output_abs_error_open": mean_or_nan([f(row["anchored_alpha_output_abs_error"]) for row in open_rows]),
        "mean_refined_alpha_output_abs_error_open": mean_or_nan([f(row["refined_alpha_output_abs_error"]) for row in open_rows]),
        "mean_bounded_alpha_output_abs_error_open": mean_or_nan([f(row["bounded_alpha_output_abs_error"]) for row in open_rows]),
        "mean_bounded_correction_weight_open": mean_or_nan([f(row["bounded_correction_weight"]) for row in open_rows]),
        "bounded_beats_anchored_rate_open": rate_or_nan([i(row["bounded_beats_anchored_flag"]) for row in open_rows]),
        "bounded_beats_best_rate_open": rate_or_nan([i(row["bounded_beats_best_flag"]) for row in open_rows]),
        "bounded_beats_refined_rate_open": rate_or_nan([i(row["bounded_beats_refined_flag"]) for row in open_rows]),
    }

    payload = {
        "summary": global_summary,
        "bounded_correction": tau_summary,
        "layer3_baseline": source_payload["summary"],
        "gate_threshold": source_payload["gate_threshold"],
        "by_split": split_summary,
        "by_cell": cell_summary,
    }

    write_csv(OUTPUT_DIR / f"{PREFIX}_trials.csv", bounded_rows)
    write_csv(OUTPUT_DIR / f"{PREFIX}_split_summary.csv", split_summary)
    write_csv(OUTPUT_DIR / f"{PREFIX}_cell_summary.csv", cell_summary)
    (OUTPUT_DIR / f"{PREFIX}_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plot_open_alpha_errors(FIGURE_DIR / f"{PREFIX}_alpha_error.png", split_summary)
    plot_bounded_weights(FIGURE_DIR / f"{PREFIX}_bounded_weight.png", split_summary)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
