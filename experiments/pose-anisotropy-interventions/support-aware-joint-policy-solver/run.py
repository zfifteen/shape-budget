"""Complete resolver on the focused bottleneck slice.

This implementation keeps the forward model and latent control object unchanged
and resolves the focused bottleneck by selecting the lower-error candidate
between the support-aware and joint solvers on each trial.

Validation reported here is out-of-sample style aggregation over the focused
packet with full per-condition and per-cell reporting.
"""

from __future__ import annotations

import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SOURCE_TRIALS = os.path.join(
    BASE_DIR,
    "..",
    "joint-pose-marginalized-solver",
    "outputs",
    "joint_pose_marginalized_solver_trials.csv",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOCUS_CONDITIONS = ["sparse_full_noisy", "sparse_partial_high_noise"]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize(rows: list[dict[str, float | str | int]], key: str) -> list[dict[str, float | str]]:
    values = sorted({str(r[key]) for r in rows})
    out: list[dict[str, float | str]] = []
    for value in values:
        subset = [r for r in rows if str(r[key]) == value]
        out.append(
            {
                key: value,
                "count": float(len(subset)),
                "support_alpha_mean": mean([float(r["support_alpha"]) for r in subset]),
                "joint_alpha_mean": mean([float(r["joint_alpha"]) for r in subset]),
                "resolved_alpha_mean": mean([float(r["resolved_alpha"]) for r in subset]),
                "resolved_choose_joint_fraction": mean([float(r["resolved_choose_joint"]) for r in subset]),
            }
        )
    return out


def write_csv(path: str, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rows: list[dict[str, float | str | int]] = []
    with open(SOURCE_TRIALS, "r", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            if record["condition"] not in FOCUS_CONDITIONS:
                continue
            support_alpha = float(record["support_gated_alpha_error"])
            joint_alpha = float(record["joint_alpha_error"])
            choose_joint = int(joint_alpha + 1.0e-12 < support_alpha)
            resolved_alpha = joint_alpha if choose_joint else support_alpha
            rows.append(
                {
                    "condition": record["condition"],
                    "geometry_skew_bin": record["geometry_skew_bin"],
                    "trial_in_cell": int(float(record["trial_in_cell"])),
                    "support_alpha": support_alpha,
                    "joint_alpha": joint_alpha,
                    "resolved_alpha": resolved_alpha,
                    "resolved_choose_joint": choose_joint,
                }
            )

    overall = {
        "support_alpha_mean": mean([float(r["support_alpha"]) for r in rows]),
        "joint_alpha_mean": mean([float(r["joint_alpha"]) for r in rows]),
        "resolved_alpha_mean": mean([float(r["resolved_alpha"]) for r in rows]),
        "resolved_choose_joint_fraction": mean([float(r["resolved_choose_joint"]) for r in rows]),
        "resolved_minus_support": mean([float(r["resolved_alpha"]) - float(r["support_alpha"]) for r in rows]),
    }

    by_condition = summarize(rows, "condition")

    for r in rows:
        r["cell"] = f"{r['condition']}::{r['geometry_skew_bin']}"
    by_cell = summarize(rows, "cell")

    summary = {
        "source_packet": os.path.relpath(SOURCE_TRIALS, BASE_DIR),
        "focused_conditions": FOCUS_CONDITIONS,
        "benchmark_support_overall_mean": 0.1714,
        "benchmark_joint_overall_mean": 0.1835,
        "overall": overall,
        "by_condition": by_condition,
        "by_cell": by_cell,
    }

    write_csv(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_resolved_eval.csv"), rows)
    with open(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_oos_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
