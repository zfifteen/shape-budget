"""Support-aware pose-marginalized policy solver (packet-level iteration).

Design goal: keep the support-aware baseline's robustness while preserving the
joint solver's complementary wins using an observable reliability gate.

This script operates on the same-trial packet produced by the current
`joint-pose-marginalized-solver` experiment and outputs a new solver packet.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
SOURCE_TRIALS = os.path.join(
    BASE_DIR,
    "..",
    "joint-pose-marginalized-solver",
    "outputs",
    "joint_pose_marginalized_solver_trials.csv",
)
os.makedirs(FIGURE_DIR, exist_ok=True)

FOCUS_CONDITIONS = ["sparse_full_noisy", "sparse_partial_high_noise"]
FOCUS_ALPHA_BIN = "moderate"
GEOMETRY_SKEW_BIN_LABELS = ["low_skew", "mid_skew", "high_skew"]


@dataclass
class TrialRow:
    condition: str
    geometry_skew_bin: str
    trial_in_cell: int
    support_gated_alpha_error: float
    support_gated_fit_rmse: float
    joint_alpha_error: float
    joint_fit_rmse: float
    joint_score: float
    joint_pose_entropy: float
    policy_alpha_error: float
    policy_fit_rmse: float
    policy_choose_joint: int
    oracle_two_alpha_error: float
    oracle_pose_alpha_error: float


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def choose_policy(row: dict[str, str]) -> int:
    condition = row["condition"]
    entropy = to_float(row, "joint_pose_entropy")
    joint_fit = to_float(row, "joint_fit_rmse")
    support_fit = to_float(row, "support_gated_fit_rmse")

    if condition == "sparse_partial_high_noise":
        return int(entropy <= 0.62 and joint_fit <= support_fit * 1.20)
    return int(entropy <= 0.76 and joint_fit <= support_fit * 1.16)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def write_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_simple_svg(path: str, title: str, labels: list[str], a: list[float], b: list[float], c: list[float]) -> None:
    width = 840
    height = 360
    margin = 50
    chart_h = height - 2 * margin
    max_v = max(a + b + c + [1e-9])
    bar_w = 28
    group_gap = 90

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width//2}" y="26" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>',
    ]

    for i, label in enumerate(labels):
        gx = margin + i * (3 * bar_w + group_gap)
        vals = [a[i], b[i], c[i]]
        colors = ["#2a9d8f", "#e76f51", "#264653"]
        for j, (v, color) in enumerate(zip(vals, colors)):
            h = int((v / max_v) * (chart_h - 10))
            x = gx + j * bar_w
            y = height - margin - h
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_w-4}" height="{h}" fill="{color}"/>')
            lines.append(f'<text x="{x+10}" y="{y-6}" font-size="10" font-family="monospace">{v:.3f}</text>')
        lines.append(f'<text x="{gx+bar_w}" y="{height-margin+18}" font-size="11" font-family="sans-serif">{label}</text>')

    lines.extend(
        [
            '<text x="560" y="70" font-size="11" font-family="sans-serif" fill="#2a9d8f">support-aware</text>',
            '<text x="560" y="88" font-size="11" font-family="sans-serif" fill="#e76f51">joint solver</text>',
            '<text x="560" y="106" font-size="11" font-family="sans-serif" fill="#264653">new policy solver</text>',
            "</svg>",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    with open(SOURCE_TRIALS, "r", encoding="utf-8") as f:
        source_rows = [row for row in csv.DictReader(f) if row["condition"] in FOCUS_CONDITIONS]

    out_rows: list[TrialRow] = []
    for row in source_rows:
        choose_joint = choose_policy(row)
        support_alpha = to_float(row, "support_gated_alpha_error")
        joint_alpha = to_float(row, "joint_alpha_error")
        support_fit = to_float(row, "support_gated_fit_rmse")
        joint_fit = to_float(row, "joint_fit_rmse")

        out_rows.append(
            TrialRow(
                condition=row["condition"],
                geometry_skew_bin=row["geometry_skew_bin"],
                trial_in_cell=int(float(row["trial_in_cell"])),
                support_gated_alpha_error=support_alpha,
                support_gated_fit_rmse=support_fit,
                joint_alpha_error=joint_alpha,
                joint_fit_rmse=joint_fit,
                joint_score=to_float(row, "joint_score"),
                joint_pose_entropy=to_float(row, "joint_pose_entropy"),
                policy_alpha_error=joint_alpha if choose_joint else support_alpha,
                policy_fit_rmse=joint_fit if choose_joint else support_fit,
                policy_choose_joint=choose_joint,
                oracle_two_alpha_error=min(support_alpha, joint_alpha),
                oracle_pose_alpha_error=to_float(row, "oracle_pose_alpha_error"),
            )
        )

    by_condition: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        subset = [r for r in out_rows if r.condition == condition]
        by_condition.append(
            {
                "condition": condition,
                "support_gated_alpha_error_mean": mean([r.support_gated_alpha_error for r in subset]),
                "joint_alpha_error_mean": mean([r.joint_alpha_error for r in subset]),
                "policy_alpha_error_mean": mean([r.policy_alpha_error for r in subset]),
                "oracle_two_alpha_error_mean": mean([r.oracle_two_alpha_error for r in subset]),
                "oracle_pose_alpha_error_mean": mean([r.oracle_pose_alpha_error for r in subset]),
                "policy_choose_joint_fraction": mean([float(r.policy_choose_joint) for r in subset]),
                "joint_pose_entropy_mean": mean([r.joint_pose_entropy for r in subset]),
            }
        )

    grouped: dict[tuple[str, str], list[TrialRow]] = defaultdict(list)
    for r in out_rows:
        grouped[(r.condition, r.geometry_skew_bin)].append(r)

    by_cell: list[dict[str, float | str]] = []
    for condition in FOCUS_CONDITIONS:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            subset = grouped[(condition, skew_bin)]
            if not subset:
                continue
            by_cell.append(
                {
                    "condition": condition,
                    "alpha_strength_bin": FOCUS_ALPHA_BIN,
                    "geometry_skew_bin": skew_bin,
                    "count": len(subset),
                    "support_gated_alpha_error_mean": mean([r.support_gated_alpha_error for r in subset]),
                    "joint_alpha_error_mean": mean([r.joint_alpha_error for r in subset]),
                    "policy_alpha_error_mean": mean([r.policy_alpha_error for r in subset]),
                    "oracle_two_alpha_error_mean": mean([r.oracle_two_alpha_error for r in subset]),
                    "policy_choose_joint_fraction": mean([float(r.policy_choose_joint) for r in subset]),
                }
            )

    focused_support = mean([r.support_gated_alpha_error for r in out_rows])
    focused_joint = mean([r.joint_alpha_error for r in out_rows])
    focused_policy = mean([r.policy_alpha_error for r in out_rows])
    focused_oracle_two = mean([r.oracle_two_alpha_error for r in out_rows])

    summary = {
        "source_packet": "joint_pose_marginalized_solver_trials.csv",
        "focused_conditions": FOCUS_CONDITIONS,
        "focused_alpha_bin": FOCUS_ALPHA_BIN,
        "focused_support_gated_mean_alpha_error": focused_support,
        "focused_joint_mean_alpha_error": focused_joint,
        "focused_policy_mean_alpha_error": focused_policy,
        "focused_oracle_two_mean_alpha_error": focused_oracle_two,
        "policy_minus_support_alpha_error": focused_policy - focused_support,
        "policy_minus_joint_alpha_error": focused_policy - focused_joint,
        "policy_gap_to_oracle_two": focused_policy - focused_oracle_two,
        "condition_means": by_condition,
    }

    complementarity = {
        "focused_support_gated_mean_alpha_error": focused_support,
        "focused_joint_mean_alpha_error": focused_joint,
        "focused_policy_mean_alpha_error": focused_policy,
        "focused_oracle_two_mean_alpha_error": focused_oracle_two,
        "policy_choose_joint_fraction": mean([float(r.policy_choose_joint) for r in out_rows]),
        "policy_oracle_alignment_fraction": mean(
            [
                float(
                    (r.policy_choose_joint == 1 and r.joint_alpha_error <= r.support_gated_alpha_error + 1.0e-12)
                    or (r.policy_choose_joint == 0 and r.support_gated_alpha_error <= r.joint_alpha_error + 1.0e-12)
                )
                for r in out_rows
            ]
        ),
    }

    write_csv(
        os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_trials.csv"),
        [r.__dict__ for r in out_rows],
    )
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_summary.csv"), by_condition)
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_cells.csv"), by_cell)

    with open(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_complementarity.json"), "w", encoding="utf-8") as f:
        json.dump(complementarity, f, indent=2)

    labels = [str(r["condition"]) for r in by_condition]
    support_vals = [float(r["support_gated_alpha_error_mean"]) for r in by_condition]
    joint_vals = [float(r["joint_alpha_error_mean"]) for r in by_condition]
    policy_vals = [float(r["policy_alpha_error_mean"]) for r in by_condition]
    build_simple_svg(
        os.path.join(FIGURE_DIR, "support_aware_joint_policy_solver_overview.svg"),
        "Support-aware joint policy solver: focused condition means",
        labels,
        support_vals,
        joint_vals,
        policy_vals,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
