"""Out-of-sample validated support-aware joint policy solver.

This script evaluates policy-gated routing between support-aware and joint
candidates on the focused bottleneck slice using disjoint validation:

1) leave-one-trial-out (LOTO)
2) leave-one-cell-out (LOCO)

Calibration and evaluation are explicitly separated in each fold.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

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
GEOMETRY_SKEW_BIN_LABELS = ["low_skew", "mid_skew", "high_skew"]
FOCUS_ALPHA_BIN = "moderate"

@dataclass
class FoldParams:
    entropy_partial: float
    ratio_partial: float
    entropy_full: float
    ratio_full: float


@dataclass
class Trial:
    row_id: int
    condition: str
    geometry_skew_bin: str
    trial_in_cell: int
    support_alpha: float
    support_fit: float
    joint_alpha: float
    joint_fit: float
    joint_entropy: float
    oracle_pose_alpha: float


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def parse_trials() -> list[Trial]:
    trials: list[Trial] = []
    with open(SOURCE_TRIALS, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            if row["condition"] not in FOCUS_CONDITIONS:
                continue
            trials.append(
                Trial(
                    row_id=idx,
                    condition=row["condition"],
                    geometry_skew_bin=row["geometry_skew_bin"],
                    trial_in_cell=int(float(row["trial_in_cell"])),
                    support_alpha=float(row["support_gated_alpha_error"]),
                    support_fit=float(row["support_gated_fit_rmse"]),
                    joint_alpha=float(row["joint_alpha_error"]),
                    joint_fit=float(row["joint_fit_rmse"]),
                    joint_entropy=float(row["joint_pose_entropy"]),
                    oracle_pose_alpha=float(row["oracle_pose_alpha_error"]),
                )
            )
    return trials


def choose_joint(trial: Trial, p: FoldParams) -> int:
    if trial.condition == "sparse_partial_high_noise":
        return int(trial.joint_entropy <= p.entropy_partial and trial.joint_fit <= trial.support_fit * p.ratio_partial)
    return int(trial.joint_entropy <= p.entropy_full and trial.joint_fit <= trial.support_fit * p.ratio_full)


def policy_alpha(trial: Trial, p: FoldParams) -> float:
    return trial.joint_alpha if choose_joint(trial, p) else trial.support_alpha


def calibrate(train_trials: list[Trial]) -> FoldParams:
    best_params = FoldParams(0.62, 1.20, 0.76, 1.16)
    best_error = float("inf")

    entropy_values = sorted({round(t.joint_entropy, 12) for t in train_trials})
    ratio_values = sorted({round(t.joint_fit / max(t.support_fit, 1.0e-12), 12) for t in train_trials})

    # add conservative/open sentinels so the search can choose "almost never" and
    # "almost always" routes.
    entropy_values = [min(entropy_values) - 1.0e-9] + entropy_values + [max(entropy_values) + 1.0e-9]
    ratio_values = [min(ratio_values) - 1.0e-9] + ratio_values + [max(ratio_values) + 1.0e-9]

    for ep in entropy_values:
        for rp in ratio_values:
            for ef in entropy_values:
                for rf in ratio_values:
                    params = FoldParams(float(ep), float(rp), float(ef), float(rf))
                    err = mean([policy_alpha(t, params) for t in train_trials])
                    if err + 1.0e-12 < best_error:
                        best_error = err
                        best_params = params
    return best_params


def summarize(rows: list[dict[str, float | str | int]], key_field: str) -> list[dict[str, float | str]]:
    keys = sorted({str(r[key_field]) for r in rows})
    out: list[dict[str, float | str]] = []
    for key in keys:
        subset = [r for r in rows if str(r[key_field]) == key]
        out.append(
            {
                key_field: key,
                "count": float(len(subset)),
                "support_alpha_mean": mean([float(r["support_alpha"]) for r in subset]),
                "joint_alpha_mean": mean([float(r["joint_alpha"]) for r in subset]),
                "policy_alpha_mean": mean([float(r["policy_alpha"]) for r in subset]),
                "oracle_two_alpha_mean": mean([float(r["oracle_two_alpha"]) for r in subset]),
                "policy_choose_joint_fraction": mean([float(r["policy_choose_joint"]) for r in subset]),
            }
        )
    return out


def write_csv(path: str, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_loto(trials: list[Trial]) -> dict[str, object]:
    eval_rows: list[dict[str, float | str | int]] = []
    calibration_rows: list[dict[str, float | str | int]] = []

    for i, holdout in enumerate(trials):
        train = [t for j, t in enumerate(trials) if j != i]
        params = calibrate(train)

        choose = choose_joint(holdout, params)
        policy = holdout.joint_alpha if choose else holdout.support_alpha
        eval_rows.append(
            {
                "row_id": holdout.row_id,
                "condition": holdout.condition,
                "geometry_skew_bin": holdout.geometry_skew_bin,
                "trial_in_cell": holdout.trial_in_cell,
                "support_alpha": holdout.support_alpha,
                "joint_alpha": holdout.joint_alpha,
                "policy_alpha": policy,
                "oracle_two_alpha": min(holdout.support_alpha, holdout.joint_alpha),
                "oracle_pose_alpha": holdout.oracle_pose_alpha,
                "policy_choose_joint": choose,
            }
        )

        calibration_rows.append(
            {
                "fold": i,
                "holdout_row_id": holdout.row_id,
                "holdout_condition": holdout.condition,
                "holdout_geometry_skew_bin": holdout.geometry_skew_bin,
                "entropy_partial": params.entropy_partial,
                "ratio_partial": params.ratio_partial,
                "entropy_full": params.entropy_full,
                "ratio_full": params.ratio_full,
                "train_policy_alpha_mean": mean([policy_alpha(t, params) for t in train]),
                "train_support_alpha_mean": mean([t.support_alpha for t in train]),
                "train_joint_alpha_mean": mean([t.joint_alpha for t in train]),
            }
        )

    overall = {
        "support_alpha_mean": mean([float(r["support_alpha"]) for r in eval_rows]),
        "joint_alpha_mean": mean([float(r["joint_alpha"]) for r in eval_rows]),
        "policy_alpha_mean": mean([float(r["policy_alpha"]) for r in eval_rows]),
        "oracle_two_alpha_mean": mean([float(r["oracle_two_alpha"]) for r in eval_rows]),
        "policy_minus_support": mean([float(r["policy_alpha"]) - float(r["support_alpha"]) for r in eval_rows]),
        "policy_minus_joint": mean([float(r["policy_alpha"]) - float(r["joint_alpha"]) for r in eval_rows]),
        "policy_choose_joint_fraction": mean([float(r["policy_choose_joint"]) for r in eval_rows]),
    }

    by_condition = summarize(eval_rows, "condition")

    for row in eval_rows:
        row["cell"] = f"{row['condition']}::{row['geometry_skew_bin']}"
    by_cell = summarize(eval_rows, "cell")

    return {
        "evaluation_rows": eval_rows,
        "calibration_rows": calibration_rows,
        "overall": overall,
        "by_condition": by_condition,
        "by_cell": by_cell,
    }


def run_loco(trials: list[Trial]) -> dict[str, object]:
    cells = sorted({(t.condition, t.geometry_skew_bin) for t in trials})
    eval_rows: list[dict[str, float | str | int]] = []
    calibration_rows: list[dict[str, float | str | int]] = []

    for fold_idx, (cond, skew) in enumerate(cells):
        holdout = [t for t in trials if t.condition == cond and t.geometry_skew_bin == skew]
        train = [t for t in trials if not (t.condition == cond and t.geometry_skew_bin == skew)]
        params = calibrate(train)

        calibration_rows.append(
            {
                "fold": fold_idx,
                "holdout_cell": f"{cond}::{skew}",
                "entropy_partial": params.entropy_partial,
                "ratio_partial": params.ratio_partial,
                "entropy_full": params.entropy_full,
                "ratio_full": params.ratio_full,
                "train_policy_alpha_mean": mean([policy_alpha(t, params) for t in train]),
                "train_support_alpha_mean": mean([t.support_alpha for t in train]),
                "train_joint_alpha_mean": mean([t.joint_alpha for t in train]),
            }
        )

        for t in holdout:
            choose = choose_joint(t, params)
            eval_rows.append(
                {
                    "row_id": t.row_id,
                    "condition": t.condition,
                    "geometry_skew_bin": t.geometry_skew_bin,
                    "trial_in_cell": t.trial_in_cell,
                    "support_alpha": t.support_alpha,
                    "joint_alpha": t.joint_alpha,
                    "policy_alpha": t.joint_alpha if choose else t.support_alpha,
                    "oracle_two_alpha": min(t.support_alpha, t.joint_alpha),
                    "policy_choose_joint": choose,
                }
            )

    overall = {
        "support_alpha_mean": mean([float(r["support_alpha"]) for r in eval_rows]),
        "joint_alpha_mean": mean([float(r["joint_alpha"]) for r in eval_rows]),
        "policy_alpha_mean": mean([float(r["policy_alpha"]) for r in eval_rows]),
        "oracle_two_alpha_mean": mean([float(r["oracle_two_alpha"]) for r in eval_rows]),
        "policy_minus_support": mean([float(r["policy_alpha"]) - float(r["support_alpha"]) for r in eval_rows]),
        "policy_minus_joint": mean([float(r["policy_alpha"]) - float(r["joint_alpha"]) for r in eval_rows]),
        "policy_choose_joint_fraction": mean([float(r["policy_choose_joint"]) for r in eval_rows]),
    }
    by_condition = summarize(eval_rows, "condition")
    for row in eval_rows:
        row["cell"] = f"{row['condition']}::{row['geometry_skew_bin']}"
    by_cell = summarize(eval_rows, "cell")

    return {
        "evaluation_rows": eval_rows,
        "calibration_rows": calibration_rows,
        "overall": overall,
        "by_condition": by_condition,
        "by_cell": by_cell,
    }


def main() -> None:
    trials = parse_trials()

    loto = run_loto(trials)
    loco = run_loco(trials)

    summary = {
        "source_packet": os.path.relpath(SOURCE_TRIALS, BASE_DIR),
        "focused_conditions": FOCUS_CONDITIONS,
        "focused_alpha_bin": FOCUS_ALPHA_BIN,
        "benchmark_support_overall_mean": 0.1714,
        "benchmark_joint_overall_mean": 0.1835,
        "validation": {
            "leave_one_trial_out": {
                "overall": loto["overall"],
                "by_condition": loto["by_condition"],
                "by_cell": loto["by_cell"],
            },
            "leave_one_cell_out": {
                "overall": loco["overall"],
                "by_condition": loco["by_condition"],
                "by_cell": loco["by_cell"],
            },
        },
    }

    write_csv(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_loto_eval.csv"), loto["evaluation_rows"]) 
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_loto_calibration.csv"), loto["calibration_rows"]) 
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_loco_eval.csv"), loco["evaluation_rows"]) 
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_loco_calibration.csv"), loco["calibration_rows"]) 

    with open(os.path.join(OUTPUT_DIR, "support_aware_joint_policy_solver_oos_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
