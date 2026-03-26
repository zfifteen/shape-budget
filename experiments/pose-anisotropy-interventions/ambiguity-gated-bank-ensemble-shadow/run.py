"""
Shadow A/B for swapping the focused ensemble gate from entropy to ambiguity.

This experiment is deliberately minimal:

- keep the current cached candidate families
- keep the current four-way chooser
- keep the same default fallback candidate
- freeze threshold selection to calibration only
- compare the current entropy gate against an ambiguity-width gate on the same
  trials
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

ensure_caches, load_merged_rows, fit_four_way_chooser, choose_names_from_model, candidate_alpha_errors, numeric_value, CALIBRATION_BLOCKS, DEFAULT_CANDIDATE, FOCUS_CONDITIONS, GEOMETRY_SKEW_BIN_LABELS, FOCUS_ALPHA_BIN, GATE_FEATURE = load_symbols(
    "run_entropy_gated_bank_ensemble_shadow_source",
    ROOT / "experiments/pose-anisotropy-interventions/entropy-gated-bank-ensemble-solver/run.py",
    "ensure_caches",
    "load_merged_rows",
    "fit_four_way_chooser",
    "choose_names_from_model",
    "candidate_alpha_errors",
    "numeric_value",
    "CALIBRATION_BLOCKS",
    "DEFAULT_CANDIDATE",
    "FOCUS_CONDITIONS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "FOCUS_ALPHA_BIN",
    "GATE_FEATURE",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AMBIGUITY_TRIALS_PATH = (
    ROOT
    / "experiments/pose-anisotropy-diagnostics/ambiguity-width-diagnostic/outputs/ambiguity_width_diagnostic_trials.csv"
)

POLICY_DEFAULT = "default_only"
POLICY_ENTROPY = "entropy_gate"
POLICY_AMBIGUITY = "ambiguity_gate"
AMBIGUITY_FEATURE = "mean_ambiguity_ratio"

BLOCK_TO_SPLIT = {
    "calibration_block_1": "calibration",
    "calibration_block_2": "calibration",
    "holdout_block_1": "holdout",
    "confirmation_block": "confirmation",
}
FRESH_BLOCKS = ("holdout_block_1", "confirmation_block")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key_from_base(row: dict[str, object]) -> tuple[str, int, str, str]:
    block_name = str(row["block_name"])
    return (
        BLOCK_TO_SPLIT[block_name],
        int(row["observation_seed"]),
        str(row["condition"]),
        str(row["geometry_skew_bin"]),
    )


def row_key_from_shadow(row: dict[str, object]) -> tuple[str, int, str, str]:
    return (
        str(row["split"]),
        int(float(row["observation_seed"])),
        str(row["condition"]),
        str(row["geometry_skew_bin"]),
    )


def load_ambiguity_values() -> dict[tuple[str, int, str, str], dict[str, float]]:
    rows = read_csv_rows(AMBIGUITY_TRIALS_PATH)
    values: dict[tuple[str, int, str, str], dict[str, float]] = {}
    for row in rows:
        values[row_key_from_shadow(row)] = {
            "mean_ambiguity_ratio": float(row["mean_ambiguity_ratio"]),
            "max_ambiguity_ratio": float(row["max_ambiguity_ratio"]),
            "mean_alpha_log_span_set": float(row["mean_alpha_log_span_set"]),
            "alpha_bank_log_span": float(row["alpha_bank_log_span"]),
            "geometry_bank_span_norm": float(row["geometry_bank_span_norm"]),
            "alpha_unstable_flag": float(row["alpha_unstable_flag"]),
        }
    return values


def select_gate_threshold(
    calibration_rows: list[dict[str, object]],
    chooser_prediction_names: list[str],
    gate_values: list[float],
) -> tuple[float, dict[str, float]]:
    gate_array = np.array(gate_values, dtype=float)
    threshold_candidates = np.unique(np.quantile(gate_array, np.linspace(0.05, 0.95, 19)))
    median_gate_value = float(np.median(gate_array))

    best_threshold = None
    best_score = None
    best_worst_block = None
    best_block_scores = None

    for threshold in threshold_candidates:
        chosen_names = [
            predicted if gate_value >= threshold else DEFAULT_CANDIDATE
            for predicted, gate_value in zip(chooser_prediction_names, gate_array)
        ]
        score = float(
            np.mean(
                [
                    candidate_alpha_errors(row)[name]
                    for row, name in zip(calibration_rows, chosen_names)
                ]
            )
        )
        per_block = {}
        for block_name in CALIBRATION_BLOCKS:
            block_values = [
                candidate_alpha_errors(row)[name]
                for row, name in zip(calibration_rows, chosen_names)
                if row["block_name"] == block_name
            ]
            per_block[block_name] = float(np.mean(block_values))
        worst_block_score = max(per_block.values())

        candidate_key = (
            score,
            worst_block_score,
            abs(float(threshold) - median_gate_value),
        )
        best_key = (
            best_score,
            best_worst_block,
            abs(float(best_threshold) - median_gate_value) if best_threshold is not None else math.inf,
        )
        if best_threshold is None or candidate_key < best_key:
            best_threshold = float(threshold)
            best_score = float(score)
            best_worst_block = float(worst_block_score)
            best_block_scores = per_block

    if best_threshold is None or best_block_scores is None:
        raise RuntimeError("Failed to select a gate threshold.")

    return best_threshold, {key: float(value) for key, value in best_block_scores.items()}


def build_policy_rows(
    rows: list[dict[str, object]],
    chooser_prediction_names: list[str],
    chooser_confidences: list[float],
    entropy_gate_values: list[float],
    ambiguity_gate_values: list[float],
    entropy_threshold: float,
    ambiguity_threshold: float,
) -> list[dict[str, object]]:
    trial_rows: list[dict[str, object]] = []
    for row, chooser_name, chooser_confidence, entropy_value, ambiguity_value in zip(
        rows,
        chooser_prediction_names,
        chooser_confidences,
        entropy_gate_values,
        ambiguity_gate_values,
    ):
        candidate_errors = candidate_alpha_errors(row)
        best_single_name = min(candidate_errors, key=candidate_errors.get)
        baseline = row["baseline"]
        density = row["density"]

        default_name = DEFAULT_CANDIDATE
        entropy_gate_open = int(entropy_value >= entropy_threshold)
        ambiguity_gate_open = int(ambiguity_value >= ambiguity_threshold)
        entropy_name = chooser_name if entropy_gate_open == 1 else DEFAULT_CANDIDATE
        ambiguity_name = chooser_name if ambiguity_gate_open == 1 else DEFAULT_CANDIDATE

        trial_rows.append(
            {
                "block_name": str(row["block_name"]),
                "split": BLOCK_TO_SPLIT[str(row["block_name"])],
                "observation_seed": int(row["observation_seed"]),
                "condition": str(row["condition"]),
                "geometry_skew_bin": str(row["geometry_skew_bin"]),
                "alpha_strength_bin": FOCUS_ALPHA_BIN,
                "chooser_prediction": chooser_name,
                "chooser_confidence": float(chooser_confidence),
                "default_candidate": default_name,
                "default_alpha_error": float(candidate_errors[default_name]),
                "entropy_gate_feature": GATE_FEATURE,
                "entropy_gate_value": float(entropy_value),
                "entropy_gate_threshold": float(entropy_threshold),
                "entropy_gate_open": int(entropy_gate_open),
                "entropy_chosen_candidate": entropy_name,
                "entropy_chosen_alpha_error": float(candidate_errors[entropy_name]),
                "ambiguity_gate_feature": AMBIGUITY_FEATURE,
                "ambiguity_gate_value": float(ambiguity_value),
                "ambiguity_gate_threshold": float(ambiguity_threshold),
                "ambiguity_gate_open": int(ambiguity_gate_open),
                "ambiguity_chosen_candidate": ambiguity_name,
                "ambiguity_chosen_alpha_error": float(candidate_errors[ambiguity_name]),
                "ambiguity_minus_entropy_alpha_error": float(candidate_errors[ambiguity_name] - candidate_errors[entropy_name]),
                "entropy_minus_default_alpha_error": float(candidate_errors[entropy_name] - candidate_errors[default_name]),
                "ambiguity_minus_default_alpha_error": float(candidate_errors[ambiguity_name] - candidate_errors[default_name]),
                "best_single_candidate": best_single_name,
                "best_single_alpha_error": float(candidate_errors[best_single_name]),
                "oracle4_alpha_error": float(min(candidate_errors.values())),
                "baseline_support_alpha_error": float(candidate_errors["baseline_support"]),
                "baseline_joint_alpha_error": float(candidate_errors["baseline_joint"]),
                "dense_support_alpha_error": float(candidate_errors["dense_support"]),
                "dense_joint_alpha_error": float(candidate_errors["dense_joint"]),
                "baseline_support_score": numeric_value(baseline, "support_score"),
                "baseline_joint_score": numeric_value(baseline, "joint_score"),
                "dense_support_score": numeric_value(density, "support_score"),
                "dense_joint_score": numeric_value(density, "joint_score"),
                "baseline_support_entropy": numeric_value(baseline, "support_entropy"),
                "baseline_joint_entropy": numeric_value(baseline, "joint_entropy"),
                "dense_support_entropy": numeric_value(density, "support_entropy"),
                "dense_joint_entropy": numeric_value(density, "joint_entropy"),
            }
        )
    return trial_rows


def policy_mean(rows: list[dict[str, object]], field_name: str) -> float:
    return float(np.mean([float(row[field_name]) for row in rows]))


def summarize_policy(rows: list[dict[str, object]], alpha_error_field: str, gate_open_field: str | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "count": len(rows),
        "alpha_error_mean": policy_mean(rows, alpha_error_field),
        "best_single_alpha_error_mean": policy_mean(rows, "best_single_alpha_error"),
        "oracle4_alpha_error_mean": policy_mean(rows, "oracle4_alpha_error"),
        "beats_best_single": policy_mean(rows, alpha_error_field) + 1.0e-12 < policy_mean(rows, "best_single_alpha_error"),
    }
    if gate_open_field is not None:
        summary["gate_open_rate"] = policy_mean(rows, gate_open_field)

    by_condition: list[dict[str, object]] = []
    for condition in FOCUS_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            continue
        row_summary = {
            "condition": condition,
            "count": len(condition_rows),
            "alpha_error_mean": policy_mean(condition_rows, alpha_error_field),
        }
        if gate_open_field is not None:
            row_summary["gate_open_rate"] = policy_mean(condition_rows, gate_open_field)
        by_condition.append(row_summary)
    summary["by_condition"] = by_condition

    by_cell: list[dict[str, object]] = []
    for condition in FOCUS_CONDITIONS:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            cell_rows = [row for row in rows if row["condition"] == condition and row["geometry_skew_bin"] == skew_bin]
            if not cell_rows:
                continue
            row_summary = {
                "condition": condition,
                "alpha_strength_bin": FOCUS_ALPHA_BIN,
                "geometry_skew_bin": skew_bin,
                "count": len(cell_rows),
                "alpha_error_mean": policy_mean(cell_rows, alpha_error_field),
            }
            if gate_open_field is not None:
                row_summary["gate_open_rate"] = policy_mean(cell_rows, gate_open_field)
            by_cell.append(row_summary)
    summary["by_cell"] = by_cell
    return summary


def paired_delta_summary(rows: list[dict[str, object]], delta_field: str) -> dict[str, object]:
    deltas = [float(row[delta_field]) for row in rows]
    better = int(sum(delta < -1.0e-12 for delta in deltas))
    worse = int(sum(delta > 1.0e-12 for delta in deltas))
    ties = int(len(deltas) - better - worse)
    return {
        "count": len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(np.array(deltas, dtype=float))),
        "better_count": better,
        "worse_count": worse,
        "tie_count": ties,
        "better_rate": float(better / len(deltas)),
    }


def block_subset(rows: list[dict[str, object]], block_name: str) -> list[dict[str, object]]:
    return [row for row in rows if row["block_name"] == block_name]


def run(force_cache: bool = False) -> dict[str, object]:
    ensure_caches(force=force_cache)

    ambiguity_lookup = load_ambiguity_values()
    calibration_rows = load_merged_rows(CALIBRATION_BLOCKS)
    holdout_rows = load_merged_rows(("holdout_block_1",))
    confirmation_rows = load_merged_rows(("confirmation_block",))
    all_rows = calibration_rows + holdout_rows + confirmation_rows

    missing_keys = [row_key_from_base(row) for row in all_rows if row_key_from_base(row) not in ambiguity_lookup]
    if missing_keys:
        raise RuntimeError(f"Missing ambiguity rows for keys: {missing_keys[:3]}")

    chooser, _ = fit_four_way_chooser(calibration_rows)
    calibration_prediction_names, calibration_frame = choose_names_from_model(chooser, calibration_rows)
    entropy_threshold, entropy_block_scores = select_gate_threshold(
        calibration_rows,
        calibration_prediction_names,
        calibration_frame[GATE_FEATURE].astype(float).tolist(),
    )
    ambiguity_threshold, ambiguity_block_scores = select_gate_threshold(
        calibration_rows,
        calibration_prediction_names,
        [ambiguity_lookup[row_key_from_base(row)][AMBIGUITY_FEATURE] for row in calibration_rows],
    )

    all_prediction_names, all_frame = choose_names_from_model(chooser, all_rows)
    probabilities = chooser.predict_proba(all_frame)
    chooser_confidences = [float(np.max(row)) for row in probabilities]
    entropy_gate_values = all_frame[GATE_FEATURE].astype(float).tolist()
    ambiguity_gate_values = [ambiguity_lookup[row_key_from_base(row)][AMBIGUITY_FEATURE] for row in all_rows]

    trial_rows = build_policy_rows(
        all_rows,
        all_prediction_names,
        chooser_confidences,
        entropy_gate_values,
        ambiguity_gate_values,
        entropy_threshold,
        ambiguity_threshold,
    )

    calibration_trials = [row for row in trial_rows if row["split"] == "calibration"]
    holdout_trials = [row for row in trial_rows if row["block_name"] == "holdout_block_1"]
    confirmation_trials = [row for row in trial_rows if row["block_name"] == "confirmation_block"]
    fresh_trials = holdout_trials + confirmation_trials

    policies = {
        POLICY_DEFAULT: ("default_alpha_error", None),
        POLICY_ENTROPY: ("entropy_chosen_alpha_error", "entropy_gate_open"),
        POLICY_AMBIGUITY: ("ambiguity_chosen_alpha_error", "ambiguity_gate_open"),
    }

    summaries: dict[str, dict[str, object]] = {}
    for policy_name, (alpha_field, gate_field) in policies.items():
        summaries[policy_name] = {
            "calibration": summarize_policy(calibration_trials, alpha_field, gate_field),
            "holdout": summarize_policy(holdout_trials, alpha_field, gate_field),
            "confirmation": summarize_policy(confirmation_trials, alpha_field, gate_field),
            "fresh_combined": summarize_policy(fresh_trials, alpha_field, gate_field),
        }

    comparisons = {
        "holdout_ambiguity_vs_entropy": paired_delta_summary(
            holdout_trials,
            "ambiguity_minus_entropy_alpha_error",
        ),
        "confirmation_ambiguity_vs_entropy": paired_delta_summary(
            confirmation_trials,
            "ambiguity_minus_entropy_alpha_error",
        ),
        "fresh_combined_ambiguity_vs_entropy": paired_delta_summary(
            fresh_trials,
            "ambiguity_minus_entropy_alpha_error",
        ),
        "holdout_entropy_vs_default": paired_delta_summary(
            holdout_trials,
            "entropy_minus_default_alpha_error",
        ),
        "confirmation_entropy_vs_default": paired_delta_summary(
            confirmation_trials,
            "entropy_minus_default_alpha_error",
        ),
        "holdout_ambiguity_vs_default": paired_delta_summary(
            holdout_trials,
            "ambiguity_minus_default_alpha_error",
        ),
        "confirmation_ambiguity_vs_default": paired_delta_summary(
            confirmation_trials,
            "ambiguity_minus_default_alpha_error",
        ),
    }

    holdout_entropy = summaries[POLICY_ENTROPY]["holdout"]["alpha_error_mean"]
    confirmation_entropy = summaries[POLICY_ENTROPY]["confirmation"]["alpha_error_mean"]
    holdout_ambiguity = summaries[POLICY_AMBIGUITY]["holdout"]["alpha_error_mean"]
    confirmation_ambiguity = summaries[POLICY_AMBIGUITY]["confirmation"]["alpha_error_mean"]

    decisive_benefit = bool(
        holdout_ambiguity + 1.0e-12 < holdout_entropy
        and confirmation_ambiguity + 1.0e-12 < confirmation_entropy
        and comparisons["fresh_combined_ambiguity_vs_entropy"]["mean_delta"] < -1.0e-12
    )

    if decisive_benefit:
        plain_language_result = (
            "The ambiguity gate beat the entropy gate on both fresh blocks under the frozen shadow protocol."
        )
        bgp_interpretation = "strengthens BGP"
    else:
        plain_language_result = (
            "The ambiguity gate did not beat the entropy gate on both fresh blocks under the frozen shadow protocol."
        )
        bgp_interpretation = "narrows the gate-control question"

    summary = {
        "experiment": "ambiguity-gated-bank-ensemble-shadow",
        "focus_alpha_bin": FOCUS_ALPHA_BIN,
        "default_candidate": DEFAULT_CANDIDATE,
        "entropy_gate_feature": GATE_FEATURE,
        "ambiguity_gate_feature": AMBIGUITY_FEATURE,
        "entropy_gate_threshold": float(entropy_threshold),
        "ambiguity_gate_threshold": float(ambiguity_threshold),
        "entropy_calibration_block_scores": entropy_block_scores,
        "ambiguity_calibration_block_scores": ambiguity_block_scores,
        "policies": summaries,
        "comparisons": comparisons,
        "decisive_benefit": decisive_benefit,
        "plain_language_result": plain_language_result,
        "bgp_interpretation": bgp_interpretation,
    }

    write_csv(OUTPUT_DIR / "ambiguity_gated_bank_ensemble_shadow_trials.csv", trial_rows)
    write_json(OUTPUT_DIR / "ambiguity_gated_bank_ensemble_shadow_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Rebuild upstream cached candidate blocks before running the shadow comparison.",
    )
    args = parser.parse_args()
    payload = run(force_cache=args.force_cache)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
