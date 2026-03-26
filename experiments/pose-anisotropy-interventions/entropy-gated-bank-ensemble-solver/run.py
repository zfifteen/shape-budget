"""
Entropy-gated bank ensemble solver for the focused anisotropic bottleneck.

The solver merges the cached candidate families from the bank-adaptive baseline
and density-ablation variants. It fits a four-way multinomial logistic chooser
on calibration blocks only, then applies an observability gate:

- default candidate: density-ablation support
- gate metric: density-ablation joint entropy
- when the gate opens, trust the frozen four-way chooser

The gate threshold is chosen from calibration only. Holdout and confirmation are
pure evaluation blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

generate_block_cache, load_csv_rows, cache_path, FOCUS_CONDITIONS, GEOMETRY_SKEW_BIN_LABELS, FOCUS_ALPHA_BIN = load_symbols(
    "run_bank_adaptive_solver_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/bank-adaptive-solver/run.py",
    "generate_block_cache",
    "load_csv_rows",
    "cache_path",
    "FOCUS_CONDITIONS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "FOCUS_ALPHA_BIN",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_NAMES = (
    "calibration_block_1",
    "calibration_block_2",
    "holdout_block_1",
    "confirmation_block",
)
CALIBRATION_BLOCKS = ("calibration_block_1", "calibration_block_2")
EVAL_BLOCKS = ("holdout_block_1", "confirmation_block")
VARIANTS = ("baseline", "density_ablation")
DEFAULT_CANDIDATE = "dense_support"
GATE_FEATURE = "d_joint_entropy"

CANDIDATE_NAMES = (
    "baseline_support",
    "baseline_joint",
    "dense_support",
    "dense_joint",
)
CANDIDATE_TO_INDEX = {name: idx for idx, name in enumerate(CANDIDATE_NAMES)}

STRING_FIELDS = {"condition", "geometry_skew_bin", "block_name"}


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


def ensure_caches(force: bool = False) -> None:
    for variant_name in VARIANTS:
        for block_name in BLOCK_NAMES:
            generate_block_cache(block_name, variant_name, force=force)


def block_rows(block_name: str, variant_name: str) -> list[dict[str, float | str]]:
    return load_csv_rows(cache_path(block_name, variant_name))


def load_merged_rows(block_names: tuple[str, ...]) -> list[dict[str, object]]:
    merged_rows: list[dict[str, object]] = []
    for block_name in block_names:
        baseline_rows = block_rows(block_name, "baseline")
        density_rows = block_rows(block_name, "density_ablation")

        baseline_by_key = {
            (
                int(float(row["observation_seed"])),
                str(row["condition"]),
                str(row["geometry_skew_bin"]),
            ): row
            for row in baseline_rows
        }
        density_by_key = {
            (
                int(float(row["observation_seed"])),
                str(row["condition"]),
                str(row["geometry_skew_bin"]),
            ): row
            for row in density_rows
        }

        shared_keys = sorted(set(baseline_by_key) & set(density_by_key))
        for observation_seed, condition, skew_bin in shared_keys:
            merged_rows.append(
                {
                    "block_name": block_name,
                    "observation_seed": observation_seed,
                    "condition": condition,
                    "geometry_skew_bin": skew_bin,
                    "baseline": baseline_by_key[(observation_seed, condition, skew_bin)],
                    "density": density_by_key[(observation_seed, condition, skew_bin)],
                }
            )
    return merged_rows


def candidate_alpha_errors(row: dict[str, object]) -> dict[str, float]:
    baseline = row["baseline"]
    density = row["density"]
    return {
        "baseline_support": float(baseline["support_alpha_error"]),
        "baseline_joint": float(baseline["joint_alpha_error"]),
        "dense_support": float(density["support_alpha_error"]),
        "dense_joint": float(density["joint_alpha_error"]),
    }


def numeric_value(container: dict[str, float | str], key: str) -> float:
    return float(container[key])


def build_feature_dict(row: dict[str, object]) -> dict[str, float | str]:
    baseline = row["baseline"]
    density = row["density"]
    features: dict[str, float | str] = {
        "condition": str(row["condition"]),
        "geometry_skew_bin": str(row["geometry_skew_bin"]),
        "obs_count": numeric_value(density, "observed_count"),
    }

    for tag, source in (("b", baseline), ("d", density)):
        for prefix in ("support", "joint"):
            out = f"{tag}_{prefix}"
            features[f"{out}_alpha"] = numeric_value(source, f"{prefix}_alpha")
            features[f"{out}_t"] = numeric_value(source, f"{prefix}_t")
            features[f"{out}_rho"] = numeric_value(source, f"{prefix}_rho")
            features[f"{out}_h"] = numeric_value(source, f"{prefix}_h")
            features[f"{out}_w1"] = numeric_value(source, f"{prefix}_w1")
            features[f"{out}_w2"] = numeric_value(source, f"{prefix}_w2")
            features[f"{out}_score"] = numeric_value(source, f"{prefix}_score")
            features[f"{out}_entropy"] = numeric_value(source, f"{prefix}_entropy")
            features[f"{out}_cv"] = numeric_value(source, f"{prefix}_cv_score")
            features[f"{out}_fit"] = numeric_value(source, f"{prefix}_fit_rmse")

        support_alpha = numeric_value(source, "support_alpha")
        joint_alpha = numeric_value(source, "joint_alpha")
        support_t = numeric_value(source, "support_t")
        joint_t = numeric_value(source, "joint_t")
        support_score = numeric_value(source, "support_score")
        joint_score = numeric_value(source, "joint_score")
        support_cv = numeric_value(source, "support_cv_score")
        joint_cv = numeric_value(source, "joint_cv_score")
        support_fit = numeric_value(source, "support_fit_rmse")
        joint_fit = numeric_value(source, "joint_fit_rmse")

        features[f"{tag}_alpha_gap"] = abs(joint_alpha - support_alpha)
        features[f"{tag}_t_gap"] = abs(joint_t - support_t)
        features[f"{tag}_score_ratio"] = joint_score / max(support_score, 1.0e-12)
        features[f"{tag}_cv_ratio"] = joint_cv / max(support_cv, 1.0e-12)
        features[f"{tag}_fit_ratio"] = joint_fit / max(support_fit, 1.0e-12)

    features["cross_support_alpha_gap"] = abs(
        numeric_value(baseline, "support_alpha") - numeric_value(density, "support_alpha")
    )
    features["cross_joint_alpha_gap"] = abs(
        numeric_value(baseline, "joint_alpha") - numeric_value(density, "joint_alpha")
    )
    features["cross_support_score_ratio"] = numeric_value(density, "support_score") / max(
        numeric_value(baseline, "support_score"),
        1.0e-12,
    )
    features["cross_joint_score_ratio"] = numeric_value(density, "joint_score") / max(
        numeric_value(baseline, "joint_score"),
        1.0e-12,
    )
    return features


def dataframe_and_targets(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, np.ndarray]:
    feature_rows: list[dict[str, float | str]] = []
    labels: list[int] = []
    for row in rows:
        feature_rows.append(build_feature_dict(row))
        best_candidate = min(candidate_alpha_errors(row), key=candidate_alpha_errors(row).get)
        labels.append(CANDIDATE_TO_INDEX[best_candidate])
    return pd.DataFrame(feature_rows), np.array(labels, dtype=int)


def fit_four_way_chooser(calibration_rows: list[dict[str, object]]) -> tuple[Pipeline, list[str]]:
    frame, labels = dataframe_and_targets(calibration_rows)
    numeric_features = [column for column in frame.columns if column not in STRING_FIELDS]
    categorical_features = ["condition", "geometry_skew_bin"]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    chooser = Pipeline(
        [
            ("pre", preprocessor),
            ("model", LogisticRegression(max_iter=3000)),
        ]
    )
    chooser.fit(frame, labels)
    return chooser, numeric_features


def choose_names_from_model(chooser: Pipeline, rows: list[dict[str, object]]) -> tuple[list[str], pd.DataFrame]:
    frame, _ = dataframe_and_targets(rows)
    predictions = chooser.predict(frame)
    prediction_names = [CANDIDATE_NAMES[int(idx)] for idx in predictions]
    return prediction_names, frame


def mean_alpha_error(rows: list[dict[str, object]], names: list[str]) -> float:
    values = []
    for row, name in zip(rows, names):
        values.append(candidate_alpha_errors(row)[name])
    return float(np.mean(values))


def block_score(rows: list[dict[str, object]], names: list[str], block_name: str) -> float:
    values = [
        candidate_alpha_errors(row)[name]
        for row, name in zip(rows, names)
        if row["block_name"] == block_name
    ]
    return float(np.mean(values))


def select_gate_threshold(
    calibration_rows: list[dict[str, object]],
    chooser_prediction_names: list[str],
    calibration_frame: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    gate_values = np.array(calibration_frame[GATE_FEATURE], dtype=float)
    threshold_candidates = np.unique(np.quantile(gate_values, np.linspace(0.05, 0.95, 19)))
    median_gate_value = float(np.median(gate_values))

    best_threshold = None
    best_score = None
    best_worst_block = None
    best_block_scores = None

    for threshold in threshold_candidates:
        chosen_names = [
            predicted if gate_value >= threshold else DEFAULT_CANDIDATE
            for predicted, gate_value in zip(chooser_prediction_names, gate_values)
        ]
        score = mean_alpha_error(calibration_rows, chosen_names)
        per_block = {
            block_name: block_score(calibration_rows, chosen_names, block_name)
            for block_name in CALIBRATION_BLOCKS
        }
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
        raise RuntimeError("Failed to select a gate threshold from calibration rows.")

    return best_threshold, {key: float(value) for key, value in best_block_scores.items()}


def apply_solver(
    rows: list[dict[str, object]],
    chooser: Pipeline,
    gate_threshold: float,
) -> list[dict[str, object]]:
    chooser_prediction_names, frame = choose_names_from_model(chooser, rows)
    probabilities = chooser.predict_proba(frame)

    trial_rows: list[dict[str, object]] = []
    for row, chooser_name, probability_vector, (_, feature_row) in zip(
        rows,
        chooser_prediction_names,
        probabilities,
        frame.iterrows(),
    ):
        gate_value = float(feature_row[GATE_FEATURE])
        gate_open = gate_value >= gate_threshold
        chosen_name = chooser_name if gate_open else DEFAULT_CANDIDATE
        candidate_errors = candidate_alpha_errors(row)
        best_single_name = min(candidate_errors, key=candidate_errors.get)

        baseline = row["baseline"]
        density = row["density"]
        trial_rows.append(
            {
                "block_name": str(row["block_name"]),
                "observation_seed": int(row["observation_seed"]),
                "condition": str(row["condition"]),
                "geometry_skew_bin": str(row["geometry_skew_bin"]),
                "gate_feature": GATE_FEATURE,
                "gate_value": gate_value,
                "gate_threshold": float(gate_threshold),
                "gate_open": int(gate_open),
                "chooser_prediction": chooser_name,
                "chooser_confidence": float(np.max(probability_vector)),
                "chosen_candidate": chosen_name,
                "chosen_alpha_error": float(candidate_errors[chosen_name]),
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


def summarize_block(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    chosen_mean = float(np.mean([float(row["chosen_alpha_error"]) for row in rows]))
    oracle4_mean = float(np.mean([float(row["oracle4_alpha_error"]) for row in rows]))
    single_means = {
        "baseline_support": float(np.mean([float(row["baseline_support_alpha_error"]) for row in rows])),
        "baseline_joint": float(np.mean([float(row["baseline_joint_alpha_error"]) for row in rows])),
        "dense_support": float(np.mean([float(row["dense_support_alpha_error"]) for row in rows])),
        "dense_joint": float(np.mean([float(row["dense_joint_alpha_error"]) for row in rows])),
    }
    best_single_mean = min(single_means.values())
    best_single_name = min(single_means, key=single_means.get)
    beats_best_single = chosen_mean + 1.0e-12 < best_single_mean

    by_condition: list[dict[str, object]] = []
    by_cell: list[dict[str, object]] = []
    for condition in FOCUS_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            continue
        by_condition.append(
            {
                "condition": condition,
                "count": len(condition_rows),
                "solver_alpha_error_mean": float(
                    np.mean([float(row["chosen_alpha_error"]) for row in condition_rows])
                ),
                "best_single_alpha_error_mean": float(
                    np.mean([float(row["best_single_alpha_error"]) for row in condition_rows])
                ),
                "gate_open_rate": float(np.mean([float(row["gate_open"]) for row in condition_rows])),
            }
        )
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            cell_rows = [row for row in condition_rows if row["geometry_skew_bin"] == skew_bin]
            if not cell_rows:
                continue
            chosen_counter = defaultdict(int)
            for row in cell_rows:
                chosen_counter[str(row["chosen_candidate"])] += 1
            by_cell.append(
                {
                    "condition": condition,
                    "alpha_strength_bin": FOCUS_ALPHA_BIN,
                    "geometry_skew_bin": skew_bin,
                    "count": len(cell_rows),
                    "solver_alpha_error_mean": float(
                        np.mean([float(row["chosen_alpha_error"]) for row in cell_rows])
                    ),
                    "best_single_alpha_error_mean": float(
                        np.mean([float(row["best_single_alpha_error"]) for row in cell_rows])
                    ),
                    "gate_open_rate": float(np.mean([float(row["gate_open"]) for row in cell_rows])),
                    "chosen_candidate_counts": dict(sorted(chosen_counter.items())),
                }
            )

    if beats_best_single:
        plain_language_result = (
            f"The entropy-gated ensemble beat the best single cached candidate on {label}: "
            f"solver {chosen_mean:.4f} vs best single {best_single_name} at {best_single_mean:.4f}."
        )
        bgp_interpretation = "strengthens BGP"
    else:
        plain_language_result = (
            f"The entropy-gated ensemble did not beat the best single cached candidate on {label}: "
            f"solver {chosen_mean:.4f} vs best single {best_single_name} at {best_single_mean:.4f}."
        )
        bgp_interpretation = "narrows the remaining solver bottleneck"

    return {
        "label": label,
        "count": len(rows),
        "solver_alpha_error_mean": chosen_mean,
        "best_single_alpha_error_mean": best_single_mean,
        "best_single_candidate": best_single_name,
        "oracle4_alpha_error_mean": oracle4_mean,
        "single_candidate_means": single_means,
        "gate_open_rate": float(np.mean([float(row["gate_open"]) for row in rows])),
        "beats_best_single": beats_best_single,
        "by_condition": by_condition,
        "by_cell": by_cell,
        "plain_language_result": plain_language_result,
        "bgp_interpretation": bgp_interpretation,
    }


def overall_summary(holdout_summary: dict[str, object], confirmation_summary: dict[str, object]) -> dict[str, object]:
    working_solver = bool(
        holdout_summary["beats_best_single"] and confirmation_summary["beats_best_single"]
    )
    if working_solver:
        plain_language_result = (
            "The entropy-gated ensemble solved the focused slice in the tested regime: "
            f"holdout {holdout_summary['solver_alpha_error_mean']:.4f} vs "
            f"{holdout_summary['best_single_alpha_error_mean']:.4f}, confirmation "
            f"{confirmation_summary['solver_alpha_error_mean']:.4f} vs "
            f"{confirmation_summary['best_single_alpha_error_mean']:.4f}."
        )
        bgp_interpretation = "strengthens BGP"
    else:
        plain_language_result = (
            "The entropy-gated ensemble improved the focused slice but did not clear both evaluation blocks."
        )
        bgp_interpretation = "narrows the remaining solver bottleneck"

    return {
        "working_solver": working_solver,
        "plain_language_result": plain_language_result,
        "bgp_interpretation": bgp_interpretation,
    }


def run(force_cache: bool = False) -> dict[str, object]:
    ensure_caches(force=force_cache)

    calibration_rows = load_merged_rows(CALIBRATION_BLOCKS)
    holdout_rows = load_merged_rows(("holdout_block_1",))
    confirmation_rows = load_merged_rows(("confirmation_block",))

    chooser, numeric_features = fit_four_way_chooser(calibration_rows)
    chooser_prediction_names, calibration_frame = choose_names_from_model(chooser, calibration_rows)
    gate_threshold, calibration_block_scores = select_gate_threshold(
        calibration_rows,
        chooser_prediction_names,
        calibration_frame,
    )

    calibration_trials = apply_solver(calibration_rows, chooser, gate_threshold)
    holdout_trials = apply_solver(holdout_rows, chooser, gate_threshold)
    confirmation_trials = apply_solver(confirmation_rows, chooser, gate_threshold)

    calibration_summary = summarize_block(calibration_trials, "calibration")
    holdout_summary = summarize_block(holdout_trials, "holdout_block_1")
    confirmation_summary = summarize_block(confirmation_trials, "confirmation_block")
    overall = overall_summary(holdout_summary, confirmation_summary)

    model = chooser.named_steps["model"]
    feature_names = chooser.named_steps["pre"].get_feature_names_out().tolist()
    model_payload = {
        "solver_family": "entropy_gated_bank_ensemble",
        "calibration_blocks": list(CALIBRATION_BLOCKS),
        "evaluation_blocks": list(EVAL_BLOCKS),
        "default_candidate": DEFAULT_CANDIDATE,
        "gate_feature": GATE_FEATURE,
        "gate_threshold": gate_threshold,
        "calibration_block_scores": calibration_block_scores,
        "candidate_names": list(CANDIDATE_NAMES),
        "feature_names": feature_names,
        "classes": [CANDIDATE_NAMES[int(idx)] for idx in model.classes_],
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "numeric_feature_count": len(numeric_features),
    }

    trial_rows = calibration_trials + holdout_trials + confirmation_trials
    summary_payload = {
        "experiment": "entropy-gated-bank-ensemble-solver",
        "focus_alpha_bin": FOCUS_ALPHA_BIN,
        "default_candidate": DEFAULT_CANDIDATE,
        "gate_feature": GATE_FEATURE,
        "gate_threshold": gate_threshold,
        "calibration": calibration_summary,
        "holdout": holdout_summary,
        "confirmation": confirmation_summary,
        "overall": overall,
        "plain_language_result": overall["plain_language_result"],
        "bgp_interpretation": overall["bgp_interpretation"],
    }

    write_csv(OUTPUT_DIR / "entropy_gated_bank_ensemble_solver_trials.csv", trial_rows)
    write_json(OUTPUT_DIR / "entropy_gated_bank_ensemble_solver_summary.json", summary_payload)
    write_json(OUTPUT_DIR / "entropy_gated_bank_ensemble_solver_model.json", model_payload)
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropy-gated bank ensemble solver.")
    parser.add_argument("--force-cache", action="store_true", help="Regenerate upstream candidate caches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(force_cache=bool(args.force_cache))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
