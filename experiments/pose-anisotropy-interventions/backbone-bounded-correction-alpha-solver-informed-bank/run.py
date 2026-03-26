"""
Bounded-correction Layer 3 on top of the corrected informed-bank stack.

This experiment freezes the Layer 3 bounded-correction tau on a separate
calibration block, then evaluates that frozen tau on fresh holdout and
confirmation rows using the explicit Layer 2 open rule.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

(
    BANK_SEEDS,
    GEOMETRY_SKEW_BIN_LABELS,
    FOCUS_ALPHA_BIN,
    sample_conditioned_parameters,
    observe_pose_free_signature,
    anisotropic_forward_signature,
    OBSERVATION_REGIMES,
    marginalized_bank_scores,
    softmin_temperature,
    make_trial_rng,
    score_band,
    geometry_span_norm,
    global_geometry_ranges,
    build_informed_method_context,
    capture_candidate_indices,
    anchored_alpha_posterior,
) = load_symbols(
    "run_bounded_correction_calibration",
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
    "build_informed_method_context",
    "capture_candidate_indices",
    "anchored_alpha_posterior",
)

(candidate_conditioned_search,) = load_symbols(
    "run_candidate_conditioned_alignment_for_bounded_correction",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "candidate_conditioned_search",
)

(CALIBRATION_SEEDS, TARGET_CONDITION) = load_symbols(
    "run_specialized_gate_constants_for_bounded_correction",
    ROOT / "experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank-specialized-ratio-sweep/run.py",
    "CALIBRATION_SEEDS",
    "TARGET_CONDITION",
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

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

SOURCE_DIR = (
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-conditional-alpha-solver-informed-bank/outputs"
)
SOURCE_ALL_REFINE = SOURCE_DIR / "backbone_conditional_alpha_solver_informed_bank_all_refine_trials.csv"
SOURCE_TRIALS = SOURCE_DIR / "backbone_conditional_alpha_solver_informed_bank_trials.csv"
SOURCE_SUMMARY = SOURCE_DIR / "backbone_conditional_alpha_solver_informed_bank_summary.json"
GATE_SUMMARY_PATH = (
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank-specialized-ratio-sweep/outputs/"
    "backbone_observability_gate_informed_bank_specialized_ratio_sweep_summary.json"
)

PREFIX = "backbone_bounded_correction_alpha_solver_informed_bank"
TOP_K_REFINEMENT_SEEDS = 3
NUMERIC_EPS = 1.0e-9
CALIBRATION_CACHE_PATH = OUTPUT_DIR / f"{PREFIX}_calibration_trials.csv"


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


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


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


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def rate_or_nan(flags: list[int]) -> float:
    if not flags:
        return float("nan")
    return float(np.mean(flags))


def threshold_compare(value: float, threshold: float, comparator: str) -> int:
    if comparator == "ge":
        return int(value >= threshold)
    if comparator == "gt":
        return int(value > threshold)
    if comparator == "le":
        return int(value <= threshold)
    if comparator == "lt":
        return int(value < threshold)
    raise ValueError(f"Unsupported comparator: {comparator}")


def load_gate_rule() -> dict[str, float | str]:
    payload = json.loads(GATE_SUMMARY_PATH.read_text(encoding="utf-8"))
    rule = dict(payload["selected_for_layer3"])
    rule["metric"] = str(rule["metric"])
    rule["open_threshold"] = float(rule.get("open_threshold", rule.get("threshold")))
    rule["open_comparator"] = str(rule.get("open_comparator", rule.get("direction")))
    return rule


def metric_value(row: dict[str, object], metric_name: str) -> float:
    candidate_count = f(row["mean_candidate_count"])
    set_span = f(row["mean_alpha_log_span_set"])
    geometry_span = f(row["mean_geometry_span_norm_set"])
    ambiguity = f(row["mean_ambiguity_ratio"])
    anchored_std = f(row["mean_anchored_alpha_log_std"])
    anchored_span = f(row["mean_anchored_alpha_log_span"])
    anchored_effective = f(row["mean_anchored_effective_count"])

    values = {
        "metric_std": anchored_std,
        "metric_ambiguity": ambiguity,
        "metric_set_span": set_span,
        "metric_anchored_span": anchored_span,
        "ratio_std_over_set_span": anchored_std / max(set_span, NUMERIC_EPS),
        "ratio_std_over_ambiguity": anchored_std / max(ambiguity, NUMERIC_EPS),
        "ratio_ambiguity_over_std": ambiguity / max(anchored_std, NUMERIC_EPS),
        "ratio_effective_over_count": anchored_effective / max(candidate_count, NUMERIC_EPS),
        "ratio_count_over_effective": candidate_count / max(anchored_effective, NUMERIC_EPS),
        "ratio_std_times_effective_over_count": anchored_std * anchored_effective / max(candidate_count, NUMERIC_EPS),
        "ratio_anchored_span_over_std": anchored_span / max(anchored_std, NUMERIC_EPS),
        "ratio_candidate_times_anchored_span_over_std": candidate_count * anchored_span / max(anchored_std, NUMERIC_EPS),
        "ratio_effective_times_anchored_span_over_std": anchored_effective * anchored_span / max(anchored_std, NUMERIC_EPS),
        "ratio_candidate_times_std_over_effective": candidate_count * anchored_std / max(anchored_effective, NUMERIC_EPS),
        "ratio_candidate_times_ambiguity_over_effective": candidate_count * ambiguity / max(anchored_effective, NUMERIC_EPS),
        "ratio_candidate_times_std_over_ambiguity": candidate_count * anchored_std / max(ambiguity, NUMERIC_EPS),
        "ratio_candidate_times_span_over_effective": candidate_count * anchored_span / max(anchored_effective, NUMERIC_EPS),
        "ratio_geomspan_over_std": geometry_span / max(anchored_std, NUMERIC_EPS),
        "ratio_geomspan_times_count_over_effective": geometry_span * candidate_count / max(anchored_effective, NUMERIC_EPS),
    }
    return float(values[metric_name])


def correction_excursion_ratio(row: dict[str, object]) -> float:
    anchored_log = math.log(f(row["anchored_alpha_output"]))
    refined_log = math.log(f(row["refined_alpha_output"]))
    excursion = abs(refined_log - anchored_log)
    return excursion / max(f(row["mean_anchored_alpha_log_span"]), NUMERIC_EPS)


def bounded_weight(row: dict[str, object], tau: float) -> float:
    ratio = correction_excursion_ratio(row)
    return float(max(0.0, 1.0 - ratio / max(tau, NUMERIC_EPS)))


def bounded_row(row: dict[str, object], gate_rule: dict[str, float | str], tau: float) -> dict[str, object]:
    out: dict[str, object] = dict(row)
    gate_open_flag = threshold_compare(
        metric_value(row, str(gate_rule["metric"])),
        float(gate_rule["open_threshold"]),
        str(gate_rule["open_comparator"]),
    )

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

    out["gate_open_flag"] = int(gate_open_flag)
    out["point_output_flag"] = int(gate_open_flag)
    out["correction_excursion_ratio"] = float(correction_excursion_ratio(row))
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


def candidate_taus(rows: list[dict[str, object]]) -> list[float]:
    ratios = sorted({correction_excursion_ratio(row) for row in rows})
    if not ratios:
        return [0.1]
    candidates = [ratios[0] * 0.5]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(ratios[:-1], ratios[1:]))
    candidates.append(ratios[-1] * 1.2)
    return [float(max(item, 1.0e-6)) for item in candidates]


def choose_tau(calibration_rows: list[dict[str, object]], gate_rule: dict[str, float | str]) -> tuple[float, float, float, int]:
    calibration_open_rows = [
        row
        for row in calibration_rows
        if threshold_compare(
            metric_value(row, str(gate_rule["metric"])),
            float(gate_rule["open_threshold"]),
            str(gate_rule["open_comparator"]),
        )
        == 1
    ]
    scored: list[tuple[float, float, float]] = []
    for tau in candidate_taus(calibration_open_rows):
        bounded_rows = [bounded_row(row, gate_rule, tau) for row in calibration_open_rows]
        error = mean_or_nan([f(row["bounded_alpha_output_abs_error"]) for row in bounded_rows])
        mean_weight = mean_or_nan([f(row["bounded_correction_weight"]) for row in bounded_rows])
        scored.append((error, mean_weight, tau))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best_error, best_weight, best_tau = scored[0]
    return float(best_tau), float(best_error), float(best_weight), int(len(calibration_open_rows))


def calibration_trial_rows() -> list[dict[str, object]]:
    if CALIBRATION_CACHE_PATH.exists():
        return [dict(row) for row in read_csv(CALIBRATION_CACHE_PATH)]

    geometry_ranges = global_geometry_ranges()
    regime = next(item for item in OBSERVATION_REGIMES if item["name"] == TARGET_CONDITION)
    rows: list[dict[str, object]] = []

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
            refined_alpha_logs: list[float] = []
            best_alpha_errors: list[float] = []
            anchored_alpha_errors: list[float] = []
            refined_alpha_errors: list[float] = []
            anchored_alpha_stds: list[float] = []
            anchored_alpha_spans: list[float] = []
            anchored_effective_counts: list[float] = []
            candidate_counts: list[int] = []
            alpha_log_span_sets: list[float] = []
            geometry_span_sets: list[float] = []
            ambiguity_ratios: list[float] = []

            for bank_seed_value in BANK_SEEDS:
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
                band_scores = scores[band_indices]
                band_geometries = context.geometries[band_indices]
                band_alpha_logs = context.alpha_logs[band_indices]

                geometry_weights = np.exp(-(band_scores - float(np.min(band_scores))) / max(band, NUMERIC_EPS))
                geometry_weights = geometry_weights / max(float(np.sum(geometry_weights)), NUMERIC_EPS)
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
                ambiguity_ratio = float(alpha_log_span / max(geometry_span, NUMERIC_EPS))

                seed_order = np.argsort(-anchored_weights)[:TOP_K_REFINEMENT_SEEDS]
                refined_logs_local: list[float] = []
                refined_scores_local: list[float] = []
                refined_base_weights: list[float] = []
                for local_idx in seed_order:
                    seed_params = tuple(float(x) for x in context.params_list[int(band_indices[int(local_idx)])])
                    refined_params, _, _, refined_score = candidate_conditioned_search(
                        observed_signature,
                        mask,
                        seed_params,
                        temperature,
                    )
                    refined_logs_local.append(math.log(float(refined_params[5])))
                    refined_scores_local.append(float(refined_score))
                    refined_base_weights.append(float(max(anchored_weights[int(local_idx)], NUMERIC_EPS)))

                refined_scores_arr = np.array(refined_scores_local, dtype=float)
                refined_base_weights_arr = np.array(refined_base_weights, dtype=float)
                score_offsets = refined_scores_arr - float(np.min(refined_scores_arr))
                refined_weights = normalize_weights(
                    refined_base_weights_arr * np.exp(-score_offsets / max(band, NUMERIC_EPS))
                )
                refined_log = float(np.sum(refined_weights * np.array(refined_logs_local, dtype=float)))
                refined_alpha = float(math.exp(refined_log))

                best_alpha_logs.append(best_alpha_log)
                anchored_alpha_logs.append(anchored_mean_log)
                refined_alpha_logs.append(refined_log)
                best_alpha_errors.append(float(abs(best_alpha - true_alpha)))
                anchored_alpha_errors.append(float(abs(anchored_alpha - true_alpha)))
                refined_alpha_errors.append(float(abs(refined_alpha - true_alpha)))
                anchored_alpha_stds.append(float(anchored_std_log))
                anchored_alpha_spans.append(float(anchored_span_log))
                anchored_effective_counts.append(float(anchored_effective_count))
                candidate_counts.append(int(len(band_indices)))
                alpha_log_span_sets.append(float(alpha_log_span))
                geometry_span_sets.append(float(geometry_span))
                ambiguity_ratios.append(float(ambiguity_ratio))

            best_alpha_output = float(math.exp(float(np.mean(best_alpha_logs))))
            anchored_alpha_output = float(math.exp(float(np.mean(anchored_alpha_logs))))
            refined_alpha_output = float(math.exp(float(np.mean(refined_alpha_logs))))

            rows.append(
                {
                    "split": "calibration",
                    "observation_seed": int(observation_seed),
                    "condition": TARGET_CONDITION,
                    "geometry_skew_bin": skew_bin,
                    "true_alpha": float(true_alpha),
                    "true_t": float(true_params[1]),
                    "true_rotation_shift": int(true_shift),
                    "mean_candidate_count": float(np.mean(candidate_counts)),
                    "mean_alpha_log_span_set": float(np.mean(alpha_log_span_sets)),
                    "mean_geometry_span_norm_set": float(np.mean(geometry_span_sets)),
                    "mean_ambiguity_ratio": float(np.mean(ambiguity_ratios)),
                    "mean_anchored_alpha_log_std": float(np.mean(anchored_alpha_stds)),
                    "mean_anchored_alpha_log_span": float(np.mean(anchored_alpha_spans)),
                    "mean_anchored_effective_count": float(np.mean(anchored_effective_counts)),
                    "best_alpha_output": float(best_alpha_output),
                    "best_alpha_output_abs_error": float(abs(best_alpha_output - true_alpha)),
                    "anchored_alpha_output": float(anchored_alpha_output),
                    "anchored_alpha_output_abs_error": float(abs(anchored_alpha_output - true_alpha)),
                    "refined_alpha_output": float(refined_alpha_output),
                    "refined_alpha_output_abs_error": float(abs(refined_alpha_output - true_alpha)),
                    "best_alpha_bank_log_span": float(np.max(best_alpha_logs) - np.min(best_alpha_logs)),
                    "anchored_alpha_bank_log_span": float(np.max(anchored_alpha_logs) - np.min(anchored_alpha_logs)),
                    "refined_alpha_bank_log_span": float(np.max(refined_alpha_logs) - np.min(refined_alpha_logs)),
                    "refined_alpha_abs_error_mean": float(np.mean(refined_alpha_errors)),
                    "refined_beats_anchored_flag": int(abs(refined_alpha_output - true_alpha) <= abs(anchored_alpha_output - true_alpha)),
                    "refined_beats_best_flag": int(abs(refined_alpha_output - true_alpha) <= abs(best_alpha_output - true_alpha)),
                }
            )

    write_csv(CALIBRATION_CACHE_PATH, rows)
    return rows


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
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
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
                    "condition": TARGET_CONDITION,
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
    ax.set_title("Bounded correction with calibration-frozen tau")
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
    ax.set_title("Bounded correction follows frozen calibration tau")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    gate_rule = load_gate_rule()
    with SOURCE_SUMMARY.open("r", encoding="utf-8") as handle:
        source_payload = json.load(handle)

    source_trial_rows = {
        (
            str(row["split"]),
            int(float(row["observation_seed"])),
            str(row["condition"]),
            str(row["geometry_skew_bin"]),
        ): row
        for row in read_csv(SOURCE_TRIALS)
    }
    fresh_rows: list[dict[str, object]] = []
    for row in read_csv(SOURCE_ALL_REFINE):
        if str(row["split"]) not in ("holdout", "confirmation"):
            continue
        key = (
            str(row["split"]),
            int(float(row["observation_seed"])),
            str(row["condition"]),
            str(row["geometry_skew_bin"]),
        )
        source_trial = source_trial_rows.get(key)
        if source_trial is None:
            raise RuntimeError(f"Missing integrated Layer 3 trial row for {key}")
        merged = dict(row)
        merged["alpha_point_recoverable_flag"] = source_trial["alpha_point_recoverable_flag"]
        merged["alpha_point_unrecoverable_flag"] = source_trial["alpha_point_unrecoverable_flag"]
        fresh_rows.append(merged)

    calibration_rows = calibration_trial_rows()
    tau, calibration_error, calibration_weight, calibration_open_count = choose_tau(calibration_rows, gate_rule)

    bounded_fresh_rows = [bounded_row(row, gate_rule, tau) for row in fresh_rows]
    bounded_calibration_rows = [bounded_row(row, gate_rule, tau) for row in calibration_rows]
    split_summary = summarize_by_split(bounded_fresh_rows)
    cell_summary = summarize_by_cell(bounded_fresh_rows)
    open_rows = [row for row in bounded_fresh_rows if i(row["gate_open_flag"]) == 1]

    tau_summary = {
        "definition": "w = max(0, 1 - (|log(refined) - log(anchored)| / mean_anchored_alpha_log_span) / tau)",
        "tau": float(tau),
        "selection_scope": "calibration open-trial cached sweep",
        "calibration_count": int(len(calibration_rows)),
        "calibration_open_count": int(calibration_open_count),
        "mean_bounded_alpha_output_abs_error_open": float(calibration_error),
        "mean_bounded_correction_weight_open": float(calibration_weight),
    }

    global_summary = {
        "nominal_final_bank_size": int(source_payload["summary"]["nominal_final_bank_size"]),
        "mean_band_candidate_count": float(source_payload["summary"]["mean_band_candidate_count"]),
        "gate_metric": str(gate_rule["metric"]),
        "gate_open_comparator": str(gate_rule["open_comparator"]),
        "trial_count": len(bounded_fresh_rows),
        "point_output_count": int(sum(i(row["point_output_flag"]) for row in bounded_fresh_rows)),
        "point_output_rate": float(np.mean([i(row["point_output_flag"]) for row in bounded_fresh_rows])),
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

    write_csv(OUTPUT_DIR / f"{PREFIX}_calibration_trials.csv", bounded_calibration_rows)
    write_csv(OUTPUT_DIR / f"{PREFIX}_trials.csv", bounded_fresh_rows)
    write_csv(OUTPUT_DIR / f"{PREFIX}_split_summary.csv", split_summary)
    write_csv(OUTPUT_DIR / f"{PREFIX}_cell_summary.csv", cell_summary)
    (OUTPUT_DIR / f"{PREFIX}_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plot_open_alpha_errors(FIGURE_DIR / f"{PREFIX}_alpha_error.png", split_summary)
    plot_bounded_weights(FIGURE_DIR / f"{PREFIX}_bounded_weight.png", split_summary)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
