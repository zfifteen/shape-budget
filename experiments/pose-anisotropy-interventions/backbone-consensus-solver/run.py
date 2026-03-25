"""
Backbone-first geometry consensus layer for the pose-free anisotropic solver challenge.

This is the first capability layer in the layered solver program.

The goal is narrower than the full solver challenge:

- recover the stable normalized-geometry backbone
- do not force a point estimate for alpha
- test whether a near-family geometry consensus is more bank-stable than
  winner-take-all best-candidate geometry

The experiment stays on the focused moderate-anisotropy slice and reuses the
bank-adaptive calibration, holdout, and confirmation observation blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

GEOMETRY_SKEW_BIN_LABELS, sample_conditioned_parameters = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "GEOMETRY_SKEW_BIN_LABELS",
    "sample_conditioned_parameters",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

ALPHA_MAX, ALPHA_MIN, anisotropic_forward_signature, build_reference_bank, control_invariants = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "control_invariants",
)

OBSERVATION_REGIMES, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "write_csv",
)

FOCUS_ALPHA_BIN, FOCUS_CONDITIONS, marginalized_bank_scores, softmin_temperature = load_symbols(
    "run_joint_pose_marginalized_solver_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py",
    "FOCUS_ALPHA_BIN",
    "FOCUS_CONDITIONS",
    "marginalized_bank_scores",
    "softmin_temperature",
)

import json
import math
import os
from dataclasses import asdict, dataclass

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

BLOCK_SPECS = {
    "calibration": (20260410, 20260411, 20260412, 20260416, 20260417, 20260418),
    "holdout": (20260422, 20260423, 20260424),
    "confirmation": (20260425, 20260426, 20260427),
}

BANK_SEEDS = (20260324, 20260325, 20260326, 20260327, 20260328)
REFERENCE_BANK_SIZE = 300
MIN_SCORE_BAND = 5.0e-5
NUMERIC_EPS = 1.0e-9
ALPHA_LOG_RANGE = float(math.log(ALPHA_MAX) - math.log(ALPHA_MIN))


@dataclass
class BankRow:
    split: str
    observation_seed: int
    bank_seed: int
    condition: str
    geometry_skew_bin: str
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    candidate_count: int
    alpha_log_span_set: float
    geometry_span_norm_set: float
    ambiguity_ratio: float
    best_geometry_mae: float
    consensus_geometry_mae: float
    best_alpha: float
    consensus_rho12: float
    consensus_rho13: float
    consensus_rho23: float


@dataclass
class TrialRow:
    split: str
    observation_seed: int
    condition: str
    geometry_skew_bin: str
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    mean_candidate_count: float
    mean_alpha_log_span_set: float
    mean_geometry_span_norm_set: float
    mean_ambiguity_ratio: float
    alpha_bank_log_span: float
    best_geometry_bank_span_norm: float
    consensus_geometry_bank_span_norm: float
    best_geometry_mae_mean: float
    consensus_geometry_mae_mean: float
    geometry_mae_gain: float
    geometry_bank_span_gain: float
    consensus_beats_best_flag: int
    consensus_tighter_flag: int


@dataclass(frozen=True)
class BankContext:
    seed: int
    params_list: list[tuple[float, float, float, float, float, float]]
    shifted_bank: np.ndarray
    geometries: np.ndarray
    alpha_logs: np.ndarray


def condition_index(condition: str) -> int:
    return FOCUS_CONDITIONS.index(condition)


def skew_index(skew_bin: str) -> int:
    return GEOMETRY_SKEW_BIN_LABELS.index(skew_bin)


def make_trial_rng(observation_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(observation_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


def canonicalize_geometry(params: tuple[float, float, float, float, float, float]) -> tuple[np.ndarray, float]:
    geometry, _, alpha = control_invariants(params)
    swapped = np.array([geometry[0], geometry[2], geometry[1]])
    if tuple(swapped) < tuple(geometry):
        return swapped, float(alpha)
    return geometry, float(alpha)


def build_bank_context(bank_seed: int) -> BankContext:
    bank_rng = np.random.default_rng(bank_seed)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, bank_rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    geometries = []
    alpha_logs = []
    for params in params_list:
        geometry, alpha = canonicalize_geometry(params)
        geometries.append(geometry)
        alpha_logs.append(math.log(alpha))
    return BankContext(
        seed=bank_seed,
        params_list=params_list,
        shifted_bank=shifted_bank,
        geometries=np.array(geometries),
        alpha_logs=np.array(alpha_logs),
    )


def empirical_geometry_ranges(bank_contexts: list[BankContext]) -> np.ndarray:
    all_geometries = np.concatenate([context.geometries for context in bank_contexts], axis=0)
    span = np.max(all_geometries, axis=0) - np.min(all_geometries, axis=0)
    return np.maximum(span, NUMERIC_EPS)


def score_band(regime: dict[str, float | str | int]) -> float:
    return max(float(regime["noise_sigma"]) ** 2, MIN_SCORE_BAND)


def select_candidate_indices(scores: np.ndarray, band: float) -> np.ndarray:
    order = np.argsort(scores)
    best_score = float(scores[order[0]])
    keep = order[scores[order] <= best_score + band]
    if len(keep) == 0:
        return order[:1]
    return keep


def geometry_mae(true_geometry: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.abs(true_geometry - estimate)))


def geometry_span_norm(geometries: np.ndarray, geometry_ranges: np.ndarray) -> float:
    span = (np.max(geometries, axis=0) - np.min(geometries, axis=0)) / geometry_ranges
    return float(np.mean(span))


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def consensus_weights(scores: np.ndarray, scale: float) -> np.ndarray:
    offsets = scores - float(np.min(scores))
    logits = np.exp(-offsets / max(scale, NUMERIC_EPS))
    total = float(np.sum(logits))
    if total <= 0.0:
        return np.full(len(scores), 1.0 / len(scores))
    return logits / total


def evaluate_trial(
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    regime: dict[str, float | str | int],
    bank_contexts: list[BankContext],
    geometry_ranges: np.ndarray,
) -> tuple[list[BankRow], TrialRow]:
    trial_rng = make_trial_rng(observation_seed, condition, skew_bin)
    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
    true_geometry, _ = canonicalize_geometry(true_params)
    clean_signature = anisotropic_forward_signature(true_params)
    _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
    temperature = softmin_temperature(regime)
    band = score_band(regime)

    bank_rows: list[BankRow] = []
    candidate_counts = []
    alpha_log_spans = []
    geometry_set_spans = []
    ambiguity_ratios = []
    best_geometries = []
    consensus_geometries = []
    best_alpha_logs = []
    best_geometry_errors = []
    consensus_geometry_errors = []

    for context in bank_contexts:
        scores, _ = marginalized_bank_scores(observed_signature, mask, context.shifted_bank, temperature)
        best_idx = int(np.argmin(scores))
        candidate_indices = select_candidate_indices(scores, band)
        candidate_scores = scores[candidate_indices]
        candidate_geometries = context.geometries[candidate_indices]
        candidate_alpha_logs = context.alpha_logs[candidate_indices]

        local_weights = consensus_weights(candidate_scores, band)
        consensus_geometry = np.sum(candidate_geometries * local_weights[:, None], axis=0)
        best_geometry = context.geometries[best_idx]

        alpha_log_span = float(np.max(candidate_alpha_logs) - np.min(candidate_alpha_logs))
        geometry_span = geometry_span_norm(candidate_geometries, geometry_ranges)
        ambiguity_ratio = float(alpha_log_span / max(geometry_span, NUMERIC_EPS))

        best_geometry_error = geometry_mae(true_geometry, best_geometry)
        consensus_geometry_error = geometry_mae(true_geometry, consensus_geometry)

        candidate_counts.append(int(len(candidate_indices)))
        alpha_log_spans.append(alpha_log_span)
        geometry_set_spans.append(geometry_span)
        ambiguity_ratios.append(ambiguity_ratio)
        best_geometries.append(best_geometry)
        consensus_geometries.append(consensus_geometry)
        best_alpha_logs.append(float(context.alpha_logs[best_idx]))
        best_geometry_errors.append(best_geometry_error)
        consensus_geometry_errors.append(consensus_geometry_error)

        bank_rows.append(
            BankRow(
                split=split,
                observation_seed=int(observation_seed),
                bank_seed=int(context.seed),
                condition=condition,
                geometry_skew_bin=skew_bin,
                true_alpha=float(true_params[5]),
                true_t=float(true_params[1]),
                true_rotation_shift=int(true_shift),
                candidate_count=int(len(candidate_indices)),
                alpha_log_span_set=alpha_log_span,
                geometry_span_norm_set=geometry_span,
                ambiguity_ratio=ambiguity_ratio,
                best_geometry_mae=best_geometry_error,
                consensus_geometry_mae=consensus_geometry_error,
                best_alpha=float(math.exp(context.alpha_logs[best_idx])),
                consensus_rho12=float(consensus_geometry[0]),
                consensus_rho13=float(consensus_geometry[1]),
                consensus_rho23=float(consensus_geometry[2]),
            )
        )

    best_geometry_bank_span = geometry_span_norm(np.array(best_geometries), geometry_ranges)
    consensus_geometry_bank_span = geometry_span_norm(np.array(consensus_geometries), geometry_ranges)
    alpha_bank_log_span = float(np.max(best_alpha_logs) - np.min(best_alpha_logs))
    best_geometry_mae_mean = float(np.mean(best_geometry_errors))
    consensus_geometry_mae_mean = float(np.mean(consensus_geometry_errors))

    trial_row = TrialRow(
        split=split,
        observation_seed=int(observation_seed),
        condition=condition,
        geometry_skew_bin=skew_bin,
        true_alpha=float(true_params[5]),
        true_t=float(true_params[1]),
        true_rotation_shift=int(true_shift),
        mean_candidate_count=float(np.mean(candidate_counts)),
        mean_alpha_log_span_set=float(np.mean(alpha_log_spans)),
        mean_geometry_span_norm_set=float(np.mean(geometry_set_spans)),
        mean_ambiguity_ratio=float(np.mean(ambiguity_ratios)),
        alpha_bank_log_span=alpha_bank_log_span,
        best_geometry_bank_span_norm=best_geometry_bank_span,
        consensus_geometry_bank_span_norm=consensus_geometry_bank_span,
        best_geometry_mae_mean=best_geometry_mae_mean,
        consensus_geometry_mae_mean=consensus_geometry_mae_mean,
        geometry_mae_gain=float(best_geometry_mae_mean - consensus_geometry_mae_mean),
        geometry_bank_span_gain=float(best_geometry_bank_span - consensus_geometry_bank_span),
        consensus_beats_best_flag=int(consensus_geometry_mae_mean <= best_geometry_mae_mean),
        consensus_tighter_flag=int(consensus_geometry_bank_span <= best_geometry_bank_span),
    )
    return bank_rows, trial_row


def summarize_by_split(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        subset = [row for row in rows if row.split == split]
        if not subset:
            continue
        summary.append(
            {
                "split": split,
                "count": len(subset),
                "mean_candidate_count": float(np.mean([row.mean_candidate_count for row in subset])),
                "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in subset])),
                "mean_best_geometry_bank_span_norm": float(np.mean([row.best_geometry_bank_span_norm for row in subset])),
                "mean_consensus_geometry_bank_span_norm": float(np.mean([row.consensus_geometry_bank_span_norm for row in subset])),
                "mean_best_geometry_mae": float(np.mean([row.best_geometry_mae_mean for row in subset])),
                "mean_consensus_geometry_mae": float(np.mean([row.consensus_geometry_mae_mean for row in subset])),
                "mean_geometry_mae_gain": float(np.mean([row.geometry_mae_gain for row in subset])),
                "mean_geometry_bank_span_gain": float(np.mean([row.geometry_bank_span_gain for row in subset])),
                "consensus_beats_best_rate": float(np.mean([row.consensus_beats_best_flag for row in subset])),
                "consensus_tighter_rate": float(np.mean([row.consensus_tighter_flag for row in subset])),
                "consensus_span_vs_alpha_span_corr": safe_corr(
                    [row.consensus_geometry_bank_span_norm for row in subset],
                    [row.alpha_bank_log_span for row in subset],
                ),
                "best_span_vs_alpha_span_corr": safe_corr(
                    [row.best_geometry_bank_span_norm for row in subset],
                    [row.alpha_bank_log_span for row in subset],
                ),
            }
        )
    return summary


def summarize_by_condition(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            subset = [row for row in rows if row.split == split and row.condition == condition]
            if not subset:
                continue
            summary.append(
                {
                    "split": split,
                    "condition": condition,
                    "count": len(subset),
                    "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in subset])),
                    "mean_best_geometry_bank_span_norm": float(np.mean([row.best_geometry_bank_span_norm for row in subset])),
                    "mean_consensus_geometry_bank_span_norm": float(np.mean([row.consensus_geometry_bank_span_norm for row in subset])),
                    "mean_best_geometry_mae": float(np.mean([row.best_geometry_mae_mean for row in subset])),
                    "mean_consensus_geometry_mae": float(np.mean([row.consensus_geometry_mae_mean for row in subset])),
                    "consensus_beats_best_rate": float(np.mean([row.consensus_beats_best_flag for row in subset])),
                    "consensus_tighter_rate": float(np.mean([row.consensus_tighter_flag for row in subset])),
                }
            )
    return summary


def summarize_by_cell(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                subset = [
                    row
                    for row in rows
                    if row.split == split
                    and row.condition == condition
                    and row.geometry_skew_bin == skew_bin
                ]
                if not subset:
                    continue
                summary.append(
                    {
                        "split": split,
                        "condition": condition,
                        "alpha_strength_bin": FOCUS_ALPHA_BIN,
                        "geometry_skew_bin": skew_bin,
                        "count": len(subset),
                        "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in subset])),
                        "mean_best_geometry_bank_span_norm": float(np.mean([row.best_geometry_bank_span_norm for row in subset])),
                        "mean_consensus_geometry_bank_span_norm": float(np.mean([row.consensus_geometry_bank_span_norm for row in subset])),
                        "mean_best_geometry_mae": float(np.mean([row.best_geometry_mae_mean for row in subset])),
                        "mean_consensus_geometry_mae": float(np.mean([row.consensus_geometry_mae_mean for row in subset])),
                        "consensus_beats_best_rate": float(np.mean([row.consensus_beats_best_flag for row in subset])),
                        "consensus_tighter_rate": float(np.mean([row.consensus_tighter_flag for row in subset])),
                    }
                )
    return summary


def plot_geometry_error_bars(path: str, split_summary: list[dict[str, float | str]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([float(item["mean_best_geometry_mae"]) for item in split_summary], dtype=float)
    consensus = np.array([float(item["mean_consensus_geometry_mae"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - width / 2.0, best, width=width, color="#9c6644", label="best candidate")
    ax.bar(x + width / 2.0, consensus, width=width, color="#2a9d8f", label="geometry consensus")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean geometry MAE")
    ax.set_title("Backbone geometry error across solver blocks")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_span_bars(path: str, split_summary: list[dict[str, float | str]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([float(item["mean_best_geometry_bank_span_norm"]) for item in split_summary], dtype=float)
    consensus = np.array([float(item["mean_consensus_geometry_bank_span_norm"]) for item in split_summary], dtype=float)
    alpha = np.array([float(item["mean_alpha_bank_log_span"]) / ALPHA_LOG_RANGE for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - width, best, width=width, color="#8d99ae", label="best-geometry bank span")
    ax.bar(x, consensus, width=width, color="#2a9d8f", label="consensus-geometry bank span")
    ax.bar(x + width, alpha, width=width, color="#e76f51", label="best-alpha bank span (normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean normalized span")
    ax.set_title("Consensus geometry span stays small while alpha remains bank-unstable")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}
    bank_contexts = [build_bank_context(bank_seed) for bank_seed in BANK_SEEDS]
    geometry_ranges = empirical_geometry_ranges(bank_contexts)

    bank_rows: list[BankRow] = []
    trial_rows: list[TrialRow] = []

    for split, observation_seeds in BLOCK_SPECS.items():
        for observation_seed in observation_seeds:
            for condition in FOCUS_CONDITIONS:
                regime = regime_map[condition]
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    new_bank_rows, trial_row = evaluate_trial(
                        split=split,
                        observation_seed=int(observation_seed),
                        condition=condition,
                        skew_bin=skew_bin,
                        regime=regime,
                        bank_contexts=bank_contexts,
                        geometry_ranges=geometry_ranges,
                    )
                    bank_rows.extend(new_bank_rows)
                    trial_rows.append(trial_row)

    split_summary = summarize_by_split(trial_rows)
    condition_summary = summarize_by_condition(trial_rows)
    cell_summary = summarize_by_cell(trial_rows)

    global_summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "bank_seeds": list(BANK_SEEDS),
        "score_band_rule": "best_score + max(noise_sigma^2, 5e-5)",
        "trial_count": len(trial_rows),
        "mean_alpha_bank_log_span": float(np.mean([row.alpha_bank_log_span for row in trial_rows])),
        "mean_best_geometry_bank_span_norm": float(np.mean([row.best_geometry_bank_span_norm for row in trial_rows])),
        "mean_consensus_geometry_bank_span_norm": float(np.mean([row.consensus_geometry_bank_span_norm for row in trial_rows])),
        "mean_best_geometry_mae": float(np.mean([row.best_geometry_mae_mean for row in trial_rows])),
        "mean_consensus_geometry_mae": float(np.mean([row.consensus_geometry_mae_mean for row in trial_rows])),
        "mean_geometry_mae_gain": float(np.mean([row.geometry_mae_gain for row in trial_rows])),
        "mean_geometry_bank_span_gain": float(np.mean([row.geometry_bank_span_gain for row in trial_rows])),
        "consensus_beats_best_rate": float(np.mean([row.consensus_beats_best_flag for row in trial_rows])),
        "consensus_tighter_rate": float(np.mean([row.consensus_tighter_flag for row in trial_rows])),
        "consensus_span_vs_alpha_span_corr": safe_corr(
            [row.consensus_geometry_bank_span_norm for row in trial_rows],
            [row.alpha_bank_log_span for row in trial_rows],
        ),
        "best_span_vs_alpha_span_corr": safe_corr(
            [row.best_geometry_bank_span_norm for row in trial_rows],
            [row.alpha_bank_log_span for row in trial_rows],
        ),
    }

    output_payload = {
        "summary": global_summary,
        "by_split": split_summary,
        "by_condition": condition_summary,
        "by_cell": cell_summary,
    }

    write_csv(os.path.join(OUTPUT_DIR, "backbone_consensus_solver_bank_rows.csv"), [asdict(row) for row in bank_rows])
    write_csv(os.path.join(OUTPUT_DIR, "backbone_consensus_solver_trials.csv"), [asdict(row) for row in trial_rows])
    write_csv(os.path.join(OUTPUT_DIR, "backbone_consensus_solver_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, "backbone_consensus_solver_condition_summary.csv"), condition_summary)
    write_csv(os.path.join(OUTPUT_DIR, "backbone_consensus_solver_cell_summary.csv"), cell_summary)
    with open(os.path.join(OUTPUT_DIR, "backbone_consensus_solver_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    plot_geometry_error_bars(
        os.path.join(FIGURE_DIR, "backbone_consensus_solver_geometry_error.png"),
        split_summary,
    )
    plot_geometry_span_bars(
        os.path.join(FIGURE_DIR, "backbone_consensus_solver_geometry_span.png"),
        split_summary,
    )

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
