"""
Layer 3 with the informed Layer 1 bank and the new Layer 2 ratio gate.

This cached implementation reuses the informed-bank candidate atlas instead of
rebuilding banks from scratch. That keeps the experiment faithful to the new
Layer 1 and Layer 2 stack while making the integrated Layer 3 pass fast enough
to iterate on.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

GEOMETRY_SKEW_BIN_LABELS, sample_conditioned_parameters = load_symbols(
    "run_candidate_conditioned_alignment_experiment_for_informed_layer3",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "GEOMETRY_SKEW_BIN_LABELS",
    "sample_conditioned_parameters",
)

candidate_conditioned_search, = load_symbols(
    "run_candidate_conditioned_alignment_experiment_refiner_for_informed_layer3",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "candidate_conditioned_search",
)

observe_pose_free_signature, = load_symbols(
    "run_pose_free_weighted_inverse_experiment_for_informed_layer3",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "observe_pose_free_signature",
)

anisotropic_forward_signature, = load_symbols(
    "run_weighted_anisotropic_inverse_experiment_for_informed_layer3",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "anisotropic_forward_signature",
)

OBSERVATION_REGIMES, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment_for_informed_layer3",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "write_csv",
)

FOCUS_ALPHA_BIN, softmin_temperature = load_symbols(
    "run_joint_pose_marginalized_solver_experiment_for_informed_layer3",
    ROOT / "experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py",
    "FOCUS_ALPHA_BIN",
    "softmin_temperature",
)

BANK_SEEDS, = load_symbols(
    "run_persistent_mode_informed_bank_constants_for_layer3",
    ROOT / "experiments/pose-anisotropy-interventions/persistent-mode-informed-bank/run.py",
    "BANK_SEEDS",
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

ATLAS_ROWS_PATH = (
    ROOT
    / "experiments/pose-anisotropy-diagnostics/persistent-mode-bank-candidate-atlas/outputs/"
    "persistent_mode_bank_candidate_atlas_rows.csv"
)
GATE_SUMMARY_PATH = (
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank-specialized-ratio-sweep/outputs/"
    "backbone_observability_gate_informed_bank_specialized_ratio_sweep_summary.json"
)
LAYER2_TRIALS_PATH = (
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-observability-gate-informed-bank/outputs/"
    "backbone_observability_gate_informed_bank_trials.csv"
)

BLOCK_SPECS = {
    "holdout": (20260422, 20260423, 20260424),
    "confirmation": (20260425, 20260426, 20260427),
}

TARGET_CONDITIONS = ("sparse_partial_high_noise",)
METHOD_NAME = "persistent_mode_informed"
NUMERIC_EPS = 1.0e-9
ALPHA_STABLE_LOG_SPAN_THRESHOLD = 0.20
ALPHA_POINT_ABS_ERROR_THRESHOLD = 0.15
TOP_K_REFINEMENT_SEEDS = 3
GATE_RULE_OVERRIDE = {
    "metric": "ratio_std_over_set_span",
    "threshold": 0.215677985967846,
    "direction": "ge",
    "calibration_balanced_accuracy": 0.5833333333333334,
}


@dataclass
class CandidateRow:
    split: str
    observation_seed: int
    condition: str
    geometry_skew_bin: str
    bank_seed: int
    candidate_index: int
    rank_by_score: int
    marginalized_score: float
    score_gap_from_best: float
    rho: float
    t: float
    h: float
    w1: float
    w2: float
    alpha: float
    log_alpha: float
    rho12: float
    rho13: float
    rho23: float
    consensus_weight_layer1: float
    anchored_weight_layer2: float
    true_alpha: float
    true_rotation_shift: int


@dataclass
class BankState:
    split: str
    observation_seed: int
    condition: str
    geometry_skew_bin: str
    bank_seed: int
    true_alpha: float
    true_rotation_shift: int
    candidate_count: int
    local_consensus_geometry: np.ndarray
    best_alpha_log: float
    best_alpha_abs_error: float
    anchored_mean_log: float
    anchored_std_log: float
    anchored_span_log: float
    anchored_effective_count: float
    anchored_alpha_abs_error: float
    seed_candidates: list[CandidateRow]


@dataclass
class TrialBase:
    split: str
    observation_seed: int
    condition: str
    geometry_skew_bin: str
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    mean_best_entropy: float
    mean_candidate_count: float
    mean_alpha_log_span_set: float
    mean_anchored_alpha_log_std: float
    mean_anchored_alpha_log_span: float
    mean_anchored_effective_count: float
    ratio_std_over_set_span: float
    ratio_candidate_times_anchored_span_over_std: float
    ratio_candidate_times_span_over_effective: float
    best_alpha_bank_log_span: float
    anchored_alpha_bank_log_span: float
    best_alpha_abs_error_mean: float
    anchored_alpha_abs_error_mean: float
    best_alpha_output: float
    best_alpha_output_abs_error: float
    anchored_alpha_output: float
    anchored_alpha_output_abs_error: float
    alpha_point_recoverable_flag: int
    alpha_point_unrecoverable_flag: int
    trial_backbone: np.ndarray


@dataclass
class TrialPrepared:
    split: str
    observation_seed: int
    condition: str
    geometry_skew_bin: str
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    observed_signature: np.ndarray
    mask: np.ndarray
    temperature: float
    trial_backbone: np.ndarray
    bank_states: list[BankState]
    trial_base: TrialBase


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
    best_entropy: float
    best_alpha: float
    best_alpha_abs_error: float
    anchored_alpha: float
    anchored_alpha_abs_error: float
    anchored_alpha_log_std: float
    anchored_alpha_log_span: float
    anchored_effective_count: float
    local_consensus_rho12: float
    local_consensus_rho13: float
    local_consensus_rho23: float
    trial_backbone_rho12: float
    trial_backbone_rho13: float
    trial_backbone_rho23: float
    gate_open_flag: int
    refined_alpha: float
    refined_alpha_abs_error: float
    refined_effective_count: float
    refined_seed_count: int


@dataclass
class TrialRow:
    split: str
    observation_seed: int
    condition: str
    geometry_skew_bin: str
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    mean_best_entropy: float
    mean_candidate_count: float
    mean_alpha_log_span_set: float
    mean_anchored_alpha_log_std: float
    mean_anchored_alpha_log_span: float
    mean_anchored_effective_count: float
    ratio_std_over_set_span: float
    ratio_candidate_times_anchored_span_over_std: float
    ratio_candidate_times_span_over_effective: float
    best_alpha_bank_log_span: float
    anchored_alpha_bank_log_span: float
    best_alpha_abs_error_mean: float
    anchored_alpha_abs_error_mean: float
    best_alpha_output: float
    best_alpha_output_abs_error: float
    anchored_alpha_output: float
    anchored_alpha_output_abs_error: float
    alpha_point_recoverable_flag: int
    alpha_point_unrecoverable_flag: int
    gate_threshold: float
    gate_open_flag: int
    point_output_flag: int
    abstain_flag: int
    gate_correct_flag: int
    gate_open_and_recoverable_flag: int
    gate_closed_and_unrecoverable_flag: int
    trial_backbone_rho12: float
    trial_backbone_rho13: float
    trial_backbone_rho23: float
    refined_alpha_output: float
    refined_alpha_output_abs_error: float
    refined_alpha_bank_log_span: float
    refined_alpha_abs_error_mean: float
    refined_beats_anchored_flag: int
    refined_beats_best_flag: int


def condition_index(condition: str) -> int:
    return TARGET_CONDITIONS.index(condition)


def skew_index(skew_bin: str) -> int:
    return GEOMETRY_SKEW_BIN_LABELS.index(skew_bin)


def make_trial_rng(observation_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(observation_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


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


def effective_count(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights * weights))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def rate_or_nan(flags: list[int]) -> float:
    if not flags:
        return float("nan")
    return float(np.mean(flags))


def balanced_accuracy(rows: list[TrialBase], metric_name: str, threshold: float, direction: str) -> float:
    labels = np.array([row.alpha_point_unrecoverable_flag for row in rows], dtype=int)
    if direction == "ge":
        preds = np.array([int(getattr(row, metric_name) >= threshold) for row in rows], dtype=int)
    else:
        preds = np.array([int(getattr(row, metric_name) <= threshold) for row in rows], dtype=int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = float(np.sum((preds == 1) & (labels == 1)) / positives)
    tnr = float(np.sum((preds == 0) & (labels == 0)) / negatives)
    return 0.5 * (tpr + tnr)


def load_gate_rule() -> dict[str, float | str]:
    if GATE_RULE_OVERRIDE:
        return GATE_RULE_OVERRIDE
    payload = json.loads(GATE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if "best_metric" in payload:
        return payload["best_metric"]
    return payload["proposed_informed_gate_rule"]


def load_candidate_rows() -> dict[tuple[str, int, str, str, int], list[CandidateRow]]:
    grouped: dict[tuple[str, int, str, str, int], list[CandidateRow]] = defaultdict(list)
    with ATLAS_ROWS_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw["method"] != METHOD_NAME:
                continue
            if raw["capture_tier"] != "band":
                continue
            row = CandidateRow(
                split=str(raw["split"]),
                observation_seed=int(raw["observation_seed"]),
                condition=str(raw["condition"]),
                geometry_skew_bin=str(raw["geometry_skew_bin"]),
                bank_seed=int(raw["bank_seed"]),
                candidate_index=int(raw["candidate_index"]),
                rank_by_score=int(raw["rank_by_score"]),
                marginalized_score=float(raw["marginalized_score"]),
                score_gap_from_best=float(raw["score_gap_from_best"]),
                rho=float(raw["rho"]),
                t=float(raw["t"]),
                h=float(raw["h"]),
                w1=float(raw["w1"]),
                w2=float(raw["w2"]),
                alpha=float(raw["alpha"]),
                log_alpha=float(raw["log_alpha"]),
                rho12=float(raw["rho12"]),
                rho13=float(raw["rho13"]),
                rho23=float(raw["rho23"]),
                consensus_weight_layer1=float(raw["consensus_weight_layer1"]),
                anchored_weight_layer2=float(raw["anchored_weight_layer2"]),
                true_alpha=float(raw["true_alpha"]),
                true_rotation_shift=int(raw["true_rotation_shift"]),
            )
            key = (row.split, row.observation_seed, row.condition, row.geometry_skew_bin, row.bank_seed)
            grouped[key].append(row)
    return grouped


def load_layer2_trial_rows() -> dict[tuple[str, int, str, str], dict[str, float | int | str]]:
    keyed: dict[tuple[str, int, str, str], dict[str, float | int | str]] = {}
    with LAYER2_TRIALS_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw["method"] != METHOD_NAME:
                continue
            key = (
                str(raw["split"]),
                int(raw["observation_seed"]),
                str(raw["condition"]),
                str(raw["geometry_skew_bin"]),
            )
            keyed[key] = {
                "mean_best_entropy": float(raw["mean_best_entropy"]),
                "mean_candidate_count": float(raw["mean_candidate_count"]),
                "mean_alpha_log_span_set": float(raw["mean_alpha_log_span_set"]),
                "mean_anchored_alpha_log_std": float(raw["mean_anchored_alpha_log_std"]),
                "mean_anchored_alpha_log_span": float(raw["mean_anchored_alpha_log_span"]),
                "mean_anchored_effective_count": float(raw["mean_anchored_effective_count"]),
                "ratio_std_over_set_span": float(raw["mean_anchored_alpha_log_std"]) / max(float(raw["mean_alpha_log_span_set"]), NUMERIC_EPS),
                "ratio_candidate_times_anchored_span_over_std": float(raw["ratio_candidate_times_anchored_span_over_std"]),
                "ratio_candidate_times_span_over_effective": float(raw["mean_candidate_count"]) * float(raw["mean_anchored_alpha_log_span"]) / max(float(raw["mean_anchored_effective_count"]), NUMERIC_EPS),
                "best_alpha_bank_log_span": float(raw["best_alpha_bank_log_span"]),
                "anchored_alpha_bank_log_span": float(raw["anchored_alpha_bank_log_span"]),
                "best_alpha_abs_error_mean": float(raw["best_alpha_abs_error_mean"]),
                "anchored_alpha_abs_error_mean": float(raw["anchored_alpha_abs_error_mean"]),
                "alpha_point_recoverable_flag": int(raw["alpha_point_recoverable_flag"]),
                "alpha_point_unrecoverable_flag": int(raw["alpha_point_unrecoverable_flag"]),
            }
    return keyed


def build_bank_state(rows: list[CandidateRow]) -> BankState:
    ordered = sorted(rows, key=lambda row: row.rank_by_score)
    consensus_weights = normalize_weights(np.array([row.consensus_weight_layer1 for row in ordered], dtype=float))
    anchored_weights = normalize_weights(np.array([row.anchored_weight_layer2 for row in ordered], dtype=float))
    geometries = np.array([[row.rho12, row.rho13, row.rho23] for row in ordered], dtype=float)
    alpha_logs = np.array([row.log_alpha for row in ordered], dtype=float)

    local_consensus_geometry = np.sum(geometries * consensus_weights[:, None], axis=0)
    anchored_mean_log = float(np.sum(anchored_weights * alpha_logs))
    anchored_var = float(np.sum(anchored_weights * (alpha_logs - anchored_mean_log) ** 2))
    anchored_std_log = float(math.sqrt(max(anchored_var, 0.0)))
    anchored_span_log = float(
        weighted_quantile(alpha_logs, anchored_weights, 0.90)
        - weighted_quantile(alpha_logs, anchored_weights, 0.10)
    )
    anchored_effective = effective_count(anchored_weights)

    best = ordered[0]
    best_alpha_abs_error = float(abs(best.alpha - best.true_alpha))
    anchored_alpha_abs_error = float(abs(math.exp(anchored_mean_log) - best.true_alpha))
    seed_candidates = sorted(
        ordered,
        key=lambda row: (-row.anchored_weight_layer2, row.rank_by_score),
    )[:TOP_K_REFINEMENT_SEEDS]

    return BankState(
        split=best.split,
        observation_seed=best.observation_seed,
        condition=best.condition,
        geometry_skew_bin=best.geometry_skew_bin,
        bank_seed=best.bank_seed,
        true_alpha=best.true_alpha,
        true_rotation_shift=best.true_rotation_shift,
        candidate_count=len(ordered),
        local_consensus_geometry=local_consensus_geometry,
        best_alpha_log=float(best.log_alpha),
        best_alpha_abs_error=best_alpha_abs_error,
        anchored_mean_log=anchored_mean_log,
        anchored_std_log=anchored_std_log,
        anchored_span_log=anchored_span_log,
        anchored_effective_count=anchored_effective,
        anchored_alpha_abs_error=anchored_alpha_abs_error,
        seed_candidates=seed_candidates,
    )


def prepare_trial(
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    regime: dict[str, float | str | int],
    candidate_rows_by_bank: dict[tuple[str, int, str, str, int], list[CandidateRow]],
    layer2_trial_rows: dict[tuple[str, int, str, str], dict[str, float | int | str]],
) -> TrialPrepared:
    trial_rng = make_trial_rng(observation_seed, condition, skew_bin)
    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
    clean_signature = anisotropic_forward_signature(true_params)
    _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
    temperature = softmin_temperature(regime)

    bank_states: list[BankState] = []
    for bank_seed in BANK_SEEDS:
        key = (split, int(observation_seed), condition, skew_bin, int(bank_seed))
        rows = candidate_rows_by_bank.get(key, [])
        if not rows:
            raise RuntimeError(f"Missing atlas rows for {key}")
        bank_states.append(build_bank_state(rows))

    trial_backbone = np.mean(np.array([state.local_consensus_geometry for state in bank_states]), axis=0)
    layer2_key = (split, int(observation_seed), condition, skew_bin)
    layer2_row = layer2_trial_rows.get(layer2_key)
    if layer2_row is None:
        raise RuntimeError(f"Missing Layer 2 trial row for {layer2_key}")
    best_alpha_logs = [state.best_alpha_log for state in bank_states]
    anchored_alpha_logs = [state.anchored_mean_log for state in bank_states]

    best_alpha_output = float(math.exp(float(np.mean(best_alpha_logs))))
    anchored_alpha_output = float(math.exp(float(np.mean(anchored_alpha_logs))))

    trial_base = TrialBase(
        split=split,
        observation_seed=int(observation_seed),
        condition=condition,
        geometry_skew_bin=skew_bin,
        true_alpha=float(true_params[5]),
        true_t=float(true_params[1]),
        true_rotation_shift=int(true_shift),
        mean_best_entropy=float(layer2_row["mean_best_entropy"]),
        mean_candidate_count=float(layer2_row["mean_candidate_count"]),
        mean_alpha_log_span_set=float(layer2_row["mean_alpha_log_span_set"]),
        mean_anchored_alpha_log_std=float(layer2_row["mean_anchored_alpha_log_std"]),
        mean_anchored_alpha_log_span=float(layer2_row["mean_anchored_alpha_log_span"]),
        mean_anchored_effective_count=float(layer2_row["mean_anchored_effective_count"]),
        ratio_std_over_set_span=float(layer2_row["ratio_std_over_set_span"]),
        ratio_candidate_times_anchored_span_over_std=float(layer2_row["ratio_candidate_times_anchored_span_over_std"]),
        ratio_candidate_times_span_over_effective=float(layer2_row["ratio_candidate_times_span_over_effective"]),
        best_alpha_bank_log_span=float(layer2_row["best_alpha_bank_log_span"]),
        anchored_alpha_bank_log_span=float(layer2_row["anchored_alpha_bank_log_span"]),
        best_alpha_abs_error_mean=float(layer2_row["best_alpha_abs_error_mean"]),
        anchored_alpha_abs_error_mean=float(layer2_row["anchored_alpha_abs_error_mean"]),
        best_alpha_output=best_alpha_output,
        best_alpha_output_abs_error=float(abs(best_alpha_output - true_params[5])),
        anchored_alpha_output=anchored_alpha_output,
        anchored_alpha_output_abs_error=float(abs(anchored_alpha_output - true_params[5])),
        alpha_point_recoverable_flag=int(layer2_row["alpha_point_recoverable_flag"]),
        alpha_point_unrecoverable_flag=int(layer2_row["alpha_point_unrecoverable_flag"]),
        trial_backbone=trial_backbone,
    )

    return TrialPrepared(
        split=split,
        observation_seed=int(observation_seed),
        condition=condition,
        geometry_skew_bin=skew_bin,
        true_alpha=float(true_params[5]),
        true_t=float(true_params[1]),
        true_rotation_shift=int(true_shift),
        observed_signature=observed_signature,
        mask=mask,
        temperature=temperature,
        trial_backbone=trial_backbone,
        bank_states=bank_states,
        trial_base=trial_base,
    )


def refine_trial(
    prepared: TrialPrepared,
    gate_metric_name: str,
    gate_threshold: float,
    gate_direction: str,
) -> tuple[list[BankRow], TrialRow]:
    trial_base = prepared.trial_base
    metric_value = float(getattr(trial_base, gate_metric_name))
    if gate_direction == "ge":
        gate_open_flag = int(metric_value < gate_threshold)
    else:
        gate_open_flag = int(metric_value > gate_threshold)

    bank_rows: list[BankRow] = []
    refined_alpha_logs: list[float] = []
    refined_alpha_errors: list[float] = []

    for state in prepared.bank_states:
        refined_alpha = float("nan")
        refined_alpha_error = float("nan")
        refined_effective = float("nan")
        refined_seed_count = 0

        if gate_open_flag:
            refined_logs_local: list[float] = []
            refined_scores_local: list[float] = []
            refined_base_weights: list[float] = []

            for candidate in state.seed_candidates:
                seed_params = (
                    candidate.rho,
                    candidate.t,
                    candidate.h,
                    candidate.w1,
                    candidate.w2,
                    candidate.alpha,
                )
                refined_params, _, _, refined_score = candidate_conditioned_search(
                    prepared.observed_signature,
                    prepared.mask,
                    seed_params,
                    prepared.temperature,
                )
                refined_logs_local.append(math.log(float(refined_params[5])))
                refined_scores_local.append(float(refined_score))
                refined_base_weights.append(float(max(candidate.anchored_weight_layer2, NUMERIC_EPS)))

            refined_scores_arr = np.array(refined_scores_local, dtype=float)
            refined_base_weights_arr = np.array(refined_base_weights, dtype=float)
            score_offsets = refined_scores_arr - float(np.min(refined_scores_arr))
            refined_weights = normalize_weights(refined_base_weights_arr * np.exp(-score_offsets))
            refined_log = float(np.sum(refined_weights * np.array(refined_logs_local, dtype=float)))
            refined_alpha = float(math.exp(refined_log))
            refined_alpha_error = float(abs(refined_alpha - prepared.true_alpha))
            refined_effective = effective_count(refined_weights)
            refined_seed_count = int(len(refined_logs_local))
            refined_alpha_logs.append(refined_log)
            refined_alpha_errors.append(refined_alpha_error)

        bank_rows.append(
            BankRow(
                split=prepared.split,
                observation_seed=int(prepared.observation_seed),
                bank_seed=int(state.bank_seed),
                condition=prepared.condition,
                geometry_skew_bin=prepared.geometry_skew_bin,
                true_alpha=float(prepared.true_alpha),
                true_t=float(prepared.true_t),
                true_rotation_shift=int(prepared.true_rotation_shift),
                candidate_count=int(state.candidate_count),
                best_entropy=float("nan"),
                best_alpha=float(math.exp(state.best_alpha_log)),
                best_alpha_abs_error=float(state.best_alpha_abs_error),
                anchored_alpha=float(math.exp(state.anchored_mean_log)),
                anchored_alpha_abs_error=float(state.anchored_alpha_abs_error),
                anchored_alpha_log_std=float(state.anchored_std_log),
                anchored_alpha_log_span=float(state.anchored_span_log),
                anchored_effective_count=float(state.anchored_effective_count),
                local_consensus_rho12=float(state.local_consensus_geometry[0]),
                local_consensus_rho13=float(state.local_consensus_geometry[1]),
                local_consensus_rho23=float(state.local_consensus_geometry[2]),
                trial_backbone_rho12=float(prepared.trial_backbone[0]),
                trial_backbone_rho13=float(prepared.trial_backbone[1]),
                trial_backbone_rho23=float(prepared.trial_backbone[2]),
                gate_open_flag=int(gate_open_flag),
                refined_alpha=float(refined_alpha),
                refined_alpha_abs_error=float(refined_alpha_error),
                refined_effective_count=float(refined_effective),
                refined_seed_count=int(refined_seed_count),
            )
        )

    if gate_open_flag:
        refined_alpha_output = float(math.exp(float(np.mean(refined_alpha_logs))))
        refined_alpha_output_abs_error = float(abs(refined_alpha_output - prepared.true_alpha))
        refined_alpha_bank_log_span = float(np.max(refined_alpha_logs) - np.min(refined_alpha_logs))
        refined_alpha_abs_error_mean = float(np.mean(refined_alpha_errors))
        refined_beats_anchored_flag = int(refined_alpha_output_abs_error <= trial_base.anchored_alpha_output_abs_error)
        refined_beats_best_flag = int(refined_alpha_output_abs_error <= trial_base.best_alpha_output_abs_error)
    else:
        refined_alpha_output = float("nan")
        refined_alpha_output_abs_error = float("nan")
        refined_alpha_bank_log_span = float("nan")
        refined_alpha_abs_error_mean = float("nan")
        refined_beats_anchored_flag = 0
        refined_beats_best_flag = 0

    trial_row = TrialRow(
        split=trial_base.split,
        observation_seed=int(trial_base.observation_seed),
        condition=trial_base.condition,
        geometry_skew_bin=trial_base.geometry_skew_bin,
        true_alpha=float(trial_base.true_alpha),
        true_t=float(trial_base.true_t),
        true_rotation_shift=int(trial_base.true_rotation_shift),
        mean_best_entropy=float(trial_base.mean_best_entropy),
        mean_candidate_count=float(trial_base.mean_candidate_count),
        mean_alpha_log_span_set=float(trial_base.mean_alpha_log_span_set),
        mean_anchored_alpha_log_std=float(trial_base.mean_anchored_alpha_log_std),
        mean_anchored_alpha_log_span=float(trial_base.mean_anchored_alpha_log_span),
        mean_anchored_effective_count=float(trial_base.mean_anchored_effective_count),
        ratio_std_over_set_span=float(trial_base.ratio_std_over_set_span),
        ratio_candidate_times_anchored_span_over_std=float(trial_base.ratio_candidate_times_anchored_span_over_std),
        ratio_candidate_times_span_over_effective=float(trial_base.ratio_candidate_times_span_over_effective),
        best_alpha_bank_log_span=float(trial_base.best_alpha_bank_log_span),
        anchored_alpha_bank_log_span=float(trial_base.anchored_alpha_bank_log_span),
        best_alpha_abs_error_mean=float(trial_base.best_alpha_abs_error_mean),
        anchored_alpha_abs_error_mean=float(trial_base.anchored_alpha_abs_error_mean),
        best_alpha_output=float(trial_base.best_alpha_output),
        best_alpha_output_abs_error=float(trial_base.best_alpha_output_abs_error),
        anchored_alpha_output=float(trial_base.anchored_alpha_output),
        anchored_alpha_output_abs_error=float(trial_base.anchored_alpha_output_abs_error),
        alpha_point_recoverable_flag=int(trial_base.alpha_point_recoverable_flag),
        alpha_point_unrecoverable_flag=int(trial_base.alpha_point_unrecoverable_flag),
        gate_threshold=float(gate_threshold),
        gate_open_flag=int(gate_open_flag),
        point_output_flag=int(gate_open_flag),
        abstain_flag=int(1 - gate_open_flag),
        gate_correct_flag=int(
            (gate_open_flag == 1 and trial_base.alpha_point_recoverable_flag == 1)
            or (gate_open_flag == 0 and trial_base.alpha_point_unrecoverable_flag == 1)
        ),
        gate_open_and_recoverable_flag=int(gate_open_flag == 1 and trial_base.alpha_point_recoverable_flag == 1),
        gate_closed_and_unrecoverable_flag=int(gate_open_flag == 0 and trial_base.alpha_point_unrecoverable_flag == 1),
        trial_backbone_rho12=float(trial_base.trial_backbone[0]),
        trial_backbone_rho13=float(trial_base.trial_backbone[1]),
        trial_backbone_rho23=float(trial_base.trial_backbone[2]),
        refined_alpha_output=float(refined_alpha_output),
        refined_alpha_output_abs_error=float(refined_alpha_output_abs_error),
        refined_alpha_bank_log_span=float(refined_alpha_bank_log_span),
        refined_alpha_abs_error_mean=float(refined_alpha_abs_error_mean),
        refined_beats_anchored_flag=int(refined_beats_anchored_flag),
        refined_beats_best_flag=int(refined_beats_best_flag),
    )
    return bank_rows, trial_row


def summarize_by_split(rows: list[TrialRow], gate_metric_name: str, gate_direction: str) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        subset = [row for row in rows if row.split == split]
        if not subset:
            continue
        open_subset = [row for row in subset if row.gate_open_flag == 1]
        unrecoverable_subset = [row for row in subset if row.alpha_point_unrecoverable_flag == 1]
        base_subset = [
            TrialBase(
                split=row.split,
                observation_seed=row.observation_seed,
                condition=row.condition,
                geometry_skew_bin=row.geometry_skew_bin,
                true_alpha=row.true_alpha,
                true_t=row.true_t,
                true_rotation_shift=row.true_rotation_shift,
                mean_best_entropy=row.mean_best_entropy,
                mean_candidate_count=row.mean_candidate_count,
                mean_alpha_log_span_set=row.mean_alpha_log_span_set,
                mean_anchored_alpha_log_std=row.mean_anchored_alpha_log_std,
                mean_anchored_alpha_log_span=row.mean_anchored_alpha_log_span,
                mean_anchored_effective_count=row.mean_anchored_effective_count,
                ratio_std_over_set_span=row.ratio_std_over_set_span,
                ratio_candidate_times_anchored_span_over_std=row.ratio_candidate_times_anchored_span_over_std,
                ratio_candidate_times_span_over_effective=row.ratio_candidate_times_span_over_effective,
                best_alpha_bank_log_span=row.best_alpha_bank_log_span,
                anchored_alpha_bank_log_span=row.anchored_alpha_bank_log_span,
                best_alpha_abs_error_mean=row.best_alpha_abs_error_mean,
                anchored_alpha_abs_error_mean=row.anchored_alpha_abs_error_mean,
                best_alpha_output=row.best_alpha_output,
                best_alpha_output_abs_error=row.best_alpha_output_abs_error,
                anchored_alpha_output=row.anchored_alpha_output,
                anchored_alpha_output_abs_error=row.anchored_alpha_output_abs_error,
                alpha_point_recoverable_flag=row.alpha_point_recoverable_flag,
                alpha_point_unrecoverable_flag=row.alpha_point_unrecoverable_flag,
                trial_backbone=np.array(
                    [row.trial_backbone_rho12, row.trial_backbone_rho13, row.trial_backbone_rho23],
                    dtype=float,
                ),
            )
            for row in subset
        ]
        summary.append(
            {
                "split": split,
                "count": len(subset),
                "point_output_count": int(sum(row.point_output_flag for row in subset)),
                "point_output_rate": float(np.mean([row.point_output_flag for row in subset])),
                "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in subset])),
                "gate_balanced_accuracy": balanced_accuracy(
                    base_subset,
                    gate_metric_name,
                    subset[0].gate_threshold,
                    gate_direction,
                ),
                "gate_precision": rate_or_nan([row.alpha_point_recoverable_flag for row in open_subset]),
                "gate_reject_unrecoverable_rate": rate_or_nan([row.gate_closed_and_unrecoverable_flag for row in unrecoverable_subset]),
                "mean_best_alpha_output_abs_error_open": mean_or_nan([row.best_alpha_output_abs_error for row in open_subset]),
                "mean_anchored_alpha_output_abs_error_open": mean_or_nan([row.anchored_alpha_output_abs_error for row in open_subset]),
                "mean_refined_alpha_output_abs_error_open": mean_or_nan([row.refined_alpha_output_abs_error for row in open_subset]),
                "mean_best_alpha_bank_log_span_open": mean_or_nan([row.best_alpha_bank_log_span for row in open_subset]),
                "mean_anchored_alpha_bank_log_span_open": mean_or_nan([row.anchored_alpha_bank_log_span for row in open_subset]),
                "mean_refined_alpha_bank_log_span_open": mean_or_nan([row.refined_alpha_bank_log_span for row in open_subset]),
                "refined_beats_anchored_rate_open": rate_or_nan([row.refined_beats_anchored_flag for row in open_subset]),
                "refined_beats_best_rate_open": rate_or_nan([row.refined_beats_best_flag for row in open_subset]),
            }
        )
    return summary


def summarize_by_condition(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        for condition in TARGET_CONDITIONS:
            subset = [row for row in rows if row.split == split and row.condition == condition]
            if not subset:
                continue
            open_subset = [row for row in subset if row.gate_open_flag == 1]
            summary.append(
                {
                    "split": split,
                    "condition": condition,
                    "count": len(subset),
                    "point_output_rate": float(np.mean([row.point_output_flag for row in subset])),
                    "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in subset])),
                    "gate_precision": rate_or_nan([row.alpha_point_recoverable_flag for row in open_subset]),
                    "mean_refined_alpha_output_abs_error_open": mean_or_nan([row.refined_alpha_output_abs_error for row in open_subset]),
                    "mean_refined_alpha_bank_log_span_open": mean_or_nan([row.refined_alpha_bank_log_span for row in open_subset]),
                    "refined_beats_anchored_rate_open": rate_or_nan([row.refined_beats_anchored_flag for row in open_subset]),
                    "refined_beats_best_rate_open": rate_or_nan([row.refined_beats_best_flag for row in open_subset]),
                }
            )
    return summary


def summarize_by_cell(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        for condition in TARGET_CONDITIONS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                subset = [
                    row
                    for row in rows
                    if row.split == split and row.condition == condition and row.geometry_skew_bin == skew_bin
                ]
                if not subset:
                    continue
                open_subset = [row for row in subset if row.gate_open_flag == 1]
                summary.append(
                    {
                        "split": split,
                        "condition": condition,
                        "alpha_strength_bin": FOCUS_ALPHA_BIN,
                        "geometry_skew_bin": skew_bin,
                        "count": len(subset),
                        "point_output_rate": float(np.mean([row.point_output_flag for row in subset])),
                        "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in subset])),
                        "gate_precision": rate_or_nan([row.alpha_point_recoverable_flag for row in open_subset]),
                        "mean_best_alpha_output_abs_error_open": mean_or_nan([row.best_alpha_output_abs_error for row in open_subset]),
                        "mean_anchored_alpha_output_abs_error_open": mean_or_nan([row.anchored_alpha_output_abs_error for row in open_subset]),
                        "mean_refined_alpha_output_abs_error_open": mean_or_nan([row.refined_alpha_output_abs_error for row in open_subset]),
                        "mean_refined_alpha_bank_log_span_open": mean_or_nan([row.refined_alpha_bank_log_span for row in open_subset]),
                        "refined_beats_anchored_rate_open": rate_or_nan([row.refined_beats_anchored_flag for row in open_subset]),
                        "refined_beats_best_rate_open": rate_or_nan([row.refined_beats_best_flag for row in open_subset]),
                    }
                )
    return summary


def plot_open_alpha_errors(path: str, split_summary: list[dict[str, float | str]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([float(item["mean_best_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    anchored = np.array([float(item["mean_anchored_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    refined = np.array([float(item["mean_refined_alpha_output_abs_error_open"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - width, best, width=width, color="#9c6644", label="best-bank ensemble")
    ax.bar(x, anchored, width=width, color="#577590", label="anchored ensemble")
    ax.bar(x + width, refined, width=width, color="#2a9d8f", label="conditional refined ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean open-trial alpha abs error")
    ax.set_title("Layer 3 alpha output error on gate-open trials")
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_open_alpha_spans(path: str, split_summary: list[dict[str, float | str]]) -> None:
    splits = [str(item["split"]) for item in split_summary]
    best = np.array([float(item["mean_best_alpha_bank_log_span_open"]) for item in split_summary], dtype=float)
    anchored = np.array([float(item["mean_anchored_alpha_bank_log_span_open"]) for item in split_summary], dtype=float)
    refined = np.array([float(item["mean_refined_alpha_bank_log_span_open"]) for item in split_summary], dtype=float)
    x = np.arange(len(splits))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.97)
    ax.bar(x - width, best, width=width, color="#8d99ae", label="best-bank span")
    ax.bar(x, anchored, width=width, color="#577590", label="anchored-bank span")
    ax.bar(x + width, refined, width=width, color="#2a9d8f", label="refined-bank span")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mean open-trial alpha bank log-span")
    ax.set_title("Layer 3 keeps alpha bank spread small on gate-open trials")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    gate_rule = load_gate_rule()
    gate_metric_name = str(gate_rule["metric"])
    gate_direction = str(gate_rule["direction"])
    gate_threshold = float(gate_rule["threshold"])
    candidate_rows_by_bank = load_candidate_rows()
    layer2_trial_rows = load_layer2_trial_rows()
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}

    prepared_trials: list[TrialPrepared] = []
    trial_bases: list[TrialBase] = []
    for split, observation_seeds in BLOCK_SPECS.items():
        for observation_seed in observation_seeds:
            for condition in TARGET_CONDITIONS:
                regime = regime_map[condition]
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    prepared = prepare_trial(
                        split=split,
                        observation_seed=int(observation_seed),
                        condition=condition,
                        skew_bin=skew_bin,
                        regime=regime,
                        candidate_rows_by_bank=candidate_rows_by_bank,
                        layer2_trial_rows=layer2_trial_rows,
                    )
                    prepared_trials.append(prepared)
                    trial_bases.append(prepared.trial_base)

    bank_rows: list[BankRow] = []
    trial_rows: list[TrialRow] = []
    for prepared in prepared_trials:
        new_bank_rows, trial_row = refine_trial(prepared, gate_metric_name, gate_threshold, gate_direction)
        bank_rows.extend(new_bank_rows)
        trial_rows.append(trial_row)

    split_summary = summarize_by_split(trial_rows, gate_metric_name, gate_direction)
    condition_summary = summarize_by_condition(trial_rows)
    cell_summary = summarize_by_cell(trial_rows)

    holdout_rows = [row for row in trial_bases if row.split == "holdout"]
    confirmation_rows = [row for row in trial_bases if row.split == "confirmation"]
    open_rows = [row for row in trial_rows if row.gate_open_flag == 1]
    unrecoverable_rows = [row for row in trial_rows if row.alpha_point_unrecoverable_flag == 1]

    threshold_summary = {
        "metric": gate_metric_name,
        "direction": gate_direction,
        "threshold": gate_threshold,
        "calibration_balanced_accuracy": float(gate_rule["calibration_balanced_accuracy"]),
        "holdout_balanced_accuracy": balanced_accuracy(holdout_rows, gate_metric_name, gate_threshold, gate_direction),
        "confirmation_balanced_accuracy": balanced_accuracy(
            confirmation_rows,
            gate_metric_name,
            gate_threshold,
            gate_direction,
        ),
        "overall_balanced_accuracy": balanced_accuracy(trial_bases, gate_metric_name, gate_threshold, gate_direction),
    }

    global_summary = {
        "final_bank_size": int(np.mean([row.candidate_count for row in bank_rows])),
        "bank_seeds": list(BANK_SEEDS),
        "top_k_refinement_seeds": TOP_K_REFINEMENT_SEEDS,
        "gate_metric": gate_metric_name,
        "gate_direction": gate_direction,
        "trial_count": len(trial_rows),
        "point_output_count": int(sum(row.point_output_flag for row in trial_rows)),
        "point_output_rate": float(np.mean([row.point_output_flag for row in trial_rows])),
        "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in trial_rows])),
        "gate_precision": rate_or_nan([row.alpha_point_recoverable_flag for row in open_rows]),
        "gate_reject_unrecoverable_rate": rate_or_nan([row.gate_closed_and_unrecoverable_flag for row in unrecoverable_rows]),
        "mean_best_alpha_output_abs_error_open": mean_or_nan([row.best_alpha_output_abs_error for row in open_rows]),
        "mean_anchored_alpha_output_abs_error_open": mean_or_nan([row.anchored_alpha_output_abs_error for row in open_rows]),
        "mean_refined_alpha_output_abs_error_open": mean_or_nan([row.refined_alpha_output_abs_error for row in open_rows]),
        "mean_best_alpha_bank_log_span_open": mean_or_nan([row.best_alpha_bank_log_span for row in open_rows]),
        "mean_anchored_alpha_bank_log_span_open": mean_or_nan([row.anchored_alpha_bank_log_span for row in open_rows]),
        "mean_refined_alpha_bank_log_span_open": mean_or_nan([row.refined_alpha_bank_log_span for row in open_rows]),
        "refined_beats_anchored_rate_open": rate_or_nan([row.refined_beats_anchored_flag for row in open_rows]),
        "refined_beats_best_rate_open": rate_or_nan([row.refined_beats_best_flag for row in open_rows]),
    }

    payload = {
        "summary": global_summary,
        "gate_threshold": threshold_summary,
        "by_split": split_summary,
        "by_condition": condition_summary,
        "by_cell": cell_summary,
    }

    prefix = "backbone_conditional_alpha_solver_informed_bank"
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_bank_rows.csv"), [asdict(row) for row in bank_rows])
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_trials.csv"), [asdict(row) for row in trial_rows])
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_condition_summary.csv"), condition_summary)
    write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_cell_summary.csv"), cell_summary)
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_summary.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    plot_open_alpha_errors(os.path.join(FIGURE_DIR, f"{prefix}_alpha_error.png"), split_summary)
    plot_open_alpha_spans(os.path.join(FIGURE_DIR, f"{prefix}_alpha_span.png"), split_summary)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
