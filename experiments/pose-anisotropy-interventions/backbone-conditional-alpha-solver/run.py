"""
Backbone-first conditional alpha recovery layer for the pose-free anisotropic solver challenge.

This is the third capability layer in the layered solver program.

Layer 1 established a stable geometry backbone across fresh bank seeds.
Layer 2 established a trustworthy observability gate for point alpha recovery.

Layer 3 keeps both layers intact:

- recover the stable backbone first
- open the Layer 2 gate only on the point-recoverable region
- run a dedicated alpha refinement only inside that region
- abstain everywhere else

This is still not the full solver challenge. It is the first conditional
alpha-recovery layer built on top of the validated backbone and gate.
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

candidate_conditioned_search, = load_symbols(
    "run_candidate_conditioned_alignment_experiment_refiner",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "candidate_conditioned_search",
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

FOCUS_ALPHA_BIN, FOCUS_CONDITIONS, SolverContext, marginalized_bank_scores, softmin_temperature = load_symbols(
    "run_joint_pose_marginalized_solver_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py",
    "FOCUS_ALPHA_BIN",
    "FOCUS_CONDITIONS",
    "SolverContext",
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
GEOMETRY_ANCHOR_SCALE = 0.10
ALPHA_STABLE_LOG_SPAN_THRESHOLD = 0.20
ALPHA_POINT_ABS_ERROR_THRESHOLD = 0.15
GATE_METRIC_NAME = "mean_anchored_alpha_log_std"
TOP_K_REFINEMENT_SEEDS = 3


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
    mean_anchored_alpha_log_std: float
    mean_anchored_alpha_log_span: float
    mean_anchored_effective_count: float
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


@dataclass(frozen=True)
class BankContext:
    seed: int
    params_list: list[tuple[float, float, float, float, float, float]]
    shifted_bank: np.ndarray
    geometries: np.ndarray
    alpha_logs: np.ndarray


@dataclass(frozen=True)
class PreparedBank:
    context: BankContext
    candidate_indices: np.ndarray
    candidate_scores: np.ndarray
    candidate_geometries: np.ndarray
    candidate_alpha_logs: np.ndarray
    local_geometry_consensus: np.ndarray
    best_entropy: float
    best_alpha_log: float
    best_alpha_abs_error: float
    anchored_mean_log: float
    anchored_std_log: float
    anchored_span_log: float
    anchored_effective_count: float
    anchored_alpha_abs_error: float


@dataclass(frozen=True)
class PreparedTrial:
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
    band: float
    trial_backbone: np.ndarray
    prepared_banks: list[PreparedBank]


@dataclass(frozen=True)
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
    mean_anchored_alpha_log_std: float
    mean_anchored_alpha_log_span: float
    mean_anchored_effective_count: float
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


def score_weights(scores: np.ndarray, scale: float) -> np.ndarray:
    offsets = scores - float(np.min(scores))
    weights = np.exp(-offsets / max(scale, NUMERIC_EPS))
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(scores), 1.0 / len(scores))
    return weights / total


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


def anchored_alpha_posterior(
    candidate_scores: np.ndarray,
    candidate_geometries: np.ndarray,
    candidate_alpha_logs: np.ndarray,
    geometry_anchor: np.ndarray,
    geometry_ranges: np.ndarray,
    score_scale: float,
) -> tuple[np.ndarray, float, float, float, float]:
    base_weights = score_weights(candidate_scores, score_scale)
    geometry_distance = np.mean(
        np.abs(candidate_geometries - geometry_anchor[None, :]) / geometry_ranges[None, :],
        axis=1,
    )
    geometry_penalty = np.exp(-geometry_distance / GEOMETRY_ANCHOR_SCALE)
    anchored_weights = normalize_weights(base_weights * geometry_penalty)
    anchored_mean_log = float(np.sum(anchored_weights * candidate_alpha_logs))
    anchored_var_log = float(np.sum(anchored_weights * (candidate_alpha_logs - anchored_mean_log) ** 2))
    anchored_std_log = float(math.sqrt(max(anchored_var_log, 0.0)))
    anchored_span_log = float(
        weighted_quantile(candidate_alpha_logs, anchored_weights, 0.90)
        - weighted_quantile(candidate_alpha_logs, anchored_weights, 0.10)
    )
    return anchored_weights, anchored_mean_log, anchored_std_log, anchored_span_log, effective_count(anchored_weights)


def balanced_accuracy(rows: list[TrialBase], metric_name: str, threshold: float) -> float:
    labels = np.array([row.alpha_point_unrecoverable_flag for row in rows], dtype=int)
    preds = np.array([int(getattr(row, metric_name) >= threshold) for row in rows], dtype=int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = float(np.sum((preds == 1) & (labels == 1)) / positives)
    tnr = float(np.sum((preds == 0) & (labels == 0)) / negatives)
    return 0.5 * (tpr + tnr)


def choose_threshold(calibration_rows: list[TrialBase], metric_name: str) -> tuple[float, float]:
    values = sorted({float(getattr(row, metric_name)) for row in calibration_rows})
    if not values:
        return 0.0, float("nan")
    candidates = [values[0] - 1.0e-6]
    candidates.extend((lo + hi) * 0.5 for lo, hi in zip(values[:-1], values[1:]))
    candidates.append(values[-1] + 1.0e-6)
    scored = [(balanced_accuracy(calibration_rows, metric_name, threshold), threshold) for threshold in candidates]
    scored.sort(key=lambda item: (item[0], -item[1]))
    best_score, best_threshold = scored[-1]
    return float(best_threshold), float(best_score)


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def rate_or_nan(flags: list[int]) -> float:
    if not flags:
        return float("nan")
    return float(np.mean(flags))


def prepare_trial(
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    regime: dict[str, float | str | int],
    bank_contexts: list[BankContext],
    geometry_ranges: np.ndarray,
) -> tuple[PreparedTrial, TrialBase]:
    trial_rng = make_trial_rng(observation_seed, condition, skew_bin)
    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
    clean_signature = anisotropic_forward_signature(true_params)
    _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
    temperature = softmin_temperature(regime)
    band = score_band(regime)
    true_alpha = float(true_params[5])

    prepared_banks: list[PreparedBank] = []
    local_consensus_geometries = []
    best_entropies = []
    candidate_counts = []
    best_alpha_logs = []
    anchored_alpha_logs = []
    best_alpha_errors = []
    anchored_alpha_errors = []
    anchored_alpha_log_stds = []
    anchored_alpha_log_spans = []
    anchored_effective_counts = []

    for context in bank_contexts:
        scores, _ = marginalized_bank_scores(observed_signature, mask, context.shifted_bank, temperature)
        best_idx = int(np.argmin(scores))
        candidate_indices = select_candidate_indices(scores, band)
        candidate_scores = scores[candidate_indices]
        candidate_geometries = context.geometries[candidate_indices]
        candidate_alpha_logs = context.alpha_logs[candidate_indices]

        geometry_weights = score_weights(candidate_scores, band)
        local_geometry_consensus = np.sum(candidate_geometries * geometry_weights[:, None], axis=0)
        _, anchored_mean_log, anchored_std_log, anchored_span_log, anchored_count = anchored_alpha_posterior(
            candidate_scores,
            candidate_geometries,
            candidate_alpha_logs,
            local_geometry_consensus,
            geometry_ranges,
            band,
        )

        solver_context = SolverContext(observed_signature, mask)
        _, _, _, best_entropy = solver_context.score_params(context.params_list[best_idx], temperature)

        best_alpha_log = float(context.alpha_logs[best_idx])
        best_alpha = float(math.exp(best_alpha_log))
        anchored_alpha = float(math.exp(anchored_mean_log))

        prepared_banks.append(
            PreparedBank(
                context=context,
                candidate_indices=candidate_indices,
                candidate_scores=candidate_scores,
                candidate_geometries=candidate_geometries,
                candidate_alpha_logs=candidate_alpha_logs,
                local_geometry_consensus=local_geometry_consensus,
                best_entropy=float(best_entropy),
                best_alpha_log=best_alpha_log,
                best_alpha_abs_error=float(abs(best_alpha - true_alpha)),
                anchored_mean_log=anchored_mean_log,
                anchored_std_log=anchored_std_log,
                anchored_span_log=anchored_span_log,
                anchored_effective_count=anchored_count,
                anchored_alpha_abs_error=float(abs(anchored_alpha - true_alpha)),
            )
        )

        local_consensus_geometries.append(local_geometry_consensus)
        best_entropies.append(float(best_entropy))
        candidate_counts.append(int(len(candidate_indices)))
        best_alpha_logs.append(best_alpha_log)
        anchored_alpha_logs.append(anchored_mean_log)
        best_alpha_errors.append(float(abs(best_alpha - true_alpha)))
        anchored_alpha_errors.append(float(abs(anchored_alpha - true_alpha)))
        anchored_alpha_log_stds.append(anchored_std_log)
        anchored_alpha_log_spans.append(anchored_span_log)
        anchored_effective_counts.append(anchored_count)

    trial_backbone = np.mean(np.array(local_consensus_geometries), axis=0)
    best_alpha_output = float(math.exp(float(np.mean(best_alpha_logs))))
    anchored_alpha_output = float(math.exp(float(np.mean(anchored_alpha_logs))))
    anchored_alpha_bank_log_span = float(np.max(anchored_alpha_logs) - np.min(anchored_alpha_logs))
    anchored_alpha_abs_error_mean = float(np.mean(anchored_alpha_errors))
    alpha_point_recoverable_flag = int(
        anchored_alpha_bank_log_span < ALPHA_STABLE_LOG_SPAN_THRESHOLD
        and anchored_alpha_abs_error_mean < ALPHA_POINT_ABS_ERROR_THRESHOLD
    )

    prepared_trial = PreparedTrial(
        split=split,
        observation_seed=int(observation_seed),
        condition=condition,
        geometry_skew_bin=skew_bin,
        true_alpha=true_alpha,
        true_t=float(true_params[1]),
        true_rotation_shift=int(true_shift),
        observed_signature=observed_signature,
        mask=mask,
        temperature=temperature,
        band=band,
        trial_backbone=trial_backbone,
        prepared_banks=prepared_banks,
    )
    trial_base = TrialBase(
        split=split,
        observation_seed=int(observation_seed),
        condition=condition,
        geometry_skew_bin=skew_bin,
        true_alpha=true_alpha,
        true_t=float(true_params[1]),
        true_rotation_shift=int(true_shift),
        mean_best_entropy=float(np.mean(best_entropies)),
        mean_candidate_count=float(np.mean(candidate_counts)),
        mean_anchored_alpha_log_std=float(np.mean(anchored_alpha_log_stds)),
        mean_anchored_alpha_log_span=float(np.mean(anchored_alpha_log_spans)),
        mean_anchored_effective_count=float(np.mean(anchored_effective_counts)),
        best_alpha_bank_log_span=float(np.max(best_alpha_logs) - np.min(best_alpha_logs)),
        anchored_alpha_bank_log_span=anchored_alpha_bank_log_span,
        best_alpha_abs_error_mean=float(np.mean(best_alpha_errors)),
        anchored_alpha_abs_error_mean=anchored_alpha_abs_error_mean,
        best_alpha_output=best_alpha_output,
        best_alpha_output_abs_error=float(abs(best_alpha_output - true_alpha)),
        anchored_alpha_output=anchored_alpha_output,
        anchored_alpha_output_abs_error=float(abs(anchored_alpha_output - true_alpha)),
        alpha_point_recoverable_flag=alpha_point_recoverable_flag,
        alpha_point_unrecoverable_flag=int(1 - alpha_point_recoverable_flag),
        trial_backbone=trial_backbone,
    )
    return prepared_trial, trial_base


def refine_open_trial(prepared_trial: PreparedTrial, trial_base: TrialBase, gate_threshold: float, geometry_ranges: np.ndarray) -> tuple[list[BankRow], TrialRow]:
    gate_open_flag = int(getattr(trial_base, GATE_METRIC_NAME) < gate_threshold)
    bank_rows: list[BankRow] = []

    refined_alpha_logs = []
    refined_alpha_errors = []

    for prepared_bank in prepared_trial.prepared_banks:
        refined_alpha = float("nan")
        refined_alpha_error = float("nan")
        refined_count = float("nan")
        refined_seed_count = 0

        if gate_open_flag:
            backbone_weights, _, _, _, _ = anchored_alpha_posterior(
                prepared_bank.candidate_scores,
                prepared_bank.candidate_geometries,
                prepared_bank.candidate_alpha_logs,
                prepared_trial.trial_backbone,
                geometry_ranges,
                prepared_trial.band,
            )
            top_order = np.argsort(-backbone_weights)[:TOP_K_REFINEMENT_SEEDS]
            refined_logs_local = []
            refined_scores_local = []
            refined_base_weights = []
            for local_idx in top_order:
                seed_idx = int(prepared_bank.candidate_indices[int(local_idx)])
                seed_params = prepared_bank.context.params_list[seed_idx]
                refined_params, _, _, refined_score = candidate_conditioned_search(
                    prepared_trial.observed_signature,
                    prepared_trial.mask,
                    seed_params,
                    prepared_trial.temperature,
                )
                refined_logs_local.append(math.log(float(refined_params[5])))
                refined_scores_local.append(float(refined_score))
                refined_base_weights.append(float(backbone_weights[int(local_idx)]))

            refined_scores_arr = np.array(refined_scores_local, dtype=float)
            refined_base_weights_arr = np.array(refined_base_weights, dtype=float)
            score_offsets = refined_scores_arr - float(np.min(refined_scores_arr))
            refined_weights = normalize_weights(
                refined_base_weights_arr * np.exp(-score_offsets / max(prepared_trial.band, NUMERIC_EPS))
            )
            refined_log = float(np.sum(refined_weights * np.array(refined_logs_local, dtype=float)))
            refined_alpha = float(math.exp(refined_log))
            refined_alpha_error = float(abs(refined_alpha - prepared_trial.true_alpha))
            refined_count = effective_count(refined_weights)
            refined_seed_count = int(len(refined_logs_local))
            refined_alpha_logs.append(refined_log)
            refined_alpha_errors.append(refined_alpha_error)

        bank_rows.append(
            BankRow(
                split=prepared_trial.split,
                observation_seed=int(prepared_trial.observation_seed),
                bank_seed=int(prepared_bank.context.seed),
                condition=prepared_trial.condition,
                geometry_skew_bin=prepared_trial.geometry_skew_bin,
                true_alpha=float(prepared_trial.true_alpha),
                true_t=float(prepared_trial.true_t),
                true_rotation_shift=int(prepared_trial.true_rotation_shift),
                candidate_count=int(len(prepared_bank.candidate_indices)),
                best_entropy=float(prepared_bank.best_entropy),
                best_alpha=float(math.exp(prepared_bank.best_alpha_log)),
                best_alpha_abs_error=float(prepared_bank.best_alpha_abs_error),
                anchored_alpha=float(math.exp(prepared_bank.anchored_mean_log)),
                anchored_alpha_abs_error=float(prepared_bank.anchored_alpha_abs_error),
                anchored_alpha_log_std=float(prepared_bank.anchored_std_log),
                anchored_alpha_log_span=float(prepared_bank.anchored_span_log),
                anchored_effective_count=float(prepared_bank.anchored_effective_count),
                local_consensus_rho12=float(prepared_bank.local_geometry_consensus[0]),
                local_consensus_rho13=float(prepared_bank.local_geometry_consensus[1]),
                local_consensus_rho23=float(prepared_bank.local_geometry_consensus[2]),
                trial_backbone_rho12=float(prepared_trial.trial_backbone[0]),
                trial_backbone_rho13=float(prepared_trial.trial_backbone[1]),
                trial_backbone_rho23=float(prepared_trial.trial_backbone[2]),
                gate_open_flag=gate_open_flag,
                refined_alpha=float(refined_alpha),
                refined_alpha_abs_error=float(refined_alpha_error),
                refined_effective_count=float(refined_count),
                refined_seed_count=int(refined_seed_count),
            )
        )

    if gate_open_flag:
        refined_alpha_output = float(math.exp(float(np.mean(refined_alpha_logs))))
        refined_alpha_output_abs_error = float(abs(refined_alpha_output - prepared_trial.true_alpha))
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
        mean_anchored_alpha_log_std=float(trial_base.mean_anchored_alpha_log_std),
        mean_anchored_alpha_log_span=float(trial_base.mean_anchored_alpha_log_span),
        mean_anchored_effective_count=float(trial_base.mean_anchored_effective_count),
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


def summarize_by_split(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary = []
    for split in BLOCK_SPECS:
        subset = [row for row in rows if row.split == split]
        if not subset:
            continue
        open_subset = [row for row in subset if row.gate_open_flag == 1]
        unrecoverable_subset = [row for row in subset if row.alpha_point_unrecoverable_flag == 1]
        summary.append(
            {
                "split": split,
                "count": len(subset),
                "point_output_count": int(sum(row.point_output_flag for row in subset)),
                "point_output_rate": float(np.mean([row.point_output_flag for row in subset])),
                "alpha_point_recoverable_rate": float(np.mean([row.alpha_point_recoverable_flag for row in subset])),
                "gate_balanced_accuracy": balanced_accuracy(
                    [TrialBase(
                        split=row.split,
                        observation_seed=row.observation_seed,
                        condition=row.condition,
                        geometry_skew_bin=row.geometry_skew_bin,
                        true_alpha=row.true_alpha,
                        true_t=row.true_t,
                        true_rotation_shift=row.true_rotation_shift,
                        mean_best_entropy=row.mean_best_entropy,
                        mean_candidate_count=row.mean_candidate_count,
                        mean_anchored_alpha_log_std=row.mean_anchored_alpha_log_std,
                        mean_anchored_alpha_log_span=row.mean_anchored_alpha_log_span,
                        mean_anchored_effective_count=row.mean_anchored_effective_count,
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
                        trial_backbone=np.array([row.trial_backbone_rho12, row.trial_backbone_rho13, row.trial_backbone_rho23], dtype=float),
                    ) for row in subset],
                    GATE_METRIC_NAME,
                    subset[0].gate_threshold,
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
        for condition in FOCUS_CONDITIONS:
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
    regime_map = {str(regime["name"]): regime for regime in OBSERVATION_REGIMES}
    bank_contexts = [build_bank_context(bank_seed) for bank_seed in BANK_SEEDS]
    geometry_ranges = empirical_geometry_ranges(bank_contexts)

    prepared_trials: list[PreparedTrial] = []
    trial_bases: list[TrialBase] = []

    for split, observation_seeds in BLOCK_SPECS.items():
        for observation_seed in observation_seeds:
            for condition in FOCUS_CONDITIONS:
                regime = regime_map[condition]
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    prepared_trial, trial_base = prepare_trial(
                        split=split,
                        observation_seed=int(observation_seed),
                        condition=condition,
                        skew_bin=skew_bin,
                        regime=regime,
                        bank_contexts=bank_contexts,
                        geometry_ranges=geometry_ranges,
                    )
                    prepared_trials.append(prepared_trial)
                    trial_bases.append(trial_base)

    calibration_bases = [row for row in trial_bases if row.split == "calibration"]
    gate_threshold, calibration_score = choose_threshold(calibration_bases, GATE_METRIC_NAME)

    bank_rows: list[BankRow] = []
    trial_rows: list[TrialRow] = []
    for prepared_trial, trial_base in zip(prepared_trials, trial_bases):
        new_bank_rows, trial_row = refine_open_trial(prepared_trial, trial_base, gate_threshold, geometry_ranges)
        bank_rows.extend(new_bank_rows)
        trial_rows.append(trial_row)

    split_summary = summarize_by_split(trial_rows)
    condition_summary = summarize_by_condition(trial_rows)
    cell_summary = summarize_by_cell(trial_rows)

    holdout_rows = [row for row in trial_bases if row.split == "holdout"]
    confirmation_rows = [row for row in trial_bases if row.split == "confirmation"]
    threshold_summary = {
        "metric": GATE_METRIC_NAME,
        "threshold": float(gate_threshold),
        "calibration_balanced_accuracy": float(calibration_score),
        "holdout_balanced_accuracy": balanced_accuracy(holdout_rows, GATE_METRIC_NAME, gate_threshold),
        "confirmation_balanced_accuracy": balanced_accuracy(confirmation_rows, GATE_METRIC_NAME, gate_threshold),
        "overall_balanced_accuracy": balanced_accuracy(trial_bases, GATE_METRIC_NAME, gate_threshold),
    }

    open_rows = [row for row in trial_rows if row.gate_open_flag == 1]
    unrecoverable_rows = [row for row in trial_rows if row.alpha_point_unrecoverable_flag == 1]
    global_summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "bank_seeds": list(BANK_SEEDS),
        "score_band_rule": "best_score + max(noise_sigma^2, 5e-5)",
        "top_k_refinement_seeds": TOP_K_REFINEMENT_SEEDS,
        "gate_metric": GATE_METRIC_NAME,
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

    output_payload = {
        "summary": global_summary,
        "gate_threshold": threshold_summary,
        "by_split": split_summary,
        "by_condition": condition_summary,
        "by_cell": cell_summary,
    }

    write_csv(os.path.join(OUTPUT_DIR, "backbone_conditional_alpha_solver_bank_rows.csv"), [asdict(row) for row in bank_rows])
    write_csv(os.path.join(OUTPUT_DIR, "backbone_conditional_alpha_solver_trials.csv"), [asdict(row) for row in trial_rows])
    write_csv(os.path.join(OUTPUT_DIR, "backbone_conditional_alpha_solver_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, "backbone_conditional_alpha_solver_condition_summary.csv"), condition_summary)
    write_csv(os.path.join(OUTPUT_DIR, "backbone_conditional_alpha_solver_cell_summary.csv"), cell_summary)
    with open(os.path.join(OUTPUT_DIR, "backbone_conditional_alpha_solver_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    plot_open_alpha_errors(
        os.path.join(FIGURE_DIR, "backbone_conditional_alpha_solver_alpha_error.png"),
        split_summary,
    )
    plot_open_alpha_spans(
        os.path.join(FIGURE_DIR, "backbone_conditional_alpha_solver_alpha_span.png"),
        split_summary,
    )

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
