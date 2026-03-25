"""
First informed-bank experiment for the backbone layer.

This experiment tests whether a scout-conditioned, mode-aware bank can reduce
Layer 1 compression load without breaking backbone recovery.

The comparison is deliberately narrow:

- one-shot random bank
- scout + random fill
- scout + persistent-mode informed expansion

The informed bank uses only observation-side scout structure:

- cluster-aware carryover anchors
- local expansion around persistent scout modes
- a fixed exploration reserve
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

GEOMETRY_SKEW_BIN_LABELS, sample_conditioned_parameters = load_symbols(
    "run_candidate_conditioned_alignment_experiment_for_persistent_mode_bank",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "GEOMETRY_SKEW_BIN_LABELS",
    "sample_conditioned_parameters",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment_for_persistent_mode_bank",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

(
    ALPHA_MAX,
    ALPHA_MIN,
    anisotropic_forward_signature,
    control_invariants,
    sample_anisotropic_parameters,
) = load_symbols(
    "run_weighted_anisotropic_inverse_experiment_for_persistent_mode_bank",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "anisotropic_forward_signature",
    "control_invariants",
    "sample_anisotropic_parameters",
)

OBSERVATION_REGIMES, write_csv, GEOMETRY_BOUNDS = load_symbols(
    "run_weighted_multisource_inverse_experiment_for_persistent_mode_bank",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "write_csv",
    "GEOMETRY_BOUNDS",
)

FOCUS_ALPHA_BIN, FOCUS_CONDITIONS, marginalized_bank_scores, softmin_temperature = load_symbols(
    "run_joint_pose_marginalized_solver_experiment_for_persistent_mode_bank",
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

RANDOM_BANK_SIZE = 300
SCOUT_BANK_SIZE = 120
FINAL_BANK_SIZE = 300
FINAL_CARRYOVER_BUDGET = 72
FINAL_LOCAL_EXPANSION_BUDGET = 192
FINAL_EXPLORATION_BUDGET = 36

MIN_SCORE_BAND = 5.0e-5
FRONTIER_CAPTURE_COUNT = 16
GEOMETRY_CLUSTER_THRESHOLD = 0.08
NUMERIC_EPS = 1.0e-9
ALPHA_LOG_RANGE = float(math.log(ALPHA_MAX) - math.log(ALPHA_MIN))

GLOBAL_RANGE_SAMPLE_SIZE = 5000
GLOBAL_RANGE_SEED = 20260301

METHOD_RANDOM = "one_shot_random"
METHOD_SCOUT_RANDOM = "scout_random_fill"
METHOD_INFORMED = "persistent_mode_informed"
METHOD_ORDER = (METHOD_RANDOM, METHOD_SCOUT_RANDOM, METHOD_INFORMED)

ARCHETYPE_BONUS = {
    "dominant_core": 1.00,
    "broad_fan": 0.90,
    "alpha_fan": 0.70,
    "compact_minor": 0.45,
    "fringe_singleton": 0.10,
}

LOCAL_MIX_RANGES = {
    "dominant_core": (0.04, 0.10),
    "broad_fan": (0.08, 0.16),
    "alpha_fan": (0.08, 0.18),
    "compact_minor": (0.06, 0.12),
    "fringe_singleton": (0.10, 0.18),
}


@dataclass(frozen=True)
class BankContext:
    seed: int
    params_list: list[tuple[float, float, float, float, float, float]]
    shifted_bank: np.ndarray
    geometries: np.ndarray
    alpha_logs: np.ndarray


@dataclass
class ScoutCluster:
    cluster_id: int
    member_indices: list[int]
    member_weights: list[float]
    cluster_mass: float
    cluster_size: int
    cluster_geometry_span: float
    cluster_alpha_span: float
    cluster_best_score: float
    archetype: str


@dataclass
class MethodBankRow:
    method: str
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
class MethodTrialRow:
    method: str
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
    consensus_geometry_bank_span_norm: float
    best_geometry_mae_mean: float
    consensus_geometry_mae_mean: float
    geometry_mae_gain: float
    compression_load: float


def condition_index(condition: str) -> int:
    return FOCUS_CONDITIONS.index(condition)


def skew_index(skew_bin: str) -> int:
    return GEOMETRY_SKEW_BIN_LABELS.index(skew_bin)


def make_trial_rng(observation_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(observation_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


def bank_rng(bank_seed: int, observation_seed: int, condition: str, skew_bin: str, stage: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [int(bank_seed), int(observation_seed), condition_index(condition), skew_index(skew_bin), int(stage)]
    )
    return np.random.default_rng(sequence)


def canonicalize_geometry(params: tuple[float, float, float, float, float, float]) -> tuple[np.ndarray, float]:
    geometry, _, alpha = control_invariants(params)
    swapped = np.array([geometry[0], geometry[2], geometry[1]])
    if tuple(swapped) < tuple(geometry):
        return swapped, float(alpha)
    return geometry, float(alpha)


def sample_random_params_list(sample_size: int, rng: np.random.Generator) -> list[tuple[float, float, float, float, float, float]]:
    return [sample_anisotropic_parameters(rng) for _ in range(sample_size)]


def build_bank_context_from_params(
    bank_seed: int,
    params_list: list[tuple[float, float, float, float, float, float]],
) -> BankContext:
    signatures = np.array([anisotropic_forward_signature(params) for params in params_list], dtype=float)
    shifted_bank = build_shift_stack(signatures)
    geometries = []
    alpha_logs = []
    for params in params_list:
        geometry, alpha = canonicalize_geometry(params)
        geometries.append(geometry)
        alpha_logs.append(math.log(alpha))
    return BankContext(
        seed=int(bank_seed),
        params_list=list(params_list),
        shifted_bank=shifted_bank,
        geometries=np.array(geometries, dtype=float),
        alpha_logs=np.array(alpha_logs, dtype=float),
    )


def global_geometry_ranges() -> np.ndarray:
    rng = np.random.default_rng(GLOBAL_RANGE_SEED)
    geometries = []
    for _ in range(GLOBAL_RANGE_SAMPLE_SIZE):
        geometry, _ = canonicalize_geometry(sample_anisotropic_parameters(rng))
        geometries.append(geometry)
    all_geometries = np.array(geometries, dtype=float)
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


def capture_candidate_indices(scores: np.ndarray, band: float) -> tuple[np.ndarray, set[int], np.ndarray]:
    order = np.argsort(scores)
    best_score = float(scores[order[0]])
    band_indices = order[scores[order] <= best_score + band]
    if len(band_indices) == 0:
        band_indices = order[:1]
    band_set = {int(idx) for idx in band_indices}
    frontier = [int(idx) for idx in order if int(idx) not in band_set][:FRONTIER_CAPTURE_COUNT]
    captured = np.array(list(map(int, band_indices)) + frontier, dtype=int)
    return captured, band_set, order


def geometry_mae(true_geometry: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.abs(true_geometry - estimate)))


def geometry_span_norm(geometries: np.ndarray, geometry_ranges: np.ndarray) -> float:
    if len(geometries) <= 1:
        return 0.0
    span = (np.max(geometries, axis=0) - np.min(geometries, axis=0)) / geometry_ranges
    return float(np.mean(span))


def consensus_weights(scores: np.ndarray, scale: float) -> np.ndarray:
    offsets = scores - float(np.min(scores))
    logits = np.exp(-offsets / max(scale, NUMERIC_EPS))
    total = float(np.sum(logits))
    if total <= 0.0:
        return np.full(len(scores), 1.0 / len(scores))
    return logits / total


def greedy_geometry_clusters(
    geometries: np.ndarray,
    geometry_ranges: np.ndarray,
    ordering_weights: np.ndarray,
) -> np.ndarray:
    if len(geometries) == 0:
        return np.array([], dtype=int)
    normed = geometries / geometry_ranges[None, :]
    assignments = np.full(len(geometries), -1, dtype=int)
    centers: list[np.ndarray] = []
    center_weights: list[float] = []
    order = np.argsort(-ordering_weights)
    for idx in order:
        point = normed[idx]
        assigned = False
        for cluster_id, center in enumerate(centers):
            distance = float(np.mean(np.abs(point - center)))
            if distance <= GEOMETRY_CLUSTER_THRESHOLD:
                assignments[idx] = cluster_id
                total = center_weights[cluster_id] + float(ordering_weights[idx])
                centers[cluster_id] = (
                    centers[cluster_id] * center_weights[cluster_id] + point * ordering_weights[idx]
                ) / max(total, NUMERIC_EPS)
                center_weights[cluster_id] = total
                assigned = True
                break
        if not assigned:
            cluster_id = len(centers)
            centers.append(point.copy())
            center_weights.append(float(ordering_weights[idx]))
            assignments[idx] = cluster_id
    return assignments


def classify_cluster(cluster_mass: float, cluster_size: int, cluster_geometry_span: float, cluster_alpha_span: float) -> str:
    if cluster_mass >= 0.18 and cluster_size >= 8:
        return "dominant_core"
    if cluster_alpha_span >= 0.35 and cluster_geometry_span >= 0.12:
        return "broad_fan"
    if cluster_alpha_span >= 0.25:
        return "alpha_fan"
    if cluster_size <= 2 and cluster_mass <= 0.02:
        return "fringe_singleton"
    return "compact_minor"


def full_weights(params: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.array([params[3], params[4], 1.0 - params[3] - params[4]], dtype=float)


def make_params(
    rho: float,
    t: float,
    h: float,
    weights: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float, float, float]:
    normalized = np.clip(np.array(weights, dtype=float), NUMERIC_EPS, None)
    normalized /= float(np.sum(normalized))
    rho = float(np.clip(rho, GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(np.clip(t, GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(np.clip(h, GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
    return rho, t, h, float(normalized[0]), float(normalized[1]), alpha


def blend_params(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
    mix: float,
) -> tuple[float, float, float, float, float, float]:
    mix = float(np.clip(mix, 0.0, 1.0))
    left_weights = full_weights(left)
    right_weights = full_weights(right)
    blended_weights = (1.0 - mix) * left_weights + mix * right_weights
    return make_params(
        (1.0 - mix) * float(left[0]) + mix * float(right[0]),
        (1.0 - mix) * float(left[1]) + mix * float(right[1]),
        (1.0 - mix) * float(left[2]) + mix * float(right[2]),
        blended_weights,
        (1.0 - mix) * float(left[5]) + mix * float(right[5]),
    )


def draw_member_index(cluster: ScoutCluster, rng: np.random.Generator) -> int:
    weights = np.array(cluster.member_weights, dtype=float)
    weights = np.maximum(weights, NUMERIC_EPS)
    weights /= float(np.sum(weights))
    return int(rng.choice(np.array(cluster.member_indices, dtype=int), p=weights))


def sample_local_candidate(
    cluster: ScoutCluster,
    scout_context: BankContext,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    primary_idx = draw_member_index(cluster, rng)
    primary = scout_context.params_list[primary_idx]

    if len(cluster.member_indices) >= 2:
        secondary_idx = primary_idx
        attempts = 0
        while secondary_idx == primary_idx and attempts < 8:
            secondary_idx = draw_member_index(cluster, rng)
            attempts += 1
        secondary = scout_context.params_list[secondary_idx]
        base = blend_params(primary, secondary, rng.uniform(0.15, 0.85))
    else:
        base = primary

    jitter_source = sample_anisotropic_parameters(rng)
    mix_lo, mix_hi = LOCAL_MIX_RANGES[cluster.archetype]
    jitter_mix = float(rng.uniform(mix_lo, mix_hi))
    return blend_params(base, jitter_source, jitter_mix)


def carryover_indices_from_scout(
    captured_indices: np.ndarray,
    captured_scores: np.ndarray,
    captured_cluster_ids: np.ndarray,
    cluster_rank_lookup: dict[int, int],
    budget: int,
) -> list[int]:
    captured_order = np.argsort(captured_scores)
    members_by_cluster: dict[int, list[int]] = {}
    for local_idx in captured_order:
        cluster_id = int(captured_cluster_ids[local_idx])
        members_by_cluster.setdefault(cluster_id, []).append(int(captured_indices[local_idx]))
    ranked_clusters = [cluster_id for cluster_id, _ in sorted(cluster_rank_lookup.items(), key=lambda item: item[1])]

    selected: list[int] = []
    seen: set[int] = set()
    cursor = 0
    while len(selected) < budget and any(cursor < len(members_by_cluster.get(cluster_id, [])) for cluster_id in ranked_clusters):
        for cluster_id in ranked_clusters:
            members = members_by_cluster.get(cluster_id, [])
            if cursor < len(members):
                candidate_idx = int(members[cursor])
                if candidate_idx not in seen:
                    selected.append(candidate_idx)
                    seen.add(candidate_idx)
                    if len(selected) >= budget:
                        break
        cursor += 1

    if len(selected) < budget:
        for local_idx in captured_order:
            candidate_idx = int(captured_indices[local_idx])
            if candidate_idx not in seen:
                selected.append(candidate_idx)
                seen.add(candidate_idx)
                if len(selected) >= budget:
                    break
    return selected[:budget]


def allocate_cluster_counts(clusters: list[ScoutCluster], budget: int) -> dict[int, int]:
    eligible = [cluster for cluster in clusters if cluster.archetype != "fringe_singleton"]
    if not eligible:
        eligible = clusters
    if not eligible or budget <= 0:
        return {}

    raw_weights = []
    for cluster in eligible:
        span_bonus = 1.0 + 0.7 * float(cluster.cluster_geometry_span) + 0.5 * float(cluster.cluster_alpha_span / max(ALPHA_LOG_RANGE, NUMERIC_EPS))
        score = math.sqrt(max(cluster.cluster_mass, NUMERIC_EPS)) * ARCHETYPE_BONUS[cluster.archetype] * span_bonus
        raw_weights.append(score)
    raw_weights_array = np.array(raw_weights, dtype=float)
    normalized = raw_weights_array / float(np.sum(raw_weights_array))
    fractional = normalized * budget
    counts = np.floor(fractional).astype(int)
    remainder = int(budget - int(np.sum(counts)))
    if remainder > 0:
        order = np.argsort(-(fractional - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {cluster.cluster_id: int(count) for cluster, count in zip(eligible, counts) if int(count) > 0}


def scout_clusters_from_context(
    scout_context: BankContext,
    observed_signature: np.ndarray,
    mask: np.ndarray,
    band: float,
    temperature: float,
    geometry_ranges: np.ndarray,
) -> tuple[list[ScoutCluster], list[int]]:
    scores, _ = marginalized_bank_scores(observed_signature, mask, scout_context.shifted_bank, temperature)
    captured_indices, band_set, order = capture_candidate_indices(scores, band)
    rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(order)}
    band_indices = np.array(sorted(band_set, key=lambda idx: rank_lookup[idx]), dtype=int)
    band_scores = scores[band_indices]
    band_geometries = scout_context.geometries[band_indices]
    geometry_weights = consensus_weights(band_scores, band)
    band_weight_lookup = {int(idx): float(weight) for idx, weight in zip(band_indices, geometry_weights)}

    captured_scores = scores[captured_indices]
    captured_geometries = scout_context.geometries[captured_indices]
    captured_alpha_logs = scout_context.alpha_logs[captured_indices]
    captured_ordering_weights = consensus_weights(captured_scores, band)
    captured_cluster_ids = greedy_geometry_clusters(captured_geometries, geometry_ranges, captured_ordering_weights)

    cluster_ids = sorted(set(int(value) for value in captured_cluster_ids))
    clusters: list[ScoutCluster] = []
    cluster_rank_lookup: dict[int, int] = {}

    mass_pairs: list[tuple[int, float]] = []
    for cluster_id in cluster_ids:
        member_local = np.where(captured_cluster_ids == cluster_id)[0]
        member_indices = [int(captured_indices[idx]) for idx in member_local]
        member_weights = [float(band_weight_lookup.get(int(captured_indices[idx]), 0.0)) for idx in member_local]
        member_geometries = captured_geometries[member_local]
        member_alpha = captured_alpha_logs[member_local]
        cluster_mass = float(sum(member_weights))
        cluster_geometry_span = geometry_span_norm(member_geometries, geometry_ranges)
        cluster_alpha_span = float(np.max(member_alpha) - np.min(member_alpha)) if len(member_alpha) > 1 else 0.0
        cluster_best_score = float(np.min(captured_scores[member_local]))
        archetype = classify_cluster(cluster_mass, len(member_indices), cluster_geometry_span, cluster_alpha_span)
        clusters.append(
            ScoutCluster(
                cluster_id=int(cluster_id),
                member_indices=member_indices,
                member_weights=member_weights,
                cluster_mass=cluster_mass,
                cluster_size=int(len(member_indices)),
                cluster_geometry_span=float(cluster_geometry_span),
                cluster_alpha_span=float(cluster_alpha_span),
                cluster_best_score=cluster_best_score,
                archetype=archetype,
            )
        )
        mass_pairs.append((int(cluster_id), cluster_mass))

    ranked_clusters = [cluster_id for cluster_id, _ in sorted(mass_pairs, key=lambda item: (-item[1], item[0]))]
    cluster_rank_lookup = {cluster_id: rank + 1 for rank, cluster_id in enumerate(ranked_clusters)}
    carryover = carryover_indices_from_scout(
        captured_indices,
        captured_scores,
        captured_cluster_ids,
        cluster_rank_lookup,
        FINAL_CARRYOVER_BUDGET,
    )
    clusters.sort(key=lambda item: (cluster_rank_lookup[item.cluster_id], item.cluster_best_score))
    return clusters, carryover


def build_informed_params(
    scout_context: BankContext,
    clusters: list[ScoutCluster],
    carryover_indices: list[int],
    rng: np.random.Generator,
) -> list[tuple[float, float, float, float, float, float]]:
    final_params = [scout_context.params_list[int(idx)] for idx in carryover_indices]
    allocation = allocate_cluster_counts(clusters, FINAL_LOCAL_EXPANSION_BUDGET)
    cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
    for cluster_id, count in allocation.items():
        cluster = cluster_lookup[int(cluster_id)]
        for _ in range(int(count)):
            final_params.append(sample_local_candidate(cluster, scout_context, rng))
    while len(final_params) < FINAL_BANK_SIZE - FINAL_EXPLORATION_BUDGET:
        cluster = clusters[0]
        final_params.append(sample_local_candidate(cluster, scout_context, rng))
    for _ in range(FINAL_EXPLORATION_BUDGET):
        final_params.append(sample_anisotropic_parameters(rng))
    return final_params[:FINAL_BANK_SIZE]


def build_scout_random_fill_params(
    scout_context: BankContext,
    carryover_indices: list[int],
    rng: np.random.Generator,
) -> list[tuple[float, float, float, float, float, float]]:
    final_params = [scout_context.params_list[int(idx)] for idx in carryover_indices]
    while len(final_params) < FINAL_BANK_SIZE:
        final_params.append(sample_anisotropic_parameters(rng))
    return final_params[:FINAL_BANK_SIZE]


def evaluate_bank_context(
    method: str,
    split: str,
    observation_seed: int,
    bank_seed_value: int,
    condition: str,
    skew_bin: str,
    true_params: tuple[float, float, float, float, float, float],
    true_geometry: np.ndarray,
    true_shift: int,
    observed_signature: np.ndarray,
    mask: np.ndarray,
    temperature: float,
    band: float,
    geometry_ranges: np.ndarray,
    context: BankContext,
) -> MethodBankRow:
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

    return MethodBankRow(
        method=method,
        split=split,
        observation_seed=int(observation_seed),
        bank_seed=int(bank_seed_value),
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
        best_alpha=float(math.exp(float(context.alpha_logs[best_idx]))),
        consensus_rho12=float(consensus_geometry[0]),
        consensus_rho13=float(consensus_geometry[1]),
        consensus_rho23=float(consensus_geometry[2]),
    )


def summarize_method_trial(
    method: str,
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    true_params: tuple[float, float, float, float, float, float],
    true_shift: int,
    geometry_ranges: np.ndarray,
    bank_rows: list[MethodBankRow],
) -> MethodTrialRow:
    candidate_counts = [row.candidate_count for row in bank_rows]
    alpha_log_spans = [row.alpha_log_span_set for row in bank_rows]
    geometry_set_spans = [row.geometry_span_norm_set for row in bank_rows]
    ambiguity_ratios = [row.ambiguity_ratio for row in bank_rows]
    consensus_geometries = np.array(
        [[row.consensus_rho12, row.consensus_rho13, row.consensus_rho23] for row in bank_rows],
        dtype=float,
    )
    best_geometry_errors = [row.best_geometry_mae for row in bank_rows]
    consensus_geometry_errors = [row.consensus_geometry_mae for row in bank_rows]
    best_alpha_logs = np.array([math.log(max(row.best_alpha, NUMERIC_EPS)) for row in bank_rows], dtype=float)

    consensus_geometry_bank_span_norm = geometry_span_norm(consensus_geometries, geometry_ranges)
    compression_load = float(np.mean(geometry_set_spans) / max(consensus_geometry_bank_span_norm, NUMERIC_EPS))

    return MethodTrialRow(
        method=method,
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
        alpha_bank_log_span=float(np.max(best_alpha_logs) - np.min(best_alpha_logs)),
        consensus_geometry_bank_span_norm=float(consensus_geometry_bank_span_norm),
        best_geometry_mae_mean=float(np.mean(best_geometry_errors)),
        consensus_geometry_mae_mean=float(np.mean(consensus_geometry_errors)),
        geometry_mae_gain=float(np.mean(best_geometry_errors) - np.mean(consensus_geometry_errors)),
        compression_load=float(compression_load),
    )


def aggregate_trial_rows(
    trial_rows: list[MethodTrialRow],
    method: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    split_summary: list[dict[str, object]] = []
    condition_summary: list[dict[str, object]] = []
    method_rows = [row for row in trial_rows if row.method == method]
    for split in BLOCK_SPECS:
        split_subset = [row for row in method_rows if row.split == split]
        split_summary.append(
            {
                "method": method,
                "split": split,
                "count": len(split_subset),
                "mean_candidate_count": float(np.mean([row.mean_candidate_count for row in split_subset])),
                "mean_geometry_span_norm_set": float(np.mean([row.mean_geometry_span_norm_set for row in split_subset])),
                "mean_compression_load": float(np.mean([row.compression_load for row in split_subset])),
                "mean_consensus_geometry_mae": float(np.mean([row.consensus_geometry_mae_mean for row in split_subset])),
                "mean_consensus_geometry_bank_span_norm": float(np.mean([row.consensus_geometry_bank_span_norm for row in split_subset])),
            }
        )
        for condition in FOCUS_CONDITIONS:
            subset = [row for row in split_subset if row.condition == condition]
            condition_summary.append(
                {
                    "method": method,
                    "split": split,
                    "condition": condition,
                    "count": len(subset),
                    "mean_candidate_count": float(np.mean([row.mean_candidate_count for row in subset])),
                    "mean_geometry_span_norm_set": float(np.mean([row.mean_geometry_span_norm_set for row in subset])),
                    "mean_compression_load": float(np.mean([row.compression_load for row in subset])),
                    "mean_consensus_geometry_mae": float(np.mean([row.consensus_geometry_mae_mean for row in subset])),
                    "mean_consensus_geometry_bank_span_norm": float(np.mean([row.consensus_geometry_bank_span_norm for row in subset])),
                }
            )
    return split_summary, condition_summary


def find_condition_value(
    condition_summary: list[dict[str, object]],
    method: str,
    split: str,
    condition: str,
    key: str,
) -> float:
    for row in condition_summary:
        if row["method"] == method and row["split"] == split and row["condition"] == condition:
            return float(row[key])
    raise KeyError((method, split, condition, key))


def plot_condition_metric(
    path: str,
    condition_summary: list[dict[str, object]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    splits = list(BLOCK_SPECS.keys())
    conditions = list(FOCUS_CONDITIONS)
    x = np.arange(len(splits))
    width = 0.24
    colors = {
        METHOD_RANDOM: "#6c757d",
        METHOD_SCOUT_RANDOM: "#2a9d8f",
        METHOD_INFORMED: "#e76f51",
    }

    fig, axes = plt.subplots(1, len(conditions), figsize=(12.6, 4.2), constrained_layout=False)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.98, wspace=0.20)
    if len(conditions) == 1:
        axes = [axes]

    for axis, condition in zip(axes, conditions):
        for method_index, method in enumerate(METHOD_ORDER):
            values = [
                find_condition_value(condition_summary, method, split, condition, metric_key)
                for split in splits
            ]
            axis.bar(x + (method_index - 1) * width, values, width=width, color=colors[method], label=method if condition == conditions[0] else None)
        axis.set_xticks(x)
        axis.set_xticklabels(splits)
        axis.set_title(condition)
        axis.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    geometry_ranges = global_geometry_ranges()

    bank_rows: list[MethodBankRow] = []
    trial_rows: list[MethodTrialRow] = []

    for split, seeds in BLOCK_SPECS.items():
        for observation_seed in seeds:
            for condition in FOCUS_CONDITIONS:
                regime = next(item for item in OBSERVATION_REGIMES if item["name"] == condition)
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    trial_rng = make_trial_rng(observation_seed, condition, skew_bin)
                    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
                    true_geometry, _ = canonicalize_geometry(true_params)
                    clean_signature = anisotropic_forward_signature(true_params)
                    _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
                    temperature = softmin_temperature(regime)
                    band = score_band(regime)

                    method_bank_rows: dict[str, list[MethodBankRow]] = {method: [] for method in METHOD_ORDER}

                    for bank_seed_value in BANK_SEEDS:
                        random_rng = bank_rng(bank_seed_value, observation_seed, condition, skew_bin, stage=0)
                        random_params = sample_random_params_list(RANDOM_BANK_SIZE, random_rng)
                        random_context = build_bank_context_from_params(bank_seed_value, random_params)
                        method_bank_rows[METHOD_RANDOM].append(
                            evaluate_bank_context(
                                METHOD_RANDOM,
                                split,
                                observation_seed,
                                bank_seed_value,
                                condition,
                                skew_bin,
                                true_params,
                                true_geometry,
                                true_shift,
                                observed_signature,
                                mask,
                                temperature,
                                band,
                                geometry_ranges,
                                random_context,
                            )
                        )

                        scout_rng = bank_rng(bank_seed_value, observation_seed, condition, skew_bin, stage=1)
                        scout_params = sample_random_params_list(SCOUT_BANK_SIZE, scout_rng)
                        scout_context = build_bank_context_from_params(bank_seed_value, scout_params)
                        clusters, carryover_indices = scout_clusters_from_context(
                            scout_context,
                            observed_signature,
                            mask,
                            band,
                            temperature,
                            geometry_ranges,
                        )

                        scout_random_rng = bank_rng(bank_seed_value, observation_seed, condition, skew_bin, stage=2)
                        scout_random_params = build_scout_random_fill_params(
                            scout_context,
                            carryover_indices,
                            scout_random_rng,
                        )
                        scout_random_context = build_bank_context_from_params(bank_seed_value, scout_random_params)
                        method_bank_rows[METHOD_SCOUT_RANDOM].append(
                            evaluate_bank_context(
                                METHOD_SCOUT_RANDOM,
                                split,
                                observation_seed,
                                bank_seed_value,
                                condition,
                                skew_bin,
                                true_params,
                                true_geometry,
                                true_shift,
                                observed_signature,
                                mask,
                                temperature,
                                band,
                                geometry_ranges,
                                scout_random_context,
                            )
                        )

                        informed_rng = bank_rng(bank_seed_value, observation_seed, condition, skew_bin, stage=3)
                        informed_params = build_informed_params(
                            scout_context,
                            clusters,
                            carryover_indices,
                            informed_rng,
                        )
                        informed_context = build_bank_context_from_params(bank_seed_value, informed_params)
                        method_bank_rows[METHOD_INFORMED].append(
                            evaluate_bank_context(
                                METHOD_INFORMED,
                                split,
                                observation_seed,
                                bank_seed_value,
                                condition,
                                skew_bin,
                                true_params,
                                true_geometry,
                                true_shift,
                                observed_signature,
                                mask,
                                temperature,
                                band,
                                geometry_ranges,
                                informed_context,
                            )
                        )

                    for method in METHOD_ORDER:
                        bank_rows.extend(method_bank_rows[method])
                        trial_rows.append(
                            summarize_method_trial(
                                method,
                                split,
                                observation_seed,
                                condition,
                                skew_bin,
                                true_params,
                                true_shift,
                                geometry_ranges,
                                method_bank_rows[method],
                            )
                        )

    bank_rows_dicts = [asdict(row) for row in bank_rows]
    trial_rows_dicts = [asdict(row) for row in trial_rows]
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_informed_bank_bank_rows.csv"), bank_rows_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_informed_bank_trial_summary.csv"), trial_rows_dicts)

    split_summary: list[dict[str, object]] = []
    condition_summary: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_split_summary, method_condition_summary = aggregate_trial_rows(trial_rows, method)
        split_summary.extend(method_split_summary)
        condition_summary.extend(method_condition_summary)

    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_informed_bank_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_informed_bank_condition_summary.csv"), condition_summary)

    hard_branch_reduction = {}
    hard_branch_geometry_ratio = {}
    for split in ("holdout", "confirmation"):
        baseline_load = find_condition_value(
            condition_summary,
            METHOD_RANDOM,
            split,
            "sparse_partial_high_noise",
            "mean_compression_load",
        )
        informed_load = find_condition_value(
            condition_summary,
            METHOD_INFORMED,
            split,
            "sparse_partial_high_noise",
            "mean_compression_load",
        )
        baseline_geometry = find_condition_value(
            condition_summary,
            METHOD_RANDOM,
            split,
            "sparse_partial_high_noise",
            "mean_consensus_geometry_mae",
        )
        informed_geometry = find_condition_value(
            condition_summary,
            METHOD_INFORMED,
            split,
            "sparse_partial_high_noise",
            "mean_consensus_geometry_mae",
        )
        hard_branch_reduction[split] = float(1.0 - informed_load / max(baseline_load, NUMERIC_EPS))
        hard_branch_geometry_ratio[split] = float(informed_geometry / max(baseline_geometry, NUMERIC_EPS))

    summary = {
        "random_bank_size": RANDOM_BANK_SIZE,
        "scout_bank_size": SCOUT_BANK_SIZE,
        "final_bank_size": FINAL_BANK_SIZE,
        "final_carryover_budget": FINAL_CARRYOVER_BUDGET,
        "final_local_expansion_budget": FINAL_LOCAL_EXPANSION_BUDGET,
        "final_exploration_budget": FINAL_EXPLORATION_BUDGET,
        "methods": list(METHOD_ORDER),
        "trial_count_per_method": len(trial_rows_dicts) // len(METHOD_ORDER),
        "bank_row_count": len(bank_rows_dicts),
        "hard_branch_compression_reduction_vs_random": hard_branch_reduction,
        "hard_branch_geometry_mae_ratio_vs_random": hard_branch_geometry_ratio,
        "meets_20pct_hard_branch_target": bool(
            hard_branch_reduction["holdout"] >= 0.20 and hard_branch_reduction["confirmation"] >= 0.20
        ),
        "preserves_hard_branch_geometry_within_5pct": bool(
            hard_branch_geometry_ratio["holdout"] <= 1.05 and hard_branch_geometry_ratio["confirmation"] <= 1.05
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "persistent_mode_informed_bank_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_condition_metric(
        os.path.join(FIGURE_DIR, "persistent_mode_informed_bank_compression_load.png"),
        condition_summary,
        "mean_compression_load",
        "Compression load by method and condition",
        "mean compression load",
    )
    plot_condition_metric(
        os.path.join(FIGURE_DIR, "persistent_mode_informed_bank_geometry_mae.png"),
        condition_summary,
        "mean_consensus_geometry_mae",
        "Backbone geometry MAE by method and condition",
        "mean consensus geometry MAE",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
