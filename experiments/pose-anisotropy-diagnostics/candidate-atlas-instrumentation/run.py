"""
Candidate-atlas instrumentation for the pose-free anisotropic solver challenge.

This diagnostic captures the internal near-best candidate family around the
validated Layer 1 and Layer 2 stack so the repo can inspect:

- which candidates survive scoring
- how they cluster in geometry space
- which candidates pull toward or away from the Layer 1 backbone
- which families carry high compression load
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

GEOMETRY_SKEW_BIN_LABELS, sample_conditioned_parameters = load_symbols(
    "run_candidate_conditioned_alignment_experiment_for_candidate_atlas",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "GEOMETRY_SKEW_BIN_LABELS",
    "sample_conditioned_parameters",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment_for_candidate_atlas",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

ALPHA_MAX, ALPHA_MIN, anisotropic_forward_signature, build_reference_bank, control_invariants = load_symbols(
    "run_weighted_anisotropic_inverse_experiment_for_candidate_atlas",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "control_invariants",
)

OBSERVATION_REGIMES, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment_for_candidate_atlas",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "write_csv",
)

FOCUS_ALPHA_BIN, FOCUS_CONDITIONS, marginalized_bank_scores, softmin_temperature = load_symbols(
    "run_joint_pose_marginalized_solver_experiment_for_candidate_atlas",
    ROOT / "experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py",
    "FOCUS_ALPHA_BIN",
    "FOCUS_CONDITIONS",
    "marginalized_bank_scores",
    "softmin_temperature",
)

import json
import math
import os
from dataclasses import dataclass

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
FRONTIER_CAPTURE_COUNT = 16
GEOMETRY_ANCHOR_SCALE = 0.10
GEOMETRY_CLUSTER_THRESHOLD = 0.08
NEIGHBOR_COUNT = 3
NUMERIC_EPS = 1.0e-9
ALPHA_LOG_RANGE = float(math.log(ALPHA_MAX) - math.log(ALPHA_MIN))


@dataclass(frozen=True)
class BankContext:
    seed: int
    params_list: list[tuple[float, float, float, float, float, float]]
    shifted_bank: np.ndarray
    geometries: np.ndarray
    weights: np.ndarray
    alpha_logs: np.ndarray


def condition_index(condition: str) -> int:
    return FOCUS_CONDITIONS.index(condition)


def skew_index(skew_bin: str) -> int:
    return GEOMETRY_SKEW_BIN_LABELS.index(skew_bin)


def make_trial_rng(observation_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(observation_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


def canonicalize_control(
    params: tuple[float, float, float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, float]:
    geometry, weights, alpha = control_invariants(params)
    swapped_geometry = np.array([geometry[0], geometry[2], geometry[1]])
    swapped_weights = np.array([weights[1], weights[0], weights[2]])
    if tuple(swapped_geometry) < tuple(geometry):
        return swapped_geometry, swapped_weights, float(alpha)
    return geometry, weights, float(alpha)


def build_bank_context(bank_seed: int) -> BankContext:
    bank_rng = np.random.default_rng(bank_seed)
    params_list, bank_signatures = build_reference_bank(REFERENCE_BANK_SIZE, bank_rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)
    geometries = []
    weights = []
    alpha_logs = []
    for params in params_list:
        geometry, canonical_weights, alpha = canonicalize_control(params)
        geometries.append(geometry)
        weights.append(canonical_weights)
        alpha_logs.append(math.log(alpha))
    return BankContext(
        seed=bank_seed,
        params_list=params_list,
        shifted_bank=shifted_bank,
        geometries=np.array(geometries),
        weights=np.array(weights),
        alpha_logs=np.array(alpha_logs),
    )


def empirical_geometry_ranges(bank_contexts: list[BankContext]) -> np.ndarray:
    all_geometries = np.concatenate([context.geometries for context in bank_contexts], axis=0)
    span = np.max(all_geometries, axis=0) - np.min(all_geometries, axis=0)
    return np.maximum(span, NUMERIC_EPS)


def score_band(regime: dict[str, float | str | int]) -> float:
    return max(float(regime["noise_sigma"]) ** 2, MIN_SCORE_BAND)


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


def geometry_span_norm(geometries: np.ndarray, geometry_ranges: np.ndarray) -> float:
    if len(geometries) <= 1:
        return 0.0
    span = (np.max(geometries, axis=0) - np.min(geometries, axis=0)) / geometry_ranges
    return float(np.mean(span))


def effective_count(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights * weights))


def weighted_entropy(weights: np.ndarray) -> float:
    weights = normalize_weights(weights)
    count = len(weights)
    if count <= 1:
        return 0.0
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1.0e-12)))
    return float(entropy / math.log(count))


def capture_candidate_indices(scores: np.ndarray, band: float) -> tuple[np.ndarray, set[int], set[int], np.ndarray]:
    order = np.argsort(scores)
    best_score = float(scores[order[0]])
    band_indices = order[scores[order] <= best_score + band]
    if len(band_indices) == 0:
        band_indices = order[:1]
    band_set = {int(idx) for idx in band_indices}
    frontier = [int(idx) for idx in order if int(idx) not in band_set][:FRONTIER_CAPTURE_COUNT]
    captured = np.array(list(map(int, band_indices)) + frontier, dtype=int)
    frontier_set = set(frontier)
    return captured, band_set, frontier_set, order


def anchored_alpha_posterior(
    candidate_scores: np.ndarray,
    candidate_geometries: np.ndarray,
    candidate_alpha_logs: np.ndarray,
    geometry_anchor: np.ndarray,
    geometry_ranges: np.ndarray,
    score_scale: float,
) -> tuple[np.ndarray, float, float, float]:
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
    return anchored_weights, anchored_mean_log, anchored_std_log, effective_count(anchored_weights)


def compute_shift_gap(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_subset: np.ndarray,
) -> np.ndarray:
    masked_subset = shifted_subset[:, :, mask]
    residual = masked_subset - observed_signature[mask][None, None, :]
    mse = np.mean(residual * residual, axis=2)
    if mse.shape[1] <= 1:
        return np.zeros(mse.shape[0], dtype=float)
    partitioned = np.partition(mse, 1, axis=1)
    return partitioned[:, 1] - partitioned[:, 0]


def normalized_geometry_distances(geometries: np.ndarray, geometry_ranges: np.ndarray) -> np.ndarray:
    normed = geometries / geometry_ranges[None, :]
    diffs = np.abs(normed[:, None, :] - normed[None, :, :])
    return np.mean(diffs, axis=2)


def normalized_joint_distances(
    geometries: np.ndarray,
    alpha_logs: np.ndarray,
    geometry_ranges: np.ndarray,
) -> np.ndarray:
    geom_part = geometries / geometry_ranges[None, :]
    alpha_part = (alpha_logs[:, None] - np.min(alpha_logs)) / max(ALPHA_LOG_RANGE, NUMERIC_EPS)
    features = np.concatenate([geom_part, alpha_part], axis=1)
    diffs = np.abs(features[:, None, :] - features[None, :, :])
    return np.mean(diffs, axis=2)


def local_density_from_distance_matrix(distance_matrix: np.ndarray, neighbor_count: int) -> np.ndarray:
    if len(distance_matrix) <= 1:
        return np.ones(len(distance_matrix), dtype=float)
    sorted_distances = np.sort(distance_matrix + np.eye(len(distance_matrix)) * 1.0e9, axis=1)
    k = min(max(neighbor_count, 1), len(distance_matrix) - 1)
    mean_neighbor_distance = np.mean(sorted_distances[:, :k], axis=1)
    return 1.0 / np.maximum(mean_neighbor_distance, NUMERIC_EPS)


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
                centers[cluster_id] = (centers[cluster_id] * center_weights[cluster_id] + point * ordering_weights[idx]) / max(total, NUMERIC_EPS)
                center_weights[cluster_id] = total
                assigned = True
                break
        if not assigned:
            cluster_id = len(centers)
            centers.append(point.copy())
            center_weights.append(float(ordering_weights[idx]))
            assignments[idx] = cluster_id
    return assignments


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def distance_to_geometry_anchor(candidate_geometry: np.ndarray, anchor: np.ndarray, geometry_ranges: np.ndarray) -> float:
    return float(np.mean(np.abs(candidate_geometry - anchor) / geometry_ranges))


def geometry_pull_projection(
    candidate_geometry: np.ndarray,
    best_geometry: np.ndarray,
    anchor_geometry: np.ndarray,
    geometry_ranges: np.ndarray,
) -> tuple[float, float]:
    direction = (anchor_geometry - best_geometry) / geometry_ranges
    offset = (candidate_geometry - best_geometry) / geometry_ranges
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= NUMERIC_EPS:
        return 0.0, 0.0
    projection = float(np.dot(offset, direction) / direction_norm)
    return max(projection, 0.0), max(-projection, 0.0)


def alpha_pull_projection(
    candidate_alpha_log: float,
    best_alpha_log: float,
    anchored_mean_log: float,
) -> tuple[float, float]:
    direction = anchored_mean_log - best_alpha_log
    if abs(direction) <= NUMERIC_EPS:
        return 0.0, 0.0
    projection = float((candidate_alpha_log - best_alpha_log) * math.copysign(1.0, direction))
    return max(projection, 0.0), max(-projection, 0.0)


def mode_persistence_rate(cluster_rows: list[dict[str, object]], geometry_ranges: np.ndarray) -> float:
    if not cluster_rows:
        return float("nan")
    centers = np.array(
        [
            [
                float(row["cluster_center_rho12"]),
                float(row["cluster_center_rho13"]),
                float(row["cluster_center_rho23"]),
            ]
            for row in cluster_rows
        ],
        dtype=float,
    )
    if len(centers) == 1:
        return 1.0
    ordering_weights = np.array([float(row["cluster_mass_layer1"]) for row in cluster_rows], dtype=float)
    assignments = greedy_geometry_clusters(centers, geometry_ranges, ordering_weights)
    total_mass = float(np.sum(ordering_weights))
    if total_mass <= 0.0:
        return float("nan")
    persistent_mass = 0.0
    for cluster_id in sorted(set(int(x) for x in assignments)):
        member_indices = np.where(assignments == cluster_id)[0]
        bank_coverage = len({int(cluster_rows[idx]["bank_seed"]) for idx in member_indices})
        cluster_mass = float(np.sum(ordering_weights[member_indices]))
        if bank_coverage >= 3:
            persistent_mass += cluster_mass
    return float(persistent_mass / total_mass)


def evaluate_trial(
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    regime: dict[str, float | str | int],
    bank_contexts: list[BankContext],
    geometry_ranges: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    trial_rng = make_trial_rng(observation_seed, condition, skew_bin)
    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
    true_geometry, _, true_alpha = canonicalize_control(true_params)
    clean_signature = anisotropic_forward_signature(true_params)
    _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
    temperature = softmin_temperature(regime)
    band = score_band(regime)

    candidate_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []

    band_candidate_counts: list[int] = []
    frontier_candidate_counts: list[int] = []
    cluster_counts: list[int] = []
    cluster_entropies: list[float] = []
    geometry_set_spans: list[float] = []
    alpha_log_spans: list[float] = []
    ambiguity_ratios: list[float] = []
    consensus_geometries: list[np.ndarray] = []
    poison_pulls: list[float] = []
    supportive_pulls: list[float] = []

    for context in bank_contexts:
        scores, best_shifts = marginalized_bank_scores(observed_signature, mask, context.shifted_bank, temperature)
        captured_indices, band_set, frontier_set, order = capture_candidate_indices(scores, band)
        rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(order)}

        band_indices = np.array(sorted(band_set, key=lambda idx: rank_lookup[idx]), dtype=int)
        frontier_indices = np.array(sorted(frontier_set, key=lambda idx: rank_lookup[idx]), dtype=int)

        band_scores = scores[band_indices]
        band_geometries = context.geometries[band_indices]
        band_alpha_logs = context.alpha_logs[band_indices]

        geometry_weights = score_weights(band_scores, band)
        geometry_consensus = np.sum(band_geometries * geometry_weights[:, None], axis=0)
        anchored_weights, anchored_mean_log, anchored_std_log, anchored_effective_count = anchored_alpha_posterior(
            band_scores,
            band_geometries,
            band_alpha_logs,
            geometry_consensus,
            geometry_ranges,
            band,
        )

        geometry_span = geometry_span_norm(band_geometries, geometry_ranges)
        alpha_log_span = float(np.max(band_alpha_logs) - np.min(band_alpha_logs))
        ambiguity_ratio = float(alpha_log_span / max(geometry_span, NUMERIC_EPS))

        captured_scores = scores[captured_indices]
        captured_best_shifts = best_shifts[captured_indices]
        captured_shift_gaps = compute_shift_gap(observed_signature, mask, context.shifted_bank[captured_indices])
        captured_geometries = context.geometries[captured_indices]
        captured_weights = context.weights[captured_indices]
        captured_alpha_logs = context.alpha_logs[captured_indices]
        captured_ordering_weights = score_weights(captured_scores, band)
        captured_cluster_ids = greedy_geometry_clusters(captured_geometries, geometry_ranges, captured_ordering_weights)
        geometry_distance_matrix = normalized_geometry_distances(captured_geometries, geometry_ranges)
        joint_distance_matrix = normalized_joint_distances(captured_geometries, captured_alpha_logs, geometry_ranges)
        geometry_density = local_density_from_distance_matrix(geometry_distance_matrix, NEIGHBOR_COUNT)
        joint_density = local_density_from_distance_matrix(joint_distance_matrix, NEIGHBOR_COUNT)

        band_consensus_weight_lookup = {int(idx): float(weight) for idx, weight in zip(band_indices, geometry_weights)}
        band_anchored_weight_lookup = {int(idx): float(weight) for idx, weight in zip(band_indices, anchored_weights)}
        best_idx = int(order[0])
        best_geometry = context.geometries[best_idx]
        best_alpha_log = float(context.alpha_logs[best_idx])

        band_distance_to_consensus = np.array(
            [distance_to_geometry_anchor(context.geometries[idx], geometry_consensus, geometry_ranges) for idx in band_indices],
            dtype=float,
        )
        poison_distance_threshold = float(np.median(band_distance_to_consensus)) if len(band_distance_to_consensus) > 0 else 0.0

        cluster_member_rows: list[dict[str, object]] = []

        for local_idx, candidate_idx in enumerate(captured_indices):
            params = context.params_list[int(candidate_idx)]
            rho, t, h, w1, w2, alpha = [float(x) for x in params]
            w3 = float(1.0 - w1 - w2)
            candidate_geometry = captured_geometries[local_idx]
            candidate_weight_vector = captured_weights[local_idx]
            candidate_alpha_log = float(captured_alpha_logs[local_idx])

            dist_best_geometry = distance_to_geometry_anchor(candidate_geometry, best_geometry, geometry_ranges)
            dist_consensus_geometry = distance_to_geometry_anchor(candidate_geometry, geometry_consensus, geometry_ranges)
            dist_anchored_alpha = float(abs(candidate_alpha_log - anchored_mean_log) / max(ALPHA_LOG_RANGE, NUMERIC_EPS))

            pull_toward_consensus, pull_away_from_consensus = geometry_pull_projection(
                candidate_geometry,
                best_geometry,
                geometry_consensus,
                geometry_ranges,
            )
            pull_toward_anchored, pull_away_from_anchored = alpha_pull_projection(
                candidate_alpha_log,
                best_alpha_log,
                anchored_mean_log,
            )

            consensus_weight_layer1 = float(band_consensus_weight_lookup.get(int(candidate_idx), 0.0))
            anchored_weight_layer2 = float(band_anchored_weight_lookup.get(int(candidate_idx), 0.0))
            poison_candidate_flag = int(
                consensus_weight_layer1 >= 0.05
                and dist_consensus_geometry >= poison_distance_threshold
                and pull_away_from_consensus > 0.0
            )

            row = {
                "split": split,
                "observation_seed": int(observation_seed),
                "condition": condition,
                "geometry_skew_bin": skew_bin,
                "bank_seed": int(context.seed),
                "candidate_index": int(candidate_idx),
                "capture_tier": "band" if int(candidate_idx) in band_set else "frontier",
                "rank_by_score": int(rank_lookup[int(candidate_idx)]),
                "marginalized_score": float(captured_scores[local_idx]),
                "score_gap_from_best": float(captured_scores[local_idx] - scores[best_idx]),
                "best_shift": int(captured_best_shifts[local_idx]),
                "best_shift_score_gap": float(captured_shift_gaps[local_idx]),
                "rho": rho,
                "t": t,
                "h": h,
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "alpha": float(math.exp(candidate_alpha_log)),
                "log_alpha": float(candidate_alpha_log),
                "rho12": float(candidate_geometry[0]),
                "rho13": float(candidate_geometry[1]),
                "rho23": float(candidate_geometry[2]),
                "geometry_vector_norm": float(np.linalg.norm(candidate_geometry)),
                "weight_entropy": float(-np.sum(candidate_weight_vector * np.log(np.maximum(candidate_weight_vector, 1.0e-12))) / math.log(3.0)),
                "in_band_flag": int(int(candidate_idx) in band_set),
                "consensus_weight_layer1": consensus_weight_layer1,
                "anchored_weight_layer2": anchored_weight_layer2,
                "distance_to_best_geometry": float(dist_best_geometry),
                "distance_to_consensus_geometry": float(dist_consensus_geometry),
                "distance_to_backbone_geometry": float(dist_consensus_geometry),
                "distance_to_anchored_alpha": float(dist_anchored_alpha),
                "nearest_neighbor_geometry_distance": float(np.partition(geometry_distance_matrix[local_idx], 1)[1]) if len(captured_indices) > 1 else 0.0,
                "nearest_neighbor_alpha_distance": float(np.partition(joint_distance_matrix[local_idx], 1)[1]) if len(captured_indices) > 1 else 0.0,
                "local_density_geometry": float(geometry_density[local_idx]),
                "local_density_joint": float(joint_density[local_idx]),
                "cluster_id": int(captured_cluster_ids[local_idx]),
                "pull_toward_consensus": float(pull_toward_consensus),
                "pull_away_from_consensus": float(pull_away_from_consensus),
                "pull_toward_anchored_alpha": float(pull_toward_anchored),
                "pull_away_from_anchored_alpha": float(pull_away_from_anchored),
                "poison_candidate_flag": poison_candidate_flag,
                "true_alpha": float(true_alpha),
                "true_rotation_shift": int(true_shift),
            }
            candidate_rows.append(row)
            cluster_member_rows.append(row)

        cluster_ids = sorted({int(row["cluster_id"]) for row in cluster_member_rows})
        cluster_masses = {
            cluster_id: float(sum(float(row["consensus_weight_layer1"]) for row in cluster_member_rows if int(row["cluster_id"]) == cluster_id))
            for cluster_id in cluster_ids
        }
        ranked_clusters = [cluster_id for cluster_id, _ in sorted(cluster_masses.items(), key=lambda item: (-item[1], item[0]))]
        cluster_rank_lookup = {cluster_id: rank + 1 for rank, cluster_id in enumerate(ranked_clusters)}

        for row in cluster_member_rows:
            row["cluster_rank_by_mass"] = int(cluster_rank_lookup[int(row["cluster_id"])])

        for cluster_id in cluster_ids:
            members = [row for row in cluster_member_rows if int(row["cluster_id"]) == cluster_id]
            member_geometries = np.array([[float(row["rho12"]), float(row["rho13"]), float(row["rho23"])] for row in members], dtype=float)
            member_alpha_logs = np.array([float(row["log_alpha"]) for row in members], dtype=float)
            cluster_rows.append(
                {
                    "split": split,
                    "observation_seed": int(observation_seed),
                    "condition": condition,
                    "geometry_skew_bin": skew_bin,
                    "bank_seed": int(context.seed),
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(members)),
                    "cluster_mass_layer1": float(sum(float(row["consensus_weight_layer1"]) for row in members)),
                    "cluster_mass_layer2": float(sum(float(row["anchored_weight_layer2"]) for row in members)),
                    "cluster_best_score": float(min(float(row["marginalized_score"]) for row in members)),
                    "cluster_mean_score_gap": float(np.mean([float(row["score_gap_from_best"]) for row in members])),
                    "cluster_center_rho12": float(np.mean(member_geometries[:, 0])),
                    "cluster_center_rho13": float(np.mean(member_geometries[:, 1])),
                    "cluster_center_rho23": float(np.mean(member_geometries[:, 2])),
                    "cluster_center_alpha": float(math.exp(float(np.mean(member_alpha_logs)))),
                    "cluster_geometry_span": float(geometry_span_norm(member_geometries, geometry_ranges)),
                    "cluster_alpha_span": float(np.max(member_alpha_logs) - np.min(member_alpha_logs)) if len(member_alpha_logs) > 1 else 0.0,
                    "cluster_rank_by_mass": int(cluster_rank_lookup[int(cluster_id)]),
                }
            )

        band_candidate_counts.append(int(len(band_indices)))
        frontier_candidate_counts.append(int(len(frontier_indices)))
        cluster_counts.append(int(len(cluster_ids)))
        cluster_entropies.append(
            weighted_entropy(np.array([cluster_masses[cluster_id] for cluster_id in cluster_ids], dtype=float))
            if cluster_ids
            else 0.0
        )
        geometry_set_spans.append(float(geometry_span))
        alpha_log_spans.append(float(alpha_log_span))
        ambiguity_ratios.append(float(ambiguity_ratio))
        consensus_geometries.append(geometry_consensus)
        poison_pulls.append(float(sum(float(row["consensus_weight_layer1"]) * float(row["pull_away_from_consensus"]) for row in cluster_member_rows if int(row["poison_candidate_flag"]) == 1)))
        supportive_pulls.append(float(sum(float(row["consensus_weight_layer1"]) * float(row["pull_toward_consensus"]) for row in cluster_member_rows)))

    consensus_geometry_bank_span_norm = geometry_span_norm(np.array(consensus_geometries), geometry_ranges)
    compression_load = float(np.mean(geometry_set_spans) / max(consensus_geometry_bank_span_norm, NUMERIC_EPS))
    trial_clusters = [
        row
        for row in cluster_rows
        if int(row["observation_seed"]) == int(observation_seed)
        and str(row["split"]) == split
        and str(row["condition"]) == condition
        and str(row["geometry_skew_bin"]) == skew_bin
    ]

    trial_summary = {
        "split": split,
        "observation_seed": int(observation_seed),
        "condition": condition,
        "geometry_skew_bin": skew_bin,
        "mean_band_candidate_count": float(np.mean(band_candidate_counts)),
        "mean_frontier_candidate_count": float(np.mean(frontier_candidate_counts)),
        "mean_cluster_count": float(np.mean(cluster_counts)),
        "mean_cluster_entropy": float(np.mean(cluster_entropies)),
        "mean_geometry_span_norm_set": float(np.mean(geometry_set_spans)),
        "consensus_geometry_bank_span_norm": float(consensus_geometry_bank_span_norm),
        "compression_load": float(compression_load),
        "mean_alpha_log_span_set": float(np.mean(alpha_log_spans)),
        "mean_ambiguity_ratio": float(np.mean(ambiguity_ratios)),
        "mean_poison_pull": float(np.mean(poison_pulls)),
        "mean_supportive_pull": float(np.mean(supportive_pulls)),
        "mode_persistence_rate": float(mode_persistence_rate(trial_clusters, geometry_ranges)),
        "true_alpha": float(true_alpha),
        "true_t": float(true_params[1]),
        "true_rotation_shift": int(true_shift),
    }
    return candidate_rows, cluster_rows, trial_summary


def summarize_by_split(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for split in BLOCK_SPECS:
        subset = [row for row in trial_rows if str(row["split"]) == split]
        if not subset:
            continue
        summary.append(
            {
                "split": split,
                "count": len(subset),
                "mean_band_candidate_count": float(np.mean([float(row["mean_band_candidate_count"]) for row in subset])),
                "mean_frontier_candidate_count": float(np.mean([float(row["mean_frontier_candidate_count"]) for row in subset])),
                "mean_cluster_count": float(np.mean([float(row["mean_cluster_count"]) for row in subset])),
                "mean_cluster_entropy": float(np.mean([float(row["mean_cluster_entropy"]) for row in subset])),
                "mean_compression_load": float(np.mean([float(row["compression_load"]) for row in subset])),
                "mean_mode_persistence_rate": float(np.mean([float(row["mode_persistence_rate"]) for row in subset])),
                "mean_poison_pull": float(np.mean([float(row["mean_poison_pull"]) for row in subset])),
                "mean_supportive_pull": float(np.mean([float(row["mean_supportive_pull"]) for row in subset])),
            }
        )
    return summary


def summarize_by_condition(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            subset = [row for row in trial_rows if str(row["split"]) == split and str(row["condition"]) == condition]
            if not subset:
                continue
            summary.append(
                {
                    "split": split,
                    "condition": condition,
                    "count": len(subset),
                    "mean_band_candidate_count": float(np.mean([float(row["mean_band_candidate_count"]) for row in subset])),
                    "mean_cluster_count": float(np.mean([float(row["mean_cluster_count"]) for row in subset])),
                    "mean_compression_load": float(np.mean([float(row["compression_load"]) for row in subset])),
                    "mean_mode_persistence_rate": float(np.mean([float(row["mode_persistence_rate"]) for row in subset])),
                    "mean_poison_pull": float(np.mean([float(row["mean_poison_pull"]) for row in subset])),
                    "mean_supportive_pull": float(np.mean([float(row["mean_supportive_pull"]) for row in subset])),
                }
            )
    return summary


def plot_compression_load(path: str, condition_summary: list[dict[str, object]]) -> None:
    if not condition_summary:
        return
    labels = [f"{item['split']}\n{item['condition']}" for item in condition_summary]
    values = np.array([float(item["mean_compression_load"]) for item in condition_summary], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(labels)), values, color="#457b9d")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean compression load")
    ax.set_title("Candidate-atlas compression load by split and condition")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_count(path: str, split_summary: list[dict[str, object]]) -> None:
    if not split_summary:
        return
    labels = [str(item["split"]) for item in split_summary]
    values = np.array([float(item["mean_cluster_count"]) for item in split_summary], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(np.arange(len(labels)), values, color="#2a9d8f")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean cluster count")
    ax.set_title("Candidate-atlas geometry clusters per trial")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    bank_contexts = [build_bank_context(seed) for seed in BANK_SEEDS]
    geometry_ranges = empirical_geometry_ranges(bank_contexts)

    candidate_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for split, seeds in BLOCK_SPECS.items():
        for observation_seed in seeds:
            for condition in FOCUS_CONDITIONS:
                regime = next(item for item in OBSERVATION_REGIMES if str(item["name"]) == condition)
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    trial_candidate_rows, trial_cluster_rows, trial_summary = evaluate_trial(
                        split,
                        int(observation_seed),
                        condition,
                        skew_bin,
                        regime,
                        bank_contexts,
                        geometry_ranges,
                    )
                    candidate_rows.extend(trial_candidate_rows)
                    cluster_rows.extend(trial_cluster_rows)
                    trial_rows.append(trial_summary)

    split_summary = summarize_by_split(trial_rows)
    condition_summary = summarize_by_condition(trial_rows)

    write_csv(os.path.join(OUTPUT_DIR, "candidate_atlas_rows.csv"), candidate_rows)
    write_csv(os.path.join(OUTPUT_DIR, "candidate_atlas_cluster_rows.csv"), cluster_rows)
    write_csv(os.path.join(OUTPUT_DIR, "candidate_atlas_trial_summary.csv"), trial_rows)
    write_csv(os.path.join(OUTPUT_DIR, "candidate_atlas_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, "candidate_atlas_condition_summary.csv"), condition_summary)

    summary_payload = {
        "experiment": "candidate-atlas-instrumentation",
        "trial_count": len(trial_rows),
        "candidate_row_count": len(candidate_rows),
        "cluster_row_count": len(cluster_rows),
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "bank_seeds": list(BANK_SEEDS),
        "frontier_capture_count": FRONTIER_CAPTURE_COUNT,
        "geometry_cluster_threshold": GEOMETRY_CLUSTER_THRESHOLD,
        "mean_band_candidate_count": float(np.mean([float(row["mean_band_candidate_count"]) for row in trial_rows])),
        "mean_frontier_candidate_count": float(np.mean([float(row["mean_frontier_candidate_count"]) for row in trial_rows])),
        "mean_cluster_count": float(np.mean([float(row["mean_cluster_count"]) for row in trial_rows])),
        "mean_compression_load": float(np.mean([float(row["compression_load"]) for row in trial_rows])),
        "mean_mode_persistence_rate": float(np.mean([float(row["mode_persistence_rate"]) for row in trial_rows])),
        "compression_load_vs_mode_persistence_corr": safe_corr(
            [float(row["compression_load"]) for row in trial_rows],
            [float(row["mode_persistence_rate"]) for row in trial_rows],
        ),
        "split_summary": split_summary,
        "condition_summary": condition_summary,
    }

    with open(os.path.join(OUTPUT_DIR, "candidate_atlas_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    plot_compression_load(
        os.path.join(FIGURE_DIR, "candidate_atlas_compression_load.png"),
        condition_summary,
    )
    plot_cluster_count(
        os.path.join(FIGURE_DIR, "candidate_atlas_cluster_count.png"),
        split_summary,
    )


if __name__ == "__main__":
    main()
