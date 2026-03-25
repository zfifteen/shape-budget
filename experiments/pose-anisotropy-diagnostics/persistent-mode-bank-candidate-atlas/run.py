"""
Candidate-atlas instrumentation for the persistent-mode informed bank pilot.

This diagnostic mirrors the original candidate-atlas capture, but it runs on
the hard-branch bank comparison between:

- one_shot_random
- persistent_mode_informed

The goal is to see how the candidate family itself changes under the informed
bank, not just whether trial-level compression load improves.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

(
    ALPHA_MAX,
    ALPHA_MIN,
    BLOCK_SPECS,
    BANK_SEEDS,
    RANDOM_BANK_SIZE,
    SCOUT_BANK_SIZE,
    FINAL_BANK_SIZE,
    FINAL_CARRYOVER_BUDGET,
    FINAL_LOCAL_EXPANSION_BUDGET,
    FINAL_EXPLORATION_BUDGET,
    METHOD_RANDOM,
    METHOD_INFORMED,
    TARGET_CONDITIONS,
    GEOMETRY_SKEW_BIN_LABELS,
    sample_conditioned_parameters,
    observe_pose_free_signature,
    anisotropic_forward_signature,
    control_invariants,
    OBSERVATION_REGIMES,
    write_csv,
    FOCUS_ALPHA_BIN,
    marginalized_bank_scores,
    softmin_temperature,
    make_trial_rng,
    bank_rng,
    sample_random_params_list,
    build_bank_context_from_params,
    scout_clusters_from_context,
    allocate_cluster_counts,
    sample_local_candidate,
    score_band,
    geometry_span_norm,
    global_geometry_ranges,
) = load_symbols(
    "run_persistent_mode_informed_bank_candidate_atlas",
    ROOT / "experiments/pose-anisotropy-interventions/persistent-mode-informed-bank/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "BLOCK_SPECS",
    "BANK_SEEDS",
    "RANDOM_BANK_SIZE",
    "SCOUT_BANK_SIZE",
    "FINAL_BANK_SIZE",
    "FINAL_CARRYOVER_BUDGET",
    "FINAL_LOCAL_EXPANSION_BUDGET",
    "FINAL_EXPLORATION_BUDGET",
    "METHOD_RANDOM",
    "METHOD_INFORMED",
    "TARGET_CONDITIONS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "sample_conditioned_parameters",
    "observe_pose_free_signature",
    "anisotropic_forward_signature",
    "control_invariants",
    "OBSERVATION_REGIMES",
    "write_csv",
    "FOCUS_ALPHA_BIN",
    "marginalized_bank_scores",
    "softmin_temperature",
    "make_trial_rng",
    "bank_rng",
    "sample_random_params_list",
    "build_bank_context_from_params",
    "scout_clusters_from_context",
    "allocate_cluster_counts",
    "sample_local_candidate",
    "score_band",
    "geometry_span_norm",
    "global_geometry_ranges",
)

(
    capture_candidate_indices,
    anchored_alpha_posterior,
    compute_shift_gap,
    normalized_geometry_distances,
    normalized_joint_distances,
    local_density_from_distance_matrix,
    greedy_geometry_clusters,
    weighted_entropy,
    distance_to_geometry_anchor,
    geometry_pull_projection,
    alpha_pull_projection,
    mode_persistence_rate,
    score_weights,
    NEIGHBOR_COUNT,
    NUMERIC_EPS,
    ALPHA_LOG_RANGE,
) = load_symbols(
    "run_candidate_atlas_helpers_for_persistent_mode_bank",
    ROOT / "experiments/pose-anisotropy-diagnostics/candidate-atlas-instrumentation/run.py",
    "capture_candidate_indices",
    "anchored_alpha_posterior",
    "compute_shift_gap",
    "normalized_geometry_distances",
    "normalized_joint_distances",
    "local_density_from_distance_matrix",
    "greedy_geometry_clusters",
    "weighted_entropy",
    "distance_to_geometry_anchor",
    "geometry_pull_projection",
    "alpha_pull_projection",
    "mode_persistence_rate",
    "score_weights",
    "NEIGHBOR_COUNT",
    "NUMERIC_EPS",
    "ALPHA_LOG_RANGE",
)

import json
import math
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def condition_index(condition: str) -> int:
    return list(TARGET_CONDITIONS).index(condition)


def skew_index(skew_bin: str) -> int:
    return list(GEOMETRY_SKEW_BIN_LABELS).index(skew_bin)


def canonicalize_control(
    params: tuple[float, float, float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, float]:
    geometry, weights, alpha = control_invariants(params)
    swapped_geometry = np.array([geometry[0], geometry[2], geometry[1]])
    swapped_weights = np.array([weights[1], weights[0], weights[2]])
    if tuple(swapped_geometry) < tuple(geometry):
        return swapped_geometry, swapped_weights, float(alpha)
    return geometry, weights, float(alpha)


def canonical_weight_array(params_list: list[tuple[float, float, float, float, float, float]]) -> np.ndarray:
    rows = []
    for params in params_list:
        _, weights, _ = canonicalize_control(params)
        rows.append(weights)
    return np.array(rows, dtype=float)


def cluster_lookup_for_scout(clusters: list[object]) -> tuple[dict[int, int], dict[int, str], dict[int, int]]:
    member_to_cluster: dict[int, int] = {}
    cluster_to_archetype: dict[int, str] = {}
    cluster_to_rank: dict[int, int] = {}
    for rank, cluster in enumerate(clusters, start=1):
        cluster_to_archetype[int(cluster.cluster_id)] = str(cluster.archetype)
        cluster_to_rank[int(cluster.cluster_id)] = rank
        for member_index in cluster.member_indices:
            member_to_cluster[int(member_index)] = int(cluster.cluster_id)
    return member_to_cluster, cluster_to_archetype, cluster_to_rank


def build_random_method_context(
    bank_seed_value: int,
    observation_seed: int,
    condition: str,
    skew_bin: str,
) -> tuple[object, list[dict[str, object]]]:
    rng = bank_rng(bank_seed_value, observation_seed, condition, skew_bin, stage=0)
    params_list = sample_random_params_list(RANDOM_BANK_SIZE, rng)
    metadata = [
        {
            "candidate_source": "random",
            "source_cluster_id": -1,
            "source_cluster_rank": -1,
            "source_archetype": "random",
        }
        for _ in params_list
    ]
    return build_bank_context_from_params(bank_seed_value, params_list), metadata


def build_informed_method_context(
    bank_seed_value: int,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    observed_signature: np.ndarray,
    mask: np.ndarray,
    band: float,
    temperature: float,
    geometry_ranges: np.ndarray,
) -> tuple[object, list[dict[str, object]]]:
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
    member_to_cluster, cluster_to_archetype, cluster_to_rank = cluster_lookup_for_scout(clusters)

    final_params: list[tuple[float, float, float, float, float, float]] = []
    metadata: list[dict[str, object]] = []

    for idx in carryover_indices:
        cluster_id = int(member_to_cluster.get(int(idx), -1))
        final_params.append(scout_context.params_list[int(idx)])
        metadata.append(
            {
                "candidate_source": "carryover",
                "source_cluster_id": cluster_id,
                "source_cluster_rank": int(cluster_to_rank.get(cluster_id, -1)),
                "source_archetype": str(cluster_to_archetype.get(cluster_id, "unknown")),
            }
        )

    informed_rng = bank_rng(bank_seed_value, observation_seed, condition, skew_bin, stage=2)
    allocation = allocate_cluster_counts(clusters, FINAL_LOCAL_EXPANSION_BUDGET)
    cluster_lookup = {int(cluster.cluster_id): cluster for cluster in clusters}
    for cluster_id, count in allocation.items():
        cluster = cluster_lookup[int(cluster_id)]
        for _ in range(int(count)):
            final_params.append(sample_local_candidate(cluster, scout_context, informed_rng))
            metadata.append(
                {
                    "candidate_source": "local_expansion",
                    "source_cluster_id": int(cluster_id),
                    "source_cluster_rank": int(cluster_to_rank.get(int(cluster_id), -1)),
                    "source_archetype": str(cluster_to_archetype.get(int(cluster_id), "unknown")),
                }
            )

    while len(final_params) < FINAL_BANK_SIZE - FINAL_EXPLORATION_BUDGET:
        cluster = clusters[0]
        final_params.append(sample_local_candidate(cluster, scout_context, informed_rng))
        metadata.append(
            {
                "candidate_source": "local_fill",
                "source_cluster_id": int(cluster.cluster_id),
                "source_cluster_rank": int(cluster_to_rank.get(int(cluster.cluster_id), -1)),
                "source_archetype": str(cluster.archetype),
            }
        )

    for _ in range(FINAL_EXPLORATION_BUDGET):
        final_params.append(sample_random_params_list(1, informed_rng)[0])
        metadata.append(
            {
                "candidate_source": "exploration",
                "source_cluster_id": -1,
                "source_cluster_rank": -1,
                "source_archetype": "exploration",
            }
        )

    final_params = final_params[:FINAL_BANK_SIZE]
    metadata = metadata[:FINAL_BANK_SIZE]
    return build_bank_context_from_params(bank_seed_value, final_params), metadata


def instrument_method_bank(
    method: str,
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    bank_seed_value: int,
    true_params: tuple[float, float, float, float, float, float],
    true_alpha: float,
    true_shift: int,
    observed_signature: np.ndarray,
    mask: np.ndarray,
    temperature: float,
    band: float,
    geometry_ranges: np.ndarray,
    context: object,
    source_metadata: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
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
    captured_alpha_logs = context.alpha_logs[captured_indices]
    captured_weights = canonical_weight_array(context.params_list)[captured_indices]
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

    candidate_rows: list[dict[str, object]] = []
    cluster_member_rows: list[dict[str, object]] = []

    for local_idx, candidate_idx in enumerate(captured_indices):
        params = context.params_list[int(candidate_idx)]
        rho, t, h, w1, w2, _alpha = [float(x) for x in params]
        w3 = float(1.0 - w1 - w2)
        candidate_geometry = captured_geometries[local_idx]
        candidate_weight_vector = captured_weights[local_idx]
        candidate_alpha_log = float(captured_alpha_logs[local_idx])
        metadata = source_metadata[int(candidate_idx)] if int(candidate_idx) < len(source_metadata) else {}

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
            "method": method,
            "split": split,
            "observation_seed": int(observation_seed),
            "condition": condition,
            "geometry_skew_bin": skew_bin,
            "bank_seed": int(bank_seed_value),
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
            "candidate_source": str(metadata.get("candidate_source", "unknown")),
            "source_cluster_id": int(metadata.get("source_cluster_id", -1)),
            "source_cluster_rank": int(metadata.get("source_cluster_rank", -1)),
            "source_archetype": str(metadata.get("source_archetype", "unknown")),
            "true_alpha": float(true_alpha),
            "true_rotation_shift": int(true_shift),
        }
        candidate_rows.append(row)
        cluster_member_rows.append(row)

    cluster_ids = sorted({int(row["cluster_id"]) for row in cluster_member_rows})
    cluster_masses = {
        cluster_id: float(
            sum(float(row["consensus_weight_layer1"]) for row in cluster_member_rows if int(row["cluster_id"]) == cluster_id)
        )
        for cluster_id in cluster_ids
    }
    ranked_clusters = [cluster_id for cluster_id, _ in sorted(cluster_masses.items(), key=lambda item: (-item[1], item[0]))]
    cluster_rank_lookup = {cluster_id: rank + 1 for rank, cluster_id in enumerate(ranked_clusters)}

    for row in cluster_member_rows:
        row["cluster_rank_by_mass"] = int(cluster_rank_lookup[int(row["cluster_id"])])

    cluster_rows: list[dict[str, object]] = []
    for cluster_id in cluster_ids:
        members = [row for row in cluster_member_rows if int(row["cluster_id"]) == cluster_id]
        member_geometries = np.array([[float(row["rho12"]), float(row["rho13"]), float(row["rho23"])] for row in members], dtype=float)
        member_alpha_logs = np.array([float(row["log_alpha"]) for row in members], dtype=float)
        source_labels = [str(row["candidate_source"]) for row in members]
        dominant_source = max(sorted(set(source_labels)), key=source_labels.count)
        cluster_rows.append(
            {
                "method": method,
                "split": split,
                "observation_seed": int(observation_seed),
                "condition": condition,
                "geometry_skew_bin": skew_bin,
                "bank_seed": int(bank_seed_value),
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
                "dominant_candidate_source": dominant_source,
            }
        )

    method_bank_row = {
        "method": method,
        "split": split,
        "observation_seed": int(observation_seed),
        "condition": condition,
        "geometry_skew_bin": skew_bin,
        "bank_seed": int(bank_seed_value),
        "true_alpha": float(true_alpha),
        "true_t": float(true_params[1]),
        "true_rotation_shift": int(true_shift),
        "mean_band_candidate_count": float(len(band_indices)),
        "mean_frontier_candidate_count": float(len(frontier_indices)),
        "mean_cluster_count": float(len(cluster_ids)),
        "mean_cluster_entropy": float(weighted_entropy(np.array([cluster_masses[cluster_id] for cluster_id in cluster_ids], dtype=float)) if cluster_ids else 0.0),
        "mean_geometry_span_norm_set": float(geometry_span),
        "mean_alpha_log_span_set": float(alpha_log_span),
        "mean_ambiguity_ratio": float(ambiguity_ratio),
        "mean_poison_pull": float(sum(float(row["consensus_weight_layer1"]) * float(row["pull_away_from_consensus"]) for row in cluster_member_rows if int(row["poison_candidate_flag"]) == 1)),
        "mean_supportive_pull": float(sum(float(row["consensus_weight_layer1"]) * float(row["pull_toward_consensus"]) for row in cluster_member_rows)),
        "consensus_geometry_rho12": float(geometry_consensus[0]),
        "consensus_geometry_rho13": float(geometry_consensus[1]),
        "consensus_geometry_rho23": float(geometry_consensus[2]),
        "anchored_alpha_log_std": float(anchored_std_log),
        "anchored_alpha_effective_count": float(anchored_effective_count),
    }
    return candidate_rows, cluster_rows, method_bank_row


def method_trial_summary(
    method: str,
    split: str,
    observation_seed: int,
    condition: str,
    skew_bin: str,
    bank_rows: list[dict[str, object]],
    cluster_rows: list[dict[str, object]],
    geometry_ranges: np.ndarray,
    true_alpha: float,
    true_t: float,
    true_shift: int,
) -> dict[str, object]:
    consensus_geometries = np.array(
        [[float(row["consensus_geometry_rho12"]), float(row["consensus_geometry_rho13"]), float(row["consensus_geometry_rho23"])] for row in bank_rows],
        dtype=float,
    )
    consensus_geometry_bank_span_norm = geometry_span_norm(consensus_geometries, geometry_ranges)
    compression_load = float(
        np.mean([float(row["mean_geometry_span_norm_set"]) for row in bank_rows]) / max(consensus_geometry_bank_span_norm, NUMERIC_EPS)
    )
    return {
        "method": method,
        "split": split,
        "observation_seed": int(observation_seed),
        "condition": condition,
        "geometry_skew_bin": skew_bin,
        "true_alpha": float(true_alpha),
        "true_t": float(true_t),
        "true_rotation_shift": int(true_shift),
        "mean_band_candidate_count": float(np.mean([float(row["mean_band_candidate_count"]) for row in bank_rows])),
        "mean_frontier_candidate_count": float(np.mean([float(row["mean_frontier_candidate_count"]) for row in bank_rows])),
        "mean_cluster_count": float(np.mean([float(row["mean_cluster_count"]) for row in bank_rows])),
        "mean_cluster_entropy": float(np.mean([float(row["mean_cluster_entropy"]) for row in bank_rows])),
        "mean_geometry_span_norm_set": float(np.mean([float(row["mean_geometry_span_norm_set"]) for row in bank_rows])),
        "consensus_geometry_bank_span_norm": float(consensus_geometry_bank_span_norm),
        "compression_load": float(compression_load),
        "mean_alpha_log_span_set": float(np.mean([float(row["mean_alpha_log_span_set"]) for row in bank_rows])),
        "mean_ambiguity_ratio": float(np.mean([float(row["mean_ambiguity_ratio"]) for row in bank_rows])),
        "mean_poison_pull": float(np.mean([float(row["mean_poison_pull"]) for row in bank_rows])),
        "mean_supportive_pull": float(np.mean([float(row["mean_supportive_pull"]) for row in bank_rows])),
        "mean_anchored_alpha_log_std": float(np.mean([float(row["anchored_alpha_log_std"]) for row in bank_rows])),
        "mode_persistence_rate": float(mode_persistence_rate(cluster_rows, geometry_ranges)),
    }


def summarize_by_split(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in (METHOD_RANDOM, METHOD_INFORMED):
        for split in BLOCK_SPECS:
            subset = [row for row in trial_rows if str(row["method"]) == method and str(row["split"]) == split]
            if not subset:
                continue
            summary.append(
                {
                    "method": method,
                    "split": split,
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


def summarize_by_cell(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in (METHOD_RANDOM, METHOD_INFORMED):
        for split in BLOCK_SPECS:
            for condition in TARGET_CONDITIONS:
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    subset = [
                        row
                        for row in trial_rows
                        if str(row["method"]) == method
                        and str(row["split"]) == split
                        and str(row["condition"]) == condition
                        and str(row["geometry_skew_bin"]) == skew_bin
                    ]
                    if not subset:
                        continue
                    summary.append(
                        {
                            "method": method,
                            "split": split,
                            "condition": condition,
                            "geometry_skew_bin": skew_bin,
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


def main() -> None:
    geometry_ranges = global_geometry_ranges()

    candidate_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    bank_summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for split, seeds in BLOCK_SPECS.items():
        for observation_seed in seeds:
            for condition in TARGET_CONDITIONS:
                regime = next(item for item in OBSERVATION_REGIMES if item["name"] == condition)
                for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                    trial_rng = make_trial_rng(observation_seed, condition, skew_bin)
                    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
                    _, _, true_alpha = canonicalize_control(true_params)
                    clean_signature = anisotropic_forward_signature(true_params)
                    _, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, regime, trial_rng)
                    temperature = softmin_temperature(regime)
                    band = score_band(regime)

                    trial_bank_rows_by_method: dict[str, list[dict[str, object]]] = {
                        METHOD_RANDOM: [],
                        METHOD_INFORMED: [],
                    }
                    trial_cluster_rows_by_method: dict[str, list[dict[str, object]]] = {
                        METHOD_RANDOM: [],
                        METHOD_INFORMED: [],
                    }

                    for bank_seed_value in BANK_SEEDS:
                        random_context, random_metadata = build_random_method_context(
                            int(bank_seed_value),
                            int(observation_seed),
                            condition,
                            skew_bin,
                        )
                        random_candidate_rows, random_cluster_rows, random_bank_row = instrument_method_bank(
                            METHOD_RANDOM,
                            split,
                            int(observation_seed),
                            condition,
                            skew_bin,
                            int(bank_seed_value),
                            true_params,
                            float(true_alpha),
                            int(true_shift),
                            observed_signature,
                            mask,
                            temperature,
                            band,
                            geometry_ranges,
                            random_context,
                            random_metadata,
                        )
                        candidate_rows.extend(random_candidate_rows)
                        cluster_rows.extend(random_cluster_rows)
                        bank_summary_rows.append(random_bank_row)
                        trial_bank_rows_by_method[METHOD_RANDOM].append(random_bank_row)
                        trial_cluster_rows_by_method[METHOD_RANDOM].extend(random_cluster_rows)

                        informed_context, informed_metadata = build_informed_method_context(
                            int(bank_seed_value),
                            int(observation_seed),
                            condition,
                            skew_bin,
                            observed_signature,
                            mask,
                            band,
                            temperature,
                            geometry_ranges,
                        )
                        informed_candidate_rows, informed_cluster_rows, informed_bank_row = instrument_method_bank(
                            METHOD_INFORMED,
                            split,
                            int(observation_seed),
                            condition,
                            skew_bin,
                            int(bank_seed_value),
                            true_params,
                            float(true_alpha),
                            int(true_shift),
                            observed_signature,
                            mask,
                            temperature,
                            band,
                            geometry_ranges,
                            informed_context,
                            informed_metadata,
                        )
                        candidate_rows.extend(informed_candidate_rows)
                        cluster_rows.extend(informed_cluster_rows)
                        bank_summary_rows.append(informed_bank_row)
                        trial_bank_rows_by_method[METHOD_INFORMED].append(informed_bank_row)
                        trial_cluster_rows_by_method[METHOD_INFORMED].extend(informed_cluster_rows)

                    for method in (METHOD_RANDOM, METHOD_INFORMED):
                        trial_rows.append(
                            method_trial_summary(
                                method,
                                split,
                                int(observation_seed),
                                condition,
                                skew_bin,
                                trial_bank_rows_by_method[method],
                                trial_cluster_rows_by_method[method],
                                geometry_ranges,
                                float(true_alpha),
                                float(true_params[1]),
                                int(true_shift),
                            )
                        )

    split_summary = summarize_by_split(trial_rows)
    cell_summary = summarize_by_cell(trial_rows)

    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_bank_candidate_atlas_rows.csv"), candidate_rows)
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_bank_candidate_atlas_cluster_rows.csv"), cluster_rows)
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_bank_candidate_atlas_bank_summary.csv"), bank_summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_bank_candidate_atlas_trial_summary.csv"), trial_rows)
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_bank_candidate_atlas_split_summary.csv"), split_summary)
    write_csv(os.path.join(OUTPUT_DIR, "persistent_mode_bank_candidate_atlas_cell_summary.csv"), cell_summary)

    summary = {
        "experiment": "persistent-mode-bank-candidate-atlas",
        "methods": [METHOD_RANDOM, METHOD_INFORMED],
        "trial_count": len(trial_rows),
        "candidate_row_count": len(candidate_rows),
        "cluster_row_count": len(cluster_rows),
        "bank_summary_row_count": len(bank_summary_rows),
        "random_bank_size": int(RANDOM_BANK_SIZE),
        "scout_bank_size": int(SCOUT_BANK_SIZE),
        "final_bank_size": int(FINAL_BANK_SIZE),
        "target_conditions": list(TARGET_CONDITIONS),
        "bank_seeds": [int(seed) for seed in BANK_SEEDS],
        "splits": list(BLOCK_SPECS.keys()),
        "mean_band_candidate_count_by_method": {
            method: float(np.mean([float(row["mean_band_candidate_count"]) for row in trial_rows if str(row["method"]) == method]))
            for method in (METHOD_RANDOM, METHOD_INFORMED)
        },
        "mean_compression_load_by_method": {
            method: float(np.mean([float(row["compression_load"]) for row in trial_rows if str(row["method"]) == method]))
            for method in (METHOD_RANDOM, METHOD_INFORMED)
        },
        "mean_mode_persistence_rate_by_method": {
            method: float(np.mean([float(row["mode_persistence_rate"]) for row in trial_rows if str(row["method"]) == method]))
            for method in (METHOD_RANDOM, METHOD_INFORMED)
        },
    }
    with open(os.path.join(OUTPUT_DIR, "persistent_mode_bank_candidate_atlas_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
