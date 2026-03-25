"""
Bank-adaptive solver with cached candidate tables and a frozen ridge chooser.

The workflow is:

1. generate one cache table per explicit block of bank / observation seeds
2. fit exactly one ridge chooser on cached calibration blocks only
3. freeze the chooser before any holdout generation or evaluation
4. validate on the requested holdout and confirmation blocks
5. if the first holdout fails, rerun exactly one density fallback branch
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

GEOMETRY_SKEW_BIN_LABELS, candidate_conditioned_search, sample_conditioned_parameters, top_k_indices = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "GEOMETRY_SKEW_BIN_LABELS",
    "candidate_conditioned_search",
    "sample_conditioned_parameters",
    "top_k_indices",
)

family_switching_refine, = load_symbols(
    "run_family_switching_refinement_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/family-switching-refinement/run.py",
    "family_switching_refine",
)

nearest_neighbor_aligned, rmse = load_symbols(
    "run_orientation_locking_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/orientation-locking/run.py",
    "nearest_neighbor_aligned",
    "rmse",
)

oracle_align_observation, = load_symbols(
    "run_oracle_alignment_ceiling_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/oracle-alignment-ceiling/run.py",
    "oracle_align_observation",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

anisotropic_forward_signature, build_reference_bank, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "symmetry_aware_errors",
)

OBSERVATION_REGIMES, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "write_csv",
)

FOCUS_CONDITIONS, FOCUS_ALPHA_BIN, SolverContext, choose_support_gated_baseline, joint_pose_marginalized_refine, marginalized_bank_scores, softmin_temperature = load_symbols(
    "run_joint_pose_marginalized_solver_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py",
    "FOCUS_CONDITIONS",
    "FOCUS_ALPHA_BIN",
    "SolverContext",
    "choose_support_gated_baseline",
    "joint_pose_marginalized_refine",
    "marginalized_bank_scores",
    "softmin_temperature",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RIDGE_LAMBDA = 1.0
NUMERIC_EPS = 1.0e-12

STRING_FIELDS = {
    "variant_name",
    "block_name",
    "block_role",
    "condition",
    "geometry_skew_bin",
}


@dataclass(frozen=True)
class BlockSpec:
    name: str
    role: str
    bank_seed: int
    obs_seeds: tuple[int, ...]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    reference_bank_size: int
    top_k_seeds: int


BLOCK_SPECS = {
    "calibration_block_1": BlockSpec(
        name="calibration_block_1",
        role="calibration",
        bank_seed=20260324,
        obs_seeds=(20260410, 20260411, 20260412),
    ),
    "calibration_block_2": BlockSpec(
        name="calibration_block_2",
        role="calibration",
        bank_seed=20260326,
        obs_seeds=(20260416, 20260417, 20260418),
    ),
    "holdout_block_1": BlockSpec(
        name="holdout_block_1",
        role="holdout",
        bank_seed=20260327,
        obs_seeds=(20260422, 20260423, 20260424),
    ),
    "confirmation_block": BlockSpec(
        name="confirmation_block",
        role="confirmation",
        bank_seed=20260328,
        obs_seeds=(20260425, 20260426, 20260427),
    ),
}

VARIANT_SPECS = {
    "baseline": VariantSpec(name="baseline", reference_bank_size=300, top_k_seeds=3),
    "density_ablation": VariantSpec(name="density_ablation", reference_bank_size=600, top_k_seeds=5),
}

CELL_NAMES = [f"{condition}__{skew_bin}" for condition in FOCUS_CONDITIONS for skew_bin in GEOMETRY_SKEW_BIN_LABELS]
FEATURE_NAMES = CELL_NAMES + [
    "support_log_alpha",
    "joint_log_alpha",
    "support_abs_t",
    "joint_abs_t",
    "support_score",
    "joint_score",
    "support_entropy",
    "joint_entropy",
    "support_cv_score",
    "joint_cv_score",
]


def cache_path(block_name: str, variant_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{variant_name}__{block_name}.csv")


def cache_meta_path(block_name: str, variant_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{variant_name}__{block_name}.json")


def chooser_model_path(variant_name: str) -> str:
    return os.path.join(MODEL_DIR, f"{variant_name}__frozen_ridge_chooser.json")


def report_json_path(variant_name: str, report_name: str) -> str:
    return os.path.join(REPORT_DIR, f"{variant_name}__{report_name}.json")


def report_csv_path(variant_name: str, report_name: str) -> str:
    return os.path.join(REPORT_DIR, f"{variant_name}__{report_name}.csv")


def condition_index(condition: str) -> int:
    return FOCUS_CONDITIONS.index(condition)


def skew_index(skew_bin: str) -> int:
    return GEOMETRY_SKEW_BIN_LABELS.index(skew_bin)


def cell_name(condition: str, skew_bin: str) -> str:
    return f"{condition}__{skew_bin}"


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def to_float(value: float | str) -> float:
    return float(value)


def numeric_row(raw_row: dict[str, str]) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    for key, value in raw_row.items():
        if key in STRING_FIELDS:
            row[key] = value
        else:
            row[key] = float(value)
    return row


def load_csv_rows(path: str) -> list[dict[str, float | str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [numeric_row(row) for row in csv.DictReader(handle)]


def load_block_rows(block_name: str, variant_name: str) -> list[dict[str, float | str]]:
    return load_csv_rows(cache_path(block_name, variant_name))


def load_rows_for_blocks(block_names: list[str], variant_name: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for block_name in block_names:
        rows.extend(load_block_rows(block_name, variant_name))
    return rows


def make_trial_rng(obs_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(obs_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


def make_cv_rng(obs_seed: int, condition: str, skew_bin: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([7777, int(obs_seed), condition_index(condition), skew_index(skew_bin)])
    return np.random.default_rng(sequence)


def shift_mse_for_mask(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    signature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shift_stack = np.stack([np.roll(signature, shift) for shift in range(len(signature))], axis=0)
    residual = shift_stack[:, mask] - observed_signature[mask][None, :]
    mse = np.mean(residual * residual, axis=1)
    return mse, shift_stack


def cross_validated_shift_score(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    params: tuple[float, float, float, float, float, float],
    rng: np.random.Generator,
) -> float:
    candidate_signature = anisotropic_forward_signature(params)
    observed_indices = np.flatnonzero(mask)
    if len(observed_indices) < 4:
        full_mse, _ = shift_mse_for_mask(observed_signature, mask, candidate_signature)
        return float(np.min(full_mse))

    shuffled = observed_indices[rng.permutation(len(observed_indices))]
    folds = [np.array(fold, dtype=int) for fold in np.array_split(shuffled, 2) if len(fold) > 0]
    cv_scores: list[float] = []

    for fold_idx, validation_idx in enumerate(folds):
        train_idx = np.concatenate([fold for idx, fold in enumerate(folds) if idx != fold_idx], axis=0)
        if len(train_idx) == 0 or len(validation_idx) == 0:
            continue

        train_mask = np.zeros_like(mask, dtype=bool)
        validation_mask = np.zeros_like(mask, dtype=bool)
        train_mask[train_idx] = True
        validation_mask[validation_idx] = True

        train_mse, shift_stack = shift_mse_for_mask(observed_signature, train_mask, candidate_signature)
        best_shift = int(np.argmin(train_mse))
        validation_residual = shift_stack[best_shift, validation_mask] - observed_signature[validation_mask]
        cv_scores.append(float(np.mean(validation_residual * validation_residual)))

    if not cv_scores:
        full_mse, _ = shift_mse_for_mask(observed_signature, mask, candidate_signature)
        return float(np.min(full_mse))
    return float(np.mean(cv_scores))


def support_and_joint_candidates(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    condition: str,
    bank_params: list[tuple[float, float, float, float, float, float]],
    shifted_bank: np.ndarray,
    top_k_seeds: int,
) -> dict[str, object]:
    regime = next(item for item in OBSERVATION_REGIMES if str(item["name"]) == condition)
    temperature = softmin_temperature(regime)

    marginalized_scores, marginalized_best_shifts = marginalized_bank_scores(
        observed_signature,
        mask,
        shifted_bank,
        temperature,
    )
    marginalized_idx = int(np.argmin(marginalized_scores))
    marginalized_params = bank_params[marginalized_idx]
    marginalized_signature = shifted_bank[marginalized_idx, int(marginalized_best_shifts[marginalized_idx])]

    conditioned_best_params = marginalized_params
    conditioned_best_signature = marginalized_signature
    conditioned_best_score = float("inf")

    family_best_params = marginalized_params
    family_best_signature = marginalized_signature
    family_best_score = float("inf")

    joint_best_params = marginalized_params
    joint_best_signature = marginalized_signature
    joint_best_score = float("inf")
    joint_best_entropy = 1.0
    joint_best_seed_rank = 1

    seed_indices = top_k_indices(marginalized_scores, top_k_seeds)
    for seed_rank, idx in enumerate(seed_indices, start=1):
        seed_params = bank_params[idx]

        conditioned_params, conditioned_signature, _, conditioned_score = candidate_conditioned_search(
            observed_signature,
            mask,
            seed_params,
            temperature,
        )
        if conditioned_score < conditioned_best_score:
            conditioned_best_score = float(conditioned_score)
            conditioned_best_params = conditioned_params
            conditioned_best_signature = conditioned_signature

        family_params, family_signature, _, family_score = family_switching_refine(
            observed_signature,
            mask,
            seed_params,
            temperature,
        )
        if family_score < family_best_score:
            family_best_score = float(family_score)
            family_best_params = family_params
            family_best_signature = family_signature

        joint_params, joint_signature, _, joint_score, joint_entropy = joint_pose_marginalized_refine(
            observed_signature,
            mask,
            seed_params,
            temperature,
            condition,
        )
        if joint_score < joint_best_score:
            joint_best_score = float(joint_score)
            joint_best_params = joint_params
            joint_best_signature = joint_signature
            joint_best_entropy = float(joint_entropy)
            joint_best_seed_rank = int(seed_rank)

    support_params, support_signature, support_score, support_choose_family = choose_support_gated_baseline(
        condition,
        conditioned_best_params,
        conditioned_best_signature,
        conditioned_best_score,
        family_best_params,
        family_best_signature,
        family_best_score,
    )

    context = SolverContext(observed_signature, mask)
    support_score_full, support_signature_full, support_shift, support_entropy = context.score_params(support_params, temperature)
    joint_score_full, joint_signature_full, joint_shift, joint_entropy_full = context.score_params(joint_best_params, temperature)

    return {
        "temperature": float(temperature),
        "support": {
            "params": support_params,
            "signature": support_signature_full,
            "score": float(support_score_full if np.isfinite(support_score_full) else support_score),
            "entropy": float(support_entropy),
            "best_shift": int(support_shift),
            "choose_family": int(support_choose_family),
        },
        "joint": {
            "params": joint_best_params,
            "signature": joint_signature_full if joint_best_score < float("inf") else joint_best_signature,
            "score": float(joint_score_full if np.isfinite(joint_score_full) else joint_best_score),
            "entropy": float(joint_entropy_full if np.isfinite(joint_entropy_full) else joint_best_entropy),
            "best_shift": int(joint_shift),
            "seed_rank": int(joint_best_seed_rank),
        },
    }


def observable_feature_row(
    condition: str,
    skew_bin: str,
    support_params: tuple[float, float, float, float, float, float],
    support_score: float,
    support_entropy: float,
    support_cv_score: float,
    joint_params: tuple[float, float, float, float, float, float],
    joint_score: float,
    joint_entropy: float,
    joint_cv_score: float,
) -> dict[str, float | str]:
    support_rho, support_t, support_h, support_w1, support_w2, support_alpha = support_params
    joint_rho, joint_t, joint_h, joint_w1, joint_w2, joint_alpha = joint_params
    return {
        "condition": condition,
        "geometry_skew_bin": skew_bin,
        "support_rho": float(support_rho),
        "support_t": float(support_t),
        "support_h": float(support_h),
        "support_w1": float(support_w1),
        "support_w2": float(support_w2),
        "support_w3": float(1.0 - support_w1 - support_w2),
        "support_alpha": float(support_alpha),
        "support_score": float(support_score),
        "support_entropy": float(support_entropy),
        "support_cv_score": float(support_cv_score),
        "joint_rho": float(joint_rho),
        "joint_t": float(joint_t),
        "joint_h": float(joint_h),
        "joint_w1": float(joint_w1),
        "joint_w2": float(joint_w2),
        "joint_w3": float(1.0 - joint_w1 - joint_w2),
        "joint_alpha": float(joint_alpha),
        "joint_score": float(joint_score),
        "joint_entropy": float(joint_entropy),
        "joint_cv_score": float(joint_cv_score),
    }


def build_trial_row(
    variant: VariantSpec,
    block: BlockSpec,
    obs_seed: int,
    condition: str,
    skew_bin: str,
    bank_params: list[tuple[float, float, float, float, float, float]],
    shifted_bank: np.ndarray,
    bank_signatures: np.ndarray,
) -> dict[str, float | str]:
    trial_rng = make_trial_rng(obs_seed, condition, skew_bin)

    true_params = sample_conditioned_parameters(trial_rng, FOCUS_ALPHA_BIN, skew_bin)
    clean_signature = anisotropic_forward_signature(true_params)
    rotated_signature, observed_signature, mask, true_shift = observe_pose_free_signature(clean_signature, next(item for item in OBSERVATION_REGIMES if str(item["name"]) == condition), trial_rng)

    candidate_data = support_and_joint_candidates(
        observed_signature,
        mask,
        condition,
        bank_params,
        shifted_bank,
        variant.top_k_seeds,
    )

    support = candidate_data["support"]
    joint = candidate_data["joint"]

    support_params = support["params"]
    joint_params = joint["params"]

    support_geometry_error, support_weight_error, support_alpha_error = symmetry_aware_errors(true_params, support_params)
    joint_geometry_error, joint_weight_error, joint_alpha_error = symmetry_aware_errors(true_params, joint_params)

    support_fit_rmse = rmse(support["signature"], rotated_signature)
    joint_fit_rmse = rmse(joint["signature"], rotated_signature)

    support_cv_score = cross_validated_shift_score(
        observed_signature,
        mask,
        support_params,
        make_cv_rng(obs_seed, condition, skew_bin),
    )
    joint_cv_score = cross_validated_shift_score(
        observed_signature,
        mask,
        joint_params,
        make_cv_rng(obs_seed, condition, skew_bin),
    )

    oracle_observed, oracle_mask = oracle_align_observation(observed_signature, mask, true_shift)
    oracle_params, oracle_signature = nearest_neighbor_aligned(
        oracle_observed,
        oracle_mask,
        bank_signatures,
        bank_params,
    )
    oracle_geometry_error, oracle_weight_error, oracle_alpha_error = symmetry_aware_errors(true_params, oracle_params)
    oracle_fit_rmse = rmse(oracle_signature, clean_signature)

    row = {
        "variant_name": variant.name,
        "reference_bank_size": float(variant.reference_bank_size),
        "top_k_seeds": float(variant.top_k_seeds),
        "block_name": block.name,
        "block_role": block.role,
        "bank_seed": float(block.bank_seed),
        "observation_seed": float(obs_seed),
        "condition": condition,
        "geometry_skew_bin": skew_bin,
        "true_rho": float(true_params[0]),
        "true_t": float(true_params[1]),
        "true_h": float(true_params[2]),
        "true_w1": float(true_params[3]),
        "true_w2": float(true_params[4]),
        "true_w3": float(1.0 - true_params[3] - true_params[4]),
        "true_alpha": float(true_params[5]),
        "true_rotation_shift": float(true_shift),
        "observed_count": float(np.sum(mask)),
        "temperature": float(candidate_data["temperature"]),
    }
    row.update(
        observable_feature_row(
            condition,
            skew_bin,
            support_params,
            float(support["score"]),
            float(support["entropy"]),
            float(support_cv_score),
            joint_params,
            float(joint["score"]),
            float(joint["entropy"]),
            float(joint_cv_score),
        )
    )
    row.update(
        {
            "support_alpha_error": float(support_alpha_error),
            "support_geometry_error": float(support_geometry_error),
            "support_weight_error": float(support_weight_error),
            "support_fit_rmse": float(support_fit_rmse),
            "support_choose_family": float(support["choose_family"]),
            "joint_alpha_error": float(joint_alpha_error),
            "joint_geometry_error": float(joint_geometry_error),
            "joint_weight_error": float(joint_weight_error),
            "joint_fit_rmse": float(joint_fit_rmse),
            "joint_seed_rank": float(joint["seed_rank"]),
            "oracle_pose_alpha_error": float(oracle_alpha_error),
            "oracle_pose_geometry_error": float(oracle_geometry_error),
            "oracle_pose_weight_error": float(oracle_weight_error),
            "oracle_pose_fit_rmse": float(oracle_fit_rmse),
        }
    )
    return row


def summarize_cache_rows(rows: list[dict[str, float | str]]) -> dict[str, object]:
    by_condition: list[dict[str, float | str]] = []
    by_cell: list[dict[str, float | str]] = []

    for condition in FOCUS_CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        by_condition.append(
            {
                "condition": condition,
                "count": len(subset),
                "support_alpha_error_mean": mean([to_float(row["support_alpha_error"]) for row in subset]),
                "joint_alpha_error_mean": mean([to_float(row["joint_alpha_error"]) for row in subset]),
                "support_cv_score_mean": mean([to_float(row["support_cv_score"]) for row in subset]),
                "joint_cv_score_mean": mean([to_float(row["joint_cv_score"]) for row in subset]),
            }
        )
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            cell_subset = [row for row in subset if row["geometry_skew_bin"] == skew_bin]
            if not cell_subset:
                continue
            by_cell.append(
                {
                    "condition": condition,
                    "alpha_strength_bin": FOCUS_ALPHA_BIN,
                    "geometry_skew_bin": skew_bin,
                    "count": len(cell_subset),
                    "support_alpha_error_mean": mean([to_float(row["support_alpha_error"]) for row in cell_subset]),
                    "joint_alpha_error_mean": mean([to_float(row["joint_alpha_error"]) for row in cell_subset]),
                    "support_cv_score_mean": mean([to_float(row["support_cv_score"]) for row in cell_subset]),
                    "joint_cv_score_mean": mean([to_float(row["joint_cv_score"]) for row in cell_subset]),
                }
            )

    return {
        "count": len(rows),
        "by_condition": by_condition,
        "by_cell": by_cell,
    }


def generate_block_cache(block_name: str, variant_name: str, force: bool = False) -> list[dict[str, float | str]]:
    block = BLOCK_SPECS[block_name]
    variant = VARIANT_SPECS[variant_name]
    path = cache_path(block_name, variant_name)
    meta_path = cache_meta_path(block_name, variant_name)
    if os.path.exists(path) and not force:
        return load_csv_rows(path)

    bank_rng = np.random.default_rng(block.bank_seed)
    bank_params, bank_signatures = build_reference_bank(variant.reference_bank_size, bank_rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_signatures)

    rows: list[dict[str, float | str]] = []
    for obs_seed in block.obs_seeds:
        for condition in FOCUS_CONDITIONS:
            for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
                rows.append(
                    build_trial_row(
                        variant,
                        block,
                        int(obs_seed),
                        condition,
                        skew_bin,
                        bank_params,
                        shifted_bank,
                        bank_signatures,
                    )
                )

    write_csv(path, rows)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "block": block.__dict__,
                "variant": variant.__dict__,
                "summary": summarize_cache_rows(rows),
            },
            handle,
            indent=2,
        )
    return rows


def build_feature_vector(row: dict[str, float | str]) -> np.ndarray:
    vector: list[float] = []
    active_cell = cell_name(str(row["condition"]), str(row["geometry_skew_bin"]))
    for name in CELL_NAMES:
        vector.append(1.0 if name == active_cell else 0.0)

    vector.extend(
        [
            float(math.log(max(to_float(row["support_alpha"]), NUMERIC_EPS))),
            float(math.log(max(to_float(row["joint_alpha"]), NUMERIC_EPS))),
            float(abs(to_float(row["support_t"]))),
            float(abs(to_float(row["joint_t"]))),
            float(to_float(row["support_score"])),
            float(to_float(row["joint_score"])),
            float(to_float(row["support_entropy"])),
            float(to_float(row["joint_entropy"])),
            float(to_float(row["support_cv_score"])),
            float(to_float(row["joint_cv_score"])),
        ]
    )
    return np.array(vector, dtype=float)


def regression_target(row: dict[str, float | str]) -> float:
    return float(to_float(row["joint_alpha_error"]) - to_float(row["support_alpha_error"]))


def fit_ridge_model(rows: list[dict[str, float | str]]) -> dict[str, object]:
    x = np.vstack([build_feature_vector(row) for row in rows])
    y = np.array([regression_target(row) for row in rows], dtype=float)

    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std > 0.0, feature_std, 1.0)
    x_scaled = (x - feature_mean) / feature_std

    x_augmented = np.concatenate([x_scaled, np.ones((x_scaled.shape[0], 1), dtype=float)], axis=1)
    penalty = np.eye(x_augmented.shape[1], dtype=float)
    penalty[-1, -1] = 0.0
    gram = x_augmented.T @ x_augmented + RIDGE_LAMBDA * penalty
    rhs = x_augmented.T @ y
    coefficients = np.linalg.solve(gram, rhs)

    return {
        "ridge_lambda": float(RIDGE_LAMBDA),
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "weights": coefficients[:-1].tolist(),
        "bias": float(coefficients[-1]),
        "train_count": len(rows),
    }


def save_model_artifact(
    variant_name: str,
    train_block_names: list[str],
    model: dict[str, object],
) -> str:
    path = chooser_model_path(variant_name)
    payload = {
        "solver_family": "bank_adaptive_ridge_chooser",
        "variant": VARIANT_SPECS[variant_name].__dict__,
        "train_blocks": train_block_names,
        "focused_conditions": FOCUS_CONDITIONS,
        "focused_alpha_bin": FOCUS_ALPHA_BIN,
        "cells": CELL_NAMES,
        **model,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_model_artifact(variant_name: str) -> dict[str, object]:
    with open(chooser_model_path(variant_name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_delta(row: dict[str, float | str], model: dict[str, object]) -> float:
    features = build_feature_vector(row)
    mean_vec = np.array(model["feature_mean"], dtype=float)
    std_vec = np.array(model["feature_std"], dtype=float)
    scaled = (features - mean_vec) / std_vec
    weights = np.array(model["weights"], dtype=float)
    return float(np.dot(scaled, weights) + float(model["bias"]))


def choose_from_cached_row(row: dict[str, float | str], model: dict[str, object]) -> dict[str, float | str]:
    predicted_delta = predict_delta(row, model)
    choose_joint = int(predicted_delta < 0.0)
    chosen_prefix = "joint" if choose_joint else "support"
    return {
        **row,
        "predicted_delta": float(predicted_delta),
        "choose_joint": float(choose_joint),
        "chooser_alpha_error": float(to_float(row[f"{chosen_prefix}_alpha_error"])),
        "chooser_geometry_error": float(to_float(row[f"{chosen_prefix}_geometry_error"])),
        "chooser_weight_error": float(to_float(row[f"{chosen_prefix}_weight_error"])),
        "chooser_fit_rmse": float(to_float(row[f"{chosen_prefix}_fit_rmse"])),
        "chooser_score": float(to_float(row[f"{chosen_prefix}_score"])),
        "chooser_entropy": float(to_float(row[f"{chosen_prefix}_entropy"])),
        "chooser_cv_score": float(to_float(row[f"{chosen_prefix}_cv_score"])),
        "chooser_alpha": float(to_float(row[f"{chosen_prefix}_alpha"])),
        "chooser_t": float(to_float(row[f"{chosen_prefix}_t"])),
    }


def summarize_predictions(rows: list[dict[str, float | str]], label: str) -> dict[str, object]:
    support_mean = mean([to_float(row["support_alpha_error"]) for row in rows])
    joint_mean = mean([to_float(row["joint_alpha_error"]) for row in rows])
    chooser_mean = mean([to_float(row["chooser_alpha_error"]) for row in rows])
    choose_joint_fraction = mean([to_float(row["choose_joint"]) for row in rows])

    beats_support = chooser_mean + NUMERIC_EPS < support_mean
    beats_joint = chooser_mean + NUMERIC_EPS < joint_mean

    by_condition: list[dict[str, float | str]] = []
    by_cell: list[dict[str, float | str]] = []

    for condition in FOCUS_CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        by_condition.append(
            {
                "condition": condition,
                "count": len(subset),
                "support_alpha_error_mean": mean([to_float(row["support_alpha_error"]) for row in subset]),
                "joint_alpha_error_mean": mean([to_float(row["joint_alpha_error"]) for row in subset]),
                "chooser_alpha_error_mean": mean([to_float(row["chooser_alpha_error"]) for row in subset]),
                "choose_joint_fraction": mean([to_float(row["choose_joint"]) for row in subset]),
            }
        )

        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            cell_subset = [row for row in subset if row["geometry_skew_bin"] == skew_bin]
            if not cell_subset:
                continue
            by_cell.append(
                {
                    "condition": condition,
                    "alpha_strength_bin": FOCUS_ALPHA_BIN,
                    "geometry_skew_bin": skew_bin,
                    "count": len(cell_subset),
                    "support_alpha_error_mean": mean([to_float(row["support_alpha_error"]) for row in cell_subset]),
                    "joint_alpha_error_mean": mean([to_float(row["joint_alpha_error"]) for row in cell_subset]),
                    "chooser_alpha_error_mean": mean([to_float(row["chooser_alpha_error"]) for row in cell_subset]),
                    "choose_joint_fraction": mean([to_float(row["choose_joint"]) for row in cell_subset]),
                }
            )

    if beats_support and beats_joint:
        plain_language = (
            f"The frozen chooser beat both cached candidates on {label}: "
            f"support {support_mean:.4f}, joint {joint_mean:.4f}, chooser {chooser_mean:.4f}."
        )
        bgp_interpretation = "strengthens BGP"
    elif beats_support:
        plain_language = (
            f"The frozen chooser beat the support-aware baseline but not the joint candidate on {label}: "
            f"support {support_mean:.4f}, joint {joint_mean:.4f}, chooser {chooser_mean:.4f}."
        )
        bgp_interpretation = "narrows the remaining solver bottleneck"
    elif beats_joint:
        plain_language = (
            f"The frozen chooser beat the joint candidate but not the support-aware baseline on {label}: "
            f"support {support_mean:.4f}, joint {joint_mean:.4f}, chooser {chooser_mean:.4f}."
        )
        bgp_interpretation = "narrows the remaining solver bottleneck"
    else:
        plain_language = (
            f"The frozen chooser did not beat either cached candidate on {label}: "
            f"support {support_mean:.4f}, joint {joint_mean:.4f}, chooser {chooser_mean:.4f}."
        )
        bgp_interpretation = "narrows the remaining solver bottleneck"

    return {
        "label": label,
        "count": len(rows),
        "support_alpha_error_mean": float(support_mean),
        "joint_alpha_error_mean": float(joint_mean),
        "chooser_alpha_error_mean": float(chooser_mean),
        "choose_joint_fraction": float(choose_joint_fraction),
        "chooser_minus_support_alpha_error": float(chooser_mean - support_mean),
        "chooser_minus_joint_alpha_error": float(chooser_mean - joint_mean),
        "beats_support": bool(beats_support),
        "beats_joint": bool(beats_joint),
        "by_condition": by_condition,
        "by_cell": by_cell,
        "plain_language_result": plain_language,
        "bgp_interpretation": bgp_interpretation,
    }


def evaluate_on_rows(
    rows: list[dict[str, float | str]],
    model: dict[str, object],
    label: str,
) -> tuple[list[dict[str, float | str]], dict[str, object]]:
    predictions = [choose_from_cached_row(row, model) for row in rows]
    summary = summarize_predictions(predictions, label)
    return predictions, summary


def write_predictions_csv(variant_name: str, report_name: str, rows: list[dict[str, float | str]]) -> str:
    path = report_csv_path(variant_name, report_name)
    write_csv(path, rows)
    return path


def save_report(variant_name: str, report_name: str, payload: dict[str, object]) -> str:
    path = report_json_path(variant_name, report_name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def fit_and_freeze_official_model(variant_name: str) -> dict[str, object]:
    train_blocks = ["calibration_block_1", "calibration_block_2"]
    train_rows = load_rows_for_blocks(train_blocks, variant_name)
    model = fit_ridge_model(train_rows)
    save_model_artifact(variant_name, train_blocks, model)
    return load_model_artifact(variant_name)


def smoke_check(variant_name: str) -> dict[str, object]:
    train_rows = load_rows_for_blocks(["calibration_block_1"], variant_name)
    smoke_model = fit_ridge_model(train_rows)
    predictions, summary = evaluate_on_rows(train_rows, smoke_model, "smoke_calibration_block_1")
    write_predictions_csv(variant_name, "smoke_calibration_block_1_predictions", predictions)
    save_report(
        variant_name,
        "smoke_calibration_block_1_summary",
        {
            "variant": VARIANT_SPECS[variant_name].__dict__,
            "summary": summary,
        },
    )
    return summary


def fit_eval_from_existing_cache(
    variant_name: str,
    include_confirmation: bool = True,
) -> dict[str, object]:
    model = fit_and_freeze_official_model(variant_name)

    calibration_rows = load_rows_for_blocks(["calibration_block_1", "calibration_block_2"], variant_name)
    holdout_rows = load_rows_for_blocks(["holdout_block_1"], variant_name)

    calibration_predictions, calibration_summary = evaluate_on_rows(calibration_rows, model, "calibration")
    holdout_predictions, holdout_summary = evaluate_on_rows(holdout_rows, model, "holdout_block_1")

    write_predictions_csv(variant_name, "calibration_predictions", calibration_predictions)
    write_predictions_csv(variant_name, "holdout_block_1_predictions", holdout_predictions)

    payload: dict[str, object] = {
        "variant": VARIANT_SPECS[variant_name].__dict__,
        "model_path": chooser_model_path(variant_name),
        "calibration": calibration_summary,
        "holdout": holdout_summary,
    }

    if include_confirmation and os.path.exists(cache_path("confirmation_block", variant_name)):
        confirmation_rows = load_rows_for_blocks(["confirmation_block"], variant_name)
        confirmation_predictions, confirmation_summary = evaluate_on_rows(
            confirmation_rows,
            model,
            "confirmation_block",
        )
        write_predictions_csv(variant_name, "confirmation_block_predictions", confirmation_predictions)
        payload["confirmation"] = confirmation_summary

    save_report(variant_name, "fit_eval_summary", payload)
    return payload


def bank_adaptive_solver(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    condition: str,
    geometry_skew_bin: str,
    bank_params: list[tuple[float, float, float, float, float, float]],
    shifted_bank: np.ndarray,
    chooser_model: dict[str, object],
    top_k_seeds: int,
) -> tuple[tuple[float, float, float, float, float, float], dict[str, float | str]]:
    candidates = support_and_joint_candidates(
        observed_signature,
        mask,
        condition,
        bank_params,
        shifted_bank,
        top_k_seeds,
    )
    support = candidates["support"]
    joint = candidates["joint"]
    feature_row = observable_feature_row(
        condition,
        geometry_skew_bin,
        support["params"],
        float(support["score"]),
        float(support["entropy"]),
        cross_validated_shift_score(observed_signature, mask, support["params"], np.random.default_rng(0)),
        joint["params"],
        float(joint["score"]),
        float(joint["entropy"]),
        cross_validated_shift_score(observed_signature, mask, joint["params"], np.random.default_rng(0)),
    )
    predicted_delta = predict_delta(feature_row, chooser_model)
    choose_joint = predicted_delta < 0.0
    chosen = joint if choose_joint else support
    diagnostics = {
        "predicted_delta": float(predicted_delta),
        "choose_joint": float(int(choose_joint)),
        "support_score": float(support["score"]),
        "joint_score": float(joint["score"]),
        "support_entropy": float(support["entropy"]),
        "joint_entropy": float(joint["entropy"]),
    }
    return chosen["params"], diagnostics


def run_variant_ladder(variant_name: str, force: bool = False) -> dict[str, object]:
    generate_block_cache("calibration_block_1", variant_name, force=force)
    smoke_summary = smoke_check(variant_name)

    generate_block_cache("calibration_block_2", variant_name, force=force)
    model = fit_and_freeze_official_model(variant_name)

    generate_block_cache("holdout_block_1", variant_name, force=force)
    official_payload = fit_eval_from_existing_cache(variant_name, include_confirmation=False)

    holdout_summary = official_payload["holdout"]
    holdout_success = bool(holdout_summary["beats_support"] and holdout_summary["beats_joint"])

    confirmation_summary = None
    if holdout_success:
        generate_block_cache("confirmation_block", variant_name, force=force)
        confirmation_rows = load_rows_for_blocks(["confirmation_block"], variant_name)
        confirmation_predictions, confirmation_summary = evaluate_on_rows(
            confirmation_rows,
            model,
            "confirmation_block",
        )
        write_predictions_csv(variant_name, "confirmation_block_predictions", confirmation_predictions)

    working_solver = bool(
        holdout_success
        and confirmation_summary is not None
        and confirmation_summary["beats_support"]
        and confirmation_summary["beats_joint"]
    )

    result = {
        "variant": VARIANT_SPECS[variant_name].__dict__,
        "smoke": smoke_summary,
        "calibration": official_payload["calibration"],
        "holdout": holdout_summary,
        "confirmation": confirmation_summary,
        "model_path": chooser_model_path(variant_name),
        "holdout_success": holdout_success,
        "working_solver": working_solver,
        "plain_language_result": (
            str(confirmation_summary["plain_language_result"])
            if confirmation_summary is not None
            else str(holdout_summary["plain_language_result"])
        ),
        "bgp_interpretation": (
            str(confirmation_summary["bgp_interpretation"])
            if confirmation_summary is not None
            else str(holdout_summary["bgp_interpretation"])
        ),
    }
    save_report(variant_name, "ladder_summary", result)
    return result


def run_full_plan(force: bool = False) -> dict[str, object]:
    baseline_result = run_variant_ladder("baseline", force=force)
    if baseline_result["working_solver"]:
        final_payload = {
            "selected_variant": "baseline",
            "baseline": baseline_result,
            "final_plain_language_result": baseline_result["plain_language_result"],
            "final_bgp_interpretation": baseline_result["bgp_interpretation"],
        }
        save_report("baseline", "full_plan_result", final_payload)
        return final_payload

    if baseline_result["holdout_success"] and not baseline_result["working_solver"]:
        final_payload = {
            "selected_variant": "baseline",
            "baseline": baseline_result,
            "final_plain_language_result": (
                "The frozen chooser cleared holdout block 1 but failed fresh-bank confirmation without recalibration; "
                "the routed solver is not yet reliable on the focused slice."
            ),
            "final_bgp_interpretation": "narrows the remaining solver bottleneck",
        }
        save_report("baseline", "full_plan_result", final_payload)
        return final_payload

    density_result = run_variant_ladder("density_ablation", force=force)
    final_payload = {
        "selected_variant": "density_ablation",
        "baseline": baseline_result,
        "density_ablation": density_result,
        "final_plain_language_result": density_result["plain_language_result"],
        "final_bgp_interpretation": density_result["bgp_interpretation"],
    }
    if not density_result["holdout_success"]:
        final_payload["final_plain_language_result"] = (
            "The support-vs-joint router stayed bank-sensitive after the allowed density ablation; "
            "current support-vs-joint routing is too bank-sensitive for a reliable solution under this solver family."
        )
        final_payload["final_bgp_interpretation"] = "narrows the remaining solver bottleneck"
    elif not density_result["working_solver"]:
        final_payload["final_plain_language_result"] = (
            "The density ablation cleared holdout block 1 but failed fresh-bank confirmation without recalibration; "
            "current support-vs-joint routing is still too bank-sensitive for a reliable solution under this solver family."
        )
        final_payload["final_bgp_interpretation"] = "narrows the remaining solver bottleneck"

    save_report("density_ablation", "full_plan_result", final_payload)
    return final_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bank-adaptive focused solver workflow.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    generate_block_parser = subparsers.add_parser("generate-block", help="Generate one cached candidate table.")
    generate_block_parser.add_argument("--block", choices=sorted(BLOCK_SPECS.keys()), required=True)
    generate_block_parser.add_argument("--variant", choices=sorted(VARIANT_SPECS.keys()), default="baseline")
    generate_block_parser.add_argument("--force", action="store_true")

    smoke_parser = subparsers.add_parser("smoke-check", help="Fit and evaluate the chooser on calibration block 1 only.")
    smoke_parser.add_argument("--variant", choices=sorted(VARIANT_SPECS.keys()), default="baseline")

    fit_eval_parser = subparsers.add_parser("fit-eval", help="Fit the official chooser from existing cache and evaluate.")
    fit_eval_parser.add_argument("--variant", choices=sorted(VARIANT_SPECS.keys()), default="baseline")
    fit_eval_parser.add_argument("--include-confirmation", action="store_true")

    run_ladder_parser = subparsers.add_parser("run-ladder", help="Run the cache, fit, holdout, and optional confirmation ladder for one variant.")
    run_ladder_parser.add_argument("--variant", choices=sorted(VARIANT_SPECS.keys()), default="baseline")
    run_ladder_parser.add_argument("--force", action="store_true")

    full_parser = subparsers.add_parser("run-full", help="Run the baseline ladder and the single allowed fallback branch if needed.")
    full_parser.add_argument("--force", action="store_true")

    parser.set_defaults(command="run-full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "generate-block":
        rows = generate_block_cache(args.block, args.variant, force=bool(args.force))
        print(
            json.dumps(
                {
                    "variant": VARIANT_SPECS[args.variant].__dict__,
                    "block": BLOCK_SPECS[args.block].__dict__,
                    "cache_path": cache_path(args.block, args.variant),
                    "summary": summarize_cache_rows(rows),
                },
                indent=2,
            )
        )
        return

    if args.command == "fit-eval":
        result = fit_eval_from_existing_cache(
            args.variant,
            include_confirmation=bool(args.include_confirmation),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "smoke-check":
        result = smoke_check(args.variant)
        print(json.dumps(result, indent=2))
        return

    if args.command == "run-ladder":
        result = run_variant_ladder(args.variant, force=bool(args.force))
        print(json.dumps(result, indent=2))
        return

    result = run_full_plan(force=bool(args.force))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
