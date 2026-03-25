"""
Scope Boundary Experiment for Shape Budget.

This experiment is a theory-hardening synthesis benchmark. It does not generate
new forward families. It reads the established experiment outputs and measures:

1. where one-knob compression succeeds exactly,
2. where one-knob compression fails but a compact higher-dimensional control
   object restores collapse,
3. where the latent object remains operational in inverse settings, and
4. where the current solver challenge is branch-specific rather than a general
   failure of BGP.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

SUMMARY_PATHS = {
    "control_knob": ROOT / "experiments/core-control-knob/control-knob/outputs/control_knob_summary.json",
    "manifold_dimension": ROOT / "experiments/core-control-knob/manifold-dimension/outputs/manifold_summary.json",
    "identifiability": ROOT / "experiments/core-control-knob/identifiability-and-baselines/outputs/experiment_summary.json",
    "representation_independence": ROOT / "experiments/core-control-knob/representation-independence/outputs/representation_independence_summary.json",
    "probe_specialization": ROOT / "experiments/core-control-knob/probe-specialization/outputs/probe_specialization_summary.json",
    "asymmetry": ROOT / "experiments/two-source-extensions/asymmetry/outputs/asymmetry_summary.json",
    "hyperbola_twin": ROOT / "experiments/two-source-extensions/hyperbola-twin/outputs/hyperbola_twin_summary.json",
    "anisotropy": ROOT / "experiments/two-source-extensions/anisotropy/outputs/anisotropy_summary.json",
    "multisource": ROOT / "experiments/multisource-control-objects/multisource/outputs/multisource_summary.json",
    "weighted_multisource": ROOT / "experiments/multisource-control-objects/weighted-multisource/outputs/weighted_multisource_summary.json",
    "weighted_multisource_inverse": ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/outputs/weighted_multisource_inverse_summary.json",
    "weighted_anisotropic_inverse": ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/outputs/weighted_anisotropic_inverse_summary.json",
    "pose_free_weighted_anisotropic_inverse": ROOT / "experiments/multisource-control-objects/pose-free-weighted-anisotropic-inverse/outputs/pose_free_weighted_anisotropic_inverse_summary.json",
}

DIMENSION_COLORS = {
    1: "#457b9d",
    2: "#2a9d8f",
    3: "#f4a261",
    5: "#d62828",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    loaded: dict[str, dict[str, Any]] = {}
    audit: dict[str, bool] = {}
    for name, path in SUMMARY_PATHS.items():
        exists = path.exists()
        audit[name] = exists
        if not exists:
            raise FileNotFoundError(f"Missing required input: {path}")
        loaded[name] = read_json(path)
    return loaded, audit


def build_branch_rows(inputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    control = inputs["control_knob"]
    manifold = inputs["manifold_dimension"]["manifold_metrics"]
    hyperbola = inputs["hyperbola_twin"]
    asymmetry = inputs["asymmetry"]
    anisotropy = inputs["anisotropy"]
    multisource = inputs["multisource"]
    weighted = inputs["weighted_multisource"]

    return [
        {
            "branch": "symmetric ellipse",
            "scope_class": "exact one-knob base case",
            "control_object": "e = c / a",
            "established_dimension": 1,
            "collapse_error": float(control["max_scale_collapse_error"]),
            "supporting_metric": float(manifold["pc1_explained_variance_ratio"]),
            "supporting_metric_label": "PC1 explained variance ratio",
            "secondary_metric": float(manifold["abs_isomap_spearman_rho_with_e"]),
            "secondary_metric_label": "abs Isomap Spearman rho with e",
            "notes": "full normalized family is one-dimensional in boundary space",
        },
        {
            "branch": "hyperbola twin",
            "scope_class": "exact one-knob twin",
            "control_object": "lambda = a / c",
            "established_dimension": 1,
            "collapse_error": float(hyperbola["max_scale_collapse_error"]),
            "supporting_metric": float(hyperbola["scale_spread"]["normalized_openness"]),
            "supporting_metric_label": "normalized openness scale spread",
            "secondary_metric": float(hyperbola["scale_spread"]["normalized_vertex_curvature"]),
            "secondary_metric_label": "vertex-curvature scale spread",
            "notes": "deficit-side twin stays one-knob under its own bounded ratio",
        },
        {
            "branch": "asymmetry",
            "scope_class": "structured one-knob failure",
            "control_object": "(e, w)",
            "established_dimension": 2,
            "collapse_error": float(asymmetry["max_two_knob_scale_collapse_error"]),
            "supporting_metric": float(asymmetry["min_one_knob_family_distance"]),
            "supporting_metric_label": "minimum one-knob family distance",
            "secondary_metric": float(asymmetry["max_one_knob_family_distance"]),
            "secondary_metric_label": "maximum one-knob family distance",
            "notes": "e alone fails; fixed (e, w) restores collapse",
        },
        {
            "branch": "raw anisotropy",
            "scope_class": "structured one-knob failure",
            "control_object": "(e, alpha) raw; e after whitening",
            "established_dimension": 2,
            "collapse_error": float(anisotropy["max_raw_scale_collapse_error"]),
            "supporting_metric": float(anisotropy["min_raw_family_distance_fixed_e"]),
            "supporting_metric_label": "minimum raw one-knob family distance",
            "secondary_metric": float(anisotropy["max_whitened_collapse_error"]),
            "secondary_metric_label": "maximum whitened one-knob collapse error",
            "notes": "raw family needs alpha; whitening restores the original one-knob base case",
        },
        {
            "branch": "equal-weight three-source",
            "scope_class": "low-dimensional control manifold",
            "control_object": "normalized source triangle",
            "established_dimension": 3,
            "collapse_error": float(multisource["max_boundary_collapse_error"]),
            "supporting_metric": float(multisource["random_pc3_cumulative_explained_variance_ratio"]),
            "supporting_metric_label": "random-family PC3 cumulative variance",
            "secondary_metric": float(multisource["equilateral_pc1_explained_variance_ratio"]),
            "secondary_metric_label": "equilateral-slice PC1 variance",
            "notes": "broad family is about three-parameter; equilateral slice is near one-parameter",
        },
        {
            "branch": "weighted three-source",
            "scope_class": "low-dimensional control manifold",
            "control_object": "normalized source triangle + weight simplex",
            "established_dimension": 5,
            "collapse_error": float(weighted["max_boundary_collapse_error"]),
            "supporting_metric": float(weighted["random_weighted_pc5_cumulative_explained_variance_ratio"]),
            "supporting_metric_label": "random-weighted PC5 cumulative variance",
            "secondary_metric": float(weighted["equilateral_weighted_pc2_cumulative_explained_variance_ratio"]),
            "secondary_metric_label": "equilateral-weighted PC2 cumulative variance",
            "notes": "broad family is about five-parameter; equilateral weighted slice is near two-parameter",
        },
    ]


def build_negative_control_rows(inputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    asymmetry = inputs["asymmetry"]
    anisotropy = inputs["anisotropy"]
    weighted = inputs["weighted_multisource"]

    rows = [
        {
            "branch": "asymmetry",
            "wrong_compression": "e only",
            "wrong_floor": float(asymmetry["min_one_knob_family_distance"]),
            "wrong_ceiling": float(asymmetry["max_one_knob_family_distance"]),
            "corrected_compression": "fixed (e, w)",
            "corrected_error": float(asymmetry["max_two_knob_scale_collapse_error"]),
            "recovered_reduced_compression": "",
            "recovered_error": "",
        },
        {
            "branch": "raw anisotropy",
            "wrong_compression": "e only in raw coordinates",
            "wrong_floor": float(anisotropy["min_raw_family_distance_fixed_e"]),
            "wrong_ceiling": float(anisotropy["max_raw_family_distance_fixed_e"]),
            "corrected_compression": "fixed (e, alpha)",
            "corrected_error": float(anisotropy["max_raw_scale_collapse_error"]),
            "recovered_reduced_compression": "whitened e only",
            "recovered_error": float(anisotropy["max_whitened_collapse_error"]),
        },
        {
            "branch": "weighted three-source",
            "wrong_compression": "fixed geometry only",
            "wrong_floor": float(weighted["min_fixed_geometry_boundary_family_distance"]),
            "wrong_ceiling": float(weighted["max_fixed_geometry_boundary_family_distance"]),
            "corrected_compression": "fixed geometry + weights",
            "corrected_error": float(weighted["max_boundary_collapse_error"]),
            "recovered_reduced_compression": "",
            "recovered_error": "",
        },
    ]

    for row in rows:
        strongest_error = float(row["corrected_error"])
        if row["recovered_error"] != "":
            strongest_error = min(strongest_error, float(row["recovered_error"]))
        row["wrong_to_right_gap_factor"] = float(row["wrong_floor"]) / strongest_error
    return rows


def build_representation_penalty_rows(inputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = inputs["representation_independence"]["by_condition_mode_representation"]
    indexed = {
        (record["representation"], record["mode"], record["condition"]): record
        for record in records
    }
    conditions = sorted({record["condition"] for record in records})
    rows: list[dict[str, Any]] = []
    for representation in ["radial", "support"]:
        for condition in conditions:
            canonical = indexed[(representation, "canonical", condition)]
            pose_free = indexed[(representation, "pose_free", condition)]
            geometry_penalty = float(pose_free["geometry_mae_mean"]) / float(canonical["geometry_mae_mean"])
            alpha_penalty = float(pose_free["alpha_mae_mean"]) / float(canonical["alpha_mae_mean"])
            rows.append(
                {
                    "representation": representation,
                    "condition": condition,
                    "geometry_penalty": geometry_penalty,
                    "alpha_penalty": alpha_penalty,
                    "selectivity": alpha_penalty / geometry_penalty,
                    "canonical_fit_improvement_factor": float(canonical["fit_improvement_factor_mean"]),
                    "pose_free_fit_improvement_factor": float(pose_free["fit_improvement_factor_mean"]),
                }
            )
    return rows


def build_posefree_penalty_rows(inputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in inputs["pose_free_weighted_anisotropic_inverse"]["penalties_vs_canonical_anisotropic"]:
        geometry_penalty = float(record["geometry_mae_penalty_factor"])
        weight_penalty = float(record["weight_mae_penalty_factor"])
        alpha_penalty = float(record["alpha_mae_penalty_factor"])
        rows.append(
            {
                "condition": record["condition"],
                "geometry_penalty": geometry_penalty,
                "weight_penalty": weight_penalty,
                "alpha_penalty": alpha_penalty,
                "alpha_over_geometry_penalty": alpha_penalty / geometry_penalty,
                "alpha_over_weight_penalty": alpha_penalty / weight_penalty,
                "fit_penalty": float(record["anisotropic_fit_rmse_penalty_factor"]),
                "fit_improvement_ratio_vs_canonical": float(record["fit_improvement_ratio_vs_canonical"]),
            }
        )
    return rows


def build_operational_rows(
    inputs: dict[str, dict[str, Any]],
    representation_penalties: list[dict[str, Any]],
    posefree_penalties: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    identifiability = inputs["identifiability"]
    weighted_inverse = inputs["weighted_multisource_inverse"]["summary"]
    anisotropic_inverse = inputs["weighted_anisotropic_inverse"]["summary"]
    probe = inputs["probe_specialization"]
    posefree = inputs["pose_free_weighted_anisotropic_inverse"]["summary"]

    baseline_advantages = []
    for values in identifiability["baseline_rmse"].values():
        baseline_advantages.append(float(values["d_and_S"]) / float(values["e_only"]))

    support_rows = [row for row in representation_penalties if row["representation"] == "support"]
    router_results = probe["router_results"]
    router_factors = []
    router_wins = 0
    conditions = sorted({row["condition"] for row in router_results})
    for condition in conditions:
        local = [row for row in router_results if row["condition"] == condition]
        fixed = min(
            float(row["mean_abs_error"])
            for row in local
            if row["method"] in {"perimeter_only", "width_only", "major_tip_only"}
        )
        router = next(float(row["mean_abs_error"]) for row in local if row["method"] == "router")
        factor = fixed / router
        router_factors.append(factor)
        if factor > 1.0:
            router_wins += 1

    return [
        {
            "branch": "symmetric known-source inverse",
            "scope_status": "operational",
            "control_object": "e = c / a",
            "primary_metric": "worst p95 absolute e error",
            "primary_value": max(
                float(record["worst_p95_abs_error"])
                for record in identifiability["identifiability"].values()
            ),
            "secondary_metric": "d_and_S over e-only RMSE ratio",
            "secondary_min_value": min(baseline_advantages),
            "secondary_max_value": max(baseline_advantages),
            "notes": "base-case control knob is recoverable and scale-generalizing",
        },
        {
            "branch": "weighted multisource inverse",
            "scope_status": "operational latent object",
            "control_object": "geometry + weights",
            "primary_metric": "fit improvement over equal-weight baseline",
            "primary_value": float(weighted_inverse["smallest_fit_improvement_factor_mean"]),
            "secondary_metric": "best fit improvement over equal-weight baseline",
            "secondary_min_value": float(weighted_inverse["smallest_fit_improvement_factor_mean"]),
            "secondary_max_value": float(weighted_inverse["largest_fit_improvement_factor_mean"]),
            "notes": "compact control object remains useful in canonical-pose inverse recovery",
        },
        {
            "branch": "weighted anisotropic inverse",
            "scope_status": "operational latent object",
            "control_object": "geometry + weights + alpha",
            "primary_metric": "fit improvement over Euclidean baseline",
            "primary_value": float(anisotropic_inverse["smallest_fit_improvement_factor_mean"]),
            "secondary_metric": "best fit improvement over Euclidean baseline",
            "secondary_min_value": float(anisotropic_inverse["smallest_fit_improvement_factor_mean"]),
            "secondary_max_value": float(anisotropic_inverse["largest_fit_improvement_factor_mean"]),
            "notes": "medium structure joins the recoverable latent object in canonical pose",
        },
        {
            "branch": "representation independence",
            "scope_status": "operational across encodings",
            "control_object": "same latent object under support encoding",
            "primary_metric": "support selective pose-penalty minimum",
            "primary_value": min(float(row["selectivity"]) for row in support_rows),
            "secondary_metric": "support fit-improvement factor",
            "secondary_min_value": min(
                min(float(row["canonical_fit_improvement_factor"]), float(row["pose_free_fit_improvement_factor"]))
                for row in support_rows
            ),
            "secondary_max_value": max(
                max(float(row["canonical_fit_improvement_factor"]), float(row["pose_free_fit_improvement_factor"]))
                for row in support_rows
            ),
            "notes": "selective alpha fragility survives a representation swap",
        },
        {
            "branch": "probe specialization",
            "scope_status": "experimental-control support",
            "control_object": "depletion phase as router signal",
            "primary_metric": "router wins",
            "primary_value": router_wins,
            "secondary_metric": "best-fixed over router factor",
            "secondary_min_value": min(router_factors),
            "secondary_max_value": max(router_factors),
            "notes": "router beats the best fixed practical probe in most tested regimes",
        },
        {
            "branch": "pose-free weighted anisotropic inverse",
            "scope_status": "branch-specific solver challenge",
            "control_object": "geometry + weights + alpha under hidden pose",
            "primary_metric": "fit improvement over Euclidean baseline",
            "primary_value": float(posefree["smallest_fit_improvement_factor_mean"]),
            "secondary_metric": "alpha over geometry penalty factor",
            "secondary_min_value": min(float(row["alpha_over_geometry_penalty"]) for row in posefree_penalties),
            "secondary_max_value": max(float(row["alpha_over_geometry_penalty"]) for row in posefree_penalties),
            "notes": "geometry stays comparatively stable while alpha takes the main pose penalty",
        },
    ]


def build_summary(
    audit: dict[str, bool],
    branch_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    operational_rows: list[dict[str, Any]],
    representation_penalties: list[dict[str, Any]],
    posefree_penalties: list[dict[str, Any]],
) -> dict[str, Any]:
    support_rows = [row for row in representation_penalties if row["representation"] == "support"]
    one_knob_rows = [row for row in branch_rows if int(row["established_dimension"]) == 1]
    expanded_rows = [row for row in branch_rows if int(row["established_dimension"]) > 1]
    probe_row = next(row for row in operational_rows if row["branch"] == "probe specialization")
    pose_row = next(row for row in operational_rows if row["branch"] == "pose-free weighted anisotropic inverse")
    symmetric_row = next(row for row in operational_rows if row["branch"] == "symmetric known-source inverse")

    return {
        "audit": {
            "required_input_files": len(audit),
            "loaded_input_files": int(sum(1 for value in audit.values() if value)),
            "minimum_wrong_to_right_gap_factor": float(min(row["wrong_to_right_gap_factor"] for row in negative_rows)),
            "minimum_support_selectivity": float(min(row["selectivity"] for row in support_rows)),
            "minimum_posefree_alpha_over_geometry_penalty": float(
                min(row["alpha_over_geometry_penalty"] for row in posefree_penalties)
            ),
        },
        "compact_scope": {
            "one_knob_branch_count": len(one_knob_rows),
            "expanded_compact_branch_count": len(expanded_rows),
            "largest_established_dimension": int(max(row["established_dimension"] for row in branch_rows)),
            "smallest_compact_collapse_error": float(min(row["collapse_error"] for row in branch_rows)),
            "largest_compact_collapse_error": float(max(row["collapse_error"] for row in branch_rows)),
        },
        "negative_controls": {
            row["branch"]: {
                "wrong_floor": float(row["wrong_floor"]),
                "wrong_ceiling": float(row["wrong_ceiling"]),
                "corrected_error": float(row["corrected_error"]),
                "recovered_error": None if row["recovered_error"] == "" else float(row["recovered_error"]),
                "wrong_to_right_gap_factor": float(row["wrong_to_right_gap_factor"]),
            }
            for row in negative_rows
        },
        "operational_scope": {
            "minimum_scale_generalization_advantage_over_d_and_S": float(symmetric_row["secondary_min_value"]),
            "maximum_scale_generalization_advantage_over_d_and_S": float(symmetric_row["secondary_max_value"]),
            "minimum_support_fit_improvement_factor": float(
                min(
                    min(row["canonical_fit_improvement_factor"], row["pose_free_fit_improvement_factor"])
                    for row in support_rows
                )
            ),
            "maximum_support_fit_improvement_factor": float(
                max(
                    max(row["canonical_fit_improvement_factor"], row["pose_free_fit_improvement_factor"])
                    for row in support_rows
                )
            ),
            "probe_router_wins": int(probe_row["primary_value"]),
            "probe_router_total_regimes": 4,
            "probe_router_factor_min": float(probe_row["secondary_min_value"]),
            "probe_router_factor_max": float(probe_row["secondary_max_value"]),
        },
        "solver_challenge": {
            "minimum_support_selectivity": float(min(row["selectivity"] for row in support_rows)),
            "maximum_support_selectivity": float(max(row["selectivity"] for row in support_rows)),
            "minimum_posefree_fit_improvement_factor": float(pose_row["primary_value"]),
            "minimum_posefree_alpha_over_geometry_penalty": float(pose_row["secondary_min_value"]),
            "maximum_posefree_alpha_over_geometry_penalty": float(pose_row["secondary_max_value"]),
        },
    }


def plot_compactness_ladder(path: Path, branch_rows: list[dict[str, Any]]) -> None:
    ordered = branch_rows
    y = np.arange(len(ordered))
    x = np.array([int(row["established_dimension"]) for row in ordered], dtype=float)
    colors = [DIMENSION_COLORS[int(value)] for value in x]

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.hlines(y, 0.9, x, color="#d9d9d9", lw=3)
    ax.scatter(x, y, s=150, c=colors, edgecolors="white", linewidths=1.5, zorder=3)

    for idx, row in enumerate(ordered):
        label = row["control_object"]
        evidence_value = float(row["supporting_metric"])
        if evidence_value >= 1e-3:
            evidence = f"{row['supporting_metric_label']}: {evidence_value:.4f}"
        else:
            evidence = f"{row['supporting_metric_label']}: {evidence_value:.2e}"
        if row["branch"] == "raw anisotropy":
            label = f"{label}\nwhitened collapse: {float(row['secondary_metric']):.2e}"
        ax.text(float(x[idx]) + 0.08, idx, f"{label}\n{evidence}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([row["branch"] for row in ordered])
    ax.set_xlim(0.9, 5.9)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Established compact control dimension")
    ax.set_title("Scope ladder: BGP stays compact, but not universally one-scalar")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_negative_controls(path: Path, negative_rows: list[dict[str, Any]]) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.2, 4.8))
    x = np.arange(len(negative_rows))
    labels = [row["branch"] for row in negative_rows]

    for idx, row in enumerate(negative_rows):
        floor = float(row["wrong_floor"])
        ceiling = float(row["wrong_ceiling"])
        ax_left.vlines(idx, floor, ceiling, color="#d62828", lw=5, alpha=0.85)
        ax_left.scatter([idx, idx], [floor, ceiling], color="#d62828", s=48, zorder=3)
        ax_left.text(idx, ceiling * 1.18, row["wrong_compression"], ha="center", va="bottom", fontsize=8)

    ax_left.set_yscale("log")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=12, ha="right")
    ax_left.set_ylabel("Boundary-family distance")
    ax_left.set_title("Wrong compression leaves visible normalized family gaps")

    bar_width = 0.28
    for idx, row in enumerate(negative_rows):
        ax_right.bar(
            idx - bar_width / 2,
            float(row["corrected_error"]),
            width=bar_width,
            color="#2a9d8f",
            label="correct control object" if idx == 0 else None,
        )
        if row["recovered_error"] != "":
            ax_right.bar(
                idx + bar_width / 2,
                float(row["recovered_error"]),
                width=bar_width,
                color="#457b9d",
                label="reduced object after coordinate recovery" if idx == 1 else None,
            )
        ax_right.text(
            idx - bar_width / 2,
            float(row["corrected_error"]) * 1.8,
            row["corrected_compression"],
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if row["recovered_error"] != "":
            ax_right.text(
                idx + bar_width / 2,
                float(row["recovered_error"]) * 1.8,
                str(row["recovered_reduced_compression"]),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax_right.set_yscale("log")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels, rotation=12, ha="right")
    ax_right.set_ylabel("Collapse error")
    ax_right.set_title("Correct control objects recover clean normalized collapse")
    ax_right.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_solver_challenge_map(
    path: Path,
    representation_penalties: list[dict[str, Any]],
    posefree_penalties: list[dict[str, Any]],
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    support_rows = [row for row in representation_penalties if row["representation"] == "support"]
    radial_rows = [row for row in representation_penalties if row["representation"] == "radial"]
    ordered_conditions = [row["condition"] for row in support_rows]
    x = np.arange(len(ordered_conditions))

    ax_left.plot(
        x,
        [row["selectivity"] for row in radial_rows],
        marker="o",
        lw=2.6,
        color="#6d597a",
        label="radial",
    )
    ax_left.plot(
        x,
        [row["selectivity"] for row in support_rows],
        marker="o",
        lw=2.6,
        color="#2a9d8f",
        label="support",
    )
    ax_left.axhline(1.0, color="#999999", lw=1.2, linestyle="--")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(ordered_conditions, rotation=20, ha="right")
    ax_left.set_ylabel("alpha-over-geometry pose-penalty selectivity")
    ax_left.set_title("Selective alpha fragility survives the representation swap")
    ax_left.legend(loc="upper right", frameon=True)

    x2 = np.arange(len(posefree_penalties))
    width = 0.24
    ax_right.bar(
        x2 - width,
        [row["geometry_penalty"] for row in posefree_penalties],
        width=width,
        color="#457b9d",
        label="geometry",
    )
    ax_right.bar(
        x2,
        [row["weight_penalty"] for row in posefree_penalties],
        width=width,
        color="#f4a261",
        label="weights",
    )
    ax_right.bar(
        x2 + width,
        [row["alpha_penalty"] for row in posefree_penalties],
        width=width,
        color="#d62828",
        label="alpha",
    )
    ax_right.set_yscale("log")
    ax_right.set_xticks(x2)
    ax_right.set_xticklabels([row["condition"] for row in posefree_penalties], rotation=20, ha="right")
    ax_right.set_ylabel("Penalty factor vs canonical pose")
    ax_right.set_title("The current pose-free solver challenge is selective, not general")
    ax_right.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    inputs, audit = load_inputs()
    branch_rows = build_branch_rows(inputs)
    negative_rows = build_negative_control_rows(inputs)
    representation_penalties = build_representation_penalty_rows(inputs)
    posefree_penalties = build_posefree_penalty_rows(inputs)
    operational_rows = build_operational_rows(inputs, representation_penalties, posefree_penalties)
    summary = build_summary(
        audit,
        branch_rows,
        negative_rows,
        operational_rows,
        representation_penalties,
        posefree_penalties,
    )

    write_csv(OUTPUT_DIR / "scope_boundary_branch_summary.csv", branch_rows)
    write_csv(OUTPUT_DIR / "scope_boundary_negative_controls.csv", negative_rows)
    write_csv(OUTPUT_DIR / "scope_boundary_operational_summary.csv", operational_rows)
    write_csv(OUTPUT_DIR / "scope_boundary_representation_penalties.csv", representation_penalties)
    write_csv(OUTPUT_DIR / "scope_boundary_posefree_penalties.csv", posefree_penalties)
    with (OUTPUT_DIR / "scope_boundary_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    plot_compactness_ladder(FIGURE_DIR / "scope_boundary_compactness_ladder.png", branch_rows)
    plot_negative_controls(FIGURE_DIR / "scope_boundary_negative_controls.png", negative_rows)
    plot_solver_challenge_map(
        FIGURE_DIR / "scope_boundary_solver_challenge_map.png",
        representation_penalties,
        posefree_penalties,
    )

    print(f"Loaded summaries: {len(audit)}")
    print(f"Negative controls: {len(negative_rows)}")
    print(f"Operational rows: {len(operational_rows)}")
    print(
        "Minimum wrong-to-right gap factor: "
        f"{summary['audit']['minimum_wrong_to_right_gap_factor']:.2e}"
    )
    print(
        "Minimum support selectivity: "
        f"{summary['audit']['minimum_support_selectivity']:.3f}"
    )
    print(
        "Minimum pose-free alpha-over-geometry penalty: "
        f"{summary['audit']['minimum_posefree_alpha_over_geometry_penalty']:.3f}"
    )


if __name__ == "__main__":
    main()
