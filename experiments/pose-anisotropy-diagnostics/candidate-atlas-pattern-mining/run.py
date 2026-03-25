"""
Pattern mining over the candidate atlas.

This diagnostic labels recurring cluster archetypes, builds trial-level burden
metrics, and checks whether atlas structure alone contains solver-relevant
signal for Layer 3 activation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

BLOCK_SPECS, GEOMETRY_SKEW_BIN_LABELS, FOCUS_ALPHA_BIN, FOCUS_CONDITIONS = load_symbols(
    "run_candidate_atlas_pattern_constants",
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-conditional-alpha-solver/run.py",
    "BLOCK_SPECS",
    "GEOMETRY_SKEW_BIN_LABELS",
    "FOCUS_ALPHA_BIN",
    "FOCUS_CONDITIONS",
)

import matplotlib
import numpy as np
import pandas as pd
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

ATLAS_DIR = (
    ROOT
    / "experiments/pose-anisotropy-diagnostics/candidate-atlas-instrumentation/outputs"
)
PRESSURE_DIR = (
    ROOT
    / "experiments/pose-anisotropy-interventions/backbone-correction-pressure-triggered-alpha-solver/outputs"
)

CLUSTERS_PATH = ATLAS_DIR / "candidate_atlas_cluster_rows.csv"
TRIALS_PATH = ATLAS_DIR / "candidate_atlas_trial_summary.csv"
ROWS_PATH = ATLAS_DIR / "candidate_atlas_rows.csv"
PRESSURE_TRIALS_PATH = (
    PRESSURE_DIR / "backbone_correction_pressure_triggered_alpha_solver_trials.csv"
)

KEY = ["split", "observation_seed", "condition", "geometry_skew_bin"]
NUMERIC_EPS = 1.0e-9


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def assign_archetype(row: pd.Series) -> str:
    if float(row["cluster_mass_layer1"]) >= 0.18 and float(row["cluster_size"]) >= 8:
        return "dominant_core"
    if (
        float(row["cluster_alpha_span"]) >= 0.35
        and float(row["cluster_geometry_span"]) >= 0.12
    ):
        return "broad_fan"
    if float(row["cluster_alpha_span"]) >= 0.25:
        return "alpha_fan"
    if float(row["cluster_size"]) <= 2 and float(row["cluster_mass_layer1"]) <= 0.02:
        return "fringe_singleton"
    return "compact_minor"


def classify_candidate(row: pd.Series) -> str:
    if (
        float(row.get("poison_candidate_flag", 0)) == 1
        or float(row.get("pull_away_from_consensus", 0)) > 0.2
    ):
        return "poison"
    if (
        float(row.get("pull_toward_consensus", 0)) > 0.25
        and int(row.get("in_band_flag", 0)) == 1
    ):
        return "reinforce"
    if (
        float(row.get("local_density_geometry", 0)) > 30
        and float(row.get("pull_toward_consensus", 0)) < 0.1
    ):
        return "clutter"
    if int(row.get("rank_by_score", 999)) <= 5 and str(
        row.get("candidate_source", "")
    ) in ("local_expansion", "carryover"):
        return "dense_sample"
    return "avoid"


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 2:
        return float("nan")
    x_arr = np.asarray(pair["x"], dtype=float)
    y_arr = np.asarray(pair["y"], dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def choose_threshold(
    calibration: pd.DataFrame, metric: str
) -> tuple[float, float, float]:
    values = np.sort(calibration[metric].dropna().unique())
    best: tuple[float, float, float] | None = None
    for threshold in values:
        use_refined = calibration[metric] >= threshold
        error = float(
            np.mean(
                np.where(
                    use_refined,
                    calibration["refined_alpha_output_abs_error"],
                    calibration["anchored_alpha_output_abs_error"],
                )
            )
        )
        fire_rate = float(np.mean(use_refined))
        candidate = (error, fire_rate, float(threshold))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return float("nan"), float("nan"), float("nan")
    return best[2], best[0], best[1]


def summarize_by_split(trial_patterns: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in BLOCK_SPECS:
        subset = trial_patterns[trial_patterns["split"] == split]
        if subset.empty:
            continue
        rows.append(
            {
                "split": split,
                "count": int(len(subset)),
                "mean_dominant_core_mass": float(
                    subset["dominant_core_mean_mass"].mean()
                ),
                "mean_broad_fan_mass": float(subset["broad_fan_mean_mass"].mean()),
                "mean_alpha_fan_mass": float(subset["alpha_fan_mean_mass"].mean()),
                "mean_fringe_singleton_mass": float(
                    subset["fringe_singleton_mean_mass"].mean()
                ),
                "mean_compact_minor_mass": float(
                    subset["compact_minor_mean_mass"].mean()
                ),
                "mean_fan_vs_core": float(subset["fan_vs_core"].mean()),
                "mean_useful_structure_ratio": float(
                    subset["useful_structure_ratio"].mean()
                ),
                "mean_fringe_burden": float(subset["fringe_burden"].mean()),
                "mean_residual_shell_alpha_mass": float(
                    subset["residual_shell_alpha_mass"].mean()
                ),
                "mean_improve": float(subset["improve"].mean()),
                "mean_correction_pressure": float(subset["correction_pressure"].mean()),
            }
        )
    return rows


def summarize_by_condition(trial_patterns: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in BLOCK_SPECS:
        for condition in FOCUS_CONDITIONS:
            subset = trial_patterns[
                (trial_patterns["split"] == split)
                & (trial_patterns["condition"] == condition)
            ]
            if subset.empty:
                continue
            rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "count": int(len(subset)),
                    "mean_dominant_core_mass": float(
                        subset["dominant_core_mean_mass"].mean()
                    ),
                    "mean_broad_fan_mass": float(subset["broad_fan_mean_mass"].mean()),
                    "mean_alpha_fan_mass": float(subset["alpha_fan_mean_mass"].mean()),
                    "mean_fringe_singleton_mass": float(
                        subset["fringe_singleton_mean_mass"].mean()
                    ),
                    "mean_compact_minor_mass": float(
                        subset["compact_minor_mean_mass"].mean()
                    ),
                    "mean_fan_vs_core": float(subset["fan_vs_core"].mean()),
                    "mean_useful_structure_ratio": float(
                        subset["useful_structure_ratio"].mean()
                    ),
                    "mean_fringe_burden": float(subset["fringe_burden"].mean()),
                    "mean_residual_shell_alpha_mass": float(
                        subset["residual_shell_alpha_mass"].mean()
                    ),
                    "mean_improve": float(subset["improve"].mean()),
                    "mean_correction_pressure": float(
                        subset["correction_pressure"].mean()
                    ),
                }
            )
    return rows


def plot_condition_masses(
    path: str, condition_summary: list[dict[str, object]]
) -> None:
    if not condition_summary:
        return
    df = pd.DataFrame(condition_summary)
    labels = [f"{row['split']}\n{row['condition']}" for _, row in df.iterrows()]
    core = df["mean_dominant_core_mass"].to_numpy(float)
    broad = df["mean_broad_fan_mass"].to_numpy(float)
    alpha = df["mean_alpha_fan_mass"].to_numpy(float)
    fringe = df["mean_fringe_singleton_mass"].to_numpy(float)
    compact = df["mean_compact_minor_mass"].to_numpy(float)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, core, color="#355070", label="dominant_core")
    ax.bar(x, broad, bottom=core, color="#e56b6f", label="broad_fan")
    ax.bar(x, alpha, bottom=core + broad, color="#eaac8b", label="alpha_fan")
    ax.bar(
        x,
        fringe,
        bottom=core + broad + alpha,
        color="#6d597a",
        label="fringe_singleton",
    )
    ax.bar(
        x,
        compact,
        bottom=core + broad + alpha + fringe,
        color="#84a59d",
        label="compact_minor",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean per-bank archetype mass")
    ax.set_title("Candidate-atlas archetype mass by split and condition")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fan_vs_core_scatter(path: str, trial_patterns: pd.DataFrame) -> None:
    if trial_patterns.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    palette = {"sparse_full_noisy": "#355070", "sparse_partial_high_noise": "#e56b6f"}
    for condition, subset in trial_patterns.groupby("condition"):
        ax.scatter(
            subset["fan_vs_core"],
            subset["correction_pressure"],
            label=condition,
            alpha=0.8,
            s=36,
            color=palette.get(condition, None),
        )
    ax.set_xlabel("broad-fan vs core share")
    ax.set_ylabel("correction pressure")
    ax.set_title("Atlas broad-fan share vs correction pressure")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    clusters = pd.read_csv(CLUSTERS_PATH)
    atlas_trials = pd.read_csv(TRIALS_PATH)
    atlas_rows = pd.read_csv(ROWS_PATH)
    pressure_trials = pd.read_csv(PRESSURE_TRIALS_PATH)

    clusters["archetype"] = clusters.apply(assign_archetype, axis=1)
    archetype_rows = clusters.copy()

    # Per-trial cluster mass is reported as the mean over the five banks.
    bank_count = 5.0
    trial_archetypes = (
        clusters.pivot_table(
            index=KEY,
            columns="archetype",
            values="cluster_mass_layer1",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    shell_metric_rows: list[dict[str, object]] = []
    for key_values, group in clusters.groupby(KEY):
        key_dict = dict(zip(KEY, key_values))
        shell_metric_rows.append(
            {
                **key_dict,
                "residual_shell_mass": float(
                    group.loc[
                        group["cluster_mean_score_gap"] > 0.00055, "cluster_mass_layer1"
                    ].sum()
                    / bank_count
                ),
                "residual_shell_alpha_mass": float(
                    group.loc[
                        (group["cluster_mean_score_gap"] > 0.00055)
                        & (group["cluster_alpha_span"] > 0.25),
                        "cluster_mass_layer1",
                    ].sum()
                    / bank_count
                ),
            }
        )
    shell_metrics = pd.DataFrame(shell_metric_rows)
    for column in [
        "dominant_core",
        "broad_fan",
        "alpha_fan",
        "fringe_singleton",
        "compact_minor",
    ]:
        if column not in trial_archetypes.columns:
            trial_archetypes[column] = 0.0
        trial_archetypes[f"{column}_mean_mass"] = trial_archetypes[column] / bank_count

    trial_archetypes["fan_vs_core"] = trial_archetypes[
        "broad_fan_mean_mass"
    ] / np.maximum(
        trial_archetypes["broad_fan_mean_mass"]
        + trial_archetypes["dominant_core_mean_mass"],
        NUMERIC_EPS,
    )
    trial_archetypes["fringe_vs_fan"] = trial_archetypes[
        "fringe_singleton_mean_mass"
    ] / np.maximum(
        trial_archetypes["fringe_singleton_mean_mass"]
        + trial_archetypes["broad_fan_mean_mass"],
        NUMERIC_EPS,
    )
    trial_archetypes["useful_structure_ratio"] = (
        trial_archetypes["broad_fan_mean_mass"]
        + trial_archetypes["alpha_fan_mean_mass"]
    ) / np.maximum(
        trial_archetypes["dominant_core_mean_mass"]
        + trial_archetypes["fringe_singleton_mean_mass"]
        + trial_archetypes["compact_minor_mean_mass"],
        NUMERIC_EPS,
    )
    trial_archetypes["fringe_burden"] = trial_archetypes[
        "fringe_singleton_mean_mass"
    ] + (trial_archetypes["compact_minor_mean_mass"] * 0.5)
    trial_archetypes = trial_archetypes.merge(shell_metrics, on=KEY, how="left")

    layer3_subset = pressure_trials[
        KEY
        + [
            "gate_open_flag",
            "anchored_alpha_output_abs_error",
            "refined_alpha_output_abs_error",
            "correction_pressure",
            "correction_flux",
            "refined_alpha_bank_log_span",
            "triggered_alpha_output_abs_error",
        ]
    ].copy()
    trial_patterns = atlas_trials.merge(trial_archetypes, on=KEY, how="left").merge(
        layer3_subset, on=KEY, how="left"
    )
    trial_patterns["improve"] = (
        trial_patterns["anchored_alpha_output_abs_error"]
        - trial_patterns["refined_alpha_output_abs_error"]
    )

    gate_open_patterns = trial_patterns[trial_patterns["gate_open_flag"] == 1].copy()
    calibration_open = gate_open_patterns[
        gate_open_patterns["split"] == "calibration"
    ].copy()
    fan_threshold, fan_cal_error, fan_fire_rate = choose_threshold(
        calibration_open, "fan_vs_core"
    )

    fan_rule_rows = []
    for _, row in gate_open_patterns.iterrows():
        fire = int(float(row["fan_vs_core"]) >= fan_threshold)
        fan_rule_rows.append(
            {
                **row.to_dict(),
                "fan_rule_fire_flag": fire,
                "fan_rule_output_abs_error": float(
                    row["refined_alpha_output_abs_error"]
                    if fire == 1
                    else row["anchored_alpha_output_abs_error"]
                ),
            }
        )
    fan_rule_df = pd.DataFrame(fan_rule_rows)

    split_summary = summarize_by_split(trial_patterns)
    condition_summary = summarize_by_condition(trial_patterns)

    cluster_mass_condition = (
        archetype_rows.groupby(["archetype", "condition"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    old_poison_candidates = atlas_rows[
        (atlas_rows["in_band_flag"] == 1)
        & (atlas_rows["consensus_weight_layer1"] >= 0.02)
    ].copy()
    old_poison_candidates["old_poison_score"] = (
        old_poison_candidates["consensus_weight_layer1"]
        * old_poison_candidates["pull_away_from_consensus"]
        * old_poison_candidates["local_density_geometry"]
    )
    old_poison_join = old_poison_candidates.merge(
        layer3_subset[
            KEY
            + [
                "anchored_alpha_output_abs_error",
                "refined_alpha_output_abs_error",
                "correction_pressure",
            ]
        ],
        on=KEY,
        how="left",
    )
    old_poison_join["improve"] = (
        old_poison_join["anchored_alpha_output_abs_error"]
        - old_poison_join["refined_alpha_output_abs_error"]
    )

    summary_payload = {
        "experiment": "candidate-atlas-pattern-mining",
        "trial_count": int(len(trial_patterns)),
        "gate_open_trial_count": int(len(gate_open_patterns)),
        "archetype_counts": archetype_rows["archetype"].value_counts().to_dict(),
        "archetype_condition_counts": {
            archetype: {
                condition: int(count) for condition, count in condition_counts.items()
            }
            for archetype, condition_counts in archetype_rows.groupby(
                ["archetype", "condition"]
            )
            .size()
            .unstack(fill_value=0)
            .to_dict("index")
            .items()
        },
        "gate_open_correlations": {
            "fan_vs_core_vs_improve": safe_corr(
                gate_open_patterns["fan_vs_core"], gate_open_patterns["improve"]
            ),
            "fan_vs_core_vs_correction_pressure": safe_corr(
                gate_open_patterns["fan_vs_core"],
                gate_open_patterns["correction_pressure"],
            ),
            "useful_structure_ratio_vs_improve": safe_corr(
                gate_open_patterns["useful_structure_ratio"],
                gate_open_patterns["improve"],
            ),
            "useful_structure_ratio_vs_refined_alpha_bank_log_span": safe_corr(
                gate_open_patterns["useful_structure_ratio"],
                gate_open_patterns["refined_alpha_bank_log_span"],
            ),
            "fringe_burden_vs_improve": safe_corr(
                gate_open_patterns["fringe_burden"], gate_open_patterns["improve"]
            ),
            "fringe_burden_vs_correction_pressure": safe_corr(
                gate_open_patterns["fringe_burden"],
                gate_open_patterns["correction_pressure"],
            ),
            "residual_shell_alpha_mass_vs_improve": safe_corr(
                gate_open_patterns["residual_shell_alpha_mass"],
                gate_open_patterns["improve"],
            ),
            "residual_shell_alpha_mass_vs_correction_pressure": safe_corr(
                gate_open_patterns["residual_shell_alpha_mass"],
                gate_open_patterns["correction_pressure"],
            ),
        },
        "fan_vs_core_rule": {
            "threshold": float(fan_threshold),
            "calibration_error": float(fan_cal_error),
            "calibration_fire_rate": float(fan_fire_rate),
            "holdout_error": float(
                mean_or_nan(
                    fan_rule_df.loc[
                        fan_rule_df["split"] == "holdout", "fan_rule_output_abs_error"
                    ]
                    .astype(float)
                    .tolist()
                )
            ),
            "holdout_fire_rate": float(
                mean_or_nan(
                    fan_rule_df.loc[
                        fan_rule_df["split"] == "holdout", "fan_rule_fire_flag"
                    ]
                    .astype(float)
                    .tolist()
                )
            ),
            "confirmation_error": float(
                mean_or_nan(
                    fan_rule_df.loc[
                        fan_rule_df["split"] == "confirmation",
                        "fan_rule_output_abs_error",
                    ]
                    .astype(float)
                    .tolist()
                )
            ),
            "confirmation_fire_rate": float(
                mean_or_nan(
                    fan_rule_df.loc[
                        fan_rule_df["split"] == "confirmation", "fan_rule_fire_flag"
                    ]
                    .astype(float)
                    .tolist()
                )
            ),
        },
        "poison_heuristic_shift": {
            "old_candidate_poison_score_vs_improve": safe_corr(
                old_poison_join["old_poison_score"], old_poison_join["improve"]
            ),
            "old_candidate_poison_score_vs_correction_pressure": safe_corr(
                old_poison_join["old_poison_score"],
                old_poison_join["correction_pressure"],
            ),
            "new_residual_shell_alpha_mass_vs_improve": safe_corr(
                gate_open_patterns["residual_shell_alpha_mass"],
                gate_open_patterns["improve"],
            ),
            "new_residual_shell_alpha_mass_vs_correction_pressure": safe_corr(
                gate_open_patterns["residual_shell_alpha_mass"],
                gate_open_patterns["correction_pressure"],
            ),
        },
        "split_summary": split_summary,
        "condition_summary": condition_summary,
    }

    write_csv(
        os.path.join(OUTPUT_DIR, "candidate_atlas_archetype_rows.csv"),
        archetype_rows.to_dict(orient="records"),
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "candidate_atlas_trial_patterns.csv"),
        trial_patterns.to_dict(orient="records"),
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "candidate_atlas_pattern_split_summary.csv"),
        split_summary,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "candidate_atlas_pattern_condition_summary.csv"),
        condition_summary,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "candidate_atlas_fan_rule_rows.csv"),
        fan_rule_df.to_dict(orient="records"),
    )

    with open(
        os.path.join(OUTPUT_DIR, "candidate_atlas_pattern_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary_payload, handle, indent=2)

    plot_condition_masses(
        os.path.join(FIGURE_DIR, "candidate_atlas_archetype_mass_by_condition.png"),
        condition_summary,
    )
    plot_fan_vs_core_scatter(
        os.path.join(FIGURE_DIR, "candidate_atlas_fan_vs_core_scatter.png"),
        gate_open_patterns,
    )


if __name__ == "__main__":
    main()
