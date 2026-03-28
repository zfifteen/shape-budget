"""
Visualize the pose-free alpha phase insight from repo experiment outputs.

The plots in this script demonstrate a sharper reading of the hard branch:

1. pre-anchor alpha ambiguity is not the final object
2. many wide alpha families collapse after backbone anchoring
3. a smaller subset stays wide even after anchoring
4. ambiguity load and entropy opportunity are different control axes

Outputs are written to ./plots/insights/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MPL_CACHE_DIR = Path("/tmp") / "matplotlib-codex-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 240,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.family": "sans-serif",
    }
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "plots" / "insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE_ORDER = ["low_ambiguity", "gauge_broad", "bundle_broad"]
PHASE_LABELS = {
    "low_ambiguity": "Low ambiguity",
    "gauge_broad": "Gauge-broad",
    "bundle_broad": "Bundle-broad",
}
PHASE_COLORS = {
    "low_ambiguity": "#7a8f99",
    "gauge_broad": "#2a9d8f",
    "bundle_broad": "#c8553d",
}
CONDITION_MARKERS = {
    "sparse_full_noisy": "o",
    "sparse_partial_high_noise": "s",
}
GATE_COLORS = {
    "both_closed": "#adb5bd",
    "entropy_only": "#f4a261",
    "ambiguity_only": "#457b9d",
    "both_open": "#2a9d8f",
}


def load_thresholds() -> tuple[float, float, float]:
    obs_summary = json.loads(
        (ROOT / "experiments/pose-anisotropy-interventions/backbone-observability-gate/outputs/backbone_observability_gate_summary.json").read_text()
    )
    shadow_summary = json.loads(
        (ROOT / "experiments/pose-anisotropy-interventions/ambiguity-gated-bank-ensemble-shadow/outputs/ambiguity_gated_bank_ensemble_shadow_summary.json").read_text()
    )

    std_threshold = None
    metric_rows = obs_summary["summary"]["threshold_rule"]["metrics"]
    for row in metric_rows:
        if row["metric"] == "mean_anchored_alpha_log_std":
            std_threshold = float(row["threshold"])
            break
    if std_threshold is None:
        raise RuntimeError("Could not find the anchored-std threshold in the observability summary.")

    ambiguity_threshold = float(shadow_summary["ambiguity_gate_threshold"])
    entropy_threshold = float(shadow_summary["entropy_gate_threshold"])
    return ambiguity_threshold, std_threshold, entropy_threshold


def load_phase_dataframe() -> pd.DataFrame:
    keys = ["split", "observation_seed", "condition", "geometry_skew_bin"]
    obs = pd.read_csv(
        ROOT / "experiments/pose-anisotropy-interventions/backbone-observability-gate/outputs/backbone_observability_gate_trials.csv"
    )
    atlas = pd.read_csv(
        ROOT / "experiments/pose-anisotropy-diagnostics/candidate-atlas-instrumentation/outputs/candidate_atlas_trial_summary.csv"
    )
    patterns = pd.read_csv(
        ROOT / "experiments/pose-anisotropy-diagnostics/candidate-atlas-pattern-mining/outputs/candidate_atlas_trial_patterns.csv"
    )

    joined = (
        obs.merge(atlas, on=keys, suffixes=("_obs", "_atlas"))
        .merge(
            patterns[keys + ["fan_vs_core", "useful_structure_ratio", "residual_shell_alpha_mass"]],
            on=keys,
            how="left",
        )
        .copy()
    )
    return joined


def assign_phase(df: pd.DataFrame, ambiguity_threshold: float, std_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["phase"] = "low_ambiguity"
    high_ambiguity = out["mean_ambiguity_ratio_obs"] >= ambiguity_threshold
    out.loc[high_ambiguity, "phase"] = "gauge_broad"
    out.loc[high_ambiguity & (out["mean_anchored_alpha_log_std"] >= std_threshold), "phase"] = "bundle_broad"
    return out


def phase_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("phase", as_index=False)
        .agg(
            count=("phase", "size"),
            point_recoverable_rate=("alpha_point_recoverable_flag", "mean"),
            anchored_beats_best_rate=("anchored_beats_best_flag", "mean"),
            mean_best_span=("best_alpha_bank_log_span", "mean"),
            mean_anchored_span=("anchored_alpha_bank_log_span", "mean"),
            mean_alpha_gain=("alpha_abs_error_gain", "mean"),
            mean_candidate_count=("mean_candidate_count", "mean"),
            mean_compression_load=("compression_load", "mean"),
            mean_fan_vs_core=("fan_vs_core", "mean"),
            mean_entropy=("mean_best_entropy", "mean"),
            mean_ambiguity=("mean_ambiguity_ratio_obs", "mean"),
            mean_anchored_std=("mean_anchored_alpha_log_std", "mean"),
        )
        .copy()
    )
    summary["phase"] = pd.Categorical(summary["phase"], categories=PHASE_ORDER, ordered=True)
    summary = summary.sort_values("phase").reset_index(drop=True)
    summary["span_collapse_factor"] = summary["mean_best_span"] / np.maximum(summary["mean_anchored_span"], 1.0e-9)
    return summary


def load_fresh_gate_dataframe(
    phase_df: pd.DataFrame,
    ambiguity_threshold: float,
    std_threshold: float,
    entropy_threshold: float,
) -> pd.DataFrame:
    keys = ["split", "observation_seed", "condition", "geometry_skew_bin"]
    fresh = pd.read_csv(
        ROOT / "experiments/pose-anisotropy-interventions/ambiguity-gated-bank-ensemble-shadow/outputs/ambiguity_gated_bank_ensemble_shadow_trials.csv"
    )
    fresh = fresh[fresh["split"].isin(["holdout", "confirmation"])].copy()
    merged = fresh.merge(
        phase_df[
            keys
            + [
                "phase",
                "mean_ambiguity_ratio_obs",
                "mean_anchored_alpha_log_std",
                "compression_load",
                "fan_vs_core",
                "useful_structure_ratio",
            ]
        ],
        on=keys,
        how="left",
    )
    merged["ambiguity_high"] = merged["mean_ambiguity_ratio_obs"] >= ambiguity_threshold
    merged["entropy_high"] = merged["entropy_gate_value"] >= entropy_threshold
    merged["std_high"] = merged["mean_anchored_alpha_log_std"] >= std_threshold
    merged["oracle_gain_vs_default"] = merged["default_alpha_error"] - merged["oracle4_alpha_error"]
    merged["entropy_gain_vs_default"] = merged["default_alpha_error"] - merged["entropy_chosen_alpha_error"]
    merged["ambiguity_gain_vs_default"] = merged["default_alpha_error"] - merged["ambiguity_chosen_alpha_error"]

    gate_state = []
    for _, row in merged.iterrows():
        if row["entropy_high"] and row["ambiguity_high"]:
            gate_state.append("both_open")
        elif row["entropy_high"]:
            gate_state.append("entropy_only")
        elif row["ambiguity_high"]:
            gate_state.append("ambiguity_only")
        else:
            gate_state.append("both_closed")
    merged["gate_state"] = gate_state
    return merged


def add_phase_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PHASE_COLORS[phase],
            markeredgecolor="black",
            markersize=9,
            label=PHASE_LABELS[phase],
        )
        for phase in PHASE_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, title="Phase")


def plot_phase_dashboard(
    phase_df: pd.DataFrame,
    summary: pd.DataFrame,
    ambiguity_threshold: float,
    std_threshold: float,
) -> Path:
    fig = plt.figure(figsize=(16.5, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])

    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_rates = fig.add_subplot(grid[0, 1])
    ax_spans = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])

    for condition, marker in CONDITION_MARKERS.items():
        subset = phase_df[phase_df["condition"] == condition]
        colors = subset["phase"].map(PHASE_COLORS)
        sizes = 52 + 2.0 * subset["mean_candidate_count"]
        edgecolors = np.where(subset["alpha_point_recoverable_flag"] == 1, "#111111", "#f8f9fa")
        linewidths = np.where(subset["alpha_point_recoverable_flag"] == 1, 1.2, 0.8)
        ax_scatter.scatter(
            subset["mean_ambiguity_ratio_obs"],
            subset["mean_anchored_alpha_log_std"],
            s=sizes,
            c=colors,
            marker=marker,
            alpha=0.88,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )

    ax_scatter.axvline(ambiguity_threshold, color="#33415c", linestyle="--", linewidth=1.5, alpha=0.9)
    ax_scatter.axhline(std_threshold, color="#7f5539", linestyle="--", linewidth=1.5, alpha=0.9)
    ax_scatter.text(
        ambiguity_threshold + 0.014,
        phase_df["mean_anchored_alpha_log_std"].max() * 0.98,
        f"ambiguity gate\n{ambiguity_threshold:.3f}",
        fontsize=10,
        va="top",
        color="#33415c",
    )
    ax_scatter.text(
        phase_df["mean_ambiguity_ratio_obs"].min() + 0.01,
        std_threshold + 0.01,
        f"anchored std gate\n{std_threshold:.3f}",
        fontsize=10,
        va="bottom",
        color="#7f5539",
    )
    ax_scatter.text(
        ambiguity_threshold + 0.02,
        std_threshold * 0.62,
        "Gauge-broad:\nwide before anchoring,\nmostly pointable after",
        fontsize=10.5,
        color="#155d55",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#edf8f6", edgecolor="#7cc6ba", alpha=0.96),
    )
    ax_scatter.text(
        ambiguity_threshold + 0.05,
        std_threshold + 0.04,
        "Bundle-broad:\nwide before and after anchoring,\npoint forcing usually fails",
        fontsize=10.5,
        color="#7f2f1d",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdf0ed", edgecolor="#d48873", alpha=0.96),
    )
    ax_scatter.text(
        phase_df["mean_ambiguity_ratio_obs"].min() + 0.02,
        std_threshold * 0.5,
        "Low ambiguity:\nalready narrow",
        fontsize=10.0,
        color="#495057",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f1f3f5", edgecolor="#ced4da", alpha=0.96),
    )

    condition_handles = [
        Line2D([0], [0], marker=marker, color="#343a40", linestyle="none", markersize=9, label=condition.replace("_", " "))
        for condition, marker in CONDITION_MARKERS.items()
    ]
    add_phase_legend(ax_scatter)
    extra_legend = ax_scatter.legend(handles=condition_handles, loc="lower right", frameon=True, title="Condition")
    ax_scatter.add_artist(extra_legend)
    ax_scatter.set_title("A. Alpha Changes Type After Backbone Anchoring")
    ax_scatter.set_xlabel("Pre-anchor ambiguity ratio")
    ax_scatter.set_ylabel("Post-anchor alpha log std")

    x = np.arange(len(summary))
    width = 0.36
    ax_rates.bar(
        x - width / 2,
        summary["point_recoverable_rate"],
        width=width,
        color=[PHASE_COLORS[p] for p in summary["phase"]],
        alpha=0.92,
        label="Point recoverable",
    )
    ax_rates.bar(
        x + width / 2,
        summary["anchored_beats_best_rate"],
        width=width,
        color="#264653",
        alpha=0.72,
        label="Anchored beats best",
    )
    for idx, row in summary.iterrows():
        ax_rates.text(
            idx - width / 2,
            row["point_recoverable_rate"] + 0.03,
            f"{row['point_recoverable_rate']:.0%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        ax_rates.text(
            idx + width / 2,
            row["anchored_beats_best_rate"] + 0.03,
            f"{row['anchored_beats_best_rate']:.0%}",
            ha="center",
            fontsize=10,
            color="#1b263b",
        )
        ax_rates.text(
            idx,
            0.03,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#212529",
        )
    ax_rates.set_ylim(0.0, 1.08)
    ax_rates.set_xticks(x, [PHASE_LABELS[p] for p in summary["phase"]], rotation=0)
    ax_rates.set_title("B. Gauge-Broad Cases Mostly Become Pointable")
    ax_rates.set_ylabel("Rate")
    ax_rates.legend(loc="upper right", frameon=True)

    span_positions = np.arange(len(summary))
    width = 0.34
    ax_spans.bar(
        span_positions - width / 2,
        summary["mean_best_span"],
        width=width,
        color="#e76f51",
        alpha=0.84,
        label="Best-candidate bank span",
    )
    ax_spans.bar(
        span_positions + width / 2,
        summary["mean_anchored_span"],
        width=width,
        color="#2a9d8f",
        alpha=0.84,
        label="Anchored bank span",
    )
    for idx, row in summary.iterrows():
        ax_spans.text(
            idx,
            max(row["mean_best_span"], row["mean_anchored_span"]) + 0.022,
            f"{row['span_collapse_factor']:.1f}x collapse",
            ha="center",
            fontsize=10,
            color="#343a40",
        )
    ax_spans.set_xticks(span_positions, [PHASE_LABELS[p] for p in summary["phase"]], rotation=0)
    ax_spans.set_ylabel("Mean cross-bank alpha log span")
    ax_spans.set_title("C. Anchoring Collapses Most Width, But Not All Width")
    ax_spans.legend(loc="upper left", frameon=True)

    heat_source = summary.set_index("phase")[
        ["mean_candidate_count", "mean_compression_load", "mean_fan_vs_core", "mean_entropy"]
    ].rename(
        columns={
            "mean_candidate_count": "candidate count",
            "mean_compression_load": "compression load",
            "mean_fan_vs_core": "fan/core",
            "mean_entropy": "entropy",
        }
    )
    normalized = heat_source.copy()
    for column in normalized.columns:
        col = normalized[column]
        normalized[column] = (col - col.min()) / max(col.max() - col.min(), 1.0e-9)
    sns.heatmap(
        normalized,
        ax=ax_heat,
        cmap=sns.color_palette(["#edf6f9", "#83c5be", "#ffddd2", "#e76f51"], as_cmap=True),
        cbar=False,
        linewidths=1.0,
        linecolor="#ffffff",
        annot=heat_source.round(2),
        fmt="",
        annot_kws={"fontsize": 10},
    )
    ax_heat.set_title("D. Bundle-Broad Means Bigger Families, Not Bigger Compression")
    ax_heat.set_xlabel("Phase mean")
    ax_heat.set_ylabel("")
    ax_heat.set_yticklabels([PHASE_LABELS[p] for p in heat_source.index], rotation=0)

    fig.suptitle(
        "Pose-Free Alpha Phase Map\n"
        "Wide alpha families split into gauge-broad and bundle-broad regimes after backbone anchoring",
        fontsize=17,
        fontweight="bold",
    )

    output_path = OUTPUT_DIR / "alpha_phase_map_dashboard.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_gate_axes(
    fresh_df: pd.DataFrame,
    ambiguity_threshold: float,
    std_threshold: float,
    entropy_threshold: float,
) -> Path:
    fig = plt.figure(figsize=(15.8, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    marker_map = {False: "o", True: "^"}
    for std_high, marker in marker_map.items():
        subset = fresh_df[fresh_df["std_high"] == std_high]
        colors = subset["gate_state"].map(GATE_COLORS)
        sizes = 75 + 1500.0 * np.clip(subset["oracle_gain_vs_default"], 0.0, None)
        ax_scatter.scatter(
            subset["mean_ambiguity_ratio_obs"],
            subset["entropy_gate_value"],
            s=sizes,
            c=colors,
            marker=marker,
            alpha=0.88,
            edgecolors="#212529",
            linewidths=0.8,
        )

    ax_scatter.axvline(ambiguity_threshold, color="#457b9d", linestyle="--", linewidth=1.5, alpha=0.9)
    ax_scatter.axhline(entropy_threshold, color="#f4a261", linestyle="--", linewidth=1.5, alpha=0.9)
    ax_scatter.text(
        ambiguity_threshold + 0.014,
        fresh_df["entropy_gate_value"].max() * 0.98,
        f"ambiguity gate\n{ambiguity_threshold:.3f}",
        fontsize=10,
        va="top",
        color="#457b9d",
    )
    ax_scatter.text(
        fresh_df["mean_ambiguity_ratio_obs"].min() + 0.01,
        entropy_threshold + 0.01,
        f"entropy gate\n{entropy_threshold:.3f}",
        fontsize=10,
        va="bottom",
        color="#b86a16",
    )
    ax_scatter.text(
        ambiguity_threshold + 0.02,
        entropy_threshold + 0.05,
        "Most action sits here:\nboth ambiguity and entropy high",
        fontsize=10.5,
        color="#155d55",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#edf8f6", edgecolor="#7cc6ba", alpha=0.96),
    )
    ax_scatter.text(
        ambiguity_threshold + 0.02,
        entropy_threshold - 0.12,
        "Ambiguity-only corner:\nreal load, but little chooser upside",
        fontsize=10.0,
        color="#2b4c7e",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef4fb", edgecolor="#8db1d5", alpha=0.96),
    )
    ax_scatter.set_title("A. Entropy And Ambiguity Are Different Control Axes")
    ax_scatter.set_xlabel("Pre-anchor ambiguity ratio")
    ax_scatter.set_ylabel("Dense-joint entropy")

    gate_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="#212529", markersize=9, label=label.replace("_", " "))
        for label, color in GATE_COLORS.items()
    ]
    std_handles = [
        Line2D([0], [0], marker=marker_map[False], color="#212529", linestyle="none", markersize=8, label=f"anchored std < {std_threshold:.3f}"),
        Line2D([0], [0], marker=marker_map[True], color="#212529", linestyle="none", markersize=8, label=f"anchored std >= {std_threshold:.3f}"),
    ]
    legend1 = ax_scatter.legend(handles=gate_handles, loc="upper left", frameon=True, title="Gate state")
    ax_scatter.add_artist(legend1)
    ax_scatter.legend(handles=std_handles, loc="lower right", frameon=True, title="Post-anchor phase")

    quadrant_summary = (
        fresh_df.groupby(["entropy_high", "ambiguity_high"], as_index=False)
        .agg(
            count=("gate_state", "size"),
            mean_oracle_gain=("oracle_gain_vs_default", "mean"),
            mean_anchored_std=("mean_anchored_alpha_log_std", "mean"),
        )
        .copy()
    )
    quadrant_summary["row"] = quadrant_summary["entropy_high"].map({False: 1, True: 0})
    quadrant_summary["col"] = quadrant_summary["ambiguity_high"].map({False: 0, True: 1})
    heat = np.full((2, 2), np.nan)
    annot = np.empty((2, 2), dtype=object)
    for _, row in quadrant_summary.iterrows():
        heat[int(row["row"]), int(row["col"])] = float(row["mean_oracle_gain"])
        annot[int(row["row"]), int(row["col"])] = (
            f"n={int(row['count'])}\n"
            f"oracle gain {row['mean_oracle_gain']:.3f}\n"
            f"anchored std {row['mean_anchored_std']:.3f}"
        )
    sns.heatmap(
        heat,
        ax=ax_heat,
        cmap=sns.color_palette(["#f1faee", "#a8dadc", "#2a9d8f", "#1d6f63"], as_cmap=True),
        cbar=False,
        linewidths=1.2,
        linecolor="#ffffff",
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 10},
        vmin=np.nanmin(heat),
        vmax=np.nanmax(heat),
    )
    ax_heat.set_title("B. The Best Gains Need Both Axes, Not One Axis")
    ax_heat.set_xticklabels(["ambiguity low", "ambiguity high"], rotation=0)
    ax_heat.set_yticklabels(["entropy high", "entropy low"], rotation=0)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")

    fig.suptitle(
        "Fresh-Block Gate Control Map\n"
        "Ambiguity measures structural load; entropy measures whether richer chooser freedom can pay off",
        fontsize=16,
        fontweight="bold",
    )

    output_path = OUTPUT_DIR / "alpha_gate_control_axes.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    ambiguity_threshold, std_threshold, entropy_threshold = load_thresholds()
    phase_df = assign_phase(load_phase_dataframe(), ambiguity_threshold, std_threshold)
    summary = phase_summary_table(phase_df)
    fresh_df = load_fresh_gate_dataframe(phase_df, ambiguity_threshold, std_threshold, entropy_threshold)

    phase_plot = plot_phase_dashboard(phase_df, summary, ambiguity_threshold, std_threshold)
    gate_plot = plot_gate_axes(fresh_df, ambiguity_threshold, std_threshold, entropy_threshold)

    print(f"Saved {phase_plot}")
    print(f"Saved {gate_plot}")


if __name__ == "__main__":
    main()
