"""
Research-grade control-knob experiment for the Shape Budget concept.

This script tests three claims about the normalized control ratio e = c / a:

1. Process reconstruction:
   The constant-sum circle construction numerically reconstructs the ellipse.
2. Scale collapse:
   For fixed e, normalized loci are invariant to absolute scale a.
3. Metric control:
   A family of normalized geometric observables are functions of e alone.

Outputs are written to ./experiment_outputs/
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse


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
OUTPUT_DIR = os.path.join(BASE_DIR, "experiment_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


@dataclass
class ExperimentRow:
    e: float
    a: float
    c: float
    b: float
    normalized_width: float
    normalized_area: float
    normalized_perimeter: float
    normalized_major_tip_curvature: float
    normalized_minor_tip_curvature: float
    max_equation_residual: float
    rms_equation_residual: float


def ellipse_parameters(a: float, e: float) -> tuple[float, float, float]:
    c = e * a
    b = math.sqrt(max(a * a - c * c, 0.0))
    return a, b, c


def normalized_radii_family(a: float, c: float, sample_count: int = 500) -> np.ndarray:
    if c == 0.0:
        return np.full(sample_count, a)
    s = np.linspace(-1.0, 1.0, sample_count)
    return a + c * s


def constant_sum_locus_points(a: float, e: float, sample_count: int = 500) -> np.ndarray:
    a, b, c = ellipse_parameters(a, e)
    if c == 0.0:
        theta = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
        return np.column_stack([a * np.cos(theta), b * np.sin(theta)])

    d = 2.0 * c
    upper_points: list[tuple[float, float]] = []
    for r1 in normalized_radii_family(a, c, sample_count):
        r2 = 2.0 * a - r1
        x_left = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
        y_sq = max(r1 * r1 - x_left * x_left, 0.0)
        y = math.sqrt(y_sq)
        x = x_left - c
        upper_points.append((x, y))

    upper = np.array(upper_points)
    lower = upper[::-1].copy()
    lower[:, 1] *= -1.0
    return np.vstack([upper, lower])


def ellipse_equation_residual(points: np.ndarray, a: float, b: float) -> np.ndarray:
    return (points[:, 0] ** 2) / (a * a) + (points[:, 1] ** 2) / (b * b) - 1.0


def approximate_perimeter(a: float, b: float) -> float:
    return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def make_rows(e_values: np.ndarray, a_values: list[float]) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    for e in e_values:
        for a in a_values:
            a, b, c = ellipse_parameters(a, e)
            points = constant_sum_locus_points(a, e)
            residuals = ellipse_equation_residual(points, a, b)
            rows.append(
                ExperimentRow(
                    e=float(e),
                    a=float(a),
                    c=float(c),
                    b=float(b),
                    normalized_width=float(b / a),
                    normalized_area=float(b / a),
                    normalized_perimeter=float(approximate_perimeter(a, b) / (2.0 * math.pi * a)),
                    normalized_major_tip_curvature=float((a * a) / (b * b)),
                    normalized_minor_tip_curvature=float(b / (1.0 * a)),
                    max_equation_residual=float(np.max(np.abs(residuals))),
                    rms_equation_residual=float(np.sqrt(np.mean(residuals**2))),
                )
            )
    return rows


def scale_collapse_errors(e_values: np.ndarray, a_values: list[float], sample_count: int = 500) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for e in e_values:
        normalized_sets = {}
        for a in a_values:
            points = constant_sum_locus_points(a, float(e), sample_count=sample_count) / a
            normalized_sets[a] = points
        for a1, a2 in combinations(a_values, 2):
            delta = normalized_sets[a1] - normalized_sets[a2]
            distances = np.sqrt(np.sum(delta * delta, axis=1))
            rows.append(
                {
                    "e": float(e),
                    "a1": float(a1),
                    "a2": float(a2),
                    "mean_collapse_error": float(np.mean(distances)),
                    "max_collapse_error": float(np.max(distances)),
                }
            )
    return rows


def write_csv(path: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_process_reconstruction(path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.26, wspace=0.22)
    gallery_e = [0.15, 0.40, 0.70, 0.92]

    for idx, (ax, e) in enumerate(zip(axes.ravel(), gallery_e)):
        a, b, c = ellipse_parameters(1.0, e)
        points = constant_sum_locus_points(1.0, e, sample_count=420)
        ax.scatter(points[:, 0], points[:, 1], s=7, color="#264653", alpha=0.55, label="circle-combination locus")
        ellipse = Ellipse((0.0, 0.0), 2 * a, 2 * b, edgecolor="#e76f51", facecolor="none", lw=2.5, label="analytic ellipse")
        ax.add_patch(ellipse)
        ax.scatter([-c, c], [0, 0], color="#d62828", s=28, zorder=4)
        ax.set_title(f"e = {e:.2f}, residue = {b / a:.3f}")
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12 if e < 0.85 else -0.45, 1.12 if e < 0.85 else 0.45)
        ax.set_aspect("equal")
        ax.set_xlabel("x / a")
        ax.set_ylabel("y / a")
        eq_res = ellipse_equation_residual(points, a, b)
        ax.text(
            0.03,
            0.94,
            f"max |residual| = {np.max(np.abs(eq_res)):.2e}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        if idx == 0:
            ax.legend(loc="lower left", frameon=True)

    fig.suptitle("Experiment 1: Process Reconstruction", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scale_collapse(path: str) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 6.6), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.22)

    fixed_e = 0.65
    scale_palette = ["#2a9d8f", "#e76f51", "#264653"]
    a_values = [1.0, 2.5, 4.0]

    for a, color in zip(a_values, scale_palette):
        _, b, c = ellipse_parameters(a, fixed_e)
        points = constant_sum_locus_points(a, fixed_e, sample_count=360)
        ax_left.scatter(points[:, 0], points[:, 1], s=7, color=color, alpha=0.35)
        ax_left.scatter([-c, c], [0, 0], color=color, s=28)
        ax_left.text(a + 0.12, b, f"a={a:.1f}", color=color, fontsize=10)

    ax_left.set_title("Raw loci at fixed e = 0.65 and different absolute scales")
    ax_left.set_xlabel("Absolute x")
    ax_left.set_ylabel("Absolute y")
    ax_left.set_aspect("equal")

    compare_e = [0.20, 0.65, 0.92]
    compare_colors = ["#2a9d8f", "#264653", "#8b1e3f"]
    for e, color in zip(compare_e, compare_colors):
        points = constant_sum_locus_points(1.0, e, sample_count=500)
        ax_right.scatter(points[:, 0], points[:, 1], s=8, color=color, alpha=0.22)
        _, _, c = ellipse_parameters(1.0, e)
        ax_right.scatter([-c, c], [0, 0], color="#d62828", s=24)

    ax_right.set_title("After normalization, families separate only by e")
    ax_right.set_xlabel("Normalized x / a")
    ax_right.set_ylabel("Normalized y / a")
    ax_right.set_xlim(-1.12, 1.12)
    ax_right.set_ylim(-1.12, 1.12)
    ax_right.set_aspect("equal")
    ax_right.legend(
        handles=[plt.Line2D([0], [0], color=color, lw=3, label=f"e = {e:.2f}") for e, color in zip(compare_e, compare_colors)],
        loc="lower left",
        frameon=True,
    )

    fig.suptitle("Experiment 2: Scale Collapse", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_response_curves(path: str) -> None:
    e = np.linspace(0.0, 0.995, 500)
    residue = np.sqrt(1.0 - e**2)
    perimeter_ratio = np.array([approximate_perimeter(1.0, float(b)) / (2.0 * math.pi) for b in residue])
    major_curvature = 1.0 / np.clip(1.0 - e**2, 1e-9, None)
    minor_curvature = residue

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11.5, 9.0), constrained_layout=False)
    fig.subplots_adjust(top=0.9, hspace=0.35)

    ax_top.plot(e, residue, lw=3, color="#2a9d8f", label="width residue b/a")
    ax_top.plot(e, perimeter_ratio, lw=3, color="#457b9d", label="normalized perimeter P/(2πa)")
    ax_top.fill_between(e, 0.0, residue, color="#2a9d8f", alpha=0.12)
    ax_top.set_ylabel("Normalized quantity")
    ax_top.set_xlim(0.0, 1.0)
    ax_top.set_ylim(0.0, 1.02)
    ax_top.legend(loc="upper right")

    ax_bottom.plot(e, major_curvature, lw=3, color="#d62828", label="a · κ_major-tip")
    ax_bottom.plot(e, minor_curvature, lw=3, color="#6a4c93", label="a · κ_minor-tip")
    ax_bottom.set_xlabel("Control knob e = c / a")
    ax_bottom.set_ylabel("Scale-normalized response")
    ax_bottom.set_xlim(0.0, 1.0)
    ax_bottom.set_ylim(0.0, 12.0)
    ax_bottom.legend(loc="upper left")

    fig.suptitle("Experiment 3: Metric Response Curves", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_map(path: str) -> None:
    d_vals = np.linspace(0.0, 10.0, 450)
    s_vals = np.linspace(0.2, 10.0, 450)
    d_grid, s_grid = np.meshgrid(d_vals, s_vals)
    e_grid = d_grid / s_grid
    residue_grid = np.where(e_grid <= 1.0, np.sqrt(np.clip(1.0 - e_grid**2, 0.0, 1.0)), np.nan)

    fig, ax = plt.subplots(figsize=(10.5, 8.2), constrained_layout=True)
    mesh = ax.pcolormesh(d_grid, s_grid, residue_grid, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    contours = ax.contour(
        d_grid,
        s_grid,
        np.where(e_grid <= 1.0, e_grid, np.nan),
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="white",
        linewidths=1.2,
    )
    ax.clabel(contours, inline=True, fmt=lambda value: f"e={value:.1f}", fontsize=10)
    ax.plot(d_vals, d_vals, linestyle="--", color="#111111", lw=1.4)
    ax.set_title("Experiment 4: Phase Map\nConstant-e rays organize the full separation-budget plane")
    ax.set_xlabel("Source separation d = 2c")
    ax.set_ylabel("Total reach budget S = 2a")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Width residue b / a")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[ExperimentRow], collapse_rows: list[dict[str, float]]) -> dict[str, float]:
    max_equation_residual = max(row.max_equation_residual for row in rows)
    rms_equation_residual = max(row.rms_equation_residual for row in rows)
    max_collapse_error = max(row["max_collapse_error"] for row in collapse_rows)
    mean_collapse_error = float(np.mean([row["mean_collapse_error"] for row in collapse_rows]))
    return {
        "num_metric_rows": len(rows),
        "num_collapse_rows": len(collapse_rows),
        "max_equation_residual": float(max_equation_residual),
        "max_rms_equation_residual": float(rms_equation_residual),
        "max_scale_collapse_error": float(max_collapse_error),
        "mean_scale_collapse_error": mean_collapse_error,
    }


def main() -> None:
    e_values = np.round(np.linspace(0.05, 0.95, 19), 4)
    a_values = [0.75, 1.0, 1.5, 2.5, 4.0]

    metric_rows = make_rows(e_values, a_values)
    collapse_rows = scale_collapse_errors(e_values, a_values)

    metric_path = os.path.join(OUTPUT_DIR, "control_knob_metrics.csv")
    collapse_path = os.path.join(OUTPUT_DIR, "control_knob_scale_collapse.csv")
    summary_path = os.path.join(OUTPUT_DIR, "control_knob_summary.json")

    write_csv(metric_path, [asdict(row) for row in metric_rows])
    write_csv(collapse_path, collapse_rows)

    summary = summarize(metric_rows, collapse_rows)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_process_reconstruction(os.path.join(FIGURE_DIR, "control_knob_process_reconstruction.png"))
    plot_scale_collapse(os.path.join(FIGURE_DIR, "control_knob_scale_collapse.png"))
    plot_response_curves(os.path.join(FIGURE_DIR, "control_knob_response_curves.png"))
    plot_phase_map(os.path.join(FIGURE_DIR, "control_knob_phase_map.png"))

    print("Control-knob experiment complete.")
    print(f"Metric rows: {len(metric_rows)}")
    print(f"Collapse rows: {len(collapse_rows)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
