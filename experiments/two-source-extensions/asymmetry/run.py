"""
Experiment 1: Unequal budget split / asymmetry test for Shape Budget.

Symmetric control-knob experiment:
    r1 + r2 = 2a

Asymmetric pilot model:
    w r1 + (1 - w) r2 = a

with w in (0, 1). The symmetric ellipse case is recovered at w = 0.5.
For w != 0.5 the locus becomes a Cartesian oval.

This experiment asks:
1. Does one-knob sufficiency fail once symmetry is broken?
2. Does a two-parameter family (e, w) recover normalized collapse?
"""

from __future__ import annotations

import sys
from pathlib import Path

_COMPAT_MODULES = Path(__file__).resolve().parents[3] / ".experiment_modules"
if str(_COMPAT_MODULES) not in sys.path:
    sys.path.insert(0, str(_COMPAT_MODULES))

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from itertools import combinations

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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


@dataclass
class AsymmetryRow:
    e: float
    w: float
    a_budget: float
    c: float
    x_left_over_a: float
    x_right_over_a: float
    y_max_over_a: float
    area_over_a2: float
    perimeter_over_2pi_a: float
    centroid_x_over_a: float


def sample_boundary(a_budget: float, e: float, w: float, sample_count: int = 1200) -> np.ndarray:
    c = e * a_budget
    d = 2.0 * c
    if d <= 0.0:
        raise ValueError("Asymmetry experiment assumes e > 0")

    diff_low = 1.0 - 2.0 * e * (1.0 - w)
    diff_high = 1.0 + 2.0 * e * (1.0 - w)
    s_min = diff_low
    s_max = diff_high
    if abs(1.0 - 2.0 * w) > 1e-12:
        sum_tangent = (2.0 * e * (1.0 - w) - 1.0) / (1.0 - 2.0 * w)
        if w < 0.5:
            s_min = max(s_min, sum_tangent)
        else:
            s_max = min(s_max, sum_tangent)

    s_min = max(0.0, s_min)
    s_max = min(1.0 / w, s_max)
    if s_min >= s_max:
        raise ValueError(f"Invalid sampling interval for e={e}, w={w}")

    upper_points: list[tuple[float, float]] = []
    s_values = np.linspace(s_min, s_max, sample_count)
    for s in s_values:
        r1 = a_budget * s
        r2 = a_budget * (1.0 - w * s) / (1.0 - w)
        if r2 < -1e-10:
            continue
        if d > r1 + r2 + 1e-10 or d < abs(r1 - r2) - 1e-10:
            continue

        x_local = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
        y_sq = r1 * r1 - x_local * x_local
        if y_sq < -1e-10:
            continue
        y = math.sqrt(max(y_sq, 0.0))
        x = -c + x_local
        upper_points.append((x, y))

    if len(upper_points) < 3:
        raise ValueError(f"Insufficient boundary points for e={e}, w={w}")

    upper = np.array(upper_points)
    lower = upper[::-1].copy()
    lower[:, 1] *= -1.0
    return np.vstack([upper, lower])


def closed_curve_metrics(points: np.ndarray) -> tuple[float, float, float]:
    closed = np.vstack([points, points[0]])
    dx = np.diff(closed[:, 0])
    dy = np.diff(closed[:, 1])
    perimeter = float(np.sum(np.sqrt(dx * dx + dy * dy)))

    x = closed[:, 0]
    y = closed[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * float(np.sum(cross))
    if abs(area) < 1e-15:
        centroid_x = 0.0
    else:
        centroid_x = float(np.sum((x[:-1] + x[1:]) * cross) / (6.0 * area))
    return abs(area), perimeter, centroid_x


def resample_closed_curve(points: np.ndarray, sample_count: int = 500) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    seg = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], sample_count, endpoint=False)
    x = np.interp(target, s, closed[:, 0])
    y = np.interp(target, s, closed[:, 1])
    return np.column_stack([x, y])


def pointwise_distance(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, float]:
    delta = points_a - points_b
    dist = np.sqrt(np.sum(delta * delta, axis=1))
    return float(np.mean(dist)), float(np.max(dist))


def metric_rows(e_values: np.ndarray, w_values: list[float], a_values: list[float]) -> list[AsymmetryRow]:
    rows: list[AsymmetryRow] = []
    for e in e_values:
        for w in w_values:
            for a_budget in a_values:
                c = e * a_budget
                points = sample_boundary(a_budget, float(e), float(w))
                area, perimeter, centroid_x = closed_curve_metrics(points)
                x_left = float(np.min(points[:, 0]))
                x_right = float(np.max(points[:, 0]))
                rows.append(
                    AsymmetryRow(
                        e=float(e),
                        w=float(w),
                        a_budget=float(a_budget),
                        c=float(c),
                        x_left_over_a=float(x_left / a_budget),
                        x_right_over_a=float(x_right / a_budget),
                        y_max_over_a=float(np.max(points[:, 1]) / a_budget),
                        area_over_a2=float(area / (a_budget * a_budget)),
                        perimeter_over_2pi_a=float(perimeter / (2.0 * math.pi * a_budget)),
                        centroid_x_over_a=float(centroid_x / a_budget),
                    )
                )
    return rows


def scale_collapse_rows(e_values: np.ndarray, w_values: list[float], a_values: list[float]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for e in e_values:
        for w in w_values:
            normalized = {}
            for a_budget in a_values:
                points = sample_boundary(a_budget, float(e), float(w)) / a_budget
                normalized[a_budget] = resample_closed_curve(points)
            for a1, a2 in combinations(a_values, 2):
                mean_dist, max_dist = pointwise_distance(normalized[a1], normalized[a2])
                rows.append(
                    {
                        "e": float(e),
                        "w": float(w),
                        "a1": float(a1),
                        "a2": float(a2),
                        "mean_collapse_error": mean_dist,
                        "max_collapse_error": max_dist,
                    }
                )
    return rows


def one_knob_failure_rows(e_values: np.ndarray, w_values: list[float], a_budget: float = 1.0) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for e in e_values:
        normalized = {}
        for w in w_values:
            points = sample_boundary(a_budget, float(e), float(w)) / a_budget
            normalized[w] = resample_closed_curve(points)
        for w1, w2 in combinations(w_values, 2):
            mean_dist, max_dist = pointwise_distance(normalized[w1], normalized[w2])
            rows.append(
                {
                    "e": float(e),
                    "w1": float(w1),
                    "w2": float(w2),
                    "mean_family_distance": mean_dist,
                    "max_family_distance": max_dist,
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


def plot_family_gallery(path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.9, hspace=0.28, wspace=0.22)

    top_e = 0.60
    top_ws = [0.30, 0.50, 0.70]
    bottom_w = 0.30
    bottom_es = [0.20, 0.60, 0.90]
    palette = ["#2a9d8f", "#264653", "#8d3149"]

    for ax, w, color in zip(axes[0], top_ws, palette):
        points = sample_boundary(1.0, top_e, w)
        ax.plot(points[:, 0], points[:, 1], color=color, lw=2.5)
        c = top_e
        ax.scatter([-c, c], [0, 0], color="#d62828", s=28)
        ax.set_title(f"Fixed e = {top_e:.2f}, w = {w:.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("x / a")
        ax.set_ylabel("y / a")

    for ax, e, color in zip(axes[1], bottom_es, palette):
        points = sample_boundary(1.0, e, bottom_w)
        ax.plot(points[:, 0], points[:, 1], color=color, lw=2.5)
        c = e
        ax.scatter([-c, c], [0, 0], color="#d62828", s=28)
        ax.set_title(f"Fixed w = {bottom_w:.2f}, e = {e:.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("x / a")
        ax.set_ylabel("y / a")

    fig.suptitle("Experiment 1A: Asymmetric Family Gallery", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_collapse(path: str) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 6.6), constrained_layout=False)
    fig.subplots_adjust(top=0.88, wspace=0.24)

    fixed_e = 0.60
    ws = [0.30, 0.50, 0.70]
    colors = ["#2a9d8f", "#264653", "#8d3149"]
    for w, color in zip(ws, colors):
        points = sample_boundary(1.0, fixed_e, w) / 1.0
        ax_left.plot(points[:, 0], points[:, 1], color=color, lw=2.5, label=f"w = {w:.2f}")
    ax_left.scatter([-fixed_e, fixed_e], [0, 0], color="#d62828", s=26)
    ax_left.set_title("One-knob failure at fixed e")
    ax_left.set_xlabel("Normalized x / a")
    ax_left.set_ylabel("Normalized y / a")
    ax_left.set_aspect("equal")
    ax_left.set_xlim(-1.45, 1.45)
    ax_left.set_ylim(-1.1, 1.1)
    ax_left.legend(loc="lower left", frameon=True)

    fixed_w = 0.70
    scales = [1.0, 2.5, 4.0]
    scale_colors = ["#2a9d8f", "#e76f51", "#264653"]
    for a_budget, color in zip(scales, scale_colors):
        points = sample_boundary(a_budget, fixed_e, fixed_w) / a_budget
        ax_right.plot(points[:, 0], points[:, 1], color=color, lw=2.3, label=f"a = {a_budget:.1f}")
    ax_right.scatter([-fixed_e, fixed_e], [0, 0], color="#d62828", s=26)
    ax_right.set_title("Two-knob collapse at fixed (e, w)")
    ax_right.set_xlabel("Normalized x / a")
    ax_right.set_ylabel("Normalized y / a")
    ax_right.set_aspect("equal")
    ax_right.set_xlim(-1.45, 1.45)
    ax_right.set_ylim(-1.1, 1.1)
    ax_right.legend(loc="lower left", frameon=True)

    fig.suptitle("Experiment 1B: Failure of One-Knob Sufficiency, Success of Two-Knob Collapse", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def heatmap_matrix(metric_rows: list[AsymmetryRow], metric_name: str, e_values: np.ndarray, w_values: list[float]) -> np.ndarray:
    metric_lookup = {(row.e, row.w): getattr(row, metric_name) for row in metric_rows if abs(row.a_budget - 1.0) < 1e-12}
    matrix = np.zeros((len(w_values), len(e_values)))
    for i, w in enumerate(w_values):
        for j, e in enumerate(e_values):
            matrix[i, j] = metric_lookup[(float(e), float(w))]
    return matrix


def plot_response_surfaces(path: str, metric_rows_list: list[AsymmetryRow], e_values: np.ndarray, w_values: list[float]) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.22)

    y_residue = heatmap_matrix(metric_rows_list, "y_max_over_a", e_values, w_values)
    centroid = heatmap_matrix(metric_rows_list, "centroid_x_over_a", e_values, w_values)

    sns.heatmap(
        y_residue,
        ax=ax_left,
        cmap="viridis",
        xticklabels=[f"{e:.2f}" for e in e_values],
        yticklabels=[f"{w:.2f}" for w in w_values],
        cbar_kws={"label": "max y / a"},
    )
    ax_left.set_title("Transverse residue surface")
    ax_left.set_xlabel("e")
    ax_left.set_ylabel("w")

    sns.heatmap(
        centroid,
        ax=ax_right,
        cmap="coolwarm",
        center=0.0,
        xticklabels=[f"{e:.2f}" for e in e_values],
        yticklabels=[f"{w:.2f}" for w in w_values],
        cbar_kws={"label": "centroid x / a"},
    )
    ax_right.set_title("Centroid shift surface")
    ax_right.set_xlabel("e")
    ax_right.set_ylabel("w")

    fig.suptitle("Experiment 1C: Two-Parameter Response Surfaces", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_error_summary(path: str, collapse_rows: list[dict[str, float]], family_rows: list[dict[str, float]], e_values: np.ndarray) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11.5, 8.8), constrained_layout=False)
    fig.subplots_adjust(top=0.9, hspace=0.34)

    collapse_by_e = []
    for e in e_values:
        vals = [row["max_collapse_error"] for row in collapse_rows if abs(row["e"] - float(e)) < 1e-12]
        collapse_by_e.append(max(vals))
    ax_top.plot(e_values, collapse_by_e, color="#2a9d8f", lw=3)
    ax_top.set_title("Scale-collapse residual across the asymmetric family")
    ax_top.set_xlabel("e")
    ax_top.set_ylabel("max normalized collapse error")
    ax_top.set_yscale("log")

    family_by_e = []
    for e in e_values:
        vals = [row["mean_family_distance"] for row in family_rows if abs(row["e"] - float(e)) < 1e-12]
        family_by_e.append(min(vals))
    ax_bottom.plot(e_values, family_by_e, color="#d62828", lw=3)
    ax_bottom.set_title("Minimum family distance across differing w at fixed e")
    ax_bottom.set_xlabel("e")
    ax_bottom.set_ylabel("min mean normalized family distance")

    fig.suptitle("Experiment 1D: Quantifying Two-Knob Sufficiency", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summarize(metric_rows_list: list[AsymmetryRow], collapse_rows: list[dict[str, float]], family_rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "num_metric_rows": len(metric_rows_list),
        "num_scale_collapse_rows": len(collapse_rows),
        "num_family_distance_rows": len(family_rows),
        "max_two_knob_scale_collapse_error": float(max(row["max_collapse_error"] for row in collapse_rows)),
        "mean_two_knob_scale_collapse_error": float(np.mean([row["mean_collapse_error"] for row in collapse_rows])),
        "min_one_knob_family_distance": float(min(row["mean_family_distance"] for row in family_rows)),
        "max_one_knob_family_distance": float(max(row["mean_family_distance"] for row in family_rows)),
    }


def main() -> None:
    e_values = np.round(np.linspace(0.10, 0.90, 17), 4)
    w_values = [0.30, 0.40, 0.50, 0.60, 0.70]
    a_values = [0.75, 1.0, 1.5, 2.5, 4.0]

    rows = metric_rows(e_values, w_values, a_values)
    collapse = scale_collapse_rows(e_values, w_values, a_values)
    family = one_knob_failure_rows(e_values, w_values, a_budget=1.0)

    metrics_path = os.path.join(OUTPUT_DIR, "asymmetry_metrics.csv")
    collapse_path = os.path.join(OUTPUT_DIR, "asymmetry_scale_collapse.csv")
    family_path = os.path.join(OUTPUT_DIR, "asymmetry_family_distances.csv")
    summary_path = os.path.join(OUTPUT_DIR, "asymmetry_summary.json")

    write_csv(metrics_path, [asdict(row) for row in rows])
    write_csv(collapse_path, collapse)
    write_csv(family_path, family)

    summary = summarize(rows, collapse, family)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_family_gallery(os.path.join(FIGURE_DIR, "asymmetry_family_gallery.png"))
    plot_collapse(os.path.join(FIGURE_DIR, "asymmetry_collapse.png"))
    plot_response_surfaces(os.path.join(FIGURE_DIR, "asymmetry_response_surfaces.png"), rows, e_values, w_values)
    plot_error_summary(os.path.join(FIGURE_DIR, "asymmetry_error_summary.png"), collapse, family, e_values)

    print("Asymmetry experiment complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
