"""
Experiment 3: hyperbola twin for Shape Budget.

This experiment tests the fixed-difference twin of the ellipse experiment.

Instead of the constant-sum rule
    r1 + r2 = 2a
we use the constant-difference rule
    |r1 - r2| = 2a

with focal separation 2c and c > a.

The clean bounded control ratio is
    lambda = a / c = fixed difference / focal separation

which lies in (0, 1). Standard hyperbolic eccentricity is its reciprocal:
    e_h = c / a = 1 / lambda.

Because the hyperbola is open, all branch comparisons are done on a shared
truncated hyperbolic-parameter window.
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

U_MAX = 1.8
BRANCH_SAMPLE_COUNT = 500


@dataclass
class HyperbolaRow:
    lambda_ratio: float
    c_scale: float
    a_deficit: float
    b_residue: float
    hyperbolic_eccentricity: float
    normalized_vertex: float
    normalized_openness: float
    asymptote_slope: float
    normalized_vertex_curvature: float
    max_equation_residual: float
    rms_equation_residual: float


def hyperbola_parameters(c: float, lambda_ratio: float) -> tuple[float, float, float]:
    a = lambda_ratio * c
    b = math.sqrt(max(c * c - a * a, 0.0))
    return a, b, c


def hyperbola_branch_from_process(c: float, lambda_ratio: float, sample_count: int = BRANCH_SAMPLE_COUNT, u_max: float = U_MAX) -> np.ndarray:
    a, _, c = hyperbola_parameters(c, lambda_ratio)
    u = np.linspace(0.0, u_max, sample_count)
    s = c * np.cosh(u)
    d = 2.0 * c
    branch: list[tuple[float, float]] = []
    for baseline in s:
        r_left = baseline + a
        r_right = baseline - a
        x_left = (r_left * r_left - r_right * r_right + d * d) / (2.0 * d)
        y_sq = max(r_left * r_left - x_left * x_left, 0.0)
        y = math.sqrt(y_sq)
        x = x_left - c
        branch.append((x, y))

    upper = np.array(branch)
    lower = upper[::-1].copy()
    lower[:, 1] *= -1.0
    return np.vstack([upper, lower])


def full_hyperbola_process_points(c: float, lambda_ratio: float, sample_count: int = BRANCH_SAMPLE_COUNT, u_max: float = U_MAX) -> np.ndarray:
    right = hyperbola_branch_from_process(c, lambda_ratio, sample_count=sample_count, u_max=u_max)
    left = right.copy()
    left[:, 0] *= -1.0
    return np.vstack([left, right])


def analytic_hyperbola_points(c: float, lambda_ratio: float, sample_count: int = BRANCH_SAMPLE_COUNT, u_max: float = U_MAX) -> np.ndarray:
    a, b, _ = hyperbola_parameters(c, lambda_ratio)
    u = np.linspace(0.0, u_max, sample_count)
    x = a * np.cosh(u)
    y = b * np.sinh(u)
    upper_right = np.column_stack([x, y])
    lower_right = np.column_stack([x[::-1], -y[::-1]])
    right = np.vstack([upper_right, lower_right])
    left = right.copy()
    left[:, 0] *= -1.0
    return np.vstack([left, right])


def hyperbola_equation_residual(points: np.ndarray, a: float, b: float) -> np.ndarray:
    return (points[:, 0] ** 2) / (a * a) - (points[:, 1] ** 2) / (b * b) - 1.0


def make_rows(lambda_values: np.ndarray, c_values: list[float]) -> list[HyperbolaRow]:
    rows: list[HyperbolaRow] = []
    for lambda_ratio in lambda_values:
        for c in c_values:
            a, b, _ = hyperbola_parameters(c, float(lambda_ratio))
            points = full_hyperbola_process_points(c, float(lambda_ratio))
            residuals = hyperbola_equation_residual(points, a, b)
            rows.append(
                HyperbolaRow(
                    lambda_ratio=float(lambda_ratio),
                    c_scale=float(c),
                    a_deficit=float(a),
                    b_residue=float(b),
                    hyperbolic_eccentricity=float(c / a),
                    normalized_vertex=float(a / c),
                    normalized_openness=float(b / c),
                    asymptote_slope=float(b / a),
                    normalized_vertex_curvature=float((c * a) / (b * b)),
                    max_equation_residual=float(np.max(np.abs(residuals))),
                    rms_equation_residual=float(np.sqrt(np.mean(residuals**2))),
                )
            )
    return rows


def scale_collapse_errors(lambda_values: np.ndarray, c_values: list[float], sample_count: int = BRANCH_SAMPLE_COUNT, u_max: float = U_MAX) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for lambda_ratio in lambda_values:
        normalized_sets = {}
        for c in c_values:
            points = full_hyperbola_process_points(c, float(lambda_ratio), sample_count=sample_count, u_max=u_max) / c
            normalized_sets[c] = points
        for c1, c2 in combinations(c_values, 2):
            delta = normalized_sets[c1] - normalized_sets[c2]
            distances = np.sqrt(np.sum(delta * delta, axis=1))
            rows.append(
                {
                    "lambda_ratio": float(lambda_ratio),
                    "c1": float(c1),
                    "c2": float(c2),
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
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.8), constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.26, wspace=0.22)
    gallery_lambda = [0.15, 0.40, 0.70, 0.92]

    for idx, (ax, lambda_ratio) in enumerate(zip(axes.ravel(), gallery_lambda)):
        a, b, c = hyperbola_parameters(1.0, lambda_ratio)
        process_points = full_hyperbola_process_points(1.0, lambda_ratio, sample_count=420, u_max=U_MAX)
        analytic_points = analytic_hyperbola_points(1.0, lambda_ratio, sample_count=420, u_max=U_MAX)
        ax.scatter(process_points[:, 0], process_points[:, 1], s=7, color="#264653", alpha=0.50, label="difference-process locus")
        ax.plot(analytic_points[:, 0], analytic_points[:, 1], color="#e76f51", lw=2.1, label="analytic hyperbola")
        ax.scatter([-c, c], [0, 0], color="#d62828", s=28, zorder=4)
        ax.set_title(f"lambda = {lambda_ratio:.2f}, openness = {b / c:.3f}")
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x / c")
        ax.set_ylabel("y / c")
        residual = hyperbola_equation_residual(process_points, a, b)
        ax.text(
            0.03,
            0.94,
            f"max |residual| = {np.max(np.abs(residual)):.2e}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        if idx == 0:
            ax.legend(loc="lower left", frameon=True)

    fig.suptitle("Experiment 3A: Hyperbola Process Reconstruction", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scale_collapse(path: str) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.8, 6.4), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.24)

    fixed_lambda = 0.60
    c_values = [1.0, 2.5, 4.0]
    palette = ["#2a9d8f", "#e76f51", "#264653"]

    for c, color in zip(c_values, palette):
        pts = full_hyperbola_process_points(c, fixed_lambda, sample_count=300, u_max=1.45)
        ax_left.plot(pts[:, 0], pts[:, 1], color=color, lw=2.0, alpha=0.86)
        ax_left.scatter([-c, c], [0, 0], color=color, s=24)

    ax_left.set_title("Raw branches at fixed lambda = 0.60 and different c")
    ax_left.set_xlabel("Absolute x")
    ax_left.set_ylabel("Absolute y")
    ax_left.set_xlim(-10.0, 10.0)
    ax_left.set_ylim(-7.0, 7.0)
    ax_left.set_aspect("equal")

    compare_lambda = [0.20, 0.60, 0.92]
    colors = ["#2a9d8f", "#264653", "#8b1e3f"]
    for lambda_ratio, color in zip(compare_lambda, colors):
        pts = full_hyperbola_process_points(1.0, lambda_ratio, sample_count=350, u_max=1.6)
        ax_right.plot(pts[:, 0], pts[:, 1], color=color, lw=2.0, alpha=0.85)

    ax_right.set_title("After normalization by c, families separate only by lambda")
    ax_right.set_xlabel("Normalized x / c")
    ax_right.set_ylabel("Normalized y / c")
    ax_right.set_xlim(-3.4, 3.4)
    ax_right.set_ylim(-3.0, 3.0)
    ax_right.set_aspect("equal")
    ax_right.legend(
        handles=[plt.Line2D([0], [0], color=color, lw=3, label=f"lambda = {lam:.2f}") for lam, color in zip(compare_lambda, colors)],
        loc="lower left",
        frameon=True,
    )

    fig.suptitle("Experiment 3B: Scale Collapse", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_response_curves(path: str) -> None:
    lam = np.linspace(0.02, 0.98, 600)
    openness = np.sqrt(1.0 - lam**2)
    asymptote_slope = openness / lam
    vertex_curvature = lam / np.maximum(1.0 - lam**2, 1e-12)
    hyperbolic_ecc = 1.0 / lam

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11.8, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.9, hspace=0.35)

    ax_top.plot(lam, openness, lw=3, color="#2a9d8f", label="openness residue b / c")
    ax_top.plot(lam, asymptote_slope, lw=3, color="#457b9d", label="asymptote slope b / a")
    ax_top.fill_between(lam, 0.0, openness, color="#2a9d8f", alpha=0.12)
    ax_top.set_ylabel("Normalized quantity")
    ax_top.set_xlim(0.0, 1.0)
    ax_top.set_ylim(0.0, max(1.1, float(np.quantile(asymptote_slope, 0.95))))
    ax_top.legend(loc="upper right", frameon=True)
    ax_top.set_title("Openness falls as the fixed difference consumes more of the separation")

    ax_bottom.plot(lam, vertex_curvature, lw=3, color="#d62828", label="normalized vertex curvature c kappa_vertex")
    ax_bottom.plot(lam, hyperbolic_ecc, lw=2.7, color="#264653", label="standard hyperbolic eccentricity c / a")
    ax_bottom.set_xlabel("control ratio lambda = a / c")
    ax_bottom.set_ylabel("Response")
    ax_bottom.set_xlim(0.0, 1.0)
    ax_bottom.set_yscale("log")
    ax_bottom.legend(loc="upper left", frameon=True)
    ax_bottom.set_title("Local sharpness rises while standard eccentricity is just the reciprocal scale")

    fig.suptitle("Experiment 3C: Hyperbola Control Curves", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_map(path: str) -> None:
    separation = np.linspace(0.6, 4.0, 320)
    difference = np.linspace(0.03, 3.97, 300)
    D_grid, d_grid = np.meshgrid(difference, separation)
    valid = D_grid < d_grid
    lambda_field = np.divide(D_grid, d_grid, out=np.full_like(D_grid, np.nan), where=valid)
    openness_field = np.sqrt(np.maximum(1.0 - lambda_field**2, 0.0))

    fig, ax = plt.subplots(figsize=(10.8, 8.0), constrained_layout=False)
    fig.subplots_adjust(top=0.88)
    mesh = ax.pcolormesh(D_grid, d_grid, openness_field, cmap="viridis", shading="auto")
    contour_levels = [0.2, 0.4, 0.6, 0.8]
    contours = ax.contour(D_grid, d_grid, lambda_field, levels=contour_levels, colors="white", linewidths=1.1)
    ax.clabel(contours, fmt=lambda val: f"lambda={val:.1f}", fontsize=9, inline=True)
    ax.plot([0.0, 4.0], [0.0, 4.0], linestyle="--", color="#d62828", lw=2.0, label="D = d boundary")
    ax.fill_between(separation, 0.0, separation, color="#000000", alpha=0.08)
    ax.text(2.65, 1.3, "invalid\nD >= d", ha="center", va="center", color="#222222", fontsize=11)
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 4.0)
    ax.set_xlabel("fixed difference D = 2a")
    ax.set_ylabel("focal separation d = 2c")
    ax.set_title("Openness residue field in the deficit-spending regime")
    ax.legend(loc="upper left", frameon=True)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.9)
    cbar.set_label("normalized openness b / c")

    fig.suptitle("Experiment 3D: Hyperbola Phase Map", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    lambda_values = np.round(np.linspace(0.05, 0.95, 19), 4)
    c_values = [0.75, 1.0, 1.5, 2.5, 4.0]

    rows = make_rows(lambda_values, c_values)
    collapse_rows = scale_collapse_errors(lambda_values, c_values)

    metrics_path = os.path.join(OUTPUT_DIR, "hyperbola_twin_metrics.csv")
    collapse_path = os.path.join(OUTPUT_DIR, "hyperbola_twin_scale_collapse.csv")
    summary_path = os.path.join(OUTPUT_DIR, "hyperbola_twin_summary.json")

    write_csv(metrics_path, [asdict(row) for row in rows])
    write_csv(collapse_path, collapse_rows)

    summary = {
        "representation": {
            "control_ratio": "lambda = a / c",
            "hyperbolic_eccentricity_relation": "e_h = c / a = 1 / lambda",
            "branch_window_u_max": U_MAX,
        },
        "max_equation_residual": float(max(row.max_equation_residual for row in rows)),
        "max_rms_equation_residual": float(max(row.rms_equation_residual for row in rows)),
        "max_scale_collapse_error": float(max(row["max_collapse_error"] for row in collapse_rows)),
        "mean_scale_collapse_error": float(np.mean([row["mean_collapse_error"] for row in collapse_rows])),
        "scale_spread": {
            "normalized_openness": float(
                max(
                    np.ptp([row.normalized_openness for row in rows if abs(row.lambda_ratio - lambda_ratio) < 1e-12])
                    for lambda_ratio in lambda_values
                )
            ),
            "asymptote_slope": float(
                max(
                    np.ptp([row.asymptote_slope for row in rows if abs(row.lambda_ratio - lambda_ratio) < 1e-12])
                    for lambda_ratio in lambda_values
                )
            ),
            "normalized_vertex_curvature": float(
                max(
                    np.ptp([row.normalized_vertex_curvature for row in rows if abs(row.lambda_ratio - lambda_ratio) < 1e-12])
                    for lambda_ratio in lambda_values
                )
            ),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_process_reconstruction(os.path.join(FIGURE_DIR, "hyperbola_process_reconstruction.png"))
    plot_scale_collapse(os.path.join(FIGURE_DIR, "hyperbola_scale_collapse.png"))
    plot_response_curves(os.path.join(FIGURE_DIR, "hyperbola_response_curves.png"))
    plot_phase_map(os.path.join(FIGURE_DIR, "hyperbola_phase_map.png"))

    print("Hyperbola twin experiment complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
