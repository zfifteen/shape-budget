"""
Experiment 4: controlled anisotropy for Shape Budget.

This experiment studies the constant-sum two-source process under an
axis-aligned quadratic anisotropic metric

    d_alpha((x, y), (u, v)) = sqrt((x - u)^2 + alpha^2 (y - v)^2)

with sources placed on the x-axis.

This is a controlled anisotropy test:
- in raw Euclidean coordinates the normalized family depends on both e and alpha
- after whitening by (x, y) -> (x, alpha y), the family reduces exactly to the
  symmetric Euclidean control-knob case
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
from itertools import combinations, product

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
class AnisotropyRow:
    e: float
    alpha: float
    a: float
    c: float
    b: float
    raw_vertical_residue: float
    whitened_vertical_residue: float
    raw_major_tip_response: float
    whitened_major_tip_response: float
    max_equation_residual: float
    rms_equation_residual: float


def ellipse_parameters(a: float, e: float) -> tuple[float, float, float]:
    c = e * a
    b = math.sqrt(max(a * a - c * c, 0.0))
    return a, b, c


def constant_sum_anisotropic_points(a: float, e: float, alpha: float, sample_count: int = 500) -> np.ndarray:
    a, b, c = ellipse_parameters(a, e)
    if c == 0.0:
        theta = np.linspace(0.0, 2.0 * math.pi, 2 * sample_count, endpoint=False)
        return np.column_stack([a * np.cos(theta), (a / alpha) * np.sin(theta)])

    r1_values = np.linspace(a - c, a + c, sample_count)
    upper_points: list[tuple[float, float]] = []
    for r1 in r1_values:
        r2 = 2.0 * a - r1
        x = (r1 * r1 - r2 * r2) / (4.0 * c)
        y_sq_scaled = max(r1 * r1 - (x + c) * (x + c), 0.0)
        y = math.sqrt(y_sq_scaled) / alpha
        upper_points.append((x, y))

    upper = np.array(upper_points)
    lower = upper[::-1].copy()
    lower[:, 1] *= -1.0
    return np.vstack([upper, lower])


def anisotropic_equation_residual(points: np.ndarray, a: float, b: float, alpha: float) -> np.ndarray:
    return (points[:, 0] ** 2) / (a * a) + (alpha * points[:, 1]) ** 2 / (b * b) - 1.0


def whiten_points(points: np.ndarray, alpha: float) -> np.ndarray:
    out = points.copy()
    out[:, 1] *= alpha
    return out


def make_rows(e_values: np.ndarray, alpha_values: list[float], a_values: list[float]) -> list[AnisotropyRow]:
    rows: list[AnisotropyRow] = []
    for e, alpha, a in product(e_values, alpha_values, a_values):
        a, b, c = ellipse_parameters(float(a), float(e))
        points = constant_sum_anisotropic_points(a, float(e), float(alpha))
        residuals = anisotropic_equation_residual(points, a, b, float(alpha))
        rows.append(
            AnisotropyRow(
                e=float(e),
                alpha=float(alpha),
                a=float(a),
                c=float(c),
                b=float(b),
                raw_vertical_residue=float((b / alpha) / a),
                whitened_vertical_residue=float(b / a),
                raw_major_tip_response=float((alpha * alpha) * (a * a) / (b * b)),
                whitened_major_tip_response=float((a * a) / (b * b)),
                max_equation_residual=float(np.max(np.abs(residuals))),
                rms_equation_residual=float(np.sqrt(np.mean(residuals**2))),
            )
        )
    return rows


def raw_scale_collapse(e_values: np.ndarray, alpha_values: list[float], a_values: list[float], sample_count: int = 500) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for e, alpha in product(e_values, alpha_values):
        normalized_sets = {}
        for a in a_values:
            normalized_sets[a] = constant_sum_anisotropic_points(float(a), float(e), float(alpha), sample_count=sample_count) / a
        for a1, a2 in combinations(a_values, 2):
            delta = normalized_sets[a1] - normalized_sets[a2]
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            rows.append(
                {
                    "e": float(e),
                    "alpha": float(alpha),
                    "a1": float(a1),
                    "a2": float(a2),
                    "mean_collapse_error": float(np.mean(dist)),
                    "max_collapse_error": float(np.max(dist)),
                }
            )
    return rows


def raw_family_distances(e_values: np.ndarray, alpha_values: list[float], a: float = 1.0, sample_count: int = 500) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for e in e_values:
        normalized_sets = {
            alpha: constant_sum_anisotropic_points(a, float(e), float(alpha), sample_count=sample_count) / a
            for alpha in alpha_values
        }
        for alpha1, alpha2 in combinations(alpha_values, 2):
            delta = normalized_sets[alpha1] - normalized_sets[alpha2]
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            rows.append(
                {
                    "e": float(e),
                    "alpha1": float(alpha1),
                    "alpha2": float(alpha2),
                    "mean_family_distance": float(np.mean(dist)),
                    "max_family_distance": float(np.max(dist)),
                }
            )
    return rows


def whitened_collapse(e_values: np.ndarray, alpha_values: list[float], a_values: list[float], sample_count: int = 500) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for e in e_values:
        normalized_sets = {}
        for alpha, a in product(alpha_values, a_values):
            points = constant_sum_anisotropic_points(float(a), float(e), float(alpha), sample_count=sample_count) / a
            normalized_sets[(alpha, a)] = whiten_points(points, float(alpha))
        for (alpha1, a1), (alpha2, a2) in combinations(normalized_sets.keys(), 2):
            delta = normalized_sets[(alpha1, a1)] - normalized_sets[(alpha2, a2)]
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            rows.append(
                {
                    "e": float(e),
                    "alpha1": float(alpha1),
                    "a1": float(a1),
                    "alpha2": float(alpha2),
                    "a2": float(a2),
                    "mean_whitened_collapse_error": float(np.mean(dist)),
                    "max_whitened_collapse_error": float(np.max(dist)),
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
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.5), constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.26, wspace=0.22)
    gallery = [(0.20, 0.60), (0.55, 0.75), (0.55, 1.50), (0.85, 2.00)]

    for idx, (ax, (e, alpha)) in enumerate(zip(axes.ravel(), gallery)):
        a, b, c = ellipse_parameters(1.0, e)
        points = constant_sum_anisotropic_points(1.0, e, alpha, sample_count=420)
        analytic = np.linspace(0.0, 2.0 * math.pi, 800, endpoint=False)
        x_curve = a * np.cos(analytic)
        y_curve = (b / alpha) * np.sin(analytic)
        ax.scatter(points[:, 0], points[:, 1], s=7, color="#264653", alpha=0.5, label="anisotropic-process locus")
        ax.plot(x_curve, y_curve, color="#e76f51", lw=2.2, label="analytic anisotropic ellipse")
        ax.scatter([-c, c], [0, 0], color="#d62828", s=28, zorder=4)
        ax.set_title(f"e = {e:.2f}, alpha = {alpha:.2f}")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xlabel("x / a")
        ax.set_ylabel("y / a")
        residual = anisotropic_equation_residual(points, a, b, alpha)
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

    fig.suptitle("Experiment 4A: Anisotropic Process Reconstruction", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_whitening_recovery(path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), constrained_layout=False)
    fig.subplots_adjust(top=0.84, wspace=0.28)

    fixed_e = 0.60
    alpha_values = [0.50, 0.75, 1.00, 1.50, 2.00]
    colors = ["#2a9d8f", "#457b9d", "#264653", "#e76f51", "#8b1e3f"]

    for alpha, color in zip(alpha_values, colors):
        pts = constant_sum_anisotropic_points(1.0, fixed_e, alpha, sample_count=420)
        axes[0].plot(pts[:, 0], pts[:, 1], color=color, lw=2.0, alpha=0.85)
        ptsw = whiten_points(pts, alpha)
        axes[1].plot(ptsw[:, 0], ptsw[:, 1], color=color, lw=2.0, alpha=0.85)

    fixed_alpha = 1.50
    a_values = [0.75, 1.0, 1.5, 2.5, 4.0]
    for a, color in zip(a_values, colors):
        pts = constant_sum_anisotropic_points(a, fixed_e, fixed_alpha, sample_count=420) / a
        axes[2].plot(pts[:, 0], pts[:, 1], color=color, lw=2.0, alpha=0.85)

    axes[0].set_title("Raw normalized shapes at fixed e and varying alpha")
    axes[1].set_title("After whitening, the same family collapses")
    axes[2].set_title("Fixed (e, alpha) still collapses across scale")

    for ax in axes:
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-2.1, 2.1)
        ax.set_aspect("equal")
        ax.set_xlabel("x / a")
        ax.set_ylabel("y / a or whitened y / a")

    axes[0].legend(
        handles=[plt.Line2D([0], [0], color=color, lw=3, label=f"alpha = {alpha:.2f}") for alpha, color in zip(alpha_values, colors)],
        loc="lower left",
        frameon=True,
    )
    axes[2].legend(
        handles=[plt.Line2D([0], [0], color=color, lw=3, label=f"a = {a:.2f}") for a, color in zip(a_values, colors)],
        loc="lower left",
        frameon=True,
    )

    fig.suptitle("Experiment 4B: Raw One-Knob Failure And Whitening Recovery", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_response_surfaces(path: str) -> None:
    e = np.linspace(0.05, 0.95, 260)
    alpha_values = [0.50, 0.75, 1.00, 1.50, 2.00]
    colors = ["#2a9d8f", "#457b9d", "#264653", "#e76f51", "#8b1e3f"]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11.8, 9.2), constrained_layout=False)
    fig.subplots_adjust(top=0.9, hspace=0.32)

    whitened_residue = np.sqrt(1.0 - e**2)
    whitened_major = 1.0 / (1.0 - e**2)

    for alpha, color in zip(alpha_values, colors):
        raw_residue = whitened_residue / alpha
        raw_major = (alpha * alpha) * whitened_major
        ax_top.plot(e, raw_residue, color=color, lw=2.5, label=f"alpha = {alpha:.2f}")
        ax_bottom.plot(e, raw_major, color=color, lw=2.5, label=f"alpha = {alpha:.2f}")

    ax_top.plot(e, whitened_residue, color="#111111", lw=2.2, linestyle="--", label="whitened collapse")
    ax_bottom.plot(e, whitened_major, color="#111111", lw=2.2, linestyle="--", label="whitened collapse")
    ax_top.set_title("Raw vertical residue spreads with alpha; whitening restores e-only behavior")
    ax_top.set_ylabel("vertical residue")
    ax_top.set_xlim(0.0, 1.0)
    ax_top.legend(loc="upper right", frameon=True, ncol=2)

    ax_bottom.set_title("Raw major-tip response absorbs alpha^2; whitening removes it")
    ax_bottom.set_xlabel("e = c / a")
    ax_bottom.set_ylabel("major-tip response")
    ax_bottom.set_xlim(0.0, 1.0)
    ax_bottom.set_yscale("log")
    ax_bottom.legend(loc="upper left", frameon=True, ncol=2)

    fig.suptitle("Experiment 4C: Anisotropy Response Curves", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_map(path: str) -> None:
    e = np.linspace(0.02, 0.98, 260)
    alpha = np.linspace(0.40, 2.10, 240)
    E, A = np.meshgrid(e, alpha)
    raw_residue = np.sqrt(np.maximum(1.0 - E**2, 0.0)) / A
    raw_major = (A * A) / np.maximum(1.0 - E**2, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.26)

    mesh0 = axes[0].pcolormesh(E, A, raw_residue, cmap="viridis", shading="auto")
    axes[0].contour(E, A, E, levels=[0.2, 0.4, 0.6, 0.8], colors="white", linewidths=1.0)
    axes[0].set_title("Raw vertical residue field")
    axes[0].set_xlabel("e")
    axes[0].set_ylabel("alpha")
    cb0 = fig.colorbar(mesh0, ax=axes[0], shrink=0.9)
    cb0.set_label("raw vertical residue")

    mesh1 = axes[1].pcolormesh(E, A, np.log10(raw_major), cmap="magma", shading="auto")
    axes[1].contour(E, A, E, levels=[0.2, 0.4, 0.6, 0.8], colors="white", linewidths=1.0)
    axes[1].set_title("Raw major-tip response field")
    axes[1].set_xlabel("e")
    axes[1].set_ylabel("alpha")
    cb1 = fig.colorbar(mesh1, ax=axes[1], shrink=0.9)
    cb1.set_label("log10 raw major-tip response")

    fig.suptitle("Experiment 4D: Two-Parameter Raw Geometry Under Controlled Anisotropy", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    e_values = np.round(np.linspace(0.10, 0.90, 17), 4)
    alpha_values = [0.50, 0.75, 1.00, 1.50, 2.00]
    a_values = [0.75, 1.0, 1.5, 2.5, 4.0]

    metric_rows = make_rows(e_values, alpha_values, a_values)
    raw_scale_rows = raw_scale_collapse(e_values, alpha_values, a_values)
    raw_family_rows = raw_family_distances(e_values, alpha_values, a=1.0)
    whitened_rows = whitened_collapse(e_values, alpha_values, a_values)

    write_csv(os.path.join(OUTPUT_DIR, "anisotropy_metrics.csv"), [asdict(row) for row in metric_rows])
    write_csv(os.path.join(OUTPUT_DIR, "anisotropy_scale_collapse.csv"), raw_scale_rows)
    write_csv(os.path.join(OUTPUT_DIR, "anisotropy_raw_family_distances.csv"), raw_family_rows)
    write_csv(os.path.join(OUTPUT_DIR, "anisotropy_whitened_collapse.csv"), whitened_rows)

    summary = {
        "representation": {
            "metric": "d_alpha((x,y),(u,v)) = sqrt((x-u)^2 + alpha^2 (y-v)^2)",
            "whitening_transform": "(x, y) -> (x, alpha y)",
        },
        "max_equation_residual": float(max(row.max_equation_residual for row in metric_rows)),
        "max_rms_equation_residual": float(max(row.rms_equation_residual for row in metric_rows)),
        "max_raw_scale_collapse_error": float(max(row["max_collapse_error"] for row in raw_scale_rows)),
        "mean_raw_scale_collapse_error": float(np.mean([row["mean_collapse_error"] for row in raw_scale_rows])),
        "min_raw_family_distance_fixed_e": float(min(row["mean_family_distance"] for row in raw_family_rows)),
        "max_raw_family_distance_fixed_e": float(max(row["max_family_distance"] for row in raw_family_rows)),
        "max_whitened_collapse_error": float(max(row["max_whitened_collapse_error"] for row in whitened_rows)),
        "mean_whitened_collapse_error": float(np.mean([row["mean_whitened_collapse_error"] for row in whitened_rows])),
    }

    with open(os.path.join(OUTPUT_DIR, "anisotropy_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_process_reconstruction(os.path.join(FIGURE_DIR, "anisotropy_process_reconstruction.png"))
    plot_whitening_recovery(os.path.join(FIGURE_DIR, "anisotropy_whitening_recovery.png"))
    plot_response_surfaces(os.path.join(FIGURE_DIR, "anisotropy_response_curves.png"))
    plot_parameter_map(os.path.join(FIGURE_DIR, "anisotropy_parameter_map.png"))

    print("Controlled anisotropy experiment complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
