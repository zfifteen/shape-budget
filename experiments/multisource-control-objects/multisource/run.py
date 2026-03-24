"""
Experiment 5: multi-source generalization for Shape Budget.

This experiment studies the three-source constant-sum boundary

    sum_i ||x - p_i|| = S

with equal source weights.

The question is not whether the two-source one-knob result survives
unchanged. It does not. With three sources, the natural control object is the
normalized source triangle relative to the total budget, which carries three
degrees of freedom after translation and rotation are factored out.

This script tests four linked claims:

1. for a fixed normalized source triangle, normalized boundaries collapse
   exactly across absolute scale
2. the per-boundary-point allocation-share loop
   (d1 / S, d2 / S, d3 / S) is also scale-invariant
3. the equilateral three-source slice forms a near one-parameter family
4. the broader normalized three-source family remains low-dimensional
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

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
class ResidualRow:
    family: str
    rho: float
    t: float
    h: float
    S: float
    d12_over_S: float
    d13_over_S: float
    d23_over_S: float
    interior_sum_over_S: float
    max_equation_residual: float
    rms_equation_residual: float

@dataclass
class CollapseRow:
    family: str
    rho: float
    t: float
    h: float
    S1: float
    S2: float
    mean_boundary_collapse_error: float
    max_boundary_collapse_error: float
    mean_simplex_loop_error: float
    max_simplex_loop_error: float

@dataclass
class SpectrumRow:
    dataset: str
    component: int
    explained_variance_ratio: float
    cumulative_explained_variance_ratio: float

def canonical_sources(rho: float, t: float, h: float, S: float = 1.0) -> np.ndarray:
    return S * np.array(
        [
            [-rho, 0.0],
            [rho, 0.0],
            [rho * t, rho * h],
        ],
        dtype=float,
    )

def equilateral_sources(rho: float, S: float = 1.0) -> np.ndarray:
    return S * np.array(
        [
            [-rho, 0.0],
            [rho, 0.0],
            [0.0, math.sqrt(3.0) * rho],
        ],
        dtype=float,
    )

def pairwise_distance_invariants(points: np.ndarray, S: float) -> tuple[float, float, float]:
    d12 = float(np.linalg.norm(points[0] - points[1]) / S)
    d13 = float(np.linalg.norm(points[0] - points[2]) / S)
    d23 = float(np.linalg.norm(points[1] - points[2]) / S)
    return d12, d13, d23

def total_distance(x: np.ndarray, points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(points - x, axis=1)))

def geometric_median(points: np.ndarray, tol: float = 1.0e-13, max_iter: int = 10_000) -> np.ndarray:
    guess = np.mean(points, axis=0)
    for _ in range(max_iter):
        diff = points - guess
        distances = np.linalg.norm(diff, axis=1)
        if np.min(distances) < tol:
            return points[int(np.argmin(distances))].copy()
        weights = 1.0 / np.maximum(distances, tol)
        next_guess = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(next_guess - guess) < tol:
            return next_guess
        guess = next_guess
    return guess

def boundary_radius_on_ray(points: np.ndarray, S: float, origin: np.ndarray, angle: float) -> float:
    direction = np.array([math.cos(angle), math.sin(angle)])

    def g(radius: float) -> float:
        x = origin + radius * direction
        return total_distance(x, points) - S

    low = 0.0
    high = max(1.0, S)
    if g(low) >= 0.0:
        raise ValueError("Origin is not inside the constant-sum region.")
    while g(high) <= 0.0:
        high *= 2.0
        if high > 1.0e6:
            raise RuntimeError("Failed to bracket boundary radius.")

    for _ in range(90):
        mid = 0.5 * (low + high)
        if g(mid) <= 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

def boundary_curve(points: np.ndarray, S: float, angle_count: int = 360) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = geometric_median(points)
    interior_sum = total_distance(origin, points)
    if interior_sum >= S:
        raise ValueError("No closed constant-sum boundary exists for this source triangle and budget.")

    angles = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    radii = np.array([boundary_radius_on_ray(points, S, origin, angle) for angle in angles])
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    curve = origin + radii[:, None] * directions
    return origin, angles, curve

def allocation_shares(curve: np.ndarray, points: np.ndarray, S: float) -> np.ndarray:
    distances = np.column_stack([np.linalg.norm(curve - point, axis=1) for point in points])
    return distances / S

def simplex_projection(weights: np.ndarray) -> np.ndarray:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
        ]
    )
    return weights @ vertices

def shape_signature(curve: np.ndarray, origin: np.ndarray, S: float) -> np.ndarray:
    return np.linalg.norm((curve - origin) / S, axis=1)

def write_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def collect_scale_collapse(
    configs: list[tuple[str, float, float, float]],
    S_values: list[float],
    angle_count: int = 360,
) -> tuple[list[ResidualRow], list[CollapseRow]]:
    residual_rows: list[ResidualRow] = []
    collapse_rows: list[CollapseRow] = []

    for family, rho, t, h in configs:
        normalized_curves: dict[float, np.ndarray] = {}
        simplex_loops: dict[float, np.ndarray] = {}
        for S in S_values:
            points = canonical_sources(rho, t, h, S=S)
            origin, _, curve = boundary_curve(points, S, angle_count=angle_count)
            shares = allocation_shares(curve, points, S)
            residuals = np.sum(np.linalg.norm(curve[:, None, :] - points[None, :, :], axis=2), axis=1) - S
            d12, d13, d23 = pairwise_distance_invariants(points, S)
            residual_rows.append(
                ResidualRow(
                    family=family,
                    rho=rho,
                    t=t,
                    h=h,
                    S=S,
                    d12_over_S=d12,
                    d13_over_S=d13,
                    d23_over_S=d23,
                    interior_sum_over_S=total_distance(origin, points) / S,
                    max_equation_residual=float(np.max(np.abs(residuals))),
                    rms_equation_residual=float(np.sqrt(np.mean(residuals**2))),
                )
            )
            normalized_curves[S] = (curve - origin) / S
            simplex_loops[S] = shares

        for S1, S2 in combinations(S_values, 2):
            boundary_delta = normalized_curves[S1] - normalized_curves[S2]
            simplex_delta = simplex_loops[S1] - simplex_loops[S2]
            boundary_dist = np.linalg.norm(boundary_delta, axis=1)
            simplex_dist = np.linalg.norm(simplex_delta, axis=1)
            collapse_rows.append(
                CollapseRow(
                    family=family,
                    rho=rho,
                    t=t,
                    h=h,
                    S1=S1,
                    S2=S2,
                    mean_boundary_collapse_error=float(np.mean(boundary_dist)),
                    max_boundary_collapse_error=float(np.max(boundary_dist)),
                    mean_simplex_loop_error=float(np.mean(simplex_dist)),
                    max_simplex_loop_error=float(np.max(simplex_dist)),
                )
            )

    return residual_rows, collapse_rows

def signature_matrix_for_equilateral(rho_values: np.ndarray, angle_count: int = 240) -> np.ndarray:
    signatures = []
    for rho in rho_values:
        points = equilateral_sources(float(rho), S=1.0)
        origin, _, curve = boundary_curve(points, 1.0, angle_count=angle_count)
        signatures.append(shape_signature(curve, origin, 1.0))
    return np.array(signatures)

def sample_random_signatures(
    sample_size: int = 180,
    angle_count: int = 240,
    seed: int = 20260324,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: list[tuple[float, float, float]] = []
    signatures: list[np.ndarray] = []

    while len(signatures) < sample_size:
        rho = float(rng.uniform(0.05, 0.24))
        t = float(rng.uniform(-0.8, 0.8))
        h = float(rng.uniform(0.45, 1.6))
        points = canonical_sources(rho, t, h, S=1.0)
        origin = geometric_median(points)
        if total_distance(origin, points) >= 0.985:
            continue
        _, _, curve = boundary_curve(points, 1.0, angle_count=angle_count)
        params.append((rho, t, h))
        signatures.append(shape_signature(curve, geometric_median(points), 1.0))

    return np.array(params), np.array(signatures)

def spectrum_rows(dataset: str, pca: PCA, component_count: int = 8) -> list[SpectrumRow]:
    rows: list[SpectrumRow] = []
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for idx in range(component_count):
        rows.append(
            SpectrumRow(
                dataset=dataset,
                component=idx + 1,
                explained_variance_ratio=float(pca.explained_variance_ratio_[idx]),
                cumulative_explained_variance_ratio=float(cumulative[idx]),
            )
        )
    return rows

def plot_gallery(path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.8), constrained_layout=False)
    fig.subplots_adjust(top=0.89, hspace=0.24, wspace=0.22)
    gallery = [
        ("balanced", 0.10, 0.0, 1.0),
        ("acute", 0.14, 0.35, 1.20),
        ("skewed", 0.16, -0.55, 0.85),
        ("tall", 0.18, 0.45, 1.45),
    ]

    for ax, (label, rho, t, h) in zip(axes.ravel(), gallery):
        points = canonical_sources(rho, t, h, S=1.0)
        origin, _, curve = boundary_curve(points, 1.0, angle_count=420)
        d12, d13, d23 = pairwise_distance_invariants(points, 1.0)
        ax.plot(curve[:, 0], curve[:, 1], color="#1d3557", lw=2.2)
        ax.scatter(points[:, 0], points[:, 1], c=["#d62828", "#f77f00", "#2a9d8f"], s=42, zorder=4)
        ax.scatter([origin[0]], [origin[1]], c="#264653", s=24, zorder=5)
        for idx, point in enumerate(points, start=1):
            ax.text(point[0] + 0.015, point[1] + 0.015, f"s{idx}", fontsize=9)
        ax.set_title(f"{label}: (d12, d13, d23) / S = ({d12:.2f}, {d13:.2f}, {d23:.2f})")
        ax.set_aspect("equal")
        ax.set_xlabel("x / S")
        ax.set_ylabel("y / S")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.75, 0.95)

    fig.suptitle("Experiment 5A: Three-Source Constant-Sum Families", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_scale_collapse(path: str, S_values: list[float]) -> None:
    rho, t, h = 0.14, 0.25, 1.15
    colors = sns.color_palette("viridis", n_colors=len(S_values))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.25)

    for color, S in zip(colors, S_values):
        points = canonical_sources(rho, t, h, S=S)
        origin, _, curve = boundary_curve(points, S, angle_count=420)
        normalized_curve = (curve - origin) / S
        shares = allocation_shares(curve, points, S)
        simplex = simplex_projection(shares)
        ax_left.plot(normalized_curve[:, 0], normalized_curve[:, 1], color=color, lw=2.0, alpha=0.9, label=f"S = {S:.2f}")
        ax_right.plot(simplex[:, 0], simplex[:, 1], color=color, lw=2.0, alpha=0.9, label=f"S = {S:.2f}")

    simplex_vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax_right.plot(simplex_vertices[:, 0], simplex_vertices[:, 1], color="#888888", lw=1.6)
    ax_right.text(-0.02, -0.03, "w1 = 1", fontsize=9)
    ax_right.text(1.01, -0.03, "w2 = 1", fontsize=9, ha="left")
    ax_right.text(0.5, math.sqrt(3.0) / 2.0 + 0.03, "w3 = 1", fontsize=9, ha="center")

    ax_left.set_title("Normalized boundary collapses across absolute scale")
    ax_left.set_xlabel("(x - x*) / S")
    ax_left.set_ylabel("(y - y*) / S")
    ax_left.set_aspect("equal")
    ax_left.legend(loc="lower left", frameon=True)

    ax_right.set_title("Allocation-share loop is also scale-invariant")
    ax_right.set_xlabel("simplex x")
    ax_right.set_ylabel("simplex y")
    ax_right.set_aspect("equal")

    fig.suptitle("Experiment 5B: Scale Collapse And Allocation-Share Loop", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_dimension(path: str, equilateral_rho_values: np.ndarray, equilateral_pca: PCA, random_pca: PCA) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=False)
    fig.subplots_adjust(top=0.84, wspace=0.28)

    colors = sns.color_palette("rocket", n_colors=len(equilateral_rho_values))
    for color, rho in zip(colors, equilateral_rho_values):
        points = equilateral_sources(float(rho), S=1.0)
        origin, _, curve = boundary_curve(points, 1.0, angle_count=320)
        normalized_curve = (curve - origin) / 1.0
        axes[0].plot(normalized_curve[:, 0], normalized_curve[:, 1], color=color, lw=1.8, alpha=0.9)
    axes[0].set_title("Equilateral slice sweeps a thin one-parameter family")
    axes[0].set_xlabel("(x - x*) / S")
    axes[0].set_ylabel("(y - y*) / S")
    axes[0].set_aspect("equal")

    eq_components = np.arange(1, 7)
    eq_cumulative = np.cumsum(equilateral_pca.explained_variance_ratio_[:6])
    axes[1].bar(eq_components, equilateral_pca.explained_variance_ratio_[:6], color="#f4a261", alpha=0.85)
    axes[1].plot(eq_components, eq_cumulative, color="#1d3557", lw=2.2, marker="o")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Equilateral slice spectrum")
    axes[1].set_xlabel("principal component")
    axes[1].set_ylabel("explained variance ratio")

    rand_components = np.arange(1, 7)
    rand_cumulative = np.cumsum(random_pca.explained_variance_ratio_[:6])
    axes[2].bar(rand_components, random_pca.explained_variance_ratio_[:6], color="#2a9d8f", alpha=0.85)
    axes[2].plot(rand_components, rand_cumulative, color="#1d3557", lw=2.2, marker="o")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Random normalized family spectrum")
    axes[2].set_xlabel("principal component")
    axes[2].set_ylabel("explained variance ratio")

    fig.suptitle("Experiment 5C: One-Parameter Slice And Low-Dimensional Family", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    configs = [
        ("balanced", 0.10, 0.00, 1.00),
        ("offset", 0.14, 0.25, 1.15),
        ("skewed", 0.16, -0.55, 0.85),
        ("tall", 0.18, 0.45, 1.45),
    ]
    S_values = [0.75, 1.00, 1.50, 2.50, 4.00]
    equilateral_rho_values = np.linspace(0.05, 0.26, 28)

    residual_rows, collapse_rows = collect_scale_collapse(configs, S_values, angle_count=360)
    random_params, random_signatures = sample_random_signatures(sample_size=180, angle_count=240)
    equilateral_signatures = signature_matrix_for_equilateral(equilateral_rho_values, angle_count=240)

    equilateral_pca = PCA().fit(equilateral_signatures)
    random_pca = PCA().fit(random_signatures)

    residual_dicts = [row.__dict__ for row in residual_rows]
    collapse_dicts = [row.__dict__ for row in collapse_rows]
    spectrum_dicts = [row.__dict__ for row in spectrum_rows("equilateral_slice", equilateral_pca)]
    spectrum_dicts += [row.__dict__ for row in spectrum_rows("random_family", random_pca)]
    random_param_rows = [
        {"rho": float(rho), "t": float(t), "h": float(h)}
        for rho, t, h in random_params
    ]

    write_csv(os.path.join(OUTPUT_DIR, "multisource_residuals.csv"), residual_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "multisource_scale_collapse.csv"), collapse_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "multisource_spectra.csv"), spectrum_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "multisource_random_parameters.csv"), random_param_rows)

    plot_gallery(os.path.join(FIGURE_DIR, "multisource_family_gallery.png"))
    plot_scale_collapse(os.path.join(FIGURE_DIR, "multisource_scale_collapse.png"), S_values)
    plot_dimension(os.path.join(FIGURE_DIR, "multisource_dimension.png"), equilateral_rho_values, equilateral_pca, random_pca)

    summary = {
        "max_equation_residual": float(max(row.max_equation_residual for row in residual_rows)),
        "mean_equation_residual": float(np.mean([row.rms_equation_residual for row in residual_rows])),
        "max_boundary_collapse_error": float(max(row.max_boundary_collapse_error for row in collapse_rows)),
        "mean_boundary_collapse_error": float(np.mean([row.mean_boundary_collapse_error for row in collapse_rows])),
        "max_simplex_loop_error": float(max(row.max_simplex_loop_error for row in collapse_rows)),
        "mean_simplex_loop_error": float(np.mean([row.mean_simplex_loop_error for row in collapse_rows])),
        "equilateral_pc1_explained_variance_ratio": float(equilateral_pca.explained_variance_ratio_[0]),
        "equilateral_pc2_cumulative_explained_variance_ratio": float(np.sum(equilateral_pca.explained_variance_ratio_[:2])),
        "random_pc1_explained_variance_ratio": float(random_pca.explained_variance_ratio_[0]),
        "random_pc3_cumulative_explained_variance_ratio": float(np.sum(random_pca.explained_variance_ratio_[:3])),
        "random_pc5_cumulative_explained_variance_ratio": float(np.sum(random_pca.explained_variance_ratio_[:5])),
    }
    with open(os.path.join(OUTPUT_DIR, "multisource_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
