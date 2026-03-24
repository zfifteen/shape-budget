"""
Post-roadmap extension: weighted multi-source generalization for Shape Budget.

This experiment studies the weighted three-source constant-sum boundary

    w1 ||x - p1|| + w2 ||x - p2|| + w3 ||x - p3|| = S

with positive weights normalized so that w1 + w2 + w3 = 1.

The natural control object is no longer just the normalized source triangle
relative to budget. It is:

- the normalized source triangle relative to budget, plus
- the weight vector in the 2-simplex.

This script tests four linked claims:

1. for fixed normalized source geometry and fixed weights, normalized
   boundaries collapse across absolute scale
2. the weighted allocation-share loop
   (w1 d1 / S, w2 d2 / S, w3 d3 / S) is also scale-invariant
3. fixing geometry and varying weights breaks equal-weight sufficiency
4. the equilateral weighted slice is near two-parameter, while the broader
   weighted family remains low-dimensional with about five principal directions
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
OUTPUT_DIR = os.path.join(BASE_DIR, "weighted_multisource_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


@dataclass
class ResidualRow:
    family: str
    rho: float
    t: float
    h: float
    w1: float
    w2: float
    w3: float
    S: float
    d12_over_S: float
    d13_over_S: float
    d23_over_S: float
    weighted_interior_sum_over_S: float
    max_equation_residual: float
    rms_equation_residual: float


@dataclass
class CollapseRow:
    family: str
    rho: float
    t: float
    h: float
    w1: float
    w2: float
    w3: float
    S1: float
    S2: float
    mean_boundary_collapse_error: float
    max_boundary_collapse_error: float
    mean_simplex_loop_error: float
    max_simplex_loop_error: float


@dataclass
class WeightVariationRow:
    family: str
    rho: float
    t: float
    h: float
    label1: str
    w1_1: float
    w2_1: float
    w3_1: float
    label2: str
    w1_2: float
    w2_2: float
    w3_2: float
    mean_boundary_family_distance: float
    max_boundary_family_distance: float
    mean_simplex_loop_family_distance: float
    max_simplex_loop_family_distance: float


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


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0.0):
        raise ValueError("Weighted multi-source experiment requires strictly positive weights.")
    return weights / np.sum(weights)


def pairwise_distance_invariants(points: np.ndarray, S: float) -> tuple[float, float, float]:
    d12 = float(np.linalg.norm(points[0] - points[1]) / S)
    d13 = float(np.linalg.norm(points[0] - points[2]) / S)
    d23 = float(np.linalg.norm(points[1] - points[2]) / S)
    return d12, d13, d23


def weighted_total_distance(x: np.ndarray, points: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * np.linalg.norm(points - x, axis=1)))


def weighted_geometric_median(
    points: np.ndarray,
    weights: np.ndarray,
    tol: float = 1.0e-13,
    max_iter: int = 10_000,
) -> np.ndarray:
    guess = np.average(points, axis=0, weights=weights)
    for _ in range(max_iter):
        diff = points - guess
        distances = np.linalg.norm(diff, axis=1)
        if np.min(distances) < tol:
            return points[int(np.argmin(distances))].copy()
        coeff = weights / np.maximum(distances, tol)
        next_guess = np.sum(points * coeff[:, None], axis=0) / np.sum(coeff)
        if np.linalg.norm(next_guess - guess) < tol:
            return next_guess
        guess = next_guess
    return guess


def boundary_radius_on_ray(
    points: np.ndarray,
    weights: np.ndarray,
    S: float,
    origin: np.ndarray,
    angle: float,
) -> float:
    direction = np.array([math.cos(angle), math.sin(angle)])

    def g(radius: float) -> float:
        x = origin + radius * direction
        return weighted_total_distance(x, points, weights) - S

    low = 0.0
    high = max(1.0, S)
    if g(low) >= 0.0:
        raise ValueError("Origin is not inside the weighted constant-sum region.")
    while g(high) <= 0.0:
        high *= 2.0
        if high > 1.0e6:
            raise RuntimeError("Failed to bracket weighted boundary radius.")

    for _ in range(90):
        mid = 0.5 * (low + high)
        if g(mid) <= 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def weighted_boundary_curve(
    points: np.ndarray,
    weights: np.ndarray,
    S: float,
    angle_count: int = 360,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = weighted_geometric_median(points, weights)
    interior_sum = weighted_total_distance(origin, points, weights)
    if interior_sum >= S:
        raise ValueError("No closed weighted constant-sum boundary exists for this configuration.")

    angles = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    radii = np.array([boundary_radius_on_ray(points, weights, S, origin, angle) for angle in angles])
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    curve = origin + radii[:, None] * directions
    return origin, angles, curve


def weighted_allocation_shares(curve: np.ndarray, points: np.ndarray, weights: np.ndarray, S: float) -> np.ndarray:
    distances = np.column_stack([np.linalg.norm(curve - point, axis=1) for point in points])
    return distances * weights[None, :] / S


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


def format_weight_label(weights: np.ndarray) -> str:
    return f"({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f})"


def write_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_scale_collapse(
    configs: list[tuple[str, float, float, float]],
    weights: np.ndarray,
    S_values: list[float],
    angle_count: int = 360,
) -> tuple[list[ResidualRow], list[CollapseRow]]:
    residual_rows: list[ResidualRow] = []
    collapse_rows: list[CollapseRow] = []
    w1, w2, w3 = [float(item) for item in weights]

    for family, rho, t, h in configs:
        normalized_curves: dict[float, np.ndarray] = {}
        simplex_loops: dict[float, np.ndarray] = {}
        for S in S_values:
            points = canonical_sources(rho, t, h, S=S)
            origin, _, curve = weighted_boundary_curve(points, weights, S, angle_count=angle_count)
            shares = weighted_allocation_shares(curve, points, weights, S)
            residuals = np.sum(
                weights[None, :] * np.linalg.norm(curve[:, None, :] - points[None, :, :], axis=2),
                axis=1,
            ) - S
            d12, d13, d23 = pairwise_distance_invariants(points, S)
            residual_rows.append(
                ResidualRow(
                    family=family,
                    rho=rho,
                    t=t,
                    h=h,
                    w1=w1,
                    w2=w2,
                    w3=w3,
                    S=S,
                    d12_over_S=d12,
                    d13_over_S=d13,
                    d23_over_S=d23,
                    weighted_interior_sum_over_S=weighted_total_distance(origin, points, weights) / S,
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
                    w1=w1,
                    w2=w2,
                    w3=w3,
                    S1=S1,
                    S2=S2,
                    mean_boundary_collapse_error=float(np.mean(boundary_dist)),
                    max_boundary_collapse_error=float(np.max(boundary_dist)),
                    mean_simplex_loop_error=float(np.mean(simplex_dist)),
                    max_simplex_loop_error=float(np.max(simplex_dist)),
                )
            )

    return residual_rows, collapse_rows


def collect_weight_variation(
    family: tuple[str, float, float, float],
    weight_sets: list[tuple[str, np.ndarray]],
    S: float = 1.0,
    angle_count: int = 320,
) -> list[WeightVariationRow]:
    family_name, rho, t, h = family
    points = canonical_sources(rho, t, h, S=S)
    boundary_sets: dict[str, np.ndarray] = {}
    simplex_sets: dict[str, np.ndarray] = {}
    weight_lookup: dict[str, np.ndarray] = {}

    for label, weights in weight_sets:
        origin, _, curve = weighted_boundary_curve(points, weights, S, angle_count=angle_count)
        boundary_sets[label] = (curve - origin) / S
        simplex_sets[label] = weighted_allocation_shares(curve, points, weights, S)
        weight_lookup[label] = weights

    rows: list[WeightVariationRow] = []
    for (label1, _), (label2, _) in combinations(weight_sets, 2):
        boundary_delta = boundary_sets[label1] - boundary_sets[label2]
        simplex_delta = simplex_sets[label1] - simplex_sets[label2]
        boundary_dist = np.linalg.norm(boundary_delta, axis=1)
        simplex_dist = np.linalg.norm(simplex_delta, axis=1)
        w1 = weight_lookup[label1]
        w2 = weight_lookup[label2]
        rows.append(
            WeightVariationRow(
                family=family_name,
                rho=rho,
                t=t,
                h=h,
                label1=label1,
                w1_1=float(w1[0]),
                w2_1=float(w1[1]),
                w3_1=float(w1[2]),
                label2=label2,
                w1_2=float(w2[0]),
                w2_2=float(w2[1]),
                w3_2=float(w2[2]),
                mean_boundary_family_distance=float(np.mean(boundary_dist)),
                max_boundary_family_distance=float(np.max(boundary_dist)),
                mean_simplex_loop_family_distance=float(np.mean(simplex_dist)),
                max_simplex_loop_family_distance=float(np.max(simplex_dist)),
            )
        )
    return rows


def equilateral_weight_grid() -> list[np.ndarray]:
    weights: list[np.ndarray] = []
    grid = np.linspace(0.15, 0.70, 12)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if 0.15 <= w3 <= 0.70:
                weights.append(np.array([w1, w2, w3], dtype=float))
    return weights


def equilateral_weight_signatures(rho: float = 0.16, angle_count: int = 180) -> tuple[np.ndarray, np.ndarray]:
    weight_vectors = equilateral_weight_grid()
    signatures = []
    for weights in weight_vectors:
        points = equilateral_sources(rho, S=1.0)
        origin, _, curve = weighted_boundary_curve(points, weights, 1.0, angle_count=angle_count)
        signatures.append(shape_signature(curve, origin, 1.0))
    return np.array(weight_vectors), np.array(signatures)


def sample_random_weighted_signatures(
    sample_size: int = 180,
    angle_count: int = 180,
    seed: int = 20260324,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: list[tuple[float, float, float, float, float, float]] = []
    signatures: list[np.ndarray] = []

    while len(signatures) < sample_size:
        rho = float(rng.uniform(0.05, 0.24))
        t = float(rng.uniform(-0.8, 0.8))
        h = float(rng.uniform(0.45, 1.6))
        weights = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
        points = canonical_sources(rho, t, h, S=1.0)
        origin = weighted_geometric_median(points, weights)
        if weighted_total_distance(origin, points, weights) >= 0.99:
            continue
        origin, _, curve = weighted_boundary_curve(points, weights, 1.0, angle_count=angle_count)
        params.append((rho, t, h, float(weights[0]), float(weights[1]), float(weights[2])))
        signatures.append(shape_signature(curve, origin, 1.0))

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


def plot_scale_collapse(path: str, weights: np.ndarray, S_values: list[float]) -> None:
    rho, t, h = 0.14, 0.25, 1.15
    colors = sns.color_palette("viridis", n_colors=len(S_values))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.25)

    for color, S in zip(colors, S_values):
        points = canonical_sources(rho, t, h, S=S)
        origin, _, curve = weighted_boundary_curve(points, weights, S, angle_count=420)
        normalized_curve = (curve - origin) / S
        shares = weighted_allocation_shares(curve, points, weights, S)
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
    ax_right.text(-0.02, -0.03, "s1 share = 1", fontsize=9)
    ax_right.text(1.01, -0.03, "s2 share = 1", fontsize=9, ha="left")
    ax_right.text(0.5, math.sqrt(3.0) / 2.0 + 0.03, "s3 share = 1", fontsize=9, ha="center")

    ax_left.set_title(f"Normalized boundary collapse at weights {format_weight_label(weights)}")
    ax_left.set_xlabel("(x - x*) / S")
    ax_left.set_ylabel("(y - y*) / S")
    ax_left.set_aspect("equal")
    ax_left.legend(loc="lower left", frameon=True)

    ax_right.set_title("Weighted allocation-share loop is scale-invariant")
    ax_right.set_xlabel("simplex x")
    ax_right.set_ylabel("simplex y")
    ax_right.set_aspect("equal")

    fig.suptitle("Weighted Experiment A: Scale Collapse And Weighted Share Loop", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_weight_variation(path: str, family: tuple[str, float, float, float], weight_sets: list[tuple[str, np.ndarray]]) -> None:
    _, rho, t, h = family
    points = canonical_sources(rho, t, h, S=1.0)
    colors = sns.color_palette("mako", n_colors=len(weight_sets))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.0, 6.1), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.26)

    for color, (label, weights) in zip(colors, weight_sets):
        origin, _, curve = weighted_boundary_curve(points, weights, 1.0, angle_count=420)
        shares = weighted_allocation_shares(curve, points, weights, 1.0)
        simplex = simplex_projection(shares)
        ax_left.plot((curve[:, 0] - origin[0]), (curve[:, 1] - origin[1]), color=color, lw=2.0, alpha=0.95, label=f"{label}: {format_weight_label(weights)}")
        ax_right.plot(simplex[:, 0], simplex[:, 1], color=color, lw=2.0, alpha=0.95, label=label)

    simplex_vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax_right.plot(simplex_vertices[:, 0], simplex_vertices[:, 1], color="#888888", lw=1.6)
    ax_left.set_title("Fixed geometry, varying weights changes the normalized family")
    ax_left.set_xlabel("(x - x*) / S")
    ax_left.set_ylabel("(y - y*) / S")
    ax_left.set_aspect("equal")
    ax_left.legend(loc="lower left", frameon=True, fontsize=9)

    ax_right.set_title("The weighted share loop also changes with weights")
    ax_right.set_xlabel("simplex x")
    ax_right.set_ylabel("simplex y")
    ax_right.set_aspect("equal")

    fig.suptitle("Weighted Experiment B: Geometry Is Not Controlled By Placement Alone", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dimension(
    path: str,
    equilateral_weight_vectors: np.ndarray,
    equilateral_pca: PCA,
    random_pca: PCA,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.5), constrained_layout=False)
    fig.subplots_adjust(top=0.84, wspace=0.28)

    rho = 0.16
    simplex_points = simplex_projection(equilateral_weight_vectors)
    cmap = plt.cm.inferno
    ordering = np.argsort(simplex_points[:, 1])
    for rank, idx in enumerate(ordering):
        color = cmap(rank / max(len(ordering) - 1, 1))
        weights = equilateral_weight_vectors[idx]
        points = equilateral_sources(rho, S=1.0)
        origin, _, curve = weighted_boundary_curve(points, weights, 1.0, angle_count=320)
        normalized_curve = curve - origin
        axes[0].plot(normalized_curve[:, 0], normalized_curve[:, 1], color=color, lw=1.7, alpha=0.88)
    axes[0].set_title("Equilateral geometry plus weights sweeps a thin 2D family")
    axes[0].set_xlabel("(x - x*) / S")
    axes[0].set_ylabel("(y - y*) / S")
    axes[0].set_aspect("equal")

    eq_components = np.arange(1, 7)
    eq_cumulative = np.cumsum(equilateral_pca.explained_variance_ratio_[:6])
    axes[1].bar(eq_components, equilateral_pca.explained_variance_ratio_[:6], color="#f4a261", alpha=0.85)
    axes[1].plot(eq_components, eq_cumulative, color="#1d3557", lw=2.2, marker="o")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Equilateral weighted slice spectrum")
    axes[1].set_xlabel("principal component")
    axes[1].set_ylabel("explained variance ratio")

    rand_components = np.arange(1, 8)
    rand_cumulative = np.cumsum(random_pca.explained_variance_ratio_[:7])
    axes[2].bar(rand_components, random_pca.explained_variance_ratio_[:7], color="#2a9d8f", alpha=0.85)
    axes[2].plot(rand_components, rand_cumulative, color="#1d3557", lw=2.2, marker="o")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Random weighted family spectrum")
    axes[2].set_xlabel("principal component")
    axes[2].set_ylabel("explained variance ratio")

    fig.suptitle("Weighted Experiment C: Two-Parameter Slice And Five-Parameter Family", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fixed_weights = normalize_weights(np.array([0.20, 0.35, 0.45], dtype=float))
    configs = [
        ("balanced", 0.10, 0.00, 1.00),
        ("offset", 0.14, 0.25, 1.15),
        ("skewed", 0.16, -0.55, 0.85),
        ("tall", 0.18, 0.45, 1.45),
    ]
    S_values = [0.75, 1.00, 1.50, 2.50, 4.00]
    weight_sets = [
        ("equal", normalize_weights(np.array([1.0, 1.0, 1.0]))),
        ("mild skew", normalize_weights(np.array([0.20, 0.35, 0.45]))),
        ("source 3 heavy", normalize_weights(np.array([0.10, 0.20, 0.70]))),
        ("source 1 heavy", normalize_weights(np.array([0.55, 0.25, 0.20]))),
    ]

    residual_rows, collapse_rows = collect_scale_collapse(configs, fixed_weights, S_values, angle_count=360)
    variation_rows = collect_weight_variation(("offset", 0.14, 0.25, 1.15), weight_sets, S=1.0, angle_count=320)
    equilateral_weight_vectors, equilateral_signatures = equilateral_weight_signatures(rho=0.16, angle_count=180)
    random_params, random_signatures = sample_random_weighted_signatures(sample_size=180, angle_count=180)

    equilateral_pca = PCA().fit(equilateral_signatures)
    random_pca = PCA().fit(random_signatures)

    residual_dicts = [row.__dict__ for row in residual_rows]
    collapse_dicts = [row.__dict__ for row in collapse_rows]
    variation_dicts = [row.__dict__ for row in variation_rows]
    spectrum_dicts = [row.__dict__ for row in spectrum_rows("equilateral_weighted_slice", equilateral_pca)]
    spectrum_dicts += [row.__dict__ for row in spectrum_rows("random_weighted_family", random_pca)]
    random_param_rows = [
        {
            "rho": float(rho),
            "t": float(t),
            "h": float(h),
            "w1": float(w1),
            "w2": float(w2),
            "w3": float(w3),
        }
        for rho, t, h, w1, w2, w3 in random_params
    ]

    write_csv(os.path.join(OUTPUT_DIR, "weighted_multisource_residuals.csv"), residual_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_multisource_scale_collapse.csv"), collapse_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_multisource_weight_variation.csv"), variation_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_multisource_spectra.csv"), spectrum_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_multisource_random_parameters.csv"), random_param_rows)

    plot_scale_collapse(os.path.join(FIGURE_DIR, "weighted_multisource_scale_collapse.png"), fixed_weights, S_values)
    plot_weight_variation(os.path.join(FIGURE_DIR, "weighted_multisource_weight_variation.png"), ("offset", 0.14, 0.25, 1.15), weight_sets)
    plot_dimension(
        os.path.join(FIGURE_DIR, "weighted_multisource_dimension.png"),
        equilateral_weight_vectors,
        equilateral_pca,
        random_pca,
    )

    summary = {
        "max_equation_residual": float(max(row.max_equation_residual for row in residual_rows)),
        "mean_equation_residual": float(np.mean([row.rms_equation_residual for row in residual_rows])),
        "max_boundary_collapse_error": float(max(row.max_boundary_collapse_error for row in collapse_rows)),
        "mean_boundary_collapse_error": float(np.mean([row.mean_boundary_collapse_error for row in collapse_rows])),
        "max_simplex_loop_error": float(max(row.max_simplex_loop_error for row in collapse_rows)),
        "mean_simplex_loop_error": float(np.mean([row.mean_simplex_loop_error for row in collapse_rows])),
        "min_fixed_geometry_boundary_family_distance": float(min(row.mean_boundary_family_distance for row in variation_rows)),
        "max_fixed_geometry_boundary_family_distance": float(max(row.mean_boundary_family_distance for row in variation_rows)),
        "min_fixed_geometry_simplex_family_distance": float(min(row.mean_simplex_loop_family_distance for row in variation_rows)),
        "max_fixed_geometry_simplex_family_distance": float(max(row.mean_simplex_loop_family_distance for row in variation_rows)),
        "equilateral_weighted_pc1_explained_variance_ratio": float(equilateral_pca.explained_variance_ratio_[0]),
        "equilateral_weighted_pc2_cumulative_explained_variance_ratio": float(np.sum(equilateral_pca.explained_variance_ratio_[:2])),
        "equilateral_weighted_pc3_cumulative_explained_variance_ratio": float(np.sum(equilateral_pca.explained_variance_ratio_[:3])),
        "random_weighted_pc1_explained_variance_ratio": float(random_pca.explained_variance_ratio_[0]),
        "random_weighted_pc3_cumulative_explained_variance_ratio": float(np.sum(random_pca.explained_variance_ratio_[:3])),
        "random_weighted_pc5_cumulative_explained_variance_ratio": float(np.sum(random_pca.explained_variance_ratio_[:5])),
        "random_weighted_pc7_cumulative_explained_variance_ratio": float(np.sum(random_pca.explained_variance_ratio_[:7])),
    }
    with open(os.path.join(OUTPUT_DIR, "weighted_multisource_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
