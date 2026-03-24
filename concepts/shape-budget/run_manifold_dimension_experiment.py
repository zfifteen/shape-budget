"""
Experiment 6: manifold-dimension test for Shape Budget.

This experiment tests whether the symmetric constant-sum two-source family,
after normalization by scale, behaves like a one-dimensional manifold in
boundary space.

To avoid baking the answer into the original generation parameter, shapes are
represented as radial signatures on a fixed polar grid rather than as raw
process samples.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


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
OUTPUT_DIR = os.path.join(BASE_DIR, "manifold_dimension_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


@dataclass
class SpectrumRow:
    component: int
    explained_variance_ratio: float
    cumulative_explained_variance_ratio: float


@dataclass
class ReconstructionRow:
    components: int
    mean_relative_error: float
    max_relative_error: float


@dataclass
class EmbeddingRow:
    dataset: str
    e: float
    a: float
    pc1: float
    pc2: float
    isomap1: float


def ellipse_parameters(a: float, e: float) -> tuple[float, float, float]:
    c = e * a
    b = math.sqrt(max(a * a - c * c, 0.0))
    return a, b, c


def normalized_radii_family(a: float, c: float, sample_count: int = 800) -> np.ndarray:
    if c == 0.0:
        return np.full(sample_count, a)
    s = np.linspace(-1.0, 1.0, sample_count)
    return a + c * s


def constant_sum_locus_points(a: float, e: float, sample_count: int = 800) -> np.ndarray:
    a, b, c = ellipse_parameters(a, e)
    if c == 0.0:
        theta = np.linspace(0.0, 2.0 * math.pi, 2 * sample_count, endpoint=False)
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


def radial_signature(points: np.ndarray, angle_count: int = 360) -> tuple[np.ndarray, np.ndarray]:
    angle = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
    radius = np.linalg.norm(points, axis=1)
    order = np.argsort(angle)
    angle = angle[order]
    radius = radius[order]

    # Extend periodically so interpolation at the wrap-around stays stable.
    angle_ext = np.concatenate([angle[-1:] - 2.0 * math.pi, angle, angle[:1] + 2.0 * math.pi])
    radius_ext = np.concatenate([radius[-1:], radius, radius[:1]])
    grid = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    signature = np.interp(grid, angle_ext, radius_ext)
    return grid, signature


def boundary_signature(a: float, e: float, angle_count: int = 360, sample_count: int = 800) -> tuple[np.ndarray, np.ndarray]:
    points = constant_sum_locus_points(a, e, sample_count=sample_count) / a
    return radial_signature(points, angle_count=angle_count)


def signatures_to_matrix(e_values: np.ndarray, a: float, angle_count: int, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    signatures = []
    angle_grid = None
    for e in e_values:
        angle_grid, sig = boundary_signature(float(a), float(e), angle_count=angle_count, sample_count=sample_count)
        signatures.append(sig)
    return angle_grid, np.array(signatures)


def write_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pca_reconstruction_errors(pca: PCA, matrix: np.ndarray, component_counts: list[int]) -> list[ReconstructionRow]:
    rows: list[ReconstructionRow] = []
    scores = pca.transform(matrix)
    norms = np.linalg.norm(matrix, axis=1)
    for k in component_counts:
        recon = scores[:, :k] @ pca.components_[:k] + pca.mean_
        rel_error = np.linalg.norm(recon - matrix, axis=1) / norms
        rows.append(
            ReconstructionRow(
                components=k,
                mean_relative_error=float(np.mean(rel_error)),
                max_relative_error=float(np.max(rel_error)),
            )
        )
    return rows


def signature_to_points(angle_grid: np.ndarray, signature: np.ndarray) -> np.ndarray:
    return np.column_stack([signature * np.cos(angle_grid), signature * np.sin(angle_grid)])


def plot_embedding(path: str, unique_rows: list[EmbeddingRow], scale_rows: list[EmbeddingRow], min_between: float, max_within_by_e: list[tuple[float, float]]) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.24)

    unique_pc1 = np.array([row.pc1 for row in unique_rows])
    unique_pc2 = np.array([row.pc2 for row in unique_rows])
    unique_e = np.array([row.e for row in unique_rows])
    scatter = ax_left.scatter(unique_pc1, unique_pc2, c=unique_e, cmap="viridis", s=18, alpha=0.9)
    ax_left.plot(unique_pc1, unique_pc2, color="#264653", alpha=0.25, lw=1.2)
    ax_left.set_title("Normalized boundaries form a thin curve in PCA space")
    ax_left.set_xlabel("PC1")
    ax_left.set_ylabel("PC2")

    cbar = fig.colorbar(scatter, ax=ax_left, shrink=0.9)
    cbar.set_label("e")

    e_vals = np.array([item[0] for item in max_within_by_e])
    spreads = np.array([item[1] for item in max_within_by_e])
    ax_right.plot(e_vals, spreads, color="#457b9d", lw=2.4, label="max within-e across-scale spread")
    ax_right.axhline(min_between, color="#d62828", linestyle="--", lw=2.0, label="smallest gap between neighboring e means")
    ax_right.set_yscale("log")
    ax_right.set_title("Scale collapse remains clean in embedding space")
    ax_right.set_xlabel("e")
    ax_right.set_ylabel("distance in PC1-PC2 plane (log scale)")
    ax_right.legend(loc="upper left", frameon=True)

    fig.suptitle("Experiment 6A: Boundary-Space Embedding", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_and_reconstruction(path: str, spectrum_rows: list[SpectrumRow], reconstruction_rows: list[ReconstructionRow]) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.2, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.84, wspace=0.26)

    components = [row.component for row in spectrum_rows]
    ratios = [row.explained_variance_ratio for row in spectrum_rows]
    cumulative = [row.cumulative_explained_variance_ratio for row in spectrum_rows]
    ax_left.bar(components, ratios, color="#2a9d8f", alpha=0.85, label="per-component ratio")
    ax_left.plot(components, cumulative, color="#1d3557", lw=2.2, marker="o", label="cumulative ratio")
    ax_left.set_title("Singular spectrum of radial-signature family")
    ax_left.set_xlabel("principal component")
    ax_left.set_ylabel("explained variance ratio")
    ax_left.set_ylim(0.0, 1.02)
    ax_left.legend(loc="upper right", frameon=True)

    ks = [row.components for row in reconstruction_rows]
    mean_err = [row.mean_relative_error for row in reconstruction_rows]
    max_err = [row.max_relative_error for row in reconstruction_rows]
    ax_right.plot(ks, mean_err, color="#457b9d", lw=2.4, marker="o", label="mean relative error")
    ax_right.plot(ks, max_err, color="#e76f51", lw=2.4, marker="s", label="max relative error")
    ax_right.set_yscale("log")
    ax_right.set_title("Low-dimensional reconstruction accuracy")
    ax_right.set_xlabel("number of principal components")
    ax_right.set_ylabel("relative reconstruction error (log scale)")
    ax_right.legend(loc="upper right", frameon=True)

    fig.suptitle("Experiment 6B: Spectrum And Reconstruction", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_isomap_and_reconstructions(
    path: str,
    e_values: np.ndarray,
    isomap_1d: np.ndarray,
    angle_grid: np.ndarray,
    signatures: np.ndarray,
    pca: PCA,
) -> None:
    fig = plt.figure(figsize=(15.2, 9.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 4, top=0.9, wspace=0.28, hspace=0.28)

    ax_embed = fig.add_subplot(gs[:, 0])
    ax_embed.scatter(e_values, isomap_1d, c=e_values, cmap="viridis", s=22)
    ax_embed.plot(e_values, isomap_1d, color="#264653", alpha=0.25, lw=1.2)
    ax_embed.set_title("1D Isomap coordinate")
    ax_embed.set_xlabel("e")
    ax_embed.set_ylabel("isomap-1 coordinate")

    chosen_e = [0.10, 0.50, 0.90]
    for row_idx, target_e in enumerate(chosen_e):
        idx = int(np.argmin(np.abs(e_values - target_e)))
        true_signature = signatures[idx]
        scores = pca.transform(true_signature[None, :])
        recon_1 = scores[:, :1] @ pca.components_[:1] + pca.mean_
        recon_2 = scores[:, :2] @ pca.components_[:2] + pca.mean_
        variants = [
            ("original", true_signature),
            ("1-PC recon", recon_1[0]),
            ("2-PC recon", recon_2[0]),
        ]
        for col_idx, (label, signature) in enumerate(variants, start=1):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            pts = signature_to_points(angle_grid, signature)
            ax.plot(pts[:, 0], pts[:, 1], color="#1d3557", lw=2.2)
            ax.fill(pts[:, 0], pts[:, 1], color="#a8dadc", alpha=0.16)
            ax.set_aspect("equal")
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.08, 1.08)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(label)
            ax.text(
                0.04,
                0.92,
                f"e = {e_values[idx]:.2f}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.92),
            )

    fig.suptitle("Experiment 6C: Nonlinear Ordering And Shape Reconstructions", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    unique_e_values = np.round(np.linspace(0.02, 0.98, 193), 4)
    scale_e_values = np.round(np.linspace(0.05, 0.95, 37), 4)
    a_values = [0.75, 1.0, 1.5, 2.5, 4.0]
    angle_count = 360
    sample_count = 800
    component_counts = [1, 2, 3, 5, 10]

    angle_grid, unique_signatures = signatures_to_matrix(unique_e_values, a=1.0, angle_count=angle_count, sample_count=sample_count)

    pca = PCA().fit(unique_signatures)
    unique_scores = pca.transform(unique_signatures)
    pc1_sign = -1.0 if spearmanr(unique_scores[:, 0], unique_e_values).statistic < 0 else 1.0
    unique_scores[:, 0] *= pc1_sign

    isomap = Isomap(n_neighbors=8, n_components=1)
    isomap_1d = isomap.fit_transform(unique_signatures)[:, 0]
    if spearmanr(isomap_1d, unique_e_values).statistic < 0:
        isomap_1d *= -1.0

    spectrum_rows = []
    cumulative = 0.0
    for idx, ratio in enumerate(pca.explained_variance_ratio_[:10], start=1):
        cumulative += float(ratio)
        spectrum_rows.append(
            SpectrumRow(
                component=idx,
                explained_variance_ratio=float(ratio),
                cumulative_explained_variance_ratio=float(cumulative),
            )
        )

    reconstruction_rows = pca_reconstruction_errors(pca, unique_signatures, component_counts)

    unique_embedding_rows = [
        EmbeddingRow(
            dataset="unique",
            e=float(e),
            a=1.0,
            pc1=float(pc1),
            pc2=float(pc2),
            isomap1=float(iso),
        )
        for e, pc1, pc2, iso in zip(unique_e_values, unique_scores[:, 0], unique_scores[:, 1], isomap_1d)
    ]

    scale_embedding_rows: list[EmbeddingRow] = []
    scale_spread_pairs: list[tuple[float, float]] = []
    mean_curve: list[np.ndarray] = []
    for e in scale_e_values:
        signatures = []
        for a in a_values:
            _, signature = boundary_signature(float(a), float(e), angle_count=angle_count, sample_count=sample_count)
            signatures.append(signature)
        signature_matrix = np.array(signatures)
        scores = pca.transform(signature_matrix)
        scores[:, 0] *= pc1_sign
        mean_point = scores[:, :2].mean(axis=0)
        mean_curve.append(mean_point)
        scale_spread_pairs.append((float(e), float(np.max(np.linalg.norm(scores[:, :2] - mean_point, axis=1)))))
        for a, (pc1, pc2) in zip(a_values, scores[:, :2]):
            scale_embedding_rows.append(
                EmbeddingRow(
                    dataset="scale_overlay",
                    e=float(e),
                    a=float(a),
                    pc1=float(pc1),
                    pc2=float(pc2),
                    isomap1=float("nan"),
                )
            )

    mean_curve = np.array(mean_curve)
    between_neighbor_distances = np.linalg.norm(np.diff(mean_curve, axis=0), axis=1)
    min_between_neighbor_distance = float(np.min(between_neighbor_distances))

    summary = {
        "representation": {
            "shape_encoding": "normalized radial signature",
            "angle_count": angle_count,
            "process_sample_count": sample_count,
        },
        "manifold_metrics": {
            "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "pc2_cumulative_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_[:2])),
            "pc3_cumulative_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_[:3])),
            "one_pc_mean_relative_error": reconstruction_rows[0].mean_relative_error,
            "one_pc_max_relative_error": reconstruction_rows[0].max_relative_error,
            "two_pc_mean_relative_error": reconstruction_rows[1].mean_relative_error,
            "two_pc_max_relative_error": reconstruction_rows[1].max_relative_error,
            "three_pc_mean_relative_error": reconstruction_rows[2].mean_relative_error,
            "three_pc_max_relative_error": reconstruction_rows[2].max_relative_error,
            "five_pc_mean_relative_error": reconstruction_rows[3].mean_relative_error,
            "five_pc_max_relative_error": reconstruction_rows[3].max_relative_error,
            "abs_isomap_spearman_rho_with_e": float(abs(spearmanr(isomap_1d, unique_e_values).statistic)),
            "max_within_e_scale_spread_pc12": float(max(spread for _, spread in scale_spread_pairs)),
            "mean_within_e_scale_spread_pc12": float(np.mean([spread for _, spread in scale_spread_pairs])),
            "min_between_neighbor_distance_pc12": min_between_neighbor_distance,
            "mean_between_neighbor_distance_pc12": float(np.mean(between_neighbor_distances)),
        },
    }

    write_csv(os.path.join(OUTPUT_DIR, "manifold_spectrum.csv"), [asdict(row) for row in spectrum_rows])
    write_csv(os.path.join(OUTPUT_DIR, "manifold_reconstruction_errors.csv"), [asdict(row) for row in reconstruction_rows])
    write_csv(
        os.path.join(OUTPUT_DIR, "manifold_embeddings.csv"),
        [asdict(row) for row in unique_embedding_rows + scale_embedding_rows],
    )

    with open(os.path.join(OUTPUT_DIR, "manifold_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_embedding(
        os.path.join(FIGURE_DIR, "manifold_embedding.png"),
        unique_embedding_rows,
        scale_embedding_rows,
        min_between_neighbor_distance,
        scale_spread_pairs,
    )
    plot_spectrum_and_reconstruction(
        os.path.join(FIGURE_DIR, "manifold_spectrum.png"),
        spectrum_rows,
        reconstruction_rows,
    )
    plot_isomap_and_reconstructions(
        os.path.join(FIGURE_DIR, "manifold_isomap_and_reconstructions.png"),
        unique_e_values,
        isomap_1d,
        angle_grid,
        unique_signatures,
        pca,
    )

    print("Manifold-dimension experiment complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
