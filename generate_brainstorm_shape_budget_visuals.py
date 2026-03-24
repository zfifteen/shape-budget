"""
Brainstorm visuals for the Shape Budget hypothesis.

These plots focus on the governing ratio

    lambda = separation / total_reach = c / a

and try to make the main intuition legible:
- the same normalized ratio should imply the same normalized shape
- absolute scale should wash out after normalization
- the ratio partitions the two-source system into distinct compression regimes

Outputs are written to ./plots/brainstorm/
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
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


OUTPUT_DIR = os.path.join("plots", "brainstorm")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ellipse_params(lambda_ratio: float, total_reach: float = 2.0) -> tuple[float, float, float]:
    """Return semimajor axis a, focal offset c, and semiminor axis b."""
    a = total_reach / 2.0
    c = lambda_ratio * a
    b = np.sqrt(max(a * a - c * c, 0.0))
    return a, c, b


def draw_ellipse(ax, lambda_ratio: float, total_reach: float = 2.0, color: str = "#1f6fb2") -> None:
    a, c, b = ellipse_params(lambda_ratio, total_reach)
    ellipse = Ellipse(
        (0.0, 0.0),
        width=2 * a,
        height=2 * b,
        edgecolor=color,
        facecolor=sns.desaturate(color, 0.65),
        alpha=0.18,
        lw=2.8,
    )
    ax.add_patch(ellipse)
    ax.scatter([-c, c], [0, 0], color="#d62828", s=36, zorder=3)


def make_governor_gallery() -> str:
    lambda_values = [0.00, 0.20, 0.40, 0.60, 0.80, 0.95]
    regime_labels = [
        "Co-located regime",
        "Slack-rich regime",
        "Balanced regime",
        "Compression onset",
        "Pinched regime",
        "Near-degenerate regime",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), constrained_layout=True)

    for ax, lam, regime in zip(axes.ravel(), lambda_values, regime_labels):
        a, _, b = ellipse_params(lam)
        residue = b / a if a else 0.0
        draw_ellipse(ax, lam)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15 if lam < 0.9 else -0.45, 1.15 if lam < 0.9 else 0.45)
        ax.set_aspect("equal")
        ax.set_title(f"{regime}\nlambda = {lam:.2f}, residue = {residue:.2f}")
        ax.set_xlabel("Expansion axis")
        ax.set_ylabel("Residual spread")
        ax.text(
            0.03,
            0.93,
            f"bridge load = {lam:.0%}\nspread left = {residue:.0%}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.95),
        )

    fig.suptitle(
        "Brainstorm View 1: Same Total Reach, Different Separation Load\n"
        "The family of shapes reads like a compression governor",
        fontsize=17,
        fontweight="bold",
    )
    path = os.path.join(OUTPUT_DIR, "shape_budget_governor_gallery.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_phase_map() -> str:
    d_vals = np.linspace(0.0, 10.0, 420)
    s_vals = np.linspace(0.2, 10.0, 420)
    d_grid, s_grid = np.meshgrid(d_vals, s_vals)
    lambda_grid = d_grid / s_grid
    residue_grid = np.sqrt(np.clip(1.0 - lambda_grid**2, 0.0, 1.0))
    residue_grid = np.where(lambda_grid <= 1.0, residue_grid, np.nan)

    cmap = LinearSegmentedColormap.from_list(
        "shape_budget",
        ["#8b1e3f", "#d95d39", "#f2cc8f", "#98c1d9", "#2a9d8f"],
    )

    fig, ax = plt.subplots(figsize=(11.5, 9.0), constrained_layout=True)
    mesh = ax.pcolormesh(d_grid, s_grid, residue_grid, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    contours = ax.contour(
        d_grid,
        s_grid,
        np.where(lambda_grid <= 1.0, lambda_grid, np.nan),
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="white",
        linewidths=1.4,
        alpha=0.95,
    )
    ax.clabel(contours, inline=True, fmt=lambda value: f"lambda={value:.1f}", fontsize=10)
    ax.plot(d_vals, d_vals, color="#222222", linestyle="--", lw=1.5, alpha=0.8)
    ax.text(7.1, 7.35, "degenerate boundary\nseparation = total reach", fontsize=10, color="#222222")

    sample_points = [
        (1.0, 5.0, "same lambda family"),
        (2.0, 10.0, "same lambda family"),
        (5.0, 8.0, "pinched"),
    ]
    for d_val, s_val, label in sample_points:
        ax.scatter(d_val, s_val, color="#111111", s=34, zorder=4)
        ax.text(d_val + 0.15, s_val + 0.1, label, fontsize=9, color="#111111")

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.92)
    cbar.set_label("Residual spread ratio")

    ax.set_title(
        "Brainstorm View 2: Phase Map of Separation vs Total Reach\n"
        "If your idea is right, normalized shape should depend on the ratio field, not absolute scale",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Source separation d")
    ax.set_ylabel("Total reach budget S")

    path = os.path.join(OUTPUT_DIR, "shape_budget_phase_map.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_scale_collapse() -> str:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.0, 6.8), constrained_layout=True)

    same_lambda = 0.60
    scales = [2.0, 4.0, 6.0]
    colors = ["#2a9d8f", "#e76f51", "#264653"]

    for total_reach, color in zip(scales, colors):
        a, c, b = ellipse_params(same_lambda, total_reach=total_reach)
        ellipse = Ellipse((0.0, 0.0), 2 * a, 2 * b, edgecolor=color, facecolor="none", lw=2.6)
        ax_left.add_patch(ellipse)
        ax_left.scatter([-c, c], [0, 0], color=color, s=26)
        ax_left.text(a + 0.12, b, f"S={total_reach:.0f}", color=color, fontsize=10)

    ax_left.set_title("Different absolute scales, same lambda = 0.60")
    ax_left.set_xlabel("Absolute expansion axis")
    ax_left.set_ylabel("Absolute residual spread")
    ax_left.set_xlim(-3.4, 3.4)
    ax_left.set_ylim(-2.4, 2.4)
    ax_left.set_aspect("equal")

    for lam, color in zip([0.20, 0.60, 0.90], ["#2a9d8f", "#264653", "#8b1e3f"]):
        draw_ellipse(ax_right, lam, total_reach=2.0, color=color)

    ax_right.set_title("After normalization, shape families separate by lambda")
    ax_right.set_xlabel("Normalized expansion axis (divide by a)")
    ax_right.set_ylabel("Normalized residual spread")
    ax_right.set_xlim(-1.15, 1.15)
    ax_right.set_ylim(-1.15, 1.15)
    ax_right.set_aspect("equal")
    ax_right.legend(
        handles=[
            plt.Line2D([0], [0], color="#2a9d8f", lw=2.6, label="lambda = 0.20"),
            plt.Line2D([0], [0], color="#264653", lw=2.6, label="lambda = 0.60"),
            plt.Line2D([0], [0], color="#8b1e3f", lw=2.6, label="lambda = 0.90"),
        ],
        loc="lower left",
        frameon=True,
    )

    fig.suptitle(
        "Brainstorm View 3: Scale Collapse\n"
        "A computational primitive often shows up as a normalized family collapse",
        fontsize=16,
        fontweight="bold",
    )
    path = os.path.join(OUTPUT_DIR, "shape_budget_scale_collapse.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_ratio_curves() -> str:
    lam = np.linspace(0.0, 0.999, 500)
    residue = np.sqrt(1.0 - lam**2)

    fig, ax = plt.subplots(figsize=(11.8, 7.0), constrained_layout=True)
    ax.plot(lam, lam, color="#d62828", lw=3, label="bridge load ratio")
    ax.plot(lam, residue, color="#2a9d8f", lw=3, label="residual spread ratio")
    ax.fill_between(lam, 0.0, residue, color="#2a9d8f", alpha=0.18)
    ax.fill_between(lam, residue, 1.0, color="#d62828", alpha=0.10)

    ax.axvspan(0.0, 0.25, color="#9fd3c7", alpha=0.18)
    ax.axvspan(0.25, 0.65, color="#f2cc8f", alpha=0.16)
    ax.axvspan(0.65, 1.0, color="#e5989b", alpha=0.16)

    ax.text(0.08, 0.17, "slack-rich", fontsize=11)
    ax.text(0.39, 0.17, "mixed", fontsize=11)
    ax.text(0.79, 0.17, "pinched", fontsize=11)

    ax.set_title(
        "Brainstorm View 4: The Governor Curve\n"
        "A single ratio moves the system from slack-rich to pinched",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Normalized separation load lambda = d / S = c / a")
    ax.set_ylabel("Normalized ratio")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="center right")

    path = os.path.join(OUTPUT_DIR, "shape_budget_governor_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    paths = [
        make_governor_gallery(),
        make_phase_map(),
        make_scale_collapse(),
        make_ratio_curves(),
    ]
    print("Generated brainstorm visuals:")
    for path in paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
