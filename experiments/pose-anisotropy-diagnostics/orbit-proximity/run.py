"""
Post-roadmap extension: orbit-proximity diagnostic for pose-free anisotropic inversion.

This experiment tests the symmetry-orbit mechanism directly in the same
centroid-centered mean-radius-normalized radial-signature space used by the
inverse artifacts.

Question:

- are alpha perturbations more easily absorbed by cyclic rotation than matched
  geometry perturbations are?
"""

from __future__ import annotations

import sys
from pathlib import Path

_COMPAT_MODULES = Path(__file__).resolve().parents[3] / ".experiment_modules"
if str(_COMPAT_MODULES) not in sys.path:
    sys.path.insert(0, str(_COMPAT_MODULES))

import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from run_weighted_anisotropic_inverse_experiment import (
    ALPHA_MAX,
    ALPHA_MIN,
    GEOMETRY_BOUNDS,
    anisotropic_forward_signature,
)
from run_weighted_multisource_inverse_experiment import SIGNATURE_ANGLE_COUNT, write_csv


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


BASE_STATE_COUNT = 96
STEP_FRACTIONS = [0.03, 0.06, 0.10, 0.14]
AUDIT_RANDOM_CASES = 40
PLOT_STEP_FOR_SCATTER = 0.10


PARAM_RANGES = {
    "rho": GEOMETRY_BOUNDS["rho_max"] - GEOMETRY_BOUNDS["rho_min"],
    "t": GEOMETRY_BOUNDS["t_max"] - GEOMETRY_BOUNDS["t_min"],
    "h": GEOMETRY_BOUNDS["h_max"] - GEOMETRY_BOUNDS["h_min"],
    "alpha": ALPHA_MAX - ALPHA_MIN,
}


@dataclass
class PerturbationRow:
    base_index: int
    perturbation_type: str
    step_fraction: float
    sign: int
    base_rho: float
    base_t: float
    base_h: float
    base_w1: float
    base_w2: float
    base_w3: float
    base_alpha: float
    delta_rho: float
    delta_t: float
    delta_h: float
    delta_alpha: float
    raw_rmse: float
    orbit_rmse: float
    orbit_absorption_fraction: float
    orbit_ratio: float
    best_shift: int
    nontrivial_best_shift: int


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def shift_stack(signature: np.ndarray) -> np.ndarray:
    return np.stack([np.roll(signature, shift) for shift in range(len(signature))], axis=0)


def orbit_metrics(base_signature: np.ndarray, perturbed_signature: np.ndarray) -> tuple[float, float, float, float, int]:
    raw = rmse(base_signature, perturbed_signature)
    stack = shift_stack(base_signature)
    distances = np.sqrt(np.mean((stack - perturbed_signature[None, :]) ** 2, axis=1))
    best_shift = int(np.argmin(distances))
    orbit = float(distances[best_shift])
    ratio = orbit / max(raw, 1.0e-12)
    absorption = 1.0 - ratio
    return raw, orbit, absorption, ratio, best_shift


def sample_interior_parameters(
    rng: np.random.Generator,
    max_step_fraction: float,
) -> tuple[float, float, float, float, float, float]:
    rho_margin = max_step_fraction * PARAM_RANGES["rho"]
    t_margin = max_step_fraction * PARAM_RANGES["t"]
    h_margin = max_step_fraction * PARAM_RANGES["h"]
    alpha_margin = max_step_fraction * PARAM_RANGES["alpha"]

    rho = float(rng.uniform(GEOMETRY_BOUNDS["rho_min"] + rho_margin, GEOMETRY_BOUNDS["rho_max"] - rho_margin))
    t = float(rng.uniform(GEOMETRY_BOUNDS["t_min"] + t_margin, GEOMETRY_BOUNDS["t_max"] - t_margin))
    h = float(rng.uniform(GEOMETRY_BOUNDS["h_min"] + h_margin, GEOMETRY_BOUNDS["h_max"] - h_margin))
    weights = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
    alpha = float(rng.uniform(ALPHA_MIN + alpha_margin, ALPHA_MAX - alpha_margin))
    return rho, t, h, float(weights[0]), float(weights[1]), alpha


def geometry_unit_direction(rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return direction


def delta_from_type(
    perturbation_type: str,
    step_fraction: float,
    sign: int,
    geometry_direction: np.ndarray,
) -> tuple[float, float, float, float]:
    if perturbation_type == "alpha":
        return 0.0, 0.0, 0.0, sign * step_fraction * PARAM_RANGES["alpha"]
    if perturbation_type == "rho":
        return sign * step_fraction * PARAM_RANGES["rho"], 0.0, 0.0, 0.0
    if perturbation_type == "t":
        return 0.0, sign * step_fraction * PARAM_RANGES["t"], 0.0, 0.0
    if perturbation_type == "h":
        return 0.0, 0.0, sign * step_fraction * PARAM_RANGES["h"], 0.0
    if perturbation_type == "geometry_random":
        return (
            sign * step_fraction * PARAM_RANGES["rho"] * float(geometry_direction[0]),
            sign * step_fraction * PARAM_RANGES["t"] * float(geometry_direction[1]),
            sign * step_fraction * PARAM_RANGES["h"] * float(geometry_direction[2]),
            0.0,
        )
    raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def summarized_rows(rows: list[PerturbationRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    perturbation_types = ["alpha", "geometry_random", "rho", "t", "h"]
    for perturbation_type in perturbation_types:
        for step_fraction in STEP_FRACTIONS:
            subset = [
                row
                for row in rows
                if row.perturbation_type == perturbation_type and math.isclose(row.step_fraction, step_fraction, rel_tol=0.0, abs_tol=1.0e-12)
            ]
            raw = np.array([row.raw_rmse for row in subset])
            orbit = np.array([row.orbit_rmse for row in subset])
            absorption = np.array([row.orbit_absorption_fraction for row in subset])
            ratio = np.array([row.orbit_ratio for row in subset])
            nontrivial = np.array([row.nontrivial_best_shift for row in subset], dtype=float)
            summary.append(
                {
                    "perturbation_type": perturbation_type,
                    "step_fraction": step_fraction,
                    "count": len(subset),
                    "raw_rmse_mean": float(np.mean(raw)),
                    "raw_rmse_p95": float(np.quantile(raw, 0.95)),
                    "orbit_rmse_mean": float(np.mean(orbit)),
                    "orbit_rmse_p95": float(np.quantile(orbit, 0.95)),
                    "orbit_absorption_mean": float(np.mean(absorption)),
                    "orbit_absorption_p95": float(np.quantile(absorption, 0.95)),
                    "orbit_ratio_mean": float(np.mean(ratio)),
                    "orbit_ratio_p95": float(np.quantile(ratio, 0.95)),
                    "nontrivial_best_shift_fraction": float(np.mean(nontrivial)),
                }
            )
    return summary


def pre_benchmark_audit(rng: np.random.Generator) -> dict[str, float | int]:
    max_shift_identity_rmse = 0.0
    max_ordering_violation = 0.0
    min_absorption = 1.0
    max_absorption = 0.0

    for _ in range(AUDIT_RANDOM_CASES):
        params = sample_interior_parameters(rng, max(STEP_FRACTIONS))
        signature = anisotropic_forward_signature(params)
        shift = int(rng.integers(0, SIGNATURE_ANGLE_COUNT))
        shifted = np.roll(signature, shift)
        _, orbit, _, _, _ = orbit_metrics(signature, shifted)
        max_shift_identity_rmse = max(max_shift_identity_rmse, orbit)

        direction = geometry_unit_direction(rng)
        perturbation_type = str(rng.choice(np.array(["alpha", "rho", "t", "h", "geometry_random"])))
        step_fraction = float(rng.choice(np.array(STEP_FRACTIONS)))
        sign = -1 if int(rng.integers(0, 2)) == 0 else 1
        delta_rho, delta_t, delta_h, delta_alpha = delta_from_type(perturbation_type, step_fraction, sign, direction)
        perturbed_params = (
            params[0] + delta_rho,
            params[1] + delta_t,
            params[2] + delta_h,
            params[3],
            params[4],
            params[5] + delta_alpha,
        )
        perturbed_signature = anisotropic_forward_signature(perturbed_params)
        raw, orbit, absorption, _, _ = orbit_metrics(signature, perturbed_signature)
        max_ordering_violation = max(max_ordering_violation, orbit - raw)
        min_absorption = min(min_absorption, absorption)
        max_absorption = max(max_absorption, absorption)

    return {
        "audit_random_cases": AUDIT_RANDOM_CASES,
        "max_shift_identity_rmse": float(max_shift_identity_rmse),
        "max_orbit_minus_raw_violation": float(max_ordering_violation),
        "min_absorption_fraction": float(min_absorption),
        "max_absorption_fraction": float(max_absorption),
    }


def plot_absorption_curves(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.09, right=0.98)

    order = ["alpha", "geometry_random", "rho", "t", "h"]
    palette = {
        "alpha": "#d62828",
        "geometry_random": "#1d3557",
        "rho": "#2a9d8f",
        "t": "#e9c46a",
        "h": "#6a4c93",
    }
    labels = {
        "alpha": "alpha",
        "geometry_random": "geometry random",
        "rho": "rho",
        "t": "t",
        "h": "h",
    }

    for perturbation_type in order:
        subset = [row for row in summary_rows if row["perturbation_type"] == perturbation_type]
        x = np.array([float(row["step_fraction"]) for row in subset])
        y = np.array([float(row["orbit_absorption_mean"]) for row in subset])
        ax.plot(x, y, marker="o", lw=2.5, color=palette[perturbation_type], label=labels[perturbation_type])

    ax.set_xlabel("normalized latent step fraction")
    ax.set_ylabel("mean orbit absorption fraction")
    ax.set_title("Orbit proximity by perturbation type")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.suptitle("Orbit Proximity A: Alpha Versus Geometry Under Rotation-Orbit Matching", fontsize=15, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_distance_scatter(path: str, rows: list[PerturbationRow]) -> None:
    chosen = [
        row
        for row in rows
        if math.isclose(row.step_fraction, PLOT_STEP_FOR_SCATTER, rel_tol=0.0, abs_tol=1.0e-12)
        and row.perturbation_type in {"alpha", "geometry_random"}
    ]

    fig, ax = plt.subplots(figsize=(7.6, 6.2), constrained_layout=False)
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.13, right=0.98)

    palette = {"alpha": "#d62828", "geometry_random": "#1d3557"}
    for perturbation_type in ["alpha", "geometry_random"]:
        subset = [row for row in chosen if row.perturbation_type == perturbation_type]
        ax.scatter(
            [row.raw_rmse for row in subset],
            [row.orbit_rmse for row in subset],
            s=34,
            alpha=0.72,
            color=palette[perturbation_type],
            label=perturbation_type,
            edgecolors="none",
        )

    max_val = max(max(row.raw_rmse for row in chosen), max(row.orbit_rmse for row in chosen))
    ax.plot([0.0, max_val], [0.0, max_val], color="#444444", linestyle="--", lw=1.2)
    ax.set_xlabel("raw signature RMSE")
    ax.set_ylabel("best-rotation-orbit RMSE")
    ax.set_title(f"Step fraction = {PLOT_STEP_FOR_SCATTER:.2f}")
    ax.legend(loc="upper left", frameon=True)
    fig.suptitle("Orbit Proximity B: Rotation Absorbs Alpha More Than Geometry", fontsize=15, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_shift_fraction(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.09, right=0.98)

    order = ["alpha", "geometry_random", "rho", "t", "h"]
    palette = {
        "alpha": "#d62828",
        "geometry_random": "#1d3557",
        "rho": "#2a9d8f",
        "t": "#e9c46a",
        "h": "#6a4c93",
    }

    x = np.arange(len(STEP_FRACTIONS))
    width = 0.15
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(order))
    for offset, perturbation_type in zip(offsets, order):
        subset = [row for row in summary_rows if row["perturbation_type"] == perturbation_type]
        y = np.array([float(row["nontrivial_best_shift_fraction"]) for row in subset])
        ax.bar(x + offset, y, width=width, color=palette[perturbation_type], label=perturbation_type)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{step:.2f}" for step in STEP_FRACTIONS])
    ax.set_xlabel("normalized latent step fraction")
    ax.set_ylabel("fraction with nonzero best shift")
    ax.set_title("How often rotation meaningfully absorbs the perturbation")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.suptitle("Orbit Proximity C: Nontrivial Orbit Absorption Frequency", fontsize=15, fontweight="bold", y=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)
    audit = pre_benchmark_audit(rng)

    rows: list[PerturbationRow] = []
    perturbation_types = ["alpha", "geometry_random", "rho", "t", "h"]
    max_step_fraction = max(STEP_FRACTIONS)

    for base_index in range(BASE_STATE_COUNT):
        params = sample_interior_parameters(rng, max_step_fraction)
        base_signature = anisotropic_forward_signature(params)
        geometry_direction = geometry_unit_direction(rng)
        w3 = 1.0 - params[3] - params[4]

        for step_fraction in STEP_FRACTIONS:
            for perturbation_type in perturbation_types:
                for sign in (-1, 1):
                    delta_rho, delta_t, delta_h, delta_alpha = delta_from_type(
                        perturbation_type,
                        step_fraction,
                        sign,
                        geometry_direction,
                    )
                    perturbed_params = (
                        params[0] + delta_rho,
                        params[1] + delta_t,
                        params[2] + delta_h,
                        params[3],
                        params[4],
                        params[5] + delta_alpha,
                    )
                    perturbed_signature = anisotropic_forward_signature(perturbed_params)
                    raw, orbit, absorption, ratio, best_shift = orbit_metrics(base_signature, perturbed_signature)

                    rows.append(
                        PerturbationRow(
                            base_index=base_index,
                            perturbation_type=perturbation_type,
                            step_fraction=float(step_fraction),
                            sign=int(sign),
                            base_rho=float(params[0]),
                            base_t=float(params[1]),
                            base_h=float(params[2]),
                            base_w1=float(params[3]),
                            base_w2=float(params[4]),
                            base_w3=float(w3),
                            base_alpha=float(params[5]),
                            delta_rho=float(delta_rho),
                            delta_t=float(delta_t),
                            delta_h=float(delta_h),
                            delta_alpha=float(delta_alpha),
                            raw_rmse=float(raw),
                            orbit_rmse=float(orbit),
                            orbit_absorption_fraction=float(absorption),
                            orbit_ratio=float(ratio),
                            best_shift=int(best_shift),
                            nontrivial_best_shift=int(best_shift != 0),
                        )
                    )

    summary_rows = summarized_rows(rows)

    write_csv(os.path.join(OUTPUT_DIR, "orbit_proximity_rows.csv"), [row.__dict__ for row in rows])
    write_csv(os.path.join(OUTPUT_DIR, "orbit_proximity_summary.csv"), summary_rows)

    plot_absorption_curves(os.path.join(FIGURE_DIR, "orbit_proximity_absorption_curves.png"), summary_rows)
    plot_distance_scatter(os.path.join(FIGURE_DIR, "orbit_proximity_distance_scatter.png"), rows)
    plot_shift_fraction(os.path.join(FIGURE_DIR, "orbit_proximity_shift_fraction.png"), summary_rows)

    alpha_rows = [row for row in summary_rows if row["perturbation_type"] == "alpha"]
    geometry_rows = [row for row in summary_rows if row["perturbation_type"] == "geometry_random"]
    alpha_by_step = {float(row["step_fraction"]): row for row in alpha_rows}
    geometry_by_step = {float(row["step_fraction"]): row for row in geometry_rows}

    summary = {
        "base_state_count": BASE_STATE_COUNT,
        "step_fractions": STEP_FRACTIONS,
        "audit": audit,
        "largest_alpha_minus_geometry_random_absorption_gap": float(
            max(
                float(alpha_by_step[step]["orbit_absorption_mean"]) - float(geometry_by_step[step]["orbit_absorption_mean"])
                for step in STEP_FRACTIONS
            )
        ),
        "smallest_alpha_minus_geometry_random_absorption_gap": float(
            min(
                float(alpha_by_step[step]["orbit_absorption_mean"]) - float(geometry_by_step[step]["orbit_absorption_mean"])
                for step in STEP_FRACTIONS
            )
        ),
        "largest_alpha_over_geometry_random_orbit_ratio_advantage": float(
            max(
                float(geometry_by_step[step]["orbit_ratio_mean"]) / max(float(alpha_by_step[step]["orbit_ratio_mean"]), 1.0e-12)
                for step in STEP_FRACTIONS
            )
        ),
        "smallest_alpha_over_geometry_random_orbit_ratio_advantage": float(
            min(
                float(geometry_by_step[step]["orbit_ratio_mean"]) / max(float(alpha_by_step[step]["orbit_ratio_mean"]), 1.0e-12)
                for step in STEP_FRACTIONS
            )
        ),
        "largest_alpha_nontrivial_shift_fraction_minus_geometry_random": float(
            max(
                float(alpha_by_step[step]["nontrivial_best_shift_fraction"]) - float(geometry_by_step[step]["nontrivial_best_shift_fraction"])
                for step in STEP_FRACTIONS
            )
        ),
        "smallest_alpha_nontrivial_shift_fraction_minus_geometry_random": float(
            min(
                float(alpha_by_step[step]["nontrivial_best_shift_fraction"]) - float(geometry_by_step[step]["nontrivial_best_shift_fraction"])
                for step in STEP_FRACTIONS
            )
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "orbit_proximity_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
