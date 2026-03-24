"""
Probe Specialization Experiment for Shape Budget.

This experiment tests whether depletion phase predicts which measurement
strategy is the best inverse probe for recovering e.

Part A:
- ideal direct-measurement benchmark with matched relative scalar noise

Part B:
- practical equal-budget benchmark with probe-specific sampling strategies
- perimeter strategy: full-boundary sampling
- width strategy: extremum-focused sampling around major and minor vertices
- major-tip strategy: curvature-focused sampling near the major-axis tips
- adaptive router: a small perimeter pilot followed by a specialized probe
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import py_compile
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.special import ellipe

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

IDEAL_RELATIVE_NOISE = 0.01
WIDTH_WINDOW = 0.22
MAJOR_TIP_WINDOW = 0.18
ROUTER_PILOT_FRACTION = 0.25

EMPIRICAL_CONDITIONS = [
    {
        "name": "dense_low_noise",
        "total_points": 128,
        "noise_sigma_over_a": 0.0025,
    },
    {
        "name": "dense_medium_noise",
        "total_points": 96,
        "noise_sigma_over_a": 0.01,
    },
    {
        "name": "sparse_medium_noise",
        "total_points": 32,
        "noise_sigma_over_a": 0.01,
    },
    {
        "name": "sparse_high_noise",
        "total_points": 24,
        "noise_sigma_over_a": 0.02,
    },
]

PROBE_ORDER = ["perimeter", "width", "major_tip"]
PROBE_LABELS = {
    "perimeter": "perimeter",
    "width": "width",
    "major_tip": "major-tip curvature",
}
PROBE_COLORS = {
    "perimeter": "#457b9d",
    "width": "#2a9d8f",
    "major_tip": "#d62828",
    "router": "#6a4c93",
    "phase_oracle": "#8d5a97",
    "trial_oracle": "#264653",
}

@dataclass
class IdealMetricRow:
    e: float
    probe: str
    mae: float

@dataclass
class EmpiricalTrial:
    condition: str
    true_e: float
    replicate: int
    pilot_perimeter_estimate: float
    perimeter_estimate: float
    width_estimate: float
    major_tip_estimate: float
    pilot_abs_error: float
    perimeter_abs_error: float
    width_abs_error: float
    major_tip_abs_error: float

@dataclass
class RouterResult:
    condition: str
    method: str
    mean_abs_error: float

def width_residue(e: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(1.0 - e**2, 0.0))

def normalized_perimeter(e: np.ndarray) -> np.ndarray:
    return (2.0 / math.pi) * ellipe(e**2)

def major_tip_response(e: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(1.0 - e**2, 1e-300)

def e_from_width(q: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(1.0 - np.clip(q, 0.0, 1.0) ** 2, 0.0))

def e_from_major_tip(q: np.ndarray) -> np.ndarray:
    clipped = np.maximum(q, 1.0)
    return np.sqrt(np.maximum(1.0 - 1.0 / clipped, 0.0))

def invert_perimeter_scalar(q: float) -> float:
    q = float(np.clip(q, 2.0 / math.pi, 1.0))
    lo = 0.0
    hi = 0.999999999
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(normalized_perimeter(np.array([mid]))[0]) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def e_from_perimeter(q: np.ndarray) -> np.ndarray:
    return np.array([invert_perimeter_scalar(float(value)) for value in q])

def ellipse_points_from_theta(a_budget: float, e: float, theta: np.ndarray) -> np.ndarray:
    b = a_budget * math.sqrt(max(1.0 - e * e, 0.0))
    return np.column_stack([a_budget * np.cos(theta), b * np.sin(theta)])

def split_counts(total_points: int, parts: int) -> list[int]:
    base = total_points // parts
    remainder = total_points % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]

def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + 2.0 * math.pi) % (2.0 * math.pi)

def sample_uniform_boundary(a_budget: float, e: float, total_points: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, total_points, endpoint=False)
    return ellipse_points_from_theta(a_budget, e, theta)

def sample_theta_windows(
    a_budget: float,
    e: float,
    total_points: int,
    centers: list[float],
    half_width: float,
) -> np.ndarray:
    counts = split_counts(total_points, len(centers))
    windows = []
    for center, count in zip(centers, counts):
        if count <= 0:
            continue
        theta = np.linspace(center - half_width, center + half_width, count, endpoint=False)
        theta += half_width / max(count, 1)
        windows.append(wrap_angle(theta))
    all_theta = np.concatenate(windows) if windows else np.array([])
    return ellipse_points_from_theta(a_budget, e, all_theta)

def add_isotropic_noise(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return points + rng.normal(scale=sigma, size=points.shape)

def sample_extremum_strategy(a_budget: float, e: float, total_points: int) -> np.ndarray:
    return sample_theta_windows(a_budget, e, total_points, [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], WIDTH_WINDOW)

def sample_major_tip_strategy(a_budget: float, e: float, total_points: int) -> np.ndarray:
    return sample_theta_windows(a_budget, e, total_points, [0.0, math.pi], MAJOR_TIP_WINDOW)

def top_abs_axis_scale(values: np.ndarray, top_k: int = 3) -> float:
    abs_values = np.sort(np.abs(values))
    k = min(max(1, top_k), len(abs_values))
    return float(np.median(abs_values[-k:]))

def ordered_points_about_origin(points: np.ndarray) -> np.ndarray:
    theta = np.arctan2(points[:, 1], points[:, 0])
    return points[np.argsort(theta)]

def circular_smooth(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points
    if window % 2 == 0:
        window += 1
    pad = window // 2
    extended = np.vstack([points[-pad:], points, points[:pad]])
    kernel = np.ones(window) / window
    smooth_x = np.convolve(extended[:, 0], kernel, mode="valid")
    smooth_y = np.convolve(extended[:, 1], kernel, mode="valid")
    return np.column_stack([smooth_x, smooth_y])

def polygon_perimeter(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    return float(np.sum(np.linalg.norm(shifted - points, axis=1)))

def estimate_perimeter_probe(points: np.ndarray) -> float:
    ordered = ordered_points_about_origin(points)
    smooth_window = max(5, 2 * (len(ordered) // 64) + 1)
    smooth = circular_smooth(ordered, smooth_window)
    a_hat = max(top_abs_axis_scale(smooth[:, 0], top_k=3), 1e-12)
    q_hat = polygon_perimeter(smooth) / (2.0 * math.pi * a_hat)
    return float(np.clip(q_hat, 2.0 / math.pi, 1.0))

def estimate_width_probe(points: np.ndarray) -> float:
    a_hat = max(top_abs_axis_scale(points[:, 0], top_k=3), 1e-12)
    b_hat = max(top_abs_axis_scale(points[:, 1], top_k=3), 0.0)
    return float(np.clip(b_hat / a_hat, 0.0, 1.0))

def fit_circle_kasa(points: np.ndarray) -> tuple[float, float, float]:
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    rhs = x**2 + y**2
    cx, cy, c0 = np.linalg.lstsq(design, rhs, rcond=None)[0]
    radius = math.sqrt(max(c0 + cx * cx + cy * cy, 1e-12))
    return float(cx), float(cy), float(radius)

def estimate_major_tip_probe(points: np.ndarray) -> float:
    right = points[points[:, 0] >= 0.0]
    left = points[points[:, 0] < 0.0]
    responses = []
    for local in [right, left]:
        if len(local) < 4:
            continue
        _, _, radius = fit_circle_kasa(local)
        a_hat = max(top_abs_axis_scale(local[:, 0], top_k=3), 1e-12)
        responses.append(a_hat / radius)
    if not responses:
        return 1.0
    return float(max(np.mean(responses), 1.0))

def noisy_direct_measurement(
    true_value: np.ndarray,
    relative_noise: float,
    rng: np.random.Generator,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    noisy = true_value * (1.0 + rng.normal(scale=relative_noise, size=true_value.shape))
    if lower is not None or upper is not None:
        noisy = np.clip(noisy, lower if lower is not None else -np.inf, upper if upper is not None else np.inf)
    return noisy

def run_ideal_benchmark(e_values: np.ndarray, replicates: int, rng: np.random.Generator) -> list[IdealMetricRow]:
    rows: list[IdealMetricRow] = []
    for e in e_values:
        width_true = float(width_residue(np.array([e]))[0])
        perimeter_true = float(normalized_perimeter(np.array([e]))[0])
        major_true = float(major_tip_response(np.array([e]))[0])

        width_obs = noisy_direct_measurement(
            np.full(replicates, width_true),
            IDEAL_RELATIVE_NOISE,
            rng,
            lower=0.0,
            upper=1.0,
        )
        perimeter_obs = noisy_direct_measurement(
            np.full(replicates, perimeter_true),
            IDEAL_RELATIVE_NOISE,
            rng,
            lower=2.0 / math.pi,
            upper=1.0,
        )
        major_obs = noisy_direct_measurement(
            np.full(replicates, major_true),
            IDEAL_RELATIVE_NOISE,
            rng,
            lower=1.0,
            upper=None,
        )

        estimates = {
            "width": e_from_width(width_obs),
            "perimeter": e_from_perimeter(perimeter_obs),
            "major_tip": e_from_major_tip(major_obs),
        }
        for probe, values in estimates.items():
            rows.append(
                IdealMetricRow(
                    e=float(e),
                    probe=probe,
                    mae=float(np.mean(np.abs(values - e))),
                )
            )
    return rows

def run_empirical_trials(
    e_values: np.ndarray,
    replicates: int,
    a_budget: float,
    rng: np.random.Generator,
) -> list[EmpiricalTrial]:
    rows: list[EmpiricalTrial] = []
    for condition in EMPIRICAL_CONDITIONS:
        total_points = int(condition["total_points"])
        sigma = float(condition["noise_sigma_over_a"]) * a_budget
        pilot_points = max(8, int(round(ROUTER_PILOT_FRACTION * total_points)))
        final_points = max(8, total_points - pilot_points)

        for e in e_values:
            for replicate in range(replicates):
                perimeter_points = add_isotropic_noise(sample_uniform_boundary(a_budget, float(e), total_points), sigma, rng)
                width_points = add_isotropic_noise(sample_extremum_strategy(a_budget, float(e), total_points), sigma, rng)
                major_points = add_isotropic_noise(sample_major_tip_strategy(a_budget, float(e), total_points), sigma, rng)
                pilot_points_cloud = add_isotropic_noise(sample_uniform_boundary(a_budget, float(e), pilot_points), sigma, rng)

                perimeter_e = float(e_from_perimeter(np.array([estimate_perimeter_probe(perimeter_points)]))[0])
                width_e = float(e_from_width(np.array([estimate_width_probe(width_points)]))[0])
                major_e = float(e_from_major_tip(np.array([estimate_major_tip_probe(major_points)]))[0])
                pilot_e = float(e_from_perimeter(np.array([estimate_perimeter_probe(pilot_points_cloud)]))[0])

                rows.append(
                    EmpiricalTrial(
                        condition=condition["name"],
                        true_e=float(e),
                        replicate=replicate,
                        pilot_perimeter_estimate=pilot_e,
                        perimeter_estimate=perimeter_e,
                        width_estimate=width_e,
                        major_tip_estimate=major_e,
                        pilot_abs_error=float(abs(pilot_e - e)),
                        perimeter_abs_error=float(abs(perimeter_e - e)),
                        width_abs_error=float(abs(width_e - e)),
                        major_tip_abs_error=float(abs(major_e - e)),
                    )
                )
    return rows

def aggregate_empirical_mae(rows: list[EmpiricalTrial], e_values: np.ndarray) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for condition in EMPIRICAL_CONDITIONS:
        name = condition["name"]
        for e in e_values:
            subset = [row for row in rows if row.condition == name and abs(row.true_e - float(e)) < 1e-12]
            for probe in PROBE_ORDER:
                summary.append(
                    {
                        "condition": name,
                        "true_e": float(e),
                        "probe": probe,
                        "mae": float(np.mean([getattr(row, f"{probe}_abs_error") for row in subset])),
                    }
                )
    return summary

def threshold_candidates(e_values: np.ndarray) -> list[float]:
    mids = [0.0]
    mids.extend(float(0.5 * (left + right)) for left, right in zip(e_values[:-1], e_values[1:]))
    mids.append(1.0)
    return mids

def route_probe_with_order(value: float, tau1: float, tau2: float, order: tuple[str, str, str]) -> str:
    if value < tau1:
        return order[0]
    if value < tau2:
        return order[1]
    return order[2]

def search_policy_from_value(
    train_rows: list[EmpiricalTrial],
    e_values: np.ndarray,
    value_getter,
) -> tuple[float, float, tuple[str, str, str]]:
    candidates = threshold_candidates(e_values)
    orders = list(itertools.permutations(PROBE_ORDER))
    best_pair = (0.35, 0.75)
    best_order = orders[0]
    best_score = float("inf")
    for order in orders:
        for tau1 in candidates:
            for tau2 in candidates:
                if tau2 < tau1:
                    continue
                errors = []
                for row in train_rows:
                    probe = route_probe_with_order(value_getter(row), tau1, tau2, order)
                    errors.append(getattr(row, f"{probe}_abs_error"))
                score = float(np.mean(errors))
                if score < best_score:
                    best_score = score
                    best_pair = (float(tau1), float(tau2))
                    best_order = order
    return best_pair[0], best_pair[1], best_order

def evaluate_router(
    rows: list[EmpiricalTrial],
    train_e_values: np.ndarray,
    test_e_values: np.ndarray,
) -> tuple[list[RouterResult], list[dict[str, float | str]]]:
    results: list[RouterResult] = []
    threshold_rows: list[dict[str, float | str]] = []
    train_set = set(float(value) for value in train_e_values)
    test_set = set(float(value) for value in test_e_values)

    for condition in EMPIRICAL_CONDITIONS:
        name = condition["name"]
        condition_rows = [row for row in rows if row.condition == name]
        train_rows = [row for row in condition_rows if row.true_e in train_set]
        test_rows = [row for row in condition_rows if row.true_e in test_set]
        tau1, tau2, router_order = search_policy_from_value(train_rows, train_e_values, lambda row: row.pilot_perimeter_estimate)
        oracle_tau1, oracle_tau2, oracle_order = search_policy_from_value(train_rows, train_e_values, lambda row: row.true_e)
        threshold_rows.append(
            {
                "condition": name,
                "tau1": tau1,
                "tau2": tau2,
                "router_order": " -> ".join(router_order),
                "oracle_tau1": oracle_tau1,
                "oracle_tau2": oracle_tau2,
                "oracle_order": " -> ".join(oracle_order),
            }
        )

        methods = {
            "perimeter_only": lambda row: row.perimeter_abs_error,
            "width_only": lambda row: row.width_abs_error,
            "major_tip_only": lambda row: row.major_tip_abs_error,
            "router": lambda row: getattr(row, f"{route_probe_with_order(row.pilot_perimeter_estimate, tau1, tau2, router_order)}_abs_error"),
            "phase_oracle": lambda row: getattr(row, f"{route_probe_with_order(row.true_e, oracle_tau1, oracle_tau2, oracle_order)}_abs_error"),
            "trial_oracle": lambda row: min(row.perimeter_abs_error, row.width_abs_error, row.major_tip_abs_error),
        }

        for method, error_fn in methods.items():
            results.append(
                RouterResult(
                    condition=name,
                    method=method,
                    mean_abs_error=float(np.mean([error_fn(row) for row in test_rows])),
                )
            )

    return results, threshold_rows

def exact_inverse_audit(e_values: np.ndarray) -> dict[str, float]:
    width_e = e_from_width(width_residue(e_values))
    perimeter_e = e_from_perimeter(normalized_perimeter(e_values))
    major_e = e_from_major_tip(major_tip_response(e_values))
    return {
        "max_width_inverse_error": float(np.max(np.abs(width_e - e_values))),
        "max_perimeter_inverse_error": float(np.max(np.abs(perimeter_e - e_values))),
        "max_major_tip_inverse_error": float(np.max(np.abs(major_e - e_values))),
    }

def clean_strategy_audit(e_values: np.ndarray, a_budget: float) -> dict[str, float]:
    perimeter_errors = []
    width_errors = []
    major_errors = []
    for e in e_values:
        perimeter_points = sample_uniform_boundary(a_budget, float(e), 1024)
        width_points = sample_extremum_strategy(a_budget, float(e), 512)
        major_points = sample_major_tip_strategy(a_budget, float(e), 512)
        perimeter_e = float(e_from_perimeter(np.array([estimate_perimeter_probe(perimeter_points)]))[0])
        width_e = float(e_from_width(np.array([estimate_width_probe(width_points)]))[0])
        major_e = float(e_from_major_tip(np.array([estimate_major_tip_probe(major_points)]))[0])
        perimeter_errors.append(abs(perimeter_e - float(e)))
        width_errors.append(abs(width_e - float(e)))
        major_errors.append(abs(major_e - float(e)))
    return {
        "max_perimeter_abs_error": float(max(perimeter_errors)),
        "max_width_abs_error": float(max(width_errors)),
        "max_major_tip_abs_error": float(max(major_errors)),
    }

def router_logic_audit() -> dict[str, float]:
    toy = [
        EmpiricalTrial(
            condition="toy",
            true_e=0.2,
            replicate=0,
            pilot_perimeter_estimate=0.18,
            perimeter_estimate=0.19,
            width_estimate=0.30,
            major_tip_estimate=0.45,
            pilot_abs_error=0.02,
            perimeter_abs_error=0.01,
            width_abs_error=0.10,
            major_tip_abs_error=0.25,
        ),
        EmpiricalTrial(
            condition="toy",
            true_e=0.6,
            replicate=1,
            pilot_perimeter_estimate=0.62,
            perimeter_estimate=0.55,
            width_estimate=0.58,
            major_tip_estimate=0.78,
            pilot_abs_error=0.02,
            perimeter_abs_error=0.05,
            width_abs_error=0.02,
            major_tip_abs_error=0.18,
        ),
        EmpiricalTrial(
            condition="toy",
            true_e=0.9,
            replicate=2,
            pilot_perimeter_estimate=0.88,
            perimeter_estimate=0.80,
            width_estimate=0.84,
            major_tip_estimate=0.91,
            pilot_abs_error=0.02,
            perimeter_abs_error=0.10,
            width_abs_error=0.06,
            major_tip_abs_error=0.01,
        ),
    ]
    tau1, tau2, order = search_policy_from_value(toy, np.array([0.2, 0.6, 0.9]), lambda row: row.pilot_perimeter_estimate)
    routed_errors = [
        getattr(row, f"{route_probe_with_order(row.pilot_perimeter_estimate, tau1, tau2, order)}_abs_error")
        for row in toy
    ]
    return {
        "toy_tau1": float(tau1),
        "toy_tau2": float(tau2),
        "toy_order": " -> ".join(order),
        "toy_router_mean_abs_error": float(np.mean(routed_errors)),
    }

def write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def plot_ideal_curves(path: str, rows: list[IdealMetricRow]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for probe in PROBE_ORDER:
        probe_rows = [row for row in rows if row.probe == probe]
        ax.plot(
            [row.e for row in probe_rows],
            [row.mae for row in probe_rows],
            lw=2.4,
            color=PROBE_COLORS[probe],
            label=PROBE_LABELS[probe],
        )
    ax.set_yscale("log")
    ax.set_xlabel("true e")
    ax.set_ylabel("MAE of direct probe inversion")
    ax.set_title("Ideal direct-measurement benchmark")
    ax.legend(frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_empirical_curves(path: str, summary_rows: list[dict[str, float | str]], e_values: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.25)
    axes = axes.ravel()

    for ax, condition in zip(axes, EMPIRICAL_CONDITIONS):
        subset = [row for row in summary_rows if row["condition"] == condition["name"]]
        for probe in PROBE_ORDER:
            probe_rows = [row for row in subset if row["probe"] == probe]
            ax.plot(
                e_values,
                [row["mae"] for row in probe_rows],
                lw=2.2,
                color=PROBE_COLORS[probe],
                label=PROBE_LABELS[probe],
            )
        ax.set_title(condition["name"].replace("_", " "))
        ax.set_xlabel("true e")
        ax.set_ylabel("probe-specific MAE")
        ax.set_yscale("log")
        ax.legend(frameon=True)

    fig.suptitle("Equal-budget dedicated probe strategies", fontsize=15, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_best_probe_map(path: str, summary_rows: list[dict[str, float | str]], e_values: np.ndarray) -> None:
    probe_to_id = {"perimeter": 0, "width": 1, "major_tip": 2}
    matrix = []
    for condition in EMPIRICAL_CONDITIONS:
        row = []
        subset = [item for item in summary_rows if item["condition"] == condition["name"]]
        for e in e_values:
            e_subset = [item for item in subset if abs(float(item["true_e"]) - float(e)) < 1e-12]
            best = min(e_subset, key=lambda item: float(item["mae"]))["probe"]
            row.append(probe_to_id[str(best)])
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(13.0, 4.2))
    cmap = ListedColormap([PROBE_COLORS["perimeter"], PROBE_COLORS["width"], PROBE_COLORS["major_tip"]])
    ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=-0.5, vmax=2.5)
    ax.set_yticks(range(len(EMPIRICAL_CONDITIONS)))
    ax.set_yticklabels([condition["name"].replace("_", " ") for condition in EMPIRICAL_CONDITIONS])
    ax.set_xticks(range(len(e_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in e_values], rotation=45, ha="right")
    ax.set_xlabel("true e")
    ax.set_title("Best fixed probe by depletion phase")

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=PROBE_COLORS["perimeter"], label="perimeter"),
        Patch(facecolor=PROBE_COLORS["width"], label="width"),
        Patch(facecolor=PROBE_COLORS["major_tip"], label="major-tip curvature"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_router_comparison(path: str, router_results: list[RouterResult]) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    methods = ["perimeter_only", "width_only", "major_tip_only", "router", "phase_oracle", "trial_oracle"]
    x = np.arange(len(EMPIRICAL_CONDITIONS))
    width = 0.13

    for idx, method in enumerate(methods):
        values = [
            next(row.mean_abs_error for row in router_results if row.condition == condition["name"] and row.method == method)
            for condition in EMPIRICAL_CONDITIONS
        ]
        ax.bar(
            x + (idx - 2) * width,
            values,
            width=width,
            color=PROBE_COLORS.get(method.replace("_only", ""), "#555555"),
            alpha=0.92,
            label=method.replace("_", " "),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([condition["name"].replace("_", " ") for condition in EMPIRICAL_CONDITIONS], rotation=15, ha="right")
    ax.set_ylabel("test MAE")
    ax.set_title("Fixed probes vs phase-adaptive router")
    ax.legend(frameon=True, ncol=3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    rng = np.random.default_rng(20260324)
    a_budget = 1.0
    e_values = np.round(np.linspace(0.05, 0.95, 19), 2)
    train_e_values = e_values[::2]
    test_e_values = e_values[1::2]

    py_compile.compile(__file__, doraise=True)

    inverse_audit = exact_inverse_audit(e_values)
    clean_audit = clean_strategy_audit(e_values, a_budget)
    router_audit = router_logic_audit()

    ideal_rows = run_ideal_benchmark(e_values, replicates=5000, rng=rng)
    empirical_trials = run_empirical_trials(e_values, replicates=140, a_budget=a_budget, rng=rng)
    empirical_summary = aggregate_empirical_mae(empirical_trials, e_values)
    router_results, threshold_rows = evaluate_router(empirical_trials, train_e_values, test_e_values)

    write_csv(os.path.join(OUTPUT_DIR, "ideal_probe_summary.csv"), [asdict(row) for row in ideal_rows])
    write_csv(os.path.join(OUTPUT_DIR, "empirical_probe_trials.csv"), [asdict(row) for row in empirical_trials])
    write_csv(os.path.join(OUTPUT_DIR, "empirical_probe_summary.csv"), empirical_summary)
    write_csv(os.path.join(OUTPUT_DIR, "router_results.csv"), [asdict(row) for row in router_results])
    write_csv(os.path.join(OUTPUT_DIR, "router_thresholds.csv"), threshold_rows)

    summary = {
        "ideal_relative_noise": IDEAL_RELATIVE_NOISE,
        "router_pilot_fraction": ROUTER_PILOT_FRACTION,
        "audit": {
            "compile_ok": True,
            "inverse_audit": inverse_audit,
            "clean_strategy_audit": clean_audit,
            "router_logic_audit": router_audit,
        },
        "test_conditions": [condition for condition in EMPIRICAL_CONDITIONS],
        "router_results": [asdict(row) for row in router_results],
        "router_thresholds": threshold_rows,
    }
    with open(os.path.join(OUTPUT_DIR, "probe_specialization_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_ideal_curves(os.path.join(FIGURE_DIR, "probe_specialization_ideal.png"), ideal_rows)
    plot_empirical_curves(
        os.path.join(FIGURE_DIR, "probe_specialization_empirical_curves.png"),
        empirical_summary,
        e_values,
    )
    plot_best_probe_map(
        os.path.join(FIGURE_DIR, "probe_specialization_best_probe_map.png"),
        empirical_summary,
        e_values,
    )
    plot_router_comparison(
        os.path.join(FIGURE_DIR, "probe_specialization_router_comparison.png"),
        router_results,
    )

if __name__ == "__main__":
    main()
