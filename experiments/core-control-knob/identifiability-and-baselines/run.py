"""
Experiment 2: identifiability and baseline comparison for Shape Budget.

Part A tests whether e = c / a can be recovered from noisy boundary observations
when the source positions are known.

Part B tests whether e is a better predictive summary variable for normalized
geometry than raw separation d, raw budget S, or the pair (d, S), using
low-capacity regression models under a scale-held-out split.
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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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


@dataclass
class IdentifiabilityTrial:
    condition: str
    true_e: float
    a_budget: float
    replicate: int
    noise_sigma_over_a: float
    num_points: int
    e_hat: float
    abs_error: float


@dataclass
class BaselineResult:
    target: str
    feature_set: str
    rmse: float
    mae: float
    r2: float


IDENTIFIABILITY_CONDITIONS = [
    {
        "name": "full_low_noise",
        "num_points": 200,
        "theta_start": 0.0,
        "theta_stop": 2.0 * math.pi,
        "noise_sigma_over_a": 0.005,
    },
    {
        "name": "full_medium_noise",
        "num_points": 200,
        "theta_start": 0.0,
        "theta_stop": 2.0 * math.pi,
        "noise_sigma_over_a": 0.02,
    },
    {
        "name": "partial_arc_medium_noise",
        "num_points": 80,
        "theta_start": 0.0,
        "theta_stop": 0.5 * math.pi,
        "noise_sigma_over_a": 0.02,
    },
    {
        "name": "sparse_full_medium_noise",
        "num_points": 16,
        "theta_start": 0.0,
        "theta_stop": 2.0 * math.pi,
        "noise_sigma_over_a": 0.02,
    },
    {
        "name": "sparse_partial_high_noise",
        "num_points": 12,
        "theta_start": 0.0,
        "theta_stop": 0.5 * math.pi,
        "noise_sigma_over_a": 0.03,
    },
]


def sample_ellipse_points(a_budget: float, e: float, num_points: int, theta_start: float, theta_stop: float) -> np.ndarray:
    b = a_budget * math.sqrt(1.0 - e * e)
    theta = np.linspace(theta_start, theta_stop, num_points, endpoint=False)
    return np.column_stack([a_budget * np.cos(theta), b * np.sin(theta)])


def add_isotropic_noise(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return points + rng.normal(scale=sigma, size=points.shape)


def estimate_a_from_known_foci(points: np.ndarray, c: float) -> float:
    focus_left = np.array([-c, 0.0])
    focus_right = np.array([c, 0.0])
    constant_sum_samples = np.linalg.norm(points - focus_left, axis=1) + np.linalg.norm(points - focus_right, axis=1)
    return 0.5 * float(np.median(constant_sum_samples))


def estimate_e_from_known_foci(points: np.ndarray, c: float) -> float:
    a_hat = estimate_a_from_known_foci(points, c)
    return c / a_hat


def run_identifiability_trials(e_values: np.ndarray, a_values: list[float], replicates: int, rng: np.random.Generator) -> list[IdentifiabilityTrial]:
    rows: list[IdentifiabilityTrial] = []
    for condition in IDENTIFIABILITY_CONDITIONS:
        for e in e_values:
            for a_budget in a_values:
                c = e * a_budget
                clean = sample_ellipse_points(
                    a_budget,
                    float(e),
                    condition["num_points"],
                    condition["theta_start"],
                    condition["theta_stop"],
                )
                sigma = condition["noise_sigma_over_a"] * a_budget
                for replicate in range(replicates):
                    noisy = add_isotropic_noise(clean, sigma, rng)
                    e_hat = estimate_e_from_known_foci(noisy, c)
                    rows.append(
                        IdentifiabilityTrial(
                            condition=condition["name"],
                            true_e=float(e),
                            a_budget=float(a_budget),
                            replicate=replicate,
                            noise_sigma_over_a=float(condition["noise_sigma_over_a"]),
                            num_points=int(condition["num_points"]),
                            e_hat=float(e_hat),
                            abs_error=float(abs(e_hat - e)),
                        )
                    )
    return rows


def aggregate_identifiability(rows: list[IdentifiabilityTrial], e_values: np.ndarray) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for condition in IDENTIFIABILITY_CONDITIONS:
        name = condition["name"]
        for e in e_values:
            errors = [row.abs_error for row in rows if row.condition == name and abs(row.true_e - float(e)) < 1e-12]
            e_hats = [row.e_hat for row in rows if row.condition == name and abs(row.true_e - float(e)) < 1e-12]
            summary.append(
                {
                    "condition": name,
                    "true_e": float(e),
                    "mae": float(np.mean(errors)),
                    "median_abs_error": float(np.median(errors)),
                    "p95_abs_error": float(np.quantile(errors, 0.95)),
                    "mean_e_hat": float(np.mean(e_hats)),
                    "std_e_hat": float(np.std(e_hats)),
                }
            )
    return summary


def normalized_perimeter_ratio(e: np.ndarray) -> np.ndarray:
    # Exact normalized ellipse perimeter: P / (2 pi a) = (2 / pi) E(e),
    # where E is the complete elliptic integral of the second kind.
    return (2.0 / math.pi) * ellipe(e**2)


def poly_features_1d(x: np.ndarray, degree: int = 5) -> np.ndarray:
    cols = [np.ones_like(x)]
    for power in range(1, degree + 1):
        cols.append(x**power)
    return np.column_stack(cols)


def poly_features_2d(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    cols = [np.ones_like(x)]
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i == 0 and j == 0:
                continue
            cols.append((x**i) * (y**j))
    return np.column_stack(cols)


def fit_and_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> tuple[float, float, float]:
    beta = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
    pred = test_x @ beta
    residual = pred - test_y
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((test_y - np.mean(test_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return rmse, mae, r2


def run_baseline_comparison(rng: np.random.Generator) -> list[BaselineResult]:
    sample_count = 5000
    e = rng.uniform(0.05, 0.95, sample_count)
    a_budget = rng.uniform(0.5, 5.0, sample_count)
    d = 2.0 * a_budget * e
    S = 2.0 * a_budget

    targets = {
        "width_residue": np.sqrt(1.0 - e**2),
        "normalized_perimeter": normalized_perimeter_ratio(e),
        "major_tip_response": 1.0 / (1.0 - e**2),
        "minor_tip_response": np.sqrt(1.0 - e**2),
    }

    train_mask = a_budget <= 2.5
    test_mask = a_budget > 2.5

    feature_sets = {
        "e_only": (
            poly_features_1d(e[train_mask], degree=5),
            poly_features_1d(e[test_mask], degree=5),
        ),
        "d_only": (
            poly_features_1d(d[train_mask], degree=5),
            poly_features_1d(d[test_mask], degree=5),
        ),
        "S_only": (
            poly_features_1d(S[train_mask], degree=5),
            poly_features_1d(S[test_mask], degree=5),
        ),
        "d_and_S": (
            poly_features_2d(d[train_mask], S[train_mask], degree=3),
            poly_features_2d(d[test_mask], S[test_mask], degree=3),
        ),
    }

    results: list[BaselineResult] = []
    for target_name, target in targets.items():
        train_y = target[train_mask]
        test_y = target[test_mask]
        for feature_name, (train_x, test_x) in feature_sets.items():
            rmse, mae, r2 = fit_and_score(train_x, train_y, test_x, test_y)
            results.append(BaselineResult(target_name, feature_name, rmse, mae, r2))
    return results


def write_csv(path: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_identifiability_heatmap(path: str, summary_rows: list[dict[str, float]], e_values: np.ndarray) -> None:
    matrix = np.zeros((len(IDENTIFIABILITY_CONDITIONS), len(e_values)))
    labels = []
    for i, condition in enumerate(IDENTIFIABILITY_CONDITIONS):
        labels.append(condition["name"])
        for j, e in enumerate(e_values):
            match = next(row for row in summary_rows if row["condition"] == condition["name"] and abs(row["true_e"] - float(e)) < 1e-12)
            matrix[i, j] = match["mae"]

    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, left=0.19, right=0.98, bottom=0.14)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        xticklabels=[f"{e:.2f}" for e in e_values],
        yticklabels=labels,
        cbar_kws={"label": "mean absolute error in e-hat"},
    )
    ax.set_title("Experiment 2A: Identifiability Heatmap", fontsize=15, fontweight="bold")
    ax.set_xlabel("true e")
    ax.set_ylabel("observation condition")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_identifiability_recovery(path: str, summary_rows: list[dict[str, float]], e_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.6), constrained_layout=False)
    fig.subplots_adjust(top=0.88, left=0.12, right=0.98, bottom=0.11)

    chosen = ["full_low_noise", "partial_arc_medium_noise", "sparse_partial_high_noise"]
    colors = ["#2a9d8f", "#457b9d", "#d62828"]
    for name, color in zip(chosen, colors):
        mean_vals = []
        err_vals = []
        for e in e_values:
            row = next(item for item in summary_rows if item["condition"] == name and abs(item["true_e"] - float(e)) < 1e-12)
            mean_vals.append(row["mean_e_hat"])
            err_vals.append(row["p95_abs_error"])
        ax.plot(e_values, mean_vals, color=color, lw=2.7, label=name)
        ax.fill_between(e_values, np.array(mean_vals) - np.array(err_vals), np.array(mean_vals) + np.array(err_vals), color=color, alpha=0.14)

    ax.plot(e_values, e_values, color="#222222", linestyle="--", lw=1.5, label="ideal recovery")
    ax.set_title("Experiment 2B: Recovery of e from noisy observations", fontsize=15, fontweight="bold")
    ax.set_xlabel("true e")
    ax.set_ylabel("recovered e-hat")
    ax.set_xlim(float(e_values[0]), float(e_values[-1]))
    ax.set_ylim(float(e_values[0]) - 0.03, float(e_values[-1]) + 0.03)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_collapse(path: str, rng: np.random.Generator) -> None:
    sample_count = 1800
    e = rng.uniform(0.05, 0.95, sample_count)
    a_budget = rng.uniform(0.5, 5.0, sample_count)
    d = 2.0 * a_budget * e
    S = 2.0 * a_budget
    width = np.sqrt(1.0 - e**2)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(top=0.84, wspace=0.28)

    for ax, x, xlabel in zip(
        axes,
        [e, d, S],
        ["e = c / a", "raw separation d", "raw budget S"],
    ):
        scatter = ax.scatter(x, width, c=a_budget, cmap="viridis", s=10, alpha=0.55)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("width residue b / a")
    axes[0].set_title("Collapse under e")
    axes[1].set_title("Aliasing under d")
    axes[2].set_title("Aliasing under S")

    cbar = fig.colorbar(scatter, ax=axes, shrink=0.92)
    cbar.set_label("a")
    fig.suptitle("Experiment 2C: Why e is a stronger summary variable", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_rmse(path: str, results: list[BaselineResult]) -> None:
    feature_order = ["e_only", "d_only", "S_only", "d_and_S"]
    target_order = ["width_residue", "normalized_perimeter", "major_tip_response", "minor_tip_response"]
    colors = ["#2a9d8f", "#e76f51", "#8d99ae", "#264653"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.34, wspace=0.24)

    for ax, target in zip(axes.ravel(), target_order):
        vals = [next(row.rmse for row in results if row.target == target and row.feature_set == feature) for feature in feature_order]
        ax.bar(feature_order, vals, color=colors)
        ax.set_title(target)
        ax.set_ylabel("test RMSE (log scale)")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=18)

    fig.suptitle("Experiment 2D: Scale-held-out baseline comparison", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summarize_identifiability(summary_rows: list[dict[str, float]]) -> dict[str, float]:
    by_condition = {}
    for condition in IDENTIFIABILITY_CONDITIONS:
        rows = [row for row in summary_rows if row["condition"] == condition["name"]]
        by_condition[condition["name"]] = {
            "mean_mae": float(np.mean([row["mae"] for row in rows])),
            "worst_p95_abs_error": float(max(row["p95_abs_error"] for row in rows)),
        }
    return by_condition


def summarize_baselines(results: list[BaselineResult]) -> dict[str, dict[str, float]]:
    output = {}
    for target in sorted({row.target for row in results}):
        output[target] = {row.feature_set: row.rmse for row in results if row.target == target}
    return output


def main() -> None:
    rng = np.random.default_rng(20260324)
    e_values = np.round(np.linspace(0.10, 0.90, 17), 4)
    a_values = [0.75, 1.0, 1.5, 2.5, 4.0]
    replicates = 120

    ident_trials = run_identifiability_trials(e_values, a_values, replicates, rng)
    ident_summary = aggregate_identifiability(ident_trials, e_values)
    baseline_results = run_baseline_comparison(rng)

    ident_trials_path = os.path.join(OUTPUT_DIR, "identifiability_trials.csv")
    ident_summary_path = os.path.join(OUTPUT_DIR, "identifiability_summary.csv")
    baseline_results_path = os.path.join(OUTPUT_DIR, "baseline_rmse.csv")
    summary_json_path = os.path.join(OUTPUT_DIR, "experiment_summary.json")

    write_csv(ident_trials_path, [asdict(row) for row in ident_trials])
    write_csv(ident_summary_path, ident_summary)
    write_csv(baseline_results_path, [asdict(row) for row in baseline_results])

    summary = {
        "identifiability": summarize_identifiability(ident_summary),
        "baseline_rmse": summarize_baselines(baseline_results),
    }
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_identifiability_heatmap(os.path.join(FIGURE_DIR, "identifiability_heatmap.png"), ident_summary, e_values)
    plot_identifiability_recovery(os.path.join(FIGURE_DIR, "identifiability_recovery.png"), ident_summary, e_values)
    plot_baseline_collapse(os.path.join(FIGURE_DIR, "baseline_collapse.png"), rng)
    plot_baseline_rmse(os.path.join(FIGURE_DIR, "baseline_rmse.png"), baseline_results)

    print("Identifiability and baseline experiment complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
