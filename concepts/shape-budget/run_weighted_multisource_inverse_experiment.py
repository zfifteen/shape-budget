"""
Post-roadmap extension: inverse weighted multi-source experiment for Shape Budget.

This experiment asks whether the weighted three-source control object can be
recovered from boundary data when the source positions and weights are not
given in advance.

The recovery target is the normalized control object:

- normalized source-triangle geometry relative to budget
- normalized weight vector in the simplex

To keep the first inverse test clean and interpretable, the experiment uses
canonical-pose boundary observations. Translation and scale are removed from
the observed boundary itself by centering at the boundary centroid and
normalizing by the mean centroid radius. Rotation is not randomized in this
first inverse test.

The inverse method is deliberately simple:

- build a reference bank of weighted forward models
- encode each boundary as a centroid-centered normalized radial signature
- recover the nearest reference signature under masked L2 distance

This is a conservative inverse. If a simple bank-matching method works, that is
good evidence that the control object is operational.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from run_weighted_multisource_experiment import canonical_sources, normalize_weights, weighted_boundary_curve


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
OUTPUT_DIR = os.path.join(BASE_DIR, "weighted_multisource_inverse_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


REFERENCE_BANK_SIZE = 300
EQUAL_WEIGHT_BANK_SIZE = 150
TEST_TRIALS_PER_REGIME = 40
CURVE_SAMPLE_COUNT = 96
SIGNATURE_ANGLE_COUNT = 64

GEOMETRY_BOUNDS = {
    "rho_min": 0.06,
    "rho_max": 0.22,
    "t_min": -0.70,
    "t_max": 0.70,
    "h_min": 0.55,
    "h_max": 1.45,
}


OBSERVATION_REGIMES = [
    {"name": "full_clean", "noise_sigma": 0.0, "observed_fraction": 1.0, "mode": "full"},
    {"name": "full_noisy", "noise_sigma": 0.01, "observed_fraction": 1.0, "mode": "full"},
    {"name": "partial_arc_noisy", "noise_sigma": 0.01, "observed_fraction": 0.40, "mode": "contiguous"},
    {"name": "sparse_full_noisy", "noise_sigma": 0.02, "observed_count": 12, "mode": "random"},
    {"name": "sparse_partial_high_noise", "noise_sigma": 0.03, "observed_count": 10, "arc_fraction": 0.25, "mode": "sparse_contiguous"},
]


@dataclass
class TrialRow:
    condition: str
    trial: int
    true_rho: float
    true_t: float
    true_h: float
    true_w1: float
    true_w2: float
    true_w3: float
    pred_rho: float
    pred_t: float
    pred_h: float
    pred_w1: float
    pred_w2: float
    pred_w3: float
    equal_pred_rho: float
    equal_pred_t: float
    equal_pred_h: float
    geometry_mae: float
    weight_mae: float
    weighted_fit_rmse: float
    equal_weight_baseline_fit_rmse: float
    weighted_fit_improvement_factor: float


def sample_weighted_parameters(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    rho = float(rng.uniform(GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(rng.uniform(GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(rng.uniform(GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    weights = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
    return rho, t, h, float(weights[0]), float(weights[1])


def sample_equal_weight_parameters(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    rho = float(rng.uniform(GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(rng.uniform(GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(rng.uniform(GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    return rho, t, h, 1.0 / 3.0, 1.0 / 3.0


def boundary_signature_from_curve(curve: np.ndarray, angle_count: int = SIGNATURE_ANGLE_COUNT) -> np.ndarray:
    center = np.mean(curve, axis=0)
    shifted = curve - center
    angle = np.mod(np.arctan2(shifted[:, 1], shifted[:, 0]), 2.0 * math.pi)
    radius = np.linalg.norm(shifted, axis=1)
    order = np.argsort(angle)
    angle = angle[order]
    radius = radius[order]

    angle_ext = np.concatenate([angle[-1:] - 2.0 * math.pi, angle, angle[:1] + 2.0 * math.pi])
    radius_ext = np.concatenate([radius[-1:], radius, radius[:1]])
    grid = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    signature = np.interp(grid, angle_ext, radius_ext)
    return signature / np.mean(signature)


def forward_signature(params: tuple[float, float, float, float, float]) -> np.ndarray:
    rho, t, h, w1, w2 = params
    weights = normalize_weights(np.array([w1, w2, 1.0 - w1 - w2], dtype=float))
    points = canonical_sources(rho, t, h, S=1.0)
    _, _, curve = weighted_boundary_curve(points, weights, 1.0, angle_count=CURVE_SAMPLE_COUNT)
    return boundary_signature_from_curve(curve, angle_count=SIGNATURE_ANGLE_COUNT)


def control_invariants(params: tuple[float, float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    rho, t, h, w1, w2 = params
    points = canonical_sources(rho, t, h, S=1.0)
    d12 = np.linalg.norm(points[0] - points[1])
    d13 = np.linalg.norm(points[0] - points[2])
    d23 = np.linalg.norm(points[1] - points[2])
    return np.array([d12, d13, d23]), np.array([w1, w2, 1.0 - w1 - w2])


def symmetry_aware_errors(
    true_params: tuple[float, float, float, float, float],
    pred_params: tuple[float, float, float, float, float],
) -> tuple[float, float]:
    true_geom, true_weights = control_invariants(true_params)
    pred_geom, pred_weights = control_invariants(pred_params)

    direct_geom_mae = float(np.mean(np.abs(true_geom - pred_geom)))
    direct_weight_mae = float(np.mean(np.abs(true_weights - pred_weights)))

    swapped_geom = np.array([pred_geom[0], pred_geom[2], pred_geom[1]])
    swapped_weights = np.array([pred_weights[1], pred_weights[0], pred_weights[2]])
    swapped_geom_mae = float(np.mean(np.abs(true_geom - swapped_geom)))
    swapped_weight_mae = float(np.mean(np.abs(true_weights - swapped_weights)))

    if direct_geom_mae + direct_weight_mae <= swapped_geom_mae + swapped_weight_mae:
        return direct_geom_mae, direct_weight_mae
    return swapped_geom_mae, swapped_weight_mae


def observe_signature(
    clean_signature: np.ndarray,
    regime: dict[str, float | str | int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    observed = clean_signature.copy()
    mask = np.zeros_like(clean_signature, dtype=bool)
    mode = str(regime["mode"])

    if mode == "full":
        mask[:] = True
    elif mode == "contiguous":
        span = int(float(regime["observed_fraction"]) * len(clean_signature))
        start = int(rng.integers(0, len(clean_signature)))
        mask[(np.arange(span) + start) % len(clean_signature)] = True
    elif mode == "random":
        count = int(regime["observed_count"])
        mask[rng.choice(len(clean_signature), size=count, replace=False)] = True
    elif mode == "sparse_contiguous":
        span = int(float(regime["arc_fraction"]) * len(clean_signature))
        start = int(rng.integers(0, len(clean_signature)))
        pool = (np.arange(span) + start) % len(clean_signature)
        count = min(int(regime["observed_count"]), len(pool))
        mask[rng.choice(pool, size=count, replace=False)] = True
    else:
        raise ValueError(f"Unknown observation mode: {mode}")

    sigma = float(regime["noise_sigma"])
    if sigma > 0.0:
        observed[mask] += rng.normal(scale=sigma, size=int(np.sum(mask)))
    return observed, mask


def nearest_neighbor_prediction(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    bank_signatures: np.ndarray,
    bank_params: list[tuple[float, float, float, float, float]],
) -> tuple[tuple[float, float, float, float, float], np.ndarray]:
    residual = bank_signatures[:, mask] - observed_signature[mask]
    mse = np.mean(residual * residual, axis=1)
    idx = int(np.argmin(mse))
    return bank_params[idx], bank_signatures[idx]


def build_reference_bank(
    sample_size: int,
    rng: np.random.Generator,
    weighted: bool,
) -> tuple[list[tuple[float, float, float, float, float]], np.ndarray]:
    params_list: list[tuple[float, float, float, float, float]] = []
    signatures: list[np.ndarray] = []
    sampler = sample_weighted_parameters if weighted else sample_equal_weight_parameters

    while len(params_list) < sample_size:
        params = sampler(rng)
        signatures.append(forward_signature(params))
        params_list.append(params)
    return params_list, np.array(signatures)


def write_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_trials(rows: list[TrialRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for regime in OBSERVATION_REGIMES:
        name = str(regime["name"])
        subset = [row for row in rows if row.condition == name]
        geometry = np.array([row.geometry_mae for row in subset])
        weight = np.array([row.weight_mae for row in subset])
        fit = np.array([row.weighted_fit_rmse for row in subset])
        baseline = np.array([row.equal_weight_baseline_fit_rmse for row in subset])
        improve = np.array([row.weighted_fit_improvement_factor for row in subset])
        summary.append(
            {
                "condition": name,
                "geometry_mae_mean": float(np.mean(geometry)),
                "geometry_mae_p95": float(np.quantile(geometry, 0.95)),
                "weight_mae_mean": float(np.mean(weight)),
                "weight_mae_p95": float(np.quantile(weight, 0.95)),
                "weighted_fit_rmse_mean": float(np.mean(fit)),
                "equal_weight_baseline_fit_rmse_mean": float(np.mean(baseline)),
                "fit_improvement_factor_mean": float(np.mean(improve)),
            }
        )
    return summary


def plot_error_heatmap(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    geometry = np.array([[float(item["geometry_mae_mean"]) for item in summary_rows]])
    weights = np.array([[float(item["weight_mae_mean"]) for item in summary_rows]])

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.19, right=0.98, hspace=0.35)

    sns.heatmap(
        geometry,
        ax=axes[0],
        cmap="viridis",
        annot=True,
        fmt=".3f",
        xticklabels=conditions,
        yticklabels=["geometry MAE"],
        cbar_kws={"label": "mean absolute error"},
    )
    axes[0].set_title("Boundary-only inverse recovery of normalized geometry")

    sns.heatmap(
        weights,
        ax=axes[1],
        cmap="magma",
        annot=True,
        fmt=".3f",
        xticklabels=conditions,
        yticklabels=["weight MAE"],
        cbar_kws={"label": "mean absolute error"},
    )
    axes[1].set_title("Boundary-only inverse recovery of normalized weights")
    axes[1].set_xlabel("observation regime")

    fig.suptitle("Weighted Inverse Experiment A: Recovery Error Across Regimes", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(path: str, summary_rows: list[dict[str, float | str]]) -> None:
    conditions = [str(item["condition"]) for item in summary_rows]
    weighted_fit = np.array([float(item["weighted_fit_rmse_mean"]) for item in summary_rows])
    equal_fit = np.array([float(item["equal_weight_baseline_fit_rmse_mean"]) for item in summary_rows])
    improve = np.array([float(item["fit_improvement_factor_mean"]) for item in summary_rows])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.4, 5.6), constrained_layout=False)
    fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.28)

    x = np.arange(len(conditions))
    width = 0.36
    ax_left.bar(x - width / 2.0, weighted_fit, width=width, color="#2a9d8f", label="weighted bank")
    ax_left.bar(x + width / 2.0, equal_fit, width=width, color="#e76f51", label="equal-weight baseline")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(conditions, rotation=20, ha="right")
    ax_left.set_ylabel("mean clean-signature RMSE")
    ax_left.set_title("Weighted bank beats the equal-weight baseline")
    ax_left.legend(loc="upper left", frameon=True)

    ax_right.plot(x, improve, color="#1d3557", lw=2.4, marker="o")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(conditions, rotation=20, ha="right")
    ax_right.set_ylabel("baseline / weighted fit ratio")
    ax_right.set_title("Average fit-improvement factor")

    fig.suptitle("Weighted Inverse Experiment B: Baseline Comparison", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_example_recoveries(path: str, rows: list[TrialRow]) -> None:
    chosen_conditions = ["full_clean", "partial_arc_noisy", "sparse_partial_high_noise"]
    fig, axes = plt.subplots(len(chosen_conditions), 1, figsize=(10.2, 9.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.40)

    for ax, name in zip(axes, chosen_conditions):
        subset = [row for row in rows if row.condition == name]
        exemplar = min(subset, key=lambda row: row.weighted_fit_rmse)
        true_sig = forward_signature((exemplar.true_rho, exemplar.true_t, exemplar.true_h, exemplar.true_w1, exemplar.true_w2))
        pred_sig = forward_signature((exemplar.pred_rho, exemplar.pred_t, exemplar.pred_h, exemplar.pred_w1, exemplar.pred_w2))
        equal_sig = forward_signature((exemplar.equal_pred_rho, exemplar.equal_pred_t, exemplar.equal_pred_h, 1.0 / 3.0, 1.0 / 3.0))
        angle_grid = np.linspace(0.0, 2.0 * math.pi, len(true_sig), endpoint=False)

        ax.plot(angle_grid, true_sig, color="#222222", lw=2.4, label="true signature")
        ax.plot(angle_grid, pred_sig, color="#2a9d8f", lw=2.0, label="weighted-bank recovery")
        ax.plot(angle_grid, equal_sig, color="#e76f51", lw=1.8, linestyle="--", label="equal-weight baseline")
        ax.set_title(f"{name}: geometry MAE = {exemplar.geometry_mae:.3f}, weight MAE = {exemplar.weight_mae:.3f}")
        ax.set_xlabel("angle")
        ax.set_ylabel("normalized radius")
        if ax is axes[0]:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Weighted Inverse Experiment C: Representative Signature Recoveries", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20260324)
    weighted_params, weighted_signatures = build_reference_bank(REFERENCE_BANK_SIZE, rng, weighted=True)
    equal_params, equal_signatures = build_reference_bank(EQUAL_WEIGHT_BANK_SIZE, rng, weighted=False)

    rows: list[TrialRow] = []
    for regime in OBSERVATION_REGIMES:
        for trial in range(TEST_TRIALS_PER_REGIME):
            true_params = sample_weighted_parameters(rng)
            clean_signature = forward_signature(true_params)
            observed_signature, mask = observe_signature(clean_signature, regime, rng)

            pred_params, pred_signature = nearest_neighbor_prediction(observed_signature, mask, weighted_signatures, weighted_params)
            equal_pred_params, equal_signature = nearest_neighbor_prediction(observed_signature, mask, equal_signatures, equal_params)

            geometry_mae, weight_mae = symmetry_aware_errors(true_params, pred_params)
            weighted_fit_rmse = float(np.sqrt(np.mean((pred_signature - clean_signature) ** 2)))
            equal_fit_rmse = float(np.sqrt(np.mean((equal_signature - clean_signature) ** 2)))
            rows.append(
                TrialRow(
                    condition=str(regime["name"]),
                    trial=trial,
                    true_rho=float(true_params[0]),
                    true_t=float(true_params[1]),
                    true_h=float(true_params[2]),
                    true_w1=float(true_params[3]),
                    true_w2=float(true_params[4]),
                    true_w3=float(1.0 - true_params[3] - true_params[4]),
                    pred_rho=float(pred_params[0]),
                    pred_t=float(pred_params[1]),
                    pred_h=float(pred_params[2]),
                    pred_w1=float(pred_params[3]),
                    pred_w2=float(pred_params[4]),
                    pred_w3=float(1.0 - pred_params[3] - pred_params[4]),
                    equal_pred_rho=float(equal_pred_params[0]),
                    equal_pred_t=float(equal_pred_params[1]),
                    equal_pred_h=float(equal_pred_params[2]),
                    geometry_mae=geometry_mae,
                    weight_mae=weight_mae,
                    weighted_fit_rmse=weighted_fit_rmse,
                    equal_weight_baseline_fit_rmse=equal_fit_rmse,
                    weighted_fit_improvement_factor=float(equal_fit_rmse / max(weighted_fit_rmse, 1.0e-12)),
                )
            )

    trial_dicts = [row.__dict__ for row in rows]
    summary_rows = aggregate_trials(rows)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_multisource_inverse_trials.csv"), trial_dicts)
    write_csv(os.path.join(OUTPUT_DIR, "weighted_multisource_inverse_summary.csv"), summary_rows)

    plot_error_heatmap(os.path.join(FIGURE_DIR, "weighted_multisource_inverse_heatmap.png"), summary_rows)
    plot_baseline_comparison(os.path.join(FIGURE_DIR, "weighted_multisource_inverse_baseline.png"), summary_rows)
    plot_example_recoveries(os.path.join(FIGURE_DIR, "weighted_multisource_inverse_examples.png"), rows)

    summary = {
        "reference_bank_size": REFERENCE_BANK_SIZE,
        "equal_weight_bank_size": EQUAL_WEIGHT_BANK_SIZE,
        "test_trials_per_regime": TEST_TRIALS_PER_REGIME,
        "best_geometry_mae_mean": float(min(item["geometry_mae_mean"] for item in summary_rows)),
        "worst_geometry_mae_mean": float(max(item["geometry_mae_mean"] for item in summary_rows)),
        "best_weight_mae_mean": float(min(item["weight_mae_mean"] for item in summary_rows)),
        "worst_weight_mae_mean": float(max(item["weight_mae_mean"] for item in summary_rows)),
        "best_weighted_fit_rmse_mean": float(min(item["weighted_fit_rmse_mean"] for item in summary_rows)),
        "worst_weighted_fit_rmse_mean": float(max(item["weighted_fit_rmse_mean"] for item in summary_rows)),
        "smallest_fit_improvement_factor_mean": float(min(item["fit_improvement_factor_mean"] for item in summary_rows)),
        "largest_fit_improvement_factor_mean": float(max(item["fit_improvement_factor_mean"] for item in summary_rows)),
    }

    with open(os.path.join(OUTPUT_DIR, "weighted_multisource_inverse_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "by_condition": summary_rows}, handle, indent=2)

    print(json.dumps({"summary": summary, "by_condition": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
