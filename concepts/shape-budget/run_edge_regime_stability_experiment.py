"""
Experiment 7: edge-regime stability for Shape Budget.

This experiment studies how normalized observables respond as e approaches the
circular end (e -> 0) and the degenerate end (e -> 1).

It measures both:
- forward sensitivity: how sharply an observable changes with e
- inverse recoverability: how much uncertainty in e is induced by a fixed
  relative measurement error in the observable
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
from scipy.special import ellipe, ellipk


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
OUTPUT_DIR = os.path.join(BASE_DIR, "edge_regime_outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

RELATIVE_NOISE_FRACTION = 0.01


@dataclass
class EdgeMetricRow:
    observable: str
    e: float
    value: float
    derivative: float
    abs_sensitivity: float
    relative_condition_number: float
    delta_e_for_1pct_relative_noise: float


def width_residue(e: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(1.0 - e**2, 0.0))


def d_width_residue(e: np.ndarray) -> np.ndarray:
    return -e / np.sqrt(np.maximum(1.0 - e**2, 1e-300))


def normalized_perimeter(e: np.ndarray) -> np.ndarray:
    return (2.0 / math.pi) * ellipe(e**2)


def d_normalized_perimeter(e: np.ndarray) -> np.ndarray:
    out = np.zeros_like(e)
    nonzero = e > 0.0
    ez = e[nonzero]
    m = ez**2
    out[nonzero] = (2.0 / math.pi) * (ellipe(m) - ellipk(m)) / ez
    return out


def major_tip_response(e: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(1.0 - e**2, 1e-300)


def d_major_tip_response(e: np.ndarray) -> np.ndarray:
    return 2.0 * e / np.maximum((1.0 - e**2) ** 2, 1e-300)


def minor_tip_response(e: np.ndarray) -> np.ndarray:
    return width_residue(e)


def d_minor_tip_response(e: np.ndarray) -> np.ndarray:
    return d_width_residue(e)


OBSERVABLES = {
    "width_residue": {
        "func": width_residue,
        "derivative": d_width_residue,
        "color": "#2a9d8f",
        "label": "width residue b / a",
    },
    "normalized_perimeter": {
        "func": normalized_perimeter,
        "derivative": d_normalized_perimeter,
        "color": "#457b9d",
        "label": "normalized perimeter P / (2 pi a)",
    },
    "major_tip_response": {
        "func": major_tip_response,
        "derivative": d_major_tip_response,
        "color": "#d62828",
        "label": "major-tip response a kappa_major",
    },
}


def build_e_grid() -> np.ndarray:
    low = np.geomspace(1e-6, 1e-2, 140, endpoint=False)
    mid = np.linspace(1e-2, 0.99, 320, endpoint=False)
    high = 1.0 - np.geomspace(1e-6, 1e-2, 180)
    grid = np.unique(np.round(np.concatenate([low, mid, high]), 12))
    return np.sort(grid)


def relative_condition_number(e: np.ndarray, value: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    return np.abs((e / np.maximum(np.abs(value), 1e-300)) * derivative)


def delta_e_for_relative_noise(value: np.ndarray, derivative: np.ndarray, noise_fraction: float) -> np.ndarray:
    return noise_fraction * np.abs(value / np.maximum(np.abs(derivative), 1e-300))


def find_condition_crossing(observable_name: str, target: float = 1.0) -> float:
    func = OBSERVABLES[observable_name]["func"]
    deriv = OBSERVABLES[observable_name]["derivative"]
    lo = 1e-9
    hi = 0.999999
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        value = float(func(np.array([mid]))[0])
        dvalue = float(deriv(np.array([mid]))[0])
        cond = abs((mid / value) * dvalue)
        if cond < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def metric_rows(e_grid: np.ndarray) -> list[EdgeMetricRow]:
    rows: list[EdgeMetricRow] = []
    for name, spec in OBSERVABLES.items():
        value = spec["func"](e_grid)
        deriv = spec["derivative"](e_grid)
        rel_cond = relative_condition_number(e_grid, value, deriv)
        delta_e = delta_e_for_relative_noise(value, deriv, RELATIVE_NOISE_FRACTION)
        for e, q, dq, cond, de in zip(e_grid, value, deriv, rel_cond, delta_e):
            rows.append(
                EdgeMetricRow(
                    observable=name,
                    e=float(e),
                    value=float(q),
                    derivative=float(dq),
                    abs_sensitivity=float(abs(dq)),
                    relative_condition_number=float(cond),
                    delta_e_for_1pct_relative_noise=float(de),
                )
            )
    return rows


def write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_conditioning_overview(path: str, e_grid: np.ndarray) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.2, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.84, wspace=0.24)

    for name, spec in OBSERVABLES.items():
        value = spec["func"](e_grid)
        deriv = spec["derivative"](e_grid)
        rel_cond = relative_condition_number(e_grid, value, deriv)
        delta_e = delta_e_for_relative_noise(value, deriv, RELATIVE_NOISE_FRACTION)
        ax_left.plot(e_grid, rel_cond, lw=2.4, color=spec["color"], label=spec["label"])
        ax_right.plot(e_grid, delta_e, lw=2.4, color=spec["color"], label=spec["label"])

    ax_left.axhline(1.0, color="#444444", linestyle="--", lw=1.5, label="kappa = 1")
    ax_left.set_yscale("log")
    ax_left.set_title("Forward sensitivity across the full e range")
    ax_left.set_xlabel("e")
    ax_left.set_ylabel("relative condition number")
    ax_left.legend(loc="upper left", frameon=True)

    ax_right.set_yscale("log")
    ax_right.set_title("Inverse recoverability for 1 percent relative noise")
    ax_right.set_xlabel("e")
    ax_right.set_ylabel("implied delta e")
    ax_right.legend(loc="upper right", frameon=True)

    fig.suptitle("Experiment 7A: Edge-Regime Conditioning Overview", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_edge_zooms(path: str) -> None:
    e_zero = np.geomspace(1e-6, 2e-1, 220)
    delta = np.geomspace(1e-6, 2e-1, 220)
    e_one = 1.0 - delta

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0), constrained_layout=False)
    fig.subplots_adjust(top=0.9, hspace=0.28, wspace=0.24)

    for name, spec in OBSERVABLES.items():
        q_zero = spec["func"](e_zero)
        dq_zero = spec["derivative"](e_zero)
        cond_zero = relative_condition_number(e_zero, q_zero, dq_zero)
        de_zero = delta_e_for_relative_noise(q_zero, dq_zero, RELATIVE_NOISE_FRACTION)

        q_one = spec["func"](e_one)
        dq_one = spec["derivative"](e_one)
        cond_one = relative_condition_number(e_one, q_one, dq_one)
        de_one = delta_e_for_relative_noise(q_one, dq_one, RELATIVE_NOISE_FRACTION)

        axes[0, 0].plot(e_zero, cond_zero, lw=2.2, color=spec["color"], label=spec["label"])
        axes[1, 0].plot(e_zero, de_zero, lw=2.2, color=spec["color"], label=spec["label"])
        axes[0, 1].plot(delta, cond_one, lw=2.2, color=spec["color"], label=spec["label"])
        axes[1, 1].plot(delta, de_one, lw=2.2, color=spec["color"], label=spec["label"])

    # Reference slopes for the circular end.
    ref_e = np.array([1e-6, 2e-1])
    axes[0, 0].plot(ref_e, 5e-1 * ref_e**2, color="#777777", linestyle="--", lw=1.4, label="quadratic reference")
    axes[1, 0].plot(ref_e, 1e-2 / ref_e, color="#777777", linestyle="--", lw=1.4, label="1 / e reference")

    # Reference slopes for the degenerate end.
    ref_d = np.array([1e-6, 2e-1])
    axes[0, 1].plot(ref_d, 5e-3 / ref_d, color="#777777", linestyle="--", lw=1.4, label="1 / (1 - e) reference")
    axes[1, 1].plot(ref_d, 1e-2 * ref_d, color="#777777", linestyle="--", lw=1.4, label="(1 - e) reference")

    axes[0, 0].set_title("Circular edge: forward sensitivity")
    axes[0, 0].set_xlabel("e")
    axes[0, 0].set_ylabel("relative condition number")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")

    axes[1, 0].set_title("Circular edge: inverse recoverability")
    axes[1, 0].set_xlabel("e")
    axes[1, 0].set_ylabel("implied delta e")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")

    axes[0, 1].set_title("Degenerate edge: forward sensitivity")
    axes[0, 1].set_xlabel("1 - e")
    axes[0, 1].set_ylabel("relative condition number")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")

    axes[1, 1].set_title("Degenerate edge: inverse recoverability")
    axes[1, 1].set_xlabel("1 - e")
    axes[1, 1].set_ylabel("implied delta e")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")

    axes[0, 0].legend(loc="upper left", frameon=True)
    axes[0, 1].legend(loc="upper left", frameon=True)

    fig.suptitle("Experiment 7B: Edge Zooms And Asymptotic Slopes", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_reference_heatmap(path: str, reference_e: list[float]) -> None:
    observable_order = ["width_residue", "normalized_perimeter", "major_tip_response"]
    matrix = np.zeros((len(observable_order), len(reference_e)))
    labels = []

    for i, name in enumerate(observable_order):
        labels.append(OBSERVABLES[name]["label"])
        func = OBSERVABLES[name]["func"]
        deriv = OBSERVABLES[name]["derivative"]
        e_arr = np.array(reference_e)
        q = func(e_arr)
        dq = deriv(e_arr)
        delta_e = delta_e_for_relative_noise(q, dq, RELATIVE_NOISE_FRACTION)
        matrix[i, :] = np.log10(delta_e)

    fig, ax = plt.subplots(figsize=(10.8, 4.4), constrained_layout=False)
    fig.subplots_adjust(top=0.82, left=0.24, right=0.96, bottom=0.18)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis_r",
        xticklabels=[f"{e:g}" for e in reference_e],
        yticklabels=labels,
        cbar_kws={"label": "log10 implied delta e for 1 percent relative noise"},
        ax=ax,
    )
    ax.set_title("Experiment 7C: Reference-Point Recoverability Map", fontsize=15, fontweight="bold")
    ax.set_xlabel("reference e")
    ax.set_ylabel("observable")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[EdgeMetricRow], reference_e: list[float]) -> dict[str, object]:
    by_observable: dict[str, dict[str, object]] = {}
    for name in OBSERVABLES:
        obs_rows = [row for row in rows if row.observable == name]
        by_observable[name] = {
            "e_where_relative_condition_equals_1": float(find_condition_crossing(name)),
            "reference_points": {},
            "near_zero": {
                "max_relative_condition_e_le_0_1": float(max(row.relative_condition_number for row in obs_rows if row.e <= 0.1)),
                "min_delta_e_for_1pct_noise_e_le_0_1": float(min(row.delta_e_for_1pct_relative_noise for row in obs_rows if row.e <= 0.1)),
                "max_delta_e_for_1pct_noise_e_le_0_1": float(max(row.delta_e_for_1pct_relative_noise for row in obs_rows if row.e <= 0.1)),
            },
            "near_one": {
                "max_relative_condition_e_ge_0_9": float(max(row.relative_condition_number for row in obs_rows if row.e >= 0.9)),
                "min_delta_e_for_1pct_noise_e_ge_0_9": float(min(row.delta_e_for_1pct_relative_noise for row in obs_rows if row.e >= 0.9)),
                "max_delta_e_for_1pct_noise_e_ge_0_9": float(max(row.delta_e_for_1pct_relative_noise for row in obs_rows if row.e >= 0.9)),
            },
        }
        for e in reference_e:
            e_arr = np.array([e], dtype=float)
            q = float(OBSERVABLES[name]["func"](e_arr)[0])
            dq = float(OBSERVABLES[name]["derivative"](e_arr)[0])
            cond = float(relative_condition_number(e_arr, np.array([q]), np.array([dq]))[0])
            de = float(delta_e_for_relative_noise(np.array([q]), np.array([dq]), RELATIVE_NOISE_FRACTION)[0])
            by_observable[name]["reference_points"][f"{e:g}"] = {
                "value": q,
                "relative_condition_number": cond,
                "delta_e_for_1pct_relative_noise": de,
            }

    # Under current normalization the minor-tip response exactly matches width residue.
    e_probe = build_e_grid()
    minor_match = np.max(np.abs(width_residue(e_probe) - minor_tip_response(e_probe)))

    return {
        "noise_model": {
            "relative_measurement_noise_fraction": RELATIVE_NOISE_FRACTION,
        },
        "observables": by_observable,
        "redundancies": {
            "minor_tip_response_equals_width_residue_under_current_normalization_max_abs_difference": float(minor_match),
        },
    }


def main() -> None:
    e_grid = build_e_grid()
    rows = metric_rows(e_grid)
    reference_e = [0.001, 0.01, 0.1, 0.9, 0.99, 0.999]
    summary = summarize(rows, reference_e)

    write_csv(os.path.join(OUTPUT_DIR, "edge_regime_metrics.csv"), [asdict(row) for row in rows])
    with open(os.path.join(OUTPUT_DIR, "edge_regime_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_conditioning_overview(os.path.join(FIGURE_DIR, "edge_conditioning_overview.png"), e_grid)
    plot_edge_zooms(os.path.join(FIGURE_DIR, "edge_zoom_panels.png"))
    plot_reference_heatmap(os.path.join(FIGURE_DIR, "edge_reference_heatmap.png"), reference_e)

    print("Edge-regime stability experiment complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
