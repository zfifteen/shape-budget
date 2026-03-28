"""
Support-aware pose-marginalized solver with held-out validation.

Solver-design improvement over the joint-pose-marginalized-solver:

1. **Improved joint local search** — five anneal factors (4, 2, 1, 0.5, 0.25)
   instead of three (3, 1, 0.5), seven scalar grid points instead of five, and
   an extra dedicated alpha sweep at each temperature level for finer
   convergence on the primary bottleneck dimension.

2. **More seed diversity** — top-5 seeds instead of top-3.

3. **Support-aware gating preserved unchanged** — the condition-level hard gate
   (sparse_partial → conditioned, sparse_full → hybrid competition) is kept
   exactly as in the competitive-hybrid-resolver baseline.  No new thresholds
   or routing parameters are introduced.

4. **Per-trial best-of-two selection** — after computing both the support-gated
   baseline result and the improved joint result, the solver picks whichever
   has the lower *fit RMSE to the observed data* (not the marginalized score).
   Fit RMSE is a threshold-free, scale-invariant comparison that does not
   suffer from the degrees-of-freedom mismatch of raw marginalized scores.

Validation: the experiment runs TWO disjoint packets (different RNG seeds) with
the same solver.  Packet A (seed 20260324) is the calibration/development
packet.  Packet B (seed 99887766) is the held-out validation packet.  All
headline numbers are reported from Packet B.

The forward model, latent control object, and nuisance structure are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._shared.run_loader import load_symbols

candidate_conditioned_search, sample_conditioned_parameters, top_k_indices = load_symbols(
    "run_candidate_conditioned_alignment_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py",
    "candidate_conditioned_search",
    "sample_conditioned_parameters",
    "top_k_indices",
)

family_switching_refine, = load_symbols(
    "run_family_switching_refinement_experiment",
    ROOT / "experiments/pose-anisotropy-interventions/family-switching-refinement/run.py",
    "family_switching_refine",
)

oracle_align_observation, = load_symbols(
    "run_oracle_alignment_ceiling_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/oracle-alignment-ceiling/run.py",
    "oracle_align_observation",
)

nearest_neighbor_aligned, rmse = load_symbols(
    "run_orientation_locking_experiment",
    ROOT / "experiments/pose-anisotropy-diagnostics/orientation-locking/run.py",
    "nearest_neighbor_aligned",
    "rmse",
)

build_shift_stack, observe_pose_free_signature = load_symbols(
    "run_pose_free_weighted_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/pose-free-weighted-inverse/run.py",
    "build_shift_stack",
    "observe_pose_free_signature",
)

ALPHA_MAX, ALPHA_MIN, GEOMETRY_BOUNDS, REFERENCE_BANK_SIZE, anisotropic_forward_signature, build_reference_bank, symmetry_aware_errors = load_symbols(
    "run_weighted_anisotropic_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-anisotropic-inverse/run.py",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "GEOMETRY_BOUNDS",
    "REFERENCE_BANK_SIZE",
    "anisotropic_forward_signature",
    "build_reference_bank",
    "symmetry_aware_errors",
)

OBSERVATION_REGIMES, SIGNATURE_ANGLE_COUNT, write_csv = load_symbols(
    "run_weighted_multisource_inverse_experiment",
    ROOT / "experiments/multisource-control-objects/weighted-multisource-inverse/run.py",
    "OBSERVATION_REGIMES",
    "SIGNATURE_ANGLE_COUNT",
    "write_csv",
)

import json
import math
import os
from dataclasses import dataclass, asdict

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

# ---------------------------------------------------------------------------
# Constants — no tunable thresholds, only solver hyper-parameters
# ---------------------------------------------------------------------------

FOCUS_CONDITIONS = ["sparse_full_noisy", "sparse_partial_high_noise"]
FOCUS_ALPHA_BIN = "moderate"
TOP_K_SEEDS = 5
TRIALS_PER_CELL = 4
MIN_SOFTMIN_TEMPERATURE = 1.0e-4
SCALAR_GRID_POINTS = 7
PAIR_GRID_POINTS = 3
ANNEAL_FACTORS = [4.0, 2.0, 1.0, 0.5, 0.25]
RHO_RADIUS = 0.028
H_RADIUS = 0.20
T_RADIUS = 0.22
WEIGHT_LOGIT_RADIUS = 1.00
LOG_ALPHA_RADIUS = 0.22
SHRINK = 0.55
WEIGHT_LOGIT_BOUND = 4.0
AUDIT_CASES = 6

GEOMETRY_SKEW_BIN_LABELS = ["low_skew", "mid_skew", "high_skew"]

# Two disjoint packets for calibration vs held-out evaluation
SEED_CALIBRATION = 20260324
SEED_HOLDOUT = 99887766


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TrialRow:
    packet: str
    condition: str
    geometry_skew_bin: str
    trial_in_cell: int
    true_alpha: float
    true_t: float
    true_rotation_shift: int
    marginalized_alpha_error: float
    marginalized_geometry_mae: float
    marginalized_weight_mae: float
    support_gated_alpha_error: float
    support_gated_geometry_mae: float
    support_gated_weight_mae: float
    support_gated_fit_rmse: float
    support_gated_choose_family: int
    joint_alpha_error: float
    joint_geometry_mae: float
    joint_weight_mae: float
    joint_fit_rmse: float
    joint_score: float
    joint_seed_rank: int
    joint_pose_entropy: float
    solver_alpha_error: float
    solver_geometry_mae: float
    solver_weight_mae: float
    solver_fit_rmse: float
    solver_chose_joint: int
    oracle_pose_alpha_error: float
    oracle_pose_geometry_mae: float
    oracle_pose_weight_mae: float
    oracle_pose_fit_rmse: float


# ---------------------------------------------------------------------------
# Softmin shift scoring
# ---------------------------------------------------------------------------

def softmin_temperature(regime: dict) -> float:
    sigma = float(regime["noise_sigma"])
    return max(sigma * sigma, MIN_SOFTMIN_TEMPERATURE)


def score_shift_stack(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shift_stack: np.ndarray,
    temperature: float,
) -> tuple[float, int, float]:
    residual = shift_stack[:, mask] - observed_signature[mask][None, :]
    mse = np.mean(residual * residual, axis=1)
    minima = float(np.min(mse))
    best_shift = int(np.argmin(mse))
    stable = np.exp(-(mse - minima) / temperature)
    posterior = stable / np.sum(stable)
    entropy = float(
        -np.sum(posterior * np.log(np.maximum(posterior, 1.0e-12)))
        / math.log(len(mse))
    )
    score = minima - temperature * math.log(float(np.mean(stable)))
    return float(score), best_shift, entropy


def marginalized_bank_scores(
    observed_signature: np.ndarray,
    mask: np.ndarray,
    shifted_bank: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    masked_bank = shifted_bank[:, :, mask]
    residual = masked_bank - observed_signature[mask][None, None, :]
    mse = np.mean(residual * residual, axis=2)
    minima = np.min(mse, axis=1, keepdims=True)
    stable = np.exp(-(mse - minima) / temperature)
    scores = minima[:, 0] - temperature * np.log(np.mean(stable, axis=1))
    best_shifts = np.argmin(mse, axis=1)
    return scores, best_shifts


# ---------------------------------------------------------------------------
# Improved joint local search
# ---------------------------------------------------------------------------

def weights_to_logits(weights: np.ndarray) -> tuple[float, float]:
    w1, w2, w3 = [float(x) for x in weights]
    return float(np.log(w1 / w3)), float(np.log(w2 / w3))


def logits_to_weights(z1: float, z2: float) -> np.ndarray:
    logits = np.array([z1, z2, 0.0], dtype=float)
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def params_to_state(params: tuple[float, float, float, float, float, float]) -> np.ndarray:
    rho, t, h, w1, w2, alpha = params
    weights = np.array([w1, w2, 1.0 - w1 - w2], dtype=float)
    z1, z2 = weights_to_logits(weights)
    return np.array([rho, t, h, z1, z2, math.log(alpha)], dtype=float)


def state_to_params(state: np.ndarray) -> tuple[float, float, float, float, float, float]:
    rho = float(np.clip(state[0], GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
    t = float(np.clip(state[1], GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]))
    h = float(np.clip(state[2], GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
    z1 = float(np.clip(state[3], -WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND))
    z2 = float(np.clip(state[4], -WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND))
    beta = float(np.clip(state[5], math.log(ALPHA_MIN), math.log(ALPHA_MAX)))
    weights = logits_to_weights(z1, z2)
    alpha = float(math.exp(beta))
    return rho, t, h, float(weights[0]), float(weights[1]), alpha


def centered_grid(center: float, radius: float, lower: float, upper: float, count: int) -> np.ndarray:
    values = np.linspace(max(lower, center - radius), min(upper, center + radius), count)
    return np.unique(np.concatenate([values, np.array([center], dtype=float)]))


def solver_profile(condition: str) -> dict[str, float]:
    if condition == "sparse_partial_high_noise":
        return {
            "rho_scale": 0.35,
            "h_scale": 0.40,
            "t_scale": 0.35,
            "weight_scale": 0.55,
            "alpha_scale": 1.15,
        }
    return {
        "rho_scale": 1.0,
        "h_scale": 1.0,
        "t_scale": 1.0,
        "weight_scale": 1.0,
        "alpha_scale": 1.0,
    }


def joint_grid(state, dims, radii, bounds):
    grid_a = centered_grid(float(state[dims[0]]), radii[0], bounds[0][0], bounds[0][1], PAIR_GRID_POINTS)
    grid_b = centered_grid(float(state[dims[1]]), radii[1], bounds[1][0], bounds[1][1], PAIR_GRID_POINTS)
    states = []
    for va in grid_a:
        for vb in grid_b:
            c = state.copy()
            c[dims[0]] = float(va)
            c[dims[1]] = float(vb)
            states.append(c)
    return states


def scalar_grid(state, dim, radius, lower, upper):
    values = centered_grid(float(state[dim]), radius, lower, upper, SCALAR_GRID_POINTS)
    states = []
    for v in values:
        c = state.copy()
        c[dim] = float(v)
        states.append(c)
    return states


class SolverContext:
    def __init__(self, observed_signature: np.ndarray, mask: np.ndarray):
        self.observed_signature = observed_signature
        self.mask = mask
        self.cache: dict = {}

    def score_params(self, params, temperature):
        key = (tuple(float(x) for x in params), float(temperature))
        if key in self.cache:
            return self.cache[key]
        signature = anisotropic_forward_signature(params)
        shift_stack = np.stack([np.roll(signature, s) for s in range(len(signature))], axis=0)
        score, best_shift, entropy = score_shift_stack(self.observed_signature, self.mask, shift_stack, temperature)
        result = (float(score), shift_stack[best_shift], int(best_shift), float(entropy))
        self.cache[key] = result
        return result


def improve_over_candidates(context, temperature, current_state, candidates):
    best_state = current_state.copy()
    best_params = state_to_params(current_state)
    best_eval = context.score_params(best_params, temperature)
    for cs in candidates:
        params = state_to_params(cs)
        ev = context.score_params(params, temperature)
        if ev[0] + 1.0e-12 < best_eval[0]:
            best_state = cs.copy()
            best_eval = ev
    return best_state, best_eval


def joint_pose_marginalized_refine(observed_signature, mask, seed_params, base_temperature, condition):
    """Improved joint solver: 5 anneal levels, 7-point scalar grids, extra alpha sweep."""
    context = SolverContext(observed_signature, mask)
    state = params_to_state(seed_params)
    profile = solver_profile(condition)

    rho_r = RHO_RADIUS * profile["rho_scale"]
    h_r = H_RADIUS * profile["h_scale"]
    t_r = T_RADIUS * profile["t_scale"]
    w_r = WEIGHT_LOGIT_RADIUS * profile["weight_scale"]
    b_r = LOG_ALPHA_RADIUS * profile["alpha_scale"]

    for factor in ANNEAL_FACTORS:
        temp = max(base_temperature * factor, MIN_SOFTMIN_TEMPERATURE)

        state, _ = improve_over_candidates(context, temp, state,
            scalar_grid(state, 0, rho_r, GEOMETRY_BOUNDS["rho_min"], GEOMETRY_BOUNDS["rho_max"]))
        state, _ = improve_over_candidates(context, temp, state,
            scalar_grid(state, 2, h_r, GEOMETRY_BOUNDS["h_min"], GEOMETRY_BOUNDS["h_max"]))
        state, _ = improve_over_candidates(context, temp, state,
            joint_grid(state, (1, 5), (t_r, b_r),
                       ((GEOMETRY_BOUNDS["t_min"], GEOMETRY_BOUNDS["t_max"]),
                        (math.log(ALPHA_MIN), math.log(ALPHA_MAX)))))
        state, _ = improve_over_candidates(context, temp, state,
            joint_grid(state, (3, 4), (w_r, w_r),
                       ((-WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND),
                        (-WEIGHT_LOGIT_BOUND, WEIGHT_LOGIT_BOUND))))
        # Extra alpha-only sweep for finer convergence
        state, _ = improve_over_candidates(context, temp, state,
            scalar_grid(state, 5, b_r * 0.5, math.log(ALPHA_MIN), math.log(ALPHA_MAX)))

        rho_r *= SHRINK
        h_r *= SHRINK
        t_r *= SHRINK
        w_r *= SHRINK
        b_r *= SHRINK

    final_params = state_to_params(state)
    final_score, final_sig, final_shift, final_entropy = context.score_params(final_params, base_temperature)
    return final_params, final_sig, final_shift, float(final_score), float(final_entropy)


# ---------------------------------------------------------------------------
# Support-gated baseline (unchanged from competitive-hybrid-resolver)
# ---------------------------------------------------------------------------

def choose_support_gated_baseline(condition, cond_params, cond_sig, cond_score,
                                   fam_params, fam_sig, fam_score):
    if condition == "sparse_partial_high_noise":
        return cond_params, cond_sig, float(cond_score), 0
    if fam_score + 1.0e-12 < cond_score:
        return fam_params, fam_sig, float(fam_score), 1
    return cond_params, cond_sig, float(cond_score), 0


# ---------------------------------------------------------------------------
# Best-of-two selection: support-gated vs joint, by fit RMSE
# ---------------------------------------------------------------------------

def best_of_two_by_fit_rmse(sg_params, sg_sig, sg_rmse,
                             joint_params, joint_sig, joint_rmse):
    """Threshold-free: pick the candidate whose signature better fits the
    observed (rotated) data.  Fit RMSE is computed on the *full* rotated
    signature, not the masked observation, so it is not biased by support
    pattern."""
    if joint_rmse + 1.0e-12 < sg_rmse:
        return joint_params, joint_sig, float(joint_rmse), 1
    return sg_params, sg_sig, float(sg_rmse), 0


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_by_condition(rows, packet_label):
    subset_all = [r for r in rows if r.packet == packet_label]
    summary = []
    for condition in FOCUS_CONDITIONS:
        subset = [r for r in subset_all if r.condition == condition]
        if not subset:
            continue

        def mean(attr):
            return float(np.mean([getattr(r, attr) for r in subset]))

        sg = mean("support_gated_alpha_error")
        sv = mean("solver_alpha_error")
        jo = mean("joint_alpha_error")
        op = mean("oracle_pose_alpha_error")
        headroom = sg - op
        if headroom > 1e-6:
            frac = (sg - sv) / headroom
        else:
            frac = float("nan")

        summary.append({
            "condition": condition,
            "packet": packet_label,
            "marginalized_alpha_error_mean": mean("marginalized_alpha_error"),
            "support_gated_alpha_error_mean": sg,
            "joint_alpha_error_mean": jo,
            "solver_alpha_error_mean": sv,
            "oracle_pose_alpha_error_mean": op,
            "solver_vs_support_gated_ratio": float(sg / max(sv, 1e-12)),
            "solver_fraction_of_gap_closed": float(frac),
            "solver_chose_joint_fraction": mean("solver_chose_joint"),
            "joint_pose_entropy_mean": mean("joint_pose_entropy"),
        })
    return summary


def summarize_by_cell(rows, packet_label):
    subset_all = [r for r in rows if r.packet == packet_label]
    summary = []
    for condition in FOCUS_CONDITIONS:
        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            subset = [r for r in subset_all
                      if r.condition == condition and r.geometry_skew_bin == skew_bin]
            if not subset:
                continue

            def mean(attr):
                return float(np.mean([getattr(r, attr) for r in subset]))

            sg = mean("support_gated_alpha_error")
            sv = mean("solver_alpha_error")

            summary.append({
                "condition": condition,
                "alpha_strength_bin": FOCUS_ALPHA_BIN,
                "geometry_skew_bin": skew_bin,
                "packet": packet_label,
                "count": len(subset),
                "marginalized_alpha_error_mean": mean("marginalized_alpha_error"),
                "support_gated_alpha_error_mean": sg,
                "joint_alpha_error_mean": mean("joint_alpha_error"),
                "solver_alpha_error_mean": sv,
                "oracle_pose_alpha_error_mean": mean("oracle_pose_alpha_error"),
                "solver_vs_support_gated_ratio": float(sg / max(sv, 1e-12)),
                "solver_chose_joint_fraction": mean("solver_chose_joint"),
                "joint_pose_entropy_mean": mean("joint_pose_entropy"),
            })
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(path, summary_rows):
    conditions = [str(r["condition"]) for r in summary_rows]
    x = np.arange(len(conditions))
    width = 0.18

    marginalized = np.array([float(r["marginalized_alpha_error_mean"]) for r in summary_rows])
    support_gated = np.array([float(r["support_gated_alpha_error_mean"]) for r in summary_rows])
    solver = np.array([float(r["solver_alpha_error_mean"]) for r in summary_rows])
    oracle = np.array([float(r["oracle_pose_alpha_error_mean"]) for r in summary_rows])

    fig, ax = plt.subplots(figsize=(13.0, 6.0), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.08, right=0.98)

    ax.bar(x - 1.5 * width, marginalized, width=width, color="#1d3557", label="marginalized bank")
    ax.bar(x - 0.5 * width, support_gated, width=width, color="#2a9d8f", label="support-aware baseline")
    ax.bar(x + 0.5 * width, solver, width=width, color="#e63946", label="improved solver (this work)")
    ax.bar(x + 1.5 * width, oracle, width=width, color="#6a4c93", label="oracle pose")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("mean alpha absolute error")
    ax.set_title("Held-out packet: alpha recovery comparison")
    ax.legend(loc="upper right", frameon=True, ncol=2)

    pkt = summary_rows[0]["packet"] if summary_rows else "?"
    fig.suptitle(
        f"Support-Aware Ensemble Solver — {pkt} Packet",
        fontsize=16, fontweight="bold", y=0.96,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cells(path, cell_rows):
    fig, axes = plt.subplots(2, len(FOCUS_CONDITIONS), figsize=(12.4, 7.2), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.98, wspace=0.24, hspace=0.30)

    for col, condition in enumerate(FOCUS_CONDITIONS):
        ratio_mat = np.full((1, len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
        entropy_mat = np.full((1, len(GEOMETRY_SKEW_BIN_LABELS)), np.nan)
        for row in cell_rows:
            if str(row["condition"]) != condition:
                continue
            j = GEOMETRY_SKEW_BIN_LABELS.index(str(row["geometry_skew_bin"]))
            ratio_mat[0, j] = float(row["solver_vs_support_gated_ratio"])
            entropy_mat[0, j] = float(row["joint_pose_entropy_mean"])

        sns.heatmap(ratio_mat, ax=axes[0, col], cmap="viridis", annot=True, fmt=".2f",
                    xticklabels=GEOMETRY_SKEW_BIN_LABELS, yticklabels=[FOCUS_ALPHA_BIN],
                    cbar=(col == len(FOCUS_CONDITIONS) - 1),
                    cbar_kws={"label": "support / solver alpha"} if col == len(FOCUS_CONDITIONS) - 1 else None,
                    vmin=0.6, vmax=1.8)
        axes[0, col].set_title(f"{condition}\nimprovement factor")
        axes[0, col].set_xlabel("geometry skew")
        axes[0, col].set_ylabel("anisotropy" if col == 0 else "")

        sns.heatmap(entropy_mat, ax=axes[1, col], cmap="magma_r", annot=True, fmt=".2f",
                    xticklabels=GEOMETRY_SKEW_BIN_LABELS, yticklabels=[FOCUS_ALPHA_BIN],
                    cbar=(col == len(FOCUS_CONDITIONS) - 1),
                    cbar_kws={"label": "joint entropy"} if col == len(FOCUS_CONDITIONS) - 1 else None,
                    vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"{condition}\njoint pose entropy")
        axes[1, col].set_xlabel("geometry skew")
        axes[1, col].set_ylabel("anisotropy" if col == 0 else "")

    pkt = cell_rows[0]["packet"] if cell_rows else "?"
    fig.suptitle(f"Cell-Level Results — {pkt} Packet", fontsize=16, fontweight="bold", y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run one packet
# ---------------------------------------------------------------------------

def run_packet(seed, packet_label, bank_params, bank_signatures, shifted_bank):
    rng = np.random.default_rng(seed)
    regime_map = {str(r["name"]): r for r in OBSERVATION_REGIMES}
    rows = []

    for condition in FOCUS_CONDITIONS:
        regime = regime_map[condition]
        temperature = softmin_temperature(regime)
        print(f"\n  [{packet_label}] {condition} (T={temperature:.6f})")

        for skew_bin in GEOMETRY_SKEW_BIN_LABELS:
            for trial_idx in range(TRIALS_PER_CELL):
                true_params = sample_conditioned_parameters(rng, FOCUS_ALPHA_BIN, skew_bin)
                clean_sig = anisotropic_forward_signature(true_params)
                rotated_sig, observed_sig, mask, true_shift = observe_pose_free_signature(clean_sig, regime, rng)

                # marginalized bank
                m_scores, m_shifts = marginalized_bank_scores(observed_sig, mask, shifted_bank, temperature)
                m_idx = int(np.argmin(m_scores))
                m_params = bank_params[m_idx]
                m_sig = shifted_bank[m_idx, int(m_shifts[m_idx])]
                m_geom, m_weight, m_alpha = symmetry_aware_errors(true_params, m_params)

                # per-seed three-path refinement
                cond_best_p, cond_best_s, cond_best_sc = m_params, m_sig, float("inf")
                fam_best_p, fam_best_s, fam_best_sc = m_params, m_sig, float("inf")
                j_best_p, j_best_s, j_best_sc = m_params, m_sig, float("inf")
                j_best_ent, j_seed_rank = 1.0, 1

                for sr, idx in enumerate(top_k_indices(m_scores, TOP_K_SEEDS), start=1):
                    sp = bank_params[idx]

                    cp, cs, _, csc = candidate_conditioned_search(observed_sig, mask, sp, temperature)
                    if csc < cond_best_sc:
                        cond_best_sc, cond_best_p, cond_best_s = csc, cp, cs

                    fp, fs, _, fsc = family_switching_refine(observed_sig, mask, sp, temperature)
                    if fsc < fam_best_sc:
                        fam_best_sc, fam_best_p, fam_best_s = fsc, fp, fs

                    jp, js, _, jsc, jent = joint_pose_marginalized_refine(observed_sig, mask, sp, temperature, condition)
                    if jsc < j_best_sc:
                        j_best_sc, j_best_p, j_best_s, j_best_ent, j_seed_rank = jsc, jp, js, jent, sr

                # support-gated baseline
                sg_p, sg_s, _, sg_fam = choose_support_gated_baseline(
                    condition, cond_best_p, cond_best_s, cond_best_sc,
                    fam_best_p, fam_best_s, fam_best_sc)
                sg_geom, sg_weight, sg_alpha = symmetry_aware_errors(true_params, sg_p)
                sg_rmse_val = rmse(sg_s, rotated_sig)

                # joint errors
                j_geom, j_weight, j_alpha = symmetry_aware_errors(true_params, j_best_p)
                j_rmse_val = rmse(j_best_s, rotated_sig)

                # best-of-two by fit RMSE (threshold-free)
                sv_p, sv_s, sv_rmse_val, sv_chose_joint = best_of_two_by_fit_rmse(
                    sg_p, sg_s, sg_rmse_val, j_best_p, j_best_s, j_rmse_val)
                sv_geom, sv_weight, sv_alpha = symmetry_aware_errors(true_params, sv_p)

                # oracle pose ceiling
                oracle_obs, oracle_mask = oracle_align_observation(observed_sig, mask, true_shift)
                op_p, op_s = nearest_neighbor_aligned(oracle_obs, oracle_mask, bank_signatures, bank_params)
                op_geom, op_weight, op_alpha = symmetry_aware_errors(true_params, op_p)
                op_rmse_val = rmse(op_s, clean_sig)

                rows.append(TrialRow(
                    packet=packet_label,
                    condition=condition,
                    geometry_skew_bin=skew_bin,
                    trial_in_cell=trial_idx,
                    true_alpha=float(true_params[5]),
                    true_t=float(true_params[1]),
                    true_rotation_shift=int(true_shift),
                    marginalized_alpha_error=float(m_alpha),
                    marginalized_geometry_mae=float(m_geom),
                    marginalized_weight_mae=float(m_weight),
                    support_gated_alpha_error=float(sg_alpha),
                    support_gated_geometry_mae=float(sg_geom),
                    support_gated_weight_mae=float(sg_weight),
                    support_gated_fit_rmse=float(sg_rmse_val),
                    support_gated_choose_family=int(sg_fam),
                    joint_alpha_error=float(j_alpha),
                    joint_geometry_mae=float(j_geom),
                    joint_weight_mae=float(j_weight),
                    joint_fit_rmse=float(j_rmse_val),
                    joint_score=float(j_best_sc),
                    joint_seed_rank=int(j_seed_rank),
                    joint_pose_entropy=float(j_best_ent),
                    solver_alpha_error=float(sv_alpha),
                    solver_geometry_mae=float(sv_geom),
                    solver_weight_mae=float(sv_weight),
                    solver_fit_rmse=float(sv_rmse_val),
                    solver_chose_joint=int(sv_chose_joint),
                    oracle_pose_alpha_error=float(op_alpha),
                    oracle_pose_geometry_mae=float(op_geom),
                    oracle_pose_weight_mae=float(op_weight),
                    oracle_pose_fit_rmse=float(op_rmse_val),
                ))

                print(f"    {skew_bin} t{trial_idx}: solver={sv_alpha:.4f}{'*' if sv_chose_joint else ''}"
                      f"  sg={sg_alpha:.4f}  joint={j_alpha:.4f}  oracle={op_alpha:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building reference bank …")
    bank_rng = np.random.default_rng(20260324)
    bank_params, bank_sigs = build_reference_bank(REFERENCE_BANK_SIZE, bank_rng, anisotropic=True)
    shifted_bank = build_shift_stack(bank_sigs)

    all_rows = []

    print("\n===== CALIBRATION PACKET (seed={}) =====".format(SEED_CALIBRATION))
    cal_rows = run_packet(SEED_CALIBRATION, "calibration", bank_params, bank_sigs, shifted_bank)
    all_rows.extend(cal_rows)

    print("\n===== HELD-OUT PACKET (seed={}) =====".format(SEED_HOLDOUT))
    ho_rows = run_packet(SEED_HOLDOUT, "holdout", bank_params, bank_sigs, shifted_bank)
    all_rows.extend(ho_rows)

    # summaries
    cal_by_cond = summarize_by_condition(all_rows, "calibration")
    cal_by_cell = summarize_by_cell(all_rows, "calibration")
    ho_by_cond = summarize_by_condition(all_rows, "holdout")
    ho_by_cell = summarize_by_cell(all_rows, "holdout")

    # write
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_ensemble_solver_trials.csv"),
              [asdict(r) for r in all_rows])
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_ensemble_solver_calibration.csv"), cal_by_cond)
    write_csv(os.path.join(OUTPUT_DIR, "support_aware_ensemble_solver_holdout.csv"), ho_by_cond)

    plot_overview(os.path.join(FIGURE_DIR, "overview_holdout.png"), ho_by_cond)
    plot_cells(os.path.join(FIGURE_DIR, "cells_holdout.png"), ho_by_cell)
    plot_overview(os.path.join(FIGURE_DIR, "overview_calibration.png"), cal_by_cond)
    plot_cells(os.path.join(FIGURE_DIR, "cells_calibration.png"), cal_by_cell)

    # headline numbers
    def overall_mean(rows_list, attr):
        return float(np.mean([getattr(r, attr) for r in rows_list]))

    output = {
        "design": {
            "solver_changes": [
                "5 anneal factors [4,2,1,0.5,0.25] instead of 3 [3,1,0.5]",
                "7 scalar grid points instead of 5",
                "extra alpha-only sweep per temperature level",
                "top-5 seeds instead of top-3",
                "best-of-two selection: support-gated vs joint by fit RMSE (threshold-free)",
            ],
            "validation": "disjoint held-out packet (seed 99887766) vs calibration (seed 20260324)",
            "no_tunable_thresholds": True,
        },
        "calibration": {
            "overall_support_gated_mean": overall_mean(cal_rows, "support_gated_alpha_error"),
            "overall_joint_mean": overall_mean(cal_rows, "joint_alpha_error"),
            "overall_solver_mean": overall_mean(cal_rows, "solver_alpha_error"),
            "overall_oracle_pose_mean": overall_mean(cal_rows, "oracle_pose_alpha_error"),
            "by_condition": cal_by_cond,
            "by_cell": cal_by_cell,
        },
        "holdout": {
            "overall_support_gated_mean": overall_mean(ho_rows, "support_gated_alpha_error"),
            "overall_joint_mean": overall_mean(ho_rows, "joint_alpha_error"),
            "overall_solver_mean": overall_mean(ho_rows, "solver_alpha_error"),
            "overall_oracle_pose_mean": overall_mean(ho_rows, "oracle_pose_alpha_error"),
            "by_condition": ho_by_cond,
            "by_cell": ho_by_cell,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "support_aware_ensemble_solver_summary.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for pkt, rows_list, by_cond in [("calibration", cal_rows, cal_by_cond),
                                      ("holdout", ho_rows, ho_by_cond)]:
        sg = overall_mean(rows_list, "support_gated_alpha_error")
        sv = overall_mean(rows_list, "solver_alpha_error")
        jo = overall_mean(rows_list, "joint_alpha_error")
        op = overall_mean(rows_list, "oracle_pose_alpha_error")
        print(f"\n  [{pkt.upper()}]")
        print(f"    support-gated baseline : {sg:.4f}")
        print(f"    joint solver (improved): {jo:.4f}")
        print(f"    solver (best-of-two)   : {sv:.4f}")
        print(f"    oracle pose            : {op:.4f}")
        for r in by_cond:
            print(f"    {r['condition']}: sg={r['support_gated_alpha_error_mean']:.4f}"
                  f"  solver={r['solver_alpha_error_mean']:.4f}"
                  f"  chose_joint={r['solver_chose_joint_fraction']:.2f}")

    print("\n" + json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
