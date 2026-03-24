# Shape Budget Research Roadmap

## Purpose

The current evidence supports a careful, specific claim:

> Under the symmetric constant-sum two-source Euclidean process, `e = c/a` behaves like a sufficient control variable for normalized geometry.

The next phase should not assume that this immediately generalizes. It should test how strong that claim really is, where it breaks, and whether the control-knob interpretation is merely elegant or actually useful.

This roadmap is organized around that goal.

## Current Status

What is already supported:

- the constant-sum two-circle process reconstructs the analytic ellipse
- normalized loci collapse across scale for fixed `e`
- multiple normalized observables behave as one-dimensional functions of `e`

What is not yet established:

- whether `e` remains sufficient once symmetry is broken
- whether `e` is recoverable from noisy or partial observations
- whether `e` outperforms more naive alternatives as a predictive variable
- whether the same budget logic survives in nearby families such as hyperbolas, anisotropic media, or multi-source systems

## Priority Order

| Priority | Experiment | Primary purpose | Success metric |
|---|---|---|---|
| 1 | Unequal growth / unequal budget split | Robustness test | Clean low-dimensional collapse under `(e, w)` with normalized residuals near floating-point error for fixed `(e, w)` and visible one-knob failure when `w` varies |
| 2 | Identifiability and baseline comparison | Operational usefulness | `e` recovers reliably from noisy boundary data and outperforms raw single-variable baselines across scale |
| 3 | Hyperbola flip | Theory extension | Parallel normalized control parameter emerges with clean scale collapse in the fixed-difference family |
| 4 | Controlled anisotropy | Universality test | Collapse survives after adding one directional descriptor or a whitened coordinate transform |
| 5 | Multi-source generalization | Higher-dimensional extension | A low-dimensional allocation space organizes normalized geometry better than raw coordinates alone |

1. Unequal growth / unequal budget split
2. Identifiability and baseline comparison
3. Hyperbola flip
4. Controlled anisotropy
5. Multi-source generalization

This order is deliberate. The first two experiments harden the evidence for the existing claim. The later ones broaden the theory.

---

## Experiment 1: Unequal Growth / Unequal Budget Split

### Question

If the two processes are no longer symmetric, does the one-knob story survive, or does the geometry require an additional parameter?

### Why this matters

This is the strongest immediate robustness test. If a single asymmetry parameter cleanly extends the collapse, then the Shape Budget principle gets stronger. If not, you learn exactly where the original sufficiency claim stops.

### Suggested setup

- Keep the same two-source geometry with separation `2c`
- Replace the symmetric split `r` and `2a-r` with a weighted split
- Examples:
  - fixed weight `w` controlling how the total budget is split
  - unequal growth rates over time
  - offset budget allocation from the start

### Pilot run checklist

- reuse the existing control-knob experiment structure and swap in a weighted split generator
- start with one asymmetry parameter `w` only, not multiple competing asymmetry knobs
- cap the initial sweep away from the degenerate edge, for example `e <= 0.95`
- generate the same core outputs as the base experiment:
  - reconstruction figure
  - fixed-`e` family comparison across `w`
  - fixed-`(e, w)` scale-collapse test
  - response curves or surfaces for a few normalized observables
- keep runtime in the same range as the original control-knob experiment so iteration stays cheap

### Measurements

- reconstruction quality of the resulting locus
- scale-collapse behavior after normalization
- whether a two-parameter family `(e, w)` collapses cleanly
- whether any single effective knob can absorb the asymmetry

### Strong outcome

Normalized geometry collapses under a small parameter set such as `(e, w)`.

### Weakening outcome

No clean low-dimensional collapse survives once symmetry is broken.

### Recommended deliverable

`ASYMMETRY_EXPERIMENT.md` plus figures showing:
- symmetric vs asymmetric families
- failure of one-knob collapse
- success or failure of two-knob collapse

---

## Experiment 2: Identifiability and Baseline Comparison

### Question

Is `e` just a mathematically available parameter, or is it actually the best practical summary variable for prediction and recovery?

### Why this matters

This is the most important “hardening” step. It tests whether the control knob is operational.

### Part A: Identifiability

Given noisy, partial, or sparsely sampled boundary data:

- can we recover `e` accurately?
- how stable is the estimate?
- which observables are best for inverse recovery?

Suggested conditions:

- full boundary with Gaussian noise
- partial arc only
- few sampled points
- heavy noise near high-eccentricity regimes

Suggested outputs:

- recovered `e` versus true `e`
- error bars versus noise level
- phase diagram of identifiability

### Part B: Baseline comparison

Compare predictive performance using:

- `e` alone
- raw separation `d` alone
- raw budget `S` alone
- the pair `(d, S)` without normalization

Prediction targets:

- normalized width `b/a`
- normalized perimeter
- tip curvatures
- regime classification such as slack-rich / mixed / pinched

### Strong outcome

`e` is the simplest variable that preserves predictive power across scale and outperforms raw single-variable baselines.

### Recommended deliverable

`IDENTIFIABILITY_AND_BASELINES.md`

---

## Experiment 3: Hyperbola Flip

### Question

If the sum-budget rule gives an ellipse family, does the fixed-difference rule yield a hyperbola family governed by an analogous control parameter?

### Why this matters

This is the cleanest nearby extension. It tests whether the “budget logic” is deeper than the ellipse case.

### Suggested setup

- Replace the constant-sum condition with a constant-difference condition
- Recreate the same analysis structure:
  - reconstruction
  - normalization
  - scale collapse
  - response curves
  - phase map

### What to look for

- a normalized openness parameter analogous to `e`
- clean collapse across scale
- whether the hyperbola case reads like a deficit-spending dual of the ellipse case

### Strong outcome

A parallel control-knob theory emerges naturally for the hyperbola family.

### Caveat

This is more of an extension than a validation test. It broadens the program but does not by itself strengthen the evidence for the original ellipse claim.

### Recommended deliverable

`HYPERBOLA_TWIN_EXPERIMENT.md`

---

## Experiment 4: Controlled Anisotropy

### Question

Does the one-knob story survive under a slightly warped distance rule, or does anisotropy immediately require additional control variables?

### Why this matters

This is the right way to test universality without jumping too quickly into fully general “warped spaces.”

### Suggested setup

Start with a quadratic anisotropic metric, not a fully general medium:

\[
d_A(x, y) = \sqrt{\begin{bmatrix}x & y\end{bmatrix} A \begin{bmatrix}x \\ y\end{bmatrix}}
\]

with positive-definite `A`.

This gives a controlled way to stretch space in one direction.

### Measurements

- whether normalization by `a` and the anisotropy parameters yields collapse
- whether `e` remains useful after coordinate whitening
- whether a new directional parameter must be added

### Strong outcome

The control-knob idea survives but requires one additional anisotropy descriptor.

### Recommended deliverable

`ANISOTROPY_EXPERIMENT.md`

---

## Experiment 5: Multi-Source Generalization

### Question

When there are three or more centers sharing a fixed total budget, does the one-knob story become a low-dimensional allocation simplex?

### Why this matters

This is the most ambitious extension. If it works, the Shape Budget concept graduates from conic interpretation to a broader multi-source allocation geometry.

### Suggested setup

- three-source constant-sum system first
- normalized placement geometry plus normalized budget allocation variables
- compare direct geometry to candidate low-dimensional control spaces

### What to look for

- whether the family of normalized shapes is still low-dimensional
- whether a simplex of allocation ratios organizes the geometry
- whether useful summary variables emerge

### Caveat

This should come after the two-source case is hardened. Otherwise too many variables change at once.

### Recommended deliverable

`MULTISOURCE_EXPERIMENT.md`

---

## Additional Experiments I Recommend

## Experiment 6: Manifold-Dimension Test

### Question

Do normalized boundaries generated by the current model truly lie on a one-dimensional manifold parameterized by `e`?

### Why this matters

This gives a geometric data-analysis confirmation of the control-knob claim itself.

### Suggested setup

- sample many normalized boundaries across `e`
- vectorize them as shapes
- use PCA or another embedding

### Strong outcome

One dominant dimension explains nearly all variation in the symmetric case.

### Recommended deliverable

`MANIFOLD_DIMENSION_EXPERIMENT.md`

---

## Experiment 7: Edge-Regime Stability

### Question

How stable are different observables as `e -> 0` and `e -> 1`?

### Why this matters

Control parameters are more persuasive when we understand their well-conditioned and ill-conditioned regimes.

### Suggested setup

- very fine sweep near `e = 0`
- very fine sweep near `e = 1`
- measure sensitivity of width, perimeter, and curvatures to small perturbations in `e`

### Strong outcome

Clear map of which downstream quantities are robust and which become unstable near degeneration.

### Recommended deliverable

`EDGE_REGIME_STABILITY.md`

---

## Recommended Near-Term Sequence

If the goal is to harden the evidence as efficiently as possible, I would do:

1. Unequal growth / unequal budget split
2. Identifiability and baseline comparison
3. Manifold-dimension test

If the goal is instead to broaden the concept as quickly as possible, I would do:

1. Hyperbola flip
2. Controlled anisotropy
3. Multi-source generalization

My recommendation is the first path.

## Bottom Line

The project now has enough evidence to justify the phrase “control knob” inside the symmetric two-source Euclidean model.

The next scientific question is not whether the phrase is evocative. It is:

> How far does the sufficiency of `e = c/a` actually extend, and when does it break?

That is the right question for the next phase.

## Philosophical Note

At a deeper level, this roadmap is testing whether Shape Budget is best understood as a process ontology or only as a parameter ontology.

- Parameter ontology:
  `e = c/a` is simply another way to rewrite standard ellipse geometry.
- Process ontology:
  the curve is the geometric residue of a constrained allocation process, and the control variables are meaningful because they organize that process.

If the asymmetry experiment still produces a clean low-dimensional collapse, that would be evidence in favor of the stronger process reading.

## Risk Register

- Risk: asymmetry introduces too many free parameters and collapse fails trivially.
  Mitigation: start with one fixed asymmetry parameter `w` and hold everything else constant.
- Risk: numerical instability near the degenerate edge obscures the result.
  Mitigation: cap early pilot sweeps away from `e -> 1` and only extend to the edge after the pipeline is stable.
- Risk: an appealing extension distracts from hardening the original claim.
  Mitigation: treat hyperbola, anisotropy, and multi-source work as phase-2 only unless Experiment 1 is complete.
