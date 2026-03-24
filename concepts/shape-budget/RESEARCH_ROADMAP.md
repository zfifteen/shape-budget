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
- the first asymmetry pilot breaks one-knob sufficiency but preserves a clean two-parameter collapse under `(e, w)`
- `e` is recoverable from noisy, partial, and sparse boundary observations when the source positions are known
- `e` strongly outperforms raw `d`, raw `S`, and low-capacity models on `(d, S)` under a scale-held-out predictive test
- in a radial-signature boundary representation, the symmetric normalized family behaves like a one-dimensional curved manifold ordered by `e`
- the circular edge is first-order flat across the tested normalized observables, while the degenerate edge splits them into sharp and moderate probes of `e`
- the fixed-difference twin yields a hyperbola family with clean one-knob collapse under `lambda = a / c`
- under a controlled quadratic anisotropy, raw geometry becomes a two-parameter family `(e, alpha)` and whitening restores the original one-knob collapse
- in the equal-weight three-source constant-sum case, normalized geometry is organized by the normalized source triangle relative to budget, the induced allocation-share loop is scale-invariant, the equilateral slice is near one-parameter, and the broader family behaves like a low-dimensional roughly three-parameter manifold
- in the weighted three-source constant-sum case, normalized geometry is organized by the normalized source triangle plus the weight simplex, fixed geometry-plus-weights collapses exactly across scale, varying weights breaks equal-weight sufficiency, the equilateral weighted slice is near two-parameter, and the broader family behaves like a low-dimensional roughly five-parameter manifold
- in the weighted three-source canonical-pose inverse setting, a simple boundary-only reference-bank inverse recovers the normalized geometry and normalized weights with useful accuracy and consistently outperforms an equal-weight baseline
- in the weighted three-source pose-free inverse setting, a cyclic-shift-aware boundary-only weighted reference-bank inverse still recovers the normalized geometry and normalized weights with useful accuracy and continues to outperform an equal-weight baseline across all tested regimes
- in the weighted three-source anisotropic canonical-pose inverse setting, a boundary-only anisotropy-aware reference-bank inverse jointly recovers normalized geometry, normalized weights, and the medium anisotropy parameter `alpha`, and it decisively outperforms a Euclidean weighted baseline across all tested regimes
- in the weighted three-source pose-free anisotropic inverse setting, a cyclic-shift-aware anisotropy-aware inverse still recovers normalized geometry and weights with useful accuracy and continues to decisively outperform a Euclidean weighted baseline, but `alpha` becomes much more weakly identified once rotation is hidden too
- a pilot direct-alpha-refinement pass under the same pose-free anisotropic setting improves fit and sometimes improves `alpha`, but it does not rescue `alpha` uniformly, which suggests the bottleneck is not only coarse search resolution
- a matched canonical-versus-pose-free ambiguity profile now shows that hidden rotation broadens the near-optimal latent family mainly in `alpha`, with pose-free top-10 `alpha` spans about `3.7x` to `5.6x` larger and pose-free best-`alpha` errors about `11.4x` to `31.0x` larger, while geometry dispersion stays nearly unchanged
- a genuinely pose-invariant low-order spectral encoding has now been tested and audited cleanly; it generally does not improve `alpha` and often makes it much worse under partial support, which means the pose-free `alpha` bottleneck is not merely a shift-search artifact and naive invariant compression is too lossy
- a soft shift-marginalized pose score has now been tested and audited cleanly; it modestly improves `alpha` in most noisy regimes, tightens the top-10 `alpha` envelope in every regime, and gives its strongest gains in the hardest sparse partial setting, which means the hard best-shift pose rule was part of the bottleneck
- a micro-pilot shift-marginalized local refinement pass has now been tested and audited cleanly; it tightens the selected local basin in every regime and sometimes improves fit or geometry, but it does not materially improve `alpha`, which means the remaining bottleneck is not just within-basin cleanup
- a sharper working mechanism now guides the next phase: in the current pose-free radial-signature pipeline, `alpha` appears much closer to the rotation symmetry orbit than normalized geometry does, so the next experiments should measure orbit proximity directly and test symmetry-breaking alignment before expanding search or bank complexity
- a direct orbit-proximity diagnostic has now been completed: in clean full-signature space, `alpha` perturbations are more rotation-absorbable than matched random geometry perturbations, especially at medium and larger step sizes, but the absolute effect is still moderate, so orbit proximity looks like a real contributor rather than a complete explanation of the pose-free `alpha` penalty
- a direct orientation-locking intervention has now been completed: two simple observation-only symmetry-breaking locks pass exact clean audits and give small `alpha` gains in fully observed conditions, but they degrade sharply under partial and sparse support, so practical alignment instability is now the main issue rather than the logical possibility of locking
- an oracle alignment ceiling has now been completed: once true pose is given, `alpha` improves by about `5.7x` to `13.2x` across all tested regimes while geometry changes only modestly, which means most of the pose-free `alpha` penalty is real alignment headroom rather than missing signal
- an alignment failure map has now been completed: practical locking success is highly structured by support regime, geometry skew, and anisotropy strength, with full observations and high-skew weak-anisotropy cells being the friendliest, and sparse or partial low-to-mid-skew cells being the main failure zones where simple locks lose large oracle headroom
- a candidate-conditioned local `shift + alpha` search has now been completed: it improves alpha recovery over observation-only shift marginalization in four of five regimes and strongly in the sparse-partial branch, but it does not uniformly rescue the moderate sparse band and still fails sharply in the sparse-full moderate family, which means the orbit-alias story is real but still incomplete

What is not yet established:

- whether `alpha` identifiability can be materially improved under the combined nuisance of unknown rotation plus unknown medium anisotropy
- whether the pose-free `alpha` bottleneck can be materially reduced by a better representation, uncertainty estimate, or joint local refinement beyond alpha-only search
- whether a richer pose-equivariant or hybrid coarse-to-fine representation can improve `alpha` without discarding as much shape information as the low-order spectral invariant
- whether broader multi-seed or joint weight-plus-geometry-plus-`alpha` refinement can improve `alpha` once the single-seed local basin has already been tightened
- whether a practical alignment method can capture a meaningful fraction of the large oracle headroom inside the identified sparse or partial low-to-mid-skew failure cells
- whether better-than-naive alignment can remain stable under partial and sparse support while still helping `alpha`
- how sharply the alignment failure boundary moves under finer sweeps of support fraction, anisotropy strength, and source geometry
- whether the sparse-full moderate failure family is mainly a wrong-family selection problem, a deeper representation-level alias, or a limitation of within-family local alpha search
- whether the same budget logic survives under unknown anisotropy axes, richer warped media, or once the source count increases further
- whether the manifold conclusion remains equally strong under alternative shape encodings or outside the symmetric setting
- how the conditioning map changes once symmetry or Euclidean distance is relaxed
- whether the weighted inverse problem remains tractable under richer non-quadratic or spatially varying media, and how the story changes as the source count increases further

## Priority Order

| Status | Experiment | Primary purpose | Success metric |
|---|---|---|---|
| completed | Unequal growth / unequal budget split | Robustness test | Completed: one-knob sufficiency fails under asymmetry, but normalized geometry collapses cleanly under `(e, w)` |
| completed | Identifiability and baseline comparison | Operational usefulness | Completed: `e` recovers reliably in the known-source setting and outperforms raw baselines across scale |
| completed | Manifold-dimension test | One-dimensionality confirmation | Completed: PC1 explains about 98.9 percent of radial-signature variance, 1D Isomap recovers the family ordering perfectly, and scale overlays collapse in embedding space |
| completed | Edge-regime stability | Conditioning map | Completed: near-circular geometry is first-order flat, while high-e geometry separates into sharp width and curvature probes versus a smoother perimeter probe |
| completed | Hyperbola flip | Theory extension | Completed: the fixed-difference twin produces a hyperbola family with one-knob collapse under `lambda = a / c` |
| completed | Controlled anisotropy | Universality test | Completed: raw geometry upgrades to `(e, alpha)` and whitening restores exact one-knob collapse |
| completed | Multi-source generalization | Higher-dimensional extension | Completed: fixed normalized source triangles give exact boundary and simplex-loop collapse across scale, the equilateral slice is near one-parameter, and the broader family is low-dimensional with about three principal directions |
| completed | Weighted multi-source generalization | Higher-dimensional robustness | Completed: fixed geometry plus weights gives exact scale collapse, varying weights breaks equal-weight sufficiency, the equilateral weighted slice is near two-parameter, and the broader family is low-dimensional with about five principal directions |
| completed | Weighted multi-source inverse | Inferential usefulness | Completed: in canonical pose, a boundary-only weighted reference-bank inverse recovers normalized geometry and weights with useful accuracy and beats an equal-weight baseline by about 1.9x to 5.5x |
| completed | Pose-free weighted inverse | Inferential robustness | Completed: with unknown rotation, a cyclic-shift-aware weighted inverse still recovers normalized geometry and weights with useful accuracy and beats the equal-weight baseline by about 1.1x to 5.4x |
| completed | Weighted anisotropic inverse | Medium-aware inferential robustness | Completed: in canonical pose with unknown `alpha`, an anisotropy-aware weighted inverse jointly recovers normalized geometry, normalized weights, and medium anisotropy, and beats a Euclidean weighted baseline by about 7.6x to 14.0x |
| completed | Pose-free weighted anisotropic inverse | Combined-nuisance robustness | Completed: with unknown rotation and unknown `alpha`, a cyclic-shift-aware anisotropy-aware inverse still recovers normalized geometry and weights with useful accuracy and beats a Euclidean weighted baseline by about 3.8x to 15.9x, but `alpha` becomes much more weakly identified |
| completed | Matched latent ambiguity profiling | Bottleneck diagnosis | Completed: matched canonical-versus-pose-free trials show that hidden rotation broadens the near-optimal latent family mainly in `alpha`, with `alpha` span ratios of about 3.7x to 5.6x and near-tied `alpha`-diverse families becoming common in the harder pose-free regimes |
| completed | Rotation-invariant spectral representation test | Representation diagnosis | Completed: a clean low-order pose-invariant spectral encoding generally failed to improve `alpha`, often worsened fit and ambiguity under partial support, and showed severe conditioning issues in the hardest partial regimes |
| completed | Shift-marginalized pose scoring | Pose-mechanism diagnosis | Completed: soft shift-marginalization modestly improves `alpha` in most noisy regimes, tightens the top-10 `alpha` envelope in every regime, and gives its strongest gains in the hardest sparse partial setting |
| completed | Shift-marginalized local refinement micro-pilot | Within-basin cleanup test | Completed: one-seed one-round local refinement tightened the selected local basin in every regime and sometimes improved fit or geometry, but it did not materially improve `alpha`, so the remaining bottleneck is not just local cleanup |
| completed | Orbit-proximity diagnostic | Mechanism test | Completed: in clean full-signature space, `alpha` perturbations show consistently higher orbit absorption and nontrivial best-shift frequency than matched random geometry perturbations, but the absolute effect remains moderate |
| completed | Orientation-locking pre-alignment | Symmetry-breaking test | Completed: simple observation-only locks passed exact clean audits and gave small gains in full observations, but they degraded sharply under partial and sparse support, so naive locking does not rescue `alpha` where it matters most |
| completed | Oracle alignment ceiling | Upper-bound test | Completed: oracle pose information improves `alpha` by about 5.7x to 13.2x across all regimes while leaving geometry roughly stable, so most of the current alpha loss is genuine alignment headroom |
| completed | Alignment failure map | Robustness map | Completed: locking success is highly structured by support, geometry skew, and anisotropy strength; full observations and high-skew cells are friendliest, while sparse or partial low-to-mid-skew cells are the main failure zones |
| completed | Candidate-conditioned local shift-alpha search | Mechanism-targeted intervention | Completed: local candidate-family disentanglement improves alpha recovery in four of five regimes and strongly in sparse-partial cells, but it does not uniformly rescue the moderate sparse band and still fails sharply in the sparse-full moderate family |
| completed | Multi-seed family-switching refinement | Upstream family-selection test | Completed: letting top-seed families move locally in geometry plus `alpha` helps the sparse-full moderate branch overall and modestly improves the original sparse-full moderate mid-skew miss, but it hurts the sparse-partial moderate branch; the dominant new motion is geometric rather than anisotropic |
| completed | Orbit-alias regime router pilot | Policy-routing test | Completed: neither a raw anisotropy-to-skew ratio nor the first support-aware orbit-alias index beats the fixed alpha-only policy overall, but the raw ratio nearly matches the sparse-full winner while the sparse-partial branch still resists scalar routing |
| completed | Representation independence | Theory-hardening test | Completed: the core inferential story survives a swap from radial signatures to support-function profiles; useful canonical recovery, strong anisotropy-aware baseline gains, and selective pose-free `alpha` fragility all persist, though the hardest sparse bottleneck changes in severity |

The original roadmap sequence is complete, and the post-roadmap multi-source robustness plus Euclidean and anisotropic inverse extensions are now complete as well. The experiments hardened the existing claim first and then broadened it in a controlled way, and they have now surfaced a specific bottleneck: under pose-free observation, the latent family broadens mainly along the medium-anisotropy direction rather than along normalized geometry. Naive invariant compression does not fix that, softer shift-marginalized pose handling does recover a real part of the lost signal, and a one-seed local cleanup pass is not enough to turn that tighter local organization into materially better `alpha` identification. The orbit-proximity diagnostic supports the careful symmetry-orbit story, the direct orientation-locking intervention shows that naive symmetry-breaking is too brittle under incomplete support, the oracle ceiling shows that most of the lost `alpha` signal is in fact recoverable with perfect pose, and the failure map shows that the remaining practical problem is not uniform: it is concentrated in sparse or partial low-to-mid-skew regions where simple locking loses large oracle headroom. A newer candidate-conditioned local `shift + alpha` intervention showed that the orbit-alias story is actionable but incomplete: local candidate-family disentanglement helps substantially in the sparse-partial branch and in several other hard cells, but it does not uniformly solve the moderate sparse band and still fails sharply in the sparse-full moderate family. The multi-seed family-switching refinement sharpened that diagnosis further: sparse-full moderate cells do benefit from geometry-level family switching, including the original mid-skew target slice, while sparse-partial moderate cells often get worse under the same added freedom. The dominant corrective motion is usually geometric rather than anisotropic. The first orbit-alias regime-router pilot then added one more useful constraint: a raw anisotropy-to-skew ratio nearly matches the right policy in the sparse-full branch, but neither that scalar nor the first support-aware alias index is enough to beat the fixed alpha-only policy overall, because the sparse-partial branch still resists one-scalar routing. The newer representation-independence test then clarified the bigger picture: the core inferential BGP story survives a swap from radial signatures to support-function profiles, including useful canonical recovery, strong anisotropy-aware advantage over a Euclidean baseline, and the selective pose-free `alpha` penalty. That means the remaining bottleneck is not a general threat to BGP. It is better read as a practical inverse bottleneck whose severity depends on representation and support regime. The current highest-value open work is therefore theory-hardening and scope-defining experimentation, not bottleneck cleanup alone.

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

1. Hyperbola flip
2. Controlled anisotropy
3. Multi-source generalization

If the goal is instead to broaden the concept as quickly as possible, I would do:

1. Hyperbola flip
2. Controlled anisotropy
3. Multi-source generalization

My recommendation is the first path.

## Bottom Line

The project now has enough evidence to justify the phrase “control knob” inside the symmetric two-source Euclidean model, enough inverse/predictive evidence to say that the knob is operational in the known-source setting, enough boundary-space evidence to say that the normalized family itself is organized by a one-dimensional manifold, and enough conditioning evidence to say where the resulting observables are informative versus flat.

It also now has a clean matched twin on the fixed-difference side and a controlled anisotropy result, which suggests the budget logic is organizing a small low-dimensional family of classical two-source geometries rather than only one isolated conic.

The next scientific question is not whether the phrase is evocative. It is:

> How compactly does Shape Budget continue to organize geometry once we leave the simplest symmetric setting?

That is the right question for the next phase.

## Philosophical Note

At a deeper level, this roadmap is testing whether Shape Budget is best understood as a process ontology or only as a parameter ontology.

- Parameter ontology:
  `e = c/a` is simply another way to rewrite standard ellipse geometry.
- Process ontology:
  the curve is the geometric residue of a constrained allocation process, and the control variables are meaningful because they organize that process.

If the asymmetry experiment still produces a clean low-dimensional collapse, that would be evidence in favor of the stronger process reading.

The asymmetry pilot and the identifiability/baseline experiment now both push in that direction:

- asymmetry did not destroy the program; it promoted the family from one control dimension to two
- the symmetric control variable is recoverable and predictive, not just algebraically available
- the symmetric family is one-dimensional in a natural nonlinear boundary-space sense, not only in a few chosen observables
- the edge behavior is now mapped: near-circular geometry is low-information, while near-degenerate geometry separates observables by sharpness
- the nearby deficit-spending twin does not break the program; it extends it into a second one-knob family with a flipped bounded ratio
- a simple directional warp does not break the program either; it adds one descriptor in raw space and disappears under whitening

## Risk Register

- Risk: known-source identifiability looks strong but the harder unknown-source inverse problem may be substantially less stable.
  Mitigation: keep future inverse experiments explicit about which latent quantities are assumed known.
- Risk: symmetric observables can be deceptively redundant because several normalized quantities collapse to the same function of `e`.
  Mitigation: include boundary-space or manifold-level tests, not only scalar observables.
- Risk: the manifold result could be overstated if it depends too strongly on one convenient shape encoding.
  Mitigation: treat the current radial-signature result as strong evidence in a natural representation, and replicate later with support-function or signed-distance encodings if needed.
- Risk: extension work may blur together genuinely new phenomena with artifacts of symmetry or Euclidean distance.
  Mitigation: treat hyperbola and anisotropy as explicit theory-extension experiments, not as automatic consequences of the base ellipse case.
