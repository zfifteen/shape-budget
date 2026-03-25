# Symmetry-Orbit Design

## Purpose

The recent inverse results point to a more specific next-phase hypothesis.

The solver challenge does not look like a generic failure of the latent-object program.

It looks more like this:

> in the current pose-free radial-signature pipeline, the anisotropy parameter `alpha` sits much closer to the rotation symmetry orbit than the normalized geometry does.

If that is right, then the design criterion for the next experiments changes.

The question is no longer just:

- can we search harder
- can we refine more locally
- can we compress pose more aggressively

It becomes:

- how far is each latent variable from the rotation orbit in the chosen representation
- and can we break or handle that symmetry in a way that separates `alpha` from pose without harming geometry

This note turns that idea into a concrete experiment sequence.

## Mechanism Under Test

The mechanism under test is:

- in the current radial-signature representation, hidden rotation broadens the near-optimal family mainly along `alpha`
- the most plausible mechanism is that `alpha` perturbations are more easily absorbed by rotation than geometry perturbations are

That is narrower than saying `alpha` is fundamentally unrecoverable.

It says only that:

- under the current encoding and pose handling, `alpha` is close to the rotation orbit

That is exactly the kind of statement we can test directly.

## Why This Is Actionable Now

This hypothesis already lines up with four existing artifacts:

- [LATENT_AMBIGUITY_EXPERIMENT.md](pose-anisotropy-diagnostics/latent-ambiguity/README.md#L1)
- [ROTATION_INVARIANT_SPECTRAL_EXPERIMENT.md](pose-anisotropy-diagnostics/rotation-invariant-spectral/README.md#L1)
- [SHIFT_MARGINALIZED_POSE_EXPERIMENT.md](pose-anisotropy-interventions/shift-marginalized-pose/README.md#L1)
- [SHIFT_MARGINALIZED_LOCAL_REFINEMENT_EXPERIMENT.md](pose-anisotropy-interventions/shift-marginalized-local-refinement/README.md#L1)

Taken together, those results say:

- geometry stays comparatively stable
- `alpha` broadens sharply once pose is hidden
- blunt invariant compression does not help
- softer pose handling helps somewhat
- within-basin local refinement does not rescue `alpha`

That pattern is exactly what a symmetry-orbit proximity mechanism predicts.

So this is not a shelf-it-later concept.

It is specific enough to steer the next experiments now.

## Design Rule

The new design rule is:

> prefer experiments that measure or break rotation-orbit aliasing directly.

In practice that means:

- measure orbit proximity before adding more bank density
- test symmetry-breaking alignment before adding more local refinement
- use oracle alignment to separate representation limits from alignment limits
- map where alignment helps and where symmetry itself makes alignment unstable

## Next Experiments

### Experiment A: Orbit-Proximity Diagnostic

Question:

- are small `alpha` changes actually closer to the rotation orbit than matched-size geometry changes in the current representation?

Setup:

- start from one true latent state
- perturb only `alpha` by a controlled amount
- perturb geometry by a matched normalized amount
- for each perturbation, compute:
  - raw signature distance
  - best-rotation-orbit distance
  - orbit-absorption ratio = best-rotation distance divided by raw distance

Primary readout:

- if the hypothesis is right, `alpha` perturbations should have much smaller orbit-minimized distances and higher orbit absorption than geometry perturbations

Strong outcome:

- `alpha` is measurably more rotation-aliased than geometry in the current radial-signature space

Weakening outcome:

- orbit absorption is similar across latent directions, which would push us away from the symmetry-orbit explanation

Recommended deliverable:

- `ORBIT_PROXIMITY_EXPERIMENT.md`

### Experiment B: Orientation-Locking Pre-Alignment

Question:

- if we break rotational symmetry before radial encoding, does `alpha` recover much more than geometry does?

Setup:

- estimate a canonical orientation from the observed boundary before radial signature encoding
- begin with simple locking rules:
  - principal-axis alignment from the boundary point cloud
  - low-order harmonic phase alignment from the radial signature
- rerun the pose-free weighted anisotropic inverse on the aligned observations

Primary prediction:

- `alpha` error ratios should move substantially back toward canonical-pose performance
- geometry error ratios should move only slightly because geometry was already comparatively stable

Strong outcome:

- large `alpha` gain with small geometry change

Weakening outcome:

- little `alpha` change after locking, which would suggest the aliasing is deeper than orientation handling alone

Recommended deliverable:

- `ORIENTATION_LOCKING_EXPERIMENT.md`

### Experiment C: Oracle Alignment Ceiling

Question:

- how much of the current `alpha` loss is due to imperfect alignment, and how much remains even with idealized orientation information?

Setup:

- use the clean full boundary to define an oracle orientation
- align the observation with that oracle before encoding
- compare:
  - current pose-free baseline
  - practical orientation-locking
  - oracle-aligned inverse
  - canonical-pose inverse

Primary readout:

- the gap between practical locking and oracle locking tells us how much headroom is in better alignment
- the gap between oracle locking and canonical tells us how much ambiguity remains even after idealized orientation handling

Strong outcome:

- oracle locking nearly recovers canonical `alpha`, which would strongly support the orbit-proximity mechanism

Recommended deliverable:

- `ORACLE_ALIGNMENT_CEILING_EXPERIMENT.md`

### Experiment D: Alignment Failure Map

Question:

- where does orientation-locking fail or become unstable?

Setup:

- map performance across:
  - near-isotropic `alpha`
  - stronger anisotropy
  - more and less symmetric source geometries
  - full, partial, and sparse support
  - low and high noise

Primary readout:

- identify regimes where alignment is well-posed versus regimes where symmetry makes orientation itself ambiguous

Why this matters:

- it prevents us from overreading one good alignment result
- it also tells us whether “alpha near the orbit” is a global issue or only a near-symmetric issue

Recommended deliverable:

- `ALIGNMENT_FAILURE_MAP_EXPERIMENT.md`

## Recommended Order

The best order is:

1. orbit-proximity diagnostic
2. orientation-locking experiment
3. oracle alignment ceiling
4. alignment failure map

That order matters.

The first experiment tests the mechanism directly.

The second tests the main intervention implied by the mechanism.

The third tells us whether the intervention is close to the right solution or still leaving large recoverable signal on the table.

The fourth tells us where the mechanism is strongest and where the intervention becomes ill-posed.

## What To Stop Doing For Now

Until this branch is tested, the next step should not be:

- a much larger bank
- a deeper single-basin local refinement
- a more aggressive low-order invariant compression

Those all assume the problem is mainly search or resolution.

The experimental record points more toward symmetry aliasing.

## Decision Rule

If Experiment A and Experiment B both land strongly, then the symmetry-orbit mechanism becomes the right organizing explanation for the next phase of the project.

If Experiment A is weak, or Experiment B fails to rescue `alpha`, then this mechanism should be downgraded and treated as only one partial factor.

That makes this a good mechanism under test:

- it is sharp
- it is falsifiable
- it gives immediate design guidance

## Plain-Language Summary

The simplest version is:

- the current inverse reads geometry well because geometry sits far from the pose symmetry orbit
- it reads `alpha` poorly because `alpha` sits much closer to that orbit
- so the next experiments should measure that orbit proximity directly and then try to break the symmetry before encoding

That is the right next branch.
