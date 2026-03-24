# Representation Independence Experiment

## Purpose

This experiment tests one of the highest-priority theory-hardening questions:

> does the main inferential Shape Budget result survive a meaningful change in the boundary representation, or is too much of the experimental record tied to the centroid-normalized radial-signature pipeline?

The alternative representation used here is a centroid-normalized support-function profile.

The forward family stays the same.

Only the boundary encoding changes.

That makes this a clean test of representation dependence rather than a different geometry problem.

The experiment script is [run_representation_independence_experiment.py](run.py#L1).

## Pre-Benchmark Logic Audit

Before the benchmark, the new script was checked explicitly.

Code sanity:

- script compiled cleanly

Support-encoding identity audit:

- audit cases: `30`
- canonical exact recovery fraction: `1.0`
- pose-free exact recovery fraction: `1.0`
- max canonical fit RMSE: `0.0`
- max pose-free fit RMSE: `0.0`

That matters because the support profile is only useful as a comparison if it behaves cleanly under the same bank machinery. It does.

## Method

The experiment compares two encodings on matched trials:

- `radial`
- `support`

For each encoding, it runs the same inverse structure in two settings:

- `canonical`
- `pose_free`

And in each setting it compares:

- anisotropy-aware bank
- Euclidean baseline bank

So the experiment is testing three things at once:

1. can the alternative representation still recover the latent control object?
2. does the anisotropy-aware advantage survive?
3. does the selective pose penalty on `alpha` survive?

### Boundary encodings

Radial:

- centroid-centered radius as a function of angle
- normalized by mean radius

Support:

- centroid-centered support value as a function of angle
- normalized by mean support

### Benchmark scale

- anisotropic bank size: `220`
- Euclidean baseline bank size: `120`
- trials per regime: `20`
- observation regimes: all `5` standard regimes

## Main Result

The result is strong.

> The core inferential BGP result survives the representation swap. Under the support encoding, canonical recovery remains useful, the anisotropy-aware inverse still decisively beats the Euclidean baseline, and pose-free observation still degrades `alpha` much more than geometry. The hard pose-free alpha bottleneck therefore does not look like a radial-signature artifact, even though its severity is representation-sensitive in some sparse regimes.

This is exactly the kind of result that hardens the theory while narrowing the bottleneck.

The summary file is [representation_independence_summary.json](outputs/representation_independence_summary.json).

## Canonical Pose: Core Recovery Survives

Under the support encoding, canonical recovery stays in the same general quality band as the radial encoding.

Examples:

- `full_clean`
  - radial alpha: `0.0202`
  - support alpha: `0.0187`
  - radial fit improvement over Euclidean: `14.13x`
  - support fit improvement over Euclidean: `14.48x`

- `partial_arc_noisy`
  - radial alpha: `0.0315`
  - support alpha: `0.0343`
  - radial fit improvement: `9.34x`
  - support fit improvement: `10.14x`

- `sparse_partial_high_noise`
  - radial alpha: `0.0729`
  - support alpha: `0.0625`
  - radial fit improvement: `4.25x`
  - support fit improvement: `5.50x`

That is already enough to say:

> the operational latent-variable result is not confined to the radial signature.

## Pose-Free Observation: Selective Alpha Penalty Survives

This is the most important theory-hardening result.

Under the support encoding, the pose-free penalty still lands much more on `alpha` than on geometry.

Support-encoding pose selectivity:

- best `alpha`-over-geometry selectivity: `4.88x`
- worst `alpha`-over-geometry selectivity: `11.96x`

That means:

> even after swapping encodings, hidden pose still hurts `alpha` far more than geometry.

Examples:

- `full_clean`
  - support geometry penalty: `0.931`
  - support alpha penalty: `10.959`
  - selectivity: `11.77x`

- `partial_arc_noisy`
  - support geometry penalty: `1.004`
  - support alpha penalty: `5.556`
  - selectivity: `5.53x`

- `sparse_partial_high_noise`
  - support geometry penalty: `0.816`
  - support alpha penalty: `3.978`
  - selectivity: `4.88x`

So the selective `alpha` bottleneck is not just a quirk of the radial representation.

## What Changed With The Representation Swap

The support encoding does not reproduce every number.

It changes the severity of the hard pose-free alpha penalty in some sparse regimes.

The most visible differences are:

- `sparse_full_noisy`, pose-free
  - radial alpha: `0.1650`
  - support alpha: `0.2461`
  - support over radial alpha ratio: `1.49`

- `sparse_partial_high_noise`, pose-free
  - radial alpha: `0.1307`
  - support alpha: `0.2486`
  - support over radial alpha ratio: `1.90`

So the bottleneck magnitude is representation-sensitive.

That matters.

But the key point is what did **not** disappear:

- the anisotropy-aware baseline advantage
- useful canonical recovery
- the selective pose penalty on `alpha`

That is why this experiment strengthens BGP even though it does not “solve” the bottleneck.

## Interpretation

This experiment changes the diagnosis in a very useful way.

What it establishes:

- the core BGP inferential result is representation-robust across at least two genuinely different encodings
- the pose-free `alpha` bottleneck is not just a radial-signature artifact
- the bottleneck magnitude still depends on representation, so practical inference design remains important

What it does not support:

- a claim that the current bottleneck has nothing to do with representation

The strongest reading is:

> BGP itself now looks substantially more representation-independent than before, while the hard pose-free anisotropic bottleneck looks partly representation-sensitive in magnitude but not in kind.

That is a strong theory-hardening result.

## Why This Matters For Priority

This experiment answers the triage question directly.

The current bottleneck does not look like a general threat to BGP.

Why:

- the core latent-variable and baseline-improvement result survived the representation swap
- the selective `alpha` penalty also survived the swap
- only the difficulty level of the hardest pose-free sparse cases changed materially

So the bottleneck still matters, but it now looks more like:

- an inference-design problem
- with representation-sensitive severity
- rather than a theory-level failure of BGP

That means it is reasonable to move higher-value theory-hardening work ahead of bottleneck cleanup for a while.

## Figures

Key figures:

- [representation_independence_alpha.png](outputs/figures/representation_independence_alpha.png)
- [representation_independence_selectivity.png](outputs/figures/representation_independence_selectivity.png)

The first figure shows that alpha recovery remains in the same broad quality band under both encodings in canonical pose, while both encodings suffer a larger alpha hit in pose-free mode.

The second figure is the main theory figure:

- left: anisotropy-aware improvement over the Euclidean baseline survives the swap
- right: pose-free `alpha` remains much more fragile than geometry under both encodings

