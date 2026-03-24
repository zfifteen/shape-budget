# Theory Hardening Triage

## Purpose

This note records the current experiment priority after the recent pose-free
anisotropic bottleneck work.

The key question is not whether that bottleneck matters at all.

It does.

The key question is:

> does the current bottleneck jeopardize the Budget Governor Principle in a general way, or is it a narrower inverse-and-representation problem that should not dominate the research program?

The bottleneck belongs to the second category.

## Current Read

The pose-free weighted anisotropic bottleneck is real, but it is not a general threat to BGP.

Why:

- the forward and inverse evidence for the core control object is already strong in the symmetric two-source, asymmetry, hyperbola, anisotropy, and multi-source branches
- the hardest unresolved issue is concentrated in a very specific branch:
  - weighted multi-source
  - anisotropic medium
  - pose-free observation
  - incomplete support
- the oracle-alignment result already shows that much of the lost `alpha` signal is present in the observation and is being lost by practical inference rather than by failure of the underlying budget-governed structure

So the current bottleneck is best read as:

- an important inverse bottleneck
- a useful mechanism testbed
- but not the main determinant of whether BGP stands or falls as a broader theory

## What Evidence Matters Most Now

The next phase should prioritize evidence that hardens BGP at the theory level.

The main evidence classes are:

1. representation independence
2. operational usefulness
3. scope boundaries and falsification structure
4. bottleneck cleanup

That order is intentional.

## Priority 1: Representation Independence

### Why it matters

Too much of the current inverse pipeline still passes through the centroid-normalized radial-signature pipeline.

If the core results survive a genuinely different boundary encoding, then BGP becomes much harder to dismiss as a representation artifact.

If the bottleneck changes substantially, that is also valuable:

- the theory gets cleaner
- the bottleneck becomes more local

### Main question

Do the key control-object and inverse results survive a representation swap?

### Recommended experiment

`REPRESENTATION_INDEPENDENCE_EXPERIMENT`

Suggested first alternative:

- centroid-centered normalized support-function profile

Core comparisons:

- canonical weighted anisotropic inverse under radial vs support encoding
- pose-free weighted anisotropic inverse under radial vs support encoding
- whether geometry remains more stable than `alpha` under both encodings

### Strong outcome

The support encoding preserves the same broad structure:

- useful canonical recovery
- anisotropy-aware improvement over Euclidean baseline
- pose-free penalty concentrated more in `alpha` than in geometry

### Weakening outcome

The support encoding destroys the core inferential pattern or removes the selective `alpha` penalty entirely.

## Priority 2: Operational Usefulness

### Why it matters

BGP is strongest if it does not only describe shape or latent structure, but improves what you do next.

### Main question

Can BGP tell you which observable or measurement strategy to trust?

### Recommended experiment

`PROBE_SPECIALIZATION_EXPERIMENT`

Focus:

- test whether depletion phase predicts which observable gives the best inverse readout
- compare fixed probes against phase-adaptive probe choice

This is the cleanest route to showing that BGP is an experimental-control principle, not just a geometric description.

## Priority 3: Scope Boundaries

### Why it matters

A stronger theory is not one that claims everything.

A stronger theory is one that states clearly where it works, where it weakens, and what kind of object it is.

### Main question

Where does BGP stay compact and where does it stop compressing the problem cleanly?

### Recommended experiment types

- failure-map summaries across representations
- negative controls where normalization should not collapse
- branches where the control object clearly requires more dimensions

The goal is a disciplined scope map, not just more positive examples.

## Priority 4: Bottleneck Cleanup

### Why it still matters

The current bottleneck is scientifically important and can still produce publishable mechanism results.

But it is now a branch-specific disentanglement problem, not the highest-priority theory-hardening problem.

### Current read on the bottleneck

- sparse-full observations seem to benefit from geometry-level family switching
- sparse-partial observations often benefit more from fixed-family local alpha handling
- the first scalar regime-router pilot did not solve this globally

So the bottleneck should remain active, but it should not dominate the program until the broader theory-hardening work above is stronger.

## Recommended Order

1. `REPRESENTATION_INDEPENDENCE_EXPERIMENT`
2. `PROBE_SPECIALIZATION_EXPERIMENT`
3. `SCOPE_BOUNDARY_EXPERIMENT`
4. return to branch-aware bottleneck routing and cleanup

## Working Decision

The current bottleneck should be treated as:

- important
- informative
- worth revisiting early

But not as the main blocker on the broader BGP program.

That means the immediate next experiment should be:

> `REPRESENTATION_INDEPENDENCE_EXPERIMENT`

## Status Update

That experiment has now been completed.

The result was the decisive one:

- the core inferential BGP result survived the representation swap
- the pose-free anisotropic bottleneck persisted in kind, though not with identical severity

So the triage decision now becomes even clearer:

> the bottleneck is still important, but it does not look like the highest-priority threat to BGP. The next theory-hardening priority should move to operational usefulness and then to scope boundaries.

That operational-usefulness step has now also been completed.

The result was useful and disciplined rather than maximal:

- the best practical inverse probe really does change with depletion phase
- under equal-budget dedicated sampling, the practical split is mainly between width and perimeter
- major-tip curvature wins in the ideal direct-scalar benchmark but loses in the practical equal-budget benchmark
- a simple perimeter-pilot router modestly beats the best fixed probe in three of four regimes

So BGP now has partial support as an experimental-control principle.

That means the next theory-hardening priority should move to:

> `SCOPE_BOUNDARY_EXPERIMENT`
