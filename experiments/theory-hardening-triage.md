# Theory Hardening Triage

## Purpose

This note records the current experiment priority after the recent pose-free
anisotropic solver milestone work.

The key question is not whether that solver challenge matters at all.

It does.

The key question is:

> did the focused solver bottleneck ever jeopardize the Budget Governor Principle in a general way, or was it a narrower inverse-and-representation problem that should not dominate the research program?

The solver challenge belongs to the second category.

## Current Read

The pose-free weighted anisotropic bottleneck was real, but it was never a general threat to BGP.

Why:

- the forward and inverse evidence for the core control object is already strong in the symmetric two-source, asymmetry, hyperbola, anisotropy, and multi-source branches
- the hardest unresolved issue is concentrated in a very specific branch:
  - weighted multi-source
  - anisotropic medium
  - pose-free observation
  - incomplete support
- the oracle-alignment result already shows that much of the lost `alpha` signal is present in the observation and is being lost by practical inference rather than by failure of the underlying budget-governed structure

The focused moderate sparse slice is now solved in the tested regime by the entropy-gated bank ensemble solver, which means the bottleneck is best read as:

- an important solver-design challenge in inverse recovery
- a useful mechanism testbed
- but not the main determinant of whether BGP stands or falls as a broader theory

## What Evidence Matters Most Now

The next phase should prioritize evidence that hardens BGP at the theory level.

The main evidence classes are:

1. representation independence
2. operational usefulness
3. scope boundaries and falsification structure
4. broader-scope solver validation and extension

That order is intentional.

## Priority 1: Representation Independence

### Why it matters

Too much of the current inverse pipeline still passes through the centroid-normalized radial-signature pipeline.

If the core results survive a genuinely different boundary encoding, then BGP becomes much harder to dismiss as a representation artifact.

If the solver challenge changes substantially, that is also valuable:

- the theory gets cleaner
- the solver challenge becomes more local

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

## Priority 4: Broader-Scope Solver Validation And Extension

### Why it still matters

The focused slice is solved in the tested regime, but broader solver work still matters scientifically and practically.

The remaining work is no longer “close the focused bottleneck at all.” It is:

- validate that the same solver-policy idea extends beyond the solved slice
- test harder support and anisotropy cells
- test broader fresh-bank stability
- and move outward to harder extension branches

### Current read on the remaining solver work

- the focused moderate sparse slice no longer needs to dominate the program
- solver extension work should now target broader validation and outward harder-scope tests
- the theory-hardening sequence above remains the right completed foundation for that extension work

So solver work should remain active, but the framing should change from “can the focused bottleneck be closed?” to “how far does the solved inverse policy generalize?”

## Recommended Order

1. `REPRESENTATION_INDEPENDENCE_EXPERIMENT`
2. `PROBE_SPECIALIZATION_EXPERIMENT`
3. `SCOPE_BOUNDARY_EXPERIMENT`
4. return to broader-scope solver validation and harder extension branches

## Working Decision

The focused solver bottleneck should be treated as:

- important
- informative
- worth extending outward carefully

But not as the main blocker on the broader BGP program.

That means the immediate theory-hardening next experiment should have been:

> `REPRESENTATION_INDEPENDENCE_EXPERIMENT`

## Status Update

That experiment has now been completed.

The result was the decisive one:

- the core inferential BGP result survived the representation swap
- the pose-free anisotropic solver challenge persisted in kind, though not with identical severity

So the triage decision now becomes even clearer:

> the solver challenge is still important, but it does not look like the highest-priority threat to BGP. The next theory-hardening priority should move to operational usefulness and then to scope boundaries.

That operational-usefulness step has now also been completed.

The result was useful and disciplined rather than maximal:

- the best practical inverse probe really does change with depletion phase
- under equal-budget dedicated sampling, the practical split is mainly between width and perimeter
- major-tip curvature wins in the ideal direct-scalar benchmark but loses in the practical equal-budget benchmark
- a simple perimeter-pilot router modestly beats the best fixed probe in three of four regimes

So BGP now has partial support as an experimental-control principle.

That scope-boundary step has now also been completed.

The result was the disciplined one the triage needed:

- exact one-knob scope is established in the symmetric ellipse base case and the hyperbola twin
- asymmetry and raw anisotropy explicitly falsify universal one-scalar compression, but compact corrected control objects restore collapse
- equal-weight three-source is about three-parameter and weighted three-source about five-parameter
- the pose-free anisotropic solver challenge is now localized as a selective `alpha` fragility, not a general loss of compact budget structure

The focused solver milestone has now also been completed:

- the entropy-gated bank ensemble solver resolves the tested moderate sparse slice
- the result is a solver-policy fix, not a theory rewrite
- the remaining open work is broader validation and harder extension branches

So the triage decision now becomes:

> the theory-hardening sequence is complete, and the focused solver slice is solved in the tested regime. The next immediate priority can move to broader-scope solver validation or outward harder-scope tests, but it no longer needs to treat the old focused bottleneck as the main blocker.
