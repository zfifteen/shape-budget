# Intelligent Bank Design

BGP is the control object. The current bank is only a finite solver approximation to that object.

The current random bank is useful for broad coverage and stability testing, but it has three visible shortcomings in the focused pose-free anisotropic solver stack:

- it spends density uniformly instead of where the observation actually needs it
- it mixes qualitatively different near-best candidates into one family and pushes that complexity downstream
- it makes solver quality too sensitive to bank seed

The next bank should be informed by the layered solver state.

## Design Goal

The next bank should be a backbone-conditioned candidate atlas, not a global random library.

That means:

- start from the recovered Layer 1 backbone
- preserve diversity across genuinely different candidate modes
- allocate density along observed ambiguity directions
- keep a smaller exploration reserve for uncovered structure

## What An Intelligent Bank Must Do

An intelligent bank must satisfy all of the following.

### 1. Respect the Layered Stack

The bank should mirror the current solver program.

- Layer 1 provides a backbone estimate and compression-load state
- Layer 2 provides anchored tolerance and point-recoverability context
- the bank should use that information before spending density on extension coordinates

This is different from the current random bank, which is blind to the recovered backbone.

### 2. Represent Modes, Not Just Volume

The near-best family often contains several plausible candidate groups.

The bank should preserve those groups explicitly instead of letting one dense region dominate because it happened to receive more random samples.

The bank therefore needs:

- cluster-aware candidate selection
- quota or diversity rules across clusters
- explicit tracking of how many candidates each mode contributes

### 3. Spend Density Where The Observation Is Ambiguous

The bank should not treat all directions in latent space equally.

It should increase density:

- near the recovered backbone
- along the directions where the near-best family actually differs
- around candidate modes that survive scoring repeatedly

It should decrease density:

- in redundant duplicate regions
- in modes that rarely survive the score frontier

### 4. Preserve Exploration

An informed bank should not collapse into a purely local deterministic search.

It still needs an exploration reserve so the solver can discover structure that the current atlas does not yet model.

The informed bank should therefore contain:

- a structured core
- plus a smaller exploratory tail

## Proposed Architecture

The next bank should be built in stages.

### Stage 0. Scout Bank

Use a small broad-coverage bank to recover a coarse candidate family.

This stage is not the final bank.
Its job is to expose:

- the backbone neighborhood
- the main candidate clusters
- the active ambiguity directions

The scout stage should move away from pure iid random sampling.
The preferred replacement is a stratified or low-discrepancy design over the control coordinates.

### Stage 1. Candidate Atlas

Construct a candidate atlas from the scout-family survivors.

The atlas should identify:

- candidate clusters
- local density
- consensus pull directions
- ambiguity directions in geometry, weights, and `alpha`
- compression-load state

This stage is the bridge between raw scoring and informed bank construction.

### Stage 2. Backbone-Conditioned Expansion Bank

Build the real bank around the atlas.

The first expansion families should be:

#### Backbone core

Dense candidates near the Layer 1 consensus backbone.

Purpose:

- support a stable geometry layer
- clean up finite-bank noise around the backbone

#### Ambiguity rays

Candidates placed along the directions where the near-best family actually differs.

Purpose:

- resolve the specific ambiguity structure the observation is expressing
- avoid wasting density in irrelevant directions

#### Mode anchors

Candidates placed at representative centers of the discovered clusters.

Purpose:

- keep different plausible solution families visible
- prevent one mode from drowning out the others

#### Exploration reserve

A smaller residual random or low-discrepancy component.

Purpose:

- preserve discovery
- prevent premature closure around a mistaken atlas

## First Allocation Rule

The first informed-bank budget should be allocated by state, not by a fixed global mixture.

The most promising state variables are:

- `compression_load`
- candidate cluster count
- candidate density near the consensus backbone
- anchored `alpha` uncertainty
- correction-pressure or correction-transmission state

The simplest first rule is:

- low compression load: spend more budget in the backbone core
- high compression load: spend more budget on ambiguity rays and mode anchors
- always keep a non-zero exploration reserve

## What Counts As Success

The intelligent bank is useful only if it improves solver-relevant behavior, not just candidate aesthetics.

The primary acceptance tests are:

- reduced bank-seed sensitivity
- lower compression load after backbone recovery
- fewer poisoned consensus families
- cleaner separation between genuinely simple and heavily compressed backbones
- better fresh-block behavior for downstream Layer 2 and Layer 3 policies

## What This Design Is Not

This design is not:

- a hand-tuned fixed library built from hindsight labels
- a regime-label router in disguise
- a replacement for the layered solver stack

It is a better search object for the existing stack.

## Immediate Next Step

Do not implement the informed bank before candidate visibility improves.

The candidate-atlas instrumentation in [Candidate Atlas Instrumentation](./CANDIDATE-ATLAS-INSTRUMENTATION.md) is the required first stage.
The first mining pass over that atlas now lives at [Candidate Atlas Pattern Mining](../experiments/pose-anisotropy-diagnostics/candidate-atlas-pattern-mining/README.md).

Those artifacts should determine:

- which candidate attributes actually matter
- which candidates poison consensus
- which ambiguity directions survive scoring
- how the informed bank should be populated
