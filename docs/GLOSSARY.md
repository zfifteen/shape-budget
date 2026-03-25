# Glossary

This glossary defines the solver-side terms used in the pose-free anisotropic experiments.

These terms describe the current engineering machinery.
They are not claims about how nature itself is organized.

## Backbone

The stable part of the recovered control object.

In the current layered solver program, the backbone is the normalized-geometry layer recovered before conditional `alpha` work begins.

## Bank

A finite reference library of sampled hypothetical solutions used by the current solver.

More concretely, a bank is:

- a list of sampled latent parameter tuples
- plus their precomputed signatures
- used as the search space for scoring one observation

Why this exists:

- the current solver does not search the continuous latent space directly
- it approximates that space with a finite reference set

Why it is random:

- random sampling gives broad coverage without hand-built directional bias
- different bank seeds let the repo test whether a solver is stable to the sampled reference set

Important:

- a bank is an engineering object in the current solver
- it is not a claim that nature is itself a random lookup table

## Bank Seed

The random seed used to generate one reference bank.

Changing the bank seed changes which sampled reference entries are available to the solver.

## Candidate

One entry inside a bank.

A candidate is one sampled hypothetical solution with:

- one latent parameter tuple
- one implied geometry / weight / `alpha` configuration
- one precomputed signature

When the solver scores an observation against a bank, each bank entry is a candidate.

## Best Candidate

The single bank entry with the lowest score for the current observation.

This is the winner-take-all output if no family aggregation is used.

## Candidate Family

The near-best subset of bank entries that all survive the score cutoff around the current best score.

This family is important because many candidates can fit the observed boundary almost equally well while still differing meaningfully in geometry or `alpha`.

## Candidate Diversity

The amount of internal variation inside the near-best candidate family.

In the current Layer 1 analysis, candidate diversity means:

- how many candidates survive the score band
- how much geometry spread remains inside that family
- how much `alpha` spread remains inside that family

High candidate diversity means the observation is compatible with many near-best solutions.

## Score Band

The tolerance around the best score used to keep the near-best candidate family.

This is how the solver decides which candidates are still plausible enough to remain in the family.

## Consensus

The aggregated estimate formed from the near-best candidate family instead of taking only the best candidate.

In Layer 1, the geometry consensus is built by averaging canonicalized geometry invariants across the family with score-decay weights.

## Collapse

The compression of a wide near-best candidate family into one narrower consensus estimate.

In plain language:

- many plausible candidates go in
- one stable-looking backbone estimate comes out

Collapse is not automatically bad.
It is often necessary to recover a stable backbone at all.

But collapse can hide how much internal candidate diversity had to be absorbed to produce that stable output.

## Compression Load

The hidden Layer 1 quantity that measures how much candidate diversity had to collapse to produce the final backbone consensus.

Current working ratio:

- `compression_load = family_geometry_span / consensus_geometry_bank_span`

In the current exported field names:

- `compression_load = mean_geometry_span_norm_set / consensus_geometry_bank_span_norm`

Interpretation:

- low compression load means the family was already fairly simple
- high compression load means the backbone looks simple only after heavy compression

## Backbone Tightness

How narrow or stable the recovered backbone looks across banks.

Important:

- backbone tightness alone is not the same thing as low compression load
- a tight backbone can be cheap to obtain or expensive to obtain

## Gate

The Layer 2 observability decision about whether a point `alpha` estimate should be trusted at all.

The gate is not the same thing as final `alpha` recovery.

## Gate-Open Trial

A trial where the Layer 2 gate allows point `alpha` recovery to be attempted.

Gate-open does not mean refine-by-default.

## Anchored Output

The Layer 2 point estimate produced after the geometry backbone is used to reweight the near-best family.

This is the protected lower-layer default for Layer 3 comparisons.

## Refined Output

The Layer 3 point estimate after additional conditional `alpha` refinement.

## Correction Flux

The average size of the anchored-to-refined move across banks.

This measures how much Layer 3 is trying to change the anchored answer.

## Correction Pressure

The current Layer 3 ratio:

- `correction_pressure = correction_sign_majority * (correction_flux / anchored_alpha_log_std)`

This measures how strong and coherent the proposed post-anchor correction is relative to anchored tolerance.

## Correction Transmission

A cross-layer ratio that adds Layer 1 compression load into the denominator:

- `correction_transmission = correction_sign_majority * (correction_flux / (anchored_alpha_log_std * compression_load))`

This is a working hidden-ratio hypothesis, not yet an established final solver law.

## Why These Terms Matter

The current solver challenge is not just to pick a better winner from a bank.

The layered program is trying to distinguish:

- a backbone that is genuinely simple
- from a backbone that only looks simple because many candidates were compressed into one consensus

That is why the glossary separates:

- backbone tightness
- candidate diversity
- collapse
- compression load
