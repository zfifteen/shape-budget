# Layer 1 Hidden Ratios

## Core Claim

Layer 1 is not only recovering a stable backbone.
It is also measuring how much candidate diversity had to collapse to make that backbone look stable.

The hidden Layer 1 state is therefore not just:

- backbone estimate
- backbone tightness

It is also:

- compression load

That changes the read of the stack.

A tight backbone can mean two different things:

- the observation was genuinely simple
- the solver had to compress a wide family into one narrow consensus

Those should not be treated the same downstream.

## Working Ratio

The strongest current Layer 1 hidden ratio is:

- `compression_load = family_geometry_span / consensus_geometry_bank_span`

Using the current field names:

- `compression_load = mean_geometry_span_norm_set / consensus_geometry_bank_span_norm`

This is dimensionless.

It measures how much within-family geometry spread had to collapse to produce the final cross-bank backbone consensus.

Low `compression_load` means:

- the family was already fairly simple relative to the recovered backbone

High `compression_load` means:

- many geometrically distinct near-best candidates were crushed into one stable-looking backbone

That is why a tight consensus is not enough by itself.

## Cross-Layer Consequence

The Layer 3 correction trigger should not read Layer 1 tightness as simple confidence.

The better cross-layer object is:

- `correction_transmission = correction_sign_majority * (correction_flux / (anchored_alpha_log_std * compression_load))`

This is also dimensionless.

In `Z = A(B / C)` form:

- `A = correction_sign_majority`
- `B = correction_flux`
- `C = anchored_alpha_log_std * compression_load`

Interpretation:

- `correction_flux` is the proposed post-anchor move
- `anchored_alpha_log_std` is Layer 2 anchored tolerance
- `compression_load` is Layer 1 structural resistance

So this is not just correction pressure.
It is correction pressure after accounting for how expensive the backbone was to obtain.

## Current Evidence

The current cached-output probe supports the compression-load reading.

### 1. Compression load separates the two support branches

Mean `compression_load` on gate-open fresh-block trials:

| Split | `sparse_full_noisy` | `sparse_partial_high_noise` |
| --- | ---: | ---: |
| Holdout | `8.4423` | `16.1059` |
| Confirmation | `7.5451` | `17.1693` |

The harder sparse-partial branch is carrying about twice the Layer 1 load.

### 2. Compression load predicts downstream correction demand

Correlation of `compression_load` with downstream Layer 3 quantities on gate-open fresh-block trials:

| Split | `correction_flux` | `refined_alpha_bank_log_span` |
| --- | ---: | ---: |
| Holdout | `0.4763` | `0.5080` |
| Confirmation | `0.7587` | `0.4205` |

So higher Layer 1 load is associated with:

- bigger correction moves
- wider refined bank spread

That is exactly what the hidden-ratio view predicts.

### 3. Collapse belongs in the denominator more than the numerator

The current official Layer 3 pressure trigger is:

- `pressure = correction_sign_majority * (correction_flux / anchored_alpha_log_std)`

On fresh gate-open blocks:

| Trigger | Holdout | Confirmation |
| --- | ---: | ---: |
| `pressure` | `0.15839` | `0.11720` |
| `pressure / compression_load` | `0.15916` | `0.11603` |

This is not yet a finished win.
But it is a useful signal:

- putting Layer 1 load in the denominator slightly helps confirmation
- multiplying by collapse terms was worse

So `compression_load` currently behaves more like resistance than like confidence.

## What This Changes

Layer 1 should no longer be treated as a pure estimator layer.

Its real outputs are:

1. `backbone estimate`
2. `backbone span`
3. `compression load`

That means the stack becomes:

1. Layer 1 exports backbone and collapse state
2. Layer 2 exports anchored tolerance
3. Layer 3 activates only when correction survives both tolerance and load

This is a better fit to the current behavior than the older reading:

- tight backbone means go ahead

The updated reading is:

- tight backbone can be cheap or expensive
- expensive backbone should make later correction harder to trust

## Testable Predictions

### P1. Matched-tightness divergence

If two trials have similar `consensus_geometry_bank_span_norm` but different `compression_load`, the higher-load trial should show:

- larger `correction_flux`
- larger `refined_alpha_bank_log_span`
- weaker default case for refinement

### P2. Support labels should compress under load

Some of the current support-type split should weaken if fresh-block trials are grouped by `compression_load` instead of by support label alone.

### P3. Layer 1 should export load explicitly

A Layer 1 redesign that exports `compression_load` as a first-class state variable should outperform a design that exports only consensus geometry and consensus span.

## Decision Rule

The next Layer 1 redesign should export at least:

- `mean_geometry_span_norm_set`
- `consensus_geometry_bank_span_norm`
- `compression_load`
- `best_geometry_bank_span_norm`
- `mean_candidate_count`
- `mean_ambiguity_ratio`

The next Layer 3 ratio sweep should treat `compression_load` as a denominator candidate, not as a direct confidence score.

## Status

This note is a cross-layer working hypothesis with direct support from the current cached-output probes.

What is already supported:

- Layer 1 hides a structural load variable beyond consensus tightness
- that load variable tracks downstream correction demand
- it behaves more like resistance than permission

What is not established yet:

- that `compression_load` is the final Layer 1 control ratio
- that the transmission ratio above is already the best deployable Layer 3 control law
- that this alone resolves the full confirmation-stable solver challenge
