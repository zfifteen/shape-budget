# Competitive Hybrid Resolver

This experiment targets the current bottleneck directly.

The question is simple:

1. run the two strongest existing refiners from the same top marginalized seeds
2. let them compete on the same marginalized score
3. see whether that resolves the sparse moderate pose-free anisotropy bottleneck

The script is [run.py](run.py#L1).

## Resolver paths

The two competing paths are:

- fixed-family candidate-conditioned `shift + alpha` search
- geometry-plus-alpha family switching

The first resolver tested here is a direct score-competitive hybrid:

- choose the refined candidate with the lower marginalized score
- if the scores tie, keep the fixed-family path

That resolver is implemented in [run.py](run.py#L1) and summarized in:

- [competitive_hybrid_resolver_summary.json](outputs/competitive_hybrid_resolver_summary.json)
- [competitive_hybrid_resolver_summary.csv](outputs/competitive_hybrid_resolver_summary.csv)
- [competitive_hybrid_resolver_cells.csv](outputs/competitive_hybrid_resolver_cells.csv)
- [competitive_hybrid_resolver_trials.csv](outputs/competitive_hybrid_resolver_trials.csv)

The key figures are:

- [competitive_hybrid_resolver_overview.png](outputs/figures/competitive_hybrid_resolver_overview.png)
- [competitive_hybrid_resolver_focus.png](outputs/figures/competitive_hybrid_resolver_focus.png)

## Main result

The direct score-competitive hybrid is **not** the full fix.

It nearly solves the `sparse_full_noisy` moderate branch:

- marginalized alpha error: `0.1244`
- fixed-family alpha error: `0.1565`
- family-switch alpha error: `0.1101`
- competitive hybrid alpha error: `0.0775`
- oracle best-of-two alpha error: `0.0772`

So in `sparse_full_noisy`, direct competition between the two paths works extremely well.

But it fails in `sparse_partial_high_noise`:

- marginalized alpha error: `0.2676`
- fixed-family alpha error: `0.1730`
- family-switch alpha error: `0.2254`
- competitive hybrid alpha error: `0.2230`
- oracle best-of-two alpha error: `0.1614`

The failure mode is clear in the trial outputs:

- the competitive hybrid chooses the family-switch path `91.7%` of the time in `sparse_partial_high_noise`
- the oracle best-of-two would only choose that path `33.3%` of the time

So score competition is a strong resolver in sparse-full, but an unreliable selector in sparse-partial.

## Bottleneck fix

The bottleneck slice is resolved much better by a **support-aware hard gate** using the already-computed paths:

- `sparse_full_noisy` -> use the competitive hybrid
- `sparse_partial_high_noise` -> use the fixed-family candidate-conditioned path

That post-run derivation is recorded in:

- [support_gated_derivation.json](outputs/support_gated_derivation.json)

Its results are:

- `sparse_full_noisy`: alpha error `0.0775`
- `sparse_partial_high_noise`: alpha error `0.1730`
- overall focused mean alpha error: `0.1253`

For comparison:

- direct competitive hybrid overall: `0.1502`
- oracle best-of-two overall: `0.1193`

So the support-aware gate closes most of the remaining gap without inventing a new latent object or a larger theory program.

## Interpretation

The current bottleneck is not that both refinement paths fail.

The current bottleneck is that:

- sparse-full wants geometry freedom
- sparse-partial wants fixed-family `shift + alpha` cleanup
- direct score competition alone misroutes too many sparse-partial cases

In plain language:

- the hard branch is now mostly a **policy-selection problem**
- not a collapse of the anisotropic control object
- and not a general failure of BGP

## Audit

The most important audit in [competitive_hybrid_resolver_summary.json](outputs/competitive_hybrid_resolver_summary.json) is:

- `max_conditioned_vs_family_score_delta_same_params = 0.0`

So the two competing paths really are being compared on the same score scale.

## Takeaway

The strongest practical resolver found here is:

- keep the competitive hybrid in sparse-full
- suppress family switching in sparse-partial

That is the smallest fix in the repo so far that materially reduces the current bottleneck.
