# Validated Support + Joint Router

This experiment tests a small solver-design change on the benchmark packet from the current joint-solver comparison.

It does **not** change the forward model.
It does **not** change the latent control object.
It does **not** broaden into a theory rewrite.

The change is a narrow router:

- default to the current support-aware baseline
- allow a joint-solver rescue only when sparse-partial evidence crosses a calibrated gate

Validation is explicit.

Primary validation:
- leave-one-trial-out

Cross-check:
- leave-one-cell-out

Both validations give the same overall mean on this packet.

## Out-of-sample result

- validated overall mean alpha error: `0.1596356869551954`
- baseline overall mean alpha error: `0.1713940930372717`
- joint-solver overall mean alpha error: `0.1835302813296511`
- oracle best-of-two overall mean alpha error: `0.1281032160699384`

So the validated router beats the current out-of-sample benchmark of `0.1714`.

## Out-of-sample by condition

- `sparse_full_noisy`: `0.1233128000232719`
- `sparse_partial_high_noise`: `0.1959585738871189`

## Plain-language read

The validated gain comes from one place only:
the sparse-partial branch.

The router leaves `sparse_full_noisy` unchanged out-of-sample and recovers a small subset of the joint solver's complementary wins in `sparse_partial_high_noise`.

That strengthens the current BGP read.

This still looks like a solver bottleneck, not a control-object failure.
More specifically, the remaining bottleneck looks narrower now:
it is a sparse-partial rescue problem, not a broad failure across both focused conditions.
