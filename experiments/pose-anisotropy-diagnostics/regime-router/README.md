# Regime Router Experiment

## Purpose

This experiment tests a sharper version of the current bottleneck hypothesis.

The working idea was:

> maybe the remaining pose-free anisotropic bottleneck is not “find a universally better alpha estimator,” but “route each trial to the right refinement policy before refinement starts.”

The candidate routing signals tested here are:

- a raw anisotropy-to-skew ratio from the top marginalized seed
- a support-aware orbit-alias index from the same top seed plus the actual observation mask

The two refinement policies are:

- fixed-family alpha-only search
- geometry-plus-alpha family-switching search

The experiment script is [run_regime_router_experiment.py](run.py#L1).

## Pre-Benchmark Logic Audit

Before the benchmark, the new script was checked explicitly.

Code sanity:

- script compiled cleanly

Support-aware alias-index invariance:

- audit cases: `20`
- max joint-shift delta: `1.3323e-15`

That matters because the new index is supposed to measure pose-related alias pressure. If jointly rotating the seed signature and the mask changed the index materially, the index itself would be suspect.

Router logic audit:

- leave-one-out toy accuracy: `1.0`
- leave-one-out toy mean routed alpha error: `0.1`

That is a direct audit of the threshold-and-direction selection logic used in the benchmark.

## Method

This is a focused pilot on the current hard branch.

Scope:

- observation regimes:
  - `sparse_full_noisy`
  - `sparse_partial_high_noise`
- anisotropy band:
  - `moderate`
- geometry-skew bins:
  - `low_skew`
  - `mid_skew`
  - `high_skew`

Trials:

- `4` trials per cell
- `24` trials total

For each trial:

1. take the top marginalized seed
2. compute two candidate router signals from that seed
3. run both refinement policies:
   - alpha-only fixed-family search
   - geometry-plus-alpha family switching
4. record which policy actually gives lower alpha error
5. evaluate leave-one-out threshold routers for each signal

The raw ratio is:

- anisotropy strength `|log(alpha_seed)|`
- divided by top-seed skew magnitude `|t_seed|`

The support-aware orbit-alias index is:

- how much alpha perturbations around the top seed can be absorbed by optimal shifts under the current mask
- divided by how visible small skew perturbations remain under the same mask

So the support-aware index tries to encode:

> hidden pose pressure relative to visible geometry anchor strength

## Main Result

The result is informative, but it does not validate the simplest scalar-router design.

> Neither scalar router beats the fixed alpha-only policy overall. The raw anisotropy-to-skew ratio carries some useful signal and tracks the sparse-full branch surprisingly well, but the current support-aware alias index does not improve routing enough to beat the simple baseline. So the routing idea has legs, but not yet in one-scalar form.

The summary file is [regime_router_summary.json](outputs/regime_router_summary.json).

Overall means:

- alpha-only fixed-family search: `0.1648`
- geometry-plus-alpha family switching: `0.1678`
- raw-ratio router: `0.1692`
- support-aware alias router: `0.2038`
- oracle router over the two methods: `0.1193`

Overall leave-one-out routing accuracy:

- raw-ratio router: `0.4167`
- support-aware router: `0.3750`

So the direct practical answer is clear:

> in this pilot, neither scalar router is good enough yet to replace the current fixed alpha-only default.

## By Condition

### Sparse full, moderate

This branch establishes a real part of the routing idea.

- alpha-only: `0.1565`
- geometry-plus-alpha: `0.1101`
- raw-ratio router: `0.1104`
- support-aware router: `0.1861`
- oracle router over the two methods: `0.0772`

This is the strongest positive surprise in the experiment.

The raw ratio almost matches the better geometry-plus-alpha policy here.

So in the sparse-full branch:

> the simple ratio already contains a real clue about when geometry freedom helps.

### Sparse partial, moderate

This branch behaves the other way.

- alpha-only: `0.1730`
- geometry-plus-alpha: `0.2254`
- raw-ratio router: `0.2281`
- support-aware router: `0.2214`
- oracle router over the two methods: `0.1614`

Here the fixed alpha-only policy is still the best practical choice.

Both routers lose ground relative to the alpha-only baseline.

So in the sparse-partial branch:

> neither scalar signal is strong enough yet to suppress harmful geometry motion reliably.

## Cell-Level Pattern

The cell summaries show why the scalar routing rule is not complete.

### Sparse full

- `low_skew`
  - alpha-only: `0.1073`
  - family switch: `0.1043`
  - raw router: `0.1043`
  - support-aware router: `0.1043`

- `mid_skew`
  - alpha-only: `0.2247`
  - family switch: `0.2088`
  - raw router: `0.2088`
  - support-aware router: `0.3128`

- `high_skew`
  - alpha-only: `0.1376`
  - family switch: `0.0172`
  - raw router: `0.0179`
  - support-aware router: `0.1411`

The raw router is very close to the right answer in every sparse-full cell.

The current support-aware index is not.

### Sparse partial

- `low_skew`
  - alpha-only: `0.2011`
  - family switch: `0.2474`
  - raw router: `0.2695`
  - support-aware router: `0.2474`

- `mid_skew`
  - alpha-only: `0.2022`
  - family switch: `0.2163`
  - raw router: `0.1988`
  - support-aware router: `0.2022`

- `high_skew`
  - alpha-only: `0.1159`
  - family switch: `0.2124`
  - raw router: `0.2159`
  - support-aware router: `0.2147`

This is the branch where the scalar-router idea still falls short.

The raw ratio keeps over-choosing geometry freedom, and the current support-aware index does not suppress that strongly enough.

## What This Means

This result does not kill the routing idea.

It sharpens it.

What it establishes:

- the bottleneck is a policy-routing problem rather than a single-estimator problem
- the raw ratio carries real signal in the sparse-full branch
- the current support-aware index is not the right compressed variable

What it does not establish:

- a single scalar based only on the top seed is already enough to route all moderate sparse cells correctly

The strongest reading is:

> support regime itself still matters too much to be compressed away by the current scalar router design. The sparse-full and sparse-partial branches remain qualitatively different even after adding the first support-aware alias metric.

That is the central outcome of the experiment.

## Strongest Clue

The most useful clue is this:

> the raw ratio nearly solves the sparse-full branch, while the sparse-partial branch resists both scalar routers.

That points to a stronger next question:

- not “find a better single scalar”
- but “can we build a two-stage router that first identifies support type or visible support geometry, then applies a scalar rule inside that branch?”

In other words:

- sparse-full is a ratio-routable branch
- sparse-partial needs an explicit support-aware gate before any scalar alias measure becomes useful

## Figures

Key figures:

- [regime_router_scatter.png](outputs/figures/regime_router_scatter.png)
- [regime_router_method_bars.png](outputs/figures/regime_router_method_bars.png)

The scatter is the right first look. It shows how each trial’s gain from geometry freedom relates to:

- the raw anisotropy-to-skew ratio
- the support-aware orbit-alias index

The bar figure then makes the practical outcome obvious:

- the raw router almost tracks the sparse-full winner
- neither router is good enough to beat alpha-only overall
