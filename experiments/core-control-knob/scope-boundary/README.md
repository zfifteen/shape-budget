# Scope Boundary Experiment

## Purpose

BGP is the latent control parameter `e = c / a` for normalized geometry in the symmetric constant-sum two-source Euclidean process.

This experiment maps the tested scope boundary directly:

- where the control object stays one-dimensional
- where it expands but stays compact
- where wrong compression leaves visible normalized family gaps
- where the current failure is a branch-specific inverse bottleneck rather than a general failure of BGP

The experiment script is [run_scope_boundary_experiment.py](run.py#L1).

## Pre-Benchmark Logic Audit

Before the synthesis benchmark, the new script was checked explicitly.

Code sanity:

- script compiled cleanly

Input audit:

- required summary files loaded: `13` of `13`

Scope-consistency audit:

- minimum wrong-to-right gap factor across explicit negative controls: `5.39e+05`
- minimum support-encoding pose selectivity: `4.88`
- minimum pose-free `alpha`-over-geometry penalty versus canonical anisotropic inversion: `3.85`

That matters because this experiment is a scope map built from established branch outputs. It only means anything if the loaded branch artifacts remain numerically separated in exactly the way the scope claims require. They do.

## Method

This is a synthesis experiment grounded in established branch summaries.

It does three things.

### 1. Compactness ladder

It records the smallest established compact control object for the main forward branches:

- symmetric ellipse
- hyperbola twin
- asymmetry
- raw anisotropy
- equal-weight three-source
- weighted three-source

### 2. Explicit negative controls

It measures three wrong-compression failures against the corrected compact object:

- asymmetry under `e`-only compression
- raw anisotropy under `e`-only compression
- weighted three-source under geometry-only compression

### 3. Operational and bottleneck summary

It pulls together the established inverse and theory-hardening results:

- known-source inverse usefulness
- weighted and anisotropic latent-object recovery
- representation-independence selectivity
- probe-specialization routing gains
- pose-free anisotropic penalty structure

## Main Result

The scope boundary is now clear.

> BGP stays compact across every tested positive branch, but it is not universally one scalar. The established map is: one scalar in the symmetric ellipse base case and the hyperbola twin; two parameters under asymmetry and raw anisotropy; about three parameters in equal-weight three-source; about five parameters in weighted three-source; operational inverse usefulness persists into weighted and anisotropic branches; and the main current scope limit is the selective `alpha` bottleneck in the pose-free anisotropic inverse rather than a general failure of compact budget-governed structure.

This is the disciplined scope map the theory-hardening triage called for.

## Part 1: Where BGP Stays One-Knob

The one-knob result is established in two tested branches.

Symmetric ellipse:

- control object: `e = c / a`
- max normalized collapse error: `3.9736e-08`
- boundary-space PC1 explained variance ratio: `0.9891`
- nonlinear 1D ordering recovery: `|rho| = 1.0`

Hyperbola twin:

- control object: `lambda = a / c`
- max normalized collapse error: `3.9736e-08`
- normalized openness scale spread: `2.2204e-16`

So the repo now has two exact one-knob branches, not just one.

## Part 2: Where One-Knob Compression Fails But Compactness Survives

The scope boundary is not vague. The wrong control object leaves visible family gaps.

Asymmetry:

- wrong compression floor under `e` only: `0.0200`
- corrected `(e, w)` collapse error: `3.7196e-08`
- gap factor: `5.39e+05`

Raw anisotropy:

- wrong compression floor under raw `e` only: `0.05694`
- corrected `(e, alpha)` collapse error: `7.9473e-08`
- whitened `e` collapse error: `3.9736e-08`
- strongest gap factor: `1.43e+06`

Weighted three-source:

- wrong compression floor under geometry-only compression: `0.01948`
- corrected geometry-plus-weights collapse error: `2.4476e-12`
- gap factor: `7.96e+09`

So the stronger claim

> one normalized scalar should compress every tested branch

is now explicitly falsified.

The tested replacement claim is stronger and cleaner:

> BGP is a compact-control-object principle. When one knob fails, the failure is structured and the corrected compact object restores collapse.

## Part 3: How Large The Compact Object Becomes

The broad equal-weight three-source family is about three-parameter:

- max boundary collapse error: `2.3333e-13`
- random-family PC3 cumulative explained variance ratio: `0.9901`
- equilateral-slice PC1 explained variance ratio: `0.9957`

The broad weighted three-source family is about five-parameter:

- max boundary collapse error: `2.4476e-12`
- random weighted-family PC5 cumulative explained variance ratio: `0.99996`
- equilateral weighted-slice PC2 cumulative explained variance ratio: `0.9988`

So the tested dimensional ladder is now:

- `1` in the symmetric ellipse base case
- `1` in the hyperbola twin
- `2` under asymmetry
- `2` under raw anisotropy
- `3` in equal-weight three-source
- `5` in weighted three-source

That is a concrete scope statement, not a vague “low-dimensional” slogan.

## Part 4: Operational Scope And The Current Limit

The compact object remains operational in the tested inverse branches.

Symmetric known-source inverse:

- worst 95th-percentile `e` error: `0.01262`
- scale-generalization advantage of `e` over raw `(d, S)`: `83.4x` to `2947.8x`

Weighted multisource inverse:

- fit improvement over equal-weight baseline: `1.88x` to `5.53x`

Weighted anisotropic inverse:

- fit improvement over Euclidean baseline: `7.61x` to `14.04x`

Representation independence:

- support-encoding selective pose penalty: `4.88x` to `11.96x`
- support fit-improvement factor: `3.93x` to `16.76x`

Probe specialization:

- router beats the best fixed practical probe in `3` of `4` regimes
- best-fixed over router factor: `0.987x` to `1.042x`

The current limit is narrower than a general theory failure.

Pose-free weighted anisotropic inverse:

- fit improvement over Euclidean baseline still: `3.78x` to `15.93x`
- geometry penalty versus canonical anisotropic inverse: `0.917x` to `1.238x`
- `alpha` penalty: `4.77x` to `14.39x`
- `alpha` over geometry penalty: `3.85x` to `15.69x`

So the current hard boundary is:

> under hidden pose in the anisotropic branch, compact latent structure survives, but the anisotropy coordinate becomes selectively weak.

That is a branch-specific inverse limit, not a collapse of the broader compact-control-object result.

## Interpretation

This experiment upgrades the repo’s scope statement.

What is now established:

- BGP is exactly one-knob in the symmetric ellipse base case and in the hyperbola twin
- the universal-one-scalar version is false in the tested richer branches
- the compact-control-object version is supported in every tested positive branch
- operational usefulness survives into inverse and measurement-strategy settings
- the main current theory boundary is not “compactness disappears” but “pose-free anisotropy makes one latent direction selectively fragile”

That is the falsification structure the program needed.

## Why This Matters For Priority

`SCOPE_BOUNDARY_EXPERIMENT` is now complete.

The theory-hardening side now has:

- representation robustness
- operational usefulness
- explicit scope boundaries with negative controls

So the repo’s main ambiguity is lower than it was before this experiment.

The next high-value choices are now clearer:

- return to the pose-free anisotropy bottleneck as a targeted branch problem
- or move outward to harder scope tests such as unknown anisotropy-axis orientation, richer media, or out-of-family application checks

## Figures

- [scope_boundary_compactness_ladder.png](outputs/figures/scope_boundary_compactness_ladder.png)
- [scope_boundary_negative_controls.png](outputs/figures/scope_boundary_negative_controls.png)
- [scope_boundary_bottleneck_map.png](outputs/figures/scope_boundary_bottleneck_map.png)

The clearest figures are:

- [scope_boundary_compactness_ladder.png](outputs/figures/scope_boundary_compactness_ladder.png) for the `1 -> 2 -> 3 -> 5` control-object expansion
- [scope_boundary_negative_controls.png](outputs/figures/scope_boundary_negative_controls.png) for the explicit wrong-compression failures and corrected collapse

## Artifacts

Data:

- [scope_boundary_branch_summary.csv](outputs/scope_boundary_branch_summary.csv)
- [scope_boundary_negative_controls.csv](outputs/scope_boundary_negative_controls.csv)
- [scope_boundary_operational_summary.csv](outputs/scope_boundary_operational_summary.csv)
- [scope_boundary_representation_penalties.csv](outputs/scope_boundary_representation_penalties.csv)
- [scope_boundary_posefree_penalties.csv](outputs/scope_boundary_posefree_penalties.csv)
- [scope_boundary_summary.json](outputs/scope_boundary_summary.json)

Code:

- [run_scope_boundary_experiment.py](run.py#L1)
