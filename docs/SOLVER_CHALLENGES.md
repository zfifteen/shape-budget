# Solver Challenges

BGP is the latent control parameter `e = c / a` for normalized geometry in the symmetric constant-sum two-source Euclidean process. The repo establishes compact corrected control objects across the tested positive branches, and the current solver challenge is the focused pose-free anisotropic inverse under hidden rotation, not a general failure of compact budget-governed structure.

This document is for external developers who want to resolve the remaining solver-design issue.

For solver vocabulary used in this document, see [Glossary](./GLOSSARY.md).
For current bank-design notes, see [Intelligent Bank Design](./INTELLIGENT-BANK-DESIGN.md) and [Candidate Atlas Instrumentation](./CANDIDATE-ATLAS-INSTRUMENTATION.md).

## Established Scope

The repo establishes the following points already.

- The base case is established in the strongest tested scope. See [technical note](../technical-note/technical_note.md).
- The weighted anisotropic canonical-pose inverse jointly recovers normalized geometry, normalized weights, and `alpha` from boundary data. See [weighted anisotropic inverse](../experiments/multisource-control-objects/weighted-anisotropic-inverse/README.md).
- The hard pose-free `alpha` recovery challenge is not a radial-signature artifact. The representation swap preserves the core inferential result while the same selective `alpha` fragility remains. See [representation independence](../experiments/core-control-knob/representation-independence/README.md).
- The scope map is established: BGP stays compact across every tested positive branch, while the main current limit is the selective pose-free anisotropic `alpha` recovery challenge. See [scope boundary](../experiments/core-control-knob/scope-boundary/README.md).

The practical conclusion is narrow and important:

- the latent object still looks real
- the current failure is solver-side
- the hard case is hidden rotation plus sparse support plus bank-sensitive routing

## Focused Solver Problem

The focused slice is:

- `sparse_full_noisy`
- `sparse_partial_high_noise`
- moderate anisotropy
- `low_skew`, `mid_skew`, `high_skew`

The current solver challenge is not “find a universally better alpha estimator.” The challenge is to build a solver or routing policy that stays reliable across fresh bank seeds in this slice.

## Layered Construction Rule

The next solver should mirror the control structure that the repo has already established.

Use this construction rule:

- build the solver backbone-first, not full-latent-first
- validate each capability layer before adding the next one
- do not force point recovery of every latent coordinate at the first stage

The default layered plan is:

1. recover the stable control backbone first
2. validate that backbone recovery stays stable across fresh bank seeds
3. add an observability or ambiguity layer for extension coordinates
4. add conditional extension recovery only where the observation supports it
5. only then attempt a full confirmation-stable solver policy

In practical terms, new solver work should begin by asking:

- what part of the control object is already stable in this observation class
- which coordinates become ambiguous under hidden rotation and sparse support
- whether the next capability layer is actually validated before the following layer is added

This repo should not default to monolithic full-latent search-and-route designs when the evidence suggests a layered observability structure instead.

## What The Repo Already Rules Out

Several interventions already narrowed the problem.

- A pure joint objective is not enough by itself. The first-principles joint solver does not beat the current support-aware baseline overall. See [joint pose-marginalized solver](../experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/README.md).
- The solver challenge splits by support type. Geometry freedom helps the sparse-full branch and often hurts the sparse-partial branch. See [family switching refinement](../experiments/pose-anisotropy-interventions/family-switching-refinement/README.md).
- Simple scalar routing is not enough. Useful routing signal exists, but one-scalar routers do not beat the fixed policy overall. See [regime router](../experiments/pose-anisotropy-diagnostics/regime-router/README.md).
- Direct score competition is not enough in sparse-partial. The competitive hybrid helps, but simple competition misroutes too many sparse-partial cases. See [competitive hybrid resolver](../experiments/pose-anisotropy-interventions/competitive-hybrid-resolver/README.md).

The repeated pattern is consistent:

- sparse-full often wants more geometry freedom
- sparse-partial often wants more conservative fixed-family cleanup
- routing signal exists, but the current compressed policies are not stable enough

## Latest Bank-Adaptive Attempt

The newest focused experiment is [bank-adaptive solver](../experiments/pose-anisotropy-interventions/bank-adaptive-solver/run.py).

Its design was:

- cache the two real competing candidates once per trial
- fit exactly one ridge chooser on cached calibration blocks only
- freeze the chooser before holdout
- require disjoint holdout success
- require fresh-bank confirmation without recalibration
- stop after one density fallback branch if confirmation still fails

The main output bundle is [full plan result](../experiments/pose-anisotropy-interventions/bank-adaptive-solver/outputs/reports/density_ablation__full_plan_result.json).

### Baseline Router

Baseline settings:

- reference bank size `300`
- `TOP_K_SEEDS = 3`

Results:

| Split | Support | Joint | Chooser | Outcome |
| --- | ---: | ---: | ---: | --- |
| Calibration | `0.1862` | `0.1596` | `0.1479` | beats both |
| Holdout block 1 | `0.1273` | `0.1180` | `0.1223` | beats support, loses to joint |

Relevant report:

- [baseline ladder summary](../experiments/pose-anisotropy-interventions/bank-adaptive-solver/outputs/reports/baseline__ladder_summary.json)

This variant failed the acceptance rule at holdout.

### Density Fallback

Fallback settings:

- reference bank size `600`
- `TOP_K_SEEDS = 5`

Results:

| Split | Support | Joint | Chooser | Outcome |
| --- | ---: | ---: | ---: | --- |
| Calibration | `0.1696` | `0.1687` | `0.1294` | beats both |
| Holdout block 1 | `0.1091` | `0.1356` | `0.0969` | beats both |
| Confirmation block | `0.1475` | `0.1104` | `0.1435` | beats support, loses to joint |

Relevant report:

- [density fallback ladder summary](../experiments/pose-anisotropy-interventions/bank-adaptive-solver/outputs/reports/density_ablation__ladder_summary.json)

This variant cleared holdout and then failed fresh-bank confirmation.

## What The Latest Result Means

The latest result says something precise.

- The router signal is real.
- The router can beat both candidates on calibration.
- The router can even beat both candidates on a disjoint holdout under a denser bank.
- But the router is still bank-sensitive.
- The same frozen chooser does not stay best on a fresh bank seed.

That is why the correct stop-condition summary is:

> current support-vs-joint routing is still too bank-sensitive for a reliable solution under this solver family

The bank-adaptive attempt narrows the remaining solver challenge. It does not weaken the broader BGP theory.

## What A Working Solver Must Do

A solver counts as working only if it does all of the following.

- Beats the current support-aware baseline on holdout block 1.
- Beats the uniform joint solver on holdout block 1.
- Beats both again on a fresh-bank confirmation block with no recalibration.
- Reports per-condition results for `sparse_full_noisy` and `sparse_partial_high_noise`.
- Reports per-cell results for all 6 focused cells.

In practice, this means a candidate solver must be confirmation-stable, not just calibration-good.

## Hard Constraints For New Work

Please keep new solver work inside a new experiment folder and preserve these evaluation rules.

- Use disjoint calibration vs holdout.
- Freeze all chooser weights before touching holdout evaluation.
- Do not use true latent errors at evaluation time.
- Do not route by regime label alone.
- Do not rely on copied outputs from older experiments as a substitute for recomputation.
- If a method fails holdout, stop or move to one explicitly defined fallback branch. Do not keep tuning heuristics on the same holdout.

## Useful Starting Points

If you are extending the solver stack, these are the best starting references.

- [backbone consensus solver](../experiments/pose-anisotropy-interventions/backbone-consensus-solver/README.md)
- [backbone observability gate](../experiments/pose-anisotropy-interventions/backbone-observability-gate/README.md)
- [backbone conditional alpha solver](../experiments/pose-anisotropy-interventions/backbone-conditional-alpha-solver/README.md)
- [backbone correction-flux-triggered alpha solver](../experiments/pose-anisotropy-interventions/backbone-correction-flux-triggered-alpha-solver/README.md)
- [backbone correction-pressure-triggered alpha solver](../experiments/pose-anisotropy-interventions/backbone-correction-pressure-triggered-alpha-solver/README.md)
- [persistent-mode informed bank](../experiments/pose-anisotropy-interventions/persistent-mode-informed-bank/README.md)
- [candidate atlas instrumentation](../experiments/pose-anisotropy-diagnostics/candidate-atlas-instrumentation/README.md)
- [candidate atlas pattern mining](../experiments/pose-anisotropy-diagnostics/candidate-atlas-pattern-mining/README.md)
- [bank-adaptive solver driver](../experiments/pose-anisotropy-interventions/bank-adaptive-solver/run.py)
- [joint pose-marginalized solver](../experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py)
- [family switching refinement](../experiments/pose-anisotropy-interventions/family-switching-refinement/run.py)
- [candidate-conditioned alignment](../experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py)
- [regime router README](../experiments/pose-anisotropy-diagnostics/regime-router/README.md)
- [competitive hybrid README](../experiments/pose-anisotropy-interventions/competitive-hybrid-resolver/README.md)

## Practical Developer Read

The control object is still the right theory object. The current solver challenge is an operational inverse problem.

External solver work is most likely to help if it improves one of these failure modes:

- bank-invariant candidate ranking
- support-aware routing that is richer than one scalar
- better uncertainty estimates for when geometry freedom is safe
- candidate generation that is less sensitive to the sampled reference bank
- solver policies that survive fresh-bank confirmation instead of only same-bank or same-packet wins

The repo does not need another same-packet heuristic victory. It needs a confirmation-stable solver on the focused pose-free anisotropic slice.
