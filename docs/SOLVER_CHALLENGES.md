# Solver Challenges

BGP is the latent control parameter `e = c / a` for normalized geometry in the symmetric constant-sum two-source Euclidean process. The repo establishes compact corrected control objects across the tested positive branches, and it now has a working focused solver on the hardest tested pose-free anisotropic slice. This document is for external developers who want to extend that solver result beyond the solved slice.

For solver vocabulary used in this document, see [Glossary](./GLOSSARY.md). For current bank-design notes, see [Intelligent Bank Design](./INTELLIGENT-BANK-DESIGN.md) and [Candidate Atlas Instrumentation](./CANDIDATE-ATLAS-INSTRUMENTATION.md).

## Established Scope

The repo establishes the following points already.

- The base case is established in the strongest tested scope. See [technical note](../technical-note/technical_note.md).
- The weighted anisotropic canonical-pose inverse jointly recovers normalized geometry, normalized weights, and `alpha` from boundary data. See [weighted anisotropic inverse](../experiments/multisource-control-objects/weighted-anisotropic-inverse/README.md).
- The hard pose-free `alpha` recovery challenge is not a radial-signature artifact. The representation swap preserves the core inferential result while the same selective `alpha` fragility remains. See [representation independence](../experiments/core-control-knob/representation-independence/README.md).
- The scope map is established: BGP stays compact across every tested positive branch. See [scope boundary](../experiments/core-control-knob/scope-boundary/README.md).
- The focused pose-free anisotropic bottleneck is now solved in the tested regime by the [entropy-gated bank ensemble solver](../experiments/pose-anisotropy-interventions/entropy-gated-bank-ensemble-solver/README.md).

The practical conclusion is narrow and important:

- the latent object still looks real
- the focused bottleneck yielded to a solver-policy fix
- the remaining open work is broader validation and broader extension, not the solved slice itself

## Focused Solver Milestone

The solved slice is:

- `sparse_full_noisy`
- `sparse_partial_high_noise`
- moderate anisotropy
- `low_skew`, `mid_skew`, `high_skew`

The working solver is the [entropy-gated bank ensemble solver](../experiments/pose-anisotropy-interventions/entropy-gated-bank-ensemble-solver/README.md).

Its tested result is:

- holdout mean `alpha` error `0.1050` vs best single cached candidate `0.1091`
- confirmation mean `alpha` error `0.1064` vs best single cached candidate `0.1104`

This is a solver-policy resolution in the tested slice. It is not a blanket claim that the entire anisotropic solver stack is finished.

## Historical Narrowing Steps

Several earlier interventions narrowed the problem before the working solver arrived.

- A pure joint objective was not enough by itself. The first-principles joint solver did not beat the support-aware baseline overall. See [joint pose-marginalized solver](../experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/README.md).
- The residual problem split by support type. Geometry freedom helped the sparse-full branch and often hurt the sparse-partial branch. See [family switching refinement](../experiments/pose-anisotropy-interventions/family-switching-refinement/README.md).
- Simple scalar routing was not enough. Useful routing signal existed, but one-scalar routers did not beat the fixed policy overall. See [regime router](../experiments/pose-anisotropy-diagnostics/regime-router/README.md).
- Direct score competition was not enough in sparse-partial. The competitive hybrid helped, but simple competition still misrouted too many sparse-partial cases. See [competitive hybrid resolver](../experiments/pose-anisotropy-interventions/competitive-hybrid-resolver/README.md).
- The bank-adaptive chooser showed real signal but still failed fresh-bank confirmation under one solver family. See [bank-adaptive solver](../experiments/pose-anisotropy-interventions/bank-adaptive-solver/run.py).

Those steps matter because they show what the current milestone actually resolved: the old bottleneck was narrowed to solver policy and observability handling before it was closed in the focused tested slice.

## What Remains Open

The remaining open work is no longer “close the focused bottleneck at all.” It is:

- broader regime coverage outside the solved moderate sparse slice
- broader fresh-bank validation outside the solved slice
- unknown anisotropy-axis orientation
- richer non-quadratic or spatially varying media
- outward extension to harder source families and application tests

## Extension Rules

If you extend the solver stack, preserve these rules.

- Keep the forward model fixed unless the task explicitly changes the medium model.
- Keep the latent control object fixed unless the task explicitly targets a new branch that requires a different object.
- Use disjoint calibration vs evaluation for any chooser or routing policy.
- Freeze all chooser weights before touching holdout evaluation.
- Do not route by regime label alone at evaluation time.
- Do not claim a new solver milestone from same-packet or same-bank wins alone.

The preferred construction rule remains layered:

1. recover the stable control backbone first
2. validate that backbone recovery stays stable
3. add an observability layer for extension coordinates
4. add conditional extension recovery only where the observation supports it

## Useful Starting Points

- [entropy-gated bank ensemble solver](../experiments/pose-anisotropy-interventions/entropy-gated-bank-ensemble-solver/README.md)
- [bank-adaptive solver driver](../experiments/pose-anisotropy-interventions/bank-adaptive-solver/run.py)
- [joint pose-marginalized solver](../experiments/pose-anisotropy-interventions/joint-pose-marginalized-solver/run.py)
- [family switching refinement](../experiments/pose-anisotropy-interventions/family-switching-refinement/run.py)
- [candidate-conditioned alignment](../experiments/pose-anisotropy-interventions/candidate-conditioned-alignment/run.py)
- [regime router README](../experiments/pose-anisotropy-diagnostics/regime-router/README.md)
- [candidate atlas instrumentation](../experiments/pose-anisotropy-diagnostics/candidate-atlas-instrumentation/README.md)

## Practical Developer Read

The control object is still the right theory object. The remaining solver work is an operational inverse extension problem.

External solver work is most likely to help if it improves one of these failure modes:

- broader holdout stability outside the solved slice
- support-aware routing that stays reliable under wider regime variation
- candidate generation that is less sensitive to the sampled reference bank
- solver policies that survive broader fresh-bank confirmation

The repo does not need another same-packet heuristic victory on the old focused slice. It needs broader-scope extensions that preserve the current working-solver standard.
