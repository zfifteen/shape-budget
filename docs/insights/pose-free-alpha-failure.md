## The Budget Governor Principle Reveals an Observability Asymmetry: Pose-Free Failure of Alpha Is a Structural Orbit-Aliasing Phase Transition, Not a Solver Deficiency

When a system hides both rotation and anisotropy at once, the anisotropy parameter does not merely become harder to recover. It crosses into a qualitatively different identifiability regime, while geometric shape remains comparatively unaffected. This is not a matter of needing more search, more local refinement, or a better representation. It is a consequence of how close the anisotropy direction sits to the rotation symmetry orbit in the observation space.

The standard view in inverse problems is that harder latent variables need more powerful solvers. This framework says something structurally different: some latent variables sit geometrically near a symmetry orbit in observation space, and for those variables, solver improvements cannot close the gap until the symmetry is explicitly broken before encoding.

What is non-obvious is the selectivity. In the same pipeline, at the same level of incompleteness, normalized geometry stays nearly as recoverable as it was in the canonical case. Only alpha collapses. This selectivity is itself a diagnostic, and it tells you something about the observation representation, not about the underlying latent object.

The oracle ceiling experiment is the crucial indicator. Once true pose is given, alpha error drops by factors ranging from about five to thirteen, while geometry barely moves. That means the anisotropy signal is genuinely in the data. The loss is not a signal absence problem. It is a symmetry-induced aliasing problem that concentrates in one latent direction.

The tipping point is not just "incomplete observations" in the ordinary sense. The failure map shows it is structured: sparse or partial data combined with mid-range source skew is the critical failure zone. This means the phase boundary is parameterized by two properties simultaneously, support completeness and geometric skew, not by noise alone.

A practical consequence follows that most inverse-problem practitioners would not predict. Adding more samples within a sparse observation regime does not help, and neither does tightening the local refinement basin or compressing pose more aggressively. The only productive intervention is breaking the rotation symmetry before the boundary is encoded, but only in regimes where the observation is complete enough for the locking procedure itself to remain stable.

This implies a regime-conditional design rule: an alignment-first strategy is beneficial in the full-support and high-skew cells, actively harmful in the sparse low-skew cells, and unhelpful everywhere else. A solver that applies alignment uniformly will perform worse on average than one that applies it never, because the harmful cells outnumber the helpful ones when observations are incomplete.
