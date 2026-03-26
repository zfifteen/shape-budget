## Historical Note: Pre-Resolution Read of Pose-Free Alpha Failure as an Observability Asymmetry

This memo predates the entropy-gated bank ensemble solver milestone. It records the earlier mechanism read before the focused moderate sparse slice was solved in the tested regime. Read it as historical mechanism context, not as the current repo-level conclusion about what solver work can or cannot achieve.

When a system hides both rotation and anisotropy at once, the anisotropy parameter does not merely become harder to recover. It crosses into a qualitatively different identifiability regime, while geometric shape remains comparatively unaffected. At the stage when this memo was written, the evidence suggested that simple search expansion, local refinement, or a representation swap were not enough by themselves. The dominant mechanism still looked like the closeness of the anisotropy direction to the rotation symmetry orbit in the observation space.

The standard view in inverse problems is that harder latent variables need more powerful solvers. This framework suggested something structurally different: some latent variables sit geometrically near a symmetry orbit in observation space, and for those variables, naive solver improvements do not close the gap until the symmetry issue is handled in a more targeted way.

What is non-obvious is the selectivity. In the same pipeline, at the same level of incompleteness, normalized geometry stays nearly as recoverable as it was in the canonical case. Only alpha collapses. This selectivity is itself a diagnostic, and it tells you something about the observation representation, not about the underlying latent object.

The oracle ceiling experiment is the crucial indicator. Once true pose is given, alpha error drops by factors ranging from about five to thirteen, while geometry barely moves. That means the anisotropy signal is genuinely in the data. The loss is not a signal absence problem. It is a symmetry-induced aliasing problem that concentrates in one latent direction.

The tipping point is not just "incomplete observations" in the ordinary sense. The failure map shows it is structured: sparse or partial data combined with mid-range source skew is the critical failure zone. This means the phase boundary is parameterized by two properties simultaneously, support completeness and geometric skew, not by noise alone.

A practical consequence followed that most inverse-problem practitioners would not predict. Adding more samples within a sparse observation regime did not help in the early interventions, and neither did tightening the local refinement basin or compressing pose more aggressively. At that stage, the productive intervention space still looked like breaking the rotation symmetry before or during encoding, but only in regimes where the locking procedure itself remained stable enough to help.

This implies a regime-conditional design rule from that earlier phase: an alignment-first strategy is beneficial in the full-support and high-skew cells, actively harmful in the sparse low-skew cells, and unhelpful everywhere else. The later entropy-gated solver milestone narrowed that read further by showing that a better solver policy can resolve the focused tested slice without changing the underlying control object.
