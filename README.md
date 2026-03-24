# Shape Budget

What if ellipse eccentricity is not just a way to describe a finished shape, but a readout of how much of a fixed geometric budget has already been spent?

That is the starting idea in this repository.

In the simplest picture, imagine two sources expanding outward while sharing a fixed total reach budget. If the sources start very close together, almost all of that budget is still available for wide, symmetric spread, so the resulting locus looks round. If the sources start far apart, much of the budget is already spent just bridging the gap between them, so the locus gets squeezed into a thinner, more elongated form.

This repository calls that intuition the **Shape Budget** idea, and its more mature version the **Budget Governor Principle (BGP)**:

> normalized separation acts like a governor on how much shape freedom remains.

The project began as a reframing of ellipse eccentricity. It has since grown into a larger research program about low-dimensional geometric control objects, inverse recovery, and when hidden budget structure can be recovered from boundary observations.

## The Easy Version

The easiest version of the idea is:

- two sources are offset from each other
- they share a fixed total reach budget
- some of that budget is consumed by the separation itself
- what remains shows up as width, spread, and curvature in the final shape

In the symmetric two-source ellipse case, the key ratio is:

```text
e = separation / total budget
```

More precisely, if the focal half-separation is `c` and the semimajor axis is `a`, then:

```text
e = c / a
```

and the remaining normalized transverse spread is:

```text
b / a = sqrt(1 - e^2)
```

So the repo’s core question is not only:

- “what shape is this?”

but also:

- “how much of the available budget was already committed before the shape had room to spread?”

## The Slightly More Technical Version

The clean mathematical base case is the constant-sum two-source Euclidean process:

```text
|x - F1| + |x - F2| = 2a
```

In standard geometry, that is just an ellipse.

In this repo’s language:

- `e = c / a` is the **allocation readout**
- `b / a = sqrt(1 - e^2)` is the **transverse residue**

That is the first form of the Budget Governor Principle:

> under the symmetric constant-sum two-source process, one normalized ratio governs the whole normalized shape family.

The work in this repo then asks:

1. Is that ratio really a control variable, or just a renamed parameter?
2. Does it stay useful when we add asymmetry, more sources, or anisotropy?
3. Can the hidden control object be recovered from boundary data?
4. Where does that recovery fail, and why?

## Key Plot 1: The Basic Compression Story

![Ellipse compression budget](plots/ellipse_compression_budget.png)

This is the plain-language picture of the whole project.

Each panel keeps the same total budget while changing only the normalized source separation. As `e` increases, more of the total budget is effectively pre-committed to bridging the gap between the two sources. The visible effect is not random: the family gets systematically narrower and more elongated.

What this plot is trying to show is that eccentricity can be read causally, not just descriptively. The shape is what is left after the separation tax is paid.

## Key Plot 2: Same Governor, Same Normalized Shape

![Control knob scale collapse](concepts/shape-budget/experiment_outputs/figures/control_knob_scale_collapse.png)

This figure is the first serious hardening test of the idea.

On the left, the same normalized separation ratio is realized at different absolute scales. On the right, after normalization, those shapes collapse onto each other almost exactly. That is the experimental basis for the claim that `e = c / a` behaves like a genuine control knob in the symmetric two-source case rather than just a label attached after the fact.

In other words:

- change the absolute size, keep the ratio fixed, and the normalized shape stays the same
- change the ratio, and the normalized family moves

That is the core one-knob result in this repo.

## What The Experiments Found

The project now supports a more mature story than the original concept note alone.

### 1. In the symmetric two-source case, the control knob is real

The main control-knob experiment showed:

- the circle-combination process reconstructs the analytic ellipse to machine precision
- fixed `e` gives normalized scale collapse
- several normalized observables behave as one-dimensional functions of `e`

The known-source inverse experiment then showed that `e` is also operational:

- it can be recovered accurately from noisy, partial, and sparse boundary observations
- it strongly outperforms raw separation `d`, raw budget `S`, or low-capacity models on `(d, S)` under scale shift

### 2. The story broadens in a structured way

Once symmetry is broken, the original one-knob family does **not** survive unchanged.

That is a good thing, not a bad one, because the failure is structured:

- asymmetry upgrades the family from one control variable to two
- the fixed-difference twin yields a hyperbola-side counterpart
- controlled anisotropy upgrades the raw family to `(e, alpha)`
- three-source families are governed by compact normalized source-placement objects rather than by one scalar

So the current view is not:

- “everything is one knob forever”

It is:

- “budget-governed shape families stay low-dimensional, but the right control object expands as the process gets richer”

## Key Plot 3: From Shape Description To Latent Variable

![Weighted multi-source inverse heatmap](technical-note/figures/figure4_weighted_multisource_inverse_heatmap.png)

This is one of the most important transitions in the repo.

By this point the project is no longer only saying that certain shape families can be *described* by compact normalized variables. It is showing that those hidden variables can be *recovered* from the boundary.

In the weighted three-source case, the important hidden object is:

- normalized source placement relative to budget
- plus normalized source participation weights

The inverse experiments show that this compact control object is recoverable from boundary-only data and clearly outperforms simpler baselines that ignore the weight degrees of freedom.

That is why the project increasingly talks about **operational latent variables** rather than just geometric descriptors.

### 3. The current bottleneck is not “the idea breaks”

The hardest branch so far is the pose-free anisotropic inverse:

- unknown rotation
- unknown anisotropy
- unknown geometry
- unknown participation weights

What the repo found is surprisingly specific:

- geometry stays fairly recoverable
- weights degrade, but remain usable
- anisotropy `alpha` becomes the weakly identified direction

That means the failure is selective, not uniform.

The current interpretation is that this is mainly a **symmetry-handling** problem: hidden rotation can impersonate medium anisotropy much more easily than it can impersonate the underlying normalized geometry.

## Key Plot 4: The Signal Is There, But Pose Handling Matters

![Oracle alignment ceiling](technical-note/figures/figure5_oracle_alignment_ceiling_alpha_methods.png)

This figure shows one of the most important diagnostics in the whole folder.

The baseline pose-free anisotropic inverse does not recover `alpha` well. But when the inverse is given the true pose, `alpha` error drops dramatically across every tested regime. That means the anisotropy signal is genuinely present in the boundary; the current pipeline is mostly losing it because practical pose handling is unstable under incomplete observations.

This is why the repo’s current bottleneck is no longer “is there really a latent variable here?” It is “how do we preserve enough broken symmetry before inference starts?”

## Key Plot 5: Where The Current Pipeline Fails

![Alignment failure map](technical-note/figures/figure6_alignment_failure_map_capture.png)

This map is the current state of the art for the hardest branch.

It shows that practical pose handling does not fail everywhere equally. The bad regions are structured:

- sparse or partial observations are much harder than full observations
- low-skew and especially mid-skew source geometries are harder than high-skew ones
- moderate and strong anisotropy are harder than weak anisotropy

That matters because it turns a vague problem into a targetable one. The next method does not need to be globally clever in an abstract way. It needs to be better in these specific failure cells.

## Where To Start Reading

If you want the shortest path through the project:

1. Read the original idea in [CONCEPT.md](concepts/shape-budget/CONCEPT.md).
2. Read the mathematical cleanup in [DERIVATION.md](concepts/shape-budget/DERIVATION.md).
3. Read the higher-level synthesis in [technical_note.md](technical-note/technical_note.md).

If you want the strongest evidence path:

1. [CONTROL_KNOB_EXPERIMENT.md](concepts/shape-budget/CONTROL_KNOB_EXPERIMENT.md)
2. [IDENTIFIABILITY_AND_BASELINES.md](concepts/shape-budget/IDENTIFIABILITY_AND_BASELINES.md)
3. [ASYMMETRY_EXPERIMENT.md](concepts/shape-budget/ASYMMETRY_EXPERIMENT.md)
4. [WEIGHTED_MULTISOURCE_INVERSE_EXPERIMENT.md](concepts/shape-budget/WEIGHTED_MULTISOURCE_INVERSE_EXPERIMENT.md)
5. [WEIGHTED_ANISOTROPIC_INVERSE_EXPERIMENT.md](concepts/shape-budget/WEIGHTED_ANISOTROPIC_INVERSE_EXPERIMENT.md)
6. [ORACLE_ALIGNMENT_CEILING_EXPERIMENT.md](concepts/shape-budget/ORACLE_ALIGNMENT_CEILING_EXPERIMENT.md)
7. [ALIGNMENT_FAILURE_MAP_EXPERIMENT.md](concepts/shape-budget/ALIGNMENT_FAILURE_MAP_EXPERIMENT.md)

If you want the full research trajectory, see [RESEARCH_ROADMAP.md](concepts/shape-budget/RESEARCH_ROADMAP.md).

## Repo Tour

### Core idea and synthesis

- [concepts/shape-budget/CONCEPT.md](concepts/shape-budget/CONCEPT.md)
- [concepts/shape-budget/DERIVATION.md](concepts/shape-budget/DERIVATION.md)
- [technical-note/technical_note.md](technical-note/technical_note.md)

### Experiment notes

- [concepts/shape-budget/CONTROL_KNOB_EXPERIMENT.md](concepts/shape-budget/CONTROL_KNOB_EXPERIMENT.md)
- [concepts/shape-budget/IDENTIFIABILITY_AND_BASELINES.md](concepts/shape-budget/IDENTIFIABILITY_AND_BASELINES.md)
- [concepts/shape-budget/ASYMMETRY_EXPERIMENT.md](concepts/shape-budget/ASYMMETRY_EXPERIMENT.md)
- [concepts/shape-budget/MULTISOURCE_EXPERIMENT.md](concepts/shape-budget/MULTISOURCE_EXPERIMENT.md)
- [concepts/shape-budget/WEIGHTED_MULTISOURCE_INVERSE_EXPERIMENT.md](concepts/shape-budget/WEIGHTED_MULTISOURCE_INVERSE_EXPERIMENT.md)
- [concepts/shape-budget/WEIGHTED_ANISOTROPIC_INVERSE_EXPERIMENT.md](concepts/shape-budget/WEIGHTED_ANISOTROPIC_INVERSE_EXPERIMENT.md)
- [concepts/shape-budget/POSE_FREE_WEIGHTED_ANISOTROPIC_INVERSE_EXPERIMENT.md](concepts/shape-budget/POSE_FREE_WEIGHTED_ANISOTROPIC_INVERSE_EXPERIMENT.md)
- [concepts/shape-budget/ORACLE_ALIGNMENT_CEILING_EXPERIMENT.md](concepts/shape-budget/ORACLE_ALIGNMENT_CEILING_EXPERIMENT.md)
- [concepts/shape-budget/ALIGNMENT_FAILURE_MAP_EXPERIMENT.md](concepts/shape-budget/ALIGNMENT_FAILURE_MAP_EXPERIMENT.md)

### Scripts and plots

- [generate_shape_budget_plots.py](generate_shape_budget_plots.py)
- [generate_brainstorm_shape_budget_visuals.py](generate_brainstorm_shape_budget_visuals.py)
- [plots](plots)
- [concepts/shape-budget](concepts/shape-budget)

## Reproducing The Main Artifacts

At the top level:

```bash
python3 generate_shape_budget_plots.py
python3 generate_brainstorm_shape_budget_visuals.py
```

For the deeper experiment suite, each note in `concepts/shape-budget/` is paired with a `run_*.py` script in the same folder. For example:

```bash
python3 concepts/shape-budget/run_control_knob_experiment.py
python3 concepts/shape-budget/run_identifiability_and_baselines_experiment.py
python3 concepts/shape-budget/run_alignment_failure_map_experiment.py
```

To build the technical note PDF:

```bash
cd technical-note
./build_pdf.sh
```

## Current Status

The repo now supports a careful version of the Budget Governor Principle:

- in the symmetric two-source Euclidean case, one normalized ratio really does govern normalized geometry
- in richer cases, that same budget logic expands into compact low-dimensional control objects
- in weighted inverse settings, those control objects behave like operational latent variables
- the main open problem is robust symmetry handling in the pose-free anisotropic branch

So the project is no longer just:

- “here is a neat way to think about ellipses”

It is now closer to:

- “here is a growing framework for budget-governed geometry and how much of its hidden structure can be recovered from what we observe”
