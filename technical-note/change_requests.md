# Technical Note Change Requests

This document tracks suggested edits for the technical note before any final implementation decisions are made.

Current intake rule:

- every new review item enters as `proposed`
- no item is marked `accepted` or `rejected` until we decide explicitly
- no item is marked `implemented` until the note is actually changed

## Status Key

- `proposed`
- `accepted`
- `rejected`
- `deferred`
- `implemented`

## Index

| ID | Status | Location | Short Label | Source |
| --- | --- | --- | --- | --- |
| CR-001 | proposed | Abstract | Shorten and split opening abstract sentences | Grok review |
| CR-002 | proposed | In-text self-citation | Replace self-citation with in-document cross-reference | Grok review |
| CR-003 | proposed | Figure 7 caption | Add sub-panel guidance to phase-map caption | Grok review; Model Council review |
| CR-004 | proposed | Section 9 summary | Add compact summary table for generalization results | Grok review; Model Council review |
| CR-005 | proposed | Artifact references | Add stronger reproducibility anchors for figures and artifacts | Model Council review |
| CR-006 | proposed | Jargon onboarding | Add one-sentence definitions for dense solver terms on first use | Model Council review |
| CR-007 | proposed | Section 5 probe discussion | Define “crossover” operationally | Model Council review |
| CR-008 | proposed | Solver milestone framing | Make “solved slice” benchmark meaning more explicit | Model Council review |
| CR-009 | proposed | External positioning | Add brief related-work / positioning section | Model Council review |
| CR-010 | proposed | Global structure | Reduce redundancy between the opening statement and Section 9 summary | Model Council review |
| CR-011 | proposed | Publication language | Tighten informal wording for publication tone | Model Council review |
| CR-012 | proposed | Abstract | Add explicit novelty sentence to the abstract | MC Copilot review |
| CR-013 | proposed | Opening statement | Add one-sentence early “why this matters” bridge | MC Copilot review |
| CR-014 | proposed | Conceptual framing | Add conservation-law-like / mathematical-interpretation framing | MC Copilot review |
| CR-015 | proposed | Mathematical core | Note that collapse error is machine-precision limited | MC Copilot review |
| CR-016 | proposed | Operational evidence | Explain why raw `(d, S)` fail under scale shift | MC Copilot review |
| CR-017 | proposed | Latent-variable framing | State that the latent variable is forced by budget structure, not arbitrary | MC Copilot review |
| CR-018 | proposed | Section 8 clarity | Tighten ambiguity-vs-entropy explanation and add bundle-broad intuition | MC Copilot review |
| CR-019 | proposed | Open problems structure | Group unresolved targets by problem class | MC Copilot review |
| CR-020 | proposed | Visual synthesis | Add diagram of control-object evolution across regimes | MC Copilot review |

## Change Requests

### Template

#### CR-XXX

- Status: `proposed`
- Source:
- Date:
- Location:
- Short label:
- Request:
- Rationale:
- Decision notes:
- Implementation notes:

#### CR-001

- Status: `proposed`
- Source: Grok review
- Date: 2026-03-29
- Location: Abstract front matter
- Short label: Shorten and split opening abstract sentences
- Request: Tighten the abstract opening by splitting the first sentence or moving some qualifier language into a second sentence so the big picture lands faster.
- Rationale: The review says the abstract is strong but dense, and that the first sentence is carrying too much structure before the reader has the main frame.
- Decision notes:
- Implementation notes:

#### CR-002

- Status: `proposed`
- Source: Grok review
- Date: 2026-03-29
- Location: In-text self-citation near the opening statement
- Short label: Replace self-citation with in-document cross-reference
- Request: Consider removing the in-note parenthetical citation to the repo note itself and replacing it with a section cross-reference or a more direct in-document pointer.
- Rationale: The review notes that citing the 2026 note inside the 2026 note itself can read awkwardly and that keeping the reader inside the document may be cleaner.
- Decision notes:
- Implementation notes:

#### CR-003

- Status: `proposed`
- Source: Grok review; reinforced by Model Council review
- Date: 2026-03-29
- Location: Figure 7 caption
- Short label: Add sub-panel guidance to phase-map caption
- Request: Expand the Figure 7 caption with a brief one-line guide for each sub-panel so readers who jump straight to figures can decode the dashboard faster.
- Rationale: Multiple reviews say Figure 7 is valuable but visually dense, and that panel-level cues in the caption would improve scanability.
- Decision notes:
- Implementation notes:

#### CR-004

- Status: `proposed`
- Source: Grok review; reinforced by Model Council review; reinforced by MC Copilot review
- Date: 2026-03-29
- Location: Section 9, `What The Repo Establishes`
- Short label: Add compact summary table for generalization results
- Request: Consider adding a small summary table contrasting symmetric Euclidean, asymmetric, anisotropic, and multi-source branches by control-object dimension and representative recovery/collapse metrics.
- Rationale: Multiple reviews suggest that a compact table would make the generalization story more scannable for publication readers and make the extension pattern visually explicit.
- Decision notes:
- Implementation notes:

#### CR-005

- Status: `proposed`
- Source: Model Council review
- Date: 2026-03-29
- Location: Global figure and artifact references
- Short label: Add stronger reproducibility anchors for figures and artifacts
- Request: Strengthen figure and artifact anchoring with clearer paths, identifiers, or version references so a publication reader can trace major claims to concrete repo artifacts more directly.
- Rationale: The review flags reproducibility anchors as one of the biggest remaining doc-quality gaps, especially for readers approaching the note as a stand-alone technical artifact.
- Decision notes:
- Implementation notes:

#### CR-006

- Status: `proposed`
- Source: Model Council review
- Date: 2026-03-29
- Location: First use of dense solver terminology
- Short label: Add one-sentence definitions for dense solver terms on first use
- Request: Add short plain-language definitions when specialized terms first appear, especially for items like `entropy-gated solver`, `ambiguity gate`, `anchored-uncertainty gate`, `gauge-broad`, and `bundle-broad`.
- Rationale: The review identifies reader onboarding as a cross-model weakness and recommends reducing concept-load friction without weakening the technical story.
- Decision notes:
- Implementation notes:

#### CR-007

- Status: `proposed`
- Source: Model Council review
- Date: 2026-03-29
- Location: Section 5 probe-sensitivity discussion
- Short label: Define “crossover” operationally
- Request: Make clear what the reported probe “crossover points” mean operationally rather than leaving the term implicit.
- Rationale: The review notes that the current wording is quantitative but not fully self-contained for readers who have not looked at the underlying experiment note.
- Decision notes:
- Implementation notes:

#### CR-008

- Status: `proposed`
- Source: Model Council review
- Date: 2026-03-29
- Location: Solver milestone framing in abstract and Section 8
- Short label: Make “solved slice” benchmark meaning more explicit
- Request: Clarify what counts as “solved” for the focused slice in benchmark terms, not only by repeating the raw `0.1050` vs `0.1091` and `0.1064` vs `0.1104` comparisons.
- Rationale: The review notes that different readers will ask whether the solver win is interpretably meaningful, statistically meaningful, or only local; a short benchmark-oriented sentence would make the claim easier to parse.
- Decision notes:
- Implementation notes:

#### CR-009

- Status: `proposed`
- Source: Model Council review
- Date: 2026-03-29
- Location: Front matter or early-body positioning
- Short label: Add brief related-work / positioning section
- Request: Consider a short positioning section that distinguishes this note from classical conic parameterization, standard inverse framing, and generic statistical shape-model language.
- Rationale: The review says this would help the note stand alone outside the repo context and make the novelty claim easier for external readers to place.
- Decision notes:
- Implementation notes:

#### CR-010

- Status: `proposed`
- Source: Model Council review
- Date: 2026-03-29
- Location: Opening statement and Section 9
- Short label: Reduce redundancy between the opening statement and Section 9 summary
- Request: Review the overlap between the early “established result set” summary and the later “What The Repo Establishes” list and decide whether one can be tightened or partially merged.
- Rationale: The review flags some repeated summary structure that may read well internally but could feel redundant in a publication-oriented note.
- Decision notes:
- Implementation notes:

#### CR-011

- Status: `proposed`
- Source: Model Council review
- Date: 2026-03-29
- Location: Global prose style
- Short label: Tighten informal wording for publication tone
- Request: Review informal words or tonal shortcuts such as `clean`, `chaotically`, and similar phrasing to decide whether the publication version should use more neutral scientific language.
- Rationale: The review notes that most of the tone is strong, but a few informal words may weaken the note’s external publication surface.
- Decision notes:
- Implementation notes:

#### CR-012

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Abstract
- Short label: Add explicit novelty sentence to the abstract
- Request: Add one sentence in the abstract that states the main scientific novelty directly, for example that the work establishes budget-normalized latent variables rather than raw geometric or scale parameters as the governing state across the tested families.
- Rationale: The review says the abstract is already strong but would land faster for external readers if the novelty is stated in one plain sentence rather than only inferred from the dense summary.
- Decision notes:
- Implementation notes:

#### CR-013

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Opening statement
- Short label: Add one-sentence early “why this matters” bridge
- Request: Add a short motivating sentence near the opening statement that tells the reader why the result matters computationally or scientifically before the note moves deeper into the program.
- Rationale: The review says the opening is clean but would benefit from one explicit sentence that anchors importance for readers new to the framework.
- Decision notes:
- Implementation notes:

#### CR-014

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Core proposal / conceptual framing
- Short label: Add conservation-law-like / mathematical-interpretation framing
- Request: Consider explicitly describing BGP as conservation-law-like, or add a short mathematical-interpretation subsection that frames the budget constraint as governing residual geometric freedom.
- Rationale: The review says this hook is already implicit and naming it would strengthen the conceptual positioning of the note.
- Decision notes:
- Implementation notes:

#### CR-015

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Mathematical core section
- Short label: Note that collapse error is machine-precision limited
- Request: Add a brief remark that the scale-collapse error in the symmetric base case is effectively machine-precision limited.
- Rationale: The review says this would sharpen the reader’s understanding of how exact the base-case result really is.
- Decision notes:
- Implementation notes:

#### CR-016

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Operational evidence section
- Short label: Explain why raw `(d, S)` fail under scale shift
- Request: Add a sentence explaining that raw separation and raw budget retain absolute scale information while `e` encodes normalized budget allocation, which is why `e` transfers under scale-held-out prediction and the raw variables do not.
- Rationale: The review says the section is empirically strong, and that one explanatory sentence would make the result more self-contained.
- Decision notes:
- Implementation notes:

#### CR-017

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: From descriptor to operational latent variable
- Short label: State that the latent variable is forced by budget structure, not arbitrary
- Request: Add a sentence clarifying that the latent control object is not an arbitrary learned descriptor but a compact state forced by the normalized budget structure of the family.
- Rationale: The review says this would sharpen the inferential significance of the latent-variable framing.
- Decision notes:
- Implementation notes:

#### CR-018

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Section 8, pose-free anisotropy discussion
- Short label: Tighten ambiguity-vs-entropy explanation and add bundle-broad intuition
- Request: Tighten the explanation of the ambiguity and entropy axes, and add one short intuition sentence for `bundle-broad` cases in reader-facing language.
- Rationale: The review says this is the densest and most interesting section, and that a little more conceptual guidance would improve accessibility without changing the technical content.
- Decision notes:
- Implementation notes:

#### CR-019

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Limits and scope / open targets
- Short label: Group unresolved targets by problem class
- Request: Reorganize the unresolved-target or open-target list into categories such as theoretical, computational, inverse-problem, and application.
- Rationale: The review says the open problems are already honest and useful, and that grouping them would make the research program structure clearer.
- Decision notes:
- Implementation notes:

#### CR-020

- Status: `proposed`
- Source: MC Copilot review
- Date: 2026-03-29
- Location: Visual synthesis
- Short label: Add diagram of control-object evolution across regimes
- Request: Consider adding a diagram that shows the evolution from symmetric to asymmetric to anisotropic to multi-source to pose-free anisotropic branches, with the control object at each stage.
- Rationale: The review says this would visually reinforce the note’s main narrative arc and make the extension structure easier to scan.
- Decision notes:
- Implementation notes:
