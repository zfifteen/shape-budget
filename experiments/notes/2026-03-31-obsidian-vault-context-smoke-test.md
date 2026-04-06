---
type: experiment
project: shape-budget
status: complete
date: 2026-03-31
hypothesis: The project README, local project context, and recent experiment notes are sufficient for Codex to recover the active thesis and propose one bounded next step without rereading the whole repo.
related_notes:
  - shape-budget/docs/PROJECT_CONTEXT.md
  - shape-budget/experiments/notes/discovery.md
  - shape-budget/experiments/notes/follow-ups.md
related_code: []
related_artifacts:
  - shape-budget/technical-note/technical_note.md
tags:
  - experiment
  - shape-budget
  - obsidian
  - codex
---

# Obsidian Vault Context Smoke Test

## Question

Can a Codex session recover the live `shape-budget` thesis and propose a bounded next experiment from the vault-native context alone?

## Context

This note validates the new Obsidian-over-`IdeaProjects` workflow. The context packet for the dry run was:

- [[shape-budget/README]]
- [[shape-budget/docs/PROJECT_CONTEXT]]
- [[shape-budget/experiments/notes/discovery]]
- [[shape-budget/experiments/notes/follow-ups]]

## Method

Read the project entrypoint, the local project context note, and the two most recent experiment notes. Then synthesize one bounded next-step experiment without traversing the entire repository.

## Commands

```bash
# No repo-side scientific code was executed for this smoke test.
# This was a vault-context dry run.
```

## Outputs

- Confirmed canonical project entrypoint: [[shape-budget/README]]
- Confirmed richer local context: [[shape-budget/docs/PROJECT_CONTEXT]]
- Proposed next experiment note path: `shape-budget/experiments/notes/2026-03-31-hyperbola-flip-scope-probe.md`

## Result

The context packet was sufficient. The active thesis is recoverable from the repo `README.md` plus the adjacent notes, and a bounded next step emerged cleanly: test the constant-difference hyperbola-side twin as the nearest structural extension of the current shape-budget story.

## Interpretation

This is a good sign for the vault contract. Codex did not need to rediscover the entire repo from scratch; the linked notes already preserved the thesis, scope, and immediate next move in a form that is legible to both Obsidian and Codex.

## Next Step

If this dry-run pattern stays useful, create the next note at `shape-budget/experiments/notes/2026-03-31-hyperbola-flip-scope-probe.md`, link it back to [[shape-budget/README]] and [[shape-budget/docs/PROJECT_CONTEXT]], and record the actual commands and artifacts there.
