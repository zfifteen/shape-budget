# Stability threshold router

This experiment extends the cached support-vs-joint chooser workflow with one explicit stability intervention.

## Idea

The bank-adaptive ridge chooser already produces an observable preference score for the joint candidate versus the support-aware candidate. The fresh-bank failure suggests the raw zero-threshold decision boundary is too eager to switch toward the joint candidate on some confirmation cases.

This experiment keeps the same two candidate generators and the same disjoint calibration/holdout protocol, but replaces the fixed zero threshold with a frozen cell-aware threshold map learned on calibration blocks only.

For each of the six focused cells, the experiment:

1. computes the ridge chooser score from observable features only
2. searches a small calibration-only threshold grid for that cell
3. freezes the chosen thresholds before holdout evaluation
4. evaluates holdout block 1 and, only if holdout passes, the fresh-bank confirmation block

## Evaluation rules preserved

- disjoint calibration versus holdout blocks
- frozen thresholds before holdout evaluation
- no true latent errors used at evaluation time
- no routing by regime label alone; the score still depends on observable candidate features and the thresholds only adjust the decision boundary by focused cell
- one explicit solver family, no repeated holdout tuning loops

## Outputs

The experiment writes cache tables, frozen threshold artifacts, prediction CSVs, and JSON summaries under `outputs/`.
