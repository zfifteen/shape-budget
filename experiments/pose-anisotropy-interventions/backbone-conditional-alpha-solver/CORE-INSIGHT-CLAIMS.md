# Core Insight Claims

## Definitions

- `gate-open trial`
  A trial with `gate_open_flag = 1` in [backbone_conditional_alpha_solver_trials.csv](outputs/backbone_conditional_alpha_solver_trials.csv).

- `anchored output`
  The Layer 2 ensemble point estimate `anchored_alpha_output`.

- `refined output`
  The Layer 3 ensemble point estimate `refined_alpha_output`.

- `error delta`
  `refined_alpha_output_abs_error - anchored_alpha_output_abs_error`.
  Negative means refinement helped.
  Positive means refinement hurt.

- `bank-wise correction`
  For one bank on one gate-open trial:
  `log(refined_alpha) - log(anchored_alpha)`.

- `correction flux`
  Trial-level mean absolute bank-wise correction:
  `F = mean_i abs(log(refined_alpha_i) - log(anchored_alpha_i))`
  over the five banks.

- `provisional event-trigger rule`
  A calibration-frozen Layer 3 policy:
  use refined only if correction-sign majority is at least `0.6` and correction flux `F >= 0.0738`.
  Otherwise stay anchored.

## Smallest Testable Claims Already Supported By This Folder

### C01. Gate-open does not imply refine-now.

Claim:
Layer 2 gate-open and Layer 3 refine-now are different decisions.

Test:
On gate-open trials, count how often refined beats anchored.

Current evidence:
There are `53` gate-open trials.
Refined beats anchored on `30`.
Anchored beats refined on `23`.

Falsified if:
Fresh reruns push the harmful count close to zero.

### C02. The Layer 2 gate still generalizes even when Layer 3 misses.

Claim:
The main holdout miss is downstream of the gate, not proof that the gate itself failed.

Test:
Compare gate balanced accuracy to Layer 3 holdout error.

Current evidence:
The gate keeps holdout balanced accuracy `0.6818` and confirmation balanced accuracy `0.7000`.
But always-refine Layer 3 still loses to anchored on holdout: `0.1650` vs `0.1598`.

Falsified if:
The gate collapses on fresh reruns or the holdout miss disappears without changing the Layer 3 trigger.

### C03. Cross-bank refined spread is not the main failure variable.

Claim:
The harmful Layer 3 cases are not explained mainly by final refined bank span.

Test:
Correlate refined bank log-span with error delta on gate-open trials.

Current evidence:
The correlation is about `-0.0631`, which is essentially no useful signal.

Falsified if:
A rerun shows strong positive correlation between refined bank span and error delta.

### C04. A post-gate correction signal exists and is observable without latent access.

Claim:
The next Layer 3 decision can be made from solver outputs alone.

Test:
Compute correction flux from [backbone_conditional_alpha_solver_bank_rows.csv](outputs/backbone_conditional_alpha_solver_bank_rows.csv) without using true `alpha` or latent error.

Current evidence:
Correction flux is fully computable from `anchored_alpha` and `refined_alpha` in the bank rows.

Falsified if:
The signal turns out to require latent labels or hidden variables at evaluation time.

### C05. High-correction events are sparse.

Claim:
Helpful Layer 3 corrections are not the default state of gate-open trials.

Test:
Calibrate a correction-flux threshold on calibration only, then measure the fraction of fresh gate-open trials that pass it.

Current evidence:
Under the provisional event-trigger rule, switch rates are:
- calibration: `7/25 = 0.28`
- holdout: `1/14 = 0.0714`
- confirmation: `2/14 = 0.1429`

Falsified if:
The best frozen threshold ends up switching most gate-open trials.

### C06. Always-refine is the wrong default for Layer 3.

Claim:
A gated trial is not enough evidence to justify refinement by default.

Test:
Compare always-refine vs anchored on holdout gate-open trials.

Current evidence:
Holdout gate-open error is:
- anchored: `0.1598`
- always-refine: `0.1650`

Falsified if:
Always-refine consistently beats anchored on holdout across reruns.

### C07. A sparse event-triggered Layer 3 can beat always-refine on holdout.

Claim:
Layer 3 improves if refinement is treated as a sparse event rather than a default action.

Test:
Freeze the provisional event-trigger rule on calibration and compare its holdout error to always-refine.

Current evidence:
Holdout gate-open error becomes `0.1592` under the provisional rule, beating always-refine at `0.1650`.

Falsified if:
The triggered policy does not beat always-refine on fresh holdout blocks.

### C08. A sparse event-triggered Layer 3 can keep confirmation gain over anchored.

Claim:
Suppressing weak correction events does not require giving up the confirmation win.

Test:
Freeze the provisional event-trigger rule on calibration and compare its confirmation error to anchored.

Current evidence:
Confirmation gate-open error becomes `0.1262` under the provisional rule, beating anchored at `0.1353`.

Falsified if:
The triggered policy loses the confirmation advantage over anchored.

### C09. The holdout miss is concentrated, not diffuse.

Claim:
The harmful always-refine behavior clusters in a small subset of cells rather than appearing uniformly everywhere.

Test:
Check gate-open cell summaries and trial rows.

Current evidence:
The sharpest blocker remains `sparse_full_noisy + mid_skew` on holdout.
That cell has:
- best output error: `0.0695`
- anchored output error: `0.1418`
- refined output error: `0.1697`

Falsified if:
Future runs show the holdout harm spreading evenly across most gate-open cells.

### C10. Layer 3 is better described as correction-event detection than as generic refinement.

Claim:
The right role of Layer 3 is to detect whether a real post-anchor correction event is present.

Test:
Compare an event-triggered Layer 3 against an always-refine Layer 3 on fresh blocks.

Current evidence:
The provisional event-trigger rule beats always-refine on holdout and keeps a confirmation gain over anchored.

Falsified if:
The event-trigger framing stops helping once tested as a first-class Layer 3 experiment.

## Smallest Directly Testable Claims For The Next Pass

### N01. Low-flux gate-open trials are usually weak-correction cases.

Test:
Freeze a calibration threshold on correction flux and compare error delta above vs below that threshold on holdout and confirmation.

Support would look like:
Low-flux trials have non-negative mean error delta.

Weakening observation:
Low-flux trials still show strong negative mean error delta.

### N02. High-flux gate-open trials contain most of the real Layer 3 gain.

Test:
Freeze a calibration threshold on correction flux and compare anchored vs refined only inside the high-flux subset.

Support would look like:
High-flux trials have clearly negative mean error delta on fresh blocks.

Weakening observation:
High-flux trials are no better than low-flux trials.

### N03. Correction flux is more useful than corrected-span for deciding whether to refine.

Test:
Build two frozen post-gate policies:
one based on correction flux and one based on refined bank span.
Compare them on holdout and confirmation.

Support would look like:
The correction-flux policy wins on fresh blocks.

Weakening observation:
Refined span matches or beats the flux policy.

### N04. Sign coherence is optional and correction flux is the core variable.

Test:
Compare two frozen policies:
one using correction flux alone and one using correction flux plus sign-majority.

Support would look like:
The flux-only rule performs similarly.

Weakening observation:
The sign-majority check is required for the gain.

### N05. The event-trigger principle scales as a general Layer 3 rule.

Test:
Implement event-triggered Layer 3 as a first-class experiment and rerun the full calibration, holdout, and confirmation sweep.

Support would look like:
The new Layer 3 beats both anchored and always-refine on holdout and confirmation inside the gate-open region.

Weakening observation:
The gain disappears once the rule is promoted from a post-hoc probe to a real experiment.
