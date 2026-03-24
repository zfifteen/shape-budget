# Alignment Failure Map Experiment

## Purpose

The oracle ceiling established that most of the lost pose-free `alpha` signal is genuinely recoverable if pose is handled perfectly.

That immediately raises the next practical question:

> where does simple practical alignment still work, and where does it become ill-posed?

This note answers that by mapping alignment behavior across observation regime, anisotropy strength, and source-geometry skew.

## Research Question

Across the pose-free weighted anisotropic inverse, which regions of the problem space let practical orientation locks recover meaningful oracle headroom on `alpha`, and which regions consistently break them?

## Pre-Benchmark Logic Audit

Before the benchmark, the script was checked in the same way as the recent alignment experiments.

Code sanity:

- script compiled cleanly

Alignment primitives re-audited:

- harmonic lock rotation invariance: max aligned RMSE `0.0`
- principal-axis lock rotation invariance: max aligned RMSE `0.0`
- harmonic clean exact-bank recovery: `1.0`
- principal-axis clean exact-bank recovery: `1.0`

So the map is not mixing together benchmark effects with unverified alignment logic.

The experiment script is [run_alignment_failure_map_experiment.py](run.py#L1).

## Method

Each trial compares four inverse paths on the same pose-free anisotropic observation:

- shift-aware baseline
- harmonic orientation lock
- principal-axis orientation lock
- oracle alignment

The map bins each trial along two latent axes:

- anisotropy strength: `|log(alpha)|`
  - `weak`: `< 0.10`
  - `moderate`: `[0.10, 0.25)`
  - `strong`: `>= 0.25`
- geometry skew: `|t|`
  - `low_skew`: `< 0.20`
  - `mid_skew`: `[0.20, 0.45)`
  - `high_skew`: `>= 0.45`

The observation regimes are:

- `full_clean`
- `full_noisy`
- `partial_arc_noisy`
- `sparse_full_noisy`
- `sparse_partial_high_noise`

Reference bank:

- anisotropy-aware bank size: `300`

Trials:

- `60` trials per observation regime

The map reports three main summaries per cell:

- fraction of oracle `alpha` gain captured
- clean alignment RMSE induced by the observation-derived shift
- fraction of trials where the lock improves `alpha` over the pose-free baseline

For the gain-capture metric, the summary uses cell means and is only interpreted directly when the cell has positive oracle headroom. That matters because ratio metrics become unstable in tiny-headroom cells.

## Main Result

The result is strong and useful.

> Practical alignment failure is highly structured, not uniform. Support regime dominates the map, higher geometry skew is generally friendlier, weak anisotropy is the easiest part of the space, and the hardest cells are sparse or partial observations with low-to-mid skew, where practical locks often move away from the large oracle headroom rather than toward it.

The summary file is [alignment_failure_map_summary.json](outputs/alignment_failure_map_summary.json).

The cleanest overall read is:

- full observation is the only region where mean oracle-gain capture stays broadly non-negative
- sparse regimes are mostly negative relative to oracle headroom even when they sometimes beat the baseline on individual trials
- `high_skew` cells are the most alignment-friendly overall
- `mid_skew` cells are the most consistently unstable
- principal-axis locking is usually a bit better than harmonic locking in the harder regions, but neither method is robust enough to approach oracle reliably

## By Observation Regime

These summaries average only over cells with `count >= 3` and positive oracle headroom.

- `full_clean`
  - harmonic mean capture: `0.0403`
  - principal-axis mean capture: `0.0065`
  - clean alignment RMSE stays around `0.071`
  - practical locks help only occasionally, but they are not broadly destructive

- `full_noisy`
  - harmonic mean capture: `0.1625`
  - principal-axis mean capture: `0.1667`
  - clean alignment RMSE stays around `0.064`
  - this is the friendliest realistic regime for simple locking

- `partial_arc_noisy`
  - harmonic mean capture: `-0.1712`
  - principal-axis mean capture: `0.1410`
  - clean alignment RMSE rises to about `0.105` harmonic and `0.092` principal-axis
  - this is a mixed regime where principal-axis locking sometimes tracks oracle meaningfully and harmonic often does not

- `sparse_full_noisy`
  - harmonic mean capture: `-0.5937`
  - principal-axis mean capture: `-0.2141`
  - clean alignment RMSE rises to about `0.138` harmonic and `0.124` principal-axis
  - once support is sparse, both locks usually lose oracle headroom overall

- `sparse_partial_high_noise`
  - harmonic mean capture: `-1.0851`
  - principal-axis mean capture: `-1.0024`
  - clean alignment RMSE stays around `0.12`
  - this is the clearest failure zone in the current map

So support regime is the dominant axis.

## By Anisotropy Strength

Again restricting to reasonably populated positive-headroom cells:

- `weak`
  - harmonic mean capture: `0.2003`
  - principal-axis mean capture: `0.6239`
  - clean alignment RMSE stays lowest, about `0.043` harmonic and `0.039` principal-axis

- `moderate`
  - harmonic mean capture: `-0.7551`
  - principal-axis mean capture: `-0.5599`
  - clean alignment RMSE rises to about `0.08`

- `strong`
  - harmonic mean capture: `-0.2080`
  - principal-axis mean capture: `-0.1811`
  - clean alignment RMSE rises further to about `0.137` harmonic and `0.126` principal-axis

The non-obvious part is that `moderate` anisotropy is the hardest slice on average in this map, not the strongest slice.

That suggests the alignment issue is not just “more anisotropy is always worse.”

## By Geometry Skew

- `low_skew`
  - harmonic mean capture: `-0.3391`
  - principal-axis mean capture: `-0.0074`
  - clean alignment RMSE is relatively high, about `0.124` harmonic and `0.115` principal-axis

- `mid_skew`
  - harmonic mean capture: `-0.6276`
  - principal-axis mean capture: `-0.5008`
  - this is the most consistently difficult skew band

- `high_skew`
  - harmonic mean capture: `0.0588`
  - principal-axis mean capture: `0.1307`
  - improvement fractions are also strongest here

So geometry asymmetry helps.

The practical locks do best when the source configuration is already far from ambiguous in orientation.

## Reading The Capture Metric Carefully

The largest positive and negative capture cells should not be read as global headlines by themselves.

For example:

- the best principal-axis cell is `sparse_partial_high_noise`, `weak`, `low_skew`, with capture `1.4428`
- the worst principal-axis cell is `sparse_partial_high_noise`, `moderate`, `mid_skew`, with capture `-6.7104`

Those cells are real, but they are also small or low-headroom enough that the ratio can swing hard.

That is why the main interpretation of the map should come from:

- the broad regime patterns
- the clean alignment RMSE panel
- the repeated sign structure across families of cells

not from any single extreme ratio cell.

## Interpretation

This experiment sharpens the alignment diagnosis considerably.

The problem is no longer just:

- simple alignment is unstable

It is now:

- simple alignment is unstable in specific parts of the latent space

The map says:

- full support can often tolerate naive locking
- partial support becomes mixed
- sparse support usually breaks simple locking
- higher source-geometry skew helps
- weak anisotropy is comparatively easy
- moderate and strong anisotropy, especially with low-to-mid skew, are where practical locking loses the oracle opportunity

The clean alignment RMSE panel helps explain why.

Once the observation-derived shift is wrong enough, usually around the `0.10` to `0.14` RMSE range in this map, the practical method often stops capturing oracle headroom even if it still wins on some individual trials.

So the next method does not need to be globally smarter in an abstract sense.

It needs to be better specifically in the bad cells.

## What This Establishes

This experiment does show:

- the pose-free alignment problem has a structured failure boundary rather than a uniform performance drop
- support regime is the dominant control axis for practical locking quality
- geometry skew is an important secondary control axis
- weak anisotropy is much easier for practical alignment than moderate or strong anisotropy
- principal-axis locking is somewhat better than harmonic locking in the harder cells, but neither is close to oracle where it matters most

This experiment does not address:

- which practical alignment method can reliably capture a large fraction of oracle headroom inside the bad cells
- whether the best next improvement should come from better alignment scoring, candidate-conditioned alignment, or a different pose-equivariant representation
- how the failure boundary changes under finer support-fraction sweeps or richer source families

## Why The Result Matters

For the BGP program, this strengthens the project.

The map says the hidden budget-governed state is still there, and the pose penalty is not random noise. It is organized by how much symmetry remains effectively unbroken under the observation regime and source configuration.

That is a better place to be than “`alpha` is just hard.”

It means the next step should target:

- the sparse and partial regimes
- especially low-skew and mid-skew cells
- with alignment methods designed to stay stable exactly where the oracle headroom is large and the naive locks fail

## Figures

- [alignment_failure_map_capture.png](outputs/figures/alignment_failure_map_capture.png)
- [alignment_failure_map_alignment_rmse.png](outputs/figures/alignment_failure_map_alignment_rmse.png)
- [alignment_failure_map_improvement_rate.png](outputs/figures/alignment_failure_map_improvement_rate.png)

The clearest figure is [alignment_failure_map_capture.png](outputs/figures/alignment_failure_map_capture.png), but it is best read together with [alignment_failure_map_alignment_rmse.png](outputs/figures/alignment_failure_map_alignment_rmse.png), because the second panel shows where the bad cells are coming from.

## Artifacts

Data:

- [alignment_failure_map_summary.json](outputs/alignment_failure_map_summary.json)
- [alignment_failure_map_summary.csv](outputs/alignment_failure_map_summary.csv)
- [alignment_failure_map_trials.csv](outputs/alignment_failure_map_trials.csv)

Code:

- [run_alignment_failure_map_experiment.py](run.py#L1)
