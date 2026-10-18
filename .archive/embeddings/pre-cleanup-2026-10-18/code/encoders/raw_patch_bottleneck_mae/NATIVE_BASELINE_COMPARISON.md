# RPB v4 native baseline comparison

Date: 2026-10-06. Protocol: archive-readout-v1. Policy: 1.2.

New validation-only readout experiment; no new encoder training. This completes
phase 1 of [the continuation plan](NEXT_ADVANCE.md). The shared evaluator accepts
saved exports without importing an encoder. PCA follows raw observations only.

**RPB-v4:** learned global bottleneck. It reconstructs from its exact 32-number
global export. Model cells below contain the tag only, with descriptions in text
beside the tables, following [the reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md).

## Conditions

All four tasks use masters 1701/1802/1903 from the saved diversity-128 cohort.
Each master/task has 256 TRAIN examples from 128 source pairs and 128 VALIDATION
examples from 64 disjoint pairs, shape 3×32×3 and 10% natural missingness. Each
task has three independently trained encoder checkpoints, frozen at 512 completed
CUDA updates, batch size 8, RTX A2000. Tasks have separate models; this does not
establish one universal encoder across tasks. No TEST archive is opened.

The new work fits readouts on CPU, one thread. Ridge penalty is 1. The neural
head has 16 tanh hidden units, Adam 0.01 and 100 updates. Three declared head
initializations (2701/2802/2903, streamed by input width) are paired across the
two 32-number methods. All three fits are retained. Scores average over the
three head repetitions within a master and then equally over the three masters.
There are 384 unique validation examples per task, not nine independent datasets.

Raw inputs contain 288 observed values and 288 visibility flags; TRAIN-only
per-channel/feature scaling zero-fills missing values. Raw PCA fits only legal
TRAIN observations. Both compact methods have ridge 66/neural 562 parameters;
raw576 has ridge 1154/neural 9266. Classifier recipes and update budgets are fixed.

## Timing relationship

The task identifies the sign of the time lag between channels. **RPB-v4** is the
learned global bottleneck; it makes this relationship much more accessible to
the fixed heads.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 48.7 | 52.8 | 100 |
| PCA only — no encoder | 32 | 51.3 | 69.6 | 100 |
| RPB-v4 | 32 | 93.8 | 99.9 | 100 |

RPB-v4 exceeds PCA-only by 42.45 percentage points with the linear head and
30.30 points with the neural head. Native linear scores by master are 98.44%,
97.66% and 85.16%; neural means are 99.74%, 100% and 100%. This variation is a
reason to re-evaluate training duration on fresh development sources. Saved
source-group intervals are conditional on each fitted encoder/head and are not
an interval over retraining seeds.

These validation scores must not be read as a decline from the historical
94.79%/100% TEST result: the scoring split and declared head seed repetitions
differ. The encoder checkpoints and their exports are unchanged.

## Amplitude

This task distinguishes a source signal's amplitude. The encoder also exposes
this property much better than either trained raw-data control.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 60.9 | 63.9 | 100 |
| PCA only — no encoder | 32 | 55.2 | 64.8 | 100 |
| RPB-v4 | 32 | 100 | 99.7 | 100 |

## Direction

Direction is the reversal task. Every method solves this simple check.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 100 | 100 | 100 |
| PCA only — no encoder | 32 | 100 | 100 | 100 |
| RPB-v4 | 32 | 100 | 100 | 100 |

## Level

Every method also solves the level check.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 100 | 100 | 100 |
| PCA only — no encoder | 32 | 100 | 100 | 100 |
| RPB-v4 | 32 | 100 | 100 | 100 |

## Encoder training

Reused timing-checkpoint training measurements, averaged over the three masters.
Error is fixed-query standardized reconstruction MAE; lower is better. The new
readout experiment performs zero encoder updates and does not alter these errors.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 | 0.0693 | 0.0730 | 15.8 |

4096 sampled row presentations equal 16 equivalent passes over 256 available
TRAIN examples, with replacement and fresh masks; they are not 16 guaranteed
epochs. The saved optimization loss is masked Huber, distinct from fixed MAE.

## Engineering and evidence

The evaluator resides in shared code and links no encoder. It independently
tracks each method's supported TRAIN/VALIDATION rows, fits only on TRAIN, and
scores each paired comparison on its common valid validation population. An
unsupported native fit does not disable legal raw controls. Native exports are
copied unchanged and receive no PCA. All artifacts use a new directory.

Focused container tests passed for validation/hidden-value/label isolation,
paired initialization, unsupported PCA rank, unequal coverage, all-missing
observations, source overlap, malformed support, numerical overflow, exact
input preservation and ambient CPU/CUDA RNG restoration. The ordinary encoder
sources/configurations remain byte-identical to the saved milestone; only the
Makefile adds the new evaluator build targets.

The experiment keeps 12 cohorts, 36 head repetitions, 108 measured method fits,
48 immutable input archives and 447 generated artifacts plus the output manifest.
Native row identity is inherited from the audited original producer lineage:
the archived feature tensors contain no row IDs. Input declarations bind exact
file SHA256s, expected feature provenance, checkpoints, scalers and source order.
An independent artifact audit passed 28,174 checks, including all 216 prediction
archives, 432 recomputed head-score counts, 108 paired comparisons, input/output
hashes and the independently recomputed evaluator source fingerprint.
A companion preprocessing audit independently reproduced observed TRAIN counts,
legal normalized raw values/flags and native/raw TRAIN normalizer statistics.

Evidence:

- [Frozen experiment plan](../../../output/runs/archive-controls/phase1-6cbaa14e0d/experiment-plan.json)
- [Identity-bound input manifest](../../../output/runs/archive-controls/phase1-6cbaa14e0d/inputs.tsv)
- [Input audit](../../../output/runs/archive-controls/input-audit-20261006T191016Z-d7b99529/input-audit.json)
- [Instantiated card](../../../output/runs/archive-controls/archive-readout-BmAkXt/results/archive-readout-card.json)
- [Full report, scores and paired intervals](../../../output/runs/archive-controls/archive-readout-BmAkXt/results/report.json)
- [Input file hashes](../../../output/runs/archive-controls/archive-readout-BmAkXt/results/input-manifest.json)
- [Output artifact hashes](../../../output/runs/archive-controls/archive-readout-BmAkXt/results/output-manifest.json)
- [Independent artifact validation](../../../output/runs/archive-controls/phase1-validation-20261006T193142Z-6f14edc3/validation.json)
- [Independent TRAIN preprocessing validation](../../../output/runs/archive-controls/phase1-validation-20261006T193142Z-6f14edc3/training-preprocessing-validation.json)
- [Historical training measurements](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/diversity-128/validation-report.json)

Generated inputs and artifacts remain local under ignored `output/`; this
document records the reproducible conditions and measured summary.

## Next action

Implement the separate native-only training/selection protocol in phase 2,
then re-evaluate RPB-v4 on fresh development masters with milestones
0/128/512/2048. The zero point tests training benefit; a common positive budget
is selected by mean native32 linear validation accuracy, before independent
TEST sources are generated. Keep raw/PCA-only controls and head recipes fixed.
Verify actual CUDA updates and checkpoint/export parity before the full curve.
The weakest seed motivates this check; it does not yet justify changing the head
or bringing every historical encoder back. Consumer acceptance remains separate.
