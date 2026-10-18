# RPB v4 native curve advance

Date: 2026-10-06. Status: completed, fresh synthetic development experiment.
Protocol: native-curve-v1; evaluation policy 1.2. No consumer acceptance claim.

Phase 2 of [the continuation plan](NEXT_ADVANCE.md) is complete. The separate
native-only evaluator selected 512 updates from VALIDATION, then scored fresh
TEST sources. RPB-v4 reached 94.01% linear and 96.09% neural timing accuracy.
Training longer improved reconstruction but reduced timing validation accuracy.
The next encoder experiment should address consistency across trained instances
and moderate missingness, rather than increase the training budget again.

RPB-v4 is the learned global bottleneck: its exact native 32-number export is
also the sole reconstruction signal. No PCA follows an encoder, including in
budget selection. PCA is a standalone raw-data baseline. This changes the
evaluation protocol and trains new instances of the same design; it does not
introduce another architecture or change legacy checkpoint defaults.

## Conditions

The [frozen card](../../evaluation/cards/native_curve_v1.md) and launch plan were
saved before generation. Masters 3101/3202/3303 use C3/H32/F3, patch8 and 10%
natural missingness. Every master/task has 128 TRAIN source pairs (256 examples),
64 VALIDATION pairs (128 examples), and 64 fresh TEST pairs (128 examples).
Paired labels share the same source and visibility mask; split source groups
are disjoint. Each task trains its own encoder. These results do not establish
one universal embedding solving all four tasks.

Timing (`lag_sign`) has milestones 0/128/512/2048 on one continuous optimizer
path per master. Direction (`reversal`), level, and amplitude use fixed 0/512.
There are twelve persistent trainers, thirty checkpoint points, and 10,752
completed encoder updates. Point zero is each trainer's exact initialization,
with its own TRAIN-fitted readouts; it is a matched control, not another design.

Training uses batch8 on the RTX A2000 CUDA device in the existing managed
container. All 225,805 parameters are on CUDA. The separate four-update
engineering gate verified CUDA inputs/loss/optimizer state, finite gradients,
changed global-pooling and decoder weights, exact checkpoint parity, exact
global32 decoding, and immutable earlier snapshots. It generated no TEST data.
The resolved recipe uses mixer1/global-mode2, mask ratio0.25, masked Huber1,
AdamW0.001/weight-decay0.0001, clip1, and the unchanged learned-global template.

Linear means ridge penalty1. Neural means tanh16, Adam0.01, 100 updates. Three
declared head seeds 2701/2802/2903 are paired for equal-width inputs. Each head
fits separately on legal TRAIN labels; these repetitions are not independent
encoder trainings. Summaries average heads within a master, then the three
masters equally. Each task has 384 unique TEST examples, not 1,152 independent
examples from repeating heads.

Raw576 contains 288 observed values plus 288 original visibility flags, with
TRAIN-only observation scaling and zero-filled missing values. Raw/PCA controls
and their heads fit once per cohort and stay fixed across milestones and stress.
Native32 and PCA32 have 66 linear / 562 neural fitted parameters. Raw576 has
1,154 / 9,266 and is a larger-input reference. Classifier recipes were not tuned.

## Timing TEST

RPB-v4 uses the selected 512-update learned global bottleneck. All rows below
use the same fresh sources and fixed heads.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 46.61 | 50.61 | 100 |
| PCA only — no encoder | 32 | 51.82 | 79.51 | 100 |
| RPB-v4 | 32 | 94.01 | 96.09 | 100 |

The exact untrained RPB-v4 initialization scores 55.47% linear and 65.19%
neural, also at 100% coverage. Training adds 38.54 and 30.90 percentage points,
respectively. The visibility-only diagnostic scores 50% with either head.
The declared analytic rule using legal observed raw values scores 100% at full
coverage. That establishes solvability; it is not a trained classifier result.

RPB-v4's per-master scores expose the remaining consistency gap:

| Master | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: |
| 3101 | 100 | 100 | 100 |
| 3202 | 85.16 | 88.54 | 100 |
| 3303 | 96.88 | 99.74 | 100 |

Paired source-group intervals favor native32 over PCA32 for the linear head in
every master. One neural comparison, master 3202/rep1, has a 4.69-point effect
with interval [-2.34, 12.50], crossing zero. The intervals condition on each
fitted checkpoint/readout; they are not an across-retraining confidence interval.
Fresh TEST scores here cannot be read as a regression from phase 1's different
TRAIN/VALIDATION cohorts and archived checkpoints.

## Direction TEST

RPB-v4 is the learned global bottleneck trained for this task at fixed 512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 100 | 100 | 100 |
| PCA only — no encoder | 32 | 100 | 100 | 100 |
| RPB-v4 | 32 | 100 | 100 | 100 |

## Level TEST

RPB-v4 is the learned global bottleneck trained for this task at fixed 512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 100 | 100 | 100 |
| PCA only — no encoder | 32 | 100 | 100 | 100 |
| RPB-v4 | 32 | 100 | 100 | 100 |

Direction and level also reach 100% through their untrained RPB-v4 controls.
These saturated checks verify the pipeline and information availability; they
do not demonstrate a training gain.

## Amplitude TEST

RPB-v4 is the learned global bottleneck trained for this task at fixed 512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 61.46 | 63.80 | 100 |
| PCA only — no encoder | 32 | 54.95 | 67.80 | 100 |
| RPB-v4 | 32 | 100 | 100 | 100 |

Its untrained control already scores 97.40% linear / 96.79% neural. The amplitude
training gain is small compared with timing. All ordinary-task raw analytic
rules reach 100% accuracy and coverage; metadata-only heads remain at chance.

## Timing training and selection

RPB-v4 is the learned global bottleneck. Errors below are fixed-query
reconstruction MAE in frozen TRAIN-scaler standardized units, averaged over the
three timing trainers. GPU seconds are cumulative synchronized encoder training
time per trainer; they exclude heads, extraction, evaluation and file writes.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 0 | 0.9054 | 0.9052 | 0 |
| RPB-v4 | 128 | 0.4545 | 0.4638 | 3.83 |
| RPB-v4 | 512 | 0.0616 | 0.0635 | 15.33 |
| RPB-v4 | 2048 | 0.0484 | 0.0503 | 62.90 |

The fixed-query Huber means at 512 are TRAIN 0.003023 / VALIDATION 0.003247;
at 2048 they are 0.001875 / 0.002104. Actual stochastic minibatch optimization
loss and gradient traces are separately saved. Neither measurement substitutes
for representation accuracy. At batch8/256 TRAIN rows, 128/512/2048 updates
represent 4/16/64 equivalent sampled passes, not guaranteed complete epochs.

Selection used only native32 linear VALIDATION accuracy, first declared ridge
repetition per master, with unchanged support. All three positive budgets were
supported; none was excluded. Exact ties would choose the smaller positive
budget. Zero, neural, raw/PCA, reconstruction, TEST, and stress did not select.

| Updates | Mean linear VALIDATION % | Decision |
| --- | ---: | --- |
| 128 | 92.19 | Supported |
| 512 | 94.79 | Selected |
| 2048 | 90.10 | Supported |

Selection was durably saved before any TEST generation. TEST uses the frozen
`native-curve-v1/fresh-testing` namespace and seed
`stream_seed(master, 0x6e63763174657374ULL)`. The engine verified the chosen and
point-zero snapshots after later training and reused their exact retained
TRAIN-fitted heads. No TEST readout refitting occurred.

Reconstruction improved in every timing master between 512 and 2048 while linear
validation accuracy declined. Both TRAIN and VALIDATION reconstruction improve;
this does not establish reconstruction overfitting. Better reconstruction alone
does not guarantee that a fixed head can recover the timing relationship.

## Timing missingness

These are selected RPB-v4 TEST readouts under fixed additional coordinate
deletion, beyond the ordinary 10% natural missingness. Every method and source
pair receives the same corruption; no component is refitted under stress.

| Extra deletion | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: |
| None | 94.01 | 96.09 | 100 |
| 10% | 92.97 | 95.14 | 100 |
| 30% | 87.76 | 92.19 | 100 |
| 60% | 59.64 | 73.35 | 100 |
| 90% | 50.78 | 50.00 | 100 |
| All observations absent | N/A | N/A | 0 |

All-absent signal rows abstain; conditional accuracy is undefined, and
full-population correctness is zero. At 60% extra deletion, the raw analytic
rule has 100% conditional accuracy on only 34.38% coverage. At 90% it has no
supported examples. Thus 100% encoder coverage at heavy deletion means it can
emit a vector, not that the task retains enough information for perfect answers.
At 30% extra deletion, the analytic rule still has 100% accuracy and coverage;
this is a useful remaining robustness target. This is approximately 37% expected
total missingness, including the natural mask. Mean degradation does not imply
a statistically established decrement in every master.

Removing either of the two timing-bearing channels reduces accuracy to chance.
Removing the third channel gives 94.53% linear / 97.83% neural. The complete
twelve-case stress report also contains contiguous blackout 25/50/75% and all
channel-absence cases. Blackout results are not uniformly monotonic; the full
case-level artifacts govern any interpretation.

## Evidence and verification

The local capsule is
[`output/runs/rpb-native-curve/native-curve-5G8O5c`](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/).
Generated checkpoints/data remain local under the ignored output directory;
this measured summary and the implementation are tracked in Git.

- [CUDA gate](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/results-cuda-gate/gpu-check.json).
- [Pre-generation launch plan](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/results.launch-plan.json)
  and [instantiated card](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/results/native-curve-card.json).
- [Durable selection](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/results/selection.json),
  [thirty validation points](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/results/validation-report.json),
  [fresh TEST results](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/results/report.json),
  and [stress results](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/results/stress-report.json).
- [Artifact SHA256 inventory](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/artifact-integrity.json):
  4,596 files, excluding the inventory itself. Inventory SHA256
  `c9922d3c817630da3d8609b7ba8d2b47cab7434a17fe28bf5ae272a841da0d95`.
- [Independent audit](../../../output/runs/rpb-native-curve/audit-tools/independent-native-20261006-v1/run-5G8O5c/validation.json):
  passed 70,619,069 checks with no repair/retry. Audit JSON SHA256
  `fce69cb11e56886e91f32e1278264cc6292d0c1964388da40ebe96567a3c6d49`.

The exact 49-file compiled source snapshot has fingerprint
`5a2c39d343e399c549b89f880c7e3a9e4b823d17b541775b7352c2de6ce8d643`.
The binary recorded Git HEAD `d3719b4d25de3e8d3baddf5e5d0cc92e56b65849`
and a dirty tree because the new native-curve source had not yet been committed.
The preserved source snapshot, not the subsequent documentation commit, names
the measured executable. Each cohort records exact resolved settings, training
dataset/scaler IDs, optimizer counters, checkpoint producer identities and hashes.

Selected timing checkpoint SHA256 values, all at 512:

| Master | Checkpoint SHA256 |
| --- | --- |
| 3101 | `71db828e63884956024a20b8223ef89a3da1b3f031a7fe84d36cc6c1cdac351b` |
| 3202 | `8cbce9f90c3c36146c2102955b279ab587b9569a0519f7815dfd0eec9e44ab73` |
| 3303 | `8bdf10f69f7c927546707f9fcabf6269d4401fb63425fb1da5c4877c9b303904` |

Focused container tests passed for the generic engine and encoder CUDA gate.
They cover TRAIN-only fitting, selection before TEST, unsupported/tied budgets,
fit-once controls, native exports without PCA, continuous optimizer/RNG isolation,
immutable snapshots, fixed-readout stress, and exact intact prediction parity.
The independent auditor checked 12 cohorts, 30 points, 198 TRAIN-only fits,
576 prediction archives, 108 fresh paired comparisons, 432 stress cases,
2,160 stress prediction archives, and 72 reconstruction archives. It recomputed
compact-head inference, preprocessing, scores, masks, and reconstruction reductions.
It did not replay bootstrap draws, execute CUDA checkpoints, or independently
replay full-width raw/mask head inference. Actual CUDA checkpoint execution was
covered by the C++ gate. Uncertainty remains within master, conditional on fitted
assets; consumer chronology and acceptance thresholds remain undefined.

## Next action

Phase 3 will test one encoder mechanism: learned global pooling directly over
aligned patch/channel states, before separate channel summaries compress them.
RPB-v4 already mixes channels at aligned patches; this proposal bypasses its
subsequent per-channel summary compression. The hypothesis is that this retains
joint temporal information more consistently at the served 32-number bottleneck. The
measurements do not establish that compression caused the remaining errors.
Register a new design tag before measuring it; do not relabel RPB-v4. There is
no measured claim for this candidate yet.

Keep native32, the reconstruction objective, fixed heads, and 512 encoder updates
for the first paired comparison. Retain the selected RPB-v4 checkpoints and
their TRAIN-fitted assets as the relevant reference; do not routinely retrain all
older designs. Primary comparisons use native linear accuracy and per-master
spread. The already declared 30% deletion case is the moderate robustness target;
neural accuracy remains secondary. Avoid making perfect-score demands after
removing task-essential information.

Freeze the candidate card and a new TEST namespace before candidate measurements.
The TEST sources opened here are now development evidence and cannot serve as
unseen confirmation for a design chosen from this report. Legal TRAIN/VALIDATION
cohorts may be reused for paired development; new TEST sources must be common
to candidate and frozen RPB-v4, with no test fitting or classifier changes.
If timing and moderate missingness improve without an unacceptable reconstruction
or cost tradeoff, move to a declared harder or consumer task instead of further
optimizing saturated direction/level scores.
