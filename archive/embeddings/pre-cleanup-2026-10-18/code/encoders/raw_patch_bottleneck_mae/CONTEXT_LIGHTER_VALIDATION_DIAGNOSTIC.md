# RPB-v7 lighter context deletion validation diagnostic

Date: 2026-10-07. Status: measured and independently audited; not promoted.
Protocol: `context-lighter-validation-v1`; policy 1.2; development only.

Lighter context deletion improves mean native linear timing accuracy in both
known VALIDATION views and lowers reconstruction error relative to RPB-v6.
It still exceeds paired RPB-v4 reconstruction error on every master, so it
fails the joint development question. No TEST or historical stress payload
was accessed; no checkpoint, deletion rate or head was selected or promoted.
RPB-v4 remains the active experimental reference. Numerical results below
come from the completed C++ measurement and passed independent archive audit v3.

## Question, conditions and model descriptions

The question is whether requesting less extra context deletion preserves the
RPB-v6 timing representation while recovering RPB-v4 reconstruction at the same
512-update budget. Use timing (`lag_sign`) on the five known masters
4404/5505/6606/7707/8808. Each master has 128 original TRAIN source pairs
(256 examples) and 64 known VALIDATION pairs (128 examples). Opposite-label
variants remain in the same source split. These observations and masters are
already known development data, not fresh confirmation.

RPB-v4 is the learned global bottleneck without extra training context deletion.
RPB-v6 has that same inference architecture with a fixed 0.30 training request
rate. RPB-v7 retains the architecture and requests 0.15 instead. All three serve
the exact native 32-number global vector and decode solely through it. All have
225,805 parameters, mode 2, one aligned channel mixer, C3/H32/F3, patch 8,
width 64, three temporal blocks, four attention heads, feed-forward 256 and
decoder width 128. No PCA or random projection follows an encoder.

Five RPB-v7 encoder trajectories were newly trained on actual CUDA; ten RPB-v4/
RPB-v6 trajectories are retained from the completed five-master replication.
Each new trajectory saves point 0 and 512. Those two points and the three
readout repetitions do not add independent encoder runs. Preserve batch size 8,
10% natural missingness, AdamW learning rate 0.001, weight decay 0.0001,
clipping 1 and dropout 0. At 512 unskipped updates, 4,096 sampled rows give
16 equivalent presentations of the 256 TRAIN examples, not guaranteed epochs.

The sole new policy is `rpb-training-context-deletion-015-v1`. It uses the
unchanged label-independent context stream and semantic-canonical coordinates.
Requests apply only to original visible context; deterministic repair clears
only extra deletion to retain two original visible patch groups in eligible
channels. Original observed support, artificial queries, targets, eligibility,
hierarchical Huber 1, TRAIN scaler and row/mask/Torch streams stay unchanged.
Typed CPU checkpoint companions retain cumulative requested, actual and
restored coordinate counts; the 0.15 request threshold is not an exact realized
deletion fraction. Inference applies neither deletion nor repair.

The [frozen card](../../evaluation/cards/context_lighter_validation_v1.md)
defines admission, permitted roles, fitting and decision requirements.

## Fixed fitting and validation views

The linear head is ridge with penalty 1. The neural head has 16 tanh hidden
units, Adam learning rate 0.01 and 100 updates. Retain all repetitions
2701/2802/2903; actual equal-width seeds are paired. Native32 and standalone
PCA32 have 66 linear and 562 neural fitted parameters. Raw576 has 1,154 and
9,266. Means average repetitions within each master, then masters equally.

Raw576 contains 288 observed TRAIN-scaled values plus 288 visibility flags.
Its observed TRAIN scaler uses floor 1e-8. The standalone PCA-only baseline
fits raw TRAIN and supplies 32 components. Unsupported PCA fits retain their
reason without disabling legal native/raw fits.

The diagnostic makes 15 positive-point archive inputs: five masters times
RPB-v4/RPB-v6/RPB-v7. It transparently refits ordinary TRAIN maps and readouts:
45 fitted pipelines per method, each containing both heads. All 135 pipelines
are supported. Native has 45 old/cache fit-parity records. These are new diagnostic fits,
not zero-constructor reuse of old heads. Each fitted pipeline scores intact
VALIDATION and the one additional-deletion view with no held-out refit.
New raw/PCA maps, fits and predictions are checked for exact equality across
versions within each master; this does not claim parity with unlisted old
raw/PCA assets.

One fixed additional-30% VALIDATION mask is shared across paired variants,
versions and repetitions. Namespace: `context-lighter-validation-v1`;
stream: `0x636c763164726f70`; view: `validation-dropout-030`. It deletes only
observed coordinates, retains remaining values exactly and zeros hidden
storage. It has no repair or support restoration. This new view differs from
every historical TEST/stress view. Its score must not be presented as the
historical replication score.

## Intact timing quality on known VALIDATION

All methods retain 100% coverage: 128 VALIDATION examples per master, 640
across the five masters. Full-population correctness therefore equals the
conditional accuracy below. Raw/PCA scores are identical across versions;
their repeated identical copies do not add encoder runs or independent data.
RPB-v4 is the learned global32 bottleneck; RPB-v6 adds 0.30 training context
deletion; RPB-v7 changes that request rate to 0.15 with the same inference model.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.69 | 50.83 | 100 |
| PCA only — no encoder | 32 | 49.06 | 70.99 | 100 |
| RPB-v4 | 32 | 93.59 | 95.99 | 100 |
| RPB-v6 | 32 | 95.31 | 99.74 | 100 |
| RPB-v7 | 32 | 97.97 | 98.96 | 100 |

RPB-v7 minus RPB-v6 is +2.65625 percentage points for the linear head and
−0.78125 for the neural head. Native32 linear accessibility improves on this
known cohort; the neural result does not show a corresponding mean improvement.

## Fixed additional 30% deletion on known VALIDATION

The same ordinary TRAIN-fitted maps and heads score this view. Coverage stays
100% for every method, master and repetition. The new mask is distinct from
the historical replication TEST corruption and its scores are not interchangeable.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 52.03 | 48.65 | 100 |
| PCA only — no encoder | 32 | 49.84 | 54.79 | 100 |
| RPB-v4 | 32 | 84.38 | 91.46 | 100 |
| RPB-v6 | 32 | 93.91 | 97.86 | 100 |
| RPB-v7 | 32 | 95.16 | 96.25 | 100 |

RPB-v7 minus RPB-v6 is +1.25 percentage points linear and −1.614583333 neural.
No corrupted-VALIDATION values or labels fit these heads or their maps.

## Per-master effects and worst-master quality

Each model cell below is linear / neural accuracy %, averaging all three
declared head repetitions. The linear head is deterministic across those
repetitions; paired intervals remain separately retained by repetition.
Delta columns are linear effects in percentage points.

Intact known VALIDATION:

| Master | RPB-v4 | RPB-v6 | RPB-v7 | v7 − v4 | v7 − v6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4404 | 98.44 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | +1.5625 | 0 |
| 5505 | 91.41 / 92.45 | 92.19 / 99.74 | 93.75 / 96.09 | +2.34375 | +1.5625 |
| 6606 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 0 | 0 |
| 7707 | 89.06 / 94.79 | 85.94 / 98.96 | 97.66 / 100.00 | +8.59375 | +11.71875 |
| 8808 | 89.06 / 92.71 | 98.44 / 100.00 | 98.44 / 98.70 | +9.375 | 0 |

Fixed additional 30% deletion:

| Master | RPB-v4 | RPB-v6 | RPB-v7 | v7 − v4 | v7 − v6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4404 | 68.75 / 94.01 | 97.66 / 100.00 | 98.44 / 100.00 | +29.6875 | +0.78125 |
| 5505 | 89.06 / 89.58 | 92.97 / 96.61 | 94.53 / 93.75 | +5.46875 | +1.5625 |
| 6606 | 97.66 / 99.48 | 97.66 / 100.00 | 98.44 / 97.92 | +0.78125 | +0.78125 |
| 7707 | 86.72 / 89.58 | 84.38 / 92.71 | 92.97 / 100.00 | +6.25 | +8.59375 |
| 8808 | 79.69 / 84.64 | 96.88 / 100.00 | 91.41 / 89.58 | +11.71875 | −5.46875 |

The worst-master linear scores improve from RPB-v6's 85.9375% / 84.375%
to RPB-v7's 93.75% / 91.40625% (intact / deleted), but this is not uniform
per-master improvement. Master 8808 loses 5.46875 points under deletion,
and its mean neural deletion accuracy falls from 100% to 89.583333333%.
Worst-master neural scores fall from 98.958333333% / 92.708333333% to
96.09375% / 89.583333333%.

For example, repetition 1's conditional 95% source-group linear intervals
for v7 minus v6 are:

| Master | Intact effect [interval], pp | Deleted effect [interval], pp |
| --- | ---: | ---: |
| 4404 | 0 [0, 0] | +0.78125 [−2.34375, +3.125] |
| 5505 | +1.5625 [−5.46875, +7.8125] | +1.5625 [−3.90625, +7.03125] |
| 6606 | 0 [0, 0] | +0.78125 [−1.5625, +3.125] |
| 7707 | +11.71875 [+6.25, +17.1875] | +8.59375 [+1.5625, +15.625] |
| 8808 | 0 [−3.125, +3.125] | −5.46875 [−10.9375, +0.78125] |

These examples are not selected favorable repetitions or averaged intervals.
The [complete paired report](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/paired-validation-report.json)
retains every repetition, both comparators, both heads and conditional/full-
population intervals. Most small positive linear effects cross zero; the
7707 gain does not establish a universal benefit across masters.

The complete matrix has 60 version/view/repetition comparison records
(5 masters × 2 views × 3 repetitions × 2 comparators). Each contains both
heads, giving 120 paired head effects. Source-group intervals use exact
row predictions on named common support within a master. Full-population
correctness counts abstentions as failures. These intervals condition on
identified fits; they are not uncertainty across five encoder runs or three
head repetitions. Crossing-zero intervals must remain visible.

## Reconstruction and training cost

Error means standardized fixed-query MAE on original
TRAIN and intact VALIDATION, through ordinary unaugmented native32 inference.
Enumerate the four original hidden-patch queries, target legally observed cells
and preserve equal cell → channel → example reduction and the original TRAIN
scale. Retain Huber, query/target/visibility/eligibility arrays and the actual
random-minibatch optimization trace separately. The classification deletion
view does not redefine reconstruction targets.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 | 0.067593 | 0.071309 | 13.85 |
| RPB-v6 | 512 | 0.091179 | 0.092984 | 15.06 |
| RPB-v7 | 512 | 0.081179 | 0.083630 | 15.69 |

Mean RPB-v7 error improves over RPB-v6 by 0.010000268211 on TRAIN and
0.009354085303 on VALIDATION, but remains above RPB-v4 by 0.013585551816
and 0.012320803007 respectively. Both reconstruction guards fail. All five
individual RPB-v7 errors exceed their paired RPB-v4 errors; two masters also
have slightly worse reconstruction than RPB-v6. Lighter training context deletion reduces
the mean tradeoff but does not remove it.

| Master | v4 TRAIN / VAL MAE | v6 TRAIN / VAL MAE | v7 TRAIN / VAL MAE | v7 GPU seconds |
| --- | ---: | ---: | ---: | ---: |
| 4404 | 0.072697 / 0.075390 | 0.097731 / 0.098094 | 0.076529 / 0.078533 | 16.53 |
| 5505 | 0.059514 / 0.064164 | 0.075800 / 0.078413 | 0.079698 / 0.083926 | 15.87 |
| 6606 | 0.062540 / 0.065255 | 0.102393 / 0.104031 | 0.087755 / 0.090371 | 15.15 |
| 7707 | 0.080216 / 0.085018 | 0.082619 / 0.084234 | 0.082668 / 0.085783 | 15.41 |
| 8808 | 0.063000 / 0.066718 | 0.097353 / 0.100148 | 0.079243 / 0.079538 | 15.49 |

RPB-v4/RPB-v6 training costs are retained parent measurements; RPB-v7 cost is
new. Synchronized loop timing includes training preparation/transfers and
bookkeeping and excludes loading, extraction, reconstruction, classifier
fitting/scoring and artifact writing. It is not GPU-kernel-only or an inference
cost benchmark. New v7 loop times span 15.153382946–16.534471545 seconds;
retained v4/v6 timing is not a contemporaneous repeated speed comparison.

## Evidence and independent audit

Completed capsule:
[lighter-validation-duyRU2](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/).
Its [launch plan](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results.launch-plan.json)
and [instantiated card](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/context-lighter-validation-card.json)
precede measurements. See the
[readout report](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/readouts/report.json),
[fixed-query report](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/report.json),
[native refit parity](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/refit-parity.json),
[raw/PCA parity](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/raw-pca-parity.json)
and [completion record](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/complete.json).
The production record reports ten candidate development points, 15 positive
inputs, all 135 supported pipelines, 45 native parity checks, 60 comparison
records and 120 effects.

The [parent-input manifest](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/parent-inputs.json)
explicitly permits 313 parent TRAIN/VALIDATION-related roles,
including identified checkpoints, companions, manifests, native features,
ordinary fits/predictions and fixed-query arrays. Role hashes are pinned before
access and preserved after it. No recursive payload discovery or parent TEST/
stress/report access is authorized. Pregenerated observations/order must match
both retained parents before fitting; point 0 must match all initialized
parameters, buffers, scalers and original streams. Candidate positive counters
all satisfy attempted = completed = 512 and sampled rows = 4,096, with 225,805
CUDA parameters. The production
[input byte guard](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/input-integrity-after.json)
and outer [preservation record](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/parent-preserved-after.txt)
retain original allowed-role bytes and hashes. The
[capsule inventory](../../../output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/artifact-integrity.json)
binds the new outputs and source/admission copies.

| Identity | SHA256 |
| --- | --- |
| New orchestration source | `f22cc8d5f2ac8af3e6a6056c6ad2693ffcc82cf39e8744c32fb13038710c5311` |
| New training producer | `d8ed2771d1615beabccbead081847813b2562b7b1f9b5347d4825c075ccc27a7` |
| New core writer | `e379b8f8abb102c256fb843466b4fbfbecb6fdc21298b862fd0b6dfeb47b853e` |
| Passed admission record 5LwmCY | `902dd80940c38b695e2dbb88a1090c586ff01b395cc2c83f16babecdfed6f68b` |
| Sealed independent auditor v3 | `a75d0a5b96e87858726bacb86c691d296b762742265994c2e2fe6f5bcf810bda` |
| Passed independent validation record | `c01050a083a52b2535a409ae7b7e7b7436a373d834534606b3754c74e8412115` |
| New capsule inventory | `d4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821` |
| Frozen human card | `7a22817ff88240f95a378aa90617191eb5dc032595123a127fcf09f9c5c01240` |

The [independent validation record](../../../output/runs/rpb-context-lighter-validation/audit-tools/run-duyRU2-v3/validation.json)
passes 67,097,380 numerical and provenance checks. It verifies 63 production
source files, 313 identified parent roles, 10 candidate development points,
15 positive inputs, 135 TRAIN fits, 405 prediction archives, 45 native parity
records, 50 reconstruction archives and all 60 comparison records/120 effects.
It independently regenerates 92,160 source-shared corruption draws and checks
ordinary/deleted support, fixed-query arithmetic and saved head inputs/logits/
predictions. Parent roles, source/admission copies, input-preservation records
and the 1,483-file/547,772,582-byte capsule inventory are bound by file hashes.
V1/v2 reader source versions were preserved without payload execution; v3's
numeric acceptance predicates were sealed before this audit.

The audit does not rerun encoder or classifier optimization or deserialize the
GPU model. Full initialization/gradient/update and checkpoint continuation
are established by the separately source/log-bound actual CUDA admission.
Continuation here means the live trainer across saved immutable snapshots,
not reloading a v7 checkpoint through ordinary workflow resume; that path rejects
nonempty training-policy identities.
Grouped bootstrap draws and bounds are not independently regenerated: the
source counts, recipe, point estimates and finite ordered bounds are checked.
Intervals remain conditional within masters, with no across-training interval
claim. Source/role guards do not supply an operating-system access trace, and
tensor-check counts do not represent independent statistical trials.
The execution records compiled Git HEAD `ee514aa` with a dirty working tree;
its exact source capsule and fingerprints identify the measured code. A later
commit saves that source rather than retroactively becoming its build identity.

The exclusive [audit replay bundle](../../../output/runs/rpb-context-lighter-validation/audit-tools/replay-bundle-duyRU2-complete-20261007T040729Z-62189f01/REPLAY.md)
preserves the exact v3 entrypoint, six helper sources and unchanged passed
validation JSON outside the measured capsule. Its source-only imports pass
with zero payload reads. Inventory SHA256 is
`628b65139b621199bc9f3aafafd0f6b0ed757f61c8b36293b4ea99e949d8ef3b`.
Replay still requires the named local data capsules and documented helper-path
resolution; the bundle does not contain the data. The
[durable numerical summary](../../../doc/results/context_lighter_validation_v1.json)
retains exact means, all masters, training scope and audit identities in source
control; large generated checkpoints and archives remain local under `output/`.
Its SHA256 is `1f9e9c0124fc4fe77d284d8b4e7f2c0cfd9649cad04999da1328ae8db7b3b98a`.

## Decision and conditional next action

The frozen development question requires RPB-v7 mean native ridge
scores no lower than paired RPB-v6 in both views with equal coverage, mean
original TRAIN and VALIDATION MAE each no worse than paired RPB-v4, and all
admission/preservation/audit gates. RPB-v7 meets the mean linear score/coverage
part and fails both reconstruction guards. Preserve individual-master effects and
worst-master changes. Neural-only gains or a reconstruction guard waiver do
not satisfy that question. Even a pass is known-VALIDATION evidence requiring
a separate explicit decision and frozen plan before fresh confirmation.

Lighter deletion fails the joint reconstruction/quality question. A prospective
next hypothesis is a fixed balanced mixture of ordinary context
and the existing 0.30-deletion view at the same 512 updates: exactly 256 updates
of each, retaining architecture, original targets/loss, streams and heads.
Clean-context updates might anchor reconstruction while strong corrupted views
retain a robustness incentive. This is unimplemented and unmeasured; halved
corruption exposure could instead weaken robustness. Freeze one schedule and
checkpoint-policy identity before any test, rather than search rates or
budgets. The prior longer-training diagnostic does not support more budget
alone as the joint remedy. The
[implementation plan](BALANCED_CONTEXT_IMPLEMENTATION_PLAN.md) specifies the
bounded follow-up and its separately frozen card/admission prerequisites.
This proposal selects or promotes no model and opens no TEST.
