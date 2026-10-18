# Learned ordered relation development screen

RPB-v21 is a separate candidate. It keeps the RPB-v20 backbone, configurable
geometry, masks, scaler and fixed spectral coordinates. A zero-start projection
adds generic ordered lag comparisons to its learned coordinates. The working
RPB-v20.alt-01 instances and historical evidence remain preserved.

The [saved TRAIN diagnosis](LEARNED_TRAIN_V1.md) motivated this change: the
learned branch varied substantially after training, but paired examples with
opposite timing remained much less separated than in the fixed coordinates.
This was a diagnosis of the saved observations, without an encoder call,
classifier fit or validation payload read.

## Architecture and controlled comparison

The [portable module](../code/encoders/raw_patch_bottleneck_mae/LEARNED_ORDERED_RELATION.md)
uses original-tick lags 1/2/4/8, canonical semantic channel pairs and every
feature pair. Odd, even and common-support measurements use only jointly
observed endpoints. Moments are accumulated on CUDA in float64; the resulting
bank and trainable projection use float32. The projection has 58,320 weights
at C3/H128/F9/D32 and starts at zero without consuming an initialization draw.

Candidate0 must export exactly the same native32 values and validity as the
saved RPB-v20.alt-01 control. Every fixed12 coordinate must stay exact at both
retained points and all observation views. Training retains waveform Huber,
the original row and mask streams, batch 8 and 512 AdamW updates on CUDA.
Labels are excluded from the optimizer.

The [prospective card](../code/evaluation/cards/configurable_learned_relation_v1.md)
declares the first two known development cohorts, matched controls, fixed head
recipes and numerical gate before quality execution. A complete joint pass
alone allocates the third known cohort. There is no fresh TEST, market quality
claim, reference promotion or serving default change.

The [separate runner and reader](../code/evaluation/protocols/configurable_learned_relation_v1/README.md)
fit only trained native32 and learned20 readouts. Initialized readouts and
trained RPB-v20 controls are reused from saved artifacts. The publisher checks
their arithmetic, normalization, predictions and provenance independently;
it does not run an encoder or fit a classifier.

## Engineering evidence

Artificial CUDA engineering passed before the quality experiment. It covers
generic geometry, masks, reversal, scalar relation witnesses, zero-start output
and RNG parity, actual new-weight gradients and updates, typed checkpoint
reload and malformed-state rejection. Ordinary CPU arithmetic witnesses are
separate from model execution; the encoder runs on CUDA.

Both engineering attempts are retained under
[the durable engineering receipts](results/configurable_learned_relation_v1/engineering/).
The first attempt stopped on an overstrict storage-permutation assertion:
the inherited backbone differed by 2.98e-08 and the candidate by 2.38e-07.
The canonical relation bank itself stayed byte-exact. The corrected test uses
a tight numerical bound for that inherited permutation behavior, while
identical-storage zero-start parity remains byte-exact. No model arithmetic
changed between these two attempts. The passed capsule is admission-BUMBAL;
the failed capsule is admission-cDE5er. Original binaries remain in ignored
local output; their hashes and exact source snapshots are durable.

## Quality evidence

The completed two-cohort screen **fails** its prospective gate: 45 of 54
conditions pass; nine fail, all involving deleted-view readouts. No third
cohort was allocated. RPB-v20.alt-01 remains the working configuration.

RPB-v21.alt-01 adds the learned ordered lag projection. RPB-v20.alt-01 is the
preserved compact spectral encoder. Size32 below is the complete embedding;
size20 is only its learned coordinate slice, a diagnostic readout. Initial
learned20 controls were approximately 50%; they were reused only after exact
candidate0 export equality. The complete initialized v20 control was already
94.92%/93.55% on intact inputs and 88.67%/86.78% with extra deletion.

**TEMPO-5, complexity5/5:** slow lead/lag beneath fast rhythms, nine grouped
features and structured distractors. Intact VALIDATION, equal means across
masters 941101/941102 and head repetitions 2701/2802/2903.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 95.12 | 94.92 | 100.00 |
| RPB-v21.alt-01 | 32 | 97.85 | 96.48 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 50.26 | 100.00 |
| RPB-v21.alt-01 | 20 | 62.50 | 61.00 | 100.00 |

**TEMPO-5, complexity5/5:** the same structured slow timing task, VALIDATION
with additional 30% coordinate deletion, the same two cohorts and repetitions.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 88.67 | 87.89 | 100.00 |
| RPB-v21.alt-01 | 32 | 65.43 | 78.06 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 49.87 | 100.00 |
| RPB-v21.alt-01 | 20 | 61.72 | 58.40 | 100.00 |

**TEMPO-5, complexity5/5:** fixed original hidden-patch query errors under
TRAIN-fitted scales, equal two-cohort means; lower MAE is better.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 512 | 0.78174 | 0.80302 | 20.6 |
| RPB-v21.alt-01 | 512 | 0.78891 | 0.81042 | 43.9 |

These v20 results are matched saved controls, not reruns. New v21 evidence
uses two independent encoder trajectories, each 512 completed AdamW updates
on CUDA:0 float32 with batch 8, C3/H128/F9/P8/W64/D32. Each cohort has 512 TRAIN
rows/256 source pairs and 256 VALIDATION rows/128 source pairs. Heads retain
ridge penalty 1 and neural 16-tanh / Adam 0.01 / 100 updates; three head repetitions
are not three encoder retrainings. No post-encoder PCA or raw/PCA fits occurred.

GPU training took 27.1/60.6 seconds for the two new runs. Query capture plus
transfer/output took 88.0/359.0 seconds; feature extraction plus transfer/output
took 7.5/16.6 seconds. These descriptive costs are separate from training and
CPU classifiers. The slowdown affected compute and transfer measurements;
the evidence does not establish a sole cause. Encoder CPU calls are zero.

Reconstruction ratios against the matched v20 means are 1.00917 TRAIN and
1.00922 VALIDATION, within the prospective protection margins. Initial/native
equality covers 65,536 cells; fixed-coordinate equality covers 49,152 cells at
both retained points. The publisher checked 12,848,191 conditions over 476
consumed files: 12 new fit pairs/24 classifiers, 30 reused fit pairs checked
arithmetically, 36 new prediction archives and 48 paired comparisons.

The learned coordinates improved, but the complete embedding lost 23.24/9.83
percentage points under extra deletion. Unchanged fixed12 does not guarantee
that a classifier using all 32 coordinates will keep using that robust signal.
This is a failure of this architecture/training pair under the fixed evaluation;
it does not damage the preserved working encoder or prove intrinsic information
loss. Conditional source-pair intervals do not measure encoder retraining
uncertainty. Stop this frozen recipe without tuning its heads, rates or budget.

## Reproducible evidence

The full local capsule is
`output/runs/rpb-v21-learned-relation/admission-0PGNmn`, completed as
`screen-fail-stop`. The original 815-artifact capacity inventory and the saved
branch inventory passed their final preservation checks. Frozen source
fingerprint is `dbe1c04cc3246655fa5419520455000e6ee97068b3b51c19e76c7cb5d9020110`;
card SHA256 is `c36ebc4517bb021de44f9762e1c3ba55ef04d7b122a08db71e5a1e524cd45b86`.
Full artifact manifest SHA256 is
`6b138475bbeac3bde35cc45f331f21f5e10cff64b5971fb883f60121c20562bf`.

The [saved summary](results/configurable_learned_relation_v1.json),
[198 exact non-tensor capsule files](results/configurable_learned_relation_v1/completed/),
[independent JSON audit](results/configurable_learned_relation_v1/independent-audit.json)
and [instance registry](configurable_learned_relation_embedding_versions.json)
retain scores, source/card/config/input/binary receipts, fitted arithmetic,
logs and checkpoint identities. Tensor arrays, weights and compiled binaries
remain in ignored local output; they were not removed or overwritten.

## Continuation

Follow the [13 October continuation](CONTINUATION_2026-10-13_AFTER_LEARNED_RELATION.md).
The exploratory [TRAIN support audit](results/configurable_learned_relation_v1/train-support-audit/README.md)
finds an active observation-density offset that cannot distinguish same-source
timing pairs. This supports testing a separate content-only relation architecture
that keeps support for validity/normalization and excludes additive density
inputs. It is a mechanism hypothesis, not a causal attribution. RPB-v22 remains
prospective. A learned-only decoder would address a different question.

## Repository boundary

Research protocols and synthetic evaluation stay in cuwacunu_embedding.
cuwacunu keeps its selected RPB-v20 implementation and settings. A failed
candidate stays research-only. A successful candidate may be mirrored as a
separate portable API after review, without changing the serving default.
