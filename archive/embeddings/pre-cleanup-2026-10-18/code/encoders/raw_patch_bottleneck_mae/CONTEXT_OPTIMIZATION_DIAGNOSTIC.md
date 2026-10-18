# RPB-v6 context optimization diagnostic

Date: 2026-10-07. Status: measured; independent archive audit v2 passed.
Protocol: `context-optimization-validation-v1`; policy 1.2; development only.

More training improves reconstruction, but none of the measured budgets both
reaches the historical reconstruction reference and preserves the replayed
512-update linear timing accuracy in both validation conditions. The original
[context-deletion result](CONTEXT_DELETION_ADVANCE.md) and its unresolved
reconstruction tradeoff remain unchanged. No checkpoint was selected or promoted;
RPB-v4 remains active. This diagnostic accessed no TEST or old stress payload.

## Conditions and unchanged training

Measure timing (`lag_sign`) on the three original masters 3101/3202/3303.
Each uses exactly 128 TRAIN source pairs (256 examples) and 64 known VALIDATION
pairs (128 examples). Opposite-label variants stay together in their source
split. The nine checkpoints are three points on three replayed trajectories,
not nine independently trained encoders or new-master confirmation. Three
classifier repetitions on each checkpoint also do not add encoder runs.

RPB-v6 is the RPB-v4 inference architecture with additional training context
deletion: mode 2, one aligned channel mixer and the exact served global 32-number
decoder input. Preserve all 225,805 parameters, C3/H32/F3, patch 8, width 64,
three temporal blocks, four heads, feed-forward 256, decoder 128 and dropout 0.
Natural missingness remains 10%, batch size 8, and updates use actual CUDA.
The AdamW recipe stays at learning rate 0.001, weight decay 0.0001 and clipping 1.

The unchanged policy `rpb-training-context-deletion-v1` requests 30% extra
coordinate deletion only from original visible context, using the independent
counter stream and semantic-canonical coordinate order. It repairs only extra
deletion to retain two original visible patch groups in originally eligible
channels. Original observed targets, whole-patch queries, eligibility,
hierarchical Huber 1 reductions, scaler and row/mask/Torch streams stay exact.
Frozen inference applies neither training deletion nor repair. The
[frozen diagnostic card](../../evaluation/cards/context_optimization_validation_v1.md)
records the complete recipe and permitted input roles.

Ordinary training still rejects the tagged checkpoint. A new wrapper uses the
existing fresh trainer unchanged, with only its absolute step ceiling extended
to 2048. It replays 0 -> 512 and requires exact parent agreement before later
updates: named parameters and buffers, active named AdamW steps/moments, complete
TRAIN scaler, dataset/schema, ordered sources, semantic IDs, attempts, completed
updates, sampled rows and cumulative context request/deletion/repair counts.
New replayed native TRAIN/VALIDATION exports and fixed-query arrays also reproduce
the original 512 point. The same live optimizer then advances to 1024 and 2048.
The captured 512 snapshot remains unchanged after continuation.

Attempted equals completed at all nine points. Per trajectory, 512/1024/2048
updates sample 4096/8192/16384 rows with replacement: 16/32/64 equivalent
presentations of the 256 TRAIN examples, not guaranteed complete epochs.
Every budget, master and repetition is retained. No new architecture, objective,
mask policy or classifier recipe was introduced.

## Fixed heads and validation views

Fit the native heads only on that checkpoint's original TRAIN export. Use
native 32 numbers with TRAIN-only normalization and no post-encoder PCA or random
projection. Ridge penalty 1 is the linear head; the secondary neural head has
16 tanh hidden units, Adam 0.01 and 100 updates. Keep all repetitions
2701/2802/2903, paired by actual input width; no best repetition is selected.

Raw input has 288 TRAIN-scaled observed values plus 288 observation flags,
576 numbers total. Standalone PCA fits raw TRAIN only and supplies 32 components.
The raw heads have 1154/9266 fitted parameters, versus 66/562 for native32 and
PCA32 (linear/neural). For each master/budget input, the driver refits raw/PCA
maps and heads on the same original TRAIN; these are new diagnostic fits, not
retained controls. Their scores are numerically constant across budgets.
The 512 native fit/prediction witnesses reproduce the original C fitting assets.

Intact VALIDATION remains the original known cohort. The one new corrupted
view deletes 30% of coordinates through a separately frozen source-based stream,
on top of natural missingness. Opposite-label variants share the requested
mask. This single view is fixed across budgets, methods and repetitions;
hidden storage is zero, retained values and row order stay exact, and evaluation
restores no support. Each point's same TRAIN-fitted heads score both views with
zero additional fitting for the view. This is not the old 12-case TEST stress
sweep. No v4 model or head is evaluated on this new deletion view.

## Intact timing quality on known VALIDATION

Means average the three fixed readout repetitions within each master, then
masters equally. RPB-v6 denotes the unchanged inference architecture with
context-deletion training; Updates distinguishes points on the same trajectory.
All rows retain 100% coverage: 128 examples per master, 384 across masters.

| Method | Updates | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw data — no encoder | N/A | 576 | 52.86 | 52.60 | 100 |
| PCA only — no encoder | N/A | 32 | 51.82 | 76.56 | 100 |
| RPB-v6 | 512 | 32 | 98.96 | 100.00 | 100 |
| RPB-v6 | 1024 | 32 | 98.96 | 100.00 | 100 |
| RPB-v6 | 2048 | 32 | 96.88 | 98.35 | 100 |

The historical v4 fixed-512 intact VALIDATION ridge mean is 94.7916667%.
It is a cached reference, not a newly trained or newly scored control here.
There is no corresponding v4 result for the new deletion view.

## Fixed additional 30% deletion on known VALIDATION

Use the same native32 exports and frozen ordinary TRAIN heads. Coverage remains
100% for every method, master and budget, so full-population correctness equals
the conditional accuracy shown here.

| Method | Updates | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw data — no encoder | N/A | 576 | 46.61 | 48.61 | 100 |
| PCA only — no encoder | N/A | 32 | 52.86 | 50.35 | 100 |
| RPB-v6 | 512 | 32 | 98.44 | 99.74 | 100 |
| RPB-v6 | 1024 | 32 | 97.40 | 99.48 | 100 |
| RPB-v6 | 2048 | 32 | 95.31 | 96.53 | 100 |

At 1024, mean intact ridge is unchanged but deletion-view ridge falls by 1.04
percentage points. At 2048, mean ridge falls by 2.08 intact and 3.13 under
deletion versus replay 512. Master 3303 carries the largest decline:
intact ridge goes from 96.875% to 90.625%, and deletion-view ridge from 95.3125%
to 86.71875%. Equal full coverage rules out a change in scored population as
the explanation for these accuracy differences. These known-cohort shifts do
not by themselves establish why training changes linear access.

## Reconstruction and training cost

The error is fixed-query standardized MAE under the frozen original TRAIN
scaler. Enumerate the same four original hidden-patch queries, target only
legally observed cells and reduce cell -> channel -> example equally. Use
ordinary unaugmented inference through the exact served native32 decoder input.
This is distinct from randomly masked minibatch Huber loss; the complete fresh
loss trace retains attempted/completed counters, target counts and gradient norms.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v6 | 512 | 0.085523 | 0.085768 | 14.26 |
| RPB-v6 | 1024 | 0.071130 | 0.072810 | 28.29 |
| RPB-v6 | 2048 | 0.035736 | 0.035845 | 58.00 |

These are mean cumulative times for the new replayed trajectories. The first
512 updates cost 14.262272 seconds; continuation adds 14.031313 seconds to reach
1024 and a further 29.710344 seconds to reach 2048. Timing is CUDA-synchronized
existing training-loop wall time, including CPU masks, context preparation,
transfers and bookkeeping. It excludes checkpoint loading/writing, replay state
audit, classifier fitting, extraction and scoring. GPU-kernel-only duration is
unmeasured. Do not add replay cost to the original C measurement or substitute
this replay time for its historical training time. These are descriptive costs,
not a repeated speed benchmark or an inference/memory measurement.

The historical v4 fixed-512 mean TRAIN/VALIDATION MAE is
0.061601474/0.063486190. At 1024, both reconstruction errors still exceed that
reference and the deletion-view linear score is lower. At 2048, both mean errors
pass the reference, but both mean linear scores and the worst-master quality
are worse than replay 512. Therefore no measured point satisfies both parts of
the diagnostic question. Lower reconstruction error does not establish a better
timing embedding, and longer training is not supported as the remedy here.

## Evidence, audit status and identities

Preserved capsule:
[context-optimization-validation-GFfFj3](../../../output/runs/rpb-context-optimization-validation/context-optimization-validation-GFfFj3/).
It contains the [launch plan](../../../output/runs/rpb-context-optimization-validation/context-optimization-validation-GFfFj3/results.launch-plan.json),
[continuation measurements](../../../output/runs/rpb-context-optimization-validation/context-optimization-validation-GFfFj3/results/continuation-report.json),
[shared readout results](../../../output/runs/rpb-context-optimization-validation/context-optimization-validation-GFfFj3/results/readouts/report.json)
and [coordinated admission](../../../output/runs/rpb-context-optimization-validation/context-optimization-validation-GFfFj3/admission/passed.json).
The 61-file compiled production footprint is
`305fd58213d8d5aba636bcdac8f3088e790af3875234497c3b61d29c257a9de8`.

Original C producer identities remain unchanged. New checkpoints retain their
current core-writer fingerprint; ordinary training companions identify the
actual training producer; replay companions bind those to the immutable parent
producer and the new orchestrator. Source IDs and serialization metadata are
not claimed equal to the old checkpoint. The replay gate compares identified
numerical state exactly, while original input bytes remain protected. Typed CPU
sidecars retain named model/buffer/AdamW/scaler witnesses, absolute counters,
context counts, full loss traces and separate replay/continuation costs.

The [independent archive audit v2](../../../output/runs/rpb-context-optimization-validation/audit-tools/run-GFfFj3-v2/validation.json)
passed 47,104,671 checks: three cohorts, nine points, three exact replay state
gates, six optimizer transitions and 621 named optimizer-state inspections.
It verifies 30 reconstruction archives, 81 new fitted pipelines, nine original
512 native-fit parity witnesses and 243 prediction archives. It checks 81
ordinary pairs, 81 deletion-view pairs and 162 view-versus-ordinary effects.
All 576 TRAIN/VALIDATION source groups are disjoint; all 52 declared original
input roles remain byte exact. The reader regenerates 55,296 source-based
coordinate draws and records zero TEST/stress payload reads.

The preserved
[v1 failure](../../../output/runs/rpb-context-optimization-validation/audit-tools/run-GFfFj3-v1/validation.json)
reported an erasure-stream mismatch because it compared an `array('b')` slice
with a Python list, rather than comparing their bit values. The
[v2 reader](../../../output/runs/rpb-context-optimization-validation/audit-tools/independent-context-optimization-20261007-v2/validate_context_optimization.py)
corrects that container comparison and adds a regression check; the RNG law,
declared numerical checks and production artifacts are unchanged. The failed
audit remains alongside the completed v2 run, rather than being overwritten.

| Evidence | SHA256 |
| --- | --- |
| Independent v2 validation JSON | `80756785306375df7dc0f5a7d64bfa219e8820dd6985b6cc8547aecf73d1bd85` |
| Independent v2 auditor | `aed4caf0414cec03432b44ded239cb7a6bd7effaf1eb762cd703f00ad79df399` |
| Coordinated admission record | `ba3dd870e59d9bf70eedd0ea14a6736e47030b226728f654ded93c49801e5746` |
| Capsule artifact inventory | `2dcfffb06074e3655c8abdb7db1b2d2b7e14f8626c497d0964d4f21e54d5dfdd` |

The [capsule inventory](../../../output/runs/rpb-context-optimization-validation/context-optimization-validation-GFfFj3/artifact-integrity.json)
contains 753 files and 286,523,588 bytes, excluding the inventory itself.
The large check count includes repeated tensor comparisons, not independent
statistical trials. Bootstrap estimates, support, recipes and bounds are checked;
bootstrap draws are not independently regenerated. The source-group intervals
remain conditional within master, not uncertainty across three new retrainings.
The no-TEST/stress claim follows explicit input allowlisting, payload guards and
frozen source review, not an operating-system syscall trace. Coordinated CUDA
admission separately establishes actual device, optimizer association and
unchanged-policy semantics; the independent archive reader executes no model.

## Next bounded action

Preserve the original fixed-512 tradeoff and all later diagnostic points. Do not
select a later budget, change the reconstruction guard or promote v6 from these
known VALIDATION scores. No TEST score or old stress result was opened to make
this decision.

The next proposed comparison is a separately frozen five-seed paired replication
at the existing 512-update budget: train fresh v4 and v6 on the same new TRAIN
sources, paired initialization and update streams, with unchanged native32
heads and an unopened TEST namespace. Equal budgets isolate the context policy;
the question is whether its quality benefit repeats across independent training,
rather than whether more reconstruction optimization can recover it. This
proposal is not a completed experiment or an authorization to tune on new TEST.
Follow the [next plan](NEXT_ADVANCE.md) and
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md).
