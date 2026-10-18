# Five-master context deletion replication

Date: 2026-10-07. Status: measured and independently audited; no promotion.
Protocol: `context-replication-v1`; policy 1.2; synthetic development only.

The fresh five-master comparison repeats the native linear accuracy gain from
training context deletion, especially when additional coordinates are missing.
Mean fixed-query TRAIN and VALIDATION reconstruction errors still exceed the
fresh, equal-budget reference. The full [frozen rule](../../evaluation/cards/context_replication_v1.md)
therefore does not establish a retained research lead. RPB-v4 remains active.
Independent archive audit v3 passed 103,787,020 checks. Both failed reader
attempts and their identified corrections remain preserved. Coordinated CUDA
admission passed separately; these are different evidence stages.

This is a new comparison on five new training masters, not another scoring of
the original three models. The [original context experiment](CONTEXT_DELETION_ADVANCE.md)
and [TRAIN/VALIDATION optimization diagnostic](CONTEXT_OPTIMIZATION_DIAGNOSTIC.md)
remain unchanged, with their original cards and dispositions.

## Conditions and exact pairing

Measure timing (`lag_sign`) only, with independently initialized masters 4404,
5505, 6606, 7707 and 8808. Each master has 128 TRAIN source pairs (256 examples),
64 VALIDATION pairs (128 examples) and 64 fresh TEST pairs (128 examples).
Across masters these are 640/320/320 independent source groups and
1,280/640/640 examples. Opposite-label variants remain in the same source split
and share observation masks; two variants are not two independent sources.
Geometry is C3/H32/F3, patch length 8, natural missingness 10%, batch 8, one CPU
Torch thread and verified RTX A2000 CUDA.

Train five fresh RPB-v4 trajectories and five fresh RPB-v6 trajectories to exactly
512 updates each. Every positive checkpoint has attempted = completed = 512,
4,096 sampled rows and no skipped attempts. This is 16 equivalent presentations
of 256 TRAIN rows, not 16 guaranteed complete epochs. Point 0 is the exact
initialization diagnostic of each trajectory, not another trained model.
Three readout repetitions per checkpoint are not three encoder runs.

The baseline preparation uses `native-development-v1/lag_sign`: it generates
TRAIN/VALIDATION only, trains point 0/512, completes all feature/readout/query
witnesses and writes a durable development-completion record without TEST,
stress or checkpoint selection. Freeze its 453 permitted files before candidate
training and fresh TEST generation. Stage 2 loads these RPB-v4 and raw/PCA/control
assets directly; recorded retained transform and readout refits are both zero.
The enclosing frozen-manifest checks report all 453 files preserved before and
after comparison. The engine separately reports preservation of its 248 declared
input roles. The successful independent audit also verifies all 453 baseline hashes.

Within each master, all 225,805 point 0 parameters and all buffers match exactly,
including the learned global pool. Legal TRAIN observations, physical channel
order, semantic IDs 0/1/2, source order, schema, frozen scaler statistics and
identity, initialization, row sampling, original patch masks and per-attempt
Torch streams are paired. Encoder fitting receives permitted TRAIN observations
and metadata only. Labels are available to TRAIN readout fits, never encoder
training; hidden clean signal and held-out observations are not fitting inputs.

Both recipes use width 64, three temporal blocks, four heads, feed-forward 256,
one aligned channel mixer, global mode 2, native export width 32 and decoder
width 128. Keep dropout 0, layer-normalization epsilon 1e-5, original patch-mask
ratio 0.25, hierarchical Huber delta 1, scaler floor 1e-6, AdamW 0.001,
weight decay 0.0001 and gradient clipping 1. The decoder receives solely the
exact served global32 plus existing channel/patch metadata. No PCA or random
projection follows either encoder. Inference uses ordinary unaugmented mode 2.

Only candidate training visibility changes. Preserve natural observations O,
original hidden-patch request A, targets Q=O&A, original eligible channels and
Huber denominators. On original visible context V0=O&~A, request 0.30 coordinate
deletion through the existing `0x6374782d64726f70ULL` counter stream, absolute
attempt and semantic-canonical B/C/H/F ordinal. Encode V=V0&~E with hidden
storage zeroed. Repair only E, when needed, to retain two originally visible
patch groups in each originally eligible channel: earliest erased original
patch, then its first original-visible coordinate. Never restore a query or
natural missingness. Companions retain requested/actual/restored counts; the
request probability is not an exact realized erasure fraction.

Audited cumulative coordinate counts at 512 are:

| Master | Requested coordinates | Deleted coordinates | Restored coordinates |
| --- | ---: | ---: | ---: |
| 4404 | 237,487 | 237,487 | 0 |
| 5505 | 238,724 | 238,724 | 0 |
| 6606 | 238,961 | 238,961 | 0 |
| 7707 | 238,322 | 238,322 | 0 |
| 8808 | 238,257 | 238,257 | 0 |

All three counts are zero at point 0. No support repair was needed in these
measured training trajectories; correctness under sparse support remains a
separate admitted test contract.

RPB-v6 checkpoints retain `rpb-training-context-deletion-v1`. Ordinary training
resume rejects that tag; this comparison uses fresh continuous training and no
augmented resume. Frozen export retains policy provenance and applies no
training deletion or repair.

## Classifiers and fresh held-out population

Fit candidate native32 heads on valid TRAIN features at points 0/512. Load the
fresh baseline's already fitted native heads and controls, reproducing original
VALIDATION feature, prediction and reconstruction witnesses before TEST.
Keep all head seeds 2701/2802/2903, paired through the declared actual-width seed
mapping. Ridge with penalty 1 is primary. The neural secondary uses 16 tanh
hidden units, Adam 0.01 and 100 updates. Native32/PCA32 heads have 66/562 fitted
parameters; raw576 heads have 1,154/9,266. These are fixed recipes with separately
TRAIN-fitted weights and normalization.

Raw data supplies 288 TRAIN-scaled observed values plus 288 observation flags.
Its scaler floor is 1e-8. Standalone PCA32 and its normalization fit only raw
TRAIN. The native pipeline uses its TRAIN outer normalizer and the head's own
TRAIN normalizer. Mask metadata and legal raw oracle are distinct controls;
the oracle is a hand-written solvability check, not a trained classifier.

All development witnesses precede the durable fixed-budget comparison manifest.
Then generate exactly one fresh TEST population per master with namespace
`context-replication-v1/fresh-testing`, stream `0x6372763174657374ULL` (`crv1test`).
Neither phase reads old three-master TEST/stress payloads. All methods receive
the same new legal observations, masks, rows and source IDs. Keep models,
scalers, maps, normalizers and heads frozen for TEST and all 12 stress cases.
There is no TEST fitting, head selection, budget selection or replacement draw.

## Intact timing quality

Means average the three declared readout repetitions within each master, then
weight the five masters equally. RPB-v4 is the learned global bottleneck.
RPB-v6 serves that same inference architecture after context-deletion training.
Raw/PCA controls are freshly fitted during reference preparation and reused
without refitting during candidate comparison.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 47.81 | 51.46 | 100 |
| PCA only — no encoder | 32 | 51.09 | 71.41 | 100 |
| RPB-v4 | 32 | 94.06 | 97.71 | 100 |
| RPB-v6 | 32 | 95.78 | 99.69 | 100 |

The paired mean effect is +1.71875 linear percentage points and +1.9791667 neural
points. Intact ridge gains are nonnegative in all five masters, with two ties;
their range is 0 to +3.90625 points. Worst-master ridge rises from 88.28125% to
90.625%. Exact matched point 0 controls score 55.15625% linear and 74.7916667%
neural on average. All methods retain 128/128 rows and 64/64 source groups per
master, so conditional accuracy equals full-population correctness. Ordinary
legal raw oracle scores 128/128 in every master; mask metadata scores 50%.

## Additional 30% coordinate deletion

This second primary deletes additional coordinates on the same fresh TEST
population, using shared source-level masks across variants and methods.
Together with natural 10% missingness this is approximately 37% expected total
missingness, with actual support retained in manifests. Evaluation applies no
training support repair and uses the original frozen TRAIN heads.

RPB-v4 is the learned global bottleneck; RPB-v6 has the same serving architecture
with additional context deletion during training.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 50.16 | 49.90 | 100 |
| PCA only — no encoder | 32 | 51.41 | 53.07 | 100 |
| RPB-v4 | 32 | 87.81 | 93.75 | 100 |
| RPB-v6 | 32 | 95.94 | 97.76 | 100 |

The mean effect is +8.125 linear points and +4.0104167 neural points. Every
master has a positive linear point effect, ranging from +1.5625 to +25 points.
Worst-master ridge rises from 75% to 91.40625%. Coverage and common populations
remain 128/128 rows and 64/64 sources in each master. These point effects repeat
the moderate-corruption benefit; they do not establish a uniform effect size.

The paired Ridge intervals below use the first predeclared repetition; all
three repetitions remain retained. Intervals are 95% source-group percentile
bootstrap with 1,000 draws, conditional within master on its fitted checkpoint
and readouts. They are not confidence intervals across five retrainings.

| Master | Condition | RPB-v4 % | RPB-v6 % | Difference pp | Paired 95% interval pp |
| --- | --- | ---: | ---: | ---: | --- |
| 4404 | Intact | 97.65625 | 100 | +2.34375 | [0, 5.46875] |
| 5505 | Intact | 92.1875 | 92.1875 | 0 | [-7.03125, 7.03125] |
| 6606 | Intact | 100 | 100 | 0 | [0, 0] |
| 7707 | Intact | 88.28125 | 90.625 | +2.34375 | [-3.90625, 8.59375] |
| 8808 | Intact | 92.1875 | 96.09375 | +3.90625 | [-1.5625, 10.15625] |
| 4404 | Additional 30% deletion | 75 | 100 | +25 | [18.75, 31.25] |
| 5505 | Additional 30% deletion | 89.84375 | 91.40625 | +1.5625 | [-6.25, 9.375] |
| 6606 | Additional 30% deletion | 97.65625 | 99.21875 | +1.5625 | [-1.5625, 4.6875] |
| 7707 | Additional 30% deletion | 91.40625 | 92.96875 | +1.5625 | [-4.6875, 7.8125] |
| 8808 | Additional 30% deletion | 85.15625 | 96.09375 | +10.9375 | [3.90625, 17.1875] |

Only two deletion-view intervals have strictly positive lower bounds; intact
intervals include zero. The ceiling case at 6606 is not proof of zero population
effect. Do not interpret the aggregate point means as a confirmed across-master
confidence statement or replace unfavorable uncertainty with the largest gain.

All 12 declared stress cases remain in the report, with 15 master/repetition
runs and seven methods per case. Secondary ridge means include 68.75% versus
85.15625% at 60% additional coordinate deletion, but 52.1875% versus 51.5625% at
90%; 74.6875% versus 74.0625% under a 75% contiguous gap. Thus the benefit is not
uniform over corruption severity. All-absent signal surfaces are invalid with
zero coverage and null conditional accuracy, while metadata retains its distinct
structural-control role. These secondaries cannot replace either primary.
Intact stress and ordinary TEST prediction/validity parity is retained in the
engineering witnesses, subject to independent archive verification.

## Reconstruction and training cost

Use original fixed whole-patch queries with ordinary unaugmented inference:
target only legal observed cells, require two remaining visible patch groups,
and decode through exact native32. TRAIN and VALIDATION targets, eligibility,
query checksums and frozen TRAIN scales are paired. Error is standardized MAE
with equal cell-to-channel-to-example reductions, not the last randomly masked
minibatch Huber loss. RPB-v4 is the learned global bottleneck; RPB-v6 uses the
same decoder route after context-deletion training.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 | 0.067593 | 0.071309 | 13.85 |
| RPB-v6 | 512 | 0.091179 | 0.092984 | 15.06 |

Full-precision means are TRAIN/VALIDATION 0.0675931436195981/0.07130906590008226
for v4 and 0.091178963646751/0.0929839542097942 for v6. Every master has worse
candidate TRAIN MAE; four of five have worse VALIDATION MAE. Master 7707 has
slightly lower candidate VALIDATION MAE (0.08423434 versus 0.08501756), which
does not rescue either failed aggregate guard. Both mean errors must be no
worse than the contemporaneous reference under the frozen card.

Timing comes from `progress.cumulative_training_seconds`, averaged over the five
new runs: v4 13.847804411 seconds, v6 15.0607659384 seconds. The difference is
1.2129615274 seconds per trajectory. This synchronized training-loop timer
includes CPU planning, transfers and bookkeeping within the loop; it excludes
readout fitting, extraction, scoring and artifact writing. Isolated GPU kernel
time, inference speed and memory were not measured. Timing is descriptive,
not a repeated performance benchmark or a new acceptance threshold.

## Evidence and audit status

The [completed capsule](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/)
contains the [reference launch plan](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/reference.launch-plan.json),
[comparison launch plan](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/results.launch-plan.json),
[development-completion record](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/reference/development-complete.json),
[453-file input ledger](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/reference-inputs.json),
[before](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/reference-preserved-before.txt)
and [after preservation checks](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/reference-preserved-after.txt),
[durable pre-TEST comparison manifest](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/results/comparison-manifest.json),
[VALIDATION report](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/results/validation-report.json),
[TEST report](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/results/report.json),
[stress report](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/results/stress-report.json)
and [input-preservation record](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/results/input-integrity-after.json).
The generic machine-card filename remains `paired-pooling-card.json`; its
protocol is `context-replication-v1`, not an architectural pooling change.

The 63-file compiled production snapshot is identified by
`9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828`.
Recorded Git HEAD is `7be9a89c4f958ce82d5b121c048d480756eb752c`, with a dirty
checkout before the replication source commit; the preserved source snapshot
binds the actual binaries. Both TRAIN producer companions identify
`2fff50c48605ee21dbbfd28d6989bdcf6b5cd057785c2acb19419571af48b895`.
Ordinary checkpoints separately record core-writer fingerprint
`587f2423758c3e70c2c7665d9e7b59bfe10af4c7f9aaba23f5318b3c55b13bca`.
These source scopes are not interchangeable with human recipe tags or individual
checkpoint/data/scaler hashes.

[Coordinated admission](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/admission/passed.json)
passed actual enabled context-training CUDA contracts, exact fresh same-mode
initialization/scaler/data/stream pairing, unchanged target/loss support,
default-protocol compatibility, tagged resume rejection and native32 serving.
All measured parameters, inputs and losses are CUDA, with finite gradients and
changed weights. The separate [CUDA serving gate](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/results-cuda-gate/gpu-check.json)
does not by itself establish augmented training correctness.

The [independent audit v3](../../../output/runs/rpb-context-replication/audit-tools/run-JjNEUc-v3/validation.json)
passed 103,787,020 checks. It verifies five cohorts and 20 initial/positive points
(ten reference, ten candidate), 105 distinct TRAIN-fitted pipelines, 315 ordinary
prediction archives and 150 paired comparisons: 75 VALIDATION and 75 fresh TEST.
It checks 15 stress reports, 180 cases, 1,260 stress prediction entries and
independently regenerates 2,211,840 source-level corruption coordinates.
Seventy reconstruction reductions are counted, with 80 unique reconstruction
archives inspected including witnesses. All 453 baseline files remain byte exact,
all 1,280 source groups across TRAIN/VALIDATION/TEST and masters are disjoint,
retained transform/readout refits are zero, and old experiment payload reads are
zero. The reader records 2,140 opened payload files in this new capsule.

The preserved [v1 failure](../../../output/runs/rpb-context-replication/audit-tools/run-JjNEUc-v1/validation.json)
stopped after 589,560 checks at `actual CUDA/canonical channel setting`.
Its [reader](../../../output/runs/rpb-context-replication/audit-tools/independent-context-replication-20261007-v1/validate_context_replication.py)
incorrectly required configured text `channel_ids=0,1,2`. The saved formatter
preserves an empty configured vector; its documented resolver supplies 0/1/2.
The correction resolves that setting while retaining exact CUDA, typed producer
channel-order and complete scaler-ID checks. Production settings and artifacts
were not changed to satisfy the reader.

The preserved [v2 failure](../../../output/runs/rpb-context-replication/audit-tools/run-JjNEUc-v2/validation.json)
stopped after 21,924,636 checks at `name 'CONTEXT_RNG' is not defined`.
Its [reader](../../../output/runs/rpb-context-replication/audit-tools/independent-context-replication-20261007-v2/validate_context_replication.py)
referenced three unbound module constants: `CONTEXT_RNG`, `CONTEXT_REPAIR` and
`CONTEXT_VISIBILITY`. The separately frozen
[v3 reader](../../../output/runs/rpb-context-replication/audit-tools/independent-context-replication-20261007-v3/validate_context_replication.py)
adds their exact strings from the pinned producer header. Independent static
review and source-only checks found no unresolved application globals across
95 lexical scopes and 422 references; its assertion self-tests exercise two
valid point budgets and 12 negative policy/rate/stream/count cases without
opening an archive. Original numerical assertions, production code, data,
cards, models, fits and scores remain unchanged. Both failed attempts are kept
alongside the successful run rather than overwritten.

The large check count includes repeated tensor comparisons, not independent
statistical trials. The independent reader executes no model and does not
independently associate CUDA tensor storage; hash-bound C++ admission supplies
that evidence. Bootstrap point estimates, support and bounds are checked, but
bootstrap draws are not independently regenerated. Conditional within-master
intervals and controlled synthetic sources do not establish across-master
confidence, consumer acceptance, other-task generalization or a production
support threshold. No-old-payload access follows the declared allowlist/read
ledger and source review, not an operating-system syscall trace.

| Evidence | SHA256 |
| --- | --- |
| Frozen human card | `461cc5944c31f545da7ae78d23976536e24b29bc1a3854228368cc1339aef76c` |
| Coordinated admission record | `b12080b0ad433db5571e4c2bf590278c43b7d895cfd7c6283945dc5672c18745` |
| Capsule artifact inventory | `13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9` |
| Failed v1 validation JSON | `0d89243cb06ed5d47db1fce8937f6b1625a5b25153796adb8e96cea9bb544651` |
| Preserved v1 reader | `15ee5da7e1f4184effcdc87596c5a4f550c0f6201f80148e263a42c12b348de5` |
| Failed v2 validation JSON | `8b8586895f911b01c07b0683f8f9519cc41fa65e317c70c11db37cf18bce11ce` |
| Preserved v2 reader | `40d16ad8af774ca9e43ad716be52db82cc8a6ae3e8a22c933ea985a0f957726e` |
| Passed v3 validation JSON | `60969d30e9c642b525ab2c29a3d788c79e40837b30109f5beb9ad928c640aee4` |
| Frozen v3 reader | `e900c82264b7f9a10a69700c91f38d0be8a503eb77846ab409abbd19298c5f7d` |

The [inventory](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/artifact-integrity.json)
contains 2,738 files and 910,266,491 bytes, excluding itself. The independent
readers and validation records live outside that capsule and are identified
separately above. Preserve all checkpoints, fits, queries, predictions, masks,
source snapshots, logs and failed audit evidence.

## Disposition and next bounded option

Both declared mean linear primaries improve, candidate coverage stays equal and
neither worst-master primary score decreases. The original-query TRAIN and
VALIDATION reconstruction guard still fails. RPB-v6 is not promoted; RPB-v4
remains active. Save this measured replication and its final audit before any
new production change. Do not reinterpret the frozen rule or tune against these
TEST/stress scores.

A proposed next option is RPB-v7 — Global bottleneck with lighter context deletion:
one separately identified 0.15 training-context policy
on the same architecture, optimizer, heads and 512-update budget, assessed first
on TRAIN and known VALIDATION. It would test whether retaining more visible
training context reduces the reconstruction tradeoff while preserving useful
native32 features. This is an inference from the tradeoff, not a measured result
or assurance that halving the request rate succeeds. Preserve the existing v6
0.30 policy and replay API exactly, retain ordinary tagged-resume rejection and
use planned policy ID `rpb-training-context-deletion-015-v1` under the prospective
[lighter-policy card](../../evaluation/cards/context_lighter_validation_v1.md).
It remains proposed, unimplemented and unmeasured. Do not reuse this
replication's TEST for rate selection; any later quality confirmation needs a
separate frozen plan and unopened held-out sources. The [next advance plan](NEXT_ADVANCE.md)
remains the broader research boundary.
