# RPB v5 paired pooling experiment

Date: 2026-10-07. Status: measured; claimed advance rejected at512.
Protocol: paired-pooling-v1; policy1.2; synthetic development only.

RPB-v5 is the direct aligned patch-state global bottleneck. RPB-v4 is the
retained learned global bottleneck and remains active. V5 failed both declared
linear primaries and fixed-query TRAIN/VALIDATION reconstruction. This rejects
its claimed advance under the [fixed512 card](../../evaluation/cards/paired_pooling_v1.md),
not every training budget or use of this design. The frozen card remains unchanged.
The [completed v4 native curve](NATIVE_CURVE_ADVANCE.md) remains preserved.

## Conditions and mechanism

This is a new timing (`lag_sign`) comparison with three masters3101/3202/3303.
Each reuses exactly128 TRAIN pairs (256 examples) and64 VALIDATION pairs
(128 examples). A new TEST draw supplies64 pairs (128 examples) per master,
384 unique TEST examples total. Two opposite-label variants stay within their
source split and share masks. Geometry C3/H32/F3, patch8, natural missingness10%,
batch8, verified RTX A2000 CUDA; candidate training completes512 attempts/updates.
Known TRAIN/VALIDATION cohorts and masters plus fresh TEST are development
source replication, not fresh-master confirmation or consumer acceptance.

Keep temporal blocks, one aligned mixer, reconstruction objective, scaler,
decoder and native32. Common-module initial weights and original sampling/mask
streams match the frozen v4 point0 assets. Restore v4 point0/512 without training
another v4. V5 pools all canonical C*K*W patch states followed by C*K visible
bits using780 ->64 ->32 with GELU; absent states are zero. V4 compresses each
channel before learned global pooling. Both decoders consume solely exact
served32. V5's per-channel D exports remain diagnostics with no reconstruction
gradient through the bypassed branch. No projection follows either encoder.

| Encoder | Total parameters | Pool parameters | Completed updates |
| --- | ---: | ---: | ---: |
| RPB-v4 | 225,805 | 8,480 | Frozen512 |
| RPB-v5 | 269,389 | 52,064 | 512 |

The accepted extra cost is43,584 parameters (+19.30%). This contrast changes
state resolution, support conditioning and pooling capacity together; the
result does not isolate any one as its cause.

Candidate heads fit TRAIN at point0/512 only. Reference/raw/PCA/metadata
normalizers, maps and heads are loaded from saved tensors without fitting
constructors and reproduce original VALIDATION predictions. Ridge penalty1 is
primary; tanh16/Adam0.01/100 updates is secondary. Repetitions2701/2802/2903 are
paired by actual width. Native32/PCA32 heads have66/562 parameters; raw576 heads
have1,154/9,266. Three head fits are not three encoder trainings.

A durable fixed512 manifest precedes all fresh TEST generation. Namespace
`paired-pooling-v1/fresh-testing`, stream `0x7070763174657374ULL` (`ppv1test`),
differs from native-curve TEST. No old TEST is an input. No TEST/stress fitting.

## Intact timing quality

Means average head repetitions within each master, then masters equally.
RPB-v4 is the learned summary-based global bottleneck; RPB-v5 is the direct
aligned patch-state global bottleneck. Raw/PCA retain their original TRAIN fits
and are newly scored on the same fresh TEST sources.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.04 | 53.99 | 100 |
| PCA only — no encoder | 32 | 48.96 | 81.34 | 100 |
| RPB-v4 | 32 | 94.53 | 96.96 | 100 |
| RPB-v5 | 32 | 84.38 | 96.53 | 100 |

Linear effect is−10.16 percentage points; neural effect is−0.43. Point-zero
controls score55.73%/68.23% for v4 and56.77%/90.89% for v5 (linear/neural).
V5 improves over its own initialization, which differs from beating trained v4.
Metadata scores50%; legal-raw oracle scores100% with full coverage for each master.

## Additional30% coordinate deletion

This second primary is additional to natural10% missingness (approximately37%
expected total missingness). Source variants, methods and repetitions receive
identical erasures and frozen readouts. RPB-v4 and RPB-v5 keep native32 without
PCA or refitting. Other fixed stress cases are diagnosis, not replacement primaries.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 50.78 | 48.61 | 100 |
| PCA only — no encoder | 32 | 50.00 | 53.13 | 100 |
| RPB-v4 | 32 | 85.16 | 90.36 | 100 |
| RPB-v5 | 32 | 66.15 | 75.87 | 100 |

Linear effect is−19.01 points; neural effect is−14.50. Both primaries retain
identical128-row/64-source-group common populations per master. Full-population
correctness equals conditional accuracy here. Allmissing signal cases have
zero coverage and null accuracy; metadata remains a structurally valid control.
Intact stress predictions exactly equal ordinary TEST predictions.

The following Ridge intervals use the first predeclared head repetition.
All repetitions are retained; Ridge predictions are identical across them.
Intervals are95% source-group percentile bootstrap,1,000 draws, conditional
within master, not an across-retraining confidence interval.

| Master | Condition | RPB-v4 % | RPB-v5 % | Difference pp | Paired95% interval pp |
| --- | --- | ---: | ---: | ---: | --- |
| 3101 | Intact | 100.00 | 83.59 | −16.41 | [−22.66, −10.16] |
| 3202 | Intact | 85.16 | 77.34 | −7.81 | [−17.19, 0.78] |
| 3303 | Intact | 98.44 | 92.19 | −6.25 | [−10.94, −1.56] |
| 3101 | Additional30% | 94.53 | 62.50 | −32.03 | [−39.84, −24.22] |
| 3202 | Additional30% | 74.22 | 66.41 | −7.81 | [−17.19, 1.56] |
| 3303 | Additional30% | 86.72 | 69.53 | −17.19 | [−25.00, −8.59] |

Every master has a lower v5 primary point score. Intervals are wholly negative
for3101/3303 and cross zero for3202. Worst-master scores also decline:77.34%
versus85.16% intact and62.50% versus74.22% under deletion.

## Reconstruction and cost

RPB-v4 and RPB-v5 reconstruct only through exact native32. Fixed-query
standardized MAE uses the same frozen TRAIN scaler, observed original-patch
targets and equal cell -> channel -> example reductions. These are newly
reproduced errors, not the final random optimization minibatch loss.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | Frozen512 | 0.06160 | 0.06349 | 15.33 |
| RPB-v5 | 512 | 0.09027 | 0.09474 | 14.37 |

V4 time is reused historical native-curve time; v5 is this run's synchronized
encoder-only mean. This is not a contemporaneous repeated speed benchmark.
V5 times are14.03/14.71/14.38 seconds. Each instance samples4,096 rows:16
equivalent presentations of256 TRAIN rows, not16 guaranteed complete epochs.
V5 TRAIN/VALIDATION MAE are worse for every master; mean differences+0.02866
and+0.03126. TEST MAE is0.09564 versus0.06375; this diagnostic selects no budget.

## Disposition and limits

Engineering, initialization, no-skip, scaler, exact export/decoder and immutable
asset gates passed. The frozen rule's two positive primary means, two
worst-master comparisons and both mean TRAIN/VALIDATION MAE guards failed.
**Reject the claimed advance at512; keep RPB-v4 active.** Neural-only, favorable
master or alternative-stress findings cannot rescue the declared comparison.
This does not universally reject direct pooling or establish that512 is its
optimal training budget.

Known development cohorts, three initializations and synthetic timing limit
interpretation. No new direction/level/amplitude TEST was opened. Consumer
data/chronology, task, support/cost limits and acceptance thresholds remain
undefined. Do not pool conditional intervals into an across-master CI.

## Evidence and identities

[Capsule](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/):
[launch plan](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results.launch-plan.json),
[machine card](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results/paired-pooling-card.json),
[input ledger](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results/input-manifest.json),
[durable comparison manifest](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results/comparison-manifest.json),
[VALIDATION](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results/validation-report.json),
[TEST](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results/report.json),
[stress](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results/stress-report.json),
[CUDA gate](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/results-cuda-gate/gpu-check.json).

The57-file source snapshot fingerprint is
`3655668a178e389f59d76d2af187d49ec7acc06d7928057bb70dbd367f559166`.
Recorded HEAD `1fba1fecc0756021bf62888ece937dd1616795d6`, dirty before candidate
source commit; preserved snapshot, not a later doc commit, identifies the run.
Unchanged human card SHA256:
`8b6269bc0a402ec47160c74b39234949a7d73e742681bbbf583dd1c35b083af1`.
Machine card SHA256:
`a9175214afc18a05473a501d38299cce65cf4d3d6d61355eb52f72e1254b6393`.
Comparison manifest SHA256:
`20eca77fc99f2f63f5dbd0e7401be06422ced85373d9147905f8b96ee9090832`.

| Master | Candidate point0 checkpoint SHA256 | Candidate point512 checkpoint SHA256 |
| --- | --- | --- |
| 3101 | `506b6352125f6b6bba37bae68181374af0a5361198823cea448c5a7cfc1b22d1` | `fc9da5eceffc252cb69f5e9b4a92e6464711cfcc8c1e7c225bb42936f24a1207` |
| 3202 | `dd727a279f30617f6856c272bc8615f9af9e3b84c1d853ed104813adc1280a3b` | `7a7d3949f9757ba7594c00cb6231e700d7ce19e822194b660126eeff582c3d1e` |
| 3303 | `04081dae601bb3641e22b91c03c2c33bc663e86bfa93b666efab5eff28996fc2` | `02c45cc957fa6beae6730df66bee0bdb5c22297ce18d0f620c4890fcb3044de0` |

Original v4 checkpoint/input identities remain pinned by the unchanged card.
New [artifact inventory](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/artifact-integrity.json):
1,326 files,405,076,455 bytes, excluding itself; SHA256
`610976e666468ac2f80404bd56884e36f9275645305e83a9d0e50b762e702be3`.

[Independent audit](../../../output/runs/rpb-paired-pooling/audit-tools/independent-paired-20261006-v1/run-BFFPX5-v1/validation.json)
passed47,753,221 checks without repair/relaxation. JSON SHA256:
`7e8b8679e440c9a7935cf5a6e3808f150316079268ab04a3167c398d73b2b56a`;
auditor SHA256:
`5de9e57ff42793123fc8d378ea366eb8be62c261edc6ceb077011ce5f6d59e1a`.
Verified3 cohorts,6 candidate points,63 fits (18 candidate/45 retained),144
ordinary prediction checks,45 VALIDATION/45 TEST pairs,108 stress cases/756
stress predictions,768 disjoint source groups. Reconstruction reductions were
audited30 times, including repeated positive-point checks. All174 original role
files remained identical; old TEST reads and retained refits were zero.

Audit replays both heads, all normalizers, raw preprocessing, masks, scores and
reconstruction reductions from CPU tensors. Bootstrap estimates/populations/
bounds were checked, not bootstrap draw replay. CUDA checkpoint execution and
exact decoder/optimizer semantics were covered by container C++ gates, including
legacy mode0/1/2 compatibility and mode3 continuity/no-skip admission.

## Next action: VALIDATION-only continuation

Save/commit this completed milestone before further source work. A separate
prospective diagnostic may resume exact v5 point512 model/AdamW/scaler states
through absolute1024 and2048 on the same TRAIN observations and streams. Record
step-ceiling overrides; do not initialize another model, seed, optimizer or
scaler. Keep native32 and fixed heads; new point heads fit TRAIN only.

Freeze a separate diagnostic recipe before running. Use existing VALIDATION
scores, fixed-query reconstruction and cumulative cost for all three masters
to distinguish insufficient optimization at512 from persistent poor linear
access. Do not open/rescore TEST or stress archives, alter the fixed512 rejection,
promote a model, or call this unseen confirmation. Any later budget comparison
needs its own declared reference/budget/namespace before another TEST draw.
