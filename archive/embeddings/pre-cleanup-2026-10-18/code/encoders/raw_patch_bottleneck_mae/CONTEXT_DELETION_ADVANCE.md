# RPB v6 context deletion experiment

Date: 2026-10-07. Status: measured; unresolved reconstruction tradeoff; no promotion.
Protocol: context-deletion-v1; policy 1.2; synthetic development only.

RPB-v6 uses the RPB-v4 inference architecture with additional visible-context
deletion during training. Both declared linear quality primaries improve, but
fixed-query TRAIN and VALIDATION reconstruction MAE worsen for every master.
The [frozen card](../../evaluation/cards/context_deletion_v1.md) therefore does
not establish a retained lead under its full rule. RPB-v4 remains active.
The card and earlier [native curve](NATIVE_CURVE_ADVANCE.md),
[v5 comparison](PAIRED_POOLING_ADVANCE.md) and
[v5 optimization diagnosis](OPTIMIZATION_DIAGNOSTIC.md) remain unchanged.

## Conditions and mechanism

This is a new timing (`lag_sign`) comparison with three masters 3101/3202/3303.
Each reuses exactly 128 TRAIN source pairs (256 examples) and 64 VALIDATION pairs
(128 examples). Each new TEST draw contains 64 pairs (128 examples): 384 unique
TEST examples across the three masters. The two opposite-label variants stay
in one source split and share the observation mask. Geometry C3/H32/F3, patch 8,
natural missingness 10%, batch 8 and verified RTX A2000 CUDA. Three readout
repetitions on one checkpoint are not three encoder training runs.

Train one fresh v6 trajectory per master continuously to 512; attempted and
completed updates both equal 512. Restore retained v4 point 0/512 checkpoints
without training another v4. Every initial parameter and buffer, including the
global pooling head, matches retained v4 exactly. Dataset/schema, semantic IDs,
TRAIN-fitted scaler tensors, row sampling, original hidden-patch masks and
per-attempt Torch RNG policies match the reference. Each instance samples 4,096
rows, equivalent to 16 presentations of 256 TRAIN rows, not 16 complete epochs.

Both models have 225,805 parameters and serve the same native 32-number global export.
Keep width 64, three temporal blocks, four heads, feed-forward 256, one aligned
channel mixer, global mode 2 and decoder 128. Preserve dropout 0, layer-normalization
epsilon 1e-5, original patch-mask ratio 0.25, hierarchical Huber delta 1, TRAIN
scaler floor 1e-6, AdamW 0.001/weight decay 0.0001 and clipping 1. The decoder receives
solely the exact served 32 numbers plus existing channel/patch metadata. No PCA or random
projection follows either encoder. Inference uses ordinary unaugmented mode 2.
Parameter equality does not constitute a measured inference-speed or memory test.

Only the training context view changes. With observation mask O and original
hidden-patch request A, retain targets Q=O&A, original eligible channels and
Huber reductions exactly. Request extra deletion at 0.30 on V0=O&~A using the
independent `0x6374782d64726f70ULL` counter stream, absolute attempt and canonical
semantic B/C/H/F coordinate ordinal. Requests do not depend on values, labels,
held-out data or Torch RNG. Encode V=V0&~E with nonvisible storage zeroed.

If extra deletion would leave an originally eligible channel with fewer than
two original visible patch groups, restore only extra-deleted coordinates:
earliest fully erased original patch, first original-visible h/f coordinate,
until two groups remain. Query and naturally missing support are never restored.
The exact counter/repair recipe is preserved in the frozen card and policy
companions. This constrained request policy is not an exact empirical 30%
erasure rate. Actual cumulative request/deletion counts at 512 were:

| Master | Requested coordinates | Deleted coordinates | Restored coordinates |
| --- | ---: | ---: | ---: |
| 3101 | 239,051 | 239,051 | 0 |
| 3202 | 238,204 | 238,204 | 0 |
| 3303 | 239,253 | 239,253 | 0 |

Point 0 counts are zero. Ordinary checkpoints carry
`rpb-training-context-deletion-v1`; ordinary training resume rejects this tag
to prevent silently switching policy. The measured run was fresh and continuous;
no augmented-resume API was used. Export may load the checkpoint and preserves
policy provenance while serving without training erasure.

Candidate native heads fit only valid TRAIN rows at point 0/512. Retained v4,
raw/PCA/metadata scalers, normalizers, PCA map and heads load directly from
tensors with zero fitting constructors. Original VALIDATION predictions,
native exports and immutable reconstruction witnesses reproduce exactly.
Ridge penalty 1 is primary; tanh with 16 hidden units, Adam 0.01 and 100 updates
is secondary. Keep all head seeds 2701/2802/2903, paired by actual input width.
Native 32-number/PCA 32-component heads have 66/562 parameters; raw 576-number
heads have 1,154/9,266. Raw data consists of 288 TRAIN-scaled observed values
plus 288 observation flags. The raw scaler and
standalone PCA use only TRAIN, with the raw scaler's 1e-8 floor.

The durable fixed 512-update comparison manifest precedes every fresh TEST draw.
Namespace `context-deletion-v1/fresh-testing`, stream
`0x6374763174657374ULL` (`ctv1test`), differs from earlier native/pooling TEST.
No old TEST/stress payload is an input, and no TEST/stress fitting occurs.
Known development masters and TRAIN/VALIDATION cohorts plus fresh TEST are
source replication, not fresh-master confirmation or consumer acceptance.

## Intact timing quality

Means average the three declared head repetitions within each master, then
masters equally. RPB-v4 is the retained learned global bottleneck. RPB-v6 has
the same inference architecture with additional training context deletion.
Raw/PCA use their retained TRAIN fits and are newly scored on these TEST sources.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 48.44 | 53.30 | 100 |
| PCA only — no encoder | 32 | 53.39 | 76.48 | 100 |
| RPB-v4 | 32 | 93.49 | 96.35 | 100 |
| RPB-v6 | 32 | 98.96 | 100.00 | 100 |

The mean linear effect is +5.47 percentage points; neural effect +3.65. Paired
point 0 models, features and fitted heads are exact matches: both score 57.29%
linear and 69.79% neural. V6 gains 41.67 linear points over its initialization,
versus 36.20 for v4. This separates learned improvement from an initialization
or extra-capacity advantage, without proving a general causal benefit on other
tasks or consumers. Metadata scores 50%; legal raw oracle scores 100%, both
with full coverage. The oracle is a solvability check, not a fitted classifier.

## Additional 30% coordinate deletion

This second primary adds 30% coordinate deletion to natural 10% missingness,
approximately 37% expected total missingness. Both source variants and all
methods/repetitions receive identical evaluation erasures. RPB-v4 and RPB-v6
keep native 32-number exports and their frozen ordinary TRAIN readouts;
evaluation applies no training support repair. The other ten fixed stress cases
cannot replace a primary.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 47.92 | 49.05 | 100 |
| PCA only — no encoder | 32 | 54.43 | 50.09 | 100 |
| RPB-v4 | 32 | 86.72 | 93.06 | 100 |
| RPB-v6 | 32 | 97.14 | 99.65 | 100 |

The mean linear effect is +10.42 points; neural effect +6.60. Both primaries
retain identical 128-row/64-source-group common populations per master, so
full-population correctness equals conditional accuracy. The legal raw oracle
still scores 100% for each master under 30% deletion; metadata remains 50%.
Point 0 controls match at 52.08% linear and 55.64% neural. Intact stress predictions
exactly equal ordinary TEST predictions. All-absent signal exports are zero
and invalid, with zero coverage/null accuracy; metadata remains a structural control.

The following Ridge intervals use the first predeclared head repetition.
All repetitions are retained; Ridge point predictions are identical across
them. Intervals are 95% source-group percentile bootstrap, 1,000 draws, conditional
within master on its fitted checkpoint/heads. They are not an across-master CI.

| Master | Condition | RPB-v4 % | RPB-v6 % | Difference pp | Paired 95% interval pp |
| --- | --- | ---: | ---: | ---: | --- |
| 3101 | Intact | 100.00 | 100.00 | 0.00 | [0.00, 0.00] |
| 3202 | Intact | 83.59 | 100.00 | +16.41 | [+10.16, +22.66] |
| 3303 | Intact | 96.88 | 96.88 | 0.00 | [−3.91, +3.91] |
| 3101 | Additional 30% | 98.44 | 99.22 | +0.78 | [−1.56, +3.13] |
| 3202 | Additional 30% | 73.44 | 98.44 | +25.00 | [+17.19, +33.59] |
| 3303 | Additional 30% | 88.28 | 93.75 | +5.47 | [−1.56, +12.50] |

The gain is concentrated in the vulnerable master 3202. Its two intervals are
positive; the other masters' intervals contain zero. No master has a lower
linear point score. Worst-master scores improve from 83.59% to 96.88% intact
and 73.44% to 93.75% under deletion. This is favorable development evidence,
not a pooled statistical confirmation across retraining.

## Reconstruction and cost

RPB-v4 and RPB-v6 reconstruct through the exact native 32-number export. These
fixed-query standardized MAE measurements newly reproduce the same frozen TRAIN scaler,
observed original-patch targets, visibility/eligibility and equal cell -> channel
-> example reductions on TRAIN/VALIDATION. They use ordinary unaugmented
inference, not the final randomly masked training loss. The actual optimization
objective remains hierarchical Huber 1, with its original query denominators;
the saved traces retain losses, gradients and target counts.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 | 0.06160 | 0.06349 | 15.33 |
| RPB-v6 | 512 | 0.08552 | 0.08577 | 15.36 |

V4 GPU time is reused historical native-curve time. V6 time is this run's
synchronized encoder-training mean, including its context preparation;
classifier fitting, extraction/scoring, stress and artifact writing are excluded.
These times are descriptive, not a contemporaneous repeated speed benchmark.
V6 master times are 14.45446/17.68482/13.95417 seconds. Both models have 225,805
parameters; no new memory or inference-time benchmark was measured.

| Master | RPB-v4 TRAIN MAE | RPB-v6 TRAIN MAE | RPB-v4 VALIDATION MAE | RPB-v6 VALIDATION MAE |
| --- | ---: | ---: | ---: | ---: |
| 3101 | 0.06551 | 0.09376 | 0.06833 | 0.09197 |
| 3202 | 0.05973 | 0.08018 | 0.06034 | 0.08298 |
| 3303 | 0.05956 | 0.08263 | 0.06179 | 0.08234 |

Mean TRAIN MAE increases 0.02392 and mean VALIDATION MAE 0.02228. Every master
worsens on both. Extra visible-context deletion makes training harder while
leaving scored query targets unchanged; this observation alone does not prove
why the clean-query reconstruction regresses or whether more optimization
would resolve it. TEST reconstruction also remains a diagnostic, not selection.

## Disposition and limits

Both positive primary means, equal coverage, both worst-master comparisons,
engineering/full-initialization/policy/source gates and the independent audit
pass. **Both mean TRAIN and VALIDATION MAE no-worse guards fail.** The exact
declared disposition is `unresolved_guardrail_tradeoff`: no silent promotion
or claim that the full lead-retention rule passed. Keep RPB-v4 active while
preserving v6 as measured tradeoff evidence. Neural-only or master 3202 gains
do not waive the reconstruction clause. This does not universally reject
context deletion or negate the two measured quality improvements.

Only synthetic timing was measured. No new direction/level/amplitude TEST was
opened. Three reused development masters, conditional source-group intervals
and retained reference heads do not establish consumer chronology, acceptable
support/cost limits, fresh-master confirmation or a consumer acceptance threshold.
Do not change this card or select a favorable alternative erasure rate, head,
master or budget from these TEST/stress scores.

Save this audited milestone before another source change. The next proposed
step is a separately frozen TRAIN/VALIDATION-only optimization diagnostic,
using deterministic replay under the unchanged context policy to reproduce
the first 512 updates before any later measurements. It must preserve this
fixed 512-update disposition and verify exact parent/source/scaler/counter
provenance. Ordinary resume intentionally rejects the v6 checkpoint tag.
No TEST/stress payload may be reopened by that diagnostic; any subsequent
quality comparison needs its own frozen recipe and unopened TEST namespace.

## Evidence and identities

[Completed capsule](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/):
[command log](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/command.log),
[launch plan](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results.launch-plan.json),
[machine card](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results/paired-pooling-card.json),
[input ledger](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results/input-manifest.json),
[durable comparison manifest](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results/comparison-manifest.json),
[VALIDATION](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results/validation-report.json),
[TEST](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results/report.json),
[stress](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results/stress-report.json),
[input preservation](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results/input-integrity-after.json).
The generic machine-card filename remains `paired-pooling-card.json`; its
recorded protocol is `context-deletion-v1` and fresh namespace is
`context-deletion-v1/fresh-testing`.

The 61-file compiled source snapshot fingerprint is
`d1f1cf3a754b5657619982b3d31eec02e96b371b299fcdd56fff7c19fea69178`.
Recorded HEAD `1c95eaad83709d8c132f873bce08fdf7feea05f8`, dirty before the C
source commit; the preserved snapshot identifies the measurement.
Unchanged human card SHA256:
`74bb4558b0431bf8209a76cfc68a94112fa00a4f8cbda504746ca0db98a0ec80`.
Machine card SHA256:
`bd804c56873606869fe5b532875268f1302e0b65a44d335422a44ed3624e3230`.
Durable comparison manifest SHA256:
`e62c83c21d068ef10d9e31cfe790e1d0d4da166795a79b0625bb1cd533539552`.
TEST report SHA256:
`3d50c1df893e7e8ff903c52a7010c2f4bc38688413ff7a11ab23a78893c2a80f`.
Stress report SHA256:
`c93936d86d4acbffd0bb953a6d591bf96764a7e538c3454c9e36c90aeb4631af`.

Admission preserves both [enabled context CUDA test evidence](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/admission/build-and-tests.log)
and the separate [mode 2 native-serving CUDA gate](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/results-cuda-gate/gpu-check.json).
The default serving gate alone is not evidence of augmented training.
[Admission record](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/admission/passed.json)
SHA256 `41b1c107c52bfd198104d2ec69bc1fbf9f6042401532b1a2d0c121085d10bd6b`;
enabled test log SHA256
`8cc6cc52132935920b869c8a77cda55c697db8617da353db4f12d27bdfcf9c7b`.
Its 73-file production/test manifest SHA256 is
`a33eeba0f0b4b5fc08fa57b347ac899a3c1415b1ee190738abcd55dba3b37a3c`.
These source-bound tests cover deterministic masks/repair/unchanged target
loss, actual enabled CUDA updates, default-path parity, frozen ordinary inference,
tagged resume rejection and legacy checkpoint contracts.

| Master | Candidate point 0 checkpoint SHA256 | Candidate point 512 checkpoint SHA256 |
| --- | --- | --- |
| 3101 | `6698c53472289f406218b07016e4e8d2f424765db33b3c9ddcf04c60d9a96ee6` | `f995739e4ec90f411ec153f2382c9e5c3afc8104675bc1d1f018960cde644c18` |
| 3202 | `8f687e4d7d58287d105a7f2753f8cc0dffb239d81e3b176f857420fbf5e0e625` | `8eec0ccdc28f36c79e0692acbd03ce862f3d02307a86925fc3e762d8bddebc38` |
| 3303 | `5a02526083014d35e3db331e10b96bd7e66b784a520a435ed06100efb1f9f0d5` | `2c81b7a728cf7ea11bbd1a6b4d3b98feba170d8bafb35ceb609874ea364926bd` |

Retained v4 checkpoint/input identities are pinned in the frozen card and
original 4,596-file inventory SHA256
`c9922d3c817630da3d8609b7ba8d2b47cab7434a17fe28bf5ae272a841da0d95`.
New [artifact inventory](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/artifact-integrity.json):
1,411 files, 402,953,359 bytes, excluding itself and including the final command
log; SHA256 `62415443de5db9ebe7509d31d79b3bbab8d95efa3bf72b7afde3466b1ae021dd`.

[Independent audit](../../../output/runs/rpb-context-deletion/audit-tools/independent-context-20261007-v1/run-KTbte3-v1/validation.json)
passed 47,751,732 checks without repair or relaxation. JSON SHA256:
`fb2f87cd1f8a89598b65917b94a3b5b4a5a16b32cca98c4d541ebb82412b588d`;
[auditor](../../../output/runs/rpb-context-deletion/audit-tools/independent-context-20261007-v1/validate_context_deletion.py)
SHA256 `e427ce98c0a532559dba3352489ac91a37af4fbc1d0603f01bba94f1fbbebc3f`.
It verifies 3 cohorts, 6 candidate points, 63 fits (18 candidate/45 retained), 144
ordinary prediction checks, 45 VALIDATION/45 TEST pairs, 108 stress cases / 756
stress predictions, 30 reconstruction reductions and 768 disjoint source groups.
All 174 original role files remain identical; retained refits and old TEST
payload reads are zero. Full point 0 features/fits/predictions match, context
counts and typed/text policy companions agree, and logged original query target
counts match retained traces.

The independent reader replays CPU fitting statistics, both fixed heads,
preprocessing, masks, support, scores and reconstruction reductions. Bootstrap
estimates/populations/bounds are checked, not independent draw regeneration.
Real CUDA model/optimizer/checkpoint semantics and policy implementation rely
on the separately hash-bound coordinated C++ tests/gates, not an independent
binary execution or operating-system trace by the archive reader.
