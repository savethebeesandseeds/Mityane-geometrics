# Paired pooling v1 development card

Protocol: `paired-pooling-v1`. Evaluation policy: 1.2. Stage: development.
Status: frozen recipe before candidate measurements. No consumer acceptance.

This card compares one new pooling mechanism, RPB-v5, with the exact retained
RPB-v4 checkpoints and TRAIN-fitted readouts from
[native-curve-5G8O5c](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/).
The [experiment record](../../encoders/raw_patch_bottleneck_mae/PAIRED_POOLING_ADVANCE.md)
records the hypothesis, planned evidence and eventual disposition. The
[native-curve record](../../encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md)
remains the completed RPB-v4 experiment.

## Scope and hypothesis

Only timing (`lag_sign`) is measured on new TEST sources in this first comparison.
Use all three existing masters, 3101/3202/3303. Train one RPB-v5 instance per
master to exactly 512 completed updates, retaining its point-zero checkpoint.
RPB-v4 point zero and 512 are frozen references; no RPB-v4 retraining occurs.
Do not open new direction, level or amplitude TEST sources under this card.
Candidate engineering/regression checks on permitted TRAIN/VALIDATION data may
follow before any promotion and must be labelled separately.

RPB-v4 already mixes channels at aligned original patch coordinates. It then
compresses each channel to a 32-number summary before its global MLP. RPB-v5
feeds the aligned patch states directly to a global MLP. The hypothesis is that
bypassing that intermediate compression improves timing access and moderate
missingness at the same served32 bottleneck. Existing measurements do not prove
that summary compression caused the gap; this is one controlled mechanism test.

The two declared primary measurements are native32 ridge accuracy on intact
fresh TEST and on the same additional30% coordinate-deletion stress case.
Neural accuracy is secondary. The twelve-case fixed-readout stress sweep is
retained as diagnosis, without selecting a favorable case after scores.

## Exact mechanism and cost

Keep C3/H32/F3, patch8, encoder width64, three temporal blocks, four attention
heads, feed-forward256, one aligned channel-mixer block and decoder128. Keep
dropout0, layer-normalization epsilon1e-5, mask ratio0.25, masked Huber delta1,
TRAIN scaler floor1e-6, AdamW0.001/weight-decay0.0001 and gradient clip1.
Train on the verified RTX A2000 CUDA device at batch8 with one continuous AdamW
state. The decoder consumes only the exact served BD32 signal plus its existing
channel/patch metadata. No additional loss, skip connection or decoder signal
is introduced.

RPB-v5 uses core global mode3 with mixer1. Restore mixed visible tokens to
original patch positions, canonicalize channels by configured semantic IDs,
and form B,C,K,W with K4/W64. Absent slots are exact zero. The MLP input contains
**all canonical C*K*W state values first, followed by C*K slot-visible bits**;
bits are not interleaved with each state. The layer sequence is
`Linear(C*K*(W+1), W) -> GELU -> Linear(W, D)` with D32: 780 -> 64 -> 32.
The visible bits derive only from the encoder's legal observed/visible mask.
Canonical channel order is 0/1/2 and K retains original patch indices.

RPB-v4's pooling MLP has 8,480 parameters; RPB-v5's has 52,064. Total parameters
are 225,805 and 269,389, respectively: +43,584 (+19.30%). This larger internal
pooling head is explicitly accepted for the bounded test. Equal export/head
width is not equal encoder capacity. Record CUDA parameter counts, synchronized
encoder training time and any measured memory/inference costs; the retained v4
training time is historical and does not establish a contemporaneous speed
comparison. No undisclosed speed threshold or speed advantage is assumed.
This contrast changes state resolution, slot-level support conditioning and
pooling-head capacity together. A gain would support this complete pooling
mechanism; it would not isolate compression avoidance from those other changes.

Per-channel D exports remain diagnostics. They are not the v5 reconstruction
bottleneck and do not receive reconstruction gradients through the bypassed
summary branch. The primary native signal is exactly the mode3 global32.

## Retained cohorts and source binding

Load the exact saved timing TRAIN and VALIDATION observations, masks, scoring
labels and ordered source IDs. Each master has 128 TRAIN source pairs (256
examples) and 64 VALIDATION pairs (128 examples), C3/H32/F3 float64 CPU raw
storage, 10% natural missingness, unitless feature units and sampling interval1.
Two opposite-label transformations of a source share its visibility mask.
Neither transformed variant crosses a source split. Labels are available to
readouts/scoring only; provider fitting receives legal TRAIN observations and
metadata, with no labels, clean hidden values or held-out observations.
If the generic stress container requires a `clean` field for schema validation,
fill it with a clone of legal, zero-masked observations only. This is a schema
placeholder, never hidden ground truth, a fitting input or a reconstruction
target source. The scoring targets remain original legally observed cells.

Pin the original capsule's integrity manifest SHA256:
`c9922d3c817630da3d8609b7ba8d2b47cab7434a17fe28bf5ae272a841da0d95`.
It binds 4,596 preserved files. The original 49-file source snapshot fingerprint
is `5a2c39d343e399c549b89f880c7e3a9e4b823d17b541775b7352c2de6ce8d643`.
The original independent audit SHA256 is
`fce69cb11e56886e91f32e1278264cc6292d0c1964388da40ebe96567a3c6d49`.

The production launch plan must enumerate every retained input with role,
path, SHA256 and byte size before candidate fitting. Inputs include each
`seed-M-lag_sign/controlled-training.pt` and `controlled-validation.pt`, original
development manifest, point0/512 checkpoint and frozen scaler assets, corresponding
native TRAIN/VALIDATION exports, three TRAIN-fitted reference heads and their
VALIDATION prediction witnesses, and original raw/PCA/metadata fitting assets.
Resolve entries against the pinned integrity manifest; refuse a missing, altered,
unexpected or ambiguous role. Preserve the original files byte for byte.

Retained point512 checkpoint hashes are:

| Master | RPB-v4 checkpoint SHA256 |
| --- | --- |
| 3101 | `71db828e63884956024a20b8223ef89a3da1b3f031a7fe84d36cc6c1cdac351b` |
| 3202 | `8cbce9f90c3c36146c2102955b279ab587b9569a0519f7815dfd0eec9e44ab73` |
| 3303 | `8bdf10f69f7c927546707f9fcabf6269d4401fb63425fb1da5c4877c9b303904` |

The reference checkpoints' original resolved step ceiling2048 remains provenance;
their actual retained counters are 0 or512. Candidate training stops at512.
No old TEST archive is an input. Its opened scores informed this development
hypothesis and cannot be represented as unseen confirmation for it.

## Pairing, fitting and fixed budget

Pair the common temporal, aligned-mixer, per-channel diagnostic and decoder
modules at initialization by original master seed and the existing initialization
stream. Exclude only the structurally different global-pooling heads from the
named-weight equality comparison. Check exact common-module equality against
the original v4 point-zero checkpoint. Do not shift shared initialization by
registering the new head earlier or changing the common constructor order.

Use the original counter-derived row, masking and per-attempt Torch RNG policy
and seeds. The first512 sampling/mask draws must match the original v4 prefix;
an unmatched attempt/skip trajectory is a pairing failure, not an excuse to
replace a seed. Record attempts/completed updates, sampled rows and RNG policy.
Use the same legal TRAIN preprocessing; candidate fitting occurs once on TRAIN
only, and its frozen scaler must match the retained reference's statistics.
Engineering/snapshot/readout measurements must restore ambient CPU/CUDA RNG
states so they cannot alter the continuous training path.

Restore existing v4 point0/512 outer normalizers, Ridge/Tiny parameters and all
raw/PCA/metadata controls **from saved tensors, without fitting constructors**.
The frozen raw observation scaler, raw outer normalizers, standalone PCA32 map
and controls remain unchanged. Verify their cached VALIDATION predictions and
fit hashes before using new TEST. Do not refit a reference to make it match the
candidate or this card's names.

Fit candidate native heads at point0 and512 only on the same TRAIN rows. The
pipeline remains outer TRAIN FeatureNormalizer then each probe's own TRAIN
normalizer, with **no PCA or random projection after either encoder**. Ridge
penalty1 is primary. The neural secondary is tanh16 with Adam0.01/100 updates.
Repetitions are rep1/rep2/rep3, seeds2701/2802/2903; actual seed is
`stream_seed(declared_probe_seed, actual_input_width)`. Equal-width native32,
initialization32 and PCA-only32 are paired. Each has 66 Ridge/562 neural
parameters; raw576 has 1,154/9,266 and metadata288 remains a labelled control.
Retain every repetition, including unsupported fits; never choose a best head.

There is no checkpoint-budget selection sweep. Persist a durable fixed512
manifest after TRAIN/VALIDATION measurement, retained snapshot/readout witnesses
and initialization/pairing audits, **before every fresh TEST draw**. No
VALIDATION, TEST, nonlinear, reconstruction or stress score changes that budget.
An unsupported primary fit or pairing/engineering failure aborts this measured
comparison and preserves its diagnostic artifacts; it does not authorize a new
seed, width or budget under the same card.

## Fresh TEST and fixed-readout stress

Use namespace `paired-pooling-v1/fresh-testing`, stream
`0x7070763174657374ULL` (ASCII `ppv1test`, decimal8102105684417803124), with actual
seed `stream_seed(master, stream)`. This differs from native-curve-v1's `ncv1test`
namespace. Generate 64 new source pairs (128 examples) per master only after the
fixed-budget manifest is durable and all retained assets are witnessed. Check
TRAIN/VALIDATION/TEST source IDs are disjoint. Candidate and reference see the
same new sources, masks, semantic IDs and row ordering.

Score trained RPB-v5 and frozen RPB-v4, their exact respective point-zero controls,
the retained raw576/PCA-only32 heads and visibility-only control. TEST heads,
normalizers, scalers and PCA never fit again. Analytic legal-raw rules remain
separately labelled solvability checks. Declare any fail-on-unsolvable gate in
the machine card before generation; failure preserves the draw and does not
authorize a replacement namespace.

Use the unchanged [fixed-readout-stress-v1](fixed_readout_stress_v1.md) recipe:
intact; additional random deletion10/30/60/90%; contiguous gaps25/50/75% of H;
semantic channel0/1/2 absent; and all absent. Three masters x three head
repetitions receive the same twelve cases. Every method and both variants of a
source receive identical requested erasures. Additional30% deletion on natural
10% missingness has approximately37% expected total missingness; it is the
declared moderate robustness primary. Other cases are diagnosis, not alternative
primaries chosen after results.

Stress reuses each exact fitted readout; no model, preprocessing or classifier
fits. Intact predictions/validity must exactly equal ordinary new TEST outputs.
Allmissing signal exports are exactly zero with zero validity; metadata may
retain structural validity and remains a control. Pooling must not infer support
for absent channel/patch slots. Query targets are zeroed before encoding.

Report native coverage and pair-specific common validity, class support and
source-group counts. The trained v5-v4 primary population is candidate AND
reference validity on the same new TEST rows; require equal coverage. Also
report full-population correctness with abstention counted as incorrect. No
argmax placeholder can count as a valid allmissing prediction.

Use 95% source-group paired percentile intervals, 1,000 bootstrap replicates,
conditional on the fitted models/readouts. Retain both source variants together.
Report three encoder-instance scores and their spread; three head fits on one
instance are not three independent encoder runs. Do not construct an
across-master CI by pooling these intervals or select the best seed.

## Reconstruction and engineering evidence

Use the same fixed original-patch query for candidate and reference: enumerate
each of the four patches, hide that patch across channels, score observed target
cells only and leave at least two visible observed patch groups per eligible
channel. Keep natural missingness. Score frozen TRAIN-scaler standardized MAE
and Huber1 with equal cell -> channel -> example reductions. Both decoders read
only their exact served32 export. Keep target/support denominators and query
arrays so any coverage or scaling change is visible.

Save TRAIN/VALIDATION reconstruction per point. Witness candidate point zero
after later training and both frozen reference checkpoints; reproduce native
features, readout predictions and fixed-query reconstruction exactly. Verify
CUDA parameters/inputs/loss/optimizer, finite gradients, changed pooling/decoder
weights, hidden-target isolation, semantic permutation invariance, missing-prefix
patch alignment, absent-slot zeroing and no per-channel bypass. Preserve mode0/1/2
checkpoints, defaults and their exact existing exports.

## Conservative development disposition

This is an experimental lead rule, not consumer acceptance or automatic active
version promotion. Compare means by averaging head repetitions within each
master, then masters equally; deterministic Ridge repetitions do not multiply
the encoder sample size. Report every per-master score and spread.

Retain v5 as a lead for follow-up only when all of these hold:

1. Both mean primary Ridge effects, intact and additional30% deletion, are
   strictly positive versus frozen v4 on the common new TEST population.
2. Coverage is equal and the worst-master score is not lower for either primary.
3. Mean fixed-query TRAIN MAE and mean VALIDATION MAE at512 are each no worse
   than the reference under the same query/scaler/target contract.
4. Initialization, support, exact32-decoding, continuity, immutable asset,
   provenance and independent audit gates pass; all masters/repetitions remain.

Crossing-zero individual source-group intervals do not erase a point estimate,
but make any improvement claim ambiguous. A passing point-score rule can name a
**research lead**, not a confirmed improvement. Report those intervals and any
per-master losses explicitly. A nonpositive mean primary rejects the claimed
advance under this recipe. A coverage, worst-master, reconstruction or cost
tradeoff leaves it unresolved; do not silently promote it, tune heads or replace
a seed to obtain a favorable result. Neural-only or best-seed gains cannot rescue
failed primaries. Document the accepted +43,584 parameter cost and actual timing;
unresolved cost judgment remains explicit rather than inventing a post-score
threshold. RPB-v4 stays active unless a meaningful lead passes the subsequent
sanity/regression review and an explicit promotion decision.

Keep the synthetic and reused-development limits visible. These three encoder
instances reuse known TRAIN/VALIDATION cohorts; fresh TEST provides a new held-out
source replication for a development hypothesis, not fresh-master or consumer
confirmation. Intended consumer chronology, support, cost limits and acceptance
thresholds remain undefined.

## Required artifacts

Use a new exclusively claimed capsule. Persist the human/machine cards, resolved
settings and source snapshot before training; retained-input SHA manifest;
common-initialization/scaler/RNG audits; candidate0/512 ordinary checkpoints and
fixed fitted heads; copied/reference pointers with exact identities; row-level
features/predictions/labels/source IDs; fixed-budget manifest; query arrays and
reconstruction reductions; ordinary new TEST and all stress masks/predictions;
paired scores/intervals/coverage; cost records; independent verification; and
the final measured disposition. Preserve all old files/reports. Follow the
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md) with tag-only
encoder table cells and descriptions in adjacent prose.
