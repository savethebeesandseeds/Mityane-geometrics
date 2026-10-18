# Context deletion v1 development card

Protocol: `context-deletion-v1`. Evaluation policy:1.2. Stage: development.
Status: frozen prospective recipe before candidate measurements.
No consumer acceptance or automatic promotion.

This card compares one training-view mechanism, RPB-v6, with the exact retained
RPB-v4 point0/512 checkpoints and TRAIN-fitted readouts from
[native-curve-5G8O5c](../../../output/runs/rpb-native-curve/native-curve-5G8O5c/).
The inference architecture is unchanged. The completed
[RPB-v5 optimization diagnosis](../../encoders/raw_patch_bottleneck_mae/OPTIMIZATION_DIAGNOSTIC.md)
and [fixed512 rejection](../../encoders/raw_patch_bottleneck_mae/PAIRED_POOLING_ADVANCE.md)
remain separate preserved evidence. This card neither rescues v5 nor uses an
old TEST archive as an input. RPB-v4 remains active pending a subsequent explicit
decision and sanity review.

## Scope and hypothesis

Measure timing (`lag_sign`) only, with masters3101/3202/3303. Train one fresh
RPB-v6 trajectory per master continuously to exactly512 completed, unskipped
updates, retaining point0 as an initialization diagnostic. RPB-v4 point0/512
are frozen references; do not retrain them or open new direction, level or
amplitude TEST sources. All three masters and head repetitions are retained.

The hypothesis is that extra missing visible context during training improves
the native32 linear readout and moderate missingness, while preserving the
reconstruction target contract and v4 inference capacity. The v5 diagnosis
showed better reconstruction with additional optimization without a mean linear
VALIDATION gain; it does not establish that this new mechanism will work.

The two primary measurements are native32 Ridge accuracy on intact fresh TEST
and on the same additional30% coordinate-deletion case. Neural accuracy and
the other fixed stress cases are secondary diagnosis. No favorable head,
master, budget or stress case is selected after scores.

## Unchanged architecture and fixed budgets

Use [learned_global.conf](../../encoders/raw_patch_bottleneck_mae/config/learned_global.conf)
with actual CUDA and a512-update ceiling. Keep C3/H32/F3, patch8, width64,
three temporal blocks, four attention heads, feed-forward256, one aligned
channel mixer, decoder128, global mode2 and native32. Preserve dropout0,
layer-normalization epsilon1e-5, original mask ratio0.25, masked Huber delta1,
TRAIN scaler floor1e-6, AdamW0.001/weight decay0.0001, clipping1 and batch8.
The model has225,805 parameters, exactly as v4: no capacity or serving-width
increase. Decode solely the exact served global32 and existing channel/patch
metadata, with no per-channel decoder bypass or additional objective.

Only the training context view changes. Keep model/configuration constructor
order and pair **every mode2 point0 parameter and buffer**, including the global
pooling head, exactly with the corresponding retained v4 initialization.
Verify the original frozen TRAIN scaler, dataset/schema, semantic channel IDs,
initialization seed, ordered source manifest, row sampling, original masking
and per-attempt Torch RNG policies. Unlike the v5 structural comparison,
there is no excluded new pooling module in this equality check.

Use the original first512 row/query-mask counter draws and one continuous AdamW
state. Attempted=completed=512 is required at admission; a skipped attempt
breaks the retained prefix pairing and rejects the comparison before fresh TEST.
The fresh candidate fits its scaler once on that same TRAIN and must match
the retained scaler tensors exactly; reference/control scalers never refit.
Engineering, snapshots and readout measurement restore ambient CPU/all-CUDA
RNG states. Do not reseed or refit the scaler to obtain a favorable run.

## Exact training context-deletion policy

Versioned policy: `rpb-training-context-deletion-v1`, enabled only for the fresh
mode2/mixer1/native32 training adapter. Disabled/default behavior is historical
training. Original observation mask `O`, hidden-patch request `A`, target
`Q=O&A`, original eligible channels, target counts and hierarchical Huber
denominators remain exact. Start with original visible context `V0=O&~A` and
request extra deletions only within `V0`; neither query targets nor naturally
missing cells can be made visible.

The request rate is0.30. Requests are independent of values, labels, held-out
data and Torch RNG. For each minibatch, every B/C/H/F coordinate has a fixed
ordinal, including unobserved and target coordinates:

`ordinal=((b*C+ascending_semantic_channel_rank)*H+h)*F+f`.

Use `base=counter_seed(actual_training_seed, absolute_attempt, stream)` and
`u=(mixed(base+ordinal)>>11)*2^-53`. Request deletion where `V0 && u<0.30`.
The dedicated stream is `0x6374782d64726f70ULL` (decimal7166485043407384432).
Physical channel storage permutations cannot shift semantic draws. The row
ordinal is its sampled minibatch position, not a scoring label or source group;
this training policy does not require transformed variants to share requests.

Support repair only clears requested **extra** deletions in an originally
eligible channel that otherwise retains fewer than two original patch groups.
Visit original patch indices in ascending order; for the earliest fully erased
originally visible patch, retain its first original-visible h/f coordinate,
continuing until two groups remain. Restore no original query or naturally
missing support. This is the implemented deterministic constrained policy,
not unconstrained exact30% realized erasure or a value-dependent repair.

Pass `V=V0&~E` with all nonvisible data zeroed to encode. Decode exact served32,
then use original legally observed standardized targets and the unchanged
hierarchical Huber calculation on original `Q` and eligibility. The decoder
never receives hidden target values as a second signal.

Preserve text companions `training_policy_id`, `context_deletion_ratio`,
`context_deletion_stream`, `context_deletion_rng_policy`,
`context_deletion_repair_policy`, `context_deletion_visibility_policy`,
`context_deletion_count_policy` and `context_deletion_resume_policy`.
Save typed `context_deletion_ratio_value`/`context_deletion_stream_value` and
cumulative `context_requested_deleted_coordinates`,
`context_actual_deleted_coordinates`, `context_restored_coordinates` in each
checkpoint audit. Counts are zero atpoint0, cumulative over eligible forward
batches and satisfy actual=requested-restored. Report request/realized/repair
counts; do not present the requested rate as an exact empirical fraction.

Ordinary checkpoints carry this nonempty training-policy ID. Ordinary
`run_cli train --resume` must reject it, preventing a silent switch to historical
training. This first comparison supports only fresh continuous training; no
augmented-resume API is authorized. Inference/embedding export may load the
tagged checkpoint while retaining its policy provenance, with context deletion
disabled and the ordinary mode2 serving contract.

## Retained data, fitting and immutable references

Each master reuses its exact128 TRAIN source pairs (256 examples) and64
VALIDATION pairs (128 examples), natural missingness0.1, float64 CPU raw
C3/H32/F3, semantic channel IDs0/1/2, unitless features and sampling interval1.
Both opposite-label variants of a source remain within one split and share
their base observation mask. Provider training receives legal TRAIN observations
and metadata only, without labels, clean hidden values or held-out observations.
Any required `clean` schema field is only a clone of zero-masked legal
observations, never hidden ground truth or a fitting/target source.

Pin the retained4,596-file inventory SHA256
`c9922d3c817630da3d8609b7ba8d2b47cab7434a17fe28bf5ae272a841da0d95` and
original49-file source snapshot
`5a2c39d343e399c549b89f880c7e3a9e4b823d17b541775b7352c2de6ce8d643`.
Bind only explicit TRAIN/VALIDATION/checkpoint/scaler/head/manifest roles,
including original point0/512 native exports and VALIDATION prediction witnesses,
before candidate training/fitting. The instantiated ledger records paths, byte
sizes and SHA256. No old TEST/stress payload is discovered, hashed or opened.
The inventory metadata may be read to resolve permitted roles.

Retained v4 point512 checkpoints are:

| Master | Checkpoint SHA256 |
| --- | --- |
| 3101 | `71db828e63884956024a20b8223ef89a3da1b3f031a7fe84d36cc6c1cdac351b` |
| 3202 | `8cbce9f90c3c36146c2102955b279ab587b9569a0519f7815dfd0eec9e44ab73` |
| 3303 | `8bdf10f69f7c927546707f9fcabf6269d4401fb63425fb1da5c4877c9b303904` |

Load retained v4 point0/512 outer/probe normalizers and heads, raw observation
scaler, raw/PCA32/metadata heads and PCA map directly from tensors, with no
fitting constructors. Verify their original VALIDATION predictions, native
exports, fit identities and immutable snapshot/reconstruction witnesses before
fresh TEST. The historical resolved step ceiling2048 stays provenance; actual
reference counters are0/512, and no new budget selection occurs.

Fit candidate native heads atpoint0/512 on valid TRAIN rows only. No PCA/random
projection follows either encoder. The fixed pipeline is outer TRAIN
FeatureNormalizer followed by each probe's own TRAIN normalizer. Ridge1 is
primary; the neural secondary is tanh16/Adam0.01/100 updates. Keep all repetitions
rep1/rep2/rep3 with seeds2701/2802/2903; actual seed is
`stream_seed(declared_probe_seed,input_width)`. Native32 and standalone PCA32
have66/562 Ridge/neural parameters; raw576 has1,154/9,266. Metadata288 is a
labelled control, not an embedding improvement. Unsupported fits retain names,
reasons and coverage; a primary admission failure preserves artifacts and
does not authorize another master, width, head or budget.

## Admission, fresh TEST and fixed readouts

Persist human/machine cards, resolved settings, retained-input ledger and
compiled source snapshot before training. Admission requires both actual-CUDA
**enabled context-deletion training contract tests** and the existing mode2
native-serving CUDA gate. The default serving gate alone does not establish
augmented training. Preserve evidence of unchanged original Q/target reductions,
support-only deterministic requests/repair, finite CUDA gradients/updates,
full initialization pairing, missing-data/semantic-order safety and default
checkpoint/export regressions. Snapshot extraction and reconstruction use
ordinary unaugmented inference.

After all candidate TRAIN/VALIDATION measurements and exact retained/init
feature/readout/reconstruction witnesses, write a durable fixed512 comparison
manifest. Only then generate each master's64 fresh TEST source pairs (128
examples). Namespace: `context-deletion-v1/fresh-testing`; stream
`0x6374763174657374ULL` (ASCII `ctv1test`, decimal7166482861831582580), actual seed
`stream_seed(master,stream)`. It differs from all earlier native/pooling TEST
namespaces. Require source-group disjointness across TRAIN/VALIDATION/new TEST.
Candidate/reference/controls share the same legal observations, masks, channel
IDs and row ordering. Do not draw another task or regenerate an unfavorable
cohort. The legal raw oracle must have supported accuracy at least0.95; record
its coverage, and preserve any failed draw without replacement.

Score seven methods: trained candidate/reference, their exact respective
point0 controls, retained raw576/PCA32 and mask metadata. Use the unchanged
[fixed-readout stress recipe](fixed_readout_stress_v1.md): intact; additional
random deletion10/30/60/90%; contiguous gaps25/50/75% of H; semantic channel0/1/2
absent; all absent. The requested masks are common across all methods and both
source variants. Evaluation stress applies no training support-repair policy.
Only intact and additional30% are primaries; additional30% on natural10%
missingness has approximately37% expected total missingness.

Frozen TEST/stress scalers, maps, normalizers, model weights and heads never
refit. Intact stress predictions/validity must exactly equal ordinary TEST.
Allmissing signal exports are exact zero/invalid; metadata may keep structural
validity as a control. Compare candidate AND reference validity on the same
rows, require equal coverage and report full-population correctness with
abstention counted as incorrect, class/source counts and pair populations.

Retain95%/1,000-replicate source-group paired percentile intervals, keeping
both variants together. Uncertainty is within-master conditional on the fitted
checkpoint/head. Report every master and spread; do not invent an across-master
CI or treat three head fits as three encoder runs.

## Reconstruction, cost and conservative disposition

Use the same unaugmented four-original-patch query for candidate/reference
TRAIN/VALIDATION/TEST: hide each patch across channels, target observed cells
only, preserve natural missingness and require at least two retained visible
patch groups in eligible channels. Score frozen TRAIN-scaler standardized
MAE/Huber1 with equal cell -> channel -> example reductions. Decode only exact
native32; save masks, eligibility, per-example losses and denominators.

Both architectures have225,805 parameters. Context planning/repair can add
training overhead; record synchronized candidate training time and actual
CUDA counts. Frozen v4 training time is historical, not a contemporary repeated
cost control. Do not claim measured inference/memory equivalence or invent a
post-score speed threshold from equal parameter count.

This lead-retention rule is development governance, not consumer acceptance.
Average head repetitions within each master, then the three masters equally.
Retain v6 as a research lead only when:

1. Both mean primary Ridge effects, intact and additional30% deletion, are
   strictly positive versus frozen v4 on the common new TEST population.
2. Coverage is equal and the minimum score among all three candidate masters
   is no lower than the corresponding reference minimum for either primary.
   Report each paired-master gain/loss as well; a mean cannot hide a loss.
3. Mean fixed-query TRAIN MAE and mean VALIDATION MAE at512 are each no worse
   than the reference under the same scaler/query/target contract.
4. Full initialization/stream/policy/target/support/decoder/CUDA/asset/source
   gates and the independent audit pass, without dropping masters or heads.

A nonpositive mean primary rejects the claimed advance under this card.
Coverage, worst-master, reconstruction or unresolved cost tradeoffs remain
explicitly unresolved. Neural-only/best-master gains cannot rescue primaries.
Crossing-zero individual conditional intervals make any improvement claim
ambiguous even when the point-score rule passes: a research lead is not a
confirmed improvement. No silent promotion, head tuning or replacement seed
is permitted. RPB-v4 remains active pending subsequent sanity/regression review
and an explicit promotion decision.

These reused development masters and controlled synthetic tasks do not establish
consumer chronology, cost limits or acceptance. New TEST is a held-out source
replication of a development hypothesis, not fresh-master confirmation.

## Required artifacts

Claim an exclusively new capsule. Preserve cards/source/input hashes;
actual enabled-training and native-serving gate evidence; candidate0/512
ordinary policy-tagged checkpoints and CPU audit companions; full pairing and
context request/actual/repair counters; exact retained fitting tensors;
TRAIN-only candidate heads; legal features/labels/source IDs and predictions;
fixed-budget manifest before TEST; fixed query arrays/reductions; all ordinary
and stress populations/masks/predictions; per-master paired scores/intervals;
cost and input preservation records; independent stdlib verification; and
the measured disposition. Never alter earlier reports/cards. Follow the
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md): tag-only
embedding cells and short descriptions in adjacent prose.
