# Context replication v1 development card

Protocol: `context-replication-v1`. Policy: 1.2. Stage: development.
Status: frozen prospective recipe before fresh data generation.
No automatic promotion, consumer acceptance, budget selection or head tuning.

This card repeats the RPB-v4/RPB-v6 comparison at exactly 512 updates on five
new, independently initialized development masters: 4404, 5505, 6606, 7707 and
8808. RPB-v4 is the unchanged learned global native32 encoder; RPB-v6 uses that
same inference architecture with the existing training context-deletion policy.
The [original context comparison](../../encoders/raw_patch_bottleneck_mae/CONTEXT_DELETION_ADVANCE.md)
and [optimization diagnostic](../../encoders/raw_patch_bottleneck_mae/CONTEXT_OPTIMIZATION_DIAGNOSTIC.md)
remain preserved evidence. Their primary gains and reconstruction tradeoff are
not waived by this replication. RPB-v4 remains active pending an explicit decision.

## Question, task and fixed population

Measure timing (`lag_sign`) only. Does the native32 linear gain on intact and
moderately corrupted held-out sources repeat across five fresh training masters
at the same 512-update cost budget, with equal coverage and without worse
original fixed-query TRAIN/VALIDATION reconstruction?

Keep every master, initialization point, head repetition and declared case.
Do not replace an unfavorable master, regenerate a cohort, change the model,
increase the update budget, select a head or add another task after scores.
These masters are new relative to the earlier three-master comparison, but this
is still controlled synthetic development evidence, not consumer confirmation.

For each master generate 128 TRAIN source pairs (256 rows) and 64 VALIDATION
pairs (128 rows), with natural missingness 0.10, through
`make_controlled_development_protocol(lag_sign,shape,128,64,master,0.10)`.
Its TEST tensors are undefined and its TEST source list is empty. Use float64
CPU legal raw observations with C3/H32/F3, semantic channel IDs 0/1/2,
unitless features, sampling interval 1 and endpoint 31. Both opposite-label
variants of a source stay within one split and share their observation mask.
Source groups must be disjoint across TRAIN, VALIDATION, fresh TEST and masters.

## Stage 1: fresh reference preparation, development only

Use the shared `native-development-v1` preparation path, with metadata namespace
`native-development-v1/lag_sign`. It trains only RPB-v4, without context deletion,
at absolute points 0 and 512. Set TEST pair count to 0 and stress sweep to false.
There is no TEST generation, TEST export, stress scoring or checkpoint selection.
Persist the human card, instantiated `native-development-card.json`, resolved settings and
compiled production/admission source snapshot before generating any cohort.

Keep one live AdamW trajectory per master. Point 0 is its exact initialization
diagnostic, not an extra training run or selection candidate. Require attempted
equals completed equals 512 at the positive point; a skipped attempt breaks
prefix pairing and stops the comparison. Preserve all five completed point
0/512 checkpoints, CPU audits, raw TRAIN/scaler archives, native TRAIN/VALIDATION
exports, TRAIN-fitted heads, cached VALIDATION predictions and fixed-query
TRAIN/VALIDATION reconstruction witnesses. Raw/PCA/metadata control fits are
TRAIN-only and remain fixed within each cohort.

Write a durable `development-complete.json` only after the entire five-master,
two-point reference matrix and its fitting/query witnesses are complete.
Reference output is the new capsule's `reference/` directory. Freeze its
permitted development-only files in `reference-inputs.sha256`; save their paths,
byte sizes, SHA256s and the manifest hash in `reference-inputs.json`. Write both
after completion and before any RPB-v6 training or fresh TEST generation.
This new baseline is measured under this card; it is not an old retained score.

Neither stage opens old three-master TEST, stress, model, data or fit payloads.
Historical reports remain descriptive context; previously frozen stdlib reader
source may be reused without executing its old experiment entry points.

## Stage 2: exactly paired RPB-v6 training

Use [learned_global.conf](../../encoders/raw_patch_bottleneck_mae/config/learned_global.conf)
with device CUDA, the declared master seed and a 512-update ceiling. Keep patch
8, encoder width 64, three temporal blocks, four heads, feed-forward width 256,
one aligned channel mixer, decoder width 128, global bottleneck mode 2 and
native export width 32. Both versions have exactly 225,805 parameters.
Preserve dropout 0, layer-normalization epsilon 1e-5, original mask ratio 0.25,
hierarchical Huber delta 1, scaler floor 1e-6, AdamW learning rate 0.001,
weight decay 0.0001, clipping 1, batch 8 and one CPU Torch thread. Record the
full resolved recipe rather than treating these summary fields as instance identity.

For each master pair every point 0 parameter and buffer, including the mode 2
global pooling head, exactly with the new RPB-v4 reference. Use its same legal
TRAIN data, physical channel order, initialization seed, ordered sources,
row-sampling stream, original patch-mask stream and per-attempt Torch RNG policy.
Training metadata namespace is `native-development-v1/lag_sign` for both.
The fresh candidate fits its TRAIN scaler once and must reproduce the reference
scaler tensors and identity exactly. Its first 512 row/query-mask draws must
match the reference prefix; no extra failed or skipped attempt is admitted.

The encoder receives legal TRAIN observations and metadata only, without labels,
clean hidden values or held-out observations. Readout fits may use TRAIN labels;
encoder fitting may not. Any `clean` field required by a shared scoring schema
is an explicitly disclosed clone of zero-masked legal observations, never a
hidden target or fitting source. Extraction, snapshots and readout measurement
restore ambient CPU and all-CUDA RNG states before the next training update.

Keep the unchanged `rpb-training-context-deletion-v1` policy. Its request rate
is 0.30 on original visible context V0=O&~A, with dedicated stream
`0x6374782d64726f70ULL` (decimal 7166485043407384432). The absolute attempt and
semantic-canonical B/C/H/F ordinal determine independent requests; values,
labels, held-out data and Torch RNG do not. Original O, A, Q=O&A, eligible
channels, standardized targets and hierarchical Huber denominators stay exact.
Repair only extra deletions in originally eligible channels to retain two
originally visible patch groups, using the earliest erased original patch and
its first original-visible h/f coordinate. No natural missingness or query
support is restored. Zero nonvisible storage before encode and decode solely
the exact served global32 plus existing metadata, without a decoder bypass.

Save all typed/text policy companions and cumulative requested, actual and
restored coordinate counts at points 0/512; actual equals requested minus
restored, and all counts are zero at point 0. Report realized counts rather
than claiming an exact 30% empirical erasure fraction. Ordinary resume rejects
the policy-tagged checkpoint; this comparison uses fresh continuous training.
Frozen inference disables training deletion and repair and keeps policy provenance.

## Frozen features, classifiers and controls

Load the new reference's point 0/512 fits, raw scaler, raw-PCA map and control
heads directly from identified tensors; do not refit retained assets in stage 2.
Require original VALIDATION prediction and native export parity before fresh
TEST. Fit candidate native heads at points 0/512 on valid TRAIN features only.
No PCA or random projection follows either encoder.

The native pipeline is outer TRAIN FeatureNormalizer followed by the head's own
TRAIN normalizer. Ridge penalty 1 is primary. The neural secondary is tanh
with 16 hidden units, Adam 0.01 and 100 updates. Preserve all three repetitions,
2701/2802/2903, with actual seed `stream_seed(rep_seed,input_width)` shared
between equal-width methods. Native32 and standalone raw-PCA32 have 66 Ridge
and 562 neural parameters; raw576 has 1,154 and 9,266. PCA32 fits only raw TRAIN:
288 standardized observed values plus 288 visibility flags, raw scaler floor
1e-8. Mask metadata is a labelled control, not a signal embedding.

Score seven methods: RPB-v6, RPB-v4, each version's own exact point 0 control,
raw576, PCA32 and mask metadata. Retain the five declared comparisons:
candidate minus reference; candidate minus its initialization; reference minus
its initialization; candidate minus PCA only; reference minus PCA only.
All unsupported fits retain names, reasons and coverage. No unfavorable support
or fit failure authorizes dropping a master/head or drawing a replacement.

Three head fits are not three independently trained encoders. For summaries,
average all three declared head repetitions within each master, then weight
the five masters equally. Report every master score, paired effect and spread.
Do not create a pooled across-master confidence interval under this card.

## Admission and unopened TEST boundary

Admission must pass actual CUDA enabled context-training tests, ordinary mode 2
serving/checkpoint regressions, complete initialization/scaler/stream pairing,
unchanged query/eligibility/Huber checks and default protocol compatibility.
Hash-bind the test log and preserved production/test sources to the binaries.
All 225,805 parameters, minibatch inputs, loss, gradients and active optimizer
moments/steps must be on actual CUDA; record finite updates and changed weights.

After stage 1 completion and frozen input hashes, train and measure all five
RPB-v6 points 0/512. Recheck immutable reference/candidate snapshots, exports,
decoder queries and loaded/fitted heads. Persist the durable fixed-budget
comparison manifest after all development witnesses and before any new TEST
generation. It records budget 512 for every master, not a selected budget.

Then generate exactly 64 fresh TEST source pairs (128 rows) per master using
namespace `context-replication-v1/fresh-testing` and stream
`0x6372763174657374ULL` (ASCII `crv1test`, decimal 7165919911878161268).
The actual fresh TEST seed is `stream_seed(master,stream)`. This namespace has
not been used by the earlier cards. All methods receive the same legal TEST
observations, masks, row/source order and semantic geometry. The ordinary legal
raw oracle must have supported accuracy at least 0.95; record coverage and
preserve any failed draw without regeneration. Do not generate another task.

Both primaries use that same fresh TEST population: intact and additional 30%
coordinate deletion. Use the unchanged
[fixed-readout stress recipe](fixed_readout_stress_v1.md), including its existing
`fixed-readout-stress-v1` source-level deletion stream law. Preserve all 12 cases:
intact; random deletion 10/30/60/90%; contiguous gaps 25/50/75%; semantic channel
0/1/2 absent; and all absent. Only intact and random deletion 30% are primaries;
the other ten conditions and neural scores are fixed secondary diagnostics.
Requested masks are shared across both variants and all methods. Stress applies
no training support repair; 30% additional deletion on 10% natural missingness
has approximately 37% expected total missingness, with actual support reported.
Stress oracle scores are descriptive, not a redraw or replacement gate.

Freeze model weights, scalers, maps, normalizers and heads for ordinary TEST
and every stress case. Intact stress predictions and validity must exactly
equal ordinary TEST. All-absent signal features are exact zero/invalid;
metadata may retain structural control validity. Compare candidate/reference
on their declared common-valid rows and report native/full-population coverage,
class/source support and correctness with abstention counted as failure.

Use within-master paired source-group percentile intervals, 95% confidence and
1,000 draws, keeping both source variants together. These intervals condition
on the saved encoder/head; they are not uncertainty across five training runs.
Preserve every interval and point score. No best master, head or condition is selected.

## Original reconstruction, costs and disposition

Candidate/reference fixed-query TRAIN/VALIDATION/TEST reconstruction uses
ordinary unaugmented inference. Enumerate each of the four original patches,
hide the same patch across channels, target legally observed cells only and
require two retained visible patch groups in eligible channels. Decode only
exact native32. Frozen TRAIN-scaler standardized MAE/Huber 1 uses equal
cell -> channel -> example reductions. Save masks, targets, predictions,
eligibility, counts and per-example losses. The extra deletion classification
case does not change these original-query reconstruction targets.

Both versions are freshly measured in this capsule at exactly 512 updates.
Record actual CUDA parameter/update/sample counts and synchronized training
loop time for each master/version. Disclose CPU planning/transfers/bookkeeping
inside that timer, and preserve admission, loading, export and readout time
separately when measured. Equal parameter count does not measure equal memory,
inference speed or isolated GPU kernel time. Do not reuse old training time as
the new reference timing or introduce a post-score speed threshold.

The conservative research-lead rule requires all of the following:

1. Strictly positive mean native Ridge candidate-minus-reference effects on
   both intact and additional 30% deletion, under the declared equal-master
   aggregation and common populations.
2. Equal candidate/reference coverage and a minimum score across the five
   candidate masters no lower than the reference minimum for either primary.
   Report each paired-master gain/loss; means do not conceal individual losses.
3. Mean original fixed-query TRAIN MAE and mean VALIDATION MAE at 512 each no
   worse than the contemporaneous five-master RPB-v4 reference, using identical
   scaler/query/target contracts. No post-hoc waiver of either reconstruction guard.
4. Full engineering, initialization, stream, policy, query, support, decoder,
   asset/source-preservation and independent audit checks pass without exclusions.

A nonpositive primary mean rejects the claimed advance under this fixed card.
Coverage, worst-master, reconstruction or unresolved cost tradeoffs stay explicit.
Neural-only or best-master gains do not rescue the primaries. Crossing-zero
within-master intervals limit the strength of a claimed gain even if point-score
rules pass; retaining a research lead is not confirmed improvement or acceptance.
No automatic promotion, larger-budget tuning, new objective or replacement seed
is authorized. RPB-v4 remains active until an explicit decision and appropriate
consumer/sanity review under separate authority.

## Required evidence and independent audit

Claim an exclusively new capsule. Preserve the human/instantiated cards,
compiled source and admission evidence before generation; the reference
development-completion record and frozen permitted input ledger before candidate
training; all point 0/512 checkpoints and audits; exact scaler/init/stream/source
pairing; policy/count companions; native features and no-post-encoder-PCA fits;
original prediction/query witnesses; the durable pre-TEST comparison manifest;
fresh TEST/stress masks, labels, source IDs, features, predictions and populations;
all master/head paired effects and intervals; costs and byte-integrity evidence.

The independent stdlib auditor reads only this new capsule's declared artifacts
and pinned helper source. It verifies actual file SHA256/role lineage, counter and
policy evidence, initialization/scaler pairing, TRAIN statistics and frozen
classifier inference, no native PCA, group/population scores and paired estimates,
intact stress parity, hidden-storage safety, fixed-query reductions, and input
preservation. It does not execute a model, fit a head or independently deserialize
CUDA storage; actual CUDA/state association is covered by hash-bound coordinated
tests. Preserve failed audit versions and use separately identified corrections.

Follow the [reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md):
quality and training tables, tag-only encoder cells and short descriptions
immediately beside the tables. Label this experiment new, historical evidence
reused, and unmeasured consumer acceptance explicitly. Do not pool different
tasks into an unspecified encoder accuracy or treat head repetitions as masters.
