# Early mixer reliability v1

Protocol: `early-mixer-reliability-v1`. Prospective development card,
9 October 2026 (Asia/Dubai). Freeze this card, the complete source closure,
actual CUDA engineering admission and independently reviewed reader before
generating these cohorts. No historical quality payload is an analysis input.

## Question and single variation

Does mixing aligned channels before the temporal blocks improve the reliability
of the same masked autoencoder's timing representation? The saved-TRAIN
diagnosis was inconclusive about mechanism; this is a bounded architecture
experiment, not a claim that earlier mixing has already been justified causally.
The completed amplitude check established bounded accessibility, with a strong
untrained control. It does not select this architecture or imply generality.

Compare a fresh **RPB-v7.alt-02** late-mixer group with **RPB-v10**, earlier
aligned channel mixing. Preserve every original v7, alt-01 and historical
checkpoint, scaler, reader, card and result. The only architectural change is
`channel_mixer_placement=1`: projected patches with original positional and
semantic channel embeddings enter the existing aligned mixer before the same
temporal blocks. The default placement 0 keeps the literal current path.

Keep mode2/mixer1/dropout0, native32 and all 225,805 named parameters and their
registration order. Preserve genuinely independent `z_local` using a separate
unmixed temporal pass. Do not change its autograd contract. Measure the added
work; equal parameters do not mean equal compute. No widened representation,
alternative pooling, auxiliary loss, gain augmentation or head tuning occurs.

## Identity and compatibility

Parse placement separately and omit default 0 from canonical settings and old
archive schemas. Placement 1 requires a typed int64 placement, distinct
architecture ID `aligned-mixer-before-temporal-v1`, and matching feature and
reconstruction semantics, validated before loading weights. Restrict this first
early candidate to mode2/mixer1/dropout0 and the existing coordinate15 policy.
Its audit/provenance tag is RPB-v10; do not label it RPB-v7. Old v4/v7 adapters
and historical checkpoint admission paths explicitly reject nonzero placement.
Ordinary historical default settings, initialization, serving and update behavior
must remain compatible. Existing source fingerprints remain their old scopes.

## Matched learning and new data

Five timing masters: 19119/20220/21321/22422/23523. For each, generate unchanged
`Task::lag_sign` TRAIN128 source pairs and VALIDATION64 pairs (256/128 rows),
C3/H32/F3 float64 CPU, natural missingness .10, semantic IDs0/1/2, interval1,
endpoint31 and zero hidden storage. Prefix IDs with this protocol. No TEST.
Both encoders use exactly the same legal label-free TRAIN values, source order,
original TRAIN-fitted scaler, initialization seed and counter seeds. Save
point0 witnesses proving equality of all named parameters/buffers/scaler;
initial features can differ because the architectures differ.
Encoder/scaler fit metadata is restricted to the original timing TRAIN namespace
`early-mixer-reliability-v1/lag_sign`; amplitude head data cannot enter it.

Train each of ten trajectories once on CUDA to exactly512 completed/unskipped
updates, batch8, AdamW .001/weight decay .0001/gradient clip1/Huber delta1,
threads1. Keep the existing coordinate15 context deletion mechanics, target
query Q, eligibility, patch sampling, repair and deterministic row/mask streams.
Per-step attempted/completed/target-cell trace counts and cumulative context
deletion counters must match exactly between the pair. Exact deterministic
target/mask generation is bound by source and paired artificial CUDA admission;
this protocol does not claim a saved full target-mask witness at every update.
Save all512 trace entries, point0/512 checkpoints and companions. No additional
decoder calibration, continuation, rate grid or rescue budget. No labels enter
encoder fitting. Total planned sampled rows:40,960 across ten trajectories.

Separate amplitude data masters24624/25725/26826/27927/29028, paired in order,
use the unchanged amplitude generator, with the same TRAIN/VALIDATION sizes and
natural masks. These are new head-fitting/evaluation data, never encoder fitting
data. The networks retain their timing TRAIN scalers. TRAIN and VALIDATION
sources/streams and both task namespaces are disjoint.

For each task, score intact VALIDATION and exactly one extra .30 coordinate
deletion view shared by all methods. Timing deletion stream is
`stream_seed(timing_master,0x656d782d74696d30ULL)`; amplitude deletion stream is
`stream_seed(amplitude_master,0x656d782d616d7030ULL)`. Namespaces are
`early-mixer-reliability-v1/lag_sign/validation-coordinate-dropout` and
`early-mixer-reliability-v1/amplitude/validation-coordinate-dropout`.

## Reusable evaluation

Seven methods per task: raw576 (288 TRAIN-scaled values plus288 masks), raw
PCA32 only, mask288, untrained late32, untrained early32, trained late32 and
trained early32. Native outputs have no PCA. Raw/PCA preprocessing fits each
task's TRAIN only. Retain both initial controls rather than assuming a shared
untrained representation from equal weights.

Use unchanged shared ridge penalty1 and neural tanh16/Adam .01/100 updates,
repetitions2701/2802/2903, actual paired `stream_seed(rep,width)`. Same existing
outer/probe TRAIN normalization rules. Fit each head once and reuse both views;
save maps, logits, classes, support, labels and source order. Unsupported fits
remain explicit. Planned210 pipelines/420 heads across seven methods, five
cohorts, two tasks and three repetitions; report actual supported counts.

Add explicit generic reference/candidate comparison names to the shared
readout wrapper while preserving its historical default v4/v7 behavior. The
helper remains independent of encoder implementations. Within-master paired
source bootstrap1000/95% uses its unchanged conditional fixed-head population
and seed rules; these are not five-encoder uncertainty intervals. Report all
five cohorts, equal-master means/min/max, and neural repetitions averaged
within cohort before cohort summaries. Do not average the two tasks together.

Only timing TRAIN and intact VALIDATION get the unchanged original-patch query
reconstruction evaluation through `write_native_patch_reconstruction`. Preserve
Q/visibility/eligibility/support/target arrays byte-for-byte between architectures.
Here eligibility is the saved channel/target eligibility; the ordinary original
patch query evaluation adds no training-context deletion tensor. Use each original
encoder512 decoder, with no extra decoder optimization. The Q-masked contexts
require distinct CUDA encodes; those are necessary query forwards, not repeated
full-observation feature extraction. Save reconstruction arrays and original
hierarchical MAE reductions separately from sampled training Huber losses.

## Gates, arithmetic and decisions

Use a new strictly protocol-bound CUDA snapshot adapter for native features and
query reconstruction. Do not use the historical CPU serving snapshots. Freeze
parameters/buffers/scaler and parent bytes; eval/no-grad for measurement, actual
CUDA input/output/loss/gradients/updates for training. Save each needed export
once, then reuse it for heads and CPU saved arithmetic.

Before quality generation, admit artificial CUDA fixtures covering default
placement parity, exact paired initialization, identity-mixer equivalence,
independent local/context interaction, original-patch alignment and semantic
permutations, hidden storage/gradient isolation, missing support, exact served
BD32 decoder input, gradients and continuous0→2→4 versus0→4 AdamW, immutable
snapshots, malformed placement/semantics rejection and old-loader rejection.
Preserve failed admissions additively.

Independent validation executes no model, optimizer, head fitting or PCA/SVD
fit. Verify closed artifact/source/admission identities, saved initialization,
traces/counters/policies, scalers, raw/PCA/readout arithmetic, masks/support,
query reductions and source-paired effects. CUDA checkpoint bodies remain byte
bound; typed CPU companions and actual CUDA admission/source establish their
training/architecture semantics. Float64 replay tolerance is abs2e-9+rel2e-9;
float32 operation replay tolerance is abs2e-6+rel2e-5. Saved logits' own argmax,
classes/support/source order and promised equality witnesses are exact.

Timing intact/deleted linear accuracy, worst-cohort accuracy and timing
TRAIN/VALIDATION MAE are the main comparison. Report neural heads and amplitude
transfer alongside them, including any regression. This bounded experiment has
no numeric automatic promotion/selection guard and opens no TEST or stress.
No quality-dependent head, objective, deletion-rate, seed-subset or budget
change is permitted within this card. Save and assess the whole result before
declaring any further separately versioned experiment.

Use exclusive capsules under `output/runs/rpb-early-mixer-reliability`. Report
CUDA training, full-feature inference, necessary query inference, binding/I/O,
CPU baseline/head/bootstrap work and independent audit costs separately, with
mixed wall time labelled honestly and pure GPU compute unmeasured when absent.
Use the standard quality/training tables and short version descriptions outside
their cells. Register exact trained-group/checkpoint identities; no original
saved encoder is replaced by these new groups.
