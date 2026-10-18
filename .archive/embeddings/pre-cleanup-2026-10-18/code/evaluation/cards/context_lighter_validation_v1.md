# Lighter context deletion validation v1 development card

Protocol: `context-lighter-validation-v1`. Policy: 1.2. Stage: development.
Status: ready for the fixed TRAIN/known-VALIDATION-only diagnostic after source
review. Freeze this card, instantiated plan, permitted input hashes and actual
CUDA admission before fitting. The diagnostic remains unmeasured until completed.
No TEST or old stress access, selection, consumer acceptance or automatic promotion.

RPB-v7 is a bounded training-policy hypothesis: reduce the extra visible-context
deletion request rate from 0.30 to 0.15, keeping the RPB-v4 inference architecture
and the remaining RPB-v6 training recipe unchanged. The choice addresses the
already known TRAIN/VALIDATION reconstruction tradeoff of stronger deletion.
It is not a threshold, architecture or budget chosen against replication TEST.
The [replication card](context_replication_v1.md), its measured artifacts and
all earlier cards keep their original bytes and disposition. RPB-v4 remains active.

## Question and fixed population

Use timing (`lag_sign`) only and the five known masters 4404, 5505, 6606, 7707
and 8808. For each master reuse exactly 128 TRAIN source pairs (256 rows) and
64 VALIDATION pairs (128 rows) from the completed new replication development
cohort. Natural missingness stays 0.10. Do not replace a cohort, introduce
a new master, change a label or mask, or discover another task or testing split.
The existing development generator may replay these exact declared draws only
after the permitted parent roles are bound. Before any candidate fit, require
bitwise equality of observations, masks and labels plus exact source order for
every master against both parent versions. The candidate fitting callback also
checks its replayed legal TRAIN input against that identified cohort. A replay
mismatch stops the run; it does not authorize another draw.
Both opposite-label variants remain grouped within their original split.

The RPB-v7 `NativeCurveRun` development phase retains ordinary points 0 and
512, including their native features, fixed TRAIN heads, original queries and
immutable witnesses. Point 0 is the exact initialization of the same live
trajectory, not another candidate, selection point or additional-deletion
classification input. Old RPB-v4/RPB-v6 point 0 assets establish strict
initialization pairing and remain preserved ordinary diagnostics.
The positive budget is fixed at 512 updates for all five masters. A failed or
unsupported fit does not authorize omitting its master/head, redrawing data
or choosing another budget. No checkpoint, head or deletion rate is selected
from this diagnostic; 0.15 is the sole proposed new threshold.

The question is whether lighter training deletion preserves RPB-v6 native32
linear access on known intact and a new fixed corrupted VALIDATION view while
bringing original fixed-query TRAIN/VALIDATION reconstruction to the paired
RPB-v4 level. This is development evidence on known masters and observations,
not an independent confirmation of generalization.

## Permitted parent roles and untouched scope

Parent capsule:
[context-replication-JjNEUc](../../../output/runs/rpb-context-replication/context-replication-JjNEUc/).
Pin its completed inventory metadata SHA256
`13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9`.
The parent orchestration source identity is
`9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828`;
the parent training producer is
`2fff50c48605ee21dbbfd28d6989bdcf6b5cd057785c2acb19419571af48b895`;
the parent core writer is
`587f2423758c3e70c2c7665d9e7b59bfe10af4c7f9aaba23f5318b3c55b13bca`.
Inventory metadata identifies permitted roles; it does not authorize opening
or hashing all parent payloads.

Before any new model, transform or head fitting, save an explicit subset
manifest of required parent TRAIN/VALIDATION observations, development
manifests, native features, initial/positive model checkpoints, raw/scaler
companions, producer and initialization audits, point records, ordinary
TRAIN-fitted heads and TRAIN/VALIDATION predictions, and original fixed-query
TRAIN/VALIDATION arrays. Include role, absolute path, bytes, SHA256, master,
budget and producer/cohort association. RPB-v4 roles are under `reference/`;
RPB-v6 roles are under `results/`, using `candidate-milestone-0` and
`candidate-milestone-512`. The exact input allowlist is fixed before access.

Only those permitted roles may be opened. Do not load parent `report.json`,
TEST archives, testing manifests/predictions, stress reports/cases or hidden
clean values. Do not hash such payloads as a preservation shortcut. No shared
testing generator or TEST/stress scoring entry point is called. Preserve every
original allowed input byte before and after; all new outputs use exclusive paths.
Frozen helper source may be reused with pinned identity without running its
historical experiment entry point or reading historical payloads.

## Same architecture, initialization and fixed training prefix

Use [learned_global.conf](../../encoders/raw_patch_bottleneck_mae/config/learned_global.conf),
with CUDA, the declared master seed and a 512-update ceiling. Keep C3/H32/F3,
semantic channel IDs 0/1/2, unitless features, sampling interval 1, endpoint 31,
patch length 8, encoder width 64, export width 32, three temporal blocks, four
attention heads, feed-forward width 256, one aligned channel mixer, decoder
width 128 and global bottleneck mode 2. Every version has 225,805 parameters.
No PCA, projection or readout compression follows the encoder. Decode solely
the exact served global32 plus the existing metadata.

Preserve dropout 0, layer-normalization epsilon 1e-5, original artificial mask
ratio 0.25, hierarchical Huber delta 1, TRAIN scaler floor 1e-6, AdamW learning
rate 0.001, weight decay 0.0001, clipping 1, batch 8, one CPU Torch thread and
the original logging/checkpoint settings. Persist the full actual settings,
not only this summary. An empty configured channel-ID list is resolved as
0/1/2 by the canonical resolver; require the actual input, scaler and producer
channel-order witnesses to agree exactly.

For every master require all point 0 parameters and buffers, including global
pooling, to match both identified RPB-v4 and RPB-v6 initialization checkpoints.
Use the exact original legal TRAIN values/support, source order, dataset/schema,
initialization seed, sampling interval and semantic channel metadata. The
fresh RPB-v7 TRAIN scaler fits once and must reproduce both parent scalers
and their identity exactly. All providers use the unchanged fitting namespace
`native-development-v1/lag_sign`; the new outer diagnostic protocol does not
rename the metadata namespace used to identify this cohort.

Keep the original row-sampling, artificial patch-mask and per-attempt Torch
streams, one live AdamW state and the absolute attempt/update/sample counters.
Require attempted equals completed equals 512, with exactly 4,096 sampled rows
at the positive point. A skipped attempt breaks prefix pairing and stops the
declared run. Snapshot creation, extraction, reconstruction and head fitting
restore ambient CPU/all-CUDA RNG states before the next training update.
Save the point 0/512 checkpoints and immutable snapshot witnesses.

The encoder trainer receives only legal TRAIN observations and metadata,
never TRAIN labels, hidden clean values or held-out observations. Readout
fitting may use TRAIN labels. No scaler or model is fitted on VALIDATION.

## Sole training-policy change: request rate 0.15

Use policy ID `rpb-training-context-deletion-015-v1`, checkpoint tag RPB-v7 and
complete typed/text companions declaring request rate 0.15. The existing
RPB-v6 rate-0.30 policy ID `rpb-training-context-deletion-v1` and API remain
unchanged. Do not relabel either policy or silently resume through ordinary
training. Bind these identities in the instantiated plan and admission before
execution; this card does not authorize a second mechanism.

Requests apply only to original visible context V0=O&~A. Keep the existing
training context stream `0x6374782d64726f70ULL` (decimal 7166485043407384432),
absolute attempted update and semantic-canonical B/C/H/F ordinal. A request
occurs where the same independent top-53-bit uniform value is below 0.15.
Do not consume Torch RNG or use values, labels or held-out data in those requests.
Before repair, rate-0.15 requests are a subset of rate-0.30 requests for the
same original visible coordinates and absolute counter draws. Post-repair
realized masks need not obey a simple subset claim.

Retain the existing repair: only originally eligible channels may have extra
deletion cleared, restoring the first original-visible h/f coordinate in the
earliest fully erased original patch until two originally visible patch groups
remain. Do not restore natural missingness, query support or hidden target
values. Original O, A, Q=O&A, eligibility, standardized targets and cell/channel/
example Huber denominators remain exact. Zero nonvisible storage before encode.
Save cumulative requested, actual and restored coordinate counts at each point;
actual equals requested minus restored, with all counts zero at point 0.
Report actual counts rather than an exact realized 15% deletion fraction.
Ordinary frozen inference performs neither training deletion nor repair.

## One fixed additional-30% VALIDATION view

View ID: `validation-dropout-030`. Request rate: 0.30.
Fixed dedicated stream: `0x636c763164726f70ULL` (`clv1drop`), decimal
7164231061750312816. Namespace: `context-lighter-validation-v1`.
The view seed is `stream_seed(master,stream)`; this namespace/stream is distinct
from all earlier VALIDATION and TEST streams. Freeze it before view generation.

For each original source ID use
`stream_seed(view_seed,FNV1a64(namespace+"/lag_sign/coordinate/"+decimal(id.size())+":"+id))`.
Use local `std::mt19937_64`, one top-53-bit draw per physical C/H/F coordinate
in declared channel order 0/1/2, including naturally absent coordinates.
Request erasure where `(draw>>11)*2^-53 < 0.30`. Share the exact requested mask
across both variants, all three positive-point versions and every head repetition.
Create and persist one view per master, with source mapping, original mask,
requested erasure, final mask, observed counts and checksums, before scoring.

Apply O'=O&~E, preserve original values exactly where O' is true and set every
hidden value to zero. This view has no training repair or support restoration.
Labels, source order, geometry and channel IDs remain exact. Its approximately
37% expected total missingness combines 10% natural missingness with 30%
additional independent deletion; disclose actual support instead of assigning
that expectation to every example. Any legal raw-oracle score is descriptive,
never a replacement-draw gate.

## Fixed TRAIN readouts and known-VALIDATION comparisons

Use the existing shared `ArchiveReadoutRun` primitives and explicit legal inputs;
the evaluator remains independent of the encoder. Its new-view comparison
includes native, raw576 and standalone raw-PCA32, without a mask-metadata
classifier. Existing ordinary mask controls preserved by `NativeCurveRun`
remain historical evidence; this card does not refit or score them on the new
deletion view.

Declare exactly 15 archive-readout inputs: five masters times RPB-v4/RPB-v6/
RPB-v7 at positive point 512. Each input's ordinary TRAIN fit scores intact
VALIDATION and the shared additional-30% view. Positive point 512 is the sole
new corrupted-view classification comparison. Do not create another initial
checkpoint deletion-view fit matrix; identical initial assets support pairing,
and candidate point 0 keeps its ordinary heads/features/queries and witnesses.
Fit outer FeatureNormalizer
and each head's own normalizer on valid original TRAIN features only, once per
identified version/checkpoint input. Score both intact VALIDATION and the fixed
deletion view with those exact maps and heads. There is no held-out refit.

RPB-v4/RPB-v6 comparisons use diagnostic TRAIN refits under the same fixed
recipe, with original ordinary fit and intact prediction parity checked where
the retained schema permits exact mapping. These are new diagnostic fits;
do not claim zero fitting constructors or relabel them as loaded old heads.
Keep the original assets untouched. Refit raw576 and standalone raw-PCA32
transparently on the same original TRAIN cohort and freeze them for both views.
Equal inputs/recipes must retain identical transforms and
ordinary predictions across the disclosed repetitions.
The new native refit parity matrix contains 45 head pairs (15 positive inputs
times three repetitions). This count covers native parity records only;
the separately disclosed raw/PCA head fits also remain in the full archive
readout matrix. Record complete input and output manifests and finish only after
all 10 candidate development points, 15 positive inputs, 45 native parity records
and 60 version/view/repetition comparison records are complete. Each comparison
record contains both heads, giving 120 paired head effects.

Ridge penalty 1 is primary. The neural secondary is tanh with 16 hidden units,
Adam learning rate 0.01 and 100 updates. Keep all three base seeds
2701/2802/2903 and actual seed `stream_seed(rep_seed,input_width)` paired across
equal-width methods. Native32 and raw-PCA32 each have 66 Ridge and 562 neural
parameters. Raw576 consists of 288 TRAIN-standardized observed values plus
288 visibility flags; its observed TRAIN scaler floor is 1e-8. PCA32 applies
only to those raw TRAIN features and preserves rank limits; it never follows
RPB-v4, RPB-v6 or RPB-v7. Unsupported raw PCA preserves legal native/raw results.

Retain all five masters, every declared positive version/head/view and
unsupported status/reason, plus every ordinary point 0 diagnostic and witness.
Require candidate/reference support equality for the primary
comparison, and report native/full-population coverage, common-valid class/
source support and correctness with abstention counted as failure. Comparisons
are named pairwise, so an unrelated control cannot alter their denominator.
Any paired intervals use within-master source groups and remain conditional
on the identified fits; do not construct an across-master or head-fit CI.

## Original fixed-query reconstruction, costs and decision

Score ordinary unaugmented frozen inference on original TRAIN and intact
VALIDATION for RPB-v4/RPB-v6/RPB-v7 point 512, retaining point 0 diagnostics.
Enumerate each of four original patches, hide the same patch across channels,
target observed cells only and require two retained visible patch groups in
eligible channels. Preserve query, target, visible and eligibility arrays.
The exact TRAIN scaler defines standardized MAE and Huber 1, with equal
cell -> channel -> example reductions. The classification deletion view does
not redefine these original reconstruction queries or denominators.

Record new RPB-v7 actual CUDA counters, finite gradients, changed weights and
synchronized training-loop time for every master. Retained RPB-v4/RPB-v6 times
remain identified historical timings from their parent run. A loop timer is
not isolated GPU kernel time and does not measure equal inference/memory cost.

A successful development diagnostic requires all of the following at the
predeclared point 512, with all three heads averaged within each master and
all five masters weighted equally:

1. Intact and deleted known-VALIDATION mean native Ridge accuracy no lower than
   paired RPB-v6, with equal coverage. Preserve every paired-master effect and
   spread; do not hide lower individual scores or a lower worst-master score.
2. Mean original fixed-query TRAIN MAE and mean VALIDATION MAE each no worse
   than the contemporaneous paired RPB-v4 point 512 reference. No guard waiver.
3. Full source/input preservation, initialization/scaler/stream/query/decoder,
   actual CUDA, checkpoint-policy, fixed fitting and independent audit gates pass.

Neural-only gains, best masters or unsupported exclusions do not satisfy the
primaries. Crossing-zero within-master intervals constrain interpretation.
Failure stays a failed or unresolved development hypothesis; no second ratio,
budget search, acceptance, automatic promotion or TEST draw follows this card.
Even passing this known-VALIDATION diagnostic requires a separate frozen card
and explicit decision before fresh TEST or independently trained confirmation.

Follow the [reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md):
quality and training tables, tag-only embedding cells with descriptions in
adjacent prose, every master/head and named view retained, and a clear division
between this new diagnostic, historical evidence reused and unmeasured acceptance.
