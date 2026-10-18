# Optimization validation v1 development card

Protocol: `optimization-validation-v1`. Policy: 1.2. Stage: development.
Status: frozen prospective TRAIN/VALIDATION-only optimization diagnostic.
No TEST, stress, budget selection, acceptance or promotion.

The [diagnostic record](../../encoders/raw_patch_bottleneck_mae/OPTIMIZATION_DIAGNOSTIC.md)
tracks this follow-up. The completed
[paired fixed512 result](../../encoders/raw_patch_bottleneck_mae/PAIRED_POOLING_ADVANCE.md)
and its [original card](paired_pooling_v1.md) remain unchanged: the claimed v5
advance was rejected at that budget, and v4 remains active. This diagnostic
cannot rescue that result or establish unseen confirmation.

## Question and fixed scope

Does the existing v5 trajectory improve native linear access and fixed-query
reconstruction with additional optimization, or does poor linear access persist?
Use timing (`lag_sign`) only, every original master3101/3202/3303, the exact saved
128 TRAIN source pairs (256 examples) and64 VALIDATION pairs (128 examples).
Do not generate data, introduce a replacement seed or open another task.
These already known development cohorts are not an unseen evaluation set.

Restore each saved ordinary v5 point512 checkpoint, its AdamW state, scaler,
settings, attempted/completed counters and exact TRAIN association. Continue
absolute512 ->1024 ->2048 on the same row/mask/per-attempt Torch RNG streams.
The existing ordinary `run_cli train --resume` loop performs updates; do not
write a second optimizer loop. Its `--steps` argument is **additional updates**:
512 more reaches1024, then1024 more reaches2048. Save every new point in a new
directory and use the exact saved1024 checkpoint as the2048 parent.

Keep mode3/mixer1/native32, C3/H32/F3, patch8, encoder width64/three blocks/four
heads/feed-forward256, decoder128, dropout0/LN epsilon1e-5, mask ratio0.25,
Huber1, TRAIN scaler floor1e-6, AdamW0.001/weight decay0.0001, clipping1 and
batch8 on actual CUDA. All269,389 model parameters stay on CUDA for updates.
No fresh model initialization, scaler refit, seed, objective, head recipe or
architecture change is permitted. Record additional-budget and checkpoint-
frequency overrides explicitly; checkpoint frequency0 prevents old paths from
being overwritten. An invalid update, mismatched parent, skip/counter mismatch
or nonfinite state preserves evidence and stops this declared diagnostic.

## Parent binding and point512 witness

Parent capsule:
[paired-pooling-BFFPX5](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/).
Pin its1,326-file inventory SHA256
`610976e666468ac2f80404bd56884e36f9275645305e83a9d0e50b762e702be3`.
It records source fingerprint
`3655668a178e389f59d76d2af187d49ec7acc06d7928057bb70dbd367f559166`.
Bind only explicitly named TRAIN/VALIDATION/checkpoint/fitting roles from that
inventory before deserialization. Reading inventory metadata is permitted;
discovering, hashing or reading its TEST/stress payloads is outside this run.

| Master | Point512 parent checkpoint SHA256 |
| --- | --- |
| 3101 | `fc9da5eceffc252cb69f5e9b4a92e6464711cfcc8c1e7c225bb42936f24a1207` |
| 3202 | `7a7d3949f9757ba7594c00cb6231e700d7ce19e822194b660126eeff582c3d1e` |
| 3303 | `02c45cc957fa6beae6730df66bee0bdb5c22297ce18d0f620c4890fcb3044de0` |

Also bind original ordinary TRAIN companions, producer audits, legal controlled
TRAIN/VALIDATION archives, native feature exports, fixed-query arrays and saved
point512 fitted readouts/predictions. Preserve original bytes and role hashes
before and after. The instantiated card/ledger records exact paths, byte sizes,
SHA256s, data/schema/scaler/source IDs, resolved settings and parent lineage.

Before any continuation, restore a frozen point512 snapshot and exactly
reproduce its native TRAIN/VALIDATION features/support and original-patch
reconstruction arrays. Witness the unchanged old fitted heads/predictions;
**do not fit heads or raw/PCA controls again at512**. Retain the frozen snapshot
for an after-continuation immutability witness. Labels and held-out observations
are excluded from the encoder resume interface, which receives legal TRAIN and
metadata only.

Verify parent optimizer association and active-parameter step512, finite CUDA
moments, CUDA weights and frozen TRAIN scaler. At1024/2048 active AdamW steps
advance exactly by512/1024, with absolute attempted/completed counters matching
the declared unskipped trajectory. Scaler tensors, schema, dataset, semantic
channel IDs and row ordering stay exact. Weights must change and remain finite.
Preserve CPU-readable state evidence or an independently inspectable ordinary
archive representation for optimizer/scaler tensor QA; numerical loop semantics
remain those of the existing workflow.

## Native readouts and raw controls

At new points1024/2048, extract the exact served global32 on legal TRAIN and
VALIDATION through immutable checkpoint snapshots. No PCA/random projection
follows the encoder. Shared measurement helpers own fixed-query arithmetic;
the encoder adapter supplies frozen features and standardized reconstruction.
Callbacks receive independent legal clones; measurement restores ambient
CPU/CUDA RNG so it cannot affect the resumed update stream.

Use the existing encoder-independent `ArchiveReadoutRun` for the six new
master/point inputs. Every input declares explicit TRAIN/VALIDATION observations,
native features, scoring labels, source IDs and hashes; it has no TEST role.
Outer normalization and each probe's own normalization fit valid TRAIN only.
Ridge penalty1 is primary. The secondary is tanh16/Adam0.01/100 updates. Keep
all repetitions2701/2802/2903, actual seed `stream_seed(rep_seed,input_width)`.
Native32 and standalone raw-PCA32 use66 Ridge/562 neural parameters; raw576
uses1,154/9,266. No best head or master is selected.

This reused archive driver refits raw ObservationScaler/outer normalization/
PCA and raw/PCA readouts on the **same TRAIN** separately for each new input.
Disclose this constructor behavior rather than claiming retained control fits.
It does not retrain an encoder or fit any preprocessing on VALIDATION. Raw
means288 legally observed standardized values followed by288 visibility flags;
missing values are zero, ObservationScaler floor1e-8. PCA32 fits raw TRAIN only.
Record actual fit counts, ranks, dimensions and control identities. Unsupported
rank/native fits stay named with coverage; never silently drop a master or row.

Score all three methods on common declared VALIDATION rows. Report native/full
coverage, pair-specific intersections, class/source counts, per-master scores
and spread. Retain95%/1,000-draw source-group paired conditional intervals where
supported. Do not pool them into an across-master CI. Any512 numbers shown are
explicitly reused existing VALIDATION results, not freshly fitted measurements.

## Reconstruction, time and interpretation

At512/1024/2048 use the unchanged four-original-patch query: hide each patch
across channels, target legally observed cells only, retain at least two visible
patch groups per eligible channel, preserve natural missingness. Decode solely
exact served32. Standardized MAE/Huber1 reduce cells -> channels -> examples
equally using the unchanged TRAIN scaler. Save target/visible masks, eligibility,
predictions and per-example reductions. All points use identical queries and
source populations; a changed denominator is reported, not treated as progress.

The adapter's synchronized whole-command wall time includes checkpoint loading,
normalization, updates, logging and saving. It is **not GPU-update-only time**.
Show GPU training seconds as unmeasured (`—`) at1024/2048 and report command
wall time separately with its scope. Original512 GPU-update mean14.37 seconds
may be shown only as a reused measurement. Do not add these different timing
scopes or infer a speed advantage from them.

Report every point/master/head, continuous counters/optimizer evidence,
reconstruction, native access and cumulative additional cost. Improved known
VALIDATION at a later point is optimization diagnosis, not a new selected
checkpoint, repaired fixed512 result, promotion or consumer acceptance. A later
comparison would require a separate declared reference/budget and fresh TEST
namespace before any unseen draw; this card authorizes none.

## Artifacts and independent audit

Claim a new output capsule; refuse existing destinations. Persist this human
card, instantiated machine recipe/input ledger and exact compiled source snapshot
before loading/fitting/training. Save512 witnesses, ordinary1024/2048 checkpoints
with optimizer state, producer/resume audits, settings overrides, legal feature
arrays, shared raw/PCA/native fits/predictions, fixed queries/loss reductions,
wall-time scope, input preservation hashes and complete VALIDATION report.
Keep earlier capsules and the fixed512 card untouched.

Independent stdlib archive inspection must verify parent binding, exact512
witnesses, continuous optimizer/counter/scaler evidence, TRAIN-only fitted
statistics/native-no-PCA, paired seeds/source populations, prediction replay,
reconstruction reductions and original bytes. No Torch/model execution or
TEST/stress discovery is needed for that audit; actual CUDA continuity is covered
by coordinated container gates. Follow the
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md): tag-only model
cells and short descriptions in adjacent prose.
