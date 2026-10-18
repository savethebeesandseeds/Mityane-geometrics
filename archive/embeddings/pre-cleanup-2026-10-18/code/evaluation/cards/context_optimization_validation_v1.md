# Context optimization validation v1 development card

Protocol: `context-optimization-validation-v1`. Policy: 1.2. Stage: development.
Status: frozen prospective TRAIN/VALIDATION-only optimization diagnostic.
No TEST access, old stress reads, budget selection, acceptance or promotion.

The [completed context-deletion result](../../encoders/raw_patch_bottleneck_mae/CONTEXT_DELETION_ADVANCE.md)
and its [original card](context_deletion_v1.md) remain unchanged. RPB-v6 improved
both fixed 512-update quality primaries but failed both reconstruction guards.
That measured disposition remains an unresolved tradeoff; RPB-v4 stays active.
This diagnostic cannot retroactively pass that rule or establish unseen quality.

## Question and fixed scope

Can unchanged context-deletion training improve fixed-query reconstruction
toward the historical RPB-v4 point 512 errors while preserving RPB-v6 point 512
native linear access on known intact and corrupted VALIDATION?

Use timing (`lag_sign`) only and all three masters 3101/3202/3303. Reuse each
master's exact 128 TRAIN source pairs (256 examples) and 64 VALIDATION pairs
(128 examples). Two opposite-label variants remain within one source split.
Natural missingness stays 0.1. Do not generate a replacement cohort or label,
introduce another task/master, or discover any old TEST/stress payload.
The one permitted additional view is a newly constructed fixed 30% coordinate
deletion of the existing VALIDATION observations, defined below.

Measure absolute budgets 512, 1024 and 2048 on one unchanged trajectory per
master. Preserve every point and head repetition. No budget is selected or
promoted from these scores; VALIDATION is already known development evidence.
Better late scores do not repair the original fixed 512-update result.

## Exact replay gate, then live continuation

The tagged RPB-v6 checkpoint deliberately cannot resume through ordinary
`run_cli train --resume`. Do not remove that guard, load the weights into an
ordinary-policy trainer or claim augmented-resume support. Instead construct
a fresh trainer using the exact original context policy and replay 0 -> 512.
This is an engineering replay of the already measured trajectory, not a new
seed, candidate, initialization choice or permission to change its recipe.

Before continuing past 512, require exact agreement with the pinned C parent:
all model parameters and buffers; named optimizer step/moment state; scaler
tensors and identity; attempted/completed/sample counters; dataset/schema,
ordered sources and semantic channel IDs; and cumulative context request,
actual deletion and restoration counts. Verify the original native TRAIN and
VALIDATION exports/support and fixed-query reconstruction arrays. Preserve
original fitted readouts and their cached VALIDATION prediction witnesses.
The diagnostic refits its point 512 heads on original TRAIN under the same
fixed recipe after this gate; those are new diagnostic fits, not cached assets.
Require numerical fit and intact prediction parity with original C 512 where
the archive schemas permit mapping. Preserve original fitting bytes unchanged.

A replay mismatch rejects this declared diagnostic before later updates or
readout fitting. Preserve all evidence, stop, and diagnose separately. No
tolerance, replacement master, seed retry or recipe change is authorized to
force the gate to pass. Checkpoint container bytes need not match serialization
metadata; the required equality is the identified state tensors and semantics.

After the gate, keep that same live trainer, AdamW state, frozen scaler and
absolute row/mask/Torch/context counters. Continue 512 -> 1024 with 512 more
updates, then 1024 -> 2048 with 1024 more. Attempted equals completed at every
point; a skipped attempt, nonfinite state or changed association stops the run.
Do not restart a model, optimizer, scaler or RNG policy between points.
Record the only permitted settings change, the absolute step ceiling, and save
only new paths. Original logging/checkpoint settings remain unchanged.

Keep the full original recipe: RPB-v6 mode 2, one aligned mixer, native 32-number
global export, C3/H32/F3, semantic channel IDs 0/1/2, patch 8, encoder width 64,
three temporal blocks, four heads, feed-forward 256, decoder 128, dropout 0,
layer-normalization epsilon 1e-5, mask ratio 0.25, hierarchical Huber delta 1,
TRAIN scaler floor 1e-6, AdamW 0.001/weight decay 0.0001, clipping 1 and batch 8
on actual CUDA. All 225,805 model parameters stay on CUDA for updates. Decode
solely the exact served global 32 numbers and existing metadata.

The unchanged training policy is `rpb-training-context-deletion-v1`: request
rate 0.30 on original visible context V0=O&~A, stream
`0x6374782d64726f70ULL` (decimal 7166485043407384432), absolute attempt and
semantic-canonical B/C/H/F ordinal. Original Q=O&A, eligibility, targets and
hierarchical Huber denominators stay exact. Repair only extra deletion to keep
two originally visible patch groups in originally eligible channels, using
earliest erased original patch/first original-visible h/f coordinate. Preserve
all text/typed policy companions and cumulative request/actual/repair counts.
Ordinary frozen inference uses no training deletion or repair.

The provider receives only legal TRAIN observations and metadata, never labels,
clean hidden values or held-out observations. Fresh replay may fit the TRAIN
scaler once, but it must exactly reproduce the parent scaler. Original C fitting
assets load for parity witnesses. Snapshot construction, measurement and probe
fitting restore ambient CPU/all-CUDA RNG state before the next training update.

## Parent roles and byte preservation

Parent capsule:
[context-deletion-KTbte3](../../../output/runs/rpb-context-deletion/context-deletion-KTbte3/).
Pin its 1,411-file inventory SHA256
`62415443de5db9ebe7509d31d79b3bbab8d95efa3bf72b7afde3466b1ae021dd` and
61-file source fingerprint
`d1f1cf3a754b5657619982b3d31eec02e96b371b299fcdd56fff7c19fea69178`.
The original human card remains SHA256
`74bb4558b0431bf8209a76cfc68a94112fa00a4f8cbda504746ca0db98a0ec80`.

| Master | RPB-v6 point 512 checkpoint SHA256 |
| --- | --- |
| 3101 | `f995739e4ec90f411ec153f2382c9e5c3afc8104675bc1d1f018960cde644c18` |
| 3202 | `8eec0ccdc28f36c79e0692acbd03ce862f3d02307a86925fc3e762d8bddebc38` |
| 3303 | `2c81b7a728cf7ea11bbd1a6b4d3b98feba170d8bafb35ceb609874ea364926bd` |

Bind explicit legal TRAIN/VALIDATION archives and parent point 512 checkpoints,
producer audit/scaler/optimizer evidence, native feature exports, point records,
fixed-query arrays and point 512 fitting/prediction assets before deserialization.
Cached RPB-v4 point 512 intact scores and reconstruction errors keep their
historical lineage and hashes; do not load or score v4 on the new deletion view.
Inventory metadata may resolve permitted roles; hashing or reading
TEST/stress payloads is forbidden. The instantiated input ledger records paths,
roles, byte sizes, SHA256s, resolved settings and parent/source/data identities.
Verify every original allowed input before and after; never overwrite a role.

Keep an immutable parent 512 snapshot and recheck its legal exports and fixed
queries after live continuation. New compiled source identity and gate evidence
remain distinct from the frozen C producer. Actual CUDA weights, inputs, loss,
finite gradients, optimizer moments/steps and changed weights require coordinated
container tests and measured gate evidence. Save CPU-readable named optimizer,
scaler and policy/counter witnesses for independent tensor comparison.

## One fixed corrupted VALIDATION view

View ID: `validation-dropout-030`. Request rate: 0.30. Stream:
`0x63766c3164726f70ULL` (decimal 7167034816401141616).
View namespace: `context-optimization-validation-v1`.
The view seed is `stream_seed(master,stream)`. For each source ID, use
`stream_seed(view_seed,FNV1a64(namespace+"/"+task_name+"/coordinate/"+decimal(id.size())+":"+id))`.
Task name is `lag_sign`; source IDs are the original ordered strings.

Use a local `std::mt19937_64` per source. Traverse physical C/H/F coordinates
in their fixed declared channel order 0/1/2; each coordinate receives one draw
regardless of its original observation support. Request deletion where
`(draw>>11)*2^-53 < 0.30`. Both variants of a source share the exact requested
mask. Apply O'=O&~E, retain values exactly where O' is true and zero hidden
storage. This view restores no support and applies no training repair.
Labels, source order, channel IDs and geometry remain unchanged.

Create and persist this single mask/view once per master, before budget scoring.
Reuse its observations, labels and IDs byte-for-byte across C replay 512, new
1024/2048 snapshots and all head repetitions. Do not seed by budget,
method, score or label. Record requested erasure, original/final observation
masks, raw observations, source mappings, seed/namespace, checksums and observed
counts. Additional 30% deletion on natural 10% missingness has approximately
37% expected total missingness; report actual support instead of assuming it.

Construct corrupted native feature exports with each point's immutable model
and original frozen TRAIN scaler. Both intact and corrupted VALIDATION must
use that point's same already TRAIN-fitted maps/normalizers/heads. Feature
extraction may use the corrupted legal observations, but no fitting constructor
receives them. Preserve original parent 512 native features/heads and verify
diagnostic fit/prediction parity; extracting its new corrupted view is inference,
not fitting on a corrupted view or an old stress read.

This is one explicitly declared VALIDATION corruption diagnostic, not the old
12-case TEST stress sweep. No new TEST generation, TEST archive discovery,
TEST scoring or old stress replay is authorized. The legal raw oracle on
the new view is descriptive; do not redraw a difficult view or gate it using
TEST solvability requirements.

## Fixed native heads and raw controls

At diagnostic points 512/1024/2048, fit each native pipeline once on valid
original TRAIN features, then score intact VALIDATION and the extra view. No PCA/random
projection follows the encoder. The pipeline is outer TRAIN FeatureNormalizer
followed by each head's own TRAIN normalizer. Ridge penalty 1 is primary;
tanh with 16 hidden units, Adam 0.01 and 100 updates is secondary. Keep all
repetitions 2701/2802/2903, actual seed `stream_seed(rep_seed,input_width)`.
Native 32-number and standalone raw-PCA 32-component heads have 66/562
parameters; raw 576-number heads have 1,154/9,266. No best repetition is selected.

Use the encoder-independent archive readout engine's optional validation views.
For each of the nine master/budget inputs, its raw ObservationScaler, outer normalization/PCA and
raw/PCA heads may refit the same original TRAIN once, as the existing driver
does. Disclose this behavior rather than calling those new controls retained.
Intact plus corrupted VALIDATION share that input's one fitted pipeline, with
zero additional map/head fits for the view. Raw input is 288 observed
TRAIN-scaled values plus 288 flags; missing values are zero, raw scaler floor
1e-8. PCA 32 fits raw TRAIN only. Save actual fit counters, maps, ranks and seeds.

At 512, preserve original C fitting tensors and replay their cached intact
VALIDATION predictions. Diagnostic 512 heads are fresh TRAIN fits; require
corresponding numeric-fit and prediction parity and disclose their constructor
counts. These and later point heads fit TRAIN labels; encoder replay/continuation
remains label-free. No v4 readout is evaluated on the new view. Corrupted
observations, source identities, labels and
feature support must match the declared view. Reject support additions,
retained-value changes, row-order drift or nonzero hidden storage.

Retain all unsupported fits with reasons and coverage. Report native coverage,
full-population correctness with abstentions incorrect, class/source counts,
pair-specific common populations and each master's intact/dropout score.
Require paired views to use identical declared rows; differing support is
disclosed, not laundered through invalid zero features. Report both mean and
worst-master score, and every paired-master change versus C replay 512.
Preserve source-group paired 95%/1,000-draw intervals where supported, grouping
both variants. These are conditional within-master intervals, not a CI across
three independently retrained encoders. Label cached original 512 results reused,
and diagnostic 512 refits/replayed measurements new, even when numerically equal.

## Original fixed-query reconstruction and cost

At 512/1024/2048 use the unchanged unaugmented four-original-patch query on
TRAIN and intact VALIDATION. Hide each original patch across channels, target
legally observed cells only, preserve natural missingness and require two
remaining visible patch groups in eligible channels. Decode solely exact
native 32. Frozen TRAIN-scaler standardized MAE/Huber 1 reduce cells -> channels
-> examples equally. Save predictions, targets, visible/query masks,
eligibility, target counts and per-example losses. All budgets use the same
queries/denominators; the corrupted classification view does not redefine them.

The frozen v4 512 mean TRAIN/VALIDATION MAE is 0.06160147/0.06348619; C 512 is
0.08552338/0.08576768. C's cached intact VALIDATION linear mean is 98.9583%
with full coverage. Record these references and all new per-master values.
Treat v4 errors and cached intact VALIDATION scores only as explicit historical
references, with no v4 score for the newly declared deletion view. Diagnose
whether reconstruction reaches that error reference while preserving C replay
512 intact/dropout linear access, coverage and worst-master score. No
validation criterion selects a budget, retires a master or promotes a model.

Record replay and additional-update costs separately, with synchronized timing
scope and exact counters. The parent C 512 mean GPU training time 15.36449
seconds and v4 historical 15.33126 seconds are reused measurements. Never add
replay cost to the old measurement or call it original training cost. If a
new field covers whole-command loading/logging/saving rather than encoder
updates only, show GPU training seconds as unmeasured and state command wall
time separately. Inference speed/memory is not inferred from parameter equality.

## Artifacts, audit and next decision

Claim an exclusively new capsule. Persist this card, instantiated recipe,
allowed-input ledger and exact compiled production/test source snapshots before
loading/training/fitting. Preserve replay admission, exact parent-state and
native/query/readout witnesses, live 1024/2048 checkpoints/optimizer/scaler/
policy audits, legal feature arrays, the single fixed corruption manifest,
TRAIN-fitted assets and both-view predictions, fixed-query arrays/reductions,
loss traces, cost scope and input integrity. Refuse existing output destinations.

The independent stdlib reader must verify parent SHA/roles, state/scaler/counter
continuity, exact replay, native-no-PCA, TRAIN-only normalization/head fits,
view masks/retained values/source pairing, one-fit reuse, frozen prediction
replay, coverage/paired intervals, reconstruction reductions and original bytes.
It must not instantiate a model or discover TEST/stress payloads. Actual CUDA
and unchanged-policy update semantics remain covered by hash-bound coordinated
container gates, not a claim of independent GPU execution by the reader.

Report every point/master/head and both VALIDATION conditions. Better known
VALIDATION or reconstruction later is optimization diagnosis, not a selected
checkpoint, repaired fixed 512-update card, confirmation or acceptance. Any
later quality comparison requires a new declared budget/reference, independently
trained seeds and unopened TEST namespace; this card authorizes none. Follow the
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md): model tags
alone in encoder table cells, with short descriptions immediately beside them.
