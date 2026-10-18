# Early mixer paired learning curve v1

Protocol: `early-mixer-learning-curve-v1`. Prospective development card,
9 October 2026 (Asia/Dubai). Freeze this card, the complete source closure,
actual CUDA engineering admission and independently reviewed reader before
generating these cohorts. Preserve the completed fixed-512 experiment's card,
source, capsule and readers, and every historical trained group. This is a
separate fresh experiment; no historical quality payload is an analysis input.

## Question and trained groups

Does the difference between late and early aligned mixing persist, improve or
reverse as the same continuous optimizer reaches 512, 1,024 and 2,048 updates?
This is a fresh paired learning curve, not a continuation or rescue of the
completed fixed-512 groups.

The fresh late group is **RPB-v7.alt-03** and the fresh early group is
**RPB-v10.alt-01**. Design tags remain RPB-v7 and RPB-v10; suffixes identify new
trained groups. Label retained points plainly, for example
`RPB-v7.alt-03 @ 1024`. Point 0 is its own untrained control. Equal initial
weights do not imply equal initial outputs because the information paths differ.

Keep native width 32, all 225,805 named parameters, module registration and
initialization order, mode 2, mixer 1 and dropout 0. Placement 0 uses the literal
late path. Placement 1 mixes aligned channel states before the shared temporal
blocks and retains the independent unmixed temporal pass for `z_local`, with
its existing autograd contract. Independent local states are a diagnostic
information path, not a second reconstruction objective. The decoder consumes
the served contextual global 32-vector. Early mixing adds compute without
adding parameters; preserve typed architecture and output/reconstruction
semantics and every historical placement-0 admission contract.

## Fresh paired cohorts and unchanged learning recipe

Timing masters: **30130, 31231, 32332, 33433, 34534**. Amplitude masters:
**35635, 36736, 37837, 38938, 40039**, paired with timing masters in that order.

Each task has 128 TRAIN source pairs and 64 VALIDATION source pairs: 256 and 128
rows. Use C = 3, H = 32, F = 3, patch size 8, float64 CPU generation, natural
missingness 0.10, semantic IDs 0/1/2, interval 1, endpoint 31 and zero hidden
storage. Prefix source IDs with the new protocol and task; TRAIN and VALIDATION
sources/streams are disjoint. Generate no TEST or stress data.

Both architectures receive the same legal label-free timing TRAIN observations,
source order, original TRAIN-fitted scaler, initialization seed and deterministic
row/query/context streams. The original encoder/scaler fit namespace is exactly
`early-mixer-learning-curve-v1/lag_sign`. Amplitude data fit only amplitude
controls and heads. Each amplitude export retains its paired timing TRAIN
scaler and original encoder fit metadata; amplitude never fits an encoder.

Keep the existing coordinate-0.15 policy
`rpb-training-context-deletion-015-v1`, its counter stream, original Q, target
eligibility, hierarchical Huber denominator and deterministic repair unchanged.
Keep CUDA batch 8, AdamW learning rate 0.001, weight decay 0.0001, gradient clip
1, Huber delta 1, one configured thread and `log_every=1` for the complete
per-attempt trace. No decoder calibration, auxiliary
loss, deletion-rate change, head tuning or learning-rate schedule is added.

Create one live controller per architecture/master with a 2,048-update ceiling
and attempted-step cap 4,096. Save at 0, train to 512 and save, continue to 1,024
and save, then continue to 2,048 and save. Do not reinitialize, resume through a
new optimizer or refit the scaler. Require attempted = completed at every point
and no skipped attempt. A skip or exhausted cap rejects the run rather than
replacing rows or shifting its schedule. The ten trajectories contain 20,480
completed updates and **163,840 sampled row exposures**; these are not
independent source counts.

The larger ceiling/cap must not alter the first 512 constant-rate updates,
sampling/query/context streams or ordinary AdamW progression. Prove horizon
equivalence with artificial CUDA fixtures before generation. Disclose the
settings-text ceiling change separately from the unchanged update recipe.

## Fixed validation views and shared readouts

Score intact VALIDATION and one additional 0.30 coordinate-deletion view per
task/master. Each source pair shares one physical-grid erasure mask, including
draws at naturally absent coordinates; there is no evaluation support repair.
Create each view once and reuse its exact observations/mask across every
architecture, budget, control and head repetition.

Timing uses `stream_seed(timing_master, 0x656d636c74696d30ULL)` and namespace
`early-mixer-learning-curve-v1/lag_sign/validation-coordinate-dropout`.
Amplitude uses `stream_seed(amplitude_master, 0x656d636c616d7030ULL)` and namespace
`early-mixer-learning-curve-v1/amplitude/validation-coordinate-dropout`.
Keep the existing unrepaired source-paired coordinate law: hash namespace,
task and length-qualified source ID with FNV-1a, combine with the derived view
seed through `stream_seed`, then draw from a local MT19937_64 once for every
physical C/H/F coordinate. Compare each top-53-bit uniform draw with 0.30.

Use Ridge penalty 1 and tanh-16 Adam learning rate 0.01 for 100 updates, with all
three head repetitions 2701/2802/2903 and the unchanged actual seed law
`stream_seed(rep, width)`. For raw, mask and native methods, keep the outer TRAIN
normalizer followed by each probe's own TRAIN normalizer. Fit raw outer
normalization and PCA 32 once on each task's TRAIN. Prepared PCA skips the
readout outer stage and keeps the probe normalizer. Native features receive no
PCA. Fit each method/repetition once and reuse TRAIN, intact VALIDATION and
deleted VALIDATION predictions. Retain unsupported fits and their reasons.

Timing has 11 unique fitted methods per master: raw 576, PCA-only 32, mask 288,
both initial native-32 controls, and six positive native-32 points (late and
early at each of 512, 1,024 and 2,048). Controls and initial heads fit once per
cohort. Use a generic reusable vector of three explicit comparison pairs:
late versus early at 512, at 1,024 and at 2,048, with effects defined as early
minus late. Reuse saved fits and predictions; do not invoke a two-model
evaluator three times and refit its baselines. Any declared descriptive initial
or adjacent-budget contrast also uses these saved fits and populations.

Amplitude has seven unique methods per master: its three task-specific controls,
both initial native controls and both final native points at 2,048. No amplitude
exports or head fits at 512 or 1,024 are included.

The fully supported matrix is **270 pipelines / 540 individual heads**:
165 timing pipelines and 105 amplitude pipelines. Every pipeline contains one
Ridge and one tiny neural head. Report actual supported counts separately; do
not average unsupported methods away. The two tasks do not share TRAIN maps
or heads.

Within-master paired source bootstrap intervals keep the existing fixed-head
law, seeds, 1,000 replicates and 95% bounds. Retain every master, every head
repetition, coverage, equal-master means and worst-master scores. Head
repetitions and nested budgets are not independent encoder replications. Do
not average tasks or confidence-interval endpoints. No pooled across-master
interval is introduced by this card.

## Continuous-state preservation and one export per quality surface

At every point, immediately save the checkpoint and all companions, exact byte
hashes, producer scopes, scaler and named parameter/buffer/ordinary AdamW state
associations, attempted/completed/sample counters, cumulative context counts
and the complete trace-prefix witness. Require paired attempted/completed/
target-cell/sample counts and source-bound deterministic row/query/context
streams. Loss and gradient values may differ between architectures. Do not
claim a full saved target-mask witness at every update unless provided by the
final schema.

After both controllers reach 2,048, recheck every earlier checkpoint and
companion's exact bytes and saved state/counter/trace-prefix witnesses. Later
training must not mutate an earlier point. Then instantiate immutable CUDA
snapshots for all retained points and export each required quality surface
and query bank exactly once. No earlier quality export is repeated to prove
preservation, and no CPU model replay occurs. Artificial CUDA admission proves
exact repeated serving and unchanged earlier snapshots after live advancement.

Save each full-observation native export once and reuse it for all heads. The
planned count is **180 exports**: 120 timing exports (two architectures × four
points × three views × five masters) and 60 amplitude exports (two architectures
× two points × three views × five masters). Views are TRAIN, intact VALIDATION
and deleted VALIDATION. Point 0 exports are initialization diagnostics.

At positive timing points only, run original-patch reconstruction on TRAIN and
intact VALIDATION with that point's original decoder. Save exact paired Q,
visibility, eligibility, target and support arrays and unchanged hierarchical
MAE reductions. These are **60 query-evaluation calls / 240 necessary Q-masked
CUDA forwards**, using four original common patch-query banks per call. Point 0
query evaluation and amplitude reconstruction are outside this scope/count.
Query contexts require distinct encodes; they are not repeated full-observation
feature extraction. Keep sampled training Huber traces separate from fixed-query
reconstruction MAE.

Admission must prove direct versus segmented continuous progression on
artificial CUDA data: named parameters, AdamW moments/steps, scaler,
sampling/query/context counters and trace prefixes agree. Interleaved
save/snapshot/export and CPU readout work must leave subsequent updates
unchanged. Test immutable earlier points after later updates, decreasing-budget
and skipped-update rejection, malformed source/namespace/placement/policy
rejection, and rejection of the new namespace/budgets by historical fixed-512
scopes. Preserve old APIs and their admission contracts; retain failed gates
additively.

Independent CPU saved arithmetic reruns no encoder, optimizer, head fitting or
PCA/SVD. CUDA checkpoint bodies remain byte-bound; CPU companions and actual
CUDA admission/source establish their state semantics. Keep float64 replay
tolerance absolute 2e-9 plus relative 2e-9, and float32 operation replay tolerance
absolute 2e-6 plus relative 2e-5. Saved-logit argmax, classes/support/source order
and promised equality witnesses remain exact. Freeze the final writer schema
and reviewed reader before generation. Any later execution-flag release must
be only the pre-reviewed flag changes, with exact byte proof.

## Reporting and interpretation

Use separate standard timing intact/deleted tables at each positive budget.
Each has raw, PCA-only, mask, both initial controls and the two native points at
that budget; adjacent prose marks reused controls. Point 0 has five unique
initial/control rows or links to those saved diagnostics. Amplitude gets
separate intact/deleted final-point tables. Model cells contain short tags only;
descriptions sit beside tables.

Training tables retain exactly these five headers:
`Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds`.
Show both groups at 512, 1,024 and 2,048. Disclose cumulative versus interval
costs beside the table. Keep pure GPU/kernel time unmeasured when not
instrumented; separate encoder-loop cost, full-feature inference, necessary
query inference, binding/I/O, CPU control/head/bootstrap work and audit cost.
The independent early local pass adds compute at every point. Equal parameters
and update counts do not establish equal compute budgets.

Describe the whole paired curve: timing Ridge means, worst-master scores,
coverage, original-query TRAIN/VALIDATION MAE and per-master tradeoffs. Neural
heads and final-point amplitude transfer are secondary. Strong initial amplitude
controls limit claims that trained amplitude accessibility was learned. Nested
points describe optimization on known validation cohorts, not independent
generalization confirmations.

No adaptive best-point selection, rescue, automatic promotion, TEST, stress,
decoder tuning, seed dropping, deletion-rate change or head change occurs.
Retain favorable and unfavorable points together. Any later selected-budget
or fresh-TEST comparison requires a separate prospective card and decision.
This card does not alter or reinterpret the frozen fixed-512 result.
