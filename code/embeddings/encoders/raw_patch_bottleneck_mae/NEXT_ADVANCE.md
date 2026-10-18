# RPB evaluation and development plan

Live direction reviewed: 2026-10-12. Research chronology below: through 2026-10-10.

The configurable RPB-v20 API and separate TEMPO-5 capacity experiment are
implemented. Read [current status](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CURRENT_STATUS.md), the
[capacity report](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_CAPACITY_V1.md) and its
[12 October continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_CAPACITY.md).
The compact RPB-v20.alt-01 configuration is retained for continuation.
Next proposed research identifies learned versus fixed-coordinate usefulness
from saved features, then predeclares matched H128/H256 quality work and, if
needed, a feature-specific challenge. Those follow-ups are not measured here.
Market preparation remains in the sibling integration repository.
Preserve original **RPB-v7** and the formal **RPB-v4** reference. Use
native exports at each card's declared width and fixed classifier head recipes.
PCA is a standalone raw-data baseline. The historical native32 budgets below
belong to their original cards, not a universal v20 dimension restriction.
Older encoder designs remain archived references and compatibility inputs;
routine research does not retrain every historical variant.

Use [the reporting standard](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/RESULTS_REPORTING_STANDARD.md) and
[the version registry](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md). Shared evaluation
stays separate from encoder training and decoding.

## Preserved 10 October spectral milestone and proposed direction

The latest completed quality record at that milestone was
[RPB-v19.alt-01](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/SPECTRAL_CONFIRMATION_V1.md),
the unchanged v19 design on fresh masters 930905, 930906 and 930907 for each of
**TEMPO-4** slow-component timing and **AMP-2** component balance, both at
designed complexity **5/5**. All 12 prospective conditions pass at native32,
512 CUDA updates, batch 8, unchanged fixed heads and 100% coverage. Intact
Linear/Neural means are 99.48%/98.44% for timing and 99.48%/98.87% for balance;
deleted means are 93.23%/93.23% and 94.79%/86.89%, respectively. Most strength
comes from the fixed prior. Modest training gains on fresh deleted means are
not uniform across cohorts; query VALIDATION MAE is 0.74248 for timing and
0.75793 for balance.

The proposed next direction at that milestone was a separate harmonic linear decoder architecture,
keeping the fixed prior, data, native32 and heads unchanged. It is prospective:
the implementation and new quality cohorts are not allocated. Follow the
[historical continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-10_AFTER_SPECTRAL_CONFIRMATION.md).
There is no default promotion. Original saved **RPB-v7** and the formal
**RPB-v4** reference remain protected.

The preceding [RPB-v19 screen](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/MULTIBAND_SCREEN_V1.md) introduced the design;
**RPB-v4 remains the formal evaluated reference**, with original saved **RPB-v7**
protected. V19 keeps 20 learned shape coordinates and 12 fixed low/high spectral
relation coordinates inside native32. Its two-cohort screen passed the
prospective joint gate on **TEMPO-4** slow-component timing and separate **AMP-2**
component balance, each at designed complexity **5/5**, with 100% coverage.
Initial controls are already strong; credit the architecture prior rather than
claiming waveform training learned these relations. Learned shape and
reconstruction still need demonstrated progress.

Its separate fresh confirmation is now complete above. Preserve native32,
the fixed heads and separate task populations, together with the
[historical screen continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-10_AFTER_MULTIBAND.md).
No TEST/stress access or default promotion follows the screen.

The preserved [RPB-v18 result](UNIT_TEMPORAL_RELATION_SCREEN.md) normalizes each
fixed four-spacing timing group inside native32. Its two-cohort gate passed,
then all five matched known TEMPO-3
cohorts (designed complexity 4/5) reach 100%/100% intact and 100%/99.95% deleted
Linear/Neural accuracy, with 100% coverage. Initial timing scores are already
almost perfect; credit the fixed prior. All matched v17 scores are reused,
with no old encoder or head refits and no promotion.

The preceding [fresh v17 confirmation](FIXED_PRIOR_FRESH_CONFIRMATION.md)
retains 100%/100% intact and 82.19%/99.90% deleted means, with 100% coverage.
Its 70.31% deleted Linear cohort fails both joint rules. Both experiments keep
the unchanged native32, .15 context deletion, B8, 512 updates and fixed heads.

Keep the [v18 milestone](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/MILESTONE_2026-10-10_RPB_V18.md)
and its [historical continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-10_AFTER_FRESH_V17.md).
The coherent two-component challenge proposed there now has passed information
admission and the completed v18/v19 screen linked above. Do not infer universal
quality from either the preserved single-component result or a two-cohort screen.
The chronology below preserves the earlier development path.

The [TEMPO-3 comparison](STRUCTURED_HARD_TIMING_DIAGNOSTIC.md) completed once
and passed its independent audit. This is structured harder timing at designed
complexity 4/5, with unchanged native32, .15/B8/fixed 512 and heads. Fresh late
RPB-v7.alt-05 and early RPB-v10.alt-05 have intact linear means55.47%/52.81%
and deleted means54.22%/53.75%, all at 100% coverage. Neural means are54.64%/54.90%
and54.01%/54.32%. Raw fixed heads also stay near chance. The observed-only
cross-feature analytic timing rule solves every saved quality row, so this is
not evidence that the legal observations lack usable timing. Neither group is
promoted, and these fresh cohorts do not replace or damage original saved v7.

The [saved-TRAIN diagnosis](STRUCTURED_HARD_TIMING_TRAIN_DIAGNOSTIC.md) is now
complete without model execution, head refits or PCA. Early-v10 fixed neural
accuracy is 82.06% on TRAIN and54.90% on intact VALIDATION; raw is99.69%/50.73%.
Same-support timing in query targets is 100%, versus68.28% for late and61.02%
for early reconstruction. Those four banks use different masked contexts and
cannot establish information loss in a single full-context native32 vector.

**RPB-v13 — Early mixer with visible adjacent differences** is now measured
under [its frozen card](../../evaluation/cards/visible_difference_v1.md).
The [report](VISIBLE_DIFFERENCE_DIAGNOSTIC.md) retains all five candidate CUDA
trajectories, all fixed heads and reused controls. Original common initialization
and all 15 point0 features match; the new branch adds 3,072 values for 228,877 total.
Mean query MAE improves 0.644282/0.682958→0.622320/0.664399 TRAIN/VAL, while intact
linear/neural accuracy changes 52.81%/54.90%→53.28%/52.03% and deleted
53.75%/54.32%→52.81%/50.52%. Coverage is 100%. Intact linear improves in two
cohorts, falls in two and ties in one; the small mean gain is not consistent.
The saved CPU audit passed 17,152,339 checks/280 witness archives. Its failed v1
basename check and additive, exact-source-bound repair are preserved. No model
was rerun on CPU and no old head was refitted. The input route and capacity
changed together; this result cannot establish encoder information loss.

The [architecture screen](TEMPORAL_ARCHITECTURE_SCREEN.md) now measures
RPB-v14's generic temporal relations and RPB-v15's label-free dynamics objective.
V14 passes the two-cohort75%/both-heads/both-views gate, then reaches97.03%/95.10%
intact and72.97%/75.21% deleted across five known cohorts. Its initial scores
are stronger, so credit the prior; waveform learning weakens timing. V15 stays
near52% and stops after two cohorts. [V16](PARTITIONED_TEMPORAL_RELATION_SCREEN.md)
separates20 shape+12 odd coordinates, reaching100% intact but failing the
continuation gate on master76373's69.53% deleted neural accuracy. Stop its expansion.
[V17](FIXED_PRIOR_TEMPORAL_RELATION_SCREEN.md) freezes only432 generic odd weights
while shape/backbone/decoder train, preserving useful coordinates inside native32.
All four designs have their own frozen card, source, tag and checkpoint namespace.

That architecture-screen milestone proposed fresh-source verification of v17.
The five-cohort RPB-v17.alt-01 follow-up is now complete, and the subsequent
v18 development screen improves deleted Linear accuracy on those known cohorts.
The [historical architecture-screen continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-10_ARCHITECTURE_SCREEN.md)
preserves the earlier decision. That milestone used the dated continuation and
two-component challenge linked above; present direction is at the top of this
document. No TEST/stress or default promotion followed.
The earlier [independent TRAIN-support proposal](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-10_AFTER_VISIBLE_DIFFERENCE.md)
is deferred and preserved, not measured. Avoid expanded comparisons of weak
candidates and do not rescue v15/v16 by tuning their frozen recipes.

## Earlier completed milestones

The [fresh early/late confirmation](EARLY_MIXER_CONFIRMATION_DIAGNOSTIC.md)
remains a separate TEMPO-1/AMP-1 fixed 512 record for RPB-v7.alt-04/RPB-v10.alt-04.
Timing Ridge means rise94.21875%→95% intact and88.28125%→92.8125% deleted, but
worst scores fall84.375%→80.46875% and78.125%→75%. Three pairs improve and two
worsen in both views; master65262 loses strongly. Its better mean reconstruction
and worse amplitude deletion score did not promote compact early-v10. Preserve
that record rather than comparing fresh harder cohorts as a decline of its weights.

The [continuous learning-curve diagnostic](EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md)
is complete for RPB-v7.alt-03 and RPB-v10.alt-01. All five paired live controllers
retain 0/512/1024/2048, with unchanged native32, coordinate15 policy, batch 8 and
heads. Longer training improves both groups' reconstruction without monotone
timing classification benefit. At 2048, late/early intact linear means are
92.1875%/92.65625% and additional-deletion means are 87.8125%/91.71875%, with
100% coverage. Final amplitude quality favors the late group. All eight standard
quality panels, budget-specific training tables and conditional intervals are
in the linked report. No budget is selected and neither group is promoted.

The completed [matched-target gain diagnostic](MATCHED_TARGET_GAIN_DIAGNOSTIC.md)
uses fresh RPB-v10.alt-02/RPB-v11 at fixed 512 under [its frozen card](../../evaluation/cards/matched_target_gain_v1.md).
Timing intact/deletion linear means are 96.40625%/92.5% for the control and
95.3125%/84.6875% for gain, at equal 100% coverage. TRAIN/VALIDATION original-query
MAE rises from 0.075181/0.076729 to 0.216641/0.214659; every gain instance is worse
on both splits. All six numeric guards fail. Stop this recipe without changing
its gain range, deletion rate, budget or heads. Amplitude deletion also worsens.

The completed [pooled-context diagnostic](POOLED_CONTEXT_DIAGNOSTIC.md)
uses fresh RPB-v10.alt-03/RPB-v12 at 512. Compact/pooled timing linear means are
98.28125%/90.9375% intact and 97.03125%/83.125% deleted. All five cohorts lose
in both views, despite both mean original-query MAE guards and equal 100%
coverage passing. All four timing numeric guards fail; the joint decision fails.
The wider route and first-layer capacity change together, so do not assign a
causal defect to the compact projection from these results. The four full
seven-method task/view panels, separate initial controls and all cohort details
are preserved. Stop pooled width and gain without tuning heads, rates or budgets.

Skip half-mixer and further width variations. The pooled milestone selected the
fresh fixed 512 confirmation now completed above. Preserve original RPB-v7,
active RPB-v4, every earlier group and every retained curve point. The
[historical pooled continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-09_AFTER_POOLED_CONTEXT.md)
preserves that prospective decision.

## Completed milestone

No new experiment. The [saved report](../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/diversity-128/report.json)
uses three synthetic development masters 1701/1802/1903. Each run has 128 training
source pairs (256 examples), 64 validation pairs (128 examples), and 64 fresh
test pairs (128 examples). Encoder training completes 512 updates at batch size
8 on the verified RTX A2000 CUDA device. The linear head is ridge with penalty
1; the neural head is tanh-16 with Adam 0.01 for 100 updates. Heads have identical
recipes and paired initialization policies; their weights fit separately.

Timing means across the three runs, with no PCA on encoder exports:

RPB-v2 uses channel mixing and channel reconstruction. RPB-v4 uses a learned
global bottleneck.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | — | — | — |
| PCA only — no encoder | 32 | — | — | — |
| RPB-v2 | 32 | 69.3 | 62.5 | 100 |
| RPB-v4 | 32 | 94.8 | 100 | 100 |

The two baseline rows are unmeasured **under these conditions**. Older raw/PCA
measurements used fewer training examples and cannot fill these cells. Analytic
raw rules score 100%, but are separately labelled solvability checks.
RPB-v4 linear scores by master are 98.44%, 99.22%, and 86.72%; its neural head
is correct on all 384 timing test examples. Three source-group conditional
intervals are not an uncertainty estimate across encoder retraining.

The [validation record](../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/diversity-128/validation-report.json)
also provides fixed-query standardized MAE and synchronized encoder time:

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v2 | 512 | 0.0421 | 0.0454 | 16.4 |
| RPB-v4 | 512 | 0.0693 | 0.0730 | 15.8 |

RPB-v2 reconstruction consumes three contextual channel vectors (96 signal
numbers); RPB-v4 uses only the exact global 32-number vector. This is a different
reconstruction capacity, not two decoders reading the same bottleneck. Timing
ranges overlap and do not establish a speed advantage. Encoder and head budgets
are separate; 4096 sampled row presentations are 16 equivalent passes, not 16
guaranteed full epochs.

RPB-v4's separate direction/level runs score 100% with either native head.
Its amplitude runs score 100% linear and 99.48% neural. Each task currently trains
its own model; this does not establish one universal embedding across tasks.
Severe missingness remains a measured weakness. These are development findings,
not consumer acceptance.

## Phase 1 Establish the no encoder comparison

**Completed:** the [native baseline comparison](NATIVE_BASELINE_COMPARISON.md)
implements this shared archive protocol and reports all four tasks. Timing
validation means are 48.7%/52.8% raw, 51.3%/69.6% PCA-only, and 93.8%/99.9%
RPB-v4 (linear/neural), all with 100% coverage. Encoder checkpoints are unchanged.
The definitions below record the frozen phase-1 procedure. Phase 2 is also
complete; its fresh TEST results are kept separate from these validation scores.

The small generic archive readout protocol lives in shared evaluation.
It consumes legal raw TRAIN/VALIDATION observations and archived native feature
surfaces, labels, source IDs, and support. It does not instantiate an encoder or
retrain one. Reuse the existing training-only normalizers, ridge, tiny head,
PCA and grouped scoring primitives. Build/test/run only in the managed container.

Freeze a new card before any fitting. Use the saved diversity-128 cohorts and
RPB-v4 milestone-512 native global TRAIN/VALIDATION features for all four tasks
and three masters. Validation-only results remain development diagnosis. No
TEST archive is opened for this phase and no historical report is overwritten.

Compare these rows:

1. Raw data: 288 observed values plus 288 visibility flags, training-only scaling
   and zero filling of missing values. This is a 576-number reference.
2. PCA only — no encoder: the same legal raw representation compressed to 32
   components using training data only.
3. RPB-v4: exact saved native 32-number global export, with no projection.

Use the same declared training rows, labels, split/source manifests, fixed head
recipes and readout budgets. Pair initialization seeds for the two 32-number
methods; both have ridge 66/neural 562 fitted parameters. Raw576 has a larger
head and is reported as a larger-input reference. Freeze the observation scaler,
PCA input definition, supports, seed policy, and repetitions before scores.
Do not use hidden clean targets or labels to fit raw preprocessing/PCA.

Save fitted assets, predictions, valid/total counts, per-master paired effects,
conditional source-group intervals, and immutable input hashes. Preserve the
complete declared seed set; do not select the most favorable repetition.
Retain RPB-v2's existing milestone as a named historical comparison rather than
adding another encoder training path.

## Phase 2 Re-evaluate RPB v4 on fresh development data

**Completed:** the [native curve advance](NATIVE_CURVE_ADVANCE.md) implements
the separate shared native-only driver and its frozen card. CUDA correctness
and focused container tests passed, followed by an independent artifact audit.
Three timing trainers continued through 0/128/512/2048. Native linear VALIDATION
selected 512 (94.79%), ahead of 128 (92.19%) and 2048 (90.10%), before any TEST
generation. Fresh timing TEST means are 94.01% linear / 96.09% neural, versus
51.82% / 79.51% PCA-only and 46.61% / 50.61% raw; coverage is 100% throughout.
Direction, level and amplitude each reach 100% with trained native32 heads.

Reconstruction improves through 2048, but native linear validation does not.
The remaining timing gap is consistency across trained instances and moderate
additional missingness. The exact untrained timing controls score 55.47% linear
/ 65.19% neural, showing a substantial training contribution. The procedure
below records the completed protocol; historical compressed drivers stay intact.

The separate native-only training/selection protocol supplies this curve.
The historical global driver requires three variants, computes compressed tiers,
and selects a budget with PCA12. Changing the displayed table cannot change that
protocol. Leave its cards, executable semantics and archived results intact.

The new protocol registers RPB-v4 as its sole trained encoder. Use fresh declared
development masters/source cohorts, native global32, and the same head recipes.
This is development re-evaluation; formal consumer confirmation still requires
the separate evaluation-policy contract and acceptance thresholds.
Carry a matched untrained RPB-v4 control where the claim needs training evidence;
it is not another older architecture. Raw/PCA-only controls fit on the same legal
training data and stay fixed across checkpoint milestones.

For the timing task, freeze milestones 0/128/512/2048 and one common positive
checkpoint budget selected by mean **native32 linear validation accuracy** over
every declared master. Zero is initialization diagnosis only. An unsupported
primary fit excludes that budget; exact ties choose the smaller positive budget.
Use no PCA after the encoder, including in selection. Other task scope and
fixed sanity-check budgets are declared in the card before generation.

Verify actual CUDA weights, inputs, loss, finite gradients, changed weights,
and exact checkpoint/export reconstruction before long training. Preserve one
continuous optimizer and immutable milestone checkpoints. Persist selection
before generating independent fresh test sources. Restore the selected model
and fitted readouts without test refitting. The neural head remains secondary;
its result cannot replace the primary after scores are known.

Report native RPB-v4 versus PCA-only32 and raw576, full coverage/counts, paired
effects and uncertainty. Repeat the fixed-readout stress protocol with identical
corruptions and no fit under stress. Do not routinely train RPB-v1/v2/v3-mean.

## Phase 3 Change the encoder only for a measured gap

**Completed:** the [paired pooling record](PAIRED_POOLING_ADVANCE.md) reports
RPB-v5 versus the frozen RPB-v4 reference at 512. Intact timing linear accuracy
is84.38% versus94.53%; additional30% deletion is66.15% versus85.16%, with equal
100% coverage. V5 TRAIN/VALIDATION fixed-query MAE also worsens. The claimed
advance is rejected at this fixed budget; RPB-v4 stays active. The unchanged
card and full independent audit preserve this measured result rather than
universally rejecting the architecture.

The following conditions record that completed experiment: learned global
pooling directly over aligned patch/channel states, before separate channel
summaries compress them.
RPB-v4 already mixes aligned channel states; the proposed pool bypasses its
subsequent per-channel compression. It aims to preserve joint temporal
relationships more consistently at the served 32-number bottleneck. These measurements
do not establish that compression caused the remaining errors.

Keep the temporal blocks, channel mixer, legal preprocessing, reconstruction
objective, exact served 32-number decoder input, sampling/mask policy, fixed heads and
512 encoder updates. Freeze pooling support/position/channel rules and pair
common-module initialization with RPB-v4's point-zero assets. Record added
parameters and training cost; equal export width does not mean equal encoder
capacity. All-absent exports remain zero and invalid.

Use only the selected RPB-v4 instances and their original TRAIN-fitted assets
as the relevant reference. Do not bring every older variant back, add auxiliary
losses, tune classifier capacity, or fit under stress. Declare intact timing
and the existing 30% additional-deletion case before scores; its legal raw rule
retains full accuracy and coverage. Primary evidence is native linear accuracy,
per-master spread and within-master paired effects, with neural results secondary.

Freeze a candidate card and durable fixed-budget manifest before generating a
new TEST namespace. The opened phase-2 TEST data are now development evidence.
Reuse legal TRAIN/VALIDATION cohorts for paired development if needed, and apply
candidate and frozen RPB-v4 to common new TEST sources and corruption masks,
without TEST refitting. Preserve every declared master/head and point-zero
diagnostic. No across-retraining confidence interval is claimed from three masters.

If raw/PCA controls and RPB-v4 already solve the simple synthetic tasks, move to
harder declared tasks or the intended consumer dataset instead of optimizing a
saturated score. Consumer confirmation requires the dataset/task, chronology,
support, thresholds and cost contract to be defined before acceptance.

## Completed diagnostic: continue v5 on TRAIN and VALIDATION only

The [optimization diagnostic](OPTIMIZATION_DIAGNOSTIC.md) is complete and
independently audited. Mean linear VALIDATION accuracy was84.38% at 512,
84.38% at1024 and82.81% at2048, while reconstruction error improved at both
new budgets. Longer training did not improve mean linear access under the fixed
heads. This does not change the fixed 512 rejection or establish universal
failure at every possible budget. The following records its frozen procedure.

Save and commit the fixed 512 milestone before further source work. A separate
prospective recipe may resume each exact saved v5 point512 model, AdamW state,
scaler and counter stream through absolute1024 and2048 updates on the same
TRAIN observations. Keep all three masters, native32 and the fixed readout
recipes; candidate heads at new points fit only TRAIN. Record any settings
step-ceiling override and cumulative cost without resetting initialization,
optimizer, preprocessing or sampling/mask progression.

This is an optimization diagnosis using known VALIDATION, fixed-query
TRAIN/VALIDATION reconstruction and native linear access. Do not open or
rescore TEST/stress, introduce a replacement seed, alter the fixed 512 rejection,
or promote v5 based on it. Its own card was frozen before running. Any later trained-budget comparison needs its own
declared reference/budget and new TEST namespace before unseen scoring.

## Completed bounded candidate: RPB-v6 context deletion

The [context-deletion advance](CONTEXT_DELETION_ADVANCE.md) is complete and
independently audited. Timing TEST means are98.96%/100% intact and97.14%/99.65%
with30% additional deletion (linear/neural), versus93.49%/96.35% and86.72%/93.06%
for frozen RPB-v4, with full coverage. The quality guards pass, but mean original
fixed-query TRAIN/VALIDATION MAE worsens to.08552/.08577 from.06160/.06349.
This unresolved reconstruction tradeoff prevents promotion. The following
records the completed card.

Keep the RPB-v4 inference architecture, all225,805 parameters and exact served
32-number decoder input. Change only its training view: remove30% of originally
visible coordinates through a separate deterministic counter stream, repairing
only extra deletion to retain two original visible patch groups for each target
eligible channel. Keep the original target queries, loss, scaler, batches,
optimizer,512-update budget and classifier recipes.

Freeze [context-deletion-v1](../../evaluation/cards/context_deletion_v1.md)
before measurement. Reuse the three retained TRAIN/VALIDATION cohorts, paired
point-zero weights and frozen RPB-v4/control readouts. Preserve enabled-context
CUDA test evidence and source identity before the new TEST namespace is opened.
Ordinary training must reject resuming a checkpoint carrying the new training
policy; ordinary inference may serve its normal native32 embedding.

Both mean linear primaries (intact and30% additional deletion) must improve,
with equal coverage, no lower worst-master primary score and no worse mean
TRAIN/VALIDATION fixed-query reconstruction error. Neural and other stress
results stay secondary. Preserve every seed and audit the artifacts before any
explicit promotion.

## Completed diagnostic: RPB-v6 optimization with the same training view

The [context optimization diagnostic](CONTEXT_OPTIMIZATION_DIAGNOSTIC.md) is
complete and independently audited. It keeps the 30% context policy, native32,
three original cohorts and fixed heads at 512, 1024 and 2048 updates. Fresh
replay matches the saved 512 weights, buffers, AdamW, scaler, absolute counters
and context deletion counts before the same live optimizer continues.

Mean TRAIN/VALIDATION fixed-query MAE reaches 0.035736/0.035845 at 2048,
passing the historical v4 fixed-512 reconstruction reference. But mean linear
VALIDATION accuracy falls to 96.875% intact and 95.3125% under the one fixed
additional 30% deletion view, versus 98.9583%/98.4375% at replay 512. At 1024,
reconstruction still exceeds the reference and deletion-view linear accuracy
also falls. No measured point satisfies both reconstruction and preserved
512-update linear quality. All budgets and masters remain in the record.

The same ordinary TRAIN-fitted heads score intact and deleted VALIDATION without
fitting on corruption. This diagnostic accessed no TEST or old stress payload,
selected no checkpoint and made no promotion decision. The original fixed-512
tradeoff and active RPB-v4 reference remain unchanged.

## Completed replication: five independent paired masters

The [five-master replication](CONTEXT_REPLICATION_ADVANCE.md) is measured under
the unchanged [context-replication-v1 card](../../evaluation/cards/context_replication_v1.md).
Fresh RPB-v4/RPB-v6 training at 512 updates gives intact native linear means of
94.0625%/95.78125% (+1.71875 percentage points) and additional 30% deletion means
of 87.8125%/95.9375% (+8.125 points), with equal 100% coverage. Mean TRAIN/VALIDATION
MAE is 0.067593/0.071309 for v4 versus 0.091179/0.092984 for v6; both declared
reconstruction guards fail. RPB-v4 remains active. Independent archive audit v3
passed 103,787,020 checks; preserve both failed reader attempts, their correction
rationale and all five outcomes.

The following records the completed procedure. Masters 4404, 5505, 6606, 7707
and 8808 used the same new TRAIN sources within each pair, with complete
initialization, scaler and original row, patch-mask and Torch streams paired.
Only the existing context policy differed; architecture, reconstruction loss,
native32 and classifier recipes stayed fixed.

Use 128 TRAIN, 64 VALIDATION and 64 fresh TEST source pairs, batch size 8,
C3/H32/F3 with patch length 8 and 10% natural missingness. Keep ridge penalty 1
and the secondary tanh-16 head with Adam 0.01 for 100 updates and all three
declared repetitions. New heads fit TRAIN only. No PCA follows an encoder;
the retained raw and standalone PCA32 controls use the same legal sources.

A shared development-only baseline phase prepares v4's exact point-zero and
512-update checkpoints, TRAIN/VALIDATION exports and fitted controls/readouts.
It performs no selection or TEST generation. The paired phase verifies the
frozen baseline assets and fresh v6 training, then durably records the complete
comparison before generating a new common TEST namespace and corruption masks.
Known VALIDATION cannot select a different replication budget. The question is
whether the fixed-512 context benefit repeats across independent training,
with reconstruction and cost reported alongside quality. The point-score gain
repeats, but the full guard does not pass. Conditional source-group intervals
are not across-master uncertainty, and this is not consumer confirmation.

## Completed lighter-policy diagnostic: RPB-v7

The [RPB-v7 record](CONTEXT_LIGHTER_VALIDATION_DIAGNOSTIC.md) completes the fixed
0.15 request hypothesis at 512 updates on all five known TRAIN/VALIDATION
masters under the frozen
[lighter-policy card](../../evaluation/cards/context_lighter_validation_v1.md).
The same mode2/mixer1/native32 architecture, optimizer, reconstruction targets,
loss, streams and fixed heads remain in use. Five new candidate trajectories
are paired with ten retained v4/v6 trajectories. Independent archive audit v3
passed 67,097,380 checks; preserve the measured capsule and save its passed audit before any
further production changes.

Mean intact/deletion native linear accuracy is 97.96875%/95.15625%, compared
with v6's 95.3125%/93.90625%, at equal 100% coverage. Mean TRAIN/VALIDATION MAE
improves over v6 to 0.081179/0.083630, but both guards against v4's
0.067593/0.071309 fail. Every master's reconstruction remains worse than v4;
neural means decline and master 8808 loses 5.46875 linear points under deletion.
The joint target is unresolved and v4 remains active.

The separate policy `rpb-training-context-deletion-015-v1` is implemented.
CUDA admission passes actual gradients/weight changes, exact continuous
2→4→6 continuation against uninterrupted 6, full initialization/scaler pairing,
original query/support loss, semantic-coordinate repair and request nesting,
snapshot immutability and unchanged disabled/v6 defaults. Expected recipe
binding rejects cross-policy initialization and the v6-only replay rejects v7.
Ordinary training still rejects tagged resume; serving applies no deletion.
The exact initialization and ordinary feature/query references are checked
before fitting. All 135 readout pipelines are supported and all 45 native
fit/ordinary-prediction parity records pass.

No TEST/stress, rate/budget search, checkpoint selection or promotion occurred.
The new fixed VALIDATION deletion view is distinct from historical TEST/stress.
Known development gains do not supply fresh held-out confirmation. The card
and measured report remain frozen evidence, separate from the current next plan.

## Completed balanced-view diagnostic: RPB-v8

**RPB-v8 — Global bottleneck with balanced context views** is implemented and
measured under the frozen [balanced-view card](../../evaluation/cards/context_balanced_validation_v1.md).
It alternates 256 ordinary and 256 deletion-0.30 attempts at 512 completed updates,
with unchanged mode2/mixer1/native32 architecture, original targets/loss and heads.
The [five-master diagnostic](CONTEXT_BALANCED_VALIDATION_DIAGNOSTIC.md) reports
mean native linear accuracy of 95.93750% intact and
91.71875% under the exact saved v7 additional-deletion view,
at 100% coverage. Mean TRAIN/VALIDATION MAE is
0.085022/0.087620.
The predeclared joint development guard **failed**. Independent archive audit v2
passed 67,099,959 checks; actual CUDA admission and prior-policy regressions passed.
V8 is not promoted, and RPB-v4 remains active. No TEST/stress, checkpoint selection,
head tuning or post-encoder PCA occurred. Five new v8 trajectories use ten
retained v4/v7 references; all five masters and both heads remain in the report.

## Completed TRAIN objective diagnosis

The [TRAIN-only diagnostic](TRAINING_OBJECTIVE_DIAGNOSTIC.md) examines fifteen
retained v4/v7/v8 instances without encoder updates or head refits. The proposed
weight1 all-pair residual-difference loss improves the local fixed-Ridge gradient
direction on two v7 masters and worsens it on three. It fails the prospective
five-master rule and will not be trained or tuned. Actual CUDA admission and
independent saved-arithmetic audit v3 passed; the failed v2 reader and its
cross-backend correction are preserved. The result does not establish a
generalization gain or prove irreversible timing-information loss.

## Earlier focused actions

The [RPB-v9 comparison](NATIVE_VIEW_AGREEMENT_VALIDATION_DIAGNOSTIC.md) is complete.
Actual CUDA admission and independent archive audit passed, but the fixed recipe
failed all six numeric guards and produced severe reconstruction instability.
RPB-v4 remains active. Stop this mechanism; do not search its coefficients,
rate or budget, and preserve all five runs including master4404.

The separately frozen [saved-TRAIN scale lesson](NATIVE_VIEW_LOSS_SCALE_TRAIN_DIAGNOSTIC.md)
is complete: no scale floor activated, and agreement grew later in every run.
Its scalar traces do not identify objective-specific gradients or establish cause.
The original dated [continuation note](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-08.md)
remains the historical pre-diagnosis record.

The [v7 decoder calibration](V7_DECODER_CALIBRATION_DIAGNOSTIC.md) then froze all
214,277 non-decoder parameters and trained the existing 11,528-parameter decoder
for 128 updates on CUDA. Mean TRAIN/known-VALIDATION MAE fell from
0.081179/0.083630 to 0.054039/0.056989. Every master is below its paired v4
fixed 512 error. Exact encoder/native/scaler invariants and the independent
25,450,287-check saved-arithmetic audit passed. Classification is cached and
unchanged; there were zero new encoder updates or head fits. This demonstrates
decoder recoverability on these cohorts, with no new embedding tag or promotion.

The separately frozen [fresh equal-budget replication](FRESH_DECODER_REPLICATION_DIAGNOSTIC.md)
is now complete. Both v4 and v7 received encoder512 and decoder128 on five new
cohorts after explicit actual-CUDA admission for both policies. Linear means
favor v7 by 0.625 percentage points intact and 7.8125 under additional deletion,
but the intact paired-master interval crosses zero and v7's worst intact score
falls to 80.46875%. Mean post-calibration TRAIN/VALIDATION MAE is
0.056375/0.059200 for v7 versus 0.053609/0.056835 for equal-budget v4. Recovery
passes against v4 before calibration; equal-budget guards and the intact
worst-master guard fail. Independent saved-evidence audit passed 67,932,331
checks. Exact frozen features justify reusing classification after calibration.
There was no TEST/stress access, promotion or new tag. The
[earlier continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-08_AFTER_DECODER_CALIBRATION.md)
remains the historical prospective record.

The [saved-TRAIN reliability diagnosis](SAVED_NATIVE_RELIABILITY_DIAGNOSTIC.md)
completed the earlier [plan](FROZEN_NATIVE_RELIABILITY_PLAN.md) and
[dated continuation](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-08_AFTER_FRESH_DECODER_REPLICATION.md).
It compares every paired v4/v7 TRAIN export, retained fixed-head margins,
within-source separation, feature variation and encoder training trace.
Master12412 is already weaker on TRAIN as well as VALIDATION; this cannot be
described only as a held-out generalization failure. Mixed geometry does not
identify a common architecture defect. The diagnosis is descriptive, with zero
model updates, head refits, new quality generation or held-out analysis payloads.
The later amplitude and early-mixer milestones are preserved separately; the
current planned direction appears at the top of this document.
