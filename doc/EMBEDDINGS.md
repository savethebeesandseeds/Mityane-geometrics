# Encoder research and leading references

This is the current scientific summary: which encoders led, what improved, what
was tried and where to continue. The [Hero records](../web/dashboard/README.md)
hold work states. The [handoff addendum](EMBEDDINGS_HANDOFF_ADDENDUM_2026-10-18.md)
holds migration locations, checkpoint identities and the exact pause boundary.
Dated archived status and version notes describe their own historical stage.
The family-owned [live CPB ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md)
holds current checkpoint standings, scoped records and their change history.

The lane seeks a compact, trustworthy representation of observed multichannel
history that retains temporal structure and relationships across tasks, views
and readouts. Masked reconstruction and uncertainty objectives train the encoder;
fixed readouts independently measure retained information. The
[project charter](operations_over_representations_design_spec_v0.1.0.md) keeps the
broader project domain-agnostic and makes operation utility a separate claim.

## Leading encoders and their roles

| Role | Encoder | Recorded achievement and remaining limit |
| --- | --- | --- |
| Broad clean-causal breakthrough | **CPB-v4.market-01** | **19/22 original conditions**, all eight TEMPO-5 retention checks, major trend/relation gains and the best saved known-validation reconstruction. Linear volatility, linear activity and TRAIN query MAE failed their floors, withholding formal promotion. |
| Historical balanced research leader | **RPB-v24.market-01.alt-01, native36** | **8/9 market conditions plus all four primary frozen TEMPO-5 transfer comparisons**, at full coverage using the same market-trained checkpoint. Later diagnosis limits its clean causal eligibility; its measured success and historical leading designation remain preserved. |
| Stable clean research baseline | **CPB-v3.market-01** | First genuinely fresh causal-training reference; still leads linear volatility. Baseline status is not an overall quality ranking or a project-wide Mityane selection. |
| Later historical market success | **RPB-v30.market-01** | **9/9 market conditions and 8/8 strict native36 floors**. Independent synthetic transfer remained pending, so the historical balanced designation stayed with native36. Its trained ancestry is outside the clean causal comparison. |
| Latest catalogued historical instance | **CPB-v14.market-01** | Full trust and coverage; 3/22 strengthened conditions and zero new protected records in that run. Its exact recipe is closed; independent metadata review and compatible application are complete. The separate six-run V15 study has its own identities. |
| Latest completed controlled study | **CPB-v15.objective-001**, three fresh V4/V15 pairs | February activity and query reconstruction improve in all three pairs; six scoped January records include two V15 and four control records. The prospective verdict is mixed because volatility, trend tolerance and timing retention fail. No broad replacement follows. |

V4 is the broad clean-causal reference to foreground when discussing successful
improvements. Later V6, V8, V11 and V13 add specific strengths that remain
protected. No clean candidate has combined every protected strength or received
overall promotion. Broad success, baseline status and formal promotion are
separate facts. Historical 19/22, 8/22 and 3/22 counts use different prospectively
frozen floors; they cannot be ranked as identical tests.

Evidence: [V4 review](../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V4_REVIEW.md),
[native36 transfer milestone](../.archive/embeddings/pre-cleanup-2026-10-18/doc/MILESTONE_2026-10-14_NATIVE36_TRANSFER.md),
[historical RPB ledger](../.archive/embeddings/pre-cleanup-2026-10-18/doc/ENCODER_SEARCH_LEDGER.md)
and [V14 results](../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V14_RESULTS.md).

## Historical V3–V14 measurement leaders

The tables below preserve the pre-V15 clean-causal final-checkpoint leaders used
as historical comparators. Use the [live ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md)
for current standings, including the new V4-control and V15 observations. These
dated comparator tables are task-specific development results.
MKT-HIST-2 readouts measure the observed past window on known chronological
VALIDATION, using 112 TRAIN and 32 VALIDATION examples. Values below are rounded;
the [saved V14 comparison](../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V14_RESULTS.md)
and [clean research ledger](../.archive/embeddings/pre-cleanup-2026-10-18/doc/clean_causal_encoder_research_ledger.json)
retain full precision and earlier results. V14 leaves these leaders unchanged.

| Past-window task | Linear R2 leader | Neural mean R2 leader |
| --- | --- | --- |
| Volatility | V3, **0.9647** | V8, **0.7360** |
| Trend | V4, **0.9072** | V4, **0.5699** |
| Activity | V11, **0.9094** | V13, **0.7625** |
| Cross-channel relation | V6, **0.6221** | V13, **0.4572** |

Reconstruction measures fixed encoder/decoder queries and remains separate from
readout quality. TRAIN and known-VALIDATION records answer different questions.
Lower is better for these saved MAE and Gaussian NLL measures.

| Fixed-query split | MAE leader | Gaussian NLL leader |
| --- | --- | --- |
| TRAIN | V7, **0.241805** | V7, **-0.408827** |
| Known VALIDATION | V4, **0.308317** | V4, **-0.045381** |

TEMPO-5 tests slow lead/lag beneath faster rhythms and distractors. Its saved
accuracy leaders depend on the input scaler, validation view and head. The
deleted view applies an additional 30% deletion to the same validation groups.

| TEMPO-5 scaler and validation view | Linear accuracy leader | Neural accuracy leader |
| --- | --- | --- |
| Synthetic TRAIN scaler, intact | V13, **98.9583%** | V6, **98.4809%** |
| Synthetic TRAIN scaler, deleted | V13, **95.8333%** | V13, **94.5747%** |
| Market TRAIN scaler, intact | V9, **99.8698%** | V8, **99.2622%** |
| Market TRAIN scaler, deleted | V6, **97.2656%** | V4, **95.6163%** |

Known validation has repeatedly guided development. The clean causal sequence
uses one encoder trajectory per historical version; three neural readout fits are not three
encoder retrainings. V3/V4 native36 versus V5–V14 native40 also changes readout
capacity and width-derived seeds. These results do not establish fresh-period
generalization, forecasting skill or a causal explanation of gains.

## What the search has tried

| Research stage | Main questions and outcomes |
| --- | --- |
| Earlier RPB designs | Pooling, channel mixing, context deletion, difference features, temporal relations, spectral priors and capacity changes. Better reconstruction often failed to preserve task information. |
| RPB-v23 through v30 | Change branches, protected trained prefixes, normalization and decoder objectives. Native36 became the balanced research leader; V30 later passed the full market gate with transfer pending. |
| CPB-v1 and v2 | Causal execution with inherited weights. Runtime cutoff invariance did not establish clean causal-training ancestry. |
| CPB-v3 and v4 | Establish fresh causal training, then change variance-gradient routing. V4 delivered the broad improvement described above. |
| CPB-v5 through v7 | Separate mean/uncertainty projections, Huber mean supervision, detached-error variance calibration and clipping changes. Gains remained mixed across tasks. |
| CPB-v8 through v10 | Original-position temporal bins, half-local/half-global pooling and a live joint Gaussian objective trial. New timing/volatility strengths accompanied other regressions. |
| CPB-v11 through v14 | Mean plus variance/dispersion summaries, uniform dispersion weights, regularized square-root dispersion and nested trailing windows. V11/V13 added activity/relation/timing records; V14 added no protected record. |
| CPB-v15 with fresh V4 controls | Completed three-seed objective comparison at fixed V4 native36. February linear/neural activity and query reconstruction improve in all three pairs. Six scoped January records include two V15 and four fresh-control records. The prospective verdict is mixed: volatility, trend tolerance and timing retention fail. V4 remains the broad reference. |

**V3 through V15 all start from fresh weights, optimizer state and TRAIN
statistics.** Source reuse is branched design ancestry: V8 uses V6 source, V11
uses V9, V13 uses V11, V14 uses V13 and V15 uses V4. V15 retains native36;
V5–V14 use native40. V1/V2 retain pretrained ancestry.
The [addendum](EMBEDDINGS_HANDOFF_ADDENDUM_2026-10-18.md) and frozen per-version
reports preserve the distinctions and failures.

## Current work and continuation

The owner selected **V16 temporal-change reconstruction** on 19 October, with
the [prospective protocol](../code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/PROTOCOL.md)
declared before building or reading new confirmation values. V16 keeps V4
native36 and its Gaussian value objective, adding a fixed small Huber loss for
hidden differences at minute lags1/4/16. Both endpoints must be original hidden
targets; context deletion supplies no extra targets. Six fresh trajectories at
three new paired seeds compare V4 and V16. March2024 is chosen by calendar
for confirmation; February remains consumed development evidence. As of
20 October, [CUDA engineering](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/engineering-review-005.json)
and [January target parity](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/target-parity-001.json)
have passed. The six fresh trajectories are being executed under the
[frozen run card](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/run-inputs-001.json);
the [execution log](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/study-execution-001.log)
records progress. No V16 quality result is established yet. The selected
direction is separate from the alternative objective blend.
Current work states and acceptance criteria are in [Hero17](../web/dashboard/heroes/17-encoder-cpb.json).

The user resumed embedding continuation on 18 October 2026, beginning with
**V14 metadata review, compatible application and preservation closeout**.
The [closeout protocol](../code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/README.md)
records this scope and the original
[handoff boundary](../.archive/embeddings/pre-cleanup-2026-10-18/doc/HANDOFF_2026-10-18_V14_PAUSED_MIGRATION.md).
The three original proposals passed 1,687 independent checks. The staged current
records passed 116 semantic checks, and their compatible application is recorded
in the [actual application receipt](../.temp/embeddings/closeout/cpb-v14/attempt-001/application.json).
The maintained [catalog](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/current.json)
now records V14, with its [measured instance](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v14.market-01.json)
and [closeout summary](../code/embeddings/evaluation/records/CPB-v14.market-01.closeout.json).
V3 remains the clean research baseline; Mityane has made no model selection.
Original proposals, scientific results and the archived ledger remain unchanged.
Final integrity sealing is recorded separately at
`.temp/embeddings/closeout/cpb-v14/attempt-001/preservation.json`.

The bounded verification resolved all 11,826 required identities. It reuses
earlier bulk observations at their original time and freshly checks declared
controls and new aliases. The original broad scan remains failed: its 11,302
rows are retained, including unrelated historical references outside the typed
scope and schema-context corrections. This is not whole-archive verification.

The [draft closeout plan](EMBEDDINGS_V14_CLOSEOUT_PLAN_2026-10-18.md) names proposed
writable targets and historical alias requirements. Its
[dated input snapshot](EMBEDDINGS_V14_CLOSEOUT_INPUTS_2026-10-18.json) captures an
earlier limited file-integrity inspection. Later documentation edits change
those document hashes; the snapshot is not a current execution admission.

On 19 October the owner approved the [V15 objective comparison](../code/embeddings/evaluation/protocols/cpb_v15_objective_v1/PROTOCOL.md):
three fresh V4/V15 pairs at native36, changing the mean objective to Huber1 while
preserving V4's architecture and variance forward. Source review, actual CUDA
engineering and exact January target parity passed. All six fresh trajectories
completed their fixed 2,048 updates and exact resume checks; the
[training completion record](../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/training-completion-001.json)
records those trajectories. All six development runs, six independent arithmetic
audits and twelve checkpoint causality proofs passed. The
[development seal](../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/development-seal.json)
fixes all six final checkpoints and their TRAIN-fitted heads before February
scoring. All six February confirmation phases and all six confirmation arithmetic
audits passed. The [completed results](../code/embeddings/evaluation/protocols/cpb_v15_objective_v1/RESULTS.md)
and [compact study record](../code/embeddings/evaluation/records/CPB-v15.objective-001.json)
bind the six-run study, all eighteen phase receipts, twelve independent arithmetic
audits and its complete reporting review. The scientific verdict is **mixed**.

**February activity improves in all three pairs**, for both linear and neural-mean
readouts. The median paired linear R2 gain is **+0.186530**. Descriptive fixed-query
MAE and Gaussian NLL also improve in all three pairs. The prospective promising
rule fails because linear volatility improves in only one pair (median
**-0.078143**), median linear trend change is **-0.098598**, below its -0.01
tolerance, and no candidate retains all eight strengthened timing conditions.
Neural-mean trend falls in all three February pairs. All six runs have full
declared coverage and pass causal correctness; timing quality is a separate
measurement. The full 22-condition matrices and every arm/seed remain in the
results, without selecting the best seed.

The study establishes **six new scoped January highs**, retained in the
[live ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md).
V15 sets TRAIN query MAE **0.24003670546837294** at seed945102 and synthetic-scaler
intact linear timing **99.609375%** at seed945101. Fresh V4 controls separately
set neural activity, TRAIN NLL and two neural timing records. All three V15
candidates improve paired January TRAIN/known-validation query MAE and
known-validation Gaussian NLL, while January linear trend falls in all three.
V4 retains the broad clean reference role and both known-validation
reconstruction records. Scoped checkpoint records and broad replacement remain
separate assessments.

The [reviewed reporting card](../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-inputs-001.json)
binds the complete machine report. A preserved reporting interruption and a
separately reviewed successor reporting generation are documented in RESULTS
and the compact record. The successor removes identical duplicate final checks;
scientific calculations are unchanged, every distinct final binding is freshly
verified and no training, head fit or completed experiment was replayed. The
local [execution record](../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/execution-start-001.json)
retains the original admitted run order and source identities. The new work used
C++, Node and shell in the managed Linux container, without Python.

The frozen [V15 source declaration](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v15/README.md)
and [version descriptor](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/CPB-v15.json)
retain their preparation-stage wording. Current progress and measurements belong
in this guide, the live ranking and Hero 17. The catalog remains at V14; this
six-run study records each arm and seed separately and performs no catalog
application or seed selection.

January remains development evidence. Calendar-selected February was scored with
the sealed models and original TRAIN-fitted heads, with no February refit or
tuning. February is now consumed confirmation evidence: a recipe chosen after
seeing these outcomes requires a new confirmation population. The 22 original
protection predicates and all earlier leaders remain preserved. The six-run
study establishes no automatic model promotion or Mityane selection.

The [saved V14 TRAIN-only study](../code/embeddings/evaluation/protocols/cpb_v14_train_diagnosis_v1/PROPOSED_PROTOCOL.md)
remains reviewed preparation, unexecuted and deferred by this V4-based choice.
Two saved batches cannot establish held-out causation. Current priorities,
acceptance criteria and milestone states remain in
[Hero 17](../web/dashboard/heroes/17-encoder-cpb.json).

## Preserved evidence

All links below lead to unchanged documents in the pre-cleanup snapshot:

| Record | Use |
| --- | --- |
| [V14 handoff](../.archive/embeddings/pre-cleanup-2026-10-18/doc/HANDOFF_2026-10-18_V14_PAUSED_MIGRATION.md) | Completed science, exact pause boundary, unfinished reconciliation and resume conditions |
| [V14 results](../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V14_RESULTS.md) | Saved task-specific measurements |
| [Research ledger](../.archive/embeddings/pre-cleanup-2026-10-18/doc/clean_causal_encoder_research_ledger.json) | Scoped historical leaders; the handoff notes V14 metadata has not been applied |
| [Evaluation policy](../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_EVALUATION_POLICY.md) | Inherited evaluation boundaries and controls |
| [Version guide](../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md) | Preserved design/instance identities |
| [Historical index](../.archive/embeddings/pre-cleanup-2026-10-18/doc/README.md) | Earlier research reports, cards, protocols and model guides |

The [archive guide](../.archive/embeddings/README.md) maps old source locations
to current ownership. Frozen source capsules, protocol cards, receipts and
hashes keep their original bytes and path identities. Historical ignored data,
checkpoints, mirrors and transfer inputs remain under `code/embeddings/`.
Archived instructions describe their original environment; current execution
uses only the root managed container and task runner. A historical reproduction
requires its own explicit path mapping and the frozen protocol's admission.

New operation claims need the charter's Operation Contract and linked
Requirements-and-Evidence Record. The inherited
[forecasting records](../code/operations/forecasting/EVIDENCE.md) distinguish
engineering checks from unmeasured forecast/readout/decision utility. Nothing
in this cleanup supplies new quality measurements or changes a historical
trained-model/checkpoint identity. Maintained source has new path-bound source
identities; use its actual current fingerprints for future artifacts.

## Keeping the current summary consistent

Keep scientific aims, research roles and study conclusions in this guide. The
family-owned [live ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md)
owns current checkpoint standings and their dated changes. Update it from bound
results and align the short family/dashboard summaries in the same change.
Preserve earlier leaders, ties and original protocol scopes; retain the tables
above as V3–V14 historical comparators. Keep migration details in the handoff
addendum and detailed evidence in its owning records. Work states and priorities
remain in the Hero JSONs, with the documented refresh after their edits.
