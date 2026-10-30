# Encoder research and leading references

This is the current scientific summary: which encoders led, what improved, what
was tried and where to continue. Hx20, the Temporal Fusion Transformer, and
Hx21, state fusion models, live in `code/fusion/` and are not part of this
ranking. The [Hero records](../web/dashboard/README.md)
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

## Latest comparison: timeframe channels — 29 October 2026

The first fresh comparison on the timeframe contract is complete. One window
is one instrument. Channels are published 1-minute, 1-hour, and 1-day bars,
torch ordinals 4, 9, and 15. Both arms are CPB-family, three layers, four
heads, feed-forward 256, native36, mask 0.25, log-return view, 2048 updates,
context deletion none. Width 64 is seed 949101. Width 128 is seed 949102.
January 2024 development only: TRAIN days 1–21, validation days 22–28, 3072
and 96 rows. The hour and day series are the publisher's own bars.

Validation R², same-window linear probes:

| Task | Width 64 | Width 128 |
| --- | ---: | ---: |
| Anchor volatility | 0.7123727934288147 | 0.7682422126010569 |
| Anchor trend | 0.18541166081492155 | 0.22475307149870904 |
| Anchor activity | 0.9954692473567033 | 0.9955882717924301 |
| Cross resolution | abstained, coverage 0 | abstained, coverage 0 |

Width 128 is ahead on volatility and trend. Activity is saturated because the
window already contains log volume. Trend remains the weak readout. Cross
resolution needs eight distinct coarser bars, and 128 minutes do not contain
them. The [29 October handoff](../code/embeddings/encoders/causal_patch_bottleneck_mae/family/migration/HANDOFF_2026-10-29.md)
binds both artifacts and is the note for bringing the width-128 instance into
Cuwacunu. **V16 seed 946103/final2048 remains the research reference.** This
comparison does not rank against that seed, does not enter `decision.mjs`,
and leaves `Mityane_application_selection` null.

The follow-up search is closed. Temporal weights 0.3 and 1, including a repeat
of weight 1, did not beat seed 949102 under the rules declared beforehand. A
480-minute history made cross resolution measurable at R² −0.23302912246862673.
The [close note](../code/data/market/experiments/timeframe_published_bars_v1/CLOSE_2026-10-29.md)
records the table. Another weight or history length does not reopen it.

## Previous comparison: V19 EMA teacher — 23 October 2026

The [V19 comparison](../code/embeddings/evaluation/protocols/cpb_v19_teacher_v1/RESULTS.md)
is complete and **did not produce a better balanced encoder**. A training-only
predictor and moving-average teacher supplied a direct target for the learned
bottleneck. The served architecture/output shape, scattered masks, base objective and established
linear evaluation stayed unchanged. Three fresh candidates trained to 1,024
updates; the three completed scattered controls and their audited heads were
reused. All three new readouts and independent arithmetic audits passed.

The method has real scoped gains: volatility improved in **3/3 pairs**, with
median candidate-minus-control R² **+0.053184**; relation improved in **2/3**,
median **+0.109583**. Trend and activity each improved in only **1/3**, with
medians **−0.029321** and **−0.053271**. All four trend/activity continuation
conditions failed; volatility protection passed. No scoped linear record or
top-three group changed. **V16 seed946103/final2048 remains the reference.**

The [record](../code/embeddings/evaluation/records/CPB-v19.teacher-001.json)
separates **3,072 new updates** from **3,072 reused control updates** within
6,144 logical comparison updates. It also records the new teacher/predictor
work, three TRAIN fits, six training processes, twelve new target heads and
three new audits. Every new endpoint and admitted control identity was sealed
before new candidate heads; the reused control heads already existed.

The [V19 component](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v19/README.md)
keeps the complete teacher/predictor/objective/optimizer/resume method portable
and exports the student alone for inference. CUDA engineering passed 1,734
assertions across native36 and native15. Nonzero batch dispersion and norms
remain descriptive diagnostics, not proof against population collapse. This
known January development comparison measures no neural, timing,
reconstruction, fresh-population, full 2,048-update or application result.
The [delivery receipt](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v19/migration/delivery-2026-10-23.json)
pins the verified 106-file source package, including the full training method,
completed results and independent review; all 94 quoted includes resolve.
The [pre-commit preservation record](ENCODER_COMMIT_PRESERVATION_2026-10-23.json)
maps the exact V19 execution-source capsule before later Git byte-protection
rules and distinguishes the retained MTF/RPB bundles from subsequent shared
guidance edits. Original numerical receipts and bundles remain unchanged.

## Previous comparison: V18 contiguous training gaps — 22 October 2026

The [V18 comparison](../code/embeddings/evaluation/protocols/cpb_v18_span_v1/RESULTS.md)
is complete and **did not produce a better encoder**. Three matched seed pairs
compared contiguous gaps with scattered training masks at 1,024 updates per arm.
Architecture, objective, optimizer and the established January linear readouts
remained shared. All six readouts and independent arithmetic audits passed.

Candidate-minus-control median R² changes were trend **−0.221846**, activity
**−0.090620**, volatility **−0.057857** and relation **+0.015733**. Trend worsened
in all three pairs; activity improved in one and relation in two. Every
predeclared continuation condition failed, including the volatility protection.
No new scoped linear record or top-three group was established. **V16
seed946103/final2048 remains the research reference.**

The study reused the three original scattered-control 256-update states and
continued them without refitting; V18 candidates started fresh with matching
initialization and TRAIN statistics. It executed **5,376 new updates**, with
768 preserved control-prefix updates giving 6,144 logical updates. All six
final states were sealed before any readout. The
[compact record](../code/embeddings/evaluation/records/CPB-v18.span-001.json)
retains every arm, actual mask/support witnesses, work counts and limitations.
This is known January development evidence at a 1,024-update budget; it measures
no neural, timing, reconstruction, fresh-population or application result.

The [V18 component](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v18/README.md)
retains configurable construction, TRAIN fitting, the complete training method
and typed checkpoint/resume. This mask recipe is not the next research default;
the negative result remains useful evidence for the continuing search.
The [delivery receipt](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v18/migration/delivery-2026-10-22.json)
pins the verified 106-file source package, including the complete training
method, results and independent review; all 95 quoted includes resolve.

## Previous comparison: V17 temporal strength — 22 October 2026

The [V17 bounded study](../code/embeddings/evaluation/protocols/cpb_v17_balance_v1/RESULTS.md)
is complete and **failed its continuation criterion**. Three fresh native36
seed pairs compared temporal weights0.03 and0.1 through the corrected
configurable component at256 updates per arm. All1536 updates, twelve actual
fresh/resume processes, exact legacy target checks and six independent linear
readout audits passed. Candidate-minus-control median January R2 changes were
trend **-0.003029**, activity **-0.008995**, volatility **-0.002419** and relation
**+0.003186**. Activity worsened3/3; trend improved only1/3. Volatility stayed
within its -0.01 median tolerance, and relation improved2/3. These gains and
regressions remain visible in the [complete record](../code/embeddings/evaluation/records/CPB-v17.balance-001.json).

**V16 seed946103/final2048 remains the current research reference.** No new
scoped linear record or better overall encoder was established. This was an
early256-update comparison on consumed January development data; neural heads,
timing, fixed-query reconstruction and fresh confirmation were not measured. Its fresh0.1
controls share the new fitter/calibration/sampling recipe and do not reproduce
historical V16 trajectories. The TRAIN-only diagnostic found small temporal
raw gradients and positive global alignment, with localized backbone conflict;
it does not identify the cause of earlier regressions or AdamW update effects.

The [V17 component](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v17/README.md)
retains configurable construction, complete TRAIN fitting/training code,
weight-bound typed resume/export and a portable source package. This negative
research result does not undo the corrected component and application boundary.
The [source-delivery receipt](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v17/migration/delivery-2026-10-22.json)
records the verified93-file package and its independent manifest pin.

## Family deliverable and application training — 22 October 2026

Versioned encoder discovery continues alongside portable component development.
Mityane owns proposing, training and comparing architecture and learning-method
candidates to seek better representations. The generalized V16 interface makes
its method configurable and exportable; it does not conclude CPB research or
replace later numbered candidates. Numbered research versions retain their
scoped protocols and measured results; an export revision alone establishes
no new quality winner.

The [shared encoder Hero instructions](../code/embeddings/encoders/AGENTS.md)
apply across **Hx15–Hx18**, including external provider tools under their actual
upstream capabilities. All four workstreams deliver reusable encoder/provider
components; the CPB family path below is one implemented example, rather than
the definition of the other families' coverage.

Mityane discovers and compares **configurable encoder families**. Cuwacunu owns
substantial application training and final docking. A research checkpoint is
evidence for one trained instance; it does not define the family's supported
input geometry, latent width or model capacity. The V16 checkpoint reference
below remains a comparison reference, not a requirement to import its weights
when constructing a fresh application model.

Keep four objects separate: the family definition and its actual architectural
constraints; the configuration choosing supported geometry and capacity; the
experimental recipe specifying initialization, TRAIN preprocessing, objective,
optimizer and progress; and each trained instance binding those choices to its
own state. Derive dimensions and parameter counts from configuration. Loading
existing weights still requires their exact saved configuration; changing a
supported shape does not imply reshaping those weights or creating another
source version merely to bypass an experimental assertion.

The [CPB configurable family and training handoff](../code/embeddings/encoders/causal_patch_bottleneck_mae/family/README.md)
is the maintained destination for supported constraints, portable construction,
preprocessing fitting, training, checkpoint/resume and their focused engineering
evidence. It must distinguish implemented portability from missing application
work. Preserve the frozen V16 source, study recipe, checkpoint identities and
mixed verdict. This direction does not start another costly research study.

The [trained-reference compatibility record](../code/embeddings/encoders/causal_patch_bottleneck_mae/family/evidence/compatibility-2026-10-22.json)
now verifies the corrected interface against the selected trained V16 state,
including losses/gradients and matched short training/resume. The
[runnable fresh-training example](../code/embeddings/encoders/causal_patch_bottleneck_mae/family/examples/TRAINING.md)
adds configured input, fitting, periodic saves and byte-bound resume. These are
component engineering results; the checkpoint rankings and mixed study verdict
retain their original scope.

The [source migration package](../code/embeddings/encoders/causal_patch_bottleneck_mae/family/migration/README.md)
exports the training method with the network: complete numerical dependencies,
fitting/masking/objective/optimizer/resume code, runnable example, method notes,
research rationale, dependencies/licenses and a verifiable source manifest.
Its Cuwacunu mapping respects the existing application training pipeline;
example, historical study and consumer recipes remain distinct.

The [component ownership contract](../code/embeddings/encoders/COMPONENT_DELIVERY.md)
incorporates a direct Cuwacunu review of the CPB delivery: Mityane delivers
reusable learning semantics and evidence; applications bind their data, complete
recipes, execution and trusted artifacts. That review found no missing numerical
delivery mechanism blocking use through Cuwacunu's existing training pipeline.

**Hx15 MTF remains a retained reference, with no current downstream application
use.** Its [portable family layer](../code/embeddings/encoders/mtf_jepa_mae_vicreg/family/README.md)
and [22 October engineering record](../code/embeddings/encoders/mtf_jepa_mae_vicreg/family/verification/RESULTS_2026-10-22.md)
now cover configurable construction, identity preparation without a fitted
scaler, training/resume, inference and source delivery. Two configurations and
two sampling recipes passed primary and isolated CPU/F32 engineering checks.
These checks establish the recorded component behavior; they add no scientific
quality ranking or downstream utility claim. Historical MTF-v1 checkpoints and
their unknown preparation associations retain their original scope.

**Hx16 RPB's portable training delivery is complete.** Its
[maintained family guide](../code/embeddings/encoders/raw_patch_bottleneck_mae/family/README.md)
and [runnable C++ example](../code/embeddings/encoders/raw_patch_bottleneck_mae/family/examples/TRAINING.md)
cover configurable fresh-v22 construction, streamed TRAIN fitting, masked
updates, typed continuation and inference/embedding export. Two configurations
passed actual CUDA/F32 engineering checks with exact continuation in the named
runtime. The [source-delivery receipt](../code/embeddings/encoders/raw_patch_bottleneck_mae/family/migration/delivery-2026-10-22.json)
binds the 117-file package, isolated recipient acceptance and documentation-only
final amendment. These checks establish component behavior; the historical
native36/V30 results and causal-eligibility limits below remain unchanged.
Applications own substantial training and final docking.

## Leading encoders and their roles

**Current research reference: V16 candidate seed 946103, final checkpoint 2048,
native36.** Use this checkpoint as the default starting reference for continued
encoder research. The owner requested this advance on **22 October 2026** after
the completed study. The family-owned [reference record](../code/embeddings/encoders/causal_patch_bottleneck_mae/research-reference.json)
binds the exact checkpoint and TRAIN preprocessing, reasons and limitations.

It leads both January trend readouts: linear R² **0.925246** and neural-mean R²
**0.693462**. Its March linear trend and activity improve its matched fresh V4
control by **+0.047276** and **+0.244398 R²**. V16 seed 946102 separately leads
neural activity, and seed 946101 leads market-scaled, deleted-view linear timing.
These are three checkpoints; the default does not combine their scores.

This is an explicit retrospective research choice. The prospective study remains
**mixed**, with failed volatility/timing protections and reconstruction regressions.
January and March are consumed evidence; future claims need fresh confirmation.
V4 remains a historical benchmark and the known-validation reconstruction leader;
V3 remains the clean baseline. Runtime/application adoption is a separate decision.

| Role | Encoder | Recorded achievement and remaining limit |
| --- | --- | --- |
| Current research reference | **V16 candidate 946103, final2048/native36** | Both January trend records; positive matched March trend/activity gains. Chosen on 22 October for continued research. The study remains mixed; this checkpoint passes 0/8 strongest timing conditions and is not the leader in every task. |
| Timeframe development instance | **CPB-family width 128, seed 949102, 2048 updates, native36** | Stronger of the two fresh 1m/1h/1d log-return arms on 29 October. Same-window volatility R² 0.7682422126010569 and trend R² 0.22475307149870904, activity saturated, cross resolution unmeasured. One seed. Offered to Cuwacunu for docking. It does not replace the research reference. |
| Historical clean-causal breakthrough and comparator | **CPB-v4.market-01** | **19/22 original conditions**, all eight TEMPO-5 retention checks, major trend/relation gains and the best saved known-validation reconstruction. Linear volatility, linear activity and TRAIN query MAE failed their floors, withholding formal promotion. |
| Historical balanced research leader | **RPB-v24.market-01.alt-01, native36** | **8/9 market conditions plus all four primary frozen TEMPO-5 transfer comparisons**, at full coverage using the same market-trained checkpoint. Later diagnosis limits its clean causal eligibility; its measured success and historical leading designation remain preserved. |
| Stable clean research baseline | **CPB-v3.market-01** | First genuinely fresh causal-training reference; still leads linear volatility. Baseline status is not an overall quality ranking or a project-wide Mityane selection. |
| Later historical market success | **RPB-v30.market-01** | **9/9 market conditions and 8/8 strict native36 floors**. Independent synthetic transfer remained pending, so the historical balanced designation stayed with native36. Its trained ancestry is outside the clean causal comparison. |
| Latest catalogued historical instance | **CPB-v14.market-01** | Full trust and coverage; 3/22 strengthened conditions and zero new protected records in that run. Its exact recipe is closed; independent metadata review and compatible application are complete. The separate six-run V15 and V16 studies have their own identities. |
| Latest completed controlled study | **CPB-v16.temporal-001**, three fresh V4/V16 pairs | Four new January candidate record scopes. Both March primary targets pass: trend median R² gain +0.047276 (2/3 pairs), activity +0.193252 (3/3). The complete verdict is mixed because volatility tolerance and strongest timing retention fail. |

Foreground V16 when naming the current research reference. Preserve V4's
historical breakthrough and the other task leaders as comparators. No clean
candidate has combined every protected strength or received overall protocol
promotion. A chosen research reference, scoped records and protocol promotion
are separate facts. Historical 19/22, 8/22 and 3/22 counts use different
prospectively frozen floors; they cannot be ranked as identical tests.

Evidence: [V4 review](../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V4_REVIEW.md),
[native36 transfer milestone](../.archive/embeddings/pre-cleanup-2026-10-18/doc/MILESTONE_2026-10-14_NATIVE36_TRANSFER.md),
[historical RPB ledger](../.archive/embeddings/pre-cleanup-2026-10-18/doc/ENCODER_SEARCH_LEDGER.md)
and [V14 results](../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V14_RESULTS.md).

## Historical V3–V14 measurement leaders

The tables below preserve the pre-V15 clean-causal final-checkpoint leaders used
as historical comparators. Use the [live ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md)
for current standings, including fresh V4 controls and V15/V16 observations. These
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
| CPB-v15 with fresh V4 controls | Completed three-seed objective comparison at fixed V4 native36. February linear/neural activity and query reconstruction improve in all three pairs. Six scoped January records include two V15 and four fresh-control records. The prospective verdict is mixed: volatility, trend tolerance and timing retention fail. V4 remained the reference at the 19 October closeout; the 22 October reference choice advances to V16. |
| CPB-v16 with fresh V4 controls | V4 Gaussian NLL plus weight0.1 hidden temporal Huber at lags1/4/16, native36. Four January candidate records and both March primary-target gains; mixed full screen because volatility and strongest timing protections fail. Known-validation query MAE/NLL worsen in all three pairs. |

**V3 through V16 all start from fresh weights, optimizer state and TRAIN
statistics.** Source reuse is branched design ancestry: V8 uses V6 source, V11
uses V9, V13 uses V11, V14 uses V13, and V15/V16 use V4. V15/V16 retain native36;
V5–V14 use native40. V1/V2 retain pretrained ancestry. V16's interrupted sixth
trajectory later resumed its own saved state, without importing another run.
The [addendum](EMBEDDINGS_HANDOFF_ADDENDUM_2026-10-18.md) and frozen per-version
reports preserve the distinctions and failures.

## Current work and continuation

**The V16 temporal comparison is complete — 21 October 2026.** The
[complete results](../code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/RESULTS.md),
[compact study record](../code/embeddings/evaluation/records/CPB-v16.temporal-001.json)
and [18-check independent review](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/final-study-review-001.json)
bind all six trajectories, eighteen phase receipts, twelve independent numerical
audits and twelve checkpoint causality proofs. All declared coverage passes.

**Four new January candidate records lead their scopes among 24 measured
checkpoints.** Each is native36, final logical update2048; the successes span
three different seeds rather than one selected checkpoint.

| Record scope | V16 candidate seed | Saved unrounded result | Former leader |
| --- | --- | --- | --- |
| Linear trend R² | 946103 | 0.9252456611897139 | Historical V4, 0.9072134297036245 |
| Neural trend mean R² | 946103 | 0.6934616212783601 | Historical V4, 0.5698752307038314 |
| Neural activity mean R² | 946102 | 0.8156121912494054 | V15-study V4 control945101, 0.7924196342764943 |
| Market-TRAIN scaler / deleted-view linear TEMPO | 946101 | 0.9752604166666666 (97.526042%) | V6, 0.97265625 |

Fresh V4 controls946101 and946103 also exceed historical V4's neural trend mean
with **0.650729417976703** and **0.621373408240522**. Candidate946103 subsequently
leads that scope. These control successes and all former leaders remain visible
in the [live ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md).
The V15 study retains five current first-place scopes; its six original record
achievements remain dated history. January is repeatedly used development evidence.

**Both predeclared March primary targets pass.** Candidate minus matched fresh
V4 control linear R² gives the following saved three-pair screen:

| March outcome | Median difference | Improving pairs | Declared role |
| --- | --- | --- | --- |
| Trend | +0.04727628106997395 | 2/3 | Primary target: pass |
| Activity | +0.1932523053809676 | 3/3 | Primary target: pass |
| Volatility | -0.03349967451837166 | 1/3 | Protection tolerance -0.01: fail |

The full prospective verdict is **mixed**. No candidate retains all eight frozen
strongest January timing conditions: candidate946101 passes1/8, the others0/8;
the fresh controls also pass0/8. Several paired timing measures improve, including
the new deleted-view linear record, without satisfying that absolute guard.
These counts use stronger floors than historical V4's original19/22 comparison.

March neural trend and activity means each improve in2/3 pairs. Linear relation
improves in3/3 but remains negative; neural relation worsens in2/3. Neural activity
means remain negative for946101/946102, and one946103 head is still negative.
January known-validation query MAE and NLL worsen in3/3 pairs. March query MAE
improves only for946102, while NLL improves for946101/946103; those scores are
descriptive and add no gate. Readout, reconstruction and timing remain separate.

The [prospective protocol](../code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/PROTOCOL.md)
keeps V4 native36/Gaussian NLL and adds weight0.1 Huber differences at raw minute
lags1/4/16, using only original hidden-target endpoints and fixed thirds including
unsupported lags. All preprocessing and heads are TRAIN-fitted. The
[all-six development seal](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/development-seal.json)
passed before March scoring. February was already consumed by V15; March is now
consumed evidence for future recipe choices. These are past-window readouts and
masked reconstruction, without forecasting or application-selection evidence.

One trajectory, candidate946103, recovered its own typed512 state under the
[additive recovery declaration](../code/embeddings/evaluation/protocols/cpb_v16_temporal_recovery_v1/RECOVERY.md).
The [publication receipt](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery001/publication.json)
binds the distinct recovery executable, exact available discarded-tail rows and
torn-fragment matches, two reconciled CSV identities and preserved original aliases.
The other five training trajectories remain unchanged. Total physical attempts
and repeated work, complete costs and original parameter-prefix counters remain
unknown; recorded recovery learning/cost evidence covers only updates513–2048.
The [execution log](../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovered-study-execution-001.log)
finishes successfully. C++, Node and shell ran in the managed container, without Python.

**V16 candidate 946103 is the current research reference from 22 October.** V4 is
the historical benchmark and V3 the baseline. The measured-instance catalog
remains at V14; the separate [reference record](../code/embeddings/encoders/causal_patch_bottleneck_mae/research-reference.json)
records the owner's retrospective choice without rewriting the mixed study or
its original selection flags. Current work states and acceptance criteria remain
in [Hx17](../web/dashboard/heroes/17-encoder-cpb.json).

**29 October 2026 — timeframe development instance.** Width 128, seed 949102,
is the encoder to bring into Cuwacunu. The
[handoff](../code/embeddings/encoders/causal_patch_bottleneck_mae/family/migration/HANDOFF_2026-10-29.md)
binds its settings, inference archive, training state, preprocessing, and the
matched width-64 control. The follow-up search is
[closed](../code/data/market/experiments/timeframe_published_bars_v1/CLOSE_2026-10-29.md)
on that instance. The research reference above is unchanged.

The next reference change uses the [family screen](../code/embeddings/evaluation/protocols/cpb_family_screen_v1/POLICY.md).
Development cannot promote. January, February and March 2024 are consumed.
[`current.json`](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/current.json)
and [`index.json`](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/index.json)
stay frozen. [Version roles](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/ROLES.md)
name the one live pointer.

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
V3 remains the clean research baseline. That V14 metadata closeout made no model
selection; the later V16 research-reference choice is recorded above.
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

At its 19 October closeout, V15 established **six new scoped January highs**, retained in the
[live ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md).
V15 sets TRAIN query MAE **0.24003670546837294** at seed945102 and synthetic-scaler
intact linear timing **99.609375%** at seed945101. Fresh V4 controls separately
set neural activity, TRAIN NLL and two neural timing records. V16 later supersedes
the activity lead; the other five V15-study leading scopes remain. All three V15
candidates improve paired January TRAIN/known-validation query MAE and
known-validation Gaussian NLL, while January linear trend falls in all three.
V4 retained the research-reference role at the V15 closeout and still holds both
known-validation reconstruction records. The later 22 October reference choice
advances to V16 while preserving those measurements.

The [reviewed reporting card](../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-inputs-001.json)
binds the complete machine report. A preserved reporting interruption and a
separately reviewed successor reporting generation are documented in RESULTS
and the compact record. The successor removes identical duplicate final checks;
scientific calculations are unchanged, every distinct final binding is freshly
verified and no training, head fit or completed experiment was replayed. The
local [execution record](../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/execution-start-001.json)
retains the original admitted run order and source identities. The new work used
C++, Node and shell in the managed Linux container, without Python.

The frozen [V15 source declaration](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v15/README.md),
its [version descriptor](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/CPB-v15.json),
the [V16 source declaration](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v16/README.md)
and its [version descriptor](../code/embeddings/encoders/causal_patch_bottleneck_mae/versions/CPB-v16.json)
retain their preparation-stage wording because execution receipts bind those bytes.
Current progress and measurements belong in this guide, the live ranking and Hx17.
The catalog remains at V14; these six-run studies record each arm and seed
separately and perform no catalog
application or prospective seed selection. The separate 22 October reference
record documents the subsequent retrospective research choice.

January remains development evidence. Calendar-selected February was scored with
the sealed models and original TRAIN-fitted heads, with no February refit or
tuning. February is now consumed confirmation evidence: a recipe chosen after
seeing these outcomes requires a new confirmation population. The 22 original
protection predicates and all earlier leaders remain preserved. The six-run
study establishes no automatic model promotion or application selection.

The [saved V14 TRAIN-only study](../code/embeddings/evaluation/protocols/cpb_v14_train_diagnosis_v1/PROPOSED_PROTOCOL.md)
remains reviewed preparation, unexecuted and deferred by this V4-based choice.
Two saved batches cannot establish held-out causation. Current priorities,
acceptance criteria and milestone states remain in
[Hx17](../web/dashboard/heroes/17-encoder-cpb.json).

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
family-owned [reference record](../code/embeddings/encoders/causal_patch_bottleneck_mae/research-reference.json)
owns the exact current research default and its dated decision basis. The
family-owned [live ranking](../code/embeddings/encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md)
owns current checkpoint standings and their dated changes. The
[29 October handoff](../code/embeddings/encoders/causal_patch_bottleneck_mae/family/migration/HANDOFF_2026-10-29.md)
owns the timeframe development instance offered to Cuwacunu. Update the ranking
from bound results and align the short family/dashboard summaries in the same
change.
Preserve earlier leaders, ties and original protocol scopes; retain the tables
above as V3–V14 historical comparators. Keep migration details in the handoff
addendum and detailed evidence in its owning records. Work states and priorities
remain in the Hero JSONs, with the documented refresh after their edits.
