# Causal patch bottleneck MAE

Copyright (c) 2026 Waajacu. All rights reserved. Preserve the inherited notices
inside the copied implementation.

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument.
Each channel is one time resolution of that instrument. The channel id is the
torch kline-interval ordinal. Different instruments are different rows. The
contract is
[timeframe_channel.h](../include/embedding/encoders/timeframe_channel.h).
Historical checkpoints that stored `{0,1,2}` keep that withdrawn instrument
binding, and their scores stay scores of that binding.

The first fresh comparison on that contract is the
[29 October handoff](family/migration/HANDOFF_2026-10-29.md). Width 128,
seed 949102, 2048 updates, is the shared-clock instance. Width 64, seed 949101,
is its matched control. The packed-channel instance to bring into Cuwacunu is
the [31 October handoff](family/migration/HANDOFF_2026-10-31.md), seed 949501.
Further Mityane training on that packed layout is closed.
V16 seed 946103 remains the research reference. The new scores are same-window
timeframe probes and are not a ranking against that seed. The follow-up search
is
[closed](../../../data/market/experiments/timeframe_published_bars_v1/CLOSE_2026-10-29.md)
on seed 949102.

## Continue versioned CPB research

The CPB workstream continues to propose and compare new encoder versions in
search of better representations. V16 seed946103/final2048/native36 is the
current research reference. Generalizing V16 for configurable construction,
training and export supplies a reusable foundation for that continuing search.

The latest candidate, [V19](versions/v19/README.md), added a training-only
predictor and EMA teacher on learned coordinates, keeping served architecture,
scattered masks, base objective and evaluation unchanged. Its
[audited comparison](../../evaluation/protocols/cpb_v19_teacher_v1/RESULTS.md)
completed three new 1,024-update candidates against preserved controls.
Volatility improved in 3/3 pairs (median ΔR² +0.053184) and relation in 2/3 (+0.109583).
Trend/activity medians fell −0.029321/−0.053271, each improving in only 1/3, so all
four primary conditions failed while volatility protection passed. These
scoped gains establish no new linear record, top-three group or better balanced
encoder. **V16 remains the research reference.** The
[record](../../evaluation/records/CPB-v19.teacher-001.json) counts 3,072 new and
3,072 reused control updates separately. Complete teacher/predictor training,
optimizer and typed resume are part of the portable method; inference exports
only the unchanged-size student. The
[verified delivery receipt](versions/v19/migration/delivery-2026-10-23.json)
pins its 106-file source bundle, including complete training-method code,
results and independent review; verification through the delivered copy passed.

The preceding candidate, [V18](versions/v18/README.md), compared contiguous training
gaps with scattered masks in three matched pairs at 1,024 updates. Its
[audited result](../../evaluation/protocols/cpb_v18_span_v1/RESULTS.md) is negative:
median candidate-minus-control R² changes were trend −0.221846, activity
−0.090620, volatility −0.057857 and relation +0.015733. Trend worsened in all
three pairs and every continuation condition failed. All six readouts/audits
passed; no scoped linear record or top-three group changed. **V16 remains the
research reference.** Controls reused their own completed 256-update prefixes;
5,376 new updates plus 768 preserved updates give 6,144 logical updates. The
[record](../../evaluation/records/CPB-v18.span-001.json) preserves this ancestry,
all mask witnesses and the known-development/1,024-budget limitations.
The [V18 migration guide](versions/v18/migration/README.md) and
[verified delivery receipt](versions/v18/migration/delivery-2026-10-22.json)
carry its complete training method and negative evidence in a 106-file source
package; verification through the delivered copy passed.

The first candidate on this foundation, [V17](versions/v17/README.md), compared
temporal weights0.03 and0.1 in three fresh matched256-update pairs. Its
[completed screen](../../evaluation/protocols/cpb_v17_balance_v1/RESULTS.md)
failed continuation: median trend/activity R2 changes were-0.003029/-0.008995,
with activity worse3/3. Volatility stayed within its tolerance and relation
improved2/3. All six trajectories and arithmetic audits passed; this is a valid
negative research result. V16 remains the reference, and no earlier record is
displaced. The V17 training/resume/export machinery remains reusable.

The [93-file delivery receipt](versions/v17/migration/delivery-2026-10-22.json)
pins the complete training-method source package, negative study evidence and
passed verification through the delivered copy.

Later candidates can investigate meaningful architecture or
learning-method changes under explicit protocols. The portable `CPB-family` interface is
an engineering identity; it neither replaces the numbered research sequence
nor establishes a new quality winner. Every research result keeps its actual
configuration, recipe, ancestry, trained instance and limitations. The export
work creates no new measured CPB candidate by itself.

## Family deliverable and training ownership — 22 October 2026

Read the [shared encoder Hero instructions](../AGENTS.md), which apply to
Hx15–Hx18. This family's portable APIs implement part of that common tool
direction; its measured checkpoint reference has a separate evidence role.

This research lane discovers and compares a **configurable encoder family**.
Cuwacunu owns substantial application training and final docking. Deliver the
family's constructible architecture, supported configuration and portable
training/checkpoint path; a pretrained checkpoint supplies research evidence
for one instance and must not determine the only supported shape.

Separate the family definition, geometry/capacity configuration, experimental
recipe and trained instance. Supported changes in input geometry, latent width
or capacity should use configuration and derived dimensions/counts. Validate
actual architectural constraints rather than copying a benchmark's geometry
or exact parameter totals into the family's fresh factory. Existing checkpoints
remain bound to their saved configurations; this does not authorize automatic
weight reshaping.

The [configurable family and training handoff](family/README.md) is the maintained
destination for portable initialization, TRAIN preprocessing fitting, training
objective, optimizer, checkpoint/resume, dependencies, constraints and focused
alternative-shape evidence. Record what is implemented and what application
docking still needs. Preserve frozen V16 source and research records unchanged;
the research reference below remains a measured comparison instance, not the
architecture's supported-shape contract. This clarification authorizes no new
expensive study or publication.

The maintained [family handoff](family/README.md#trained-reference-compatibility-and-runnable-example)
now includes trained-reference equivalence and a runnable fresh/resume CLI.
It preserves V16's numerical behavior at the reference configuration while
keeping fresh configured instances separate from the saved research checkpoint.
Application data docking, scale and utility remain separate work.

The [source migration package](family/migration/README.md) carries the network
and its training method together: complete numerical dependencies, fresh/resume
example, preprocessing/objective rules, research context, licenses and a
verifiable file manifest. Its Cuwacunu mapping preserves that application's
own training recipe and artifact publication.

The [22 October delivery record](family/migration/delivery-2026-10-22.json)
retains the independently verified 76-file bundle manifest pin, passed 20-test
export checks, and the read-only Cuwacunu source comparison. No training or
checkpoint ranking changed during packaging.

The subsequent [Cuwacunu consumer review](family/migration/CONSUMER_REVIEW_2026-10-22.md)
confirmed sufficient numerical delivery and clarified producer/application
ownership. The [reviewed delivery record](family/migration/delivery-2026-10-22-consumer-review.json)
pins the current 78-file source bundle containing that review and the
[shared component contract](../COMPONENT_DELIVERY.md). The original 76-file
bundle and all numerical evidence remain preserved.

## Active ranking and agent instructions

Maintain the live [`causal_patch_bottleneck_mae.ranking.md`](causal_patch_bottleneck_mae.ranking.md) in this
family directory. Read it and the authoritative
[`research-reference.json`](research-reference.json) before answering which
encoder is best or choosing comparison references. The record names the current
research default; the ranking owns scoped quality standings. Dated notes below
retain their historical context. Hero JSON owns work states.

1. **Update the ranking with new evidence.** After an admitted measurement and
   its required trust, coverage and arithmetic review pass, update its scoped
   ranking and the dated change log. List audited development records from an
   ongoing study with a clear pending-study marker; wait for the complete
   protocol result before declaring its overall verdict.
2. **Name the exact model.** Use the measured instance or study/arm/encoder seed,
   native width/layout and checkpoint identity. Link the hash-bound card or
   receipt containing source, checkpoint SHA256, TRAIN fits, masks, data split,
   population/horizon, fixed heads, seeds, coverage and audit. A version name
   alone does not identify a trained encoder.
3. **Use the explicit current research reference.** The default is V16 candidate
   seed 946103, final logical update 2048/native36, selected on 22 October 2026.
   Keep that choice, V3's baseline, V4's historical breakthrough and task leaders
   as distinct roles. Record scoped gains even when a protection fails. Change
   the reference through an explicit evidence-based assessment and dated record;
   retain the former reference, its achievements and the reasons for the change.
   A research default does not establish universal quality or promotion under
   the frozen protocol. The next reference change uses
   [`cpb_family_screen_v1`](../../evaluation/protocols/cpb_family_screen_v1/POLICY.md):
   a development population cannot promote, January, February and March 2024 are
   consumed, and only a sealed later window whose training-state `recipe_identity`
   begins `cpb-family-train-example-v1;` at the 2048-update budget can promote. [`versions/current.json`](versions/current.json)
   and [`versions/index.json`](versions/index.json) are frozen. The research
   default remains [`research-reference.json`](research-reference.json). See
   [version roles](versions/ROLES.md).
4. **Rank comparable scores separately.** Keep TRAIN, known validation, fresh
   confirmation, tasks, heads and TEMPO scaler/views separate. Sort R²/accuracy
   descending and MAE/NLL ascending. Use unrounded values, preserve ties
   under the ranking's stated reporting tolerance, and keep frozen predicates
   unchanged. Different historical pass counts are not a common ranking score;
   introduce no undeclared weighted overall score.
5. **Retain the whole study.** Show every admitted encoder seed/arm, gains,
   regressions, support failures and unresolved outcomes. Head repetitions are
   not independent encoder trials. A record-setting seed is not a prospectively
   selected study winner. A later owner-directed reference choice must be dated
   and described as retrospective, preserving the original study's flags and
   verdict. Keep initial checkpoints separate from final outcome ranks.
6. **Preserve ancestry and scope.** Keep pretrained/historically ineligible
   references separate from clean-causal comparisons. Record width/head-seed
   confounds. Reconstruction, readout, timing and forecasting utility remain
   separate claims.
7. **Keep summaries consistent.** Update the current quality summary and README
   headline when the ranking changes; link to the ranking instead of maintaining
   conflicting live tables. Preserve dated historical comparators. Update Hero
   work states through their existing schema and refresh procedure only when
   that separate work-state change is in scope.
8. **Preserve evidence boundaries.** Update this live Markdown from saved
   results; do not replay experiments to populate it. Never rewrite archived
   evidence, frozen version/source declarations or protocols as live status.
   Ranking maintenance alone authorizes no training, catalog application,
   runtime deployment, publication or Mityane model selection.

## Current research reference — 22 October 2026

**Use V16 candidate seed 946103, final logical update 2048/native36, as the
reference for continued encoder research.** The
[authoritative reference record](research-reference.json) binds its exact
checkpoint, TRAIN input scaler and bank standardizer. It leads both January
trend readouts and has positive March linear/neural trend and activity scores.
This is the owner's explicit retrospective choice after the completed study;
the study's original mixed verdict and selection flags remain unchanged.

V16 seed 946102 retains the January neural-activity record, and seed 946101
retains the market-scaled/deleted-view linear TEMPO record. Their scores belong
to those separate checkpoints. V4 remains the historical broad clean-causal
breakthrough and current known-validation reconstruction leader; retain it as
a comparator for those strengths. V3 remains the clean baseline. The reference
choice changes no runtime default, deployment or Mityane application selection.

## Completed V16 temporal comparison — 21 October 2026

V16 delivers **four new January candidate record scopes** across three seeds:
seed946103 linear trend R² **0.925246** and neural trend mean **0.693462**,
seed946102 neural activity mean **0.815612**, and seed946101 market-TRAIN-scaled,
deleted-view linear TEMPO accuracy **97.526042%**. All are native36/final logical
update2048. Fresh V4 controls946101/946103 also surpass historical V4's neural
trend mean; candidate946103 subsequently leads that scope. The
[live ranking](causal_patch_bottleneck_mae.ranking.md) preserves each checkpoint,
former leader and dated success alongside all 24 measured checkpoints.

**Both declared March primary targets pass:** linear trend median gain
**+0.047276 R²**, improving2/3 pairs, and linear activity **+0.193252**, improving3/3.
The full prospective verdict is **mixed**: linear volatility median **-0.033500**
fails its -0.01 tolerance, and no candidate retains all eight strongest January
timing conditions (1/8 for946101,0/8 for946102/946103; fresh controls also0/8).
Several paired timing views improve without satisfying those absolute floors.
Historical V4's original19/22 used earlier floors and is a distinct comparison.

March neural trend/activity means improve2/3 pairs. Linear relation improves3/3
but remains negative; neural relation worsens2/3. Neural activity means remain
negative for946101/946102. January known-validation query MAE and NLL worsen3/3.
March query MAE improves only946102, while NLL improves946101/946103; these are
descriptive scores. The current reference retains these failures and tradeoffs.

The [complete results](../../evaluation/protocols/cpb_v16_temporal_v1/RESULTS.md)
and [compact study record](../../evaluation/records/CPB-v16.temporal-001.json) bind
six completed trajectories, all eighteen phase receipts, twelve independent
arithmetic audits and twelve saved-checkpoint causal proofs. Full coverage passes.
The [prospective protocol](../../evaluation/protocols/cpb_v16_temporal_v1/PROTOCOL.md)
adds weight0.1 Huber differences at lags1/4/16 to V4 Gaussian NLL without changing
native36 capacity, original hidden targets, heads or TRAIN-only preprocessing.
The [development seal](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/development-seal.json)
passed before March scoring. January remains development evidence; February and
March are consumed evidence for later recipe choices. The original study made
no checkpoint or Mityane application selection. The separate 22 October research
reference choice is recorded above; current work states remain in Hx17.

Candidate946103 restored its own typed512 state under the
[recovery declaration](../../evaluation/protocols/cpb_v16_temporal_recovery_v1/RECOVERY.md)
and [publication receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery001/publication.json).
Exact available discarded-tail rows/fragments match; the distinct executable,
two reconciled CSV identities and retained originals are explicit. The other five
training trajectories remain unchanged. Total physical attempts/repeats, complete
costs and prefix parameter counters remain unknown; measured recovery evidence
covers only logical updates513–2048. The frozen V16 source declaration retains
its preparation-stage wording because execution receipts bind those bytes.

## Completed V15 comparison — 19 October 2026

The three-seed paired V4/V15 native36 study is complete and valid, with a
**mixed** prospective verdict. V15 improves February linear/neural activity and
descriptive query MAE/NLL in all three pairs. It sets two scoped January records;
fresh V4 controls set four more. The [live ranking](causal_patch_bottleneck_mae.ranking.md)
preserves all six runs and earlier leaders. Volatility, trend tolerance and
timing retention fail the prospective promising rule; **V4 remained the
broad clean reference at that 19 October closeout**. See the [complete results](../../evaluation/protocols/cpb_v15_objective_v1/RESULTS.md)
and [bound study record](../../evaluation/records/CPB-v15.objective-001.json).

## Research references — 18 October 2026

**CPB-v4.market-01 established the broad clean-causal breakthrough.** It passed
19/22 original conditions against V3, retained all eight hard-timing checks
(seven improvements and one tie), substantially improved trend/relation, and
achieved the best saved known-VALIDATION reconstruction MAE and Gaussian NLL.
Its failures were linear volatility, linear activity and TRAIN query MAE; those
withheld formal promotion without cancelling its successes. Read the preserved
[V4 results](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V4_RESULTS.md)
and [review](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V4_REVIEW.md).

| Reference role | Encoder and meaning |
| --- | --- |
| Stable clean baseline | **CPB-v3.market-01**; first fresh-seeded causal baseline, not a claim of best overall quality or a project-wide Mityane selection |
| Broad clean breakthrough | **CPB-v4.market-01**; the broad success above, including the protected linear/neural trend and known-VALIDATION reconstruction leaders |
| Later market leaders | **V8** neural volatility; **V11** linear activity; **V13** neural activity and neural relation; **V6** linear relation. V3 retains linear volatility. Timing leaders depend on scaler, head and deletion condition. |
| Latest catalogued historical instance | **CPB-v14.market-01**; trusted fresh causal run, full coverage, 3/22 strengthened conditions and no new protected record in that run; independent metadata review and compatible application complete |

In this 18 October snapshot, V7 held the TRAIN reconstruction MAE/NLL records.
The [live ranking](causal_patch_bottleneck_mae.ranking.md) records the later
V15/V16-study observations, including fresh-control gains. TRAIN reconstruction,
known-VALIDATION reconstruction, fixed-readout quality and forecasting utility
are separate measures. No single clean encoder has combined every protected
strength or received overall promotion. Acceptance floors changed over time, so
V4's 19/22 and V14's 3/22 are not a common ranking scale.

The [current research guide](../../../../doc/EMBEDDINGS.md) is the maintained
entry for aims, the comparison map, evidence and current work state. The
[post-migration addendum](../../../../doc/EMBEDDINGS_HANDOFF_ADDENDUM_2026-10-18.md)
clarifies leadership, checkpoint locations and the exact pause boundary.
[Saved V14 results](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V14_RESULTS.md)
contain all per-task tables; the
[archived family guide](../../../../.archive/embeddings/pre-cleanup-2026-10-18/code/encoders/causal_patch_bottleneck_mae/README.md)
preserves the dated V1–V14 continuation chronology. Its former “current” and
“next” notes describe those historical stages.

## Aim, ancestry and pause boundary

This lane seeks a compact causal representation of observed multichannel
history that retains useful temporal structure and relationships across tasks,
readout heads and missingness conditions. Known validation has repeatedly guided
development; these past-window measurements do not establish fresh-period or
forecasting utility. The project-wide scope is defined by the
[Operations over Representations charter](../../../../doc/operations_over_representations_design_spec_v0.1.0.md).

V1 imports RPB-v30 weights; V2 adapts V1 and retains pretrained ancestry. V3–V16
start from fresh learned weights, optimizer state and TRAIN statistics. Later
versions branch source designs; they do not accumulate earlier trained weights.
V3/V4 native36 versus V5–V14 native40 remains a capacity/head confound. V15/V16
branch V4 source at native36, making their objective comparisons independent of
the older Huber experiments' width change.

The user resumed **V14 metadata review/application/preservation** on 18 October
2026. The saved proposals passed independent review, and the maintained
[catalog](versions/current.json) now records V14 with its
[measured instance](versions/instances/CPB-v14.market-01.json) and
[closeout summary](../../evaluation/records/CPB-v14.market-01.closeout.json).
The [actual application receipt](../../../../.temp/embeddings/closeout/cpb-v14/attempt-001/application.json)
records the reviewed writes; final integrity sealing is recorded separately at
`.temp/embeddings/closeout/cpb-v14/attempt-001/preservation.json`.
Original scientific evidence and proposals remain unchanged. On 19 October the
owner authorized the [V15 objective comparison](../../evaluation/protocols/cpb_v15_objective_v1/PROTOCOL.md):
three fresh V4/V15 pairs, native36, Huber1 mean supervision and detached variance
calibration, with a separate February confirmation population. Source, CUDA,
target-parity and all six development/audit checks passed. The six final models
and TRAIN-fitted heads were sealed before February. All six confirmation phases
and their independent arithmetic audits passed. The final report is complete;
the prospective verdict is mixed for the reasons recorded above. February is
now consumed confirmation evidence, so a recipe chosen after these results
needs a new confirmation population. The [live ranking](causal_patch_bottleneck_mae.ranking.md)
retains all new scoped measurements. No Mityane model selection is established. The
[saved V14 TRAIN-only study](../../evaluation/protocols/cpb_v14_train_diagnosis_v1/PROPOSED_PROTOCOL.md)
remains reviewed, unexecuted preparation, deferred by this choice.

## Physical source ownership

Each CPB version owns a complete implementation under `versions/vN/`, including
its `include/`, `config/`, `tests/`, make fragment and provenance. For example,
[V4](versions/v4/) and [V14](versions/v14/) are current physical source roots;
the V1 public entry is
[`versions/v1/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h`](versions/v1/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h),
in namespace `embedding::encoders::causal_patch_bottleneck_mae::v1`, with settings
in [`versions/v1/config/v1.conf`](versions/v1/config/v1.conf).

Owned private headers and parser utilities do not forward encoder execution
into the raw family, another causal version or an external checkout. Torch and
the C++ standard library are external dependencies. There is no mutable
family-level encoder implementation shared between evaluated CPB versions.
Maintained relocated source has a new path-bound identity; historical receipts
continue to bind archived originals or exact frozen capsules.

Use the root launcher, task runner and
[current environment guide](../../../../doc/ENVIRONMENT.md) for container
orchestration. Frozen source/protocol paths remain evidence identities, not
current lifecycle instructions.

## Cutoff contract

The public `Encoder::encode_at` and `trace_at` paths take raw CPU float64 history,
boolean observation masks, semantic channel IDs, a timestamp grid and explicit
cutoffs. They discard later values **and masks** before observed-finiteness
checks, rounding, normalization, token packing, support counts or bank statistics.
Each output `z_t` uses the last at most H observations ending at its own cutoff.
Unavailable earlier slots are left-padded with hidden zeros. Temporal attention
uses original patch endpoints and blocks later keys while retaining padding
masks; channel mixing stays at the same patch time.

Every returned timestamp must use its own cutoff path. A final global vector
must not be broadcast to earlier times. The public API has no unguarded
full-history `encode` bypass. Old decoder helpers are not advertised as an
autoregressive forecasting API. The mandatory CUDA proof checks suffix values
and masks, future poison, short and missing prefixes, semantic permutations,
internal temporal states and legal past positive controls before quality work.

## Historical V1 metadata and evidence

The [version index](versions/index.json) and
[prospective CPB-v1 manifest](versions/CPB-v1.json) are admitted declaration
snapshots. Their pending wording records the evidence stage when they were
frozen; they are preserved rather than rewritten as a live status page.

The [verified instance record](versions/instances/CPB-v1.market-01.json) binds
actual source/configuration, the usable typed checkpoint, both engineering
attempts, proof/admission/quality evidence, report and full-precision task
results. The [portable guide](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1.md)
contains measured task tables and reuse instructions. Checkpoints and data remain local artifacts; the portable guide records
the research measurements. These additive records carry later status outside the
admitted physical implementation and prospective manifest.

The passing CUDA proof covers all 128 cutoff indices, 2,350 suffix cases,
30 internal temporal cases and 14 typed negative cases. All 76 imported
parameter tensors (297,560 values), frozen statistics and saved/reloaded state
match. The proof is bounded correctness evidence. Legal past positive controls
change outputs/support; future values and masks do not.

MKT-HIST-2 complexity4/5 retains 112 TRAIN/32 known chronological VALIDATION
windows and four separate past-window targets. At the V1 stage, CPB-v1 established
matched trained-final comparison highs in all four linear tasks and neural
volatility/activity; RPB-v25 retained neural trend and relation means in that scope.
The neural trend repetitions are -0.33939984380671695,
0.657262428919963 and 0.4132006631059153, so the full market gate fails despite
passing every mean-score floor. TEMPO-5 complexity5/5 uses all three saved
cohorts, both scaler policies and intact/30%-erased views. Coverage is 100%
throughout; five of eight frozen transfer floors fail. Strong absolute scores
do not satisfy the full retention rule.

The recorded usable typed checkpoint has SHA256
`54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895`,
1,290,592 bytes. `proof/refusal-1.pt` through `proof/refusal-10.pt` are deliberately
malformed typed-loader test inputs, preserved as engineering evidence and
unusable as models. Attempt001's failed attention-mask proof, its exact source
and all three binaries remain preserved; attempt002's reviewed repair passed
before the single quality run. There were no new encoder updates, input/bank
fits, decoder calls, old encoder replays or head tuning. New fixed heads were
fitted once on TRAIN. Historical RPB targets retain their original window
semantics and are not relabeled causal or forecasting results.

RPB native36 remains the historical balanced research reference and all scoped leaders remain
preserved. Preserve historical source, checkpoints, cards and results, including
the interrupted, unscored V30 synthetic extraction. This guide starts no new
research, training, commit or push.

