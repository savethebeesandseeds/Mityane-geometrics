Frozen 2026-10-16T01:14:03.2527202Z. Machine card SHA256 `6acfaadfe758c0d252c8138fc2830807d40444f9a9adfa61f85ec13dde7c63a6`.

<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v8.market-01: four original-position temporal bins

**Frozen prospectively after root and independent source/card review. Runtime proofs and quality are unmeasured; execution must pass the frozen source admission.** The machine
card is [CAUSAL_FRESH_V8_CARD.json](CAUSAL_FRESH_V8_CARD.json); its final hash and
immutable version descriptor will be created only after owner source freeze.
The card and descriptor drafts are complete physical V6 copies followed by
owned V8 edits. V6/V7 and all historical source/result/checkpoint bytes stay exact.

CPB-v3.market-01 remains the clean baseline and sole dev encoder. Protect every
V3/V4/V5/V6/V7 scoped strength. This candidate uses the already reviewed
strongest V3–V7 [22-condition protocol](causal_fresh_v8/goal-protocol.json),
SHA256 `c0a46527388032ea4f43dd9780ceedef486e5db7b4502ce37165d6e8616368b7`,
with its [review](causal_fresh_v8/goal-protocol-review.json) and
[application admission](causal_fresh_v8/goal-protocol-admission.json).
Historical pass counts retain their own cards; they are not equivalent floors.
TRAIN Gaussian NLL remains a preserved descriptive record, never a new gate.
No winner, application change, automatic promotion or publication follows.

## One architectural change from the V6 recipe

V8 is initialized from its own seeded constructor, not from V6 weights. It keeps
V6's constructor draw order, parameter shapes, fresh TRAIN statistics,2048/B8
budget, streams, objective, global clip1, AdamW and D40 heads. It replaces one
learned temporal pool per channel with four fixed original-position bins. This
is an information-allocation change with unchanged parameter capacity, not a
change in loss coefficient or clipping cap.

The precise policy is
`original-position-four-bin-disjoint-export-rows-v1`. With history128/patch8,
K=16 patch positions and D=32 export coordinates, the typed partition is
Long[5] `[4,16,32,4,8]`: four bins,16 original positions,32 export coordinates,
four positions per bin,eight export rows per bin. Each bin uses the existing
learned scores and a within-bin masked softmax. It projects its pooled64-vector
through its own eight rows of the existing32×64 export weight and matching bias;
the four slices concatenate to32 per channel. The unchanged99→64→32 global
route supplies learned20; fixed spectral12, mean4 and uncertainty4 retain their
routes. Native40 is20+12+4+4. No new learned tensor or changed RNG draw is added.

Membership uses original patch positions, never compacted order after masking.
All-empty score rows become finite zeros before softmax; nonmembers and unsafe
hidden values contribute zero. Empty-bin output is zeroed after projection bias.
The ordinary and early-contextual paths use the same owned pooling kernel.
Causal contextual states already contain positions and temporal context; this
card does not claim the previous summary was order-invariant.

The [saved TRAIN V6/V7 diagnosis](CAUSAL_V6_V7_TRAIN_DIAGNOSIS.md) and its
[independent616228-check audit](causal_v6_v7_train_diagnosis/independent-audit.json)
precede this declaration. Both TRAIN readouts are strong, the oldest bin has the
lowest first8 error, and unequal support prevents an age-controlled conclusion.
The diagnosis did not prove pooling caused held-out losses. Four-bin pooling is
a controlled structural hypothesis. It does not rescue the stopped V7 clipping
recipe or combine V7 branch clipping with a pooling change.

## Freshness, objective and update policy

All76 learned tensors/294752 values start fresh under seed943001. The3888
literal-prior values are fixed; total parameters298640 across77 tensors. No old
model, teacher, checkpoint or fitted statistic may initialize the run. Calibration
fits input and270-coordinate bank population statistics once from the same4096
saved TRAIN anchors/recent128 windows; validation/test use is zero and calibration
encoder calls are zero. The fresh statistics are then frozen.

The V6 objective remains

`R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v + log(v))]`,

with coefficients1 each, delta1, original eligible-example/channel/target-cell
hierarchy, sanitized F32 mean/target operands promoted separately to CUDAF64
before subtraction, and the same variance link. Reporting Gaussian NLL/MAE use
unchanged formulas; reporting NLL never backpropagates. Variance and total losses
may be negative. A single global norm2 cap1 and single AdamW remain: learning
rate0.001,weight_decay0.0001. Mean residual changes and optimizer history retain
coupling; zero direct cross-path gradient does not imply optimizer independence.

One fresh2048-successful-update, batch8 CUDA:0 trajectory uses initialization /
row / mask / context seeds943001/943002/943003/943004, query fraction0.25 and extra
context deletion0.15. Keep the existing saved fixed-query plans. Save inference0
and2048, typed full training512/1024/2048. Resume512 must continue the actual
trajectory with model/buffers/moments/order/counters/streams/recipe/data identity
checked at the saved state; the one actual trajectory then continues without
replayed steps. A separate bounded engineering fixture tests next-step bit parity.
No training replay, old encoder execution, head
retuning or checkpoint selection beyond initial0/final2048 is allowed.

## Prospective learning and mechanism admission

The scalar `core.encoder.level.base.backbone.pool_score.bias` is a softmax gauge:
a constant shift cancels inside every bin's softmax. It remains registered,
freshly drawn and in the same AdamW. Its observed gradient/movement is recorded
separately; weight decay alone is not evidence of data-learning. Exactly this
one scalar is exempt from mandatory nonzero data-gradient learning. All other
75 identifiable learned tensors/294751 values must satisfy the declared
nonzero-gradient/movement admission; do not claim every scalar coordinate moves.

Require actual main-trajectory and nonzero CUDA-fixture evidence that all four
existing export weight/bias eight-row groups receive nonzero data gradients,
move and differ finally. Require each bin's centered four-position
`pool_positions.weight` contrasts to receive nonzero data-gradient/movement
and final centered change. A common positional offset within a bin is itself
a softmax gauge, so raw positional-offset decay is not the contrast proof.
These are at-least-one-event requirements, not every-update requirements.

The run records8192 group rows (2048×4) and2048 scalar-gauge observations from
existing backward/optimizer steps, without extra encoder forwards. Data-gradient
records are observed after the actual global clip; movement includes AdamW
weight decay and is labelled separately. The pinned temporal learning receipt
binds both CSVs and named group counts. Independent auditing checks the stored
records without claiming to reconstruct upstream gradients.

A separate admitted seventh binary proves original-position manual pooling
parity, nontrivial mask/empty-bin behavior, finite exact-zero unsupported output,
four disjoint projection-row/position-contrast gradients and actual optimizer
movement, and scalar softmax-gauge invariance. Manual F32 reference tolerance is
abs/rel2e-5; membership/empty output are exact, gauge tolerance2e-6. The dedicated
PASS prefix is `PASS CPB-v8 original-position-four-bin-disjoint-export-rows-v1:`.
Per-bin mass tolerance is2e-6 and flat-trace mass tolerance8e-6. Its explicit
scope is one backbone Model::encode covering both real temporal paths,two direct
kernel calls,zero public Encoder/head calls and one synthetic AdamW step. The
machine card names the exact membership/trace/manual-gradient/four-row-group/
centered-position/gauge booleans and binary/log pins. The unchanged
composite-objective CUDA fixture remains a separate proof; it must not be
mislabeled as the new temporal mechanism proof.

All128 cutoffs,2350 future-value/mask suffix cases and30 internal cases must pass
before/after engineering updates and for actual saved initial0/final2048, with
unchanged saved states and zero quality-training updates. Typed inference and
TRAIN roots/payloads reject wrong temporal policy, malformed partition, old
model lineage and mismatched objective/layout/progress/optimizer state. The
partition is stored as `temporal_pooling_partition_v1` in archives; common JSON
uses `temporal_pooling_partition` alongside `temporal_pooling_policy`.
The fresh-constructor JSON intentionally omits only `gradient_policy`; this
single known schema exception is explicit, while objective/temporal/gauge fields
remain required and typed/saved/engineering/training policy fields stay strict.

## Initial pairing is intentionally partial

V8 changes the learned content20 forward calculation at initialization. Do not
require or report whole40 initial equality. The13 references are DATA/statistic
archives from V6 initial; no reference checkpoint is numerically opened:

- Eight feature archives compare **only coordinates20:40**: unchanged fixed
  spectral12, mean4 and uncertainty4. Their declared valid/row/source/label
  fields compare whole. Learned content0:20 may differ.
- Five freshly fitted statistic archives compare their declared whole numeric
  fields: input scaler,bank normalizer and three synthetic TRAIN scalers.
- Metadata/file bytes may differ independently of numeric equality. Initial
  V8 embeddings and all own initial heads are newly produced; no initial score
  or old fitted statistic is reused.

All13 pairing witnesses and independent training/quality audits are mandatory
before overall trust/advancement. Raw report/comparison producers must label
those pending separately from runtime engineering admission and22 numeric
conditions/full coverage. No producer may mark overall establishment true from
scores alone. A later explicit evidence join can establish the measured
candidate without selecting an overall winner or application encoder.

## Fixed quality recipes and protection

Market evaluation has112 TRAIN/32 known chronological VALIDATION probe windows
on one January2024 BTC/ETH/SOL history, not independent market cohorts. It uses
four past-statistic regression targets and the unchanged saved targets/scalers.
Each new initial/final native40 stage gets its own ridge1 and tanh16/Adam0.01/
100-update head fits, repetitions2701/2802/2903. D40 seeds remain
6049515622977958931 /8262941649267994858 /2095371932330426401; market head
parameters164/724. No post-encoder PCA or head tuning. Three head fits are not
three independently trained encoders.

TEMPO-5 retains cohorts941101/941102/941103, each512 TRAIN examples from256 paired
sources and256 VALIDATION examples from128 distinct paired sources. Both stages
use all three unchanged fixed-head fits, head parameters82/690, intact and30%
extra coordinate deletion, under freshly fitted synthetic TRAIN scaler and
fresh market TRAIN scaler policies. Synthetic TRAIN statistics are fitted once
per cohort; no synthetic encoder training occurs. The four panels stay separate.

Use all five exact final V3/V4/V5/V6/V7 control tables and their original costs.
V6/V7 raw table pending-trust fields are preserved; their separate exact final
establishment receipts supply full-trust admission. V3 remains selected. Detailed
encoder-free raw controls are reused byteexact; no old encoder/head is replayed.
Report task-separated standard five-column tables,physical MAE/RMSE,all three
head-fit scores,all four timing panels,query MAE/NLL/components and separately
labelled synchronized training/extraction/head/I/O costs. V3's historical144s is
reused descriptive evidence, not a newly matched benchmark. Complexity4/5 and
5/5 are ordinal designed recipe labels, not accuracy or measured task difficulty.

The exact22 criterion identities/rules/floors are copied literally from the
reviewed c0a465… protocol: eight market retention,eight timing retention,two
query MAE,strict known-validation NLL,all-three positive neural trend andtwo
positive relation requirements. No score is invented or old count recomputed.
Coverage must be100% market32/32 and each timing panel256/256, separately from
trust and numeric conditions. No TRAIN NLL advancement criterion is added.

Stop at the declared2048 budget and report every failure. No cap/coefficient,
rate/seed/budget/head rescue. Preserve all new scoped gains even if not all22
conditions pass, together with every prior checkpoint/result/source/failure.
No automatic application replacement, overall-winner selection, commit or push.
Use only the existing GPU-enabled managed cuwacunu container with exclusive
source/build/run/quality leaves and frozen source/compiler/binary closure.
