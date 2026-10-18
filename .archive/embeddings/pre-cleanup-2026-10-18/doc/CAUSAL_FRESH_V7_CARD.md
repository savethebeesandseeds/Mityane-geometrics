<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v7.market-01 frozen prospective card

**Frozen before any build, calibration, model, training or quality execution.**
Root has reviewed the mechanism/source contract. Independent preflight and
source/compiler/binary/input admission must precede execution. The machine
authority is [the exact JSON card](CAUSAL_FRESH_V7_CARD.json). This card began as a
complete physical V6 card copy; the original remains unchanged. The immutable
prospective CPB-v7 descriptor will be created only after the final card hash,
before build or measurement, with a later additive measured instance record.

CPB-v3.market-01 remains the clean baseline and sole dev encoder. All V3/V4/V5/V6
sources, checkpoints, initial/final results, fixed heads, failures and displaced
whole scope records stay preserved. V1/V2/RPB are historical evidence, never
pretrained initializers, teachers or oracle quality floors. No application
selection, publication, commit or push follows this declaration.

## One controlled change

CPB-v7.market-01 tests **separate norm2 gradient clipping at1 for the disjoint
mean and uncertainty parameter sets, followed by the same single AdamW step**.
It keeps V6's forward architecture, native40, fresh constructor draw order,
TRAIN-only calibration, objective and coefficients,2048/B8 budget, data/mask
streams and fixed heads. It does not continue or import the trained V6 weights,
optimizer, fitted scaler or bank statistics.

The objective remains

    R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v + log(v))]

with coefficients1/1, original supported-cell/channel/example reductions,
independently sanitized/promoted F64 operands before subtraction and unchanged
positive-variance link. Mean/parent receive Huber gradients; uncertainty4 and
variance head receive Gaussian calibration gradients on detached mean error.
The separate reporting Gaussian NLL is unchanged numerically and never an
optimized mean loss. Variance/composite values may be negative.

The mechanism kind is `disjoint-mean-uncertainty-clipping-v1`, distinct from the
unchanged objective policy `huber1-mean-plus-detached-error-gaussian-variance-v1`.
The exact gradient policy is
`huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;disjoint-mean-uncertainty-clip1-v1`.
The exact clip policy is `disjoint-mean-uncertainty-norm2-clip1-v1`.

Mean has73 learned tensors/293537 values: every learned parameter outside the
three uncertainty names. Uncertainty has3 tensors/1215 values:
`core.uncertainty_projection`, `core.variance_projection`, `core.variance_bias`.
The fixed literal prior is1 tensor/3888 values and is excluded. Every learned
parameter must belong to exactly one set, without overlap, omission or a new
unrecognized assignment. The76 learned tensors/294752 values and77 total
tensors/298640 values match V6. All learned values start fresh.

Each group's cap is fixed prospectively at1, norm type2. Two disjoint unit
balls permit a combined post-clip norm up tosqrt2; the original single joint
ball capped the total at1. This changes clipping geometry and the possible
combined norm budget. It is a confound to disclose, not equivalent-budget or
complete optimizer independence. One AdamW lifecycle and residual-dependent
future variance targets remain. There is no coefficient/cap/seed/rate/head
sweep, rescue or quality-driven checkpoint choice.

The [audited TRAIN diagnosis](CAUSAL_V5_V6_TRAIN_DIAGNOSIS.md) motivates this
question: observed V6 factor below1 on1326/2048 updates, with frequent later
clipping. V6's TRAIN volatility/activity readouts were strong; trend weakened.
The evidence does not identify which branch caused clipping, infer gradient
dominance from loss sizes/cross-basis norms or demonstrate a held-out cause.

## Fresh geometry, data and update policy

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute-history retention ·
**complexity 4/5** · unlabeled chronological TRAIN,4096 saved anchors.

Native40 remains learned content20 + literal spectral12 + mean4 + uncertainty4.
Raw input is F64; C3/H128/F9/P8, encoder width64,3 layers,4 heads, feedforward256
and decoder hidden128 stay unchanged. Both bank270→4 projections retain their
V6 construction order; the uncertainty projection remains constructed last.
Fresh seed943001 constructs all learned tensors directly, source_updates=0.
No old checkpoint/statistics artifact is opened to initialize or calibrate.

Fresh input and relative-bank statistics use the same4096 TRAIN recent128
windows and frozen population-statistics recipes, scale floor1e-6. No encoder
forward occurs during calibration; no validation statistic is used. Freeze all
new statistics before training, preserving their values and the3888 priors.

One2048-successful-update run, batch8, AdamW learning rate0.001/decay0.0001,
dropout0, target patch fraction0.25 and separate0.15 context deletion. Seeds
row943002/mask943003/context943004 are unchanged. Context deletion never becomes
target support. No attempt/budget extension, alternate seed or rerun rescue.
Inference checkpoints are initial0/final2048; typed training states512/1024/2048.
Actual resume512 continues this same trajectory with exact next-step parity.

Typed V7 artifacts bind the exact objective/gradient/native layout and five
clipping fields at inference outer, nested payload and training root:

- `gradient_clip_policy`: exact policy text.
- `gradient_clip_max_norms_f64`: Double[2] values[mean1,uncertainty1].
- `gradient_clip_norm_type_f64`: Double scalar2.
- `gradient_clip_partition_v1`: Long[6] values[73,293537,3,1215,1,3888].
- `gradient_clip_parameter_names`: canonical newline text, all `mean:*` names in
  registration order, then `uncertainty:*`, then `fixed-excluded:*`.

Direct-route text is
`mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;disjoint-mean-uncertainty-clip1`.
Reject old V6 joint-clip lineage, wrong kinds/layout/objective/caps/norm type,
partition/name/order/counter/recipe/dataset mismatch and corrupted moments.
Keep complete model/optimizer/moments/progress identity; no inference-only
resume or silent optimizer reset. Exact JSON selector applicability must follow
the final producer schemas without weakening typed artifact guards. Main
`fresh-initialization.json` intentionally omits `gradient_policy`; verify this
specific absence while requiring its common objective/clip/direct-route fields.
Engineering, training and saved-state proof records retain their required
gradient policy. The training-only `encoder_gradient_kind` selector must not be
invented in other JSON record schemas. Initial typed model/training artifacts
retain every exact objective/gradient/clip field, including initial0.

## Proof and saved evidence required before quality

The scientific proof requirements and exact receipt contract are fixed before
measurement:

- Direct fresh-constructor state matches the saved initial artifact, no learned
  ancestry or old-stat import, source0, all76 learned tensors freshly seeded.
- Strict masks/raw cutoff and full native40 causal preservation: all128 cutoffs,
  2350 future-value/mask cases and30 internal interventions before/after small
  engineering updates and on actual saved main initial0/final2048; past positive
  interventions, frozen tested states, no quality-time training.
- A nonzero CUDA fixture preserves forward/objective values, manual composite
  gradients and direct route separation; proves complete/disjoint canonical
  mean/uncertainty sets, independent factor formulas/caps, cross-branch clipping
  isolation and one actual AdamW learning step. It discloses the sqrt2 bound.
- Actual CUDA finite nonzero gradients and movement for every learned
  tensor/category, fixed priors/statistics unchanged; do not claim every scalar
  coordinate moved. Full typed state and actual next-step512 resume with
  refusal tests for wrong old policy/group fields and malformed state.
- Hash-bound exclusive source/compiler/binary/input/checkpoint closure, full
  ownership of V7 files and immutable prospective descriptor before measurement.

All2048 rows record original components/report NLL plus observed mean and
uncertainty pre-clip norms/factors and exact membership. First/last actual
pre-step component witnesses come from existing forwards, without new updates.
These are observed group norms; loss scalar ratios do not establish dominance.
Independent saved-evidence training/quality audits and all mandatory trust
receipts are required before overall advancement. Producer runtime admission
must remain separate from later pending pairing/audit trust.

## Fixed-head evaluation and pairing

Initial0 and final2048 each get their own unchanged fixed-head fits. Reuse exact
saved V3/V4/V5/V6 final control scores/costs and the exact prior raw controls;
no historical encoder/head execution. No PCA, post-encoder transform, fitted
subspace, seed rescue, head tuning, target/scaler change or checkpoint selection.
V6's immutable raw table retains its original pending trust fields; the exact
separate [V6 establishment receipt](causal_fresh_v7/baseline-v6-establishment.json)
binds that table and completed trust. Never rewrite old producer fields to
admit the reference.

Dataset: **MKT-HIST-2** · **complexity 4/5** ·112 TRAIN/32 chronological
development VALIDATION windows per stage; four retrospective regression tasks.
Ridge penalty1, tanh16/Adam0.01/100 updates, repetitions2701/2802/2903. D40 heads
have164 linear/724 neural parameters; unchanged unsigned derived seeds are
6049515622977958931,8262941649267994858,2095371932330426401. Report each task,
all three neural fits, physical errors, coverage and query/cost scopes separately.

Dataset: **TEMPO-5** · slow/fast nine-feature lag-sign timing · **complexity 5/5** ·
three original cohorts941101/941102/941103. Each has512 TRAIN examples/256 source
groups and256 VALIDATION examples/128 source groups. Retain all four panels:
synthetic_train versus market_scaler, intact versus extra30% coordinate deletion.
The cohort TRAIN scaler is fitted fresh once per cohort and shared by initial/
final; market_scaler uses the new market TRAIN fit. Heads have82 linear/690
neural parameters, the same D40 seeds, ridge1 and tanh16/Adam0.01/100 steps.
Three cohorts/readout fits are not three independent encoder trajectories.

Thirteen V6 saved initial DATA/statistic artifacts are pairing witnesses only:
eight whole-native40 feature archives and five fresh-statistic archives. Compare
the declared logical numeric tensors exactly, distinguishing metadata/file-byte
differences. Never open old model tensors or import old fitted statistics.
Do not reuse initial quality scores: V7 fits its own initial heads regardless.
Pairing is mandatory trust before advancement, not a quality-selection rule.

## Strongest-final retention and interpretation

Use the independently reviewed/admitted
[V3–V6 strongest-final22 protocol](clean_causal_encoder_search_goal_v6_retention.json),
SHA0965842331056b61faee827a5ea782703205b23bd326298929ed3ecfca4604fb,
with its separate completed review/admission. It raises9 floors and preserves13
relative to the47fa parent; both old protocols and all historical pass counts
remain byte exact. Copy its22 criteria literally, never recompute or loosen a
floor from candidate scores. Protect16 quality scopes, TRAIN/validation MAE,
strict best known validation Gaussian NLL and the three existing positivity
conditions. **No TRAIN NLL criterion** is added; its scoped record is preserved
descriptively. Complete coverage and all trust/pairing/audits are separate gates.

No V7 score or outcome exists. All22 numeric conditions plus complete trust and
coverage are needed for advancement; no automatic application promotion follows.
Known development VALIDATION is neither independent confirmation nor forecasting.
If this exact fresh recipe fails, preserve its scoped gains and stop it without
cap/coefficient/seed/rate/budget/head/checkpoint rescue. Any next mechanism needs
its own prospective source/card and retention declaration.

Use only the existing managed GPU `cuwacunu` container with exclusive admitted
source/build/engineering/main/proof/quality leaves. New source hashes belong to
the independent preflight/capsule after the final card hash, avoiding a cycle;
the card pins inputs, historical preservation and prior diagnostic evidence.
The V6 card and its measured records stay unchanged. The pre-freeze V7 draft
bytes are preserved separately. Card freeze is prospective declaration, not
runtime admission, a quality result or overall advancement.
