# V9/V11 saved TRAIN information diagnosis — owned source preparation

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

These five files began as complete physical copies of the final five-file V8/V9
TRAIN diagnosis. All ten main/auditor parent files were copied, hashed and
preserved before any owned-source edit in
`output/engineering/causal-v9-v11-train-diagnosis-source-preparation/attempt-001/parent-copies/`.
The inert archive reader remains byte-for-byte identical. No older algorithm is
imported, forwarded, modified, or executed. The independent auditor has its own
physical reader and arithmetic source; it imports no diagnosis algorithm.

**Current disposition: SOURCE PREPARATION ONLY, pending independent full-source
review. No actual card, preflight, Stage, Schema, diagnosis, audit or model run
is admitted or performed by creating these files.** V3 remains the sole DEV
encoder. The exact V11 recipe is closed; there is no new encoder experiment.

The authoritative scope is
`doc/causal_v9_v11_train_diagnosis/DESIGN-001.PROPOSED.md` and refined
`design-inventory-002.PROPOSED.json`. The source is V11-only computation plus four
closed cached V9 selectors. It removes inherited V8 computation, old raw V9
reductions, four age-bin panels, variance-link derivatives, local adjoints,
finite differences, trajectory/gauge/Hadamard learning-trust replay and gradient
cosines. Those earlier results remain preserved and valid in their original
scope. No old numeric panel is silently advertised as newly recomputed.

## Closed data access

Exactly25 numeric roles are defined by `ALLOWED_IDS` and `expected_fields()`:
20V11 initial/final feature/query/summary/checkpoint/readout roles; two saved
actual update1/2048 witnesses; one V11 moment CSV; one existing shared TRAIN
112-row target archive; one cached audited V9 diagnosis JSON. The target file
lives in DEV and is read-only; no DEV source or artifact is changed.

Eight V11 feature/head/PT archives and metadata containers can contain other
fields. Numeric tensor access is restricted to explicit allowlisted logical
fields through the inert reader. Metadata unpickling does not import Torch or
run modules. It reads DATA metadata; only selected storage values are decoded.

- Features: `row_ids`, `values`, `valid`.
- Targets: `row_ids`, `group_ids`, `targets`, `valid`.
- Saved head predictions: row/group IDs, targets/support, validity and ridge/
  neural predictions. Fitted heads expose only saved TRAIN feature/target
  normalizers, ridge weights/intercept and seed. Neural weights are excluded.
- Queries: exactly11 fields: row IDs, target support, targets, mean, variance,
  native40, row MAE/NLL and three saved objective-component rows. No raw values,
  observed/hidden masks, visible masks, timestamps or age-bin panel is read.
- Checkpoints: exactly45 logical fields, explicitly enumerated at source. Typed
  provenance/layout/objective and all seven temporal/moment fields are checked
  at outer/payload nodes. Numeric parameter access is only the same-stage
  `payload.model.encoder.change_decoder` matrix3456x4. No full encoder, optimizer,
  variance projection/bias or variance-link parameters are read.
- Actual witnesses: exactly20 fields: update/attempt/clipfactor; before-step
  export weight/bias; packed positions/valid/rows; each of two routes' mixture,
  bin-valid, mean, centered variance, upstream-defined and PRECLIP upstream.
  Original hidden states, separate local/global weights, clipped gradients,
  after-step parameters and optimizer arrays are excluded.
- Query-summary JSON is decoded whole but returns only `train`. Cached JSON is
  decoded whole but returns exactly `models.v9.stages.initial`,
  `models.v9.stages.final`, `actual_pre_step_batch_diagnosis.v9` and
  `temporal_learning_observations.v9`. V8 and old comparison keys are excluded.

All28 old V9 raw-artifact bindings and six excluded V11 bindings are hash-only
preservation guards, verified before numeric values. There is no archive/CSV
numeric API for those guards. The existing V11 temporal receipt, global training
CSV, whole-group CSV, gauge CSV, between-bin CSV and witness bindings JSON remain
hash-only. Their already completed audit conclusions are context, not replay.

## Fixed panels and arithmetic disclosure

All new numeric panels describe MKT-HIST-2, joint BTC/ETH/SOL minute histories,
C3/H128/F9, 112 chronological TRAIN windows, complexity4/5 as an ordinal label.
Known VALIDATION32, TEST/stress, raw market caches and TEMPO files are excluded.
One historical fresh2048/B8 trajectory exists per version. Three head fits are
readout repetitions, not independent encoder training. Initial/final are saved
0/2048; actual update2048 is a pre-step state2047 witness.

For both V11 states, retain all four tasks, both saved readout predictions, all
three repetitions and fixed early56/late56 descriptions. Compute TRAIN R²,
physical MAE/RMSE and support from saved predictions. This is in-sample diagnostic
information, never a new method, holdout result, forecast or head selection.
Native learned20/spectral12/mean4/uncertainty4 get scalar population covariance,
participation and coordinate-to-target descriptions. No eigensolver, PCA,
projection, fitted transform or basis alignment exists.

Manual scalar contractions reconstruct the *existing saved ridge* from its saved
coefficients and normalizers, then decompose four native blocks in physical task
units. All four tasks/repetitions/halves retain target/residual correlations and
pairwise contribution covariance. No block-removal predictor or refit is formed.
The saved mean4 residual uses the matching stage's change_decoder and first-eight
query native mean4. The parent remainder is stored prediction minus this F64
manual affine residual, approximating original F32 addition. No nonlinear parent
decoder is invoked and no content/spectral decoder split is invented.

Reuse all112 saved query MAE/NLL/Huber/variance/objective rows and their distinct
units. For every task/head/repetition and all112/early56/late56 scopes, describe
Pearson associations between each row component and signed/absolute saved
readout residual. Eligible rows require saved readout support and positive query
support. These associations cannot establish a cause of validation changes.

The moment CSV reads all11 declared columns and all16384 rows. It describes eight
bin/moment subgroups overall and fixed updates1–512/513–1024/1025–1536/1537–2048.
The four gradient/movement norms and two recorded element counts get means and
fixed quantiles. No old event/partition/learning guard is recomputed; magnitudes
are coordinate dependent and movement includes AdamW decay.

Actual update1/2048 descriptions keep both ordinary/contextual routes, all four
bins and all packed rows. Moment, mixture-concentration and PRECLIP-upstream
quantiles use bin-present rows; mixture concentration spans valid tokens of the
whole half-local/half-global mixture. Saved before-step affine exports are
computed for every packed row, using fixed first4 mean/last4 variance export rows,
then zero the whole empty-local slice after bias. Missing present-bin summaries
are null. Undefined upstream with its explicit saved zero fallback is retained,
not interpreted as a dead branch. These traces cannot identify moment-to-readout
contributions for the112 query rows; V9 has no paired intermediate traces.

`manual_arithmetic_counters` discloses five families of saved affine arithmetic:
ridge_total, ridge_block, mean4_decoder, temporal_mean4_export and
 temporal_variance4_export. `dot_outputs` counts scalar dot outputs;
`multiply_add_terms` counts products in those dot sums, excluding bias additions.
AllR export rows count even when post-bias masking yields zero. Instrumented
mean/quantile/variance/Pearson/score invocation counts are implementation-specific
summary calls, not primitive FLOPs. Model/encoder/decoder/head module forwards,
neural forwards, fits and training updates are separately zero. Manual saved
contractions are performed arithmetic and are never hidden in those zero counts.

V9 reference differences subtract its existing cached descriptive scalars only.
The complete four selected cached subtrees remain unchanged. Its old independent
687825-check audit is reused; the new auditor must state it independently
recomputes V11 only and does **not** perform a second V9 reduction.

## Future card boundary

No wrapper or actual card exists as a result of this source preparation.
`Inputs` requires an externally pinned, separately admitted card with exact25
allowlisted numeric roles, 34 hash-only guards, twelve context inputs, actual
source pins and a separate pre-execution preservation receipt. The source/schema
card kind is `cpb_v9_v11_saved_train_diagnosis_card_v1`; stage is
`prospective_before_diagnostic_arithmetic`. It declares computed_versions V11
and cached_reference_versions V9; manual affine arithmetic is explicit and
module forwards/new quality/selection are forbidden.

Context binds the old prospective card, old independent audit, both old mirrors,
completed V11 TRAIN audit/final trust join, actual final109-checkpoint broad
preservation, reviewed/applied stronger future retention protocol, and current
ledger/catalog. Six exact final-context assertions require passed preservation,
109 critical checkpoints, passed retention review, applied future floors and V3
selection. Their actual selector paths/hashes must be reviewed when evidence
exists. No future hash, diagnostic outcome, check total or next encoder is
invented here. All context/guards are byte-verified before numeric storage access.

Later root-approved workflow preparation must physically own its metadata/card/
preflight/Stage/Schema/Run/audit/mirror tools. Preserve the earlier PowerShell
lesson: parenthesize array concatenation and reject literal `payload.` or duplicate
field-list entries. Actual card review precedes preflight. Every phase needs
separate admission in the existing managed cuwacunu container, exclusive leaves,
source before/after hashes, independently reviewed results and full preservation.
No CPU/host project execution or old inference/head fitting is permitted.

After independent source review and actual context admission, the future CLI is:

```text
python3 schema.py ACTUAL_CARD EXTERNAL_CARD_SHA256
python3 diagnose.py --card ACTUAL_CARD --card-sha256 EXTERNAL_CARD_SHA256 --output NEW_EXCLUSIVE_LEAF
```

These lines describe interfaces; they are not execution authorization. No live
report, ledger, catalog, source selection, checkpoint, DEV or Git publication is
changed by this source preparation.
