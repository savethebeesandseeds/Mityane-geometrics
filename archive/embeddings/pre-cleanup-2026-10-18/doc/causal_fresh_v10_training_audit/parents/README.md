# V9 independent saved-training audit workflow — source preparation

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

Status: source and metadata preparation only. No actual audit card, Docker
invocation, AST/schema check, auditor body, model, head, fit or update has run
through this workflow. Each later phase requires its own root admission.
These helpers do not authorize main encoder or head execution.

The stage/execute/mirror files began as full physical V8 helper copies. Their
exact original bytes remain under `parents/`, with original repository paths
and hashes in `parents/manifest.json`. The V9 quality-card generator is the
complete physical metadata-generator parent because V8's training card was
written directly. Two V9 quality helpers supplied independently preserved
stronger transport-guard reference bytes. The new host run helper is owned
orchestration, rather than an import of an old implementation. No helper
imports an old or producer module.

All numerical audit work stays inside the existing managed `cuwacunu`
container, immutable ID
`82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e`, image
`sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7`.
Native calls use `docker.exe`; no function named Docker shadows the executable.
Nothing creates, starts, stops, replaces, cleans or deletes a container or volume.

The sole planned audit attempt is
`/opt/cuwacunu/research/causal-fresh-v9-training-audit/attempt-001`, mirrored at
`output/engineering/causal-fresh-v9-training-audit/attempt-001`. Preserve any
existing attempt, partial mirror, invocation, failure and log; helpers never
retry or overwrite evidence. Main input scope is the completed
`/opt/cuwacunu/research/causal-fresh-v9/attempt-001/{source,run-001}` and explicit
completed engineering/objective/temporal proof files. The audit workflow never
recurses through concurrently running main saved-proof or quality directories.

`prepare-card.ps1` is a later host file/JSON/hash operation only. It requires
`-SourcesReviewed`, external main-card/manifest SHA256 values, a passed completed
Training mirror SHA and an independently passed source-review path/SHA. It
verifies the three current auditor pins in that review, actual main package,
completed training stage/summary and exact pooling declaration before writing
one exclusive `audit-card.json`. A quality run or Complete mirror is not a
dependency. Unknown main source count and hashes are read from those actual
externally pinned artifacts; no old count or future score is invented.

The numerical source is the separately owned
`code/research/causal_fresh_v9_training_audit/` three-file auditor. Its declaration
keeps exactly2048 successful updates while allowing2048–4096 attempted batches,
with dense attempt/skip rows, actual seed streams, typed checkpoint attempts and
actual first/last witness associations. It checks the same unit Huber1 mean plus
detached-error variance objective, global clip1, fresh TRAIN statistics and exact
typed resume without replay. Only75 identifiable learned tensors are mandatory;
their294751 values are numel capacity, never every-scalar gradient proof. The
retained scalar bias gauge is observed separately. Existing8192 group/2048 gauge
rows and new2048 between-bin rows are checked; the fixed unit R16 Hadamard
directions require gradient/movement events and positive OVERALL final projection,
not every final coefficient nonzero. Actual raw arrays remain restricted to
updates1/2048, with F64 operands before subtraction.

Later CLI sequence, shown as interface documentation and not executed here:

1. Generate the declaration with `prepare-card.ps1 -SourcesReviewed
   -MainCardSHA256 ... -MainManifestSHA256 ... -TrainingMirrorSHA256 ...
   -SourceReviewPath doc/.../independent-source-review.json
   -SourceReviewSHA256 ...`. The review fields are `source`, `readme`,
   `provenance`, `status=passed`, `blockers=[]`.
2. Stage with `stage.ps1 -StageAdmitted -AuditCardSHA256 ...`. It checks the
   external card before parsing, its own source pin and the existing container,
   creates exclusive leaves, then verifies raw ZIP/manifest/card hashes before
   extraction. The capsule has six payload files plus `SOURCE_MANIFEST.json`;
   only the auditor and recorder receive AST checks. No import or numerical body.
3. After a separate numerical admission, invoke `run.ps1 -RunAdmitted
   -AuditCardSHA256 ... -StageReceiptSha256 ... -ManifestSHA256 ... -ZipSHA256 ...`.
   The spelling `-StageReceiptSha256` is the actual mandatory API. The host
   checks all four raw hashes before JSON parse, records a once marker, and
   verifies the container recorder's exact source hash before execution. The
   recorder repeats transport/source checks, requires completed actual training,
   obtains a nonblocking lock, and starts one auditor subprocess. Exact CLI
   includes `--temporal-proof`. It preserves argv/stdout/stderr/raw result,
   source before/after, execution disposition and a complete audit-leaf inventory.
4. Mirror with `mirror.ps1 -AuditCardSHA256 ... -StageReceiptSha256 ...
   -ManifestSHA256 ... -ZipSHA256 ... -ExecutionSHA256 ... -InventorySHA256 ...`.
   It copies to an exclusive `container-mirror`, verifies every file and raw
   receipt before parsing, and preserves any failed audit disposition. The
   canonical successful raw JSON is copied byte-exactly to
   `doc/causal_fresh_v9/independent-training-audit.json`; execution and mirror
   receipts remain separately bound. No audit is replayed during mirroring.

Core checks and all five module counts remain separate:
`numeric_statistics`, `saved_tensor_state`, `manual_saved_Gaussian`,
`actual_pre_step_composite_objective`, `temporal_pooling_saved_learning`.
The recorder copies those counts and does not invent a total. Any later total
must explicitly sum core plus five modules. Quality/partial13 pairings/e792
retention gates and host checkpoint availability/ignore checks remain separate.
No winner, next candidate, application replacement or Git action follows.
