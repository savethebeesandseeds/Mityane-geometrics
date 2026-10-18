# V6 metadata proposal tools

These owned standard-library tools derive from the complete V5 metadata
finalizer. They read saved JSON/text and hash artifacts. They do not construct
models, fit heads, choose checkpoints, apply live metadata, change dev, rewrite
the prospective CPB-v6 descriptor or publish Git changes.

The generic source does not hard-code the V6 numeric pass count. Root will supply
a separately reviewed input plan after reports, all13 initial numeric pairings,
independent training/quality audits, the complete mirror and a separate final
establishment receipt exist. Nothing in this directory has been executed.

## Later execution contract

Use only the existing managed cuwacunu container and a new exclusive leaf.
The original completed model attempt is
`/opt/cuwacunu/research/causal-fresh-v6/attempt-001`.

1. Stage both tools, the root-reviewed plan and completed saved metadata.
2. Run `python3 prepare_card.py --plan /ABS/plan.json --output /ABS/NEW_SOURCE`.
3. Independently review the resulting pinned card/source before running
   `python3 /ABS/NEW_SOURCE/finalize.py --card /ABS/NEW_SOURCE/finalization-card.json --output /ABS/NEW_PROPOSALS`.
4. Review the three generated metadata proposals independently. Applying them
   is a separate root action. A failed output leaf and its failure.json stay
   preserved; retry only under a new leaf with additive declarations.

The plan kind is `cpb_v6_metadata_finalization_input_plan_v1`, with
`approved_for_metadata_card_generation: true`, `client_date`, the exact
`container_attempt`, `generator: {bytes,sha256}`, `next_direction:
{status: proposal_only..., action: ...}`, `evidence_contract`, and31 inputs.
Each input supplies `id`, research-relative durable `path`, absolute existing
container `source_path`, `bytes`, `sha256`, and `format` (json or text).

The exact input IDs are ledger, catalog, prior_snapshot, descriptor,
experiment_card, source_manifest, build, training, resume, initialization,
calibration, engineering, gradient, initial_proof, final_proof, proof_recovery,
proof_recovery_stage, training_audit, quality_audit, pairings, establishment,
mirror, quality_receipt, comparison, v3_table, v4_table, v5_table, v6_table,
results, review and detail. Ledger/catalog must be the six-file prior-live
snapshot bound by manifest SHA256
5eb6f38f1b03c95116b832003ddb86c03f4aa4ad7712363f54bad2c5238a0358.
The prospective encoder card and descriptor hashes are fixed in prepare_card.py.

The evidence contract maps actual receipt schemas through explicit JSON key
paths; it does not invent field names. `assertions` contains exactly:

- pairing_status: pairings, equals passed; pairing_count: pairings, equals13.
- training_audit_status and quality_audit_status: their own receipts, passed.
- establishment_status: establishment, passed.
- establishment_runtime, establishment_trust, establishment_coverage and
  establishment_measured: establishment, true.

Each assertion is `{input, path: [key,...], equals}`. The contract also supplies
`training_audit_checks_path`, `quality_audit_checks_path` and
`establishment_bindings_path`, plus `establishment_binding_keys` mapping
experiment_card, comparison, v6_table, pairings, training_audit and quality_audit
to their actual evidence-map keys (for example card and table_rows). Each carries
the exact corresponding bytes/sha256 pin. This separate receipt is mandatory
for full trust. Its measured_candidate_established may be true while overall
establishment and advancement remain false; the latter must agree with the
derived numeric22 result. Original producer comparisons retain false/pending
overall trust unchanged; numeric22 outcomes are recomputed only from their
saved boolean records. No additional floor is introduced.

## Proposal behavior

Only ledger, current catalog and a new CPB-v6.market-01 instance are emitted.
The CPB-v6.json descriptor remains a pinned immutable input. CPB-v3 remains
baseline/application. All old candidates, scorecards, history and the complete
V5 current entry are retained. Six prior scorecards gain V6 initial/final, giving
eight. The16 quality scopes use the maximum saved final score acrossV3–V6;
the four reconstruction scopes use the minimum final value. Stable chronological
ties retain the earlier leader. Every old full leader record is added to history;
initial results remain descriptive and never select a checkpoint. TRAIN NLL is
a descriptive reconstruction leader and adds no advancement floor.

The new instance binds102 frozen source files, six binaries, five checkpoints,
the corrected final-proof-attempt002 recovery evidence, objective/routing fields,
cost, fixed query components and separate independent audits. Shared global clip1
and AdamW coupling remain explicit. All model/fit/update/live-write counters for
this metadata operation are zero. The search remains active; any next direction
is a saved-data proposal, not a training authorization.
