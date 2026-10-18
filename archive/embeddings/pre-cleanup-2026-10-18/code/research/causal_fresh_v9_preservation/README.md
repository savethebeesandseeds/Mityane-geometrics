<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# V9 final-preservation source workflow

Source preparation only. No actual V9 preservation input plan/card, verifier
invocation, Docker invocation or project/model execution occurs here. The
existing94 critical checkpoint pins and1030 historical pins already exist;
the future99 count is admitted only after five actual V9 checkpoint files exist.

verify.ps1 is a complete owned physical copy of the final V8 verifier
5a66c68e, extended by one preservation-chain generation. The byte-exact
v8-preservation-card.template.json is historical schema/data copied from the
134629-byte V8 card04058b51, not a newly authorized V9 verification card.
The new prepare-card.ps1 is an owned metadata generator because V8 had no
separate generator source. Provenance binds both direct parents and the final
V8 preservation receipt919b4f11. No old source is modified or forwarded to.

The verifier keeps V6/V7/V8 full metadata aliases and adds the six whole pre-V9
live snapshots. It checks all old1030 pins through those exact generation
aliases, old Complete mirrors/ZIPs,94 inherited checkpoint hashes,47 selected
V3 encoder sources in each repository and the sole selected V3 checkpoint
50e1d651. Concurrent DEV Latent Roadmap work is recorded without reset or a
clean-worktree claim. Only explicitly declared old DEV integration-control
exceptions are allowed; selected encoder sources/checkpoints never qualify.
Missing repository fields are accepted only for original V5 checkpoint paths.

The actual V9 Complete manifest determines source count; neither117 reviewed
assets nor118 capsule members is hardcoded as a future body assumption.
All seven V9 compiled binaries are separately mirrored and checked against
the build-admission byte/hash pins and the separate binary-mirror receipt;
all three sets must contain exactly the same seven unique binary identities.
They are not Complete.files members.
The original container build directory is independently hashed, not executed.
The three temporal CSVs, all source/AST evidence, read-only administrative
failures and pre-admission repair snapshots are preserved. PT ignore checks
cover every pinned historical/new PT file; archives include data/heads/tests,
not a count of trained encoders. No checkpoint is deleted or replaced.

prepare-card.ps1 requires an externally hashed actual-input JSON plan plus
RootReviewedActualInputs and FinalMetadataApplied. It refuses absent actual
five checkpoint files, Complete mirror, separate measured-candidate trust join
and reviewed/applied metadata. Only then may it write one exclusive card to
doc/causal_fresh_v9/preservation-card.json. No candidate score/pass count is
anticipated;22 outcomes are carried from the separate evidence join.

The future plan schema is:

- kind: causal_fresh_v9_final_preservation_input_plan_v1
- status: root_reviewed_actual_inputs; authorized: true
- twelve path/bytes/sha256 pins: prior_live_snapshot, complete_mirror,
  source_manifest, source_zip, main_card, metadata_application,
  proposal_review, final_establishment, compiled_binary_mirror,
  reporting_recovery_card, reporting_recovery_receipt and failure_mirror
- compiled_binaries: exactly seven actual host path/bytes/sha256 pins
- concurrent_DEV_exclusions: explicit array, empty when none are authorized
- protected_inventories: explicit array of pinned receipts with inventory_field,
  optional entry_pin_field and optional repository-relative prefix, as in V8
- protected_pins: explicit additional actual file pins, empty if none
- additional_protected_roots: explicit physical research-relative directory
  paths to completed audit/establishment/finalization mirrors or other actual
  failure/evidence leaves; empty only when all are covered by declared receipts

No actual plan is generated during source preparation. The final plan/card
must be independently reviewed after root supplies the real evidence paths.
Required post-V8 retention/diagnosis and V9 source-repair roots are included by
the generator; inherited V8 protected pins/inventories are retained whole.
The root must supply actual later audit/finalization mirrors as extra declared
inventories or roots. Numeric tensor/model contents are never deserialized.

The separately admitted reporting-only recovery is mandatory. Pin actual
card9c61e451, receipt5f96dfab and pre-recovery failure mirrore30f6cd7. The failed
original quality/report stages remain failed; the recovery binds all531 input
files before/after, the immutable118 source identity, five passed scientific
stages and zero scientific replay. The generator requires the complete host
recovery attempt folder and preparation snapshots as protected roots; the
verifier also confirms that folder is covered by exact Complete membership.
No quality result or scientific criterion is modified by this preservation.

Later invocation boundaries, only after separate root admission:

1. prepare-card.ps1 -InputPlan PATH -InputPlanSHA256 SHA
   -RootReviewedActualInputs -FinalMetadataApplied
2. Independent read-only review of that actual card and all source pins.
3. verify.ps1 -PreservationCard PATH -PreservationCardSHA256 SHA
   -FinalMetadataApplied

Both operations refuse existing output paths and preserve failed attempts.
The verifier's only result write is doc/causal_fresh_v9/final-preservation.json.
Its body performs bounded file/hash/ZIP/JSON and Git-ignore inspection, plus
byte hashing in the existing managed container. No encoder/head/optimizer
forward, fit, training update, quality replay, Git mutation or deletion occurs.
