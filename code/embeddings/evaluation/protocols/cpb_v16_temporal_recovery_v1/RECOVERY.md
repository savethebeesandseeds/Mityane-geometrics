# V16 external-interruption recovery declaration

Declared 21 October 2026, Asia/Dubai, after an external interruption and before
recovery execution, development extraction, readout fitting or March scoring.
This is an additive operational amendment for `cpb-v16-temporal-001`,
`attempt-001`. The owner authorized the V16 temporal-reconstruction study with
“go for your second choice, let's see”; recovering its interrupted execution
within the fixed scientific recipe continues that authorization. Neither the
original protocol nor the project instructions impose a separate human approval
requirement for this bounded engineering continuation. This declaration does not
authorize a replacement experiment, another recipe or a performance rescue.

The original [prospective protocol](../cpb_v16_temporal_v1/PROTOCOL.md), run card,
configuration, producer source and compiled binaries remain unchanged. Their
original paths, bytes and hashes remain their evidence identities. The new
recovery source, binary, wrapper, declaration, reviews and execution receipts
receive separate identities and explicit links to those original bindings.
Preparing this declaration establishes no successful recovery or scientific
result. Only the actual reviewed execution and downstream audits can do that.

## Interruption and exact scope

Five of the six declared fresh trajectories already have passed training
receipts at logical update 2048. Preserve those receipts and all their output
bytes; do not replay any of those five trajectories. The sixth trajectory is
the V16 candidate with encoder seed 946103, row seed 947103, mask seed 948103 and
context seed 949103. Its original process was externally interrupted; no original
training process survives. Bind the actual interruption observations rather
than claiming an inferred exception from the producer. The observed Docker
client exit was -1 while the existing managed container remained running; that
client status alone does not identify the cause of process termination.

The candidate already performed the originally prescribed, in-process typed
resume at 512. Its saved `typed-resume-512.json` reports exact parameters,
buffers, moments, optimizer steps, parameter order, counters and seed streams.
The latest retained typed training state is `training/checkpoints/training-512.pt`.
Use that fixed latest persisted state because it is the recoverable state,
without inspecting or comparing checkpoint quality. Do not restart from fresh
weights, reset the optimizer, refit preprocessing or select another checkpoint.

The original stdout log ends at update 512, but this is not the actual stopping
point. The original general training CSV contains complete rows through 553 and
a fragment for 554; the temporal CSV contains complete rows through 565 and a
fragment for 566. Thus work occurred after the saved checkpoint. The exact
number of successfully completed updates and the final in-memory state at
termination are unknown. In particular, complete temporal rows through 565 prove
more discarded work than the general CSV alone records. Preserve the fragments;
do not convert a fragment into a certified complete update.

Recovery reloads the saved 512 state and recomputes its discarded continuation,
then proceeds to the original logical final 2048. It cannot claim an
uninterrupted process, exact restoration of the lost interruption state,
zero repeated computations, or that the stdout stopping point 512 was the true
last update. Distinguish the final logical trajectory's 2048 updates from
discarded work and recovery work. The exact repeated computation count and
complete original execution cost remain unknown. No extra successful updates
are added to the accepted final model beyond logical update 2048.

The frozen recipe remains V4 Gaussian NLL plus 0.1 times the fixed equal-third
hidden-endpoint temporal Huber losses at lags 1, 4, 16. Retain all original
coefficients, masks, support rules, TRAIN-only data and preprocessing,
initialization, batch size, AdamW settings, clip 1, attempt counters, deterministic
row/mask/context stream derivation and final 2048 outcome. No seed, mask,
coefficient, learning-rate, checkpoint, head or training-budget choice is made
from scores. The logical attempt limit remains 4096; do not reset its counter
when loading 512. The original first-update witnesses remain original artifacts;
produce the actual final-update 2048 witnesses during recovery.

## Preserve original bytes before changing canonical outputs

Before model execution or canonical reconciliation, retain all 16 files inside
the interrupted candidate's `training/` directory under
`.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/interruption/snapshot001/`.
The retention manifest must bind every original locator and every new retained
alias with complete repository, path, byte count and SHA256 identity, and verify
identical retained bytes. These are new path-bound copies of historical bytes,
not a claim that the aliases were the original producer paths. Preserve the
sibling original `training.log` unchanged and bind its original identity
separately; it is not one of those 16 training-directory files. If it is also
copied, record that additional retained alias explicitly.

The planned snapshot includes that stdout log as an explicit 17th source-to-alias
pair. Its manifest kind is `cpb_v16_external_interruption_snapshot_v1`.

In particular, preserve both original CSV files including their incomplete
last rows, their 512 checkpoint, calibration artifacts, initialization and first
witnesses. Preserve the original complete-study orchestration log and any
actual interruption observation record. Do not invent a producer failure
receipt or raw stderr that the terminated process never wrote.

The recovery executable writes a new, exclusive output leaf. It must not edit
the original partial training leaf or stdout log. Retain its actual arguments,
binary identity, new log, exit status and outputs. A separately reviewed bridge
may publish newly completed artifacts at previously missing canonical paths
after recovery verification. Existing non-CSV canonical files must remain
byte-identical. Conflicts with any previously existing final artifact abort;
the bridge does not replace final evidence silently.

There are exactly two intended replacements of original canonical files:
`training/training.csv` and `training/temporal-training.csv`. Their full file
identities necessarily change and must be recorded as changes, with the
original identities and retained aliases still available. Canonical CSVs are
assembled from complete original prefix rows through logical update 512 plus new
recovery rows 513 through 2048, with the original header and ordering. Do not
include discarded original post-512 rows in that accepted logical trajectory,
duplicate 512, splice torn rows, fabricate missing values or retain stale final
fragments. Hash and retain the complete recovery CSVs and the reconciled files;
the manifest explains which byte/row ranges supplied each canonical file.

## Required recovery checks

Before loading the checkpoint, freshly verify the original run card, relevant
source/build/configuration/data bindings, retained 512 checkpoint and preserved
original artifacts. The typed loader must enforce the original model and
objective kind, configuration, dataset/split/recipe identity, parameter order,
optimizer state identities and steps, scaler/normalizer identity, logical
completed/attempted counters and all three seed streams. Loading an inference
checkpoint as training state, resetting moments or using fresh calibration is
not recovery.

Start from logical update 512 and its saved attempted-batch counter. Use the original
counter-derived streams and unchanged batch/target/context construction. For
every recoverable comparison in the discarded tail, require exact agreement
with available original scalar fields, counters, support counts and stream/row
records. Compare the general and temporal CSVs over their own available scopes;
one file's longer complete tail does not repair the other file's missing rows.
Require each preserved final CSV fragment to be an exact byte prefix of the
corresponding complete replayed row; its missing suffix is not evidence. Compare
complete rows' raw lexemes and parsed finite F64 bits, retaining U64 seeds,
counters and anchors without lossy numeric conversion. Abort on any available
replay-prefix scalar,
counter, support, stream, anchor or valid fragment-prefix mismatch. Do not pick
a looser tolerance, discard a mismatching row or launch another recovery to
obtain agreement. Such agreement is a bounded retained-output check, not proof
of equality to the unavailable lost in-memory state.

The reconciled CSVs must cover every logical successful update 1 through 2048
exactly once with the original column contracts and paired stream rules. Keep
first-update original witnesses and bind actual recovered final-update witnesses
to the same logical update/attempt contract. Validate the final model and
optimizer counters before saving final artifacts. Preserve the original typed
resume 512 proof; add a distinct external recovery proof. The earlier proof is
not evidence that the later external recovery already passed.

The typed checkpoint does not retain all cumulative parameter-learning evidence,
loader counters or wall/GPU timings from the interrupted process. Do not invent
complete-prefix counts, infer full training costs from partial logs, add
overlapping costs, or claim the recovery suffix's measurements cover 2048 steps.
Keep genuinely recoverable prefix aggregates and measured suffix aggregates
with their precise scopes and derivation. Unavailable complete-trajectory
evidence and full execution/recomputation costs are explicitly null or unknown.
Where an independent check requires evidence that cannot be recovered, report
that limitation and abstain from the unsupported claim rather than substituting
a fabricated value. Retain unchanged first/final scalar-arithmetic, output
adjoint, causal-proof, fit, prediction, support and metric checks.

## Mandatory admission and receipt bridge

New source and binary are not silently substituted into the original build
record or run card. Bind the exact recovery source closure, actual reviewed
build and new binary, this declaration, original card/protocol, interruption
and retention manifests, selected512 state, original and new locator mappings,
actual process call, recovery checks and canonical reconciliation in a new
mandatory recovery receipt. It must state repeated discarded-tail work occurred
and that its exact count and full cost are unknown. It must distinguish
source-only reviews, executed checks and actual scientific outputs.

The reconciliation and publication records use
`cpb_v16_external_interruption_merge_v1` and
`cpb_v16_external_interruption_publication_v1`. The canonical training receipt's
mandatory `operational_recovery` object binds `amendment`, `source_review`,
`build_admission`, `snapshot`, `suffix_execution`, `merge` and `publication`.
It requires `actual_new_binary_call:true`, `original_call_completed:false`,
`physical_repeated_updates_exact:null` and `no_replay_claim:false`.
`observed_post_checkpoint_complete_temporal_updates:53` denotes the retained
complete temporal CSV rows 513 through 565 only; it is not the exact physical
repeat count or a bound that accounts for every observed partial computation.

The recovered candidate's canonical training phase receipt retains the original
study-card identity and the normal phase/seed/arm/output contract, plus a
mandatory recovery binding. Its call inventory truthfully names the recovery
binary and actual arguments; it must not claim the old candidate training binary
produced new artifacts. Every canonical output has its current identity; changed
CSV identities are linked to their preserved originals. There is no earlier
passed sixth training receipt to overwrite. Preserve any partial/failed receipt
if one is later discovered and resolve that conflict explicitly before writing.

The frozen runner's generic receipt checks do not by themselves verify this
exception. Do not rely silently on permissive phase-receipt or call checks.
Before any development invocation, a separately admitted recovery-aware wrapper
must require and verify the mandatory recovery receipt, actual new source and
binary bindings, output mappings, tail comparisons and reconciliation. The
original five training receipts remain bound to the unchanged original card.
The wrapper establishes six complete logical update 2048 trajectories, rather than
asserting six uninterrupted original-process completions.

The recovery-aware report wrapper must perform those checks again and bind the
recovery declaration and complete evidence chain before admitting the scientific
report. Preserve the original reporter's metric calculations, fixed readout
recipes, protection predicates, aggregations and prospective March screen;
add explicit recovery and incomplete-evidence disclosures to its actual saved
report. A successful numerical screen does not erase the operational amendment
or confer unsupported execution-evidence claims. Source-only review of this
plan is not sufficient admission for report generation.

## Evaluation boundary and permitted claims

Do not extract development embeddings or fit/read out task heads until all six
logical update 2048 training trajectories and the recovery checks are complete. Then
run the original initial/final causal proofs and independent development audits.
Seal all six final checkpoints, their January TRAIN-fitted heads and all six
development audits before March features, predictions or outcomes are opened.
Every original mask/support, head, audit, coverage and confirmation protection
continues to apply. Missing required evidence or a failed check prevents the
corresponding success claim and must not be treated as a numerical pass.

Report the interrupted and recovered execution alongside the five original
completions. Scientific outcomes, if the required audits pass, concern the same
fixed three-seed V4-versus-V16 comparison and the same OR-05 past-window readout
and OR-09 fixed-query populations. They do not establish forecasting,
intervention, decision utility, an overall encoder selection or automatic
promotion. Preserve all earlier results and the original frozen protocol.
Continue using C++, Node and Bash only in the existing managed Linux container;
this amendment authorizes no Python fallback, dependency/container replacement,
cleanup, commit, push, publication or catalog application.
