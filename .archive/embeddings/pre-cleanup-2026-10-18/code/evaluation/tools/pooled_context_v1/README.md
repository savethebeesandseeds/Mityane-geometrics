# Pooled-context saved-arithmetic source tools

These are byte-exact copies of the reviewed **blocked** `pooled-context-v1`
reader and reporting sources, plus their sole pinned stdlib CPU archive-codec
source. `source-manifest.json` records original role paths, lengths and SHA256
for the seven preserved sources. The source stager is separately peer-reviewed.
No fixtures, seals, validation outputs or tensor/model artifacts are bundled as
a fresh admission. No source tool is a CPU encoder, head trainer or optimizer.

The copies live outside the compiled Make source closure. A future run still
needs the frozen card, actual CUDA admission, complete source capture and its
own independent reader triple before generation. Existing historical sources,
fixtures, capsules, validations and report bytes remain unchanged.

Run inside the project's existing managed Linux container with the repository
at `/embedding`; use the documented launcher/setup and internal SDK. Do not run
these tools on Windows, recreate a container, install a host toolchain or use a
CPU model fallback. Commands below begin from a clean checkout and the verified
container environment. They generate only new source/fixture evidence.

```bash
cd /embedding
mkdir -p output/runs/rpb-pooled-context/audit-tools
python3 -B code/evaluation/tools/pooled_context_v1/stage_source_tools.py \
  --stage-directory /embedding/output/runs/rpb-pooled-context/audit-tools/independent-pooled-context-staged-UNIQUE
```

Replace `UNIQUE` with a fresh name. The stager verifies every manifest source,
then creates the codec and blocked report-source paths only when absent. Exact
existing sources are reusable; conflicting, redirected or partial files cause
failure and remain preserved. It writes a fresh FALSE reader copy, runs only
that reader's artificial/source fixtures, creates a new seal, and records a
source-stage proof. It derives `release_staged_reader.py` by changing only the
reviewed release template's `ORIGINAL` and `PINS` declarations; every function
AST remains identical. It never invokes the release, generation or reporting
branches. A failed stage remains available for diagnosis; use another exclusive
name after resolving its cause.

`reader.paths` lists this new source/fixture/seal triple for the existing
launcher. In a clean checkout, publish it without overwriting another pointer:

```bash
test ! -e output/runs/rpb-pooled-context/reader-approved.paths
cp --no-clobber \
  output/runs/rpb-pooled-context/audit-tools/independent-pooled-context-staged-UNIQUE/reader.paths \
  output/runs/rpb-pooled-context/reader-approved.paths
```

The source-only stage is not a CUDA admission. Use the existing documented
`check-pooled-context.sh` and `evaluate-pooled-context` Make target only within
the fixed card's authorized scope. They capture the new FALSE triple, source,
actual test log and binary before generating fresh data. Do not import a past
fixture/seal as the new triple or substitute historical quality payloads.

After a **completed** new capsule and exact inventory are authorized, its staged
release utility can create a new TRUE reader copy with the two-line reverse-byte
proof. The complete original triple remains FALSE and unchanged:

```bash
python3 -B output/runs/rpb-pooled-context/audit-tools/independent-pooled-context-staged-UNIQUE/release_staged_reader.py \
  --authorized-completed-capsule /embedding/output/runs/rpb-pooled-context/pooled-context-NEW \
  --inventory-sha256 EXACT_COMPLETED_INVENTORY_SHA \
  --released-directory /embedding/output/runs/rpb-pooled-context/audit-tools/independent-pooled-context-released-NEW
```

The release reads only completion/inventory/source metadata and never runs an
audit. Invoke the released reader once with its explicit completed capsule and
exclusive output directory, as described by `--help`. A passing numerical
direction is separate from an audit PASS. Reporting tools require exact closed
completed-metadata SHA records and create separate one-flag releases; report
generation and durable copying are explicit later commands, not stage side
effects. Durable destinations must be absent and existing records are never
replaced. Read each tool's `--help` for its exact record fields and arguments.

Audit limits remain the frozen reader's limits: CPU saved arithmetic, typed
companions, named initialization/live-state witnesses, fixed query reductions,
saved logits/own argmax and source intervals are checked. CUDA checkpoint bodies
are byte-bound; actual CUDA admission establishes their association. No model,
autodiff, AdamW, head fit or PCA/SVD fit is independently rerun. No TEST/stress,
selection, promotion or broad generalization claim follows from these tools.
