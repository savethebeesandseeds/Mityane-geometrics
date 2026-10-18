# Configurable RPB-v20 capacity experiment

This is a new experiment boundary for the portable v20 encoder. The CUDA
producer owns model training/checkpoint/native exports. The CPU reader owns
shared fixed heads and saved arithmetic. Neither implementation imports market
ingestion or rewrites historical quality protocols.

The prospective contract is
[`configurable_capacity_v1.md`](../../cards/configurable_capacity_v1.md).
TEMPO-5 uses three streams,128 candles and nine structured feature roles;
the observed slow lead/lag target has designed complexity5/5. The benchmark is
[`configurable_capacity`](../../../../../.archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/benchmarks/configurable_capacity). It is a
dimension-aware synthetic benchmark, not the market dataloader. Primary H128
and engineering H256 share the same period/delay law in original candle units.

Read the existing model-only configuration, then use closed profile overrides:
A/W64/D32 (`RPB-v20.alt-01`), B/W128/D32 (`RPB-v20.alt-02`),
C/W64/D64 (`RPB-v20.alt-03`). All other settings, data and budgets are fixed.
All three masters941101..103 are predetermined, with512 AdamW updates per run.
Baseline PCA is permitted on raw inputs only. Encoder exports go directly to
the common heads with no PCA between encoder and head.

Use the existing managed `cuwacunu_embedding` container. Source its documented
runtime profile; build products stay in its named internal build session:

```sh
source /etc/profile.d/embedding.sh
bash code/embeddings/evaluation/protocols/configurable_capacity_v1/admit.sh
# Use the exact new path printed by CAPACITY_ADMITTED:
bash code/embeddings/evaluation/protocols/configurable_capacity_v1/run_capacity.sh /embedding/output/runs/rpb-v20-capacity/admission-XXXXXX
```

Admission creates a unique capsule, builds via `code/scripts/task.sh`, captures
source/card/settings/binaries/SDK/device receipts, and runs generator, analytic
information, actual CUDA two-update/reload, and CPU saved-head engineering
checks. It constructs no quality model. Failures remain in their capsules.

The run generates the nine quality role archives once and checks all observed
information decisions before any quality model. It binds input bytes, runs all
nine CUDA trajectories once, saves initial/trained native arrays/checkpoints
and compact original query banks, and fits fixed CPU heads once per method,
master and repetition. The same fitted heads score intact and extra-deleted
validation. The source/binary lock prevents a concurrent rebuild in this session.

`readouts/report.json` retains scores, populations, source-group intervals and
paired differences. `arithmetic-audit.json` independently replays the saved
fits without training or encoder calls. `publish.py` audits saved decisions,
traces and both fixed query points using a pinned stdlib archive codec, then
writes the agreed tables in `report.md` and all master details in `summary.json`.
`artifacts.sha256` addresses the whole capsule; `complete.json` is written only
after all runs, fixed evaluation and audits succeed. Reuse existing evidence
for rescoring; do not run a CPU encoder or train a trajectory again to read it.

No TEST or market data are accessed. This experiment compares capacity at a
fixed short budget; improvement or failure does not automatically promote a
profile, alter cutoffs, or authorize an adaptive seed/budget extension.

The 12 October code review strengthens admission, immutable archive publication,
original-input and TRAIN-scaler associations, checkpoint settings, and paired
interval replay. Runtime checks remain active under `python -O`. Regression
targets are `test-configurable-capacity-data`,
`test-configurable-capacity-runtime`, and `test-configurable-capacity-audit`.
New native exports include source IDs and CPU scaler witnesses; older exports
remain auditable from their preserved checkpoint metadata, without model calls.

To audit a completed historical capsule, build the corrected tools in a separate
named session. Use a new existing review directory for the arithmetic output and
a new absent child directory for publisher output:

```sh
"$review_build/embedding_configurable_capacity_readout" \
  --audit-only "$original/readouts" --audit-output "$review/arithmetic-audit.json"
python3 -B code/embeddings/evaluation/protocols/configurable_capacity_v1/publish.py \
  --capsule "$original" --arithmetic-audit "$review/arithmetic-audit.json" \
  --output-directory "$review/published-audit"
```

This reads saved heads, predictions and tensors without fitting or running an
encoder. Keep the corrected auditor's source and binary fingerprints separate
from the original experiment's fingerprint, and verify the original artifact
manifest afterward. See [the review record](../../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CODE_REVIEW_2026-10-12.md).
