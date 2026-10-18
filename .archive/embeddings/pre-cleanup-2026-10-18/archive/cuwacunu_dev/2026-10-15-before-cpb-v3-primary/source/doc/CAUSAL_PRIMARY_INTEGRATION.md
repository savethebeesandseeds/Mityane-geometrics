# Current provenance qualification — 15 October 2026

**The selected CPB-v1 artifact remains a pretrained historical reference.
Its runtime is causal, but its learned weights descend from noncausal training.
It is not a clean causal training baseline or a quality gate for new models.**
Read [the correction](CAUSAL_TRAINING_PROVENANCE_CORRECTION_2026-10-15.md).
Keep its source/checkpoint and existing runtime selection. Fresh causal training
belongs in cuwacunu_embedding, with no imported learned weights and no oracle
retention targets. Do not add competing implementations here before a verified
replacement is selected. Older baseline language below carries this qualification.

A fresh causal research baseline, CPB-v3.market-01, is now measured in the sibling embedding repository: all learned weights start fresh, with TRAIN-only normalization and new initial/final fixed heads. See its [review](../../cuwacunu_embedding/doc/CAUSAL_FRESH_V3_REVIEW.md). This does not change the selected dev checkpoint or add another active implementation.

---
<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# Causal encoder selected for cuwacunu_dev — 15 October 2026

The user explicitly selected **CPB-v1.market-01** from the
`causal_patch_bottleneck_mae` family for application integration and requested
removal of competing encoder implementations from the development repository.
This supersedes older instructions to keep RPB-v22 as the application primary.
Research comparisons, their original decisions and all historical results remain
preserved. Selecting the causal application encoder does not change those scores
or declare it an overall research winner.

## Current entry and artifact

| Item | Selection |
| --- | --- |
| Application API | `code/integration/include/cuwacunu/encoder/encoder.h`, `cuwacunu::encoder::primary` |
| Compatibility include | `cuwacunu/encoder/primary_encoder.h`, same implementation |
| Sole active family | `code/encoders/causal_patch_bottleneck_mae/` |
| Owned implementation | `versions/v1/`, unchanged evaluated bytes in both repositories |
| Geometry | C3 / H128 / F9 / P8, native36 = level32 + change4, CUDA:0 |
| Core settings | `config/encoder.conf`, exact copy of `versions/v1/config/v1.conf` |
| Runtime selection | `config/encoder_runtime.conf`, exact instance/checkpoint SHA required |
| Local checkpoint | `output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/inference.pt` |
| Checkpoint SHA-256 | `54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895` |
| Checkpoint size | 1,290,592 bytes; local and ignored by Git |
| Input scaler | `rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969` |
| Bank normalizer | `rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476` |

The historical `rpb-` strings are identities embedded in the imported frozen
normalization data. They do not select another encoder implementation.
Application loading uses the existing typed CPB artifact; it never imports an
old checkpoint, initializes a substitute or fits normalization on application data.
The checkpoint must accompany a fresh checkout separately from Git.

## Integration changes

The default Makefile includes only the causal version's encoder headers and the
application/data/configuration layers. The old encoder aliases, training drivers,
experiment extractors and alternate application settings were removed from active
development after exact source archival. Encoder-independent saved-feature
regression tools and unrelated latent-market/network work remain available.

The mmap dataset/loader now accepts an explicit Float64 feature option, preserving
original record doubles directly. Existing generic callers keep their Float32
default. The causal application loader requests Float64, unnormalized records,
actual timestamps and zero future horizon. This avoids silently rounding the raw
relative bank's observations through Float32 before encoding.

The new adapter requires explicit cutoffs and semantic channel IDs. It checks the
common actual timestamp grid, preserves row/per-feature observation masks and
passes raw values to CPB's cutoff boundary. Future sample fields are excluded.
Later values and masks are cropped before finiteness checks, normalization or
bank construction. Retained observed nonfinite values are rejected; unavailable
individual coordinates require false per-feature masks. Downstream consumers
must honor the returned validity marker. Native36 is returned without PCA.

The convenience CLI reads existing raw caches, requires three sources in
BTC/ETH/SOL semantic order and an actual common candle-close cutoff. CSV paths
require an already existing fresh raw cache; inference does not rebuild data.
See [the API and commands](../code/integration/README.md).

## Verification

All project builds and execution ran in the existing managed **cuwacunu** GPU
container, immutable ID
`82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e`.
The named session was `primary-causal-integration`:

```bash
bash code/scripts/task.sh primary-causal-integration -j2 test
```

The complete default build/test command exited successfully. Setup fixtures,
mmap/data precision, storage safety and core configuration checks passed.
The selected-checkpoint integration test passed **4,383 checks**, including:

- Exact original-double parity for 4,320 CACHE88 feature cells.
- Native36 CUDA output, full-history versus rolling-H128 parity and mixed cutoffs.
- Fourteen future-value/mask interventions, including nonfinite suffix poison.
- Explicit-cutoff requirements, input/configuration/artifact rejection cases,
  frozen checkpoint/scaler state and a legal-past positive control.

The test recorded 29 CUDA encoding attempts; some deliberately invalid inputs
are rejected before a forward. One additional CLI forward on the existing raw
BTC/ETH/SOL caches at close time `1706486399999` returned `valid=true` and a finite
36-coordinate vector. There were **zero encoder updates and zero head fits**.
No historical model training, quality scoring or fixed-head tuning was repeated.

The actual compiler dependencies contain 67 first-party source/header files
across six translation units. Both application translation units use the same
42 version-owned encoder headers. No retired encoder, archived source or sibling
checkout occurs in their build dependencies. Five resulting binaries are hashed.
After testing, concurrent latent-market work appended an optional Makefile
include. That addition was preserved. The exact tested Makefile is retained;
all other 72 captured files still match. A dry run of the current Makefile
confirms the same default five test commands without additional compilation.

Logs, the real-cache CLI vector and build/source/binary hashes are in
`output/engineering/primary-causal-integration/attempt-001/` (local/ignored).
The metadata-capture helper's first GCC dependency-parser failure and original
helper are retained there; correcting that parser required no model/test rerun.
The source-only [selection receipt](causal_primary_integration/selection.json)
binds these outputs and [preservation audit](causal_primary_integration/preservation.json).

The selected settings retain the evaluated file's exact bytes, including its
preexisting trailing blank lines. Whitespace verification excludes only the
`blank-at-eof` rule for that intentional exact copy.

## Historical source and artifacts

Before removal, the exact then-current integration source was copied to:

`C:\Work\cuwacunu\cuwacunu_embedding\archive\cuwacunu_dev\2026-10-15-before-causal-primary\`

For any retired relative path `P`, its exact old content is available at
`archive/cuwacunu_dev/2026-10-15-before-causal-primary/source/P` in the research
repository. This mapping also resolves old frozen receipts that name paths no
longer active in `cuwacunu_dev`. Restore a separately isolated historical source
tree if a future task explicitly requires reproducing an old run.

The archive contains **580 files / 4,423,528 source bytes**; every archive copy was
verified against its original SHA-256. Its manifest SHA-256 is
`72913640cddbd0e4e1b83548ef605fb6d5cfb625022d8c7fdefe9139a0c188ed`.
Only **556** archived source/config/script files were removed. The remaining
archived files include the old Makefile/settings/front-door docs that were
replaced, and retained historical evidence. The guarded removal script and exact
path/hash list are in [the migration directory](causal_primary_integration/).

No data, output directory, checkpoint or historical result was deleted. Existing
result/checkpoint entries are independently checked against the captured
4,552-entry baseline; the prior complete 80,105-entry inventory remains preserved.
The physical CPB-v1 implementation in both repositories is rechecked against
that earlier inventory. Research retains its encoder version collection, stopped
cards and all scoped leaders, including the native40/native36 references.

## Research status after integration

The [existing CPB result report](CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md) remains
the quality authority. It reports original market 9/9 and strict 8/8 mean floors,
but only 2/3 positive neural trend fits and 3/8 exact TEMPO-5 retention floors.
The application update adds correctness/integration evidence, not a new quality
experiment. CPB-v1 still uses imported V30 weights with zero fresh causal training.

Further improvement belongs in `cuwacunu_embedding` under a separately named
causal version, preserving this selected checkpoint and both market/hard-signal
guards. No new training or search was started by this integration task.
