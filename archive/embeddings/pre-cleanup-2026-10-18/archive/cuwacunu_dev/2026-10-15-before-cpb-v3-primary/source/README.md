# Current provenance qualification — 15 October 2026

**The selected CPB-v1 artifact remains a pretrained historical reference.
Its runtime is causal, but its learned weights descend from noncausal training.
It is not a clean causal training baseline or a quality gate for new models.**
Read [the correction](doc/CAUSAL_TRAINING_PROVENANCE_CORRECTION_2026-10-15.md).
Keep its source/checkpoint and existing runtime selection. Fresh causal training
belongs in cuwacunu_embedding, with no imported learned weights and no oracle
retention targets. Do not add competing implementations here before a verified
replacement is selected. Older baseline language below carries this qualification.

A fresh causal research baseline, CPB-v3.market-01, is now measured in the sibling embedding repository: all learned weights start fresh, with TRAIN-only normalization and new initial/final fixed heads. See its [review](../cuwacunu_embedding/doc/CAUSAL_FRESH_V3_REVIEW.md). This does not change the selected dev checkpoint or add another active implementation.

---
# Cuwacunu

Cuwacunu uses one active encoder: **CPB-v1.market-01**, from
**causal_patch_bottleneck_mae**. The user selected this strictly causal native36
instance for application integration on 15 October 2026. This operational choice
does not declare an overall research winner.

Use the application facade
[encoder.h](code/integration/include/cuwacunu/encoder/encoder.h), namespace
**cuwacunu::encoder::primary**. The complete encoder implementation lives in
[versions/v1](code/encoders/causal_patch_bottleneck_mae/versions/v1/).
[CAUSAL_PRIMARY_INTEGRATION.md](doc/CAUSAL_PRIMARY_INTEGRATION.md) is the current
selection and verification record. The existing
[research results](doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md) describe its
measured strengths and remaining trend/retention weaknesses.

## Selected settings and model

[config/encoder.conf](config/encoder.conf) fixes the evaluated geometry:
three semantic channels, 128 history steps, nine raw kline features, patch8,
parent32 plus four change coordinates, native36, and CUDA:0. Its time unit is
milliseconds, with a 60,000 ms candle-close grid. Geometry changes belong in
separately declared research; these settings are not a menu of serving models.

[config/encoder_runtime.conf](config/encoder_runtime.conf) separately selects
the exact instance and checkpoint path. Paths are relative to that runtime
configuration file. The local Git-ignored checkpoint is:

    output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/inference.pt

SHA-256:

    54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895

The file is 1,290,592 bytes. The facade verifies that hash and typed CPB metadata.
A missing or different checkpoint fails; there is no random initialization or
previous-encoder fallback. This saved model contains the frozen input scaler
and bank normalizer. It imports exact v30 weights with zero new causal updates.

## Causal input and inference

Load with **primary::load_selected()** and call **primary::encode_at** with raw
CPU float64 [B,3,N,9] values, boolean observation masks, explicit semantic
channel IDs, actual timestamps and one explicit cutoff per batch row.
Use [causal_encoder_input.h](code/integration/include/cuwacunu/integration/causal_encoder_input.h)
to adapt a raw sample_t; it requires genuine F64 source values and actual
past_keys, rather than reconstructed timestamps or F32 values upcast to F64.

The version-owned cutoff boundary removes later values and masks before value
validation, normalization, feature construction or pooling. It uses at most
the last 128 observations ending at each cutoff and blocks later temporal
attention. Return values include native36 coordinates, validity and cutoffs.
Never reuse an end-of-window embedding as an embedding for earlier times.
The application entry is frozen inference; it does not fit scalers or train.

The market CLI opens three distinct existing raw CACHE88 files read-only.
Sources follow semantic IDs 0,1,2: BTCUSDT, ETHUSDT, SOLUSDT. A CSV path requires
its existing fresh raw cache. The explicit cutoff must equal an actual candle
close; the CLI does not choose a nearby time or rebuild caches.

## Build and use

Follow [ENVIRONMENT.md](doc/ENVIRONMENT.md). Build and run only in the existing
GPU-enabled managed cuwacunu container. From this checkout in PowerShell:

~~~powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'causal-primary', '-j2', 'test')
~~~

The normal all, test, test-primary, test-encoder and test-integration targets
select the causal integration. Build products stay in the container. Use the
selection record for verification status; these commands describe entry points.

Inside the container at /cuwacunu, an explicit market inference example is:

~~~bash
bash code/scripts/task.sh causal-primary encode-history \
  'ARGS=--cutoff 1706486399999 --source data/mkt-hist-1/csv/BTCUSDT-1m-2024-01.cache.bin --source data/mkt-hist-1/csv/ETHUSDT-1m-2024-01.cache.bin --source data/mkt-hist-1/csv/SOLUSDT-1m-2024-01.cache.bin'
~~~

The command returns JSON containing the selected instance, cutoff, validity
and native36 coordinates. It requires the selected local checkpoint and source
files. Use encode_history --help on the built container binary for arguments.

## Source ownership and preserved research

Active code consists of the causal encoder, raw-data/config utilities,
application adapters, inference CLI and independent saved-feature readouts.
Previous RPB implementations, compatibility roots and research training
launchers have left the active development tree.

Their exact former dev source is preserved in the sibling research repository:
[archive entry](../cuwacunu_embedding/archive/cuwacunu_dev/2026-10-15-before-causal-primary/README.md)
and [per-file manifest](../cuwacunu_embedding/archive/cuwacunu_dev/2026-10-15-before-causal-primary/manifest.json).
The snapshot contains 580 files and 4,423,528 bytes; its source/ tree retains
the original repository-relative paths. The
[retirement receipt](doc/causal_primary_integration/retired-source.json)
records the 556 SHA-guarded removals. Historical docs, results, datasets and
ignored checkpoints retain their original locations and identities.

Encoder research continues in cuwacunu_embedding only when separately
requested. Keep every prior candidate and scoped leader. Reusable encoder
changes must be synchronized there; market loading and application selection
remain here. Each future causal version owns a complete independent
implementation. This update starts no training or quality comparison.

The local checkout is C:\Work\cuwacunu\cuwacunu_dev; the project/repository name
is **cuwacunu**. Its authorized remote is the private
[savethebeesandseeds/cuwacunu](https://github.com/savethebeesandseeds/cuwacunu).
Keep it private. New original code/documentation is all rights reserved under
[LICENSE](LICENSE); preserve inherited notices in [LICENSES/NOTICE](LICENSES/NOTICE).
