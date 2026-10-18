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
# Cuwacunu instructions

## Current operational selection — 15 October 2026

The user explicitly selected **CPB-v1.market-01** as the only active application
encoder in cuwacunu_dev. Family: causal_patch_bottleneck_mae; native36;
exact saved v30 weights, zero new causal training updates. This operational
selection supersedes historical instructions to keep RPB-v22 primary. It does
not replace research leader records or declare an overall quality winner.

Read [README.md](README.md), the
[current integration record](doc/CAUSAL_PRIMARY_INTEGRATION.md),
[the causal family entry](code/encoders/causal_patch_bottleneck_mae/README.md)
and [the frozen quality report](doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md).
The integration record carries current verification status; do not infer a
passed runtime test from an existing research proof or from this selection.

## Application contract

- Use cuwacunu::encoder::primary from
  code/integration/include/cuwacunu/encoder/encoder.h. The model implementation
  is solely code/encoders/causal_patch_bottleneck_mae/versions/v1/.
- config/encoder.conf fixes the evaluated C3/H128/F9/P8/native36 geometry and
  60,000 ms close-time grid on CUDA:0. config/encoder_runtime.conf
  separately selects the instance/path/hash. Do not add model-selection toggles
  or silently change evaluated geometry.
- Selected local, ignored checkpoint:
  output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/inference.pt;
  SHA256 54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895,
  1,290,592 bytes. Preserve it and fail on missing/hash-mismatched artifacts;
  do not fall back to initialized weights, old RPB or another instance.
- Use frozen load_selected / encode_at inference with genuine raw CPU F64
  values, boolean masks, explicit semantic IDs, actual timestamp grids and an
  explicit cutoff for each row. No F32 upcast substitute, fabricated timestamps,
  silent channel alignment, future-to-past embedding reuse or scaler refitting.
  Crop future values and masks through the version-owned cutoff boundary.
- Market CLI: encode_history --cutoff CLOSE_TIME_MS and three ordered
  --source arguments for IDs 0/1/2 (BTC/ETH/SOL). Existing raw caches are read
  only; CSV paths require fresh existing caches. Never rebuild source data as
  an inference side effect.
- Keep data loading, encoder implementation, core settings, runtime artifact
  selection and evaluation separate. No PCA after native encoder exports.

## Preservation and research boundary

Only the causal family remains active here. Retired RPB source, old adapters,
research drivers, settings and top-level pointers were copied byte-exactly to
../cuwacunu_embedding/archive/cuwacunu_dev/2026-10-15-before-causal-primary/.
Its 580-file manifest and
[retirement receipt](doc/causal_primary_integration/retired-source.json)
map original paths to preserved bytes before 556 guarded source removals.
Do not reintroduce historical encoder roots or forwarding shims into live dev.

All historical results, cards, ledgers, source capsules, datasets and local
ignored checkpoints remain protected at their original locations. Never
relabel them as causal or rewrite their old protocol semantics. Keep RPB-v23,
both v24 widths, all 55 checkpoint pins and every scoped leader. The source
archive is provenance, not a newly advertised execution workspace.

The continuous research search remains stopped. This integration update
authorizes no new training, head tuning, quality rerun, automatic next experiment,
commit or push. Future encoder research belongs in cuwacunu_embedding when
requested; preserve current research leader records and the frozen CPB baseline.
Reusable changes must also be synchronized there. Every new causal version
owns complete copied implementation/config/tests; do not edit evaluated v1
arithmetic or share mutable implementation across versions. Application market
loading and selection stay integration-owned.

## Development and repository rules

Use C++20 and Makefiles. Build, test and run only in the existing managed
GPU-enabled cuwacunu Linux container; follow [ENVIRONMENT.md](doc/ENVIRONMENT.md).
Keep toolchains, caches and build products inside it. Preserve Docker objects
and local data; do not create replacement containers or install host toolchains.

Use bash code/scripts/task.sh SESSION ... . Normal primary checks are
test-primary; the default test includes primary, data/config and setup
fixtures. Run focused integration checks for integration changes; do not repeat
research comparisons. Separate new verification from reused quality evidence,
and make no test-pass claim before observing its result.

This local checkout is C:\Work\cuwacunu\cuwacunu_dev; repository name remains
cuwacunu. Its authorized GitHub remote is private
savethebeesandseeds/cuwacunu. Verify destination before any separately
authorized push; keep visibility and credentials unchanged. The earlier
native36 publication instruction is not standing authorization for later work.

New original code/documentation is all rights reserved under LICENSE.
Preserve inherited notices in LICENSES/NOTICE. Local synchronization does
not authorize publication to the public embedding remote. Preserve unrelated
changes, including doc/LATENT_MARKET_DYNAMICS_ROADMAP.md.

For new research measurements, follow the embedding reporting standard and
identify exact model, dataset, evidence stage and checkpoint. This source
integration changes operational selection, not measured quality.
