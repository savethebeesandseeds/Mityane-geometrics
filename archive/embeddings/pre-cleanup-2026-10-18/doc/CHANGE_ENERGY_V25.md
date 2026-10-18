<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# RPB-v25: balanced change-moment reconstruction

This is the local portable counterpart of the integration
[RPB-v25.market-01 card](../../cuwacunu_dev/doc/MARKET_CHANGE_ENERGY_CARD.md).
RPB-v22 stays primary. Preserve the leading research reference
RPB-v24.market-01.alt-01 and all earlier native40/native36 artifacts.
This new objective has its own design tag and typed archives; v24 weights are
never relabelled as v25.

The unchanged v24 encode/decode route preserves the trained v22 native32 and
appends four learned change coordinates. The export is native36, with24 learned
coordinates and12 fixed spectral coordinates. A zero-initialized, bias-free
change4-to270 auxiliary decoder is used only in training. It reconstructs the
existing signed-change/RMS bank from the same final visible input after query
and context masking. It adds1080 parameters and no exported coordinates.

At C3/H128/F9/W64 the full bundle has298505 parameters:15984 updated and282521
protected. Input scaling, the waveform residual decoder, AdamW options,
2048-new-update/B8 budget, data streams and fixed heads follow the card. The
matched start is the saved native36 initialization, which already contains
2048 inherited trained-parent updates. Both decoders begin at zero.

## Portable implementation

Files live under `code/encoders/raw_patch_bottleneck_mae`:

- `include/embedding/encoders/raw_patch_bottleneck_mae/energy_change_config.h`
  defines the distinct tag, layout and exact objective constants.
- `energy_change_model.h` in the same include directory wraps the unchanged
  separate-change model and adds the training-only moment decoder.
- `energy_change_moments.h` implements validity, frozen TRAIN RMS scaling and
  equal signed/energy group losses.
- `energy_change_checkpoint.h` stores typed inference, both normalizers and
  the full bundle; export remains36.
- `energy_change_training_state.h` adds typed optimizer/counter resume,
  protected-parent validation and warm start from the saved v24 initialization.
- `energy_change_settings.h` requires explicit `MODEL_TAG = RPB-v25`.
- `include/cuwacunu/encoder/energy_encoder.h` is the optional facade.
- `config/energy_changes.conf` and `tests/energy_change_test.cpp` match their
  integration counterparts exactly. The optional `energy_change.mk` provides
  `rpb-v25` and `test-rpb-v25` targets.

Only the integration fixture is executed for this milestone. The standalone
Make targets do not imply another encoder training run or quality measurement.
Market ingestion, orchestration, targets and fixed evaluation stay in integration.
The [54-pair receipt](../../cuwacunu_dev/doc/results/market_change_energy_v1/portable-sync-final.json)
checks all45 older pairs unchanged and all9 new pairs byte for byte.

## Objective and archive contract

Moment validity comes from at least two genuinely visible adjacent pairs,
including the original-time quarter support. Valid zero energy stays valid;
gaps are never bridged. The bank values retain the existing global and
quarter-over-global-RMS arithmetic. Targets are detached from gradients.

The auxiliary RMS normalizer is fitted once without centering on4096 full
natural-mask TRAIN anchors. This calls the moment bank and never the encoder.
It stores float64 sums of squares and scales, int64 counts, boolean support,
geometry and source identity. Validation, TEST and task labels do not enter it. The original input
scaler is unchanged. Empty support produces differentiable zero.

Each eligible example averages signed and energy Huber1 losses separately,
then combines them equally. The fixed objective is
`L_waveform + 0.1 * L_auxiliary`. Typed inference/resume bind the exact layouts,
coefficient, group weights, both scalers and completed update count. Resume
also binds parameter ordering, optimizer moments, deterministic counters and
source/split/recipe identities. Protected parent tensors have no gradients or
optimizer moments. Immutable writes reject existing filenames.

The model's `forward` retains the inherited waveform behavior. A training
caller explicitly adds `energy_change_moment_loss`; the integration runner
assembles and records the full v25 objective.

The existing-container CUDA fixture passed masking/poison/gaps, valid zeros,
independent scalar moment/support/RMS/loss calculations, exact initialized
native36 and protected prefix32, actual GPU learning, wrong-kind/identity
rejection and exact next-update resume for all losses, tensors and moments.
Its retained log is
[engineering-01.log](../../cuwacunu_dev/doc/results/market_change_energy_v1/engineering-01.log).

## Evidence and selection

The new quality trajectory completed on 15 October 2026 as
**RPB-v25.market-01**, with 2048 new CUDA updates/B8 after2048 inherited parent
updates. The full market gate **fails**: original conditions **7/9**, strict
native36 floors **6/8**. Both volatility heads fail; trend, activity and
relation improve over native36 v24. All three final neural trend scores are
positive. Keep RPB-v22 primary and RPB-v24.market-01.alt-01 the leading balanced
research reference. Preserve v25 as a stronger trend reference and stop this
exact objective without coefficient/rate/width/seed/budget/head rescue.

The following are new saved initial/final evaluations, with the v24 row reused
from its frozen report. Same4096 TRAIN anchors,112 TRAIN/32 known VALIDATION
readout windows; TRAIN Jan1–21 and VALIDATION Jan22–28, January2024. Evaluation
uses complete native36, ridge1 and neural16-tanh/Adam0.01/100 updates,
three repetitions2701/2802/2903 and148/660 head parameters. No old encoder
replay, old head refit or post-encoder PCA occurred. Neural scores average all
three fixed fits. R² is not classification accuracy.

### Past volatility

Dataset: **MKT-HIST-2 · complexity 4/5** · BTC/ETH/SOL January2024 minute
history, nine candle features, recent128 past-only volatility; intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v24.market-01.alt-01 | 36 | 0.446 | 0.393 | 100 |
| RPB-v25.market-01.init | 36 | 0.504 | 0.355 | 100 |
| RPB-v25.market-01 | 36 | 0.409 | 0.236 | 100 |

Strict required floors are0.6781069298378846 linear and0.3926141918082198
mean neural; both fail. The unchanged original v22 retention rule also fails
for both heads. Protected parent information remains exact; this result does
not prove why the jointly fitted heads generalize poorly.

### Past trend

Dataset: **MKT-HIST-2 · complexity 4/5** · BTC/ETH/SOL January2024 minute
history, nine candle features, recent128 past-only trend; intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v24.market-01.alt-01 | 36 | 0.655 | 0.134 | 100 |
| RPB-v25.market-01.init | 36 | -0.561 | -0.870 | 100 |
| RPB-v25.market-01 | 36 | 0.774 | 0.488 | 100 |

### Past activity

Dataset: **MKT-HIST-2 · complexity 4/5** · BTC/ETH/SOL January2024 minute
history, nine candle features, recent128 past-only activity; intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v24.market-01.alt-01 | 36 | 0.751 | 0.563 | 100 |
| RPB-v25.market-01.init | 36 | 0.745 | 0.608 | 100 |
| RPB-v25.market-01 | 36 | 0.755 | 0.598 | 100 |

### Past relation

Dataset: **MKT-HIST-2 · complexity 4/5** · BTC/ETH/SOL January2024 minute
history, nine candle features, recent128 past-only cross-channel relation;
intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v24.market-01.alt-01 | 36 | 0.341 | 0.350 | 100 |
| RPB-v25.market-01.init | 36 | 0.342 | 0.201 | 100 |
| RPB-v25.market-01 | 36 | 0.374 | 0.407 | 100 |

### Fixed-query reconstruction and execution

Dataset: **MKT-HIST-2 · complexity 4/5** · same nine-feature candle history,
fixed hidden-cell queries on112 TRAIN/32 known VALIDATION windows. Errors are
TRAIN-channel/feature-scaled MAE, not task-head errors.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v25.market-01.init | 0 | 0.278625 | 0.390528 | — |
| RPB-v25.market-01 | 2048 | 0.259617 | 0.388090 | 170.846 |

Updates above are new branch updates; inherited parent2048 is separate. GPU
training time includes checkpoint I/O. Auxiliary TRAIN calibration took2.192792s;
saved-feature head fitting/scoring/I/O took2.971975s, with pure fit time not
separately measured. Saved-evidence reporting/hash work took26.404274s.
The independent arithmetic audit passed31,851 checks. Initial full native36 and
both saved prefix32 banks match the historical controls exactly. All five new
checkpoints are local and ignored by Git; their paths, hashes, sizes, complete
scores/repetitions, identities and54-pair receipt are in the
[v25 registry](change_energy_embedding_versions.json) and
[frozen full report](../../cuwacunu_dev/doc/MARKET_CHANGE_ENERGY_RESULTS.md).

The market gate failed, so **v25 TEMPO-5 transfer was not executed**. Do not
infer synthetic retention from improved market trend or from v24's measured
transfer. Next propose a saved-TRAIN-only dimensionless-return versus raw-change
scale/energy representation diagnosis before declaring a separate branch
mechanism. This is a proposal, with no new experiment started here.

All new work remains local. The earlier native36/transfer publication request
does not authorize publishing this later objective. Checkpoints, features and
optimizer files stay ignored on the integration machine. No default change,
promotion, fresh chronological confirmation or TEST score follows automatically.
