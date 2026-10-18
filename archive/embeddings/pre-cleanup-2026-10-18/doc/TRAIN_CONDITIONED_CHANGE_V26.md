<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# RPB-v26: TRAIN-conditioned change bank

RPB-v26.market-01 is measured and stopped at its declared budget: original
market conditions8/9, strict native36 protection5/8. RPB-v22 remains primary;
RPB-v24.market-01.alt-01 remains the balanced research reference. V26 preserves
a new individual neural-volatility reference, R²0.41926036121396154, while its
linear-volatility R²0.33649049583839497 fails the0.6781069298378846 floor.
Strict linear activity and neural relation also fail. Keep every v26 checkpoint
and the stronger earlier task/head references in the mirrored search ledger.

The new optional module uniformly subtracts frozen TRAIN means and divides by
frozen TRAIN population standard deviations inside the unchanged270-coordinate
change bank, before its learned four-coordinate projection. It concatenates
the complete trained v22 native32 prefix exactly, giving native36, and uses the
same separate waveform residual decoder. It adds no auxiliary objective or
information. Centering changes the implicit offset/prior of the bias-free
branch; this is not an arithmetically neutral rewrite of v24.

The admitted geometry is C3/H128/F9/P8/W64, with14904 optimizer-updated and282521
protected parameters (297425 total). Calibration used4096 natural-mask TRAIN
anchors, zero encoder calls, with per-coordinate valid counts/sums/sumsquares
in CPU F64. Count>=2 defines support; population std floor1e-6; invalid or
unsupported entries become literal0. Valid zeros remain valid. Its complete
frozen standardizer is part of the new typed inference/TRAIN state, with exact
source, input-scaler, geometry, layout and content identity validation.

One new CUDA trajectory completed2048/B8 after2048 inherited parent updates.
Same input scaler/data/targets/masks/streams and waveform loss; fixed saved-bank
ridge1 and neural16-tanh/Adam0.01/100-step readouts, three repetitions, no PCA.
GPU training123.391642384s includes checkpoint I/O; calibration3.268507533s is
separate. Fixed-query TRAIN/VALIDATION standardized MAE ends at
0.273081120635782/0.38193804025650024. Better reconstruction is not a full gate win.

See the frozen integration [per-task result and physical errors](../../cuwacunu_dev/doc/MARKET_CONDITIONED_CHANGE_RESULTS.md),
[prospective card](../../cuwacunu_dev/doc/MARKET_CONDITIONED_CHANGE_CARD.md),
[exact local version index](conditioned_change_embedding_versions.json), and
[live search ledger](ENCODER_SEARCH_LEDGER.md). MKT-HIST-2 is BTC/ETH/SOL
January2024 minute history, recent128 past-only targets, designed complexity4/5;
112 TRAIN/32 known VALIDATION readout windows. This is known-source development,
not forecasting, TEST or a fresh chronological confirmation. No v26 TEMPO-5
transfer ran because the full market gate failed.

The artificial GPU fixture passed its first invocation: scalar bank/normalizer
oracles, masks/poison/gaps/order/partial history, exact parent/reconstruction,
actual decoder then projection learning, no protected optimizer state, immutable
typed archives and exact next-update resume. The saved arithmetic audit passed
1,280,472 checks. All64 reusable source pairs (54 previous plus10 new) match
locally. The optional `conditioned_change.mk` is a context-specific build adapter;
it is not claimed byte-identical to the integration Make fragment. Only the
integration container ran the fixture; no separate sibling build is claimed.

Headers live under `code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_*.h`;
the portable facade is `include/cuwacunu/encoder/conditioned_encoder.h`, and core
settings are `config/conditioned_changes.conf` under that module. The primary
selection and historical targets stay fixed. Checkpoints and native arrays remain local ignored files in integration; Git alone cannot recover
them. Verify the recorded hashes before reuse.

The continuing goal now explores the separately declared RPB-v27.market-01
parent-conditional bank under its own card, before any admission or quality.
V26's exact mechanism stays stopped: no coefficient/rate/width/seed/budget/head
rescue. This later work remains local and unpublished.
