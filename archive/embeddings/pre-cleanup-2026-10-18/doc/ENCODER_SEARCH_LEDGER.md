# Encoder search ledger

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

RPB-v30.market-01 passes the complete market gate; separately verified frozen TEMPO-5 transfer remains pending. RPB-v22 remains primary; RPB-v24.market-01.alt-01 remains the balanced research reference.
All10 earlier candidate entries are preserved semantically unchanged. Every checkpoint, failed condition and useful per-task reference remains available.
This ledger reuses verified saved measurements. It executes no encoder, deserializes no tensor, fits no head/scaler and performs no Git commit, push or publication.

This Markdown and the exact JSON are identical local mirrors in both repositories. Evidence paths resolve from C:/Work/cuwacunu/cuwacunu_dev. Checkpoints and frozen calibration weights remain local ignored files.

## Matched final checkpoints

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · intact known chronological VALIDATION.
Same C3/H128/F9/P8/W64, original TRAIN CF scaler,4096 TRAIN anchors,112 TRAIN/32 known-VALIDATION probes. TRAIN Jan1–21; VALIDATION Jan22–28; Jan29–31 unused.
Core methods: RPB-v22.market-02, RPB-v23.market-01, RPB-v24.market-01, RPB-v24.market-01.alt-01, RPB-v25.market-01, RPB-v26.market-01, RPB-v27.market-01, RPB-v28.market-01, RPB-v29.market-01, RPB-v30.market-01. Native widths, width-derived head seeds/costs and inherited exposure differ; this comparison describes known development evidence.
Latest mechanism: Unchanged raw-relative native36 and V28 normalizer; unmodulated mean plus change-conditioned diagonal Gaussian waveform variance and proper NLL64.

### past volatility

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · past volatility · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728107 | 0.380336 | 100 |
| RPB-v23.market-01 | 32 | 0.654630 | 0.259498 | 100 |
| RPB-v24.market-01 | 40 | 0.599739 | 0.324488 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.446315 | 0.392614 | 100 |
| RPB-v25.market-01 | 36 | 0.409235 | 0.236161 | 100 |
| RPB-v26.market-01 | 36 | 0.336490 | 0.419260 | 100 |
| RPB-v27.market-01 | 36 | 0.523796 | 0.192784 | 100 |
| RPB-v28.market-01 | 36 | 0.333838 | -0.105883 | 100 |
| RPB-v29.market-01 | 36 | 0.603912 | 0.089783 | 100 |
| RPB-v30.market-01 | 36 | 0.953913 | 0.588277 | 100 |

### past trend

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · past trend · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | -0.383904 | -0.542353 | 100 |
| RPB-v23.market-01 | 32 | 0.387131 | -0.166048 | 100 |
| RPB-v24.market-01 | 40 | 0.599060 | 0.066226 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.654911 | 0.134296 | 100 |
| RPB-v25.market-01 | 36 | 0.774228 | 0.488465 | 100 |
| RPB-v26.market-01 | 36 | 0.723823 | 0.275948 | 100 |
| RPB-v27.market-01 | 36 | 0.117594 | -0.155343 | 100 |
| RPB-v28.market-01 | 36 | 0.684885 | 0.298249 | 100 |
| RPB-v29.market-01 | 36 | -0.664295 | -0.438662 | 100 |
| RPB-v30.market-01 | 36 | 0.794366 | 0.320210 | 100 |

### past activity

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · past activity · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728431 | 0.512167 | 100 |
| RPB-v23.market-01 | 32 | 0.720187 | 0.564187 | 100 |
| RPB-v24.market-01 | 40 | 0.764260 | 0.678189 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.751430 | 0.563153 | 100 |
| RPB-v25.market-01 | 36 | 0.755266 | 0.597520 | 100 |
| RPB-v26.market-01 | 36 | 0.750337 | 0.667897 | 100 |
| RPB-v27.market-01 | 36 | 0.741773 | 0.621087 | 100 |
| RPB-v28.market-01 | 36 | 0.748209 | 0.607966 | 100 |
| RPB-v29.market-01 | 36 | 0.825582 | 0.635718 | 100 |
| RPB-v30.market-01 | 36 | 0.865732 | 0.613572 | 100 |

### past relation

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · past relation · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.388583 | 0.257996 | 100 |
| RPB-v23.market-01 | 32 | 0.447747 | 0.329479 | 100 |
| RPB-v24.market-01 | 40 | 0.328862 | -0.017315 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.340685 | 0.349583 | 100 |
| RPB-v25.market-01 | 36 | 0.373557 | 0.407150 | 100 |
| RPB-v26.market-01 | 36 | 0.343038 | 0.208225 | 100 |
| RPB-v27.market-01 | 36 | 0.360892 | 0.268986 | 100 |
| RPB-v28.market-01 | 36 | 0.364335 | 0.315183 | 100 |
| RPB-v29.market-01 | 36 | 0.453961 | 0.401638 | 100 |
| RPB-v30.market-01 | 36 | 0.487476 | 0.406836 | 100 |

## Own frozen decisions and exposure

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · counts belong to each own card; older recipes are never rescored under a later gate.

| Method | Native size | Own conditions | New/inherited updates | Status |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | No nine-condition score | 2048/0 | primary architecture parent/reference; no retroactive nine-condition score |
| RPB-v23.market-01 | 32 | 5/8 | 2048/0 | stopped measured recipe; useful native32 trend/relation reference |
| RPB-v23.market-01.alt-01 | 32 | 4/8 | 2048/2048 | stopped measured frozen-parent recipe; separate preserved reference |
| RPB-v24.market-01 | 40 | 5/9 | 2048/2048 | stopped native40 recipe; useful trend/activity reference |
| RPB-v24.market-01.alt-01 | 36 | 8/9 | 2048/2048 | leading balanced research reference; original8/9 with linear-volatility failure |
| RPB-v25.market-01 | 36 | 7/9 | 2048/2048 | stopped measured objective; stronger trend/activity/relation reference; both volatility heads fail |
| RPB-v26.market-01 | 36 | 8/9 | 2048/2048 | stopped exact conditioned-bank mechanism; preserved task references |
| RPB-v27.market-01 | 36 | 5/9 | 2048/2048 | stopped exact declared mechanism; preserved task references |
| RPB-v28.market-01 | 36 | 7/9 | 2048/2048 | stopped exact declared mechanism; preserved task references |
| RPB-v29.market-01 | 36 | 4/9 | 2048/2048 | stopped exact declared mechanism; preserved task references |
| RPB-v30.market-01 | 36 | 9/9 | 2048/2048 | market gate passed; separate conditional transfer pending; no balanced promotion |

Latest strict native36 floors:8/8; all three final neural trend fits positive:True; full32/32 coverage:True.
Old parent N/A, fresh/frozen-v23 eight-condition decisions, native40/native36/v25/v26 outcomes and per-task leaders remain intact. Three fixed neural repetitions on one checkpoint are head variation, not independent market cohorts.

## Reconstruction and training cost

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · unchanged fixed hidden TRAIN/known-VALIDATION query MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 2048 | 0.278625 | 0.390528 | 110.059 |
| RPB-v23.market-01 | 2048 | 0.310953 | 0.346302 | 130.636 |
| RPB-v23.market-01.alt-01 | 2048 | 0.265508 | 0.366152 | 94.655 |
| RPB-v24.market-01 | 2048 | 0.259922 | 0.382801 | 69.688 |
| RPB-v24.market-01.alt-01 | 2048 | 0.263832 | 0.384721 | 93.377 |
| RPB-v25.market-01 | 2048 | 0.259617 | 0.388090 | 170.846 |
| RPB-v26.market-01 | 2048 | 0.273081 | 0.381938 | 123.392 |
| RPB-v27.market-01 | 2048 | 0.274922 | 0.388020 | 148.800 |
| RPB-v28.market-01 | 2048 | 0.274429 | 0.379518 | 124.340 |
| RPB-v29.market-01 | 2048 | 0.265796 | 0.382868 | 159.606 |
| RPB-v30.market-01 | 2048 | 0.282098 | 0.390187 | 131.756 |

All trajectories use B8 on RTX A2000 8GB/cuda:0. Errors are original fixed-query TRAIN-CF-scaled MAE. GPU training includes checkpoint I/O and excludes calibration/extraction/head fitting; inherited exposure is separate.
The latest result separately reports necessary frozen calibration forwards, map fitting, reused statistics, extraction, fixed saved-head fitting/scoring/I/O and audit durations. Frozen calibration coefficients are storage, not trainable encoder parameters.

## Fixed head costs and task leaders

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · TRAIN-only normalizers; ridge1; neural16-tanh/Adam0.01/100 steps; repetitions2701/2802/2903.

| Native size | Ridge parameters | Neural parameters | Exact uint64 seeds for2701/2802/2903 |
| --- | ---: | ---: | ---: |
| 32 | 132 | 596 | 15952039642448263928, 10925781080001991545, 14020430036991234725 |
| 36 | 148 | 660 | 13088411117524848790, 10018485049408248344, 9083751155097190455 |
| 40 | 164 | 724 | 6049515622977958931, 8262941649267994858, 2095371932330426401 |

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · Core10 final checkpoints · intact known VALIDATION · descriptive task/head maxima.

| Task | Head | Method | R² |
| --- | ---: | ---: | ---: |
| past_volatility | linear_r2 | RPB-v30.market-01 | 0.95391286529698172 |
| past_volatility | neural_mean_r2 | RPB-v30.market-01 | 0.58827656467282696 |
| past_trend | linear_r2 | RPB-v30.market-01 | 0.79436550162468511 |
| past_trend | neural_mean_r2 | RPB-v25.market-01 | 0.48846466053793119 |
| past_activity | linear_r2 | RPB-v30.market-01 | 0.86573187689252273 |
| past_activity | neural_mean_r2 | RPB-v24.market-01 | 0.67818940135179373 |
| past_relation | linear_r2 | RPB-v30.market-01 | 0.48747626119524834 |
| past_relation | neural_mean_r2 | RPB-v25.market-01 | 0.4071495455067296 |

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5 · All11 preserved final records including frozen-v23 alternative · intact known VALIDATION · descriptive task/head maxima.

| Task | Head | Method | R² |
| --- | ---: | ---: | ---: |
| past_volatility | linear_r2 | RPB-v30.market-01 | 0.95391286529698172 |
| past_volatility | neural_mean_r2 | RPB-v30.market-01 | 0.58827656467282696 |
| past_trend | linear_r2 | RPB-v30.market-01 | 0.79436550162468511 |
| past_trend | neural_mean_r2 | RPB-v25.market-01 | 0.48846466053793119 |
| past_activity | linear_r2 | RPB-v30.market-01 | 0.86573187689252273 |
| past_activity | neural_mean_r2 | RPB-v24.market-01 | 0.67818940135179373 |
| past_relation | linear_r2 | RPB-v30.market-01 | 0.48747626119524834 |
| past_relation | neural_mean_r2 | RPB-v25.market-01 | 0.4071495455067296 |

Core eight-axis Pareto frontier: RPB-v24.market-01, RPB-v25.market-01, RPB-v26.market-01, RPB-v27.market-01, RPB-v29.market-01, RPB-v30.market-01. Exact saved scalar comparisons describe tradeoffs and do not select a universal winner.
The frozen-parent v23 alternative stays outside the core scope and inside the broader maxima. V25 and v26 partial leaders remain visible even when a future balanced candidate supersedes them.

## Protection and next direction

Candidate records are append-only. Corrections append linked records. Preserve all previous leader views, exact own criteria, physical errors, every neural repetition, cards/reports, initialization and recovery checkpoints.
Changing the balanced pointer requires all original9 conditions, strict native36 eight floors, all three trend repetitions positive, full coverage and separately verified compatible TEMPO-5 retention under both unchanged input-scaler policies. This market-only append changes no balanced or primary pointer.
Only the declared absolute1e-12 arithmetic tolerance applies to quality. Per-task maxima do not silently become new mandatory floors. No stopped map/normalizer/loss/head/rate/width/seed/budget rescue or failed-market transfer.
Next measure the exact frozen checkpoint on the declared compatible TEMPO-5 cohorts with every new calibration state frozen. Transfer is pending; no new balanced selection is recorded.
Fresh chronology, TEMPO-4 and AMP-2 remain separate; no forecasting, TEST or universal-best claim follows.

## Evidence

- [Exact machine ledger](encoder_search_ledger.json)
- Latest report: doc/MARKET_LOCATION_SCALE_DECODER_RESULTS.md SHA256 f5519bc661fb5072a74f36ebe4258c25862ebfb5c13117056ab000533a296ff8
- Latest card: doc/MARKET_LOCATION_SCALE_DECODER_CARD.md SHA256 f620298e9ecf9f73a32ae61737b67295f8dcde709416aa3d6ed16feee613e45c
- Previous JSON: doc/results/market_location_scale_decoder_v1/ledger-update-before.json SHA256 bf0087116423b018ee2b11b042a9363939d492ae346034c51a2a97d3e03224a4
- Previous Markdown: doc/results/market_location_scale_decoder_v1/ledger-update-before.md SHA256 d699853a8ee2f552b1bbdf8e1753b1f184d6730b16505f200299a9655f02d6d9

All55 candidate checkpoints were byte-hash checked and remain Git-ignored. This update performs no broad historical scan, deserialization or weight export. Earlier preservation receipts stay bound. The unrelated roadmap remains untouched.
