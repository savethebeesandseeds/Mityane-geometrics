<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# RPB-v28: raw-F64 relative change branch

No new experiment was run to write this guide. **RPB-v28.market-01 passes 7/9
original conditions and 4/8 strict native36 floors; its full market gate fails.**
Both volatility heads fail. All three final neural trend fits are positive and
coverage is 100%. Stop this exact recipe. RPB-v22 remains primary and
RPB-v24.market-01.alt-01 remains the 8/9 balanced research reference. Preserve
v26's neural-volatility head leader and every older task reference.

One CUDA trajectory completed 2048 new branch updates/B8 after 2048 inherited
parent updates on RTX A2000 8GB/cuda:0. Same 4096 TRAIN anchors, 112 TRAIN/32
known VALIDATION readout windows, input scaler, targets, queries and streams.
TRAIN Jan 1–21; VALIDATION Jan 22–28; Jan 29–31 unused. Three fixed head
repetitions measure variation on one encoder/split, not independent cohorts.
This is past-window representation evidence, not forecasting or TEST.

## What changed

The optional v28 module receives original raw-F64 observations and computes
adjacent symmetric-relative increments using max-abs-rescaled
`2*(a-b)/(|a|+|b|)`. Both-zero observed endpoints give valid zero; missing
endpoints give invalid zero. No gap bridging occurs. Each increment is rounded
to F32, then global/quarter means and RMS use F64 reductions before the final
F32 bank. The 270-coordinate layout, original-time quarters, minimum two pairs
and RMS floor 1e-6 stay fixed. Positive multiplicative units cancel in the bank;
the complete encoder retains parent levels and is not scale or offset invariant.

A new frozen TRAIN standardizer uses 4096 natural windows and zero encoder
calls: F32 bank values, CPU-F64 counts/sum/sumsquares, population mean/SD,
support two and floor 1e-6. Runtime F32 subtraction/division emits zero for
invalid/unsupported entries. It does not reuse v26's numeric normalizer.

The parent view casts raw to F32 before the existing CPU-F64 scaler transform
and CUDA-F32 output. The branch and parent share one final visible mask and
semantic channel IDs. This preserves the trained parent32 exactly. The learned
four-coordinate projection, bias-free zero-start waveform residual decoder
and waveform Huber1 objective stay fixed. No auxiliary objective, conditional
map, head tuning or post-encoder PCA is applied.

Geometry C3/H128/F9/P8/W64, canonical IDs 0/1/2, sampling interval 60000 ms.
Native36 is exact parent32 plus learned4. Parameters total 297425, optimizer
updated 14904, protected 282521. Initial encoder weights match the same saved
v24 initialization; full initial36 changes through the new relative bank.

## Separate task results

Fixed saved-bank heads use TRAIN-only normalization, ridge1 and neural16-tanh/
Adam0.01/100 updates, repetitions 2701/2802/2903. At D36 the heads have 148/660
parameters and seeds 13088411117524848790, 10018485049408248344 and
9083751155097190455. Native32 retains its original settings. All rows below are
reused saved measurements; the full ledger also retains v23/native40/v27.

### Past volatility

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only volatility · complexity 4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728107 | 0.380336 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.446315 | 0.392614 | 100 |
| RPB-v25.market-01 | 36 | 0.409235 | 0.236161 | 100 |
| RPB-v26.market-01 | 36 | 0.336490 | 0.419260 | 100 |
| RPB-v28.market-01.init | 36 | 0.455403 | 0.228191 | 100 |
| RPB-v28.market-01 | 36 | 0.333838 | -0.105883 | 100 |

### Past trend

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only trend · complexity 4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | -0.383904 | -0.542353 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.654911 | 0.134296 | 100 |
| RPB-v25.market-01 | 36 | 0.774228 | 0.488465 | 100 |
| RPB-v26.market-01 | 36 | 0.723823 | 0.275948 | 100 |
| RPB-v28.market-01.init | 36 | -0.078349 | -0.450465 | 100 |
| RPB-v28.market-01 | 36 | 0.684885 | 0.298249 | 100 |

### Past activity

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only activity · complexity 4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728431 | 0.512167 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.751430 | 0.563153 | 100 |
| RPB-v25.market-01 | 36 | 0.755266 | 0.597520 | 100 |
| RPB-v26.market-01 | 36 | 0.750337 | 0.667897 | 100 |
| RPB-v28.market-01.init | 36 | 0.729677 | 0.724605 | 100 |
| RPB-v28.market-01 | 36 | 0.748209 | 0.607966 | 100 |

### Past relation

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only relation · complexity 4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.388583 | 0.257996 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.340685 | 0.349583 | 100 |
| RPB-v25.market-01 | 36 | 0.373557 | 0.407150 | 100 |
| RPB-v26.market-01 | 36 | 0.343038 | 0.208225 | 100 |
| RPB-v28.market-01.init | 36 | 0.379086 | 0.194693 | 100 |
| RPB-v28.market-01 | 36 | 0.364335 | 0.315183 | 100 |

The [frozen integration report](../../cuwacunu_dev/doc/MARKET_RELATIVE_CHANGE_RESULTS.md)
retains all 80 task rows, raw controls, every neural repetition, physical-unit
MAE/RMSE and full-precision conditions. Initial neural activity 0.7246054298126288
remains an initialization control, not a trained-final task leader. No task is
averaged into one encoder accuracy.

## Reconstruction and cost

MKT-HIST-2 · BTC/ETH/SOL recent128 past-only history · complexity 4/5 · unchanged fixed hidden TRAIN/known-VALIDATION queries.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v28.market-01.init | 0 | 0.278625 | 0.390528 | — |
| RPB-v28.market-01 | 2048 | 0.274429 | 0.379518 | 124.340 |

Errors are original fixed-query TRAIN-CF-scaled MAE. Initial TRAIN/VALIDATION
0.27862535204206196/0.39052824676036835 match the saved parent; final
0.27442921165909084/0.379517525434494 improve without meeting the gates.
The 2048 inherited parent updates remain separate from the new branch budget.

TRAIN calibration 2.929513154 seconds includes input I/O and zero encoder calls.
GPU training 124.340118798 seconds includes checkpoints and loader parity,
excluding calibration and captures. Initial/final feature I/O took
0.933559986/0.748246063 seconds; fixed-head fitting/scoring/I/O took 2.598573382
seconds. Pure head-fit compute is unavailable. Every loaded raw batch is checked
against the historical F32 loader, and parent preprocessing includes CPU
rounding/scaling. These descriptive timings support no fair inference-speed claim.

The decoder has nonzero loss gradients on all 2048 updates; projection gradients
are nonzero on 2047. The zero decoder learns first. Exact parent values/buffers,
absent protected optimizer moments and exact typed next-update resume pass.

## Portable API and evidence

The optional facade is `cuwacunu::encoder::relative` in
`include/cuwacunu/encoder/relative_encoder.h`. Portable headers are
`relative_change_*.h` under the standalone raw-patch module; settings are
`config/relative_changes.conf`. Its input contract requires genuine raw F64.
Typed inference/TRAIN archives bind geometry, raw/bank family, recipes, input
scaler, frozen bank standardizer, optimizer handles and counters. The primary
API stays fixed. The application raw-record loader remains integration-owned.

All 83 reusable pairs match locally: 73 preserved plus 10 new. Only the
integration container ran the CUDA fixture and quality trajectory. The first
producer compile failed on a missing quote in its summary JSON source; exact
failed source/helper and build01 are preserved with successful build02. No model
execution or calibration occurred in that compile failure. Saved arithmetic
passes 1,549,413 checks. Report verification checks 184 asset bindings, 277 durable
members and 72 unchanged old plus 8 new rows. All 45 indexed checkpoints were
rehashed and confirmed available, Git-ignored and untracked. The earlier
77,831-file preservation receipts remain bound; this guide performs no broad scan.

- [Prospective card](../../cuwacunu_dev/doc/MARKET_RELATIVE_CHANGE_CARD.md)
- [Full frozen result](../../cuwacunu_dev/doc/MARKET_RELATIVE_CHANGE_RESULTS.md)
- [Exact instance registry](raw_relative_change_embedding_versions.json)
- [Live search ledger](ENCODER_SEARCH_LEDGER.md)
- [Completion and checkpoint pins](../../cuwacunu_dev/doc/results/market_relative_changes_v1/completion.json)

All five checkpoints, raw/bank/decoder witnesses, native banks and optimizer
archives remain local and ignored in integration. Durable docs contain compact
metadata, exact hashes and captured source. Verify availability/hashes before
reuse; Git alone cannot recover the weights.

## Next direction

No v28 TEMPO-5 transfer ran after market failure. Stop its exact bank mechanism
without normalizer, head, rate, width, seed or budget rescue. Keep native36
balanced, v26's neural-volatility leader and all other task references.

The separate [v29 card](../../cuwacunu_dev/doc/MARKET_PARENT_MODULATED_DECODER_CARD.md)
reuses the exact relative bank and saved normalizer, adding a bias-free zero-start
parent-context gain to the waveform residual decoder. Fresh saved v24
initialization is required; no trained-v28 warm start. Native36 and 2048/B8 stay
fixed, with zero new calibration fits/forwards. After exact full36 parity with
saved v28 initialization, reuse initial fixed-head results and fit only final
heads. Admission/quality are pending. No primary promotion, commit, push or
publication is implied.
