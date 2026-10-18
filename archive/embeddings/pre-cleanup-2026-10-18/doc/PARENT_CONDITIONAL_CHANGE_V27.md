<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# RPB-v27: parent-conditional change branch

This guide reuses the completed experiment; no new experiment was run to write
it. **RPB-v27.market-01 fails the full market gate: original5/9 and strict
native36 protection2/8, with all three final neural trend fits negative and100%
coverage.** Stop its exact mechanism. RPB-v22 remains primary and
RPB-v24.market-01.alt-01 remains the8/9 balanced research reference. V26 retains
the neural-volatility head record0.41926036121396154; keep every older task record.

One CUDA trajectory completed2048 new branch updates/B8 after2048 inherited
parent updates on RTX A2000 8GB/cuda:0. Same4096 TRAIN anchors,112 TRAIN/32 known
VALIDATION readout windows, input scaler, targets, queries and streams. TRAIN
Jan1–21; VALIDATION Jan22–28; Jan29–31 unused. Three fixed neural repetitions
measure head variation on one encoder/split, not independent market cohorts.
This is known chronological representation evidence, not forecasting or TEST.

## What changed

The optional v27 module subtracts a frozen unlabeled TRAIN prediction of the
existing conditioned270-coordinate adjacent-change bank from parent32, then
projects the residual to four learned coordinates. Native36 concatenates the
exact parent prefix. The separate bias-free waveform residual decoder and
waveform Huber1 objective stay unchanged. No task labels, auxiliary objective,
residual restandardization, clipping, head tuning or post-encoder PCA.

It reuses v26's exact frozen bank standardizer. A new map uses4096 natural-mask
TRAIN windows and512 necessary frozen CUDA parent forwards with zero calibration
updates. CPU F64 parent mean/population-SD normalization has floor1e-6. Each
supported bank coordinate uses ridge1 on32 parent slopes and an unpenalized
intercept, requiring count>=2. Runtime normalization, map multiplication and
bank correction use F32. Invalid/unsupported entries emit literal zero.

This is approximate regularized residualization, without an orthogonality or
causal explanation of validation loss. The affine-span equivalence applies
only with complete valid banks and a fixed projection; varying masks change it.

Geometry C3/H128/F9/P8/W64, canonical IDs0/1/2 and sampling interval60000ms.
Total encoder parameters297425;14904 receive optimizer updates and282521 parent
parameters are protected. The frozen map adds8910 coefficients:71,280 bytes in
F64 or35,640 bytes after F32 conversion. All frozen map tensors, including
statistics, occupy2,571,950 bytes. They are not trainable encoder parameters.

## Separate task results

Fixed saved-bank heads use TRAIN-only normalization, ridge1 and neural16-tanh/
Adam0.01/100 updates, repetitions2701/2802/2903. At D36 they have148/660 parameters
and uint64 seeds13088411117524848790,10018485049408248344,9083751155097190455.
Native32 uses its original different counts/seeds. Older rows below are reused.
The full ledger separately retains v23/native40 and the frozen-v23 alternative.

### Past volatility

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only volatility · complexity4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728107 | 0.380336 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.446315 | 0.392614 | 100 |
| RPB-v25.market-01 | 36 | 0.409235 | 0.236161 | 100 |
| RPB-v26.market-01 | 36 | 0.336490 | 0.419260 | 100 |
| RPB-v27.market-01.init | 36 | 0.269641 | 0.275640 | 100 |
| RPB-v27.market-01 | 36 | 0.523796 | 0.192784 | 100 |

### Past trend

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only trend · complexity4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | -0.383904 | -0.542353 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.654911 | 0.134296 | 100 |
| RPB-v25.market-01 | 36 | 0.774228 | 0.488465 | 100 |
| RPB-v26.market-01 | 36 | 0.723823 | 0.275948 | 100 |
| RPB-v27.market-01.init | 36 | 0.182454 | -0.390806 | 100 |
| RPB-v27.market-01 | 36 | 0.117594 | -0.155343 | 100 |

### Past activity

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only activity · complexity4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728431 | 0.512167 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.751430 | 0.563153 | 100 |
| RPB-v25.market-01 | 36 | 0.755266 | 0.597520 | 100 |
| RPB-v26.market-01 | 36 | 0.750337 | 0.667897 | 100 |
| RPB-v27.market-01.init | 36 | 0.732908 | 0.543634 | 100 |
| RPB-v27.market-01 | 36 | 0.741773 | 0.621087 | 100 |

### Past relation

MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only relation · complexity4/5 · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.388583 | 0.257996 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.340685 | 0.349583 | 100 |
| RPB-v25.market-01 | 36 | 0.373557 | 0.407150 | 100 |
| RPB-v26.market-01 | 36 | 0.343038 | 0.208225 | 100 |
| RPB-v27.market-01.init | 36 | 0.328523 | 0.204915 | 100 |
| RPB-v27.market-01 | 36 | 0.360892 | 0.268986 | 100 |

The [frozen integration report](../../cuwacunu_dev/doc/MARKET_PARENT_CONDITIONAL_CHANGE_RESULTS.md)
retains all72 task rows, raw controls, every neural repetition, physical-unit
MAE/RMSE and full-precision condition differences. No tasks are averaged into
one encoder accuracy.

## Reconstruction and cost

MKT-HIST-2 · BTC/ETH/SOL recent128 past-only history · complexity4/5 · unchanged fixed hidden TRAIN/known-VALIDATION queries.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v27.market-01.init | 0 | 0.278625 | 0.390528 | — |
| RPB-v27.market-01 | 2048 | 0.274922 | 0.388020 | 148.800 |

Errors are original fixed-query TRAIN-CF-scaled MAE. Initial TRAIN/VALIDATION
0.27862535204206196/0.39052824676036835 match the saved parent exactly; final
0.27492174186876844/0.3880198262631893 improve slightly without meeting the gates.
The2048 inherited parent updates remain separate from the new branch budget.

GPU training148.800206036s includes checkpoint I/O and excludes calibration/
captures. Parent calibration14.210417854s and map fit1.730320335s include I/O.
Initial/final feature extraction and I/O0.914576875s/0.872558284s; fixed saved-head
fitting/scoring/artifact I/O2.690933644s. Pure head-fit compute is unavailable;
audit and query-capture times remain separately recorded in the full evidence.

Actual CUDA learning includes2048 decoder-gradient and2047 nonzero projection-
gradient updates, with2048 weight changes in each branch tensor. The zero
decoder learns first, then projection loss gradients appear. Parent values/
buffers remain exact, with undefined gradients and no optimizer moments.

## Portable API and preservation

The optional facade is `cuwacunu::encoder::parent_conditional` in
`include/cuwacunu/encoder/parent_conditional_encoder.h`. Headers are under
`code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_*.h`;
settings are `code/encoders/raw_patch_bottleneck_mae/config/parent_conditional_changes.conf`.
Typed inference/TRAIN archives bind geometry, source/split/recipe, input scaler,
the frozen normalizer/map, optimizer mappings and counters. The primary API stays fixed.

All73 reusable pairs match locally:64 preserved plus9 new. The standalone Make
fragment is an environment-specific adapter, not a claimed identical integration
file. Only the integration container executed the CUDA fixture and experiment.
The failed first fixture assertion and passed second attempt are preserved.
Numerical cross-layout permutation checks and exact direct bank/map/projection
arithmetic pass; model headers and producer did not change for that correction.

Saved arithmetic passes2,716,060 checks. Separate report verification checks184
asset bindings/166 durable members and all64 old plus8 new rows. The ledger append
preserves all seven earlier entries and40 ignored checkpoints. No broad two-repo
scan was repeated; previous receipts bind the77,831-file baseline honestly.

- [Prospective card](../../cuwacunu_dev/doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md)
- [Full frozen report](../../cuwacunu_dev/doc/MARKET_PARENT_CONDITIONAL_CHANGE_RESULTS.md)
- [Exact local instance registry](parent_conditional_change_embedding_versions.json)
- [Live search ledger](ENCODER_SEARCH_LEDGER.md)
- [Completion and checkpoint pins](../../cuwacunu_dev/doc/results/market_parent_conditional_changes_v1/completion.json)

The five checkpoints, map coefficients/statistics, native banks, witnesses and
optimizer archives remain local and ignored in integration. Only compact metadata
and hashes enter docs; Git alone cannot recover the weights. Verify hashes before reuse.

## Next direction

No v27 TEMPO-5 transfer ran after market failure. Stop this exact map without
penalty, normalizer, head, rate, width, seed or budget rescue. Keep native36
balanced and every task/head leader.

The separately declared [RPB-v28 card](../../cuwacunu_dev/doc/MARKET_RELATIVE_CHANGE_CARD.md)
uses symmetric-relative adjacent increments from raw F64 levels and a new TRAIN
bank standardizer, waveform-only native36 and unchanged2048/B8. Engineering
admission/calibration/quality are pending. It is a separate mechanism, not a
measured result or v27 extension. No primary promotion, commit, push or
publication is implied.
