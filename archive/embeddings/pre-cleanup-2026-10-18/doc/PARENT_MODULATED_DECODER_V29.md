<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# RPB-v29: parent-modulated waveform residual decoder

No new experiment was run to write this guide. **RPB-v29.market-01 fails the full market gate: original 4/9 and strict native36 protection 4/8.** Volatility and trend fail; all three final neural trend fits are negative and coverage is 100%. Stop this exact mechanism. RPB-v22 stays primary and RPB-v24.market-01.alt-01 stays the 8/9 balanced research reference.

V29 sets two new linear-head records: activity 0.8255824303229565 and relation 0.4539605718451353. Neural-head leaders remain unchanged, including v26 volatility, v25 trend/relation and native40 activity. These are scoped task records, without an overall-winner claim.

One CUDA trajectory completed 2048 new updates/B8 after 2048 inherited parent updates on RTX A2000 8GB/cuda:0. Same 4096 TRAIN anchors, 112 TRAIN/32 known VALIDATION probes, scaler, targets, masks and streams. TRAIN Jan 1–21; known VALIDATION Jan 22–28; Jan 29–31 unused. Three head repetitions measure head variation on one encoder/split, not independent market cohorts. This is past-window representation evidence, not forecasting or TEST.

## What changed

A bias-free zero-start matrix A[27,32] produces a canonical channel/feature gain from detached exact parent32. It multiplies only the waveform residual, with the gain broadcast over the original time axis before semantic channel gathering:

```text
R = reshape(z @ D.T, [B,C,H,F])
G = 1 + reshape(L.detach() @ A.T, [B,C,1,F])
prediction = frozen_parent_decode(L, ids) + semantic_gather(R * G, ids)
```

New arithmetic is CUDA F32. The unchanged raw-F64 relative bank, original V28 frozen standardizer, waveform Huber1, learned4/native36, masks and fixed heads stay fixed. No task labels, auxiliary loss, conditional map, learned bias, gain constraint or post-encoder PCA is added. Unbounded gains can cross zero, amplify gradients or overfit; no causal explanation of generalization follows.

A adds 864 trainable parameters: total 298289, updated 15768, protected 282521. Geometry is C3/H128/F9/P8/W64 with canonical IDs 0/1/2 and interval 60000 ms. The exact parent32 remains NoGrad with no optimizer gradients, moments, decay or drift.

Fresh saved v24 initialization is used, never trained V28 weights. New full initial36, validity and row IDs equal saved V28 initialization exactly. Its initialized fixed-head scores are reused after that parity proof; zero initial head refits. Only new final banks receive one ridge pipeline/four targets and three neural fits. The initial neural activity score 0.7246054298126288 remains an initialization control, not a trained-final record.

The normalizer is the exact saved V28 archive and original statistics/source identity: zero new calibration fits or encoder calls. V28 originally fit it once on 4096 natural TRAIN windows, with zero encoder calls. The genuine raw-F64 adjacent-relative bank and historical raw-F32-before-scaler parent view share one final mask. Original V28 raw/calibration witnesses remain bound; new decoder/conditioning witnesses are saved locally.

## Separate task results

Fixed heads use TRAIN-only feature/target normalization, ridge1 and neural16-tanh/Adam0.01/100 steps. D36 heads have 148/660 parameters, repetitions 2701/2802/2903 and uint64 seeds 13088411117524848790, 10018485049408248344, 9083751155097190455. Native32 retains its original settings. All earlier rows and the V29 initialized scoring are reused; V29 final scoring is new.

### Past volatility

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only targets · complexity 4/5 · past volatility · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728107 | 0.380336 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.446315 | 0.392614 | 100 |
| RPB-v25.market-01 | 36 | 0.409235 | 0.236161 | 100 |
| RPB-v26.market-01 | 36 | 0.336490 | 0.419260 | 100 |
| RPB-v28.market-01 | 36 | 0.333838 | -0.105883 | 100 |
| RPB-v29.market-01.init | 36 | 0.455403 | 0.228191 | 100 |
| RPB-v29.market-01 | 36 | 0.603912 | 0.089783 | 100 |

### Past trend

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only targets · complexity 4/5 · past trend · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | -0.383904 | -0.542353 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.654911 | 0.134296 | 100 |
| RPB-v25.market-01 | 36 | 0.774228 | 0.488465 | 100 |
| RPB-v26.market-01 | 36 | 0.723823 | 0.275948 | 100 |
| RPB-v28.market-01 | 36 | 0.684885 | 0.298249 | 100 |
| RPB-v29.market-01.init | 36 | -0.078349 | -0.450465 | 100 |
| RPB-v29.market-01 | 36 | -0.664295 | -0.438662 | 100 |

### Past activity

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only targets · complexity 4/5 · past activity · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728431 | 0.512167 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.751430 | 0.563153 | 100 |
| RPB-v25.market-01 | 36 | 0.755266 | 0.597520 | 100 |
| RPB-v26.market-01 | 36 | 0.750337 | 0.667897 | 100 |
| RPB-v28.market-01 | 36 | 0.748209 | 0.607966 | 100 |
| RPB-v29.market-01.init | 36 | 0.729677 | 0.724605 | 100 |
| RPB-v29.market-01 | 36 | 0.825582 | 0.635718 | 100 |

### Past relation

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only targets · complexity 4/5 · past relation · intact known VALIDATION.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.388583 | 0.257996 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.340685 | 0.349583 | 100 |
| RPB-v25.market-01 | 36 | 0.373557 | 0.407150 | 100 |
| RPB-v26.market-01 | 36 | 0.343038 | 0.208225 | 100 |
| RPB-v28.market-01 | 36 | 0.364335 | 0.315183 | 100 |
| RPB-v29.market-01.init | 36 | 0.379086 | 0.194693 | 100 |
| RPB-v29.market-01 | 36 | 0.453961 | 0.401638 | 100 |

The [frozen integration report](../../cuwacunu_dev/doc/MARKET_PARENT_MODULATED_DECODER_RESULTS.md) and exact registry preserve all 88 rows, every neural repetition, physical-unit MAE/RMSE and full-precision conditions. All 80 historical rows are unchanged, four initialized rows reuse scoring and only four final rows are newly scored. No task average becomes one encoder accuracy.

## Reconstruction and cost

MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only targets · complexity 4/5 · unchanged fixed hidden TRAIN/known-VALIDATION queries.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v29.market-01.init | 0 | 0.278625 | 0.390528 | — |
| RPB-v29.market-01 | 2048 | 0.265796 | 0.382868 | 159.606 |

Errors are original fixed-query TRAIN-CF-scaled MAE. Initial TRAIN/VALIDATION 0.27862535204206196/0.39052824676036835 match the saved parent. Final 0.2657958780016218/0.38286811858415604 improve without meeting the representation gates. Inherited 2048 parent updates stay separate from new branch exposure.

GPU training took 159.606196866 seconds including checkpoint/witness/loader-parity I/O. Initial/final feature I/O took 1.494165288/1.259348142 seconds; final fixed-head fitting/scoring/I/O took 2.558239264 seconds. Pure head-fit compute is unavailable. The saved scalar audit took 1.7932028629984416 seconds. New calibration fit time is zero; the original V28 normalizer fit cost 2.929513154 seconds is historical, not newly charged. Raw-loader checks and CPU parent preprocessing are included, so these descriptive times support no fair speed comparison.

The first decoder loss gradient opens D while projection/gain loss gradients are zero; A stays literal zero on that step. Projection can move through AdamW decay despite its zero first loss gradient. Projection/gain then have 2047 nonzero-gradient updates; D has 2048. Weight-change counts are projection 2048, D 2048 and A 2047. Actual CUDA learning, exact protected state and typed next-update resume pass.

## Portable API, evidence and reporting correction

The optional facade is `cuwacunu::encoder::parent_modulated` in `include/cuwacunu/encoder/parent_modulated_encoder.h`. Headers are `parent_modulated_decoder_*.h` under the standalone raw-patch module, with `config/parent_modulated_decoder.conf`. Input requires genuine raw F64; the parent view uses historical CPU raw-F32 rounding before the original scaler. New typed inference/TRAIN archives bind the V29 decoder, unchanged relative-bank family, reused normalizer/scaler/source, optimizer handles and counters. Old V28/wrong-kind/width/scaler/recipe state is rejected. The primary API stays fixed.

All 91 reusable pairs match: 83 preserved plus 8 new. Only the integration container executed the CUDA fixture and one quality trajectory. The first engineering compile failed on a const helper signature before any GPU invocation; its exact source/log and the corrected second build remain preserved. No encoder header/formula change was needed for that fixture correction.

Saved arithmetic passes 567,804 checks. Independent report verification checks 194 asset bindings and 562 durable members. The frozen first reporter stopped before report output because an inherited check expected new raw witnesses. The approved additive `report-market-parent-modulated-decoder-v2.py` instead pins the original V28 raw/record witnesses and passed audit, as prospectively declared. Original reporter/source/stderr and the exact three-hunk diff remain preserved. No encoder, head or audit was rerun, no missing evidence was synthesized and no scores/gates changed. Full raw/decoder/gain witnesses remain ignored locally and hash-only in durable inventories.

The ledger append preserves all nine earlier records exactly and indexes 50 rehashed, available, Git-ignored/untracked checkpoints. It updates the two descriptive linear task leaders while preserving primary/balanced pointers and all neural leaders. Prior 77,831-file preservation receipts remain bound; no broad scan was repeated for this guide.

## Local checkpoint pins

Paths below resolve beneath `C:/Work/cuwacunu/cuwacunu_dev/output/runs/market-parent-modulated-decoder/run-p4XMUY/checkpoints/`. All five files are local, ignored and untracked; Git alone cannot recover weights. Verify size/hash before reuse.

- `inference-0.pt` — SHA256 `6ab1c927075df1ceb20bc94abbd74f2a08bf9d9978d3adbf3e104d542b38cc8c`, 1286671 bytes.
- `inference-2048.pt` — SHA256 `100a8bef1cf3ef57eafd9e2e58fe5fb2d3f8e7b62449250d9ee0ad10175a6975`, 1287607 bytes.
- `training-1024.pt` — SHA256 `aed7ddf376fb377eb6a837271c6a8f4fe451ab7f9bd61742a6408833382fe485`, 1437318 bytes.
- `training-2048.pt` — SHA256 `c0d881bc12cf1722a49b4008ff745bd96608fe48bc2ab55d8fa885af94af117e`, 1437318 bytes.
- `training-512.pt` — SHA256 `8aba45e189742e87ee5dcb14674d49c640be91f1c3aa321c6efad74fd382e1c2`, 1436971 bytes.

The reused normalizer SHA256 is `c3e4f9e850ebbb355ae58e8b434173e4db0d1ff98055c844af4fe16ba7879fbc`; its identity is `rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476`. It remains V28 fit state rather than a relabeled V29 calibration.

- [Prospective card](../../cuwacunu_dev/doc/MARKET_PARENT_MODULATED_DECODER_CARD.md)
- [Full frozen result](../../cuwacunu_dev/doc/MARKET_PARENT_MODULATED_DECODER_RESULTS.md)
- [Exact instance registry](parent_modulated_decoder_embedding_versions.json)
- [Live search ledger](ENCODER_SEARCH_LEDGER.md)
- [Completion and local pins](../../cuwacunu_dev/doc/results/market_parent_modulated_decoder_v1/completion.json)

## Next direction

No V29 TEMPO-5 transfer ran after market failure. Stop the exact gain mechanism without gain, normalizer, head, rate, width, seed or budget rescue. Keep native36 balanced, the new activity/relation linear records and all old references.

The separate [RPB-v30 card](../../cuwacunu_dev/doc/MARKET_LOCATION_SCALE_DECODER_CARD.md) is declared, pending admission/quality. It tests proper Gaussian waveform location/scale reconstruction, with unmodulated V28 mean and a change-conditioned variance matrix/bias. Native36, 2048/B8, data/scaler/normalizer and initial scoring reuse stay fixed. The fixed data-free softplus link must produce actual initial variance1; valid negative F64 NLL is retained and mean query MAE stays separate. This is a new objective/distribution, not a gain-recipe rescue. No primary promotion, commit, push or publication follows.
