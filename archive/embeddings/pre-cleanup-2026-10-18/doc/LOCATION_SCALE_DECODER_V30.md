# RPB-v30: Gaussian waveform location and scale

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

**RPB-v30.market-01 passes the complete declared market gate: 9/9 original conditions, 8/8 strict native36 floors, all three final neural trend fits positive and 100% coverage.** The frozen TEMPO-5 transfer is separately declared and pending; it has no measured V30 result yet. Keep RPB-v22 primary and RPB-v24.market-01.alt-01 the balanced research reference until that transfer is verified. All work here remains local and unpublished.

## Evidence scope and fixed heads

This guide reuses completed saved evidence; no new experiment. One encoder trajectory used CUDA:0 on the RTX A2000 8GB, batch8 and 2048 new branch updates after 2048 inherited parent updates. MKT-HIST-2 uses BTC/ETH/SOL January 2024 minute data, C3/H128/F9, four separate past-window tasks, 4096 TRAIN training anchors, 112 TRAIN head probes and 32 chronological VALIDATION probes. TRAIN is January 1–21 and known VALIDATION January 22–28; January 29–31 is unused. This is known-source representation development, not forecasting or unseen confirmation. Three head fits on one encoder/split measure head variation, not independent market cohorts.

Direct native embeddings have no post-encoder PCA. Fixed heads retain TRAIN-only feature/target scaling, ridge penalty1, and 16-tanh neural/Adam0.01/100 updates with repetitions2701/2802/2903. D36 has148 linear and660 neural fitted parameters; actual uint64 seeds are13088411117524848790/10018485049408248344/9083751155097190455. The native32 parent retains its original dimension-dependent head settings. The exact preserved evaluator binary SHA256 is `5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14`.

Fresh saved v24 initial weights are used, never trained v28/v29 change weights. New full initial36 values, sample validity and row IDs match saved v28 initialization exactly. Consequently the four initial task scores are explicitly reused from v28 with **zero initial head refits**. Only final scoring is new: one ridge pipeline/four targets and three neural fits. All older controls remain saved/reused, with zero old encoder replay. The complete frozen table retains all96 task rows:88 historical,4 reused initial and4 newly scored final.

## Four separate market tasks

RPB-v22.market-02 is the protected native32 parent. RPB-v24.market-01 is the preserved native40 trend/activity reference; its `.alt-01` is the balanced native36 reference. V25, v26 and v29 retain their own stopped recipes and partial records. V30 exports that exact parent32 plus four learned change coordinates. All comparison rows below reuse the frozen report; only V30 final head measurements were newly fitted in this experiment.

### Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-market past-window signals · **complexity 4/5** · intact chronological VALIDATION, 32 rows; heads fit on 112 TRAIN rows.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728107 | 0.380336 | 100 |
| RPB-v24.market-01 | 40 | 0.599739 | 0.324488 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.446315 | 0.392614 | 100 |
| RPB-v25.market-01 | 36 | 0.409235 | 0.236161 | 100 |
| RPB-v26.market-01 | 36 | 0.336490 | 0.419260 | 100 |
| RPB-v29.market-01 | 36 | 0.603912 | 0.089783 | 100 |
| RPB-v30.market-01 | 36 | 0.953913 | 0.588277 | 100 |

V30 neural repetitions2701/2802/2903: 0.6486061493538446, 0.6758376672339298, 0.44038587743070645. Its physical-unit linear MAE/RMSE are 5.515113434489571e-05/7.252789459441617e-05 logreturn; mean neural MAE/RMSE are 0.00017633025349411837/0.00021511728155497738. All individual physical scores and initial scores are in the exact registry.

### Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-market past-window signals · **complexity 4/5** · intact chronological VALIDATION, 32 rows; heads fit on 112 TRAIN rows.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | -0.383904 | -0.542353 | 100 |
| RPB-v24.market-01 | 40 | 0.599060 | 0.066226 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.654911 | 0.134296 | 100 |
| RPB-v25.market-01 | 36 | 0.774228 | 0.488465 | 100 |
| RPB-v26.market-01 | 36 | 0.723823 | 0.275948 | 100 |
| RPB-v29.market-01 | 36 | -0.664295 | -0.438662 | 100 |
| RPB-v30.market-01 | 36 | 0.794366 | 0.320210 | 100 |

V30 neural repetitions2701/2802/2903: 0.008674474092963624, 0.5350811106482072, 0.4168737921607343. Its physical-unit linear MAE/RMSE are 2.4230709341539103e-05/3.0365980660320684e-05 logreturn/tick; mean neural MAE/RMSE are 4.260128935952923e-05/5.4489015910244984e-05. All individual physical scores and initial scores are in the exact registry.

### Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-market past-window signals · **complexity 4/5** · intact chronological VALIDATION, 32 rows; heads fit on 112 TRAIN rows.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728431 | 0.512167 | 100 |
| RPB-v24.market-01 | 40 | 0.764260 | 0.678189 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.751430 | 0.563153 | 100 |
| RPB-v25.market-01 | 36 | 0.755266 | 0.597520 | 100 |
| RPB-v26.market-01 | 36 | 0.750337 | 0.667897 | 100 |
| RPB-v29.market-01 | 36 | 0.825582 | 0.635718 | 100 |
| RPB-v30.market-01 | 36 | 0.865732 | 0.613572 | 100 |

V30 neural repetitions2701/2802/2903: 0.5263652355780768, 0.6968960216889721, 0.6174561986987163. Its physical-unit linear MAE/RMSE are 0.17204336315408053/0.21057992154616248 log1p(base_volume); mean neural MAE/RMSE are 0.2845960447905132/0.35578059924246735. All individual physical scores and initial scores are in the exact registry.

### Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-market past-window signals · **complexity 4/5** · intact chronological VALIDATION, 32 rows; heads fit on 112 TRAIN rows.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.388583 | 0.257996 | 100 |
| RPB-v24.market-01 | 40 | 0.328862 | -0.017315 | 100 |
| RPB-v24.market-01.alt-01 | 36 | 0.340685 | 0.349583 | 100 |
| RPB-v25.market-01 | 36 | 0.373557 | 0.407150 | 100 |
| RPB-v26.market-01 | 36 | 0.343038 | 0.208225 | 100 |
| RPB-v29.market-01 | 36 | 0.453961 | 0.401638 | 100 |
| RPB-v30.market-01 | 36 | 0.487476 | 0.406836 | 100 |

V30 neural repetitions2701/2802/2903: 0.43457053564584935, 0.5043528390709461, 0.2815848611143552. Its physical-unit linear MAE/RMSE are 0.04992962898357469/0.061703956974893316 correlation; mean neural MAE/RMSE are 0.05403187640561657/0.06618132921367507. All individual physical scores and initial scores are in the exact registry.

All original and protection decisions use unchanged full-precision arithmetic tolerance1e-12, never rounded table entries. V30 sets all four linear-head maxima and the neural-volatility maximum in this recorded matched scope. Preserve v25 neural trend0.4884646605379312/relation0.4071495455067296 and native40 neural activity0.6781894013517937, together with displaced v26/v29 records. Initial neural activity0.7246054298126288 is a reused initialization control, not a trained-final leader. There is no universal encoder ranking.

## What changed in training

The unmodulated waveform mean is the protected NoGrad parent reconstruction plus the bias-free change residual. The relative bank, exact raw-F64 input contract and frozen V28 TRAIN standardizer are unchanged. New zero-initialized variance matrix A_sigma[27,4] and bias b_sigma[27] add135 parameters. Their score depends on change4, not parent32, and produces one variance per canonical channel/feature, broadcast across time before semantic gather. Total/updated/protected parameter counts are297560/15039/282521.

The fixed positive F32 variance link uses anchored softplus with beta1/threshold20, epsilon1e-6 and c=0.5413232726357509. It computes `epsilon + coefficient * (softplus(score+c) / softplus(c))`, in that exact divide/multiply/add order, with coefficient F32(1-epsilon). Actual initial variance is bitwise1 and initial mean equals the parent. Target, mean and variance are cast to CUDA F64 before subtraction/squaring/division/log. NLL is `.5*((mean-target)^2/variance + log(variance))`, omitting the constant. Targets outside the mask are sanitized before arithmetic; masked losses are zero. Existing cell→channel→eligible-example averaging is retained. **Valid negative NLL is retained.** No task labels, auxiliary coefficient, covariance head or head tuning is added.

First-batch mean decoder and variance matrix/bias gradients and updates are nonzero; projection loss-gradient is zero initially and learns on the next batch. AdamW decay may still move the initial projection. The protected parent has no gradients, optimizer moments, decay or drift. CUDA admission proves masking/hidden poison handling, bitwise parent32, wrong typed-kind/recipe/scaler/geometry rejection and exact next-update optimizer/weights/counters resume.

Learned variance includes mean error and is not the measured market-volatility target. Its bias may bypass informative change coordinates, the Gaussian loss changes outlier weighting relative to Huber, and its per-CF variance is constant across time. Objective and distribution both change, so this run does not isolate a causal variance-head effect.

## Reconstruction and cost

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-market masked reconstruction · **complexity 4/5** · fixed hidden queries on TRAIN/known VALIDATION. Errors below are TRAIN-CF-scaled mean MAE, separate from NLL.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v30.market-01.init | 0 | 0.278625352 | 0.390528247 | — |
| RPB-v30.market-01 | 2048 | 0.282097576 | 0.390187301 | 131.75605979 |

Updates in this table are new branch updates; both rows include the same2048 inherited parent updates. Initialization has no new training segment. TRAIN mean MAE worsens slightly; known VALIDATION MAE changes only marginally. Do not infer overall mean-reconstruction improvement from the lower NLL.

NLL first/last minibatches:0.3399173832780379/−0.27151144758246093; first64/last64 means:0.21427880773368144/−0.26330228646329235. There are1449 legitimate negative minibatches of2048. Initial variance min/mean/max is1/1/1; final training variance min/mean/max is0.002306993817910552/0.7960859330756918/6.173368453979492. These are distribution diagnostics, not classifier R² or calibrated market-volatility estimates.

GPU segment time131.75605979s includes checkpoint/witness I/O and loader parity. Initial/final feature extraction and I/O take1.1259591679999998/0.782120961s; final saved-head fitting/scoring/I/O2.177286234s; pure fit time is unavailable. New calibration is0 fits/0 forwards/0s; the reused V28 standardizer historical fit2.929513154s stays separate. Scalar audit1.1815329909986758s and full wrapper286.8784265200011s are separate from GPU training. Every raw batch is checked against the legacy F32 loader, and parent preprocessing includes CPU rounding/scaling. These descriptive costs do not support a fair inference-speed comparison.

## Portable API and preservation

The optional `rpb_location_scale` module exposes `cuwacunu/encoder/location_scale_encoder.h`, namespace `cuwacunu::encoder::location_scale`. It provides Config/Encoder/Input/FrozenScaler/BankStandardizer, typed settings/checkpoint helpers, native_embedding and mask utilities. Genuine raw-F64 observations are required; the parent view deliberately rounds raw values to F32 before CPU-F64 scaling and final CUDA-F32 conversion. Use the typed V30 checkpoint and its embedded unchanged V28 bank standardizer. Wrong legacy kinds are refused rather than relabeled. Primary config/API still selects RPB-v22.

All99 reusable pairs match locally:91 preserved plus8 V30 header/facade/settings/fixture pairs. Application data loading, producer, orchestration and fixed heads stay integration-owned. Only the existing integration GPU container ran the engineering fixture and quality trajectory. Two fixture compiles passed; the first emitted one formatting warning and ran no GPU. A fixture-only brace/newline cleanup produced a warning-free second compile before the first and only successful GPU fixture. Both exact capsules/logs are retained; producer build and report each passed on their first invocation.

Independent saved arithmetic passes941,513 checks; F64 cell-NLL normalized error is about1.71e-16. Report verification binds198 assets,570 durable members and every old/reused/new task row. Wrapper protection confirms3219 preexisting integration files unchanged and all50 prior checkpoint pins. The broader historical77,831-file preservation receipts are bound rather than rescanned. All55 candidate checkpoint files were rehashed, available, ignored and untracked. Weights, raw arrays and full decoder/loss/conditioning witnesses remain ignored local assets; only hashes and compact evidence are durable.

The five new checkpoints are local under `C:/Work/cuwacunu/cuwacunu_dev/output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints`; none is exported into this repository.

- `inference-0.pt` — SHA256 `52b6c9881f13e852a86e3e7b22558f27d68616e06c6d638ebbd97344e20ac3c5`, 1284638 bytes.

- `inference-2048.pt` — SHA256 `c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb`, 1285583 bytes.

- `training-1024.pt` — SHA256 `8728c3908b95c9685bc28ade55b47e2eecc22be6cfbfb15d4b57f9ce9dfba691`, 1430905 bytes.

- `training-2048.pt` — SHA256 `57c01efe74e2fadf3941ca6c713ddcb8976074c156904372fa2af87e0e777907`, 1430905 bytes.

- `training-512.pt` — SHA256 `bc383e51f7e8dc411acb568b0428d1a7cb8fc10267d3d5cc750e5635603a90d3`, 1430551 bytes.

Use final inference SHA256 `c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb` for the separately declared frozen transfer. The reused normalizer SHA256 is `c3e4f9e850ebbb355ae58e8b434173e4db0d1ff98055c844af4fe16ba7879fbc`, identity `rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476`; its source remains V28, with no new normalization fits or calibration forwards.

- [Prospective market card](../../cuwacunu_dev/doc/MARKET_LOCATION_SCALE_DECODER_CARD.md)
- [Full frozen market result](../../cuwacunu_dev/doc/MARKET_LOCATION_SCALE_DECODER_RESULTS.md)
- [Exact instance registry](location_scale_decoder_embedding_versions.json)
- [Completion](../../cuwacunu_dev/doc/results/market_location_scale_decoder_v1/completion.json)
- [Report verification](../../cuwacunu_dev/doc/results/market_location_scale_decoder_v1/report-verification.json)
- [Live search ledger](ENCODER_SEARCH_LEDGER.md)

## Next direction

The [frozen TEMPO-5 transfer card](../../cuwacunu_dev/doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md) is declared, pending admission/extraction/scoring. It checks the exact saved V30 final checkpoint with zero encoder updates under both preserved scaler policies and all eight native36 head/view retention floors, plus full coverage. TEMPO-5 is the compatible C3/H128/F9 slow lead/lag signal family, designed complexity5/5. No V30 synthetic result is measured here. Keep the frozen market result independent, RPB-v22 primary and native36 balanced until transfer is verified. Preserve every reference, stopped recipe and checkpoint; no primary promotion, commit, push or publication follows.
