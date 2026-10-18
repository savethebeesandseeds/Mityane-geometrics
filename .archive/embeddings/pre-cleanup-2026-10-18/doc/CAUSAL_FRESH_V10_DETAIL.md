<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# CPB-v10.market-01 detailed saved evidence

**Full trust and measured-candidate admission passed.** The [final evidence join](causal_fresh_v10/final-establishment.json) confirms the passed runtime/CUDA/typed-resume proofs, training audit, quality audit and all 13 exact whole-native40/statistic DATA pairings versus saved V9. It records 326495 training-audit checks and 13089 quality-audit checks with zero mismatches, 3/22 numeric conditions and 100% coverage. Overall advancement, winner and winner selection remain false. Frozen producer pending-trust records stay untouched; the join completes their admission separately. V3 remains the clean baseline and sole dev encoder. Read [RESULTS](CAUSAL_FRESH_V10_RESULTS.md) for all seven final controls and [REVIEW](CAUSAL_FRESH_V10_REVIEW.md) for the bounded interpretation.

These are literal saved report sections, not refits or new reductions of arrays. Four market tasks retain separate R², physical MAE/RMSE and all three full-precision neural-head repetitions, plus unchanged encoder-free raw controls. Four timing panels retain both input-scaler policies, intact/deleted views and every cohort/head repetition. MKT-HIST-2 uses 112 TRAIN/32 known chronological VALIDATION windows and 4096 TRAIN calibration anchors. TEMPO-5 has three independent generator cohorts 941101/941102/941103, each 512 TRAIN/256 VALIDATION examples from 256/128 paired source groups. Deleted views use the same validation source groups. One fresh CUDA:0 encoder, batch 8, 2048 completed updates/2048 attempts/zero skips; paired transforms and repeated heads do not create new encoder runs or independent market cohorts.

Fixed ridge 1 and 16-tanh/Adam 0.01/100-step heads retain repetitions 2701/2802/2903 and D40 seeds 6049515622977958931, 8262941649267994858, 2095371932330426401:164/724 market and 82/690 timing parameters. Both initial/final states receive new exports/heads, with 1572 exports recorded. TRAIN-fitted feature/target scales remain fixed per allowed dataset. Synthetic input scales are freshly fitted per cohort and shared between checkpoints; the other policy uses fresh market TRAIN input scaling. No synthetic encoder training, PCA, tuning, old encoder/head replay or initial-score reuse occurs. V3–V9 frozen rows are historical evidence, not freshly repeated matched training controls. Complexity 4/5 and 5/5 are ordinal recipe labels, not accuracy or observed task difficulty.

The unchanged raw controls supply6912 recent128 or 13824 full256 numbers and consequently larger fixed-head parameter capacity than native40. Their original saved fits are reused, not newly fitted by V10; width and capacity remain explicit in their rows. R² is a regression score, not classification accuracy.

## Objective and forward contract

V10 retains V9's full forward/native40/constructor draw order and changes one training route:

`total = R[Huber_delta1(mean-target)] + R[0.5*((mean-target)^2/variance + log(variance))]`.

Both coefficients are 1. Targets stay detached; mean and variance are live in ONE joint Gaussian term. Mean receives Huber and Gaussian mean gradients; uncertainty4 receives the Gaussian variance derivative once. No direct variance-output gradient goes into parent32/mean4. The reporting Gaussian copy detaches both distribution tensors and never calls backward. At identical distributions the scalar objective equals V9's detached-error expression, while its derivative differs. Shared global clip 1/single AdamW retains scaling/history coupling and future mean changes affect later variance targets. This is not full optimizer independence or a duplicate variance objective.

Native40=20 learned content + 12 fixed spectral + mean4 + uncertainty4. All 294752 learned values/76 registered tensors, optimizer state and TRAIN statistics start fresh, source_updates=0/constructor seed 943001. The fixed prior has 3888 values, total298640. Seventy-five mandatory identifiable tensors have nonzero data-gradient/movement events; 294751 is their numel capacity, not proof for each scalar. The retained pool-score bias is a softmax gauge, observed separately. Original-position half-local/half-shared-global pooling still applies each nonempty bin's8 existing export rows/bias once; empty slices remain zero after bias. Both temporal routes use the same owned kernel, and every slice describes the current cutoff endpoint using no future values or masks.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.61655559 | 0.26788204 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.75086921 | 0.02651728 | 100.00 |
| CPB-v10.market-01.init | 40 | 0.26294158 | -0.36300499 | 100.00 |
| CPB-v10.market-01 | 40 | 0.93567490 | 0.61766705 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical prediction errors; units: logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.0001457632481993121 | 0.00020920228421032871 | 0.00023436261930595529 | 0.00028233840324221054 |
| Raw data — no encoder (full256) | 0.00012113393348219792 | 0.00016862770230196986 | 0.00027850334296948618 | 0.00032813516320482082 |
| CPB-v10.market-01.init | 0.00025055934059789686 | 0.0002900455267242687 | 0.00033064468709028631 | 0.00039183500147358719 |
| CPB-v10.market-01 | 6.5287997780750208e-05 | 8.5685126325557373e-05 | 0.00016553113935653321 | 0.00020628828806851446 |

CPB-v10.market-01.init neural R² for repetitions 2701/2802/2903: -0.040167667755234282, -0.79467427575727778, -0.254173032572929.

CPB-v10.market-01 neural R² for repetitions 2701/2802/2903: 0.72007615110758816, 0.69181981181918362, 0.44110517635270319.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.07548049 | -0.03159348 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.36673586 | -0.62097135 | 100.00 |
| CPB-v10.market-01.init | 40 | -0.16691292 | -0.56215422 | 100.00 |
| CPB-v10.market-01 | 40 | 0.89708528 | 0.54823037 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical prediction errors; units: logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 4.2538095061369338e-05 | 6.4386877535939344e-05 | 5.249607559998899e-05 | 6.7754225595132475e-05 |
| Raw data — no encoder (full256) | 4.0222433762679972e-05 | 5.3288288447226955e-05 | 6.7286706934599854e-05 | 8.4157641119802349e-05 |
| CPB-v10.market-01.init | 4.7890874995537686e-05 | 7.2336675702662165e-05 | 6.5665639526452536e-05 | 8.3085055160316454e-05 |
| CPB-v10.market-01 | 1.8171657656549772e-05 | 2.1482166107369831e-05 | 3.5201613064165647e-05 | 4.4900113759407961e-05 |

CPB-v10.market-01.init neural R² for repetitions 2701/2802/2903: -0.8810937230834377, -0.74511489507803841, -0.060254036024300017.

CPB-v10.market-01 neural R² for repetitions 2701/2802/2903: 0.57976322595647356, 0.45906737154334909, 0.60586050485844944.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.52820248 | 0.41069546 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.65964729 | 0.32128228 | 100.00 |
| CPB-v10.market-01.init | 40 | 0.47793118 | 0.54183249 | 100.00 |
| CPB-v10.market-01 | 40 | 0.79719960 | 0.48936013 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical prediction errors; units: log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.30772366365230841 | 0.39473744667109539 | 0.37233804145499749 | 0.44020959039793733 |
| Raw data — no encoder (full256) | 0.27840914010158291 | 0.33527031837769694 | 0.37995325638639649 | 0.47290779350881545 |
| CPB-v10.market-01.init | 0.33226073894129032 | 0.41523540869404407 | 0.30863033067568429 | 0.38557242044955425 |
| CPB-v10.market-01 | 0.2274853144292828 | 0.25880037558047531 | 0.34348949405388052 | 0.4074104964475353 |

CPB-v10.market-01.init neural R² for repetitions 2701/2802/2903: 0.46838905936437514, 0.45346021527006519, 0.70364818411382757.

CPB-v10.market-01 neural R² for repetitions 2701/2802/2903: 0.54095073425183049, 0.61798736364853224, 0.30914228784494668.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | -1.18503359 | 0.09860018 | 100.00 |
| Raw data — no encoder (full256) | 13824 | -0.33498693 | -0.01139620 | 100.00 |
| CPB-v10.market-01.init | 40 | 0.29558925 | -0.06594906 | 100.00 |
| CPB-v10.market-01 | 40 | 0.08166440 | 0.36820496 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical prediction errors; units: correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.10081525450811848 | 0.12740459195953027 | 0.065765884738787181 | 0.081095482611407546 |
| Raw data — no encoder (full256) | 0.083363206746537438 | 0.09958513805506701 | 0.067221551946341093 | 0.086505902968647921 |
| CPB-v10.market-01.init | 0.058987775053747998 | 0.072338418333731072 | 0.071894606008359715 | 0.088802710622309947 |
| CPB-v10.market-01 | 0.067838038573812459 | 0.082595565898654508 | 0.054592098178096418 | 0.068483079456814464 |

CPB-v10.market-01.init neural R² for repetitions 2701/2802/2903: 0.029204570193499535, -0.26357952487012359, 0.036527770809100857.

CPB-v10.market-01 neural R² for repetitions 2701/2802/2903: 0.33117714401163179, 0.3597728694916188, 0.41366487926256246.

## Hard timing information

TEMPO-5 has cohorts 941101/941102/941103, each 512 TRAIN examples from 256 paired sources and 256 VALIDATION examples from 128 distinct paired sources. Both stages use newly fitted three-seed fixed heads per policy/cohort. Classification heads contain82/690 parameters. Synthetic TRAIN input statistics are freshly fitted once per cohort and shared by initial/final; the secondary policy uses this run’s fresh market TRAIN scaler. No synthetic encoder training occurs.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v10.market-01.init | 40 | 94.9219 | 94.2274 | 100.0000 |
| CPB-v10.market-01 | 40 | 97.5260 | 97.0920 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v10.market-01.init | 40 | 89.0625 | 87.2830 | 100.0000 |
| CPB-v10.market-01 | 40 | 91.9271 | 91.1458 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v10.market-01.init | 40 | 94.6615 | 94.1840 | 100.0000 |
| CPB-v10.market-01 | 40 | 98.9583 | 98.3941 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v10.market-01.init | 40 | 88.5417 | 84.4618 | 100.0000 |
| CPB-v10.market-01 | 40 | 93.3594 | 91.9271 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. initial, synthetic_train; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.95703125 | 0.94140625 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.8828125 | 0.86328125 | 1 |
| 941101 | rep-2802 | validation_intact | 0.95703125 | 0.953125 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.8828125 | 0.890625 | 1 |
| 941101 | rep-2903 | validation_intact | 0.95703125 | 0.9453125 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.8828125 | 0.87890625 | 1 |
| 941102 | rep-2701 | validation_intact | 0.9453125 | 0.92578125 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.88671875 | 0.8828125 | 1 |
| 941102 | rep-2802 | validation_intact | 0.9453125 | 0.93359375 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.88671875 | 0.84765625 | 1 |
| 941102 | rep-2903 | validation_intact | 0.9453125 | 0.93359375 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.88671875 | 0.859375 | 1 |
| 941103 | rep-2701 | validation_intact | 0.9453125 | 0.953125 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.90234375 | 0.90625 | 1 |
| 941103 | rep-2802 | validation_intact | 0.9453125 | 0.9453125 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.90234375 | 0.875 | 1 |
| 941103 | rep-2903 | validation_intact | 0.9453125 | 0.94921875 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.90234375 | 0.8515625 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. initial, market_scaler; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.9609375 | 0.953125 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.88671875 | 0.83984375 | 1 |
| 941101 | rep-2802 | validation_intact | 0.9609375 | 0.953125 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.88671875 | 0.8671875 | 1 |
| 941101 | rep-2903 | validation_intact | 0.9609375 | 0.9375 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.88671875 | 0.859375 | 1 |
| 941102 | rep-2701 | validation_intact | 0.9375 | 0.93359375 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.87890625 | 0.828125 | 1 |
| 941102 | rep-2802 | validation_intact | 0.9375 | 0.94921875 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.87890625 | 0.828125 | 1 |
| 941102 | rep-2903 | validation_intact | 0.9375 | 0.94140625 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.87890625 | 0.8125 | 1 |
| 941103 | rep-2701 | validation_intact | 0.94140625 | 0.91796875 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.890625 | 0.84765625 | 1 |
| 941103 | rep-2802 | validation_intact | 0.94140625 | 0.953125 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.890625 | 0.875 | 1 |
| 941103 | rep-2903 | validation_intact | 0.94140625 | 0.9375 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.890625 | 0.84375 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. final, synthetic_train; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.9765625 | 0.9765625 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.9296875 | 0.9140625 | 1 |
| 941101 | rep-2802 | validation_intact | 0.9765625 | 0.97265625 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.9296875 | 0.91796875 | 1 |
| 941101 | rep-2903 | validation_intact | 0.9765625 | 0.97265625 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.9296875 | 0.92578125 | 1 |
| 941102 | rep-2701 | validation_intact | 0.98046875 | 0.9609375 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.8984375 | 0.90625 | 1 |
| 941102 | rep-2802 | validation_intact | 0.98046875 | 0.984375 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.8984375 | 0.8984375 | 1 |
| 941102 | rep-2903 | validation_intact | 0.98046875 | 0.9609375 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.8984375 | 0.90234375 | 1 |
| 941103 | rep-2701 | validation_intact | 0.96875 | 0.96484375 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.9296875 | 0.90234375 | 1 |
| 941103 | rep-2802 | validation_intact | 0.96875 | 0.97265625 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.9296875 | 0.91796875 | 1 |
| 941103 | rep-2903 | validation_intact | 0.96875 | 0.97265625 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.9296875 | 0.91796875 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. final, market_scaler; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.9921875 | 0.984375 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.93359375 | 0.9375 | 1 |
| 941101 | rep-2802 | validation_intact | 0.9921875 | 0.984375 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.93359375 | 0.94140625 | 1 |
| 941101 | rep-2903 | validation_intact | 0.9921875 | 0.98828125 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.93359375 | 0.9375 | 1 |
| 941102 | rep-2701 | validation_intact | 0.984375 | 0.9765625 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.9296875 | 0.8984375 | 1 |
| 941102 | rep-2802 | validation_intact | 0.984375 | 0.98828125 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.9296875 | 0.90625 | 1 |
| 941102 | rep-2903 | validation_intact | 0.984375 | 0.9765625 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.9296875 | 0.8984375 | 1 |
| 941103 | rep-2701 | validation_intact | 0.9921875 | 0.98046875 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.9375 | 0.9140625 | 1 |
| 941103 | rep-2802 | validation_intact | 0.9921875 | 0.984375 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.9375 | 0.91015625 | 1 |
| 941103 | rep-2903 | validation_intact | 0.9921875 | 0.9921875 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.9375 | 0.9296875 | 1 |

## Training, fixed queries and separate cost

Errors use the immutable saved hidden reconstruction queries with hierarchical standardized MAE. Gaussian NLL is an unchanged reporting metric and can be negative; the optimized total is Huber(delta 1) plus one joint Gaussian NLL. The synchronized step timer covers forward/objective/backward/clip/AdamW and in-step checks; loading, masking, checkpoint/resume I/O, evidence, extraction and readout fitting remain separate.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Fixed saved TRAIN/VALIDATION query; standardized MAE, source_updates=0.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v10.market-01.init | 0 | 0.66425819770853367 | 0.95821285302915538 | 0 |
| CPB-v10.market-01 | 2048 | 0.25142589402513021 | 0.38612232096966237 | 146.06388198500014 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Same saved fixed queries; Gaussian waveform NLL, distinct from MAE.

| Encoder | Updates | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- | --- |
| CPB-v10.market-01.init | 0 | 0.42384730158338346 | 0.79549181757104681 |
| CPB-v10.market-01 | 2048 | -0.37746072830305494 | 1.6198607484755174 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Same fixed query masks; descriptive objective components, not additional advancement criteria. Joint Gaussian NLL and total may be negative.

| Encoder | Split | Mean Huber1 | Joint Gaussian NLL | Total objective |
| --- | --- | --- | --- | --- |
| CPB-v10.market-01.init | train | 0.3237604271966214 | 0.42384730158338346 | 0.74760772878000492 |
| CPB-v10.market-01.init | validation | 0.58273177713710489 | 0.79549181757104681 | 1.3782235947081518 |
| CPB-v10.market-01 | train | 0.085718591403455058 | -0.37746072830305494 | -0.29174213689959988 |
| CPB-v10.market-01 | validation | 0.1494667350941816 | 1.6198607484755174 | 1.769327483569699 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. and Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. One-run descriptive cost; components are not interchangeable.

| Stage | Seconds |
| --- | --- |
| Fresh market TRAIN calibration including input/witness I/O | 2.8985313819999998 |
| Initial+final export including fresh synthetic statistics and input/state I/O | 76.32270304100166 |
| initial market CPU fixed heads and I/O | 1.2325232099974528 |
| final market CPU fixed heads and I/O | 1.307364623000467 |
| initial synthetic CPU fixed heads/scoring and I/O | 1.9172491569988779 |
| final synthetic CPU fixed heads/scoring and I/O | 2.3478964290006843 |

## Actual optimization, learning and proof evidence

The following actual minibatches are pre-step observations of the one trajectory, not the fixed112/32 query population and not head quality. Only objective.total_loss is backpropagated. Gaussian reporting NLL equals the joint component numerically here but its detached copy has no gradient. Neither total/component sign nor scalar magnitude proves gradient dominance.

Dataset: **MKT-HIST-2** · retained past-window information · **complexity 4/5** · TRAIN-only actual B8 pre-step minibatches at updates 1/2048, separate from fixed queries.

| Update | Huber1 mean component | ONE joint Gaussian component | Total objective | Reporting Gaussian NLL |
| --- | --- | --- | --- | --- |
| 1 | 0.47003086138400735 | 0.8024352489237474 | 1.2724661103077548 | 0.8024352489237474 |
| 2048 | 0.061238782210412526 | -0.5536577050113076 | -0.4924189228008951 | -0.5536577050113076 |

The final join records 326495 training-audit checks: 25581 core, 61153 numeric-statistic, 1177 saved-state, 117864 Gaussian, 69301 actual-component and 51419 temporal-learning checks, with zero mismatches. This total and the module labels are copied from the saved join, without a new human-report reduction. The audit independently checks support/F64 reductions, exact source/typed objective routes, actual seed streams and row choices, fresh TRAIN statistics, checkpoint optimizer state/order/counters, exact-state typed resume512 and the continued trajectory. It executes no model, head fit, optimizer replay or full encoder Jacobian. One global clip 1 is observed on 1767 of 2048 updates; its input norm is recorded, not independently reconstructed. This observation does not establish which branch dominated or a counterfactual trajectory.

All four export8-row groups record2048 nonzero data-gradient and movement events for weights/biases. Each of four within-bin centered position contrasts records 2048 gradient/movement events. The three unit-R16 Hadamard directions [ + + --], [ + - + -], [ + -- + ], repeated across the four positions per bin and divided by4, each record2048 gradient and 2048 movement events. Signed final changes are −0.09296585619449615, 0.026230327785015106, −0.10600896179676056, with positive overall projection change0.14341750414406226. The declared main gate requires per-direction events and positive OVERALL final change; individual final coefficients are descriptive. All movement includes AdamW decay; decay is not counted as data-gradient learning. The two actual raw-array witnesses are updates 1/2048 only, not a reconstruction of 2046 unsaved gradient arrays.

The objective fixture passes 92 engineering checks, including 36 independent smooth-composite F64 finite differences; 12 CUDA decoder calls, one synthetic AdamW step and five selected fixture tensors moved, with 0 encoder forwards/old artifacts/quality fits. It verifies joint mean + uncertainty routes, variance counted once and manual native40 gradient parity. The temporal fixture passes 100 checks with one backbone encode, five direct pooling-kernel calls, zero public Encoder/head calls and one synthetic AdamW step. These fixtures establish local implementation behavior, not quality or an additional encoder training run.

Full engineering records 303517 checks, 7372 CUDA encoder calls and 391 typed negative cases. Saved initial 0 and final 2048 causal proofs each record150674 checks/3685 CUDA encoder calls, with 128 cutoffs, 2350 suffix cases and 30 internal temporal cases per stage. Their durations are separate from training; each proof performs0 updates. Five typed metadata nodes reject wrong/missing/policy/dtype/shape values, including the new live-error objective flag. Actual main resume512 checks exact saved state and continues the one trajectory; no counterfactual next-step replay claim is made for main training.

Fresh TRAIN calibration uses 4096 anchors/512 loader calls and zero VALIDATION calibration rows; old model/statistic opens are 0. The source query masks are the saved seed 942005 plans; 943005 is not consumed. Initial and final query scalars above cover112 TRAIN/32 known VALIDATION rows; first 8 cell archives remain bounded witnesses. Whole40 initial pairing is declared against saved V9 DATA/statistics, not old trained checkpoint loading; the independent quality audit verifies all 13 pairings exactly. V8/V9 historical partial pairing semantics stay unchanged.

## Additional timing scopes

The synchronized GPU step timer includes forward/objective/backward/global clip/AdamW and in-step checks, rather than a pure-kernel microbenchmark. Loading/masking, checkpoint/resume/logging, gradient-evidence capture, fixed-query/calibration I/O, wrappers and readout work have separate scopes. Timings are descriptive single-run measurements; do not sum overlapping scopes or infer matched benchmark wins from historical costs or UTC timestamps.

Dataset: **MKT-HIST-2** · past-window retention · **complexity 4/5** · TRAIN 4096 calibration / fixed TRAIN 112 / known VALIDATION 32. Dataset: **TEMPO-5** · nine-feature slow lead/lag recipe · **complexity 5/5** · three cohorts, each TRAIN 512/VALIDATION 256. Separate actual recorded phase scopes.

| Scope | Seconds | Interpretation |
| --- | --- | --- |
| Build wrapper | 85.96029924899995 | Source compilation/closure, no model quality |
| Objective fixture wrapper | 4.271785037999507 | Independent engineered CUDA objective proof |
| Temporal fixture wrapper | 1.5613151150027988 | Engineered CUDA pooling proof |
| Full engineering wrapper | 367.5820077009994 | Causal/typed/learning fixture proof, separate updates |
| Synchronized GPU training steps | 146.06388198500014 | Forward/objective/backward/clip/AdamW and in-step checks |
| Training loader/mask time | 2.9921161450000033 | Recorded separately |
| Checkpoint/resume/logging time | 0.917233562000001 | Recorded separately |
| Gradient evidence time | 43.20515033699987 | Recorded separately |
| Main trajectory wall | 195.119894663 | One actual training trajectory, broader than GPU step timer |
| Training wrapper | 204.356952225 | Includes launch/calibration/other wrapper scope |
| Saved initial0 proof wrapper | 171.008157389002 | No training update |
| Saved final2048 proof wrapper | 176.7826903899986 | No training update |
| Quality wrapper | 83.40623863400106 | Normal extraction + all fixed heads + two aggregators once |

Normal quality has five completed scientific stage durations and seven passed local stage files: extraction, initial/final market heads, initial/final synthetic heads, report and comparison. The outer wrapper pins its Python interpreter/log/frozen driver; that interpreter is not an eighth model binary. Current V10 has no reporting recovery. Historical V9 failed report/audit stages and report-only recovery remain preserved opaque context, without scientific replay.

## Evidence bindings and completed admission

All 127 frozen main capsule files remain exact; seven build binaries and 11 dependency-closure files are separately mirrored. Complete contains 718 files/1119401019 bytes and 621 ignored PT archives. These include features, heads, witnesses and test states, not 621 trained encoders. The original raw report/comparison keep mandatory postmeasurement trust pending. The quality audit and final establishment join have passed; their separate receipts complete measured-candidate/full-trust admission without changing any frozen producer fields.

The preserved draft and final report manifests bind the frozen main card/MD/input/goal/protocol closure, raw table/comparison and their literal Markdown sections, main training summary/learning receipt, passed training audit, actual stage/proof receipts, Complete mirror and all five canonical checkpoints. All four market tasks and four timing panels are retained separately; no new score, head fit, tensor arithmetic, candidate choice, current metadata edit or publication occurred while drafting.

Read the [independent training audit](causal_fresh_v10/independent-training-audit.json), [independent quality audit](causal_fresh_v10/quality-independent-audit.json), [frozen prospective card](CAUSAL_FRESH_V10_CARD.json) and [Complete mirror](causal_fresh_v10/complete-mirror-attempt-001.json). The [final establishment receipt](causal_fresh_v10/final-establishment.json) now completes the trust chain.

Keep the five checkpoints and every prior source/result/failure. V3 remains selected; the goal remains active. Every pending draft generation remains preserved; these canonical reports bind the final-join receipt alongside the actual [quality audit](causal_fresh_v10/quality-independent-audit.json). No next encoder or rescue coefficient/seed/rate/budget/head change is selected.

## Proposed direction after closure — not a measured candidate

Retain V3 and every scoped V3–V9 leader. The next source review may consider one fresh V9-based temporal-summary hypothesis: each anchored bin would use four mean-export rows and four centered-dispersion rows under the SAME normalized half-local/half-shared-global attention mixture. Keep V9's unit Huber(delta 1) mean plus detached-error Gaussian variance calibration, native40, parameter count, seeds, budget of 2048 updates at batch 8 and D40 fixed heads. Do not combine the failed V10 live-joint mean loss or rescue it by changing coefficients.

This tests variation retention, not a demonstrated cause of known-VALIDATION errors or a promised fix. Splitting existing export rows changes functional capacity, scales and gradients even if tensor count/width stays fixed. The eventual exact dispersion definition, typed moment/layout policy, support/empty behavior and source closure must be prospectively specified. Independent full attention-dependent adjoints, equal-mean/different-spread witnesses, constant/empty/NaN cases, actual GPU learning and saved cutoff/resume proofs are required before any new quality. No V11 code, card, training or quality run is admitted by this proposal. The goal remains active; no overall winner or application change is selected.