<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v9.market-01 detailed saved evidence

The final [evidence join](causal_fresh_v9/final-establishment.json) establishes full trust, one measured fresh causal candidate and full coverage;6/22 advancement conditions pass. V3 stays the clean baseline and sole dev encoder. The original producer JSON stays immutable with pending postmeasurement trust; this report describes the subsequently completed independent chain. Read [RESULTS](CAUSAL_FRESH_V9_RESULTS.md) for all six protected final controls plus own initial/final quality tables, and [REVIEW](CAUSAL_FRESH_V9_REVIEW.md) for the decision and limitations.

These detailed tables are literal saved report sections, not refits or newly reduced numeric arrays. The four market tasks keep separate R² and physical MAE/RMSE tables, original larger raw controls and all three neural repetitions at full precision. Timing retains both scaler policies, intact/deleted views and every cohort/head repetition. MKT-HIST-2 uses112 TRAIN/32 known chronological VALIDATION windows from BTC/ETH/SOL,4096 TRAIN calibration anchors and one2048/B8 CUDA:0 encoder run. TEMPO-5 has three source-independent cohorts941101/941102/941103; each has512 TRAIN/256 VALIDATION examples from256/128 paired source groups. The deleted view shares the intact validation groups. Neither repeated heads nor paired deletion create independent encoder runs.

Fixed ridge1 and16-tanh/Adam0.01/100-step heads retain repetitions2701/2802/2903 and D40 seeds6049515622977958931,8262941649267994858,2095371932330426401:164/724 market and82/690 timing parameters. Both checkpoints receive new embeddings/heads, no initial score reuse. All scales are TRAIN-only. The source seed942005 saved query plans are used directly; query943005 is not sampled. No PCA, head tuning, prior model/normalizer import, old encoder replay or fresh-period test is performed. Numeric complexity levels are ordinal recipe descriptions only.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.61655559 | 0.26788204 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.75086921 | 0.02651728 | 100.00 |
| CPB-v9.market-01.init | 40 | 0.26294158 | -0.36300499 | 100.00 |
| CPB-v9.market-01 | 40 | 0.90992222 | 0.61852296 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.0001457632481993121 | 0.00020920228421032871 | 0.00023436261930595529 | 0.00028233840324221054 |
| Raw data — no encoder (full256) | 0.00012113393348219792 | 0.00016862770230196986 | 0.00027850334296948618 | 0.00032813516320482082 |
| CPB-v9.market-01.init | 0.00025055934059789686 | 0.0002900455267242687 | 0.00033064468709028631 | 0.00039183500147358719 |
| CPB-v9.market-01 | 8.374560038259041e-05 | 0.00010139675442362568 | 0.00016564358658353818 | 0.00019914782760568 |

CPB-v9.market-01.init neural R² for repetitions2701/2802/2903: -0.040167667755234282, -0.79467427575727778, -0.254173032572929.

CPB-v9.market-01 neural R² for repetitions2701/2802/2903: 0.75645983927471727, 0.81728114750943837, 0.28182788764527578.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.07548049 | -0.03159348 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.36673586 | -0.62097135 | 100.00 |
| CPB-v9.market-01.init | 40 | -0.16691292 | -0.56215422 | 100.00 |
| CPB-v9.market-01 | 40 | 0.66081920 | -0.13367387 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 4.2538095061369338e-05 | 6.4386877535939344e-05 | 5.249607559998899e-05 | 6.7754225595132475e-05 |
| Raw data — no encoder (full256) | 4.0222433762679972e-05 | 5.3288288447226955e-05 | 6.7286706934599854e-05 | 8.4157641119802349e-05 |
| CPB-v9.market-01.init | 4.7890874995537686e-05 | 7.2336675702662165e-05 | 6.5665639526452536e-05 | 8.3085055160316454e-05 |
| CPB-v9.market-01 | 3.0269737531295928e-05 | 3.8999133191908177e-05 | 5.8180396630821728e-05 | 6.9751324011107688e-05 |

CPB-v9.market-01.init neural R² for repetitions2701/2802/2903: -0.8810937230834377, -0.74511489507803841, -0.060254036024300017.

CPB-v9.market-01 neural R² for repetitions2701/2802/2903: 0.1203974960089037, -0.81807504548409504, 0.2966559283843212.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.52820248 | 0.41069546 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.65964729 | 0.32128228 | 100.00 |
| CPB-v9.market-01.init | 40 | 0.47793118 | 0.54183249 | 100.00 |
| CPB-v9.market-01 | 40 | 0.83053287 | 0.60329485 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.30772366365230841 | 0.39473744667109539 | 0.37233804145499749 | 0.44020959039793733 |
| Raw data — no encoder (full256) | 0.27840914010158291 | 0.33527031837769694 | 0.37995325638639649 | 0.47290779350881545 |
| CPB-v9.market-01.init | 0.33226073894129032 | 0.41523540869404407 | 0.30863033067568429 | 0.38557242044955425 |
| CPB-v9.market-01 | 0.21066512118066966 | 0.23657739109926865 | 0.2898182269297801 | 0.36146398283866338 |

CPB-v9.market-01.init neural R² for repetitions2701/2802/2903: 0.46838905936437514, 0.45346021527006519, 0.70364818411382757.

CPB-v9.market-01 neural R² for repetitions2701/2802/2903: 0.56790198197628372, 0.66058400463235822, 0.58139856708350313.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native40 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | -1.18503359 | 0.09860018 | 100.00 |
| Raw data — no encoder (full256) | 13824 | -0.33498693 | -0.01139620 | 100.00 |
| CPB-v9.market-01.init | 40 | 0.29558925 | -0.06594906 | 100.00 |
| CPB-v9.market-01 | 40 | 0.53169120 | 0.32655005 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.10081525450811848 | 0.12740459195953027 | 0.065765884738787181 | 0.081095482611407546 |
| Raw data — no encoder (full256) | 0.083363206746537438 | 0.09958513805506701 | 0.067221551946341093 | 0.086505902968647921 |
| CPB-v9.market-01.init | 0.058987775053747998 | 0.072338418333731072 | 0.071894606008359715 | 0.088802710622309947 |
| CPB-v9.market-01 | 0.050756518277857307 | 0.058982365270566557 | 0.055639302623323929 | 0.070003039522161858 |

CPB-v9.market-01.init neural R² for repetitions2701/2802/2903: 0.029204570193499535, -0.26357952487012359, 0.036527770809100857.

CPB-v9.market-01 neural R² for repetitions2701/2802/2903: 0.46601771163285322, 0.043039595069326309, 0.47059283895806892.

## Hard timing information

TEMPO-5 has cohorts941101/941102/941103, each512 TRAIN examples from256 paired sources and256 VALIDATION examples from128 distinct paired sources. Both stages use newly fitted three-seed fixed heads per policy/cohort. Classification heads contain82/690 parameters. Synthetic TRAIN input statistics are freshly fitted once per cohort and shared by initial/final; the secondary policy uses this run’s fresh market TRAIN scaler. No synthetic encoder training occurs.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v9.market-01.init | 40 | 94.9219 | 94.2274 | 100.0000 |
| CPB-v9.market-01 | 40 | 98.6979 | 97.8733 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v9.market-01.init | 40 | 89.0625 | 87.2830 | 100.0000 |
| CPB-v9.market-01 | 40 | 95.5729 | 93.6198 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v9.market-01.init | 40 | 94.6615 | 94.1840 | 100.0000 |
| CPB-v9.market-01 | 40 | 99.8698 | 98.4809 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v9.market-01.init | 40 | 88.5417 | 84.4618 | 100.0000 |
| CPB-v9.market-01 | 40 | 93.3594 | 92.8385 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. initial, synthetic_train; every intact/deleted VALIDATION repetition; exact accuracy fractions.

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

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. initial, market_scaler; every intact/deleted VALIDATION repetition; exact accuracy fractions.

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

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. final, synthetic_train; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.98828125 | 0.984375 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.9453125 | 0.93359375 | 1 |
| 941101 | rep-2802 | validation_intact | 0.98828125 | 0.99609375 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.9453125 | 0.9453125 | 1 |
| 941101 | rep-2903 | validation_intact | 0.98828125 | 0.9921875 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.9453125 | 0.9453125 | 1 |
| 941102 | rep-2701 | validation_intact | 0.9921875 | 0.9765625 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.96484375 | 0.9453125 | 1 |
| 941102 | rep-2802 | validation_intact | 0.9921875 | 0.96484375 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.96484375 | 0.92578125 | 1 |
| 941102 | rep-2903 | validation_intact | 0.9921875 | 0.96484375 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.96484375 | 0.92578125 | 1 |
| 941103 | rep-2701 | validation_intact | 0.98046875 | 0.97265625 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.95703125 | 0.94921875 | 1 |
| 941103 | rep-2802 | validation_intact | 0.98046875 | 0.9765625 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.95703125 | 0.92578125 | 1 |
| 941103 | rep-2903 | validation_intact | 0.98046875 | 0.98046875 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.95703125 | 0.9296875 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. final, market_scaler; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 1 | 0.98828125 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.91015625 | 0.9375 | 1 |
| 941101 | rep-2802 | validation_intact | 1 | 0.984375 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.91015625 | 0.94921875 | 1 |
| 941101 | rep-2903 | validation_intact | 1 | 0.98828125 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.91015625 | 0.9453125 | 1 |
| 941102 | rep-2701 | validation_intact | 1 | 0.9609375 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.9453125 | 0.90625 | 1 |
| 941102 | rep-2802 | validation_intact | 1 | 0.9765625 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.9453125 | 0.90625 | 1 |
| 941102 | rep-2903 | validation_intact | 1 | 0.9765625 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.9453125 | 0.90234375 | 1 |
| 941103 | rep-2701 | validation_intact | 0.99609375 | 0.99609375 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.9453125 | 0.9296875 | 1 |
| 941103 | rep-2802 | validation_intact | 0.99609375 | 0.99609375 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.9453125 | 0.953125 | 1 |
| 941103 | rep-2903 | validation_intact | 0.99609375 | 0.99609375 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.9453125 | 0.92578125 | 1 |

## Training, fixed queries and separate cost

Errors use the immutable saved hidden reconstruction queries with hierarchical standardized MAE. Gaussian NLL is an unchanged reporting metric and can be negative; the optimized total is Huber1 plus detached-error variance NLL. The synchronized step timer covers forward/objective/backward/clip/AdamW and in-step checks; loading, masking, checkpoint/resume I/O, evidence, extraction and readout fitting remain separate.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Fixed saved TRAIN/VALIDATION query; standardized MAE, source updates0.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v9.market-01.init | 0 | 0.66425819770853367 | 0.95821285302915538 | 0 |
| CPB-v9.market-01 | 2048 | 0.24454034650075321 | 0.32610751498844048 | 147.32844396600021 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same saved fixed queries; Gaussian waveform NLL, distinct from MAE.

| Encoder | Updates | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- | --- |
| CPB-v9.market-01.init | 0 | 0.42384730158338346 | 0.79549181757104681 |
| CPB-v9.market-01 | 2048 | -0.39245162593123745 | 0.47538792454983048 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same fixed query masks; descriptive objective components, not additional advancement criteria. Variance and total may be negative.

| Encoder | Split | Mean Huber1 | Detached-error variance NLL | Total objective |
| --- | --- | --- | --- | --- |
| CPB-v9.market-01.init | train | 0.3237604271966214 | 0.42384730158338346 | 0.74760772878000492 |
| CPB-v9.market-01.init | validation | 0.58273177713710489 | 0.79549181757104681 | 1.3782235947081518 |
| CPB-v9.market-01 | train | 0.083059716065924286 | -0.39245162593123745 | -0.30939190986531312 |
| CPB-v9.market-01 | validation | 0.11735324032731209 | 0.47538792454983048 | 0.59274116487714246 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. and Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. One-run descriptive cost; components are not interchangeable.

| Stage | Seconds |
| --- | --- |
| Fresh market TRAIN calibration including input/witness I/O | 3.0297476269999999 |
| Initial+final export including fresh synthetic statistics and input/state I/O | 71.207207712999661 |
| initial market CPU fixed heads and I/O | 1.2260921310007689 |
| final market CPU fixed heads and I/O | 1.1312482829998771 |
| initial synthetic CPU fixed heads/scoring and I/O | 2.2698193190008169 |
| final synthetic CPU fixed heads/scoring and I/O | 2.2994907649990637 |

## Finer temporal learning and causal admission

The [temporal receipt](../output/runs/causal-fresh-v9/run-001/temporal-projection-learning.json) binds8192 four-bin rows,2048 common-gauge rows and2048 nine-column between-bin rows. All four eight-row export groups and four centered within-bin position contrasts have recorded data-gradient/movement events. The three unit Hadamard directions over16 positions use bin signs++--,+-+-,+--+ repeated four times and divided by4, with F64 operands before projection/subtraction. Each direction records2048 nonzero data-gradient events and2048 movement events. Saved final signed changes are−0.15708683710545301,0.009199167601764202,−0.1647594692185521; the overall projection-change L2 is0.2278301599295833. Individual final coefficients are descriptive, not extra nonzero gates. Observed post-global-clip gradients are not an independently reconstructed upstream gradient/Jacobian. Actual optimization-step1/2048 witnesses reuse the existing forward and twelve parameter/gradient arrays; no extra encoder call is added.

All76 learned tensors remain registered/optimized. The75 identifiable tensor events cover294751 values of capacity; they do not prove every scalar moved or had a nonzero gradient. The common scalar pool_score.bias remains a softmax gauge. Its tiny floating recorded gradients and decay-related movement do not establish data learning. There is no parameter deletion or optimizer-independence claim.

The initialized and saved final artifacts each pass all128 cutoffs,2350 future-value/future-mask suffix cases and30 internal temporal cases, with state unchanged and no extra training updates. Causal attention/backward relation-bank semantics stay intact. Four slice coordinates describe the current endpoint; a global context contribution may use later-in-window past states only within that endpoint cutoff. Engineering passes303159 checks/212 typed negative cases/7372 CUDA calls, plus exact resume. Actual main typed resume512 is exact, and strict policy/partition/mix/empty metadata bind inference root/payload and TRAIN root/encoder/encoder.payload. The dedicated pooling fixture has one backbone encode, five direct kernel calls and one synthetic optimizer step; it is not the main trajectory or a head fit.

All13 initial pairings versus V8 pass: eight DATA feature archives compare only native coordinates20:40 and whole declared masks/IDs/labels, and five fresh-statistic archives compare whole statistics. Content0:20 is expected to differ. Neither whole40 equality nor trained-weight reuse is claimed. Both stages' fixed heads are independently fitted anew.

## Independent audit, timing and preservation boundaries

Training audit:25555 core +61153 numeric statistics +1177 saved tensor state +117864 Gaussian arithmetic +69301 actual pre-step components +51419 temporal learning =326469 checks, zero mismatches. Quality audit002:13411 checks, zero mismatches. These are verification checks, not additional training examples or measured models. The [final join](causal_fresh_v9/final-establishment.json) binds the audits and explicit reporting recovery; trust/full coverage pass, advancement/winner selection do not.

Dataset: **MKT-HIST-2** · past-window retention · **complexity 4/5** · TRAIN112/known VALIDATION32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms and structured distractors · **complexity 5/5** · three cohorts each TRAIN512/VALIDATION256. Descriptive one-run cost scopes; no repeated timing benchmark.

| Recorded scope | Seconds | Interpretation |
| --- | ---: | --- |
| Synchronized main encoder training | 147.32844396600021 | Forward/objective/backward/global clip/AdamW/in-step checks; excludes external I/O/evidence |
| Training trajectory wall | 196.732696927 | Includes loader/masking/checkpoint/evidence overhead |
| Outer main training wrapper | 206.1132579289988 | Broader process wall; not compute-only training time |
| Gradient/parameter evidence logging | 43.56861853400001 | Recorded separately outside synchronized step timer |
| Original quality wrapper, failed at report guard | 78.24492377099887 | Five scientific stages completed once; preserves failed disposition |
| External report/comparison recovery | 0.926021336 | Saved JSON aggregation only, no encoder/head/update calls |
| Independent quality audit002 | 3.361644904 | Saved-data verification only |
| Engineering proof | 337.89078325599985 | Separate initialized/trained proof fixture, not main-training compute |
| Initial saved causal proof | 216.33056785299777 | Read-only saved checkpoint checks |
| Final saved causal proof | 168.62452934799876 | Read-only saved checkpoint checks |

Do not sum overlapping scopes into a new end-to-end benchmark. Export/head/calibration times above remain their own scopes; proof processes and audit work are not training compute. No memory/inference-latency benchmark or FLOP measurement was declared. Parameter capacity is unchanged from V8, while the mixture changes routing/scales/work; no speed superiority follows from one training timer.

## Frozen22 conditions and earlier references

The following decision table is copied from the saved comparison. It retains the exact e792 strongest-V3–V8 floors and operators. No TRAIN-NLL gate is added; historical pass counts keep their original cards.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. All22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Frozen floor/ceiling | Source | Rule | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.90992222 | 0.964673434 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.618522958 | 0.735969246 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:linear_r2 | 0.660819198 | 0.90721343 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:neural_mean_r2 | -0.133673874 | 0.569875231 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:linear_r2 | 0.83053287 | 0.862840311 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.603294851 | 0.687370545 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:linear_r2 | 0.531691195 | 0.622100687 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.326550049 | 0.370505348 | CPB-v5.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.986979167 | 0.986979167 | CPB-v6.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.978732639 | 0.984809028 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.955729167 | 0.953125 | CPB-v6.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.936197917 | 0.934027778 | CPB-v6.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.998697917 | 0.997395833 | CPB-v6.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.984809028 | 0.992621528 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.93359375 | 0.97265625 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.928385417 | 0.956163194 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | train:standardized_MAE | 0.244540347 | 0.241805338 | CPB-v7.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | validation:standardized_MAE | 0.326107515 | 0.308316924 | CPB-v4.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | 0.475387925 | -0.0453814881 | CPB-v4.market-01 | candidate < floor - 1e-12 | FAIL |
| MKT-HIST-2 | past_trend:all3_neural_R2_positive | 0.120397496, -0.818075045, 0.296655928 | 0 | strict positivity | all three final neural trend R2 values > 0 | FAIL |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.531691195 | 0 | strict positivity | candidate > 0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.326550049 | 0 | strict positivity | candidate > 0 | PASS |

## Source and failure chain

The scientific card94488bf2…, source manifest43a44dbb… and ZIPb54ffb00… remain exact:118 captured sources,82 family/research owner files,7 binaries and11 translation units. They are separate from postmeasurement/auditor/reporting recovery sources. The full card and manifests are linked below; prefix hashes here are readable labels, not substitute bindings.

The original quality/report stages remain failed. All five extraction/fixed-head stages and531 pre-recovery files were preserved before the external report guard repair. Recovery declared9c61e451… and receipt5f96dfab… bind the same118 scientific sources and all original files unchanged, with zero scientific-stage replay. The repaired reporter changes metadata admission/prose compatibility only; the original comparator and all metrics remain unchanged. Audit001 subsequently refused before numeric/archive checks on a host-label/runtime-path selector; its card/source/Stage/logs/execution and failed disposition remain. Additive auditor002 changes only supplied-runtime-path binding, then passes. The separate receipts make these workflow repairs visible rather than replacing failed history.

All five main checkpoints are available locally and ignored. The Complete mirror verifies549 files/596796521 bytes, including442 ignored PT archives; PT includes heads/data/tests and is not a trained-encoder count. The earlier94 critical checkpoint pins and47 selected V3 sources in each repo were protected before the build. Final99-checkpoint/whole-history preservation is a separate postmetadata receipt and must not be presumed complete before it runs. No DEV promotion, Git mutation/publication, deletion or cleanup is performed here.

- [Prospective card](CAUSAL_FRESH_V9_CARD.json), [freeze](causal_fresh_v9/prospective-freeze.json), [source preflight](causal_fresh_v9/protocol-preflight.json).
- [Saved tables](../output/runs/causal-fresh-v9/run-001/quality-001/report/table_rows.json), [comparison](../output/runs/causal-fresh-v9/run-001/quality-001/comparison/comparison.json), [quality execution](../output/runs/causal-fresh-v9/run-001/quality-001/quality-receipt.json).
- [Training audit](causal_fresh_v9/independent-training-audit.json), [quality audit](causal_fresh_v9/quality-independent-audit.json), [final trust join](causal_fresh_v9/final-establishment.json).
- [Failure preservation](causal_fresh_v9/failure-mirror-before-report-recovery-001.json), [report recovery declaration](causal_fresh_v9/reporting-recovery-card-001.json), [report-only receipt](causal_fresh_v9/reporting-recovery-receipt-001.json), [failed audit001 mirror](causal_fresh_v9/quality-audit-mirror-attempt001.json).
- [Complete mirror](causal_fresh_v9/complete-mirror-attempt-001.json), [version-owned source](../code/encoders/causal_patch_bottleneck_mae/versions/v9/README.md).
