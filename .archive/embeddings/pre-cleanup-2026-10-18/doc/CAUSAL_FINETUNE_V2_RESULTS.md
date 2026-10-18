# CPB-v2: causal fine-tuning and fixed-head results

**CPB-v1.market-01 remains the frozen application baseline.** CPB-v2 is a separately named research candidate. The baseline initial embeddings and head scores are reused after exact initial parity, with zero initial refits. Only the final checkpoint receives one new quality extraction and fixed-head pass.

Market fitting: TRAIN112 / known VALIDATION32, CUDA:0 batch8, native36; one ridge pipeline/four target fits and three tanh16/Adam0.01/100-update neural fits. Repetitions2701/2802/2903 retain exact D36 seeds; regression heads148/660 parameters. The targets measure retained past-window information, not forecasting.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; R² is dimensionless.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.61655559 | 0.26788204 | 100.00 |
| RPB-v24.market-01.alt-01 | 36 | 0.44631541 | 0.39261419 | 100.00 |
| CPB-v1.market-01 | 36 | 0.95442168 | 0.60593058 | 100.00 |
| CPB-v2.market-01 | 36 | 0.92924761 | 0.46229551 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.0001457632481993121 | 0.00020920228421032871 | 0.00023436261930595529 | 0.00028233840324221054 |
| RPB-v24.market-01.alt-01 | 0.00019273861126322043 | 0.00025138909498817691 | 0.00021007226044545535 | 0.00026201098576414281 |
| CPB-v1.market-01 | 5.4406701903979004e-05 | 7.212641896829908e-05 | 0.00016624754519196607 | 0.00021158217223371328 |
| CPB-v2.market-01 | 7.1651183688005421e-05 | 8.9864003034231766e-05 | 0.00020002600996599581 | 0.00024762648516590878 |

Every final neural R², repetitions2701/2802/2903: 0.4840558093120586, 0.48594319674548336, 0.4168875288858862.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; R² is dimensionless.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.07548049 | -0.03159348 | 100.00 |
| RPB-v24.market-01.alt-01 | 36 | 0.65491126 | 0.13429552 | 100.00 |
| CPB-v1.market-01 | 36 | 0.79680365 | 0.24368775 | 100.00 |
| CPB-v2.market-01 | 36 | 0.79703414 | 0.23900308 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 4.2538095061369338e-05 | 6.4386877535939344e-05 | 5.249607559998899e-05 | 6.7754225595132475e-05 |
| RPB-v24.market-01.alt-01 | 3.4078053718289387e-05 | 3.9337315625123313e-05 | 5.0265262984775125e-05 | 6.2292678985356823e-05 |
| CPB-v1.market-01 | 2.4165928465186302e-05 | 3.0185423887803161e-05 | 4.4348940829996791e-05 | 5.5999286171599476e-05 |
| CPB-v2.market-01 | 2.4258157886903527e-05 | 3.0168298948841671e-05 | 4.7210174918133048e-05 | 5.6218280249865059e-05 |

Every final neural R², repetitions2701/2802/2903: -0.30107408705294536, 0.68479504330300445, 0.3332882956295069.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; R² is dimensionless.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.52820248 | 0.41069546 | 100.00 |
| RPB-v24.market-01.alt-01 | 36 | 0.75142968 | 0.56315290 | 100.00 |
| CPB-v1.market-01 | 36 | 0.87512521 | 0.68664662 | 100.00 |
| CPB-v2.market-01 | 36 | 0.86999430 | 0.69448146 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.30772366365230841 | 0.39473744667109539 | 0.37233804145499749 | 0.44020959039793733 |
| RPB-v24.market-01.alt-01 | 0.22716902256464511 | 0.28652012668984 | 0.3014507811535253 | 0.37072305659896349 |
| CPB-v1.market-01 | 0.16656231911834701 | 0.20308033874250978 | 0.2619879309461553 | 0.32169440536048749 |
| CPB-v2.market-01 | 0.15278453782484902 | 0.2072104683388368 | 0.25694199764015635 | 0.31595738726218853 |

Every final neural R², repetitions2701/2802/2903: 0.7772506035041713, 0.63174464618413273, 0.67444912219296849.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; R² is dimensionless.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | -1.18503359 | 0.09860018 | 100.00 |
| RPB-v24.market-01.alt-01 | 36 | 0.34068539 | 0.34958349 | 100.00 |
| CPB-v1.market-01 | 36 | 0.49370525 | 0.38971631 | 100.00 |
| CPB-v2.market-01 | 36 | 0.39899814 | 0.16496865 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.10081525450811848 | 0.12740459195953027 | 0.065765884738787181 | 0.081095482611407546 |
| RPB-v24.market-01.alt-01 | 0.060321601964518218 | 0.069984581991297007 | 0.057151362081462875 | 0.069393204999402047 |
| CPB-v1.market-01 | 0.049624023547790309 | 0.061327849082980085 | 0.054681411533969798 | 0.067227872559793511 |
| CPB-v2.market-01 | 0.054464319516696585 | 0.066818070692507198 | 0.065598630865627172 | 0.078586912048309879 |

Every final neural R², repetitions2701/2802/2903: 0.14012632853791329, 0.045403014480634774, 0.30937659662644068.

## Hard timing retention

TEMPO-5 uses cohorts941101/941102/941103, each512 TRAIN variants from256 sources and256 VALIDATION variants from128 distinct sources. Both saved input-scaler policies are frozen. Ridge1 and neural tanh16/Adam0.01/100 retain three D36 seeds per cohort/policy; classification heads74/626 parameters. No synthetic encoder training occurs.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| RPB-v22.alt-01 | 32 | 97.1354 | 95.9635 | 100.0000 |
| RPB-v24.market-01.alt-01 | 36 | 98.9583 | 97.7431 | 100.0000 |
| CPB-v1.market-01 | 36 | 98.8281 | 97.8733 | 100.0000 |
| CPB-v2.market-01 | 36 | 98.3073 | 97.4392 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| RPB-v22.alt-01 | 32 | 91.2760 | 91.0590 | 100.0000 |
| RPB-v24.market-01.alt-01 | 36 | 93.3594 | 92.5347 | 100.0000 |
| CPB-v1.market-01 | 36 | 92.7083 | 92.1441 | 100.0000 |
| CPB-v2.market-01 | 36 | 93.3594 | 92.8819 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| RPB-v22.alt-01 | 32 | 97.1354 | 95.9635 | 100.0000 |
| RPB-v24.market-01.alt-01 | 36 | 100.0000 | 99.5226 | 100.0000 |
| CPB-v1.market-01 | 36 | 100.0000 | 99.3490 | 100.0000 |
| CPB-v2.market-01 | 36 | 99.3490 | 99.0451 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| RPB-v22.alt-01 | 32 | 91.2760 | 91.0590 | 100.0000 |
| RPB-v24.market-01.alt-01 | 36 | 96.0938 | 95.0521 | 100.0000 |
| CPB-v1.market-01 | 36 | 96.0938 | 94.8785 | 100.0000 |
| CPB-v2.market-01 | 36 | 96.2240 | 93.5330 | 100.0000 |

## Every final synthetic repetition

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, intact/deleted VALIDATION; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.9921875 | 0.98046875 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.921875 | 0.93359375 | 1 |
| 941101 | rep-2802 | validation_intact | 0.9921875 | 0.9765625 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.921875 | 0.921875 | 1 |
| 941101 | rep-2903 | validation_intact | 0.9921875 | 0.98046875 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.921875 | 0.921875 | 1 |
| 941102 | rep-2701 | validation_intact | 0.98046875 | 0.98046875 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.9296875 | 0.92578125 | 1 |
| 941102 | rep-2802 | validation_intact | 0.98046875 | 0.9765625 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.9296875 | 0.9296875 | 1 |
| 941102 | rep-2903 | validation_intact | 0.98046875 | 0.984375 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.9296875 | 0.92578125 | 1 |
| 941103 | rep-2701 | validation_intact | 0.9765625 | 0.95703125 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.94921875 | 0.9375 | 1 |
| 941103 | rep-2802 | validation_intact | 0.9765625 | 0.96875 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.94921875 | 0.9296875 | 1 |
| 941103 | rep-2903 | validation_intact | 0.9765625 | 0.96484375 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.94921875 | 0.93359375 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, intact/deleted VALIDATION; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.99609375 | 0.98828125 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.96484375 | 0.94140625 | 1 |
| 941101 | rep-2802 | validation_intact | 0.99609375 | 0.98828125 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.96484375 | 0.9453125 | 1 |
| 941101 | rep-2903 | validation_intact | 0.99609375 | 0.98828125 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.96484375 | 0.9453125 | 1 |
| 941102 | rep-2701 | validation_intact | 1 | 0.9921875 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.9609375 | 0.91796875 | 1 |
| 941102 | rep-2802 | validation_intact | 1 | 0.99609375 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.9609375 | 0.921875 | 1 |
| 941102 | rep-2903 | validation_intact | 1 | 0.99609375 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.9609375 | 0.90234375 | 1 |
| 941103 | rep-2701 | validation_intact | 0.984375 | 0.98828125 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.9609375 | 0.953125 | 1 |
| 941103 | rep-2802 | validation_intact | 0.984375 | 0.98828125 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.9609375 | 0.94921875 | 1 |
| 941103 | rep-2903 | validation_intact | 0.984375 | 0.98828125 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.9609375 | 0.94140625 | 1 |

## Training, fixed queries and cost

The complete training/query measurements, including standardized fixed-query MAE and separate Gaussian NLL, are retained without substitution in `table_rows.json`. Per-update trajectories stay in the training artifacts. Encoder training time, frozen extraction time and CPU head fitting/scoring time are reported separately. Initial/final rows count new causal updates; inherited source updates2048 are separate. The synchronized GPU step measurement is wall time around forward/NLL/backward/clip/AdamW and includes in-step contract checks; loading, masks, checkpoints and evidence logging are excluded.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Unchanged fixed hidden-target query; errors are standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v2.market-01 | 0 | 0.27671297318712434 | 0.38879820744200222 | 0 |
| CPB-v2.market-01 | 2048 | 0.27157563278641378 | 0.41617756793800925 | 126.55104771099985 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same saved query; Gaussian waveform NLL is distinct from MAE and can be negative.

| Encoder | Updates | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- | --- |
| CPB-v2.market-01 | 0 | -0.3794596571636123 | 0.80594673045023768 |
| CPB-v2.market-01 | 2048 | -0.54815195173950104 | 1.107129785299853 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. and Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. Final frozen extraction and head-scoring cost; one run, descriptive timing.

| Stage | Seconds |
| --- | --- |
| Final CUDA extraction, full API including input/state checks | 33.374608043999999 |
| Market CPU heads/scoring including process and I/O | 2.368041373003507 |
| Synthetic CPU heads/scoring including I/O | 1.668461537 |

## Fixed prospective gates

```json
{
  "conditions": {
    "original_market": [
      {
        "criterion": "positive_linear_r2_trend",
        "floor": 0.0,
        "candidate": 0.7970341369141087,
        "passed": true
      },
      {
        "criterion": "positive_neural_mean_r2_trend",
        "floor": 0.0,
        "candidate": 0.23900308395985534,
        "passed": true
      },
      {
        "criterion": "past_volatility_linear_r2_v22_retention",
        "floor": 0.6781069298378846,
        "candidate": 0.9292476143211095,
        "passed": true
      },
      {
        "criterion": "past_volatility_neural_mean_r2_v22_retention",
        "floor": 0.3303364424760756,
        "candidate": 0.4622955116478094,
        "passed": true
      },
      {
        "criterion": "past_activity_linear_r2_v22_retention",
        "floor": 0.6784311857420708,
        "candidate": 0.8699943031113067,
        "passed": true
      },
      {
        "criterion": "past_activity_neural_mean_r2_v22_retention",
        "floor": 0.46216745965172695,
        "candidate": 0.6944814572937575,
        "passed": true
      },
      {
        "criterion": "past_relation_linear_r2_v22_retention",
        "floor": 0.3385830932968374,
        "candidate": 0.3989981375336701,
        "passed": true
      },
      {
        "criterion": "past_relation_neural_mean_r2_v22_retention",
        "floor": 0.20799583291493073,
        "candidate": 0.16496864654832957,
        "passed": false
      },
      {
        "criterion": "linear_trend_exceeds_saved_v23",
        "floor": 0.3871309972495327,
        "candidate": 0.7970341369141087,
        "passed": true
      }
    ],
    "strict_native36_market": [
      {
        "criterion": "past_volatility_linear_r2_native36_guard",
        "floor": 0.6781069298378846,
        "candidate": 0.9292476143211095,
        "passed": true
      },
      {
        "criterion": "past_volatility_neural_mean_r2_native36_guard",
        "floor": 0.3926141918082198,
        "candidate": 0.4622955116478094,
        "passed": true
      },
      {
        "criterion": "past_trend_linear_r2_native36_guard",
        "floor": 0.6549112553681312,
        "candidate": 0.7970341369141087,
        "passed": true
      },
      {
        "criterion": "past_trend_neural_mean_r2_native36_guard",
        "floor": 0.1342955201053261,
        "candidate": 0.23900308395985534,
        "passed": true
      },
      {
        "criterion": "past_activity_linear_r2_native36_guard",
        "floor": 0.7514296819362523,
        "candidate": 0.8699943031113067,
        "passed": true
      },
      {
        "criterion": "past_activity_neural_mean_r2_native36_guard",
        "floor": 0.5631529010630465,
        "candidate": 0.6944814572937575,
        "passed": true
      },
      {
        "criterion": "past_relation_linear_r2_native36_guard",
        "floor": 0.3406853885730756,
        "candidate": 0.3989981375336701,
        "passed": true
      },
      {
        "criterion": "past_relation_neural_mean_r2_native36_guard",
        "floor": 0.3495834881128966,
        "candidate": 0.16496864654832957,
        "passed": false
      }
    ],
    "causal_baseline_market": [
      {
        "criterion": "past_volatility_linear_r2",
        "floor": 0.954421680571244,
        "candidate": 0.9292476143211095,
        "passed": false
      },
      {
        "criterion": "past_volatility_neural_mean_r2",
        "floor": 0.6059305827863365,
        "candidate": 0.4622955116478094,
        "passed": false
      },
      {
        "criterion": "past_trend_linear_r2",
        "floor": 0.7968036456603806,
        "candidate": 0.7970341369141087,
        "passed": true
      },
      {
        "criterion": "past_trend_neural_mean_r2",
        "floor": 0.24368774940638713,
        "candidate": 0.23900308395985534,
        "passed": false
      },
      {
        "criterion": "past_activity_linear_r2",
        "floor": 0.8751252139115532,
        "candidate": 0.8699943031113067,
        "passed": false
      },
      {
        "criterion": "past_activity_neural_mean_r2",
        "floor": 0.6866466216076083,
        "candidate": 0.6944814572937575,
        "passed": true
      },
      {
        "criterion": "past_relation_linear_r2",
        "floor": 0.49370525358462336,
        "candidate": 0.3989981375336701,
        "passed": false
      },
      {
        "criterion": "past_relation_neural_mean_r2",
        "floor": 0.38971630978947336,
        "candidate": 0.16496864654832957,
        "passed": false
      }
    ],
    "individual_neural_trend": [
      {
        "repetition": 2701,
        "candidate": -0.30107408705294536,
        "passed": false
      },
      {
        "repetition": 2802,
        "candidate": 0.6847950433030044,
        "passed": true
      },
      {
        "repetition": 2903,
        "candidate": 0.3332882956295069,
        "passed": true
      }
    ],
    "synthetic_retention": [
      {
        "mode": "synthetic_train",
        "view": "validation_intact",
        "head": "ridge",
        "baseline": 0.98828125,
        "native36_floor": 0.9895833333333334,
        "floor": 0.9895833333333334,
        "coverage_floor": 1.0,
        "candidate": 0.9830729166666666,
        "passed": false
      },
      {
        "mode": "synthetic_train",
        "view": "validation_intact",
        "head": "tiny_secondary",
        "baseline": 0.978732638888889,
        "native36_floor": 0.9774305555555556,
        "floor": 0.978732638888889,
        "coverage_floor": 1.0,
        "candidate": 0.974392361111111,
        "passed": false
      },
      {
        "mode": "synthetic_train",
        "view": "validation_deleted",
        "head": "ridge",
        "baseline": 0.9270833333333334,
        "native36_floor": 0.93359375,
        "floor": 0.93359375,
        "coverage_floor": 1.0,
        "candidate": 0.93359375,
        "passed": true
      },
      {
        "mode": "synthetic_train",
        "view": "validation_deleted",
        "head": "tiny_secondary",
        "baseline": 0.9214409722222222,
        "native36_floor": 0.9253472222222222,
        "floor": 0.9253472222222222,
        "coverage_floor": 1.0,
        "candidate": 0.9288194444444444,
        "passed": true
      },
      {
        "mode": "market_scaler",
        "view": "validation_intact",
        "head": "ridge",
        "baseline": 1,
        "native36_floor": 1.0,
        "floor": 1.0,
        "coverage_floor": 1.0,
        "candidate": 0.9934895833333334,
        "passed": false
      },
      {
        "mode": "market_scaler",
        "view": "validation_intact",
        "head": "tiny_secondary",
        "baseline": 0.9934895833333334,
        "native36_floor": 0.9952256944444444,
        "floor": 0.9952256944444444,
        "coverage_floor": 1.0,
        "candidate": 0.9904513888888888,
        "passed": false
      },
      {
        "mode": "market_scaler",
        "view": "validation_deleted",
        "head": "ridge",
        "baseline": 0.9609375,
        "native36_floor": 0.9609375,
        "floor": 0.9609375,
        "coverage_floor": 1.0,
        "candidate": 0.9622395833333334,
        "passed": true
      },
      {
        "mode": "market_scaler",
        "view": "validation_deleted",
        "head": "tiny_secondary",
        "baseline": 0.9487847222222222,
        "native36_floor": 0.9505208333333334,
        "floor": 0.9505208333333334,
        "coverage_floor": 1.0,
        "candidate": 0.935329861111111,
        "passed": false
      }
    ]
  },
  "counts": {
    "original_market": {
      "passed": 8,
      "total": 9
    },
    "strict_native36_market": {
      "passed": 7,
      "total": 8
    },
    "causal_baseline_market": {
      "passed": 2,
      "total": 8
    },
    "individual_neural_trend": {
      "passed": 2,
      "total": 3
    },
    "synthetic_retention": {
      "passed": 3,
      "total": 8
    }
  },
  "neural_trend_mean_strict_improvement": false,
  "full_market_coverage": true,
  "full_synthetic_coverage": true,
  "overall_gate_passed": false,
  "absolute_tolerance": 1e-12
}
```

## What changed and next direction

The candidate performs separately declared causal fine-tuning from the preserved saved baseline. Its native size, input populations, scalers and fixed-head recipes remain fixed. All eight market baseline floors and the stronger of each causal/native36 synthetic floor are reported, alongside the historical gates. No application pointer or historical leader is changed by this report. A failed frozen recipe is retained and stopped; a later mechanism requires a separately declared version/card.

