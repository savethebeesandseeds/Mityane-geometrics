# CPB-v4: fresh causal training and fixed-head measurements

CPB-v4 starts every learned tensor from its declared fresh constructor, with source updates0 and new TRAIN-only input/bank statistics. Initial0 and final2048 embeddings and heads are newly measured. Pretrained historical encoders supply no quality target, floor, teacher, weights or fitted statistics.

Trust admission covers fresh ancestry, actual CUDA learning, typed resume and saved initial/final causal proofs. Complete finite fixed-head results with full coverage establish a measured clean causal candidate; CPB-v3 remains the clean baseline. The separate clean-baseline comparison evaluates all 22 frozen advancement conditions. No overall winner or automatic promotion is declared.

The sole controlled change stops the variance input gradient at change4. The conditional variance head still learns and its forward dependence is preserved; mean/residual gradients remain attached. Gaussian NLL values use the unchanged formula, but encoder updates use its mean-path partial gradient rather than the full joint NLL gradient.

One CUDA:0 trajectory uses2048 updates/B8. Four market outputs use fixed ridge1 and tanh16/Adam0.01/100 updates, TRAIN112 / known chronological VALIDATION32. Repetitions2701/2802/2903 retain the exact D36 seeds, with148/660 head parameters. Both stages fit their own heads; there is no initial score reuse or post-encoder PCA. Raw controls are reused encoder-free results at their original larger input sizes. Their original head sizes are (D+1)×4 linear and16×(D+1)+68 neural. These targets measure retained past-window information, not forecasting.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.61655559 | 0.26788204 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.75086921 | 0.02651728 | 100.00 |
| CPB-v4.market-01.init | 36 | 0.45161550 | 0.27158568 | 100.00 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.55517623 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.0001457632481993121 | 0.00020920228421032871 | 0.00023436261930595529 | 0.00028233840324221054 |
| Raw data — no encoder (full256) | 0.00012113393348219792 | 0.00016862770230196986 | 0.00027850334296948618 | 0.00032813516320482082 |
| CPB-v4.market-01.init | 0.00020329626814686734 | 0.00025018300246663807 | 0.0002323444353786061 | 0.00028403651233036893 |
| CPB-v4.market-01 | 0.00010309772654730662 | 0.00013103692356775142 | 0.00017593986478183261 | 0.00022023279862042359 |

CPB-v4.market-01.init neural R² for repetitions2701/2802/2903: 0.10569529906697439, 0.5993227865507641, 0.10973894966309383.

CPB-v4.market-01 neural R² for repetitions2701/2802/2903: 0.27550702165012397, 0.68909438026645409, 0.70092728541468285.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.07548049 | -0.03159348 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.36673586 | -0.62097135 | 100.00 |
| CPB-v4.market-01.init | 36 | -0.79042258 | -1.29101305 | 100.00 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.56987523 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 4.2538095061369338e-05 | 6.4386877535939344e-05 | 5.249607559998899e-05 | 6.7754225595132475e-05 |
| Raw data — no encoder (full256) | 4.0222433762679972e-05 | 5.3288288447226955e-05 | 6.7286706934599854e-05 | 8.4157641119802349e-05 |
| CPB-v4.market-01.init | 6.4356509338262314e-05 | 8.9601874607961695e-05 | 7.4851659770490132e-05 | 0.00010004134608569896 |
| CPB-v4.market-01 | 1.5979596504966024e-05 | 2.0397732125200584e-05 | 3.499181232931064e-05 | 4.3266197431183492e-05 |

CPB-v4.market-01.init neural R² for repetitions2701/2802/2903: -1.4400303669384704, -0.36449282120763549, -2.0685159767141794.

CPB-v4.market-01 neural R² for repetitions2701/2802/2903: 0.64055209499749255, 0.35811636672973135, 0.71095723038427039.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.52820248 | 0.41069546 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.65964729 | 0.32128228 | 100.00 |
| CPB-v4.market-01.init | 36 | 0.16845197 | 0.60751248 | 100.00 |
| CPB-v4.market-01 | 36 | 0.83514307 | 0.68737055 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.30772366365230841 | 0.39473744667109539 | 0.37233804145499749 | 0.44020959039793733 |
| Raw data — no encoder (full256) | 0.27840914010158291 | 0.33527031837769694 | 0.37995325638639649 | 0.47290779350881545 |
| CPB-v4.market-01.init | 0.4231467113802751 | 0.52405174171614266 | 0.28884576972561293 | 0.35656206046360589 |
| CPB-v4.market-01 | 0.18614467583169114 | 0.23333726255631509 | 0.25168436472763089 | 0.31249136498691593 |

CPB-v4.market-01.init neural R² for repetitions2701/2802/2903: 0.68230477909300413, 0.6924019526619386, 0.44783072225932108.

CPB-v4.market-01 neural R² for repetitions2701/2802/2903: 0.86826703651099679, 0.63522616762538675, 0.55861843089646523.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | -1.18503359 | 0.09860018 | 100.00 |
| Raw data — no encoder (full256) | 13824 | -0.33498693 | -0.01139620 | 100.00 |
| CPB-v4.market-01.init | 36 | 0.08343718 | -0.10617897 | 100.00 |
| CPB-v4.market-01 | 36 | 0.32640507 | 0.20951046 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.10081525450811848 | 0.12740459195953027 | 0.065765884738787181 | 0.081095482611407546 |
| Raw data — no encoder (full256) | 0.083363206746537438 | 0.09958513805506701 | 0.067221551946341093 | 0.086505902968647921 |
| CPB-v4.market-01.init | 0.069197289610486656 | 0.082515804883195681 | 0.073010465514330283 | 0.090273208956608078 |
| CPB-v4.market-01 | 0.056297452839669512 | 0.070738431929249612 | 0.061592812121547401 | 0.076622346597282495 |

CPB-v4.market-01.init neural R² for repetitions2701/2802/2903: -0.11773962267266325, -0.3444212979580612, 0.14362401118698986.

CPB-v4.market-01 neural R² for repetitions2701/2802/2903: 0.18221053804312992, 0.23983438970069992, 0.20648646572074181.

## Hard timing information

TEMPO-5 has cohorts941101/941102/941103, each512 TRAIN examples from256 paired sources and256 VALIDATION examples from128 distinct paired sources. Both stages use newly fitted three-seed fixed heads per policy/cohort. Classification heads contain74/626 parameters. Synthetic TRAIN input statistics are freshly fitted once per cohort and shared by initial/final; the secondary policy uses this run’s fresh market TRAIN scaler. No synthetic encoder training occurs.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 94.0104 | 92.7517 | 100.0000 |
| CPB-v4.market-01 | 36 | 98.3073 | 96.9618 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 88.5417 | 85.8941 | 100.0000 |
| CPB-v4.market-01 | 36 | 94.6615 | 93.3594 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 94.7917 | 93.5764 | 100.0000 |
| CPB-v4.market-01 | 36 | 99.3490 | 98.8281 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 88.1510 | 83.9410 | 100.0000 |
| CPB-v4.market-01 | 36 | 95.8333 | 95.6163 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. initial, synthetic_train; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.9453125 | 0.9375 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.87890625 | 0.875 | 1 |
| 941101 | rep-2802 | validation_intact | 0.9453125 | 0.9296875 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.87890625 | 0.8359375 | 1 |
| 941101 | rep-2903 | validation_intact | 0.9453125 | 0.9296875 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.87890625 | 0.8515625 | 1 |
| 941102 | rep-2701 | validation_intact | 0.94140625 | 0.91796875 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.890625 | 0.8515625 | 1 |
| 941102 | rep-2802 | validation_intact | 0.94140625 | 0.9296875 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.890625 | 0.8515625 | 1 |
| 941102 | rep-2903 | validation_intact | 0.94140625 | 0.9296875 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.890625 | 0.859375 | 1 |
| 941103 | rep-2701 | validation_intact | 0.93359375 | 0.9296875 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.88671875 | 0.85546875 | 1 |
| 941103 | rep-2802 | validation_intact | 0.93359375 | 0.91015625 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.88671875 | 0.88671875 | 1 |
| 941103 | rep-2903 | validation_intact | 0.93359375 | 0.93359375 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.88671875 | 0.86328125 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. initial, market_scaler; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.95703125 | 0.953125 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.875 | 0.8515625 | 1 |
| 941101 | rep-2802 | validation_intact | 0.95703125 | 0.9453125 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.875 | 0.83203125 | 1 |
| 941101 | rep-2903 | validation_intact | 0.95703125 | 0.94921875 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.875 | 0.81640625 | 1 |
| 941102 | rep-2701 | validation_intact | 0.93359375 | 0.92578125 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.86328125 | 0.8125 | 1 |
| 941102 | rep-2802 | validation_intact | 0.93359375 | 0.92578125 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.86328125 | 0.8203125 | 1 |
| 941102 | rep-2903 | validation_intact | 0.93359375 | 0.9140625 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.86328125 | 0.8125 | 1 |
| 941103 | rep-2701 | validation_intact | 0.953125 | 0.94140625 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.90625 | 0.890625 | 1 |
| 941103 | rep-2802 | validation_intact | 0.953125 | 0.92578125 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.90625 | 0.859375 | 1 |
| 941103 | rep-2903 | validation_intact | 0.953125 | 0.94140625 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.90625 | 0.859375 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. final, synthetic_train; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 0.9921875 | 0.96875 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.94921875 | 0.94921875 | 1 |
| 941101 | rep-2802 | validation_intact | 0.9921875 | 0.9765625 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.94921875 | 0.9453125 | 1 |
| 941101 | rep-2903 | validation_intact | 0.9921875 | 0.96875 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.94921875 | 0.9453125 | 1 |
| 941102 | rep-2701 | validation_intact | 0.9921875 | 0.97265625 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.9609375 | 0.9140625 | 1 |
| 941102 | rep-2802 | validation_intact | 0.9921875 | 0.98046875 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.9609375 | 0.94140625 | 1 |
| 941102 | rep-2903 | validation_intact | 0.9921875 | 0.96875 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.9609375 | 0.9296875 | 1 |
| 941103 | rep-2701 | validation_intact | 0.96484375 | 0.96484375 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.9296875 | 0.91796875 | 1 |
| 941103 | rep-2802 | validation_intact | 0.96484375 | 0.96484375 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.9296875 | 0.9296875 | 1 |
| 941103 | rep-2903 | validation_intact | 0.96484375 | 0.9609375 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.9296875 | 0.9296875 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. final, market_scaler; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 1 | 1 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.95703125 | 0.96875 | 1 |
| 941101 | rep-2802 | validation_intact | 1 | 1 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.95703125 | 0.96484375 | 1 |
| 941101 | rep-2903 | validation_intact | 1 | 1 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.95703125 | 0.95703125 | 1 |
| 941102 | rep-2701 | validation_intact | 0.9921875 | 0.9921875 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.94921875 | 0.94140625 | 1 |
| 941102 | rep-2802 | validation_intact | 0.9921875 | 0.98828125 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.94921875 | 0.94140625 | 1 |
| 941102 | rep-2903 | validation_intact | 0.9921875 | 0.98828125 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.94921875 | 0.94140625 | 1 |
| 941103 | rep-2701 | validation_intact | 0.98828125 | 0.96875 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.96875 | 0.9609375 | 1 |
| 941103 | rep-2802 | validation_intact | 0.98828125 | 0.9765625 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.96875 | 0.96484375 | 1 |
| 941103 | rep-2903 | validation_intact | 0.98828125 | 0.98046875 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.96875 | 0.96484375 | 1 |

## Training, fixed queries and separate cost

Errors use the immutable saved hidden reconstruction queries with hierarchical standardized MAE. Gaussian NLL is reported separately and can be negative. The synchronized step timer covers forward/NLL/backward/clip/AdamW and in-step checks; loading, masking, checkpoint/resume I/O, evidence, extraction and readout fitting remain separate.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Fixed saved TRAIN/VALIDATION query; standardized MAE, source updates0.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 0 | 0.66416785662510747 | 0.95806493432361817 | 0 |
| CPB-v4.market-01 | 2048 | 0.27250803234958165 | 0.30831692401624544 | 172.86005468699975 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same saved fixed queries; Gaussian waveform NLL, distinct from MAE.

| Encoder | Updates | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- | --- |
| CPB-v4.market-01.init | 0 | 0.4238153534628793 | 0.79537651115920871 |
| CPB-v4.market-01 | 2048 | -0.35669277205499406 | -0.045381488090816241 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. and Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. One-run descriptive cost; components are not interchangeable.

| Stage | Seconds |
| --- | --- |
| Fresh market TRAIN calibration including input/witness I/O | 4.0218010030000002 |
| Initial+final export including fresh synthetic statistics and input/state I/O | 87.246106992002751 |
| initial market CPU fixed heads and I/O | 1.44750117300282 |
| final market CPU fixed heads and I/O | 1.3217964850045973 |
| initial synthetic CPU fixed heads/scoring and I/O | 2.7086816629962414 |
| final synthetic CPU fixed heads/scoring and I/O | 2.6865203880006447 |

## Interpretation and next direction

Each initial-to-final task/head difference is recorded at full precision in `table_rows.json`. The frozen protocol measures whether fresh causal training changes useful information; it does not require retention of any pretrained encoder’s scores. Mixed or weak quality remains a valid reported candidate, with no head or checkpoint rescue. The frozen 22-condition comparison against clean CPB-v3 is reported separately; CPB-v3 remains selected. Keep this run and every earlier artifact; use a separately declared mechanism for any later improvement. The application selection is unchanged and no overall quality winner is declared.

