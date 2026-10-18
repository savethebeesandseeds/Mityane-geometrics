# CPB-v3: fresh causal training and fixed-head measurements

CPB-v3 starts every learned tensor from its declared fresh constructor, with source updates0 and new TRAIN-only input/bank statistics. Initial0 and final2048 embeddings and heads are newly measured. Pretrained historical encoders supply no quality target, floor, teacher, weights or fitted statistics.

Trust admission covers fresh ancestry, actual CUDA learning, typed resume and saved initial/final causal proofs. Complete finite fixed-head results with full coverage establish a clean fresh-training baseline. Quality below is descriptive; no numerical quality result is treated as an advancement gate or overall accuracy.

One CUDA:0 trajectory uses2048 updates/B8. Four market outputs use fixed ridge1 and tanh16/Adam0.01/100 updates, TRAIN112 / known chronological VALIDATION32. Repetitions2701/2802/2903 retain the exact D36 seeds, with148/660 head parameters. Both stages fit their own heads; there is no initial score reuse or post-encoder PCA. Raw controls are reused encoder-free results at their original larger input sizes. Their original head sizes are (D+1)×4 linear and16×(D+1)+68 neural. These targets measure retained past-window information, not forecasting.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.61655559 | 0.26788204 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.75086921 | 0.02651728 | 100.00 |
| CPB-v3.market-01.init | 36 | 0.45161550 | 0.27158568 | 100.00 |
| CPB-v3.market-01 | 36 | 0.96467343 | 0.46305192 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.0001457632481993121 | 0.00020920228421032871 | 0.00023436261930595529 | 0.00028233840324221054 |
| Raw data — no encoder (full256) | 0.00012113393348219792 | 0.00016862770230196986 | 0.00027850334296948618 | 0.00032813516320482082 |
| CPB-v3.market-01.init | 0.00020329626814686734 | 0.00025018300246663807 | 0.0002323444353786061 | 0.00028403651233036893 |
| CPB-v3.market-01 | 5.1438244510906345e-05 | 6.3498860082341889e-05 | 0.00021558140567355788 | 0.00024583633271647962 |

CPB-v3.market-01.init neural R² for repetitions2701/2802/2903: 0.10569529906697439, 0.5993227865507641, 0.10973894966309383.

CPB-v3.market-01 neural R² for repetitions2701/2802/2903: 0.34452549910387642, 0.41467285432586121, 0.62995741473494571.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.07548049 | -0.03159348 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.36673586 | -0.62097135 | 100.00 |
| CPB-v3.market-01.init | 36 | -0.79042258 | -1.29101305 | 100.00 |
| CPB-v3.market-01 | 36 | 0.26313296 | -0.25397900 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 4.2538095061369338e-05 | 6.4386877535939344e-05 | 5.249607559998899e-05 | 6.7754225595132475e-05 |
| Raw data — no encoder (full256) | 4.0222433762679972e-05 | 5.3288288447226955e-05 | 6.7286706934599854e-05 | 8.4157641119802349e-05 |
| CPB-v3.market-01.init | 6.4356509338262314e-05 | 8.9601874607961695e-05 | 7.4851659770490132e-05 | 0.00010004134608569896 |
| CPB-v3.market-01 | 4.3723491614761766e-05 | 5.7482266990266961e-05 | 5.8573278918616742e-05 | 7.4973086124059424e-05 |

CPB-v3.market-01.init neural R² for repetitions2701/2802/2903: -1.4400303669384704, -0.36449282120763549, -2.0685159767141794.

CPB-v3.market-01 neural R² for repetitions2701/2802/2903: -0.31924515917541196, -0.23774789512546346, -0.20494394912226821.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | 0.52820248 | 0.41069546 | 100.00 |
| Raw data — no encoder (full256) | 13824 | 0.65964729 | 0.32128228 | 100.00 |
| CPB-v3.market-01.init | 36 | 0.16845197 | 0.60751248 | 100.00 |
| CPB-v3.market-01 | 36 | 0.86284031 | 0.54142606 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.30772366365230841 | 0.39473744667109539 | 0.37233804145499749 | 0.44020959039793733 |
| Raw data — no encoder (full256) | 0.27840914010158291 | 0.33527031837769694 | 0.37995325638639649 | 0.47290779350881545 |
| CPB-v3.market-01.init | 0.4231467113802751 | 0.52405174171614266 | 0.28884576972561293 | 0.35656206046360589 |
| CPB-v3.market-01 | 0.17814355298975235 | 0.21283534231476031 | 0.30355288545421949 | 0.38715069272080227 |

CPB-v3.market-01.init neural R² for repetitions2701/2802/2903: 0.68230477909300413, 0.6924019526619386, 0.44783072225932108.

CPB-v3.market-01 neural R² for repetitions2701/2802/2903: 0.55195976639400202, 0.64999403029895886, 0.42232437005499801.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Intact chronological VALIDATION; initial/final native36 and original encoder-free raw controls.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 6912 | -1.18503359 | 0.09860018 | 100.00 |
| Raw data — no encoder (full256) | 13824 | -0.33498693 | -0.01139620 | 100.00 |
| CPB-v3.market-01.init | 36 | 0.08343718 | -0.10617897 | 100.00 |
| CPB-v3.market-01 | 36 | -0.00700445 | -0.08556567 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical prediction errors; units: correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| Raw data — no encoder (recent128) | 0.10081525450811848 | 0.12740459195953027 | 0.065765884738787181 | 0.081095482611407546 |
| Raw data — no encoder (full256) | 0.083363206746537438 | 0.09958513805506701 | 0.067221551946341093 | 0.086505902968647921 |
| CPB-v3.market-01.init | 0.069197289610486656 | 0.082515804883195681 | 0.073010465514330283 | 0.090273208956608078 |
| CPB-v3.market-01 | 0.072398292894631594 | 0.086491159022817898 | 0.070219921481160796 | 0.089445883268242155 |

CPB-v3.market-01.init neural R² for repetitions2701/2802/2903: -0.11773962267266325, -0.3444212979580612, 0.14362401118698986.

CPB-v3.market-01 neural R² for repetitions2701/2802/2903: -0.30065428105903469, 0.16101463449470332, -0.1170573678474649.

## Hard timing information

TEMPO-5 has cohorts941101/941102/941103, each512 TRAIN examples from256 paired sources and256 VALIDATION examples from128 distinct paired sources. Both stages use newly fitted three-seed fixed heads per policy/cohort. Classification heads contain74/626 parameters. Synthetic TRAIN input statistics are freshly fitted once per cohort and shared by initial/final; the secondary policy uses this run’s fresh market TRAIN scaler. No synthetic encoder training occurs.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01.init | 36 | 94.0104 | 92.7517 | 100.0000 |
| CPB-v3.market-01 | 36 | 96.4844 | 95.7031 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01.init | 36 | 88.5417 | 85.8941 | 100.0000 |
| CPB-v3.market-01 | 36 | 91.1458 | 91.7535 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01.init | 36 | 94.7917 | 93.5764 | 100.0000 |
| CPB-v3.market-01 | 36 | 99.3490 | 98.1771 | 100.0000 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01.init | 36 | 88.1510 | 83.9410 | 100.0000 |
| CPB-v3.market-01 | 36 | 94.9219 | 94.2708 | 100.0000 |

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
| 941101 | rep-2701 | validation_intact | 0.98828125 | 0.9765625 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.93359375 | 0.9375 | 1 |
| 941101 | rep-2802 | validation_intact | 0.98828125 | 0.9609375 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.93359375 | 0.92578125 | 1 |
| 941101 | rep-2903 | validation_intact | 0.98828125 | 0.9765625 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.93359375 | 0.9375 | 1 |
| 941102 | rep-2701 | validation_intact | 0.953125 | 0.94140625 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.8984375 | 0.91796875 | 1 |
| 941102 | rep-2802 | validation_intact | 0.953125 | 0.94921875 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.8984375 | 0.91015625 | 1 |
| 941102 | rep-2903 | validation_intact | 0.953125 | 0.94921875 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.8984375 | 0.89453125 | 1 |
| 941103 | rep-2701 | validation_intact | 0.953125 | 0.96484375 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.90234375 | 0.91015625 | 1 |
| 941103 | rep-2802 | validation_intact | 0.953125 | 0.94921875 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.90234375 | 0.90234375 | 1 |
| 941103 | rep-2903 | validation_intact | 0.953125 | 0.9453125 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.90234375 | 0.921875 | 1 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. final, market_scaler; every intact/deleted VALIDATION repetition; exact accuracy fractions.

| Cohort | Fit | View | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- |
| 941101 | rep-2701 | validation_intact | 1 | 0.9921875 | 1 |
| 941101 | rep-2701 | validation_deleted | 0.94921875 | 0.93359375 | 1 |
| 941101 | rep-2802 | validation_intact | 1 | 0.98828125 | 1 |
| 941101 | rep-2802 | validation_deleted | 0.94921875 | 0.9296875 | 1 |
| 941101 | rep-2903 | validation_intact | 1 | 0.9921875 | 1 |
| 941101 | rep-2903 | validation_deleted | 0.94921875 | 0.94140625 | 1 |
| 941102 | rep-2701 | validation_intact | 0.98828125 | 0.97265625 | 1 |
| 941102 | rep-2701 | validation_deleted | 0.95703125 | 0.96484375 | 1 |
| 941102 | rep-2802 | validation_intact | 0.98828125 | 0.97265625 | 1 |
| 941102 | rep-2802 | validation_deleted | 0.95703125 | 0.96484375 | 1 |
| 941102 | rep-2903 | validation_intact | 0.98828125 | 0.9765625 | 1 |
| 941102 | rep-2903 | validation_deleted | 0.95703125 | 0.95703125 | 1 |
| 941103 | rep-2701 | validation_intact | 0.9921875 | 0.98046875 | 1 |
| 941103 | rep-2701 | validation_deleted | 0.94140625 | 0.94140625 | 1 |
| 941103 | rep-2802 | validation_intact | 0.9921875 | 0.98046875 | 1 |
| 941103 | rep-2802 | validation_deleted | 0.94140625 | 0.92578125 | 1 |
| 941103 | rep-2903 | validation_intact | 0.9921875 | 0.98046875 | 1 |
| 941103 | rep-2903 | validation_deleted | 0.94140625 | 0.92578125 | 1 |

## Training, fixed queries and separate cost

Errors use the immutable saved hidden reconstruction queries with hierarchical standardized MAE. Gaussian NLL is reported separately and can be negative. The synchronized step timer covers forward/NLL/backward/clip/AdamW and in-step checks; loading, masking, checkpoint/resume I/O, evidence, extraction and readout fitting remain separate.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Fixed saved TRAIN/VALIDATION query; standardized MAE, source updates0.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01.init | 0 | 0.66416785662510747 | 0.95806493432361817 | 0 |
| CPB-v3.market-01 | 2048 | 0.25904599043427573 | 0.37641423057615903 | 144.34111624900029 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same saved fixed queries; Gaussian waveform NLL, distinct from MAE.

| Encoder | Updates | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- | --- |
| CPB-v3.market-01.init | 0 | 0.4238153534628793 | 0.79537651115920871 |
| CPB-v3.market-01 | 2048 | -0.38017807983375768 | 1.521796577918092 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. and Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. One-run descriptive cost; components are not interchangeable.

| Stage | Seconds |
| --- | --- |
| Fresh market TRAIN calibration including input/witness I/O | 2.9916498730000001 |
| Initial+final export including fresh synthetic statistics and input/state I/O | 70.202964225005417 |
| initial market CPU fixed heads and I/O | 1.3098437119988375 |
| final market CPU fixed heads and I/O | 1.2524065260004136 |
| initial synthetic CPU fixed heads/scoring and I/O | 2.3774248179979622 |
| final synthetic CPU fixed heads/scoring and I/O | 2.4179064359996119 |

## Interpretation and next direction

Each initial-to-final task/head difference is recorded at full precision in `table_rows.json`. The frozen protocol measures whether fresh causal training changes useful information; it does not require retention of any pretrained encoder’s scores. Mixed or weak quality remains a valid reported baseline, with no head or checkpoint rescue. Keep this run and every earlier artifact; use a separately declared mechanism for any later improvement. The application selection is unchanged and no overall quality winner is declared.

