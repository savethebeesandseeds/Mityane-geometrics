<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v4.market-01 results: 19/22, preserve CPB-v3

**CPB-v3.market-01 remains the clean causal baseline and sole dev encoder.**
CPB-v4.market-01 is a newly measured, clean causal research reference: stronger
trend/relation, all eight hard-timing retention conditions passed, but lower
linear volatility/activity and worse TRAIN reconstruction MAE. It fails the full
advancement rule. Preserve both versions and all checkpoints; no promotion or
later encoder run follows this experiment.

This is one newly trained encoder on MKT-HIST-2, 4096 TRAIN anchors, 2048 CUDA:0
updates at batch8, fresh seed943001 weights, fresh TRAIN-only input/bank statistics
and a fresh optimizer. Source updates and inherited learned values are zero.
Native36 consists of learned20, literal fixed spectral12 and learned change4.
The complete owned implementation is [versions/v4](../code/encoders/causal_patch_bottleneck_mae/versions/v4/),
with its [per-design declaration](../code/encoders/causal_patch_bottleneck_mae/versions/CPB-v4.json)
and [measured instance](../code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v4.market-01.json).

The market probes use112 TRAIN/32 known chronological VALIDATION anchors. Ridge1
and a joint D36→16tanh→4 neural head (Adam0.01,100 updates, three fixed repetitions)
have148/660 parameters. TEMPO-5 has three saved cohorts941101/941102/941103, each
512 TRAIN/256 VALIDATION examples from256/128 independent paired source groups;
its binary heads have74/626 parameters. Each policy scores intact and extra30%
coordinate deletion. The synthetic-TRAIN and market-TRAIN scaler policies are
separate panels; zero synthetic encoder updates occurred. Head variation is not
independent encoder replication. These known-development tasks measure retained
past information, not forecasting or new-period confirmation.

Both saved v4 stages were extracted once and received newly fitted unchanged
heads. V3 scores are reused without model execution or refitting. No PCA follows
the encoder, and no checkpoint, seed, head or budget was tuned after quality.
All13 paired initial feature/statistic archives match v3 numerically bit for bit;
the bank identity differs only through the versioned source-provenance tag.

The [prospective card](CAUSAL_FRESH_V4_CARD.md) remains unchanged. The dedicated
variance test passed109 checks, proving that conditional forward capacity and
head learning remain while variance-only native gradients stop. Full engineering
passed7372 CUDA encoder calls; both actual saved stages separately passed all128
cutoffs,2350 future-value/mask interventions and30 internal temporal cases with
frozen state. Actual learning moved all75 learned tensors; typed resume continued
at512 with no repeated training updates.

The [independent training audit](causal_fresh_v4/independent-training-audit.json)
passed192545 checks, and the [independent quality audit](causal_fresh_v4/quality-independent-audit.json)
passed10927 checks, both with zero mismatches and no model/head replay. Original
reporting failed on the reused raw-control kind; an additive, separately declared
[report-only correction](causal_fresh_v4/reporting-attempt002-card.json) fixed that
metadata predicate and added an exact raw-file hash check. All original failures
and audit-only corrections remain preserved. No extraction, fits or training were
repeated. The [detailed report](CAUSAL_FRESH_V4_DETAIL.md) retains raw controls,
per-cohort results, all head repetitions, physical errors, compute and provenance.

The [comparison JSON](results/causal_fresh_v4/comparison.json) records each frozen
condition at full precision. Training times below are descriptive synchronized
compute measurements, not a repeated speed benchmark. Final preservation and
artifact inventory are in the [preservation receipt](causal_fresh_v4/final-preservation.json).


The candidate passes **19/22** prospective numeric conditions. Trust admission: passed; full coverage: true.

CPB-v3 remains the selected clean baseline. These are known chronological development measurements; no overall winner, automatic application promotion or forecasting result is declared. V3 measurements are reused unchanged; both v4 initial and final embeddings have new unchanged-recipe heads. No pretrained v1/v2/RPB score is a floor.

Only the variance input gradient is stopped at change4. The conditional variance head still learns and the forward Gaussian formula is unchanged. Encoder updates use the mean-path partial gradient, not the full joint-NLL gradient. One fresh 2048/B8 CUDA trajectory uses the same data, TRAIN-only calibration recipe, initialization/data/mask seeds and native36 fixed-head recipes. No head tuning, additional encoder training or v3 replay is performed by this comparison.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_volatility; v3 reused, v4 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.96467343381106097 | 0.46305192272156109 | 100 |
| CPB-v4.market-01.init | 36 | 0.45161550426506369 | 0.27158567842694409 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221031565105 | 0.55517622911042031 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 5.1438244510906345e-05 | 6.3498860082341889e-05 | 0.00021558140567355788 | 0.00024583633271647962 |
| CPB-v4.market-01.init | 0.00020329626814686734 | 0.00025018300246663807 | 0.0002323444353786061 | 0.00028403651233036893 |
| CPB-v4.market-01 | 0.00010309772654730662 | 0.00013103692356775142 | 0.00017593986478183261 | 0.00022023279862042359 |

CPB-v3.market-01 neural R² repetitions 2701/2802/2903: 0.34452549910387642, 0.41467285432586121, 0.62995741473494571.
CPB-v4.market-01.init neural R² repetitions 2701/2802/2903: 0.10569529906697439, 0.5993227865507641, 0.10973894966309383.
CPB-v4.market-01 neural R² repetitions 2701/2802/2903: 0.27550702165012397, 0.68909438026645409, 0.70092728541468285.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_trend; v3 reused, v4 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.2631329634696965 | -0.25397900114104788 | 100 |
| CPB-v4.market-01.init | 36 | -0.79042257691793782 | -1.2910130549534284 | 100 |
| CPB-v4.market-01 | 36 | 0.90721342970362451 | 0.56987523070383139 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 4.3723491614761766e-05 | 5.7482266990266961e-05 | 5.8573278918616742e-05 | 7.4973086124059424e-05 |
| CPB-v4.market-01.init | 6.4356509338262314e-05 | 8.9601874607961695e-05 | 7.4851659770490132e-05 | 0.00010004134608569896 |
| CPB-v4.market-01 | 1.5979596504966024e-05 | 2.0397732125200584e-05 | 3.499181232931064e-05 | 4.3266197431183492e-05 |

CPB-v3.market-01 neural R² repetitions 2701/2802/2903: -0.31924515917541196, -0.23774789512546346, -0.20494394912226821.
CPB-v4.market-01.init neural R² repetitions 2701/2802/2903: -1.4400303669384704, -0.36449282120763549, -2.0685159767141794.
CPB-v4.market-01 neural R² repetitions 2701/2802/2903: 0.64055209499749255, 0.35811636672973135, 0.71095723038427039.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_activity; v3 reused, v4 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.86284031106087822 | 0.541426055582653 | 100 |
| CPB-v4.market-01.init | 36 | 0.16845197492995934 | 0.60751248467142127 | 100 |
| CPB-v4.market-01 | 36 | 0.83514307454564629 | 0.68737054501094963 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.17814355298975235 | 0.21283534231476031 | 0.30355288545421949 | 0.38715069272080227 |
| CPB-v4.market-01.init | 0.4231467113802751 | 0.52405174171614266 | 0.28884576972561293 | 0.35656206046360589 |
| CPB-v4.market-01 | 0.18614467583169114 | 0.23333726255631509 | 0.25168436472763089 | 0.31249136498691593 |

CPB-v3.market-01 neural R² repetitions 2701/2802/2903: 0.55195976639400202, 0.64999403029895886, 0.42232437005499801.
CPB-v4.market-01.init neural R² repetitions 2701/2802/2903: 0.68230477909300413, 0.6924019526619386, 0.44783072225932108.
CPB-v4.market-01 neural R² repetitions 2701/2802/2903: 0.86826703651099679, 0.63522616762538675, 0.55861843089646523.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_relation; v3 reused, v4 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.0070044468343222199 | -0.08556567147059875 | 100 |
| CPB-v4.market-01.init | 36 | 0.083437182541613142 | -0.10617896981457819 | 100 |
| CPB-v4.market-01 | 36 | 0.32640506704184069 | 0.20951046448819055 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.072398292894631594 | 0.086491159022817898 | 0.070219921481160796 | 0.089445883268242155 |
| CPB-v4.market-01.init | 0.069197289610486656 | 0.082515804883195681 | 0.073010465514330283 | 0.090273208956608078 |
| CPB-v4.market-01 | 0.056297452839669512 | 0.070738431929249612 | 0.061592812121547401 | 0.076622346597282495 |

CPB-v3.market-01 neural R² repetitions 2701/2802/2903: -0.30065428105903469, 0.16101463449470332, -0.1170573678474649.
CPB-v4.market-01.init neural R² repetitions 2701/2802/2903: -0.11773962267266325, -0.3444212979580612, 0.14362401118698986.
CPB-v4.market-01 neural R² repetitions 2701/2802/2903: 0.18221053804312992, 0.23983438970069992, 0.20648646572074181.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact; v3 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01.init | 36 | 94.010416666666657 | 92.7517361111111 | 100 |
| CPB-v4.market-01 | 36 | 98.307291666666657 | 96.961805555555557 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted; v3 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.145833333333343 | 91.753472222222214 | 100 |
| CPB-v4.market-01.init | 36 | 88.541666666666657 | 85.894097222222214 | 100 |
| CPB-v4.market-01 | 36 | 94.661458333333343 | 93.359375 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact; v3 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.348958333333343 | 98.177083333333343 | 100 |
| CPB-v4.market-01.init | 36 | 94.791666666666657 | 93.576388888888886 | 100 |
| CPB-v4.market-01 | 36 | 99.348958333333343 | 98.828125 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted; v3 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.270833333333343 | 100 |
| CPB-v4.market-01.init | 36 | 88.151041666666657 | 83.940972222222214 | 100 |
| CPB-v4.market-01 | 36 | 95.833333333333343 | 95.616319444444443 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Immutable fixed-query standardized MAE. V3 cost is its reused historical 144.341116249s measurement, not a new run.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.25904599043427573 | 0.37641423057615903 | 144.34111624900029 |
| CPB-v4.market-01.init | 0 | 0.66416785662510747 | 0.95806493432361817 | 0 |
| CPB-v4.market-01 | 2048 | 0.27250803234958165 | 0.30831692401624544 | 172.86005468699975 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same fixed queries; Gaussian NLL is separate and may be negative.

| Encoder | Updates | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | -0.38017807983375768 | 1.521796577918092 |
| CPB-v4.market-01.init | 0 | 0.4238153534628793 | 0.79537651115920871 |
| CPB-v4.market-01 | 2048 | -0.35669277205499406 | -0.045381488090816241 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. All 22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Baseline/floor | Rule | Result |
| --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.84956221031565105 | 0.96467343381106097 | candidate+1e-12>=baseline | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.55517622911042031 | 0.46305192272156109 | candidate+1e-12>=baseline | PASS |
| MKT-HIST-2 | past_trend:linear_r2 | 0.90721342970362451 | 0.2631329634696965 | candidate+1e-12>=baseline | PASS |
| MKT-HIST-2 | past_trend:neural_mean_r2 | 0.56987523070383139 | -0.25397900114104788 | candidate+1e-12>=baseline | PASS |
| MKT-HIST-2 | past_activity:linear_r2 | 0.83514307454564629 | 0.86284031106087822 | candidate+1e-12>=baseline | FAIL |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.68737054501094963 | 0.541426055582653 | candidate+1e-12>=baseline | PASS |
| MKT-HIST-2 | past_relation:linear_r2 | 0.32640506704184069 | -0.0070044468343222199 | candidate+1e-12>=baseline | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.20951046448819055 | -0.08556567147059875 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.98307291666666663 | 0.96484375 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.96961805555555558 | 0.95703125 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.94661458333333337 | 0.91145833333333337 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.93359375 | 0.91753472222222221 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.99348958333333337 | 0.99348958333333337 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.98828125 | 0.98177083333333337 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.95833333333333337 | 0.94921875 | candidate+1e-12>=baseline | PASS |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.95616319444444442 | 0.94270833333333337 | candidate+1e-12>=baseline | PASS |
| MKT-HIST-2 | train:standardized_MAE | 0.27250803234958165 | 0.25904599043427573 | candidate<=baseline+1e-12 | FAIL |
| MKT-HIST-2 | validation:standardized_MAE | 0.30831692401624544 | 0.37641423057615903 | candidate<=baseline+1e-12 | PASS |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | -0.045381488090816241 | 1.521796577918092 | candidate<baseline-1e-12 | PASS |
| MKT-HIST-2 | past_trend:all3_neural_R2_positive | 0.64055209499749255, 0.35811636672973135, 0.71095723038427039 | 0 | All3 final neural trend R2 strictly>0 | PASS |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.32640506704184069 | 0 | candidate>0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.20951046448819055 | 0 | candidate>0 | PASS |

Scoped gains and regressions are preserved at full precision in [comparison.json](results/causal_fresh_v4/comparison.json). Initial-to-final changes do not establish causation, and retention gates do not create an overall accuracy score. Any further experiment needs a separate declaration; this report authorizes no rescue or promotion.

