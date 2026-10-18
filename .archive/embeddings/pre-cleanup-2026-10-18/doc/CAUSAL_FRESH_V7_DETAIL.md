<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v7.market-01 detailed measurements and evidence

**Measured evidence and final independent establishment passed; overall advancement fails4/22.**
The following tables/physical errors/repetition values and22 decisions are
verbatim saved-data readouts from the completed V7 comparison, with standard
quality column labels. No values are recomputed for this human document.
V3/V4/V5/V6 final measurements are reused; V7 initial0/final2048 and each stage's
own fixed heads are new. Trust is supplied by the completed independent audits/evidence join while
frozen producer pending-trust fields remain unchanged. V7 passes4/22 with 100%
coverage; there is no overall winner, application replacement or new quality run.

Read [results](CAUSAL_FRESH_V7_RESULTS.md) for the full populations, native40
composition, fresh ancestry, fixed heads/seeds, dimensions and protocol.
This is one 2048/B8 CUDA encoder trajectory with3 neural head fits per stage,
not3 encoder trainings. Every table below retains its adjacent MKT-HIST-2 or
TEMPO-5 recipe/complexity legend. Error units differ by target; R² is not a
classification accuracy. Physical MAE/RMSE and all three repetitions are shown
for all four clean controls plus V7 initial/final.
Complexity 4/5 and 5/5 are ordinal recipe labels, not accuracy or observed task difficulty.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_volatility; V3/V4/V5/V6 final rows reused, V7 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673434 | 0.463051923 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.555176229 | 100 |
| CPB-v5.market-01 | 40 | 0.95221109 | 0.648190685 | 100 |
| CPB-v6.market-01 | 40 | 0.877120189 | 0.58463377 | 100 |
| CPB-v7.market-01.init | 40 | 0.412200077 | -0.118115229 | 100 |
| CPB-v7.market-01 | 40 | 0.880509577 | 0.673304976 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 5.14382445e-05 | 6.34988601e-05 | 0.000215581406 | 0.000245836333 |
| CPB-v4.market-01 | 0.000103097727 | 0.000131036924 | 0.000175939865 | 0.000220232799 |
| CPB-v5.market-01 | 5.54373122e-05 | 7.38548092e-05 | 0.00016620698 | 0.000198930285 |
| CPB-v6.market-01 | 9.71580786e-05 | 0.000118428297 | 0.00017129703 | 0.000212519589 |
| CPB-v7.market-01.init | 0.000207721498 | 0.00025901802 | 0.000288208788 | 0.000354742789 |
| CPB-v7.market-01 | 9.79104568e-05 | 0.000116783575 | 0.000149546807 | 0.000188788891 |

CPB-v3.market-01 neural R² for2701/2802/2903: 0.344525499, 0.414672854, 0.629957415.
CPB-v4.market-01 neural R² for2701/2802/2903: 0.275507022, 0.68909438, 0.700927285.
CPB-v5.market-01 neural R² for2701/2802/2903: 0.539831415, 0.658428065, 0.746312576.
CPB-v6.market-01 neural R² for2701/2802/2903: 0.634143112, 0.778464525, 0.341293674.
CPB-v7.market-01.init neural R² for2701/2802/2903: 0.127535207, 0.0143159115, -0.496196806.
CPB-v7.market-01 neural R² for2701/2802/2903: 0.701381536, 0.825073085, 0.493460306.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_trend; V3/V4/V5/V6 final rows reused, V7 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263132963 | -0.253979001 | 100 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.569875231 | 100 |
| CPB-v5.market-01 | 40 | 0.868645445 | 0.523928784 | 100 |
| CPB-v6.market-01 | 40 | 0.675860749 | 0.270702751 | 100 |
| CPB-v7.market-01.init | 40 | -1.03185689 | -0.5918479 | 100 |
| CPB-v7.market-01 | 40 | 0.685042973 | 0.10961324 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 4.37234916e-05 | 5.7482267e-05 | 5.85732789e-05 | 7.49730861e-05 |
| CPB-v4.market-01 | 1.59795965e-05 | 2.03977321e-05 | 3.49918123e-05 | 4.32661974e-05 |
| CPB-v5.market-01 | 2.00877722e-05 | 2.42695594e-05 | 3.60577464e-05 | 4.60696486e-05 |
| CPB-v6.market-01 | 3.03383766e-05 | 3.81245859e-05 | 4.64219318e-05 | 5.71545039e-05 |
| CPB-v7.market-01.init | 6.69073927e-05 | 9.54521865e-05 | 6.32738855e-05 | 8.31063899e-05 |
| CPB-v7.market-01 | 3.07774069e-05 | 3.75807094e-05 | 4.92686076e-05 | 6.27498501e-05 |

CPB-v3.market-01 neural R² for2701/2802/2903: -0.319245159, -0.237747895, -0.204943949.
CPB-v4.market-01 neural R² for2701/2802/2903: 0.640552095, 0.358116367, 0.71095723.
CPB-v5.market-01 neural R² for2701/2802/2903: 0.512587238, 0.442624323, 0.61657479.
CPB-v6.market-01 neural R² for2701/2802/2903: 0.204803498, 0.321815119, 0.285489636.
CPB-v7.market-01.init neural R² for2701/2802/2903: -1.42117034, -0.296082931, -0.0582904283.
CPB-v7.market-01 neural R² for2701/2802/2903: 0.239513425, -0.194818323, 0.284144619.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_activity; V3/V4/V5/V6 final rows reused, V7 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840311 | 0.541426056 | 100 |
| CPB-v4.market-01 | 36 | 0.835143075 | 0.687370545 | 100 |
| CPB-v5.market-01 | 40 | 0.844256819 | 0.617983776 | 100 |
| CPB-v6.market-01 | 40 | 0.672769992 | 0.323179998 | 100 |
| CPB-v7.market-01.init | 40 | 0.11098842 | 0.546273981 | 100 |
| CPB-v7.market-01 | 40 | 0.712390048 | 0.352638034 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.178143553 | 0.212835342 | 0.303552885 | 0.387150693 |
| CPB-v4.market-01 | 0.186144676 | 0.233337263 | 0.251684365 | 0.312491365 |
| CPB-v5.market-01 | 0.196732151 | 0.226795807 | 0.283586415 | 0.346423083 |
| CPB-v6.market-01 | 0.274086555 | 0.328743415 | 0.381790574 | 0.471796096 |
| CPB-v7.market-01.init | 0.439655457 | 0.541856403 | 0.308563906 | 0.384794713 |
| CPB-v7.market-01 | 0.259703709 | 0.308199873 | 0.382186321 | 0.461718526 |

CPB-v3.market-01 neural R² for2701/2802/2903: 0.551959766, 0.64999403, 0.42232437.
CPB-v4.market-01 neural R² for2701/2802/2903: 0.868267037, 0.635226168, 0.558618431.
CPB-v5.market-01 neural R² for2701/2802/2903: 0.777561455, 0.374143234, 0.702246639.
CPB-v6.market-01 neural R² for2701/2802/2903: 0.19661957, 0.386097575, 0.386822848.
CPB-v7.market-01.init neural R² for2701/2802/2903: 0.457036142, 0.503704623, 0.678081177.
CPB-v7.market-01 neural R² for2701/2802/2903: 0.253826868, 0.414419927, 0.389667307.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_relation; V3/V4/V5/V6 final rows reused, V7 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.00700444683 | -0.0855656715 | 100 |
| CPB-v4.market-01 | 36 | 0.326405067 | 0.209510464 | 100 |
| CPB-v5.market-01 | 40 | 0.252328556 | 0.370505348 | 100 |
| CPB-v6.market-01 | 40 | 0.622100687 | 0.322683609 | 100 |
| CPB-v7.market-01.init | 40 | 0.0873076083 | -0.253444377 | 100 |
| CPB-v7.market-01 | 40 | 0.600800572 | 0.324861744 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.0723982929 | 0.086491159 | 0.0702199215 | 0.0894458833 |
| CPB-v4.market-01 | 0.0562974528 | 0.0707384319 | 0.0615928121 | 0.0766223466 |
| CPB-v5.market-01 | 0.0586723303 | 0.074526619 | 0.0547690214 | 0.0682288649 |
| CPB-v6.market-01 | 0.0456908024 | 0.0529839185 | 0.055655748 | 0.0702095263 |
| CPB-v7.market-01.init | 0.0683223609 | 0.0823413983 | 0.0774026141 | 0.0961035738 |
| CPB-v7.market-01 | 0.0474929235 | 0.0544566571 | 0.056510372 | 0.0703589445 |

CPB-v3.market-01 neural R² for2701/2802/2903: -0.300654281, 0.161014634, -0.117057368.
CPB-v4.market-01 neural R² for2701/2802/2903: 0.182210538, 0.23983439, 0.206486466.
CPB-v5.market-01 neural R² for2701/2802/2903: 0.317728582, 0.48696529, 0.306822173.
CPB-v6.market-01 neural R² for2701/2802/2903: 0.0570793218, 0.525771962, 0.385199543.
CPB-v7.market-01.init neural R² for2701/2802/2903: 0.0468624742, -0.471401748, -0.335793857.
CPB-v7.market-01 neural R² for2701/2802/2903: 0.115499431, 0.489241068, 0.369844732.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact; V3/V4/V5/V6 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01 | 36 | 98.3072917 | 96.9618056 | 100 |
| CPB-v5.market-01 | 40 | 97.1354167 | 96.5711806 | 100 |
| CPB-v6.market-01 | 40 | 98.6979167 | 98.4809028 | 100 |
| CPB-v7.market-01.init | 40 | 94.2708333 | 93.8368056 | 100 |
| CPB-v7.market-01 | 40 | 98.5677083 | 97.8732639 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted; V3/V4/V5/V6 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.1458333 | 91.7534722 | 100 |
| CPB-v4.market-01 | 36 | 94.6614583 | 93.359375 | 100 |
| CPB-v5.market-01 | 40 | 92.96875 | 92.7951389 | 100 |
| CPB-v6.market-01 | 40 | 95.3125 | 93.4027778 | 100 |
| CPB-v7.market-01.init | 40 | 88.28125 | 85.8072917 | 100 |
| CPB-v7.market-01 | 40 | 93.359375 | 92.2309028 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact; V3/V4/V5/V6 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.3489583 | 98.1770833 | 100 |
| CPB-v4.market-01 | 36 | 99.3489583 | 98.828125 | 100 |
| CPB-v5.market-01 | 40 | 99.21875 | 98.8715278 | 100 |
| CPB-v6.market-01 | 40 | 99.7395833 | 99.21875 | 100 |
| CPB-v7.market-01.init | 40 | 94.4010417 | 93.7934028 | 100 |
| CPB-v7.market-01 | 40 | 99.21875 | 98.4809028 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted; V3/V4/V5/V6 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.2708333 | 100 |
| CPB-v4.market-01 | 36 | 95.8333333 | 95.6163194 | 100 |
| CPB-v5.market-01 | 40 | 94.140625 | 94.0972222 | 100 |
| CPB-v6.market-01 | 40 | 97.265625 | 95.3125 | 100 |
| CPB-v7.market-01.init | 40 | 86.71875 | 84.0711806 | 100 |
| CPB-v7.market-01 | 40 | 94.53125 | 91.8836806 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.25904599 | 0.376414231 | 144.341116 (reused) |
| CPB-v4.market-01 | 2048 | 0.272508032 | 0.308316924 | 172.860055 (reused) |
| CPB-v5.market-01 | 2048 | 0.254463936 | 0.375196294 | 134.983334 (reused) |
| CPB-v6.market-01 | 2048 | 0.250172579 | 0.354519735 | 175.575555 (reused) |
| CPB-v7.market-01.init | 0 | 0.664167857 | 0.958064934 | 0 |
| CPB-v7.market-01 | 2048 | 0.241805338 | 0.370420511 | 219.730088 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same query distributions; NLL omits constant and can be negative.

| Method | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- |
| CPB-v3.market-01 | -0.38017808 | 1.52179658 |
| CPB-v4.market-01 | -0.356692772 | -0.0453814881 |
| CPB-v5.market-01 | -0.324377373 | 2.68745022 |
| CPB-v6.market-01 | -0.400207362 | 0.648448045 |
| CPB-v7.market-01.init | 0.423815353 | 0.795376511 |
| CPB-v7.market-01 | -0.408827278 | 0.868789874 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. All22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Frozen floor/ceiling | Source | Rule | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.880509577 | 0.964673434 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.673304976 | 0.648190685 | CPB-v5.market-01 | candidate + 1e-12 >= floor | PASS |
| MKT-HIST-2 | past_trend:linear_r2 | 0.685042973 | 0.90721343 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:neural_mean_r2 | 0.10961324 | 0.569875231 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:linear_r2 | 0.712390048 | 0.862840311 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.352638034 | 0.687370545 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:linear_r2 | 0.600800572 | 0.622100687 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.324861744 | 0.370505348 | CPB-v5.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.985677083 | 0.986979167 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.978732639 | 0.984809028 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.93359375 | 0.953125 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.922309028 | 0.934027778 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.9921875 | 0.997395833 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.984809028 | 0.9921875 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.9453125 | 0.97265625 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.918836806 | 0.956163194 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | train:standardized_MAE | 0.241805338 | 0.250172579 | CPB-v6.market-01 | candidate <= floor + 1e-12 | PASS |
| MKT-HIST-2 | validation:standardized_MAE | 0.370420511 | 0.308316924 | CPB-v4.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | 0.868789874 | -0.0453814881 | CPB-v4.market-01 | candidate < floor - 1e-12 | FAIL |
| MKT-HIST-2 | past_trend:all3_neural_R2_positive | 0.239513425, -0.194818323, 0.284144619 | 0 | strict positivity | all three final neural trend R2 values > 0 | FAIL |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.600800572 | 0 | strict positivity | candidate > 0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.324861744 | 0 | strict positivity | candidate > 0 | PASS |

The exact declared trajectory is stopped at2048. Retain every failed condition and all four references and their scoped strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue.



## Procedure, causal proof and independent evidence

MKT-HIST-2 training retains one fresh 2048/B8 CUDA:0 trajectory with no skips,
actual typed continuation at512 and five saved artifacts. Fresh constructor
943001 creates all76 learned tensors/294752 values without old model or fitted
statistic import. The fixed prior has3888 values. Fresh scaler/bank calibration
uses4096 TRAIN-only anchors and no VALIDATION rows or encoder forwards; query
masks/readout recipes remain fixed. Native40 and all76 learned tensor categories,
including uncertainty projection, receive actual CUDA gradients and movement.

The unchanged F64 objective is R[Huber_delta1(mu-y)] plus
R[0.5*(stopgrad(mu-y)^2/v+log(v))], coefficients1/1, with original hierarchical
eligible cell/channel/example reduction and variance link. Reporting Gaussian
NLL uses the unchanged numeric kernel and never backpropagates. Variance and
total objective can be negative. V7's change is disjoint norm2 max1 clipping
(mean73/293537, uncertainty3/1215; fixed1/3888 excluded), followed by one AdamW.
The combined norm may reachsqrt2; optimizer/future residual coupling remains.

Engineering213824 checks/7372 CUDA calls and110 negative typed-load/resume cases
pass. The separate engineering-only nonzero fixture reports359 CUDA checks,
10 decoder calls, manual composite native40 gradient parity, detached-error
variance mean-gradient zero, live mean/uncertainty gradients, clipping isolation,
one AdamW step and5 fixture parameter tensors moved; encoder forwards0, old
artifacts0, quality fits0. Its synthetic tensors are not quality measurements.
Initial/final saved artifacts each pass128 cutoffs,2350 future value/mask cases
and30 internal temporal interventions, unchanged state/source0 and actual
causal counters0/2048, with no further updates. Concurrent proof wall times
241.13/240.44s are not encoder training time or a speed benchmark.

Training audit:283252 total =33566 core +61153 statistics +1346 saved state
+117864 saved Gaussian +69323 actual pre-step composite checks; zero mismatches.
It retains all2048 records, fresh TRAIN statistics, typed moments/counters,
actual resume and update1/2048 component witnesses. The last actual batch is a
pre-step state2047 forward; it is not paired with post-step2048 decoder weights
for a fabricated local gradient. Recorded branch factors/coverage are audited;
full gradient norms are not independently reconstructed. Actual clipping is
mean0/2048 and uncertainty1313/2048, not a V6 counterfactual.

Quality audit:12493 checks, zero mismatches, all 13 full40 initial DATA/statistic
pairings versus V6 exact, all actual22 decisions and100% coverage verified.
Eight initial feature archives and five fresh-statistic archives match
numerically; metadata identities/file bytes may differ. V7 still fits its own
initial heads. Original producer pending-trust fields stay unchanged, and a
separate root-owned receipt now binds final independent evidence without changing producer bytes.

Evidence pins:

- [Prospective machine card](CAUSAL_FRESH_V7_CARD.json): f913044fd4bdbeff637991087f91d0b47bca451b31dfe664fb70b9555bbae593.
- [Source manifest](results/causal_fresh_v7/source-manifest.json):09c151d24eb91ea4ce8e653601aeee9d5e42d098df2ecd7204af92e528d67f80;106 frozen files, six admitted binaries, full ZIPac43ef587383cbe3d610a5f73736a4d10cc852a64ebca249602f37db4b716999.
- [Training audit](causal_fresh_v7/independent-training-audit.json):a36254f4392c4411793ef0b9fe9db858c7e16f3238c75ce4de12aa8b9869c6d4.
- [Quality audit](causal_fresh_v7/quality-independent-audit.json):7e70495f509000063ae32f145c705fe0662fb8121b06a2e5cc3d4ec519926404.
- [Final establishment](causal_fresh_v7/final-establishment.json):7678e84a676453c0294f0654af6117e06246f1639f4e67f9ff78e9cf51049512; full trust, measured candidate,13 pairings and coverage pass;4/22 fails advancement, winner/winner-selection remain false.
- [Complete mirror](causal_fresh_v7/complete-mirror-attempt-001.json):67e720225b5f45e16039c068f7f1516408766b33c46adf49e7f87adc5e4415d2;431 files/290853182 bytes,340 PT archives local and ignored. PT count includes features, heads and engineering fixtures, not340 trained encoders.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, retained past-window signals · **complexity 4/5** · one fresh TRAIN2048/B8 trajectory, saved checkpoints; no scoring is performed by hashing them.

| Checkpoint | Bytes | SHA256 |
| --- | --- | --- |
| inference-0.pt | 1312612 | 66b7d4bb29f9b18154f735ebd98c28c3ef69c10d41e8251a1adf38bc0a0c3727 |
| training-512.pt | 3800143 | 7eb641d4067e8966d369e409855819a85e1d8eb03b6c3181a1526ae7cc8df326 |
| training-1024.pt | 3800853 | 0ee7e92829713e5a217b030094b0de9d65f7db349b6a611e1be702772e52a155 |
| training-2048.pt | 3800853 | b74afefcde72a181a5a195901fdf07eaffebd848bfc36f9fc53de0d9983aa12f |
| inference-2048.pt | 1313777 | 25f082bb22076aea26739c8ec9157d385c45e4413b42668d51d14ef40ce1f6a6 |

Main GPU compute time219.730088148s is synchronized cumulative encoder work;
trajectory wall272.302167537s and wrapper281.478890819s include other work/I/O.
The quality wrapper is about 79.922443504s, separate from training. Reused V3/V4/V5/V6 costs retain their
original scopes and are descriptive, not a matched repeated runtime benchmark.
All old checkpoints/results/source/failures remain preserved. The original
V7 main source, raw reports and independent audit bytes are unchanged by human
reporting or metadata-only finalization. Finalization source's pre-execution
winner-selection clarification preserved its previous five files; no encoder,
training or head run was repeated for that metadata clarification.
