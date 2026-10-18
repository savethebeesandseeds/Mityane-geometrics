<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v5 complete measured detail

This document formats completed saved evidence; no new experiment or scoring run is performed to write it. Both clean reference rows are reused, and V5 initial/final rows are new. Independent quality audit passes 11300 checks with zero mismatches; TRAIN audit 192589, engineering 213674/38 refusals and the dedicated gradient fixture 31 checks pass. Both actual saved stages pass 128/2350/30 causal matrices. Read [the summary](CAUSAL_FRESH_V5_RESULTS.md) and [review](CAUSAL_FRESH_V5_REVIEW.md).

## Compared with both preserved clean references

Candidate passes **7/22** exact frozen goal conditions. Trust admission: passed; full coverage: true.

CPB-v3 remains selected; CPB-v4 scoped gains remain protected. V3/V4 rows and training costs are reused unchanged. V5 initial 0/final 2048 and their fixed heads are new. No overall winner, automatic promotion, forecasting or fresh-period confirmation follows.

New native40 has 1080 extra learned projection values; D40 market heads 164/724 versus D36 148/660; TEMPO 82/690 versus 74/626; dimension-derived seeds differ. No same-capacity causal attribution.

Independent bank projections feed mean4 and uncertainty4; true Gaussian NLL trains disjoint mean/variance paths. The learned variance still reweights mean errors. All learned values/statistics start fresh; no old model execution or head tuning.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_volatility; V3/V4 reused, V5 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673434 | 0.463051923 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.555176229 | 100 |
| CPB-v5.market-01.init | 40 | 0.412200077 | -0.118115229 | 100 |
| CPB-v5.market-01 | 40 | 0.95221109 | 0.648190685 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 5.14382445e-05 | 6.34988601e-05 | 0.000215581406 | 0.000245836333 |
| CPB-v4.market-01 | 0.000103097727 | 0.000131036924 | 0.000175939865 | 0.000220232799 |
| CPB-v5.market-01.init | 0.000207721498 | 0.00025901802 | 0.000288208788 | 0.000354742789 |
| CPB-v5.market-01 | 5.54373122e-05 | 7.38548092e-05 | 0.00016620698 | 0.000198930285 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: 0.344525499, 0.414672854, 0.629957415.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.275507022, 0.68909438, 0.700927285.
CPB-v5.market-01.init neural R² for repetitions 2701/2802/2903: 0.127535207, 0.0143159115, -0.496196806.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.539831415, 0.658428065, 0.746312576.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_trend; V3/V4 reused, V5 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263132963 | -0.253979001 | 100 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.569875231 | 100 |
| CPB-v5.market-01.init | 40 | -1.03185689 | -0.5918479 | 100 |
| CPB-v5.market-01 | 40 | 0.868645445 | 0.523928784 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 4.37234916e-05 | 5.7482267e-05 | 5.85732789e-05 | 7.49730861e-05 |
| CPB-v4.market-01 | 1.59795965e-05 | 2.03977321e-05 | 3.49918123e-05 | 4.32661974e-05 |
| CPB-v5.market-01.init | 6.69073927e-05 | 9.54521865e-05 | 6.32738855e-05 | 8.31063899e-05 |
| CPB-v5.market-01 | 2.00877722e-05 | 2.42695594e-05 | 3.60577464e-05 | 4.60696486e-05 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: -0.319245159, -0.237747895, -0.204943949.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.640552095, 0.358116367, 0.71095723.
CPB-v5.market-01.init neural R² for repetitions 2701/2802/2903: -1.42117034, -0.296082931, -0.0582904283.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.512587238, 0.442624323, 0.61657479.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_activity; V3/V4 reused, V5 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840311 | 0.541426056 | 100 |
| CPB-v4.market-01 | 36 | 0.835143075 | 0.687370545 | 100 |
| CPB-v5.market-01.init | 40 | 0.11098842 | 0.546273981 | 100 |
| CPB-v5.market-01 | 40 | 0.844256819 | 0.617983776 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.178143553 | 0.212835342 | 0.303552885 | 0.387150693 |
| CPB-v4.market-01 | 0.186144676 | 0.233337263 | 0.251684365 | 0.312491365 |
| CPB-v5.market-01.init | 0.439655457 | 0.541856403 | 0.308563906 | 0.384794713 |
| CPB-v5.market-01 | 0.196732151 | 0.226795807 | 0.283586415 | 0.346423083 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: 0.551959766, 0.64999403, 0.42232437.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.868267037, 0.635226168, 0.558618431.
CPB-v5.market-01.init neural R² for repetitions 2701/2802/2903: 0.457036142, 0.503704623, 0.678081177.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.777561455, 0.374143234, 0.702246639.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_relation; V3/V4 reused, V5 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.00700444683 | -0.0855656715 | 100 |
| CPB-v4.market-01 | 36 | 0.326405067 | 0.209510464 | 100 |
| CPB-v5.market-01.init | 40 | 0.0873076083 | -0.253444377 | 100 |
| CPB-v5.market-01 | 40 | 0.252328556 | 0.370505348 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.0723982929 | 0.086491159 | 0.0702199215 | 0.0894458833 |
| CPB-v4.market-01 | 0.0562974528 | 0.0707384319 | 0.0615928121 | 0.0766223466 |
| CPB-v5.market-01.init | 0.0683223609 | 0.0823413983 | 0.0774026141 | 0.0961035738 |
| CPB-v5.market-01 | 0.0586723303 | 0.074526619 | 0.0547690214 | 0.0682288649 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: -0.300654281, 0.161014634, -0.117057368.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.182210538, 0.23983439, 0.206486466.
CPB-v5.market-01.init neural R² for repetitions 2701/2802/2903: 0.0468624742, -0.471401748, -0.335793857.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.317728582, 0.48696529, 0.306822173.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_intact; V3/V4 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01 | 36 | 98.3072917 | 96.9618056 | 100 |
| CPB-v5.market-01.init | 40 | 94.2708333 | 93.8368056 | 100 |
| CPB-v5.market-01 | 40 | 97.1354167 | 96.5711806 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_deleted; V3/V4 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.1458333 | 91.7534722 | 100 |
| CPB-v4.market-01 | 36 | 94.6614583 | 93.359375 | 100 |
| CPB-v5.market-01.init | 40 | 88.28125 | 85.8072917 | 100 |
| CPB-v5.market-01 | 40 | 92.96875 | 92.7951389 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_intact; V3/V4 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.3489583 | 98.1770833 | 100 |
| CPB-v4.market-01 | 36 | 99.3489583 | 98.828125 | 100 |
| CPB-v5.market-01.init | 40 | 94.4010417 | 93.7934028 | 100 |
| CPB-v5.market-01 | 40 | 99.21875 | 98.8715278 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_deleted; V3/V4 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.2708333 | 100 |
| CPB-v4.market-01 | 36 | 95.8333333 | 95.6163194 | 100 |
| CPB-v5.market-01.init | 40 | 86.71875 | 84.0711806 | 100 |
| CPB-v5.market-01 | 40 | 94.140625 | 94.0972222 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.

| Method | Updates | TRAIN MAE | VALIDATION MAE | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.25904599 | 0.376414231 | 144.341116 (reused) |
| CPB-v4.market-01 | 2048 | 0.272508032 | 0.308316924 | 172.860055 (reused) |
| CPB-v5.market-01.init | 0 | 0.664167857 | 0.958064934 | 0 |
| CPB-v5.market-01 | 2048 | 0.254463936 | 0.375196294 | 134.983334 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Same query distributions; NLL omits constant and can be negative.

| Method | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- |
| CPB-v3.market-01 | -0.38017808 | 1.52179658 |
| CPB-v4.market-01 | -0.356692772 | -0.0453814881 |
| CPB-v5.market-01.init | 0.423815353 | 0.795376511 |
| CPB-v5.market-01 | -0.324377373 | 2.68745022 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. All 22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Frozen floor/ceiling | Source | Rule | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.95221109 | 0.964673434 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.648190685 | 0.555176229 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| MKT-HIST-2 | past_trend:linear_r2 | 0.868645445 | 0.90721343 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:neural_mean_r2 | 0.523928784 | 0.569875231 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:linear_r2 | 0.844256819 | 0.862840311 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.617983776 | 0.687370545 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:linear_r2 | 0.252328556 | 0.326405067 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.370505348 | 0.209510464 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.971354167 | 0.983072917 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.965711806 | 0.969618056 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.9296875 | 0.946614583 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.927951389 | 0.93359375 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.9921875 | 0.993489583 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.988715278 | 0.98828125 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.94140625 | 0.958333333 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.940972222 | 0.956163194 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | train:standardized_MAE | 0.254463936 | 0.25904599 | CPB-v3.market-01 | candidate <= floor + 1e-12 | PASS |
| MKT-HIST-2 | validation:standardized_MAE | 0.375196294 | 0.308316924 | CPB-v4.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | 2.68745022 | -0.0453814881 | CPB-v4.market-01 | candidate < floor - 1e-12 | FAIL |
| MKT-HIST-2 | past_trend:all3_neural_R2_positive | 0.512587238, 0.442624323, 0.61657479 | 0 | strict positivity | all three final neural trend R2 values > 0 | PASS |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.252328556 | 0 | strict positivity | candidate > 0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.370505348 | 0 | strict positivity | candidate > 0 | PASS |

The exact declared trajectory is stopped at 2048. Retain every failed condition and both reference strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue.


## Every saved market neural repetition

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. past_volatility; physical MAE/RMSE in logreturn.

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | 0.34452549910387642 | 0.00023591343985291372 | 0.00027352253772491852 |
| CPB-v3.market-01 | 2802 | 0.41467285432586121 | 0.00023639862741390167 | 0.00025847262252737751 |
| CPB-v3.market-01 | 2903 | 0.62995741473494571 | 0.00017443214975385827 | 0.00020551383789714291 |
| CPB-v4.market-01 | 2701 | 0.27550702165012397 | 0.00021858422420986478 | 0.00028756253717808527 |
| CPB-v4.market-01 | 2802 | 0.68909438026645409 | 0.00015630685364439475 | 0.00018837770537763807 |
| CPB-v4.market-01 | 2903 | 0.70092728541468285 | 0.00015292851649123827 | 0.00018475815330554746 |
| CPB-v5.market-01.init | 2701 | 0.12753520698223764 | 0.00025782611004414877 | 0.00031556524928591116 |
| CPB-v5.market-01.init | 2802 | 0.014315911487901012 | 0.00026318239141584422 | 0.00033541624244850225 |
| CPB-v5.market-01.init | 2903 | -0.49619680623717621 | 0.00034361786124379129 | 0.0004132468749866081 |
| CPB-v5.market-01 | 2701 | 0.5398314153274193 | 0.00019641478626913396 | 0.0002291784087142298 |
| CPB-v5.market-01 | 2802 | 0.65842806455623526 | 0.00015750320745241375 | 0.00019744962201088365 |
| CPB-v5.market-01 | 2903 | 0.74631257625349345 | 0.00014470294524033649 | 0.00017016282439398178 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. past_trend; physical MAE/RMSE in logreturn/tick.

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | -0.31924515917541196 | 6.3963534136833448E-05 | 7.6913411986060163E-05 |
| CPB-v3.market-01 | 2802 | -0.23774789512546346 | 5.5858253568333586E-05 | 7.4499853938872243E-05 |
| CPB-v3.market-01 | 2903 | -0.20494394912226821 | 5.5898049050683185E-05 | 7.3505992447245853E-05 |
| CPB-v4.market-01 | 2701 | 0.64055209499749255 | 3.2547644876162492E-05 | 4.0147388573822634E-05 |
| CPB-v4.market-01 | 2802 | 0.35811636672973135 | 4.5291096313565925E-05 | 5.3649721785431941E-05 |
| CPB-v4.market-01 | 2903 | 0.71095723038427039 | 2.7136695798203504E-05 | 3.6001481934295893E-05 |
| CPB-v5.market-01.init | 2701 | -1.4211703413214605 | 7.8507654504916639E-05 | 0.00010419622759658911 |
| CPB-v5.market-01.init | 2802 | -0.29608293140274866 | 5.8996207726486635E-05 | 7.6235230667838066E-05 |
| CPB-v5.market-01.init | 2903 | -0.058290428341196243 | 5.2317794235575962E-05 | 6.8887711459334242E-05 |
| CPB-v5.market-01 | 2701 | 0.51258723763894287 | 3.8131670295723817E-05 | 4.6750662684554344E-05 |
| CPB-v5.market-01 | 2802 | 0.44262432315537048 | 3.8079898067553624E-05 | 4.9993475486283194E-05 |
| CPB-v5.market-01 | 2903 | 0.61657479041229912 | 3.1961670878049018E-05 | 4.1464807630593263E-05 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. past_activity; physical MAE/RMSE in log1p(base_volume).

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | 0.55195976639400202 | 0.29089899562705657 | 0.38467061163255906 |
| CPB-v3.market-01 | 2802 | 0.64999403029895886 | 0.27367612028609056 | 0.33999163060351023 |
| CPB-v3.market-01 | 2903 | 0.42232437005499801 | 0.3460835404495114 | 0.43678983592633752 |
| CPB-v4.market-01 | 2701 | 0.86826703651099679 | 0.17215990526731362 | 0.20858243419875477 |
| CPB-v4.market-01 | 2802 | 0.63522616762538675 | 0.28349025597915939 | 0.34709018947973269 |
| CPB-v4.market-01 | 2903 | 0.55861843089646523 | 0.29940293293641962 | 0.38180147128226033 |
| CPB-v5.market-01.init | 2701 | 0.45703614217843513 | 0.31372706235863207 | 0.42346347927293043 |
| CPB-v5.market-01.init | 2802 | 0.50370462337023281 | 0.34518020453247861 | 0.40485603288746769 |
| CPB-v5.market-01.init | 2903 | 0.67808117711672522 | 0.26678445196170103 | 0.32606462619408588 |
| CPB-v5.market-01 | 2701 | 0.77756145523406128 | 0.22372946167609226 | 0.27104133209336251 |
| CPB-v5.market-01 | 2802 | 0.37414323416363171 | 0.38269948825671996 | 0.45464034884933957 |
| CPB-v5.market-01 | 2903 | 0.70224663889313699 | 0.24433029644033016 | 0.31358756785903819 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. past_relation; physical MAE/RMSE in correlation.

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | -0.30065428105903469 | 0.074939390348140164 | 0.098296252324748715 |
| CPB-v3.market-01 | 2802 | 0.16101463449470332 | 0.062049591126866768 | 0.078946560826134118 |
| CPB-v3.market-01 | 2903 | -0.1170573678474649 | 0.073670782968475457 | 0.09109483665384363 |
| CPB-v4.market-01 | 2701 | 0.18221053804312992 | 0.059011701866965194 | 0.077942938990507568 |
| CPB-v4.market-01 | 2802 | 0.23983438970069992 | 0.064438729719716106 | 0.075146737755645995 |
| CPB-v4.market-01 | 2903 | 0.20648646572074181 | 0.061328004777960909 | 0.076777363045693922 |
| CPB-v5.market-01.init | 2701 | 0.046862474170524249 | 0.067086994358783084 | 0.084146064120077968 |
| CPB-v5.market-01.init | 2802 | -0.47140174835152826 | 0.086456986254590695 | 0.10454942691653586 |
| CPB-v5.market-01.init | 2903 | -0.33579385652958704 | 0.078663861678739802 | 0.099615230360170642 |
| CPB-v5.market-01 | 2701 | 0.31772858216902866 | 0.060875889075655422 | 0.071192560285472814 |
| CPB-v5.market-01 | 2802 | 0.48696529005645239 | 0.047522223128582251 | 0.061734707830253283 |
| CPB-v5.market-01 | 2903 | 0.30682217256380928 | 0.055908951980067567 | 0.071759326505342599 |

The immutable [machine table](results/causal_fresh_v5/table_rows.json) and [comparison](results/causal_fresh_v5/comparison.json) retain full-precision physical errors, all source references and every numeric condition. [Quality audit](causal_fresh_v5/quality-independent-audit.json): 203294 bytes, SHA256 `f6a11ba91f750ab9d39760e713a3759cdd6b7223bcaa42d8301205464f7e8dc5`; [TRAIN audit](causal_fresh_v5/independent-training-audit.json): 33798 bytes, SHA256 `265846da7bc58ba650e21e0b19f89bb49922e0cbad7dfe04320d352c1d3b4fa1`. No winner, automatic dev promotion, forecast, rescue or new-period confirmation is declared.

