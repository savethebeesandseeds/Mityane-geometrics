<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v6 complete measured detail

This formats saved completed quality; it executes no model, fit or scoring run.
V3/V4/V5 final rows/costs are reused; V6 initial 0/final 2048 and their own fixed heads
are new. Numeric conditions 12/22, full coverage 100%; independent training/quality audits
pass 270721/12218 with zero mismatches. The separate final establishment receipt confirms full trust and measured-candidate
status while numeric 12/22 fails overall advancement. The original failed
audit and all frozen model/training/head/report bytes remain preserved. No overall
winner or application promotion follows. See [results](CAUSAL_FRESH_V6_RESULTS.md)
and [review](CAUSAL_FRESH_V6_REVIEW.md).

V6 matches V5 native40, 294752 learned values, fresh constructor draws, TRAIN-only
calibration, 2048/B8 CUDA budget and D40 readouts/seeds. Its sole mechanism change
is fixed Huber(delta=1) mean plus detached-error Gaussian variance calibration,
unit weights, live uncertainty4 and one shared clip 1/AdamW. Reporting Gaussian
MAE/NLL remain unchanged and are never the backward tensor. Direct separated
routes do not establish optimizer independence; future residuals still couple
training. V3/V4 native36 capacity/head-seed confounds remain explicit.

The prospective 47fa… protocol protects all prior final V3/V4/V5 scopes, with
unchanged strict best V4 validation NLL/positivity and no TRAIN NLL gate.
Historic pass counts used their own old thresholds. Full native40 initial numeric pairing
and independent audits are mandatory separate trust, not numeric conditions.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_volatility; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673434 | 0.463051923 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.555176229 | 100 |
| CPB-v5.market-01 | 40 | 0.95221109 | 0.648190685 | 100 |
| CPB-v6.market-01.init | 40 | 0.412200077 | -0.118115229 | 100 |
| CPB-v6.market-01 | 40 | 0.877120189 | 0.58463377 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 5.14382445e-05 | 6.34988601e-05 | 0.000215581406 | 0.000245836333 |
| CPB-v4.market-01 | 0.000103097727 | 0.000131036924 | 0.000175939865 | 0.000220232799 |
| CPB-v5.market-01 | 5.54373122e-05 | 7.38548092e-05 | 0.00016620698 | 0.000198930285 |
| CPB-v6.market-01.init | 0.000207721498 | 0.00025901802 | 0.000288208788 | 0.000354742789 |
| CPB-v6.market-01 | 9.71580786e-05 | 0.000118428297 | 0.00017129703 | 0.000212519589 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: 0.344525499, 0.414672854, 0.629957415.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.275507022, 0.68909438, 0.700927285.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.539831415, 0.658428065, 0.746312576.
CPB-v6.market-01.init neural R² for repetitions 2701/2802/2903: 0.127535207, 0.0143159115, -0.496196806.
CPB-v6.market-01 neural R² for repetitions 2701/2802/2903: 0.634143112, 0.778464525, 0.341293674.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_trend; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263132963 | -0.253979001 | 100 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.569875231 | 100 |
| CPB-v5.market-01 | 40 | 0.868645445 | 0.523928784 | 100 |
| CPB-v6.market-01.init | 40 | -1.03185689 | -0.5918479 | 100 |
| CPB-v6.market-01 | 40 | 0.675860749 | 0.270702751 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 4.37234916e-05 | 5.7482267e-05 | 5.85732789e-05 | 7.49730861e-05 |
| CPB-v4.market-01 | 1.59795965e-05 | 2.03977321e-05 | 3.49918123e-05 | 4.32661974e-05 |
| CPB-v5.market-01 | 2.00877722e-05 | 2.42695594e-05 | 3.60577464e-05 | 4.60696486e-05 |
| CPB-v6.market-01.init | 6.69073927e-05 | 9.54521865e-05 | 6.32738855e-05 | 8.31063899e-05 |
| CPB-v6.market-01 | 3.03383766e-05 | 3.81245859e-05 | 4.64219318e-05 | 5.71545039e-05 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: -0.319245159, -0.237747895, -0.204943949.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.640552095, 0.358116367, 0.71095723.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.512587238, 0.442624323, 0.61657479.
CPB-v6.market-01.init neural R² for repetitions 2701/2802/2903: -1.42117034, -0.296082931, -0.0582904283.
CPB-v6.market-01 neural R² for repetitions 2701/2802/2903: 0.204803498, 0.321815119, 0.285489636.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_activity; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840311 | 0.541426056 | 100 |
| CPB-v4.market-01 | 36 | 0.835143075 | 0.687370545 | 100 |
| CPB-v5.market-01 | 40 | 0.844256819 | 0.617983776 | 100 |
| CPB-v6.market-01.init | 40 | 0.11098842 | 0.546273981 | 100 |
| CPB-v6.market-01 | 40 | 0.672769992 | 0.323179998 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.178143553 | 0.212835342 | 0.303552885 | 0.387150693 |
| CPB-v4.market-01 | 0.186144676 | 0.233337263 | 0.251684365 | 0.312491365 |
| CPB-v5.market-01 | 0.196732151 | 0.226795807 | 0.283586415 | 0.346423083 |
| CPB-v6.market-01.init | 0.439655457 | 0.541856403 | 0.308563906 | 0.384794713 |
| CPB-v6.market-01 | 0.274086555 | 0.328743415 | 0.381790574 | 0.471796096 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: 0.551959766, 0.64999403, 0.42232437.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.868267037, 0.635226168, 0.558618431.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.777561455, 0.374143234, 0.702246639.
CPB-v6.market-01.init neural R² for repetitions 2701/2802/2903: 0.457036142, 0.503704623, 0.678081177.
CPB-v6.market-01 neural R² for repetitions 2701/2802/2903: 0.19661957, 0.386097575, 0.386822848.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_relation; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.00700444683 | -0.0855656715 | 100 |
| CPB-v4.market-01 | 36 | 0.326405067 | 0.209510464 | 100 |
| CPB-v5.market-01 | 40 | 0.252328556 | 0.370505348 | 100 |
| CPB-v6.market-01.init | 40 | 0.0873076083 | -0.253444377 | 100 |
| CPB-v6.market-01 | 40 | 0.622100687 | 0.322683609 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Physical errors in correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.0723982929 | 0.086491159 | 0.0702199215 | 0.0894458833 |
| CPB-v4.market-01 | 0.0562974528 | 0.0707384319 | 0.0615928121 | 0.0766223466 |
| CPB-v5.market-01 | 0.0586723303 | 0.074526619 | 0.0547690214 | 0.0682288649 |
| CPB-v6.market-01.init | 0.0683223609 | 0.0823413983 | 0.0774026141 | 0.0961035738 |
| CPB-v6.market-01 | 0.0456908024 | 0.0529839185 | 0.055655748 | 0.0702095263 |

CPB-v3.market-01 neural R² for repetitions 2701/2802/2903: -0.300654281, 0.161014634, -0.117057368.
CPB-v4.market-01 neural R² for repetitions 2701/2802/2903: 0.182210538, 0.23983439, 0.206486466.
CPB-v5.market-01 neural R² for repetitions 2701/2802/2903: 0.317728582, 0.48696529, 0.306822173.
CPB-v6.market-01.init neural R² for repetitions 2701/2802/2903: 0.0468624742, -0.471401748, -0.335793857.
CPB-v6.market-01 neural R² for repetitions 2701/2802/2903: 0.0570793218, 0.525771962, 0.385199543.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_intact; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01 | 36 | 98.3072917 | 96.9618056 | 100 |
| CPB-v5.market-01 | 40 | 97.1354167 | 96.5711806 | 100 |
| CPB-v6.market-01.init | 40 | 94.2708333 | 93.8368056 | 100 |
| CPB-v6.market-01 | 40 | 98.6979167 | 98.4809028 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_deleted; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.1458333 | 91.7534722 | 100 |
| CPB-v4.market-01 | 36 | 94.6614583 | 93.359375 | 100 |
| CPB-v5.market-01 | 40 | 92.96875 | 92.7951389 | 100 |
| CPB-v6.market-01.init | 40 | 88.28125 | 85.8072917 | 100 |
| CPB-v6.market-01 | 40 | 95.3125 | 93.4027778 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_intact; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.3489583 | 98.1770833 | 100 |
| CPB-v4.market-01 | 36 | 99.3489583 | 98.828125 | 100 |
| CPB-v5.market-01 | 40 | 99.21875 | 98.8715278 | 100 |
| CPB-v6.market-01.init | 40 | 94.4010417 | 93.7934028 | 100 |
| CPB-v6.market-01 | 40 | 99.7395833 | 99.21875 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_deleted; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.2708333 | 100 |
| CPB-v4.market-01 | 36 | 95.8333333 | 95.6163194 | 100 |
| CPB-v5.market-01 | 40 | 94.140625 | 94.0972222 | 100 |
| CPB-v6.market-01.init | 40 | 86.71875 | 84.0711806 | 100 |
| CPB-v6.market-01 | 40 | 97.265625 | 95.3125 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.

| Method | Updates | TRAIN MAE | VALIDATION MAE | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.25904599 | 0.376414231 | 144.341116 (reused) |
| CPB-v4.market-01 | 2048 | 0.272508032 | 0.308316924 | 172.860055 (reused) |
| CPB-v5.market-01 | 2048 | 0.254463936 | 0.375196294 | 134.983334 (reused) |
| CPB-v6.market-01.init | 0 | 0.664167857 | 0.958064934 | 0 |
| CPB-v6.market-01 | 2048 | 0.250172579 | 0.354519735 | 175.575555 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Same query distributions; NLL omits constant and can be negative.

| Method | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- |
| CPB-v3.market-01 | -0.38017808 | 1.52179658 |
| CPB-v4.market-01 | -0.356692772 | -0.0453814881 |
| CPB-v5.market-01 | -0.324377373 | 2.68745022 |
| CPB-v6.market-01.init | 0.423815353 | 0.795376511 |
| CPB-v6.market-01 | -0.400207362 | 0.648448045 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. All 22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Frozen floor/ceiling | Source | Rule | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.877120189 | 0.964673434 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.58463377 | 0.648190685 | CPB-v5.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:linear_r2 | 0.675860749 | 0.90721343 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:neural_mean_r2 | 0.270702751 | 0.569875231 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:linear_r2 | 0.672769992 | 0.862840311 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.323179998 | 0.687370545 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:linear_r2 | 0.622100687 | 0.326405067 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.322683609 | 0.370505348 | CPB-v5.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.986979167 | 0.983072917 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.984809028 | 0.969618056 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.953125 | 0.946614583 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.934027778 | 0.93359375 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.997395833 | 0.993489583 | CPB-v3.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.9921875 | 0.988715278 | CPB-v5.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.97265625 | 0.958333333 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.953125 | 0.956163194 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | train:standardized_MAE | 0.250172579 | 0.254463936 | CPB-v5.market-01 | candidate <= floor + 1e-12 | PASS |
| MKT-HIST-2 | validation:standardized_MAE | 0.354519735 | 0.308316924 | CPB-v4.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | 0.648448045 | -0.0453814881 | CPB-v4.market-01 | candidate < floor - 1e-12 | FAIL |
| MKT-HIST-2 | past_trend:all 3_neural_R2_positive | 0.204803498, 0.321815119, 0.285489636 | 0 | strict positivity | all three final neural trend R2 values > 0 | PASS |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.622100687 | 0 | strict positivity | candidate > 0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.322683609 | 0 | strict positivity | candidate > 0 | PASS |

The exact declared trajectory is stopped at 2048. Retain every failed condition and all three references and their scoped strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue.


## Every saved market neural repetition

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_volatility; physical MAE/RMSE in logreturn. V3/V4/V5 final rows reused.

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | 0.34452549910387642 | 0.00023591343985291372 | 0.00027352253772491852 |
| CPB-v3.market-01 | 2802 | 0.41467285432586121 | 0.00023639862741390167 | 0.00025847262252737751 |
| CPB-v3.market-01 | 2903 | 0.62995741473494571 | 0.00017443214975385827 | 0.00020551383789714291 |
| CPB-v4.market-01 | 2701 | 0.27550702165012397 | 0.00021858422420986478 | 0.00028756253717808527 |
| CPB-v4.market-01 | 2802 | 0.68909438026645409 | 0.00015630685364439475 | 0.00018837770537763807 |
| CPB-v4.market-01 | 2903 | 0.70092728541468285 | 0.00015292851649123827 | 0.00018475815330554746 |
| CPB-v5.market-01 | 2701 | 0.5398314153274193 | 0.00019641478626913396 | 0.0002291784087142298 |
| CPB-v5.market-01 | 2802 | 0.65842806455623526 | 0.00015750320745241375 | 0.00019744962201088365 |
| CPB-v5.market-01 | 2903 | 0.74631257625349345 | 0.00014470294524033649 | 0.00017016282439398178 |
| CPB-v6.market-01.init | 2701 | 0.12753520698223764 | 0.00025782611004414877 | 0.00031556524928591116 |
| CPB-v6.market-01.init | 2802 | 0.014315911487901012 | 0.00026318239141584422 | 0.00033541624244850225 |
| CPB-v6.market-01.init | 2903 | -0.49619680623717621 | 0.00034361786124379129 | 0.0004132468749866081 |
| CPB-v6.market-01 | 2701 | 0.63414311205428331 | 0.00016582920056088554 | 0.00020434820834698217 |
| CPB-v6.market-01 | 2802 | 0.77846452532821797 | 0.00012412590780687448 | 0.00015901454703187086 |
| CPB-v6.market-01 | 2903 | 0.34129367365915408 | 0.00022393598071355461 | 0.00027419601179503212 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_trend; physical MAE/RMSE in logreturn/tick. V3/V4/V5 final rows reused.

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | -0.31924515917541196 | 6.3963534136833448E-05 | 7.6913411986060163E-05 |
| CPB-v3.market-01 | 2802 | -0.23774789512546346 | 5.5858253568333586E-05 | 7.4499853938872243E-05 |
| CPB-v3.market-01 | 2903 | -0.20494394912226821 | 5.5898049050683185E-05 | 7.3505992447245853E-05 |
| CPB-v4.market-01 | 2701 | 0.64055209499749255 | 3.2547644876162492E-05 | 4.0147388573822634E-05 |
| CPB-v4.market-01 | 2802 | 0.35811636672973135 | 4.5291096313565925E-05 | 5.3649721785431941E-05 |
| CPB-v4.market-01 | 2903 | 0.71095723038427039 | 2.7136695798203504E-05 | 3.6001481934295893E-05 |
| CPB-v5.market-01 | 2701 | 0.51258723763894287 | 3.8131670295723817E-05 | 4.6750662684554344E-05 |
| CPB-v5.market-01 | 2802 | 0.44262432315537048 | 3.8079898067553624E-05 | 4.9993475486283194E-05 |
| CPB-v5.market-01 | 2903 | 0.61657479041229912 | 3.1961670878049018E-05 | 4.1464807630593263E-05 |
| CPB-v6.market-01.init | 2701 | -1.4211703413214605 | 7.8507654504916639E-05 | 0.00010419622759658911 |
| CPB-v6.market-01.init | 2802 | -0.29608293140274866 | 5.8996207726486635E-05 | 7.6235230667838066E-05 |
| CPB-v6.market-01.init | 2903 | -0.058290428341196243 | 5.2317794235575962E-05 | 6.8887711459334242E-05 |
| CPB-v6.market-01 | 2701 | 0.2048034983255218 | 5.011183113829932E-05 | 5.9714053577817311E-05 |
| CPB-v6.market-01 | 2802 | 0.32181511909160265 | 4.2539136603292595E-05 | 5.5145918528851938E-05 |
| CPB-v6.market-01 | 2903 | 0.2854896361852024 | 4.6614827775815047E-05 | 5.6603539530083729E-05 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_activity; physical MAE/RMSE in log1p(base_volume). V3/V4/V5 final rows reused.

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | 0.55195976639400202 | 0.29089899562705657 | 0.38467061163255906 |
| CPB-v3.market-01 | 2802 | 0.64999403029895886 | 0.27367612028609056 | 0.33999163060351023 |
| CPB-v3.market-01 | 2903 | 0.42232437005499801 | 0.3460835404495114 | 0.43678983592633752 |
| CPB-v4.market-01 | 2701 | 0.86826703651099679 | 0.17215990526731362 | 0.20858243419875477 |
| CPB-v4.market-01 | 2802 | 0.63522616762538675 | 0.28349025597915939 | 0.34709018947973269 |
| CPB-v4.market-01 | 2903 | 0.55861843089646523 | 0.29940293293641962 | 0.38180147128226033 |
| CPB-v5.market-01 | 2701 | 0.77756145523406128 | 0.22372946167609226 | 0.27104133209336251 |
| CPB-v5.market-01 | 2802 | 0.37414323416363171 | 0.38269948825671996 | 0.45464034884933957 |
| CPB-v5.market-01 | 2903 | 0.70224663889313699 | 0.24433029644033016 | 0.31358756785903819 |
| CPB-v6.market-01.init | 2701 | 0.45703614217843513 | 0.31372706235863207 | 0.42346347927293043 |
| CPB-v6.market-01.init | 2802 | 0.50370462337023281 | 0.34518020453247861 | 0.40485603288746769 |
| CPB-v6.market-01.init | 2903 | 0.67808117711672522 | 0.26678445196170103 | 0.32606462619408588 |
| CPB-v6.market-01 | 2701 | 0.19661956996740371 | 0.40345948780854074 | 0.51509949566463853 |
| CPB-v6.market-01 | 2802 | 0.38609757480700313 | 0.37059451964450096 | 0.45027742624408723 |
| CPB-v6.market-01 | 2903 | 0.38682284805737521 | 0.37131771312820122 | 0.45001136548998122 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_relation; physical MAE/RMSE in correlation. V3/V4/V5 final rows reused.

| Method | Repetition | Neural R² | Neural MAE | Neural RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2701 | -0.30065428105903469 | 0.074939390348140164 | 0.098296252324748715 |
| CPB-v3.market-01 | 2802 | 0.16101463449470332 | 0.062049591126866768 | 0.078946560826134118 |
| CPB-v3.market-01 | 2903 | -0.1170573678474649 | 0.073670782968475457 | 0.09109483665384363 |
| CPB-v4.market-01 | 2701 | 0.18221053804312992 | 0.059011701866965194 | 0.077942938990507568 |
| CPB-v4.market-01 | 2802 | 0.23983438970069992 | 0.064438729719716106 | 0.075146737755645995 |
| CPB-v4.market-01 | 2903 | 0.20648646572074181 | 0.061328004777960909 | 0.076777363045693922 |
| CPB-v5.market-01 | 2701 | 0.31772858216902866 | 0.060875889075655422 | 0.071192560285472814 |
| CPB-v5.market-01 | 2802 | 0.48696529005645239 | 0.047522223128582251 | 0.061734707830253283 |
| CPB-v5.market-01 | 2903 | 0.30682217256380928 | 0.055908951980067567 | 0.071759326505342599 |
| CPB-v6.market-01.init | 2701 | 0.046862474170524249 | 0.067086994358783084 | 0.084146064120077968 |
| CPB-v6.market-01.init | 2802 | -0.47140174835152826 | 0.086456986254590695 | 0.10454942691653586 |
| CPB-v6.market-01.init | 2903 | -0.33579385652958704 | 0.078663861678739802 | 0.099615230360170642 |
| CPB-v6.market-01 | 2701 | 0.057079321768938263 | 0.065641615496310562 | 0.083693860865312497 |
| CPB-v6.market-01 | 2802 | 0.52577196162877549 | 0.047811467092534254 | 0.059353951376918762 |
| CPB-v6.market-01 | 2903 | 0.38519954260645128 | 0.053514161265242277 | 0.067580766571134057 |

## Composite objective on the unchanged fixed queries

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · fixed query TRAIN 112 / known chronological VALIDATION 32. Components are standardized supported-cell/channel/example means. Variance NLL equals ordinary Gaussian NLL numerically at the same forward; its detached-error gradient route differs. Negative variance/total values are valid.

| Stage and split | Mean Huber(delta=1) | Variance NLL | Unit-sum objective |
| --- | --- | --- | --- |
| Initial TRAIN | 0.32371866461853954 | 0.4238153534628793 | 0.7475340180814188 |
| Initial VALIDATION | 0.5826438083137091 | 0.7953765111592087 | 1.378020319472918 |
| Final TRAIN | 0.0843757275086087 | -0.400207361741669 | -0.3158316342330604 |
| Final VALIDATION | 0.13370756455552302 | 0.6484480452157971 | 0.7821556097713201 |

The actual first/last training-batch witnesses are distinct from fixed queries.
Their composite scalars, support/context and CSV associations are independently
checked without replay; no full parameter-gradient Jacobian is reconstructed.
Recorded global norm is observed evidence; factor arithmetic is checked as
min(1, 1/(norm+ 1e-6)). The nonzero CUDA fixture and typed policy fields bind routing,
which cannot be inferred from numerical equality of variance/reporting NLL.

## Verification, costs and protected artifacts

Engineering 213752 checks/7372 CUDA calls/77 typed negative cases and the 48-check
nonzero CUDA mechanism test pass. The actual saved initial 0/final 2048 models each
pass 106120 checks, 3685 CUDA calls, 128 cutoffs/2350 suffix cases/30 internal cases;
no extra updates and state remains frozen. The final-proof settings typo stopped
before model loading/forwards. Its failure plus declared recovery attempt 002
remain exact, with no repeated training, quality or failed-command forwards.

The independent training audit passes 270721 checks: core 21202, fresh stats 61153,
saved states 1201, fixed query Gaussian/components117864, actual pre-step composite
witnesses 69301; zero mismatches. It makes no encoder/decoder/head/update calls.
All 76 learned tensors receive nonzero CUDA gradients and change; protected prior
and fitted statistics stay exact. This does not assert all individual scalars move.
Quality audit 001 passed all 13 full native40 initial DATA/statistic pairings against V5,
but its overall status failed a missing fresh-initialization JSON selector.
Reviewed additive audit 002 passes 12218 checks with zero mismatches; it explicitly
accepts the legitimate selector absence and retains typed policy/source/numeric
checks. No frozen producer bytes/numbers change.
Earlier V5/V4 pairing covered only prefix36 plus statistics. V6's own initial
heads are newly fitted even when all 40 initial values match numerically.

Step compute 175.575555496 s, trajectory 227.065203786 s and wrapper 242.243735468 s
are separate scopes. Step compute includes the F64 composite/reporting kernels,
backward, clip/AdamW and in-step contracts; loading/masks, checkpoint/evidence I/O,
proofs, export, fixed-head fitting and audits are separate. Single-pass quality
157.282324796 s is not encoder training. Old costs are reused, not a speed benchmark.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · local checkpoint file metadata for the one TRAIN trajectory; no new quality measurement.

| Artifact | Updates | Bytes | SHA256 |
| --- | --- | --- | --- |
| inference-0.pt | 0 | 1299890 | 7fc69034f46682bfeb608c152368f2bbb23c10db9ccfecb93264b8d6e49e3ed1 |
| inference-2048.pt | 2048 | 1301025 | 8cf6f12609b647871c4c42df39fc1c1d36b53a908e962585f17fe515d8ef193e |
| training-512.pt | 512 | 3780661 | 7a1845f5675f8ef061beed66b04f5aa0e26bc337616cdc89d3a68df21f59a8d5 |
| training-1024.pt | 1024 | 3781292 | 2b6ac0100d093b14326fbaf124bb984c12f8930a36e2d4f6f15a943aa6ef546c |
| training-2048.pt | 2048 | 3781292 | 598fa2fe8cb1d4322e448cf35026f01a2deec090922b1b534b51393c3419d7b1 |

Local root: output/runs/causal-fresh-v6/run-001/checkpoints/. The complete 400-file /
218709469-byte mirror passes; 307 PT files ignored. All old checkpoints, sources,
results and failed attempts remain preserved. Unmeasured report drafts and exact
V5 report parents were copied before measured population under
output/engineering/causal-fresh-v6-human-reports/attempt-001/.

The durable [machine table](results/causal_fresh_v6/table_rows.json) is 394229 B,
SHA 0e4b665b7e7086271fe5a2556303c90383f467508ef1ab0e76507737414c491c;
[comparison](results/causal_fresh_v6/comparison.json) is 226032 B,
SHA d4af012a653f85b8fcbd98f2d3d38bb7737305a61c8a6270477794c6b5035203.
[TRAIN audit](causal_fresh_v6/independent-training-audit.json)38112 B,
SHA ce548d9100d6137381c7c863c796c947fbce940e52f7e45406718b9dd3be4169.
Read the [card](CAUSAL_FRESH_V6_CARD.md),
[retention protocol](clean_causal_encoder_search_goal_v5_retention.json),
[initial proof](results/causal_fresh_v6/initial-causal-proof.json),
[final proof](results/causal_fresh_v6/final-causal-proof.json) and
[proof recovery](causal_fresh_v6/saved-proof-recovery-attempt002.json).
The [quality audit](causal_fresh_v6/quality-independent-audit.json) is 219068 B,
SHA a1bcd600e7be558132eb2e77911e423a7f008d048951aed68c579671fb255fe3:
12218 checks, zero mismatches, 13 full native40 pairings exact. The separate
[final establishment](causal_fresh_v6/final-establishment.json) confirms full trust
and measured-candidate status while numeric 12/22 fails overall advancement.
No overall winner, automatic promotion, next candidate, commit or push follows.
A separate metadata-establishment invocation 001 failed during PowerShell argument
construction before its saved-data source executed. Preserved correction 002 uses
the same frozen source and passes; it repeats no encoder/head/training execution.
Final establishment SHA256:
`ff3a26ea261fb9a8aa9ce681ee37ff882ef703f0056dce421eefc3e1a641c6fd`.