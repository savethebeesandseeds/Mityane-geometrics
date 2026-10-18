<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v6.market-01 results: 12/22, preserve timing and relation gains

**CPB-v3.market-01 remains the clean baseline and sole dev encoder.** V6's new
fixed-head measurements pass 12/22 prospectively strengthened V3/V4/V5 retention
conditions, with 100% scoring coverage. Seven TEMPO-5 scopes, linear market
relation and TRAIN query MAE improve the strongest clean references. Volatility,
trend, activity, validation reconstruction/calibration and one timing condition
still fail protection. This is a scoped result, not balanced advancement.

**Evidence stage: measured quality and independent audits passed.** Independent
training/quality audits pass 270721/12218 checks with zero mismatches; all 13
full native40 initial DATA/statistic pairings versus V5 are exact. The separate final establishment receipt confirms the measured candidate and
full trust; numeric 12/22 fails overall advancement.
The first quality audit failed one selector that required gradient_policy in
fresh-initialization JSON, which legitimately omits that key. Its complete failure
is preserved; reviewed additive audit 002 checks the absence explicitly and passes.
No model/training/head/report bytes, fixed checks or scores changed.

V6 changes only the training objective/routing relative to V5's forward capacity
and fresh construction order. Native40 is 20 learned content + 12 literal spectral
+ 4 mean + 4 uncertainty coordinates. All 294752 learned values start fresh under
943001 across 76 tensors; 3888 deterministic prior values stay fixed, 77 tensors /
298640 values total. Input/bank statistics and AdamW are freshly fitted/created
from TRAIN, source_updates=0, 2048 successful CUDA:0 updates at batch 8. No old
learned weights/statistics, teacher, repeated old encoder training or synthetic
encoder updates.

The objective is fixed unweighted Huber(delta=1) mean supervision plus Gaussian
variance NLL on detached error, both coefficients 1. Parent32/mean4 receive live
Huber gradients; uncertainty4 and the variance head receive live variance
calibration gradients. The Gaussian reporting MAE/NLL kernels stay unchanged,
and reporting NLL never backpropagates. Variance NLL and total may be negative.
One shared global clip 1/AdamW and future changing residuals retain coupling:
zero direct variance-loss mean gradient is not full optimizer independence.

MKT-HIST-2 retains 4096 TRAIN anchors, 112 TRAIN/32 known chronological VALIDATION
readout windows: TRAIN Jan 1–21, VALIDATION Jan 22–28, Jan 29–31 unused. Fixed heads
use ridge penalty 1 and tanh 16 / Adam 0.01 /100 updates, repetitions 2701/2802/2903.
V5/V6 native40 use identical width-derived seeds 6049515622977958931,
8262941649267994858, 2095371932330426401, market heads 164/724 parameters and timing
heads 82/690. V3/V4 native36 retain 148/660 and 74/626 with different seeds. V6 versus
V5 is a same-capacity loss/routing experiment; comparison to V3/V4 also differs
in encoder/readout width and head seeds.

TEMPO-5 uses saved cohorts 941101/941102/941103, each 512 TRAIN/256 VALIDATION examples
from 256/128 source groups, both declared scaler policies and intact/extra 30%
coordinate deletion. V6 initial 0/final 2048 banks are newly extracted and each gets
its own unchanged fixed heads. V3/V4/V5 final scores/costs are reused. No initial
score reuse, PCA after encoding, best head or intermediate-checkpoint selection.
Three head fits are not three encoder trajectories. These are retained past
signals on known development populations, not forecasting or a fresh period.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_volatility; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673434 | 0.463051923 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.555176229 | 100 |
| CPB-v5.market-01 | 40 | 0.95221109 | 0.648190685 | 100 |
| CPB-v6.market-01.init | 40 | 0.412200077 | -0.118115229 | 100 |
| CPB-v6.market-01 | 40 | 0.877120189 | 0.58463377 | 100 |

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_trend; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263132963 | -0.253979001 | 100 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.569875231 | 100 |
| CPB-v5.market-01 | 40 | 0.868645445 | 0.523928784 | 100 |
| CPB-v6.market-01.init | 40 | -1.03185689 | -0.5918479 | 100 |
| CPB-v6.market-01 | 40 | 0.675860749 | 0.270702751 | 100 |

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_activity; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840311 | 0.541426056 | 100 |
| CPB-v4.market-01 | 36 | 0.835143075 | 0.687370545 | 100 |
| CPB-v5.market-01 | 40 | 0.844256819 | 0.617983776 | 100 |
| CPB-v6.market-01.init | 40 | 0.11098842 | 0.546273981 | 100 |
| CPB-v6.market-01 | 40 | 0.672769992 | 0.323179998 | 100 |

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. past_relation; V3/V4/V5 final rows reused, V6 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.00700444683 | -0.0855656715 | 100 |
| CPB-v4.market-01 | 36 | 0.326405067 | 0.209510464 | 100 |
| CPB-v5.market-01 | 40 | 0.252328556 | 0.370505348 | 100 |
| CPB-v6.market-01.init | 40 | 0.0873076083 | -0.253444377 | 100 |
| CPB-v6.market-01 | 40 | 0.622100687 | 0.322683609 | 100 |

## Timing: synthetic_train / validation_intact

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_intact; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01 | 36 | 98.3072917 | 96.9618056 | 100 |
| CPB-v5.market-01 | 40 | 97.1354167 | 96.5711806 | 100 |
| CPB-v6.market-01.init | 40 | 94.2708333 | 93.8368056 | 100 |
| CPB-v6.market-01 | 40 | 98.6979167 | 98.4809028 | 100 |

## Timing: synthetic_train / validation_deleted

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. synthetic_train, validation_deleted; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.1458333 | 91.7534722 | 100 |
| CPB-v4.market-01 | 36 | 94.6614583 | 93.359375 | 100 |
| CPB-v5.market-01 | 40 | 92.96875 | 92.7951389 | 100 |
| CPB-v6.market-01.init | 40 | 88.28125 | 85.8072917 | 100 |
| CPB-v6.market-01 | 40 | 95.3125 | 93.4027778 | 100 |

## Timing: market_scaler / validation_intact

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_intact; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.3489583 | 98.1770833 | 100 |
| CPB-v4.market-01 | 36 | 99.3489583 | 98.828125 | 100 |
| CPB-v5.market-01 | 40 | 99.21875 | 98.8715278 | 100 |
| CPB-v6.market-01.init | 40 | 94.4010417 | 93.7934028 | 100 |
| CPB-v6.market-01 | 40 | 99.7395833 | 99.21875 | 100 |

## Timing: market_scaler / validation_deleted

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN 512 / VALIDATION 256. market_scaler, validation_deleted; V3/V4/V5 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.2708333 | 100 |
| CPB-v4.market-01 | 36 | 95.8333333 | 95.6163194 | 100 |
| CPB-v5.market-01 | 40 | 94.140625 | 94.0972222 | 100 |
| CPB-v6.market-01.init | 40 | 86.71875 | 84.0711806 | 100 |
| CPB-v6.market-01 | 40 | 97.265625 | 95.3125 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN 112 / known chronological VALIDATION 32. Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
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

## What the gains and failures mean

Linear relation reaches 0.6221006868548108, above V4's 0.3264050670418407 and V5's
0.25232855646478336. TRAIN query MAE reaches 0.2501725785157887, below V5's
0.25446393568049436. V6 improves all eight timing measures versus V5 and exceeds
all three prior final references in seven of eight protected timing scopes.
Market-scaler deleted neural timing 95.3125% remains below V4's 95.6163194%.
Preserve these new scopes alongside every displaced/historical reference.

Market balance still fails: volatility 0.877120188581859/0.5846337703472184 loses
to the strongest linear/neural references; trend 0.6758607489688313/0.27070275120077564
and activity 0.6727699916736933/0.323179997610594 lose to their protected leaders.
Neural activity also falls from its own initial 0.5462739808884643. Neural relation
0.322683608668055 stays below V5's 0.3705053482630968. All three final neural trend
fits remain positive; both relation heads are positive. Lower reconstruction
error does not imply uniform retained-signal gains.

Validation MAE 0.35451973488525984 and NLL 0.6484480452157971 improve on V5 but fail
V4's 0.30831692401624544 and −0.04538148809081624. TRAIN NLL −0.400207361741669 is
reported separately; no new TRAIN NLL advancement guard is introduced. The 12
passes comprise market 1/8, timing 7/8, MAE 1/2, strict validation NLL 0/1, positive
trend 1/1 and positive relation 2/2. The [detail](CAUSAL_FRESH_V6_DETAIL.md) keeps
all 22 exact rules, physical errors and every neural repetition visible.

The stronger 47fa165d325230ffca4f99c19bbec494d04f1daf0d99ae7e6bd197fdc47bec0d
protocol protects final V3/V4/V5 scopes. Original 28f6bd35b4b2c66f3f906a374604952e79b07cc876b74e4ded16339fd535de25
stays exact. V4's historical 19/22 and V5's 7/22 used earlier thresholds; do not rank
those counts as the same protocol. Numeric gates, coverage and final trust are
separate. No overall winner, automatic dev replacement, commit or push follows.

## Training, verification and preservation

The main trajectory runs once for 2048/B8. Synchronized step compute is
175.575555496 s; trajectory wall time 227.065203786 s; wrapper 242.243735468 s. Step
compute includes forward_at, F64 composite/reporting loss, backward, shared clip,
AdamW and in-step contracts; loader/masks, checkpoints, gradient evidence, proofs,
export, head fitting and audits have separate scopes. Old V3/V4/V5 times are
reused descriptive costs, not a controlled speed comparison. Single-pass quality
took 157.282324796 s and is not encoder training time.

Engineering passes 213752 checks, 7372 CUDA calls and 77 typed negative cases;
the nonzero CUDA mechanism fixture passes 48 checks. Both actual saved initial 0
and final 2048 models pass 128 cutoffs/2350 future-value-or-mask cases/30 internal
temporal interventions, with frozen state and no new updates. The erroneous
first final-proof settings path stopped before model loading/forwards; its failure
and declared recovery attempt 002 remain preserved. No quality/training replay.

The independent TRAIN audit passes 270721 checks, zero mismatches, including
fresh TRAIN calibration, typed actual512 continuation, saved moment/state checks,
all 2048 CSV schedules/components/clip factors and first/last actual pre-step
witness reductions. Every one of the 76 learned tensors receives nonzero CUDA gradients and
changes; no claim that every scalar value moves. Prior/statistics stay exact.
The independent quality audit passes 12218 checks with zero mismatches, including
all 13 full native40 saved initial pairings. Original producer pending-trust fields stay
byte exact; the separate [final establishment](causal_fresh_v6/final-establishment.json)
binds full trust and measured-candidate status, with overall advancement false.
Scores and numeric decisions are not rewritten.

All 400 main artifacts /218709469 bytes are mirrored; 307 PT files remain ignored.
All five main checkpoints, old source/results/checkpoints and failed attempts
remain protected. Read the [card](CAUSAL_FRESH_V6_CARD.md),
[comparison](results/causal_fresh_v6/comparison.json),
[saved detailed rows](results/causal_fresh_v6/table_rows.json),
[TRAIN audit](causal_fresh_v6/independent-training-audit.json),
[quality audit](causal_fresh_v6/quality-independent-audit.json) and
[saved-proof recovery](causal_fresh_v6/saved-proof-recovery-attempt002.json).

## Next direction

Stop this exact recipe at 2048 without coefficient, seed, rate, budget, head or
checkpoint rescue. Keep V3 selected and preserve V4/V5 strengths plus V6's new
scopes. Next propose a separately declared saved TRAIN-only V5/V6 diagnosis of
content20/mean4/uncertainty4, existing ridge contributions and reconstruction
components. Inspect all 112 TRAIN summaries separately from first 8 cellwise
witnesses; no held-out diagnosis, old encoder/head replay or new fit. These
measurements do not by themselves identify why trend/activity declined. No next
candidate or mechanism is declared here. Read [the review](CAUSAL_FRESH_V6_REVIEW.md).