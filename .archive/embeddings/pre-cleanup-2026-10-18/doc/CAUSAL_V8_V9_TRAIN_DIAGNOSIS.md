<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE. -->
# Audited saved TRAIN diagnosis: CPB-v8 and CPB-v9

No new encoder experiment. This report reuses saved CPB-v8.market-01 and
CPB-v9.market-01 artifacts. One bounded diagnostic body and one independent
numeric audit ran in the existing `cuwacunu` container. The audit passed
**687,825 checks, zero mismatches**, including **256 scalar finite differences**
with maximum absolute error **1.7045298505991013e-10**.

V9 reconstructs the fixed TRAIN query slightly better, while seven of eight
final TRAIN readout R² values are slightly lower than V8. The temporal routes
learned; the evidence does not establish a dead bin, missing global access,
loss-gradient dominance, or a cause of held-out failures. A single generic
objective-route experiment is a more informative next hypothesis than another
pooling mixture, subject to a separate prospective declaration. No V10 or
other new candidate is selected here.

## Evidence stage and what changed

Dataset: **MKT-HIST-2** — BTC/ETH/SOL January 2024 minute-candle histories,
C3/H128/F9, four recent-128 history regressions, **designed complexity 4/5**.
The complexity is an ordinal recipe label fixed before measurement; it is not
accuracy, a measured difficulty, or an audit-check count. These are past-window
retention tasks, not forecasts. The three assets are joint input channels in one
historical panel, not three independent market cohorts.

Each saved head fit uses 112 TRAIN examples, and this diagnosis scores those
same 112 examples. These are **in-sample TRAIN readouts**, not new VALIDATION
scores. The historical split also has 32 VALIDATION examples; none of their
numeric tensor fields is read by this diagnosis. Existing held-out results remain
in [V8's report](CAUSAL_FRESH_V8_RESULTS.md) and
[V9's report](CAUSAL_FRESH_V9_RESULTS.md), under their original cards and pass
counts. Those results are not recomputed or ranked by the TRAIN tables below.

There is one historical fresh CUDA encoder trajectory per version, each with
2,048 optimizer updates, batch 8, the paired 943001–943004 constructor/row/mask/
context seeds, and freshly fitted permitted-TRAIN statistics. There is no new
encoder training, calibration, export, neural forward, readout fit or replay here.
Initial means the saved fresh 0-update checkpoint; final means the saved
2,048-update checkpoint. Three head repetitions are head variation on one
checkpoint/split, not three independent encoder retrainings.

Both methods serve native40: learned temporal20, fixed spectral12, mean4 and
uncertainty4. They retain the same registered shapes/draw order and capacity:
298,640 parameter values, 294,752 fresh learned values in 76 tensors and 3,888
fixed prior values. The common scalar `pool_score.bias` is a softmax gauge:
its recorded gradient/movement is observed separately, not required as meaningful
data learning. The historical learning criterion covers 75 identifiable tensors
and named projection/position groups, not every scalar's nonzero gradient.

V8 separately pools four original-position bins. V9 gives each nonempty bin's
existing eight export rows a fixed half local-pool/half whole-history-pool hidden
vector before applying its projection and bias once. Empty local bins still zero
the whole export slice after bias. Both causal temporal paths use this rule;
causality is relative to the requested history cutoff, not an earlier bin endpoint.
There is no new tensor, head width, mixture search or learned-state import.

The loss remains unit Huber(delta 1) mean reconstruction plus unit Gaussian
variance NLL with detached mean error, using the same hierarchical support and
F64 arithmetic. Ordinary Gaussian NLL is a reporting value, not the backward
mean loss. Global norm clipping at 1 and one AdamW remain unchanged. The direct
mean/variance native routes are separated, but shared clipping and subsequent
residual/data/statistics/AdamW dynamics do not imply optimizer independence.

Fixed saved readouts: ridge penalty 1; neural 16 tanh units/four outputs,
Adam 0.01, 100 full-batch masked standardized-target-MSE updates. Feature/target
normalization was fitted on permitted probe TRAIN only. D40 gives 164 linear and
724 neural fitted values. Repetitions 2701/2802/2903 use the unchanged unsigned
seeds `6049515622977958931`, `8262941649267994858`,
`2095371932330426401`. Deterministic ridge is fitted once per state/target and
reused across repetition receipts. No best repetition is selected. No PCA,
head tuning or post-encoder projection is used.

## Four TRAIN task tables

The tables follow [reporting standard 1.2](RESULTS_REPORTING_STANDARD.md).
Method labels ending in `.init` identify the saved **INITIAL, 0-update** checkpoint.
Labels without `.init` identify the **FINAL, 2,048-update** checkpoint. The detailed
tables also give the state explicitly. Values are the already
saved means over all three readout repetitions, formatted to ten significant
digits. They are not newly reduced statistics. Coverage is 112/112 throughout.
R² uses the scored TRAIN population's mean; it is not an accuracy percentage.

### Past volatility

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns in the recent128 suffix · **complexity 4/5 (ordinal)** · score: saved TRAIN112, first pair initial / second pair final.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v8.market-01.init | 40 | 0.8803339965 | 0.9414784295 | 100 (112/112) |
| CPB-v9.market-01.init | 40 | 0.8681049024 | 0.9346514182 | 100 (112/112) |
| CPB-v8.market-01 | 40 | 0.9914216872 | 0.9633322409 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.990979636 | 0.9611169145 | 100 (112/112) |

### Past trend

Dataset: **MKT-HIST-2** · channel-mean supported adjacent close log-return trend in the recent128 suffix · **complexity 4/5 (ordinal)** · score: saved TRAIN112, first pair initial / second pair final.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v8.market-01.init | 40 | 0.7254991662 | 0.9602816863 | 100 (112/112) |
| CPB-v9.market-01.init | 40 | 0.7078987245 | 0.9591511634 | 100 (112/112) |
| CPB-v8.market-01 | 40 | 0.913545227 | 0.9526524124 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.9075641665 | 0.9524614608 | 100 (112/112) |

### Past activity

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) in the recent128 suffix · **complexity 4/5 (ordinal)** · score: saved TRAIN112, first pair initial / second pair final.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v8.market-01.init | 40 | 0.9201412833 | 0.9602650006 | 100 (112/112) |
| CPB-v9.market-01.init | 40 | 0.9382282393 | 0.9562879264 | 100 (112/112) |
| CPB-v8.market-01 | 40 | 0.9754597999 | 0.9736358764 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.9753968809 | 0.9731731582 | 100 (112/112) |

### Past relation

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns in the recent128 suffix · **complexity 4/5 (ordinal)** · score: saved TRAIN112, first pair initial / second pair final.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v8.market-01.init | 40 | 0.6589078418 | 0.9198362066 | 100 (112/112) |
| CPB-v9.market-01.init | 40 | 0.6549993607 | 0.9101974546 | 100 (112/112) |
| CPB-v8.market-01 | 40 | 0.713897615 | 0.8759586143 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.7176859025 | 0.8734249472 | 100 (112/112) |

Only final relation-linear improves: V8 R² 0.7138976150333574 versus
V9 0.717685902495712. The other seven final task/head means are slightly lower.
Final trend-neural stays strong on TRAIN: V8 0.9526524124305243 and
V9 0.9524614608176227. This is evidence of fitted TRAIN information, not evidence
that the representation generalizes or that the historical held-out trend failure
has been explained. Exact fields: `comparison.saved_TRAIN_head_scores` and
`models.{v8,v9}.stages.{initial,final}.per_task_mean_across_saved_repetitions`.

## Saved head repetitions and physical errors

Every saved repetition appears below; these are reused results, not newly fitted
heads. MAE/RMSE are inverse-scaled original target units, separate from standardized
reconstruction MAE. Physical errors remain task-specific. The repeated linear
scores are the same deterministic ridge fit; tiny last-digit formatting differences
in receipts are not separate fits. No intervals across independent market cohorts
are inferred from these repetitions.

### Past volatility — all saved repetitions

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns in the recent128 suffix · **complexity 4/5 (ordinal)** · score: in-sample TRAIN112 · errors in **logreturn**.

| Method | State | Rep | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE | Coverage % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v8.market-01 | initial | 2701 | 0.8803339965 | 0.9513096303 | 0.0001303462567 | 0.0001741174034 | 8.067858447E-05 | 0.0001110652822 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2802 | 0.8803339965 | 0.9314237379 | 0.0001303462567 | 0.0001741174034 | 9.341726615E-05 | 0.0001318085836 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2903 | 0.8803339965 | 0.9417019203 | 0.0001303462567 | 0.0001741174034 | 8.698646884E-05 | 0.0001215301162 | 100 (112/112) |
| CPB-v8.market-01 | final | 2701 | 0.9914216872 | 0.9586312178 | 3.701699176E-05 | 4.661843906E-05 | 7.998586614E-05 | 0.0001023748213 | 100 (112/112) |
| CPB-v8.market-01 | final | 2802 | 0.9914216872 | 0.9696190647 | 3.701699176E-05 | 4.661843906E-05 | 6.818292945E-05 | 8.773186426E-05 | 100 (112/112) |
| CPB-v8.market-01 | final | 2903 | 0.9914216872 | 0.96174644 | 3.701699176E-05 | 4.661843906E-05 | 7.421010318E-05 | 9.844478564E-05 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2701 | 0.8681049024 | 0.9473639558 | 0.0001357620951 | 0.0001827978634 | 8.374444037E-05 | 0.0001154777756 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2802 | 0.8681049024 | 0.9164209768 | 0.0001357620951 | 0.0001827978634 | 0.0001007397256 | 0.0001455142194 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2903 | 0.8681049024 | 0.9401693221 | 0.0001357620951 | 0.0001827978634 | 8.969779312E-05 | 0.0001231172058 | 100 (112/112) |
| CPB-v9.market-01 | final | 2701 | 0.990979636 | 0.9519440313 | 3.828330273E-05 | 4.780450429E-05 | 8.344023794E-05 | 0.0001103393589 | 100 (112/112) |
| CPB-v9.market-01 | final | 2802 | 0.990979636 | 0.9686011417 | 3.828330273E-05 | 4.780450429E-05 | 6.93494579E-05 | 8.918949739E-05 | 100 (112/112) |
| CPB-v9.market-01 | final | 2903 | 0.990979636 | 0.9628055703 | 3.828330273E-05 | 4.780450429E-05 | 7.379219978E-05 | 9.707239406E-05 | 100 (112/112) |

### Past trend — all saved repetitions

Dataset: **MKT-HIST-2** · channel-mean supported adjacent close log-return trend in the recent128 suffix · **complexity 4/5 (ordinal)** · score: in-sample TRAIN112 · errors in **logreturn/tick**.

| Method | State | Rep | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE | Coverage % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v8.market-01 | initial | 2701 | 0.7254991662 | 0.9606384034 | 3.426405543E-05 | 4.374549567E-05 | 1.273844938E-05 | 1.65652449E-05 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2802 | 0.7254991662 | 0.9500059172 | 3.426405543E-05 | 4.374549567E-05 | 1.406968696E-05 | 1.866898944E-05 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2903 | 0.7254991662 | 0.9702007383 | 3.426405543E-05 | 4.374549567E-05 | 1.078948212E-05 | 1.441332767E-05 | 100 (112/112) |
| CPB-v8.market-01 | final | 2701 | 0.913545227 | 0.9638720934 | 1.885914435E-05 | 2.45502543E-05 | 1.141526331E-05 | 1.587021851E-05 | 100 (112/112) |
| CPB-v8.market-01 | final | 2802 | 0.913545227 | 0.9450962917 | 1.885914435E-05 | 2.45502543E-05 | 1.357624916E-05 | 1.95642114E-05 | 100 (112/112) |
| CPB-v8.market-01 | final | 2903 | 0.913545227 | 0.9489888522 | 1.885914435E-05 | 2.45502543E-05 | 1.442727926E-05 | 1.885793158E-05 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2701 | 0.7078987245 | 0.9592748549 | 3.61331303E-05 | 4.512614499E-05 | 1.269176249E-05 | 1.684972542E-05 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2802 | 0.7078987245 | 0.9701443191 | 3.61331303E-05 | 4.512614499E-05 | 1.140190942E-05 | 1.442696566E-05 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2903 | 0.7078987245 | 0.9480343163 | 3.61331303E-05 | 4.512614499E-05 | 1.429618367E-05 | 1.903355146E-05 | 100 (112/112) |
| CPB-v9.market-01 | final | 2701 | 0.9075641665 | 0.958519908 | 1.949013121E-05 | 2.538526425E-05 | 1.301889172E-05 | 1.70051851E-05 | 100 (112/112) |
| CPB-v9.market-01 | final | 2802 | 0.9075641665 | 0.9500248002 | 1.949013121E-05 | 2.538526425E-05 | 1.307000305E-05 | 1.866546344E-05 | 100 (112/112) |
| CPB-v9.market-01 | final | 2903 | 0.9075641665 | 0.9488396743 | 1.949013121E-05 | 2.538526425E-05 | 1.416545635E-05 | 1.888548567E-05 | 100 (112/112) |

### Past activity — all saved repetitions

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) in the recent128 suffix · **complexity 4/5 (ordinal)** · score: in-sample TRAIN112 · errors in **log1p(base_volume)**.

| Method | State | Rep | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE | Coverage % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v8.market-01 | initial | 2701 | 0.9201412833 | 0.9617334314 | 0.1367496117 | 0.175472971 | 0.09586621968 | 0.1214671455 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2802 | 0.9201412833 | 0.9458786062 | 0.1367496117 | 0.175472971 | 0.1060197862 | 0.1444553237 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2903 | 0.9201412833 | 0.973182964 | 0.1367496117 | 0.175472971 | 0.08078202196 | 0.1016844287 | 100 (112/112) |
| CPB-v8.market-01 | final | 2701 | 0.9754597999 | 0.9761101146 | 0.07570950481 | 0.09727206053 | 0.07218593053 | 0.09597455309 | 100 (112/112) |
| CPB-v8.market-01 | final | 2802 | 0.9754597999 | 0.9742461222 | 0.07570950481 | 0.09727206053 | 0.0795241958 | 0.09964841182 | 100 (112/112) |
| CPB-v8.market-01 | final | 2903 | 0.9754597999 | 0.9705513923 | 0.07570950481 | 0.09727206053 | 0.08245574405 | 0.1065568685 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2701 | 0.9382282393 | 0.9600611918 | 0.1178825777 | 0.1543277611 | 0.09582546921 | 0.1240928092 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2802 | 0.9382282393 | 0.9482640856 | 0.1178825777 | 0.1543277611 | 0.1043937188 | 0.1412359094 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2903 | 0.9382282393 | 0.9605385018 | 0.1178825777 | 0.1543277611 | 0.09582267762 | 0.1233490618 | 100 (112/112) |
| CPB-v9.market-01 | final | 2701 | 0.9753968809 | 0.9752602014 | 0.07655912449 | 0.09739667937 | 0.07772520976 | 0.09766684214 | 100 (112/112) |
| CPB-v9.market-01 | final | 2802 | 0.9753968809 | 0.9733753218 | 0.07655912449 | 0.09739667937 | 0.08165318163 | 0.1013190826 | 100 (112/112) |
| CPB-v9.market-01 | final | 2903 | 0.9753968809 | 0.9708839513 | 0.07655912449 | 0.09739667937 | 0.08011155984 | 0.1059534942 | 100 (112/112) |

### Past relation — all saved repetitions

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns in the recent128 suffix · **complexity 4/5 (ordinal)** · score: in-sample TRAIN112 · errors in **correlation**.

| Method | State | Rep | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE | Coverage % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v8.market-01 | initial | 2701 | 0.6589078418 | 0.9138169294 | 0.04687048595 | 0.05932233774 | 0.02249781842 | 0.02981902033 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2802 | 0.6589078418 | 0.9396828141 | 0.04687048595 | 0.05932233774 | 0.01854809387 | 0.0249461153 | 100 (112/112) |
| CPB-v8.market-01 | initial | 2903 | 0.6589078418 | 0.9060088764 | 0.04687048595 | 0.05932233774 | 0.02365520416 | 0.03114051641 | 100 (112/112) |
| CPB-v8.market-01 | final | 2701 | 0.713897615 | 0.8634692126 | 0.04399508861 | 0.05433042829 | 0.02933678157 | 0.03753165254 | 100 (112/112) |
| CPB-v8.market-01 | final | 2802 | 0.713897615 | 0.9145666663 | 0.04399508861 | 0.05433042829 | 0.02338953143 | 0.02968903395 | 100 (112/112) |
| CPB-v8.market-01 | final | 2903 | 0.713897615 | 0.849839964 | 0.04399508861 | 0.05433042829 | 0.03095261353 | 0.03936040617 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2701 | 0.6549993607 | 0.8921401614 | 0.0475858637 | 0.05966124879 | 0.02576022723 | 0.03335894234 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2802 | 0.6549993607 | 0.9544438088 | 0.0475858637 | 0.05966124879 | 0.01700984778 | 0.02167984102 | 100 (112/112) |
| CPB-v9.market-01 | initial | 2903 | 0.6549993607 | 0.8840083936 | 0.0475858637 | 0.05966124879 | 0.02764398335 | 0.03459359297 | 100 (112/112) |
| CPB-v9.market-01 | final | 2701 | 0.7176859025 | 0.869802951 | 0.04315089728 | 0.05396953451 | 0.02849304531 | 0.03665075791 | 100 (112/112) |
| CPB-v9.market-01 | final | 2802 | 0.7176859025 | 0.8999899724 | 0.04315089728 | 0.05396953451 | 0.02568035272 | 0.03212211726 | 100 (112/112) |
| CPB-v9.market-01 | final | 2903 | 0.7176859025 | 0.8504819182 | 0.04315089728 | 0.05396953451 | 0.03104264049 | 0.03927618057 | 100 (112/112) |

## Fixed TRAIN reconstruction query

The fixed query uses the unchanged saved masks. These 112-row aggregates are
separate from the first-eight detailed diagnostic tensors. No VALIDATION query
value is selected here. GPU training times are not numeric inputs to this saved
TRAIN diagnosis and are therefore left unreported, not called zero. Historical
training costs remain in the original V8/V9 reports.

Dataset: **MKT-HIST-2** · hidden-patch reconstruction of C3/H128/F9 histories ·
**complexity 4/5 (ordinal)** · score: fixed TRAIN112 query · standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| CPB-v8.market-01 | 0 | 0.6643606324 | — | — |
| CPB-v9.market-01 | 0 | 0.6642581977 | — | — |
| CPB-v8.market-01 | 2048 | 0.2471462607 | — | — |
| CPB-v9.market-01 | 2048 | 0.2445403465 | — | — |

Dataset: **MKT-HIST-2** · the same fixed hidden-target query ·
**complexity 4/5 (ordinal)** · score: TRAIN112 only · Gaussian NLL and objective
components are separate from MAE and readout quality.

| Encoder | State | Reporting Gaussian NLL | Mean Huber1 | Detached-error variance NLL | Total objective |
| --- | --- | ---: | ---: | ---: | ---: |
| CPB-v8.market-01 | initial | 0.4238833104 | 0.3238084184 | 0.4238833104 | 0.7476917288 |
| CPB-v9.market-01 | initial | 0.4238473016 | 0.3237604272 | 0.4238473016 | 0.7476077288 |
| CPB-v8.market-01 | final | -0.3812151448 | 0.0838644686 | -0.3812151448 | -0.2973506762 |
| CPB-v9.market-01 | final | -0.3924516259 | 0.08305971607 | -0.3924516259 | -0.3093919099 |

The saved final comparison reports V9-minus-V8 MAE **−0.0026059141893047255**
and NLL **−0.01123648115700332**. Lower fixed TRAIN reconstruction therefore
coexists with seven slightly lower TRAIN readout means. These values do not
establish a general representation improvement or justify selecting a checkpoint.
Negative NLL is valid for a continuous density and is not a failure or a
classification probability.

## Detailed scope, temporal learning and clipping

The independent audit checked all four model stages. First-eight fixed-query
raw/target/mask/distribution tensors support bounded scalar likelihood derivatives
and saved linear mean4 residual/variance-link contractions. They do not support
full temporal-encoder/parent-decoder/optimizer Jacobians. Initial mean4 and
uncertainty4 native adjoints are zero because of the declared zero-start residual
and variance heads; that is not a dead-branch finding. Final local adjoints are
nonzero. Mean-NLL and MAE derivatives are hypothetical diagnostic alternatives;
only Huber1 mean plus detached-error variance was actually trained.

Within V9's final mean4 basis on the first eight queries, saved Huber-versus-NLL
cosine is 0.7182744739426327 and Huber-versus-MAE is 0.8103533270407922.
Those directions differ but are positively aligned. Their norms do not establish
loss dominance, and mean4/uncertainty4 bases must not be compared by cosine or
magnitude. The direct variance-into-mean4 adjoint is exactly zero by design in
both versions. See `comparison.local_native_adjoints` for the full stage-matched
context.

Both trajectories expose exactly 2,048 saved successful scalar rows, 8,192
four-bin export/within-bin-position records, 2,048 separately observed scalar-gauge
records, and two actual update-1/update-2,048 witnesses. Each witness uses the
same twelve before/after/clipped-gradient arrays for export weight/bias, pool
positions and scalar pool-score bias. Values are promoted from F32 before F64
movement subtraction/centering. Actual minibatch components/support/row IDs and
observed clipping can be associated with their CSV rows; no pre-step-2,047 decoder
was reconstructed and no actual-batch decoder adjoint or optimizer was replayed.
The known zero-skip attempts belong to these two historical runs, not a generic
assumption for future training.

For all four original bin row groups in both versions, recorded export weight,
bias and within-bin centered-position data-gradient and movement event counts
are 2,048. The saved final weights/biases changed. This is actual named-group
learning evidence, not every-scalar gradient proof and not a claim that weight
decay movement is data learning. The common scalar gauge remains separately
observed and exempt from the nonzero data-gradient condition.

V9 also has 2,048 between-bin CSV rows. All three fixed unit Hadamard R16
directions have 2,048 recorded gradient and movement events. Saved final signed
contrast changes are −0.15708683710545301, 0.009199167601764202,
−0.1647594692185521; the overall final between-bin projection-change norm is
0.2278301599295833. Signed final coefficients are descriptive: the declared gate
requires all three direction event counts and positive overall final movement,
not a nonzero final value for every coefficient. This confirms that the restored
relative-position routes were used; it does not show why a held-out task changed.

Dataset: **MKT-HIST-2** · saved encoder optimization trajectory ·
**complexity 4/5 (ordinal)** · TRAIN observations only; counts copied from each
trajectory's existing scalar summary, not a reconstructed branch gradient.

| Method | Clipped updates / 2048 | Saved frequency | Clipped updates in final512 | Saved final512 frequency |
| --- | ---: | ---: | ---: | ---: |
| CPB-v8.market-01 | 1333 | 0.6508789062 | 437 | 0.853515625 |
| CPB-v9.market-01 | 1339 | 0.6538085938 | 443 | 0.865234375 |

Similar clipping incidence is descriptive. Neither trajectory contains the missing
mean/uncertainty branch split norms, so this report cannot infer branch dominance,
a counterfactual clipping factor, or an optimizer-coupling cause.

The four raw/reconstruction age bins cover only the first eight TRAIN queries.
Their target support is unequal and their per-bin hierarchy is different from the
full-query reduction; bin errors cannot be summed to produce the full loss.

Dataset: **MKT-HIST-2** · original-position masked reconstruction bins ·
**complexity 4/5 (ordinal)** · score: first8 TRAIN, final saved states · standardized
MAE under each bin's eligible-example/channel/cell hierarchy.

| Original history ticks | Eligible examples / 8 | Target cells | V8 MAE | V9 MAE |
| --- | ---: | ---: | ---: | ---: |
| [0, 32) | 4 | 288 | 0.09397459721 | 0.1053071428 |
| [32, 64) | 6 | 720 | 0.402717793 | 0.3795928289 |
| [64, 96) | 2 | 216 | 0.3573283032 | 0.3430495903 |
| [96, 128) | 5 | 504 | 0.2235177837 | 0.2236564078 |

The two middle bins improve here, the oldest worsens, and the newest is almost
unchanged. Different support/target distributions and eight examples do not
isolate age, prove an oldest-history failure or establish pooling capacity as the
cause. Whole-history causal states may already contain earlier context. These are
observations of the selected saved queries, not ablation or counterfactual tests.

Native learned20 participation ratio changes from V8 3.234146893169757 to V9
3.3339087571323156 on the final112 TRAIN rows. This descriptive covariance
statistic is not rank, intrinsic dimension or a causal explanation. Saved fitted
ridge block decompositions and mean4 residual allocation remain descriptive;
component correlations are not gradient conflict and cannot substitute for block
ablation/refitting. No such fit or intervention was performed.

## Diagnosis and audit cost

Dataset: **MKT-HIST-2** · saved TRAIN-only metadata/inert-array algebra ·
**complexity 4/5 (ordinal)** · one producer and one independent audit, existing
container CPU execution; no new training, classifier or GPU computation.

| Operation | Body seconds | Recorder/wrapper seconds | New encoder updates | New head fits |
| --- | ---: | ---: | ---: | ---: |
| Saved TRAIN diagnosis | 1.995412661 | 2.4694229 | 0 | 0 |
| Independent saved TRAIN audit | 1.930579447 | 1.981298885 | 0 | 0 |

Body and wrapper times are separate measurements; they are not added, called GPU
training time, or offered as a repeated performance benchmark. Extraction, original
encoder/readout training, artifact copies, staging, static reviews and the rejected
metadata generation are outside these body values.

## Admission, audit and preservation

The first generated metadata card (`5e6b5591…`, 90,868 bytes) was rejected during
actual-card static inspection before any preflight, AST, Schema or diagnosis.
An unparenthesized PowerShell array/concatenation expression created duplicate
outer checkpoint keys and literal `payload.` fields. All nine original card,
Markdown, generator, context and review files were preserved. The only source
repair parenthesized the payload concatenation. The replacement card has exact
unique checkpoint allowlists: 39 fields for V8 and 43 for V9. No scientific
source, data, score or head recipe changed; no producer run was replayed to repair
that metadata error. See the
[rejected-card review](causal_v8_v9_train_diagnosis/rejected-card-001-independent-static-review.json)
and [replacement review](causal_v8_v9_train_diagnosis/actual-card-independent-static-review-002.json).

The admitted main scope binds 56 TRAIN roles, 36 preservation roles and one final
99-checkpoint broad receipt: 93 file guards. Sixteen context metadata copies belong
to the 24-file main capsule. The exact main mirror has 38 files; the independent
10-payload/11-ZIP-member, AST4 source audit mirror has 19 files. Before/after
source and guard bytes are exact. The old 1,030-pin history, 99 critical
checkpoints and selected V3 source/checkpoint protections remain references to
preserved evidence; this report does not bulk reaudit or rewrite them.

All saved diagnostic counters are zero for encoder/decoder/model/GPU calls,
neural-head forwards, new head/scaler fits, training updates, raw-cache numeric
reads, VALIDATION/TEST numeric reads and new quality scores. Four mixed query JSON
files are parsed whole, but only their TRAIN numeric key is selected. Head-fit
archives can contain other fields; only the admitted TRAIN ridge/normalizer fields
are accessed. The independent audit imports no producer code, Torch or model.
Its 256 finite differences validate bounded scalar contracts, not an encoder or
optimizer Jacobian.

Exact artifact anchors (full SHA256):

- [Prospective admitted card](CAUSAL_V8_V9_TRAIN_DIAGNOSIS_CARD.json), 90,296 bytes:
  `ceb4c2648a2e4b3d594b2028ae91eab6a9a626f1846a1d11b2b9b0bd1f4cd950`.
- [Saved main diagnosis](causal_v8_v9_train_diagnosis/diagnosis.json), 4,788,176 bytes:
  `805d80db8f260e59a86b1b0184fe54fb3e28659e642f09f6e4e8ab99aa2a36cf`.
- [Independent audit card](causal_v8_v9_train_diagnosis/independent-audit-card.json), 8,540 bytes:
  `5b5f6d2b784ff94e00d48f7c781ab898582b0ab3706b1ae94d05c1e82e072457`.
- [Independent audit](causal_v8_v9_train_diagnosis/independent-audit.json), 89,509 bytes:
  `c0c8d5c93e95d24c8a79f3e1d6a94b597a4ac853407c38aaeea27235715d8da9`.
- [Exact main mirror](causal_v8_v9_train_diagnosis/complete-mirror.json), 7,199 bytes:
  `6e1f55f7b18dd311552d3cf341d6bbdf17a1d40f0c7c8be749c4a0443f00cc08`.
- [Exact independent audit mirror](causal_v8_v9_train_diagnosis/independent-audit-mirror.json), 9,612 bytes:
  `a3c0efb1a66b94d92cfd80175457f0a6211e27e2e33248df0f856386da814977`.
- [Independent audit execution](causal_v8_v9_train_diagnosis/independent-audit-execution.json), 6,746 bytes:
  `170a89627debe0b81ae7bb409e8699a59fab86e03774f298970b2a0c6c5bb643`.

## Next direction and unchanged selection

The joint observation is better TRAIN reconstruction with little broad TRAIN
readout benefit, despite live temporal groups and restored relative-position
learning. That makes one generic objective-route comparison more informative than
another pooling mixture. Conditional on a separately reviewed prospective design,
keep V9 pooling fixed and test one change in how the existing self-supervised
reconstruction gradient reaches the learned representation. Declare its intended
information/gradient route and falsifiable criteria before training; this report
does not choose a coefficient, loss variant, card, schedule, seed or new encoder.
Separate source-only preparation is now authorized for an objective-route proposal.
It has no training or quality admission and contributes no measurements here.
The local Huber/NLL/MAE alternatives do not identify which route would improve
retention, and TRAIN readouts do not explain VALIDATION failures.

CPB-v3.market-01 remains the sole selected baseline and dev application encoder.
All V3–V9 checkpoints, historical results/pass counts, scoped leaders, future
retention floors and prior failures remain unchanged. The search goal stays
active. No V10 selection, promotion, publication, commit or push is performed by
this diagnostic/reporting work.
