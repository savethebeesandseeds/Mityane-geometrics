<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v5 review and next direction

CPB-v3.market-01 remains selected. CPB-v5.market-01 is a clean scoped reference,
not balanced advancement: 7/22 strongest-clean-reference conditions pass, with
complete trust admission and 100% coverage. Independent saved-quality audit
passes 11300 checks and independent TRAIN audit 192589, both with zero mismatches.
Stop this exact separate-projection recipe without coefficient/seed/rate/budget/head rescue.
The active clean encoder goal continues; preserve all V3/V4/V5 strengths.

V5 separates mean4 and uncertainty4 bank projections and exports native40.
All 294752 learned values start fresh; every one of the 76 learned tensors
receives nonzero gradients and changes. The 3888 fixed-prior values
and new TRAIN statistics stay protected. Gaussian NLL and its positive link are
unchanged. The new uncertainty projection receives live variance gradients,
separate from parent32/mean4; mean learning still uses error/variance. This tests
added representation capacity; independent unweighted mean supervision was not tested. Wider
D40 heads and different width-derived seeds prevent attributing changes solely
to the extra encoder branch. No historical weights, teacher or fitted statistics
initialize the candidate.

Neural volatility reaches 0.6481906853790493 versus V4's 0.5551762291104203;
neural relation reaches 0.3705053482630968 versus 0.20951046448819055. All three
final neural trend and relation fits are positive. Market-scaler intact neural
timing reaches 98.87152777777778%, narrowly above V4's 98.828125%. TRAIN query
MAE 0.25446393568049436 improves on V3's 0.25904599043427573. These are scoped
records on known development populations, with capacity confounds, not universal
encoder or forecasting improvements.

The protected failures remain visible. Linear volatility 0.9522110896824942 is
below V3's 0.964673433811061; both trend heads are below V4. Linear activity is
below V3, and neural activity/linear relation are below V4. Only 1/8 strongest
TEMPO-5 retention conditions passes. Validation query MAE 0.37519629363955453
loses to V4's 0.30831692401624544, and NLL 2.6874502201003447 loses to −0.04538148809081624.
New records and lower TRAIN MAE do not cancel those failures. The 7/22 count
uses stronger best-V3/V4 criteria than V4's historical 19/22 protocol.

This tradeoff does not prove uncertainty coordinates caused poor calibration,
that decoder variance is market volatility, or that one gradient route explains
generalization. It is one trained encoder, one known market period and three
fixed head fits, plus three saved synthetic cohorts. The 134.9833336049999s
training time is descriptive synchronized step time; reused previous costs do
not establish a controlled speed improvement. Thirteen initial prefix/statistic
pairings are numerically exact, while the new uncertainty4 is finite/nonzero;
full native40 heads still need their own initial evaluation and receive it.

Next propose bounded saved TRAIN-only V4/V5 diagnosis before choosing another
mechanism. Compare 112 natural-mask TRAIN feature rows, existing ridge
components/predictions and each block under its saved normalizers. Inspect
all 112 aggregate TRAIN query MAE/NLL rows separately from the first 8 saved
cellwise witnesses. For V5, distinguish mean4[32:36] and uncertainty4[36:40],
and use masked query natives for local derivatives. Examine whether remaining
inverse-variance weighting misaligns mean-reconstruction pressure, or whether
another representation effect is better supported. This may motivate
unweighted Huber1 mean supervision with separate variance calibration; exact
routing and geometry require a prospective declaration. It is not selected or
declared yet.

Do not use held-out tensors to select a loss/coefficient, tune heads, refit
readouts, retrain an old encoder or import pretrained state. Any next mechanism
needs an independent named clean source copy and prospective card, protecting
all scoped strengths already found. No next encoder version, dev selection,
commit/push or generalization claim follows from this report.

Read [results](CAUSAL_FRESH_V5_RESULTS.md), [detail](CAUSAL_FRESH_V5_DETAIL.md),
[comparison](results/causal_fresh_v5/comparison.json),
[quality audit](causal_fresh_v5/quality-independent-audit.json) and
[TRAIN audit](causal_fresh_v5/independent-training-audit.json). All source,
checkpoints, typed states, embeddings, fitted heads and failed attempts remain
protected. The exact frozen goal stays active.
