<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Saved TRAIN diagnosis: CPB-v3 and CPB-v4

16 October 2026. No new encoder experiment or head fit. This is a separately
declared arithmetic diagnosis of saved TRAIN artifacts, with no model execution,
decoder/neural forward, autograd, fitted normalizer or held-out tensor analysis.
CPB-v3.market-01 remains the clean baseline and dev selection; preserve v4's
scoped gains and all historical artifacts.

The [prospective card](CAUSAL_V3_V4_TRAIN_DIAGNOSIS_CARD.json) allows exactly41
inputs:37 TRAIN archives and four aggregate JSON files accessed only through
their TRAIN fields. Full-context features, targets and saved predictions cover112
TRAIN windows from the same MKT-HIST-2 population. Detailed masked-query cellwise
witnesses cover only the first8 windows, separately from all112 scalar query
summaries. Both saved initial/final stages of both versions are retained.

The original experiments each trained one encoder for2048 updates at B8 on
CUDA:0. Existing ridge1 and tanh16/Adam0.01/100-step heads are reused, including all
three neural repetitions2701/2802/2903. These are descriptive in-sample TRAIN
scores, not fresh quality measurements, generalization estimates or three
independent encoder runs. Full task rows and provenance are in the
[diagnosis](causal_v3_v4_train_diagnosis/diagnosis.json).

Dataset: **MKT-HIST-2** · past volatility · **complexity4/5** · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01 | 36 | 0.984327 | 0.934212 | 100 |
| CPB-v4.market-01 | 36 | 0.956503 | 0.937637 | 100 |

Dataset: **MKT-HIST-2** · past trend · **complexity4/5** · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01 | 36 | 0.610912 | 0.859802 | 100 |
| CPB-v4.market-01 | 36 | 0.951367 | 0.959067 | 100 |

Dataset: **MKT-HIST-2** · past activity · **complexity4/5** · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01 | 36 | 0.966995 | 0.958349 | 100 |
| CPB-v4.market-01 | 36 | 0.974613 | 0.968751 | 100 |

Dataset: **MKT-HIST-2** · past relation · **complexity4/5** · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01 | 36 | 0.714607 | 0.853676 | 100 |
| CPB-v4.market-01 | 36 | 0.754399 | 0.903724 | 100 |

Both exports contain learned20, literal spectral12 and learned change4. The
spectral block and all paired initial TRAIN features remain numerically exact.
V4 loses some linear volatility information on TRAIN, but activity improves.
Consequently these data do not support an activity-erasure explanation for its
known VALIDATION activity regression.

Change4's raw covariance participation ratio increases1.199→3.220. This is a
scale-dependent measure of variance concentration, not numerical rank or proof
of useful capacity. V3's change coordinates each correlate strongly with
volatility; v4 distributes them across more signals. The saved jointly fitted
ridge change contribution remains dominant for volatility, but its target
correlation falls0.9866→0.9665. Parent/change contribution correlation becomes
less negative,−0.668→−0.086, while their sum-to-marginal variance ratio moves
0.869→1.008. Harmful readout cancellation therefore has weak direct support.
These are components of the existing joint ridge, not newly fitted block heads.

The first8 fixed queries improve in MAE,0.27455→0.26901, although all112 worsen,
0.25905→0.27251. The eight detailed windows cannot establish the cause of the
population-wide reconstruction regression. Their v4 NLL/Huber1 change-gradient
cosine is+0.789 and NLL/MAE is+0.690: partial alignment, not systematic conflict.
Inverse-variance weighting creates unequal local pressure across features, but
that observation alone does not explain quality differences.

The hypothetical v4 variance-native gradient norm0.09374 was never applied:
the actual variance input is detached and its applied gradient is exactly zero.
Initial change gradients are zero because both decoder projections start at
zero. The local adjoints describe fixed saved states; they are not replayed
training trajectories or full parameter Jacobians.

The next hypothesis is to separate mean4 and uncertainty4 projections from the
same causal normalized bank, exporting native40. The existing mean path keeps
v4's NLL derivative; a distinct uncertainty path can learn from variance without
directing those gradients through mean4. This adds1080 learned values and changes
dimension-dependent head capacity/seeds. It is a new architecture experiment,
not proof that four extra coordinates are necessary or that decoder uncertainty
equals market volatility. Declare its source, head settings and all22 strongest
clean-reference protection conditions before testing. Unweighted mean supervision
remains a later alternative; independent parent/residual supervision has weaker
support in these saved data.

The frozen [search-goal protocol](clean_causal_encoder_search_goal.json) protects
both v3 and v4 rather than forgetting previous gains. No application change,
commit or push is part of this research continuation. The
[pre-execution preservation receipt](causal_v3_v4_train_diagnosis/preservation-before.json)
verifies1030 existing files and74 critical checkpoint paths. The
[independent audit](causal_v3_v4_train_diagnosis/independent-audit.json) passes154629
checks and all256 prespecified finite differences, with zero mismatches. All1030
existing files remain exact after arithmetic and audit execution.
