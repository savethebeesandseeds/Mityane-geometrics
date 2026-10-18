<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v3 saved TRAIN diagnosis

New arithmetic on existing saved data; no new encoder experiment or held-out
quality test. No model, encoder, decoder or neural-head forward was executed,
and no head/statistics fit or parameter update occurred. The independent audit
passed477404 checks. All513 before-diagnosis preserved files remain exact.

V3 stays the clean baseline. Scores below reuse its original fitted predictions
on112 TRAIN anchors from one market split, with3 head repetitions; they are
in-sample diagnostics, not independent encoder replications or forecasting evidence.
The saved cellwise reconstruction witnesses cover only the first8 TRAIN windows.

## Saved TRAIN head errors

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity 4/5** · TRAIN112, saved predictions; past_volatility.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01.init | 36 | 0.84830286 | 0.90617984 | 100.00 |
| CPB-v3.market-01 | 36 | 0.98432683 | 0.93421246 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity 4/5** · TRAIN112, saved predictions; past_trend.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01.init | 36 | 0.59732709 | 0.89630266 | 100.00 |
| CPB-v3.market-01 | 36 | 0.61091217 | 0.85980249 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity 4/5** · TRAIN112, saved predictions; past_activity.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01.init | 36 | 0.94623663 | 0.94966086 | 100.00 |
| CPB-v3.market-01 | 36 | 0.96699523 | 0.95834935 | 100.00 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity 4/5** · TRAIN112, saved predictions; past_relation.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v3.market-01.init | 36 | 0.65141514 | 0.83109054 | 100.00 |
| CPB-v3.market-01 | 36 | 0.71460716 | 0.85367619 | 100.00 |

Relation is already learnable in the saved TRAIN population (final0.7146 linear,
0.8537 neural mean R²). This does not establish generalization. It gives less
immediate support for adding task-shaped correlation features than for testing
the existing training pressure. Detailed physical MAE/RMSE and every repetition
remain in the linked JSON; no new fit was selected.

## Native block concentration

Dataset: **MKT-HIST-2** · past-window retention · **complexity 4/5** · TRAIN112, saved native features.
The table uses the raw-coordinate covariance participation ratio, a measure of
variance concentration. It depends on coordinate scale and is not numerical rank.

| Block | Width | Initial ratio | Final ratio |
| --- | ---: | ---: | ---: |
| learned20 | 20 | 2.718428 | 3.467925 |
| fixed_spectral12 | 12 | 6.855212 | 6.855212 |
| change4 | 4 | 3.659154 | 1.199182 |
| complete36 | 36 | 9.997528 | 4.136468 |

## Bounded variance-gradient finding

Dataset: **MKT-HIST-2** · masked historical reconstruction · **complexity 4/5** · first8 TRAIN query windows only.
Analytic scalar arithmetic from saved coefficients reproduces the Gaussian NLL
and its local derivative with respect to change4. The variance-path norm is
6.942428× the mean-path norm; flattened cosine is+0.101062, with negative dot
products in3/8 rows. This is not a total-parameter gradient comparison or proof
that variance gradients are universally harmful. Independent finite differences
agree to4.03e-12 maximum absolute discrepancy.

## Chosen next experiment

CPB-v4.market-01 changes only the variance head input gradient rule: stop
gradients into change4 from that head while preserving its conditional forward
capacity, trainable108+27 parameters and the live mean residual. Train all learned
values fresh under the same paired seeds, data/calibration recipe, native36 and
2048/B8 budget. Compare both saved stages with unchanged fixed heads.

The [prospective card](CAUSAL_FRESH_V4_CARD.md) freezes22 advancement conditions
against clean v3 and all trust/proof requirements before execution. V1/V2/RPB
weights and quality floors are excluded. V3 is preserved regardless of outcome.

Evidence: [diagnosis](causal_v3_train_diagnosis/diagnosis.json),
[independent audit](causal_v3_train_diagnosis/independent-audit.json),
[before preservation](causal_v3_train_diagnosis/preservation-before.json),
[diagnosis declaration](CAUSAL_V3_TRAIN_DIAGNOSIS_CARD.md).
