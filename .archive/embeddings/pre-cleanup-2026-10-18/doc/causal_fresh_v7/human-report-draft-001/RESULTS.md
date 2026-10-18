<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v7.market-01 results: 4/22, retain neural volatility and TRAIN reconstruction

**CPB-v3.market-01 remains the clean baseline and sole dev encoder.** V7 passes
4/22 prospectively strengthened V3/V4/V5/V6 retention conditions with100% coverage.
It sets a new neural-volatility reference and improves TRAIN fixed-query MAE/NLL.
It fails18 protected conditions, including all eight TEMPO-5 floors, both trend
and activity floors, validation reconstruction/calibration, and all-three neural
trend positivity. It is a scoped reference, not an overall winner or dev replacement.

**Provisional human draft: independent training and quality audits passed; the
separate root establishment/mirror binding is pending.** No final report freeze
or live metadata application follows from this draft. The frozen producers keep
their original pending-trust fields; later independent receipts are additive.
Training audit passes283252 checks (33566 core +61153 statistics +1346 saved-state
+117864 Gaussian +69323 actual components), and quality audit passes12493, both
with zero mismatches. All13 full40 initial DATA/statistic pairings versus V6 are
exact. These pairings verify fresh paired data/statistics, not pretrained ancestry.

V7 changes only clipping relative to V6: separate disjoint mean and uncertainty
parameter groups each receive norm2 max1 clipping before one unchanged AdamW.
The forward architecture, native40, Huber1 mean plus detached-error Gaussian
variance objective, constructor draws, data, initialization943001,2048/B8 budget
and heads remain fixed. Native40 is20 learned content +12 literal spectral
+4 learned mean +4 learned uncertainty coordinates. All294752 learned values
start fresh across76 tensors; each tensor receives nonzero CUDA gradients and
changes. The deterministic3888-value prior remains fixed,77 tensors/298640 total.
Fresh input/bank TRAIN statistics and optimizer state use no old model/stat import.

Mean parameters are73 tensors/293537 values; uncertainty parameters are3/1215.
Each branch cap is1, allowing combined norm up tosqrt2 rather than the former
shared cap1. One AdamW and changing future residuals retain other coupling.
This capacity of the update budget is a confound: this experiment cannot prove
that cross-branch attenuation caused any earlier failure. The audit observes
mean clipping on0/2048 updates and uncertainty clipping on1313/2048; it checks
recorded factors/coverage, without independently reconstructing full gradients.

One fresh CUDA:0 trajectory completes2048 successful updates at batch8 without
skips, including actual typed resume at512. MKT-HIST-2 uses4096 TRAIN anchors and
112 TRAIN/32 known chronological VALIDATION readout windows: TRAIN Jan1–21,
VALIDATION Jan22–28, Jan29–31 unused. Targets are past-window retained signals,
not forecasts. Fixed heads are ridge penalty1 and tanh16/Adam0.01/100 updates,
repetitions2701/2802/2903. D40 seeds remain6049515622977958931,
8262941649267994858,2095371932330426401; market164/724 and TEMPO82/690 fitted head
parameters. Native36 V3/V4 use148/660 and74/626 with their declared different
width-derived seeds. V7 versus V6 is same-capacity; V3/V4 comparisons also carry
encoder/readout width and seed confounds.

TEMPO-5 uses saved cohorts941101/941102/941103, each512 TRAIN/256 VALIDATION from
256/128 source groups, both TRAIN-fitted synthetic and saved market scaler
policies, intact and extra30% coordinate-deletion views. V7 initial0/final2048
embeddings and fixed heads are newly measured. V3/V4/V5/V6 final scores/costs
are reused unchanged. There is no initial score reuse, post-encoder PCA, best
head/intermediate checkpoint selection, repeated old encoder training or head
rescue. Three head fits are not three encoder trajectories; known development
populations do not establish fresh-period generalization.
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
The four passed conditions are neural volatility retention, TRAIN MAE, and the
two relation-positivity conditions. TRAIN Gaussian NLL is a descriptive record,
not an added advancement gate. V7's trend neural repetitions are0.239513425,
-0.194818323 and0.284144619. The protected best linear relation and neural
relation remain V6 and V5 respectively. Preserve all V3–V6 timing, trend,
activity, volatility and validation-reconstruction strengths.

The GPU compute-only cumulative time is219.730088148s; trajectory wall time is
272.302167537s and wrapper time281.478890819s. The new quality wrapper is79.922s,
separate from training. Costs are descriptive single runs, not a repeated speed
benchmark. Initial/final proof runs are separate zero-update engineering work.

Stop this exact disjoint-clipping recipe without cap/rate/seed/coefficient/head
rescue. Next is a separately declared saved TRAIN-only V6/V7 diagnosis, then a
review of a broader causal temporal-summary architecture. No next model is
selected or running. The stronger22-condition goal remains active.

See [detail](CAUSAL_FRESH_V7_DETAIL.md), [review](CAUSAL_FRESH_V7_REVIEW.md),
[the frozen prospective card](CAUSAL_FRESH_V7_CARD.md),
[comparison](results/causal_fresh_v7/comparison.json) and
[the preserved ledger](clean_causal_encoder_research_ledger.json).
