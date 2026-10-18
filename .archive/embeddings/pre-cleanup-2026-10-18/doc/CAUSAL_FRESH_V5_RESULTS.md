<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v5.market-01 results: 7/22, preserve the scoped gains

**CPB-v3.market-01 remains the clean baseline and sole dev encoder.** Newly
measured CPB-v5.market-01 passes 7/22 frozen strongest-clean-reference conditions,
complete trust admission and 100% scoring coverage. It adds scoped records in
neural volatility, neural relation, market-scaler intact neural timing and TRAIN
reconstruction MAE. It fails balanced advancement. Preserve V3, V4, V5 and every
checkpoint; stop this exact V5 recipe without coefficient, seed, rate, budget,
head or checkpoint rescue. The active clean search goal continues. No dev
replacement, commit or push follows.

This is one new encoder trajectory: all 76 learned tensors/294752 values start
fresh under seed 943001, with 3888 deterministic fixed-prior values, fresh
TRAIN-only input/bank statistics and optimizer, source_updates=0 and 2048 CUDA:0
updates at batch 8. Native width 40 is learned20 + fixed spectral12 + mean4 +
uncertainty4: 28 learned coordinates and 12 fixed-formula coordinates. Independent
270-to-4 bank projections feed the mean residual and live conditional variance
head separately. Gaussian NLL keeps its numerical formula and full gradient
through disjoint paths. Variance still weights mean errors through 1/variance.
Decoder uncertainty is not the market-volatility target.

MKT-HIST-2 uses 4096 TRAIN anchors, 112 TRAIN/32 known chronological VALIDATION
readout windows. TRAIN Jan 1–21, VALIDATION Jan 22–28; Jan 29–31 is unused. Fixed
heads retain ridge penalty 1 and 16 tanh hidden units / Adam 0.01 / 100 updates, repetitions 2701/2802/2903.
V5 D40 has 164/724 market and 82/690 timing head parameters, using width-derived
seeds 6049515622977958931/8262941649267994858/2095371932330426401. V3/V4 D36 had
148/660 and 74/626, with their original different seeds. V5 adds 1080 learned
encoder values and wider readouts: this is a capacity/representation experiment,
not a same-capacity isolation of one gradient effect.

TEMPO-5 reuses cohorts 941101/941102/941103 under both declared scaler policies
and intact/extra 30%-deleted views; zero synthetic encoder updates. Both new saved
V5 initial 0/final 2048 banks are extracted once and receive their own fixed-recipe
heads. V3/V4 final scores and costs are reused without encoder or head execution.
No post-encoder PCA, best-head or intermediate-checkpoint selection occurs.
Three head fits are not three encoder runs. These known-development tasks measure
retained past information, not forecasting or fresh chronological confirmation.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_volatility.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673 | 0.463052 | 100.00 |
| CPB-v4.market-01 | 36 | 0.849562 | 0.555176 | 100.00 |
| CPB-v5.market-01.init | 40 | 0.412200 | -0.118115 | 100.00 |
| CPB-v5.market-01 | 40 | 0.952211 | 0.648191 | 100.00 |

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_trend.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263133 | -0.253979 | 100.00 |
| CPB-v4.market-01 | 36 | 0.907213 | 0.569875 | 100.00 |
| CPB-v5.market-01.init | 40 | -1.031857 | -0.591848 | 100.00 |
| CPB-v5.market-01 | 40 | 0.868645 | 0.523929 | 100.00 |

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_activity.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840 | 0.541426 | 100.00 |
| CPB-v4.market-01 | 36 | 0.835143 | 0.687371 | 100.00 |
| CPB-v5.market-01.init | 40 | 0.110988 | 0.546274 | 100.00 |
| CPB-v5.market-01 | 40 | 0.844257 | 0.617984 | 100.00 |

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Task: past_relation.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.007004 | -0.085566 | 100.00 |
| CPB-v4.market-01 | 36 | 0.326405 | 0.209510 | 100.00 |
| CPB-v5.market-01.init | 40 | 0.087308 | -0.253444 | 100.00 |
| CPB-v5.market-01 | 40 | 0.252329 | 0.370505 | 100.00 |

## Timing: synthetic_train / validation_intact

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each 512 TRAIN/256 VALIDATION examples from 256/128 independent source groups. Policy: synthetic_train; view: validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.48 | 95.70 | 100.00 |
| CPB-v4.market-01 | 36 | 98.31 | 96.96 | 100.00 |
| CPB-v5.market-01.init | 40 | 94.27 | 93.84 | 100.00 |
| CPB-v5.market-01 | 40 | 97.14 | 96.57 | 100.00 |

## Timing: synthetic_train / validation_deleted

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each 512 TRAIN/256 VALIDATION examples from 256/128 independent source groups. Policy: synthetic_train; view: validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.15 | 91.75 | 100.00 |
| CPB-v4.market-01 | 36 | 94.66 | 93.36 | 100.00 |
| CPB-v5.market-01.init | 40 | 88.28 | 85.81 | 100.00 |
| CPB-v5.market-01 | 40 | 92.97 | 92.80 | 100.00 |

## Timing: market_scaler / validation_intact

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each 512 TRAIN/256 VALIDATION examples from 256/128 independent source groups. Policy: market_scaler; view: validation_intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.35 | 98.18 | 100.00 |
| CPB-v4.market-01 | 36 | 99.35 | 98.83 | 100.00 |
| CPB-v5.market-01.init | 40 | 94.40 | 93.79 | 100.00 |
| CPB-v5.market-01 | 40 | 99.22 | 98.87 | 100.00 |

## Timing: market_scaler / validation_deleted

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each 512 TRAIN/256 VALIDATION examples from 256/128 independent source groups. Policy: market_scaler; view: validation_deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.92 | 94.27 | 100.00 |
| CPB-v4.market-01 | 36 | 95.83 | 95.62 | 100.00 |
| CPB-v5.market-01.init | 40 | 86.72 | 84.07 | 100.00 |
| CPB-v5.market-01 | 40 | 94.14 | 94.10 | 100.00 |

## Reconstruction and training cost

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Fixed saved query masks; TRAIN 112/VALIDATION 32. Errors are hierarchical standardized mean MAE.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.259046 | 0.376414 | 144.341116 (reused) |
| CPB-v4.market-01 | 2048 | 0.272508 | 0.308317 | 172.860055 (reused) |
| CPB-v5.market-01.init | 0 | 0.664168 | 0.958065 | 0.000000 |
| CPB-v5.market-01 | 2048 | 0.254464 | 0.375196 | 134.983334 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute history, past-window retention · **complexity 4/5** · intact known chronological VALIDATION 32; heads fit on 112 TRAIN windows. Same fixed queries; Gaussian NLL omits its constant and valid negative values remain valid.

| Method | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- |
| CPB-v3.market-01 | -0.380178 | 1.521797 |
| CPB-v4.market-01 | -0.356693 | -0.045381 |
| CPB-v5.market-01.init | 0.423815 | 0.795377 |
| CPB-v5.market-01 | -0.324377 | 2.687450 |

GPU seconds are synchronized forward_at + F64 NLL + backward + clip + AdamW
wall time, including in-step contract checks; loader/masks, checkpoints, proofs,
export, head fitting and audits are separate. V3/V4 times are old measurements,
not reruns or a controlled speed benchmark.

## Exact frozen advancement decision

The seven passes are neural volatility, neural relation, market-scaler intact
neural timing, TRAIN MAE, all 3 positive neural trend fits, and both positive
relation heads. The other 15 conditions fail. Category counts are market 2/8,
TEMPO-5 retention 1/8, fixed-query MAE 1/2, validation NLL 0/1, trend robustness 1/1
and relation positivity 2/2. Read all 22 exact rules and physical errors in
[the detail](CAUSAL_FRESH_V5_DETAIL.md).

V5's 7/22 uses the stronger best-V3/V4 goal. V4's historical 19/22 used its own
previous floors; these counts are not a comparable ranking. New neural records
and lower TRAIN MAE do not cancel losses in other tasks, timing retention or
validation reconstruction/calibration.

## Verification and preservation

Engineering passes 213674 checks and 38 typed refusals. Both actual saved stages
pass 128 cutoffs/2350 future-value-or-mask cases/30 internal temporal cases, with
frozen state. The dedicated nonzero CUDA mechanism fixture passes 31 checks,
including disjoint native gradients, uncertainty-projection learning, manual
full-NLL gradient parity and one synthetic optimizer step. Independent TRAIN
audit passes 192589 checks; actual resume 512 continues the same trajectory.
Independent saved-quality audit passes 11300 checks with zero mismatches and 339
artifact bindings. All 13 saved initial prefix/statistic pairings are exact;
the added uncertainty4 is finite/nonzero. This pairing is not initial-score reuse.

The frozen [card](CAUSAL_FRESH_V5_CARD.md), [goal](clean_causal_encoder_search_goal.json),
[machine comparison](results/causal_fresh_v5/comparison.json),
[detailed saved rows](results/causal_fresh_v5/table_rows.json),
[TRAIN audit](causal_fresh_v5/independent-training-audit.json) and
[quality audit](causal_fresh_v5/quality-independent-audit.json) bind the evidence.
Failed build attempts and successful admitted attempt 003 remain preserved.
Checkpoints, typed states, embeddings, heads, source/data and every older result
remain local and protected. No model/head run was repeated to write these docs.

## Next direction

Keep V3 selected and preserve V4/V5 scoped records. Next propose a separately
declared saved TRAIN-only V4/V5 comparison of mean4/uncertainty4, existing fitted
linear contributions and variance-weighted reconstruction pressure. This can
inform whether independent mean supervision (original option A) or another
mechanism is justified. No next version is declared here. Do not use held-out
tensors to choose a loss/coefficient, refit heads or replay either encoder.
The goal continues; no automatic application replacement or publication follows.
Read [the review](CAUSAL_FRESH_V5_REVIEW.md).
