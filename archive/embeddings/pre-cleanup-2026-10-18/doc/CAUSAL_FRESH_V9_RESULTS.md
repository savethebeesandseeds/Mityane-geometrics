<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v9.market-01: 6/22, three new timing references

**CPB-v3.market-01 remains the clean baseline and sole dev encoder.** V9 completes one fresh2048/B8 CUDA trajectory and passes6/22 prospectively frozen strongest-V3–V8 conditions, with100% coverage. It adds three scoped TEMPO-5 references: synthetic-TRAIN-scaler deleted-view linear accuracy0.9557291666666666 and neural accuracy0.9361979166666666, and market-scaler intact-view linear accuracy0.9986979166666666. Synthetic-TRAIN-scaler intact linear accuracy0.9869791666666666 ties V6; the earlier reference remains. All eight protected market quality conditions fail. V9 is not an overall winner or an application replacement.

**Full trust and measured-candidate admission passed.** The independent training audit passes326469 checks:25555 core +61153 statistics +1177 saved state +117864 Gaussian arithmetic +69301 actual optimization components +51419 temporal learning. The independent quality audit passes13411 checks, with zero mismatches in both audits. The separate [final evidence join](causal_fresh_v9/final-establishment.json) binds these results, fresh ancestry, CUDA learning, typed resume, causal proofs and13 initial DATA/statistic pairings. Advancement and winner selection remain false. Frozen producer records retain their original pending-trust fields; the separate join supplies their completed admission.

V9 changes V8's temporal summary: each nonempty original-position bin receives a fixed half anchored-local and half shared whole-history hidden vector, before its existing eight export rows and bias are applied once. The32-row export tensor is one **freshly initialized shared tensor**, with unchanged shapes/registration/draw order; no trained V8/V6 weights or fitted statistics are loaded. Empty local bins zero the whole eight-coordinate slice after bias. Both temporal routes use the same owned kernel. Each slice belongs to the embedding at the current cutoff, and shared context uses valid states at or before that cutoff. Future values and future masks cannot affect that endpoint embedding.

Native40 remains20 learned content +12 fixed spectral +mean4 +uncertainty4. All294752 learned values across76 registered tensors start fresh, with3888 fixed values and298640 total. Seventy-five identifiable tensors show nonzero CUDA data-gradient/movement events;294751 is their tensor-numel capacity, **not an every-scalar learning proof**. Only the registered scalar pool-score softmax gauge is exempt and observed separately. Four export groups/four within-bin position contrasts and all three between-bin Hadamard directions learn. Observed movement includes AdamW decay; decay is not counted as data learning.

The retained V6 training recipe is unit Huber(delta1) mean loss plus detached-error Gaussian variance calibration, one global norm2 clip1 and one AdamW. This keeps future-residual and optimizer coupling. The fixed half blend changes routing, numerical scales and FLOPs while downstream global mixing remains; one fresh trajectory cannot establish a causal pooling explanation or generalization.

Before each table: MKT-HIST-2 measures retained past-window volatility/trend/activity/relation, not forecasting, from three BTC/ETH/SOL channel histories. Training calibration uses4096 TRAIN anchors; each readout run uses112 TRAIN and32 known chronological VALIDATION windows. TEMPO-5 uses three independent generator cohorts941101/941102/941103, each512 TRAIN examples from256 paired source groups and256 VALIDATION examples from128 different paired groups. Deleted views are transformations of those same validation groups. There is one newly trained encoder, batch8, CUDA:0,2048 completed updates/2048 attempts, zero skips. Three neural-head repetitions on one market checkpoint are head variation, not independent market-cohort uncertainty or three encoder trainings.

All initial0/final2048 embeddings and heads are new:1572 CUDA exports total. Fixed ridge1 and16-tanh/Adam0.01/100-step heads use repetitions2701/2802/2903 and the unchanged D40 seeds6049515622977958931,8262941649267994858,2095371932330426401. Market heads have164/724 fitted parameters, timing heads82/690. Each head's feature/target scales use its permitted TRAIN data. Synthetic-TRAIN input statistics are fresh per cohort and shared across the two checkpoints; market-scaler panels use this run's fresh TRAIN scaler. No synthetic encoder training, PCA after exports, tuning, checkpoint/seed selection or old encoder/head replay occurs. V3–V8 rows/costs are reused frozen evidence; V3/V4 native36 differ in capacity and width-derived head seeds from native40. Complexity4/5 and5/5 are ordinal designed recipe labels, not accuracy or measured task difficulty.

The following tables copy saved comparison rows. Physical errors, full-precision neural repetitions, all timing cohort/head repetitions, raw controls, objective components, proofs, costs and hashes are in [DETAIL](CAUSAL_FRESH_V9_DETAIL.md) and the immutable [table JSON](../output/runs/causal-fresh-v9/run-001/quality-001/report/table_rows.json).

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_volatility; V3/V4/V5/V6/V7/V8 final rows reused, V9 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673434 | 0.463051923 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.555176229 | 100 |
| CPB-v5.market-01 | 40 | 0.95221109 | 0.648190685 | 100 |
| CPB-v6.market-01 | 40 | 0.877120189 | 0.58463377 | 100 |
| CPB-v7.market-01 | 40 | 0.880509577 | 0.673304976 | 100 |
| CPB-v8.market-01 | 40 | 0.907581398 | 0.735969246 | 100 |
| CPB-v9.market-01.init | 40 | 0.262941581 | -0.363004992 | 100 |
| CPB-v9.market-01 | 40 | 0.90992222 | 0.618522958 | 100 |

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_trend; V3/V4/V5/V6/V7/V8 final rows reused, V9 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263132963 | -0.253979001 | 100 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.569875231 | 100 |
| CPB-v5.market-01 | 40 | 0.868645445 | 0.523928784 | 100 |
| CPB-v6.market-01 | 40 | 0.675860749 | 0.270702751 | 100 |
| CPB-v7.market-01 | 40 | 0.685042973 | 0.10961324 | 100 |
| CPB-v8.market-01 | 40 | 0.69630659 | -0.0374078453 | 100 |
| CPB-v9.market-01.init | 40 | -0.166912916 | -0.562154218 | 100 |
| CPB-v9.market-01 | 40 | 0.660819198 | -0.133673874 | 100 |

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_activity; V3/V4/V5/V6/V7/V8 final rows reused, V9 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840311 | 0.541426056 | 100 |
| CPB-v4.market-01 | 36 | 0.835143075 | 0.687370545 | 100 |
| CPB-v5.market-01 | 40 | 0.844256819 | 0.617983776 | 100 |
| CPB-v6.market-01 | 40 | 0.672769992 | 0.323179998 | 100 |
| CPB-v7.market-01 | 40 | 0.712390048 | 0.352638034 | 100 |
| CPB-v8.market-01 | 40 | 0.819083477 | 0.654256366 | 100 |
| CPB-v9.market-01.init | 40 | 0.477931176 | 0.541832486 | 100 |
| CPB-v9.market-01 | 40 | 0.83053287 | 0.603294851 | 100 |

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_relation; V3/V4/V5/V6/V7/V8 final rows reused, V9 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.00700444683 | -0.0855656715 | 100 |
| CPB-v4.market-01 | 36 | 0.326405067 | 0.209510464 | 100 |
| CPB-v5.market-01 | 40 | 0.252328556 | 0.370505348 | 100 |
| CPB-v6.market-01 | 40 | 0.622100687 | 0.322683609 | 100 |
| CPB-v7.market-01 | 40 | 0.600800572 | 0.324861744 | 100 |
| CPB-v8.market-01 | 40 | 0.504487017 | 0.0389512195 | 100 |
| CPB-v9.market-01.init | 40 | 0.295589255 | -0.0659490613 | 100 |
| CPB-v9.market-01 | 40 | 0.531691195 | 0.326550049 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact; V3/V4/V5/V6/V7/V8 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01 | 36 | 98.3072917 | 96.9618056 | 100 |
| CPB-v5.market-01 | 40 | 97.1354167 | 96.5711806 | 100 |
| CPB-v6.market-01 | 40 | 98.6979167 | 98.4809028 | 100 |
| CPB-v7.market-01 | 40 | 98.5677083 | 97.8732639 | 100 |
| CPB-v8.market-01 | 40 | 97.65625 | 97.2222222 | 100 |
| CPB-v9.market-01.init | 40 | 94.921875 | 94.2274306 | 100 |
| CPB-v9.market-01 | 40 | 98.6979167 | 97.8732639 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted; V3/V4/V5/V6/V7/V8 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.1458333 | 91.7534722 | 100 |
| CPB-v4.market-01 | 36 | 94.6614583 | 93.359375 | 100 |
| CPB-v5.market-01 | 40 | 92.96875 | 92.7951389 | 100 |
| CPB-v6.market-01 | 40 | 95.3125 | 93.4027778 | 100 |
| CPB-v7.market-01 | 40 | 93.359375 | 92.2309028 | 100 |
| CPB-v8.market-01 | 40 | 93.2291667 | 93.0121528 | 100 |
| CPB-v9.market-01.init | 40 | 89.0625 | 87.2829861 | 100 |
| CPB-v9.market-01 | 40 | 95.5729167 | 93.6197917 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact; V3/V4/V5/V6/V7/V8 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.3489583 | 98.1770833 | 100 |
| CPB-v4.market-01 | 36 | 99.3489583 | 98.828125 | 100 |
| CPB-v5.market-01 | 40 | 99.21875 | 98.8715278 | 100 |
| CPB-v6.market-01 | 40 | 99.7395833 | 99.21875 | 100 |
| CPB-v7.market-01 | 40 | 99.21875 | 98.4809028 | 100 |
| CPB-v8.market-01 | 40 | 99.609375 | 99.2621528 | 100 |
| CPB-v9.market-01.init | 40 | 94.6614583 | 94.1840278 | 100 |
| CPB-v9.market-01 | 40 | 99.8697917 | 98.4809028 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted; V3/V4/V5/V6/V7/V8 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.2708333 | 100 |
| CPB-v4.market-01 | 36 | 95.8333333 | 95.6163194 | 100 |
| CPB-v5.market-01 | 40 | 94.140625 | 94.0972222 | 100 |
| CPB-v6.market-01 | 40 | 97.265625 | 95.3125 | 100 |
| CPB-v7.market-01 | 40 | 94.53125 | 91.8836806 | 100 |
| CPB-v8.market-01 | 40 | 96.2239583 | 95.0520833 | 100 |
| CPB-v9.market-01.init | 40 | 88.5416667 | 84.4618056 | 100 |
| CPB-v9.market-01 | 40 | 93.359375 | 92.8385417 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.25904599 | 0.376414231 | 144.341116 (reused) |
| CPB-v4.market-01 | 2048 | 0.272508032 | 0.308316924 | 172.860055 (reused) |
| CPB-v5.market-01 | 2048 | 0.254463936 | 0.375196294 | 134.983334 (reused) |
| CPB-v6.market-01 | 2048 | 0.250172579 | 0.354519735 | 175.575555 (reused) |
| CPB-v7.market-01 | 2048 | 0.241805338 | 0.370420511 | 219.730088 (reused) |
| CPB-v8.market-01 | 2048 | 0.247146261 | 0.324973012 | 178.284957 (reused) |
| CPB-v9.market-01.init | 0 | 0.664258198 | 0.958212853 | 0 |
| CPB-v9.market-01 | 2048 | 0.244540347 | 0.326107515 | 147.328444 |

Neural trend remains weak: mean R²−0.1336738736969567, with individual values0.1203974960089037,−0.818075045484095,0.2966559283843212. Relation becomes positive for both heads, satisfying the two positivity conditions, but its linear/neural means remain below the protected V6/V5 floors. The four timing retention passes are three strict improvements and one tie; together with relation positivity they account for6/22. No reconstruction record is set. VALIDATION NLL0.47538792454983048 is worse than V8's0.424167868 and the required V4 ceiling; TRAIN NLL is descriptive and adds no gate.

Two reporting workflow failures remain explicit. The original quality wrapper failed after all five extraction/readout stages passed, at a report prose-contract guard. A separately declared external reporting-only repair generated the report and unchanged comparison once in0.926021336s; all531 original input files and118 scientific sources remained exact. Quality-audit001 then refused before numeric/archive checks because a durable host label was mistaken for a runtime path. Additive audit002 corrected only that selector and passed; the failed sources/cards/logs remain preserved. Neither repair repeated training, extraction or any head fit. [Recovery](causal_fresh_v9/reporting-recovery-receipt-001.json), [audit](causal_fresh_v9/quality-independent-audit.json), and [review](CAUSAL_FRESH_V9_REVIEW.md) retain that distinction.

Stop this exact fixed-half V9 recipe without blend/seed/rate/budget/head rescue. Preserve the three new timing references and every earlier scoped strength, checkpoint and result. Next is a **separately declared saved TRAIN-only V8/V9 diagnosis** before selecting any V10 mechanism. The research goal remains active; no next encoder, dev promotion, Git publication or new scientific execution is authorized by these reports.
