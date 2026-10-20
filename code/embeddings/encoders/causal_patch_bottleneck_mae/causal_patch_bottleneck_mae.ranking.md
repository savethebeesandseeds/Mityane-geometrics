# Causal patch bottleneck MAE encoder ranking

Last evidence review: **19 October 2026**. This is the maintained CPB leaderboard for measured checkpoint quality. Work states remain in [Hero 17](../../../../web/dashboard/heroes/17-encoder-cpb.json); [README](README.md#active-ranking-and-agent-instructions) defines how agents must maintain this file.

**Leading broad clean-causal reference: CPB-v4.market-01, native36.** Its original comparison delivered 19/22 conditions, all eight hard-timing checks, large trend/relation gains and the best saved known-validation reconstruction. Its remaining failures and lack of formal promotion do not erase that leading role. **CPB-v3.market-01 remains the clean baseline.**

The January quantitative rankings below include 12 completed historical instances, V3–V14, plus all six fresh trajectories in the completed V4/V15 study, trained on January only. The separate February confirmation is complete with a **mixed prospective screen verdict**. Scoped records remain visible alongside their former leaders; they do not select a seed or replace the broad reference automatically.

## Research standing

| Standing | Encoder | Reason and limit |
| --- | --- | --- |
| Current broad clean lead | [CPB-v4.market-01][v4] | 19/22 original conditions, all eight original timing protections, trend/relation breakthrough and known-validation reconstruction leader. Linear volatility/activity and TRAIN MAE failed their original floors. |
| Stable clean baseline | [CPB-v3.market-01][v3] | Fresh causal-training baseline; retains the highest observed linear volatility R². Baseline is not the broad quality ranking. |
| Specialist references | [V6][v6], [V8][v8], [V11][v11], [V13][v13] | Retain relation, volatility, activity and timing strengths in the task rankings below. Earlier leaders remain recorded when a new observation exceeds them. |
| Latest catalogued closed instance | [CPB-v14.market-01][v14] | Full trust/coverage; 3/22 strengthened conditions and no new protected record in that historical run. Its catalog position is not a quality rank. |
| Completed controlled study | V4 controls and V15, seeds 945101/945102/945103 | All six fresh native36 final2048 trajectories and twelve arithmetic audits passed. February's declared screen is mixed: activity and reconstruction gains coexist with volatility/trend losses and failed timing protections. |

V4's broad-lead position is the retained scientific assessment supported by its [review](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V4_REVIEW.md), rather than an invented weighted score. Historical 19/22 and later 3/22 counts used different floors and must not be sorted as one metric. Ranking positions confer no Mityane application selection or automatic promotion.

## How to read the quantitative ranks

V3 through V14 abbreviate the exact `CPB-vN.market-01` instance, never every checkpoint of that version. `V4 control 945101` and similar labels identify new control-arm checkpoints in `cpb-v15-objective-001`; they are distinct from `CPB-v4.market-01`. `V15 seed 945102` identifies that study's candidate arm. Every new run is listed in the [identity register](#study-run-identities).

Columns 1–3 are **distinct score groups**, using dense ranks: scores within absolute **1e-12** share a group, and every tied member is retained. This reporting convention avoids ordering floating-point averaging noise. It changes no frozen acceptance predicate. Values are displayed to six decimals; ranking uses unrounded values, and the linked JSON retains individual scores. Higher R²/accuracy is better; lower MAE/NLL is better.

V3/V4 and the new study have native36; V5–V14 have native40. Width changes readout capacity and width-derived head seeds. These are observed checkpoint rankings, not controlled claims that one architecture causes a gain. Historical versions each have one encoder training trajectory; three head repetitions are not three encoder retrainings.

## Market readout ranking

Scope: **OR-05 past-window readout**, MKT-HIST-2, BTC/ETH/SOL January2024, TRAIN days1–21 and known validation days22–28. Encoders use recent128 history from the saved H256 probe geometry. Heads use 112 TRAIN and 32 validation probes, TRAIN-fitted preprocessing, ridge1 or four-output tanh16/Adam0.01/100 steps. All ranked rows have 32/32 coverage. Neural R² is the arithmetic mean of the three fixed head R² values, not pooled predictions. January is repeatedly used development evidence.

| Task / head | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Volatility / linear | [V3][v3] · **0.964673** | [V5][v5] · **0.952211** | [V10][v10] · **0.935675** |
| Volatility / neural mean | [V8][v8] · **0.735969** | [V7][v7] · **0.673305** | [V5][v5] · **0.648191** |
| Trend / linear | [V4][v4] · **0.907213** | [V10][v10] · **0.897085** | [V5][v5] · **0.868645** |
| Trend / neural mean | [V4][v4] · **0.569875** | [V10][v10] · **0.548230** | [V5][v5] · **0.523929** |
| Activity / linear | [V11][v11] · **0.909439** | [V15 seed 945103][n945103] · **0.906534** | [V14][v14] · **0.887832** |
| Activity / neural mean | [V4 control 945101][c945101] · **0.792420** | [V13][v13] · **0.762544** | [V4 control 945102][c945102] · **0.756138** |
| Relation / linear | [V6][v6] · **0.622101** | [V7][v7] · **0.600801** | [V14][v14] · **0.589564** |
| Relation / neural mean | [V13][v13] · **0.457197** | [V11][v11] · **0.448775** | [V14][v14] · **0.419915** |

The new activity-neural high is **0.7924196342764943**, from fresh V4 control945101, against the prior closed-study V13 value **0.762543630810775**. Its three saved head values are 0.8569086791075013, 0.7904946597000803 and 0.7298555640219014. This is a new checkpoint observation from the V15 study's control arm, not a V15-candidate gain.

## Fixed query reconstruction ranking

Scope: **OR-09 fixed-query decoding**, the unchanged saved masked MKT-HIST-2 query, final checkpoint2048. TRAIN contains112 windows and known validation32. The two splits are ranked separately. Gaussian NLL is a reported measurement; V15 trains a different composite objective. TRAIN NLL remains descriptive and adds no protection gate.

| Population / metric | 1 | 2 | 3 |
| --- | --- | --- | --- |
| TRAIN / MAE | [V15 seed 945102][n945102] · **0.240037** | [V7][v7] · **0.241805** | [V9][v9] · **0.244540** |
| TRAIN / Gaussian NLL | [V4 control 945101][c945101] · **-0.410369** | [V7][v7] · **-0.408827** | [V6][v6] · **-0.400207** |
| Known validation / MAE | [V4][v4] · **0.308317** | [V12][v12] · **0.315911** | [V11][v11] · **0.320134** |
| Known validation / Gaussian NLL | [V4][v4] · **-0.045381** | [V12][v12] · **0.297064** | [V11][v11] · **0.312349** |

The new TRAIN MAE is **0.24003670546837294** for V15 seed945102, below V7's **0.24180533789857825**. Fresh V4 control945101 has TRAIN NLL **-0.41036896330440875**, below V7's **-0.40882727825899934**. These training-population records do not establish fresh-period superiority. V4 retains both known-validation reconstruction records.

## TEMPO timing ranking

Scope: **OR-05 slow lead/lag readout**, TEMPO-5 complexity5/5, masters941101/941102/941103, 512 TRAIN and256 validation rows per master. Both input scalers are TRAIN-fitted; the deleted validation view applies the saved additional30% deletion. Scores average three master means, each averaging the three fixed head repetitions; all ranked rows have full coverage. Input scaler, view and head define separate rankings. This measures synthetic transfer, separately from market quality and fresh-month generalization.

| Input scaler / validation view / head | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Synthetic TRAIN / intact / linear | [V15 seed 945101][n945101] · **99.609375%** | [V15 seed 945103][n945103] · **99.348958%** | [V13][v13] · **98.958333%** |
| Synthetic TRAIN / intact / neural | [V6][v6] · **98.480903%** | [V15 seed 945102][n945102], [V15 seed 945103][n945103], [V11][v11] · **98.263889%** | [V13][v13] · **98.220486%** |
| Synthetic TRAIN / deleted / linear | [V13][v13] · **95.833333%** | [V9][v9], [V15 seed 945103][n945103] · **95.572917%** | [V6][v6] · **95.312500%** |
| Synthetic TRAIN / deleted / neural | [V4 control 945101][c945101] · **94.835069%** | [V13][v13] · **94.574653%** | [V15 seed 945103][n945103] · **94.401042%** |
| Market TRAIN / intact / linear | [V9][v9] · **99.869792%** | [V6][v6], [V11][v11], [V12][v12], [V14][v14], [V15 seed 945103][n945103] · **99.739583%** | [V8][v8], [V13][v13], [V15 seed 945101][n945101], [V4 control 945102][c945102] · **99.609375%** |
| Market TRAIN / intact / neural | [V4 control 945102][c945102] · **99.305556%** | [V8][v8] · **99.262153%** | [V6][v6] · **99.218750%** |
| Market TRAIN / deleted / linear | [V6][v6] · **97.265625%** | [V15 seed 945101][n945101] · **96.484375%** | [V15 seed 945103][n945103] · **96.354167%** |
| Market TRAIN / deleted / neural | [V4][v4] · **95.616319%** | [V15 seed 945103][n945103] · **95.486111%** | [V6][v6] · **95.312500%** |

These new timing highs coexist with failures against other strongest timing protections. The completed study reports every run's eight original flags and the separate prospective February screen below.

## All six January study trajectories

All runs use native36, 2048 successful updates and fresh weights, optimizer and TRAIN statistics. Each encoder-seed pair holds the architecture and declared initialization/data/mask/context seeds fixed while changing the objective. Pairing receipts record identical row/mask-seed and anchor CSV fields and matching initialization FNV summaries; those metadata checks do not independently compare every parameter bit or context-mask waveform. All six January results remain visible so a record-setting seed does not stand in for the entire study. Individual neural repetitions, masks, row IDs and audited support remain in the linked reports.

### Linear R²

| Run | Volatility | Trend | Activity | Relation |
| --- | --- | --- | --- | --- |
| [V4 control 945101][c945101] | 0.762364 | 0.732087 | 0.874762 | 0.430174 |
| [V15 seed 945101][n945101] | 0.536357 | 0.619406 | 0.855497 | 0.469471 |
| [V4 control 945102][c945102] | 0.749644 | 0.814244 | 0.840677 | 0.494371 |
| [V15 seed 945102][n945102] | 0.626864 | 0.571334 | 0.834800 | 0.489265 |
| [V4 control 945103][c945103] | 0.687369 | 0.802083 | 0.795132 | 0.164000 |
| [V15 seed 945103][n945103] | 0.690095 | 0.757609 | 0.906534 | 0.060133 |

### Neural mean R²

| Run | Volatility | Trend | Activity | Relation |
| --- | --- | --- | --- | --- |
| [V4 control 945101][c945101] | 0.445879 | 0.099855 | 0.792420 | 0.088102 |
| [V15 seed 945101][n945101] | 0.411758 | -0.268647 | 0.720172 | -0.066863 |
| [V4 control 945102][c945102] | 0.635973 | 0.414591 | 0.756138 | 0.064325 |
| [V15 seed 945102][n945102] | 0.236982 | 0.070592 | 0.571972 | 0.186924 |
| [V4 control 945103][c945103] | 0.322828 | 0.238552 | 0.629837 | -0.095595 |
| [V15 seed 945103][n945103] | 0.468324 | -0.112153 | 0.632892 | -0.061140 |

### Reconstruction

| Run | TRAIN MAE | TRAIN NLL | Known-VAL MAE | Known-VAL NLL |
| --- | --- | --- | --- | --- |
| [V4 control 945101][c945101] | 0.247850 | -0.410369 | 0.399577 | 1.278563 |
| [V15 seed 945101][n945101] | 0.245746 | -0.385437 | 0.354997 | 0.508550 |
| [V4 control 945102][c945102] | 0.264637 | -0.365964 | 0.383207 | 0.755391 |
| [V15 seed 945102][n945102] | 0.240037 | -0.367805 | 0.375532 | 0.429499 |
| [V4 control 945103][c945103] | 0.258744 | -0.386967 | 0.475570 | 1.103325 |
| [V15 seed 945103][n945103] | 0.248416 | -0.390166 | 0.393124 | 0.732632 |

V15 lowers paired TRAIN/known-validation MAE and known-validation NLL in all three pairs. January linear trend R² falls in all three pairs. Its third pair improves linear activity from **0.7951324467818991** to **0.9065340022031494**. These gains and regressions remain part of the record together.

## February confirmation and broader replacement

**Work is complete: six of six trajectories and all twelve independent arithmetic audits passed. The saved prospective screen verdict is `mixed`; `promising` is false.** Activity linear R² improves in all three candidate/control pairs, and descriptive fixed-query MAE and NLL fall in all three pairs. Volatility and trend linear R² improve in one pair each; the volatility median, trend tolerance and candidate timing protections fail the declared screen. These scoped gains and losses do not replace the broad historical V4 reference or select a seed.

Scope: a separate **32-probe calendar-selected February2024 population**, opened after the [six-run development seal](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/development-seal.json), with the original January TRAIN scalers and fitted heads reused. Every task/head has32/32 support in all six runs. Neural values below are means of the same three fixed head repetitions. No February fit, tuning or encoder update occurs. January and February remain separate populations; values below are within-study comparisons, not a cross-month family ranking.

### February paired linear R² differences

Each value is the report's saved **candidate minus its same-seed control** difference. Relation is descriptive and does not enter the prospective screen.

| Encoder seed | Volatility Δ | Trend Δ | Activity Δ | Relation Δ, nongating |
| --- | --- | --- | --- | --- |
| 945101 | -0.203014 | +0.007876 | +0.228295 | -1.862306 |
| 945102 | +0.105940 | -0.218994 | +0.186530 | +0.548564 |
| 945103 | -0.078143 | -0.098598 | +0.122493 | -3.178873 |

The following median, range and positive-pair counts are copied from the saved screen, without adding a relation gate or changing its rules.

| Screen target | Saved median Δ | Saved range Δ | Improving pairs |
| --- | --- | --- | --- |
| Volatility | -0.078143 | [-0.203014, +0.105940] | 1/3 |
| Activity | +0.186530 | [+0.122493, +0.228295] | 3/3 |
| Trend | -0.098598 | [-0.218994, +0.007876] | 1/3 |

The primary rule requires positive medians and at least2/3 improving pairs for **both** volatility and activity. Trend requires a median difference at least-0.01. Every candidate must also retain all eight strongest historical timing conditions. The report gives these exact failure reasons:

- `past_volatility requires positive median and at least two improving pairs`
- `February linear trend median difference is below -0.01 or undefined`
- `at least one candidate fails a frozen hard timing condition`

### February absolute linear R²

| Run | Volatility | Trend | Activity | Relation |
| --- | --- | --- | --- | --- |
| [V4 control 945101][c945101] | 0.699687 | 0.290588 | 0.589823 | -1.108819 |
| [V15 seed 945101][n945101] | 0.496673 | 0.298464 | 0.818118 | -2.971125 |
| [V4 control 945102][c945102] | 0.652745 | 0.346180 | 0.655552 | -1.874316 |
| [V15 seed 945102][n945102] | 0.758685 | 0.127186 | 0.842081 | -1.325752 |
| [V4 control 945103][c945103] | 0.790686 | 0.491474 | 0.400313 | -1.248019 |
| [V15 seed 945103][n945103] | 0.712543 | 0.392876 | 0.522806 | -4.426892 |

### February absolute neural mean R²

| Run | Volatility | Trend | Activity | Relation |
| --- | --- | --- | --- | --- |
| [V4 control 945101][c945101] | 0.218620 | -0.147026 | 0.350338 | -1.115418 |
| [V15 seed 945101][n945101] | 0.098630 | -0.436559 | 0.525858 | -0.913773 |
| [V4 control 945102][c945102] | 0.246331 | -0.438837 | 0.555476 | -2.553742 |
| [V15 seed 945102][n945102] | -0.048022 | -0.645956 | 0.635421 | -0.951631 |
| [V4 control 945103][c945103] | 0.308937 | 0.013534 | 0.249503 | -0.637141 |
| [V15 seed 945103][n945103] | 0.534598 | -1.205357 | 0.396074 | -1.543608 |

### February fixed query reconstruction and passed evidence

These descriptive measurements add no protection gate or checkpoint selection. All32 waveform examples are independently audited for every run. Individual neural head values and the full January/final/February score tables remain in [human results](../../evaluation/protocols/cpb_v15_objective_v1/RESULTS.md) and the full-precision [machine report](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-001.json).

| Run | MAE | Gaussian NLL | Passed February evidence |
| --- | --- | --- | --- |
| [V4 control 945101][c945101] | 0.802806 | 9.337928 | [confirmation](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/confirmation-audit.json) |
| [V15 seed 945101][n945101] | 0.714172 | 5.154861 | [confirmation](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/confirmation-audit.json) |
| [V4 control 945102][c945102] | 0.835078 | 10.078783 | [confirmation](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/confirmation-audit.json) |
| [V15 seed 945102][n945102] | 0.723069 | 4.969757 | [confirmation](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/confirmation-audit.json) |
| [V4 control 945103][c945103] | 0.903748 | 8.747334 | [confirmation](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/confirmation-audit.json) |
| [V15 seed 945103][n945103] | 0.833747 | 6.845502 | [confirmation](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/confirmation-audit.json) |

### Final January historical protections

All22 criterion values are defined and supported for each run. Counts below use the unchanged strengthened historical floors; they are not comparable to V4's original19/22 count. **None of the six new trajectories passes all eight timing protections.** P/F flags follow the eight TEMPO table rows above in order: synthetic/intact linear, synthetic/intact neural, synthetic/deleted linear, synthetic/deleted neural, market/intact linear, market/intact neural, market/deleted linear, market/deleted neural. The complete132 criterion objects, thresholds, values and flags are retained in the report and human results.

| Run | Passed historical conditions | Eight timing flags, P=pass/F=fail |
| --- | --- | --- |
| [V4 control 945101][c945101] | 4/22 | F F F P F F F F |
| [V15 seed 945101][n945101] | 2/22 | P F F F F F F F |
| [V4 control 945102][c945102] | 4/22 | F F F F F P F F |
| [V15 seed 945102][n945102] | 3/22 | F F F F F F F F |
| [V4 control 945103][c945103] | 1/22 | F F F F F F F F |
| [V15 seed 945103][n945103] | 2/22 | P F F F F F F F |

February has now been consumed by this study and is observed evidence for future recipe choices. Three encoder pairs on one additional month establish a screen, not general-purpose superiority. Head repetitions are not encoder or population replications. No rank establishes forecasting, intervention effects or downstream decision utility.

The authoritative [machine report](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-001.json) is `complete`/`passed:true` (2463957bytes; SHA256 `cfebbb5fddb53008a0a460c7cc5f0f545897ef5808720cbd049e8b08d7cbf9ae`). Its [actual input card](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-inputs-001.json) binds all eighteen phase receipts and twelve audits; [human results](../../evaluation/protocols/cpb_v15_objective_v1/RESULTS.md), the [compact study record](../../evaluation/records/CPB-v15.objective-001.json) and the [frozen scientific protocol](../../evaluation/protocols/cpb_v15_objective_v1/PROTOCOL.md) provide the reviewable closeout.

Reporting-generation history is explicit: the original report derivation was stopped before writing a report artifact, with its source and logs preserved in the [transition record](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-transition-001.json). A separately [reviewed source generation](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/source-review-report-002.json) coalesces identical full-pin checks only inside the final verification barrier; each retained identity still receives the unchanged fresh physical-file checks. Scientific calculations are preserved. This report switch performed no training, head fitting or independent-audit replay.

## Study run identities

Every row below points to passed training/development and independent arithmetic evidence; the February reconstruction table above links each run's passed confirmation receipt and audit. The checkpoint tuple is copied from the actual passed training receipt. Raw checkpoints and datasets remain ignored local artifacts. Each development receipt binds the three TEMPO master reports as well as the market outputs; the query summary binds all112/32 saved row aggregates. January independent waveform witnesses cover the first8 examples per population; full means are checked from saved row metrics. Model/optimizer trajectories are not independently reimplemented by those arithmetic audits.

| Run | Final checkpoint bytes / SHA256 | Saved scores | Passed evidence |
| --- | --- | --- | --- |
| [V4 control 945101][c945101] | 1291409 / `b3201a1aa25236b9a69e416851d658c4564bcf0ae0335a93445b57eea672eea1` | [market](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/independent-audit.json) |
| [V15 seed 945101][n945101] | 1295937 / `64f13a7701ef75c3b0cd7857ddddeea37c5808338564664ba800a216e565f3ef` | [market](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/independent-audit.json) |
| [V4 control 945102][c945102] | 1291409 / `850a4e5b9a0fc292a51ef47718343a02833b1b36a46863069804fded9f36c206` | [market](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/independent-audit.json) |
| [V15 seed 945102][n945102] | 1295937 / `c6c3c5cbe1c91e004371ad814abbaa9c857f5e18cdc962de3427dc1459eb8b5e` | [market](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/independent-audit.json) |
| [V4 control 945103][c945103] | 1291409 / `5908ec3e790dc8add93f73b71dcb6a6e3cfe1f68e0ee461a4b2644035e0b7408` | [market](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/independent-audit.json) |
| [V15 seed 945103][n945103] | 1295937 / `68ce3f91b7509937548fc9e34c2cd77bf5b87074201bcdf95075822fef9f9ba9` | [market](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/independent-audit.json) |

The [immutable study card](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/run-inputs-001.json) binds source, binaries, environment, input datasets, scalers, protocol and all seed/arm roles. Reps2701/2802/2903 use width36 seeds13088411117524848790,10018485049408248344,9083751155097190455. The [historical V4/control comparison](../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/historical-v4-control-recipe-comparison-002.json) found recipe/source consistency with different declared seed families; it does not prove cross-build bit equality or attribute the historical score gap to initialization alone.

The broad historical V4 reference is separately identified by its [instance metadata][v4]: native36, initialization seed943001, final2048, 1291217bytes, SHA256 `ed1b2611f10f3cb0e14b5636e9fd07d6dbfbbce4ac9ed4769b8e98e792676fad`.

## Historical ancestry and cross-family context

V3–V14 and all six new trajectories are fresh-weight clean-causal research; source branching is not trained-weight inheritance. CPB V1/V2 retain pretrained ancestry and stay outside these clean score tables. Across the earlier RPB family, **RPB-v24.market-01.alt-01 native36** remains the historical balanced leader:8/9 market conditions plus all four primary TEMPO-5 transfer comparisons. RPB-v30 later passed9/9 market and8/8 strict native36 conditions with transfer pending. Their later causal-eligibility diagnosis is preserved, so their historical achievements are not folded into the clean CPB ranking. See the [reviewed handoff addendum](../../../../doc/EMBEDDINGS_HANDOFF_ADDENDUM_2026-10-18.md).

## Ranking changes

**19 October2026 — initial active ranking.** V4's broad-lead role and V3 baseline remain. The following six January score leaders were first registered after the development measurements passed, while the full study was still incomplete. They remain recorded after study completion, alongside each displaced leader and its original evidence.

| Scope | Previous closed-study leader | Latest observed leader |
| --- | --- | --- |
| Activity / neural mean R² | [V13][v13] · 0.762544 | [V4 control 945101][c945101] · **0.792420** |
| TRAIN / MAE | [V7][v7] · 0.241805 | [V15 seed 945102][n945102] · **0.240037** |
| TRAIN / NLL | [V7][v7] · -0.408827 | [V4 control 945101][c945101] · **-0.410369** |
| TEMPO synthetic_train / validation_intact / linear | [V13][v13] · 98.958333% | [V15 seed 945101][n945101] · **99.609375%** |
| TEMPO synthetic_train / validation_deleted / neural | [V13][v13] · 94.574653% | [V4 control 945101][c945101] · **94.835069%** |
| TEMPO market_scaler / validation_intact / neural | [V8][v8] · 99.262153% | [V4 control 945102][c945102] · **99.305556%** |

**19 October2026 — completed-study finalization.** The authoritative final-unique report closes all six trajectories and twelve arithmetic audits with the prospective verdict **mixed**, while retaining all six January highs above. The separate February section now records every pair, all six absolute readout/query results and every timing flag. February is consumed; V4's historical broad-reference role and V3's baseline role remain, with no seed/checkpoint selection, automatic promotion, catalog application or Mityane selection.

Future entries must record date, former/new leader, exact comparison scope, evidence links and the reason for the change. Retractions and invalidated eligibility need an explicit entry rather than silently deleting a result. Frozen historical protection floors, measured-instance catalogs and Mityane selection are independent records.

## Historical evidence and precision

- [V14 full-precision comparison](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/results/causal_fresh_v14/comparison.json): all12 completed final instances,32/32 market coverage, all four TEMPO panels and fixed-query split aggregates. 665850bytes; SHA256 `08ab7cd076b10de3bf6c36ce424655259bf42310d52a4cf33d4349d7e03b7d36`.
- [Archived clean research ledger](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/clean_causal_encoder_research_ledger.json): pre-V15 leaders, source/card identities and displaced-leader history;1746343bytes, SHA256 `4ac9a0a3b73225b3ca33915176a2785dd23524d2464021a421a99c33f5464270`.
- [V14 human results](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V14_RESULTS.md) and [current research guide](../../../../doc/EMBEDDINGS.md) give the broader search history. Their pre-V15 tables remain historical comparators; this file carries the live observations above.

No rank here establishes forecasting, identified intervention effects or downstream decision utility. Those are separate operation tests under the [project charter](../../../../doc/operations_over_representations_design_spec_v0.1.0.md).

[v3]: versions/instances/CPB-v3.market-01.json
[v4]: versions/instances/CPB-v4.market-01.json
[v5]: versions/instances/CPB-v5.market-01.json
[v6]: versions/instances/CPB-v6.market-01.json
[v7]: versions/instances/CPB-v7.market-01.json
[v8]: versions/instances/CPB-v8.market-01.json
[v9]: versions/instances/CPB-v9.market-01.json
[v10]: versions/instances/CPB-v10.market-01.json
[v11]: versions/instances/CPB-v11.market-01.json
[v12]: versions/instances/CPB-v12.market-01.json
[v13]: versions/instances/CPB-v13.market-01.json
[v14]: versions/instances/CPB-v14.market-01.json
[c945101]: #study-run-identities
[n945101]: #study-run-identities
[c945102]: #study-run-identities
[n945102]: #study-run-identities
[c945103]: #study-run-identities
[n945103]: #study-run-identities
