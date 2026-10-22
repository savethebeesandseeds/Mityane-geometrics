# Causal patch bottleneck MAE encoder ranking

Scientific evidence reviewed: **23 October 2026**. Current research reference chosen: **22 October 2026**.

This is the maintained CPB leaderboard for measured checkpoint quality. Work states remain in [Hx17](../../../../web/dashboard/heroes/17-encoder-cpb.json); [README](README.md#active-ranking-and-agent-instructions) defines how agents must maintain this file.

**Current research reference: [V16 candidate seed 946103][n946103], final logical update 2048/native36.** The owner's explicit 22 October choice is recorded in [research-reference.json](research-reference.json), including the exact checkpoint and TRAIN preprocessing identities. It leads both January trend readouts and has positive March linear/neural trend and activity scores. V4 remains the historical broad clean-causal breakthrough and current known-validation reconstruction leader; V3 remains the clean baseline.

The January evidence now contains **39 checkpoint observations**: 24 prior final2048 checkpoints (12 historical instances V3–V14, six fresh V4/V15 trajectories and six fresh V4/V16 trajectories), six V17-study final256 checkpoints, six V18-study final1024 checkpoints and three fresh V19 teacher-student final1024 checkpoints. V19 reuses the three existing V18 scattered-control final1024 observations and heads; those controls are not counted again. The controls' 256 and1024 observations share ancestry rather than independent retrainings. V18 span candidates and V19 teacher students start fresh. V17, V18 and V19 add no scoped linear record or top-three group; neural, reconstruction and TEMPO standings still draw on the prior24 checkpoints.

V16 establishes four candidate record scopes across three checkpoints: seed946103 linear/neural trend, seed946102 neural activity and seed946101 market-scaled/deleted-view linear TEMPO. Both March primary targets pass: median paired linear R² gains are +0.047276 for trend (2/3 improving pairs) and +0.193252 for activity (3/3). The full March verdict remains **mixed** because volatility and strongest timing protections fail. The separate February V15 verdict is also **mixed**. Both studies retain their original no-selection flags; the later research-reference choice is retrospective.

## Research standing

| Standing | Encoder | Reason and limit |
| --- | --- | --- |
| Current research reference | [V16 candidate seed 946103][n946103], final2048/native36 | Explicit 22 October retrospective choice for continued research. Holds both January trend records; March linear/neural trend and activity are positive. Mixed full study, 5/22 strengthened conditions and 0/8 strongest timing conditions remain recorded. |
| Historical broad breakthrough and reconstruction comparator | [CPB-v4.market-01][v4] | 19/22 original conditions, all eight original timing protections and major trend/relation gains. Still leads known-validation reconstruction. Linear volatility/activity and TRAIN MAE failed its original floors. |
| Stable clean baseline | [CPB-v3.market-01][v3] | Fresh causal-training baseline; established the historical linear-volatility R² lead; current scoped positions are in the table. Baseline is not the broad quality ranking. |
| Specialist references | [V16 seed 946102][n946102], [V16 seed 946101][n946101], [V6][v6], [V8][v8], [V11][v11], [V13][v13] | V16 seeds946102/946101 lead January neural activity and market-scaled/deleted linear TEMPO respectively. Other relation, volatility, activity and timing strengths and former leaders remain in the scoped rankings below. |
| Latest catalogued closed instance | [CPB-v14.market-01][v14] | Full trust/coverage; 3/22 strengthened conditions and no new protected record in that historical run. Its catalog position is not a quality rank. |
| Completed controlled study | V4 controls and V15, seeds 945101/945102/945103 | All six fresh native36 final2048 trajectories and twelve arithmetic audits passed. February's declared screen is mixed: activity and reconstruction gains coexist with volatility/trend losses and failed timing protections. |
| Completed temporal study | Fresh V4 controls/V16, seeds946101/946102/946103 | Six final2048 trajectories and twelve arithmetic audits complete; both March primary targets pass and the saved full verdict is mixed. The study made no automatic promotion or checkpoint selection. |
| Completed span-mask comparison | Continued scattered controls and fresh V18 span candidates, seeds 947101/947102/947103, final1024/native36 | All six endpoints and six independent ridge audits passed; all five continuation predicates failed. Limited volatility/activity/relation gains remain recorded below. No scoped record, top-three entry or reference change. |
| Completed teacher-method comparison | Reused scattered controls and fresh V19 teacher students, seeds947101/947102/947103, final1024/native36 | Three new students and three readout/audit pairs passed; control readouts/audits are inherited. Volatility improves 3/3 and relation 2/3, but trend/activity medians fail the primary conditions. No new scoped record, top-three entry or reference change. |

The current research default follows the explicit [22 October reference decision](research-reference.json) and completed readout evidence. V4's historical role remains supported by its [review](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V4_REVIEW.md). Neither role is an invented weighted score or a universal quality claim. Historical 19/22 and later 3/22 counts used different floors and must not be sorted as one metric. The reference choice changes no frozen verdict, runtime default, deployment or Mityane application selection.

## Latest bounded comparison: V19 EMA teacher, final1024

The [audited results](../../evaluation/protocols/cpb_v19_teacher_v1/RESULTS.md)
and [complete instance/evidence record](../../evaluation/records/CPB-v19.teacher-001.json)
close the three-pair teacher-method comparison on **23 October 2026**. Three
fresh V19 students train with the original scattered masks and base
GaussianNLL + 0.1 temporal objective, plus a fixed 0.1 normalized-dot target on
the 20 learned coordinates. An EMA teacher with decay 0.99 uses the full observed history at the same
cutoff; the 20→40→20 GELU predictor and teacher are training-only dependencies.
The served student retains native36 and its original inference capacity.

The three scattered-control final1024 checkpoints **and their audited heads
are reused from V18**. They are existing observations, not three additional
retrainings or newly fitted control heads. All three new candidate endpoints
and the admitted control identities seal before any new candidate head; the
control heads already existed before that seal. New work is **3,072 candidate
updates**, three TRAIN fits, six training processes, twelve target-head fits
and three independent scalar audits. Another 3,072 control updates are inherited
(768 V17 prefix updates and 2,304 V18 continuations), giving 6,144 logical updates
in the comparison. No control training or head replay was performed.

Rows retain the unchanged 112 TRAIN/32 January known-validation anchors, exact
legacy targets/associations, TRAIN-standardized ridge1 and independent scalar
audit. Every row has 32/32 coverage. Shared initial student/scaler/bank numerical
identities and realized row/attempt streams match within each seed pair. These
1024 checkpoints use the newer common fitter/calibration/sampling recipe and
do not reproduce historical 2,048 trajectories.

| Seed | Arm / readout provenance | Volatility R2 | Trend R2 | Activity R2 | Relation R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 947101 | [Scattered control / reused][c947101u1024] | 0.775581 | 0.727207 | 0.860521 | 0.515340 |
| 947101 | [V19 teacher student / new][n947101v19u1024] | 0.828765 | 0.654755 | 0.762943 | 0.220432 |
| 947102 | [Scattered control / reused][c947102u1024] | 0.809778 | 0.790894 | 0.776192 | 0.450636 |
| 947102 | [V19 teacher student / new][n947102v19u1024] | 0.839335 | 0.761573 | 0.827093 | 0.560219 |
| 947103 | [Scattered control / reused][c947103u1024] | 0.643408 | 0.772619 | 0.835143 | 0.241880 |
| 947103 | [V19 teacher student / new][n947103v19u1024] | 0.789696 | 0.810264 | 0.781873 | 0.432549 |

Every difference below is V19 candidate minus its same-seed existing control.

| Seed | Volatility delta R2 | Trend delta R2 | Activity delta R2 | Relation delta R2, descriptive |
| --- | ---: | ---: | ---: | ---: |
| 947101 | +0.053184 | -0.072452 | -0.097579 | -0.294908 |
| 947102 | +0.029557 | -0.029321 | +0.050902 | +0.109583 |
| 947103 | +0.146289 | +0.037645 | -0.053271 | +0.190669 |

**Volatility improves in all three pairs**, with median R2 gain **+0.053184**.
Relation improves in two pairs, with descriptive median **+0.109583**; its first
pair nevertheless loses **-0.294908**. Seed 947103 also improves trend +0.037645,
and seed 947102 improves activity +0.050902. These are measured scoped gains and
remain visible alongside the regressions.

**The continuation criterion fails four of five predicates.** Trend's median
-0.029321 and only 1/3 nonnegative pairs fail both primary trend rules.
Activity's median -0.053271 and only 1/3 nonnegative pairs fail both activity
rules. Volatility's positive median passes its -0.01 protection floor. Relation
is descriptive and cannot replace the declared trend/activity conditions. No
new candidate reaches an existing third linear score group or scoped record,
so the quantitative leaders below remain unchanged.

**V16 seed 946103/final2048 remains the research reference.** This teacher method
has useful volatility/relation gains but does not improve the declared balance
at 1,024. January and the inherited control scores are exposed development
evidence; January/February/March are consumed for later selection. Neural
heads, fixed-query reconstruction, TEMPO timing, full 2,048 comparison, fresh
confirmation, forecasting and application utility were not measured. Actual
TRAIN batch dispersion/near-zero-norm diagnostics are retained in the record;
they are not a population-wide collapse proof or an extra continuation gate.
The configurable construction, complete teacher/predictor learning state and
typed resume/export remain reusable component tools.

### V19 study run identities

These are the **three new** final1024/native36 served-student observations.
Existing controls retain their [V18 identities](#v18-study-run-identities) and
are not counted again. The [compact record](../../evaluation/records/CPB-v19.teacher-001.json)
binds the student checkpoints, full teacher/predictor/optimizer continuation,
TRAIN statistics, source/configuration/executables and reused-control evidence.

| Run | Checkpoint bytes / SHA256 | Saved outcomes and passed audit |
| --- | --- | --- |
| V19 teacher student 947101 | 1300259 / `2193ffb666897bdef4d74686e953fc69267941981808c733756369fc97439f7a` | [linear readout](../../../../.temp/embeddings/runs/cpb-v19-teacher-001/attempt-001/readouts/seed-947101-candidate/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v19-teacher-001/attempt-001/audits/seed-947101-candidate/summary.json) |
| V19 teacher student 947102 | 1300259 / `ef68423589313beb65c0c2d02a7ac0afce995a70ffe072f9efa2c6078dc4e4fb` | [linear readout](../../../../.temp/embeddings/runs/cpb-v19-teacher-001/attempt-001/readouts/seed-947102-candidate/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v19-teacher-001/attempt-001/audits/seed-947102-candidate/summary.json) |
| V19 teacher student 947103 | 1300259 / `f94fa47024948cbff950629ea25ed03cfd907f104acaac000bafa602a33b84ef` | [linear readout](../../../../.temp/embeddings/runs/cpb-v19-teacher-001/attempt-001/readouts/seed-947103-candidate/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v19-teacher-001/attempt-001/audits/seed-947103-candidate/summary.json) |

## Earlier bounded comparison: V18 span masks, final1024

The [audited results](../../evaluation/protocols/cpb_v18_span_v1/RESULTS.md)
and [complete instance/evidence record](../../evaluation/records/CPB-v18.span-001.json)
close the three-pair contiguous-versus-scattered training-mask comparison. Both
arms retain temporal coefficient 0.1 and the same architecture, native36 shape,
TRAIN fitting and fixed evaluation. The controls continue their exact original
V17 typed256 states through 640 to 1024 with the preserved trainer; fresh V18
candidates train to 512 and resume their own typed states to 1024. All six final
states were sealed before any head fit. **5376 new optimizer updates plus 768
preserved control-prefix updates account for 6144 logical updates**; the reused
prefixes were not replayed. All 6144 logical attempts updated successfully.

Rows use the unchanged 112 TRAIN/32 January known-validation probes, exact
legacy target/association checks, TRAIN-standardized ridge1 and six independent
scalar audits. All rows have 32/32 coverage. The common Chan fitter, calibration,
sampling and context recipe follows the V17 study and differs from historical
V16; these 1024 checkpoints do not reproduce historical 2048 trajectories. Their
scores are scoped checkpoint observations, with unequal historical budgets and
recipe differences explicit.

| Seed | Arm / training mask, temporal weight 0.1 | Volatility R2 | Trend R2 | Activity R2 | Relation R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 947101 | [Continued control / scattered][c947101u1024] | 0.775581 | 0.727207 | 0.860521 | 0.515340 |
| 947101 | [V18 candidate / contiguous span][n947101u1024] | 0.785656 | 0.505361 | 0.756590 | 0.531074 |
| 947102 | [Continued control / scattered][c947102u1024] | 0.809778 | 0.790894 | 0.776192 | 0.450636 |
| 947102 | [V18 candidate / contiguous span][n947102u1024] | 0.728684 | 0.524526 | 0.685572 | 0.448741 |
| 947103 | [Continued control / scattered][c947103u1024] | 0.643408 | 0.772619 | 0.835143 | 0.241880 |
| 947103 | [V18 candidate / contiguous span][n947103u1024] | 0.585550 | 0.679696 | 0.849355 | 0.409086 |

Every paired difference below is candidate minus its same-seed continued control.

| Seed | Volatility delta R2 | Trend delta R2 | Activity delta R2 | Relation delta R2, descriptive |
| --- | ---: | ---: | ---: | ---: |
| 947101 | +0.010075 | -0.221846 | -0.103932 | +0.015733 |
| 947102 | -0.081094 | -0.266367 | -0.090620 | -0.001895 |
| 947103 | -0.057857 | -0.092923 | +0.014212 | +0.167206 |

**Continuation failed: all five predeclared predicates failed.** Trend's median
change was -0.221846 with 0/3 nonnegative pairs, missing both the nonnegative
median and the requirement for at least two nonnegative pairs. Activity's median was -0.090620 with 1/3 nonnegative pairs,
missing the same two rules. Volatility's median -0.057857 missed its -0.01 floor.
Relation's descriptive median was +0.015733 with 2/3 improving pairs. The first
pair's volatility gain and third pair's activity/relation gains remain measured
successes within those scopes; they do not satisfy the declared continuation
rule. The best new scores for all four tasks remain below their respective
third score groups, so no scoped linear record or top-three entry changes.
Control observations remain control results and are not attributed to the span
mechanism.

**V16 seed 946103/final2048 remains the current research reference.** This valid
negative result completes the bounded 1024 comparison. January is consumed
development evidence, and February/March were already consumed by earlier
choices. Neural heads, fixed-query reconstruction, TEMPO timing, a full 2048
comparison, fresh confirmation, forecasting and application utility were not
measured. No broader winner, automatic promotion or further experiment follows.
The reusable construction/training/resume/export component and exact candidate
mask/support/lag witnesses remain available with the negative evidence.

### V18 study run identities

All rows below are final1024/native36. The [compact record](../../evaluation/records/CPB-v18.span-001.json)
binds source/configuration/executables, TRAIN statistics, stream identities,
all-six seal, complete attempts and actual resume histories. The original three
control prefixes remain bound to the separate [V17 record](../../evaluation/records/CPB-v17.balance-001.json).

| Run | Checkpoint bytes / SHA256 | Saved outcomes and passed audit |
| --- | --- | --- |
| Continued scattered control 947101 | 1300204 / `e762a08e4f1419bbcda802fc7db359a2601ca296c6d0b79846c320463adb941c` | [linear readout](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/readouts/seed-947101-control/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/audits/seed-947101-control/summary.json) |
| V18 span candidate 947101 | 1300725 / `f9afb6ca845451b13a4dacc524e8239d0c478d86125265066e175ed19b48aa2d` | [linear readout](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/readouts/seed-947101-candidate/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/audits/seed-947101-candidate/summary.json) |
| Continued scattered control 947102 | 1300204 / `c95b901c4e86beb34907d5d96ec771d0b983c7c49d1c8465cf6330272ecfb95d` | [linear readout](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/readouts/seed-947102-control/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/audits/seed-947102-control/summary.json) |
| V18 span candidate 947102 | 1300725 / `a6b2b3253022a516ca92b9443d359ebd9120e23117d588a232f691848fbe9c4e` | [linear readout](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/readouts/seed-947102-candidate/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/audits/seed-947102-candidate/summary.json) |
| Continued scattered control 947103 | 1300204 / `4e637264da7b3a4c364c75b0ee618f95241a8de72ac4ad8defc64658d34eeb5c` | [linear readout](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/readouts/seed-947103-control/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/audits/seed-947103-control/summary.json) |
| V18 span candidate 947103 | 1300725 / `b90219d742b91c312373be65be2fc9b499f91fda331ed672a1588d5e1025439e` | [linear readout](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/readouts/seed-947103-candidate/summary.json), [scalar audit](../../../../.temp/embeddings/runs/cpb-v18-span-001/attempt-001/audits/seed-947103-candidate/summary.json) |

## Earlier bounded comparison: V17, final256

The [audited results](../../evaluation/protocols/cpb_v17_balance_v1/RESULTS.md)
and [complete instance/evidence record](../../evaluation/records/CPB-v17.balance-001.json)
close the three-pair temporal-strength screen. All six fresh native36
trajectories completed256 successful updates, with actual128-to256 resume,
1536 total updates/attempts, exact legacy target parity and six independent
ridge audits. Rows use112 TRAIN/32 January known-validation probes and ridge1;
all have full coverage. The new common Chan fitter/calibration/sampler/context
recipe differs from historical V16, equally in both arms. These are early
development instances, not historical V16 reproductions or a full-budget test.

| Seed | Arm / temporal weight | Volatility R2 | Trend R2 | Activity R2 | Relation R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 947101 | Fresh control /0.1 | 0.614767 | 0.253784 | 0.724115 | 0.424003 |
| 947101 | V17 /0.03 | 0.599008 | 0.247524 | 0.701170 | 0.389874 |
| 947102 | Fresh control /0.1 | 0.733644 | -0.910284 | 0.629928 | 0.368489 |
| 947102 | V17 /0.03 | 0.731226 | -0.788282 | 0.626376 | 0.384690 |
| 947103 | Fresh control /0.1 | 0.592213 | 0.307701 | 0.820555 | 0.463791 |
| 947103 | V17 /0.03 | 0.592498 | 0.304672 | 0.811560 | 0.466977 |

**Continuation failed.** Candidate-minus-control median R2 changes were trend
-0.003029 (1/3 nonnegative), activity-0.008995 (0/3), volatility-0.002419
(within its -0.01 floor) and relation+0.003186 (2/3 improving). Seed947102's
trend gain+0.122001 and two relation gains remain recorded; they do not outweigh
the declared trend/activity failures. No new scoped linear record or better
overall encoder was established. V16 seed946103/final2048 remains the reference.
Neural heads, fixed-query reconstruction, TEMPO and fresh confirmation were not
measured. The configurable training/resume/export component remains reusable.

## How to read the quantitative ranks

V3 through V14 abbreviate the exact `CPB-vN.market-01` instance, never every checkpoint of that version. `V4 control 945101` and similar labels identify new control-arm checkpoints in `cpb-v15-objective-001`; they are distinct from `CPB-v4.market-01`. `V15 seed 945102` identifies that study's candidate arm. V4 control946101 and V16 seed946101 identify the separate V16 study. Later bounded comparisons retain their own seed/arm, mask recipe and update endpoint. Identity registers for [V15](#study-run-identities), [V16](#v16-study-run-identities), [V18](#v18-study-run-identities) and [V19](#v19-study-run-identities), plus the [V17 record](../../evaluation/records/CPB-v17.balance-001.json), bind the exact observations.

Columns 1–3 are **distinct score groups**, using dense ranks: scores within absolute **1e-12** share a group, and every tied member is retained. This reporting convention avoids ordering floating-point averaging noise. It changes no frozen acceptance predicate. Values are displayed to six decimals; ranking uses unrounded values, and the linked JSON retains individual scores. Higher R²/accuracy is better; lower MAE/NLL is better.

V3/V4 and the V15–V19 studies have native36; V5–V14 have native40. Width changes readout capacity and width-derived head seeds. Training budgets and declared recipes also differ across studies: 2048 for the prior 24 checkpoints, 256 for V17 and1024 for V18/V19. V19 reuses the existing control1024 scores; only its three fresh candidates add observations. These are observed checkpoint rankings, not controlled claims that one architecture causes a gain. The continued controls' 256/1024 observations share ancestry; historical versions each have one encoder training trajectory, and three head repetitions are not three encoder retrainings.

## Market readout ranking

Scope: **OR-05 past-window readout**, MKT-HIST-2, BTC/ETH/SOL January2024, TRAIN days1–21 and known validation days22–28. Encoders use recent128 history from the saved H256 probe geometry. Heads use 112 TRAIN and 32 validation probes, TRAIN-fitted preprocessing, ridge1 or four-output tanh16/Adam0.01/100 steps. All ranked rows have 32/32 coverage. Neural R² is the arithmetic mean of the three fixed head R² values, not pooled predictions. January is repeatedly used development evidence.

| Task / head | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Volatility / linear | [V3][v3] · **0.964673** | [V5][v5] · **0.952211** | [V10][v10] · **0.935675** |
| Volatility / neural mean | [V8][v8] · **0.735969** | [V7][v7] · **0.673305** | [V5][v5] · **0.648191** |
| Trend / linear | [V16 seed 946103][n946103] · **0.925246** | [V4][v4] · **0.907213** | [V10][v10] · **0.897085** |
| Trend / neural mean | [V16 seed 946103][n946103] · **0.693462** | [V4 control 946101][c946101] · **0.650729** | [V4 control 946103][c946103] · **0.621373** |
| Activity / linear | [V11][v11] · **0.909439** | [V15 seed 945103][n945103] · **0.906534** | [V14][v14] · **0.887832** |
| Activity / neural mean | [V16 seed 946102][n946102] · **0.815612** | [V4 control 945101][c945101] · **0.792420** | [V13][v13] · **0.762544** |
| Relation / linear | [V6][v6] · **0.622101** | [V7][v7] · **0.600801** | [V14][v14] · **0.589564** |
| Relation / neural mean | [V13][v13] · **0.457197** | [V11][v11] · **0.448775** | [V14][v14] · **0.419915** |

The V15-study activity-neural record was **0.7924196342764943**, from fresh V4 control945101, against the prior closed-study V13 value **0.762543630810775**. Its three saved head values are 0.8569086791075013, 0.7904946597000803 and 0.7298555640219014. This is a new checkpoint observation from the V15 study's control arm, not a V15-candidate gain.

## Fixed query reconstruction ranking

Scope: **OR-09 fixed-query decoding**, the unchanged saved masked MKT-HIST-2 query, final checkpoint2048. TRAIN contains112 windows and known validation32. The two splits are ranked separately. Gaussian NLL is a reported measurement; V15 and V16 train different declared composite objectives. TRAIN NLL remains descriptive and adds no protection gate.

| Population / metric | 1 | 2 | 3 |
| --- | --- | --- | --- |
| TRAIN / MAE | [V15 seed 945102][n945102] · **0.240037** | [V7][v7] · **0.241805** | [V9][v9] · **0.244540** |
| TRAIN / Gaussian NLL | [V4 control 945101][c945101] · **-0.410369** | [V7][v7] · **-0.408827** | [V4 control 946103][c946103] · **-0.407006** |
| Known validation / MAE | [V4][v4] · **0.308317** | [V12][v12] · **0.315911** | [V11][v11] · **0.320134** |
| Known validation / Gaussian NLL | [V4][v4] · **-0.045381** | [V12][v12] · **0.297064** | [V11][v11] · **0.312349** |

The V15-study TRAIN MAE record was **0.24003670546837294** for V15 seed945102, below V7's **0.24180533789857825**. Fresh V4 control945101 has TRAIN NLL **-0.41036896330440875**, below V7's **-0.40882727825899934**. These training-population records do not establish fresh-period superiority. Historical V4 established both known-validation reconstruction records; current scoped positions are in the table.

## TEMPO timing ranking

Scope: **OR-05 slow lead/lag readout**, TEMPO-5 complexity5/5, masters941101/941102/941103, 512 TRAIN and256 validation rows per master. Both input scalers are TRAIN-fitted; the deleted validation view applies the saved additional30% deletion. Scores average three master means, each averaging the three fixed head repetitions; all ranked rows have full coverage. Input scaler, view and head define separate rankings. This measures synthetic transfer, separately from market quality and fresh-month generalization.

| Input scaler / validation view / head | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Synthetic TRAIN / intact / linear | [V15 seed 945101][n945101] · **99.609375%** | [V15 seed 945103][n945103] · **99.348958%** | [V13][v13] · **98.958333%** |
| Synthetic TRAIN / intact / neural | [V6][v6] · **98.480903%** | [V16 seed 946101][n946101] · **98.350694%** | [V15 seed 945102][n945102], [V15 seed 945103][n945103], [V11][v11] · **98.263889%** |
| Synthetic TRAIN / deleted / linear | [V13][v13] · **95.833333%** | [V9][v9], [V15 seed 945103][n945103] · **95.572917%** | [V6][v6], [V16 seed 946101][n946101] · **95.312500%** |
| Synthetic TRAIN / deleted / neural | [V4 control 945101][c945101] · **94.835069%** | [V13][v13] · **94.574653%** | [V15 seed 945103][n945103] · **94.401042%** |
| Market TRAIN / intact / linear | [V9][v9] · **99.869792%** | [V6][v6], [V11][v11], [V12][v12], [V14][v14], [V15 seed 945103][n945103], [V16 seed 946101][n946101] · **99.739583%** | [V8][v8], [V13][v13], [V15 seed 945101][n945101], [V4 control 945102][c945102] · **99.609375%** |
| Market TRAIN / intact / neural | [V4 control 945102][c945102] · **99.305556%** | [V8][v8] · **99.262153%** | [V6][v6], [V16 seed 946102][n946102] · **99.218750%** |
| Market TRAIN / deleted / linear | [V16 seed 946101][n946101] · **97.526042%** | [V6][v6] · **97.265625%** | [V15 seed 945101][n945101] · **96.484375%** |
| Market TRAIN / deleted / neural | [V4][v4] · **95.616319%** | [V15 seed 945103][n945103] · **95.486111%** | [V6][v6] · **95.312500%** |

These V15-study timing highs coexist with failures against other strongest timing protections. The completed study reports every run's eight original flags and the separate prospective February screen below.

## All six January V15-study trajectories

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

**Work is complete: six of six trajectories and all twelve independent arithmetic audits passed. The saved prospective screen verdict is `mixed`; `promising` is false.** Activity linear R² improves in all three candidate/control pairs, and descriptive fixed-query MAE and NLL fall in all three pairs. Volatility and trend linear R² improve in one pair each; the volatility median, trend tolerance and candidate timing protections fail the declared screen. At the 19 October closeout, those scoped gains and losses left V4 as the reference and selected no seed. The later 22 October research-reference decision is recorded separately above.

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

## V16 temporal study: all six January trajectories

All six newly initialized V4 controls/V16 candidates use native36 and final2048/B8. They are distinct from the V15 controls and historical CPB-v4.market-01. Control retains V4 Gaussian NLL; candidate adds0.1 times the fixed mean of hidden-endpoint difference Huber1 at raw-minute lags1/4/16, with sqrt2 scaling. One shared clip1/AdamW remains. These are saved audited observations, not an independently reconstructed optimizer trajectory.

**V16 operational recovery.** The five previously completed trajectories remain unchanged. Candidate946103 restored its original typed checkpoint512 and completed logical updates513–2048 using a separately admitted recovery executable. Available discarded-tail complete CSV rows and torn fragment prefixes matched exactly before reconciliation. The physical repeated-update count, complete original execution cost and whole-trajectory per-parameter counters remain unknown; no uninterrupted or zero-recomputation claim is made. The [operational amendment](../../evaluation/protocols/cpb_v16_temporal_recovery_v1/RECOVERY.md), [exact recovered-report verification](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery-report-verification-001.json) and [actual publication record](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery001/publication.json) bind that extension. These execution limitations do not change the saved score definitions,24-checkpoint population, dense-rank grouping or original predicates.

Recovery source review: [exact source extension](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery-source-review-001.json); [actual compiled build](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery-build-admission-001.json); [explicit operational admission](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery-admission-001.json); [actual suffix execution](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/recovery-suffix-execution-001.json). New recovery binary: `/opt/mityane-geometrics/embeddings/build/cpb-v16-temporal-001/cpb_v16_temporal_recovery_candidate_train`, 927984bytes, SHA256 `41cf522307e00185eee02001bb603d9a45f3012d0bfd22a3fa50961f630610ab`. Candidate946103 whole-trajectory costs are null; its recorded learning counters and suffix timing cover logical513–2048 only. See the human results for original-stage calibration and separately scoped suffix measurements. No cost-efficiency ranking is inferred.

### January Linear R² — V16 study

| Run | Volatility | Trend | Activity | Relation | Coverage |
| --- | --- | --- | --- | --- | --- |
| [V4 control 946101][c946101] | 0.768341 | 0.854608 | 0.789226 | 0.429196 | full declared |
| [V16 seed 946101][n946101] | 0.691376 | 0.762465 | 0.784860 | 0.422845 | full declared |
| [V4 control 946102][c946102] | 0.790769 | 0.744868 | 0.836326 | 0.406025 | full declared |
| [V16 seed 946102][n946102] | 0.688165 | 0.818543 | 0.842149 | 0.486499 | full declared |
| [V4 control 946103][c946103] | 0.615560 | 0.841987 | 0.844730 | 0.291859 | full declared |
| [V16 seed 946103][n946103] | 0.745787 | 0.925246 | 0.855231 | 0.249267 | full declared |

### January Neural mean R² — V16 study

| Run | Volatility | Trend | Activity | Relation | Coverage |
| --- | --- | --- | --- | --- | --- |
| [V4 control 946101][c946101] | 0.302772 | 0.650729 | 0.742071 | 0.117352 | full declared |
| [V16 seed 946101][n946101] | 0.398748 | 0.278750 | 0.650478 | 0.209820 | full declared |
| [V4 control 946102][c946102] | 0.037965 | 0.450068 | 0.727045 | 0.183345 | full declared |
| [V16 seed 946102][n946102] | -0.028552 | 0.498439 | 0.815612 | 0.197636 | full declared |
| [V4 control 946103][c946103] | -0.019702 | 0.621373 | 0.672556 | 0.105447 | full declared |
| [V16 seed 946103][n946103] | 0.037713 | 0.693462 | 0.673931 | 0.194984 | full declared |

### January reconstruction — V16 study

| Run | TRAIN MAE | TRAIN NLL | Known-VAL MAE | Known-VAL NLL |
| --- | --- | --- | --- | --- |
| [V4 control 946101][c946101] | 0.267098 | -0.358751 | 0.390534 | 0.922563 |
| [V16 seed 946101][n946101] | 0.258649 | -0.363471 | 0.419242 | 1.117587 |
| [V4 control 946102][c946102] | 0.256308 | -0.393660 | 0.366233 | 0.432826 |
| [V16 seed 946102][n946102] | 0.267641 | -0.384654 | 0.381787 | 0.529572 |
| [V4 control 946103][c946103] | 0.250390 | -0.407006 | 0.402902 | 0.928599 |
| [V16 seed 946103][n946103] | 0.253763 | -0.400952 | 0.430649 | 1.411167 |

Candidate fixed-query lag16 remains unsupported/null because the unchanged query hides one eight-step patch. No target mask is expanded; Gaussian query scores remain distinct from the composite training loss. Full initial/final scores, individual head repetitions and all132 January criterion objects are retained in [human results](../../evaluation/protocols/cpb_v16_temporal_v1/RESULTS.md) and [machine JSON](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/report.json).

### January protection flags — V16 study

| Run | Saved conditions | Eight timing flags | All eight retained |
| --- | --- | --- | --- |
| [V4 control 946101][c946101] | 4/22 | F F F F F F F F | no |
| [V16 seed 946101][n946101] | 3/22 | F F F F F F P F | no |
| [V4 control 946102][c946102] | 3/22 | F F F F F F F F | no |
| [V16 seed 946102][n946102] | 4/22 | F F F F F F F F | no |
| [V4 control 946103][c946103] | 4/22 | F F F F F F F F | no |
| [V16 seed 946103][n946103] | 5/22 | F F F F F F F F | no |

## March confirmation — V16 study

**Work is complete; saved prospective screen: mixed; promising: false.** Trend and activity are the primary targets (positive medians and at least2/3 improving pairs separately); volatility has median tolerance≥-0.01. Every candidate must retain all eight unchanged January timing protections. Relation and query reconstruction are descriptive. This rule differs from the V15 February screen; January, February and March are not mixed into one population ranking.

| Encoder seed | Δ trend | Δ activity | Δ volatility | Δ relation (descriptive) |
| --- | --- | --- | --- | --- |
| 946101 | -0.073135 | +0.022193 | +0.018031 | +0.669197 |
| 946102 | +0.096483 | +0.193252 | -0.033500 | +0.223590 |
| 946103 | +0.047276 | +0.244398 | -0.041547 | +0.237152 |

| Saved screen target | Median Δ | Range Δ | Improving pairs |
| --- | --- | --- | --- |
| past_trend | +0.047276 | [-0.073135, +0.096483] | 2/3 |
| past_activity | +0.193252 | [+0.022193, +0.244398] | 3/3 |
| past_volatility | -0.033500 | [-0.041547, +0.018031] | 1/3 |

Saved screen reasons: `March linear volatility median difference is below -0.01 or undefined`; `at least one candidate fails a frozen hard timing condition`. No verdict, difference, median, interval or metric is recalculated by this ranking helper.

### March absolute linear R²

| Run | Volatility | Trend | Activity | Relation |
| --- | --- | --- | --- | --- |
| [V4 control 946101][c946101] | 0.827999 | 0.868563 | 0.068591 | -0.904167 |
| [V16 seed 946101][n946101] | 0.846030 | 0.795428 | 0.090784 | -0.234971 |
| [V4 control 946102][c946102] | 0.890406 | 0.707554 | 0.159191 | -0.765573 |
| [V16 seed 946102][n946102] | 0.856907 | 0.804037 | 0.352443 | -0.541983 |
| [V4 control 946103][c946103] | 0.806104 | 0.871328 | 0.002308 | -0.754805 |
| [V16 seed 946103][n946103] | 0.764557 | 0.918605 | 0.246707 | -0.517653 |

### March absolute neural mean R²

| Run | Volatility | Trend | Activity | Relation |
| --- | --- | --- | --- | --- |
| [V4 control 946101][c946101] | 0.500988 | 0.506033 | -0.140147 | -0.022026 |
| [V16 seed 946101][n946101] | 0.577511 | 0.662105 | -0.242164 | 0.035381 |
| [V4 control 946102][c946102] | 0.456415 | 0.529338 | -0.167314 | -0.100474 |
| [V16 seed 946102][n946102] | 0.342331 | 0.490709 | -0.148026 | -0.529505 |
| [V4 control 946103][c946103] | 0.415820 | 0.659982 | -0.102542 | -0.127128 |
| [V16 seed 946103][n946103] | 0.470166 | 0.698410 | 0.125801 | -0.290792 |

### March descriptive fixed query

| Run | MAE | Gaussian NLL | Passed confirmation evidence |
| --- | --- | --- | --- |
| [V4 control 946101][c946101] | 4.902232 | 125.819852 | [receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/confirmation-audit.json) |
| [V16 seed 946101][n946101] | 4.984493 | 119.781735 | [receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/confirmation-audit.json) |
| [V4 control 946102][c946102] | 4.989452 | 182.473701 | [receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/confirmation-audit.json) |
| [V16 seed 946102][n946102] | 4.970938 | 182.892624 | [receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/confirmation-audit.json) |
| [V4 control 946103][c946103] | 4.857849 | 213.362216 | [receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/confirmation-audit.json) |
| [V16 seed 946103][n946103] | 4.892664 | 193.686043 | [receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/confirmation-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/confirmation-audit.json) |

March is now consumed evidence for later recipe choices; February was already consumed by V15. Three paired encoders on one additional month are a screen, not general superiority. Head repetitions are not encoder or population replications; no forecasting or intervention claim follows. At the 21 October study closeout, V4 remained the broad reference and V3 the baseline, without automatic promotion, best-seed/checkpoint selection, catalog application or Mityane selection. The separately dated 22 October choice now makes V16 candidate946103 the research reference while preserving those original results and flags.

## V16 study run identities

| Run | Final2048 checkpoint bytes / SHA256 | Saved outcomes | Passed evidence |
| --- | --- | --- | --- |
| [V4 control 946101][c946101] | 1291409 / `7e5c3b613cdc9ed56da0f043e0df812bde09f01cb320f44fa00dca3fdf888d68` | [market](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/control/independent-audit.json) |
| [V16 seed 946101][n946101] | 1295473 / `65a47a5fa27617b2c514af376c457223ec3ec67a6b8f5933abf54117daff1695` | [market](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946101/candidate/independent-audit.json) |
| [V4 control 946102][c946102] | 1291409 / `65ce122e2b79c7fbfbb2325d48f174cd2ad7c2bb9f5cf9a2841f18ed725f6bc6` | [market](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/control/independent-audit.json) |
| [V16 seed 946102][n946102] | 1295473 / `69ebd13213e953b8bc86fefa73035eacb3e10ab483d2fb09e08acd7c9240868b` | [market](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946102/candidate/independent-audit.json) |
| [V4 control 946103][c946103] | 1291409 / `6541439e55a9da59d38000cea31faef2fbb945a46064f7531b89004d86432887` | [market](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/control/independent-audit.json) |
| [V16 seed 946103][n946103] | 1295473 / `b6f3cbe711235307f362bf2f999571d65d00c85825f8a4a5c2fbc4fa4242e51e` | [market](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/heads-final/market/report.json), [query](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/training/training-summary.json), [TEMPO 941101](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/heads-final/synthetic/master-941101/report.json), [TEMPO 941102](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/heads-final/synthetic/master-941102/report.json), [TEMPO 941103](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/heads-final/synthetic/master-941103/report.json) | [training](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/train-receipt.json), [development](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/development-receipt.json), [arithmetic audit](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/independent-audit.json) |

Actual [complete report](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/report.json) (2496656bytes; SHA256 `263a8a4debbcd34ea6fe1364dc314a73a9804311fd77d30d0fbda57ae11c40f3`), [report card with eighteen phase receipts/twelve audits](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/report-inputs-001.json), [study card](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/run-inputs-001.json), [development seal](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/development-seal.json), [human results](../../evaluation/protocols/cpb_v16_temporal_v1/RESULTS.md), [compact record](../../evaluation/records/CPB-v16.temporal-001.json) and [final peer review](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/final-study-review-001.json) retain exact bindings. V16 uses its authoritative report.mjs/report.json derivation.

## Historical ancestry and cross-family context

V3–V14 and all twelve fresh study trajectories are fresh-weight clean-causal research; source branching is not trained-weight inheritance. CPB V1/V2 retain pretrained ancestry and stay outside these clean score tables. Across the earlier RPB family, **RPB-v24.market-01.alt-01 native36** remains the historical balanced leader:8/9 market conditions plus all four primary TEMPO-5 transfer comparisons. RPB-v30 later passed9/9 market and8/8 strict native36 conditions with transfer pending. Their later causal-eligibility diagnosis is preserved, so their historical achievements are not folded into the clean CPB ranking. See the [reviewed handoff addendum](../../../../doc/EMBEDDINGS_HANDOFF_ADDENDUM_2026-10-18.md).

## Ranking changes

**19 October2026 — initial active ranking.** V4's broad-lead role and V3 baseline were retained at that stage. The following six January score leaders were first registered after the development measurements passed, while the full study was still incomplete. They remain recorded after study completion, alongside each displaced leader and its original evidence.

| Scope | Previous closed-study leader | Latest observed leader |
| --- | --- | --- |
| Activity / neural mean R² | [V13][v13] · 0.762544 | [V4 control 945101][c945101] · **0.792420** |
| TRAIN / MAE | [V7][v7] · 0.241805 | [V15 seed 945102][n945102] · **0.240037** |
| TRAIN / NLL | [V7][v7] · -0.408827 | [V4 control 945101][c945101] · **-0.410369** |
| TEMPO synthetic_train / validation_intact / linear | [V13][v13] · 98.958333% | [V15 seed 945101][n945101] · **99.609375%** |
| TEMPO synthetic_train / validation_deleted / neural | [V13][v13] · 94.574653% | [V4 control 945101][c945101] · **94.835069%** |
| TEMPO market_scaler / validation_intact / neural | [V8][v8] · 99.262153% | [V4 control 945102][c945102] · **99.305556%** |

**19 October2026 — completed-study finalization.** The authoritative final-unique report closes all six trajectories and twelve arithmetic audits with the prospective verdict **mixed**, while retaining all six January highs above. The separate February section now records every pair, all six absolute readout/query results and every timing flag. February is consumed; V4's historical broad-reference role and V3's baseline role remain, with no seed/checkpoint selection, automatic promotion, catalog application or Mityane selection.

**21 October 2026 — audited ongoing V16 first pair; full study pending.** Candidate946101 sets market-scaler/deleted TEMPO linear accuracy 0.9752604166666666 above V6 0.97265625. Its fresh V4 control946101 sets January neural trend mean R² 0.650729417976703 above historical V4 0.5698752307038314. All first-pair gains and regressions remain in the [retained pending section](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-section-001.md) and [hash-bound provisional receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-001.json). The old18 tables remain the completed-study snapshot; remaining pairs/March are outside this entry. No overall verdict, seed selection or protection-floor pass is inferred.

**21 October 2026 — all-six audited January V16 update; March verdict pending.** Candidate946103 now leads linear/neural trend, candidate946102 leads neural activity, and candidate946101 retains the market-scaler/deleted linear TEMPO record. Both control946101/control946103 neural-trend observations above historical V4 remain visible; candidate946103 supersedes the first-pair control leader. The [all-six retained section](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-section-002.md) and [receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-002.json) preserve these current observations and the full old evidence chain. Original18 tables remain the completed-study snapshot until final24 closeout. No March result, protection-floor pass or overall verdict is inferred.

**21 October 2026 — completed V16 temporal study.** All six fresh controls/candidates and twelve arithmetic audits are retained; March saved screen **mixed**. January tables now include all24 checkpoints with unrounded1e-12 dense tie groups. V15 February evidence/history and all former leaders remain. At that closeout, V4 broad reference/V3 baseline were retained, without seed selection, automatic promotion, catalog application or Mityane selection.

| January scope | Former score group1 | Current score group1 |
| --- | --- | --- |
| past_trend:linear_r2 | [V4][v4] · **0.907213** | [V16 seed 946103][n946103] · **0.925246** |
| past_trend:neural_mean_r2 | [V4][v4] · **0.569875** | [V16 seed 946103][n946103] · **0.693462** |
| past_activity:neural_mean_r2 | [V4 control 945101][c945101] · **0.792420** | [V16 seed 946102][n946102] · **0.815612** |
| market_scaler:validation_deleted:linear_head_accuracy | [V6][v6] · **97.265625%** | [V16 seed 946101][n946101] · **97.526042%** |

[Audited V16 report](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/report.json); [human results](../../evaluation/protocols/cpb_v16_temporal_v1/RESULTS.md); [complete compact record](../../evaluation/records/CPB-v16.temporal-001.json).

The audited first-pair observations remain traceable in the [first-pair section](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-section-001.md) and [first-pair receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-001.json); the later [all-six January section](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-section-002.md) and [second provisional receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-provisional-002.json) preserve the superseding candidate trend/activity leaders and both fresh-control achievements. The [Hx17 naming receipt](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-hx17-naming-001.json), [named first-pair ranking](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-before-provisional-002.md) and [prior completed18 ranking](../../../../.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/ranking-before-provisional-001.md) retain the exact earlier identities. Full24 results now replace only the live all-six pending block; both dated stages of history and former leaders remain.

**22 October 2026 — current research reference changed from historical V4 to [V16 candidate seed 946103][n946103], final2048/native36.** The owner requested that the best current encoder become the research reference. The [explicit decision and artifact identities](research-reference.json) select this checkpoint for continued research because it holds both January trend records and has positive March linear/neural trend and activity scores, improving its matched control on both linear primary tasks. V16 seeds946102/946101 retain their separate activity/timing records. V4 remains the historical breakthrough and known-validation reconstruction comparator; V3 remains the baseline. This is a retrospective reference choice after observing the completed study. The frozen mixed verdict, failed volatility/timing protections, all earlier no-selection flags and unknown recovery cost/repetition limits remain unchanged. No runtime default, deployment or Mityane application selection changes.

**22 October 2026 — completed V18 contiguous-span mask comparison.** The [final1024 record](../../evaluation/records/CPB-v18.span-001.json) adds six audited January linear checkpoint observations, bringing the documented total to 36 alongside the 24 prior final2048 and six V17 final256 observations. Three scattered controls continue their exact original V17 coefficient0.1 state ancestry; three span candidates start fresh. All five predeclared continuation predicates fail: median candidate-minus-control trend/activity/volatility R2 changes are -0.221846/-0.090620/-0.057857. Relation's descriptive median gain +0.015733, first-pair volatility gain and third-pair activity gain are retained. No observation displaces a linear top-three group or scoped record; earlier leaders and the selected V16 reference remain unchanged. New physical work is 5376 updates plus 768 reused prefix updates; no fresh population, full2048 comparison or additional quality panel was measured.

**23 October 2026 — completed V19 EMA-teacher comparison.** The [final1024 record](../../evaluation/records/CPB-v19.teacher-001.json) adds three fresh teacher-student observations, bringing the documented January total from36 to39. Its three original scattered controls and audited heads are reused from V18 and are not counted or fitted again. Volatility improves in all three pairs (median+0.053184), relation improves in two (median+0.109583), seed947103 improves trend and seed947102 improves activity. The large first-pair relation regression-0.294908 remains visible. The continuation criterion fails four primary trend/activity predicates; volatility passes its protection. None of the three new candidates enters an existing linear top-three group or scoped record, and V16 remains the selected reference. New work is3072 updates/twelve head fits/three scalar audits, with3072 control updates and their heads/audits inherited. No new neural, reconstruction, timing, fresh-period or full2048 comparison was measured.
Future entries must record date, former/new leader or research reference, exact comparison scope, evidence links and the reason for the change. Retractions and invalidated eligibility need an explicit entry rather than silently deleting a result. Frozen historical protection floors, measured-instance catalogs and Mityane selection are independent records.

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

[c946101]: #v16-study-run-identities
[n946101]: #v16-study-run-identities
[c946102]: #v16-study-run-identities
[n946102]: #v16-study-run-identities
[c946103]: #v16-study-run-identities
[n946103]: #v16-study-run-identities
[c947101u1024]: #v18-study-run-identities
[n947101u1024]: #v18-study-run-identities
[c947102u1024]: #v18-study-run-identities
[n947102u1024]: #v18-study-run-identities
[c947103u1024]: #v18-study-run-identities
[n947103u1024]: #v18-study-run-identities
[n947101v19u1024]: #v19-study-run-identities
[n947102v19u1024]: #v19-study-run-identities
[n947103v19u1024]: #v19-study-run-identities
