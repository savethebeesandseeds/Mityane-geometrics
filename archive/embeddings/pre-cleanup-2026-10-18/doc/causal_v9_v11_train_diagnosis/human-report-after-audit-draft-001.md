<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# Audited saved TRAIN diagnosis: CPB-v9 and CPB-v11

**The independent saved-TRAIN arithmetic audit passes68264 checks, zero mismatches.** No new encoder experiment. One main diagnosis and one independent arithmetic audit ran once in the existing cuwacunu container. All displayed measurements copy their saved results; no score or diagnostic metric was recomputed for this report. The main/audit mirrors report exact before/after bytes. Separate root/peer documentary completion review follows this authored report; no new encoder or application change is selected. The whole preliminary report remains preserved.

V11 retains strong fitted TRAIN information, but its changes are mixed. Relative to cached V9 finals, linear volatility, trend and relation R² are higher; linear activity and all four neural means are lower. All112 reconstruction MAE and Gaussian NLL are also worse. This is not a uniform TRAIN information gain or an explanation of the separately recorded known-VALIDATION gains and failures. CPB-v3.market-01 remains the clean baseline and sole DEV encoder.

## Evidence and methods

Dataset: **MKT-HIST-2** · joint BTC/ETH/SOL January2024 minute histories, C3/H128/F9, recent128 past-window retention · **complexity4/5, ordinal**. The three assets are joint channels in one market panel, not three independent cohorts. These are retrospective retention tasks, not forecasts.

Each saved readout was fitted on the same112 chronological TRAIN windows scored here. Historical quality used32 known chronological VALIDATION windows, but their numerical fields, all TEST/stress/TEMPO inputs and raw caches are outside this diagnosis. Initial/final mean saved0/2048 updates. Each version has one historical fresh2048-update/B8 CUDA trajectory; three saved readout repetitions2701/2802/2903 are head variation on one checkpoint/split, not three encoder runs. This diagnosis performs no new calibration, export, model/head forward, fit or optimizer update.

Both representations are native40: content20 + literal spectral12 + mean4 + independent uncertainty4. Saved fixed heads are ridge penalty1 and16-tanh/four-output neural heads, Adam0.01/100 updates,164/724 fitted values, permitted TRAIN-only normalizers and unchanged D40 seeds6049515622977958931/8262941649267994858/2095371932330426401. All repetitions remain in the saved result; no best fit, seed, checkpoint or post-encoder transform is selected.

V11 changes V9's eight mean-export rows per original temporal bin into four mean and four componentwise centered-variance rows under the same half-local/half-global attention. The existing two-sum mean and post-bias empty-local whole-slice zeros remain; native40 and parameter shapes/counts remain. Intermediate variance differs from native uncertainty4 and decoder conditional variance. Both use unit Huber1 mean plus unit detached-error Gaussian variance calibration, shared global clip1 and one AdamW. V10's live joint Gaussian mean route is excluded.

V9 is V11's scientific source/objective parent only. All V11 learned values, optimizer and permitted TRAIN statistics start fresh, source_updates0; no trained parent or oracle weights are inherited. Initial DATA pairing remains native20:40 plus five whole statistic tensors. Learned content0:20 is unpaired, so different initial whole40 readouts or query means are permitted; whole40 equality is not claimed. Historical V10 whole40 pairing keeps its original different semantics.

## Four saved TRAIN task tables

The five-column regression layout follows reporting standard1.2. `.init` is initial0; the unsuffixed instance is final2048. Values display ten significant digits; full precision and all saved repetitions/physical errors remain in the linked diagnosis. Neural columns are the saved mean over all three head repetitions. R² uses this scored TRAIN population's constant mean as zero; it is not accuracy or generalization.

### Past volatility

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | 0.8681049024 | 0.9346514182 | 100 (112/112) |
| CPB-v11.market-01.init | 40 | 0.8704379066 | 0.9304516397 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.990979636 | 0.9611169145 | 100 (112/112) |
| CPB-v11.market-01 | 40 | 0.9915892992 | 0.9535589501 | 100 (112/112) |

Physical prediction errors, unit: **logreturn**. These are copied saved mean errors over the original repetitions, not reconstructed predictions or newly fitted heads.

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns in recent128 · **complexity4/5, ordinal** · saved TRAIN112.

| Method | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 0.0001357620951 | 0.0001827978634 | 9.139398637E-05 | 0.0001280364003 |
| CPB-v11.market-01.init | 0.0001387027591 | 0.0001811739557 | 9.516026118E-05 | 0.0001321480457 |
| CPB-v9.market-01 | 3.828330273E-05 | 4.780450429E-05 | 7.552729854E-05 | 9.886708345E-05 |
| CPB-v11.market-01 | 3.717972113E-05 | 4.616075248E-05 | 7.910417677E-05 | 0.0001078014187 |

### Past trend

Dataset: **MKT-HIST-2** · channel-mean supported adjacent close log-return trend in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | 0.7078987245 | 0.9591511634 | 100 (112/112) |
| CPB-v11.market-01.init | 40 | 0.6937123027 | 0.9444130642 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.9075641665 | 0.9524614608 | 100 (112/112) |
| CPB-v11.market-01 | 40 | 0.9110286797 | 0.9382186199 | 100 (112/112) |

Physical prediction errors, unit: **logreturn/tick**. These are copied saved mean errors over the original repetitions, not reconstructed predictions or newly fitted heads.

Dataset: **MKT-HIST-2** · channel-mean supported adjacent close log-return trend in recent128 · **complexity4/5, ordinal** · saved TRAIN112.

| Method | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 3.61331303E-05 | 4.512614499E-05 | 1.279661852E-05 | 1.677008085E-05 |
| CPB-v11.market-01.init | 3.566489438E-05 | 4.620896957E-05 | 1.439798006E-05 | 1.961435648E-05 |
| CPB-v9.market-01 | 1.949013121E-05 | 2.538526425E-05 | 1.341811704E-05 | 1.818537807E-05 |
| CPB-v11.market-01 | 1.887022258E-05 | 2.490499881E-05 | 1.483424212E-05 | 2.071447203E-05 |

### Past activity

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | 0.9382282393 | 0.9562879264 | 100 (112/112) |
| CPB-v11.market-01.init | 40 | 0.9264675744 | 0.964113309 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.9753968809 | 0.9731731582 | 100 (112/112) |
| CPB-v11.market-01 | 40 | 0.9676749422 | 0.9699659773 | 100 (112/112) |

Physical prediction errors, unit: **log1p(base_volume)**. These are copied saved mean errors over the original repetitions, not reconstructed predictions or newly fitted heads.

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) in recent128 · **complexity4/5, ordinal** · saved TRAIN112.

| Method | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 0.1178825777 | 0.1543277611 | 0.09868062187 | 0.1295592601 |
| CPB-v11.market-01.init | 0.132608976 | 0.1683792272 | 0.09089358311 | 0.116806301 |
| CPB-v9.market-01 | 0.07655912449 | 0.09739667937 | 0.07982998374 | 0.101646473 |
| CPB-v11.market-01 | 0.08901114748 | 0.1116397154 | 0.08690566221 | 0.1075073407 |

### Past relation

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | 0.6549993607 | 0.9101974546 | 100 (112/112) |
| CPB-v11.market-01.init | 40 | 0.6710415617 | 0.8933798703 | 100 (112/112) |
| CPB-v9.market-01 | 40 | 0.7176859025 | 0.8734249472 | 100 (112/112) |
| CPB-v11.market-01 | 40 | 0.7345649307 | 0.8704845323 | 100 (112/112) |

Physical prediction errors, unit: **correlation**. These are copied saved mean errors over the original repetitions, not reconstructed predictions or newly fitted heads.

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns in recent128 · **complexity4/5, ordinal** · saved TRAIN112.

| Method | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 0.0475858637 | 0.05966124879 | 0.02347135278 | 0.02987745878 |
| CPB-v11.market-01.init | 0.04625253306 | 0.05825764237 | 0.02526688974 | 0.03306402499 |
| CPB-v9.market-01 | 0.04315089728 | 0.05396953451 | 0.02840534618 | 0.03601635191 |
| CPB-v11.market-01 | 0.04309224556 | 0.05233130202 | 0.0284738675 | 0.03650128335 |

The final V11 volatility information is highly readable on these fitted TRAIN rows: linear R²0.9915892992 and neural mean0.9535589501. That rules out an absence of fitted TRAIN signal in this panel, not a validation problem or a forecasting limitation. Trend linear improves over V9 while neural mean declines. Activity falls on both TRAIN heads despite its separate known-VALIDATION records; relation linear improves while neural mean declines and remains below its own initial mean. The different directions require preservation, not a single overall encoder score.

Exact navigation: `models.v11.stages.{initial,final}.per_task_mean_across_saved_repetitions` contains the new V11 scores; `cached_v9_TRAIN.stages.{initial,final}` contains the unchanged old audited V9 panel. Each stage's `saved_TRAIN_head_scores[]` retains all2701/2802/2903 R², MAE/RMSE, predictions/support and fixed early56/late56 descriptions. The diagnosis compares saved scalars in `comparison.saved_TRAIN_readout_scalar_differences`; it performs zero new old-V9 numerical reductions.

## Reconstruction and native/readout geometry

Dataset: **MKT-HIST-2** · standardized masked historical reconstruction · **complexity4/5, ordinal** · fixed TRAIN112,24192 supported target cells per state. Each metric retains the declared eligible-example/channel/cell reduction. Lower MAE is better; Gaussian NLL omits its constant and may be negative. Huber mean, variance calibration and total are distinct objective values.

| Method | Query MAE | Reporting Gaussian NLL | Mean Huber1 | Detached-error variance NLL | Total objective |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 0.6642581977 | 0.4238473016 | 0.3237604272 | 0.4238473016 | 0.7476077288 |
| CPB-v11.market-01.init | 0.6640733605 | 0.4237330498 | 0.3236635132 | 0.4237330498 | 0.747396563 |
| CPB-v9.market-01 | 0.2445403465 | -0.3924516259 | 0.08305971607 | -0.3924516259 | -0.3093919099 |
| CPB-v11.market-01 | 0.2517975124 | -0.369659381 | 0.08466702545 | -0.369659381 | -0.2849923556 |

V11 final all112 MAE0.2517975124 versus V9's0.2445403465 and NLL−0.3696593810 versus−0.3924516259 are less favorable. The saved comparison records MAE difference+0.007257165916 and NLL difference+0.02279224489; these are copied saved differences. First8 query MAE is0.2630482911 versus0.2584784768, also less favorable. Those eight detailed rows cannot stand for all112 or explain the task/head differences. TRAIN NLL remains descriptive and adds no retention gate.

The native feature descriptions remain concentrated in parameter-dependent coordinates: final learned20 participation ratio is V9 3.333908757 versus V11 3.276767195; uncertainty4 is1.050884883 versus1.048875014. These are population-covariance concentration statistics, not numerical rank, intrinsic dimension or proof that more coordinates/decorrelation are required. The literal spectral block retains its saved statistics.

The existing joint ridge decomposition shows useful allocation in both versions. At repetition2701, V11 uncertainty4's volatility contribution has target Pearson0.9818530187 (V9 0.9786446628); V11 mean4's trend contribution has target Pearson0.9237425690 (V9 0.9225836932). For activity, learned20/mean4 contribution correlation is0.8554130954 in V11 and0.8804221060 in V9. Relation uses several correlated blocks. These are decompositions of an already fitted predictor in physical target units, not independent block heads, ablation, gradient conflict or proof of branch competition. Complete four-task/all-repetition/half/covariance records remain under `saved_linear_readout_block_decomposition`.

The first8 mean4 residual panel manually contracts the same-stage saved change decoder. The parent remainder is stored total minus that F64 contraction, approximating original F32 addition. No parent nonlinear decoder, separate content/spectral decoder attribution or counterfactual predictor is formed. It is separate from the actual training witnesses described below.

## New V11 associations and temporal observations

The result contains new V11-only `reconstruction_readout_associations` for every task, head, repetition and all112/early56/late56 scope. It retains signed and absolute physical readout residual associations with all five standardized query-component rows. These are descriptive, in-sample row associations; there is no matched newly computed V9 association panel. The full result retains contrary signs and weak associations without selecting a favorable subset.

For example, final TRAIN112 reconstruction-MAE versus absolute neural volatility residual Pearson is0.1281580491/0.1480090347/0.2305717505 across2701/2802/2903. Trend is0.3350077808/0.3726992097/0.2833781899; activity0.2242428586/0.1816599220/0.01371876974; relation−0.1174517098/−0.004355822827/+0.02222015217. All112 rows are eligible in these selected fixed all112 examples. No correlation here identifies a validation cause, an objective coefficient or a task-specific training remedy; complete components, heads and fixed halves remain available.

Dataset: **MKT-HIST-2** · existing postclip export-subgroup records · **complexity4/5, ordinal** · all2048 historical V11 updates, eight bin/moment groups,16384 CSV rows. The entries below are already saved means of parameter-coordinate magnitudes, not new reductions. Movement includes AdamW decay.

| Original bin | Moment rows | Recorded rows | Weight gradient L2 mean | Bias gradient L2 mean | Weight movement L2 mean | Bias movement L2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | mean | 2048 | 0.008813839777 | 0.001410660074 | 0.002559086183 | 0.0002601005402 |
| 0 | centered_variance | 2048 | 0.001298440467 | 0.0008583272502 | 0.002709726305 | 0.0002844791214 |
| 1 | mean | 2048 | 0.009470987799 | 0.001481106932 | 0.002690457412 | 0.0002749760933 |
| 1 | centered_variance | 2048 | 0.001227150446 | 0.0008170369549 | 0.002428820784 | 0.0002716271614 |
| 2 | mean | 2048 | 0.01223881131 | 0.001973526408 | 0.002565176811 | 0.0002600609508 |
| 2 | centered_variance | 2048 | 0.001461729294 | 0.000918990993 | 0.00266621598 | 0.000294728354 |
| 3 | mean | 2048 | 0.01137127991 | 0.002098249717 | 0.002777608034 | 0.0002902464901 |
| 3 | centered_variance | 2048 | 0.001415239652 | 0.0008702144324 | 0.002515110956 | 0.0002779033825 |

The recorded means for variance-row weight gradients are numerically smaller than mean-row means while movement magnitudes are of similar displayed order. Different units, parameterization, clipping and AdamW prevent interpreting that as branch importance, starvation or causal gradient dominance. The same result retains every subgroup's quantiles, both recorded element counts and all four fixed512-update blocks. Existing learning/event/gauge/Hadamard trust proofs are reused as preserved context rather than replayed by this diagnosis.

The two actual witnesses describe the original saved forwards at updates1/2048. Each has24 packed rows and four present local bins per route in these saved observations. Both ordinary/contextual moment/export descriptions are retained. The ordinary upstream-defined flag is false with the explicit saved zero fallback; contextual is true. That flag describes the captured training route, not a disconnected encoder branch.

Captured centered-variance median values in the later witness are lower than in the first witness across these displayed bins, but they are different sampled batches and different learned states. They are not matched-input evidence of collapsed dispersion or a training-wide distribution estimate. Manual mean/variance export contractions show their actual signed values and before-step coefficient allocation; no task contribution is inferred from their magnitudes. Both route details and full-mixture concentration stay in `temporal_moment_observations.actual_witnesses`.

**Update2048 is a pre-step2047 witness, not the final2048 checkpoint.** Its affine exports use its own saved before-step export coefficients. Never pair it with the final2048 decoder. There are no matched V9 intermediate moment captures and no all112 query-row intermediate moments; this diagnosis does not manufacture them. Two actual batches cannot connect moment rows to all112 downstream task errors or reconstruct full optimizer/trajectory gradients.

## Arithmetic, timing and preservation scope

Dataset: **MKT-HIST-2** · saved TRAIN scalar/manual-affine diagnosis · **complexity4/5, ordinal** · one main producer, no new encoder or head training. The actual main internal elapsed time is0.6928441099998963 seconds. Main wrapper, independent audit and independent wrapper times are separately recorded below; they are not added to historical GPU training time.

Manual arithmetic was performed despite zero model/module forward counters:

| Saved affine family | Scalar dot outputs | Multiply-add terms |
| --- | ---: | ---: |
| mean4_decoder | 55296 | 221184 |
| ridge_block | 10752 | 107520 |
| ridge_total | 2688 | 107520 |
| temporal_mean4_export | 1536 | 98304 |
| temporal_variance4_export | 1536 | 98304 |

Dot outputs count scalar saved-array contractions, including packed export rows later masked to zero. Multiply-add terms count pair products entering those sums, excluding bias additions. Instrumented scalar-summary calls are implementation-dependent function calls, not primitive FLOPs. No claim of zero arithmetic or absent saved linear-decoder computation is made.

Actual main counters are zero for model/encoder/decoder/head/module/neural forwards, fits, head fits, training updates, held-out tensor reads, old V9 raw reads/reductions and V8/old-adjoint/learning-trust replay. Saved neural predictions are read; no neural weight forward is run. The old V9 cached subtrees retain their original687825-check independent audit authority, not a newly rerun V9 audit.

The main binds25 numeric roles,34 hash-only guards,12 metadata contexts and its exact admitted card. Its twelve metadata copies and pre-execution alias preserve the prior completed109-checkpoint V11 receipt and unchanged baseline/catalog context; before/after source and guard evidence must remain exact through completion. All4.2MB of main diagnosis is saved evidence, not a new encoder artifact. No checkpoint or previous result is replaced.

## Interpretation and next decision

The observed TRAIN pattern is mixed readout/representation redistribution: three linear improvements, five task/head regressions, less favorable reconstruction and concentration that does not establish missing capacity. It supplies no single demonstrated bottleneck. The new moment captures add transparency about the two actual training forwards; they do not identify the cause of known-VALIDATION changes. In-sample performance, three head fits and one history/split cannot establish generalization or seed robustness.

The arithmetic audit is now complete. The tentative next design discussion is how reconstruction supervision allocates information to the native export and preserves geometry usable by the unchanged small heads. Separate parent/residual supervision and a self-supervised representation-consistency objective were previously discussed but deferred; they remain unselected questions, not conclusions from these associations. Either would add objective/compute assumptions requiring a separately frozen declaration. These saved data do not rank them or choose a coefficient, width, pooling operator or encoder.

Do not duplicate the already closed V10 live-joint-NLL route, V7 clipping-cap experiment, V9 blend rescue or V11 mean/variance-row rescue. Keeping every protected scope matters more than optimizing one reconstruction scalar. Exact V11 remains closed without coefficient/seed/rate/budget/head/checkpoint rescue. Root/peer report and completion review is the immediate remaining documentary step; any new scientific candidate requires a distinct prospective source/card and the current strengthened22-condition protocol. Preserve every historical leader and original pass count. TRAIN diagnosis does not rescore or promote candidates; V3 remains baseline and sole DEV encoder.

## Exact sources and navigation

- Main result: [diagnosis.json](causal_v9_v11_train_diagnosis/diagnosis.json),4218354 bytes, SHA256 `62f80b93ef4ad7b2a2a994832ab3f2f6921e09f37e96aca441ec283064f3b742`; actual producer status `passed_readonly_saved_TRAIN_diagnosis`.
- Admitted main card: [CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.ADMITTED.json](CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.ADMITTED.json),63742 bytes, SHA256 `9efb9c5bda49abbc8049e85664bfa58634d4a9dfbe286b6f2f44f620eb52eb8a`.
- Preserved field map/outline: [report-outline-001.PROPOSED.md](causal_v9_v11_train_diagnosis/report-outline-001.PROPOSED.md),23587 bytes, SHA256 `3dd4dd172ed07aa7630d0703e5b7beffe0fa1bc82c7831237c214bfa08c665ab`.
- [Reporting standard1.2](RESULTS_REPORTING_STANDARD.md), [V11 established quality report](CAUSAL_FRESH_V11_RESULTS.md), [V9 established quality report](CAUSAL_FRESH_V9_RESULTS.md), [current prospective retention protocol](clean_causal_encoder_search_goal_v11_retention.md).

Host report preparation decoded this already computed saved result and copied/formatted existing scalar fields. It performed no tensor/array reductions, score recomputation, contractions, head/model call, fit, update, helper, schema, AST or Docker operation. No scientific source, card, runner or live baseline metadata was changed. The new final report adds actual audit receipt metadata and saved per-repetition fields while preserving the preliminary bytes. Documentary completion remains a separate metadata review.

## Every saved readout repetition and physical error

These tables copy all three saved fits for both stages and versions. Method labels distinguish saved0/2048 stages. Deterministic ridge repeats retain their recorded values; they are not three independent ridge or encoder experiments. No best fit is selected. Units refer to physical task prediction errors; standardized reconstruction errors stay in their separate panel. Coverage is112/112 for each head/repetition shown. Full precision remains in the source JSON; display uses ten significant digits.

### Past volatility — all repetitions

Dataset: **MKT-HIST-2** · standard deviation of supported adjacent close log returns in recent128 · **complexity4/5, ordinal** · saved TRAIN112, in-sample. Physical MAE/RMSE unit: **logreturn**.

| Method | Repetition | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 2701 | 0.8681049024 | 0.9473639558 | 0.0001357620951 | 0.0001827978634 | 8.374444037E-05 | 0.0001154777756 |
| CPB-v9.market-01.init | 2802 | 0.8681049024 | 0.9164209768 | 0.0001357620951 | 0.0001827978634 | 0.0001007397256 | 0.0001455142194 |
| CPB-v9.market-01.init | 2903 | 0.8681049024 | 0.9401693221 | 0.0001357620951 | 0.0001827978634 | 8.969779312E-05 | 0.0001231172058 |
| CPB-v11.market-01.init | 2701 | 0.8704379066 | 0.9480124223 | 0.0001387027591 | 0.0001811739557 | 8.606292762E-05 | 0.0001147642386 |
| CPB-v11.market-01.init | 2802 | 0.8704379066 | 0.9248865428 | 0.0001387027591 | 0.0001811739557 | 9.910184037E-05 | 0.0001379480812 |
| CPB-v11.market-01.init | 2903 | 0.8704379066 | 0.918455954 | 0.0001387027591 | 0.0001811739557 | 0.0001003160156 | 0.0001437318172 |
| CPB-v9.market-01 | 2701 | 0.990979636 | 0.9519440313 | 3.828330273E-05 | 4.780450429E-05 | 8.344023794E-05 | 0.0001103393589 |
| CPB-v9.market-01 | 2802 | 0.990979636 | 0.9686011417 | 3.828330273E-05 | 4.780450429E-05 | 6.93494579E-05 | 8.918949739E-05 |
| CPB-v9.market-01 | 2903 | 0.990979636 | 0.9628055703 | 3.828330273E-05 | 4.780450429E-05 | 7.379219978E-05 | 9.707239406E-05 |
| CPB-v11.market-01 | 2701 | 0.9915892992 | 0.9397402825 | 3.717972113E-05 | 4.616075248E-05 | 8.997031077E-05 | 0.0001235578476 |
| CPB-v11.market-01 | 2802 | 0.9915892992 | 0.9648279068 | 3.717972113E-05 | 4.616075248E-05 | 6.831640848E-05 | 9.43965009E-05 |
| CPB-v11.market-01 | 2903 | 0.9915892992 | 0.956108661 | 3.717972113E-05 | 4.616075248E-05 | 7.902581107E-05 | 0.0001054499075 |

### Past trend — all repetitions

Dataset: **MKT-HIST-2** · supported adjacent close log-return trend in recent128 · **complexity4/5, ordinal** · saved TRAIN112, in-sample. Physical MAE/RMSE unit: **logreturn/tick**.

| Method | Repetition | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 2701 | 0.7078987245 | 0.9592748549 | 3.61331303E-05 | 4.512614499E-05 | 1.269176249E-05 | 1.684972542E-05 |
| CPB-v9.market-01.init | 2802 | 0.7078987245 | 0.9701443191 | 3.61331303E-05 | 4.512614499E-05 | 1.140190942E-05 | 1.442696566E-05 |
| CPB-v9.market-01.init | 2903 | 0.7078987245 | 0.9480343163 | 3.61331303E-05 | 4.512614499E-05 | 1.429618367E-05 | 1.903355146E-05 |
| CPB-v11.market-01.init | 2701 | 0.6937123027 | 0.9472332648 | 3.566489438E-05 | 4.620896957E-05 | 1.313720415E-05 | 1.917969161E-05 |
| CPB-v11.market-01.init | 2802 | 0.6937123027 | 0.93154477 | 3.566489438E-05 | 4.620896957E-05 | 1.619516159E-05 | 2.184564262E-05 |
| CPB-v11.market-01.init | 2903 | 0.6937123027 | 0.9544611578 | 3.566489438E-05 | 4.620896957E-05 | 1.386157443E-05 | 1.781773519E-05 |
| CPB-v9.market-01 | 2701 | 0.9075641665 | 0.958519908 | 1.949013121E-05 | 2.538526425E-05 | 1.301889172E-05 | 1.70051851E-05 |
| CPB-v9.market-01 | 2802 | 0.9075641665 | 0.9500248002 | 1.949013121E-05 | 2.538526425E-05 | 1.307000305E-05 | 1.866546344E-05 |
| CPB-v9.market-01 | 2903 | 0.9075641665 | 0.9488396743 | 1.949013121E-05 | 2.538526425E-05 | 1.416545635E-05 | 1.888548567E-05 |
| CPB-v11.market-01 | 2701 | 0.9110286797 | 0.9409886975 | 1.887022258E-05 | 2.490499881E-05 | 1.457169087E-05 | 2.028285603E-05 |
| CPB-v11.market-01 | 2802 | 0.9110286797 | 0.9277573259 | 1.887022258E-05 | 2.490499881E-05 | 1.535159796E-05 | 2.244183756E-05 |
| CPB-v11.market-01 | 2903 | 0.9110286797 | 0.9459098363 | 1.887022258E-05 | 2.490499881E-05 | 1.457943752E-05 | 1.94187225E-05 |

### Past activity — all repetitions

Dataset: **MKT-HIST-2** · mean log1p(nonnegative observed base volume) in recent128 · **complexity4/5, ordinal** · saved TRAIN112, in-sample. Physical MAE/RMSE unit: **log1p(base_volume)**.

| Method | Repetition | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 2701 | 0.9382282393 | 0.9600611918 | 0.1178825777 | 0.1543277611 | 0.09582546921 | 0.1240928092 |
| CPB-v9.market-01.init | 2802 | 0.9382282393 | 0.9482640856 | 0.1178825777 | 0.1543277611 | 0.1043937188 | 0.1412359094 |
| CPB-v9.market-01.init | 2903 | 0.9382282393 | 0.9605385018 | 0.1178825777 | 0.1543277611 | 0.09582267762 | 0.1233490618 |
| CPB-v11.market-01.init | 2701 | 0.9264675744 | 0.9604736197 | 0.132608976 | 0.1683792272 | 0.09674711797 | 0.1234504247 |
| CPB-v11.market-01.init | 2802 | 0.9264675744 | 0.9565085167 | 0.132608976 | 0.1683792272 | 0.1037722261 | 0.129494456 |
| CPB-v11.market-01.init | 2903 | 0.9264675744 | 0.9753577905 | 0.132608976 | 0.1683792272 | 0.07216140524 | 0.09747402243 |
| CPB-v9.market-01 | 2701 | 0.9753968809 | 0.9752602014 | 0.07655912449 | 0.09739667937 | 0.07772520976 | 0.09766684214 |
| CPB-v9.market-01 | 2802 | 0.9753968809 | 0.9733753218 | 0.07655912449 | 0.09739667937 | 0.08165318163 | 0.1013190826 |
| CPB-v9.market-01 | 2903 | 0.9753968809 | 0.9708839513 | 0.07655912449 | 0.09739667937 | 0.08011155984 | 0.1059534942 |
| CPB-v11.market-01 | 2701 | 0.9676749422 | 0.9666138639 | 0.08901114748 | 0.1116397154 | 0.09054964755 | 0.1134572219 |
| CPB-v11.market-01 | 2802 | 0.9676749422 | 0.9702249784 | 0.08901114748 | 0.1116397154 | 0.08762217514 | 0.1071457892 |
| CPB-v11.market-01 | 2903 | 0.9676749422 | 0.9730590895 | 0.08901114748 | 0.1116397154 | 0.08254516395 | 0.101919011 |

### Past relation — all repetitions

Dataset: **MKT-HIST-2** · eligible-pair Pearson correlation of supported adjacent close returns in recent128 · **complexity4/5, ordinal** · saved TRAIN112, in-sample. Physical MAE/RMSE unit: **correlation**.

| Method | Repetition | Linear R² | Neural R² | Linear MAE | Linear RMSE | Neural MAE | Neural RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 2701 | 0.6549993607 | 0.8921401614 | 0.0475858637 | 0.05966124879 | 0.02576022723 | 0.03335894234 |
| CPB-v9.market-01.init | 2802 | 0.6549993607 | 0.9544438088 | 0.0475858637 | 0.05966124879 | 0.01700984778 | 0.02167984102 |
| CPB-v9.market-01.init | 2903 | 0.6549993607 | 0.8840083936 | 0.0475858637 | 0.05966124879 | 0.02764398335 | 0.03459359297 |
| CPB-v11.market-01.init | 2701 | 0.6710415617 | 0.8962817312 | 0.04625253306 | 0.05825764237 | 0.02456989577 | 0.03271222006 |
| CPB-v11.market-01.init | 2802 | 0.6710415617 | 0.9124124261 | 0.04625253306 | 0.05825764237 | 0.02334426332 | 0.03006101484 |
| CPB-v11.market-01.init | 2903 | 0.6710415617 | 0.8714454537 | 0.04625253306 | 0.05825764237 | 0.02788651013 | 0.03641884007 |
| CPB-v9.market-01 | 2701 | 0.7176859025 | 0.869802951 | 0.04315089728 | 0.05396953451 | 0.02849304531 | 0.03665075791 |
| CPB-v9.market-01 | 2802 | 0.7176859025 | 0.8999899724 | 0.04315089728 | 0.05396953451 | 0.02568035272 | 0.03212211726 |
| CPB-v9.market-01 | 2903 | 0.7176859025 | 0.8504819182 | 0.04315089728 | 0.05396953451 | 0.03104264049 | 0.03927618057 |
| CPB-v11.market-01 | 2701 | 0.7345649307 | 0.8593115988 | 0.04309224556 | 0.05233130202 | 0.02879099296 | 0.03809882103 |
| CPB-v11.market-01 | 2802 | 0.7345649307 | 0.8898075441 | 0.04309224556 | 0.05233130202 | 0.02699234473 | 0.03371772939 |
| CPB-v11.market-01 | 2903 | 0.7345649307 | 0.862334454 | 0.04309224556 | 0.05233130202 | 0.0296382648 | 0.03768729962 |

## Actual audit, costs and durable closure

The independent audit newly recomputes two V11 native/readout/query/mean4 stages, all112 per-task/readout/reconstruction associations, the16384 moment-subgroup rows with fixed blocks and both actual ordinary/contextual moment/export captures. It imports no producer algorithm. It compares all newly declared V11 outputs and the exact manually counted affine output/product quantities. The four V9 cached subtrees are carried under their original687825-check audit; no second old-V9 numerical reduction or256-finite-difference proof is claimed. Existing V11 causal/typed-resume/local-adjoint/gauge/Hadamard/learning proofs remain preserved context.

Dataset: **MKT-HIST-2** · saved TRAIN diagnosis and independent arithmetic audit · **complexity4/5, ordinal** · existing container CPU scalar arithmetic, no new encoder training or GPU computation. Body and recorder seconds are separate scopes and are not added or offered as a matched performance benchmark.

| Operation | Body seconds | Recorder seconds | New encoder updates | New head fits |
| --- | ---: | ---: | ---: | ---: |
| Saved TRAIN diagnosis | 0.69284411 | 1.132756622 | 0 | 0 |
| Independent saved TRAIN audit | 0.674598772 | 0.726899459 | 0 | 0 |

The main source capsule has19 payload files; its full run mirror contains33 files. The independent audit mirror contains19 files. Both saved mirror receipts report exact source before/after and complete file byte/hash equality. Runtime zero model/head/fit/update counters are separate from the explicit performed manual contractions. The audit matches the five affine-family counter objects and checks producer summary calls as implementation-dependent counts, not equal FLOPs.

## Preserved metadata failures — no scientific replay

Two host metadata defects were stopped and repaired before the respective scientific phases. Their whole original files and failure receipts remain preserved:

1. The first prospective-card metadata call stopped on PowerShell's read-only `$Host` variable before any card JSON/Markdown existed. Root preserved the original generator, provenance, source-readiness and failure, then repaired four local variable tokens to `taskHostKey` and obtained independent source/card review. The source-repair receipt also retains the initial expected-token-count guard refusal before source edit. No Docker, AST, schema or numeric body ran for that metadata failure.
2. The original preflight passed its1436 checks, but `Group-Object path` collapsed the displayed source inventory to one entry. The next Stage invocation refused the missing exact `archive_reader.py` pin before workspace creation, Docker exec, project AST, schema or numerical execution; its one Docker inspect was metadata only. The full34152-byte original preflight and1436-byte Stage-refusal receipt remain preserved. Root independently reconstructed the28 literal-path inventory pins and appended explicit provenance, preserving every other original check/binding. An independently reviewed external42166-byte amendment was applied byte-exactly; held helper/source/card bytes stayed frozen and the preflight was not rerun. The next Stage received the new external preflight hash.

These are metadata preparation/transport failures, not failed diagnostic arithmetic or numerical reruns. The producer and independent arithmetic auditor each executed once. Their evidence was not regenerated to repair reporting metadata. Failed generations, old declarations and prospective/admitted distinctions remain readable.

Exact provenance anchors:

- `$Host` failure: `output/engineering/causal-v9-v11-train-diagnosis-workflow-source-preparation/attempt-001/main-repair-002/failure.json`,749 bytes, SHA256 `c420086c18f9738f98c2a111aa8bdd58aaf320434bc1a0272b07b7a6894513bf`; original whole source copies and `before-manifest.json` remain beside it.
- Original preflight: `output/engineering/causal-v9-v11-train-diagnosis-preflight-inventory-amendment/attempt-001/preflight-review.original.json`,34152 bytes, SHA256 `dac2938d0198e126f3aa859988ca91e76220f1ffa7e4041127651ed926f510b2`.
- Stage refusal: same amendment directory `stage-refusal.json`,1436 bytes, SHA256 `0a9ed862277fc3c5fde830ed581a2fb1c4858cabd7a57a5a23f85e336c2a17db`.
- Reviewed/applied external preflight: [preflight-review.json](causal_v9_v11_train_diagnosis/preflight-review.json),42166 bytes, SHA256 `03741fdc72c33d29a771278b4a4f3629300ff21048549ca0c376527d33b19fcb`. Its frozen nested proposal-state string is historical provenance; [application receipt](causal_v9_v11_train_diagnosis/preflight-inventory-amendment-application-001.json),2398 bytes, SHA256 `18534e461175e62d192e60e90840dcb7ee6d2c0c81ae561db0a1dbf0c9bf9018`, supplies the completed application separately.

## Additional exact result anchors

- [Independent arithmetic audit](causal_v9_v11_train_diagnosis/independent-audit.json),48403 bytes, SHA256 `70697acc72fd010b48431516cb15dc1537b615cde5835ed90536cd4de625bc8c`, actual status `passed`,68264 checks, zero mismatches.
- [Independent execution](causal_v9_v11_train_diagnosis/independent-audit-execution.json),6876 bytes, SHA256 `ee966ec7e8c4d06627053841a957deddf31d41b5a068b4c6716011f0bd05b867`.
- [Independent complete19-file mirror](causal_v9_v11_train_diagnosis/independent-audit-mirror.json),9657 bytes, SHA256 `c237f09563c48d45b9641fd28f5c886418db9bc704f308ec510452a62483dd28`.
- [Main execution](causal_v9_v11_train_diagnosis/execution.json),3111 bytes, SHA256 `7c0c613e6e745c8ecd0efb97f191bd0661a33818126f518125e7bc4eb4a69384`, actual status `passed`, phase `Run`, exit0, once_only true.
- [Main complete33-file mirror](causal_v9_v11_train_diagnosis/mirror.json),6844 bytes, SHA256 `27d2b9151bf2c45d3409ff49d58ac300fefecde8b2900b5d4b998d8ee701cadb`.
- [Independent ADMITTED card](causal_v9_v11_train_diagnosis/independent-audit-card.ADMITTED.json),7503 bytes, SHA256 `6648ee62237b9393d15b7a11f7fd820cb5a0769ab80340f4a0952568b5b6bd64`.
- [Preserved preliminary human report](causal_v9_v11_train_diagnosis/human-report-preliminary-001.md),22614 bytes, SHA256 `783d9abcd62b029b87b3620b6297afedd72e264152c41e1126e7765f76fc48c4`.

This report adds no model, head, training, extraction, new metric reduction or old scientific replay. It does not claim a completed documentary join before that separate root/peer operation. Main card, all ten scientific sources, ten current helpers, ledger, catalog, current future protocol and completed109-checkpoint preservation receipt were rehashed unchanged before and after report preparation. Every V3–V11 scoped leader and all historical checkpoint/result bytes remain protected under their original records.
