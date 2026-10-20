# V15 objective comparison: six-run results

Execution is **complete and valid** for all six declared trajectories and their development/confirmation audits. The exact prospective screen verdict is **mixed**; the saved promising rule does not pass. Work completion and scientific success are separate judgments. No encoder is automatically promoted or selected for Mityane.

V15 improves February **linear and neural-mean activity R² and descriptive fixed-query MAE/NLL in all three pairs**. Linear volatility, trend and relation each improve in one pair; neural-mean trend falls in all three. January reconstruction also improves across all three pairs, while January trend readouts decline. The paired tables below preserve these gains and regressions together.

The [family-owned live ranking](../../../encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md) retains six new scoped January highs from this study: V15 TRAIN MAE and synthetic-scaler intact linear timing; fresh V4-control neural activity, TRAIN NLL, synthetic-scaler deleted neural timing and market-scaler intact neural timing. Control-arm records are distinct from V15 gains. All earlier leaders remain recorded with their original scope.

This document formats saved values from the [authoritative machine report](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-001.json). Numbers are displayed to six decimal places; full-precision JSON, original criterion objects, null/abstention states and individual results remain authoritative. No readout, metric, median, interval or scientific verdict is recomputed by this renderer.

The [actual report card](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-inputs-001.json) binds all18 phase receipts and12 independent arithmetic wrappers. See the [frozen protocol](PROTOCOL.md) and [compact closeout record (written separately)](../../records/CPB-v15.objective-001.json).

## Prospective February screen

All differences below are saved candidate minus paired fresh V4 control linear R². Volatility and activity are primary; trend has the predeclared median tolerance of -0.01. Relation is descriptive and does not enter the screen.

| Encoder seed | Δ volatility | Δ activity | Δ trend | Δ relation (descriptive) |
| --- | --- | --- | --- | --- |
| 945101 | -0.203014 | +0.228295 | +0.007876 | -1.862306 |
| 945102 | +0.105940 | +0.186530 | -0.218994 | +0.548564 |
| 945103 | -0.078143 | +0.122493 | -0.098598 | -3.178873 |

| Saved screen target | Median Δ | Range Δ | Improving pairs | Role |
| --- | --- | --- | --- | --- |
| Volatility | -0.078143 | [-0.203014, +0.105940] | 1/3 | positive median and ≥2/3 improving |
| Activity | +0.186530 | [+0.122493, +0.228295] | 3/3 | positive median and ≥2/3 improving |
| Trend | -0.098598 | [-0.218994, +0.007876] | 1/3 | median ≥ -0.01 |

| Saved screen condition | Verdict |
| --- | --- |
| Primary targets | FAIL |
| Trend tolerance | FAIL |
| All candidates retain eight January timing conditions | FAIL |

Saved reasons: `past_volatility requires positive median and at least two improving pairs`; `February linear trend median difference is below -0.01 or undefined`; `at least one candidate fails a frozen hard timing condition`.

## January gains and tradeoffs remain visible

All three January pairs showed lower candidate TRAIN/known-VALIDATION query MAE and known-VALIDATION Gaussian NLL. Candidate linear trend R² fell in all three pairs. The third pair’s activity gain is retained alongside losses elsewhere; these development observations do not substitute for the prospective February screen.

| Seed | TRAIN MAE control / V15 | Known-VAL MAE control / V15 | Known-VAL NLL control / V15 | January linear trend Δ | January linear activity Δ | January neural activity Δ |
| --- | --- | --- | --- | --- | --- | --- |
| 945101 | 0.247850 / 0.245746 | 0.399577 / 0.354997 | 1.278563 / 0.508550 | -0.112681 | -0.019265 | -0.072247 |
| 945102 | 0.264637 / 0.240037 | 0.383207 / 0.375532 | 0.755391 / 0.429499 | -0.242910 | -0.005877 | -0.184165 |
| 945103 | 0.258744 / 0.248416 | 0.475570 / 0.393124 | 1.103325 / 0.732632 | -0.044474 | +0.111402 | +0.003055 |

The seed945102 candidate TRAIN query MAE is **0.240037**, against the frozen V7-derived floor **0.241805** (saved predicate: PASS). Its full-precision saved value is in the machine report. This scoped gain is retained without choosing that seed as the study winner.

Fresh V4 control seed945101 records TRAIN Gaussian NLL **-0.410369**, below the [saved V7 reference](../../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/results/causal_fresh_v7/training-summary.json) of **-0.40882727825899934**. This is the lowest observed final-checkpoint TRAIN NLL in the saved clean-causal V3–V14 references and these six new runs on the unchanged query. It is descriptive and adds no protection gate or seed selection.

## Absolute final readout R²

### January known validation (development)

| Encoder seed | Arm | Past-window task | Linear R² | Neural mean R² | Coverage |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | Volatility | 0.762364 | 0.445879 | 1.000000 |
| 945101 | control | Trend | 0.732087 | 0.099855 | 1.000000 |
| 945101 | control | Activity | 0.874762 | 0.792420 | 1.000000 |
| 945101 | control | Relation | 0.430174 | 0.088102 | 1.000000 |
| 945101 | candidate | Volatility | 0.536357 | 0.411758 | 1.000000 |
| 945101 | candidate | Trend | 0.619406 | -0.268647 | 1.000000 |
| 945101 | candidate | Activity | 0.855497 | 0.720172 | 1.000000 |
| 945101 | candidate | Relation | 0.469471 | -0.066863 | 1.000000 |
| 945102 | control | Volatility | 0.749644 | 0.635973 | 1.000000 |
| 945102 | control | Trend | 0.814244 | 0.414591 | 1.000000 |
| 945102 | control | Activity | 0.840677 | 0.756138 | 1.000000 |
| 945102 | control | Relation | 0.494371 | 0.064325 | 1.000000 |
| 945102 | candidate | Volatility | 0.626864 | 0.236982 | 1.000000 |
| 945102 | candidate | Trend | 0.571334 | 0.070592 | 1.000000 |
| 945102 | candidate | Activity | 0.834800 | 0.571972 | 1.000000 |
| 945102 | candidate | Relation | 0.489265 | 0.186924 | 1.000000 |
| 945103 | control | Volatility | 0.687369 | 0.322828 | 1.000000 |
| 945103 | control | Trend | 0.802083 | 0.238552 | 1.000000 |
| 945103 | control | Activity | 0.795132 | 0.629837 | 1.000000 |
| 945103 | control | Relation | 0.164000 | -0.095595 | 1.000000 |
| 945103 | candidate | Volatility | 0.690095 | 0.468324 | 1.000000 |
| 945103 | candidate | Trend | 0.757609 | -0.112153 | 1.000000 |
| 945103 | candidate | Activity | 0.906534 | 0.632892 | 1.000000 |
| 945103 | candidate | Relation | 0.060133 | -0.061140 | 1.000000 |

### February confirmation

| Encoder seed | Arm | Past-window task | Linear R² | Neural mean R² | Coverage |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | Volatility | 0.699687 | 0.218620 | 1.000000 |
| 945101 | control | Trend | 0.290588 | -0.147026 | 1.000000 |
| 945101 | control | Activity | 0.589823 | 0.350338 | 1.000000 |
| 945101 | control | Relation | -1.108819 | -1.115418 | 1.000000 |
| 945101 | candidate | Volatility | 0.496673 | 0.098630 | 1.000000 |
| 945101 | candidate | Trend | 0.298464 | -0.436559 | 1.000000 |
| 945101 | candidate | Activity | 0.818118 | 0.525858 | 1.000000 |
| 945101 | candidate | Relation | -2.971125 | -0.913773 | 1.000000 |
| 945102 | control | Volatility | 0.652745 | 0.246331 | 1.000000 |
| 945102 | control | Trend | 0.346180 | -0.438837 | 1.000000 |
| 945102 | control | Activity | 0.655552 | 0.555476 | 1.000000 |
| 945102 | control | Relation | -1.874316 | -2.553742 | 1.000000 |
| 945102 | candidate | Volatility | 0.758685 | -0.048022 | 1.000000 |
| 945102 | candidate | Trend | 0.127186 | -0.645956 | 1.000000 |
| 945102 | candidate | Activity | 0.842081 | 0.635421 | 1.000000 |
| 945102 | candidate | Relation | -1.325752 | -0.951631 | 1.000000 |
| 945103 | control | Volatility | 0.790686 | 0.308937 | 1.000000 |
| 945103 | control | Trend | 0.491474 | 0.013534 | 1.000000 |
| 945103 | control | Activity | 0.400313 | 0.249503 | 1.000000 |
| 945103 | control | Relation | -1.248019 | -0.637141 | 1.000000 |
| 945103 | candidate | Volatility | 0.712543 | 0.534598 | 1.000000 |
| 945103 | candidate | Trend | 0.392876 | -1.205357 | 1.000000 |
| 945103 | candidate | Activity | 0.522806 | 0.396074 | 1.000000 |
| 945103 | candidate | Relation | -4.426892 | -1.543608 | 1.000000 |

## Final fixed-query reconstruction

MAE and Gaussian NLL are separate saved descriptive scores; lower is better. February reconstruction has no gate. The V15 training objective is the composite Huber1 plus variance calibration, not its reported Gaussian NLL.

| Seed | Arm | Population | MAE | Gaussian NLL | Valid rows | Waveform audit scope |
| --- | --- | --- | --- | --- | --- | --- |
| 945101 | control | January TRAIN | 0.247850 | -0.410369 | 112 | first8 / full112 saved row means |
| 945101 | control | January known VALIDATION | 0.399577 | 1.278563 | 32 | first8 / full32 saved row means |
| 945101 | control | February confirmation | 0.802806 | 9.337928 | 32 | all32 waveforms |
| 945101 | candidate | January TRAIN | 0.245746 | -0.385437 | 112 | first8 / full112 saved row means |
| 945101 | candidate | January known VALIDATION | 0.354997 | 0.508550 | 32 | first8 / full32 saved row means |
| 945101 | candidate | February confirmation | 0.714172 | 5.154861 | 32 | all32 waveforms |
| 945102 | control | January TRAIN | 0.264637 | -0.365964 | 112 | first8 / full112 saved row means |
| 945102 | control | January known VALIDATION | 0.383207 | 0.755391 | 32 | first8 / full32 saved row means |
| 945102 | control | February confirmation | 0.835078 | 10.078783 | 32 | all32 waveforms |
| 945102 | candidate | January TRAIN | 0.240037 | -0.367805 | 112 | first8 / full112 saved row means |
| 945102 | candidate | January known VALIDATION | 0.375532 | 0.429499 | 32 | first8 / full32 saved row means |
| 945102 | candidate | February confirmation | 0.723069 | 4.969757 | 32 | all32 waveforms |
| 945103 | control | January TRAIN | 0.258744 | -0.386967 | 112 | first8 / full112 saved row means |
| 945103 | control | January known VALIDATION | 0.475570 | 1.103325 | 32 | first8 / full32 saved row means |
| 945103 | control | February confirmation | 0.903748 | 8.747334 | 32 | all32 waveforms |
| 945103 | candidate | January TRAIN | 0.248416 | -0.390166 | 112 | first8 / full112 saved row means |
| 945103 | candidate | January known VALIDATION | 0.393124 | 0.732632 | 32 | first8 / full32 saved row means |
| 945103 | candidate | February confirmation | 0.833747 | 6.845502 | 32 | all32 waveforms |

## Frozen January protection report

Each seed is checked separately on final January development; original criterion objects, master/repetition means, strict inequalities and positivity quantifiers are unchanged. Counts are not comparable to V4 historical19/22 under its earlier floors.

Counts below summarize saved predicate flags only. Every original rule, floor, candidate value and verdict is retained in each22-row matrix; rounding here does not change a saved predicate. TRAIN Gaussian NLL is not an additional gate.

| Seed | Arm | Saved passed / total | Timing passed /8 | Full declared support | All22 pass |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | 4/22 | 1/8 | yes | no |
| 945101 | candidate | 2/22 | 1/8 | yes | no |
| 945102 | control | 4/22 | 1/8 | yes | no |
| 945102 | candidate | 3/22 | 0/8 | yes | no |
| 945103 | control | 1/22 | 0/8 | yes | no |
| 945103 | candidate | 2/22 | 1/8 | yes | no |

### Seed 945101 — control

| Criterion ID | Original rule | Floor | Saved candidate value | Saved verdict |
| --- | --- | --- | --- | --- |
| `past_volatility:linear_r2` | candidate + 1e-12 >= floor | 0.964673 | 0.762364 | FAIL |
| `past_volatility:neural_mean_r2` | candidate + 1e-12 >= floor | 0.735969 | 0.445879 | FAIL |
| `past_trend:linear_r2` | candidate + 1e-12 >= floor | 0.907213 | 0.732087 | FAIL |
| `past_trend:neural_mean_r2` | candidate + 1e-12 >= floor | 0.569875 | 0.099855 | FAIL |
| `past_activity:linear_r2` | candidate + 1e-12 >= floor | 0.909439 | 0.874762 | FAIL |
| `past_activity:neural_mean_r2` | candidate + 1e-12 >= floor | 0.762544 | 0.792420 | PASS |
| `past_relation:linear_r2` | candidate + 1e-12 >= floor | 0.622101 | 0.430174 | FAIL |
| `past_relation:neural_mean_r2` | candidate + 1e-12 >= floor | 0.457197 | 0.088102 | FAIL |
| `synthetic_train:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.989583 | 0.984375 | FAIL |
| `synthetic_train:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.984809 | 0.975260 | FAIL |
| `synthetic_train:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.958333 | 0.947917 | FAIL |
| `synthetic_train:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.945747 | 0.948351 | PASS |
| `market_scaler:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.998698 | 0.993490 | FAIL |
| `market_scaler:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.992622 | 0.988715 | FAIL |
| `market_scaler:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.972656 | 0.955729 | FAIL |
| `market_scaler:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.956163 | 0.940104 | FAIL |
| `train:standardized_MAE` | candidate <= floor + 1e-12 | 0.241805 | 0.247850 | FAIL |
| `validation:standardized_MAE` | candidate <= floor + 1e-12 | 0.308317 | 0.399577 | FAIL |
| `known_validation:Gaussian_NLL_F64` | candidate < floor - 1e-12 | -0.045381 | 1.278563 | FAIL |
| `past_trend:all3_neural_R2_positive` | all three final neural trend R2 values > 0 | 0.000000 | [0.491896, -0.232065, 0.039733] | FAIL |
| `past_relation:linear_r2:positive` | candidate > 0 | 0.000000 | 0.430174 | PASS |
| `past_relation:neural_mean_r2:positive` | candidate > 0 | 0.000000 | 0.088102 | PASS |

### Seed 945101 — candidate

| Criterion ID | Original rule | Floor | Saved candidate value | Saved verdict |
| --- | --- | --- | --- | --- |
| `past_volatility:linear_r2` | candidate + 1e-12 >= floor | 0.964673 | 0.536357 | FAIL |
| `past_volatility:neural_mean_r2` | candidate + 1e-12 >= floor | 0.735969 | 0.411758 | FAIL |
| `past_trend:linear_r2` | candidate + 1e-12 >= floor | 0.907213 | 0.619406 | FAIL |
| `past_trend:neural_mean_r2` | candidate + 1e-12 >= floor | 0.569875 | -0.268647 | FAIL |
| `past_activity:linear_r2` | candidate + 1e-12 >= floor | 0.909439 | 0.855497 | FAIL |
| `past_activity:neural_mean_r2` | candidate + 1e-12 >= floor | 0.762544 | 0.720172 | FAIL |
| `past_relation:linear_r2` | candidate + 1e-12 >= floor | 0.622101 | 0.469471 | FAIL |
| `past_relation:neural_mean_r2` | candidate + 1e-12 >= floor | 0.457197 | -0.066863 | FAIL |
| `synthetic_train:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.989583 | 0.996094 | PASS |
| `synthetic_train:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.984809 | 0.980035 | FAIL |
| `synthetic_train:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.958333 | 0.946615 | FAIL |
| `synthetic_train:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.945747 | 0.930990 | FAIL |
| `market_scaler:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.998698 | 0.996094 | FAIL |
| `market_scaler:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.992622 | 0.986111 | FAIL |
| `market_scaler:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.972656 | 0.964844 | FAIL |
| `market_scaler:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.956163 | 0.944010 | FAIL |
| `train:standardized_MAE` | candidate <= floor + 1e-12 | 0.241805 | 0.245746 | FAIL |
| `validation:standardized_MAE` | candidate <= floor + 1e-12 | 0.308317 | 0.354997 | FAIL |
| `known_validation:Gaussian_NLL_F64` | candidate < floor - 1e-12 | -0.045381 | 0.508550 | FAIL |
| `past_trend:all3_neural_R2_positive` | all three final neural trend R2 values > 0 | 0.000000 | [-0.579520, -0.133860, -0.092562] | FAIL |
| `past_relation:linear_r2:positive` | candidate > 0 | 0.000000 | 0.469471 | PASS |
| `past_relation:neural_mean_r2:positive` | candidate > 0 | 0.000000 | -0.066863 | FAIL |

### Seed 945102 — control

| Criterion ID | Original rule | Floor | Saved candidate value | Saved verdict |
| --- | --- | --- | --- | --- |
| `past_volatility:linear_r2` | candidate + 1e-12 >= floor | 0.964673 | 0.749644 | FAIL |
| `past_volatility:neural_mean_r2` | candidate + 1e-12 >= floor | 0.735969 | 0.635973 | FAIL |
| `past_trend:linear_r2` | candidate + 1e-12 >= floor | 0.907213 | 0.814244 | FAIL |
| `past_trend:neural_mean_r2` | candidate + 1e-12 >= floor | 0.569875 | 0.414591 | FAIL |
| `past_activity:linear_r2` | candidate + 1e-12 >= floor | 0.909439 | 0.840677 | FAIL |
| `past_activity:neural_mean_r2` | candidate + 1e-12 >= floor | 0.762544 | 0.756138 | FAIL |
| `past_relation:linear_r2` | candidate + 1e-12 >= floor | 0.622101 | 0.494371 | FAIL |
| `past_relation:neural_mean_r2` | candidate + 1e-12 >= floor | 0.457197 | 0.064325 | FAIL |
| `synthetic_train:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.989583 | 0.980469 | FAIL |
| `synthetic_train:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.984809 | 0.979167 | FAIL |
| `synthetic_train:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.958333 | 0.946615 | FAIL |
| `synthetic_train:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.945747 | 0.937066 | FAIL |
| `market_scaler:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.998698 | 0.996094 | FAIL |
| `market_scaler:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.992622 | 0.993056 | PASS |
| `market_scaler:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.972656 | 0.954427 | FAIL |
| `market_scaler:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.956163 | 0.937066 | FAIL |
| `train:standardized_MAE` | candidate <= floor + 1e-12 | 0.241805 | 0.264637 | FAIL |
| `validation:standardized_MAE` | candidate <= floor + 1e-12 | 0.308317 | 0.383207 | FAIL |
| `known_validation:Gaussian_NLL_F64` | candidate < floor - 1e-12 | -0.045381 | 0.755391 | FAIL |
| `past_trend:all3_neural_R2_positive` | all three final neural trend R2 values > 0 | 0.000000 | [0.453764, 0.153479, 0.636530] | PASS |
| `past_relation:linear_r2:positive` | candidate > 0 | 0.000000 | 0.494371 | PASS |
| `past_relation:neural_mean_r2:positive` | candidate > 0 | 0.000000 | 0.064325 | PASS |

### Seed 945102 — candidate

| Criterion ID | Original rule | Floor | Saved candidate value | Saved verdict |
| --- | --- | --- | --- | --- |
| `past_volatility:linear_r2` | candidate + 1e-12 >= floor | 0.964673 | 0.626864 | FAIL |
| `past_volatility:neural_mean_r2` | candidate + 1e-12 >= floor | 0.735969 | 0.236982 | FAIL |
| `past_trend:linear_r2` | candidate + 1e-12 >= floor | 0.907213 | 0.571334 | FAIL |
| `past_trend:neural_mean_r2` | candidate + 1e-12 >= floor | 0.569875 | 0.070592 | FAIL |
| `past_activity:linear_r2` | candidate + 1e-12 >= floor | 0.909439 | 0.834800 | FAIL |
| `past_activity:neural_mean_r2` | candidate + 1e-12 >= floor | 0.762544 | 0.571972 | FAIL |
| `past_relation:linear_r2` | candidate + 1e-12 >= floor | 0.622101 | 0.489265 | FAIL |
| `past_relation:neural_mean_r2` | candidate + 1e-12 >= floor | 0.457197 | 0.186924 | FAIL |
| `synthetic_train:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.989583 | 0.988281 | FAIL |
| `synthetic_train:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.984809 | 0.982639 | FAIL |
| `synthetic_train:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.958333 | 0.937500 | FAIL |
| `synthetic_train:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.945747 | 0.930990 | FAIL |
| `market_scaler:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.998698 | 0.994792 | FAIL |
| `market_scaler:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.992622 | 0.987847 | FAIL |
| `market_scaler:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.972656 | 0.950521 | FAIL |
| `market_scaler:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.956163 | 0.924479 | FAIL |
| `train:standardized_MAE` | candidate <= floor + 1e-12 | 0.241805 | 0.240037 | PASS |
| `validation:standardized_MAE` | candidate <= floor + 1e-12 | 0.308317 | 0.375532 | FAIL |
| `known_validation:Gaussian_NLL_F64` | candidate < floor - 1e-12 | -0.045381 | 0.429499 | FAIL |
| `past_trend:all3_neural_R2_positive` | all three final neural trend R2 values > 0 | 0.000000 | [0.258758, -0.387215, 0.340232] | FAIL |
| `past_relation:linear_r2:positive` | candidate > 0 | 0.000000 | 0.489265 | PASS |
| `past_relation:neural_mean_r2:positive` | candidate > 0 | 0.000000 | 0.186924 | PASS |

### Seed 945103 — control

| Criterion ID | Original rule | Floor | Saved candidate value | Saved verdict |
| --- | --- | --- | --- | --- |
| `past_volatility:linear_r2` | candidate + 1e-12 >= floor | 0.964673 | 0.687369 | FAIL |
| `past_volatility:neural_mean_r2` | candidate + 1e-12 >= floor | 0.735969 | 0.322828 | FAIL |
| `past_trend:linear_r2` | candidate + 1e-12 >= floor | 0.907213 | 0.802083 | FAIL |
| `past_trend:neural_mean_r2` | candidate + 1e-12 >= floor | 0.569875 | 0.238552 | FAIL |
| `past_activity:linear_r2` | candidate + 1e-12 >= floor | 0.909439 | 0.795132 | FAIL |
| `past_activity:neural_mean_r2` | candidate + 1e-12 >= floor | 0.762544 | 0.629837 | FAIL |
| `past_relation:linear_r2` | candidate + 1e-12 >= floor | 0.622101 | 0.164000 | FAIL |
| `past_relation:neural_mean_r2` | candidate + 1e-12 >= floor | 0.457197 | -0.095595 | FAIL |
| `synthetic_train:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.989583 | 0.972656 | FAIL |
| `synthetic_train:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.984809 | 0.970920 | FAIL |
| `synthetic_train:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.958333 | 0.925781 | FAIL |
| `synthetic_train:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.945747 | 0.914931 | FAIL |
| `market_scaler:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.998698 | 0.990885 | FAIL |
| `market_scaler:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.992622 | 0.985677 | FAIL |
| `market_scaler:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.972656 | 0.953125 | FAIL |
| `market_scaler:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.956163 | 0.937500 | FAIL |
| `train:standardized_MAE` | candidate <= floor + 1e-12 | 0.241805 | 0.258744 | FAIL |
| `validation:standardized_MAE` | candidate <= floor + 1e-12 | 0.308317 | 0.475570 | FAIL |
| `known_validation:Gaussian_NLL_F64` | candidate < floor - 1e-12 | -0.045381 | 1.103325 | FAIL |
| `past_trend:all3_neural_R2_positive` | all three final neural trend R2 values > 0 | 0.000000 | [0.394190, 0.387101, -0.065634] | FAIL |
| `past_relation:linear_r2:positive` | candidate > 0 | 0.000000 | 0.164000 | PASS |
| `past_relation:neural_mean_r2:positive` | candidate > 0 | 0.000000 | -0.095595 | FAIL |

### Seed 945103 — candidate

| Criterion ID | Original rule | Floor | Saved candidate value | Saved verdict |
| --- | --- | --- | --- | --- |
| `past_volatility:linear_r2` | candidate + 1e-12 >= floor | 0.964673 | 0.690095 | FAIL |
| `past_volatility:neural_mean_r2` | candidate + 1e-12 >= floor | 0.735969 | 0.468324 | FAIL |
| `past_trend:linear_r2` | candidate + 1e-12 >= floor | 0.907213 | 0.757609 | FAIL |
| `past_trend:neural_mean_r2` | candidate + 1e-12 >= floor | 0.569875 | -0.112153 | FAIL |
| `past_activity:linear_r2` | candidate + 1e-12 >= floor | 0.909439 | 0.906534 | FAIL |
| `past_activity:neural_mean_r2` | candidate + 1e-12 >= floor | 0.762544 | 0.632892 | FAIL |
| `past_relation:linear_r2` | candidate + 1e-12 >= floor | 0.622101 | 0.060133 | FAIL |
| `past_relation:neural_mean_r2` | candidate + 1e-12 >= floor | 0.457197 | -0.061140 | FAIL |
| `synthetic_train:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.989583 | 0.993490 | PASS |
| `synthetic_train:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.984809 | 0.982639 | FAIL |
| `synthetic_train:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.958333 | 0.955729 | FAIL |
| `synthetic_train:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.945747 | 0.944010 | FAIL |
| `market_scaler:validation_intact:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.998698 | 0.997396 | FAIL |
| `market_scaler:validation_intact:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.992622 | 0.980035 | FAIL |
| `market_scaler:validation_deleted:linear_head_accuracy` | candidate + 1e-12 >= floor | 0.972656 | 0.963542 | FAIL |
| `market_scaler:validation_deleted:neural_head_accuracy` | candidate + 1e-12 >= floor | 0.956163 | 0.954861 | FAIL |
| `train:standardized_MAE` | candidate <= floor + 1e-12 | 0.241805 | 0.248416 | FAIL |
| `validation:standardized_MAE` | candidate <= floor + 1e-12 | 0.308317 | 0.393124 | FAIL |
| `known_validation:Gaussian_NLL_F64` | candidate < floor - 1e-12 | -0.045381 | 0.732632 | FAIL |
| `past_trend:all3_neural_R2_positive` | all three final neural trend R2 values > 0 | 0.000000 | [0.004404, -0.168022, -0.172843] | FAIL |
| `past_relation:linear_r2:positive` | candidate > 0 | 0.000000 | 0.060133 | PASS |
| `past_relation:neural_mean_r2:positive` | candidate > 0 | 0.000000 | -0.061140 | FAIL |

## Final TEMPO retention scores

Accuracies are saved fractions, retaining both input scaler policies and intact/deleted validation views. Full master/repetition results and conditional producer intervals remain in JSON; those intervals were not independently audited.

| Seed | Arm | Input policy | Validation view | Linear accuracy | Neural accuracy | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 945101 | control | synthetic_train | validation_intact | 0.984375 | 0.975260 | 1.000000 |
| 945101 | control | synthetic_train | validation_deleted | 0.947917 | 0.948351 | 1.000000 |
| 945101 | control | market_scaler | validation_intact | 0.993490 | 0.988715 | 1.000000 |
| 945101 | control | market_scaler | validation_deleted | 0.955729 | 0.940104 | 1.000000 |
| 945101 | candidate | synthetic_train | validation_intact | 0.996094 | 0.980035 | 1.000000 |
| 945101 | candidate | synthetic_train | validation_deleted | 0.946615 | 0.930990 | 1.000000 |
| 945101 | candidate | market_scaler | validation_intact | 0.996094 | 0.986111 | 1.000000 |
| 945101 | candidate | market_scaler | validation_deleted | 0.964844 | 0.944010 | 1.000000 |
| 945102 | control | synthetic_train | validation_intact | 0.980469 | 0.979167 | 1.000000 |
| 945102 | control | synthetic_train | validation_deleted | 0.946615 | 0.937066 | 1.000000 |
| 945102 | control | market_scaler | validation_intact | 0.996094 | 0.993056 | 1.000000 |
| 945102 | control | market_scaler | validation_deleted | 0.954427 | 0.937066 | 1.000000 |
| 945102 | candidate | synthetic_train | validation_intact | 0.988281 | 0.982639 | 1.000000 |
| 945102 | candidate | synthetic_train | validation_deleted | 0.937500 | 0.930990 | 1.000000 |
| 945102 | candidate | market_scaler | validation_intact | 0.994792 | 0.987847 | 1.000000 |
| 945102 | candidate | market_scaler | validation_deleted | 0.950521 | 0.924479 | 1.000000 |
| 945103 | control | synthetic_train | validation_intact | 0.972656 | 0.970920 | 1.000000 |
| 945103 | control | synthetic_train | validation_deleted | 0.925781 | 0.914931 | 1.000000 |
| 945103 | control | market_scaler | validation_intact | 0.990885 | 0.985677 | 1.000000 |
| 945103 | control | market_scaler | validation_deleted | 0.953125 | 0.937500 | 1.000000 |
| 945103 | candidate | synthetic_train | validation_intact | 0.993490 | 0.982639 | 1.000000 |
| 945103 | candidate | synthetic_train | validation_deleted | 0.955729 | 0.944010 | 1.000000 |
| 945103 | candidate | market_scaler | validation_intact | 0.997396 | 0.980035 | 1.000000 |
| 945103 | candidate | market_scaler | validation_deleted | 0.963542 | 0.954861 | 1.000000 |

## Methods

The comparison tests one objective change against fresh V4 controls. Both arms
use the same native36 architecture: learned20 + fixed spectral12 + mean4,
75 learned tensors, 293672 learned values and 3888 fixed values. Architecture,
pooling, parameter registration and initialization draw order are matched.
The controls retain V4's Gaussian-NLL mean gradient, weighted by conditional
variance. V15 instead uses unit Huber(delta1) mean supervision plus unit Gaussian
variance calibration on detached mean error. The variance projection still
receives detached mean4 input, so its direct gradients reach the variance head only;
the Huber route reaches the mean/backbone. Both arms retain one optimizer and
one shared global clip1. Gaussian NLL is reported separately; it is not the
V15 composite training objective.

There are exactly three fresh paired encoder seeds: 945101, 945102 and 945103.
Each pair uses corresponding row/mask/context streams at seed+1000/+2000/+3000.
Each of the six trajectories completed 2048 successful updates with batch8,
AdamW lr0.001/weight_decay0.0001, mask25%, context deletion15%, zero dropout
and at most4096 attempts. Actual typed continuation at update512 preserves the
trajectory. Initial0 and final2048 checkpoints are retained; final2048 is the
sole outcome checkpoint. No old trained weights, optimizer state or fitted
statistics are imported. There is no checkpoint, seed, coefficient or budget
rescue. CUDA engineering checks verified matched fresh initialization and the
intended objective/gradient routes before scientific execution.

Preprocessing and analytic-bank calibration are freshly fitted on January
TRAIN only. January2024 days1-21 supply TRAIN, days22-28 known validation;
saved readout populations contain112 TRAIN and32 validation probes. Encoders
use recent128 observations from the declared H256 anchor geometry. Encoder
raw inputs retain CACHE88 F64 precision. The separate historical target-label
recipe uses the legacy F32 loader followed by F64 target arithmetic; masks,
IDs and all112/32 saved January targets passed exact parity before confirmation
target preparation.

All readouts use unprojected native36. Market heads are fixed ridge1 and
four-output tanh16/Adam0.01/100-update heads for past volatility, trend,
activity and relation. Reps2701/2802/2903 retain uint64 seeds
13088411117524848790, 10018485049408248344 and9083751155097190455.
Ridge is deterministic and reused across repetitions. TEMPO uses original
masters941101/941102/941103,512 TRAIN and256 validation rows per cohort,
both synthetic_train and market_scaler policies, and intact/deleted views.
Its heads remain fixed as well. Head repetitions are not extra encoder pairs.

All six training trajectories finished before development extraction/scoring.
Each initial/final checkpoint passed the saved-checkpoint causality proof.
The passed development seal binds all six final checkpoint/head generations,
12 train/development receipts, six independent arithmetic audits and1662 output
files before February extraction or scoring. February2024 BTC/ETH/SOL Spot1m
is the separately declared32-probe, H256-disjoint calendar-selected month
screen. It reuses the exact January TRAIN-fitted heads and scalers, with no
February fitting, row replacement or tuning. Blind target preparation exposed
no numerical outcomes. February query reconstruction is descriptive only.

The prospective primary outcomes are paired February linear-R2 differences
for volatility and activity. A promising screen requires a positive median
and at least two improving pairs for each, a median February linear trend
difference at least -0.01, and all eight frozen January TEMPO timing checks
for every candidate. All three pairs and all head repetitions must be retained.
Undefined R2, incomplete support or invalid evidence prevents that verdict.
Execution completeness and scientific success are separate judgments.

Direct gradient separation does not establish optimization independence: the single shared global clip1 factor can couple the scaling of backbone and variance-head updates.

## Limits and historical context

January has been used repeatedly and remains development evidence. February
compares the two new arms on one additional month; it cannot replace or be
ranked directly against historical January scores. Three encoder pairs on one
month are a screen, not a population-wide superiority or significance claim.
All four tasks, individual heads, supports and initial/final changes remain
visible, including gains and losses across different pairs.

The separate22-predicate January protection report retains the strongest
V3-V13 floors frozen for V14. V4's historical19/22 used earlier floors and
V14's3/22 used stronger floors; these counts are not a ranking scale. The
new screen's0.01 trend tolerance changes no historical predicate. V3 remains
the clean research baseline, and V4's broad historical success remains a
reference. Nothing here automatically promotes an encoder or selects one for
Mityane.

The bounded historical-control comparison found the same numerical V4
architecture, objective, preprocessing recipe, cohort identities and fixed
heads, with intentionally different declared initialization/training-stream
seed families. Source identities and artifacts remain distinct. That review
does not prove identical historical/new random draws, fitted arrays or CUDA
numerics, nor explain a score difference from one historical trajectory.

Independent native arithmetic checks reconstruct saved normalizers, ridge
equations, head predictions, metrics, mask reductions and row associations.
January query waveform witnesses cover the first8 examples of each112/32
population; full-population means are checked from saved row metrics.
February waveform witnesses cover all32 probes. Audits do not independently
reimplement encoder forwards or optimizer trajectories, prove gradient routes
from saved objective arithmetic, or audit the producer's bootstrap intervals.
Engineering route tests and saved-array audits establish different evidence.
Reconstruction and past-window readouts do not establish forecasting,
identified intervention effects, decision quality or downstream utility.

February has now been inspected for this declared study. A later recipe chosen after inspecting these results needs a newly declared confirmation population for a fresh test.

Additional machine-report limitations:

- Three paired encoder seeds on one additional month constitute a screen, not general-purpose superiority.
- January was repeatedly used and remains development evidence; February comparisons are within the new paired arms.
- Head repetitions are not independent encoder or market-population replications.
- Saved-fit arithmetic is independently reconstructed; encoder forwards and optimizer trajectories are not independently reimplemented.
- Training fixed-query waveform witnesses cover first8 examples only; full112/32 saved row reductions are checked without claiming full waveform reconstruction.
- Conditional bootstrap intervals from the frozen producer are retained but are not independently audited.
- Past-window readout, reconstruction and TEMPO do not establish forecasting, interventions or downstream decision utility.

## Recorded compute scope

These are copied saved timing fields. Overlapping scopes are not summed; unrecorded market-head CPU and confirmation wall times are not estimated. Receipt-to-receipt gaps are not GPU or hashing timers.

| Seed | Arm | Synchronized GPU step seconds | Training trajectory wall seconds | Calibration seconds | Development extraction seconds (incl. I/O) |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | 115.165788 | 151.142781 | 3.034951 | 60.892020 |
| 945101 | candidate | 119.224921 | 154.794322 | 2.580726 | 61.943388 |
| 945102 | control | 114.734635 | 150.605027 | 2.631968 | 60.876696 |
| 945102 | candidate | 120.212487 | 156.159352 | 2.598005 | 60.832361 |
| 945103 | control | 114.523375 | 150.385973 | 2.690416 | 59.948426 |
| 945103 | candidate | 120.083475 | 156.154780 | 2.559339 | 58.801129 |

## Actual evidence and identities

- [Complete machine report](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-001.json)
- [Report card with all actual phase/audit pins](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-inputs-001.json)
- [Actual immutable study card](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/run-inputs-001.json)
- [Six-run development seal, established before February](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/development-seal.json)
- [Reporting source review](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/source-review-report-002.json)
- [Reporting protocol review](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-protocol-review-002.json)
- [Reporting admission](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-admission-002.json)
- [Bounded historical V4/control source comparison](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/historical-v4-control-recipe-comparison-002.json)
- [Compact hash-bound closeout record (created separately)](../../records/CPB-v15.objective-001.json)

| Seed | Arm | Training | Development | Confirmation | Development arithmetic | Confirmation arithmetic |
| --- | --- | --- | --- | --- | --- | --- |
| 945101 | control | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/train-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/development-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/confirmation-receipt.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/independent-audit.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/control/confirmation-audit.json) |
| 945101 | candidate | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/train-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/development-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/confirmation-receipt.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/independent-audit.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945101/candidate/confirmation-audit.json) |
| 945102 | control | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/train-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/development-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/confirmation-receipt.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/independent-audit.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/control/confirmation-audit.json) |
| 945102 | candidate | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/train-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/development-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/confirmation-receipt.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/independent-audit.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945102/candidate/confirmation-audit.json) |
| 945103 | control | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/train-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/development-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/confirmation-receipt.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/independent-audit.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/control/confirmation-audit.json) |
| 945103 | candidate | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/train-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/development-receipt.json) | [receipt](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/confirmation-receipt.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/independent-audit.json) | [audit](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/subruns/seed-945103/candidate/confirmation-audit.json) |

The reporting successor coalesces only identical final-barrier bindings: 19754 declared file-pin occurrences receive 2018 fresh distinct file checks. Initial and per-read barriers remain fresh and separate; metric calculations are unchanged. Exact duplicate checks are coalesced within this final barrier on completed sealed artifacts; this is not observationally identical to repeated checks with concurrent writers. Initial and per-read checks remain separate and fresh; unique final files retain unchanged canonical, single-link and full-hash checks.

The first report derivation was interrupted when the managed container stopped before writing a report. The [interruption record](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-interruption-001.json) preserves the observed state; the [resumption record](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-resumption-002.json) identifies the same reused container. The unfinished [original resumed log](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-execution-002.log) is retained, and the [reporting transition](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-transition-001.json) records its disposition. The independently reviewed successor has a distinct source, card and output, with a [completed execution log](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/report-final-unique-execution-001.log). No training, head fitting or arithmetic audit was replayed. The container stop cause was not identified.

The development seal and report/card preserve exact source, settings, data, environment, checkpoint, fixed-head, receipt and binary identities. The side records [paired-stream metadata 945101](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/paired-stream-check-seed-945101.json), [paired-stream metadata 945102](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/paired-stream-check-seed-945102.json), [paired-stream metadata 945103](../../../../../.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/paired-stream-check-seed-945103.json) compare CSV/JSON pairing metadata only; equal FNV/seed fields do not independently establish full tensor or realized context-mask equality. Raw data and tensors remain ignored local artifacts. No publication, commit, catalog application or Mityane selection follows from this document.

## Appendix A: January initial readout R²

Initial scores are descriptive and use their separately fitted frozen-recipe heads; final2048 is the sole outcome checkpoint.

| Seed | Arm | Past-window task | Linear R² | Neural mean R² | Coverage |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | Volatility | 0.479068 | 0.026938 | 1.000000 |
| 945101 | control | Trend | -1.129942 | -1.620970 | 1.000000 |
| 945101 | control | Activity | 0.100790 | 0.604307 | 1.000000 |
| 945101 | control | Relation | -0.020456 | -0.194457 | 1.000000 |
| 945101 | candidate | Volatility | 0.479068 | 0.026938 | 1.000000 |
| 945101 | candidate | Trend | -1.129942 | -1.620970 | 1.000000 |
| 945101 | candidate | Activity | 0.100790 | 0.604307 | 1.000000 |
| 945101 | candidate | Relation | -0.020456 | -0.194457 | 1.000000 |
| 945102 | control | Volatility | 0.449220 | -0.323629 | 1.000000 |
| 945102 | control | Trend | -0.747550 | -1.797860 | 1.000000 |
| 945102 | control | Activity | 0.515085 | 0.548198 | 1.000000 |
| 945102 | control | Relation | 0.265812 | -0.139911 | 1.000000 |
| 945102 | candidate | Volatility | 0.449220 | -0.323629 | 1.000000 |
| 945102 | candidate | Trend | -0.747550 | -1.797860 | 1.000000 |
| 945102 | candidate | Activity | 0.515085 | 0.548198 | 1.000000 |
| 945102 | candidate | Relation | 0.265812 | -0.139911 | 1.000000 |
| 945103 | control | Volatility | -0.893082 | -0.129898 | 1.000000 |
| 945103 | control | Trend | -0.399590 | -0.951758 | 1.000000 |
| 945103 | control | Activity | 0.736982 | 0.617690 | 1.000000 |
| 945103 | control | Relation | 0.175525 | -0.191136 | 1.000000 |
| 945103 | candidate | Volatility | -0.893082 | -0.129898 | 1.000000 |
| 945103 | candidate | Trend | -0.399590 | -0.951758 | 1.000000 |
| 945103 | candidate | Activity | 0.736982 | 0.617690 | 1.000000 |
| 945103 | candidate | Relation | 0.175525 | -0.191136 | 1.000000 |

## Appendix B: individual neural readout repetitions

Columns retain reps2701/2802/2903 in their fixed order. They are head repetitions, not additional encoder seeds or market populations.

### January initial

| Encoder seed | Arm | Past-window task | Rep2701 R² | Rep2802 R² | Rep2903 R² |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | Volatility | 0.101352 | 0.343258 | -0.363797 |
| 945101 | control | Trend | -1.963264 | -1.270693 | -1.628951 |
| 945101 | control | Activity | 0.651555 | 0.624468 | 0.536899 |
| 945101 | control | Relation | -0.081189 | -0.276857 | -0.225324 |
| 945101 | candidate | Volatility | 0.101352 | 0.343258 | -0.363797 |
| 945101 | candidate | Trend | -1.963264 | -1.270693 | -1.628951 |
| 945101 | candidate | Activity | 0.651555 | 0.624468 | 0.536899 |
| 945101 | candidate | Relation | -0.081189 | -0.276857 | -0.225324 |
| 945102 | control | Volatility | -0.789755 | -0.112887 | -0.068245 |
| 945102 | control | Trend | -3.473377 | -0.776573 | -1.143629 |
| 945102 | control | Activity | 0.586895 | 0.459914 | 0.597786 |
| 945102 | control | Relation | -0.499182 | 0.113399 | -0.033950 |
| 945102 | candidate | Volatility | -0.789755 | -0.112887 | -0.068245 |
| 945102 | candidate | Trend | -3.473377 | -0.776573 | -1.143629 |
| 945102 | candidate | Activity | 0.586895 | 0.459914 | 0.597786 |
| 945102 | candidate | Relation | -0.499182 | 0.113399 | -0.033950 |
| 945103 | control | Volatility | -0.540894 | 0.398173 | -0.246972 |
| 945103 | control | Trend | -0.782234 | -1.367071 | -0.705968 |
| 945103 | control | Activity | 0.702842 | 0.548108 | 0.602119 |
| 945103 | control | Relation | -0.071002 | -0.152038 | -0.350367 |
| 945103 | candidate | Volatility | -0.540894 | 0.398173 | -0.246972 |
| 945103 | candidate | Trend | -0.782234 | -1.367071 | -0.705968 |
| 945103 | candidate | Activity | 0.702842 | 0.548108 | 0.602119 |
| 945103 | candidate | Relation | -0.071002 | -0.152038 | -0.350367 |

### January final known validation

| Encoder seed | Arm | Past-window task | Rep2701 R² | Rep2802 R² | Rep2903 R² |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | Volatility | 0.550802 | 0.221188 | 0.565646 |
| 945101 | control | Trend | 0.491896 | -0.232065 | 0.039733 |
| 945101 | control | Activity | 0.856909 | 0.790495 | 0.729856 |
| 945101 | control | Relation | 0.398568 | 0.009014 | -0.143276 |
| 945101 | candidate | Volatility | 0.046505 | 0.593881 | 0.594887 |
| 945101 | candidate | Trend | -0.579520 | -0.133860 | -0.092562 |
| 945101 | candidate | Activity | 0.594817 | 0.807567 | 0.758132 |
| 945101 | candidate | Relation | -0.206689 | -0.023108 | 0.029207 |
| 945102 | control | Volatility | 0.452146 | 0.746651 | 0.709122 |
| 945102 | control | Trend | 0.453764 | 0.153479 | 0.636530 |
| 945102 | control | Activity | 0.668999 | 0.763155 | 0.836259 |
| 945102 | control | Relation | -0.010005 | 0.152533 | 0.050448 |
| 945102 | candidate | Volatility | -0.251313 | 0.508464 | 0.453794 |
| 945102 | candidate | Trend | 0.258758 | -0.387215 | 0.340232 |
| 945102 | candidate | Activity | 0.473640 | 0.554228 | 0.688049 |
| 945102 | candidate | Relation | -0.080654 | 0.390083 | 0.251342 |
| 945103 | control | Volatility | 0.135492 | 0.336267 | 0.496725 |
| 945103 | control | Trend | 0.394190 | 0.387101 | -0.065634 |
| 945103 | control | Activity | 0.621411 | 0.532454 | 0.735645 |
| 945103 | control | Relation | -0.249141 | -0.006137 | -0.031506 |
| 945103 | candidate | Volatility | 0.334467 | 0.414921 | 0.655583 |
| 945103 | candidate | Trend | 0.004404 | -0.168022 | -0.172843 |
| 945103 | candidate | Activity | 0.566218 | 0.655601 | 0.676856 |
| 945103 | candidate | Relation | 0.038893 | -0.435350 | 0.213037 |

### February final confirmation

| Encoder seed | Arm | Past-window task | Rep2701 R² | Rep2802 R² | Rep2903 R² |
| --- | --- | --- | --- | --- | --- |
| 945101 | control | Volatility | 0.465236 | 0.301876 | -0.111252 |
| 945101 | control | Trend | -0.235259 | -0.118659 | -0.087159 |
| 945101 | control | Activity | 0.410492 | 0.384336 | 0.256185 |
| 945101 | control | Relation | -1.406626 | -0.765607 | -1.174022 |
| 945101 | candidate | Volatility | 0.461028 | -0.474699 | 0.309562 |
| 945101 | candidate | Trend | 0.135527 | -0.971598 | -0.473606 |
| 945101 | candidate | Activity | 0.656186 | 0.357890 | 0.563498 |
| 945101 | candidate | Relation | -0.890717 | -0.179624 | -1.670980 |
| 945102 | control | Volatility | 0.125633 | 0.488667 | 0.124692 |
| 945102 | control | Trend | -0.284641 | -0.562464 | -0.469405 |
| 945102 | control | Activity | 0.525718 | 0.693459 | 0.447250 |
| 945102 | control | Relation | -1.312482 | -4.351214 | -1.997530 |
| 945102 | candidate | Volatility | -0.737624 | 0.309819 | 0.283738 |
| 945102 | candidate | Trend | -0.111621 | -1.033874 | -0.792374 |
| 945102 | candidate | Activity | 0.637486 | 0.630177 | 0.638599 |
| 945102 | candidate | Relation | -0.628379 | -0.792388 | -1.434125 |
| 945103 | control | Volatility | 0.296795 | 0.163671 | 0.466344 |
| 945103 | control | Trend | -0.044159 | 0.264892 | -0.180130 |
| 945103 | control | Activity | 0.541344 | 0.549027 | -0.341863 |
| 945103 | control | Relation | -0.593843 | -0.814076 | -0.503503 |
| 945103 | candidate | Volatility | 0.468070 | 0.623637 | 0.512087 |
| 945103 | candidate | Trend | -1.439265 | -1.158879 | -1.017927 |
| 945103 | candidate | Activity | 0.408124 | 0.314100 | 0.465996 |
| 945103 | candidate | Relation | -2.809450 | -1.283974 | -0.537400 |
