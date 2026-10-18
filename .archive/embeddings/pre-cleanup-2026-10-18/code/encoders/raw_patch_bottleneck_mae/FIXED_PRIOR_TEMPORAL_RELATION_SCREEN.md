# Fixed-prior temporal relation screen

New CUDA architecture/objective runs; known TRAIN/VALIDATION development data.
TRAIN256 examples/128 pairs and VALIDATION128/64 pairs per run. B8,512 updates,
native32; fixed Ridge1 and tanh16/Adam .01/100, three head repetitions.
No PCA after either encoder. Existing baselines reuse saved metadata.

**RPB-v17** dedicates20 native coordinates to learned waveform shape and12
to fixed grouped antisymmetric timing relations. Total226,877 parameters;226,445 trainable.

Untrained scores measure the built-in prior; they are not evidence of learning.

## Two-cohort screen

Masters: 75272, 76373.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.95 | 52.60 | 100 |
| PCA only — no encoder | 32 | 47.66 | 51.43 | 100 |
| RPB-v10.alt-05 | 32 | 50.00 | 53.26 | 100 |
| RPB-v17 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.22 | 51.43 | 100 |
| PCA only — no encoder | 32 | 48.05 | 51.30 | 100 |
| RPB-v10.alt-05 | 32 | 54.30 | 54.56 | 100 |
| RPB-v17 | 32 | 89.06 | 99.74 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 | 32 | 89.06 | 99.74 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 |512| 0.674393 | 0.719119 | 20.09 |

## Five known cohorts

Masters: 75272, 76373, 77474, 78575, 79676.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.25 | 50.73 | 100 |
| PCA only — no encoder | 32 | 48.44 | 51.15 | 100 |
| RPB-v10.alt-05 | 32 | 52.81 | 54.90 | 100 |
| RPB-v17 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.53 | 50.78 | 100 |
| PCA only — no encoder | 32 | 48.75 | 50.68 | 100 |
| RPB-v10.alt-05 | 32 | 53.75 | 54.32 | 100 |
| RPB-v17 | 32 | 85.94 | 99.84 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 | 32 | 86.09 | 99.74 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17 |512| 0.689181 | 0.723849 | 20.23 |

The screen gate is75% in both fixed heads and both views on each of the
two starting cohorts, with100% coverage. Retain all failures and initial
scores. Additional known cohorts are development checks after screening,
not fresh-data confirmation. No automatic promotion or reference replacement.

- RPB-v17: passes the predeclared continuation gate.

The fixed-prior recipe passes both starting-cohort gates and completes the three
remaining known cohorts once. Five-cohort intact accuracy is100% with both
heads. Extra deletion gives85.94% Linear and99.84% Neural, at100% coverage.
The corresponding untrained encoder scores are86.09%/99.74% deleted; classifier
heads are fitted on TRAIN even when the encoder is untrained. Credit the fixed
generic timing calculation, rather than claiming timing was learned from the
waveform loss. The learned20-coordinate shape block still changes during training.

This is the strongest measured TEMPO-3 timing candidate. It is not a production
promotion or independent unseen-data confirmation: all five source cohorts
were already known, and three were evaluated after a screen-selected design.
Original v7, formal v4 and TEMPO-1 v10.alt-03 remain unchanged. RPB-v14/v15/v16
artifacts remain available; stopped recipes are not retuned or overwritten.

Waveform reconstruction remains a tradeoff. Five-cohort query MAE is
0.689181/0.723849 TRAIN/VALIDATION, versus the reused early-v10 reference's
0.644282/0.682958. Keeping timing intact does not solve reconstruction or prove
a universally useful representation. Total parameters are226,877, with226,445
trainable and432 frozen. Size32 Ridge/tanh16 heads have66/562 fitted parameters;
size576 raw heads have1,154/9,266. No PCA follows an encoder.

Five new quality trajectories ran once on CUDA:2,560 updates/20,480 sampled rows,
30 native exports,30 head pipelines/60 heads and10 original-query writers/40
necessary masked forwards. Two retained points per trajectory use the same
fixed head recipe; no baseline refits, CPU encoder execution, TEST or stress.
Actual CUDA model and serialized-loader admission passed. Saved-only checks
passed5,828,142 checks/180 CPU archives, including exact equality of30,720
odd-coordinate values at0/512 on TRAIN, intact and deleted validation. These
checks load no CUDA model/optimizer, perform no forwards, and fit no heads.
GPU training seconds are mean synchronized training-loop wall time including
batch preparation/transfers, excluding extraction, readouts, query and checkpoint
I/O. Timings are descriptive, not a repeated speed benchmark.

Next: freeze a fresh-source TRAIN/VALIDATION confirmation card for v17 with the
same encoder/head/update budgets, retaining every result. If it passes, specify
a separate coherent multi-component timing dataset before measurement, with a
new codename and designed complexity. Keep the fixed relation prior available
while testing what the learned shape path contributes beyond this lag-sign task.
