# Partitioned temporal relation screen

New CUDA architecture/objective runs; known TRAIN/VALIDATION development data.
TRAIN256 examples/128 pairs and VALIDATION128/64 pairs per run. B8,512 updates,
native32; fixed Ridge1 and tanh16/Adam .01/100, three head repetitions.
No PCA after either encoder. Existing baselines reuse saved metadata.

**RPB-v16** dedicates20 native coordinates to learned waveform shape and12
to grouped antisymmetric timing relations. Total226,877 parameters.

Untrained scores measure the built-in prior; they are not evidence of learning.

## Two-cohort screen

Masters: 75272, 76373.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.95 | 52.60 | 100 |
| PCA only — no encoder | 32 | 47.66 | 51.43 | 100 |
| RPB-v10.alt-05 | 32 | 50.00 | 53.26 | 100 |
| RPB-v16 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v16 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.22 | 51.43 | 100 |
| PCA only — no encoder | 32 | 48.05 | 51.30 | 100 |
| RPB-v10.alt-05 | 32 | 54.30 | 54.56 | 100 |
| RPB-v16 | 32 | 83.20 | 80.34 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v16 | 32 | 89.06 | 99.74 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v16 |512| 0.668228 | 0.707265 | 21.08 |

The screen gate is75% in both fixed heads and both views on each of the
two starting cohorts, with100% coverage. Retain all failures and initial
scores. Additional known cohorts are development checks after screening,
not fresh-data confirmation. No automatic promotion or reference replacement.

- RPB-v16: fails the predeclared continuation gate.

The dedicated coordinates preserve intact accuracy at100% on both known
screen cohorts. The mean deleted scores improve over v14's matched screen,
but master76373's neural score is69.53%, below the fixed75% gate. Stop this
version after two cohorts; do not silently expand its budget or retune it.
Untrained deleted scores are89.06%/99.74%, so waveform training still weakens
the prior, despite structural oddness. The [separate v17 card](../../evaluation/cards/fixed_prior_relation_screen_v1.md)
tests freezing that generic odd projection while learning the shape block.
No old version, classifier recipe or dataset is changed.

CUDA primitive and serialized-loader admission passed; the saved-only checker
passed2,331,315 checks/72 CPU archives without loading the model or fitting
heads. Two new trajectories/1,024 CUDA updates,12 native exports,12 fixed-head
pipelines/24 heads and4 original-query writers/16 masked forwards were run once.
There were no baseline refits, CPU encoder calls, TEST or stress access.
GPU training seconds include synchronized training-loop batch preparation and
transfers, and exclude head fitting/export/query/checkpoint I/O. The fixed
size32 Ridge/tanh16 heads have66/562 parameters; size576 raw heads have1,154/9,266.
