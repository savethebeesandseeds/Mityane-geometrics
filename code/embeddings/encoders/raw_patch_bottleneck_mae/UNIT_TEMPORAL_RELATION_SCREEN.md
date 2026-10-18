# Unit temporal relation screen

New CUDA architecture/objective runs; known TRAIN/VALIDATION development data.
TRAIN256 examples/128 pairs and VALIDATION128/64 pairs per run. B8,512 updates,
native32; fixed Ridge1 and tanh16/Adam .01/100, three head repetitions.
No PCA after either encoder. Matched v17 scores reuse saved metadata.

**RPB-v18** keeps20 learned shape coordinates and normalizes each generic
four-spacing fixed relation group to unit length inside native32.
Total226,877 parameters;226,445 trainable. No new weights or heads.

Untrained scores measure the built-in prior; they are not evidence of learning.

## Two-cohort screen

Masters: 80787, 84090.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 100.00 | 100.00 | 100 |
| RPB-v18 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 75.00 | 99.74 | 100 |
| RPB-v18 | 32 | 100.00 | 99.87 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18 | 32 | 100.00 | 99.74 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18 |512| 0.692434 | 0.717017 | 21.70 |

## Five known cohorts

Masters: 80787, 81888, 82989, 84090, 85191.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 100.00 | 100.00 | 100 |
| RPB-v18 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 82.19 | 99.90 | 100 |
| RPB-v18 | 32 | 100.00 | 99.95 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18 | 32 | 100.00 | 99.90 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18 |512| 0.692694 | 0.719535 | 21.89 |

The screen gate is75% in both fixed heads and both views on each of the
two starting cohorts, with100% coverage, no-worse Neural than saved v17
per cohort/view, and at least5pp mean deleted Linear gain. Retain all failures and initial
scores. Additional known cohorts are development checks after screening,
not fresh-data confirmation. No automatic promotion or reference replacement.

Master84090 was selected because of its known weak linear score;80787 is
a second known cohort. This is targeted development, not unseen confirmation.
Unit normalization can amplify weak unrelated pairs and removes relation
magnitude. Credit the fixed prior, and retain waveform reconstruction tradeoffs.

- RPB-v18: passes the predeclared continuation gate.
