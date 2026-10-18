# RPB-v17 fresh-source confirmation

New **RPB-v17.alt-01** instances reuse the exact RPB-v17 design:20 learned
shape coordinates and12 fixed generic temporal-relation coordinates, native32.
TRAIN256/VALIDATION128 per cohort, CUDA B8/512 updates; original .15 context
deletion and waveform Huber1. Ridge1; tanh16/Adam .01/100, repetitions2701/2802/2903.
Five new source-disjoint cohorts run once, with all results retained.

Masters: 80787, 81888, 82989, 84090, 85191.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0; heads still fit TRAIN.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 100.00 | 100.00 | 100 |

Worst cohort Linear/Neural: 100.00%/100.00%.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 82.19 | 99.90 | 100 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0; heads still fit TRAIN.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 82.03 | 99.84 | 100 |

Worst cohort Linear/Neural: 70.31%/99.48%.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed hidden-target TRAIN/intact VALIDATION queries; standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 |512| 0.705814 | 0.731698 | 21.82 |

Time is synchronized training-loop wall time, including CPU batch preparation/transfers;
it excludes native extraction, head fitting, queries and checkpoint I/O.

The primary continuity rule is75% for both heads in both views on EACH cohort;
the secondary strong-capability target is95% intact for both heads,95% deleted Neural
and75% deleted Linear, with100% coverage. Both were fixed before quality measurement.
Continuity: **fails**.
Strong capability: **fails**.

Credit the fixed12-coordinate prior when the untrained encoder already solves timing.
The learned20-coordinate block and decoder still train, but timing accuracy alone does
not establish learned improvement or broad representation quality. This is same-law
fresh-source replication, not a new complexity level or automatic default promotion.

Historical development scores are preserved in the milestone note. No older encoder
or matched baseline was refitted here, and no PCA follows the encoder.

Next: a separately preregistered coherent multi-component timing/shape dataset,
with its new codename, numeric complexity and fixed tasks before measurement.

