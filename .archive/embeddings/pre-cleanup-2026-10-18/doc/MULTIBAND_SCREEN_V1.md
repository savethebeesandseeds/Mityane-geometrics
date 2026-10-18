# Multiband representation screen

Two new cohorts per task, TRAIN256/128 source pairs and VALIDATION128/64 pairs.
Each candidate ran once per task/cohort on CUDA for512 updates; initial0 controls
are retained. Native32 and fixed Ridge1/tanh16 heads; no post-encoder PCA.

**RPB-v18.alt-01**: new instances of the unchanged unit temporal-relation design.
**RPB-v19**: same learned shape20 and fixed12 allocation, with mask-aware generic
low/high harmonic complex relations and joint per-pair normalization.
Both have226,877 registered/226,445 trainable/432 frozen parameters.

Initial success credits the architectural prior; learned shape requires gains
over that instance’s own initial control. No old model or head was rerun.

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 95.31 | 95.83 | 100.00 |
| RPB-v19 | 32 | 98.83 | 98.83 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 95.31 | 93.88 | 100.00 |
| RPB-v19 | 32 | 98.83 | 98.57 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 78.12 | 75.26 | 100.00 |
| RPB-v19 | 32 | 92.97 | 94.27 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 78.52 | 76.17 | 100.00 |
| RPB-v19 | 32 | 92.97 | 92.32 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 |512| 0.710965 | 0.748360 | 20.90 |
| RPB-v19 |512| 0.711928 | 0.751925 | 19.76 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 54.30 | 58.20 | 100.00 |
| RPB-v19 | 32 | 98.44 | 97.92 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 51.95 | 55.21 | 100.00 |
| RPB-v19 | 32 | 98.44 | 97.92 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 60.16 | 56.64 | 100.00 |
| RPB-v19 | 32 | 94.53 | 88.41 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 | 32 | 50.00 | 55.99 | 100.00 |
| RPB-v19 | 32 | 94.14 | 86.07 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v18.alt-01 |512| 0.726795 | 0.767214 | 21.66 |
| RPB-v19 |512| 0.727008 | 0.767751 | 20.46 |

The frozen joint continuation rule **passed**.
separate prospective confirmation required; no further seeds allocated.

Per-cohort results, initial controls, populations, source-group intervals, costs
and every gate condition are retained in the [durable JSON](results/multiband_screen_v1.json).
The [prospective card](../code/evaluation/cards/multiband_screen_v1.md) fixes the
decision before quality. Both datasets remain separate; complexity is a designed
ordinal level, not an accuracy-derived score or a claim of equal task difficulty.

All12 fixed values per row are byte-identical at0/512 in both designs.
CPU saved-only verification replays predictions from retained fitted heads and
query arithmetic, with zero model execution, head fits or frequency searches.
Full artifacts remain in `output/runs/rpb-multiband-screen/admission-0Yzi6p`.
Source, card, SDK, input and artifact hashes are retained in the JSON.

TEMPO-3 RPB-v18, formal RPB-v4, original RPB-v7 and all older results remain
preserved. No TEST, stress evaluation, default change or reference promotion.
