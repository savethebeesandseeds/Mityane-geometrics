# Fresh spectral representation confirmation

Three fresh cohorts per task; TRAIN256/128 source pairs and VALIDATION128/64 pairs.
Each instance ran once on CUDA for512 updates. Initial0 controls, native32,
fixed Ridge1/tanh16 heads and original masked waveform queries are retained.

**RPB-v19.alt-01**: new instances of unchanged RPB-v19, with20 learned shape
coordinates and12 fixed generic low/high complex spectral relations.
226,877 registered/226,445 trainable/432 frozen parameters.

This tests repeatability on fresh sources. Strong initial results credit the
fixed prior; trained gains over those controls are a separate learning question.
No old encoder/head, raw baseline or information-frequency search was rerun.

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 99.48 | 98.44 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 98.70 | 98.18 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 93.23 | 93.23 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 91.67 | 89.76 | 100.00 |

Dataset: **TEMPO-4** · slow-component lead/lag in a coherent two-rhythm mixture · **complexity5/5** · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 |512| 0.717581 | 0.742483 | 20.69 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 99.48 | 98.87 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_intact · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 99.48 | 98.87 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 94.79 | 86.89 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · validation_deleted · updates0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 | 32 | 92.45 | 85.42 | 100.00 |

Dataset: **AMP-2** · relative slow/fast strength in a coherent two-rhythm mixture · **complexity5/5** · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v19.alt-01 |512| 0.726132 | 0.757934 | 21.30 |

The frozen confirmation gate **passed** across all12 conditions.
confirmation complete; a learned-path change requires its own prospective card.

Per-cohort results, initial controls, support populations, source-group intervals,
costs and gates are in the [durable JSON](results/spectral_confirmation_v1.json).
The [prospective card](../code/evaluation/cards/spectral_confirmation_v1.md)
fixed all three fresh masters and the gate before generation. No additional
quality seeds are allocated. All12 fixed values are byte-identical at0/512.
Saved-only verification performs zero encoder forwards, updates or head fits.
Full artifacts: `output/runs/rpb-spectral-confirmation/admission-hzyMa2`.

The original v19 screen, TEMPO-3 v18, formal v4, original v7, older evidence
and defaults remain preserved. No TEST, stress or promotion.
