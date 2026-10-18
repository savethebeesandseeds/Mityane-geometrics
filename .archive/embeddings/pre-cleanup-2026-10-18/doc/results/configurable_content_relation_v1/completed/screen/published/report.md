# RPB-v22 content-only relation development evidence

New candidate training; saved controls reused without refitting. RPB-v20 remains preserved.

Dataset: **TEMPO-5** · complexity **5/5** · slow timing beneath fast rhythms and nine grouped features.

RPB-v22.alt-01 learns ordered content relations without additive mask-density inputs. RPB-v20.alt-01 is the saved complete32 control.

Intact VALIDATION. Dataset: **TEMPO-5**, complexity **5/5**, slow timing with structured distractors.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 95.12 | 94.92 | 100.00 |
| RPB-v22.alt-01 | 32 | 97.46 | 96.55 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 50.26 | 100.00 |
| RPB-v22.alt-01 | 20 | 63.28 | 61.52 | 100.00 |

Size32 is the complete native export; size20 is the first20 learned coordinates, a diagnostic subset.

Extra30% deletion VALIDATION. Dataset: **TEMPO-5**, complexity **5/5**, slow timing with structured distractors.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 88.67 | 87.89 | 100.00 |
| RPB-v22.alt-01 | 32 | 90.04 | 89.71 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 49.87 | 100.00 |
| RPB-v22.alt-01 | 20 | 61.91 | 62.17 | 100.00 |

Size32 is the complete native export; size20 is the first20 learned coordinates, a diagnostic subset.

Dataset: **TEMPO-5**, complexity **5/5**. Fixed original hidden-query standardized MAE; lower is better.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.alt-01 | 512 | 0.78213 | 0.80820 | 24.8 |

Prospective joint gate: **PASS**.

Every allocated cohort and fixed head repetition is retained. No TEST, market measurement or default replacement.
