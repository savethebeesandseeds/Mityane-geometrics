# RPB-v21 learned relation development evidence

New candidate training; saved controls reused without refitting. RPB-v20 remains preserved.

Dataset: **TEMPO-5** · complexity **5/5** · slow timing beneath fast rhythms and nine grouped features.

RPB-v21.alt-01 adds a zero-start learned ordered relation path. RPB-v20.alt-01 is the saved complete32 control.

Intact VALIDATION. Dataset: **TEMPO-5**, complexity **5/5**, slow timing with structured distractors.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 95.12 | 94.92 | 100.00 |
| RPB-v21.alt-01 | 32 | 97.85 | 96.48 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 50.26 | 100.00 |
| RPB-v21.alt-01 | 20 | 62.50 | 61.00 | 100.00 |

Size32 is the complete native export; size20 is the first20 learned coordinates, a diagnostic subset.

Extra30% deletion VALIDATION. Dataset: **TEMPO-5**, complexity **5/5**, slow timing with structured distractors.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 88.67 | 87.89 | 100.00 |
| RPB-v21.alt-01 | 32 | 65.43 | 78.06 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 49.87 | 100.00 |
| RPB-v21.alt-01 | 20 | 61.72 | 58.40 | 100.00 |

Size32 is the complete native export; size20 is the first20 learned coordinates, a diagnostic subset.

Dataset: **TEMPO-5**, complexity **5/5**. Fixed original hidden-query standardized MAE; lower is better.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v21.alt-01 | 512 | 0.78891 | 0.81042 | 43.9 |

Prospective joint gate: **FAIL**.

Every allocated cohort and fixed head repetition is retained. No TEST, market measurement or default replacement.
