TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

New saved-feature diagnostic; zero encoder execution or updates. RPB-v20.alt-01 is unchanged. Size32 means its complete export; size20 means the learned coordinates; size12 means the fixed spectral coordinates. The fixed12 arrays are exactly identical at0/512, so their nine TRAIN-fitted head pairs are shared across both points. Learned20 uses18 new pairs. Complete32 and raw results reuse original fitted decisions without new fits.

Linear head = ridge1; Neural head = tanh16,100 Adam steps at0.01. Accuracy is conditional on valid rows; coverage is the fraction scored. Means retain all three cohorts and all three head repetitions. Changing branch width changes head parameter counts and initialization seeds; this tests branch sufficiency, rather than providing a perfectly controlled attribution of each coordinate.

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Initial, zero updates; intact VALIDATION. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 94.01 | 93.23 | 100.00 |
| RPB-v20.alt-01 | 20 | 50.13 | 49.78 | 100.00 |
| RPB-v20.alt-01 | 12 | 95.18 | 94.66 | 100.00 |
| Raw data — no encoder | 6912 | 49.61 | 50.69 | 100.00 |

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Initial, zero updates; VALIDATION with an additional30% coordinate deletion. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 89.32 | 87.89 | 100.00 |
| RPB-v20.alt-01 | 20 | 50.91 | 50.69 | 100.00 |
| RPB-v20.alt-01 | 12 | 89.97 | 89.54 | 100.00 |
| Raw data — no encoder | 6912 | 49.22 | 50.35 | 100.00 |

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Trained,512 updates; intact VALIDATION. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 94.66 | 94.31 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.74 | 50.17 | 100.00 |
| RPB-v20.alt-01 | 12 | 95.18 | 94.66 | 100.00 |
| Raw data — no encoder | 6912 | 49.61 | 50.69 | 100.00 |

TEMPO-5 — complexity **5/5** (designed): three streams,128 candles,nine grouped features; slow lead/lag beneath fast rhythms and unrelated structured signals, with natural missing cells/gaps. TRAIN512 rows/256 source pairs; VALIDATION256 rows/128 source pairs per cohort. Three cohorts and three head repetitions.

Trained,512 updates; VALIDATION with an additional30% coordinate deletion. The TRAIN-fitted head is reused unchanged across views.

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 90.10 | 88.72 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.61 | 49.91 | 100.00 |
| RPB-v20.alt-01 | 12 | 89.97 | 89.54 | 100.00 |
| Raw data — no encoder | 6912 | 49.22 | 50.35 | 100.00 |

Reused original training record; no training occurred in this diagnostic.

| Encoder | Updates | Train error | Validation error | GPU training seconds |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 |512|0.7825|0.7986|20.8|

Training errors are original fixed-query standardized MAE; the source training usedCUDA float32,batch8. These known VALIDATION results do not establish unseen TEST or market quality.
