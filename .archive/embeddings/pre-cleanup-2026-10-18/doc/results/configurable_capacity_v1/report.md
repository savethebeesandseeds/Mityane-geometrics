TEMPO-5 — complexity **5/5** (designed): three streams, 128 candles, nine structured features; infer slow lead/lag beneath faster patterns and unrelated structured signals. Natural missing values and contiguous candle gaps. TRAIN: 256 source pairs; VALIDATION: 128 source pairs per master. Three masters, three fixed heads per method.

RPB-v20.alt-01: A, internal width64, native32. RPB-v20.alt-02: B, internal width128, native32. RPB-v20.alt-03: C, internal width64, native64. Initial means zero encoder updates; trained means512. All encoder operations ran on CUDA.

Linear head = ridge classifier. Neural head = fixed tanh16 classifier. Accuracy is conditional on valid rows. Coverage is the fraction scored. Each percentage is the mean across three masters and three fixed head repetitions; parentheses show the range of master means. Raw data includes visibility masks: 6,912 numbers, with13,826 linear-head or110,642 neural-head parameters. Equal-size methods use identical head initialization seeds.

Intact VALIDATION, trained encoders:

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| Raw data — no encoder | 6912 | 49.6% (48.8–50.4) | 50.7% (49.6–52.7) | 100.0% |
| PCA only — no encoder | 32 | 49.6% (48.8–50.4) | 50.4% (49.3–52.0) | 100.0% |
| PCA only — no encoder | 64 | 50.5% (50.0–51.6) | 49.5% (48.8–50.3) | 100.0% |
| RPB-v20.alt-01 | 32 | 94.7% (93.8–95.3) | 94.3% (93.1–95.8) | 100.0% |
| RPB-v20.alt-02 | 32 | 94.1% (92.2–96.1) | 93.8% (91.8–96.2) | 100.0% |
| RPB-v20.alt-03 | 64 | 94.8% (92.6–96.5) | 93.4% (92.6–94.8) | 100.0% |

TEMPO-5 — complexity **5/5** (designed): three streams, 128 candles, nine structured features; infer slow lead/lag beneath faster patterns and unrelated structured signals. Natural missing values and contiguous candle gaps. TRAIN: 256 source pairs; VALIDATION: 128 source pairs per master. Three masters, three fixed heads per method.

VALIDATION with an additional30% coordinate deletion; the intact TRAIN-fitted heads remain fixed:

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| Raw data — no encoder | 6912 | 49.2% (45.7–51.2) | 50.3% (49.9–51.0) | 100.0% |
| PCA only — no encoder | 32 | 49.7% (48.8–50.8) | 50.0% (48.0–52.1) | 100.0% |
| PCA only — no encoder | 64 | 51.0% (50.4–51.6) | 49.0% (48.2–50.1) | 100.0% |
| RPB-v20.alt-01 | 32 | 90.1% (87.5–93.0) | 88.7% (86.7–90.4) | 100.0% |
| RPB-v20.alt-02 | 32 | 89.3% (88.3–90.6) | 88.5% (86.5–89.8) | 100.0% |
| RPB-v20.alt-03 | 64 | 89.1% (87.5–91.4) | 88.5% (86.1–90.4) | 100.0% |

TEMPO-5 — complexity **5/5** (designed): three streams, 128 candles, nine structured features; infer slow lead/lag beneath faster patterns and unrelated structured signals. Natural missing values and contiguous candle gaps. TRAIN: 256 source pairs; VALIDATION: 128 source pairs per master. Three masters, three fixed heads per method.

Initial encoder exports, zero updates, intact VALIDATION:

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 32 | 94.0% (92.2–96.1) | 93.2% (92.6–94.1) | 100.0% |
| RPB-v20.alt-02 | 32 | 94.0% (91.4–95.7) | 92.8% (92.1–94.0) | 100.0% |
| RPB-v20.alt-03 | 64 | 93.2% (91.0–94.9) | 92.4% (90.5–94.3) | 100.0% |

TEMPO-5 — complexity **5/5** (designed): three streams, 128 candles, nine structured features; infer slow lead/lag beneath faster patterns and unrelated structured signals. Natural missing values and contiguous candle gaps. TRAIN: 256 source pairs; VALIDATION: 128 source pairs per master. Three masters, three fixed heads per method.

Fixed leave-one-patch-out reconstruction error is standardized MAE, averaged within examples and then over examples. Values below are means across three masters. Training used4096 sampled rows /512 available rows =8 equivalent presentations; sampling is with replacement, so these are not complete epochs.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
|---|---:|---:|---:|---:|
| RPB-v20.alt-01 | 512 | 0.7825 | 0.7986 | 20.8 |
| RPB-v20.alt-02 | 512 | 0.7821 | 0.8000 | 21.6 |
| RPB-v20.alt-03 | 512 | 0.7752 | 0.7950 | 27.9 |

Optimization used waveform hierarchical Huber, AdamW0.001, batch8, patch masking25% plus context deletion15%; no dynamics auxiliary. Loss traces, initial/final minibatch losses and the final32-update mean are retained separately from the fixed reconstruction error. GPU memory and registered/trainable parameter counts remain in the JSON report.


The compact ranges describe variation between the three dataset masters. Source-group confidence intervals and paired architecture/training differences remain in the saved reports. This screen uses VALIDATION only; it does not establish held-out TEST or market performance.
