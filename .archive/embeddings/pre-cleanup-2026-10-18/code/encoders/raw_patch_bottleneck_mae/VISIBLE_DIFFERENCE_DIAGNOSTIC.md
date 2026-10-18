# Visible first difference diagnostic

RPB-v13 adds a zero-initialized, bias-free projection of currently visible adjacent backward differences to the early mixer. The original 225,805 parameter values, buffers and TRAIN scaler matched the retained initial model before AdamW; the new 3,072 values increase total capacity to 228,877.

Five candidate trajectories used 512 updates and the same retained TEMPO-3 data, masks, training streams, query targets and fixed heads. All seven parent baseline/control rows are reused from their passed audit. The candidate initial features matched saved untrained early features in all three views before each cohort’s training and head fitting.

Timing task: 256 TRAIN rows / 128 source pairs and 128 VALIDATION rows / 64 source pairs per master, CUDA batch size 8. Each fixed head uses repetitions 2701/2802/2903: Ridge penalty 1, or Tiny tanh width 16 trained with Adam learning rate 0.01 for 100 steps. The native embedding width is 32, with no PCA afterward.

This is a descriptive comparison. The input route and capacity changed together, so results do not isolate a causal effect of differences. No TEST, stress evaluation, model selection or promotion occurred.

## TRAIN fixed-head quality

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| Raw data — no encoder | 576 | 86.41 | 99.69 | 100.00 |
| Mask metadata — no encoder | 288 | 50.00 | 50.00 | 100.00 |
| PCA only — no encoder | 32 | 50.62 | 92.42 | 100.00 |
| Untrained late mixer | 32 | 53.12 | 84.84 | 100.00 |
| Untrained early mixer | 32 | 53.52 | 84.48 | 100.00 |
| RPB-v7.alt-05 | 32 | 58.36 | 81.59 | 100.00 |
| RPB-v10.alt-05 | 32 | 55.70 | 82.06 | 100.00 |
| RPB-v13 | 32 | 56.48 | 78.28 | 100.00 |

Percent accuracy is conditional on supported rows. Means weight all five cohorts equally, then all three fixed repetitions equally; any undefined required score keeps its aggregate undefined. Worst-cohort values and every population, head and interval are retained in the JSON.

## Intact validation

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| Raw data — no encoder | 576 | 51.25 | 50.73 | 100.00 |
| Mask metadata — no encoder | 288 | 50.00 | 50.00 | 100.00 |
| PCA only — no encoder | 32 | 48.44 | 51.15 | 100.00 |
| Untrained late mixer | 32 | 50.78 | 51.56 | 100.00 |
| Untrained early mixer | 32 | 50.47 | 51.87 | 100.00 |
| RPB-v7.alt-05 | 32 | 55.47 | 54.64 | 100.00 |
| RPB-v10.alt-05 | 32 | 52.81 | 54.90 | 100.00 |
| RPB-v13 | 32 | 53.28 | 52.03 | 100.00 |

Percent accuracy is conditional on supported rows. Means weight all five cohorts equally, then all three fixed repetitions equally; any undefined required score keeps its aggregate undefined. Worst-cohort values and every population, head and interval are retained in the JSON.

## Validation with extra 30% coordinate deletion

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

| Method | Size | Linear head % | Neural head % | Coverage % |
|---|---:|---:|---:|---:|
| Raw data — no encoder | 576 | 49.53 | 50.78 | 100.00 |
| Mask metadata — no encoder | 288 | 50.00 | 50.00 | 100.00 |
| PCA only — no encoder | 32 | 48.75 | 50.68 | 100.00 |
| Untrained late mixer | 32 | 48.91 | 50.05 | 100.00 |
| Untrained early mixer | 32 | 49.53 | 50.57 | 100.00 |
| RPB-v7.alt-05 | 32 | 54.22 | 54.01 | 100.00 |
| RPB-v10.alt-05 | 32 | 53.75 | 54.32 | 100.00 |
| RPB-v13 | 32 | 52.81 | 50.52 | 100.00 |

Percent accuracy is conditional on supported rows. Means weight all five cohorts equally, then all three fixed repetitions equally; any undefined required score keeps its aggregate undefined. Worst-cohort values and every population, head and interval are retained in the JSON.

## Candidate cohort spread

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

| Master | TRAIN linear / neural | Intact linear / neural | Deleted linear / neural | Coverage TRAIN / intact / deleted |
|---|---:|---:|---:|---:|
| 75272 | 54.30 / 75.39 | 51.56 / 51.04 | 52.34 / 50.26 | 100.00 / 100.00 / 100.00 |
| 76373 | 56.64 / 75.91 | 53.12 / 52.86 | 50.78 / 48.96 | 100.00 / 100.00 / 100.00 |
| 77474 | 58.98 / 85.42 | 57.03 / 56.25 | 54.69 / 48.44 | 100.00 / 100.00 / 100.00 |
| 78575 | 53.52 / 75.78 | 50.00 / 46.61 | 51.56 / 46.61 | 100.00 / 100.00 / 100.00 |
| 79676 | 58.98 / 78.91 | 54.69 / 53.39 | 54.69 / 58.33 | 100.00 / 100.00 / 100.00 |

## Reconstruction and training cost

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

| Encoder | Updates | Train error | Validation error | GPU training seconds |
|---|---:|---:|---:|---:|
| RPB-v7.alt-05 | 512 | 0.649343 | 0.689378 | 15.877 |
| RPB-v10.alt-05 | 512 | 0.644282 | 0.682958 | 16.789 |
| RPB-v13 | 512 | 0.622320 | 0.664399 | 19.526 |

RPB-v7.alt-05 and RPB-v10.alt-05 each have 225,805 parameters; their reconstruction errors and timers are reused without encoder/query execution. RPB-v13 has 228,877 parameters and is newly measured here.

MAE uses the same original frozen TRAIN scaler and original masked query targets. The GPU training seconds column reports the synchronized loop timer, including CUDA work plus scalar/counter trace work. Full CPU live model/AdamW state capture and checkpoint work belong to the separately measured binding/checkpoint stage; this is not a pure GPU kernel timer.

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

| Master | Candidate TRAIN MAE | Candidate VAL MAE | Candidate loop seconds |
|---|---:|---:|---:|
| 75272 | 0.580456 | 0.639370 | 14.794 |
| 76373 | 0.643044 | 0.712791 | 20.419 |
| 77474 | 0.682413 | 0.676627 | 20.863 |
| 78575 | 0.610844 | 0.640907 | 20.466 |
| 79676 | 0.594843 | 0.652301 | 21.087 |

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

| New candidate stage | Mean seconds per cohort |
|---|---:|
| parent_archive_verification_and_load_seconds | 0.058 |
| binding_checkpoint_and_snapshot_IO_seconds | 6.605 |
| CUDA_initial_parity_transfer_and_verification_seconds | 2.970 |
| CUDA_query_transfer_verification_IO_seconds | 8.622 |
| CUDA_quality_native_transfer_seconds | 3.032 |
| CPU_candidate_heads_and_paired_bootstrap_IO_seconds | 1.468 |

## Linear matched validation effects

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

Candidate minus retained control, in percentage points. Brackets are saved 95% source-group intervals conditional on each fitted head; intervals were copied after the audit, without new resampling.

| Master | Control | Repetition | View | Common rows | Effect [95% interval] |
|---|---|---:|---|---:|---:|
| 75272 | native_early | rep-2701 | validation_intact | 128 | 0.00 [-6.25, 6.25] |
| 75272 | native_early | rep-2701 | validation_deleted | 128 | -1.56 [-7.81, 5.47] |
| 75272 | native_early | rep-2802 | validation_intact | 128 | 0.00 [-6.25, 6.25] |
| 75272 | native_early | rep-2802 | validation_deleted | 128 | -1.56 [-7.81, 5.47] |
| 75272 | native_early | rep-2903 | validation_intact | 128 | 0.00 [-6.25, 6.25] |
| 75272 | native_early | rep-2903 | validation_deleted | 128 | -1.56 [-8.59, 4.69] |
| 75272 | native_late | rep-2701 | validation_intact | 128 | 2.34 [-5.47, 10.16] |
| 75272 | native_late | rep-2701 | validation_deleted | 128 | 3.12 [-3.91, 10.16] |
| 75272 | native_late | rep-2802 | validation_intact | 128 | 2.34 [-6.25, 10.16] |
| 75272 | native_late | rep-2802 | validation_deleted | 128 | 3.12 [-3.91, 10.16] |
| 75272 | native_late | rep-2903 | validation_intact | 128 | 2.34 [-6.25, 10.94] |
| 75272 | native_late | rep-2903 | validation_deleted | 128 | 3.12 [-3.91, 10.16] |
| 76373 | native_early | rep-2701 | validation_intact | 128 | 4.69 [-1.56, 10.94] |
| 76373 | native_early | rep-2701 | validation_deleted | 128 | -3.91 [-10.94, 2.34] |
| 76373 | native_early | rep-2802 | validation_intact | 128 | 4.69 [-1.56, 10.94] |
| 76373 | native_early | rep-2802 | validation_deleted | 128 | -3.91 [-10.16, 2.34] |
| 76373 | native_early | rep-2903 | validation_intact | 128 | 4.69 [-1.56, 10.94] |
| 76373 | native_early | rep-2903 | validation_deleted | 128 | -3.91 [-10.94, 2.34] |
| 76373 | native_late | rep-2701 | validation_intact | 128 | 1.56 [-5.47, 9.38] |
| 76373 | native_late | rep-2701 | validation_deleted | 128 | 0.00 [-6.25, 6.25] |
| 76373 | native_late | rep-2802 | validation_intact | 128 | 1.56 [-6.25, 9.38] |
| 76373 | native_late | rep-2802 | validation_deleted | 128 | 0.00 [-5.47, 6.25] |
| 76373 | native_late | rep-2903 | validation_intact | 128 | 1.56 [-6.25, 8.59] |
| 76373 | native_late | rep-2903 | validation_deleted | 128 | 0.00 [-6.25, 5.47] |
| 77474 | native_early | rep-2701 | validation_intact | 128 | 7.03 [0.78, 14.06] |
| 77474 | native_early | rep-2701 | validation_deleted | 128 | 0.78 [-6.25, 8.59] |
| 77474 | native_early | rep-2802 | validation_intact | 128 | 7.03 [-0.78, 14.84] |
| 77474 | native_early | rep-2802 | validation_deleted | 128 | 0.78 [-6.25, 7.81] |
| 77474 | native_early | rep-2903 | validation_intact | 128 | 7.03 [0.00, 13.28] |
| 77474 | native_early | rep-2903 | validation_deleted | 128 | 0.78 [-6.25, 7.81] |
| 77474 | native_late | rep-2701 | validation_intact | 128 | -2.34 [-8.59, 4.69] |
| 77474 | native_late | rep-2701 | validation_deleted | 128 | -0.78 [-7.03, 5.47] |
| 77474 | native_late | rep-2802 | validation_intact | 128 | -2.34 [-9.38, 4.69] |
| 77474 | native_late | rep-2802 | validation_deleted | 128 | -0.78 [-6.25, 4.69] |
| 77474 | native_late | rep-2903 | validation_intact | 128 | -2.34 [-9.38, 4.69] |
| 77474 | native_late | rep-2903 | validation_deleted | 128 | -0.78 [-7.03, 5.47] |
| 78575 | native_early | rep-2701 | validation_intact | 128 | -4.69 [-11.72, 2.34] |
| 78575 | native_early | rep-2701 | validation_deleted | 128 | -0.78 [-7.81, 6.25] |
| 78575 | native_early | rep-2802 | validation_intact | 128 | -4.69 [-11.72, 2.34] |
| 78575 | native_early | rep-2802 | validation_deleted | 128 | -0.78 [-8.59, 6.25] |
| 78575 | native_early | rep-2903 | validation_intact | 128 | -4.69 [-11.72, 2.34] |
| 78575 | native_early | rep-2903 | validation_deleted | 128 | -0.78 [-7.81, 6.25] |
| 78575 | native_late | rep-2701 | validation_intact | 128 | -5.47 [-14.06, 3.12] |
| 78575 | native_late | rep-2701 | validation_deleted | 128 | -3.12 [-10.94, 5.47] |
| 78575 | native_late | rep-2802 | validation_intact | 128 | -5.47 [-12.50, 3.12] |
| 78575 | native_late | rep-2802 | validation_deleted | 128 | -3.12 [-11.72, 5.47] |
| 78575 | native_late | rep-2903 | validation_intact | 128 | -5.47 [-14.06, 2.34] |
| 78575 | native_late | rep-2903 | validation_deleted | 128 | -3.12 [-11.72, 5.47] |
| 79676 | native_early | rep-2701 | validation_intact | 128 | -4.69 [-11.72, 3.12] |
| 79676 | native_early | rep-2701 | validation_deleted | 128 | 0.78 [-6.25, 8.59] |
| 79676 | native_early | rep-2802 | validation_intact | 128 | -4.69 [-12.50, 2.34] |
| 79676 | native_early | rep-2802 | validation_deleted | 128 | 0.78 [-7.03, 8.59] |
| 79676 | native_early | rep-2903 | validation_intact | 128 | -4.69 [-11.72, 3.12] |
| 79676 | native_early | rep-2903 | validation_deleted | 128 | 0.78 [-7.03, 8.59] |
| 79676 | native_late | rep-2701 | validation_intact | 128 | -7.03 [-14.06, 0.00] |
| 79676 | native_late | rep-2701 | validation_deleted | 128 | -6.25 [-14.06, 1.56] |
| 79676 | native_late | rep-2802 | validation_intact | 128 | -7.03 [-14.06, -0.78] |
| 79676 | native_late | rep-2802 | validation_deleted | 128 | -6.25 [-14.06, 1.56] |
| 79676 | native_late | rep-2903 | validation_intact | 128 | -7.03 [-14.06, 0.00] |
| 79676 | native_late | rep-2903 | validation_deleted | 128 | -6.25 [-14.06, 1.56] |

## Neural secondary matched validation effects

**TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps

Candidate minus retained control, in percentage points. Brackets are saved 95% source-group intervals conditional on each fitted head; intervals were copied after the audit, without new resampling.

| Master | Control | Repetition | View | Common rows | Effect [95% interval] |
|---|---|---:|---|---:|---:|
| 75272 | native_early | rep-2701 | validation_intact | 128 | -3.12 [-12.50, 7.03] |
| 75272 | native_early | rep-2701 | validation_deleted | 128 | -4.69 [-14.84, 4.69] |
| 75272 | native_early | rep-2802 | validation_intact | 128 | -2.34 [-10.16, 5.47] |
| 75272 | native_early | rep-2802 | validation_deleted | 128 | -0.78 [-8.59, 7.03] |
| 75272 | native_early | rep-2903 | validation_intact | 128 | 2.34 [-6.25, 10.16] |
| 75272 | native_early | rep-2903 | validation_deleted | 128 | -3.12 [-10.94, 4.69] |
| 75272 | native_late | rep-2701 | validation_intact | 128 | 3.12 [-3.91, 10.16] |
| 75272 | native_late | rep-2701 | validation_deleted | 128 | -3.91 [-13.28, 5.47] |
| 75272 | native_late | rep-2802 | validation_intact | 128 | -1.56 [-9.38, 4.69] |
| 75272 | native_late | rep-2802 | validation_deleted | 128 | -3.12 [-10.16, 4.69] |
| 75272 | native_late | rep-2903 | validation_intact | 128 | 7.03 [0.00, 14.06] |
| 75272 | native_late | rep-2903 | validation_deleted | 128 | 1.56 [-6.25, 10.16] |
| 76373 | native_early | rep-2701 | validation_intact | 128 | -8.59 [-17.97, 0.00] |
| 76373 | native_early | rep-2701 | validation_deleted | 128 | -7.81 [-14.84, -0.78] |
| 76373 | native_early | rep-2802 | validation_intact | 128 | 1.56 [-8.59, 11.72] |
| 76373 | native_early | rep-2802 | validation_deleted | 128 | -6.25 [-14.84, 1.56] |
| 76373 | native_early | rep-2903 | validation_intact | 128 | 2.34 [-6.25, 10.94] |
| 76373 | native_early | rep-2903 | validation_deleted | 128 | -7.03 [-14.84, 0.78] |
| 76373 | native_late | rep-2701 | validation_intact | 128 | 0.78 [-8.59, 9.38] |
| 76373 | native_late | rep-2701 | validation_deleted | 128 | 0.00 [-9.38, 7.81] |
| 76373 | native_late | rep-2802 | validation_intact | 128 | -3.12 [-13.28, 6.25] |
| 76373 | native_late | rep-2802 | validation_deleted | 128 | -1.56 [-10.16, 7.03] |
| 76373 | native_late | rep-2903 | validation_intact | 128 | -0.78 [-9.38, 8.59] |
| 76373 | native_late | rep-2903 | validation_deleted | 128 | -3.12 [-10.94, 4.69] |
| 77474 | native_early | rep-2701 | validation_intact | 128 | 1.56 [-7.03, 10.16] |
| 77474 | native_early | rep-2701 | validation_deleted | 128 | -5.47 [-14.06, 3.91] |
| 77474 | native_early | rep-2802 | validation_intact | 128 | -3.12 [-12.50, 6.25] |
| 77474 | native_early | rep-2802 | validation_deleted | 128 | -8.59 [-17.19, 0.00] |
| 77474 | native_early | rep-2903 | validation_intact | 128 | 6.25 [-3.91, 15.62] |
| 77474 | native_early | rep-2903 | validation_deleted | 128 | -5.47 [-13.28, 2.34] |
| 77474 | native_late | rep-2701 | validation_intact | 128 | 3.91 [-5.47, 13.28] |
| 77474 | native_late | rep-2701 | validation_deleted | 128 | -2.34 [-10.16, 6.25] |
| 77474 | native_late | rep-2802 | validation_intact | 128 | -2.34 [-10.94, 6.25] |
| 77474 | native_late | rep-2802 | validation_deleted | 128 | -5.47 [-14.06, 3.12] |
| 77474 | native_late | rep-2903 | validation_intact | 128 | 10.16 [0.00, 20.31] |
| 77474 | native_late | rep-2903 | validation_deleted | 128 | -6.25 [-15.62, 2.34] |
| 78575 | native_early | rep-2701 | validation_intact | 128 | -4.69 [-12.50, 3.12] |
| 78575 | native_early | rep-2701 | validation_deleted | 128 | -3.12 [-12.50, 7.03] |
| 78575 | native_early | rep-2802 | validation_intact | 128 | -6.25 [-15.62, 3.12] |
| 78575 | native_early | rep-2802 | validation_deleted | 128 | 2.34 [-6.25, 10.94] |
| 78575 | native_early | rep-2903 | validation_intact | 128 | -6.25 [-14.84, 3.91] |
| 78575 | native_early | rep-2903 | validation_deleted | 128 | -9.38 [-17.19, -1.56] |
| 78575 | native_late | rep-2701 | validation_intact | 128 | -10.94 [-22.66, 0.78] |
| 78575 | native_late | rep-2701 | validation_deleted | 128 | -1.56 [-10.94, 7.03] |
| 78575 | native_late | rep-2802 | validation_intact | 128 | -11.72 [-21.09, -2.34] |
| 78575 | native_late | rep-2802 | validation_deleted | 128 | -9.38 [-19.53, 0.78] |
| 78575 | native_late | rep-2903 | validation_intact | 128 | -11.72 [-20.31, -3.12] |
| 78575 | native_late | rep-2903 | validation_deleted | 128 | -14.84 [-24.22, -5.47] |
| 79676 | native_early | rep-2701 | validation_intact | 128 | -10.16 [-17.97, -2.34] |
| 79676 | native_early | rep-2701 | validation_deleted | 128 | -0.78 [-8.59, 7.81] |
| 79676 | native_early | rep-2802 | validation_intact | 128 | -8.59 [-17.19, 0.78] |
| 79676 | native_early | rep-2802 | validation_deleted | 128 | 1.56 [-7.03, 10.16] |
| 79676 | native_early | rep-2903 | validation_intact | 128 | -3.91 [-14.06, 5.47] |
| 79676 | native_early | rep-2903 | validation_deleted | 128 | 1.56 [-7.03, 9.38] |
| 79676 | native_late | rep-2701 | validation_intact | 128 | -6.25 [-14.84, 1.56] |
| 79676 | native_late | rep-2701 | validation_deleted | 128 | -3.91 [-10.94, 3.12] |
| 79676 | native_late | rep-2802 | validation_intact | 128 | -9.38 [-18.75, 0.00] |
| 79676 | native_late | rep-2802 | validation_deleted | 128 | 2.34 [-5.47, 10.16] |
| 79676 | native_late | rep-2903 | validation_intact | 128 | -6.25 [-14.84, 1.56] |
| 79676 | native_late | rep-2903 | validation_deleted | 128 | -0.78 [-10.16, 8.59] |

## Evidence

The sole saved-arithmetic audit passed 17152339 checks and decoded 280 CPU witness archives in 170.612 seconds. Ordinary CUDA checkpoint bodies were bound by bytes and admission; they were not decoded or run by the CPU reader.

The candidate produced 15 quality exports plus 15 initial-parity exports, 10 query writers / 40 masked forwards, and 15 supported pipelines / 30 individual heads. No baseline, control or untrained head was refitted, and no parent encoder was re-exported.

Full five-cohort objective traces, reconstruction populations, individual intervals, matched pairs and retained parent metadata are saved in [the durable JSON](../../../doc/results/visible_difference_v1.json).

- Card SHA: `75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc`
- Producer SOURCE: `14c99891d6a4a9b1831116431a31e3eef9b4b352565db63f1df6889828332bdb`
- Reader SHA: `74fde8c189e22bc9e590a0f5c7059fadface9540500f8861e035b18d11c164c8`
- Completed inventory SHA: `384ed378bd0cd563e2f0e38f65c35a20d4e8db746dac4830c86cb8f29129f2fb`
- report SHA: `24c63efd046d8065db7bdfcd598d2715670ad47f134eff77ceb210ab61926f4b`
- audit SHA: `b4849d9aca2530523b12093706f8f52229bc512cf9947447c68870d2d31ee070`
- parent_summary SHA: `1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4`
