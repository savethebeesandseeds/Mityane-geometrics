# TEMPO-3 structured harder timing comparison

New measurements on a separately specified timing recipe. Designed complexity 4/5 is ordinal, fixed before measurement; it is not an accuracy score.

Each of five paired masters has 256 TRAIN examples from 128 independent sources and 128 VALIDATION examples from 64 separate sources. Ten fresh CUDA trajectories complete 512 updates at batch 8 with no skips (16 equivalent presentations). Native exports are 32 numbers; each design has 225,805 parameters. No TEST/stress or architecture promotion.

The late RPB-v7.alt-05 mixer follows the temporal pass. RPB-v10.alt-05 mixes before it and retains an independent unmixed local temporal pass. Equal parameters do not imply equal compute. These fresh harder-data results do not change original saved v7 weights.

Ridge penalty 1 and tanh16/Adam .01/100 updates use repetitions 2701/2802/2903, fitted once on TRAIN and retained for both VALIDATION views. Raw/PCA share one TRAIN outer map per cohort; encoder features have no PCA. Raw/mask heads contain 1,154/578 linear and 9,266/4,658 neural parameters; native/PCA32 heads contain 66/562.

Tables use equal-master means after averaging all three fixed head repetitions. Every cohort, unsupported fit, conditional source-group interval and paired common-support record remains in the JSON. Undefined cohorts keep means undefined; they are not removed. Conditional interval bounds are never averaged into retraining uncertainty.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation intact.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder | 576 | 51.25 | 50.73 | 100.00 |
| Mask metadata — no encoder | 288 | 50.00 | 50.00 | 100.00 |
| PCA only — no encoder | 32 | 48.44 | 51.15 | 100.00 |
| Untrained late mixer | 32 | 50.78 | 51.56 | 100.00 |
| Untrained early mixer | 32 | 50.47 | 51.88 | 100.00 |
| RPB-v7.alt-05 | 32 | 55.47 | 54.64 | 100.00 |
| RPB-v10.alt-05 | 32 | 52.81 | 54.90 | 100.00 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation intact · variation across five retrainings.

| Encoder | Mean linear % | Worst linear % | Linear range % | Mean neural % | Worst neural % | Neural range % | Coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RPB-v7.alt-05 | 55.47 | 49.22 | 49.22–61.72 | 54.64 | 48.18 | 48.18–60.68 | 100.00 |
| RPB-v10.alt-05 | 52.81 | 48.44 | 48.44–59.38 | 54.90 | 52.08 | 52.08–60.94 | 100.00 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation intact · all paired cohorts.

| Master | RPB-v7.alt-05 linear % | RPB-v10.alt-05 linear % | Difference pp | Late/early coverage % |
| --- | --- | --- | --- | --- |
| 75272 | 49.22 | 51.56 | 2.34 | 100.00/100.00 |
| 76373 | 51.56 | 48.44 | -3.12 | 100.00/100.00 |
| 77474 | 59.38 | 50.00 | -9.38 | 100.00/100.00 |
| 78575 | 55.47 | 54.69 | -0.78 | 100.00/100.00 |
| 79676 | 61.72 | 59.38 | -2.34 | 100.00/100.00 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation intact · all paired cohorts · fixed neural head.

| Master | RPB-v7.alt-05 neural % | RPB-v10.alt-05 neural % | Difference pp | Late/early coverage % |
| --- | --- | --- | --- | --- |
| 75272 | 48.18 | 52.08 | 3.91 | 100.00/100.00 |
| 76373 | 53.91 | 54.43 | 0.52 | 100.00/100.00 |
| 77474 | 52.34 | 54.69 | 2.34 | 100.00/100.00 |
| 78575 | 58.07 | 52.34 | -5.73 | 100.00/100.00 |
| 79676 | 60.68 | 60.94 | 0.26 | 100.00/100.00 |

Per-cohort differences above use each method’s own support. The saved paired intervals use exact common support; both populations and every repetition are retained in JSON.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation intact · within-master conditional paired uncertainty.

| Master | Head repetition | Common coverage % | Linear effect [95% interval] pp | Neural effect [95% interval] pp |
| --- | --- | --- | --- | --- |
| 75272 | rep-2701 | 100.00 | 2.34 [-5.47, 10.16] | 6.25 [-3.12, 15.62] |
| 75272 | rep-2802 | 100.00 | 2.34 [-4.69, 10.16] | 0.78 [-7.03, 8.59] |
| 75272 | rep-2903 | 100.00 | 2.34 [-5.47, 10.16] | 4.69 [-2.34, 12.50] |
| 76373 | rep-2701 | 100.00 | -3.12 [-10.94, 4.69] | 9.38 [-1.56, 18.75] |
| 76373 | rep-2802 | 100.00 | -3.12 [-10.94, 4.69] | -4.69 [-12.50, 3.91] |
| 76373 | rep-2903 | 100.00 | -3.12 [-11.72, 4.69] | -3.12 [-12.50, 5.47] |
| 77474 | rep-2701 | 100.00 | -9.38 [-14.84, -4.69] | 2.34 [-6.25, 10.94] |
| 77474 | rep-2802 | 100.00 | -9.38 [-14.84, -3.91] | 0.78 [-7.81, 8.59] |
| 77474 | rep-2903 | 100.00 | -9.38 [-14.06, -4.69] | 3.91 [-6.25, 13.28] |
| 78575 | rep-2701 | 100.00 | -0.78 [-7.81, 7.03] | -6.25 [-17.19, 4.69] |
| 78575 | rep-2802 | 100.00 | -0.78 [-8.59, 7.03] | -5.47 [-14.84, 3.91] |
| 78575 | rep-2903 | 100.00 | -0.78 [-8.59, 7.03] | -5.47 [-15.62, 3.91] |
| 79676 | rep-2701 | 100.00 | -2.34 [-8.59, 3.91] | 3.91 [-3.12, 11.72] |
| 79676 | rep-2802 | 100.00 | -2.34 [-9.38, 3.91] | -0.78 [-10.16, 7.81] |
| 79676 | rep-2903 | 100.00 | -2.34 [-9.38, 3.91] | -2.34 [-10.94, 7.03] |

Effects are early minus late on common source support. These source-group percentile intervals hold each saved encoder/readout fixed; they do not quantify encoder retraining uncertainty. Every fixed head repetition is shown.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation deleted.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder | 576 | 49.53 | 50.78 | 100.00 |
| Mask metadata — no encoder | 288 | 50.00 | 50.00 | 100.00 |
| PCA only — no encoder | 32 | 48.75 | 50.68 | 100.00 |
| Untrained late mixer | 32 | 48.91 | 50.05 | 100.00 |
| Untrained early mixer | 32 | 49.53 | 50.57 | 100.00 |
| RPB-v7.alt-05 | 32 | 54.22 | 54.01 | 100.00 |
| RPB-v10.alt-05 | 32 | 53.75 | 54.32 | 100.00 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation deleted · variation across five retrainings.

| Encoder | Mean linear % | Worst linear % | Linear range % | Mean neural % | Worst neural % | Neural range % | Coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RPB-v7.alt-05 | 54.22 | 49.22 | 49.22–60.94 | 54.01 | 50.52 | 50.52–59.11 | 100.00 |
| RPB-v10.alt-05 | 53.75 | 52.34 | 52.34–54.69 | 54.32 | 50.00 | 50.00–57.55 | 100.00 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation deleted · all paired cohorts.

| Master | RPB-v7.alt-05 linear % | RPB-v10.alt-05 linear % | Difference pp | Late/early coverage % |
| --- | --- | --- | --- | --- |
| 75272 | 49.22 | 53.91 | 4.69 | 100.00/100.00 |
| 76373 | 50.78 | 54.69 | 3.91 | 100.00/100.00 |
| 77474 | 55.47 | 53.91 | -1.56 | 100.00/100.00 |
| 78575 | 54.69 | 52.34 | -2.34 | 100.00/100.00 |
| 79676 | 60.94 | 53.91 | -7.03 | 100.00/100.00 |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation deleted · all paired cohorts · fixed neural head.

| Master | RPB-v7.alt-05 neural % | RPB-v10.alt-05 neural % | Difference pp | Late/early coverage % |
| --- | --- | --- | --- | --- |
| 75272 | 52.08 | 53.12 | 1.04 | 100.00/100.00 |
| 76373 | 50.52 | 55.99 | 5.47 | 100.00/100.00 |
| 77474 | 53.12 | 54.95 | 1.82 | 100.00/100.00 |
| 78575 | 55.21 | 50.00 | -5.21 | 100.00/100.00 |
| 79676 | 59.11 | 57.55 | -1.56 | 100.00/100.00 |

Per-cohort differences above use each method’s own support. The saved paired intervals use exact common support; both populations and every repetition are retained in JSON.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: validation deleted · within-master conditional paired uncertainty.

| Master | Head repetition | Common coverage % | Linear effect [95% interval] pp | Neural effect [95% interval] pp |
| --- | --- | --- | --- | --- |
| 75272 | rep-2701 | 100.00 | 4.69 [-2.34, 10.94] | 0.78 [-7.03, 8.59] |
| 75272 | rep-2802 | 100.00 | 4.69 [-1.56, 10.94] | -2.34 [-10.16, 5.47] |
| 75272 | rep-2903 | 100.00 | 4.69 [-1.56, 10.94] | 4.69 [-3.12, 11.72] |
| 76373 | rep-2701 | 100.00 | 3.91 [-3.91, 10.94] | 7.81 [-1.56, 17.19] |
| 76373 | rep-2802 | 100.00 | 3.91 [-3.12, 10.94] | 4.69 [-4.69, 14.84] |
| 76373 | rep-2903 | 100.00 | 3.91 [-3.12, 10.94] | 3.91 [-5.47, 12.50] |
| 77474 | rep-2701 | 100.00 | -1.56 [-8.59, 6.25] | 3.12 [-6.25, 12.50] |
| 77474 | rep-2802 | 100.00 | -1.56 [-7.81, 5.47] | 3.12 [-5.47, 11.72] |
| 77474 | rep-2903 | 100.00 | -1.56 [-8.59, 6.25] | -0.78 [-9.38, 7.81] |
| 78575 | rep-2701 | 100.00 | -2.34 [-9.38, 4.69] | 1.56 [-8.59, 10.94] |
| 78575 | rep-2802 | 100.00 | -2.34 [-10.16, 4.69] | -11.72 [-21.09, -2.34] |
| 78575 | rep-2903 | 100.00 | -2.34 [-9.38, 4.69] | -5.47 [-14.84, 3.12] |
| 79676 | rep-2701 | 100.00 | -7.03 [-13.28, -0.78] | -3.12 [-11.72, 5.47] |
| 79676 | rep-2802 | 100.00 | -7.03 [-13.28, -0.78] | 0.78 [-6.25, 8.59] |
| 79676 | rep-2903 | 100.00 | -7.03 [-13.28, -0.78] | -2.34 [-10.94, 6.25] |

Effects are early minus late on common source support. These source-group percentile intervals hold each saved encoder/readout fixed; they do not quantify encoder retraining uncertainty. Every fixed head repetition is shown.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: original whole-patch TRAIN and intact VALIDATION queries.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| RPB-v7.alt-05 | 512 | 0.649343 | 0.689378 | 15.88 |
| RPB-v10.alt-05 | 512 | 0.644282 | 0.682958 | 16.79 |

Errors are standardized MAE on fixed hidden-target queries, distinct from the random-mask Huber trace and classifier accuracy. The synchronized training-loop timer includes CPU trace capture and excludes checkpoint writing, feature/query extraction and CPU head work; it is not pure GPU kernel time. Ordinary updates jointly train the decoder; extra decoder-only calibration is zero. Every 512-entry loss/gradient trace is retained.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: observed-only analytic information diagnostic.

| View | Conditional accuracy % | Coverage % | Full-population correctness % | Valid rows per master / total |
| --- | --- | --- | --- | --- |
| training | 100.00 | 100.00 | 100.00 | 256/256/256/256/256 / 256 |
| validation_intact | 100.00 | 100.00 | 100.00 | 128/128/128/128/128 / 128 |
| validation_deleted | 100.00 | 100.00 | 100.00 | 128/128/128/128/128 / 128 |

The analytic rule uses legal observed values/masks only, cancels constant offsets and has zero fitted heads. It abstains below four supported centres or at zero margin. It is a declared information check, not Bayes optimality or a classifier head result.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: premodel information admission v1 · fresh engineering cohorts.

| Rule | Master | View | Valid / total | Conditional accuracy % | Coverage % | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| same-feature v1 | 80383 | intact | 1024/1024 | 100.00 | 100.0000 | PASS |
| same-feature v1 | 80383 | deleted | 950/1024 | 100.00 | 92.7734 | FAIL |
| same-feature v1 | 81484 | intact | 1024/1024 | 100.00 | 100.0000 | PASS |
| same-feature v1 | 81484 | deleted | 900/1024 | 100.00 | 87.8906 | FAIL |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: premodel information admission v2 · fresh engineering cohorts.

| Rule | Master | View | Valid / total | Conditional accuracy % | Coverage % | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| cross-feature v2 | 82585 | intact | 1024/1024 | 100.00 | 100.0000 | PASS |
| cross-feature v2 | 82585 | deleted | 1020/1024 | 100.00 | 99.6094 | PASS |
| cross-feature v2 | 83686 | intact | 1024/1024 | 100.00 | 100.0000 | PASS |
| cross-feature v2 | 83686 | deleted | 1024/1024 | 100.00 | 100.0000 | PASS |

The same-feature v1 rule failed its predeclared deleted-coverage gate. The separately specified cross-feature v2 rule passed unchanged accuracy/coverage thresholds on fresh seeds. Both use the identical TEMPO-3 generator; this is not a paired change estimate or encoder-quality result. The original failed receipt/report/JSON remain preserved.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains, offsets and 3-tick channel gaps · view: whole experiment · descriptive mean per cohort.

| Cost scope | Seconds |
| --- | --- |
| generation and observation io seconds | 0.05 |
| binding and checkpoint io seconds | 23.70 |
| CUDA query transfer verification io seconds | 16.70 |
| CUDA native transfer verification seconds | 12.58 |
| CPU baseline preparation io seconds | 0.09 |
| CPU head bootstrap io seconds | 6.27 |
| CPU analytic observed only and asset IO seconds | 0.05 |

Mixed scopes include transfers, verification and/or I/O as their names state. CPU head scope includes fixed fit, prediction, bootstrap and asset writing. Audit cost is separate. Retain all five cohorts when interpreting means and worst-cohort reliability; no automatic numerical promotion rule or best-seed selection applies.

Provenance, complete counters, all analytic counts, head predictions summarized as metadata, marginal/paired intervals, original-query summaries and traces are retained in `structured_hard_timing_comparison_v2.json`. The formatter reads JSON metadata only; it runs no encoder, head, PCA, bootstrap or independent audit.

Audit: 52934423 checks / 715 saved archive decodes. Source `9cd4af69f71b1cdc0368987185aa836040d349ec7936b2ef6b11e1838c2b6eb8`, reader `5a77eee23619d7efbc71dcb086680508b9f54f68fb761a39bfeb5aeabd2b9819`, inventory `9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa`.
