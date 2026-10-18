# Early mixer reliability diagnostic

Protocol: `early-mixer-reliability-v1`. Completed known-VALIDATION comparison with an independent saved-arithmetic audit. No TEST or stress was generated or read.

RPB-v7.alt-02 is a fresh late-mixer group using the preserved coordinate-deletion 0.15 recipe. RPB-v10 moves aligned width-64 channel mixing before temporal encoding. Both have 225,805 parameters and serve 32 native numbers; the early model retains a separate independent local temporal pass. Equal parameters do not mean equal compute.

Five paired timing masters: 19119/20220/21321/22422/23523. Each trajectory used 256 TRAIN rows from 128 source groups, 128 VALIDATION rows from 64 disjoint groups, batch size 8 and 512 unskipped CUDA AdamW updates. There are ten new encoder trajectories, not thirty runs from repeated heads. Each saw 4,096 sampled rows (16 equivalent presentations, not guaranteed epochs).

Amplitude data masters 24624/25725/26826/27927/29028 provide separate TRAIN/VALIDATION head data. They never enter encoder or scaler fitting. Three fixed head repetitions use ridge penalty 1 and 16 tanh units, Adam at 0.01 and 100 updates. Native heads receive exact 32 native numbers without PCA; raw data at width 576 includes 288 scaled values and 288 masks. PCA at width 32 is fitted only to raw TRAIN; mask-only has 288 inputs. Head parameter counts are 66/562 at width 32, 578/4,658 at width 288 and 1,154/9,266 at width 576 (linear/neural).

Both untrained controls are retained because equal initial weights do not imply equal outputs for different computation order. Each fitted head is reused intact and with one fixed extra 30% coordinate deletion view. Coverage is valid / declared 128 VALIDATION rows; accuracy is conditional on that support. All fixed head and paired source intervals are retained in the durable JSON.

Earlier mixing improved mean timing linear accuracy by 1.40625 percentage points intact and 3.59375 points with deletion. The worst timing cohort rose from 85.9375% to 90.625% intact and from 78.125% to 86.71875% with deletion. Three pairs improved and two regressed in both views; master 22422 fell by 7.03125 and 7.8125 points. All methods retained 100% coverage.

This is a promising timing result with a transfer tradeoff. Deleted amplitude linear accuracy fell by 1.40625 points and neural accuracy by 3.8541667 points. Intact amplitude is near ceiling even for both untrained controls, so it supplies little evidence of a gain from learned timing training. Mean fixed-query reconstruction errors improved slightly, while three of five individual pairs worsened. RPB-v10 remains experimental; no encoder is promoted or replaced.

## Timing (lag sign): intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.7188 | 51.8229 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 52.1875 | 74.2188 | 100.0000 |
| Untrained late encoder | 32 | 50.6250 | 60.9896 | 100.0000 |
| Untrained early encoder | 32 | 52.0312 | 64.1667 | 100.0000 |
| RPB-v7.alt-02 | 32 | 94.3750 | 99.1667 | 100.0000 |
| RPB-v10 | 32 | 95.7812 | 99.2708 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-02 linear / neural % | RPB-v10 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 19119 / 19119 | 100.0000 / 100.0000 | 98.4375 / 99.7396 | -1.5625 | 100.0000 / 100.0000 |
| 20220 / 20220 | 96.8750 / 100.0000 | 99.2188 / 100.0000 | +2.3438 | 100.0000 / 100.0000 |
| 21321 / 21321 | 85.9375 / 98.4375 | 90.6250 / 100.0000 | +4.6875 | 100.0000 / 100.0000 |
| 22422 / 22422 | 97.6562 / 100.0000 | 90.6250 / 96.6146 | -7.0312 | 100.0000 / 100.0000 |
| 23523 / 23523 | 91.4062 / 97.3958 | 100.0000 / 100.0000 | +8.5938 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Timing (lag sign): extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.2188 | 48.8542 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 51.4062 | 55.2604 | 100.0000 |
| Untrained late encoder | 32 | 48.5938 | 52.0833 | 100.0000 |
| Untrained early encoder | 32 | 49.0625 | 53.4375 | 100.0000 |
| RPB-v7.alt-02 | 32 | 90.6250 | 92.4479 | 100.0000 |
| RPB-v10 | 32 | 94.2188 | 97.1875 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-02 linear / neural % | RPB-v10 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 19119 / 19119 | 97.6562 / 97.1354 | 96.0938 / 99.7396 | -1.5625 | 100.0000 / 100.0000 |
| 20220 / 20220 | 94.5312 / 98.9583 | 98.4375 / 99.4792 | +3.9062 | 100.0000 / 100.0000 |
| 21321 / 21321 | 78.1250 / 80.2083 | 86.7188 / 90.6250 | +8.5938 | 100.0000 / 100.0000 |
| 22422 / 22422 | 98.4375 / 99.2188 | 90.6250 / 97.3958 | -7.8125 | 100.0000 / 100.0000 |
| 23523 / 23523 | 84.3750 / 86.7188 | 99.2188 / 98.6979 | +14.8438 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Amplitude transfer: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 64.5312 | 62.5521 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 54.0625 | 65.6250 | 100.0000 |
| Untrained late encoder | 32 | 98.9062 | 98.9583 | 100.0000 |
| Untrained early encoder | 32 | 98.9062 | 98.6458 | 100.0000 |
| RPB-v7.alt-02 | 32 | 99.5312 | 99.5833 | 100.0000 |
| RPB-v10 | 32 | 99.6875 | 98.9583 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-02 linear / neural % | RPB-v10 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 19119 / 24624 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 20220 / 25725 | 99.2188 / 98.4375 | 99.2188 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 21321 / 26826 | 100.0000 / 100.0000 | 99.2188 / 97.3958 | -0.7812 | 100.0000 / 100.0000 |
| 22422 / 27927 | 98.4375 / 99.4792 | 100.0000 / 97.9167 | +1.5625 | 100.0000 / 100.0000 |
| 23523 / 29028 | 100.0000 / 100.0000 | 100.0000 / 99.4792 | +0.0000 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Amplitude transfer: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 64.5312 | 46.4062 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 52.3438 | 56.6146 | 100.0000 |
| Untrained late encoder | 32 | 89.8438 | 87.4479 | 100.0000 |
| Untrained early encoder | 32 | 89.0625 | 88.6458 | 100.0000 |
| RPB-v7.alt-02 | 32 | 97.9688 | 95.2083 | 100.0000 |
| RPB-v10 | 32 | 96.5625 | 91.3542 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-02 linear / neural % | RPB-v10 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 19119 / 24624 | 99.2188 / 97.6562 | 96.0938 / 92.7083 | -3.1250 | 100.0000 / 100.0000 |
| 20220 / 25725 | 96.0938 / 92.1875 | 99.2188 / 96.0938 | +3.1250 | 100.0000 / 100.0000 |
| 21321 / 26826 | 99.2188 / 96.3542 | 96.8750 / 89.3229 | -2.3438 | 100.0000 / 100.0000 |
| 22422 / 27927 | 95.3125 / 93.7500 | 96.0938 / 86.9792 | +0.7812 | 100.0000 / 100.0000 |
| 23523 / 29028 | 100.0000 / 96.0938 | 94.5312 / 91.6667 | -5.4688 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Training and reconstruction

Fixed ordinary whole-patch query standardized MAE uses the same original Q, targets, support and TRAIN scaler in each pair. Inference adds no training context deletion. Sampled optimization Huber and its complete 512-step traces are separate evidence; amplitude reconstruction was not measured.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-02 | 512 | 0.072575341 | 0.074894305 | 14.225696 |
| RPB-v10 | 512 | 0.071023592 | 0.073852599 | 14.443317 |

| Timing master | RPB-v7.alt-02 TRAIN / VALIDATION MAE | RPB-v10 TRAIN / VALIDATION MAE | GPU update seconds late / early |
| --- | ---: | ---: | ---: |
| 19119 | 0.066785788 / 0.070699063 | 0.074006805 / 0.079711106 | 14.936121 / 14.654205 |
| 20220 | 0.066110291 / 0.067751829 | 0.069079518 / 0.069956523 | 14.539450 / 14.545427 |
| 21321 | 0.069001431 / 0.069804535 | 0.072005228 / 0.073312761 | 13.811788 / 14.325689 |
| 22422 | 0.083228087 / 0.086510588 | 0.077125886 / 0.081924096 | 14.051766 / 14.307997 |
| 23523 | 0.077751108 / 0.079705509 | 0.062900526 / 0.064358506 | 13.789355 / 14.383270 |

Synchronized cumulative update-loop times exclude head fitting, extraction, reconstruction queries and checkpoint writing. They are descriptive single-run measurements. Additional stages below include CPU transfer, parent verification, file hashing and I/O; pure CUDA kernel time is unmeasured.

| Mixed wall-time scope | Total seconds | Mean per paired cohort seconds |
| --- | ---: | ---: |
| `generation_and_observation_io_seconds` | 0.684812 | 0.136962 |
| `binding_and_checkpoint_io_seconds` | 40.783997 | 8.156799 |
| `CUDA_query_transfer_verification_io_seconds` | 82.150832 | 16.430166 |
| `CUDA_native_transfer_verification_seconds` | 97.938261 | 19.587652 |
| `CPU_baseline_preparation_io_seconds` | 0.801433 | 0.160287 |
| `CPU_head_bootstrap_io_seconds` | 59.310860 | 11.862172 |

## Evidence and limits

Actual supported budgets: 210 pipelines / 420 heads (planned 210 / 420). The five paired cohorts retain 20 point0/512 checkpoints, both initial controls, all ten complete traces and all original-query supports. 120 full native export callbacks and 20 query-writer calls produced 80 patch-query banks. No extra decoder calibration occurred.

Independent audit PASS: 96,352,345 checks, 1,215 CPU archive decodes, 380.147750 seconds. It independently replays saved maps/logits/support/query reductions and source effects; it runs no model, CUDA forward, autodiff, optimizer, head fit or PCA/SVD fit. Ordinary CUDA checkpoint bodies remain byte-bound, with semantics established by source/engineering admission and CPU companions. Original sampler/Torch streams are source/admission-bound, not GPU reenacted.

No numeric automatic promotion rule was declared. Assess timing intact/deleted linear accuracy, worst-cohort accuracy, fixed TRAIN/VALIDATION MAE, neural regressions and amplitude separately before any next change. This report does not replace original RPB-v7 weights or claim a causal internal mechanism.

[Frozen prospective card](../../evaluation/cards/early_mixer_reliability_v1.md). [Durable full summary](../../../doc/results/early_mixer_reliability_v1.json).

The first engineering admission passed with a 134-file source closure. Before any quality generation, the closure was corrected to 146 files and a new actual CUDA admission passed under the measured source fingerprint. Both admission records are preserved; the earlier capture is not used as a quality binding. [Initial admission](../../../output/runs/rpb-early-mixer-reliability/admission/admission-sGWNLS/passed.json) and [final admission](../../../output/runs/rpb-early-mixer-reliability/admission/admission-0zaRdL/passed.json).

The original RPB-v7 and its previous decoder-calibration bundle remain unchanged: 1,710 files verified, zero changed files, zero tensor decodes and zero model/head execution in the preservation check. [Preservation record](../../../output/runs/rpb-early-mixer-reliability/report-tools/original-v7-preserved-20261009-before-quality-result.json).

[Completed capsule](../../../output/runs/rpb-early-mixer-reliability/early-mixer-reliability-b6DsL0/results/report.json), [inventory](../../../output/runs/rpb-early-mixer-reliability/early-mixer-reliability-b6DsL0/artifact-integrity.json), [independent validation](../../../output/runs/rpb-early-mixer-reliability/audit-tools/run-b6DsL0-v2/validation.json), [released reader](../../../output/runs/rpb-early-mixer-reliability/audit-tools/independent-early-mixer-20261009-v2-released-b6DsL0/validate_early_mixer_reliability.py) and [renderer release proof](../../../output/runs/rpb-early-mixer-reliability/report-tools/released-renderer-b6DsL0-v1/renderer-release-proof.json).

The proposed next action is a fresh matched continuous learning curve for RPB-v7.alt-03 and RPB-v10.alt-01, saved at 0, 512, 1,024 and 2,048 updates under one unchanged live optimizer per trajectory and fixed heads. Earlier checkpoint bytes and full trace prefixes would be bound before each quality feature/query is exported once after training finishes. This requires a separate frozen card and genuinely new source seeds. Timing masters are 30130/31231/32332/33433/34534; amplitude data masters 35635/36736/37837/38938/40039 are reserved for initial and final-point transfer only. It is not a new result or permission to rescue either model by tuning losses, deletion rates or heads against these cohorts.

Metadata/source bindings:

- `admission_log_sha256`: `b7294025466203152f87f4e0b4cc707407ff66d05ee071c99bcc72b844b00831`
- `admission_sha256`: `d2a08ef26f03d00dcdb4e4c06ef9ff40c69ff2569c0f92fd3ae251d80329ce66`
- `audit_reader`: `/embedding/output/runs/rpb-early-mixer-reliability/audit-tools/independent-early-mixer-20261009-v2-released-b6DsL0/validate_early_mixer_reliability.py`
- `audit_sha256`: `e07931e860bec3c719bb377a7b76454d6b13c331b89c7fa1cf3c13062c710ccd`
- `audit_validation`: `/embedding/output/runs/rpb-early-mixer-reliability/audit-tools/run-b6DsL0-v2/validation.json`
- `capsule`: `/embedding/output/runs/rpb-early-mixer-reliability/early-mixer-reliability-b6DsL0`
- `card_sha256`: `a4c3aa956a28e97d39f4b9c181c46b4d7cf5125b27516a5a5a1043b7745718f0`
- `inventory_sha256`: `426a1c17288f6b2781d25ca6c5ea94af7f1bb8f041bcc07a9a5412b92e1d50f5`
- `reader_sha256`: `9d68892a664c019fb158dc8f58c9a1a6192b7aa5161ac33cee0cdcdcf8f172fa`
- `renderer_sha256`: `d86e7b669e798fb85efc69e9e4ac866b9c003d664d4e794c10a0ec8b07c17d29`
- `source_fingerprint`: `583ee5362d73b689910c8700fea748f2092506856489f37961e50e9e0bdfb006`
