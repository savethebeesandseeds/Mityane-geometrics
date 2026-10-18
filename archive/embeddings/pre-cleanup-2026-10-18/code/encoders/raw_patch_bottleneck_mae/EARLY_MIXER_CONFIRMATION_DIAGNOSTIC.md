# Early mixer architecture confirmation

Protocol: `early-mixer-confirmation-v1`. Completed known-VALIDATION comparison with an independent saved-arithmetic audit. No TEST or stress was generated or read.

RPB-v7.alt-04 is a fresh late-mixer group using the preserved coordinate15 recipe. RPB-v10.alt-04 moves aligned W64 channel mixing before temporal encoding. Both have 225,805 parameters and serve native32; the early model retains a separate independent local temporal pass. Equal parameters do not mean equal compute.

Five paired timing masters: 64161/65262/66363/67464/68565. Each trajectory used 256 TRAIN rows from 128 source groups, 128 VALIDATION rows from 64 disjoint groups, batch 8 and 512 unskipped CUDA AdamW updates. There are ten new encoder trajectories, not thirty runs from repeated heads. Each saw 4,096 sampled rows (16 equivalent presentations, not guaranteed epochs).

Amplitude data masters 69666/70767/71868/72969/74070 provide separate TRAIN/VALIDATION head data. They never enter encoder or scaler fitting. Three fixed head repetitions use ridge penalty 1 and tanh 16 / Adam 0.01 / 100 updates. Native heads receive exact 32 without PCA; raw 576 includes 288 scaled values and 288 masks. PCA 32 is fitted only to raw TRAIN; mask-only has 288 inputs. Head parameter counts are 66/562 at 32, 578/4,658 at 288 and 1,154/9,266 at 576 (linear/neural).

External cohort fit identity is early-mixer-confirmation-v1/lag_sign; the unchanged training implementation receives an explicit copy labelled early-mixer-reliability-v1/lag_sign. The old checkpoint and snapshot metadata keep that truthful implementation identity. A new fifth .confirmation.pt companion and separate confirmation snapshot audit bind the new cohort, original TRAIN/source order/scaler, counters, source scopes and four original parent bytes. No historical quality payload is an input. The new binding is admission metadata and changes no numerical training or serving function.

Both untrained controls are retained because equal initial weights do not imply equal outputs for different computation order. Each fitted head is reused intact and with one fixed extra 30% coordinate deletion view. Coverage is valid / declared 128 VALIDATION rows; accuracy is conditional on that support. All fixed head and paired source intervals are retained in the durable JSON.

## Timing (lag sign): intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 50.7812 | 54.2708 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 47.8125 | 74.4271 | 100.0000 |
| Untrained late encoder | 32 | 54.0625 | 69.1667 | 100.0000 |
| Untrained early encoder | 32 | 52.3438 | 68.4896 | 100.0000 |
| RPB-v7.alt-04 | 32 | 94.2188 | 96.7188 | 100.0000 |
| RPB-v10.alt-04 | 32 | 95.0000 | 96.9271 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-04 linear / neural % | RPB-v10.alt-04 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 64161 / 64161 | 97.6562 / 99.7396 | 100.0000 / 100.0000 | +2.3438 | 100.0000 / 100.0000 |
| 65262 / 65262 | 96.8750 / 100.0000 | 80.4688 / 84.6354 | -16.4062 | 100.0000 / 100.0000 |
| 66363 / 66363 | 96.8750 / 100.0000 | 100.0000 / 100.0000 | +3.1250 | 100.0000 / 100.0000 |
| 67464 / 67464 | 95.3125 / 95.8333 | 94.5312 / 100.0000 | -0.7812 | 100.0000 / 100.0000 |
| 68565 / 68565 | 84.3750 / 88.0208 | 100.0000 / 100.0000 | +15.6250 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Timing (lag sign): extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 52.1875 | 50.7812 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.7500 | 54.7396 | 100.0000 |
| Untrained late encoder | 32 | 50.0000 | 52.2396 | 100.0000 |
| Untrained early encoder | 32 | 50.4687 | 53.4896 | 100.0000 |
| RPB-v7.alt-04 | 32 | 88.2812 | 94.7396 | 100.0000 |
| RPB-v10.alt-04 | 32 | 92.8125 | 95.8854 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-04 linear / neural % | RPB-v10.alt-04 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 64161 / 64161 | 79.6875 / 98.6979 | 100.0000 / 100.0000 | +20.3125 | 100.0000 / 100.0000 |
| 65262 / 65262 | 97.6562 / 100.0000 | 75.0000 / 79.9479 | -22.6562 | 100.0000 / 100.0000 |
| 66363 / 66363 | 92.9688 / 100.0000 | 98.4375 / 100.0000 | +5.4688 | 100.0000 / 100.0000 |
| 67464 / 67464 | 92.9688 / 91.1458 | 90.6250 / 99.4792 | -2.3438 | 100.0000 / 100.0000 |
| 68565 / 68565 | 78.1250 / 83.8542 | 100.0000 / 100.0000 | +21.8750 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Amplitude transfer: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 62.5000 | 62.9167 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 54.8438 | 66.3542 | 100.0000 |
| Untrained late encoder | 32 | 99.0625 | 99.2708 | 100.0000 |
| Untrained early encoder | 32 | 99.2188 | 98.7500 | 100.0000 |
| RPB-v7.alt-04 | 32 | 99.8438 | 99.4271 | 100.0000 |
| RPB-v10.alt-04 | 32 | 99.8438 | 99.7917 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-04 linear / neural % | RPB-v10.alt-04 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 64161 / 69666 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 65262 / 70767 | 99.2188 / 98.1771 | 100.0000 / 100.0000 | +0.7812 | 100.0000 / 100.0000 |
| 66363 / 71868 | 100.0000 / 99.2188 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 67464 / 72969 | 100.0000 / 99.7396 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 68565 / 74070 | 100.0000 / 100.0000 | 99.2188 / 98.9583 | -0.7812 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Amplitude transfer: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 62.6562 | 46.3021 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 57.3438 | 54.7917 | 100.0000 |
| Untrained late encoder | 32 | 86.4062 | 80.4688 | 100.0000 |
| Untrained early encoder | 32 | 86.7188 | 79.6875 | 100.0000 |
| RPB-v7.alt-04 | 32 | 97.6562 | 92.8646 | 100.0000 |
| RPB-v10.alt-04 | 32 | 96.8750 | 94.4792 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-04 linear / neural % | RPB-v10.alt-04 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 64161 / 69666 | 100.0000 / 94.5312 | 96.8750 / 93.7500 | -3.1250 | 100.0000 / 100.0000 |
| 65262 / 70767 | 96.0938 / 89.5833 | 100.0000 / 97.9167 | +3.9062 | 100.0000 / 100.0000 |
| 66363 / 71868 | 96.0938 / 91.4062 | 98.4375 / 94.7917 | +2.3438 | 100.0000 / 100.0000 |
| 67464 / 72969 | 96.8750 / 93.4896 | 98.4375 / 95.3125 | +1.5625 | 100.0000 / 100.0000 |
| 68565 / 74070 | 99.2188 / 95.3125 | 90.6250 / 90.6250 | -8.5938 | 100.0000 / 100.0000 |

These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.

## Training and reconstruction

Fixed ordinary whole-patch query standardized MAE uses the same original Q, targets, support and TRAIN scaler in each pair. Inference adds no training context deletion. Sampled optimization Huber and its complete 512-step traces are separate evidence; amplitude reconstruction was not measured.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-04 | 512 | 0.077806255 | 0.080484228 | 15.832232 |
| RPB-v10.alt-04 | 512 | 0.073898432 | 0.075583393 | 16.602317 |

| Timing master | RPB-v7.alt-04 TRAIN / VALIDATION MAE | RPB-v10.alt-04 TRAIN / VALIDATION MAE | GPU update seconds late / early |
| --- | ---: | ---: | ---: |
| 64161 | 0.070009859 / 0.071833299 | 0.069404525 / 0.070303313 | 14.349920 / 15.736125 |
| 65262 | 0.086692801 / 0.090181257 | 0.077293014 / 0.082955453 | 15.467193 / 16.864874 |
| 66363 | 0.074258780 / 0.076719574 | 0.072489777 / 0.072060982 | 15.664069 / 16.196346 |
| 67464 | 0.081503926 / 0.083350294 | 0.071830598 / 0.073708584 | 19.185895 / 18.762474 |
| 68565 | 0.076565907 / 0.080336716 | 0.078474244 / 0.078888632 | 14.494083 / 15.451766 |

Synchronized cumulative update-loop times exclude head fitting, extraction, reconstruction queries and checkpoint writing. They are descriptive single-run measurements. Additional stages below include CPU transfer, parent verification, file hashing and I/O; pure CUDA kernel time is unmeasured.

| Mixed wall-time scope | Total seconds | Mean per paired cohort seconds |
| --- | ---: | ---: |
| `generation_and_observation_io_seconds` | 0.572576 | 0.114515 |
| `binding_and_checkpoint_io_seconds` | 124.413438 | 24.882688 |
| `CUDA_query_transfer_verification_io_seconds` | 141.424453 | 28.284891 |
| `CUDA_native_transfer_verification_seconds` | 219.072979 | 43.814596 |
| `CPU_baseline_preparation_io_seconds` | 0.931199 | 0.186240 |
| `CPU_head_bootstrap_io_seconds` | 50.669822 | 10.133964 |

## Descriptive guard checks

| Check | Passed |
| --- | --- |
| validation_intact_ridge_mean_no_worse | true |
| validation_intact_ridge_worst_no_worse | false |
| validation_intact_equal_per_master_coverage | true |
| validation_deleted_ridge_mean_no_worse | true |
| validation_deleted_ridge_worst_no_worse | false |
| validation_deleted_equal_per_master_coverage | true |
| training_original_query_mae_mean_no_worse | true |
| validation_original_query_mae_mean_no_worse | true |

These guard flags describe timing mean/worst linear scores, equal per-master coverage and original-query mean MAE. They do not authorize automatic promotion or rescue by secondary amplitude/neural scores.

## Evidence and limits

Actual supported budgets: 210 pipelines / 420 heads (planned 210/420). The five paired cohorts retain 20 point-0/512 checkpoints, both initial controls, all ten complete traces and all original-query supports. 120 full native export callbacks and 20 query-writer calls produced 80 patch-query banks. Joint reconstruction updates still train the decoder; no extra decoder-only calibration occurred.

Independent audit PASS: 96,153,106 checks, 1,255 CPU archive decodes, 780.004451 seconds. It independently replays saved maps/logits/support/query reductions and source effects; it runs no model, CUDA forward, autodiff, optimizer, head fit or PCA/SVD fit. Ordinary CUDA checkpoint bodies remain byte-bound, with semantics established by source/engineering admission and CPU companions. Original sampler/Torch streams are source/admission-bound, not GPU reenacted.

No numeric automatic promotion rule was declared. Assess timing intact/deleted linear accuracy, worst-cohort accuracy, fixed TRAIN/VALIDATION MAE, neural regressions and amplitude separately before any next change. This report does not replace original RPB-v7 weights or claim a causal internal mechanism.

[Frozen prospective card](../../evaluation/cards/early_mixer_confirmation_v1.md). [Durable full summary](../../../doc/results/early_mixer_confirmation_v1.json).

Metadata/source bindings:

- `admission_log_sha256`: `55bf81f6099165fccefc21f9f6ddbe28b16f9497f2c55c7c238835f864225efe`
- `admission_sha256`: `b1ebfcd04d53f08e3cbad74429638582586279fbb530696875ed367632c55723`
- `audit_reader`: `/embedding/output/runs/rpb-early-mixer-confirmation/audit-tools/independent-confirmation-cached-source-adjunct-released-uJ1Ack-v4/validate_early_mixer_confirmation.py`
- `audit_sha256`: `63826de216aff8a09b26b203d10fb9020fc08208276eec27347c9570e839957e`
- `audit_validation`: `/embedding/output/runs/rpb-early-mixer-confirmation/audit-tools/run-uJ1Ack-v4/validation.json`
- `capsule`: `/embedding/output/runs/rpb-early-mixer-confirmation/early-mixer-confirmation-uJ1Ack`
- `card_sha256`: `98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea`
- `inventory_sha256`: `d7f6db6dccad050eec984196d28e99b2cc1091c537edf40049ce2be117454846`
- `reader_sha256`: `979601d1c2acb2a7886b0d0f39070bf8738dd3f5788ce57e3dbdc6ac9e4a3c66`
- `renderer_sha256`: `d23f7ddfea1a25351490a403650296907596e3a482145e85f83f69a9df5112c9`
- `root_completed_metadata_record`: `/embedding/output/runs/rpb-early-mixer-confirmation/report-tools/released-renderer-uJ1Ack-v4/completed-binding.json`
- `root_completed_metadata_record_sha256`: `b3c86dd3f248d096fce477e3e3c0290f31e9f74bbfeacf15616ef8d3c404f7db`
- `source_fingerprint`: `fa1cbd0e67b4ce6226d6d3a0c5c5c39a71298764e64158dfdbc50decad7e1f12`
