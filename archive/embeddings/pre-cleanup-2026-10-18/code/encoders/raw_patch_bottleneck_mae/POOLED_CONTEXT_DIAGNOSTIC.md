# Pooled temporal context diagnostic

Completed fixed `pooled-context-v1` development comparison with an independent saved-arithmetic audit. All five fresh paired cohorts and both separate initial controls are retained. No TEST or stress was generated or read, no point was selected, and no model is promoted.

RPB-v10.alt-03 is the projected-D32 early-mixer control. RPB-v12 retains each channel's learned W64 temporal summary until the global native32 bottleneck. Both use the original 0.15 context deletion, original TRAIN scaler, reconstruction targets, optimizer, masks and heads. The wider route also increases the first global layer capacity, so this comparison does not isolate projection loss as a cause. Both independent and contextual W summaries use semantic channel order and support bits; the exact served BD32 remains the decoder's only observation signal.

The control registers 225,805 parameters; the candidate registers 231,949, including 6,144 additional first-layer values. Exactly 219,469 common same-name values and all buffers are copied from the saved zero-update control before AdamW. Only the wider first matrix retains its own initialization. Candidate diagnostic projection values number 2,080; they stay unchanged and have no AdamW moments. Its reconstruction graph reaches 229,869 registered values, 4,064 more than the compact control; this does not imply every scalar updates. Actual named optimizer activity remains in the JSON. CUDA inference still computes the declared diagnostic outputs. Initial native outputs are different functions and receive separate untrained controls.

Timing masters are 53151/54252/55353/56454/57555; separate amplitude-data masters are 58656/59757/60858/61959/63060. Each timing trajectory uses 256 TRAIN rows from 128 sources and 128 known VALIDATION rows from 64 disjoint sources. Exactly 512 joint encoder/decoder updates at batch 8 give 4,096 sampled row exposures per trajectory: 16 equivalent presentations, not guaranteed epochs. Ten trajectories give 5,120 updates and 40,960 exposures. There are zero extra decoder-only calibration updates; amplitude never fits an encoder or its scaler.

Raw data has 288 scaled values plus 288 masks. Mask only has 288 inputs. PCA only — no encoder — fits raw TRAIN and yields 32 coordinates; native 32 receives no PCA. Three fixed repetitions 2701/2802/2903 use Ridge penalty 1 and tanh 16/Adam 0.01 for 100 full-TRAIN updates. Width 32 has 66 linear/562 neural parameters; raw 576 has 1,154/9,266 and mask 288 has 578/4,658. Each TRAIN-fitted head is reused across both VALIDATION views.

## Timing (lag sign): intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 52.0312 | 52.6042 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 47.9688 | 75.3125 | 100.0000 |
| Untrained compact encoder | 32 | 55.4688 | 76.9271 | 100.0000 |
| Untrained pooled encoder | 32 | 55.9375 | 76.5104 | 100.0000 |
| RPB-v10.alt-03 | 32 | 98.2812 | 99.9479 | 100.0000 |
| RPB-v12 | 32 | 90.9375 | 96.5104 | 100.0000 |

Equal-master means average head repetitions within each master first. Unsupported cohorts remain present and make the corresponding aggregate undefined. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-03 linear / neural % | RPB-v12 linear / neural % | Paired linear effect pp | Coverage compact / pooled % |
| --- | ---: | ---: | ---: | ---: |
| 53151 / 53151 | 98.4375 / 100.0000 | 97.6562 / 100.0000 | -0.7812 | 100.0000 / 100.0000 |
| 54252 / 54252 | 93.7500 / 99.7396 | 73.4375 / 86.1979 | -20.3125 | 100.0000 / 100.0000 |
| 55353 / 55353 | 100.0000 / 100.0000 | 99.2188 / 100.0000 | -0.7812 | 100.0000 / 100.0000 |
| 56454 / 56454 | 99.2188 / 100.0000 | 85.1562 / 96.3542 | -14.0625 | 100.0000 / 100.0000 |
| 57555 / 57555 | 100.0000 / 100.0000 | 99.2188 / 100.0000 | -0.7812 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-population paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Timing (lag sign): extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 48.4375 | 49.3750 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.5938 | 51.5625 | 100.0000 |
| Untrained compact encoder | 32 | 49.2188 | 54.8438 | 100.0000 |
| Untrained pooled encoder | 32 | 51.4062 | 58.4375 | 100.0000 |
| RPB-v10.alt-03 | 32 | 97.0312 | 99.3750 | 100.0000 |
| RPB-v12 | 32 | 83.1250 | 89.4271 | 100.0000 |

Equal-master means average head repetitions within each master first. Unsupported cohorts remain present and make the corresponding aggregate undefined. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-03 linear / neural % | RPB-v12 linear / neural % | Paired linear effect pp | Coverage compact / pooled % |
| --- | ---: | ---: | ---: | ---: |
| 53151 / 53151 | 96.8750 / 100.0000 | 92.9688 / 100.0000 | -3.9062 | 100.0000 / 100.0000 |
| 54252 / 54252 | 89.0625 / 96.8750 | 53.1250 / 55.2083 | -35.9375 | 100.0000 / 100.0000 |
| 55353 / 55353 | 100.0000 / 100.0000 | 96.8750 / 100.0000 | -3.1250 | 100.0000 / 100.0000 |
| 56454 / 56454 | 99.2188 / 100.0000 | 84.3750 / 91.9271 | -14.8438 | 100.0000 / 100.0000 |
| 57555 / 57555 | 100.0000 / 100.0000 | 88.2812 / 100.0000 | -11.7188 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-population paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Frozen amplitude transfer: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 60.1562 | 60.9896 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 54.3750 | 65.2083 | 100.0000 |
| Untrained compact encoder | 32 | 98.4375 | 98.6458 | 100.0000 |
| Untrained pooled encoder | 32 | 99.0625 | 98.6458 | 100.0000 |
| RPB-v10.alt-03 | 32 | 98.5938 | 98.6458 | 100.0000 |
| RPB-v12 | 32 | 100.0000 | 99.9479 | 100.0000 |

Equal-master means average head repetitions within each master first. Unsupported cohorts remain present and make the corresponding aggregate undefined. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-03 linear / neural % | RPB-v12 linear / neural % | Paired linear effect pp | Coverage compact / pooled % |
| --- | ---: | ---: | ---: | ---: |
| 53151 / 58656 | 100.0000 / 100.0000 | 100.0000 / 99.7396 | +0.0000 | 100.0000 / 100.0000 |
| 54252 / 59757 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 55353 / 60858 | 92.9688 / 95.0521 | 100.0000 / 100.0000 | +7.0312 | 100.0000 / 100.0000 |
| 56454 / 61959 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 57555 / 63060 | 100.0000 / 98.1771 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-population paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Frozen amplitude transfer: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 61.8750 | 46.2500 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 51.4062 | 55.5208 | 100.0000 |
| Untrained compact encoder | 32 | 90.3125 | 86.3021 | 100.0000 |
| Untrained pooled encoder | 32 | 87.5000 | 87.2917 | 100.0000 |
| RPB-v10.alt-03 | 32 | 92.1875 | 90.0521 | 100.0000 |
| RPB-v12 | 32 | 98.7500 | 97.8646 | 100.0000 |

Equal-master means average head repetitions within each master first. Unsupported cohorts remain present and make the corresponding aggregate undefined. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-03 linear / neural % | RPB-v12 linear / neural % | Paired linear effect pp | Coverage compact / pooled % |
| --- | ---: | ---: | ---: | ---: |
| 53151 / 58656 | 99.2188 / 97.1354 | 96.0938 / 95.3125 | -3.1250 | 100.0000 / 100.0000 |
| 54252 / 59757 | 97.6562 / 95.5729 | 100.0000 / 100.0000 | +2.3438 | 100.0000 / 100.0000 |
| 55353 / 60858 | 84.3750 / 83.5938 | 97.6562 / 96.0938 | +13.2812 | 100.0000 / 100.0000 |
| 56454 / 61959 | 86.7188 / 86.1979 | 100.0000 / 98.4375 | +13.2812 | 100.0000 / 100.0000 |
| 57555 / 63060 | 92.9688 / 87.7604 | 100.0000 / 99.4792 | +7.0312 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-population paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Training and reconstruction

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v10.alt-03 | 512 | 0.079615491 | 0.080886286 | 13.827282377 |
| RPB-v12 | 512 | 0.075831372 | 0.078075544 | 13.886893782 |

Errors are fixed original-query standardized MAE in common original-scaler units. Full sampled Huber objective/gradient/target-count traces remain separate. Synchronized CUDA-loop wall time includes CPU trace/live-state capture and is descriptive, not pure kernel cost. Checkpoint/continuation writing, query/native transfer and parent hashing, baseline/head/bootstrap and audit stages are separate.

| Timing master | RPB-v10.alt-03 TRAIN / VALIDATION MAE | RPB-v12 TRAIN / VALIDATION MAE | Loop seconds compact / pooled |
| --- | ---: | ---: | ---: |
| 53151 | 0.086682538 / 0.093483956 | 0.083193831 / 0.088955453 | 13.919096448 / 13.805070677 |
| 54252 | 0.074286469 / 0.075684636 | 0.071608434 / 0.071972067 | 13.891262319 / 13.880047854 |
| 55353 | 0.074655751 / 0.073817043 | 0.087045546 / 0.089279980 | 13.708193016 / 13.846147144 |
| 56454 | 0.087256078 / 0.086673276 | 0.068714616 / 0.070446372 | 13.766081793 / 13.929576187 |
| 57555 | 0.075196618 / 0.074772515 | 0.068594436 / 0.069723848 | 13.851778307 / 13.973627050 |

## Frozen joint direction

| Guard | Passed |
| --- | --- |
| `validation_intact_ridge_mean_no_worse` | false |
| `validation_intact_ridge_worst_no_worse` | false |
| `validation_intact_equal_per_master_coverage` | true |
| `validation_deleted_ridge_mean_no_worse` | false |
| `validation_deleted_ridge_worst_no_worse` | false |
| `validation_deleted_equal_per_master_coverage` | true |
| `training_original_query_mae_mean_no_worse` | true |
| `validation_original_query_mae_mean_no_worse` | true |

Joint direction passed: **false**. Six numeric guards cover timing linear mean/worst scores in both views and mean original TRAIN/VALIDATION MAE; equal per-master coverage is checked separately. A failure stops this route without width/rate/budget/head rescue. A pass requires separately frozen fresh confirmation before promotion. All secondary neural/amplitude tradeoffs remain reported; strong initial amplitude controls limit learning-credit claims.

| Mixed wall-time scope | Total seconds | Mean paired-cohort seconds |
| --- | ---: | ---: |
| `generation_and_observation_io_seconds` | 0.640556 | 0.128111 |
| `binding_and_checkpoint_io_seconds` | 72.457214 | 14.491443 |
| `CUDA_query_transfer_verification_io_seconds` | 95.079657 | 19.015931 |
| `CUDA_unique_quality_native_transfer_verification_seconds` | 106.916858 | 21.383372 |
| `CPU_baseline_preparation_io_seconds` | 0.909071 | 0.181814 |
| `CPU_head_bootstrap_io_seconds` | 62.530913 | 12.506183 |

## Evidence and limits

Independent audit PASS: 96,170,207 checks/1,235 CPU archive decodes/402.684740 seconds. Supported pipelines/heads: 210/420 (planned 210/420). Actual full native callbacks: 120 distinct quality exports, with no counterpart reuse. Original timing queries use 20 writers/80 masked forwards. All 20 checkpoint points and five-role companions are retained.

The CPU audit checks saved common/nonshared initialization, unchanged diagnostic projection and absent moments, original scaler/data/support, fixed query reductions, maps/logits/argmax and source intervals. It performs no CUDA/model/autodiff/optimizer/head fit or PCA/SVD fit. CUDA checkpoint bodies are byte-bound; actual named live CPU state/AdamW/scaler witnesses and captured CUDA admission establish association without independently reenacting the optimizer trajectory. No causal bottleneck claim or broad generalization claim follows from these synthetic tasks. Pure GPU kernel time is unmeasured. Historical quality payloads, TEST and stress are absent. Original RPB-v7 artifacts and the active v4 reference remain unchanged.

[Frozen prospective card](../../evaluation/cards/pooled_context_v1.md). [Full durable summary](../../../doc/results/pooled_context_v1.json).

Metadata/source bindings:

- `admission_log_sha256`: `51bb61cdf37678940bda75366de976f50642896e7b08db1019ea15ec61cee794`
- `admission_sha256`: `7ff8469d677c5f7d3fa7f51338b9322ddd5c18786057f7c71c4369b65fae0c8d`
- `audit_reader`: `/embedding/output/runs/rpb-pooled-context/audit-tools/independent-pooled-context-released-Na1nEQ-v2/validate_pooled_context.py`
- `audit_sha256`: `6820cd40ec4ff79a715ff7521a4ff1315c78908ae81c8d2fa8e27395b7f23964`
- `audit_validation`: `/embedding/output/runs/rpb-pooled-context/audit-tools/run-Na1nEQ-v2/validation.json`
- `capsule`: `/embedding/output/runs/rpb-pooled-context/pooled-context-Na1nEQ`
- `card_sha256`: `b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7`
- `inventory_sha256`: `66a86efd911b5c98485b8540120c348105fb7be6853e9180b9733c53e4b3d700`
- `reader_sha256`: `61616c501391ae118ea78e628849e60f661f1b3434795743560ade1bb2a9e63a`
- `renderer_sha256`: `96f501d4771eeea21f0fb5b70c732c861b4f1946c08bd41f1cb3d94fd3a8ca54`
- `root_completed_metadata_record`: `/embedding/output/runs/rpb-pooled-context/report-tools/released-renderer-Na1nEQ-v1/completed-binding.json`
- `root_completed_metadata_record_sha256`: `195c2a836b1ddb2a6b143f6344e377217e72da522a52472d17446b141e7aa3a7`
- `source_fingerprint`: `be129553c8712415d3f009dce8b2131c88123ff2e39f3822baf3bb55e0310059`
