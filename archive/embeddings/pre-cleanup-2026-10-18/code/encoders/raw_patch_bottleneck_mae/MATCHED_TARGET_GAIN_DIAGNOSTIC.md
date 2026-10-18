# Matched-target TRAIN magnitude-support diagnostic

Completed fixed `matched-target-gain-v1` development comparison with independent saved-arithmetic audit. No TEST or stress was generated or read; no point was selected and no model is promoted. All five fresh paired cohorts are retained.

RPB-v10.alt-02 is the fresh early-mixer control with ordinary coordinate-0.15 context deletion. RPB-v11 keeps the exact same 225,805-parameter/native-32 inference architecture, original TRAIN scaler, optimizer, masks and heads. It adds one static source-paired legal float64 TRAIN gain before original centering/scaling, with gained input and reconstruction target matched. Serving adds no gain or context deletion. These are new trained instances; all original v7 artifacts remain preserved.

Timing TRAIN/VALIDATION masters are 41140/42241/43342/44443/45544; separate amplitude-data masters are 46645/47746/48847/49948/51049. Each timing controller receives 256 TRAIN rows from 128 sources and 128 known VALIDATION rows from 64 disjoint sources. Exactly 512 ordinary joint encoder/decoder updates at batch 8 give 4,096 sampled exposures per trajectory: 16 equivalent presentations, not guaranteed epochs. There are ten trajectories, 5,120 updates and 40,960 exposures and zero extra decoder-only calibration updates. Amplitude never fits an encoder or its scaler.

Raw data contains 288 scaled values plus 288 masks. Mask only has 288 inputs. PCA only — no encoder — is fitted from raw TRAIN and has 32 coordinates; native 32 receives no PCA. The one untrained early control is justified by 30 additional actual candidate-initial CUDA counterpart exports that match the shared initial values/support exactly on all six timing/amplitude surfaces before heads. Three fixed repetitions 2701/2802/2903 use Ridge penalty 1 and tanh 16/Adam 0.01 for 100 full-TRAIN updates. Width 32 has 66 linear/562 neural parameters; raw 576 has1,154/9,266 and mask 288 has578/4,658. Each fixed TRAIN head is reused across both VALIDATION views.

## Timing (lag sign): intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 50.7812 | 52.2396 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 44.0625 | 77.0312 | 100.0000 |
| Untrained encoder | 32 | 54.8438 | 67.8646 | 100.0000 |
| RPB-v10.alt-02 | 32 | 96.4062 | 99.8958 | 100.0000 |
| RPB-v11 | 32 | 95.3125 | 97.5000 | 100.0000 |

Equal-master means; head repetitions are averaged within master first. Unsupported cohorts remain present and make an aggregate unsupported. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-02 linear / neural % | RPB-v11 linear / neural % | Paired linear effect pp | Coverage control / gain % |
| --- | ---: | ---: | ---: | ---: |
| 41140 / 41140 | 90.6250 / 100.0000 | 100.0000 / 100.0000 | +9.3750 | 100.0000 / 100.0000 |
| 42241 / 42241 | 98.4375 / 100.0000 | 84.3750 / 91.6667 | -14.0625 | 100.0000 / 100.0000 |
| 43342 / 43342 | 100.0000 / 99.7396 | 96.8750 / 98.6979 | -3.1250 | 100.0000 / 100.0000 |
| 44443 / 44443 | 92.9688 / 99.7396 | 95.3125 / 97.1354 | +2.3438 | 100.0000 / 100.0000 |
| 45544 / 45544 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-pop paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Timing (lag sign): extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 50.6250 | 48.9062 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 43.9062 | 54.8958 | 100.0000 |
| Untrained encoder | 32 | 50.4687 | 53.5937 | 100.0000 |
| RPB-v10.alt-02 | 32 | 92.5000 | 98.0208 | 100.0000 |
| RPB-v11 | 32 | 84.6875 | 90.4167 | 100.0000 |

Equal-master means; head repetitions are averaged within master first. Unsupported cohorts remain present and make an aggregate unsupported. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-02 linear / neural % | RPB-v11 linear / neural % | Paired linear effect pp | Coverage control / gain % |
| --- | ---: | ---: | ---: | ---: |
| 41140 / 41140 | 83.5938 / 94.2708 | 74.2188 / 99.2188 | -9.3750 | 100.0000 / 100.0000 |
| 42241 / 42241 | 93.7500 / 98.6979 | 73.4375 / 75.7812 | -20.3125 | 100.0000 / 100.0000 |
| 43342 / 43342 | 96.8750 / 97.9167 | 89.0625 / 89.5833 | -7.8125 | 100.0000 / 100.0000 |
| 44443 / 44443 | 88.2812 / 99.2188 | 90.6250 / 89.8438 | +2.3438 | 100.0000 / 100.0000 |
| 45544 / 45544 | 100.0000 / 100.0000 | 96.0938 / 97.6562 | -3.9062 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-pop paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Amplitude transfer: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 60.6250 | 61.9271 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 54.3750 | 66.8750 | 100.0000 |
| Untrained encoder | 32 | 98.7500 | 98.9583 | 100.0000 |
| RPB-v10.alt-02 | 32 | 99.8438 | 99.6875 | 100.0000 |
| RPB-v11 | 32 | 99.5312 | 99.7917 | 100.0000 |

Equal-master means; head repetitions are averaged within master first. Unsupported cohorts remain present and make an aggregate unsupported. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-02 linear / neural % | RPB-v11 linear / neural % | Paired linear effect pp | Coverage control / gain % |
| --- | ---: | ---: | ---: | ---: |
| 41140 / 46645 | 100.0000 / 99.4792 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 42241 / 47746 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 43342 / 48847 | 100.0000 / 98.9583 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 44443 / 49948 | 99.2188 / 100.0000 | 97.6562 / 98.9583 | -1.5625 | 100.0000 / 100.0000 |
| 45544 / 51049 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-pop paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Amplitude transfer: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 576 | 63.9062 | 46.0938 | 100.0000 |
| Mask only | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 55.9375 | 56.6146 | 100.0000 |
| Untrained encoder | 32 | 82.9688 | 81.6667 | 100.0000 |
| RPB-v10.alt-02 | 32 | 97.5000 | 95.0521 | 100.0000 |
| RPB-v11 | 32 | 86.2500 | 93.4375 | 100.0000 |

Equal-master means; head repetitions are averaged within master first. Unsupported cohorts remain present and make an aggregate unsupported. Coverage is valid/declared rows; accuracy is conditional on that support.

| Timing master / data master | RPB-v10.alt-02 linear / neural % | RPB-v11 linear / neural % | Paired linear effect pp | Coverage control / gain % |
| --- | ---: | ---: | ---: | ---: |
| 41140 / 46645 | 99.2188 / 94.5312 | 99.2188 / 99.4792 | +0.0000 | 100.0000 / 100.0000 |
| 42241 / 47746 | 99.2188 / 98.9583 | 78.9062 / 90.1042 | -20.3125 | 100.0000 / 100.0000 |
| 43342 / 48847 | 94.5312 / 91.4062 | 88.2812 / 93.7500 | -6.2500 | 100.0000 / 100.0000 |
| 44443 / 49948 | 96.0938 / 94.2708 | 79.6875 / 91.1458 | -16.4062 | 100.0000 / 100.0000 |
| 45544 / 51049 | 98.4375 / 96.0938 | 85.1562 / 92.7083 | -13.2812 | 100.0000 / 100.0000 |

Full ranges, every marginal interval and all common-pop paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.

## Training and reconstruction

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v10.alt-02 | 512 | 0.075180955 | 0.076728894 | 14.177860099 |
| RPB-v11 | 512 | 0.216640799 | 0.214658641 | 14.335222582 |

Training errors are fixed original-query standardized MAE in common original-scaler units. Sampled Huber loss/gradient/target-count traces remain separate: gain changes target variance, so training loss is not a common quality score. Synchronized CUDA-loop wall time includes trace/sample-index capture and candidate-only actual-target CPU copies. It is descriptive, not pure kernel cost. Checkpoint/gain-state writing, query/native transfer and parent hashing, baseline/head/bootstrap and audit stages are separate.

| Timing master | RPB-v10.alt-02 TRAIN / VALIDATION MAE | RPB-v11 TRAIN / VALIDATION MAE | Loop seconds control / gain |
| --- | ---: | ---: | ---: |
| 41140 | 0.076623245 / 0.076463840 | 0.254186450 / 0.254603176 | 13.997977581 / 14.252936285 |
| 42241 | 0.065536250 / 0.068289240 | 0.167289785 / 0.168511609 | 14.052916118 / 14.473876112 |
| 43342 | 0.082801979 / 0.083179851 | 0.214926303 / 0.212914258 | 14.655154923 / 14.184408250 |
| 44443 | 0.071373074 / 0.072060307 | 0.217008800 / 0.216158286 | 14.531806073 / 14.436315399 |
| 45544 | 0.079570228 / 0.083651232 | 0.229792658 / 0.221105874 | 13.651445801 / 14.328576866 |

## Frozen joint direction

| Guard | Passed |
| --- | --- |
| `validation_intact_ridge_mean_no_worse` | false |
| `validation_intact_ridge_worst_no_worse` | false |
| `validation_intact_equal_per_master_coverage` | true |
| `validation_deleted_ridge_mean_no_worse` | false |
| `validation_deleted_ridge_worst_no_worse` | false |
| `validation_deleted_equal_per_master_coverage` | true |
| `training_original_query_mae_mean_no_worse` | false |
| `validation_original_query_mae_mean_no_worse` | false |

Joint direction passed: **false**. The six numeric guards cover timing linear mean/worst score in both views and mean original TRAIN/VALIDATION MAE; coverage equality is checked separately. A failed joint direction stops this recipe without magnitude/rate/budget/head rescue. Even a pass requires separately frozen confirmation before promotion. Neural and amplitude tradeoffs are secondary and remain reported; strong untrained amplitude controls limit learning-credit claims.

| Mixed wall-time scope | Total seconds | Mean paired-cohort seconds |
| --- | ---: | ---: |
| `generation_and_observation_io_seconds` | 0.646353 | 0.129271 |
| `binding_and_checkpoint_io_seconds` | 71.747151 | 14.349430 |
| `CUDA_query_transfer_verification_io_seconds` | 99.320079 | 19.864016 |
| `CUDA_unique_quality_native_transfer_verification_seconds` | 83.331823 | 16.666365 |
| `CUDA_initial_counterpart_transfer_verification_io_seconds` | 22.520371 | 4.504074 |
| `CPU_baseline_preparation_io_seconds` | 0.931026 | 0.186205 |
| `CPU_head_bootstrap_io_seconds` | 51.281916 | 10.256383 |

## Evidence and limits

Independent audit PASS: 88,154,488 checks/1,165 CPU archive decodes/358.420464 seconds. Supported pipelines/heads: 180/360 (planned 180/360). Actual full native callbacks: 120 = 90 unique quality exports plus 30 initial counterparts; original timing queries use 20 writers/80 masked forwards. All 20 checkpoints and six-file companions are retained.

The CPU audit replays source gain ranks/uniforms/gains, all actual candidate normalized targets, maps/logits/support/query reductions and source effects. It performs no CUDA/model/autodiff/optimizer/head fit or PCA/SVD fit. CUDA checkpoint bodies are byte-bound; live CPU model/AdamW/scaler state and captured actual CUDA admission establish their interpretation. It does not reenact the optimizer trajectory or prove a causal architecture bottleneck. No historical quality payloads, TEST or stress enter this comparison.

[Frozen prospective card](../../evaluation/cards/matched_target_gain_v1.md). [Full durable summary](../../../doc/results/matched_target_gain_v1.json).

Metadata/source bindings:

- `admission_log_sha256`: `71604f7cbe11c05b761873163b72e88358dc9fe7aeb4d7b47550abed10f04cf6`
- `admission_sha256`: `a8eae291a3d2854a6ac2f555a48d92a0263ee58fcc841f26c5af6c6d20d39f4b`
- `audit_reader`: `/embedding/output/runs/rpb-matched-target-gain/audit-tools/independent-matched-target-gain-released-YmAtKW-v2/validate_matched_target_gain.py`
- `audit_sha256`: `9807e8439c60c3beefeb83431cd7660667a10aae449fbdc4b281c98b1b631682`
- `audit_validation`: `/embedding/output/runs/rpb-matched-target-gain/audit-tools/run-YmAtKW-v2/validation.json`
- `capsule`: `/embedding/output/runs/rpb-matched-target-gain/matched-target-gain-YmAtKW`
- `card_sha256`: `00d5d84815fd64a3e1787e3f25dddef045b1ca208458c19d5ca4000a5ca04eec`
- `inventory_sha256`: `e653ab95509af9eb321a0ff90e50f16f4566051894a3e4df4d162f55dda0f4f4`
- `reader_sha256`: `f9c2851fb145e20f068f32ca555cb8573a60a817ea5e8b04557f51dbf6bc90e6`
- `renderer_sha256`: `508c6fb26b94787681c21b8db11eb9781a706e368d1eaad7b769219e693a15ef`
- `root_completed_metadata_record`: `/embedding/output/runs/rpb-matched-target-gain/report-tools/released-renderer-YmAtKW-v1/completed-binding.json`
- `root_completed_metadata_record_sha256`: `2a0d6257e9038a3f60f13402f35f2a3e1c8f460595adb18875d2a92a56580cee`
- `source_fingerprint`: `1b734e72069f46aef80ecf9270edfa186aa50557b715ae3f101380b80f519215`
