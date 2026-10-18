# Fresh decoder replication diagnostic

New synthetic development replication under `fresh-decoder-replication-v1`. Five fresh masters (9109, 10210, 11311, 12412, 13513) each provide 256 TRAIN examples from 128 source groups and 128 VALIDATION examples from 64 disjoint source groups. This is ten encoder trajectories and ten decoder calibration trajectories, rather than repeated classifier fits being counted as encoder runs.

Each RPB-v4/RPB-v7 pair trained once for 512 unskipped updates on the NVIDIA RTX A2000 8 GB, batch 8. Both have 225,805 parameters and serve the exact native 32-number global export. Fresh decoder calibration then trained the six decoder tensors (11,528 parameters) for 128 updates with 214,277 non-decoder parameters, buffers and the TRAIN scaler frozen. Every same-CUDA native feature/support witness remained exact.

Geometry is C3/H32/F3 with 10% natural missingness and CPU float64 observations: patch 8, encoder width 64, three temporal blocks, four attention heads, feedforward width 256, one channel mixer, global mode 2, decoder width 128 and dropout 0. AdamW uses learning rate 0.001, weight decay 0.0001 and gradient clipping 1; the original query mask ratio is 0.25 and Huber delta is 1. One Torch CPU thread is used. Decoder calibration keeps this optimizer recipe but starts fresh decoder-only AdamW, uses absolute attempts 512–639 and applies no context deletion.

Intact/deletion mean linear effects are +0.62500/+7.81250 percentage points. The classification guard is FAIL; reconstruction recovery against v4 before decoder calibration is PASS; the equal-decoder reconstruction guard is FAIL. The joint development guard is FAIL. No automatic promotion occurred.

## Native timing quality

RPB-v4 is the ordinary learned global bottleneck. RPB-v7 has the same inference architecture with 15% training context deletion. The shared untrained native control is their paired point-0 initialization. Raw includes 288 TRAIN-scaled legal values plus 288 visibility flags; mask metadata contains only the 288 flags. PCA is fitted to raw TRAIN features and never follows an encoder.

Intact VALIDATION. All figures are means of three fixed head repetitions within each master, then of the five masters with equal weight.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder | 576 | 51.09375 | 51.61458 | 100 |
| PCA only — no encoder | 32 | 49.375 | 74.53125 | 100 |
| Mask metadata | 288 | 50 | 50 | 100 |
| Untrained native control | 32 | 48.75 | 60.67708 | 100 |
| RPB-v4 | 32 | 92.34375 | 98.125 | 100 |
| RPB-v7 | 32 | 92.96875 | 95.26042 | 100 |

Additional 30% coordinate deletion on the same VALIDATION sources. One named, pair-shared mask is reused by every method; there is no repair or held-out fit.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| Raw data — no encoder | 576 | 51.875 | 49.47917 | 100 |
| PCA only — no encoder | 32 | 49.21875 | 55.52083 | 100 |
| Mask metadata | 288 | 50 | 50 | 100 |
| Untrained native control | 32 | 50.78125 | 50.46875 | 100 |
| RPB-v4 | 32 | 82.03125 | 90.41667 | 100 |
| RPB-v7 | 32 | 89.84375 | 92.44792 | 100 |

Linear heads use ridge penalty 1. Neural heads use 16 tanh hidden units, Adam at 0.01 and 100 updates. All three declared seeds 2701/2802/2903 are retained, with `stream_seed(repetition, feature_width)`. Size-32 heads have 66/562 fitted parameters; raw576 heads have 1,154/9,266, and mask288 heads have 578/4,658. Raw/native/mask/untrained use an outer TRAIN normalizer followed by each probe's TRAIN normalizer. PCA-only uses raw TRAIN normalization → PCA32 → the probe's TRAIN normalizer.

There are 90 fitted pipelines and 180 individual heads. Controls and raw PCA are prepared once per cohort. Each head is fitted once on TRAIN and scores both VALIDATION views. Classification at decoder 128 reuses the decoder-0 scores and assets only after exact same-CUDA feature/support witnesses; it has zero additional head fits and zero additional score repetitions.

## Reconstruction and cost

Errors are original fixed-query standardized MAE, with cell → channel → example reduction and equal example weight. All four original patch queries are retained; these are not final minibatch losses. Encoder update counts are successful optimizer updates. At batch 8, 512 updates give 4,096 sampled row presentations (16 TRAIN-size equivalents), and decoder 128 adds 1,024 presentations (four equivalents).

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | --- | --- | --- |
| RPB-v4 | 512 encoder | 0.069948 | 0.072527 | 12.60 encoder |
| RPB-v4 | 512 encoder + 128 decoder | 0.053609 | 0.056835 | 12.60 encoder + 2.08 decoder loop† |
| RPB-v7 | 512 encoder | 0.080413 | 0.082911 | 13.40 encoder |
| RPB-v7 | 512 encoder + 128 decoder | 0.056375 | 0.059200 | 13.40 encoder + 1.99 decoder loop† |

† Encoder seconds are synchronized encoder training-loop time. Decoder-loop seconds include CPU evidence capture and are not a pure GPU-kernel benchmark; they are shown separately. The encoder cost repeats in the post-decoder row as provenance of that same trajectory, rather than another encoder run. CUDA extraction/query evidence and CPU head fitting/prediction/bootstrap/I/O are separate, descriptive timing scopes in the machine record.

## Every master and uncertainty

| Master | Intact v4/v7 linear % | Deletion v4/v7 linear % | Intact/deletion v7−v4 pp | v4 post TRAIN/VAL MAE | v7 post TRAIN/VAL MAE |
| --- | --- | --- | --- | --- | --- |
| 9109 | 89.84375/91.40625 | 85.9375/86.71875 | +1.56250/+0.78125 | 0.056490/0.060831 | 0.057453/0.061704 |
| 10210 | 96.875/97.65625 | 82.8125/89.84375 | +0.78125/+7.03125 | 0.052827/0.053928 | 0.056891/0.057106 |
| 11311 | 89.0625/95.3125 | 84.375/94.53125 | +6.25000/+10.15625 | 0.050196/0.054456 | 0.054654/0.059038 |
| 12412 | 93.75/80.46875 | 81.25/81.25 | -13.28125/+0.00000 | 0.054894/0.058286 | 0.055156/0.057592 |
| 13513 | 92.1875/100 | 75.78125/96.875 | +7.81250/+21.09375 | 0.053639/0.056676 | 0.057721/0.060560 |

Descriptive paired five-master 95% intervals for v7−v4 are [-6.40625, 6.09375] pp intact and [1.87500, 14.84375] pp under deletion for linear heads; neural intervals are [-8.59375, 1.40625] pp and [-4.79167, 8.28125] pp. These use the fixed 10,000-draw whole-master bootstrap with each master's three head repetitions retained together. A crossing-zero interval leaves the direction imprecise even if a point guard passes. Within-master source-group intervals remain separate in the linked record; no bounds are averaged.

## Predeclared guards

| Predicate | Result |
| --- | --- |
| validation intact positive mean linear | PASS |
| validation intact worst master not lower | FAIL |
| validation intact equal full coverage | PASS |
| validation deleted positive mean linear | PASS |
| validation deleted worst master not lower | PASS |
| validation deleted equal full coverage | PASS |
| train recovery vs v4 pre | PASS |
| train equal decoder vs v4 post | FAIL |
| validation recovery vs v4 pre | PASS |
| validation equal decoder vs v4 post | FAIL |
| joint development guard | FAIL |

Recovery asks whether post-calibration v7 reaches ordinary v4 before calibration. The fair reconstruction comparison gives both v4 and v7 the same extra decoder budget. The two questions are reported separately; recovery cannot be described as equal-cost superiority. Neural results cannot rescue failed linear or reconstruction guards.

## Evidence and scope

Independent saved-arithmetic audit passed 67,932,331 checks and decoded 670 CPU archives. It verified the complete fresh role/source/card/admission matrix, saved maps/logits/classes, grouped bootstrap, decoder update arithmetic, fixed queries and frozen state/native witnesses. It does not rerun CUDA models, AdamW or head optimization; those depend on the hash-bound producer/admission and recorded CUDA/state evidence. Original row choices from `std::shuffle` are structurally checked and source-bound. This is synthetic TRAIN/VALIDATION development evidence, with zero historical quality inputs, TEST/stress access, selection or automatic promotion.

- [Frozen card](../../evaluation/cards/fresh_decoder_replication_v1.md): `2236fb794d608c8f7b8bde81814f199abcc3e27c641cfba5902cdc3f4c4a25cf`.
- [Measured report](../../../output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b/results/report.json): `0352202e27d692adad8a5041bff259116a9dd6aef7911edf685c624a3d6db2d9`.
- [Capsule inventory](../../../output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b/artifact-integrity.json): `a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90`; 948 files / 432,090,961 bytes, excluding the inventory itself.
- [Captured source](../../../output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b/source-manifest.json): `6463926ae71ee5e5547aa660d654afb02832fa843c978c41609cbd814266d78c`; 77 compiled source entries.
- [Actual CUDA admission](../../../output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b/admission/passed.json): `eda1ea8f2d041f227634f120a0007fad629a4950f1c19c16bff412d2900fd7f4`; log `5288ed9ea645389163d179c6efb4fd40e440393e50c4c45f1a95f7cc255ae9ce`.
- [Passed independent audit](../../../output/runs/rpb-fresh-decoder-replication/audit-tools/run-4p9U4b-v2/validation.json): `241efd4705e4f6d67edf7070fa650f41e5a2a1b2d6e1e371e3d2835391c567e4`; [sealed reader](../../../output/runs/rpb-fresh-decoder-replication/audit-tools/independent-fresh-decoder-replication-20261008-v2/validate_fresh_decoder_replication.py) `ab9d2a241ee746389cd1c4b32de8db1e9e9fca2943ec3ecd6d26e16d4148ca88`.
- [Complete machine summary](../../../doc/results/fresh_decoder_replication_v1.json), including every master, fixed head/source intervals, separate timing scopes and all guards.

The first admission `admission-qrEtH2` FAILED and is preserved. Compilation, historical CUDA and shared CPU readout checks passed; fresh numerical lifecycle assertions reached the final broad RNG witness, which also enclosed unchanged historical fixture preparation whose trainer intentionally seeds RNG and has no ambient-isolation contract. The subsequent test-only repair isolates only those fixture calls and strengthens the scoped witnesses for all unwrapped fresh API lifecycle callbacks. New admission `admission-evzwzh` passed. No quality data was generated during the failed admission, and no production/backend/card/training recipe changed.

RPB-v4 remains the active reference and RPB-v7 remains the working candidate, without promotion or a new tag. Stop decoder/rate/budget/head tuning. The next focused action is a separately frozen saved-TRAIN reliability diagnosis of timing features, fixed-head margins, paired separation and training traces across all ten v4/v7 trajectories, with zero model updates or head refits. Distinguish weak TRAIN separation from a generalization gap rather than assigning a cause from one weak master. Do not exclude an outlier or open another task before that diagnosis.
