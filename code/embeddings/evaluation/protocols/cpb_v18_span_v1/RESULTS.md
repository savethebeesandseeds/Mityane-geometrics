# CPB V18 contiguous-mask comparison

The fixed1024 development continuation criterion failed; this declared mask-policy comparison does not support advancement.

**V16 seed946103/final2048 remains the selected research reference.** This known-development comparison does not automatically establish a new best encoder.

The candidate changes training from scattered hidden patches to one contiguous four-patch span per eligible dense channel. Architecture/native36, temporal coefficient0.1, Gaussian/temporal objective, TRAIN fitting, optimizer and evaluation stay shared. All six final1024 states were sealed before any new readout.

Controls continue their own already completed256 states through640 and1024 using the exact original executable/recipe, with no refit. Candidates start fresh, stop at512 and resume their typed state to1024. Existing controls are reused evidence, not new prefix retrainings.

## Fixed January linear readouts

The unchanged evaluation uses112 TRAIN/32 known-validation probe anchors, legacy target arithmetic and TRAIN-standardized ridge1. All six runs have complete support and independent scalar arithmetic audits.

| Seed | Arm / mask | Volatility R2 | Trend R2 | Activity R2 | Relation R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 947101 | control / scattered | 0.77558063 | 0.72720709 | 0.86052121 | 0.51534038 |
| 947101 | candidate / contiguous | 0.78565584 | 0.50536120 | 0.75658953 | 0.53107372 |
| 947102 | candidate / contiguous | 0.72868365 | 0.52452649 | 0.68557180 | 0.44874056 |
| 947102 | control / scattered | 0.80977758 | 0.79089372 | 0.77619163 | 0.45063605 |
| 947103 | control / scattered | 0.64340764 | 0.77261921 | 0.83514318 | 0.24188029 |
| 947103 | candidate / contiguous | 0.58555022 | 0.67969631 | 0.84935545 | 0.40908597 |

Candidate minus same-seed control:

| Task | Seed947101 | Seed947102 | Seed947103 | Median | Minimum | Maximum | Nonnegative pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| past_volatility | +0.010075214 | -0.081093928 | -0.057857421 | -0.057857421 | -0.081093928 | +0.010075214 | 1/3 |
| past_trend | -0.22184589 | -0.26636724 | -0.092922906 | -0.22184589 | -0.26636724 | -0.092922906 | 0/3 |
| past_activity | -0.10393168 | -0.090619824 | +0.014212272 | -0.090619824 | -0.10393168 | +0.014212272 | 1/3 |
| past_relation | +0.015733337 | -0.0018954941 | +0.16720569 | +0.015733337 | -0.0018954941 | +0.16720569 | 2/3 |

| Predeclared condition | Result |
| --- | --- |
| trend median nonnegative | FAIL |
| trend at least two nonnegative pairs | FAIL |
| activity median nonnegative | FAIL |
| activity at least two nonnegative pairs | FAIL |
| volatility median floor | FAIL |

The continuation criterion **failed**. Trend improved in 0/3 pairs; activity improved in 1/3 pairs (947103). All gains and regressions remain visible; a tie is not a gain. Relation is descriptive and does not replace the declared primary conditions.

## Actual work and reused evidence

Six complete1024 histories contain **6144 logical optimizer updates**. This study executed **5376 new physical updates**:2304 control continuation updates and3072 fresh candidate updates. The remaining **768 control-prefix updates** were already executed in V17 and were preserved without replay. Twelve new training processes implement the planned continuation boundaries. New TRAIN preprocessing fits:3; preserved control fits:3.

Actual attempts: 5376 new plus768 preserved. New scientific readout work:24 target-head fits, six independent scalar audits and24 separate scalar target refits for verification. Dataset preparation and the old TRAIN gradient diagnostic were not rerun. Artificial engineering work remains separately counted in the compact record.

## Actual contiguous-mask witnesses

Candidate logs retain returned starts, legal-span counts, quotas, exact decimal-string counter seeds, selected patch/target counts, unsupported reasons and all three temporal pair counts. On the declared dense B8/C3/H128/P8 geometry, each of24 channel draws has quota4 and13 legal starts0..12; each update selects96 patches,6912 target cells and lag1/4/16 pair counts6696/6048/3456. These are actual returned values checked per event; no second mask replay manufactured them. The original control CLI did not log detailed mask/lag fields, so those remain unavailable.

| Candidate seed | Saved attempts | Flattened channel draws | Lag1 pairs | Lag4 pairs | Lag16 pairs | Unsupported channels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 947101 | 1024 | 24576 | 6856704 | 6193152 | 3538944 | 0 |
| 947102 | 1024 | 24576 | 6856704 | 6193152 | 3538944 | 0 |
| 947103 | 1024 | 24576 | 6856704 | 6193152 | 3538944 | 0 |

## Evidence and limits

Known January development data and one1024 budget do not establish fresh-period, universal or full2048-budget superiority. Neural heads, TEMPO timing, fixed-query reconstruction, forecasting and application utility were not measured. January/February/March2024 remain consumed evidence for later choices.

The preserved `cpb_v17_ridge_witness_v1` is a generic structural tensor codec reused by the unchanged audit. It is not the encoder identity: actual v17-scattered/v18-contiguous kind, mask policy, checkpoint SHA and1024 update count are separately bound.

The [compact record](../../records/CPB-v18.span-001.json) links the admitted [protocol](PROTOCOL.md), [Operation Contract](CONTRACT.md), engineering proof, original108-file control source capsule, exact reused prefix states, twelve new training reports, all-six seal, six readouts/audits and verified source/input manifest. Full generated evidence remains under `.temp/embeddings/runs/cpb-v18-span-001/attempt-001/`. The49 frozen V16 files, three family headers and selected reference checkpoint/statistics were checked unchanged.

The reusable component retains configurable construction, TRAIN fitting, complete mask/objective/optimizer methods and typed training/resume/inference export. A benchmark geometry or checkpoint does not define the full family interface.

Generated at 2026-10-22T18:53:31.989Z by metadata-only closeout; zero model/process calls and no tensor archive decoding.
