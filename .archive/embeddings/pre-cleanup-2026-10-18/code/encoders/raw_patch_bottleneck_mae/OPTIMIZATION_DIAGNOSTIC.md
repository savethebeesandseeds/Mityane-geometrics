# RPB v5 optimization diagnostic

Date: 2026-10-07. Status: measured; independent audit passed.
Protocol: optimization-validation-v1; policy1.2; development, TRAIN/VALIDATION only.

Additional optimization lowers fixed-query reconstruction error and improves
the fixed neural readout, but does not improve mean native linear VALIDATION
accuracy:84.38% at512 and1024 updates, then82.81% at2048. This diagnoses the
existing v5 trajectories on known development data. It does not select a budget,
repair the [rejected fixed512 comparison](PAIRED_POOLING_ADVANCE.md), open TEST
or promote v5. RPB-v4 remains active.

## Fixed conditions and quality

Timing (`lag_sign`) only. Each of the three existing trajectories, masters
3101/3202/3303, retains128 TRAIN source pairs (256 examples) and64 VALIDATION
source pairs (128 examples). No quality cohort was generated and no TEST/stress
payload was opened. A separate engineering CUDA gate used tiny development
fixtures before admission; it supplies no quality score.

The [frozen card](../../evaluation/cards/optimization_validation_v1.md) binds
each exact saved512 checkpoint, ordinary AdamW state, scaler, settings and
attempt counters. Ordinary `run_cli train --resume --steps` adds512 updates to
reach1024, then adds1024 from that exact checkpoint to reach2048. All six new
points have attempted=completed=absolute budget, finite changed weights and
unchanged TRAIN/scaler association. No initialization or architecture changed.
C3/H32/F3, patch8, mode3/mixer1/native32,269,389 parameters and batch8 remain
fixed on actual CUDA, with unchanged objective, AdamW, clipping and row/mask RNG
policies. These are three continuations, not nine independently trained encoders.

RPB-v5 pools the canonical aligned patch/channel states and visibility bits
directly into its sole32-number served reconstruction bottleneck. No PCA or
random projection follows this encoder. Raw data supplies288 legally observed
TRAIN-standardized values plus288 visibility flags. PCA only fits32 components
on that raw TRAIN representation, with no encoder.

The primary readout is Ridge with penalty1. The secondary has16 tanh hidden
units, Adam0.01 and100 updates. Both fit their normalizers and weights on TRAIN
only. All three paired head seeds2701/2802/2903 are retained; equal-width
methods use the same actual `stream_seed(rep_seed,width)`. Native/PCA32 heads
have66/562 parameters, versus1,154/9,266 for raw576. Table means retain all
three masters and three head fits per point; the head fits are not independent
encoder runs. Ridge is identical across its three seed repetitions here.

| Method | Size | Encoder updates | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | N/A | 52.86 | 52.60 | 100 |
| PCA only — no encoder | 32 | N/A | 51.82 | 76.56 | 100 |
| RPB-v5 | 32 | 512 | 84.38 | 95.83 | 100 |
| RPB-v5 | 32 | 1024 | 84.38 | 96.01 | 100 |
| RPB-v5 | 32 | 2048 | 82.81 | 97.83 | 100 |

The512 encoder row reuses its original saved VALIDATION heads and predictions;
there were no512 head refits. The six new-point inputs use the shared
`ArchiveReadoutRun`: raw ObservationScaler, raw outer normalization/PCA and
raw/PCA heads are refitted on the same TRAIN separately per input. Their fitted
tensors and predictions are exactly equal between1024/2048 for each master/head,
so one control row is shown. All raw PCA fits have numerical TRAIN rank223 and
support32 components. These refits use no VALIDATION preprocessing and retrain
no reference encoder. No fit or scoring row was unsupported or dropped.

The native linear VALIDATION scores vary by master:

| Master | 512 linear % | 1024 linear % | 2048 linear % |
| --- | ---: | ---: | ---: |
| 3101 | 84.38 | 80.47 | 88.28 |
| 3202 | 81.25 | 81.25 | 79.69 |
| 3303 | 87.50 | 91.41 | 80.47 |

The per-master spread changes rather than improving uniformly. All methods
retain the full128 VALIDATION examples/64 source groups per master, and every
declared paired comparison uses that same population. The
[readout report](../../../output/runs/rpb-optimization-validation/optimization-validation-sHD27P/results/readouts/report.json)
retains every master/head score and54 native/raw/PCA paired comparisons with
95%/1,000-draw source-group intervals. Those intervals are conditional within
master on the fitted checkpoint/readout. No across-master or cross-milestone
confidence interval was estimated.

## Reconstruction and cost

RPB-v5 reconstructs solely through exact native32. Errors use the same four
original-patch query over legally observed target cells, the frozen TRAIN scaler
and equal cell -> channel -> example reductions. Natural missingness remains;
all points retain the same requested/eligible target cells and full coverage.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v5 | 512 | 0.09027 | 0.09474 | 14.37 |
| RPB-v5 | 1024 | 0.06569 | 0.06902 | — |
| RPB-v5 | 2048 | 0.05531 | 0.05964 | — |

Errors are mean standardized MAE across the three masters. The512 errors and
GPU time are reused measurements. At new points, GPU-update-only time is
unmeasured (`—`). The adapter measured a different scope: CUDA-synchronized
whole ordinary resume command, including load/normalize/update/log/save and
excluding adapter checks, feature extraction and classifier fitting.

| Continuation | Additional updates | Mean command wall seconds |
| --- | ---: | ---: |
| 512 ->1024 | 512 | 13.32 |
| 1024 ->2048 | 1024 | 26.43 |

The two continuation commands total39.75 seconds per trajectory on average.
Do not add that whole-command duration to the reused14.37-second GPU-update
measurement or infer a speed advantage from the different scopes. The
[continuation report](../../../output/runs/rpb-optimization-validation/optimization-validation-sHD27P/results/continuation-report.json)
retains per-master MAE/Huber, target counts, command timing, parent/settings
overrides and optimizer/counter evidence.

## Evidence and identities

Measured capsule:
[optimization-validation-sHD27P](../../../output/runs/rpb-optimization-validation/optimization-validation-sHD27P/).
The58-file compiled source snapshot is
`9ad33e0a91b98ac4c74efb94e5dd1ad1cdb992b1377b502967eba943f71d9238`;
recorded Git head `101612d290dcbbfb4fd2aab4ed549a4acd6a1f07`, dirty at build.
The immutable source snapshot identifies the compiled working tree.

| Artifact | SHA256 |
| --- | --- |
| Frozen human card | `e24c439e769384f27aeece95a99475ff5677de944f3d23746964a5eb4e5e74e7` |
| Machine launch plan | `13bb69e438aafa422ca41e02c6371be84dc1c489363346e0cc4a665879a88145` |
| Retained-role manifest | `eaa41452332f95e3f8b343d63192f21233f38f1037a81aae6b7a1e2a4e9904b4` |
| Continuation report | `ee1c0a3a0dbaf6b058df6932ae145cf5dd68546f52586f7eae4884672c7628d5` |
| Shared readout report | `f7a124257db776a8319d00e5cdcca6f1ac9d585b8241e8fdac1e7252ace65475` |
| Shared readout card | `f766e1e615a6f78915db07cd7ea55cfae58ae67677a742be0a79c8b9cfbccff3` |
| Shared output manifest | `a4732b568f587e5999080f5f99914413d0179ad7638b0159905c8fc7a53b77f3` |

The [original parent capsule](../../../output/runs/rpb-paired-pooling/paired-pooling-BFFPX5/)
retains its1,326-file inventory SHA256
`610976e666468ac2f80404bd56884e36f9275645305e83a9d0e50b762e702be3`.
Only its52 explicitly declared TRAIN/VALIDATION/checkpoint/fitting roles were
hashed/read in this diagnostic. All52 remain byte exact, and the inventory
metadata is unchanged; this did not rescan its excluded TEST/stress payloads.
The frozen card records each exact512 parent hash. New ordinary checkpoint
identities are:

| Master | Updates | Checkpoint SHA256 |
| --- | ---: | --- |
| 3101 | 1024 | `8134d7b75ac281e10290be57cd5dc5ca54d04e14660d43266f9fed11c22b0f33` |
| 3101 | 2048 | `1f1aeb1223c07fa359a7a2cba98d2762f5ecb6f2ad656d31895471b628439469` |
| 3202 | 1024 | `35a79508f41c28f6ddeb6cef60f295b8176155148f92991089f79d61300b6b64` |
| 3202 | 2048 | `694795adda242d33ad95c6d04fcfa8238dd3d6fbb0888b91b270e21afcd51e48` |
| 3303 | 1024 | `675b5edf4aa2464c28c34c59ab51c5b5c6ccdf68abde356c06ba76945164674d` |
| 3303 | 2048 | `df334b90e94858fc3094863d22645694b711f2d8bfc54ff51511b04eee38b232` |

Independent stdlib QA
[passed41,001,198 checks](../../../output/runs/rpb-optimization-validation/audit-tools/independent-optimization-20261007-v1/run-sHD27P-v1/validation.json):
three cohorts/six new points, nine cached512 native fits,54 new fits,
117 prediction archives,54 paired comparisons, six optimizer transitions,
768 named optimizer-state inspections and30 fixed-query reconstruction audits.
It checks512 exports/queries before and after continuation, TRAIN-only fitted
statistics, native-no-PCA, prediction replay, PCA scope/rank, target reductions,
exact counters/scalers and1024 ->2048 named AdamW moment continuity. It reads
576 distinct TRAIN/VALIDATION source groups and zero TEST/stress payloads.

Auditor SHA256:
`bde3f83747842e215aec9ca13d207cd32ff0833571832ec7b8f9898b2e4137ed`.
Audit JSON SHA256:
`07bc3533e53193f986140183b744c7d563c86aa11b2950e40cd23ee7270e4238`.
The41-million checks include repeated numerical comparisons; they are not
independent statistical trials. Bootstrap populations/estimates/bounds were
checked, not every bootstrap draw. CUDA checkpoint/moment association and
finite-weight/device continuity rely on the coordinated C++ tests/gate plus
CPU state witnesses; the independent reader executes no model.

## Interpretation and next action

More optimization helps v5's reconstruction route and fixed nonlinear access
on known VALIDATION, without a mean linear gain at these measured budgets.
The observed per-master shifts and secondary neural improvement do not establish
an unseen benefit or a repaired fixed512 result. No checkpoint was selected or
promoted. RPB-v4 remains active, and the v5 fixed512 rejection remains bounded
to its unchanged original card.

The next bounded mechanism is a separate RPB-v6 candidate: retain v4's inference
architecture/native32 and add fixed30% deletion of visible TRAIN context,
preserving the target query, Huber objective, heads, scaler and512-update budget.
It needs its own prospective card, paired admission checks and fresh TEST
namespace. This completed optimization diagnostic authorizes none of those
future TEST measurements and will not be used to promote v5.
