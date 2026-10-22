# CPB V17 temporal-strength bounded study

The bounded development criterion failed; these results do not support advancing this strength ablation under the declared rule.

**The selected research reference remains V16 seed 946103/final 2048.** This 256-update development screen does not establish a new best encoder.

The candidate uses temporal weight 0.03; the fresh control uses 0.1. Both share the corrected configurable native36 component, 1024-window Chan TRAIN fitting, context deletion NONE, B8, the same seed streams and one AdamW/global clip 1. Each trajectory starts fresh, stops at 128 and resumes its exact typed state to 256. These controls do not reproduce the historical V16 training trajectory.

## Final256 January development readouts

R2 below uses 112 TRAIN/32 known-validation probe anchors, original legacy target arithmetic and TRAIN-standardized ridge 1. All six readout runs have complete support and independent arithmetic audits. Full-precision R2/MAE/RMSE/coverage and checkpoint identities are retained in the compact JSON record.

| Seed | Arm | Volatility R2 | Trend R2 | Activity R2 | Relation R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 947101 | control (0.1) | 0.6147668 | 0.2537839 | 0.7241147 | 0.4240029 |
| 947101 | candidate (0.03) | 0.5990076 | 0.2475242 | 0.7011696 | 0.3898740 |
| 947102 | candidate (0.03) | 0.7312258 | -0.7882823 | 0.6263762 | 0.3846897 |
| 947102 | control (0.1) | 0.7336444 | -0.9102837 | 0.6299277 | 0.3684888 |
| 947103 | control (0.1) | 0.5922134 | 0.3077009 | 0.8205550 | 0.4637911 |
| 947103 | candidate (0.03) | 0.5924983 | 0.3046718 | 0.8115596 | 0.4669771 |

Candidate minus same-seed control:

| Task | Seed 947101 | Seed 947102 | Seed 947103 | Median | Minimum | Maximum | Nonnegative pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| past_volatility | -0.01575917 | -0.002418553 | +0.0002848377 | -0.002418553 | -0.01575917 | +0.0002848377 | 1/3 |
| past_trend | -0.006259664 | +0.1220014 | -0.003029098 | -0.003029098 | -0.006259664 | +0.1220014 | 1/3 |
| past_activity | -0.02294518 | -0.003551464 | -0.008995412 | -0.008995412 | -0.02294518 | -0.003551464 | 0/3 |
| past_relation | -0.03412885 | +0.01620092 | +0.003185947 | +0.003185947 | -0.03412885 | +0.01620092 | 2/3 |

| Predeclared condition | Result |
| --- | --- |
| trend median nonnegative | FAIL |
| trend at least two nonnegative pairs | FAIL |
| activity median nonnegative | FAIL |
| activity at least two nonnegative pairs | FAIL |
| volatility median floor | PASS |

The declared continuation criterion **failed**. An observed positive trend/activity pair difference occurred. Equal scores satisfy nonnegative conditions but are not an improvement. No automatic promotion or longer execution follows.

## TRAIN-only zero-update diagnostic

Eight fixed B8 batches from 64 TRAIN windows were evaluated using the selected frozen V16/final 2048 checkpoint and its original fitted statistics. The 32 parameter-derivative evaluations performed no preprocessing fit, optimizer step or artifact export. These local gradients do not identify the cause of earlier quality changes.

Distributions below are medians [minimum, maximum] over the eight batches; null denominator cases remain null in the JSON record. The coefficient-weighted ratio is temporal-gradient norm divided by Gaussian-gradient norm. Clipping scales are hypothetical.

| Parameter group | Gaussian L2 | Temporal L2 | Cosine | Weighted ratio 0.1 | Weighted ratio 0.03 | Clip scale 0.1 | Clip scale 0.03 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all learned | 0.6557786 [0.2503504, 8.422768] | 0.03013188 [0.02324884, 0.06434740] | 0.08667559 [0.02522118, 0.2428225] | 0.005474344 [0.0002760238, 0.01343785] | 0.001642303 [0.00008280714, 0.004031356] | 1 [0.1187250, 1] | 1 [0.1187255, 1] |
| encoder and mean decoder | 0.6057470 [0.2299374, 8.421918] | 0.03013188 [0.02324884, 0.06434740] | 0.09201350 [0.02522373, 0.2639269] | 0.005763336 [0.0002760517, 0.01357286] | 0.001729001 [0.00008281550, 0.004071858] | 1 [0.1187370, 1] | 1 [0.1187375, 1] |
| backbone without waveform decoder | 0.2381305 [0.09306268, 2.109331] | 0.003285935 [0.002107123, 0.005302945] | 0.1087353 [-0.2280247, 0.5314128] | 0.001045662 [0.0001667635, 0.002814862] | 0.0003136986 [0.00005002904, 0.0008444585] | 1 [0.4740852, 1] | 1 [0.4740841, 1] |
| waveform mean decoder | 0.5240878 [0.1894363, 8.036188] | 0.02841848 [0.02206136, 0.06324969] | 0.09932615 [0.02713663, 0.2928121] | 0.006668671 [0.0002745252, 0.01510450] | 0.002000601 [0.00008235755, 0.004531350] | 1 [0.1244361, 1] | 1 [0.1244368, 1] |
| variance head | 0.1005791 [0.06737138, 0.3868224] | 0 [0, 0] | null (8/8) | 0 [0, 0] | 0 [0, 0] | 1 [1, 1] | 1 [1, 1] |

Global learned-gradient cosines were positive in 8/8 batches and negative in 0/8. The backbone subgroup had 4/8 negative cosines. Global alignment and subgroup conflict are different observations; neither establishes behavior at every fresh-training point. Raw gradient norms and cosines do not measure AdamW-preconditioned update contributions. The variance-head temporal gradient was zero in 8/8 batches.

## Actual work and preserved failures

Scientific work: six fresh trajectories, 12 planned training processes, 1536 physical/logical optimizer updates, 1536 attempts, six TRAIN preprocessing fits, 24 linear target-head fits and six independent scalar audits. Those audits separately recomputed 24 target fits for arithmetic verification. The diagnostic added 32 derivative evaluations and zero training updates.

Artificial engineering work is separate: 859 assertions, 126 named gradient tensors compared, 16 synthetic optimizer updates, 2 preprocessing fits and 23 deliberate runtime refusals. The initial failed compile executed zero models/updates. Its preserved record is a transcription of the actual Codex tool output; a separately captured raw compiler log is unavailable. Earlier source admissions and the corrected build remain linked.

## Scope and evidence

Fixed-query reconstruction, TEMPO timing, neural heads, fresh-population confirmation, full-budget superiority, forecasting and application utility were not measured. January/February/March 2024 are consumed evidence for later selection. V16’s historical mixed verdict and timing limitations remain in its original records.

The reusable component retains configurable construction, TRAIN fitting, the complete training objective, typed resume/export and the training source. A research checkpoint and its benchmark shape do not define the family’s full supported interface.

The [OR-05 Operation Contract and Requirements-and-Evidence Record](CONTRACT.md) restates this admitted readout scope. Its supplemental admission binds the unchanged original protocol, study admission and completed training seal before any head execution.

The [compact closeout record](../../records/CPB-v17.balance-001.json) links the full prospective admission, phase source/executable pins, engineering record, three preparation receipts, diagnostic batches, training seal, 12 training reports, six readouts, six independent audits and final summary. Full ignored evidence is preserved at `.temp/embeddings/runs/cpb-v17-balance-001/attempt-001/`; the preserved `closeout-input-manifest.json` there retains verified source and artifact pins. Frozen 49 V16 files, three family numerical headers and the selected reference checkpoint/statistics were checked unchanged at closeout.

Generated from actual saved evidence by `closeout-study.mjs` at 2026-10-22T17:30:56.783Z; no model execution or tensor decoding.
