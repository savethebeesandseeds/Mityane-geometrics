# OR-05: CPB bounded past-window linear readout

**Operation:** `OR-05 / CPB-bounded-past-window-linear-readout`  
**Revision:** `0.1.0` · **Recorded:** 22 October 2026  
**Evidence at this supplement's declaration:** bounded development study in progress; readout results pending. Current outcomes are recorded in the linked closeout, without rewriting this declaration.

This documentation supplement makes the existing
[prospective protocol](PROTOCOL.md) explicit in the
[project charter's contract and evidence format](../../../../../doc/operations_over_representations_design_spec_v0.1.0.md#4-common-contract-and-evidence-standard).
It is prepared during encoder training and before readout/head execution.
It restates the declared operation, comparison and limitations; it changes no
arm, population, execution rule, budget, threshold or permission. The admitted
protocol remains unchanged. The supplement receives its own source pin before
head execution rather than being represented as an earlier admitted artifact.

## Operation Contract

| Required field | This bounded operation |
| --- | --- |
| **Operation ID and revision** | `OR-05 / CPB-bounded-past-window-linear-readout`, revision `0.1.0`. |
| **Primary question** | At 256 successful encoder updates, how does temporal coefficient 0.03 change linear access to four past-window quantities relative to the same-seed coefficient 0.1 control on the declared January development split? |
| **Input contract** | A frozen final256 native36 observation representation, its exact source/configuration and typed inference-checkpoint identity, actual coefficient, initialization seed and embedded frozen TRAIN scaler/bank. Geometry is C3/H128/F9, minute interval 60000, channel IDs 0/1/2 for BTC/ETH/SOL; native coordinates are ordered level32 plus change4. Each input retains the most recent 128 ticks from a legal original H256 nonoverlapping probe selection, with genuine F64 observations, Boolean support, cutoff, row/group associations and declared role. Use exactly 112 TRAIN and 32 known-VALIDATION rows. The four targets below must equal the original legacy F32-loader-to-F64 target values/support and original row/group IDs. [Configuration](../../../encoders/causal_patch_bottleneck_mae/versions/v17/config/research.conf), [typed artifact contract](../../../encoders/causal_patch_bottleneck_mae/versions/v17/README.md#artifact-contract), and protocol sections “Data identities and temporal boundary” and “Fixed final256 January development readouts” bind the details. |
| **Output contract** | Four CPU F64 point predictions per scored row, in the target order/units below, together with target and feature validity, cutoff/row/group associations, fitted TRAIN feature/target means and scales, ridge weights/intercept, predictions/targets and per-target coverage, R2, MAE and RMSE. The saved `ridge-witness.pt` carries these equations and associations; `summary.json` carries the metrics and artifact bindings. Predictions are linear point estimates with no predictive intervals or output clipping. R2 and coverage are dimensionless; MAE/RMSE use the corresponding target unit. |
| **Required representation structure** | Finite native36 coordinates in the checkpoint's exact ordered layout, valid causal encoding at the supplied cutoff, and unchanged model/scaler/bank state during extraction. The operation requires stable coordinates for fitting/applying the same readout, not semantic axes, a distance metric or shared coordinates across independently trained instances. Each instance fits its own readout. |
| **Other dependencies** | Declared January CACHE88 observations and role-contained timestamp selections; the frozen target definitions and legacy target archives; complete target/support parity; TRAIN-only feature and target normalization; the unchanged centered ridge penalty 1 with unpenalized intercept. Standardization uses TRAIN population means/scales with scale floor 1e-8. Source, data, configuration, executable, checkpoint, seed and fitted-state bindings remain attached. [readout.cpp](readout.cpp) implements the operation, the [preserved target/scoring source](../cpb_v16_temporal_v1/deps/src/fixed_regression.cpp) supplies definitions, and [audit.cpp](audit.cpp) independently checks the saved scalar ridge arithmetic. |
| **Scope of claim** | A fixed-capacity observational readout comparison on known January development data, conditioned on the declared fresh training recipe. The encoder is frozen during this operation; no future target, forecasting, application utility, raw-feature superiority, timing retention, fresh confirmation, identified causal effect or decision-policy claim follows. The selected historical V16 reference is retained; the fresh 0.1 arm is the paired baseline under the new common fitting/sampling recipe. |
| **Failure and abstention conditions** | Missing/mismatched source, archive, row/group, cutoff, checkpoint, configuration or recipe identity; TRAIN/validation overlap; nonfinite input/features/targets/predictions/arithmetic; invalid/incomplete feature or target support; zero TRAIN or validation target variance; failed frozen-state or independent arithmetic checks; incomplete/exhausted trajectories or absent pairs. Unsupported metrics remain unavailable and cannot support a positive screen. No target expansion, row replacement, rescue seed or budget extension is introduced. These are the existing protocol/readout conditions. |

### Target point estimates

All targets describe the retained past 128 ticks, with the original target
arithmetic applied to F32-rounded observations promoted to F64. Encoder input
retains genuine F64 values. Exact legacy value/support and row association
checks are required by the existing readout protocol.

| Output column | Target definition and support | Unit/scale |
| --- | --- | --- |
| `past_volatility` | Mean across channels of the population standard deviation of valid adjacent close-price log returns. Each channel needs at least eight returns with both closes observed and positive. | `logreturn` |
| `past_trend` | Mean across channels of the mean valid adjacent close-price log return, using the same movement support. | `logreturn/tick` |
| `past_activity` | Mean across channels of the mean `log1p(base_volume)` over observed nonnegative base-volume cells; each channel needs at least eight supported cells. | `log1p(base_volume)` |
| `past_relation` | Mean correlation across eligible channel pairs using common valid adjacent log returns. A pair needs at least eight common returns and population variance at least 1e-12 in each series; at least two channel pairs must qualify. | `correlation`, dimensionless target in [-1, 1]; a linear prediction is not clipped. |

## Linked Requirements-and-Evidence Record

The record consists of this contract, the unchanged [protocol](PROTOCOL.md),
the actual source/input/admission and execution receipts, and the pending
[RESULTS.md](RESULTS.md) and
[compact record](../../records/CPB-v17.balance-001.json). The latter two are
populated at closeout and must distinguish completion from scientific verdict.
No result or promotion is asserted by this documentation supplement.

| Required field | Declaration and evidence location |
| --- | --- |
| **Hypothesis** | Under the shared fresh native36 recipe and fixed 256-update budget, lowering temporal weight from 0.1 to 0.03 can retain/improve past trend/activity readouts without crossing the declared volatility loss floor. Protocol “Question and controlled change” and “Predeclared screen decision” give the falsifiable comparison. The TRAIN-only gradient diagnostic is descriptive and cannot select a different coefficient or rescue arm. |
| **Data and provenance** | January 2024 Spot1m CACHE88, ordered BTC/ETH/SOL and nine features, with original cache byte/SHA256 identities and exact close-time role bounds in protocol “Data identities and temporal boundary.” H256 nonoverlapping selection yields 112 TRAIN/32 known-VALIDATION probes, with recent H128 encoder/target support. Exact target source and archived target/row/group parity, final checkpoint/recipe and frozen scaler/bank identities are recorded in admission, preparation, training seal, readout and audit receipts. The compact record links the actual pins. |
| **Controls and baselines** | Paired fresh coefficient 0.1 control versus coefficient 0.03 candidate, with the same configuration, admitted observations, TRAIN fitter/order, initial parameter tensors and stochastic streams within each seed pair. Historical V16 is the preserved reference and Stage A diagnostic state, not a fresh ancestor or an identically reproduced trajectory. No raw-feature, trivial, random-encoder or downstream-utility baseline is run in this bounded screen, so superiority to those alternatives is untested. |
| **Evaluation protocol** | The existing protocol fixes a zero-update 64-TRAIN-window diagnostic, then six fresh trajectories at successful B8 update256, with total attempt ceiling512 and planned exact typed resume at128. All six final states are sealed before probe extraction/head fitting. Ridge1 and its feature/target normalizers fit only the 112 TRAIN rows; scoring uses the 32 chronological known-VALIDATION rows. Target/support/association parity, frozen encoder state and independent scalar ridge audit retain the declared tolerances. There is no new tuning, early stop or outcome-dependent extension. |
| **Measures of success** | Candidate-minus-same-seed-control R2: trend median >=0 and at least two of three pairs nonnegative; activity median >=0 and at least two of three nonnegative; volatility median >=-0.01. All six trajectories and existing identity/support/arithmetic checks must be valid. Equal scores pass a nonnegative condition but do not establish a gain; report observed gains separately. Relation and any verified fixed-query reconstruction are descriptive. All per-arm R2/MAE/RMSE/coverage and condition pass/fail outcomes are retained in results/compact records. |
| **Robustness checks** | Encoder seeds 947101/947102/947103 with matched row/mask/context offsets +1000/+2000/+3000, and the predeclared alternating arm order. Report all three paired differences, medians and minimum/maximum ranges. No confidence interval, bootstrap, neural-head repetition, fresh-month subgroup/shift evaluation or TEMPO timing test is performed. Engineering parity/resume/export/refusal checks establish numerical mechanisms, not representation quality. |
| **Results and limitations** | Pending in [RESULTS.md](RESULTS.md), [compact record](../../records/CPB-v17.balance-001.json), and the actual `screen-summary.json`, sealed trajectories, readout witnesses and independent audits under the named run leaf below. Known January development exposure, small probe population and one early budget limit the inference. January/February/March are consumed evidence; timing, full-budget superiority, downstream utility and fresh confirmation remain unmeasured here. Preserve failures and incomplete attempts as execution outcomes, not valid negative scientific scores. |
| **Reproduction pointer** | [run-study.mjs](run-study.mjs), [readout.cpp](readout.cpp), [audit.cpp](audit.cpp), the preserved target/scoring source linked above, versioned settings and full source/configuration/executable/input pins. Actual artifacts reside under `.temp/embeddings/runs/cpb-v17-balance-001/attempt-001/`: admission/phase receipts, `training-seal.json`, per-seed fresh/resume reports and checkpoints, probe preparation, readouts/audits and `screen-summary.json`. Data and checkpoints remain separately controlled artifacts; the public source package carries the method, contract and compact evidence pointers. |
| **Decision** | Pending bounded development verdict, to be stated with reasons in results and the compact record. A valid screen pass is only a continuation signal; a valid failure/inconclusive result may still complete this bounded work. No automatic reference promotion, best-seed selection, application update or authority for further experimental execution follows. V16 remains the selected research reference unless a separate justified selection changes it. |
