# OR-05: CPB contiguous-mask training comparison

**Operation:** `OR-05 / CPB-span-training-fixed-linear-readout`  
**Revision:** `0.1.0` · **Declared:**22 October2026  
**Evidence at declaration:** prospective; no V18 result asserted.

This contract is saved before this study's engineering/model execution and
new readouts. It makes the unchanged evaluation and the one training-form
intervention explicit under the [project charter](../../../../../doc/operations_over_representations_design_spec_v0.1.0.md#4-common-contract-and-evidence-standard).
The [prospective protocol](PROTOCOL.md) supplies exact numerical, data,
inherited-control, timing and work-accounting bindings. Later results and
compact receipts extend the Requirements-and-Evidence Record; they do not
rewrite this declaration or the V17 evidence.

## Operation Contract

| Required field | Binding |
| --- | --- |
| Operation ID and revision | `OR-05 / CPB-span-training-fixed-linear-readout`, revision0.1.0. |
| Primary question | At the fixed1024 successful-update endpoint, does contiguous original-grid target masking improve past-window linear readouts compared with scattered masking under the same architecture/loss/optimization/data recipe? |
| Input contract | Six frozen native36 final1024 instances, exact inference/TRAIN/source/config/recipe identities, seed, scaler and bank; three scattered controls continued from their immutable typed256 states and three fresh V18 span candidates. Study inputs are C3/H128/F9/P8 at minute interval60000, ordered semantic IDs0/1/2. Reuse existing112 TRAIN/32 known January VALIDATION H256 nonoverlap probe anchors, retaining genuine F64 recent128 observations, Boolean support, cutoffs and source/row/group associations. Original target archives and legacy F32→F64 target/support/association parity are mandatory. |
| Output contract | Per row, four CPU F64 point predictions in fixed order: past volatility (`logreturn`), trend (`logreturn/tick`), activity (`log1p(base_volume)`) and relation (`correlation`). Save TRAIN feature/target means/scales, ridge weights/intercepts, features/predictions/targets, support and row/group/cutoff IDs; report R2/MAE/RMSE/coverage for every arm and task, all three paired differences and median/minimum/maximum. No predictive interval or output clipping is introduced. |
| Required representation structure | Finite valid native36 coordinates in each instance's own learned20+spectral12+mean-change4 layout, causal cutoff encoding and stable frozen coordinates during fitting/scoring. Each instance fits its own head; shared latent axes, distance geometry or an affine alignment across instances are not assumed. |
| Other dependencies | Existing V17 TRAIN1024 fit/training archive and112/32 readout archives, original CACHE88/source/calendar admission and target archives; unchanged target definitions, TRAIN population standardization with1e-8 floor, ridge penalty1 with unpenalized intercept, original scoring and independent scalar long-double Cholesky audit at abs1e-9+rel1e-8. Producer changes are limited to typed checkpoint loading,1024 binding and evidence naming. All six final states seal before any new extraction/head fit. |
| Scope of claim | Controlled encoder-training intervention measured by fixed-capacity observational readouts on consumed January development data. The intervention is training-mask geometry; the test is unchanged. It is not historical V16 reproduction, fresh confirmation, a2048-budget comparison, forecasting, application utility, raw-baseline superiority, timing retention, identified causal effect or an ultimate-encoder claim. |
| Failure and abstention | Identity/source/config/checkpoint/recipe/counter mismatch; unavailable exact control executable or prefix; nonfinite/invalid features, targets or predictions; target/row/group/cutoff mismatch; TRAIN/validation overlap; zero scored-target variance; incomplete support or arithmetic failure; missing pair or exhausted2048 attempt ceiling. For span construction, M<3 or no legal original-grid q-span selects no targets for that channel, with explicit reason and no fallback; a batch without eligible channels consumes an attempt without updating. Preserve failures/partial states rather than changing rows, seeds, quotas or budgets. |

Target definitions and support stay exactly those in the
[prior operation contract](../cpb_v17_balance_v1/CONTRACT.md#target-point-estimates)
and [frozen target source](../cpb_v16_temporal_v1/deps/src/fixed_regression.cpp):
per-channel supported adjacent close-logreturn mean/stddev, observed nonnegative
base-volume log1p means, and supported pairwise return correlation. At least
eight supported observations/returns and the original relation variance/pair
requirements remain unchanged. V18 does not add supervised task targets to
encoder training.

## Linked Requirements-and-Evidence Record

| Required field | Declaration and evidence |
| --- | --- |
| Hypothesis | Contiguous original-grid masking, with the same target-patch quota, may improve temporal information retained by a fixed native36 encoder compared with scattered targets. Protocol “Question and one controlled change” and “Portable mask contract” fix the intervention. The causal explanation is a hypothesis, not an established diagnosis. |
| Data and provenance | The exact existing TRAIN1024 and readout112/32 archive hashes, original CACHE88/calendar roles, original target hashes, control report/state/executable/settings identities and fresh candidate fitting bindings are in PROTOCOL.md and later admitted manifests. TRAIN is January1–21, known validation January22–28, recent128 from original H256 readout support. No new data is opened. |
| Controls and baselines | All three existing coefficient0.1 scattered-control TRAIN states at256 are continued unchanged to1024; all three candidates start fresh with matching same-seed initial/scaler/bank numerical identities. All rows, optimizer/stream settings and objective semantics match; training masks intentionally differ. Coefficient0.03 trajectories are not extended. Historical V16 is a retained research reference, not an initializer or identically reproduced control. No raw/trivial/random-encoder utility comparison is added. |
| Evaluation protocol | Fixed1024 successful-update endpoint and2048 total-attempt ceiling. Controls256→640→1024; candidates0→512→1024, with actual typed resume, no refit on resume and no intermediate/256 head gate. All six seal before new extraction/fitting. Preserve original112/32 selections, exact target parity, TRAIN-only normalizers, centered ridge1 equations, scoring and independent audit. No head tuning, seed replacement, optional stopping or outcome-dependent extension. |
| Measures of success | Valid complete six-arm evidence plus unchanged predicates: trend median paired R2≥0 and≥2/3 nonnegative pairs; activity same; volatility median≥-0.01. Relation descriptive. Report all arms/tasks/metrics, conditions and ties/gains separately. A pass is only a development continuation signal and not promotion. |
| Robustness checks | Three paired seeds947101/2/3 with matched row/mask/context offsets+1000/+2000/+3000 and alternating arm order. Report paired ranges, candidate actual span/support/lag counts and both arms' recorded attempts/skips. Original control logs lack per-lag counts; any optional mask-only reconstruction is labeled reconstructed, never measured. Engineering covers supported alternative geometry, missing/partial original positions, quota, no-legal-span abstention, exact continuation/export and refusal. No confidence interval, bootstrap, new month, neural heads, fixed-query reconstruction or TEMPO panel is run. |
| Results and limitations | Unmeasured at declaration; later all-arm results/compact record/audits are authoritative. Known January and reused controls are exposed development evidence. A1024 endpoint cannot settle2048 behavior or fresh generalization. Different mask geometry intentionally changes pair support; raw gradients do not quantify AdamW-preconditioned effects. Valid negative work may complete; invalid/partial execution cannot establish a scientific negative. |
| Reproduction pointer | This protocol/contract; actual new source/config/executable and phase manifests; immutable V17 control-prefix reports/states and original trainer/settings/data pins; candidate initial/fit and typed512/1024 state; new control640/1024 state; all-six seal; unchanged probe/target archives; adapted loader/update binding, saved ridge equations/predictions and independent audits. Record new physical5376, reused historical768 and logical6144 updates separately, plus actual attempts/fits/engineering/failed work. Full numerical method and supported configuration stay exportable with the component. |
| Decision | Prospective, unmeasured. Later record passed/failed/inconclusive reasons under the fixed criterion. Keep V16 seed946103/final2048 as reference. No automatic new-best designation, checkpoint/seed/application selection, longer study, external coordination, commit/push or publication follows. |

The benchmark remains model-owned evaluation under `code/embeddings/evaluation/`;
this contract does not move operation implementation or application ownership.
The [shared encoder instructions](../../../encoders/AGENTS.md) require reusable
construction, complete training semantics, supported configuration and typed
artifacts alongside research evidence. Application projects own substantial
application training and final docking.
