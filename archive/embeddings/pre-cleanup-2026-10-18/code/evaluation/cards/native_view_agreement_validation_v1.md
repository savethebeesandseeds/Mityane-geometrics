# Native view agreement validation v1 development card

Date: 2026-10-07. Protocol: `native-view-agreement-validation-v1`. Policy: 1.2.
Status: prospective; implementation/admission/measurement not yet completed.
RPB-v4 remains active. Freeze this card, compiled sources, exact parent role
manifest and coordinated actual-CUDA admission before any quality fitting.

## One fixed mechanism

RPB-v9 uses the [native view agreement plan](../../encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_AGREEMENT_PLAN.md).
It retains RPB-v4's mode2/mixer1/native32 architecture, all 225,805 parameters,
C3/H32/F3, semantic IDs 0/1/2, patch8, width64, temporal layers3, heads4,
feed-forward256 and decoder128. No new module, teacher, learned projection,
decoder capacity, serving transform or head recipe. Inference remains ordinary.

Policy: `rpb-training-native-view-agreement-015-v1`. For the original legal
TRAIN minibatch, preserve O, A, Q=O&A, original eligibility and scaler.
Let V_o=O&~A and let V_s=V_o&~E, where E is the existing repaired
`coordinate15_v1` plan at the same absolute attempted counter. Preserve
original sampled rows, mask/Torch streams and context stream/semantic ordinals.
Zero nonvisible values before both encodes. Compute exact served32 z_o and z_s.
Decode z_o only, using original target metadata and hierarchical Huber delta1.

Before update0, calibrate s0 on all legal TRAIN rows under original O using
the candidate point0 CUDA model in evaluation mode. No labels or clean hidden
values enter calibration. Save native32 calibration features, valid rows,
source/order and the per-coordinate population standard deviation computed
in float64, floored at1e-6, then stored as float32 s0. Require finite positive
s0 and at least two valid source groups. Fit once, preserve its identity and
never recompute. Restore RNG/threads around calibration. This is a loss-only
scale; feature exports and TRAIN-fitted head normalizers stay unchanged.

With u_s=z_s/s0 and u_o=detach(z_o)/s0, use exactly:

```text
L = original_hierarchical_Huber(decode(z_o), Q)
  + 0.05 * mean((u_s - detach(u_o))^2)
  + 0.01 * mean_d relu(0.5 - sqrt(population_variance(u_s)_d + 1e-4))
```

Agreement averages the32 coordinates and supported rows. Auxiliary support S
requires original reconstruction eligibility, joint sample validity, matching
row IDs/channel IDs/endpoints and equal channel-valid masks in both views.
The positive is the exact same row, never opposite-label siblings. Variance
uses the first supported sampled occurrence of each distinct source ID; with
fewer than two distinct groups it is differentiable zero. Use population
variance. Unsupported auxiliary rows retain their ordinary reconstruction.
The target receives no agreement gradient; shared encoder parameters receive
ordinary reconstruction and student auxiliary gradients. All coefficients,
floor, epsilon, threshold and support rules are fixed prospectively.

## Fresh fixed512 training and snapshots

Train five fresh candidate encoders for masters4404/5505/6606/7707/8808.
Use learned_global.conf, actual CUDA, threads1, batch8, dropout0,
AdamW0.001, weight decay0.0001 and clipping1. Keep original mask ratio0.25
and legal TRAIN scaler floor1e-6. Save0/512 on one live trajectory.
Require attempted=completed=512 and sampled_rows4096; an ineligible original
attempt aborts before scoring, without replacements. Save s0, calibration,
component losses, eligible/auxiliary/unique-group counts, original mask and
extra-deletion trace identities, CUDA flags and synchronized training-loop
seconds. Calibration time is reported separately from update-loop time.

Ordinary resume must reject the tagged policy. No optimizer reload contract
is introduced. Immutable saved snapshots use existing frozen providers;
neither scoring nor checkpoint inspection may change live parameters,
optimizer state, counters, RNG, scalers or s0.

## Two retained parents and exact permitted roles

Task is timing `lag_sign`:256 TRAIN rows/128 source pairs and128 known
VALIDATION rows/64 pairs per master, with10% natural missingness. Retain
five v4 and five v7 encoders; saved points and repetitions are not new runs.

Parent v4:
`output/runs/rpb-context-replication/context-replication-JjNEUc`,
inventory SHA256 `13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9`,
orchestration source `9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828`.
Parent v7:
`output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2`,
inventory SHA256 `d4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821`,
orchestration source `f22cc8d5f2ac8af3e6a6056c6ad2693ffcc82cf39e8744c32fb13038710c5311`.

The exact326 roles are generated metadata-only before payload access from
the following closed template. Let prefix(v4)=`reference`,
prefix(v7)=`results/candidate-development`, and cohort(master)=
`seed-<master>-lag_sign` for the five declared masters.

- Under each parent/prefix: `native-development-card.json`,
  `development-complete.json`, `validation-report.json`.
- Under each parent/prefix/cohort: `controlled-training.pt`,
  `controlled-validation.pt`, `development-manifest.json`, `trainer-audit.json`.
- Under each parent/prefix/cohort/milestone-0 and milestone-512:
  `checkpoint.pt`, `checkpoint.pt.audit.pt`, `checkpoint.pt.scaler.pt`,
  `checkpoint.pt.training-raw.pt`, `native-training.pt`, `native-validation.pt`,
  `point.json`, `training-reconstruction.pt`, `validation-reconstruction.pt`.
- At milestone-512 only, under rep-1/rep-2/rep-3: `fit.pt`,
  `training-predictions.pt`, `validation-predictions.pt`.
- Under v7 only, `results/validation-views/<cohort>/validation-dropout-030.pt`
  and the matching `.json`.

There are158 v4 and168 v7 roles. Manifest binding is exact
`SHA256  parent_id/relative_path`; reject missing, duplicate, redirected,
extra or undeclared paths before the first role hash/decode. Read the pinned
inventory metadata and these roles only. No recursive historical discovery,
TEST/stress/report.json payload or preservation hashing of such payloads.
The TRAIN objective diagnosis's195-role permission is not reused here.

Pregenerate only TRAIN/known VALIDATION and require exact values, masks,
labels and source order against both retained parents. Encoder callbacks
receive legal TRAIN observations/metadata under `native-development-v1/lag_sign`,
never labels/hidden values/heldout fitting inputs. Before training512, require
all common initialized parameters/buffers, scaler and original stream plans
to agree against retained v4 point0; independently gate retained v7 point0
through its unchanged explicit0.15 admission. Candidate scale calibration
must not change any common initialization or stream.

## Fixed views, heads and reconstruction

Use intact known VALIDATION and the exact saved v7 additional30% coordinate
deletion view, copied without a new draw or support repair. Preserve namespace
`context-lighter-validation-v1`, stream `0x636c763164726f70`, source-paired
erasures, physical channel order0/1/2 and original labels/source order.
Verify requested erasures, O'=O&~E, retained values and zero hidden storage.

Use native32 with no PCA. Keep Ridge penalty1 and tanh16/Adam0.01/100 updates,
all repetitions2701/2802/2903 and existing width-derived actual seed law.
Fit weights and affine maps only on ordinary TRAIN, separately per checkpoint;
apply the same fit to both VALIDATION views. The positive matrix has15 inputs
(five masters times v4/v7/v9),45 pipelines per supported method and135 total.
Refit retained controls transparently, then require exact native fit and
TRAIN/intact prediction parity against each retained cache. Preserve all
unsupported reasons and all masters/repetitions.

Raw576 uses TRAIN-standardized observed values plus observation flags;
standalone PCA32 is fitted only on raw TRAIN, labelled "PCA only — no encoder".
Verify common raw/PCA maps/fits/predictions across versions. No head tuning,
corrupted-view fitting, best-head choice or encoder selection. Report paired
v9-v4 and v9-v7 effects on exact common support, with abstentions counted as
failures. Use1000 source-group bootstrap replicates and95% within-master
intervals; do not treat them as across-master confidence intervals.

Retain original unaugmented four-patch TRAIN/VALIDATION query arrays and
standardized MAE/Huber reductions. Retained exports/queries reproduce their
saved values/support; candidate exports match its saved checkpoint provider.
Report all per-master costs, distinguish new versus retained timings, and
state that equal updates are not equal compute because v9 uses two encodes.

## Actual CUDA admission and fixed decision

Before quality fitting, admission must prove real CUDA parameters, ordinary
and student inputs, all loss components/gradients and actual weight updates;
exact ordinary scalar/gradient parity when auxiliaries are disabled; target
detach; original Q/eligibility/scaler preservation; semantic-coordinate
repair; hidden-value poisoning invariance; finite constant-input variance
gradients; unique-source/unsupported handling; full common initialization;
and deterministic uninterrupted0→4 versus live0→1→2→4 weights, buffers,
AdamW, scaler, s0, counters and trace parity. Preserve old policies and their
regression tests, tagged ordinary-resume rejection and earlier snapshots.
Quality measurement binds passed admission source/log SHA. The generic
native32 serving gate alone is insufficient. Independent saved-arithmetic
audit and all source/input preservation gates must pass before reporting.

At512 the joint guard requires mean AND worst-master native Ridge accuracy
in BOTH VALIDATION views at least their retained v7 values, equal coverage,
and mean TRAIN AND VALIDATION fixed-query MAE each no worse than retained v4.
Neural accuracy is secondary and cannot rescue a failure. Finish all five
declared candidates, disclose per-master tradeoffs, then stop this mechanism
if any joint guard fails. No coefficient, rate, budget or pair-subset rescue.
Even a pass does not open TEST/stress, select a checkpoint or promote v9.
Known-VALIDATION development evidence needs separate fresh confirmation.
