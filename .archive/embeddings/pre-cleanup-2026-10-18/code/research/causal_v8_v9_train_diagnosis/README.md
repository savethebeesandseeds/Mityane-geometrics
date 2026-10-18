<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE. -->
# Saved TRAIN V8/V9 diagnosis — source preparation only

These five files are complete physical copies of the frozen V6/V8 diagnosis,
independently adapted in this owned directory. No old implementation is imported;
parent source/card/results remain exact. No card/context freeze, AST, schema/body,
diagnostic arithmetic, model/head/fit or training execution occurs in preparation.
Final V9 metadata/99-checkpoint preservation and reviewed/applied future retention
context remain unresolved. Root owns their later exact pins and guard-role count.

Question: what do existing TRAIN representations/readouts, reconstruction and
observed temporal learning say about anchored-local V8 versus half shared-global
V9? Known quality motivates the question but no held-out tensor/prediction
arithmetic is permitted. TRAIN evidence cannot explain held-out tradeoffs or
establish pooling causation. No next encoder or mix/seed/rate/budget/head rescue.

V8/V9 share native40,298640 total/294752 fresh learned/3888 fixed capacity,
constructor draw order, unit Huber1 mean plus detached-error Gaussian variance,
shared global norm2 clip1/single AdamW,2048/B8 and fixed D40 heads. V8 has four
original-position local pools; V9 supplies each nonempty bin fixed half local/
half shared whole-history hidden before its same eight export rows and bias.
Empty local bins zero the whole V9 export slice. Changed routing also changes
scales/compute and subsequent gradients/optimizer trajectory despite paired
draws. Downstream global mixing already existed. Shared capacity does not remove
latent-basis, one-seed, trajectory/support or known-population confounds.

## Exact56 numeric roles

One shared train_targets plus20 common roles and four trajectory roles per
version make49. Three temporal roles per version plus one V9between role make56.
Roots: /opt/cuwacunu/research/causal-fresh-{v8,v9}/attempt-001. TRAIN targets:
/cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/train-targets.pt.
No raw cache or held-out tensor values. Whole-file SHA checks of mixed preserved
reports are metadata and do not admit their held-out values.

For each version:

- initial/final_features:112 TRAIN values/valid/row_ids; blocks learned20,
  fixed_spectral12,mean4,uncertainty4.
- initial/final_head/fit_{2701,2802,2903}: existing predictions_train.pt and
  permitted fitted ridge/normalizer/seed fields only; no neural weights/forward,
  refit, ablation predictor or seed selection.
- initial/final_query:first8 fixed TRAIN rawF64/masks/targets/native/mean/variance
  and saved component rows. initial/final_query_summary:mixed JSON decoded,
  explicit train subtree only/full112 TRAIN scalar rows.
- initial/final_checkpoint:typed objective/native/inference metadata and only
  change_decoder[3456,4],variance_projection[27,4],variance_bias[27], link constants.
  Outer/payload V8 policy/partition; V9 adds Double[2]mix[.5,.5]/empty-local policy.
  No whole encoder tensors or optimizer state.
- actual_update_1/2048:two actual pre-step support/component/native/row witnesses,
  plus the same twelve observed temporal arrays in BOTH versions.
- actual_bindings:the two saved hash/update/pre-step associations.
- training_csv:2048 successful rows and seven numeric columns: update,
  Gaussian_NLL_F64,mean_Huber_F64,variance_NLL_F64,total_objective_F64,
  pre_clip_global_norm,clip_factor. No numeric seed/attempt/row-column expansion.
- temporal_learning_receipt:restricted TRAIN-only temporal-projection-learning.json,
  with exact version-specific top-level/nested allowlists.
- temporal_learning_csv:16 columns/8192 update-major rows, four original bins.
- temporal_gauge_csv:6 columns/2048 observations, no data-learning threshold.

V9 also has v9_temporal_between_csv:9 columns/2048 rows: update,
between_bin_gradient_l2,between_bin_movement_l2,contrast_0_gradient,
contrast_1_gradient,contrast_2_gradient,contrast_0_movement,contrast_1_movement,
contrast_2_movement. Its restricted receipt includes declared basis/signs/divisor,
width/rows/events/final-change/scope. expected_fields(identity),
temporal_receipt_fields(version),temporal_receipt_nested(version) and fixed CSV
constants define exact field allowlists; the future card must reproduce them.

Twelve actual fields use stems temporal_export_weight[32,64],
temporal_export_bias[32],temporal_pool_positions[16,1],temporal_pool_score_bias[1],
with suffixes before/after/clipped_gradient. Original FloatStorage operands become
Python F64 BEFORE subtraction/centering. No other encoder weights are read.

## Four bounded panels

1. All112 TRAIN native features/targets/saved predictions: R2/MAE/RMSE,
   covariance/participation/target correlation, saved-ridge block contributions,
   fixed early56/late56. No head/scaler fit, new exported feature or ablation.
2. Stage-matched initial/final first8 residual/variance and local mean4
   Huber1/diagnostic-MAE/diagnostic-mean-NLL adjoints, separate uncertainty4
   variance adjoints and bounded saved-linear residual/variance-link arithmetic.
   No nonlinear parent decoder or parent/encoder Jacobian. Four32-tick raw/error
   bins retain unequal supports: hierarchical bin losses are not additive full
   loss portions and are not age-controlled causal evidence.
3. Both2048 scalar traces/global norms/factors/components, overall/four512 blocks.
   Actual1/2048 distributions/support/CSV associations are checked. Update2048
   forward is state2047; no matched saved pre-step decoder exists, so final2048
   decoder coefficients must never be paired with that actual minibatch.
4. BOTH8192 row/within-bin and2048 gauge observations, fixed512 summaries and two
   actual12-array checks. Registered groups are compared descriptively, without
   counterfactual gradients/norms. V9's2048 signed Hadamard records use the three
   unit R16 directions [++--],[+-+-],[+--+] repeatedfour/divided4. Check Parseval,
   signed telescoping, each direction's gradient/movement events and positive
   overall final projection using actual1 before/actual2048 after. Only those
   two raw gradients are independently reduced; other2046 remain CSV observations.
   No individual final coefficient nonzero requirement.

Seventy-five mandatory tensors have294751 numel capacity, not every-scalar
learning proof. Common pool-score bias is a softmax gauge in BOTH models;
tiny numeric gradients, movement and AdamW decay are not data-learning proof.
V9's three restored relative-bin directions differ from that common gauge.
Initial pairing is native20:40/support only; learned content0:20 and first
mean/variance may differ. Whole40 equality is not required. Mean4/uncertainty4
and cross-model/stage bases are independent: no cross-basis cosine or norm
dominance. Both expose global norms only; no branch split, full-gradient replay
or optimizer independence is inferred. No actual-batch decoder adjoint pairing.

## Later card and admission

Kind cpb_v8_v9_saved_train_diagnosis_card_v1, stage
prospective_before_diagnostic_arithmetic,exact56 IDs/fields, TRAIN112/query8,
four32 bins/four512 blocks. Declare distinct temporal_policy_by_version,
V9_temporal_mix[.5,.5],V9_empty_local_bin_policy, Hadamard basis/signs and
individual_final_coefficient_nonzero_gate=false. Root later supplies exact
context/preservation pins and broad pre_execution_preservation; none is guessed.
Each Stage/Schema/Run requires separate root admission after independent review.

schema.py reads inert descriptors only. diagnose.py imports its owned neutral
reader and standard-library scalar tools, writes exclusive diagnosis.json or
preserved failure.json, then rechecks source/input/preservation hashes.
Output temporal_learning_observations has v8/v9 entries; V9 adds
between_bin_positions. Zero executable model/decoder calls does not mean zero
bounded manual saved-linear/link/finite-difference arithmetic. No next candidate,
new quality, dev/current metadata or Git mutation follows from this diagnosis.
