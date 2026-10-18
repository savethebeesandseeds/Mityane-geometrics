<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE. -->
# Saved TRAIN V6/V7 diagnosis preparation

These five files are complete physical copies of the frozen V5/V6 diagnosis,
then independently adapted here. No historical module imports, Torch/NumPy,
encoder/decoder/neural forwards, head/statistic fits or training occur. Source
preparation authorizes no execution. A separately reviewed prospective card and
existing-container source capsule are still required; no card is frozen here.

The same native40 geometry, fresh constructor, Huber1 mean objective and
detached-error Gaussian variance objective are compared. V6 applies one global
norm2 clip1; V7 clips disjoint mean/uncertainty groups at1 each before one AdamW.
The V7 combined clipped norm may reach sqrt2. Future residual targets, common
data/frozen statistics and optimizer history retain coupling. V7 branch norms
cannot reconstruct V6's unavailable branch split or explain held-out results.

The three panels use saved TRAIN evidence only:

1. Both2048-row CSV traces and first/last actual pre-step component witnesses:
   recompute support, Huber/variance/total/reporting components and each recorded
   norm-factor formula. Report all2048 plus four fixed512-update blocks. V6
   global norms and V7 branch norms retain distinct names. No counterfactual
   global clipping, loss-ratio dominance or gradient replay is computed.
2. All112 saved TRAIN native40 rows, targets and all three already-fitted heads:
   block20/12/mean4/uncertainty4 variance/participation, saved prediction metrics
   and exact saved-ridge block contributions. Neural predictions are read only;
   neural weights/forwards, new fits and block-ablation predictors are forbidden.
3. Initial/final first8 fixed TRAIN query witnesses only: same support, saved
   reconstruction/components and stage-matched decoder-local adjoints. Four
   fixed history bins[0,32),[32,64),[64,96),[96,128) describe supported errors and
   observed rawF64 means/std/within-bin first differences by row/channel/feature.
   Raw summaries receive no market task labels. Their eligible support differs
   from masked-query support; bin hierarchical means cannot be summed as loss.

Local mean-NLL/MAE alternatives are descriptive adjoints in the same mean4 basis;
both actual models use Huber1. Uncertainty4 has an independent basis, so no
cross-basis/model/stage cosine or norm-dominance inference is made. Parent32 and
upstream parameter gradients are unavailable. Only saved initial0/final2048
decoder coefficients are paired with their matching fixed queries. Actual
update2048 was forwarded at state2047: it never receives final2048 coefficients.
The first actual fresh forward is checked equal; later model trajectories differ.

Exactly49 numeric input IDs are required: one shared train_targets plus20 common
roles and four trajectory roles per version. For each v6/v7:

- {version}_{initial,final}_features:112 TRAIN native40 values/valid/row_ids.
- {version}_{initial,final}_query:first8 saved TRAIN raw/masks/targets/native/
  mean/variance/component tensors. Raw must be original DoubleStorage[8,3,128,9].
- {version}_{initial,final}_query_summary:mixed JSON decoded, only train key
  used; both have saved112-row Huber/variance/total components.
- {version}_{initial,final}_checkpoint:typed inference metadata and only saved
  change_decoder[3456,4], variance_projection[27,4], variance_bias[27] and link
  constants. No other parameter/optimizer tensors are read. V7 additionally
  binds typed disjoint clipping policy/caps/norm/partition/names.
- {version}_{initial,final}_{head,fit}_{2701,2802,2903}:predictions_train.pt and
  fit.pt restricted to saved ridge/normalization/seed fields; no neural weights.
- {version}_actual_update_{1,2048}:actual saved pre-step component witnesses.
- {version}_actual_bindings:optimization-component-witnesses.json, exactly
  two forward/update/hash bindings, no new model calls.
- {version}_training_csv:only declared scalar columns across2048 updates.

V6 CSV fields are update,Gaussian_NLL_F64,mean_Huber_F64,variance_NLL_F64,
total_objective_F64,pre_clip_global_norm,clip_factor. V7 replaces the final two
with mean_pre_clip_norm,uncertainty_pre_clip_norm,mean_clip_factor,
uncertainty_clip_factor. No attempt/anchor/seed/gradient-tensor CSV fields are
numerically accessed. expected_fields(id) defines exact per-file selectors;
the future card must copy them without widening. All source/card/allowed-input/
preservation pins are checked before and after arithmetic. Broad preservation
receipt verification happens before any tensor storage access.

Canonical existing-container roots are
/opt/cuwacunu/research/causal-fresh-{v6,v7}/attempt-001. Features/heads come from
quality-001, query/component witnesses and typed checkpoints from run-001.
Shared targets remain /cuwacunu/output/runs/market-capacity-study/
shared-MKT-HIST-2-v1/features/train-targets.pt. No raw cache needs to be read.
The future card declares all49 exact paths/sizes/hashes, per-file fields, the
preservation-only pin list/count and frozen final V7 preservation receipt.

Required prospective card identity is cpb_v6_v7_saved_train_diagnosis_card_v1,
stage prospective_before_diagnostic_arithmetic, versions[v6,v7], TRAIN112,
query8, native40 each. It must declare temporal_bins as the four pairs above,
V6_split_norm_inference='forbidden', full112_component_rows='saved-both-versions',
actual_batch_tensors='saved-both-versions-no-replay',actual_batches[1,2048] and
actual_batch_decoder_adjoints='forbidden-no-matched-prestep-decoder-weights'.
Other inherited support/derivative/mask/seed restrictions remain in Inputs.

After independent source/card preflight and explicit root admission only:

    python3 schema.py /ABS/SOURCE/CAUSAL_V6_V7_TRAIN_DIAGNOSIS_CARD.json
    python3 diagnose.py --card /ABS/SOURCE/CAUSAL_V6_V7_TRAIN_DIAGNOSIS_CARD.json \
      --output /ABS/NEW_DIAGNOSIS_LEAF

schema.py reads allowlisted tensor metadata without storage values. diagnose.py
writes exclusive diagnosis.json or preserves failure.json. It declares no new
candidate, coefficient/seed/rate/cap/head/budget rescue or quality floor. Preserve
V3 baseline/application, every V3-V7 strength and old result/source/checkpoint.
No dev change, model/head replay, held-out tensor diagnosis or Git publication.
