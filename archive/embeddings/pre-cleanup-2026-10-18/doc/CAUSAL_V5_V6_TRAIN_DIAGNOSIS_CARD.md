<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Prospective saved TRAIN V5/V6 diagnosis

This declaration precedes diagnostic arithmetic. No new experiment, encoder,
quality measurement, head fit or next candidate is declared. The machine
authority is [the exact JSON card](CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.json).
It implements the scope in the [planning handoff](CAUSAL_V5_V6_TRAIN_DIAGNOSIS_PLAN.md).
All prior cards, sources, results, saved heads/checkpoints and failed attempts
remain unchanged. V3 remains the clean baseline and sole dev encoder.

Dataset: **MKT-HIST-2** · retrospective market targets · **complexity4/5** ·
TRAIN only through1705881599999. Both prior trajectories are fresh2048/B8 CUDA;
this diagnosis uses only saved data and has no model/GPU/training execution.

The two owned native40 versions share20 learned content,12 literal spectral,
4 mean and4 uncertainty coordinates. V5 applies joint Gaussian NLL. V6 applies
unweighted Huber(delta1) mean plus detached-error Gaussian variance calibration,
unit coefficients, with shared globalclip1/AdamW. Reporting Gaussian MAE/NLL
are unchanged; direct route separation does not make optimization independent.

The45 pinned numeric/evidence inputs comprise41 common roles (initial/final
TRAIN features, query distributions/summaries, decoder checkpoints, three saved
head predictions and coefficient archives per stage/version, shared targets),
two V6 actual batch archives, their binding JSON and V6 CSV. The CSV is limited
to seven numeric columns for all2048 successful updates. Every archive has
an exact independent tensor-field allowlist. Verify all45 input pins, declared
preservation pins and final V6 broad preservation before numeric access, then
recheck after. Root source-stage binds all five owned files plus both cards.

Planned saved-data arithmetic:

- All112 natural-mask TRAIN rows: four-task R²/physical MAE/RMSE from saved ridge
  and all three neural predictions; chronological early56/late56 without refits.
  Reconstruct the existing ridge using its original normalization/coefficients.
  Describe four-block coordinates, covariance/participation ratio and existing
  ridge contributions/covariances. No new predictor or neural-head forward.
- Full112 TRAIN query MAE/NLL/support rows and halves. Four aggregate JSON files
  contain both splits; only their train numeric subtree is accessed. Only V6
  supplies full112 Huber/variance/total rows. V5 lacks these fields; do not infer
  them from aggregate MAE/NLL.
- First8 TRAIN fixed-query distributions only: support/timestamps, channel-feature
  errors, variance/inverse-variance pressure and bounded local adjoints from
  stage-matched saved decoder coefficients. V5 Huber1 is hypothetical/unapplied;
  V6 mean-NLL/MAE adjoints are diagnostic alternatives. Compare NLL/Huber1/MAE
  only within each same mean4 basis. Both live uncertainty4 branches stay
  separate; no mean4/uncertainty4 or cross-model/stage cosine/conflict inference.
- Saved linear mean4 residual allocation and approximate total-minus-residual
  parent remainder, with F32 rounding limits. No nonlinear parent decoder call,
  content20/spectral12 decoder split or counterfactual block predictor.
- V6-only actual update1/2048 pre-step batch distributions: recompute supported
  Huber/variance/total/reporting losses and rows, exact target/context/visible
  masks, TRAIN timestamps, binding/CSV update identity and observed clip-factor
  arithmetic. V5 has no matching tensor
  evidence. The two batches differ and do not establish a trajectory curve.
- V6 TRAIN trajectory scalars only: update, mean_Huber_F64, variance_NLL_F64,
  total_objective_F64, Gaussian_NLL_F64, pre_clip_global_norm and clip_factor for
  all2048 successful updates. Check update order, finite/range/unit-component
  identities and each recorded factor against min(1,1/(observed_norm+1e-6)).
  Report overall and four fixed512-update blocks: observed clipping frequency
  (recorded factor below1) and component/norm/factor minimum, mean, median and
  maximum. No other CSV column is numerically accessed. Loss-scalar sizes or
  ratios do not establish objective gradient dominance or explain causation;
  a recorded global norm is not a reconstructed full gradient.

The last actual witness comes from pre-step state2047; the final decoder is
post-step2048. **No actual-batch decoder adjoint** is reconstructed with unmatched
weights. Observed full gradient norm is not independently reconstructed. Query
local adjoints omit parent32/upstream parameter Jacobians and optimizer state.
Scalar algebra separately promotes each F32 mean/target operand to F64 before
subtraction and follows the declared supported-cell/channel/example hierarchy;
variance-link differentiation approximates saved F32 intermediates, not CUDA
gradient bit parity. Raw latent adjoint norms depend on scale/parameterization.

No raw-cache numeric access, learned-module loading, neural weights/forward,
fit, calibration, Torch/NumPy, training replay, held-out numeric tensor access,
head/quality rerun, PCA or post-encoder transform is admitted. Metadata/hash
preservation of broader files does not authorize reading held-out numeric fields.
Any mismatch writes a new exclusive failure receipt and stops without fallback.

The exact once-only arithmetic command, after root and independent preflight:

    python3 diagnose.py --card /ABS/SOURCE/CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.json \
      --output /ABS/NEW_DIAGNOSIS_LEAF

Use only existing managed cuwacunu, with separate exclusive source/output leaves.
Source preparation and card declaration have not executed numerical diagnosis.
The preceding five-source/two-card draft is preserved byte exact under
output/engineering/causal-v5-v6-train-diagnosis/attempt-001/before-trajectory-scalar-scope
with its manifest. This additive revision admits more already saved TRAIN
scalar evidence prospectively; it adds no file, tensor stage, model, head fit,
held-out access, candidate or rescue after results.
An independent saved-evidence audit must follow before a human interpretation.
Known TRAIN associations cannot establish causation, forecasting or held-out
improvement. Choose no next mechanism from latent norms or an unmatched batch.

Protect every V3/V4/V5/V6 final quality scope, validation reconstruction and V6
TRAIN MAE/descriptive NLL gain, including displaced whole historical records.
The47fa… retention protocol is preserved parent context. The separately reviewed
strongest V3–V6 future protocol0965… is additional preservation context; this
diagnosis neither derives its floors nor measures a new candidate against them.
Keep the same22 criteria/trust/coverage/positivity and no TRAIN NLL gate.
Read [V6 review](CAUSAL_FRESH_V6_REVIEW.md),
[ledger](clean_causal_encoder_research_ledger.json) and
[final establishment](causal_fresh_v6/final-establishment.json).
