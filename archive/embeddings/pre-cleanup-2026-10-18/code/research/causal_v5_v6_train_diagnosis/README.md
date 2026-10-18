<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE. -->
# Saved TRAIN V5/V6 diagnosis

Prospective protocol: doc/CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.json and .md.
These five files began as complete physical copies of the neutral V4/V5
diagnosis, then received owned V5/V6 adaptations. No historical imports,
Torch/NumPy, learned-module execution or fitting occurs.

Exactly45 input IDs comprise41 common saved TRAIN roles, two actual V6 batch
archives, their binding JSON and one CSV restricted to seven numeric columns
across all2048 successful updates.
All allowed/preservation/source/card pins must match before/after arithmetic.
The final V6 preservation receipt is mandatory. Only the train subtree of four
mixed-split query JSONs is accessed. Per-file tensor selectors exclude held-out
tensors, neural weights, raw cache and arbitrary model/optimizer parameters.

Both versions have native40: learned20, fixed spectral12, mean4, uncertainty4.
All112 TRAIN feature/target/head-prediction rows and three saved repetitions
remain descriptive, with existing-ridge coefficient reconstruction and block
contributions, covariance, participation ratio and fixed early56/late56 halves.
There is no new predictor, head fit, neural forward or block-ablation selection.

Both first8 fixed-query witnesses admit cellwise errors and saved-decoder local
adjoints. V5 applies joint Gaussian NLL to its mean and independent uncertainty
branch. V6 applies fixed Huber(delta1) to mean and Gaussian variance calibration
on detached error; its reporting Gaussian NLL never backpropagates. Diagnostic
NLL/Huber1/MAE vectors are compared only inside the same version/stage mean4
basis. Independent uncertainty4 bases have no cross-basis cosine/conflict
comparison; all native derivative norms are scale-dependent. No parent32 or
upstream parameter Jacobian is reconstructed. Each saved F32 mean and target is
promoted separately to F64 before subtraction. Saved linear mean4 residual and
approximate parent remainder describe allocation, not a new ablation predictor.

V5 full112 query JSON has only MAE/NLL/support. V5 Huber is hypothetical on its
first8 distributions only. V6 adds saved full112 Huber/variance/total components.
These asymmetric scopes are labelled explicitly and never enlarged by replay.

Only V6 saved actual pre-step update1/2048 batch tensors. Their supported losses,
mask/context rules, CSV update/component association and observed clip arithmetic
are recomputed. No actual-batch decoder adjoints are calculated: the final
checkpoint is state2048 while the last witness is a forward from state2047.
The two different batches are not a matched query trajectory; observed global
norm is not independently reconstructed. Shared clip1/AdamW and future residual
targets preserve optimizer coupling despite direct route separation.

The CSV contributes only update, mean_Huber_F64, variance_NLL_F64,
total_objective_F64, Gaussian_NLL_F64, pre_clip_global_norm and clip_factor.
For all2048 and four fixed512-update blocks, report observed clipping frequency
and scalar min/mean/median/max; independently check the recorded factor formula.
No attempt, anchor, seed or gradient-tensor column is numerically accessed.
Loss scalar sizes or ratios do not establish objective gradient dominance, and
the observed norm is not a reconstruction of a full parameter gradient.
The prior seven-file draft is preserved before this prospective CSV revision.

After independent preflight and root authorization, existing cuwacunu only:

    python3 schema.py /ABS/SOURCE/CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.json
    python3 diagnose.py --card /ABS/SOURCE/CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.json \
      --output /ABS/NEW_DIAGNOSIS_LEAF

schema.py inspects allowlisted archive metadata without storage values; it is
also held until admission. diagnose.py writes exclusive diagnosis.json or a
preserved failure.json. No numeric execution occurred during source preparation.

No next candidate, coefficient/seed/rate/head/budget rescue or quality floor is
chosen here. Preserve V3 baseline/application and all V3/V4/V5/V6 scoped gains.
Future strongest-final22 retention is separately declared/reviewed and supplies
preservation context before any candidate quality; no TRAIN NLL gate. No dev
changes, publication or Git action.
