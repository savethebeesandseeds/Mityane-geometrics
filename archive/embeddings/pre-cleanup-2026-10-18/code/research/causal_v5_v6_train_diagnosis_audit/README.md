<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../LICENSE. -->
# Independent saved TRAIN diagnosis audit

This directory owns complete physical copies of the prior neutral TRAIN
arithmetic auditor and inert archive reader, plus an independent owned scalar
component/trajectory helper. Execution requires the exact frozen main-card hash,
diagnosis hash and independent audit source declaration, in an exclusive
capsule admitted by root inside the existing managed container.

The independent review distinguishes these evidence scopes:

- Natural-mask TRAIN112 features, targets and saved fixed-head predictions:
  all four tasks, three repetitions, fixed early56/late56 partitions and the four
  native40 blocks. Saved ridge coefficients may reconstruct their existing
  predictions and contributions; no neural forward, refit or block ablation.
- Fixed masked-query TRAIN112 row scalars: ordinary MAE/NLL in both versions;
  Huber/variance/total component rows only where V6 actually saved them. V5 has
  no full112 Huber/component rows to reconstruct from its aggregate losses.
- The first8 saved query distributions per initial/final stage: supported F64
  kernels, error allocation and matched-stage bounded decoder adjoints. V5's
  applied mean path is Gaussian NLL; V6's applied mean path is Huber1. Gaussian
  mean-NLL sensitivity on V6 is hypothetical, not detached-report autograd.
- Two V6 actual pre-step training batches, updates1 and2048: masks, support,
  components, recorded CSV association and observed clipping-factor arithmetic.
  They are different random batches. State2047 decoder weights are unavailable,
  so final2048 weights must not reconstruct update2048 pre-step adjoints.
- All2048 recorded V6 TRAIN CSV steps: only update and six declared observed
  component/reporting/norm/factor columns. Fixed overall/four512-step summaries
  and clip-factor formula checks do not reconstruct full parameter gradients or
  infer branch gradient dominance from scalar loss magnitudes.

The allowlist is the common41 roles plus two V6 component archives, their binding
JSON and the TRAIN CSV: exactly45 files. Selectors and pins must agree with the
frozen schema. Training-summary JSON is hash-only
preservation evidence; aggregate query JSON exposes only its TRAIN subtree.
No held-out tensors, TEMPO data, extra checkpoint stages, old model replay,
Torch/NumPy, fitting, calibration or encoder/parameter Jacobian is authorized.

Any vector cosine is confined to one saved mean4 basis. Mean4 and uncertainty4
are unrelated bases, and trained bases across versions/stages are not compared
as if their coordinates were aligned. Approximate F64 scalar adjoints disclose
the original F32 forward-rounding boundary. Shared global clip1/AdamW remains a
coupled optimizer; its observed norm cannot be independently recomputed without
the unsaved full parameter-gradient arrays.

All old sources, cards, results, failed attempts and baseline checkpoints remain
immutable. A later additive audit result supplies evidence for the TRAIN
diagnosis; it neither changes historical quality criteria nor selects a new
candidate, coefficient, head recipe or application encoder.

Run only after the source/card capsule is frozen and admitted:

    python3 audit.py --card /ABS/MAIN_CARD.json --diagnosis /ABS/diagnosis.json \
      --audit-card /ABS/independent-audit-card.json --output /ABS/NEW/audit.json

The scalar finite-difference plan retains256 checks: the first16 supported cells
in canonical b,c,f,h traversal for each of four fixed-query model/stage pairs;
Gaussian mean/variance, MAE and Huber1 kernels use the inherited positive step
rules. Scalar Gaussian finite differences are diagnostic kernels and do not
claim that V6's detached reporting NLL backpropagates.
