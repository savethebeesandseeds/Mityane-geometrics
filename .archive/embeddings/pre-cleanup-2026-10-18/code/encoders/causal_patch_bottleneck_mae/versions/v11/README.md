# CPB-v11: four mean rows and four centered-variance rows per temporal bin

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
All inherited copyright/license notices remain in their physical source copies.

SOURCE PREPARATION ONLY. No AST, build, test, CUDA, training, head fit or quality
execution is admitted. CPB-v3.market-01 remains the sole DEV encoder and baseline.
Preserve all V3–V10 scoped leaders, sources, checkpoints, reports and failures.

The 50 files in this directory and external versions/v11.h are complete physical
copies of the FINAL V9 family followed by owned V11 edits. Source ancestry is not
weight ancestry: ALL new learned numeric values, AdamW state and TRAIN statistics
start fresh. No old learned weights, checkpoint statistics or implementation files
are imported. The external header is a full byte-identical copy of the owned
public header. Parent bytes/pins are preserved under the separately pinned
causal-fresh-v11-family-preparation/parent-copies-001 leaf.

Only the intermediate temporal moment/export operator changes. For each nonempty
original-position bin, retain the exact V9 two-sum mean expression:
mu = .5*sum(local_softmax*safe_hidden) + .5*sum(global_softmax*safe_hidden).
Its actual normalized mixture is p = .5*local_softmax + .5*global_softmax.
The new componentwise centered variance is sum(p*(safe_hidden-mu)^2), with no
square root, Bessel correction, extra epsilon, detached mean or detached weights.
The bin's existing first four projection rows/bias read mu; its last four rows/bias
read centered variance. Bias is applied once per row. A missing LOCAL bin keeps
its WHOLE eight-coordinate slice exactly zero AFTER bias, even if global history
exists. All-padding softmax uses a finite fallback before normalization. Hidden
storage outside valid support is sanitized BEFORE subtraction/square/sums;
supported nonfinite hidden/scores are rejected. Both ordinary and contextual
paths call the same owned kernel. Every slice describes the CURRENT cutoff
endpoint; shared history is exclusively valid states at or before that cutoff.

The sixteen centered-variance export coordinates are INTERMEDIATE coordinates
inside the 32-wide per-channel export. They feed the unchanged global MLP/content20
path. They are distinct from native uncertainty4 and the decoder's conditional
variance branch, which are unchanged. Native40 remains content20 + fixed spectral12
+ mean4 + uncertainty4. The V9 unit Huber(delta1) mean objective plus detached-error
Gaussian variance calibration, global clip1, one AdamW, D40 heads, seeds/data and
2048 successful updates at batch8 are unchanged. Parameter shapes/registration and
constructor draws are unchanged: source inspection must derive and bind them;
no reused numeric weight values are permitted. Fresh initial DATA20:40 and five
whole statistic contents are expected to match SAVED V9 evidence, without replay;
no whole40/initial query mean equality is promised.

Typed metadata at inference root/payload and TRAIN root/encoder/encoder.payload:
- temporal_pooling_policy: original-position-four-bin-half-mixture-mean4-centered-variance4-v1.
- temporal_moment_policy: componentwise-centered-variance-two-sum-mean-live-attention-v1.
- temporal_export_layout: per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1.
- temporal_moment_partition_v1: Long[2]=[4,4].
- Existing partition Long[5]=[4,16,32,4,8], mix Double[2]=[.5,.5] and
  empty_local_bin_policy=zero-whole-export-slice-after-bias-v1 remain strict.
Missing/wrong value/dtype/shape and old kinds/policies must refuse. Export width
must equal32 as well as be divisible by8, so every accepted config has the declared
four-mean/four-variance layout. Moment traces expose ordinary/contextual [R,4,64] bin means
and variances. Local/global/per-bin mixture traces remain actual weights;
aggregate weights are a SUM of bin mixtures, with mass=#nonempty, not one probability.
LocationScaleForwardTrace.temporal comes from the ONE existing encode call.
Actual update1/2048 witnesses may save its moments/support/hidden/weights and
retained post-empty export upstream gradients for
independent bounded scalar reconstruction; no added model call is allowed.
Retained nonleaf upstream gradients are PRECLIP: parameter clipping does not
modify them. Undefined unused-route gradients require explicit flags and exact
zero fallbacks; no arbitrary route activity is required. Sum both route local
adjoints and multiply the recorded global clip factor ONCE before comparing
postclip parameter-gradient arrays. No full encoder gradient/global norm is
reconstructed. Archive counts derive from the expanded actual writer, not the
old twelve parameter arrays alone.

Full independent adjoints follow the ACTUAL mean expression. With upstream
vectors a for mean, b for centered variance, delta=h-mu and r=sum(p*delta), the
mean adjoint is a-2*b*r. Add direct hidden contribution 2*p*b*delta and direct
mixture-weight contribution dot(b,delta^2), then propagate both local/global
softmax and score(tanh(hidden)) paths. Never assume r is exactly zero or detach
attention/mean. Variance is centered rather than E[h^2]-E[h]^2. Present constant
hidden may have a small positive variance from F32 rounding; it is NOT an exact
zero gate. Exact zeros are reserved for absent export slices/sanitized empty paths.

The dedicated synthetic fixture prospectively declares manual F32 absolute and
relative tolerance3e-5 for forward AND full hidden/score/export/position adjoints; common-shift tolerance2e-6; bin mass2e-6; aggregate mass8e-6;
present-constant F32 variance bound1e-10*(1+max_abs_hidden^2). Its independent scalar
F64 central differences use step1e-5 and abs/rel2e-8; actual CUDA F32 kernel central
differences use step2e-3 and abs/rel2e-3, covering18 coordinates each (hidden,
projection weights, original-position offsets), with at least three actual reference
FD derivatives of absolute magnitude above the fixed1e-2 noise-floor threshold.
The two references cover all
attention-dependent paths. Equal-mean/different-spread, unsupported NaN, exact
empty slices, both actual backbone paths, each mean4/variance4 projection subgroup,
within-bin contrasts, three unit R16 Hadamard directions and common score-bias
gauge are checked. The source declares one backbone encode (both paths),44 direct
kernel calls, zero public Encoder/head calls and one synthetic AdamW step. These
are prospective fixture scopes, not claimed results. Runtime counters/logs must
confirm them before admission.

75 identifiable tensors require actual data-gradient/movement events; the common
pool_score.bias gauge remains registered in the same optimizer and observed
separately. Tensor value capacity does not prove every scalar moved. Four mean4
and four centered-variance4 row groups must each show weight/bias data-gradient
and movement events plus final change. Existing four within-bin contrasts and
three Hadamard directions retain their event requirements and positive OVERALL
final between-bin projection change; signed final coefficients are descriptive.
The constructed fixture may check stronger per-direction single-step movement
without adding an every-final-coefficient main-trajectory gate. Weight decay is
not data-learning evidence. No validation cause, winner or guaranteed gain is
claimed by this unexecuted architecture hypothesis.
