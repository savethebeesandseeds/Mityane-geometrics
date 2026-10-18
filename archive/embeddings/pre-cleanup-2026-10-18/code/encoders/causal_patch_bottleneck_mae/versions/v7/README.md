# CPB-v7: disjoint mean and uncertainty clipping

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited notices remain in all complete copied files and LICENSES/NOTICE.

Prospective owned source only. Build, causal/CUDA proof, typed resume and quality
have not been executed or claimed. CPB-v3 remains baseline/application; every
V3/V4/V5/V6 result, scoped leader, source and checkpoint remains preserved.

This version owns a complete physical V6 family copy under V7 namespaces,
include paths, settings and checkpoint kinds. The external versions/v7.h is a
full byte-identical copy of the internal public header. No forwarding, old model
initializer, checkpoint import or fitted-statistic import is provided.

Forward/native40, learned/fixed capacity, module/parameter registration,
constructor draw order, raw F64 causal boundary, mean/variance decoders,
preprocessing and objective are unchanged from V6. All294752 learned values
start fresh. Native40 remains parent32+mean4+uncertainty4;3888 analytic prior
values stay fixed. The objective is still unit Huber1 mean plus detached-error
Gaussian variance NLL with the same F64 support/hierarchical reduction. Reporting
Gaussian NLL never receives backward.

The only optimization change partitions actual named parameters into:

- Mean:73 learned tensors/293537 values, including the parent encoder/decoder
  and mean4 projection/residual decoder.
- Uncertainty:3 learned tensors/1215 values: core.variance_projection,
  core.variance_bias and core.uncertainty_projection.
- Fixed:1 literal prior tensor/3888 values, excluded from clipping and AdamW.

training::parameter_partition returns the actual tensors, canonical full names
and value counts. It checks exact counts, non-aliasing, all three uncertainty
names and complete learned coverage. training::clip_gradients validates both
branches before mutation, applies norm2 clip1 independently to each and returns
both pre-clip norms/factors plus membership. Undefined gradients remain valid
for currently unused parameters. One unchanged single-group AdamW step follows;
the optimizer, rates, decay, objective and training budget are not changed.

The maximum combined clipped norm can approach sqrt2, versus V6's joint bound1.
Any quality change therefore does not isolate coupling removal from this changed
joint norm budget. Current clipping cannot let one branch's norm attenuate the
other branch's gradients, but future mean residuals still alter variance targets;
the input data/frozen statistics and optimizer history remain part of the common
training process. Complete optimizer/statistical independence is not claimed.

New inference/payload/training kinds strictly bind gradient_clip_policy,
gradient_clip_max_norms_f64 Double[1,1], gradient_clip_norm_type_f64 Double2,
gradient_clip_partition_v1 Long[73,293537,3,1215,1,3888] and canonical named
membership. Missing/malformed fields, changed names/counts and old global clip
or old-version policies are rejected. Existing model/moment/order/seed/counter/
dataset/split/recipe resume bindings remain.

The owned fixture retains manual composite-objective native40 gradient parity,
direct route checks and one actual CUDA AdamW step. Its additional artificial
nonzero CUDA gradient witness covers all76 actual learned tensors, independent
F64 group-norm/clipping parity, both cross-branch scaling directions, fixed-prior
exclusion and the nonvacuous joint sqrt2 bound. This gradient-only witness is not
an encoder replay or training objective. Whole-family engineering separately
checks genuine reconstruction learning, causal suffix invariance and typed resume.

No head tuning, PCA, task-label supervision, old encoder execution or application
promotion follows. A prospective card and frozen source admission are required
before any numerical execution.
