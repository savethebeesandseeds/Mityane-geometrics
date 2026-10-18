# CPB-v11 owned fresh causal research

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited notices remain in copied dependencies and LICENSES/NOTICE.

Source preparation only. No AST, build, test, model, training, extraction,
head fitting or quality execution is admitted by these files. The separately
reviewed prospective declaration and immutable capsule are required first.
There are31 complete owned research files:20 evaluation files, ten owned
non-evaluation sources (including this provenance and objective fixture), and
one separately owned temporal fixture. No older implementation is imported.

## One architectural hypothesis

The scientific recipe parent is V9 SOURCE ONLY. Every learned value, optimizer
moment and permitted TRAIN statistic is fresh. Eight saved historical controls
V3 through V10 remain protected; the singular scoped_reference V10 means latest
measured control, never best, winner, learned ancestry or pairing parent.

Each original bin retains V9's normalized half-local/half-global attention.
The existing two-sum mean feeds its first4 export rows; componentwise centered
variance under that actual mixture feeds its last4 rows. Attention and mean
remain live in the variance derivative. The same32x64 weight and32 bias retain
registration, shapes and constructor draw order. Empty original bins zero their
whole slice after bias. Four8-row slices feed the unchanged global99->64->32 and
shape20 projection. Centered variance here is an INTERMEDIATE pooling feature,
not a new native coordinate or the decoder conditional-variance branch.

Native40 stays content20+fixed spectral12+mean4+uncertainty4. Total298640 values
are294752 fresh learned values in76 tensors plus3888 analytic fixed values.
Same capacity reassigns half the intermediate mean rows, with information,
scale and trajectory confounds. No diagnosis or validation establishes cause.

Exact current temporal metadata at inference root/payload and TRAIN
root/encoder/encoder.payload:

- temporal_pooling_policy: original-position-four-bin-half-mixture-mean4-centered-variance4-v1
- temporal_pooling_partition_v1: Long[5]=[4,16,32,4,8]
- temporal_pooling_mix_f64: Double[2]=[0.5,0.5]
- empty_local_bin_policy: zero-whole-export-slice-after-bias-v1
- temporal_moment_policy: componentwise-centered-variance-two-sum-mean-live-attention-v1
- temporal_export_layout: per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1
- temporal_moment_partition_v1: Long[2]=[4,4]

JSON uses temporal_pooling_partition, temporal_pooling_mix and
 temporal_moment_partition without typed suffixes. Strict wrong/missing/value,
dtype/shape and old-version loads are rejected at all five nodes.

## Restored V9 objective and unchanged recipe

R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v+log(v))], unit coefficients.
Supported mean/target/variance operands are sanitized and separately promoted
before F64 arithmetic and the same eligible-channel/example reductions.
Only objective.total_loss backpropagates; objective.mean_huber and
objective.variance_nll are component traces. trace.nll is the detached diagnostic
copy. JSON/archives use variance_NLL_F64 and row_variance_nll, never V10's joint
Gaussian labels or live-mean/36-FD claims.

objective_policy: huber1-mean-plus-detached-error-gaussian-variance-v1

gradient_policy: huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1

direct_gradient_routes: mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1

variance_error_detached is Long1/JSONtrue; reporting_nll_backpropagated Long0/
JSONfalse. Fresh constructor JSON omits only gradient_policy. All other common
objective/direct-route/temporal fields remain mandatory. The same global norm2
clip1 and AdamW couple scaling and future residuals; no optimizer independence.

Same constructor943001, row943002, mask943003, context943004; fresh4096 TRAIN
calibration anchors/512 loads, batch8,2048 successful updates with at most4096
attempts, mask25/context15/dropout0, AdamW .001/decay .0001, resume512 exact.
Raw loads=512+actual attempts+36 because data loading precedes eligibility skip.
No old model/normalizer is opened, no update replay and no label supervision.
D40 fixed heads, three repetitions and all22 literal221f criteria are unchanged.
Initial pairing uses eight saved V9 DATA archives at20:40 only plus five whole
fresh-statistic archives. New own initial heads are fitted; no old encoder/head
rerun. Historical V10 full40 semantics remain historical.

## Actual learning evidence

75 mandatory identifiable tensors require nonzero CUDA data-gradient/movement
events. Their294751 numel capacity is not a per-scalar proof. The common
pool_score.bias scalar is registered/optimized/observed but is the sole
nonzero-data-gradient exemption; weight decay never proves data learning.

Retain2048 scalar CSV rows,8192 whole-export/within-position rows,2048 gauge rows,
2048 Hadamard between-bin rows and add16384 moment-subgroup rows. The new
 temporal-moment-subgroup-learning.csv columns are update,bin,moment,
export_row_begin,export_row_end,weight_gradient_l2,bias_gradient_l2,
weight_movement_l2,bias_movement_l2,nonzero_export_gradient_values,
moved_export_values. moment is mean or centered_variance, each four rows.
Every subgroup needs at least one weight AND bias data-gradient/movement event
and final weight/bias changes, never every scalar or every update. The eight
subgroups exactly partition the four existing8-row groups. Receipt adds
moment_subgroup_rows=16384 and moment_subgroups[8]; summary flag is
all_eight_temporal_moment_subgroups_learned. Engineering groups are
 temporal_pool_mean_bin_0..3 and temporal_pool_centered_variance_bin_0..3.

Actual update1/2048 archives have56 fields:23 base+12 parameter before/after/
postclip-gradient arrays+21 same-forward temporal trace fields. The17 ordinary
trace fields save original positions/support/rows, two hidden paths, local/
global/mixture weights, bin validity, means and centered variances. Two retained
projection-output upstreams are PRECLIP nonleaf gradients; two Long scalar
flags mark defined routes, with zero fallback for unused routes. Independent
local adjoints sum both routes and apply recorded clip_factor ONCE before
tight comparison with postclip export weight/bias and pool_positions
arrays: abs1e-9+rel3e-4 per coordinate. At each actual witness the reconstructed postclip export weight OR bias maximum must exceed1e-6. Common score-bias actual/reference/error is finite descriptive evidence only, with no tight parity or nonzero gate. Saved scalar moments retain abs/rel3e-5. No whole-encoder adjoint or global-norm reconstruction is claimed.
There is one existing forward, no extra encode/replay or additional loss.

The objective fixture is the complete V9 partial-gradient source copy under V11.
The separately owned temporal fixture proves full-attention moments/adjoints,
F64 and actualF32 finite differences, empty/constant/equal-mean-spread controls,
common gauge, relative-bin sensitivity and actual subgroup learning.

Source-only preparation preserves all old scientific source and results. V3
remains the baseline and sole DEV application encoder; no selection/publication.
