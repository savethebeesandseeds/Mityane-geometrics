# CPB-v10: Huber1 plus one live joint Gaussian objective

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
Preserve inherited notices in every copied file and LICENSES/NOTICE.

SOURCE PREPARATION ONLY. No AST, build, fixture, encoder, CUDA, training,
head or quality execution has occurred. No frozen prospective scientific card
or canonical version descriptor is created by this source preparation. CPB-v3
remains the sole dev encoder. Every V3–V9 result/source/checkpoint stays intact.

This family owns all50 physical files copied from versions/v9/, with complete
version-local namespace/include/kind adaptations, plus the complete external
versions/v10.h convenience copy (51 assets). The convenience header is byte
identical to include/embedding/encoders/causal_patch_bottleneck_mae/versions/v10.h.
There is no old implementation include, forwarding shim, learned-weight import,
optimizer import or historical-statistic initializer. Source ancestry is not
learned-weight ancestry: every learned tensor and TRAIN statistic starts fresh.

The ONE scientific change is training gradient routing. With e=mean-target,
G=.5*(e*e/variance+log(variance)), and H=Huber(delta1), the objective is:

    total = R[H(e)] + R[G(mean,variance)]

Exactly one Gaussian term has BOTH predicted mean and variance live. It replaces
V9's detached-mean Gaussian term; no second detached-error term remains. Both
coefficients are1. The target is detached and unsupported operands are sanitized
before individual F64 promotion and subtraction. The unchanged reduction averages
supported cells within channel, eligible channels within example and eligible
examples. Huber is nonnegative; Gaussian and total may be negative. No log2pi
constant, rescue clamp, weighting search, new head or new data is introduced.

Per supported cell (before reduction), mean derivative is clip(e,-1,1)+e/v;
variance derivative is .5*(1/v-e*e/(v*v)), ONCE. At identical saved distribution
values the total scalar equals V9's total; the graph differs. Mean derivatives
reach parent32/backbone and mean4/residual paths. Variance reads only live
uncertainty4, whose independent projection reads the fixed TRAIN-standardized
bank270; it has no direct path to the learned temporal backbone or mean4.
Shared global clipping, one AdamW/history and future residual values still
couple optimization. No optimizer-independence or causal validation claim follows.

Public policy strings are:
- objective: huber1-mean-plus-joint-gaussian-nll-v1;
- gradient: huber1-mean-live;joint-gaussian-mean-live-and-uncertainty4-live;shared-global-clip1-v1;
- direct routes: Huber->parent32+mean4;Gaussian-mean->parent32+mean4;Gaussian-variance->uncertainty4;mean-reads0:36;variance-reads36:40;shared-global-clip1.

Training trace exposes objective.mean_huber, objective.joint_gaussian_nll and
objective.total_loss. Only total_loss is backpropagated. trace.nll remains an
ordinary reporting Gaussian COPY with BOTH distribution tensors detached.
Research component labels must be joint_Gaussian_NLL_F64/row_joint_gaussian_nll,
not variance-only labels. Coefficient order is [mean_huber,joint_gaussian_nll].

Inference root/payload and TRAIN root/encoder/encoder.payload strictly bind
objective_policy and direct_gradient_routes as UTF8 Byte vectors,
mean_huber_delta_f64 Double scalar1, loss_coefficients_f64 Double[2]=[1,1],
variance_error_detached Long scalar0, reporting_nll_backpropagated Long scalar0,
and the new gradient policy/kinds. Missing, wrong, old V9 routing, wrong dtype
or wrong shape must refuse. Typed model/optimizer hashes, moments, parameter
order, counters, seeds, recipe and dataset/stat identities remain enforced.
Fixture resume is a declared engineering witness; main resume512 must continue
one actual trajectory after exact state restoration, without replay claims.

Inference/forward capacity, parameter registration/draw order and numerical
computation are unchanged from V9. Native40 is20 learned content+12 literal
spectral+mean4+uncertainty4;76 learned tensors/294752 fresh scalar capacity,
fixed1/3888, total77/298640. Exactly pool_score.bias[1] remains the registered
common-softmax gauge, separately observed. The other75 identifiable tensors
cover294751 capacity and require tensor-level data-gradient/movement evidence,
not a claim about every scalar. Existing four export row/local-position groups
and three between-bin direction event requirements remain.

Both ordinary and early-contextual routes retain the SAME owned half-global/
half-local kernel. For each NONEMPTY anchored original bin, average its local
softmax hidden and the shared valid-history softmax hidden at1/2 each, then
apply that bin's existing8 export rows/bias ONCE. An empty local bin keeps the
whole zero8 slice after bias. Policy remains
original-position-four-bin-shared-global-half-mixture-v1, partition
[4,16,32,4,8], mix Double[.5,.5], empty policy
zero-whole-export-slice-after-bias-v1. Truthful local/global/per-bin-mixture
traces and aggregate SUM mass=#nonempty bins are preserved.

Each slice describes the CURRENT cutoff endpoint, not its bin's earlier endpoint.
Shared context contains only retained times<=t; cutoff masking, causal attention,
backward relation banks and all128/2350/30 planned invariance proofs stay intact.
The downstream global network and all inference/distribution operations are
unchanged. Fresh same-seed full native40 initial DATA and five-statistic pairing
against SAVED V9 evidence is appropriate; no V9 model or weight/stat load is used.

The owned objective fixture uses artificial TRAIN-only statistics, nonzero
residual/variance-head values, unequal support, nonzero errors and unequal
variances. Planned checks include isolated Gaussian mean/variance routes,
Huber+ONE Gaussian adjoint addition, exact-once variance-head gradients,
manual full-native40 and mean4/uncertainty4 adjoints, unsupported NaN sanitation,
and36 independent F64 scalar finite differences away from Huber/link boundaries.
Finite differences validate the attached scalar objective, not detached report
NLL autograd. One synthetic AdamW step checks five selected projection/head
parameter tensors; no full encoder forward or head fit is called. F32 manual
native gradient tolerance remains2e-5 relative/2e-7 absolute, local coordinate
oracle2e-4/2e-6; scalar finite differences2e-8*(1+abs(reference)) with step1e-5.
No runtime pass is claimed. Temporal fixture math is inherited unchanged except
version paths/namespaces; the main fixture adds strict five-node objective
negative cases while preserving cutoff and resume witnesses.

This is a separately proposed objective mechanism. Prospective source/card,
strongest-V3–V9 retention221f…/22 conditions, fixed heads/seeds/budget and independent
admission must freeze before any execution. No TRAIN-NLL gate, winner, dev
replacement, Git publication, cleanup or historical rewrite is authorized here.
