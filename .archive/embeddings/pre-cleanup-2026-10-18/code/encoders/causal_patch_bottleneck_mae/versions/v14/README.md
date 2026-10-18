# CPB-v14: fixed nested trailing attention windows

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited notices remain in every complete physical source copy.

SOURCE PREPARATION ONLY. No helper, AST, import, compiler, test, model, head fit,
training or quality execution is admitted by these sources. CPB-v3.market-01
remains the clean baseline and sole DEV encoder. Preserve every prior source,
checkpoint, measurement, failed attempt, review and scoped leader.

This family owns50 files, the complete external versions/v14.h, and three full
research fixtures. Complete V13 source parents and canonical V13 descriptor DATA
were preserved before any owned copy/edit under the central V14 parent manifest.
V13 supplies SOURCE only: no learned values, fitted statistics, optimizer state
or checkpoints initialize V14. Both public headers have identical complete
bodies; private includes remain in this owned family plus Torch/STL.

The single scientific change is local temporal membership. H128/P8 retains16
original patch positions. Windows of trailing lengths4,8,12,16 have supports
[12,16),[8,16),[4,16),[0,16), ordered shortest to longest. Membership is valid AND
original_position>=16-length[j] AND original_position<16. Edge16 follows the
retained left-padded history ending at the requested cutoff. Packed length,
valid count, last observed position and future raw-array end never set edges.
Position0 belongs only to the longest window; position15 belongs to all four.
Sparse gaps remain gaps. Parameter/data/native shapes do not change.

Each present window retains the actual V13 attention graph:
p=.5*local_softmax+.5*global_softmax;
mu=.5*sum(local*safe_hidden)+.5*sum(global*safe_hidden);
v=sum(p*(safe_hidden-mu)^2);
root=sqrt(v+0.0009765625); s=root-0.03125.
Root calculation is actual F32 add, sqrt, then separate subtraction. The first4
existing projection rows read mean; the last4 read s. Globalhalf uses all valid
retained causal states, so slices are horizon emphases for the same cutoff, not
isolated horizon-only information. Length16 local/global supports coincide, but
the actual half-plus-half two-sum expression/graph remains unsimplified.

Tau2^-10, offset1/32, raw variance, full live hidden/center/attention score/position
routes and finite-F32 center-residual correction remain unchanged. For mean
upstream a and dispersion upstream c, beta=c/(2*ACTUAL saved root),
residual=sum(p*(h-mu)), center upstream=a-2*beta*residual. Mean coefficients follow
two separate sums, not a rounded stored mixture. No uniform weights, detach,
scale, Bessel correction, new parameters/buffers/RNG draws or sweep is added.
Intermediate dispersion16, native uncertainty4 and decoder variance are distinct.

Both hidden and broadcast-center operands are sanitized BEFORE subtraction and
square, including invalid tokens inside present windows. Supported nonfinite
hidden/scores and nonfinite/negative/overflowed raw variance are refused.
An absent local window has mixture0 and whole8 served rows0 AFTER bias even with
global history. All-padding softmax has a finite fallback. Raw zero-variance
roots intentionally remain1/32 for absent/all-padding windows; dispersion is0.

The actual-step archive stays60 fields/25 temporal fields, with TWO explicit
instrumentation changes: pool_local_weights and contextual_pool_local_weights
(archive temporal_* fields) now hold exact SAME-FORWARD per-window locals[R,4,T],
replacing V13's disjoint SUM[R,T]. No mixture inversion is used for adjoints.
Global weights remain[R,T]. pool_weights/contextual_pool_weights remain SUMs of
mixtures: mass=#present windows, not one probability. Historical V13 archives
remain untouched and incompatible with this new local-weight shape. No new
encoder forward/capture replay is introduced.

All five checkpoint nodes enforce new pooling String
original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1;
temporal_pooling_geometry_v1 Long[4]=[4,16,32,8];
temporal_pooling_trailing_lengths_v1 Long[4]=[4,8,12,16];
layout String
per-trailing-window-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1.
Nodes are inference root/payload and TRAIN root/encoder/encoder.payload. Old
temporal_pooling_partition_v1 is neither written nor accepted, even as an extra
field beside correct new metadata. Missing/wrong dtype/rank/length/value/order,
swapped lengths, parent policy/layout/artifact and legacy conflicts are refused.
The transform String and Float64[1] tau remain strict at all five nodes.
Own fresh typed resume retains full model/optimizer/counter/RNG parity proofs.

The independent scalar reference uses fixed original edge16 and evaluates the
full attention/moment/root/projection/score graph on both routes. Captured F32
adjoints use actual per-window local/global/mixture/mean/variance/root fields,
never observed gradients or an inverse of rounded mixture. The primary oracle
checks fixed-position membership, exact local outside zeros, mixture mass,
moments and transforms at all cutoffs. No full encoder/global-norm reconstruction
is inferred from bounded witnesses.

Temporal proof SOURCE contract:53 direct kernel calls=36 F32 FD evaluations
+14 ordinary successes+3 supported numerical refusals;18 independent F64 FD
coordinates,18 actual F32 FD coordinates,4 isolated autograd::grad calls,
1 actual backbone encode,0 public encoder calls,0 heads,1 synthetic AdamW step.
Cases cover original0/15 and sparse boundary positions, longest local/global
equality, empty-local/nonempty-global, all-padding, invalid NaN/Inf padding,
packed-length invariance, huge singleton, zero/tiny/large dispersion and supported
overflow refusal. Both full temporal fixture copies are byte-identical.

Thirteen prospective flags comprise the nine V13 checks plus
nested_trailing_supports_fixed_edge16, per_window_local_weights_captured,
longest_local_equals_global_two_sum and packed_padding_membership_invariance.
Manual F32 abs/relative tolerance3e-5; F64 FD step1e-5/tolerance2e-8;
F32 FD step2e-3/tolerance2e-3; gauge2e-6; present-constant raw variance bound
1e-10*(1+max_abs_hidden_squared); at least3 FD reference derivatives>1e-2;
actual-root analytic slope bound16. These are declarations, not executed passes.
Original four-position-quarter/Hadamard diagnostics remain a fixed R16 basis,
distinct from overlapping local supports; export groups correspond to windows.

Research primary declares821 negative CASES=33 initial+10*(21 objective rows
+55 temporal rows)+19 late archive cases+4 direct loader requests+5 legacy-extra
cases, writing817 full invalid archives. Family primary declares676 cases
=33+10*(6+56)+14+4+5, writing672 full invalid archives. JSON distinguishes cases
from files. Compact archives/restored-target compact controls both count0. The
optional compact path was not implemented: intended-validator and restored-target
payload-boundary controls would require a separate engineering change.
Full positive roundtrips, all five real checkpoints, late identity/nested parity/
optimizer cases, causal matrices and exact resume remain. Legacy-extra cases
assert the specific legacy-field refusal. Live model/scaler/bank/optimizer
validation follows the negative matrix.

Primary fixtures retain128-cutoff future VALUE and MASK invariance, left-padding,
longer histories, original timestamps, internal causal attention and past-positive
controls. Research separation/family variance-gradient fixtures retain objective
and decoder routing proofs. Registration/draw order and expected76 learned
tensors/294752 values+3888 fixed values/native40 must be proved later.

Future science remains one fresh2048-update/B8 CUDA trajectory with seeds
943001/943002/943003/943004, shared globalclip1, one AdamW and unchanged unit
Huber1 mean plus detached-error Gaussian variance objective and report-only NLL.
Initial comparison uses eight V13 DATA tails20:40 and five whole fresh statistics,
with no old encoder/head execution or fitted-state import. A separately reviewed
prospective22-condition protocol protects all leaders without a TRAIN-NLL gate.
No tuning, PCA, old training replay, winner, DEV change, publication or execution
follows from this preparation.