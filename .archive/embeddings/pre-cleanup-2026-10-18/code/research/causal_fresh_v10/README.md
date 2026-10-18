# CPB-v10 owned fresh causal research

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited notices remain in copied dependencies and LICENSES/NOTICE.

Source preparation only. No AST, build, fixture, encoder, training, extraction,
readout or quality execution is admitted until the separate prospective card,
independent preflight and immutable source capsule are reviewed and frozen.
This research family consists of 31 complete owned V9-derived physical files:
20 evaluation files, ten non-evaluation files (including this provenance), and
one separately owned objective fixture. Source owners prepare different files;
all actual source hashes must agree before capture. There is no forwarding or
runtime include/import of a mutable older version.

## Sole scientific change

V10 keeps the exact V9 forward/pooling/native40/constructor registration and
random draw order. Its only scientific change is the objective/gradient route:

`R[Huber_delta1(mu-y)] + R[0.5*((mu-y)^2/v + log(v))]`.

Both terms have unit coefficients and exactly the same supported hierarchical
reduction. Mean, target and variance are sanitized and promoted to F64 before
residual arithmetic. The Gaussian term is ONE ordinary joint mean/variance NLL
with live residual/mean and live uncertainty gradients. It replaces V9's
detached-error variance-NLL term; a second variance-only term is not added.
`objective.mean_huber` and `objective.joint_gaussian_nll` are separate traces;
only `objective.total_loss` is backpropagated. `trace.nll` remains a separately
detached reporting copy with the ordinary Gaussian forward value, never the
backward tensor. Fixed-query MAE and Gaussian NLL remain reporting measures.

The exact objective policy is `huber1-mean-plus-joint-gaussian-nll-v1`.
The gradient policy is
`huber1-mean-live;joint-gaussian-mean-live-and-uncertainty4-live;shared-global-clip1-v1`.
The typed direct route is
`Huber->parent32+mean4;Gaussian-mean->parent32+mean4;Gaussian-variance->uncertainty4;mean-reads0:36;variance-reads36:40;shared-global-clip1`.
Typed `variance_error_detached` is Long scalar 0 and
`reporting_nll_backpropagated` is Long scalar 0 at inference root/payload and
training root/encoder/encoder.payload. Huber delta remains Double scalar 1;
loss coefficients remain Double[2]=[1,1] in Huber/joint-Gaussian order.
JSON receipts use `variance_error_detached=false` and include the same direct
route; fresh-initialization deliberately omits `gradient_policy`, as in the
established schema, but retains the objective/route contract.

## Unchanged geometry, initialization and recipe

Native40 remains learned temporal20, fixed spectral12, mean4 and uncertainty4.
There are 298640 total values: 294752 freshly initialized learned values in
76 registered tensors plus 3888 analytic fixed-prior values. All weights,
optimizer moments and permitted TRAIN statistics start fresh; no old checkpoint,
normalizer or learned tensor is opened for training. Constructor seed943001,
row/mask/context streams943002/943003/943004, 4096 recent128 TRAIN calibration
windows, batch8, exactly2048 successful updates (at most4096 attempts),
query-mask25/context-deletion15, dropout0, AdamW .001/decay .0001 and global
norm clip1 remain unchanged. The saved evaluation-query masks remain unchanged;
seed943005 is not used to redraw them. Typed resume512 continues the actual
trajectory without replaying an update or resetting moments/counters/streams.
One optimizer/global clip still couples branch scaling; no optimizer independence
or branch dominance is claimed.

Both temporal paths retain V9's four original-position bins. Each nonempty bin
mixes local-softmax and shared-valid-history-softmax hidden vectors at fixed
one-half each before its existing eight export rows and bias are applied once.
An empty local bin zeros its whole export slice after bias, even when another
bin has global support. Original positions are not packed ranks. Local weights
sum1 per nonempty bin, global weights sum1 over active history, each nonempty
bin's mixture sums1, and the aggregate sums the number of nonempty bins. This
is causal at the requested history cutoff, not each bin's earlier endpoint.
The downstream global32-to-shape20 path, fixed spectral12 and independent
mean4/uncertainty4 frozen-bank projections are unchanged.

## Evidence and typed refusals

The common `pool_score.bias` scalar remains a softmax gauge, freshly registered
in the same optimizer. It is the only nonzero-data-gradient exemption. The other
75 mandatory tensors have numel capacity294751; tensor-level gradient/movement
and finer named groups are proved, not every scalar's nonzero learning. Recorded
movement capacity is sum of numel of tensors with movement events. Weight decay
is never counted as data-gradient evidence.

All2048 ordinary CSV rows record `Gaussian_NLL_F64`, `mean_Huber_F64`,
`joint_Gaussian_NLL_F64` and `total_objective_F64` separately, plus unchanged
observed global norm/factor and stream/row evidence. Fixed-query JSON/witnesses
and actual optimization component witnesses use `row_joint_gaussian_nll` and
`joint_Gaussian_NLL_F64`; no detached-error variance term is mislabeled as joint.
First/last summary values and the two actual pre-step component archives retain
all support, row, distribution and objective associations.

Temporal evidence remains8192 row-group records,2048 gauge records and2048
between-bin records. Every eight-row export weight/bias group and corresponding
four-position within-bin-centered contrast needs a nonzero data-gradient and
movement event plus final change. Three fixed unit Hadamard R16 directions use
[1,1,-1,-1], [1,-1,1,-1], [1,-1,-1,1] bin signs, repeated over each four-position
bin and divided by4. Each direction needs observed gradient/movement events;
only the overall final between-bin projection must change, not every final
signed coefficient. The same twelve weight/bias/position/gauge before/after/
clipped-gradient arrays at actual updates1/2048 suffice, with F64 promotion before
subtraction and no extra encoder forward. No unsaved upstream gradient or
optimizer/decoder trajectory is reconstructed. Evidence I/O remains outside
synchronized training-compute time and is separately timed.

The owned Makefile names seven binaries and eleven translation units. The rich
engineering fixture covers128 cutoffs/2350 suffix interventions/30 internal
interventions per initialized/trained stage, actual CUDA category/group learning,
seeded-constructor parity, unchanged frozen state and exact typed next-step resume.
Objective/gradient/delta/coefficient/detach/route/reporting fields are tested for
wrong/missing/dtype/shape values at all five actual archive nodes, and old V9
inference/payload/training kinds are explicitly rejected. The separate owned
objective fixture tests live joint mean and uncertainty routes, manual component
and gradient parity, supported/outlier cases and the retained global-clip coupling.
The temporal fixture retains one backbone encode plus five direct kernel calls,
one synthetic AdamW step, zero public Encoder calls and zero head fits.

## Fixed quality protocol and historical controls

Quality must independently export OWN initial0 and final2048 native40 banks and
fit unchanged heads once per stage. Because forward/initialization are identical,
all eight initial DATA banks must match SAVED V9 in their whole40 coordinates,
and all five fresh statistic archives must match numerically. These thirteen
pairings open no old model and reuse no initial quality score. Fresh V10 initial
heads are still fitted under the prospective fixed recipe. No trained-weight
import, old encoder/head replay, post-encoder PCA, seed selection or head tuning.

D40 fixed capacities remain MKT164/724 fitted values and TEMPO82/690; ridge1 and
16-tanh/Adam.01/100update heads and dimension-derived repetitions2701/2802/2903
remain unchanged. All seven V3-V9 final control tables and separate V6/V7/V8/V9
establishment receipts are preserved. The applied V9-retention protocol221f7cc7
keeps exactly22 criteria and no TRAIN-NLL gate. Actual cards, runtime quality
results, scores or trust/winner counts are not anticipated by source preparation.
CPB-v3 remains sole baseline/dev selection. Nothing in this source proposal
promotes a model, publishes changes or rewrites historical evidence.
