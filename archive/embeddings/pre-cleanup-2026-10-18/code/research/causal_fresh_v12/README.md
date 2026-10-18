# CPB-v12 owned fresh causal training research

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited SPDX/notices remain in the complete physically copied files.

SOURCE PREPARATION ONLY. This directory does not admit an AST, build, test,
model, training, extraction, head fit or quality run. The separate admitted
source plan is doc/causal_fresh_v12/design-source-plan.ADMITTED-001.json.
Root must review the complete owned sources, exact schema and prospective card
before any execution. V3 remains the clean baseline and sole DEV encoder.

## Owned source and scientific boundary

Eight complete non-test/non-evaluation research parents were preserved before
any edit under output/engineering/causal-fresh-v12-source-preparation/attempt-001/
training/parent-copies/, with an exact manifest. All eight live V11 parents match
the completed V11 source capsule. New V12 files began as complete physical copies;
there is no forwarding/import of old learned implementation or old project body.
The family and three research fixtures, evaluation sources and independent
training/quality auditors are owned separately. The Makefile refers only to the
new V12 family, tests and evaluation paths; it is not invoked during preparation.

V11 is the scientific/normal-pipeline SOURCE parent. Every learned value,
optimizer and permitted TRAIN statistic starts fresh; no trained V11/older/oracle
weight, teacher, fitted statistic or optimizer state initializes this candidate.
Historical final controls V3 through V11 and all109 preexisting critical
checkpoints remain preserved. V11 is the latest scoped reference, not an overall
winner, ancestry initializer or permission to replace DEV.

## One fixed parameter-neutral operator change

The four existing mean rows per original temporal bin retain V11's actual
half-local/half-shared-global learned-attention two-sum mean, byte-expression
unchanged. Only each bin's four dispersion-row inputs change. For valid retained
original patch set V and nonempty original local bin L_j:

u_j = .5 * 1[L_j]/|L_j| + .5 * 1[V]/|V|

mu_u,j = .5*sum(local_uniform_j*safe_h) + .5*sum(global_uniform*safe_h)

v_j = sum(u_j*(safe_h-mu_u,j)^2), componentwise population centered variance.

The uniform center is its OWN actual F32 two-sum center, never the attention
mean. Uniform support is per valid patch and depends only on mask/original
geometry. Unsupported operands are sanitized before subtraction/square and safe
count denominators keep empty/all-padding behavior finite. An absent local bin
still zeros the entire eight-row export slice after the one projection/bias,
even if global support exists. All four slices belong to the current requested
endpoint embedding; the global set contains only valid states at or before its
cutoff and is not a separate earlier-bin endpoint.

Hidden states and their uniform center remain live. Uniform dispersion has no
direct derivative into temporal pool scores or original-position score offsets;
learned-attention mean and hidden Transformer-attention derivatives remain live.
Independent scalar/F64/F32 adjoints must follow the actual two-sum F32 center and
retain the weighted-centered-residual correction rather than assuming exact
normalization cancellation. No detach, sqrt/RMS/Bessel/new-epsilon modification,
coefficient/support sweep, new parameter/pool, fitted statistic or post-encoder
transform is part of the source plan.

Existing32x64 export weights and32 bias, registration order/constructor draws,
global99->64->32/content20 bottleneck and native40 are unchanged by intent.
Native40 remains content20+literal spectral12+mean4+uncertainty4. Intended values
remain76 learned tensors/294752 learned values plus3888 fixed values. Actual
source/build/runtime parity must be verified later; this source preparation
claims no numerical parity or measurement. Equal parameters do not imply equal
compute, activation units, gradient routes, clipped AdamW state or quality.
Intermediate uniform variance is distinct from native uncertainty4 and decoder
conditional variance. The TRAIN diagnosis did not establish collapsed moments
or missing fitted TRAIN volatility information.

## Strict new typed identity

V12 owns its model tag/inference/TRAIN kinds and both new policy values. Inference
root/payload and TRAIN root/encoder/encoder.payload must all bind:

- temporal_pooling_policy: original-position-four-bin-attention-mean4-uniform-centered-variance4-v1
- temporal_pooling_partition_v1: Long[5]=[4,16,32,4,8]
- temporal_pooling_mix_f64: Double[2]=[0.5,0.5]
- empty_local_bin_policy: zero-whole-export-slice-after-bias-v1
- temporal_moment_policy: componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1
- temporal_export_layout: per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1
- temporal_moment_partition_v1: Long[2]=[4,4]

JSON uses the existing unsuffixed partition/mix/moment names. Strict wrong/missing
old policy/kind/value/dtype/shape refusal applies at every required node. No
legacy checkpoint conversion or import API is introduced. Typed resume512 must
remain exact on the single real trajectory; no update replay is permitted.

## Unchanged objective, data and streams

R[Huber_delta1(mu-y)] + R[.5*(stopgrad(mu-y)^2/v+log(v))], coefficients1/1.
Only objective.total_loss backpropagates. Parent32/mean4 receive the live Huber
route; uncertainty4 receives live detached-error Gaussian calibration. The ordinary
Gaussian reporting NLL copy is detached. Supported F32 mean/target operands are
sanitized and promoted separately before F64 arithmetic; hierarchical reduction
and positive variance link remain unchanged. V10 live joint mean-NLL and V7 split
clipping are excluded. Shared global norm2 clip1, one AdamW and later residual
changes retain optimization coupling; no independence claim is made.

Same fresh constructor943001, row943002, mask943003 and context943004 streams;
4096 TRAIN anchors, batch8,2048 successful updates with at most4096 attempts;
query25/context15/dropout0; AdamW .001/decay .0001. Calibration is fresh TRAIN-only
4096 windows/512 loader calls and executes no encoder. Immutable saved query
plans use fraction1/16, with no new query seed draw. Raw loader calls remain
512 + actual attempted batches +36, with skips derived from attempts minus2048.
Initial/final query observations and extraction/head phases retain their original
separate timing and admission boundaries.

The future candidate card must bind all22 exact strongest V3-V11 criteria from
the applied V11-retention protocol866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b,
plus its external passed review/application. No TRAIN-NLL gate or historical
rescoring is added. Same D40 ridge1/tanh16/Adam.01/100-update head recipes and
three repetitions/seeds are declared before quality. Initial reference is eight
saved V11 DATA archives20:40 ONLY plus five whole fresh-statistic archives;
content0:20 and whole40 equality are not required. No old model/head is replayed,
no initial score is reused and no PCA or head/checkpoint selection is introduced.

## Existing learning logs with new dispersion semantics

Keep2048 scalar training rows,8192 whole-export/position rows,2048 common-score
bias gauge rows,2048 between-bin Hadamard rows and16384 moment-subgroup rows.
The eight subgroups still partition four existing eight-row export slices into
mean4 and centered_variance4. CSV names/columns/labels and successful-update
blocks are unchanged. The label centered_variance now means the uniform-support
statistic with its own uniform center; mean still means learned-attention mean.
Receipt metadata carries the new pooling/moment policies explicitly.

Every mean/variance subgroup needs observed weight AND bias data-gradient and
movement events plus final changes, not every scalar or every update. Preserve
75 mandatory identifiable tensors and the common scalar pool-score-bias gauge
exemption;294751 values describe tensor capacity, not per-scalar learning. Gauge
movement/AdamW decay alone is never data learning. New direct dispersion-score
zeros do not eliminate score/relative-position gradients from the mean route.

## Exact same-existing-forward actual witnesses

At successful updates1/2048, the existing training forward saves the same base,
parameter and temporal evidence plus four explicit uniform arrays. No extra
encode or evidence forward is added. Actual2048 is pre-step2047. Its saved
before-step export coefficients and PRECLIP projection-output upstream are the
matching state; the final2048 decoder must never be substituted.

The actual writer has60 fields:23 unchanged base fields +12 unchanged parameter
before/after/postclip-gradient arrays +25 same-forward temporal fields. The latter
are21 explicit geometry/hidden/attention/moment/support tensors and four dynamic
upstream flag/value fields. The exact derived allowlist is in this directory's
training-helper-provenance.json, to be reviewed before any card freeze.

New keys and shapes:

- temporal_pool_dispersion_weights: Float[R,4,T]
- temporal_pool_dispersion_mean: Float[R,4,64]
- temporal_contextual_pool_dispersion_weights: Float[R,4,T]
- temporal_contextual_pool_dispersion_mean: Float[R,4,64]

They directly capture LocationScaleForwardTrace's pool_dispersion_weights,
pool_dispersion_mean, contextual_pool_dispersion_weights and
contextual_pool_dispersion_mean from the same owned kernel. Existing attention
local/global/mixture weights and pool_bin_mean remain attention quantities;
pool_bin_variance now records the independently centered uniform variance.
Both ordinary/contextual paths and original packed positions/support/row mapping
remain available. Undefined ordinary upstream retains its explicit Long flag and
zero fallback; this is not a dead-branch claim. Retained nonleaf upstream is
PRECLIP, scaled once by the recorded factor for independent local adjoints.
Parameter postclip arrays remain separately recorded. No full encoder/global-norm
or missing2046 raw-gradient reconstruction is claimed.

The source copies every historical measurement and checkpoint by reference only
as preserved DATA metadata for later card/evaluation phases. No current result,
passed proof, learning event, causal invariance, typed resume or quality is claimed
by source preparation. All execution requires a separately frozen exact source
capsule/card and independent admission in the existing managed GPU container.
