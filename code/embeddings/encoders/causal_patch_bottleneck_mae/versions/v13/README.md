# CPB-v13: attention-centered regularized dispersion

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
Inherited copyright and license notices remain in each complete physical copy.

SOURCE PREPARATION ONLY. No AST, build, fixture, model, training, head fit or
quality execution has occurred in this preparation. The canonical descriptor and
experiment card remain absent. CPB-v3.market-01 is the clean baseline and sole DEV
encoder. Preserve every V3–V12 checkpoint, scoped leader, measurement and failure.

The50 files here, full external versions/v13.h, three owned research fixtures and
descriptor draft originate from complete FINAL V11 source parents preserved
before copying under the central causal-fresh-v13-source-preparation manifest.
Both public headers contain the same full body; neither forwards to another
version. Private includes and namespaces stay within this V13 family. Source
ancestry imports no weights: all76 learned tensors/294752 values, AdamW state and
TRAIN statistics must start fresh. Fixed3888 values, total298640, native40,
registration order, constructor draws, geometry and all other objectives remain
the V11 source recipe. These are source declarations, not executed parity proof.

Only the last four existing export rows of each original temporal bin change.
With unchanged original-support local/global softmax weights, compute

```
p    = .5 * local_softmax + .5 * global_softmax
mu   = .5 * sum(local_softmax * safe_hidden)
     + .5 * sum(global_softmax * safe_hidden)
v    = sum(p * (safe_hidden - mu)^2)
root = sqrt(v + 0.0009765625)       // F32 add, then F32 sqrt
s    = root - 0.03125              // separate F32 subtraction
```

The first four projection rows and their once-applied bias still read the exact
parent two-sum attention mean. The last four rows now read s. No rationalization,
hard clamp, detach, Bessel correction, learned scale, new normalization, V12
uniform support/center or changed attention policy is introduced. Tau is exactly
2^-10 and its root offset exactly1/32. No regularizer sweep is allowed.

Every nonempty slice describes the CURRENT cutoff, not its earlier bin endpoint.
Only valid retained original positions at/before that cutoff reach either
ordinary or contextual path, both of which use the same owned kernel. Hidden
values outside token support are sanitized before arithmetic. For an absent LOCAL
bin, BOTH hidden and center operands are zeroed BEFORE subtract/square; this
avoids unsupported0*Inf for a finite huge singleton in another bin without
changing present-bin arithmetic. Supported nonfinite hidden/scores and overflowed
or negative raw variance, nonfinite root or dispersion are refused before export.
All-padding softmax uses a finite fallback. An absent bin's whole EIGHT served
coordinates are exactly zero AFTER bias, even with nonempty global history.

Raw witness semantics deliberately differ from served padding: v=0 has raw
regularized root1/32 even at an absent/all-padding bin. Raw root is UNMASKED;
dispersion is zero there. Model trace initialization follows the same rule.
Both routes expose raw bin_variance, bin_regularized_root and bin_dispersion
[R,4,64] from the existing forward, without another encoder call. Four added
TemporalExecutionTrace members are pool_bin_regularized_root, pool_bin_dispersion,
contextual_pool_bin_regularized_root and contextual_pool_bin_dispersion. TRAIN
owns four corresponding temporal_* archive arrays; its actual allowlist must be
text-derived and frozen separately. Intermediate16 dispersion export coordinates
are distinct from native uncertainty4 and decoder variance.

Full attention/center derivatives remain live. For mean upstream a and dispersion
upstream c, beta=c/(2*ACTUAL saved root), delta=h-mu, r=sum(p*delta), and center
upstream is a-2*beta*r. Direct hidden and mixture routes contribute
2*p*beta*delta and beta*delta^2. Propagate these through local/global softmax,
hidden-dependent scores and positional offsets. Do not assume captured F32 r=0.
The mean coefficient follows its two separate sums; a rounded stored p is not
silently substituted in the captured-F32 adjoint. Transform derivative is bounded
by16 and equals16 at v=0. No full encoder/global norm reconstruction is claimed.

Checkpoint identity includes exact pooling/moment/export policies, String
temporal_dispersion_transform_policy, and temporal_dispersion_regularizer as a
Float64 vector shape[1], value0.0009765625. Common writer/reader enforce these at
inference root/payload and TRAIN root/encoder/encoder.payload. Reject
missing/old/changed strings and wrong dtype/rank/length/value, including Float32,
0-D Double, empty, length2, rank2, NaN and negative tau. Engineering also mutates
old V11 artifact kinds at all five nodes. Own typed reload/resume must prove
state/optimizer/RNG/counter equality; no old checkpoint is initialized.

Temporal fixture's source-derived contract is51 direct kernel calls:
44 inherited successful/finite-difference calls, three zero/tiny/large cases,
one finite-huge-singleton case, and three refused supported NaN/Inf/raw-overflow
calls. It has18 independent full-expression F64 finite-difference coordinates,
18 actual CUDA F32 coordinates, four isolated autograd::grad calls, one actual
backbone encode, no public encoder calls or heads, and one synthetic AdamW step.
Nine true proof tokens are mean4_regularized_stddev4_export_layout,
full_attention_dependent_regularized_moment_adjoints,
live_dispersion_score_gradient, raw_variance_root_dispersion_trace,
zero_tiny_large_analytic_dispersion_adjoints,
supported_nonfinite_and_variance_overflow_refused,
absent_local_huge_singleton_sanitized, captured_F32_residual_correction and
raw_root_absent_offset_exception. These are planned checks, not measured passes.

Prospective tolerances remain manual F32 abs/relative3e-5; F64 finite differences
step1e-5/tolerance2e-8; actual F32 step2e-3/tolerance2e-3; common gauge2e-6;
present constant raw variance1e-10*(1+max_abs_hidden_squared). At least three
finite-difference reference derivatives must exceed1e-2. Zero/tiny/large cases
compare independent captured-F32 scalar hidden/score adjoints and actual-root
derivative bound16. Scalar F64 forward/FD reference independently computes the
entire attention/moment/root graph; captured-F32 mode uses saved values, never
observed gradients. Primary fixtures compare all new traces under future
values/masks and verify root transform bits at every checked cutoff.

Research engineering negative cases are source-derived636, not an observed count:
33 initial mutations +10*(21 objective rows+37 temporal rows)+19 standalone
archive mutations+4 cross-type/request failures. Family primary has491
(33+10*(6+38)+14+4); normal research uses its richer complete owned fixture.
Both retain128-cutoff suffix value/mask and past-positive causal matrices.
Three research fixtures retain objective/decoder separation and fresh CUDA
learning/resume checks; labels are mean/regularized_stddev and
temporal_pool_regularized_stddev_bin_0..3. Raw variance trace names stay raw.

Future training stays one fresh2048-update/B8 trajectory, globalclip1 and one
AdamW, seeds943001/943002/943003/943004, unchanged unit Huber1 mean plus
detached-error Gaussian variance objective and detached reporting NLL. Fixed
initial/final heads, strongest866 protocol22 and all leaders stay unchanged.
Initial references compare eight saved V11 DATA tails20:40 and five whole fresh
statistics only; learned0:20 and whole40 remain unpaired. No old model/head replay,
PCA, rescue/tuning, DEV replacement, Git publication or measured winner follows.
