# CPB-v16: hidden temporal differences alongside V4 values

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument. Each channel is one time resolution of that instrument: minutes, hours, weeks, and the other kline intervals. The channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding from historical checkpoints. It is not this axis, and scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

Copyright (c) 2026 Waajacu. See LICENSE. Inherited notices remain in copied files.

**Source preparation only.** No build, fixture, model, GPU or scientific execution
has occurred for this version. Actual research/build protocols and admission are
separate. Earlier versions, source identities and frozen evidence remain unchanged.

This version owns a complete copy of the current48-file V4 closure under V16
namespaces/include paths/typed identities. The internal public entry is
`include/embedding/encoders/causal_patch_bottleneck_mae/versions/v16.h`.
Torch/STL are the only runtime dependencies. V4 is used only by the independent
synthetic comparison fixture. Source ancestry imports no learned tensor, fitted
statistic or optimizer state. The only initialization API remains a fresh seeded
constructor:75 learned tensors/293672 values,3888 fixed literal-prior values,
297560 total values and native36. Forward arithmetic/parameter registration stay
unchanged from V4, including the detached mean4 input of the variance head.

The unchanged original hidden-observed target mask Q defines the auxiliary:

```
loss = original_V4_Gaussian_NLL + 0.1 * (L1 + L4 + L16) / 3
error_l(t) = ((mu(t)-mu(t-l)) - (stop(y(t))-stop(y(t-l)))) / sqrt(2)
pair_support_l(t) = Q(t) AND Q(t-l), t=l..127
Ll = hierarchical_supported_mean(Huber_delta1(error_l))
```

Pairs use stride1 along the retained raw-minute grid within the same example,
channel and feature. There is no roll/wrap, index compression, new scaler,
per-window normalization, target head, uncertainty loss or division by lag.
Each endpoint is sanitized under pair support and separately promoted F32 to F64
before subtraction. Each lag averages supported time/feature pairs per channel,
then eligible channels per example, then eligible examples. An empty lag remains
connected zero with support counts0 and its fixed one-third weight; it is an
unsupported diagnostic, not perfect reconstruction. Gaussian eligibility and
update decisions remain unchanged. Context deletion removes only original-visible
inputs and cannot add Q endpoints. The level-blind auxiliary retains the value
Gaussian term to penalize constant level errors.

`trace.nll` remains the original V4 Gaussian object. Backward uses only
`trace.output.loss == trace.objective.total_loss`. Temporal direct gradients
reach mean/residual/backbone paths; variance-head gradients still come from the
one unchanged Gaussian term. Shared global norm clip1/single AdamW remain, so
direct routing separation does not establish optimization independence.

`trace.objective.temporal.lags` stores ordered1/4/16 terms. Each exposes `lag`,
`pair_mask`, `pair_counts`, `eligible_channels`, `eligible_examples`, `cell_error`,
`cell_huber`, `channel_loss`, `example_loss`, `loss`, `pair_count`,
`eligible_channel_count`, `eligible_example_count`. Cell tensors are
`[B,C,128-lag,F]`, counts `[B,C]`, channel/example eligibility `[B,C]`/`[B]`.
`temporal.loss` is the fixed three-lag mean; `temporal.active_lag_count` counts
supported terms. Saved fixed queries hide one8-step patch per channel, so lag16
is unsupported there; those masks remain unchanged.

Inference root/payload and TRAIN root/encoder/payload strictly bind objective,
lag order/type/length, F64 scale/delta/coefficient, base coefficient, support and
reduction before learned payloads. Missing, wrong-value/type/rank/order and old
or conflicting nested fields are refused. The full inherited causal/CUDA-learning
and exact-next-step typed-resume fixture remains, with full negative archives;
the dedicated variance-gradient fixture remains. The new temporal mechanism
fixture independently calculates scalar reductions, endpoint adjoints and F64
finite differences; covers unequal/empty support, nonfinite sanitation, no-wrap,
offsets/slopes, reorder/context separation, fixed-query lag16 absence, same-seed
V4 parameter/forward parity and shared clipping. Source is not an execution pass.

`v16.mk` supplies separate full, variance and temporal mechanism targets with
explicit seed/output arguments where relevant. Planned scientific seeds946101–3
and2048/B8 budget do not authorize execution without the separately reviewed
protocol/source/build gate. No root Makefile, index/catalog, older version or
application selection is changed here. The provenance receipt binds the actual
V4 source-copy identities and owned source bytes without a self-hash cycle.
