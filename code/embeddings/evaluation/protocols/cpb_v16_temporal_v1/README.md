# CPB V16 temporal reconstruction study

The owner approved the multiscale temporal-change reconstruction direction on
19 October 2026. The [prospective protocol](PROTOCOL.md) defines the exact
V4-native36 candidate, six fresh trajectories at three paired seeds, unchanged
evaluation heads/protection floors and March2024 confirmation chosen before
values. Source preparation and review precede all builds and execution.

Candidate V16 adds `0.1*(Huber_lag1+Huber_lag4+Huber_lag16)/3` to V4 Gaussian
value reconstruction. Both endpoints must be original hidden targets, with
fixed sqrt2 scaling, Huberδ1 and pair/channel/example reductions. No extra
encoder parameters, supervision from readout labels or new fitted statistics.
Empty lags retain fixed weight and are separately marked unsupported.

New owned source derives from the completed V15 study infrastructure; originals
and previous results are preserved. The local source-copy records distinguish
historical bytes, prepared derivatives and actual execution fingerprints.
The source is not itself a claim of successful engineering or model quality.

Use the existing managed container via root `container.ps1`, Makefile and named
task `cpb-v16-temporal-001`. Builds stay under
`/opt/mityane-geometrics/embeddings/build/cpb-v16-temporal-001`; persistent local
artifacts stay under `.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001`.
Data-handling code belongs to
[`code/data/market/experiments/cpb_v16_temporal_v1`](../../../../data/market/experiments/cpb_v16_temporal_v1/).

Current progress belongs to [Hero17](../../../../../web/dashboard/heroes/17-encoder-cpb.json)
and the [research guide](../../../../../doc/EMBEDDINGS.md). Audited scoped
standings belong to the [family ranking](../../../encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md).
Preserve prior leaders. Complete results and independently reviewed bindings
will be saved separately after the actual execution; nothing here selects a
checkpoint, applies a catalog version or claims a new best encoder.
