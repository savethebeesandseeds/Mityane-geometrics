> Live roles: [`index.json`](index.json) is the frozen v1-only declaration snapshot (`metadata_only`, 2026-10-15, `quality_measured` false) and is not a status page. [`current.json`](current.json) is the V14 historical catalog (2026-10-18, `active_application` null, clean baseline CPB-v3.market-01) and is not the research default. The only research default is [`research-reference.json`](../research-reference.json): V16 seed 946103, final logical update 2048, native36, selected on 22 October 2026. The next reference change uses [`cpb_family_screen_v1`](../../../evaluation/protocols/cpb_family_screen_v1/POLICY.md). Development cannot promote. January, February and March 2024 are consumed. See [ROLES.md](ROLES.md).

# Causal patch bottleneck MAE: version index

## Channel axis

cuwacunu_torch `interval_type_e` is the standard in every version below. One window is one instrument. Each channel is one time resolution of that instrument, and the channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding. Scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

Copyright(c)2026 Waajacu. All rights reserved. Inherited copied notices remain authoritative.

[CPB-v1](CPB-v1.json) is prospectively declared under the exact card pinned by the [index](index.json). Implementation, CUDA causality admission and fixed-head quality are pending. No quality score or trained checkpoint is invented here.

The authoritative physical implementation root is `versions/v1/` beneath the family. It owns its complete `include/`, `config/`, `tests/`, `v1.mk` and provenance. The sole public header, relative to that version root, is `include/embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h`; copied private implementation is inside that same version-owned include closure. Future versions require separate full implementation copies, without mutable shared encoder implementation. The original card and exact layout addendum are both pinned in the JSON. Family-level index files contain metadata only. It adds no includes, aliases, forwarding wrappers, inheritance or runtime calls into the historical family. Derivation/source-checkpoint references in JSON are provenance, not implementation dependencies.

CPB-v1 returns z_t through an explicit cutoff path: crop away future values AND masks before any validation/arithmetic/statistics, left-pad unavailable past, and enforce original-patch-coordinate causal temporal attention. Every sequence item uses its own cutoff. The mandatory suffix-value/mask poison, nonconstant positive-control, prefix-hidden-state and typed artifact proofs precede all quality work.

The first instance imports the exact V30 tensors as data with zero new encoder updates or scaler/bank fits. Causal execution may change old native outputs; old window-level results are not relabeled causal. The fixed MKT-HIST-2 four-task and TEMPO-5 two-policy/intact-deleted head recipes stay unchanged. All55 old checkpoints,99 portable pairs,44 frozen document bindings and23 interrupted V30 transfer files are preserved by the compact preflight. No old encoder/head replay, primary promotion, commit or publication follows.
