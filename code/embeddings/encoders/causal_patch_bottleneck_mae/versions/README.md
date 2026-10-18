# Causal patch bottleneck MAE: version index

Copyright(c)2026 Waajacu. All rights reserved. Inherited copied notices remain authoritative.

[CPB-v1](CPB-v1.json) is prospectively declared under the exact card pinned by the [index](index.json). Implementation, CUDA causality admission and fixed-head quality are pending. No quality score or trained checkpoint is invented here.

The authoritative physical implementation root is `versions/v1/` beneath the family. It owns its complete `include/`, `config/`, `tests/`, `v1.mk` and provenance. The sole public header, relative to that version root, is `include/embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h`; copied private implementation is inside that same version-owned include closure. Future versions require separate full implementation copies, without mutable shared encoder implementation. The original card and exact layout addendum are both pinned in the JSON. Family-level index files contain metadata only. It adds no includes, aliases, forwarding wrappers, inheritance or runtime calls into the historical family. Derivation/source-checkpoint references in JSON are provenance, not implementation dependencies.

CPB-v1 returns z_t through an explicit cutoff path: crop away future values AND masks before any validation/arithmetic/statistics, left-pad unavailable past, and enforce original-patch-coordinate causal temporal attention. Every sequence item uses its own cutoff. The mandatory suffix-value/mask poison, nonconstant positive-control, prefix-hidden-state and typed artifact proofs precede all quality work.

The first instance imports the exact V30 tensors as data with zero new encoder updates or scaler/bank fits. Causal execution may change old native outputs; old window-level results are not relabeled causal. The fixed MKT-HIST-2 four-task and TEMPO-5 two-policy/intact-deleted head recipes stay unchanged. All55 old checkpoints,99 portable pairs,44 frozen document bindings and23 interrupted V30 transfer files are preserved by the compact preflight. No old encoder/head replay, primary promotion, commit or publication follows.
