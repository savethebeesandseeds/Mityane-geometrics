# CPB-v1

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.

This version owns its complete implementation and parser closure. Its only
external code dependencies are Torch and the C++ standard library. Include
`embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h` with this version's
`include/` directory on the include path. The public namespace is
`embedding::encoders::causal_patch_bottleneck_mae::v1`.

`RawHistory` holds CPU float64 observations, boolean masks, semantic channel
IDs, a uniform float64 timestamp grid and its sampling interval. `encode_at`
and `trace_at` require explicit per-example float64 cutoff timestamps. They
first select at most H observations through each cutoff, left-pad unavailable
past slots as hidden zeros, and only then validate observed values, round,
scale and compute features. There is no public unguarded `encode` or decoder.
An observed nonfinite value inside the retained window rejects; arbitrary
suffix values and masks are never read by those operations.

Patches are aligned to the retained H-length window ending at the cutoff.
Temporal attention blocks keys with a later original patch index. The channel
mixer continues to mix only channels at the same patch. All pooled, spectral,
lag and relative-change statistics use that retained window. LayerNorm remains
over feature width. A trace records actual temporal states and pooling weights
from the same guarded forward, alongside the retained inputs and bank values.

The first declared instance imports exact parameter, input-scaler and bank
normalizer data from the hash-pinned saved V30 inference artifact, with zero new
encoder updates or normalization fits. The importer is implemented entirely
inside this version. It verifies source SHA256, all parameter/buffer bytes,
scaler/normalizer tensors and metadata. Causal attention changes execution;
identical old native32/native36 outputs are not promised.

Ordinary `load_checkpoint` accepts only `cpb_v1_inference_checkpoint_v1`, tag
`CPB-v1`, the fixed causal policy and bound source/config/scaler/normalizer/state
identities. `save_checkpoint` publishes immutably. Its nested `payload` keeps
the imported data schema for exact tensor transfer; this is private serialization
data, with no dependency on or execution of another encoder implementation.
External input scaler policies may be supplied to `encode_at`; they never
replace the checkpoint scaler or the frozen market bank normalization.

The copied private training/archive helpers are retained as part of the complete
source closure. This version entry exposes no training API, and this zero-update
card declares no causal TRAIN artifact or optimizer run. Causal training needs
a separate prospective protocol.

`provenance.json` records original source hashes and new destination hashes.
Inherited MIT notices apply to the inherited portions; explicitly marked CPB
adaptations and new original files remain all rights reserved under the
repository license. This version's source, settings, fixtures and implementation
closure remain inside `versions/v1/`; future versions receive independent copies.
Saved checkpoints and measured evidence live in the instance's external artifact
directories and are bound by its provenance and result metadata.

Implementation status: created, pending engineering admission and fixed-head
evaluation. No quality claim follows from source review alone.
