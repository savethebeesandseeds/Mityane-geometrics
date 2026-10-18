<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE. -->
# Immutable artifact sidecars

`versions/v1/` owns the small metadata utility for module and snapshot identities.
It has no Encoder or numerical-network dependency; `tensor_state.h` is the
separate LibTorch adapter. Future metadata encodings get an independent version
folder. Application adapters own archive interpretation and source-file lists.

Include `cuwacunu/artifacts/v1/sidecar.h` in namespace
`cuwacunu::artifacts::v1`. Each archive has exactly one immutable adjacent
`<archive>.manifest.json` using schema `model_artifact_sidecar_v1`.

| Identity | Version 1 representation |
| --- | --- |
| ModuleRef | Full component name, `vN` implementation version, actual source-closure SHA256. |
| WeightsRef | `weights_ref.state_sha256`: sorted component-local tensor names, dtype, shape and exact contiguous little-endian bytes, including adapter-supplied frozen state. |
| ModelRef | Derived digest of ModuleRef, WeightsRef, resolved inference configuration digest and ordered exact parents. |
| BundleRef | Derived digest of selected sidecar hashes and component ModelRefs, ordered by component name. |

Record hashes use versioned domains, fixed field order and length-delimited
strings. Paths and raw archive hashes do not enter ModelRef. Moving identical
bytes preserves identities; repackaging changes archive/sidecar/BundleRef pins.

Every component has an explicit parent array:

| Component | Required parent roles |
| --- | --- |
| Encoder | Empty array. |
| State Representation Network | `embedding`: Encoder. |
| Target State Representation Network | `embedding`: Encoder. |
| Dynamics Model | `input_history`: State Representation Network; `prediction_target`: Target State Representation Network. |
| Portfolio Policy | Reserved; selection is rejected. |

`make_component` derives ModelRef and normalizes parent order. `validate_manifest`
checks schema contents, derived IDs and fixed role declarations.
`validate_parents` resolves the whole selected closure, including external
Encoder and joint-archive components; every upstream ModelRef must match.

`publish_sidecar(archive, manifest)` verifies raw bytes and publishes without
replacing an existing sidecar. `read_verified(archive, sidecar_sha256, kind)`
requires an independently pinned sidecar SHA256, validates it and hashes the
archive bytes returned in memory. Adapters must deserialize those same bytes,
check the archive's internal kind, recompute state/configuration/source IDs and
validate the whole parent selection before exposing frozen inference access.
Hashes establish integrity relative to the trusted selection, not authenticity.

Kinds are exactly `training` and `inference`. Only training sidecars contain
`resume_identity_sha256`; existing optimizer, RNG, sampler, counters, EMA and
execution checks remain mandatory for exact resume. Inference configuration
excludes training-only settings. Existing joint predictive archives stay atomic.

Legacy adoption adds only a sidecar after independently verifying historical
implementation and parent identities; it must preserve every archive byte.
Missing evidence blocks adoption. The utility supplies no registry, dependency
solver, numerical model, retraining or fallback behavior.

Run `bash code/scripts/task.sh SESSION test-artifacts-v1` inside the existing
managed development container. The focused artificial tests cover hash goldens,
state identity, stale/swapped parents, malformed metadata, archive corruption,
strict kinds and immutable additive publication. They make no quality claim.
