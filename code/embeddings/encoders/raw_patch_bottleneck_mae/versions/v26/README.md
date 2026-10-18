# RPB-v26 current source entry

Existing TRAIN-conditioned change model, bank and normalizer.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: ConditionedChangeModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v26/include/embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_model.h).
- integration: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v26/include/embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_model.h).

Configurations:

- embedding: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v26/config/conditioned_changes.conf](config/conditioned_changes.conf)
- integration: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v26/config/conditioned_changes.conf](config/conditioned_changes.conf)

Existing dependencies: v24 protected parent/checkpoint/frozen-level helpers; v23 change bank; v20 checkpoint utilities; shared core.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v26.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/embeddings/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
