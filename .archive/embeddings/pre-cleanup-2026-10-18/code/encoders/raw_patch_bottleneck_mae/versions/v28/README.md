# RPB-v28 current source entry

Existing raw-F64 relative-change model, bank and normalizer.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: RelativeChangeModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v28/include/embedding/encoders/raw_patch_bottleneck_mae/relative_change_model.h](../../versions/v28/include/embedding/encoders/raw_patch_bottleneck_mae/relative_change_model.h).
- integration: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v28/include/embedding/encoders/raw_patch_bottleneck_mae/relative_change_model.h](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v28/include/embedding/encoders/raw_patch_bottleneck_mae/relative_change_model.h).

Configurations:

- embedding: [code/encoders/raw_patch_bottleneck_mae/versions/v28/config/relative_changes.conf](../../versions/v28/config/relative_changes.conf)
- integration: [code/encoders/raw_patch_bottleneck_mae/versions/v28/config/relative_changes.conf](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v28/config/relative_changes.conf)

Existing dependencies: v26 conditioned bank/normalizer; v24 protected parent/checkpoint/frozen-level helpers; v20 configurable model/checkpoint; shared core.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v28.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
