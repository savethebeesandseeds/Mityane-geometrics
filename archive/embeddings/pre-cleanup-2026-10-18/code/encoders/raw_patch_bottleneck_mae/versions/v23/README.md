# RPB-v23 current source entry

Existing consecutive-change API plus its frozen-level helper.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: ConsecutiveChangeModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v23/include/embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_model.h](../../versions/v23/include/embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_model.h).
- integration: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v23/include/embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_model.h](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v23/include/embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_model.h).

Configurations:

- embedding: [code/encoders/raw_patch_bottleneck_mae/versions/v23/config/consecutive_changes.conf](../../versions/v23/config/consecutive_changes.conf)
- integration: [code/encoders/raw_patch_bottleneck_mae/versions/v23/config/consecutive_changes.conf](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v23/config/consecutive_changes.conf)

Existing dependencies: v22 content model/config/checkpoint; v20 checkpoint utilities; shared core.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v23.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
