# RPB-v24 current source entry

Existing separate-change API. Native40 and native36 are separate declared configurations of this same implementation; native36 balanced reference is unchanged.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: SeparatedChangeModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v24/include/embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h](../../versions/v24/include/embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h).
- integration: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v24/include/embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v24/include/embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h).

Configurations:

- embedding: [code/encoders/raw_patch_bottleneck_mae/versions/v24/config/separated_changes.conf](../../versions/v24/config/separated_changes.conf); [code/encoders/raw_patch_bottleneck_mae/versions/v24/config/separated_changes_native36.conf](../../versions/v24/config/separated_changes_native36.conf)
- integration: [code/encoders/raw_patch_bottleneck_mae/versions/v24/config/separated_changes.conf](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v24/config/separated_changes.conf); [code/encoders/raw_patch_bottleneck_mae/versions/v24/config/separated_changes_native36.conf](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v24/config/separated_changes_native36.conf)

Existing dependencies: v22 protected content parent; v23 change bank/config/TRAIN helpers; v20 checkpoint utilities; shared core.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v24.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
