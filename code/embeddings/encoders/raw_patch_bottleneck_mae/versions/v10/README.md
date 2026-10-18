# RPB-v10 current source entry

Current shared Model exposes channel_mixer_placement. Cross-version early-mixer comparison adapters stay shared; focused architecture tests belong here in embedding.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: Model.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/shared/include/embedding/encoders/raw_patch_bottleneck_mae/model.h](../../shared/include/embedding/encoders/raw_patch_bottleneck_mae/model.h).
- integration: shared current component; original experiment adapter/recipe closure is not asserted. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/shared/include/embedding/encoders/raw_patch_bottleneck_mae/model.h](../../shared/include/embedding/encoders/raw_patch_bottleneck_mae/model.h).

Configurations:

- embedding: no dedicated current configuration template; use the original pinned card/registry and existing shared recipe controls.
- integration: no dedicated current configuration template; use the original pinned card/registry and existing shared recipe controls.

Existing dependencies: shared Model/core; shared early_mixer and confirmation adapters in embedding.

Exact equivalence to every historical training definition is not established by this current locator. Preserve original design record, registries and frozen capsules.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v10.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/embeddings/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
