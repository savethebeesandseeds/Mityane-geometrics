# RPB-v7 current source entry

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument. Each channel is one time resolution of that instrument: minutes, hours, weeks, and the other kline intervals. The channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding from historical checkpoints. It is not this axis, and scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

Existing coordinate15 recipe. Embedding also owns the existing decoder-only calibration/frozen-native components here.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: context_deletion::descriptor.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/shared/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h](../../shared/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h).
- integration: shared current component; original experiment adapter/recipe closure is not asserted. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/shared/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h](../../shared/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h).

Configurations:

- embedding: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v4/config/learned_global.conf](../v4/config/learned_global.conf)
- integration: no dedicated current configuration template; use the original pinned card/registry and existing shared recipe controls.

Existing dependencies: shared Model/core and context_deletion; shared workflow; embedding v7 calibration helpers.

Exact equivalence to every historical training definition is not established by this current locator. Preserve original design record, registries and frozen capsules.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v7.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/embeddings/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
