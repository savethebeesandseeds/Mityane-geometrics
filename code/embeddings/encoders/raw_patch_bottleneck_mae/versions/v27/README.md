# RPB-v27 current source entry

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument. Each channel is one time resolution of that instrument: minutes, hours, weeks, and the other kline intervals. The channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding from historical checkpoints. It is not this axis, and scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

Existing parent-conditional change model and map.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: ParentConditionalChangeModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v27/include/embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_model.h).
- integration: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v27/include/embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_model.h).

Configurations:

- embedding: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v27/config/parent_conditional_changes.conf](config/parent_conditional_changes.conf)
- integration: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v27/config/parent_conditional_changes.conf](config/parent_conditional_changes.conf)

Existing dependencies: v26 conditioned normalizer/bank; v24 protected parent/checkpoint/frozen-level helpers; shared core.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v27.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/embeddings/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
