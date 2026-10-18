# RPB-v22 current source entry

Existing content-only ordered-relation API and v22 training_state support. Primary selection stays RPB-v22.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: ContentOrderedRelationModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/include/embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_model.h).
- integration: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/include/embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_model.h).

Configurations:

- embedding: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/config/content_ordered_relation.conf](config/content_ordered_relation.conf)
- integration: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/config/content_ordered_relation.conf](config/content_ordered_relation.conf)

Existing dependencies: v20 configurable spectral model/config/checkpoint; shared core/parser; own content bank.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v22.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/embeddings/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
