# RPB-v29 current source entry

Existing parent-modulated decoder API and typed state.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: ParentModulatedDecoderModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v29/include/embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_model.h](../../versions/v29/include/embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_model.h).
- integration: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v29/include/embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_model.h](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v29/include/embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_model.h).

Configurations:

- embedding: [code/encoders/raw_patch_bottleneck_mae/versions/v29/config/parent_modulated_decoder.conf](../../versions/v29/config/parent_modulated_decoder.conf)
- integration: [code/encoders/raw_patch_bottleneck_mae/versions/v29/config/parent_modulated_decoder.conf](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v29/config/parent_modulated_decoder.conf)

Existing dependencies: v28 relative config/normalizer/bank; v24 protected parent/checkpoint/frozen-level helpers; shared core.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v29.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
