# RPB-v30 current source entry

Existing location/scale decoder API and typed state.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: LocationScaleDecoderModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v30/include/embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_model.h).
- integration: existing current implementation/component. Entry: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v30/include/embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_model.h](include/embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_model.h).

Configurations:

- embedding: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v30/config/location_scale_decoder.conf](config/location_scale_decoder.conf)
- integration: [code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v30/config/location_scale_decoder.conf](config/location_scale_decoder.conf)

Existing dependencies: v28 relative config/normalizer/bank; v24 protected parent/checkpoint/frozen-level helpers; shared core.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v30.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/embeddings/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
