# RPB-v20 current source entry

Existing configurable spectral API and typed inference checkpoint/settings implementation.

This folder owns only its existing version-specific files (and this navigation). The entry may still depend on shared code and other versions.

Public symbol/component: ConfigurableSpectralModel.

Repository availability and exact entry:

- embedding: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v20/include/embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_model.h](../../versions/v20/include/embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_model.h).
- integration: existing current implementation/component. Entry: [code/encoders/raw_patch_bottleneck_mae/versions/v20/include/embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_model.h](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v20/include/embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_model.h).

Configurations:

- embedding: [code/encoders/raw_patch_bottleneck_mae/versions/v20/config/configurable_spectral.conf](../../versions/v20/config/configurable_spectral.conf)
- integration: [code/encoders/raw_patch_bottleneck_mae/versions/v20/config/configurable_spectral.conf](../../../../../../cuwacunu_dev/code/encoders/raw_patch_bottleneck_mae/versions/v20/config/configurable_spectral.conf)

Existing dependencies: shared Model/preprocessing/core; shared parser/config utilities; shared StreamingScaler remains separate support.

Current source entry only; original instance registries, checkpoint pins and frozen receipts remain authoritative.

Public APIs, namespaces, model/version tags, settings/defaults, computation, initialization, registered names and typed checkpoint formats are unchanged. This source-layout cleanup runs no training, fitting or quality experiment. Primary RPB-v22, balanced native36 and the separate causal family stay unchanged.

The [family version index](../README.md) is the sole navigation registry. The original [design record](../RPB-v20.json) preserves the historical declarations and pins. Canonical file moves are recorded in code/scripts/rpb-layout-plan.json; old include paths remain compatibility forwards.
