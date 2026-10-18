# Shared RPB components

This directory owns the current reusable Model/core, preprocessing, common workflow and cross-version adapters/tests. Sharing is unchanged by relocation. Headers retain namespace embedding::encoders::raw_patch_bottleneck_mae and the public include prefix embedding/encoders/raw_patch_bottleneck_mae/. Existing include roots forward to the canonical files; they contain no separate model implementation.

The current Model is shared by the v1-v13 mechanisms and by later backbones. Its Config, encoder, types, tokenization, masking, objectives, preprocessing and training utilities stay shared. Model still includes the existing v13 visible-difference helper. context_deletion remains shared because its existing descriptors serve v6/v7/v8. Cross-version evaluation, reconstruction, replay, comparison, admission and learning-curve adapters remain shared. Embedding's v7 decoder-only calibration helpers retain their existing restricted parent contract.

The implementation dependencies remain:

- v14 uses shared Model; v16 uses v14; v17 uses v16; v18 uses v17; v19 uses v18 and its own spectral bank.
- v20 uses shared Model/preprocessing. v21 and v22 use v20. v23 uses v22.
- v24 uses the protected v22 parent and v23 change-bank/config/TRAIN helpers.
- v25-v30 use the protected v24 parent; v26 also uses the v23 bank; v27 uses v26; v28 uses the v26 bank/normalizer; v29/v30 use v28.
- StreamingScaler stays shared. The distinct training_state API stays v22-owned.

Folder ownership does not imply independence. These dependencies preserve the existing includes, inheritance, registered names, random-number consumption, serialization keys and checkpoint compatibility. No historical version receives a copied current implementation.

Integration contains the imported shared header closure plus its existing application facades/tests; embedding additionally contains standalone workflow/evaluation sources and historical experiment adapters. Absence of an integration adapter is recorded in the existing versions/index.json live_layout, not filled by reconstruction. External shared parser/archive/evaluation utilities remain at their existing repository-owned locations.

Read the [version index](../versions/README.md) for all version entry points, configurations and availability. Original import manifests, historical source snapshots, registries and receipts stay at their original paths. Legacy roots are compatibility surfaces; canonical source ownership is here or in versions/vN/.
