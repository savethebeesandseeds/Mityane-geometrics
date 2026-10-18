# Raw patch bottleneck MAE: version index

Copyright(c)2026 Waajacu. All rights reserved. Existing implementation notices are unchanged.

This is the single navigation index for the current source layout and the preserved historical design records. The physical cleanup changes paths only. Public APIs, namespaces, model tags, configurations, defaults, computations, initialization order, registered names and checkpoint formats stay unchanged. RPB-v22 stays primary; RPB-v24.market-01.alt-01 stays the balanced native36 reference. The separate causal_patch_bottleneck_mae family is unchanged.

Implementation files live under versions/vN/ when a current version-specific component exists, and under shared/ when the current code is shared. Versions that use the same Model or training-policy implementation do not receive invented model copies. Old public include roots remain compatibility headers. Existing application configuration paths remain compatibility selections. Read [shared components](../shared/README.md) for the dependency boundary.

The exact index.json retains its original historical fields and adds live_layout for current entry paths and repository availability. The 30 RPB-vN.json design records and all historical source/config/evidence pins remain unchanged. Their old paths and hashes identify the recorded stage; they are not rewritten to resemble the new layout. Use code/embeddings/scripts/rpb-layout-plan.json for the planned old-to-new path map and the cleanup verification record for before/after hashes.

Current entry means a locator for code that actually exists. In particular, current shared Model options do not prove byte-exact equivalence to every v1-v19 historical training definition. Original registries, cards and frozen capsules remain authoritative. No historical implementation is reconstructed here, and no new quality result follows from this cleanup.

Paths in the table are relative to this model family. Per-version README files give exact source entry points and repository-specific configurations. Both means the current header/component exists in both repositories; it does not imply that both contain the original experiment driver or that both ran a check. Integration lacks the dedicated v9/v11/v15/v21 components; their table links lead to embedding documentation.

| Design record | Current entry/navigation | Configuration | Availability |
| --- | --- | --- | --- |
| [RPB-v1](RPB-v1.json) | [Model](v1/README.md) | shared/config/evaluation.conf; shared/config/smoke.conf | both: shared component; full historical recipe unresolved |
| [RPB-v2](RPB-v2.json) | [Model](v2/README.md) | versions/v2/config/channel_mixer.conf | both: shared component; full historical recipe unresolved |
| [RPB-v3-mean](RPB-v3.json) | [Model](v3/README.md) | versions/v3/config/mean_global.conf | both: shared component; full historical recipe unresolved |
| [RPB-v4](RPB-v4.json) | [Model](v4/README.md) | versions/v4/config/learned_global.conf | both: shared component; full historical recipe unresolved |
| [RPB-v5](RPB-v5.json) | [Model](v5/README.md) | versions/v5/config/learned_patch_global.conf | both: shared component; full historical recipe unresolved |
| [RPB-v6](RPB-v6.json) | [context_deletion::descriptor](v6/README.md) | versions/v4/config/learned_global.conf | both: shared component; full historical recipe unresolved |
| [RPB-v7](RPB-v7.json) | [context_deletion::descriptor](v7/README.md) | versions/v4/config/learned_global.conf | both: shared component; full historical recipe unresolved |
| [RPB-v8](RPB-v8.json) | [context_deletion::descriptor](v8/README.md) | versions/v4/config/learned_global.conf | both: shared component; full historical recipe unresolved |
| [RPB-v9](RPB-v9.json) | [native_view_agreement::training_forward](v9/README.md) | Pinned historical card/registry; no dedicated current template | embedding only |
| [RPB-v10](RPB-v10.json) | [Model](v10/README.md) | Pinned historical card/registry; no dedicated current template | both: shared component; full historical recipe unresolved |
| [RPB-v11](RPB-v11.json) | [training_source_gain](v11/README.md) | Pinned historical card/registry; no dedicated current template | embedding only |
| [RPB-v12](RPB-v12.json) | [Model](v12/README.md) | Pinned historical card/registry; no dedicated current template | both: shared component; full historical recipe unresolved |
| [RPB-v13](RPB-v13.json) | [visible_first_differences](v13/README.md) | Pinned historical card/registry; no dedicated current template | both |
| [RPB-v14](RPB-v14.json) | [TemporalRelationModel](v14/README.md) | Pinned historical card/registry; no dedicated current template | both |
| [RPB-v15](RPB-v15.json) | [temporal_dynamics_targets](v15/README.md) | Pinned historical card/registry; no dedicated current template | embedding only |
| [RPB-v16](RPB-v16.json) | [PartitionedTemporalRelationModel](v16/README.md) | Pinned historical card/registry; no dedicated current template | both |
| [RPB-v17](RPB-v17.json) | [FixedPriorTemporalRelationModel](v17/README.md) | Pinned historical card/registry; no dedicated current template | both |
| [RPB-v18](RPB-v18.json) | [UnitTemporalRelationModel](v18/README.md) | Pinned historical card/registry; no dedicated current template | both |
| [RPB-v19](RPB-v19.json) | [SpectralTemporalRelationModel](v19/README.md) | Pinned historical card/registry; no dedicated current template | both |
| [RPB-v20](RPB-v20.json) | [ConfigurableSpectralModel](v20/README.md) | versions/v20/config/configurable_spectral.conf | both |
| [RPB-v21](RPB-v21.json) | [LearnedOrderedRelationModel](v21/README.md) | versions/v21/config/learned_ordered_relation.conf | embedding only |
| [RPB-v22](RPB-v22.json) | [ContentOrderedRelationModel](v22/README.md) | versions/v22/config/content_ordered_relation.conf | both |
| [RPB-v23](RPB-v23.json) | [ConsecutiveChangeModel](v23/README.md) | versions/v23/config/consecutive_changes.conf | both |
| [RPB-v24](RPB-v24.json) | [SeparatedChangeModel](v24/README.md) | versions/v24/config/separated_changes.conf; versions/v24/config/separated_changes_native36.conf | both |
| [RPB-v25](RPB-v25.json) | [EnergyChangeModel](v25/README.md) | versions/v25/config/energy_changes.conf | both |
| [RPB-v26](RPB-v26.json) | [ConditionedChangeModel](v26/README.md) | versions/v26/config/conditioned_changes.conf | both |
| [RPB-v27](RPB-v27.json) | [ParentConditionalChangeModel](v27/README.md) | versions/v27/config/parent_conditional_changes.conf | both |
| [RPB-v28](RPB-v28.json) | [RelativeChangeModel](v28/README.md) | versions/v28/config/relative_changes.conf | both |
| [RPB-v29](RPB-v29.json) | [ParentModulatedDecoderModel](v29/README.md) | versions/v29/config/parent_modulated_decoder.conf | both |
| [RPB-v30](RPB-v30.json) | [LocationScaleDecoderModel](v30/README.md) | versions/v30/config/location_scale_decoder.conf | both |

For integration, canonical settings are versions/v20/config/configurable_spectral.conf, versions/v22/config/content_ordered_relation.conf, and the v23-v30 configuration basenames shown above. Those are documented byte copies of existing application settings. config/encoder.conf and its current primary/default selection stay unchanged. Native40/native36 keep their distinct v24 configurations.

v1-v5 use the current shared Model with existing settings; v6-v8 share context_deletion descriptors. v10/v12 use shared Model options and shared comparison adapters in embedding. v9/v11/v13/v15 have existing current recipe/helper components, rather than newly reconstructed historical Model bodies. For v6-v8, the learned_global template alone does not bind the complete recipe: use its existing descriptor and pinned card/registry.

The preserved historical registry is archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md with its exact referenced instance registries. No second design registry is introduced. Checkpoints, results, scalers, fitted heads, experiment capsules and frozen provenance stay at their original paths and retain their original meanings. This navigation adds no causal or forecasting claim, training, fitting, benchmark or model selection.
