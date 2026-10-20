#!/usr/bin/env node
import { createPreparation, isMain, main } from './prepare-fresh-decoder-replication.mjs';
import { RPB, EARLY_SOURCES, GAIN_SOURCES, EARLY_MARKERS, READOUT_MARKERS, EARLY_GATES, READOUT_GATE,
  POINT_PLAN, POINT_COMPLETE, NO_EXTRA_DECODER, protocolSources } from './preparation-policies.mjs';
const protocol = 'early-mixer-confirmation-v1', card = 'code/embeddings/evaluation/cards/early_mixer_confirmation_v1.md';
export const preparation = createPreparation({
  protocol, card, cardSha: '98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea', planCardBinding: true,
  script: 'code/embeddings/scripts/prepare-early-mixer-confirmation.mjs', outputRoot: '.temp/embeddings/runs/rpb-early-mixer-confirmation',
  prefix: 'early-mixer-confirmation-', binary: 'embedding_early_mixer_confirmation', reader: true, readerBlockedFlags: true,
  launchCard: true, singleLinkSources: true, sdkGate: true,
  requiredSources: protocolSources('early-mixer-confirmation', card, [...EARLY_SOURCES, ...GAIN_SOURCES,
    `${RPB}/shared/include/embedding/encoders/raw_patch_bottleneck_mae/early_mixer_confirmation_adapter.h`,
    `${RPB}/shared/src/early_mixer_confirmation_adapter.cpp`, `${RPB}/shared/tests/early_mixer_confirmation_adapter_test.cpp`]),
  markers: [...EARLY_MARKERS, 'Early mixer confirmation CUDA admission passed', ...READOUT_MARKERS, 'Container SDK proof:'],
  gates: [...EARLY_GATES, 'truthful delegated fit namespace plus external confirmation/scaler/source/counter and five-parent-role binding', READOUT_GATE],
  plan: { protocol, ...POINT_PLAN, ...NO_EXTRA_DECODER, timing_master_seeds: [64161, 65262, 66363, 67464, 68565],
    amplitude_data_master_seeds: [69666, 70767, 71868, 72969, 74070], tags: ['RPB-v7.alt-04', 'RPB-v10.alt-04'],
    parameter_count_each: 225805, channel_mixer_placements: [0, 1], global_pool_input_sources: [0, 0], checkpoint_roles_per_point: 5,
    external_fit_protocol: protocol + '/lag_sign', implementation_fit_protocol: 'early-mixer-reliability-v1/lag_sign', point_binding_suffix: '.confirmation.pt',
    planned_pipelines: 210, planned_heads: 420, methods_each_task: 7, separate_initial_controls: 2,
    planned_unique_quality_native_exports: 120, planned_initial_counterpart_exports: 0, helper_outer_train_fits: 50 },
  launchFields: { encoder_trajectories: 10, new_heads_planned: 420, retained_points: 20, sampled_rows: 40960,
    historical_quality_inputs: [], reader_execution_enabled_before_generation: false },
  complete: { ...POINT_COMPLETE, ...NO_EXTRA_DECODER, planned_pipelines: 210, planned_heads: 420, unique_quality_native_exports: 120,
    initial_counterpart_exports: 0, initial_shared_state_exact_before_training_and_heads: true, separate_initial_controls: 2 },
  completeRanges: { head_pipelines: [0, 210], helper_outer_train_fits: [0, 50] }, pairedHeads: true,
});
if (isMain(import.meta.url)) main(preparation);
