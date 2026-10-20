#!/usr/bin/env node
import { createPreparation, isMain, main } from './prepare-fresh-decoder-replication.mjs';
import { RPB, EARLY_SOURCES, GAIN_SOURCES, EARLY_MARKERS, READOUT_MARKERS, EARLY_GATES, READOUT_GATE,
  POINT_PLAN, POINT_COMPLETE, NO_EXTRA_DECODER, protocolSources } from './preparation-policies.mjs';
const protocol = 'pooled-context-v1', card = 'code/embeddings/evaluation/cards/pooled_context_v1.md';
export const preparation = createPreparation({
  protocol, card, cardSha: 'b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7', planCardBinding: true,
  script: 'code/embeddings/scripts/prepare-pooled-context.mjs', outputRoot: '.temp/embeddings/runs/rpb-pooled-context',
  prefix: 'pooled-context-', binary: 'embedding_pooled_context', reader: true, readerBlockedFlags: true,
  launchCard: true, singleLinkSources: true, sdkGate: true,
  requiredSources: protocolSources('pooled-context', card, [...EARLY_SOURCES, ...GAIN_SOURCES,
    `${RPB}/shared/include/embedding/encoders/raw_patch_bottleneck_mae/pooled_context_adapter.h`,
    `${RPB}/shared/src/pooled_context_adapter.cpp`, `${RPB}/shared/tests/pooled_context_adapter_test.cpp`,
    `${RPB}/versions/v12/tests/pooled_context_model_test.cpp`, `${RPB}/versions/v12/tests/pooled_context_old_model_reference.h`]),
  markers: [...EARLY_MARKERS, 'Matched target gain CUDA admission passed', 'Pooled context model CUDA admission passed',
    'Pooled context adapter CUDA admission passed', ...READOUT_MARKERS, 'Container SDK proof:'],
  gates: [...EARLY_GATES, 'historical matched-target gain compatibility',
    'pooled shared initialization, independent wider matrix, inactive projection and live CUDA AdamW state', READOUT_GATE],
  plan: { protocol, ...POINT_PLAN, ...NO_EXTRA_DECODER, timing_master_seeds: [53151, 54252, 55353, 56454, 57555],
    amplitude_data_master_seeds: [58656, 59757, 60858, 61959, 63060], tags: ['RPB-v10.alt-03', 'RPB-v12'],
    parameter_counts: [225805, 231949], global_pool_input_sources: [0, 1], shared_parameter_values: 219469,
    candidate_inactive_projection_values: 2080, candidate_loss_reachable_parameter_values: 229869, checkpoint_roles_per_point: 5,
    planned_pipelines: 210, planned_heads: 420, methods_each_task: 7, separate_initial_controls: 2,
    planned_unique_quality_native_exports: 120, planned_initial_counterpart_exports: 0, helper_outer_train_fits: 50 },
  launchFields: { encoder_trajectories: 10, new_heads_planned: 420, retained_points: 20, sampled_rows: 40960,
    historical_quality_inputs: [], reader_execution_enabled_before_generation: false },
  complete: { ...POINT_COMPLETE, planned_pipelines: 210, planned_heads: 420, unique_quality_native_exports: 120,
    initial_counterpart_exports: 0, initial_shared_state_exact_before_training_and_heads: true, separate_initial_controls: 2 },
  completeRanges: { head_pipelines: [0, 210], helper_outer_train_fits: [0, 50] }, pairedHeads: true,
});
if (isMain(import.meta.url)) main(preparation);
