#!/usr/bin/env node
import { createPreparation, isMain, main } from './prepare-fresh-decoder-replication.mjs';
import { RPB, EARLY_SOURCES, EARLY_MARKERS, READOUT_MARKERS, EARLY_GATES, READOUT_GATE, INITIAL_PLAN, protocolSources } from './preparation-policies.mjs';
const protocol = 'early-mixer-learning-curve-v1', card = 'code/embeddings/evaluation/cards/early_mixer_learning_curve_v1.md';
export const preparation = createPreparation({
  protocol, card, cardSha: '869d1a113ed8b6493295632555d69700a279f34e85b01754459761abcc7c4a98',
  script: 'code/embeddings/scripts/prepare-early-mixer-learning-curve.mjs',
  outputRoot: '.temp/embeddings/runs/rpb-early-mixer-learning-curve', prefix: 'early-mixer-learning-curve-',
  binary: 'embedding_early_mixer_learning_curve', reader: true, readerBlockedFlags: true, launchCard: true, singleLinkSources: true,
  requiredSources: protocolSources('early-mixer-learning-curve', card, [...EARLY_SOURCES, `${RPB}/shared/tests/early_mixer_curve_adapter_test.cpp`]),
  markers: [...EARLY_MARKERS, 'Early mixer curve CUDA admission passed', ...READOUT_MARKERS],
  gates: [...EARLY_GATES, 'curve horizon/direct/segmented/interleaved CUDA AdamW and live CPU-state witnesses', READOUT_GATE],
  plan: { protocol, ...INITIAL_PLAN, timing_master_seeds: [30130, 31231, 32332, 33433, 34534],
    amplitude_data_master_seeds: [35635, 36736, 37837, 38938, 40039], tags: ['RPB-v7.alt-03', 'RPB-v10.alt-01'],
    encoder_updates_each: 2048, milestones: [0, 512, 1024, 2048], attempt_limit: 4096, skipped_attempts_permitted: false,
    sampled_rows: 163840, parameter_count_each: 225805, planned_pipelines: 270, planned_heads: 540,
    timing_methods: 11, amplitude_methods: 7, planned_native_export_calls: 180, planned_query_writer_calls: 60,
    planned_query_forwards: 240, quality_exports_repeated: false, raw_outer_fit_shared_with_PCA: true },
  launchFields: { encoder_trajectories: 10, new_heads_planned: 540, milestones: [0, 512, 1024, 2048], sampled_rows: 163840,
    historical_quality_inputs: [], reader_execution_enabled_before_generation: false },
  complete: { encoder_trajectories: 10, encoder_updates_each: 2048, attempt_limit: 4096, sampled_rows: 163840,
    milestones: [0, 512, 1024, 2048], retained_points: 40, skipped_attempts: 0, planned_pipelines: 270, planned_heads: 540,
    full_native_export_calls: 180, query_evaluation_calls: 60, necessary_query_forwards: 240, quality_export_replays: 0,
    driver_raw_outer_fits: 10, earlier_parent_bytes_and_prefixes_preserved: true, selection: false, promotion: false,
    CPU_encoder_training: false, CPU_encoder_forward: false, testing_accessed: false, stress_accessed: false },
  completeRanges: { head_pipelines: [-Infinity, 270], helper_outer_train_fits: [-Infinity, 70] }, pairedHeads: true,
});
if (isMain(import.meta.url)) main(preparation);
