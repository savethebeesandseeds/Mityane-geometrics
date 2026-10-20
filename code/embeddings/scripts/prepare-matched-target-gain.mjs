#!/usr/bin/env node
import { createPreparation, isMain, main } from './prepare-fresh-decoder-replication.mjs';
import { EARLY_SOURCES, GAIN_SOURCES, EARLY_MARKERS, READOUT_MARKERS, EARLY_GATES, READOUT_GATE,
  POINT_PLAN, POINT_COMPLETE, protocolSources } from './preparation-policies.mjs';
const protocol = 'matched-target-gain-v1', card = 'code/embeddings/evaluation/cards/matched_target_gain_v1.md';
export const preparation = createPreparation({
  protocol, card, cardSha: '00d5d84815fd64a3e1787e3f25dddef045b1ca208458c19d5ca4000a5ca04eec', planCardBinding: true,
  script: 'code/embeddings/scripts/prepare-matched-target-gain.mjs', outputRoot: '.temp/embeddings/runs/rpb-matched-target-gain',
  prefix: 'matched-target-gain-', binary: 'embedding_matched_target_gain', reader: true, readerBlockedFlags: true,
  launchCard: true, singleLinkSources: true, sdkGate: true,
  requiredSources: protocolSources('matched-target-gain', card, [...EARLY_SOURCES, ...GAIN_SOURCES]),
  markers: [...EARLY_MARKERS, 'Matched target gain CUDA admission passed', ...READOUT_MARKERS, 'Container SDK proof:'],
  gates: [...EARLY_GATES, 'same early initialization, gain law/target pairing, immutable snapshots and live AdamW witnesses', READOUT_GATE],
  plan: { protocol, ...POINT_PLAN, timing_master_seeds: [41140, 42241, 43342, 44443, 45544],
    amplitude_data_master_seeds: [46645, 47746, 48847, 49948, 51049], tags: ['RPB-v10.alt-02', 'RPB-v11'],
    parameter_count_each: 225805, planned_pipelines: 180, planned_heads: 360, methods_each_task: 6, shared_initial_controls: 1,
    planned_unique_quality_native_exports: 90, planned_initial_counterpart_exports: 30, helper_outer_train_fits: 40 },
  launchFields: { encoder_trajectories: 10, new_heads_planned: 360, retained_points: 20, sampled_rows: 40960,
    historical_quality_inputs: [], reader_execution_enabled_before_generation: false },
  complete: { ...POINT_COMPLETE, planned_pipelines: 180, planned_heads: 360, unique_quality_native_exports: 90,
    initial_counterpart_exports: 30, original_initial_all_views_exact_before_heads: true },
  completeRanges: { head_pipelines: [0, 180], helper_outer_train_fits: [0, 40] }, pairedHeads: true,
});
if (isMain(import.meta.url)) main(preparation);
