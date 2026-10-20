#!/usr/bin/env node
import { createPreparation, isMain, main } from './prepare-fresh-decoder-replication.mjs';
import { EARLY_SOURCES, EARLY_MARKERS, READOUT_MARKERS, EARLY_GATES, READOUT_GATE, INITIAL_PLAN, protocolSources } from './preparation-policies.mjs';
const protocol = 'early-mixer-reliability-v1', card = 'code/embeddings/evaluation/cards/early_mixer_reliability_v1.md';
export const preparation = createPreparation({
  protocol, card, script: 'code/embeddings/scripts/prepare-early-mixer-reliability.mjs',
  outputRoot: '.temp/embeddings/runs/rpb-early-mixer-reliability', prefix: 'early-mixer-reliability-',
  binary: 'embedding_early_mixer_reliability', reader: true, launchCard: true, strictMetadata: false,
  requiredSources: protocolSources('early-mixer-reliability', card, EARLY_SOURCES, { setup: false }),
  markers: [...EARLY_MARKERS, ...READOUT_MARKERS], gates: [...EARLY_GATES, READOUT_GATE],
  plan: { protocol, ...INITIAL_PLAN, timing_master_seeds: [19119, 20220, 21321, 22422, 23523],
    amplitude_data_master_seeds: [24624, 25725, 26826, 27927, 29028], tags: ['RPB-v7.alt-02', 'RPB-v10'],
    encoder_updates_each: 512, sampled_rows: 40960, parameter_count_each: 225805, planned_pipelines: 210, planned_heads: 420 },
  launchFields: { encoder_trajectories: 10, new_heads_planned: 420 },
  complete: { encoder_trajectories: 10, encoder_updates_each: 512, sampled_rows: 40960,
    CPU_encoder_training: false, CPU_encoder_forward: false, testing_accessed: false, stress_accessed: false },
  completeRanges: { head_pipelines: [-Infinity, 210] }, pairedHeads: true,
});
if (isMain(import.meta.url)) main(preparation);
