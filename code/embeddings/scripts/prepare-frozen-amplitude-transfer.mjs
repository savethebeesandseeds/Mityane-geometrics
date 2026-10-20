#!/usr/bin/env node
/** Bind the fixed saved-parent matrix; no encoder execution or head fitting. */
import path from 'node:path';
import { createPreparation, isMain, main } from './prepare-fresh-decoder-replication.mjs';
import { POLICY_SOURCE, RPB } from './preparation-policies.mjs';
const protocol = 'frozen-amplitude-transfer-v1', card = 'code/embeddings/evaluation/cards/frozen_amplitude_transfer_v1.md';
const oldMasters = [9109, 10210, 11311, 12412, 13513], newMasters = [14614, 15715, 16816, 17917, 19018];
export const HEADER = ['id', 'display_tag', 'old_master', 'new_master', 'method', 'policy', 'updates',
  'controlled_training', 'checkpoint', 'audit', 'scaler', 'training_raw', 'native_training'];
export function declaredRows(parent) {
  const rows = [];
  oldMasters.forEach((oldMaster, i) => {
    const root = path.join(parent, 'results', `seed-${oldMaster}-lag_sign`);
    for (const [method, tag, version, updates, policy] of [
      ['untrained_native', 'Untrained encoder', 'v4', 0, 'ordinary_v4'],
      ['native_v4', 'RPB-v4.alt-01', 'v4', 512, 'ordinary_v4'],
      ['native_v7', 'RPB-v7.alt-01', 'v7', 512, 'coordinate15_v7'],
    ]) {
      const checkpoint = path.join(root, version, `point-${updates}`, 'checkpoint.pt');
      rows.push([`${method}-${oldMaster}`, tag, String(oldMaster), String(newMasters[i]), method, policy, String(updates),
        path.join(root, 'controlled-training.pt'), checkpoint, checkpoint + '.audit.pt', checkpoint + '.scaler.pt',
        checkpoint + '.training-raw.pt', path.join(root, 'readouts', method, 'training-features.pt')]);
    }
  });
  return rows;
}
export const preparation = createPreparation({
  protocol, card, script: 'code/embeddings/scripts/prepare-frozen-amplitude-transfer.mjs',
  outputRoot: '.temp/embeddings/runs/rpb-frozen-amplitude-transfer', prefix: 'frozen-amplitude-transfer-',
  binary: 'embedding_frozen_amplitude_transfer', reader: true, launchCard: true, strictMetadata: false,
  requiredSources: [POLICY_SOURCE, card, 'Makefile', 'dependencies.lock',
    'code/embeddings/evaluation/src/frozen_amplitude_transfer_main.cpp', 'code/embeddings/evaluation/include/frozen_role_guard.h',
    'code/shared/representation/src/fixed_feature_readouts.cpp',
    `${RPB}/versions/v7/include/embedding/encoders/raw_patch_bottleneck_mae/frozen_native_feature_adapter.h`,
    `${RPB}/versions/v7/src/frozen_decoder_calibration.cpp`, `${RPB}/versions/v7/tests/frozen_native_feature_adapter_test.cpp`,
    'code/embeddings/scripts/prepare-fresh-decoder-replication.mjs', 'code/embeddings/scripts/prepare-frozen-amplitude-transfer.mjs',
    'code/embeddings/scripts/check-frozen-amplitude-transfer.sh', 'code/embeddings/scripts/evaluate-frozen-amplitude-transfer.sh'],
  markers: ['Frozen native feature CUDA admission passed', 'Fresh decoder replication CUDA admission passed',
    'Fixed feature readout tests passed', 'Frozen role guard checks passed'],
  gates: ['new feature-only strict parent CUDA serving fixtures', 'legacy frozen decoder contracts',
    'fixed feature readout and whole-role guard fixtures'],
  plan: { protocol, old_master_seeds: oldMasters, new_data_master_seeds: newMasters, display_tags: ['RPB-v4.alt-01', 'RPB-v7.alt-01'],
    retained_snapshots: 15, unique_parent_roles: 80, encoder_updates: 0, decoder_updates: 0, planned_pipelines: 90, planned_heads: 180,
    train_pairs: 128, validation_pairs: 64, test_pairs: 0, testing_accessed: false, stress_accessed: false, quality_generated: false },
  parent: { rows: declaredRows, header: HEADER, roleOffset: 7, roles: 80, countKey: 'unique_parent_roles', heldoutKey: 'old_heldout_roles' },
  launchFields: { encoder_updates: 0, decoder_updates: 0, new_heads_planned: 180 },
});
if (isMain(import.meta.url)) main(preparation);
