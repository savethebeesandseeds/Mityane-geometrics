#!/usr/bin/env node
/** Bind exactly the saved TRAIN roles. This module never decodes tensors or fits heads. */
import path from 'node:path';
import { createPreparation, admitMatrix, require, equal, readText, isMain, main } from './prepare-fresh-decoder-replication.mjs';
import { POLICY_SOURCE } from './preparation-policies.mjs';
const protocol = 'saved-native-reliability-v1', card = 'code/embeddings/evaluation/cards/saved_native_reliability_v1.md';
const masters = [9109, 10210, 11311, 12412, 13513];
export const HEADER = ['id', 'display_tag', 'master_seed', 'controlled_training', 'native_training', 'encoder_progress',
  'fit_2701', 'pred_2701', 'fit_2802', 'pred_2802', 'fit_2903', 'pred_2903'];
export function declaredRows(parent) {
  const rows = [];
  for (const master of masters) {
    const root = path.join(parent, 'results', `seed-${master}-lag_sign`);
    for (const method of ['v4', 'v7']) {
      const files = [path.join(root, 'controlled-training.pt'), path.join(root, 'readouts', `native_${method}`, 'training-features.pt'),
        path.join(root, method, 'encoder-progress.json')];
      for (const repetition of [2701, 2802, 2903]) {
        const rep = path.join(root, 'readouts', `native_${method}`, `rep-${repetition}`);
        files.push(path.join(rep, 'fit.pt'), path.join(rep, 'training-predictions.pt'));
      }
      rows.push([`${method}-${master}`, `RPB-${method}.alt-01`, String(master), ...files]);
    }
  }
  return rows;
}

/** Source-only check retained from the historical helper; reads only three admitted C++ files. */
export function sourceSchemaCheck(repo) {
  const names = ['code/embeddings/evaluation/src/fresh_decoder_replication_main.cpp',
    'code/shared/representation/src/fixed_feature_readouts.cpp', 'code/shared/representation/src/saved_feature_reliability.cpp'];
  const admitted = new Map(admitMatrix(names.map(name => path.join(repo, name)).sort(), repo).map(row => [row.path, row]));
  const [fresh, readout, reader] = names.map(name => readText(path.join(repo, name), admitted.get(path.join(repo, name))));
  function body(text, signature) {
    const start = text.indexOf(signature), end = text.indexOf('\n}', start);
    require(start >= 0 && end > start, 'producer source signature absent: ' + signature);
    return text.slice(start, end + 2);
  }
  const writes = (text, variable) => new Set([...text.matchAll(new RegExp('\\b' + variable + '\\.write\\("([a-z0-9_]+)"', 'g'))].map(match => match[1]));
  const gets = (text, variable) => new Set([...text.matchAll(new RegExp('get\\(' + variable + ', "([a-z0-9_]+)"\\)', 'g'))].map(match => match[1]));
  const controlled = writes(body(fresh, 'void save_observations('), 'a'); controlled.delete('requested_erasure');
  const actual = { controlled, native: writes(body(readout, 'void save_surface('), 'out'),
    fit: writes(body(readout, 'void save_fit('), 'out'), prediction: writes(body(readout, 'void save_prediction('), 'out') };
  const needed = { controlled: gets(reader, 'observations'), native: gets(reader, 'features'),
    fit: new Set([...gets(body(reader, 'SavedFeatureFit load_fit('), 'in'), 'outer_fitted_rows', 'fitted_rows', 'tiny_hidden', 'tiny_steps']),
    prediction: gets(body(reader, 'SavedFeaturePrediction load_prediction('), 'in') };
  require(equal(actual, needed), 'reader/synthetic assumptions differ from actual producer archive keys');
  const bad = new Set(needed.controlled); bad.delete('observations'); bad.add('observed');
  require(!equal(actual.controlled, bad), 'wrong observation key fixture');
  return Object.fromEntries(Object.entries(actual).map(([key, value]) => [key, [...value].sort()]));
}

export const preparation = createPreparation({
  protocol, card, script: 'code/embeddings/scripts/prepare-saved-native-reliability.mjs',
  outputRoot: '.temp/embeddings/runs/rpb-saved-native-reliability', prefix: 'saved-native-reliability-',
  binary: 'embedding_saved_feature_reliability', reader: true, launchCard: true, strictMetadata: false, headFits: 0,
  requiredSources: [POLICY_SOURCE, card, 'Makefile', 'dependencies.lock',
    'code/embeddings/evaluation/src/saved_feature_reliability_main.cpp', 'code/embeddings/evaluation/include/frozen_role_guard.h',
    'code/shared/representation/include/embedding/shared/saved_feature_reliability.h',
    'code/shared/representation/src/saved_feature_reliability.cpp', 'code/shared/representation/tests/saved_feature_reliability_test.cpp',
    'code/embeddings/scripts/prepare-fresh-decoder-replication.mjs', 'code/embeddings/scripts/prepare-saved-native-reliability.mjs',
    'code/embeddings/scripts/check-saved-native-reliability.sh', 'code/embeddings/scripts/evaluate-saved-native-reliability.sh',
    'code/embeddings/evaluation/src/fresh_decoder_replication_main.cpp', 'code/shared/representation/src/fixed_feature_readouts.cpp'],
  markers: ['Saved native reliability CPU fixtures passed', 'Frozen role guard checks passed'],
  gates: ['synthetic CPU saved-map/logit/class/margin/geometry/trace fixtures',
    'closed role/path/alias rejection before payload hashing; no encoder or fitted-head code linked'],
  plan: { protocol, master_seeds: masters, display_tags: ['RPB-v4.alt-01', 'RPB-v7.alt-01'], instances: 10, unique_train_roles: 85,
    encoder_updates: 0, decoder_updates: 0, head_refits: 0, heldout_roles: 0, testing_accessed: false, stress_accessed: false },
  parent: { rows: declaredRows, header: HEADER, roleOffset: 3, roles: 85, countKey: 'unique_train_roles', heldoutKey: 'heldout_roles' },
  launchStatus: 'frozen_before_TRAIN_analysis', launchFields: { encoder_updates: 0, decoder_updates: 0, head_refits: 0, heldout_roles: 0 },
});
if (isMain(import.meta.url)) main(preparation);
