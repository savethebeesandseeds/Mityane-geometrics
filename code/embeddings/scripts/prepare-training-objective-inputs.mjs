#!/usr/bin/env node
// Copyright (c) 2026 Waajacu. See ../../../LICENSE.
// Maintained source cutover; managed-container validation is still pending.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {
  JSON_ENCODING, admitFiles, canonicalDirectory, createExclusiveDirectory,
  fileSHA256, hashAdmitted, inside, jsonBytes, parseJSON, readAdmitted,
  requireCondition as require, safeRelative, sha256, writeNew,
} from '../../shared/artifacts/node/v1/files.mjs';

export const PROTOCOL = 'training-objective-diagnostic-v1';
export const BINDING_FORMAT = 'training-objective-inputs-node-v1';
export const MASTERS = Object.freeze([4404, 5505, 6606, 7707, 8808]);
export const TRAINING_NAMESPACE = 'native-development-v1/lag_sign';
export const PARENTS = Object.freeze([
  {id: 'v4', tag: 'RPB-v4', policy_id: '',
    capsule: 'output/runs/rpb-context-replication/context-replication-JjNEUc', cohort_prefix: 'reference',
    inventory_sha256: '13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9',
    orchestration_source_id: '9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828',
    training_producer_source_id: '2fff50c48605ee21dbbfd28d6989bdcf6b5cd057785c2acb19419571af48b895',
    core_writer_source_id: '587f2423758c3e70c2c7665d9e7b59bfe10af4c7f9aaba23f5318b3c55b13bca'},
  {id: 'v7', tag: 'RPB-v7', policy_id: 'rpb-training-context-deletion-015-v1',
    capsule: 'output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2', cohort_prefix: 'results/candidate-development',
    inventory_sha256: 'd4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821',
    orchestration_source_id: 'f22cc8d5f2ac8af3e6a6056c6ad2693ffcc82cf39e8744c32fb13038710c5311',
    training_producer_source_id: 'd8ed2771d1615beabccbead081847813b2562b7b1f9b5347d4825c075ccc27a7',
    core_writer_source_id: 'e379b8f8abb102c256fb843466b4fbfbecb6fdc21298b862fd0b6dfeb47b853e'},
  {id: 'v8', tag: 'RPB-v8', policy_id: 'rpb-training-context-balanced-030-v1',
    capsule: 'output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV', cohort_prefix: 'results/candidate-development',
    inventory_sha256: 'e0cdfbc729c8d989e16bb5e8515b5c5d73b7b445bdf91d2d938d88e09c969ac7',
    orchestration_source_id: '78ab6ffdb0a3edf141e0ad81e58fe10f36e276d4b7d8af1288859004b5f0f38c',
    training_producer_source_id: '3532deae194828f2640ff4fadacbcd94bbccfb8af6eaf37182cb8137ae84cf41',
    core_writer_source_id: '073ec84b7c781cf48a578e58a4c9602c6b765c9c23f6065f27338780be8a1835'},
].map(Object.freeze));
const PARENT_METADATA = Object.freeze([
  ['parent-inputs.json', '220767022994e5f7bab0497057034593de4cc47e3bf9d1f64de479e8de0fcd6a'],
  ['parent-inputs.sha256', 'd65ff6deacac126c29011ae3097be9bd1e2b4d520249119a61b38e9f95423b61'],
].map(Object.freeze));
const COHORT_FIELDS = Object.freeze([
  ['controlled_training_path', 'controlled-training.pt', 'TRAIN-observations-labels-source-order'],
  ['checkpoint_path', 'milestone-512/checkpoint.pt', 'frozen-checkpoint'],
  ['checkpoint_audit_path', 'milestone-512/checkpoint.pt.audit.pt', 'TRAIN-producer-audit'],
  ['checkpoint_scaler_path', 'milestone-512/checkpoint.pt.scaler.pt', 'TRAIN-fitted-model-scaler'],
  ['checkpoint_raw_training_path', 'milestone-512/checkpoint.pt.training-raw.pt', 'checkpoint-TRAIN-observations'],
  ['native_training_features_path', 'milestone-512/native-training.pt', 'TRAIN-native-export'],
  ['training_reconstruction_path', 'milestone-512/training-reconstruction.pt', 'TRAIN-fixed-query'],
].map(Object.freeze));
export const TSV_COLUMNS = Object.freeze([
  'input_id', 'tag', 'master_seed', 'budget', 'expected_policy_id', 'training_namespace',
  'orchestration_source_id', 'training_producer_source_id', 'core_writer_source_id',
  ...COHORT_FIELDS.map(([field]) => field),
  ...[1, 2, 3].flatMap(rep => [`rep_${rep}_native_fit_path`, `rep_${rep}_training_predictions_path`]),
]);
const isSHA = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
const json = bytes => parseJSON(bytes, {integers: 'number', duplicates: 'last'});
export function safeTrainRole(relative) {
  safeRelative(relative);
  require(!relative.toLowerCase().split(/[^a-z0-9]+/).some(token =>
    ['validation', 'test', 'testing', 'stress'].includes(token)), 'held-out role path forbidden');
  return relative;
}

// Pure closed role declaration. It performs no filesystem discovery or reads.
export function declaredTrainingRoles() {
  const rows = [];
  for (const master of MASTERS) for (const parent of PARENTS) {
    const inputId = `seed-${master}-${parent.tag}-updates-512`;
    for (const [field, suffix, kind] of COHORT_FIELDS) rows.push({
      inputId, instanceParent: parent.id, physicalParent: parent.id, master,
      relative: safeTrainRole(`${parent.cohort_prefix}/seed-${master}-lag_sign/${suffix}`), field, kind,
    });
    for (const rep of [1, 2, 3]) {
      const prefix = `results/readouts/${inputId}/rep-${rep}/`;
      rows.push({inputId, instanceParent: parent.id, physicalParent: 'v8', master,
        relative: safeTrainRole(`${prefix}native-fit.pt`), field: `rep_${rep}_native_fit_path`, kind: 'TRAIN-fitted-native-heads'});
      rows.push({inputId, instanceParent: parent.id, physicalParent: 'v8', master,
        relative: safeTrainRole(`${prefix}native-training-predictions.pt`), field: `rep_${rep}_training_predictions_path`,
        kind: 'TRAIN-native-head-predictions'});
    }
  }
  require(rows.length === 195 && new Set(rows.map(row => `${row.physicalParent}/${row.relative}`)).size === 195,
    'exact 195 declared physical TRAIN roles');
  return rows;
}
export function rolePlan(repoRoot, {historicalParentRoot} = {}) {
  canonicalDirectory(repoRoot); canonicalDirectory(historicalParentRoot);
  const metadataFiles = [
    ...PARENTS.map(parent => path.join(historicalParentRoot, parent.capsule, 'artifact-integrity.json')),
    ...PARENT_METADATA.map(([name]) => path.join(historicalParentRoot, PARENTS[2].capsule, name)),
  ];
  const admissions = admitFiles(metadataFiles, {root: historicalParentRoot});
  const metadata = [], indices = new Map();
  for (const [index, parent] of PARENTS.entries()) {
    const raw = readAdmitted(admissions[index]);
    require(sha256(raw) === parent.inventory_sha256, `pinned inventory metadata ${parent.id}`);
    const inventory = json(raw), entries = new Map();
    require(Array.isArray(inventory.files), 'inventory file list required');
    for (const item of inventory.files) {
      require(item && typeof item.path === 'string' && !entries.has(item.path) &&
        Number.isSafeInteger(item.bytes) && item.bytes >= 0 && isSHA(item.sha256), 'unambiguous inventory record');
      entries.set(item.path, item);
    }
    require(inventory.file_count === entries.size, 'inventory declared count');
    indices.set(parent.id, entries);
    metadata.push({path: metadataFiles[index], bytes: raw.length, sha256: sha256(raw), role: 'pinned-inventory-metadata'});
  }
  let parentRows;
  for (const [index, [filename, expected]] of PARENT_METADATA.entries()) {
    const admitted = admissions[PARENTS.length + index], raw = readAdmitted(admitted);
    require(sha256(raw) === expected, `pinned V8 parent boundary metadata ${filename}`);
    metadata.push({path: admitted.path, bytes: raw.length, sha256: sha256(raw), role: 'prior-explicit-role-metadata'});
    if (filename.endsWith('.json')) {
      const document = json(raw);
      require(Array.isArray(document.inputs), 'prior parent input metadata');
      parentRows = new Map(document.inputs.map(item => [JSON.stringify([item.parent_id, item.path]), item]));
    }
  }
  require(parentRows?.size === 326, 'prior role metadata count; not a payload allowlist');
  const instances = [], instanceMap = new Map(), inputs = [];
  for (const role of declaredTrainingRoles()) {
    const instanceParent = PARENTS.find(parent => parent.id === role.instanceParent);
    const physicalParent = PARENTS.find(parent => parent.id === role.physicalParent);
    let instance = instanceMap.get(role.inputId);
    if (!instance) {
      instance = {input_id: role.inputId, tag: instanceParent.tag, master_seed: role.master, budget: 512,
        expected_policy_id: instanceParent.policy_id, training_namespace: TRAINING_NAMESPACE,
        orchestration_source_id: instanceParent.orchestration_source_id,
        training_producer_source_id: instanceParent.training_producer_source_id, core_writer_source_id: instanceParent.core_writer_source_id};
      instances.push(instance); instanceMap.set(role.inputId, instance);
    }
    const item = indices.get(physicalParent.id).get(role.relative);
    require(item, `exact metadata role exists: ${physicalParent.id}/${role.relative}`);
    if (physicalParent.id !== 'v8') {
      const bound = parentRows.get(JSON.stringify([physicalParent.id, role.relative]));
      require(bound && bound.bytes === item.bytes && bound.sha256 === item.sha256,
        'original selected role agrees with V8 prior boundary');
    }
    const absolute = path.join(historicalParentRoot, physicalParent.capsule, role.relative);
    require(!/[\t\r\n]/.test(absolute), 'single-line absolute binding');
    instance[role.field] = absolute;
    inputs.push({input_id: instance.input_id, parent_id: physicalParent.id, path: role.relative,
      manifest_path: `${physicalParent.id}/${role.relative}`, absolute_path: absolute, kind: role.kind, api_field: role.field,
      tag: instance.tag, master_seed: instance.master_seed, budget: 512, split: 'TRAIN', expected_policy_id: instance.expected_policy_id,
      bytes: item.bytes, sha256: item.sha256, asset_writer_orchestration_source_id: physicalParent.orchestration_source_id,
      asset_parent_inventory_sha256: physicalParent.inventory_sha256, original_orchestration_source_id: instance.orchestration_source_id,
      original_training_producer_source_id: instance.training_producer_source_id, original_core_writer_source_id: instance.core_writer_source_id});
  }
  require(inputs.length === 195 && new Set(inputs.map(item => item.absolute_path)).size === 195 && instances.length === 15,
    'exact 195 physical roles and 15 instances');
  return {format_version: 1, protocol: PROTOCOL, binding_format: BINDING_FORMAT, json_encoding: JSON_ENCODING,
    historical_parent_root: historicalParentRoot, repository_root: repoRoot, stage: 'TRAIN-only-development-diagnostic',
    metadata_only: true, payload_bytes_read: 0, payloads_decoded: 0,
    parents: PARENTS, metadata_sources: metadata, inputs, instances, instances_tsv_columns: TSV_COLUMNS,
    input_count: 195, instance_count: 15, metadata_declared_bytes: inputs.reduce((sum, item) => sum + item.bytes, 0),
    typed_cohort_contract: {rows: 256, source_pairs: 128, shape: [256, 3, 32, 3], raw_dtype: 'float64', observation_mask_dtype: 'bool',
      source_ids_key: 'source_ids_json',
      source_ids_encoding: 'UTF8 uint8 JSON array in controlled_training_path; validate actual IDs against TRAIN audit fit_source_manifest',
      labels_key: 'labels_scoring_only', labels_dtype: 'int64', label_access: 'TRAIN scoring/grouping only; not encoder fitting',
      channel_ids: [0, 1, 2], feature_units: 'unitless,unitless,unitless', training_namespace: TRAINING_NAMESPACE,
      native_width: 32, native_post_encoder_pca: false, readout_repetitions: [2701, 2802, 2903]},
    scope: 'explicit TRAIN-only files; no point0, VAL manifest/payload, raw/PCA fits, TEST/stress or directory discovery'};
}
export function freeze(plan, {repoRoot, historicalParentRoot, output, frozenCard, frozenCardSHA256} = {}) {
  require(output && frozenCard && isSHA(frozenCardSHA256), 'freeze requires new output and exact frozen-card path/SHA');
  const fresh = rolePlan(repoRoot, {historicalParentRoot});
  require(jsonBytes(plan, {sortKeys: false}).equals(jsonBytes(fresh, {sortKeys: false})), 'role plan must match current pinned metadata');
  const [card] = admitFiles([frozenCard], {root: repoRoot});
  require(hashAdmitted(card) === frozenCardSHA256, 'card verified before first allowed payload hash');
  const outputRoot = path.join(repoRoot, '.temp/embeddings/runs/rpb-training-objective-diagnostic');
  require(path.isAbsolute(output) && output !== outputRoot && inside(outputRoot, output), 'new .temp diagnostic input directory required');
  require(!fs.existsSync(output), 'exclusive input output directory required');
  for (const parent of PARENTS) require(!inside(path.join(historicalParentRoot, parent.capsule), output), 'never write into historical parent');
  const admitted = admitFiles(plan.inputs.map(item => item.absolute_path), {root: historicalParentRoot, singleLink: true});
  // Strengthened admission: every size must match before any TRAIN payload hash.
  for (const [index, row] of admitted.entries()) require(row.bytes === plan.inputs[index].bytes,
    `allowed TRAIN byte count matches frozen inventory: ${plan.inputs[index].manifest_path}`);
  for (const [index, row] of admitted.entries()) {
    const expected = plan.inputs[index];
    require(hashAdmitted(row) === expected.sha256,
      `actual allowed TRAIN bytes match frozen inventory: ${expected.manifest_path}`);
  }
  for (const source of plan.metadata_sources) require(fileSHA256(source.path) === source.sha256, 'metadata unchanged during freeze');
  require(hashAdmitted(card) === frozenCardSHA256, 'frozen card unchanged');
  const planBytes = jsonBytes(plan, {sortKeys: false});
  const manifest = Buffer.from([...plan.inputs].sort((a, b) => a.absolute_path < b.absolute_path ? -1 : a.absolute_path > b.absolute_path ? 1 : 0)
    .map(item => `${item.sha256}  ${item.absolute_path}\n`).join(''), 'utf8');
  const tsvRows = [TSV_COLUMNS.join('\t'), ...plan.instances.map(instance => TSV_COLUMNS.map(column => {
    require(Object.hasOwn(instance, column), `missing TSV field: ${column}`); return String(instance[column]);
  }).join('\t'))];
  const tsv = Buffer.from(`${tsvRows.join('\n')}\n`, 'utf8');
  const bound = {...plan, metadata_only: false, actual_allowed_files_hashed: 195, payloads_decoded: 0,
    frozen_card_path: frozenCard, frozen_card_sha256: frozenCardSHA256, input_role_plan_sha256: sha256(planBytes),
    manifest_sha256: sha256(manifest), instances_tsv_sha256: sha256(tsv)};
  createExclusiveDirectory(output, {root: repoRoot});
  const files = [['input-role-plan.json', planBytes], ['inputs.json', jsonBytes(bound, {sortKeys: false})],
    ['inputs.sha256', manifest], ['instances.tsv', tsv]];
  for (const [name, bytes] of files) writeNew(path.join(output, name), bytes);
  return {status: 'frozen', protocol: PROTOCOL, binding_format: BINDING_FORMAT, output,
    inputs: 195, instances: 15, actual_allowed_files_hashed: 195, payloads_decoded: 0,
    files: files.map(([name, bytes]) => ({path: path.join(output, name), bytes: bytes.length, sha256: sha256(bytes)}))};
}
export function main(args = process.argv.slice(2)) {
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv') &&
    fileURLToPath(import.meta.url) === '/mityane/code/embeddings/scripts/prepare-training-objective-inputs.mjs',
  'run only through container.ps1 in mityane-geometrics');
  const {values} = parseArgs({args, options: {
    'role-plan': {type: 'boolean'}, freeze: {type: 'boolean'}, 'repo-root': {type: 'string', default: '/mityane'},
    'historical-parent-root': {type: 'string'}, output: {type: 'string'}, 'frozen-card': {type: 'string'},
    'frozen-card-sha256': {type: 'string'},
  }});
  require(Boolean(values['role-plan']) !== Boolean(values.freeze), 'choose exactly --role-plan or --freeze');
  require(values['historical-parent-root'], 'explicit --historical-parent-root mapping required');
  const repoRoot = canonicalDirectory(path.resolve(values['repo-root']));
  const historicalParentRoot = canonicalDirectory(path.resolve(values['historical-parent-root']));
  const plan = rolePlan(repoRoot, {historicalParentRoot});
  const result = values['role-plan'] ? plan : freeze(plan, {repoRoot, historicalParentRoot,
    output: values.output && path.resolve(values.output), frozenCard: values['frozen-card'] && path.resolve(values['frozen-card']),
    frozenCardSHA256: values['frozen-card-sha256']});
  process.stdout.write(jsonBytes(result, {sortKeys: false}));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
