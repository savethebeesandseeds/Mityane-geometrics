// Copyright (c) 2026 Waajacu. See ../../../LICENSE.
// Maintained runner bookkeeping. Scientific protocols and historical hashes stay
// unchanged; newly written receipts identify their JavaScript implementation.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual} from 'node:util';
import {JSON_ENCODING, requireCondition as require, safeRelative, canonicalDirectory,
  admitFiles, readAdmitted, hashAdmitted, fileSHA256, readJSON, parseJSON, jsonBytes,
  writeNew, writeJSON} from '../../shared/artifacts/node/v1/files.mjs';

export const METADATA_FORMAT = 'embedding-runner-metadata-node-v1';
export const MASTERS = Object.freeze([4404, 5505, 6606, 7707, 8808]);
export const CARD_HASHES = Object.freeze({
  'native-view-agreement-validation-v1': ['native_view_agreement_validation_v1.md', '1f33eff0e6ca3e3241fd972fdc855900c535defa330f4eaec7effdf0c90d1c2e'],
  'v7-decoder-calibration-v1': ['v7_decoder_calibration_v1.md', '5ae05a3b5ec253d1743842f82fc28b45e0f20c8c84da1b58bf4ab9c514d292e9'],
});
const self = fileURLToPath(import.meta.url);
const shared = fileURLToPath(new URL('../../shared/artifacts/node/v1/files.mjs', import.meta.url));
const digest = value => require(typeof value === 'string' && /^[0-9a-f]{64}$/.test(value), 'exact SHA256 required');
const same = (value, expected, label) => require(isDeepStrictEqual(value, expected), label);
const lexical = (left, right) => left < right ? -1 : left > right ? 1 : 0;
const text = row => new TextDecoder('utf-8', {fatal: true}).decode(readAdmitted(row));
function identity() {
  return {format: METADATA_FORMAT, json_encoding: JSON_ENCODING,
    sources: [self, shared].map(filename => ({path: filename, sha256: fileSHA256(filename)}))};
}
function receipt(record) { return {...record, metadata_implementation: identity()}; }
function immutableJSON(filename, record) { return writeJSON(filename, receipt(record), {sortKeys: false}); }
function exactProtocol(protocol) {
  require(Object.hasOwn(ADMISSIONS, protocol), 'supported fixed admission protocol required');
  return ADMISSIONS[protocol];
}
const ADMISSIONS = Object.freeze({
  'context-deletion-v1': {
    augmented_cuda_contract: 'context_deletion_adapter_test: inputs/loss/gradients/updates, default-path parity and frozen inference',
    other_gates: ['pure context masks/loss', 'training policy compatibility/resume rejection',
      'shared protocol defaults/custom namespaces', 'ordinary CPU modes/resume', 'legacy checkpoint exports'],
    verify_marker: 'RPB context-deletion CUDA/default-parity tests passed'},
  'context-optimization-validation-v1': {
    augmented_cuda_contract: 'fresh exact-parent replay gate; uninterrupted live AdamW continuation; hidden-value and snapshot isolation',
    other_gates: ['original enabled/default context CUDA contracts', 'tagged ordinary resume rejection',
      'optional validation views reuse ordinary TRAIN readouts', 'pure deletion stream compatibility',
      'legacy stress remains TEST-only', 'paired protocol defaults']},
  'context-replication-v1': {cuda: true,
    gates: ['TRAIN/VALIDATION-only preparation and all-point isolation', 'default native and paired compatibility',
      'fresh equal mode2 initialization/scaler/data/counter CUDA contract', 'unchanged context training policy',
      'tagged ordinary resume rejection', 'fixed readout TEST-only corruption', 'actual CUDA/native32 serving gate']},
  'context-lighter-validation-v1': {cuda: true,
    gates: ['actual CUDA .15 training and native32 serving', 'strict declared policy and full initialization pairing',
      'original query/loss support and deterministic repaired context', 'legacy disabled/.30 defaults and replay',
      'tagged ordinary resume rejection', 'development-only completion and default paired protocols',
      'fixed original-TRAIN heads reused across validation views']},
  'context-balanced-validation-v1': {cuda: true,
    gates: ['actual CUDA balanced policy and native32 serving', 'strict policy/schedule and full initialization pairing',
      'absolute-attempt ordinary/deletion counts and skip abort', 'original query/loss support and repaired context',
      'exact live continuation across saved snapshots', 'legacy disabled/.30/.15 defaults and v6-only replay',
      'tagged ordinary resume rejection', 'development-only completion and shared fixed readouts']},
  'native-view-agreement-validation-v1': {cuda: true, marker: 'Native view agreement CUDA admission passed',
    gates: ['actual CUDA native-view-agreement objective and native32 serving', 'strict objective/calibration policy and full initialization pairing',
      'ordinary reconstruction anchor, detached target, fixed TRAIN calibration and skip abort', 'original query/loss support and repaired context',
      'exact live continuation across saved snapshots and immutable calibration', 'legacy disabled/.30/.15/balanced defaults and v6-only replay',
      'tagged ordinary resume rejection', 'development-only completion and shared fixed readouts']},
  'training-objective-diagnostic-v1': {objective: true, marker: 'TRAIN objective CUDA admission passed',
    gates: ['actual CUDA parameters, input, loss and gradients', 'zero weight and buffer updates',
      'original support and hierarchical pair-Huber math', 'both frozen Ridge affine maps and descent sign',
      'real encoder gradients separate from frozen latent leaf', 'explicit195 TRAIN role metadata and SHA256']},
  'v7-decoder-calibration-v1': {decoder: true, marker: 'V7 decoder calibration CUDA admission passed',
    gates: ['closed-role admission before hashes; redirects and mutation rejection',
      'actual CUDA decoder-only finite gradients and updates; encoder/native/scaler exact',
      'original Q and hidden-value isolation; direct/split AdamW parity',
      'typed immutable snapshots and parent/data/policy/seed/overwrite rejection']},
});

export function checkCard(protocol, repo) {
  canonicalDirectory(repo);
  require(Object.hasOwn(CARD_HASHES, protocol), 'supported frozen card required');
  const [name, expected] = CARD_HASHES[protocol];
  require(fileSHA256(path.join(repo, 'code/embeddings/evaluation/cards', name), {root: repo}) === expected, 'frozen human card differs');
}
export function passAdmission(protocol, root, source) {
  const config = exactProtocol(protocol);
  canonicalDirectory(root); digest(source);
  const names = ['build-and-tests.log'];
  if (config.cuda) names.push('cuda-gate.log');
  if (config.objective) names.push('metadata-gate.log', 'source-inputs.sha256');
  else if (!config.decoder) names.push('admission-source-inputs.sha256');
  const admitted = admitFiles(names.map(name => path.join(root, name)), {root});
  const rows = new Map(names.map((name, index) => [name, admitted[index]]));
  const log = text(rows.get('build-and-tests.log'));
  if (config.marker) require(log.includes(config.marker), 'actual CUDA admission marker absent');
  if (config.decoder) require(log.includes(source) && log.includes('Frozen role guard checks passed'), 'decoder source/role admission marker absent');
  const record = {protocol, status: 'passed', source_fingerprint: source};
  if (CARD_HASHES[protocol]) record.human_card_sha256 = CARD_HASHES[protocol][1];
  record.source_preserved = true;
  if (config.objective) Object.assign(record, {payload_files_read: 0, encoder_updates: 0, head_fits: 0});
  if (config.decoder) Object.assign(record, {quality_payloads_read: 0, new_quality_encoder_updates: 0, head_fits: 0});
  for (const key of ['augmented_cuda_contract', 'other_gates', 'gates']) if (config[key]) record[key] = config[key];
  if (config.objective) record.logs = Object.fromEntries(names.map(name => [name, hashAdmitted(rows.get(name))]));
  else {
    record.log_sha256 = hashAdmitted(rows.get('build-and-tests.log'));
    if (config.cuda) record.cuda_gate_log_sha256 = hashAdmitted(rows.get('cuda-gate.log'));
    if (!config.decoder) record.admission_source_manifest_sha256 = hashAdmitted(rows.get('admission-source-inputs.sha256'));
  }
  immutableJSON(path.join(root, 'passed.json'), record);
  return record;
}
function parseManifest(source) {
  const seen = new Set();
  return source.trimEnd().split('\n').map(line => {
    require(/^[0-9a-f]{64}  .+$/.test(line), 'exact source checksum row required');
    const relative = safeRelative(line.slice(66));
    require(!seen.has(relative), 'duplicate source checksum path'); seen.add(relative);
    return {sha256: line.slice(0, 64), relative};
  });
}
export function verifyAdmission(protocol, root, parent, source) {
  const config = exactProtocol(protocol);
  canonicalDirectory(parent); canonicalDirectory(root, {root: parent}); digest(source);
  require(path.dirname(root) === parent && path.basename(root).startsWith('admission-'), 'admission pointer escapes fixed admission root');
  const record = readJSON(path.join(root, 'passed.json'), {root});
  require(record.protocol === protocol && record.status === 'passed' && record.source_fingerprint === source &&
    record.source_preserved === true, 'admission protocol/source/status differs');
  if (CARD_HASHES[protocol]) require(record.human_card_sha256 === CARD_HASHES[protocol][1], 'admission human card differs');
  if (config.objective) require(record.payload_files_read === 0 && record.encoder_updates === 0 && record.head_fits === 0, 'metadata admission accessed quality payloads');
  if (config.decoder) require(record.quality_payloads_read === 0 && record.head_fits === 0, 'decoder admission accessed quality payloads');
  const checks = config.objective ? Object.entries(record.logs) : [
    ['build-and-tests.log', record.log_sha256],
    ...(config.cuda ? [['cuda-gate.log', record.cuda_gate_log_sha256]] : []),
    ...(!config.decoder ? [['admission-source-inputs.sha256', record.admission_source_manifest_sha256]] : []),
  ];
  if (config.objective) same(checks.map(([name]) => name).sort(), ['build-and-tests.log', 'metadata-gate.log', 'source-inputs.sha256'], 'closed objective admission log set');
  const manifestName = config.objective || config.decoder ? 'source-inputs.sha256' : 'admission-source-inputs.sha256';
  const identityName = config.objective || config.decoder ? manifestName : 'production-source-inputs.sha256';
  const metadata = [...new Set([...checks.map(([name]) => safeRelative(name)), manifestName, identityName])];
  const rows = admitFiles(metadata.map(name => path.join(root, name)), {root});
  const indexed = new Map(metadata.map((name, index) => [name, rows[index]]));
  for (const [name, expected] of checks) { digest(expected); require(hashAdmitted(indexed.get(name)) === expected, `admission changed: ${name}`); }
  require(hashAdmitted(indexed.get(identityName)) === source, 'admission production source identity differs');
  if (config.verify_marker) require(text(indexed.get('build-and-tests.log')).includes(config.verify_marker), 'original CUDA/default-parity marker absent');
  const manifest = parseManifest(text(indexed.get(manifestName))), sourceRoot = path.join(root, 'source');
  canonicalDirectory(sourceRoot, {root});
  const files = admitFiles(manifest.map(item => path.join(sourceRoot, item.relative)), {root: sourceRoot});
  files.forEach((file, index) => require(hashAdmitted(file) === manifest[index].sha256, 'copied admission source changed'));
  return record;
}

function inventoryIndex(inventory) {
  require(Array.isArray(inventory.files), 'artifact inventory file array required');
  const rows = new Map();
  for (const item of inventory.files) {
    safeRelative(item.path); digest(item.sha256);
    require(Number.isSafeInteger(item.bytes) && item.bytes >= 0 && !rows.has(item.path), 'unique exact artifact inventory entries required');
    rows.set(item.path, item);
  }
  return rows;
}
export function retainedPaths(inventory, mode) {
  const indexed = inventoryIndex(inventory);
  if (mode === 'native') return [...indexed.keys()].filter(relative => {
    if (['results/native-curve-card.json', 'results/selection.json'].includes(relative)) return true;
    const parts = relative.split('/');
    return parts.length >= 3 && parts[0] === 'results' &&
      ['seed-3101-lag_sign', 'seed-3202-lag_sign', 'seed-3303-lag_sign'].includes(parts[1]) &&
      !parts.some(part => part.includes('testing') || part.includes('stress') || part.includes('selected-test')) &&
      !parts.some(part => ['milestone-128', 'milestone-2048'].includes(part));
  });
  require(mode === 'candidate-512', 'supported retained inventory recipe required');
  const allowed = new Set(['results/validation-report.json']);
  for (const master of [3101, 3202, 3303]) {
    const base = `results/seed-${master}-lag_sign`, point = `${base}/candidate-milestone-512`;
    for (const name of ['controlled-training.pt', 'controlled-validation.pt', 'candidate-training-reconstruction.pt', 'candidate-validation-reconstruction.pt']) allowed.add(`${base}/${name}`);
    for (const name of ['checkpoint.pt', 'checkpoint.pt.audit.pt', 'checkpoint.pt.scaler.pt', 'checkpoint.pt.training-raw.pt', 'native-training.pt', 'native-validation.pt', 'point.json']) allowed.add(`${point}/${name}`);
    for (const repetition of ['rep-1', 'rep-2', 'rep-3']) for (const name of ['fit.pt', 'validation-predictions.pt']) allowed.add(`${point}/${repetition}/${name}`);
  }
  require([...allowed].every(relative => indexed.has(relative)), 'missing declared retained inputs');
  return [...allowed].sort();
}
export function writeRetained(inventoryFile, output, mode) {
  const inventory = readJSON(inventoryFile), indexed = inventoryIndex(inventory);
  const rows = retainedPaths(inventory, mode);
  writeNew(output, rows.map(relative => `${indexed.get(relative).sha256}  ${relative}\n`).join(''));
  immutableJSON(`${output}.metadata.json`, {role: 'retained-input-allowlist', recipe: mode,
    inventory_sha256: fileSHA256(inventoryFile), manifest_sha256: fileSHA256(output), payload_files_read: 0});
}
function walkFiles(root, {forbidden = false} = {}) {
  canonicalDirectory(root);
  const files = [];
  const visit = directory => {
    for (const name of fs.readdirSync(directory).sort()) {
      if (forbidden) require(!name.includes('testing') && !name.includes('stress') && name !== 'selection.json', 'forbidden reference artifact');
      const filename = path.join(directory, name), info = fs.lstatSync(filename);
      require(!info.isSymbolicLink(), 'artifact inventory redirect');
      if (info.isDirectory()) { canonicalDirectory(filename, {root}); visit(filename); }
      else { require(info.isFile(), 'inventory accepts regular files only'); files.push(filename); }
    }
  };
  visit(root); return files.sort();
}
function recordsFor(root, filenames) {
  return admitFiles(filenames, {root, singleLink: true}).map(file => ({path: path.relative(root, file.path).split(path.sep).join('/'), bytes: file.bytes, sha256: hashAdmitted(file)}));
}
export function inventoryCapsule(protocol, root, {now = () => new Date().toISOString()} = {}) {
  exactProtocol(protocol);
  require(!fs.existsSync(path.join(root, 'artifact-integrity.json')), 'inventory must exclude itself and never overwrite');
  const records = recordsFor(root, walkFiles(root));
  immutableJSON(path.join(root, 'artifact-integrity.json'), {schema_version: 1, protocol,
    created_utc: now(), checksum_algorithm: 'sha256-file-bytes', inventory_excludes_itself: true,
    files: records, file_count: records.length, total_bytes: records.reduce((sum, item) => sum + item.bytes, 0)});
}
export function freezeReference(root, manifestFile, recordFile) {
  const card = readJSON(path.join(root, 'native-development-card.json'), {root});
  const completion = readJSON(path.join(root, 'development-complete.json'), {root});
  require(card.protocol === 'native-development-v1' && card.test_pairs === 0 && card.development_only === true &&
    card.selection_performed === false && card.testing_generated === false && card.stress_evaluated === false, 'reference is not development-only');
  require(completion.protocol === 'native-development-v1' && completion.all_declared_milestones_complete === true &&
    completion.all_retained_witnesses_exact === true && completion.controls_immutable === true && completion.selection_performed === false &&
    completion.testing_generated === false && completion.stress_evaluated === false && completion.cohorts === 5 && completion.checked_points === 10, 'incomplete development reference');
  same(completion.milestones, [0, 512], 'reference milestone recipe differs'); same(card.masters, MASTERS, 'reference cohort recipe differs');
  const records = recordsFor(root, walkFiles(root, {forbidden: true}));
  writeNew(manifestFile, records.map(row => `${row.sha256}  ${row.path}\n`).join(''));
  immutableJSON(recordFile, {protocol: 'context-replication-v1', role: 'reference-preparation',
    checksum_algorithm: 'sha256-file-bytes', files: records, frozen_before_candidate_and_testing: true,
    manifest_sha256: fileSHA256(manifestFile)});
}

export const PARENTS = Object.freeze({
  v4: {capsule: 'output/runs/rpb-context-replication/context-replication-JjNEUc', inventory_sha256: '13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9', protocol: 'context-replication-v1'},
  v7: {capsule: 'output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2', inventory_sha256: 'd4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821', protocol: 'context-lighter-validation-v1'},
});
export function closedNativeRoles() {
  const roles = new Set();
  for (const id of ['v4', 'v7']) {
    const prefix = id === 'v4' ? 'reference' : 'results/candidate-development';
    for (const name of ['native-development-card.json', 'development-complete.json', 'validation-report.json']) roles.add(`${id}/${prefix}/${name}`);
    for (const master of MASTERS) {
      const base = `${prefix}/seed-${master}-lag_sign`;
      for (const name of ['controlled-training.pt', 'controlled-validation.pt', 'development-manifest.json', 'trainer-audit.json']) roles.add(`${id}/${base}/${name}`);
      for (const budget of [0, 512]) {
        const point = `${base}/milestone-${budget}`;
        for (const suffix of ['', '.audit.pt', '.scaler.pt', '.training-raw.pt']) roles.add(`${id}/${point}/checkpoint.pt${suffix}`);
        for (const name of ['native-training.pt', 'native-validation.pt', 'point.json', 'training-reconstruction.pt', 'validation-reconstruction.pt']) roles.add(`${id}/${point}/${name}`);
        if (budget === 512) for (const rep of ['rep-1', 'rep-2', 'rep-3']) for (const name of ['fit.pt', 'training-predictions.pt', 'validation-predictions.pt']) roles.add(`${id}/${point}/${rep}/${name}`);
      }
      if (id === 'v7') for (const suffix of ['.pt', '.json']) roles.add(`${id}/results/validation-views/seed-${master}-lag_sign/validation-dropout-030${suffix}`);
    }
  }
  return roles;
}
function allowedRole(role, {id, strict = false} = {}) {
  const relative = safeRelative(role.path), parts = relative.split('/');
  require(!parts.some(part => part.includes('testing') || part.includes('stress') || part === 'report.json'), 'forbidden parent role');
  require(typeof role.role === 'string' && role.role.length > 0, 'named parent role required');
  if (id) require(role.tag === (id === 'v4' ? 'RPB-v4' : 'RPB-v7'), 'parent tag differs');
  else require(['reference', 'results'].includes(parts[0]) && ['RPB-v4', 'RPB-v6'].includes(role.tag), 'lighter parent role/tag differs');
  require(MASTERS.includes(role.master_seed) || (role.master_seed === null && role.role === 'development-metadata' && (id || role.tag === 'RPB-v4')), 'parent master binding differs');
  require([0, 512, null].includes(role.budget), 'parent budget binding differs');
  if (strict) {
    const expectedMaster = MASTERS.find(master => relative.includes(`/seed-${master}-lag_sign/`)) ?? null;
    const expectedBudget = relative.includes('/milestone-0/') ? 0 : relative.includes('/milestone-512/') ? 512 : null;
    require(role.master_seed === expectedMaster && role.budget === expectedBudget && (expectedMaster !== null || role.role === 'development-metadata'), 'closed parent role metadata differs');
  }
  return relative;
}
export function validateParentRoles(protocol, plan) {
  require(['context-lighter-validation-v1', 'context-balanced-validation-v1', 'native-view-agreement-validation-v1'].includes(protocol), 'fixed parent binding protocol required');
  require(plan.protocol === protocol && plan.payload_access === false && plan.testing_accessed === false && plan.stress_accessed === false, 'parent plan admits forbidden payload scope');
  same(plan.masters, MASTERS, 'parent cohort recipe differs');
  require(Array.isArray(plan.inputs) && plan.inputs.length > 0, 'explicit parent input roles required');
  const lighter = protocol === 'context-lighter-validation-v1', native = protocol === 'native-view-agreement-validation-v1';
  const roles = [...plan.inputs].sort((left, right) => lexical(`${left.parent_id ?? 'v4'}/${left.path}`, `${right.parent_id ?? 'v4'}/${right.path}`));
  const keys = new Set();
  for (const role of roles) {
    const id = lighter ? 'v4' : role.parent_id;
    require(Object.hasOwn(PARENTS, id), 'unknown input parent');
    const relative = allowedRole(role, {id: lighter ? undefined : id, strict: native});
    const key = lighter ? relative : `${id}/${relative}`;
    require(!keys.has(key) && (lighter || !Object.hasOwn(role, 'manifest_path') || role.manifest_path === key), 'duplicate or redirected parent manifest role');
    keys.add(key);
  }
  if (!lighter) {
    require(keys.size === 326 && roles.filter(row => row.parent_id === 'v4').length === 158 && roles.filter(row => row.parent_id === 'v7').length === 168, 'exact parent role counts required');
    if (native) same([...keys].sort(), [...closedNativeRoles()].sort(), 'closed native parent role matrix differs');
  }
  return roles;
}
// historicalRoot is the explicit preserved component root containing output/.
// Capsule identities in plans/inventories remain the frozen output/... paths.
export function freezeParents(protocol, historicalRoot, capsule) {
  require(['context-lighter-validation-v1', 'context-balanced-validation-v1', 'native-view-agreement-validation-v1'].includes(protocol), 'fixed parent binding protocol required');
  canonicalDirectory(historicalRoot); canonicalDirectory(capsule);
  const planFile = path.join(capsule, 'parent-role-plan.json'), planRow = admitFiles([planFile], {root: capsule})[0];
  const plan = parseJSON(readAdmitted(planRow));
  const roles = validateParentRoles(protocol, plan);
  const lighter = protocol === 'context-lighter-validation-v1';
  const declared = lighter ? [{id: 'v4', ...PARENTS.v4}] : plan.parents;
  if (lighter) require(plan.parent_capsule === PARENTS.v4.capsule, 'lighter parent identity differs');
  require(Array.isArray(declared) && declared.length === (lighter ? 1 : 2), 'complete exact parent set required');
  const parents = new Map(), inventoryPaths = [];
  for (const item of declared) {
    require(Object.hasOwn(PARENTS, item.id) && !parents.has(item.id), 'duplicate or unknown parent identity');
    const frozen = PARENTS[item.id];
    require(item.capsule === frozen.capsule && item.inventory_sha256 === frozen.inventory_sha256, 'frozen parent identity differs');
    const directory = canonicalDirectory(path.join(historicalRoot, frozen.capsule), {root: historicalRoot});
    parents.set(item.id, {item, directory}); inventoryPaths.push(path.join(directory, 'artifact-integrity.json'));
  }
  const metadata = admitFiles(inventoryPaths, {root: historicalRoot});
  [...parents.values()].forEach((parent, index) => {
    const row = metadata[index], frozen = PARENTS[parent.item.id];
    require(hashAdmitted(row) === frozen.inventory_sha256, 'original parent inventory differs');
    const inventory = parseJSON(readAdmitted(row));
    require(inventory.protocol === frozen.protocol && inventory.inventory_excludes_itself === true, 'parent inventory protocol differs');
    parent.indexed = inventoryIndex(inventory);
  });
  const pending = [];
  for (const role of roles) {
    const id = lighter ? 'v4' : role.parent_id;
    require(parents.has(id), 'unknown input parent');
    const relative = safeRelative(role.path);
    const key = lighter ? relative : `${id}/${relative}`;
    const parent = parents.get(id), original = parent.indexed.get(relative);
    require(original, 'parent role absent from frozen inventory');
    pending.push({role, key, filename: path.join(parent.directory, relative), original});
  }
  // Validate the entire path/inode matrix before opening the first tensor.
  const admitted = admitFiles(pending.map(item => item.filename), {root: historicalRoot, singleLink: true});
  const records = admitted.map((file, index) => {
    const {role, key, original} = pending[index], actual = hashAdmitted(file);
    require(file.bytes === original.bytes && actual === original.sha256, 'parent payload differs from original frozen inventory');
    return {...role, ...(!lighter ? {manifest_path: key} : {}), absolute_path: file.path, bytes: file.bytes, sha256: actual};
  });
  metadata.forEach((row, index) => require(hashAdmitted(row) === [...parents.values()][index].item.inventory_sha256, 'parent metadata changed during binding'));
  hashAdmitted(planRow);
  const manifestFile = path.join(capsule, 'parent-inputs.sha256');
  writeNew(manifestFile, records.map((row, index) => `${row.sha256}  ${pending[index].key}\n`).join(''));
  const record = {protocol, role: 'known-development-parent-inputs', checksum_algorithm: 'sha256-file-bytes',
    historical_parent_root: historicalRoot, frozen_before_new_fitting: true, testing_accessed: false, stress_accessed: false,
    manifest_sha256: fileSHA256(manifestFile), role_plan_sha256: hashAdmitted(planRow), inputs: records};
  if (lighter) Object.assign(record, {parent: parents.get('v4').directory, parent_inventory_sha256: PARENTS.v4.inventory_sha256,
    parent_source_fingerprint: '9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828',
    parent_training_producer: '2fff50c48605ee21dbbfd28d6989bdcf6b5cd057785c2acb19419571af48b895',
    parent_core_writer: '587f2423758c3e70c2c7665d9e7b59bfe10af4c7f9aaba23f5318b3c55b13bca'});
  else record.parents = [...parents.values()].map(({item, directory}) => ({...item, absolute_path: directory}));
  immutableJSON(path.join(capsule, 'parent-inputs.json'), record);
}
export function verifyParents(protocol, capsule) {
  const record = readJSON(path.join(capsule, 'parent-inputs.json'), {root: capsule});
  require(record.protocol === protocol && record.frozen_before_new_fitting === true && record.testing_accessed === false && record.stress_accessed === false, 'parent receipt scope differs');
  canonicalDirectory(record.historical_parent_root);
  require(fileSHA256(path.join(capsule, 'parent-inputs.sha256')) === record.manifest_sha256 &&
    fileSHA256(path.join(capsule, 'parent-role-plan.json')) === record.role_plan_sha256, 'parent binding metadata changed');
  const matrix = admitFiles(record.inputs.map(item => item.absolute_path), {root: record.historical_parent_root, singleLink: true});
  matrix.forEach((file, index) => {
    require(file.bytes === record.inputs[index].bytes && hashAdmitted(file) === record.inputs[index].sha256, 'parent payload changed');
  });
  immutableJSON(path.join(capsule, 'parent-preserved-after.json'), {protocol, status: 'passed', parent_inputs: matrix.length, all_allowed_bytes_preserved: true});
}
export function main(argv = process.argv.slice(2)) {
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv') && process.cwd() === '/mityane' &&
    self === '/mityane/code/embeddings/scripts/runner-metadata.mjs', 'run through the managed container source entry');
  const [command, ...args] = argv;
  const arity = count => require(args.length === count, `fixed ${command} arguments required`);
  if (command === 'check-card') { arity(2); checkCard(...args); }
  else if (command === 'pass-admission') { arity(3); passAdmission(...args); }
  else if (command === 'verify-admission') { arity(4); verifyAdmission(...args); }
  else if (command === 'retained-inputs') { arity(3); writeRetained(...args); }
  else if (command === 'freeze-reference') { arity(3); freezeReference(...args); }
  else if (command === 'freeze-parents') { arity(3); freezeParents(...args); }
  else if (command === 'verify-parents') { arity(2); verifyParents(...args); }
  else if (command === 'inventory') { arity(2); inventoryCapsule(...args); }
  else if (command === 'sdk-proof') { arity(0); process.stdout.write(`Container SDK proof: ${jsonBytes(readJSON('/opt/cuwacunu_embedding/setup/libtorch-installed.json')).toString('utf8')}`); }
  else throw new Error('unknown fixed runner metadata operation');
}
if (process.argv[1] && path.resolve(process.argv[1]) === self) {
  try { main(); } catch (error) { process.stderr.write(`Runner metadata refused: ${error.message}\n`); process.exitCode = 1; }
}
