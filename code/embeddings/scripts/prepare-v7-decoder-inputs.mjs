#!/usr/bin/env node
// Copyright (c) 2026 Waajacu. See ../../../LICENSE.
// Maintained source cutover; managed-container validation is still pending.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {
  JSON_ENCODING, admitFiles, canonicalDirectory, fileSHA256, hashAdmitted, inside,
  jsonBytes, parseJSON, readAdmitted, readJSON, requireCondition as require,
  safeRelative, sha256, writeJSON, writeNew,
} from '../../shared/artifacts/node/v1/files.mjs';

export const PROTOCOL = 'v7-decoder-calibration-v1';
export const BINDING_FORMAT = 'v7-decoder-inputs-node-v1';
export const MASTERS = Object.freeze([4404, 5505, 6606, 7707, 8808]);
export const PARENTS = Object.freeze({
  v4: Object.freeze({capsule: 'output/runs/rpb-context-replication/context-replication-JjNEUc',
    inventorySHA256: '13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9', prefix: 'reference'}),
  v7: Object.freeze({capsule: 'output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2',
    inventorySHA256: 'd4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821', prefix: 'results/candidate-development'}),
});
export function template() {
  const roles = {};
  for (const [identity, {capsule, prefix}] of Object.entries(PARENTS)) {
    const names = ['native-development-card.json', 'validation-report.json'].map(name => `${prefix}/${name}`);
    if (identity === 'v7') names.push(`${prefix}/development-complete.json`);
    for (const master of MASTERS) {
      const base = `${prefix}/seed-${master}-lag_sign`, point = `${base}/milestone-512`;
      if (identity === 'v7') {
        names.push(...['controlled-training.pt', 'controlled-validation.pt', 'development-manifest.json', 'trainer-audit.json']
          .map(name => `${base}/${name}`));
        names.push(...['checkpoint.pt', 'checkpoint.pt.audit.pt', 'checkpoint.pt.scaler.pt', 'checkpoint.pt.training-raw.pt',
          'native-training.pt', 'native-validation.pt'].map(name => `${point}/${name}`));
      }
      names.push(...['point.json', 'training-reconstruction.pt', 'validation-reconstruction.pt'].map(name => `${point}/${name}`));
    }
    for (const name of names) roles[`${identity}/${name}`] = `${capsule}/${name}`;
  }
  require(Object.keys(roles).length === 85, 'exact 85 closed legal inputs');
  return roles;
}
export function validateRoleRows(rows) {
  const expected = template();
  require(Array.isArray(rows) && rows.length === 85, 'closed 85-role matrix required');
  const names = new Set();
  for (const row of rows) {
    require(row && typeof row.manifest_path === 'string' && !names.has(row.manifest_path) &&
      Object.hasOwn(expected, row.manifest_path) && row.relative_path === expected[row.manifest_path], 'exact unique closed role binding');
    safeRelative(row.manifest_path); safeRelative(row.relative_path); names.add(row.manifest_path);
  }
  return [...rows].sort((a, b) => a.manifest_path < b.manifest_path ? -1 : a.manifest_path > b.manifest_path ? 1 : 0);
}
function context(repoRoot, capsule, historicalParentRoot) {
  canonicalDirectory(repoRoot); canonicalDirectory(capsule); canonicalDirectory(historicalParentRoot);
  const outputRoot = path.join(repoRoot, '.temp/embeddings/runs/rpb-v7-decoder-calibration');
  require(capsule !== outputRoot && inside(outputRoot, capsule), 'new .temp decoder capsule required');
  for (const parent of Object.values(PARENTS)) require(!inside(path.join(historicalParentRoot, parent.capsule), capsule),
    'never write into historical parent');
}
function inventoryRows(historicalParentRoot) {
  const parents = Object.entries(PARENTS);
  const admitted = admitFiles(parents.map(([, parent]) => path.join(historicalParentRoot, parent.capsule, 'artifact-integrity.json')),
    {root: historicalParentRoot});
  const inventories = new Map();
  for (const [index, [identity, parent]] of parents.entries()) {
    const raw = readAdmitted(admitted[index]);
    require(sha256(raw) === parent.inventorySHA256, `pinned ${identity} inventory metadata`);
    const inventory = parseJSON(raw, {integers: 'number'});
    require(inventory.inventory_excludes_itself === true && Array.isArray(inventory.files), 'parent inventory contract');
    const rows = new Map();
    for (const row of inventory.files) {
      require(row && typeof row.path === 'string' && !rows.has(row.path), 'unambiguous parent inventory paths');
      rows.set(row.path, row);
    }
    inventories.set(identity, rows);
  }
  return {inventories, admitted};
}
const validEntry = row => row && Number.isSafeInteger(row.bytes) && row.bytes >= 0 &&
  typeof row.sha256 === 'string' && /^[0-9a-f]{64}$/.test(row.sha256);
export function freeze(repoRoot, capsule, {historicalParentRoot} = {}) {
  context(repoRoot, capsule, historicalParentRoot);
  for (const name of ['parent-inputs.sha256', 'parent-inputs.json']) {
    let occupied = true;
    try { fs.lstatSync(path.join(capsule, name)); }
    catch (error) { if (error.code === 'ENOENT') occupied = false; else throw error; }
    require(!occupied, `exclusive parent binding already exists: ${name}`);
  }
  const [planSource] = admitFiles([path.join(capsule, 'parent-role-plan.json')], {root: capsule, singleLink: true});
  const planBytes = readAdmitted(planSource), plan = parseJSON(planBytes, {integers: 'number'});
  require(plan.protocol === PROTOCOL && plan.payload_access === false &&
    plan.testing_accessed === false && plan.stress_accessed === false, 'metadata-only closed role plan');
  const rows = validateRoleRows(plan.inputs), {inventories, admitted: metadataAdmissions} = inventoryRows(historicalParentRoot);
  const pending = rows.map(item => {
    const [identity, ...parts] = item.manifest_path.split('/'), relative = parts.join('/');
    const original = inventories.get(identity).get(relative);
    require(validEntry(original), 'exact pinned parent inventory role');
    return {item, original, absolute: path.join(historicalParentRoot, item.relative_path)};
  });
  // No payload bytes are read until every declared role/path/alias is admitted.
  const admissions = admitFiles(pending.map(item => item.absolute), {root: historicalParentRoot, singleLink: true});
  // Check the whole size matrix before hashing even the first allowed payload.
  for (const [index, admitted] of admissions.entries()) require(admitted.bytes === pending[index].original.bytes,
    'allowed parent byte count matches frozen inventory');
  const records = [];
  for (const [index, {item, original, absolute}] of pending.entries()) {
    const admitted = admissions[index];
    require(hashAdmitted(admitted) === original.sha256, 'allowed parent bytes match frozen inventory');
    records.push({...item, absolute_path: absolute, bytes: admitted.bytes, sha256: original.sha256});
  }
  // These re-reads are metadata only, and catch changes before publishing evidence.
  for (const admitted of metadataAdmissions) readAdmitted(admitted);
  require(readAdmitted(planSource).equals(planBytes), 'role plan unchanged before evidence publication');
  const manifest = Buffer.from(records.map(row => `${row.sha256}  ${row.manifest_path}\n`).join(''), 'utf8');
  const record = {protocol: PROTOCOL, binding_format: BINDING_FORMAT, json_encoding: JSON_ENCODING,
    historical_parent_root: historicalParentRoot, checksum_algorithm: 'sha256-file-bytes', closed_roles: true,
    frozen_before_new_fitting: true, testing_accessed: false, stress_accessed: false,
    parents: Object.fromEntries(Object.entries(PARENTS).map(([identity, parent]) => [identity,
      {capsule: parent.capsule, inventory_sha256: parent.inventorySHA256}])),
    inputs: records, manifest_sha256: sha256(manifest), role_plan_sha256: sha256(planBytes)};
  writeNew(path.join(capsule, 'parent-inputs.sha256'), manifest);
  writeJSON(path.join(capsule, 'parent-inputs.json'), record, {sortKeys: false});
  return record;
}
export function verify(repoRoot, capsule, {historicalParentRoot} = {}) {
  context(repoRoot, capsule, historicalParentRoot);
  const record = readJSON(path.join(capsule, 'parent-inputs.json'), {root: capsule, singleLink: true, integers: 'number'});
  require(record.protocol === PROTOCOL && record.binding_format === BINDING_FORMAT &&
    record.json_encoding === JSON_ENCODING && record.historical_parent_root === historicalParentRoot &&
    record.closed_roles === true && record.checksum_algorithm === 'sha256-file-bytes' &&
    record.frozen_before_new_fitting === true && record.testing_accessed === false && record.stress_accessed === false,
  'new explicit Node parent-binding identity required');
  require(fileSHA256(path.join(capsule, 'parent-inputs.sha256')) === record.manifest_sha256 &&
    fileSHA256(path.join(capsule, 'parent-role-plan.json')) === record.role_plan_sha256, 'frozen metadata evidence unchanged');
  for (const [identity, parent] of Object.entries(PARENTS)) require(record.parents?.[identity]?.capsule === parent.capsule &&
    record.parents[identity].inventory_sha256 === parent.inventorySHA256, 'original parent identity preserved');
  const rows = validateRoleRows(record.inputs);
  const {inventories, admitted: metadataAdmissions} = inventoryRows(historicalParentRoot);
  const pending = rows.map(row => {
    require(validEntry(row) && row.absolute_path === path.join(historicalParentRoot, row.relative_path), 'exact saved parent path binding');
    const [identity, ...parts] = row.manifest_path.split('/');
    const original = inventories.get(identity).get(parts.join('/'));
    require(validEntry(original) && original.bytes === row.bytes && original.sha256 === row.sha256,
      'saved binding still matches original pinned parent inventory');
    return row.absolute_path;
  });
  const admissions = admitFiles(pending, {root: historicalParentRoot, singleLink: true});
  for (const [index, admitted] of admissions.entries()) require(admitted.bytes === rows[index].bytes,
    'allowed parent byte count preserved');
  for (const [index, admitted] of admissions.entries()) require(hashAdmitted(admitted) === rows[index].sha256,
    'allowed parent input preserved');
  const expectedManifest = Buffer.from(rows.map(row => `${row.sha256}  ${row.manifest_path}\n`).join(''), 'utf8');
  require(sha256(expectedManifest) === record.manifest_sha256, 'exact closed manifest rows');
  for (const admitted of metadataAdmissions) readAdmitted(admitted);
  return {status: 'verified', protocol: PROTOCOL, binding_format: BINDING_FORMAT,
    legal_parent_inputs: 85, testing_accessed: false, stress_accessed: false};
}
export function main(args = process.argv.slice(2)) {
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv') &&
    fileURLToPath(import.meta.url) === '/mityane/code/embeddings/scripts/prepare-v7-decoder-inputs.mjs',
  'run only through container.ps1 in mityane-geometrics');
  const {values} = parseArgs({args, options: {
    'repo-root': {type: 'string', default: '/mityane'}, capsule: {type: 'string'},
    'historical-parent-root': {type: 'string'}, verify: {type: 'boolean'},
  }});
  require(values.capsule && values['historical-parent-root'], 'explicit --capsule and --historical-parent-root required');
  const repoRoot = canonicalDirectory(path.resolve(values['repo-root']));
  const capsule = canonicalDirectory(path.resolve(values.capsule));
  const options = {historicalParentRoot: canonicalDirectory(path.resolve(values['historical-parent-root']))};
  process.stdout.write(jsonBytes((values.verify ? verify : freeze)(repoRoot, capsule, options), {sortKeys: false}));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
