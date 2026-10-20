// Source-only migration tests. Execute only inside the managed Linux container.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {METADATA_FORMAT, MASTERS, PARENTS, closedNativeRoles, validateParentRoles,
  retainedPaths, writeRetained, passAdmission, verifyAdmission, freezeReference,
  inventoryCapsule} from '../runner-metadata.mjs';
import {readJSON, writeJSON} from '../../../shared/artifacts/node/v1/files.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const fixtureRoot = '/opt/mityane-geometrics/migration/runner-metadata-tests';
function fixture(t) {
  assert.equal(process.platform, 'linux');
  assert.ok(fs.existsSync('/.dockerenv'));
  fs.mkdirSync(fixtureRoot, {recursive: true});
  const root = fs.mkdtempSync(`${fixtureRoot}/fixture-`);
  t.after(() => { assert.ok(root.startsWith(`${fixtureRoot}/fixture-`)); fs.rmSync(root, {recursive: true}); });
  return root;
}
function rowsForNative() {
  return [...closedNativeRoles()].map(key => {
    const [parent_id, ...parts] = key.split('/'), relative = parts.join('/');
    const master = MASTERS.find(value => relative.includes(`/seed-${value}-lag_sign/`)) ?? null;
    return {parent_id, path: relative, tag: parent_id === 'v4' ? 'RPB-v4' : 'RPB-v7',
      role: master === null ? 'development-metadata' : 'declared-development-payload', master_seed: master,
      budget: relative.includes('/milestone-0/') ? 0 : relative.includes('/milestone-512/') ? 512 : null};
  });
}
function nativePlan() {
  return {protocol: 'native-view-agreement-validation-v1', payload_access: false, testing_accessed: false,
    stress_accessed: false, masters: [...MASTERS], inputs: rowsForNative()};
}

test('native parent matrix contains exactly 326 TRAIN/VALIDATION roles and fixed original identities', () => {
  const plan = nativePlan(), roles = validateParentRoles(plan.protocol, plan);
  assert.equal(roles.length, 326);
  assert.equal(roles.filter(row => row.parent_id === 'v4').length, 158);
  assert.equal(roles.filter(row => row.parent_id === 'v7').length, 168);
  assert.equal(PARENTS.v4.inventory_sha256, '13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9');
  assert.equal(PARENTS.v7.inventory_sha256, 'd4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821');
  assert.ok(roles.every(row => !/testing|stress/.test(row.path)));
});
test('closed matrix rejects TEST poison, duplicates, missing roles, wrong seeds, budgets and manifest redirects without filesystem access', () => {
  const mutations = [
    plan => { plan.inputs[0].path = 'results/testing-poison.pt'; },
    plan => { plan.inputs[0] = {...plan.inputs[1]}; },
    plan => { plan.inputs.pop(); },
    plan => { plan.inputs.find(row => row.master_seed !== null).master_seed = 123; },
    plan => { plan.inputs.find(row => row.budget === 512).budget = 0; },
    plan => { plan.inputs[0].manifest_path = '../outside'; },
    plan => { plan.payload_access = true; },
  ];
  for (const mutate of mutations) {
    const plan = nativePlan(); mutate(plan);
    assert.throws(() => validateParentRoles(plan.protocol, plan));
  }
});
test('native retained allowlist uses metadata only and excludes TEST/stress/late milestones', () => {
  const paths = ['results/native-curve-card.json', 'results/selection.json',
    'results/seed-3101-lag_sign/milestone-512/native-training.pt',
    'results/seed-3101-lag_sign/milestone-128/native-training.pt',
    'results/seed-3101-lag_sign/milestone-2048/native-training.pt',
    'results/seed-3101-lag_sign/selected-test/poison.pt',
    'results/seed-3101-lag_sign/stress/poison.pt', 'results/seed-3101-lag_sign/testing.pt'];
  const inventory = {files: paths.map(relative => ({path: relative, bytes: 1, sha256: 'a'.repeat(64)}))};
  assert.deepEqual(retainedPaths(inventory, 'native'), paths.slice(0, 3));
  assert.throws(() => retainedPaths({files: [...inventory.files, inventory.files[0]]}, 'native'), /unique/);
  assert.throws(() => retainedPaths(inventory, 'candidate-512'), /missing declared/);
});
test('retained checksums retain exact plain text while recording a distinct metadata identity', t => {
  const root = fixture(t), inventory = path.join(root, 'inventory.json'), output = path.join(root, 'retained.sha256');
  writeJSON(inventory, {files: [{path: 'results/native-curve-card.json', bytes: 20, sha256: 'a'.repeat(64)}]});
  writeRetained(inventory, output, 'native');
  assert.equal(fs.readFileSync(output, 'utf8'), `${'a'.repeat(64)}  results/native-curve-card.json\n`);
  const record = readJSON(`${output}.metadata.json`);
  assert.equal(record.payload_files_read, 0);
  assert.equal(record.metadata_implementation.format, METADATA_FORMAT);
  assert.throws(() => writeRetained(inventory, output, 'native'), /EEXIST/);
});
test('admission verification binds log bytes, source identity and copied sources', t => {
  const parent = fixture(t), root = path.join(parent, 'admission-fixture');
  fs.mkdirSync(root); fs.mkdirSync(path.join(root, 'source'));
  fs.writeFileSync(path.join(root, 'source', 'fixture.cpp'), 'int fixture;\n');
  const manifest = `${hash('int fixture;\n')}  fixture.cpp\n`, source = hash(manifest);
  fs.writeFileSync(path.join(root, 'admission-source-inputs.sha256'), manifest);
  fs.writeFileSync(path.join(root, 'production-source-inputs.sha256'), manifest);
  fs.writeFileSync(path.join(root, 'build-and-tests.log'), 'RPB context-deletion CUDA/default-parity tests passed\n');
  passAdmission('context-deletion-v1', root, source);
  assert.equal(verifyAdmission('context-deletion-v1', root, parent, source).status, 'passed');
  assert.throws(() => verifyAdmission('context-deletion-v1', root, parent, '0'.repeat(64)), /protocol\/source/);
  fs.appendFileSync(path.join(root, 'source', 'fixture.cpp'), 'changed');
  assert.throws(() => verifyAdmission('context-deletion-v1', root, parent, source), /copied admission source changed/);
});
test('missing actual CUDA marker cannot publish an objective admission', t => {
  const root = fixture(t);
  for (const name of ['build-and-tests.log', 'metadata-gate.log', 'source-inputs.sha256']) fs.writeFileSync(path.join(root, name), 'unconfirmed');
  assert.throws(() => passAdmission('training-objective-diagnostic-v1', root, 'a'.repeat(64)), /marker absent/);
  assert.equal(fs.existsSync(path.join(root, 'passed.json')), false);
});
test('development reference refuses held-out paths before writing a manifest', t => {
  const capsule = fixture(t), root = path.join(capsule, 'reference'); fs.mkdirSync(root);
  writeJSON(path.join(root, 'native-development-card.json'), {protocol: 'native-development-v1', test_pairs: 0,
    development_only: true, selection_performed: false, testing_generated: false, stress_evaluated: false, masters: [...MASTERS]});
  writeJSON(path.join(root, 'development-complete.json'), {protocol: 'native-development-v1', all_declared_milestones_complete: true,
    all_retained_witnesses_exact: true, controls_immutable: true, selection_performed: false, testing_generated: false,
    stress_evaluated: false, cohorts: 5, checked_points: 10, milestones: [0, 512]});
  fs.mkdirSync(path.join(root, 'testing-poison'));
  const manifest = path.join(capsule, 'reference-inputs.sha256');
  assert.throws(() => freezeReference(root, manifest, path.join(capsule, 'reference-inputs.json')), /forbidden reference/);
  assert.equal(fs.existsSync(manifest), false);
});
test('capsule inventory is exclusive, excludes itself and rejects redirects', t => {
  const root = fixture(t); fs.writeFileSync(path.join(root, 'payload.txt'), 'fixture');
  inventoryCapsule('training-objective-diagnostic-v1', root, {now: () => '2026-10-18T00:00:00.000Z'});
  const record = readJSON(path.join(root, 'artifact-integrity.json'));
  assert.equal(record.file_count, 1); assert.equal(record.total_bytes, 7);
  assert.deepEqual(record.files.map(row => row.path), ['payload.txt']);
  assert.equal(record.files[0].sha256, hash('fixture'));
  assert.equal(record.created_utc, '2026-10-18T00:00:00.000Z');
  assert.throws(() => inventoryCapsule('training-objective-diagnostic-v1', root), /never overwrite/);
  const other = fixture(t); fs.symlinkSync(path.join(root, 'payload.txt'), path.join(other, 'redirect.txt'));
  assert.throws(() => inventoryCapsule('training-objective-diagnostic-v1', other), /redirect/);
});
