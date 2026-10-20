/** Artificial source/metadata fixtures only. Run through the managed container. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { hashAdmitted } from '../../shared/artifacts/node/v1/files.mjs';
import { freshDecoder, createPreparation, fileSha, sha256, manifestBytes, relativeName, admitMatrix, writeNew,
  readJson, writeJson, parseArgs, PARENT_IDENTITY, PARENT_SHA } from '../scripts/prepare-fresh-decoder-replication.mjs';
import { preparation as saved, declaredRows as savedRows, sourceSchemaCheck } from '../scripts/prepare-saved-native-reliability.mjs';
import { preparation as transfer, declaredRows as transferRows } from '../scripts/prepare-frozen-amplitude-transfer.mjs';
import { preparation as reliability } from '../scripts/prepare-early-mixer-reliability.mjs';
import { preparation as curve } from '../scripts/prepare-early-mixer-learning-curve.mjs';
import { preparation as gain } from '../scripts/prepare-matched-target-gain.mjs';
import { preparation as confirmation } from '../scripts/prepare-early-mixer-confirmation.mjs';
import { preparation as pooled } from '../scripts/prepare-pooled-context.mjs';

assert.equal(process.platform, 'linux', 'source fixtures run exclusively in Linux');
assert.ok(fs.existsSync('/.dockerenv'), 'existing managed container required');
const preparations = [freshDecoder, saved, transfer, reliability, curve, gain, confirmation, pooled];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const disposition = JSON.parse(fs.readFileSync(path.join(repo, 'doc/PYTHON_MIGRATION_DISPOSITION_2026-10-18.json'), 'utf8'));
const retiredPython = new Set(disposition.files.filter(row => row.disposition !== 'retain-scientific-python').map(row => row.path));

function fixture(t) {
  const parent = '/opt/mityane-geometrics/tests/python-migration';
  fs.mkdirSync(parent, { recursive: true });
  const root = fs.mkdtempSync(path.join(parent, 'preparation-source-'));
  t.after(() => fs.rmSync(root, { recursive: true }));
  return root;
}
function materialize(root, names, text = 'source-only fixture\n') {
  for (const name of names) { const file = path.join(root, name); fs.mkdirSync(path.dirname(file), { recursive: true }); writeNew(file, text); }
}
function plan(preparation) {
  return { ...structuredClone(preparation.config.plan), ...(preparation.config.planCardBinding ?
    { human_card_sha256: preparation.config.cardSha, source_fingerprint: '0'.repeat(64) } : {}) };
}

test('protocol policies retain distinct cohorts, budgets and protection bindings', () => {
  assert.deepEqual(preparations.map(item => item.config.protocol), ['fresh-decoder-replication-v1', 'saved-native-reliability-v1',
    'frozen-amplitude-transfer-v1', 'early-mixer-reliability-v1', 'early-mixer-learning-curve-v1',
    'matched-target-gain-v1', 'early-mixer-confirmation-v1', 'pooled-context-v1']);
  assert.deepEqual([reliability, curve, gain, confirmation, pooled].map(item => item.config.plan.timing_master_seeds), [
    [19119, 20220, 21321, 22422, 23523], [30130, 31231, 32332, 33433, 34534], [41140, 42241, 43342, 44443, 45544],
    [64161, 65262, 66363, 67464, 68565], [53151, 54252, 55353, 56454, 57555],
  ]);
  assert.equal(curve.config.plan.encoder_updates_each, 2048);
  assert.equal(curve.config.plan.planned_heads, 540);
  assert.equal(gain.config.plan.planned_heads, 360);
  assert.equal(confirmation.config.plan.implementation_fit_protocol, 'early-mixer-reliability-v1/lag_sign');
  assert.deepEqual(pooled.config.plan.parameter_counts, [225805, 231949]);
  assert.equal(pooled.config.plan.candidate_inactive_projection_values, 2080);
  for (const preparation of preparations) {
    assert.ok(Object.isFrozen(preparation.config.plan));
    assert.ok(preparation.config.outputRoot.startsWith('.temp/embeddings/runs/'));
    for (const source of preparation.requiredSources) {
      assert.ok(!retiredPython.has(source), `retired helper is not a current preparation dependency: ${source}`);
      assert.ok(fs.statSync(path.join(repo, source)).isFile(), `current preparation source exists: ${source}`);
    }
    preparation.checkPlan(plan(preparation));
    for (const key of Object.keys(preparation.config.plan)) {
      const bad = plan(preparation); bad[key] = { invalid: true };
      assert.throws(() => preparation.checkPlan(bad), /fixed recipe/);
    }
    if (preparation.config.planCardBinding) {
      for (const [key, value] of [['human_card_sha256', 'f'.repeat(64)], ['source_fingerprint', 'old-source']])
        assert.throws(() => preparation.checkPlan({ ...plan(preparation), [key]: value }), /plan frozen card/);
    }
  }
});

test('closed role enumeration contains 85 TRAIN roles and 80 parent roles without payload reads', () => {
  const parent = '/synthetic/' + PARENT_IDENTITY;
  const savedMatrix = savedRows(parent), transferMatrix = transferRows(parent);
  assert.equal(savedMatrix.length, 10); assert.ok(savedMatrix.every(row => row.length === 12));
  assert.equal(new Set(savedMatrix.flatMap(row => row.slice(3))).size, 85);
  assert.equal(transferMatrix.length, 15); assert.ok(transferMatrix.every(row => row.length === 13));
  assert.equal(new Set(transferMatrix.flatMap(row => row.slice(7))).size, 80);
  const roles = [...savedMatrix.flatMap(row => row.slice(3)), ...transferMatrix.flatMap(row => row.slice(7))];
  assert.ok(roles.every(file => file.startsWith(parent + '/')));
  assert.ok(roles.every(file => !/validation|testing|stress/i.test(file)));
  assert.deepEqual(transferMatrix[0].slice(0, 7), ['untrained_native-9109', 'Untrained encoder', '9109', '14614', 'untrained_native', 'ordinary_v4', '0']);
  assert.deepEqual(savedMatrix[0].slice(0, 3), ['v4-9109', 'RPB-v4.alt-01', '9109']);
});

test('source manifests use GNU SHA bytes and reject non-source paths', () => {
  assert.equal(manifestBytes([{ path: 'code/source.cpp', sha256: 'a'.repeat(64) }]).toString(), 'a'.repeat(64) + '  code/source.cpp\n');
  for (const name of ['../model.pt', '/model.pt', 'code/model.pt', 'code\\source.cpp', 'code/a b.cpp', 'code//a.cpp', './a.cpp'])
    assert.throws(() => relativeName(name));
  assert.equal(relativeName('code/preparation.mjs'), 'code/preparation.mjs');
});

test('whole source matrices reject late missing, symlink and alias entries before any hash', t => {
  const root = fixture(t), hashes = [], counted = createPreparation(saved.config, { fileSha: file => { hashes.push(file); return fileSha(file); } });
  const names = counted.requiredSources; materialize(root, names);
  const cardSha = sha256(Buffer.from('source-only fixture\n'));
  assert.throws(() => counted.sourceRecords(root, [...names, 'zz_missing.cpp'].sort(), cardSha));
  assert.deepEqual(hashes, []);
  fs.linkSync(path.join(root, 'Makefile'), path.join(root, 'zz_alias.cpp'));
  assert.throws(() => counted.sourceRecords(root, [...names, 'zz_alias.cpp'].sort(), cardSha), /alias/);
  assert.deepEqual(hashes, []);
  fs.unlinkSync(path.join(root, 'zz_alias.cpp')); fs.symlinkSync('Makefile', path.join(root, 'zz_redirect.cpp'));
  assert.throws(() => counted.sourceRecords(root, [...names, 'zz_redirect.cpp'].sort(), cardSha), /canonical|redirect/);
  assert.deepEqual(hashes, []);
  const records = counted.sourceRecords(root, names, cardSha);
  assert.equal(records.length, names.length); assert.equal(hashes.length, names.length);
});

test('captured-source extras and missing copies fail before current source hashes', t => {
  const root = fixture(t), repo = path.join(root, 'repo'), target = path.join(root, 'capsule');
  fs.mkdirSync(repo); fs.mkdirSync(target);
  const hashes = [], counted = createPreparation(curve.config, { fileSha: file => { hashes.push(file); return fileSha(file); } });
  materialize(repo, counted.requiredSources);
  const cardSha = sha256(Buffer.from('source-only fixture\n')), records = counted.sourceRecords(repo, counted.requiredSources, cardSha);
  counted.copySources(repo, target, records); hashes.length = 0;
  writeNew(path.join(target, 'source', 'zz_extra.cpp'), 'not admitted');
  assert.throws(() => counted.sourceState(repo, target, cardSha), /closed captured/); assert.deepEqual(hashes, []);
  fs.unlinkSync(path.join(target, 'source', 'zz_extra.cpp')); fs.unlinkSync(path.join(target, 'source', 'Makefile'));
  assert.throws(() => counted.sourceState(repo, target, cardSha), /closed captured/); assert.deepEqual(hashes, []);
});

test('source changes between matrix admission and a later hash are refused', t => {
  const root = fixture(t), names = saved.requiredSources;
  materialize(root, names);
  let changed = false;
  const raced = createPreparation(saved.config, { fileSha: (file, admitted) => {
    if (!changed) { changed = true; fs.appendFileSync(path.join(root, names.at(-1)), 'changed after admission'); }
    return hashAdmitted(admitted);
  } });
  assert.throws(() => raced.sourceRecords(root, names, sha256(Buffer.from('source-only fixture\n'))), /changed after admission/);
});

test('role admission rejects external hardlinks and ancestor symlinks before byte reads', t => {
  const root = fixture(t); materialize(root, ['a.json', 'dir/b.json'], '{}\n');
  fs.linkSync(path.join(root, 'a.json'), path.join(root, 'outside-alias.json'));
  assert.throws(() => admitMatrix([path.join(root, 'a.json')], root), /single-link/);
  fs.symlinkSync('dir', path.join(root, 'redirect'));
  assert.throws(() => admitMatrix([path.join(root, 'redirect/b.json')], root), /redirect/);
  assert.throws(() => readJson(path.join(root, 'outside-alias.json')), /single-link/);
});

test('reader source-only flags, fixture seals and replacement identity refusal remain separate', t => {
  const repo = fixture(t), tools = path.join(repo, curve.config.outputRoot, 'audit-tools'); fs.mkdirSync(tools, { recursive: true });
  const source = path.join(tools, 'source.py'), fixtures = path.join(tools, 'fixtures.json'), seal = path.join(tools, 'seal.json');
  writeNew(source, 'REVIEWED_SCHEMA = False\r\nMEASURED_IMPLEMENTATION = False\r\n');
  writeJson(fixtures, { status: 'passed', measured_execution_enabled: false });
  writeJson(seal, { status: 'passed', source_sha256: fileSha(source), fixtures_sha256: fileSha(fixtures) });
  const args = { reader_source: source, reader_fixtures: fixtures, reader_seal: seal };
  assert.deepEqual(curve.readerPreflight(repo, args).paths, [source, fixtures, seal]);
  const replacement = path.join(tools, 'source.mjs'); writeNew(replacement, 'export const REVIEWED_SCHEMA = false;\n');
  assert.throws(() => curve.readerPreflight(repo, { ...args, reader_source: replacement }), /identity has not been accepted/);
  fs.writeFileSync(source, 'REVIEWED_SCHEMA = True\nMEASURED_IMPLEMENTATION = False\n');
  fs.writeFileSync(seal, JSON.stringify({ status: 'passed', source_sha256: fileSha(source), fixtures_sha256: fileSha(fixtures) }));
  assert.throws(() => curve.readerPreflight(repo, args), /both execution flags FALSE/);
});

test('historical input matrix missing a late role performs zero hashes', t => {
  const parent = fixture(t), hashes = [], counted = createPreparation(saved.config, { fileSha: file => { hashes.push(file); return fileSha(file); } });
  const rows = savedRows(parent), files = [...new Set(rows.flatMap(row => row.slice(3)))].sort();
  materialize(parent, files.slice(0, -1).map(file => path.relative(parent, file)), 'synthetic role');
  writeJson(path.join(parent, 'artifact-integrity.json'), { protocol: 'fixture' });
  assert.throws(() => counted.inputRecords({ historicalParentCapsule: parent }, rows)); assert.deepEqual(hashes, []);
  assert.throws(() => counted.inputRecords({}, rows), /explicit --historical-parent-capsule/);
});

test('the complete pinned role association passes before any selected role is hashed', t => {
  const parent = fixture(t), rows = savedRows(parent), files = [...new Set(rows.flatMap(row => row.slice(3)))].sort();
  materialize(parent, files.map(file => path.relative(parent, file)), 'synthetic role');
  const inventory = path.join(parent, 'artifact-integrity.json'), hashes = [];
  const entries = files.map(file => ({ path: path.relative(parent, file), bytes: 14, sha256: sha256(Buffer.from('synthetic role')) }));
  while (entries.length < 948) entries.push({ path: 'unselected-source-only-' + entries.length, bytes: 0, sha256: '0'.repeat(64) });
  // Only this artificial inventory hash is injected; no historical witness or producer is used.
  const counted = createPreparation(saved.config, { fileSha: file => { hashes.push(file); return file === inventory ? PARENT_SHA : fileSha(file); } });
  const last = entries[files.length - 1], originalSize = last.bytes; last.bytes = originalSize + 1;
  writeJson(inventory, { protocol: 'fresh-decoder-replication-v1', file_count: 948, files: entries });
  assert.throws(() => counted.inputRecords({ historicalParentCapsule: parent }, rows), /whole pinned role/);
  assert.deepEqual(hashes, [inventory]);
  last.bytes = originalSize;
  fs.writeFileSync(inventory, JSON.stringify({ protocol: 'fresh-decoder-replication-v1', file_count: 948, files: entries }));
  hashes.length = 0;
  const bound = counted.inputRecords({ historicalParentCapsule: parent }, rows);
  assert.equal(bound.records.length, 85); assert.deepEqual(hashes, [inventory, ...files]);
  assert.equal(bound.mapping.historical_identity, PARENT_IDENTITY);
});

test('new capsules require .temp roots, named binaries and exclusive writes', t => {
  const repo = fixture(t), target = path.join(repo, saved.config.outputRoot, 'saved-native-reliability-fixture');
  fs.mkdirSync(target, { recursive: true }); assert.equal(saved.directory(repo, target), target);
  const old = path.join(repo, 'output/runs/rpb-saved-native-reliability/saved-native-reliability-fixture');
  fs.mkdirSync(old, { recursive: true }); assert.throws(() => saved.directory(repo, old), /\.temp protocol directory/);
  for (const binary of ['/usr/bin/python3', '/opt/mityane-geometrics/embeddings/build/wrong', 'embedding_saved_feature_reliability'])
    assert.throws(() => saved.binaryRecord(binary), /named container binary/);
  const record = path.join(target, 'complete.json'); writeJson(record, { status: 'complete' });
  assert.throws(() => writeJson(record, { status: 'changed' }), /EEXIST/);
  assert.equal(readJson(record).status, 'complete');
  assert.throws(() => writeJson(path.join(target, 'bad.json'), { score: NaN }), /finite exact/);
  assert.equal(fs.existsSync(path.join(target, 'bad.json')), false);
});

test('complete gates reject changed counters, extra fitting and forbidden heldout access', () => {
  for (const preparation of [reliability, curve, gain, confirmation, pooled]) {
    const complete = { protocol: preparation.config.protocol, status: 'complete', ...structuredClone(preparation.config.complete), head_pipelines: 1, individual_heads: 2 };
    if (preparation.config.completeRanges?.helper_outer_train_fits) complete.helper_outer_train_fits = 0;
    preparation.checkComplete(complete);
    assert.throws(() => preparation.checkComplete({ ...complete, individual_heads: 3 }), /paired heads/);
    for (const key of ['CPU_encoder_training', 'CPU_encoder_forward', 'testing_accessed', 'stress_accessed'])
      assert.throws(() => preparation.checkComplete({ ...complete, [key]: true }), /completed measurement/);
    for (const [key, [, upper]] of Object.entries(preparation.config.completeRanges))
      assert.throws(() => preparation.checkComplete({ ...complete, [key]: upper + 1 }), /completed measurement range/);
  }
});

test('producer/source schema association stays source-only', () => {
  const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
  const schema = sourceSchemaCheck(repo);
  assert.ok(schema.controlled.includes('observations'));
  assert.ok(!schema.controlled.includes('observed'));
});

test('CLI keeps explicit mappings and rejects unknown or repeated bindings', () => {
  assert.deepEqual(parseArgs(['freeze-run', '--historical-parent-capsule', '/mityane/retained/capsule', '--sources', 'Makefile', 'a.cpp']),
    { mode: 'freeze-run', repo_root: '.', historicalParentCapsule: '/mityane/retained/capsule', sources: ['Makefile', 'a.cpp'] });
  for (const argv of [['freeze-run', '--unknown', 'x'], ['freeze-run', '--target'], ['freeze-run', '--target', '/a', '--target', '/b'],
    ['freeze-run', '--historical-parent-root', '/ambiguous-old-option']])
    assert.throws(() => parseArgs(argv));
});
