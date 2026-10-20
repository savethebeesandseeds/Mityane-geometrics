// Authored while execution is disabled. These are future engineering fixtures.
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { capture, compileCommands, shellWords } from './saved_run_support.mjs';
import { sha256 } from '../../../../../shared/artifacts/node/v1/files.mjs';
import { parentRoles, selectedRoles, progressText, compileRecords, BINARY, MASTERS } from '../../visible_difference_v1/prepare_saved_run.mjs';
import { validateInformation, validatePlan, expectedPlan, RULE } from '../../structured_hard_timing_v1/prepare_saved_run.mjs';

test('new saved-run capture uses the current SDK verifier without the retired Python installer', t => {
  assert.equal(process.platform, 'linux');
  assert.equal(fs.existsSync('/.dockerenv'), true, 'managed container fixtures only');
  const base = '/opt/mityane-geometrics/python-migration-fixtures';
  fs.mkdirSync(base, { recursive: true });
  const repo = fs.mkdtempSync(path.join(base, 'saved-run-source-'));
  t.after(() => fs.rmSync(repo, { recursive: true, force: true }));
  const target = path.join(repo, 'capture'); fs.mkdirSync(target);
  const code = 'code/fixture-protocol', card = `${code}/card.md`, content = 'artificial source fixture\n';
  const names = [card, 'Makefile', 'dependencies.lock', 'setup.sh', 'code/scripts/verify-libtorch.mjs',
    'code/scripts/node.sh', 'code/scripts/node-runtime.sh',
    'code/embeddings/evaluation/protocols/node/v1/saved_run_support.mjs',
    'code/shared/artifacts/node/v1/files.mjs', `${code}/prepare_saved_run.mjs`].sort();
  for (const name of names) {
    const filename = path.join(repo, name);
    fs.mkdirSync(path.dirname(filename), { recursive: true }); fs.writeFileSync(filename, content);
  }
  const config = { card, code, cardSHA: sha256(content), protocol: 'artificial-source-fixture' };
  assert.throws(() => capture(repo, target, names.filter(name => !name.endsWith('verify-libtorch.mjs')), config),
    /missing exact source dependency: code\/scripts\/verify-libtorch.mjs/);
  const identity = capture(repo, target, names, config);
  const manifest = JSON.parse(fs.readFileSync(path.join(target, 'source-manifest.json'), 'utf8'));
  assert.equal(manifest.source_sha256, identity);
  assert.deepEqual(manifest.files.map(row => row.path), names);
  assert.equal(fs.existsSync(path.join(repo, 'code/scripts/install-libtorch.py')), false);
});

test('compiler receipt parsing preserves quoted macros and rejects missing/duplicate objects', () => {
  assert.deepEqual(shellWords('g++ -DID=\\"abc\\" -I"a b" -c code/a.cpp -o /opt/a.o'),
    ['g++', '-DID="abc"', '-Ia b', '-c', 'code/a.cpp', '-o', '/opt/a.o']);
  assert.throws(() => shellWords('g++ "unterminated'));
  const required = [['code/a.cpp', '/opt/a.o'], ['code/b.cpp', '/opt/b.o']];
  const log = 'g++ -c code/a.cpp -o /opt/a.o\ng++ -c code/b.cpp -o /opt/b.o\ng++ /opt/a.o /opt/b.o -o /opt/bin\n';
  assert.equal(Object.keys(compileCommands(log, required, '/opt/bin').commands).length, 2);
  for (const bad of [log + 'g++ -c code/a.cpp -o /opt/a.o\n',
    log.replace('g++ /opt/a.o /opt/b.o', 'g++ /opt/b.o'),
    log.replace('-o /opt/a.o', '-o /other/a.o'), log + '/embedding/.external/libtorch/libtorch.so'])
    assert.throws(() => compileCommands(bad, required, '/opt/bin'));
});

test('visible-difference retains exactly130 original roles and complete parent progress traces', () => {
  const roles = parentRoles(); assert.equal(roles.length, 130); assert.equal(new Set(roles).size, 130);
  const rows = roles.map(name => ({ path: name, bytes: 7, sha256: 'a'.repeat(64) }));
  assert.deepEqual(selectedRoles({ status: 'complete', files: rows }), rows);
  assert.throws(() => selectedRoles({ status: 'complete', files: rows.slice(1) }));
  assert.throws(() => selectedRoles({ status: 'complete', files: [...rows, rows[0]] }));
  const audit = { per_master: MASTERS.map(timing_master => ({ timing_master, encoders: [
    { role: 'early', attempted: 512, completed: 512, sampled_rows: 4096, context_counts: [23, 21, 2] }] })) };
  const summary = { cohorts: MASTERS.map(timing_master => ({ timing_master, encoder_points: [
    { placement: 1, encoder_progress: { attempted: 512, completed: 512, sampled_rows: 4096,
      losses: Array.from({ length: 512 }, (_, i) => [i + 1, i + 1, 4, .1, .2]) } }] })) };
  const output = progressText(audit, summary); assert.equal(output.trimEnd().split('\n').length, 6);
  assert.equal(output.split('\n')[1].split('\t')[7].split(',').length, 512);
  summary.cohorts[0].encoder_points[0].encoder_progress.losses.pop(); assert.throws(() => progressText(audit, summary));
});

test('compiled visible scopes bind actual current object paths and exact macro identity', () => {
  const base = BINARY.slice(0, BINARY.lastIndexOf('/')), old = 'code/embeddings/encoders/raw_patch_bottleneck_mae/shared/src/';
  const od = `${base}/code/encoders/raw_patch_bottleneck_mae`, nd = `${base}/code/protocols/visible_difference_v1`;
  const specs = [[old + 'workflow.cpp', `${od}/workflow.o`, 'RPB_SOURCE_ID'],
    [old + 'learning_curve_adapter.cpp', `${od}/learning_curve_adapter.o`, 'EVALUATION_SOURCE_ID'],
    [old + 'early_mixer_adapter.cpp', `${od}/early_mixer_adapter.o`, 'EARLY_MIXER_ADAPTER_SOURCE_ID'],
    ['code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v13/src/visible_difference_adapter.cpp', `${nd}/visible_difference_adapter.o`, 'VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID'],
    ['code/embeddings/evaluation/protocols/visible_difference_v1/visible_difference_main.cpp', `${nd}/visible_difference_main.o`, 'VISIBLE_DIFFERENCE_SOURCE_ID']];
  const scopes = Object.fromEntries(['core_writer', 'curve_training', 'early_adapter', 'visible_adapter'].map(name => [name, { source_sha256: 'a'.repeat(64) }]));
  const lines = specs.map(([source, object, macro]) => `g++ -D${macro}=\\"${'a'.repeat(64)}\\" -c ${source} -o ${object}`);
  const log = [...lines, `g++ ${specs.map(row => row[1]).join(' ')} -o ${BINARY}`].join('\n');
  assert.equal(Object.keys(compileRecords(log, BINARY, scopes).commands).length, 5);
  assert.throws(() => compileRecords(log.replace('-DRPB_SOURCE_ID=\\"' + 'a'.repeat(64), '-DRPB_SOURCE_ID=\\"' + 'b'.repeat(64)), BINARY, scopes));
});

test('information gates preserve rational counts, closed seeds and negative outcomes', () => {
  const score = { total: 1024, valid: 1024, correct: 1024, accuracy: 1, coverage: 1, full_population_correctness: 1, passed: true };
  const value = { protocol: 'structured-hard-timing-information-v2', dataset_id: 'TEMPO-3', rule_id: RULE, complexity: 4, complexity_maximum: 5,
    seeds: [82585, 83686].map(seed => ({ seed, training_source_pairs: 512, validation_source_pairs: 512, intact: { ...score }, deleted: { ...score } })),
    passed: true, encoder_or_head_fits: 0, TEST_generated: false };
  assert.equal(validateInformation(value), true);
  for (const change of [v => { v.seeds[1].seed = 1; }, v => { v.seeds[1].deleted.accuracy = .5; },
    v => { v.seeds[1].deleted.valid = 1025; }, v => { v.TEST_generated = true; }, v => { v.passed = false; }]) {
    const bad = structuredClone(value); change(bad); assert.throws(() => validateInformation(bad));
  }
  const failed = structuredClone(value); Object.assign(failed.seeds[1].deleted,
    { correct: 0, accuracy: 0, full_population_correctness: 0, passed: false }); failed.passed = false;
  assert.equal(validateInformation(failed), false);
  const plan = { ...expectedPlan(), source_fingerprint: 'a'.repeat(64) }; validatePlan(plan);
  assert.throws(() => validatePlan({ ...plan, testing_accessed: true }));
  assert.throws(() => validatePlan({ ...plan, source_fingerprint: 'wrong' }));
});
