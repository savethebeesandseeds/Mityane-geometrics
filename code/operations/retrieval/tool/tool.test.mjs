// Engineering boundary for OR01-RET. Saved vectors only; no relevance claim.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {engineeringRequest} from './fixture.mjs';
import {writeExample} from './example.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
assert.equal(process.platform, 'linux', 'Run retrieval tool tests in the managed Linux container.');
assert.equal(ROOT, '/mityane', 'Use the authoritative managed-container checkout.');
const CLI = path.join(ROOT, 'code/operations/tools.mjs');
const TEST_PARENT = path.join(ROOT, '.temp/operations/retrieval/toolset-tests');
fs.mkdirSync(TEST_PARENT, {recursive: true});
const TEST_ROOT = fs.mkdtempSync(path.join(TEST_PARENT, 'boundary-'));
let sequence = 0;
const readJSON = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const writeJSON = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);

function call(args, expected = 0) {
  const run = spawnSync(process.execPath, [CLI, ...args], {
    cwd: ROOT, env: process.env, encoding: 'utf8', timeout: 30000, maxBuffer: 8 * 1024 * 1024,
  });
  assert.equal(run.error, undefined, `CLI process failed: ${run.error?.message}`);
  assert.equal(run.signal, null, `CLI unexpectedly received ${run.signal}`);
  assert.equal(run.status, expected,
    `CLI ${args.join(' ')} exited ${run.status}, expected ${expected}\nstdout: ${run.stdout}\nstderr: ${run.stderr}`);
  return run;
}
function jsonCall(args, expected = 0) { return JSON.parse(call(args, expected).stdout); }
function directory() { return path.join(TEST_ROOT, `case-${++sequence}`); }
function writeBundle(spec) {
  const fixture = engineeringRequest(spec);
  const destination = directory();
  fs.mkdirSync(path.join(destination, 'artifacts'), {recursive: true});
  const seen = new Set();
  const artifacts = [];
  for (const artifact of fixture.artifacts) {
    const key = `${artifact.id}\0${artifact.sha256}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const relative = `artifacts/${String(artifacts.length + 1).padStart(4, '0')}-${artifact.sha256}.json`;
    fs.writeFileSync(path.join(destination, relative), artifact.bytes);
    artifacts.push({id: artifact.id, sha256: artifact.sha256, path: relative});
  }
  const requestFile = path.join(destination, 'request.json');
  writeJSON(requestFile, {
    schema_version: 'mityane.operation.request/v1',
    operation: {id: 'OR01-RET', revision: '0.1.0'},
    input: fixture.input, artifacts,
  });
  return {destination, requestFile};
}
function runBundle(spec, expected = 0) {
  const bundle = writeBundle(spec);
  const output = directory();
  const stdout = jsonCall(['run', '--request', bundle.requestFile, '--output', output], expected);
  const record = expected === 2 ? null : readJSON(path.join(output, 'result.json'));
  return {bundle, output, stdout, record};
}

test('describe lists the engineering retrieval tool', () => {
  const list = jsonCall(['list']);
  assert.ok(list.some(row => row.id === 'OR01-RET' && row.revision === '0.1.0' &&
    row.claim_levels.length === 1 && row.claim_levels[0] === 'exploratory_description'));
  const described = jsonCall(['describe', 'OR01-RET']);
  assert.equal(described.model_execution, false);
  assert.equal(described.family, 'OR-01');
});

test('toy-a example ranks the nearer saved vector and keeps lineage', async () => {
  const generated = await writeExample({provider: 'toy-a', output: directory()});
  const output = directory();
  const stdout = jsonCall(['run', '--request', generated.request_file, '--output', output]);
  assert.equal(stdout.status, 'completed');
  const record = readJSON(path.join(output, 'result.json'));
  assert.equal(record.claim_level, 'exploratory_description');
  assert.equal(record.result.engineering_only, true);
  assert.equal(record.result.outcome, 'ranked');
  assert.equal(record.result.operation_run_id, 'run-engineering-1');
  assert.equal(record.result.hits[0].source_id, 'near');
  assert.equal(record.result.hits[0].rank, 1);
  assert.equal(record.result.hits[1].source_id, 'far');
  assert.equal(record.result.hits[0].parent_id, 'parent-near');
  assert.equal(record.result.fitted_transform_id, 'fitted-artifact');
  assert.equal(record.summary.succeeded, 1);
  const loaded = jsonCall(['read', '--result', path.join(output, 'result.json')]);
  assert.equal(loaded.record.result.operation_run_id, record.result.operation_run_id);
});

test('excluding the only member is an empty completion', () => {
  const {record} = runBundle({
    query_source_id: 'only', query: [1, 0], corpus: [{source_id: 'only', values: [1, 0]}], k: 1,
  });
  assert.equal(record.status, 'completed');
  assert.equal(record.result.outcome, 'empty');
  assert.equal(record.result.hits.length, 0);
  assert.equal(record.result.eligible_count, 0);
  assert.equal(record.result.skipped[0].reason, 'self_match');
});

test('a zero cosine query abstains', () => {
  const {record} = runBundle({
    metric: 'cosine', normalization: 'none', query: [0, 0],
    corpus: [{source_id: 'defined', values: [1, 0]}], k: 1,
  });
  assert.equal(record.status, 'completed');
  assert.equal(record.result.outcome, 'abstained');
  assert.equal(record.summary.abstained, 1);
  assert.equal(record.result.hits.length, 0);
});

test('a causal claim and a live call are rejected without hits', () => {
  for (const spec of [{claim_level: 'controlled_model_response'}, {live_call: true}, {claim_level: 'task_association'}]) {
    const {record} = runBundle(spec, 3);
    assert.equal(record.status, 'rejected');
    assert.equal(record.result.outcome, 'rejected');
    assert.equal(record.result.hits.length, 0);
    assert.equal(record.result.skipped.length, 0);
    assert.equal(record.summary.succeeded, 0);
  }
});

test('as-of mode skips a future member and rejects a future fitted artifact', () => {
  const ranked = runBundle({
    temporal_mode: 'as_of', as_of: 10, non_temporal_justification: '',
    fitted_available_at: 5, query_available_at: 10, k: 2,
    corpus: [
      {source_id: 'past', values: [1, 0], available_at: 5},
      {source_id: 'future', values: [2, 0], available_at: 11},
    ],
  });
  assert.equal(ranked.record.result.outcome, 'ranked');
  assert.equal(ranked.record.result.hits.length, 1);
  assert.equal(ranked.record.result.hits[0].source_id, 'past');
  assert.equal(ranked.record.result.skipped[0].source_id, 'future');
  assert.equal(ranked.record.result.skipped[0].reason, 'temporal_violation');

  const stale = runBundle({
    temporal_mode: 'as_of', as_of: 10, non_temporal_justification: '',
    fitted_available_at: 12, query_available_at: 10, k: 1,
    corpus: [{source_id: 'past', values: [1, 0], available_at: 5}],
  }, 3);
  assert.equal(stale.record.result.outcome, 'rejected');
  assert.equal(stale.record.result.admission_diagnostics[0].code, 'STALE_FITTED_ARTIFACT');
  assert.equal(stale.record.result.hits.length, 0);
});

test('overlapping source windows are reported and not ranked', () => {
  const {record} = runBundle({
    temporal_mode: 'retrospective', overlap_policy: 'exclude', non_temporal_justification: '',
    query_window: [0, 10], k: 2,
    corpus: [
      {source_id: 'overlap', values: [1, 0], window_start: 5, window_end: 15},
      {source_id: 'clear', values: [3, 0], window_start: 20, window_end: 25},
    ],
  });
  assert.equal(record.result.hits.map(hit => hit.source_id).join(','), 'clear');
  assert.equal(record.result.skipped[0].reason, 'source_window_overlap');
});

test('duplicate source ids are rejected', () => {
  const {record} = runBundle({
    corpus: [{source_id: 'a', values: [1, 0]}, {source_id: 'a', values: [1, 0]}], k: 1,
  }, 3);
  assert.equal(record.result.admission_diagnostics[0].code, 'DUPLICATE_SOURCE');
  assert.equal(record.result.hits.length, 0);
});

test('inline values that disagree with the saved artifact fail before ranking', () => {
  const bundle = writeBundle({});
  const request = readJSON(bundle.requestFile);
  request.input.query.values = [9, 9];
  writeJSON(bundle.requestFile, request);
  call(['run', '--request', bundle.requestFile, '--output', directory()], 2);
});
