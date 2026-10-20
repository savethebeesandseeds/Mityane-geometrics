// User-boundary checks: real JSON files, retained bytes, and the actual C++ tool.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { writeExample } from './example.mjs';
import { checkTool, runTool, inspectResult } from '../../../shared/representation/tools/v1/runtime.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
assert.equal(process.platform, 'linux', 'Run tool tests in the managed Linux container.');
assert.equal(ROOT, '/mityane', 'Use the authoritative managed-container checkout.');
const CLI = path.join(ROOT, 'code/operations/tools.mjs');
const TEST_PARENT = path.join(ROOT, '.temp/operations/counterfactuals/toolset-tests');
fs.mkdirSync(TEST_PARENT, { recursive: true });
const TEST_ROOT = fs.mkdtempSync(path.join(TEST_PARENT, 'boundary-'));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const readJSON = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const writeJSON = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
let sequence = 0;

function call(args, expected = 0) {
  const run = spawnSync(process.execPath, [CLI, ...args], {
    cwd: ROOT, env: process.env, encoding: 'utf8', timeout: 30000,
    maxBuffer: 8 * 1024 * 1024,
  });
  assert.equal(run.error, undefined, `CLI process failed: ${run.error?.message}`);
  assert.equal(run.signal, null, `CLI unexpectedly received ${run.signal}`);
  assert.equal(run.status, expected,
    `CLI ${args.join(' ')} exited ${run.status}, expected ${expected}\nstdout: ${run.stdout}\nstderr: ${run.stderr}`);
  return run;
}
function jsonCall(args, expected = 0) { return JSON.parse(call(args, expected).stdout); }
async function bundle(provider = 'toy-a') {
  const directory = path.join(TEST_ROOT, `request-${++sequence}`);
  const example = await writeExample({ provider, output: directory });
  return { directory, requestFile: example.request_file, envelope: readJSON(example.request_file) };
}
function save(request) { writeJSON(request.requestFile, request.envelope); }
function newOutput() { return path.join(TEST_ROOT, `result-${++sequence}`); }
function runBundle(request, expected = 0) {
  const output = newOutput();
  const stdout = jsonCall(['run', '--request', request.requestFile, '--output', output], expected);
  return { output, stdout, record: readJSON(path.join(output, 'result.json')) };
}
function replaceDigest(value, id, sha256) {
  if (!value || typeof value !== 'object') return;
  if (!Array.isArray(value) && value.id === id && Object.hasOwn(value, 'sha256')) value.sha256 = sha256;
  for (const child of Object.values(value)) replaceDigest(child, id, sha256);
}
function rewriteArtifact(request, id, payload) {
  const artifact = request.envelope.artifacts.find(row => row.id === id);
  assert.ok(artifact, `fixture artifact ${id} exists`);
  const bytes = Buffer.from(`${JSON.stringify(payload)}\n`);
  fs.writeFileSync(path.join(request.directory, artifact.path), bytes);
  replaceDigest(request.envelope, id, digest(bytes));
  save(request);
}
function separateArtifact(request, oldRef, payload) {
  const old = request.envelope.artifacts.find(row => row.id === oldRef.id);
  assert.ok(old, `fixture artifact ${oldRef.id} exists`);
  const suffix = ++sequence;
  const id = `${oldRef.id}-separate-record-${suffix}`;
  const relative = `artifacts/extra-${suffix}.json`;
  const bytes = payload === undefined ? fs.readFileSync(path.join(request.directory, old.path))
    : Buffer.from(`${JSON.stringify(payload)}\n`);
  fs.writeFileSync(path.join(request.directory, relative), bytes, { flag: 'wx' });
  const ref = { id, sha256: digest(bytes) };
  request.envelope.artifacts.push({ ...ref, path: relative });
  return ref;
}
function separatePair(request) {
  const second = structuredClone(request.envelope.input.pairs[0]);
  second.id = `separately-recorded-pair-${++sequence}`;
  // A distinct recording of the same declared edit may have identical content,
  // but keeps distinct input, representation and response record identities.
  for (const key of ['input', 'representation', 'response']) {
    second.alternative[key] = separateArtifact(request, second.alternative[key]);
  }
  second.alternative.response_execution_id += '-separate-record';
  return second;
}
function assertCounts(summary, expected) {
  assert.deepEqual(summary, expected);
  assert.equal(summary.requested,
    summary.succeeded + summary.abstained + summary.failed + summary.unresolved);
}

test('catalogue describes the callable operation and explicit input schema', () => {
  const list = jsonCall(['list']);
  assert.ok(Array.isArray(list));
  assert.ok(list.some(row => row.id === 'OR06-PCR' && row.revision === '0.1.0'));
  const descriptor = jsonCall(['describe', 'OR06-PCR']);
  assert.equal(descriptor.id, 'OR06-PCR');
  assert.equal(descriptor.revision, '0.1.0');
  assert.deepEqual(descriptor.claim_levels, ['controlled_model_response']);
  const schema = descriptor.input_schema;
  assert.equal(schema.additionalProperties, false);
  assert.ok(schema.required.includes('expected_space'));
  assert.ok(schema.required.includes('pairs'));
  call(['describe', 'unknown-operation'], 2);
});

test('common runtime accepts a different operation payload and a JavaScript implementation', async () => {
  let executions = 0;
  let checks = 0;
  const mock = {
    outputRoot: '.temp/operations/counterfactuals/toolset-tests',
    descriptor: { id: 'engineering-generic-fixture', revision: '1',
      limitations: ['Engineering fixture only; no research capability is registered or claimed.'] },
    encode(input) {
      assert.deepEqual(Object.keys(input), ['items']);
      assert.ok(input.items.every(Number.isFinite));
      return Buffer.from(JSON.stringify(input.items));
    },
    references() { return []; },
    verify(_input, artifacts) { assert.equal(artifacts.size, 0); },
    context() { return { operation_run_id: 'generic-engineering-run', claim_level: 'engineering_fixture' }; },
    async check(wire) {
      checks++;
      return { result: { doubled: JSON.parse(wire).map(item => item * 2) },
        native_binary: null, source_files: [] };
    },
    async execute(wire) {
      executions++;
      return { result: { doubled: JSON.parse(wire).map(item => item * 2) },
        native_binary: null, source_files: [] };
    },
    outcome(result, input) {
      assert.equal(Object.hasOwn(input, 'pairs'), false);
      assert.equal(Object.hasOwn(result, 'admitted'), false);
      return { status: 'completed', details: [`Doubled values: ${result.doubled.join(', ')}`],
        summary: { requested: input.items.length, succeeded: input.items.length,
          abstained: 0, failed: 0, unresolved: 0, success_rate: 1 } };
    },
  };
  const requestFile = path.join(TEST_ROOT, `generic-request-${++sequence}.json`);
  writeJSON(requestFile, { schema_version: 'mityane.operation.request/v1',
    operation: { id: mock.descriptor.id, revision: mock.descriptor.revision },
    input: { items: [2, -3, 0.5] }, artifacts: [] });
  const checked = await checkTool([mock], requestFile);
  assert.deepEqual(checked.result, { doubled: [4, -6, 1] });
  assert.equal(checks, 1);
  assert.equal(executions, 0, 'checking uses the explicit read-only implementation');
  const published = await runTool([mock], requestFile, newOutput());
  const saved = readJSON(published.result_file);
  assert.deepEqual(saved.result, { doubled: [4, -6, 1] });
  assert.equal(saved.execution.native_binary, null);
  assert.deepEqual(saved.verified_artifacts, []);
  assert.match(inspectResult(published.result_file), /Doubled values: 4, -6, 1/);
  assert.equal(executions, 1, 'inspection validates saved files without re-executing an operation');
});

test('same tool accepts two frozen provider identities and different widths with manual contrasts', async () => {
  const providers = [];
  for (const [provider, dimensions] of [['toy-a', 4], ['toy-b', 7]]) {
    const request = await bundle(provider);
    providers.push(request.envelope.input.expected_space.provider.id);
    assert.deepEqual(request.envelope.input.expected_space.native_shape, [dimensions]);
    const originalRequest = fs.readFileSync(request.requestFile);
    const { output, stdout, record } = runBundle(request);
    assert.equal(stdout.status, 'completed');
    assert.equal(record.schema_version, 'mityane.operation.result/v1');
    assert.equal(record.result.admitted, true);
    assertCounts(record.summary, { requested: 1, succeeded: 1, abstained: 0, failed: 0,
      unresolved: 0, success_rate: 1 });
    assert.deepEqual(record.result.pairs[0].contrast,
      { baseline: [2, -1], alternative: [5, -3], delta: [3, -2] });
    assert.equal(record.input.sha256, digest(originalRequest));
    assert.deepEqual(fs.readFileSync(path.join(output, 'request.json')), originalRequest);
    assert.equal(record.verified_artifacts.length, request.envelope.artifacts.length);
    for (const artifact of record.verified_artifacts) {
      assert.equal(digest(fs.readFileSync(path.join(output, artifact.retained_path))), artifact.sha256);
    }
    assert.match(call(['inspect', '--result', path.join(output, 'result.json')]).stdout, /delta = \[3, -2\]/);
  }
  assert.notEqual(providers[0], providers[1]);
});

test('common example command writes a complete bundle and preserves an existing destination', () => {
  const output = path.join(TEST_ROOT, `cli-example-${++sequence}`);
  const generated = jsonCall(['example', 'OR06-PCR', '--provider', 'toy-b', '--output', output]);
  assert.equal(generated.representation_dimensions, 7);
  assert.equal(readJSON(generated.request_file).operation.id, 'OR06-PCR');
  const original = fs.readFileSync(generated.request_file);
  call(['example', 'OR06-PCR', '--provider', 'toy-a', '--output', output], 2);
  assert.deepEqual(fs.readFileSync(generated.request_file), original);
});

test('check uses the actual comparison without publishing or changing input files', async () => {
  const request = await bundle();
  const before = request.envelope.artifacts.map(row => digest(fs.readFileSync(path.join(request.directory, row.path))));
  const original = fs.readFileSync(request.requestFile);
  const checked = jsonCall(['check', '--request', request.requestFile]);
  assert.equal(checked.schema_version, 'mityane.operation.check/v1');
  assert.equal(checked.valid, true);
  assert.deepEqual(checked.result.pairs[0].contrast.delta, [3, -2]);
  assert.deepEqual(fs.readFileSync(request.requestFile), original);
  assert.deepEqual(request.envelope.artifacts.map(row => digest(fs.readFileSync(path.join(request.directory, row.path)))), before);
  assert.equal(fs.existsSync(path.join(request.directory, 'result.json')), false);
});

test('Unicode record identities survive JSON and the native framing boundary', async () => {
  const request = await bundle();
  const pairId = '応答-Δ-🙂';
  request.envelope.input.purpose = 'Comparer une représentation figée — 科学';
  request.envelope.input.pairs[0].id = pairId;
  save(request);
  const checked = jsonCall(['check', '--request', request.requestFile]);
  assert.equal(checked.result.pairs[0].pair_id, pairId);
  assert.deepEqual(checked.result.pairs[0].contrast.delta, [3, -2]);
});

test('unsafe numeric, omitted optional, invalid enum and malformed Unicode metadata reject', async () => {
  const mutations = [
    input => { input.selected_output_position = 0.5; },
    input => { input.as_of = Number.MAX_SAFE_INTEGER + 1; },
    input => { delete input.as_of; },
    input => { input.pairs[0].alternative.origin = 'unregistered-origin'; },
    input => { input.purpose = '\ud800'; },
  ];
  for (const mutate of mutations) {
    const request = await bundle();
    mutate(request.envelope.input);
    save(request);
    const output = newOutput();
    call(['run', '--request', request.requestFile, '--output', output], 2);
    assert.equal(fs.existsSync(output), false);
  }
});

test('malformed JSON, missing typed metadata and unknown fields are rejected before publication', async () => {
  const malformed = await bundle();
  fs.writeFileSync(malformed.requestFile, '{"schema_version":');
  call(['check', '--request', malformed.requestFile], 2);
  const missing = await bundle();
  delete missing.envelope.input.expected_space;
  save(missing);
  const output = newOutput();
  call(['run', '--request', missing.requestFile, '--output', output], 2);
  assert.equal(fs.existsSync(output), false);
  const unknown = await bundle();
  unknown.envelope.input.pairs[0].pretend_causal_effect = true;
  save(unknown);
  call(['check', '--request', unknown.requestFile], 2);
});

test('duplicate JSON keys cannot silently replace the intended claim', async () => {
  const request = await bundle();
  const text = fs.readFileSync(request.requestFile, 'utf8');
  fs.writeFileSync(request.requestFile, text.replace('"claim_level": "controlled_model_response"',
    '"claim_level": "identified_intervention_effect", "claim_level": "controlled_model_response"'));
  const rejected = call(['check', '--request', request.requestFile], 2);
  assert.match(rejected.stderr, /duplicate/i);
});

test('changed bytes and missing inventory identities are rejected without a result directory', async () => {
  const corrupt = await bundle();
  const artifact = corrupt.envelope.artifacts[0];
  fs.appendFileSync(path.join(corrupt.directory, artifact.path), '\nchanged after hashing\n');
  const output = newOutput();
  const rejected = call(['run', '--request', corrupt.requestFile, '--output', output], 2);
  assert.match(rejected.stderr, /hash mismatch/i);
  assert.equal(fs.existsSync(output), false);
  const missing = await bundle();
  missing.envelope.artifacts.pop();
  save(missing);
  call(['check', '--request', missing.requestFile], 2);
});

test('even correctly rehashed response artifacts must agree with declared response values', async () => {
  const request = await bundle();
  const branch = request.envelope.input.pairs[0].alternative;
  rewriteArtifact(request, branch.response.id, { axes: branch.axes, values: [500, -3] });
  const rejected = call(['check', '--request', request.requestFile], 2);
  assert.match(rejected.stderr, /response|payload/i);
});

test('relative artifact paths cannot escape the request bundle', async () => {
  const request = await bundle();
  request.envelope.artifacts[0].path = '../outside.json';
  save(request);
  const rejected = call(['check', '--request', request.requestFile], 2);
  assert.match(rejected.stderr, /relative|escapes|path/i);
});

test('supported and abstained pairs retain the full denominator and never invent a zero contrast', async () => {
  const request = await bundle();
  const second = separatePair(request);
  second.id = 'outside-declared-support';
  second.support.state = 'rejected';
  second.support.detail = 'This alternative is outside the declared supported region.';
  second.support.evidence = separateArtifact(request, second.support.evidence,
    { check: 'support', state: 'rejected', scope: 'Artificial engineering assertion' });
  request.envelope.input.pairs.push(second);
  save(request);
  const { record } = runBundle(request);
  assertCounts(record.summary, { requested: 2, succeeded: 1, abstained: 1, failed: 0,
    unresolved: 0, success_rate: 0.5 });
  assert.deepEqual(record.result.pairs.map(row => row.state), ['success', 'abstained']);
  assert.equal(record.result.pairs[1].contrast, null);
  assert.ok(record.result.pairs[1].diagnostics.some(row => row.code === 'OUTSIDE_DECLARED_SUPPORT'));
});

test('incompatible spaces yield a failed pair while preserving a successful pair and denominator', async () => {
  const request = await bundle();
  const second = separatePair(request);
  second.id = 'incompatible-space';
  second.alternative.space.geometry = 'incompatible-geometry';
  request.envelope.input.pairs.push(second);
  save(request);
  const { record } = runBundle(request, 4);
  assert.equal(record.status, 'partial');
  assertCounts(record.summary, { requested: 2, succeeded: 1, abstained: 0, failed: 1,
    unresolved: 0, success_rate: 0.5 });
  assert.equal(record.result.pairs[1].contrast, null);
  assert.ok(record.result.pairs[1].diagnostics.some(row => row.code === 'INCOMPATIBLE_ROLE_OR_SPACE'));
});

test('causal claim admission is rejected explicitly without invented executed pairs', async () => {
  const request = await bundle();
  request.envelope.input.claim_level = 'identified_intervention_effect';
  save(request);
  const { record, output } = runBundle(request, 3);
  assert.equal(record.status, 'rejected');
  assert.equal(record.result.admitted, false);
  assert.deepEqual(record.result.pairs, []);
  assert.deepEqual(record.summary, { requested: request.envelope.input.pairs.length,
    succeeded: 0, abstained: 0, failed: 0,
    unresolved: 0, success_rate: null });
  assert.ok(record.result.admission_diagnostics.some(row => row.code === 'UNSUPPORTED_CLAIM'));
  assert.equal(readJSON(path.join(output, 'request.json')).input.pairs.length, 1);
});

test('publication collision preserves the complete original result', async () => {
  const request = await bundle();
  const { output } = runBundle(request);
  const files = ['request.json', 'result.json', 'summary.txt', 'manifest.json'];
  const before = files.map(file => digest(fs.readFileSync(path.join(output, file))));
  call(['run', '--request', request.requestFile, '--output', output], 2);
  assert.deepEqual(files.map(file => digest(fs.readFileSync(path.join(output, file)))), before);
});

test('inspect detects retained artifact tampering before returning a trusted summary', async () => {
  const request = await bundle();
  const { output, record } = runBundle(request);
  const retained = path.join(output, record.verified_artifacts[0].retained_path);
  fs.appendFileSync(retained, 'tampered');
  const rejected = call(['inspect', '--result', path.join(output, 'result.json')], 2);
  assert.match(rejected.stderr, /hash|bytes|mismatch/i);
  assert.equal(rejected.stdout.trim(), '');
});
