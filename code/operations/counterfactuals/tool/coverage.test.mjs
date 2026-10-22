// Toolset coverage: real saved bytes, public APIs and provider-neutral workflows.
// Run only in the managed Linux container; generated fixtures remain in .temp/.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {operations} from '../../tools.mjs';
import {createToolset, formatSummary} from '../../../shared/representation/tools/v1/runtime.mjs';
import {validateSchema} from '../../../shared/representation/tools/v1/schemas.mjs';
import {sha256} from '../../../shared/artifacts/node/v1/files.mjs';
import {writeExample} from './example.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
assert.equal(process.platform, 'linux', 'Run coverage checks in the managed Linux container.');
assert.equal(ROOT, '/mityane', 'Use the authoritative managed-container checkout.');
const CLI = path.join(ROOT, 'code/operations/tools.mjs');
const PARENT = path.join(ROOT, '.temp/operations/counterfactuals/toolset-tests');
fs.mkdirSync(PARENT, {recursive: true});
const WORK = fs.mkdtempSync(path.join(PARENT, 'coverage-'));
let sequence = 0;
const next = name => path.join(WORK, `${name}-${++sequence}`);
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const writeJSON = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);

function call(args, expected = 0) {
  const result = spawnSync(process.execPath, [CLI, ...args], {cwd: ROOT, env: process.env,
    encoding: 'utf8', timeout: args[0] === 'workflow' ? 300000 : args[0] === 'prepare' ? 180000 : 30000,
    maxBuffer: 16 * 1024 * 1024});
  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.signal, null, result.stderr);
  assert.equal(result.status, expected,
    `${args.join(' ')} exited ${result.status}, expected ${expected}\n${result.stdout}\n${result.stderr}`);
  return result;
}
const jsonCall = (args, expected = 0) => JSON.parse(call(args, expected).stdout);

async function source(provider = 'toy-a') {
  const directory = next('source');
  const example = await writeExample({provider, output: directory});
  const spec = json(example.request_file);
  spec.schema_version = 'mityane.operation.prepare/v1';
  const specFile = path.join(directory, 'prepare.json');
  writeJSON(specFile, spec);
  return {directory, specFile, spec};
}
async function saved() {
  const supplied = await source();
  const preparedDirectory = next('prepared');
  const prepared = await operations.prepare(supplied.specFile, preparedDirectory);
  const directory = next('result');
  const result = await operations.run(prepared.request_file, directory);
  return {supplied, preparedDirectory, prepared, directory, result};
}

// Change the saved result and update both its readable summary and all manifest
// hashes. These fixtures test semantic joins beyond ordinary byte tampering.
function rewrittenCopy(original, mutate) {
  const directory = next('rewritten');
  fs.cpSync(original, directory, {recursive: true, errorOnExist: true});
  const recordFile = path.join(directory, 'result.json');
  const record = json(recordFile), request = json(path.join(directory, 'request.json'));
  mutate(record, request, directory);
  writeJSON(recordFile, record);
  writeJSON(path.join(directory, 'request.json'), request);
  // A changed request must retain a coherent recorded request digest too.
  record.input.sha256 = sha256(fs.readFileSync(path.join(directory, 'request.json')));
  writeJSON(recordFile, record);
  fs.writeFileSync(path.join(directory, 'summary.txt'), formatSummary(record));
  const manifest = json(path.join(directory, 'manifest.json'));
  for (const row of manifest.files) {
    const bytes = fs.readFileSync(path.join(directory, row.path));
    row.bytes = bytes.length;
    row.sha256 = sha256(bytes);
  }
  writeJSON(path.join(directory, 'manifest.json'), manifest);
  return path.join(directory, 'result.json');
}

test('describe publishes complete schemas and rejects malformed native/check/result records', async () => {
  const description = operations.describe('OR06-PCR');
  for (const name of ['preparation_schema', 'prepared_schema', 'request_schema', 'check_schema',
    'result_schema', 'files_schema', 'loaded_schema', 'native_result_schema', 'response_artifact_schema',
    'workflow_schema', 'workflow_source_schema', 'workflow_expectations_schema']) {
    assert.equal(description[name].$schema, 'https://json-schema.org/draft/2020-12/schema', name);
    assert.equal(typeof description[name].$id, 'string', name);
  }
  for (const name of ['prepare', 'read', 'workflow']) assert.ok(description.commands.includes(name));
  const input = await source();
  validateSchema(description.preparation_schema, input.spec);
  const prepared = await operations.prepare(input.specFile, next('prepared'));
  validateSchema(description.prepared_schema, prepared);
  const checked = await operations.check(prepared.request_file);
  validateSchema(description.check_schema, checked);
  const invalidCount = structuredClone(checked);
  invalidCount.summary.requested = '1';
  assert.throws(() => validateSchema(description.check_schema, invalidCount));
  const invalidNative = structuredClone(checked.result);
  invalidNative.pairs[0].state = 'invented-success';
  assert.throws(() => validateSchema(description.native_result_schema, invalidNative));
  const invalidContrast = structuredClone(checked.result);
  invalidContrast.pairs[0].state = 'abstained';
  assert.throws(() => validateSchema(description.native_result_schema, invalidContrast));
  const output = await operations.run(prepared.request_file, next('result'));
  const loaded = operations.read(output.result_file);
  validateSchema(description.loaded_schema, loaded);
  validateSchema(description.result_schema, loaded.record);
  validateSchema(description.files_schema, loaded.manifest);
  const extraResult = {...loaded.record, pretend_causal_certificate: true};
  assert.throws(() => validateSchema(description.result_schema, extraResult));
});

test('CLI and reusable preparation preserve supplied metadata and exact artifact bytes', async () => {
  const supplied = await source('toy-b'), original = fs.readFileSync(supplied.specFile);
  const prepared = jsonCall(['prepare', '--spec', supplied.specFile, '--output', next('cli-prepared')]);
  const request = json(prepared.request_file);
  assert.deepEqual(request.input, supplied.spec.input);
  assert.deepEqual(request.operation, supplied.spec.operation);
  assert.deepEqual(fs.readFileSync(supplied.specFile), original);
  assert.equal(prepared.artifact_count, supplied.spec.artifacts.length);
  for (const artifact of request.artifacts) {
    const suppliedRow = supplied.spec.artifacts.find(row => row.id === artifact.id);
    assert.equal(artifact.sha256, suppliedRow.sha256);
    assert.deepEqual(fs.readFileSync(path.join(path.dirname(prepared.request_file), artifact.path)),
      fs.readFileSync(path.join(supplied.directory, suppliedRow.path)));
  }
  assert.equal(jsonCall(['check', '--request', prepared.request_file]).valid, true);
});

test('preparation refuses incomplete/conflicting metadata and changed bytes before creating output', async () => {
  const mutations = [
    row => { delete row.spec.input.permissions_ref; },
    row => { row.spec.artifacts.pop(); },
    row => { row.spec.artifacts[0].sha256 = '0'.repeat(64); },
    row => { row.spec.artifacts.push({...row.spec.artifacts[0]}); },
    row => { fs.appendFileSync(path.join(row.directory, row.spec.artifacts[0].path), 'changed'); },
  ];
  for (const mutate of mutations) {
    const supplied = await source();
    mutate(supplied);
    writeJSON(supplied.specFile, supplied.spec);
    const output = next('refused-preparation');
    await assert.rejects(operations.prepare(supplied.specFile, output));
    assert.equal(fs.existsSync(output), false);
  }
});

test('preparation refuses malformed/duplicate JSON and preserves an existing destination', async () => {
  const supplied = await source(), destination = next('prepared');
  const prepared = await operations.prepare(supplied.specFile, destination);
  const before = fs.readFileSync(prepared.request_file);
  await assert.rejects(operations.prepare(supplied.specFile, destination), /exists/);
  assert.deepEqual(fs.readFileSync(prepared.request_file), before);
  const malformed = await source();
  fs.writeFileSync(malformed.specFile, '{"schema_version":');
  const malformedOutput = next('malformed-output');
  assert.equal(call(['prepare', '--spec', malformed.specFile, '--output', malformedOutput], 2).stdout, '');
  assert.equal(fs.existsSync(malformedOutput), false);
  const duplicate = await source();
  fs.writeFileSync(duplicate.specFile, fs.readFileSync(duplicate.specFile, 'utf8').replace(
    '"controlled_model_response"', '"identified_intervention_effect", "claim_level": "controlled_model_response"'));
  await assert.rejects(operations.prepare(duplicate.specFile, next('duplicate-output')), /duplicate/i);
});

test('preparation rejects redirected source files and output ancestors without following them', async () => {
  const supplied = await source();
  const first = supplied.spec.artifacts[0], original = path.join(supplied.directory, first.path);
  const link = path.join(supplied.directory, 'redirected-artifact.json');
  fs.symlinkSync(original, link);
  first.path = 'redirected-artifact.json';
  writeJSON(supplied.specFile, supplied.spec);
  const refused = next('redirected-source-output');
  await assert.rejects(operations.prepare(supplied.specFile, refused), /canonical|redirect|regular/i);
  assert.equal(fs.existsSync(refused), false);
  const clean = await source(), actual = next('actual-parent'), redirected = next('redirected-parent');
  fs.mkdirSync(actual);
  fs.symlinkSync(actual, redirected, 'dir');
  await assert.rejects(operations.prepare(clean.specFile, path.join(redirected, 'output')), /canonical|redirect/i);
  assert.deepEqual(fs.readdirSync(actual), []);
});

test('structured result loading works after original source bundles are detached', async () => {
  const bundle = await saved();
  fs.renameSync(bundle.supplied.directory, `${bundle.supplied.directory}-detached`);
  fs.renameSync(bundle.preparedDirectory, `${bundle.preparedDirectory}-detached`);
  const loaded = operations.read(bundle.result.result_file);
  assert.equal(loaded.record.status, 'completed');
  assert.deepEqual(loaded.record.result.pairs[0].contrast.delta, [3, -2]);
  assert.equal(loaded.artifacts.length, loaded.request.artifacts.length);
  for (const row of loaded.artifacts) {
    assert.ok(row.path.startsWith(`${bundle.directory}/artifacts/`));
    assert.equal(sha256(fs.readFileSync(row.path)), row.sha256);
  }
  assert.deepEqual(jsonCall(['read', '--result', bundle.result.result_file]).record, loaded.record);
});

test('structured reader rejects coherently rehashed claim, revision, run, roster and arithmetic corruption', async () => {
  const bundle = await saved();
  const mutations = [
    record => { record.claim_level = 'identified_intervention_effect'; },
    record => { record.operation.revision = 'invented-revision'; },
    record => { record.input.operation_run_id += '-wrong-run'; },
    record => { record.result.pairs[0].pair_id += '-wrong-pair'; },
    record => { record.verified_artifacts[0].source_path = 'artifacts/unrelated.bin'; },
    record => {
      const pair = record.result.pairs[0];
      pair.contrast.delta[0] += 1;
      // Keep the explanatory text coherent with the forged numerical output.
      record.details[0] = `${pair.pair_id}: success; delta = [${pair.contrast.delta.join(', ')}]`;
    },
  ];
  for (const mutate of mutations) {
    const resultFile = rewrittenCopy(bundle.directory, mutate);
    assert.throws(() => operations.read(resultFile), /mismatch|disagrees|revision|roster|contrast|arithmetic/i);
    assert.equal(call(['read', '--result', resultFile], 2).stdout, '');
  }
});

test('structured reader rejects retained-byte tampering and manifest traversal', async () => {
  const bundle = await saved();
  const artifact = json(bundle.result.result_file).verified_artifacts[0];
  fs.appendFileSync(path.join(bundle.directory, artifact.retained_path), 'tampered');
  assert.throws(() => operations.read(bundle.result.result_file), /mismatch/i);
  assert.equal(call(['read', '--result', bundle.result.result_file], 2).stdout, '');
  const other = await saved();
  const manifestFile = path.join(other.directory, 'manifest.json'), manifest = json(manifestFile);
  manifest.files[0].path = '../request.json';
  writeJSON(manifestFile, manifest);
  assert.throws(() => operations.read(other.result.result_file), /pattern|relative|path/i);
});

test('generic JavaScript operation prepares and reuses structured results without pair/native assumptions', async () => {
  const calls = {check: 0, execute: 0};
  const tool = {
    outputRoot: '.temp/operations/counterfactuals/toolset-tests',
    descriptor: {id: 'coverage-generic-operation', revision: '1', family: 'engineering',
      title: 'Generic engineering fixture', claim_levels: ['engineering_fixture'],
      limitations: ['Engineering fixture only; no additional operation is registered.']},
    encode(input) {
      assert.deepEqual(Object.keys(input), ['items']);
      assert.ok(Array.isArray(input.items) && input.items.every(Number.isFinite));
      return Buffer.from(JSON.stringify(input.items));
    },
    references() { return []; },
    verify(_input, artifacts) { assert.equal(artifacts.size, 0); },
    context() { return {operation_run_id: 'generic-run', claim_level: 'engineering_fixture'}; },
    async check(wire) { calls.check++; return {result: {doubled: JSON.parse(wire).map(x => 2 * x)}, source_files: []}; },
    async execute(wire) { calls.execute++; return {result: {doubled: JSON.parse(wire).map(x => 2 * x)}, source_files: []}; },
    outcome(result, input) {
      assert.equal(Object.hasOwn(input, 'pairs'), false);
      assert.equal(Object.hasOwn(result, 'admitted'), false);
      return {status: 'completed', details: [`Doubled ${result.doubled.join(', ')}`],
        summary: {requested: input.items.length, succeeded: input.items.length,
          abstained: 0, failed: 0, unresolved: 0, success_rate: 1}};
    },
  };
  const generic = createToolset([tool]), inputDirectory = next('generic-source');
  fs.mkdirSync(inputDirectory);
  const specFile = path.join(inputDirectory, 'prepare.json');
  writeJSON(specFile, {schema_version: 'mityane.operation.prepare/v1',
    operation: {id: tool.descriptor.id, revision: tool.descriptor.revision}, input: {items: [2, -3, 0.5]}, artifacts: []});
  const prepared = await generic.prepare(specFile, next('generic-prepared'));
  assert.deepEqual(calls, {check: 0, execute: 0}, 'preparation never calls an execution hook');
  const checked = await generic.check(prepared.request_file);
  assert.deepEqual(checked.result, {doubled: [4, -6, 1]});
  const result = await generic.run(prepared.request_file, next('generic-result'));
  const loaded = generic.read(result.result_file);
  assert.deepEqual(loaded.record.result, {doubled: [4, -6, 1]});
  assert.equal(loaded.record.execution.native_binary, null);
  assert.deepEqual(loaded.artifacts, []);
  assert.match(generic.inspect(result.result_file), /Doubled 4, -6, 1/);
  assert.deepEqual(calls, {check: 1, execute: 1}, 'saved reads and inspection never re-execute');
});

test('API and CLI workflows preserve five shared-baseline alternatives across two providers', async () => {
  const reports = [
    await operations.workflow('OR06-PCR', {provider: 'toy-a', output: next('api-workflow')}),
    jsonCall(['workflow', 'OR06-PCR', '--provider', 'toy-b', '--output', next('cli-workflow')], 4),
  ];
  assert.notEqual(reports[0].provider, reports[1].provider);
  assert.deepEqual(reports.map(row => row.representation_dimensions), [4, 7]);
  for (const report of reports) {
    const descriptor = operations.describe('OR06-PCR');
    validateSchema(descriptor.workflow_schema, report);
    validateSchema(descriptor.workflow_expectations_schema, report.expected);
    assert.equal(report.verified, true);
    assert.equal(report.status, 'partial');
    assert.deepEqual(report.summary, {requested: 5, succeeded: 3, abstained: 1, failed: 1,
      unresolved: 0, success_rate: 0.6});
    const loaded = operations.read(report.result_file);
    assert.deepEqual(loaded.record.result.pairs.map(row => row.state), ['success', 'success', 'success', 'abstained', 'failed']);
    assert.deepEqual(loaded.record.result.pairs.map(row => row.contrast?.delta ?? null),
      [[1.5, -1], [3, -2], [-1.5, 1], null, null]);
    assert.deepEqual(report.expected.pairs.map(row => row.signed_change), [1, 2, -1, 10, 1]);
    assert.deepEqual(report.expected.pairs.map(row => row.magnitude), [1, 2, 1, 10, 1]);
    const baselines = loaded.request.input.pairs.map(row => row.baseline.input.id);
    assert.equal(new Set(baselines).size, 1, 'repeated baseline dependence stays explicit');
    assert.match(loaded.request.input.sampling_rule, /not independent/);
    assert.equal(jsonCall(['read', '--result', report.result_file], 4).record.summary.requested, 5);
    assert.match(call(['inspect', '--result', report.result_file]).stdout, /failed 1; unresolved 0/);
  }
});

test('preparation preserves unsupported claim metadata and typed admission refusal remains reusable', async () => {
  const supplied = await source();
  supplied.spec.input.claim_level = 'identified_intervention_effect';
  writeJSON(supplied.specFile, supplied.spec);
  const prepared = await operations.prepare(supplied.specFile, next('unsupported-prepared'));
  assert.equal(json(prepared.request_file).input.claim_level, 'identified_intervention_effect');
  const checked = await operations.check(prepared.request_file);
  assert.equal(checked.status, 'rejected');
  assert.equal(checked.valid, false);
  assert.ok(checked.result.admission_diagnostics.some(row => row.code === 'UNSUPPORTED_CLAIM'));
  const result = await operations.run(prepared.request_file, next('unsupported-result'));
  const loaded = jsonCall(['read', '--result', result.result_file], 3);
  assert.equal(loaded.record.result.admitted, false);
  assert.deepEqual(loaded.record.result.pairs, []);
  assert.deepEqual(loaded.record.summary, {requested: 1, succeeded: 0, abstained: 0, failed: 0,
    unresolved: 0, success_rate: null});
  assert.equal(loaded.request.input.pairs.length, 1);
});
