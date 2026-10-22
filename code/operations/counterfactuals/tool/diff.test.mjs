// SPDX-License-Identifier: MIT
// Engineering records only. Run in the managed Linux container; these tests
// do not execute an encoder or constitute evaluation evidence for a model.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {operations} from '../../tools.mjs';
import {formatSummary} from '../../../shared/representation/tools/v1/runtime.mjs';
import {validateSchema} from '../../../shared/representation/tools/v1/schemas.mjs';
import {sha256} from '../../../shared/artifacts/node/v1/files.mjs';
import {makeExampleRequest} from './fixture.mjs';
import {collectArtifactRefs} from './request.mjs';
import {runDiff, diffResultSchema, diffOutputSchema, diffFilesSchema, experimentalRosterFingerprint} from './diff.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
assert.equal(process.platform, 'linux', 'Run diff tests in the managed Linux container.');
assert.equal(ROOT, '/mityane', 'Use the authoritative managed-container checkout.');
const PARENT = path.join(ROOT, '.temp/operations/counterfactuals/toolset-tests');
fs.mkdirSync(PARENT, {recursive: true});
const WORK = fs.mkdtempSync(path.join(PARENT, 'diff-'));
let sequence = 0, cached;
const next = name => path.join(WORK, `${name}-${++sequence}`);
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const writeJSON = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
const diff = options => runDiff({...options, operations});
function artifact(fixture, id, payload) {
  const bytes = Buffer.from(`${JSON.stringify(payload, null, 2)}\n`), hash = sha256(bytes);
  fixture.artifacts.push({id, sha256: hash, bytes});
  return {id, sha256: hash};
}
function cloneArtifact(fixture, ref, suffix) {
  const source = fixture.artifacts.find(row => row.id === ref.id && row.sha256 === ref.sha256);
  return artifact(fixture, `${ref.id}-${suffix}`, JSON.parse(source.bytes));
}
function addPair(fixture, suffix) {
  const row = structuredClone(fixture.input.pairs[0]);
  row.id += `-${suffix}`;
  for (const key of ['input', 'representation', 'response'])
    row.alternative[key] = cloneArtifact(fixture, row.alternative[key], suffix);
  row.alternative.response_execution_id += `-${suffix}`;
  fixture.input.pairs.push(row);
}
function publishFixture(fixture) {
  const directory = next('source'); fs.mkdirSync(directory); fs.mkdirSync(path.join(directory, 'artifacts'));
  const artifacts = collectArtifactRefs(fixture.input).map((ref, index) => {
    const row = fixture.artifacts.find(value => value.id === ref.id && value.sha256 === ref.sha256);
    assert.ok(row, `fixture contains ${ref.id}`);
    const relative = `artifacts/${index}.json`;
    fs.writeFileSync(path.join(directory, relative), row.bytes);
    return {...ref, path: relative};
  });
  const filename = path.join(directory, 'request.json');
  writeJSON(filename, {schema_version: 'mityane.operation.request/v1',
    operation: {id: 'OR06-PCR', revision: '0.1.0'}, input: fixture.input, artifacts});
  return filename;
}
async function fixtures() {
  if (cached) return cached;
  const a = makeExampleRequest({providerId: 'diff-provider-A', dimensions: 4});
  const b = makeExampleRequest({providerId: 'diff-provider-B', dimensions: 7});
  for (const row of [a, b]) { addPair(row, 'support'); addPair(row, 'upstream'); }
  // Both providers consume exactly the same declared raw experiment. Their
  // representation/pipeline/recorded response artifacts remain distinct.
  for (const key of ['population_id', 'sampling_rule', 'response_horizon', 'permitted_interpretation',
    'temporal_mode', 'as_of', 'non_temporal_justification', 'response_axes', 'selected_output_position',
    'attribute_rules', 'control_tolerance']) b.input[key] = structuredClone(a.input[key]);
  b.artifacts.push(...a.artifacts);
  for (const role of ['support_policy', 'constructor']) {
    const shared = a.input.expected_pipeline.dependencies.find(row => row.role === role);
    Object.assign(b.input.expected_pipeline.dependencies.find(row => row.role === role), structuredClone(shared));
  }
  for (const [index, row] of b.input.pairs.entries()) {
    const common = a.input.pairs[index];
    for (const key of ['id', 'unit_id', 'entity_id', 'group_id', 'constructor', 'control_evidence',
      'change_definition', 'interpretation']) row[key] = structuredClone(common[key]);
    for (const side of ['baseline', 'alternative']) for (const key of ['input', 'source_observation',
      'parent_input_id', 'origin', 'timing', 'attributes', 'output_position', 'mask', 'axes'])
      row[side][key] = structuredClone(common[side][key]);
  }
  b.input.expected_pipeline.artifact = artifact(b, 'diff-provider-B:shared-experiment-pipeline',
    {identity: 'engineering-diff-pipeline-v1', space: b.input.expected_space.artifact,
      response_model: b.input.expected_pipeline.response_model,
      configuration_sha256: b.input.expected_pipeline.configuration_sha256,
      dependencies: b.input.expected_pipeline.dependencies, scope: 'Artificial recorded engineering pipeline'});
  for (const row of b.input.pairs) for (const side of ['baseline', 'alternative'])
    row[side].pipeline = b.input.expected_pipeline.artifact;
  b.input.pairs[0].alternative.response_values = [5.5, -3];
  b.input.pairs[0].alternative.response = artifact(b, 'diff-provider-B:altered-recorded-response',
    {axes: b.input.response_axes, values: [5.5, -3]});
  a.input.pairs[1].support = {state: 'rejected', evidence: artifact(a, 'diff-support-refusal',
    {scope: 'Artificial engineering support refusal', state: 'rejected'}), detail: 'Artificial support rejection'};
  a.input.pairs[2].alternative.upstream_succeeded = false;
  const left = await operations.run(publishFixture(a), next('left'));
  const right = await operations.run(publishFixture(b), next('right'));
  cached = {left: left.result_file, right: right.result_file,
    leftRecord: operations.read(left.result_file), rightRecord: operations.read(right.result_file)};
  return cached;
}
function rewrittenCopy(original, mutate) {
  const directory = next('rewritten'); fs.cpSync(path.dirname(original), directory, {recursive: true});
  const record = json(path.join(directory, 'result.json')), request = json(path.join(directory, 'request.json'));
  mutate(record, request, directory);
  writeJSON(path.join(directory, 'request.json'), request);
  record.input.sha256 = sha256(fs.readFileSync(path.join(directory, 'request.json')));
  writeJSON(path.join(directory, 'result.json'), record);
  fs.writeFileSync(path.join(directory, 'summary.txt'), formatSummary(record));
  const manifest = json(path.join(directory, 'manifest.json'));
  for (const row of manifest.files) {
    const bytes = fs.readFileSync(path.join(directory, row.path)); row.bytes = bytes.length; row.sha256 = sha256(bytes);
  }
  writeJSON(path.join(directory, 'manifest.json'), manifest);
  return path.join(directory, 'result.json');
}

test('diff aligns genuine distinct pipelines on common raw records and preserves mixed states', async () => {
  const source = await fixtures();
  assert.notDeepEqual(source.leftRecord.request.input.expected_space, source.rightRecord.request.input.expected_space);
  assert.equal(experimentalRosterFingerprint(source.leftRecord.request.input),
    experimentalRosterFingerprint(source.rightRecord.request.input));
  const output = await diff({left: source.left, right: source.right, output: next('comparison')});
  validateSchema(diffOutputSchema, output);
  assert.equal(output.status, 'partial');
  const record = json(output.comparison_file);
  validateSchema(diffResultSchema, record); validateSchema(diffFilesSchema, json(output.manifest_file));
  assert.deepEqual(record.pairs[0].right_minus_left, [0.5, 0]);
  assert.deepEqual(record.pairs[0].absolute_difference, [0.5, 0]);
  assert.equal(record.pairs[0].within_tolerance, false);
  assert.equal(record.summary.requested, 3); assert.equal(record.summary.comparable, 1);
  assert.equal(record.summary.diverged, 1); assert.equal(record.summary.missing_contrasts, 2);
  for (const [index, state] of [[1, 'abstained'], [2, 'failed']]) {
    assert.equal(record.pairs[index].left.state, state); assert.equal(record.pairs[index].right.state, 'success');
    assert.equal(record.pairs[index].right_minus_left, null); assert.equal(record.pairs[index].absolute_difference, null);
    assert.equal(record.pairs[index].within_tolerance, null);
    assert.equal(record.summary.state_cross[`${state}/success`], 1);
    assert.ok(record.pairs[index].left.diagnostics.length);
  }
  for (const side of ['left', 'right']) operations.read(path.join(path.dirname(output.comparison_file), side, 'result.json'));
  for (const row of json(output.manifest_file).files) {
    const bytes = fs.readFileSync(path.join(path.dirname(output.manifest_file), row.path));
    assert.equal(bytes.length, row.bytes); assert.equal(sha256(bytes), row.sha256);
  }
});

test('diff accepts declared tolerances and retains all successful/abstained source states', async () => {
  const source = await fixtures();
  const output = await diff({left: source.right, right: source.right, output: next('self'), absoluteTolerance: 0.5});
  assert.equal(output.status, 'completed'); assert.equal(output.summary.agreed, 3);
  assert.equal(output.summary.diverged, 0); assert.equal(output.summary.missing_contrasts, 0);
  const tolerant = await diff({left: source.left, right: source.right, output: next('tolerant'), relativeTolerance: 1 / 7});
  assert.equal(tolerant.summary.agreed, 1); assert.equal(tolerant.summary.diverged, 0);
  assert.equal(tolerant.status, 'partial', 'numeric tolerance never repairs a failed source pair');
  for (const value of [-1, NaN, Infinity, '1']) {
    const directory = next('bad-tolerance');
    await assert.rejects(diff({left: source.left, right: source.right, output: directory, absoluteTolerance: value}), /finite nonnegative/);
    assert.equal(fs.existsSync(directory), false);
  }
});

test('diff rejects incompatible ordered rosters/support/raw identities before creating output', async () => {
  const source = await fixtures();
  const mutations = [
    (_record, request) => { request.input.population_id += '-different'; },
    (_record, request) => { request.input.pairs[0].change_definition += '; different edit'; },
    (_record, request) => { request.input.attribute_rules[0].maximum = 9; },
    (_record, request) => { request.input.pairs[0].baseline.timing.vintage += '-different'; },
  ];
  for (const mutate of mutations) {
    const changed = rewrittenCopy(source.right, mutate), output = next('roster-refused');
    await assert.rejects(diff({left: source.left, right: changed, output}), /roster mismatch/);
    assert.equal(fs.existsSync(output), false);
  }
  const input = structuredClone(source.leftRecord.request.input), original = experimentalRosterFingerprint(input);
  input.pairs.reverse(); assert.notEqual(experimentalRosterFingerprint(input), original);
  input.pairs.reverse(); input.pairs[0].baseline.input.id += '-new-identity';
  assert.notEqual(experimentalRosterFingerprint(input), original, 'byte-equal raw inputs with different IDs are distinct records');
  input.pairs[0].baseline.input = source.leftRecord.request.input.pairs[0].baseline.input;
  input.pairs[0].baseline.mask.id += '-provider-local';
  assert.equal(experimentalRosterFingerprint(input), original, 'byte-identical masks can use provider-local IDs');
  input.pairs[0].baseline.mask.sha256 = '0'.repeat(64);
  assert.notEqual(experimentalRosterFingerprint(input), original, 'mask contents remain bound');
  input.pairs[0].baseline.mask.sha256 = source.leftRecord.request.input.pairs[0].baseline.mask.sha256;
  input.response_axes[0].units = 'different-units';
  assert.notEqual(experimentalRosterFingerprint(input), original, 'response semantics remain bound');
});

test('diff independently verifies saved bytes, response joins and admission before publication', async () => {
  const source = await fixtures();
  const tampered = rewrittenCopy(source.right, () => {});
  fs.appendFileSync(tampered, ' ');
  const tamperedOutput = next('tampered-refused');
  await assert.rejects(diff({left: source.left, right: tampered, output: tamperedOutput}), /size mismatch|hash mismatch/);
  assert.equal(fs.existsSync(tamperedOutput), false);
  const forged = rewrittenCopy(source.right, record => { record.result.pairs[0].contrast.delta[0] += 1; });
  await assert.rejects(diff({left: source.left, right: forged, output: next('forged-refused')}), /arithmetic|binding/);
  const unsupported = makeExampleRequest(); unsupported.input.claim_level = 'observational_contrast';
  const rejected = await operations.run(publishFixture(unsupported), next('rejected-result'));
  const rejectedOutput = next('admission-refused');
  await assert.rejects(diff({left: rejected.result_file, right: rejected.result_file, output: rejectedOutput}),
    error => error.exitCode === 3 && /admitted controlled_model_response/.test(error.message));
  assert.equal(fs.existsSync(rejectedOutput), false);
});

test('diff preserves retained unresolved outcomes and never invokes numerical execution', async () => {
  const source = await fixtures();
  // Explicit engineering interruption record, coherently resealed for the
  // reader's record-level invariants. It is not a new native execution claim.
  const interrupted = rewrittenCopy(source.right, record => {
    const row = record.result.pairs[0]; row.state = 'unresolved'; row.contrast = null;
    row.diagnostics = [{code: 'INTERRUPTED', record_id: row.pair_id, message: 'Engineering retained interruption fixture'}];
    record.summary = {...record.summary, succeeded: 2, unresolved: 1, success_rate: 2 / 3};
    record.result.summary = {...record.summary}; record.status = 'partial';
    record.details = record.result.pairs.flatMap(pair => [
      `${pair.pair_id}: ${pair.state}${pair.contrast ? `; delta = [${pair.contrast.delta.join(', ')}]` : ''}`,
      ...pair.diagnostics.map(item => `  ${item.code}: ${item.message}`),
    ]);
  });
  const previous = process.env.MITYANE_COUNTERFACTUALS_TOOL_BINARY;
  process.env.MITYANE_COUNTERFACTUALS_TOOL_BINARY = '/opt/mityane-geometrics/nonexistent-diff-must-not-execute';
  try {
    const result = await diff({left: source.right, right: interrupted, output: next('unresolved')});
    assert.equal(result.status, 'partial'); assert.equal(result.summary.state_cross['success/unresolved'], 1);
    assert.equal(result.comparison.pairs[0].right_minus_left, null);
    assert.equal(result.comparison.pairs[0].right.diagnostics[0].code, 'INTERRUPTED');
  } finally {
    if (previous === undefined) delete process.env.MITYANE_COUNTERFACTUALS_TOOL_BINARY;
    else process.env.MITYANE_COUNTERFACTUALS_TOOL_BINARY = previous;
  }
});
