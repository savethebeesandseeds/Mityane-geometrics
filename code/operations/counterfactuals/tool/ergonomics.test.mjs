// SPDX-License-Identifier: MIT
// The common public Node API and CLI are exercised together on a small grid of
// already recorded engineering alternatives, including one support abstention.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import test from 'node:test';
import {operations} from '../../tools.mjs';
import {createWorkflowSource} from './workflow.mjs';
import {validateSchema} from '../../../shared/representation/tools/v1/schemas.mjs';

assert.equal(process.platform, 'linux', 'Run integration checks in the managed Linux container.');
const ROOT = '/mityane', CLI = path.join(ROOT, 'code/operations/tools.mjs');
const PARENT = path.join(ROOT, '.temp/operations/counterfactuals/toolset-tests');
fs.mkdirSync(PARENT, {recursive: true});
const WORK = fs.mkdtempSync(path.join(PARENT, 'ergonomics-'));
let sequence = 0, cached;
const next = name => path.join(WORK, `${name}-${++sequence}`);
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const write = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
function call(args, expected = 0) {
  const result = spawnSync(process.execPath, [CLI, ...args], {cwd: ROOT, env: process.env,
    encoding: 'utf8', timeout: 300000, maxBuffer: 32 * 1024 * 1024});
  assert.equal(result.error, undefined, result.error?.message); assert.equal(result.signal, null, result.stderr);
  assert.equal(result.status, expected,
    `CLI ${args.join(' ')} exited ${result.status}, expected ${expected}\n${result.stdout}\n${result.stderr}`);
  return result;
}
const jsonCall = (args, expected = 0) => JSON.parse(call(args, expected).stdout);
async function scenario() {
  if (cached) return cached;
  const source = await createWorkflowSource({provider: 'toy-a', output: next('source')});
  const prepared = await operations.prepare(source.spec_file, next('prepared'));
  const envelope = json(prepared.request_file), operands = [-1, 1, 2, 10];
  const space = next('space') + '.json', catalog = next('catalog') + '.json';
  write(space, {schema_version: 'mityane.operation.sweep-space/v1',
    baseline_pair_id: envelope.input.pairs[0].id, request_id: 'ergonomics-grid-request-v1',
    operation_run_id: 'ergonomics-grid-run-v1',
    sampling_rule: 'The complete declared artificial Cartesian grid shares one observed baseline; no population estimate',
    axes: [{name: 'setting', operation: 'add', values: operands}], hold: ['load']});
  write(catalog, {schema_version: 'mityane.operation.sweep-catalog/v1', entries: operands.map(value => ({
    coordinate: [value], request: path.relative(WORK, prepared.request_file),
    pair_id: envelope.input.pairs.find(pair => pair.alternative.attributes[0].value === 1 + value &&
      pair.alternative.space.artifact.id === envelope.input.expected_space.artifact.id).id}))});
  const grid = await operations.sweep('OR06-PCR', {request: prepared.request_file, space, catalog, output: next('grid')});
  const left = await operations.run(grid.request_file, next('left'));
  const right = await operations.run(grid.request_file, next('right'));
  cached = {request: prepared.request_file, envelope, space, catalog, grid, left, right};
  return cached;
}

test('public Node API packages a complete recorded grid, checks/runs twice and diffs without imputing abstention', async () => {
  const source = await scenario();
  assert.equal(source.grid.mode, 'recorded'); assert.equal(source.grid.cells, 4);
  assert.equal(source.grid.blocked_support, 1); assert.equal(source.grid.materialization_required, false);
  const checked = await operations.check(source.grid.request_file);
  assert.deepEqual(checked.summary, {requested: 4, succeeded: 3, abstained: 1,
    failed: 0, unresolved: 0, success_rate: 0.75});
  for (const result of [source.left, source.right]) {
    const loaded = operations.read(result.result_file);
    assert.equal(loaded.record.status, 'completed'); assert.deepEqual(loaded.record.summary, checked.summary);
    assert.equal(loaded.record.result.pairs[3].state, 'abstained');
    assert.equal(loaded.record.result.pairs[3].contrast, null);
    assert.ok(loaded.record.result.pairs[3].diagnostics.some(row => row.code === 'OUTSIDE_DECLARED_SUPPORT'));
  }
  const result = await operations.diff('OR06-PCR', {left: source.left.result_file,
    right: source.right.result_file, output: next('api-diff')});
  assert.equal(result.status, 'completed'); assert.equal(result.summary.requested, 4);
  assert.equal(result.summary.comparable, 3); assert.equal(result.summary.agreed, 3);
  assert.equal(result.summary.missing_contrasts, 1); assert.equal(result.summary.state_cross['abstained/abstained'], 1);
  assert.ok(result.comparison.pairs.slice(0, 3).every(row => JSON.stringify(row.right_minus_left) === '[0,0]'));
  assert.equal(result.comparison.pairs[3].right_minus_left, null);
  assert.equal(result.comparison.pairs[3].within_tolerance, null);
  const descriptor = operations.describe('OR06-PCR');
  for (const name of ['sweep_space_schema', 'sweep_catalog_schema', 'sweep_result_schema',
    'diff_options_schema', 'diff_result_schema', 'diff_files_schema', 'verify_study_result_schema'])
    assert.equal(descriptor[name].$schema, 'https://json-schema.org/draft/2020-12/schema', name);
  validateSchema(descriptor.diff_result_schema, result.comparison);
  assert.deepEqual(descriptor.claim_levels, ['controlled_model_response']);
  assert.deepEqual(descriptor.unsupported_profiles.map(row => row.claim_level),
    ['observational_contrast', 'identified_intervention_effect']);
  assert.ok(descriptor.unsupported_profiles.every(row => row.callable === false));
});

test('CLI exposes scenario plans, verified diffs, tolerances and unchanged typed exit codes', async () => {
  const source = await scenario();
  const plan = jsonCall(['sweep', 'OR06-PCR', '--request', source.request,
    '--space', source.space, '--output', next('cli-plan')]);
  assert.equal(plan.mode, 'plan'); assert.equal(plan.materialization_required, true);
  assert.equal(plan.request_file, null); assert.equal(plan.blocked_support, 1);
  const compared = jsonCall(['diff', 'OR06-PCR', '--left', source.left.result_file,
    '--right', source.right.result_file, '--output', next('cli-diff'), '--abs-tolerance', '0.0', '--rel-tolerance', '1e-12']);
  assert.equal(compared.status, 'completed'); assert.equal(compared.summary.agreed, 3);
  assert.equal(compared.summary.missing_contrasts, 1);
  call(['diff', 'OR06-PCR', '--left', source.left.result_file, '--right', source.right.result_file,
    '--output', next('cli-invalid-tolerance'), '--abs-tolerance', '-1'], 2);
  call(['verify-study', 'OR06-PCR', '--attempt', next('missing-retained-study'), '--output', next('cli-missing-study')], 2);
  const unsupportedDirectory = next('unsupported'); fs.cpSync(path.dirname(source.request), unsupportedDirectory, {recursive: true});
  const unsupported = path.join(unsupportedDirectory, 'request.json'), envelope = json(unsupported);
  envelope.input.claim_level = 'identified_intervention_effect'; write(unsupported, envelope);
  call(['sweep', 'OR06-PCR', '--request', unsupported, '--space', source.space, '--output', next('cli-unsupported-claim')], 3);
  const failedDirectory = next('failed'); fs.cpSync(path.dirname(source.grid.request_file), failedDirectory, {recursive: true});
  const failedRequest = path.join(failedDirectory, 'request.json'), failed = json(failedRequest);
  failed.input.pairs[0].alternative.upstream_succeeded = false; write(failedRequest, failed);
  const partial = jsonCall(['run', '--request', failedRequest, '--output', next('cli-partial-run')], 4);
  const partialDiff = jsonCall(['diff', 'OR06-PCR', '--left', source.left.result_file,
    '--right', partial.result_file, '--output', next('cli-partial-diff')], 4);
  assert.equal(partialDiff.status, 'partial'); assert.equal(partialDiff.summary.state_cross['success/failed'], 1);
  assert.equal(partialDiff.comparison.pairs[0].right_minus_left, null);
  const changedDirectory = next('mismatched'); fs.cpSync(path.dirname(source.grid.request_file), changedDirectory, {recursive: true});
  const changedRequest = path.join(changedDirectory, 'request.json'), changed = json(changedRequest);
  changed.input.pairs[0].change_definition += '; separately declared edit'; write(changedRequest, changed);
  const changedRun = jsonCall(['run', '--request', changedRequest, '--output', next('cli-mismatch-run')]);
  call(['diff', 'OR06-PCR', '--left', source.left.result_file, '--right', changedRun.result_file,
    '--output', next('cli-mismatch-refused')], 2);
  const description = jsonCall(['describe', 'OR06-PCR']);
  assert.deepEqual(description.claim_levels, ['controlled_model_response']);
  for (const command of ['list', 'describe', 'example', 'prepare', 'check', 'run', 'inspect', 'read', 'workflow', 'sweep', 'diff', 'verify-study']) {
    if (command !== 'list') assert.ok(description.commands.includes(command), command);
  }
});
