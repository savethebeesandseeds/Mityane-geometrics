// SPDX-License-Identifier: MIT
// Engineering checks use artificial pre-recorded values, never scientific data.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {operations} from '../../tools.mjs';
import {createWorkflowSource} from './workflow.mjs';
import {runSweep, sweepSpaceSchema, sweepCatalogSchema, sweepResultSchema} from './sweep.mjs';
import {validateSchema} from '../../../shared/representation/tools/v1/schemas.mjs';

assert.equal(process.platform, 'linux', 'Run sweep checks in the managed Linux container.');
const PARENT = '/mityane/.temp/operations/counterfactuals/toolset-tests';
fs.mkdirSync(PARENT, {recursive: true});
const WORK = fs.mkdtempSync(path.join(PARENT, 'sweep-'));
let sequence = 0;
const next = name => path.join(WORK, `${name}-${++sequence}`);
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const write = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
let supplied;
async function source() {
  if (supplied) return supplied;
  const recorded = await createWorkflowSource({provider: 'toy-a', output: next('source')});
  const prepared = await operations.prepare(recorded.spec_file, next('prepared'));
  supplied = {request: prepared.request_file, input: json(prepared.request_file).input};
  return supplied;
}
async function specification(axis = {name: 'setting', operation: 'set', values: [0, 2, 3, 11]}) {
  const base = await source(), filename = next('space') + '.json';
  const value = {schema_version: 'mityane.operation.sweep-space/v1', baseline_pair_id: base.input.pairs[0].id,
    request_id: `sweep-request-${sequence}`, operation_run_id: `sweep-run-${sequence}`,
    sampling_rule: 'Complete declared artificial grid sharing one observed baseline; no population estimate',
    axes: [axis], hold: ['load']};
  validateSchema(sweepSpaceSchema, value); write(filename, value); return {filename, value};
}
const sweep = options => runSweep({...options, operations});

test('plan retains complete unsupported grid cells without fabricating model records', async () => {
  const base = await source(), space = await specification();
  const result = await sweep({request: base.request, space: space.filename, output: next('plan')});
  validateSchema(sweepResultSchema, result);
  assert.equal(result.mode, 'plan'); assert.equal(result.status, 'completed');
  assert.equal(result.materialization_required, true); assert.equal(result.request_file, null);
  assert.equal(result.check, null); assert.equal(result.cells, 4); assert.equal(result.blocked_support, 1);
  const plan = json(result.plan_file);
  assert.deepEqual(plan.cells.map(row => row.attributes[0].value), [0, 2, 3, 11]);
  assert.equal(plan.cells[3].planning_status, 'blocked_support');
  assert.ok(plan.cells.every(row => !Object.hasOwn(row, 'response_values') && !Object.hasOwn(row, 'state')));
  assert.equal(fs.existsSync(path.join(path.dirname(result.plan_file), 'prepared')), false);
});

test('decimal inclusive range and scale planning are deterministic', async () => {
  const base = await source(), space = await specification({name: 'setting', operation: 'add', start: 0.1, end: 0.3, step: 0.1});
  const result = await sweep({request: base.request, space: space.filename, output: next('decimal')});
  assert.deepEqual(json(result.plan_file).cells.map(row => row.attributes[0].value), [1.1, 1.2, 1.3]);
  const scaled = await specification({name: 'setting', operation: 'scale', values: [0, 0.25, 1]});
  const scaling = await sweep({request: base.request, space: scaled.filename, output: next('scale')});
  assert.deepEqual(json(scaling.plan_file).cells.map(row => row.attributes[0].value), [0, 0.25, 1]);
  assert.equal(json(scaling.plan_file).cells[2].attributes[0].value, base.input.pairs[0].baseline.attributes[0].value,
    'A declared no-change cell remains in the grid');
});

test('recorded catalog packages all cells and retains actual support abstention for check/run', async () => {
  const base = await source(), space = await specification(), catalogFile = next('catalog') + '.json';
  const wanted = [0, 2, 3, 11];
  const catalog = {schema_version: 'mityane.operation.sweep-catalog/v1', entries: wanted.map(value => ({coordinate: [value],
    request: path.relative(WORK, base.request),
    pair_id: base.input.pairs.find(pair => pair.alternative.attributes[0].value === value &&
      pair.alternative.space.artifact.id === base.input.expected_space.artifact.id).id}))};
  validateSchema(sweepCatalogSchema, catalog); write(catalogFile, catalog);
  const result = await sweep({request: base.request, space: space.filename, catalog: catalogFile, output: next('recorded')});
  validateSchema(sweepResultSchema, result);
  assert.equal(result.mode, 'recorded'); assert.equal(result.materialization_required, false);
  assert.deepEqual(result.check.summary, {requested: 4, succeeded: 3, abstained: 1, failed: 0, unresolved: 0, success_rate: 0.75});
  const generated = json(result.request_file);
  assert.deepEqual(generated.input.expected_pipeline, base.input.expected_pipeline);
  assert.ok(generated.input.pairs.every(pair => JSON.stringify(pair.baseline) === JSON.stringify(base.input.pairs[0].baseline)));
  assert.deepEqual(generated.input.pairs.map(pair => pair.support), catalog.entries.map(entry =>
    base.input.pairs.find(pair => pair.id === entry.pair_id).support));
  const run = await operations.run(result.request_file, next('run'));
  const loaded = operations.read(run.result_file);
  assert.deepEqual(loaded.record.summary, result.check.summary);
  assert.equal(loaded.record.result.pairs[3].state, 'abstained');
  assert.equal(loaded.record.result.pairs[3].contrast, null);
  assert.ok(loaded.record.result.pairs[3].diagnostics.some(row => row.code === 'OUTSIDE_DECLARED_SUPPORT'));
});

test('grid rejects illegal masks, incomplete catalogs, duplicate values and unbounded or ambiguous ranges', async () => {
  const base = await source();
  for (const axis of [
    {name: 'load', operation: 'set', values: [3]},
    {name: 'setting', operation: 'set', values: [2, 2]},
    {name: 'setting', operation: 'set', start: 0, end: 1, step: 0},
    {name: 'setting', operation: 'set', start: 0, end: 1, step: -1},
    {name: 'setting', operation: 'set', start: 0, end: 1, step: 0.3},
    {name: 'setting', operation: 'set', start: 0, end: 10000, step: 1},
  ]) {
    const space = await specification(axis);
    await assert.rejects(sweep({request: base.request, space: space.filename, output: next('invalid')}));
  }
  const space = await specification(); space.value.hold = []; write(space.filename, space.value);
  await assert.rejects(sweep({request: base.request, space: space.filename, output: next('no-hold')}), /partition/);
  space.value.hold = ['load']; write(space.filename, space.value);
  const catalogFile = next('incomplete-catalog') + '.json';
  write(catalogFile, {schema_version: 'mityane.operation.sweep-catalog/v1', entries: [
    {coordinate: [2], request: path.relative(WORK, base.request), pair_id: base.input.pairs[0].id}]});
  await assert.rejects(sweep({request: base.request, space: space.filename, catalog: catalogFile, output: next('incomplete')}), /every Cartesian/);
});

test('packaging retains an already recorded upstream failure and returns partial', async () => {
  const base = await source(), space = await specification({name: 'setting', operation: 'set', values: [3]});
  const failedSource = next('recorded-failure'); fs.cpSync(path.dirname(base.request), failedSource, {recursive: true});
  const filename = path.join(failedSource, 'request.json'), envelope = json(filename);
  const selected = envelope.input.pairs.find(pair => pair.alternative.attributes[0].value === 3);
  selected.alternative.upstream_succeeded = false; write(filename, envelope);
  const catalogFile = next('failed-catalog') + '.json';
  write(catalogFile, {schema_version: 'mityane.operation.sweep-catalog/v1', entries: [
    {coordinate: [3], request: path.relative(WORK, filename), pair_id: selected.id}]});
  const result = await sweep({request: base.request, space: space.filename, catalog: catalogFile, output: next('failed-grid')});
  assert.equal(result.status, 'partial');
  assert.equal(result.check.summary.failed, 1);
  assert.equal(result.check.result.pairs[0].state, 'failed');
  assert.equal(result.check.result.pairs[0].contrast, null);
});

test('sweep claims remain restricted and artifact tampering is rejected before publication', async () => {
  const base = await source(), space = await specification();
  const altered = next('altered'); fs.cpSync(path.dirname(base.request), altered, {recursive: true});
  const filename = path.join(altered, 'request.json'), envelope = json(filename);
  envelope.input.claim_level = 'identified_intervention_effect'; write(filename, envelope);
  await assert.rejects(sweep({request: filename, space: space.filename, output: next('causal')}), error => error.exitCode === 3);
  envelope.input.claim_level = 'controlled_model_response'; write(filename, envelope);
  fs.appendFileSync(path.join(altered, envelope.artifacts[0].path), ' ');
  const output = next('tampered');
  await assert.rejects(sweep({request: filename, space: space.filename, output}), /hash mismatch/);
  assert.equal(fs.existsSync(output), false);
});
