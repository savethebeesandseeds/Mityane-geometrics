// SPDX-License-Identifier: MIT
// Artificial recorded alternatives exercise the public file boundary. No model
// execution, fitting, scientific inference or frozen-study replay occurs here.
import path from 'node:path';
import {mkdirSync} from 'node:fs';
import {makeExampleRequest} from './fixture.mjs';
import {collectArtifactRefs} from './request.mjs';
import {nativeResultSchema} from './schemas.mjs';
import {operationSchema, summarySchema, loadedSchema, validateSchema}
  from '../../../shared/representation/tools/v1/schemas.mjs';
import {createExclusiveDirectory, jsonBytes, sha256, writeNew, requireCondition as require}
  from '../../../shared/artifacts/node/v1/files.mjs';

const OUTPUT_ROOT = '/mityane/.temp/operations/counterfactuals';
const OPERATION = Object.freeze({id: 'OR06-PCR', revision: '0.1.0'});
const PROVIDERS = Object.freeze({
  'toy-a': {providerId: 'example-provider-A', dimensions: 4},
  'toy-b': {providerId: 'example-provider-B', dimensions: 7},
});
const SCOPE = 'Artificial recorded-response engineering workflow; no scientific or provider-quality evidence';

const vectorSchema = {type: 'array', items: {type: 'number'}};
const expectedPairSchema = {
  type: 'object', additionalProperties: false,
  required: ['pair_id', 'signed_change', 'magnitude', 'direction', 'state', 'diagnostic', 'delta'],
  properties: {
    pair_id: {type: 'string'}, signed_change: {type: 'number'}, magnitude: {type: 'number', minimum: 0},
    direction: {enum: ['increase', 'decrease']}, state: {enum: ['success', 'abstained', 'failed']},
    diagnostic: {anyOf: [{type: 'null'}, {type: 'string'}]},
    delta: {anyOf: [{type: 'null'}, vectorSchema]},
  },
};

// Other record schemas are published by the common boundary/operation adapter.
// This descriptor covers the workflow-owned expectations independently.
export const workflowExpectationsSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:workflow-expectations:1',
  type: 'object', additionalProperties: false, required: ['summary', 'pairs'],
  properties: {
    summary: {type: 'object', additionalProperties: false,
      required: ['requested', 'succeeded', 'abstained', 'failed', 'unresolved', 'success_rate'],
      properties: {requested: {const: 5}, succeeded: {const: 3}, abstained: {const: 1}, failed: {const: 1},
        unresolved: {const: 0}, success_rate: {const: 0.6}}},
    pairs: {type: 'array', minItems: 5, maxItems: 5, items: expectedPairSchema},
  },
};

const object = properties => ({type: 'object', additionalProperties: false,
  required: Object.keys(properties), properties});
const filenameSchema = {type: 'string', minLength: 1};
const providerSchema = {type: 'string', enum: ['toy-a', 'toy-b']};
const dimensionsSchema = {type: 'integer', enum: [4, 7]};
export const workflowSourceSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:workflow-source:1',
  title: 'OR06-PCR artificial recorded workflow source report v1',
  ...object({schema_version: {const: 'mityane.operation.workflow-source/v1'},
    operation: {...operationSchema, const: OPERATION}, provider: providerSchema,
    representation_dimensions: dimensionsSchema, spec_file: filenameSchema, expected: workflowExpectationsSchema}),
};
export const workflowResultSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:workflow:1',
  title: 'OR06-PCR artificial recorded workflow result v1',
  description: 'Engineering expectations and verified saved comparison references. The expected partial result '
    + 'includes a support abstention and an incompatible-space failure; this is not scientific evidence.',
  ...object({schema_version: {const: 'mityane.operation.workflow/v1'},
    operation: {...operationSchema, const: OPERATION}, provider: providerSchema,
    representation_dimensions: dimensionsSchema, source_spec_file: filenameSchema,
    request_file: filenameSchema, result_file: filenameSchema, summary_file: filenameSchema,
    status: {const: 'partial'}, summary: summarySchema,
    pairs: {...nativeResultSchema.properties.pairs, minItems: 5, maxItems: 5},
    artifacts: loadedSchema().properties.artifacts, expected: workflowExpectationsSchema,
    inspection: {type: 'string'}, verified: {const: true},
    limitations: {type: 'array', items: {type: 'string'}, minItems: 3, maxItems: 3},
    workflow_file: filenameSchema}),
};

function selectedProvider(provider) {
  const selected = Object.hasOwn(PROVIDERS, provider) && PROVIDERS[provider];
  require(selected, `Unknown engineering provider ${provider}; use toy-a or toy-b.`);
  return selected;
}

function destination(output) {
  require(typeof output === 'string' && path.isAbsolute(output) && path.normalize(output) === output,
    'Workflow output must be an absolute canonical new directory.');
  require(output.startsWith(`${OUTPUT_ROOT}/`), `Workflow output must be a new directory within ${OUTPUT_ROOT}/.`);
  return output;
}

export function makeWorkflowRequest({provider = 'toy-a'} = {}) {
  const selected = selectedProvider(provider), prefix = `${selected.providerId}-d${selected.dimensions}:workflow-v1`;
  const fixture = makeExampleRequest(selected), input = fixture.input;
  const artifacts = new Map(fixture.artifacts.map(row => [row.id, row]));
  const artifact = (name, value) => {
    const id = `${prefix}:${name}`, bytes = jsonBytes(value), hash = sha256(bytes);
    require(!artifacts.has(id), `duplicate workflow artifact ${id}`);
    artifacts.set(id, {id, sha256: hash, bytes});
    return {id, sha256: hash};
  };
  const template = input.pairs[0], baseline = template.baseline;
  const constructor = artifact('constructor', {identity: 'fixture-explicit-setting-edit-v1',
    changed_attribute: 'setting', declared_alternative_settings: [2, 3, 0, 11, 2],
    held_fixed: {load: 2}, scope: SCOPE,
    execution: 'Inputs and responses are recorded constants; no constructor or model is executed'});
  input.expected_pipeline.dependencies = input.expected_pipeline.dependencies.map(row =>
    row.role === 'constructor' ? {...row, artifact: constructor} : row);
  const pipeline = artifact('pipeline', {identity: 'fixture-recorded-alternative-roster-v1',
    space: input.expected_space.artifact, response_model: input.expected_pipeline.response_model,
    configuration_sha256: input.expected_pipeline.configuration_sha256,
    dependencies: input.expected_pipeline.dependencies, scope: SCOPE});
  input.expected_pipeline.artifact = pipeline;
  baseline.pipeline = pipeline;
  const foreignSpaceArtifact = artifact('incompatible-space', {identity: 'fixture-foreign-space-v1',
    provider: input.expected_space.provider, native_shape: [selected.dimensions],
    geometry: 'euclidean', scope: SCOPE,
    note: 'Same width and provider but a distinct space identity; no cross-space mapping is bound'});
  const incompatibleSpace = {...input.expected_space, artifact: foreignSpaceArtifact};
  const declarations = [
    {name: 'increase-one', change: 1, response: [3.5, -2], state: 'success', delta: [1.5, -1], diagnostic: null},
    {name: 'increase-two', change: 2, response: [5, -3], state: 'success', delta: [3, -2], diagnostic: null},
    {name: 'decrease-one', change: -1, response: [0.5, 0], state: 'success', delta: [-1.5, 1], diagnostic: null},
    {name: 'outside-support', change: 10, response: [17, -11], state: 'abstained', delta: null,
      diagnostic: 'OUTSIDE_DECLARED_SUPPORT'},
    {name: 'incompatible-space', change: 1, response: [3.5, -2], state: 'failed', delta: null,
      diagnostic: 'INCOMPATIBLE_ROLE_OR_SPACE'},
  ];
  const expectedPairs = [];
  input.pairs = declarations.map(row => {
    const pairId = `${prefix}:${row.name}`, setting = 1 + row.change;
    const space = row.state === 'failed' ? incompatibleSpace : input.expected_space;
    const alternativeInput = artifact(`${row.name}-input`, {setting, load: 2, parent_input_id: baseline.input.id,
      constructor, signed_change: row.change, scope: SCOPE});
    const representation = artifact(`${row.name}-representation`, {space_id: space.artifact.id,
      values: Array.from({length: selected.dimensions}, (_, index) => index === 0 ? setting : index + 1),
      scope: SCOPE});
    const response = artifact(`${row.name}-response`, {axes: input.response_axes, values: row.response});
    const mask = artifact(`${row.name}-mask`, {selected_output_position: 0, valid: true});
    const alternative = {...template.alternative, input: alternativeInput, representation, response, mask,
      space, pipeline, response_execution_id: `${pairId}:recorded-response`,
      attributes: [{name: 'setting', units: 'unit', value: setting}, {name: 'load', units: 'unit', value: 2}],
      response_values: row.response};
    const check = (name, state = 'passed') => ({state, evidence: artifact(`${row.name}-${name}`, {
      check: name, state, scope: SCOPE, note: 'Explicit artificial supplied assertion, not independently proved'}),
      detail: `Artificial fixture ${name}: ${state}`});
    const controlEvidence = artifact(`${row.name}-control`, {attribute: 'setting', before: 1, after: setting,
      signed_change: row.change, magnitude: Math.abs(row.change), direction: row.change > 0 ? 'increase' : 'decrease',
      held_fixed: {load: 2}, scope: SCOPE});
    expectedPairs.push({pair_id: pairId, signed_change: row.change, magnitude: Math.abs(row.change),
      direction: row.change > 0 ? 'increase' : 'decrease', state: row.state, diagnostic: row.diagnostic, delta: row.delta});
    return {...template, id: pairId, baseline, alternative, constructor,
      change_definition: `Change setting from 1 to ${setting} (signed change ${row.change}); hold load fixed at 2`,
      control_evidence: controlEvidence, alternative_validity: check('validity'),
      support: check('support', row.state === 'abstained' ? 'rejected' : 'passed'),
      interpretability: check('interpretability')};
  });
  input.request_id = `${prefix}:request`;
  input.operation_run_id = `${prefix}:run`;
  input.purpose = SCOPE;
  input.population_id = `${prefix}:one-shared-source-unit`;
  input.sampling_rule = 'The complete five-pair artificial alternative roster shares one observed baseline; alternatives are not independent samples';
  input.resource_budget = 'Five recorded comparisons; CPU arithmetic only; no model execution, fitting or frozen-study replay';
  const references = collectArtifactRefs(input);
  const exactArtifacts = references.map(ref => {
    const found = artifacts.get(ref.id);
    require(found && found.sha256 === ref.sha256, `workflow omitted artifact ${ref.id}`);
    return found;
  });
  return {input, artifacts: exactArtifacts, provider, representation_dimensions: selected.dimensions,
    expected: {summary: {requested: 5, succeeded: 3, abstained: 1, failed: 1, unresolved: 0, success_rate: 0.6},
      pairs: expectedPairs}};
}

export async function createWorkflowSource({provider = 'toy-a', output}) {
  const fixture = makeWorkflowRequest({provider}), target = destination(output);
  createExclusiveDirectory(target, {root: '/mityane'});
  mkdirSync(path.join(target, 'artifacts'));
  const artifacts = fixture.artifacts.map((row, index) => {
    const relative = `artifacts/${String(index).padStart(5, '0')}.json`;
    writeNew(path.join(target, relative), row.bytes);
    return {id: row.id, sha256: row.sha256, path: relative};
  });
  const spec = {schema_version: 'mityane.operation.prepare/v1', operation: OPERATION, input: fixture.input, artifacts};
  const specFile = path.join(target, 'prepare.json');
  writeNew(specFile, jsonBytes(spec));
  writeNew(path.join(target, 'expected.json'), jsonBytes(fixture.expected));
  const report = {schema_version: 'mityane.operation.workflow-source/v1', operation: OPERATION,
    provider, representation_dimensions: fixture.representation_dimensions, spec_file: specFile,
    expected: fixture.expected};
  validateSchema(workflowSourceSchema, report);
  return report;
}

function verifyExpected(record, expected) {
  require(record.status === 'partial' && record.result.admitted === true, 'workflow expected admitted partial execution');
  require(Object.entries(expected.summary).every(([name, value]) => record.summary[name] === value),
    'workflow counts differ from the complete artificial roster');
  require(record.result.pairs.length === expected.pairs.length, 'workflow truncated the alternative roster');
  for (const [index, row] of record.result.pairs.entries()) {
    const target = expected.pairs[index];
    require(row.pair_index === index && row.pair_id === target.pair_id && row.state === target.state,
      `workflow pair outcome differs: ${target.pair_id}`);
    require(target.delta === null ? row.contrast === null :
      row.contrast !== null && row.contrast.delta.length === target.delta.length &&
      row.contrast.delta.every((value, axis) => value === target.delta[axis]),
    `workflow contrast differs: ${target.pair_id}`);
    require(target.diagnostic === null ? row.diagnostics.length === 0 :
      row.diagnostics.some(diagnostic => diagnostic.code === target.diagnostic),
    `workflow typed refusal differs: ${target.pair_id}`);
  }
}

export async function runWorkflow({operations, provider = 'toy-a', output}) {
  for (const name of ['prepare', 'check', 'run', 'inspect', 'read'])
    require(typeof operations?.[name] === 'function', `workflow needs operations.${name}`);
  selectedProvider(provider);
  const target = destination(output);
  createExclusiveDirectory(target, {root: '/mityane'});
  const source = await createWorkflowSource({provider, output: path.join(target, 'source')});
  const prepared = await operations.prepare(source.spec_file, path.join(target, 'prepared'));
  const checked = await operations.check(prepared.request_file);
  verifyExpected(checked, source.expected);
  const run = await operations.run(prepared.request_file, path.join(target, 'run'));
  const inspection = await operations.inspect(run.result_file);
  const loaded = await operations.read(run.result_file);
  verifyExpected(loaded.record, source.expected);
  const reportFile = path.join(target, 'workflow.json');
  const report = {schema_version: 'mityane.operation.workflow/v1', operation: OPERATION, provider,
    representation_dimensions: source.representation_dimensions, source_spec_file: source.spec_file,
    request_file: prepared.request_file, result_file: run.result_file, summary_file: run.summary_file,
    status: loaded.record.status, summary: loaded.record.summary, pairs: loaded.record.result.pairs,
    artifacts: loaded.artifacts, expected: source.expected, inspection, verified: true,
    limitations: ['Artificial recorded alternatives; no scientific claim, encoder execution or fitting.',
      'All alternatives share one baseline; pairs are not independent samples.',
      'Contrasts are signed differences, not derivatives, uncertainty estimates or causal effects.'],
    workflow_file: reportFile};
  validateSchema(workflowResultSchema, report);
  writeNew(reportFile, jsonBytes(report));
  return report;
}
