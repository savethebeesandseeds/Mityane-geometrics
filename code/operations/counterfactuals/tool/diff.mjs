// SPDX-License-Identifier: MIT
// Compare independently verified retained results. No encoder, readout, fitting
// or native operation is executed here, and missing contrasts remain missing.
import path from 'node:path';
import fs from 'node:fs';
import {isDeepStrictEqual} from 'node:util';
import {admitFiles, readAdmitted, sha256, jsonBytes, parseJSON, writeNew,
  createExclusiveDirectory, requireCondition as require}
  from '../../../shared/artifacts/node/v1/files.mjs';

const object = properties => ({type: 'object', additionalProperties: false,
  required: Object.keys(properties), properties});
const text = {type: 'string'}, count = {type: 'integer', minimum: 0};
const nullable = schema => ({anyOf: [{type: 'null'}, schema]});
const vector = {type: 'array', items: {type: 'number'}};
const states = ['success', 'abstained', 'failed', 'unresolved'];
const diagnostic = object({code: text, record_id: text, message: text});
const outcome = object({state: {type: 'string', enum: states},
  diagnostics: {type: 'array', items: diagnostic}});
const pin = object({path: text, bytes: count, sha256: {type: 'string', pattern: '^[a-f0-9]{64}$'}});
const pipeline = object({result_sha256: text, request_sha256: text, manifest_sha256: text,
  retained_result: text, status: {type: 'string', enum: ['completed', 'partial']},
  operation_run_id: text, space: {type: 'object'}, pipeline: {type: 'object'},
  code_sha256: text, configuration_sha256: text});
const axis = object({id: text, definition_revision: text, units: text});
const pair = object({pair_index: count, pair_id: text, left: outcome, right: outcome,
  reason_codes_agree: {type: 'boolean'}, states_agree: {type: 'boolean'},
  left_delta: nullable(vector), right_delta: nullable(vector),
  right_minus_left: nullable(vector), absolute_difference: nullable(vector),
  within_tolerance: nullable({type: 'boolean'}),
  axis_within_tolerance: nullable({type: 'array', items: {type: 'boolean'}})});

export const diffResultSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:diff:1',
  title: 'Verified OR06-PCR experimental-roster comparison v1',
  description: 'Right-minus-left response contrasts for success/success only. '
    + 'Pipeline disagreement is not an accuracy or scientific-quality verdict.',
  ...object({schema_version: {const: 'mityane.operation.diff/v1'},
    operation: object({id: {const: 'OR06-PCR'}, revision: {const: '0.1.0'}}),
    claim_level: {const: 'controlled_model_response'},
    status: {type: 'string', enum: ['completed', 'partial']},
    roster: object({schema_version: {const: 'mityane.operation.experimental-roster/v1'},
      sha256: {type: 'string', pattern: '^[a-f0-9]{64}$'}, pairs: count}),
    response_axes: {type: 'array', items: axis},
    tolerance: object({absolute: {type: 'number', minimum: 0}, relative: {type: 'number', minimum: 0},
      rule: {const: 'abs(right-left) <= absolute + relative * max(abs(left), abs(right))'}}),
    sources: object({left: pipeline, right: pipeline}),
    summary: object({requested: count, comparable: count, agreed: count, diverged: count,
      missing_contrasts: count, same_state: count, same_reason_codes: count,
      state_cross: {type: 'object', additionalProperties: count}}),
    pairs: {type: 'array', items: pair}, limitations: {type: 'array', items: text}}),
};
export const diffFilesSchema = {$schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:diff-files:1',
  ...object({schema_version: {const: 'mityane.operation.diff-files/v1'}, files: {type: 'array', items: pin}})};
export const diffOutputSchema = {$schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:diff-output:1',
  title: 'OR06-PCR retained comparison CLI and Node API output v1',
  ...object({schema_version: {const: 'mityane.operation.diff-output/v1'},
    operation: diffResultSchema.properties.operation, status: diffResultSchema.properties.status,
    comparison_file: text, summary_file: text, manifest_file: text,
    roster_sha256: {type: 'string', pattern: '^[a-f0-9]{64}$'},
    summary: diffResultSchema.properties.summary, comparison: diffResultSchema})};
export const diffOptionsSchema = {$schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:diff-options:1', type: 'object', additionalProperties: false,
  required: ['left', 'right', 'output'], properties: {left: text, right: text, output: text,
    absoluteTolerance: {type: 'number', minimum: 0, default: 0},
    relativeTolerance: {type: 'number', minimum: 0, default: 0}}};

// Exact raw records and edit/control declarations identify the experiment.
// Representation, model, response values, execution IDs and diagnostic outcomes
// identify the pipelines being compared; including them would prevent a useful
// comparison. Mask contents remain bound even when provider-local IDs differ.
export function experimentalRoster(input) {
  const branch = row => row === null ? null : ({input: row.input,
    source_observation: row.source_observation, parent_input_id: row.parent_input_id,
    origin: row.origin, timing: row.timing, attributes: row.attributes,
    output_position: row.output_position, mask_sha256: row.mask.sha256, axes: row.axes});
  return {schema_version: 'mityane.operation.experimental-roster/v1',
    contract_id: input.contract_id, contract_revision: input.contract_revision,
    rer_id: input.rer_id, rer_revision: input.rer_revision, claim_level: input.claim_level,
    population_id: input.population_id, sampling_rule: input.sampling_rule,
    response_horizon: input.response_horizon, permitted_interpretation: input.permitted_interpretation,
    temporal_mode: input.temporal_mode, as_of: input.as_of,
    non_temporal_justification: input.non_temporal_justification,
    response_axes: input.response_axes, selected_output_position: input.selected_output_position,
    attribute_rules: input.attribute_rules, control_tolerance: input.control_tolerance,
    support_policies: input.expected_pipeline.dependencies.filter(row => row.role === 'support_policy'),
    pairs: input.pairs.map(row => ({id: row.id, unit_id: row.unit_id,
      entity_id: row.entity_id, group_id: row.group_id,
      baseline: branch(row.baseline), alternative: branch(row.alternative),
      constructor: row.constructor, change_definition: row.change_definition,
      interpretation: row.interpretation, control_evidence: row.control_evidence}))};
}
export function experimentalRosterFingerprint(input) { return sha256(jsonBytes(experimentalRoster(input))); }

function sourceBundle(loaded) {
  const directory = path.dirname(loaded.result_file);
  const entries = [...loaded.manifest.files, {path: 'manifest.json'}];
  const rows = admitFiles(entries.map(entry => path.join(directory, entry.path)), {root: directory});
  const pins = rows.map((row, index) => {
    const bytes = readAdmitted(row), expected = entries[index], digest = sha256(bytes);
    if (expected.path === 'manifest.json') require(isDeepStrictEqual(parseJSON(bytes,
      {duplicates: 'reject', integers: 'number', allowBOM: false}), loaded.manifest),
    'source manifest changed after verification');
    else require(row.bytes === expected.bytes && digest === expected.sha256,
      `source bundle changed after verification: ${expected.path}`);
    return {path: expected.path, bytes: row.bytes, sha256: digest};
  });
  return {rows, pins};
}
const reasonCodes = row => [...new Set(row.diagnostics.map(item => item.code))].sort();
function provenance(loaded, source, side) {
  const input = loaded.request.input;
  const digest = name => source.pins.find(row => row.path === name).sha256;
  return {result_sha256: digest('result.json'), request_sha256: digest('request.json'),
    manifest_sha256: digest('manifest.json'), retained_result: `${side}/result.json`,
    status: loaded.record.status, operation_run_id: input.operation_run_id,
    space: input.expected_space, pipeline: input.expected_pipeline,
    code_sha256: input.code_sha256, configuration_sha256: input.configuration_sha256};
}
function toleranceNumber(value, label) {
  require(typeof value === 'number' && Number.isFinite(value) && value >= 0,
    `${label} must be a finite nonnegative number`);
  return value;
}

export async function runDiff({left, right, output, operations, absoluteTolerance = 0, relativeTolerance = 0}) {
  const absolute = toleranceNumber(absoluteTolerance, 'absoluteTolerance');
  const relative = toleranceNumber(relativeTolerance, 'relativeTolerance');
  require(typeof left === 'string' && typeof right === 'string', 'diff requires two retained result.json files');
  const first = operations.read(left), second = operations.read(right);
  for (const loaded of [first, second]) {
    require(loaded.record.operation.id === 'OR06-PCR' && loaded.record.operation.revision === '0.1.0',
      'diff requires OR06-PCR 0.1.0 results');
    if (loaded.record.claim_level !== 'controlled_model_response' || !loaded.record.result.admitted ||
        loaded.record.status === 'rejected') throw Object.assign(new Error(
      'diff requires admitted controlled_model_response results with an executed pair roster'), {exitCode: 3});
  }
  require(isDeepStrictEqual(first.record.operation, second.record.operation), 'diff operation revisions differ');
  const fingerprint = experimentalRosterFingerprint(first.request.input);
  require(fingerprint === experimentalRosterFingerprint(second.request.input),
    'diff experimental roster mismatch: raw inputs, ordered pairs, changes, masks, support, population, timing or response semantics differ');
  // Admit and hash both entire retained source matrices before publication.
  const leftSource = sourceBundle(first), rightSource = sourceBundle(second);
  const stateCross = Object.fromEntries(states.flatMap(a => states.map(b => [`${a}/${b}`, 0])));
  const summary = {requested: first.record.result.pairs.length, comparable: 0, agreed: 0,
    diverged: 0, missing_contrasts: 0, same_state: 0, same_reason_codes: 0, state_cross: stateCross};
  const pairs = first.record.result.pairs.map((a, index) => {
    const b = second.record.result.pairs[index];
    require(a.pair_id === b.pair_id && a.pair_index === b.pair_index, 'diff pair alignment mismatch');
    const both = a.state === 'success' && b.state === 'success';
    const leftDelta = both ? a.contrast.delta : null, rightDelta = both ? b.contrast.delta : null;
    const delta = both ? leftDelta.map((value, axis) => {
      const difference = rightDelta[axis] - value;
      require(Number.isFinite(difference), 'diff contrast subtraction is nonfinite');
      return difference;
    }) : null;
    const difference = delta?.map(Math.abs) ?? null;
    const tolerances = both ? leftDelta.map((value, axis) => {
      const bound = absolute + relative * Math.max(Math.abs(value), Math.abs(rightDelta[axis]));
      require(Number.isFinite(bound), 'diff tolerance arithmetic is nonfinite');
      return difference[axis] <= bound;
    }) : null;
    const agree = tolerances?.every(Boolean) ?? null;
    const statesAgree = a.state === b.state, reasonsAgree = isDeepStrictEqual(reasonCodes(a), reasonCodes(b));
    summary.state_cross[`${a.state}/${b.state}`]++;
    summary.same_state += Number(statesAgree); summary.same_reason_codes += Number(reasonsAgree);
    if (both) { summary.comparable++; summary[agree ? 'agreed' : 'diverged']++; }
    else summary.missing_contrasts++;
    return {pair_index: index, pair_id: a.pair_id,
      left: {state: a.state, diagnostics: a.diagnostics}, right: {state: b.state, diagnostics: b.diagnostics},
      reason_codes_agree: reasonsAgree, states_agree: statesAgree,
      left_delta: leftDelta, right_delta: rightDelta, right_minus_left: delta,
      absolute_difference: difference, within_tolerance: agree, axis_within_tolerance: tolerances};
  });
  const status = [first.record, second.record].some(row => row.status === 'partial') ? 'partial' : 'completed';
  const record = {schema_version: 'mityane.operation.diff/v1', operation: first.record.operation,
    claim_level: 'controlled_model_response', status,
    roster: {schema_version: 'mityane.operation.experimental-roster/v1', sha256: fingerprint, pairs: summary.requested},
    response_axes: first.request.input.response_axes,
    tolerance: {absolute, relative, rule: 'abs(right-left) <= absolute + relative * max(abs(left), abs(right))'},
    sources: {left: provenance(first, leftSource, 'left'), right: provenance(second, rightSource, 'right')},
    summary, pairs, limitations: [
      'This retained-record comparison executes no encoder, readout, fitting or native operation.',
      'The fingerprint binds the declared experiment, raw input lineage and mask contents; it does not authenticate semantic attestations.',
      'Only success/success pairs receive numerical comparisons; other states and diagnostics remain explicit without imputation.',
      'Pipeline agreement or divergence does not establish accuracy, usefulness, encoder contribution or a causal effect.',
      'All compared records retain their original exposure status; this is not a new scientific assessment.',
    ]};
  // Resolve arithmetic and the shared artifact encoding before creating an
  // output directory, so invalid numeric output cannot leave a published stub.
  const comparisonBytes = jsonBytes(record), rosterBytes = jsonBytes(experimentalRoster(first.request.input));
  const text = `OR06-PCR retained-record comparison: ${status}\nExperimental roster: ${fingerprint}\n`
    + `Pairs: ${summary.requested}; comparable: ${summary.comparable}; agree: ${summary.agreed}; diverge: ${summary.diverged}; missing contrasts: ${summary.missing_contrasts}\n`
    + `State alignment: ${summary.same_state}/${summary.requested}; reason-code alignment: ${summary.same_reason_codes}/${summary.requested}\n`
    + 'Differences are right minus left. Agreement is not a model-quality verdict.\n';
  createExclusiveDirectory(output);
  const files = [];
  for (const [side, source] of [['left', leftSource], ['right', rightSource]]) {
    createExclusiveDirectory(path.join(output, side));
    for (const [index, admitted] of source.rows.entries()) {
      const pin = source.pins[index], bytes = readAdmitted(admitted);
      require(sha256(bytes) === pin.sha256, `source changed during diff publication: ${side}/${pin.path}`);
      const target = path.join(output, side, pin.path);
      fs.mkdirSync(path.dirname(target), {recursive: true});
      writeNew(target, bytes); files.push({...pin, path: `${side}/${pin.path}`});
    }
  }
  const publish = (name, bytes) => {
    writeNew(path.join(output, name), bytes); files.push({path: name, bytes: bytes.length, sha256: sha256(bytes)});
  };
  publish('comparison.json', comparisonBytes);
  publish('roster.json', rosterBytes);
  publish('summary.txt', Buffer.from(text));
  for (const source of [leftSource, rightSource]) source.rows.forEach((row, index) =>
    require(sha256(readAdmitted(row)) === source.pins[index].sha256, 'source changed during diff publication'));
  writeNew(path.join(output, 'manifest.json'), jsonBytes({schema_version: 'mityane.operation.diff-files/v1', files}));
  return {schema_version: 'mityane.operation.diff-output/v1', operation: record.operation, status,
    comparison_file: path.join(output, 'comparison.json'), summary_file: path.join(output, 'summary.txt'),
    manifest_file: path.join(output, 'manifest.json'), roster_sha256: fingerprint, summary, comparison: record};
}
