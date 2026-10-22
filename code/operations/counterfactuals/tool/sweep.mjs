// SPDX-License-Identifier: MIT
// Scenario planning and packaging of already recorded alternatives. This module
// never encodes inputs, fits a readout, executes a model or invents native states.
import fs from 'node:fs';
import path from 'node:path';
import {isDeepStrictEqual} from 'node:util';
import {ROOT} from '../../../shared/representation/tools/v1/runtime.mjs';
import {validateSchema} from '../../../shared/representation/tools/v1/schemas.mjs';
import {collectArtifactRefs, MAX_PAIRS} from './request.mjs';
import {admitFiles, readAdmitted, parseJSON, jsonBytes, sha256, writeNew,
  createExclusiveDirectory, safeRelative, requireCondition as require}
  from '../../../shared/artifacts/node/v1/files.mjs';

const OPERATION = Object.freeze({id: 'OR06-PCR', revision: '0.1.0'});
const MAX_BYTES = 64 * 1024 * 1024, MAX_ARTIFACT_BYTES = 256 * 1024 * 1024;
const object = properties => ({type: 'object', additionalProperties: false,
  required: Object.keys(properties), properties});
const text = {type: 'string', minLength: 1};
const axisCommon = {name: text, operation: {enum: ['add', 'set', 'scale']}};
export const sweepSpaceSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:sweep-space:1',
  title: 'OR06-PCR Cartesian scenario-space specification v1',
  ...object({schema_version: {const: 'mityane.operation.sweep-space/v1'}, baseline_pair_id: text,
    request_id: text, operation_run_id: text, sampling_rule: text,
    axes: {type: 'array', minItems: 1, maxItems: 64, items: {anyOf: [
      object({...axisCommon, values: {type: 'array', minItems: 1, maxItems: MAX_PAIRS, items: {type: 'number'}}}),
      object({...axisCommon, start: {type: 'number'}, end: {type: 'number'}, step: {type: 'number'}}),
    ]}}, hold: {type: 'array', maxItems: 10000, items: text}}),
};
export const sweepCatalogSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:sweep-catalog:1',
  title: 'OR06-PCR complete precomputed alternative catalog v1',
  ...object({schema_version: {const: 'mityane.operation.sweep-catalog/v1'},
    entries: {type: 'array', minItems: 1, maxItems: MAX_PAIRS, items: object({
      coordinate: {type: 'array', minItems: 1, maxItems: 64, items: {type: 'number'}},
      request: text, pair_id: text})}}),
};
export const sweepResultSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:sweep:1',
  title: 'OR06-PCR scenario plan or recorded request packaging result v1',
  ...object({schema_version: {const: 'mityane.operation.sweep/v1'},
    operation: {const: OPERATION}, status: {enum: ['completed', 'partial', 'rejected']},
    mode: {enum: ['plan', 'recorded']}, materialization_required: {type: 'boolean'},
    plan_file: text, request_file: {anyOf: [{type: 'null'}, text]},
    cells: {type: 'integer', minimum: 1, maximum: MAX_PAIRS},
    blocked_support: {type: 'integer', minimum: 0, maximum: MAX_PAIRS},
    check: {anyOf: [{type: 'null'}, {type: 'object'}]},
    note: text}),
};

// Decimal operations use the spelling of the JSON number, not repeated binary
// addition. A range is inclusive and must reach its declared endpoint exactly.
function decimal(number) {
  require(typeof number === 'number' && Number.isFinite(number), 'finite sweep value required');
  const [mantissa, exponent = '0'] = String(number).toLowerCase().split('e');
  const fraction = (mantissa.split('.')[1] ?? '').length;
  return {n: BigInt(mantissa.replace('.', '')), scale: fraction - Number(exponent)};
}
function align(values) {
  const scale = Math.max(...values.map(value => value.scale));
  return {scale, values: values.map(value => value.n * 10n ** BigInt(scale - value.scale))};
}
function numberOf(n, scale) {
  const result = Number(`${n}e${-scale}`);
  require(Number.isFinite(result) && (n === 0n || result !== 0), 'sweep arithmetic overflow or underflow');
  return Object.is(result, -0) ? 0 : result;
}
function axisValues(axis) {
  let values;
  if (Object.hasOwn(axis, 'values')) values = axis.values.map(value => numberOf(decimal(value).n, decimal(value).scale));
  else {
    const aligned = align([axis.start, axis.end, axis.step].map(decimal));
    const [start, end, step] = aligned.values, distance = end - start;
    require(step !== 0n, 'sweep range step must be nonzero');
    require(distance === 0n || (distance > 0n) === (step > 0n), 'sweep range step has wrong direction');
    require(distance % step === 0n, 'sweep range must reach its inclusive end exactly');
    const count = distance / step + 1n;
    require(count >= 1n && count <= BigInt(MAX_PAIRS), 'sweep axis exceeds 10000 values');
    values = Array.from({length: Number(count)}, (_, index) => numberOf(start + BigInt(index) * step, aligned.scale));
  }
  require(new Set(values).size === values.length, `duplicate or numerically collapsed sweep values: ${axis.name}`);
  return values;
}
function changedValue(before, operation, value) {
  if (operation === 'set') return value;
  const a = decimal(before), b = decimal(value);
  if (operation === 'scale') return numberOf(a.n * b.n, a.scale + b.scale);
  const aligned = align([a, b]);
  return numberOf(aligned.values[0] + aligned.values[1], aligned.scale);
}
function same(left, right, message) { require(isDeepStrictEqual(left, right), message); }
function coordinateKey(values) { return JSON.stringify(values.map(value => Object.is(value, -0) ? 0 : value)); }
function admission(condition, message) {
  if (!condition) { const error = new Error(message); error.exitCode = 3; throw error; }
}

export async function runSweep({operations, request, space, catalog, output}) {
  for (const name of ['check', 'prepare']) require(typeof operations?.[name] === 'function', `sweep needs operations.${name}`);
  require(typeof output === 'string' && path.isAbsolute(output) && path.normalize(output) === output &&
    output.startsWith(`${path.join(ROOT, '.temp/operations/counterfactuals')}${path.sep}`),
  'sweep output must be an absolute new directory under .temp/operations/counterfactuals/');
  const files = new Map(), sourceRequests = new Map(), artifactMap = new Map();
  let inputBytes = 0, artifactBytes = 0;
  const load = (filename, artifact = false) => {
    const absolute = path.resolve(ROOT, filename);
    if (files.has(absolute)) return files.get(absolute);
    const admitted = admitFiles([absolute])[0];
    require(admitted.bytes <= (artifact ? MAX_ARTIFACT_BYTES : MAX_BYTES), 'sweep source exceeds byte limit');
    const bytes = readAdmitted(admitted), row = {absolute, admitted, bytes, sha256: sha256(bytes)};
    if (artifact) { artifactBytes += bytes.length; require(artifactBytes <= MAX_ARTIFACT_BYTES, 'sweep artifacts exceed 256 MiB'); }
    else { inputBytes += bytes.length; require(inputBytes <= MAX_BYTES, 'sweep source JSON exceeds 64 MiB'); }
    files.set(absolute, row); return row;
  };
  const json = row => parseJSON(row.bytes, {duplicates: 'reject', integers: 'number', allowBOM: false});
  const spaceRow = load(space), specification = json(spaceRow);
  validateSchema(sweepSpaceSchema, specification, {label: 'sweep space'});
  const sourceRequest = async filename => {
    const absolute = path.resolve(ROOT, filename);
    if (sourceRequests.has(absolute)) return sourceRequests.get(absolute);
    const row = load(absolute), envelope = json(row);
    require(envelope.schema_version === 'mityane.operation.request/v1', 'sweep source must be a request envelope');
    same(envelope.operation, OPERATION, 'sweep supports only OR06-PCR 0.1.0');
    admission(envelope.input?.claim_level === 'controlled_model_response', 'sweep admits only controlled_model_response');
    const checked = await operations.check(absolute);
    admission(checked.result.admitted, 'sweep source request was rejected by current admission');
    for (const artifact of envelope.artifacts) {
      const file = load(path.join(path.dirname(absolute), safeRelative(artifact.path)), true);
      require(file.sha256 === artifact.sha256, `sweep artifact changed after validation: ${artifact.id}`);
      const previous = artifactMap.get(artifact.id);
      require(!previous || previous.sha256 === artifact.sha256, `conflicting sweep artifact identity: ${artifact.id}`);
      artifactMap.set(artifact.id, {...file, id: artifact.id});
    }
    const result = {row, envelope, checked}; sourceRequests.set(absolute, result); return result;
  };
  const base = await sourceRequest(request), input = base.envelope.input;
  const template = input.pairs.find(pair => pair.id === specification.baseline_pair_id);
  require(template?.baseline, 'sweep baseline_pair_id must select an existing non-null baseline');
  require(input.pairs.filter(pair => pair.id === specification.baseline_pair_id).length === 1,
    'sweep baseline pair identity must be unique');
  require(specification.request_id.trim() && specification.operation_run_id.trim() && specification.sampling_rule.trim(),
    'sweep requires explicit nonempty request, run and sampling identities');
  require(specification.request_id !== input.request_id && specification.operation_run_id !== input.operation_run_id,
    'sweep request and run identities must differ from the base request');
  const rules = new Map(input.attribute_rules.map(rule => [rule.name, rule]));
  const before = new Map(template.baseline.attributes.map(attribute => [attribute.name, attribute]));
  require(rules.size === input.attribute_rules.length && before.size === rules.size &&
    [...rules.keys()].every(name => before.has(name)), 'sweep baseline must cover each scalar rule exactly');
  const varied = new Set(), held = new Set(specification.hold);
  require(held.size === specification.hold.length, 'duplicate sweep hold-list entry');
  const axes = specification.axes.map(axis => {
    require(rules.has(axis.name) && !varied.has(axis.name) && !held.has(axis.name),
      `unknown, repeated or held sweep axis: ${axis.name}`);
    require(rules.get(axis.name).role === 'changed', `sweep cannot vary held-fixed or coupled rule: ${axis.name}`);
    varied.add(axis.name); return {...axis, values: axisValues(axis)};
  });
  require([...held].every(name => rules.has(name)) && [...rules.keys()].every(name => varied.has(name) || held.has(name)),
    'sweep axes and explicit hold-list must partition the complete scalar manifest');
  let count = 1;
  for (const axis of axes) { count *= axis.values.length; require(count <= MAX_PAIRS, 'Cartesian sweep exceeds 10000 cells'); }
  // Bound the expanded scalar matrix before constructing it. The native pair
  // limit alone permits a pathological 10000 x 10000 attribute allocation.
  const expandedAttributes = count * template.baseline.attributes.length;
  require(expandedAttributes <= 1000000, 'sweep expanded scalar matrix exceeds 1000000 values');
  const attributeBytes = jsonBytes(template.baseline.attributes).length;
  require(attributeBytes * count <= MAX_BYTES / 2, 'sweep expanded scalar manifest exceeds bounded plan capacity');
  let coordinates = [[]];
  for (const axis of axes) coordinates = coordinates.flatMap(prefix => axis.values.map(value => [...prefix, value]));
  const cells = coordinates.map((coordinate, index) => {
    const attributes = template.baseline.attributes.map(attribute => ({...attribute}));
    for (const [axisIndex, axis] of axes.entries()) {
      const attribute = attributes.find(item => item.name === axis.name);
      attribute.value = changedValue(attribute.value, axis.operation, coordinate[axisIndex]);
    }
    const violations = [['baseline', template.baseline.attributes], ['alternative', attributes]].flatMap(([branch, values]) =>
      values.filter(attribute => attribute.value < rules.get(attribute.name).minimum ||
        attribute.value > rules.get(attribute.name).maximum).map(attribute => ({branch, name: attribute.name,
        value: attribute.value, minimum: rules.get(attribute.name).minimum, maximum: rules.get(attribute.name).maximum})));
    return {index, coordinate, attributes, planning_status: violations.length ? 'blocked_support' : 'ready_for_materialization',
      support_violations: violations, recorded_pair_id: null};
  });
  // Distinct operands can map to the same actual scenario (e.g. scaling zero).
  // Reject rather than pretending duplicated scenarios are independent cells.
  require(new Set(cells.map(cell => JSON.stringify(cell.attributes))).size === cells.length,
    'sweep operands collapse to duplicate actual alternatives');
  let catalogRow = null, selectedPairs = [];
  if (catalog !== undefined) {
    catalogRow = load(catalog);
    const supplied = json(catalogRow);
    validateSchema(sweepCatalogSchema, supplied, {label: 'sweep catalog'});
    require(supplied.entries.length === cells.length, 'catalog must cover every Cartesian cell; missing records require materialization');
    const entries = new Map();
    for (const entry of supplied.entries) {
      require(entry.coordinate.length === axes.length, 'catalog coordinate width differs from sweep axes');
      const key = coordinateKey(entry.coordinate);
      require(!entries.has(key), 'duplicate catalog coordinate'); entries.set(key, entry);
    }
    const pairIds = new Set();
    for (const cell of cells) {
      const entry = entries.get(coordinateKey(cell.coordinate));
      require(entry, 'catalog omits a Cartesian coordinate; missing records require materialization');
      const source = await sourceRequest(path.join(path.dirname(catalogRow.absolute), safeRelative(entry.request)));
      const candidate = source.envelope.input;
      for (const name of Object.keys(input)) if (!['request_id', 'operation_run_id', 'sampling_rule', 'pairs'].includes(name))
        same(candidate[name], input[name], `catalog changed the base request binding: ${name}`);
      const matches = candidate.pairs.filter(pair => pair.id === entry.pair_id);
      require(matches.length === 1 && matches[0].alternative, 'catalog must select one complete recorded pair');
      const pair = matches[0], alternative = pair.alternative;
      require(!pairIds.has(pair.id), 'catalog pair IDs must be unique across the Cartesian roster'); pairIds.add(pair.id);
      same(pair.baseline, template.baseline, 'catalog changed the original baseline');
      for (const name of ['unit_id', 'entity_id', 'group_id', 'constructor', 'interpretation'])
        same(pair[name], template[name], `catalog changed baseline scope or constructor: ${name}`);
      same(alternative.attributes, cell.attributes, 'catalog attributes do not match the planned change/hold mask');
      same(alternative.pipeline, input.expected_pipeline.artifact, 'catalog alternative changed the frozen pipeline pin');
      same(alternative.space, input.expected_space, 'catalog alternative changed the frozen representation space');
      same(alternative.timing, template.alternative?.timing ?? template.baseline.timing, 'catalog changed alternative timing');
      same(alternative.axes, input.response_axes, 'catalog changed response axes');
      require(alternative.output_position === input.selected_output_position, 'catalog changed selected output position');
      const native = source.checked.result.pairs.find(row => row.pair_id === pair.id);
      require(native, 'catalog pair has no current validated comparison');
      if (cell.support_violations.length) require(native.state === 'abstained' &&
        native.diagnostics.some(row => row.code === 'OUTSIDE_DECLARED_SUPPORT'),
      'outside-support catalog cell must retain an actual native support abstention; no placeholder responses may be invented');
      selectedPairs.push(pair); cell.recorded_pair_id = pair.id;
    }
  }
  // Re-admit all original bytes before publishing any output. No source file is
  // rewritten and a failed publication is retained for inspection.
  for (const row of files.values()) require(sha256(readAdmitted(row.admitted)) === row.sha256, 'sweep source changed before publication');
  const plan = {schema_version: 'mityane.operation.sweep-plan/v1', operation: OPERATION,
    scope: 'Scenario planning only; attributes are proposed inputs, not encoded or executed model records',
    base_request_sha256: base.row.sha256, baseline_pair_id: template.id,
    baseline_input: template.baseline.input, claim_level: input.claim_level,
    pipeline: input.expected_pipeline, space: input.expected_space, attribute_rules: input.attribute_rules,
    response_axes: input.response_axes, temporal_mode: input.temporal_mode, as_of: input.as_of,
    specification, cells, materialization_required: !catalogRow,
    notes: ['Planning statuses are not native pair states. Unsupported cells remain in the complete roster.',
      'Materialization requires real recorded input, representation, response, lineage and evidence artifacts.',
      'Recorded support abstentions, failures and unresolved outcomes retain current native semantics.']};
  require(jsonBytes(plan).length <= MAX_BYTES, 'sweep plan exceeds 64 MiB');
  createExclusiveDirectory(output, {root: ROOT});
  try {
    fs.mkdirSync(path.join(output, 'sources'));
    const pins = [], retainedPaths = new Map();
    for (const [index, row] of [...files.values()].entries()) {
      const retained = `sources/${String(index).padStart(5, '0')}.bin`;
      writeNew(path.join(output, retained), row.bytes);
      retainedPaths.set(row.absolute, retained);
      pins.push({source_path: row.absolute, retained_path: retained, sha256: row.sha256, bytes: row.bytes.length});
    }
    writeNew(path.join(output, 'source-pins.json'), jsonBytes({schema_version: 'mityane.operation.sweep-sources/v1', files: pins}));
    writeNew(path.join(output, 'base-request.json'), base.row.bytes);
    writeNew(path.join(output, 'space.json'), spaceRow.bytes);
    if (catalogRow) writeNew(path.join(output, 'catalog.json'), catalogRow.bytes);
    const planFile = path.join(output, 'plan.json');
    writeNew(planFile, jsonBytes(plan));
    let prepared = null, checked = null;
    if (catalogRow) {
      const packaged = {...input, request_id: specification.request_id, operation_run_id: specification.operation_run_id,
        sampling_rule: specification.sampling_rule, pairs: selectedPairs};
      const inventory = collectArtifactRefs(packaged).map(reference => {
        const row = artifactMap.get(reference.id);
        require(row?.sha256 === reference.sha256, `missing recorded sweep artifact: ${reference.id}`);
        return {...reference, path: retainedPaths.get(row.absolute)};
      });
      const specFile = path.join(output, 'prepare.json');
      writeNew(specFile, jsonBytes({schema_version: 'mityane.operation.prepare/v1', operation: OPERATION,
        input: packaged, artifacts: inventory}));
      prepared = await operations.prepare(specFile, path.join(output, 'prepared'));
      checked = await operations.check(prepared.request_file);
      require(checked.result.admitted && checked.summary.requested === cells.length,
        'generated recorded sweep did not preserve full native admission and roster');
    }
    const report = {schema_version: 'mityane.operation.sweep/v1', operation: OPERATION,
      status: checked?.status ?? 'completed', mode: catalogRow ? 'recorded' : 'plan',
      materialization_required: !catalogRow, plan_file: planFile, request_file: prepared?.request_file ?? null,
      cells: cells.length, blocked_support: cells.filter(cell => cell.support_violations.length).length,
      check: checked,
      note: catalogRow ? 'Packaged already recorded alternatives; use request_file with check/run. No encoding or fitting occurred.' :
        'Complete scenario plan only. Supply real recorded alternatives with a complete catalog before check/run.'};
    validateSchema(sweepResultSchema, report); writeNew(path.join(output, 'sweep.json'), jsonBytes(report));
    return report;
  } catch (error) {
    try { writeNew(path.join(output, 'failure.json'), jsonBytes({status: 'publication_failed', message: error.message})); } catch {}
    throw error;
  }
}
