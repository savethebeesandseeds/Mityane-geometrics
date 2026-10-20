// SPDX-License-Identifier: MIT
// Structural JSON boundary for the recorded-response tool. Admission semantics
// remain in the unchanged C++ operation. Private wire fields follow the public
// C++ structs, in their declaration order, after one version token.
export const MAX_REQUEST_BYTES = 64 * 1024 * 1024;
export const MAX_PAIRS = 10000;
const MAX_STRING_BYTES = 65536;
const SAFE = Number.MAX_SAFE_INTEGER;
const string = {kind: 'string'};
const number = {kind: 'number'};
const integer = {kind: 'integer', minimum: -SAFE, maximum: SAFE};
const size = {kind: 'integer', minimum: 0, maximum: SAFE};
const boolean = {kind: 'boolean'};
const enumeration = (...values) => ({kind: 'enum', values});
const array = (item, maximum = 10000) => ({kind: 'array', item, maximum});
const optional = item => ({kind: 'optional', item});
const object = (fields, artifact = false) => ({kind: 'object', fields, artifact});

const artifact = object({id: string, sha256: string}, true);
const dependency = object({
  role: string, artifact, fit_population: string, selection_population: string,
  pretraining_exposure: string, available_at: optional(integer),
});
const space = object({
  artifact, provider: artifact, preprocessing: artifact, fitted_transform: artifact,
  output_selector: string, native_shape: array(size, 64), layout: string,
  normalization: string, geometry: string, input_role: string,
});
const axis = object({id: string, definition_revision: string, units: string});
const pipeline = object({
  artifact, response_model: artifact, configuration_sha256: string,
  stochastic_policy: string, provider_revision_known: boolean,
  dependencies: array(dependency, 1000),
});
const rule = object({
  name: string, units: string,
  role: enumeration('unspecified', 'changed', 'held_fixed', 'coupled'),
  minimum: number, maximum: number,
});
const scalar = object({name: string, units: string, value: number});
const check = object({
  state: enumeration('unspecified', 'passed', 'rejected'), evidence: artifact, detail: string,
});
const timing = object({
  support_start: optional(integer), support_end: optional(integer),
  available_at: optional(integer), vintage: string,
});
const branch = object({
  input: artifact, source_observation: artifact, parent_input_id: string,
  origin: enumeration('unspecified', 'observed', 'edited_input', 'simulated_input', 'latent_candidate'),
  representation: artifact,
  representation_kind: enumeration('unspecified', 'observation_embedding', 'task_state',
    'predicted_state', 'transformed_candidate', 'generated_content'),
  parent_representation_id: string, space, pipeline: artifact, response: artifact,
  response_execution_id: string, output_position: integer, mask: artifact,
  output_valid: boolean, upstream_succeeded: boolean, timing,
  attributes: array(scalar), axes: array(axis), response_values: array(number),
});
const pair = object({
  id: string, unit_id: string, entity_id: string, group_id: string,
  baseline: optional(branch), alternative: optional(branch), constructor: artifact,
  change_definition: string, interpretation: string, control_evidence: artifact,
  alternative_validity: check, support: check, interpretability: check,
});
const request = object({
  request_id: string, operation_run_id: string, contract_id: string, contract_revision: string,
  rer_id: string, rer_revision: string,
  claim_level: enumeration('unspecified', 'controlled_model_response',
    'observational_contrast', 'identified_intervention_effect'),
  purpose: string, population_id: string, sampling_rule: string, response_horizon: string,
  permitted_interpretation: string, permissions_ref: string, resource_budget: string,
  evidence_destination: string, code_sha256: string, configuration_sha256: string,
  temporal_mode: enumeration('unspecified', 'non_temporal', 'retrospective', 'as_of'),
  as_of: optional(integer), non_temporal_justification: string,
  expected_space: space, expected_pipeline: pipeline, response_axes: array(axis),
  selected_output_position: integer, attribute_rules: array(rule),
  control_tolerance: number, pairs: array(pair, MAX_PAIRS),
});

function schema(type) {
  switch (type.kind) {
    case 'string': return {type: 'string', maxLength: MAX_STRING_BYTES, pattern: '^[^\\u0000]*$',
      description: 'Valid Unicode without NUL, at most 65536 UTF-8 bytes (codec-enforced).'};
    case 'number': return {type: 'number', description: 'Finite IEEE-754 binary64 value.'};
    case 'integer': return {type: 'integer', minimum: type.minimum, maximum: type.maximum};
    case 'boolean': return {type: 'boolean'};
    case 'enum': return {type: 'string', enum: type.values};
    case 'array': return {type: 'array', items: schema(type.item), maxItems: type.maximum};
    case 'optional': return {anyOf: [{type: 'null'}, schema(type.item)]};
    case 'object': return {type: 'object', additionalProperties: false,
      required: Object.keys(type.fields),
      properties: Object.fromEntries(Object.entries(type.fields).map(([key, item]) => [key, schema(item)]))};
    default: throw new Error('Unknown internal schema descriptor');
  }
}

export const inputSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:input:1',
  title: 'OR06-PCR recorded paired-response request, tool boundary version 1',
  description: 'All typed Request fields are explicit, including null optional fields. '
    + 'Structural validity does not establish operation admission, actual artifact integrity, '
    + 'semantic truth, permissions or scientific evidence. Encoded requests are limited to 64 MiB.',
  ...schema(request),
};

function fail(location, detail) { throw new Error(`${location}: ${detail}`); }
function validText(value, location) {
  if (typeof value !== 'string') fail(location, 'expected string');
  if (value.includes('\0')) fail(location, 'NUL is not supported');
  // Buffer.from() replaces unmatched surrogates; reject before encoding instead.
  for (let index = 0; index < value.length; ++index) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) fail(location, 'unpaired Unicode surrogate');
    } else if (code >= 0xdc00 && code <= 0xdfff) fail(location, 'unpaired Unicode surrogate');
  }
  if (Buffer.byteLength(value, 'utf8') > MAX_STRING_BYTES) fail(location, 'string exceeds 65536 UTF-8 bytes');
}

function walk(type, value, location, token, references) {
  switch (type.kind) {
    case 'string': validText(value, location); token(value); return;
    case 'number':
      if (typeof value !== 'number' || !Number.isFinite(value)) fail(location, 'expected finite number');
      token(Object.is(value, -0) ? '-0' : String(value)); return;
    case 'integer':
      if (!Number.isSafeInteger(value) || value < type.minimum || value > type.maximum)
        fail(location, 'expected safe integer in the declared range');
      token(String(value)); return;
    case 'boolean':
      if (typeof value !== 'boolean') fail(location, 'expected boolean');
      token(String(value)); return;
    case 'enum':
      if (!type.values.includes(value)) fail(location, `expected one of ${type.values.join(', ')}`);
      token(value); return;
    case 'optional':
      token(value === null ? '0' : '1');
      if (value !== null) walk(type.item, value, location, token, references);
      return;
    case 'array':
      if (!Array.isArray(value) || value.length > type.maximum) fail(location, `expected array of at most ${type.maximum} items`);
      if (Reflect.ownKeys(value).length !== value.length + 1)
        fail(location, 'sparse arrays or additional array properties are unsupported');
      for (let index = 0; index < value.length; ++index) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
        if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value'))
          fail(`${location}[${index}]`, 'expected enumerable array data item');
      }
      token(String(value.length));
      value.forEach((item, index) => walk(type.item, item, `${location}[${index}]`, token, references));
      return;
    case 'object': {
      if (value === null || typeof value !== 'object' || Array.isArray(value) ||
          ![Object.prototype, null].includes(Object.getPrototypeOf(value))) fail(location, 'expected plain object');
      const names = Object.keys(type.fields), own = Reflect.ownKeys(value);
      for (const key of own) {
        if (typeof key !== 'string' || !Object.hasOwn(type.fields, key)) fail(location, `unknown field ${String(key)}`);
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor.enumerable || !Object.hasOwn(descriptor, 'value')) fail(`${location}.${key}`, 'expected enumerable data field');
      }
      for (const key of names) {
        if (!Object.hasOwn(value, key)) fail(location, `missing field ${key}`);
        walk(type.fields[key], value[key], `${location}.${key}`, token, references);
      }
      if (type.artifact) references?.push({id: value.id, sha256: value.sha256});
      return;
    }
    default: throw new Error('Unknown internal schema descriptor');
  }
}

function process(input, keepBytes, references) {
  const chunks = [];
  let length = 0;
  const token = value => {
    const size = Buffer.byteLength(value, 'utf8');
    length += String(size).length + size + 2;
    if (length > MAX_REQUEST_BYTES) throw new Error('Request exceeds the 64 MiB private wire limit');
    if (keepBytes) chunks.push(Buffer.from(`${size}:${value},`, 'utf8'));
  };
  token('mityane-paired-response-request-v1');
  walk(request, input, 'input', token, references);
  return keepBytes ? Buffer.concat(chunks, length) : undefined;
}

export function encodeRequest(input) { return process(input, true); }
export function collectArtifactRefs(input) {
  const references = [];
  process(input, false, references);
  const seen = new Set();
  return references.filter(ref => {
    const key = JSON.stringify([ref.id, ref.sha256]);
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
}
