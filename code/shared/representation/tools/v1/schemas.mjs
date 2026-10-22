// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Public, provider-neutral tool records. Operation payload schemas stay with
// their owning adapter; schema validation does not establish scientific claims.
const DIALECT = 'https://json-schema.org/draft/2020-12/schema';
const text = {type: 'string'};
const nonempty = {type: 'string', minLength: 1};
const count = {type: 'integer', minimum: 0, maximum: Number.MAX_SAFE_INTEGER};
const hash = {type: 'string', pattern: '^[0-9a-f]{64}$'};
const status = {type: 'string', enum: ['completed', 'rejected', 'partial']};
const strings = {type: 'array', items: text};
const nullable = schema => ({anyOf: [{type: 'null'}, schema]});
const object = properties => ({type: 'object', additionalProperties: false,
  required: Object.keys(properties), properties});
const document = (id, title, schema) => ({$schema: DIALECT,
  $id: `urn:mityane:operation:${id}:1`, title, ...schema});

export const operationSchema = object({id: nonempty, revision: nonempty});
export const summarySchema = object({requested: count, succeeded: count, abstained: count,
  failed: count, unresolved: count,
  success_rate: nullable({type: 'number', minimum: 0, maximum: 1})});

// Runtime path admission additionally verifies canonical spelling, file type,
// containment, aliases and the exact current bytes. A schema cannot certify these.
const relativePath = {type: 'string', minLength: 1,
  pattern: '^(?!\\.{1,2}(?:/|$))(?!.*\\/\\.{1,2}(?:/|$))[^/\\\\:\\u0000\\t\\r\\n]+(?:/[^/\\\\:\\u0000\\t\\r\\n]+)*$',
  description: 'Canonical relative file path; no redirects, empty segments or parent traversal.'};
export const artifactBindingSchema = object({id: nonempty, sha256: hash, path: relativePath});
export const verifiedArtifactSchema = object({id: nonempty, sha256: hash, bytes: count,
  source_path: relativePath, retained_path: relativePath});
const verifiedArtifacts = {type: 'array', maxItems: 100000, items: verifiedArtifactSchema};

export const requestEnvelopeSchema = document('request', 'Operation request envelope v1', object({
  schema_version: {const: 'mityane.operation.request/v1'}, operation: operationSchema,
  input: {type: 'object', description: 'Operation-owned input schema from describe.'},
  artifacts: {type: 'array', maxItems: 100000, items: artifactBindingSchema},
}));
export const preparationSpecSchema = document('prepare', 'Explicit request preparation specification v1',
  {type: 'object', additionalProperties: false, required: requestEnvelopeSchema.required,
    properties: {...requestEnvelopeSchema.properties,
    schema_version: {const: 'mityane.operation.prepare/v1'}}});

export const preparedSchema = document('prepared', 'Prepared operation request report v1', object({
  schema_version: {const: 'mityane.operation.prepared/v1'}, operation: operationSchema,
  request_file: nonempty, artifact_count: count,
}));

export function checkSchema(native = {}) {
  return document('check', 'Operation check result v1', object({
    schema_version: {const: 'mityane.operation.check/v1'}, operation: operationSchema,
    valid: {type: 'boolean'}, status, summary: summarySchema, result: native,
    verified_artifacts: verifiedArtifacts,
  }));
}

const fileIdentity = object({path: nonempty, sha256: hash});
const executionSchema = object({
  started_at: {type: 'string', pattern: '^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}Z$'},
  finished_at: {type: 'string', pattern: '^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}Z$'},
  elapsed_ms: {type: 'number', minimum: 0}, node_version: nonempty,
  platform: nonempty, architecture: nonempty, native_binary: nullable(fileIdentity),
  source_files: {type: 'array', maxItems: 100000, items: fileIdentity},
});
export function resultSchema(native = {}) {
  return document('result', 'Saved operation result v1', object({
    schema_version: {const: 'mityane.operation.result/v1'}, operation: operationSchema, status,
    input: object({request_file: {const: 'request.json'}, sha256: hash, operation_run_id: text}),
    claim_level: text, summary: summarySchema, details: strings, result: native,
    verified_artifacts: verifiedArtifacts, execution: executionSchema, warnings: strings,
  }));
}

export const filesSchema = document('files', 'Saved operation file manifest v1', object({
  schema_version: {const: 'mityane.operation.files/v1'},
  files: {type: 'array', minItems: 3, maxItems: 100003,
    items: object({path: relativePath, bytes: count, sha256: hash})},
}));

export function loadedSchema(native = {}, input = {type: 'object'}) {
  return document('loaded', 'Verified structured operation result v1', object({
    schema_version: {const: 'mityane.operation.loaded/v1'}, result_file: nonempty,
    record: resultSchema(native), request: {...requestEnvelopeSchema,
      properties: {...requestEnvelopeSchema.properties, input}}, manifest: filesSchema,
    artifacts: {type: 'array', maxItems: 100000,
      items: object({id: nonempty, sha256: hash, bytes: count, path: nonempty})},
  }));
}

// Dependency-free validation for the exact JSON Schema vocabulary emitted here
// and by operation adapters. This is not a general JSON Schema implementation:
// unknown assertion keywords fail closed instead of being silently ignored.
const keywords = new Set(['$schema', '$id', 'title', 'description', 'default', 'examples',
  'type', 'const', 'enum', 'properties', 'required', 'additionalProperties', 'items',
  'minItems', 'maxItems', 'minLength', 'maxLength', 'minimum', 'maximum', 'pattern',
  'anyOf', 'allOf', 'if', 'then', 'else']);
function same(left, right) {
  if (left === right) return true;
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object' ||
      Array.isArray(left) !== Array.isArray(right)) return false;
  const names = Object.keys(left);
  return names.length === Object.keys(right).length && names.every(name =>
    Object.hasOwn(right, name) && same(left[name], right[name]));
}
function fail(at, detail) { throw new Error(`${at}: ${detail}`); }
function walk(schema, value, at, depth = 0) {
  if (depth > 128) fail(at, 'schema nesting exceeds 128 levels');
  if (schema === true) return;
  if (schema === false) fail(at, 'schema forbids value');
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) fail(at, 'invalid schema');
  for (const keyword of Object.keys(schema))
    if (!keywords.has(keyword)) fail(at, `unsupported schema keyword ${keyword}`);
  if (Object.hasOwn(schema, 'const') && !same(value, schema.const)) fail(at, 'value differs from const');
  if (schema.enum && !schema.enum.some(item => same(value, item))) fail(at, 'value is outside enum');
  if (schema.type !== undefined) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    const matches = type => type === 'null' ? value === null :
      type === 'array' ? Array.isArray(value) :
      type === 'object' ? value !== null && typeof value === 'object' && !Array.isArray(value) :
      type === 'integer' ? Number.isInteger(value) && Number.isFinite(value) :
      type === 'number' ? typeof value === 'number' && Number.isFinite(value) :
      type === 'string' ? typeof value === 'string' :
      type === 'boolean' ? typeof value === 'boolean' : fail(at, `unsupported schema type ${type}`);
    if (!types.some(matches)) fail(at, `expected ${types.join(' or ')}`);
  }
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    for (const name of schema.required ?? [])
      if (!Object.hasOwn(value, name)) fail(at, `missing ${name}`);
    for (const [name, item] of Object.entries(value)) {
      if (Object.hasOwn(schema.properties ?? {}, name))
        walk(schema.properties[name], item, `${at}.${name}`, depth + 1);
      else if (schema.additionalProperties === false) fail(at, `unexpected ${name}`);
      else if (schema.additionalProperties && typeof schema.additionalProperties === 'object')
        walk(schema.additionalProperties, item, `${at}.${name}`, depth + 1);
    }
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) fail(at, 'too few items');
    if (schema.maxItems !== undefined && value.length > schema.maxItems) fail(at, 'too many items');
    if (schema.items !== undefined) value.forEach((item, index) => walk(schema.items, item, `${at}[${index}]`, depth + 1));
  }
  if (typeof value === 'string') {
    const length = [...value].length;
    if (schema.minLength !== undefined && length < schema.minLength) fail(at, 'string is too short');
    if (schema.maxLength !== undefined && length > schema.maxLength) fail(at, 'string is too long');
    if (schema.pattern !== undefined && !new RegExp(schema.pattern, 'u').test(value)) fail(at, 'string does not match pattern');
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail(at, 'expected finite JSON number');
    if (schema.minimum !== undefined && value < schema.minimum) fail(at, 'number is below minimum');
    if (schema.maximum !== undefined && value > schema.maximum) fail(at, 'number is above maximum');
  }
  if (schema.allOf) for (const option of schema.allOf) walk(option, value, at, depth + 1);
  if (schema.anyOf) {
    let matched = false;
    for (const option of schema.anyOf) {
      try { walk(option, value, at, depth + 1); matched = true; break; } catch {}
    }
    if (!matched) fail(at, 'value matches no permitted schema');
  }
  if (schema.if) {
    let matched = false;
    try { walk(schema.if, value, at, depth + 1); matched = true; } catch {}
    const branch = matched ? schema.then : schema.else;
    if (branch !== undefined) walk(branch, value, at, depth + 1);
  }
}
export function validateSchema(schema, value, {label = 'value'} = {}) {
  walk(schema, value, label);
  return value;
}
