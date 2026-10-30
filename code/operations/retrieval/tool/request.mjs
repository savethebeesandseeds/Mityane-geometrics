// Structural JSON boundary for OR01-RET. Admission stays in the C++ bridge.
// Private wire fields follow the object-field order below, after one version token.
export const MAX_REQUEST_BYTES = 64 * 1024 * 1024;
export const MAX_CORPUS = 4096;
const MAX_STRING_BYTES = 65536;
const MAX_DIMENSIONS = 8192;
const SAFE = Number.MAX_SAFE_INTEGER;
const string = {kind: 'string'};
const hash = {kind: 'hash'};
const number = {kind: 'number'};
const integer = {kind: 'integer', minimum: -SAFE, maximum: SAFE};
const size = {kind: 'integer', minimum: 0, maximum: SAFE};
const boolean = {kind: 'boolean'};
const nothing = {kind: 'null'};
const enumeration = (...values) => ({kind: 'enum', values});
const array = (item, maximum = MAX_CORPUS) => ({kind: 'array', item, maximum});
const optional = item => ({kind: 'optional', item});
const object = (fields, artifact = false) => ({kind: 'object', fields, artifact});
const kinds = enumeration('observation_embedding', 'task_state', 'predicted_state',
  'transformed_candidate', 'generated_content');

const artifact = object({id: string, sha256: hash}, true);
const timed = {
  available_at: optional(integer),
  window_start: optional(integer),
  window_end: optional(integer),
};
const vector = {
  representation_kind: kinds,
  values: array(number, MAX_DIMENSIONS),
  ...timed,
};
const member = object({
  source_id: string, parent: artifact, ...vector,
});
const space = object({
  artifact, provider: artifact, revision: artifact, fitted_transform: artifact,
  space_id: string, snapshot_id: string, provider_id: string, revision_id: string,
  dimensions: {kind: 'integer', minimum: 1, maximum: MAX_DIMENSIONS},
  normalization: enumeration('none', 'l2'),
  metric: enumeration('cosine', 'l2'),
  task_role: string,
  representation_kind: kinds,
  fitted_available_at: optional(integer),
});
const query = object({
  id: string, source_id: string, parent: artifact, ...vector,
});
const request = object({
  request_id: string, operation_run_id: string,
  contract_id: string, contract_revision: string, rer_id: string, rer_revision: string,
  claim_level: enumeration('exploratory_description', 'task_association',
    'controlled_model_response', 'identified_intervention_effect'),
  profile: enumeration('local-saved-vectors', 'text-gemini-embedding-2'),
  live_call: boolean, purpose: string,
  code_sha256: hash, configuration_sha256: hash, evidence_destination: string,
  temporal_mode: enumeration('non_temporal', 'retrospective', 'as_of'),
  as_of: optional(integer), non_temporal_justification: string,
  space, query, corpus: array(member), k: size, exclude_self: boolean,
  excluded_source_ids: array(string),
  overlap_policy: enumeration('none', 'exclude', 'fail'),
  relevance_rule: nothing,
});

function schema(type) {
  switch (type.kind) {
    case 'string': return {type: 'string', maxLength: MAX_STRING_BYTES, pattern: '^[^\\u0000]*$'};
    case 'hash': return {type: 'string', pattern: '^[0-9a-f]{64}$'};
    case 'number': return {type: 'number'};
    case 'integer': return {type: 'integer', minimum: type.minimum, maximum: type.maximum};
    case 'boolean': return {type: 'boolean'};
    case 'null': return {type: 'null'};
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
  $id: 'urn:mityane:operations:retrieval:input:1',
  title: 'OR01-RET ranked retrieval request, tool boundary version 1',
  description: 'Saved vectors only. A null relevance rule means this call is exploratory. '
    + 'Structural validity does not admit a corpus, call Gemini, or establish relevance.',
  ...schema(request),
};

function fail(location, detail) { throw new Error(`${location}: ${detail}`); }
function validText(value, location) {
  if (typeof value !== 'string') fail(location, 'expected string');
  if (value.includes('\0')) fail(location, 'NUL is not supported');
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
    case 'hash':
      if (typeof value !== 'string' || !/^[0-9a-f]{64}$/.test(value)) fail(location, 'expected sha256');
      token(value); return;
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
    case 'null':
      if (value !== null) fail(location, 'expected null');
      token('null'); return;
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
      token(String(value.length));
      value.forEach((item, index) => walk(type.item, item, `${location}[${index}]`, token, references));
      return;
    case 'object': {
      if (value === null || typeof value !== 'object' || Array.isArray(value) ||
          ![Object.prototype, null].includes(Object.getPrototypeOf(value))) fail(location, 'expected plain object');
      const names = Object.keys(type.fields);
      for (const key of Reflect.ownKeys(value)) {
        if (typeof key !== 'string' || !Object.hasOwn(type.fields, key)) fail(location, `unknown field ${String(key)}`);
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
  token('mityane-retrieval-request-v1');
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
    seen.add(key);
    return true;
  });
}
