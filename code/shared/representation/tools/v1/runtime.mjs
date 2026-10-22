// Common file boundary for operation tools. Numerical semantics stay with owners.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual} from 'node:util';
import {requestEnvelopeSchema, preparationSpecSchema, preparedSchema, filesSchema,
  checkSchema, resultSchema, loadedSchema, validateSchema} from './schemas.mjs';
export {requestEnvelopeSchema} from './schemas.mjs';
import {
  admitFiles, readAdmitted, parseJSON, jsonBytes, sha256, writeNew,
  createExclusiveDirectory, requireCondition as require, inside, safeRelative,
} from '../../../artifacts/node/v1/files.mjs';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../..');
export const REQUEST_VERSION = 'mityane.operation.request/v1';
export const RESULT_VERSION = 'mityane.operation.result/v1';
const HASH = /^[0-9a-f]{64}$/;
const MAX_REQUEST = 64 * 1024 * 1024;
const MAX_ARTIFACTS = 256 * 1024 * 1024;

export function keys(value, expected, label) {
  require(value && typeof value === 'object' && !Array.isArray(value), `${label}: object required`);
  require(Object.keys(value).length === expected.length && expected.every(key => Object.hasOwn(value, key)),
    `${label}: expected exactly ${expected.join(', ')}`);
}
export function readBoundJSON(filename, limit = MAX_REQUEST) {
  const row = admitFiles([path.resolve(filename)])[0];
  require(row.bytes <= limit, `file exceeds ${limit} byte limit: ${filename}`);
  const bytes = readAdmitted(row);
  return {bytes, value: parseJSON(bytes, {duplicates: 'reject', integers: 'number', allowBOM: false})};
}
function definition(registry, id) {
  const found = registry.find(tool => tool.descriptor.id === id);
  require(found, `unknown operation: ${id}`);
  return found;
}
export function listTools(registry) {
  return registry.map(({descriptor: {id, revision, family, title, claim_levels}}) =>
    ({id, revision, family, title, claim_levels}));
}
export function describeTool(registry, id) {
  const descriptor = definition(registry, id).descriptor;
  const native = descriptor.native_result_schema ?? {};
  const specialize = schema => ({...schema, properties: {...schema.properties,
    operation: {const: {id: descriptor.id, revision: descriptor.revision}}}});
  const loaded = loadedSchema(native, descriptor.input_schema ?? {type: 'object'});
  loaded.properties = {...loaded.properties, record: specialize(loaded.properties.record),
    request: specialize(loaded.properties.request)};
  return {...descriptor,
    request_schema: {...specialize(requestEnvelopeSchema), properties: {
      ...specialize(requestEnvelopeSchema).properties, input: descriptor.input_schema ?? {type: 'object'}}},
    preparation_schema: {...specialize(preparationSpecSchema), properties: {
      ...specialize(preparationSpecSchema).properties, input: descriptor.input_schema ?? {type: 'object'}}},
    prepared_schema: specialize(preparedSchema), check_schema: specialize(checkSchema(native)),
    result_schema: specialize(resultSchema(native)), files_schema: filesSchema,
    loaded_schema: loaded,
  };
}
export function createToolset(registry) {
  const extension = async (method, id, options) => {
    require(options && typeof options === 'object' && !Array.isArray(options), 'explicit operation options required');
    const tool = definition(registry, id);
    require(typeof tool[method] === 'function', `operation supplies no ${method} implementation: ${id}`);
    const schema = tool.descriptor[`${method.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`)}_options_schema`];
    if (schema) validateSchema(schema, options, {label: `${method} options`});
    return tool[method]({...options, operations: createToolset(registry), output: outputPath(tool, options.output)});
  };
  return Object.freeze({
    list: () => listTools(registry), describe: id => describeTool(registry, id),
    check: request => checkTool(registry, request), run: (request, output) => runTool(registry, request, output),
    prepare: (spec, output) => prepareTool(registry, spec, output),
    read: result => readResult(result, registry), inspect: result => inspectResult(result, registry),
    workflow: async (id, options) => {
      const tool = definition(registry, id);
      require(typeof tool.workflow === 'function', `operation supplies no workflow: ${id}`);
      return tool.workflow({...options, operations: createToolset(registry), output: outputPath(tool, options?.output)});
    },
    sweep: (id, options) => extension('sweep', id, options),
    diff: (id, options) => extension('diff', id, options),
    verifyStudy: (id, options) => extension('verifyStudy', id, options),
  });
}

export function outputPath(tool, output) {
  require(typeof output === 'string' && output.length > 0, 'explicit new output directory required');
  const target = path.resolve(ROOT, output);
  const allowed = path.resolve(ROOT, tool.outputRoot);
  require(target !== allowed && inside(allowed, target), `output must be a new directory inside ${allowed}`);
  require(!fs.existsSync(target), `output already exists: ${target}`);
  return target;
}
function admitInventory(envelope, requestFile, references) {
  require(Array.isArray(envelope.artifacts) && envelope.artifacts.length <= 100000,
    'bounded artifact inventory required');
  const needed = new Map();
  for (const ref of references) {
    require(typeof ref.id === 'string' && ref.id.trim() && HASH.test(ref.sha256), 'valid ArtifactRef required');
    require(!needed.has(ref.id) || needed.get(ref.id) === ref.sha256, `conflicting artifact identity: ${ref.id}`);
    needed.set(ref.id, ref.sha256);
  }
  require(envelope.artifacts.length === needed.size, 'inventory must cover exactly the referenced artifact identities');
  const seen = new Set(), base = path.dirname(requestFile);
  let total = 0;
  const rows = envelope.artifacts.map((entry, index) => {
    keys(entry, ['id', 'sha256', 'path'], `artifacts[${index}]`);
    require(!seen.has(entry.id) && needed.get(entry.id) === entry.sha256,
      `missing, extra, duplicate or mismatched inventory identity: ${entry.id}`);
    seen.add(entry.id);
    require(typeof entry.path === 'string' && safeRelative(entry.path) === entry.path &&
      !entry.path.includes('\\'), `canonical relative artifact path required: ${entry.path}`);
    const absolute = path.resolve(base, entry.path);
    require(inside(base, absolute), 'artifact path escapes request directory');
    return {...entry, absolute, retained_path: `artifacts/${String(index).padStart(5, '0')}.bin`};
  });
  const admitted = admitFiles(rows.map(row => row.absolute), {root: base});
  for (const row of admitted) { total += row.bytes; require(total <= MAX_ARTIFACTS, 'artifact inventory exceeds 256 MiB'); }
  return rows.map((row, index) => {
    const bytes = readAdmitted(admitted[index]);
    require(sha256(bytes) === row.sha256, `artifact hash mismatch: ${row.id}`);
    return {...row, bytes};
  });
}
async function prepare(registry, requestFile) {
  const absolute = path.resolve(ROOT, requestFile);
  const {bytes, value: envelope} = readBoundJSON(absolute);
  keys(envelope, ['schema_version', 'operation', 'input', 'artifacts'], 'request');
  require(envelope.schema_version === REQUEST_VERSION, 'unsupported request schema_version');
  keys(envelope.operation, ['id', 'revision'], 'operation');
  const tool = definition(registry, envelope.operation.id);
  require(envelope.operation.revision === tool.descriptor.revision, 'unsupported operation revision');
  const wire = tool.encode(envelope.input);
  const inventory = admitInventory(envelope, absolute, tool.references(envelope.input));
  tool.verify(envelope.input, new Map(inventory.map(row => [row.id, row.bytes])));
  return {tool, envelope, requestBytes: bytes, requestFile: absolute, inventory, wire};
}

// Packaging is structural and byte-bound only. It deliberately never calls an
// adapter's check/execute hooks or infers scientific metadata.
export async function prepareTool(registry, specFile, output) {
  const absolute = path.resolve(ROOT, specFile);
  const {bytes, value: spec} = readBoundJSON(absolute);
  validateSchema(preparationSpecSchema, spec, {label: 'preparation specification'});
  const tool = definition(registry, spec.operation.id);
  require(spec.operation.revision === tool.descriptor.revision, 'unsupported operation revision');
  tool.encode(spec.input);
  const inventory = admitInventory(spec, absolute, tool.references(spec.input));
  tool.verify(spec.input, new Map(inventory.map(row => [row.id, row.bytes])));
  const target = outputPath(tool, output);
  const request = {schema_version: REQUEST_VERSION, operation: spec.operation, input: spec.input,
    artifacts: inventory.map(row => ({id: row.id, sha256: row.sha256, path: row.retained_path}))};
  const requestBytes = jsonBytes(request);
  require(requestBytes.length <= MAX_REQUEST, 'prepared request exceeds byte limit');
  createExclusiveDirectory(target, {root: ROOT});
  try {
    fs.mkdirSync(path.join(target, 'artifacts'));
    for (const row of inventory) writeNew(path.join(target, row.retained_path), row.bytes);
    writeNew(path.join(target, 'preparation.json'), bytes);
    writeNew(path.join(target, 'request.json'), requestBytes);
  } catch (error) {
    try { writeNew(path.join(target, 'failure.json'), jsonBytes({status: 'publication_failed', message: error.message})); } catch {}
    throw error;
  }
  return validateSchema(preparedSchema, {schema_version: 'mityane.operation.prepared/v1',
    operation: request.operation, request_file: path.join(target, 'request.json'), artifact_count: inventory.length});
}
function inventoryRecords(inventory) {
  return inventory.map(({id, sha256: hash, path: source_path, retained_path, bytes}) =>
    ({id, sha256: hash, bytes: bytes.length, source_path, retained_path}));
}
export function exitCode(status) { return status === 'rejected' ? 3 : status === 'partial' ? 4 : 0; }
function validateOutcome(outcome) {
  require(['completed', 'rejected', 'partial'].includes(outcome.status) &&
    Array.isArray(outcome.details) && outcome.details.every(line => typeof line === 'string'), 'invalid tool outcome');
  const summary = outcome.summary;
  keys(summary, ['requested', 'succeeded', 'abstained', 'failed', 'unresolved', 'success_rate'], 'native summary');
  for (const key of ['requested', 'succeeded', 'abstained', 'failed', 'unresolved'])
    require(Number.isSafeInteger(summary[key]) && summary[key] >= 0, 'invalid native count');
  if (outcome.status === 'rejected') {
    require(!summary.succeeded && !summary.abstained && !summary.failed && !summary.unresolved &&
      summary.success_rate === null, 'rejected admission invented executed counts');
    return;
  }
  require(summary.requested === summary.succeeded + summary.abstained + summary.failed + summary.unresolved,
    'native denominator mismatch');
  require(summary.requested === 0 ? summary.success_rate === null :
    typeof summary.success_rate === 'number' && Number.isFinite(summary.success_rate) &&
    Math.abs(summary.success_rate - summary.succeeded / summary.requested) <= Number.EPSILON,
  'invalid success rate');
  require(outcome.status !== 'completed' || (!summary.failed && !summary.unresolved), 'completed outcome contains unfinished/failed items');
}
async function execute(prepared, checking = false) {
  const started = new Date(), timer = performance.now();
  require(!checking || typeof prepared.tool.check === 'function', 'operation supplies no read-only check implementation');
  const execution = checking ? await prepared.tool.check(prepared.wire) : await prepared.tool.execute(prepared.wire);
  const outcome = prepared.tool.outcome(execution.result, prepared.envelope.input);
  validateOutcome(outcome);
  return {
    result: execution.result,
    ...outcome,
    execution: {
      started_at: started.toISOString(), finished_at: new Date().toISOString(),
      elapsed_ms: performance.now() - timer, node_version: process.version,
      platform: process.platform, architecture: process.arch,
      native_binary: execution.native_binary ?? null, source_files: execution.source_files,
    },
  };
}
export async function checkTool(registry, requestFile) {
  const prepared = await prepare(registry, requestFile);
  const executed = await execute(prepared, true);
  return validateSchema(checkSchema(prepared.tool.descriptor.native_result_schema ?? {}), {
    schema_version: 'mityane.operation.check/v1', operation: prepared.envelope.operation,
    valid: executed.status === 'completed', status: executed.status,
    summary: executed.summary, result: executed.result,
    verified_artifacts: inventoryRecords(prepared.inventory),
  });
}
export function formatSummary(record) {
  const s = record.summary;
  const lines = [`${record.operation.id} ${record.operation.revision}: ${record.status}`,
    `Claim: ${record.claim_level}`, `Requested ${s.requested}; succeeded ${s.succeeded}; abstained ${s.abstained}; failed ${s.failed}; unresolved ${s.unresolved}.`];
  lines.push(...record.details, ...record.warnings, '');
  return lines.join('\n');
}
export async function runTool(registry, requestFile, output) {
  const prepared = await prepare(registry, requestFile);
  const target = outputPath(prepared.tool, output);
  const executed = await execute(prepared);
  const record = {
    schema_version: RESULT_VERSION, operation: prepared.envelope.operation, status: executed.status,
    input: {request_file: 'request.json', sha256: sha256(prepared.requestBytes),
      operation_run_id: prepared.tool.context(prepared.envelope.input).operation_run_id},
    claim_level: prepared.tool.context(prepared.envelope.input).claim_level, summary: executed.summary,
    details: executed.details,
    result: executed.result, verified_artifacts: inventoryRecords(prepared.inventory),
    execution: executed.execution, warnings: prepared.tool.descriptor.limitations,
  };
  validateSchema(resultSchema(prepared.tool.descriptor.native_result_schema ?? {}), record);
  createExclusiveDirectory(target, {root: ROOT});
  try {
    fs.mkdirSync(path.join(target, 'artifacts'));
    const files = [
      {path: 'request.json', bytes: prepared.requestBytes},
      {path: 'result.json', bytes: jsonBytes(record)},
      {path: 'summary.txt', bytes: Buffer.from(formatSummary(record))},
      ...prepared.inventory.map(row => ({path: row.retained_path, bytes: row.bytes})),
    ];
    for (const row of files) writeNew(path.join(target, row.path), row.bytes);
    writeNew(path.join(target, 'manifest.json'), jsonBytes({
      schema_version: 'mityane.operation.files/v1',
      files: files.map(row => ({path: row.path, bytes: row.bytes.length, sha256: sha256(row.bytes)})),
    }));
  } catch (error) {
    // Preserve the incomplete directory; never remove user data after a write failure.
    try { writeNew(path.join(target, 'failure.json'), jsonBytes({status: 'publication_failed', message: error.message})); } catch {}
    throw error;
  }
  return {result_file: path.join(target, 'result.json'), summary_file: path.join(target, 'summary.txt'),
    status: record.status, summary: record.summary};
}
export function readResult(resultFile, registry = undefined) {
  const filename = path.resolve(ROOT, resultFile), directory = path.dirname(filename);
  require(path.basename(filename) === 'result.json', 'result loading expects result.json');
  const manifest = readBoundJSON(path.join(directory, 'manifest.json')).value;
  validateSchema(filesSchema, manifest, {label: 'saved file manifest'});
  keys(manifest, ['schema_version', 'files'], 'file manifest');
  require(manifest.schema_version === 'mityane.operation.files/v1' && Array.isArray(manifest.files), 'unsupported file manifest');
  require(manifest.files.length <= 100003, 'file manifest exceeds entry limit');
  const seen = new Set();
  let total = 0;
  for (const entry of manifest.files) {
    keys(entry, ['path', 'bytes', 'sha256'], 'file entry');
    require(typeof entry.path === 'string' && safeRelative(entry.path) === entry.path && !seen.has(entry.path) &&
      HASH.test(entry.sha256) && Number.isSafeInteger(entry.bytes) && entry.bytes >= 0, 'invalid file manifest row');
    seen.add(entry.path);
    total += entry.bytes;
    require(entry.bytes <= MAX_ARTIFACTS && total <= MAX_ARTIFACTS + 3 * MAX_REQUEST, 'saved bundle exceeds byte limit');
  }
  for (const name of ['request.json', 'result.json', 'summary.txt']) require(seen.has(name), `manifest omits ${name}`);
  const admitted = admitFiles(manifest.files.map(entry => path.resolve(directory, entry.path)), {root: directory});
  admitted.forEach((row, index) => require(row.bytes === manifest.files[index].bytes, `saved file size mismatch: ${manifest.files[index].path}`));
  const retained = new Map();
  admitted.forEach((row, index) => {
    const bytes = readAdmitted(row), entry = manifest.files[index];
    require(sha256(bytes) === entry.sha256, `saved file hash mismatch: ${entry.path}`);
    if (['request.json', 'result.json', 'summary.txt'].includes(entry.path)) {
      require(bytes.length <= MAX_REQUEST, `saved record exceeds byte limit: ${entry.path}`);
    }
    retained.set(entry.path, bytes);
  });
  const record = parseJSON(retained.get('result.json'), {duplicates: 'reject', integers: 'number', allowBOM: false});
  validateSchema(resultSchema(), record, {label: 'saved result'});
  validateOutcome(record);
  const requestBytes = retained.get('request.json');
  require(sha256(requestBytes) === record.input.sha256, 'saved request identity mismatch');
  require(record.input.request_file === 'request.json', 'saved request path mismatch');
  const request = parseJSON(requestBytes, {duplicates: 'reject', integers: 'number', allowBOM: false});
  validateSchema(requestEnvelopeSchema, request, {label: 'retained request'});
  require(isDeepStrictEqual(request.operation, record.operation), 'saved operation/revision mismatch');
  require(request.artifacts.length === record.verified_artifacts.length, 'saved inventory length mismatch');
  require(manifest.files.length === record.verified_artifacts.length + 3, 'saved manifest contains extra or missing files');
  const seenIds = new Set(), seenRetained = new Set(), artifactBytes = new Map();
  for (const [index, artifact] of record.verified_artifacts.entries()) {
    const supplied = request.artifacts[index];
    require(!seenIds.has(artifact.id) && !seenRetained.has(artifact.retained_path), 'duplicate retained artifact identity or path');
    require(artifact.retained_path.startsWith('artifacts/') && safeRelative(artifact.retained_path) === artifact.retained_path,
      'invalid retained artifact path');
    require(supplied.id === artifact.id && supplied.sha256 === artifact.sha256 && supplied.path === artifact.source_path,
      `saved request artifact mismatch: ${artifact.id}`);
    seenIds.add(artifact.id); seenRetained.add(artifact.retained_path);
    const row = manifest.files.find(entry => entry.path === artifact.retained_path);
    require(row && row.sha256 === artifact.sha256 && row.bytes === artifact.bytes, `retained artifact identity mismatch: ${artifact.id}`);
    artifactBytes.set(artifact.id, retained.get(artifact.retained_path));
  }
  if (registry) {
    const tool = definition(registry, record.operation.id);
    require(tool.descriptor.revision === record.operation.revision, 'unsupported saved operation revision');
    validateSchema(resultSchema(tool.descriptor.native_result_schema ?? {}), record, {label: 'saved operation result'});
    tool.encode(request.input);
    const refs = tool.references(request.input), expected = new Map();
    for (const ref of refs) {
      require(!expected.has(ref.id) || expected.get(ref.id) === ref.sha256, 'conflicting saved artifact reference');
      expected.set(ref.id, ref.sha256);
    }
    require(expected.size === seenIds.size && record.verified_artifacts.every(row => expected.get(row.id) === row.sha256),
      'retained inventory does not cover operation references');
    tool.verify(request.input, artifactBytes);
    const context = tool.context(request.input), outcome = tool.outcome(record.result, request.input);
    validateOutcome(outcome);
    require(context.operation_run_id === record.input.operation_run_id && context.claim_level === record.claim_level,
      'saved claim or run identity mismatch');
    for (const key of ['status', 'summary', 'details']) require(isDeepStrictEqual(outcome[key], record[key]), `saved ${key} disagrees with operation outcomes`);
    require(isDeepStrictEqual(record.warnings, tool.descriptor.limitations), 'saved operation limitations mismatch');
  }
  const summary = formatSummary(record);
  require(retained.get('summary.txt').equals(Buffer.from(summary)), 'saved summary does not describe result');
  return validateSchema(loadedSchema(), {schema_version: 'mityane.operation.loaded/v1', result_file: filename,
    record, request, manifest, artifacts: record.verified_artifacts.map(row => ({id: row.id, sha256: row.sha256,
      bytes: row.bytes, path: path.join(directory, row.retained_path)}))});
}
export function inspectResult(resultFile, registry = undefined) {
  return formatSummary(readResult(resultFile, registry).record);
}
