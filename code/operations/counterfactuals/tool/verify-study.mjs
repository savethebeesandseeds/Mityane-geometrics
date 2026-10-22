// Verification of byte-pinned, already exposed study records. This module never
// imports a study runner, generator, fitting routine or statistical auditor.
import fs from 'node:fs';
import path from 'node:path';
import {gzipSync} from 'node:zlib';
import {randomUUID} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {ROOT} from '../../../shared/representation/tools/v1/runtime.mjs';
import {admitFiles, canonicalDirectory, createExclusiveDirectory, hashAdmitted,
  inside, jsonBytes, parseJSON, readAdmitted, requireCondition as require,
  safeRelative, sha256, writeNew} from '../../../shared/artifacts/node/v1/files.mjs';

const SELF = fileURLToPath(import.meta.url);
const HASH = /^[0-9a-f]{64}$/;
const PROFILES = {
  'OR06-SYNTH-RESP-001': {name: 'synthetic-response-v1',
    final_sha256: '82900d0336472d6457bf8cb30fea310059a2fe323154b38f7670bdd24ab39b61',
    original_attempt: '/mityane/.temp/operations/counterfactuals/synthetic-response-v1/attempt-001',
    seal: 'model-seal.json', seal_key: 'model_seal_sha256', contribution: false},
  'OR06-REP-CONTRIB-001': {name: 'representation-contribution-v1',
    final_sha256: '9730ed8355839ccfa537d527cb763604145b8e3a7217dcbb75272108fc609642',
    original_attempt: '/mityane/.temp/operations/counterfactuals/representation-contribution-v1/attempt-001',
    seal: 'seal.json', seal_key: 'seal_sha256', contribution: true},
};
const integer = (value, label) => {
  require(/^-?(0|[1-9][0-9]*)$/.test(value), `invalid integer ${label}`);
  const number = Number(value); require(Number.isSafeInteger(number), `unsafe integer ${label}`); return number;
};
const number = (value, label) => {
  require(typeof value === 'string' && value.trim() === value && value.length > 0 && value !== 'NA', `missing numerical ${label}`);
  const result = Number(value); require(Number.isFinite(result), `nonfinite ${label}`); return result;
};
function read(filename) { return readAdmitted(admitFiles([filename])[0]); }
function parse(bytes) { return parseJSON(bytes, {duplicates: 'reject', integers: 'number', allowBOM: false}); }
function json(filename) { return parse(read(filename)); }
function pin(filename, root) {
  const row = admitFiles([filename], root ? {root} : {})[0];
  return {path: root ? path.relative(root, filename) : filename, bytes: row.bytes, sha256: hashAdmitted(row)};
}
function table(bytes, header, label) {
  const lines = new TextDecoder('utf8', {fatal: true}).decode(bytes).split('\n');
  require(lines.pop() === '', `${label}: final newline required`);
  const actual = lines.shift().split('\t');
  require(JSON.stringify(actual) === JSON.stringify(header), `${label}: unsupported columns`);
  return lines.map((line, index) => {
    const values = line.split('\t'); require(values.length === header.length, `${label}: malformed row ${index + 2}`);
    return Object.fromEntries(header.map((key, i) => [key, values[i]]));
  });
}
function filesBelow(directory, root = directory) {
  canonicalDirectory(directory);
  return fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const filename = path.join(directory, entry.name);
    require(!entry.isSymbolicLink(), `retained artifact redirect: ${filename}`);
    if (entry.isDirectory()) return filesBelow(filename, root);
    require(entry.isFile(), `unsupported retained member: ${filename}`);
    return [path.relative(root, filename)];
  }).sort();
}
function relativePin(row, attempt, profile) {
  require(row && HASH.test(row.sha256) && Number.isSafeInteger(row.bytes) && row.bytes >= 0, 'invalid retained pin');
  let relative = row.path;
  if (path.isAbsolute(relative)) {
    require(inside(profile.original_attempt, relative) && relative !== profile.original_attempt, 'retained pin escapes original attempt');
    relative = path.relative(profile.original_attempt, relative);
  }
  safeRelative(relative);
  return {...row, path: relative, absolute: path.join(attempt, relative)};
}

// Scratch is an exact serialization of the in-memory archive, rather than a
// separate source of truth. Keep its initial admissions through current check.
export function bindVerificationScratch(directory, expected) {
  canonicalDirectory(directory);
  require(Array.isArray(expected) && expected.length > 0, 'explicit verification scratch inventory required');
  const names = expected.map(row => safeRelative(row.path)).sort();
  require(new Set(names).size === names.length, 'duplicate verification scratch member');
  const admissions = admitFiles(expected.map(row => path.join(directory, row.path)), {root: directory});
  const recheck = () => {
    require(JSON.stringify(filesBelow(directory)) === JSON.stringify(names), 'verification scratch roster changed');
    for (let i = 0; i < expected.length; i++) {
      const bytes = readAdmitted(admissions[i]), row = expected[i];
      require(bytes.length === row.bytes && sha256(bytes) === row.sha256, `verification scratch bytes changed: ${row.path}`);
    }
  };
  recheck();
  return recheck;
}

// Exported for focused integrity tests. The public tool selects a hard-coded,
// reviewed final-record identity; callers cannot supply a replacement anchor.
export function verifyRetainedStudyFiles(attempt, profile) {
  canonicalDirectory(attempt);
  const finalAdmission = admitFiles([path.join(attempt, 'final.json')], {root: attempt})[0];
  const finalBytes = readAdmitted(finalAdmission);
  const finalPin = {path: 'final.json', bytes: finalBytes.length, sha256: sha256(finalBytes)};
  require(finalPin.sha256 === profile.final_sha256, 'unsupported or changed retained final identity');
  const final = parse(finalBytes);
  require(final.disposition === 'completed' && final.assessment_opened === true && final.source_unchanged === true,
    'only completed, exposed retained studies can be verified');
  const manifestAdmission = admitFiles([path.join(attempt, 'file-manifest.json')], {root: attempt})[0];
  const manifestBytes = readAdmitted(manifestAdmission);
  const manifestPin = {path: 'file-manifest.json', bytes: manifestBytes.length, sha256: sha256(manifestBytes)};
  require(manifestPin.sha256 === final.file_manifest_sha256, 'retained file-manifest hash mismatch');
  const manifest = parse(manifestBytes);
  require(Array.isArray(manifest.files) && manifest.files.length <= 10000, 'bounded retained file manifest required');
  const rows = manifest.files.map(row => relativePin(row, attempt, profile));
  require(new Set(rows.map(row => row.path)).size === rows.length, 'duplicate retained manifest member');
  const expected = [...rows.map(row => row.path), 'final.json', 'file-manifest.json'].sort();
  require(JSON.stringify(filesBelow(attempt)) === JSON.stringify(expected), 'missing or extraneous retained study members');
  const admitted = admitFiles(rows.map(row => row.absolute), {root: attempt});
  for (let i = 0; i < rows.length; i++) require(admitted[i].bytes === rows[i].bytes && hashAdmitted(admitted[i]) === rows[i].sha256,
    `retained member changed: ${rows[i].path}`);
  const inventory = new Map(rows.map(row => [row.path, row]));
  const admissions = new Map(rows.map((row, index) => [row.path, admitted[index]]));
  // Every later read retains its ORIGINAL admission metadata. Re-admitting a
  // restored file would miss a changed-and-restored record used mid-verification.
  const boundRead = name => {
    safeRelative(name);
    require(admissions.has(name), `unbound retained member: ${name}`);
    const bytes = readAdmitted(admissions.get(name));
    require(sha256(bytes) === inventory.get(name).sha256, `retained member changed: ${name}`);
    return bytes;
  };
  const requiredPin = (name, expectedHash) => {
    require(inventory.has(name) && inventory.get(name).sha256 === expectedHash, `retained binding mismatch: ${name}`);
    return parse(boundRead(name));
  };
  const freeze = requiredPin('freeze.json', final.freeze_sha256);
  const audit = requiredPin('audit.json', final.audit_sha256);
  const seal = requiredPin(profile.seal, final[profile.seal_key]);
  require(seal.freeze_sha256 === final.freeze_sha256 && seal.assessment_opened === false, 'model seal/freeze binding mismatch');
  const configuration = requiredPin('configuration.json', freeze.configuration_sha256);
  requiredPin('permissions.json', freeze.permissions_sha256);
  require(configuration.study_id === final.study_id && freeze.study_id === final.study_id &&
    configuration.claim_level === 'controlled_model_response' && freeze.assessment_opened === false && audit.valid === true,
    'retained study identity, claim or audit mismatch');
  require(Array.isArray(freeze.sources) && sha256(jsonBytes(freeze.sources)) === freeze.source_id, 'retained source identity mismatch');
  for (const source of freeze.sources) {
    safeRelative(source.path);
    const row = inventory.get(`source/${source.path}`);
    require(row && row.bytes === source.bytes && row.sha256 === source.sha256, `source capsule mismatch: ${source.path}`);
  }
  const sources = rows.filter(row => row.path.startsWith('source/'));
  require(sources.length === freeze.sources.length, 'extraneous source capsule members');
  const sealed = profile.contribution ? [...seal.models, ...seal.training] : seal.models_and_fit_inputs;
  require(Array.isArray(sealed), 'retained model/input seals required');
  for (const saved of sealed) {
    const row = relativePin(saved, attempt, profile), present = inventory.get(row.path);
    require(present && present.bytes === row.bytes && present.sha256 === row.sha256, `sealed fit/input binding mismatch: ${row.path}`);
  }
  const opened = inventory.has('assessment-open.json') && parse(boundRead('assessment-open.json'));
  require(opened && opened.freeze_sha256 === final.freeze_sha256 &&
    (opened.model_seal_sha256 ?? opened.seal_sha256) === final[profile.seal_key], 'assessment opening/seal mismatch');
  const recheck = () => {
    require(JSON.stringify(filesBelow(attempt)) === JSON.stringify(expected), 'missing or extraneous retained study members');
    require(hashAdmitted(finalAdmission) === finalPin.sha256 && hashAdmitted(manifestAdmission) === manifestPin.sha256,
      'retained final or manifest changed during verification');
    for (let i = 0; i < rows.length; i++) require(hashAdmitted(admitted[i]) === rows[i].sha256,
      `retained member changed: ${rows[i].path}`);
  };
  return {final, freeze, audit, configuration, inventory, finalPin, manifestPin,
    files_verified: rows.length + 2, read: boundRead, recheck};
}

const ORIGINAL_HEADER = ['arm','unit','change_num','a_num','b_num','alt_a_num','state','reason','recorded_base','recorded_alt','delta','z_base','z_alt'];
const CONTRIBUTION_HEADER = ['axis','arm','unit','change_num','a_num','b_num','state','reason','base_prediction','alternative_prediction','delta','truth','z_base','z_alt'];
const INPUT_HEADER = ['unit','a_num','b_num','noise_sign'];
const states = () => ({requested: 0, succeeded: 0, abstained: 0, failed: 0, unresolved: 0});
const publicPin = row => ({path: row.path, bytes: row.bytes, sha256: row.sha256});
function addState(counts, state) { counts.requested++; counts[state === 'success' ? 'succeeded' : state]++; }
const sameNumber = (a, b) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));

function parsePartition(attempt, bound, profile, seed, split) {
  const config = bound.configuration, file = profile.contribution ? `seed-${seed}/${split}-pairs.tsv` : `seed-${seed}/${split}/pairs.tsv`;
  require(bound.inventory.has(file), `missing retained fixture: ${file}`);
  const sourceFile = `seed-${seed}/${split}.tsv`;
  require(bound.inventory.has(sourceFile), `missing retained source population: ${sourceFile}`);
  const sources = table(bound.read(sourceFile), INPUT_HEADER, sourceFile);
  require(sources.length === config.counts[split], 'retained source population count mismatch');
  const sourceMap = new Map(sources.map(row => [row.unit, row]));
  require(sourceMap.size === sources.length, 'duplicate retained source unit');
  const rows = table(bound.read(file), profile.contribution ? CONTRIBUTION_HEADER : ORIGINAL_HEADER, file);
  const axes = profile.contribution ? config.axes : ['polynomial'];
  require(rows.length === axes.length * config.arms.length * sources.length * config.changes.length, 'incomplete retained pair roster');
  let index = 0;
  for (const axis of axes) for (const arm of config.arms) for (const source of sources) for (const change of config.changes) {
    const row = rows[index++]; row.axis ??= 'polynomial';
    require(row.axis === axis && row.arm === arm && row.unit === source.unit && integer(row.change_num, 'change') === change &&
      row.a_num === source.a_num && row.b_num === source.b_num, `retained pair identity/order mismatch: ${file}/${index}`);
    const a = integer(row.a_num, 'a'), b = integer(row.b_num, 'b');
    require(config.denominator === 4096 && Math.abs(a) <= 3072 && Math.abs(b) <= 4096, 'unsupported retained source bounds');
    row.alternative_a_num = a + change;
    if (!profile.contribution) require(integer(row.alt_a_num, 'alternative a') === row.alternative_a_num, 'retained alternative input mismatch');
    row.saved_state = row.state === 'succeeded' ? 'success' : row.state;
    require(['success','abstained','failed','unresolved'].includes(row.saved_state), 'unsupported retained pair state');
    const supported = Math.abs(row.alternative_a_num) <= 4096;
    require(row.saved_state === (supported ? 'success' : 'abstained') && row.reason ===
      (supported ? 'NONE' : profile.contribution ? 'out_of_support' : 'OUTSIDE_DECLARED_SUPPORT'), 'retained support/state/reason mismatch');
    row.pair_id = `${bound.final.study_id}/${seed}/${split}/${axis}/${arm}/${row.unit}/change/${change}`;
    row.base_text = row.base_prediction ?? row.recorded_base; row.alt_text = row.alternative_prediction ?? row.recorded_alt;
    row.executable = row.base_text !== 'NA' && row.alt_text !== 'NA' && row.z_base !== 'NA' && row.z_alt !== 'NA';
    if (!row.executable) require(!supported && profile.contribution &&
      ['base_prediction','alternative_prediction','delta','truth','z_base','z_alt'].every(key => row[key] === 'NA'), 'missing retained executable records');
    else {
      row.base = number(row.base_text, 'base response'); row.alternative = number(row.alt_text, 'alternative response');
      row.base_z = row.z_base.split(',').map(value => number(value, 'base representation'));
      row.alternative_z = row.z_alt.split(',').map(value => number(value, 'alternative representation'));
      require(row.base_z.length === row.alternative_z.length && row.base_z.length > 0 && row.base_z.length <= 1024, 'representation shape mismatch');
      if (supported) require(sameNumber(number(row.delta, 'saved contrast'), row.alternative - row.base), 'retained subtraction mismatch');
      else require(row.delta === 'NA', 'abstention has numerical contrast');
      if (profile.contribution) row.saved_truth = number(row.truth, 'saved oracle truth');
    }
  }
  return {file, sourceFile, rows, sources, sourceMap};
}

function makeBundle({bound, profile, seed, split, axis, arm, rows, scratch, tablePin, sourcePin, bridgePin, bridgeBytes}) {
  const prefix = `${bound.final.study_id}/${seed}/${split}/${axis}/${arm}`;
  const artifacts = new Map();
  const artifact = (name, bytes) => {
    if (!Buffer.isBuffer(bytes)) bytes = jsonBytes(bytes);
    const ref = {id: `${prefix}/${name}`, sha256: sha256(bytes)};
    if (artifacts.has(ref.id)) require(artifacts.get(ref.id).ref.sha256 === ref.sha256, `contradictory adapter artifact: ${ref.id}`);
    else artifacts.set(ref.id, {ref, bytes});
    return ref;
  };
  const context = {schema: 'mityane.hx06.retained-response-context/v1', study_id: bound.final.study_id,
    final_sha256: bound.finalPin.sha256, freeze_sha256: bound.final.freeze_sha256, source_id: bound.freeze.source_id,
    configuration_sha256: bound.freeze.configuration_sha256, table: tablePin, source_population: sourcePin,
    model: publicPin(bound.inventory.get(`seed-${seed}/model.tsv`)), seed, split, axis, arm,
    scope: 'Adapter for already recorded synthetic responses; no provider, preprocessing, fitting or model execution is asserted by this verification'};
  // Records are new adapter identities. Original identities remain in context;
  // these descriptors do not pretend to be original study execution receipts.
  const contextRef = artifact('retained-context', context);
  const model = artifact('retained-model', bound.read(`seed-${seed}/model.tsv`));
  const provider = artifact('recorded-provider', {context: contextRef, provider: arm, revision: 'retained-synthetic-table-v1', execution: false});
  const preprocessing = artifact('recorded-preprocessing', {context: contextRef, model, semantics: 'Raw feature coordinates are retained; fitted standardization belongs to the sealed readout. No transformation is executed.'});
  const constructor = artifact('verification-constructor', {revision: 'retained-study-verification-v1', context: contextRef,
    operation: 'Retain saved a alternative numerator; hold saved b numerator fixed. No new alternative is evaluated upstream.'});
  const supportPolicy = artifact('support-policy', {context: contextRef, minimum: -1, maximum: 1, held_fixed: ['b'], source: 'Frozen synthetic support declarations'});
  const operationSource = artifact('verification-source', bridgeBytes);
  const configuration = artifact('configuration', {context: contextRef, bridge: bridgePin, claim_level: 'controlled_model_response'});
  const permissions = artifact('verification-permissions', {context: contextRef,
    user_request: 'Implement verification of retained study records; do not fit, encode or regenerate study data',
    permitted: 'Read exposed pinned records and compare saved responses with the current pure comparison tool'});
  const dependencies = [['provider',provider],['preprocessing',preprocessing],['fitted_transform',model],['response_model',model],
    ['constructor',constructor],['support_policy',supportPolicy],['operation_source',operationSource],
    ['operation_configuration',configuration],['permissions_record',permissions],['retained_context',contextRef]].map(([role, ref]) => ({role, artifact: ref,
      fit_population: ['fitted_transform','response_model'].includes(role) ? `${bound.final.study_id}/${seed}/training` : 'not_fitted',
      selection_population: 'not_fitted', pretraining_exposure: 'excluded_by_design', available_at: null}));
  const shape = rows[0].base_z.length;
  const spaceRef = artifact('space', {context: contextRef, provider, preprocessing, fitted_transform: model, native_shape: [shape],
    normalization: 'none', geometry: 'retained_synthetic_feature_coordinates'});
  const space = {artifact: spaceRef, provider, preprocessing, fitted_transform: model, output_selector: 'saved-raw-feature-vector',
    native_shape: [shape], layout: 'dense', normalization: 'none', geometry: 'retained_synthetic_feature_coordinates', input_role: 'response_input'};
  const pipelineRef = artifact('pipeline', {context: contextRef, space: spaceRef, response_model: model, dependencies, configuration});
  const pipeline = {artifact: pipelineRef, response_model: model, configuration_sha256: configuration.sha256,
    stochastic_policy: 'deterministic', provider_revision_known: true, dependencies};
  const axes = [{id: axis, definition_revision: `${bound.final.study_id}/1.0.0/${axis}`, units: 'synthetic_response_unit'}];
  const mask = artifact('mask', {selected_output_position: 0, valid: true, source: 'Finite retained scalar response'});
  const evidence = artifact('recorded-controls', {context: contextRef, constructor, support_policy: supportPolicy,
    verification: 'Numerator pairing and saved scalar/vector records checked against the byte-pinned full roster; scientific validity is inherited, not re-established'});
  const checked = (state = 'passed') => ({state, evidence, detail: 'Retained static synthetic record; declared control and support verified without upstream execution'});
  const pairs = rows.map(row => {
    const id = `${row.unit}/change/${row.change_num}`;
    const observation = artifact(`${row.unit}/source`, {source_population: sourcePin, unit: row.unit, a_num: Number(row.a_num), b_num: Number(row.b_num)});
    const baseInput = artifact(`${row.unit}/input`, {unit: row.unit, a_num: Number(row.a_num), b_num: Number(row.b_num), source: observation});
    const alternativeInput = artifact(`${id}/input`, {unit: row.unit, a_num: row.alternative_a_num, b_num: Number(row.b_num), parent: baseInput, constructor});
    const baseRepresentation = artifact(`${row.unit}/representation`, {space: spaceRef, values: row.base_z});
    const alternativeRepresentation = artifact(`${id}/representation`, {space: spaceRef, values: row.alternative_z, parent: baseRepresentation});
    const branch = (alternative) => {
      const response = artifact(`${alternative ? id : row.unit}/response`, {axes, values: [alternative ? row.alternative : row.base]});
      return {input: alternative ? alternativeInput : baseInput, source_observation: observation,
        parent_input_id: alternative ? baseInput.id : '', origin: alternative ? 'edited_input' : 'observed',
        representation: alternative ? alternativeRepresentation : baseRepresentation,
        representation_kind: alternative ? 'transformed_candidate' : 'observation_embedding',
        parent_representation_id: alternative ? baseRepresentation.id : '', space, pipeline: pipelineRef, response,
        response_execution_id: `${prefix}/${alternative ? id : row.unit}/retained-response`, output_position: 0, mask,
        output_valid: true, upstream_succeeded: true,
        timing: {support_start: null, support_end: null, available_at: null, vintage: 'already-exposed-retained-synthetic-records'},
        attributes: [{name: 'a', units: 'synthetic_coordinate', value: (alternative ? row.alternative_a_num : Number(row.a_num)) / 4096},
          {name: 'b', units: 'synthetic_coordinate', value: Number(row.b_num) / 4096}], axes,
        response_values: [alternative ? row.alternative : row.base]};
    };
    return {id: row.pair_id, unit_id: `${seed}/${row.unit}`, entity_id: `${seed}/${row.unit}`, group_id: `${seed}/${split}`,
      baseline: branch(false), alternative: branch(true), constructor,
      change_definition: `Saved a_num change ${row.change_num}; b_num held fixed`, interpretation: 'model_input_response',
      control_evidence: evidence, alternative_validity: checked(), support: checked(row.saved_state === 'abstained' ? 'rejected' : 'passed'), interpretability: checked()};
  });
  const input = {request_id: `${prefix}/verification-request`, operation_run_id: `${prefix}/verification-run`,
    contract_id: 'OR06-PCR', contract_revision: '0.1.0', rer_id: 'OR06-PCR-RER', rer_revision: '0.1.0',
    claim_level: 'controlled_model_response', purpose: 'Mechanical verification of exposed retained synthetic responses; no fresh scientific assessment',
    population_id: `${bound.final.study_id}/${seed}/${split}`, sampling_rule: 'Every retained declared pair in this fixed family/axis partition',
    response_horizon: 'static', permitted_interpretation: 'model_input_response', permissions_ref: permissions.id,
    resource_budget: 'CPU recorded subtraction only; no providers, fitting, data generation or bootstrap', evidence_destination: 'caller-selected verification record',
    code_sha256: operationSource.sha256, configuration_sha256: configuration.sha256, temporal_mode: 'non_temporal', as_of: null,
    non_temporal_justification: 'Frozen project-authored static synthetic coordinates and responses', expected_space: space,
    expected_pipeline: pipeline, response_axes: axes, selected_output_position: 0,
    attribute_rules: [{name: 'a', units: 'synthetic_coordinate', role: 'changed', minimum: -1, maximum: 1},
      {name: 'b', units: 'synthetic_coordinate', role: 'held_fixed', minimum: -1, maximum: 1}], control_tolerance: 0, pairs};
  const directory = path.join(scratch.directory, `${seed}-${split}-${axis}-${arm}`);
  fs.mkdirSync(directory); fs.mkdirSync(path.join(directory, 'artifacts'));
  const inventory = [];
  const expected = [];
  for (const [index, {ref, bytes}] of [...artifacts.values()].entries()) {
    const relative = `artifacts/${index}.bin`; fs.writeFileSync(path.join(directory, relative), bytes, {flag: 'wx', mode: 0o600});
    inventory.push({...ref, path: relative});
    expected.push({path: relative, bytes: bytes.length, sha256: ref.sha256});
  }
  const envelope = {schema_version: 'mityane.operation.request/v1', operation: {id: 'OR06-PCR', revision: '0.1.0'}, input, artifacts: inventory};
  const request = path.join(directory, 'request.json'), requestBytes = jsonBytes(envelope);
  fs.writeFileSync(request, requestBytes, {flag: 'wx', mode: 0o600});
  expected.push({path: 'request.json', bytes: requestBytes.length, sha256: sha256(requestBytes)});
  return {request, envelope, recheck: bindVerificationScratch(directory, expected),
    artifacts: [...artifacts.values()].map(({ref, bytes}) => ({...ref, payload_base64: bytes.toString('base64')}))};
}

export const verifyStudyOptionsSchema = {$schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:verify-study:options:1', type: 'object', additionalProperties: false,
  required: ['attempt','output'], properties: {attempt: {type: 'string'}, output: {type: 'string'},
    study: {enum: ['synthetic-response-v1','representation-contribution-v1']}}};
const countsSchema = {type: 'object', additionalProperties: false,
  required: ['requested','succeeded','abstained','failed','unresolved'], properties: Object.fromEntries(
    ['requested','succeeded','abstained','failed','unresolved'].map(key => [key, {type: 'integer', minimum: 0}]))};
const savedMetricSchema = {type: 'object', additionalProperties: false,
  required: ['source_units','source_unit_mae','aggregation'], properties: {source_units: {type: 'integer', minimum: 1},
    source_unit_mae: {type: 'number', minimum: 0}, aggregation: {type: 'string'}}};
export const verifyStudyResultSchema = {$schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:verify-study:result:1', type: 'object', additionalProperties: false,
  description: 'Public command/API receipt. report_file identifies the richer verification.json evidence record, whose source counts distinguish current checks from retained-only abstentions.',
  required: ['schema_version','status','study_id','report_file','summary','current_check_counts','retained_only_count',
    'recomputed_saved_truth_metrics','archived_metrics','archived_metrics_note'],
  properties: {schema_version: {const: 'mityane.hx06.retained-study-verification/v1'}, status: {enum: ['completed','partial']},
    study_id: {enum: ['OR06-SYNTH-RESP-001','OR06-REP-CONTRIB-001']}, report_file: {type: 'string'}, summary: countsSchema,
    current_check_counts: countsSchema, retained_only_count: {type: 'integer', minimum: 0},
    recomputed_saved_truth_metrics: {type: 'object', additionalProperties: savedMetricSchema},
    archived_metrics: {type: 'object', description: 'Pinned original audit pooled statistics, including archived uncertainty; never presented as a new audit.'},
    archived_metrics_note: {type: 'string'}}};

export async function runVerifyStudy({attempt, output, study, operations}) {
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv') && ROOT === '/mityane', 'Use the managed Linux container');
  require(typeof attempt === 'string' && attempt.length > 0, 'explicit retained attempt directory required');
  attempt = path.resolve(ROOT, attempt);
  const final = json(path.join(attempt, 'final.json')), profile = PROFILES[final.study_id];
  require(profile && (study === undefined || study === profile.name), 'unsupported retained study identity');
  require(typeof output === 'string' && path.isAbsolute(output) && path.normalize(output) === output &&
    inside(path.join(ROOT, '.temp/operations/counterfactuals'), output), 'new operation-local verification output required');
  require(!inside(attempt, output) && !inside(output, attempt), 'verification output must be separate from retained attempt');
  const bound = verifyRetainedStudyFiles(attempt, profile);
  const bridgeAdmission = admitFiles([SELF])[0], bridgeBytes = readAdmitted(bridgeAdmission);
  const bridgePin = {path: SELF, bytes: bridgeBytes.length, sha256: sha256(bridgeBytes)};
  const sourceCounts = states(), currentCounts = states(), retainedOnly = [], partitions = [], metrics = {};
  const scratchRoot = '/opt/mityane-geometrics/operations/counterfactuals/retained-verification';
  fs.mkdirSync(scratchRoot, {recursive: true}); canonicalDirectory(scratchRoot);
  const scratch = {directory: createExclusiveDirectory(path.join(scratchRoot, randomUUID())), attempt};
  createExclusiveDirectory(output, {root: ROOT});
  const published = [];
  const publish = (name, bytes) => {
    const expected = {path: name, bytes: bytes.length, sha256: sha256(bytes)};
    writeNew(path.join(output, name), bytes);
    const actual = pin(path.join(output, name), output);
    require(actual.bytes === expected.bytes && actual.sha256 === expected.sha256, `published verification bytes changed: ${name}`);
    published.push(expected);
  };
  try {
    for (const seed of bound.configuration.seeds) for (const split of ['development','assessment']) {
      const partition = parsePartition(attempt, bound, profile, seed, split);
      const tablePin = bound.inventory.get(partition.file), sourcePin = bound.inventory.get(partition.sourceFile);
      for (const row of partition.rows) {
        addState(sourceCounts, row.saved_state);
        if (!row.executable) retainedOnly.push({pair_id: row.pair_id, state: row.saved_state, reason: row.reason,
          delta: null, current_check: 'not_executable_missing_retained_upstream_records', source_file: partition.file});
        if (profile.contribution && row.saved_state === 'success') {
          const key = `${split}/${row.axis}/${row.arm}`; metrics[key] ??= new Map();
          const unit = `${seed}/${row.unit}`; const errors = metrics[key].get(unit) ?? [];
          errors.push(Math.abs((row.alternative - row.base) - row.saved_truth)); metrics[key].set(unit, errors);
        }
      }
      const axes = profile.contribution ? bound.configuration.axes : ['polynomial'];
      for (const axis of axes) for (const arm of bound.configuration.arms) {
        const rows = partition.rows.filter(row => row.axis === axis && row.arm === arm && row.executable);
        require(rows.length > 0 && rows.length <= 10000, 'unsupported bounded retained partition');
        const bundle = makeBundle({bound, profile, seed, split, axis, arm, rows, scratch, tablePin: publicPin(tablePin),
          sourcePin: publicPin(sourcePin), bridgePin, bridgeBytes});
        bundle.recheck();
        const checked = await operations.check(bundle.request);
        bundle.recheck();
        require(checked.result.admitted === true && checked.result.pairs.length === rows.length, 'current tool rejected retained verification adapter');
        for (let i = 0; i < rows.length; i++) {
          const old = rows[i], current = checked.result.pairs[i];
          require(current.pair_id === old.pair_id && current.state === old.saved_state, `current/retained pair state mismatch: ${old.pair_id}`);
          if (old.saved_state === 'success') require(current.contrast && sameNumber(current.contrast.delta[0], number(old.delta, 'saved delta')),
            `current/retained contrast mismatch: ${old.pair_id}`);
          else require(current.contrast === null && current.diagnostics.some(row => row.code === 'OUTSIDE_DECLARED_SUPPORT'), 'current abstention reason mismatch');
          addState(currentCounts, current.state);
        }
        const name = `${seed}-${split}-${axis}-${arm}.json.gz`;
        publish(name, gzipSync(jsonBytes({schema_version: 'mityane.hx06.checked-record-archive/v1',
          note: 'A checked request plus exact artifact payloads and check output; this is not a run result bundle. No model execution.',
          request: bundle.envelope, artifacts: bundle.artifacts, check: checked}), {level: 6}));
        partitions.push({seed, split, axis, arm, archive: name, current_check_counts: checked.summary});
      }
    }
    const recomputed = Object.fromEntries(Object.entries(metrics).map(([key, units]) => [key, {
      source_units: units.size, source_unit_mae: [...units.values()].reduce((sum, values) => sum + values.reduce((a,b) => a+b, 0) / values.length, 0) / units.size,
      aggregation: 'Mean absolute error across supported alternatives within source unit, then mean across all equal-size seed populations; saved truth only'}]));
    if (profile.contribution) for (const axis of bound.configuration.axes) for (const arm of bound.configuration.arms) {
      const archived = bound.audit.metrics.pooled[axis].all.families[arm].source_unit_mae;
      require(sameNumber(recomputed[`assessment/${axis}/${arm}`].source_unit_mae, archived), 'recomputed saved-truth MAE differs from archived audit');
    }
    // Keep the original admissions through the final check, including metadata
    // signatures, rather than accepting restored files under fresh admissions.
    bound.recheck();
    require(hashAdmitted(bridgeAdmission) === bridgePin.sha256, 'verification adapter changed during invocation');
    publish('retained-only.json', jsonBytes({rows: retainedOnly}));
    const report = {schema_version: 'mityane.hx06.retained-study-verification/v1',
      status: sourceCounts.failed || sourceCounts.unresolved ? 'partial' : 'completed', study_id: bound.final.study_id,
      attempt, final_pin: bound.finalPin, file_manifest_pin: bound.manifestPin, files_verified: bound.files_verified,
      bridge: bridgePin, source_counts: sourceCounts, current_check_counts: currentCounts,
      retained_only_count: retainedOnly.length, retained_only_file: 'retained-only.json', partitions,
      original_verdict: bound.final.verdict, recomputed_saved_truth_metrics: recomputed,
      archived_metrics: bound.audit.metrics, archived_metrics_source: {path: 'audit.json', sha256: bound.final.audit_sha256,
        note: profile.contribution ? 'Intervals and breakdowns are archived; only scalar saved-truth MAE is recomputed here.' :
          'Original pairs do not retain per-pair oracle truth. These MAE statistics are verified archived audit values, not recomputed MAE.'},
      restrictions: ['Already exposed records; no fresh assessment or new scientific evidence',
        'No study runner, auditor, generator, encoder, model, fit or bootstrap executes',
        'New verification adapter identities retain original pins; they do not replace historical source or execution identities',
        'Current check covers every retained executable pair. Rows without recorded upstream values remain explicit retained-only abstentions with null contrasts.',
        'Only controlled_model_response; no observational or causal claims'], scratch_directory: scratch.directory};
    publish('verification.json', jsonBytes(report));
    writeNew(path.join(output, 'manifest.json'), jsonBytes({schema_version: 'mityane.hx06.verification-files/v1', files: published}));
    return {schema_version: report.schema_version, status: report.status, study_id: report.study_id,
      report_file: path.join(output, 'verification.json'), summary: sourceCounts,
      current_check_counts: currentCounts, retained_only_count: retainedOnly.length,
      recomputed_saved_truth_metrics: recomputed, archived_metrics: bound.audit.metrics.pooled,
      archived_metrics_note: report.archived_metrics_source.note};
  } catch (error) {
    try { writeNew(path.join(output, 'failure.json'), jsonBytes({status: 'verification_failed', message: error.message,
      published_files: published, scratch_directory: scratch.directory})); } catch {}
    throw error;
  }
}
