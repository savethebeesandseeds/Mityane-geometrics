// Copyright (c) 2026 Waajacu. See repository LICENSE for importer code.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {canonicalDirectory, admitFiles, readAdmitted} from '../../../../shared/artifacts/node/v1/files.mjs';
import {LOCK, parseCSV, validateTable} from './import.mjs';

const directory = path.dirname(fileURLToPath(import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const requireValue = (value, message) => { if (!value) throw new Error(message); };
const relative = filename => path.relative('/mityane', filename).split(path.sep).join('/');
const pin = (filename, bytes) => ({path: relative(filename), bytes: bytes.length, sha256: digest(bytes)});
const read = filename => readAdmitted(admitFiles([filename], {root: '/mityane', singleLink: true})[0]);
const encoded = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');
const source = fileURLToPath(import.meta.url);

export function baselineRecords(bytes, namespace = LOCK.dataset_id, expected = LOCK) {
  validateTable(bytes, expected);
  const [fields, ...rows] = parseCSV(bytes), baseline = fields.indexOf('re74'), assignment = fields.indexOf('train');
  return rows.map((row, index) => ({unit_id: `${namespace}:source-row-${String(index + 1).padStart(4, '0')}`,
    source_row_ordinal: index + 1, assignment: Number(row[assignment]), values: [Number(row[baseline])]}));
}

export function materialize(importLeaf, output) {
  requireValue(process.platform === 'linux' && fs.existsSync('/.dockerenv') && process.cwd() === '/mityane', 'managed Linux container required');
  canonicalDirectory('/mityane');
  requireValue(/^\.temp\/data\/hx06-nsw-reference-v1\/import-\d{3}$/.test(importLeaf), 'exact named source import required');
  requireValue(/^\.temp\/data\/hx06-nsw-reference-v1\/binding-preparation-\d{3}$/.test(output), 'new named materialization leaf required');
  const input = path.join('/mityane', importLeaf), destination = path.join('/mityane', output);
  canonicalDirectory(input, {root: '/mityane'}); canonicalDirectory(path.dirname(destination), {root: '/mityane'});
  const receiptPath = path.join(input, 'import-receipt.json'), receiptBytes = read(receiptPath), receipt = JSON.parse(receiptBytes);
  requireValue(receipt.kind === 'mityane_nsw_wooldridge_import_v1' && receipt.passed === true
    && receipt.dataset_id === LOCK.dataset_id && receipt.source.source_commit === LOCK.source_commit, 'passed exact imported snapshot required');
  const csvPin = receipt.members.find(member => member.role === 'csv_serialization')?.file;
  requireValue(csvPin?.path === `${importLeaf}/jtrain2.csv`, 'source CSV binding differs');
  const csv = read(path.join('/mityane', csvPin.path));
  requireValue(csv.length === csvPin.bytes && digest(csv) === csvPin.sha256, 'source CSV bytes changed after import');
  const admissionPath = path.join(path.dirname(input), 'data-admission-001.json');
  const admissionBytes = read(admissionPath), admission = JSON.parse(admissionBytes);
  requireValue(digest(admissionBytes) === receipt.admission.sha256 && admission.authorized === true
    && admission.matching_authorized === false && admission.effect_estimation_authorized === false, 'same explicit data-only admission required');
  const definitionPath = path.join(directory, 'binding-definition.json'), definitionBytes = read(definitionPath), definition = JSON.parse(definitionBytes);
  requireValue(definition.provider_id === 'nsw-re74-identity/v1' && definition.input_field === 're74'
    && JSON.stringify(definition.native_shape) === '[1]' && definition.geometry.id === 'absolute_distance/v1', 'baseline-only definition changed');
  const records = baselineRecords(csv);
  fs.mkdirSync(destination); canonicalDirectory(destination, {root: '/mityane'});
  const written = [];
  const write = (name, value) => {
    const bytes = encoded(value), filename = path.join(destination, name);
    fs.writeFileSync(filename, bytes, {flag: 'wx'}); const result = pin(filename, bytes); written.push(result); return result;
  };
  try {
    const provider = write('provider-definition.json', definition);
    const space = write('spaceref.json', {kind: 'mityane_spaceref_v1', version: '1.0.0',
      space_id: `nsw-re74-identity/v1:${provider.sha256}`, provider_id: definition.provider_id, provider_artifact: provider,
      dataset_id: LOCK.dataset_id, representation_kind: definition.representation_kind, surface: definition.surface,
      native_shape: definition.native_shape, dtype: 'binary64', normalization: 'none', masks: definition.masks,
      geometry: definition.geometry, preprocessing: 'not_fitted', readout: 'none', timing: definition.timing,
      suitability_evidence: 'unassessed; descriptor binding only'});
    const snapshot = {path: csvPin.path, bytes: csvPin.bytes, sha256: csvPin.sha256};
    const roster = write('roster.json', {kind: 'mityane_nsw_snapshot_roster_v1', version: '1.0.0', dataset_id: LOCK.dataset_id,
      source_snapshot: snapshot, unit_identity: definition.unit_identity, assignment_definition: 'train=1 assigned to job training; train=0 control, as documented by wooldridge jtrain2',
      rows: records.map(({unit_id, source_row_ordinal, assignment}) => ({unit_id, source_row_ordinal, assignment}))});
    const representations = write('representations.json', {kind: 'mityane_nsw_re74_representations_v1', version: '1.0.0',
      dataset_id: LOCK.dataset_id, source_snapshot: snapshot, provider_artifact: provider, space_ref: space, roster,
      native_shape: [1], dtype: 'binary64', coordinate: {name: 're74', units: 'thousands of real dollars'},
      rows: records.map(({unit_id, source_row_ordinal, values}) => ({representation_id: `${unit_id}:nsw-re74-identity/v1`,
        observation_ref: {unit_id, source_snapshot_sha256: snapshot.sha256, source_row_ordinal}, values}))});
    const sourceClosure = ['materialize.mjs', 'import.mjs', 'source-lock.json', 'binding-definition.json'].map(name => {
      const filename = path.join(directory, name); return pin(filename, read(filename));
    });
    const result = {kind: 'mityane_nsw_baseline_materialization_v1', version: '1.0.0', status: 'passed', passed: true,
      created_utc: new Date().toISOString(), dataset_id: LOCK.dataset_id, import_receipt: pin(receiptPath, receiptBytes),
      admission: pin(admissionPath, admissionBytes), source_snapshot: snapshot, source_closure: sourceClosure,
      artifacts: {provider, space_ref: space, roster, representations}, rows: records.length,
      assignment_counts: receipt.validation.assignment_counts, outcome_definition: {field: 're78', source_snapshot: snapshot,
        units: 'thousands of real dollars', period: '1978', values_copied_to_derived_artifacts: false},
      available_at: null, as_of_availability_claim: false, matching_executed: false, effect_estimated: false,
      fitted_preprocessing: false, encoder_training: false, readout_fitting: false, outcome_aggregates_computed: false,
      limitations: ['Retrospective archival reference only; no protected-assessment claim.',
        'Row ordinals identify this exact source snapshot, not persistent subject identities.',
        'No balance, overlap, confounding, outcome or identification diagnostics were calculated.']};
    write('materialization-receipt.json', result);
    return result;
  } catch (error) {
    fs.writeFileSync(path.join(destination, 'materialization-failure.json'), encoded({status: 'failed', reason: error.message,
      created_utc: new Date().toISOString(), retained_partial_leaf: true, artifacts: written}), {flag: 'wx'}); throw error;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === source) {
  try {
    requireValue(process.argv.length === 6 && process.argv[2] === '--import' && process.argv[4] === '--output',
      'usage: materialize.mjs --import .temp/data/hx06-nsw-reference-v1/import-NNN --output .temp/data/hx06-nsw-reference-v1/binding-preparation-NNN');
    console.log(JSON.stringify(materialize(process.argv[3], process.argv[5])));
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
