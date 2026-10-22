// Copyright (c) 2026 Waajacu. Importer code: repository MIT license.
// The separately retained wooldridge data and documentation declare GPL-3.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {canonicalDirectory} from '../../../../shared/artifacts/node/v1/files.mjs';

const directory = path.dirname(fileURLToPath(import.meta.url));
export const LOCK = JSON.parse(fs.readFileSync(path.join(directory, 'source-lock.json'), 'utf8'));
const requireValue = (value, message) => { if (!value) throw new Error(message); };
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const pin = (locator, bytes) => ({path: locator, bytes: bytes.length, sha256: digest(bytes)});
const relative = (root, filename) => path.relative(root, filename).split(path.sep).join('/');

/** Strict bounded CSV reader; no evaluation, imputation or type coercion. */
export function parseCSV(bytes) {
  requireValue(Buffer.isBuffer(bytes) && bytes.length > 0 && bytes.length <= 1000000, 'bounded nonempty CSV bytes required');
  const source = new TextDecoder('utf-8', {fatal: true}).decode(bytes);
  requireValue(!source.includes('\0') && !source.includes('\uFEFF'), 'NUL/BOM are not part of the declared CSV serialization');
  const records = [], row = [];
  let field = '', quoted = false, afterQuote = false, atStart = true;
  const cell = () => { row.push(field); field = ''; afterQuote = false; atStart = true; };
  const record = () => { cell(); records.push(row.splice(0)); };
  for (let index = 0; index < source.length; ++index) {
    const char = source[index];
    if (quoted) {
      if (char === '"' && source[index + 1] === '"') { field += '"'; ++index; }
      else if (char === '"') { quoted = false; afterQuote = true; }
      else { requireValue(char !== '\r' && char !== '\n', 'multiline CSV fields are unsupported'); field += char; }
    } else if (char === ',' ) cell();
    else if (char === '\n') record();
    else if (char === '\r') { requireValue(source[++index] === '\n', 'lone CR is unsupported'); record(); }
    else if (char === '"') { requireValue(atStart && !afterQuote, 'misplaced CSV quote'); quoted = true; atStart = false; }
    else { requireValue(!afterQuote, 'characters after closing CSV quote'); field += char; atStart = false; }
    requireValue(records.length <= 1000 && row.length <= 30 && field.length <= 1000, 'CSV shape/field bound exceeded');
  }
  requireValue(!quoted, 'unterminated CSV quote');
  if (field.length || row.length || afterQuote) record();
  requireValue(records.length > 1, 'CSV header and data required');
  return records;
}

export function validateTable(bytes, expected = LOCK) {
  const [fields, ...rows] = parseCSV(bytes);
  requireValue(JSON.stringify(fields) === JSON.stringify(expected.fields), 'exact nineteen-column source schema/order required');
  requireValue(rows.length === expected.rows, 'fixed source population row count differs');
  const assignment = fields.indexOf(expected.assignment_field);
  const baseline = fields.indexOf('re74'), outcome = fields.indexOf(expected.outcome_field);
  requireValue(assignment >= 0 && baseline >= 0 && outcome >= 0, 'source role fields missing');
  requireValue(JSON.stringify(expected.approved_matching_fields) === '["re74"]', 'only re74 is admitted for matching');
  const counts = {'0': 0, '1': 0};
  const decimal = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/;
  for (const values of rows) {
    requireValue(values.length === fields.length, 'ragged CSV row');
    for (let column = 0; column < values.length; ++column) {
      requireValue(values[column] === 'NA' || (decimal.test(values[column]) && Number.isFinite(Number(values[column]))), 'only finite numeric source values or explicit NA allowed');
    }
    requireValue(['0', '1'].includes(values[assignment]), 'assignment must be the literal binary source indicator');
    requireValue(values[baseline] !== 'NA' && values[outcome] !== 'NA', 're74 and re78 must be observed finite source values');
    requireValue(Number(values[baseline]) >= 0 && Number(values[outcome]) >= 0, 'source earnings cannot be negative');
    ++counts[values[assignment]];
  }
  requireValue(JSON.stringify(counts) === JSON.stringify(expected.assignment_counts), 'declared 260-control/185-assigned roster differs');
  return {fields, rows: rows.length, assignment_counts: counts, approved_matching_fields: ['re74'],
    assignment_field: expected.assignment_field, outcome_field: expected.outcome_field,
    outcome_values_aggregated: false, matching_executed: false, effect_estimated: false};
}

export function validateMetadata(bytes, expected = LOCK) {
  const value = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes));
  requireValue(value.Package === expected.package && value.Version === expected.package_version
    && value.RemoteSha === expected.source_commit && value.License === expected.publisher_license,
  'package identity, source revision, version or license differs');
  const dataset = value._datasets?.find(item => item.name === 'jtrain2');
  requireValue(dataset?.rows === expected.rows && JSON.stringify(dataset.fields) === JSON.stringify(expected.fields),
    'publisher dataset metadata differs from fixed schema/population');
  return {package: value.Package, version: value.Version, source_commit: value.RemoteSha, license: value.License,
    dataset: {name: dataset.name, rows: dataset.rows, fields: dataset.fields}, source_rdata_decoded: false};
}

const origins = new Set(['https://justinmshea.r-universe.dev', 'https://raw.githubusercontent.com',
  'https://users.nber.org', 'https://www.gnu.org', 'https://r2.ropensci.org']);
async function download(asset) {
  let url = asset.url;
  for (let redirects = 0; redirects < 5; ++redirects) {
    requireValue(origins.has(new URL(url).origin), 'download redirect left declared publisher origins');
    const response = await fetch(url, {redirect: 'manual', signal: AbortSignal.timeout(30000),
      headers: {'user-agent': 'Mityane-geometrics NSW archival-reference importer/1.0'}});
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      requireValue(response.headers.has('location'), 'redirect has no location');
      await response.body?.cancel(); url = new URL(response.headers.get('location'), url).href; continue;
    }
    requireValue(response.status === 200 && response.body, `publisher download failed: ${asset.role} HTTP ${response.status}`);
    const pieces = []; let length = 0;
    for await (const part of response.body) {
      length += part.length; requireValue(length <= asset.max_bytes, `publisher asset exceeds bound: ${asset.role}`); pieces.push(part);
    }
    const bytes = Buffer.concat(pieces);
    requireValue(bytes.length > 0, 'publisher returned empty asset');
    if (asset.sha256) requireValue(digest(bytes) === asset.sha256, `publisher artifact changed: ${asset.role}`);
    return {bytes, final_url: url, content_type: response.headers.get('content-type'), retrieved_utc: new Date().toISOString()};
  }
  throw new Error('publisher exceeded redirect limit');
}

export async function importReference(output) {
  requireValue(process.platform === 'linux' && fs.existsSync('/.dockerenv') && process.cwd() === '/mityane',
    'run through the existing managed Linux container at /mityane');
  const root = canonicalDirectory('/mityane');
  requireValue(/^\.temp\/data\/hx06-nsw-reference-v1\/import-\d{3}$/.test(output), 'new named NSW import leaf required');
  const destination = path.join(root, output), parent = path.dirname(destination);
  const admissionPath = path.join(parent, 'data-admission-001.json');
  const admissionBytes = fs.readFileSync(admissionPath);
  const admission = JSON.parse(admissionBytes);
  requireValue(admission.kind === 'hx06_nsw_data_only_admission_v1' && admission.authorized === true
    && admission.effect_estimation_authorized === false && admission.matching_authorized === false,
  'explicit data-only import/materialization admission required');
  // Check every ancestor before recursive creation; existing leaves are preserved.
  for (let ancestor = parent; ancestor !== root; ancestor = path.dirname(ancestor)) {
    if (fs.existsSync(ancestor)) canonicalDirectory(ancestor, {root});
  }
  fs.mkdirSync(parent, {recursive: true}); canonicalDirectory(parent, {root});
  fs.mkdirSync(destination); canonicalDirectory(destination, {root});
  const members = [], content = new Map();
  const sourceFiles = ['import.mjs', 'source-lock.json'];
  const closure = sourceFiles.map(file => {
    const absolute = path.join(directory, file), bytes = fs.readFileSync(absolute);
    fs.writeFileSync(path.join(destination, `input-${file}`), bytes, {flag: 'wx'});
    return pin(relative(root, absolute), bytes);
  });
  try {
    for (const asset of LOCK.assets) {
      const value = await download(asset);
      const filename = path.join(destination, asset.file);
      fs.writeFileSync(filename, value.bytes, {flag: 'wx'}); content.set(asset.role, value.bytes);
      members.push({role: asset.role, requested_url: asset.url, final_url: value.final_url,
        retrieved_utc: value.retrieved_utc, content_type: value.content_type,
        file: {...pin(relative(root, filename), value.bytes), container_path: filename,
          host_path: `C:\\Work\\Mityane-geometrics\\${relative(root, filename).replaceAll('/', '\\')}`}});
    }
    const metadata = validateMetadata(content.get('publisher_package_metadata'));
    const description = content.get('package_description').toString('utf8');
    requireValue(/^Version: 1\.4-7$/m.test(description) && /^License: GPL-3$/m.test(description), 'pinned DESCRIPTION identity differs');
    const documentation = content.get('dataset_documentation').toString('utf8');
    requireValue(documentation.includes('445 observations on 19 variables') && documentation.includes('randomly assigned')
      && documentation.includes('real earns., 1974, $1000s') && documentation.includes('real earns., 1978, $1000s'), 'source assignment/schema/units documentation differs');
    const original = content.get('original_author_sample_documentation').toString('utf8');
    requireValue(original.includes('pre-intervention variables') && original.includes('260 observations') && original.includes('185 observations'), 'original author subset documentation differs');
    requireValue(content.get('gpl_license_text').toString('utf8').includes('GNU GENERAL PUBLIC LICENSE'), 'GPL notice is missing');
    const validation = validateTable(content.get('csv_serialization'));
    for (const file of closure) requireValue(digest(fs.readFileSync(path.join(root, file.path))) === file.sha256, 'import source changed during retrieval');
    const receipt = {kind: 'mityane_nsw_wooldridge_import_v1', version: '1.0.0', status: 'passed', passed: true,
      created_utc: new Date().toISOString(), dataset_id: LOCK.dataset_id, source: metadata, members, source_closure: closure,
      admission: pin(relative(root, admissionPath), admissionBytes),
      validation, license: {publisher_declaration: 'GPL-3', repository_code_license_is_not_data_license: true,
        retained_license_text: 'GPL-3.0.txt', attribution: 'Wooldridge data package, Justin M. Shea; R. J. LaLonde (1986), NSW Demonstration; Dehejia and Wahba archival subset documentation',
        raw_data_and_documentation_published_by_this_importer: false, use: 'Authorized local archival research reference; preserve publisher attribution and GPL terms'},
      limitations: ['The CSV is the exact retained R-universe serialization bound to package metadata; the separately retained commit-pinned RData is not decoded or asserted numerically equivalent by this importer.',
        'Widely published archival benchmark; not fresh, unexposed or protected assessment.',
        'Retrospective source documentation establishes year labels, not actual decision-time information availability or original assignment probabilities.',
        'Only re74 is admitted for matching. The remaining variables are retained source fields, not automatically admissible covariates.',
        'Counts and field validation are import checks, not matching quality, causal identification or an effect estimate.'],
      matching_executed: false, encoder_calls: 0, fitted_readouts: 0, effect_estimated: false, outcome_aggregates_computed: false};
    fs.writeFileSync(path.join(destination, 'import-receipt.json'), JSON.stringify(receipt, null, 2) + '\n', {flag: 'wx'});
    return receipt;
  } catch (error) {
    fs.writeFileSync(path.join(destination, 'import-failure.json'), JSON.stringify({kind: 'mityane_nsw_wooldridge_import_failure_v1',
      status: 'failed', created_utc: new Date().toISOString(), reason: error.message, members, source_closure: closure,
      retained_partial_leaf: true}, null, 2) + '\n', {flag: 'wx'});
    throw error;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    requireValue(process.argv.length === 4 && process.argv[2] === '--output', 'usage: import.mjs --output .temp/data/hx06-nsw-reference-v1/import-NNN');
    const receipt = await importReference(process.argv[3]);
    console.log(JSON.stringify({status: receipt.status, dataset_id: receipt.dataset_id, source: receipt.source,
      validation: receipt.validation, receipt: `${process.argv[3]}/import-receipt.json`, members: receipt.members.map(item => ({role: item.role, ...item.file}))}));
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
