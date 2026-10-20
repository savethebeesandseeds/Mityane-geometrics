// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Local non-production research preparation; publisher data/terms retain their
// separate license. No training, target generation, head fitting or scoring.
import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import {inflateRawSync} from 'node:zlib';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
import {admitFiles, canonicalDirectory, createExclusiveDirectory,
  jsonBytes, parseJSON, readAdmitted, requireCondition as require, safeRelative,
  sha256, writeNew} from '../../../../shared/artifacts/node/v1/files.mjs';

const DIRECTORY = 'code/data/market/experiments/cpb_v16_temporal_v1';
const SHARED = 'code/shared/artifacts/node/v1/files.mjs';
export const SOURCES = ['prepare.mjs', 'prepare.test.mjs', 'cache88.h', 'csv_to_cache88.cpp', 'converter_test.cpp', 'converter.mk', 'source-copy-provenance.json'].map(name => `${DIRECTORY}/${name}`).concat(SHARED);
export const SYMBOLS = Object.freeze(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']);
export const GEOMETRY = Object.freeze({month: '2024-03', interval: '1m', market: 'spot', rows: 44640,
  first_open_ms: 1709251200000, first_close_ms: 1709251259999, last_close_ms: 1711929599999});
export const METADATA = Object.freeze([
  {role: 'publisher_format_and_checksums', url: 'https://raw.githubusercontent.com/binance/binance-public-data/master/README.md'},
  {role: 'publisher_dataset_terms', url: 'https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md'},
]);
const NAME = symbol => `${symbol}-1m-2024-03`;
const URLS = symbol => ({symbol, zip_url: `https://data.binance.vision/data/spot/monthly/klines/${symbol}/1m/${NAME(symbol)}.zip`,
  checksum_url: `https://data.binance.vision/data/spot/monthly/klines/${symbol}/1m/${NAME(symbol)}.zip.CHECKSUM`});
const exactBytes = (left, right) => left?.bytes === right?.bytes && left?.sha256 === right?.sha256;
const fields = pin => ({path: pin.path, bytes: pin.bytes, sha256: pin.sha256});
const self = fileURLToPath(import.meta.url);

export function validateDeclaration(value) {
  require(value?.kind === 'cpb_v16_temporal_confirmation_data_selection_v1' && value.status === 'prospective_selection_before_download_or_numeric_access', 'prospective actual V16 data selection required');
  for (const [key, expected] of Object.entries(GEOMETRY)) require(value[key] === expected, `fixed March selection differs: ${key}`);
  require(Array.isArray(value.sources) && value.sources.length === 3 && SYMBOLS.every((symbol, index) => {
    const actual = value.sources[index], expected = URLS(symbol);
    return actual?.symbol === expected.symbol && actual.zip_url === expected.zip_url && actual.checksum_url === expected.checksum_url;
  }), 'only the three prospectively declared official March spot sources are admitted');
  require(Array.isArray(value.metadata_sources) && value.metadata_sources.length === 2 && METADATA.every((expected, index) =>
    value.metadata_sources[index]?.role === expected.role && value.metadata_sources[index]?.url === expected.url), 'exact publisher README and terms metadata sources required');
  require(value.confirmation?.role === 'confirmation_only' && value.confirmation.training_use === false
    && value.confirmation.head_fit === false && value.confirmation.normalizer_fit === false
    && value.confirmation.probes === 32 && value.confirmation.history_legal === 256 && value.confirmation.encoder_history === 128,
  'confirmation-only fixed32/H256 population without fitted confirmation state required');
  require(value.license?.publisher === 'Binance Vision' && value.license.commercial_use_authorized === false
    && value.license.dataset_and_derived_artifacts_published === false && value.license.attribution_required === true
    && value.license.data_license === 'CC BY-NC-SA 4.0 plus Binance Vision Dataset Terms'
    && value.license.terms_version === '1.0' && value.license.terms_updated === '2026-08-26', 'exact local non-commercial research license declaration required');
  require(value.resumption && typeof value.resumption.path === 'string', 'actual authorized research admission binding required');
  return value;
}

export function checksumDigest(bytes, expectedName) {
  const text = Buffer.from(bytes).toString('utf8');
  require(Buffer.byteLength(text) === bytes.length, 'ASCII checksum payload required');
  const match = /^([0-9a-fA-F]{64}) [ *]([^\r\n]+)(?:\r?\n)?$/.exec(text);
  require(match && match[2] === expectedName && !match[2].includes('/') && !match[2].includes('\\'), 'publisher checksum must bind exactly the declared ZIP basename');
  return match[1].toLowerCase();
}
const crcTable = Array.from({length: 256}, (_, value) => {
  for (let bit = 0; bit < 8; ++bit) value = (value & 1) ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  return value >>> 0;
});
export function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
/** Bounded, non-executing, single-member ZIP reader. No paths are extracted. */
export function extractCSV(bytes, expectedName, maxCSVBytes = 50_000_000) {
  require(Buffer.isBuffer(bytes) && bytes.length >= 22 && bytes.length <= 15_000_000, 'bounded complete ZIP bytes required');
  require(Number.isSafeInteger(maxCSVBytes) && maxCSVBytes > 0 && maxCSVBytes <= 50_000_000, 'bounded CSV expansion limit required');
  let end = -1;
  for (let offset = bytes.length - 22; offset >= Math.max(0, bytes.length - 65557); --offset) {
    if (bytes.readUInt32LE(offset) === 0x06054b50 && offset + 22 + bytes.readUInt16LE(offset + 20) === bytes.length) { end = offset; break; }
  }
  require(end >= 0 && bytes.readUInt16LE(end + 4) === 0 && bytes.readUInt16LE(end + 6) === 0
    && bytes.readUInt16LE(end + 8) === 1 && bytes.readUInt16LE(end + 10) === 1, 'one-member non-spanned ZIP required');
  const directoryBytes = bytes.readUInt32LE(end + 12), directory = bytes.readUInt32LE(end + 16);
  require(directory + directoryBytes === end && directoryBytes >= 46 && directory + 46 <= end
    && bytes.readUInt32LE(directory) === 0x02014b50, 'exact central ZIP directory required');
  const flags = bytes.readUInt16LE(directory + 8), method = bytes.readUInt16LE(directory + 10);
  const crc = bytes.readUInt32LE(directory + 16), packed = bytes.readUInt32LE(directory + 20), size = bytes.readUInt32LE(directory + 24);
  const nameLength = bytes.readUInt16LE(directory + 28), extraLength = bytes.readUInt16LE(directory + 30), commentLength = bytes.readUInt16LE(directory + 32);
  require((flags & ~(8 | 2048)) === 0 && [0, 8].includes(method) && bytes.readUInt16LE(directory + 34) === 0
    && packed > 0 && packed < 0xffffffff && size > 0 && size <= maxCSVBytes
    && directoryBytes === 46 + nameLength + extraLength + commentLength && bytes.readUInt32LE(directory + 42) === 0, 'unsupported ZIP flags, layout, compression or size');
  const name = bytes.subarray(directory + 46, directory + 46 + nameLength);
  require(name.equals(Buffer.from(expectedName, 'ascii')) && /^[A-Z]+-1m-2024-03\.csv$/.test(expectedName), 'ZIP member must be the one exact declared CSV basename');
  const attributes = bytes.readUInt32LE(directory + 38), unixType = (attributes >>> 16) & 0xf000;
  require((attributes & 16) === 0 && [0, 0x8000].includes(unixType), 'regular ZIP member required; links and directories refused');
  require(directory >= 30 && bytes.readUInt32LE(0) === 0x04034b50 && bytes.readUInt16LE(6) === flags && bytes.readUInt16LE(8) === method, 'exact local ZIP header required');
  const localNameLength = bytes.readUInt16LE(26), localExtraLength = bytes.readUInt16LE(28);
  const dataOffset = 30 + localNameLength + localExtraLength;
  require(dataOffset <= directory && bytes.subarray(30, 30 + localNameLength).equals(name) && dataOffset + packed <= directory, 'local ZIP member differs from central binding');
  if (flags & 8) {
    for (const [offset, expected] of [[14, crc], [18, packed], [22, size]]) require([0, expected].includes(bytes.readUInt32LE(offset)), 'local descriptor header conflict');
    const descriptor = dataOffset + packed, descriptorBytes = directory - descriptor;
    require([12, 16].includes(descriptorBytes), 'exact ZIP data descriptor required');
    const d = descriptorBytes === 16 ? descriptor + 4 : descriptor;
    if (descriptorBytes === 16) require(bytes.readUInt32LE(descriptor) === 0x08074b50, 'descriptor signature mismatch');
    require(bytes.readUInt32LE(d) === crc && bytes.readUInt32LE(d + 4) === packed && bytes.readUInt32LE(d + 8) === size, 'ZIP descriptor identity mismatch');
  } else require(bytes.readUInt32LE(14) === crc && bytes.readUInt32LE(18) === packed && bytes.readUInt32LE(22) === size
    && dataOffset + packed === directory, 'local ZIP identity/length mismatch');
  const compressed = bytes.subarray(dataOffset, dataOffset + packed);
  const output = method === 0 ? Buffer.from(compressed) : inflateRawSync(compressed, {maxOutputLength: maxCSVBytes});
  require(output.length === size && crc32(output) === crc, 'CSV whole length or ZIP CRC differs');
  return output;
}

function readPhysical(filename, {root, max = 70_000_000} = {}) {
  const row = admitFiles([filename], {root, singleLink: true})[0];
  require(row.bytes <= max, 'bounded data/control input required');
  const bytes = readAdmitted(row);
  return {bytes, pin: {path: root ? path.relative(root, filename).split(path.sep).join('/') : filename, bytes: bytes.length, sha256: sha256(bytes)}};
}
function control(root, filename) {
  const absolute = path.resolve(root, filename);
  const value = readPhysical(absolute, {root});
  return {...value, data: parseJSON(value.bytes, {duplicates: 'reject', integers: 'bigint'})};
}
function assertPin(root, value, expected, allowBuild = false) {
  require(expected && Number.isSafeInteger(expected.bytes) && /^[0-9a-f]{64}$/.test(expected.sha256 ?? '')
    && (expected.repository === undefined || expected.repository === 'Mityane-geometrics'), 'current full path/bytes/SHA control pin required');
  const absolute = path.resolve(root, expected.path);
  const locator = path.relative(root, absolute).split(path.sep).join('/');
  if (!allowBuild) safeRelative(locator);
  else require(absolute.startsWith('/opt/mityane-geometrics/') && absolute.endsWith('/csv_to_cache88'), 'reviewed managed native converter path required');
  const observed = value ?? readPhysical(absolute, allowBuild ? {} : {root});
  const normalized = allowBuild ? absolute : locator;
  require(observed.pin.path === normalized && exactBytes(observed.pin, expected), `bound input identity changed: ${normalized}`);
  return observed;
}
function sameCurrent(root, left, right) {
  return left && right && (left.repository === undefined || left.repository === 'Mityane-geometrics')
    && (right.repository === undefined || right.repository === 'Mityane-geometrics')
    && path.resolve(root, left.path) === path.resolve(root, right.path) && exactBytes(left, right);
}
function checkReview(root, review) {
  require(review.data.kind === 'cpb_v16_temporal_independent_source_review_v1' && review.data.passed === true && Array.isArray(review.data.files), 'passed independent actual source review required');
  const sealed = [];
  for (const name of SOURCES) {
    const pin = review.data.files.find(row => row.path === name);
    require(pin, `missing reviewed data source: ${name}`);
    assertPin(root, null, pin); sealed.push(fields(pin));
  }
  return sealed;
}
function currentPinSchema(pin, {native = false} = {}) {
  require(pin && Number.isSafeInteger(pin.bytes) && pin.bytes >= 0 && /^[0-9a-f]{64}$/.test(pin.sha256 ?? '')
    && typeof pin.path === 'string' && (pin.repository === undefined || pin.repository === 'Mityane-geometrics'), 'full current physical control identity required');
  if (native) require(pin.path.startsWith('/opt/mityane-geometrics/') && path.posix.normalize(pin.path) === pin.path, 'managed native binary identity required');
  else safeRelative(pin.path);
  return pin;
}
function nativePin(expected) {
  currentPinSchema(expected, {native: true});
  const value = readPhysical(expected.path);
  require(exactBytes(value.pin, expected), 'bound native binary identity changed'); return value;
}
export function validateParityReceipt(value) {
  require(value?.kind === 'cpb_v16_temporal_target_precision_parity_v1' && value.passed === true && value.status === 'passed'
    && value.train_rows === 112 && value.validation_rows === 32 && value.target_precision === 'legacy_f32_then_f64'
    && Array.isArray(value.sources) && value.sources.length > 0
    && Array.isArray(value.january_caches) && value.january_caches.length === 3
    && Array.isArray(value.original_targets) && value.original_targets.length === 2,
  'exact passed January target-precision parity required before March values');
  for (const name of ['all_target_values_bitwise_equal', 'all_masks_and_ids_exact']) require(value[name] === true, `January parity ${name} required`);
  require(value.March_values_opened === false && value.encoder_calls === 0 && value.head_fits === 0, 'January-only zero-model parity required');
  for (const pin of [...value.sources, ...value.january_caches, ...value.original_targets,
    value.source_review, value.protocol_review, value.numeric_result, value.log]) currentPinSchema(pin);
  currentPinSchema(value.binary, {native: true});
  require(new Set(value.january_caches.map(pin => pin.path)).size === 3 && new Set(value.original_targets.map(pin => pin.path)).size === 2, 'January parity inputs must be distinct bound files');
  return value;
}
function recheck(root, pins) { for (const pin of pins) assertPin(root, null, pin); }
async function download(url, cap) {
  const parsed = new URL(url);
  require(parsed.protocol === 'https:' && !parsed.username && !parsed.password && !parsed.port
    && ['data.binance.vision', 'raw.githubusercontent.com'].includes(parsed.hostname), 'declared official HTTPS origin required');
  return await new Promise((resolve, reject) => {
    const request = https.get(parsed, {headers: {'user-agent': 'Mityane-geometrics-local-research/1'}}, response => {
      if (response.statusCode !== 200) { response.resume(); reject(new Error(`Publisher HTTP status ${response.statusCode}; redirect/substitution refused`)); return; }
      const length = response.headers['content-length'];
      if (length && (!/^\d+$/.test(length) || Number(length) > cap)) { response.destroy(); reject(new Error('Publisher response exceeds declared byte cap')); return; }
      const chunks = []; let count = 0;
      response.on('data', chunk => { count += chunk.length; if (count > cap) response.destroy(new Error('Publisher streamed response exceeds byte cap')); else chunks.push(chunk); });
      response.on('error', reject);
      response.on('end', () => {
        if (length && count !== Number(length)) { reject(new Error('Publisher content length mismatch')); return; }
        resolve({bytes: Buffer.concat(chunks), url, retrieved_utc: new Date().toISOString(), status: response.statusCode});
      });
    });
    request.setTimeout(120000, () => request.destroy(new Error('Publisher response timeout')));
    request.on('error', reject);
  });
}
function guardRoot(root, output, mode) {
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv') && root === '/mityane', 'existing managed Linux /mityane container required');
  canonicalDirectory(root);
  const middle = mode === 'metadata' ? '/publisher-metadata' : '';
  require(new RegExp(`^/mityane/\\.temp/data/cpb-v16-temporal-001${middle}/attempt-\\d{3}$`).test(output), 'new named admitted data-preparation leaf required');
  require(!fs.existsSync(output), 'refusing existing data-preparation leaf');
}
function createSubdir(root, output, name) { return createExclusiveDirectory(path.join(output, name), {root}); }
function savedFile(root, filename, bytes) {
  writeNew(filename, bytes, {mode: 0o600}); return readPhysical(filename, {root}).pin;
}

export async function prepareMetadata({root, declarationPath, sourceReviewPath, output, fetch = download}) {
  guardRoot(root, output, 'metadata');
  const declaration = control(root, declarationPath), review = control(root, sourceReviewPath);
  validateDeclaration(declaration.data); const sourceClosure = checkReview(root, review);
  const admission = assertPin(root, null, declaration.data.resumption);
  createExclusiveDirectory(output, {root});
  const members = [];
  try {
    for (const row of METADATA) {
      const fetched = await fetch(row.url, 2_000_000);
      require(fetched.url === row.url && fetched.status === 200, 'exact publisher metadata response required');
      const name = row.role === 'publisher_dataset_terms' ? 'TERMS_AND_CONDITIONS.md' : 'README.md';
      members.push({...row, retrieved_utc: fetched.retrieved_utc, file: savedFile(root, path.join(output, name), fetched.bytes)});
    }
    recheck(root, [declaration.pin, review.pin, admission.pin, ...sourceClosure]);
    for (const member of members) assertPin(root, null, member.file);
    const receipt = {kind: 'cpb_v16_temporal_publisher_metadata_fetch_v1', passed: true, status: 'passed', created_utc: new Date().toISOString(),
      declaration: declaration.pin, source_review: review.pin, source_closure: sourceClosure, members,
      interpretation: 'Exact publisher metadata saved for independent license/format review before numeric import.', numeric_market_values_opened: false};
    writeNew(path.join(output, 'metadata-receipt.json'), jsonBytes(receipt)); return receipt;
  } catch (error) {
    writeNew(path.join(output, 'failure.json'), jsonBytes({kind: 'cpb_v16_temporal_publisher_metadata_fetch_failure_v1', passed: false,
      declaration: declaration.pin, source_review: review.pin, members, error: error.message})); throw error;
  }
}

export async function importConfirmation({root, declarationPath, sourceReviewPath, buildReceiptPath, targetParityPath, publisherReviewPath, output, fetch = download}) {
  guardRoot(root, output, 'import');
  const declaration = control(root, declarationPath), review = control(root, sourceReviewPath);
  validateDeclaration(declaration.data); const sourceClosure = checkReview(root, review);
  const admission = assertPin(root, null, declaration.data.resumption);
  const build = control(root, buildReceiptPath), parity = control(root, targetParityPath), publisher = control(root, publisherReviewPath);
  require(build.data.kind === 'cpb_v16_temporal_data_converter_build_v1' && build.data.passed === true
    && sameCurrent(root, build.data.source_review, review.pin) && Array.isArray(build.data.sources), 'reviewed actual native converter build receipt required');
  for (const name of ['cache88.h', 'csv_to_cache88.cpp']) {
    const pin = build.data.sources.find(row => row.path === `${DIRECTORY}/${name}`);
    require(pin && sourceClosure.some(row => sameCurrent(root, pin, row)), 'native build sources must match reviewed source closure');
  }
  const converter = assertPin(root, null, build.data.converter, true);
  validateParityReceipt(parity.data);
  const parityPins = [...parity.data.sources, ...parity.data.january_caches, ...parity.data.original_targets,
    parity.data.source_review, parity.data.protocol_review, parity.data.numeric_result, parity.data.log];
  for (const pin of parityPins) assertPin(root, null, pin);
  const parityBinary = nativePin(parity.data.binary);
  const paritySourceReview = control(root, parity.data.source_review.path), parityProtocolReview = control(root, parity.data.protocol_review.path);
  require(paritySourceReview.data.passed === true && parityProtocolReview.data.passed === true, 'passed parity source/protocol reviews required');
  const parityNumeric = control(root, parity.data.numeric_result.path);
  require(parityNumeric.data.kind === 'cpb_v16_temporal_target_precision_parity_numeric_v1' && parityNumeric.data.passed === true
    && parityNumeric.data.status === 'passed' && parityNumeric.data.train_rows === 112 && parityNumeric.data.validation_rows === 32
    && parityNumeric.data.target_precision === 'legacy_f32_then_f64' && parityNumeric.data.all_target_values_bitwise_equal === true
    && parityNumeric.data.all_masks_and_ids_exact === true && parityNumeric.data.March_values_opened === false
    && parityNumeric.data.encoder_calls === 0 && parityNumeric.data.head_fits === 0, 'actual exact January-only numeric parity result required');
  require(publisher.data.kind === 'cpb_v16_temporal_publisher_metadata_review_v1' && publisher.data.passed === true && publisher.data.status === 'passed'
    && sameCurrent(root, publisher.data.declaration, declaration.pin)
    && publisher.data.local_noncommercial_research_authorized === true && publisher.data.no_external_publication === true
    && same(publisher.data.license, declaration.data.license),
  'independent exact publisher metadata/license review required');
  const metadata = assertPin(root, null, publisher.data.metadata_receipt);
  const metadataData = parseJSON(metadata.bytes, {duplicates: 'reject', integers: 'bigint'});
  require(metadataData.kind === 'cpb_v16_temporal_publisher_metadata_fetch_v1' && metadataData.passed === true && metadataData.status === 'passed'
    && sameCurrent(root, metadataData.declaration, declaration.pin) && metadataData.members?.length === 2, 'exact completed publisher metadata fetch bound to declaration required');
  for (const [index, expected] of METADATA.entries()) {
    const member = metadataData.members[index];
    require(member?.role === expected.role && member.url === expected.url, 'exact publisher metadata member binding required');
    assertPin(root, null, member.file);
  }
  const sealed = [declaration.pin, review.pin, admission.pin, build.pin, parity.pin, publisher.pin, metadata.pin,
    ...metadataData.members.map(row => row.file), ...sourceClosure, ...parityPins];
  createExclusiveDirectory(output, {root});
  const archives = createSubdir(root, output, 'archives'), csv = createSubdir(root, output, 'csv'), logs = createSubdir(root, output, 'logs');
  const members = [];
  try {
    for (const selected of declaration.data.sources) {
      const zipName = `${NAME(selected.symbol)}.zip`, csvName = `${NAME(selected.symbol)}.csv`;
      const row = {symbol: selected.symbol, zip_url: selected.zip_url, checksum_url: selected.checksum_url}; members.push(row);
      const checksum = await fetch(selected.checksum_url, 4096);
      require(checksum.url === selected.checksum_url && checksum.status === 200, 'exact publisher checksum response required');
      row.publisher_checksum = savedFile(root, path.join(archives, `${zipName}.CHECKSUM`), checksum.bytes);
      row.publisher_sha256 = checksumDigest(checksum.bytes, zipName);
      const zip = await fetch(selected.zip_url, 15_000_000);
      require(zip.url === selected.zip_url && zip.status === 200, 'exact publisher ZIP response required');
      row.zip = savedFile(root, path.join(archives, zipName), zip.bytes);
      require(row.zip.sha256 === row.publisher_sha256, 'downloaded ZIP differs from publisher SHA256');
      row.retrieved_utc = zip.retrieved_utc;
      const csvBytes = extractCSV(zip.bytes, csvName);
      row.csv = savedFile(root, path.join(csv, csvName), csvBytes);
      recheck(root, sealed); assertPin(root, null, build.data.converter, true); nativePin(parityBinary.pin);
      const cachePath = path.join(csv, `${NAME(selected.symbol)}.cache.bin`);
      const command = [converter.pin.path, '--input', path.join(root, row.csv.path), '--output', cachePath,
        '--first-open', String(GEOMETRY.first_open_ms), '--rows', String(GEOMETRY.rows)];
      const execution = spawnSync(command[0], command.slice(1), {encoding: 'utf8', timeout: 120000, maxBuffer: 1024 * 1024,
        env: {...process.env, LC_ALL: 'C', LANG: 'C'}});
      row.stdout = savedFile(root, path.join(logs, `${selected.symbol}.stdout`), Buffer.from(execution.stdout ?? ''));
      row.stderr = savedFile(root, path.join(logs, `${selected.symbol}.stderr`), Buffer.from(execution.stderr ?? ''));
      row.command = command; row.exit_status = execution.status;
      assertPin(root, null, row.csv); assertPin(root, null, build.data.converter, true);
      require(!execution.error && execution.status === 0 && execution.signal === null, 'native converter failed; logs/partial cache retained');
      const result = parseJSON(execution.stdout, {duplicates: 'reject', integers: 'bigint'});
      require(result.kind === 'cpb_v16_temporal_cache88_conversion_v1' && result.passed === true
        && result.rows === GEOMETRY.rows && result.first_open_ms === GEOMETRY.first_open_ms
        && result.first_close_ms === GEOMETRY.first_close_ms && result.last_close_ms === GEOMETRY.last_close_ms
        && result.cache_bytes === GEOMETRY.rows * 88 && result.normalizations === 0 && result.gap_fills === 0 && result.rows_filtered === 0,
      'native whole-population conversion geometry/semantics differ');
      row.cache = readPhysical(cachePath, {root}).pin;
      require(row.cache.bytes === GEOMETRY.rows * 88, 'exact CACHE88 byte length required');
      row.conversion = result;
    }
    recheck(root, sealed); assertPin(root, null, build.data.converter, true); nativePin(parityBinary.pin);
    for (const row of members) for (const pin of [row.zip, row.publisher_checksum, row.csv, row.cache, row.stdout, row.stderr]) assertPin(root, null, pin);
    const receipt = {kind: 'cpb_v16_temporal_confirmation_data_import_v1', passed: true, status: 'passed', created_utc: new Date().toISOString(),
      declaration: declaration.pin, resumption: admission.pin, source_review: review.pin, source_closure: sourceClosure,
      build_receipt: build.pin, converter: converter.pin, target_precision_parity: parity.pin, publisher_review: publisher.pin,
      publisher_metadata_receipt: metadata.pin, publisher_metadata: metadataData.members, license: declaration.data.license,
      dataset_id: 'MKT-HIST-2-MAR2024-confirmation', ...GEOMETRY, channels: SYMBOLS, members,
      validation_policy: 'Exact12 numeric fields; finite binary64 decimal values, positive OHLC, nonnegative activity, int32 trade count cast to binary64, no normalization/filter/gap filling. Complete declared minute lattice.',
      training_use: false, scaler_fits: 0, bank_fits: 0, targets_generated: 0, head_fits: 0, model_calls: 0, quality_outcomes_computed: false,
      usage: 'Confirmation only. All six models and TRAIN-fitted heads must be sealed before scoring/outcome inspection. Local non-production research; attribution and publisher data terms retained.'};
    writeNew(path.join(output, 'import-receipt.json'), jsonBytes(receipt)); return receipt;
  } catch (error) {
    writeNew(path.join(output, 'failure.json'), jsonBytes({kind: 'cpb_v16_temporal_confirmation_data_import_failure_v1', passed: false,
      declaration: declaration.pin, source_review: review.pin, build_receipt: build.pin,
      target_precision_parity: parity.pin, publisher_review: publisher.pin, members, error: error.message,
      preservation: 'Existing and partial inputs/outputs/logs retained. No retry or overwrite in this leaf.'})); throw error;
  }
}

async function cli() {
  const args = process.argv.slice(2), mode = args.shift(), options = {};
  const names = mode === 'metadata' ? ['declaration', 'source-review', 'output']
    : mode === 'import' ? ['declaration', 'source-review', 'build-receipt', 'target-parity', 'publisher-review', 'output'] : [];
  require(names.length > 0, 'prepare.mjs metadata|import with explicit reviewed input paths required');
  while (args.length) { const flag = args.shift(); require(names.some(name => flag === `--${name}`) && args.length && !Object.hasOwn(options, flag), `invalid argument: ${flag}`); options[flag] = args.shift(); }
  for (const name of names) require(options[`--${name}`], `missing --${name}`);
  const input = {root: '/mityane', declarationPath: options['--declaration'], sourceReviewPath: options['--source-review'], output: path.resolve(options['--output']),
    buildReceiptPath: options['--build-receipt'], targetParityPath: options['--target-parity'], publisherReviewPath: options['--publisher-review']};
  const receipt = mode === 'metadata' ? await prepareMetadata(input) : await importConfirmation(input);
  console.log(JSON.stringify({status: receipt.status, kind: receipt.kind, members: receipt.members.length}));
}
if (process.argv[1] && path.resolve(process.argv[1]) === self) cli().catch(error => { console.error(error.stack); process.exitCode = 1; });
