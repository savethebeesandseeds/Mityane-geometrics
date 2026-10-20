// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Applies only the three independently reviewed V14 machine records.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {
  admitFiles, canonicalDirectory, createExclusiveDirectory, hashAdmitted, inside,
  jsonBytes, parseJSON, readAdmitted, requireCondition as require, safeRelative,
  sha256, writeNew,
} from '../../../../shared/artifacts/node/v1/files.mjs';

const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const SHARED = 'code/shared/artifacts/node/v1/files.mjs';
const FAMILY = 'code/embeddings/encoders/causal_patch_bottleneck_mae/versions';
export const APPLY_TARGETS = Object.freeze([
  `${FAMILY}/current.json`, `${FAMILY}/instances/CPB-v14.market-01.json`,
  'code/embeddings/evaluation/records/CPB-v14.market-01.closeout.json',
]);
const REQUIRED_SOURCES = [`${PROTOCOL}/stage.mjs`, `${PROTOCOL}/apply.mjs`, SHARED];
const equal = (left, right) => jsonBytes(left).equals(jsonBytes(right));
const absolute = (root, relative) => path.join(root, safeRelative(relative));
function exists(filename) {
  try { fs.lstatSync(filename); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}
function pinFile(root, relative) {
  const row = admitFiles([absolute(root, relative)], {root, singleLink: true})[0];
  return {path: relative, bytes: row.bytes, sha256: hashAdmitted(row)};
}
function readFile(root, relative) {
  const row = admitFiles([absolute(root, relative)], {root, singleLink: true})[0];
  const bytes = readAdmitted(row);
  return {pin: {path: relative, bytes: bytes.length, sha256: sha256(bytes)}, bytes, mode: row.mode};
}
function readJSON(root, relative) { const file = readFile(root, relative); return {...file, data: parseJSON(file.bytes)}; }
function expectPin(actual, expected, message) {
  require(expected && actual.path === expected.path && actual.bytes === expected.bytes &&
    actual.sha256 === expected.sha256, message);
}
function checkSourcePins(root, pins, required, message) {
  require(Array.isArray(pins), `${message}: explicit source pins required`);
  const names = new Set();
  for (const expected of pins) {
    require(!names.has(expected.path), `${message}: duplicate source pin`); names.add(expected.path);
    expectPin(pinFile(root, expected.path), expected, `${message}: changed ${expected.path}`);
  }
  for (const name of required) require(names.has(name), `${message}: missing ${name}`);
}
function checkAbsent(root, relative) {
  const filename = absolute(root, relative);
  require(!exists(filename), `expected absent target already exists: ${relative}`);
  let parent = path.dirname(filename);
  while (!exists(parent)) parent = path.dirname(parent);
  canonicalDirectory(parent, {root});
}
function checkBefore(root, output) {
  const before = output.expected_before;
  require(before?.path === output.path && typeof before.exists === 'boolean', 'exact expected-before target required');
  if (before.exists) expectPin(pinFile(root, output.path), before, `target changed from expected-before: ${output.path}`);
  else checkAbsent(root, output.path);
}
function ensureParent(root, filename) {
  const parent = path.dirname(filename);
  if (!exists(parent)) createExclusiveDirectory(parent, {root});
  canonicalDirectory(parent, {root});
}
function syncDirectory(directory) {
  const descriptor = fs.openSync(directory, fs.constants.O_RDONLY | fs.constants.O_DIRECTORY);
  try { fs.fsyncSync(descriptor); } finally { fs.closeSync(descriptor); }
}
function observed(root, output) {
  try {
    if (!exists(absolute(root, output.path))) return {path: output.path, exists: false,
      matches_staged: false, matches_before: output.expected_before.exists === false};
    const pin = pinFile(root, output.path);
    return {...pin, exists: true,
      matches_staged: pin.bytes === output.bytes && pin.sha256 === output.sha256,
      matches_before: output.expected_before.exists && pin.bytes === output.expected_before.bytes &&
        pin.sha256 === output.expected_before.sha256};
  } catch (error) {
    return {path: output.path, exists: true, matches_staged: false, matches_before: false,
      observation_error: String(error.message)};
  }
}

export function apply({root, attempt, manifestPath, stagedReviewPath, sourceReviewPath}) {
  canonicalDirectory(root); canonicalDirectory(attempt, {root});
  const attemptRelative = path.relative(root, attempt).split(path.sep).join('/');
  require(/^\.temp\/embeddings\/closeout\/cpb-v14\/attempt-\d+$/.test(attemptRelative), 'named V14 attempt required');
  require(manifestPath === `${attemptRelative}/staged-change-manifest.json`, 'exact staged manifest in admitted attempt required');
  const manifest = readJSON(root, manifestPath), review = readJSON(root, stagedReviewPath),
    sourceReview = readJSON(root, sourceReviewPath);
  require(manifest.data.kind === 'cpb_v14_postmigration_staged_change_manifest_v1' &&
    manifest.data.status === 'staged_only_not_applied' && manifest.data.applied === false &&
    manifest.data.protocol === PROTOCOL,
    'unapplied staged V14 manifest required');
  require(review.data.kind === 'cpb_v14_postmigration_staged_review_v1' &&
    review.data.status === 'passed' && review.data.passed === true,
    'independently passed staged review required');
  expectPin(manifest.pin, review.data.manifest, 'staged review does not bind this exact manifest');
  require(sourceReview.data.passed === true, 'independently passed application source review required');
  checkSourcePins(root, sourceReview.data.files, [`${PROTOCOL}/apply.mjs`, SHARED], 'application source review');
  checkSourcePins(root, review.data.source_closure, REQUIRED_SOURCES, 'staged review source closure');
  const outputs = manifest.data.outputs;
  require(Array.isArray(outputs) && outputs.length === 3 && outputs.every((row, index) => row.path === APPLY_TARGETS[index]),
    'only the exact three admitted V14 machine outputs may be applied');
  require(equal(outputs, review.data.outputs), 'staged review does not bind every exact output and expected-before row');
  require(outputs[0].expected_before?.exists === true && outputs.slice(1).every(row => row.expected_before?.exists === false),
    'replace only existing catalog and create only absent instance/summary');
  for (const output of outputs) {
    require(output.staged_path === `${attemptRelative}/staged/${output.path}`, 'staged output escapes the admitted attempt');
    require(Number.isSafeInteger(output.bytes) && output.bytes > 0 && /^[0-9a-f]{64}$/.test(output.sha256),
      'exact staged byte/hash identity required');
  }
  const immutable = manifest.data.immutable_preapplication_inputs;
  require(immutable && typeof immutable === 'object' && !Array.isArray(immutable), 'explicit preapplication input bindings required');
  for (const role of ['admission', 'inputs', 'aliases', 'alias_verification', 'proposal_review', 'source_review']) {
    require(immutable[role]?.path && Number.isSafeInteger(immutable[role].bytes) &&
      /^[0-9a-f]{64}$/.test(immutable[role].sha256 ?? ''), `required immutable preapplication binding: ${role}`);
  }
  const recheckInputs = () => {
    expectPin(pinFile(root, manifestPath), manifest.pin, 'staged manifest changed before application');
    expectPin(pinFile(root, stagedReviewPath), review.pin, 'staged review changed before application');
    expectPin(pinFile(root, sourceReviewPath), sourceReview.pin, 'application source review changed before application');
    checkSourcePins(root, sourceReview.data.files, [`${PROTOCOL}/apply.mjs`, SHARED], 'application source review');
    checkSourcePins(root, review.data.source_closure, REQUIRED_SOURCES, 'staged review source closure');
    for (const expected of Object.values(immutable)) expectPin(pinFile(root, expected.path), expected,
      `immutable preapplication input changed: ${expected.path}`);
    for (const output of outputs) {
      checkBefore(root, output);
      expectPin(pinFile(root, output.staged_path), {...output, path: output.staged_path},
        `reviewed staged output changed: ${output.path}`);
    }
  };
  // Complete every admission and content check before acquiring the persistent
  // application lock or writing any marker, before-copy, temp file or live file.
  recheckInputs();
  const applicationRelative = `${attemptRelative}/application.json`;
  const startRelative = `${attemptRelative}/application-start.json`;
  const lockRelative = '.temp/embeddings/closeout/cpb-v14/application-lock.json';
  checkAbsent(root, applicationRelative); checkAbsent(root, startRelative); checkAbsent(root, lockRelative);
  const beforeBuffers = outputs.map(output => output.expected_before.exists ? readFile(root, output.path) : null);
  const stagedBuffers = outputs.map(output => readFile(root, output.staged_path));
  for (const [index, output] of outputs.entries()) {
    if (beforeBuffers[index]) expectPin(beforeBuffers[index].pin, output.expected_before, 'before snapshot changed');
    expectPin(stagedBuffers[index].pin, {...output, path: output.staged_path}, 'staged snapshot changed');
  }
  const inputs = {manifest: manifest.pin, staged_review: review.pin, application_source_review: sourceReview.pin};
  const started = {kind: 'cpb_v14_metadata_application_start_v1', attempt: attemptRelative,
    created_utc: new Date().toISOString(), protocol: PROTOCOL, inputs,
    targets: APPLY_TARGETS, policy: 'Exclusive persistent application claim. Preserve this marker after success or failure; a failed attempt requires explicit reviewed repair.'};
  const lockBytes = jsonBytes(started);
  // This exclusive shared claim serializes all V14 attempts. It is deliberately
  // retained, including on failure, so another attempt cannot guess a recovery.
  writeNew(absolute(root, lockRelative), lockBytes);
  const lockPin = {path: lockRelative, bytes: lockBytes.length, sha256: sha256(lockBytes)};
  const completed = [], beforeCopies = [], temporaryFiles = [];
  let startPin = null;
  const receipt = (status, error = null) => ({
    kind: 'cpb_v14_postmigration_metadata_application_v1', status, passed: status === 'passed',
    created_utc: new Date().toISOString(), protocol: PROTOCOL, inputs,
    immutable_preapplication_inputs: immutable,
    application_source_closure: sourceReview.data.files, staged_review_source_closure: review.data.source_closure,
    shared_application_claim: lockPin, application_start: startPin,
    expected_outputs: outputs, completed_write_operations: completed, before_copies: beforeCopies,
    observed_after: outputs.map(output => observed(root, output)),
    retained_temporary_files: temporaryFiles.filter(relative => exists(absolute(root, relative))).map(relative => {
      try { return pinFile(root, relative); }
      catch (failure) { return {path: relative, observation_error: String(failure.message)}; }
    }),
    error: error ? {name: String(error.name), message: String(error.message)} : null,
    rollback_performed: false,
    preservation_receipt_location: `${attemptRelative}/preservation.json`,
    preservation_policy: 'The named later receipt is not claimed or hashed here. It must independently bind this actual application receipt and the resulting records.',
  });
  try {
    const startBytes = jsonBytes(started);
    writeNew(absolute(root, startRelative), startBytes);
    startPin = {path: startRelative, bytes: startBytes.length, sha256: sha256(startBytes)};
    recheckInputs();
    for (const [index, output] of outputs.entries()) {
      if (!beforeBuffers[index]) continue;
      const relative = `${attemptRelative}/before/${output.path}`;
      if (output.expected_before.whole_before_copy !== undefined) require(output.expected_before.whole_before_copy === relative,
        'admitted whole-before copy must belong to this attempt and exact target');
      const filename = absolute(root, relative);
      if (!exists(filename)) { ensureParent(root, filename); writeNew(filename, beforeBuffers[index].bytes); }
      const pin = pinFile(root, relative);
      require(pin.bytes === output.expected_before.bytes && pin.sha256 === output.expected_before.sha256,
        'whole before-copy differs from admitted target generation');
      beforeCopies.push({target: output.path, ...pin});
    }
    // All target and staged hashes are rechecked together immediately before
    // the first live write. Each target is also checked immediately at its turn.
    recheckInputs();
    for (const index of [1, 2, 0]) {
      const output = outputs[index], filename = absolute(root, output.path);
      checkBefore(root, output);
      expectPin(pinFile(root, output.staged_path), {...output, path: output.staged_path},
        `reviewed staged output changed before write: ${output.path}`);
      ensureParent(root, filename);
      if (!output.expected_before.exists) writeNew(filename, stagedBuffers[index].bytes);
      else {
        const tempRelative = `${path.posix.dirname(output.path)}/.current.cpb-v14-${path.posix.basename(attemptRelative)}-${output.sha256.slice(0, 16)}.new`;
        const temporary = absolute(root, tempRelative);
        temporaryFiles.push(tempRelative);
        writeNew(temporary, stagedBuffers[index].bytes, {mode: beforeBuffers[index].mode});
        checkBefore(root, output);
        // Same-directory rename is atomic. The retained shared claim serializes
        // this protocol; the immediate hash check detects other prior edits.
        fs.renameSync(temporary, filename); syncDirectory(path.dirname(filename));
      }
      expectPin(pinFile(root, output.path), output, `written output differs from reviewed bytes: ${output.path}`);
      completed.push({path: output.path, bytes: output.bytes, sha256: output.sha256,
        operation: output.expected_before.exists ? 'atomic_replace' : 'exclusive_create'});
    }
    const final = receipt('passed');
    require(final.observed_after.every(row => row.matches_staged), 'final output observations differ from reviewed staged bytes');
    writeNew(absolute(root, applicationRelative), jsonBytes(final));
    return {...final, receipt_path: applicationRelative};
  } catch (error) {
    const failed = receipt('partial_failure', error);
    try { writeNew(absolute(root, applicationRelative), jsonBytes(failed)); }
    catch (receiptError) {
      const failure = new Error(`Application failed and its exclusive receipt could not be saved: ${error.message}; receipt: ${receiptError.message}`,
        {cause: error});
      failure.actual_state = failed; throw failure;
    }
    const failure = new Error(`Application stopped; preserve actual state and review ${applicationRelative}: ${error.message}`, {cause: error});
    failure.receipt = failed; failure.receipt_path = applicationRelative; throw failure;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const flags = {'--root': 'root', '--attempt': 'attempt', '--manifest': 'manifestPath',
    '--staged-review': 'stagedReviewPath', '--source-review': 'sourceReviewPath'}, args = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const name = process.argv[index];
    require(Object.hasOwn(flags, name) && process.argv[index + 1] && !Object.hasOwn(args, name),
      'exact paired V14 application arguments required');
    args[name] = process.argv[index + 1];
  }
  require(Object.keys(args).length === Object.keys(flags).length && process.platform === 'linux' &&
    fs.existsSync('/.dockerenv') && args['--root'] === '/mityane',
    'application CLI requires all five arguments and the managed Linux container at /mityane');
  const options = Object.fromEntries(Object.entries(flags).map(([flag, name]) => [name, args[flag]]));
  try { const result = apply(options); process.stdout.write(jsonBytes({status: result.status, receipt_path: result.receipt_path})); }
  catch (error) {
    process.stderr.write(`${error.message}\n`);
    if (error.actual_state) process.stderr.write(jsonBytes(error.actual_state));
    process.exitCode = 1;
  }
}
