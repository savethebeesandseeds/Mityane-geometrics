// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Isolated metadata fixtures; run through the managed Linux container only.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {apply, APPLY_TARGETS} from './apply.mjs';
import {jsonBytes, parseJSON, sha256} from '../../../../shared/artifacts/node/v1/files.mjs';

const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const SHARED = 'code/shared/artifacts/node/v1/files.mjs';
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cpb-v14-apply-fixture-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const relativeAttempt = '.temp/embeddings/closeout/cpb-v14/attempt-001';
  const attempt = path.join(root, relativeAttempt);
  fs.mkdirSync(attempt, {recursive: true});
  const write = (relative, data) => {
    const bytes = Buffer.isBuffer(data) ? data : jsonBytes(data);
    const filename = path.join(root, relative);
    fs.mkdirSync(path.dirname(filename), {recursive: true}); fs.writeFileSync(filename, bytes);
    return {path: relative, bytes: bytes.length, sha256: sha256(bytes)};
  };
  const sources = [
    write(`${PROTOCOL}/stage.mjs`, fs.readFileSync(fileURLToPath(new URL('./stage.mjs', import.meta.url)))),
    write(`${PROTOCOL}/apply.mjs`, fs.readFileSync(fileURLToPath(new URL('./apply.mjs', import.meta.url)))),
    write(SHARED, fs.readFileSync(fileURLToPath(new URL('../../../../shared/artifacts/node/v1/files.mjs', import.meta.url)))),
  ];
  const beforeBytes = jsonBytes({kind: 'fixture_catalog', candidate: 'V13', seed: 18446744073709551615n});
  const old = write(APPLY_TARGETS[0], beforeBytes);
  const outputs = APPLY_TARGETS.map((name, index) => {
    const staged = write(`${relativeAttempt}/staged/${name}`, {
      kind: ['fixture_catalog', 'fixture_instance', 'fixture_summary'][index], candidate: 'V14',
      seed: 18446744073709551615n, mityane_application_selection: null,
    });
    return {path: name, staged_path: staged.path, bytes: staged.bytes, sha256: staged.sha256,
      expected_before: index === 0 ? {...old, exists: true,
        whole_before_copy: `${relativeAttempt}/before/${name}`} : {path: name, exists: false}};
  });
  const immutable = Object.fromEntries(['admission', 'inputs', 'aliases', 'alias_verification', 'proposal_review', 'source_review']
    .map(role => [role, write(`${relativeAttempt}/${role}.json`, {kind: `fixture_${role}`, passed: true})]));
  const manifest = {kind: 'cpb_v14_postmigration_staged_change_manifest_v1', protocol: PROTOCOL,
    status: 'staged_only_not_applied', applied: false, outputs, immutable_preapplication_inputs: immutable};
  const manifestPath = `${relativeAttempt}/staged-change-manifest.json`;
  const sourceReviewPath = `${relativeAttempt}/application-source-review.json`;
  const stagedReviewPath = `${relativeAttempt}/staged-review.json`;
  const sourceReview = {passed: true, files: sources.filter(row => row.path !== `${PROTOCOL}/stage.mjs`)};
  const review = {kind: 'cpb_v14_postmigration_staged_review_v1', status: 'passed', passed: true,
    source_closure: sources, outputs};
  const refresh = () => {
    review.manifest = write(manifestPath, manifest);
    write(sourceReviewPath, sourceReview); write(stagedReviewPath, review);
  };
  refresh();
  const read = relative => parseJSON(fs.readFileSync(path.join(root, relative)));
  const options = {root, attempt, manifestPath, stagedReviewPath, sourceReviewPath};
  return {root, attempt, relativeAttempt, write, read, outputs, manifest, sourceReview, review, refresh,
    options, beforeBytes, sources, immutable,
    assertNoApplication() {
      assert.ok(fs.readFileSync(path.join(root, APPLY_TARGETS[0])).equals(beforeBytes));
      for (const relative of APPLY_TARGETS.slice(1)) assert.equal(fs.existsSync(path.join(root, relative)), false);
      assert.equal(fs.existsSync(path.join(attempt, 'application.json')), false);
      assert.equal(fs.existsSync(path.join(attempt, 'application-start.json')), false);
      assert.equal(fs.existsSync(path.join(root, '.temp/embeddings/closeout/cpb-v14/application-lock.json')), false);
    }};
}

test('applies exactly reviewed bytes, retains whole before-copy and binds exact sources and outputs', t => {
  const f = fixture(t), result = apply(f.options);
  assert.equal(result.status, 'passed');
  assert.equal(result.passed, true);
  assert.deepEqual(result.completed_write_operations.map(row => row.path), [APPLY_TARGETS[1], APPLY_TARGETS[2], APPLY_TARGETS[0]]);
  for (const output of f.outputs) {
    const bytes = fs.readFileSync(path.join(f.root, output.path));
    assert.equal(sha256(bytes), output.sha256);
    assert.equal(parseJSON(bytes).seed, 18446744073709551615n);
    assert.equal(parseJSON(bytes).mityane_application_selection, null);
  }
  assert.ok(fs.readFileSync(path.join(f.attempt, 'before', APPLY_TARGETS[0])).equals(f.beforeBytes));
  const receipt = f.read(`${f.relativeAttempt}/application.json`);
  assert.equal(receipt.observed_after.every(row => row.matches_staged), true);
  assert.equal(receipt.rollback_performed, false);
  assert.deepEqual(receipt.expected_outputs, f.outputs);
  assert.deepEqual(receipt.application_source_closure, f.sourceReview.files);
  assert.deepEqual(receipt.staged_review_source_closure, f.review.source_closure);
  assert.equal(Object.hasOwn(receipt, 'preservation_sha256'), false);
  assert.equal(receipt.preservation_receipt_location, `${f.relativeAttempt}/preservation.json`);
  assert.equal(sha256(fs.readFileSync(path.join(f.root, f.options.manifestPath))), receipt.inputs.manifest.sha256);
});

test('stale before hash leaves every live record and application artifact untouched', t => {
  const f = fixture(t);
  const changed = jsonBytes({candidate: 'concurrent edit'});
  fs.writeFileSync(path.join(f.root, APPLY_TARGETS[0]), changed);
  assert.throws(() => apply(f.options), /target changed from expected-before/);
  assert.ok(fs.readFileSync(path.join(f.root, APPLY_TARGETS[0])).equals(changed));
  assert.equal(fs.existsSync(path.join(f.attempt, 'application-start.json')), false);
  assert.equal(fs.existsSync(path.join(f.attempt, 'application.json')), false);
  assert.equal(fs.existsSync(path.join(f.root, APPLY_TARGETS[1])), false);
  assert.equal(fs.existsSync(path.join(f.root, APPLY_TARGETS[2])), false);
});

test('tampered staged bytes or missing late staged output stops before any first write', t => {
  const f = fixture(t), filename = path.join(f.root, f.outputs[2].staged_path);
  const bytes = fs.readFileSync(filename);
  fs.appendFileSync(filename, ' ');
  assert.throws(() => apply(f.options), /reviewed staged output changed/); f.assertNoApplication();
  fs.unlinkSync(filename);
  assert.throws(() => apply(f.options)); f.assertNoApplication();
  fs.writeFileSync(filename, bytes);
});

test('failed independent review and a receipt for another manifest are rejected', t => {
  const f = fixture(t);
  f.review.passed = false; f.refresh();
  assert.throws(() => apply(f.options), /independently passed staged review/); f.assertNoApplication();
  f.review.passed = true; f.refresh();
  f.review.manifest.sha256 = '0'.repeat(64);
  f.write(f.options.stagedReviewPath, f.review);
  assert.throws(() => apply(f.options), /does not bind this exact manifest/); f.assertNoApplication();
});

test('every staged output must be explicitly reviewed and target set cannot include another file', t => {
  const f = fixture(t);
  f.review.outputs = f.outputs.slice(0, 2); f.refresh();
  assert.throws(() => apply(f.options), /every exact output/); f.assertNoApplication();
  f.review.outputs = f.outputs;
  f.manifest.outputs[2].path = '.archive/embeddings/forbidden.json'; f.refresh();
  assert.throws(() => apply(f.options), /exact three admitted/); f.assertNoApplication();
});

test('source closure omissions, source tampering and immutable review-input tampering are rejected', t => {
  const f = fixture(t);
  f.review.source_closure = f.sources.slice(1); f.refresh();
  assert.throws(() => apply(f.options), /staged review source closure: missing/); f.assertNoApplication();
  f.review.source_closure = f.sources; f.refresh();
  const source = path.join(f.root, `${PROTOCOL}/apply.mjs`), bytes = fs.readFileSync(source);
  fs.appendFileSync(source, '\n// concurrent source edit\n');
  assert.throws(() => apply(f.options), /application source review: changed/); f.assertNoApplication();
  fs.writeFileSync(source, bytes);
  f.write(f.immutable.alias_verification.path, {passed: false});
  assert.throws(() => apply(f.options), /immutable preapplication input changed/); f.assertNoApplication();
});

test('dangling link at an expected-absent target is preserved and cannot be overwritten', t => {
  const f = fixture(t), filename = path.join(f.root, APPLY_TARGETS[1]);
  fs.mkdirSync(path.dirname(filename), {recursive: true}); fs.symlinkSync(path.join(f.root, 'missing-file'), filename);
  assert.throws(() => apply(f.options), /expected absent target already exists/);
  assert.equal(fs.lstatSync(filename).isSymbolicLink(), true);
  assert.equal(fs.existsSync(path.join(f.attempt, 'application-start.json')), false);
  assert.equal(fs.existsSync(path.join(f.attempt, 'application.json')), false);
});

test('shared application claim rejects a competing attempt without replacing its evidence', t => {
  const f = fixture(t), relative = '.temp/embeddings/closeout/cpb-v14/application-lock.json';
  const claim = f.write(relative, {attempt: 'other inspected attempt'});
  assert.throws(() => apply(f.options), /expected absent target already exists/);
  assert.equal(sha256(fs.readFileSync(path.join(f.root, relative))), claim.sha256);
  assert.ok(fs.readFileSync(path.join(f.root, APPLY_TARGETS[0])).equals(f.beforeBytes));
  assert.equal(fs.existsSync(path.join(f.root, APPLY_TARGETS[1])), false);
  assert.equal(fs.existsSync(path.join(f.root, APPLY_TARGETS[2])), false);
  assert.equal(fs.existsSync(path.join(f.attempt, 'application.json')), false);
});

test('partial failure preserves actual new records, catalog before bytes, temp file and failure receipt', t => {
  const f = fixture(t), rename = fs.renameSync;
  fs.renameSync = function (from, to) {
    if (to === path.join(f.root, APPLY_TARGETS[0])) throw new Error('fixture forced atomic catalog failure');
    return rename.call(this, from, to);
  };
  let failure;
  try { apply(f.options); } catch (error) { failure = error; }
  finally { fs.renameSync = rename; }
  assert.ok(failure);
  assert.match(failure.message, /preserve actual state/);
  const receipt = f.read(`${f.relativeAttempt}/application.json`);
  assert.equal(receipt.status, 'partial_failure');
  assert.equal(receipt.passed, false);
  assert.equal(receipt.rollback_performed, false);
  assert.deepEqual(receipt.completed_write_operations.map(row => row.path), APPLY_TARGETS.slice(1));
  assert.deepEqual(receipt.observed_after.map(row => row.matches_staged), [false, true, true]);
  assert.deepEqual(receipt.observed_after.map(row => row.matches_before), [true, false, false]);
  assert.ok(fs.readFileSync(path.join(f.root, APPLY_TARGETS[0])).equals(f.beforeBytes));
  assert.ok(fs.readFileSync(path.join(f.attempt, 'before', APPLY_TARGETS[0])).equals(f.beforeBytes));
  assert.equal(receipt.retained_temporary_files.length, 1);
  assert.equal(receipt.retained_temporary_files[0].sha256, f.outputs[0].sha256);
  const receiptBytes = fs.readFileSync(path.join(f.attempt, 'application.json'));
  assert.throws(() => apply(f.options));
  assert.ok(fs.readFileSync(path.join(f.attempt, 'application.json')).equals(receiptBytes));
});

test('a conflicting whole-before copy yields an honest failure without changing live records', t => {
  const f = fixture(t);
  f.write(`${f.relativeAttempt}/before/${APPLY_TARGETS[0]}`, {wrong: 'generation'});
  assert.throws(() => apply(f.options), /whole before-copy differs/);
  const receipt = f.read(`${f.relativeAttempt}/application.json`);
  assert.equal(receipt.status, 'partial_failure');
  assert.deepEqual(receipt.completed_write_operations, []);
  assert.equal(receipt.observed_after.every(row => row.matches_before), true);
  assert.ok(fs.readFileSync(path.join(f.root, APPLY_TARGETS[0])).equals(f.beforeBytes));
  assert.equal(fs.existsSync(path.join(f.root, APPLY_TARGETS[1])), false);
  assert.equal(fs.existsSync(path.join(f.root, APPLY_TARGETS[2])), false);
});
