// Artificial retained-ledger integrity tests. No study is fitted or replayed.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {jsonBytes, sha256} from '../../../shared/artifacts/node/v1/files.mjs';
import {verifyRetainedStudyFiles, bindVerificationScratch, runVerifyStudy, verifyStudyOptionsSchema, verifyStudyResultSchema} from './verify-study.mjs';
import {validateSchema} from '../../../shared/representation/tools/v1/schemas.mjs';

assert.equal(process.platform, 'linux', 'Use the managed Linux container');
const parent = '/opt/mityane-geometrics/operations/counterfactuals/verification-tests';
fs.mkdirSync(parent, {recursive: true});
const directory = fs.mkdtempSync(path.join(parent, 'integrity-'));
let sequence = 0;
function fixture() {
  const attempt = path.join(directory, `fixture-${++sequence}`); fs.mkdirSync(attempt);
  fs.mkdirSync(path.join(attempt, 'source'));
  const rows = [];
  const put = (name, value) => {
    const bytes = Buffer.isBuffer(value) ? value : jsonBytes(value);
    fs.writeFileSync(path.join(attempt, name), bytes, {flag: 'wx'});
    const row = {path: name, bytes: bytes.length, sha256: sha256(bytes)}; rows.push(row); return row;
  };
  const source = put('source/example.txt', Buffer.from('Independent artificial test evidence\n'));
  const sources = [{...source, path: 'example.txt'}];
  const config = put('configuration.json', {study_id: 'artificial-integrity-test', claim_level: 'controlled_model_response'});
  const permissions = put('permissions.json', {scope: 'Engineering integrity checks only'});
  const freeze = put('freeze.json', {study_id: 'artificial-integrity-test', assessment_opened: false,
    configuration_sha256: config.sha256, permissions_sha256: permissions.sha256, sources,
    source_id: sha256(jsonBytes(sources))});
  const model = put('model.tsv', Buffer.from('Independently supplied fixture, never fitted\n'));
  const seal = put('model-seal.json', {freeze_sha256: freeze.sha256, assessment_opened: false,
    models_and_fit_inputs: [model]});
  const audit = put('audit.json', {valid: true, scope: 'Artificial integrity fixture; no scientific verdict'});
  put('assessment-open.json', {freeze_sha256: freeze.sha256, model_seal_sha256: seal.sha256});
  const manifest = jsonBytes({files: [...rows]}); fs.writeFileSync(path.join(attempt, 'file-manifest.json'), manifest);
  const final = jsonBytes({study_id: 'artificial-integrity-test', disposition: 'completed', assessment_opened: true,
    source_unchanged: true, freeze_sha256: freeze.sha256, audit_sha256: audit.sha256,
    model_seal_sha256: seal.sha256, file_manifest_sha256: sha256(manifest)});
  fs.writeFileSync(path.join(attempt, 'final.json'), final);
  return {attempt, profile: {final_sha256: sha256(final), original_attempt: attempt,
    contribution: false, seal: 'model-seal.json', seal_key: 'model_seal_sha256'}};
}

test('retained closure verifies exact final, freeze, seal, source capsule and every member', () => {
  const {attempt, profile} = fixture();
  const bound = verifyRetainedStudyFiles(attempt, profile);
  assert.equal(bound.final.study_id, 'artificial-integrity-test');
  assert.equal(bound.files_verified, 10);
  assert.equal(bound.inventory.get('model.tsv').sha256, bound.audit.valid &&
    sha256(fs.readFileSync(path.join(attempt, 'model.tsv'))));
});

test('changed numerical/model record is refused rather than rerun or regenerated', () => {
  const {attempt, profile} = fixture();
  fs.appendFileSync(path.join(attempt, 'model.tsv'), 'tampered\n');
  assert.throws(() => verifyRetainedStudyFiles(attempt, profile), /retained member changed/);
});

test('later reads and final checks reject a changed-and-restored retained file', async () => {
  const {attempt, profile} = fixture();
  const bound = verifyRetainedStudyFiles(attempt, profile);
  const original = bound.read('model.tsv');
  // The container filesystem may coalesce sub-millisecond metadata updates.
  await new Promise(resolve => setTimeout(resolve, 20));
  fs.writeFileSync(path.join(attempt, 'model.tsv'), Buffer.from('Altered record during verification\n'));
  fs.writeFileSync(path.join(attempt, 'model.tsv'), original);
  assert.equal(sha256(fs.readFileSync(path.join(attempt, 'model.tsv'))), bound.inventory.get('model.tsv').sha256,
    'content has been restored, so content hash alone cannot catch the change');
  assert.throws(() => bound.read('model.tsv'), /file changed after admission/);
  assert.throws(() => bound.recheck(), /file changed after admission/);
});

test('discoverable schemas describe the complete public verification receipt', () => {
  assert.equal(verifyStudyOptionsSchema.$schema, 'https://json-schema.org/draft/2020-12/schema');
  assert.equal(verifyStudyResultSchema.$schema, verifyStudyOptionsSchema.$schema);
  assert.notEqual(verifyStudyResultSchema.$id, verifyStudyOptionsSchema.$id);
  const counts = {requested: 3, succeeded: 2, abstained: 1, failed: 0, unresolved: 0};
  const result = {schema_version: 'mityane.hx06.retained-study-verification/v1', status: 'completed',
    study_id: 'OR06-REP-CONTRIB-001', report_file: '/explicit/verification.json', summary: counts,
    current_check_counts: {...counts, requested: 2, abstained: 0}, retained_only_count: 1,
    recomputed_saved_truth_metrics: {'assessment/polynomial/control': {source_units: 1, source_unit_mae: 0.1,
      aggregation: 'Engineering schema fixture'}}, archived_metrics: {}, archived_metrics_note: 'Archived only'};
  assert.doesNotThrow(() => validateSchema(verifyStudyResultSchema, result));
  assert.throws(() => validateSchema(verifyStudyResultSchema, {...result, retained_only_count: -1}), /minimum/);
  assert.throws(() => validateSchema(verifyStudyResultSchema, {...result, current_check_counts: {...counts, failed: -1}}), /minimum/);
});

test('current-check scratch must match cached bytes and rejects changed-and-restored records', async () => {
  const scratch = path.join(directory, `scratch-${++sequence}`); fs.mkdirSync(scratch);
  const bytes = Buffer.from('Recorded response fixture\n'), filename = path.join(scratch, 'response.json');
  fs.writeFileSync(filename, bytes);
  const expected = [{path: 'response.json', bytes: bytes.length, sha256: sha256(bytes)}];
  const recheck = bindVerificationScratch(scratch, expected);
  assert.doesNotThrow(recheck);
  await new Promise(resolve => setTimeout(resolve, 20));
  fs.writeFileSync(filename, Buffer.from('A different recorded response\n'));
  fs.writeFileSync(filename, bytes);
  assert.throws(recheck, /file changed after admission/);
  const wrong = path.join(directory, `scratch-${++sequence}`); fs.mkdirSync(wrong);
  fs.writeFileSync(path.join(wrong, 'response.json'), 'Never matches the cached payload');
  assert.throws(() => bindVerificationScratch(wrong, expected), /scratch bytes changed/);
});

test('missing retained fixtures and extraneous members are refused', () => {
  const absent = fixture(); fs.unlinkSync(path.join(absent.attempt, 'model.tsv'));
  assert.throws(() => verifyRetainedStudyFiles(absent.attempt, absent.profile), /missing or extraneous/);
  const extra = fixture(); fs.writeFileSync(path.join(extra.attempt, 'new-assessment.tsv'), 'not a retained fixture');
  assert.throws(() => verifyRetainedStudyFiles(extra.attempt, extra.profile), /missing or extraneous/);
});

test('redirected capsule and replaced final anchor are refused', () => {
  const redirected = fixture();
  fs.unlinkSync(path.join(redirected.attempt, 'source/example.txt'));
  fs.symlinkSync(path.join(redirected.attempt, 'model.tsv'), path.join(redirected.attempt, 'source/example.txt'));
  assert.throws(() => verifyRetainedStudyFiles(redirected.attempt, redirected.profile), /redirect/);
  const changed = fixture(); fs.appendFileSync(path.join(changed.attempt, 'final.json'), '\n');
  assert.throws(() => verifyRetainedStudyFiles(changed.attempt, changed.profile), /changed retained final identity/);
});

test('public helper cannot accept a caller-authored study or replacement trust anchor', async () => {
  const {attempt} = fixture();
  await assert.rejects(runVerifyStudy({attempt, output: '/mityane/.temp/operations/counterfactuals/never-published', operations: {
    check() { throw new Error('must never execute'); },
  }}), /unsupported retained study identity/);
  assert.equal(fs.existsSync('/mityane/.temp/operations/counterfactuals/never-published'), false);
});
