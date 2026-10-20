// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Metadata fixtures only. Original small JSONs are copied as immutable byte
// fixtures; no saved numerical evaluation or model is executed.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {preserve, requireStoredCoverage, verifyBoundedClassification, FINAL_TARGETS, SMALL_ORIGINALS} from './preservation.mjs';
import {ORIGINAL_PROPOSALS} from './stage.mjs';
import {jsonBytes, parseJSON, sha256} from '../../../../shared/artifacts/node/v1/files.mjs';

const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const SHARED = 'code/shared/artifacts/node/v1/files.mjs';
const REPOSITORY = fileURLToPath(new URL('../../../../../', import.meta.url));
const COVERAGE = {scope_cleanup_whole_before: 14, final_cleaned_guides: 14, prior_six_live_files: 6,
  intermediate_guides: 4, paused_pointer_before: 2, paused_pointer_after: 2, source_note_aliases: 4,
  external_portable_checkpoints: 57, historical_critical_checkpoints: 119, canonical_v14_checkpoints: 5,
  complete_critical_checkpoints: 124, paused_finalization_payload: 114, card_inputs: 49,
  frozen_scientific_sources: 145, main_binaries: 7, complete_main_mirror: 1149,
  v13_singleton_registered_before_repair_traversal: true, paused_payload_exact_member_set: true,
  earlier_preservation_generations_registered_before_traversal: 8, transitive_documents: 300, transitive_identities: 999};
function fixture(t, {reusedBulk = false} = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cpb-v14-preservation-fixture-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const relative = '.temp/embeddings/closeout/cpb-v14/attempt-001', attempt = path.join(root, relative);
  fs.mkdirSync(attempt, {recursive: true});
  const write = (name, value) => {
    const bytes = Buffer.isBuffer(value) ? value : jsonBytes(value), filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), {recursive: true}); fs.writeFileSync(filename, bytes);
    return {path: name, bytes: bytes.length, sha256: sha256(bytes)};
  };
  const read = name => parseJSON(fs.readFileSync(path.join(root, name)));
  const sources = ['preservation.mjs', 'stage.mjs', 'apply.mjs', 'aliases.mjs', 'review.mjs', 'inspect.mjs', 'scope.mjs'].map(name => `${PROTOCOL}/${name}`).concat(SHARED)
    .map(name => write(name, fs.readFileSync(path.join(REPOSITORY, name))));
  if (reusedBulk) sources.push(write(`${PROTOCOL}/repair.mjs`, Buffer.from('// Fixture-only reviewed repair source.\n')));
  for (const original of SMALL_ORIGINALS) write(original.path, fs.readFileSync(path.join(REPOSITORY, original.path)));
  const sourceReview = write(`${relative}/source-review-preservation.json`, {passed: true, files: sources});
  const oldSourceReview = write(`${relative}/source-review-old.json`, {passed: true, files: sources});
  const finalData = [
    {fresh_research_candidate: {instance: 'CPB-v14.market-01'}, clean_causal_training_baseline: {instance: 'CPB-v3.market-01'}, active_application: null},
    {instance: 'CPB-v14.market-01', clean_baseline: 'CPB-v3.market-01', application_selection: null,
      quality: {numeric_passed: 3, numeric_total: 22, strict_new_scoped_quality_leaders: 0}, advancement_passed: false,
      seed: 18446744073709551615n},
    {scientific_conclusion: {mityane_application_selection: null}, future_receipt_locations: {
      application: `${relative}/application.json`, preservation: `${relative}/preservation.json`}},
    ...FINAL_TARGETS.slice(3).map(name => ({fixture_final_document: name})),
  ];
  const before = FINAL_TARGETS.map((name, index) => index === 1 || index === 2 ? {path: name, exists: false} : {
    path: name, exists: true, ...(() => {
      const backup = write(`${relative}/before/${name}`, {fixture_prior: name});
      return {bytes: backup.bytes, sha256: backup.sha256, whole_before_copy: backup.path};
    })(),
  });
  const finals = FINAL_TARGETS.map((name, index) => write(name, finalData[index]));
  const admissionData = {kind: 'cpb_v14_postmigration_closeout_admission_v1', artifact_root: relative,
    scope: {compatible_reviewed_catalog_instance_summary_application: true, new_model_training: false,
      old_encoder_or_head_replay: false, head_refits: false}, target_observations: before};
  const admission = write(`${relative}/admission.json`, admissionData);
  const bulkAliases = Array.from({length: 145}, (_, index) => ({
    original: {repository: 'cuwacunu_embedding', path: `output/fixture-bulk/${index}.bin`, bytes: 1, sha256: 'f'.repeat(64)},
    current_path: `fixture-bulk-not-read/${index}.bin`, observed: {bytes: 1, sha256: 'f'.repeat(64)},
    provenance: [{fixture: true}],
  }));
  const aliasesData = {kind: 'cpb_v14_historical_identity_alias_manifest_v1', coverage: COVERAGE,
    aliases: bulkAliases, protocol_identity: sources.find(row => row.path.endsWith('/aliases.mjs'))};
  const aliases = write(`${relative}/aliases.json`, aliasesData);
  const aliasVerificationData = {kind: 'cpb_v14_historical_identity_alias_verification_v1', passed: true, status: 'passed',
    created_utc: '2026-10-18T18:00:00.000Z', coverage: COVERAGE, aliases_verified: bulkAliases.length,
    failures: [], alias_manifest: aliases, protocol_identity: aliasesData.protocol_identity,
    shared_parser_identity: sources.find(row => row.path === SHARED)};
  if (reusedBulk) {
    const baseManifest = write(`${relative}/aliases-base.json`, {...aliasesData, created_utc: '2026-10-18T17:59:00.000Z'});
    const baseFailures = [{identity: bulkAliases[1].original, error: 'Fixture missing alias in original failed generation'}];
    const baseVerification = write(`${relative}/alias-verification-base.json`, {status: 'failed', passed: false,
      created_utc: '2026-10-18T18:00:00.000Z', alias_manifest: baseManifest, failures: baseFailures});
    Object.assign(aliasVerificationData, {created_utc: '2026-10-18T18:30:00.000Z',
      bulk_observation_reused: true, all_aliases_freshly_hashed_this_generation: false,
      base_pair: {manifest: baseManifest, verification: baseVerification},
      original_bulk_observation: {manifest_created_utc: '2026-10-18T17:59:00.000Z',
        verification_created_utc: '2026-10-18T18:00:00.000Z', verification_status: 'failed',
        protocol_identity: aliasesData.protocol_identity, source_review: oldSourceReview,
        limitation: 'Fixture original observation; no per-file timestamps.'},
      original_failures: baseFailures, base_failure_accounting: [{index: 0, identity: bulkAliases[1].original, required: false,
        resolved: false, disposition: 'outside_declared_recursive_reverification'}],
      reused_aliases: 145, new_physical_hash_reads: 1, metadata_documents_rewalked: 20});
  }
  const typedCoverage = {...COVERAGE, prior_v13_protected_pins: 8506, prior_v13_protected_inventories: 30,
    prior_v13_current_protected_inventories: 4, prior_v13_source_files: 143, main_dependency_files: 11,
    proposal_outputs: 3, proposal_receipts: 1, current_v14_external_input_pins: 20,
    selected_v3_source_files_each_repo: 47, selected_v3_source_bindings: 94};
  const originalFailures = aliasVerificationData.original_failures ?? [];
  const boundedScopeData = {kind: 'cpb_v14_typed_bounded_scope_v1', status: 'declared_metadata_scope',
    admission, source_review: sourceReview, base_pair: aliasVerificationData.base_pair ?? null, coverage: typedCoverage,
    required_pins: [{original: bulkAliases[0].original, required_by: [{manifest: admission, pointer: '/fixture', selector: 'fixture'}],
      known_locators: [bulkAliases[0].current_path]}],
    control_documents: [{role: 'fixture_control', original: {repository: 'Mityane-geometrics', ...admission},
      current_path: admission.path, observed: {bytes: admission.bytes, sha256: admission.sha256}, selectors: []}], mapping_documents: [],
    historical_failure_classification: originalFailures.map((row, index) => ({original_failure_index: index,
      original_identity: row.identity, record_qualified_identity: row.identity, required_in_scope: false,
      disposition: 'outside_declared_traversal', prior_failure_retained: true}))};
  const boundedScope = write(`${relative}/bounded-scope.json`, boundedScopeData);
  const scopeReviewData = {kind: 'cpb_v14_bounded_scope_review_v1', status: 'passed', passed: true, scope: boundedScope,
    required_pin_count: 1, coverage: typedCoverage, source_review: sourceReview, source_closure: sources};
  const scopeReview = write(`${relative}/scope-review.json`, scopeReviewData);
  Object.assign(aliasVerificationData, {bounded_scope_verified: true, bounded_scope: boundedScope, scope_review: scopeReview,
    required_pin_count: 1, required_pins_verified: 1, whole_reference_graph_verified: false, all_historical_refs_resolved: false,
    verification_scope: 'declared_bounded_v14_metadata_closeout', original_failures: originalFailures,
    base_failure_accounting: aliasVerificationData.base_failure_accounting ?? [],
    outside_declared_recursive_reverification_count: originalFailures.length, record_schema_overinterpretation_count: 0});
  const aliasVerification = write(`${relative}/alias-verification.json`, aliasVerificationData);
  const proposalReviewData = {kind: 'mityane_cpb_v14_independent_metadata_proposal_review_v1', passed: true, status: 'passed',
    source_closure: sources, alias_manifest: aliases, alias_verification: aliasVerification,
    inputs: ORIGINAL_PROPOSALS, card: SMALL_ORIGINALS.find(row => row.role === 'original_card'),
    generator_receipt: SMALL_ORIGINALS.find(row => row.role === 'original_proposal_receipt')};
  const proposalReview = write(`${relative}/proposal-review.json`, proposalReviewData);
  const snapshot = write(`${relative}/inputs.json`, {targets: before});
  const outputs = finals.slice(0, 3).map((pin, index) => {
    const staged = write(`${relative}/staged/${pin.path}`, finalData[index]);
    return {...pin, staged_path: staged.path, expected_before: before[index]};
  });
  const manifestData = {kind: 'cpb_v14_postmigration_staged_change_manifest_v1', protocol: PROTOCOL, outputs,
    immutable_preapplication_inputs: {admission, inputs: snapshot, aliases, alias_verification: aliasVerification,
      proposal_review: proposalReview, source_review: oldSourceReview}};
  const manifest = write(`${relative}/staged-change-manifest.json`, manifestData);
  const stagedReviewData = {kind: 'cpb_v14_postmigration_staged_review_v1', passed: true, status: 'passed',
    manifest, source_closure: sources, outputs};
  const stagedReview = write(`${relative}/staged-review.json`, stagedReviewData);
  const lock = write('.temp/embeddings/closeout/cpb-v14/application-lock.json', {fixture: 'claim'});
  const start = write(`${relative}/application-start.json`, {fixture: 'start'});
  const applicationData = {kind: 'cpb_v14_postmigration_metadata_application_v1', passed: true, status: 'passed',
    inputs: {manifest, staged_review: stagedReview, application_source_review: oldSourceReview},
    expected_outputs: outputs, application_source_closure: sources, staged_review_source_closure: sources,
    shared_application_claim: lock, application_start: start,
    completed_write_operations: finals.slice(0, 3), before_copies: [{target: FINAL_TARGETS[0],
      path: before[0].whole_before_copy, bytes: before[0].bytes, sha256: before[0].sha256}],
    observed_after: finals.slice(0, 3).map(pin => ({...pin, matches_staged: true})),
    preservation_receipt_location: `${relative}/preservation.json`};
  const application = write(`${relative}/application.json`, applicationData);
  const docReviewData = {kind: 'cpb_v14_postmigration_documentary_review_v1', passed: true, status: 'passed',
    independent_review: true, admission, application,
    outputs: finals.slice(3).map((pin, index) => ({...pin, expected_before: before[index + 3]}))};
  const docReview = write(`${relative}/doc-review.json`, docReviewData);
  const fixtureLog = write(`${relative}/fixture.log`, Buffer.from('Fixture engineering checks passed.\n'));
  const fixtureResult = write(`${relative}/tests.json`, {passed: true, tests: 10, pass: 10, fail: 0, source_review: sourceReview, log: fixtureLog});
  const semanticData = {kind: 'cpb_v14_postmigration_final_semantic_review_v1', status: 'passed', passed: true,
    application, staged_review: stagedReview, manifest, outputs, all_adaptations_independently_checked: true,
    unchanged_scientific_data: true, source: sources.find(row => row.path === `${PROTOCOL}/inspect.mjs`),
    source_review: sourceReview, source_closure: sources};
  const semanticResult = write(`${relative}/final-semantic-review.json`, semanticData);
  const validationData = {kind: 'cpb_v14_postmigration_final_validation_v1', passed: true, status: 'passed',
    application, source_review: sourceReview, reviewed_outputs: finals,
    checks: ['focused_fixtures', 'record_semantics', 'local_links', 'dashboard_refresh', 'dashboard_freshness'].map(id => ({
      id, passed: true, exit_code: 0, command: ['fixture-only', id], log: fixtureLog,
      ...(id === 'focused_fixtures' ? {receipt: fixtureResult} : {}),
      ...(id === 'record_semantics' ? {receipt: semanticResult} : {}),
    }))};
  const validation = write(`${relative}/validation.json`, validationData);
  const options = {root, attempt, admissionPath: admission.path, applicationPath: application.path,
    stagedReviewPath: stagedReview.path, sourceReviewPath: sourceReview.path,
    proposalReviewPath: proposalReview.path, aliasManifestPath: aliases.path,
    aliasVerificationPath: aliasVerification.path, docReviewPath: docReview.path, validationPath: validation.path};
  return {root, attempt, relative, write, read, options, finals, outputs, before, sources,
    admissionData, aliasesData, aliasVerificationData, proposalReviewData, manifestData, stagedReviewData,
    applicationData, docReviewData, validationData, fixtureLog, sourceReview, semanticData, semanticResult, boundedScopeData, scopeReviewData,
    assertFailure(error) {
      assert.ok(error); assert.equal(fs.existsSync(path.join(attempt, 'preservation.json')), false);
      const result = read(error.receipt_path); assert.equal(result.passed, false); assert.equal(result.status, 'failed');
      for (const pin of finals) assert.equal(sha256(fs.readFileSync(path.join(root, pin.path))), pin.sha256);
      return result;
    }};
}
function failed(call) { try { call(); } catch (error) { return error; } assert.fail('expected preservation failure'); }

test('preserves all seven actual outputs and exact before generations without rereading bulk aliases', t => {
  const f = fixture(t), result = preserve(f.options);
  assert.equal(result.passed, true); assert.equal(result.final_outputs.length, 7);
  assert.equal(result.whole_before_copies_or_absences.length, 7);
  assert.equal(result.whole_before_copies_or_absences.filter(row => row.existed_before).length, 5);
  assert.equal(result.freshly_verified_small_originals.length, 7);
  assert.equal(result.historical_bulk_verification.reused_passed_verification, true);
  assert.equal(result.historical_bulk_verification.fresh_bulk_rehash_performed, false);
  assert.equal(result.historical_bulk_verification.observation_utc, '2026-10-18T18:00:00.000Z');
  assert.equal(fs.existsSync(path.join(f.root, 'fixture-bulk-not-read')), false);
  assert.equal(result.scientific_execution_performed, false);
  assert.equal(result.whole_reference_graph_verified, false);
  assert.equal(result.bounded_verification.required_pins_verified, 1);
  assert.equal(f.read(FINAL_TARGETS[1]).seed, 18446744073709551615n);
  assert.equal(Object.hasOwn(result, 'sha256'), false);
  assert.equal(Object.hasOwn(result, 'preservation_sha256'), false);
  const bytes = fs.readFileSync(path.join(f.attempt, 'preservation.json'));
  assert.throws(() => preserve(f.options), /already exists/);
  assert.ok(fs.readFileSync(path.join(f.attempt, 'preservation.json')).equals(bytes));
});

test('repair receipt retains failed base outcome and earlier observation times honestly', t => {
  const f = fixture(t, {reusedBulk: true}), result = preserve(f.options), bulk = result.historical_bulk_verification;
  assert.equal(bulk.observation_utc, null);
  assert.equal(bulk.receipt_created_utc, '2026-10-18T18:30:00.000Z');
  assert.equal(bulk.original_bulk_observation.verification_created_utc, '2026-10-18T18:00:00.000Z');
  assert.equal(bulk.original_bulk_observation.verification_status, 'failed');
  assert.equal(bulk.bulk_observation_reused_by_alias_generation, true);
  assert.equal(bulk.all_aliases_freshly_hashed_this_generation, false);
  assert.equal(f.read(bulk.base_pair.verification.path).status, 'failed');
  assert.equal(bulk.original_failures.length, 1);
  assert.equal(result.bounded_verification.outside_declared_recursive_reverification_count, 1);
  assert.equal(bulk.base_failure_accounting[0].resolved, false);
  assert.equal(fs.existsSync(path.join(f.root, 'fixture-bulk-not-read')), false);
});

test('bounded classification cannot convert an outside failure into a verified required original', t => {
  const f = fixture(t, {reusedBulk: true});
  const result = verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData);
  assert.equal(result.whole_reference_graph_verified, false);
  assert.equal(result.original_failed_rows, 1);
  f.aliasVerificationData.base_failure_accounting[0].resolved = true;
  assert.throws(() => verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData), /must remain unresolved/);
  f.aliasVerificationData.base_failure_accounting[0].resolved = false;
  f.aliasVerificationData.base_failure_accounting = [];
  assert.throws(() => verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData), /every original failed row/);
});

test('a required identity must verify and repository corrections preserve the original failure', t => {
  const f = fixture(t, {reusedBulk: true}), classification = f.boundedScopeData.historical_failure_classification[0],
    accounting = f.aliasVerificationData.base_failure_accounting[0], required = f.boundedScopeData.required_pins[0].original;
  f.boundedScopeData.required_pins[0].original = classification.original_identity;
  classification.required_in_scope = true; classification.disposition = 'required_current_scope';
  accounting.required = true; accounting.disposition = 'required_identity_unresolved';
  f.aliasVerificationData.outside_declared_recursive_reverification_count = 0;
  assert.throws(() => verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData), /must actually resolve/);
  f.boundedScopeData.required_pins[0].original = required;
  classification.record_qualified_identity = required; classification.disposition = 'record_schema_overinterpretation';
  Object.assign(accounting, {required: false, resolved: false, disposition: 'record_schema_overinterpretation',
    record_qualified_identity: required, record_qualified_required: true, record_qualified_verified: true});
  f.aliasVerificationData.record_schema_overinterpretation_count = 1;
  const result = verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData);
  assert.equal(result.record_schema_overinterpretation_count, 1);
  assert.equal(accounting.resolved, false);
  accounting.record_qualified_verified = false;
  assert.throws(() => verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData), /cannot relabel/);
});

test('bounded declaration review and counts must match the required set without claiming whole closure', t => {
  const f = fixture(t);
  for (const change of [
    value => { value.required_pins_verified = 0; },
    value => { value.bounded_scope_verified = false; },
    value => { value.whole_reference_graph_verified = true; },
    value => { value.all_historical_refs_resolved = true; },
  ]) {
    const value = parseJSON(jsonBytes(f.aliasVerificationData)); change(value);
    assert.throws(() => verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, value, f.aliasesData), /complete bounded/);
  }
  const changedReview = {...f.scopeReviewData, required_pin_count: 0};
  assert.throws(() => verifyBoundedClassification(f.boundedScopeData, changedReview, f.aliasVerificationData, f.aliasesData), /complete bounded/);
});

test('explicit null aggregate identity cannot fall back to a historical file tuple', t => {
  const f = fixture(t, {reusedBulk: true}), classification = f.boundedScopeData.historical_failure_classification[0],
    accounting = f.aliasVerificationData.base_failure_accounting[0];
  Object.assign(classification, {record_qualified_identity: null, required_in_scope: false, disposition: 'record_schema_overinterpretation'});
  Object.assign(accounting, {record_qualified_identity: null, record_qualified_required: false,
    record_qualified_verified: false, required: false, resolved: false, disposition: 'record_schema_overinterpretation'});
  f.aliasVerificationData.outside_declared_recursive_reverification_count = 0;
  f.aliasVerificationData.record_schema_overinterpretation_count = 1;
  const result = verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData);
  assert.equal(result.record_schema_overinterpretation_count, 1);
  accounting.record_qualified_identity = accounting.identity;
  assert.throws(() => verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData), /cannot relabel/);
  accounting.record_qualified_identity = null; delete classification.record_qualified_identity;
  assert.throws(() => verifyBoundedClassification(f.boundedScopeData, f.scopeReviewData, f.aliasVerificationData, f.aliasesData), /explicitly distinguish/);
});

test('failed application cannot become a passed preservation and failure output is retained', t => {
  const f = fixture(t); f.applicationData.passed = false; f.applicationData.status = 'partial_failure';
  f.write(f.options.applicationPath, f.applicationData);
  const first = failed(() => preserve(f.options)); f.assertFailure(first);
  const bytes = fs.readFileSync(path.join(f.root, first.receipt_path));
  const second = failed(() => preserve(f.options)); f.assertFailure(second);
  assert.notEqual(first.receipt_path, second.receipt_path);
  assert.ok(fs.readFileSync(path.join(f.root, first.receipt_path)).equals(bytes));
});

test('missing stored historical coverage or failed bulk verification blocks preservation', t => {
  const f = fixture(t); f.aliasVerificationData.passed = false;
  f.write(f.options.aliasVerificationPath, f.aliasVerificationData);
  const error = failed(() => preserve(f.options)); f.assertFailure(error);
  assert.match(error.message, /input generation conflict|bulk alias verification|bound input changed: .*\/alias-verification\.json/);
  // A second isolated fixture tests the exact declared coverage requirement.
  const other = fixture(t); other.aliasesData.coverage = {...COVERAGE, complete_critical_checkpoints: 123};
  other.write(other.options.aliasManifestPath, other.aliasesData);
  const second = failed(() => preserve(other.options)); other.assertFailure(second);
  assert.match(second.message, /input generation conflict|coverage|alias|bound input changed: .*\/aliases\.json/);
});

test('stored coverage needs every exact count and both complete historical traversal flags', () => {
  requireStoredCoverage({coverage: COVERAGE}, {coverage: COVERAGE});
  for (const [key, value] of Object.entries(COVERAGE)) {
    const changed = {...COVERAGE, [key]: typeof value === 'boolean' ? false : key.startsWith('transitive_') ? 0 : value - 1};
    assert.throws(() => requireStoredCoverage({coverage: changed}, {coverage: changed}), /coverage/);
  }
  assert.throws(() => requireStoredCoverage({coverage: COVERAGE}, {coverage: {...COVERAGE, frozen_scientific_sources: 144}}), /coverage conflict/);
});

test('independent documentary review must cover exactly all four admitted final documents', t => {
  const f = fixture(t); f.docReviewData.outputs.pop(); f.write(f.options.docReviewPath, f.docReviewData);
  const error = failed(() => preserve(f.options)); f.assertFailure(error); assert.match(error.message, /exact four documentary/);
});

test('changed final document or before-copy blocks completion without writing live data', t => {
  const f = fixture(t), filename = path.join(f.root, FINAL_TARGETS[3]), bytes = fs.readFileSync(filename);
  fs.appendFileSync(filename, 'changed');
  const error = failed(() => preserve(f.options));
  assert.match(error.message, /documentary output changed/);
  assert.equal(fs.existsSync(path.join(f.attempt, 'preservation.json')), false);
  assert.ok(fs.readFileSync(filename).equals(Buffer.concat([bytes, Buffer.from('changed')])));
  fs.writeFileSync(filename, bytes);
  fs.appendFileSync(path.join(f.root, f.before[0].whole_before_copy), 'changed');
  const second = failed(() => preserve(f.options)); f.assertFailure(second); assert.match(second.message, /whole-before copy changed/);
});

test('validation needs actual passed freshness, bound logs and all seven final output hashes', t => {
  const f = fixture(t); f.validationData.checks.find(row => row.id === 'dashboard_freshness').passed = false;
  f.write(f.options.validationPath, f.validationData);
  const error = failed(() => preserve(f.options)); f.assertFailure(error); assert.match(error.message, /dashboard_freshness/);
  f.validationData.checks.find(row => row.id === 'dashboard_freshness').passed = true;
  f.validationData.reviewed_outputs = f.validationData.reviewed_outputs.slice(0, 6);
  f.write(f.options.validationPath, f.validationData);
  const second = failed(() => preserve(f.options)); f.assertFailure(second); assert.match(second.message, /all seven actual/);
});

test('final semantic receipt is mandatory and must pass with exact application/output bindings', t => {
  const f = fixture(t), check = f.validationData.checks.find(row => row.id === 'record_semantics');
  delete check.receipt; f.write(f.options.validationPath, f.validationData);
  const missing = failed(() => preserve(f.options)); f.assertFailure(missing); assert.match(missing.message, /record semantics need/);
  const unrelated = {...f.semanticData, kind: 'unrelated_passed_receipt'};
  check.receipt = f.write(f.semanticResult.path, unrelated); f.write(f.options.validationPath, f.validationData);
  const wrongKind = failed(() => preserve(f.options)); f.assertFailure(wrongKind); assert.match(wrongKind.message, /semantic review schema/);
  check.receipt = f.write(f.semanticResult.path, {...f.semanticData, application: {...f.semanticData.application, sha256: '0'.repeat(64)}});
  f.write(f.options.validationPath, f.validationData);
  const wrongApplication = failed(() => preserve(f.options)); f.assertFailure(wrongApplication); assert.match(wrongApplication.message, /semantic review application conflict/);
  check.receipt = f.write(f.semanticResult.path, {...f.semanticData, outputs: f.outputs.slice(0, 2)});
  f.write(f.options.validationPath, f.validationData);
  const wrongOutputs = failed(() => preserve(f.options)); f.assertFailure(wrongOutputs); assert.match(wrongOutputs.message, /independently verify every actual output/);
});

test('stale source review, original proposal tampering and changed validation logs are rejected', t => {
  const f = fixture(t), source = path.join(f.root, `${PROTOCOL}/preservation.mjs`), sourceBytes = fs.readFileSync(source);
  fs.appendFileSync(source, '\n// changed\n');
  const first = failed(() => preserve(f.options)); f.assertFailure(first); assert.match(first.message, /reviewed source changed/);
  fs.writeFileSync(source, sourceBytes);
  const proposal = path.join(f.root, ORIGINAL_PROPOSALS[0].path), proposalBytes = fs.readFileSync(proposal);
  fs.appendFileSync(proposal, ' ');
  const second = failed(() => preserve(f.options)); f.assertFailure(second); assert.match(second.message, /immutable original changed/);
  fs.writeFileSync(proposal, proposalBytes);
  fs.appendFileSync(path.join(f.root, f.fixtureLog.path), 'changed');
  const third = failed(() => preserve(f.options)); f.assertFailure(third); assert.match(third.message, /validation log changed/);
});

test('another review generation and a claimed passing output with mismatched application observations are rejected', t => {
  const f = fixture(t); f.docReviewData.application.sha256 = '0'.repeat(64);
  f.write(f.options.docReviewPath, f.docReviewData);
  const first = failed(() => preserve(f.options)); f.assertFailure(first); assert.match(first.message, /documentary review application conflict/);
  const other = fixture(t); other.applicationData.observed_after[0].matches_staged = false;
  other.write(other.options.applicationPath, other.applicationData);
  const second = failed(() => preserve(other.options)); other.assertFailure(second); assert.match(second.message, /all final output matches/);
});
