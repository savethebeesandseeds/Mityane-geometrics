// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Final metadata preservation only: no tensor reads or second bulk-data scan.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {ORIGINAL_PROPOSALS, TARGETS} from './stage.mjs';
import {admitFiles, canonicalDirectory, hashAdmitted, inside, jsonBytes, parseJSON,
  readAdmitted, requireCondition as require, safeRelative, sha256, writeNew}
  from '../../../../shared/artifacts/node/v1/files.mjs';

const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const SHARED = 'code/shared/artifacts/node/v1/files.mjs';
const PAYLOAD = 'code/embeddings/output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001/payload';
export const FINAL_TARGETS = Object.freeze([...TARGETS, 'doc/EMBEDDINGS.md',
  'code/embeddings/encoders/causal_patch_bottleneck_mae/README.md',
  'web/dashboard/heroes/17-encoder-cpb.json', 'web/dashboard/public/index.html']);
export const SMALL_ORIGINALS = Object.freeze([...ORIGINAL_PROPOSALS,
  {role: 'original_card', path: `${PAYLOAD}/card-source-001/finalization-card.json`,
    bytes: 18917, sha256: '00d8b0c1a7c7aa2f13c1733a6160a5a1ac56f9b00e29e3e01836d89bf6575629'},
  {role: 'original_proposal_receipt', path: `${PAYLOAD}/proposals-001/finalization-proposal.json`,
    bytes: 22518, sha256: 'b7ad512c114c57731f08e7f88ce6b1cf28a19c9da3d5624c3619a8ee359007a4'},
  {role: 'archived_ledger', path: '.archive/embeddings/pre-cleanup-2026-10-18/doc/clean_causal_encoder_research_ledger.json',
    bytes: 1746343, sha256: '4ac9a0a3b73225b3ca33915176a2785dd23524d2464021a421a99c33f5464270'},
  {role: 'unchanged_descriptor', path: 'code/embeddings/encoders/causal_patch_bottleneck_mae/versions/CPB-v14.json',
    bytes: 6439, sha256: '4572c0c7b55604ad02822462bdd757da7c8f3f2618e1ee0ef7f6ba43b4bbf185'},
]);
const COVERAGE = {scope_cleanup_whole_before: 14, final_cleaned_guides: 14, prior_six_live_files: 6,
  intermediate_guides: 4, paused_pointer_before: 2, paused_pointer_after: 2, source_note_aliases: 4,
  external_portable_checkpoints: 57, historical_critical_checkpoints: 119, canonical_v14_checkpoints: 5,
  complete_critical_checkpoints: 124, paused_finalization_payload: 114, card_inputs: 49,
  frozen_scientific_sources: 145, main_binaries: 7, complete_main_mirror: 1149,
  v13_singleton_registered_before_repair_traversal: true, paused_payload_exact_member_set: true,
  earlier_preservation_generations_registered_before_traversal: 8};
const TYPED_COVERAGE = {...COVERAGE, prior_v13_protected_pins: 8506, prior_v13_protected_inventories: 30,
  prior_v13_current_protected_inventories: 4, prior_v13_source_files: 143, main_dependency_files: 11,
  proposal_outputs: 3, proposal_receipts: 1, current_v14_external_input_pins: 20,
  selected_v3_source_files_each_repo: 47, selected_v3_source_bindings: 94};
const SOURCES = ['preservation.mjs', 'stage.mjs', 'apply.mjs', 'aliases.mjs', 'review.mjs', 'inspect.mjs', 'scope.mjs']
  .map(name => `${PROTOCOL}/${name}`).concat(SHARED);
const CHECKS = ['focused_fixtures', 'record_semantics', 'local_links', 'dashboard_refresh', 'dashboard_freshness'];
const equal = (left, right) => jsonBytes(left).equals(jsonBytes(right));
const absolute = (root, relative) => path.join(root, safeRelative(relative));
function exists(filename) {
  try { fs.lstatSync(filename); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}
function rootPin(root, pin) {
  require(pin && typeof pin.path === 'string', 'explicit exact receipt pin required');
  if (!path.isAbsolute(pin.path)) return pin;
  require(inside(root, pin.path), 'receipt pin escapes the repository');
  return {...pin, path: path.relative(root, pin.path).split(path.sep).join('/')};
}
function observedPin(root, relative) {
  const row = admitFiles([absolute(root, relative)], {root, singleLink: true})[0];
  return {path: relative, bytes: row.bytes, sha256: hashAdmitted(row)};
}
function boundJSON(root, relative) {
  const row = admitFiles([absolute(root, relative)], {root, singleLink: true})[0], bytes = readAdmitted(row);
  return {pin: {path: relative, bytes: bytes.length, sha256: sha256(bytes)}, data: parseJSON(bytes)};
}
function expectPin(root, actual, expected, message) {
  const pin = rootPin(root, expected);
  require(actual.path === pin.path && actual.bytes === pin.bytes && actual.sha256 === pin.sha256, message);
}
function passed(record, message) {
  require(record?.passed === true && (record.status === undefined || record.status === 'passed'), message);
}
function verifyPin(root, expected, message = 'bound input changed') {
  const pin = rootPin(root, expected), actual = observedPin(root, pin.path);
  expectPin(root, actual, pin, `${message}: ${pin.path}`); return actual;
}
function verifySources(root, rows, required = []) {
  require(Array.isArray(rows), 'explicit source closure required');
  const seen = new Set();
  for (const row of rows) {
    const pin = rootPin(root, row);
    require(!seen.has(pin.path), 'duplicate source closure member'); seen.add(pin.path);
    verifyPin(root, pin, 'reviewed source changed');
  }
  for (const name of required) require(seen.has(name), `reviewed source closure missing ${name}`);
}
export function requireStoredCoverage(aliases, verification) {
  require(equal(aliases.coverage, verification.coverage), 'alias manifest and verification coverage conflict');
  for (const [key, value] of Object.entries(COVERAGE)) require(aliases.coverage[key] === value, `missing stored alias coverage: ${key}`);
  for (const key of ['transitive_documents', 'transitive_identities']) require(Number.isSafeInteger(aliases.coverage[key]) && aliases.coverage[key] > 0,
    `positive stored transitive alias coverage required: ${key}`);
}
function identityKey(pin) {
  require(pin && ['cuwacunu_embedding', 'cuwacunu_dev', 'Mityane-geometrics'].includes(pin.repository) &&
    typeof pin.path === 'string' && pin.path.length > 0 && Number.isSafeInteger(pin.bytes) && pin.bytes >= 0 &&
    /^[0-9a-f]{64}$/.test(pin.sha256 ?? ''), 'explicit full repository/path/bytes/hash scope identity required');
  return JSON.stringify([pin.repository, pin.path, pin.bytes, pin.sha256]);
}
export function verifyBoundedClassification(scope, scopeReview, verification, aliases) {
  require(scope.kind === 'cpb_v14_typed_bounded_scope_v1' && scope.status === 'declared_metadata_scope' &&
    scopeReview.kind === 'cpb_v14_bounded_scope_review_v1', 'actual typed bounded scope and independent review required');
  passed(scopeReview, 'independent bounded scope review must have passed');
  require(equal(scope.coverage, scopeReview.coverage), 'scope and independent review coverage conflict');
  for (const [key, expected] of Object.entries(TYPED_COVERAGE)) require(scope.coverage[key] === expected, `typed scope coverage missing ${key}`);
  require(Array.isArray(scope.required_pins), 'explicit typed required-pin set required');
  const required = new Set();
  for (const row of scope.required_pins) {
    const key = identityKey(row.original);
    require(!required.has(key) && Array.isArray(row.required_by) && row.required_by.length > 0,
      'unique required identity with typed selector provenance required'); required.add(key);
  }
  require(verification.bounded_scope_verified === true && verification.verification_scope === 'declared_bounded_v14_metadata_closeout' &&
    verification.whole_reference_graph_verified === false && verification.all_historical_refs_resolved === false &&
    verification.required_pin_count === required.size && verification.required_pins_verified === required.size &&
    scopeReview.required_pin_count === required.size, 'complete bounded required-pin verification without a whole-graph claim required');
  const registered = new Set(aliases.aliases.map(row => identityKey(row.original)));
  for (const key of required) require(registered.has(key), `required bounded identity missing from alias manifest: ${key}`);
  const originals = verification.original_failures, accounting = verification.base_failure_accounting,
    classifications = scope.historical_failure_classification;
  require(Array.isArray(originals) && Array.isArray(accounting) && Array.isArray(classifications) &&
    accounting.length === originals.length && classifications.length === originals.length, 'every original failed row must retain exact bounded accounting');
  const indexed = new Map();
  for (const row of classifications) {
    const index = row.original_failure_index;
    require(Number.isInteger(index) && index >= 0 && index < originals.length && !indexed.has(index), 'one exact classification for every original failure index required');
    indexed.set(index, row);
  }
  const seen = new Set(); let outside = 0, overinterpretation = 0, verifiedOriginals = 0;
  for (const row of accounting) {
    const index = row.index;
    require(Number.isInteger(index) && index >= 0 && index < originals.length && !seen.has(index), 'one exact repair accounting row for every original failure required');
    seen.add(index);
    const classification = indexed.get(index), originalKey = identityKey(originals[index].identity);
    require(identityKey(row.identity) === originalKey && identityKey(classification.original_identity) === originalKey &&
      classification.prior_failure_retained === true, 'original failed identity/outcome must remain unchanged');
    require(Object.hasOwn(classification, 'record_qualified_identity'), 'scope must explicitly distinguish original, qualified and non-file aggregate identities');
    const qualified = classification.record_qualified_identity,
      qualifiedKey = qualified ? identityKey(qualified) : null, qualifiedRequired = qualifiedKey !== null && required.has(qualifiedKey);
    require(classification.required_in_scope === qualifiedRequired, 'scope classification contradicts actual typed required-pin membership');
    if (classification.disposition === 'record_schema_overinterpretation' || qualifiedKey !== originalKey) {
      require(row.required === false && row.resolved === false && row.disposition === 'record_schema_overinterpretation' &&
        Object.hasOwn(row, 'record_qualified_identity') && equal(row.record_qualified_identity, qualified) &&
        row.record_qualified_required === qualifiedRequired && (!qualifiedRequired || row.record_qualified_verified === true),
        'a context correction cannot relabel the original failed tuple as verified');
      overinterpretation++;
    } else if (required.has(originalKey)) {
      require(classification.disposition === 'required_current_scope' && row.required === true && row.resolved === true &&
        row.disposition === 'verified_required_identity', 'required original identity must actually resolve');
      verifiedOriginals++;
    } else {
      require(classification.disposition === 'outside_declared_traversal' && row.required === false && row.resolved === false &&
        row.disposition === 'outside_declared_recursive_reverification', 'outside-scope original failures must remain unresolved');
      outside++;
    }
  }
  require(verification.outside_declared_recursive_reverification_count === outside &&
    verification.record_schema_overinterpretation_count === overinterpretation, 'bounded failure classification counts conflict');
  return {required_pin_count: required.size, required_pins_verified: required.size,
    original_failed_rows: originals.length, verified_required_original_rows: verifiedOriginals,
    outside_declared_recursive_reverification_count: outside, record_schema_overinterpretation_count: overinterpretation,
    whole_reference_graph_verified: false, all_historical_refs_resolved: false};
}

export function preserve(options) {
  const {root, attempt} = options;
  canonicalDirectory(root); canonicalDirectory(attempt, {root});
  const relativeAttempt = path.relative(root, attempt).split(path.sep).join('/');
  require(/^\.temp\/embeddings\/closeout\/cpb-v14\/attempt-\d{3}$/.test(relativeAttempt), 'bounded named V14 attempt required');
  const finalPath = `${relativeAttempt}/preservation.json`;
  require(!exists(absolute(root, finalPath)), 'exclusive final preservation receipt already exists');
  const inputs = {}, checks = [];
  try {
    const names = {admission: 'admissionPath', application: 'applicationPath', staged_review: 'stagedReviewPath',
      source_review: 'sourceReviewPath', proposal_review: 'proposalReviewPath', aliases: 'aliasManifestPath',
      alias_verification: 'aliasVerificationPath', documentary_review: 'docReviewPath', validation: 'validationPath'};
    const data = {};
    for (const [role, key] of Object.entries(names)) {
      require(typeof options[key] === 'string' && options[key].startsWith(`${relativeAttempt}/`) &&
        options[key].endsWith('.json') && options[key].split('/').length === relativeAttempt.split('/').length + 1,
        `one admitted attempt receipt leaf required: ${key}`);
      const loaded = boundJSON(root, options[key]); inputs[role] = loaded.pin; data[role] = loaded.data;
    }
    const {admission, application, staged_review: stagedReview, source_review: sourceReview,
      proposal_review: proposalReview, aliases, alias_verification: aliasVerification,
      documentary_review: docReview, validation} = data;
    require(admission.kind === 'cpb_v14_postmigration_closeout_admission_v1' && admission.artifact_root === relativeAttempt &&
      admission.scope?.compatible_reviewed_catalog_instance_summary_application === true &&
      admission.scope.new_model_training === false && admission.scope.old_encoder_or_head_replay === false &&
      admission.scope.head_refits === false, 'actual scoped human closeout admission required');
    require(options.applicationPath === `${relativeAttempt}/application.json`, 'exact applied generation required');
    require(application.kind === 'cpb_v14_postmigration_metadata_application_v1', 'actual V14 application receipt required');
    passed(application, 'application must have passed'); passed(sourceReview, 'preservation source review must have passed');
    verifySources(root, sourceReview.files, SOURCES);
    require(stagedReview.kind === 'cpb_v14_postmigration_staged_review_v1', 'independent staged review schema');
    passed(stagedReview, 'staged review must have passed'); verifySources(root, stagedReview.source_closure);
    expectPin(root, inputs.staged_review, application.inputs.staged_review, 'application used another staged review');
    const manifestPin = verifyPin(root, application.inputs.manifest), manifest = boundJSON(root, manifestPin.path).data;
    expectPin(root, manifestPin, stagedReview.manifest, 'staged review/actual application manifest conflict');
    require(manifest.kind === 'cpb_v14_postmigration_staged_change_manifest_v1' && manifest.protocol === PROTOCOL,
      'actual current staged manifest required');
    const machine = application.expected_outputs;
    require(Array.isArray(machine) && machine.length === 3 && machine.every((row, index) => row.path === TARGETS[index]) &&
      equal(machine, manifest.outputs) && equal(machine, stagedReview.outputs), 'exact reviewed three-machine output set required');
    const appSourcePin = verifyPin(root, application.inputs.application_source_review);
    const appSource = boundJSON(root, appSourcePin.path).data; passed(appSource, 'application source review must have passed');
    require(equal(application.application_source_closure, appSource.files), 'application source closure mismatch');
    require(equal(application.staged_review_source_closure, stagedReview.source_closure), 'staged source closure mismatch');
    verifySources(root, appSource.files); verifySources(root, application.staged_review_source_closure);
    verifyPin(root, application.shared_application_claim); verifyPin(root, application.application_start);
    require(Array.isArray(application.completed_write_operations) && application.completed_write_operations.length === 3 &&
      TARGETS.every(name => application.completed_write_operations.some(row => row.path === name)), 'all three writes must have completed');
    const immutable = manifest.immutable_preapplication_inputs;
    for (const role of ['admission', 'inputs', 'aliases', 'alias_verification', 'proposal_review', 'source_review'])
      require(immutable?.[role]?.path, `missing immutable preapplication input: ${role}`);
    for (const [role, pin] of Object.entries(immutable)) {
      verifyPin(root, pin);
      if (['admission', 'aliases', 'alias_verification', 'proposal_review'].includes(role))
        expectPin(root, inputs[role], pin, `application input generation conflict: ${role}`);
    }
    passed(proposalReview, 'proposal review must have passed');
    require(proposalReview.kind === 'mityane_cpb_v14_independent_metadata_proposal_review_v1', 'independent proposal review schema');
    verifySources(root, proposalReview.source_closure);
    passed(aliasVerification, 'bulk alias verification must have passed');
    require(aliases.kind === 'cpb_v14_historical_identity_alias_manifest_v1' &&
      aliasVerification.kind === 'cpb_v14_historical_identity_alias_verification_v1' &&
      Array.isArray(aliasVerification.failures) && aliasVerification.failures.length === 0 &&
      Array.isArray(aliases.aliases) && aliases.aliases.length >= 145 && aliasVerification.aliases_verified === aliases.aliases.length &&
      equal(aliases.coverage, aliasVerification.coverage), 'complete stored alias-verification coverage required');
    expectPin(root, inputs.aliases, aliasVerification.alias_manifest, 'alias receipt binds another manifest');
    expectPin(root, inputs.aliases, proposalReview.alias_manifest, 'proposal review binds another alias manifest');
    expectPin(root, inputs.alias_verification, proposalReview.alias_verification, 'proposal review binds another alias verification');
    verifyPin(root, aliases.protocol_identity); verifyPin(root, aliasVerification.protocol_identity);
    verifyPin(root, aliasVerification.shared_parser_identity);
    requireStoredCoverage(aliases, aliasVerification);
    require(typeof aliasVerification.created_utc === 'string' && Number.isFinite(Date.parse(aliasVerification.created_utc)), 'bulk verification observation time required');
    for (const row of aliases.aliases) {
      require(row.original && row.observed && ['cuwacunu_embedding', 'cuwacunu_dev', 'Mityane-geometrics'].includes(row.original.repository) &&
        typeof row.original.path === 'string' && row.original.path.length > 0 &&
        Number.isSafeInteger(row.original.bytes) && row.original.bytes >= 0 && /^[0-9a-f]{64}$/.test(row.original.sha256 ?? '') &&
        row.original.bytes === row.observed.bytes &&
        row.original.sha256 === row.observed.sha256 && Array.isArray(row.provenance) && row.provenance.length > 0,
        'exact stored alias identity and provenance required'); safeRelative(row.current_path);
    }
    let bulkObservation = {observation_utc: aliasVerification.created_utc, base_pair: null,
      bulk_observation_reused_by_alias_generation: false,
      all_aliases_freshly_hashed_this_generation: aliasVerification.all_aliases_freshly_hashed_this_generation ?? true};
    if (aliasVerification.bulk_observation_reused === true) {
      require(aliasVerification.all_aliases_freshly_hashed_this_generation === false && aliasVerification.base_pair &&
        aliasVerification.original_bulk_observation, 'reused bulk observation must carry its original observation and exact base pair');
      verifySources(root, sourceReview.files, [`${PROTOCOL}/repair.mjs`]);
      const baseManifestPin = verifyPin(root, aliasVerification.base_pair.manifest),
        baseVerificationPin = verifyPin(root, aliasVerification.base_pair.verification);
      const baseManifest = boundJSON(root, baseManifestPin.path).data, baseVerification = boundJSON(root, baseVerificationPin.path).data,
        observation = aliasVerification.original_bulk_observation;
      expectPin(root, baseManifestPin, baseVerification.alias_manifest, 'reused bulk base pair conflict');
      require(observation.manifest_created_utc === baseManifest.created_utc &&
        observation.verification_created_utc === baseVerification.created_utc &&
        observation.verification_status === baseVerification.status &&
        equal(aliasVerification.original_failures, baseVerification.failures), 'original bulk observation outcome/time must remain unchanged');
      verifyPin(root, observation.protocol_identity); verifyPin(root, observation.source_review);
      require(Number.isSafeInteger(aliasVerification.reused_aliases) && aliasVerification.reused_aliases > 0 &&
        Number.isSafeInteger(aliasVerification.new_physical_hash_reads) && aliasVerification.new_physical_hash_reads >= 0 &&
        Number.isSafeInteger(aliasVerification.metadata_documents_rewalked) && aliasVerification.metadata_documents_rewalked > 0 &&
        Array.isArray(aliasVerification.base_failure_accounting), 'repair reuse and metadata-closure accounting required');
      bulkObservation = {observation_utc: null, base_pair: {manifest: baseManifestPin, verification: baseVerificationPin},
        bulk_observation_reused_by_alias_generation: true, all_aliases_freshly_hashed_this_generation: false,
        original_bulk_observation: observation, original_failures: aliasVerification.original_failures,
        base_failure_accounting: aliasVerification.base_failure_accounting,
        reused_aliases: aliasVerification.reused_aliases, new_physical_hash_reads: aliasVerification.new_physical_hash_reads,
        metadata_documents_rewalked: aliasVerification.metadata_documents_rewalked};
    }
    const boundedScopePin = verifyPin(root, aliasVerification.bounded_scope),
      scopeReviewPin = verifyPin(root, aliasVerification.scope_review),
      boundedScope = boundJSON(root, boundedScopePin.path).data, scopeReview = boundJSON(root, scopeReviewPin.path).data;
    expectPin(root, boundedScopePin, scopeReview.scope, 'independent scope review binds another declaration');
    expectPin(root, inputs.admission, boundedScope.admission, 'bounded scope uses another admission');
    if (aliasVerification.base_pair) {
      for (const role of ['manifest', 'verification']) expectPin(root, rootPin(root, aliasVerification.base_pair[role]),
        boundedScope.base_pair?.[role], `bounded scope uses another failed base ${role}`);
    }
    const boundedResult = verifyBoundedClassification(boundedScope, scopeReview, aliasVerification, aliases);
    for (const expected of [boundedScope.source_review, scopeReview.source_review]) {
      const pin = verifyPin(root, expected), reviewed = boundJSON(root, pin.path).data;
      passed(reviewed, 'bounded declaration/review source review must have passed');
      verifySources(root, reviewed.files, [`${PROTOCOL}/scope.mjs`, SHARED]);
    }
    verifySources(root, scopeReview.source_closure, [`${PROTOCOL}/scope.mjs`, SHARED]);
    const scopeDocuments = new Map();
    for (const rows of [boundedScope.control_documents, boundedScope.mapping_documents]) {
      require(Array.isArray(rows), 'typed scope control and mapping document arrays required');
      for (const row of rows) {
        identityKey(row.original);
        require(row.observed?.bytes === row.original.bytes && row.observed?.sha256 === row.original.sha256,
          'typed control document historical/current identity conflict');
        const pin = {...row.observed, path: row.current_path}; scopeDocuments.set(pin.path, pin);
      }
    }
    for (const row of boundedScope.required_pins) for (const origin of row.required_by) {
      const pin = rootPin(root, origin.manifest);
      require(typeof origin.pointer === 'string' && typeof origin.selector === 'string', 'exact typed selector provenance required');
      const previous = scopeDocuments.get(pin.path);
      require(!previous || previous.bytes === pin.bytes && previous.sha256 === pin.sha256, 'conflicting typed selector manifest generation');
      scopeDocuments.set(pin.path, pin);
    }
    for (const pin of scopeDocuments.values()) verifyPin(root, pin, 'typed scope control/selector document changed');
    const originals = SMALL_ORIGINALS.map(expected => ({role: expected.role, ...verifyPin(root, expected, 'immutable original changed')}));
    for (const expected of ORIGINAL_PROPOSALS) {
      const pin = proposalReview.inputs?.find(row => row.role === expected.role);
      expectPin(root, expected, pin, `proposal review omits exact original ${expected.role}`);
    }
    expectPin(root, SMALL_ORIGINALS.find(row => row.role === 'original_card'), proposalReview.card, 'proposal review card generation conflict');
    expectPin(root, SMALL_ORIGINALS.find(row => row.role === 'original_proposal_receipt'), proposalReview.generator_receipt, 'proposal receipt generation conflict');
    const admitted = admission.target_observations;
    require(Array.isArray(admitted) && admitted.length === 7 && FINAL_TARGETS.every(name => admitted.filter(row => row.path === name).length === 1),
      'exactly seven admitted target observations required');
    const before = FINAL_TARGETS.map(name => {
      const row = admitted.find(item => item.path === name), newlyCreated = TARGETS.slice(1).includes(name);
      require(row.exists === !newlyCreated, `wrong admitted prior existence: ${name}`);
      if (!row.exists) return {target: name, existed_before: false};
      const relative = `${relativeAttempt}/before/${name}`;
      require(row.whole_before_copy === relative, `whole-before copy must be exact admitted attempt target: ${name}`);
      const pin = observedPin(root, relative);
      require(pin.bytes === row.bytes && pin.sha256 === row.sha256, `whole-before copy changed: ${name}`);
      return {target: name, existed_before: true, ...pin};
    });
    require(Array.isArray(application.before_copies) && application.before_copies.length === 1 &&
      application.before_copies[0].target === TARGETS[0], 'actual catalog application before-copy binding required');
    expectPin(root, before[0], application.before_copies[0], 'application before-copy generation conflict');
    require(Array.isArray(application.observed_after) && application.observed_after.length === 3 &&
      application.observed_after.every(row => row.matches_staged === true), 'application must record all final output matches');
    for (const output of machine) {
      require(equal(output.expected_before, admitted.find(row => row.path === output.path)), 'application/admission before-state conflict');
      const pin = observedPin(root, output.path); expectPin(root, pin, output, 'applied machine record changed');
      expectPin(root, pin, application.observed_after?.find(row => row.path === output.path), 'actual application output observation conflict');
      verifyPin(root, {...output, path: output.staged_path}, 'preserved staged output changed');
    }
    require(docReview.kind === 'cpb_v14_postmigration_documentary_review_v1' && docReview.independent_review === true,
      'independent final documentary review required'); passed(docReview, 'documentary review must have passed');
    expectPin(root, inputs.admission, docReview.admission, 'documentary review admission conflict');
    expectPin(root, inputs.application, docReview.application, 'documentary review application conflict');
    require(Array.isArray(docReview.outputs) && docReview.outputs.length === 4 && FINAL_TARGETS.slice(3).every(name =>
      docReview.outputs.filter(row => row.path === name).length === 1), 'exact four documentary final outputs required');
    for (const output of docReview.outputs) {
      require(equal(output.expected_before, admitted.find(row => row.path === output.path)), 'documentary review/admission before-state conflict');
      verifyPin(root, output, 'reviewed documentary output changed');
    }
    require(validation.kind === 'cpb_v14_postmigration_final_validation_v1', 'actual final validation receipt required');
    passed(validation, 'final validation must have passed');
    expectPin(root, inputs.application, validation.application, 'validation application conflict');
    expectPin(root, inputs.source_review, validation.source_review, 'validation source generation conflict');
    const finalOutputs = FINAL_TARGETS.map(name => observedPin(root, name));
    require(Array.isArray(validation.reviewed_outputs) && validation.reviewed_outputs.length === 7 &&
      finalOutputs.every(pin => validation.reviewed_outputs.filter(row => row.path === pin.path && row.bytes === pin.bytes && row.sha256 === pin.sha256).length === 1),
      'validation must bind all seven actual final outputs');
    for (const id of CHECKS) {
      const rows = validation.checks?.filter(row => row.id === id) ?? [];
      require(rows.length === 1 && rows[0].passed === true && rows[0].exit_code === 0 &&
        Array.isArray(rows[0].command) && rows[0].command.length > 0 && rows[0].command.every(word => typeof word === 'string'),
        `actual passed validation check required: ${id}`);
      const row = rows[0], log = verifyPin(root, row.log, `validation log changed: ${id}`);
      const result = row.receipt ? verifyPin(root, row.receipt, `validation result changed: ${id}`) : null;
      if (id === 'focused_fixtures') {
        require(result, 'focused fixtures need an actual result receipt');
        const fixtureResult = boundJSON(root, result.path).data; passed(fixtureResult, 'focused fixture result must have passed');
        require(Number.isSafeInteger(fixtureResult.tests) && fixtureResult.tests > 0 && fixtureResult.pass === fixtureResult.tests && fixtureResult.fail === 0,
          'actual all-passed focused fixture counts required');
        expectPin(root, inputs.source_review, fixtureResult.source_review, 'focused fixtures tested another source generation');
        expectPin(root, log, fixtureResult.log, 'focused fixture receipt binds another test log');
      }
      if (id === 'record_semantics') {
        require(result, 'record semantics need an actual final independent semantic review receipt');
        const semantic = boundJSON(root, result.path).data;
        require(semantic.kind === 'cpb_v14_postmigration_final_semantic_review_v1', 'actual final semantic review schema required');
        passed(semantic, 'final independent semantic review must have passed');
        expectPin(root, inputs.application, semantic.application, 'semantic review application conflict');
        expectPin(root, inputs.staged_review, semantic.staged_review, 'semantic review staged-review conflict');
        expectPin(root, manifestPin, semantic.manifest, 'semantic review manifest conflict');
        require(equal(semantic.outputs, machine) && semantic.all_adaptations_independently_checked === true &&
          semantic.unchanged_scientific_data === true, 'final semantic review must independently verify every actual output and adaptation');
        const semanticSource = verifyPin(root, semantic.source);
        require(semanticSource.path === `${PROTOCOL}/inspect.mjs`, 'semantic review must bind the independent inspector source');
        verifySources(root, semantic.source_closure, [`${PROTOCOL}/inspect.mjs`, SHARED]);
        const semanticSourceReviewPin = verifyPin(root, semantic.source_review),
          semanticSourceReview = boundJSON(root, semanticSourceReviewPin.path).data;
        passed(semanticSourceReview, 'semantic inspector source review must have passed');
        verifySources(root, semanticSourceReview.files, [`${PROTOCOL}/inspect.mjs`, SHARED]);
        require(equal(semantic.source_closure, semanticSourceReview.files), 'semantic review source closure conflict');
      }
      checks.push({id, log, receipt: result, command: row.command, exit_code: 0, passed: true});
    }
    const catalog = boundJSON(root, TARGETS[0]).data, instance = boundJSON(root, TARGETS[1]).data,
      summary = boundJSON(root, TARGETS[2]).data;
    require(catalog.fresh_research_candidate?.instance === 'CPB-v14.market-01' &&
      catalog.clean_causal_training_baseline?.instance === 'CPB-v3.market-01' && catalog.active_application === null &&
      instance.instance === 'CPB-v14.market-01' && instance.clean_baseline === 'CPB-v3.market-01' && instance.application_selection === null &&
      instance.quality?.numeric_passed === 3 && instance.quality.numeric_total === 22 &&
      instance.quality.strict_new_scoped_quality_leaders === 0 && instance.advancement_passed === false &&
      summary.scientific_conclusion?.mityane_application_selection === null,
      'final records must preserve V14 result, V3 baseline and no project selection');
    require(summary.future_receipt_locations?.application === options.applicationPath &&
      summary.future_receipt_locations.preservation === finalPath && application.preservation_receipt_location === finalPath,
      'later receipt locations must remain acyclic and exact');
    for (const [role, pin] of Object.entries(inputs)) verifyPin(root, pin, `receipt changed during final checks: ${role}`);
    verifyPin(root, boundedScopePin); verifyPin(root, scopeReviewPin);
    for (const pin of [...finalOutputs, ...originals]) verifyPin(root, pin, 'final observed file changed');
    verifySources(root, sourceReview.files, SOURCES);
    const receipt = {kind: 'cpb_v14_postmigration_final_preservation_v1', status: 'passed', passed: true,
      created_utc: new Date().toISOString(), protocol: PROTOCOL, inputs, final_outputs: finalOutputs,
      whole_before_copies_or_absences: before, freshly_verified_small_originals: originals,
      bounded_verification: {scope: boundedScopePin, independent_scope_review: scopeReviewPin, ...boundedResult,
        coverage: boundedScope.coverage,
        control_and_selector_documents_freshly_hashed: [...scopeDocuments.values()],
        policy: 'Only the independently reviewed typed declaration defines recursive reverification. Required identities must verify; excluded and schema-overinterpreted original failures retain their original unresolved status.'},
      whole_reference_graph_verified: false,
      historical_bulk_verification: {alias_manifest: inputs.aliases, verification_receipt: inputs.alias_verification,
        receipt_created_utc: aliasVerification.created_utc, ...bulkObservation, coverage: aliases.coverage,
        reused_passed_verification: true, fresh_bulk_rehash_performed: false,
        whole_reference_graph_verified: false,
        coverage_observation_policy: 'Inherited inventory counts retain their bound earlier observation; only bounded_verification.coverage describes the current typed reverification declaration.',
        limitation: 'The whole historical reference graph is not verified. The passed bounded alias receipt can carry earlier physical observations and a separately preserved failed base receipt; outside-declaration and schema-overinterpreted original references remain unresolved. Receipt creation time is not a per-alias observation time. Earlier outcomes remain unchanged; fresh new-alias hashes and the declared typed closure belong to the repair generation. This final step freshly hashes small original pins, receipts, reviewed sources, before-copies and all seven final outputs; it does not repeat the bulk checkpoint scan.'},
      validation: checks, source_closure: sourceReview.files, failures: [],
      scientific_execution_performed: false, tensor_decodes: 0, model_calls: 0, head_fits: 0, training_updates: 0,
      hash_cycle_policy: 'This final receipt binds earlier actual artifacts. No self hash and no future receipt hash is included.'};
    writeNew(absolute(root, finalPath), jsonBytes(receipt)); return {...receipt, receipt_path: finalPath};
  } catch (error) {
    const failed = {kind: 'cpb_v14_postmigration_final_preservation_v1', status: 'failed', passed: false,
      created_utc: new Date().toISOString(), protocol: PROTOCOL, inputs, checks_completed: checks,
      failures: [{name: String(error.name), message: String(error.message)}], final_outputs_changed_by_this_step: false,
      fresh_bulk_rehash_performed: false, scientific_execution_performed: false};
    let failedPath;
    for (let index = 1; index <= 999; index++) {
      const candidate = `${relativeAttempt}/preservation-failed-${String(index).padStart(3, '0')}.json`;
      try { writeNew(absolute(root, candidate), jsonBytes(failed)); failedPath = candidate; break; }
      catch (failure) { if (failure.code !== 'EEXIST') { failure.actual_failed_result = failed; throw failure; } }
    }
    require(failedPath, 'all exclusive preservation failure slots are occupied');
    const failure = new Error(`Final preservation failed; retain ${failedPath}: ${error.message}`, {cause: error});
    failure.receipt_path = failedPath; failure.receipt = failed; throw failure;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const flags = {'--root': 'root', '--attempt': 'attempt', '--admission': 'admissionPath', '--application': 'applicationPath',
    '--staged-review': 'stagedReviewPath', '--source-review': 'sourceReviewPath', '--proposal-review': 'proposalReviewPath',
    '--aliases': 'aliasManifestPath', '--alias-verification': 'aliasVerificationPath', '--doc-review': 'docReviewPath', '--validation': 'validationPath'}, args = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const flag = process.argv[index];
    require(Object.hasOwn(flags, flag) && process.argv[index + 1] && !Object.hasOwn(args, flag), 'exact paired preservation arguments required');
    args[flag] = process.argv[index + 1];
  }
  require(Object.keys(args).length === Object.keys(flags).length && process.platform === 'linux' &&
    fs.existsSync('/.dockerenv') && args['--root'] === '/mityane', 'preservation requires all eleven flags and the managed Linux container at /mityane');
  try {
    const result = preserve(Object.fromEntries(Object.entries(flags).map(([flag, name]) => [name, args[flag]])));
    process.stdout.write(jsonBytes({status: result.status, receipt_path: result.receipt_path}));
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    if (error.actual_failed_result) process.stderr.write(jsonBytes(error.actual_failed_result));
    process.exitCode = 1;
  }
}
