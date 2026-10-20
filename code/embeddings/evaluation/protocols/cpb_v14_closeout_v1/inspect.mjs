// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Independent semantic review. Imports no staging/application/proposer code.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
import {admitFiles, canonicalDirectory, hashAdmitted, inside, jsonBytes, parseJSON,
  readAdmitted, requireCondition as require, safeRelative, sha256, writeNew} from '../../../../shared/artifacts/node/v1/files.mjs';

const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const FAMILY = 'code/embeddings/encoders/causal_patch_bottleneck_mae/versions';
const SHARED = 'code/shared/artifacts/node/v1/files.mjs';
const SELF = `${PROTOCOL}/inspect.mjs`;
export const TARGETS = [`${FAMILY}/current.json`, `${FAMILY}/instances/CPB-v14.market-01.json`, 'code/embeddings/evaluation/records/CPB-v14.market-01.closeout.json'];
const SOURCES = ['aliases.mjs', 'review.mjs', 'stage.mjs', 'apply.mjs', 'inspect.mjs'].map(name => `${PROTOCOL}/${name}`).concat(SHARED);
const PAYLOAD = 'code/embeddings/output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001/payload/proposals-001';
const ORIGINAL = [
  {role: 'catalog_proposal', path: `${PAYLOAD}/code/encoders/causal_patch_bottleneck_mae/versions/current.json`, bytes: 130259, sha256: '4159faa7a9177721d338466fe9e6fd260f5e54b393ecbeb0a2c11bd07d1f9204'},
  {role: 'instance_proposal', path: `${PAYLOAD}/code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v14.market-01.json`, bytes: 29471, sha256: '77262a4050a88e8eb7a86d8e675aa26522111bc7e406299b43a6c07347a05d02'},
  {role: 'ledger_proposal', path: `${PAYLOAD}/doc/clean_causal_encoder_research_ledger.json`, bytes: 1987000, sha256: '47360e36128eac9c27a5a4609f946eae2fcd2c6c2cf158bbf2b378bb041dbd45'},
];
const ARCHIVED_LEDGER = {path: '.archive/embeddings/pre-cleanup-2026-10-18/doc/clean_causal_encoder_research_ledger.json', bytes: 1746343, sha256: '4ac9a0a3b73225b3ca33915176a2785dd23524d2464021a421a99c33f5464270'};
const PIN_KEYS = ['path', 'bytes', 'sha256'];
const pick = (value, keys = PIN_KEYS) => Object.fromEntries(keys.map(key => [key, value[key]]));
const subset = (value, excluded) => Object.fromEntries(Object.entries(value).filter(([key]) => !excluded.includes(key)));
const escape = value => String(value).replaceAll('~', '~0').replaceAll('/', '~1');

// Exact tree comparison keeps unknown existing fields and unsafe integer seeds.
// Only the named adaptation keys may differ; each is checked separately below.
export function inspectRecords({catalog, instance, summary, catalogProposal: C, instanceProposal: I,
  ledgerProposal: L, aliases, bindings, proposalPins, attemptRelative, archivedLedger = ARCHIVED_LEDGER}) {
  let checks = 0;
  const eq = (actual, expected, message) => { checks++; require(same(actual, expected), message); };
  const fields = (actual, expected, message) => { for (const [key, value] of Object.entries(expected)) eq(actual[key], value, `${message}: ${key}`); };
  const exactExcept = (actual, original, allowed, message) => eq(subset(actual, allowed), subset(original, allowed), message);
  eq(C.clean_causal_training_baseline.instance, 'CPB-v3.market-01', 'Proposal V3 clean baseline');
  eq(I.clean_baseline, L.baseline, 'Instance and ledger baseline'); eq(L.baseline, 'CPB-v3.market-01', 'Ledger V3 baseline');
  fields(I, {instance: 'CPB-v14.market-01', advancement_passed: false, overall_winner: false}, 'Saved V14 outcome');
  fields(I.quality, {numeric_passed: 3, numeric_total: 22, strict_new_scoped_quality_leaders: 0}, 'Saved 3/22 without promotion');
  fields(C.fresh_research_candidate, {instance: I.instance, numeric_passed: 3, numeric_total: 22, advancement_passed: false}, 'Saved catalog result');
  const predecessor = {repository: 'cuwacunu_embedding', catalog: pick(C, ['active_application', 'interpretation', 'application_selection_record']),
    instance: pick(I, ['application_selection', 'application_selection_changed']),
    policy: 'These are preserved predecessor decisions. They do not establish a Mityane project-wide encoder selection.'};
  const selection = {instance: null, status: 'none_established_by_this_closeout'};
  const closeout = {protocol: PROTOCOL, derivation: 'reviewed historical proposal DATA with explicit field adaptations',
    immutable_preapplication_inputs: bindings, historical_proposals: proposalPins, summary: TARGETS[2], research_execution_authorized: false};
  const catalogChanges = {active_application: null, application_selection_record: null,
    interpretation: 'V3 is the clean causal research baseline. V4 is the broad clean-causal breakthrough; later versions retain scoped improvements. Mityane has no project-wide application encoder selection. The latest measured candidate is V14, whose exact failed recipe is closed.',
    historical_predecessor_application: predecessor, mityane_application_selection: selection, metadata_closeout: closeout};
  exactExcept(catalog, C, [...Object.keys(catalogChanges), 'fresh_research_candidate'], 'Unchanged complete catalog fields and history');
  fields(catalog, catalogChanges, 'Only approved catalog application/binding adaptation');
  const oldNavigation = Object.fromEntries(['design_file', 'physical_version_root', 'measured_instance_file', 'instance_record'].filter(key => Object.hasOwn(C.fresh_research_candidate, key)).map(key => [key, C.fresh_research_candidate[key]]));
  const freshChanges = {design_file: `${FAMILY}/CPB-v14.json`, physical_version_root: `${FAMILY}/v14`, measured_instance_file: TARGETS[1],
    navigation_path_base: 'Mityane repository root', historical_navigation: {original_repository: 'cuwacunu_embedding', fields: oldNavigation,
      source_identity_policy: 'Frozen source identities remain bound to original paths and source-manifest bytes; maintained locations have separate identities.'}};
  if (Object.hasOwn(C.fresh_research_candidate, 'instance_record')) freshChanges.instance_record = TARGETS[1];
  const locator = original => {
    const located = [];
    const visit = (value, pointer = '') => {
      if (!value || typeof value !== 'object') return;
      if (typeof value.path === 'string' && Number.isSafeInteger(value.bytes) && /^[0-9a-f]{64}$/.test(value.sha256 ?? '')) {
        const repository = value.repository ?? value.original_repository ?? 'cuwacunu_embedding';
        const matches = aliases.filter(row => row.original.repository === repository && same(pick(row.original), pick(value)));
        require(matches.length > 0 && new Set(matches.map(row => row.current_path)).size === 1, `Unique exact repository-qualified locator required: ${pointer}`);
        const row = matches[0]; eq(pick(row.observed, ['bytes', 'sha256']), pick(value, ['bytes', 'sha256']), `Exact observed alias ${pointer}`);
        safeRelative(row.current_path);
        located.push({pointer, original_repository: repository, historical_pin: value, current_path: row.current_path,
          observed_bytes: row.observed.bytes, observed_sha256: row.observed.sha256});
      }
      for (const [key, child] of Object.entries(value)) visit(child, `${pointer}/${escape(key)}`);
    };
    visit(original);
    return {located, historical_pins_without_locator: [], policy: 'Historical path/bytes/hash tuples retain their original repository identity. Locators are separate navigation data; no relocated source substitutes for a frozen source capsule.'};
  };
  freshChanges.evidence_locators = locator(C.fresh_research_candidate);
  exactExcept(catalog.fresh_research_candidate, C.fresh_research_candidate, Object.keys(freshChanges), 'Unchanged complete candidate scores, seeds, protocols and evidence');
  fields(catalog.fresh_research_candidate, freshChanges, 'Only approved candidate navigation/locator adaptation');
  const instanceChanges = {application_selection: null, application_selection_changed: false, historical_predecessor_application: predecessor,
    mityane_application_selection: selection, historical_source_identity: {original_repository: 'cuwacunu_embedding', version_root: I.version_root,
      source_manifest: I.source_manifest, source_capsule: I.source_capsule,
      policy: 'Historical execution used this frozen source identity. Maintained relocated or edited code is not substituted for it.'},
    version_root: `${FAMILY}/v14`, navigation_path_base: 'Mityane repository root', current_locations: {
      design_file: `${FAMILY}/CPB-v14.json`, measured_instance_file: TARGETS[1],
      ...Object.fromEntries(['local_run_root', 'local_quality_root', 'local_engineering_root'].map(key => [key, `code/embeddings/${I[key]}`])), historical_runtime_root: I.container_root},
    evidence_locators: locator(I), metadata_closeout: closeout};
  exactExcept(instance, I, Object.keys(instanceChanges), 'Unchanged complete instance quality, counters, seeds and scientific evidence');
  fields(instance, instanceChanges, 'Only approved instance application/navigation adaptation');
  const expectedSummary = {kind: 'cpb_v14_postmigration_metadata_closeout_v1', date: '2026-10-18', instance: 'CPB-v14.market-01', protocol: PROTOCOL,
    status: 'reviewed_metadata_derivation_application_and_preservation_recorded_separately', scientific_conclusion: {
      clean_research_baseline: 'CPB-v3.market-01', broad_clean_causal_breakthrough: 'CPB-v4.market-01', historical_balanced_leader: 'RPB-v24.market-01.alt-01 native36',
      measured_candidate_established: true, numeric_passed: 3, numeric_total: 22, strict_new_scoped_quality_leaders: 0, advancement_passed: false,
      exact_v14_recipe_closed: true, mityane_application_selection: null, fresh_weights_versions: 'CPB-v3 through CPB-v14; source-design branches do not imply accumulated trained weights'},
    immutable_preapplication_inputs: bindings, historical_proposals: proposalPins, preserved_archived_ledger: archivedLedger,
    ledger_data_policy: 'The reviewed ledger proposal is immutable historical DATA. Its leaders, scores, histories and failures remain there; the archived prior ledger is unchanged. No parallel mutable progress ledger is created.',
    maintained_records: TARGETS.slice(0, 2), future_receipt_locations: {application: `${attemptRelative}/application.json`, preservation: `${attemptRelative}/preservation.json`},
    future_receipt_policy: 'Locations name future records and claim no result. Those records bind actual applied bytes; this summary does not hash itself or any later receipt.',
    scope: {metadata_only: true, new_training: false, model_or_head_replay: false, head_refits: false, python_executed_by_protocol: false, checkpoint_selection: false},
    next_scientific_work: {status: 'requires_separate_saved_TRAIN_only_protocol', question: 'Window overlap, native covariance and actual saved shared-position adjoints; two captured batches do not establish a complete training-trajectory explanation.', validation_tensor_diagnosis: false, v15_selected: false}};
  eq(summary, expectedSummary, 'Exact bounded closeout summary; no hidden fields or future receipt hashes');
  return {checks, unchanged_scientific_data: true, historical_application_preserved: true, mityane_selection: null,
    numeric_passed: 3, numeric_total: 22, advancement_passed: false, new_leaders: 0};
}

export function changedFields(before, after, pointer = '') {
  if (same(before, after)) return [];
  const object = value => value !== null && typeof value === 'object';
  if (object(before) && object(after) && Array.isArray(before) === Array.isArray(after)) {
    return [...new Set([...Object.keys(before), ...Object.keys(after)])].flatMap(key => {
      const at = `${pointer}/${escape(key)}`, left = Object.hasOwn(before, key), right = Object.hasOwn(after, key);
      if (left && right) return changedFields(before[key], after[key], at);
      return [{pointer: at, operation: left ? 'remove' : 'add', prior_present: left, after_present: right,
        ...(left ? {before: before[key]} : {}), ...(right ? {after: after[key]} : {})}];
    });
  }
  return [{pointer, operation: 'replace', prior_present: true, after_present: true, before, after}];
}
const absolute = (root, relative) => path.join(root, safeRelative(relative));
function actualPin(root, relative) {
  const row = admitFiles([absolute(root, relative)], {root, singleLink: true})[0];
  return {path: relative, bytes: row.bytes, sha256: hashAdmitted(row)};
}
function load(root, relative) {
  const row = admitFiles([absolute(root, relative)], {root, singleLink: true})[0], bytes = readAdmitted(row);
  return {pin: {path: relative, bytes: bytes.length, sha256: sha256(bytes)}, data: parseJSON(bytes, {duplicates: 'last', integers: 'bigint'})};
}
function relativePin(root, value) {
  require(value && typeof value.path === 'string', 'Explicit pin required');
  if (!path.isAbsolute(value.path)) return pick(value);
  require(inside(root, value.path), 'Pin escapes repository');
  return {...pick(value), path: path.relative(root, value.path).split(path.sep).join('/')};
}
function exactPin(root, expected) {
  const normalized = relativePin(root, expected);
  require(same(actualPin(root, normalized.path), normalized), `Changed bound file: ${normalized.path}`);
  return normalized;
}
function approvedSources(root, record, required) {
  require(record.passed === true && Array.isArray(record.files), 'Passed independent source review required');
  const index = new Map();
  for (const row of record.files) { const checked = exactPin(root, row); require(!index.has(checked.path), 'Duplicate source pin'); index.set(checked.path, checked); }
  for (const name of required) require(index.has(name), `Source review omits ${name}`);
  return index;
}

export function inspect({root, attempt, sourceReviewPath, outputPath, mode = 'staged', stagedReviewPath, applicationPath}) {
  canonicalDirectory(root); canonicalDirectory(attempt, {root});
  require(['staged', 'final'].includes(mode), 'Explicit staged/final mode required');
  const relativeAttempt = path.relative(root, attempt).split(path.sep).join('/');
  require(/^\.temp\/embeddings\/closeout\/cpb-v14\/attempt-\d+$/.test(relativeAttempt), 'Named V14 attempt required');
  require(outputPath.startsWith(`${relativeAttempt}/`) && !outputPath.slice(relativeAttempt.length + 1).includes('/'), 'Exclusive review JSON in this attempt required');
  require(outputPath.endsWith('.json') && !fs.existsSync(absolute(root, outputPath)), 'New review receipt required');
  canonicalDirectory(path.dirname(absolute(root, outputPath)), {root});
  const sourceReview = load(root, sourceReviewPath), reviewedSources = approvedSources(root, sourceReview.data, SOURCES);
  const manifest = load(root, `${relativeAttempt}/staged-change-manifest.json`), M = manifest.data;
  require(M.kind === 'cpb_v14_postmigration_staged_change_manifest_v1' && M.status === 'staged_only_not_applied' && M.applied === false && M.protocol === PROTOCOL, 'Unapplied exact staged manifest required');
  require(Array.isArray(M.outputs) && M.outputs.length === 3 && M.outputs.every((row, i) => row.path === TARGETS[i] && row.staged_path === `${relativeAttempt}/staged/${TARGETS[i]}`), 'Exact ordered three staged outputs required');
  const loaded = {};
  for (const role of ['admission', 'inputs', 'aliases', 'alias_verification', 'proposal_review', 'source_review']) {
    const expected = exactPin(root, M.immutable_preapplication_inputs[role]); loaded[role] = load(root, expected.path);
  }
  require(Object.keys(M.immutable_preapplication_inputs).length === 6, 'Exact six immutable review inputs');
  const {admission, inputs, aliases, alias_verification: AV, proposal_review: PR, source_review: SR} = loaded;
  require(admission.data.artifact_root === relativeAttempt && admission.data.scope?.compatible_reviewed_catalog_instance_summary_application === true && admission.data.scope.new_model_training === false && admission.data.scope.old_encoder_or_head_replay === false && admission.data.scope.head_refits === false, 'Actual metadata-only admission');
  require(aliases.data.kind === 'cpb_v14_historical_identity_alias_manifest_v1'
    && PR.data.kind === 'mityane_cpb_v14_independent_metadata_proposal_review_v1'
    && AV.data.status === 'passed' && AV.data.passed === true && Array.isArray(AV.data.failures) && AV.data.failures.length === 0 && PR.data.status === 'passed' && PR.data.passed === true, 'Passed complete alias/proposal reviews');
  for (const expected of [AV.data.alias_manifest, PR.data.alias_manifest]) require(same(relativePin(root, expected), aliases.pin), 'Reviews bind another alias manifest');
  require(same(relativePin(root, PR.data.alias_verification), AV.pin), 'Proposal review binds another alias result');
  const priorSources = approvedSources(root, SR.data, [`${PROTOCOL}/stage.mjs`, SHARED]);
  for (const row of [...PR.data.source_closure, PR.data.source, AV.data.protocol_identity, AV.data.shared_parser_identity, aliases.data.protocol_identity]) {
    const checked = exactPin(root, row); require(same(reviewedSources.get(checked.path), checked), `Reviewed source generation mismatch: ${checked.path}`);
  }
  for (const [name, pin] of priorSources) require(same(reviewedSources.get(name), pin), `Staging source review generation differs: ${name}`);
  const proposalData = {};
  require(same(M.historical_proposals, ORIGINAL), 'Only original immutable three proposal DATA admitted');
  for (const row of ORIGINAL) {
    exactPin(root, row); proposalData[row.role] = load(root, row.path).data;
    for (const list of [inputs.data.proposals, PR.data.inputs]) require(list.filter(item => item.role === row.role).length === 1 && same(relativePin(root, list.find(item => item.role === row.role)), pick(row)), `Review/input omits exact ${row.role}`);
  }
  exactPin(root, ARCHIVED_LEDGER);
  const I = proposalData.instance_proposal;
  const descriptor = {...pick(I.prospective_descriptor), path: `${FAMILY}/CPB-v14.json`};
  exactPin(root, descriptor);
  const navigationDirectories = [`${FAMILY}/v14`, ...['local_run_root', 'local_quality_root', 'local_engineering_root'].map(key => {
    require(typeof I[key] === 'string' && I[key].startsWith('output/'), `Explicit historical navigation ${key}`);
    return `code/embeddings/${safeRelative(I[key])}`;
  })];
  for (const directory of navigationDirectories) canonicalDirectory(absolute(root, directory), {root});
  const documents = M.outputs.map((row, i) => {
    require(same(pick(actualPin(root, row.staged_path), ['bytes', 'sha256']), pick(row, ['bytes', 'sha256'])), `Changed staged bytes: ${row.path}`);
    const expected = row.expected_before;
    require(expected.path === row.path && expected.exists === (i === 0), 'Exact expected-before state');
    for (const rows of [inputs.data.targets, admission.data.target_observations]) require(rows.filter(item => item.path === row.path).length === 1 && same(rows.find(item => item.path === row.path), expected), 'Admission/input target binding differs');
    if (i === 0) {
      require(expected.whole_before_copy === `${relativeAttempt}/before/${row.path}`, 'Exact whole before-copy location');
      require(same(pick(actualPin(root, expected.whole_before_copy), ['bytes', 'sha256']), pick(expected, ['bytes', 'sha256'])), 'Before-copy does not match admitted generation');
    }
    if (mode === 'staged') {
      if (expected.exists) require(same(actualPin(root, row.path), pick(expected)), 'Live catalog changed from before pin');
      else { try { fs.lstatSync(absolute(root, row.path)); throw new Error(`Expected absent live target: ${row.path}`); } catch (error) { if (error.code !== 'ENOENT') throw error; } }
    } else require(same(actualPin(root, row.path), pick(row)), 'Applied bytes differ from reviewed stage');
    return load(root, mode === 'staged' ? row.staged_path : row.path).data;
  });
  const priorCatalog = load(root, M.outputs[0].expected_before.whole_before_copy).data, C = proposalData.catalog_proposal;
  require(same(C.fresh_research_candidate_history.slice(0, -1), priorCatalog.fresh_research_candidate_history) && same(C.fresh_research_candidate_history.at(-1), priorCatalog.fresh_research_candidate), 'Whole prior candidate history and V13 preserved');
  const result = inspectRecords({catalog: documents[0], instance: documents[1], summary: documents[2], catalogProposal: C, instanceProposal: proposalData.instance_proposal,
    ledgerProposal: proposalData.ledger_proposal, aliases: aliases.data.aliases, bindings: M.immutable_preapplication_inputs,
    proposalPins: M.historical_proposals, attemptRelative: relativeAttempt});
  require(Array.isArray(M.field_adaptations) && M.field_adaptations.length === 3, 'Exactly three complete adaptation groups');
  for (let i = 0; i < 3; i++) {
    const group = M.field_adaptations[i]; require(group.target === TARGETS[i] && group.proposal_role === (i === 0 ? 'catalog_proposal' : i === 1 ? 'instance_proposal' : null), 'Exact adaptation target/role');
    const actual = i === 2 ? [{pointer: '', operation: 'add', prior_present: false, after_present: true, after: documents[2]}] : changedFields(i === 0 ? C : proposalData.instance_proposal, documents[i]);
    require(group.fields.every(row => typeof row.reason === 'string' && row.reason.length > 0) && same(group.fields.map(row => subset(row, ['reason'])), actual), 'Full field adaptation manifest differs from independently observed changes');
  }
  const receipt = {kind: mode === 'staged' ? 'cpb_v14_postmigration_staged_review_v1' : 'cpb_v14_postmigration_final_semantic_review_v1', status: 'passed', passed: true,
    mode, ...result, manifest: manifest.pin, outputs: M.outputs, source_review: sourceReview.pin,
    source_closure: sourceReview.data.files, source: actualPin(root, SELF), immutable_preapplication_inputs: M.immutable_preapplication_inputs,
    original_proposals: ORIGINAL, all_adaptations_independently_checked: true, preserved_archived_ledger_unchanged: true,
    current_navigation_verification: {descriptor, directories: navigationDirectories},
    model_calls: 0, head_fits: 0, training_updates: 0, live_files_written: 0};
  if (mode === 'final') {
    require(stagedReviewPath === `${relativeAttempt}/staged-review.json` && applicationPath === `${relativeAttempt}/application.json`, 'Exact prior staged/application receipt names');
    const staged = load(root, stagedReviewPath), applied = load(root, applicationPath), A = applied.data;
    require(staged.data.kind === 'cpb_v14_postmigration_staged_review_v1' && staged.data.status === 'passed' && staged.data.passed === true && same(staged.data.manifest, manifest.pin) && same(staged.data.outputs, M.outputs), 'Passed exact staged review required');
    for (const pin of staged.data.source_closure) exactPin(root, pin);
    require(A.kind === 'cpb_v14_postmigration_metadata_application_v1' && A.status === 'passed' && A.passed === true && A.rollback_performed === false && same(A.expected_outputs, M.outputs) && same(A.immutable_preapplication_inputs, M.immutable_preapplication_inputs), 'Passed exact application receipt required');
    require(same(A.inputs.manifest, manifest.pin) && same(A.inputs.staged_review, staged.pin), 'Application binds different manifest/review');
    exactPin(root, A.inputs.application_source_review);
    const applicationSources = load(root, A.inputs.application_source_review.path);
    const actualApplicationSources = approvedSources(root, applicationSources.data, [`${PROTOCOL}/apply.mjs`, SHARED]);
    for (const [name, pin] of actualApplicationSources) require(same(reviewedSources.get(name), pin), `Application source generation differs: ${name}`);
    require(same(A.application_source_closure, applicationSources.data.files) && same(A.staged_review_source_closure, staged.data.source_closure), 'Application source closure records differ');
    exactPin(root, A.shared_application_claim); exactPin(root, A.application_start);
    require(A.observed_after.length === 3 && A.observed_after.every((row, i) => row.matches_staged === true && same(pick(row), pick(M.outputs[i]))), 'Application observations differ from live reviewed bytes');
    require(same(A.completed_write_operations, [1, 2, 0].map(i => ({...pick(M.outputs[i]), operation: i === 0 ? 'atomic_replace' : 'exclusive_create'}))), 'All three exact reviewed writes must have completed');
    receipt.staged_review = staged.pin; receipt.application = applied.pin;
  }
  // Verify every admitted source and immutable input once more before sealing.
  for (const pin of sourceReview.data.files) exactPin(root, pin);
  for (const pin of Object.values(M.immutable_preapplication_inputs)) exactPin(root, pin);
  for (const pin of ORIGINAL) exactPin(root, pin);
  exactPin(root, descriptor);
  for (const directory of navigationDirectories) canonicalDirectory(absolute(root, directory), {root});
  for (const output of M.outputs) {
    exactPin(root, {...pick(output), path: output.staged_path});
    if (mode === 'final') exactPin(root, output);
  }
  if (mode === 'final') { exactPin(root, receipt.application); exactPin(root, receipt.staged_review); }
  exactPin(root, manifest.pin); exactPin(root, sourceReview.pin);
  writeNew(absolute(root, outputPath), jsonBytes(receipt));
  return receipt;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const names = {'--mode': 'mode', '--root': 'root', '--attempt': 'attempt', '--source-review': 'sourceReviewPath', '--output': 'outputPath', '--staged-review': 'stagedReviewPath', '--application': 'applicationPath'}, args = {};
  for (let i = 2; i < process.argv.length; i += 2) { require(Object.hasOwn(names, process.argv[i]) && process.argv[i + 1] && !Object.hasOwn(args, names[process.argv[i]]), 'Exact paired inspection arguments'); args[names[process.argv[i]]] = process.argv[i + 1]; }
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv') && args.root === '/mityane', 'Managed Linux container at /mityane only');
  try { const result = inspect(args); process.stdout.write(jsonBytes({status: result.status, mode: result.mode, checks: result.checks})); }
  catch (error) { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; }
}
