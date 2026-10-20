// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Metadata derivation only. Run through the managed Linux container.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {
  admitFiles, canonicalDirectory, createExclusiveDirectory, hashAdmitted,
  inside, jsonBytes, parseJSON, readAdmitted, requireCondition as require,
  safeRelative, sha256, writeNew,
} from '../../../../shared/artifacts/node/v1/files.mjs';

const FAMILY = 'code/embeddings/encoders/causal_patch_bottleneck_mae/versions';
export const TARGETS = Object.freeze([
  `${FAMILY}/current.json`,
  `${FAMILY}/instances/CPB-v14.market-01.json`,
  'code/embeddings/evaluation/records/CPB-v14.market-01.closeout.json',
]);
const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const SHARED = 'code/shared/artifacts/node/v1/files.mjs';
const ROLES = ['catalog_proposal', 'instance_proposal', 'ledger_proposal'];
const SAVED_PROPOSALS = 'code/embeddings/output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001/payload/proposals-001';
export const ORIGINAL_PROPOSALS = Object.freeze([
  {role: ROLES[0], path: `${SAVED_PROPOSALS}/code/encoders/causal_patch_bottleneck_mae/versions/current.json`,
    bytes: 130259, sha256: '4159faa7a9177721d338466fe9e6fd260f5e54b393ecbeb0a2c11bd07d1f9204'},
  {role: ROLES[1], path: `${SAVED_PROPOSALS}/code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v14.market-01.json`,
    bytes: 29471, sha256: '77262a4050a88e8eb7a86d8e675aa26522111bc7e406299b43a6c07347a05d02'},
  {role: ROLES[2], path: `${SAVED_PROPOSALS}/doc/clean_causal_encoder_research_ledger.json`,
    bytes: 1987000, sha256: '47360e36128eac9c27a5a4609f946eae2fcd2c6c2cf158bbf2b378bb041dbd45'},
]);
const ARCHIVED_LEDGER = {
  path: '.archive/embeddings/pre-cleanup-2026-10-18/doc/clean_causal_encoder_research_ledger.json',
  bytes: 1746343, sha256: '4ac9a0a3b73225b3ca33915176a2785dd23524d2464021a421a99c33f5464270',
};
const clone = value => parseJSON(jsonBytes(value));
export const sameValue = (left, right) => jsonBytes(left).equals(jsonBytes(right));
const pointerPart = key => String(key).replace(/~/g, '~0').replace(/\//g, '~1');

// Each changed JSON field is recorded, including new fields. Numeric values
// remain Number or BigInt exactly as the shared artifact parser admitted them.
export function fieldAdaptations(before, after, pointer = '') {
  const rows = [];
  const visit = (left, right, location, leftPresent, rightPresent) => {
    if (leftPresent && rightPresent && sameValue(left, right)) return;
    const object = value => value !== null && typeof value === 'object';
    if (leftPresent && rightPresent && object(left) && object(right) &&
        Array.isArray(left) === Array.isArray(right)) {
      for (const key of new Set([...Object.keys(left), ...Object.keys(right)])) {
        visit(left[key], right[key], `${location}/${pointerPart(key)}`,
          Object.hasOwn(left, key), Object.hasOwn(right, key));
      }
      return;
    }
    // An added subtree is one explicit field adaptation, containing its full
    // value. Existing changed subtrees are expanded to their changed leaves.
    const row = {pointer: location, operation: !leftPresent ? 'add' : !rightPresent ? 'remove' : 'replace',
      prior_present: leftPresent, after_present: rightPresent};
    if (leftPresent) row.before = left;
    if (rightPresent) row.after = right;
    row.reason = location.includes('application') || location === '/interpretation' ?
      'Separate the predecessor application decision from current Mityane selection.' :
      'Add explicit post-migration navigation, provenance or closeout binding; historical evidence tuples remain unchanged.';
    rows.push(row);
  };
  visit(before, after, pointer, true, true);
  return rows;
}

function visitPins(value, callback, pointer = '') {
  if (!value || typeof value !== 'object') return;
  if (typeof value.path === 'string' && Number.isSafeInteger(value.bytes) &&
      /^[0-9a-f]{64}$/.test(value.sha256 ?? '')) callback(value, pointer);
  for (const [key, child] of Object.entries(value)) visitPins(child, callback, `${pointer}/${pointerPart(key)}`);
}

function aliasLocators(value, aliases) {
  const located = [], unavailable = [];
  visitPins(value, (pin, pointer) => {
    const repository = pin.repository ?? pin.original_repository ?? 'cuwacunu_embedding';
    const matches = aliases.filter(row => row.original_repository === repository && row.original_path === pin.path &&
      row.expected_bytes === pin.bytes && row.expected_sha256 === pin.sha256);
    const currents = new Set(matches.map(row => row.current_path));
    require(currents.size <= 1, `conflicting current locator for ${pin.path}:${pin.sha256}`);
    if (!matches.length) { unavailable.push({pointer, original_repository: repository, historical_pin: clone(pin)}); return; }
    const row = matches[0];
    located.push({pointer, original_repository: row.original_repository,
      historical_pin: clone(pin), current_path: row.current_path,
      observed_bytes: row.observed_bytes, observed_sha256: row.observed_sha256});
  });
  require(unavailable.length === 0, `unresolved exact historical evidence aliases: ${jsonBytes(unavailable).toString('utf8')}`);
  return {located, historical_pins_without_locator: unavailable,
    policy: 'Historical path/bytes/hash tuples retain their original repository identity. Locators are separate navigation data; no relocated source substitutes for a frozen source capsule.'};
}

// Pure derivation, exported so fixtures can check lossless integer handling,
// immutable history and the complete adaptation set without project execution.
export function deriveRecords({catalogProposal, instanceProposal, ledgerProposal, priorCatalog,
  proposalPins, bindings, aliases, attemptRelative, archivedLedger}) {
  require(catalogProposal.kind === 'causal_encoder_live_catalog_v1', 'catalog proposal kind');
  require(instanceProposal.kind === 'causal_encoder_measured_instance_v1', 'instance proposal kind');
  require(ledgerProposal.kind === 'clean_causal_encoder_research_ledger_v1', 'ledger proposal kind');
  const fresh = catalogProposal.fresh_research_candidate;
  require(fresh?.instance === 'CPB-v14.market-01' && instanceProposal.instance === fresh.instance,
    'one measured V14 instance');
  require(instanceProposal.quality?.numeric_passed === 3 && instanceProposal.quality.numeric_total === 22 &&
    instanceProposal.quality.strict_new_scoped_quality_leaders === 0 &&
    fresh.numeric_passed === 3 && fresh.numeric_total === 22 &&
    fresh.advancement_passed === false && instanceProposal.advancement_passed === false,
    'saved failed 3/22 V14 outcome required');
  require(catalogProposal.clean_causal_training_baseline?.instance === 'CPB-v3.market-01' &&
    instanceProposal.clean_baseline === 'CPB-v3.market-01' && ledgerProposal.baseline === 'CPB-v3.market-01',
    'V3 baseline must remain unchanged');
  const previous = priorCatalog.fresh_research_candidate_history;
  const proposed = catalogProposal.fresh_research_candidate_history;
  require(Array.isArray(previous) && Array.isArray(proposed) && proposed.length === previous.length + 1 &&
    previous.every((row, index) => sameValue(row, proposed[index])) &&
    sameValue(proposed.at(-1), priorCatalog.fresh_research_candidate),
    'prior catalog history must remain an exact prefix with intact prior V13 appended');
  for (const key of Object.keys(priorCatalog)) {
    if (!['fresh_research_candidate', 'fresh_research_candidate_history'].includes(key)) {
      require(sameValue(priorCatalog[key], catalogProposal[key]), `unexpected proposal catalog change: ${key}`);
    }
  }
  require(priorCatalog.fresh_research_candidate?.instance === 'CPB-v13.market-01', 'expected actual prior V13 catalog');
  const catalog = clone(catalogProposal), instance = clone(instanceProposal);
  const predecessor = {
    repository: 'cuwacunu_embedding',
    catalog: {active_application: catalog.active_application, interpretation: catalog.interpretation,
      application_selection_record: catalog.application_selection_record},
    instance: {application_selection: instance.application_selection,
      application_selection_changed: instance.application_selection_changed},
    policy: 'These are preserved predecessor decisions. They do not establish a Mityane project-wide encoder selection.',
  };
  catalog.active_application = null;
  catalog.application_selection_record = null;
  catalog.interpretation = 'V3 is the clean causal research baseline. V4 is the broad clean-causal breakthrough; later versions retain scoped improvements. Mityane has no project-wide application encoder selection. The latest measured candidate is V14, whose exact failed recipe is closed.';
  catalog.historical_predecessor_application = clone(predecessor);
  catalog.mityane_application_selection = {instance: null, status: 'none_established_by_this_closeout'};
  const oldNavigation = {};
  for (const key of ['design_file', 'physical_version_root', 'measured_instance_file', 'instance_record']) {
    if (Object.hasOwn(fresh, key)) oldNavigation[key] = clone(fresh[key]);
  }
  catalog.fresh_research_candidate.design_file = `${FAMILY}/CPB-v14.json`;
  catalog.fresh_research_candidate.physical_version_root = `${FAMILY}/v14`;
  catalog.fresh_research_candidate.measured_instance_file = TARGETS[1];
  if (Object.hasOwn(fresh, 'instance_record')) catalog.fresh_research_candidate.instance_record = TARGETS[1];
  catalog.fresh_research_candidate.navigation_path_base = 'Mityane repository root';
  catalog.fresh_research_candidate.historical_navigation = {
    original_repository: 'cuwacunu_embedding', fields: oldNavigation,
    source_identity_policy: 'Frozen source identities remain bound to original paths and source-manifest bytes; maintained locations have separate identities.',
  };
  instance.application_selection = null;
  instance.application_selection_changed = false;
  instance.historical_predecessor_application = clone(predecessor);
  instance.mityane_application_selection = {instance: null, status: 'none_established_by_this_closeout'};
  instance.historical_source_identity = {original_repository: 'cuwacunu_embedding',
    version_root: instance.version_root, source_manifest: clone(instance.source_manifest),
    source_capsule: clone(instance.source_capsule),
    policy: 'Historical execution used this frozen source identity. Maintained relocated or edited code is not substituted for it.'};
  instance.version_root = `${FAMILY}/v14`;
  instance.navigation_path_base = 'Mityane repository root';
  instance.current_locations = {
    design_file: `${FAMILY}/CPB-v14.json`, measured_instance_file: TARGETS[1],
    local_run_root: `code/embeddings/${instance.local_run_root}`,
    local_quality_root: `code/embeddings/${instance.local_quality_root}`,
    local_engineering_root: `code/embeddings/${instance.local_engineering_root}`,
    historical_runtime_root: instance.container_root,
  };
  instance.evidence_locators = aliasLocators(instanceProposal, aliases);
  catalog.fresh_research_candidate.evidence_locators = aliasLocators(fresh, aliases);
  const binding = {protocol: PROTOCOL, derivation: 'reviewed historical proposal DATA with explicit field adaptations',
    immutable_preapplication_inputs: clone(bindings), historical_proposals: clone(proposalPins),
    summary: TARGETS[2], research_execution_authorized: false};
  catalog.metadata_closeout = clone(binding);
  instance.metadata_closeout = clone(binding);
  const summary = {
    kind: 'cpb_v14_postmigration_metadata_closeout_v1', date: '2026-10-18',
    instance: 'CPB-v14.market-01', protocol: PROTOCOL,
    status: 'reviewed_metadata_derivation_application_and_preservation_recorded_separately',
    scientific_conclusion: {clean_research_baseline: 'CPB-v3.market-01',
      broad_clean_causal_breakthrough: 'CPB-v4.market-01',
      historical_balanced_leader: 'RPB-v24.market-01.alt-01 native36',
      measured_candidate_established: true, numeric_passed: 3, numeric_total: 22,
      strict_new_scoped_quality_leaders: 0, advancement_passed: false,
      exact_v14_recipe_closed: true, mityane_application_selection: null,
      fresh_weights_versions: 'CPB-v3 through CPB-v14; source-design branches do not imply accumulated trained weights'},
    immutable_preapplication_inputs: clone(bindings), historical_proposals: clone(proposalPins),
    preserved_archived_ledger: clone(archivedLedger),
    ledger_data_policy: 'The reviewed ledger proposal is immutable historical DATA. Its leaders, scores, histories and failures remain there; the archived prior ledger is unchanged. No parallel mutable progress ledger is created.',
    maintained_records: TARGETS.slice(0, 2),
    future_receipt_locations: {application: `${attemptRelative}/application.json`,
      preservation: `${attemptRelative}/preservation.json`},
    future_receipt_policy: 'Locations name future records and claim no result. Those records bind actual applied bytes; this summary does not hash itself or any later receipt.',
    scope: {metadata_only: true, new_training: false, model_or_head_replay: false,
      head_refits: false, python_executed_by_protocol: false, checkpoint_selection: false},
    next_scientific_work: {status: 'requires_separate_saved_TRAIN_only_protocol',
      question: 'Window overlap, native covariance and actual saved shared-position adjoints; two captured batches do not establish a complete training-trajectory explanation.',
      validation_tensor_diagnosis: false, v15_selected: false},
  };
  return {catalog, instance, summary, adaptations: [
    {target: TARGETS[0], proposal_role: 'catalog_proposal', fields: fieldAdaptations(catalogProposal, catalog)},
    {target: TARGETS[1], proposal_role: 'instance_proposal', fields: fieldAdaptations(instanceProposal, instance)},
    {target: TARGETS[2], proposal_role: null, fields: [{pointer: '', operation: 'add',
      prior_present: false, after_present: true, after: summary,
      reason: 'Create the admitted current lane-owned summary; no historical summary bytes are claimed.'}]},
  ]};
}

function absolute(root, relative) { return path.join(root, safeRelative(relative)); }
function readBound(root, relative) {
  const filename = absolute(root, relative), row = admitFiles([filename], {root})[0];
  const bytes = readAdmitted(row);
  return {pin: {path: relative, bytes: bytes.length, sha256: sha256(bytes)}, data: parseJSON(bytes)};
}
function expectPin(actual, expected, message) {
  require(expected && actual.path === expected.path && actual.bytes === expected.bytes && actual.sha256 === expected.sha256,
    message);
}
function relativePin(root, pin) {
  require(pin && typeof pin.path === 'string', 'explicit receipt pin required');
  if (!path.isAbsolute(pin.path)) return pin;
  require(inside(root, pin.path), 'receipt pin escapes repository');
  return {...pin, path: path.relative(root, pin.path).split(path.sep).join('/')};
}
function pathExists(filename) {
  try { fs.lstatSync(filename); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}
function checkTargets(root, targets) {
  for (const expected of targets) {
    const filename = absolute(root, expected.path);
    if (!expected.exists) {
      require(!pathExists(filename), `expected absent target already exists: ${expected.path}`);
      let parent = path.dirname(filename);
      while (!pathExists(parent)) parent = path.dirname(parent);
      canonicalDirectory(parent, {root});
      continue;
    }
    const row = admitFiles([filename], {root})[0];
    require(row.bytes === expected.bytes && hashAdmitted(row) === expected.sha256,
      `target differs from current before pin: ${expected.path}`);
  }
}

export function stage({root, attempt, admissionPath, inputsPath, aliasManifestPath,
  aliasVerificationPath, proposalReviewPath, sourceReviewPath}) {
  canonicalDirectory(root); canonicalDirectory(attempt, {root});
  const relativeAttempt = path.relative(root, attempt).split(path.sep).join('/');
  require(/^\.temp\/embeddings\/closeout\/cpb-v14\/attempt-\d+$/.test(relativeAttempt), 'named V14 attempt directory required');
  const loaded = Object.fromEntries(Object.entries({admission: admissionPath, inputs: inputsPath,
    aliases: aliasManifestPath, alias_verification: aliasVerificationPath,
    proposal_review: proposalReviewPath, source_review: sourceReviewPath})
    .map(([role, relative]) => [role, readBound(root, relative)]));
  const {admission, inputs, aliases, alias_verification: aliasVerification,
    proposal_review: proposalReview, source_review: sourceReview} = loaded;
  require(admission.data.kind === 'cpb_v14_postmigration_closeout_admission_v1' &&
    admission.data.scope?.compatible_reviewed_catalog_instance_summary_application === true &&
    admission.data.scope.new_model_training === false && admission.data.scope.old_encoder_or_head_replay === false &&
    admission.data.scope.head_refits === false, 'metadata-only human resumption admission required');
  require(admission.data.artifact_root === relativeAttempt, 'admission belongs to this attempt');
  require(sourceReview.data.passed === true && Array.isArray(sourceReview.data.files), 'passed independent source review required');
  for (const name of [`${PROTOCOL}/stage.mjs`, SHARED]) {
    const observed = readBoundSource(root, name);
    const pin = sourceReview.data.files.find(row => row.path === name);
    require(pin, `source review omits ${name}`); expectPin(observed, pin, `source review stale for ${name}`);
  }
  require(aliasVerification.data.passed === true && aliasVerification.data.status === 'passed' &&
    proposalReview.data.passed === true && proposalReview.data.status === 'passed',
    'passed independent proposal and alias reviews required before staging');
  expectPin(aliases.pin, relativePin(root, aliasVerification.data.alias_manifest),
    'alias verification does not bind the actual manifest');
  expectPin(aliases.pin, relativePin(root, proposalReview.data.alias_manifest),
    'proposal review used another alias manifest');
  expectPin(aliasVerification.pin, relativePin(root, proposalReview.data.alias_verification),
    'proposal review used another alias verification result');
  const targetRows = TARGETS.map(name => {
    const matches = inputs.data.targets?.filter(row => row.path === name) ?? [];
    require(matches.length === 1, `one actual before binding required: ${name}`);
    const admitted = admission.data.target_observations?.find(row => row.path === name);
    require(admitted && sameValue(matches[0], admitted), `inputs/admission before pin conflict: ${name}`);
    return matches[0];
  });
  checkTargets(root, targetRows);
  const proposalPins = ROLES.map(role => {
    const rows = inputs.data.proposals?.filter(row => row.role === role) ?? [];
    require(rows.length === 1, `one proposal input required for ${role}`);
    const expected = rows[0], loadedProposal = readBound(root, expected.path);
    expectPin(loadedProposal.pin, expected, `proposal changed: ${role}`);
    expectPin(loadedProposal.pin, ORIGINAL_PROPOSALS.find(row => row.role === role),
      `only the immutable saved original proposal is admitted: ${role}`);
    const reviewed = proposalReview.data.inputs?.find(row => row.role === role);
    require(reviewed, `proposal review omits ${role}`);
    expectPin(loadedProposal.pin, relativePin(root, reviewed), `proposal review does not bind ${role}`);
    return {role, ...loadedProposal.pin, data: loadedProposal.data};
  });
  const prior = readBound(root, TARGETS[0]);
  const archivedLedger = readBound(root, ARCHIVED_LEDGER.path).pin;
  expectPin(archivedLedger, ARCHIVED_LEDGER, 'preserved archived ledger differs from its immutable generation');
  const bindings = Object.fromEntries(Object.entries(loaded).map(([role, item]) => [role, item.pin]));
  const proposalByRole = Object.fromEntries(proposalPins.map(item => [item.role, item.data]));
  require(Array.isArray(aliases.data.aliases), 'explicit aliases array required');
  const aliasRows = aliases.data.aliases.map(row => {
    require(row.original && row.observed && typeof row.original.repository === 'string' &&
      typeof row.current_path === 'string' && row.original.bytes === row.observed.bytes &&
      row.original.sha256 === row.observed.sha256, 'verified exact alias tuple required');
    safeRelative(row.current_path);
    return {original_repository: row.original.repository, original_path: row.original.path,
      expected_bytes: row.original.bytes, expected_sha256: row.original.sha256,
      current_path: row.current_path, observed_bytes: row.observed.bytes, observed_sha256: row.observed.sha256};
  });
  const descriptor = readBoundSource(root, `${FAMILY}/CPB-v14.json`);
  require(descriptor.bytes === proposalByRole.instance_proposal.prospective_descriptor?.bytes &&
    descriptor.sha256 === proposalByRole.instance_proposal.prospective_descriptor?.sha256,
    'maintained prospective descriptor no longer matches saved immutable descriptor');
  canonicalDirectory(absolute(root, `${FAMILY}/v14`), {root});
  for (const key of ['local_run_root', 'local_quality_root', 'local_engineering_root']) {
    const original = proposalByRole.instance_proposal[key];
    require(typeof original === 'string' && original.startsWith('output/'), `historical ${key} must be explicit`);
    safeRelative(original);
    canonicalDirectory(absolute(root, `code/embeddings/${original}`), {root});
  }
  const results = deriveRecords({catalogProposal: proposalByRole.catalog_proposal,
    instanceProposal: proposalByRole.instance_proposal, ledgerProposal: proposalByRole.ledger_proposal,
    priorCatalog: prior.data, proposalPins: proposalPins.map(({data, ...pin}) => pin),
    bindings, aliases: aliasRows, attemptRelative: relativeAttempt, archivedLedger});
  // Recheck just before creating exclusive output leaves. Never write live files.
  checkTargets(root, targetRows);
  for (const {pin} of Object.values(loaded)) expectPin(readBound(root, pin.path).pin, pin, `input changed before stage: ${pin.path}`);
  for (const {data, ...pin} of proposalPins) expectPin(readBound(root, pin.path).pin, pin,
    `proposal changed before stage: ${pin.path}`);
  expectPin(readBoundSource(root, archivedLedger.path), archivedLedger, 'archive changed before stage');
  expectPin(readBoundSource(root, descriptor.path), descriptor, 'descriptor changed before stage');
  const stagedDirectory = path.join(attempt, 'staged');
  const manifestPath = path.join(attempt, 'staged-change-manifest.json');
  require(!fs.existsSync(stagedDirectory) && !fs.existsSync(manifestPath), 'staged generation already exists; preserve it and choose another attempt');
  createExclusiveDirectory(stagedDirectory, {root: attempt});
  const documents = [results.catalog, results.instance, results.summary];
  const outputs = TARGETS.map((target, index) => {
    const filename = path.join(stagedDirectory, target);
    const parent = path.dirname(filename);
    if (!fs.existsSync(parent)) createExclusiveDirectory(parent, {root: stagedDirectory});
    const bytes = jsonBytes(documents[index], {sortKeys: false});
    writeNew(filename, bytes);
    return {path: target, staged_path: `${relativeAttempt}/staged/${target}`, bytes: bytes.length,
      sha256: sha256(bytes), expected_before: targetRows[index]};
  });
  const manifest = {kind: 'cpb_v14_postmigration_staged_change_manifest_v1',
    status: 'staged_only_not_applied', artifact_json_encoding: 'mityane-artifact-json-v1',
    protocol: PROTOCOL, immutable_preapplication_inputs: bindings,
    historical_proposals: proposalPins.map(({data, ...pin}) => pin),
    outputs, field_adaptations: results.adaptations,
    byte_policy: 'New maintained JSON encoding; historical proposal bytes are immutable. Float spelling may differ; numeric values, exact unsafe integers and unchanged proposal fields are preserved.',
    required_next_step: 'Independently review staged content and every adaptation, then recheck expected-before pins immediately before any application.',
    applied: false};
  writeNew(manifestPath, jsonBytes(manifest));
  return {manifest_path: `${relativeAttempt}/staged-change-manifest.json`,
    manifest_sha256: sha256(jsonBytes(manifest)), outputs};
}

function readBoundSource(root, relative) {
  const row = admitFiles([absolute(root, relative)], {root})[0];
  return {path: relative, bytes: row.bytes, sha256: hashAdmitted(row)};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const flag = process.argv[index];
    require(/^--[a-z-]+$/.test(flag) && process.argv[index + 1], 'explicit paired stage arguments required');
    require(!Object.hasOwn(args, flag), `duplicate stage argument ${flag}`);
    args[flag] = process.argv[index + 1];
  }
  const names = {'--root': 'root', '--attempt': 'attempt', '--admission': 'admissionPath',
    '--inputs': 'inputsPath', '--aliases': 'aliasManifestPath', '--alias-verification': 'aliasVerificationPath',
    '--proposal-review': 'proposalReviewPath', '--source-review': 'sourceReviewPath'};
  require(Object.keys(args).length === Object.keys(names).length && Object.keys(args).every(key => names[key]), 'all eight named stage arguments required');
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv') && args['--root'] === '/mityane',
    'stage CLI requires the admitted managed Linux container at /mityane');
  const options = Object.fromEntries(Object.entries(names).map(([flag, name]) => [name, args[flag]]));
  process.stdout.write(jsonBytes(stage(options)));
}
