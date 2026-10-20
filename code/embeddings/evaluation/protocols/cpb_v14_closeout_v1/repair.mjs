// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Additive repair of an observed failed alias closure. Preserved tools are DATA.
import { createHash } from 'node:crypto';
import { lstat, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual as same } from 'node:util';
import { admitFiles, readAdmitted, parseJSON, jsonBytes } from '../../../../shared/artifacts/node/v1/files.mjs';
import { ARCHIVE, identity, identityKey, historicalCandidates, hashFile, preservedSnapshotLocator } from './aliases.mjs';

const SELF = fileURLToPath(import.meta.url);
const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/repair.mjs';
const ALIASES_PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/aliases.mjs';
const MANIFEST_KIND = 'cpb_v14_historical_identity_alias_manifest_v1';
const VERIFICATION_KIND = 'cpb_v14_historical_identity_alias_verification_v1';
const REPO = 'cuwacunu_embedding';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const bytesEqual = (a, b) => a?.bytes === b?.bytes && a?.sha256 === b?.sha256;
const fail = message => { throw new Error(message); };
const pinOnly = value => ({ path: value.path, bytes: value.bytes, sha256: value.sha256 });
const full = (pin, repository = REPO) => identity(pin, repository);
const quotePointer = value => value.replaceAll('~', '~0').replaceAll('/', '~1');

function relative(value) {
  if (typeof value !== 'string' || !value || /[\\:\0\r\n\t]/.test(value)
      || path.posix.isAbsolute(value) || value.split('/').some(part => !part || part === '.' || part === '..')) fail(`Exact relative locator required: ${value}`);
  return value;
}
async function currentPath(root, locator) {
  relative(locator);
  const absolute = path.resolve(root, locator);
  if (await realpath(root) !== path.resolve(root) || await realpath(absolute) !== absolute) fail(`Redirected locator: ${locator}`);
  const info = await lstat(absolute);
  if (!info.isFile() || info.isSymbolicLink()) fail(`Canonical regular locator required: ${locator}`);
  return { absolute, bytes: info.size };
}
function localLocator(root, inputPath) {
  if (path.isAbsolute(inputPath)) {
    const rel = path.relative(root, inputPath).split(path.sep).join('/');
    return relative(rel);
  }
  return relative(inputPath);
}
export function sameCurrentIdentity(root, left, right) {
  const a = identity(left, 'Mityane-geometrics');
  const b = identity(right, 'Mityane-geometrics');
  return a.repository === 'Mityane-geometrics' && b.repository === a.repository
    && localLocator(root, a.path) === localLocator(root, b.path) && bytesEqual(a, b);
}
function *pins(value, pointer = '') {
  if (!value || typeof value !== 'object') return;
  if (!Array.isArray(value) && typeof value.path === 'string' && Number.isSafeInteger(value.bytes) && /^[0-9a-f]{64}$/.test(value.sha256 ?? '')) yield { pin: value, pointer };
  for (const [key, child] of Object.entries(value)) yield *pins(child, `${pointer}/${quotePointer(key)}`);
}
export function atPointer(value, pointer) {
  if (pointer === '') return value;
  if (typeof pointer !== 'string' || !pointer.startsWith('/')) fail('An exact JSON pointer is required');
  for (const part of pointer.slice(1).split('/')) {
    const key = part.replaceAll('~1', '/').replaceAll('~0', '~');
    if (!value || typeof value !== 'object' || !Object.hasOwn(value, key)) fail(`Missing JSON pointer: ${pointer}`);
    value = value[key];
  }
  return value;
}

/** Only named historical mirror schemas may establish container/host pairs. */
export function mirrorMappings(data, binding, { repository = REPO } = {}) {
  const mappings = [];
  const kind = data?.kind ?? '';
  const version = '(?:5|6|7|8|9|10|11|12|13|14)';
  const mappedKind = new RegExp(`^cpb_v${version}_(?:training_independent_audit|quality_independent_audit|metadata_establishment|finalization_complete_container)_mirror_v1$`).test(kind)
    || kind === 'cpb_v14_paused_finalization_migration_mirror_v1';
  const fileKind = new RegExp(`^causal_fresh_v${version}_container_to_local_evidence_mirror_v1$`).test(kind)
    || kind === 'causal_fresh_v4_container_to_local_evidence_mirror_v1'
    || kind === 'causal_fresh_v6_proof_recovery_container_to_local_evidence_mirror_v1'
    || /^cpb_v(?:9|10|11|12|13|14)_(?:seven_build_binaries|actual_build_binary)_mirror_v1$/.test(kind);
  const trainPair = '(?:6_v7|6_v8|8_v9|9_v11|11_v12|11_v13)';
  const nestedFilesKind = new RegExp(`^cpb_v${trainPair}_TRAIN_independent_audit_mirror_v1$`).test(kind)
    || /^clean_causal_v(?:7|8|9|11)_retention_complete_mirror_v1$/.test(kind);
  const relativeFilesKind = new RegExp(`^cpb_v${trainPair}_TRAIN_diagnosis_exact_mirror_v1$`).test(kind);
  const rootedFlatKind = kind === 'causal_fresh_v6_quality_audit_attempt002_mirror_v1'
    || kind === 'cpb_v5_v6_independent_TRAIN_audit_exact_mirror_v1';
  if (!mappedKind && !fileKind && !nestedFilesKind && !relativeFilesKind && !rootedFlatKind) return mappings;
  if (repository !== REPO) fail('Known embedding mirror requires its original embedding repository context');
  if (!['passed', 'passed_file_preservation_only'].includes(data.status)) fail(`Historical mirror did not pass preservation: ${kind}`);
  if (relativeFilesKind && (data.all_sizes_sha256_exact !== true || data.container_before_after_exact !== true)) fail(`TRAIN diagnosis requires both exact preservation flags: ${kind}`);
  if (nestedFilesKind && data.all_hashes_exact !== true) fail(`Historical mapped inventory requires its exact hash flag: ${kind}`);
  if (kind === 'cpb_v5_v6_independent_TRAIN_audit_exact_mirror_v1' && data.all_container_bytes_preserved !== true) fail('V5/V6 independent audit requires preserved container bytes');
  const rows = mappedKind ? data.mapped_files : data.files;
  if (!Array.isArray(rows)) fail(`Exact historical mirror member array required: ${kind}`);
  if (Number.isInteger(data.file_count) && data.file_count !== rows.length) fail(`Historical mirror member count mismatch: ${kind}`);
  const seen = new Set();
  for (const [index, row] of rows.entries()) {
    let host, container, memberRelative;
    if (relativeFilesKind) {
      const member = relative(row.path);
      const hostRoot = legacyRepositoryRelative(data.host_root);
      if (typeof data.container_root !== 'string' || !data.container_root.startsWith('/opt/cuwacunu/research/')) fail('Exact TRAIN container root required');
      host = full({ ...row, path: `${hostRoot}/${member}` });
      container = `${data.container_root}/${member}`;
      memberRelative = member;
    } else {
      host = full(mappedKind || nestedFilesKind ? row.host : row);
      container = mappedKind || nestedFilesKind ? row.container_path ?? row.container : row.container_path;
      memberRelative = row.relative;
    }
    if (typeof container !== 'string' || !container.startsWith('/opt/cuwacunu/') || container.includes('/../')
        || (row.container && row.container_path && row.container !== row.container_path)) fail(`Exact historical container member required: ${kind}/${index}`);
    if ((nestedFilesKind || rootedFlatKind) && memberRelative === undefined) fail(`Exact relative member is mandatory for the declared roots: ${kind}/${index}`);
    if (memberRelative !== undefined) {
      relative(memberRelative);
      if (!container.endsWith(`/${memberRelative}`) || !host.path.replaceAll('\\', '/').endsWith(`/${memberRelative}`)) fail(`Mirror relative member mismatch: ${kind}/${index}`);
      if (data.container_root && container !== `${data.container_root}/${memberRelative}`) fail(`Mirror root/member mismatch: ${kind}/${index}`);
      if (nestedFilesKind && host.path !== `${legacyRepositoryRelative(data.host_root)}/${memberRelative}`) fail(`Mirror declared host root mismatch: ${kind}/${index}`);
      if (rootedFlatKind && host.path !== `${legacyRepositoryRelative(data.local_root)}/${memberRelative}`) fail(`Mirror declared local root mismatch: ${kind}/${index}`);
    }
    if (seen.has(container)) fail(`Duplicate historical container member: ${container}`);
    seen.add(container);
    mappings.push({ original: { ...host, path: container }, host,
      historical_spellings: relativeFilesKind ? [full(row)] : [],
      provenance: { manifest: binding, pointer: `/${mappedKind ? 'mapped_files' : 'files'}/${index}` } });
  }
  return mappings;
}

function legacyRepositoryRelative(value) {
  if (typeof value !== 'string') fail('An explicit historical host root is required');
  let normal = value.replaceAll('\\', '/');
  const oldRoot = 'C:/Work/cuwacunu/cuwacunu_embedding/';
  if (normal.startsWith(oldRoot)) normal = normal.slice(oldRoot.length);
  relative(normal);
  if (!normal.startsWith('output/')) fail(`Historical mirror host root must belong to the recorded embedding output namespace: ${value}`);
  return normal;
}

/** Exact source/destination byte-copy receipts; never a global hash lookup. */
export function byteCopyMappings(data, binding, { repository = REPO } = {}) {
  const kind = data?.kind ?? '';
  const v4 = kind === 'causal_fresh_v4_durable_saved_evidence_mirror_v1';
  const v5 = kind === 'causal_fresh_v5_durable_evidence_mirror_v1';
  const v9 = kind === 'cpb_v9_curated_durable_JSON_bytecopy_mirror_v1';
  const later = /^cpb_v(?:10|11|12|13|14)_curated_durable_JSON_bytecopy_mirror_v1$/.test(kind);
  if (!v4 && !v5 && !v9 && !later) return [];
  if (repository !== REPO || data.status !== 'passed') fail('An explicitly passed embedding byte-copy receipt is required');
  const rows = later ? data.copied_files : data.files;
  if (!Array.isArray(rows)) fail('Exact durable byte-copy member array required');
  if (Number.isInteger(data.file_count) && data.file_count !== rows.length) fail('Durable byte-copy member count mismatch');
  return rows.map((row, index) => {
    const original = full(v4 || v5 ? { path: row.source, bytes: row.bytes, sha256: row.sha256 } : row.source);
    const destination = full(v4 ? row : v5 ? { path: row.destination, bytes: row.bytes, sha256: row.sha256 } : row.destination);
    if (!bytesEqual(original, destination)) fail('Durable source/destination full tuples must have equal bytes/hash');
    const version = /^cpb_v(\d+)_/.exec(kind)?.[1];
    const canonicalNames = v9 ? ['reporting-recovery-card-001.json', 'reporting-recovery-receipt-001.json'] : later ? ['prospective-freeze.json', 'protocol-preflight.json'] : [];
    const canonicalCopy = canonicalNames.some(name => original.path === `doc/causal_fresh_v${version}/${name}` && destination.path === `doc/results/causal_fresh_v${version}/${name}`);
    if ((v4 && !original.path.replaceAll('\\', '/').startsWith('C:/Work/cuwacunu/cuwacunu_embedding/'))
        || (!v4 && !original.path.startsWith('output/') && !canonicalCopy) || !destination.path.startsWith('doc/results/')) fail('Durable byte-copy repository namespace mismatch');
    if (v9 && (row.byte_exact !== true || row.path !== destination.path || !bytesEqual(row, destination))) fail('Exact V9 byte-copy wrapper binding required');
    return { original, destination, provenance: { manifest: binding, pointer: `/${later ? 'copied_files' : 'files'}/${index}` } };
  });
}

function sourceCapsule(data) {
  return /^causal_fresh_v(?:5|6|7|8|9|10|11|12|13|14)_source_capsule_v1$/.test(data?.kind ?? '')
    && typeof data.container_root === 'string' && data.container_root.startsWith('/opt/cuwacunu/research/') && Array.isArray(data.files);
}

export function repairObservationFields({ baseManifest, baseVerification, basePair, sourceReview, protocolIdentity, sharedParser }) {
  if (baseVerification.status !== 'failed' || baseVerification.passed !== false) fail('Original bulk failure status must remain explicit');
  for (const value of [basePair.manifest, basePair.verification, sourceReview, protocolIdentity, sharedParser]) identity(value, 'Mityane-geometrics');
  return { protocol_identity: pinOnly(protocolIdentity), shared_parser_identity: pinOnly(sharedParser), base_pair: basePair,
    original_bulk_observation: { manifest_created_utc: baseManifest.created_utc, verification_created_utc: baseVerification.created_utc,
      verification_status: baseVerification.status, protocol_identity: baseManifest.protocol_identity, source_review: sourceReview,
      limitation: 'Original receipt dates bind the completed original observation. No per-file observation times or fresh-all claim.' },
    bulk_observation_reused: true, all_aliases_freshly_hashed_this_generation: false };
}

// Only independent canonical/regular/size checks use this bounded concurrency.
// Every unique path is checked, including paths queued after another has failed.
// No byte observation or metadata-graph dependency is parallelized here.
export async function inspectBaseLocators(root, plans) {
  const locators = new Set();
  for (const plan of plans) {
    relative(plan.current_path);
    full(plan.expected);
    if (locators.has(plan.current_path)) fail(`A unique base locator check plan is required: ${plan.current_path}`);
    locators.add(plan.current_path);
  }
  const checks = new Array(plans.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(4, plans.length) }, async () => {
    for (;;) {
      const index = next++;
      if (index >= plans.length) return;
      const plan = plans[index];
      try {
        const current = await currentPath(root, plan.current_path);
        if (current.bytes !== plan.expected.bytes) fail(`Original observation locator size changed: ${plan.current_path}`);
        checks[index] = { current_path: plan.current_path, bytes: current.bytes, passed: true, checked_utc: new Date().toISOString() };
      } catch (error) {
        checks[index] = { current_path: plan.current_path, passed: false, checked_utc: new Date().toISOString(), error: error.message };
      }
    }
  }));
  const failures = checks.filter(check => !check.passed);
  if (failures.length) {
    const error = new AggregateError(failures.map(check => new Error(check.error)),
      `Base canonical locator checks failed: ${failures.map(check => `${check.current_path}: ${check.error}`).join('; ')}`);
    error.locator_checks = checks;
    throw error;
  }
  return checks;
}

export function validateScopeDeclaration(scope, originalFailures) {
  if (scope?.kind !== 'cpb_v14_typed_bounded_scope_v1' || !Array.isArray(scope.required_pins)
      || !Array.isArray(scope.control_documents) || !Array.isArray(scope.mapping_documents)
      || !Array.isArray(scope.historical_failure_classification)) fail('A complete typed bounded scope declaration is required');
  const controls = new Map();
  for (const row of [...scope.control_documents, ...scope.mapping_documents]) {
    const original = full(row.original);
    relative(row.current_path);
    if (!bytesEqual(original, row.observed) || !row.current_path.endsWith('.json')) fail('Typed control documents require exact observed JSON identities');
    const key = identityKey({ repository: 'Mityane-geometrics', path: row.current_path, ...row.observed });
    controls.set(key, row);
  }
  const required = new Set();
  for (const row of scope.required_pins) {
    const key = identityKey(row.original);
    if (required.has(key) || !Array.isArray(row.known_locators) || !Array.isArray(row.required_by) || !row.required_by.length) fail('Unique typed required pins, provenance and declared locators are mandatory');
    required.add(key);
    for (const locator of row.known_locators) relative(locator);
    for (const source of row.required_by) {
      if (!controls.has(identityKey(source.manifest, 'Mityane-geometrics')) || typeof source.pointer !== 'string') fail('Every required pin must be selected by an explicit whole-byte-bound control document');
    }
  }
  if (scope.historical_failure_classification.length !== originalFailures.length) fail('Every original failure needs one explicit scope classification');
  const seen = new Set();
  for (const row of scope.historical_failure_classification) {
    const index = row.original_failure_index;
    if (!Number.isInteger(index) || seen.has(index) || !originalFailures[index]
        || !same(full(row.original_identity), full(originalFailures[index].identity)) || row.prior_failure_retained !== true
        || !Object.hasOwn(row, 'record_qualified_identity')
        || !['required_current_scope', 'outside_declared_traversal', 'record_schema_overinterpretation'].includes(row.disposition)) fail('Original failure scope classification differs from preserved failed evidence');
    seen.add(index);
    const qualified = row.record_qualified_identity === null ? null : full(row.record_qualified_identity);
    const qualifiedRequired = qualified !== null && required.has(identityKey(qualified));
    if (row.required_in_scope !== qualifiedRequired) fail('Record-qualified failure requirement must match the declared required pin set');
    if (qualified === null && row.disposition !== 'record_schema_overinterpretation') fail('A non-file schema object requires explicit schema-overinterpretation classification');
  }
  return scope;
}

/** Metadata rewalk, reusing explicitly bound original successful observations. */
export async function repairAliasGraph({ root, baseManifest, baseVerificationBinding, originalFailures,
  external = null, externalBinding = null, inventory = null, extraDocuments = [], requiredScope = null, requiredScopeBinding = null }) {
  root = path.resolve(root);
  if (requiredScope) validateScopeDeclaration(requiredScope, originalFailures);
  const aliases = new Map();
  const physical = new Map();
  const documents = new Map();
  const graphNodes = new Map();
  const jobs = new Map();
  const mirrorPairs = new Map();
  const contexts = new Map();
  const adapterErrors = new Map();
  const documentErrors = new Map();
  const requiredRows = requiredScope ? new Map(requiredScope.required_pins.map(row => [identityKey(row.original), row])) : null;
  if (requiredRows && requiredRows.size !== requiredScope.required_pins.length) fail('Bounded required identities must be unique');
  const declaredDocuments = requiredScope ? [...requiredScope.control_documents, ...requiredScope.mapping_documents] : [];
  const declaredDocumentKeys = new Set(declaredDocuments.map(row => identityKey(row.original)));
  const wanted = pin => !requiredRows || requiredRows.has(identityKey(pin)) || declaredDocumentKeys.has(identityKey(pin));
  let freshHashReads = 0;
  let metadataReads = 0;
  let reusedAliases = 0;
  let baseMetadataChecks = 0;
  let baseDuplicateMetadataChecksReused = 0;
  const baseReference = { manifest: baseVerificationBinding.manifest, verification: baseVerificationBinding.verification };
  const originalKeys = new Set();
  const basePlans = [];
  const locatorPlans = new Map();
  for (const old of baseManifest.aliases) {
    const original = full(old.original);
    const key = identityKey(original);
    if (!bytesEqual(original, old.observed) || !Array.isArray(old.provenance) || old.provenance.length === 0) fail(`Malformed original observation: ${key}`);
    if (originalKeys.has(key)) fail(`Duplicate original alias identity: ${key}`);
    originalKeys.add(key);
    const already = locatorPlans.get(old.current_path);
    if (already && !bytesEqual(already.expected, original)) fail(`Contradictory observations at one physical locator: ${old.current_path}`);
    if (!already) locatorPlans.set(old.current_path, { current_path: old.current_path, expected: original });
    else baseDuplicateMetadataChecksReused++;
    basePlans.push({ old, original, key });
  }
  const checkedLocators = await inspectBaseLocators(root, [...locatorPlans.values()]);
  for (const checked of checkedLocators) {
    const original = locatorPlans.get(checked.current_path).expected;
    physical.set(checked.current_path, { bytes: original.bytes, sha256: original.sha256, reused: true,
      locator_metadata_checked_utc: checked.checked_utc });
    baseMetadataChecks++;
  }
  // Preserve original alias order and mutate the registry only after all base
  // physical checks passed; a failed path cannot leave an accepted partial base.
  for (const { old, original, key } of basePlans) {
    aliases.set(key, { ...old, original, observation_reused_from: baseReference,
      locator_metadata_checked_utc: physical.get(old.current_path).locator_metadata_checked_utc });
    reusedAliases++;
  }
  async function observe(locator, expected) {
    const current = await currentPath(root, locator);
    const previous = physical.get(locator);
    if (previous && bytesEqual(previous, expected)) {
      if (current.bytes !== expected.bytes) fail(`Observed locator size changed: ${locator}`);
      return previous;
    }
    const value = await hashFile(current.absolute);
    freshHashReads++;
    if (!bytesEqual(value, expected)) fail(`Fresh exact identity mismatch: ${locator}`);
    const result = { ...value, reused: false, observed_utc: new Date().toISOString() };
    physical.set(locator, result);
    return result;
  }
  async function add(originalPin, locator, evidence, { forceFresh = false } = {}) {
    const original = full(originalPin);
    const key = identityKey(original);
    const existing = aliases.get(key);
    if (existing) return existing;
    if (!evidence?.length) fail(`Bound provenance required: ${key}`);
    if (forceFresh) physical.delete(locator);
    const observed = await observe(locator, original);
    const record = { original, current_path: locator, observed: { bytes: observed.bytes, sha256: observed.sha256 }, provenance: evidence };
    if (observed.reused) record.observation_reused_from = baseReference;
    else record.fresh_observed_utc = observed.observed_utc;
    aliases.set(key, record);
    return record;
  }
  async function metadata(entry, repository = entry.original.repository) {
    const locator = entry.current_path;
    if (!documents.has(locator)) {
      const target = await currentPath(root, locator);
      const bytes = readAdmitted(admitFiles([target.absolute], { root })[0]);
      metadataReads++;
      const observed = { bytes: bytes.length, sha256: digest(bytes) };
      if (!bytesEqual(observed, entry.original)) fail(`Metadata changed since original observation: ${locator}`);
      physical.set(locator, { ...observed, reused: false, observed_utc: new Date().toISOString() });
      documents.set(locator, { data: parseJSON(bytes, { duplicates: 'last', integers: 'bigint' }),
        binding: { repository: 'Mityane-geometrics', path: locator, ...observed } });
    }
    const nodeKey = JSON.stringify([locator, repository]);
    if (!graphNodes.has(nodeKey)) graphNodes.set(nodeKey, { ...documents.get(locator), locator, repository, adapted: false, walked: false });
    return documents.get(locator);
  }
  async function resolve(pin, repository, evidence) {
    const original = full(pin, repository);
    const key = identityKey(original);
    if (aliases.has(key)) return aliases.get(key);
    const errors = [];
    for (const locator of historicalCandidates(original)) {
      try { return await add(original, locator, [evidence]); } catch (error) { errors.push(error.message); }
    }
    fail(`No exact admitted locator for ${key}; ${errors.join('; ')}`);
  }
  if (external) {
    if (!inventory || !externalBinding) fail('Explicit external receipt and bound inventory required');
    for (const [index, row] of external.entries.entries()) {
      const original = full(row.original);
      const declared = atPointer(inventory, row.provenance.pointer);
      if (original.repository !== 'cuwacunu_dev' || !same(original, full(declared)) || !bytesEqual(original, row.observed)) fail(`External source tuple/provenance mismatch: ${index}`);
      await add(original, row.current_path, [{ manifest: externalBinding, pointer: `/entries/${index}` }, row.provenance], { forceFresh: true });
    }
  }
  if (requiredScope) {
    // Inventory members are whole-file terminals. Only independently admitted
    // typed control/mapping documents may have their JSON bodies interpreted.
    for (const row of declaredDocuments) {
      const original = full(row.original);
      if (!bytesEqual(original, row.observed)) fail('Bounded control document observed tuple mismatch');
      try {
        await metadata({ original, current_path: row.current_path }, original.repository);
      } catch (error) { documentErrors.set(identityKey(original), { identity: original, error: error.message }); }
    }
  } else {
    // Retained prototype behavior for boundary fixtures only. Actual CLI
    // execution requires an independently reviewed typed bounded scope.
    for (const entry of aliases.values()) if (entry.current_path.endsWith('.json')) {
      try { await metadata(entry); } catch (error) { documentErrors.set(identityKey(entry.original), { identity: entry.original, error: error.message }); }
    }
  }
  for (const extra of extraDocuments) {
    if (requiredScope && !declaredDocuments.some(row => same(full(row.original), full(extra.original)) && row.current_path === extra.current_path)) fail('Bounded extra mapping documents must be explicitly declared scope controls');
    const entry = await add(extra.original, extra.current_path, [{ manifest: extra.binding, pointer: 'explicit-bound-mapping-input' }]);
    await metadata(entry, extra.original.repository);
  }
  async function adapter(node) {
    const { data, binding, locator, repository } = node;
    // A failed historical mirror is retained negative evidence. It contributes
    // no successful copy aliases; its declared pins are still walked below.
    const pairs = ['passed', 'passed_file_preservation_only'].includes(data.status) ? mirrorMappings(data, binding, { repository }) : [];
    for (const pair of pairs) {
      if (!wanted(pair.original) && !wanted(pair.host) && !(pair.historical_spellings ?? []).some(wanted)) continue;
      const host = await resolve(pair.host, pair.host.repository, pair.provenance);
      const record = await add(pair.original, host.current_path, [pair.provenance]);
      for (const spelling of pair.historical_spellings ?? []) await add(spelling, host.current_path, [pair.provenance]);
      const key = identityKey(pair.original);
      mirrorPairs.set(key, { ...pair, current_path: record.current_path });
      const list = contexts.get(record.current_path) ?? [];
      if (!list.some(old => identityKey(old.original) === key)) list.push({ ...pair, current_path: record.current_path });
      contexts.set(record.current_path, list);
    }
    const copies = data.status === 'passed' ? byteCopyMappings(data, binding, { repository }) : [];
    for (const copy of copies) {
      if (!wanted(copy.original) && !wanted(copy.destination)) continue;
      let destination;
      try { destination = await resolve(copy.destination, repository, copy.provenance); }
      catch { destination = await resolve(copy.original, repository, copy.provenance); }
      await add(copy.original, destination.current_path, [copy.provenance]);
      await add(copy.destination, destination.current_path, [copy.provenance]);
    }
    if (sourceCapsule(data)) for (const [index, pin] of data.files.entries()) {
      relative(pin.path);
      const qualified = { ...full(pin, repository), path: `${data.container_root}/source/${pin.path}` };
      if (!wanted(full(pin, repository)) && !wanted(qualified)) continue;
      const evidence = { manifest: binding, pointer: `/files/${index}` };
      const source = await resolve(pin, repository, evidence);
      await add(qualified, source.current_path, [evidence]);
    }
    if (data.kind === 'causal_fresh_v14_build_admission_v1' && data.source_manifest && wanted(full(data.source_manifest, repository))) {
      const expected = full(data.source_manifest, repository);
      const candidates = [...graphNodes.values()].filter(candidate => candidate.repository === REPO
        && candidate.data.kind === 'causal_fresh_v14_source_capsule_v1'
        && `${candidate.data.container_root}/source/CAPSULE_MANIFEST.json` === expected.path
        && bytesEqual(candidate.binding, expected));
      if (repository !== REPO || candidates.length === 0) fail('V14 capsule self alias needs the exact build pin and declared matching source-capsule root');
      const capsule = candidates.sort((a, b) => a.locator.localeCompare(b.locator))[0];
      await add(expected, capsule.locator, [{ manifest: binding, pointer: '/source_manifest' },
        { manifest: capsule.binding, pointer: '/container_root', selector: 'exact V14 CAPSULE_MANIFEST self tuple from bound build.source_manifest' }]);
    }
    // Exact whole-copy relationships, never name/hash search across the archive.
    for (const { pin, pointer } of pins(data)) {
      if (typeof pin.snapshot === 'string' && wanted(full(pin, repository))) await add(full(pin, repository), historicalCandidates(full({ path: pin.snapshot, bytes: pin.bytes, sha256: pin.sha256 }, repository))[0], [{ manifest: binding, pointer }]);
    }
    if (data.kind === 'causal_fresh_v5_live_metadata_before_finalization_v1') for (const [i, pin] of data.files.entries()) {
      if (!wanted(full(pin, repository))) continue;
      const snapshot = preservedSnapshotLocator(data.kind, pin);
      const copy = await resolve(snapshot, repository, { manifest: binding, pointer: `/files/${i}/snapshot_path` });
      await add(full(pin, repository), copy.current_path, [{ manifest: binding, pointer: `/files/${i}` }]);
    }
    node.adapted = true;
    adapterErrors.delete(`${locator}|${repository}`);
  }
  async function pairedMember(pin, node, pointer) {
    const declared = full(pin, node.repository);
    relative(declared.path);
    const matches = [];
    for (const context of contexts.get(node.locator) ?? []) {
      const absolute = path.posix.join(path.posix.dirname(context.original.path), declared.path);
      const pair = mirrorPairs.get(identityKey({ ...declared, path: absolute }));
      if (pair && pair.provenance.manifest.path === context.provenance.manifest.path) matches.push(pair);
    }
    const targets = new Set(matches.map(match => match.current_path));
    if (targets.size !== 1) fail(`No unambiguous same-mirror relative source member: ${identityKey(declared)}`);
    return add(declared, matches[0].current_path, [{ manifest: node.binding, pointer }, matches[0].provenance]);
  }
  if (requiredScope) for (const [index, row] of requiredScope.required_pins.entries()) {
    const key = identityKey(row.original);
    const original = full(row.original);
    jobs.set(key, { original, row, node: { binding: requiredScopeBinding, repository: original.repository, locator: '' },
      pointer: `/required_pins/${index}`, resolved: false });
  }
  let changed = true;
  while (changed) {
    const before = [aliases.size, graphNodes.size, jobs.size];
    for (const node of graphNodes.values()) if (!node.adapted) {
      try { await adapter(node); } catch (error) { adapterErrors.set(`${node.locator}|${node.repository}`, { manifest: node.binding, repository: node.repository, error: error.message }); }
    }
    for (const node of graphNodes.values()) if (!node.walked && !requiredScope) {
      for (const { pin, pointer } of pins(node.data)) {
        const original = full(pin, node.repository);
        const key = identityKey(original);
        if (!jobs.has(key)) jobs.set(key, { original, node, pointer, resolved: false });
      }
      node.walked = true;
    }
    for (const job of jobs.values()) if (!job.resolved) {
      try {
        let entry;
        try {
          if (requiredScope) {
            entry = aliases.get(identityKey(job.original));
            const errors = [];
            if (!entry) for (const locator of job.row.known_locators) {
              try { entry = await add(job.original, locator, job.row.required_by); break; }
              catch (error) { errors.push(error.message); }
            }
            if (!entry) fail(`Required exact identity has no declared verified locator: ${identityKey(job.original)}; ${errors.join('; ')}`);
          } else entry = await resolve(job.original, job.node.repository, { manifest: job.node.binding, pointer: job.pointer });
        }
        catch (error) {
          if (requiredScope) throw error;
          if (path.posix.isAbsolute(job.original.path) || /^[A-Za-z]:/.test(job.original.path)) throw error;
          entry = await pairedMember(job.original, job.node, job.pointer);
        }
        if (!requiredScope && entry.current_path.endsWith('.json')) await metadata(entry, job.original.repository);
        job.resolved = true;
        job.error = undefined;
      } catch (error) { job.error = error.message; }
    }
    changed = aliases.size !== before[0] || graphNodes.size !== before[1] || jobs.size !== before[2];
  }
  const failures = [...documentErrors.values(), ...adapterErrors.values(), ...[...jobs.values()].filter(job => !job.resolved).map(job => ({ identity: job.original, manifest: job.node.binding, pointer: job.pointer, error: job.error }))];
  const baseFailureAccounting = [];
  const declaredFailureClasses = requiredScope ? new Map(requiredScope.historical_failure_classification.map(row => [row.original_failure_index, row])) : null;
  for (const [index, old] of originalFailures.entries()) {
    const key = identityKey(old.identity);
    const entry = aliases.get(key);
    const error = documentErrors.get(key);
    const classification = declaredFailureClasses?.get(index);
    if (requiredScope && (!classification || !same(full(classification.original_identity), full(old.identity)) || classification.prior_failure_retained !== true)) fail(`Exact original failure classification missing: ${index}`);
    const qualified = classification && classification.record_qualified_identity === null ? null
      : classification ? full(classification.record_qualified_identity) : full(old.identity);
    const changedContext = qualified === null || identityKey(qualified) !== key;
    const required = requiredRows ? !changedContext && classification.disposition !== 'record_schema_overinterpretation' && requiredRows.has(key) : true;
    const qualifiedRequired = qualified !== null && (requiredRows ? requiredRows.has(identityKey(qualified)) : required);
    const resolved = required && Boolean(entry) && !error && (jobs.get(key)?.resolved ?? !requiredScope);
    const disposition = resolved ? 'verified_required_identity' : required ? 'required_identity_unresolved'
      : changedContext || classification?.disposition === 'record_schema_overinterpretation' ? 'record_schema_overinterpretation' : 'outside_declared_recursive_reverification';
    baseFailureAccounting.push({ index, identity: full(old.identity), required, resolved, disposition,
      record_qualified_identity: qualified, record_qualified_required: qualifiedRequired,
      record_qualified_verified: qualifiedRequired && jobs.get(identityKey(qualified))?.resolved === true,
      current_path: entry?.current_path ?? null, provenance: entry?.provenance ?? [], error: error?.error ?? null });
    if (required && !resolved && !failures.some(row => row.identity && identityKey(row.identity) === key)) failures.push({ identity: old.identity, base_failure_index: index, error: 'Required original failed identity remains unresolved' });
  }
  return { aliases: [...aliases.values()], failures, original_failures: originalFailures, base_failure_accounting: baseFailureAccounting,
    reused_aliases: reusedAliases, new_physical_hash_reads: freshHashReads + metadataReads,
    new_non_metadata_hash_reads: freshHashReads, metadata_documents_rewalked: documents.size,
    base_unique_locator_metadata_checks: baseMetadataChecks, base_duplicate_locator_metadata_checks_reused: baseDuplicateMetadataChecksReused,
    metadata_file_reads: metadataReads, transitive_identities: jobs.size, adapter_contexts: mirrorPairs.size,
    verification_scope: requiredScope ? 'declared_bounded_v14_metadata_closeout' : 'prototype_recursive_graph',
    bounded_scope_verified: Boolean(requiredScope) && failures.length === 0,
    whole_reference_graph_verified: false, all_historical_refs_resolved: false,
    required_pin_count: requiredRows?.size ?? jobs.size, required_pins_verified: [...jobs.values()].filter(job => job.resolved).length,
    outside_declared_recursive_reverification_count: baseFailureAccounting.filter(row => row.disposition === 'outside_declared_recursive_reverification').length,
    record_schema_overinterpretation_count: baseFailureAccounting.filter(row => row.disposition === 'record_schema_overinterpretation').length };
}

async function loadBound(root, pin) {
  if (identity(pin, 'Mityane-geometrics').repository !== 'Mityane-geometrics') fail('Current repair control inputs require their explicit Mityane repository context');
  const locator = localLocator(root, pin.path);
  const target = await currentPath(root, locator);
  const bytes = readAdmitted(admitFiles([target.absolute], { root })[0]);
  const binding = { path: locator, bytes: bytes.length, sha256: digest(bytes) };
  if (!bytesEqual(binding, pin)) fail(`Bound repair input changed: ${locator}`);
  return { data: parseJSON(bytes, { duplicates: 'reject', integers: 'bigint' }), binding };
}
export async function repairAliasInventory({ root, inputs, inputsBinding }) {
  root = path.resolve(root);
  if (inputs.kind !== 'cpb_v14_alias_repair_inputs_v1') fail('An explicit alias repair input card is required');
  if (!inputs.bounded_scope || !inputs.scope_review) fail('Actual repair requires an exact typed bounded scope and independent passed scope approval');
  const inputCard = await loadBound(root, inputsBinding);
  if (!same(inputCard.data, inputs) || inputs.root !== root) fail('Actual input-card DATA and root must match their exact binding');
  const admission = await loadBound(root, inputs.admission);
  const base = await loadBound(root, inputs.base_manifest);
  const prior = await loadBound(root, inputs.base_verification);
  const oldReview = await loadBound(root, inputs.base_source_review);
  const currentReview = await loadBound(root, inputs.current_source_review);
  const external = await loadBound(root, inputs.external_evidence);
  const boundedScope = await loadBound(root, inputs.bounded_scope);
  const scopeReview = await loadBound(root, inputs.scope_review);
  if (base.data.kind !== MANIFEST_KIND || prior.data.kind !== VERIFICATION_KIND || prior.data.status !== 'failed'
      || prior.data.passed !== false || !prior.data.failures?.length
      || !sameCurrentIdentity(root, prior.data.alias_manifest, base.binding)) fail('An exact completed failed base pair is required');
  if (!sameCurrentIdentity(root, base.data.admission, admission.binding) || !sameCurrentIdentity(root, prior.data.admission, admission.binding)
      || admission.data.scope?.physical_root !== '/mityane' || admission.data.scope?.historical_file_hashing_and_alias_verification !== true) fail('The original admitted alias scope must match exactly');
  if (oldReview.binding.path !== `${admission.data.artifact_root}/source-review-004.json` || oldReview.data.passed !== true
      || oldReview.data.kind !== 'cpb_v14_closeout_independent_source_review_v1'
      || !oldReview.data.files?.some(file => file.path === ALIASES_PROTOCOL && sameCurrentIdentity(root, file, base.data.protocol_identity))
      || !sameCurrentIdentity(root, base.data.protocol_identity, prior.data.protocol_identity)) fail('Original source-review-004 must bind the actual bulk verifier');
  if (currentReview.data.kind !== 'cpb_v14_closeout_independent_source_review_v1' || currentReview.data.passed !== true) fail('The repaired protocol needs an independently passed source review');
  const scopeSource = PROTOCOL.replace(/repair\.mjs$/, 'scope.mjs');
  const requiredSources = [PROTOCOL, PROTOCOL.replace(/\.mjs$/, '.test.mjs'), ALIASES_PROTOCOL,
    'code/shared/artifacts/node/v1/files.mjs', scopeSource, scopeSource.replace(/\.mjs$/, '.test.mjs')];
  for (const sourcePath of requiredSources) {
    const expected = currentReview.data.files?.find(file => file.path === sourcePath);
    if (!expected) fail(`Missing current reviewed repair dependency: ${sourcePath}`);
    await loadBoundSource(root, expected);
  }
  validateScopeDeclaration(boundedScope.data, prior.data.failures);
  if (scopeReview.data.kind !== 'cpb_v14_bounded_scope_review_v1' || scopeReview.data.status !== 'passed' || scopeReview.data.passed !== true
      || !sameCurrentIdentity(root, scopeReview.data.scope, boundedScope.binding)
      || !sameCurrentIdentity(root, scopeReview.data.source_review, currentReview.binding)
      || !sameCurrentIdentity(root, boundedScope.data.source_review, currentReview.binding)
      || !sameCurrentIdentity(root, boundedScope.data.admission, admission.binding)
      || !sameCurrentIdentity(root, boundedScope.data.base_pair?.manifest, base.binding)
      || !sameCurrentIdentity(root, boundedScope.data.base_pair?.verification, prior.binding)
      || scopeReview.data.required_pin_count !== boundedScope.data.required_pins.length) fail('Actual independently reviewed typed scope must bind this admission, source review, base pair and exact required population');
  if (!same(base.data.coverage, prior.data.coverage) || prior.data.aliases_verified !== base.data.aliases.length) fail('Original alias coverage/count must agree exactly');
  for (const [key, expected] of Object.entries({ scope_cleanup_whole_before: 14, prior_six_live_files: 6, intermediate_guides: 4,
    final_cleaned_guides: 14, paused_pointer_before: 2, paused_pointer_after: 2, source_note_aliases: 4,
    external_portable_checkpoints: 57, complete_critical_checkpoints: 124, canonical_v14_checkpoints: 5,
    paused_finalization_payload: 114, frozen_scientific_sources: 145, main_binaries: 7, complete_main_mirror: 1149 })) {
    if (base.data.coverage[key] !== expected) fail(`Original required coverage is incomplete: ${key}`);
  }
  if (external.data.kind !== 'cpb_v14_explicit_external_evidence_preservation_v1' || external.data.status !== 'passed' || external.data.passed !== true
      || external.data.source_aliases !== 47 || external.data.copied_files !== 20 || external.data.entries?.length !== 67) fail('The explicitly preserved 47+20 external identities are required');
  const inventory = await loadBound(root, external.data.source_inventory);
  if (external.data.entries.filter(row => row.copied === false).length !== 47 || external.data.entries.filter(row => row.copied === true).length !== 20) fail('External actual copy/source counts differ from the admitted 47+20');
  const seen = new Set();
  for (const row of external.data.entries) {
    const original = full(row.original);
    const key = identityKey(original);
    if (seen.has(key) || !sameCurrentIdentity(root, row.provenance.manifest, inventory.binding)) fail('External evidence must use unique inventory-bound full tuples');
    seen.add(key);
    if (row.copied === false) {
      if (!original.path.startsWith('code/encoders/causal_patch_bottleneck_mae/versions/v3/') || row.current_path !== `${ARCHIVE}/${original.path}`
          || !inventory.data.pins.some(pin => pin.repository === REPO && pin.path === original.path && bytesEqual(pin, original))) fail('External V3 source alias needs the exact recorded dual-repository counterpart');
    } else if (row.copied !== true || row.current_path !== `${admission.data.artifact_root}/portable-external-evidence/${original.path}`) fail('Portable external evidence must use its exact admitted destination');
  }
  const extraDocuments = [];
  for (const row of inputs.mapping_manifests ?? []) {
    const value = await loadBound(root, row.pin);
    const original = full({ ...value.binding, path: row.original_path, repository: row.original_repository });
    if (original.repository !== REPO || !historicalCandidates(original).includes(value.binding.path)) fail('Mapping input requires an exact explicit historical embedding repository/path context');
    if (![...boundedScope.data.control_documents, ...boundedScope.data.mapping_documents].some(doc => same(full(doc.original), original) && doc.current_path === value.binding.path)) fail('Extra mapping inputs must also be explicit independently reviewed scope controls');
    extraDocuments.push({ original, current_path: value.binding.path, binding: value.binding });
  }
  const result = await repairAliasGraph({ root, baseManifest: base.data,
    baseVerificationBinding: { manifest: base.binding, verification: prior.binding }, originalFailures: prior.data.failures,
    external: external.data, externalBinding: external.binding, inventory: inventory.data, extraDocuments,
    requiredScope: boundedScope.data, requiredScopeBinding: boundedScope.binding });
  const protocolIdentity = { path: PROTOCOL, ...await hashFile(path.join(root, PROTOCOL)) };
  const sharedParser = currentReview.data.files.find(file => file.path === 'code/shared/artifacts/node/v1/files.mjs');
  const common = { created_utc: new Date().toISOString(), ...repairObservationFields({ baseManifest: base.data, baseVerification: prior.data,
    basePair: { manifest: base.binding, verification: prior.binding }, sourceReview: oldReview.binding,
    protocolIdentity, sharedParser }), admission: admission.binding,
    repair_inputs: inputsBinding, current_source_review: currentReview.binding, external_evidence: external.binding,
    bounded_scope: boundedScope.binding, scope_review: scopeReview.binding,
  };
  const manifest = { ...base.data, ...common, kind: MANIFEST_KIND, aliases: result.aliases,
    verification_scope: result.verification_scope, bounded_scope_verified: result.bounded_scope_verified,
    whole_reference_graph_verified: false, all_historical_refs_resolved: false,
    coverage: { ...base.data.coverage, repair_metadata_documents: result.metadata_documents_rewalked, repair_identities: result.transitive_identities,
      explicit_external_source_aliases: 47, explicit_external_evidence_copies: 20 },
    repair_policy: 'Retain original observed byte identities. Interpret only explicitly reviewed typed controls; verify declared required whole-file identities. Inventory-member JSON bodies are terminals. The historical recursive reference graph is not verified.' };
  const verification = { ...common, kind: VERIFICATION_KIND, status: result.failures.length ? 'failed' : 'passed', passed: result.failures.length === 0,
    coverage: manifest.coverage, aliases_verified: result.aliases.length, ...Object.fromEntries(Object.entries(result).filter(([key]) => key !== 'aliases')),
    tensor_decodes: 0, model_calls: 0, head_fits: 0, training_updates: 0 };
  const inputSeals = [inputCard.binding, admission.binding, base.binding, prior.binding, oldReview.binding,
    currentReview.binding, external.binding, inventory.binding, boundedScope.binding, scopeReview.binding,
    ...extraDocuments.map(row => row.binding), ...[...boundedScope.data.control_documents, ...boundedScope.data.mapping_documents].map(doc => ({ path: doc.current_path, bytes: doc.observed.bytes, sha256: doc.observed.sha256 }))];
  const sourceSeals = currentReview.data.files.filter(file => requiredSources.includes(file.path));
  const recheck = async () => {
    for (const pin of [...inputSeals, ...sourceSeals]) await loadBoundSource(root, pin);
  };
  await recheck();
  verification.input_seals = inputSeals;
  verification.source_seals = sourceSeals;
  verification.final_input_and_source_seals_passed = true;
  return { manifest, verification, recheck };
}
async function loadBoundSource(root, pin) {
  const target = await currentPath(root, pin.path);
  const actual = await hashFile(target.absolute);
  if (!bytesEqual(actual, pin)) fail(`Reviewed repair source changed: ${pin.path}`);
}

async function cli() {
  if (process.platform !== 'linux' || await realpath('/mityane') !== '/mityane') fail('The existing managed Linux /mityane container is required');
  await lstat('/.dockerenv');
  const flags = {};
  for (let i = 2; i < process.argv.length; i += 2) {
    const flag = process.argv[i];
    if (!['--inputs', '--output', '--verification'].includes(flag) || !process.argv[i + 1] || flags[flag]) fail('Usage: repair.mjs --inputs BOUND-CARD --output NEW-MANIFEST --verification NEW-RECEIPT');
    flags[flag] = process.argv[i + 1];
  }
  if (Object.keys(flags).length !== 3) fail('Three explicit repair paths are required');
  const root = '/mityane';
  const outputs = [flags['--output'], flags['--verification']].map(value => path.resolve(value));
  if (outputs[0] === outputs[1]) fail('Exclusive output paths must differ');
  for (const output of outputs) {
    if (!/^\/mityane\/\.temp\/embeddings\/closeout\/cpb-v14\/attempt-\d{3}\/[^/]+\.json$/.test(output)) fail('Fresh admitted numbered attempt JSON outputs are required');
    if (await realpath(path.dirname(output)) !== path.dirname(output)) fail('Redirected output parent');
    try { await lstat(output); fail(`Refusing overwrite: ${output}`); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  if (path.dirname(outputs[0]) !== path.dirname(outputs[1])) fail('Both repair outputs must belong to the same admitted attempt');
  const inputsPath = localLocator(root, path.resolve(flags['--inputs']));
  const inputBytes = readAdmitted(admitFiles([path.join(root, inputsPath)], { root })[0]);
  const inputsBinding = { path: inputsPath, bytes: inputBytes.length, sha256: digest(inputBytes) };
  const inputs = parseJSON(inputBytes, { duplicates: 'reject', integers: 'bigint' });
  if (inputs.root !== root || path.dirname(path.join(root, localLocator(root, inputs.admission.path))) !== path.dirname(outputs[0])) fail('Repair outputs must use the original admitted root and attempt');
  const { manifest, verification, recheck } = await repairAliasInventory({ root, inputs, inputsBinding });
  await recheck();
  await writeFile(outputs[0], jsonBytes(manifest), { flag: 'wx' });
  verification.alias_manifest = { path: outputs[0], ...await hashFile(outputs[0]) };
  await recheck();
  await writeFile(outputs[1], jsonBytes(verification), { flag: 'wx' });
  process.stdout.write(JSON.stringify({ status: verification.status, aliases: manifest.aliases.length, failures: verification.failures.length, reused_aliases: verification.reused_aliases }) + '\n');
  if (!verification.passed) process.exitCode = 1;
}
if (process.argv[1] && path.resolve(process.argv[1]) === SELF) cli().catch(error => { console.error(error.stack); process.exitCode = 1; });
