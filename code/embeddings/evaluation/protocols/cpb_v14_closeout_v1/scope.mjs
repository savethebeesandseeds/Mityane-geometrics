// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Declares byte-identity obligations. Does not inspect tensors or execute history.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
import {admitFiles, canonicalDirectory, jsonBytes, parseJSON, readAdmitted,
  requireCondition as require, safeRelative, sha256, writeNew} from '../../../../shared/artifacts/node/v1/files.mjs';

const PROTOCOL = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const SELF = `${PROTOCOL}/scope.mjs`, SHARED = 'code/shared/artifacts/node/v1/files.mjs';
const ARCHIVE = '.archive/embeddings/pre-cleanup-2026-10-18';
const PAYLOAD = 'code/embeddings/output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001/payload';
const EMB = 'cuwacunu_embedding', CURRENT = 'Mityane-geometrics';
const BASELINE = `${ARCHIVE}/archive/cuwacunu_dev/2026-10-15-before-causal-primary/artifact-baseline.json`;
const quote = value => String(value).replaceAll('~', '~0').replaceAll('/', '~1');
const byteEqual = (a, b) => a?.bytes === b?.bytes && a?.sha256 === b?.sha256;
const pinFields = pin => ({path: pin.path, bytes: pin.bytes, sha256: pin.sha256});

export function fullIdentity(pin, repository = EMB) {
  require(pin && typeof pin.path === 'string' && pin.path && Number.isSafeInteger(pin.bytes) && pin.bytes >= 0
    && /^[0-9a-f]{64}$/.test(pin.sha256 ?? ''), 'full exact file identity required');
  repository = pin.repository ?? repository;
  require([EMB, CURRENT, 'cuwacunu_dev'].includes(repository), 'explicit known repository context required');
  return {repository, ...pinFields(pin)};
}
export const identityKey = pin => { const p = fullIdentity(pin); return JSON.stringify([p.repository, p.path, p.bytes, p.sha256]); };

// This traversal is allowed ONLY within the two explicitly named proposal DATA
// subtrees. It never follows a pin into another file or accepts aggregate sizes.
export function proposalPins(value, pointer = '') {
  const found = [];
  function visit(node, p) {
    if (!node || typeof node !== 'object') return;
    if (!Array.isArray(node) && typeof node.path === 'string' && Number.isSafeInteger(node.bytes)
      && /^[0-9a-f]{64}$/.test(node.sha256 ?? '')) {
      found.push({pin: fullIdentity(node, node.original_repository ?? EMB), pointer: p});
    }
    for (const [key, child] of Object.entries(node)) visit(child, `${p}/${quote(key)}`);
  }
  visit(value, pointer); return found;
}

// Exact selector transcription from the frozen V13 REPAIR-001 verifier. JSON
// inventory members are terminal byte objects, even when their names end .json.
export function inventoryMembers(descriptor, data) {
  require(['mapped_files', 'copied_files', 'files'].includes(descriptor.inventory_field)
    && ['host', 'destination'].includes(descriptor.entry_pin_field), 'declared V13 inventory selector required');
  require(descriptor.prefix === undefined || descriptor.prefix === '', 'inspected V13 inventories have no nonempty prefix');
  const rows = data[descriptor.inventory_field];
  require(Array.isArray(rows) && rows.length > 0, 'nonempty declared inventory member array required');
  return rows.map((row, index) => ({pin: fullIdentity(row[descriptor.entry_pin_field]),
    pointer: `/${descriptor.inventory_field}/${index}/${descriptor.entry_pin_field}`}));
}

export function classifyFailures(failures, required, baseline, baselinePath = BASELINE) {
  const requiredKeys = new Set(required.map(row => identityKey(row.original ?? row)));
  return failures.map((failure, index) => {
    const original = fullIdentity(failure.identity);
    let qualified = original, reason = 'No independently declared control role follows this historical member body.';
    if (failure.manifest?.path === baselinePath) {
      const match = /^\/entries\/(\d+)$/.exec(failure.pointer);
      if (match) {
        const entry = baseline.entries[Number(match[1])];
        require(entry && byteEqual(entry, original) && entry.path === original.path, 'baseline entry failure must match exact declared record');
        require([EMB, 'cuwacunu_dev'].includes(entry.repo), 'baseline explicit repo context required');
        qualified = fullIdentity(pinFields(entry), entry.repo);
        reason = 'artifact-baseline.entries repository context is the explicit repo field.';
      } else if (failure.pointer === '/passed_after_audit') {
        qualified = null;
        reason = 'Legacy passed_after_audit.bytes is the aggregate audited artifact size, not the observation report file byte length.';
      }
    }
    const inScope = qualified !== null && requiredKeys.has(identityKey(qualified));
    const overinterpreted = qualified === null || identityKey(qualified) !== identityKey(original);
    return {original_failure_index: index, original_identity: failure.identity,
      record_qualified_identity: qualified, required_in_scope: inScope,
      disposition: overinterpreted ? 'record_schema_overinterpretation' : inScope ? 'required_current_scope' : 'outside_declared_traversal',
      reason, prior_failure_retained: true};
  });
}

export function requireCount(label, rows, count) {
  require(Array.isArray(rows) && rows.length === count, `${label}: exactly ${count} members required`); return rows;
}

export function validatePreservationRoots(before, final, v13Original) {
  require(before?.kind === 'cpb_v14_before_execution_file_preservation_v1' && before.status === 'passed'
    && final?.kind === 'causal_fresh_v13_final_preservation_v1' && final.status === 'passed', 'distinct actual V14 before and V13 final preservation schemas required');
  require(same(fullIdentity(before.baseline_preservation), fullIdentity(v13Original)), 'V14 before control binds the declared V13 final full tuple');
  requireCount('V14_before_critical', before.critical_checkpoint_pins, 119);
  require(same(before.critical_checkpoint_pins, final.critical_checkpoint_pins), 'V14 before and V13 final checkpoint inventories exactly equal');
}

/** Select metadata roots and inventory members, not an arbitrary reference graph. */
export function buildScope({root, attempt, admissionPath, baseManifestPath, baseVerificationPath, sourceReviewPath, clarificationPath}) {
  root = path.resolve(root); attempt = path.resolve(attempt);
  canonicalDirectory(root); canonicalDirectory(attempt, {root});
  const relative = filename => safeRelative(path.relative(root, path.resolve(filename)).split(path.sep).join('/'));
  const required = new Map(), controls = new Map(), documents = new Map();
  let baseIndex = new Map();
  function currentRead(locator, expected) {
    safeRelative(locator);
    if (documents.has(locator)) {
      const old = documents.get(locator); require(!expected || byteEqual(old.binding, expected), `changed cached control binding: ${locator}`); return old;
    }
    const bytes = readAdmitted(admitFiles([path.join(root, locator)], {root})[0]);
    const binding = {repository: CURRENT, path: locator, bytes: bytes.length, sha256: sha256(bytes)};
    require(!expected || byteEqual(binding, expected), `control identity mismatch: ${locator}`);
    const data = parseJSON(bytes, {duplicates: 'last', integers: 'bigint'});
    const loaded = {data, binding}; documents.set(locator, loaded); return loaded;
  }
  function knownLocators(pin, explicit = []) {
    const row = baseIndex.get(identityKey(pin));
    return [...new Set([...explicit, ...(row ? [row.current_path] : [])])].map(safeRelative);
  }
  function add(pin, manifest, pointer, selector, locators = []) {
    const original = fullIdentity(pin), key = identityKey(original);
    const provenance = {manifest: fullIdentity(manifest, CURRENT), pointer, ...(selector ? {selector} : {})};
    const old = required.get(key) ?? {original, required_by: [], known_locators: []};
    if (!old.required_by.some(row => same(row, provenance))) old.required_by.push(provenance);
    old.known_locators = [...new Set([...old.known_locators, ...knownLocators(original, locators)])];
    required.set(key, old); return original;
  }
  function register(loaded, role, original = loaded.binding, mapping = false) {
    const key = identityKey(loaded.binding);
    const record = controls.get(key) ?? {role, original: fullIdentity(original), current_path: loaded.binding.path,
      observed: {bytes: loaded.binding.bytes, sha256: loaded.binding.sha256}, selectors: [], mapping: false};
    record.mapping ||= mapping; controls.set(key, record);
    add(original, loaded.binding, '', 'explicit whole control document', [loaded.binding.path]);
    return {...loaded, record};
  }
  function select(doc, pin, pointer, selector, locators = []) {
    if (!doc.record.selectors.some(row => row.pointer === pointer && row.selector === selector)) doc.record.selectors.push({pointer, selector});
    return add(pin, doc.binding, pointer, selector, locators);
  }
  function loadPin(pin, role, locators = [], mapping = true) {
    pin = fullIdentity(pin);
    const candidates = knownLocators(pin, locators);
    require(candidates.length > 0, `explicit/base-observed control locator required: ${identityKey(pin)}`);
    let loaded;
    for (const locator of candidates) {
      try { loaded = currentRead(locator, pin); break; } catch (error) { if (locator === candidates.at(-1)) throw error; }
    }
    return register(loaded, role, pin, mapping);
  }
  const admission = register(currentRead(relative(admissionPath)), 'original_admission');
  const base = register(currentRead(relative(baseManifestPath)), 'original_failed_alias_manifest');
  const verification = register(currentRead(relative(baseVerificationPath)), 'original_failed_alias_verification');
  const sourceReview = register(currentRead(relative(sourceReviewPath)), 'actual_source_review');
  const clarification = register(currentRead(relative(clarificationPath)), 'scope_clarification');
  require(admission.data.kind === 'cpb_v14_postmigration_closeout_admission_v1' && admission.data.artifact_root === relative(attempt), 'original closeout admission required');
  require(base.data.kind === 'cpb_v14_historical_identity_alias_manifest_v1' && verification.data.kind === 'cpb_v14_historical_identity_alias_verification_v1'
    && verification.data.passed === false && verification.data.status === 'failed', 'complete original failed base pair required');
  requireCount('original_failed_references', verification.data.failures, 11302);
  require(base.data.aliases.length === 18660 && verification.data.aliases_verified === 18660 && same(base.data.coverage, verification.data.coverage), 'complete original observations required');
  const actualBase = pinFields(base.binding), recordedBase = verification.data.alias_manifest;
  require(recordedBase && relative(path.resolve(root, recordedBase.path)) === actualBase.path && byteEqual(recordedBase, actualBase), 'failed verification binds failed manifest');
  require(clarification.data.kind === 'cpb_v14_closeout_scope_clarification_v1'
    && same(clarification.data.original_admission, pinFields(admission.binding))
    && same(clarification.data.failed_broad_manifest, actualBase)
    && same(clarification.data.failed_broad_verification, pinFields(verification.binding)), 'clarification binds original admission and failed pair');
  require(clarification.data.prior_clarification, 'additive clarification generation must bind preserved initial declaration');
  select(clarification, {...clarification.data.prior_clarification, repository: CURRENT}, '/prior_clarification', 'preserved prior scope clarification', [clarification.data.prior_clarification.path]);
  register(currentRead(clarification.data.prior_clarification.path, clarification.data.prior_clarification), 'preserved_initial_scope_clarification');
  require(sourceReview.data.passed === true && Array.isArray(sourceReview.data.files), 'passed actual source review required');
  const sourceClosure = sourceReview.data.files.map(pin => {
    const rel = safeRelative(pin.path), bytes = readAdmitted(admitFiles([path.join(root, rel)], {root})[0]);
    require(byteEqual({bytes: bytes.length, sha256: sha256(bytes)}, pin), `reviewed source changed: ${rel}`);
    select(sourceReview, {...pin, repository: CURRENT}, `/files/${sourceReview.data.files.indexOf(pin)}`, 'actual reviewed source generation', [rel]);
    return pinFields(pin);
  });
  require([SELF, `${PROTOCOL}/scope.test.mjs`, SHARED].every(name => sourceClosure.some(pin => pin.path === name)), 'scope implementation, fixtures and exact JSON parser reviewed');
  for (const row of base.data.aliases) {
    require(byteEqual(row.original, row.observed), 'base alias observations must retain exact bytes');
    const key = identityKey(row.original), prior = baseIndex.get(key);
    require(!prior || prior.current_path === row.current_path, 'conflicting original alias observation');
    baseIndex.set(key, row);
  }
  // The binding snapshot admits these exact control roots; it does not readmit
  // current documentary targets after their already recorded changes.
  const snapshotPath = 'doc/EMBEDDINGS_V14_CLOSEOUT_INPUTS_2026-10-18.json';
  require(clarification.data.documentary_control_snapshot?.path === snapshotPath, 'explicit documentary control snapshot root');
  const snapshot = register(currentRead(snapshotPath, clarification.data.documentary_control_snapshot), 'documentary_control_snapshot', undefined, true);
  const snapshotPins = new Map(snapshot.data.existing_files.map(pin => [pin.path, pin]));
  function snapshotControl(originalPath, current = `code/embeddings/${originalPath}`) {
    const pin = snapshotPins.get(current); require(pin, `snapshot control pin missing: ${current}`);
    select(snapshot, {...pin, repository: CURRENT}, `/existing_files/${snapshot.data.existing_files.indexOf(pin)}`, 'explicit generation/migration control', [current]);
    return register(currentRead(current, pin), 'snapshot_bound_control', {...pin, path: originalPath, repository: EMB}, true);
  }
  function observedControl(oldPath, role) {
    const current = oldPath.startsWith('output/') ? `code/embeddings/${oldPath}` : `${ARCHIVE}/${oldPath}`;
    const rows = base.data.aliases.filter(row => row.original.repository === EMB && row.original.path === oldPath && row.current_path === current);
    require(rows.length === 1, `unique base-observed explicit control root required: ${oldPath}`);
    return loadPin(rows[0].original, role, [current]);
  }
  const migration = name => snapshotControl(`doc/causal_fresh_v14_migration/${name}.json`, `${ARCHIVE}/doc/causal_fresh_v14_migration/${name}.json`);
  const finalGeneration = snapshotControl('output/engineering/causal-fresh-v14-concurrent-documentary-change/final-cleanup-generation-001/manifest.json');
  const before = observedControl('output/engineering/embedding-scope-cleanup-2026-10-18/before.json', 'scope_cleanup_before');
  requireCount('scope_cleanup_whole_before', before.data, 14); requireCount('final_cleaned_guides', finalGeneration.data.files, 14);
  function snapshotMember(doc, file, pointer, field = 'snapshot') {
    require(typeof file[field] === 'string', 'exact whole snapshot locator required');
    const current = `code/embeddings/${safeRelative(file[field])}`;
    select(doc, file, pointer, 'whole preserved generation; terminal bytes', [current]);
    select(doc, {...pinFields(file), path: file[field]}, `${pointer}/${field}`, 'whole snapshot identity; terminal bytes', [current]);
  }
  for (const [i, old] of before.data.entries()) {
    const matches = finalGeneration.data.files.filter(file => file.prior_whole?.path === old.path && file.prior_whole.sha256 === old.sha256);
    require(matches.length === 1, 'unique typed prior whole-generation length required');
    const file = matches[0].prior_whole, p = `/files/${finalGeneration.data.files.indexOf(matches[0])}/prior_whole`;
    snapshotMember(finalGeneration, file, p);
    select(before, file, `/${i}`, 'length from exact final-generation prior_whole', [`code/embeddings/${file.snapshot}`]);
  }
  const guideGroups = [
    ['prior_six_live_files', 'output/engineering/causal-fresh-v14-finalization/prior-live-001/manifest.json', 6],
    ['intermediate_guides', 'output/engineering/causal-fresh-v14-concurrent-documentary-change/attempt-001/manifest.json', 4],
    ['paused_pointer_before', 'output/engineering/causal-fresh-v14-migration/attempt-001/guide-before-001/manifest.json', 2],
  ];
  for (const [label, original, count] of guideGroups) {
    const doc = snapshotControl(original); requireCount(label, doc.data.files, count);
    doc.data.files.forEach((file, i) => snapshotMember(doc, file, `/files/${i}`));
  }
  finalGeneration.data.files.forEach((file, i) => snapshotMember(finalGeneration, file, `/files/${i}`));
  const pointer = migration('pointer-update'); requireCount('paused_pointer_after', pointer.data.files, 2);
  pointer.data.files.forEach((pin, i) => select(pointer, pin, `/files/${i}`, 'postmigration historical pointer generation'));
  const notes = observedControl('doc/causal_fresh_v14/front-source-notes-001.json', 'V14_front_note_alias_control');
  requireCount('source_note_aliases', notes.data.files, 4);
  notes.data.files.forEach((pin, i) => {
    select(notes, pin, `/files/${i}`, 'new whole front-note generation');
    const locator = `code/embeddings/${safeRelative(pin.whole_prior_copy)}`;
    const rows = base.data.aliases.filter(row => row.original.repository === EMB && row.original.path === pin.path && row.current_path === locator);
    require(rows.length === 1, 'exact prior note generation observed in bound base');
    select(notes, rows[0].original, `/files/${i}/whole_prior_copy`, 'base-observed prior whole copy; suffix checked by original alias generation', [locator]);
  });
  const availability = migration('checkpoint-availability');
  select(availability, availability.data.historical_inventory, '/historical_inventory', 'actual migration historical checkpoint inventory is the V14 before-execution preservation control');
  const v14before = loadPin(availability.data.historical_inventory, 'V14_before_execution_preservation');
  const declaredV13 = clarification.data.typed_v13_final_preservation;
  require(declaredV13?.path === `${ARCHIVE}/doc/causal_fresh_v13/final-preservation.json`, 'explicit admitted V13 final location');
  const v13Original = {...declaredV13, path: 'doc/causal_fresh_v13/final-preservation.json', repository: EMB};
  const v13final = register(currentRead(declaredV13.path, declaredV13), 'V13_final_preservation', v13Original, true);
  validatePreservationRoots(v14before.data, v13final.data, v13Original);
  select(v14before, v14before.data.baseline_preservation, '/baseline_preservation', 'exact explicit link from V14 before-execution inventory to V13 final preservation', [declaredV13.path]);
  const selectedV3 = observedControl('doc/causal_v3_v4_train_diagnosis/preservation-before.json', 'selected_V3_source_identity_control');
  require(v13final.data.selected_V3_source_files_each_repo === 47 && selectedV3.data.counts.v3_owned_source_pairs === 47
    && selectedV3.data.counts.v3_source_bindings === 94, 'frozen selected V3 paired-source contract');
  const v3SourceRows = selectedV3.data.pins.map((pin, index) => ({pin, index}))
    .filter(row => Array.isArray(row.pin.roles) && row.pin.roles.includes('selected_clean_v3_source47_each'));
  requireCount('selected_v3_source_bindings', v3SourceRows, 94);
  for (const repository of [EMB, 'cuwacunu_dev']) require(v3SourceRows.filter(row => row.pin.repository === repository).length === 47, '47 exact selected V3 source identities in each original repository');
  require(new Set(v3SourceRows.map(row => identityKey(fullIdentity(row.pin)))).size === 94, 'selected V3 source tuples unique');
  v3SourceRows.forEach(({pin, index}) => select(selectedV3, pin, `/pins/${index}`, 'selected_clean_v3_source47_each role only; old baseline513 graph remains terminal'));
  const v13CardCurrent = clarification.data.typed_v13_preservation_card;
  const v13 = register(currentRead(v13CardCurrent.path, v13CardCurrent), 'V13_REPAIR_001_typed_preservation_card',
    {...v13CardCurrent, path: 'doc/causal_fresh_v13/preservation-card.REPAIR-001.json', repository: EMB}, true);
  require(v13.data.kind === 'causal_fresh_v13_final_preservation_card_v1', 'actual repaired preservation card kind');
  requireCount('prior_v13_protected_pins', v13.data.protected_pins, 8506);
  require(same(v13.data.protected_pins, v13final.data.additional_protected_pins), 'repaired V13 protected pins exactly equal final preservation pins');
  requireCount('prior_v13_protected_inventories', v13.data.protected_inventories, 30);
  requireCount('prior_v13_current_protected_inventories', v13.data.current_protected_inventories, 4);
  for (const descriptor of v13.data.current_protected_inventories) require(v13.data.protected_inventories.filter(row => same(row, descriptor)).length === 1, 'current protected selector occurs exactly once in combined selectors');
  requireCount('V13_final_inventory_controls', v13final.data.additional_protected_inventories, 30);
  v13.data.protected_pins.forEach((pin, i) => select(v13, pin, `/protected_pins/${i}`, 'frozen repaired preservation protected member; terminal bytes'));
  for (const [i, descriptor] of v13.data.protected_inventories.entries()) {
    const doc = loadPin(descriptor.receipt, 'V13_declared_protected_inventory');
    select(v13, descriptor.receipt, `/protected_inventories/${i}/receipt`, 'exact declared inventory control');
    const members = inventoryMembers(descriptor, doc.data), finalDescriptor = v13final.data.additional_protected_inventories[i];
    require(same(finalDescriptor.receipt, descriptor.receipt) && finalDescriptor.inventory_field === descriptor.inventory_field
      && finalDescriptor.members === members.length, 'V13 final inventory receipt/selector/member count equality');
    for (const member of members) select(doc, member.pin, member.pointer,
      `V13 REPAIR selector inventory_field=${descriptor.inventory_field};entry_pin_field=${descriptor.entry_pin_field};prefix=${descriptor.prefix ?? ''}; terminal member bytes`);
  }
  const singleton = loadPin(v13.data.intermediate_front_note_preservation, 'V13_front_note_singleton');
  require(singleton.data.prior_bytes_preserved_as_exact_suffix === true && byteEqual(singleton.data.prior, singleton.data.whole_prior_snapshot), 'singleton exact prior whole copy binding');
  select(singleton, singleton.data.prior, '/prior', 'whole prior singleton generation', [`code/embeddings/${singleton.data.whole_prior_snapshot.path}`]);
  select(singleton, singleton.data.whole_prior_snapshot, '/whole_prior_snapshot', 'whole snapshot byte identity', [`code/embeddings/${singleton.data.whole_prior_snapshot.path}`]);
  select(singleton, singleton.data.new, '/new', 'whole new singleton generation');
  // Exactly the documented V13->V6 control chain, never arbitrary child JSON.
  const finals = [v13final];
  for (let version = 12; version >= 6; version--) finals.push(loadPin(finals.at(-1).data[`prior_V${version}_final`], `V${version}_final_preservation_control`));
  for (const doc of finals) {
    for (const [i, file] of (doc.data.old_authorized_snapshots ?? []).entries()) snapshotMember(doc,
      {path: file.path, bytes: file.old_bytes, sha256: file.old_sha256, snapshot: file.snapshot}, `/old_authorized_snapshots/${i}`);
    for (const [i, file] of (doc.data.V5_metadata6_chain ?? []).entries()) snapshotMember(doc,
      {path: file.path, bytes: file.prior_bytes, sha256: file.prior_sha256, snapshot: file.prior_snapshot}, `/V5_metadata6_chain/${i}`);
    for (const [name, pin] of Object.entries(doc.data)) if (/^V\d+_prior_snapshot$/.test(name)) {
      select(doc, pin, `/${name}`, 'explicit earlier whole snapshot control');
      const snap = loadPin(pin, name);
      const field = snap.data.kind === 'causal_fresh_v5_live_metadata_before_finalization_v1' ? 'snapshot_path' : 'snapshot';
      require(Array.isArray(snap.data.files), 'typed prior snapshot files');
      snap.data.files.forEach((file, i) => snapshotMember(snap, file, `/files/${i}`, field));
    }
  }
  // Additional V13 repair controls are explicit role bindings. Their member
  // bodies are not recursively expanded merely because they contain old pins.
  for (const name of ['source_manifest', 'source_zip', 'complete_mirror', 'compiled_binary_mirror', 'main_card',
    'metadata_application', 'proposal_review', 'final_establishment', 'preexecution_draft_amendment', 'preexecution_draft_amendment_before_snapshot']) {
    const pin = v13.data[name]; require(pin, `V13 explicit control missing: ${name}`);
    select(v13, pin, `/${name}`, 'explicit repaired V13 preservation role');
    if (pin.path.endsWith('.json')) loadPin(pin, `V13_${name}`);
  }
  for (const [name, value] of Object.entries(v13.data.additive_repair)) {
    if (name === 'owned_sources') value.forEach((pin, i) => select(v13, pin, `/additive_repair/${name}/${i}`, 'declared V13 repair source'));
    else if (value && typeof value === 'object' && typeof value.path === 'string') {
      select(v13, value, `/additive_repair/${name}`, 'explicit historical repair control; terminal unless named here');
      if (value.path.endsWith('.json')) loadPin(value, `V13_repair_${name}`);
    }
  }
  const v13sources = loadPin(v13.data.source_manifest, 'V13_source_capsule_manifest');
  require(v13.data.main_source_files === 143, 'V13 143-source contract'); requireCount('prior_v13_source_files', v13sources.data.files, 143);
  v13sources.data.files.forEach((pin, i) => select(v13sources, pin, `/files/${i}`, 'V13 frozen capsule source member; terminal bytes'));
  const external = migration('portable-external-checkpoints'); requireCount('external_portable_checkpoints', external.data.whole_copy_aliases, 57);
  external.data.whole_copy_aliases.forEach((row, i) => {
    require(byteEqual(row.original_typed_record, row.whole_copy), 'portable whole copy exact bytes');
    for (const name of ['original_typed_record', 'whole_copy']) select(external, row[name], `/whole_copy_aliases/${i}/${name}`, 'explicit portable whole-file checkpoint', [`code/embeddings/${safeRelative(row.whole_copy.path)}`]);
  });
  requireCount('historical_critical_checkpoints', v14before.data.critical_checkpoint_pins, 119);
  requireCount('canonical_v14_checkpoints', availability.data.new_V14_verified_pins, 5);
  const critical = [...v14before.data.critical_checkpoint_pins, ...availability.data.new_V14_verified_pins];
  require(new Set(critical.map(pin => identityKey(fullIdentity(pin)))).size === 124 && critical.reduce((n, pin) => n + pin.bytes, 0) === 287027201, '124 unique critical checkpoint bindings and exact byte total');
  v14before.data.critical_checkpoint_pins.forEach((pin, i) => select(v14before, pin, `/critical_checkpoint_pins/${i}`, 'actual migration historical inventory critical checkpoint whole-file identity'));
  availability.data.new_V14_verified_pins.forEach((pin, i) => select(availability, pin, `/new_V14_verified_pins/${i}`, 'V14 critical checkpoint whole-file identity'));
  const paused = migration('paused-finalization-mirror'); requireCount('paused_finalization_payload', paused.data.mapped_files, 114);
  require(paused.data.mapped_files.reduce((n, row) => n + row.host.bytes, 0) === 21316710, 'paused payload exact byte sum');
  const relatives = new Set();
  paused.data.mapped_files.forEach((row, i) => {
    require(!relatives.has(row.relative) && row.host.path === `${paused.data.host_payload}/${row.relative}` && row.container_path === `${paused.data.container_root}/${row.relative}`, 'paused mirror explicit roots/member relation'); relatives.add(row.relative);
    const current = `code/embeddings/${safeRelative(row.host.path)}`;
    select(paused, row.host, `/mapped_files/${i}/host`, 'paused whole payload member; terminal bytes', [current]);
    select(paused, {...pinFields(row.host), path: row.container_path}, `/mapped_files/${i}/container_path`, 'explicit paused container/host identity pair', [current]);
  });
  const card = register(currentRead(`${PAYLOAD}/card-source-001/finalization-card.json`, paused.data.card), 'original_finalization_card', paused.data.card, true);
  requireCount('card_inputs', card.data.inputs, 49);
  const roles = new Map();
  card.data.inputs.forEach((pin, i) => {
    require(!roles.has(pin.id), 'unique finalization card roles');
    const locator = `${PAYLOAD}/card-source-001/${safeRelative(pin.staged_path)}`;
    select(card, pin, `/inputs/${i}`, 'original independent reviewer input role; member body terminal unless independently typed', [locator]);
    if (pin.format === 'json') roles.set(pin.id, loadPin(pin, `reviewer_${pin.id}`, [locator]));
    else roles.set(pin.id, {pin});
  });
  const scientific = roles.get('experiment_card'); requireCount('current_v14_external_input_pins', scientific.data.input_pins, 20);
  scientific.data.input_pins.forEach((pin, i) => { require(pin.repository === 'cuwacunu_dev', 'actual V14 external input repository'); select(scientific, pin, `/input_pins/${i}`, 'actual V14 scientific external input pin; whole bytes'); });
  const sources = roles.get('source_manifest'); requireCount('frozen_scientific_sources', sources.data.files, 145);
  require(new Set(sources.data.files.map(pin => pin.path)).size === 145, 'unique V14 capsule source paths');
  sources.data.files.forEach((pin, i) => {
    select(sources, pin, `/files/${i}`, 'frozen V14 capsule relative source identity; terminal bytes');
    select(sources, {...pinFields(pin), path: `${card.data.container_attempt}/source/${pin.path}`}, `/files/${i}`, 'same capsule member qualified by frozen card container_attempt/source');
  });
  const build = roles.get('build'); requireCount('main_binaries', build.data.binaries, 7);
  require(build.data.dependency_closures && !Array.isArray(build.data.dependency_closures) && Object.keys(build.data.dependency_closures).length === 11, 'exactly eleven named dependency closures');
  build.data.binaries.forEach((pin, i) => select(build, pin, `/binaries/${i}`, 'seven frozen binary identities; no execution'));
  Object.entries(build.data.dependency_closures).forEach(([name, row]) => select(build, row.file, `/dependency_closures/${quote(name)}/file`, 'eleven frozen translation-unit dependency file identities'));
  select(build, build.data.source_manifest, '/source_manifest', 'build source manifest identity');
  requireCount('explicit_admitted_mapping_roots', clarification.data.mapping_roots, 1);
  const mirrorRoot = clarification.data.mapping_roots[0];
  require(mirrorRoot.original_repository === EMB && mirrorRoot.original_path === 'doc/causal_fresh_v14/build-binaries-mirror.json'
    && mirrorRoot.pin.path === `${ARCHIVE}/${mirrorRoot.original_path}` && mirrorRoot.expected_members === 18, 'exact declared V14 build mirror mapping root');
  select(clarification, {...mirrorRoot.pin, repository: CURRENT}, '/mapping_roots/0/pin', 'explicit supplemental mapping root', [mirrorRoot.pin.path]);
  const buildMirror = register(currentRead(mirrorRoot.pin.path, mirrorRoot.pin), 'V14_explicit_binary_dependency_mapping',
    {...mirrorRoot.pin, path: mirrorRoot.original_path, repository: mirrorRoot.original_repository}, true);
  requireCount('build_binary_dependency_mapping', buildMirror.data.files, 18);
  buildMirror.data.files.forEach((pin, i) => {
    const current = `code/embeddings/${safeRelative(pin.path)}`;
    select(buildMirror, pin, `/files/${i}`, 'explicit binary/dependency mirror member', [current]);
    select(buildMirror, {...pinFields(pin), path: pin.container_path}, `/files/${i}/container_path`, 'explicit binary/dependency container identity', [current]);
  });
  const mirror = roles.get('mirror'); requireCount('complete_main_mirror', mirror.data.files, 1149);
  mirror.data.files.forEach((pin, i) => {
    const current = `code/embeddings/${safeRelative(pin.path)}`;
    select(mirror, pin, `/files/${i}`, 'complete V14 main mirror whole-file member; terminal bytes', [current]);
    if (pin.container_path) select(mirror, {...pinFields(pin), path: pin.container_path}, `/files/${i}/container_path`, 'explicit main container/host pair', [current]);
  });
  const proposalReceiptPath = `${PAYLOAD}/proposals-001/finalization-proposal.json`;
  const proposalPin = admission.data.documentary_inputs.find(pin => pin.path === proposalReceiptPath); require(proposalPin, 'original admission pins proposal receipt');
  const proposalReceipt = register(currentRead(proposalReceiptPath, proposalPin), 'original_proposal_receipt', {...proposalPin, repository: CURRENT}, true);
  requireCount('proposal_outputs', proposalReceipt.data.outputs, 3);
  for (const [i, pin] of proposalReceipt.data.outputs.entries()) {
    const current = `${PAYLOAD}/proposals-001/${safeRelative(pin.path)}`;
    select(proposalReceipt, pin, `/outputs/${i}`, 'immutable proposed output DATA, not metadata application', [current]);
    const doc = loadPin(pin, 'original_proposed_output', [current]);
    if (pin.path.endsWith('/instances/CPB-v14.market-01.json')) for (const row of proposalPins(doc.data)) select(doc, row.pin, row.pointer, 'exact historical pin objects in complete instance proposal; no file-body traversal');
    if (pin.path.endsWith('/versions/current.json')) for (const row of proposalPins(doc.data.fresh_research_candidate, '/fresh_research_candidate')) select(doc, row.pin, row.pointer, 'exact historical pin objects in fresh V14 catalog subtree; no file-body traversal');
  }
  // Classification reads this one declared historical schema, without admitting
  // its reused 80,105-entry legacy DEV inventory as a new raw-input obligation.
  require(clarification.data.opaque_classification_record?.path === BASELINE, 'exact bound classification schema record');
  const baseline = register(currentRead(BASELINE, clarification.data.opaque_classification_record), 'classification_only_legacy_artifact_baseline', undefined, false);
  require(baseline.data.kind === 'causal_primary_source_archive_artifact_baseline_v1' && baseline.data.current_artifacts_rehashed === false, 'exact reused historical baseline schema');
  const coverage = {
    scope_cleanup_whole_before: 14, final_cleaned_guides: 14, prior_six_live_files: 6, intermediate_guides: 4,
    paused_pointer_before: 2, paused_pointer_after: 2, source_note_aliases: 4, external_portable_checkpoints: 57,
    historical_critical_checkpoints: 119, canonical_v14_checkpoints: 5, complete_critical_checkpoints: 124,
    paused_finalization_payload: 114, card_inputs: 49, frozen_scientific_sources: 145, main_binaries: 7, complete_main_mirror: 1149,
    v13_singleton_registered_before_repair_traversal: true, paused_payload_exact_member_set: base.data.coverage.paused_payload_exact_member_set,
    earlier_preservation_generations_registered_before_traversal: finals.length,
    prior_v13_protected_pins: 8506, prior_v13_protected_inventories: 30, prior_v13_current_protected_inventories: 4,
    prior_v13_source_files: 143, main_dependency_files: 11, proposal_outputs: 3, proposal_receipts: 1, current_v14_external_input_pins: 20,
    selected_v3_source_files_each_repo: 47, selected_v3_source_bindings: 94,
  };
  for (const [key, value] of Object.entries(coverage)) if (Object.hasOwn(base.data.coverage, key)) require(same(base.data.coverage[key], value), `base observation coverage changed: ${key}`);
  const requiredPins = [...required.values()].sort((a, b) => identityKey(a.original).localeCompare(identityKey(b.original)));
  const controlDocuments = [...controls.values()].map(({mapping, ...row}) => row);
  const controlKeys = new Set(controlDocuments.map(row => identityKey({repository: CURRENT, path: row.current_path, ...row.observed})));
  require(requiredPins.every(row => row.required_by.length && row.required_by.every(provenance => controlKeys.has(identityKey(provenance.manifest)))), 'all member provenance derives from explicitly bound control documents');
  const classification = classifyFailures(verification.data.failures, requiredPins, baseline.data);
  const counts = {}; for (const row of classification) counts[row.disposition] = (counts[row.disposition] ?? 0) + 1;
  return {kind: 'cpb_v14_typed_bounded_scope_v1', status: 'declared_metadata_scope', created_utc: new Date().toISOString(),
    admission: pinFields(admission.binding), source_review: pinFields(sourceReview.binding), clarification: pinFields(clarification.binding),
    base_pair: {manifest: pinFields(base.binding), verification: pinFields(verification.binding)}, source_closure: sourceClosure, coverage,
    required_pin_count: requiredPins.length, required_pins: requiredPins, control_documents: controlDocuments,
    mapping_documents: [...controls.values()].filter(row => row.mapping).map(({mapping, ...row}) => row),
    terminal_policy: {kind: 'explicit_record_qualified_inventory_members_v1', inventory_members: 'Whole-byte identities only; JSON member bodies remain terminal unless separately named as an explicit control/reviewer/mapping role.',
      proposal_subtree_exceptions: ['complete instance proposal', 'catalog proposal /fresh_research_candidate'], arbitrary_child_reference_traversal: false,
      failed_base_observations_retained: true, prior_verified_observations_are_time_qualified: true, outside_scope_means_verified: false},
    historical_failure_classification: classification, historical_failure_classification_counts: counts,
    original_failed_reference_count: verification.data.failures.length, original_failed_result_unchanged: true,
    all_historical_refs_resolved: false, whole_reference_graph_verified: false,
    physical_member_set_evidence: {source: pinFields(verification.binding), field: '/coverage/paused_payload_exact_member_set', fresh_scan_claimed: false},
    tensor_decodes: 0, model_calls: 0, head_fits: 0, training_updates: 0, live_metadata_writes: 0};
}

function cli() {
  require(process.platform === 'linux' && fs.existsSync('/.dockerenv'), 'managed Linux container only');
  const args = process.argv.slice(2), options = {};
  require(args.shift() === 'build', 'scope.mjs build --root /mityane --attempt ... --admission ... --base-manifest ... --base-verification ... --source-review ... --clarification ... --output ...');
  const names = ['root', 'attempt', 'admission', 'base-manifest', 'base-verification', 'source-review', 'clarification', 'output'];
  while (args.length) { const flag = args.shift(); require(names.some(name => flag === `--${name}`) && args.length && !Object.hasOwn(options, flag), `invalid argument: ${flag}`); options[flag] = args.shift(); }
  for (const name of names) require(options[`--${name}`], `missing --${name}`);
  const root = path.resolve(options['--root']), attempt = path.resolve(options['--attempt']);
  require(root === '/mityane' && /^\/mityane\/\.temp\/embeddings\/closeout\/cpb-v14\/attempt-\d{3}$/.test(attempt), 'actual admitted managed attempt required');
  canonicalDirectory(root); canonicalDirectory(attempt, {root});
  const output = path.resolve(options['--output']);
  require(path.dirname(output) === attempt && path.basename(output) === 'bounded-scope.json' && !fs.existsSync(output), 'fresh bounded-scope.json leaf required');
  const scope = buildScope({root, attempt, admissionPath: options['--admission'], baseManifestPath: options['--base-manifest'], baseVerificationPath: options['--base-verification'],
    sourceReviewPath: options['--source-review'], clarificationPath: options['--clarification']});
  writeNew(output, jsonBytes(scope));
  console.log(JSON.stringify({status: scope.status, required_pins: scope.required_pin_count, original_failures_retained: scope.original_failed_reference_count}));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { cli(); } catch (error) { console.error(error.stack); process.exitCode = 1; }
}
