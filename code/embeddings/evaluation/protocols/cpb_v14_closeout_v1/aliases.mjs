// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Metadata and byte identity only. This program never executes preserved sources.
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { access, lstat, readFile, readdir, realpath, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseJSON } from '../../../../shared/artifacts/node/v1/files.mjs';

export const ARCHIVE = '.archive/embeddings/pre-cleanup-2026-10-18';
export const PAYLOAD = 'code/embeddings/output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001/payload';
const HISTORICAL_REPOSITORY = 'cuwacunu_embedding';
const shaPattern = /^[0-9a-f]{64}$/;
const slash = value => value.replaceAll('\\', '/');
const fail = message => { throw new Error(message); };

export function identity(pin, repository = HISTORICAL_REPOSITORY) {
  if (!pin || typeof pin.path !== 'string' || !pin.path || !Number.isSafeInteger(pin.bytes)
      || pin.bytes < 0 || !shaPattern.test(pin.sha256 ?? '')) fail('A full path/bytes/lowercase-SHA256 identity is required');
  const repo = pin.repository ?? repository;
  if (!['cuwacunu_embedding', 'cuwacunu_dev', 'Mityane-geometrics'].includes(repo)) fail(`Unknown historical repository: ${repo}`);
  return { repository: repo, path: pin.path, bytes: pin.bytes, sha256: pin.sha256 };
}

export function identityKey(pin, repository) {
  const id = identity(pin, repository);
  return JSON.stringify([id.repository, id.path, id.bytes, id.sha256]);
}

// V5's frozen receipt used snapshot_path. Later receipts used snapshot. This is
// a versioned schema distinction, never a guess based on whichever field exists.
export function preservedSnapshotLocator(manifestKind, file) {
  const laterKinds = new Set([
    'causal_fresh_v6_prior_live_metadata_preservation_v1',
    'causal_fresh_v7_prior_live_metadata_snapshot_v1',
    'causal_fresh_v8_prior_live_metadata_snapshot_v1',
    'cpb_v9_prior_live_whole_metadata_snapshot_v1',
    ...[10, 11, 12, 13].map(version => `causal_fresh_v${version}_prior_live_metadata_snapshot_v1`),
  ]);
  const field = manifestKind === 'causal_fresh_v5_live_metadata_before_finalization_v1' ? 'snapshot_path'
    : laterKinds.has(manifestKind) ? 'snapshot' : null;
  if (!field) fail(`Unsupported preservation snapshot schema: ${manifestKind}`);
  identity(file);
  const locator = file[field];
  if (typeof locator !== 'string' || !locator || path.posix.isAbsolute(locator) || path.win32.isAbsolute(locator)
      || slash(locator).split('/').some(part => !part || part === '.' || part === '..')
      || Object.hasOwn(file, field === 'snapshot' ? 'snapshot_path' : 'snapshot')) {
    fail(`Exact ${manifestKind}.${field} locator required for ${file.path}`);
  }
  return { path: locator, bytes: file.bytes, sha256: file.sha256, field };
}

export async function hashFile(absolutePath) {
  const link = await lstat(absolutePath);
  if (!link.isFile() || link.isSymbolicLink()) fail(`Expected unlinked regular file: ${absolutePath}`);
  const before = await stat(absolutePath, { bigint: true });
  if (!before.isFile()) fail(`Expected regular file: ${absolutePath}`);
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(absolutePath)) { hash.update(chunk); bytes += chunk.length; }
  const after = await stat(absolutePath, { bigint: true });
  for (const field of ['size', 'mtimeNs', 'ctimeNs', 'ino', 'dev']) {
    if (before[field] !== after[field]) fail(`File changed while hashing: ${absolutePath}`);
  }
  if (BigInt(bytes) !== before.size) fail(`Short file read: ${absolutePath}`);
  return { bytes, sha256: hash.digest('hex') };
}

async function containedPath(root, relative) {
  if (typeof relative !== 'string' || path.isAbsolute(relative) || slash(relative).split('/').includes('..')) fail(`Unsafe current locator: ${relative}`);
  const absolute = path.resolve(root, relative);
  const realRoot = await realpath(root);
  const actual = await realpath(absolute);
  const rel = path.relative(realRoot, actual);
  if (rel.startsWith(`..${path.sep}`) || rel === '..' || path.isAbsolute(rel)) fail(`Locator escapes repository: ${relative}`);
  if (actual !== absolute || realRoot !== path.resolve(root)) fail(`Redirected current locator: ${relative}`);
  return absolute;
}

function sameBytes(expected, observed) {
  return expected.bytes === observed.bytes && expected.sha256 === observed.sha256;
}

// These are candidate locators only. A candidate never establishes an identity.
export function historicalCandidates(pin) {
  const id = identity(pin);
  if (id.repository === 'Mityane-geometrics') {
    const p = slash(id.path);
    if (p.startsWith('output/')) return [`.temp/${p.slice(7)}`];
    if (p.startsWith('archive/')) return [`.archive/${p.slice(8)}`];
    return [p];
  }
  if (id.repository !== HISTORICAL_REPOSITORY) return [];
  let p = slash(id.path);
  const oldRoot = 'C:/Work/cuwacunu/cuwacunu_embedding/';
  if (p.startsWith(oldRoot)) p = p.slice(oldRoot.length);
  if (p.startsWith('/') || /^[A-Za-z]:/.test(p) || p.split('/').includes('..')) return [];
  if (['output/', '.build/', '.external/'].some(prefix => p.startsWith(prefix))) return [`code/embeddings/${p}`];
  return [`${ARCHIVE}/${p}`];
}

export function createIdentityResolver({ root, manifest }) {
  if (!manifest || manifest.kind !== 'cpb_v14_historical_identity_alias_manifest_v1' || !Array.isArray(manifest.aliases)) fail('Unsupported alias manifest');
  const index = new Map();
  for (const entry of manifest.aliases) {
    const key = identityKey(entry.original);
    if (!sameBytes(entry.original, entry.observed) || typeof entry.current_path !== 'string' || !entry.provenance?.length) fail(`Invalid alias entry: ${key}`);
    const existing = index.get(key);
    if (existing && existing.current_path !== entry.current_path) fail(`Conflicting locators for exact identity: ${key}`);
    index.set(key, entry);
  }
  async function resolve(pin, { repository = HISTORICAL_REPOSITORY } = {}) {
    const id = identity(pin, repository);
    const entry = index.get(identityKey(id));
    if (!entry) fail(`Unregistered exact historical identity: ${identityKey(id)}`);
    const absolute_path = await containedPath(root, entry.current_path);
    const observed = await hashFile(absolute_path);
    if (!sameBytes(id, observed)) fail(`Historical alias bytes changed: ${identityKey(id)}`);
    return { ...id, absolute_path, current_path: entry.current_path, provenance: entry.provenance };
  }
  return { resolve, verify: resolve };
}

export async function createResolver({ repoRoot, manifestPath }) {
  return createIdentityResolver({ root: repoRoot, manifest: parseJSON(await readFile(manifestPath), { duplicates: 'reject', integers: 'bigint' }) });
}

function *pinsIn(value, pointer = '') {
  if (!value || typeof value !== 'object') return;
  if (!Array.isArray(value) && typeof value.path === 'string' && Number.isSafeInteger(value.bytes) && shaPattern.test(value.sha256 ?? '')) yield { pin: value, pointer };
  for (const [key, child] of Object.entries(value)) yield *pinsIn(child, `${pointer}/${key.replaceAll('~', '~0').replaceAll('/', '~1')}`);
}

export async function buildAliasInventory({ root }) {
  root = path.resolve(root);
  const aliases = new Map();
  const hashCache = new Map();
  const documents = new Map();
  const failures = [];
  const coverage = {};
  const roots = [];
  let hashReads = 0;
  async function observe(current) {
    const absolute = await containedPath(root, current);
    if (!hashCache.has(current)) { hashCache.set(current, await hashFile(absolute)); hashReads++; }
    return hashCache.get(current);
  }
  async function document(current, expected) {
    const observed = await observe(current);
    if (expected && !sameBytes(expected, observed)) fail(`Bound manifest changed: ${current}`);
    if (!documents.has(current)) documents.set(current, parseJSON(await readFile(await containedPath(root, current)), { duplicates: 'last', integers: 'bigint' }));
    return { data: documents.get(current), binding: { repository: 'Mityane-geometrics', path: current, ...observed } };
  }
  async function add(pin, candidates, provenance) {
    const original = identity(pin);
    const key = identityKey(original);
    const prior = aliases.get(key);
    const evidence = Array.isArray(provenance) ? provenance : [provenance];
    if (prior) {
      for (const item of evidence) if (!prior.provenance.some(old => JSON.stringify(old) === JSON.stringify(item))) prior.provenance.push(item);
      return prior;
    }
    const attempts = [];
    for (const current of [...new Set(candidates)]) {
      try {
        const observed = await observe(current);
        if (!sameBytes(original, observed)) { attempts.push({ current_path: current, reason: 'identity mismatch', observed }); continue; }
        const entry = { original, current_path: current, observed, provenance: evidence };
        aliases.set(key, entry);
        return entry;
      } catch (error) { attempts.push({ current_path: current, reason: error.message }); }
    }
    fail(`No verified locator for ${key}; ${JSON.stringify(attempts)}`);
  }
  const source = (binding, pointer) => ({ manifest: binding, pointer });
  const historicalOutput = rel => `code/embeddings/${rel}`;
  async function loadOriginal(originalPath, expected) {
    const current = historicalCandidates({ path: originalPath, bytes: expected?.bytes ?? 0, sha256: expected?.sha256 ?? '0'.repeat(64) })[0];
    const loaded = await document(current, expected);
    roots.push(loaded.binding);
    await add({ path: originalPath, bytes: loaded.binding.bytes, sha256: loaded.binding.sha256 }, [current], source(loaded.binding, 'self-observed-root'));
    return loaded;
  }
  function requireCount(label, value, expected) {
    if (!Array.isArray(value) || value.length !== expected) fail(`${label}: expected ${expected} members, found ${value?.length}`);
    coverage[label] = value.length;
  }

  // Root migration receipts are pinned by the documentary snapshot; current
  // documentary targets are deliberately not admitted using this stale snapshot.
  const snapshot = await document('doc/EMBEDDINGS_V14_CLOSEOUT_INPUTS_2026-10-18.json');
  const snapshotPins = new Map(snapshot.data.existing_files.map(pin => [pin.path, pin]));
  async function migration(name) {
    const current = `${ARCHIVE}/doc/causal_fresh_v14_migration/${name}.json`;
    const expected = snapshotPins.get(current);
    if (!expected) fail(`Missing migration receipt root pin: ${current}`);
    const result = await document(current, expected);
    roots.push(result.binding);
    await add({ repository: HISTORICAL_REPOSITORY, path: `doc/causal_fresh_v14_migration/${name}.json`, bytes: expected.bytes, sha256: expected.sha256 }, [current], source(snapshot.binding, `/existing_files/${snapshot.data.existing_files.indexOf(expected)}`));
    return result;
  }
  async function pinnedManifest(originalPath) {
    const current = historicalOutput(originalPath);
    const expected = snapshotPins.get(current);
    if (!expected) fail(`Missing generation manifest root pin: ${current}`);
    const loaded = await document(current, expected);
    roots.push(loaded.binding);
    await add({ path: originalPath, bytes: expected.bytes, sha256: expected.sha256 }, [current], source(snapshot.binding, `/existing_files/${snapshot.data.existing_files.indexOf(expected)}`));
    return loaded;
  }
  const finalGeneration = await pinnedManifest('output/engineering/causal-fresh-v14-concurrent-documentary-change/final-cleanup-generation-001/manifest.json');
  requireCount('final_cleaned_guides', finalGeneration.data.files, 14);
  const before = await loadOriginal('output/engineering/embedding-scope-cleanup-2026-10-18/before.json');
  requireCount('scope_cleanup_whole_before', before.data, 14);
  for (const [i, old] of before.data.entries()) {
    const matches = finalGeneration.data.files.filter(file => file.prior_whole?.path === old.path && file.prior_whole.sha256 === old.sha256);
    if (matches.length !== 1) fail(`Unique prior_whole length unavailable: ${old.path}`);
    const prior = matches[0].prior_whole;
    await add(prior, [historicalOutput(prior.snapshot)], [source(before.binding, `/${i}`), source(finalGeneration.binding, `/files/${finalGeneration.data.files.indexOf(matches[0])}/prior_whole`)]);
    await add({ path: prior.snapshot, bytes: prior.bytes, sha256: prior.sha256 }, [historicalOutput(prior.snapshot)], source(finalGeneration.binding, `/files/${finalGeneration.data.files.indexOf(matches[0])}/prior_whole`));
  }
  for (const [label, original, count] of [
    ['prior_six_live_files', 'output/engineering/causal-fresh-v14-finalization/prior-live-001/manifest.json', 6],
    ['intermediate_guides', 'output/engineering/causal-fresh-v14-concurrent-documentary-change/attempt-001/manifest.json', 4],
    ['paused_pointer_before', 'output/engineering/causal-fresh-v14-migration/attempt-001/guide-before-001/manifest.json', 2],
  ]) {
    const loaded = await pinnedManifest(original);
    requireCount(label, loaded.data.files, count);
    for (const [i, file] of loaded.data.files.entries()) {
      await add(file, [historicalOutput(file.snapshot)], source(loaded.binding, `/files/${i}`));
      await add({ path: file.snapshot, bytes: file.bytes, sha256: file.sha256 }, [historicalOutput(file.snapshot)], source(loaded.binding, `/files/${i}`));
    }
  }
  for (const [i, file] of finalGeneration.data.files.entries()) {
    await add(file, [historicalOutput(file.snapshot)], source(finalGeneration.binding, `/files/${i}`));
    await add({ path: file.snapshot, bytes: file.bytes, sha256: file.sha256 }, [historicalOutput(file.snapshot)], source(finalGeneration.binding, `/files/${i}`));
  }
  const pointer = await migration('pointer-update');
  requireCount('paused_pointer_after', pointer.data.files, 2);
  for (const [i, file] of pointer.data.files.entries()) await add(file, historicalCandidates(file), source(pointer.binding, `/files/${i}`));

  const notes = await loadOriginal('doc/causal_fresh_v14/front-source-notes-001.json');
  requireCount('source_note_aliases', notes.data.files, 4);
  for (const [i, file] of notes.data.files.entries()) {
    const beforePath = historicalOutput(file.whole_prior_copy);
    const observed = await observe(beforePath);
    // The prefix receipt establishes the whole new generation and its exact
    // preserved suffix. The prior tuple is derived only after suffix checking.
    const bytes = await readFile(await containedPath(root, beforePath));
    const current = aliases.get(identityKey(file));
    if (!current) fail(`Source-note generation was not registered by whole snapshots: ${file.path}`);
    const next = await readFile(await containedPath(root, current.current_path));
    if (file.prefix_bytes + bytes.length !== file.bytes || !next.subarray(file.prefix_bytes).equals(bytes)) fail(`Source-note suffix mismatch: ${file.path}`);
    await add({ path: file.path, ...observed }, [beforePath], source(notes.binding, `/files/${i}/whole_prior_copy`));
  }
  const singleton = await loadOriginal('doc/causal_fresh_v13/front-notes-preservation-001.json');
  if (singleton.data.prior_bytes_preserved_as_exact_suffix !== true || !sameBytes(singleton.data.prior, singleton.data.whole_prior_snapshot)) fail('Invalid V13 singleton prior binding');
  await add(singleton.data.prior, [historicalOutput(singleton.data.whole_prior_snapshot.path)], source(singleton.binding, '/prior+whole_prior_snapshot'));
  const v13final = await loadOriginal('doc/causal_fresh_v13/final-preservation.json');
  const priorV13 = await loadOriginal(v13final.data.V13_prior_snapshot.path, v13final.data.V13_prior_snapshot);
  for (const [i, file] of priorV13.data.files.entries()) await add(file, [historicalOutput(file.snapshot)], source(priorV13.binding, `/files/${i}`));
  await add(singleton.data.new, [historicalOutput(priorV13.data.files.find(file => file.path === singleton.data.new.path && sameBytes(file, singleton.data.new))?.snapshot ?? '')], source(singleton.binding, '/new'));
  const singletonBefore = await readFile(await containedPath(root, historicalOutput(singleton.data.whole_prior_snapshot.path)));
  const singletonAfter = await readFile(await containedPath(root, aliases.get(identityKey(singleton.data.new)).current_path));
  if (singleton.data.prefix_bytes + singletonBefore.length !== singletonAfter.length || !singletonAfter.subarray(singleton.data.prefix_bytes).equals(singletonBefore)) fail('V13 singleton exact prior suffix mismatch');
  for (const [i, file] of v13final.data.old_authorized_snapshots.entries()) {
    await add({ path: file.path, bytes: file.old_bytes, sha256: file.old_sha256 }, [historicalOutput(file.snapshot)], source(v13final.binding, `/old_authorized_snapshots/${i}`));
  }
  coverage.v13_singleton_registered_before_repair_traversal = true;

  // Register the complete earlier documentary chain before walking any of its
  // pins. These bindings are the same whole-file generations declared by the
  // frozen V13 repaired preservation verifier; no live guide substitutes for one.
  const historicalFinals = [v13final];
  let generation = v13final;
  for (let version = 12; version >= 6; version--) {
    const pin = generation.data[`prior_V${version}_final`];
    if (!pin) fail(`Missing prior V${version} preservation link`);
    generation = await loadOriginal(pin.path, pin);
    historicalFinals.push(generation);
  }
  for (const receipt of historicalFinals) {
    for (const [i, file] of (receipt.data.old_authorized_snapshots ?? []).entries()) {
      await add({ path: file.path, bytes: file.old_bytes, sha256: file.old_sha256 }, [historicalOutput(file.snapshot)], source(receipt.binding, `/old_authorized_snapshots/${i}`));
    }
    for (const [i, file] of (receipt.data.V5_metadata6_chain ?? []).entries()) {
      await add({ path: file.path, bytes: file.prior_bytes, sha256: file.prior_sha256 }, [historicalOutput(file.prior_snapshot)], source(receipt.binding, `/V5_metadata6_chain/${i}`));
    }
    for (const [key, pin] of Object.entries(receipt.data)) {
      if (!/^V\d+_prior_snapshot$/.test(key)) continue;
      const snapshot = await loadOriginal(pin.path, pin);
      for (const [i, file] of snapshot.data.files.entries()) {
        const locator = preservedSnapshotLocator(snapshot.data.kind, file);
        await add(file, [historicalOutput(locator.path)], source(snapshot.binding, `/files/${i}`));
        await add(locator, [historicalOutput(locator.path)], source(snapshot.binding, `/files/${i}/${locator.field}`));
      }
    }
  }
  coverage.earlier_preservation_generations_registered_before_traversal = historicalFinals.length;

  const external = await migration('portable-external-checkpoints');
  const externalFiles = external.data.whole_copy_aliases;
  requireCount('external_portable_checkpoints', externalFiles, 57);
  for (const [i, record] of externalFiles.entries()) {
    if (!sameBytes(record.original_typed_record, record.whole_copy)) fail(`Portable checkpoint copy identity mismatch: ${i}`);
    await add(record.original_typed_record, [historicalOutput(record.whole_copy.path)], source(external.binding, `/whole_copy_aliases/${i}`));
    await add(record.whole_copy, [historicalOutput(record.whole_copy.path)], source(external.binding, `/whole_copy_aliases/${i}/whole_copy`));
  }
  const availability = await migration('checkpoint-availability');
  const preservation = await loadOriginal(availability.data.historical_inventory.path, availability.data.historical_inventory);
  requireCount('historical_critical_checkpoints', preservation.data.critical_checkpoint_pins, 119);
  requireCount('canonical_v14_checkpoints', availability.data.new_V14_verified_pins, 5);
  const critical = [...preservation.data.critical_checkpoint_pins, ...availability.data.new_V14_verified_pins];
  requireCount('complete_critical_checkpoints', critical, 124);
  if (new Set(critical.map(pin => identityKey(pin))).size !== 124) fail('Critical checkpoint identities must be unique');
  if (critical.reduce((total, pin) => total + pin.bytes, 0) !== 287027201) fail('Critical checkpoint byte total changed');
  for (const [i, pin] of critical.entries()) await add(pin, historicalCandidates(pin), source(i < 119 ? preservation.binding : availability.binding, i < 119 ? `/critical_checkpoint_pins/${i}` : `/new_V14_verified_pins/${i - 119}`));

  const paused = await migration('paused-finalization-mirror');
  requireCount('paused_finalization_payload', paused.data.mapped_files, 114);
  if (paused.data.mapped_files.reduce((sum, file) => sum + file.host.bytes, 0) !== 21316710) fail('Paused payload total changed');
  const relativeSet = new Set();
  for (const [i, file] of paused.data.mapped_files.entries()) {
    if (relativeSet.has(file.relative) || file.host.path !== `${paused.data.host_payload}/${file.relative}` || file.container_path !== `${paused.data.container_root}/${file.relative}`) fail(`Invalid paused payload mapping: ${file.relative}`);
    relativeSet.add(file.relative);
    const current = historicalOutput(file.host.path);
    await add(file.host, [current], source(paused.binding, `/mapped_files/${i}/host`));
    await add({ path: file.container_path, bytes: file.host.bytes, sha256: file.host.sha256 }, [current], source(paused.binding, `/mapped_files/${i}`));
  }
  const physicalPayloadMembers = [];
  async function inventoryPayload(relative = '') {
    const directory = path.join(root, PAYLOAD, relative);
    if (await realpath(directory) !== directory) fail(`Redirected paused payload directory: ${directory}`);
    for (const member of await readdir(directory, { withFileTypes: true })) {
      const name = relative ? `${relative}/${member.name}` : member.name;
      if (member.isSymbolicLink()) fail(`Redirected paused payload member: ${name}`);
      if (member.isDirectory()) await inventoryPayload(name);
      else if (member.isFile()) physicalPayloadMembers.push(name);
      else fail(`Non-regular paused payload member: ${name}`);
    }
  }
  await inventoryPayload();
  if (physicalPayloadMembers.length !== relativeSet.size || physicalPayloadMembers.some(member => !relativeSet.has(member))) fail('Paused payload physical member set differs from receipt');
  coverage.paused_payload_exact_member_set = true;
  const card = await document(`${PAYLOAD}/card-source-001/finalization-card.json`, paused.data.card);
  requireCount('card_inputs', card.data.inputs, 49);
  for (const [i, pin] of card.data.inputs.entries()) {
    await add(pin, [`${PAYLOAD}/card-source-001/${pin.staged_path}`, ...historicalCandidates(pin)], source(card.binding, `/inputs/${i}`));
  }
  const sourceManifestPin = card.data.inputs.find(pin => pin.id === 'source_manifest');
  const sourceManifest = await document(aliases.get(identityKey(sourceManifestPin)).current_path, sourceManifestPin);
  requireCount('frozen_scientific_sources', sourceManifest.data.files, 145);
  if (new Set(sourceManifest.data.files.map(pin => pin.path)).size !== 145) fail('Frozen source paths must be unique');
  for (const [i, pin] of sourceManifest.data.files.entries()) {
    const provenance = source(sourceManifest.binding, `/files/${i}`);
    const entry = await add(pin, historicalCandidates(pin), provenance);
    await add({ ...pin, path: `${card.data.container_attempt}/source/${pin.path}` }, [entry.current_path], provenance);
  }
  const buildPin = card.data.inputs.find(pin => pin.id === 'build');
  const build = await document(aliases.get(identityKey(buildPin)).current_path, buildPin);
  requireCount('main_binaries', build.data.binaries, 7);
  for (const [i, pin] of build.data.binaries.entries()) await add(pin, [`code/embeddings/output/engineering/causal-fresh-v14/attempt-001/build-binaries/${path.posix.basename(pin.path)}`], source(build.binding, `/binaries/${i}`));

  // The complete main mirror also binds absolute historical container locations.
  const mirrorPin = card.data.inputs.find(pin => pin.id === 'mirror');
  const mirror = await document(aliases.get(identityKey(mirrorPin)).current_path, mirrorPin);
  requireCount('complete_main_mirror', mirror.data.files, 1149);
  for (const [i, file] of mirror.data.files.entries()) {
    const current = historicalOutput(file.path);
    await add(file, [current], source(mirror.binding, `/files/${i}`));
    if (file.container_path) await add({ ...file, path: file.container_path }, [current], source(mirror.binding, `/files/${i}/container_path`));
  }

  // Close the saved evidence graph, including V13's failure/repair record.
  // Full tuples are visited once. Context-sensitive capsule source paths have
  // already been qualified above; changed guide generations resolve only through
  // whole copies registered before traversal.
  const pending = [...historicalFinals, preservation, card, sourceManifest, build, mirror,
    ...await Promise.all(card.data.inputs.filter(pin => pin.format === 'json').map(async pin => document(aliases.get(identityKey(pin)).current_path, pin)))];
  const visitedDocuments = new Set();
  const visitedPins = new Set();
  for (let cursor = 0; cursor < pending.length; cursor++) {
    const current = pending[cursor];
    if (visitedDocuments.has(current.binding.path)) continue;
    visitedDocuments.add(current.binding.path);
    for (const { pin, pointer: jsonPointer } of pinsIn(current.data)) {
      const key = identityKey(pin);
      if (visitedPins.has(key)) continue;
      visitedPins.add(key);
      try {
        const entry = await add(pin, historicalCandidates(pin), source(current.binding, jsonPointer));
        if (entry.current_path.endsWith('.json')) pending.push(await document(entry.current_path, pin));
      } catch (error) {
        failures.push({ identity: identity(pin), manifest: current.binding, pointer: jsonPointer, error: error.message });
      }
    }
  }
  coverage.transitive_documents = visitedDocuments.size;
  coverage.transitive_identities = visitedPins.size;
  const manifest = {
    kind: 'cpb_v14_historical_identity_alias_manifest_v1',
    created_utc: new Date().toISOString(),
    protocol: 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/aliases.mjs',
    roots, coverage,
    policy: 'Exact repository/path/bytes/SHA256 tuples only; no path-only or global byte-identity substitution.',
    protocol_identity: { path: 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/aliases.mjs', ...await observe('code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/aliases.mjs') },
    aliases: [...aliases.values()].sort((a, b) => identityKey(a.original).localeCompare(identityKey(b.original))),
  };
  const verification = {
    kind: 'cpb_v14_historical_identity_alias_verification_v1',
    created_utc: new Date().toISOString(),
    status: failures.length ? 'failed' : 'passed',
    passed: failures.length === 0,
    protocol_identity: manifest.protocol_identity,
    shared_parser_identity: { path: 'code/shared/artifacts/node/v1/files.mjs', ...await observe('code/shared/artifacts/node/v1/files.mjs') },
    coverage, aliases_verified: aliases.size, unique_physical_file_hash_reads: hashReads,
    failures, tensor_decodes: 0, model_calls: 0, head_fits: 0, training_updates: 0,
  };
  return { manifest, verification };
}

async function cli() {
  if (process.platform !== 'linux') fail('Managed Linux container only');
  await lstat('/.dockerenv');
  const args = process.argv.slice(2);
  if (args.shift() !== 'build') fail('Usage: aliases.mjs build --root /mityane --output <new aliases.json> --verification <new receipt.json>');
  const options = {};
  while (args.length) {
    const key = args.shift();
    if (!['--root', '--output', '--verification'].includes(key) || !args.length || Object.hasOwn(options, key)) fail(`Invalid argument: ${key}`);
    options[key] = args.shift();
  }
  for (const key of ['--root', '--output', '--verification']) if (!options[key]) fail(`Missing ${key}`);
  if (path.resolve(options['--root']) !== '/mityane' || await realpath('/mityane') !== '/mityane') fail('The admitted /mityane managed-container repository is required');
  const out = path.resolve(options['--output']);
  const verificationPath = path.resolve(options['--verification']);
  if (out === verificationPath) fail('Output paths must differ');
  const admittedPrefix = '/mityane/.temp/embeddings/closeout/cpb-v14/';
  for (const target of [out, verificationPath]) {
    const relative = target.startsWith(admittedPrefix) ? target.slice(admittedPrefix.length) : '';
    if (!/^attempt-\d{3}\/[^/]+\.json$/.test(relative)) fail(`Fresh admitted closeout-leaf JSON output required: ${target}`);
    for (let ancestor = path.dirname(target); ; ancestor = path.dirname(ancestor)) {
      const info = await lstat(ancestor);
      if (!info.isDirectory() || info.isSymbolicLink() || await realpath(ancestor) !== ancestor) fail(`Redirected output ancestor: ${ancestor}`);
      if (ancestor === path.dirname(ancestor)) break;
    }
  }
  if (path.dirname(out) !== path.dirname(verificationPath)) fail('Both outputs must belong to the same admitted attempt');
  const admissionPath = path.join(path.dirname(out), 'admission.json');
  const admission = { path: admissionPath, ...await hashFile(admissionPath) };
  for (const target of [out, verificationPath]) {
    try { await access(target); fail(`Refusing to overwrite: ${target}`); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  const { manifest, verification } = await buildAliasInventory({ root: options['--root'] });
  manifest.admission = admission;
  verification.admission = admission;
  await writeFile(out, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
  verification.alias_manifest = { path: out, ...await hashFile(out) };
  await writeFile(verificationPath, `${JSON.stringify(verification, null, 2)}\n`, { flag: 'wx' });
  console.log(JSON.stringify({ status: verification.status, aliases: manifest.aliases.length, failures: verification.failures.length }));
  if (verification.status !== 'passed') process.exitCode = 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) cli().catch(error => { console.error(error.stack); process.exitCode = 1; });
