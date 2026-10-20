// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createIdentityResolver, hashFile, historicalCandidates, identity, identityKey, preservedSnapshotLocator } from './aliases.mjs';

async function fixture(callback) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'cpb-v14-alias-test-'));
  try { await callback(root); } finally { await rm(root, { recursive: true, force: true }); }
}

async function row(root, current_path, text, original_path = 'doc/README.md', repository = 'cuwacunu_embedding') {
  await writeFile(path.join(root, current_path), text);
  const observed = await hashFile(path.join(root, current_path));
  return {
    original: { repository, path: original_path, ...observed }, current_path, observed,
    provenance: [{ manifest: { repository: 'Mityane-geometrics', path: 'test-manifest.json', ...observed }, pointer: '/files/0' }],
  };
}
const manifest = aliases => ({ kind: 'cpb_v14_historical_identity_alias_manifest_v1', aliases });

test('full tuple keeps generations and repositories distinct', async () => fixture(async root => {
  const first = await row(root, 'first.txt', 'first generation');
  const second = await row(root, 'second.txt', 'second generation');
  const external = await row(root, 'external.txt', 'first generation', first.original.path, 'cuwacunu_dev');
  const resolver = createIdentityResolver({ root, manifest: manifest([first, second, external]) });
  assert.notEqual(identityKey(first.original), identityKey(second.original));
  assert.notEqual(identityKey(first.original), identityKey(external.original));
  assert.equal((await resolver.resolve(first.original)).current_path, 'first.txt');
  assert.equal((await resolver.resolve(second.original)).current_path, 'second.txt');
  assert.equal((await resolver.resolve(external.original)).current_path, 'external.txt');
  assert.equal((await resolver.resolve(first.original)).absolute_path, path.join(root, 'first.txt'));
}));

test('path-only, wrong size, unknown repository, and missing tuple fail closed', async () => fixture(async root => {
  const entry = await row(root, 'first.txt', 'bytes');
  const resolver = createIdentityResolver({ root, manifest: manifest([entry]) });
  await assert.rejects(resolver.resolve({ path: entry.original.path }), /full path/);
  await assert.rejects(resolver.resolve({ ...entry.original, bytes: entry.original.bytes + 1 }), /Unregistered exact/);
  await assert.rejects(resolver.resolve({ ...entry.original, repository: 'other' }), /Unknown historical repository/);
  await assert.rejects(resolver.resolve({ ...entry.original, path: 'other-path' }), /Unregistered exact/);
}));

test('resolver freshly hashes physical bytes after manifest creation', async () => fixture(async root => {
  const entry = await row(root, 'first.txt', 'saved bytes');
  const resolver = createIdentityResolver({ root, manifest: manifest([entry]) });
  await writeFile(path.join(root, 'first.txt'), 'other bytes');
  await assert.rejects(resolver.resolve(entry.original), /alias bytes changed/);
}));

test('conflicting and internally contradictory aliases are rejected', async () => fixture(async root => {
  const first = await row(root, 'first.txt', 'saved bytes');
  const second = { ...first, current_path: 'second.txt' };
  assert.throws(() => createIdentityResolver({ root, manifest: manifest([first, second]) }), /Conflicting locators/);
  assert.throws(() => createIdentityResolver({ root, manifest: manifest([{ ...first, observed: { bytes: 0, sha256: '0'.repeat(64) } }]) }), /Invalid alias/);
  assert.throws(() => createIdentityResolver({ root, manifest: manifest([{ ...first, provenance: [] }]) }), /Invalid alias/);
}));

test('current locator traversal and symlink escape are refused', async () => fixture(async root => {
  const first = await row(root, 'first.txt', 'saved bytes');
  const bad = { ...first, current_path: '../elsewhere' };
  await assert.rejects(createIdentityResolver({ root, manifest: manifest([bad]) }).resolve(first.original), /Unsafe current locator/);
  await fixture(async outside => {
    await writeFile(path.join(outside, 'outside.txt'), 'saved bytes');
    await symlink(path.join(outside, 'outside.txt'), path.join(root, 'escape'));
    await assert.rejects(createIdentityResolver({ root, manifest: manifest([{ ...first, current_path: 'escape' }]) }).resolve(first.original), /escapes repository/);
  });
}));

test('symlink redirects inside the repository cannot satisfy an alias', async () => fixture(async root => {
  const first = await row(root, 'first.txt', 'saved bytes');
  await symlink(path.join(root, 'first.txt'), path.join(root, 'internal-link'));
  await assert.rejects(createIdentityResolver({ root, manifest: manifest([{ ...first, current_path: 'internal-link' }]) }).resolve(first.original), /Redirected current locator/);
}));

test('historical namespace mapping never applies Mityane output mapping inside component', () => {
  const full = { path: 'output/checkpoint.pt', bytes: 1, sha256: 'a'.repeat(64) };
  assert.deepEqual(historicalCandidates(full), ['code/embeddings/output/checkpoint.pt']);
  assert.deepEqual(historicalCandidates({ ...full, repository: 'Mityane-geometrics' }), ['.temp/checkpoint.pt']);
  assert.deepEqual(historicalCandidates({ ...full, repository: 'cuwacunu_dev' }), []);
  assert.deepEqual(historicalCandidates({ ...full, path: 'doc/README.md' }), ['.archive/embeddings/pre-cleanup-2026-10-18/doc/README.md']);
  assert.deepEqual(historicalCandidates({ ...full, path: '../secret' }), []);
  assert.deepEqual(historicalCandidates({ ...full, path: '/opt/unknown' }), []);
  assert.throws(() => identity({ ...full, bytes: -1 }), /full path/);
});

test('preserved whole-file locators obey exact V5 and later receipt schemas', () => {
  const pin = { path: 'doc/README.md', bytes: 11, sha256: 'a'.repeat(64) };
  const originalCopy = 'output/engineering/example/prior-live/doc/README.md';
  const v5 = 'causal_fresh_v5_live_metadata_before_finalization_v1';
  const v6 = 'causal_fresh_v6_prior_live_metadata_preservation_v1';
  assert.deepEqual(preservedSnapshotLocator(v5, { ...pin, snapshot_path: originalCopy }), { path: originalCopy, bytes: 11, sha256: pin.sha256, field: 'snapshot_path' });
  assert.deepEqual(preservedSnapshotLocator(v6, { ...pin, snapshot: originalCopy }), { path: originalCopy, bytes: 11, sha256: pin.sha256, field: 'snapshot' });
  assert.throws(() => preservedSnapshotLocator(v5, { ...pin, snapshot: originalCopy }), /snapshot_path locator required/);
  assert.throws(() => preservedSnapshotLocator(v6, { ...pin, snapshot_path: originalCopy }), /snapshot locator required/);
  assert.throws(() => preservedSnapshotLocator(v5, { ...pin, snapshot_path: originalCopy, snapshot: originalCopy }), /snapshot_path locator required/);
  assert.throws(() => preservedSnapshotLocator(v5, { ...pin, bytes: undefined, snapshot_path: originalCopy }), /full path/);
  assert.throws(() => preservedSnapshotLocator('unknown', { ...pin, snapshot: originalCopy }), /Unsupported preservation snapshot schema/);
});
