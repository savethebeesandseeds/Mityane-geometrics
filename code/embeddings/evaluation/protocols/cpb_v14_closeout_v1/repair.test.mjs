// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { repairAliasGraph, repairAliasInventory, mirrorMappings, byteCopyMappings, atPointer, repairObservationFields, sameCurrentIdentity, inspectBaseLocators, validateScopeDeclaration } from './repair.mjs';
import { createIdentityResolver } from './aliases.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const bind = (file, bytes) => ({ path: file, bytes: Buffer.byteLength(bytes), sha256: hash(bytes) });
const basePair = { manifest: bind('base.json', '{}'), verification: bind('base-verification.json', '{}') };
const evidence = [{ manifest: bind('frozen-record.json', '{}'), pointer: '/files/0' }];
const base = aliases => ({ kind: 'cpb_v14_historical_identity_alias_manifest_v1', aliases });
async function fixture(callback) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'cpb-v14-repair-'));
  try { await callback(root); } finally { await rm(root, { recursive: true, force: true }); }
}
async function row(root, locator, text, original = locator, repository = 'cuwacunu_embedding') {
  const bytes = typeof text === 'string' ? text : JSON.stringify(text);
  await mkdir(path.dirname(path.join(root, locator)), { recursive: true });
  await writeFile(path.join(root, locator), bytes);
  const pin = { repository, ...bind(original, bytes) };
  return { original: pin, current_path: locator, observed: { bytes: pin.bytes, sha256: pin.sha256 }, provenance: evidence };
}
const failure = original => ({ identity: original, manifest: bind('root.json', '{}'), pointer: '/target', error: 'No exact locator' });
const run = (root, entries, failures = [], options = {}) => repairAliasGraph({ root, baseManifest: base(entries), baseVerificationBinding: basePair, originalFailures: failures, ...options });

test('reuse is explicit and later resolver freshly detects changed bulk bytes', async () => fixture(async root => {
  const artifact = await row(root, 'saved.pt', 'saved bulk bytes', 'output/saved.pt');
  const parent = await row(root, 'parent.json', { target: artifact.original }, 'doc/parent.json');
  await writeFile(path.join(root, 'saved.pt'), 'other bulk bytes');
  const result = await run(root, [artifact, parent]);
  assert.equal(result.failures.length, 0);
  assert.equal(result.reused_aliases, 2);
  assert.equal(result.new_non_metadata_hash_reads, 0);
  assert.equal(result.new_physical_hash_reads, 1);
  assert.deepEqual(result.aliases.find(entry => entry.current_path === 'saved.pt').observation_reused_from, basePair);
  const resolver = createIdentityResolver({ root, manifest: base(result.aliases) });
  await assert.rejects(resolver.resolve(artifact.original), /alias bytes changed/);
}));

test('JSON metadata bytes are rechecked and changed generations remain a failure', async () => fixture(async root => {
  const parent = await row(root, 'parent.json', '{"x":1}', 'doc/parent.json');
  await writeFile(path.join(root, 'parent.json'), '{"x":2}');
  const result = await run(root, [parent], [failure(parent.original)]);
  assert.ok(result.failures.some(item => /Metadata changed/.test(item.error)));
  assert.equal(result.base_failure_accounting[0].resolved, false);
}));

test('new external alias requires the exact inventory pointer and fresh current bytes', async () => fixture(async root => {
  const copy = await row(root, 'portable/source.h', 'source bytes', 'code/source.h', 'cuwacunu_dev');
  const originalFailure = failure(copy.original);
  const external = { entries: [{ original: copy.original, current_path: copy.current_path, observed: copy.observed,
    provenance: { manifest: bind('inventory.json', '{}'), pointer: '/pins/0' } }] };
  const result = await run(root, [], [originalFailure], { external, externalBinding: bind('external.json', '{}'), inventory: { pins: [copy.original] } });
  assert.equal(result.failures.length, 0);
  assert.equal(result.base_failure_accounting[0].resolved, true);
  assert.equal(result.new_physical_hash_reads, 1);
  assert.deepEqual(result.original_failures, [originalFailure]);
  assert.deepEqual(result.aliases[0].original, copy.original);
  const bad = structuredClone(external);
  bad.entries[0].original.sha256 = '0'.repeat(64);
  await assert.rejects(run(root, [], [], { external: bad, externalBinding: bind('external.json', '{}'), inventory: { pins: [copy.original] } }), /tuple\/provenance mismatch/);
}));

test('all eleven dependency paths resolve through their exact build mirror', async () => fixture(async root => {
  const artifacts = [];
  const files = [];
  for (let i = 0; i < 11; i++) {
    const name = `unit-${i}.d`;
    const artifact = await row(root, `build/${name}`, `owned dependency ${i}`, `output/engineering/causal-fresh-v14/attempt-001/build-binaries/${name}`);
    artifacts.push(artifact);
    files.push({ ...artifact.original, container_path: `/opt/cuwacunu/build/causal-fresh-v14-attempt001/${name}` });
  }
  const mirror = await row(root, 'build-mirror.json', { kind: 'cpb_v14_actual_build_binary_mirror_v1', status: 'passed', files }, 'doc/causal_fresh_v14/build-binaries-mirror.json');
  const misses = files.map(file => failure({ ...file, path: file.container_path }));
  const result = await run(root, [...artifacts, mirror], misses);
  assert.equal(result.failures.length, 0);
  assert.equal(result.adapter_contexts, 11);
  assert.ok(result.base_failure_accounting.every(item => item.resolved));
  assert.equal(result.new_non_metadata_hash_reads, 0);
}));

async function capsuleFixture(root, wrongMember = false) {
  const script = await row(root, 'package/audit.py', 'never executed', 'output/engineering/audit/container-mirror/source/audit.py');
  const bare = { path: wrongMember ? 'other.py' : 'audit.py', bytes: script.original.bytes, sha256: script.original.sha256 };
  const capsule = await row(root, 'package/CAPSULE_MANIFEST.json', { kind: 'fixture_capsule', files: [bare] }, 'output/engineering/audit/container-mirror/source/CAPSULE_MANIFEST.json');
  const container_root = '/opt/cuwacunu/research/causal-fresh-v13-establishment/attempt-001';
  const mapped_files = [script, capsule].map(entry => ({ relative: `source/${path.basename(entry.current_path)}`,
    container: `${container_root}/source/${path.basename(entry.current_path)}`, host: entry.original }));
  const mirror = await row(root, 'mirror.json', { kind: 'cpb_v13_metadata_establishment_mirror_v1', status: 'passed', container_root, mapped_files }, 'doc/mirror.json');
  return { entries: [script, capsule, mirror], bare };
}
test('bare capsule member must match the same paired mirror and exact member path', async () => fixture(async root => {
  const { entries, bare } = await capsuleFixture(root);
  const result = await run(root, entries, [failure(bare)]);
  assert.equal(result.failures.length, 0);
  assert.equal(result.base_failure_accounting[0].resolved, true);
  assert.equal(result.aliases.find(entry => entry.original.path === 'audit.py').current_path, 'package/audit.py');
}));

test('matching bytes elsewhere in a mirror cannot repair a different member path', async () => fixture(async root => {
  const { entries, bare } = await capsuleFixture(root, true);
  const result = await run(root, entries, [failure(bare)]);
  assert.ok(result.failures.some(item => item.identity?.path === 'other.py'));
  assert.equal(result.base_failure_accounting[0].resolved, false);
  assert.equal(result.aliases.some(entry => entry.original.path === 'other.py'), false);
}));

test('unqualified children inherit the external repository instead of embedding', async () => fixture(async root => {
  const child = await row(root, '.archive/embeddings/pre-cleanup-2026-10-18/doc/child.txt', 'only embedding copy', 'doc/child.txt');
  const parent = await row(root, 'external-parent.json', { child: { path: child.original.path, bytes: child.original.bytes, sha256: child.original.sha256 } }, 'doc/parent.json', 'cuwacunu_dev');
  const result = await run(root, [child, parent]);
  assert.ok(result.failures.some(item => item.identity?.repository === 'cuwacunu_dev' && item.identity.path === child.original.path));
  assert.equal(result.aliases.filter(entry => entry.original.path === child.original.path).length, 1);
}));

test('unknown pins remain failures and original failure objects are preserved verbatim', async () => fixture(async root => {
  const unknown = { path: '/opt/unknown/model.pt', bytes: 5, sha256: 'a'.repeat(64) };
  const parent = await row(root, 'parent.json', { unknown }, 'doc/parent.json');
  const old = failure(unknown);
  const saved = structuredClone(old);
  const result = await run(root, [parent], [old]);
  assert.deepEqual(old, saved);
  assert.deepEqual(result.original_failures, [saved]);
  assert.equal(result.base_failure_accounting[0].resolved, false);
  assert.ok(result.failures.some(item => item.identity?.path === unknown.path));
}));

test('base observations reject contradictions and redirected locators', async () => fixture(async root => {
  const item = await row(root, 'saved.txt', 'saved', 'output/saved.txt');
  await assert.rejects(run(root, [{ ...item, observed: { ...item.observed, bytes: 6 } }]), /Malformed original observation/);
  await symlink(path.join(root, 'saved.txt'), path.join(root, 'redirect'));
  await assert.rejects(run(root, [{ ...item, current_path: 'redirect' }]), /Redirected locator/);
  await writeFile(path.join(root, 'saved.txt'), 'length changed');
  await assert.rejects(run(root, [item]), /locator size changed/);
}));

test('mirror adapters reject guessed roots, duplicate mappings and failed receipts', () => {
  const host = { path: 'output/host.txt', bytes: 5, sha256: 'a'.repeat(64) };
  const row = { ...host, container_path: '/opt/cuwacunu/build/host.txt' };
  const receipt = { kind: 'cpb_v14_actual_build_binary_mirror_v1', status: 'passed', files: [row] };
  assert.equal(mirrorMappings(receipt, bind('mirror.json', '{}')).length, 1);
  assert.throws(() => mirrorMappings({ ...receipt, files: [row, row] }, bind('mirror.json', '{}')), /Duplicate historical/);
  assert.throws(() => mirrorMappings({ ...receipt, status: 'failed' }, bind('mirror.json', '{}')), /did not pass/);
  assert.throws(() => mirrorMappings({ ...receipt, files: [{ ...row, container_path: '/etc/passwd' }] }, bind('mirror.json', '{}')), /Exact historical container/);
  assert.throws(() => mirrorMappings(receipt, bind('mirror.json', '{}'), { repository: 'cuwacunu_dev' }), /original embedding repository context/);
  assert.deepEqual(mirrorMappings({ ...receipt, kind: 'invented_mirror' }, bind('mirror.json', '{}')), []);
  assert.equal(atPointer({ 'a/b': { '~value': 7 } }, '/a~1b/~0value'), 7);
  assert.throws(() => atPointer({}, '/missing'), /Missing JSON pointer/);
});

test('control admission and inventory bindings compare repository and normalized exact path', () => {
  const first = bind('attempt/admission.json', '{}');
  assert.equal(sameCurrentIdentity('/mityane', first, { ...first, path: '/mityane/attempt/admission.json' }), true);
  assert.equal(sameCurrentIdentity('/mityane', first, { ...first, path: 'other/admission.json' }), false);
  assert.equal(sameCurrentIdentity('/mityane', first, { ...first, repository: 'cuwacunu_dev' }), false);
  const inventory = bind('frozen/inventory.json', '{}');
  assert.equal(sameCurrentIdentity('/mityane', inventory, { ...inventory, path: 'elsewhere/inventory.json' }), false);
});

test('repair provenance retains failed bulk status and carries the reviewed current parser identity', () => {
  const parser = bind('code/shared/artifacts/node/v1/files.mjs', 'current reviewed parser');
  const args = { baseManifest: { created_utc: '2026-10-18T18:00:00Z', protocol_identity: bind('aliases.mjs', 'original verifier') },
    baseVerification: { created_utc: '2026-10-18T18:01:00Z', status: 'failed', passed: false }, basePair,
    sourceReview: bind('source-review-004.json', '{}'), protocolIdentity: bind('repair.mjs', 'repair'), sharedParser: parser };
  const fields = repairObservationFields(args);
  assert.deepEqual(fields.shared_parser_identity, parser);
  assert.equal(fields.original_bulk_observation.verification_status, 'failed');
  assert.equal(fields.original_bulk_observation.verification_created_utc, args.baseVerification.created_utc);
  assert.equal(fields.bulk_observation_reused, true);
  assert.equal(fields.all_aliases_freshly_hashed_this_generation, false);
  assert.throws(() => repairObservationFields({ ...args, sharedParser: undefined }), /full path/);
  assert.throws(() => repairObservationFields({ ...args, baseVerification: { status: 'passed', passed: true } }), /failure status must remain explicit/);
});

test('V4 and V6 recovery main mirrors retain explicit full container/host pairs', () => {
  const pin = bind('output/engineering/example/result.json', '{}');
  for (const kind of ['causal_fresh_v4_container_to_local_evidence_mirror_v1', 'causal_fresh_v6_proof_recovery_container_to_local_evidence_mirror_v1']) {
    const rows = mirrorMappings({ kind, status: 'passed', file_count: 1, files: [{ ...pin, container_path: '/opt/cuwacunu/research/example/result.json' }] }, bind('mirror.json', '{}'));
    assert.equal(rows[0].original.path, '/opt/cuwacunu/research/example/result.json');
    assert.equal(rows[0].host.path, pin.path);
  }
});

test('audited TRAIN and retention mapped-file schemas require exact roots, flags and counts', () => {
  const pairs = ['6_v7', '6_v8', '8_v9', '9_v11', '11_v12', '11_v13'];
  const kinds = [...pairs.map(pair => `cpb_v${pair}_TRAIN_independent_audit_mirror_v1`),
    ...[7, 8, 9, 11].map(version => `clean_causal_v${version}_retention_complete_mirror_v1`)];
  const member = { relative: 'run-001/audit.json', container: '/opt/cuwacunu/research/audit/attempt-001/run-001/audit.json', host: bind('output/engineering/audit/container-mirror/run-001/audit.json', '{}') };
  for (const kind of kinds) {
    const receipt = { kind, status: 'passed', all_hashes_exact: true, file_count: 1, container_root: '/opt/cuwacunu/research/audit/attempt-001',
      host_root: 'output/engineering/audit/container-mirror', files: [member] };
    assert.equal(mirrorMappings(receipt, bind('mirror.json', '{}')).length, 1);
    assert.throws(() => mirrorMappings({ ...receipt, all_hashes_exact: false }, bind('mirror.json', '{}')), /exact hash flag/);
    assert.throws(() => mirrorMappings({ ...receipt, file_count: 2 }, bind('mirror.json', '{}')), /member count mismatch/);
    assert.throws(() => mirrorMappings({ ...receipt, files: [{ ...member, relative: undefined }] }, bind('mirror.json', '{}')), /relative member is mandatory/);
    assert.throws(() => mirrorMappings({ ...receipt, host_root: 'output/unrelated' }, bind('mirror.json', '{}')), /declared host root mismatch/);
    assert.throws(() => mirrorMappings({ ...receipt, files: [{ ...member, relative: 'other.json' }] }, bind('mirror.json', '{}')), /relative member mismatch/);
  }
});

test('TRAIN relative inventories use only their declared Windows host and container roots', () => {
  for (const pair of ['6_v7', '6_v8', '8_v9', '9_v11', '11_v12', '11_v13']) {
    const receipt = { kind: `cpb_v${pair}_TRAIN_diagnosis_exact_mirror_v1`, status: 'passed',
      all_sizes_sha256_exact: true, container_before_after_exact: true,
      container_root: '/opt/cuwacunu/research/diagnosis/attempt-001',
      host_root: 'C:\\Work\\cuwacunu\\cuwacunu_embedding\\output\\engineering\\diagnosis\\attempt-001\\mirror-003',
      files: [bind('source/audit.py', 'preserved source')] };
    const mapping = mirrorMappings(receipt, bind('mirror.json', '{}'))[0];
    assert.equal(mapping.original.path, '/opt/cuwacunu/research/diagnosis/attempt-001/source/audit.py');
    assert.equal(mapping.host.path, 'output/engineering/diagnosis/attempt-001/mirror-003/source/audit.py');
    assert.equal(mapping.historical_spellings[0].path, 'source/audit.py');
    assert.throws(() => mirrorMappings({ ...receipt, host_root: 'C:\\Work\\another_repository' }, bind('mirror.json', '{}')), /relative locator|host root/);
    assert.throws(() => mirrorMappings({ ...receipt, container_before_after_exact: false }, bind('mirror.json', '{}')), /both exact preservation flags/);
  }
});

test('V6 quality audit and V5/V6 TRAIN audit require their exact local-root relationship', () => {
  for (const kind of ['causal_fresh_v6_quality_audit_attempt002_mirror_v1', 'cpb_v5_v6_independent_TRAIN_audit_exact_mirror_v1']) {
    const receipt = { kind, status: 'passed', all_container_bytes_preserved: true,
      container_root: '/opt/cuwacunu/research/audit/attempt-002', local_root: 'output/engineering/audit/attempt-002/container-mirror',
      files: [{ ...bind('output/engineering/audit/attempt-002/container-mirror/audit.json', '{}'), relative: 'audit.json', container_path: '/opt/cuwacunu/research/audit/attempt-002/audit.json' }] };
    assert.equal(mirrorMappings(receipt, bind('mirror.json', '{}')).length, 1);
    assert.throws(() => mirrorMappings({ ...receipt, local_root: 'output/unrelated' }, bind('mirror.json', '{}')), /declared local root mismatch/);
    assert.throws(() => mirrorMappings({ ...receipt, files: [{ ...receipt.files[0], relative: undefined }] }, bind('mirror.json', '{}')), /relative member is mandatory/);
    assert.throws(() => mirrorMappings({ ...receipt, files: [{ ...receipt.files[0], relative: 'other.json' }] }, bind('mirror.json', '{}')), /relative member mismatch/);
  }
});

test('durable V4/V5 and V9-V14 byte-copy adapters bind the exact declared pair', () => {
  const source = bind('output/runs/example/result.json', '{}');
  const destination = bind('doc/results/example/result.json', '{}');
  const forms = [
    { kind: 'causal_fresh_v4_durable_saved_evidence_mirror_v1', files: [{ ...destination, source: `C:\\Work\\cuwacunu\\cuwacunu_embedding\\${source.path}` }] },
    { kind: 'causal_fresh_v5_durable_evidence_mirror_v1', files: [{ source: source.path, destination: destination.path, bytes: source.bytes, sha256: source.sha256 }] },
    { kind: 'cpb_v9_curated_durable_JSON_bytecopy_mirror_v1', file_count: 1, files: [{ source, destination, ...destination, byte_exact: true }] },
    ...[10, 11, 12, 13, 14].map(version => ({ kind: `cpb_v${version}_curated_durable_JSON_bytecopy_mirror_v1`, file_count: 1, copied_files: [{ source, destination }] })),
  ];
  for (const form of forms) {
    const mappings = byteCopyMappings({ ...form, status: 'passed' }, bind('mirror.json', '{}'));
    assert.equal(mappings.length, 1);
    assert.equal(mappings[0].destination.path, destination.path);
    assert.throws(() => byteCopyMappings({ ...form, status: 'passed' }, bind('mirror.json', '{}'), { repository: 'cuwacunu_dev' }), /passed embedding/);
  }
  assert.throws(() => byteCopyMappings({ kind: 'cpb_v14_curated_durable_JSON_bytecopy_mirror_v1', status: 'passed', copied_files: [{ source, destination: { ...destination, sha256: '0'.repeat(64) } }] }, bind('mirror.json', '{}')), /equal bytes\/hash/);
  for (const version of [10, 11, 12, 13, 14]) {
    for (const name of ['prospective-freeze.json', 'protocol-preflight.json']) {
      const row = { source: { ...source, path: `doc/causal_fresh_v${version}/${name}` }, destination: { ...destination, path: `doc/results/causal_fresh_v${version}/${name}` } };
      assert.equal(byteCopyMappings({ kind: `cpb_v${version}_curated_durable_JSON_bytecopy_mirror_v1`, status: 'passed', copied_files: [row] }, bind('mirror.json', '{}')).length, 1);
      assert.throws(() => byteCopyMappings({ kind: `cpb_v${version}_curated_durable_JSON_bytecopy_mirror_v1`, status: 'passed', copied_files: [{ ...row, destination: { ...row.destination, path: 'doc/results/unrelated.json' } }] }, bind('mirror.json', '{}')), /namespace mismatch/);
    }
  }
});

test('declared TRAIN roots resolve a frozen bare member without global path search', async () => fixture(async root => {
  const member = await row(root, 'saved/execution-stage.json', { status: 'preserved' }, 'output/engineering/diagnosis/attempt-001/mirror-003/execution-stage.json');
  const bare = { ...member.original, path: 'execution-stage.json' };
  const receipt = await row(root, 'diagnosis-mirror.json', { kind: 'cpb_v6_v8_TRAIN_diagnosis_exact_mirror_v1', status: 'passed',
    all_sizes_sha256_exact: true, container_before_after_exact: true,
    container_root: '/opt/cuwacunu/research/diagnosis/attempt-001', host_root: 'C:\\Work\\cuwacunu\\cuwacunu_embedding\\output\\engineering\\diagnosis\\attempt-001\\mirror-003', files: [bare] }, 'doc/diagnosis-mirror.json');
  const result = await run(root, [member, receipt], [failure(bare)]);
  assert.equal(result.failures.length, 0);
  assert.equal(result.base_failure_accounting[0].resolved, true);
}));

test('durable byte-copy evidence repairs a missing raw pin through its exact saved destination', async () => fixture(async root => {
  const destination = await row(root, 'curated/result.json', { score: 1 }, 'doc/results/example/result.json');
  const raw = { ...destination.original, path: 'output/runs/example/result.json' };
  const receipt = await row(root, 'durable-mirror.json', { kind: 'cpb_v14_curated_durable_JSON_bytecopy_mirror_v1', status: 'passed', file_count: 1,
    copied_files: [{ source: raw, destination: destination.original }] }, 'doc/durable-evidence-mirror.json');
  const result = await run(root, [destination, receipt], [failure(raw)]);
  assert.equal(result.failures.length, 0);
  assert.equal(result.aliases.find(entry => entry.original.path === raw.path).current_path, 'curated/result.json');
}));

test('base canonical metadata checks are reused only for aliases sharing one exact physical identity', async () => fixture(async root => {
  const artifact = await row(root, 'saved.pt', 'preserved bytes', 'output/saved.pt');
  const qualified = { ...artifact, original: { ...artifact.original, path: '/opt/cuwacunu/research/example/saved.pt' } };
  const result = await run(root, [artifact, qualified]);
  assert.equal(result.failures.length, 0);
  assert.equal(result.base_unique_locator_metadata_checks, 1);
  assert.equal(result.base_duplicate_locator_metadata_checks_reused, 1);
  assert.equal(result.reused_aliases, 2);
  assert.equal(result.new_physical_hash_reads, 0);
  assert.equal(result.aliases[0].locator_metadata_checked_utc, result.aliases[1].locator_metadata_checked_utc);
  const contradictory = { ...qualified, original: { ...qualified.original, sha256: '0'.repeat(64) }, observed: { ...qualified.observed, sha256: '0'.repeat(64) } };
  await assert.rejects(run(root, [artifact, contradictory]), /Contradictory observations at one physical locator/);
}));

test('bounded base inspection covers every unique path and refuses the entire base if one path fails', async () => fixture(async root => {
  const entries = [];
  const plans = [];
  for (let i = 0; i < 9; i++) {
    const artifact = await row(root, `saved-${i}.pt`, `preserved-${i}`, `output/saved-${i}.pt`);
    entries.push(artifact);
    plans.push({ current_path: artifact.current_path, expected: artifact.original });
    for (let copy = 0; copy < 3; copy++) entries.push({ ...artifact,
      original: { ...artifact.original, path: `/opt/cuwacunu/research/copy-${copy}/saved-${i}.pt` } });
  }
  const passed = await run(root, entries);
  assert.equal(passed.base_unique_locator_metadata_checks, 9);
  assert.equal(passed.base_duplicate_locator_metadata_checks_reused, 27);
  assert.deepEqual(passed.aliases.map(entry => entry.original.path), entries.map(entry => entry.original.path));
  await writeFile(path.join(root, 'saved-0.pt'), 'changed length');
  await assert.rejects(inspectBaseLocators(root, plans), error => {
    assert.equal(error.locator_checks.length, 9);
    assert.deepEqual(error.locator_checks.map(check => check.current_path), plans.map(plan => plan.current_path));
    assert.equal(error.locator_checks.filter(check => !check.passed).length, 1);
    assert.equal(error.locator_checks.filter(check => check.passed).length, 8);
    assert.ok(error.locator_checks.every(check => typeof check.checked_utc === 'string'));
    return true;
  });
  await assert.rejects(run(root, entries), /Base canonical locator checks failed/);
}));

function boundedScope(controls, required, failures, qualified = new Map()) {
  const control_documents = controls.map(entry => ({ role: 'fixture-control', original: entry.original,
    current_path: entry.current_path, observed: entry.observed, selectors: ['fixture exact declared members'] }));
  const source = controls[0];
  return { kind: 'cpb_v14_typed_bounded_scope_v1', control_documents, mapping_documents: [],
    required_pins: required.map((row, index) => ({ original: row.original, known_locators: row.known_locators ?? [],
      required_by: [{ manifest: { repository: 'Mityane-geometrics', path: source.current_path, ...source.observed }, pointer: `/members/${index}` }] })),
    historical_failure_classification: failures.map((old, index) => {
      const record = qualified.has(index) ? qualified.get(index) : old.identity;
      return { original_failure_index: index, original_identity: old.identity, record_qualified_identity: record,
        required_in_scope: record !== null && required.some(row => row.original.repository === (record.repository ?? 'cuwacunu_embedding') && row.original.path === record.path && row.original.sha256 === record.sha256 && row.original.bytes === record.bytes),
        disposition: qualified.has(index) ? 'record_schema_overinterpretation' : required.some(row => row.original.path === record.path) ? 'required_current_scope' : 'outside_declared_traversal', prior_failure_retained: true };
    }) };
}

test('bounded scope verifies declared whole members while terminal JSON references remain explicitly unverified', async () => fixture(async root => {
  const historical = { repository: 'cuwacunu_embedding', path: '/old/unrelated/preservation-after.json', bytes: 20955220255, sha256: 'a'.repeat(64) };
  const terminal = await row(root, 'opaque.json', { historical_reference: historical }, 'archive/historical/artifact-baseline.json');
  const control = await row(root, 'control.json', { members: [terminal.original] }, 'doc/typed-inventory.json');
  const failures = [failure(historical)];
  const scope = boundedScope([control], [{ original: terminal.original }], failures);
  const result = await run(root, [control, terminal], failures, { requiredScope: scope, requiredScopeBinding: bind('scope.json', '{}') });
  assert.equal(result.bounded_scope_verified, true);
  assert.equal(result.metadata_documents_rewalked, 1);
  assert.equal(result.required_pin_count, 1);
  assert.equal(result.required_pins_verified, 1);
  assert.equal(result.base_failure_accounting[0].resolved, false);
  assert.equal(result.base_failure_accounting[0].disposition, 'outside_declared_recursive_reverification');
  assert.equal(result.outside_declared_recursive_reverification_count, 1);
  assert.equal(result.whole_reference_graph_verified, false);
  assert.equal(result.all_historical_refs_resolved, false);
  assert.equal(result.aliases.some(entry => entry.original.path === historical.path), false);
  assert.deepEqual(result.original_failures, failures);
}));

test('a missing declared required identity fails bounded scope', async () => fixture(async root => {
  const original = { repository: 'cuwacunu_embedding', ...bind('output/missing.pt', 'missing') };
  const control = await row(root, 'control.json', { members: [original] }, 'doc/typed-inventory.json');
  const failures = [failure(original)];
  const scope = boundedScope([control], [{ original, known_locators: ['does-not-exist.pt'] }], failures);
  const result = await run(root, [control], failures, { requiredScope: scope, requiredScopeBinding: bind('scope.json', '{}') });
  assert.equal(result.bounded_scope_verified, false);
  assert.equal(result.required_pins_verified, 0);
  assert.equal(result.base_failure_accounting[0].required, true);
  assert.equal(result.base_failure_accounting[0].disposition, 'required_identity_unresolved');
  assert.ok(result.failures.length > 0);
}));

test('corrected record context does not relabel the original wrong-repository failure as resolved', async () => fixture(async root => {
  const external = await row(root, 'portable.txt', 'external bytes', 'doc/external.txt', 'cuwacunu_dev');
  const wrong = { ...external.original, repository: 'cuwacunu_embedding' };
  const control = await row(root, 'control.json', { members: [external.original] }, 'doc/typed-inventory.json');
  const failures = [failure(wrong)];
  const scope = boundedScope([control], [{ original: external.original, known_locators: ['portable.txt'] }], failures, new Map([[0, external.original]]));
  const result = await run(root, [control], failures, { requiredScope: scope, requiredScopeBinding: bind('scope.json', '{}') });
  assert.equal(result.bounded_scope_verified, true);
  assert.equal(result.base_failure_accounting[0].resolved, false);
  assert.equal(result.base_failure_accounting[0].required, false);
  assert.equal(result.base_failure_accounting[0].disposition, 'record_schema_overinterpretation');
  assert.equal(result.base_failure_accounting[0].record_qualified_required, true);
  assert.equal(result.base_failure_accounting[0].record_qualified_verified, true);
  assert.equal(result.record_schema_overinterpretation_count, 1);
  assert.equal(result.outside_declared_recursive_reverification_count, 0);
  const collisionScope = boundedScope([control], [
    { original: external.original, known_locators: ['portable.txt'] },
    { original: wrong, known_locators: ['portable.txt'] },
  ], failures, new Map([[0, external.original]]));
  const collision = await run(root, [control], failures, { requiredScope: collisionScope, requiredScopeBinding: bind('scope.json', '{}') });
  assert.equal(collision.bounded_scope_verified, true);
  assert.equal(collision.required_pins_verified, 2);
  assert.equal(collision.base_failure_accounting[0].required, false);
  assert.equal(collision.base_failure_accounting[0].resolved, false);
  assert.equal(collision.base_failure_accounting[0].disposition, 'record_schema_overinterpretation');
  assert.equal(collision.base_failure_accounting[0].record_qualified_verified, true);
}));

test('an aggregate schema object retains a null qualified identity and is never a required file', async () => fixture(async root => {
  const aggregate = { repository: 'cuwacunu_embedding', path: '/old/unrelated/preservation-after.json', bytes: 20955220255, sha256: 'a'.repeat(64) };
  const control = await row(root, 'control.json', { passed_after_audit: aggregate, members: [] }, 'doc/typed-inventory.json');
  const failures = [failure(aggregate)];
  const scope = boundedScope([control], [], failures, new Map([[0, null]]));
  const result = await run(root, [control], failures, { requiredScope: scope, requiredScopeBinding: bind('scope.json', '{}') });
  assert.equal(result.bounded_scope_verified, true);
  assert.equal(result.metadata_documents_rewalked, 1);
  assert.equal(result.base_failure_accounting[0].record_qualified_identity, null);
  assert.equal(result.base_failure_accounting[0].record_qualified_required, false);
  assert.equal(result.base_failure_accounting[0].record_qualified_verified, false);
  assert.equal(result.base_failure_accounting[0].required, false);
  assert.equal(result.base_failure_accounting[0].resolved, false);
  assert.equal(result.base_failure_accounting[0].disposition, 'record_schema_overinterpretation');
  assert.equal(result.new_non_metadata_hash_reads, 0);
  assert.equal(result.aliases.some(entry => entry.original.path === aggregate.path), false);
  const missingField = structuredClone(scope);
  delete missingField.historical_failure_classification[0].record_qualified_identity;
  assert.throws(() => validateScopeDeclaration(missingField, failures), /differs from preserved failed evidence/);
}));

async function capsuleSelfFixture(root, mismatch = false) {
  const expectedRoot = '/opt/cuwacunu/research/causal-fresh-v14/attempt-001';
  const capsule = await row(root, 'source-manifest.json', { kind: 'causal_fresh_v14_source_capsule_v1',
    container_root: mismatch ? '/opt/cuwacunu/research/unrelated/attempt-001' : expectedRoot, files: [] }, 'output/engineering/causal-fresh-v14/attempt-001/source-manifest.json');
  const self = { ...capsule.original, path: `${expectedRoot}/source/CAPSULE_MANIFEST.json` };
  const build = await row(root, 'build.json', { kind: 'causal_fresh_v14_build_admission_v1', source_manifest: self }, 'output/engineering/causal-fresh-v14/attempt-001/build-admission.json');
  const failures = [failure(self)];
  const scope = boundedScope([build, capsule], [{ original: self }], failures);
  return run(root, [build, capsule], failures, { requiredScope: scope, requiredScopeBinding: bind('scope.json', '{}') });
}
test('V14 capsule self alias requires both the exact build pin and matching declared capsule root', async () => fixture(async root => {
  const result = await capsuleSelfFixture(root);
  assert.equal(result.bounded_scope_verified, true);
  assert.equal(result.base_failure_accounting[0].resolved, true);
  const self = result.aliases.find(entry => entry.original.path.endsWith('/source/CAPSULE_MANIFEST.json'));
  assert.equal(self.provenance.length, 2);
  await fixture(async other => {
    const mismatch = await capsuleSelfFixture(other, true);
    assert.equal(mismatch.bounded_scope_verified, false);
    assert.ok(mismatch.failures.some(row => /matching source-capsule root/.test(row.error)));
  });
}));

test('bounded scope rejects unbound provenance, incomplete failure accounting and actual unbounded execution', async () => fixture(async root => {
  const control = await row(root, 'control.json', { members: [] }, 'doc/typed-inventory.json');
  const target = { repository: 'cuwacunu_embedding', ...bind('output/a.pt', 'a') };
  const failures = [failure(target)];
  const scope = boundedScope([control], [{ original: target }], failures);
  const unbound = structuredClone(scope);
  unbound.required_pins[0].required_by[0].manifest.path = 'unlisted-control.json';
  assert.throws(() => validateScopeDeclaration(unbound, failures), /explicit whole-byte-bound control/);
  assert.throws(() => validateScopeDeclaration({ ...scope, historical_failure_classification: [] }, failures), /Every original failure/);
  await assert.rejects(repairAliasInventory({ root, inputs: { kind: 'cpb_v14_alias_repair_inputs_v1' } }), /typed bounded scope and independent passed scope approval/);
}));
