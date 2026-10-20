// Artificial SDK/evidence fixtures only. Never create or repair a real SDK.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { KIND, VERSION, ABI, POINTER, OLD_PROFILE, NEW_PROFILE, parse, tree,
  hashFile, verifyComplete, safeRelative } from '../verify-libtorch.mjs';

const sha = raw => createHash('sha256').update(raw).digest('hex');
const encode = object => `${JSON.stringify(object, null, 2)}\n`;

export function assertFixtureContainer() {
  assert.equal(process.platform, 'linux', 'fixtures run only inside the Linux container');
  assert.ok(fs.statSync('/.dockerenv').isFile());
}

// Independently assemble proof bytes instead of using verifier inventory code.
// This catches ordering, mode, byte-count, digest and evidence-format mistakes.
function fixtureInventory(root) {
  const rows = [];
  const visit = directory => {
    for (const name of fs.readdirSync(directory).sort()) {
      const filename = path.join(directory, name);
      const relative = path.relative(root, filename);
      const info = fs.lstatSync(filename);
      if (info.isSymbolicLink()) rows.push({ path: relative, kind: 'symlink', target: fs.readlinkSync(filename) });
      else if (info.isDirectory()) {
        rows.push({ path: relative, kind: 'directory', mode: info.mode & 0o7777 }); visit(filename);
      } else rows.push({ path: relative, kind: 'file', bytes: info.size, mode: info.mode & 0o7777,
        sha256: sha(fs.readFileSync(filename)) });
    }
  };
  visit(root);
  return rows.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
}

export function fixtureBundle(root) {
  fs.mkdirSync(root);
  const files = {
    'build-version': `${VERSION}\n`,
    'share/cmake/Torch/TorchConfig.cmake': 'set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")\n',
    'share/cmake/Torch/TorchConfigVersion.cmake': 'set(PACKAGE_VERSION "2.6.0")\n',
    'include/torch/csrc/api/include/torch/torch.h': '// artificial header\n',
  };
  for (const name of ['libtorch.so', 'libtorch_cpu.so', 'libtorch_cuda.so', 'libc10.so', 'libc10_cuda.so']) {
    files[`lib/${name}`] = Buffer.concat([Buffer.from([0x7f, 0x45, 0x4c, 0x46]), Buffer.from(` artificial fixture ${name}`)]);
  }
  for (const [name, contents] of Object.entries(files)) {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, contents);
    fs.chmodSync(filename, 0o644);
  }
  fs.symlinkSync('libtorch.so', path.join(root, 'lib/internal.so'));
}

export function writeEvidence(fixture) {
  const { source, target, setup, proofPath, inventoryPath, pointer } = fixture;
  const entries = fixtureInventory(target);
  const metadataNames = ['build-version', 'share/cmake/Torch/TorchConfig.cmake', 'share/cmake/Torch/TorchConfigVersion.cmake'];
  const inventory = { artifact_kind: KIND, source, target, entries };
  fs.writeFileSync(inventoryPath, encode(inventory));
  const files = entries.filter(row => row.kind === 'file');
  const proof = { artifact_kind: KIND, status: 'complete', source, target, build_version: VERSION, cxx11_abi: ABI,
    inventory_sha256: sha(fs.readFileSync(inventoryPath)), regular_files: files.length,
    regular_file_bytes: files.reduce((total, row) => total + row.bytes, 0),
    bundle_identity: { build_version: VERSION, cxx11_abi: ABI,
      metadata_sha256: Object.fromEntries(metadataNames.map(name => [name, sha(fs.readFileSync(path.join(target, name)))])) } };
  fs.writeFileSync(proofPath, encode(proof));
  fs.writeFileSync(pointer, encode({ artifact_kind: KIND, target, proof_path: proofPath, proof_sha256: sha(fs.readFileSync(proofPath)) }));
  assert.equal(path.dirname(path.dirname(proofPath)), setup);
}

export function createFixture() {
  assertFixtureContainer();
  const parent = '/opt/mityane-geometrics/setup/libtorch-fixtures';
  fs.mkdirSync(parent, { recursive: true });
  const base = fs.mkdtempSync(`${parent}/verify-`);
  const fixture = { base, source: `${base}/source`, target: `${base}/installed`, setup: `${base}/setup`, profile: `${base}/embedding.sh` };
  fixtureBundle(fixture.source);
  fs.cpSync(fixture.source, fixture.target, { recursive: true, verbatimSymlinks: true });
  const evidence = `${fixture.setup}/sdk-artificial`;
  fs.mkdirSync(evidence, { recursive: true });
  Object.assign(fixture, { proofPath: `${evidence}/proof.json`, inventoryPath: `${evidence}/inventory.json`, pointer: `${fixture.setup}/${POINTER}` });
  fs.writeFileSync(fixture.profile, NEW_PROFILE);
  fs.chmodSync(fixture.profile, 0o644);
  writeEvidence(fixture);
  return fixture;
}

// Include bytes, type, permissions and change timestamps, excluding access time
// which a read-only verifier can legitimately update on an ordinary filesystem.
export function snapshot(root) {
  const rows = [];
  function visit(filename) {
    const info = fs.lstatSync(filename, { bigint: true });
    const row = { path: path.relative(root, filename), mode: info.mode, size: info.size, mtime: info.mtimeNs, ctime: info.ctimeNs };
    if (info.isSymbolicLink()) row.target = fs.readlinkSync(filename);
    else if (info.isFile()) row.sha256 = sha(fs.readFileSync(filename));
    rows.push(row);
    if (info.isDirectory()) for (const name of fs.readdirSync(filename).sort()) visit(path.join(filename, name));
  }
  visit(root);
  return rows;
}

export function withFixture(call) {
  const fixture = createFixture();
  try { return call(fixture); }
  finally {
    // Only the exact uniquely created artificial fixture is removed.
    assert.ok(fixture.base.startsWith('/opt/mityane-geometrics/setup/libtorch-fixtures/verify-'));
    fs.rmSync(fixture.base, { recursive: true });
  }
}

export function check(fixture, options = {}) {
  return verifyComplete(fixture.source, fixture.target, fixture.setup, { profile: fixture.profile, ...options });
}

function rejectUnchanged(fixture, call, match) {
  const before = snapshot(fixture.base);
  assert.throws(call, match);
  assert.deepEqual(snapshot(fixture.base), before, 'rejection preserves every artificial SDK/evidence/profile byte and metadata');
}

export const fixtureCases = [
  ['valid proof is reusable without writes', fixture => {
    const before = snapshot(fixture.base);
    const result = check(fixture);
    assert.equal(result.status, 'verified'); assert.equal(result.sdk_mutations, 0);
    assert.equal(result.regular_files, 9); assert.equal(result.runtime_profile.state, 'internal');
    assert.deepEqual(check(fixture), result);
    assert.deepEqual(snapshot(fixture.base), before);
  }],
  ['changed source is compared only when requested', fixture => {
    fs.appendFileSync(`${fixture.source}/build-version`, 'changed');
    rejectUnchanged(fixture, () => check(fixture));
    assert.equal(check(fixture, { compareSource: false }).status, 'verified');
  }],
  ['missing historical source is accepted', fixture => {
    fs.renameSync(fixture.source, `${fixture.base}/retained-source`);
    assert.equal(check(fixture).status, 'verified');
  }],
  ['same-size SDK byte tampering is rejected', fixture => {
    const filename = `${fixture.target}/lib/libtorch_cpu.so`;
    const raw = fs.readFileSync(filename); raw[raw.length - 1] ^= 1; fs.writeFileSync(filename, raw);
    rejectUnchanged(fixture, () => check(fixture), /SHA unchanged/);
  }],
  ['extra SDK member is rejected', fixture => {
    fs.writeFileSync(`${fixture.target}/extra`, 'retained');
    rejectUnchanged(fixture, () => check(fixture), /paths\/types\/sizes\/modes/);
  }],
  ['SDK permission tampering is rejected', fixture => {
    fs.chmodSync(`${fixture.target}/lib/libtorch.so`, 0o755);
    rejectUnchanged(fixture, () => check(fixture), /paths\/types\/sizes\/modes/);
  }],
  ['privileged SDK mode is rejected', fixture => {
    fs.chmodSync(`${fixture.target}/lib/libtorch.so`, 0o4644);
    rejectUnchanged(fixture, () => check(fixture), /privileged SDK mode/);
  }],
  ['missing pointer leaves partial target intact', fixture => {
    fs.unlinkSync(fixture.pointer);
    rejectUnchanged(fixture, () => check(fixture), /no completion pointer/);
  }],
  ['proof bytes are hash bound', fixture => {
    fs.appendFileSync(fixture.proofPath, ' ');
    rejectUnchanged(fixture, () => check(fixture), /exact completed SDK proof/);
  }],
  ['duplicate proof keys are rejected even with a matching pointer digest', fixture => {
    const raw = fs.readFileSync(fixture.proofPath, 'utf8').replace('{', '{"status":"complete",');
    fs.writeFileSync(fixture.proofPath, raw);
    const pointer = JSON.parse(fs.readFileSync(fixture.pointer));
    pointer.proof_sha256 = sha(raw); fs.writeFileSync(fixture.pointer, encode(pointer));
    rejectUnchanged(fixture, () => check(fixture), /duplicate proof JSON key/);
  }],
  ['pointer cannot redirect proof outside its exclusive setup record', fixture => {
    const pointer = JSON.parse(fs.readFileSync(fixture.pointer));
    pointer.proof_path = `${fixture.base}/proof.json`;
    fs.copyFileSync(fixture.proofPath, pointer.proof_path); fs.writeFileSync(fixture.pointer, encode(pointer));
    rejectUnchanged(fixture, () => check(fixture), /proof outside SDK/);
  }],
  ['inventory bytes are hash bound', fixture => {
    fs.appendFileSync(fixture.inventoryPath, ' ');
    rejectUnchanged(fixture, () => check(fixture), /exact completed file inventory/);
  }],
  ['proof symlink is rejected', fixture => {
    const original = `${fixture.proofPath}.original`;
    fs.renameSync(fixture.proofPath, original); fs.symlinkSync(path.basename(original), fixture.proofPath);
    rejectUnchanged(fixture, () => check(fixture), /canonical regular evidence/);
  }],
  ['evidence hardlink is rejected', fixture => {
    fs.linkSync(fixture.inventoryPath, `${fixture.base}/retained-inventory`);
    rejectUnchanged(fixture, () => check(fixture), /canonical regular evidence/);
  }],
  ['redirected evidence directory is rejected', fixture => {
    const original = `${fixture.setup}/sdk-artificial`;
    fs.renameSync(original, `${fixture.setup}/retained`); fs.symlinkSync('retained', original);
    rejectUnchanged(fixture, () => check(fixture), /canonical regular evidence|redirected directory/);
  }],
  ['escaping SDK symlink is rejected', fixture => {
    fs.symlinkSync('../../embedding.sh', `${fixture.target}/lib/escape`);
    rejectUnchanged(fixture, () => check(fixture), /symlink escape/);
  }],
  ['absolute SDK symlink is rejected', fixture => {
    fs.symlinkSync(`${fixture.target}/lib/libtorch.so`, `${fixture.target}/lib/absolute`);
    rejectUnchanged(fixture, () => check(fixture), /relative and internal/);
  }],
  ['broken SDK symlink is rejected', fixture => {
    fs.symlinkSync('absent', `${fixture.target}/lib/broken`);
    rejectUnchanged(fixture, () => check(fixture));
  }],
  ['SDK root symlink target is rejected', fixture => {
    fs.symlinkSync('..', `${fixture.target}/lib/root`);
    rejectUnchanged(fixture, () => check(fixture), /invalid root target/);
  }],
  ['wrong version is rejected even with a matching inventory', fixture => {
    fs.writeFileSync(`${fixture.target}/build-version`, '2.7.0\n'); writeEvidence(fixture);
    rejectUnchanged(fixture, () => check(fixture), /expected LibTorch/);
  }],
  ['wrong ABI is rejected even with a matching inventory', fixture => {
    fs.writeFileSync(`${fixture.target}/share/cmake/Torch/TorchConfig.cmake`, 'set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")\n');
    writeEvidence(fixture);
    rejectUnchanged(fixture, () => check(fixture), /exact Linux C\+\+11 ABI/);
  }],
  ['non-ELF library is rejected even with a matching inventory', fixture => {
    fs.writeFileSync(`${fixture.target}/lib/libtorch.so`, 'fake'); writeEvidence(fixture);
    rejectUnchanged(fixture, () => check(fixture), /Linux ELF/);
  }],
  ['legacy profile is preserved and rejected', fixture => {
    fs.writeFileSync(fixture.profile, OLD_PROFILE);
    rejectUnchanged(fixture, () => check(fixture), /legacy SDK runtime profile/);
  }],
  ['custom profile is preserved and rejected', fixture => {
    fs.writeFileSync(fixture.profile, '# custom profile\n');
    rejectUnchanged(fixture, () => check(fixture), /unknown\/custom runtime profile/);
  }],
  ['absent profile retains the inherited optional-profile semantics', fixture => {
    fs.unlinkSync(fixture.profile);
    assert.deepEqual(check(fixture).runtime_profile, { state: 'absent', path: fixture.profile });
  }],
  ['file changed after tree admission is rejected', fixture => {
    const row = tree(fixture.target).find(item => item.path === 'build-version');
    fs.appendFileSync(`${fixture.target}/build-version`, 'changed');
    rejectUnchanged(fixture, () => hashFile(`${fixture.target}/build-version`, row._signature), /changed after path admission/);
  }],
  ['file redirected after tree admission is rejected without following it', fixture => {
    const row = tree(fixture.target).find(item => item.path === 'build-version');
    fs.renameSync(`${fixture.target}/build-version`, `${fixture.target}/retained-version`);
    fs.symlinkSync('retained-version', `${fixture.target}/build-version`);
    rejectUnchanged(fixture, () => hashFile(`${fixture.target}/build-version`, row._signature), /ELOOP/);
  }],
];

export function checkParsing() {
  assert.deepEqual(parse('{"a":[1,true,null,{"b":"c"}]}'), { a: [1, true, null, { b: 'c' }] });
  for (const raw of ['{"a":1,"a":2}', '{"a":{"x":1,"\\u0078":2}}', '{"a":NaN}', '{"a":Infinity}',
    '{"a":1e999}', '{"a":9007199254740993}', '{"a":1} trailing', '{"a":01}', '[1,]', '{"a":1,}', '"bad\nstring"']) {
    assert.throws(() => parse(raw), undefined, raw);
  }
  const object = parse('{"__proto__":{"polluted":true}}');
  assert.ok(Object.hasOwn(object, '__proto__')); assert.equal({}.polluted, undefined);
  for (const name of ['', '.', '..', '../escape', '/absolute', 'lib//x', 'lib/./x', 'lib/../x', 'lib\\x', 'lib\nx']) {
    assert.throws(() => safeRelative(name));
  }
}

export function selfTest() {
  assertFixtureContainer();
  for (const [, run] of fixtureCases) withFixture(run);
  checkParsing();
  return { status: 'passed', fixture_cases: fixtureCases.length, negative_cases: 24,
    real_SDK_reads: 0, real_SDK_mutations: 0, downloads: 0, build_or_SDK_execution: false, fixture_only: true };
}
