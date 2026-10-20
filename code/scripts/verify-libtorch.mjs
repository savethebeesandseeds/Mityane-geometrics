#!/usr/bin/env node
/**
 * Read-only verification of the inherited pinned Linux LibTorch SDK.
 * The /embedding source path is an immutable proof identity, never a source to
 * install. No verifier path downloads, copies, repairs or executes SDK content.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { fileURLToPath } from 'node:url';

export const VERSION = '2.6.0+cu124';
export const ABI = 1;
export const SOURCE = '/embedding/.external/libtorch';
export const TARGET = '/opt/cuwacunu_embedding/libtorch';
export const SETUP = '/opt/cuwacunu_embedding/setup';
export const POINTER = 'libtorch-installed.json';
export const PROFILE = '/etc/profile.d/embedding.sh';
export const KIND = 'cuwacunu_libtorch_installation_v1';
export const OLD_PROFILE = Buffer.from(`export CUDA_VERSION=12.4
export CUDNN_VERSION=9
case ":$PATH:" in
  *:/usr/local/cuda-12.4/bin:*) ;;
  *) export PATH="/usr/local/cuda-12.4/bin:$PATH" ;;
esac
case "\${LD_LIBRARY_PATH:-}" in
  /embedding/.external/libtorch/lib:/usr/local/cuda-12.4/lib64*) ;;
  *) export LD_LIBRARY_PATH="/embedding/.external/libtorch/lib:/usr/local/cuda-12.4/lib64\${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
`);
export const NEW_PROFILE = Buffer.from(OLD_PROFILE.toString().replaceAll(
  '/embedding/.external/libtorch/lib', '/opt/cuwacunu_embedding/libtorch/lib'));
const CHUNK = 8 * 1024 * 1024;

export function requireCondition(ok, message) {
  if (!ok) throw new Error(message);
}

export function exists(filename) {
  try { fs.lstatSync(filename); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

export function digest(raw) { return createHash('sha256').update(raw).digest('hex'); }

// JSON.parse alone silently accepts duplicate object keys. Scan recursively so
// duplicate/escaped duplicate keys and nonfinite numbers cannot enter evidence.
export function parse(raw) {
  const source = typeof raw === 'string' ? raw : new TextDecoder('utf-8', { fatal: true }).decode(raw);
  let index = 0;
  const fail = () => { throw new Error('invalid proof JSON'); };
  const whitespace = () => { while (/[\t\n\r ]/.test(source[index] ?? '\0')) index += 1; };
  const string = () => {
    const start = index++;
    while (index < source.length) {
      if (source[index] === '\\') { index += 2; continue; }
      if (source[index++] === '"') return JSON.parse(source.slice(start, index));
    }
    return fail();
  };
  const value = () => {
    whitespace();
    const next = source[index];
    if (next === '"') return string();
    if (next === '{') {
      const result = {};
      index += 1; whitespace();
      if (source[index] === '}') { index += 1; return result; }
      for (;;) {
        if (source[index] !== '"') return fail();
        const key = string();
        requireCondition(!Object.hasOwn(result, key), 'duplicate proof JSON key');
        whitespace();
        if (source[index++] !== ':') return fail();
        Object.defineProperty(result, key, { value: value(), enumerable: true, writable: true, configurable: true });
        whitespace();
        const delimiter = source[index++];
        if (delimiter === '}') return result;
        if (delimiter !== ',') return fail();
        whitespace();
      }
    }
    if (next === '[') {
      const result = [];
      index += 1; whitespace();
      if (source[index] === ']') { index += 1; return result; }
      for (;;) {
        result.push(value()); whitespace();
        const delimiter = source[index++];
        if (delimiter === ']') return result;
        if (delimiter !== ',') return fail();
      }
    }
    for (const [token, result] of [['true', true], ['false', false], ['null', null]]) {
      if (source.startsWith(token, index)) { index += token.length; return result; }
    }
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(source.slice(index));
    if (!match) return fail();
    index += match[0].length;
    const result = Number(match[0]);
    requireCondition(Number.isFinite(result), 'nonfinite proof JSON');
    // SDK sizes/counts are integral and well below the exact integer boundary.
    requireCondition(!Number.isInteger(result) || Number.isSafeInteger(result), 'inexact proof JSON integer');
    return result;
  };
  const result = value(); whitespace();
  requireCondition(index === source.length, 'invalid trailing proof JSON');
  return result;
}

export function canonicalDirectory(directory) {
  const info = fs.lstatSync(directory);
  requireCondition(path.isAbsolute(directory) && info.isDirectory() && !info.isSymbolicLink() &&
    fs.realpathSync(directory) === directory, `canonical existing directory: ${directory}`);
  for (let parent = path.dirname(directory); ; parent = path.dirname(parent)) {
    requireCondition(!fs.lstatSync(parent).isSymbolicLink(), `redirected directory ancestor: ${parent}`);
    if (parent === path.dirname(parent)) break;
  }
}

export function canonicalFile(filename) {
  const info = fs.lstatSync(filename);
  requireCondition(path.isAbsolute(filename) && info.isFile() && !info.isSymbolicLink() &&
    fs.realpathSync(filename) === filename && info.nlink === 1, `canonical regular evidence file: ${filename}`);
  canonicalDirectory(path.dirname(filename));
}

export function safeRelative(value) {
  requireCondition(typeof value === 'string' && value.length > 0 && !/[\\\n\r]/.test(value) &&
    !path.isAbsolute(value) && value.split('/').every(part => part !== '' && part !== '.' && part !== '..') &&
    path.posix.normalize(value) === value, 'safe SDK relative path');
  return value;
}

function internal(root, member) {
  const relative = path.relative(root, member);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function signature(info) { return [info.dev, info.ino, info.size, info.mtimeNs]; }
function sameSignature(left, right) { return left.every((item, index) => item === right[index]); }
function publicRow(row) { const { _signature, ...result } = row; return result; }
// Python's original ordering uses Unicode code points, rather than UTF-16 units.
function lexical(left, right) {
  const a = Array.from(left, char => char.codePointAt(0));
  const b = Array.from(right, char => char.codePointAt(0));
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
}

export function tree(root) {
  canonicalDirectory(root);
  const rows = [];
  function visit(directory) {
    for (const name of fs.readdirSync(directory).sort(lexical)) {
      const filename = path.join(directory, name);
      const relative = safeRelative(path.relative(root, filename));
      const info = fs.lstatSync(filename, { bigint: true });
      const mode = Number(info.mode & 0o7777n);
      requireCondition(!(mode & 0o6000), 'privileged SDK mode rejected');
      if (info.isSymbolicLink()) {
        const target = fs.readlinkSync(filename);
        requireCondition(target.length > 0 && !path.isAbsolute(target), 'SDK symlink must be relative and internal');
        requireCondition(internal(root, fs.realpathSync(filename)), 'SDK symlink escape or invalid root target');
        rows.push({ path: relative, kind: 'symlink', target });
      } else if (info.isDirectory()) {
        requireCondition(fs.realpathSync(filename) === filename, 'SDK directory redirect');
        rows.push({ path: relative, kind: 'directory', mode });
        visit(filename);
      } else {
        requireCondition(info.isFile(), 'non-regular SDK member rejected');
        requireCondition(info.size <= BigInt(Number.MAX_SAFE_INTEGER), 'exact SDK member size required');
        rows.push({ path: relative, kind: 'file', bytes: Number(info.size), mode, _signature: signature(info) });
      }
    }
  }
  visit(root);
  requireCondition(rows.length > 0 && new Set(rows.map(row => row.path)).size === rows.length, 'nonempty unique SDK tree');
  return rows.sort((a, b) => lexical(a.path, b.path));
}

export function hashFile(filename, admittedSignature) {
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const before = fs.fstatSync(descriptor, { bigint: true });
    requireCondition(before.isFile() && (!admittedSignature || sameSignature(signature(before), admittedSignature)),
      `SDK file changed after path admission: ${filename}`);
    const value = createHash('sha256');
    const buffer = Buffer.allocUnsafe(Math.min(CHUNK, Number(before.size) + 1));
    let size = 0;
    for (;;) {
      const count = fs.readSync(descriptor, buffer, 0, buffer.length, null);
      if (count === 0) break;
      value.update(buffer.subarray(0, count)); size += count;
    }
    requireCondition(sameSignature(signature(fs.fstatSync(descriptor, { bigint: true })), signature(before)),
      'SDK file changed while hashing');
    return { size, sha256: value.digest('hex') };
  } finally { fs.closeSync(descriptor); }
}

export function hashedTree(root, rows = tree(root)) {
  return rows.map(row => {
    const result = publicRow(row);
    if (row.kind === 'file') {
      const hashed = hashFile(path.join(root, safeRelative(row.path)), row._signature);
      requireCondition(hashed.size === row.bytes, 'SDK member size');
      result.sha256 = hashed.sha256;
    }
    return result;
  });
}

export function bundleIdentity(root) {
  const names = ['build-version', 'share/cmake/Torch/TorchConfig.cmake',
    'share/cmake/Torch/TorchConfigVersion.cmake', 'include/torch/csrc/api/include/torch/torch.h'];
  requireCondition(names.every(name => fs.statSync(path.join(root, name)).isFile()),
    'complete LibTorch headers and CMake metadata required');
  const metadata = names.slice(0, 3).map(name => fs.readFileSync(path.join(root, name)));
  requireCondition(metadata[0].toString().trim() === VERSION, `expected LibTorch ${VERSION}`);
  const abis = [...metadata[1].toString().matchAll(/set\(TORCH_CXX_FLAGS\s+"-D_GLIBCXX_USE_CXX11_ABI=([01])"\)/g)].map(match => match[1]);
  requireCondition(isDeepStrictEqual(abis, ['1']), 'exact Linux C++11 ABI=1 CMake declaration');
  const versions = [...metadata[2].toString().matchAll(/set\(PACKAGE_VERSION\s+"([^"]+)"\)/g)].map(match => match[1]);
  requireCondition(isDeepStrictEqual(versions, ['2.6.0']), 'matching Torch CMake package version');
  for (const name of ['libtorch.so', 'libtorch_cpu.so', 'libtorch_cuda.so', 'libc10.so', 'libc10_cuda.so']) {
    const filename = path.join(root, 'lib', name);
    requireCondition(fs.statSync(filename).isFile() && internal(root, fs.realpathSync(filename)), `internal Linux library required: ${name}`);
    const descriptor = fs.openSync(filename, fs.constants.O_RDONLY);
    try {
      const magic = Buffer.alloc(4);
      requireCondition(fs.readSync(descriptor, magic, 0, 4, 0) === 4 && magic.equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46])),
        `Linux ELF library required: ${name}`);
    } finally { fs.closeSync(descriptor); }
  }
  return { build_version: VERSION, cxx11_abi: ABI,
    metadata_sha256: Object.fromEntries(names.slice(0, 3).map((name, index) => [name, digest(metadata[index])])) };
}

export function verifyProfile(profile) {
  if (profile === null) return { state: 'not_requested' };
  canonicalDirectory(path.dirname(profile));
  if (!exists(profile)) return { state: 'absent', path: profile };
  canonicalFile(profile);
  const raw = fs.readFileSync(profile);
  requireCondition(raw.equals(OLD_PROFILE) || raw.equals(NEW_PROFILE), 'unknown/custom runtime profile preserved without changes');
  requireCondition(!raw.equals(OLD_PROFILE), 'legacy SDK runtime profile remains active; restore the recorded environment profile');
  return { state: 'internal', path: profile, sha256: digest(raw), mode: fs.statSync(profile).mode & 0o7777 };
}

export function verifyComplete(source, target, setup, { compareSource = true, profile = null } = {}) {
  canonicalDirectory(target); canonicalDirectory(setup);
  const pointer = path.join(setup, POINTER);
  requireCondition(exists(pointer), 'unproven/partial SDK target preserved; no completion pointer');
  canonicalFile(pointer);
  const reference = parse(fs.readFileSync(pointer));
  requireCondition(reference.artifact_kind === KIND && reference.target === target, 'SDK pointer identity');
  const proofPath = reference.proof_path;
  requireCondition(typeof proofPath === 'string' && path.basename(proofPath) === 'proof.json' &&
    path.dirname(path.dirname(proofPath)) === setup && path.basename(path.dirname(proofPath)).startsWith('sdk-'),
  'proof outside SDK in an exclusive setup record');
  canonicalFile(proofPath);
  const proofRaw = fs.readFileSync(proofPath);
  requireCondition(digest(proofRaw) === reference.proof_sha256, 'exact completed SDK proof');
  const proof = parse(proofRaw);
  requireCondition(proof.artifact_kind === KIND && proof.status === 'complete' && proof.target === target &&
    proof.source === source && proof.build_version === VERSION && proof.cxx11_abi === ABI, 'pinned completed SDK identity');
  const inventoryPath = path.join(path.dirname(proofPath), 'inventory.json');
  canonicalFile(inventoryPath);
  const inventoryRaw = fs.readFileSync(inventoryPath);
  requireCondition(digest(inventoryRaw) === proof.inventory_sha256, 'exact completed file inventory');
  const inventory = parse(inventoryRaw);
  requireCondition(inventory.artifact_kind === KIND && inventory.source === source && inventory.target === target, 'inventory source and target');
  const expected = inventory.entries;
  const targetRows = tree(target);
  requireCondition(Array.isArray(expected) && isDeepStrictEqual(targetRows.map(publicRow), expected.map(row => {
    const { sha256, ...result } = row; return result;
  })), 'complete SDK paths/types/sizes/modes');
  requireCondition(isDeepStrictEqual(hashedTree(target, targetRows), expected), 'every installed SDK member SHA unchanged');
  requireCondition(isDeepStrictEqual(bundleIdentity(target), proof.bundle_identity), 'installed SDK ABI/version metadata');
  if (compareSource && exists(source)) {
    const sourceRows = tree(source);
    requireCondition(isDeepStrictEqual(bundleIdentity(source), proof.bundle_identity) &&
      isDeepStrictEqual(hashedTree(source, sourceRows), expected), 'existing local source remains identical to installed SDK');
  }
  return { status: 'verified', artifact_kind: KIND, target, proof_path: proofPath, proof_sha256: reference.proof_sha256,
    inventory_sha256: proof.inventory_sha256, regular_files: proof.regular_files, regular_file_bytes: proof.regular_file_bytes,
    sdk_mutations: 0, runtime_profile: verifyProfile(profile) };
}

export function mountFor(filename) {
  const mounts = fs.readFileSync('/proc/self/mountinfo', 'utf8').split('\n').filter(Boolean).map(line =>
    line.split(/\s+/)[4].replaceAll('\\040', ' ').replaceAll('\\011', '\t').replaceAll('\\134', '\\'));
  const parents = mounts.filter(mount => filename === mount || internal(mount, filename));
  requireCondition(parents.length > 0, 'container filesystem mount identity');
  return parents.sort((a, b) => b.split('/').length - a.split('/').length)[0];
}

export function containerEnvironment() {
  requireCondition(process.platform === 'linux' && fs.statSync('/.dockerenv').isFile(), 'managed Linux container only');
  requireCondition(process.geteuid() === 0, 'dependency setup requires root inside the managed container');
  requireCondition(fs.realpathSync(fileURLToPath(import.meta.url)) === '/mityane/code/scripts/verify-libtorch.mjs' &&
    fs.statSync('/mityane/container.ps1').isFile(), 'authoritative project container entry only');
  const release = fs.readFileSync('/etc/os-release', 'utf8');
  requireCondition(/^ID=debian$/m.test(release) && /^VERSION_ID="?12"?$/m.test(release), 'approved Debian 12 container');
  requireCondition(mountFor(TARGET) === '/' && mountFor(SETUP) === '/',
    'SDK and evidence must be in the container filesystem, not a workspace bind or other mount');
}

async function main(args) {
  requireCondition(args.every(arg => ['--check', '--self-test', '--help', '-h'].includes(arg)), 'usage: verify-libtorch.mjs [--check | --self-test]');
  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: verify-libtorch.mjs [--check | --self-test]\nRead-only inherited SDK proof, inventory, ABI, version and runtime profile verification.');
    return;
  }
  if (args.includes('--self-test')) {
    requireCondition(!args.includes('--check'), 'fixture mode cannot verify a real SDK');
    const { selfTest } = await import('./tests/verify-libtorch-fixtures.mjs');
    console.log(JSON.stringify(selfTest()));
    return;
  }
  containerEnvironment();
  requireCondition(exists(TARGET), 'copied internal SDK is absent; restore the recorded environment image, not a host SDK');
  console.log(JSON.stringify(verifyComplete(SOURCE, TARGET, SETUP, { compareSource: false, profile: PROFILE })));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  // Complete module evaluation before fixture helpers import these exports.
  main(process.argv.slice(2)).catch(error => {
    console.error(`LibTorch verification failed: ${error.message}`); process.exitCode = 1;
  });
}
