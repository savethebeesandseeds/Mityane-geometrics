// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Source copying and metadata only. No model or training execution.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const DEFAULT_PROFILE = 'code/embeddings/encoders/raw_patch_bottleneck_mae/family/migration/package.json';
const PROFILE_SCHEMA = 'rpb-family-source-package-v1';
const BUNDLE_SCHEMA = 'rpb-family-source-bundle-v1';
const SHA = /^[a-f0-9]{64}$/;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = message => { throw new Error(`[RPB source bundle] ${message}`); };

function relative(value) {
  if (typeof value !== 'string' || !value || /[\\:\x00-\x1f]/.test(value) ||
      path.posix.isAbsolute(value) || value.split('/').some(part => !part || part === '.' || part === '..')) {
    fail(`safe POSIX relative path required: ${String(value)}`);
  }
  return value;
}

function noSymlinkAncestors(absolute) {
  const parsed = path.parse(absolute);
  let current = parsed.root;
  for (const part of absolute.slice(parsed.root.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, part);
    if (fs.lstatSync(current).isSymbolicLink()) fail(`symlink refused: ${current}`);
  }
}

function directory(value) {
  const absolute = path.resolve(value);
  noSymlinkAncestors(absolute);
  if (!fs.lstatSync(absolute).isDirectory()) fail(`directory required: ${absolute}`);
  return absolute;
}

function regularFile(root, value) {
  relative(value);
  const absolute = path.join(root, ...value.split('/'));
  noSymlinkAncestors(absolute);
  if (!fs.lstatSync(absolute).isFile()) fail(`regular file required: ${value}`);
  return absolute;
}

function admittedSource(value) {
  relative(value);
  if (value.split('/').some(part => ['.git', '.temp', '.build', '.external', 'node_modules',
    'output', 'checkpoints', 'libtorch'].includes(part)) ||
      /(?:^|\/)\.env(?:\.|$)/i.test(value) || /\.(?:pt|pth|ckpt|so|a|dll|exe|bin)$/i.test(value)) {
    fail(`source-only package refuses artifact/environment path: ${value}`);
  }
}

function parseProfile(bytes, profilePath) {
  const profile = JSON.parse(bytes.toString('utf8'));
  if (profile.schema !== PROFILE_SCHEMA || !Array.isArray(profile.files) || !profile.files.length ||
      !Array.isArray(profile.include_roots) || !Array.isArray(profile.dependency_roots)) fail('invalid package profile');
  const seen = new Set();
  for (const entry of profile.files) {
    admittedSource(entry.path);
    if (seen.has(entry.path) || typeof entry.role !== 'string' || !entry.role.trim() ||
        entry.role === 'quoted-include-dependency') fail(`unique root file and explicit role required: ${entry.path}`);
    seen.add(entry.path);
  }
  if (!seen.has(profilePath)) fail('package profile must include itself');
  for (const key of ['include_roots', 'dependency_roots']) {
    const roots = new Set();
    for (const value of profile[key]) {
      admittedSource(value);
      if (roots.has(value)) fail(`duplicate ${key} entry`);
      roots.add(value);
    }
  }
  return profile;
}

const inRoots = (value, roots) => roots.some(root => value === root || value.startsWith(`${root}/`));
const compare = (a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0;

// The profile curates entry files and bounds every possible auto-discovered local
// dependency. Relative compatibility wrappers retain their repository paths.
function discoverSources(root, profile) {
  const selected = new Map();
  for (const entry of profile.files) {
    const content = fs.readFileSync(regularFile(root, entry.path));
    selected.set(entry.path, {...entry, content});
  }
  const queue = [...selected.keys()];
  let checked = 0;
  for (let i = 0; i < queue.length; i++) {
    const entry = selected.get(queue[i]);
    if (!/\.(?:h|hpp|hh|cpp|cc|cxx)$/i.test(entry.path)) continue;
    for (const match of entry.content.toString('utf8').matchAll(/^\s*#\s*include\s*"([^"\r\n]+)"/gm)) {
      const include = match[1];
      if (/[\\:\x00-\x1f]/.test(include) || path.posix.isAbsolute(include)) {
        fail(`unsafe quoted include in ${entry.path}: ${include}`);
      }
      const candidates = [...new Set([path.posix.dirname(entry.path), ...profile.include_roots]
        .map(base => path.posix.normalize(path.posix.join(base, include)))
        .filter(candidate => candidate !== '..' && !candidate.startsWith('../')))];
      const matches = [];
      for (const candidate of candidates) {
        const absolute = path.join(root, ...candidate.split('/'));
        let info;
        try { info = fs.lstatSync(absolute); }
        catch (error) { if (error.code === 'ENOENT' || error.code === 'ENOTDIR') continue; throw error; }
        if (!info.isFile() && !info.isSymbolicLink()) continue;
        admittedSource(candidate);
        if (!selected.has(candidate) && !inRoots(candidate, profile.dependency_roots)) {
          fail(`quoted dependency outside curated roots in ${entry.path}: ${candidate}`);
        }
        regularFile(root, candidate);
        matches.push(candidate);
      }
      if (matches.length !== 1) fail(`missing or ambiguous quoted include in ${entry.path}: ${include}`);
      const dependency = matches[0];
      if (!selected.has(dependency)) {
        selected.set(dependency, {path: dependency, role: 'quoted-include-dependency',
          content: fs.readFileSync(regularFile(root, dependency))});
        queue.push(dependency);
      }
      checked++;
    }
  }
  return {files: [...selected.values()].map(entry => ({...entry,
    bytes: entry.content.length, sha256: hash(entry.content)})).sort(compare), checked};
}

function closure(files) {
  return hash(JSON.stringify(files.map(({path: origin, bytes, sha256}) => ({path: origin, bytes, sha256}))));
}

function gitHead(root) {
  const result = spawnSync('git', ['rev-parse', '--verify', 'HEAD'], {cwd: root, encoding: 'utf8', timeout: 10000});
  const value = result.status === 0 ? result.stdout.trim() : '';
  return /^[a-f0-9]{40,64}$/.test(value) ? value : null;
}

/** Export a curated source closure to a fresh leaf; never overwrite a delivery. */
export function exportBundle({root, output, profilePath = DEFAULT_PROFILE}) {
  root = directory(root);
  relative(profilePath);
  const profileBytes = fs.readFileSync(regularFile(root, profilePath));
  const profile = parseProfile(profileBytes, profilePath);
  const selected = discoverSources(root, profile);
  if (!selected.files.find(entry => entry.path === profilePath).content.equals(profileBytes)) fail('profile changed during source discovery');
  for (const entry of selected.files) {
    if (!fs.readFileSync(regularFile(root, entry.path)).equals(entry.content)) fail(`source changed during discovery: ${entry.path}`);
  }
  const bundle = path.resolve(output);
  directory(path.dirname(bundle));
  try { fs.lstatSync(bundle); fail(`output must be a new leaf: ${bundle}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const files = selected.files.map(({path: origin, role, bytes, sha256}) => ({
    path: origin, bundle_path: `sources/${origin}`, role, bytes, sha256,
  }));
  const manifest = {
    schema: BUNDLE_SCHEMA,
    source_snapshot: {kind: 'selected-working-tree-snapshot', git_head: gitHead(root),
      git_head_is_snapshot_identity: false,
      identity: 'Per-member hashes, source_closure_sha256 and independently retained manifest SHA256 bind delivered working-tree bytes'},
    profile: {path: profilePath, sha256: hash(profileBytes)},
    include_roots: [...profile.include_roots], dependency_roots: [...profile.dependency_roots],
    quoted_includes_checked: selected.checked, source_closure_sha256: closure(files), files,
  };
  // Admission and dependency discovery finish before any output mutation.
  fs.mkdirSync(bundle);
  for (const entry of selected.files) {
    const target = path.join(bundle, 'sources', ...entry.path.split('/'));
    fs.mkdirSync(path.dirname(target), {recursive: true});
    fs.writeFileSync(target, entry.content, {flag: 'wx'});
  }
  const manifestPath = path.join(bundle, 'manifest.json');
  const bytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  fs.writeFileSync(manifestPath, bytes, {flag: 'wx'});
  const manifestSha256 = hash(bytes);
  verifyBundle({bundle, expectedManifestSha256: manifestSha256});
  return {bundle, manifestPath, manifestSha256, files: files.length,
    sourceClosureSha256: manifest.source_closure_sha256, quotedIncludesChecked: selected.checked};
}

function inventory(root, prefix = '') {
  const found = [];
  for (const name of fs.readdirSync(path.join(root, prefix)).sort()) {
    const member = prefix ? `${prefix}/${name}` : name;
    const info = fs.lstatSync(path.join(root, ...member.split('/')));
    if (info.isSymbolicLink()) fail(`symlink refused: ${member}`);
    if (info.isDirectory()) found.push(...inventory(root, member));
    else if (info.isFile()) found.push(member);
    else fail(`non-regular bundle member: ${member}`);
  }
  return found;
}

/** Offline verification uses only bundle bytes and an independently retained pin. */
export function verifyBundle({bundle, expectedManifestSha256}) {
  if (!SHA.test(expectedManifestSha256 ?? '')) fail('independent manifest SHA256 required');
  bundle = directory(bundle);
  const bytes = fs.readFileSync(regularFile(bundle, 'manifest.json'));
  if (hash(bytes) !== expectedManifestSha256) fail('manifest SHA256 mismatch');
  const manifest = JSON.parse(bytes.toString('utf8'));
  if (manifest.schema !== BUNDLE_SCHEMA || !Array.isArray(manifest.files) || !manifest.files.length ||
      !manifest.profile || !Array.isArray(manifest.include_roots) || !Array.isArray(manifest.dependency_roots) ||
      manifest.source_snapshot?.kind !== 'selected-working-tree-snapshot' ||
      manifest.source_snapshot?.git_head_is_snapshot_identity !== false) fail('invalid source bundle manifest');
  const seen = new Set();
  const files = manifest.files.map(entry => {
    admittedSource(entry.path);
    if (entry.bundle_path !== `sources/${entry.path}` || seen.has(entry.path) ||
        !Number.isSafeInteger(entry.bytes) || entry.bytes < 0 || !SHA.test(entry.sha256) ||
        typeof entry.role !== 'string' || !entry.role.trim()) fail('invalid or duplicate source member');
    seen.add(entry.path);
    const content = fs.readFileSync(regularFile(bundle, entry.bundle_path));
    if (content.length !== entry.bytes || hash(content) !== entry.sha256) fail(`member byte/hash mismatch: ${entry.path}`);
    return {...entry, content};
  });
  if (JSON.stringify(files.map(entry => entry.path)) !== JSON.stringify([...files].sort(compare).map(entry => entry.path))) {
    fail('source members must be canonically ordered');
  }
  if (closure(files) !== manifest.source_closure_sha256) fail('source closure SHA256 mismatch');
  relative(manifest.profile.path);
  const profileFile = files.find(entry => entry.path === manifest.profile.path);
  if (!profileFile || profileFile.sha256 !== manifest.profile.sha256) fail('profile binding mismatch');
  const profile = parseProfile(profileFile.content, manifest.profile.path);
  for (const key of ['include_roots', 'dependency_roots']) {
    if (JSON.stringify(profile[key]) !== JSON.stringify(manifest[key])) fail(`${key} binding mismatch`);
  }
  const discovered = discoverSources(directory(path.join(bundle, 'sources')), profile);
  const mapping = values => values.map(({path: origin, role, bytes: size, sha256}) => ({path: origin, role, bytes: size, sha256}));
  if (JSON.stringify(mapping(discovered.files)) !== JSON.stringify(mapping(files))) fail('profile source-closure binding mismatch');
  if (discovered.checked !== manifest.quoted_includes_checked) fail('quoted include count mismatch');
  const expected = ['manifest.json', ...files.map(entry => entry.bundle_path)].sort();
  if (JSON.stringify(inventory(bundle).sort()) !== JSON.stringify(expected)) fail('unexpected or missing bundle members');
  return {bundle, manifestSha256: expectedManifestSha256, files: files.length,
    sourceClosureSha256: manifest.source_closure_sha256, quotedIncludesChecked: discovered.checked};
}

function main(args) {
  const usage = 'Usage: export-rpb-family.mjs --output NEWLEAF\n' +
    '   or: export-rpb-family.mjs --verify BUNDLE --expected-manifest-sha256 SHA256';
  if (args.length === 1 && args[0] === '--help') { process.stdout.write(`${usage}\n`); return; }
  const options = new Map();
  if (args.length % 2) fail(usage);
  for (let i = 0; i < args.length; i += 2) {
    if (!['--output', '--verify', '--expected-manifest-sha256'].includes(args[i]) || options.has(args[i]) || !args[i + 1]) fail(usage);
    options.set(args[i], args[i + 1]);
  }
  let result;
  if (options.size === 1 && options.has('--output')) {
    result = exportBundle({root: path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..'), output: options.get('--output')});
  } else if (options.size === 2 && options.has('--verify') && options.has('--expected-manifest-sha256')) {
    result = verifyBundle({bundle: options.get('--verify'), expectedManifestSha256: options.get('--expected-manifest-sha256')});
  } else fail(usage);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(process.argv.slice(2)); }
  catch (error) { process.stderr.write(`${error.message}\n`); process.exitCode = 1; }
}
