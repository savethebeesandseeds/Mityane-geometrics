// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Source packaging and metadata only; this command performs no model execution.
// Adapted from the CPB source exporter; MTF has its own profile and bundle identity.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const DEFAULT_PROFILE = 'code/embeddings/encoders/mtf_jepa_mae_vicreg/family/migration/package.json';
const PROFILE_SCHEMA = 'mtf-family-source-package-v1';
const BUNDLE_SCHEMA = 'mtf-family-source-bundle-v1';
const SHA = /^[a-f0-9]{64}$/;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = message => { throw new Error(`[MTF source bundle] ${message}`); };

function relative(value) {
  if (typeof value !== 'string' || !value || value.includes('\\') || value.includes('\0') ||
      path.posix.isAbsolute(value) || /^[a-z]:/i.test(value) ||
      value.split('/').some(part => !part || part === '.' || part === '..')) {
    fail(`safe POSIX relative path required: ${String(value)}`);
  }
  return value;
}

function directory(value) {
  const absolute = path.resolve(value);
  noSymlinkAncestors(absolute);
  if (!fs.lstatSync(absolute).isDirectory()) fail(`directory required: ${absolute}`);
  return absolute;
}

function noSymlinkAncestors(absolute) {
  const parsed = path.parse(absolute);
  let current = parsed.root;
  for (const part of absolute.slice(parsed.root.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, part);
    if (fs.lstatSync(current).isSymbolicLink()) fail(`symlink refused: ${current}`);
  }
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
      /(?:^|\/)\.env(?:\.|$)/i.test(value) ||
      /\.(?:pt|pth|ckpt|so|a|dll|exe|bin|zip|tar|gz|csv|tsv|parquet|npy|npz|png|jpe?g|webp)$/i.test(value)) {
    fail(`source-only package refuses artifact/environment path: ${value}`);
  }
}

function parseProfile(bytes, profilePath) {
  const profile = JSON.parse(bytes.toString('utf8'));
  if (profile.schema !== PROFILE_SCHEMA || !Array.isArray(profile.files) || !profile.files.length ||
      !Array.isArray(profile.include_roots)) fail('invalid package profile');
  const seen = new Set();
  for (const entry of profile.files) {
    admittedSource(entry.path);
    if (seen.has(entry.path) || typeof entry.role !== 'string' || !entry.role.trim()) {
      fail(`unique file and nonempty role required: ${entry.path}`);
    }
    seen.add(entry.path);
  }
  if (!seen.has(profilePath)) fail('package profile must include itself');
  const roots = new Set();
  for (const value of profile.include_roots) {
    relative(value);
    if (roots.has(value)) fail('duplicate include root');
    roots.add(value);
  }
  return profile;
}

function checkIncludes(files, includeRoots) {
  const paths = new Set(files.map(entry => entry.path));
  let checked = 0;
  for (const entry of files) {
    if (!/\.(?:h|hpp|hh|cpp|cc|cxx)$/i.test(entry.path)) continue;
    const text = entry.content.toString('utf8');
    for (const match of text.matchAll(/^\s*#\s*include\s*"([^"\r\n]+)"/gm)) {
      const include = match[1];
      if (include.includes('\\') || include.includes('\0') || path.posix.isAbsolute(include) || /^[a-z]:/i.test(include)) {
        fail(`unsafe include in ${entry.path}: ${include}`);
      }
      const candidates = [path.posix.dirname(entry.path), ...includeRoots]
        .map(root => path.posix.normalize(path.posix.join(root, include)))
        .filter(candidate => candidate !== '..' && !candidate.startsWith('../'));
      const matches = [...new Set(candidates.filter(candidate => paths.has(candidate)))];
      if (matches.length !== 1) fail(`missing or ambiguous quoted include in ${entry.path}: ${include}`);
      checked++;
    }
  }
  return checked;
}

function closure(files) {
  return hash(JSON.stringify(files.map(({path: origin, bytes, sha256}) => ({path: origin, bytes, sha256}))));
}

function gitHead(root) {
  const result = spawnSync('git', ['rev-parse', '--verify', 'HEAD'], {cwd: root, encoding: 'utf8', timeout: 10000});
  const value = result.status === 0 ? result.stdout.trim() : '';
  return /^[a-f0-9]{40,64}$/.test(value) ? value : null;
}

/** Copy only the profile's explicit source closure into a new directory. */
export function exportBundle({root, output, profilePath = DEFAULT_PROFILE}) {
  root = directory(root);
  relative(profilePath);
  const profileBytes = fs.readFileSync(regularFile(root, profilePath));
  const profile = parseProfile(profileBytes, profilePath);
  const selected = profile.files.map(entry => {
    const content = fs.readFileSync(regularFile(root, entry.path));
    return {...entry, content, bytes: content.length, sha256: hash(content)};
  }).sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  if (!selected.find(entry => entry.path === profilePath).content.equals(profileBytes)) {
    fail('profile changed while reading sources');
  }
  const checked = checkIncludes(selected, profile.include_roots);
  const bundle = path.resolve(output);
  const parent = directory(path.dirname(bundle));
  if (fs.existsSync(bundle)) fail(`output must be a new leaf: ${bundle}`);
  // Source and dependency admission completes before the output leaf is created.
  const files = selected.map(({path: origin, role, bytes, sha256}) => ({
    path: origin, bundle_path: `sources/${origin}`, role, bytes, sha256,
  }));
  const manifest = {
    schema: BUNDLE_SCHEMA,
    source_snapshot: {
      kind: 'selected-working-tree-snapshot',
      git_head: gitHead(root),
      git_head_is_snapshot_identity: false,
      identity: 'source_closure_sha256 and the independently retained manifest SHA256 bind the actual selected bytes',
    },
    profile: {path: profilePath, sha256: hash(profileBytes)},
    include_roots: [...profile.include_roots],
    quoted_includes_checked: checked,
    source_closure_sha256: closure(files),
    files,
  };
  fs.mkdirSync(path.join(parent, path.basename(bundle)));
  for (const entry of selected) {
    const target = path.join(bundle, 'sources', ...entry.path.split('/'));
    fs.mkdirSync(path.dirname(target), {recursive: true});
    fs.writeFileSync(target, entry.content, {flag: 'wx'});
  }
  const manifestPath = path.join(bundle, 'manifest.json');
  const bytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  fs.writeFileSync(manifestPath, bytes, {flag: 'wx'});
  const manifestSha256 = hash(bytes);
  verifyBundle({bundle, expectedManifestSha256: manifestSha256});
  return {bundle, manifestPath, manifestSha256, files: files.length, sourceClosureSha256: manifest.source_closure_sha256,
    quotedIncludesChecked: checked};
}

function inventory(root, prefix = '') {
  const found = [];
  for (const name of fs.readdirSync(path.join(root, prefix)).sort()) {
    const relativePath = prefix ? `${prefix}/${name}` : name;
    const info = fs.lstatSync(path.join(root, ...relativePath.split('/')));
    if (info.isSymbolicLink()) fail(`symlink refused: ${relativePath}`);
    if (info.isDirectory()) found.push(...inventory(root, relativePath));
    else if (info.isFile()) found.push(relativePath);
    else fail(`non-regular bundle member: ${relativePath}`);
  }
  return found;
}

/** Verify a relocated bundle using an independent manifest pin; no origin checkout is read. */
export function verifyBundle({bundle, expectedManifestSha256}) {
  if (!SHA.test(expectedManifestSha256 ?? '')) fail('independent manifest SHA256 required');
  bundle = directory(bundle);
  const bytes = fs.readFileSync(regularFile(bundle, 'manifest.json'));
  if (hash(bytes) !== expectedManifestSha256) fail('manifest SHA256 mismatch');
  const manifest = JSON.parse(bytes.toString('utf8'));
  if (manifest.schema !== BUNDLE_SCHEMA || !Array.isArray(manifest.files) || !manifest.files.length ||
      !manifest.profile || !Array.isArray(manifest.include_roots) ||
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
  const sorted = [...files].sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  if (JSON.stringify(files.map(entry => entry.path)) !== JSON.stringify(sorted.map(entry => entry.path))) {
    fail('source members must be canonically ordered');
  }
  if (closure(files) !== manifest.source_closure_sha256) fail('source closure SHA256 mismatch');
  relative(manifest.profile.path);
  const profileFile = files.find(entry => entry.path === manifest.profile.path);
  if (!profileFile || profileFile.sha256 !== manifest.profile.sha256) fail('profile binding mismatch');
  const profile = parseProfile(profileFile.content, manifest.profile.path);
  if (JSON.stringify(profile.include_roots) !== JSON.stringify(manifest.include_roots)) fail('include root binding mismatch');
  const intended = [...profile.files].sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  if (JSON.stringify(intended.map(({path: origin, role}) => ({path: origin, role}))) !==
      JSON.stringify(files.map(({path: origin, role}) => ({path: origin, role})))) fail('profile member binding mismatch');
  const checked = checkIncludes(files, manifest.include_roots);
  if (checked !== manifest.quoted_includes_checked) fail('quoted include count mismatch');
  const expected = ['manifest.json', ...files.map(entry => entry.bundle_path)].sort();
  if (JSON.stringify(inventory(bundle).sort()) !== JSON.stringify(expected)) fail('unexpected or missing bundle members');
  return {bundle, manifestSha256: expectedManifestSha256, files: files.length,
    sourceClosureSha256: manifest.source_closure_sha256, quotedIncludesChecked: checked};
}

function main(args) {
  const usage = 'Usage: export-mtf-family.mjs --output NEWLEAF\n' +
    '   or: export-mtf-family.mjs --verify BUNDLE --expected-manifest-sha256 SHA256';
  if (args.length === 1 && args[0] === '--help') { process.stdout.write(`${usage}\n`); return; }
  const options = new Map();
  if (args.length % 2) fail(usage);
  for (let i = 0; i < args.length; i += 2) {
    if (!['--output', '--verify', '--expected-manifest-sha256'].includes(args[i]) || options.has(args[i]) || !args[i + 1]) fail(usage);
    options.set(args[i], args[i + 1]);
  }
  let result;
  if (options.size === 1 && options.has('--output')) {
    const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
    result = exportBundle({root, output: options.get('--output')});
  } else if (options.size === 2 && options.has('--verify') && options.has('--expected-manifest-sha256')) {
    result = verifyBundle({bundle: options.get('--verify'), expectedManifestSha256: options.get('--expected-manifest-sha256')});
  } else fail(usage);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(process.argv.slice(2)); }
  catch (error) { process.stderr.write(`${error.message}\n`); process.exitCode = 1; }
}
