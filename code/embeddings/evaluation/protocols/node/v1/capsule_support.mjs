// Metadata receipt support for the maintained Node capsule launchers.
// New receipts use mityane-artifact-json-v1; historical receipts stay intact.
import fs from 'node:fs';
import path from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import {
  requireCondition, safeRelative, canonicalDirectory, admitFiles, readAdmitted, hashAdmitted,
  fileSHA256, readJSON, writeNew, jsonBytes, JSON_ENCODING, sha256, parseJSON,
} from '../../../../../shared/artifacts/node/v1/files.mjs';

export const check = requireCondition;
export const CAPSULE_CLI_SOURCES = Object.freeze([
  'code/embeddings/evaluation/protocols/node/v1/capsule_cli.mjs',
  'code/embeddings/evaluation/protocols/node/v1/metadata.mjs',
  'code/scripts/node.sh', 'code/scripts/node-runtime.sh',
]);
export const equal = isDeepStrictEqual;
export const join = path.join;
export function lines(value) {
  if (value === '') return [];
  const result = value.split(/\r\n|[\n\r\v\f\u001c-\u001e\u0085\u2028\u2029]/);
  if (result[result.length - 1] === '') result.pop();
  return result;
}
export const sha = file => typeof file === 'string' ? fileSHA256(file, { singleLink: true }) : hashAdmitted(file);
export const exists = file => fs.existsSync(file);
export const text = file => new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(
  readAdmitted(typeof file === 'string' ? admitFiles([file], { singleLink: true })[0] : file));
export const document = file => typeof file === 'string' ?
  readJSON(file, { singleLink: true, integers: 'number', duplicates: 'last', allowBOM: false }) :
  parseJSON(readAdmitted(file), { integers: 'number', duplicates: 'last', allowBOM: false });
export const direct = directory => canonicalDirectory(directory);

export function regular(file) {
  return admitFiles([file], { singleLink: true })[0];
}

export function write(file, value) {
  direct(path.dirname(file));
  writeNew(file, typeof value === 'string' ? Buffer.from(value, 'utf8') :
    jsonBytes(value, { sortKeys: false, ensureAscii: true, indent: 2 }));
}

export function copy(source, target) {
  writeNew(target, readAdmitted(typeof source === 'string' ? regular(source) : source));
}

export function mkdir(directory, recursive = false) {
  check(typeof directory === 'string' && path.isAbsolute(directory) && path.normalize(directory) === directory,
    'absolute canonical directory required');
  if (exists(directory)) {
    direct(directory);
    check(recursive, 'exclusive directory already exists');
    return;
  }
  const missing = [directory];
  let parent = path.dirname(directory);
  while (!exists(parent)) { missing.unshift(parent); parent = path.dirname(parent); }
  direct(parent);
  check(recursive || missing.length === 1, 'parent directory must already exist');
  for (const target of missing) {
    direct(path.dirname(target));
    fs.mkdirSync(target);
    direct(target);
  }
}

export function manifestRows(file) {
  return lines(text(file)).map(line => {
    const delimiter = line.indexOf('  ');
    check(delimiter === 64 && /^[a-f0-9]{64}$/.test(line.slice(0, delimiter)), 'canonical SHA256 manifest line');
    const name = line.slice(delimiter + 2);
    check(name.length > 0 && !name.includes('\n') && !name.includes('\r'), 'nonempty manifest role');
    return { digest: line.slice(0, delimiter), name };
  });
}

export function verifyConsumed(file) {
  const rows = manifestRows(file);
  check(rows.length > 0, 'complete consumed-artifact ledger');
  // Admit the complete matrix before the first payload read and reject aliases.
  const admitted = admitFiles(rows.map(row => row.name), { singleLink: true });
  for (let i = 0; i < rows.length; i += 1) {
    check(hashAdmitted(admitted[i]) === rows[i].digest, 'consumed saved evidence remains exact');
  }
}

export function retainedAdmissions(root) {
  direct(root);
  const files = [];
  function visit(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const file = join(directory, entry.name);
      check(!entry.isSymbolicLink(), 'retained capsule cannot contain symlinks');
      if (entry.isDirectory()) { direct(file); visit(file); }
      else { check(entry.isFile(), 'retained capsule regular files only'); files.push(file); }
    }
  }
  visit(root);
  files.sort();
  return admitFiles(files, { singleLink: true });
}

export const retainedFiles = root => retainedAdmissions(root).map(row => row.path);

export function finalize(root, completion, { admittedFiles } = {}) {
  check(!exists(join(root, 'artifacts.sha256')) && !exists(join(root, 'complete.json')),
    'completed capsule cannot be finalized again');
  const current = retainedAdmissions(root);
  const files = [...(admittedFiles ?? current)].sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  check(equal(files.map(row => row.path).sort(), current.map(row => row.path).sort()),
    'retained artifact inventory changed after admission');
  const manifest = files.map(row =>
    `${hashAdmitted(row)}  ${path.relative(root, row.path).split(path.sep).join('/')}\n`).join('');
  write(join(root, 'artifacts.sha256'), manifest);
  write(join(root, 'complete.json'), { ...completion, artifacts_manifest_sha256: sha256(Buffer.from(manifest, 'utf8')) });
}

export function finite(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

export function timing(root, stage) {
  const raw = text(join(root, `${stage}.seconds`)).trim();
  check(raw.length > 0, 'finite stage wall time');
  const value = Number(raw);
  check(finite(value) && value >= 0, 'finite stage wall time');
  return value;
}

export function assertNewRoot(root, runRoot, prefix = 'admission-') {
  direct(root);
  direct(runRoot);
  check(path.dirname(root) === runRoot && path.basename(root).startsWith(prefix) &&
    path.basename(root).length > prefix.length, 'new capsule stays in its declared experiment namespace');
}

export function implementationReceipt(sourceId, pathMapping) {
  return { language: 'node', receipt_encoding: JSON_ENCODING, source_identity: sourceId,
    equivalence: 'not-certified', historical_path_mapping: pathMapping };
}

// Callers supply the complete current source list and a command runner explicitly.
// No implicit make, Python, model, or saved-data execution is available here.
export function captureSources(root, policy) {
  const { repositoryRoot, buildDirectory, sourceList, binaries, commandOutput,
    requiredSources = [], sourceIdBinaries = binaries.slice(0, 1) } = policy;
  direct(repositoryRoot); direct(buildDirectory);
  check(typeof sourceList === 'string', 'explicit complete source list required');
  check(typeof commandOutput === 'function', 'source-id command runner must be explicitly bound');
  const names = lines(sourceList);
  check(names.length > 0 && equal(names, [...new Set(names)].sort()), 'closed sorted source list');
  names.forEach(safeRelative);
  check(Array.isArray(binaries) && binaries.length > 0, 'explicit compiled binary role matrix');
  binaries.forEach(safeRelative);
  check(sourceIdBinaries.every(name => binaries.includes(name)), 'source-id command uses declared binary roles only');
  check(requiredSources.every(name => names.includes(name)), 'new implementation and support sources must be captured');
  // Admit every source and binary before hashing or invoking the source-id
  // callback. The same signatures remain authoritative through every read.
  const allAdmitted = admitFiles([...names.map(name => join(repositoryRoot, name)),
    ...binaries.map(name => join(buildDirectory, name))], { singleLink: true });
  const admitted = allAdmitted.slice(0, names.length), binaryAdmissions = allAdmitted.slice(names.length);
  const sourceRows = new Map(names.map((name, i) => [name, admitted[i]]));
  const hashes = new Map(names.map((name, i) => [name, hashAdmitted(admitted[i])]));
  const out = join(root, 'freeze');
  mkdir(out);
  write(join(out, 'source-list.txt'), sourceList);
  const sourceManifest = names.map(name => `${hashes.get(name)}  ${name}\n`).join('');
  write(join(out, 'sources.sha256'), sourceManifest);
  const sourceId = sha256(Buffer.from(sourceManifest, 'utf8'));
  for (const name of sourceIdBinaries) {
    check(commandOutput(join(buildDirectory, name), ['--source-id']).trim() === sourceId,
      'compiled producer/reader matches captured source');
  }
  for (const name of names) {
    const target = join(out, 'source', name);
    mkdir(path.dirname(target), true);
    copy(sourceRows.get(name), target);
    check(sha(target) === hashes.get(name) && hashAdmitted(sourceRows.get(name)) === hashes.get(name),
      'source changed during capture');
  }
  write(join(out, 'binaries.sha256'), binaryAdmissions.map(row => `${hashAdmitted(row)}  ${row.path}\n`).join(''));
  return { out, sourceId, sourceFiles: names.length, hashes, sourceRows, binaryAdmissions };
}

export function verifyCapturedSources(root, policy, receipt) {
  const rows = manifestRows(join(root, 'freeze/sources.sha256'));
  const names = rows.map(row => row.name);
  check(equal(names, [...new Set(names)].sort()) && names.length === receipt.source_files &&
    equal(names, lines(text(join(root, 'freeze/source-list.txt')))), 'closed captured source population');
  names.forEach(safeRelative);
  const binaries = manifestRows(join(root, 'freeze/binaries.sha256'));
  check(equal(binaries.map(row => row.name), policy.binaries.map(name => join(policy.buildDirectory, name))),
    'exact compiled binary roles');
  const sourceFiles = rows.flatMap(row => [join(policy.repositoryRoot, row.name), join(root, 'freeze/source', row.name)]);
  const allAdmitted = admitFiles([...sourceFiles, ...binaries.map(row => row.name)], { singleLink: true });
  for (let index = 0; index < rows.length; index += 1) {
    check(hashAdmitted(allAdmitted[2 * index]) === rows[index].digest &&
      hashAdmitted(allAdmitted[2 * index + 1]) === rows[index].digest, 'actual live and captured source bytes');
  }
  for (let index = 0; index < binaries.length; index += 1)
    check(hashAdmitted(allAdmitted[sourceFiles.length + index]) === binaries[index].digest, 'actual compiled binary bytes');
}

export function checksumLog(root, name, manifest, count) {
  const actual = lines(text(join(root, name)));
  const expected = manifestRows(manifest).map(row => `${row.name}: OK`);
  check(equal(actual, expected) && actual.length === count, `exact final checksum role population: ${name}`);
}
