// Source-only migration implementation. Runtime equivalence remains unverified.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { admitFiles, canonicalDirectory, createExclusiveDirectory, hashAdmitted,
  inside, parseJSON, readAdmitted, requireCondition, safeRelative, sha256,
  writeJSON, writeNew } from '../../../../../shared/artifacts/node/v1/files.mjs';

export { fs, path, requireCondition as check, sha256, inside };
export const utc = () => new Date().toISOString();
export function options(args, allowed, required = []) {
  requireCondition(args && typeof args === 'object', 'explicit metadata options');
  requireCondition(Object.keys(args).every(name => allowed.includes(name)), 'closed metadata option set');
  requireCondition(required.every(name => Object.hasOwn(args, name)), 'missing required metadata option');
}
export function same(actual, expected, reason = 'exact metadata association') {
  try { assert.deepStrictEqual(actual, expected); } catch { throw new Error(reason); }
}
export const directory = value => canonicalDirectory(value);
export const admission = names => admitFiles(names, { singleLink: true });
export const text = name => new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(readAdmitted(admission([name])[0])).replace(/\r\n?/g, '\n');
export const document = name => parseJSON(readAdmitted(admission([name])[0]), { integers: 'number', duplicates: 'last', allowBOM: false });
export const hash = name => hashAdmitted(admission([name])[0]);
export const records = names => admission(names).map(row => ({ path: row.path, bytes: row.bytes, sha256: hashAdmitted(row) }));
export function preserved(rows) { same(records(rows.map(row => row.path)), rows, 'admitted inputs changed'); }
export function appendAdmitted(filename, value) {
  const row = admission([filename])[0], bytes = Buffer.from(value, 'utf8');
  const descriptor = fs.openSync(filename, fs.constants.O_WRONLY | fs.constants.O_APPEND | fs.constants.O_NOFOLLOW);
  try {
    const before = fs.fstatSync(descriptor, { bigint: true });
    same([before.dev, before.ino, before.size, before.mode, before.mtimeNs, before.ctimeNs], row.signature, 'admitted log redirected before append');
    let offset = 0;
    while (offset < bytes.length) {
      const count = fs.writeSync(descriptor, bytes, offset, bytes.length - offset);
      checkWrite(count > 0); offset += count;
    }
    fs.fsyncSync(descriptor);
    const after = fs.fstatSync(descriptor, { bigint: true }), current = fs.lstatSync(filename, { bigint: true });
    requireCondition(after.size === before.size + BigInt(bytes.length) && current.dev === before.dev && current.ino === before.ino &&
      current.size === after.size && !current.isSymbolicLink() && current.nlink === 1n && current.mode === before.mode, 'admitted log changed during append');
    directory(path.dirname(filename));
  } finally { fs.closeSync(descriptor); }
}
function checkWrite(value) { requireCondition(value, 'short admitted log append'); }
export function copyMatrix(names, destinations) {
  const rows = admission(names);
  requireCondition(names.length === destinations.length, 'complete copy matrix');
  for (let i = 0; i < rows.length; ++i) {
    const parent = path.dirname(destinations[i]);
    makeParents(parent);
    writeNew(destinations[i], readAdmitted(rows[i]));
  }
  preserved(rows.map(row => ({ path: row.path, bytes: row.bytes, sha256: hashAdmitted(row) })));
}
export function makeParents(target) {
  if (fs.existsSync(target)) return directory(target);
  const missing = []; let ancestor = target;
  while (!fs.existsSync(ancestor)) { missing.push(ancestor); ancestor = path.dirname(ancestor); }
  directory(ancestor);
  for (const part of missing.reverse()) { fs.mkdirSync(part); directory(part); }
  return target;
}
export function emptyTarget(repo, target, runRoot, family, prefix = '') {
  directory(repo);
  requireCondition(path.isAbsolute(target) && path.normalize(target) === target &&
    path.dirname(target) === path.join(repo, runRoot, family) && path.basename(target).startsWith(prefix), 'exclusive named target family');
  directory(path.dirname(target));
  if (fs.existsSync(target)) { directory(target); requireCondition(fs.readdirSync(target).length === 0, 'preserve existing target'); }
  else createExclusiveDirectory(target, { root: repo });
  return target;
}
export function safeSource(name) {
  safeRelative(name);
  requireCondition(['Makefile', 'dependencies.lock', 'setup.sh', '.archive/embeddings/pre-cleanup-2026-10-18/doc/dataset_registry.json'].includes(name) || name.startsWith('code/'), 'closed SOURCE namespace');
  return name;
}
export const manifestText = rows => rows.map(row => `${row.sha256}  ${row.path}\n`).join('');
export function capture(repo, target, names, config) {
  requireCondition(names.length > 0, 'nonempty SOURCE closure');
  same(names, [...new Set(names)].sort(), 'canonical unique SOURCE order');
  names.forEach(safeSource);
  for (const required of [config.card, 'Makefile', 'dependencies.lock', 'setup.sh',
    'code/scripts/verify-libtorch.mjs', 'code/scripts/node.sh', 'code/scripts/node-runtime.sh',
    'code/embeddings/evaluation/protocols/node/v1/saved_run_support.mjs',
    'code/shared/artifacts/node/v1/files.mjs', `${config.code}/prepare_saved_run.mjs`]) {
    requireCondition(names.includes(required), `missing exact source dependency: ${required}`);
  }
  const before = records(names.map(name => path.join(repo, name)));
  const rows = before.map((row, i) => ({ ...row, path: names[i] }));
  requireCondition(rows.find(row => row.path === config.card).sha256 === config.cardSHA, 'frozen prospective card');
  const body = manifestText(rows), fingerprint = sha256(body);
  copyMatrix(before.map(row => row.path), names.map(name => path.join(target, 'source', name)));
  preserved(before);
  writeJSON(path.join(target, 'source-manifest.json'), { protocol: config.protocol, source_sha256: fingerprint, files: rows });
  writeNew(path.join(target, 'source-inputs.sha256'), body);
  writeNew(path.join(target, 'source-fingerprint.txt'), fingerprint + '\n');
  return fingerprint;
}
export function verifySources(repo, target) {
  const value = document(path.join(target, 'source-manifest.json'));
  const names = value.files.map(row => safeSource(row.path));
  same(names, [...new Set(names)].sort(), 'complete sorted SOURCE names');
  const actual = records([...names.map(name => path.join(repo, name)), ...names.map(name => path.join(target, 'source', name))]);
  value.files.forEach((row, i) => {
    for (const got of [actual[i], actual[i + names.length]]) requireCondition(got.bytes === row.bytes && got.sha256 === row.sha256, `SOURCE changed: ${row.path}`);
  });
  const body = manifestText(value.files);
  requireCondition(sha256(body) === value.source_sha256 && text(path.join(target, 'source-inputs.sha256')) === body &&
    text(path.join(target, 'source-fingerprint.txt')) === `${value.source_sha256}\n`, 'canonical SOURCE identity');
  return value;
}
// Only explicitly permitted metadata commands. No shell, tensor loader or fit.
export function command(executable, args, cwd) {
  requireCondition(executable === 'make' || executable.startsWith('/opt/mityane-geometrics/embeddings/build/'), 'closed metadata command');
  if (executable !== 'make') requireCondition(args.length === 1 && ['--plan', '--source-id'].includes(args[0]), 'binary metadata command only');
  const result = spawnSync(executable, args, { cwd, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  requireCondition(!result.error && result.status === 0, `metadata command failed: ${executable}: ${result.stderr ?? result.error}`);
  return result.stdout.trim();
}
export function freezeScopes(repo, target, names, config, scopeNames, enclosing) {
  const identity = capture(repo, target, names, config);
  const expanded = command('make', ['-s', '-f', `${config.code}/Makefile`, config.scopeTarget], repo);
  const source = document(path.join(target, 'source-manifest.json'));
  const index = new Map(source.files.map(row => [row.path, row]));
  const scopes = {};
  for (const line of expanded.split('\n')) {
    const split = line.indexOf('|'), name = line.slice(0, split), filename = line.slice(split + 1);
    requireCondition(split > 0 && scopeNames.includes(name), 'closed compiler scope'); safeSource(filename);
    (scopes[name] ??= []).push(filename);
  }
  same(Object.keys(scopes).sort(), scopeNames.slice().sort(), 'all compiler scopes');
  const out = {};
  for (const [name, paths] of Object.entries(scopes)) {
    same(paths, [...new Set(paths)].sort(), 'canonical scope subset');
    requireCondition(paths.every(name => index.has(name)), 'scope inside SOURCE closure');
    const files = paths.map(name => index.get(name)); out[name] = { source_sha256: sha256(manifestText(files)), files };
  }
  requireCondition(out[enclosing].source_sha256 === identity, 'enclosing compiler scope');
  verifySources(repo, target); writeJSON(path.join(target, 'source-scopes.json'), out); return identity;
}
// POSIX shlex rules needed by compiler receipts; expansions are never evaluated.
export function shellWords(line) {
  const words = []; let word = '', quote = '', active = false;
  for (let i = 0; i < line.length; ++i) {
    const ch = line[i];
    if (!quote && /\s/.test(ch)) { if (active) { words.push(word); word = ''; active = false; } continue; }
    if (ch === '\\' && quote !== "'") {
      requireCondition(i + 1 < line.length, 'unterminated shell escape');
      const next = line[++i];
      if (quote === '"' && !['$', '`', '"', '\\', '\n'].includes(next)) word += '\\';
      if (next !== '\n') word += next; active = true; continue;
    }
    if (ch === '"' || ch === "'") {
      if (!quote) { quote = ch; active = true; continue; }
      if (quote === ch) { quote = ''; continue; }
    }
    word += ch; active = true;
  }
  requireCondition(!quote, 'unterminated compiler quote'); if (active) words.push(word); return words;
}
export function compileCommands(log, required, binary, scopes = {}) {
  requireCondition(!log.includes('/embedding/.external/libtorch') && !log.includes('not found'), 'internal SDK runtime');
  const lines = log.split(/\r?\n/).flatMap(line => { try { return [[line, shellWords(line)]]; } catch { return []; } });
  const commands = {}, outputs = [];
  for (const [source, object, macro, scope] of required) {
    const found = lines.filter(([, args]) => args.includes('-c') && args.includes('-o') && args[args.indexOf('-c') + 1] === source && args[args.indexOf('-o') + 1] === object);
    requireCondition(found.length === 1, `one current exact object compile: ${source}`);
    if (macro) same(found[0][1].filter(value => value.startsWith(`-D${macro}=`)).map(value => value.slice(macro.length + 3).replace(/^"|"$/g, '')), [scopes[scope].source_sha256], 'exact object compiler scope');
    commands[source] = found[0][0]; outputs.push(object);
  }
  const links = lines.filter(([, args]) => !args.includes('-c') && args.includes('-o') && args[args.indexOf('-o') + 1] === binary);
  requireCondition(links.length === 1 && outputs.every(object => links[0][1].filter(value => value === object).length === 1), 'all current objects linked once');
  return { commands, link: links[0][0] };
}
export function walk(root) {
  directory(root); const names = [];
  const visit = current => { for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
    const full = path.join(current, entry.name); requireCondition(!entry.isSymbolicLink(), 'artifact symlink forbidden');
    if (entry.isDirectory()) visit(full); else { requireCondition(entry.isFile(), 'regular artifact member'); names.push(full); }
  } }; visit(root); return names.sort((a, b) => path.relative(root, a) < path.relative(root, b) ? -1 : path.relative(root, a) > path.relative(root, b) ? 1 : 0);
}
export function inventory(target, protocol) {
  const names = walk(target).map(name => path.relative(target, name).split(path.sep).join('/')).sort();
  const files = records(names.map(name => path.join(target, name))).map((row, i) => ({ ...row, path: names[i] }));
  writeJSON(path.join(target, 'artifact-integrity.json'), { protocol, status: 'complete', files, files_count: files.length,
    total_bytes: files.reduce((sum, row) => sum + row.bytes, 0), excludes: ['artifact-integrity.json'] });
  return { status: 'complete', inventory_sha256: hash(path.join(target, 'artifact-integrity.json')), files: files.length };
}
export function cli(meta, run) {
  if (process.argv[1] !== fileURLToPath(meta)) return;
  try {
    requireCondition(process.platform === 'linux' && fs.existsSync('/.dockerenv') && process.cwd() === '/mityane', 'managed container /mityane required');
    const [action, ...tokens] = process.argv.slice(2), args = {};
    for (let i = 0; i < tokens.length; ++i) {
      requireCondition(tokens[i].startsWith('--'), 'explicit option required'); const name = tokens[i].slice(2).replaceAll('-', '_');
      if (name === 'sources') { args[name] = tokens.slice(i + 1); requireCondition(args[name].every(item => !item.startsWith('--')), 'sources must be final option'); break; }
      requireCondition(!Object.hasOwn(args, name) && i + 1 < tokens.length, 'unique valued option'); args[name] = tokens[++i];
    }
    const result = run(action, args); if (result !== undefined) process.stdout.write(JSON.stringify(result) + '\n');
  } catch (error) { process.stderr.write(`${error.message}\n`); process.exitCode = 1; }
}
