#!/usr/bin/env node
/** Read-only cleanup/evidence and maintained-document link checks.
 * Run through bash code/scripts/node.sh code/scripts/check-layout.mjs.
 * Only the recorded nonignored snapshot is hashed; ignored historical artifact
 * trees are never enumerated.
 */
import { createHash } from 'node:crypto';
import {
  closeSync, existsSync, openSync, readFileSync, readSync, readdirSync, realpathSync, statSync,
} from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
export const DEFAULT_RECEIPT = 'doc/EMBEDDINGS_CLEANUP_2026-10-18.json';
export const DEFAULT_RETIREMENT_RECEIPT = 'doc/PYTHON_MIGRATION_REMOVAL_2026-10-18.json';
export const HISTORICAL_ROOT = '.archive/embeddings/pre-cleanup-2026-10-18';
export const IGNORED_ARTIFACTS = Object.freeze(['output', '.build', '.external'].map(name => `code/embeddings/${name}`));
const REMOVED_COMPONENT_PATHS = ['code', 'doc', 'archive', 'container.ps1', 'setup.sh', 'dependencies.lock', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'Makefile'];
const htmlEntities = JSON.parse(readFileSync(new URL('./html5-entities.json', import.meta.url), 'utf8')).entities;
const numericReplacements = new Map([
  [0, '\ufffd'], [13, '\r'], ...Array.from('€\u0081‚ƒ„…†‡ˆ‰Š‹Œ\u008dŽ\u008f\u0090‘’“”•–—˜™š›œ\u009džŸ', (value, index) => [128 + index, value]),
]);

export class Checks {
  errors = [];
  files = 0;
  links = 0;
  relocations = 0;
  retirements = 0;
  shellRoots = 0;
  require(condition, message) {
    if (!condition) this.errors.push(message);
    return condition;
  }
}

function object(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function readText(source) { return readFileSync(source, 'utf8').replace(/^\ufeff/, ''); }
function inside(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return !path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`);
}
function relative(root, candidate) { return path.relative(root, candidate).split(path.sep).join('/'); }
function isDirectory(candidate) {
  try { return statSync(candidate).isDirectory(); } catch (error) { if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return false; throw error; }
}
function isFile(candidate) {
  try { return statSync(candidate).isFile(); } catch (error) { if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return false; throw error; }
}

export function repositoryPath(value, label, root = ROOT) {
  if (typeof value !== 'string' || !value) throw new Error(`${label} must be a nonempty repository-relative path`);
  let normalized = value.replaceAll('\\', '/');
  if (normalized.startsWith('/') || normalized.split('/').includes('..') || value.includes(':') || value.includes('\0')) {
    throw new Error(`${label} is not a safe repository-relative path: ${value}`);
  }
  // Dated receipts keep archive/... as evidence identities. Navigation alone
  // follows the directory rename; never rewrite the frozen receipt bytes.
  normalized = normalized.split('/').filter(part => part && part !== '.').join('/');
  if (normalized === 'archive' || normalized.startsWith('archive/')) normalized = `.archive${normalized.slice(7)}`;
  return path.join(root, normalized);
}

export function readReceipt(source, root = ROOT) {
  const seen = new Set();
  while (true) {
    if (seen.has(source)) throw new Error('cleanup detailed_manifest redirect contains a cycle');
    seen.add(source);
    const document = JSON.parse(readText(source));
    if (!object(document)) throw new Error(`cleanup receipt is not an object: ${source}`);
    if (Object.hasOwn(document, 'preserved_files')) return document;
    if (document.detailed_manifest == null) throw new Error('cleanup receipt has neither preserved_files nor detailed_manifest');
    source = repositoryPath(document.detailed_manifest, 'detailed_manifest', root);
  }
}

export function hashFile(source) {
  const digest = createHash('sha256');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  const fd = openSync(source, 'r');
  let size = 0;
  try {
    for (let length; (length = readSync(fd, buffer)) !== 0;) {
      digest.update(buffer.subarray(0, length));
      size += length;
    }
  } finally { closeSync(fd); }
  return [digest.digest('hex'), size];
}

function* walk(directory, prune = new Set()) {
  // Like os.walk(..., followlinks=False), visit ordinary directories only.
  const entries = readdirSync(directory, { withFileTypes: true });
  const directories = [];
  const files = [];
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory() || (entry.isSymbolicLink() && isDirectory(target))) {
      if (entry.isDirectory() && !prune.has(entry.name)) directories.push(target);
    } else files.push(target);
  }
  yield [directory, files];
  for (const child of directories) yield* walk(child, prune);
}

export function checkSnapshot(receipt, checks, { root = ROOT, log = console.log } = {}) {
  const historical = receipt.historical_root ?? HISTORICAL_ROOT;
  const archive = repositoryPath(historical, 'historical_root', root);
  if (!checks.require(isDirectory(archive), `historical snapshot missing: ${historical}`)) return;
  const entries = receipt.preserved_files;
  if (!Array.isArray(entries) || !entries.length) throw new Error('preserved_files must be a nonempty list');
  const expected = new Set();
  for (const item of entries) {
    if (!object(item)) throw new Error('each preserved_files entry must be an object');
    let candidate = repositoryPath(item.path, 'preserved file path', root);
    let name;
    if (inside(archive, candidate)) name = relative(archive, candidate);
    else {
      name = item.path.replaceAll('\\', '/');
      candidate = repositoryPath(`${relative(root, archive)}/${name}`, 'preserved file path', root);
    }
    if (!checks.require(!expected.has(name), `duplicate preserved file: ${name}`)) continue;
    expected.add(name);
    if (!Number.isSafeInteger(item.bytes) || item.bytes < 0) throw new Error(`invalid preserved byte count: ${name}`);
    if (typeof item.sha256 !== 'string' || !/^[0-9a-fA-F]{64}$/.test(item.sha256)) throw new Error(`invalid preserved SHA256: ${name}`);
    let digest, size;
    try { [digest, size] = hashFile(candidate); } catch (error) {
      checks.require(false, `cannot read preserved file: ${name}: ${error.message}`);
      continue;
    }
    checks.require(size === item.bytes, `preserved size differs: ${name}`);
    checks.require(digest === item.sha256.toLowerCase(), `preserved SHA256 differs: ${name}`);
    checks.files += 1;
    if (checks.files % 1000 === 0) log(`Snapshot hashes: ${checks.files}/${entries.length} files`);
  }
  log('Checking exact archived file inventory...');
  const actual = new Set();
  for (const [, files] of walk(archive)) for (const file of files) actual.add(relative(archive, file));
  for (const name of [...expected].filter(name => !actual.has(name)).sort()) checks.require(false, `snapshot inventory member missing: ${name}`);
  for (const name of [...actual].filter(name => !expected.has(name)).sort()) checks.require(false, `snapshot has an unrecorded file: ${name}`);
  checks.require(actual.size === entries.length, 'snapshot file count differs from preserved_files');
}

export function checkRetirements(document, checks, root = ROOT) {
  if (!object(document) || document.schemaVersion !== 1 || !Array.isArray(document.files) || !document.files.length) {
    throw new Error('retirement receipt must have schemaVersion 1 and a nonempty files list');
  }
  const retired = new Map();
  const archive = path.join(root, '.archive');
  for (const item of document.files) {
    if (!object(item) || !object(item.preservation)) throw new Error('each retirement must record a preserved source');
    const source = repositoryPath(item.path, 'retired path', root);
    const replacement = repositoryPath(item.replacement, 'replacement path', root);
    const preserved = repositoryPath(item.preservation.path, 'preserved retirement path', root);
    const name = relative(root, source);
    if (!name.endsWith('.py') || !(name.startsWith('code/') || name.startsWith('web/dashboard/scripts/'))) {
      throw new Error(`retirement is not a maintained Python source: ${name}`);
    }
    if (!inside(archive, preserved) || preserved === archive) throw new Error(`retirement preservation is outside the archive: ${name}`);
    if (!['baseline-original', 'removed-maintained-source'].includes(item.preservation.role)) {
      throw new Error(`invalid retirement preservation role: ${name}`);
    }
    for (const digest of [item.baselineSha256, item.removedSha256, item.preservation.sha256]) {
      if (typeof digest !== 'string' || !/^[0-9a-f]{64}$/.test(digest)) throw new Error(`invalid retirement SHA256: ${name}`);
    }
    if (!Number.isSafeInteger(item.removedBytes) || item.removedBytes < 0 || item.preservation.bytes !== item.removedBytes ||
        item.preservation.sha256 !== item.removedSha256) throw new Error(`retirement preservation does not bind removed bytes: ${name}`);
    if (!checks.require(!retired.has(name), `duplicate retired source: ${name}`)) continue;
    retired.set(name, item);
    checks.require(!existsSync(source), `retired maintained source remains: ${name}`);
    checks.require(source !== replacement && isFile(replacement), `retirement replacement missing: ${relative(root, replacement)}`);
    try {
      if (!inside(realpathSync(archive), realpathSync(preserved))) throw new Error('preserved source resolves outside the archive');
      const [digest, size] = hashFile(preserved);
      checks.require(digest === item.preservation.sha256, `retired source preservation SHA256 differs: ${name}`);
      checks.require(size === item.preservation.bytes, `retired source preservation size differs: ${name}`);
    } catch (error) { checks.require(false, `cannot read retired source preservation: ${name}: ${error.message}`); }
    checks.retirements += 1;
  }
  return retired;
}

export function checkRelocations(receipt, checks, root = ROOT) {
  // Keep the structural-cleanup receipt immutable. A later, hash-bound
  // retirement receipt explains destinations deliberately removed after porting.
  const retirementPath = path.join(root, DEFAULT_RETIREMENT_RECEIPT);
  const retired = isFile(retirementPath) ? checkRetirements(JSON.parse(readText(retirementPath)), checks, root) : new Map();
  const entries = receipt.relocations;
  if (!Array.isArray(entries) || !entries.length) throw new Error('relocations must be a nonempty list');
  for (const item of entries) {
    if (!object(item)) throw new Error('each relocation must be an object');
    const old = repositoryPath(item.old, 'old relocation path', root);
    const current = repositoryPath(item.new, 'new relocation path', root);
    checks.require(existsSync(current) || retired.has(relative(root, current)), `relocated destination missing: ${relative(root, current)}`);
    if (old !== current) checks.require(!existsSync(old), `retired source path remains: ${relative(root, old)}`);
    checks.relocations += 1;
  }
  for (const name of REMOVED_COMPONENT_PATHS) {
    const target = path.join(root, 'code/embeddings', name);
    checks.require(!existsSync(target), `redundant component path remains: ${relative(root, target)}`);
  }
}

export function checkIgnoreProtection(checks, root = ROOT) {
  const probes = IGNORED_ARTIFACTS.map(name => `${name}/__layout_ignore_probe__`);
  const result = spawnSync('git', ['-c', `safe.directory=${root}`, 'check-ignore', '--no-index', '--stdin'], {
    cwd: root, input: `${probes.join('\n')}\n`, encoding: 'utf8',
  });
  if (![0, 1].includes(result.status)) {
    checks.require(false, `git check-ignore failed: ${result.error?.message ?? result.stderr?.trim()}`);
    return;
  }
  const ignored = new Set(result.stdout.split(/\r?\n/));
  for (let index = 0; index < probes.length; index += 1) {
    checks.require(ignored.has(probes[index]), `historical artifact directory is not ignored: ${IGNORED_ARTIFACTS[index]}`);
    checks.require(isDirectory(path.join(root, IGNORED_ARTIFACTS[index])), `historical artifact directory is missing: ${IGNORED_ARTIFACTS[index]}`);
  }
}

export function htmlUnescape(text) {
  // HTML5 table and replacement rules match the predecessor's html.unescape,
  // including semicolon-free names, invalid code points and Windows-1252 C1.
  return text.replace(/&(#(?:[0-9]+|[xX][0-9a-fA-F]+);?|[^\t\n\f <&#;]{1,32};?)/g, (match, name) => {
    if (name.startsWith('#')) {
      const code = Number.parseInt(name.slice(/^#x/i.test(name) ? 2 : 1), /^#x/i.test(name) ? 16 : 10);
      if (numericReplacements.has(code)) return numericReplacements.get(code);
      if ((code >= 0xd800 && code <= 0xdfff) || code > 0x10ffff) return '\ufffd';
      if ((code >= 1 && code <= 8) || code === 11 || (code >= 14 && code <= 31) || (code >= 0x7f && code <= 0x9f) ||
          (code >= 0xfdd0 && code <= 0xfdef) || (code & 0xffff) >= 0xfffe) return '';
      return String.fromCodePoint(code);
    }
    if (Object.hasOwn(htmlEntities, name)) return htmlEntities[name];
    for (let length = name.length - 1; length > 1; length -= 1) {
      if (Object.hasOwn(htmlEntities, name.slice(0, length))) return htmlEntities[name.slice(0, length)] + name.slice(length);
    }
    return match;
  });
}

export function* markdownDestinations(text) {
  const lines = [];
  let fence = null;
  for (const line of text.match(/[^\n]*\n|[^\n]+$/g) ?? []) {
    const marker = /^ {0,3}(`{3,}|~{3,})/.exec(line);
    if (marker) {
      const run = marker[1];
      if (fence === null) fence = run;
      else if (run[0] === fence[0] && run.length >= fence.length) fence = null;
      lines.push(line.endsWith('\n') ? '\n' : '');
    } else if (fence === null) lines.push(line.replace(/(`+).*?\1/g, ''));
    else lines.push(line.endsWith('\n') ? '\n' : '');
  }
  const source = lines.join('');
  const lineAt = offset => source.slice(0, offset).split('\n').length;
  for (const match of source.matchAll(/\]\(\s*/g)) {
    const start = match.index + match[0].length;
    if (start >= source.length) continue;
    if (source[start] === '<') {
      const end = source.indexOf('>', start + 1);
      if (end !== -1) yield [lineAt(start), source.slice(start + 1, end)];
      continue;
    }
    let depth = 0, end = start;
    while (end < source.length) {
      const char = source[end];
      if (char === '\\' && end + 1 < source.length) { end += 2; continue; }
      if (/\s/u.test(char) || (char === ')' && depth === 0)) break;
      if (char === '(') depth += 1;
      else if (char === ')') depth -= 1;
      end += 1;
    }
    if (end > start) yield [lineAt(start), source.slice(start, end)];
  }
  for (const match of source.matchAll(/^ {0,3}\[[^\]\n]+\]:\s*(?:<([^>]+)>|(\S+))/gm)) {
    yield [lineAt(match.index), match[1] || match[2]];
  }
}

export function checkShellRoots(source, checks, root = ROOT) {
  const pattern = /^\s*cd\s+--\s+"\$\(dirname\s+--\s+"\$\{BASH_SOURCE\[0\]\}"\)\/((?:\.\.\/)*\.\.)"\s*$/;
  for (const [index, text] of readText(source).split(/\r?\n/).entries()) {
    const match = pattern.exec(text);
    if (!match) continue;
    checks.shellRoots += 1;
    const target = realpathSync(path.resolve(path.dirname(source), match[1]));
    checks.require(target === root, `shell repository root differs: ${relative(root, source)}:${index + 1}: ${match[1]} resolves to ${target}`);
  }
}

export function checkMarkdownLinks(checks, root = ROOT) {
  const sources = [], existence = new Map();
  const prune = new Set(['.git', '.build', '.external', 'output', '__pycache__', 'archive', '.archive']);
  for (const [, files] of walk(path.join(root, 'code'), prune)) {
    for (const file of files) {
      if (file.toLowerCase().endsWith('.md')) sources.push(file);
      if (file.toLowerCase().endsWith('.sh')) checkShellRoots(file, checks, root);
    }
  }
  sources.push(path.join(root, 'doc/EMBEDDINGS.md'));
  for (const source of sources.sort()) {
    if (!checks.require(isFile(source), `maintained document missing: ${relative(root, source)}`)) continue;
    for (let [line, destination] of markdownDestinations(readText(source))) {
      destination = htmlUnescape(destination.replace(/\\([()<> ])/g, '$1'));
      if (!destination || destination.startsWith('#')) continue;
      let local = destination.split('#', 1)[0].split('?', 1)[0];
      local = local.replace(/(?:%[0-9a-fA-F]{2})+/g, encoded => Buffer.from(encoded.replaceAll('%', ''), 'hex').toString('utf8'));
      local = local.replace(/:\d+(?::\d+)?$/, '');
      if (!local || /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(local)) continue;
      const target = local.startsWith('/') ? path.resolve(root, local.replace(/^\/+/, '')) : path.resolve(path.dirname(source), local);
      if (inside(path.join(root, '.archive'), target)) continue;
      checks.links += 1;
      if (!existence.has(target)) existence.set(target, existsSync(target));
      checks.require(existence.get(target), `broken Markdown link: ${relative(root, source)}:${line}: ${destination}`);
    }
  }
}

export function main(args = process.argv.slice(2)) {
  if (!isFile('/.dockerenv') || ROOT !== '/mityane') {
    console.error('Run through the managed mityane-geometrics container.');
    return 2;
  }
  const { values } = parseArgs({ args, options: {
    receipt: { type: 'string', default: DEFAULT_RECEIPT },
    'links-only': { type: 'boolean', default: false }, help: { type: 'boolean', short: 'h' },
  } });
  if (values.help) {
    console.log('Usage: check-layout.mjs [--receipt PATH] [--links-only]\nRead-only cleanup/evidence and maintained-document link checks.\n--links-only checks ignores, links and shell roots without hashing or relocation checks.');
    return 0;
  }
  const checks = new Checks();
  try {
    if (!values['links-only']) {
      const receipt = readReceipt(repositoryPath(values.receipt, 'receipt'));
      console.log('Checking archived original bytes and SHA256 identities...');
      checkSnapshot(receipt, checks);
      console.log('Checking relocated sources and removed component configuration...');
      checkRelocations(receipt, checks);
    }
    console.log('Checking historical artifact ignore protection...');
    checkIgnoreProtection(checks);
    console.log('Checking maintained Markdown links and literal shell roots...');
    checkMarkdownLinks(checks);
  } catch (error) { checks.errors.push(error.message); }
  if (checks.errors.length) {
    for (const error of checks.errors) console.error(`FAIL ${error}`);
    console.error(`Layout verification failed with ${checks.errors.length} problem(s).`);
    return 1;
  }
  const summary = `${checks.links} maintained Markdown links, ${checks.shellRoots} literal shell roots; historical artifact trees remain ignored.`;
  console.log(values['links-only'] ? `Links verified: ${summary}` : `Layout verified: ${checks.files} preserved files, ${checks.relocations} relocations, ${checks.retirements} later source retirements, ${summary}`);
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { process.exitCode = main(); } catch (error) { console.error(error.message); process.exitCode = 2; }
}
