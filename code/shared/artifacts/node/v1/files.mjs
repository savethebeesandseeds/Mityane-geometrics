// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Versioned artifact implementation; engineering validation uses the container.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';

export const JSON_ENCODING = 'mityane-artifact-json-v1';
export function requireCondition(value, message) {
  if (!value) throw new Error(message);
}
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
export function inside(root, filename) {
  const relative = path.relative(root, filename);
  return !path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`);
}
export function safeRelative(value) {
  requireCondition(typeof value === 'string' && value.length > 0 &&
    !/[\\:\0\t\r\n]/.test(value) && !path.posix.isAbsolute(value) &&
    value.split('/').every(part => part && part !== '.' && part !== '..'), 'safe exact relative artifact path');
  return value;
}
export function canonicalDirectory(directory, {root} = {}) {
  requireCondition(typeof directory === 'string' && path.isAbsolute(directory) &&
    path.normalize(directory) === directory, 'absolute canonical directory required');
  const info = fs.lstatSync(directory);
  requireCondition(info.isDirectory() && !info.isSymbolicLink() &&
    fs.realpathSync(directory) === directory, `directory redirect or conflict: ${directory}`);
  for (let parent = path.dirname(directory); ; parent = path.dirname(parent)) {
    requireCondition(!fs.lstatSync(parent).isSymbolicLink(), `redirected directory ancestor: ${parent}`);
    if (parent === path.dirname(parent)) break;
  }
  if (root !== undefined) requireCondition(inside(root, directory), `directory escapes admitted root: ${directory}`);
  return directory;
}
function signature(info) {
  return [info.dev, info.ino, info.size, info.mode, info.mtimeNs, info.ctimeNs];
}
function sameSignature(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

// Admit the complete matrix before any caller hashes or reads member payloads.
// Filesystem metadata is read here; content bytes are not.
export function admitFiles(filenames, {root, singleLink = false, rejectAliases = true} = {}) {
  requireCondition(Array.isArray(filenames), 'explicit file matrix required');
  if (root !== undefined) canonicalDirectory(root);
  const names = new Set(), inodes = new Set();
  return filenames.map(filename => {
    requireCondition(typeof filename === 'string' && path.isAbsolute(filename) &&
      path.normalize(filename) === filename && !/[\0\t\r\n]/.test(filename), 'absolute exact file binding required');
    requireCondition(!names.has(filename), `duplicate admitted path: ${filename}`);
    names.add(filename);
    if (root !== undefined) requireCondition(inside(root, filename), `file escapes admitted root: ${filename}`);
    canonicalDirectory(path.dirname(filename));
    const info = fs.lstatSync(filename, {bigint: true});
    requireCondition(info.isFile() && !info.isSymbolicLink() && fs.realpathSync(filename) === filename,
      `regular canonical file required: ${filename}`);
    requireCondition(!singleLink || info.nlink === 1n, `single-link file required: ${filename}`);
    requireCondition(info.size <= BigInt(Number.MAX_SAFE_INTEGER), `file byte count exceeds exact range: ${filename}`);
    const inode = `${info.dev}:${info.ino}`;
    requireCondition(!rejectAliases || !inodes.has(inode), `aliased admitted file: ${filename}`);
    inodes.add(inode);
    return Object.freeze({path: filename, bytes: Number(info.size), mode: Number(info.mode & 0o7777n),
      signature: Object.freeze(signature(info)), singleLink});
  });
}
function validateAdmission(row) {
  canonicalDirectory(path.dirname(row.path));
  const info = fs.lstatSync(row.path, {bigint: true});
  requireCondition(info.isFile() && !info.isSymbolicLink() && fs.realpathSync(row.path) === row.path &&
    sameSignature(signature(info), row.signature) && (!row.singleLink || info.nlink === 1n),
  `file changed after admission: ${row.path}`);
}
function withAdmittedFile(row, operation) {
  validateAdmission(row);
  const descriptor = fs.openSync(row.path, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    requireCondition(sameSignature(signature(fs.fstatSync(descriptor, {bigint: true})), row.signature),
      `file redirected before read: ${row.path}`);
    const result = operation(descriptor);
    requireCondition(sameSignature(signature(fs.fstatSync(descriptor, {bigint: true})), row.signature),
      `file changed during read: ${row.path}`);
    validateAdmission(row);
    return result;
  } finally { fs.closeSync(descriptor); }
}
export function readAdmitted(row) {
  return withAdmittedFile(row, descriptor => {
    const bytes = fs.readFileSync(descriptor);
    requireCondition(bytes.length === row.bytes, `admitted byte count differs: ${row.path}`);
    return bytes;
  });
}
export function hashAdmitted(row) {
  return withAdmittedFile(row, descriptor => {
    const hash = createHash('sha256'), buffer = Buffer.allocUnsafe(1024 * 1024);
    let count = 0;
    for (;;) {
      const length = fs.readSync(descriptor, buffer, 0, buffer.length, null);
      if (!length) break;
      hash.update(buffer.subarray(0, length)); count += length;
    }
    requireCondition(count === row.bytes, `admitted byte count differs: ${row.path}`);
    return hash.digest('hex');
  });
}
export function fileSHA256(filename, options = {}) {
  return hashAdmitted(admitFiles([filename], options)[0]);
}

// Safe integers remain Number. Unsafe integer tokens become BigInt, or reject
// under integers='number'. Never round an identity or seed through JS Number.
// Duplicate policy is explicit: legacy json.loads behavior is 'last'.
export function parseJSON(raw, {duplicates = 'last', integers = 'bigint', allowBOM = true} = {}) {
  requireCondition(['last', 'reject'].includes(duplicates) && ['bigint', 'number'].includes(integers),
    'explicit supported JSON compatibility policy required');
  let source = typeof raw === 'string' ? raw : new TextDecoder('utf-8', {fatal: true, ignoreBOM: true}).decode(raw);
  if (allowBOM) source = source.replace(/^\uFEFF/, '');
  let index = 0;
  const fail = () => { throw new Error(`invalid artifact JSON at character ${index}`); };
  const space = () => { while (/[ \t\r\n]/.test(source[index] ?? '\0')) index += 1; };
  const string = () => {
    const start = index++;
    while (index < source.length) {
      if (source[index] === '\\') { index += 2; continue; }
      if (source[index++] === '"') return JSON.parse(source.slice(start, index));
    }
    return fail();
  };
  const value = () => {
    space();
    if (source[index] === '"') return string();
    if (source[index] === '{') {
      const result = {};
      index += 1; space();
      if (source[index] === '}') { index += 1; return result; }
      for (;;) {
        if (source[index] !== '"') return fail();
        const key = string();
        requireCondition(duplicates === 'last' || !Object.hasOwn(result, key), `duplicate JSON key: ${key}`);
        space(); if (source[index++] !== ':') return fail();
        Object.defineProperty(result, key, {value: value(), enumerable: true, configurable: true, writable: true});
        space(); const delimiter = source[index++];
        if (delimiter === '}') return result;
        if (delimiter !== ',') return fail();
        space();
      }
    }
    if (source[index] === '[') {
      const result = [];
      index += 1; space();
      if (source[index] === ']') { index += 1; return result; }
      for (;;) {
        result.push(value()); space(); const delimiter = source[index++];
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
    if (!/[.eE]/.test(match[0])) {
      const integer = BigInt(match[0]);
      if (integer < BigInt(Number.MIN_SAFE_INTEGER) || integer > BigInt(Number.MAX_SAFE_INTEGER)) {
        requireCondition(integers === 'bigint', 'unsafe integer in number-only artifact JSON');
        return integer;
      }
    }
    const number = Number(match[0]);
    requireCondition(Number.isFinite(number), 'nonfinite artifact JSON number');
    return number;
  };
  const result = value(); space();
  requireCondition(index === source.length, 'trailing artifact JSON content');
  return result;
}
export function readJSON(filename, options = {}) {
  const {root, singleLink = false, ...jsonOptions} = options;
  return parseJSON(readAdmitted(admitFiles([filename], {root, singleLink})[0]), jsonOptions);
}
function lexical(left, right) {
  const a = Array.from(left, char => char.codePointAt(0));
  const b = Array.from(right, char => char.codePointAt(0));
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
}
// New, named artifact encoding. Integer/string-only records can match Python's
// pretty JSON, but floating lexeme/format parity is not claimed by this API.
export function jsonBytes(value, {sortKeys = true, indent = 2, ensureAscii = true} = {}) {
  requireCondition(Number.isInteger(indent) && indent >= 0 && indent <= 16, 'bounded JSON indent');
  const ancestors = new Set(), padding = depth => ' '.repeat(indent * depth);
  const quote = text => {
    const encoded = JSON.stringify(text);
    return ensureAscii ? encoded.replace(/[\u007f-\uffff]/g,
      char => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`) : encoded;
  };
  const encode = (item, depth) => {
    if (item === null) return 'null';
    if (typeof item === 'string') return quote(item);
    if (typeof item === 'boolean' || typeof item === 'bigint') return String(item);
    if (typeof item === 'number') {
      requireCondition(Number.isFinite(item) && (!Number.isInteger(item) || Number.isSafeInteger(item)),
        'finite exact JSON number required; use BigInt for large integers');
      return Object.is(item, -0) ? '-0.0' : JSON.stringify(item);
    }
    requireCondition(typeof item === 'object' && (Array.isArray(item) ||
      Object.getPrototypeOf(item) === Object.prototype || Object.getPrototypeOf(item) === null), 'plain artifact JSON value required');
    requireCondition(!ancestors.has(item), 'cyclic artifact JSON');
    ancestors.add(item);
    try {
      const array = Array.isArray(item), keys = Object.keys(item);
      if (!array && sortKeys) keys.sort(lexical);
      const values = array ? Array.from(item, child => encode(child, depth + 1)) :
        keys.map(key => `${quote(key)}: ${encode(item[key], depth + 1)}`);
      if (!values.length) return array ? '[]' : '{}';
      return `${array ? '[' : '{'}\n${padding(depth + 1)}${values.join(`,\n${padding(depth + 1)}`)}\n${padding(depth)}${array ? ']' : '}'}`;
    } finally { ancestors.delete(item); }
  };
  return Buffer.from(`${encode(value, 0)}\n`, 'utf8');
}
export function writeNew(filename, data, {mode = 0o644} = {}) {
  requireCondition(typeof filename === 'string' && path.isAbsolute(filename) &&
    path.normalize(filename) === filename, 'absolute exclusive artifact output required');
  canonicalDirectory(path.dirname(filename));
  const bytes = Buffer.isBuffer(data) ? data : Buffer.from(data, 'utf8');
  const descriptor = fs.openSync(filename,
    fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW, mode);
  try {
    let offset = 0;
    while (offset < bytes.length) {
      const length = fs.writeSync(descriptor, bytes, offset, bytes.length - offset);
      requireCondition(length > 0, 'short artifact write'); offset += length;
    }
    fs.fsyncSync(descriptor);
  } finally { fs.closeSync(descriptor); }
  canonicalDirectory(path.dirname(filename));
  const parent = fs.openSync(path.dirname(filename), fs.constants.O_RDONLY | fs.constants.O_DIRECTORY);
  try { fs.fsyncSync(parent); } finally { fs.closeSync(parent); }
  return sha256(bytes);
}
export function writeJSON(filename, value, options = {}) {
  return writeNew(filename, jsonBytes(value, options));
}
export function createExclusiveDirectory(directory, {root} = {}) {
  requireCondition(path.isAbsolute(directory) && path.normalize(directory) === directory,
    'absolute exclusive artifact directory required');
  if (root !== undefined) {
    canonicalDirectory(root);
    requireCondition(directory !== root && inside(root, directory), 'exclusive output must be inside admitted root');
  }
  const missing = [];
  let existing = path.dirname(directory);
  while (!fs.existsSync(existing)) { missing.unshift(existing); existing = path.dirname(existing); }
  canonicalDirectory(existing);
  for (const parent of missing) { fs.mkdirSync(parent); canonicalDirectory(parent); }
  canonicalDirectory(path.dirname(directory));
  fs.mkdirSync(directory);
  return canonicalDirectory(directory, {root});
}
