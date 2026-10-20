// Copyright (c) 2026 Waajacu. Artificial fixtures for the managed container.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {
  admitFiles, canonicalDirectory, createExclusiveDirectory, hashAdmitted,
  jsonBytes, parseJSON, readAdmitted, safeRelative, sha256, writeNew,
} from '../files.mjs';

function fixture(t) {
  assert.equal(process.platform, 'linux', 'fixtures run only in the managed Linux container');
  assert.ok(fs.existsSync('/.dockerenv'));
  const parent = '/opt/mityane-geometrics/fixtures/node-artifacts-v1';
  fs.mkdirSync(parent, {recursive: true});
  const root = fs.mkdtempSync(path.join(parent, 'case-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return root;
}
test('JSON preserves duplicate-key policy, exact uint64 integers and safe prototype keys', () => {
  const document = parseJSON('{"n":18446744073709551615,"n2":42,"k":1,"k":2,"__proto__":{"x":1}}');
  assert.equal(document.n, 18446744073709551615n);
  assert.equal(document.n2, 42);
  assert.equal(document.k, 2);
  assert.ok(Object.hasOwn(document, '__proto__'));
  assert.equal({}.x, undefined);
  assert.throws(() => parseJSON('{"k":1,"\\u006b":2}', {duplicates: 'reject'}), /duplicate/);
  assert.throws(() => parseJSON('{"n":18446744073709551615}', {integers: 'number'}), /unsafe integer/);
  assert.throws(() => parseJSON('{"n":1e999}'), /nonfinite/);
  assert.deepEqual(parseJSON(Buffer.from('\uFEFF{"n":1}')), {n: 1});
  assert.throws(() => parseJSON(Buffer.from('\uFEFF{"n":1}'), {allowBOM: false}), /invalid/);
  for (const invalid of ['[1,]', '{"x":1,}', '01', 'true false', 'NaN']) assert.throws(() => parseJSON(invalid));
});
test('new artifact serialization has a recorded integer/string golden and explicit float limits', () => {
  const bytes = jsonBytes({z: 18446744073709551615n, a: 'é\n', empty: [], flag: false});
  assert.equal(bytes.toString(), '{\n  "a": "\\u00e9\\n",\n  "empty": [],\n  "flag": false,\n  "z": 18446744073709551615\n}\n');
  assert.deepEqual(parseJSON(bytes), {a: 'é\n', empty: [], flag: false, z: 18446744073709551615n});
  assert.equal(jsonBytes({z: 1, a: 2}, {sortKeys: false}).toString(), '{\n  "z": 1,\n  "a": 2\n}\n');
  assert.throws(() => jsonBytes({n: Infinity}), /finite/);
  assert.throws(() => jsonBytes({n: Number.MAX_SAFE_INTEGER + 1}), /exact/);
  const cyclic = {}; cyclic.self = cyclic;
  assert.throws(() => jsonBytes(cyclic), /cyclic/);
});
test('whole-matrix admission rejects missing, duplicate, aliased and redirected paths before reads', t => {
  const root = fixture(t), first = path.join(root, 'first'), second = path.join(root, 'second');
  fs.writeFileSync(first, 'first'); fs.writeFileSync(second, 'second');
  assert.throws(() => admitFiles([first, path.join(root, 'missing')], {root}));
  assert.throws(() => admitFiles([first, first], {root}), /duplicate/);
  const alias = path.join(root, 'alias'); fs.linkSync(first, alias);
  assert.throws(() => admitFiles([first, alias], {root}), /aliased/);
  assert.throws(() => admitFiles([first], {root, singleLink: true}), /single-link/);
  fs.unlinkSync(alias);
  const link = path.join(root, 'redirect'); fs.symlinkSync(first, link);
  assert.throws(() => admitFiles([link], {root}), /canonical/);
  fs.mkdirSync(path.join(root, 'real')); fs.symlinkSync('real', path.join(root, 'indirect'));
  assert.throws(() => canonicalDirectory(path.join(root, 'indirect')), /redirect/);
  assert.equal(fs.readFileSync(first, 'utf8'), 'first');
});
test('admitted hashes retain bytes and reject identity, timestamp or permission changes', t => {
  const root = fixture(t), filename = path.join(root, 'payload');
  const bytes = Buffer.from([0, 1, 0xff, 0, 13, 10]);
  fs.writeFileSync(filename, bytes);
  let [admitted] = admitFiles([filename], {root, singleLink: true});
  assert.equal(hashAdmitted(admitted), sha256(bytes));
  assert.deepEqual(readAdmitted(admitted), bytes);
  fs.writeFileSync(filename, Buffer.from([1, 1, 0xff, 0, 13, 10]));
  // Immediate same-size writes can share mtime/ctime even with nanosecond stat
  // fields. Exercise the promised metadata-change guard deterministically.
  const changedTime = Number(admitted.signature[4] / 1_000_000_000n + 2n);
  fs.utimesSync(filename, changedTime, changedTime);
  assert.throws(() => hashAdmitted(admitted), /changed/);
  [admitted] = admitFiles([filename], {root});
  fs.renameSync(filename, path.join(root, 'old')); fs.writeFileSync(filename, 'replacement');
  assert.throws(() => readAdmitted(admitted), /changed/);
  fs.chmodSync(filename, 0o644);
  [admitted] = admitFiles([filename], {root}); fs.chmodSync(filename, 0o600);
  assert.throws(() => hashAdmitted(admitted), /changed/);
});
test('exclusive durable output refuses replacement and creates only admitted directories', t => {
  const root = fixture(t), destination = path.join(root, 'new', 'leaf');
  createExclusiveDirectory(destination, {root});
  const record = path.join(destination, 'record.json'), bytes = jsonBytes({ok: true});
  assert.equal(writeNew(record, bytes), sha256(bytes));
  assert.throws(() => writeNew(record, 'changed'));
  assert.deepEqual(fs.readFileSync(record), bytes);
  assert.throws(() => createExclusiveDirectory(destination, {root}));
  assert.throws(() => createExclusiveDirectory(path.join(root, '..', 'outside'), {root}));
  for (const value of ['../x', '/x', 'a//b', 'a/./b', 'a\\b', 'x\n']) assert.throws(() => safeRelative(value));
});
