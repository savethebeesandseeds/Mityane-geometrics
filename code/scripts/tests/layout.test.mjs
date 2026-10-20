import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import {
  Checks, DEFAULT_RETIREMENT_RECEIPT, checkMarkdownLinks, checkRelocations, checkRetirements, checkShellRoots, checkSnapshot,
  htmlUnescape, markdownDestinations, readReceipt, repositoryPath,
} from '../check-layout.mjs';

const scratch = '/opt/mityane-geometrics/maintenance-tests';
mkdirSync(scratch, { recursive: true });
function fixture(t) {
  const root = mkdtempSync(`${scratch}/layout-`);
  t.after(() => rmSync(root, { recursive: true, force: true }));
  return root;
}
function write(root, name, contents) {
  const target = path.join(root, name);
  mkdirSync(path.dirname(target), { recursive: true });
  writeFileSync(target, contents);
  return target;
}
function entry(name, contents) {
  return { path: name, bytes: Buffer.byteLength(contents), sha256: createHash('sha256').update(contents).digest('hex') };
}
const quiet = () => {};

test('repository paths reject traversal, absolute and drive paths and preserve archive navigation mapping', t => {
  const root = fixture(t);
  for (const bad of ['', null, 4, '/etc/passwd', '../outside', 'nested/../outside', 'C:\\outside', '\\outside', 'x\0y']) {
    assert.throws(() => repositoryPath(bad, 'path', root), /nonempty|safe repository-relative/);
  }
  assert.equal(repositoryPath('archive\\embeddings/receipt.json', 'path', root), path.join(root, '.archive/embeddings/receipt.json'));
  assert.equal(repositoryPath('./code//README.md', 'path', root), path.join(root, 'code/README.md'));
});

test('receipt redirects handle BOM, historical paths, cycles and malformed records', t => {
  const root = fixture(t);
  const source = write(root, 'summary.json', '\ufeff' + JSON.stringify({ detailed_manifest: 'archive/detail.json' }));
  write(root, '.archive/detail.json', JSON.stringify({ preserved_files: [entry('file', 'bytes')] }));
  assert.equal(readReceipt(source, root).preserved_files.length, 1);
  write(root, '.archive/detail.json', JSON.stringify({ detailed_manifest: 'summary.json' }));
  assert.throws(() => readReceipt(source, root), /cycle/);
  write(root, '.archive/detail.json', '[]');
  assert.throws(() => readReceipt(source, root), /not an object/);
  write(root, '.archive/detail.json', JSON.stringify({ detailed_manifest: '../outside' }));
  assert.throws(() => readReceipt(source, root), /safe repository-relative/);
});

test('snapshot accepts both path forms and checks exact bytes and exact inventory', t => {
  const root = fixture(t);
  write(root, '.archive/frozen/a', 'a\r\n');
  write(root, '.archive/frozen/nested/b', 'β');
  const receipt = { historical_root: 'archive/frozen', preserved_files: [entry('a', 'a\r\n'), entry('archive/frozen/nested/b', 'β')] };
  const checks = new Checks();
  checkSnapshot(receipt, checks, { root, log: quiet });
  assert.equal(checks.files, 2);
  assert.deepEqual(checks.errors, []);
  write(root, '.archive/frozen/a', 'changed');
  write(root, '.archive/frozen/unrecorded', 'new');
  rmSync(path.join(root, '.archive/frozen/nested/b'));
  const damaged = new Checks();
  checkSnapshot(receipt, damaged, { root, log: quiet });
  assert.ok(damaged.errors.some(value => value === 'preserved size differs: a'));
  assert.ok(damaged.errors.some(value => value === 'preserved SHA256 differs: a'));
  assert.ok(damaged.errors.some(value => value === 'snapshot inventory member missing: nested/b'));
  assert.ok(damaged.errors.some(value => value === 'snapshot has an unrecorded file: unrecorded'));
});

test('snapshot rejects invalid and duplicate evidence records', t => {
  const root = fixture(t);
  write(root, '.archive/frozen/a', 'x');
  const valid = entry('a', 'x');
  for (const malformed of [{ ...valid, bytes: true }, { ...valid, bytes: -1 }, { ...valid, sha256: 'bad' }]) {
    assert.throws(() => checkSnapshot({ historical_root: '.archive/frozen', preserved_files: [malformed] }, new Checks(), { root, log: quiet }), /invalid preserved/);
  }
  const checks = new Checks();
  checkSnapshot({ historical_root: '.archive/frozen', preserved_files: [valid, valid] }, checks, { root, log: quiet });
  assert.ok(checks.errors.includes('duplicate preserved file: a'));
  assert.ok(checks.errors.includes('snapshot file count differs from preserved_files'));
});

test('relocations require the new destination and absent retired source/configuration', t => {
  const root = fixture(t);
  write(root, 'current', 'x');
  const receipt = { relocations: [{ old: 'retired', new: 'current' }] };
  const checks = new Checks();
  checkRelocations(receipt, checks, root);
  assert.deepEqual(checks.errors, []);
  write(root, 'retired', 'x');
  write(root, 'code/embeddings/Makefile', 'old');
  rmSync(path.join(root, 'current'));
  const bad = new Checks();
  checkRelocations(receipt, bad, root);
  assert.deepEqual(bad.errors, ['relocated destination missing: current', 'retired source path remains: retired', 'redundant component path remains: code/embeddings/Makefile']);
});

function retirement(root, contents = 'retired Python source\n') {
  const preserved = entry('.archive/python-migration/retired/code/old.py', contents);
  write(root, preserved.path, contents);
  write(root, 'code/new.mjs', 'export {};\n');
  return {
    path: 'code/old.py', baselineSha256: preserved.sha256, removedSha256: preserved.sha256,
    removedBytes: preserved.bytes, replacement: 'code/new.mjs',
    preservation: { ...preserved, role: 'removed-maintained-source' },
  };
}

test('later source retirements preserve exact removed bytes without rewriting the relocation receipt', t => {
  const root = fixture(t), item = retirement(root);
  write(root, DEFAULT_RETIREMENT_RECEIPT, JSON.stringify({ schemaVersion: 1, files: [item] }));
  const receipt = { relocations: [{ old: 'code/embeddings/code/old.py', new: item.path }] };
  const before = JSON.stringify(receipt), checks = new Checks();
  checkRelocations(receipt, checks, root);
  assert.deepEqual(checks.errors, []);
  assert.equal(checks.relocations, 1);
  assert.equal(checks.retirements, 1);
  assert.equal(JSON.stringify(receipt), before);
  const unrelated = new Checks();
  checkRelocations({ relocations: [{ old: 'other', new: 'code/unrecorded.py' }] }, unrelated, root);
  assert.deepEqual(unrelated.errors, ['relocated destination missing: code/unrecorded.py']);
});

test('source retirement detects leftover working copies, absent replacements and damaged preservation', t => {
  const root = fixture(t), item = retirement(root);
  write(root, item.path, 'still present');
  rmSync(path.join(root, item.replacement));
  write(root, item.preservation.path, 'changed');
  const checks = new Checks();
  checkRetirements({ schemaVersion: 1, files: [item] }, checks, root);
  assert.deepEqual(checks.errors, [
    'retired maintained source remains: code/old.py', 'retirement replacement missing: code/new.mjs',
    'retired source preservation SHA256 differs: code/old.py', 'retired source preservation size differs: code/old.py',
  ]);
  rmSync(path.join(root, item.preservation.path));
  const missing = new Checks();
  checkRetirements({ schemaVersion: 1, files: [item] }, missing, root);
  assert.ok(missing.errors.some(message => message.startsWith('cannot read retired source preservation: code/old.py:')));
});

test('retirement records reject unbound bytes, unsafe archive paths and duplicate sources', t => {
  const root = fixture(t), item = retirement(root);
  for (const malformed of [
    { ...item, removedBytes: true },
    { ...item, removedSha256: 'bad' },
    { ...item, preservation: { ...item.preservation, sha256: '0'.repeat(64) } },
    { ...item, preservation: { ...item.preservation, path: 'code/not-archived.py' } },
    { ...item, preservation: { ...item.preservation, role: 'unverified-reference' } },
    { ...item, path: 'code/current.mjs' },
  ]) {
    assert.throws(() => checkRetirements({ schemaVersion: 1, files: [malformed] }, new Checks(), root), /retirement/);
  }
  for (const malformed of [{}, { schemaVersion: 2, files: [item] }, { schemaVersion: 1, files: [] }]) {
    assert.throws(() => checkRetirements(malformed, new Checks(), root), /retirement receipt/);
  }
  const checks = new Checks();
  checkRetirements({ schemaVersion: 1, files: [item, item] }, checks, root);
  assert.deepEqual(checks.errors, ['duplicate retired source: code/old.py']);
});

test('Markdown parser skips code and preserves escaped/parenthesized/reference paths and line numbers', () => {
  const source = '[one](a(b).md)\n`[skip](no)`\n```md\n[skip](no)\n```\n[two](<two words.md>)\n[three](escaped\\(name\\).md "title")\n[ref]: reference.md\n';
  assert.deepEqual([...markdownDestinations(source)], [[1, 'a(b).md'], [6, 'two words.md'], [7, 'escaped\\(name\\).md'], [8, 'reference.md']]);
});

test('HTML decoder handles HTML5 named aliases, longest prefixes and numeric invalid/replacement rules', () => {
  assert.equal(htmlUnescape('&amp;&NotEqualTilde;&notit;&doesnotexist;'), '&≂̸¬it;&doesnotexist;');
  assert.equal(htmlUnescape('&#0;&#128;&#13;&#x1f600;&#xD800;&#1114112;&#1;&#xFFFF;'), '\ufffd€\r😀\ufffd\ufffd');
});

test('HTML5 decoder preserves the Python golden across every named entity and numeric boundary', () => {
  const { entities } = JSON.parse(readFileSync(new URL('../html5-entities.json', import.meta.url), 'utf8'));
  const numbers = [...Array.from({ length: 160 }, (_, index) => index), 0xd800, 0xdfff, 0x10ffff, 0x110000,
    ...Array.from({ length: 32 }, (_, index) => 0xfdd0 + index)];
  const cases = [...Object.keys(entities).sort().map(name => `&${name}`), ...numbers.map(value => `&#${value};`),
    '&notit;', '&unknown;', '&ampnot', '&', '&#;', '&#x;'];
  const digest = createHash('sha256');
  for (const value of cases) digest.update(htmlUnescape(value)).update(Buffer.from([0]));
  // Captured once from the predecessor's Python html.unescape implementation.
  assert.equal(digest.digest('hex'), '40e0fd3edaf8373afca78f597765948920f25633c5b3079ad5e4ddc589e40ef7');
});

test('links normalize escapes, entities, percent paths and source line suffixes while pruning ignored trees', t => {
  const root = fixture(t);
  write(root, 'code/links.md', '[space](two%20words.md:12:4#part)\n[escaped](a\\(b\\).md)\n[entity](a&amp;b.md)\n[remote](https://example.com/no)\n[archive](../.archive/missing)\n[anchor](#local)\n[broken](missing.md)\n');
  write(root, 'code/two words.md', '');
  write(root, 'code/a(b).md', '');
  write(root, 'code/a&b.md', '');
  write(root, 'code/output/ignored.md', '[bad](missing)');
  write(root, 'code/.archive/ignored.md', '[bad](missing)');
  write(root, 'doc/EMBEDDINGS.md', '[root](/code/two%20words.md?query)');
  const checks = new Checks();
  checkMarkdownLinks(checks, root);
  assert.equal(checks.links, 5);
  assert.deepEqual(checks.errors, ['broken Markdown link: code/links.md:7: missing.md']);
});

test('literal shell roots are checked without executing shell code', t => {
  const root = fixture(t);
  const source = write(root, 'code/scripts/runner.sh', 'cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."\ncd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."\necho dangerous_command\n');
  const checks = new Checks();
  checkShellRoots(source, checks, root);
  assert.equal(checks.shellRoots, 2);
  assert.equal(checks.errors.length, 1);
  assert.match(checks.errors[0], /^shell repository root differs: code\/scripts\/runner.sh:2:/);
});
