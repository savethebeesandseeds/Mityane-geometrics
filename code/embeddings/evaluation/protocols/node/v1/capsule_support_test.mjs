// Artificial source/evidence fixtures. Not executed during Docker recovery.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { captureSources, lines, sha, mkdir, retainedAdmissions, finalize } from './capsule_support.mjs';
import { sha256 } from '../../../../../shared/artifacts/node/v1/files.mjs';
import { createCapsule as learned } from '../../configurable_learned_relation_v1/capsule.mjs';
import { createCapsule as content } from '../../configurable_content_relation_v1/capsule.mjs';

function fixture(t) {
  assert.equal(fs.existsSync('/.dockerenv'), true, 'run only inside managed Linux container');
  const base = '/opt/mityane-geometrics/python-migration-fixtures'; fs.mkdirSync(base, { recursive: true });
  const repositoryRoot = fs.mkdtempSync(path.join(base, 'capsule-capture-'));
  t.after(() => fs.rmSync(repositoryRoot, { recursive: true, force: true }));
  const root = path.join(repositoryRoot, 'new-capsule'), buildDirectory = path.join(repositoryRoot, 'build');
  fs.mkdirSync(root); fs.mkdirSync(buildDirectory);
  fs.writeFileSync(path.join(repositoryRoot, 'entry.mjs'), '// artificial source A\n');
  fs.writeFileSync(path.join(repositoryRoot, 'support.mjs'), '// artificial source B\n');
  fs.writeFileSync(path.join(buildDirectory, 'reader'), 'opaque non-executable source-id fixture\n');
  const names = ['entry.mjs', 'support.mjs'], sourceList = `${names.join('\n')}\n`;
  const sourceID = sha256(Buffer.from(names.map(name => `${sha(path.join(repositoryRoot, name))}  ${name}\n`).join('')));
  return { root, repositoryRoot, buildDirectory, sourceList, sourceID, binaries: ['reader'], requiredSources: names };
}

test('splitlines retains the historical text boundary rules', () => {
  assert.deepEqual(lines(''), []); assert.deepEqual(lines('\n'), ['']);
  assert.deepEqual(lines('a\r\nb\rc\u0085d\u2028e\n\n'), ['a', 'b', 'c', 'd', 'e', '']);
});
test('capture requires an explicit runner and captures source identity separately', t => {
  const f = fixture(t);
  assert.throws(() => captureSources(f.root, f), /explicitly bound/);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze')), false);
  const calls = [];
  const out = captureSources(f.root, { ...f, commandOutput: (file, args) => {
    calls.push([file, args]); return `${f.sourceID}\n`;
  } });
  assert.equal(out.sourceId, f.sourceID); assert.equal(out.sourceFiles, 2);
  assert.deepEqual(calls, [[path.join(f.buildDirectory, 'reader'), ['--source-id']]]);
  assert.equal(sha(path.join(out.out, 'source/entry.mjs')), sha(path.join(f.repositoryRoot, 'entry.mjs')));
  assert.throws(() => captureSources(f.root, { ...f, commandOutput: () => f.sourceID }));
});
test('a source change after source-id capture cannot become a sealed snapshot', t => {
  const f = fixture(t);
  assert.throws(() => captureSources(f.root, { ...f, commandOutput: () => {
    fs.writeFileSync(path.join(f.repositoryRoot, 'entry.mjs'), '// changed during capture\n');
    return f.sourceID;
  } }), /source changed|changed after admission/);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze/receipt.json')), false);
});
test('missing source roles and aliases are rejected before capture', t => {
  const f = fixture(t);
  assert.throws(() => captureSources(f.root, { ...f, sourceList: 'entry.mjs\n', commandOutput: () => f.sourceID }), /must be captured/);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze')), false);
  fs.unlinkSync(path.join(f.repositoryRoot, 'support.mjs'));
  fs.linkSync(path.join(f.repositoryRoot, 'entry.mjs'), path.join(f.repositoryRoot, 'support.mjs'));
  assert.throws(() => captureSources(f.root, { ...f, commandOutput: () => f.sourceID }), /single-link|aliased/);
});
test('a late missing source is rejected before the first source payload open', t => {
  const f = fixture(t), sources = new Set(['entry.mjs', 'support.mjs'].map(name => path.join(f.repositoryRoot, name)));
  fs.unlinkSync(path.join(f.repositoryRoot, 'support.mjs'));
  let sourceOpens = 0;
  const originalOpen = fs.openSync;
  t.mock.method(fs, 'openSync', (file, ...args) => {
    if (sources.has(file)) sourceOpens += 1;
    return originalOpen(file, ...args);
  });
  assert.throws(() => captureSources(f.root, { ...f, commandOutput: () => f.sourceID }), /ENOENT/);
  assert.equal(sourceOpens, 0);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze')), false);
});
test('a same-byte binary replacement inside the source-id callback is rejected', t => {
  const f = fixture(t), binary = path.join(f.buildDirectory, 'reader');
  assert.throws(() => captureSources(f.root, { ...f, commandOutput: () => {
    const bytes = fs.readFileSync(binary);
    fs.renameSync(binary, `${binary}.original`); fs.writeFileSync(binary, bytes);
    return f.sourceID;
  } }), /changed after admission/);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze/binaries.sha256')), false);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze/receipt.json')), false);
});
test('a same-byte source replacement cannot pass by being re-admitted for copying', t => {
  const f = fixture(t), source = path.join(f.repositoryRoot, 'entry.mjs');
  assert.throws(() => captureSources(f.root, { ...f, commandOutput: () => {
    const bytes = fs.readFileSync(source);
    fs.renameSync(source, `${source}.original`); fs.writeFileSync(source, bytes);
    return f.sourceID;
  } }), /changed after admission/);
  assert.equal(fs.existsSync(path.join(f.root, 'freeze/source/entry.mjs')), false);
});
test('final inventory hashes retain original admission identity through validation', t => {
  const f = fixture(t), file = path.join(f.root, 'opaque-evidence.txt');
  fs.writeFileSync(file, 'fixture evidence\n');
  const retained = retainedAdmissions(f.root), bytes = fs.readFileSync(file);
  fs.renameSync(file, path.join(f.repositoryRoot, 'prior-evidence.txt')); fs.writeFileSync(file, bytes);
  assert.throws(() => finalize(f.root, { status: 'fixture-only' }, { admittedFiles: retained }), /changed after admission/);
  assert.equal(fs.existsSync(path.join(f.root, 'artifacts.sha256')), false);
  assert.equal(fs.existsSync(path.join(f.root, 'complete.json')), false);
});
test('recursive snapshot directories reject redirect ancestors before writing', t => {
  const f = fixture(t), destination = path.join(f.repositoryRoot, 'unrelated');
  fs.mkdirSync(destination);
  const redirect = path.join(f.repositoryRoot, 'redirect'); fs.symlinkSync(destination, redirect, 'dir');
  assert.throws(() => mkdir(path.join(redirect, 'new-parent/leaf'), true), /directory redirect|redirected/);
  assert.equal(fs.existsSync(path.join(destination, 'new-parent')), false);
});
test('relation factories require explicit new build and historical path bindings', () => {
  assert.throws(() => learned(), /explicit migration/);
  const options = { repositoryRoot: '/mityane', buildDirectory: '/opt/mityane-geometrics/fixtures/relation',
    originalCapsule: '/mityane/.temp/fixture/original', branchCapsule: '/mityane/.temp/fixture/branch' };
  assert.doesNotThrow(() => learned(options));
  assert.throws(() => content(options), /preserved v21/);
  assert.throws(() => learned({ ...options, buildDirectory: '/opt/cuwacunu_embedding/build/historical' }), /new isolated build/);
});
