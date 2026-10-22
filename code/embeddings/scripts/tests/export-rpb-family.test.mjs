// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {exportBundle, verifyBundle} from '../export-rpb-family.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const profilePath = 'package.json';

function fixture(t) {
  assert.equal(process.platform, 'linux', 'run through the managed Linux container');
  assert(fs.existsSync('/.dockerenv'), 'managed container required');
  const parent = path.resolve(process.env.RPB_PACKAGE_TEST_TMP ||
    '/opt/mityane-geometrics/embeddings/build/hx16-rpb-package-tests');
  assert(parent.startsWith('/opt/mityane-geometrics/'), 'container-owned temporary location required');
  fs.mkdirSync(parent, {recursive: true});
  const temporary = fs.mkdtempSync(path.join(parent, 'rpb-source-export-'));
  t.after(() => {
    assert.equal(path.dirname(temporary), parent);
    assert(path.basename(temporary).startsWith('rpb-source-export-'));
    fs.rmSync(temporary, {recursive: true});
  });
  const root = path.join(temporary, 'origin'), output = path.join(temporary, 'bundle');
  fs.mkdirSync(path.join(root, 'include', 'detail'), {recursive: true});
  fs.mkdirSync(path.join(root, 'src'));
  fs.writeFileSync(path.join(root, 'include', 'detail', 'value.h'), '#pragma once\ninline int value(){return 3;}\n');
  fs.writeFileSync(path.join(root, 'include', 'core.h'), '#pragma once\n#include "detail/value.h"\n');
  fs.writeFileSync(path.join(root, 'src', 'main.cpp'), '#include "core.h"\nint main(){return value();}\n');
  fs.writeFileSync(path.join(root, 'LICENSE'), 'test-only license bytes\r\n');
  const profile = {schema: 'rpb-family-source-package-v1', include_roots: ['include'],
    dependency_roots: ['include'], files: [
      {path: 'src/main.cpp', role: 'training-example'}, {path: 'LICENSE', role: 'license'},
      {path: profilePath, role: 'export-profile'}]};
  const writeProfile = () => fs.writeFileSync(path.join(root, profilePath), `${JSON.stringify(profile)}\n`);
  writeProfile();
  return {temporary, root, output, profile, writeProfile,
    run: () => exportBundle({root, output, profilePath})};
}

test('curated roots discover recursive dependencies; relocated verification reads no origin', t => {
  const f = fixture(t), result = f.run();
  assert.equal(result.files, 5);
  assert.equal(result.quotedIncludesChecked, 2);
  assert.equal(result.manifestSha256, hash(fs.readFileSync(result.manifestPath)));
  const manifest = JSON.parse(fs.readFileSync(result.manifestPath));
  assert.equal(manifest.source_snapshot.git_head_is_snapshot_identity, false);
  assert.equal(manifest.files.filter(entry => entry.role === 'quoted-include-dependency').length, 2);
  for (const entry of manifest.files) {
    assert.deepEqual(fs.readFileSync(path.join(f.output, entry.bundle_path)), fs.readFileSync(path.join(f.root, entry.path)));
  }
  const relocated = path.join(f.temporary, 'relocated');
  fs.renameSync(f.output, relocated);
  fs.renameSync(f.root, path.join(f.temporary, 'origin-unavailable'));
  assert.equal(verifyBundle({bundle: relocated, expectedManifestSha256: result.manifestSha256}).files, 5);
});

test('missing transitive dependency refuses before output creation', t => {
  const f = fixture(t);
  fs.unlinkSync(path.join(f.root, 'include', 'detail', 'value.h'));
  assert.throws(f.run, /missing or ambiguous quoted include/);
  assert(!fs.existsSync(f.output));
});

test('dependency discovery cannot silently expand beyond curated roots', t => {
  const f = fixture(t);
  f.profile.dependency_roots = ['include/detail'];
  f.writeProfile();
  assert.throws(f.run, /quoted dependency outside curated roots/);
  assert(!fs.existsSync(f.output));
});

test('existing output and dangling output symlink are preserved', async t => {
  for (const kind of ['directory', 'symlink']) await t.test(kind, child => {
    const f = fixture(child);
    if (kind === 'directory') {
      fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, 'preserve.txt'), 'original');
    } else fs.symlinkSync(path.join(f.temporary, 'absent'), f.output);
    assert.throws(f.run, /output must be a new leaf/);
    if (kind === 'directory') assert.equal(fs.readFileSync(path.join(f.output, 'preserve.txt'), 'utf8'), 'original');
    else assert(fs.lstatSync(f.output).isSymbolicLink());
  });
});

test('member tampering, deletion and unlisted extra bytes are rejected', async t => {
  for (const kind of ['edit', 'delete', 'extra']) await t.test(kind, child => {
    const f = fixture(child), result = f.run();
    const target = path.join(f.output, 'sources', 'include', 'core.h');
    if (kind === 'edit') fs.appendFileSync(target, '// altered');
    else if (kind === 'delete') fs.unlinkSync(target);
    else fs.writeFileSync(path.join(f.output, 'unexpected.txt'), 'extra');
    assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: result.manifestSha256}),
      kind === 'edit' ? /member byte\/hash mismatch/ : kind === 'delete' ? /ENOENT/ : /unexpected or missing/);
  });
});

test('independent manifest pin is required and metadata edits invalidate it', t => {
  const f = fixture(t), result = f.run();
  assert.throws(() => verifyBundle({bundle: f.output}), /independent manifest SHA256 required/);
  assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: '0'.repeat(64)}), /manifest SHA256 mismatch/);
  fs.appendFileSync(result.manifestPath, ' ');
  assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: result.manifestSha256}), /manifest SHA256 mismatch/);
});

test('unsafe source paths and binary/environment artifacts are refused', async t => {
  for (const value of ['../outside.h', '/absolute.h', 'C:/outside.h', 'include\\core.h',
    'include/core.h:stream', '.temp/state.pt', 'weights.pt', '.env.local', 'output/run.json']) {
    await t.test(value, child => {
      const f = fixture(child);
      f.profile.files.push({path: value, role: 'invalid'}); f.writeProfile();
      assert.throws(f.run, /safe POSIX relative path|source-only package refuses/);
      assert(!fs.existsSync(f.output));
    });
  }
});

test('source and bundle symlinks and ambiguous include resolution are refused', async t => {
  await t.test('source symlink', child => {
    const f = fixture(child), header = path.join(f.root, 'include', 'core.h');
    const external = path.join(f.temporary, 'outside.h');
    fs.renameSync(header, external); fs.symlinkSync(external, header);
    assert.throws(f.run, /symlink refused/); assert(!fs.existsSync(f.output));
  });
  await t.test('bundle symlink', child => {
    const f = fixture(child), result = f.run(), header = path.join(f.output, 'sources', 'include', 'core.h');
    fs.unlinkSync(header); fs.symlinkSync(path.join(f.root, 'include', 'core.h'), header);
    assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: result.manifestSha256}), /symlink refused/);
  });
  await t.test('ambiguous include', child => {
    const f = fixture(child);
    fs.writeFileSync(path.join(f.root, 'src', 'core.h'), '#pragma once\n');
    f.profile.dependency_roots.push('src'); f.writeProfile();
    assert.throws(f.run, /missing or ambiguous quoted include/); assert(!fs.existsSync(f.output));
  });
});

test('bundle path redirection fails even when given a new manifest pin', t => {
  const f = fixture(t), result = f.run(), manifest = JSON.parse(fs.readFileSync(result.manifestPath));
  manifest.files[0].bundle_path = '../outside';
  const bytes = Buffer.from(JSON.stringify(manifest)); fs.writeFileSync(result.manifestPath, bytes);
  assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: hash(bytes)}), /invalid or duplicate source member/);
});
