// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {exportBundle, verifyBundle} from '../export-cpb-family.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const profilePath = 'package.json';

function fixture(t) {
  assert.equal(process.platform, 'linux', 'run through the managed Linux container');
  assert(fs.existsSync('/.dockerenv'), 'managed container required');
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'cpb-source-export-'));
  t.after(() => {
    assert(path.basename(temporary).startsWith('cpb-source-export-'));
    assert.equal(path.dirname(temporary), path.resolve(os.tmpdir()));
    fs.rmSync(temporary, {recursive: true});
  });
  const root = path.join(temporary, 'origin');
  const output = path.join(temporary, 'bundle');
  fs.mkdirSync(path.join(root, 'include'), {recursive: true});
  fs.mkdirSync(path.join(root, 'src'));
  fs.writeFileSync(path.join(root, 'include', 'core.h'), '#pragma once\ninline int value(){return 3;}\n');
  fs.writeFileSync(path.join(root, 'src', 'main.cpp'), '#include "core.h"\nint main(){return value();}\n');
  fs.writeFileSync(path.join(root, 'LICENSE'), 'test-only license bytes\r\n');
  const profile = {
    schema: 'cpb-family-source-package-v1',
    include_roots: ['include'],
    files: [{path: 'src/main.cpp', role: 'training-example'}, {path: 'include/core.h', role: 'numerical-core'},
      {path: 'LICENSE', role: 'license'}, {path: profilePath, role: 'export-profile'}],
  };
  const writeProfile = () => fs.writeFileSync(path.join(root, profilePath), `${JSON.stringify(profile)}\n`);
  writeProfile();
  const run = () => exportBundle({root, output, profilePath});
  return {temporary, root, output, profile, writeProfile, run};
}

test('complete source closure exports exact bytes and verifies after origin relocation', t => {
  const f = fixture(t);
  const result = f.run();
  assert.equal(result.files, 4);
  assert.equal(result.quotedIncludesChecked, 1);
  assert.equal(result.manifestSha256, hash(fs.readFileSync(result.manifestPath)));
  for (const entry of f.profile.files) {
    assert.deepEqual(fs.readFileSync(path.join(f.output, 'sources', entry.path)), fs.readFileSync(path.join(f.root, entry.path)));
  }
  const manifest = JSON.parse(fs.readFileSync(result.manifestPath));
  assert.equal(manifest.source_snapshot.kind, 'selected-working-tree-snapshot');
  assert.equal(manifest.source_snapshot.git_head_is_snapshot_identity, false);
  const relocated = path.join(f.temporary, 'relocated');
  fs.renameSync(f.output, relocated);
  fs.renameSync(f.root, path.join(f.temporary, 'origin-is-unavailable'));
  assert.equal(verifyBundle({bundle: relocated, expectedManifestSha256: result.manifestSha256}).files, 4);
});

test('missing quoted dependency refuses before output is created', t => {
  const f = fixture(t);
  f.profile.files = f.profile.files.filter(entry => entry.path !== 'include/core.h');
  f.writeProfile();
  assert.throws(f.run, /missing or ambiguous quoted include/);
  assert(!fs.existsSync(f.output));
});

test('existing output is preserved', t => {
  const f = fixture(t);
  fs.mkdirSync(f.output);
  fs.writeFileSync(path.join(f.output, 'preserve.txt'), 'original');
  assert.throws(f.run, /output must be a new leaf/);
  assert.equal(fs.readFileSync(path.join(f.output, 'preserve.txt'), 'utf8'), 'original');
});

test('edited and deleted members and unexpected files are rejected', async t => {
  for (const mutation of ['edited', 'deleted', 'added']) {
    await t.test(mutation, child => {
      const f = fixture(child), result = f.run();
      const target = path.join(f.output, 'sources', 'include', 'core.h');
      if (mutation === 'edited') fs.appendFileSync(target, '// changed');
      else if (mutation === 'deleted') fs.unlinkSync(target);
      else fs.writeFileSync(path.join(f.output, 'unexpected.txt'), 'extra');
      assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: result.manifestSha256}),
        mutation === 'edited' ? /member byte\/hash mismatch/ : mutation === 'added' ? /unexpected or missing/ : /ENOENT/);
    });
  }
});

test('manifest requires an independent pin and rejects altered metadata', t => {
  const f = fixture(t), result = f.run();
  assert.throws(() => verifyBundle({bundle: f.output}), /independent manifest SHA256 required/);
  assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: '0'.repeat(64)}), /manifest SHA256 mismatch/);
  fs.appendFileSync(result.manifestPath, ' ');
  assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: result.manifestSha256}), /manifest SHA256 mismatch/);
});

test('source path escape, binary artifacts and source symlinks are refused', async t => {
  for (const value of ['../outside.h', '/absolute.h', 'C:/outside.h', 'include\\core.h', '.temp/state.pt', 'weights.pt']) {
    await t.test(value, child => {
      const f = fixture(child);
      f.profile.files.push({path: value, role: 'invalid'});
      f.writeProfile();
      assert.throws(f.run, /safe POSIX relative path|source-only package refuses/);
      assert(!fs.existsSync(f.output));
    });
  }
  await t.test('symlink', child => {
    const f = fixture(child);
    const header = path.join(f.root, 'include', 'core.h');
    const realHeader = path.join(f.temporary, 'external.h');
    fs.renameSync(header, realHeader);
    fs.symlinkSync(realHeader, header);
    assert.throws(f.run, /symlink refused/);
    assert(!fs.existsSync(f.output));
  });
});

test('bundle path redirection is refused even under a newly supplied manifest pin', t => {
  const f = fixture(t), result = f.run();
  const manifest = JSON.parse(fs.readFileSync(result.manifestPath));
  manifest.files[0].bundle_path = '../outside';
  const bytes = Buffer.from(JSON.stringify(manifest));
  fs.writeFileSync(result.manifestPath, bytes);
  assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: hash(bytes)}), /invalid or duplicate source member/);
});

test('bundle symlinks and ambiguous includes are rejected', async t => {
  await t.test('bundle symlink', child => {
    const f = fixture(child), result = f.run();
    const target = path.join(f.output, 'sources', 'include', 'core.h');
    fs.unlinkSync(target);
    fs.symlinkSync(path.join(f.root, 'include', 'core.h'), target);
    assert.throws(() => verifyBundle({bundle: f.output, expectedManifestSha256: result.manifestSha256}), /symlink refused/);
  });
  await t.test('ambiguous include', child => {
    const f = fixture(child);
    fs.writeFileSync(path.join(f.root, 'src', 'core.h'), '#pragma once\n');
    f.profile.files.push({path: 'src/core.h', role: 'second-core'});
    f.writeProfile();
    assert.throws(f.run, /missing or ambiguous quoted include/);
    assert(!fs.existsSync(f.output));
  });
});
