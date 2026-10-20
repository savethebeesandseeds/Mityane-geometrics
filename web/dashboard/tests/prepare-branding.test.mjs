// SPDX-License-Identifier: AGPL-3.0-only
import assert from 'node:assert/strict';
import {mkdtemp, readFile, writeFile, rm} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import test from 'node:test';
import {brandingDirectory, nativeBinary, prepareArtwork, selectedArtwork, updateArtwork} from '../scripts/prepare-branding.mjs';

const fixture = {global: 'global.png', heroes: {example: 'hero.png'}};
const manifest = {heroes: [{id: 'example'}]};
const parse = async path => JSON.parse((await readFile(new URL(path, import.meta.url), 'utf8')).replace(/^\uFEFF/, ''));

test('artwork assignments preserve manifest order, global dimensions and distinct sources', () => {
  assert.deepEqual(selectedArtwork(fixture, manifest), [['global', 'global.png', 384], ['example', 'hero.png', 256]]);
  assert.throws(() => selectedArtwork({...fixture, heroes: {}}, manifest), /every manifest Hero/);
  assert.throws(() => selectedArtwork({...fixture, extra: true}, manifest), /every manifest Hero/);
  assert.throws(() => selectedArtwork({...fixture, global: 'hero.png'}, manifest), /distinct source/);
});

test('native artwork preserves all checked-in output bytes and provenance', async () => {
  const selection = await parse('../../branding/dashboard-assignments.json');
  const maintainedManifest = await parse('../manifest.json');
  const {files, total, heroCount} = await prepareArtwork(selection, maintainedManifest);
  assert.equal(heroCount, maintainedManifest.heroes.length);
  assert.equal(files.size, heroCount + 3);
  assert.ok(total <= 512 * 1024);
  for (const [filename, bytes] of files) assert.deepEqual(bytes, await readFile(join(brandingDirectory, filename)), filename);
  await updateArtwork(files, {check: true});
});

test('artwork preparation rejects paths escaping the source collection', async () => {
  await assert.rejects(prepareArtwork({...fixture, global: '../../README.md'}, manifest), /PNG inside web\/branding/);
});

test('check mode detects stale and missing outputs without writing', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'branding-check-'));
  try {
    const file = join(directory, 'global.webp');
    await writeFile(file, 'stale');
    const files = new Map([['global.webp', Buffer.from('replacement')]]);
    await assert.rejects(updateArtwork(files, {check: true, destination: directory}), /out of date/);
    assert.equal(await readFile(file, 'utf8'), 'stale');
    await updateArtwork(files, {destination: directory});
    await updateArtwork(files, {check: true, destination: directory});
    await assert.rejects(updateArtwork(new Map([['missing.webp', Buffer.from('new')]]), {check: true, destination: directory}), /out of date/);
  } finally {
    await rm(directory, {recursive: true});
  }
});

test('native helper rejects invalid image data and unsupported dimensions', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'branding-invalid-'));
  try {
    const file = join(directory, 'invalid.png');
    await writeFile(file, 'not an image');
    const result = spawnSync(nativeBinary, [file, '256']);
    assert.equal(result.status, 1);
    assert.match(result.stderr.toString(), /Cannot decode/);
    assert.equal(result.stdout.length, 0);
    const invalidSize = spawnSync(nativeBinary, [file, '1024']);
    assert.equal(invalidSize.status, 1);
    assert.match(invalidSize.stderr.toString(), /Usage:/);
  } finally {
    await rm(directory, {recursive: true});
  }
});
