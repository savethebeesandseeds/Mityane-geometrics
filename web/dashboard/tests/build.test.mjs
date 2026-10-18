// SPDX-License-Identifier: AGPL-3.0-only
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {renderStandalone, updateStandalone} from '../scripts/build-standalone.mjs';
import {readProject, sourceFingerprint} from '../scripts/project.mjs';
import {readBranding} from '../scripts/branding.mjs';
import {fixtureRecord} from './fixtures.mjs';

test('the maintained standalone project includes working links, embedded artwork and exact script hashes', async () => {
  const record = await readProject();
  const html = await renderStandalone(record);
  assert.ok(html.includes('Project dashboard · Mityane geometrics'));
  assert.ok(html.includes('href="../../../README.md"'));
  assert.ok(html.includes('href="../SCHEMA.md"'));
  assert.ok(html.includes('href="../README.md"'));
  assert.ok(!/NonVerba|Non Verba|non-verba-dev|__BRAND_/.test(html));
  const branding = await readBranding();
  assert.ok(html.includes(branding.mark));
  assert.ok(html.includes(branding.portrait));
  assert.ok(html.includes("connect-src 'none'"));
  assert.ok(html.includes("style-src-attr 'none'"));
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);
  assert.equal(scripts.length, 4);
  for (const script of scripts) {
    const digest = createHash('sha256').update(script).digest('base64');
    assert.ok(html.includes(`'sha256-${digest}'`));
  }
  assert.deepEqual(JSON.parse(scripts[2].match(/window\.MityaneDashboardRecord = ([\s\S]*);/)[1]), record);
});

test('every maintained Hero has distinct embedded artwork and the global image is separate', async () => {
  const record = await readProject();
  const branding = await readBranding();
  assert.deepEqual(Object.keys(branding.heroes).sort(), record.heroes.map(hero => hero.id).sort());
  assert.equal(branding.mark, branding.portrait);
  assert.equal(branding.defaultHero, branding.portrait);
  const images = [branding.portrait, ...Object.values(branding.heroes)];
  assert.equal(new Set(images).size, record.heroes.length + 1);
  for (const image of images) {
    assert.match(image, /^data:image\/webp;base64,/);
    const bytes = Buffer.from(image.split(',')[1], 'base64');
    assert.equal(bytes.subarray(0, 4).toString(), 'RIFF');
    assert.equal(bytes.subarray(8, 12).toString(), 'WEBP');
  }
});

test('synthetic Hero content remains literal inside the hash-protected standalone record', async () => {
  const record = structuredClone(fixtureRecord);
  record.heroes[0].title = '</script><script>window.injected = true</script>';
  const html = await renderStandalone(record);
  assert.equal([...html.matchAll(/<script>/g)].length, 4);
  assert.ok(!html.includes(record.heroes[0].title));
  const data = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][2][1];
  assert.deepEqual(JSON.parse(data.match(/window\.MityaneDashboardRecord = ([\s\S]*);/)[1]), record);
});

test('refresh refuses changed source before writing the generated HTML', async () => {
  assert.match(await sourceFingerprint(), /^[0-9a-f]{64}$/);
  await assert.rejects(updateStandalone(await readProject(), {fingerprint: 'changed'}), /source changed during refresh/);
});
